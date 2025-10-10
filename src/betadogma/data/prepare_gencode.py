# src/betadogma/data/prepare_gencode.py
"""
Prepare GENCODE annotations into binned, model-ready labels for structural heads:
- splice donor/acceptor
- TSS
- polyA

Outputs Parquet shards with columns:
  chrom, start, end, seq, bin_size, donor, acceptor, tss, polya

Usage:
  python -m betadogma.data.prepare_gencode \
      --fasta /path/GRCh38.primary_assembly.genome.fa \
      --gtf /path/gencode.v44.annotation.gtf \
      --out data/cache/gencode_v44_structural_base \
      --window 131072 \
      --bin-size 1 \
      --chroms chr1,chr2
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any, cast

import numpy as np
import pandas as pd

try:
    import pyfaidx
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pyfaidx is required. Install with `pip install pyfaidx`.") from e


# We infer TSS/polyA from 'transcript' rows and junctions from 'exon' adjacency
GTF_FEATURES_TSS = {"transcript"}
GTF_FEATURES_POLYA = {"transcript"}
GTF_FEATURES_EXON = {"exon"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=131072)  # 128k
    ap.add_argument("--stride", type=int, default=65536)   # 64k
    ap.add_argument("--bin-size", type=int, default=1)
    ap.add_argument("--chroms", type=str, default="", help="Comma-separated list; if empty, use FASTA contigs starting with 'chr'.")
    ap.add_argument("--max-shard-bases", type=int, default=50_000_000)
    return ap.parse_args()


def _parse_attrs(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in s.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if " " in item:
            k, v = item.split(" ", 1)
            out[k] = v.strip().strip('"')
    return out


def read_gtf_minimal(gtf_path: str, allowed_chroms: set[str] | None) -> Iterable[Dict]:
    """
    Minimal GTF parser yielding dicts:
      chrom, source, feature, start, end, strand, attrs (dict)
    Coordinates converted to 0-based, half-open [start, end).
    """
    keep_feats = (GTF_FEATURES_EXON | GTF_FEATURES_TSS | GTF_FEATURES_POLYA | {"CDS"})
    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feat, start, end, score, strand, frame, attrs = parts
            if feat not in keep_feats:
                continue
            if allowed_chroms and chrom not in allowed_chroms:
                continue
            try:
                s0 = int(start) - 1
                e1 = int(end)
            except ValueError:
                continue
            yield {
                "chrom": chrom,
                "source": source,
                "feature": feat,
                "start": s0,
                "end": e1,
                "strand": strand,
                "attrs": _parse_attrs(attrs),
            }


def collect_sites(records: Iterable[Dict]) -> Tuple[Dict[str, set[int]], Dict[str, set[int]], Dict[str, set[int]], Dict[str, set[int]]]:
    """
    From GTF records, collect base-resolution sets:
      donors, acceptors, tss, polya

    Heuristics:
      - donor = exon end (5' splice site) on '+'; exon start on '-'
      - acceptor = exon start (3' splice site) on '+'; exon end-1 on '-'
      - tss = transcript 5' end by strand
      - polyA = transcript 3' end by strand
    """
    donors: Dict[str, set[int]] = defaultdict(set)    # chrom -> set[int]
    acceptors: Dict[str, set[int]] = defaultdict(set)  # chrom -> set[int]
    tss: Dict[str, set[int]] = defaultdict(set)        # chrom -> set[int]
    polya: Dict[str, set[int]] = defaultdict(set)      # chrom -> set[int]

    # transcript bounds to infer TSS/polyA
    tx_bounds: Dict[str, Dict[str, Optional[Union[str, int]]]] = defaultdict(
        lambda: {"chrom": None, "strand": None, "start": None, "end": None}
    )
    # per-transcript exon lists to infer junctions
    exons_by_tx = defaultdict(list)

    for r in records:
        chrom = r["chrom"]; feat = r["feature"]; strand = r["strand"]; a = r["attrs"]

        if feat == "transcript":
            tid = a.get("transcript_id")
            if not tid:
                continue
            s, e = r["start"], r["end"]
            t = tx_bounds[tid]
            t["chrom"] = chrom
            t["strand"] = strand
            t["start"] = s if t["start"] is None else min(t["start"], s)
            t["end"] = e if t["end"] is None else max(t["end"], e)

        elif feat == "exon":
            tid = a.get("transcript_id")
            if not tid:
                continue
            exons_by_tx[tid].append((r["start"], r["end"], strand, chrom))

        # CDS lines are not used here; ORF labels would require a different pass.

    # TSS / polyA
    for tid, info in tx_bounds.items():
        chrom = cast(Optional[str], info["chrom"])
        strand = cast(Optional[str], info["strand"])
        s = cast(Optional[int], info["start"])
        e = cast(Optional[int], info["end"])
        
        if chrom is None or strand is None or s is None or e is None:
            continue
            
        if strand == "+":
            tss[chrom].add(s)
            polya[chrom].add(e - 1)
        else:
            tss[chrom].add(e - 1)
            polya[chrom].add(s)

    # Splice donors/acceptors via adjacent exons on the same transcript
    for tid, exons in exons_by_tx.items():
        exons.sort()  # sort by genomic coordinate
        for i in range(len(exons) - 1):
            s1, e1, strand, chrom = exons[i]
            s2, e2, strand2, chrom2 = exons[i + 1]
            if chrom != chrom2:
                continue
            if strand == "+":
                donors[chrom].add(e1 - 1)  # last base of exon i
                acceptors[chrom].add(s2)   # first base of exon i+1
            else:
                donors[chrom].add(s2)      # first base of exon i+1 (5' end on -)
                acceptors[chrom].add(e1 - 1)  # last base of exon i (3' end on -)

    return donors, acceptors, tss, polya


def _bin_index(start: int, end: int, bin_size: int, pos: int) -> int:
    if pos < start or pos >= end:
        return -1
    return (pos - start) // bin_size


def make_binned_tracks(
    chrom: str,
    wstart: int,
    wend: int,
    bin_size: int,
    donors: set[int],
    acceptors: set[int],
    tss: set[int],
    polya: set[int],
) -> Dict[str, List[int]]:
    # Lr must be integral
    total = wend - wstart
    assert total % bin_size == 0, f"Window length {total} not divisible by bin_size {bin_size}"
    Lr = total // bin_size
    arrs = {k: [0] * Lr for k in ("donor", "acceptor", "tss", "polya")}
    for site in donors:
        b = _bin_index(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["donor"][b] = 1
    for site in acceptors:
        b = _bin_index(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["acceptor"][b] = 1
    for site in tss:
        b = _bin_index(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["tss"][b] = 1
    for site in polya:
        b = _bin_index(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["polya"][b] = 1
    return arrs


def prepare_gencode(
    fasta_path: str,
    gtf_path: str,
    out_dir: str,
    window: int = 131072,
    stride: int = 65536,
    bin_size: int = 1,
    chroms: str = "",
    max_shard_bases: int = 50_000_000
) -> None:
    """Prepare GENCODE annotations into binned, model-ready labels for structural heads.
    
    Args:
        fasta_path: Path to the reference genome FASTA file
        gtf_path: Path to the GTF annotation file
        out_dir: Output directory for the processed files
        window: Window size for binning (default: 131072)
        stride: Stride for sliding window (default: 65536)
        bin_size: Size of each bin (default: 1)
        chroms: Comma-separated list of chromosomes to process (default: "", process all)
        max_shard_bases: Maximum number of bases per output shard (default: 50,000,000)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Sanity checks: ensure nice binning
    if not (window > 0 and stride > 0 and bin_size > 0):
        raise ValueError("window, stride, and bin_size must be positive integers")
    if window % bin_size != 0:
        raise ValueError("window must be a multiple of bin_size")
    if stride % bin_size != 0:
        raise ValueError("stride must be a multiple of bin_size")

    # Chromosome selection
    allowed_chroms = set([c.strip() for c in chroms.split(",") if c.strip()]) if chroms else None

    # 1) parse GTF (stream)
    print("[prepare_gencode] parsing GTF...")
    gtf_iter = read_gtf_minimal(gtf_path, allowed_chroms)
    donors_dict, acceptors_dict, tss_dict, polya_dict = collect_sites(gtf_iter)

    # 2) reference genome
    fasta = pyfaidx.Fasta(fasta_path, as_raw=True, sequence_always_upper=True)

    # 3) sliding windows per chromosome
    shard_rows: List[Dict] = []
    bases_in_shard = 0
    shard_idx = 0

    if allowed_chroms:
        chrom_list = [c for c in allowed_chroms if c in fasta]
    else:
        chrom_list = [c for c in fasta.keys() if c.startswith("chr")]

    print(f"[prepare_gencode] processing {len(chrom_list)} contigs ...")
    for chrom in chrom_list:
        if chrom not in fasta:
            continue
        clen = len(fasta[chrom])
        w = window
        stride = stride
        print(f"  - {chrom} (len={clen:,})")
        for wstart in range(0, max(1, clen - w + 1), stride):
            wend = wstart + w
            if wend > clen:
                break
            seq = str(fasta[chrom][wstart:wend])

            tracks = make_binned_tracks(
                chrom, wstart, wend, bin_size,
                donors_dict.get(chrom, set()),
                acceptors_dict.get(chrom, set()),
                tss_dict.get(chrom, set()),
                polya_dict.get(chrom, set()),
            )
            row = {
                "chrom": chrom,
                "start": int(wstart),
                "end": int(wend),
                "seq": seq,
                "bin_size": int(bin_size),
                "donor": [int(x) for x in tracks["donor"]],
                "acceptor": [int(x) for x in tracks["acceptor"]],
                "tss": [int(x) for x in tracks["tss"]],
                "polya": [int(x) for x in tracks["polya"]],
            }
            shard_rows.append(row)
            bases_in_shard += (wend - wstart)

            if bases_in_shard >= max_shard_bases:
                df = pd.DataFrame(shard_rows)
                outp = os.path.join(out_dir, f"shard_{shard_idx:04d}.parquet")
                df.to_parquet(outp, index=False)
                print(f"[prepare_gencode] wrote {outp} ({len(df)} rows)")
                shard_rows.clear()
                bases_in_shard = 0
                shard_idx += 1

    if shard_rows:
        df = pd.DataFrame(shard_rows)
        outp = os.path.join(out_dir, f"shard_{shard_idx:04d}.parquet")
        df.to_parquet(outp, index=False)
        print(f"[prepare_gencode] wrote {outp} ({len(df)} rows)")
    print(f"[prepare_gencode] done. Wrote {shard_idx} shards to {out_dir}")


def main() -> None:
    args = parse_args()
    prepare_gencode(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        out_dir=args.out,
        window=args.window,
        stride=args.stride,
        bin_size=args.bin_size,
        chroms=args.chroms,
        max_shard_bases=args.max_shard_bases
    )

if __name__ == "__main__":
    main()