# betadogma/data/prepare_gencode.py
"""
Prepare GENCODE annotations into binned, model-ready labels for structural heads:
- splice donor/acceptor
- TSS
- polyA

Outputs Parquet shards with (seq, donor, acceptor, tss, polya) at encoder bin resolution.

Usage:
  python -m betadogma.data.prepare_gencode \
      --fasta /path/GRCh38.primary_assembly.genome.fa \
      --gtf /path/gencode.v44.annotation.gtf \
      --out data/cache/gencode_v44_structural \
      --window 131072 \
      --stride 65536 \
      --bin-size 128 \
      --chroms chr1,chr2
"""
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple
from collections import defaultdict

import pyfaidx
import pandas as pd

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
    ap.add_argument("--bin-size", type=int, default=128)
    ap.add_argument("--chroms", type=str, default="")
    ap.add_argument("--max-shard-bases", type=int, default=50_000_000)
    return ap.parse_args()

def read_gtf_minimal(gtf_path: str, allowed_chroms: set[str] | None):
    """
    Minimal GTF parser yielding dicts with fields:
    chrom, source, feature, start, end, strand, attrs (dict)
    """
    def parse_attrs(s: str) -> Dict[str, str]:
        out = {}
        for item in s.strip().split(";"):
            item = item.strip()
            if not item:
                continue
            if " " in item:
                k, v = item.split(" ", 1)
                out[k] = v.strip().strip('"')
        return out

    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feat, start, end, score, strand, frame, attrs = parts
            if allowed_chroms and chrom not in allowed_chroms:
                continue
            if feat not in (GTF_FEATURES_EXON | GTF_FEATURES_TSS | GTF_FEATURES_POLYA | {"CDS"}):
                continue
            yield {
                "chrom": chrom,
                "source": source,
                "feature": feat,
                "start": int(start) - 1,  # to 0-based
                "end": int(end),          # half-open
                "strand": strand,
                "attrs": parse_attrs(attrs),
            }

def collect_sites(records):
    """
    From GTF records, collect base-resolution sets:
      donors, acceptors, tss, polya
    Simple heuristics:
      - donor = exon end at splice (5' splice site) depending on strand
      - acceptor = exon start at splice (3' splice site)
      - tss = transcript start
      - polyA = transcript end
    """
    donors = defaultdict(set)
    acceptors = defaultdict(set)
    tss = defaultdict(set)
    polya = defaultdict(set)

    # transcript boundaries
    tx_bounds = defaultdict(lambda: {"chrom": None, "strand": None, "start": None, "end": None})

    # exon boundaries per transcript to infer splice junctions
    exons_by_tx = defaultdict(list)

    for r in records:
        chrom = r["chrom"]
        feat = r["feature"]
        strand = r["strand"]
        a = r["attrs"]

        if feat == "transcript":
            tid = a.get("transcript_id")
            if not tid:
                continue
            s, e = r["start"], r["end"]
            tx_bounds[tid]["chrom"] = chrom
            tx_bounds[tid]["strand"] = strand
            tx_bounds[tid]["start"] = s if tx_bounds[tid]["start"] is None else min(tx_bounds[tid]["start"], s)
            tx_bounds[tid]["end"] = e if tx_bounds[tid]["end"] is None else max(tx_bounds[tid]["end"], e)

        if feat == "exon":
            tid = a.get("transcript_id")
            if not tid:
                continue
            exons_by_tx[tid].append((r["start"], r["end"], strand, chrom))

    # TSS / polyA from transcript bounds
    for tid, info in tx_bounds.items():
        chrom = info["chrom"]
        strand = info["strand"]
        s, e = info["start"], info["end"]
        if chrom is None or strand is None or s is None or e is None:
            continue
        if strand == "+":
            tss[chrom].add(s)
            polya[chrom].add(e - 1)
        else:
            tss[chrom].add(e - 1)
            polya[chrom].add(s)

    # Donor/acceptor from exon adjacency
    for tid, exons in exons_by_tx.items():
        # sort by genomic coordinate then use strand to decide adjacency
        exons.sort()
        # Internal boundaries produce junctions
        for i in range(len(exons) - 1):
            s1, e1, strand, chrom = exons[i]
            s2, e2, strand2, chrom2 = exons[i + 1]
            if chrom != chrom2:
                continue
            if strand == "+":
                # exon1 end -> donor at e1-1, exon2 start -> acceptor at s2
                donors[chrom].add(e1 - 1)
                acceptors[chrom].add(s2)
            else:
                # negative strand: reversed roles
                donors[chrom].add(s2)
                acceptors[chrom].add(e1 - 1)

    return donors, acceptors, tss, polya

def bin_indices(start: int, end: int, bin_size: int, pos: int) -> int:
    if pos < start or pos >= end:
        return -1
    return (pos - start) // bin_size

def make_binned_tracks(chrom: str, wstart: int, wend: int, bin_size: int,
                       donors: set[int], acceptors: set[int], tss: set[int], polya: set[int]) -> Dict[str, List[int]]:
    Lr = (wend - wstart) // bin_size
    arrs = {
        "donor": [0] * Lr,
        "acceptor": [0] * Lr,
        "tss": [0] * Lr,
        "polya": [0] * Lr,
    }
    for site in donors:
        b = bin_indices(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["donor"][b] = 1
    for site in acceptors:
        b = bin_indices(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["acceptor"][b] = 1
    for site in tss:
        b = bin_indices(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["tss"][b] = 1
    for site in polya:
        b = bin_indices(wstart, wend, bin_size, site)
        if b >= 0:
            arrs["polya"][b] = 1
    return arrs

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    allowed_chroms = set(args.chroms.split(",")) if args.chroms else None

    # 1) parse GTF (stream)
    gtf_iter = read_gtf_minimal(args.gtf, allowed_chroms)
    donors_dict, acceptors_dict, tss_dict, polya_dict = collect_sites(gtf_iter)

    # 2) reference genome
    fasta = pyfaidx.Fasta(args.fasta, as_raw=True, sequence_always_upper=True)

    # 3) sliding windows per chromosome
    shard_rows = []
    bases_in_shard = 0
    shard_idx = 0

    chrom_list = allowed_chroms or [c for c in fasta.keys() if c.startswith("chr")]
    for chrom in chrom_list:
        if chrom not in fasta:
            continue
        clen = len(fasta[chrom])
        w = args.window
        stride = args.stride
        for wstart in range(0, max(1, clen - w + 1), stride):
            wend = wstart + w
            if wend > clen:
                break
            seq = str(fasta[chrom][wstart:wend])
            tracks = make_binned_tracks(
                chrom, wstart, wend, args.bin_size,
                donors_dict.get(chrom, set()),
                acceptors_dict.get(chrom, set()),
                tss_dict.get(chrom, set()),
                polya_dict.get(chrom, set()),
            )
            row = {
                "chrom": chrom,
                "start": wstart,
                "end": wend,
                "seq": seq,
                "bin_size": args.bin_size,
                "donor": tracks["donor"],
                "acceptor": tracks["acceptor"],
                "tss": tracks["tss"],
                "polya": tracks["polya"],
            }
            shard_rows.append(row)
            bases_in_shard += (wend - wstart)
            if bases_in_shard >= args.max_shard_bases:
                df = pd.DataFrame(shard_rows)
                outp = os.path.join(args.out, f"shard_{shard_idx:04d}.parquet")
                df.to_parquet(outp, index=False)
                shard_rows.clear()
                bases_in_shard = 0
                shard_idx += 1

    if shard_rows:
        df = pd.DataFrame(shard_rows)
        outp = os.path.join(args.out, f"shard_{shard_idx:04d}.parquet")
        df.to_parquet(outp, index=False)

    print(f"Wrote shards to {args.out}")

if __name__ == "__main__":
    main()