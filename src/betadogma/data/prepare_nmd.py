# src/betadogma/data/prepare_nmd.py
"""
Create transcript-level NMD labels from GTF using the 55-nt rule.

Usage:
  python -m betadogma.data.prepare_nmd \
    --gtf /abs/path/gencode.v44.annotation.gtf \
    --out data/cache/nmd_labels.parquet \
    [--chroms chr1,chr2] [--ptc-threshold 55]

Output columns:
  transcript_id, gene_id, chrom, strand,
  n_exons, cds_tx_start, cds_tx_end, last_junction_tx, distance_ptc_to_last_junc,
  nmd_label
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chroms", type=str, default="")
    ap.add_argument("--ptc-threshold", type=int, default=55)
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


def _iter_gtf(gtf_path: str, allowed: Optional[set[str]]):
    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) != 9:
                continue
            chrom, source, feat, start, end, score, strand, frame, attrs = p
            if allowed and chrom not in allowed:
                continue
            yield {
                "chrom": chrom, "feature": feat, "start": int(start) - 1, "end": int(end),
                "strand": strand, "attrs": _parse_attrs(attrs)
            }


def run(gtf: str, out: str, chroms: str = "", ptc_threshold: int = 55):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    allowed = set([c.strip() for c in chroms.split(",") if c.strip()]) if chroms else None

    # Collect exons and CDS per transcript
    tx_meta: Dict[str, Dict] = {}  # tid -> meta
    tx_exons: Dict[str, List[Tuple[int, int]]] = {}
    tx_cds: Dict[str, List[Tuple[int, int]]] = {}

    for r in _iter_gtf(gtf, allowed):
        a = r["attrs"]
        tid = a.get("transcript_id"); gid = a.get("gene_id")
        if not tid or not gid:
            continue
        strand = r["strand"]; chrom = r["chrom"]

        if r["feature"] == "exon":
            tx_exons.setdefault(tid, []).append((r["start"], r["end"]))
            tx_meta.setdefault(tid, {"gene_id": gid, "chrom": chrom, "strand": strand})
        elif r["feature"] == "CDS":
            tx_cds.setdefault(tid, []).append((r["start"], r["end"]))
            tx_meta.setdefault(tid, {"gene_id": gid, "chrom": chrom, "strand": strand})

    rows: List[Dict] = []

    for tid, exons in tx_exons.items():
        if tid not in tx_cds:
            continue  # non-coding or no CDS annotated
        meta = tx_meta[tid]
        strand = meta["strand"]

        # Sort exons in genomic order and then get transcription order
        exons.sort()
        tx_order = exons if strand == "+" else list(reversed(exons))

        if len(tx_order) < 2:
            continue  # single-exon → NMD rule not applicable

        # Build transcript coordinate mapping to compute junction coordinate
        exon_lengths = [e1 - e0 for (e0, e1) in tx_order]
        last_junction_tx = sum(exon_lengths[:-1])  # coordinate where the last junction occurs

        # CDS span in transcript coordinates
        cds_list = sorted(tx_cds[tid])
        cds_genomic_start = cds_list[0][0] if strand == "+" else cds_list[-1][0]
        cds_genomic_end   = cds_list[-1][1] if strand == "+" else cds_list[0][1]

        # Convert genomic positions → transcript coords by walking exons in tx order
        def genomic_to_tx(gpos: int) -> Optional[int]:
            offset = 0
            for (s, e) in tx_order:
                if s <= gpos < e:
                    # position inside this exon
                    if strand == "+":
                        return offset + (gpos - s)
                    else:  # '-' strand: count from exon 3' end backwards
                        return offset + ((e - 1) - gpos)
                offset += (e - s)
            return None

        cds_tx_start = genomic_to_tx(cds_genomic_start)
        cds_tx_end   = genomic_to_tx(cds_genomic_end - 1)  # last base in CDS inclusive
        if cds_tx_start is None or cds_tx_end is None:
            continue

        # PTC position = start of stop codon on coding strand
        # For '+' strand, stop begins at cds_end - 2; for '-' strand, at cds_start
        ptc_tx = (cds_tx_end - 2) if strand == "+" else cds_tx_start
        distance = last_junction_tx - ptc_tx  # positive if PTC is upstream of last junction

        nmd_label = 1 if (distance > ptc_threshold) else 0

        rows.append({
            "transcript_id": tid,
            "gene_id": meta["gene_id"],
            "chrom": meta["chrom"],
            "strand": strand,
            "n_exons": len(tx_order),
            "cds_tx_start": int(cds_tx_start),
            "cds_tx_end": int(cds_tx_end),
            "last_junction_tx": int(last_junction_tx),
            "distance_ptc_to_last_junc": int(distance),
            "nmd_label": int(nmd_label),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[prepare_nmd] wrote {out} ({len(df):,} transcripts)")


def main():
    args = parse_args()
    run(args.gtf, args.out, chroms=args.chroms, ptc_threshold=int(args.ptc_threshold))


if __name__ == "__main__":
    main()