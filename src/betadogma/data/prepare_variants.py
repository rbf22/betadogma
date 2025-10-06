# src/betadogma/data/prepare_variants.py
"""
Precompute variant-aligned channels for model windows.

Usage:
  python -m betadogma.data.prepare_variants \
    --vcf data/variants/cohort.vcf.gz \
    --windows "data/cache/gencode_v44_structural_base/*.parquet" \
    --out data/cache/variants_aligned \
    [--apply-alt] [--max-per-window 64]

Outputs: Parquet shards in <out>/shard_*.parquet with rows:
  chrom, start, end, bin_size, seq (REF window), seq_alt (if --apply-alt),
  variant_spec, var_type, in_window_idx, span_in_window,
  ch_snp, ch_ins, ch_del, ch_any  (list[int], length = end-start)
"""

from __future__ import annotations
import argparse
import os
from glob import glob
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd

# local helpers
from .encode import encode_variant, apply_variant_to_sequence, build_variant_channels


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="VCF path or glob; .vcf(.gz) or .txt with VCF columns.")
    ap.add_argument("--windows", required=True, help="Glob for window Parquet shards from prepare_gencode.py")
    ap.add_argument("--out", required=True, help="Output directory for variant-aligned Parquet shards.")
    ap.add_argument("--apply-alt", action="store_true", help="Also store seq_alt (sequence with variant applied).")
    ap.add_argument("--max-per-window", type=int, default=0, help="Optional cap of variants per window (0 = unlimited).")
    ap.add_argument("--shard-size", type=int, default=50_000, help="Rows per output shard.")
    return ap.parse_args()


def _iter_vcf(path: str):
    """
    Minimal VCF parser that yields dicts:
      chrom, pos (1-based), id, ref, alt (list[str]), qual, filter, info
    """
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            t = line.rstrip("\n").split("\t")
            if len(t) < 8: 
                continue
            chrom, pos, vid, ref, alt, qual, flt, info = t[:8]
            yield {
                "chrom": chrom,
                "pos": int(pos),
                "id": None if vid == "." else vid,
                "ref": ref,
                "alt": alt.split(","),
                "qual": None if qual == "." else float(qual),
                "filter": flt,
                "info": info,
            }


def _read_vcf_glob(vcf_glob: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(vcf_glob)) if not os.path.isfile(vcf_glob) else [vcf_glob]
    assert paths, f"No VCF files matched: {vcf_glob}"
    out: List[Dict[str, Any]] = []
    for p in paths:
        for r in _iter_vcf(p):
            for alt in r["alt"]:
                spec = f"{r['chrom']}:{r['pos']}{r['ref']}>{alt}"
                out.append({"chrom": r["chrom"], "pos": r["pos"], "ref": r["ref"], "alt": alt, "spec": spec})
    return out


def _read_windows(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob(glob_pat))
    assert files, f"No Parquet windows matched: {glob_pat}"
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    need = {"chrom", "start", "end", "seq", "bin_size"}
    missing = need - set(df.columns)
    assert not missing, f"Missing window columns: {missing}"
    return df


def run(vcf: str, windows: str, out: str, apply_alt: bool = False, max_per_window: int = 0, shard_size: int = 50_000):
    os.makedirs(out, exist_ok=True)

    variants = _read_vcf_glob(vcf)
    win_df = _read_windows(windows)

    # group variants by chrom for faster filter
    variants_by_chr: Dict[str, List[Dict[str, Any]]] = {}
    for v in variants:
        variants_by_chr.setdefault(v["chrom"], []).append(v)

    rows: List[Dict[str, Any]] = []
    shard_idx = 0

    for i, w in win_df.iterrows():
        chrom = w["chrom"]; w0 = int(w["start"]); w1 = int(w["end"])
        seq = w["seq"]; L = len(seq)
        window = {"chrom": chrom, "start": w0, "end": w1}

        hits = [v for v in variants_by_chr.get(chrom, []) if w0 <= (v["pos"] - 1) < w1]
        if max_per_window > 0:
            hits = hits[:max_per_window]

        for v in hits:
            enc = encode_variant(v["spec"], window=window)
            if not enc["in_window"]:
                continue

            ch = build_variant_channels(L, enc)
            rec = {
                "chrom": chrom,
                "start": w0,
                "end": w1,
                "bin_size": int(w.get("bin_size", 1)),
                "seq": seq,
                "variant_spec": enc["spec"],
                "var_type": enc["type"],
                "in_window_idx": enc["in_window_idx"],
                "span_in_window": list(enc["span_in_window"]) if enc["span_in_window"] else None,
                "ch_snp": ch["snp"],
                "ch_ins": ch["ins"],
                "ch_del": ch["del"],
                "ch_any": ch["any"],
            }
            if apply_alt:
                rec["seq_alt"] = apply_variant_to_sequence(seq, w0, enc)

            rows.append(rec)

            # shard out
            if len(rows) >= shard_size:
                df = pd.DataFrame(rows)
                outp = os.path.join(out, f"shard_{shard_idx:04d}.parquet")
                df.to_parquet(outp, index=False)
                print(f"[prepare_variants] wrote {outp} ({len(df)} rows)")
                rows.clear()
                shard_idx += 1

    if rows:
        df = pd.DataFrame(rows)
        outp = os.path.join(out, f"shard_{shard_idx:04d}.parquet")
        df.to_parquet(outp, index=False)
        print(f"[prepare_variants] wrote {outp} ({len(df)} rows)")

    print(f"[prepare_variants] DONE → {out}")


def main():
    args = parse_args()
    run(
        vcf=args.vcf,
        windows=args.windows,
        out=args.out,
        apply_alt=bool(args.apply_alt),
        max_per_window=int(args.max_per_window),
        shard_size=int(args.shard_size),
    )


if __name__ == "__main__":
    main()