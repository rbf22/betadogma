"""
I/O helpers for FASTA/VCF/Parquet/HDF5 (minimal, dependency-light).
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Generator, Optional, List

import os
import pandas as pd

# FASTA (pyfaidx is a light dependency used elsewhere in the repo)
try:
    from pyfaidx import Fasta as _Fasta
except Exception:  # pragma: no cover
    _Fasta = None


def read_fasta_window(fasta_path: str, chrom: str, start: int, end: int, uppercase: bool = True) -> str:
    """
    Return FASTA slice [start, end) for a chromosome, padding with 'N' if out-of-range.
    """
    if _Fasta is None:
        raise RuntimeError("pyfaidx is not installed; `pip install pyfaidx`.")
    fa = _Fasta(fasta_path, as_raw=True, sequence_always_upper=uppercase)
    if chrom not in fa:
        raise KeyError(f"Chromosome {chrom} not found in FASTA.")
    contig_len = len(fa[chrom])
    s = max(0, start)
    e = min(end, contig_len)
    core = str(fa[chrom][s:e])
    left_pad = "N" * max(0, 0 - min(0, start))
    right_pad = "N" * max(0, end - contig_len)
    return left_pad + core + right_pad


def read_parquet_shards(paths_glob_or_list) -> pd.DataFrame:
    """
    Read one or multiple Parquet files and return a concatenated DataFrame.
    Accepts a glob string or a list of paths.
    """
    if isinstance(paths_glob_or_list, str):
        from glob import glob
        paths = sorted(glob(paths_glob_or_list))
    else:
        paths = list(paths_glob_or_list)
    if not paths:
        raise FileNotFoundError("No Parquet files matched.")
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, axis=0, ignore_index=True)


def write_parquet(df: pd.DataFrame, out_path: str, exist_ok: bool = True) -> None:
    """
    Write a DataFrame to Parquet, creating parent dirs if needed.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if (not exist_ok) and os.path.exists(out_path):
        raise FileExistsError(out_path)
    df.to_parquet(out_path, index=False)


def iter_vcf_records(vcf_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Minimal VCF reader (no external deps). Yields dicts with keys:
      chrom, pos (1-based), id, ref, alt (list[str]), qual, filter, info (raw str)

    Note: This does not parse INFO/FORMAT; it’s enough for simple variant sweeps.
    """
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, flt, info = parts[:8]
            yield {
                "chrom": chrom,
                "pos": int(pos),
                "id": vid if vid != "." else None,
                "ref": ref,
                "alt": alt.split(","),
                "qual": None if qual == "." else float(qual),
                "filter": flt,
                "info": info,
            }