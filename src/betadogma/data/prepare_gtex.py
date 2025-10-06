# src/betadogma/data/prepare_gtex.py
"""
Prepare GTEx-like junction usage (PSI) from junction count tables.

Input
-----
One or more junction-count tables with columns (minimally):
  - sample_id : str
  - chrom     : str
  - donor     : int   (0-based, donor site coordinate)
  - acceptor  : int   (0-based, acceptor site coordinate)
  - strand    : str   ('+' or '-')
  - count     : int

Accepted formats: .parquet, .csv, .tsv (auto-detected by file extension).
You may pass a glob for --junctions.

Additionally requires a GTF file to map junctions to genes (by span + strand).

Output
------
- <out_dir>/junction_psi.parquet
  Columns: sample_id, chrom, donor, acceptor, strand, count, donor_total, acceptor_total,
           psi_donor, psi_acceptor, gene_id (if assigned)

- <out_dir>/gene_psi_summary.parquet
  Per-gene aggregates:
    gene_id, n_samples_covered, n_junctions, mean_psi_donor, median_psi_donor,
    mean_psi_acceptor, median_psi_acceptor

Usage
-----
python -m betadogma.data.prepare_gtex \
  --junctions "data/jx/*.parquet" \
  --gtf /abs/path/gencode.v44.annotation.gtf \
  --out data/cache/gtex_psi \
  --min-count 5 --min-total 20 --min-samples 5
"""

from __future__ import annotations
import argparse
import os
from glob import glob
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd


# -----------------------------
# GTF parsing (minimal & fast)
# -----------------------------

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


def iter_gtf_genes(gtf_path: str, allowed_chroms: Optional[set[str]] = None):
    """
    Yield minimal gene records from a GTF: (chrom, start, end, strand, gene_id).
    Coordinates become 0-based, half-open [start, end).
    """
    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feat, start, end, score, strand, frame, attrs = parts
            if feat != "gene":
                continue
            if allowed_chroms and chrom not in allowed_chroms:
                continue
            a = _parse_attrs(attrs)
            gid = a.get("gene_id")
            if not gid:
                continue
            s0 = int(start) - 1  # to 0-based
            e1 = int(end)        # half-open
            yield {"chrom": chrom, "start": s0, "end": e1, "strand": strand, "gene_id": gid}


def build_gene_index(gtf_path: str, allowed_chroms: Optional[List[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int, str]]]:
    """
    Build a simple per-(chrom,strand) index:
      index[(chrom, strand)] = list of (start, end, gene_id), sorted by start
    """
    allowed = set(allowed_chroms) if allowed_chroms else None
    buckets: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = {}
    for rec in iter_gtf_genes(gtf_path, allowed):
        key = (rec["chrom"], rec["strand"])
        buckets.setdefault(key, []).append((rec["start"], rec["end"], rec["gene_id"]))
    for key in buckets:
        buckets[key].sort(key=lambda t: t[0])
    return buckets


def assign_gene_for_junction(
    chrom: str,
    donor: int,
    acceptor: int,
    strand: str,
    gene_index: Dict[Tuple[str, str], List[Tuple[int, int, str]]],
) -> Optional[str]:
    """
    Assign gene_id by containing-span rule: both donor and acceptor must lie
    within [gene.start, gene.end) for the same chrom/strand bucket.

    Returns first matching gene_id (if multiple overlap). If none, returns None.
    """
    key = (chrom, strand)
    spans = gene_index.get(key)
    if not spans:
        return None
    lo = min(donor, acceptor)
    hi = max(donor, acceptor) + 1
    # linear scan is fine for moderate lists; can be replaced with interval tree if needed
    for gs, ge, gid in spans:
        if gs <= lo and hi <= ge:
            return gid
    return None


# -----------------------------
# Junction counts I/O
# -----------------------------

def read_junction_tables(junctions_glob_or_list) -> pd.DataFrame:
    """
    Read one or multiple junction-count files (parquet/csv/tsv), stack into a DataFrame.

    Required columns:
      sample_id, chrom, donor, acceptor, strand, count
    """
    if isinstance(junctions_glob_or_list, str):
        paths = sorted(glob(junctions_glob_or_list))
    else:
        paths = list(junctions_glob_or_list)
    if not paths:
        raise FileNotFoundError("No junction files matched.")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        p_lower = p.lower()
        if p_lower.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif p_lower.endswith(".csv"):
            df = pd.read_csv(p)
        elif p_lower.endswith(".tsv") or p_lower.endswith(".txt"):
            df = pd.read_csv(p, sep="\t")
        else:
            raise ValueError(f"Unsupported junction file extension: {p}")

        # Normalize required columns and dtypes
        required = {"sample_id", "chrom", "donor", "acceptor", "strand", "count"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{p} is missing required columns: {sorted(missing)}")

        df = df[list(required)].copy()
        df["donor"] = df["donor"].astype(int)
        df["acceptor"] = df["acceptor"].astype(int)
        df["count"] = df["count"].astype(int)
        df["strand"] = df["strand"].astype(str)
        df["chrom"] = df["chrom"].astype(str)
        dfs.append(df)

    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


# -----------------------------
# PSI computation
# -----------------------------

def compute_junction_psi(
    df: pd.DataFrame,
    min_count: int = 5,
    min_total: int = 20,
) -> pd.DataFrame:
    """
    Compute per-sample PSI for each junction, two ways:
      - psi_donor   = count / sum counts for junctions sharing the same donor (per sample)
      - psi_acceptor = count / sum counts for junctions sharing the same acceptor (per sample)

    Filters:
      - drop rows with (count < min_count)
      - donor_total and/or acceptor_total < min_total -> PSI becomes NaN

    Returns a new DataFrame with added columns: donor_total, acceptor_total, psi_donor, psi_acceptor
    """
    df = df.copy()

    # Filter low-count junctions early
    df = df[df["count"] >= int(min_count)]

    # donor totals per (sample, chrom, strand, donor)
    donor_tot = (
        df.groupby(["sample_id", "chrom", "strand", "donor"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "donor_total"})
    )

    # acceptor totals per (sample, chrom, strand, acceptor)
    accept_tot = (
        df.groupby(["sample_id", "chrom", "strand", "acceptor"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "acceptor_total"})
    )

    # join totals back
    df = df.merge(donor_tot, on=["sample_id", "chrom", "strand", "donor"], how="left")
    df = df.merge(accept_tot, on=["sample_id", "chrom", "strand", "acceptor"], how="left")

    # PSI with coverage gating
    min_total = float(min_total)
    df["psi_donor"] = df["count"] / df["donor_total"]
    df["psi_acceptor"] = df["count"] / df["acceptor_total"]
    df.loc[df["donor_total"] < min_total, "psi_donor"] = float("nan")
    df.loc[df["acceptor_total"] < min_total, "psi_acceptor"] = float("nan")

    return df


def annotate_genes(
    df: pd.DataFrame,
    gene_index: Dict[Tuple[str, str], List[Tuple[int, int, str]]],
) -> pd.DataFrame:
    """
    Assign gene_id to each junction by containment within gene span.
    """
    def _assign(row):
        return assign_gene_for_junction(
            row["chrom"], int(row["donor"]), int(row["acceptor"]), row["strand"], gene_index
        )
    df = df.copy()
    df["gene_id"] = df.apply(_assign, axis=1)
    return df


def summarize_gene_psi(
    df: pd.DataFrame,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Aggregate per-gene PSI across samples (ignoring NaNs).
    """
    # We'll consider a sample "covered" for a gene if it has >=1 junction row after filtering.
    # PSI stats are computed over available (non-NaN) values.
    grp = df.groupby("gene_id", dropna=True)

    # Helper aggregations
    def _n_samples(g):
        return g["sample_id"].nunique()

    agg = grp.agg(
        n_samples_covered=("sample_id", _n_samples),
        n_junctions=("acceptor", "count"),
        mean_psi_donor=("psi_donor", "mean"),
        median_psi_donor=("psi_donor", "median"),
        mean_psi_acceptor=("psi_acceptor", "mean"),
        median_psi_acceptor=("psi_acceptor", "median"),
    ).reset_index()

    agg = agg[agg["n_samples_covered"] >= int(min_samples)]
    return agg


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compute junction PSI and gene-level summaries from junction count tables.")
    ap.add_argument("--junctions", required=True, help="Glob or list of files (.parquet/.csv/.tsv) with junction counts.")
    ap.add_argument("--gtf", required=True, help="GENCODE GTF path (used to map junctions to genes).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--chroms", type=str, default="", help="Optional comma-separated list of chromosomes to keep (e.g., chr1,chr2).")
    ap.add_argument("--min-count", type=int, default=5, help="Min junction count to keep per sample.")
    ap.add_argument("--min-total", type=int, default=20, help="Min total counts in donor/acceptor group to compute PSI.")
    ap.add_argument("--min-samples", type=int, default=5, help="Min samples per gene for inclusion in gene summary.")
    return ap.parse_args()


def prepare_gtex(
    junctions: str | List[str],
    gtf: str,
    out: str,
    chroms: Optional[str] = None,
    min_count: int = 5,
    min_total: int = 20,
    min_samples: int = 5
) -> None:
    """Prepare GTEx junction data with PSI calculations and gene annotations.
    
    Args:
        junctions: Path or glob pattern to junction count files
        gtf: Path to GTF annotation file
        out: Output directory
        chroms: Comma-separated list of chromosomes to include (None for all)
        min_count: Minimum read count for a junction to be included
        min_total: Minimum total read count for donor/acceptor sites
        min_samples: Minimum number of samples with coverage for gene summary
    """
    os.makedirs(out, exist_ok=True)

    # 1) load junctions
    df = read_junction_tables(junctions)

    # optional chromosome filter
    if chroms:
        keep = set([c.strip() for c in chroms.split(",") if c.strip()])
        df = df[df["chrom"].isin(keep)].copy()

    # 2) compute PSI
    df_psi = compute_junction_psi(df, min_count=min_count, min_total=min_total)

    # 3) gene assignment
    chroms_list = [c for c in sorted(df_psi["chrom"].unique())]
    gene_index = build_gene_index(gtf, allowed_chroms=chroms_list)
    df_psi = annotate_genes(df_psi, gene_index)

    # 4) write junction-level PSI
    junc_out = os.path.join(out, "junction_psi.parquet")
    df_psi.to_parquet(junc_out, index=False)
    print(f"[prepare_gtex] wrote {junc_out} ({len(df_psi):,} rows)")

    # 5) per-gene summary
    gene_sum = summarize_gene_psi(df_psi, min_samples=min_samples)
    gene_out = os.path.join(out, "gene_psi_summary.parquet")
    gene_sum.to_parquet(gene_out, index=False)
    print(f"[prepare_gtex] wrote {gene_out} ({len(gene_sum):,} genes)")


def main():
    args = parse_args()
    prepare_gtex(
        junctions=args.junctions,
        gtf=args.gtf,
        out=args.out,
        chroms=args.chroms,
        min_count=args.min_count,
        min_total=args.min_total,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()