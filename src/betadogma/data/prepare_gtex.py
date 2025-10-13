# src/betadogma/data/prepare_gtex.py
"""
Prepare GTEx-like junction usage (PSI) from junction count tables.

Enhanced with detailed progress reporting and ETA calculations.
"""

from __future__ import annotations
import os
import sys
import gzip
import json
import logging
import argparse
import atexit
import multiprocessing as mp
from glob import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable, Iterable, cast, BinaryIO, TextIO, IO
import re
from datetime import datetime, timedelta
from typing_extensions import Literal, TypedDict

import pandas as pd
from tqdm import tqdm  # type: ignore[import-untyped]


# =============================================================================
# Progress Reporting Utilities
# =============================================================================

def log(message: str, flush: bool = True):
    """Print with immediate flush to ensure real-time output."""
    print(message, flush=flush)
    sys.stdout.flush()


class ETACalculator:
    """Calculate estimated time remaining and processing rates."""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    def elapsed_time(self) -> str:
        """Get elapsed time as formatted string."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return str(timedelta(seconds=int(elapsed)))
    
    def get_eta(self, current: int, total: int) -> str:
        """Get ETA string."""
        if current == 0 or total == 0:
            return "calculating..."
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return "calculating..."
        
        rate = current / elapsed  # items per second
        remaining = total - current
        eta_seconds = remaining / rate if rate > 0 else 0
        
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return f"{int(eta_seconds/60)}m {int(eta_seconds%60)}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def get_rate(self, current: int) -> str:
        """Get processing rate."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return "calculating..."
        
        rate = current / elapsed
        if rate > 1:
            return f"{rate:.1f}/s"
        elif rate > 0:
            return f"{1/rate:.1f}s/item"
        else:
            return "0/s"


def print_section_header(title: str):
    """Print a formatted section header."""
    log("\n" + "="*80)
    log(f"  {title}")
    log("="*80)


def print_step_header(step_num: int, title: str):
    """Print a formatted step header."""
    log(f"\n{'─'*80}")
    log(f"Step {step_num}: {title}")
    log(f"{'─'*80}")


def print_summary(title: str, stats: Dict[str, Any]):
    """Print a formatted summary of statistics."""
    log(f"\n✓ {title}:")
    for key, value in stats.items():
        if isinstance(value, int):
            log(f"  • {key}: {value:,}")
        elif isinstance(value, float):
            log(f"  • {key}: {value:.2f}")
        else:
            log(f"  • {key}: {value}")


# =============================================================================
# GCT Header Parsing
# =============================================================================

def parse_gct_header(gct_path: str, max_rows: Optional[int] = None) -> Tuple[List[str], int]:
    """Parse GCT header to get sample names and data start line.
    
    Args:
        gct_path: Path to the GCT file (can be gzipped)
        max_rows: If provided, limit the number of rows to process
        
    Returns:
        Tuple of (sample_ids, n_rows) where:
        - sample_ids: List of sample IDs from the GCT header
        - n_rows: Number of data rows to process (capped at max_rows if provided)
    """
    is_gzipped = str(gct_path).endswith('.gz')
    
    def read_header_lines(f: IO[str]) -> List[str]:
        """Read header lines from GCT file."""
        lines = []
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip())
        return lines
    
    if is_gzipped:
        with gzip.open(gct_path, 'rt') as f:
            lines = read_header_lines(f)
    else:
        with open(gct_path, 'r') as f:
            lines = read_header_lines(f)
    
    if len(lines) < 3:
        raise ValueError(f"Invalid GCT file: expected at least 3 lines, got {len(lines)}")
    
    version = lines[0]
    dim_line = lines[1]
    header_line = lines[2]
    
    if version not in ('#1.2', '#1.3'):
        raise ValueError(f"Unsupported GCT version: {version}. Expected '#1.2' or '#1.3'")
    
    dims = dim_line.split()
    if len(dims) < 2:
        raise ValueError(f"Invalid GCT dimensions. Expected at least 2 values, got {len(dims)}: {dims}")
    
    n_rows = int(dims[0])
    n_cols = int(dims[1])
    
    if header_line.endswith('\t'):
        header_line = header_line.rstrip('\t')
    headers = header_line.split('\t')
    
    sample_ids = [h for h in headers[1:] if h]
    
    if version == '#1.3' and len(headers) > 1 and headers[1] == 'Description':
        sample_ids = [h for h in headers[2:] if h]
    
    if len(sample_ids) == n_cols + 1 and (not sample_ids[-1] or sample_ids[-1].isspace()):
        sample_ids = sample_ids[:-1]
    
    if len(sample_ids) != n_cols:
        log(f"Note: Adjusting column count from {n_cols} to {len(sample_ids)} to match header")
        n_cols = len(sample_ids)
    
    if len(headers) < 2:
        raise ValueError(f"Invalid GCT header line. Expected at least 2 columns, got {len(headers)}")
    
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
    
    return sample_ids, n_rows


# =============================================================================
# Multiprocessing Configuration
# =============================================================================

try:
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    
    def cleanup():
        if mp.current_process().name == 'MainProcess':
            for p in mp.active_children():
                p.terminate()
                p.join()
    
    atexit.register(cleanup)
    
except Exception as e:
    log(f"Warning: Could not set multiprocessing start method: {e}")


# =============================================================================
# GCT to Parquet Conversion
# =============================================================================

def convert_gct_to_parquet(
    gct_path: str,
    output_dir: Union[str, Path],
    min_count: int = 2,
    min_samples: int = 3,
    min_intron_length: int = 20,
    max_intron_length: int = 500000,
    smoke: bool = False,
    chroms: Optional[str] = None
) -> List[str]:
    """Convert GTEx junction GCT file to per-sample Parquet files.

    Args:
        gct_path: Path to the input GCT file (gzipped)
        output_dir: Directory to save the output Parquet files
        min_count: Minimum reads per junction per sample
        min_samples: Minimum samples with ≥min_count reads for junction inclusion
        min_intron_length: Minimum intron length in base pairs
        max_intron_length: Maximum intron length in base pairs
        smoke: If True, process a small subset for testing
        chroms: Comma-separated list of chromosomes to filter
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_section_header("CONVERTING GCT TO PARQUET FORMAT")
    
    log(f"Input: {gct_path}")
    log(f"Output: {output_dir}")
    log(f"Filters:")
    log(f"  • Intron length: {min_intron_length:,} - {max_intron_length:,} bp")
    log(f"  • Min count per sample: {min_count}")
    log(f"  • Min samples: {min_samples}")
    
    # Step 1: Read header
    print_step_header(1, "Reading GCT header")
    eta_calc = ETACalculator()
    
    with gzip.open(gct_path, 'rt') as f:
        version = next(f).strip()
        n_rows_total, n_cols_total = map(int, next(f).strip().split('\t')[:2])
        header = next(f).strip().split('\t')
        all_sample_ids = [h for h in header[2:] if h]
    
    print_summary("Header information", {
        "GCT version": version,
        "Total junctions": n_rows_total,
        "Total samples": len(all_sample_ids)
    })
    
    # Configure filters
    target_chroms = None
    if chroms:
        target_chroms = set([c.strip() for c in chroms.split(",") if c.strip()])
        log(f"\n🔍 Chromosome filter: {sorted(target_chroms)}")
    
    # Configure smoke test
    max_junctions: Union[int, float]
    if smoke:
        max_junctions = 5
        max_samples = 10
        sample_ids = all_sample_ids[:min(max_samples, len(all_sample_ids))]
        log(f"\n⚠️  SMOKE TEST MODE")
        log(f"  • Processing {max_junctions} junctions")
        log(f"  • Using {len(sample_ids)} samples")
    else:
        max_junctions = float('inf')
        sample_ids = all_sample_ids
        log(f"\n✓ Processing all samples: {len(sample_ids):,}")
    
    # Step 2: Parse junctions
    print_step_header(2, "Parsing junction data from GCT")
    
    junctions: List[Dict[str, Union[str, int]]] = []
    sample_data: Dict[str, List[int]] = {sample_id: [] for sample_id in sample_ids}
    
    # Statistics
    stats = {
        'processed': 0,
        'valid': 0,
        'filtered_length': 0,
        'filtered_chrom': 0,
        'parse_errors': 0
    }
    
    with gzip.open(gct_path, 'rt') as f:
        # Skip header
        for _ in range(3):
            next(f)
        
        pbar_total = max_junctions if smoke else n_rows_total
        with tqdm(total=pbar_total, desc="Parsing junctions", unit="jxn") as pbar:
            eta_calc_parse = ETACalculator()
            
            while True:
                if smoke and stats['valid'] >= max_junctions:
                    break
                
                try:
                    line = next(f)
                    stats['processed'] += 1
                    
                    fields = line.strip().split('\t')
                    
                    if len(fields) < 2:
                        stats['parse_errors'] += 1
                        continue
                    
                    junction_id = fields[0]
                    gene_id = fields[1]
                    counts = fields[2:]
                    
                    if smoke and len(counts) > len(sample_ids):
                        counts = counts[:len(sample_ids)]
                    
                    if len(counts) == len(sample_ids) - 1:
                        counts.append('0')
                    elif len(counts) != len(sample_ids):
                        stats['parse_errors'] += 1
                        continue
                    
                    # Parse junction coordinates
                    try:
                        if ':' in junction_id:
                            chrom, coords_strand = junction_id.split(':', 1)
                            coords, strand = coords_strand.rsplit('_', 1)
                            start, end = map(int, coords.split('-'))
                        else:
                            parts = junction_id.split('_')
                            chrom = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])
                            strand = '+'
                        
                        # Filter by intron length
                        intron_length = abs(end - start)
                        if intron_length < min_intron_length or intron_length > max_intron_length:
                            stats['filtered_length'] += 1
                            continue
                        
                        # Filter by chromosome
                        if target_chroms and chrom not in target_chroms:
                            stats['filtered_chrom'] += 1
                            continue
                        
                        # Add junction
                        junctions.append({
                            'chrom': chrom,
                            'donor': start if strand == '+' else end - 1,
                            'acceptor': end if strand == '+' else start,
                            'strand': strand,
                            'junction_id': junction_id,
                            'gene_id': gene_id
                        })
                        
                        # Store counts
                        for sample_id, count_str in zip(sample_ids, counts):
                            try:
                                count = int(float(count_str))
                                sample_data[sample_id].append(count)
                            except ValueError:
                                sample_data[sample_id].append(0)
                                stats['parse_errors'] += 1
                        
                        stats['valid'] += 1
                        pbar.update(1)
                        
                        # Update progress info
                        if stats['processed'] % 1000 == 0:
                            pbar.set_postfix({
                                'valid': f"{stats['valid']:,}",
                                'filtered': f"{stats['filtered_length'] + stats['filtered_chrom']:,}",
                                'rate': eta_calc_parse.get_rate(stats['processed']),
                                'ETA': eta_calc_parse.get_eta(stats['processed'], n_rows_total)
                            })
                        
                    except Exception as e:
                        stats['parse_errors'] += 1
                        if stats['parse_errors'] <= 5:
                            log(f"\n⚠️  Parse error on line {stats['processed']}: {e}")
                        continue
                        
                except StopIteration:
                    break
    
    print_summary("Parsing complete", {
        "Lines processed": stats['processed'],
        "Valid junctions": stats['valid'],
        "Filtered by length": stats['filtered_length'],
        "Filtered by chromosome": stats['filtered_chrom'],
        "Parse errors": stats['parse_errors'],
        "Elapsed time": eta_calc.elapsed_time()
    })
    
    if not junctions:
        raise ValueError("No valid junctions found in GCT file")
    
    # Step 3: Apply coverage filters
    print_step_header(3, "Applying coverage filters")
    
    log(f"Criteria:")
    log(f"  • Min {min_count} reads per junction per sample")
    log(f"  • Min {min_samples} samples with coverage")
    
    junctions_df = pd.DataFrame(junctions)
    qualifying_junctions = []
    
    with tqdm(total=len(junctions_df), desc="Filtering junctions", unit="jxn") as pbar:
        for idx in range(len(junctions_df)):
            junction = junctions_df.iloc[idx]
            junction_counts = [sample_data[sample_id][idx] for sample_id in sample_ids]
            samples_with_coverage = sum(1 for count in junction_counts if count >= min_count)
            
            if samples_with_coverage >= min_samples:
                qualifying_junctions.append(junction)
            
            pbar.update(1)
            
            if idx % 1000 == 0:
                pbar.set_postfix({
                    'passing': f"{len(qualifying_junctions):,}",
                    'rate': f"{100*len(qualifying_junctions)/(idx+1):.1f}%"
                })
    
    print_summary("Coverage filtering complete", {
        "Input junctions": len(junctions_df),
        "Output junctions": len(qualifying_junctions),
        "Pass rate": f"{100*len(qualifying_junctions)/len(junctions_df):.1f}%"
    })
    
    if not qualifying_junctions:
        raise ValueError(f"No junctions meet coverage criteria (≥{min_count} reads in ≥{min_samples} samples)")
    
    junctions_df = pd.DataFrame(qualifying_junctions)
    
    # Rebuild sample data with only qualifying junctions
    filtered_sample_data: Dict[str, List[int]] = {sample_id: [] for sample_id in sample_ids}
    for idx in range(len(junctions_df)):
        for sample_id in sample_ids:
            filtered_sample_data[sample_id].append(sample_data[sample_id][idx])
    
    sample_data = filtered_sample_data
    
    # Step 4: Write per-sample files
    print_step_header(4, "Writing per-sample Parquet files")
    
    output_files = []
    with tqdm(sample_ids, desc="Writing samples", unit="file") as pbar:
        for sample_id in pbar:
            pbar.set_postfix({'sample': sample_id[:30]})
            
            sample_df = junctions_df.copy()
            sample_df['sample_id'] = sample_id
            sample_df['count'] = sample_data[sample_id]
            
            if not sample_df.empty:
                safe_sample_id = "".join(c if c.isalnum() else "_" for c in sample_id)
                output_file = output_dir / f"{safe_sample_id}.parquet"
                sample_df = sample_df[['chrom', 'donor', 'acceptor', 'strand', 'count', 'sample_id']]
                sample_df.to_parquet(output_file, index=False)
                output_files.append(str(output_file))
    
    print_summary("Conversion complete", {
        "Junctions written": len(junctions_df),
        "Sample files created": len(output_files),
        "Output directory": str(output_dir),
        "Total time": eta_calc.elapsed_time()
    })
    
    return output_files


# =============================================================================
# Junction Input Processing
# =============================================================================

def process_junctions_input(
    junctions: Union[str, List[str]],
    min_count: int = 5,
    min_samples: int = 3,
    min_intron_length: int = 20,
    max_intron_length: int = 500000,
    smoke: bool = False,
    chroms: Optional[str] = None
) -> List[str]:
    """Process junctions input, handling both GCT and existing Parquet files."""
    
    if not junctions:
        raise ValueError("No junction files or GCT file provided")
    
    if isinstance(junctions, str) and junctions.endswith(('.gct', '.gct.gz')):
        output_dir = Path(junctions).parent / 'junctions_parquet'
        return convert_gct_to_parquet(
            gct_path=junctions,
            output_dir=output_dir,
            min_count=min_count,
            min_samples=min_samples,
            min_intron_length=min_intron_length,
            max_intron_length=max_intron_length,
            smoke=smoke,
            chroms=chroms
        )
    
    if isinstance(junctions, str):
        files = sorted(glob(junctions))
        if not files:
            raise FileNotFoundError(f"No files matched: {junctions}")
        return files
    
    return list(junctions)


# =============================================================================
# GTF Parsing
# =============================================================================

def _parse_attrs(s: str) -> Dict[str, str]:
    """Parse GTF attribute string."""
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
    """Yield gene records from GTF."""
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
            s0 = int(start) - 1
            e1 = int(end)
            yield {"chrom": chrom, "start": s0, "end": e1, "strand": strand, "gene_id": gid}


def build_gene_index(gtf_path: str, allowed_chroms: Optional[List[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int, str]]]:
    """Build per-(chrom,strand) gene index."""
    print_step_header(0, "Building gene index from GTF")
    
    allowed = set(allowed_chroms) if allowed_chroms else None
    buckets: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = {}
    
    gene_count = 0
    for rec in tqdm(iter_gtf_genes(gtf_path, allowed), desc="Reading genes", unit="gene"):
        key = (rec["chrom"], rec["strand"])
        buckets.setdefault(key, []).append((rec["start"], rec["end"], rec["gene_id"]))
        gene_count += 1
    
    for key in buckets:
        buckets[key].sort(key=lambda t: t[0])
    
    print_summary("Gene index built", {
        "Total genes": gene_count,
        "Chromosomes": len(set(k[0] for k in buckets.keys())),
        "Buckets": len(buckets)
    })
    
    return buckets


def assign_gene_for_junction(
    chrom: str,
    donor: int,
    acceptor: int,
    strand: str,
    gene_index: Dict[Tuple[str, str], List[Tuple[int, int, str]]],
) -> Optional[str]:
    """Assign gene_id by containment."""
    key = (chrom, strand)
    spans = gene_index.get(key)
    if not spans:
        return None
    lo = min(donor, acceptor)
    hi = max(donor, acceptor) + 1
    for gs, ge, gid in spans:
        if gs <= lo and hi <= ge:
            return gid
    return None


# =============================================================================
# Junction Reading
# =============================================================================

def read_junction_tables(
    junctions_glob_or_list: Union[str, List[str]],
    min_count: int = 5,
    min_samples: int = 3,
    smoke: bool = False,
    chroms: Optional[str] = None
) -> pd.DataFrame:
    """Read junction count files and stack into DataFrame."""
    
    print_step_header(0, "Reading junction tables")
    
    if isinstance(junctions_glob_or_list, str):
        paths = [junctions_glob_or_list]
    else:
        paths = list(junctions_glob_or_list)
    
    if not paths:
        raise FileNotFoundError("No junction files provided.")
    
    gct_file = next((p for p in paths if p.lower().endswith(('.gct', '.gct.gz'))), None)
    if gct_file:
        log(f"Detected GCT file: {gct_file}")
        output_dir = Path(gct_file).parent / 'junctions_parquet'
        paths = convert_gct_to_parquet(gct_file, output_dir, min_count=min_count, smoke=smoke, chroms=chroms)
    else:
        if isinstance(junctions_glob_or_list, str):
            paths = sorted(glob(junctions_glob_or_list))
            if not paths:
                raise FileNotFoundError(f"No files matched: {junctions_glob_or_list}")
    
    if not paths:
        raise FileNotFoundError("No junction files found after processing input.")
    
    if smoke and len(paths) > 10:
        log(f"[SMOKE TEST] Processing first 10 of {len(paths)} files")
        paths = paths[:10]
    
    log(f"Reading {len(paths):,} junction files...")
    
    dfs: List[pd.DataFrame] = []
    skipped = 0
    
    with tqdm(total=len(paths), desc="Reading files", unit="file") as pbar:
        for p in paths:
            p_str = str(p)
            p_lower = p_str.lower()
            filename = Path(p_str).name
            pbar.set_postfix({'file': filename[:30]})
            
            try:
                if p_lower.endswith(".parquet"):
                    df = pd.read_parquet(p_str)
                elif p_lower.endswith(".csv"):
                    df = pd.read_csv(p_str)
                elif p_lower.endswith((".tsv", ".txt")):
                    df = pd.read_csv(p_str, sep="\t")
                else:
                    skipped += 1
                    continue
                
                required = {"sample_id", "chrom", "donor", "acceptor", "strand", "count"}
                missing = required - set(df.columns)
                if missing:
                    log(f"\n⚠️  Skipping {filename}: missing columns {sorted(missing)}")
                    skipped += 1
                    continue
                
                df = df[list(required)].copy()
                df["count"] = df["count"].astype(int)
                df["strand"] = df["strand"].astype(str)
                df["chrom"] = df["chrom"].astype(str)
                
                if not df.empty:
                    dfs.append(df)
                
            except Exception as e:
                log(f"\n⚠️  Error reading {filename}: {str(e)[:100]}")
                skipped += 1
            
            pbar.update(1)
    
    if not dfs:
        raise ValueError("No valid junction data was loaded.")
    
    log(f"\nMerging {len(dfs):,} junction tables...")
    out = pd.concat(dfs, ignore_index=True)
    
    print_summary("Junction reading complete", {
        "Files read": len(dfs),
        "Files skipped": skipped,
        "Total junctions": len(out),
        "Unique samples": out['sample_id'].nunique()
    })
    
    return out


# =============================================================================
# PSI Computation
# =============================================================================

def compute_junction_psi(
    df: pd.DataFrame,
    min_count: int = 5,
    min_samples: int = 3,
    min_total: int = 20,
) -> pd.DataFrame:
    """Compute per-sample PSI for each junction."""
    
    print_step_header(0, "Computing PSI values")
    
    log(f"Filters:")
    log(f"  • Min count: {min_count}")
    log(f"  • Min samples: {min_samples}")
    log(f"  • Min total: {min_total}")
    
    df = df.copy()
    
    if 'sample_id' not in df.columns:
        raise ValueError(f"Missing 'sample_id' column. Available: {list(df.columns)}")
    
    initial_count = len(df)
    df = df[df["count"] >= int(min_count)]
    log(f"\n✓ Filtered by count: {len(df):,} / {initial_count:,} junctions retained")
    
    log("\nCalculating donor totals...")
    donor_tot = (
        df.groupby(["sample_id", "chrom", "strand", "donor"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "donor_total"})
    )
    
    log("Calculating acceptor totals...")
    accept_tot = (
        df.groupby(["sample_id", "chrom", "strand", "acceptor"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "acceptor_total"})
    )
    
    log("Merging totals...")
    df = df.merge(donor_tot, on=["sample_id", "chrom", "strand", "donor"], how="left")
    df = df.merge(accept_tot, on=["sample_id", "chrom", "strand", "acceptor"], how="left")
    
    log("Computing PSI...")
    min_total_float = float(min_total)
    df["psi_donor"] = df["count"].astype(float) / df["donor_total"]
    df["psi_acceptor"] = df["count"].astype(float) / df["acceptor_total"]
    df.loc[df["donor_total"] < min_total_float, "psi_donor"] = float("nan")
    df.loc[df["acceptor_total"] < min_total_float, "psi_acceptor"] = float("nan")
    
    print_summary("PSI computation complete", {
        "Total junctions": len(df),
        "Mean PSI (donor)": df["psi_donor"].mean(),
        "Mean PSI (acceptor)": df["psi_acceptor"].mean()
    })
    
    return df


def annotate_genes(
    df: pd.DataFrame,
    gene_index: Dict[Tuple[str, str], List[Tuple[int, int, str]]],
) -> pd.DataFrame:
    """Assign gene_id to each junction by containment."""
    
    print_step_header(0, "Annotating junctions with genes")
    
    def _assign(row):
        return assign_gene_for_junction(
            row["chrom"], int(row["donor"]), int(row["acceptor"]), row["strand"], gene_index
        )
    
    df = df.copy()
    
    if 'sample_id' not in df.columns:
        raise ValueError(f"Missing 'sample_id' column. Available: {list(df.columns)}")
    
    log(f"Assigning genes to {len(df):,} junctions...")
    
    tqdm.pandas(desc="Annotating genes")
    df["gene_id"] = df.progress_apply(_assign, axis=1)
    
    assigned = df["gene_id"].notna().sum()
    print_summary("Gene annotation complete", {
        "Total junctions": len(df),
        "Junctions with genes": assigned,
        "Assignment rate": f"{100*assigned/len(df):.1f}%",
        "Unique genes": df["gene_id"].nunique()
    })
    
    return df


def summarize_gene_psi(
    df: pd.DataFrame,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Aggregate per-gene PSI across samples."""
    
    print_step_header(0, "Generating gene summary")
    
    if 'sample_id' not in df.columns:
        raise ValueError(f"Missing 'sample_id' column. Available: {list(df.columns)}")
    
    grp = df.groupby("gene_id", dropna=True)
    
    def _n_samples(x):
        return x.nunique()
    
    log("Aggregating gene statistics...")
    agg = grp.agg(
        n_samples_covered=("sample_id", _n_samples),
        n_junctions=("acceptor", "count"),
        mean_psi_donor=("psi_donor", "mean"),
        median_psi_donor=("psi_donor", "median"),
        mean_psi_acceptor=("psi_acceptor", "mean"),
        median_psi_acceptor=("psi_acceptor", "median"),
    ).reset_index()
    
    initial_genes = len(agg)
    agg = agg[agg["n_samples_covered"] >= int(min_samples)]
    
    print_summary("Gene summary complete", {
        "Total genes": initial_genes,
        "Genes with coverage": len(agg),
        "Mean junctions per gene": agg["n_junctions"].mean()
    })
    
    return agg


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_gtex(
    junctions: Union[str, List[str]],
    gtf: str,
    out: Union[str, Path],
    chroms: Optional[str] = None,
    min_count: int = 5,
    min_samples: int = 5,
    min_total: int = 20,
    min_intron_length: int = 20,
    max_intron_length: int = 500000,
    n_jobs: int = -1,
    smoke: bool = False
) -> None:
    """Prepare GTEx junction data with PSI calculations and gene annotations."""
    
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if n_jobs == -1:
        n_jobs_actual = mp.cpu_count()
    else:
        n_jobs_actual = max(1, n_jobs)
    
    print_section_header("GTEX JUNCTION PSI PREPARATION")
    
    log(f"Configuration:")
    log(f"  • Output directory: {out_path}")
    log(f"  • Chromosomes: {chroms if chroms else 'all'}")
    log(f"  • CPU cores: {n_jobs_actual}")
    log(f"  • Smoke test: {'ON' if smoke else 'OFF'}")
    log(f"  • Intron length: {min_intron_length:,} - {max_intron_length:,} bp")
    log(f"  • Min count: {min_count}")
    log(f"  • Min samples: {min_samples}")
    log(f"  • Min total: {min_total}")
    
    overall_start = datetime.now()
    
    # Process junctions input
    junction_files = process_junctions_input(
        junctions=junctions,
        min_count=min_count,
        min_samples=min_samples,
        min_intron_length=min_intron_length,
        max_intron_length=max_intron_length,
        smoke=smoke,
        chroms=chroms
    )
    
    if not junction_files:
        raise ValueError("No valid junction files found after processing input")
    
    if smoke and len(junction_files) > 10:
        junction_files = junction_files[:10]
        log(f"\n[SMOKE TEST] Using first 10 of {len(junction_files)} junction files")
    
    # Read junction tables
    df = read_junction_tables(
        junction_files,
        min_count=min_count,
        min_samples=min_samples,
        smoke=smoke,
        chroms=chroms
    )
    
    if df.empty:
        raise ValueError("No valid junction data found after processing")
    
    # Filter by intron length (for non-GCT inputs)
    if not str(junctions).endswith(('.gct', '.gct.gz')):
        print_step_header(0, "Filtering by intron length")
        initial_count = len(df)
        df['intron_length'] = abs(df['acceptor'] - df['donor'])
        df = df[
            (df['intron_length'] >= min_intron_length) & 
            (df['intron_length'] <= max_intron_length)
        ]
        print_summary("Intron length filtering", {
            "Input junctions": initial_count,
            "Output junctions": len(df),
            "Pass rate": f"{100*len(df)/initial_count:.1f}%"
        })
        
        if df.empty:
            raise ValueError(f"No junctions remaining after intron length filter")
        
        df = df.drop(columns=['intron_length'])
    
    # Compute PSI
    df_psi = compute_junction_psi(
        df,
        min_count=min_count,
        min_samples=min_samples,
        min_total=min_total
    )
    
    # Annotate genes
    gene_index = build_gene_index(gtf)
    df_psi = annotate_genes(df_psi, gene_index)
    
    # Save junction PSI
    print_step_header(0, "Saving results")
    
    output_file = out_path / "junction_psi.parquet"
    log(f"Writing junction PSI to {output_file}...")
    df_psi.to_parquet(output_file, index=False)
    log(f"✓ Saved {len(df_psi):,} junctions ({output_file.stat().st_size / 1e6:.1f} MB)")
    
    # Generate gene summary
    gene_summary = summarize_gene_psi(df_psi, min_samples=min_samples)
    summary_file = out_path / "gene_psi_summary.parquet"
    log(f"Writing gene summary to {summary_file}...")
    gene_summary.to_parquet(summary_file, index=False)
    log(f"✓ Saved {len(gene_summary):,} genes ({summary_file.stat().st_size / 1e6:.1f} MB)")
    
    # Final summary
    total_time = (datetime.now() - overall_start).total_seconds()
    
    print_section_header("PIPELINE COMPLETE")
    print_summary("Final statistics", {
        "Total junctions": len(df_psi),
        "Total genes": len(gene_summary),
        "Output files": 2,
        "Output directory": str(out_path),
        "Total time": str(timedelta(seconds=int(total_time)))
    })


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare GTEx junction PSI data')
    parser.add_argument('--junctions', required=True, help='Path or glob pattern to junction count files or GCT file')
    parser.add_argument('--gtf', required=True, help='Path to GTF annotation file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--chroms', default='', help='Comma-separated list of chromosomes to include (default: all)')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum reads per junction per sample')
    parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples with sufficient coverage')
    parser.add_argument('--min-total', type=int, default=20, help='Minimum total read count for donor/acceptor sites')
    parser.add_argument('--min-intron-length', type=int, default=20, help='Minimum intron length (bp)')
    parser.add_argument('--max-intron-length', type=int, default=500000, help='Maximum intron length (bp)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--smoke', action='store_true', help='Run in smoke test mode (process only a subset of data)')
    
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    prepare_gtex(
        junctions=args.junctions,
        gtf=args.gtf,
        out=str(out_path),
        chroms=args.chroms if args.chroms else None,
        min_count=args.min_count,
        min_samples=args.min_samples,
        min_total=args.min_total,
        min_intron_length=args.min_intron_length,
        max_intron_length=args.max_intron_length,
        n_jobs=args.n_jobs,
        smoke=args.smoke
    )


if __name__ == "__main__":
    main()