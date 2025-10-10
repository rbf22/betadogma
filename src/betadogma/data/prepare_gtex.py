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
  --out data/cache/gtex_psi \
  --min-count 5 --min-total 20 --min-samples 5
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
from typing_extensions import Literal, TypedDict

import pandas as pd
from tqdm import tqdm  # type: ignore[import-untyped]


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
    # Open gzipped or regular file
    is_gzipped = str(gct_path).endswith('.gz')
    
    def read_header_lines(f: IO[str]) -> List[str]:
        """Read header lines from GCT file."""
        lines = []
        for _ in range(3):  # Read first 3 lines for version, dimensions, and headers
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
    
    # Parse dimensions
    dims = dim_line.split()
    if len(dims) < 2:
        raise ValueError(f"Invalid GCT dimensions. Expected at least 2 values, got {len(dims)}: {dims}")
    
    n_rows = int(dims[0])
    n_cols = int(dims[1])
    
    # Parse headers
    if header_line.endswith('\t'):
        header_line = header_line.rstrip('\t')
    headers = header_line.split('\t')
    
    # First column is 'Name', rest are sample IDs
    sample_ids = [h for h in headers[1:] if h]  # Filter out any empty strings
    
    # In v1.3, there's an extra description column we need to skip
    if version == '#1.3' and len(headers) > 1 and headers[1] == 'Description':
        sample_ids = [h for h in headers[2:] if h]
    
    # Handle cases where the header might have an extra tab
    if len(sample_ids) == n_cols + 1 and (not sample_ids[-1] or sample_ids[-1].isspace()):
        sample_ids = sample_ids[:-1]  # Remove the last empty column
    
    # If we're still off by one, adjust n_cols to match actual data
    if len(sample_ids) != n_cols:
        print(f"Adjusting column count from {n_cols} to {len(sample_ids)} to match actual header")
        n_cols = len(sample_ids)
    
    if len(headers) < 2:
        raise ValueError(f"Invalid GCT header line. Expected at least 2 columns, got {len(headers)}")
    
    # Cap the number of rows if max_rows is provided
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
    
    return sample_ids, n_rows

# Configure multiprocessing for macOS
try:
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    
    # Register cleanup function to handle multiprocessing resources
    def cleanup():
        if mp.current_process().name == 'MainProcess':
            # Cleanup any remaining processes
            for p in mp.active_children():
                p.terminate()
                p.join()
    
    atexit.register(cleanup)
    
except Exception as e:
    print(f"Warning: Could not set multiprocessing start method: {e}")


def convert_gct_to_parquet(
    gct_path: str,
    output_dir: Union[str, Path],
    min_count: int = 2,  # ≥2 reads per junction (two-read coverage)
    min_samples: int = 3,  # ≥N samples with coverage
    smoke: bool = False,
    chroms: Optional[str] = None
) -> List[str]:
    """Convert GTEx junction GCT file to per-sample Parquet files.

    Uses two-read junction coverage criteria:
    - Include junction if ≥2 reads in ≥min_samples samples

    Args:
        gct_path: Path to the input GCT file (gzipped)
        output_dir: Directory to save the output Parquet files
        min_count: Minimum reads per junction per sample (for two-read coverage)
        min_samples: Minimum samples with ≥min_count reads for a junction to be included
        smoke: If True, process a small subset of rows and columns for testing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing GCT file: {gct_path}")
    
    # First, get the header and sample IDs
    with gzip.open(gct_path, 'rt') as f:
        # Read the version and dimension lines
        version = next(f).strip()
        n_rows_total, n_cols_total = map(int, next(f).strip().split('\t')[:2])
        
        # Get header line with sample IDs
        header = next(f).strip().split('\t')
        all_sample_ids = [h for h in header[2:] if h]  # Skip 'Name' and 'Description'
    
    # Configure chromosome filter
    target_chroms = None
    if chroms:
        target_chroms = set([c.strip() for c in chroms.split(",") if c.strip()])
        print(f"[GCT] Filtering to chromosomes: {sorted(target_chroms)}")
    
    # Configure smoke test limits
    max_junctions: Union[int, float]  # Type hint for max_junctions
    if smoke:
        # For smoke test, process only a small subset of junctions and samples
        max_junctions = 5  # Minimal requirement for testing
        max_samples = 10     # Process only first 10 samples for smoke test
        
        # Take first N samples (or all if fewer than max_samples)
        sample_ids = all_sample_ids[:min(max_samples, len(all_sample_ids))]
        
        print(f"[SMOKE TEST] Processing exactly {max_junctions} valid junctions")
        print(f"[SMOKE TEST] Using {len(sample_ids)} of {len(all_sample_ids)} samples")
    else:
        max_junctions = float('inf')
        sample_ids = all_sample_ids
        print(f"Processing all junctions across {len(sample_ids)} samples")
    
    # Initialize data structures with type hints
    junctions: List[Dict[str, Union[str, int]]] = []
    sample_data: Dict[str, List[int]] = {sample_id: [] for sample_id in sample_ids}
    
    # Process the GCT file
    with gzip.open(gct_path, 'rt') as f:
        # Skip header lines (3 lines)
        for _ in range(3):
            next(f)
        
        # Initialize counters
        debug_count = 0
        error_count = 0
        max_errors_to_show = 10
        processed_count = 0
        valid_junctions = 0
        rows_processed = 0  # Initialize rows_processed counter
        
        with tqdm(total=max_junctions if smoke else None, desc="Processing junctions") as progress_bar:
            # Main processing loop - we'll break when we hit max_junctions valid junctions
            while True:
                # If we've reached our target, stop processing
                if smoke and valid_junctions >= max_junctions:
                    break
                try:
                    line = next(f)
                    processed_count += 1
                    rows_processed += 1  # Track total rows read
                    
                    fields = line.strip().split('\t')
                    
                    # Debug: Print first few lines
                    if processed_count <= 5:
                        print(f"\nLine {processed_count} first 5 fields: {fields[:5]}...")
                        print(f"Number of fields: {len(fields)}")
                    
                    # The format appears to be: Junction ID, Gene ID, then sample counts
                    if len(fields) < 2:  # At least need junction ID and gene ID
                        if error_count < max_errors_to_show:
                            print(f"Skipping line {processed_count}: Not enough fields (need at least 2, got {len(fields)})")
                            error_count += 1
                        progress_bar.update(1)
                        continue
                    
                    junction_id = fields[0]
                    gene_id = fields[1]  # Extract the gene ID
                    counts = fields[2:]  # The rest are sample counts
                    
                    # For smoke test, only use counts for the samples we're processing
                    if smoke and len(counts) > len(sample_ids):
                        counts = counts[:len(sample_ids)]
                    
                    # Handle case where we're missing one count (likely a trailing tab in the header)
                    if len(counts) == len(sample_ids) - 1:
                        if processed_count <= 5:
                            print(f"Note: Line {processed_count} has {len(counts)} counts, adding a zero for the last sample")
                        counts.append('0')  # Add a zero count for the missing sample
                    elif len(counts) != len(sample_ids):
                        if error_count < max_errors_to_show:
                            print(f"Warning: Line {processed_count} has {len(counts)} counts but expected {len(sample_ids)} samples")
                            error_count += 1
                        progress_bar.update(1)
                        continue
                    
                    # Debug: Print first few junction IDs
                    if debug_count < 5:
                        print(f"\nSample junction ID {debug_count+1}: {junction_id}")
                        print(f"Sample counts (first 5): {counts[:5]}")
                        debug_count += 1
                    
                    try:
                        # Parse junction ID
                        if ':' in junction_id:
                            chrom, coords_strand = junction_id.split(':', 1)
                            coords, strand = coords_strand.rsplit('_', 1)
                            start, end = map(int, coords.split('-'))
                        else:
                            parts = junction_id.split('_')
                            chrom = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])
                            strand = '+'  # Assume + strand if not specified
                        
                        if processed_count <= 5:
                            print(f"Parsed: chrom={chrom}, start={start}, end={end}, strand={strand}")
                        
                        # Skip if not in target chromosomes
                        if target_chroms and chrom not in target_chroms:
                            if processed_count <= 5:
                                print(f"Skipping junction on {chrom} (not in target chromosomes)")
                            progress_bar.update(1)
                            continue
                        
                        # Add junction to our list
                        junctions.append({
                            'chrom': chrom,
                            'donor': start if strand == '+' else end - 1,  # 0-based
                            'acceptor': end if strand == '+' else start,   # 0-based
                            'strand': strand,
                            'junction_id': junction_id,
                            'gene_id': gene_id
                        })
                        
                        # Store counts for each sample
                        for sample_id, count_str in zip(sample_ids, counts):
                            try:
                                count = int(float(count_str))  # Handle scientific notation
                                sample_data[sample_id].append(count)
                            except ValueError as e:
                                if error_count < max_errors_to_show:
                                    print(f"Warning: Could not parse count '{count_str}' for sample {sample_id}: {e}")
                                    error_count += 1
                                sample_data[sample_id].append(0)  # Default to 0 for invalid counts
                        
                        progress_bar.update(1)
                        
                    except Exception as e:
                        if error_count < max_errors_to_show:
                            print(f"Error parsing junction {processed_count}: {junction_id} - {e}")
                            error_count += 1
                        progress_bar.update(1)
                        
                except StopIteration:
                    if smoke and valid_junctions < max_junctions:
                        print(f"[SMOKE TEST] WARNING: File ended after finding only {valid_junctions} valid junctions (wanted {max_junctions})")
                    elif smoke:
                        print(f"[SMOKE TEST] Successfully processed {valid_junctions} valid junctions")
                    else:
                        print(f"Reached end of file. Processed {valid_junctions} valid junctions")
                    break
        
        print(f"\nProcessed {valid_junctions} valid junctions out of {rows_processed} total lines")
        if error_count > 0:
            print(f"Encountered {error_count} errors (first {min(error_count, max_errors_to_show)} shown)")
        
        if smoke:
            if valid_junctions >= max_junctions:
                print(f"[SMOKE TEST] Successfully processed {max_junctions} valid junctions")
            else:
                print(f"[SMOKE TEST] WARNING: Only found {valid_junctions} valid junctions (wanted {max_junctions})")
    
    if not junctions:
        raise ValueError("No valid junctions found in the GCT file")
    
    # Apply two-read junction coverage filter
    print("\nApplying two-read junction coverage criteria:")
    print(f"  - ≥{min_count} reads per junction per sample")
    print(f"  - ≥{min_samples} samples with coverage per junction")

    # Convert junction info to DataFrame for filtering
    junctions_df = pd.DataFrame(junctions)

    # Filter junctions based on two-read coverage criteria
    qualifying_junctions = []

    for idx, junction in junctions_df.iterrows():
        # Get counts for this junction across all samples
        junction_counts = [sample_data[sample_id][idx] for sample_id in sample_ids]

        # Count samples with ≥2 reads (two-read junction coverage)
        samples_with_coverage = sum(1 for count in junction_counts if count >= min_count)

        if samples_with_coverage >= min_samples:
            qualifying_junctions.append(junction)

    print(f"Filtered to {len(qualifying_junctions)} junctions meeting two-read coverage criteria")

    if not qualifying_junctions:
        raise ValueError(f"No junctions meet the two-read coverage criteria (≥{min_count} reads in ≥{min_samples} samples)")

    # Update junctions_df with only qualifying junctions
    junctions_df = pd.DataFrame(qualifying_junctions)

    # Rebuild sample_data with only qualifying junctions
    filtered_sample_data: Dict[str, List[int]] = {sample_id: [] for sample_id in sample_ids}
    for idx, junction in junctions_df.iterrows():
        for sample_id in sample_ids:
            filtered_sample_data[sample_id].append(sample_data[sample_id][idx])

    sample_data = filtered_sample_data

    output_files = []

    # Write per-sample Parquet files
    for sample_id in tqdm(sample_ids, desc="Writing sample files"):
        # Create sample DataFrame with junction info and counts
        sample_df = junctions_df.copy()
        sample_df['sample_id'] = sample_id
        sample_df['count'] = sample_data[sample_id]

        if not sample_df.empty:
            # Create output filename from sample ID
            safe_sample_id = "".join(c if c.isalnum() else "_" for c in sample_id)
            output_file = output_dir / f"{safe_sample_id}.parquet"

            # Select and order required columns
            sample_df = sample_df[['chrom', 'donor', 'acceptor', 'strand', 'count', 'sample_id']]
            sample_df.to_parquet(output_file, index=False)
            output_files.append(str(output_file))

    print(f"Completed: Processed {len(junctions_df)} junctions meeting criteria, wrote {len(output_files)} files to {output_dir}")
    return output_files

def process_junctions_input(
    junctions: Union[str, List[str]],
    min_count: int = 5,
    min_samples: int = 3,
    smoke: bool = False,
    chroms: Optional[str] = None
) -> List[str]:
    """Process junctions input, handling both GCT and existing Parquet files.
    
    Args:
        junctions: Path or list of paths to junction files or GCT file
        min_count: Minimum reads per junction per sample (for two-read coverage)
        min_samples: Minimum samples with ≥min_count reads for a junction to be included
        smoke: If True, run in smoke test mode (process only a subset of data)
        chroms: Comma-separated list of chromosomes to include (None for all)
    
    Returns:
        List of paths to processed junction files
    """
    if not junctions:
        raise ValueError("No junction files or GCT file provided")
    
    # If it's a single GCT file
    if isinstance(junctions, str) and junctions.endswith(('.gct', '.gct.gz')):
        output_dir = Path(junctions).parent / 'junctions_parquet'
        return convert_gct_to_parquet(
            gct_path=junctions,
            output_dir=output_dir,
            min_count=min_count,
            min_samples=min_samples,
            smoke=smoke,
            chroms=chroms
        )
    
    # If it's a list or glob pattern
    if isinstance(junctions, str):
        files = sorted(glob(junctions))
        if not files:
            raise FileNotFoundError(f"No files matched: {junctions}")
        return files
    
    return list(junctions)

def parse_junction_id(junction_id: str) -> Tuple[str, int, int, str]:
    """Parse a junction ID string into its components.
    
    Expected format: 'chr1:12345-12456:strand' or 'chr1_12345_12456_strand' or 'chr1:12345-12456'
    
    Returns:
        Tuple of (chrom, start, end, strand)
    """
    # Try colon format first: chr1:12345-12456:+
    if ':' in junction_id and '-' in junction_id:
        parts = junction_id.split(':')
        if len(parts) >= 2:
            chrom = parts[0]
            positions = parts[1].split('-')
            if len(positions) == 2:
                start = int(positions[0])
                end = int(positions[1])
                strand = parts[2] if len(parts) > 2 else '+'
                return chrom, start, end, strand
    
    # Try underscore format: chr1_12345_12456_+
    if '_' in junction_id:
        parts = junction_id.split('_')
        if len(parts) >= 3:
            # Reconstruct chrom name in case it contains underscores
            chrom = '_'.join(parts[:-3])
            try:
                start = int(parts[-3])
                end = int(parts[-2])
                strand = parts[-1] if parts[-1] in '+-' else '+'
                return chrom, start, end, strand
            except (ValueError, IndexError):
                pass
    
    # If we get here, try to extract numbers in any format
    numbers = list(map(int, re.findall(r'\d+', junction_id)))
    if len(numbers) >= 2:
        start = min(numbers[0], numbers[1])
        end = max(numbers[0], numbers[1])
        chrom = re.sub(r'[^a-zA-Z0-9_]+', '', junction_id.split(str(numbers[0]))[0])
        strand = '+'  # Default strand if not specified
        return chrom, start, end, strand
    
    raise ValueError(f"Could not parse junction ID: {junction_id}")

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

def read_junction_tables(
    junctions_glob_or_list: Union[str, List[str]],
    min_count: int = 5,
    min_samples: int = 3,
    smoke: bool = False,
    chroms: Optional[str] = None
) -> pd.DataFrame:
    """
    Read one or multiple junction-count files (parquet/csv/tsv/gct), stack into a DataFrame.
    If a GCT file is provided, it will be converted to Parquet format first.

    Required columns:
      sample_id, chrom, donor, acceptor, strand, count
    """
    from tqdm import tqdm
    
    if isinstance(junctions_glob_or_list, str):
        paths = [junctions_glob_or_list]  # Keep as single item to check for GCT
    else:
        paths = list(junctions_glob_or_list)
    
    if not paths:
        raise FileNotFoundError("No junction files provided.")

    # Check if we have a GCT file
    gct_file = next((p for p in paths if p.lower().endswith(('.gct', '.gct.gz'))), None)
    if gct_file:
        print(f"Processing GCT file: {gct_file}")
        output_dir = Path(gct_file).parent / 'junctions_parquet'
        paths = convert_gct_to_parquet(gct_file, output_dir, min_count=min_count, smoke=smoke, chroms=chroms)
    else:
        # Handle glob patterns for non-GCT files
        if isinstance(junctions_glob_or_list, str):
            paths = sorted(glob(junctions_glob_or_list))
            if not paths:
                raise FileNotFoundError(f"No files matched: {junctions_glob_or_list}")

    if not paths:
        raise FileNotFoundError("No junction files found after processing input.")

    # For smoke test, limit to first 10 files
    if smoke and len(paths) > 10:
        print(f"[smoke test] Processing first 10 of {len(paths)} files")
        paths = paths[:10]

    dfs: List[pd.DataFrame] = []
    processed_files = 0
    
    with tqdm(total=len(paths), desc="Reading junction files", unit="file") as pbar:
        for p in paths:
            p_str = str(p)
            p_lower = p_str.lower()
            pbar.set_postfix(file=Path(p_str).name[:20] + ("..." if len(Path(p_str).name) > 20 else ""))
            
            try:
                if p_lower.endswith(".parquet"):
                    df = pd.read_parquet(p_str)
                elif p_lower.endswith(".csv"):
                    df = pd.read_csv(p_str)
                elif p_lower.endswith(".tsv") or p_lower.endswith(".txt"):
                    df = pd.read_csv(p_str, sep="\t")
                else:
                    print(f"\nSkipping unsupported file: {p_str}")
                    continue

                # Normalize required columns and dtypes
                required = {"sample_id", "chrom", "donor", "acceptor", "strand", "count"}
                missing = required - set(df.columns)
                if missing:
                    print(f"\nWarning: {p_str} is missing required columns: {sorted(missing)}")
                    continue

                df = df[list(required)].copy()
                df["donor_total"] = df["donor_total"].astype(float)  # Keep as float for PSI calculation
                df["acceptor_total"] = df["acceptor_total"].astype(float)  # Keep as float for PSI calculation
                df["count"] = df["count"].astype(int)
                df["strand"] = df["strand"].astype(str)
                df["chrom"] = df["chrom"].astype(str)
                
                if not df.empty:
                    dfs.append(df)
                    processed_files += 1
                    pbar.update(1)
                else:
                    print(f"\nWarning: {p_str} is empty after filtering")
                
            except Exception as e:
                print(f"\nError reading {p_str}: {str(e)[:200]}")
                continue

    if not dfs:
        raise ValueError("No valid junction data was loaded from the provided files.")
    
    print(f"\nMerging {len(dfs)} junction tables...")
    with tqdm(total=len(dfs), desc="Merging tables", unit="chunk") as pbar:
        # Process in chunks to avoid high memory usage
        chunk_size = 1000
        chunks = []
        for i in range(0, len(dfs), chunk_size):
            chunk = pd.concat(dfs[i:i+chunk_size], ignore_index=True)
            chunks.append(chunk)
            pbar.update(min(chunk_size, len(dfs) - i))
        
        # Merge chunks
        out = pd.concat(chunks, ignore_index=True)
    
    print(f"Merged {len(dfs)} files into {len(out):,} junctions")
    return out


# -----------------------------
# PSI computation
# -----------------------------

def compute_junction_psi(
    df: pd.DataFrame,
    min_count: int = 5,
    min_samples: int = 3,
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

    # Ensure sample_id column exists
    if 'sample_id' not in df.columns:
        raise ValueError(f"Input dataframe missing required 'sample_id' column. Available columns: {list(df.columns)}")

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
    min_total_float = float(min_total)
    df["psi_donor"] = df["count"].astype(float) / df["donor_total"]
    df["psi_acceptor"] = df["count"].astype(float) / df["acceptor_total"]
    df.loc[df["donor_total"] < min_total_float, "psi_donor"] = float("nan")
    df.loc[df["acceptor_total"] < min_total_float, "psi_acceptor"] = float("nan")

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

    # Ensure sample_id column exists
    if 'sample_id' not in df.columns:
        raise ValueError(f"Input dataframe missing required 'sample_id' column. Available columns: {list(df.columns)}")

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

    # Ensure sample_id column exists
    if 'sample_id' not in df.columns:
        raise ValueError(f"Input dataframe missing required 'sample_id' column. Available columns: {list(df.columns)}")

    grp = df.groupby("gene_id", dropna=True)

    # Helper aggregations
    def _n_samples(g):
        print(f"DEBUG: Group type: {type(g)}")
        if hasattr(g, 'columns'):
            print(f"DEBUG: Grouped dataframe columns: {list(g.columns)}")
            print(f"DEBUG: sample_id in columns: {'sample_id' in g.columns}")
            if 'sample_id' in g.columns:
                print(f"DEBUG: sample_id column dtype: {g['sample_id'].dtype}")
                print(f"DEBUG: sample_id unique values: {g['sample_id'].unique()}")
                return g["sample_id"].nunique()
            else:
                print(f"DEBUG: sample_id not found in group columns")
                return 0  # Return 0 if no sample_id column
        else:
            print(f"DEBUG: Group is Series")
            return 0  # Fallback for Series groups

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



def prepare_gtex(
    junctions: Union[str, List[str]],
    gtf: str,
    out: Union[str, Path],
    chroms: Optional[str] = None,
    min_count: int = 5,
    min_samples: int = 5,
    min_total: int = 20,
    smoke: bool = False
) -> None:
    """Prepare GTEx junction data with PSI calculations and gene annotations.
    
    Args:
        junctions: Path or glob pattern to junction count files or GCT file
        gtf: Path to GTF annotation file
        out: Output directory
        chroms: Comma-separated list of chromosomes to include (None for all)
        min_count: Minimum reads per junction per sample
        min_samples: Minimum samples with sufficient coverage
        min_total: Minimum total read count for donor/acceptor sites
        smoke: If True, run in smoke test mode (process only a subset of data)
    """
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing GTEx data (smoke test: {'ON' if smoke else 'OFF'})")
    print("="*80)
    
    # Process junctions input
    junction_files = process_junctions_input(
        junctions=junctions,
        min_count=min_count,
        min_samples=min_samples,
        smoke=smoke,
        chroms=chroms
    )
    
    if not junction_files:
        raise ValueError("No valid junction files found after processing input")
    
    if smoke and len(junction_files) > 10:
        junction_files = junction_files[:10]
        print(f"[smoke] Using first 10 of {len(junction_files)} junction files")
    else:
        print(f"Processing {len(junction_files)} junction files")
    
    # Read and process junction tables
    df = read_junction_tables(
        junction_files,
        min_count=min_count,
        min_samples=min_samples,
        smoke=smoke,
        chroms=chroms
    )
    
    if df.empty:
        raise ValueError("No valid junction data found after processing")
    
    # Compute PSI values
    df_psi = compute_junction_psi(
        df,
        min_count=min_count,
        min_samples=min_samples,
        min_total=min_total
    )
    
    # Annotate genes
    gene_index = build_gene_index(gtf)
    df_psi = annotate_genes(df_psi, gene_index)
    
    # Save results
    output_file = out_path / "junction_psi.parquet"
    df_psi.to_parquet(output_file, index=False)
    print(f"Saved junction PSI to {output_file}")
    
    # Generate and save gene summary
    gene_summary = summarize_gene_psi(df_psi, min_samples=min_samples)
    summary_file = out_path / "gene_psi_summary.parquet"
    gene_summary.to_parquet(summary_file, index=False)
    print(f"Saved gene summary to {summary_file}")
    print("="*80)
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare GTEx junction PSI data')
    parser.add_argument('--junctions', required=True, help='Path or glob pattern to junction count files or GCT file')
    parser.add_argument('--gtf', required=True, help='Path to GTF annotation file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--chroms', default='', help='Comma-separated list of chromosomes to include (default: all)')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum reads per junction per sample')
    parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples with sufficient coverage')
    parser.add_argument('--min-total', type=int, default=20, help='Minimum total read count for donor/acceptor sites')
    parser.add_argument('--smoke', action='store_true', help='Run in smoke test mode (process only a subset of data)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
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
        smoke=args.smoke
    )

if __name__ == "__main__":
    main()
