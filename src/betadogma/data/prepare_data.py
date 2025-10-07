"""
Convenience CLI: prepare structural training data from FASTA + GTF.

This script delegates to prepare_gencode.main() with the same arguments.
Usage:
  python -m betadogma.data.prepare_data \
      --fasta /path/GRCh38.fa \
      --gtf /path/gencode.v44.gtf \
      --out data/cache/gencode_v44_structural \
      --window 131072 --stride 65536 --bin-size 128 --chroms chr1,chr2
"""

from __future__ import annotations
import argparse
from . import prepare_gencode


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=131072)
    ap.add_argument("--stride", type=int, default=65536)
    ap.add_argument("--bin-size", type=int, default=128)
    ap.add_argument("--chroms", type=str, default="")
    ap.add_argument("--max-shard-bases", type=int, default=50_000_000)
    return ap.parse_args()


def main():
    # This simply forwards to prepare_gencode with the same signature.
    args = parse_args()
    # prepare_gencode.main() already handles the exact same arg set.
    prepare_gencode.main()


import os
import json
import glob
from collections import defaultdict
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import gc  # Added missing import

def load_parquet_files(glob_pattern: str) -> pd.DataFrame:
    """Load and concatenate parquet files matching the glob pattern."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {glob_pattern}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            raise
    
    if not dfs:
        raise ValueError(f"No valid parquet files found in {glob_pattern}")
    
    return pd.concat(dfs, ignore_index=True)

def prepare_data(
    input_dir: str,
    output_dir: str,
    gtex_dir: Optional[str] = None,
    variant_dir: Optional[str] = None,
    split_by: Optional[str] = "chrom",
    keep_columns: Optional[List[str]] = None,
    max_variants_per_window: int = 64,  # Default max variants per window
    **kwargs
) -> None:
    """
    Prepare final training data by aggregating processed features.
    
    Args:
        input_dir: Directory containing input data (from prepare_gencode)
        output_dir: Directory to write processed data
        gtex_dir: Directory containing GTEx junction data
        variant_dir: Directory containing variant data
        split_by: Column to split data by (e.g., 'chrom')
        keep_columns: List of columns to keep in the final output
        max_variants_per_window: Maximum number of variants to track per window
        **kwargs: Additional configuration parameters
    """
    from pathlib import Path
    
    # Set default columns to keep if not specified
    if keep_columns is None:
        keep_columns = [
            "chrom", "start", "end", "seq", 
            "donor", "acceptor", "tss", "polya"
        ]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[prepare_data] Preparing data from {input_dir} to {output_dir}")
    
    # Load genomic windows data
    print("[prepare_data] Loading genomic windows...")
    windows_glob = str(Path(input_dir) / "*.parquet")
    windows_df = load_parquet_files(windows_glob)
    
    # Filter columns if specified
    if keep_columns:
        missing_cols = [col for col in keep_columns if col not in windows_df.columns]
        if missing_cols:
            print(f"[prepare_data] Warning: Missing columns in windows data: {missing_cols}")
        windows_df = windows_df[[col for col in keep_columns if col in windows_df.columns]]
    
    # Process GTEx junction data if available
    if gtex_dir and os.path.exists(gtex_dir):
        print("[prepare_data] Processing GTEx junction data...")
        try:
            gtex_glob = str(Path(gtex_dir) / "*.parquet")
            gtex_df = load_parquet_files(gtex_glob)
            
            # Aggregate GTEx data by junction (sum counts across samples)
            gtex_agg = gtex_df.groupby(["chrom", "donor", "acceptor", "strand", "gene_id"]).agg({
                'count': 'sum',
                'psi_donor': 'mean',
                'psi_acceptor': 'mean'
            }).reset_index()
            
            # Add junction length and midpoint for window matching
            gtex_agg['junction_length'] = gtex_agg['acceptor'] - gtex_agg['donor']
            gtex_agg['junction_midpoint'] = (gtex_agg['donor'] + gtex_agg['acceptor']) // 2
            
            # Merge GTEx data with windows
            # We'll match junctions to windows where the junction midpoint falls within the window
            windows_df['window_midpoint'] = (windows_df['start'] + windows_df['end']) // 2
            
            # For each window, find all junctions where midpoint falls within the window
            # and merge the junction information
            merged_dfs = []
            for _, window in windows_df.iterrows():
                # Find junctions in this window
                window_junctions = gtex_agg[
                    (gtex_agg['chrom'] == window['chrom']) &
                    (gtex_agg['junction_midpoint'] >= window['start']) &
                    (gtex_agg['junction_midpoint'] < window['end'])
                ].copy()
                
                if not window_junctions.empty:
                    # Add window information to junctions
                    for col in ['start', 'end', 'window_midpoint']:
                        window_junctions[col] = window[col]
                    merged_dfs.append(window_junctions)
            
            if merged_dfs:
                # Combine all window-junction mappings
                window_junctions = pd.concat(merged_dfs)
                
                # Aggregate junction metrics per window
                window_metrics = window_junctions.groupby(['chrom', 'start', 'end']).agg({
                    'count': ['sum', 'count', 'mean'],
                    'psi_donor': 'mean',
                    'psi_acceptor': 'mean',
                    'junction_length': ['mean', 'min', 'max']
                }).reset_index()
                
                # Flatten column names
                window_metrics.columns = [f"gtex_{a}_{b}" if b != '' else a for a, b in window_metrics.columns]
                
                # Merge back with windows
                windows_df = pd.merge(
                    windows_df,
                    window_metrics,
                    on=['chrom', 'start', 'end'],
                    how='left'
                )
                
                print(f"[prepare_data] Processed {len(window_junctions)} junctions in {len(window_metrics)} windows")
            else:
                print("[prepare_data] No junctions found in any windows")
                
        except Exception as e:
            print(f"[prepare_data] Error processing GTEx data: {e}")
            raise
    
    # Process variant data if available
    if variant_dir and os.path.exists(variant_dir):
        print("[prepare_data] Processing variant data...")
        try:
            variant_glob = str(Path(variant_dir) / "*.parquet")
            variant_files = sorted(glob.glob(variant_glob))
            
            # Initialize variant counts using a dictionary to track unique variants per window
            variant_counts = {}
            
            # First pass: Collect all unique variant IDs and their window assignments
            variant_windows = {}
            variant_types = {}
            
            # Process variant files in chunks to manage memory
            chunk_size = 5  # Process fewer files at a time to reduce memory usage
            
            # Track which variants we've seen to detect duplicates
            seen_variants = set()
            duplicate_count = 0
            variant_type_distribution = defaultdict(int)
            
            for i in range(0, len(variant_files), chunk_size):
                chunk_files = variant_files[i:i+chunk_size]
                
                # Load chunk of variant files
                for f in chunk_files:
                    try:
                        df = pd.read_parquet(f)
                        if df.empty:
                            continue
                            
                        # Ensure we have the required columns
                        if 'variant_spec' not in df.columns:
                            print(f"[prepare_data] Warning: 'variant_spec' column missing in {f}")
                            continue
                        
                        # Debug: Check for variant types
                        if 'var_type' in df.columns:
                            # Track variant type distribution
                            type_counts = df['var_type'].value_counts().to_dict()
                            for t, count in type_counts.items():
                                variant_type_distribution[t] += count
                            
                            # Specifically check for insertions
                            ins_count = (df['var_type'] == 'INS').sum()
                            if ins_count > 0:
                                print(f"[DEBUG] Found {ins_count} insertions in {f}")
                            
                            # Debug: Show some example insertions if any exist
                            if ins_count > 0:
                                ins_examples = df[df['var_type'] == 'INS'].head(3)
                                print(f"[DEBUG] Insertion examples from {f}:")
                                for _, row in ins_examples.iterrows():
                                    print(f"  - {row['chrom']}:{row['start']} - {row['variant_spec']}")
                            
                        # Create a unique identifier for each variant
                        df['variant_id'] = df['chrom'] + ':' + df['start'].astype(str) + ':' + df['variant_spec']
                        
                        # For each variant, find all windows it belongs to
                        for _, row in df.iterrows():
                            variant_id = row['variant_id']
                            chrom = row['chrom']
                            pos = row['start']
                            
                            # Store variant type if available
                            if 'var_type' in row:
                                variant_types[variant_id] = row['var_type']
                                
                            # Debug: Check for duplicate variants
                            if variant_id in seen_variants:
                                duplicate_count += 1
                                if duplicate_count <= 5:  # Only print first few duplicates
                                    print(f"[DEBUG] Duplicate variant detected: {variant_id}")
                                    print(f"[DEBUG] File: {f}, Position: {chrom}:{pos}, Type: {row.get('var_type', 'UNK')}")
                            else:
                                seen_variants.add(variant_id)
                            
                            # Find all windows containing this variant
                            matching_windows = windows_df[
                                (windows_df['chrom'] == chrom) &
                                (windows_df['start'] <= pos) &
                                (windows_df['end'] > pos)
                            ][['chrom', 'start', 'end']].drop_duplicates()
                            
                            # Update variant_windows with this variant's windows
                            for _, win in matching_windows.iterrows():
                                window_key = (win['chrom'], win['start'], win['end'])
                                if variant_id not in variant_windows:
                                    variant_windows[variant_id] = set()
                                variant_windows[variant_id].add(window_key)
                                
                    except Exception as e:
                        print(f"[prepare_data] Warning: Could not process {f}: {e}")
            
            # Print variant type distribution info
            print(f"[DEBUG] Variant type distribution in input files:")
            for vtype, count in sorted(variant_type_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {vtype}: {count:,} variants")
            
            # Second pass: Count all variants per window and track their types
            # First, sort variants to ensure consistent selection when capping
            sorted_variants = sorted(variant_windows.items(), key=lambda x: x[0])
            
            # Track windows exceeding max variants
            windows_exceeding_max = []
            
            for variant_id, windows in sorted_variants:
                var_type = variant_types.get(variant_id, 'UNK')
                # Sort windows to ensure consistent processing order
                for window_key in sorted(windows):
                    if window_key not in variant_counts:
                        variant_counts[window_key] = {'total': set(), 'types': defaultdict(int)}
                    
                    # Always add variant to the count for accurate statistics
                    current_count = len(variant_counts[window_key]['total'])
                    
                    # Issue warning if we've exceeded max variants but still count them
                    if current_count == max_variants_per_window:
                        chrom, start, end = window_key
                        window_id = f"{chrom}:{start}-{end}"
                        windows_exceeding_max.append(window_id)
                        print(f"[WARNING] Window {window_id} exceeds {max_variants_per_window} variants specified in data.base.config")
                            
                    # Always add the variant to our tracking
                    if variant_id not in variant_counts[window_key]['total']:
                        variant_counts[window_key]['total'].add(variant_id)
                        variant_counts[window_key]['types'][var_type] += 1
            
            if windows_exceeding_max:
                print(f"[WARNING] Total of {len(windows_exceeding_max)} windows exceed the maximum variant count ({max_variants_per_window})")
                print(f"          This may affect training as only up to {max_variants_per_window} variants can be used per window")
            
            # Clear memory
            if 'df' in locals():
                del df
            gc.collect()
            
            # Convert counts to list of dicts with type information
            variant_stats = []
            type_totals = defaultdict(int)
            
            for (chrom, start, end), counts in variant_counts.items():
                total = len(counts['total'])
                type_counts = counts['types']
                
                # Update global type totals
                for t, c in type_counts.items():
                    type_totals[t] += c
                
                variant_stats.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'variant_count': total,
                    **{f"{t.lower()}_count": c for t, c in type_counts.items()}
                })
            
            # Print duplicate variant stats
            print(f"[DEBUG] Total unique variants: {len(seen_variants):,}")
            print(f"[DEBUG] Duplicate variants found: {duplicate_count:,}")
            
            # Debug: Print distribution of variant counts per window
            if variant_stats:
                counts = [vs['variant_count'] for vs in variant_stats]
                print(f"[prepare_data] Variant count distribution:")
                print(f"  - Windows with variants: {len(counts)} of {len(windows_df)} ({len(counts)/len(windows_df)*100:.1f}%)")
                print(f"  - Min variants per window: {min(counts) if counts else 0}")
                print(f"  - Max variants per window: {max(counts) if counts else 0}")
                print(f"  - Mean variants per window: {sum(counts)/len(counts) if counts else 0:.1f}")
                
                # Print variant type distribution
                total_variants = sum(counts)
                if type_totals:
                    print("  - Variant type distribution in assigned windows:")
                    for t, c in sorted(type_totals.items(), key=lambda x: x[1], reverse=True):
                        print(f"    - {t}: {c:,} ({c/total_variants*100:.1f}%)")
                
                # Print top windows by variant count
                top_windows = sorted(variant_stats, key=lambda x: x['variant_count'], reverse=True)[:5]
                print("  - Top 5 windows by variant count:")
                for win in top_windows:
                    win_types = {k[:-6]: v for k, v in win.items() 
                               if k.endswith('_count') and k != 'variant_count'}
                    type_str = ", ".join(f"{k}:{v}" for k, v in win_types.items() if v > 0)
                    print(f"    - {win['chrom']}:{win['start']}-{win['end']}: "
                          f"{win['variant_count']} variants ({type_str})")
            
            if variant_stats:
                # Create DataFrame from variant_stats which already has the correct structure
                variant_stats_df = pd.DataFrame(variant_stats)
                
                # Ensure we don't have duplicate columns before merge
                cols_to_use = variant_stats_df.columns.difference(windows_df.columns)
                cols_to_use = list(cols_to_use) + ['chrom', 'start', 'end']
                
                # Merge variant stats with windows
                windows_df = pd.merge(
                    windows_df,
                    variant_stats_df[cols_to_use],
                    on=['chrom', 'start', 'end'],
                    how='left'
                )
                windows_df["variant_count"] = windows_df["variant_count"].fillna(0).astype(int)
                
                # Calculate variant statistics
                total_variants = sum(vs['variant_count'] for vs in variant_stats)
                windows_with_variants = len([vs for vs in variant_stats if vs['variant_count'] > 0])
                avg_variants_per_window = total_variants / len(windows_df) if not windows_df.empty else 0
                max_variants_in_window = max((vs['variant_count'] for vs in variant_stats), default=0)
                
                # Print detailed statistics
                print(f"[prepare_data] Variant Statistics:")
                print(f"  - Total variants: {total_variants:,}")
                print(f"  - Windows with variants: {windows_with_variants:,} of {len(windows_df):,} ({windows_with_variants/len(windows_df):.1%})")
                print(f"  - Average variants per window: {avg_variants_per_window:.1f}")
                print(f"  - Max variants in a window: {max_variants_in_window:,}")
                
                # Print top 5 windows with most variants
                if max_variants_in_window > 0:
                    top_windows = windows_df.nlargest(5, 'variant_count')[['chrom', 'start', 'end', 'variant_count']]
                    print("  - Top 5 windows by variant count:")
                    for _, row in top_windows.iterrows():
                        print(f"    - {row['chrom']}:{row['start']}-{row['end']}: {row['variant_count']:,} variants")
                        
            else:
                print("[prepare_data] No variants found in any windows")
                windows_df["variant_count"] = 0
                
        except Exception as e:
            print(f"[prepare_data] Error processing variant data: {e}")
            raise
    
    # Save processed data
    print("[prepare_data] Saving processed data...")
    
    # Split data if requested
    if split_by and split_by in windows_df.columns:
        for chrom, group in windows_df.groupby(split_by):
            chrom_dir = output_dir / str(chrom)
            chrom_dir.mkdir(exist_ok=True)
            output_file = chrom_dir / "data.parquet"
            group.to_parquet(output_file, index=False)
        print(f"[prepare_data] Saved data split by {split_by} to {output_dir}")
    else:
        # Save as single file
        output_file = output_dir / "data.parquet"
        windows_df.to_parquet(output_file, index=False)
        print(f"[prepare_data] Saved data to {output_file}")
    
    # Save metadata
    metadata = {
        "num_samples": len(windows_df),
        "columns": list(windows_df.columns),
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_dir": str(input_dir),
        "gtex_dir": str(gtex_dir) if gtex_dir else None,
        "variant_dir": str(variant_dir) if variant_dir else None,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[prepare_data] Data preparation complete. Processed {len(windows_df)} windows.")


if __name__ == "__main__":
    main()