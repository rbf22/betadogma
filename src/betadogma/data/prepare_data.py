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
import logging
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
    ap.add_argument("--safety-limit", type=int, default=2000, 
                    help="Safety limit for variants per window (prevents memory issues, 0 = unlimited)")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    return ap.parse_args()


def setup_logging(debug=False):
    """Configure logging based on debug flag"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    # This simply forwards to prepare_gencode with the same signature.
    args = parse_args()
    # Setup logging
    global logger
    logger = setup_logging(args.debug)
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

# Initialize logger at module level
logger = logging.getLogger(__name__)


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
            logger.error(f"Error reading {f}: {e}")
            raise
    
    if not dfs:
        raise ValueError(f"No valid parquet files found in {glob_pattern}")
    
    return pd.concat(dfs, ignore_index=True)


def prepare_variants_wrapper(
    vcf_path: str, 
    windows_path: str, 
    out_path: str, 
    apply_alt: bool = False,
    safety_limit: int = 2000,
    shard_size: int = 50_000,
    seed: int = 42,
    debug: bool = False
):
    """
    Wrapper for prepare_variants.run with renamed parameters.
    
    This ensures we correctly pass safety_limit to the run function.
    """
    from .prepare_variants import run as prepare_variants_run
    
    # Log what we're doing
    logger.info(f"Running prepare_variants with safety_limit={safety_limit}")
    
    # Call with updated parameter names
    prepare_variants_run(
        vcf=vcf_path,
        windows=windows_path,
        out=out_path,
        apply_alt=apply_alt,
        safety_limit=safety_limit,  # Pass the safety_limit parameter
        max_per_window=0,  # Set to 0 as it's deprecated
        shard_size=shard_size,
        seed=seed,
        debug=debug,
    )


def prepare_data(
    input_dir: str,
    output_dir: str,
    gtex_dir: Optional[str] = None,
    variant_dir: Optional[str] = None,
    split_by: Optional[str] = "chrom",
    keep_columns: Optional[List[str]] = None,
    safety_limit: int = 2000,  # Renamed from max_variants_per_window
    vcf_path: Optional[str] = None,  # Added for direct variant processing
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
        safety_limit: Safety limit for variants per window (prevents memory issues, 0 = unlimited)
        vcf_path: Optional path to VCF file for direct variant processing
        **kwargs: Additional configuration parameters
    """
    from pathlib import Path
    
    # Handle backward compatibility with max_variants_per_window
    if 'max_variants_per_window' in kwargs:
        safety_limit = kwargs['max_variants_per_window']
        logger.info(f"Using max_variants_per_window={safety_limit} as safety_limit")
        
    # If we're directly processing variants from VCF
    if vcf_path and not variant_dir:
        logger.info(f"Processing variants directly from VCF: {vcf_path}")
        variant_out_dir = os.path.join(output_dir, "variants_cache")
        os.makedirs(variant_out_dir, exist_ok=True)
        
        # Call our wrapper to ensure safety_limit is passed correctly
        prepare_variants_wrapper(
            vcf_path=vcf_path,
            windows_path=str(Path(input_dir) / "*.parquet"),
            out_path=variant_out_dir,
            safety_limit=safety_limit,
            debug=kwargs.get('debug', False)
        )
        
        # Update variant_dir to point to our processed output
        variant_dir = variant_out_dir
        logger.info(f"Set variant_dir to {variant_dir}")

    # Set default columns to keep if not specified
    if keep_columns is None:
        keep_columns = [
            "chrom", "start", "end", "seq", 
            "donor", "acceptor", "tss", "polya"
        ]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing data from {input_dir} to {output_dir}")
    
    # Load genomic windows data
    logger.info("Loading genomic windows...")
    windows_glob = str(Path(input_dir) / "*.parquet")
    windows_df = load_parquet_files(windows_glob)
    
    # Filter columns if specified
    if keep_columns:
        missing_cols = [col for col in keep_columns if col not in windows_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in windows data: {missing_cols}")
        windows_df = windows_df[[col for col in keep_columns if col in windows_df.columns]]
    
    # Process GTEx junction data if available
    if gtex_dir and os.path.exists(gtex_dir):
        logger.info("Processing GTEx junction data...")
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
                
                logger.info(f"Processed {len(window_junctions)} junctions in {len(window_metrics)} windows")
            else:
                logger.info("No junctions found in any windows")
                
        except Exception as e:
            logger.error(f"Error processing GTEx data: {e}")
            raise
    
    # Process variant data if available
    if variant_dir and os.path.exists(variant_dir):
        logger.info("Processing variant data...")
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
                            logger.warning(f"'variant_spec' column missing in {f}")
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
                                logger.debug(f"Found {ins_count} insertions in {f}")
                            
                            # Debug: Show some example insertions if any exist
                            if ins_count > 0 and logger.isEnabledFor(logging.DEBUG):
                                ins_examples = df[df['var_type'] == 'INS'].head(3)
                                logger.debug(f"Insertion examples from {f}:")
                                for _, row in ins_examples.iterrows():
                                    logger.debug(f"  - {row['chrom']}:{row['start']} - {row['variant_spec']}")
                            
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
                                    logger.debug(f"Duplicate variant detected: {variant_id}")
                                    logger.debug(f"File: {f}, Position: {chrom}:{pos}, Type: {row.get('var_type', 'UNK')}")
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
                        logger.warning(f"Could not process {f}: {e}")
            
            # Print variant type distribution info
            logger.info(f"Variant type distribution in input files:")
            for vtype, count in sorted(variant_type_distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {vtype}: {count:,} variants")
            
            # Second pass: Count all variants per window and track their types
            # First, sort variants to ensure consistent selection when capping
            sorted_variants = sorted(variant_windows.items(), key=lambda x: x[0])
            
            # Track windows exceeding safety limit
            windows_exceeding_safety = []
            
            for variant_id, windows in sorted_variants:
                var_type = variant_types.get(variant_id, 'UNK')
                # Sort windows to ensure consistent processing order
                for window_key in sorted(windows):
                    if window_key not in variant_counts:
                        variant_counts[window_key] = {'total': set(), 'types': defaultdict(int)}
                    
                    # Always add variant to the count for accurate statistics
                    current_count = len(variant_counts[window_key]['total'])
                    
                    # Track windows that exceed safety limit but still count all variants
                    if safety_limit > 0 and current_count == safety_limit:
                        chrom, start, end = window_key
                        window_id = f"{chrom}:{start}-{end}"
                        windows_exceeding_safety.append(window_id)
                        if len(windows_exceeding_safety) <= 10:  # Limit excessive logging
                            logger.warning(
                                f"Window {window_id} has {current_count:,} variants "
                                f"(exceeding safety_limit={safety_limit}). "
                                f"This is expected with population-level data."
                            )
                            
                    # Always add the variant to our tracking
                    if variant_id not in variant_counts[window_key]['total']:
                        variant_counts[window_key]['total'].add(variant_id)
                        variant_counts[window_key]['types'][var_type] += 1
            
            if windows_exceeding_safety:
                total_exceeded = len(windows_exceeding_safety)
                logger.warning(
                    f"{total_exceeded} windows ({total_exceeded/len(windows_df)*100:.1f}%) "
                    f"have more than {safety_limit:,} variants. This is expected with population-level "
                    f"data and won't affect training as variants are tracked naturally."
                )
            
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
            logger.info(f"Total unique variants: {len(seen_variants):,}")
            logger.info(f"Duplicate variants found: {duplicate_count:,}")
            
            # Enhanced variant statistics
            if variant_stats:
                counts = [vs['variant_count'] for vs in variant_stats]
                windows_with_variants = sum(1 for c in counts if c > 0)
                
                # Calculate percentiles
                if counts:
                    percentiles = np.percentile(counts, [5, 25, 50, 75, 95])
                
                logger.info(f"Variant count distribution:")
                logger.info(f"  - Windows with variants: {windows_with_variants} of {len(windows_df)} ({windows_with_variants/len(windows_df)*100:.1f}%)")
                if counts:
                    logger.info(f"  - Min variants per window: {min(counts)}")
                    logger.info(f"  - 5th percentile: {percentiles[0]:.1f}")
                    logger.info(f"  - 25th percentile: {percentiles[1]:.1f}")
                    logger.info(f"  - Median variants per window: {percentiles[2]:.1f}")
                    logger.info(f"  - 75th percentile: {percentiles[3]:.1f}")
                    logger.info(f"  - 95th percentile: {percentiles[4]:.1f}")
                    logger.info(f"  - Max variants per window: {max(counts)}")
                    logger.info(f"  - Mean variants per window: {sum(counts)/len(counts):.1f}")
                
                # Print variant type distribution
                total_variants = sum(counts)
                if type_totals:
                    logger.info("  - Variant type distribution in assigned windows:")
                    for t, c in sorted(type_totals.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"    - {t}: {c:,} ({c/total_variants*100:.1f}%)")
                
                # Print top windows by variant count
                top_windows = sorted(variant_stats, key=lambda x: x['variant_count'], reverse=True)[:5]
                logger.info("  - Top 5 windows by variant count:")
                for win in top_windows:
                    win_types = {k[:-6]: v for k, v in win.items() 
                               if k.endswith('_count') and k != 'variant_count'}
                    type_str = ", ".join(f"{k}:{v}" for k, v in win_types.items() if v > 0)
                    logger.info(f"    - {win['chrom']}:{win['start']}-{win['end']}: "
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
                
                # Calculate variant statistics for logging
                total_variants = sum(vs['variant_count'] for vs in variant_stats)
                windows_with_variants = len([vs for vs in variant_stats if vs['variant_count'] > 0])
                avg_variants_per_window = total_variants / len(windows_df) if not windows_df.empty else 0
                max_variants_in_window = max((vs['variant_count'] for vs in variant_stats), default=0)
                
                # Log detailed statistics
                logger.info(f"Variant Statistics:")
                logger.info(f"  - Total variants: {total_variants:,}")
                logger.info(f"  - Windows with variants: {windows_with_variants:,} of {len(windows_df):,} ({windows_with_variants/len(windows_df):.1%})")
                logger.info(f"  - Average variants per window: {avg_variants_per_window:.1f}")
                logger.info(f"  - Max variants in a window: {max_variants_in_window:,}")
                
            else:
                logger.info("No variants found in any windows")
                windows_df["variant_count"] = 0
                
        except Exception as e:
            logger.error(f"Error processing variant data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # Save processed data
    logger.info("Saving processed data...")
    
    # Split data if requested
    if split_by and split_by in windows_df.columns:
        for group_val, group in windows_df.groupby(split_by):
            group_dir = output_dir / str(group_val)
            group_dir.mkdir(exist_ok=True)
            output_file = group_dir / "data.parquet"
            group.to_parquet(output_file, index=False)
        logger.info(f"Saved data split by {split_by} to {output_dir}")
    else:
        # Save as single file
        output_file = output_dir / "data.parquet"
        windows_df.to_parquet(output_file, index=False)
        logger.info(f"Saved data to {output_file}")
    
    # Save metadata
    metadata = {
        "num_samples": len(windows_df),
        "columns": list(windows_df.columns),
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_dir": str(input_dir),
        "gtex_dir": str(gtex_dir) if gtex_dir else None,
        "variant_dir": str(variant_dir) if variant_dir else None,
        "safety_limit": safety_limit
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Data preparation complete. Processed {len(windows_df)} windows.")


if __name__ == "__main__":
    main()