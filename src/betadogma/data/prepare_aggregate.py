#!/usr/bin/env python3
"""
Aggregate all processed features into final training format with train/test split.

This script merges outputs from all previous processing steps:
  1. Base structural windows (gencode_windows_base/)
  2. GTEx PSI values (gtex_psi/)
  3. Population variants (variants_overlapping/ or variants_base/)
  4. Validated splice variants (splice_variants/)

Then creates an 80/20 train/test split and saves the final datasets.

Usage:
  python prepare_aggregate.py \\
    --input-dir data/cache/chr21/gencode_windows_base \\
    --gtex-dir data/cache/chr21/gtex_psi \\
    --variant-dir data/cache/chr21/variants_overlapping \\
    --splice-dir data/cache/chr21/splice_variants \\
    --output-dir data/processed/chr21 \\
    --split-by chrom \\
    --test-size 0.2 \\
    --keep-columns chrom start end seq variant_mask psi training_label
"""

import argparse
import os
import sys
import json
import logging
import gc
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate all processed features into final training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script merges outputs from all previous processing steps:
  1. Base structural windows (gencode_windows_base/)
  2. GTEx PSI values (gtex_psi/)
  3. Population variants (variants_overlapping/ or variants_base/)
  4. Validated splice variants (splice_variants/)

Example:
  python prepare_aggregate.py \\
    --input-dir data/cache/chr21/gencode_windows_base \\
    --gtex-dir data/cache/chr21/gtex_psi \\
    --variant-dir data/cache/chr21/variants_overlapping \\
    --splice-dir data/cache/chr21/splice_variants \\
    --output-dir data/processed/chr21 \\
    --split-by chrom \\
    --test-size 0.2 \\
    --random-seed 42
        """
    )
    
    # Input directories (all outputs from previous steps)
    parser.add_argument('--input-dir', required=True,
                        help='Base structural windows (gencode_windows_base/)')
    parser.add_argument('--gtex-dir', required=True,
                        help='GTEx PSI values (gtex_psi/)')
    parser.add_argument('--variant-dir', required=True,
                        help='Variant windows (variants_overlapping/ or variants_base/)')
    parser.add_argument('--splice-dir', required=False, default=None,
                        help='Splice variant annotations (splice_variants/)')
    
    # Output configuration
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for final training data')
    parser.add_argument('--split-by', default='chrom',
                        choices=['chrom', 'none'],
                        help='Split output by chromosome')
    
    # Train/test split
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--stratify', action='store_true', default=True,
                        help='Stratify split by training label (default: True)')
    parser.add_argument('--group-by-chrom', action='store_true', default=True,
                        help='Keep chromosomes together in train/test (default: True)')
    
    # Data format
    parser.add_argument('--format', default='parquet',
                        choices=['parquet', 'hdf5'],
                        help='Output format (default: parquet)')
    parser.add_argument('--compression', default='snappy',
                        help='Compression codec (default: snappy)')
    
    # Column selection
    parser.add_argument('--keep-columns', action='append', dest='keep_columns',
                        help='Columns to keep in final output (can specify multiple)')
    parser.add_argument('--drop-columns', action='append', dest='drop_columns',
                        help='Columns to drop from final output')
    
    # Filtering
    parser.add_argument('--max-variants-per-window', type=int, default=0,
                        help='Max variants per window (0=unlimited, default: 0)')
    parser.add_argument('--min-coverage', type=float, default=0.0,
                        help='Minimum sequence coverage required (default: 0.0)')
    parser.add_argument('--min-psi', type=float, default=0.0,
                        help='Minimum PSI value for inclusion (default: 0.0)')
    
    # Processing
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate output data (default: True)')
    parser.add_argument('--write-index', action='store_true',
                        help='Write index files')
    parser.add_argument('--write-stats', action='store_true', default=True,
                        help='Write statistics (default: True)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

def load_parquet_dir(directory: str, pattern: str = "*.parquet") -> pd.DataFrame:
    """Load all parquet files from a directory."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    files = sorted(glob(os.path.join(directory, pattern)))
    if not files:
        raise ValueError(f"No parquet files found in {directory}")
    
    logger.info(f"  Loading {len(files)} parquet files...")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# =============================================================================
# GTEx Merging
# =============================================================================

def merge_gtex_by_coordinates(
    windows: pd.DataFrame, 
    gtex_dir: str
) -> pd.DataFrame:
    """
    Merge GTEx PSI values with windows by genomic coordinates.
    
    GTEx has junction coordinates (chrom, donor, acceptor positions).
    Windows have regions (chrom, start, end).
    We need to find junctions that overlap with each window.
    
    Args:
        windows: Base windows DataFrame
        gtex_dir: Path to GTEx PSI directory
    
    Returns:
        DataFrame with GTEx data merged
    """
    logger.info("\n📂 Loading GTEx PSI values...")
    
    if not os.path.exists(gtex_dir):
        logger.warning(f"  GTEx directory not found: {gtex_dir}")
        logger.warning(f"  Continuing without GTEx data")
        
        # Add empty columns
        windows['psi_mean'] = 0.0
        windows['psi_max'] = 0.0
        windows['num_junctions'] = 0
        windows['has_gtex_data'] = False
        return windows
    
    gtex_data = load_parquet_dir(gtex_dir)
    logger.info(f"  Loaded {len(gtex_data):,} GTEx records")
    logger.info(f"  GTEx columns: {list(gtex_data.columns)}")
    
    # GTEx has: chrom, donor (position), acceptor (position), psi_donor, psi_acceptor
    # Windows have: chrom, start, end
    # We need to find junctions where donor/acceptor fall within [start, end)
    
    logger.info("  Merging GTEx data with windows (interval overlap)...")
    
    # Prepare GTEx data for merging
    # Rename to avoid confusion with window's donor/acceptor boolean flags
    gtex_prep = gtex_data.rename(columns={
        'donor': 'junction_donor',
        'acceptor': 'junction_acceptor'
    }).copy()
    
    # For efficiency, only keep relevant columns
    gtex_cols = ['chrom', 'junction_donor', 'junction_acceptor']
    
    # Add PSI columns (handle different naming conventions)
    if 'psi_donor' in gtex_prep.columns:
        gtex_cols.append('psi_donor')
    if 'psi_acceptor' in gtex_prep.columns:
        gtex_cols.append('psi_acceptor')
    if 'mean_psi_donor' in gtex_prep.columns:
        gtex_cols.append('mean_psi_donor')
    if 'mean_psi_acceptor' in gtex_prep.columns:
        gtex_cols.append('mean_psi_acceptor')
    if 'gene_id' in gtex_prep.columns:
        gtex_cols.append('gene_id')
    
    available_cols = [c for c in gtex_cols if c in gtex_prep.columns]
    gtex_prep = gtex_prep[available_cols]
    
    # Method 1: Use interval join (more accurate but slower)
    # For each window, find all junctions that overlap
    
    logger.info("  Building interval index for efficient junction lookup...")
    
    # Group by chromosome for efficiency
    merged_parts = []
    
    for chrom in windows['chrom'].unique():
        chrom_windows = windows[windows['chrom'] == chrom].copy()
        chrom_gtex = gtex_prep[gtex_prep['chrom'] == chrom]
        
        if len(chrom_gtex) == 0:
            # No GTEx data for this chromosome
            chrom_windows['psi_mean'] = 0.0
            chrom_windows['psi_max'] = 0.0
            chrom_windows['num_junctions'] = 0
            chrom_windows['has_gtex_data'] = False
            merged_parts.append(chrom_windows)
            continue
        
        logger.info(f"    {chrom}: {len(chrom_windows)} windows, {len(chrom_gtex):,} junctions")
        
        # For each window, find overlapping junctions
        window_gtex_data = []
        
        for idx, window in chrom_windows.iterrows():
            w_start = window['start']
            w_end = window['end']
            
            # Find junctions where either donor or acceptor falls in window
            # Or where the junction spans the window
            overlapping = chrom_gtex[
                (
                    # Donor in window
                    (chrom_gtex['junction_donor'] >= w_start) & 
                    (chrom_gtex['junction_donor'] < w_end)
                ) | (
                    # Acceptor in window
                    (chrom_gtex['junction_acceptor'] >= w_start) & 
                    (chrom_gtex['junction_acceptor'] < w_end)
                ) | (
                    # Junction spans window
                    (chrom_gtex['junction_donor'] < w_start) & 
                    (chrom_gtex['junction_acceptor'] >= w_end)
                )
            ]
            
            if len(overlapping) > 0:
                # Calculate PSI statistics for this window
                psi_values = []
                
                if 'psi_donor' in overlapping.columns:
                    psi_values.extend(overlapping['psi_donor'].dropna().tolist())
                if 'psi_acceptor' in overlapping.columns:
                    psi_values.extend(overlapping['psi_acceptor'].dropna().tolist())
                if 'mean_psi_donor' in overlapping.columns:
                    psi_values.extend(overlapping['mean_psi_donor'].dropna().tolist())
                if 'mean_psi_acceptor' in overlapping.columns:
                    psi_values.extend(overlapping['mean_psi_acceptor'].dropna().tolist())
                
                psi_values = [p for p in psi_values if p > 0]  # Filter out zeros
                
                window_gtex_data.append({
                    'window_idx': idx,
                    'psi_mean': np.mean(psi_values) if psi_values else 0.0,
                    'psi_max': np.max(psi_values) if psi_values else 0.0,
                    'num_junctions': len(overlapping),
                    'has_gtex_data': True
                })
            else:
                window_gtex_data.append({
                    'window_idx': idx,
                    'psi_mean': 0.0,
                    'psi_max': 0.0,
                    'num_junctions': 0,
                    'has_gtex_data': False
                })
        
        # Create DataFrame and merge back
        gtex_summary = pd.DataFrame(window_gtex_data)
        chrom_windows = chrom_windows.reset_index(drop=True)
        chrom_windows['window_idx'] = chrom_windows.index
        
        chrom_merged = chrom_windows.merge(
            gtex_summary,
            on='window_idx',
            how='left'
        ).drop(columns=['window_idx'])
        
        merged_parts.append(chrom_merged)
    
    merged = pd.concat(merged_parts, ignore_index=True)
    
    # Fill any remaining NaN values
    merged['psi_mean'] = merged['psi_mean'].fillna(0.0)
    merged['psi_max'] = merged['psi_max'].fillna(0.0)
    merged['num_junctions'] = merged['num_junctions'].fillna(0).astype(int)
    merged['has_gtex_data'] = merged['has_gtex_data'].fillna(False)
    
    windows_with_gtex = merged['has_gtex_data'].sum()
    total_junctions = merged['num_junctions'].sum()
    
    logger.info(f"  Windows with GTEx data: {windows_with_gtex:,} / {len(merged):,}")
    logger.info(f"  Total junctions mapped: {total_junctions:,}")
    
    if windows_with_gtex > 0:
        avg_psi = merged[merged['has_gtex_data']]['psi_mean'].mean()
        logger.info(f"  Average PSI (windows with data): {avg_psi:.3f}")
    
    return merged


# =============================================================================
# Variant Merging
# =============================================================================

def merge_variants_with_windows(
    windows: pd.DataFrame,
    variant_dir: str
) -> pd.DataFrame:
    """Merge population variant data with windows."""
    logger.info("\n📂 Loading population variants...")
    
    if not os.path.exists(variant_dir):
        logger.warning(f"  Variant directory not found: {variant_dir}")
        windows['has_variant'] = False
        windows['num_variants'] = 0
        windows['variant_mask'] = None
        windows['variant_info'] = None
        if 'seq_alt' not in windows.columns:
            windows['seq_alt'] = windows.get('seq', '')
        return windows
    
    variants = load_parquet_dir(variant_dir)
    logger.info(f"  Loaded {len(variants):,} variant records")
    
    # Check for duplicates in variant data
    variant_dupes = variants.duplicated(subset=['chrom', 'start', 'end'], keep=False)
    if variant_dupes.any():
        logger.info(f"  Found {variant_dupes.sum():,} duplicate variant records")
        logger.info(f"  Unique windows in variants: {variants[['chrom', 'start', 'end']].drop_duplicates().shape[0]:,}")
    
    # Build aggregation dictionary
    agg_dict = {}
    
    if 'var_type' in variants.columns:
        agg_dict['var_type'] = lambda x: list(x)
    
    if 'in_window_idx' in variants.columns:
        agg_dict['in_window_idx'] = lambda x: list(x)
    
    if 'variant_spec' in variants.columns:
        agg_dict['variant_spec'] = lambda x: list(x)
    
    if 'seq_alt' in variants.columns:
        agg_dict['seq_alt'] = 'first'
    
    logger.info("  Aggregating variants per window...")
    variant_agg = variants.groupby(['chrom', 'start', 'end'], as_index=False).agg(agg_dict)
    
    logger.info(f"  Aggregated to {len(variant_agg):,} unique windows")
    
    variant_agg['num_variants'] = variant_agg['var_type'].apply(len) if 'var_type' in variant_agg.columns else 1
    variant_agg['has_variant'] = True
    
    # Create variant_mask
    if 'in_window_idx' in variant_agg.columns:
        def create_mask(indices, window_size=131072):
            mask = np.zeros(window_size, dtype=bool)
            for idx in indices:
                if isinstance(idx, (int, np.integer)) and 0 <= idx < window_size:
                    mask[idx] = True
            return mask
        
        variant_agg['variant_mask'] = variant_agg['in_window_idx'].apply(create_mask)
    
    # Create variant_info
    if all(c in variant_agg.columns for c in ['var_type', 'in_window_idx', 'variant_spec']):
        def create_variant_info(row):
            return [
                {
                    'type': row['var_type'][i],
                    'pos': row['in_window_idx'][i] if i < len(row['in_window_idx']) else -1,
                    'spec': row['variant_spec'][i] if i < len(row['variant_spec']) else '',
                }
                for i in range(len(row['var_type']))
            ]
        
        variant_agg['variant_info'] = variant_agg.apply(create_variant_info, axis=1)
    
    # Count by type
    if 'var_type' in variant_agg.columns:
        variant_agg['num_snp'] = variant_agg['var_type'].apply(lambda x: sum(1 for t in x if t == 'SNP'))
        variant_agg['num_ins'] = variant_agg['var_type'].apply(lambda x: sum(1 for t in x if t == 'INS'))
        variant_agg['num_del'] = variant_agg['var_type'].apply(lambda x: sum(1 for t in x if t == 'DEL'))
    else:
        variant_agg['num_snp'] = 0
        variant_agg['num_ins'] = 0
        variant_agg['num_del'] = 0
    
    # Select merge columns
    merge_cols = ['chrom', 'start', 'end', 'has_variant', 'num_variants', 
                  'num_snp', 'num_ins', 'num_del']
    
    if 'variant_mask' in variant_agg.columns:
        merge_cols.append('variant_mask')
    if 'variant_info' in variant_agg.columns:
        merge_cols.append('variant_info')
    if 'seq_alt' in variant_agg.columns:
        merge_cols.append('seq_alt')
    
    # Merge
    logger.info("  Merging with base windows...")
    logger.info(f"    Base windows: {len(windows):,}")
    logger.info(f"    Variant windows: {len(variant_agg):,}")
    
    merged = windows.merge(
        variant_agg[merge_cols],
        on=['chrom', 'start', 'end'],
        how='left',
        validate='one_to_one'  # Ensure 1:1 merge
    )
    
    logger.info(f"    After merge: {len(merged):,}")
    
    # Check for unexpected duplication
    if len(merged) != len(windows):
        logger.error(f"  ❌ DUPLICATION DETECTED: {len(windows):,} → {len(merged):,}")
        logger.error(f"  Keeping only first occurrence of each window")
        merged = merged.drop_duplicates(subset=['chrom', 'start', 'end'], keep='first')
        logger.info(f"    After deduplication: {len(merged):,}")
    
    # Fill NaN values
    merged['has_variant'] = merged['has_variant'].fillna(False)
    merged['num_variants'] = merged['num_variants'].fillna(0).astype(int)
    merged['num_snp'] = merged['num_snp'].fillna(0).astype(int)
    merged['num_ins'] = merged['num_ins'].fillna(0).astype(int)
    merged['num_del'] = merged['num_del'].fillna(0).astype(int)
    
    if 'seq_alt' in merged.columns and 'seq' in merged.columns:
        merged['seq_alt'] = merged['seq_alt'].fillna(merged['seq'])
    
    windows_with_variants = merged['has_variant'].sum()
    logger.info(f"  Windows with variants: {windows_with_variants:,} / {len(merged):,}")
    
    if 'var_type' in variant_agg.columns:
        logger.info(f"  Variant counts: SNP={merged['num_snp'].sum():,}, "
                   f"INS={merged['num_ins'].sum():,}, DEL={merged['num_del'].sum():,}")
    
    return merged


# =============================================================================
# Splice Variant Processing
# =============================================================================

def flatten_splice_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten splice variant lists into top-level columns.
    
    Handles multiple splice variants per window by selecting the most
    significant one (prioritizing STRONG > MILD > NONE, then highest score).
    
    Args:
        df: DataFrame with 'splice_variants' column (list of dicts)
    
    Returns:
        DataFrame with flattened splice variant columns
    """
    logger.info("  Flattening splice variant data...")
    
    df = df.copy()
    
    # Initialize columns with defaults
    splice_cols = {
        'splice_effect': '',
        'splice_score': 0.0,
        'splice_method': '',
        'splice_location': '',
        'splice_distance_to_exon': np.inf,
        'splice_site_type': '',
        'splice_gene': '',
        'splice_hgvs': '',
        'is_splice_altering': False,
        'is_canonical_site': False,
        'is_deep_intronic': False,
        'is_exonic': False,
    }
    
    for col, default in splice_cols.items():
        if col not in df.columns:
            df[col] = default
    
    # Only process windows that have splice variants
    has_splice = df['has_splice_variant'].fillna(False)
    
    if not has_splice.any():
        logger.info("    No splice variants to flatten")
        return df
    
    logger.info(f"    Processing {has_splice.sum():,} windows with splice variants")
    
    # Define effect priority for selecting primary variant
    effect_priority = {'STRONG': 3, 'MILD': 2, 'NONE': 1, '': 0}
    
    # Track successful extractions
    extracted = 0
    
    for idx in df[has_splice].index:
        variants = df.at[idx, 'splice_variants']
        
        # Check for None, empty, or invalid types
        if variants is None:
            continue
        if isinstance(variants, float) and np.isnan(variants):
            continue
        
        # Accept both list and numpy array
        if not isinstance(variants, (list, np.ndarray)):
            logger.warning(f"    Row {idx}: splice_variants is invalid type: {type(variants)}")
            continue
        
        # Convert numpy array to list for easier handling
        if isinstance(variants, np.ndarray):
            variants = variants.tolist()
        
        if len(variants) == 0:
            continue
        
        # Select primary variant (highest priority effect, then highest score)
        try:
            primary = max(
                variants,
                key=lambda v: (
                    effect_priority.get(v.get('splice_effect', ''), 0),
                    v.get('splice_score', 0.0) if isinstance(v.get('splice_score', 0), (int, float)) else 0.0
                )
            )
        except Exception as e:
            logger.warning(f"    Row {idx}: Error selecting primary variant: {e}")
            continue
        
        # Flatten primary variant to top-level columns
        df.at[idx, 'splice_effect'] = primary.get('splice_effect', '')
        df.at[idx, 'splice_score'] = float(primary.get('splice_score', 0.0))
        df.at[idx, 'splice_method'] = primary.get('method', '')
        df.at[idx, 'splice_location'] = primary.get('location', '')
        df.at[idx, 'splice_distance_to_exon'] = float(primary.get('distance_to_exon', np.inf))
        df.at[idx, 'splice_site_type'] = primary.get('site_type', '')
        df.at[idx, 'splice_gene'] = primary.get('gene', '')
        df.at[idx, 'splice_hgvs'] = primary.get('hgvs', '')
        
        # Derive boolean flags
        effect = primary.get('splice_effect', '')
        site_type = primary.get('site_type', '')
        
        df.at[idx, 'is_splice_altering'] = effect in ['STRONG', 'MILD']
        df.at[idx, 'is_canonical_site'] = site_type == 'canonical'
        df.at[idx, 'is_deep_intronic'] = site_type == 'deep_intronic'
        df.at[idx, 'is_exonic'] = site_type == 'exonic'
        
        extracted += 1
    
    logger.info(f"    Successfully extracted {extracted} splice variants")
    
    # Log statistics
    logger.info(f"    Splice variant statistics:")
    logger.info(f"      Total windows with splice variants: {has_splice.sum():,}")
    
    if has_splice.any():
        logger.info(f"      By effect:")
        for effect in ['STRONG', 'MILD', 'NONE']:
            count = (df['splice_effect'] == effect).sum()
            if count > 0:
                logger.info(f"        {effect}: {count:,}")
        
        # Check for empty effects
        empty_effects = (df[has_splice]['splice_effect'] == '').sum()
        if empty_effects > 0:
            logger.warning(f"        Empty/missing: {empty_effects:,}")
        
        logger.info(f"      By site type:")
        for site_type in ['canonical', 'near_splice', 'splice_region', 'deep_intronic', 'exonic']:
            count = (df['splice_site_type'] == site_type).sum()
            if count > 0:
                logger.info(f"        {site_type}: {count:,}")
        
        logger.info(f"      Derived flags:")
        logger.info(f"        is_splice_altering: {df['is_splice_altering'].sum():,}")
        logger.info(f"        is_canonical_site: {df['is_canonical_site'].sum():,}")
        logger.info(f"        is_deep_intronic: {df['is_deep_intronic'].sum():,}")
        logger.info(f"        is_exonic: {df['is_exonic'].sum():,}")
    
    return df


def merge_splice_variants_with_windows(
    windows: pd.DataFrame,
    splice_dir: str
) -> pd.DataFrame:
    """Merge splice variant annotations with windows."""
    logger.info("\n📂 Loading splice variant annotations...")
    
    merged_windows_dir = os.path.join(splice_dir, 'merged_windows')
    
    if not os.path.exists(merged_windows_dir):
        logger.warning(f"  Splice variants directory not found: {merged_windows_dir}")
        windows['has_splice_variant'] = False
        windows['num_splice_variants'] = 0
        windows['max_splice_score'] = 0.0
        windows = flatten_splice_variants(windows)
        return windows
    
    splice_windows = load_parquet_dir(merged_windows_dir)
    logger.info(f"  Loaded {len(splice_windows):,} splice variant records")
    
    # DEBUG: Check structure
    logger.info(f"  Columns: {list(splice_windows.columns)}")
    if len(splice_windows) > 0:
        first_row = splice_windows.iloc[0]
        logger.info(f"  Sample row:")
        for col in splice_windows.columns:
            val = first_row[col]
            if isinstance(val, list):
                logger.info(f"    {col}: list with {len(val)} items")
                if len(val) > 0:
                    logger.info(f"      First item: {val[0]}")
            else:
                logger.info(f"    {col}: {val}")
    
    # Check for duplicates
    splice_dupes = splice_windows.duplicated(subset=['chrom', 'start', 'end'], keep=False)
    if splice_dupes.any():
        logger.warning(f"  Found {splice_dupes.sum():,} duplicate splice variant records")
        logger.warning(f"  Keeping first occurrence of each window")
        splice_windows = splice_windows.drop_duplicates(subset=['chrom', 'start', 'end'], keep='first')
        logger.info(f"  After deduplication: {len(splice_windows):,}")
    
    # Select columns
    splice_cols = ['chrom', 'start', 'end']
    
    if 'has_splice_variant' in splice_windows.columns:
        splice_cols.append('has_splice_variant')
    if 'splice_variants' in splice_windows.columns:
        splice_cols.append('splice_variants')
    if 'num_splice_variants' in splice_windows.columns:
        splice_cols.append('num_splice_variants')
    if 'splice_effects' in splice_windows.columns:
        splice_cols.append('splice_effects')
    if 'max_splice_score' in splice_windows.columns:
        splice_cols.append('max_splice_score')
    
    logger.info(f"  Selecting columns: {splice_cols}")
    splice_windows = splice_windows[splice_cols]
    
    # Merge
    logger.info("  Merging with base windows...")
    logger.info(f"    Base windows: {len(windows):,}")
    logger.info(f"    Splice windows: {len(splice_windows):,}")
    
    merged = windows.merge(
        splice_windows,
        on=['chrom', 'start', 'end'],
        how='left',
        validate='one_to_one',
        suffixes=('', '_splice')
    )
    
    logger.info(f"    After merge: {len(merged):,}")
    
    # Check for unexpected duplication
    if len(merged) != len(windows):
        logger.error(f"  ❌ DUPLICATION DETECTED: {len(windows):,} → {len(merged):,}")
        logger.error(f"  Keeping only first occurrence of each window")
        merged = merged.drop_duplicates(subset=['chrom', 'start', 'end'], keep='first')
        logger.info(f"    After deduplication: {len(merged):,}")
    
    # Fill NaN
    if 'has_splice_variant' in merged.columns:
        merged['has_splice_variant'] = merged['has_splice_variant'].fillna(False)
    if 'num_splice_variants' in merged.columns:
        merged['num_splice_variants'] = merged['num_splice_variants'].fillna(0).astype(int)
    if 'max_splice_score' in merged.columns:
        merged['max_splice_score'] = merged['max_splice_score'].fillna(0.0)
    
    windows_with_splice = merged['has_splice_variant'].sum() if 'has_splice_variant' in merged.columns else 0
    logger.info(f"  Windows with splice variants: {windows_with_splice:,} / {len(merged):,}")
    
    # DEBUG: Check what we're passing to flatten
    if windows_with_splice > 0 and 'splice_variants' in merged.columns:
        sample_idx = merged[merged['has_splice_variant']].index[0]
        sample = merged.at[sample_idx, 'splice_variants']
        logger.info(f"  DEBUG: Sample splice_variants before flatten: {type(sample)}")
        if isinstance(sample, list):
            logger.info(f"  DEBUG: Length: {len(sample)}")
            if len(sample) > 0:
                logger.info(f"  DEBUG: First item: {sample[0]}")
    
    # Flatten
    merged = flatten_splice_variants(merged)
    
    return merged


# =============================================================================
# Training Label Creation
# =============================================================================

def create_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create training labels based on splice variants and other features.
    
    Labeling strategy:
      - Label = 1: Has STRONG splice effect
      - Label = 0: No splice variant OR splice effect = NONE/MILD
      - Weight = 3.0: STRONG splice-altering variant
      - Weight = 2.0: MILD effect (experimentally validated)
      - Weight = 1.5: NONE effect (validated negative)
      - Weight = 1.0: No experimental validation
    
    Args:
        df: DataFrame with splice variant columns
    
    Returns:
        DataFrame with training_label and training_weight columns
    """
    logger.info("\n🏷️  Creating training labels...")
    
    df = df.copy()
    
    # Initialize
    df['training_label'] = 0
    df['training_weight'] = 1.0
    
    # Positive labels: STRONG splice effect
    if 'splice_effect' in df.columns:
        strong_mask = df['splice_effect'] == 'STRONG'
        df.loc[strong_mask, 'training_label'] = 1
        df.loc[strong_mask, 'training_weight'] = 3.0
        
        # Higher weight for MILD validated variants (still negatives for training)
        mild_mask = df['splice_effect'] == 'MILD'
        df.loc[mild_mask, 'training_weight'] = 2.0
        
        # Even NONE effects get some weight (they're validated negatives)
        none_mask = df['splice_effect'] == 'NONE'
        df.loc[none_mask, 'training_weight'] = 1.5
    
    # Log label distribution
    label_counts = df['training_label'].value_counts().sort_index()
    logger.info(f"  Training label distribution:")
    for label, count in label_counts.items():
        pct = 100 * count / len(df) if len(df) > 0 else 0
        logger.info(f"    Label {label}: {count:,} ({pct:.1f}%)")
    
    # Log weight distribution
    logger.info(f"  Training weight distribution:")
    for weight in sorted(df['training_weight'].unique()):
        count = (df['training_weight'] == weight).sum()
        pct = 100 * count / len(df) if len(df) > 0 else 0
        logger.info(f"    Weight {weight}: {count:,} ({pct:.1f}%)")
    
    # Additional statistics
    if 'splice_effect' in df.columns:
        logger.info(f"\n  By splice effect:")
        for effect in ['STRONG', 'MILD', 'NONE', '']:
            count = (df['splice_effect'] == effect).sum()
            if count > 0:
                avg_label = df[df['splice_effect'] == effect]['training_label'].mean()
                avg_weight = df[df['splice_effect'] == effect]['training_weight'].mean()
                effect_name = effect if effect else 'None'
                logger.info(f"    {effect_name}: {count:,} windows, "
                          f"avg_label={avg_label:.2f}, avg_weight={avg_weight:.2f}")
    
    return df


# =============================================================================
# Train/Test Split
# =============================================================================

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
    group_by_chrom: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    logger.info(f"\n✂️  Splitting data into train/test (test_size={test_size})...")
    
    # Check if we have multiple chromosomes
    chroms = sorted(df['chrom'].unique()) if 'chrom' in df.columns else []
    num_chroms = len(chroms)
    
    logger.info(f"  Found {num_chroms} chromosome(s): {chroms}")
    
    # If only 1 chromosome, cannot do chromosome-level split
    if group_by_chrom and num_chroms <= 1:
        logger.warning(f"  ⚠️  Only {num_chroms} chromosome - cannot do chromosome-level split")
        logger.warning(f"  Switching to random split")
        group_by_chrom = False
    
    if group_by_chrom and num_chroms > 1:
        logger.info("  Using chromosome-level split (no data leakage)")
        
        # [previous chromosome split code...]
        
    else:
        logger.info("  Using random split")
        
        if stratify and 'training_label' in df.columns:
            logger.info("    Stratifying by training_label")
            stratify_col = df['training_label']
        else:
            stratify_col = None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_col
        )
    
    # Report statistics
    logger.info(f"\n  Split statistics:")
    logger.info(f"    Train: {len(train_df):,} windows ({len(train_df)/len(df)*100:.1f}%)")
    
    if len(train_df) > 0:
        train_pos = (train_df['training_label']==1).sum()
        logger.info(f"      Positives: {train_pos:,} ({train_pos/len(train_df)*100:.1f}%)")
        logger.info(f"      Negatives: {(train_df['training_label']==0).sum():,} "
                   f"({(train_df['training_label']==0).sum()/len(train_df)*100:.1f}%)")
    
    logger.info(f"    Test: {len(test_df):,} windows ({len(test_df)/len(df)*100:.1f}%)")
    
    if len(test_df) > 0:
        test_pos = (test_df['training_label']==1).sum()
        logger.info(f"      Positives: {test_pos:,} ({test_pos/len(test_df)*100:.1f}%)")
        logger.info(f"      Negatives: {(test_df['training_label']==0).sum():,} "
                   f"({(test_df['training_label']==0).sum()/len(test_df)*100:.1f}%)")
    
    return train_df, test_df


# =============================================================================
# Validation
# =============================================================================

def validate_training_data(df: pd.DataFrame, name: str = "data"):
    """Validate the final training data."""
    logger.info(f"  Validating {name}...")
    
    errors = []
    warnings = []
    
    # Required columns
    required = ['chrom', 'start', 'end', 'training_label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check coordinate validity
    if 'start' in df.columns and 'end' in df.columns:
        invalid_coords = df[df['start'] >= df['end']]
        if len(invalid_coords) > 0:
            errors.append(f"{len(invalid_coords)} windows have start >= end")
    
    # Check for duplicates
    if all(c in df.columns for c in ['chrom', 'start', 'end']):
        dupes = df.duplicated(subset=['chrom', 'start', 'end'], keep=False)
        if dupes.any():
            warnings.append(f"{dupes.sum()} duplicate windows found")
    
    # Validate splice variant columns if present
    if 'has_splice_variant' in df.columns:
        # Check consistency
        has_splice = df['has_splice_variant']
        has_effect = df['splice_effect'].notna() & (df['splice_effect'] != '')
        
        inconsistent = (has_splice & ~has_effect).sum()
        if inconsistent > 0:
            warnings.append(
                f"{inconsistent} windows marked has_splice_variant=True "
                f"but have no splice_effect"
            )
        
        # Check derived flags match
        if 'is_splice_altering' in df.columns:
            splice_altering_expected = df['splice_effect'].isin(['STRONG', 'MILD'])
            mismatch = (df['is_splice_altering'] != splice_altering_expected).sum()
            if mismatch > 0:
                warnings.append(
                    f"{mismatch} windows have inconsistent is_splice_altering flag"
                )
    
    # Check training labels
    if 'training_label' in df.columns:
        invalid_labels = df[~df['training_label'].isin([0, 1])]
        if len(invalid_labels) > 0:
            errors.append(f"{len(invalid_labels)} windows have invalid training labels")
    
    # Report results
    if errors:
        for err in errors:
            logger.error(f"    ❌ {err}")
        raise ValueError(f"Validation failed for {name}")
    
    if warnings:
        for warn in warnings:
            logger.warning(f"    ⚠️  {warn}")
    else:
        logger.info(f"    ✅ All validation checks passed")


# =============================================================================
# Output Saving
# =============================================================================

def save_training_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    split_by: str = 'chrom',
    format: str = 'parquet',
    compression: str = 'snappy',
    write_index: bool = False,
    write_stats: bool = True
):
    """Save final training data."""
    logger.info(f"\n💾 Saving training data to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def save_split(df, split_dir, split_name):
        logger.info(f"\n  Saving {split_name} set ({len(df):,} windows)...")
        
        if split_by == 'chrom':
            for chrom in sorted(df['chrom'].unique()):
                chrom_df = df[df['chrom'] == chrom]
                outfile = os.path.join(split_dir, f"{chrom}.{format}")
                
                if format == 'parquet':
                    chrom_df.to_parquet(outfile, compression=compression, index=False)
                elif format == 'hdf5':
                    chrom_df.to_hdf(outfile, key='data', mode='w', complevel=9)
                
                logger.info(f"    {chrom}: {len(chrom_df):,} windows → {outfile}")
        else:
            outfile = os.path.join(split_dir, f"data.{format}")
            if format == 'parquet':
                df.to_parquet(outfile, compression=compression, index=False)
            elif format == 'hdf5':
                df.to_hdf(outfile, key='data', mode='w', complevel=9)
            logger.info(f"    Saved {len(df):,} windows → {outfile}")
    
    save_split(train_df, train_dir, 'train')
    save_split(test_df, test_dir, 'test')
    
    # Write statistics
    if write_stats:
        logger.info(f"\n  Writing statistics...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_windows': len(train_df) + len(test_df),
            'train_windows': len(train_df),
            'test_windows': len(test_df),
            'test_fraction': len(test_df) / (len(train_df) + len(test_df)),
            
            'train_stats': {
                'positives': int((train_df['training_label'] == 1).sum()),
                'negatives': int((train_df['training_label'] == 0).sum()),
                'positive_rate': float((train_df['training_label'] == 1).mean()),
                'chromosomes': sorted(train_df['chrom'].unique().tolist()) if 'chrom' in train_df.columns else [],
            },
            
            'test_stats': {
                'positives': int((test_df['training_label'] == 1).sum()),
                'negatives': int((test_df['training_label'] == 0).sum()),
                'positive_rate': float((test_df['training_label'] == 1).mean()),
                'chromosomes': sorted(test_df['chrom'].unique().tolist()) if 'chrom' in test_df.columns else [],
            },
            
            'columns': list(train_df.columns),
        }
        
        # Add splice variant stats if available
        if 'has_splice_variant' in train_df.columns:
            stats['splice_variant_stats'] = {
                'train_with_splice': int(train_df['has_splice_variant'].sum()),
                'test_with_splice': int(test_df['has_splice_variant'].sum()),
            }
        
        # Add variant stats if available
        if 'has_variant' in train_df.columns:
            stats['variant_stats'] = {
                'train_with_variants': int(train_df['has_variant'].sum()),
                'test_with_variants': int(test_df['has_variant'].sum()),
                'train_total_variants': int(train_df['num_variants'].sum()),
                'test_total_variants': int(test_df['num_variants'].sum()),
            }
        
        stats_file = os.path.join(output_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"    Saved statistics → {stats_file}")
    
    # Write index if requested
    if write_index:
        logger.info(f"\n  Writing index files...")
        
        def write_index_file(df, index_file):
            index_data = df[['chrom', 'start', 'end']].copy()
            index_data.to_parquet(index_file, compression='snappy', index=False)
        
        write_index_file(train_df, os.path.join(train_dir, 'index.parquet'))
        write_index_file(test_df, os.path.join(test_dir, 'index.parquet'))
        
        logger.info(f"    Saved index files")


# =============================================================================
# Main Processing
# =============================================================================

def main():
    args = parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*80)
    logger.info("BetaDogma Training Data Aggregation")
    logger.info("="*80)
    logger.info(f"Base windows:     {args.input_dir}")
    logger.info(f"GTEx PSI:         {args.gtex_dir}")
    logger.info(f"Variants:         {args.variant_dir}")
    logger.info(f"Splice variants:  {args.splice_dir or 'None'}")
    logger.info(f"Output:           {args.output_dir}")
    logger.info(f"Test size:        {args.test_size}")
    logger.info(f"Random seed:      {args.random_seed}")
    logger.info("="*80)
    
    # Step 1: Load base structural windows
    logger.info("\n📂 Step 1: Loading base structural windows...")
    base_windows = load_parquet_dir(args.input_dir)
    logger.info(f"  Loaded {len(base_windows):,} windows")
    logger.info(f"  Columns: {list(base_windows.columns)}")
    
    # Step 2: Merge GTEx data
    merged = merge_gtex_by_coordinates(base_windows, args.gtex_dir)
    del base_windows
    gc.collect()
    
    # Step 3: Merge population variants
    merged = merge_variants_with_windows(merged, args.variant_dir)
    gc.collect()
    
    # Step 4: Merge splice variants (optional)
    if args.splice_dir:
        merged = merge_splice_variants_with_windows(merged, args.splice_dir)
    else:
        logger.info("\n⚠️  No splice variant directory provided")
        logger.info("    Continuing without splice variant data")
        
        # Add empty splice columns for consistency
        merged['has_splice_variant'] = False
        merged['num_splice_variants'] = 0
        merged = flatten_splice_variants(merged)
    
    gc.collect()
    
    # Step 5: Create training labels
    merged = create_training_labels(merged)
    
    # Step 6: Apply filters
    if args.max_variants_per_window > 0:
        before = len(merged)
        merged = merged[merged['num_variants'] <= args.max_variants_per_window]
        logger.info(f"\n  Filtered by max variants: {before:,} → {len(merged):,}")
    
    if args.min_coverage > 0:
        if 'coverage' in merged.columns:
            before = len(merged)
            merged = merged[merged['coverage'] >= args.min_coverage]
            logger.info(f"  Filtered by coverage: {before:,} → {len(merged):,}")
    
    if args.min_psi > 0:
        if 'psi' in merged.columns:
            before = len(merged)
            merged = merged[merged['psi'] >= args.min_psi]
            logger.info(f"  Filtered by PSI: {before:,} → {len(merged):,}")
    
    # Step 7: Select columns
    if args.keep_columns:
        available_cols = [c for c in args.keep_columns if c in merged.columns]
        missing_cols = [c for c in args.keep_columns if c not in merged.columns]
        
        if missing_cols:
            logger.warning(f"\n  Requested columns not found: {missing_cols}")
        
        merged = merged[available_cols]
        logger.info(f"\n  Selected {len(available_cols)} columns")
    
    if args.drop_columns:
        merged = merged.drop(columns=args.drop_columns, errors='ignore')
    
    # Step 8: Split into train/test
    train_df, test_df = split_train_test(
        merged,
        test_size=args.test_size,
        random_seed=args.random_seed,
        stratify=args.stratify,
        group_by_chrom=args.group_by_chrom
    )
    
    del merged
    gc.collect()
    
    # Step 9: Validate
    if args.validate:
        logger.info("\n✓ Validating output...")
        validate_training_data(train_df, name="train set")
        validate_training_data(test_df, name="test set")
    
    # Step 10: Save final training data
    save_training_data(
        train_df,
        test_df,
        args.output_dir,
        split_by=args.split_by,
        format=args.format,
        compression=args.compression,
        write_index=args.write_index,
        write_stats=args.write_stats
    )
    
    logger.info("\n" + "="*80)
    logger.info("✅ Aggregation complete!")
    logger.info("="*80)
    logger.info(f"Train set: {len(train_df):,} windows")
    logger.info(f"Test set:  {len(test_df):,} windows")
    logger.info(f"Output:    {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()