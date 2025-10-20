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
    
    # Log detailed GTEx statistics
    windows_with_gtex = merged['has_gtex_data'].sum()
    total_junctions = merged['num_junctions'].sum()
    
    logger.info("\n🌐 GTEx Splicing Data:")
    logger.info("=" * 50)
    logger.info(f"  Windows with GTEx data: {windows_with_gtex:,}/{len(merged):,} "
               f"({windows_with_gtex/len(merged)*100:.1f}%)")
    logger.info(f"  Total junctions mapped: {total_junctions:,}")
    
    if windows_with_gtex > 0:
        gtex_windows = merged[merged['has_gtex_data']]
        avg_psi = gtex_windows['psi_mean'].mean()
        median_psi = gtex_windows['psi_mean'].median()
        
        logger.info("\n  PSI Statistics (windows with data):")
        logger.info(f"    - Mean: {avg_psi:.3f}")
        logger.info(f"    - Median: {median_psi:.3f}")
        logger.info(f"    - Min: {gtex_windows['psi_mean'].min():.3f}")
        logger.info(f"    - Max: {gtex_windows['psi_mean'].max():.3f}")
        
        # Junction count distribution
        junction_counts = gtex_windows['num_junctions']
        logger.info("\n  Junction Counts per Window:")
        logger.info(f"    - Mean: {junction_counts.mean():.1f}")
        logger.info(f"    - Median: {junction_counts.median():.1f}")
        logger.info(f"    - Min: {junction_counts.min():,}")
        logger.info(f"    - Max: {junction_counts.max():,}")
    
    logger.info("=" * 50)
    
    return merged


# =============================================================================
# Variant Merging
# =============================================================================

def merge_variants_with_windows(
    windows: pd.DataFrame,
    variant_dir: str
) -> pd.DataFrame:
    """
    Merge population variant data with windows.
    
    Handles multiple variant types (1000 Genomes, ClinVar, etc.) and aggregates
    them into a single variant representation per window.
    
    Args:
        windows: DataFrame with genomic windows
        variant_dir: Directory containing variant parquet files
        
    Returns:
        DataFrame with merged variant information
    """
    logger.info("\n📂 Loading population variants...")
    
    # Define merge columns that will be used in the final merge
    merge_cols = [
        'chrom', 'start', 'end', 'strand', 
        'has_variant', 'num_variants',
        'variant_info', 'variant_mask', 
        'variant_type',  # Track SNP/INS/DEL
        'variant_af',    # Track allele frequency
        'is_pathogenic', # Track pathogenicity
        'SNP', 'INS', 'DEL'  # Legacy type indicators
    ]
    
    if not os.path.exists(variant_dir):
        logger.warning(f"  Variant directory not found: {variant_dir}")
        windows['has_variant'] = False
        windows['num_variants'] = 0
        windows['variant_mask'] = None
        windows['variant_info'] = None
        return windows
    
    # Load and process variants
    variants = load_parquet_dir(variant_dir)
    logger.info(f"  Loaded {len(variants):,} variant records")
    
    # Add source information if not present
    if 'source' not in variants.columns:
        variants['source'] = '1kg'  # Default to 1000 Genomes if source not specified
    
    # Ensure variant type is properly set
    if 'variant_type' not in variants.columns:
        # Infer from existing type indicators if available
        if all(t in variants.columns for t in ['SNP', 'INS', 'DEL']):
            variants['variant_type'] = np.select(
                [variants['SNP'], variants['INS'], variants['DEL']],
                ['SNP', 'INS', 'DEL'],
                default='UNKNOWN'
            )
        else:
            variants['variant_type'] = 'UNKNOWN'
    
    # Ensure allele frequency is set
    if 'variant_af' not in variants.columns and 'af' in variants.columns:
        variants['variant_af'] = variants['af']
    elif 'variant_af' not in variants.columns:
        variants['variant_af'] = 0.0  # Default to 0 if not available
    
    # Check for required columns and handle missing ones
    required_columns = ['chrom', 'pos', 'ref', 'alt']
    missing_columns = [col for col in required_columns if col not in variants.columns]
    
    if missing_columns:
        logger.warning(f"  Missing required columns in variant data: {missing_columns}")
        logger.info(f"  Available columns: {variants.columns.tolist()}")
        
        # If we don't have the required columns for deduplication, just continue with all variants
        logger.info("  Cannot perform deduplication without required columns. Using all variants as-is.")
        variant_dupes = pd.Series(False, index=variants.index)
    else:
        # Check for duplicates in variant data using available columns
        variant_dupes = variants.duplicated(subset=required_columns, keep=False)
        if variant_dupes.any():
            logger.info(f"  Found {variant_dupes.sum():,} duplicate variant records")
            logger.info(f"  Unique variants: {variants[required_columns].drop_duplicates().shape[0]:,}")
        else:
            variant_dupes = pd.Series(False, index=variants.index)
    
    # Build aggregation dictionary - only keep essential fields
    agg_dict = {}
    
    # Add standard fields if they exist
    for col in ['ref', 'alt', 'af', 'variant_type', 'variant_af']:
        if col in variants.columns:
            agg_dict[col] = 'first'
    
    # Handle pathogenicity
    if 'is_pathogenic' in variants.columns:
        agg_dict['is_pathogenic'] = 'any'
    
    # Add list-type aggregations
    for col in ['source', 'var_type', 'in_window_idx', 'variant_spec']:
        if col in variants.columns:
            agg_dict[col] = lambda x: list(x) if len(x) > 1 else x.iloc[0]
    
    # Add boolean aggregations
    for col in ['is_pathogenic', 'is_benign']:
        if col in variants.columns:
            agg_dict[col] = 'any'
    
    # Add ClinVar-specific fields
    for col in ['clinvar_significance', 'clinvar_review_status', 'clinvar_allele_freq']:
        if col in variants.columns:
            agg_dict[col] = 'first'
    
    # Group by genomic position to merge variants at the same location
    logger.info("    # Convert variants to genomic positions for interval mapping")
    if 'pos' in variants.columns:
        variant_positions = variants['pos'].values
    elif 'start' in variants.columns:
        variant_positions = variants['start'].values
    else:
        logger.warning("  No position information available for variants. Cannot map to windows.")
        windows['has_variant'] = False
        windows['num_variants'] = 0
        windows['variant_mask'] = None
        windows['variant_info'] = None
        return windows
    
    # Determine grouping columns based on available data
    if 'pos' in variants.columns:
        group_cols = ['chrom', 'pos']
    elif 'start' in variants.columns:
        group_cols = ['chrom', 'start']
    else:
        logger.warning("  No position column found. Using all variants as-is.")
        group_cols = ['chrom']  # Fallback to just chromosome if no position column
    
    # If we have a valid grouping, perform the aggregation
    if len(group_cols) > 1:
        variant_agg = variants.groupby(group_cols, as_index=False).agg(agg_dict)
    else:
        # If we can't group by position, just take the first variant per chromosome
        variant_agg = variants.drop_duplicates(subset=group_cols, keep='first')
    
    # Add window-based information
    logger.info("  Mapping variants to windows...")
    variant_windows = []
    
    # Convert windows to interval index for faster lookup
    windows['interval'] = pd.IntervalIndex.from_arrays(
        windows['start'], windows['end'], closed='both'
    )
    
    # Create interval tree for faster window lookups
    from intervaltree import IntervalTree
    chrom_trees = {}
    has_strand = 'strand' in windows.columns
    
    for chrom, group in windows.groupby('chrom'):
        tree = IntervalTree()
        for idx, row in group.iterrows():
            # Use strand if available, otherwise default to '+'
            strand = row['strand'] if has_strand else '+'
            # Store window info as (start, end, strand, index) to keep track of the original row
            tree[row['start']:row['end']+1] = (row['start'], row['end'], strand, idx)
        chrom_trees[chrom] = tree
    
    # Map each variant to its containing windows
    for _, var in variant_agg.iterrows():
        chrom = var['chrom']
        
        # Use 'pos' if available, otherwise use 'start' as position
        if 'pos' in var:
            pos = var['pos']
        elif 'start' in var:
            pos = var['start']
        else:
            logger.warning(f"Variant missing position information: {var}")
            continue
            
        if chrom not in chrom_trees:
            continue
            
        # Find all windows containing this variant
        for interval in chrom_trees[chrom][pos]:
            window_start, window_end, strand, _ = interval.data
            
            # Calculate position within window (1-based)
            in_window_pos = pos - window_start + 1
            
            # Create variant info with available data
            variant_info = {
                'chrom': chrom,
                'start': window_start,
                'end': window_end,
                'strand': strand,
                'variant_chrom': chrom,
                'variant_pos': pos,
                'in_window_pos': in_window_pos,
                'source': var.get('source', 'unknown'),
                'type': var.get('var_type', 'UNK'),
                'has_splice_variant': var.get('has_splice_variant', False),
                'num_splice_variants': var.get('num_splice_variants', 0),
                'max_splice_score': var.get('max_splice_score', 0.0)
            }
            
            # Add sequence information if available
            if 'seq' in var and 'seq_alt' in var:
                variant_info.update({
                    'ref': var.get('seq', 'N'),
                    'alt': var.get('seq_alt', 'N'),
                    'is_snp': var.get('ch_snp', False),
                    'is_ins': var.get('ch_ins', False),
                    'is_del': var.get('ch_del', False)
                })
            
            # Add any additional fields that might be useful
            for field in ['af', 'is_pathogenic', 'clinvar_significance', 'splice_effect',
                                'is_snp', 'is_ins', 'is_del']:
                if field in var:
                    variant_info[field] = var[field]
            
            # Add optional fields
            for field in ['af', 'is_pathogenic', 'clinvar_significance', 'splice_effect']:
                if field in var:
                    variant_info[field] = var[field]
            
            variant_windows.append(variant_info)
    
    # Convert to DataFrame and aggregate by window
    if variant_windows:
        try:
            variant_windows_df = pd.DataFrame(variant_windows)
            
            # Ensure we have the required columns for grouping
            required_cols = ['chrom', 'start', 'end', 'strand']
            for col in required_cols:
                if col not in variant_windows_df.columns:
                    logger.warning(f"Missing required column: {col}")
                    variant_windows_df[col] = ''
            
            # Add in_window_pos if missing
            if 'in_window_pos' not in variant_windows_df.columns:
                if 'pos' in variant_windows_df.columns and 'start' in variant_windows_df.columns:
                    variant_windows_df['in_window_pos'] = variant_windows_df['pos'] - variant_windows_df['start'] + 1
                else:
                    variant_windows_df['in_window_pos'] = 0
            
            # Group by window coordinates
            grouped = variant_windows_df.groupby(
                ['chrom', 'start', 'end', 'strand'], 
                group_keys=False
            )
            
            # Initialize list to store window variant data
            window_data = []
            
            # Process each window group
            for (chrom, start, end, strand), group in grouped:
                # Initialize variant info list for this window
                variant_info = []
                
                # Process each variant in the window
                for _, row in group.iterrows():
                    # Create variant info dictionary with all position information
                    variant = {
                        # Original genomic coordinates
                        'chrom': str(row.get('chrom', '')),  # Add chromosome
                        'pos': int(row.get('pos', row.get('start', 0))),  # Genomic position
                        'ref': str(row.get('ref', row.get('seq', 'N'))),  # Reference allele
                        'alt': str(row.get('alt', row.get('seq_alt', 'N'))),  # Alternate allele
                        
                        # Window-relative position
                        'in_window_pos': int(row.get('in_window_pos', 0)),
                        'window_start': int(row.get('start', 0)),
                        'window_end': int(row.get('end', 0)),
                        'strand': str(row.get('strand', '+')),  # Default to '+' if missing
                        
                        # Variant metadata
                        'source': str(row.get('source', 'unknown')),
                        'type': str(row.get('var_type', 'UNK')),
                        'has_splice_variant': bool(row.get('has_splice_variant', False)),
                        'num_splice_variants': int(row.get('num_splice_variants', 0)),
                        'max_splice_score': float(row.get('max_splice_score', 0.0))
                    }
                    
                    # Add any additional fields that might be useful
                    for field in ['af', 'is_pathogenic', 'clinvar_significance', 'splice_effect',
                                'is_snp', 'is_ins', 'is_del']:
                        if field in row:
                            variant[field] = row[field]
                    
                    # Add optional fields
                    for field in ['af', 'is_pathogenic', 'clinvar_significance', 'splice_effect']:
                        if field in row:
                            variant[field] = row[field]
                    
                    variant_info.append(variant)
                
                # Count variant types
                snp_count = 0
                ins_count = 0
                del_count = 0
                
                for v in variant_info:
                    vt = str(v.get('type', '')).upper()
                    if 'INS' in vt:
                        ins_count += 1
                    elif 'DEL' in vt:
                        del_count += 1
                    else:
                        snp_count += 1
                
                # Add window data
                window_data.append({
                    'chrom': str(chrom),
                    'start': int(start),
                    'end': int(end),
                    'strand': str(strand),
                    'has_variant': len(variant_info) > 0,
                    'num_variants': len(variant_info),
                    'variant_info': variant_info,
                    'variant_mask': [v.get('pos', 0) for v in variant_info],
                    'SNP': snp_count,
                    'INS': ins_count,
                    'DEL': del_count
                })
            
            # Convert to DataFrame
            if window_data:
                window_variants = pd.DataFrame(window_data)
            else:
                window_variants = pd.DataFrame(columns=[
                    'chrom', 'start', 'end', 'strand', 'has_variant', 'num_variants',
                    'variant_info', 'variant_mask', 'SNP', 'INS', 'DEL'
                ])
                
        except Exception as e:
            logger.error(f"Error processing variants: {str(e)}")
            logger.error(traceback.format_exc())
            window_variants = pd.DataFrame(columns=[
                'chrom', 'start', 'end', 'strand', 'has_variant', 'num_variants',
                'variant_info', 'variant_mask', 'SNP', 'INS', 'DEL'
            ])
    else:
        # No variants found
        window_variants = pd.DataFrame(columns=['chrom', 'start', 'end', 'strand'])
        window_variants['has_variant'] = False
        window_variants['num_variants'] = 0
        window_variants['variant_info'] = None
        window_variants['variant_mask'] = None
        window_variants[['SNP', 'INS', 'DEL']] = 0
        
        # Ensure all required columns exist
        for col in merge_cols:
            if col not in window_variants.columns:
                window_variants[col] = None
        
        # Select only the required columns
        window_variants = window_variants[merge_cols]
        
        # Add pathogenic variant counts if available in variant_info
        if not window_variants.empty and 'variant_info' in window_variants.columns:
            def count_pathogenic(variants):
                if not isinstance(variants, list):
                    return 0
                return sum(1 for v in variants if isinstance(v, dict) and v.get('is_pathogenic', False))
            
            window_variants['num_pathogenic'] = window_variants['variant_info'].apply(count_pathogenic)
            merge_cols.append('num_pathogenic')
    
    # Merge
    logger.info("  Merging with base windows...")
    logger.info(f"    Base windows: {len(windows):,}")
    logger.info(f"    Variant windows: {len(window_variants):,}")
    
    # Ensure we only keep the columns that exist in window_variants
    valid_merge_cols = [col for col in merge_cols if col in window_variants.columns]
    
    merged = windows.merge(
        window_variants[valid_merge_cols],
        on=['chrom', 'start', 'end'],
        how='left',
        validate='one_to_one'  # Ensure 1:1 merge
    )
    
    # Fill NA values for variant columns
    merged['has_variant'] = merged['has_variant'].fillna(False)
    merged['num_variants'] = merged['num_variants'].fillna(0).astype(int)
    
    # Fill in 0 for variant type counts
    for col in ['SNP', 'INS', 'DEL']:
        merged[col] = merged[col].fillna(0).astype(int)
    
    # Log detailed variant statistics
    num_windows_with_variants = merged['has_variant'].sum()
    total_variants = merged['num_variants'].sum()
    
    logger.info("\n📊 Variant Statistics:")
    logger.info("=" * 50)
    logger.info(f"  Windows with variants: {num_windows_with_variants:,}/{len(merged):,} "
               f"({num_windows_with_variants/len(merged)*100:.1f}%)")
    logger.info(f"  Total variants: {total_variants:,}")
    
    # Variant type breakdown
    logger.info("\n  Variant Types:")
    logger.info(f"    - SNPs: {merged['SNP'].sum():,}")
    logger.info(f"    - Insertions: {merged['INS'].sum():,}")
    logger.info(f"    - Deletions: {merged['DEL'].sum():,}")
    
    # Pathogenic variants
    if 'num_pathogenic' in merged.columns:
        patho_count = merged['num_pathogenic'].sum()
        patho_windows = (merged['num_pathogenic'] > 0).sum()
        logger.info(f"\n  Pathogenic Variants:")
        logger.info(f"    - Total: {patho_count:,}")
        logger.info(f"    - Windows with pathogenic variants: {patho_windows:,} "
                  f"({patho_windows/len(merged)*100:.1f}%)")
    
    # Variant sources (if available)
    if 'variant_info' in merged.columns:
        try:
            sources = {}
            for variants in merged['variant_info'].dropna():
                for var in variants:
                    source = var.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            
            if sources:
                logger.info("\n  Variant Sources:")
                for source, count in sorted(sources.items()):
                    logger.info(f"    - {source}: {count:,}")
        except Exception as e:
            logger.debug(f"Could not extract variant sources: {str(e)}")
    
    logger.info("=" * 50)
    
    # Clean up
    del windows, variant_windows_df, window_variants
    gc.collect()
    
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
    
    # Initialize columns with defaults and track their presence
    splice_cols = {
        'splice_effect': ('', 'categorical'),
        'splice_score': (0.0, 'float'),
        'splice_method': ('', 'categorical'),
        'splice_location': ('', 'categorical'),
        'splice_distance_to_exon': (np.inf, 'float'),
        'splice_site_type': ('', 'categorical'),
        'splice_gene': ('', 'categorical'),
        'splice_hgvs': ('', 'categorical'),
        'splice_variant_type': ('UNKNOWN', 'categorical'),
        'splice_variant_af': (0.0, 'float'),
        'is_splice_altering': (False, 'bool'),
        'is_canonical_site': (False, 'bool'),
        'is_deep_intronic': (False, 'bool'),
        'is_exonic': (False, 'bool'),
        'is_pathogenic': (False, 'bool'),
    }
    
    # Initialize missing columns with appropriate defaults
    for col, (default, dtype) in splice_cols.items():
        if col not in df.columns:
            df[col] = default
            # Ensure correct data type
            if dtype == 'float':
                df[col] = df[col].astype(float)
            elif dtype == 'bool':
                df[col] = df[col].astype(bool)
            elif dtype == 'categorical':
                df[col] = df[col].astype('category')
    
    # Only process windows that have splice variants
    has_splice = df['has_splice_variant'].fillna(False)
    
    if not has_splice.any():
        logger.info("    No splice variants to flatten (no windows with has_splice_variant=True)")
        return df
    
    logger.info(f"    Processing {has_splice.sum():,} windows with splice variants")
    
    # Debug: Check if we have the splice_variants column
    if 'splice_variants' not in df.columns:
        logger.warning("    No 'splice_variants' column found in the input data")
        return df
        
    # Debug: Check the type of the splice_variants column
    logger.info(f"    Type of splice_variants column: {df['splice_variants'].dtype}")
    
    # Debug: Count non-null values in splice_variants
    non_null = df[has_splice]['splice_variants'].notna().sum()
    logger.info(f"    Non-null splice_variants: {non_null:,} out of {has_splice.sum():,} windows with has_splice_variant=True")
    
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
        
        # Flatten primary variant to top-level columns with validation
        try:
            # Basic variant info
            df.at[idx, 'splice_effect'] = str(primary.get('splice_effect', ''))
            df.at[idx, 'splice_score'] = float(primary.get('splice_score', 0.0))
            df.at[idx, 'splice_method'] = str(primary.get('method', ''))
            df.at[idx, 'splice_location'] = str(primary.get('location', ''))
            df.at[idx, 'splice_distance_to_exon'] = float(primary.get('distance_to_exon', np.inf))
            df.at[idx, 'splice_site_type'] = str(primary.get('site_type', ''))
            df.at[idx, 'splice_gene'] = str(primary.get('gene', ''))
            df.at[idx, 'splice_hgvs'] = str(primary.get('hgvs', ''))
            
            # Additional variant info if available
            if 'variant_type' in primary:
                df.at[idx, 'splice_variant_type'] = str(primary['variant_type'])
            if 'af' in primary:
                df.at[idx, 'splice_variant_af'] = float(primary['af'])
            
            # Derive boolean flags
            effect = str(primary.get('splice_effect', '')).upper()
            site_type = str(primary.get('site_type', '')).lower()
            
            df.at[idx, 'is_splice_altering'] = effect in ['STRONG', 'MILD']
            df.at[idx, 'is_canonical_site'] = site_type == 'canonical'
            df.at[idx, 'is_deep_intronic'] = 'deep' in site_type
            df.at[idx, 'is_exonic'] = 'exon' in site_type
            
            # Pathogenicity if available
            if 'is_pathogenic' in primary:
                df.at[idx, 'is_pathogenic'] = bool(primary['is_pathogenic'])
            
            extracted += 1
            
        except Exception as e:
            logger.warning(f"    Row {idx}: Error processing variant: {e}")
            logger.debug(f"    Variant data: {primary}")
            continue
        
        extracted += 1
    
    logger.info(f"    Successfully extracted {extracted} splice variants")
    
    # Log comprehensive statistics
    logger.info("\n🔍 Splice Variant Statistics")
    logger.info("=" * 80)
    logger.info(f"  Total windows processed: {len(df):,}")
    logger.info(f"  Windows with splice variants: {has_splice.sum():,} ({has_splice.mean()*100:.1f}%)")
    
    if has_splice.any():
        # Variant type distribution
        if 'splice_variant_type' in df.columns:
            type_counts = df[has_splice]['splice_variant_type'].value_counts()
            logger.info("\n  Variant Type Distribution:")
            for var_type, count in type_counts.items():
                logger.info(f"    - {var_type}: {count:,} ({count/has_splice.sum()*100:.1f}%)")
        
        # Allele frequency distribution
        if 'splice_variant_af' in df.columns:
            af_stats = df[has_splice]['splice_variant_af'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
            logger.info("\n  Allele Frequency Distribution:")
            for stat in ['mean', 'min', '10%', '50%', '90%', 'max']:
                if stat in af_stats:
                    logger.info(f"    - {stat}: {af_stats[stat]:.6f}")
        
        # Pathogenicity
        if 'is_pathogenic' in df.columns:
            path_count = df[has_splice]['is_pathogenic'].sum()
            logger.info(f"\n  Pathogenic Variants: {path_count:,} ({path_count/has_splice.sum()*100:.1f}% of splice variants)")
    
    logger.info("=" * 80)
    
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
    if 'splice_effect' in splice_windows.columns:
        splice_cols.append('splice_effect')
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
    else:
        logger.warning("  No splice_variants column found in merged data")
    
    # Flatten
    merged = flatten_splice_variants(merged)
    
    # Debug: Check if any splice effects were actually set
    if 'splice_effect' in merged.columns:
        effect_counts = merged[merged['has_splice_variant']]['splice_effect'].value_counts()
        logger.info("\n  Splice effects after flattening:")
        for effect, count in effect_counts.items():
            logger.info(f"    {effect}: {count}")
    
    # Log detailed splice variant statistics
    windows_with_splice = merged['has_splice_variant'].sum()
    total_splice_vars = merged['num_splice_variants'].sum()
    
    logger.info("\n🧬 Splice Variant Statistics:")
    logger.info("=" * 50)
    logger.info(f"  Windows with splice variants: {windows_with_splice:,}/{len(merged):,} "
               f"({windows_with_splice/len(merged)*100:.1f}%)")
    logger.info(f"  Total splice variants: {total_splice_vars:,}")
    
    if 'splice_effect' in merged.columns:
        # Detailed effect distribution
        effects = merged[merged['has_splice_variant']]['splice_effect'].value_counts()
        if not effects.empty:
            logger.info("\n  Splice Effect Distribution:")
            for effect, count in effects.items():
                logger.info(f"    - {effect}: {count:,} windows "
                          f"({count/windows_with_splice*100:.1f}%)")
    
    # Splice score statistics if available
    if 'max_splice_score' in merged.columns:
        splice_scores = merged[merged['has_splice_variant']]['max_splice_score']
        if not splice_scores.empty:
            logger.info("\n  SpliceAI Score Statistics (max per window):")
            logger.info(f"    - Mean: {splice_scores.mean():.3f}")
            logger.info(f"    - Median: {splice_scores.median():.3f}")
            logger.info(f"    - Min: {splice_scores.min():.3f}")
            logger.info(f"    - Max: {splice_scores.max():.3f}")
            
            # Score distribution in bins
            bins = [0, 0.2, 0.5, 0.8, 1.0]
            score_bins = pd.cut(splice_scores, bins=bins, right=False)
            bin_counts = score_bins.value_counts().sort_index()
            
            logger.info("\n  SpliceAI Score Distribution:")
            for score_range, count in bin_counts.items():
                logger.info(f"    - {score_range}: {count:,} windows "
                          f"({count/len(splice_scores)*100:.1f}%)")
    
    logger.info("=" * 50)
    
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
    
    # Log comprehensive data summary
    logger.info("\n📊 Training Data Summary:")
    logger.info("=" * 50)
    
    # Basic dataset stats
    logger.info(f"  Total windows: {len(df):,}")
    logger.info(f"  Training set: {len(train_df):,} windows ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test set:     {len(test_df):,} windows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Label distribution
    train_pos = train_df['training_label'].sum()
    test_pos = test_df['training_label'].sum()
    
    logger.info("\n🏷️  Label Distribution:")
    logger.info("  Training set:")
    logger.info(f"    - Positive: {train_pos:,} ({train_pos/len(train_df)*100:.1f}%)")
    logger.info(f"    - Negative: {len(train_df)-train_pos:,} ({(len(train_df)-train_pos)/len(train_df)*100:.1f}%)")
    logger.info("  Test set:")
    logger.info(f"    - Positive: {test_pos:,} ({test_pos/len(test_df)*100:.1f}%)")
    logger.info(f"    - Negative: {len(test_df)-test_pos:,} ({(len(test_df)-test_pos)/len(test_df)*100:.1f}%)")
    
    # Feature statistics
    logger.info("\n🔍 Feature Statistics:")
    
    # GTEx coverage (if available)
    if 'has_gtex_data' in train_df.columns:
        train_gtex = train_df['has_gtex_data'].sum()
        test_gtex = test_df['has_gtex_data'].sum()
        logger.info("  GTEx Coverage:")
        logger.info(f"    - Training: {train_gtex:,} windows with data ({train_gtex/len(train_df)*100:.1f}%)")
        logger.info(f"    - Test:     {test_gtex:,} windows with data ({test_gtex/len(test_df)*100:.1f}%)")
    else:
        logger.info("  GTEx Data: No GTEx data available in this dataset")
    
    # Variant statistics (if available)
    if 'has_variant' in train_df.columns:
        train_vars = train_df['has_variant'].sum()
        test_vars = test_df['has_variant'].sum()
        logger.info("  Variant Coverage:")
        logger.info(f"    - Training: {train_vars:,} windows with variants ({train_vars/len(train_df)*100:.1f}%)")
        logger.info(f"    - Test:     {test_vars:,} windows with variants ({test_vars/len(test_df)*100:.1f}%)")
    else:
        logger.info("  Variant Data: No variant data available in this dataset")
    
    # Splice variant statistics (if available)
    if 'has_splice_variant' in train_df.columns:
        train_splice = train_df['has_splice_variant'].sum()
        test_splice = test_df['has_splice_variant'].sum()
        logger.info("  Splice Variants:")
        logger.info(f"    - Training: {train_splice:,} windows with splice variants ({train_splice/len(train_df)*100:.1f}%)")
        logger.info(f"    - Test:     {test_splice:,} windows with splice variants ({test_splice/len(test_df)*100:.1f}%)")
    else:
        logger.info("  Splice Variants: No splice variant data available in this dataset")
    
    # Class weights
    logger.info("\n⚖️  Class Weights:")
    if 'training_weight' in train_df.columns:
        weight_stats = train_df.groupby('training_label')['training_weight'].agg(['mean', 'std', 'min', 'max'])
        for label, stats in weight_stats.iterrows():
            logger.info(f"  Label {int(label)}: mean={stats['mean']:.2f} ± {stats['std']:.2f} "
                      f"(range: {stats['min']:.2f}-{stats['max']:.2f})")
    
    logger.info("=" * 50)
    
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