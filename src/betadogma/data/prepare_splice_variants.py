#!/usr/bin/env python3
"""
Prepare experimentally validated splice variants from SpliceVarDB.

This script:
1. Reads SpliceVarDB VCF (converted from TSV)
2. Parses experimental splice effects (STRONG/MILD/NONE)
3. Annotates location relative to exons (NO filtering - trust experimental data)
4. Merges with base genomic windows
5. Outputs splice variant annotations

Key Feature: Includes deep intronic variants (>50bp from exons)
             that are experimentally proven to affect splicing.
             NO proximity-based filtering!

Usage:
    python prepare_splicevar.py \\
        --splicevar-vcf data/raw/variants/splicevar_hg38.vcf.gz \\
        --gtf data/raw/gencode/gencode.v44.annotation.gtf \\
        --base-windows data/cache/chr21/variants_base/*.parquet \\
        --out data/cache/chr21/splice_variants \\
        --chroms chr21
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# VCF Parsing
# =============================================================================

def parse_info_field(info_str: str) -> Dict[str, str]:
    """Parse VCF INFO field into dictionary."""
    info = {}
    for item in info_str.split(';'):
        if '=' in item:
            key, value = item.split('=', 1)
            info[key] = value
        else:
            info[item] = True
    return info


def load_splice_variants_from_vcf(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    include_effects: Optional[List[str]] = None,
    min_qual: float = 0.0
) -> pd.DataFrame:
    """
    Load experimentally validated splice variants from SpliceVarDB VCF.
    
    Args:
        vcf_path: Path to SpliceVarDB VCF (bgzipped with index)
        chromosomes: List of chromosomes to include (e.g., ['chr21'])
        include_effects: List of effects to include: ['STRONG', 'MILD', 'NONE']
        min_qual: Minimum quality score (default 0 = no filter)
    
    Returns:
        DataFrame with columns:
            - chrom, pos, ref, alt
            - splice_effect (STRONG/MILD/NONE)
            - splice_score (1.0/0.5/0.0)
            - gene, hgvs, method, location, doi
    """
    logger.info(f"Loading splice variants from: {vcf_path}")
    
    if include_effects is None:
        include_effects = ['STRONG', 'MILD', 'NONE']
    
    variants = []
    total_variants = 0
    filtered_counts = defaultdict(int)
    
    try:
        vcf = pysam.VariantFile(vcf_path)
        
        # Get chromosomes to process
        vcf_chroms = list(vcf.header.contigs)
        if chromosomes:
            target_chroms = [c for c in chromosomes if c in vcf_chroms]
            if not target_chroms:
                logger.warning(f"No matching chromosomes found in VCF")
                logger.warning(f"  Requested: {chromosomes}")
                logger.warning(f"  Available: {vcf_chroms[:10]}...")
        else:
            target_chroms = vcf_chroms
        
        logger.info(f"Processing {len(target_chroms)} chromosome(s): {target_chroms}")
        
        # Process each chromosome
        for chrom in target_chroms:
            logger.info(f"  Reading {chrom}...")
            
            try:
                for record in vcf.fetch(chrom):
                    total_variants += 1
                    
                    # Quality filter
                    if record.qual is not None and record.qual < min_qual:
                        filtered_counts['low_quality'] += 1
                        continue
                    
                    # Parse INFO field
                    info = record.info
                    
                    # Get splice effect
                    splice_effect = info.get('SPLICE_EFFECT', 'UNKNOWN')
                    
                    # Filter by effect
                    if splice_effect not in include_effects:
                        filtered_counts[f'excluded_effect_{splice_effect}'] += 1
                        continue
                    
                    # Get other fields - ensure all are strings, not tuples
                    splice_score = float(info.get('SPLICE_SCORE', 0.0))

                    # Convert to string, handling tuples from multi-value INFO fields
                    def info_to_str(value, default=''):
                        """Convert VCF INFO value to string."""
                        if value is None:
                            return default
                        if isinstance(value, (list, tuple)):
                            return ';'.join(str(v) for v in value)
                        return str(value)

                    gene = info_to_str(info.get('GENE'), '')
                    hgvs = info_to_str(info.get('HGVS'), '')
                    method = info_to_str(info.get('METHOD'), '')
                    location = info_to_str(info.get('LOCATION'), '')
                    classification = info_to_str(info.get('CLASSIFICATION'), '')
                    doi = info_to_str(info.get('DOI'), '')

                    
                    # Store variant
                    variants.append({
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'ref': record.ref,
                        'alt': record.alts[0] if record.alts else '.',
                        'qual': record.qual if record.qual is not None else 0.0,
                        'splice_effect': splice_effect,
                        'splice_score': splice_score,
                        'gene': gene,
                        'hgvs': hgvs,
                        'method': method,
                        'location': location,  # From original TSV
                        'classification': classification,  # Original classification
                        'doi': doi
                    })
                    
            except Exception as e:
                logger.warning(f"Error fetching {chrom}: {e}")
                continue
        
        vcf.close()
        
    except Exception as e:
        logger.error(f"Error reading VCF: {e}")
        raise
    
    # Convert to DataFrame
    df = pd.DataFrame(variants)
    
    # Log statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"Splice Variant Loading Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total variants in VCF:     {total_variants:,}")
    logger.info(f"Variants loaded:           {len(df):,}")
    logger.info(f"Variants filtered:         {sum(filtered_counts.values()):,}")
    
    if filtered_counts:
        logger.info(f"\nFiltering breakdown:")
        for reason, count in sorted(filtered_counts.items()):
            logger.info(f"  {reason:30s}: {count:6,}")
    
    if len(df) > 0:
        logger.info(f"\nBy splice effect:")
        for effect in ['STRONG', 'MILD', 'NONE']:
            count = (df['splice_effect'] == effect).sum()
            pct = 100 * count / len(df)
            logger.info(f"  {effect:10s}: {count:6,} ({pct:5.1f}%)")
        
        logger.info(f"\nBy experimental method:")
        method_counts = df['method'].value_counts()
        for method, count in method_counts.head(10).items():
            pct = 100 * count / len(df)
            logger.info(f"  {method:20s}: {count:6,} ({pct:5.1f}%)")
    
    return df


# =============================================================================
# Exon Annotation
# =============================================================================

def load_exon_boundaries(gtf_path: str) -> pd.DataFrame:
    """
    Load exon boundaries from GTF for location annotation.
    
    Returns DataFrame with columns: chrom, start, end, strand, gene_name, transcript_id
    """
    logger.info(f"Loading exon boundaries from: {gtf_path}")
    
    exons = []
    
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            feature_type = fields[2]
            if feature_type != 'exon':
                continue
            
            chrom = fields[0]
            start = int(fields[3]) - 1  # Convert to 0-based
            end = int(fields[4])
            strand = fields[6]
            
            # Parse attributes
            attrs = {}
            for item in fields[8].split(';'):
                item = item.strip()
                if ' ' in item:
                    key, value = item.split(' ', 1)
                    attrs[key] = value.strip('"')
            
            exons.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'gene_name': attrs.get('gene_name', ''),
                'transcript_id': attrs.get('transcript_id', '')
            })
    
    df = pd.DataFrame(exons)
    logger.info(f"  Loaded {len(df):,} exons")
    
    return df


def annotate_variant_locations(
    variants_df: pd.DataFrame,
    exons_df: pd.DataFrame,
    location_categories: Dict[str, Tuple[Optional[int], Optional[int]]] = None
) -> pd.DataFrame:
    """
    Annotate variant locations relative to exons.
    
    This does NOT filter variants - it only adds location metadata for analysis.
    ALL variants are kept regardless of distance from exons!
    
    Args:
        variants_df: Variants with chrom, pos
        exons_df: Exons with chrom, start, end
        location_categories: Distance categories (in bp from exon boundary)
            Example: {
                'canonical': (-2, 2),      # ±2bp from splice site
                'near_splice': (3, 8),     # 3-8bp from splice site
                'splice_region': (9, 50),  # 9-50bp
                'deep_intronic': (51, None) # >50bp
            }
    
    Returns:
        DataFrame with added columns:
            - distance_to_exon: Minimum distance to any exon boundary
            - site_type: canonical/near_splice/splice_region/deep_intronic/exonic
            - is_canonical: Boolean
            - is_deep_intronic: Boolean
            - is_exonic: Boolean
            - nearest_exon_gene: Gene of nearest exon
    """
    logger.info("Annotating variant locations relative to exons...")
    
    if location_categories is None:
        location_categories = {
            'canonical': (-2, 2),
            'near_splice': (3, 8),
            'splice_region': (9, 50),
            'deep_intronic': (51, None)
        }
    
    # Make a copy to avoid modifying original
    df = variants_df.copy()
    
    # Initialize columns
    df['distance_to_exon'] = np.inf
    df['site_type'] = 'unknown'
    df['is_canonical'] = False
    df['is_deep_intronic'] = False
    df['is_exonic'] = False
    df['nearest_exon_gene'] = ''
    
    # Process each chromosome
    for chrom in df['chrom'].unique():
        logger.info(f"  Processing {chrom}...")
        
        chrom_variants = df['chrom'] == chrom
        chrom_exons = exons_df[exons_df['chrom'] == chrom]
        
        if len(chrom_exons) == 0:
            logger.warning(f"  No exons found for {chrom}")
            continue
        
        # For each variant, find distance to nearest exon
        for idx in df[chrom_variants].index:
            pos = df.loc[idx, 'pos']
            
            # Check if within any exon
            in_exon = ((chrom_exons['start'] <= pos) & (pos < chrom_exons['end']))
            
            if in_exon.any():
                df.loc[idx, 'distance_to_exon'] = 0
                df.loc[idx, 'site_type'] = 'exonic'
                df.loc[idx, 'is_exonic'] = True
                df.loc[idx, 'nearest_exon_gene'] = chrom_exons[in_exon].iloc[0]['gene_name']
            else:
                # Calculate distance to nearest exon boundary
                distances_to_start = np.abs(pos - chrom_exons['start'].values)
                distances_to_end = np.abs(pos - chrom_exons['end'].values)
                min_dist = min(distances_to_start.min(), distances_to_end.min())
                
                df.loc[idx, 'distance_to_exon'] = min_dist
                
                # Find nearest exon gene
                nearest_idx = np.argmin(np.minimum(distances_to_start, distances_to_end))
                df.loc[idx, 'nearest_exon_gene'] = chrom_exons.iloc[nearest_idx]['gene_name']
                
                # Categorize by distance
                categorized = False
                for cat_name, (min_d, max_d) in location_categories.items():
                    if min_d is None:
                        min_d = -np.inf
                    if max_d is None:
                        max_d = np.inf
                    
                    if min_d <= min_dist <= max_d:
                        df.loc[idx, 'site_type'] = cat_name
                        categorized = True
                        break
                
                if not categorized:
                    df.loc[idx, 'site_type'] = 'other'
    
    # Set boolean flags
    df['is_canonical'] = df['site_type'] == 'canonical'
    df['is_deep_intronic'] = df['site_type'] == 'deep_intronic'
    
    # Log statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"Location Annotation Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total variants: {len(df):,}")
    
    logger.info(f"\nBy site type:")
    for site_type in ['exonic', 'canonical', 'near_splice', 'splice_region', 'deep_intronic', 'other', 'unknown']:
        count = (df['site_type'] == site_type).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {site_type:20s}: {count:6,} ({pct:5.1f}%)")
    
    logger.info(f"\nDeep intronic variants (>50bp from exons):")
    deep = df[df['is_deep_intronic']]
    logger.info(f"  Count: {len(deep):,}")
    if len(deep) > 0:
        logger.info(f"  By effect:")
        for effect in ['STRONG', 'MILD', 'NONE']:
            count = (deep['splice_effect'] == effect).sum()
            logger.info(f"    {effect}: {count}")
    
    return df


# =============================================================================
# Window Merging
# =============================================================================

def load_base_windows(base_windows_input: Union[str, List[str]]) -> pd.DataFrame:
    """Load base windows from previous step (variants_base)."""
    from glob import glob
    
    logger.info(f"Loading base windows from: {base_windows_input}")
    
    # Handle both glob pattern (single string) and expanded files (list)
    if isinstance(base_windows_input, list):
        if len(base_windows_input) == 1:
            # Single pattern - try glob expansion
            parquet_files = glob(base_windows_input[0])
            if not parquet_files:
                # Maybe it's just one file
                parquet_files = base_windows_input
        else:
            # Multiple files already provided
            parquet_files = base_windows_input
    else:
        # Single string pattern
        parquet_files = glob(base_windows_input)
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found: {base_windows_input}")
    
    logger.info(f"  Found {len(parquet_files)} parquet files")
    
    dfs = []
    for f in tqdm(parquet_files, desc="Loading base windows"):
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Loaded {len(combined):,} windows")
    
    return combined


def assign_variants_to_windows(
    windows_df: pd.DataFrame,
    variants_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign splice variants to genomic windows.
    
    Returns windows with added columns:
        - has_splice_variant: bool
        - splice_variants: list of variant info dicts
        - num_splice_variants: int
        - splice_effects: list of effects in window
        - max_splice_score: max score in window
    """
    logger.info("Assigning splice variants to windows...")
    
    # Initialize columns
    windows_df = windows_df.copy()
    windows_df['has_splice_variant'] = False
    windows_df['splice_variants'] = [[] for _ in range(len(windows_df))]
    windows_df['num_splice_variants'] = 0
    windows_df['splice_effects'] = [[] for _ in range(len(windows_df))]
    windows_df['max_splice_score'] = 0.0
    
    # Process each chromosome
    for chrom in windows_df['chrom'].unique():
        logger.info(f"  Processing {chrom}...")
        
        chrom_windows = windows_df[windows_df['chrom'] == chrom]
        chrom_variants = variants_df[variants_df['chrom'] == chrom]
        
        if len(chrom_variants) == 0:
            logger.info(f"    No variants for {chrom}")
            continue
        
        logger.info(f"    {len(chrom_windows):,} windows, {len(chrom_variants):,} variants")
        
        # For each window, find overlapping variants
        for idx, window in chrom_windows.iterrows():
            start = window['start']
            end = window['end']
            
            # Find variants in window
            in_window = (
                (chrom_variants['pos'] >= start) &
                (chrom_variants['pos'] < end)
            )
            
            window_variants = chrom_variants[in_window]
            
            if len(window_variants) > 0:
                windows_df.at[idx, 'has_splice_variant'] = True
                windows_df.at[idx, 'num_splice_variants'] = len(window_variants)
                
                # Store variant info
                variant_list = []
                effects = []
                scores = []
                
                for _, var in window_variants.iterrows():
                    variant_list.append({
                        'pos': int(var['pos']),
                        'ref': var['ref'],
                        'alt': var['alt'],
                        'splice_effect': var['splice_effect'],
                        'splice_score': float(var['splice_score']),
                        'gene': var['gene'],
                        'hgvs': var['hgvs'],
                        'method': var['method'],
                        'site_type': var.get('site_type', ''),
                        'distance_to_exon': float(var.get('distance_to_exon', 0))
                    })
                    effects.append(var['splice_effect'])
                    scores.append(var['splice_score'])
                
                windows_df.at[idx, 'splice_variants'] = variant_list
                windows_df.at[idx, 'splice_effects'] = effects
                windows_df.at[idx, 'max_splice_score'] = max(scores)
    
    # Summary
    num_with_variants = windows_df['has_splice_variant'].sum()
    logger.info(f"\n  Windows with splice variants: {num_with_variants:,} / {len(windows_df):,}")
    
    return windows_df


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare experimentally validated splice variants from SpliceVarDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_splicevar.py \\
      --splicevar-vcf data/raw/variants/splicevar_hg38.vcf.gz \\
      --gtf data/raw/gencode/gencode.v44.annotation.gtf \\
      --base-windows "data/cache/chr21/variants_base/*.parquet" \\
      --out data/cache/chr21/splice_variants \\
      --chroms chr21
  
  # Include only high-confidence variants
  python prepare_splicevar.py \\
      --splicevar-vcf data/raw/variants/splicevar_hg38.vcf.gz \\
      --gtf data/raw/gencode/gencode.v44.annotation.gtf \\
      --base-windows "data/cache/chr21/variants_base/*.parquet" \\
      --out data/cache/chr21/splice_variants \\
      --chroms chr21 \\
      --include-effects STRONG
        """
    )
    
    # Input files
    parser.add_argument('--splicevar-vcf', required=True,
                        help='SpliceVarDB VCF file (bgzipped with index)')
    parser.add_argument('--gtf', required=True,
                        help='GENCODE GTF file for exon boundaries')
    parser.add_argument('--base-windows', required=True, nargs='+',
                        help='Base windows from previous step (glob pattern or file paths)')
    # Output
    parser.add_argument('--out', required=True,
                        help='Output directory')
    
    # Filtering
    parser.add_argument('--chroms', nargs='+',
                        help='Chromosomes to process (e.g., chr21)')
    parser.add_argument('--include-effects', action='append',
                        default=[], choices=['STRONG', 'MILD', 'NONE'],
                        help='Splice effects to include (default: all)')
    parser.add_argument('--min-qual', type=float, default=0.0,
                        help='Minimum variant quality (default: 0.0)')
    
    # Processing
    parser.add_argument('--shard-size', type=int, default=1000,
                        help='Windows per output shard')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (not yet implemented)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = out_dir / 'merged_windows'
    merged_dir.mkdir(exist_ok=True)
    
    logger.info(f"{'='*80}")
    logger.info(f"SpliceVarDB Preparation")
    logger.info(f"{'='*80}")
    logger.info(f"SpliceVar VCF: {args.splicevar_vcf}")
    logger.info(f"GTF:           {args.gtf}")
    logger.info(f"Base windows:  {args.base_windows}")
    logger.info(f"Output:        {args.out}")
    logger.info(f"Chromosomes:   {args.chroms}")
    logger.info(f"Include effects: {args.include_effects}")
    logger.info(f"{'='*80}\n")
    
    # Step 1: Load splice variants from VCF
    variants_df = load_splice_variants_from_vcf(
        args.splicevar_vcf,
        chromosomes=args.chroms,
        include_effects=args.include_effects,
        min_qual=args.min_qual
    )
    
    if len(variants_df) == 0:
        logger.warning("No variants passed filters!")
        logger.warning("Creating empty output files...")
        
        # Create empty outputs
        variants_df.to_parquet(out_dir / 'splice_variants.parquet', index=False)
        
        metadata = {
            'total_variants': 0,
            'chromosomes': args.chroms,
            'include_effects': args.include_effects,
            'timestamp': datetime.now().isoformat()
        }
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return
    
    # Step 2: Load exon boundaries
    exons_df = load_exon_boundaries(args.gtf)
    
    # Step 3: Annotate variant locations (NO filtering!)
    variants_df = annotate_variant_locations(variants_df, exons_df)
    
    # Save annotated variants
    variants_path = out_dir / 'splice_variants.parquet'
    variants_df.to_parquet(variants_path, index=False)
    logger.info(f"\n✅ Saved annotated variants: {variants_path}")
    
    # Step 4: Load base windows
    windows_df = load_base_windows(args.base_windows)
    
    # Step 5: Assign variants to windows
    windows_df = assign_variants_to_windows(windows_df, variants_df)
    
    # Step 6: Save merged windows
    logger.info(f"\nSaving merged windows to: {merged_dir}")
    
    num_windows = len(windows_df)
    num_shards = (num_windows + args.shard_size - 1) // args.shard_size
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * args.shard_size
        end_idx = min((shard_idx + 1) * args.shard_size, num_windows)
        
        shard_df = windows_df.iloc[start_idx:end_idx]
        shard_path = merged_dir / f'shard_{shard_idx:04d}.parquet'
        shard_df.to_parquet(shard_path, index=False)
    
    logger.info(f"  Saved {num_shards} shards")
    
    # Step 7: Generate statistics
    stats = {
        'total_variants': len(variants_df),
        'chromosomes': args.chroms,
        'include_effects': args.include_effects,
        
        'by_effect': {
            'STRONG': int((variants_df['splice_effect'] == 'STRONG').sum()),
            'MILD': int((variants_df['splice_effect'] == 'MILD').sum()),
            'NONE': int((variants_df['splice_effect'] == 'NONE').sum())
        },
        
        'by_site_type': {
            site_type: int((variants_df['site_type'] == site_type).sum())
            for site_type in variants_df['site_type'].unique()
        },
        
        'deep_intronic': {
            'total': int(variants_df['is_deep_intronic'].sum()),
            'STRONG': int(((variants_df['is_deep_intronic']) & 
                          (variants_df['splice_effect'] == 'STRONG')).sum()),
            'MILD': int(((variants_df['is_deep_intronic']) & 
                        (variants_df['splice_effect'] == 'MILD')).sum()),
            'NONE': int(((variants_df['is_deep_intronic']) & 
                        (variants_df['splice_effect'] == 'NONE')).sum())
        },
        
        'windows': {
            'total': int(num_windows),
            'with_splice_variants': int(windows_df['has_splice_variant'].sum()),
            'shards': num_shards
        },
        
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✅ Saved metadata: {metadata_path}")
    
    # Save detailed location stats
    location_stats_path = out_dir / 'location_stats.json'
    location_stats = {
        'by_effect_and_location': {}
    }
    
    for effect in ['STRONG', 'MILD', 'NONE']:
        location_stats['by_effect_and_location'][effect] = {}
        effect_vars = variants_df[variants_df['splice_effect'] == effect]
        
        for site_type in variants_df['site_type'].unique():
            count = (effect_vars['site_type'] == site_type).sum()
            location_stats['by_effect_and_location'][effect][site_type] = int(count)
    
    with open(location_stats_path, 'w') as f:
        json.dump(location_stats, f, indent=2)
    logger.info(f"✅ Saved location stats: {location_stats_path}")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Variants processed:     {len(variants_df):,}")
    logger.info(f"  STRONG:               {stats['by_effect']['STRONG']:,}")
    logger.info(f"  MILD:                 {stats['by_effect']['MILD']:,}")
    logger.info(f"  NONE:                 {stats['by_effect']['NONE']:,}")
    logger.info(f"Deep intronic:          {stats['deep_intronic']['total']:,}")
    logger.info(f"Windows with variants:  {stats['windows']['with_splice_variants']:,} / {num_windows:,}")
    logger.info(f"\nOutputs:")
    logger.info(f"  {variants_path}")
    logger.info(f"  {merged_dir}/ ({num_shards} shards)")
    logger.info(f"  {metadata_path}")
    logger.info(f"  {location_stats_path}")


if __name__ == '__main__':
    main()
