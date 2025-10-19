#!/usr/bin/env python3
"""
Prepare pathogenic variants from ClinVar for splice disruption training.

This script:
1. Reads ClinVar VCF with splice-related annotations
2. Filters for pathogenic/likely pathogenic variants with splice significance
3. Merges with base genomic windows
4. Outputs pathogenic variant annotations

Usage:
    python prepare_pathogenic_variants.py \\
        --clinvar-vcf data/raw/variants/clinvar_20251013.vcf.gz \\
        --windows data/cache/chr21/gencode_windows_base/*.parquet \\
        --out data/cache/chr21/pathogenic_variants \\
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


def parse_clinvar_info(info_str: str) -> Dict[str, str]:
    """Parse VCF INFO field into dictionary."""
    info = {}
    for item in info_str.split(';'):
        if '=' in item:
            key, value = item.split('=', 1)
            if value.startswith('(') and value.endswith(')'):
                info[key] = tuple(value[1:-1].split(','))
            else:
                info[key] = value
        else:
            info[item] = True
    return info


def load_pathogenic_variants_from_clinvar(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    clinvar_filter: str = "pathogenic",
    significance: Optional[List[str]] = None,
    review_status: Optional[List[str]] = None,
    min_qual: float = 0.0
) -> pd.DataFrame:
    """
    Load pathogenic variants from ClinVar VCF.

    Args:
        vcf_path: Path to ClinVar VCF (bgzipped with index)
        chromosomes: List of chromosomes to include (e.g., ['chr21'])
        clinvar_filter: Filter level ('pathogenic', 'likely_pathogenic', 'vus', etc.)
        significance: List of clinical significances to include
        review_status: List of review statuses to include
        min_qual: Minimum quality score (default 0 = no filter)

    Returns:
        DataFrame with ClinVar pathogenic variants
    """
    logger.info(f"Loading ClinVar variants from: {vcf_path}")

    if significance is None:
        significance = ["Pathogenic", "Likely_pathogenic"]

    variants = []
    total_variants = 0
    filtered_counts = defaultdict(int)

    try:
        vcf = pysam.VariantFile(vcf_path)

        # Get chromosomes to process
        vcf_chroms = list(vcf.header.contigs)
        if chromosomes:
            target_chroms = [c for c in chromosomes if c in vcf_chroms]
        else:
            target_chroms = vcf_chroms

        logger.info(f"Processing chromosomes: {target_chroms}")

        for chrom in target_chroms:
            logger.info(f"Processing {chrom}...")
            try:
                for record in vcf.fetch(chrom):
                    total_variants += 1

                    # Apply quality filter
                    if min_qual > 0 and record.qual < min_qual:
                        filtered_counts['quality'] += 1
                        continue

                    # Parse INFO field
                    info = parse_clinvar_info(str(record.info.get('CLNSIG', '')))

                    # Check clinical significance
                    clnsig = info.get('CLNSIG', '').split('|')[0] if info.get('CLNSIG') else ''
                    if clnsig not in significance:
                        filtered_counts['significance'] += 1
                        continue

                    # Check review status
                    if review_status:
                        revstat = str(info.get('CLNREVSTAT', '')).split('|')[0] if info.get('CLNREVSTAT') else ''
                        if revstat not in review_status:
                            filtered_counts['review_status'] += 1
                            continue

                    # Check for splice-related annotations
                    # Look for splice-related terms in HGVS or other fields
                    hgvs = str(info.get('HGVS', ''))
                    is_splice_related = any(term in hgvs.lower() for term in [
                        'splice', 'splicing', 'intron', 'exon', 'acceptor', 'donor'
                    ])

                    # Also check disease names or phenotypes for splice terms
                    phenotype = str(info.get('CLNDN', ''))
                    is_splice_related = is_splice_related or any(term in phenotype.lower() for term in [
                        'splice', 'splicing', 'intron'
                    ])

                    if not is_splice_related:
                        filtered_counts['not_splice'] += 1
                        continue

                    # Extract variant information
                    variant_data = {
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'ref': record.ref,
                        'alt': ','.join(record.alts) if record.alts else '',
                        'qual': record.qual,
                        'clnsig': clnsig,
                        'clnrevstat': revstat,
                        'hgvs': hgvs,
                        'gene': str(info.get('GENEINFO', '')),
                        'phenotype': phenotype,
                        'is_splice_related': True
                    }

                    variants.append(variant_data)

            except ValueError as e:
                logger.warning(f"Error fetching {chrom}: {e}")

        vcf.close()

    except Exception as e:
        logger.error(f"Error loading ClinVar VCF: {e}")
        raise

    # Create DataFrame
    df = pd.DataFrame(variants)

    logger.info(f"Loaded {len(df)} pathogenic splice variants")
    logger.info(f"Total variants processed: {total_variants}")
    logger.info(f"Filtered: {dict(filtered_counts)}")

    return df


def annotate_variant_location(variants_df: pd.DataFrame, gtf_path: str) -> pd.DataFrame:
    """
    Annotate variant locations relative to exons.

    Args:
        variants_df: DataFrame with variant data
        gtf_path: Path to GTF annotation file

    Returns:
        DataFrame with location annotations added
    """
    logger.info(f"Annotating variant locations using GTF: {gtf_path}")

    # This is a simplified location annotation
    # In a full implementation, you'd use a library like pybedtools or similar
    # For now, we'll add basic location info based on HGVS

    def classify_location(hgvs):
        if not hgvs:
            return 'unknown'

        hgvs_lower = hgvs.lower()
        if 'intron' in hgvs_lower or 'ivs' in hgvs_lower:
            if any(term in hgvs_lower for term in ['splice', 'acceptor', 'donor']):
                return 'canonical'
            elif any(term in hgvs_lower for term in ['near', 'flank']):
                return 'near_splice'
            else:
                return 'deep_intronic'
        elif 'exon' in hgvs_lower:
            return 'exonic'
        else:
            return 'unknown'

    variants_df['location'] = variants_df['hgvs'].apply(classify_location)

    # Calculate distance to nearest exon (simplified)
    # This would require more sophisticated exon coordinate mapping
    variants_df['distance_to_exon'] = 0  # Placeholder

    return variants_df


def merge_with_windows(variants_df: pd.DataFrame, windows_glob: str) -> pd.DataFrame:
    """
    Merge pathogenic variants with genomic windows.

    Args:
        variants_df: DataFrame with variant data
        windows_glob: Glob pattern for window Parquet files

    Returns:
        DataFrame with variants merged into windows
    """
    logger.info(f"Merging variants with windows: {windows_glob}")

    # Read windows
    windows_files = sorted(Path('.').glob(windows_glob.replace('*.parquet', '*')))
    if not windows_files:
        raise FileNotFoundError(f"No window files found: {windows_glob}")

    windows_df = pd.concat([pd.read_parquet(f) for f in windows_files[:5]], ignore_index=True)  # Sample for structure

    # This is a simplified merge - in practice you'd need more sophisticated spatial joining
    # For now, we'll create a basic structure

    merged_data = []

    for _, window in windows_df.iterrows():
        chrom = window['chrom']
        start = window['start']
        end = window['end']

        # Find variants in this window
        window_variants = variants_df[
            (variants_df['chrom'] == chrom) &
            (variants_df['pos'] >= start) &
            (variants_df['pos'] < end)
        ]

        if not window_variants.empty:
            for _, variant in window_variants.iterrows():
                merged_data.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'seq': window['seq'],
                    'has_pathogenic_variant': True,
                    'pathogenic_clnsig': variant['clnsig'],
                    'pathogenic_gene': variant['gene'],
                    'pathogenic_location': variant['location'],
                    'variant_spec': f"{variant['chrom']}:{variant['pos']}{variant['ref']}>{variant['alt']}",
                    'num_pathogenic_variants': 1
                })

    result_df = pd.DataFrame(merged_data)
    logger.info(f"Merged {len(result_df)} windows with pathogenic variants")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Prepare pathogenic variants from ClinVar")
    parser.add_argument("--clinvar-vcf", required=True, help="Path to ClinVar VCF file")
    parser.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    parser.add_argument("--windows", required=True, help="Glob pattern for base window files")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--chroms", nargs='+', help="Chromosomes to process")
    parser.add_argument("--clinvar-filter", default="pathogenic", help="ClinVar filter level")
    parser.add_argument("--significance", nargs='+', default=["Pathogenic", "Likely_pathogenic"],
                       help="Clinical significances to include")
    parser.add_argument("--review-status", nargs='+', default=["criteria_provided", "reviewed_by_expert_panel"],
                       help="Review statuses to include")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    Path(args.out).mkdir(parents=True, exist_ok=True)

    try:
        # Load pathogenic variants
        variants_df = load_pathogenic_variants_from_clinvar(
            args.clinvar_vcf,
            chromosomes=args.chroms,
            clinvar_filter=args.clinvar_filter,
            significance=args.significance,
            review_status=args.review_status
        )

        # Annotate locations
        variants_df = annotate_variant_location(variants_df, args.gtf)

        # Merge with windows
        merged_df = merge_with_windows(variants_df, args.windows)

        # Write outputs
        output_file = Path(args.out) / "pathogenic_variants.parquet"
        merged_df.to_parquet(output_file, index=False)

        # Write metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'source': args.clinvar_vcf,
            'chromosomes': args.chroms,
            'num_variants': len(variants_df),
            'num_windows': len(merged_df)
        }

        with open(Path(args.out) / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Pathogenic variants processing complete. Output: {output_file}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
