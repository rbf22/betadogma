#!/usr/bin/env python3
"""
Prepare pathogenic variants from ClinVar for splice effect prediction.

This script processes pathogenic variants from ClinVar, validates them
against reference sequences, and prepares them for model training.

Key Features:
- Processes ClinVar pathogenic variants
- Validates reference sequences
- Handles both single and multi-nucleotide variants
- Maintains consistent output format with other variant types
- Preserves all variant metadata
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from betadogma.data.variant_loader import VariantLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathogenicVariantLoader(VariantLoader):
    """Loader for ClinVar pathogenic variants."""
    
    def __init__(self, 
                clinical_significance: Optional[List[str]] = None,
                review_status: Optional[List[str]] = None,
                **kwargs):
        """Initialize the pathogenic variant loader.
        
        Args:
            clinical_significance: List of clinical significance terms to include
            review_status: List of review statuses to include
            **kwargs: Additional arguments for VariantLoader
        """
        super().__init__(**kwargs)
        self.clinical_significance = clinical_significance or [
            'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'
        ]
        self.review_status = review_status or [
            'criteria_provided', 'reviewed_by_expert_panel', 'practice_guideline'
        ]
        # Add pathogenic-specific columns to variant columns
        self._variant_columns.extend([
            'clinical_significance', 'review_status', 'gene', 'phenotype',
            'variant_type', 'variant_length', 'clinvar_id', 'origin'
        ])
    
    def _process_variant_record(self, record) -> Optional[Dict[str, Any]]:
        """Process a single variant record from a VCF."""
        try:
            # Skip based on quality (handled by base class)
            if self.min_qual > 0 and record.qual is not None and record.qual < self.min_qual:
                return None
            
            # Parse INFO field
            info = {k: v[0] if isinstance(v, tuple) and len(v) == 1 else v 
                   for k, v in dict(record.info).items()}
            
            # Get clinical significance
            clnsig = info.get('CLNSIG', '')
            if isinstance(clnsig, (list, tuple)):
                clnsig = clnsig[0] if clnsig else ''
            
            # Filter by clinical significance
            if clnsig not in self.clinical_significance:
                return None
            
            # Get review status
            clnrevstat = info.get('CLNREVSTAT', '')
            if isinstance(clnrevstat, (list, tuple)):
                clnrevstat = clnrevstat[0] if clnrevstat else ''
            
            # Filter by review status if specified
            if self.review_status and not any(rs in clnrevstat for rs in self.review_status):
                return None
            
            # Get variant details
            ref = str(record.ref).upper()
            alts = [str(alt).upper() for alt in record.alts] if record.alts else ['.']
            
            variants = []
            
            # Create entry for each alternate allele
            for alt in alts:
                # Skip non-SNV variants if needed
                if len(ref) > 1 or len(alt) > 1:
                    logger.debug(f"Skipping non-SNV variant: {record.chrom}:{record.pos}{ref}>{alt}")
                    continue
                
                # Get allele frequency
                af = self._get_allele_frequency(info)
                
                # Get gene and phenotype
                gene = info.get('GENEINFO', '').split('|')[0] if 'GENEINFO' in info else ''
                phenotype = info.get('CLNDN', '')
                if isinstance(phenotype, (list, tuple)):
                    phenotype = phenotype[0] if phenotype else ''
                
                variant = {
                    'chrom': str(record.chrom),
                    'pos': int(record.pos),
                    'ref': ref,
                    'alt': alt,
                    'af': af,
                    'source': 'ClinVar',
                    'is_pathogenic': True,
                    'clinical_significance': clnsig,
                    'review_status': clnrevstat,
                    'gene': gene,
                    'phenotype': phenotype,
                    'variant_type': self._get_variant_type(ref, alt),
                    'variant_length': max(len(ref), len(alt)),
                    'clinvar_id': record.id if record.id != '.' else '',
                    'origin': info.get('ORIGIN', [''])[0] if isinstance(info.get('ORIGIN'), list) else info.get('ORIGIN', ''),
                    'filter': 'PASS' if not record.filter.keys() else ';'.join(record.filter.keys())
                }
                
                variants.append(variant)
            
            return variants if len(variants) > 1 else variants[0] if variants else None
            
        except Exception as e:
            logger.warning(f"Error processing variant at {record.chrom}:{record.pos}: {str(e)}")
            return None
    
    def _get_variant_type(self, ref: str, alt: str) -> str:
        """Determine the type of variant."""
        if len(ref) == len(alt):
            return 'SNP' if len(ref) == 1 else 'MNP'
        elif len(ref) > len(alt):
            return 'DEL'
        else:
            return 'INS'


def load_pathogenic_variants_from_vcf(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    clinical_significance: Optional[List[str]] = None,
    review_status: Optional[List[str]] = None,
    min_qual: float = 0.0,
    min_af: float = 0.0,
    max_af: float = 1.0,
    filter_pass: bool = False
) -> pd.DataFrame:
    """
    Load pathogenic variants from ClinVar VCF.
    
    Args:
        vcf_path: Path to VCF file (bgzipped with index)
        chromosomes: List of chromosomes to include
        clinical_significance: List of clinical significance terms to include
        review_status: List of review statuses to include
        min_qual: Minimum quality score
        min_af: Minimum allele frequency
        max_af: Maximum allele frequency
        filter_pass: Only include variants with FILTER=PASS
        
    Returns:
        DataFrame with standardized variant columns
    """
    loader = PathogenicVariantLoader(
        clinical_significance=clinical_significance,
        review_status=review_status,
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
        filter_pass=filter_pass
    )
    return loader.load_from_vcf(vcf_path, chromosomes)


def merge_pathogenic_variants_with_windows(
    variants_df: pd.DataFrame,
    windows_glob: str
) -> pd.DataFrame:
    """
    Merge pathogenic variants with genomic windows.
    
    Args:
        variants_df: DataFrame containing variant information
        windows_glob: Glob pattern for window Parquet files
        
    Returns:
        DataFrame with variants merged into windows
    """
    loader = PathogenicVariantLoader()
    return loader.merge_with_windows(variants_df, windows_glob)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare pathogenic variants for training')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--windows', required=True, help='Glob pattern for window files')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--chromosomes', nargs='+', help='Chromosomes to include')
    parser.add_argument('--min-qual', type=float, default=0.0, help='Minimum quality score')
    parser.add_argument('--min-af', type=float, default=0.0, help='Minimum allele frequency')
    parser.add_argument('--max-af', type=float, default=1.0, help='Maximum allele frequency')
    parser.add_argument('--filter-pass', action='store_true', help='Only include PASS variants')
    parser.add_argument('--clinical-sig', nargs='+', 
                       default=['Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'],
                       help='Clinical significance terms to include')
    parser.add_argument('--review-status', nargs='+',
                       default=['criteria_provided', 'reviewed_by_expert_panel', 'practice_guideline'],
                       help='Review statuses to include')
    
    args = parser.parse_args()
    
    # Load variants
    variants = load_pathogenic_variants_from_vcf(
        vcf_path=args.vcf,
        chromosomes=args.chromosomes,
        clinical_significance=args.clinical_sig,
        review_status=args.review_status,
        min_qual=args.min_qual,
        min_af=args.min_af,
        max_af=args.max_af,
        filter_pass=args.filter_pass
    )
    
    if not variants.empty:
        # Merge with windows
        result = merge_pathogenic_variants_with_windows(
            variants_df=variants,
            windows_glob=args.windows
        )
        
        # Save results
        result.to_parquet(args.output)
        logger.info(f"Saved {len(result)} records to {args.output}")
    else:
        logger.warning("No variants found matching criteria")
