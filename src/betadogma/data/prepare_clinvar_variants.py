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
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from betadogma.data.variant_loader import VariantLoader
from betadogma.data.vcf_processor import VCFProcessor
import pysam

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinvarVariantLoader(VariantLoader):
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
            
    def get_variants_for_sequence(self, sequence: str, chrom: Optional[str] = None, 
                                start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get variants that overlap with the specified genomic region.
        
        Args:
            sequence: Chromosome/contig name
            start: Start position (0-based)
            end: End position (1-based)
            
        Returns:
            List of variant dictionaries
        """
        if not hasattr(self, 'variants_df'):
            return []
            
        # Filter variants by chromosome and position
        mask = (self.variants_df['chrom'] == sequence)
        if start is not None:
            mask &= (self.variants_df['pos'] >= start)
        if end is not None:
            mask &= (self.variants_df['pos'] <= end)
            
        return self.variants_df[mask].to_dict('records')


def load_clinvar_variants_from_vcf(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    clinical_significance: Optional[List[str]] = None,
    review_status: Optional[List[str]] = None,
    min_qual: float = 0.0,
    min_af: float = 0.0,
    max_af: float = 1.0,
    filter_pass: bool = False,
    keep_processed_vcf: bool = False
) -> pd.DataFrame:
    """
    Load ClinVar variants from VCF file using VCFProcessor for consistent handling.
    
    This function loads variants from a ClinVar VCF file, processes them with VCFProcessor
    for consistent chromosome naming and filtering, and then applies ClinVar-specific filters.
    
    Args:
        vcf_path: Path to VCF file (can be gzipped)
        chromosomes: List of chromosomes to include (with or without 'chr' prefix)
        clinical_significance: List of clinical significance terms to include
        review_status: List of review statuses to include
        min_qual: Minimum quality score
        min_af: Minimum allele frequency
        max_af: Maximum allele frequency
        filter_pass: Only include variants with FILTER=PASS
        keep_processed_vcf: Keep the processed VCF file for debugging
        
    Returns:
        DataFrame with standardized variant columns
    """
    logger.info(f"Loading ClinVar variants with settings:")
    logger.info(f"  VCF: {vcf_path}")
    logger.info(f"  Chromosomes: {chromosomes}")
    logger.info(f"  Clinical significance: {clinical_significance}")
    logger.info(f"  Review status: {review_status}")
    logger.info(f"  Min quality: {min_qual}")
    logger.info(f"  AF range: {min_af} - {max_af}")
    logger.info(f"  Filter PASS only: {filter_pass}")
    
    # Check if VCF file exists
    vcf_path = Path(vcf_path)
    if not vcf_path.exists():
        logger.error(f"VCF file not found: {vcf_path}")
        return pd.DataFrame()
    
    # Create a temporary directory for processed VCF
    with tempfile.TemporaryDirectory(prefix='clinvar_') as temp_dir:
        temp_dir = Path(temp_dir)
        processed_vcf = temp_dir / 'processed.vcf.gz'
        processed_is_temp = False
        
        try:
            # Skip VCF processing and use the original file
            # This avoids the chromosome normalization issue
            processed_vcf = Path(vcf_path)
            processed_is_temp = False
            
            # Ensure the index exists (create alongside original if missing)
            tbi_path = Path(f"{vcf_path}.tbi")
            csi_path = Path(f"{vcf_path}.csi")
            if not tbi_path.exists() and not csi_path.exists():
                logger.info("Indexing VCF...")
                try:
                    # Prefer CSI for very large VCFs; fall back to TBI
                    pysam.tabix_index(str(processed_vcf), preset="vcf", force=False)
                except Exception:
                    # As a fallback, try creating CSI via pysam (if configured) or ignore
                    pass
            
            # Initialize the ClinVar variant loader
            loader = ClinvarVariantLoader(
                clinical_significance=clinical_significance or [
                    'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'
                ],
                review_status=review_status or [
                    'criteria_provided', 'reviewed_by_expert_panel', 'practice_guideline'
                ],
                min_qual=min_qual,
                min_af=min_af,
                max_af=max_af,
                filter_pass=filter_pass
            )
            
            # Load variants from the processed VCF
            logger.info("Loading variants from processed VCF...")
            variants = loader.load_from_vcf(processed_vcf, chromosomes)
            
            # Log summary
            if not variants.empty:
                logger.info(f"Successfully loaded {len(variants)} ClinVar variants")
                if 'clinical_significance' in variants.columns:
                    sig_counts = variants['clinical_significance'].value_counts()
                    logger.info("Clinical significance counts:\n" + 
                              "\n".join(f"  {k}: {v}" for k, v in sig_counts.items()))
            else:
                logger.warning("No variants were loaded. This might indicate an issue with the VCF or filtering criteria.")
                
                # Try to diagnose the issue
                try:
                    with pysam.VariantFile(processed_vcf) as vcf:
                        sample_records = []
                        for i, record in enumerate(vcf):
                            if i >= 5:  # Just get first 5 records for debugging
                                break
                            sample_records.append({
                                'chrom': record.chrom,
                                'pos': record.pos,
                                'id': record.id,
                                'ref': record.ref,
                                'alts': record.alts,
                                'qual': record.qual,
                                'filter': list(record.filter) if record.filter else [],
                                'info': {k: v for k, v in record.info.items() if k in ['CLNSIG', 'CLNREVSTAT', 'AF']}
                            })
                        if sample_records:
                            logger.debug("Sample VCF records (first 5):")
                            for rec in sample_records:
                                logger.debug(f"  {rec}")
                except Exception as e:
                    logger.error(f"Error reading VCF file for debugging: {str(e)}")
            
            # Optionally keep the processed VCF for debugging (only for temp files)
            if keep_processed_vcf and processed_is_temp and processed_vcf.exists():
                debug_vcf = Path("clinvar_processed.vcf.gz")
                import shutil
                shutil.copy2(processed_vcf, debug_vcf)
                # Copy index if present
                for idx_ext in (".tbi", ".csi"):
                    src = Path(f"{processed_vcf}{idx_ext}")
                    dst = Path(f"{debug_vcf}{idx_ext}")
                    if src.exists():
                        shutil.copy2(src, dst)
                logger.info(f"Processed VCF saved to: {debug_vcf}")
            
            return variants
            
        except Exception as e:
            logger.error(f"Error processing ClinVar VCF: {str(e)}", exc_info=True)
            return pd.DataFrame()
        
        finally:
            # Clean up only temporary processed files, never the original VCF
            if processed_is_temp:
                try:
                    if not keep_processed_vcf and processed_vcf.exists():
                        processed_vcf.unlink(missing_ok=True)
                    for idx_ext in (".tbi", ".csi"):
                        idx_path = Path(f"{processed_vcf}{idx_ext}")
                        if idx_path.exists():
                            idx_path.unlink()
                except Exception:
                    # Best-effort cleanup; ignore failures
                    pass


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
    loader = ClinvarVariantLoader()
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
    variants = load_clinvar_variants_from_vcf(
        vcf_path=args.vcf,
        chromosomes=args.chromosomes,
        clinical_significance=args.clinical_sig,
        review_status=args.review_status,
        min_qual=args.min_qual,
        min_af=args.min_af,
        max_af=args.max_af,
        filter_pass=args.filter_pass
    )
    
    if variants.empty:
        logger.error("No variants found matching the specified criteria. This could be due to:")
        logger.error(f"- No variants in the specified chromosomes: {args.chromosomes}")
        logger.error(f"- No variants with the specified clinical significance: {args.clinical_sig}")
        logger.error(f"- No variants with the specified review status: {args.review_status}")
        logger.error("Please check your input VCF file and filtering criteria.")
        
        # Try to get some diagnostic information
        try:
            import pysam
            with pysam.VariantFile(args.vcf) as vcf:
                total_variants = sum(1 for _ in vcf)
                logger.error(f"Total variants in VCF: {total_variants}")
                
                # Get first few records for debugging
                vcf.reset()
                sample_records = []
                for i, record in enumerate(vcf):
                    if i >= 3:  # Just get first 3 records for debugging
                        break
                    sample_records.append({
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'id': record.id,
                        'ref': record.ref,
                        'alts': record.alts,
                        'qual': record.qual,
                        'filter': list(record.filter) if record.filter else [],
                        'info': {k: v for k, v in record.info.items() if k in ['CLNSIG', 'CLNREVSTAT', 'AF']}
                    })
                
                if sample_records:
                    logger.error("Sample VCF records (first 3):")
                    for rec in sample_records:
                        logger.error(f"  {rec}")
                        
        except Exception as e:
            logger.error(f"Error reading VCF file for diagnostics: {str(e)}")
            
        sys.exit(1)
    
    # If we get here, we have variants to process
    # Expand windows glob and merge
    try:
        import glob as _glob
        window_files = sorted(_glob.glob(args.windows))
        if not window_files:
            logger.warning(f"No window files found matching pattern: {args.windows}")
        else:
            logger.info(f"Merging ClinVar variants with {len(window_files)} window files")
        result = merge_pathogenic_variants_with_windows(
            variants_df=variants,
            windows_glob=window_files if window_files else args.windows
        )
        # Validate expected columns are present post-merge
        expected_cols = {'has_variant','variant_pos','variant_ref','variant_alt'}
        missing = [c for c in expected_cols if c not in result.columns]
        if missing:
            raise RuntimeError(f"Post-merge result missing expected columns: {missing}. Aborting to avoid saving unusable output.")
    except Exception as e:
        import traceback
        logger.error("Failed during ClinVar merge with windows:")
        logger.error(traceback.format_exc())
        raise
    
    # Save results (ensure directory exists)
    try:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path)
        logger.info(f"Successfully processed and saved {len(result)} records to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save ClinVar results to {args.output}: {e}")
        raise
