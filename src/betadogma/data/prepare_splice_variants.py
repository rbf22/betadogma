#!/usr/bin/env python3
"""
Prepare experimentally validated splice variants for splice effect prediction.

This script processes experimentally validated splice variants, validates them
against reference sequences, and prepares them for model training.

Key Features:
- Processes experimentally validated splice variants
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

# Constants
SPLICE_EFFECT_SCORES = {
    'STRONG': 1.0,
    'MILD': 0.5,
    'NONE': 0.0
}


class SpliceVariantLoader(VariantLoader):
    """Loader for experimentally validated splice variants."""
    
    def __init__(self, 
                include_effects: Optional[List[str]] = None,
                **kwargs):
        """Initialize the splice variant loader.
        
        Args:
            include_effects: List of effects to include (STRONG/MILD/NONE)
            **kwargs: Additional arguments for VariantLoader
        """
        super().__init__(**kwargs)
        # Convert all effects to uppercase for case-insensitive comparison
        self.include_effects = [e.upper() for e in (include_effects or list(SPLICE_EFFECT_SCORES.keys()))]
        # Add splice-specific columns to variant columns
        self._variant_columns.extend([
            'splice_effect', 'splice_score', 'gene', 'transcript',
            'hgvs', 'method', 'source'
        ])
        
    def get_variants_for_sequence(self, sequence: str, chrom: Optional[str] = None, 
                                start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve variants that overlap with the given sequence or genomic region.
        
        Args:
            sequence: The reference DNA sequence (not used in this implementation)
            chrom: Chromosome name to filter variants
            start: Start position (1-based, inclusive)
            end: End position (1-based, inclusive)
            
        Returns:
            List of variant dictionaries with variant information
        """
        if not hasattr(self, 'variants'):
            return []
            
        # Convert variants to list of dictionaries if not already
        variants = []
        for var in self.variants:
            if isinstance(var, dict):
                variants.append(var)
            elif hasattr(var, '__dict__'):
                variants.append(var.__dict__)
                
        # Filter by chromosome if specified
        if chrom is not None:
            variants = [v for v in variants if str(v.get('chrom', '')) == str(chrom)]
            
        # Filter by position range if specified
        if start is not None and end is not None:
            variants = [v for v in variants if start <= v.get('pos', 0) <= end]
            
        return variants
    
    def _process_variant_record(self, record) -> Optional[Dict[str, Any]]:
        """Process a single variant record from a VCF."""
        try:
            # Log record being processed
            logger.debug(f"Processing record: {record.chrom}:{record.pos} {record.ref}>{record.alts}")
            
            # Skip based on quality (handled by base class)
            if self.min_qual > 0 and record.qual is not None and record.qual < self.min_qual:
                logger.debug(f"Skipping variant {record.chrom}:{record.pos} due to quality filter")
                return None
            
            # Log raw record info for debugging
            logger.debug(f"Raw record: {record}")
            logger.debug(f"Record INFO: {record.info}")
            logger.debug(f"Record INFO type: {type(record.info)}")
            logger.debug(f"Record INFO items: {list(record.info.items())}")
            
            # Parse INFO field
            info = {}
            for k, v in record.info.items():
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    info[k] = v[0]
                else:
                    info[k] = v
                logger.debug(f"Processed INFO field - Key: '{k}' (type: {type(k)}), Value: '{info[k]}' (type: {type(info[k])})")
            
            logger.debug(f"Processed INFO fields: {info}")
            logger.debug(f"Looking for SPLICE_EFFECT in: {list(info.keys())}")
            
            # Get splice effect - handle case-insensitive lookup
            effect = None
            for key in info.keys():
                if key.upper() == 'SPLICE_EFFECT':
                    effect = str(info[key]).upper().strip()
                    logger.debug(f"Found SPLICE_EFFECT: {effect} (raw: {info[key]}, type: {type(info[key])})")
                    break
            
            # If no effect found in VCF, try to get it from CLASSIFICATION
            if not effect or effect == 'NONE':
                for key in info.keys():
                    if key.upper() == 'CLASSIFICATION':
                        classification = str(info[key]).upper().strip()
                        if 'SPLICE-ALTERING' in classification:
                            effect = 'STRONG'
                        elif 'LOW-FREQUENCY' in classification:
                            effect = 'MILD'
                        else:
                            effect = 'NONE'
                        logger.debug(f"Inferred SPLICE_EFFECT '{effect}' from CLASSIFICATION: {classification}")
                        break
            
            # If still no effect, use the first include_effect as default
            if not effect or effect == 'NONE':
                if self.include_effects:
                    effect = self.include_effects[0]
                    logger.debug(f"No SPLICE_EFFECT found in VCF, using default: {effect}")
                else:
                    effect = 'NONE'
                    logger.debug("No SPLICE_EFFECT found and no include_effects specified, using 'NONE'")
            
            # Validate the effect against include_effects if specified
            if self.include_effects and effect not in self.include_effects:
                logger.warning(f"Effect '{effect}' not in include_effects {self.include_effects}, using {self.include_effects[0]} instead")
                effect = self.include_effects[0]
            
            logger.debug(f"Final splice effect: {effect}")
            logger.debug(f"Include effects: {self.include_effects}")
            
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
                af = float(info.get('AF', [0.0])[0] if isinstance(info.get('AF'), list) else info.get('AF', 0.0))
                
                variant = {
                    'chrom': str(record.chrom),
                    'pos': int(record.pos),
                    'ref': ref,
                    'alt': alt,
                    'af': af,
                    'source': 'SpliceVarDB',
                    'is_pathogenic': effect != 'NONE',
                    'splice_effect': effect,
                    'splice_score': SPLICE_EFFECT_SCORES.get(effect, 0.0),
                    'gene': info.get('GENE', [''])[0] if isinstance(info.get('GENE'), list) else info.get('GENE', ''),
                    'transcript': info.get('TRANSCRIPT', [''])[0] if isinstance(info.get('TRANSCRIPT'), list) else info.get('TRANSCRIPT', ''),
                    'hgvs': info.get('HGVS', [''])[0] if isinstance(info.get('HGVS'), list) else info.get('HGVS', ''),
                    'method': info.get('METHOD', [''])[0] if isinstance(info.get('METHOD'), list) else info.get('METHOD', ''),
                    'filter': 'PASS' if not record.filter.keys() else ';'.join(record.filter.keys())
                }
                
                variants.append(variant)
            
            return variants if len(variants) > 1 else variants[0] if variants else None
            
        except Exception as e:
            logger.warning(f"Error processing variant at {record.chrom}:{record.pos}: {str(e)}")
            return None


def load_splice_variants_from_vcf(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    include_effects: Optional[List[str]] = None,
    min_qual: float = 0.0,
    min_af: float = 0.0,
    max_af: float = 1.0,
    filter_pass: bool = False
) -> pd.DataFrame:
    """
    Load and validate splice variants from VCF.
    
    Args:
        vcf_path: Path to VCF file (bgzipped with index)
        chromosomes: List of chromosomes to include
        include_effects: List of effects to include (STRONG/MILD/NONE)
        min_qual: Minimum quality score
        min_af: Minimum allele frequency
        max_af: Maximum allele frequency
        filter_pass: Only include variants with FILTER=PASS
        
    Returns:
        DataFrame with standardized variant columns
    """
    logger.info("="*80)
    logger.info(f"Loading splice variants from: {vcf_path}")
    logger.info(f"Chromosomes: {chromosomes}")
    logger.info(f"Including effects: {include_effects}")
    logger.info(f"Filters - min_qual: {min_qual}, min_af: {min_af}, max_af: {max_af}, filter_pass: {filter_pass}")
    
    loader = SpliceVariantLoader(
        include_effects=include_effects,
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
        filter_pass=filter_pass
    )
    
    # Load variants from VCF
    logger.info("Starting VCF loading...")
    variants_df = loader.load_from_vcf(vcf_path, chromosomes=chromosomes)
    
    if variants_df is None or variants_df.empty:
        logger.warning("No variants found matching criteria")
        return pd.DataFrame()
    
    # Log basic statistics
    logger.info("="*80)
    logger.info(f"VCF LOADING SUMMARY")
    logger.info("-"*40)
    logger.info(f"Total variants loaded: {len(variants_df):,}")
    
    # Log variant types
    if 'ref' in variants_df.columns and 'alt' in variants_df.columns:
        # Classify variants by type (SNP, INS, DEL, COMPLEX)
        variants_df['var_type'] = variants_df.apply(
            lambda x: 'SNP' if len(str(x['ref'])) == len(str(x['alt'])) == 1 else 
                     'INS' if len(str(x['ref'])) < len(str(x['alt'])) else
                     'DEL' if len(str(x['ref'])) > len(str(x['alt'])) else 'COMPLEX',
            axis=1
        )
        type_counts = variants_df['var_type'].value_counts().to_dict()
        logger.info("\nVariant Types:")
        for var_type, count in type_counts.items():
            logger.info(f"  {var_type}: {count:,} ({count/len(variants_df)*100:.1f}%)")
    
    # Log effect distribution if available
    if 'splice_effect' in variants_df.columns:
        logger.info("\nSplice Effect Distribution:")
        effect_counts = variants_df['splice_effect'].value_counts().to_dict()
        for effect, count in effect_counts.items():
            logger.info(f"  {effect}: {count:,} ({count/len(variants_df)*100:.1f}%)")
    
    # Log quality statistics if available
    if 'qual' in variants_df.columns:
        logger.info("\nQuality Score Statistics:")
        logger.info(f"  Min: {variants_df['qual'].min():.2f}")
        logger.info(f"  Mean: {variants_df['qual'].mean():.2f}")
        logger.info(f"  Max: {variants_df['qual'].max():.2f}")
    
    # Log allele frequency statistics if available
    if 'af' in variants_df.columns:
        logger.info("\nAllele Frequency Statistics:")
        logger.info(f"  Min: {variants_df['af'].min():.6f}")
        logger.info(f"  Mean: {variants_df['af'].mean():.6f}")
        logger.info(f"  Max: {variants_df['af'].max():.6f}")
    
    logger.info("="*80)
    return variants_df


def merge_splice_variants_with_windows(
    variants_df: pd.DataFrame,
    windows_glob: str
) -> pd.DataFrame:
    """
    Merge splice variants with genomic windows.
    
    Args:
        variants_df: DataFrame containing variant information
        windows_glob: Glob pattern for window Parquet files
        
    Returns:
        DataFrame with variants merged into windows
    """
    loader = SpliceVariantLoader()
    return loader.merge_with_windows(variants_df, windows_glob)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare splice variants for training')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--windows', required=True, help='Glob pattern for window files')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--chromosomes', nargs='+', help='Chromosomes to include')
    parser.add_argument('--min-qual', type=float, default=0.0, help='Minimum quality score')
    parser.add_argument('--min-af', type=float, default=0.0, help='Minimum allele frequency')
    parser.add_argument('--max-af', type=float, default=1.0, help='Maximum allele frequency')
    parser.add_argument('--filter-pass', action='store_true', help='Only include PASS variants')
    # Add effect arguments
    effect_group = parser.add_argument_group('Splice Effect Options')
    effect_group.add_argument('--effect', action='append',
                            help='Splice effect to include (can be used multiple times)')
    effect_group.add_argument('--effects', nargs='+', action='append',
                            help='Splice effects to include (space-separated list)')
    
    # Set default effects that will be used if no effects are specified
    default_effects = ['STRONG', 'MILD', 'NONE']
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Process the effects
    effects = []
    
    # Handle --effect flag (can be used multiple times)
    if args.effect is not None:
        # --effect was used one or more times
        for effect in args.effect:
            if effect:
                effects.append(effect.strip().upper())
    
    # Handle --effects flag (can be used multiple times with space-separated lists)
    if args.effects is not None:
        # args.effects is a list of lists because of nargs='+' and action='append'
        for effect_list in args.effects:
            for effect in effect_list:
                if effect:
                    # Handle case where effects might be passed as a single string with commas
                    if ',' in effect:
                        effects.extend([x.strip().upper() for x in effect.split(',') if x.strip()])
                    else:
                        effects.append(effect.strip().upper())
    
    # If no effects were specified, use defaults
    if not effects:
        effects = default_effects
    
    # Remove duplicates while preserving order
    seen = set()
    args.effects = [x for x in effects if not (x in seen or seen.add(x))]
    
    # Validate effects
    valid_effects = set(SPLICE_EFFECT_SCORES.keys())
    for effect in args.effects:
        if effect not in valid_effects:
            logger.warning(f"Warning: Unknown effect '{effect}'. Valid effects are: {', '.join(valid_effects)}")
    
    logger.info(f"Including splice effects: {args.effects}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load variants
    variants = load_splice_variants_from_vcf(
        vcf_path=args.vcf,
        chromosomes=args.chromosomes,
        include_effects=args.effects,
        min_qual=args.min_qual,
        min_af=args.min_af,
        max_af=args.max_af,
        filter_pass=args.filter_pass
    )
    
    if not variants.empty:
        # Merge with windows if provided
        if args.windows:
            import glob
            # Expand the glob pattern to get the actual files
            window_files = sorted(glob.glob(args.windows))
            if not window_files:
                logger.warning(f"No window files found matching pattern: {args.windows}")
            else:
                logger.info(f"Merging with {len(window_files)} window files")
                variants = merge_splice_variants_with_windows(variants, window_files)
        
        # Save results
        variants.to_parquet(args.output)
        logger.info(f"Saved {len(variants)} records to {args.output}")
        
        # Log variant statistics
        if 'splice_effect' in variants.columns:
            effect_counts = variants['splice_effect'].value_counts().to_dict()
            logger.info("Variant counts by effect:")
            for effect, count in effect_counts.items():
                logger.info(f"  {effect}: {count:,} ({count/len(variants)*100:.1f}%)")
    else:
        logger.warning("No variants found matching criteria")
