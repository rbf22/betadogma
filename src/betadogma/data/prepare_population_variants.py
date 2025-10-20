#!/usr/bin/env python3
"""
Prepare common variants for model training.

This script processes common variants from population VCFs, applies them to
reference sequences, and prepares them for model training with balanced
representation of variant types.

Key Features:
- Processes common variants from VCF files
- Maintains natural variant type distribution
- Handles SNPs, insertions, and deletions
- Balances variant types within windows
- Outputs sharded Parquet files for efficient storage
"""

import argparse
import gc
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Generator

import numpy as np
import pandas as pd
from tqdm import tqdm

from betadogma.data.variant_loader import VariantLoader
from betadogma.data.encode import (
    encode_variant, 
    apply_variants_to_sequence,
    build_variant_channels,
    rescue_insertion_variant
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class CommonVariantLoader(VariantLoader):
    """Loader for common variants from population VCFs."""
    
    def __init__(self, 
                max_per_window: int = 100,
                balance_variants: bool = True,
                **kwargs):
        """Initialize the common variant loader.
        
        Args:
            max_per_window: Maximum variants per window (0 = unlimited)
            balance_variants: Whether to balance variant types
            **kwargs: Additional arguments for VariantLoader
        """
        super().__init__(**kwargs)
        self.max_per_window = max_per_window
        self.balance_variants = balance_variants
        # Add common variant specific columns
        self._variant_columns.extend([
            'var_type', 'variant_length', 'in_window_idx', 'span_in_window',
            'ch_snp', 'ch_ins', 'ch_del', 'ch_any'
        ])
    
    def get_variants_for_sequence(self, sequence: str, chrom: Optional[str] = None, 
                                start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve variants that overlap with the given sequence or genomic region.
        
        Args:
            sequence: The reference DNA sequence
            chrom: Optional chromosome name to filter variants
            start: Optional start position (1-based)
            end: Optional end position (1-based, inclusive)
            
        Returns:
            List of variant dictionaries with 'pos', 'ref', 'alt' keys
        """
        if not hasattr(self, 'vcf_reader') or self.vcf_reader is None:
            logger.warning("VCF reader not initialized. Call load_vcf() first.")
            return []
            
        if chrom is None:
            # If no chromosome is provided, try to use the first one available
            if not self.variants:
                logger.warning("No variants loaded. Call load_vcf() first.")
                return []
            chrom = self.variants[0]['chrom']
            
        # If positions not provided, use full sequence length
        if start is None:
            start = 1
        if end is None:
            end = len(sequence)
            
        # Fetch variants from the VCF for this region
        variants = []
        try:
            for record in self.vcf_reader.fetch(chrom, start-1, end):  # pysam uses 0-based, half-open
                # Process the variant record
                var = self._process_variant_record(record)
                if var:
                    if isinstance(var, list):
                        variants.extend(var)
                    else:
                        variants.append(var)
        except Exception as e:
            logger.warning(f"Error fetching variants for {chrom}:{start}-{end}: {str(e)}")
            
        return variants
        
    def _process_variant_record(self, record) -> Optional[Dict[str, Any]]:
        """Process a single variant record from a VCF."""
        try:
            # Skip based on quality (handled by base class)
            if self.min_qual > 0 and record.qual is not None and record.qual < self.min_qual:
                return None
            
            # Parse INFO field
            info = {k: v[0] if isinstance(v, tuple) and len(v) == 1 else v 
                   for k, v in dict(record.info).items()}
            
            # Get allele frequency
            af = self._get_allele_frequency(info)
            
            # Filter by allele frequency
            if af < self.min_af or af > self.max_af:
                return None
            
            # Get reference and alternate alleles
            ref = str(record.ref).upper()
            alts = [str(alt).upper() for alt in record.alts] if record.alts else ['.']
            
            variants = []
            
            # Create entry for each alternate allele
            for alt in alts:
                # Determine variant type
                var_type = self._get_variant_type(ref, alt)
                
                variant = {
                    'chrom': str(record.chrom),
                    'pos': int(record.pos),
                    'ref': ref,
                    'alt': alt,
                    'af': af,
                    'source': 'CommonVariant',
                    'is_pathogenic': False,
                    'var_type': var_type,
                    'variant_length': max(len(ref), len(alt)),
                    'id': record.id if record.id != '.' else '',
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
    
    def _get_variant_span(self, variant: Dict[str, Any]) -> Tuple[int, int]:
        """Get the 0-based span (start, end) that a variant affects."""
        pos = variant['pos'] - 1  # Convert to 0-based
        ref = variant['ref']
        
        if variant['var_type'] == 'INS':
            # Insertions don't consume reference bases beyond the anchor
            return (pos, pos + 1)
        else:
            # For SNPs and DELs, the span is the reference bases
            return (pos, pos + len(ref))
    
    def _select_balanced_variants(self, variants: List[Dict], max_variants: int, seed: int) -> List[Dict]:
        """Select variants maintaining natural type distribution while avoiding conflicts."""
        if not variants or max_variants <= 0:
            return []
        
        # Count variants by type
        type_counts = {}
        for v in variants:
            var_type = v.get('var_type', 'UNK')
            type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        # If we have few variants or don't need to balance, return all
        if len(variants) <= max_variants or not self.balance_variants:
            return variants[:max_variants]
        
        # Calculate target counts for each type
        total = sum(type_counts.values())
        target_counts = {}
        remaining = max_variants
        
        # First pass: allocate based on natural distribution
        for var_type, count in type_counts.items():
            target = max(1, int((count / total) * max_variants))
            target = min(target, count, remaining)
            target_counts[var_type] = target
            remaining -= target
        
        # Second pass: distribute any remaining slots
        types = list(target_counts.keys())
        random.Random(seed).shuffle(types)  # For deterministic behavior
        
        for var_type in types:
            if remaining <= 0:
                break
            if target_counts[var_type] < type_counts[var_type]:
                target_counts[var_type] += 1
                remaining -= 1
        
        # Select variants
        selected = []
        by_type = {}
        
        # Group by type
        for v in variants:
            var_type = v.get('var_type', 'UNK')
            if var_type not in by_type:
                by_type[var_type] = []
            by_type[var_type].append(v)
        
        # Select from each type
        for var_type, variants_of_type in by_type.items():
            count = min(target_counts.get(var_type, 0), len(variants_of_type))
            selected.extend(random.Random(seed).sample(variants_of_type, count))
        
        return selected
    
    def merge_with_windows(
        self,
        variants_df: pd.DataFrame,
        windows_glob: str,
        output_dir: str,
        shard_size: int = 1000,
        seed: int = 42,
        apply_alt: bool = False,
        debug: bool = False
    ) -> None:
        """Merge variants with genomic windows and write sharded output.
        
        Args:
            variants_df: DataFrame containing variant information
            windows_glob: Glob pattern for window Parquet files
            output_dir: Directory to write output shards
            shard_size: Number of rows per output shard
            seed: Random seed for reproducibility
            apply_alt: Whether to store sequences with variants applied
            debug: Enable debug output
        """
        import pyarrow.parquet as pq
        import pyarrow as pa
        from pathlib import Path
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process windows in streaming fashion to save memory
        shard_idx = 0
        current_shard = []
        
        # Get list of window files
        window_files = sorted(glob(str(windows_glob)))
        if not window_files:
            raise ValueError(f"No window files found matching: {windows_glob}")
        
        # Process each window file
        for window_file in tqdm(window_files, desc="Processing window files"):
            try:
                # Read windows in chunks
                table = pq.read_table(window_file)
                
                for i in range(0, len(table), 1000):  # Process in chunks of 1000
                    chunk = table.slice(i, min(1000, len(table) - i))
                    df_chunk = chunk.to_pandas()
                    
                    for _, window in df_chunk.iterrows():
                        # Find variants in this window
                        window_variants = self._find_variants_in_window(
                            variants_df, 
                            window['chrom'],
                            window['start'],
                            window['end']
                        )
                        
                        # Skip if no variants in this window
                        if window_variants.empty:
                            continue
                        
                        # Select balanced variants if needed
                        if self.max_per_window > 0 and len(window_variants) > self.max_per_window:
                            window_variants = self._select_balanced_variants(
                                window_variants.to_dict('records'),
                                self.max_per_window,
                                seed
                            )
                            window_variants = pd.DataFrame(window_variants)
                        
                        # Process variants in this window
                        result = self._process_window_variants(
                            window, 
                            window_variants,
                            apply_alt
                        )
                        
                        if result:
                            current_shard.append(result)
                            
                            # Write shard if full
                            if len(current_shard) >= shard_size:
                                self._write_shard(current_shard, shard_idx, output_dir)
                                shard_idx += 1
                                current_shard = []
                                
                                # Log memory usage if psutil is available
                                if HAS_PSUTIL and debug:
                                    process = psutil.Process()
                                    logger.debug(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            except Exception as e:
                logger.error(f"Error processing {window_file}: {str(e)}")
                if debug:
                    raise
        
        # Write any remaining rows in the last shard
        if current_shard:
            self._write_shard(current_shard, shard_idx, output_dir)
    
    def _find_variants_in_window(
        self,
        variants_df: pd.DataFrame,
        chrom: str,
        start: int,
        end: int
    ) -> pd.DataFrame:
        """Find variants within a genomic window."""
        # Convert to 1-based position for comparison
        window_variants = variants_df[
            (variants_df['chrom'] == str(chrom)) &
            (variants_df['pos'] > start) &  # 1-based position > 0-based start
            (variants_df['pos'] <= end)     # 1-based position <= 1-based end (inclusive)
        ].copy()
        
        return window_variants
    
    def _process_window_variants(
        self,
        window: pd.Series,
        variants: pd.DataFrame,
        apply_alt: bool = False
    ) -> Optional[Dict]:
        try:
            # Initialize result with window data
            result = {
                'chromosome': window['chrom'],  
                'start': window['start'],
                'end': window['end'],
                'seq': window['seq'],
                'has_variant': True,
                'variant_pos': [],
                'variant_ref': [],
                'variant_alt': [],
                'variant_af': [],
                'variant_type': [],
                'is_pathogenic': False,
                **{k: window.get(k, np.zeros(len(str(window['seq'])), dtype=np.float32))
                   for k in ['donor', 'acceptor', 'tss', 'polya']}
            }
            
            # Apply each variant to the sequence
            seq = str(window['seq'])  # Ensure seq is a string
            
            for _, variant in variants.iterrows():
                # Calculate position relative to window (0-based)
                rel_pos = int(variant['pos']) - int(window['start']) - 1
                
                # Skip variants outside the sequence
                if rel_pos < 0 or rel_pos >= len(seq):
                    continue
                
                # Verify reference allele matches
                if seq[rel_pos:rel_pos+len(variant['ref'])] != variant['ref']:
                    logger.debug(
                        f"Reference mismatch at {variant['chrom']}:{variant['pos']}. "
                        f"Expected '{variant['ref']}', found '{seq[rel_pos:rel_pos+len(variant['ref'])]}'"
                    )
                    continue
                
                # Add variant information
                result['variant_pos'].append(variant['pos'])
                result['variant_ref'].append(variant['ref'])
                result['variant_alt'].append(variant['alt'])
                result['variant_af'].append(variant.get('af', 0.0))
                result['variant_type'].append(variant.get('var_type', 'UNK'))
                
                # Apply variant to sequence if needed
                if apply_alt:
                    seq = seq[:rel_pos] + variant['alt'] + seq[rel_pos + len(variant['ref']):]
            
            # If no valid variants, return None
            if not result['variant_pos']:
                return None
                
            # Ensure all list fields have the same length
            expected_len = len(result['variant_pos'])
            for field in ['variant_ref', 'variant_alt', 'variant_af', 'variant_type']:
                if len(result[field]) != expected_len:
                    logger.warning(f"Inconsistent field lengths in result. Expected {expected_len}, got {len(result[field])} for {field}")
                    return None
            
            # Store the modified sequence if needed
            if apply_alt:
                result['seq_alt'] = seq
            
            # Convert lists to numpy arrays for better serialization
            for key in ['variant_pos', 'variant_af']:
                if result[key]:
                    result[key] = np.array(result[key], dtype=np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing window {window.get('chrom', 'unknown')}:{window.get('start', '?')}-{window.get('end', '?')}: {str(e)}")
            if 'debug' in window and window['debug']:
                raise
            return None
    
    def _write_shard(self, rows: List[Dict], shard_idx: int, output_dir: Path) -> int:
        """Write a shard of data to disk."""
        if not rows:
            return 0
            
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Write to Parquet
        output_file = output_dir / f"shard_{shard_idx:05d}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        
        logger.info(f"Wrote {len(rows)} rows to {output_file}")
        return len(rows)


def load_common_variants_from_vcf(
    vcf_path: str,
    chromosomes: Optional[List[str]] = None,
    min_qual: float = 0.0,
    min_af: float = 0.0,
    max_af: float = 1.0,
    filter_pass: bool = False
) -> pd.DataFrame:
    """Load common variants from a VCF file.

    Args:
        vcf_path: Path to VCF file (bgzipped with index)
        chromosomes: List of chromosomes to include
        min_qual: Minimum quality score
        min_af: Minimum allele frequency (0.0-1.0)
        max_af: Maximum allele frequency (0.0-1.0)
        filter_pass: Only include variants with FILTER=PASS

    Returns:
        DataFrame containing the loaded variants with standardized variant columns
    """
    loader = CommonVariantLoader(
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
        filter_pass=filter_pass
    )
    return loader.load_from_vcf(vcf_path, chromosomes)


def prepare_common_variants(
    vcf: str,
    windows: str,
    output_dir: str,
    apply_alt: bool = False,
    max_per_window: int = 100,
    shard_size: int = 1000,
    seed: int = 42,
    debug: bool = False,
    min_af: float = 0.0,
    max_af: float = 1.0,
    min_qual: float = 0.0,
    filter_pass: bool = False
) -> None:
    """
    Prepare common variant data for training.
    
    Args:
        vcf: Path to VCF file or glob pattern for VCF files
        windows: Glob pattern for window Parquet shards from prepare_gencode.py
        output_dir: Output directory for variant-aligned Parquet shards
        apply_alt: Whether to store sequences with variants applied
        max_per_window: Maximum variants per window (maintains type balance, 0 = unlimited)
        shard_size: Number of rows per output shard
        seed: Random seed for reproducibility
        debug: Enable debug logging
        min_af: Minimum allele frequency (0.0 = no filter)
        max_af: Maximum allele frequency (1.0 = no filter)
        min_qual: Minimum variant quality score (0.0 = no filter)
        filter_pass: Only include variants with FILTER=PASS
    """
    # Load variants
    logger.info(f"Loading variants from: {vcf}")
    variants = load_common_variants_from_vcf(
        vcf_path=vcf,
        chromosomes=None,  # Process all chromosomes
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
        filter_pass=filter_pass
    )
    
    if variants.empty:
        logger.warning("No variants found matching criteria")
        return
    
    logger.info(f"Loaded {len(variants)} variants")
    
    # Create loader for merging with windows
    loader = CommonVariantLoader(
        max_per_window=max_per_window,
        balance_variants=True,
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
        filter_pass=filter_pass
    )
    
    # Merge with windows and write shards
    logger.info(f"Merging variants with windows: {windows}")
    loader.merge_with_windows(
        variants_df=variants,
        windows_glob=windows,
        output_dir=output_dir,
        shard_size=shard_size,
        seed=seed,
        apply_alt=apply_alt,
        debug=debug
    )
    
    logger.info(f"Finished processing variants. Output written to: {output_dir}")
    
    # Print variant type summary
    if not variants.empty:
        logger.info("\n" + "="*80)
        logger.info("Variant Type Summary")
        logger.info("="*80)
        
        # Basic counts
        total_variants = len(variants)
        logger.info(f"Total variants processed: {total_variants:,}")
        
        # Variant type distribution
        if 'var_type' in variants.columns:
            type_counts = variants['var_type'].value_counts()
            logger.info("\nVariant Types:")
            for var_type, count in type_counts.items():
                pct = (count / total_variants) * 100
                logger.info(f"  - {var_type}: {count:,} ({pct:.1f}%)")
        
        # Variant length distribution
        if 'variant_length' in variants.columns:
            logger.info("\nVariant Length Distribution:")
            length_stats = variants['variant_length'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            for stat, value in length_stats.items():
                if stat != 'count':  # Skip count since we already have total
                    logger.info(f"  - {stat}: {value:.2f}")
        
        # Allele frequency distribution if available
        if 'af' in variants.columns:
            logger.info("\nAllele Frequency Distribution:")
            af_stats = variants['af'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            for stat, value in af_stats.items():
                if stat != 'count':  # Skip count since we already have total
                    logger.info(f"  - {stat}: {value:.4f}")
        
        # Variants per window statistics if available
        if 'window_id' in variants.columns:
            variants_per_window = variants['window_id'].value_counts()
            logger.info("\nVariants per Window:")
            win_stats = variants_per_window.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            for stat, value in win_stats.items():
                if stat != 'count':  # Skip count since it's not meaningful here
                    logger.info(f"  - {stat}: {value:.2f}")
        
        logger.info("="*80 + "\n")


def main():
    """Command-line interface for preparing common variants."""
    parser = argparse.ArgumentParser(description='Prepare common variants for training')
    parser.add_argument('--vcf', required=True, help='Input VCF file or glob pattern')
    parser.add_argument('--windows', required=True, help='Glob pattern for window files')
    parser.add_argument('--output-dir', required=True, help='Output directory for shards')
    parser.add_argument('--apply-alt', action='store_true', help='Store sequences with variants applied')
    parser.add_argument('--max-per-window', type=int, default=100, 
                        help='Maximum variants per window (0 = unlimited)')
    parser.add_argument('--shard-size', type=int, default=1000, 
                        help='Rows per output shard')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging')
    parser.add_argument('--min-af', type=float, default=0.0, 
                        help='Minimum allele frequency (0.0 = no filter)')
    parser.add_argument('--max-af', type=float, default=1.0, 
                        help='Maximum allele frequency (1.0 = no filter)')
    parser.add_argument('--min-qual', type=float, default=0.0, 
                        help='Minimum variant quality score (0.0 = no filter)')
    parser.add_argument('--filter-pass', action='store_true', 
                        help='Only include variants with FILTER=PASS')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the pipeline
    prepare_common_variants(
        vcf=args.vcf,
        windows=args.windows,
        output_dir=args.output_dir,
        apply_alt=args.apply_alt,
        max_per_window=args.max_per_window,
        shard_size=args.shard_size,
        seed=args.seed,
        debug=args.debug,
        min_af=args.min_af,
        max_af=args.max_af,
        min_qual=args.min_qual,
        filter_pass=args.filter_pass
    )


if __name__ == "__main__":
    main()
