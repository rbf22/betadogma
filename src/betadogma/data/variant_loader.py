"""
Base class for variant loading and processing.

This module provides a base class for loading and processing different types of
variants (common, pathogenic, splice) with consistent interfaces and utilities.
"""

import logging
import os
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VariantLoader(ABC):
    """Base class for loading and processing genomic variants.
    
    This class provides common functionality for loading variants from VCF files,
    processing them, and merging with genomic windows. Specific variant types
    should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, 
                min_qual: float = 0.0,
                min_af: float = 0.0,
                max_af: float = 1.0,
                filter_pass: bool = False):
        """Initialize the variant loader.
        
        Args:
            min_qual: Minimum quality score (default: 0.0, no filter)
            min_af: Minimum allele frequency (default: 0.0, no filter)
            max_af: Maximum allele frequency (default: 1.0, no filter)
            filter_pass: Only include variants with FILTER=PASS (default: False)
        """
        self.min_qual = min_qual
        self.min_af = min_af
        self.max_af = max_af
        self.filter_pass = filter_pass
        self.variants = []
        self._variant_columns = [
            'chrom', 'pos', 'ref', 'alt', 'af', 'source', 'is_pathogenic'
        ]
    
    @abstractmethod
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
        pass
        
    @abstractmethod
    def _process_variant_record(self, record) -> Optional[Dict[str, Any]]:
        """Process a single variant record from a VCF.
        
        This method should be implemented by subclasses to handle specific
        variant types and extract relevant information.
        
        Args:
            record: A pysam.VariantRecord object
            
        Returns:
            Dictionary with variant information or None if variant should be skipped
        """
        pass
    
    def _parse_info_field(self, info_str: str) -> Dict[str, Any]:
        """Parse VCF INFO field into a dictionary with proper type conversion.
        
        Args:
            info_str: The INFO field string from a VCF
            
        Returns:
            Dictionary with parsed INFO fields
        """
        info = {}
        if not info_str or info_str == '.':
            return info
            
        for item in info_str.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                # Convert to appropriate type
                if value.isdigit():
                    info[key] = int(value)
                else:
                    try:
                        info[key] = float(value)
                    except ValueError:
                        info[key] = value
            else:
                info[item] = True
        return info
    
    def _get_allele_frequency(self, info: Dict[str, Any]) -> float:
        """Extract allele frequency from INFO dictionary.
        
        Args:
            info: Dictionary of INFO fields
            
        Returns:
            Allele frequency as a float, or 0.0 if not found
        """
        # Try common AF tags
        for tag in ['AF', 'MAF', 'FREQ', 'AF_popmax']:
            if tag in info:
                try:
                    af = info[tag]
                    if isinstance(af, (list, tuple)):
                        af = af[0]  # Take first value for multi-allelic variants
                    return float(af)
                except (ValueError, TypeError):
                    continue
        return 0.0
    
    def load_from_vcf(self, 
                     vcf_path: str, 
                     chromosomes: Optional[List[str]] = None) -> pd.DataFrame:
        """Load variants from a VCF file.
        
        Args:
            vcf_path: Path to VCF file (bgzipped with index)
            chromosomes: List of chromosomes to include (None for all)
            
        Returns:
            DataFrame with loaded variants
        """
        logger.info(f"Loading variants from: {vcf_path}")
        
        variants = []
        total_variants = 0
        processed_variants = 0
        skipped_counts = {
            'quality': 0,
            'af': 0,
            'filter': 0,
            'other': 0
        }
        
        try:
            logger.debug(f"Opening VCF file: {vcf_path}")
            vcf = pysam.VariantFile(vcf_path)
            
            # Get chromosomes to process, adapting requested names to header convention
            vcf_chroms = list(vcf.header.contigs)
            if chromosomes:
                from .chrom_utils import match_chroms_to_header
                target_chroms = match_chroms_to_header(chromosomes, vcf_chroms)
                # Keep only those present in header
                target_chroms = [c for c in target_chroms if str(c) in vcf_chroms]
            else:
                target_chroms = vcf_chroms
            
            logger.info(f"Processing {len(target_chroms)} chromosomes: {', '.join(target_chroms)}")
            logger.debug(f"VCF contigs: {vcf_chroms}")
            
            for chrom in target_chroms:
                logger.info(f"Processing {chrom}...")
                chrom_variants = 0
                
                try:
                    for record in vcf.fetch(chrom):
                        total_variants += 1
                        
                        if total_variants % 1000 == 0:
                            logger.debug(f"Processed {total_variants} variants, found {len(variants)} so far...")
                        
                        # Apply basic filters
                        if self.min_qual > 0 and record.qual is not None and record.qual < self.min_qual:
                            skipped_counts['quality'] += 1
                            continue
                            
                        if self.filter_pass and record.filter is not None and 'PASS' not in record.filter:
                            skipped_counts['filter'] += 1
                            continue
                        
                        # Let subclass process the variant
                        variant = self._process_variant_record(record)
                        if variant is not None:
                            if isinstance(variant, list):
                                variants.extend(variant)
                                chrom_variants += len(variant)
                            else:
                                variants.append(variant)
                                chrom_variants += 1
                    
                    logger.info(f"Found {chrom_variants} variants on {chrom}")
                    processed_variants += chrom_variants
                            
                except Exception as e:
                    logger.warning(f"Error processing {chrom}: {str(e)}")
                    continue
                    
            vcf.close()
            
        except Exception as e:
            logger.error(f"Error loading VCF: {str(e)}")
            raise
        
        # Create DataFrame
        if not variants:
            logger.warning("No variants found matching criteria")
            return pd.DataFrame(columns=self._variant_columns)
        
        df = pd.DataFrame(variants)
        
        # Log statistics
        logger.info(f"Processed {total_variants} total variants")
        logger.info(f"  Kept: {len(df)} variants")
        logger.info(f"  Skipped: {sum(skipped_counts.values())} variants")
        for reason, count in skipped_counts.items():
            if count > 0:
                logger.info(f"    - {reason}: {count}")
        
        return df
    
    def merge_with_windows(self, 
                         variants_df: pd.DataFrame,
                         windows_files: Union[List[str], str]) -> pd.DataFrame:
        """Merge variants with genomic windows.
        
        Args:
            variants_df: DataFrame containing variants
            windows_files: List of paths to window files (parquet format)
            
        Returns:
            DataFrame with merged variant and window information
        """
        # Expand glob pattern if a string was provided
        if isinstance(windows_files, str):
            windows_files = sorted(glob.glob(windows_files))
        
        # Pre-compute variant positions for faster lookup
        variants_df = variants_df.copy()
        variants_df['pos'] = variants_df['pos'].astype(int)
        variants_df = variants_df.sort_values(['chrom', 'pos'])

        # Normalize variant chrom naming to match windows convention if possible
        try:
            from .chrom_utils import detect_convention, normalize_chroms, UCSC, NCBI
            # Peek into first non-empty window file to detect convention
            window_chroms_sample = []
            for wf in windows_files:
                try:
                    df_sample = pd.read_parquet(wf, columns=['chrom'])
                    if not df_sample.empty:
                        window_chroms_sample = df_sample['chrom'].astype(str).head(10).tolist()
                        break
                except Exception:
                    continue
            if window_chroms_sample:
                win_conv = detect_convention(window_chroms_sample)
                # If conventions differ, normalize variants
                var_conv = detect_convention(variants_df['chrom'].astype(str).head(10).tolist())
                if var_conv != win_conv:
                    variants_df['chrom'] = normalize_chroms(variants_df['chrom'].astype(str).tolist(), win_conv)
        except Exception:
            # Best-effort normalization; continue if detection fails
            pass
        
        # Use the output directory for intermediate files
        # Get the directory from the first window file as a fallback
        default_output_dir = os.path.dirname(os.path.dirname(os.path.dirname(windows_files[0]))) if windows_files else 'output'
        output_dir = os.path.join(default_output_dir, 'splice_variants')
        
        # Try to get the output directory from the command line arguments if available
        try:
            import sys
            if '--output' in sys.argv:
                output_path = sys.argv[sys.argv.index('--output') + 1]
                output_dir = os.path.dirname(os.path.abspath(output_path))
        except (IndexError, ValueError):
            pass  # Fall back to default output_dir
            
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        total_variants = 0
        skipped_variants = 0
        processed_windows = 0
        
        # Process one window file at a time
        for i, window_file in enumerate(tqdm(windows_files, desc="Processing window files")):
                try:
                    # Read the window file
                    window_df = pd.read_parquet(window_file)
                    if window_df.empty:
                        logger.debug(f"Empty window file: {window_file}")
                        continue
                        
                    # Process each window in the file
                    batch_data = []
                    for _, window in window_df.iterrows():
                        try:
                            if 'chrom' not in window:
                                logger.warning("Skipping window with missing 'chrom' column")
                                continue
                                
                            chrom = window['chrom']
                            start = window['start']
                            end = window['end']
                            seq = window.get('seq', '')
                            
                            # Skip if sequence is not available
                            if not isinstance(seq, str) or len(seq) == 0:
                                continue
                            # Normalize sequence case to avoid mismatches on lowercase
                            seq_upper = seq.upper()
                            
                            processed_windows += 1
                            
                            # Find variants in this window
                            window_variants = variants_df[
                                (variants_df['chrom'] == chrom) &
                                (variants_df['pos'] > start) &
                                (variants_df['pos'] <= end)
                            ]
                            
                            # Process variants in this window
                            if not window_variants.empty:
                                for _, variant in window_variants.iterrows():
                                    rel_pos = variant['pos'] - start
                                    
                                    # Skip if variant is out of bounds
                                    if rel_pos <= 0 or rel_pos > len(seq):
                                        skipped_variants += 1
                                        continue
                                    
                                    ref = variant['ref']
                                    alt = variant['alt']
                                    
                                    # Verify reference allele matches the sequence
                                    if seq_upper[rel_pos-1:rel_pos-1+len(ref)] != ref:
                                        skipped_variants += 1
                                        continue
                                    
                                    # Add variant to batch
                                    # Create base variant data
                                    variant_data = {
                                        'chromosome': chrom,
                                        'start': start,
                                        'end': end,
                                        'seq': seq,
                                        'has_variant': True,
                                        'variant_pos': rel_pos,
                                        'variant_ref': ref,
                                        'variant_alt': alt,
                                        'variant_af': variant.get('af', 0.0),
                                        'is_pathogenic': variant.get('is_pathogenic', False)
                                    }
                                    
                                    # Add splice effect information if available
                                    if 'splice_effect' in variant:
                                        variant_data['splice_effect'] = variant['splice_effect']
                                    if 'splice_score' in variant:
                                        variant_data['splice_score'] = variant['splice_score']
                                    if 'gene' in variant:
                                        variant_data['splice_gene'] = variant['gene']
                                    if 'hgvs' in variant:
                                        variant_data['splice_hgvs'] = variant['hgvs']
                                    if 'method' in variant:
                                        variant_data['splice_method'] = variant['method']
                                    
                                    batch_data.append(variant_data)
                                    total_variants += 1
                            else:
                                # Add as negative example (no variants in this window)
                                batch_data.append({
                                    'chromosome': chrom,
                                    'start': start,
                                    'end': end,
                                    'seq': seq,
                                    'has_variant': False,
                                    'variant_pos': None,
                                    'variant_ref': None,
                                    'variant_alt': None,
                                    'variant_af': 1.0,
                                    'is_pathogenic': False,
                                    'splice_effect': 'NONE',
                                    'splice_score': 0.0,
                                    'splice_gene': None,
                                    'splice_hgvs': None,
                                    'splice_method': None
                                })
                            
                        except Exception as e:
                            logger.warning(f"Error processing window: {e}")
                            continue
                    
                    # Save batch to temporary file if we have data
                    if batch_data:
                        batch_df = pd.DataFrame(batch_data)
                        output_file = os.path.join(output_dir, f'batch_{i:06d}.parquet')
                        batch_df.to_parquet(output_file, index=False)
                        output_files.append(output_file)
                            
                except Exception as e:
                    logger.error(f"Error processing window file {window_file}: {e}")
                    continue
                    
                    # If no variants, add as negative example
                    if window_variants.empty:
                        batch_data.append({
                            'chromosome': chrom,
                            'start': start,
                            'end': end,
                            'seq': seq,
                            'has_variant': False,
                            'variant_pos': None,
                            'variant_ref': None,
                            'variant_alt': None,
                            'variant_af': 1.0,
                            'is_pathogenic': False,
                            'splice_effect': 'NONE',
                            'splice_score': 0.0,
                            'splice_gene': None,
                            'splice_hgvs': None,
                            'splice_method': None
                        })
                        continue
                    
                    # Process each variant in the window
                    for _, variant in window_variants.iterrows():
                        rel_pos = variant['pos'] - start
                        
                        # Skip if variant is out of bounds
                        if rel_pos <= 0 or rel_pos > len(seq):
                            skipped_variants += 1
                            continue
                        
                        ref = variant['ref']
                        alt = variant['alt']
                        
                        # Verify reference allele matches the sequence
                        if seq[rel_pos-1:rel_pos-1+len(ref)] != ref:
                            skipped_variants += 1
                            continue
                        
                        # Add variant to batch
                        batch_data.append({
                            'chromosome': chrom,
                            'start': start,
                            'end': end,
                            'seq': seq,
                            'has_variant': True,
                            'variant_pos': rel_pos,
                            'variant_ref': ref,
                            'variant_alt': alt,
                            'variant_af': variant.get('af', 0.0),
                            'is_pathogenic': variant.get('is_pathogenic', False)
                        })
                        total_variants += 1
                
                    # Save batch to temporary file if we have data
                    if batch_data:
                        batch_df = pd.DataFrame(batch_data)
                        output_file = os.path.join(output_dir, f'batch_{i:06d}.parquet')
                        batch_df.to_parquet(output_file, index=False)
                        output_files.append(output_file)
                        
                except Exception as e:
                    logger.error(f"Error processing {window_file}: {str(e)}")
                    continue
        
        # Combine all temporary files if we have any
        if not output_files:
            logger.warning("No valid data to process")
            return pd.DataFrame(columns=[
                'chromosome', 'start', 'end', 'seq', 'has_variant',
                'variant_pos', 'variant_ref', 'variant_alt', 'variant_af', 'is_pathogenic',
                'donor', 'acceptor', 'tss', 'polya'
            ])
            
        # Process files in chunks to avoid memory issues
        chunk_size = 50  # Adjust based on available memory
        chunks = [output_files[i:i + chunk_size] for i in range(0, len(output_files), chunk_size)]
        result_chunks = []
        
        for chunk_files in chunks:
            # Read and combine a chunk of files, filtering out empty DataFrames
            dfs = []
            for f in chunk_files:
                try:
                    df = pd.read_parquet(f)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {f}: {str(e)}")
            
            if dfs:  # Only concatenate if we have non-empty DataFrames
                chunk_df = pd.concat(dfs, ignore_index=True)
                result_chunks.append(chunk_df)
            
            # No need to clean up files as they're in the cache directory
        
        # Combine all non-empty chunks
        if not result_chunks:
            result = pd.DataFrame()
        else:
            # Filter out any empty DataFrames that might have been added
            non_empty_chunks = [df for df in result_chunks if not df.empty]
            if non_empty_chunks:
                result = pd.concat(non_empty_chunks, ignore_index=True)
            else:
                result = pd.DataFrame()
        
        # Log statistics
        logger.info(f"Processed {processed_windows} windows, found {total_variants} valid variants")
        if skipped_variants > 0:
            logger.info(f"Skipped {skipped_variants} variants due to reference mismatches or out-of-bounds")
            
        if result.empty:
            return pd.DataFrame(columns=[
                'chromosome', 'start', 'end', 'seq', 'has_variant',
                'variant_pos', 'variant_ref', 'variant_alt', 'variant_af', 'is_pathogenic',
                'donor', 'acceptor', 'tss', 'polya'
            ])
        
        return result
