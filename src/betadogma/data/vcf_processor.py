"""
VCF Processing Module

This module provides functionality for processing VCF files with consistent
chromosome naming and quality control.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pysam
from tqdm import tqdm

from .chrom_utils import ChromosomeNormalizer, normalize_chrom_convention, detect_convention
from ..config import get_config

logger = logging.getLogger(__name__)

class VCFProcessor:
    """Process VCF files with consistent chromosome naming and filtering."""
    
    def __init__(self, config: Optional[Union[str, Path, object]] = None):
        """Initialize the VCF processor.
        
        Args:
            config: Either a config object or path to configuration file
        """
        if isinstance(config, (str, Path)) or config is None:
            self.config = get_config(config)
        else:
            self.config = config
        self.chroms = self.config.get_chromosomes()
        # Determine target convention from configured chromosomes
        self.target_convention = 'ucsc' if any(str(c).lower().startswith('chr') for c in self.chroms) else 'ncbi'
        
    def process_vcf(
        self,
        input_vcf: Union[str, Path],
        output_vcf: Optional[Union[str, Path]] = None,
        force_overwrite: bool = False
    ) -> str:
        """Process a VCF file with consistent chromosome naming.
        
        Args:
            input_vcf: Path to input VCF file (can be bgzipped)
            output_vcf: Path to output VCF file (default: input_vcf + ".normalized")
            force_overwrite: Overwrite output file if it exists
            
        Returns:
            Path to the processed VCF file
        """
        input_vcf = Path(input_vcf)
        if not input_vcf.exists():
            raise FileNotFoundError(f"Input VCF not found: {input_vcf}")
            
        if output_vcf is None:
            output_vcf = input_vcf.with_suffix(".normalized.vcf.gz")
        output_vcf = Path(output_vcf)
        
        if output_vcf.exists() and not force_overwrite:
            logger.info(f"Using existing file: {output_vcf}")
            return str(output_vcf)
        
        logger.info(f"Processing VCF: {input_vcf}")
        logger.debug(f"Target chromosomes: {sorted(self.chroms)}")
        
        # Ensure output directory exists
        output_vcf.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the VCF
        with ChromosomeNormalizer(
            input_path=input_vcf,
            output_path=output_vcf,
            target_chroms=self.chroms,
            target_convention=self.target_convention
        ) as normalizer:
            normalizer.process()
        
        logger.info(f"Processed VCF saved to: {output_vcf}")
        return str(output_vcf)
    
    def filter_vcf(
        self,
        input_vcf: Union[str, Path],
        output_vcf: Optional[Union[str, Path]] = None,
        min_qual: float = 0.0,
        min_af: float = 0.0,
        max_af: float = 1.0,
        filter_pass: bool = False,
        force_overwrite: bool = False
    ) -> str:
        """Filter a VCF file based on quality and allele frequency.
        
        Args:
            input_vcf: Path to input VCF file (can be bgzipped)
            output_vcf: Path to output VCF file (default: input_vcf + ".filtered")
            min_qual: Minimum quality score
            min_af: Minimum allele frequency
            max_af: Maximum allele frequency
            filter_pass: Only include variants with FILTER=PASS
            force_overwrite: Overwrite output file if it exists
            
        Returns:
            Path to the filtered VCF file
        """
        input_vcf = Path(input_vcf)
        if not input_vcf.exists():
            raise FileNotFoundError(f"Input VCF not found: {input_vcf}")
            
        if output_vcf is None:
            output_vcf = input_vcf.with_suffix(".filtered.vcf.gz")
        output_vcf = Path(output_vcf)
        
        if output_vcf.exists() and not force_overwrite:
            logger.info(f"Using existing file: {output_vcf}")
            return str(output_vcf)
        
        logger.info(f"Filtering VCF: {input_vcf}")
        logger.debug(f"Filters: QUAL >= {min_qual}, AF: {min_af}-{max_af}, PASS only: {filter_pass}")
        
        # Ensure output directory exists
        output_vcf.parent.mkdir(parents=True, exist_ok=True)
        
        # Open input and output VCFs
        with pysam.VariantFile(input_vcf) as vcf_in:
            # Create output header
            header = vcf_in.header
            
            # Add filters to header if they don't exist
            if 'AF' not in header.info:
                header.add_meta('INFO', items=[('ID', 'AF'), ('Number', 'A'), ('Type', 'Float'),
                                            ('Description', 'Allele frequency')])
            
            with pysam.VariantFile(output_vcf, 'w', header=header) as vcf_out:
                for record in tqdm(vcf_in, desc="Filtering variants"):
                    # Skip based on FILTER
                    if filter_pass and record.filter.keys() and 'PASS' not in record.filter:
                        continue
                    
                    # Skip based on QUAL
                    if min_qual > 0 and record.qual is not None and record.qual < min_qual:
                        continue
                    
                    # Skip if not in target chromosomes or if chromosome format is incorrect
                    if self.chroms:
                        normalized_chrom = normalize_chrom_convention(record.chrom, self.target_convention)
                        if normalized_chrom not in self.chroms:
                            continue
                        record.chrom = normalized_chrom
                        
                    # Get allele frequency
                    af = self._get_allele_frequency(record)
                    logger.debug(f"Variant at {record.chrom}:{record.pos} - AF: {af}")
                    
                    if af is not None and not (min_af <= af <= max_af):
                        logger.debug(f"Skipping variant at {record.chrom}:{record.pos} - AF {af} outside range [{min_af}, {max_af}]")
                        continue
                    
                    logger.debug(f"Including variant at {record.chrom}:{record.pos} - AF: {af}")
                    vcf_out.write(record)
        
        logger.info(f"Filtered VCF saved to: {output_vcf}")
        return str(output_vcf)
    
    def _get_allele_frequency(self, record) -> Optional[float]:
        """Get allele frequency from a VCF record."""
        # Try common AF fields
        for af_field in ['AF', 'AF_popmax', 'AF_global', 'AF_adj']:
            if af_field in record.info:
                af = record.info[af_field]
                if af is None:
                    continue
                if isinstance(af, (list, tuple)) and af:
                    af = af[0]  # Take the first AF value if it's a list
                try:
                    return float(af)
                except (TypeError, ValueError):
                    continue
        
        # Calculate from AC/AN if available
        if 'AC' in record.info and 'AN' in record.info:
            try:
                ac = record.info['AC']
                an = record.info['AN']
                if an <= 0:
                    return None
                if isinstance(ac, (list, tuple)) and ac:
                    return float(ac[0]) / an
                return float(ac) / an
            except (TypeError, ValueError):
                pass
        
        return None


def process_vcf(
    input_vcf: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[object] = None,
    **kwargs
) -> Dict[str, str]:
    """Process a VCF file with the default pipeline.
    
    Args:
        input_vcf: Path to input VCF file
        output_dir: Output directory (default: configured processed data directory)
        config_path: Path to configuration file (ignored if config is provided)
        config: Optional pre-configured config object
        **kwargs: Additional arguments for VCF processing
        
    Returns:
        Dictionary with paths to processed files:
        {
            'normalized': path to normalized VCF,
            'filtered': path to filtered VCF
        }
    """
    if config is None:
        config = get_config(config_path)
    processor = VCFProcessor(config)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(config.get_path("paths", "processed_data")) / "variants"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the VCF
    input_vcf = Path(input_vcf)
    base_name = input_vcf.stem.replace(".vcf", "").replace(".gz", "")
    
    normalized_vcf = output_dir / f"{base_name}.normalized.vcf.gz"
    filtered_vcf = output_dir / f"{base_name}.filtered.vcf.gz"
    
    # Run normalization and filtering
    normalized_path = processor.process_vcf(
        input_vcf=input_vcf,
        output_vcf=normalized_vcf,
        force_overwrite=kwargs.get('force_overwrite', False)
    )
    
    filtered_path = processor.filter_vcf(
        input_vcf=normalized_path,
        output_vcf=filtered_vcf,
        min_qual=kwargs.get('min_qual', 0.0),
        min_af=kwargs.get('min_af', 0.0),
        max_af=kwargs.get('max_af', 1.0),
        filter_pass=kwargs.get('filter_pass', False),
        force_overwrite=kwargs.get('force_overwrite', False)
    )
    
    return {
        'normalized': normalized_path,
        'filtered': filtered_path
    }
