"""
Chromosome naming and processing utilities.

This module provides functions for normalizing and validating chromosome names
across different naming conventions (UCSC, NCBI, Ensembl).
"""
from typing import Set, List, Union, Optional
import re
from pathlib import Path
import pysam

# Standard chromosome names (UCSC format)
STANDARD_CHROMS = [f"chr{chrom}" for chrom in (*range(1, 23), 'X', 'Y', 'M')]

# Common chromosome name variants
CHROM_ALIASES = {
    'MT': 'M',
    'chrMT': 'chrM',
    '23': 'X',
    '24': 'Y',
    '25': 'M',
    'chr23': 'chrX',
    'chr24': 'chrY',
    'chr25': 'chrM'
}

def normalize_chrom(chrom: str, prefix: str = 'chr') -> str:
    """
    Normalize chromosome name to standard format.
    
    Args:
        chrom: Input chromosome name (e.g., '1', 'chr1', 'MT')
        prefix: Desired prefix ('chr' or '')
        
    Returns:
        Normalized chromosome name (e.g., 'chr1', 'chrM')
    """
    if not chrom:
        raise ValueError("Chromosome name cannot be empty")
        
    # Convert to string and strip any existing prefix
    chrom = str(chrom)
    chrom = re.sub(r'^chr|^chrom', '', chrom, flags=re.IGNORECASE)
    
    # Handle special cases and aliases
    chrom = CHROM_ALIASES.get(chrom, chrom)
    
    # Add prefix if needed
    if prefix and not chrom.startswith(prefix):
        chrom = f"{prefix}{chrom}"
    
    return chrom

def get_chrom_set(include_sex: bool = True, include_mito: bool = False) -> Set[str]:
    """
    Get standard set of chromosome names.
    
    Args:
        include_sex: Include sex chromosomes (X, Y)
        include_mito: Include mitochondrial DNA (M/MT)
        
    Returns:
        Set of chromosome names
    """
    chroms = [f"chr{i}" for i in range(1, 23)]
    if include_sex:
        chroms.extend(['chrX', 'chrY'])
    if include_mito:
        chroms.append('chrM')
    return set(chroms)

def validate_vcf_chroms(vcf_path: Union[str, Path], 
                       expected_chroms: Optional[Set[str]] = None) -> bool:
    """
    Validate that a VCF contains expected chromosomes.
    
    Args:
        vcf_path: Path to VCF file (can be bgzipped)
        expected_chroms: Set of expected chromosome names
        
    Returns:
        True if all expected chromosomes are found
    """
    if expected_chroms is None:
        expected_chroms = get_chrom_set()
    
    try:
        with pysam.VariantFile(str(vcf_path)) as vcf:
            found_chroms = {normalize_chrom(rec.chrom) for rec in vcf}
    except Exception as e:
        raise ValueError(f"Error reading VCF file: {e}")
    
    return bool(found_chroms & expected_chroms)

def get_vcf_chroms(vcf_path: Union[str, Path]) -> List[str]:
    """Get list of chromosomes present in a VCF file."""
    try:
        with pysam.VariantFile(str(vcf_path)) as vcf:
            return sorted({normalize_chrom(rec.chrom) for rec in vcf})
    except Exception as e:
        raise ValueError(f"Error reading VCF file: {e}")

class ChromosomeNormalizer:
    """Context manager for handling chromosome name normalization in VCF files."""
    
    def __init__(self, input_path: Union[str, Path], 
                 output_path: Union[str, Path],
                 target_chroms: Optional[Set[str]] = None):
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.target_chroms = target_chroms
        
    def __enter__(self):
        self.vcf_in = pysam.VariantFile(self.input_path)
        self.vcf_out = pysam.VariantFile(self.output_path, 'w', header=self.vcf_in.header)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vcf_in.close()
        self.vcf_out.close()
        
    def process(self):
        """Process the VCF, normalizing chromosome names and filtering by target chromosomes."""
        for record in self.vcf_in:
            orig_chrom = record.chrom
            normalized_chrom = normalize_chrom(orig_chrom)
            
            # Only process and write records for target chromosomes
            if self.target_chroms is None or normalized_chrom in self.target_chroms:
                record.chrom = normalized_chrom
                self.vcf_out.write(record)
            
        return str(self.output_path)
