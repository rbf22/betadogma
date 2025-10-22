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

# Supported naming conventions
UCSC = "ucsc"   # chr1..chr22, chrX, chrY, chrM
NCBI = "ncbi"   # 1..22, X, Y, MT

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

def strip_chr_prefix(chrom: str) -> str:
    """Remove common chromosome prefixes (e.g., 'chr', 'chrom')."""
    if chrom is None:
        return chrom
    return re.sub(r'^(chr|chrom)', '', str(chrom), flags=re.IGNORECASE)

def _to_ucsc(core: str) -> str:
    """Convert a core chrom token (e.g., '1','X','Y','M','MT') to UCSC form."""
    token = CHROM_ALIASES.get(core, core)
    # UCSC uses chrM for mitochondrial
    if token in {"MT", "M"}:
        return "chrM"
    return f"chr{token}"

def _to_ncbi(core: str) -> str:
    """Convert a core chrom token (e.g., '1','X','Y','M','MT') to NCBI form."""
    token = CHROM_ALIASES.get(core, core)
    # NCBI uses MT for mitochondrial
    if token in {"M", "MT"}:
        return "MT"
    return f"{token}"

def normalize_chrom_convention(chrom: str, convention: str = UCSC) -> str:
    """
    Normalize a chromosome name to a target convention.

    Args:
        chrom: Input chromosome (e.g., '1','chr1','MT','chrM')
        convention: 'ucsc' or 'ncbi'

    Returns:
        Normalized chromosome string in the target convention.
    """
    if not chrom:
        raise ValueError("Chromosome name cannot be empty")
    core = strip_chr_prefix(str(chrom))
    core = CHROM_ALIASES.get(core, core)
    # Normalize special cases like lower-case x,y,m
    core = core.upper() if core.isalpha() else core
    if convention == UCSC:
        return _to_ucsc(core)
    elif convention == NCBI:
        return _to_ncbi(core)
    else:
        # Default to UCSC if unknown
        return _to_ucsc(core)

def normalize_chroms(chroms: List[str], convention: str = UCSC) -> List[str]:
    """Normalize a list of chromosome names to the given convention."""
    return [normalize_chrom_convention(c, convention) for c in chroms]

def detect_convention(chroms: List[str]) -> str:
    """Heuristically detect naming convention from a list of chromosomes."""
    if not chroms:
        return UCSC
    if any(str(c).lower().startswith('chr') for c in chroms):
        return UCSC
    return NCBI

def match_chroms_to_header(requested: Optional[List[str]], header_chroms: List[str]) -> List[str]:
    """
    Convert requested chromosome names to match the convention of VCF header contigs.

    If requested is None, returns header chromosomes as-is.
    """
    if not requested:
        return list(header_chroms)
    header_conv = detect_convention(list(header_chroms))
    return normalize_chroms([str(c) for c in requested], header_conv)

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
                 target_chroms: Optional[Set[str]] = None,
                 target_convention: str = UCSC):
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.target_chroms = target_chroms
        self.target_convention = target_convention
        
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
            normalized_chrom = normalize_chrom_convention(orig_chrom, self.target_convention)
            
            # Only process and write records for target chromosomes
            if self.target_chroms is None or normalized_chrom in self.target_chroms:
                record.chrom = normalized_chrom
                self.vcf_out.write(record)
            
        return str(self.output_path)
