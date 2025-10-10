"""
Rule-based NMD heuristics (55-nt rule, last-exon exceptions, etc.).

This module implements Nonsense-Mediated mRNA Decay (NMD) prediction rules based on
the position of premature termination codons (PTCs) relative to exon-exon junctions.

Conventions
-----------
- Exon coordinates are genomic, half-open: [start, end).
- Isoform.exons are in transcription order for the given strand.
- Isoform.cds is a genomic interval (start, end) covering the CDS, also half-open
  by default. If your CDS is inclusive, set cds_inclusive=True in calls.
- All positions are 0-based, following Python conventions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, TypeVar, TypedDict, Union, cast

from ..decoder.types import Exon, Isoform

# Type variable for generic numeric types
N = TypeVar('N', int, float)

# Default PTC rule threshold in nucleotides (commonly 50-55)
DEFAULT_PTC_THRESHOLD: Final[int] = 55

# Type aliases for clarity
Strand = Literal['+', '-']
GenomicPosition = int
TranscriptPosition = int


def _get_cds(iso: Isoform, cds_inclusive: bool = False) -> Optional[Tuple[int, int]]:
    """Helper to get CDS coordinates with proper handling of inclusive/exclusive ranges.
    
    Args:
        iso: Isoform with CDS information
        cds_inclusive: If True, treat CDS coordinates as inclusive
        
    Returns:
        Tuple of (cds_start, cds_end) in half-open coordinates,
        or None if no CDS is present.
    """
    if iso.cds is None:
        return None
    
    cds_start, cds_end = iso.cds
    if cds_inclusive:
        cds_end += 1  # Convert inclusive end to half-open
    
    return (cds_start, cds_end)


def _last_junction_tx_coord(iso: Isoform) -> Optional[TranscriptPosition]:
    """
    Calculate the transcript coordinate of the last exon-exon junction.

    Args:
        iso: Isoform with exons in transcription order

    Returns:
        Transcript coordinate (0-based) of the last exon-exon junction,
        or None if the isoform has fewer than 2 exons.

    Notes:
        - For + strand, this is the end of the second-to-last exon
        - For - strand, this is the start of the second exon
        - Coordinates are 0-based, matching Python string/sequence indexing
    """
    if not hasattr(iso, 'exons') or len(iso.exons) < 2:
        return None

    try:
        if iso.strand == '+':
            # For + strand, last junction is at the end of the second-to-last exon
            return sum(e.end - e.start for e in iso.exons[:-1]) - 1
        elif iso.strand == '-':
            # For - strand, last junction is at the start of the second exon
            return sum(e.end - e.start for e in iso.exons[1:])
        else:
            raise ValueError(f"Invalid strand: {iso.strand}. Must be '+' or '-'")
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Error calculating last junction position: {e}")

def _ptc_genomic_start(iso: Isoform, cds_inclusive: bool = False) -> Optional[GenomicPosition]:
    """
    Calculate the genomic coordinate of the start of the stop codon (PTC).
{{ ... }}
    Args:
        iso: Isoform with CDS information
        cds_inclusive: If True, treat CDS coordinates as inclusive

    Returns:
        Genomic coordinate (0-based) of the start of the stop codon (PTC),
        or None if CDS is missing or invalid.

    Notes:
        - For cds_inclusive=True, cds=(start,end) is treated as inclusive.
        - On the + strand, stop codon starts at end-2 (inclusive) or end-3 (half-open).
        - On the - strand, stop codon always starts at cds_start.
        
    Examples:
        >>> # Plus strand, half-open CDS [100, 200)
{{ ... }}
        100
        
    Raises:
        ValueError: If the strand is invalid or CDS coordinates are invalid
        TypeError: If the input types are incorrect
    """
    try:
        if iso.cds is None:
            return None
            
        cds_start, cds_end = iso.cds
        
        # Validate CDS coordinates
        if not (isinstance(cds_start, int) and isinstance(cds_end, int)):
            raise TypeError(f"CDS coordinates must be integers, got {type(cds_start)} and {type(cds_end)}")
            
        if cds_start >= cds_end:
            raise ValueError(f"Invalid CDS coordinates: start ({cds_start}) must be < end ({cds_end})")
        
        # Calculate PTC position based on strand
        if iso.strand == '+':
            if cds_inclusive:
                if cds_end < 2:
                    raise ValueError(f"CDS end ({cds_end}) too small for inclusive coordinates")
                return cds_end - 2  # Last 3 bases: end-2, end-1, end
            else:
                if cds_end < 3:
                    raise ValueError(f"CDS end ({cds_end}) too small for half-open coordinates")
                return cds_end - 3  # Last 3 bases: end-3, end-2, end-1
                
        elif iso.strand == '-':
            return cds_start  # Stop codon is at the start of CDS on - strand
            
        else:
            raise ValueError(f"Invalid strand: {iso.strand}. Must be '+' or '-'")
            
    except (TypeError, ValueError) as e:
        raise type(e)(f"Error in _ptc_genomic_start: {str(e)}") from e


def _map_genomic_to_tx(iso: Isoform, gpos: GenomicPosition) -> Optional[TranscriptPosition]:
    """
    Map a genomic position to its transcript (cDNA) coordinate.

    Args:
        iso: Isoform with exons in transcription order
        gpos: Genomic position (0-based) to map

    Returns:
        0-based transcript coordinate if gpos is within an exon, None otherwise.

    Notes:
        - For '+' strand: coordinate is offset from exon.start
        - For '-' strand: coordinate is offset from exon 3' end (exon.end - 1 - gpos)
        
    Examples:
        >>> # Plus strand example
        >>> iso_plus = Isoform(exons=[(100, 200), (300, 400)], strand='+', meta={})
        >>> _map_genomic_to_tx(iso_plus, 150)  # 150-100 = 50
        50
        >>> _map_genomic_to_tx(iso_plus, 350)  # 100 + (350-300) = 150
        150
        
        >>> # Minus strand example
        >>> iso_minus = Isoform(exons=[(100, 200), (300, 400)], strand='-', meta={})
        >>> # First exon: 199-150 = 49 (0-based from end of first exon)
        >>> _map_genomic_to_tx(iso_minus, 150)
        49
    """
    tx_pos = 0
    for ex in iso.exons:  # already in transcription order
        if ex.start <= gpos < ex.end:
            if iso.strand == '+':
                return tx_pos + (gpos - ex.start)
            return tx_pos + (ex.end - 1 - gpos)
        tx_pos += ex.length
    return None


def ptc_distance_to_last_junction_tx(
    isoform: Isoform,
    cds_inclusive: bool = False,
) -> Optional[int]:
    """
    Calculate the distance from the PTC to the last exon-exon junction.

    Args:
        isoform: Isoform with exons and CDS information
        cds_inclusive: If True, treat CDS coordinates as inclusive

    Returns:
        Signed distance in nucleotides (last_junction_tx - ptc_tx):
        - Positive: PTC is upstream of last junction (NMD-prone)
        - Negative: PTC is at/after last junction (NMD-escaping)
        - None: if single-exon, missing CDS, or PTC not mappable
    """
    last_junc = _last_junction_tx_coord(isoform)
    if last_junc is None:
        return None

    ptc_g = _ptc_genomic_start(isoform, cds_inclusive=cds_inclusive)
    if ptc_g is None:
        return None

    ptc_tx = _map_genomic_to_tx(isoform, ptc_g)
    if ptc_tx is None:
        return None  # CDS stop not inside provided exons

    return last_junc - ptc_tx


def rule_ptc_before_last_junction(
    isoform: Isoform,
    ptc_rule_threshold: int = DEFAULT_PTC_THRESHOLD,
    cds_inclusive: bool = False,
    strict_gt: bool = True,
) -> bool:
    """
    Determine if a PTC is likely to trigger NMD based on the 55-nt rule.

    Args:
        isoform: Isoform with exons in transcription order and CDS information
        ptc_rule_threshold: Distance threshold in nucleotides (default: 55)
        cds_inclusive: If True, treat CDS coordinates as inclusive [start, end]
        strict_gt: If True, use '>' for comparison; if False, use '>='

    Returns:
        True if the PTC is more than `ptc_rule_threshold` nt upstream of the
        last exon-exon junction, indicating likely NMD targeting.

    Notes:
        - Only meaningful for multi-exon isoforms with a mapped CDS stop.
        - The 55-nt rule is an empirical proxy for EJC deposition.
        - For single-exon transcripts or missing CDS, returns False.
    """
    dist = ptc_distance_to_last_junction_tx(isoform, cds_inclusive=cds_inclusive)
    if dist is None:
        return False
    return (dist > ptc_rule_threshold) if strict_gt else (dist >= ptc_rule_threshold)