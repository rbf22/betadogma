"""
Rule-based NMD heuristics (55-nt rule, last-exon exceptions, etc.).

Conventions
-----------
- Exon coordinates are genomic, half-open: [start, end).
- Isoform.exons are in transcription order for the given strand.
- Isoform.cds is a genomic interval (start, end) covering the CDS, also half-open
  by default. If your CDS is inclusive, set cds_inclusive=True in calls.
"""

from __future__ import annotations
from typing import Optional, Tuple

from ..decoder.types import Isoform


def _last_junction_tx_coord(iso: Isoform) -> Optional[int]:
    """
    Transcript index (0-based, in nucleotides) of the last exon–exon junction.
    Returns None for single-exon isoforms.
    """
    if len(iso.exons) < 2:
        return None
    # Sum of all exon lengths except the last one.
    return sum(e.end - e.start for e in iso.exons[:-1])


def _ptc_genomic_start(iso: Isoform, cds_inclusive: bool = False) -> Optional[int]:
    """
    Genomic coordinate (0-based) of the *start* of the stop codon (PTC).
    Returns None if CDS is missing.

    If cds_inclusive=True, cds=(start,end) is treated as inclusive, so the
    stop codon starts at end-2 on '+' and at start on '-' (after reverse logic).
    In the default half-open case, stop codon starts at end-3 on '+'.
    """
    if iso.cds is None:
        return None
    cds_start, cds_end = iso.cds  # genomic
    if cds_end <= cds_start:
        return None

    if iso.strand == '+':
        # Plus strand: stop is at the *end* of CDS region.
        if cds_inclusive:
            # ...XYZ[stop] with end pointing to the last base (inclusive)
            return cds_end - 2  # start of the 3-bp stop codon
        else:
            # Half-open: CDS is [start, end), stop codon occupies end-3..end-1
            return cds_end - 3
    else:
        # Minus strand: CDS runs right->left; the stop codon is at genomic CDS start.
        # For both inclusive and half-open, the *start* of the stop codon is cds_start
        # because the three bases are cds_start..cds_start+2 on the minus strand.
        return cds_start


def _map_genomic_to_tx(iso: Isoform, gpos: int) -> Optional[int]:
    """
    Map a genomic position inside an exon to transcript (cDNA) index (0-based).
    Returns None if gpos is not within any exon.

    For '+' strand: offset from exon.start
    For '-' strand: offset from exon 3' end, i.e., (exon.end - 1 - gpos)
    """
    tx = 0
    for ex in iso.exons:  # already in transcription order
        if ex.start <= gpos < ex.end:
            if iso.strand == '+':
                return tx + (gpos - ex.start)
            else:
                return tx + (ex.end - 1 - gpos)
        tx += (ex.end - ex.start)
    return None


def ptc_distance_to_last_junction_tx(
    isoform: Isoform,
    cds_inclusive: bool = False,
) -> Optional[int]:
    """
    Compute signed distance (in nt) from the PTC (start of stop codon) to the last
    exon–exon junction, both measured along the transcript (cDNA) coordinate.

    Returns:
        distance = last_junction_tx - ptc_tx
        - Positive: PTC is upstream of the last junction (NMD-prone).
        - Negative: PTC is at/after last junction (usually NMD-escaping).
        - None: if single-exon, missing CDS, or PTC not mappable.
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
    ptc_rule_threshold: int = 55,
    cds_inclusive: bool = False,
    strict_gt: bool = True,
) -> bool:
    """
    Return True if the PTC is more than `ptc_rule_threshold` nt upstream of the
    last exon–exon junction (the classic "55-nt rule").

    Args:
        isoform: Isoform with exons in transcription order; isoform.cds is genomic (start,end).
        ptc_rule_threshold: Threshold in nucleotides (commonly 50–55).
        cds_inclusive: Set True if isoform.cds is inclusive [start,end] instead of half-open.
        strict_gt: If True, use 'distance > threshold'; if False, use '>= threshold'.

    Notes:
        - This rule is only meaningful for multi-exon isoforms with a mapped CDS stop.
        - This function does not model EJC deposition (~20–24 nt upstream of junction);
          the 55-nt rule is an empirical proxy for that effect.
    """
    dist = ptc_distance_to_last_junction_tx(isoform, cds_inclusive=cds_inclusive)
    if dist is None:
        return False
    return (dist > ptc_rule_threshold) if strict_gt else (dist >= ptc_rule_threshold)