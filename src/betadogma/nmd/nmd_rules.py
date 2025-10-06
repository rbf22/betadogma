"""
Rule-based NMD heuristics (55-nt rule, last-exon exceptions, etc.).
"""
from ..decoder.types import Isoform


def rule_ptc_before_last_junction(isoform: Isoform, ptc_rule_threshold: int = 55) -> bool:
    """
    Return True if PTC > `ptc_rule_threshold` nt upstream of the last exon junction.

    This function implements the classic "55-nucleotide rule" for nonsense-mediated
    decay (NMD). An isoform is predicted to be an NMD target if its stop codon
    (premature termination codon or PTC) is located more than a certain distance
    upstream of the final exon-exon junction.

    Args:
        isoform: The isoform object to check. It must have its `cds` attribute
                 populated with the genomic coordinates of the coding sequence.
                 The `exons` attribute must be in transcription order.
        ptc_rule_threshold: The distance threshold in nucleotides.

    Returns:
        True if the isoform is predicted to be an NMD target, False otherwise.
    """
    # Rule only applies to spliced transcripts (more than one exon)
    if len(isoform.exons) < 2:
        return False

    # Rule requires a defined CDS to locate the PTC
    if isoform.cds is None:
        return False

    # 1. Find the transcript coordinate of the last exon-exon junction.
    # `isoform.exons` is in transcription order. The junction is after the second-to-last exon.
    last_junction_tx_coord = sum(e.end - e.start for e in isoform.exons[:-1])

    # 2. Find the transcript coordinate of the PTC (start of the stop codon).
    if isoform.strand == '+':
        # For plus strand, the stop codon is at the end of the CDS.
        # cds format is (start, end) of the entire CDS region.
        # The stop codon starts at end - 3.
        ptc_genomic_coord = isoform.cds[1] - 3
    else:  # strand == '-'
        # For minus strand, the stop codon is at the start of the CDS in genomic terms.
        ptc_genomic_coord = isoform.cds[0]

    ptc_tx_coord = -1
    len_before_exon = 0
    for exon in isoform.exons:  # Exons are in transcription order
        if exon.start <= ptc_genomic_coord < exon.end:
            # Found the exon containing the PTC
            if isoform.strand == '+':
                offset_in_exon = ptc_genomic_coord - exon.start
            else:  # strand == '-'
                # Transcription is right-to-left. The coordinate is relative to the exon's 3' end.
                offset_in_exon = exon.end - 1 - ptc_genomic_coord
            ptc_tx_coord = len_before_exon + offset_in_exon
            break
        len_before_exon += (exon.end - exon.start)

    # If PTC wasn't found in any exon, something is wrong.
    if ptc_tx_coord == -1:
        return False

    # 3. Calculate the distance and check the rule.
    # The distance is from the PTC to the junction.
    distance_from_ptc_to_junction = last_junction_tx_coord - ptc_tx_coord

    # A positive distance means the PTC is upstream of the junction.
    return distance_from_ptc_to_junction > ptc_rule_threshold