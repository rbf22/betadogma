"""
Synthetic mutagenesis utilities for training variant sensitivity.
"""

from __future__ import annotations
from typing import List, Optional

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
def _rc(s: str) -> str: return "".join(_COMP.get(b, "N") for b in reversed(s.upper()))


# --- Nonsense (PTC) creation ---

def _single_snp_to_stop(triplet: str):
    stops = {"TAA", "TAG", "TGA"}
    bases = ["A", "C", "G", "T"]
    for i in range(3):
        for b in bases:
            if b == triplet[i]:
                continue
            if triplet[:i] + b + triplet[i+1:] in stops:
                return i, b
    return None


def make_ptc_variant(chrom: str, codon_start: int, strand: str, ref_triplet: str) -> Optional[str]:
    """Create a single-base edit that turns a codon into a stop."""
    edit = _single_snp_to_stop(ref_triplet.upper())
    if not edit:
        return None
    off, new_base = edit
    if strand == "+":
        pos0, ref = codon_start + off, ref_triplet[off]
        alt = new_base
    else:
        pos0 = codon_start + (2 - off)
        ref = _COMP[ref_triplet[off]]
        alt = _COMP[new_base]
    return f"{chrom}:{pos0 + 1}{ref}>{alt}"


# --- Splice / Kozak / PolyA ---

def make_splice_site_variant(chrom: str, site_pos0: int, kind: str, strand: str, enforce=True) -> str:
    """Mutate a splice site donor/acceptor motif."""
    canon_plus = "GT" if kind == "donor" else "AG"
    canon_minus = _rc(canon_plus)
    ref = canon_plus[0] if strand == "+" else canon_minus[0]
    alt = ref if enforce else {"A", "C", "G", "T"}.difference({ref}).pop()
    return f"{chrom}:{site_pos0 + 1}{ref}>{alt}"


def make_kozak_variant(chrom: str, atg_start: int, strand: str, strengthen=True) -> List[str]:
    """Mutate -3 and +3 positions around ATG to strengthen/weaken Kozak consensus."""
    base_m3 = "G" if strengthen else "C"
    base_p3 = "G" if strengthen else "A"
    specs = []
    if strand == "+":
        if atg_start - 3 >= 0:
            specs.append(f"{chrom}:{atg_start - 3 + 1}N>{base_m3}")
        specs.append(f"{chrom}:{atg_start + 3 + 1}N>{base_p3}")
    else:
        specs.append(f"{chrom}:{atg_start + 5 + 1}N>{_COMP[base_m3]}")
        specs.append(f"{chrom}:{atg_start - 3 + 1}N>{_COMP[base_p3]}")
    return specs


def make_polya_disruption_variant(chrom: str, signal_start0: int, strand: str) -> str:
    """Disrupt 'AATAAA' polyA motif."""
    if strand == "+":
        return f"{chrom}:{signal_start0 + 1}A>C"
    elif strand == "-":
        return f"{chrom}:{signal_start0 + 1}T>A"
    raise ValueError("strand must be '+' or '-'")