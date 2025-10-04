"""
Rule-based NMD heuristics (55-nt rule, last-exon exceptions, etc.).
"""
from typing import Dict, Any

def rule_ptc_before_last_junction(isoform: Dict[str, Any]) -> bool:
    """Return True if PTC > ~50-55 nt upstream of last exon junction. Placeholder."""
    return False
