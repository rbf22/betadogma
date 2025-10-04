"""
Encode variants (SNP/indel) into an auxiliary input channel aligned with the sequence window.
"""
from typing import Any, Dict

def encode_variant(spec: str, window=None) -> Dict[str, Any]:
    """Parse 'chr:posREF>ALT' into structured format and alignment. Placeholder."""
    return {"spec": spec}
