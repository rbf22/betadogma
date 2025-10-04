"""
Isoform decoder: builds exon chains from splice/TSS/polyA signals.
Also computes ψ (relative isoform usage) via a scoring head.
"""
from typing import Any, Dict, List

class IsoformDecoder:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def build_splice_graph(self, heads_outputs: Dict[str, Any]) -> Any:
        """Construct a splice graph from per-base head outputs. Placeholder."""
        return {}

    def decode_isoforms(self, graph: Any) -> List[Dict[str, Any]]:
        """Enumerate candidate isoforms (exon chains). Placeholder."""
        return []

    def score_abundance(self, isoforms: list) -> Dict[str, float]:
        """Return ψ per isoform (softmax scores). Placeholder."""
        return {}
