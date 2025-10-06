"""
Learned NMD classifier using features derived from transcript structure.
This version uses a simple rule-based approach as a baseline.
"""
from typing import Dict, Any

from ..decoder.types import Isoform
from .nmd_rules import rule_ptc_before_last_junction


class NMDModel:
    """
    A simple rule-based model for predicting nonsense-mediated decay (NMD).
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Allow overriding the NMD rule threshold from the main model config
        self.ptc_rule_threshold = self.config.get("ptc_rule_threshold", 55)

    def predict(self, isoform: Isoform) -> float:
        """
        Predicts the probability of an isoform undergoing NMD.

        For this rule-based model, the probability is binary (0.0 or 1.0).

        Args:
            isoform: The isoform to be evaluated.

        Returns:
            A probability (float) indicating the likelihood of NMD.
        """
        is_nmd_target = rule_ptc_before_last_junction(
            isoform,
            ptc_rule_threshold=self.ptc_rule_threshold
        )
        return 1.0 if is_nmd_target else 0.0