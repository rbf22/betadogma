# src/betadogma/nmd/nmd_model.py
"""
NMD predictor

Supports three modes:
  - "rule"   : classic 55-nt rule (binary 0/1) — default (preserves old behavior)
  - "learned": tiny MLP over simple transcript features -> probability
  - "hybrid" : blend of soft rule signal and learned head

API:
    nmd = NMDModel(config)
    p = nmd.predict(isoform)                   # float in [0,1]
    info = nmd.predict_with_details(isoform)  # dict with p, rule, distance, features
"""

from __future__ import annotations
from typing import Dict, Any, Literal, Optional, TypedDict, Union, final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nmd_rules import (
    rule_ptc_before_last_junction,
    ptc_distance_to_last_junction_tx,
    DEFAULT_PTC_THRESHOLD,
)
from ..decoder.types import Isoform

# Type definitions
NMDMode = Literal["rule", "learned", "hybrid"]

class NMDConfig(TypedDict, total=False):
    """Configuration dictionary for NMDModel.
    
    All fields are optional with the following defaults:
    - mode: 'rule' | 'learned' | 'hybrid' = 'rule'
    - ptc_rule_threshold: int = 55
    - cds_inclusive: bool = False
    - strict_gt: bool = True
    - alpha: float = 0.5
    - tau: float = 10.0
    - hidden: int = 16
    - init_bias: float = -1.5
    """
    mode: NMDMode
    ptc_rule_threshold: int
    cds_inclusive: bool
    strict_gt: bool
    alpha: float
    tau: float
    hidden: int
    init_bias: float

# Default configuration
DEFAULT_CONFIG: NMDConfig = {
    "mode": "rule",
    "ptc_rule_threshold": DEFAULT_PTC_THRESHOLD,
    "cds_inclusive": False,
    "strict_gt": True,
    "alpha": 0.5,
    "tau": 10.0,
    "hidden": 16,
    "init_bias": -1.5,
}


class NMDModel(nn.Module):
    """Nonsense-mediated decay (NMD) predictor.
    
    Supports three prediction modes:
    - "rule": Classic 55-nt rule (binary 0/1)
    - "learned": MLP over transcript features -> probability
    - "hybrid": Blend of soft rule signal and learned prediction
    
    Attributes:
        mode: Current prediction mode ("rule", "learned", or "hybrid")
        ptc_rule_threshold: Distance threshold for NMD (default: 55)
        cds_inclusive: If True, CDS coordinates are treated as inclusive
        strict_gt: If True, use '>' for threshold comparison
        alpha: Blend weight for rule in hybrid mode (0=learned only, 1=rule only)
        tau: Softness parameter for sigmoid in nt
        mlp: Neural network for learned predictions
    """
    
    def __init__(self, config: NMDConfig | None = None) -> None:
        """Initialize NMD predictor with the given configuration.
        
        Args:
            config: Optional configuration dictionary. See NMDConfig for available options.
        """
        super().__init__()
        # Merge with defaults
        cfg: NMDConfig = {**DEFAULT_CONFIG, **(config or {})}
        
        # Validate mode
        self.mode: NMDMode = str(cfg["mode"]).lower()  # type: ignore
        if self.mode not in ("rule", "learned", "hybrid"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of: rule, learned, hybrid")
            
        self.ptc_rule_threshold: int = int(cfg["ptc_rule_threshold"])
        self.cds_inclusive: bool = bool(cfg["cds_inclusive"])
        self.strict_gt: bool = bool(cfg["strict_gt"])
        self.alpha: float = float(cfg["alpha"])
        self.tau: float = float(cfg["tau"])
        
        # Initialize MLP for learned predictions
        hidden_size: int = int(cfg["hidden"])
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_size),  # 5 simple features (see _features)
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Initialize bias to predict non-NMD by default
        init_bias: float = float(cfg["init_bias"])
        with torch.no_grad():
            if isinstance(self.mlp[-1], nn.Linear):
                self.mlp[-1].bias.fill_(init_bias)

    # ---------- feature engineering (no CDS re-decoding required)

    def _soft_rule_prob(self, iso: Isoform) -> float:
        """Compute soft probability using a sigmoid over the PTC-junction distance.
        
        The probability is computed as:
            p = sigmoid((distance - threshold) / tau)
            
        where distance = last_junction_tx - ptc_tx (positive when PTC is upstream)
        
        Args:
            iso: Isoform to evaluate
            
        Returns:
            Probability in [0, 1], or 0.0 if distance is unavailable
        """
        dist = ptc_distance_to_last_junction_tx(iso, cds_inclusive=self.cds_inclusive)
        if dist is None:
            return 0.0
            
        # Compute sigmoid with numerical stability
        x = (dist - self.ptc_rule_threshold) / max(1e-6, self.tau)
        return float(torch.sigmoid(torch.tensor(x)).item())

    def _features(self, iso: Isoform) -> torch.Tensor:
        """Extract features from an isoform for NMD prediction.
        
        Features:
            x0: Number of exons
            x1: Transcript length in kilobases
            x2: Binary indicator for multi-exon transcript
            x3: Fraction of transcript length in last exon
            x4: Soft rule probability
            
        Args:
            iso: Isoform to extract features from
            
        Returns:
            Tensor of shape (5,) containing the extracted features
        """
        # Calculate basic features
        n_exons = len(iso.exons)
        total_length = float(sum(e.end - e.start for e in iso.exons)) if iso.exons else 0.0
        
        # Calculate last exon features
        last_exon_length = float(iso.exons[-1].end - iso.exons[-1].start) if iso.exons else 0.0
        has_multiple_exons = 1.0 if n_exons > 1 else 0.0
        last_exon_frac = (last_exon_length / total_length) if total_length > 0 else 0.0
        
        # Get soft rule probability
        soft_rule = self._soft_rule_prob(iso)
        
        # Create feature vector
        return torch.tensor([
            float(n_exons),
            total_length / 1000.0,
            has_multiple_exons,
            last_exon_frac,
            soft_rule
        ], dtype=torch.float32)

    # ---------- public API

    @torch.no_grad()
    def predict(self, isoform: Isoform) -> float:
        """Predict NMD probability for a given isoform.
        
        The prediction mode is determined by self.mode:
        - "rule": Returns 0.0 or 1.0 based on the hard 55-nt rule
        - "learned": Returns MLP prediction in [0, 1]
        - "hybrid": Returns weighted average of soft rule and MLP prediction
        
        Args:
            isoform: Input isoform to predict NMD for
            
        Returns:
            Probability of NMD in the range [0, 1]
        """
        # Rule-based prediction
        if self.mode == "rule":
            is_nmd = rule_ptc_before_last_junction(
                isoform,
                ptc_rule_threshold=self.ptc_rule_threshold,
                cds_inclusive=self.cds_inclusive,
                strict_gt=self.strict_gt,
            )
            return 1.0 if is_nmd else 0.0
        
        # Extract features and get MLP prediction
        feats = self._features(isoform)
        learned_logits: torch.Tensor = self.mlp(feats)
        learned_p: float = float(torch.sigmoid(learned_logits).item())
        
        # Return based on mode
        if self.mode == "learned":
            return learned_p
            
        # Hybrid mode: blend soft rule with learned prediction
        soft_rule = self._soft_rule_prob(isoform)
        return float(self.alpha * soft_rule + (1.0 - self.alpha) * learned_p)

    @torch.no_grad()
    def predict_with_details(self, isoform: Isoform) -> Dict[str, float | str]:
        """Predict NMD probability with detailed debugging information.
        
        Returns:
            Dictionary containing:
            - p: Final probability in [0,1] (same as predict())
            - rule_hard: 0/1 from classic rule (0 if distance unavailable)
            - rule_soft: sigmoid((dist - threshold)/tau) in [0,1]
            - distance: Signed distance in nt (NaN if unavailable)
            - mode: The prediction mode used ("rule", "learned", or "hybrid")
        """
        # Calculate distance and rule-based predictions
        dist = ptc_distance_to_last_junction_tx(isoform, cds_inclusive=self.cds_inclusive)
        
        if dist is None:
            rule_hard = 0.0
            rule_soft = 0.0
            dist_val = float("nan")
        else:
            # Calculate hard rule prediction
            if self.strict_gt:
                rule_hard = 1.0 if dist > self.ptc_rule_threshold else 0.0
            else:
                rule_hard = 1.0 if dist >= self.ptc_rule_threshold else 0.0
                
            # Calculate soft rule probability
            rule_soft = float(torch.sigmoid(
                torch.tensor((dist - self.ptc_rule_threshold) / max(1e-6, self.tau))
            ).item())
            dist_val = float(dist)
        
        # Get MLP prediction if needed
        if self.mode != "rule":
            feats = self._features(isoform)
            learned_p = float(torch.sigmoid(self.mlp(feats)).item())
        
        # Determine final prediction based on mode
        if self.mode == "rule":
            p = rule_hard
        elif self.mode == "learned":
            p = learned_p
        else:  # hybrid
            p = float(self.alpha * rule_soft + (1.0 - self.alpha) * learned_p)
        
        return {
            "p": p,
            "rule_hard": rule_hard,
            "rule_soft": rule_soft,
            "distance": dist_val,
            "mode": self.mode,
        }