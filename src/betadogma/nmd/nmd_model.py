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
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .nmd_rules import (
    rule_ptc_before_last_junction,
    ptc_distance_to_last_junction_tx,
)
from ..decoder.types import Isoform


class NMDModel(nn.Module):
    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Config options (all optional):
          mode: "rule" | "learned" | "hybrid"  (default: "rule")
          ptc_rule_threshold: int (default: 55)
          cds_inclusive: bool    (default: False)  # set True if CDS is inclusive
          strict_gt: bool        (default: True)   # '>' vs '>=' in the rule
          # hybrid settings
          alpha: float           (default: 0.5)    # blend weight for rule in hybrid
          tau: float             (default: 10.0)   # softness (nt) for the rule sigmoid
          # learned head
          hidden: int            (default: 16)
          init_bias: float       (default: -1.5)   # bias toward non-NMD before training
        """
        super().__init__()
        cfg = config or {}
        self.mode = str(cfg.get("mode", "rule")).lower()
        self.ptc_rule_threshold = int(cfg.get("ptc_rule_threshold", 55))
        self.cds_inclusive = bool(cfg.get("cds_inclusive", False))
        self.strict_gt = bool(cfg.get("strict_gt", True))

        # Hybrid softness/weight
        self.alpha = float(cfg.get("alpha", 0.5))
        self.tau = float(cfg.get("tau", 10.0))

        # Tiny learned head
        hidden = int(cfg.get("hidden", 16))
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden),  # 5 simple features (see _features)
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        init_bias = float(cfg.get("init_bias", -1.5))
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    # ---------- feature engineering (no CDS re-decoding required)

    def _soft_rule_prob(self, iso: Isoform) -> float:
        """
        Smooth version of the 55-nt rule using a logistic over signed distance:
          distance = last_junction_tx - ptc_tx  (positive = upstream of junction)
          soft_p = sigmoid((distance - threshold) / tau)
        Falls back to 0.0 if distance is unavailable (single-exon / no CDS).
        """
        dist = ptc_distance_to_last_junction_tx(iso, cds_inclusive=self.cds_inclusive)
        if dist is None:
            return 0.0
        return float(torch.sigmoid(torch.tensor((dist - self.ptc_rule_threshold) / max(1e-6, self.tau))).item())

    def _features(self, iso: Isoform) -> torch.Tensor:
        """
        Construct a tiny feature vector:
            x0 = exon_count
            x1 = transcript_length_kb
            x2 = has_multiple_exons (0/1)
            x3 = last_exon_frac (last_exon_len / total_len)
            x4 = soft_rule_prob (smooth distance-based rule)
        """
        n_ex = float(len(iso.exons))
        total = float(sum(e.end - e.start for e in iso.exons)) if iso.exons else 0.0
        last_len = float(iso.exons[-1].end - iso.exons[-1].start) if iso.exons else 0.0
        multi = 1.0 if n_ex > 1 else 0.0
        last_frac = (last_len / total) if total > 0 else 0.0
        soft_rule = self._soft_rule_prob(iso)
        vec = torch.tensor([n_ex, total / 1000.0, multi, last_frac, soft_rule], dtype=torch.float32)
        return vec

    # ---------- public API

    @torch.no_grad()
    def predict(self, isoform: Isoform) -> float:
        """
        Return p(NMD) in [0,1] according to the configured mode.
        - "rule"   : returns 0.0/1.0 from the hard rule
        - "learned": returns sigmoid(MLP(features))
        - "hybrid" : alpha * soft_rule + (1-alpha) * learned_prob
        """
        if self.mode == "rule":
            is_nmd = rule_ptc_before_last_junction(
                isoform,
                ptc_rule_threshold=self.ptc_rule_threshold,
                cds_inclusive=self.cds_inclusive,
                strict_gt=self.strict_gt,
            )
            return 1.0 if is_nmd else 0.0

        feats = self._features(isoform)
        learned_p = float(torch.sigmoid(self.mlp(feats)).item())

        if self.mode == "learned":
            return learned_p

        # hybrid
        soft_rule = self._soft_rule_prob(isoform)
        return float(self.alpha * soft_rule + (1.0 - self.alpha) * learned_p)

    @torch.no_grad()
    def predict_with_details(self, isoform: Isoform) -> Dict[str, float]:
        """
        Return a dict with extra introspection for debugging/metrics.
        Keys:
            p          : final probability in [0,1]
            rule_hard  : 0/1 from classic rule (or 0 if distance unavailable)
            rule_soft  : sigmoid((dist - threshold)/tau) in [0,1]
            distance   : signed distance in nt (None -> NaN)
        """
        dist = ptc_distance_to_last_junction_tx(isoform, cds_inclusive=self.cds_inclusive)
        if dist is None:
            rule_hard = 0.0
            rule_soft = 0.0
            dist_val = float("nan")
        else:
            rule_hard = 1.0 if (dist > self.ptc_rule_threshold if self.strict_gt else dist >= self.ptc_rule_threshold) else 0.0
            rule_soft = float(torch.sigmoid(torch.tensor((dist - self.ptc_rule_threshold) / max(1e-6, self.tau))).item())
            dist_val = float(dist)

        feats = self._features(isoform)
        learned_p = float(torch.sigmoid(self.mlp(feats)).item())

        if self.mode == "rule":
            p = rule_hard
        elif self.mode == "learned":
            p = learned_p
        else:
            p = float(self.alpha * rule_soft + (1.0 - self.alpha) * learned_p)

        return {"p": p, "rule_hard": rule_hard, "rule_soft": rule_soft, "distance": dist_val}