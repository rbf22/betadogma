"""
Loss functions for multi-task training:
- BCE for per-base binary maps (donor, acceptor, tss, polya, orf start/stop)
- CE for per-base multi-class maps (orf frame 0/1/2)
- KL divergence for PSI distributions across isoforms
- Optional consistency loss (rule vs learned NMD)

All losses support padding masks so PAD tokens don't contribute.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    logits, targets: (B, L, 1) or broadcastable
    pad_mask: (B, L) True for PAD
    """
    logits = logits.float()
    targets = targets.float()
    if pad_mask is not None:
        # expand to (B, L, 1)
        m = pad_mask.unsqueeze(-1).expand_as(logits)
        logits = logits.masked_select(~m)
        targets = targets.masked_select(~m)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    return loss


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    logits: (B, L, C)
    targets: (B, L) int64 class indices
    pad_mask: (B, L) True for PAD
    """
    B, L, C = logits.shape
    logits = logits.reshape(B * L, C).float()
    targets = targets.reshape(B * L).long()
    if pad_mask is not None:
        keep = ~pad_mask.reshape(B * L)
        logits = logits[keep]
        targets = targets[keep]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    return F.cross_entropy(logits, targets, reduction="mean")


def structural_bce_ce_loss(
    head_outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: Dict[str, Dict[str, torch.Tensor]],
    pad_mask: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute a weighted sum of per-base losses aligned with your current heads:

    Expected keys:
        head_outputs["splice"] -> {"donor": (B,L,1), "acceptor": (B,L,1)}
        head_outputs["tss"]    -> {"tss":   (B,L,1)}
        head_outputs["polya"]  -> {"polya": (B,L,1)}
        head_outputs["orf"]    -> {"start": (B,L,1), "stop": (B,L,1), "frame": (B,L,3)}

        labels mirrors the same structure; for "frame", labels["orf"]["frame"] is (B,L) int64 in {0,1,2}.

    weights: optional per-task weights (defaults in code).
    """
    w = {
        "donor": 1.0, "acceptor": 1.0,
        "tss": 1.0, "polya": 1.0,
        "orf_start": 0.5, "orf_stop": 0.5, "orf_frame": 0.5,
    }
    if weights:
        w.update(weights)

    loss = 0.0

    # Splice donor/acceptor (BCE)
    loss += w["donor"] * _masked_bce_with_logits(head_outputs["splice"]["donor"], labels["splice"]["donor"], pad_mask)
    loss += w["acceptor"] * _masked_bce_with_logits(head_outputs["splice"]["acceptor"], labels["splice"]["acceptor"], pad_mask)

    # TSS / PolyA (BCE)
    loss += w["tss"] * _masked_bce_with_logits(head_outputs["tss"]["tss"], labels["tss"]["tss"], pad_mask)
    loss += w["polya"] * _masked_bce_with_logits(head_outputs["polya"]["polya"], labels["polya"]["polya"], pad_mask)

    # ORF start/stop (BCE)
    loss += w["orf_start"] * _masked_bce_with_logits(head_outputs["orf"]["start"], labels["orf"]["start"], pad_mask)
    loss += w["orf_stop"]  * _masked_bce_with_logits(head_outputs["orf"]["stop"],  labels["orf"]["stop"],  pad_mask)

    # ORF frame (CE over 3 classes)
    loss += w["orf_frame"] * _masked_ce(head_outputs["orf"]["frame"], labels["orf"]["frame"], pad_mask)

    return loss


def psi_kl_loss(scores: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    KL divergence between predicted PSI (softmax over isoform scores) and target PSI.

    Args:
        scores: (B, N) raw isoform scores (logits)
        target_probs: (B, N) target PSI distribution (sum=1 per row)
    """
    log_q = F.log_softmax(scores, dim=-1)           # (B, N)
    p = (target_probs + epsilon) / (target_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # KL(P || Q) = sum P * (log P - log Q)
    kl = torch.sum(p * (torch.log(p + epsilon) - log_q), dim=-1)  # (B,)
    return kl.mean()


def nmd_consistency_loss(p_rule: torch.Tensor, p_learned: torch.Tensor) -> torch.Tensor:
    """
    Encourage agreement between a rule-derived NMD probability and the learned model's NMD p.
    Both tensors should be in [0,1], shape (B,) or broadcastable.

    Uses symmetric BCE: BCE(p_rule -> p_learned) + BCE(p_learned -> p_rule)
    """
    p_rule = p_rule.clamp(0.0, 1.0)
    p_learned = p_learned.clamp(0.0, 1.0)
    # Convert to logits safely
    def to_logit(p):
        eps = 1e-6
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))
    loss = F.binary_cross_entropy_with_logits(to_logit(p_learned), p_rule) + \
           F.binary_cross_entropy_with_logits(to_logit(p_rule), p_learned)
    return loss * 0.5