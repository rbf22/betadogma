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


def _masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Optional[torch.BoolTensor] = None,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional masking.
    
    Args:
        logits: (B, L, C) tensor of logits
        targets: (B, L) tensor of target class indices
        pad_mask: Optional (B, L) boolean tensor where True indicates padding
        weight: Optional (C,) tensor of class weights
        ignore_index: Target value to ignore (default: -100)
        
    Returns:
        Mean CE loss over non-padded elements
    """
    B, L, C = logits.shape
    logits = logits.reshape(B * L, C)
    targets = targets.reshape(B * L)
    
    if pad_mask is not None:
        keep = ~pad_mask.reshape(B * L)
        logits = logits[keep]
        targets = targets[keep]
    
    if logits.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
        
    return F.cross_entropy(
        logits, 
        targets, 
        weight=weight,
        ignore_index=ignore_index,
        reduction="mean"
    )


def _masked_bce_with_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    pad_mask: Optional[torch.BoolTensor] = None
) -> torch.Tensor:
    """
    Compute binary cross-entropy with optional masking.
    
    Args:
        logits: (B, L, 1) or broadcastable tensor of logits
        targets: (B, L, 1) or broadcastable tensor of target probabilities
        pad_mask: Optional (B, L) boolean tensor where True indicates padding
        
    Returns:
        Mean BCE loss over non-padded elements
    """
    B, L = logits.shape[0], logits.shape[1]
    logits = logits.reshape(B * L)
    targets = targets.reshape(B * L)
    if pad_mask is not None:
        keep = ~pad_mask.reshape(B * L)
        logits = logits[keep]
        targets = targets[keep]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

def structural_bce_ce_loss(
    head_outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: Dict[str, Dict[str, torch.Tensor]],
    pad_mask: Optional[torch.BoolTensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute a weighted sum of per-base losses for different genomic features.

    Expected keys in head_outputs and labels:
        - "splice": dict with "donor" and "acceptor" tensors (B, L, 1)
        - "polya": dict with "polya" tensor (B, L, 1)
        - "orf": dict with "start" (B, L, 1), "stop" (B, L, 1), "frame" (B, L, 3)

    For the "frame" task, labels["orf"]["frame"] should be (B, L) int64 in {0,1,2}.
    """
    # Initialize weights with default values
    w: Dict[str, float] = {
        "donor": 1.0, 
        "acceptor": 1.0,
        "tss": 1.0, 
        "polya": 1.0,
        "orf_start": 0.5, 
        "orf_stop": 0.5, 
        "orf_frame": 0.5,
    }
    if weights:
        w.update(weights)

    # Initialize loss on the correct device
    device = next(iter(head_outputs.values()))[next(iter(next(iter(head_outputs.values()))))].device
    loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Splice donor/acceptor (BCE)
    loss = loss + w["donor"] * _masked_bce_with_logits(
        head_outputs["splice"]["donor"], 
        labels["splice"]["donor"], 
        pad_mask
    )
    loss = loss + w["acceptor"] * _masked_bce_with_logits(
        head_outputs["splice"]["acceptor"], 
        labels["splice"]["acceptor"], 
        pad_mask
    )

    # TSS / PolyA (BCE)
    loss = loss + w["tss"] * _masked_bce_with_logits(
        head_outputs["tss"]["tss"], 
        labels["tss"]["tss"], 
        pad_mask
    )
    loss = loss + w["polya"] * _masked_bce_with_logits(
        head_outputs["polya"]["polya"], 
        labels["polya"]["polya"], 
        pad_mask
    )

    # ORF start/stop (BCE)
    loss = loss + w["orf_start"] * _masked_bce_with_logits(
        head_outputs["orf"]["start"], 
        labels["orf"]["start"], 
        pad_mask
    )
    loss = loss + w["orf_stop"] * _masked_bce_with_logits(
        head_outputs["orf"]["stop"],  
        labels["orf"]["stop"],  
        pad_mask
    )

    # ORF frame (CE over 3 classes)
    loss = loss + w["orf_frame"] * _masked_ce(
        head_outputs["orf"]["frame"], 
        labels["orf"]["frame"], 
    )

    return loss


def psi_kl_loss(
    scores: torch.Tensor, 
    target_probs: torch.Tensor, 
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute KL divergence between predicted and target PSI (Percent Spliced In) distributions.

    Args:
        scores: Tensor of shape (B, N) containing raw isoform scores (logits)
        target_probs: Tensor of shape (B, N) containing target PSI distributions (sum=1 per row)
        epsilon: Small constant for numerical stability
        
    Returns:
        torch.Tensor: KL divergence loss, averaged over the batch
    """
    log_q = F.log_softmax(scores, dim=-1)           # (B, N)
    p = (target_probs + epsilon) / (target_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # KL(P || Q) = sum P * (log P - log Q)
    kl = torch.sum(p * (torch.log(p + epsilon) - log_q), dim=-1)  # (B,)
    return kl.mean()


def nmd_consistency_loss(
    p_rule: torch.Tensor, 
    p_learned: torch.Tensor
) -> torch.Tensor:
    """
    Compute a symmetric consistency loss between rule-based and learned NMD probabilities.
    
    Uses symmetric binary cross-entropy: BCE(p_rule -> p_learned) + BCE(p_learned -> p_rule)
    
    Args:
        p_rule: Tensor of shape (B,) containing rule-based NMD probabilities in [0,1]
        p_learned: Tensor of shape (B,) containing learned NMD probabilities in [0,1]
        
    Returns:
        torch.Tensor: Symmetric consistency loss, averaged over the batch
    """
    p_rule = p_rule.clamp(0.0, 1.0)
    p_learned = p_learned.clamp(0.0, 1.0)
    # Convert to logits safely
    def to_logit(p: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))
    loss = F.binary_cross_entropy_with_logits(to_logit(p_learned), p_rule) + \
           F.binary_cross_entropy_with_logits(to_logit(p_rule), p_learned)
    return loss * 0.5