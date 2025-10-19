"""
core/heads.py
--------------
Per-base prediction heads used by BetaDogmaModel.

Each head maps encoder embeddings (B, L, D) -> logits with a small MLP,
optionally preceded by a lightweight depthwise/pointwise Conv1d stack
for local context.

Heads API (what BetaDogmaModel/decoder expect):
- SpliceHead.forward -> {"donor": (B,L,1), "acceptor": (B,L,1)}
- TSSHead.forward   -> {"tss":    (B,L,1)}
- PolyAHead.forward -> {"polya":  (B,L,1)}
- ORFHead.forward   -> {"start": (B,L,1), "stop": (B,L,1), "frame": (B,L,3)}
"""

from __future__ import annotations

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.init as init

# Import logger from train module
try:
    from ...train.train import logger
except ImportError:
    # Fallback logger if import fails
    logger = logging.getLogger("betadogma_heads")


class StableLayerNorm(nn.LayerNorm):
    """LayerNorm with enhanced numerical stability."""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.stability_eps = 1e-6  # Additional epsilon for extreme cases
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check for NaN/inf in input
        if torch.isnan(input).any() or torch.isinf(input).any():
            logger.warning("NaN/Inf in LayerNorm input - replacing with zeros")
            return torch.zeros_like(input)
            
        # Compute mean and variance with stability
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        
        # Debug: Check computed statistics
        if torch.isnan(mean).any() or torch.isnan(var).any():
            logger.debug(f"LayerNorm stats: mean={mean}, var={var}")
            logger.debug(f"LayerNorm input range: [{input.min().item():.4f}, {input.max().item():.4f}]")
            logger.debug(f"LayerNorm input mean: {input.mean().item():.4f}")
        
        # Add stability epsilon to prevent division by zero
        var = var + self.stability_eps
        
        # Check for NaN in computed statistics
        if torch.isnan(mean).any() or torch.isnan(var).any():
            logger.warning("NaN in LayerNorm statistics - using identity")
            return input
            
        # Normalize
        input = (input - mean) / torch.sqrt(var)
        
        # Scale and shift if affine
        if self.elementwise_affine:
            input = input * self.weight + self.bias
            
        return input


class _ConvHead(nn.Module):
    """
    A flexible head block.
    If use_conv=True: LN -> depthwise conv -> GELU -> dilated depthwise -> GELU
                      -> pointwise -> GELU -> Dropout -> pointwise(out_ch)
    Else:             LN -> Linear -> GELU -> Dropout -> Linear(out_ch)
    """
    def __init__(self, d_in: int, d_hidden: int, out_ch: int, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.norm = StableLayerNorm(d_in)  # Use stable LayerNorm
            self.net = nn.Sequential(
                nn.Conv1d(d_in, d_in, kernel_size=11, groups=d_in, padding=5),               # depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_in, kernel_size=5, groups=d_in, padding=4, dilation=2),    # dilated depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_hidden, kernel_size=1),                                    # pointwise
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_hidden, out_ch, kernel_size=1),
            )
        else:
            self.net = nn.Sequential(
                StableLayerNorm(d_in),  # Use stable LayerNorm
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, out_ch),
            )

        # Initialize weights properly to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values to prevent NaN."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # Use smaller initialization for better numerical stability
                init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    init.zeros_(module.bias)
                logger.debug(f"Initialized {type(module).__name__} with std=0.02")
            elif isinstance(module, (nn.LayerNorm, StableLayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.zeros_(module.bias)
                logger.debug(f"Initialized {type(module).__name__} with ones/zeros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.use_conv:
            # Check for NaN in input before LayerNorm
            if torch.isnan(x).any():
                logger.error("NaN in input to LayerNorm!")
                return torch.zeros_like(x)
            
            x = self.norm(x)          # (B, L, D) - StableLayerNorm handles NaN internally
            
            # Check for NaN after LayerNorm
            if torch.isnan(x).any():
                logger.error("NaN after StableLayerNorm in _ConvHead!")
                logger.debug(f"Post-LayerNorm range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                logger.debug(f"Post-LayerNorm mean: {x.mean().item():.4f}")
                x = torch.zeros_like(x)
            
            x = x.transpose(1, 2)     # (B, D, L)
            if torch.isnan(x).any():
                logger.error("NaN after transpose in _ConvHead!")
                x = torch.zeros_like(x)
                
            y = self.net(x)           # (B, out_ch, L)
            if torch.isnan(y).any():
                logger.error("NaN after self.net in _ConvHead!")
                y = torch.zeros_like(y)
                
            y = y.transpose(1, 2)     # (B, L, out_ch)
        else:
            # Check for NaN in input to non-conv path
            if torch.isnan(x).any():
                logger.error("NaN in input to non-conv _ConvHead!")
                return torch.zeros_like(x)
                
            y = self.net(x)           # (B, L, out_ch)
            
            # Check for NaN in non-conv output
            if torch.isnan(y).any():
                logger.error("NaN in non-conv output of _ConvHead!")
                y = torch.zeros_like(y)
        
        # Final check for NaN in output
        if torch.isnan(y).any():
            logger.error("NaN in final output of _ConvHead!")
            y = torch.zeros_like(y)
            
        return y


class SpliceHead(nn.Module):
    """Produces donor/acceptor logits as two (B,L,1) maps."""
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=2, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.proj(embeddings)                 # (B, L, 2)
        return {"donor": logits[..., 0:1], "acceptor": logits[..., 1:2]}


class TSSHead(nn.Module):
    """Produces a single (B,L,1) TSS logit map."""
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"tss": self.proj(embeddings)}          # (B, L, 1)


class PolyAHead(nn.Module):
    """Produces a single (B,L,1) polyA logit map."""
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"polya": self.proj(embeddings)}        # (B, L, 1)


class ORFHead(nn.Module):
    """
    Produces three maps:
      - start: (B,L,1) start-codon evidence
      - stop:  (B,L,1) stop-codon evidence
      - frame: (B,L,3) in-frame channel probs/logits for frames 0/1/2
    """
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.start = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.stop  = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.frame = _ConvHead(d_in, d_hidden, out_ch=3, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "start": self.start(embeddings),  # (B, L, 1)
            "stop":  self.stop(embeddings),   # (B, L, 1)
            "frame": self.frame(embeddings),  # (B, L, 3)
        }