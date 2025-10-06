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

from typing import Dict
import torch
import torch.nn as nn


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
            self.norm = nn.LayerNorm(d_in)
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
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.use_conv:
            x = self.norm(x)          # (B, L, D)
            x = x.transpose(1, 2)     # (B, D, L)
            y = self.net(x)           # (B, out_ch, L)
            y = y.transpose(1, 2)     # (B, L, out_ch)
        else:
            y = self.net(x)           # (B, L, out_ch)
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