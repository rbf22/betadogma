import torch
import torch.nn as nn
from typing import Dict

class _ConvHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, out_ch: int, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.norm = nn.LayerNorm(d_in)
            self.net = nn.Sequential(
                nn.Conv1d(d_in, d_in, kernel_size=11, groups=d_in, padding=5),      # depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_in, kernel_size=5, groups=d_in, padding=4, dilation=2),  # dilated depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_hidden, kernel_size=1),  # pointwise
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
        # x: [B, Lr, D]
        if self.use_conv:
            x = self.norm(x)              # Apply norm on [B, Lr, D]
            x = x.transpose(1, 2)         # [B, D, Lr]
            y = self.net(x)               # [B, out_ch, Lr]
            y = y.transpose(1, 2)         # [B, Lr, out_ch]
        else:
            y = self.net(x)               # [B, Lr, out_ch]
        return y

class SpliceHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=2, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.proj(embeddings)    # [B, Lr, 2]
        return {"donor": logits[..., 0:1], "acceptor": logits[..., 1:2]}

class TSSHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"tss": self.proj(embeddings)}  # [B, Lr, 1]

class PolyAHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"polya": self.proj(embeddings)}  # [B, Lr, 1]

class ORFHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.start = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.stop  = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.frame = _ConvHead(d_in, d_hidden, out_ch=3, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "start": self.start(embeddings),  # [B, Lr, 1]
            "stop":  self.stop(embeddings),   # [B, Lr, 1]
            "frame": self.frame(embeddings),  # [B, Lr, 3]
        }