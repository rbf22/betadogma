
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# ============================================================================
# Model Heads
# ============================================================================

class PredictionHead(nn.Module):
    """Prediction head with optional gradient checkpointing."""

    def __init__(self, d_in: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.1, use_checkpointing: bool = False):
        super().__init__()

        # Force checkpointing on MPS for memory efficiency
        if torch.backends.mps.is_available():
            self.use_checkpointing = True
        else:
            self.use_checkpointing = use_checkpointing

        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward computation."""
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = cast(torch.Tensor, self.fc(lstm_out).squeeze(-1))  # Squeeze last dim: (B, L, 1) -> (B, L)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return cast(torch.Tensor, checkpoint(self._forward_impl, x, use_reentrant=False))
        else:
            return self._forward_impl(x)
