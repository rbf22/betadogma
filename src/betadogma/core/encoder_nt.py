# SPDX-License-Identifier: MIT
# src/betadogma/core/encoder_nt.py
from __future__ import annotations
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

class NTEncoder:
    """Transformers-native nucleotide encoder. Outputs [B, L, D] at base resolution (bin_size=1)."""
    def __init__(self, model_id: str = DEFAULT_NT, device: str = "auto"):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.bin_size = 1

    @torch.no_grad()
    def forward(self, seqs: list[str]) -> torch.Tensor:
        toks = self.tok(seqs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model(**toks)
        return out.last_hidden_state  # [B, L, D]