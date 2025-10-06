# SPDX-License-Identifier: MIT
# src/betadogma/core/encoder_nt.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"


class NTEncoder:
    """
    Transformers-native nucleotide encoder.

    Returns base-resolution embeddings (bin_size=1) that align with your heads.
    Also returns a clean base-level input_ids tensor for downstream ORF scoring.

    Notes:
      - Many nucleotide tokenizers add BOS/EOS special tokens. We strip them by
        slicing [:, 1:-1] on the model output and tokenizer input_ids.
      - If you pass sequences of unequal lengths, they will be right-padded.
        We return a boolean pad_mask with True at padding positions.
    """

    def __init__(self, model_id: str = DEFAULT_NT, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.bin_size = 1

        # Map A/C/G/T/N to small ints for decoder ORF scoring
        self.base_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    @torch.no_grad()
    def forward(self, seqs: List[str]) -> Dict[str, Any]:
        """
        Args:
            seqs: list of DNA strings (A/C/G/T/N), variable length allowed.

        Returns:
            {
                "embeddings": (B, L, D) float32,
                "input_ids": (B, L) int64 in {0..4} mapping A/C/G/T/N,
                "pad_mask": (B, L) bool, True for PAD positions,
                "lengths": List[int] true (unpadded) lengths per sequence
            }
        """
        # Tokenize with specials so the model is happy
        toks = self.tok(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(self.device)

        out = self.model(**toks)  # last_hidden_state: (B, L_tok, D)

        # Most HF nucleotide models place one BOS at position 0 and one EOS at last.
        # We remove them to get base-aligned embeddings.
        emb_tok = out.last_hidden_state  # (B, L_tok, D)
        emb = emb_tok[:, 1:-1, :]        # (B, L, D) base-only

        # Strip specials from tokenizer IDs and attn mask to compute lengths/padding
        input_ids_tok = toks["input_ids"][:, 1:-1]       # (B, L)
        attn_tok = toks["attention_mask"][:, 1:-1]       # (B, L), 1 for real tokens

        # Build a clean base-level input_ids in {0..4} from original seq text
        # (safer than trying to invert tokenizer vocab)
        base_ids = []
        lengths = []
        max_L = attn_tok.size(1)
        for s in seqs:
            s_up = s.upper()
            ids = [self.base_vocab.get(ch, 4) for ch in s_up]  # map unknown to N=4
            lengths.append(len(ids))
            if len(ids) < max_L:
                ids = ids + [4] * (max_L - len(ids))  # pad with N
            else:
                ids = ids[:max_L]
            base_ids.append(ids)
        base_ids = torch.tensor(base_ids, dtype=torch.long, device=self.device)  # (B, L)

        # Pad mask: True at padded positions (i.e., where attention_mask==0)
        pad_mask = attn_tok == 0  # (B, L) bool

        return {
            "embeddings": emb,          # (B, L, D)
            "input_ids": base_ids,      # (B, L) ints 0..4
            "pad_mask": pad_mask.bool(),# (B, L)
            "lengths": lengths,         # list[int]
        }