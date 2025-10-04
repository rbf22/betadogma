# betadogma/experiments/train.py
"""
Phase 1: Structural fine-tuning for splice/TSS/polyA heads.

Usage:
  python -m betadogma.experiments.train --config betadogma/experiments/config/default.yaml
"""
# betadogma/experiments/train.py
"""
Phase 1: Structural fine-tuning for splice/TSS/polyA heads.

Usage:
  python -m betadogma.experiments.train --config betadogma/experiments/config/default.yaml
"""
from __future__ import annotations
import argparse
import os
import yaml
from glob import glob
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from betadogma.model import BetaDogmaModel

# ---------------- Data ----------------

class StructuralDataset(Dataset):
    def __init__(self, parquet_paths: List[str], max_shards: int | None = None):
        self.paths = sorted(parquet_paths)[:max_shards] if max_shards else sorted(parquet_paths)
        self._rows = []
        for p in self.paths:
            df = pd.read_parquet(p)
            self._rows.extend(df.to_dict("records"))

    def __len__(self): return len(self._rows)

    def __getitem__(self, idx):
        r = self._rows[idx]
        seq = r["seq"]
        # Convert binned labels to tensors
        donor = torch.tensor(r["donor"], dtype=torch.float32)
        acceptor = torch.tensor(r["acceptor"], dtype=torch.float32)
        tss = torch.tensor(r["tss"], dtype=torch.float32)
        polya = torch.tensor(r["polya"], dtype=torch.float32)
        return {"seq": seq, "donor": donor, "acceptor": acceptor, "tss": tss, "polya": polya}

def collate(batch):
    # Pad to max label length (Lr) in batch
    max_label_len = max(len(x["donor"]) for x in batch)

    def pad1d(x, target_len):
        return torch.nn.functional.pad(x, (0, target_len - len(x)))

    seqs = [b["seq"] for b in batch]
    donor = torch.stack([pad1d(b["donor"], max_label_len) for b in batch])
    acceptor = torch.stack([pad1d(b["acceptor"], max_label_len) for b in batch])
    tss = torch.stack([pad1d(b["tss"], max_label_len) for b in batch])
    polya = torch.stack([pad1d(b["polya"], max_label_len) for b in batch])

    return {"seqs": seqs, "donor": donor, "acceptor": acceptor, "tss": tss, "polya": polya}


# --------------- Tokenization ----------------

def tokenize_dna(seqs: List[str]) -> torch.Tensor:
    """Converts a list of DNA sequences to a tensor of token indices."""
    token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    batch_tokens = []
    for seq in seqs:
        # upper() handles 'acgt' and 'N'
        tokens = [token_map.get(char, token_map['N']) for char in seq.upper()]
        batch_tokens.append(tokens)
    return torch.tensor(batch_tokens, dtype=torch.long)


# --------------- Training ----------------

def train(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize Model and Data ---
    # The model is now instantiated with the full config, ensuring all components
    # (encoder, heads, decoder) are correctly configured.
    model = BetaDogmaModel(config=cfg).to(device)

    shards_glob = os.path.join(cfg["data"]["out_cache"], "*.parquet")
    paths = glob(shards_glob)
    assert paths, f"No parquet shards found at {shards_glob}"

    ds = StructuralDataset(paths, max_shards=cfg["trainer"].get("max_shards"))
    dl = DataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True, collate_fn=collate)

    # --- Optimizer and Loss Function ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["loss"]["pos_weight"]).to(device))
    loss_weights = cfg["loss"]

    # --- Training Loop ---
    model.train()
    for epoch in range(cfg["trainer"]["epochs"]):
        total_loss = 0.0
        for batch in dl:
            opt.zero_grad()

            # Tokenize sequences using our simple DNA tokenizer
            # The sequences are expected to be of fixed length from the data prep script.
            input_ids = tokenize_dna(batch["seqs"]).to(device)

            # Forward pass through the unified model
            # The BetaDogmaEncoder does not use an attention mask.
            outputs = model(input_ids=input_ids)

            # --- Calculate Multi-Task Loss ---
            # Splice loss
            splice_logits_donor = outputs['splice']['donor'].squeeze(-1)
            splice_logits_acceptor = outputs['splice']['acceptor'].squeeze(-1)
            labels_donor = batch['donor'].to(device)
            labels_acceptor = batch['acceptor'].to(device)

            min_len = min(splice_logits_donor.shape[1], labels_donor.shape[1])
            loss_d = criterion(splice_logits_donor[:, :min_len], labels_donor[:, :min_len])
            loss_a = criterion(splice_logits_acceptor[:, :min_len], labels_acceptor[:, :min_len])
            splice_loss = (loss_d + loss_a) * loss_weights['w_splice']

            # TSS loss
            tss_logits = outputs['tss']['tss'].squeeze(-1)
            labels_tss = batch['tss'].to(device)
            min_len_tss = min(tss_logits.shape[1], labels_tss.shape[1])
            tss_loss = criterion(tss_logits[:, :min_len_tss], labels_tss[:, :min_len_tss]) * loss_weights['w_tss']

            # PolyA loss
            polya_logits = outputs['polya']['polya'].squeeze(-1)
            labels_polya = batch['polya'].to(device)
            min_len_polya = min(polya_logits.shape[1], labels_polya.shape[1])
            polya_loss = criterion(polya_logits[:, :min_len_polya], labels_polya[:, :min_len_polya]) * loss_weights['w_polya']

            # Combine losses
            loss = splice_loss + tss_loss + polya_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"].get("grad_clip", 1.0))
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dl))
        print(f"epoch {epoch+1}/{cfg['trainer']['epochs']}  loss={avg_loss:.4f}")

    # --- Save Checkpoint ---
    ckpt_dir = cfg["trainer"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "betadogma_structural.pt"))
    print("Saved checkpoint:", ckpt_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()