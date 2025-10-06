"""
Phase 1: Structural fine-tuning for splice / TSS / polyA heads.

Usage:
  python -m betadogma.experiments.train --config betadogma/experiments/config/default.yaml
"""
from __future__ import annotations
import argparse
import os
from glob import glob
from typing import Dict, List

import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from betadogma.core.encoder_nt import NTEncoder
from betadogma.model import BetaDogmaModel
from betadogma.core.losses import structural_bce_ce_loss


# ---------------- Data ----------------

class StructuralDataset(Dataset):
    """
    Expects Parquet shards where each row has:
      - "seq": str DNA sequence (A/C/G/T/N)
      - "donor", "acceptor", "tss", "polya": lists/arrays of 0/1 labels (length Lr)
    """
    def __init__(self, parquet_paths: List[str], max_shards: int | None = None):
        self.paths = sorted(parquet_paths)[:max_shards] if max_shards else sorted(parquet_paths)
        self.rows = []
        for p in self.paths:
            df = pd.read_parquet(p)
            self.rows.extend(df.to_dict("records"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "seq": r["seq"],
            "donor": torch.as_tensor(r["donor"], dtype=torch.float32),
            "acceptor": torch.as_tensor(r["acceptor"], dtype=torch.float32),
            "tss": torch.as_tensor(r["tss"], dtype=torch.float32),
            "polya": torch.as_tensor(r["polya"], dtype=torch.float32),
        }


def collate_structural(batch):
    # Variable-length sequences allowed; encoder will pad at token level.
    seqs = [b["seq"] for b in batch]

    # For labels we’ll pad to the max label length in this mini-batch
    def pad1d(x, T):
        L = x.numel()
        if L < T:
            return torch.nn.functional.pad(x, (0, T - L))
        return x[:T]

    max_Lr = max(b["donor"].numel() for b in batch)
    donor   = torch.stack([pad1d(b["donor"],   max_Lr) for b in batch])  # (B, Lr)
    accept  = torch.stack([pad1d(b["acceptor"],max_Lr) for b in batch])
    tss     = torch.stack([pad1d(b["tss"],     max_Lr) for b in batch])
    polya   = torch.stack([pad1d(b["polya"],   max_Lr) for b in batch])

    return {"seqs": seqs, "donor": donor, "acceptor": accept, "tss": tss, "polya": polya}


# ---------------- Training ----------------

def _align_labels_to_L(labels_B_Lr: torch.Tensor, L: int) -> torch.Tensor:
    """Pad/crop labels along length dim to match encoder/heads length L."""
    B, Lr = labels_B_Lr.shape
    if Lr == L:
        return labels_B_Lr
    if Lr < L:
        return torch.nn.functional.pad(labels_B_Lr, (0, L - Lr))
    return labels_B_Lr[:, :L]


def train(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Encoder ---
    enc = NTEncoder(
        model_id=cfg["encoder"].get("model_id", "InstaDeepAI/nucleotide-transformer-500m-human-ref"),
        device=cfg["encoder"].get("device", "auto"),
    )

    # --- Model ---
    d_in = enc.hidden_size
    model = BetaDogmaModel(d_in=d_in, config=cfg).to(device)
    model.train()

    # --- Data ---
    shard_glob = os.path.join(cfg["data"]["out_cache"], "*.parquet")
    paths = glob(shard_glob)
    assert paths, f"No parquet shards found at {shard_glob}"

    ds = StructuralDataset(paths, max_shards=cfg["trainer"].get("max_shards"))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["trainer"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["trainer"].get("num_workers", 0)),
        collate_fn=collate_structural,
    )

    # --- Optimizer ---
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg["optimizer"]["lr"]),
                            weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)))

    # Optional loss weights
    w = {
        "donor": cfg["loss"].get("w_splice", 1.0),
        "acceptor": cfg["loss"].get("w_splice", 1.0),
        "tss": cfg["loss"].get("w_tss", 1.0),
        "polya": cfg["loss"].get("w_polya", 1.0),
        "orf_start": cfg["loss"].get("w_orf_start", 0.0),
        "orf_stop": cfg["loss"].get("w_orf_stop", 0.0),
        "orf_frame": cfg["loss"].get("w_orf_frame", 0.0),
    }

    epochs = int(cfg["trainer"]["epochs"])
    for ep in range(epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[Structural] Epoch {ep+1}/{epochs}")
        for batch in pbar:
            # Encode sequences -> embeddings + masks
            enc_out = enc.forward(batch["seqs"])
            embeddings = enc_out["embeddings"].to(device)   # (B, L, D)
            input_ids  = enc_out["input_ids"].to(device)    # (B, L)
            pad_mask   = enc_out["pad_mask"].to(device)     # (B, L)
            B, L, _ = embeddings.shape

            # Forward heads
            head_outs = model(embeddings=embeddings, input_ids=input_ids)

            # Align labels (B, Lr) to L
            donor   = _align_labels_to_L(batch["donor"].to(device),   L).unsqueeze(-1)  # (B,L,1)
            accept  = _align_labels_to_L(batch["acceptor"].to(device),L).unsqueeze(-1)
            tss     = _align_labels_to_L(batch["tss"].to(device),     L).unsqueeze(-1)
            polya   = _align_labels_to_L(batch["polya"].to(device),   L).unsqueeze(-1)

            labels = {
                "splice": {"donor": donor, "acceptor": accept},
                "tss":    {"tss": tss},
                "polya":  {"polya": polya},
                # No ORF supervision in this phase; skip keys to weight=0 losses
                "orf":    {"start": torch.zeros(B, L, 1, device=device),
                           "stop":  torch.zeros(B, L, 1, device=device),
                           "frame": torch.zeros(B, L, 3, device=device, dtype=torch.long)},
            }

            loss = structural_bce_ce_loss(head_outs, labels, pad_mask=pad_mask, weights=w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"].get("grad_clip", 1.0)))
            opt.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / max(1, len(dl))
        print(f"Epoch {ep+1}/{epochs}  avg_loss={avg:.4f}")

    # --- Save ---
    ckpt_dir = cfg["trainer"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "betadogma_structural.pt"))
    print(f"Saved checkpoint to {ckpt_dir}/betadogma_structural.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()