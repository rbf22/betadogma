"""
Phase 2a: Decoder fine-tuning for structural accuracy.

Trains the IsoformScorer with a listwise ranking loss.
Encoder + per-base heads are frozen; we feed mock head outputs so we can
exercise the decoder/scorer end-to-end without HF downloads.

Usage:
  python -m betadogma.experiments.train_decoder --config betadogma/experiments/config/default.yaml
"""
from __future__ import annotations
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from betadogma.model import BetaDogmaModel
from betadogma.decoder.types import Exon, Isoform


# ---------------------------
# Helpers / dataset
# ---------------------------

def are_isoforms_equal(a: Isoform, b: Isoform) -> bool:
    if a.strand != b.strand or len(a.exons) != len(b.exons):
        return False
    return all(x.start == y.start and x.end == y.end for x, y in zip(a.exons, b.exons))


class GroundTruthIsoformDataset(Dataset):
    """
    Minimal synthetic dataset that yields mock head outputs which
    strongly support a fixed ground-truth 2-exon isoform.
    """
    def __init__(self, num_samples: int = 32, L: int = 1024, strand: str = "+"):
        super().__init__()
        self.num_samples = num_samples
        self.L = L
        self.strand = strand

        # GT isoform: [100,200) + [300,400)
        self.true_iso = Isoform(
            exons=[Exon(100, 200), Exon(300, 400)],
            strand=strand,
            score=0.0,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> Dict:
        L = self.L
        # Strong logits at GT splice sites; low elsewhere
        donor = torch.full((L,), -8.0)
        accept = torch.full((L,), -8.0)
        donor[200] = 6.0
        accept[100] = 6.0
        donor[400] = 5.0
        accept[300] = 5.0
        # Some distractors
        donor[250] = 3.5
        accept[150] = 3.0

        # TSS and PolyA weak but centered at isoform boundaries
        tss = torch.full((L,), -6.0); tss[100] = 2.0
        polya = torch.full((L,), -6.0); polya[400] = 2.0

        # ORF maps optional (decoder scorer can work without them);
        # provide zeros so shapes are correct if enabled later
        start = torch.full((L,), -8.0)
        stop  = torch.full((L,), -8.0)
        frame = torch.zeros(L, 3)

        head_outputs = {
            "splice": {"donor": donor.view(1, L, 1), "acceptor": accept.view(1, L, 1)},
            "tss":    {"tss": tss.view(1, L, 1)},
            "polya":  {"polya": polya.view(1, L, 1)},
            "orf":    {"start": start.view(1, L, 1), "stop": stop.view(1, L, 1), "frame": frame.view(1, L, 3)},
            # passthroughs the decoder may look for (not used in this synthetic task)
            "embeddings": torch.zeros(1, L, 8),
            "input_ids":  torch.full((1, L), 4, dtype=torch.long),  # all N's
        }
        return {"head_outputs": head_outputs, "true_isoform": self.true_iso, "strand": self.strand}


def collate_keep_first(batch):
    # Train on single examples; return the first item
    return batch[0]


# ---------------------------
# Loss
# ---------------------------

def listwise_hinge_loss(pos: torch.Tensor, negs: List[torch.Tensor], margin: float = 0.1) -> torch.Tensor:
    if not negs:
        return pos.new_tensor(0.0)
    diffs = pos - torch.stack(negs)
    return torch.clamp(margin - diffs, min=0).mean()


# ---------------------------
# Training
# ---------------------------

def train_decoder(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal model just to access the decoder/scorer (encoder/heads are irrelevant here)
    d_in = int(cfg.get("encoder", {}).get("hidden_size", 256))
    model = BetaDogmaModel(d_in=d_in, config=cfg).to(device)

    # Freeze everything except the scorer
    for p in model.parameters():
        p.requires_grad = False
    for p in model.isoform_decoder.scorer.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.isoform_decoder.scorer.parameters(), lr=float(cfg["optimizer"]["lr"]))
    ds = GroundTruthIsoformDataset(
        num_samples=int(cfg["trainer"].get("num_samples", 32)),
        L=int(cfg["trainer"].get("L", 1024)),
        strand=cfg.get("data", {}).get("strand", "+"),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_keep_first)

    model.isoform_decoder.scorer.train()
    epochs = int(cfg["trainer"]["epochs"])
    for epoch in range(epochs):
        total = 0.0
        steps = 0
        for batch in dl:
            head_outputs = {k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else (v.to(device) if torch.is_tensor(v) else v)
                            for k, v in batch["head_outputs"].items()}

            # Decode candidates
            cands = model.isoform_decoder.decode(head_outputs, strand=batch["strand"])

            # Split pos/neg relative to GT
            gt = batch["true_isoform"]
            pos = None; negs = []
            for c in cands:
                (pos := c) if are_isoforms_equal(c, gt) else negs.append(c)

            # Skip if GT not in candidates
            if pos is None:
                continue

            # Score with the learnable scorer
            pos_s = model.isoform_decoder.scorer(pos, head_outputs)
            neg_s = [model.isoform_decoder.scorer(n, head_outputs) for n in negs]

            loss = listwise_hinge_loss(pos_s, neg_s, margin=float(cfg["loss"].get("margin", 0.1)))
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()); steps += 1

        avg = total / max(1, steps)
        print(f"[Decoder] Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    print("Decoder fine-tuning complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_decoder(cfg)


if __name__ == "__main__":
    main()