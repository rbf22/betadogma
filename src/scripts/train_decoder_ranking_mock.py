# scripts/train_decoder_ranking_mock.py
import argparse
import yaml
from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from betadogma.model import BetaDogmaModel
from betadogma.decoder import Isoform, Exon

class PairwiseHingeLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    def forward(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.margin - (pos - neg)).mean()

def get_mock_batch(device: torch.device, seq_len: int = 2048) -> Dict[str, Any]:
    donor = torch.full((1, seq_len), -10.0, device=device)
    acceptor = torch.full((1, seq_len), -10.0, device=device)
    tss = torch.full((1, seq_len), -10.0, device=device)
    polya = torch.full((1, seq_len), -10.0, device=device)
    orf_start = torch.full((1, seq_len), -10.0, device=device)
    orf_stop  = torch.full((1, seq_len), -10.0, device=device)
    orf_frame = torch.randn(1, seq_len, 3, device=device)

    tss[0, 50] = 5.0
    donor[0, 200] = 5.0
    acceptor[0, 300] = 5.0
    polya[0, 500] = 5.0
    acceptor[0, 100] = 2.0
    donor[0, 400] = 2.0

    head_outputs = {
        "splice": {"donor": donor.unsqueeze(-1), "acceptor": acceptor.unsqueeze(-1)},
        "tss": {"tss": tss.unsqueeze(-1)},
        "polya": {"polya": polya.unsqueeze(-1)},
        "orf": {"start": orf_start.unsqueeze(-1), "stop": orf_stop.unsqueeze(-1), "frame": orf_frame},
    }
    true_isoform = Isoform(exons=[Exon(50, 200), Exon(300, 500)], strand="+")
    input_ids = torch.randint(0, 5, (1, seq_len), device=device)
    return {"head_outputs": head_outputs, "true_isoform": true_isoform, "strand": "+", "input_ids": input_ids}

def select_candidates(cands: List[Isoform], true_iso: Isoform):
    tp = tuple((e.start, e.end) for e in true_iso.exons)
    pos, neg = [], []
    for c in cands:
        pairs = tuple((e.start, e.end) for e in c.exons)
        (pos if pairs == tp else neg).append(c)
    return pos, neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/betadogma/experiments/config/default.yaml")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    d_in = int(cfg["encoder"]["hidden_size"])
    model = BetaDogmaModel(d_in=d_in, config=cfg).to(device)

    # train only scorer
    for p in model.parameters():
        p.requires_grad = False
    for p in model.isoform_decoder.scorer.parameters():
        p.requires_grad = True

    opt = optim.Adam(model.isoform_decoder.scorer.parameters(), lr=cfg["optimizer"]["lr"])
    loss_fn = PairwiseHingeLoss(0.5)

    print("Training decoder scorer on mock data...")
    for epoch in range(args.epochs):
        total = 0.0
        for _ in range(args.steps):
            batch = get_mock_batch(device)
            head_outputs = {k: {k2: v2 for k2, v2 in v.items()} for k, v in batch["head_outputs"].items()}
            input_ids = batch["input_ids"]

            cands = model.isoform_decoder.decode(head_outputs, strand=batch["strand"], input_ids=input_ids)
            pos, neg = select_candidates(cands, batch["true_isoform"])
            if not pos or not neg:
                continue
            scorer = model.isoform_decoder.scorer
            pos_score = scorer(pos[0], head_outputs, input_ids=input_ids)
            neg_score = scorer(neg[0], head_outputs, input_ids=input_ids)
            loss = loss_fn(pos_score, neg_score)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg = total / max(1, args.steps)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}")

    # quick validation
    batch = get_mock_batch(device)
    cands = model.isoform_decoder.decode(batch["head_outputs"], strand="+", input_ids=batch["input_ids"])
    pos, neg = select_candidates(cands, batch["true_isoform"])
    if pos and neg:
        s_pos = model.isoform_decoder.scorer(pos[0], batch["head_outputs"], input_ids=batch["input_ids"]).item()
        s_neg = model.isoform_decoder.scorer(neg[0], batch["head_outputs"], input_ids=batch["input_ids"]).item()
        print(f"Validation: pos={s_pos:.3f}  neg={s_neg:.3f}  (expect pos>neg)")

if __name__ == "__main__":
    main()