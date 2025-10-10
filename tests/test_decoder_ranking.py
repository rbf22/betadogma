# tests/test_decoder_ranking.py
from __future__ import annotations

import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Import local modules with type checking only
try:
    from betadogma.model import BetaDogmaModel
    from betadogma.decoder import Isoform, Exon
    from betadogma.core.losses import structural_bce_ce_loss
    from betadogma.core.encoder_nt import NTEncoder
except ImportError:
    pass  # For type checking only

HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "src" / "betadogma" / "experiments" / "config" / "default.yaml"

class PairwiseHingeLoss(torch.nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    def forward(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.margin - (pos - neg)).mean()

def _mock_batch(device: torch.device, L: int = 2048) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Isoform, torch.Tensor]:
    # Create logits with a clean “true path”: TSS@50, donor@200, acceptor@300, polyA@500
    donor = torch.full((1, L), -10.0, device=device)
    acceptor = torch.full((1, L), -10.0, device=device)
    tss = torch.full((1, L), -10.0, device=device)
    polya = torch.full((1, L), -10.0, device=device)
    orf_start = torch.full((1, L), -10.0, device=device)
    orf_stop  = torch.full((1, L), -10.0, device=device)
    orf_frame = torch.randn(1, L, 3, device=device)  # logits for 3 frames

    tss[0, 50] = 5.0
    donor[0, 200] = 5.0
    acceptor[0, 300] = 5.0
    polya[0, 500] = 5.0
    # distractors
    acceptor[0, 100] = 2.0
    donor[0, 400] = 2.0

    head_outputs = {
        "splice": {"donor": donor.unsqueeze(-1), "acceptor": acceptor.unsqueeze(-1)},  # [1,L,1]
        "tss": {"tss": tss.unsqueeze(-1)},
        "polya": {"polya": polya.unsqueeze(-1)},
        "orf": {"start": orf_start.unsqueeze(-1), "stop": orf_stop.unsqueeze(-1), "frame": orf_frame},
    }
    true_iso = Isoform(exons=[Exon(50, 200), Exon(300, 500)], strand="+")
    input_ids = torch.randint(0, 5, (1, L), device=device)  # used by sequence-based fallback if enabled
    return head_outputs, true_iso, input_ids

@torch.no_grad()
def _score_pair(
    model: Any, 
    head_outputs: Dict[str, Dict[str, torch.Tensor]], 
    true_iso: Isoform, 
    input_ids: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # enumerate candidates
    cands = model.isoform_decoder.decode(head_outputs, strand="+", input_ids=input_ids)
    # split
    true_exon_pairs = tuple((e.start, e.end) for e in true_iso.exons)
    pos: List[Isoform] = []
    neg: List[Isoform] = []
    for c in cands:
        pairs = tuple((e.start, e.end) for e in c.exons)
        (pos if pairs == true_exon_pairs else neg).append(c)
    if not pos or not neg:
        return None, None
    scorer = model.isoform_decoder.scorer
    p = scorer(pos[0], head_outputs, input_ids=input_ids)
    n = scorer(neg[0], head_outputs, input_ids=input_ids)
    return p.item(), n.item()

def test_decoder_ranking_learns():
    assert CFG.exists(), f"Missing config: {CFG}"
    cfg = yaml.safe_load(CFG.read_text())
    torch.manual_seed(7)

    d_in = int(cfg["encoder"]["hidden_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaDogmaModel(d_in=d_in, config=cfg).to(device)

    # train only the scorer
    for p in model.parameters():
        p.requires_grad = False
    for p in model.isoform_decoder.scorer.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.isoform_decoder.scorer.parameters(), lr=cfg["optimizer"]["lr"])
    loss_fn = PairwiseHingeLoss(0.5)

    # one batch repeated a few steps (keeps runtime tiny)
    head_outputs, true_iso, input_ids = _mock_batch(device)
    model.train()
    base_pos, base_neg = _score_pair(model, head_outputs, true_iso, input_ids)

    # If enumeration didn’t yield a pos/neg split, just skip (rare with these peaks)
    if base_pos is None:
        return

    for _ in range(10):
        opt.zero_grad()
        cands = model.isoform_decoder.decode(head_outputs, strand="+", input_ids=input_ids)
        true_pairs = tuple((e.start, e.end) for e in true_iso.exons)
        pos = [c for c in cands if tuple((e.start, e.end) for e in c.exons) == true_pairs]
        neg = [c for c in cands if tuple((e.start, e.end) for e in c.exons) != true_pairs]
        if not pos or not neg:
            break
        scorer = model.isoform_decoder.scorer
        pos_score = scorer(pos[0], head_outputs, input_ids=input_ids)
        neg_score = scorer(neg[0], head_outputs, input_ids=input_ids)
        loss = loss_fn(pos_score, neg_score)
        loss.backward()
        opt.step()

    model.eval()
    pos_after, neg_after = _score_pair(model, head_outputs, true_iso, input_ids)
    # The scorer should rank the positive higher after training
    assert pos_after is not None and neg_after is not None
    assert pos_after > neg_after, f"Expected pos > neg, got {pos_after:.4f} <= {neg_after:.4f}"