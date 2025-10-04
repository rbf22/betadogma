"""
Phase 2a Training: Isoform Structure Ranking

This script trains the IsoformScorer to rank candidate isoforms.
The goal is to teach the model to assign higher scores to the annotated/correct
isoform compared to other enumerated candidates.
"""

import argparse
import yaml
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Import the module so we can patch it
import betadogma.model
from betadogma.model import BetaDogmaModel
from betadogma.decoder import Isoform, Exon

# --- Monkeypatching for lightweight training ---
class DummyEncoder(nn.Module):
    """A dummy encoder that returns random tensors of the correct shape."""
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # This should match the encoder's hidden_size from the config
        return torch.randn(batch_size, seq_len, 1536)

def monkeypatch_encoder():
    """Replaces the real encoder class with a dummy one."""
    print("Monkeypatching BetaDogmaEncoder with DummyEncoder.")
    betadogma.model.BetaDogmaEncoder = DummyEncoder

# --- Loss Function ---
class PairwiseHingeLoss(nn.Module):
    """Pairwise hinge loss for ranking."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        loss = torch.relu(self.margin - (positive_scores - negative_scores))
        return loss.mean()

# --- Training Logic ---
def get_mock_batch(device: torch.device) -> Dict[str, Any]:
    """Creates a mock batch of data for demonstration purposes."""
    seq_len = 2048

    # Create logits with a few high-confidence peaks to prevent combinatorial explosion
    donor_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    acceptor_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    tss_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    polya_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_start_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_stop_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_frame_logits = torch.randn(1, 1, seq_len, 3, device=device)


    # Define sites for a true isoform and some distractors
    # True isoform path: [Exon(50, 200), Exon(300, 500)]
    # This requires: TSS@50, don@200, acc@300, polyA@500
    tss_logits[0, 0, 50] = 5.0
    donor_logits[0, 0, 200] = 5.0
    acceptor_logits[0, 0, 300] = 5.0
    polya_logits[0, 0, 500] = 5.0

    # Add some distractor sites
    acceptor_logits[0, 0, 100] = 2.0
    donor_logits[0, 0, 400] = 2.0

    head_outputs = {
        "splice": {"donor": donor_logits.transpose(1, 2), "acceptor": acceptor_logits.transpose(1, 2)},
        "tss": {"tss": tss_logits.transpose(1, 2)},
        "polya": {"polya": polya_logits.transpose(1, 2)},
        "orf": {
            "start": orf_start_logits.transpose(1, 2),
            "stop": orf_stop_logits.transpose(1, 2),
            "frame": orf_frame_logits.squeeze(0), # Remove channel dim
        }
    }

    # A "true" isoform for ranking against
    true_isoform = Isoform(exons=[Exon(50, 200), Exon(300, 500)], strand='+')

    # Mock input_ids for sequence-based scoring fallback
    mock_input_ids = torch.randint(0, 5, (1, seq_len), device=device)

    return {"head_outputs": head_outputs, "true_isoform": true_isoform, "strand": "+", "input_ids": mock_input_ids}

def select_candidates(candidates: List[Isoform], true_isoform: Isoform) -> (List[Isoform], List[Isoform]):
    """Separates candidate isoforms into positive (matching ground truth) and negative."""
    # This is a simplified matching logic. A real implementation would be more robust.
    positive, negative = [], []
    true_exons = tuple((e.start, e.end) for e in true_isoform.exons)
    for cand in candidates:
        cand_exons = tuple((e.start, e.end) for e in cand.exons)
        if cand_exons == true_exons:
            positive.append(cand)
        else:
            negative.append(cand)
    return positive, negative

def train_one_epoch(
    model: BetaDogmaModel,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mock_loader: List[Dict]
):
    model.train()
    total_loss = 0.0

    for batch in mock_loader:
        head_outputs = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in batch['head_outputs'].items()}
        input_ids = batch['input_ids'].to(device)

        # The decoder is used as a utility here, not trained directly.
        candidates = model.isoform_decoder.decode(head_outputs, strand=batch['strand'], input_ids=input_ids)

        pos_cands, neg_cands = select_candidates(candidates, batch['true_isoform'])

        if not pos_cands or not neg_cands:
            continue

        # Score candidates using the learnable scorer
        scorer = model.isoform_decoder.scorer
        pos_scores = torch.stack([scorer(iso, head_outputs, input_ids=input_ids) for iso in pos_cands])
        # For simplicity, we only use one negative example per positive
        neg_scores = torch.stack([scorer(iso, head_outputs, input_ids=input_ids) for iso in neg_cands[:len(pos_cands)]])

        loss = loss_fn(pos_scores, neg_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(mock_loader) if mock_loader else 0.0

def main():
    parser = argparse.ArgumentParser(description="Train the IsoformScorer for ranking.")
    parser.add_argument("--config", default="src/betadogma/experiments/config/default.yaml", help="Path to config.")
    parser.add_argument("--no-patch", action="store_true", help="Do not monkeypatch the encoder.")
    args = parser.parse_args()

    if not args.no_patch:
        monkeypatch_encoder()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BetaDogmaModel(config)
    model.to(device)

    # We are only training the scorer's weights
    trainable_parameters = model.isoform_decoder.scorer.parameters()

    optimizer = optim.Adam(trainable_parameters, lr=config['optimizer']['lr'])
    loss_fn = PairwiseHingeLoss(margin=0.5)

    # Create a mock data loader for demonstration
    mock_loader = [get_mock_batch(device) for _ in range(10)]

    print("Starting training with mock data...")
    for epoch in range(config['trainer']['epochs']):
        epoch_loss = train_one_epoch(model, optimizer, loss_fn, device, mock_loader)
        print(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Mock Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()