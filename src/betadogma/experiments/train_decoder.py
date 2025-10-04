"""
Phase 2a: Decoder fine-tuning for structural accuracy.

This script trains the IsoformScorer's parameters using a listwise
ranking loss. It freezes the underlying encoder and structural heads,
focusing only on teaching the decoder to rank ground-truth isoforms highly.

Usage:
  python -m betadogma.experiments.train_decoder --config betadogma/experiments/config/default.yaml
"""
import argparse
import yaml
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import betadogma.model # For monkeypatching
from betadogma.model import BetaDogmaModel
from betadogma.decoder.types import Exon, Isoform

# --- Dummy Encoder ---
class DummyEncoder(nn.Module):
    """A dummy encoder that avoids loading the real model from Hugging Face."""
    def __init__(self, model_name: str):
        super().__init__()
        print(f"Using DummyEncoder. Model '{model_name}' will not be loaded.")

    def forward(self, *args, **kwargs):
        # This should not be called during decoder training as we provide head_outputs directly.
        raise NotImplementedError("The dummy encoder's forward method should not be called.")

# --- Helper function to compare isoforms ---
def are_isoforms_equal(iso1: Isoform, iso2: Isoform) -> bool:
    """Checks if two isoforms have the same exon chain."""
    if iso1.strand != iso2.strand or len(iso1.exons) != len(iso2.exons):
        return False
    for ex1, ex2 in zip(iso1.exons, iso2.exons):
        if ex1.start != ex2.start or ex1.end != ex2.end:
            return False
    return True

# --- Dataset for Ground-Truth Isoforms ---
class GroundTruthIsoformDataset(Dataset):
    """
    A placeholder dataset that yields model inputs and ground-truth isoforms.
    """
    def __init__(self, num_samples: int = 20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict:
        # For this test, we create a dummy sequence and a ground-truth isoform.
        # The model will try to predict this isoform from the sequence.
        true_exons = [Exon(start=100, end=200, score=0.9), Exon(start=300, end=400, score=0.8)]
        true_isoform = Isoform(exons=true_exons, strand="+", score=0.85)

        # We also need to provide mock head outputs that would lead to this isoform.
        # This is a bit artificial but necessary for a self-contained training script.
        donor_logits = torch.full((1000,), -10.0)
        acceptor_logits = torch.full((1000,), -10.0)
        donor_logits[200] = 5.0
        acceptor_logits[100] = 5.0
        donor_logits[400] = 4.0
        acceptor_logits[300] = 4.0
        # Add some negative distractors
        acceptor_logits[150] = 3.0
        donor_logits[250] = 3.0

        head_outputs = {
            "splice": {
                "donor": donor_logits.unsqueeze(0).unsqueeze(-1),
                "acceptor": acceptor_logits.unsqueeze(0).unsqueeze(-1),
            }
        }
        return {"head_outputs": head_outputs, "true_isoform": true_isoform, "strand": "+"}

def collate_decoder(batch):
    # In this simple case, we process one sample at a time, so no collating needed.
    return batch[0]

# --- Ranking Loss ---
def listwise_hinge_loss(positive_score: torch.Tensor, negative_scores: List[torch.Tensor], margin: float = 0.1):
    """
    Calculates a simple hinge loss to rank the positive sample above negatives.
    Loss = sum(max(0, margin - (positive_score - negative_score)))
    """
    if not negative_scores:
        return torch.tensor(0.0)
    negative_scores_tensor = torch.stack(negative_scores)
    losses = torch.clamp(margin - (positive_score - negative_scores_tensor), min=0)
    return losses.mean()

# --- Training Loop ---
def train_decoder(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Monkeypatch the real encoder with our dummy version before creating the model
    betadogma.model.BetaDogmaEncoder = DummyEncoder

    # 2. Initialize Model and freeze all params
    model = BetaDogmaModel(config=cfg).to(device)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze only the scorer parameters
    for param in model.isoform_decoder.scorer.parameters():
        param.requires_grad = True

    print("Model loaded. Encoder and heads are FROZEN.")
    print("Training decoder scorer parameters...")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.isoform_decoder.scorer.parameters(), lr=cfg["optimizer"]["lr"])

    # 5. Data
    ds = GroundTruthIsoformDataset()
    # Batch size must be 1 for this simple implementation
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_decoder)

    # 6. Training loop
    model.isoform_decoder.scorer.train()
    for epoch in range(cfg["trainer"]["epochs"]):
        total_loss = 0
        for batch in dl:
            optimizer.zero_grad()

            # a. Get candidate isoforms from the decoder
            # We use the mock head_outputs directly from the dataset for this test
            head_outputs = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in batch["head_outputs"].items()}
            candidates = model.isoform_decoder.decode(head_outputs, strand=batch["strand"])

            # b. Find positive and negative samples
            true_isoform = batch["true_isoform"]
            positive_candidate = None
            negative_candidates = []
            for cand in candidates:
                if are_isoforms_equal(cand, true_isoform):
                    positive_candidate = cand
                else:
                    negative_candidates.append(cand)

            if not positive_candidate:
                # If the true isoform was not even found, we can't compute loss for this sample.
                continue

            # c. Calculate scores for all candidates
            positive_score = model.isoform_decoder.scorer(positive_candidate, head_outputs)
            negative_scores = [model.isoform_decoder.scorer(neg, head_outputs) for neg in negative_candidates]

            # d. Compute loss
            loss = listwise_hinge_loss(positive_score, negative_scores, margin=0.1)

            # e. Backpropagate
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dl) if dl else 0
        print(f"Epoch {epoch+1}/{cfg['trainer']['epochs']} | Avg Loss: {avg_loss:.4f}")

    print("Decoder training complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_decoder(cfg)

if __name__ == "__main__":
    main()