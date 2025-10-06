# tests/test_full_workflow.py
import os
from pathlib import Path
import torch
import yaml

from betadogma.model import BetaDogmaModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_CFG = PROJECT_ROOT / "src" / "betadogma" / "experiments" / "config" / "default.yaml"

def load_cfg(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def test_full_model_workflow_shapes_and_decode():
    assert DEFAULT_CFG.exists(), f"Config not found at {DEFAULT_CFG}"
    cfg = load_cfg(DEFAULT_CFG)
    d_in = int(cfg["encoder"]["hidden_size"])
    model = BetaDogmaModel(d_in=d_in, config=cfg).eval()

    # Dummy embeddings (avoid heavy HF download in CI)
    B, L = 2, 1024
    embeddings = torch.randn(B, L, d_in)

    # Forward through heads
    outs = model(embeddings=embeddings)

    # --- Shape checks ---
    # splice head: dict with donor, acceptor [B, L, 1]
    assert "splice" in outs and "donor" in outs["splice"] and "acceptor" in outs["splice"]
    assert outs["splice"]["donor"].shape == (B, L, 1)
    assert outs["splice"]["acceptor"].shape == (B, L, 1)

    # tss / polya [B, L, 1]
    assert outs["tss"]["tss"].shape == (B, L, 1)
    assert outs["polya"]["polya"].shape == (B, L, 1)

    # orf head: start/stop [B,L,1], frame [B,L,3]
    assert outs["orf"]["start"].shape == (B, L, 1)
    assert outs["orf"]["stop"].shape  == (B, L, 1)
    assert outs["orf"]["frame"].shape == (B, L, 3)

    # --- Decode to isoforms (uses batch 0 by design) ---
    # The decoder expects per-sample tensors; we pass the full dict (B,L,*) and it will squeeze/index internally.
    preds = model.predict(outs, strand='+')
    # preds is a Prediction wrapper; access isoforms list and derived properties
    assert hasattr(preds, "isoforms")
    # psi: dict mapping isoform keys -> PSI
    psi = preds.psi
    assert isinstance(psi, dict)
    # dominant isoform and NMD prob should always be computable (0.0 if no isoforms)
    _ = preds.dominant_isoform
    _ = preds.p_nmd  # float

    # The test is intentionally permissive about content—this is a smoke test.