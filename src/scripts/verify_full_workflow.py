# scripts/verify_full_workflow.py
import torch
from pathlib import Path
from betadogma.model import BetaDogmaModel
from betadogma.core.encoder_nt import NTEncoder

def main():
    print("--- Verifying Full Model Workflow (E2E) ---")
    config_path = Path(__file__).resolve().parent.parent / "src/betadogma/experiments/config/default.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"

    # 1) Model
    print("Loading BetaDogmaModel from config...")
    model = BetaDogmaModel.from_config_file(str(config_path))
    model.eval()

    # 2) Encoder (real HF model)
    print("Loading NTEncoder...")
    enc_cfg = model.config["encoder"]
    encoder = NTEncoder(model_id=enc_cfg.get("model_id") or "InstaDeepAI/nucleotide-transformer-500m-human-ref")

    # Sanity check: dimensions
    # Works for conv-head path (LayerNorm is present); if non-conv, skip the norm query.
    if hasattr(model.splice_head, "proj") and hasattr(model.splice_head.proj, "norm"):
        head_d_in = model.splice_head.proj.norm.normalized_shape[0]
        assert head_d_in == encoder.hidden_size, \
            f"Model d_in ({head_d_in}) != encoder hidden_size ({encoder.hidden_size})"
    else:
        # Fallback: trust config
        print("Warning: conv head norm not found; relying on config hidden_size.")

    # 3) Dummy sequence
    dummy_sequence = "N" * 4096
    print(f"Creating embeddings for sequence len={len(dummy_sequence)}...")
    with torch.no_grad():
        embeddings = encoder.forward([dummy_sequence])  # [1, L, D]
    print(f"Embeddings shape: {tuple(embeddings.shape)}")

    # 4) Forward heads
    with torch.no_grad():
        outputs = model(embeddings)

    # 5) Print shapes
    print("\n--- Head output shapes ---")
    print(f"Splice (donor):    {outputs['splice']['donor'].shape}")
    print(f"Splice (acceptor): {outputs['splice']['acceptor'].shape}")
    print(f"TSS:               {outputs['tss']['tss'].shape}")
    print(f"PolyA:             {outputs['polya']['polya'].shape}")
    print(f"ORF (start):       {outputs['orf']['start'].shape}")
    print(f"ORF (stop):        {outputs['orf']['stop'].shape}")
    print(f"ORF (frame):       {outputs['orf']['frame'].shape}")

    # 6) Decode and NMD on the same outputs
    pred = model.predict(outputs, strand='+')
    dom = pred.dominant_isoform
    p_nmd = pred.p_nmd
    n_iso = len(pred.isoforms)
    print(f"\nDecoded {n_iso} isoform candidates.")
    if dom is not None:
        print(f"Dominant isoform score: {dom.score:.3f} | p(NMD): {p_nmd:.3f}")
    else:
        print("No dominant isoform (empty candidate set).")

    print("\nFull E2E workflow verified successfully!")

if __name__ == "__main__":
    main()