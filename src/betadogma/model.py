"""
High-level BetaDogma model API.

This module exposes a simple interface for inference:
- BetaDogmaModel: orchestrates encoder, heads, decoder, and NMD predictor.
- preprocess_sequence / preprocess_variant: utilities to create inputs.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from .core.encoder import BetaDogmaEncoder
from .core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead


@dataclass
class Prediction:
    dominant_isoform: Dict[str, Any]
    psi: Dict[str, float]
    p_nmd: float
    aux: Dict[str, Any]


class BetaDogmaModel(nn.Module):
    """
    Coordinates the backbone encoder, per-base heads, isoform decoder, and NMD predictor.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = BetaDogmaEncoder(model_name=self.config["encoder"]["model_name"])

        # Heads
        d_in = self.config["encoder"]["hidden_size"]
        head_config = self.config["heads"]

        self.splice_head = SpliceHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.tss_head = TSSHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.polya_head = PolyAHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.orf_head = ORFHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Runs the encoder and all prediction heads.
        """
        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        outputs = {
            "splice": self.splice_head(embeddings),
            "tss": self.tss_head(embeddings),
            "polya": self.polya_head(embeddings),
            "orf": self.orf_head(embeddings),
            "embeddings": embeddings,
        }
        return outputs

    @classmethod
    def from_pretrained(cls, tag: str, config: Optional[Dict[str, Any]] = None):
        """Load a pretrained BetaDogma model (weights + config). Placeholder."""
        # In a real scenario, this would load a config from the hub if not provided
        # and also load weights.
        if config is None:
            # Placeholder config
            config = {
                "encoder": {
                    "model_name": "arm-genomics/enformer-finetuned-human-128k",
                    "hidden_size": 1536,
                },
                "heads": {
                    "hidden": 768,
                    "dropout": 0.1,
                    "use_conv": True,
                },
                "tag": tag,
            }
        return cls(config=config)

    @torch.no_grad()
    def predict(self, sequence: str, variant: Optional[Dict[str, Any]] = None) -> Prediction:
        """
        Run end-to-end inference.
        Args:
            sequence: DNA bases A/C/G/T/N for a genomic window.
            variant: optional structured variant encoding dict.
        Returns:
            Prediction: dominant isoform, ψ distribution, P(NMD), and aux head outputs.
        """
        # 1. Preprocess sequence into token IDs (placeholder)
        # This would involve a tokenizer specific to the genomic model.
        # For now, creating a dummy input tensor.
        dummy_input_ids = torch.randint(0, 5, (1, 1000))  # Batch size 1, sequence length 1000

        # 2. Run the model's forward pass
        head_outputs = self.forward(dummy_input_ids)

        # 3. Decode head outputs into biological structures (placeholder)
        # This is where the isoform decoder and NMD model would run.
        dominant_isoform = {"exons": [], "cds": None, "note": "decoded from model output (stub)"}
        psi = {"note": "decoded psi (stub)"}
        p_nmd = 0.0  # from NMD model (stub)

        # 4. Package for output
        # Aux contains the raw head logits for inspection
        aux = {
            "splice_logits": head_outputs["splice"],
            "tss_logits": head_outputs["tss"],
            "polya_logits": head_outputs["polya"],
            "orf_logits": head_outputs["orf"],
        }

        return Prediction(dominant_isoform, psi, p_nmd, aux)

def preprocess_sequence(chrom: str, start: int, end: int) -> str:
    """
    Fetch and normalize a genomic window. Placeholder.
    Replace with FASTA-backed retrieval (pyfaidx/pysam) and uppercase normalization.
    """
    # TODO: implement FASTA retrieval
    return "N" * (end - start)

def preprocess_variant(vcf_like: str, window: Any = None) -> Dict[str, Any]:
    """
    Convert a simple variant spec (e.g., '17:43051000A>T') into a structured encoding
    suitable for the variant channel.
    """
    # TODO: parse and align to window coords
    return {"spec": vcf_like}
