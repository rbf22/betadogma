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

from .core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead
from .decoder.isoform_decoder import IsoformDecoder


@dataclass
class Prediction:
    dominant_isoform: Dict[str, Any]
    psi: Dict[str, float]
    p_nmd: float
    aux: Dict[str, Any]


class BetaDogmaModel(nn.Module):
    """
    Coordinates the backbone encoder, per-base heads, and isoform decoder.
    This model is instantiated with a configuration dictionary that defines the
    architecture of all its components.
    """

    def __init__(self, d_in: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # --- 2. Prediction Heads ---
        # These modules process the encoder embeddings to make per-base predictions.
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

        # --- 3. Isoform Decoder ---
        # This component assembles predictions into transcript structures.
        self.isoform_decoder = IsoformDecoder(config=self.config.get("decoder", {}))


    def forward(self, embeddings: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Runs all prediction heads on pre-computed embeddings.
        This is the main forward pass for training, returning logits from each head.
        """
        # Pass embeddings through each prediction head
        outputs = {
            "splice": self.splice_head(embeddings),
            "tss": self.tss_head(embeddings),
            "polya": self.polya_head(embeddings),
            "orf": self.orf_head(embeddings),
            "embeddings": embeddings,
            "input_ids": input_ids, # Pass through for downstream use (e.g., sequence-based ORF scoring)
        }
        return outputs

    @classmethod
    def from_config_file(cls, config_path: str):
        """Load model from a YAML config file."""
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        d_in = config.get("encoder", {}).get("hidden_size")
        if d_in is None:
            raise ValueError("`encoder.hidden_size` must be specified in the config.")

        return cls(d_in=d_in, config=config)

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
