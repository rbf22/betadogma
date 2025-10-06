"""
High-level BetaDogma model API.

This module exposes a simple interface for inference:
- BetaDogmaModel: orchestrates encoder, heads, decoder, and NMD predictor.
- preprocess_sequence / preprocess_variant: utilities to create inputs.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead
from .decoder.isoform_decoder import IsoformDecoder
from .decoder.types import Isoform
from .nmd.nmd_model import NMDModel


def _isoform_to_key(iso: Isoform) -> str:
    """Creates a unique string identifier for an isoform."""
    exon_str = ",".join(f"{e.start}-{e.end}" for e in iso.exons)
    return f"{iso.strand}:{exon_str}"


@dataclass
class Prediction:
    """
    Encapsulates the full output of a BetaDogma prediction.

    This object holds the raw candidate isoforms and computes final values
    (PSI, dominant isoform, NMD status) on demand via properties.
    """
    isoforms: list[Isoform]
    _nmd_model: NMDModel # A reference to the NMD model for on-the-fly prediction

    @property
    def psi(self) -> Dict[str, float]:
        """Calculate Percent Spliced In (PSI) values for all isoforms."""
        if not self.isoforms:
            return {}

        scores = torch.tensor([iso.score for iso in self.isoforms])
        psi_values = F.softmax(scores, dim=0)

        return {
            _isoform_to_key(iso): psi.item()
            for iso, psi in zip(self.isoforms, psi_values)
        }

    @property
    def dominant_isoform(self) -> Optional[Isoform]:
        """Return the isoform with the highest PSI value."""
        if not self.isoforms:
            return None
        # The dominant isoform corresponds to the one with the maximum raw score
        return max(self.isoforms, key=lambda iso: iso.score)

    @property
    def p_nmd(self) -> float:
        """Predict the NMD fate of the dominant isoform."""
        dom_iso = self.dominant_isoform
        if dom_iso is None:
            return 0.0
        return self._nmd_model.predict(dom_iso)


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

        # --- 4. NMD Model ---
        # This component predicts NMD fate.
        self.nmd_model = NMDModel(config=self.config.get("nmd", {}))


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

    def predict(self, head_outputs: Dict[str, Any], strand: str = '+') -> Prediction:
        """
        Decodes the raw outputs from the heads into a final, structured prediction
        object that can compute isoform abundances and NMD status.

        Args:
            head_outputs: A dictionary containing the raw tensor outputs from the
                          model's forward pass.
            strand: The strand of the gene, either '+' or '-'.

        Returns:
            A Prediction object containing the candidate isoforms.
        """
        input_ids = head_outputs.get("input_ids")

        # 1. Decode all candidate isoforms from the graph
        candidate_isoforms = self.isoform_decoder.decode(
            head_outputs,
            strand=strand,
            input_ids=input_ids,
        )

        # 2. Return the Prediction object, which handles all downstream calculations
        return Prediction(isoforms=candidate_isoforms, _nmd_model=self.nmd_model)

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
