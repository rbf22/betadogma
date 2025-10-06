"""
High-level BetaDogma model API.

This module exposes a simple interface for inference:
- BetaDogmaModel: orchestrates encoder, heads, decoder, and NMD predictor.
- preprocess_sequence / preprocess_variant: utilities to create inputs.
"""

from __future__ import annotations

import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Heads each return a dict of per-base logits with fixed keys:
#   SpliceHead -> {"donor": (B,L,1), "acceptor": (B,L,1)}
#   TSSHead    -> {"tss":   (B,L,1)}
#   PolyAHead  -> {"polya": (B,L,1)}
#   ORFHead    -> {"start": (B,L,1), "stop": (B,L,1), "frame": (B,L,3)}
from .core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead

# IsoformDecoder consumes the dict of head outputs and returns a list[Isoform]
from .decoder.isoform_decoder import IsoformDecoder
from .decoder.types import Isoform  # assumes fields: exons:List[Exon], strand:str, score:float

# Lightweight NMD predictor; must expose .predict(Isoform)->float
from .nmd.nmd_model import NMDModel


# ---------- Small helpers

def _isoform_to_key(iso: Isoform) -> str:
    """Create a stable string identifier for an isoform based on exon boundaries and strand."""
    exon_str = ",".join(f"{e.start}-{e.end}" for e in iso.exons)
    return f"{iso.strand}:{exon_str}"


# ---------- Prediction container

@dataclass
class Prediction:
    """
    Encapsulates the full output of a BetaDogma prediction.

    Holds candidate isoforms and exposes convenience properties:
    - psi: dict[isoform_key] -> PSI
    - dominant_isoform: Isoform with highest raw score
    - p_nmd: NMD probability for the dominant isoform
    """
    isoforms: List[Isoform]
    _nmd_model: NMDModel  # reference used on-demand

    @property
    def psi(self) -> Dict[str, float]:
        if not self.isoforms:
            return {}
        scores = torch.tensor([iso.score for iso in self.isoforms], dtype=torch.float32)
        probs = F.softmax(scores, dim=0)
        return {_isoform_to_key(iso): float(p) for iso, p in zip(self.isoforms, probs)}

    @property
    def dominant_isoform(self) -> Optional[Isoform]:
        if not self.isoforms:
            return None
        return max(self.isoforms, key=lambda iso: iso.score)

    @property
    def p_nmd(self) -> float:
        dom = self.dominant_isoform
        if dom is None:
            return 0.0
        return float(self._nmd_model.predict(dom))


# ---------- Main model

class BetaDogmaModel(nn.Module):
    """
    Coordinates (precomputed) embeddings -> per-base heads -> isoform decoding -> NMD.
    You supply d_in (embedding size) and a config dict (or use from_config_file()).

    Expected config keys:
      heads:
        hidden: int
        dropout: float
        use_conv: bool
      decoder: { ... }  # passed to IsoformDecoder
      nmd: { ... }      # passed to NMDModel
    """

    def __init__(self, d_in: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        head_cfg = config.get("heads", {})
        d_hidden = int(head_cfg.get("hidden", 768))
        dropout = float(head_cfg.get("dropout", 0.1))
        use_conv = bool(head_cfg.get("use_conv", True))

        # Per-base structural heads
        self.splice_head = SpliceHead(d_in=d_in, d_hidden=d_hidden, dropout=dropout, use_conv=use_conv)
        self.tss_head    = TSSHead(   d_in=d_in, d_hidden=d_hidden, dropout=dropout, use_conv=use_conv)
        self.polya_head  = PolyAHead( d_in=d_in, d_hidden=d_hidden, dropout=dropout, use_conv=use_conv)
        self.orf_head    = ORFHead(   d_in=d_in, d_hidden=d_hidden, dropout=dropout, use_conv=use_conv)

        # Isoform decoder (graph + scoring)
        self.isoform_decoder = IsoformDecoder(config=config.get("decoder", {}))

        # NMD predictor
        self.nmd_model = NMDModel(config=config.get("nmd", {}))

    def forward(self, embeddings: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run the per-base heads on precomputed embeddings.

        Args:
            embeddings: (B, L, d_in) tensor
            input_ids: optional sequence tokens aligned to L (e.g., A/C/G/T/N vocab)

        Returns:
            A dict bundling all head outputs, plus embeddings/input_ids passthrough:
            {
                "splice": {"donor":(B,L,1), "acceptor":(B,L,1)},
                "tss":    {"tss":(B,L,1)},
                "polya":  {"polya":(B,L,1)},
                "orf":    {"start":(B,L,1), "stop":(B,L,1), "frame":(B,L,3)},
                "embeddings": embeddings,
                "input_ids": input_ids,
            }
        """
        outputs = {
            "splice": self.splice_head(embeddings),
            "tss":    self.tss_head(embeddings),
            "polya":  self.polya_head(embeddings),
            "orf":    self.orf_head(embeddings),
            "embeddings": embeddings,
            "input_ids": input_ids,
        }
        return outputs

    @torch.no_grad()
    def predict(self, head_outputs: Dict[str, Any], strand: str = "+") -> Prediction:
        """
        Decode head outputs into isoforms and wrap them in a Prediction object.
        """
        input_ids = head_outputs.get("input_ids", None)
        isoforms = self.isoform_decoder.decode(head_outputs, strand=strand, input_ids=input_ids)
        return Prediction(isoforms=isoforms, _nmd_model=self.nmd_model)

    @classmethod
    def from_config_file(cls, config_path: str) -> "BetaDogmaModel":
        """Build a model from a YAML config file. Expects encoder.hidden_size."""
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        d_in = cfg.get("encoder", {}).get("hidden_size")
        if d_in is None:
            raise ValueError("`encoder.hidden_size` must be specified in the config.")
        return cls(d_in=d_in, config=cfg)


# ---------- Convenience preprocessing utilities

def preprocess_sequence(chrom: str, start: int, end: int, fasta_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract an uppercased DNA window (A/C/G/T/N) from FASTA.

    Returns: {"chrom":str, "start":int, "end":int, "seq":str}
    """
    if end <= start:
        raise ValueError("end must be > start")

    seq = "N" * (end - start)
    if fasta_path:
        from pyfaidx import Fasta
        fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        if chrom not in fa:
            raise KeyError(f"Chromosome {chrom} not found in FASTA.")
        contig_len = len(fa[chrom])
        s = max(0, start)
        e = min(end, contig_len)
        core = str(fa[chrom][s:e])
        left_pad = "N" * (0 - min(0, start))
        right_pad = "N" * max(0, end - contig_len)
        seq = left_pad + core + right_pad

    return {"chrom": chrom, "start": start, "end": end, "seq": seq}


def preprocess_variant(vcf_like: str, window: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse strings like '17:43051000A>T' or 'chr1:12345-DEL3' into a normalized record.

    Returns:
      {
        "chrom": str,
        "pos": int,           # 0-based genomic position of the first ref base (or insertion point)
        "ref": str,
        "alt": str,
        "type": "SNP"|"DEL"|"INS"|"INDEL",
        "in_window_idx": Optional[int],  # 0-based index into `window["seq"]`, if applicable
      }
    """
    m = re.match(r'(?P<chrom>[^:]+):(?P<pos>\d+)(?P<ref>[ACGTN\-]+)>(?P<alt>[ACGTN\-]+)$', vcf_like)
    if not m:
        raise ValueError(f"Unrecognized variant format: {vcf_like}")

    chrom = m["chrom"]
    pos0 = int(m["pos"]) - 1
    ref = m["ref"]
    alt = m["alt"]

    if len(ref) == len(alt) == 1:
        vtype = "SNP"
    elif alt == "-":
        vtype = "DEL"
    elif ref == "-":
        vtype = "INS"
    else:
        vtype = "INDEL"

    in_idx = None
    if window and window.get("chrom") == chrom and window.get("start") is not None and window.get("end") is not None:
        if window["start"] <= pos0 < window["end"]:
            in_idx = pos0 - window["start"]

    return {"chrom": chrom, "pos": pos0, "ref": ref, "alt": alt, "type": vtype, "in_window_idx": in_idx}