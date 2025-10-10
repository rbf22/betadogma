"""
High-level BetaDogma model API.

This module exposes a simple interface for inference:
- BetaDogmaModel: orchestrates encoder, heads, decoder, and NMD predictor.
- preprocess_sequence / preprocess_variant: utilities to create inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Type aliases for better readability
TensorDict = Dict[str, torch.Tensor]  # e.g., {"donor": (B,L,1), "acceptor": (B,L,1)}
HeadOutputs = Dict[str, TensorDict]    # e.g., {"splice": {"donor": ..., "acceptor": ...}}

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

# Type for window information
class WindowInfo(TypedDict, total=False):
    """Type definition for window information dictionary.
    
    Attributes:
        chrom: Chromosome name (e.g., 'chr1')
        start: Start position (0-based, inclusive)
        end: End position (0-based, exclusive)
        seq: DNA sequence in the window (uppercase A/C/G/T/N)
    """
    chrom: str
    start: int
    end: int
    seq: str

# Type for variant information
class VariantInfo(TypedDict, total=False):
    """Type definition for variant information dictionary."""
    chrom: str
    pos: int
    ref: str
    alt: str
    type: Literal["SNP", "DEL", "INS", "INDEL"]
    in_window_idx: Optional[int]

@dataclass
class Prediction:
    """Encapsulates the full output of a BetaDogma prediction.
    
    This class provides a convenient interface to access prediction results,
    including isoform probabilities and NMD predictions.
    
    Example:
        >>> # Assuming we have a prediction result
        >>> prediction = Prediction(isoforms=predicted_isoforms, _nmd_model=nmd_model)
        >>> 
        >>> # Get PSI (percent spliced in) for each isoform
        >>> psi = prediction.psi  # {'+:100-200,300-400': 0.7, '+:100-500': 0.3}
        >>> 
        >>> # Get the dominant isoform
        >>> dom_iso = prediction.dominant_isoform
        >>> 
        >>> # Get NMD probability for the dominant isoform
        >>> nmd_prob = prediction.p_nmd
    """
    isoforms: List[Isoform]
    _nmd_model: NMDModel  # reference to NMD model for on-demand prediction

    @property
    def psi(self) -> Dict[str, float]:
        """Calculate percent spliced-in (PSI) for each isoform.
        
        Returns:
            Dictionary mapping isoform keys to their PSI values (sums to 1.0).
            The keys are generated using _isoform_to_key().
            
        Note:
            PSI is calculated using softmax over the raw isoform scores.
            An empty dictionary is returned if no isoforms are present.
        """
        if not self.isoforms:
            return {}
        scores = torch.tensor([iso.score for iso in self.isoforms], dtype=torch.float32)
        probs = F.softmax(scores, dim=0)
        return {_isoform_to_key(iso): float(p) for iso, p in zip(self.isoforms, probs)}

    @property
    def dominant_isoform(self) -> Optional[Isoform]:
        """Get the isoform with the highest raw score.
        
        Returns:
            The highest-scoring Isoform, or None if no isoforms are present.
            
        Note:
            In case of a tie, returns the first isoform with the maximum score.
        """
        if not self.isoforms:
            return None
        return max(self.isoforms, key=lambda iso: iso.score)

    @property
    def p_nmd(self) -> float:
        """Get the NMD probability for the dominant isoform.
        
        Returns:
            NMD probability as a float between 0.0 and 1.0.
            Returns 0.0 if no isoforms are present.
            
        Note:
            This evaluates the NMD model on the dominant (highest scoring) isoform.
            To get NMD probabilities for all isoforms, use the NMDModel directly.
        """
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

    def __init__(self, d_in: int, config: Dict[str, Any]) -> None:
        """Initialize BetaDogmaModel with given configuration.
        
        Args:
            d_in: Input dimension (embedding size)
            config: Configuration dictionary with the following structure:
                {
                    "heads": {
                        "hidden": int,     # Hidden dimension size (default: 768)
                        "dropout": float,  # Dropout rate (default: 0.1)
                        "use_conv": bool   # Whether to use convolutional layers (default: True)
                    },
                    "decoder": { ... },    # Configuration for IsoformDecoder
                    "nmd": { ... }         # Configuration for NMDModel
                }
        """
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

        self.isoform_decoder = IsoformDecoder(config=config.get("decoder", {}))

        # NMD predictor
        self.nmd_model = NMDModel(config=config.get("nmd", {}))

    def forward(
        self, 
        embeddings: torch.Tensor, 
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, None]]:
        """
        Run the per-base heads on precomputed embeddings.

        Args:
            embeddings: (B, L, d_in) tensor
            input_ids: Optional sequence tokens aligned to L (e.g., A/C/G/T/N vocab)

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

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'BetaDogmaModel':
        """Build a model from a YAML config file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            An initialized BetaDogmaModel instance
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            KeyError: If required configuration keys are missing
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        if "encoder" not in config or "hidden_size" not in config["encoder"]:
            raise KeyError("Config must contain 'encoder.hidden_size'")
            
        return cls(d_in=config["encoder"]["hidden_size"], config=config)


# ---------- Convenience preprocessing utilities

def preprocess_sequence(
    chrom: str, 
    start: int, 
    end: int, 
    fasta_path: Optional[Union[str, Path]] = None
) -> WindowInfo:
    """Extract a DNA window from a FASTA file.
    
    Args:
        chrom: Chromosome name (e.g., 'chr1')
        start: Start position (0-based, inclusive)
        end: End position (0-based, exclusive)
        fasta_path: Path to the FASTA file. If None, returns a mock sequence.
        
    Returns:
        A dictionary containing:
        {
            "chrom": str,  # Chromosome name
            "start": int,  # Start position (0-based, inclusive)
            "end": int,    # End position (0-based, exclusive)
            "seq": str     # Uppercased DNA sequence (A/C/G/T/N)
        }
        
    Raises:
        ValueError: If start >= end or if coordinates are invalid
        FileNotFoundError: If fasta_path is provided but doesn't exist
    """
    if end <= start:
        raise ValueError("end must be > start")

    seq = "N" * (end - start)
    if fasta_path:
        try:
            from pyfaidx import Fasta  # type: ignore
            fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
            if chrom not in fa:
                raise KeyError(f"Chromosome {chrom} not found in FASTA.")
            contig_len = len(fa[chrom])
        except ImportError:
            raise ImportError("pyfaidx is required for FASTA processing. Install with: pip install pyfaidx")
        s = max(0, start)
        e = min(end, contig_len)
        core = str(fa[chrom][s:e])
        left_pad = "N" * (0 - min(0, start))
        right_pad = "N" * max(0, end - contig_len)
        seq = left_pad + core + right_pad

    return {"chrom": chrom, "start": start, "end": end, "seq": seq}


def preprocess_variant(
    vcf_like: str, 
    window: Optional[WindowInfo] = None
) -> VariantInfo:
    """Parse a VCF-like variant string into a normalized record.
    
    Args:
        vcf_like: Variant string in format 'CHR:POSREF>ALT' or 'CHR:POS-TYPE[LEN]'
                 (e.g., '17:43051000A>T', 'chr1:12345-DEL3')
        window: Optional window information to calculate in_window_idx
        
    Returns:
        A dictionary containing:
        {
            "chrom": str,           # Chromosome name
            "pos": int,             # 0-based genomic position
            "ref": str,             # Reference allele
            "alt": str,             # Alternate allele
            "type": Literal[        # Variant type
                "SNP", "DEL", "INS", "INDEL"
            ],
            "in_window_idx": Optional[int]  # 0-based index into window["seq"] if window provided
        }
        
    Raises:
        ValueError: If the variant string is malformed
    """
    m = re.match(r'(?P<chrom>[^:]+):(?P<pos>\d+)(?P<ref>[ACGTN\-]+)>(?P<alt>[ACGTN\-]+)$', vcf_like)
    if not m:
        raise ValueError(f"Unrecognized variant format: {vcf_like}")
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

    # If we have a window, compute the in-window index
    in_window_idx = None
    if window and window.get("chrom") == m["chrom"] and window.get("start", 0) <= pos0 < window.get("end", 0):
        in_window_idx = pos0 - window["start"]

    # Cast vtype to Literal type for type checking
    variant_type: Literal["SNP", "DEL", "INS", "INDEL"] = vtype  # type: ignore

    return {
        "chrom": m["chrom"],
        "pos": pos0,
        "ref": ref,
        "alt": alt,
        "type": variant_type,
        "in_window_idx": in_window_idx,
    }