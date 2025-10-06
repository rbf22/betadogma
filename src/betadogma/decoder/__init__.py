"""Isoform graph construction and decoding."""

from .types import Isoform, Exon
from .isoform_decoder import (
    IsoformDecoder,
    IsoformScorer,
    IsoformEnumerator,
    SpliceGraphBuilder,
    SpliceGraph,
)
from .evaluate import (
    exon_chain_match,
    junction_f1,
    top_k_recall,
    evaluate_set,
    EvalSummary,
)

__all__ = [
    # types
    "Isoform", "Exon",
    # decoder components
    "IsoformDecoder", "IsoformScorer", "IsoformEnumerator",
    "SpliceGraphBuilder", "SpliceGraph",
    # evaluation
    "exon_chain_match", "junction_f1", "top_k_recall", "evaluate_set", "EvalSummary",
    # convenience
    "decode_isoforms",
]

def decode_isoforms(head_outputs, strand: str = "+", input_ids=None, config=None):
    """
    Convenience wrapper: build a decoder and return candidate isoforms.
    Useful for quick notebooks/tests without importing the class.
    """
    dec = IsoformDecoder(config or {})
    return dec.decode(head_outputs, strand=strand, input_ids=input_ids)