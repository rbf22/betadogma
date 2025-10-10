"""Isoform graph construction and decoding.

This module provides tools for constructing splice graphs from model outputs,
enumerating candidate isoforms, and scoring them based on various features.
It also includes utilities for evaluating the quality of predicted isoforms
against ground truth annotations.

Types:
    Isoform: A class representing a transcript isoform with exons and metadata.
    Exon: A class representing a single exon with start, end, and score.
    IsoformDecoder: Main class for decoding isoforms from model outputs.
    IsoformScorer: Class for scoring candidate isoforms.
    IsoformEnumerator: Class for enumerating candidate isoforms.
    SpliceGraph: Graph representation of exons and their junctions.
    SpliceGraphBuilder: Builder class for constructing splice graphs.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union

import torch

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

__all__: List[str] = [
    # types
    "Isoform", 
    "Exon",
    
    # decoder components
    "IsoformDecoder", 
    "IsoformScorer", 
    "IsoformEnumerator",
    "SpliceGraphBuilder", 
    "SpliceGraph",
    
    # evaluation
    "exon_chain_match", 
    "junction_f1", 
    "top_k_recall", 
    "evaluate_set", 
    "EvalSummary",
    
    # convenience
    "decode_isoforms",
]

def decode_isoforms(
    head_outputs: Dict[str, torch.Tensor], 
    strand: str = "+", 
    input_ids: Optional[torch.Tensor] = None, 
    config: Optional[Dict[str, Any]] = None
) -> List[Isoform]:
    """Convenience wrapper to decode isoforms from model outputs.
    
    This is a simplified interface that creates an IsoformDecoder instance
    and returns the top candidate isoforms in a single function call.
    
    Args:
        head_outputs: Dictionary of model outputs, typically containing keys like
                     'splice', 'tss', 'polya', 'orf_start', 'orf_stop', 'orf_frame'.
        strand: Strand of the input sequence ('+' or '-'). Defaults to "+".
        input_ids: Optional input token IDs, used for ORF scoring if provided.
        config: Optional configuration dictionary for the decoder.
               If None, default settings will be used.
               
    Returns:
        List of candidate Isoform objects, sorted by score in descending order.
        
    Example:
        >>> import torch
        >>> from typing import Dict
        >>> from betadogma.decoder import decode_isoforms
        
        >>> # Create mock model outputs
        >>> model_outputs: Dict[str, torch.Tensor] = {
        ...     'splice': torch.sigmoid(torch.randn(1, 1000, 1)),
        ...     'tss': torch.sigmoid(torch.randn(1, 1000, 1)),
        ...     'polya': torch.sigmoid(torch.randn(1, 1000, 1)),
        ...     'orf_start': torch.sigmoid(torch.randn(1, 1000, 1)),
        ...     'orf_stop': torch.sigmoid(torch.randn(1, 1000, 1)),
        ...     'orf_frame': torch.sigmoid(torch.randn(1, 1000, 3)),
        ... }
        
        >>> # Decode isoforms
        >>> isoforms: List[Isoform] = decode_isoforms(
        ...     head_outputs=model_outputs,
        ...     strand='+',
        ...     config={'beam_size': 10}
        ... )
        >>> print(f"Predicted {len(isoforms)} isoforms")
    """
    dec = IsoformDecoder(config or {})
    return dec.decode(head_outputs, strand=strand, input_ids=input_ids)