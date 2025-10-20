"""
Variant handling utilities for the BetaDogma model.

This module provides functions to sample and apply variants to reference sequences
during training, with support for both pathogenic and benign variants.
"""
from typing import Dict, List, Optional, Tuple, Set, Union
import random
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def apply_variant(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    Apply a variant to a reference sequence.
    
    Args:
        seq: Reference sequence
        pos: 1-based position of the variant
        ref: Reference allele
        alt: Alternate allele
        
    Returns:
        Modified sequence with the variant applied
    """
    # Convert to 0-based indexing for Python
    pos_0 = pos - 1
    
    # Verify the reference matches
    if seq[pos_0:pos_0 + len(ref)] != ref:
        raise ValueError(f"Reference mismatch at position {pos}: expected '{ref}', got '{seq[pos_0:pos_0 + len(ref)]}'")
    
    # Apply the variant
    return seq[:pos_0] + alt + seq[pos_0 + len(ref):]


def sample_variants(
    variants: List[Dict[str, any]],
    max_variants: int = 5,
    min_distance: int = 10,
    rng: Optional[random.Random] = None
) -> List[Dict[str, any]]:
    """
    Sample non-overlapping variants from a list of possible variants.
    
    Args:
        variants: List of variant dictionaries, each with 'pos', 'ref', 'alt' keys
        max_variants: Maximum number of variants to sample
        min_distance: Minimum distance between selected variants
        rng: Random number generator for reproducibility
        
    Returns:
        List of selected variant dictionaries
    """
    if rng is None:
        rng = random.Random()
    
    if not variants:
        return []
    
    # Sort variants by position
    sorted_variants = sorted(variants, key=lambda x: x['pos'])
    
    selected = []
    last_pos = -min_distance
    
    # Try to sample up to max_variants, maintaining minimum distance
    for var in sorted_variants:
        if len(selected) >= max_variants:
            break
            
        # Check minimum distance from previous variant
        if var['pos'] >= last_pos + min_distance:
            selected.append(var)
            last_pos = var['pos'] + len(var['ref']) - 1  # End position of this variant
    
    return selected


def apply_variants_to_sequence(
    seq: str,
    variants: List[Dict[str, any]],
    max_variants: int = 5,
    min_distance: int = 10,
    rng: Optional[random.Random] = None
) -> Tuple[str, List[Dict[str, any]]]:
    """
    Apply sampled variants to a reference sequence.
    
    Args:
        seq: Reference sequence
        variants: List of possible variants to apply
        max_variants: Maximum number of variants to apply
        min_distance: Minimum distance between applied variants
        rng: Random number generator for reproducibility
        
    Returns:
        Tuple of (modified_sequence, applied_variants)
    """
    if not variants:
        return seq, []
    
    # Sample variants
    selected_variants = sample_variants(
        variants,
        max_variants=max_variants,
        min_distance=min_distance,
        rng=rng
    )
    
    if not selected_variants:
        return seq, []
    
    # Sort variants by position in descending order to handle multiple variants correctly
    selected_variants.sort(key=lambda x: x['pos'], reverse=True)
    
    # Apply variants from right to left to maintain positions
    modified_seq = seq
    applied_vars = []
    
    for var in selected_variants:
        try:
            modified_seq = apply_variant(modified_seq, var['pos'], var['ref'], var['alt'])
            applied_vars.append({
                'pos': var['pos'],
                'ref': var['ref'],
                'alt': var['alt'],
                'is_pathogenic': var.get('is_pathogenic', False),
                'af': var.get('af', 0.0)
            })
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to apply variant {var}: {e}")
            continue
    
    return modified_seq, applied_vars
