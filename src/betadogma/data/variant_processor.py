"""
Variant processing utilities for on-the-fly sequence modification during training.

This module provides functions to apply variants to reference sequences during training,
allowing for dynamic generation of alternative sequences without pre-computation.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
import torch
from pathlib import Path

from betadogma.data.encode import apply_variants_to_sequence, rescue_insertion_variant
from betadogma.data.variant_loader import VariantLoader


def generate_alt_sequence(
    ref_seq: str,
    variants: List[Dict[str, Any]],
    max_variants: int = 5,
    balance_variants: bool = True,
    seed: Optional[int] = None,
    strict_ref_check: bool = False
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate an alternative sequence by applying variants to a reference sequence.
    
    Args:
        ref_seq: Reference DNA sequence
        variants: List of variant dictionaries with 'pos', 'ref', and 'alt' keys
        max_variants: Maximum number of variants to apply (0 for no limit)
        balance_variants: Whether to balance variant types when selecting
        seed: Random seed for reproducibility
        strict_ref_check: If True, raises error on reference mismatches
        
    Returns:
        Tuple of (modified_sequence, applied_variants)
    """
    if not variants:
        return ref_seq, []
    
    # Create a local random generator for reproducibility
    rng = random.Random(seed)
    
    # Filter and prepare variants
    valid_variants = []
    for var in variants:
        # Ensure variant is within sequence bounds
        pos = var['pos'] - 1  # Convert to 0-based
        ref_len = len(var['ref'])
        if 0 <= pos < len(ref_seq) and (pos + ref_len) <= len(ref_seq):
            # Check reference sequence matches
            if ref_seq[pos:pos+ref_len] == var['ref'] or not strict_ref_check:
                valid_variants.append(var)
    
    if not valid_variants:
        return ref_seq, []
    
    # Limit number of variants if needed
    if 0 < max_variants < len(valid_variants):
        if balance_variants:
            # Group by variant type
            by_type = {}
            for var in valid_variants:
                var_type = 'SNP' if len(var['ref']) == len(var['alt']) else 'INDEL'
                if var_type not in by_type:
                    by_type[var_type] = []
                by_type[var_type].append(var)
            
            # Sample proportionally from each type
            selected = []
            remaining = max_variants
            total = len(valid_variants)
            
            # First pass: allocate based on distribution
            allocations = {}
            for var_type, vars_of_type in by_type.items():
                count = max(1, int((len(vars_of_type) / total) * max_variants))
                count = min(count, len(vars_of_type), remaining)
                allocations[var_type] = count
                remaining -= count
            
            # Second pass: fill any remaining slots
            while remaining > 0:
                for var_type in by_type:
                    if allocations[var_type] < len(by_type[var_type]) and remaining > 0:
                        allocations[var_type] += 1
                        remaining -= 1
                    if remaining == 0:
                        break
            
            # Select variants
            for var_type, count in allocations.items():
                selected.extend(rng.sample(by_type[var_type], count))
            
            valid_variants = selected
        else:
            valid_variants = rng.sample(valid_variants, max_variants)
    
    # Sort variants by position (required for apply_variants_to_sequence)
    valid_variants = sorted(valid_variants, key=lambda x: x['pos'])
    
    try:
        # Apply variants to generate alternative sequence
        alt_seq = apply_variants_to_sequence(
            ref_seq,
            valid_variants,
            strict_ref_check=strict_ref_check
        )
        return alt_seq, valid_variants
    except Exception as e:
        # Fallback to reference sequence if variant application fails
        print(f"Warning: Failed to apply variants: {str(e)}")
        return ref_seq, []


class OnTheFlyVariantProcessor:
    """Process variants on-the-fly during training."""
    
    def __init__(
        self,
        variant_loader: VariantLoader,
        max_variants: int = 5,
        balance_variants: bool = True,
        strict_ref_check: bool = False,
        variant_prob: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize the variant processor.
        
        Args:
            variant_loader: Loader for accessing variants
            max_variants: Maximum number of variants to apply per sequence
            balance_variants: Whether to balance variant types when selecting
            strict_ref_check: If True, raises error on reference mismatches
            variant_prob: Probability of applying variants to a sequence
            seed: Random seed for reproducibility
        """
        self.variant_loader = variant_loader
        self.max_variants = max(max_variants, 0)
        self.balance_variants = balance_variants
        self.strict_ref_check = strict_ref_check
        self.variant_prob = max(0.0, min(1.0, variant_prob))
        self.rng = random.Random(seed)
        self._rng_state = None
    
    def set_epoch(self, epoch: int) -> None:
        """Set the random seed based on the current epoch."""
        if self.seed is not None:
            self.rng.seed(self.seed + epoch)
    
    def process_batch(
        self,
        batch: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of sequences by applying variants on-the-fly.
        
        Args:
            batch: Dictionary containing batch data with 'seqs' (list of sequences)
            device: Device to move tensors to
            
        Returns:
            Dictionary with additional 'seqs_alt' and 'has_variant' keys
        """
        if 'seqs' not in batch:
            return batch
        
        seqs = batch['seqs']
        batch_size = len(seqs)
        
        # Initialize output
        seqs_alt = []
        has_variant = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, seq in enumerate(seqs):
            # Skip variant application based on probability
            if self.rng.random() > self.variant_prob:
                seqs_alt.append(seq)
                continue
            
            # Get variants for this sequence (implement this method in your variant_loader)
            variants = self.variant_loader.get_variants_for_sequence(seq)
            
            if not variants:
                seqs_alt.append(seq)
                continue
            
            # Generate alternative sequence
            alt_seq, applied_variants = generate_alt_sequence(
                ref_seq=seq,
                variants=variants,
                max_variants=self.max_variants,
                balance_variants=self.balance_variants,
                seed=self.rng.randint(0, 2**32 - 1),
                strict_ref_check=self.strict_ref_check
            )
            
            if alt_seq != seq and applied_variants:
                seqs_alt.append(alt_seq)
                has_variant[i] = True
            else:
                seqs_alt.append(seq)
        
        # Add to batch
        batch['seqs_alt'] = seqs_alt
        batch['has_variant'] = has_variant
        
        return batch
