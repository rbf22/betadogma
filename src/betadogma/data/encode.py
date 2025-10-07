# src/betadogma/data/encode.py
"""Variant encoding utilities for BetaDogma."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def encode_variant(ref_allele: str, alt_allele: str) -> Dict[str, Any]:
    """
    Encode a variant (SNP/INS/DEL) into a dictionary of features.
    
    Args:
        ref_allele: Reference allele sequence
        alt_allele: Alternate allele sequence
        
    Returns:
        Dictionary with variant features including type, length, and encoded representation
    """
    var_type = "SNP"
    if len(ref_allele) > len(alt_allele):
        var_type = "DEL"
    elif len(ref_allele) < len(alt_allele):
        var_type = "INS"
    
    return {
        "type": var_type,
        "ref_len": len(ref_allele),
        "alt_len": len(alt_allele),
        "is_snp": var_type == "SNP",
        "is_ins": var_type == "INS",
        "is_del": var_type == "DEL",
    }

class VariantError(ValueError):
    """Custom error class for variant processing errors."""
    def __init__(self, message, variant=None, sequence_context=None):
        self.variant = variant
        self.sequence_context = sequence_context
        super().__init__(message)


def _format_variant(variant, sequence=None):
    """Format variant information for error messages with context."""
    if sequence is None or not isinstance(sequence, str):
        return f"pos={variant['pos']} ref='{variant['ref']}' alt='{variant['alt']}'"
    
    pos_0 = variant['pos'] - 1  # 0-based position
    ref = variant['ref']
    alt = variant['alt']
    
    # Get context (3 bases before and after)
    context_start = max(0, pos_0 - 3)
    context_end = min(len(sequence), pos_0 + len(ref) + 3)
    context = sequence[context_start:context_end]
    
    # Calculate positions for the variant in the context
    var_start = pos_0 - context_start
    var_end = var_start + len(ref)
    
    # Create a visual representation of the variant in context
    pointer = ' ' * var_start + '^' * len(ref)
    
    # Create the alternate sequence in context
    alt_context = context[:var_start] + alt + context[var_end:]
    
    return (
        f"pos={variant['pos']} ref='{ref}' alt='{alt}'\n"
        f"  ref: {context}\n"
        f"       {pointer}\n"
        f"  alt: {alt_context}"
    )

def rescue_insertion_variant(sequence: str, pos: int, ref: str, alt: str, window_start: int) -> tuple:
    """
    Try to rescue an insertion variant by finding a better position match.
    
    Args:
        sequence: Window sequence
        pos: Original 1-based position in reference coordinates
        ref: Reference allele
        alt: Alternate allele (insertion)
        window_start: 0-based window start position in reference coordinates
        
    Returns:
        Tuple of (new_pos, new_ref, new_alt) with adjusted coordinates if rescue successful,
        or (pos, ref, alt) if no rescue needed or possible
    """
    # Skip if not an insertion
    if len(ref) >= len(alt):
        return pos, ref, alt
    
    # Convert to window coordinates
    pos_0 = pos - 1 - window_start  # 0-based position in window
    
    # Skip if outside window
    if pos_0 < 0 or pos_0 >= len(sequence):
        return pos, ref, alt
    
    # Check if reference already matches
    ref_in_seq = sequence[pos_0:pos_0+len(ref)]
    if ref_in_seq == ref:
        return pos, ref, alt  # No rescue needed
    
    # Try shifting position by ±1 base
    for offset in [-1, 1]:
        test_pos = pos_0 + offset
        
        # Skip if out of bounds
        if test_pos < 0 or test_pos + len(ref) > len(sequence):
            continue
        
        test_ref = sequence[test_pos:test_pos+len(ref)]
        
        # If we found an exact match
        if test_ref == ref:
            # Convert back to reference coordinates (1-based)
            new_pos = test_pos + 1 + window_start
            return new_pos, ref, alt
    
    # If no exact match found, try to use the actual reference
    new_ref = ref_in_seq
    new_alt = alt
    
    # For insertions, try to maintain the correct inserted sequence
    if len(ref) > 0 and len(ref_in_seq) > 0 and len(alt) > 0:
        # If ref is one base and alt starts with that base (common format)
        if len(ref) == 1 and alt.startswith(ref):
            # Replace the first base of alt with actual ref base
            new_alt = ref_in_seq + alt[1:]
    
    return pos, new_ref, new_alt

def apply_variants_to_sequence(sequence: str, variants: List[Dict[str, Any]], 
                              strict_ref_check: bool = False) -> str:
    """
    Apply multiple variants to a reference sequence in a single left-to-right pass.
    
    Args:
        sequence: Reference sequence
        variants: List of variant dictionaries with 'pos', 'ref', and 'alt' keys
                (pos is 1-based, ref/alt are strings)
        strict_ref_check: If True, raises error on reference mismatches; if False, tries to continue
    
    Returns:
        Sequence with all variants applied
    
    Raises:
        VariantError: If variants overlap or reference sequence doesn't match (with strict_ref_check=True)
    """
    if not variants:
        return sequence
    
    # Sort variants by position
    variants = sorted(variants, key=lambda v: v['pos'])
    
    result = []
    i = 0  # Current position in the reference (0-based)
    seq_len = len(sequence)
    
    # Track variant statistics
    mismatch_counts = {"SNP": 0, "INS": 0, "DEL": 0, "TOTAL": 0}
    
    for idx, v in enumerate(variants, 1):
        pos_1 = v['pos']  # 1-based position
        pos_0 = pos_1 - 1  # 0-based position
        ref = v['ref']
        # Handle case where alt is a list (multiple alternate alleles)
        alt = v['alt'][0] if isinstance(v['alt'], (list, tuple)) else v['alt']
        
        # Determine variant type
        var_type = "SNP"
        if len(ref) > len(alt):
            var_type = "DEL"
        elif len(ref) < len(alt):
            var_type = "INS"
        
        # Check if position is within sequence bounds
        if pos_0 >= seq_len:
            raise VariantError(
                f"Variant position {pos_1} is outside the reference sequence (length: {seq_len})",
                variant=v
            )
        
        # Check for overlapping variants
        if pos_0 < i:
            # Calculate overlap details
            overlap = i - pos_0
            raise VariantError(
                f"Variant at position {pos_1} overlaps with previous variant by {overlap} base(s)",
                variant=v
            )
        
        # Add sequence up to this variant
        result.append(sequence[i:pos_0])
        
        # Get the reference sequence at this position
        ref_end = min(pos_0 + len(ref), seq_len)
        ref_in_sequence = sequence[pos_0:ref_end]
        
        # Check reference allele matches
        if ref_in_sequence != ref:
            mismatch_counts[var_type] += 1
            mismatch_counts["TOTAL"] += 1
            
            if strict_ref_check:
                # Get some context around the mismatch
                context_start = max(0, pos_0 - 5)
                context_end = min(seq_len, ref_end + 5)
                context = sequence[context_start:context_end]
                
                # Show the expected and actual reference sequence
                pointer = ' ' * (pos_0 - context_start) + '^' * len(ref)
                
                raise VariantError(
                    f"Reference allele mismatch at position {pos_1} ({var_type}):\n"
                    f"  Expected: {ref}\n"
                    f"  Found:    {ref_in_sequence}\n"
                    f"  Context:  {context}\n"
                    f"            {pointer}",
                    variant=v
                )
            else:
                # In non-strict mode, handle mismatches more gracefully
                if var_type == "INS":
                    # For insertions with mismatched reference, try to preserve insertion
                    # as long as the first base matches (or we can make it match)
                    if len(ref) > 0 and len(ref_in_sequence) > 0:
                        if ref[0] == ref_in_sequence[0]:
                            # First base matches, apply insertion
                            pass
                        else:
                            # Mismatch, but we'll use actual reference base and insert after it
                            if len(alt) > 0:
                                # Replace first alt base with actual ref base
                                alt = ref_in_sequence[0] + alt[1:] if len(alt) > 1 else ref_in_sequence[0]
                    ref = ref_in_sequence  # Use actual reference
                elif var_type == "DEL":
                    # For deletions, use actual reference sequence
                    ref = ref_in_sequence
        
        # Apply the variant
        result.append(alt)
        i = ref_end
    
    # Add remaining sequence
    result.append(sequence[i:])
    
    # Print info about mismatches if any were encountered in non-strict mode
    if not strict_ref_check and mismatch_counts["TOTAL"] > 0:
        print(f"WARNING: Applied variants with {mismatch_counts['TOTAL']} reference mismatches "
              f"(SNP: {mismatch_counts['SNP']}, "
              f"INS: {mismatch_counts['INS']}, "
              f"DEL: {mismatch_counts['DEL']})")
    
    return ''.join(result)


def apply_variant_to_sequence(sequence: str, pos: int, ref_allele: str, alt_allele: str) -> str:
    """
    Apply a single variant to a reference sequence.
    
    This is a convenience wrapper around apply_variants_to_sequence for backward compatibility.
    
    Args:
        sequence: Reference sequence
        pos: 1-based position of the variant
        ref_allele: Reference allele sequence
        alt_allele: Alternate allele sequence
        
    Returns:
        Sequence with the variant applied
    """
    return apply_variants_to_sequence(sequence, [{
        'pos': pos,
        'ref': ref_allele,
        'alt': alt_allele
    }])

def build_variant_channels(
    sequence: str, 
    variants: List[Tuple[int, str, str]],  # List of (pos, ref, alt)
    window_start: int,
    window_end: int
) -> Dict[str, List[float]]:
    """
    Build variant annotation channels for a sequence window.
    
    Args:
        sequence: Reference sequence for the window
        variants: List of (pos, ref, alt) tuples for variants in this window
        window_start: Start position of the window (0-based)
        window_end: End position of the window (0-based, exclusive)
        
    Returns:
        Dictionary with variant channel annotations:
        - snp: 1.0 for SNP positions, 0.0 otherwise
        - ins: 1.0 for insertion positions, 0.0 otherwise
        - del_: 1.0 for deletion positions, 0.0 otherwise
        - any: 1.0 for any variant position, 0.0 otherwise
    """
    # Initialize empty channels
    seq_len = window_end - window_start
    channels = {
        'snp': [0.0] * seq_len,
        'ins': [0.0] * seq_len,
        'del_': [0.0] * seq_len,
        'any': [0.0] * seq_len
    }
    
    for pos, ref, alt in variants:
        # Convert to window coordinates
        pos_in_window = pos - 1 - window_start  # Convert to 0-based and adjust for window
        
        # Skip if variant is outside the window
        if pos_in_window < 0 or pos_in_window >= seq_len:
            continue
            
        # Get variant type
        var_info = encode_variant(ref, alt)
        
        # Update channels based on variant type
        if var_info['is_snp']:
            channels['snp'][pos_in_window] = 1.0
        elif var_info['is_ins']:
            channels['ins'][pos_in_window] = 1.0
        elif var_info['is_del']:
            # For deletions, mark all deleted bases
            for i in range(pos_in_window, min(pos_in_window + len(ref), seq_len)):
                channels['del_'][i] = 1.0
        
        # Update 'any' channel for all variant types
        channels['any'][pos_in_window] = 1.0
    
    return channels
