"""Helper methods for BetaDogmaDataset - to be integrated into train.py"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# Amino acid vocabulary
AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY*'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
AA_TO_IDX['<PAD>'] = len(AA_VOCAB)

def parse_isoforms(isoforms_json: str) -> Dict:
    """Parse isoform metadata from JSON string.
    
    Returns dict with:
        - proteins: List of protein sequences
        - nmd_flags: List of NMD flags
        - tpms: List of expression levels
        - cds_coords: List of (start, end) tuples
        - is_canonical: List of canonical flags
    """
    if not isoforms_json or isoforms_json == '[]':
        return {
            'proteins': [],
            'nmd_flags': [],
            'tpms': [],
            'cds_coords': [],
            'is_canonical': []
        }
    
    try:
        isoforms = json.loads(isoforms_json)
        
        return {
            'proteins': [iso.get('protein_seq', '') for iso in isoforms],
            'nmd_flags': [iso.get('has_nmd', False) for iso in isoforms],
            'tpms': [iso.get('expression_tpm', 0.0) for iso in isoforms],
            'cds_coords': [(iso.get('cds_start', -1), iso.get('cds_end', -1)) for iso in isoforms],
            'is_canonical': [iso.get('is_canonical', False) for iso in isoforms]
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse isoforms: {e}")
        return {
            'proteins': [],
            'nmd_flags': [],
            'tpms': [],
            'cds_coords': [],
            'is_canonical': []
        }

def create_protein_labels(protein_seq: str, cds_start: int, cds_end: int, seq_len: int) -> np.ndarray:
    """Convert protein sequence to per-position amino acid labels.
    
    Args:
        protein_seq: Protein sequence (amino acids)
        cds_start: CDS start position (relative to window)
        cds_end: CDS end position (relative to window)
        seq_len: Total sequence length
        
    Returns:
        Array of shape (seq_len,) with amino acid indices (-1 = ignore, 0-19 = AA, 20 = stop)
    """
    # Initialize with -1 (ignore index for loss computation)
    labels = np.full(seq_len, -1, dtype=np.int64)
    
    # If no valid CDS or protein, return all ignore
    if not protein_seq or cds_start < 0 or cds_end < 0 or cds_end > seq_len:
        return labels
    
    # Label each codon position with its amino acid
    cds_len = cds_end - cds_start
    codon_positions = range(cds_start, min(cds_end, seq_len), 3)
    
    for i, pos in enumerate(codon_positions):
        if i >= len(protein_seq):
            break
            
        aa = protein_seq[i]
        if aa in AA_TO_IDX:
            aa_idx = AA_TO_IDX[aa]
            # Label all 3 positions of the codon with the same AA
            for j in range(3):
                if pos + j < seq_len:
                    labels[pos + j] = aa_idx
    
    return labels

def create_cds_boundary_labels(cds_start: int, cds_end: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create binary labels for CDS start and end positions.
    
    Args:
        cds_start: CDS start position (relative to window), can be None
        cds_end: CDS end position (relative to window), can be None
        seq_len: Total sequence length
        
    Returns:
        Tuple of (cds_start_labels, cds_end_labels), each of shape (seq_len,)
    """
    start_labels = np.zeros(seq_len, dtype=np.float32)
    end_labels = np.zeros(seq_len, dtype=np.float32)
    
    if cds_start is not None and 0 <= cds_start < seq_len:
        start_labels[cds_start] = 1.0
    
    if cds_end is not None and 0 <= cds_end < seq_len:
        end_labels[cds_end] = 1.0
    
    return start_labels, end_labels

def extract_canonical_isoform(isoform_data: Dict) -> Dict:
    """Extract canonical isoform data from parsed isoforms.
    
    Args:
        isoform_data: Dict from parse_isoforms()
        
    Returns:
        Dict with canonical isoform data:
            - protein: Protein sequence
            - nmd: NMD flag
            - tpm: Expression level
            - cds_start, cds_end: CDS coordinates
    """
    # Find canonical isoform
    canonical_idx = None
    for i, is_canon in enumerate(isoform_data['is_canonical']):
        if is_canon:
            canonical_idx = i
            break
    
    # If no canonical, use highest expressed
    if canonical_idx is None and isoform_data['tpms']:
        canonical_idx = np.argmax(isoform_data['tpms'])
    
    # Extract canonical data
    if canonical_idx is not None and canonical_idx < len(isoform_data['proteins']):
        cds_start, cds_end = isoform_data['cds_coords'][canonical_idx]
        return {
            'protein': isoform_data['proteins'][canonical_idx],
            'nmd': isoform_data['nmd_flags'][canonical_idx],
            'tpm': isoform_data['tpms'][canonical_idx],
            'cds_start': cds_start,
            'cds_end': cds_end
        }
    
    # Return empty if no isoforms
    return {
        'protein': '',
        'nmd': False,
        'tpm': 0.0,
        'cds_start': -1,
        'cds_end': -1
    }

def to_tensor(data, length: int) -> torch.Tensor:
    """Convert data to tensor with proper padding/truncation.
    
    Args:
        data: List, numpy array, or JSON string
        length: Target length
        
    Returns:
        Tensor of shape (length,)
    """
    # Parse JSON if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return torch.zeros(length, dtype=torch.float32)
    
    # Convert to numpy
    if isinstance(data, (list, np.ndarray)):
        data = np.asarray(data, dtype=np.float32)
        
        # Truncate if too long (center crop)
        if len(data) > length:
            start = (len(data) - length) // 2
            data = data[start:start + length]
        
        # Pad if too short
        elif len(data) < length:
            pad = (0, length - len(data))
            data = np.pad(data, pad, 'constant')
        
        return torch.from_numpy(data)
    
    return torch.zeros(length, dtype=torch.float32)
