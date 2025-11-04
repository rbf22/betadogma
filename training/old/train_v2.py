#!/usr/bin/env python3
"""
train_v2.py - Phase 1: Central Dogma Modeling with Rich Isoform Data

Phase 1 Features:
- DNA → RNA: TSS, polyA, expression prediction
- RNA → Splicing: Donor/acceptor sites  
- RNA → Protein: Protein sequence prediction, CDS boundaries
- Quality Control: NMD prediction
- Optimized for 300k sequences (matches data generation)

Phase 2 (Next): Variant augmentation and effect prediction
"""

import os
import json
import random
import time
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import random

# Add parent directory to path for local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import warnings
import yaml
from pathlib import Path
warnings.filterwarnings('ignore')

# Load configuration from YAML
CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


# ============================================================================
# GPU Detection with Memory-Optimized Configs
# ============================================================================

def get_gpu_config():
    """Auto-detect GPU and return memory-optimized config."""
    
    if not torch.cuda.is_available():
        return {
            'max_seq_len': 1000,
            'batch_size': 1,
            'accumulate_grad_batches': 64,
            'device_name': 'CPU',
            'use_gradient_checkpointing': False,
        }
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    gpu_memory_gb = gpu_props.total_memory / (1024**3)
    
    print(f"\n🔍 Detected GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb >= 75:  # A100-80GB
        config = {
            'max_seq_len': 450000,  # Full 450k!
            'batch_size': 2,
            'accumulate_grad_batches': 8,
            'device_name': 'A100-80GB',
            'use_gradient_checkpointing': False,
        }
        print("   ✅ A100-80GB: Full 450k sequences, batch_size=2")
        
    elif gpu_memory_gb >= 35:  # A100-40GB
        config = {
            'max_seq_len': 450000,  # Full 450k!
            'batch_size': 1,
            'accumulate_grad_batches': 16,
            'device_name': 'A100-40GB',
            'use_gradient_checkpointing': True,
        }
        print("   ✅ A100-40GB: Full 450k sequences with gradient checkpointing")
        
    elif gpu_memory_gb >= 14:  # T4, RTX 3080 (16GB)
        config = {
            'max_seq_len': 450000,  # YES! Full 450k!
            'batch_size': 1,
            'accumulate_grad_batches': 32,
            'device_name': 'T4/RTX3080',
            'use_gradient_checkpointing': True,  # Critical!
            'empty_cache_freq': 1,  # Clear cache every batch
        }
        print("   ✅ T4: Full 450k sequences (with optimizations)")
        print("      - Frozen encoder")
        print("      - Gradient checkpointing enabled")
        print("      - Aggressive memory management")
        
    else:  # <12GB
        config = {
            'max_seq_len': 100000,  # Reduced
            'batch_size': 1,
            'accumulate_grad_batches': 48,
            'device_name': 'Small GPU',
            'use_gradient_checkpointing': True,
        }
        print("   ⚠️  Limited memory: 100k sequences max")
    
    print(f"   Effective batch size: {config['batch_size'] * config['accumulate_grad_batches']}")
    return config


class Config:
    """Configuration class that loads settings from config.yaml."""
    def __init__(self):
        # GPU Configuration
        self.gpu_config = get_gpu_config()
        
        # Paths
        self.data_dir = Path(__file__).parent.parent / CONFIG['data']['data_dir']
        self.output_dir = Path(__file__).parent.parent / CONFIG['output']['output_dir']
        
        # Model architectures
        model_cfg = CONFIG['model']
        self.model_name = model_cfg['name']
        self.hidden_size = model_cfg.get('hidden_size', 768)
        self.num_layers = model_cfg.get('num_layers', 24)
        self.max_seq_len = model_cfg['max_seq_len']
        self.splice_hidden = 128
        self.splice_layers = 1
        self.tss_hidden = 64
        self.tss_layers = 1
        self.polya_hidden = 64
        self.polya_layers = 1
        
        # NEW Phase 1: Protein prediction head dimensions
        self.protein_hidden = 256
        self.protein_layers = 2
        
        self.dropout = 0.1
        
        # Memory optimization
        self.use_gradient_checkpointing = model_cfg['use_gradient_checkpointing']
        self.empty_cache_freq = self.gpu_config.get('empty_cache_freq', 0)
        
        # Loss weights
        self.w_splice_donor = 1.0
        self.w_splice_acceptor = 1.0
        self.w_tss = 0.5
        self.w_polya = 0.5
        self.w_splice_effect = 1.0
        
        # NEW Phase 1: Loss weights for new tasks
        self.w_protein = 2.0          # Protein sequence prediction
        self.w_cds_start = 0.5        # CDS start boundary
        self.w_cds_end = 0.5          # CDS end boundary
        self.w_nmd = 1.0              # NMD prediction
        self.w_expression = 1.0       # Expression prediction
        
        self.pos_weight = 20.0  # For positive class in BCEWithLogitsLoss
        
        # Coupling parameters
        self.coupling_strength = 0.1  # Controls strength of coupling between tasks
        self.consistency_weight = 0.1  # Weight for consistency loss
        
        # Training configuration
        train_cfg = CONFIG['training']
        self.limit_val_batches = train_cfg['limit_val_batches']
        self.limit_train_batches = train_cfg['limit_train_batches']
        self.batch_size = train_cfg['batch_size']
        self.accumulate_grad_batches = train_cfg['accumulate_grad_batches']
        self.num_workers = train_cfg['num_workers']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.max_epochs = train_cfg['max_epochs']
        self.gradient_clip_val = train_cfg['gradient_clip_val']
        self.precision = train_cfg['precision']
        self.devices = train_cfg['devices']
        self.save_top_k = train_cfg.get('save_top_k', 1)
        self.monitor = train_cfg.get('monitor', 'val/loss/total')
        self.mode = train_cfg.get('mode', 'min')
        self.patience = train_cfg.get('patience', 5)
        self.freeze_encoder = train_cfg.get('freeze_encoder', True)
        
        # Set encoder dimension based on model name
        if 'medium' in self.model_name:
            self.encoder_dim = 768
        elif 'large' in self.model_name:
            self.encoder_dim = 1024
        else:  # small or base
            self.encoder_dim = 256


# ============================================================================
# Character Tokenizer
# ============================================================================

class CharacterTokenizer:
    """Character-level DNA tokenizer."""
    
    def __init__(self, max_length: int = 300000):
        self.max_length = max_length  # Phase 1: 300k sequences
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.pad_token_id = 4
        
    def __call__(self, sequence: str, return_tensors: str = "pt", 
                 padding: str = "max_length", max_length: int = None,
                 truncation: bool = True):
        if max_length is None:
            max_length = self.max_length
        
        sequence = sequence.upper()
        
        if truncation and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        tokens = [self.vocab.get(char, 4) for char in sequence]
        attention_mask = [1] * len(tokens)
        
        if padding == "max_length":
            pad_length = max_length - len(tokens)
            if pad_length > 0:
                tokens = tokens + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        if return_tensors == "pt":
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }


# ============================================================================
# Dataset
# ============================================================================

class BetaDogmaDataset(Dataset):
    """Modernized dataset for Central Dogma modeling with rich isoform and variant data."""
    
    # Amino acid vocabulary (20 AA + stop + padding)
    AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY*'
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
    AA_TO_IDX['<PAD>'] = len(AA_VOCAB)
    
    def __init__(
        self, 
        parquet_files: List[Path],
        tokenizer,
        max_seq_len: int = 300000,  # Updated to match data generation
        mode: str = "train",
        augment_prob: float = 0.0,  # Variant augmentation probability
        seed: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.augment_prob = augment_prob
        self.seed = seed or 42
        
        # Store file paths
        self.parquet_files = [str(Path(f).resolve()) for f in parquet_files if Path(f).exists()]
        if not self.parquet_files:
            raise ValueError(f"No valid data files found for {mode}")
            
        print(f"\n{'='*60}")
        print(f"Loading {mode} dataset")
        print(f"{'='*60}")
        print(f"Found {len(self.parquet_files)} {mode} files")
        
        # Pre-load all file metadata
        self.file_metas = []
        total_examples = 0
        
        for f in self.parquet_files:
            try:
                with open(f, 'rb') as pf:
                    parquet_file = pq.ParquetFile(pf)
                    num_rows = parquet_file.metadata.num_rows
                    
                    self.file_metas.append({
                        'path': f,
                        'start_idx': total_examples,
                        'end_idx': total_examples + num_rows - 1,
                        'num_rows': num_rows
                    })
                    
                    total_examples += num_rows
                    print(f"  ✓ {Path(f).name}: {num_rows:,} examples")
                    
            except Exception as e:
                print(f"  ✗ {Path(f).name}: {e}")
                continue
                
        self.length = total_examples
        print(f"\n  Total: {self.length:,} examples")
        print(f"  Max sequence length: {self.max_seq_len:,} bp")
        print(f"  Variant augmentation: {self.augment_prob*100:.1f}%")
        print(f"{'='*60}\n")
        
        # For deterministic behavior in val/test
        if self.mode in ['val', 'test']:
            np.random.seed(self.seed)
            random.seed(self.seed)
    
    def __len__(self):
        return self.length
    
    def _get_file_and_row(self, idx):
        """Find which file and row contains the given index."""
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds [0, {self.length-1}]")
            
        # Binary search to find the right file
        low, high = 0, len(self.file_metas) - 1
        while low <= high:
            mid = (low + high) // 2
            meta = self.file_metas[mid]
            
            if idx < meta['start_idx']:
                high = mid - 1
            elif idx > meta['end_idx']:
                low = mid + 1
            else:
                # Found the file
                row_idx = idx - meta['start_idx']
                return meta['path'], row_idx
                
        raise IndexError(f"Could not find index {idx} in any file")
    
    @staticmethod
    def parse_isoforms(isoforms_json: str) -> Dict:
        """Parse isoform metadata from JSON string."""
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
        except (json.JSONDecodeError, KeyError, TypeError):
            return {
                'proteins': [],
                'nmd_flags': [],
                'tpms': [],
                'cds_coords': [],
                'is_canonical': []
            }
    
    @staticmethod
    def extract_canonical_isoform(isoform_data: Dict) -> Dict:
        """Extract canonical isoform data from parsed isoforms."""
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
    
    @classmethod
    def create_protein_labels(cls, protein_seq: str, cds_start: int, cds_end: int, seq_len: int) -> np.ndarray:
        """Convert protein sequence to per-position amino acid labels."""
        # Initialize with -1 (ignore index for loss computation)
        labels = np.full(seq_len, -1, dtype=np.int64)
        
        # If no valid CDS or protein, return all ignore
        if not protein_seq or cds_start is None or cds_end is None or cds_start < 0 or cds_end < 0 or cds_end > seq_len:
            return labels
        
        # Label each codon position with its amino acid
        codon_positions = range(cds_start, min(cds_end, seq_len), 3)
        
        for i, pos in enumerate(codon_positions):
            if i >= len(protein_seq):
                break
                
            aa = protein_seq[i]
            if aa in cls.AA_TO_IDX:
                aa_idx = cls.AA_TO_IDX[aa]
                # Label all 3 positions of the codon with the same AA
                for j in range(3):
                    if pos + j < seq_len:
                        labels[pos + j] = aa_idx
        
        return labels
    
    @staticmethod
    def create_cds_boundary_labels(cds_start: int, cds_end: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create binary labels for CDS start and end positions."""
        start_labels = np.zeros(seq_len, dtype=np.float32)
        end_labels = np.zeros(seq_len, dtype=np.float32)
        
        if cds_start is not None and 0 <= cds_start < seq_len:
            start_labels[cds_start] = 1.0
        
        if cds_end is not None and 0 <= cds_end < seq_len:
            end_labels[cds_end] = 1.0
        
        return start_labels, end_labels
    
    @staticmethod
    def to_tensor(data, length: int) -> torch.Tensor:
        """Convert data to tensor with proper padding/truncation."""
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
    
    def _apply_synthetic_variant(self, seq: str, seed: int) -> tuple:
        """Apply random variant (SNV/INS/DEL) to sequence.
        
        Args:
            seq: Reference sequence
            seed: Random seed for reproducibility
            
        Returns:
            (mutated_sequence, variant_info_dict)
        """
        rng = np.random.RandomState(seed)
        
        seq_len = len(seq)
        if seq_len < 200:
            return seq, None
        
        # Pick random position (avoid edges)
        pos = rng.randint(100, seq_len - 100)
        
        # Choose variant type: 70% SNV, 15% INS, 15% DEL
        var_type = rng.choice(['SNV', 'INS', 'DEL'], p=[0.7, 0.15, 0.15])
        
        bases = ['A', 'C', 'G', 'T']
        ref_base = seq[pos]
        
        if ref_base not in bases:
            return seq, None
        
        if var_type == 'SNV':
            # Simple substitution
            alt_base = rng.choice([b for b in bases if b != ref_base])
            ref = ref_base
            alt = alt_base
        elif var_type == 'INS':
            # Insert 1-5 random bases
            ins_len = rng.randint(1, 6)
            alt = ''.join(rng.choice(bases) for _ in range(ins_len))
            ref = ref_base
            # Insert after the anchor base
            alt = ref + alt
        else:  # DEL
            # Delete 1-5 bases
            del_len = min(rng.randint(1, 6), len(seq) - pos - 1)
            ref = ref_base + seq[pos+1:pos+1+del_len]
            alt = ref_base
        
        # Apply mutation
        seq_list = list(seq)
        
        if var_type == 'SNV':
            seq_list[pos] = alt
        elif var_type == 'INS':
            # Insert after the anchor base
            seq_list[pos+1:pos+1] = list(alt[1:])  # alt includes the anchor base
        else:  # DEL
            # Remove deleted bases
            del seq_list[pos+1:pos+1+len(ref)-1]
        
        seq_alt = ''.join(seq_list)
        
        variant_info = {
            'pos': pos,
            'ref': ref,
            'alt': alt,
            'type': f'synthetic_{var_type.lower()}',
            'pathogenic': False,
        }
        
        return seq_alt, variant_info
    
    def _apply_clinvar_variant(self, seq: str, variant: Dict) -> tuple:
        """Apply ClinVar variant to sequence.
        
        Args:
            seq: Reference sequence
            variant: ClinVar variant dict with 'pos', 'ref', 'alt', 'type', 'pathogenic'
            
        Returns:
            (mutated_sequence, variant_info_dict)
        """
        pos = variant['pos']
        ref = variant['ref']
        alt = variant['alt']
        var_type = variant.get('type', 'SNV')
        
        if pos < 0 or pos >= len(seq):
            return seq, None
        
        # Verify reference matches for the first base (anchor)
        if seq[pos] != ref[0] if ref else False:
            return seq, None
        
        # Apply mutation based on variant type
        seq_list = list(seq)
        
        if var_type == 'SNV':
            # Simple substitution
            seq_list[pos] = alt
        elif var_type == 'INS':
            # Insertion: insert after the anchor base
            seq_list[pos+1:pos+1] = list(alt)
        elif var_type == 'DEL':
            # Deletion: remove reference bases after anchor
            del_len = len(ref)
            seq_list[pos+1:pos+1+del_len] = []
        
        seq_alt = ''.join(seq_list)
        
        variant_info = {
            'pos': pos,
            'ref': ref,
            'alt': alt,
            'type': f'clinvar_{var_type.lower()}',
            'pathogenic': variant['pathogenic'],
        }
        
        return seq_alt, variant_info
    
    def _get_file_and_row(self, idx):
        """Find which file and row contains the given index."""
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds [0, {self.length-1}]")
            
        # Binary search to find the right file
        low, high = 0, len(self.file_metas) - 1
        while low <= high:
            mid = (low + high) // 2
            meta = self.file_metas[mid]
            
            if idx < meta['start_idx']:
                high = mid - 1
            elif idx > meta['end_idx']:
                low = mid + 1
            else:
                # Found the file
def _apply_synthetic_variant(self, seq: str, seed: int) -> tuple:
    """Apply random variant (SNV/INS/DEL) to sequence.
        
    Args:
        seq: Reference sequence
        seed: Random seed for reproducibility
            
    Returns:
        (mutated_sequence, variant_info_dict)
    """
    rng = np.random.RandomState(seed)
        
    seq_len = len(seq)
    if seq_len < 200:
        return seq, None
        
    # Pick random position (avoid edges)
    pos = rng.randint(100, seq_len - 100)
        
    # Choose variant type: 70% SNV, 15% INS, 15% DEL
    var_type = rng.choice(['SNV', 'INS', 'DEL'], p=[0.7, 0.15, 0.15])
        
    bases = ['A', 'C', 'G', 'T']
    ref_base = seq[pos]
        
    if ref_base not in bases:
        return seq, None
        
    if var_type == 'SNV':
        # Simple substitution
        alt_base = rng.choice([b for b in bases if b != ref_base])
        ref = ref_base
        alt = alt_base
    elif var_type == 'INS':
        # Insert 1-5 random bases
        ins_len = rng.randint(1, 6)
        alt = ''.join(rng.choice(bases) for _ in range(ins_len))
        ref = ref_base
        # Insert after the anchor base
        alt = ref + alt
    else:  # DEL
        # Delete 1-5 bases
        del_len = min(rng.randint(1, 6), len(seq) - pos - 1)
        ref = ref_base + seq[pos+1:pos+1+del_len]
        alt = ref_base
        
    # Apply mutation
    seq_list = list(seq)
        
    if var_type == 'SNV':
        seq_list[pos] = alt
    elif var_type == 'INS':
        # Insert after the anchor base
        seq_list[pos+1:pos+1] = list(alt[1:])  # alt includes the anchor base
    else:  # DEL
        # Remove deleted bases
        del seq_list[pos+1:pos+1+len(ref)-1]
        
    seq_alt = ''.join(seq_list)
        
    variant_info = {
        'pos': pos,
        'ref': ref,
        'alt': alt,
        'type': f'synthetic_{var_type.lower()}',
        'pathogenic': False,
    }
        
    return seq_alt, variant_info
    
def _apply_clinvar_variant(self, seq: str, variant: Dict) -> tuple:
    """Apply ClinVar variant to sequence.
    
    Args:
        seq: Reference sequence
        variant: ClinVar variant dict with 'pos', 'ref', 'alt', 'type', 'pathogenic'
            
    Returns:
        (mutated_sequence, variant_info_dict)
    """
    pos = variant['pos']
    ref = variant['ref']
    alt = variant['alt']
    var_type = variant.get('type', 'SNV')
        
    if pos < 0 or pos >= len(seq):
        return seq, None
        
    # Verify reference matches for the first base (anchor)
    if seq[pos] != ref[0] if ref else False:
        return seq, None
        
    # Apply mutation based on variant type
    seq_list = list(seq)
        
    if var_type == 'SNV':
        # Simple substitution
        seq_list[pos] = alt
    elif var_type == 'INS':
        # Insertion: insert after the anchor base
        seq_list[pos+1:pos+1] = list(alt)
    elif var_type == 'DEL':
        # Deletion: remove reference bases after anchor
        del_len = len(ref)
        seq_list[pos+1:pos+1+del_len] = []
        
    seq_alt = ''.join(seq_list)
        
    variant_info = {
        'pos': pos,
        'ref': ref,
        'alt': alt,
        'type': f'clinvar_{var_type.lower()}',
        'pathogenic': variant['pathogenic'],
    }
        
    return seq_alt, variant_info
    
def _get_file_and_row(self, idx):
    """Find which file and row contains the given index."""
    if idx < 0 or idx >= self.length:
        raise IndexError(f"Index {idx} out of bounds [0, {self.length-1}]")
            
    # Binary search to find the right file
    low, high = 0, len(self.file_metas) - 1
    while low <= high:
        mid = (low + high) // 2
        meta = self.file_metas[mid]
            
        if idx < meta['start_idx']:
            high = mid - 1
        elif idx > meta['end_idx']:
            low = mid + 1
        else:
            # Found the file
            row_idx = idx - meta['start_idx']
            return meta['path'], row_idx
                
    raise IndexError(f"Could not find index {idx} in any file")

@staticmethod
def to_tensor(data, length):
    """Convert data to tensor with proper padding/truncation."""
    if isinstance(data, (list, np.ndarray)):
        data = np.asarray(data, dtype=np.float32)
        if len(data) > length:
            start = (len(data) - length) // 2
            data = data[start:start + length]
        elif len(data) < length:
            pad = (0, length - len(data))
            data = np.pad(data, pad, 'constant')
        return torch.from_numpy(data)
    return torch.zeros(length, dtype=torch.float32)

def __getitem__(self, idx):
    """Get a single item from the dataset."""
    # Find which file and row contains this index
    file_meta, row_idx = self._get_file_and_row(idx)
        
    try:
        # Read only the specific row we need
        df = pd.read_parquet(
            file_meta['path'],
            filters=[('index', '=', row_idx)],
            columns=['seq', 'start','end', 'donor', 'acceptor', 'tss', 'polya', 'splice_variants', 'clinvar_variants','benign_variants']
        )
            
        if len(df) == 0:
            raise IndexError(f"Row {row_idx} not found in {file_meta['path']}")
                
        row = df.iloc[0]
            
        # Process sequence
        seq = str(row['seq'])
        if not seq:
            raise ValueError(f"Empty sequence at index {idx}")
                
        # Ensure sequence is not empty after conversion
        if not seq:
            seq = 'N' * self.max_seq_len
                
        # Truncate sequence if needed (center crop)
        if len(seq) > self.max_seq_len:
            start = (len(seq) - self.max_seq_len) // 2
            seq = seq[start:start + self.max_seq_len]
                
        # Ensure sequence is exactly max_seq_len (pad if needed)
        if len(seq) < self.max_seq_len:
            # Pad with Ns if sequence is too short
            pad_left = (self.max_seq_len - len(seq)) // 2
            pad_right = self.max_seq_len - len(seq) - pad_left
            seq = ('N' * pad_left) + seq + ('N' * pad_right)
                
        # Tokenize the sequence
        seq = self.tokenizer(
            seq,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
            
        # Convert labels to tensors
        labels = {}
        for label_type in ['donor', 'acceptor', 'tss', 'polya']:
            if label_type in row and row[label_type] is not None:
                labels[label_type] = self.to_tensor(
                    row[label_type],
                    self.max_seq_len
                )
            
        return {
            'input_ids': seq['input_ids'].squeeze(0),
            'attention_mask': seq['attention_mask'].squeeze(0),
            'labels': labels
        }
            
    except Exception as e:
        print(f"Error in __getitem__ for index {idx}: {e}")
        # Return dummy data on error
        seq = self.tokenizer(
            'N' * self.max_seq_len,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
            
        return {
            'input_ids': seq['input_ids'].squeeze(0),
            'attention_mask': seq['attention_mask'].squeeze(0),
            'labels': {
                'donor': torch.zeros(self.max_seq_len, dtype=torch.float32),
                'acceptor': torch.zeros(self.max_seq_len, dtype=torch.float32),
                'tss': torch.zeros(self.max_seq_len, dtype=torch.float32),
                'polya': torch.zeros(self.max_seq_len, dtype=torch.float32),
            }
        }


# ============================================================================
# Model with Gradient Checkpointing
# ============================================================================

class PredictionHead(nn.Module):
    """Prediction head with optional gradient checkpointing."""
    
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int, 
                 dropout: float = 0.1, use_checkpointing: bool = False):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        
        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def _forward_impl(self, x):
        """Actual forward computation."""
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out).squeeze(-1)
        return logits
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class HyenaDNAEncoder(nn.Module):
    """HyenaDNA encoder with improved model loading and error handling."""
    
    def __init__(self, model_name: str, freeze: bool = True, device: str = None):
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing HyenaDNA model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            # Import required modules
            from transformers import AutoModel, AutoConfig
            
            # Load config first to verify model type
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"Model config: {config.model_type}")
            
            # Load model with appropriate settings
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map='auto' if str(self.device) != 'cpu' else None,
            ).to(self.device)
            
            print(f"✅ Successfully loaded model: {model_name}")
            print(f"Model device: {next(self.model.parameters()).device}")
            
            # Freeze parameters if needed
            self.frozen = freeze
            if freeze:
                print("✅ Freezing encoder parameters")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            else:
                print("✅ Keeping encoder parameters trainable")
                
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def forward(self, input_ids, attention_mask=None):
        try:
            # Ensure input is on the correct device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass with appropriate gradient context
            with torch.set_grad_enabled(not self.frozen):
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Handle different output formats
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                elif isinstance(outputs, (tuple, list)):
                    embeddings = outputs[0]  # First output is typically the hidden states
                else:
                    embeddings = outputs
                
                return embeddings
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("⚠️ CUDA out of memory. Try reducing batch size or sequence length.")
            print(f"❌ Forward pass failed: {str(e)}")
            raise


class BetaDogmaModel(nn.Module):
    """BetaDogma model with memory optimizations and splice effect prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing BetaDogmaModel on {self.device}")
        
        # Initialize encoder with proper device handling
        self.encoder = HyenaDNAEncoder(
            model_name=config.model_name,
            freeze=config.freeze_encoder,
            device=self.device
        )
        
        # Get the actual encoder dimension from the model
        if hasattr(self.encoder.model.config, 'hidden_size'):
            self.encoder_dim = self.encoder.model.config.hidden_size
        else:
            # Fallback to config value if not available
            self.encoder_dim = getattr(config, 'encoder_dim', 768)
            print(f"⚠️ Could not determine encoder hidden size, using: {self.encoder_dim}")
        
        use_checkpointing = config.use_gradient_checkpointing
        
        # Initialize prediction heads with proper dimensions
        self.donor_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        ).to(self.device)
        
        self.acceptor_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        ).to(self.device)
        
        # Splice effect prediction head (regression)
        self.splice_effect_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        ).to(self.device)
        
        # Initialize cross-attention layers with proper device placement
        self.splice_effect_to_donor = nn.Linear(1, 1).to(self.device)
        self.splice_effect_to_acceptor = nn.Linear(1, 1).to(self.device)
        self.donor_to_effect = nn.Linear(1, 1).to(self.device)
        self.acceptor_to_effect = nn.Linear(1, 1).to(self.device)
        
        # Initialize coupling layers
        for layer in [self.splice_effect_to_donor, self.splice_effect_to_acceptor,
                     self.donor_to_effect, self.acceptor_to_effect]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        # Other prediction heads
        self.tss_head = PredictionHead(
            self.encoder_dim, 
            config.tss_hidden,
            config.tss_layers, 
            config.dropout, 
            use_checkpointing
        ).to(self.device)
        
        self.polya_head = PredictionHead(
            self.encoder_dim, 
            config.polya_hidden,
            config.polya_layers, 
            config.dropout, 
            use_checkpointing
        ).to(self.device)
        
        # Initialize weights for all heads
        for head in [self.donor_head, self.acceptor_head, self.splice_effect_head, 
                    self.tss_head, self.polya_head]:
            self._init_weights(head)
            
        print(f"✅ BetaDogmaModel initialized with encoder_dim={self.encoder_dim}")
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass with proper error handling and device management."""
        try:
            # Ensure inputs are on the correct device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Get sequence embeddings
            with torch.set_grad_enabled(self.training and not self.encoder.frozen):
                embeddings = self.encoder(input_ids, attention_mask=attention_mask)
                
                # Ensure embeddings are on the correct device
                if embeddings.device != self.device:
                    embeddings = embeddings.to(self.device)
            
            # Apply prediction heads with gradient checkpointing if needed
            def run_heads(emb):
                # Get base predictions
                donor_logits = self.donor_head(emb)
                acceptor_logits = self.acceptor_head(emb)
                tss_logits = self.tss_head(emb)
                polya_logits = self.polya_head(emb)
                splice_effect = self.splice_effect_head(emb)
                
                # Apply cross-attention between tasks
                if hasattr(self.config, 'coupling_strength'):
                    # Update donor/acceptor with splice effect information
                    effect_reshaped = splice_effect.unsqueeze(-1)
                    donor_update = self.splice_effect_to_donor(effect_reshaped).squeeze(-1)
                    acceptor_update = self.splice_effect_to_acceptor(effect_reshaped).squeeze(-1)
                    
                    donor_logits = donor_logits + self.config.coupling_strength * donor_update.detach()
                    acceptor_logits = acceptor_logits + self.config.coupling_strength * acceptor_update.detach()
                    
                    # Update splice effect with donor/acceptor information
                    donor_reshaped = torch.sigmoid(donor_logits).unsqueeze(-1)
                    acceptor_reshaped = torch.sigmoid(acceptor_logits).unsqueeze(-1)
                    
                    effect_update = (self.donor_to_effect(donor_reshaped) + 
                                  self.acceptor_to_effect(acceptor_reshaped)).squeeze(-1)
                    splice_effect = (splice_effect + self.config.coupling_strength * effect_update.detach()) / 2
                
                return {
                    'donor': donor_logits,
                    'acceptor': acceptor_logits,
                    'tss': tss_logits,
                    'polya': polya_logits,
                    'splice_effect': splice_effect
                }
            
            # Use gradient checkpointing during training if enabled
            if self.training and hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                outputs = checkpoint(run_heads, embeddings, use_reentrant=False)
            else:
                outputs = run_heads(embeddings)
            
            return outputs
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("⚠️ CUDA out of memory in forward pass. Try reducing batch size or sequence length.")
                print(f"Current batch size: {input_ids.size(0)}, sequence length: {input_ids.size(1)}")
                
                # Clear cache and try one more time
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("✅ Cleared CUDA cache, trying forward pass again...")
                    return self.forward(input_ids, attention_mask)
            
            print(f"❌ Forward pass failed: {str(e)}")
            print(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
            if hasattr(self, 'encoder') and hasattr(self.encoder, 'model'):
                print(f"Encoder device: {next(self.encoder.model.parameters()).device}")
            raise
        
        return {
            'donor': donor_logits,
            'acceptor': acceptor_logits,
            'tss': self.tss_head(outputs),
            'polya': self.polya_head(outputs),
            'splice_effect': splice_effect
        }


# ============================================================================
# Lightning Module with Memory Management
# ============================================================================

class BetaDogmaLightning(pl.LightningModule):
    """Lightning module with aggressive memory management."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.model = BetaDogmaModel(config)
        self.pos_weight = torch.tensor(config.pos_weight)
        
        # Add loss function for splice effect (regression)
        self.splice_effect_loss = nn.MSELoss()
        
        self.batch_count = 0
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def _compute_loss(self, outputs, batch):
        """Compute loss for a batch."""
        labels = batch['labels']
        
        # Binary cross-entropy for splice site prediction
        loss_donor = F.binary_cross_entropy_with_logits(
            outputs['donor'], 
            labels['donor'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_acceptor = F.binary_cross_entropy_with_logits(
            outputs['acceptor'], 
            labels['acceptor'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        # TSS and polyA site prediction
        loss_tss = F.binary_cross_entropy_with_logits(
            outputs['tss'], 
            labels['tss'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_polya = F.binary_cross_entropy_with_logits(
            outputs['polya'], 
            labels['polya'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        # Splice effect prediction (regression)
        if 'splice_effect' in outputs and 'splice_effect' in labels:
            # Only compute loss on positions with non-zero effect
            mask = (labels['splice_effect'] > 0).float()
            if mask.sum() > 0:
                loss_splice = self.splice_effect_loss(
                    outputs['splice_effect'] * mask,
                    labels['splice_effect'] * mask
                )
                
                # Consistency loss between splice effect and donor/acceptor predictions
                if self.config.consistency_weight > 0:
                    # Get sigmoid of donor/acceptor logits
                    donor_probs = torch.sigmoid(outputs['donor'])
                    acceptor_probs = torch.sigmoid(outputs['acceptor'])
                    
                    # Compute max probability of donor/acceptor at each position
                    max_probs = torch.max(donor_probs, acceptor_probs)
                    
                    # Only compute consistency where we have splice effect labels
                    consistency_loss = F.mse_loss(
                        outputs['splice_effect'] * mask,
                        max_probs.detach() * mask
                    )
                    
                    # Add to splice effect loss
                    loss_splice = loss_splice + self.config.consistency_weight * consistency_loss
            else:
                loss_splice = torch.tensor(0.0, device=self.device)
        else:
            loss_splice = torch.tensor(0.0, device=self.device)
        
        # Combine losses with weights
        loss = (
            self.config.w_splice_donor * loss_donor +
            self.config.w_splice_acceptor * loss_acceptor +
            self.config.w_tss * loss_tss +
            self.config.w_polya * loss_polya +
            self.config.w_splice_effect * loss_splice
        )
        
        return {
            'loss': loss,
            'loss/donor': loss_donor,
            'loss/acceptor': loss_acceptor,
            'loss/tss': loss_tss,
            'loss/polya': loss_polya,
            'loss/splice_effect': loss_splice,
        }
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss, loss_dict = self._compute_loss(outputs, batch)
        
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, prog_bar=True, on_step=True, on_epoch=True)
        
        # Clear cache periodically on T4
        if self.config.empty_cache_freq > 0:
            self.batch_count += 1
            if self.batch_count % self.config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss, loss_dict = self._compute_loss(outputs, batch)
        
        for k, v in loss_dict.items():
            self.log(f'val/{k}', v, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Only trainable parameters (heads only)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# ============================================================================
# Data Module
# ============================================================================

class BetaDogmaDataModule(pl.LightningDataModule):
    """Data module."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = CharacterTokenizer(max_length=config.max_seq_len)
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        data_dir = Path(self.config.data_dir)
        
        # Find all parquet files in the data directory subfolders
        train_files = list((data_dir / 'train').glob("*.parquet"))
        val_files = list((data_dir / 'val').glob("*.parquet"))
        test_files = list((data_dir / 'test').glob("*.parquet"))
        # Check if we have any data
        if not train_files and not val_files and not test_files:
            raise ValueError(
                f"No training files found in {self.config.data_dir}. "
                f"Expected to find parquet files in {data_dir}/{{train,val,test}}/"
            )
            
        if stage == "fit" or stage is None:
            if not train_files:
                raise ValueError(
                    f"No training files found in {data_dir}/train/. "
                    f"Expected parquet files in {data_dir}/train/"
                )
                
            print(f"Found {len(train_files)} training files")
            self.data_train = BetaDogmaDataset(
                train_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="train",
                augment_prob=0.8,
            )
            
            if not val_files:
                print(f"WARNING: No validation files found in {data_dir}/val/. Using training data for validation.")
                val_files = train_files
            
            print(f"Using {len(val_files)} validation files")
            self.data_val = BetaDogmaDataset(
                val_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="val",
            )
            
        if stage == "test" or stage is None:
            if not test_files:
                print(f"WARNING: No test files found in {data_dir}/test/. Using validation data for testing.")
                test_files = val_files if val_files else train_files
            
            print(f"Using {len(test_files)} test files")
            self.data_test = BetaDogmaDataset(
                test_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="test",
            )
    
    @staticmethod
    def _worker_init_fn(worker_id):
        """Worker init function for reproducibility."""
        # Get a unique seed for this worker
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        
        # Set seeds for reproducibility
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        # Set NumPy random seed for this worker
        np_seed = int(worker_seed % 2**32 - 1)
        np.random.seed(np_seed)
        
        # Set Python random seed for this worker
        py_seed = int(worker_seed % 2**32 - 2)
        random.seed(py_seed)
    
    def train_dataloader(self):
        if not hasattr(self, 'data_train'):
            self.setup(stage='fit')
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        if not hasattr(self, 'data_val'):
            self.setup(stage='validate')
        return DataLoader(
            self.data_val,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        if not hasattr(self, 'data_test'):
            self.setup(stage='test')
        return DataLoader(
            self.data_test,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True,
        )


# ============================================================================
# Training
# ============================================================================

def train(
    data_dir: str = None,
    output_dir: str = None,
    max_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    warmup_epochs: int = 1,
    num_workers: int = None,
    accelerator: str = "auto",
    devices: int = None,
    precision: str = None,
    monitor: str = None,
):
    """Train the model with the given configuration.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save model checkpoints and logs
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training/validation
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs for learning rate
        num_workers: Number of data loader workers
        accelerator: Hardware accelerator to use ('cpu', 'gpu', 'tpu', 'auto')
        devices: Number of devices to use
        precision: Training precision (16 or 32 bit)
        monitor: Metric to monitor for checkpointing
    """
    # Ensure monitor has a default value if None
    if monitor is None:
        monitor = "val_loss"
        print("Warning: monitor was None, defaulting to 'val_loss'")
    
    # Initialize configuration
    config = Config()
    
    # Override config with function arguments if provided
    if data_dir is not None:
        config.data_dir = Path(data_dir)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    if max_epochs is not None:
        config.max_epochs = max_epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if weight_decay is not None:
        config.weight_decay = weight_decay
    if num_workers is not None:
        config.num_workers = num_workers
    if devices is not None:
        config.devices = devices
    if precision is not None:
        config.precision = precision
    if monitor is not None:
        config.monitor = monitor
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    pl.seed_everything(42)
    
    print("\n" + "="*80)
    print("BETADOGMA - 450K SEQUENCES ON T4!")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {config.gpu_config['device_name']}")
    print(f"  Max sequence length: {config.max_seq_len:,} bp")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.accumulate_grad_batches}")
    print(f"  Effective batch: {config.batch_size * config.accumulate_grad_batches}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Encoder frozen: {config.freeze_encoder}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"\nGPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize data module and model
    data_module = BetaDogmaDataModule(config)
    model = BetaDogmaLightning(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        monitor=monitor,
        mode='min',
        save_last=True,
        auto_insert_metric_name=False  # Prevents formatting issues with None values
    )
    
    # Initialize learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up callbacks
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Only enable early stopping if we have validation data
    if hasattr(data_module, 'val_dataset') and len(data_module.val_dataset) > 0:
        early_stop_callback = EarlyStopping(
            monitor=config.monitor,
            patience=config.patience,
            mode=config.mode,
            verbose=True,
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stop_callback)
    
    logger = TensorBoardLogger(save_dir=config.output_dir, name="logs")
    
    # Configure trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=config.devices,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=str(config.output_dir),
        num_sanity_val_steps=0,  # Skip validation sanity check
        limit_val_batches=config.limit_val_batches,
        limit_train_batches=config.limit_train_batches,
        gradient_clip_algorithm="norm",  # More stable gradient clipping
    )
    
    print("Starting training...")
    print("="*80 + "\n")
    
    try:
        trainer.fit(model, data_module)
        
        print("\n✅ Training complete!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        else:
            print("No validation metrics were recorded")
        
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ OUT OF MEMORY!")
            print(f"Current config: {config.max_seq_len:,} bp")
            print("Try reducing max_seq_len in Config")
        raise


if __name__ == "__main__":
    # Load configuration from YAML
    config = Config()
    
    # Start training with the loaded configuration
    train(
        data_dir=str(config.data_dir),
        output_dir=str(config.output_dir),
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_workers=config.num_workers,
        devices=config.devices,
        precision=config.precision,
        monitor=config.monitor
    )