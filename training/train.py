#!/usr/bin/env python3
"""train.py - Optimized for training 450k sequences on T4 (16GB)."""

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForMaskedLM
import warnings
import yaml

# Add parent directory to path for local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

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
            'max_seq_len': 300000,  # Full 450k!
            'batch_size': 2,
            'accumulate_grad_batches': 8,
            'device_name': 'A100-80GB',
            'use_gradient_checkpointing': False,
        }
        print("   ✅ A100-80GB: Full 450k sequences, batch_size=2")
        
    elif gpu_memory_gb >= 35:  # A100-40GB
        config = {
            'max_seq_len': 300000,  # Full 450k!
            'batch_size': 1,
            'accumulate_grad_batches': 16,
            'device_name': 'A100-40GB',
            'use_gradient_checkpointing': True,
        }
        print("   ✅ A100-40GB: Full 450k sequences with gradient checkpointing")
        
    elif gpu_memory_gb >= 14:  # T4, RTX 3080 (16GB)
        config = {
            'max_seq_len': 300000,  # YES! Full 450k!
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
        self.pos_weight = 20.0  # For positive class in BCEWithLogitsLoss

        # Phase 1: Protein prediction
        self.protein_hidden = 256
        self.protein_layers = 2
        
        # Phase 1: Loss weights
        self.w_protein = 2.0
        self.w_cds_start = 0.5
        self.w_cds_end = 0.5
        self.w_nmd = 1.0
        self.w_expression = 1.0
        
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
        self.max_length = max_length  # Hard cap at 450k
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
            return {'proteins': [], 'nmd_flags': [], 'tpms': [], 'cds_coords': [], 'is_canonical': []}
        
        try:
            isoforms = json.loads(isoforms_json)
            return {
                'proteins': [iso.get('protein_seq', '') for iso in isoforms],
                'nmd_flags': [iso.get('has_nmd', False) for iso in isoforms],
                'tpms': [iso.get('expression_tpm', 0.0) for iso in isoforms],
                'cds_coords': [(iso.get('cds_start', -1), iso.get('cds_end', -1)) for iso in isoforms],
                'is_canonical': [iso.get('is_canonical', False) for iso in isoforms]
            }
        except:
            return {'proteins': [], 'nmd_flags': [], 'tpms': [], 'cds_coords': [], 'is_canonical': []}
    
    @staticmethod
    def extract_canonical_isoform(isoform_data: Dict) -> Dict:
        """Extract canonical isoform data from parsed isoforms."""
        canonical_idx = None
        for i, is_canon in enumerate(isoform_data['is_canonical']):
            if is_canon:
                canonical_idx = i
                break
        
        if canonical_idx is None and isoform_data['tpms']:
            canonical_idx = np.argmax(isoform_data['tpms'])
        
        if canonical_idx is not None and canonical_idx < len(isoform_data['proteins']):
            cds_start, cds_end = isoform_data['cds_coords'][canonical_idx]
            return {
                'protein': isoform_data['proteins'][canonical_idx],
                'nmd': isoform_data['nmd_flags'][canonical_idx],
                'tpm': isoform_data['tpms'][canonical_idx],
                'cds_start': cds_start,
                'cds_end': cds_end
            }
        
        return {'protein': '', 'nmd': False, 'tpm': 0.0, 'cds_start': -1, 'cds_end': -1}
    
    @classmethod
    def create_protein_labels(cls, protein_seq: str, cds_start: int, cds_end: int, seq_len: int) -> np.ndarray:
        """Convert protein sequence to per-position amino acid labels."""
        labels = np.full(seq_len, -1, dtype=np.int64)
        
        if not protein_seq or cds_start is None or cds_end is None or cds_start < 0 or cds_end < 0 or cds_end > seq_len:
            return labels
        
        codon_positions = range(cds_start, min(cds_end, seq_len), 3)
        
        for i, pos in enumerate(codon_positions):
            if i >= len(protein_seq):
                break
            aa = protein_seq[i]
            if aa in cls.AA_TO_IDX:
                aa_idx = cls.AA_TO_IDX[aa]
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
    
    def _apply_variant_to_sequence(self, seq: str, variant: Dict) -> str:
        """Apply a variant to the sequence.
        
        Args:
            seq: Reference sequence
            variant: Variant dict with 'pos', 'ref', 'alt'
            
        Returns:
            Sequence with variant applied
        """
        pos = variant.get('pos', 0)
        ref = variant.get('ref', '')
        alt = variant.get('alt', '')
        
        if pos < 0 or pos >= len(seq):
            return seq
        
        # Verify reference matches
        if seq[pos:pos+len(ref)] != ref:
            return seq
        
        # Apply variant
        seq_list = list(seq)
        seq_list[pos:pos+len(ref)] = list(alt)
        return ''.join(seq_list)
    
    def _recompute_labels_for_variant(self, labels: Dict, variant: Dict, seq_len: int) -> Dict:
        """Recompute labels for a variant (Phase 2B).
        
        For benign variants: keep labels the same (sequence changed, function didn't)
        For pathogenic variants: modify labels based on variant effect
        
        Args:
            labels: Reference labels dict
            variant: Variant dict with 'is_benign', 'splice_effect_score'
            seq_len: Sequence length
            
        Returns:
            Updated labels dict
        """
        labels_alt = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
        
        # For benign variants, keep labels the same (this is the teaching signal)
        if variant.get('is_benign', False):
            return labels_alt
        
        # For pathogenic variants with splice effects, modify splice labels
        if variant.get('has_splice_effect', False):
            effect_score = variant.get('splice_effect_score', 0.0)
            
            # If strong effect, flip splice site labels at variant position
            if effect_score > 0.5:
                pos = variant.get('pos', 0)
                if 0 <= pos < seq_len:
                    # Flip donor label
                    if 'donor' in labels_alt:
                        labels_alt['donor'][pos] = 1.0 - labels_alt['donor'][pos]
                    # Flip acceptor label
                    if 'acceptor' in labels_alt:
                        labels_alt['acceptor'][pos] = 1.0 - labels_alt['acceptor'][pos]
        
        return labels_alt
    
    def _get_augmentation_mode(self) -> str:
        """Decide augmentation mode: reference, benign, or pathogenic (33/33/33).
        
        Returns:
            'reference', 'benign', or 'pathogenic'
        """
        if self.mode in ['val', 'test']:
            return 'reference'  # No augmentation in val/test
        
        p = random.random()
        if p < 0.33:
            return 'reference'
        elif p < 0.66:
            return 'benign'
        else:
            return 'pathogenic'
    
    def __getitem__(self, idx):
        """Get a single item with Phase 2 variant augmentation support."""
        file_path, row_idx = self._get_file_and_row(idx)
        
        try:
            # Read row with Phase 2 variant data
            df = pd.read_parquet(
                file_path,
                columns=['seq', 'donor', 'acceptor', 'tss', 'polya', 'isoforms', 'variants']
            )
                
            if len(df) <= row_idx:
                raise IndexError(f"Row {row_idx} not found")
                    
            row = df.iloc[row_idx]
            
            # ================================================================
            # Phase 2A: Decide augmentation mode (33/33/33 strategy)
            # ================================================================
            aug_mode = self._get_augmentation_mode()
            variant = None
            
            # Parse variants from row
            try:
                variants_json = row.get('variants', '[]')
                if isinstance(variants_json, str):
                    variants = json.loads(variants_json)
                else:
                    variants = variants_json if variants_json else []
            except:
                variants = []
            
            # Select variant based on augmentation mode
            if aug_mode == 'benign' and variants:
                benign_vars = [v for v in variants if v.get('is_benign', False)]
                if benign_vars:
                    variant = random.choice(benign_vars)
                    aug_mode = 'benign'
                else:
                    aug_mode = 'reference'
            
            elif aug_mode == 'pathogenic' and variants:
                path_vars = [v for v in variants if v.get('is_pathogenic', False) or v.get('has_splice_effect', False)]
                if path_vars:
                    variant = random.choice(path_vars)
                    aug_mode = 'pathogenic'
                else:
                    aug_mode = 'reference'
            
            # ================================================================
            # Process sequence
            # ================================================================
            seq = str(row['seq'])
            if not seq or len(seq) == 0:
                seq = 'N' * self.max_seq_len
            
            # Debug: Print sequence length before processing
            if idx < 3:  # Only print for first 3 examples
                print(f"\n=== Processing sequence {idx} ===")
                print(f"Original sequence length: {len(seq)}")
                print(f"Target max_seq_len: {self.max_seq_len}")
            
            # Truncate sequence if needed
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len]  # Simple truncation from start
                if idx < 3:
                    print(f"Truncated sequence to {len(seq)} bases")
            elif len(seq) < self.max_seq_len:
                # Only pad if absolutely necessary, but better to avoid this case
                pad_len = self.max_seq_len - len(seq)
                seq = seq + 'N' * pad_len
                if idx < 3:
                    print(f"Padded sequence to {len(seq)} bases")
            
            if idx < 3:
                print(f"Final sequence length: {len(seq)}")
                print(f"First 10 bases: {seq[:10]}")
            
            # Phase 2A: Apply variant to sequence if selected
            if variant and aug_mode in ['benign', 'pathogenic']:
                seq = self._apply_variant_to_sequence(seq, variant)
            
            # Tokenize with debug info
            if idx < 3:  # Debug print for first 3 examples
                print(f"\nBefore tokenization:")
                print(f"  Sequence length: {len(seq)}")
                print(f"  First 10 bases: {seq[:10]}")
            
            # Tokenize without padding/truncation since we already handled length
            tokenized = self.tokenizer(
                seq,
                max_length=None,  # No max_length since we already handled it
                padding=False,    # No padding since we already handled length
                truncation=False  # No truncation since we already handled it
            )
            
            if idx < 3:  # Debug print for first 3 examples
                print(f"After tokenization:")
                print(f"  Input IDs shape: {tokenized['input_ids'].shape}")
                print(f"  Attention mask shape: {tokenized['attention_mask'].shape}")
            
            # ================================================================
            # Process labels
            # ================================================================
            labels = {
                'donor': self.to_tensor(row['donor'], self.max_seq_len),
                'acceptor': self.to_tensor(row['acceptor'], self.max_seq_len),
                'tss': self.to_tensor(row['tss'], self.max_seq_len),
                'polya': self.to_tensor(row['polya'], self.max_seq_len),
            }
            
            # Phase 1: Isoform data
            isoform_data = self.parse_isoforms(row.get('isoforms', '[]'))
            canonical = self.extract_canonical_isoform(isoform_data)
            
            protein_labels = self.create_protein_labels(canonical['protein'], canonical['cds_start'], canonical['cds_end'], self.max_seq_len)
            labels['protein'] = torch.from_numpy(protein_labels)
            
            cds_start_labels, cds_end_labels = self.create_cds_boundary_labels(canonical['cds_start'], canonical['cds_end'], self.max_seq_len)
            labels['cds_start'] = torch.from_numpy(cds_start_labels)
            labels['cds_end'] = torch.from_numpy(cds_end_labels)
            
            nmd_value = canonical['nmd']
            if isinstance(nmd_value, str):
                nmd_value = nmd_value.lower() == 'true'
            labels['nmd'] = torch.tensor(float(nmd_value), dtype=torch.float32)
            
            labels['expression'] = torch.tensor(np.log1p(canonical['tpm']), dtype=torch.float32)
            
            # ================================================================
            # Phase 2B: Recompute labels for variant if needed
            # ================================================================
            if variant and aug_mode in ['benign', 'pathogenic']:
                labels = self._recompute_labels_for_variant(labels, variant, self.max_seq_len)
                # Add variant effect label for training
                labels['variant_effect'] = torch.tensor(variant.get('splice_effect_score', 0.0), dtype=torch.float32)
            else:
                labels['variant_effect'] = torch.tensor(0.0, dtype=torch.float32)
            
            return {
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0),
                'labels': labels,
                'augmentation_mode': aug_mode
            }
                
        except Exception as e:
            print(f"⚠️  Error loading example {idx}: {e}")
            tokenized = self.tokenizer('N' * self.max_seq_len, max_length=self.max_seq_len, padding='max_length', truncation=True)
            
            return {
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0),
                'labels': {
                    'donor': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'acceptor': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'tss': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'polya': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'protein': torch.full((self.max_seq_len,), -1, dtype=torch.long),
                    'cds_start': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'cds_end': torch.zeros(self.max_seq_len, dtype=torch.float32),
                    'nmd': torch.tensor(0.0, dtype=torch.float32),
                    'expression': torch.tensor(0.0, dtype=torch.float32),
                    'variant_effect': torch.tensor(0.0, dtype=torch.float32),
                },
                'augmentation_mode': 'error'
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
    """Wrapper for the HyenaDNA model with memory optimizations for long sequences.
    
    This wrapper provides several key features:
    - Automatic device management (CPU/GPU)
    - Gradient checkpointing for memory efficiency
    - Detailed error reporting and recovery
    - Memory usage monitoring
    - Support for very long sequences (up to 300k tokens)
    """
    
    def __init__(self, model_name: str, device: str = None, use_gradient_checkpointing: bool = False, freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.frozen = freeze
        
        print("\n" + "="*80)
        print(f"=== 🧬 INITIALIZING HYENA DNA ENCODER ===")
        print("="*80)
        
        # Force CPU for now to debug MPS issues
        self.device = torch.device('cpu')
        print(f"\n⚠️  FORCING CPU USAGE FOR DEBUGGING ⚠️\n")
        
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Gradient Checkpointing: {use_gradient_checkpointing}")
        print(f"  Freeze: {freeze}")
        
        try:
            # Initialize the model with memory-efficient settings
            print("\n[1/3] 🚀 LOADING MODEL WEIGHTS")
            print(f"  Loading {model_name}...")
            
            # Clear any cached memory before loading the model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Import the correct Auto class - HyenaDNA needs AutoModel, not AutoModelForMaskedLM
            from transformers import AutoModel
            
            # Load model with memory-efficient settings
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                dtype=torch.float32,  # Use float32 for stability
                device_map=None  # Disable device_map to handle manually
            )
            
            # Move model to device
            print(f"  Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing if requested
            if use_gradient_checkpointing:
                print("  Enabling gradient checkpointing...")
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                else:
                    print("  ⚠️  Model does not support gradient checkpointing")
            
            # Print model info
            print("\n[2/3] ✅ MODEL LOADED SUCCESSFULLY")
            print(f"  Model class: {self.model.__class__.__name__}")
            print(f"  Model device: {next(self.model.parameters()).device}")
            print(f"  Model dtype: {next(self.model.parameters()).dtype}")
            
            # Get model config to determine hidden size
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'd_model'):
                    self.hidden_size = self.model.config.d_model
                    print(f"  Hidden size (d_model): {self.hidden_size}")
                elif hasattr(self.model.config, 'hidden_size'):
                    self.hidden_size = self.model.config.hidden_size
                    print(f"  Hidden size: {self.hidden_size}")
                else:
                    self.hidden_size = 768  # Default fallback
                    print(f"  ⚠️  Could not determine hidden size, using default: {self.hidden_size}")
            else:
                self.hidden_size = 768
                print(f"  ⚠️  No config found, using default hidden size: {self.hidden_size}")
            
            # Print memory usage
            if torch.cuda.is_available():
                print("\n[3/3] 📊 GPU MEMORY USAGE")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            
            # Test with a small forward pass
            print("\n[3/3] 🧪 TESTING FORWARD PASS")
            with torch.no_grad():
                test_input = torch.zeros(1, 100, dtype=torch.long, device=self.device)
                test_output = self.model(test_input)
                
                # Check if output has the expected structure
                if hasattr(test_output, 'last_hidden_state'):
                    print(f"  ✅ Output shape: {test_output.last_hidden_state.shape}")
                elif isinstance(test_output, tuple) and len(test_output) > 0:
                    print(f"  ✅ Output shape (tuple): {test_output[0].shape}")
                    self.hidden_size = test_output[0].shape[-1]
                else:
                    print(f"  ⚠️  Unexpected output type: {type(test_output)}")
            
            print(f"\n✅ Successfully loaded model: {model_name}")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Hidden size: {self.hidden_size}")
            
            # Freeze parameters if needed
            if freeze:
                print("✅ Freezing encoder parameters")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            else:
                print("✅ Keeping encoder parameters trainable")
            
            print("\n✅ HYENA DNA ENCODER INITIALIZED")
            print("="*80 + "\n")
            
        except Exception as e:
            print("\n❌ FAILED TO INITIALIZE HYENA DNA ENCODER")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            
            # Print memory stats if available
            if torch.cuda.is_available():
                try:
                    print("\n💾 GPU MEMORY AT FAILURE:")
                    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                    print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                except Exception as me:
                    print(f"  Could not get GPU memory info: {str(me)}")
            
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("  1. Check if the model name is correct")
            print("  2. Verify you have enough disk space for the model weights")
            print("  3. Try with a smaller model (e.g., 'LongSafari/hyenadna-small-32k-seqlen')")
            print("  4. Check for CUDA/CPU compatibility")
            print("  5. Make sure transformers library is up to date: pip install --upgrade transformers")
            
            # Re-raise the error with more context
            raise RuntimeError(f"Failed to initialize HyenaDNA model: {str(e)}") from e
    

    def forward(self, input_ids, attention_mask=None):
        print("\n=== HyenaDNAEncoder.forward ===")
        print(f"[1/6] Input shape: {input_ids.shape}, device: {input_ids.device}, dtype: {input_ids.dtype}")
        
        # Ensure we're in eval mode if frozen
        if self.frozen:
            print("[2/6] Setting model to eval mode (frozen)")
            self.model.eval()
        
        # Move inputs to the correct device
        print("[3/6] Moving input to device...")
        input_ids = input_ids.to(self.device)
        
        # Ensure input is long type
        if input_ids.dtype != torch.long:
            print(f"Converting input dtype from {input_ids.dtype} to long")
            input_ids = input_ids.long()
        
        # Note: HyenaDNA doesn't use attention_mask, so we ignore it
        if attention_mask is not None:
            print("[4/6] Note: HyenaDNA doesn't use attention_mask (ignored)")
        
        print(f"[4/6] Input device: {input_ids.device}, shape: {input_ids.shape}")
        
        try:
            print("[5/6] Starting model forward...")
            with torch.no_grad() if self.frozen else torch.enable_grad():
                # Enable gradient checkpointing if not frozen
                if not self.frozen and hasattr(self.model, 'gradient_checkpointing_enable'):
                    print("  Enabling gradient checkpointing")
                    self.model.gradient_checkpointing_enable()
                
                # HyenaDNA only takes input_ids, no attention_mask
                print("  Running model forward pass (HyenaDNA - no attention mask)...")
                outputs = self.model(input_ids)
                print("  Forward pass completed")
                
                # Handle different output formats
                if hasattr(outputs, 'last_hidden_state'):
                    print("  Using last_hidden_state from outputs")
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    print("  Using first element from tuple output")
                    hidden_states = outputs[0]
                    # Create a simple object to hold the hidden states
                    class SimpleOutput:
                        def __init__(self, hidden_states):
                            self.last_hidden_state = hidden_states
                    outputs = SimpleOutput(hidden_states)
                elif isinstance(outputs, torch.Tensor):
                    print("  Output is a tensor, wrapping it")
                    # Create a simple object to hold the hidden states
                    class SimpleOutput:
                        def __init__(self, hidden_states):
                            self.last_hidden_state = hidden_states
                    outputs = SimpleOutput(outputs)
                else:
                    raise ValueError(f"Unexpected output format: {type(outputs)}")
                
                print(f"[6/6] Output shape: {outputs.last_hidden_state.shape}")
                return outputs
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("\n❌ OUT OF MEMORY ERROR")
                print(f"Input shape: {input_ids.shape}")
                print(f"Batch size: {input_ids.size(0)}")
                print(f"Sequence length: {input_ids.size(1)}")
                print("\nTry reducing batch size or sequence length")
                
                # Clear cache and try again
                if torch.cuda.is_available():
                    print("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
            
            raise


class BetaDogmaModel(nn.Module):
    """BetaDogma model with memory optimizations and splice effect prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        print("\n" + "="*80)
        print(f"Initializing BetaDogmaModel")
        print(f"  Sequence length: {config.max_seq_len}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Number of layers: {config.num_layers}")
        print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
        print("="*80 + "\n")
        
        # Determine device - prioritize CPU for now to debug
        self.device = torch.device('cpu')
        print(f"\n⚠️  FORCING CPU USAGE FOR DEBUGGING ⚠️\n")
        
        print(f"Using device: {self.device}")
        
        # Initialize encoder with the determined device
        self.encoder = HyenaDNAEncoder(
            model_name=config.model_name,
            freeze=config.freeze_encoder,
            device=str(self.device),  # Pass the device as string
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
        
        # Ensure encoder is on the correct device
        self.encoder = self.encoder.to(self.device)
        print(f"Model initialized on device: {self.device}")
        
        # Get the encoder dimension from the encoder itself
        self.encoder_dim = self.encoder.hidden_size
        print(f"Using encoder dimension: {self.encoder_dim}")
        
        use_checkpointing = config.use_gradient_checkpointing
        
        # Initialize prediction heads with proper dimensions
        self.donor_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        )
        
        self.acceptor_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        )
        
        # Splice effect prediction head (regression)
        self.splice_effect_head = PredictionHead(
            self.encoder_dim, 
            config.splice_hidden,
            config.splice_layers, 
            config.dropout, 
            use_checkpointing
        )
        
        # Other prediction heads
        self.tss_head = PredictionHead(
            self.encoder_dim, 
            config.tss_hidden,
            config.tss_layers, 
            config.dropout, 
            use_checkpointing
        )
        
        self.polya_head = PredictionHead(
            self.encoder_dim, 
            config.polya_hidden,
            config.polya_layers, 
            config.dropout, 
            use_checkpointing
        )
        
        # Phase 1: Protein prediction heads
        self.protein_head = nn.Sequential(
            nn.LSTM(self.encoder_dim, config.protein_hidden, num_layers=config.protein_layers, 
                   bidirectional=True, batch_first=True, dropout=config.dropout if config.protein_layers > 1 else 0.0),
            nn.Dropout(config.dropout),
        )
        self.protein_fc = nn.Linear(config.protein_hidden * 2, 21)  # 20 AA + stop
        
        self.cds_start_head = PredictionHead(self.encoder_dim, config.protein_hidden, 1, config.dropout, use_checkpointing)
        self.cds_end_head = PredictionHead(self.encoder_dim, config.protein_hidden, 1, config.dropout, use_checkpointing)
        
        # NMD and expression prediction
        self.nmd_head = nn.Sequential(
            nn.Linear(self.encoder_dim, config.protein_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.protein_hidden, 1)
        )
        
        self.expression_head = nn.Sequential(
            nn.Linear(self.encoder_dim, config.protein_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.protein_hidden, 1)
        )
        
        # Initialize weights for all heads
        for head in [self.donor_head, self.acceptor_head, self.splice_effect_head, 
                    self.tss_head, self.polya_head]:
            self._init_weights(head)
        
        # Initialize protein head separately
        for module in self.protein_head.modules():
            self._init_weights(module)
        self._init_weights(self.protein_fc)
        
        # Initialize other heads
        for head in [self.cds_start_head, self.cds_end_head, self.nmd_head, self.expression_head]:
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

    def forward(self, input_ids, attention_mask=None):
        """Forward pass with memory optimizations for long sequences.
        
        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing model outputs for each task
        """
        print("\n" + "="*80)
        print(f"=== 🚀 BETA DOGMA FORWARD PASS ===")
        print("="*80)
        
        try:
            # 1. Input validation and logging
            print("\n[1/6] 🔍 INPUT VALIDATION")
            if input_ids.dim() != 2:
                raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}")
                
            print(f"Input shape: {tuple(input_ids.shape)}")
            print(f"Input device: {input_ids.device}")
            print(f"Input dtype: {input_ids.dtype}")
            print(f"Input stats - min: {input_ids.min()}, max: {input_ids.max()}, mean: {input_ids.float().mean():.2f}")
            
            if attention_mask is not None:
                print(f"Attention mask shape: {tuple(attention_mask.shape)}")
                print(f"Attention mask device: {attention_mask.device}")
                print(f"Attention mask stats - min: {attention_mask.min()}, max: {attention_mask.max()}")
            
            # 2. Move inputs to correct device
            print("\n[2/6] 🚚 MOVING TO DEVICE")
            input_device = input_ids.device
            target_device = self.device
            print(f"Moving tensors from {input_device} to {target_device}")
            
            if input_device != target_device:
                print(f"Moving input_ids to {target_device}...")
                input_ids = input_ids.to(target_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(target_device)
            
            # 3. Ensure correct dtypes
            print("\n[3/6] ✅ DATA TYPES")
            if input_ids.dtype != torch.long:
                print(f"Converting input_ids from {input_ids.dtype} to long")
                input_ids = input_ids.long()
                
            if attention_mask is not None and attention_mask.dtype != torch.long:
                print(f"Converting attention_mask from {attention_mask.dtype} to long")
                attention_mask = attention_mask.long()
            
            # 4. Check for invalid values
            print("\n[4/6] 🔍 INPUT VALIDATION")
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                nan_count = torch.isnan(input_ids).sum().item()
                inf_count = torch.isinf(input_ids).sum().item()
                print(f"❌ ERROR: Input contains invalid values!")
                print(f"  NaN count: {nan_count}")
                print(f"  Inf count: {inf_count}")
                print(f"  Input shape: {tuple(input_ids.shape)}")
                print(f"  Input min/max: {input_ids.min()}/{input_ids.max()}")
                raise ValueError("Input contains NaN or Inf values")
            
            # 5. Run the encoder
            print("\n[5/6] 🧬 ENCODER FORWARD")
            print("Starting encoder forward pass...")
            
            # Clear any cached memory before running the encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Run encoder with memory optimizations
                with torch.autocast(device_type='cpu', enabled=False):
                    print("  Encoder input shape:", tuple(input_ids.shape))
                    print("  Encoder input device:", input_ids.device)
                    print("  Encoder input dtype:", input_ids.dtype)
                    
                    # Print memory stats if on CUDA
                    if torch.cuda.is_available():
                        print(f"  CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        print(f"  CUDA memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                    
                    # Run the encoder
                    print("  Calling encoder...")
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    print("✅ Encoder forward pass completed")
                    
                    # Verify encoder outputs
                    if not hasattr(outputs, 'last_hidden_state'):
                        if hasattr(outputs, 'hidden_states'):
                            print("  Using last hidden state from hidden_states")
                            outputs.last_hidden_state = outputs.hidden_states[-1]
                        else:
                            raise ValueError("No valid hidden states found in encoder outputs")
                    
                    print(f"  Hidden states shape: {tuple(outputs.last_hidden_state.shape)}")
                    print(f"  Hidden states device: {outputs.last_hidden_state.device}")
                    print(f"  Hidden states dtype: {outputs.last_hidden_state.dtype}")
                    
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("\n❌ OUT OF MEMORY IN HYENA DNA ENCODER")
                    print(f"  Input shape: {tuple(input_ids.shape)}")
                    print(f"  Batch size: {input_ids.size(0)}")
                    print(f"  Sequence length: {input_ids.size(1)}")
                    
                    if torch.cuda.is_available():
                        print("\n💾 GPU MEMORY USAGE:")
                        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                        
                        # Clear cache and suggest solutions
                        print("\n🔄 Clearing CUDA cache...")
                        torch.cuda.empty_cache()
                    
                    print("\n💡 SUGGESTED SOLUTIONS:")
                    print("  1. Reduce batch size (currently 1)")
                    print("  2. Reduce sequence length (currently 300,000)")
                    print("  3. Enable gradient checkpointing in config")
                    print("  4. Use a smaller model")
                    print("  5. Use CPU instead of GPU (slower but more memory)")
                
                # Re-raise the error with more context
                raise RuntimeError(f"Error in HyenaDNAEncoder forward pass: {str(e)}") from e
            
            # 6. Process through prediction heads
            print("\n[6/6] 🎯 PREDICTION HEADS")
            hidden_states = outputs.last_hidden_state
            
            # Get predictions from each head
            results = {
                'donor': self.donor_head(hidden_states),
                'acceptor': self.acceptor_head(hidden_states),
                'splice_effect': self.splice_effect_head(hidden_states),
                'tss': self.tss_head(hidden_states),
                'polya': self.polya_head(hidden_states),
            }
            
            # Protein prediction
            protein_lstm_out, _ = self.protein_head[0](hidden_states)
            protein_lstm_out = self.protein_head[1](protein_lstm_out)
            results['protein'] = self.protein_fc(protein_lstm_out)
            
            # CDS boundaries
            results['cds_start'] = self.cds_start_head(hidden_states)
            results['cds_end'] = self.cds_end_head(hidden_states)
            
            # NMD and expression (using mean pooling)
            pooled = hidden_states.mean(dim=1)
            results['nmd'] = self.nmd_head(pooled).squeeze(-1)
            results['expression'] = self.expression_head(pooled).squeeze(-1)
            
            print("\n✅ FORWARD PASS COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            
            return results
                
        except Exception as e:
            print(f"\n❌ UNHANDLED ERROR IN FORWARD PASS: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print("\nCURRENT TENSOR INFO:")
            print(f"  Input shape: {tuple(input_ids.shape) if 'input_ids' in locals() else 'N/A'}")
            print(f"  Input device: {input_ids.device if 'input_ids' in locals() else 'N/A'}")
            print(f"  Input dtype: {input_ids.dtype if 'input_ids' in locals() else 'N/A'}")
            
            # Print model device info
            print("\nMODEL INFO:")
            print(f"  Model device: {self.device}")
            if hasattr(self, 'encoder') and hasattr(self.encoder, 'parameters'):
                try:
                    param = next(self.encoder.parameters())
                    print(f"  Encoder device: {param.device}")
                    print(f"  Encoder dtype: {param.dtype}")
                except Exception as pe:
                    print(f"  Could not get encoder parameter info: {str(pe)}")
            
            # Print memory stats if available
            if torch.cuda.is_available():
                try:
                    print("\nGPU MEMORY INFO:")
                    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                    print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                except Exception as me:
                    print(f"  Could not get GPU memory info: {str(me)}")
            
            # Re-raise the original error
            raise


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
        
        # Phase 1: Protein prediction with NaN handling
        # Check if we have any valid protein labels (not -1)
        protein_mask = (labels['protein'] != -1).view(-1)
        if protein_mask.sum() > 0:
            # Only compute loss on valid positions
            loss_protein = F.cross_entropy(
                outputs['protein'].view(-1, 21)[protein_mask],
                labels['protein'].view(-1)[protein_mask]
            )
        else:
            # No valid protein labels in this batch, use zero loss
            loss_protein = torch.tensor(0.0, device=self.device)
        
        loss_cds_start = F.binary_cross_entropy_with_logits(
            outputs['cds_start'],
            labels['cds_start'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_cds_end = F.binary_cross_entropy_with_logits(
            outputs['cds_end'],
            labels['cds_end'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_nmd = F.binary_cross_entropy_with_logits(
            outputs['nmd'],
            labels['nmd']
        )
        
        loss_expression = F.mse_loss(
            outputs['expression'],
            labels['expression']
        )
        
        # Phase 2B: Variant effect prediction loss
        loss_variant_effect = torch.tensor(0.0, device=self.device)
        if 'variant_effect' in labels and labels['variant_effect'].sum() > 0:
            # Only compute loss on examples with variants
            mask = (labels['variant_effect'] > 0).float()
            if mask.sum() > 0:
                loss_variant_effect = F.mse_loss(
                    outputs.get('variant_effect', torch.zeros_like(labels['variant_effect'])) * mask,
                    labels['variant_effect'] * mask
                )
        
        # Combine losses with weights
        # Only include protein loss if it's valid (not zero from no labels)
        protein_weight = self.config.w_protein if protein_mask.sum() > 0 else 0.0
        
        loss = (
            self.config.w_splice_donor * loss_donor +
            self.config.w_splice_acceptor * loss_acceptor +
            self.config.w_tss * loss_tss +
            self.config.w_polya * loss_polya +
            self.config.w_splice_effect * loss_splice +
            protein_weight * loss_protein +
            self.config.w_cds_start * loss_cds_start +
            self.config.w_cds_end * loss_cds_end +
            self.config.w_nmd * loss_nmd +
            self.config.w_expression * loss_expression +
            0.5 * loss_variant_effect  # Phase 2B weight
        )
        
        # Additional safety check for NaN
        if torch.isnan(loss):
            print("\n⚠️ WARNING: Total loss is NaN!")
            print("Individual losses:")
            print(f"  loss_donor: {loss_donor.item()}")
            print(f"  loss_acceptor: {loss_acceptor.item()}")
            print(f"  loss_tss: {loss_tss.item()}")
            print(f"  loss_polya: {loss_polya.item()}")
            print(f"  loss_splice: {loss_splice.item()}")
            print(f"  loss_protein: {loss_protein.item()}")
            print(f"  loss_cds_start: {loss_cds_start.item()}")
            print(f"  loss_cds_end: {loss_cds_end.item()}")
            print(f"  loss_nmd: {loss_nmd.item()}")
            print(f"  loss_expression: {loss_expression.item()}")
            print(f"  loss_variant_effect: {loss_variant_effect.item()}")
            
            # Replace NaN with a large value to continue training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        return {
            'loss': loss,
            'loss/donor': loss_donor,
            'loss/acceptor': loss_acceptor,
            'loss/tss': loss_tss,
            'loss/polya': loss_polya,
            'loss/splice_effect': loss_splice,
            'loss/protein': loss_protein,
            'loss/cds_start': loss_cds_start,
            'loss/cds_end': loss_cds_end,
            'loss/nmd': loss_nmd,
            'loss/expression': loss_expression,
            'loss/variant_effect': loss_variant_effect,
        }

    
    def training_step(self, batch, batch_idx):
        print("\n" + "="*80)
        print(f"=== 🚀 TRAINING STEP {batch_idx} ===")
        print("="*80)
        
        try:
            # 1. Print batch info
            print("\n[1/5] 📦 BATCH INFO")
            print(f"- Batch ID: {batch_idx}")
            print(f"- Batch keys: {list(batch.keys())}")
            
            # 2. Print tensor details
            print("\n[2/5] 🧮 TENSOR DETAILS")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}:")
                    print(f"    shape: {tuple(v.shape)}")
                    print(f"    device: {v.device}")
                    print(f"    dtype: {v.dtype}")
                    print(f"    requires_grad: {v.requires_grad}")
                    print(f"    min/mean/max: {v.min().item():.2f}/{v.float().mean().item():.2f}/{v.max().item():.2f}")
                    print(f"    isnan: {torch.isnan(v).any().item()}, isinf: {torch.isinf(v).any().item()}")
                else:
                    print(f"  {k}: {type(v).__name__}")
            
            # 3. Move batch to device
            print("\n[3/5] 🚚 MOVING TO DEVICE")
            print(f"Target device: {self.device}")
            
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  Moving {k} to {self.device}...")
                    device_batch[k] = v.to(self.device)
                    print(f"  ✅ Moved {k} to {device_batch[k].device}")
                else:
                    device_batch[k] = v
            
            # 4. Forward pass
            print("\n[4/5] 🚀 FORWARD PASS")
            try:
                with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=False):
                    print("  Starting model forward...")
                    outputs = self(device_batch['input_ids'], 
                                 attention_mask=device_batch.get('attention_mask'))
                    print("  ✅ Forward pass completed successfully!")
                    
                    # Print output shapes
                    if isinstance(outputs, dict):
                        print("  Model outputs:")
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor):
                                print(f"    {k}: {tuple(v.shape)} | {v.device} | {v.dtype}")
                    
                    # 5. Compute loss
                    print("\n[5/5] 📉 LOSS COMPUTATION")
                    print("  Computing losses...")
                    loss_dict = self._compute_loss(outputs, device_batch)
                    
                    if not isinstance(loss_dict, dict):
                        raise ValueError(f"_compute_loss should return a dict, got {type(loss_dict)}")
                    
                    if 'loss' not in loss_dict:
                        raise ValueError("'loss' key not found in loss_dict")
                    
                    loss = loss_dict['loss']
                    if not isinstance(loss, torch.Tensor):
                        raise ValueError(f"loss should be a tensor, got {type(loss)}")
                    
                    print("  ✅ Loss computation completed!")
                    print("\n📊 LOSS BREAKDOWN:")
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: {v.item():.6f}")
                            self.log(f'train/{k}', v, prog_bar=True, on_step=True, on_epoch=True)
                    
                    print(f"\n✅ BATCH {batch_idx} COMPLETED SUCCESSFULLY!")
                    return loss
                    
            except Exception as e:
                print("\n❌ ERROR DURING FORWARD/LOSS:")
                print(f"  Type: {type(e).__name__}")
                print(f"  Message: {str(e)}")
                print("\nMODEL STATE:")
                print(f"  Device: {self.device}")
                print(f"  Training mode: {self.training}")
                
                if 'out of memory' in str(e).lower():
                    print("\n💡 MEMORY USAGE:")
                    if torch.cuda.is_available():
                        print(f"  CUDA allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        print(f"  CUDA reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                    print("  Try reducing batch size or sequence length")
                
                raise
                
        except Exception as e:
            print("\n" + "❌"*30)
            print(f"❌ CRITICAL ERROR IN TRAINING STEP {batch_idx}")
            print("❌"*30)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            import traceback
            print("\nTRACEBACK:")
            traceback.print_exc()
            
            raise
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss_dict = self._compute_loss(outputs, batch)
        
        for k, v in loss_dict.items():
            self.log(f'val/{k}', v, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss_dict['loss']
    
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
            persistent_workers=True if self.config.num_workers > 0 else False,
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
            persistent_workers=True if self.config.num_workers > 0 else False,
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
            persistent_workers=True if self.config.num_workers > 0 else False,
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
    """Train the model with the given configuration."""
    # Ensure monitor has a default value if None
    if monitor is None:
        monitor = "val/loss"
        print("Warning: monitor was None, defaulting to 'val/loss'")
    
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
        auto_insert_metric_name=False
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
    # Use CPU on Mac due to HyenaDNA MPS incompatibility with PyTorch Lightning device movement
    accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
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
        num_sanity_val_steps=0,
        limit_val_batches=config.limit_val_batches,
        limit_train_batches=config.limit_train_batches,
        gradient_clip_algorithm="norm",
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