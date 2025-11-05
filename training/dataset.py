
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import pyarrow.parquet as pq
import random
import numpy as np
import pandas as pd
import json

from typing import List, Dict, Optional, Tuple, Union, Any

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
