"""Clean __getitem__ implementation for BetaDogmaDataset"""

import json
import pandas as pd
import torch
import numpy as np
from dataset_helpers import (
    parse_isoforms, create_protein_labels, create_cds_boundary_labels,
    extract_canonical_isoform, to_tensor
)

def __getitem__(self, idx):
    """Get a single training example with all Central Dogma labels.
    
    Returns dict with:
        - input_ids: Tokenized DNA sequence
        - attention_mask: Attention mask
        - labels: Dict with all prediction targets
    """
    # Find which file and row contains this index
    file_path, row_idx = self._get_file_and_row(idx)
    
    try:
        # Read the row from parquet
        # Load ALL fields including new isoforms and variants
        df = pd.read_parquet(
            file_path,
            columns=[
                'seq', 'chrom', 'start', 'end',
                'transcript_id', 'gene_name', 'strand',
                'donor', 'acceptor', 'tss', 'polya',
                'isoforms',  # NEW: Rich isoform metadata
                'variants'   # NEW: Variant metadata
            ]
        )
        
        if len(df) <= row_idx:
            raise IndexError(f"Row {row_idx} not found in {file_path}")
        
        row = df.iloc[row_idx]
        
        # ====================================================================
        # 1. PROCESS SEQUENCE
        # ====================================================================
        seq = str(row['seq'])
        if not seq or len(seq) == 0:
            seq = 'N' * self.max_seq_len
        
        # Ensure sequence is exactly max_seq_len
        if len(seq) > self.max_seq_len:
            # Center crop
            start = (len(seq) - self.max_seq_len) // 2
            seq = seq[start:start + self.max_seq_len]
        elif len(seq) < self.max_seq_len:
            # Center pad
            pad_left = (self.max_seq_len - len(seq)) // 2
            pad_right = self.max_seq_len - len(seq) - pad_left
            seq = ('N' * pad_left) + seq + ('N' * pad_right)
        
        # Tokenize
        tokenized = self.tokenizer(
            seq,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
        
        # ====================================================================
        # 2. PROCESS SPLICE SITE LABELS (existing)
        # ====================================================================
        labels = {
            'donor': to_tensor(row['donor'], self.max_seq_len),
            'acceptor': to_tensor(row['acceptor'], self.max_seq_len),
            'tss': to_tensor(row['tss'], self.max_seq_len),
            'polya': to_tensor(row['polya'], self.max_seq_len),
        }
        
        # ====================================================================
        # 3. PROCESS ISOFORM DATA (NEW)
        # ====================================================================
        isoform_data = parse_isoforms(row.get('isoforms', '[]'))
        canonical = extract_canonical_isoform(isoform_data)
        
        # 3a. Protein sequence labels
        protein_labels = create_protein_labels(
            canonical['protein'],
            canonical['cds_start'],
            canonical['cds_end'],
            self.max_seq_len
        )
        labels['protein'] = torch.from_numpy(protein_labels)
        
        # 3b. CDS boundary labels
        cds_start_labels, cds_end_labels = create_cds_boundary_labels(
            canonical['cds_start'],
            canonical['cds_end'],
            self.max_seq_len
        )
        labels['cds_start'] = torch.from_numpy(cds_start_labels)
        labels['cds_end'] = torch.from_numpy(cds_end_labels)
        
        # 3c. NMD label (scalar)
        labels['nmd'] = torch.tensor(float(canonical['nmd']), dtype=torch.float32)
        
        # 3d. Expression label (log TPM, scalar)
        labels['expression'] = torch.tensor(
            np.log1p(canonical['tpm']),  # log(TPM + 1)
            dtype=torch.float32
        )
        
        # ====================================================================
        # 4. PROCESS VARIANT DATA (NEW - for future variant effect prediction)
        # ====================================================================
        # For now, just store variant count as metadata
        # In Phase 3, we'll use this for variant augmentation
        try:
            variants = json.loads(row.get('variants', '[]'))
            labels['num_variants'] = torch.tensor(len(variants), dtype=torch.long)
        except:
            labels['num_variants'] = torch.tensor(0, dtype=torch.long)
        
        # ====================================================================
        # 5. RETURN COMPLETE EXAMPLE
        # ====================================================================
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels,
            # Metadata (not used in training, but useful for debugging)
            'metadata': {
                'chrom': row.get('chrom', ''),
                'start': int(row.get('start', 0)),
                'end': int(row.get('end', 0)),
                'gene_name': row.get('gene_name', ''),
            }
        }
        
    except Exception as e:
        print(f"⚠️  Error loading example {idx}: {e}")
        
        # Return dummy example on error
        tokenized = self.tokenizer(
            'N' * self.max_seq_len,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
        
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
                'num_variants': torch.tensor(0, dtype=torch.long),
            },
            'metadata': {
                'chrom': '',
                'start': 0,
                'end': 0,
                'gene_name': '',
            }
        }
