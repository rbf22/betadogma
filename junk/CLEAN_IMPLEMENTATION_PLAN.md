# Clean Implementation Plan - Phase 1 + 2

## ✅ Files Created

1. **`training/model_helpers.py`** ✅
   - `PredictionHead` - Binary classification
   - `ProteinPredictionHead` - 21-way classification
   - `ScalarPredictionHead` - NMD, expression
   - `HyenaDNAEncoder` - Encoder wrapper

2. **`training/dataset_helpers.py`** ✅ (already exists)
   - `parse_isoforms()`
   - `extract_canonical_isoform()`
   - `create_protein_labels()`
   - `create_cds_boundary_labels()`
   - `to_tensor()`

## 📋 What to Add to train.py

### Section 1: Imports (Lines 1-50)
```python
#!/usr/bin/env python3
import os, json, random, sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# Import helpers
from training.model_helpers import *
from training.dataset_helpers import *

# Load config
CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)
```

### Section 2: Config Class (Lines 51-150)
Copy from `old/train_backup.py` lines 52-200, then add:
```python
# NEW Phase 1 parameters
self.protein_hidden = 256
self.protein_layers = 2

# NEW Phase 1 loss weights
self.w_protein = 2.0
self.w_cds_start = 0.5
self.w_cds_end = 0.5
self.w_nmd = 1.0
self.w_expression = 1.0
```

### Section 3: Tokenizer (Lines 151-180)
Copy from backup, change max_length to 300000

### Section 4: Dataset Class (Lines 181-600)
```python
class BetaDogmaDataset(Dataset):
    AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY*'
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
    
    def __init__(self, parquet_files, tokenizer, max_seq_len=300000, ...):
        # Copy from backup lines 270-328
        
    def __len__(self):
        return self.length
    
    def _get_file_and_row(self, idx):
        # Copy from backup lines 333-353
    
    # Import all helper methods from dataset_helpers.py
    parse_isoforms = staticmethod(parse_isoforms)
    extract_canonical_isoform = staticmethod(extract_canonical_isoform)
    create_protein_labels = classmethod(create_protein_labels)
    create_cds_boundary_labels = staticmethod(create_cds_boundary_labels)
    to_tensor = staticmethod(to_tensor)
    
    def __getitem__(self, idx):
        # Copy from dataset_getitem.py lines 13-165
```

### Section 5: Model Class (Lines 601-800)
```python
class BetaDogmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder
        self.encoder = HyenaDNAEncoder(config.model_name, freeze=True, device=self.device)
        self.encoder_dim = config.encoder_dim
        
        # Existing heads
        self.donor_head = PredictionHead(self.encoder_dim, config.splice_hidden, ...)
        self.acceptor_head = PredictionHead(...)
        self.tss_head = PredictionHead(...)
        self.polya_head = PredictionHead(...)
        
        # NEW Phase 1 heads
        self.protein_head = ProteinPredictionHead(self.encoder_dim, config.protein_hidden, ...)
        self.cds_start_head = PredictionHead(...)
        self.cds_end_head = PredictionHead(...)
        self.nmd_head = ScalarPredictionHead(self.encoder_dim, 128)
        self.expression_head = ScalarPredictionHead(self.encoder_dim, 128)
    
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.encoder(input_ids, attention_mask)
        
        return {
            'donor': self.donor_head(embeddings),
            'acceptor': self.acceptor_head(embeddings),
            'tss': self.tss_head(embeddings),
            'polya': self.polya_head(embeddings),
            'protein': self.protein_head(embeddings),
            'cds_start': self.cds_start_head(embeddings),
            'cds_end': self.cds_end_head(embeddings),
            'nmd': self.nmd_head(embeddings),
            'expression': self.expression_head(embeddings),
        }
```

### Section 6: Lightning Module (Lines 801-1100)
```python
class BetaDogmaLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BetaDogmaModel(config)
        self.pos_weight = torch.tensor(config.pos_weight)
    
    def _compute_loss(self, outputs, batch):
        labels = batch['labels']
        
        # Existing losses
        loss_donor = F.binary_cross_entropy_with_logits(...)
        loss_acceptor = F.binary_cross_entropy_with_logits(...)
        loss_tss = F.binary_cross_entropy_with_logits(...)
        loss_polya = F.binary_cross_entropy_with_logits(...)
        
        # NEW Phase 1 losses
        loss_protein = F.cross_entropy(
            outputs['protein'].view(-1, 21),
            labels['protein'].view(-1),
            ignore_index=-1
        )
        loss_cds_start = F.binary_cross_entropy_with_logits(...)
        loss_cds_end = F.binary_cross_entropy_with_logits(...)
        loss_nmd = F.binary_cross_entropy_with_logits(...)
        loss_expression = F.mse_loss(...)
        
        # Total loss
        loss = (
            self.config.w_splice_donor * loss_donor +
            self.config.w_splice_acceptor * loss_acceptor +
            self.config.w_tss * loss_tss +
            self.config.w_polya * loss_polya +
            self.config.w_protein * loss_protein +
            self.config.w_cds_start * loss_cds_start +
            self.config.w_cds_end * loss_cds_end +
            self.config.w_nmd * loss_nmd +
            self.config.w_expression * loss_expression
        )
        
        return loss, {...}  # Return all losses for logging
```

## 🎯 Quick Build Script

Run this to build train.py:

```bash
cd /Users/robert_fenwick/SWE/betadogma

# Copy structure from backup
head -50 training/old/train_backup.py > training/train.py

# Add imports
cat >> training/train.py << 'EOF'
from training.model_helpers import *
from training.dataset_helpers import *
EOF

# Then manually add the sections above
```

## ✅ Verification

After building, test with:
```bash
poetry run python -c "from training.train import *; print('✅ Import successful')"
```

## 📊 Expected Result

A clean ~1000 line train.py that:
- ✅ Imports from model_helpers.py and dataset_helpers.py
- ✅ Has Phase 1 dataset with protein/NMD/expression labels
- ✅ Has Phase 1 model with 5 new prediction heads
- ✅ Has Phase 1 loss computation
- ✅ Ready for Phase 2A variant augmentation

## 🚀 Next Session

Once train.py is complete:
1. Test Phase 1 (1 batch)
2. Add Phase 2A variant augmentation
3. Full training run

Want me to create the complete train.py file piece by piece?
