# Phase 1 Integration Guide

## Status: Ready to Integrate

I've created helper files and identified all changes needed. Due to the size of train.py, I recommend integrating in steps.

## Files Created

1. ✅ `training/dataset_helpers.py` - All helper functions
2. ✅ `training/dataset_getitem.py` - New __getitem__ implementation  
3. ✅ `training/train_backup.py` - Backup of original
4. ✅ `config.yaml` - Updated (max_seq_len = 300000)

## Integration Steps

### Step 1: Clean Up Dataset Class (Lines 238-732)

**Remove**: All duplicate/broken methods between lines 330-640
**Keep**: Lines 238-329 (clean __init__ and __len__)
**Add**: Helper methods from dataset_helpers.py
**Replace**: __getitem__ with version from dataset_getitem.py

### Step 2: Add New Model Heads (After line 915)

Add these new prediction heads to BetaDogmaModel:

```python
# NEW: Protein prediction head (21-way classification)
self.protein_head = ProteinPredictionHead(
    self.encoder_dim,
    config.protein_hidden,  # 256
    config.protein_layers,  # 2
    num_classes=21,  # 20 AA + stop
    dropout=config.dropout,
    use_checkpointing=use_checkpointing
).to(self.device)

# NEW: CDS boundary prediction heads
self.cds_start_head = PredictionHead(
    self.encoder_dim,
    config.splice_hidden,
    config.splice_layers,
    config.dropout,
    use_checkpointing
).to(self.device)

self.cds_end_head = PredictionHead(
    self.encoder_dim,
    config.splice_hidden,
    config.splice_layers,
    config.dropout,
    use_checkpointing
).to(self.device)

# NEW: NMD prediction head (scalar)
self.nmd_head = ScalarPredictionHead(
    self.encoder_dim,
    hidden_dim=128,
    dropout=config.dropout
).to(self.device)

# NEW: Expression prediction head (scalar)
self.expression_head = ScalarPredictionHead(
    self.encoder_dim,
    hidden_dim=128,
    dropout=config.dropout
).to(self.device)
```

### Step 3: Update Model Forward Pass (Line ~990)

Add to the outputs dict:

```python
outputs = {
    'donor': donor_logits,
    'acceptor': acceptor_logits,
    'tss': tss_logits,
    'polya': polya_logits,
    # NEW Phase 1 outputs:
    'protein': self.protein_head(embeddings),      # [B, L, 21]
    'cds_start': self.cds_start_head(embeddings),  # [B, L]
    'cds_end': self.cds_end_head(embeddings),      # [B, L]
    'nmd': self.nmd_head(embeddings),              # [B]
    'expression': self.expression_head(embeddings) # [B]
}
```

### Step 4: Update Loss Computation (Line ~1050)

Add new loss terms in `_compute_loss`:

```python
# NEW: Protein sequence prediction (cross-entropy, ignore -1)
if 'protein' in labels:
    loss_protein = F.cross_entropy(
        outputs['protein'].view(-1, 21),  # [B*L, 21]
        labels['protein'].view(-1),        # [B*L]
        ignore_index=-1  # Ignore non-CDS positions
    )
else:
    loss_protein = torch.tensor(0.0, device=self.device)

# NEW: CDS boundary prediction
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

# NEW: NMD prediction (binary classification)
loss_nmd = F.binary_cross_entropy_with_logits(
    outputs['nmd'],
    labels['nmd']
)

# NEW: Expression prediction (regression on log TPM)
loss_expression = F.mse_loss(
    outputs['expression'],
    labels['expression']
)

# Update total loss
loss = (
    self.config.w_splice_donor * loss_donor +
    self.config.w_splice_acceptor * loss_acceptor +
    self.config.w_tss * loss_tss +
    self.config.w_polya * loss_polya +
    self.config.w_protein * loss_protein +           # NEW
    self.config.w_cds_start * loss_cds_start +       # NEW
    self.config.w_cds_end * loss_cds_end +           # NEW
    self.config.w_nmd * loss_nmd +                   # NEW
    self.config.w_expression * loss_expression       # NEW
)

# Add to loss dict for logging
return {
    'loss': loss,
    'loss/donor': loss_donor,
    'loss/acceptor': loss_acceptor,
    'loss/tss': loss_tss,
    'loss/polya': loss_polya,
    'loss/protein': loss_protein,           # NEW
    'loss/cds_start': loss_cds_start,       # NEW
    'loss/cds_end': loss_cds_end,           # NEW
    'loss/nmd': loss_nmd,                   # NEW
    'loss/expression': loss_expression,     # NEW
}
```

### Step 5: Add New Model Components

Before BetaDogmaModel class, add:

```python
class ProteinPredictionHead(nn.Module):
    """Protein sequence prediction head (21-way classification per position)."""
    
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int,
                 num_classes: int = 21, dropout: float = 0.1, 
                 use_checkpointing: bool = False):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def _forward_impl(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # [B, L, 21]
        return logits
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class ScalarPredictionHead(nn.Module):
    """Scalar prediction head (e.g., for NMD, expression)."""
    
    def __init__(self, d_in: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.pool(x).squeeze(-1)  # [B, D]
        return self.fc(x).squeeze(-1)  # [B]
```

### Step 6: Update Config Class

Add to Config.__init__ (around line 140):

```python
# NEW: Task-specific head dimensions for Phase 1
self.protein_hidden = 256
self.protein_layers = 2

# NEW: Loss weights for Phase 1 tasks
self.w_protein = 2.0
self.w_cds_start = 0.5
self.w_cds_end = 0.5
self.w_nmd = 1.0
self.w_expression = 1.0
```

## Testing

After integration, test with:

```bash
# Quick test on 1 batch
poetry run python training/train.py \
    --limit_train_batches 1 \
    --limit_val_batches 1 \
    --max_epochs 1

# Check that all losses are computed
# Look for these in logs:
# - train/loss/protein
# - train/loss/cds_start
# - train/loss/cds_end
# - train/loss/nmd
# - train/loss/expression
```

## Expected Output Shape

After Phase 1, model outputs:

```python
{
    'donor': [B, L],           # Splice donor logits
    'acceptor': [B, L],        # Splice acceptor logits
    'tss': [B, L],             # TSS logits
    'polya': [B, L],           # PolyA logits
    'protein': [B, L, 21],     # Protein logits (21 classes)
    'cds_start': [B, L],       # CDS start logits
    'cds_end': [B, L],         # CDS end logits
    'nmd': [B],                # NMD probability
    'expression': [B]          # Log TPM prediction
}
```

## Next: Would you like me to:

A) Create a script that does the integration automatically
B) Walk through each step manually with you
C) Create a minimal test script first to verify the data loads correctly

Recommend: **C** - Test data loading first before modifying the model
