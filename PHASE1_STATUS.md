# Phase 1 Implementation Status

## ✅ Completed

### 1. Configuration
- ✅ Updated `config.yaml`: `max_seq_len = 300000`

### 2. Helper Functions  
- ✅ Created `training/dataset_helpers.py` with all helper functions
- ✅ Created `training/dataset_getitem.py` with new `__getitem__`
- ✅ Tested all helpers with `training/test_dataset_phase1.py` - ALL TESTS PASS

### 3. Documentation
- ✅ `TRAINING_ANALYSIS.md` - Complete analysis
- ✅ `PHASE1_INTEGRATION_GUIDE.md` - Step-by-step guide
- ✅ `PHASE1_COMPLETE_SUMMARY.md` - Summary
- ✅ `VARIANT_AUGMENTATION_DESIGN.md` - Phase 2 design
- ✅ `PHASE2_VARIANT_STRATEGY.md` - Phase 2 strategy
- ✅ `VARIANT_EFFECTS_CLARIFICATION.md` - Sequence vs function

### 4. Backup
- ✅ `training/train_backup.py` - Original saved

## ⏳ In Progress

### Integration into train.py

The file `train.py` became too messy during editing. We need a clean integration.

**Recommended Approach**: Create a new clean `train_v2.py` with all Phase 1 changes, then replace `train.py`.

## 📋 What Needs to Be Done

### Step 1: Create Clean train_v2.py

Copy from `train_backup.py` and make these changes:

1. **Dataset Class** (lines ~238-732):
   - Add helper methods from `dataset_helpers.py`
   - Replace `__getitem__` with version from `dataset_getitem.py`

2. **Model Heads** (after line ~738):
   - Add `ProteinPredictionHead` class
   - Add `ScalarPredictionHead` class

3. **BetaDogmaModel** (lines ~1000-1100):
   - Add 5 new prediction heads:
     - `self.protein_head`
     - `self.cds_start_head`
     - `self.cds_end_head`
     - `self.nmd_head`
     - `self.expression_head`
   - Update `forward()` to return all outputs

4. **Config Class** (lines ~100-200):
   - Add `protein_hidden = 256`
   - Add `protein_layers = 2`
   - Add loss weights for new tasks

5. **Loss Computation** (lines ~1200-1300):
   - Add `loss_protein` (CrossEntropyLoss, ignore_index=-1)
   - Add `loss_cds_start` (BCEWithLogitsLoss)
   - Add `loss_cds_end` (BCEWithLogitsLoss)
   - Add `loss_nmd` (BCEWithLogitsLoss)
   - Add `loss_expression` (MSELoss)

### Step 2: Test Phase 1

```bash
poetry run python training/train_v2.py \
    --limit_train_batches 1 \
    --limit_val_batches 1 \
    --max_epochs 1
```

### Step 3: Move to Phase 2

Once Phase 1 works, add variant augmentation.

## 🎯 Quick Start Guide

Since the file got messy, here's the fastest path forward:

### Option A: Manual Integration (30 min)
1. Open `train_backup.py`
2. Follow `PHASE1_INTEGRATION_GUIDE.md` step by step
3. Test each change

### Option B: Script Integration (5 min)
1. Run integration script (to be created)
2. Test immediately

### Option C: Use Prepared Files (1 min)
1. Copy helper code from `dataset_helpers.py` into dataset class
2. Copy `__getitem__` from `dataset_getitem.py`
3. Add new model heads (see guide)

## 📊 Expected Results After Phase 1

```python
# Model outputs:
{
    'donor': [B, L],           # Existing
    'acceptor': [B, L],        # Existing
    'tss': [B, L],             # Existing
    'polya': [B, L],           # Existing
    'protein': [B, L, 21],     # NEW
    'cds_start': [B, L],       # NEW
    'cds_end': [B, L],         # NEW
    'nmd': [B],                # NEW
    'expression': [B]          # NEW
}

# Loss components:
{
    'loss/donor': ...,
    'loss/acceptor': ...,
    'loss/tss': ...,
    'loss/polya': ...,
    'loss/protein': ...,       # NEW
    'loss/cds_start': ...,     # NEW
    'loss/cds_end': ...,       # NEW
    'loss/nmd': ...,           # NEW
    'loss/expression': ...     # NEW
}
```

## 🚀 Next: Phase 2

After Phase 1 works:
1. Add variant augmentation (33/33/33 strategy)
2. Implement protein sequence recomputation
3. Add variant effect prediction
4. Train full model

## Files Reference

- `training/dataset_helpers.py` - All helper functions ✅
- `training/dataset_getitem.py` - New `__getitem__` ✅
- `training/test_dataset_phase1.py` - Tests (all pass) ✅
- `training/train_backup.py` - Clean original ✅
- `training/train.py` - Needs clean integration ⏳
- `PHASE1_INTEGRATION_GUIDE.md` - Step-by-step guide ✅

## Decision Point

**Recommend**: I can create a clean `train_v2.py` with all Phase 1 changes integrated properly. This will take ~10 minutes and give you a working, tested implementation.

**Alternative**: You can manually integrate following the guide (30 min).

What would you prefer?
