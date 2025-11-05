# Testing Guide for train.py

## 🧪 Testing Strategy

There are several levels of testing, from quick smoke tests to full training runs.

---

## Level 1: Import & Syntax Check (30 seconds)

**Verify the script loads without errors:**

```bash
poetry run python -c "from training.train import *; print('✅ All imports successful')"
```

**Expected output:**
```
✅ All imports successful
```

---

## Level 2: Dataset Loading Test (2-5 minutes)

**Test that the dataset can load data correctly:**

```bash
poetry run python << 'EOF'
from training.train import Config, BetaDogmaDataset, CharacterTokenizer
from pathlib import Path

# Load config
config = Config()

# Create tokenizer
tokenizer = CharacterTokenizer(config.max_seq_len)

# Load training dataset
train_files = sorted((config.data_dir / 'train').glob('*.parquet'))
print(f"Found {len(train_files)} training files")

if train_files:
    dataset = BetaDogmaDataset(train_files[:1], tokenizer, config.max_seq_len, mode='train')
    print(f"✅ Dataset created: {len(dataset)} examples")
    
    # Try loading one example
    example = dataset[0]
    print(f"✅ Example loaded successfully")
    print(f"   - input_ids shape: {example['input_ids'].shape}")
    print(f"   - attention_mask shape: {example['attention_mask'].shape}")
    print(f"   - labels keys: {list(example['labels'].keys())}")
    print(f"   - augmentation_mode: {example.get('augmentation_mode', 'N/A')}")
else:
    print("❌ No training files found")
EOF
```

**Expected output:**
```
Found X training files
✅ Dataset created: Y examples
✅ Example loaded successfully
   - input_ids shape: torch.Size([300000])
   - attention_mask shape: torch.Size([300000])
   - labels keys: ['donor', 'acceptor', 'tss', 'polya', 'protein', 'cds_start', 'cds_end', 'nmd', 'expression', 'variant_effect']
   - augmentation_mode: reference (or benign/pathogenic)
```

---

## Level 3: Model Initialization Test (1 minute)

**Test that the model can be created and run forward pass:**

```bash
poetry run python << 'EOF'
import torch
from training.train import Config, BetaDogmaModel

# Load config
config = Config()

# Create model
print("Creating model...")
model = BetaDogmaModel(config)
print("✅ Model created successfully")

# Create dummy input
batch_size = 1
seq_len = 300000
input_ids = torch.randint(0, 5, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

# Forward pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(input_ids, attention_mask)

print("✅ Forward pass successful")
print(f"   Output keys: {list(outputs.keys())}")
for key, val in outputs.items():
    print(f"   - {key}: {val.shape}")
EOF
```

**Expected output:**
```
Creating model...
✅ Model created successfully
Running forward pass...
✅ Forward pass successful
   Output keys: ['donor', 'acceptor', 'tss', 'polya', 'protein', 'cds_start', 'cds_end', 'nmd', 'expression']
   - donor: torch.Size([1, 300000])
   - acceptor: torch.Size([1, 300000])
   - tss: torch.Size([1, 300000])
   - polya: torch.Size([1, 300000])
   - protein: torch.Size([1, 300000, 21])
   - cds_start: torch.Size([1, 300000])
   - cds_end: torch.Size([1, 300000])
   - nmd: torch.Size([1])
   - expression: torch.Size([1])
```

---

## Level 4: Quick Training Test (5-15 minutes)

**Test training on 1 batch to catch runtime errors:**

```bash
poetry run python training/train.py \
    --limit_train_batches 1 \
    --limit_val_batches 1 \
    --max_epochs 1
```

**What this tests:**
- ✅ Data loading works
- ✅ Model forward pass works
- ✅ Loss computation works
- ✅ Backward pass works
- ✅ Optimizer step works
- ✅ Logging works

**Expected output:**
```
🔍 GPU: [GPU_NAME] ([MEMORY] GB)
✅ BetaDogmaModel initialized
   Encoder: [model_name] (768D)
   Phase 1: protein, CDS, NMD, expression heads added

Epoch 1: 100%|████| 1/1 [00:XX<00:00, XXs/it]

train/loss: 0.45
train/loss/donor: 0.05
train/loss/acceptor: 0.04
train/loss/tss: 0.02
train/loss/polya: 0.02
train/loss/protein: 0.15
train/loss/cds_start: 0.05
train/loss/cds_end: 0.05
train/loss/nmd: 0.02
train/loss/expression: 0.05
train/loss/splice_effect: 0.01
train/loss/variant_effect: 0.00

✅ Training completed successfully
```

---

## Level 5: Short Training Run (30-60 minutes)

**Test training on a few batches to verify convergence:**

```bash
poetry run python training/train.py \
    --limit_train_batches 10 \
    --limit_val_batches 5 \
    --max_epochs 2
```

**What this tests:**
- ✅ Training loop stability
- ✅ Loss convergence
- ✅ Validation works
- ✅ Checkpointing works
- ✅ Memory management works

**Expected behavior:**
- Loss should decrease over batches
- Validation loss should be similar to training loss
- No out-of-memory errors
- Checkpoints saved

---

## Level 6: Full Training Run

**Full training on all data:**

```bash
poetry run python training/train.py
```

**Configuration in `config.yaml` controls:**
- `max_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `limit_train_batches`: Fraction of training data (1.0 = all)
- `limit_val_batches`: Fraction of validation data (1.0 = all)

---

## 🔍 Debugging Tests

### Test 1: Check Dataset Augmentation

**Verify Phase 2A/2B augmentation is working:**

```bash
poetry run python << 'EOF'
from training.train import Config, BetaDogmaDataset, CharacterTokenizer
from pathlib import Path

config = Config()
tokenizer = CharacterTokenizer(config.max_seq_len)
train_files = sorted((config.data_dir / 'train').glob('*.parquet'))[:1]

if train_files:
    dataset = BetaDogmaDataset(train_files, tokenizer, config.max_seq_len, mode='train')
    
    # Load multiple examples to see augmentation modes
    modes = {'reference': 0, 'benign': 0, 'pathogenic': 0}
    
    for i in range(30):
        example = dataset[i]
        mode = example.get('augmentation_mode', 'unknown')
        modes[mode] = modes.get(mode, 0) + 1
    
    print("Augmentation mode distribution (should be ~33/33/33):")
    for mode, count in sorted(modes.items()):
        print(f"  {mode}: {count}/30 ({count/30*100:.1f}%)")
EOF
```

**Expected output:**
```
Augmentation mode distribution (should be ~33/33/33):
  benign: 10/30 (33.3%)
  pathogenic: 10/30 (33.3%)
  reference: 10/30 (33.3%)
```

### Test 2: Check Loss Computation

**Verify all losses are computed correctly:**

```bash
poetry run python << 'EOF'
import torch
from training.train import Config, BetaDogmaModel, BetaDogmaLightning

config = Config()
model = BetaDogmaLightning(config)

# Create dummy batch
batch = {
    'input_ids': torch.randint(0, 5, (1, 300000)),
    'attention_mask': torch.ones(1, 300000),
    'labels': {
        'donor': torch.zeros(1, 300000),
        'acceptor': torch.zeros(1, 300000),
        'tss': torch.zeros(1, 300000),
        'polya': torch.zeros(1, 300000),
        'protein': torch.full((1, 300000), -1, dtype=torch.long),
        'cds_start': torch.zeros(1, 300000),
        'cds_end': torch.zeros(1, 300000),
        'nmd': torch.zeros(1),
        'expression': torch.zeros(1),
        'variant_effect': torch.zeros(1),
    }
}

# Compute loss
outputs = model(batch['input_ids'], batch['attention_mask'])
loss_dict = model._compute_loss(outputs, batch)

print("Loss computation successful!")
print("Loss terms:")
for key, val in loss_dict.items():
    print(f"  {key}: {val.item():.4f}")
EOF
```

**Expected output:**
```
Loss computation successful!
Loss terms:
  loss: 0.XXXX
  loss/donor: 0.XXXX
  loss/acceptor: 0.XXXX
  loss/tss: 0.XXXX
  loss/polya: 0.XXXX
  loss/splice_effect: 0.XXXX
  loss/protein: 0.XXXX
  loss/cds_start: 0.XXXX
  loss/cds_end: 0.XXXX
  loss/nmd: 0.XXXX
  loss/expression: 0.XXXX
  loss/variant_effect: 0.XXXX
```

---

## 📋 Recommended Testing Sequence

### First Time Setup:
1. **Level 1**: Import check (30 sec)
2. **Level 2**: Dataset loading (2-5 min)
3. **Level 3**: Model initialization (1 min)
4. **Level 4**: Quick training test (5-15 min)

### Before Full Training:
5. **Level 5**: Short training run (30-60 min)
6. **Debug Test 1**: Augmentation check
7. **Debug Test 2**: Loss computation check

### Full Training:
8. **Level 6**: Full training run

---

## 🚀 Quick Test Commands

**Copy-paste these for quick testing:**

```bash
# 1. Import check
poetry run python -c "from training.train import *; print('✅ OK')"

# 2. Quick training (1 batch, 1 epoch)
poetry run python training/train.py --limit_train_batches 1 --limit_val_batches 1 --max_epochs 1

# 3. Short training (10 batches, 2 epochs)
poetry run python training/train.py --limit_train_batches 10 --limit_val_batches 5 --max_epochs 2

# 4. Full training
poetry run python training/train.py
```

---

## ⚠️ Common Issues & Solutions

### Issue: "No valid data files found"
**Solution**: Check that data files exist in `data/processed/train/`, `data/processed/val/`, `data/processed/test/`

### Issue: "CUDA out of memory"
**Solution**: Reduce `batch_size` or `max_seq_len` in `config.yaml`

### Issue: "Model not converging"
**Solution**: Check learning rate, try different `learning_rate` in `config.yaml`

### Issue: "Augmentation modes not balanced"
**Solution**: Increase number of samples (should converge to 33/33/33 with more data)

---

## 📊 Expected Performance

### Quick Test (1 batch):
- Should complete in < 2 minutes
- All losses should be computed
- No errors

### Short Test (10 batches, 2 epochs):
- Should complete in 30-60 minutes
- Loss should decrease
- Validation loss similar to training loss

### Full Training:
- Depends on dataset size
- Monitor loss curves in TensorBoard
- Check checkpoints in `outputs/checkpoints/`

---

## 🎯 Verification Checklist

After each test level, verify:
- ✅ No errors in console
- ✅ All losses computed
- ✅ Memory usage reasonable
- ✅ Training progresses
- ✅ Checkpoints saved

**Ready to test!** Start with Level 1 and work your way up. 🚀
