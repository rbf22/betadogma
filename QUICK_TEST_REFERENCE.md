# Quick Test Reference Card

## 🚀 One-Liners for Testing

### Level 1: Import Check (30 sec)
```bash
poetry run python -c "from training.train import *; print('✅ Imports OK')"
```

### Level 2: Dataset Test (2-5 min)
```bash
poetry run python << 'EOF'
from training.train import Config, BetaDogmaDataset, CharacterTokenizer
from pathlib import Path
config = Config()
tokenizer = CharacterTokenizer(config.max_seq_len)
train_files = sorted((config.data_dir / 'train').glob('*.parquet'))[:1]
dataset = BetaDogmaDataset(train_files, tokenizer, config.max_seq_len, mode='train')
example = dataset[0]
print(f"✅ Dataset OK: {len(dataset)} examples")
print(f"   Labels: {list(example['labels'].keys())}")
EOF
```

### Level 3: Model Test (1 min)
```bash
poetry run python << 'EOF'
import torch
from training.train import Config, BetaDogmaModel
config = Config()
model = BetaDogmaModel(config)
input_ids = torch.randint(0, 5, (1, 300000))
attention_mask = torch.ones(1, 300000)
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
print(f"✅ Model OK: {len(outputs)} outputs")
EOF
```

### Level 4: Quick Training (5-15 min)
```bash
poetry run python training/train.py \
    --limit_train_batches 1 \
    --limit_val_batches 1 \
    --max_epochs 1
```

### Level 5: Short Training (30-60 min)
```bash
poetry run python training/train.py \
    --limit_train_batches 10 \
    --limit_val_batches 5 \
    --max_epochs 2
```

### Level 6: Full Training
```bash
poetry run python training/train.py
```

---

## 🔍 Debugging Commands

### Check Augmentation Distribution
```bash
poetry run python << 'EOF'
from training.train import Config, BetaDogmaDataset, CharacterTokenizer
from pathlib import Path
config = Config()
tokenizer = CharacterTokenizer(config.max_seq_len)
train_files = sorted((config.data_dir / 'train').glob('*.parquet'))[:1]
dataset = BetaDogmaDataset(train_files, tokenizer, config.max_seq_len, mode='train')
modes = {}
for i in range(30):
    mode = dataset[i].get('augmentation_mode', 'unknown')
    modes[mode] = modes.get(mode, 0) + 1
print("Augmentation modes (should be ~33/33/33):")
for mode, count in sorted(modes.items()):
    print(f"  {mode}: {count}/30 ({count/30*100:.1f}%)")
EOF
```

### Check Loss Computation
```bash
poetry run python << 'EOF'
import torch
from training.train import Config, BetaDogmaLightning
config = Config()
model = BetaDogmaLightning(config)
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
outputs = model(batch['input_ids'], batch['attention_mask'])
loss_dict = model._compute_loss(outputs, batch)
print("✅ Loss computation OK")
for key, val in loss_dict.items():
    print(f"  {key}: {val.item():.4f}")
EOF
```

---

## 📊 Expected Outputs

### Level 1: Import Check
```
✅ Imports OK
```

### Level 2: Dataset Test
```
✅ Dataset OK: 190 examples
   Labels: ['donor', 'acceptor', 'tss', 'polya', 'protein', 'cds_start', 'cds_end', 'nmd', 'expression', 'variant_effect']
```

### Level 3: Model Test
```
✅ Model OK: 9 outputs
```

### Level 4: Quick Training
```
Epoch 1: 100%|████| 1/1 [00:XX<00:00, XXs/it]
train/loss: 0.45
train/loss/donor: 0.05
...
✅ Training completed successfully
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No valid data files found" | Check `data/processed/train/` exists |
| "CUDA out of memory" | Reduce `batch_size` in `config.yaml` |
| "Module not found" | Run `poetry install` |
| "Augmentation not balanced" | Use more samples (converges to 33/33/33) |
| "Loss is NaN" | Check label ranges, try different learning rate |

---

## 🎯 Recommended Testing Path

1. **First time**: Level 1 → 2 → 3 → 4
2. **Before full training**: Level 5 + Debug tests
3. **Production**: Level 6

**Total time**: ~1 hour for full test suite

---

## 📝 Notes

- All tests use `poetry run` to ensure correct environment
- Adjust `--limit_train_batches` and `--limit_val_batches` for faster testing
- Check `outputs/logs/` for TensorBoard logs
- Check `outputs/checkpoints/` for saved models

**Start with Level 1!** 🚀
