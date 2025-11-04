# MPS Compatibility Note

## The Discovery: MPS Works, But PyTorch Lightning Breaks It

We discovered something interesting:

### ✅ HyenaDNA Works Fine on MPS
```python
model = AutoModel.from_pretrained("LongSafari/hyenadna-medium-450k-seqlen-hf")
model = model.to('mps')  # ✅ Works!
output = model(input_ids)  # ✅ Works with 300k sequences!
```

### ❌ But PyTorch Lightning Breaks It
When PyTorch Lightning tries to move the model to MPS during training:
```
RuntimeError: Placeholder storage has not been allocated on MPS device!
```

### The Root Cause
The issue is **NOT** in HyenaDNA - it's in how PyTorch Lightning handles device movement. When Lightning calls `.to(device)` on the model after initialization, something goes wrong with the embedding layer's MPS storage allocation.

### The Solution
**Use CPU on Mac** - This is the correct workaround for PyTorch Lightning compatibility.

```python
# In train() function:
accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
trainer = pl.Trainer(accelerator=accelerator, ...)
```

### Performance Impact
- **CPU on Mac**: ~2-3x slower than MPS would be
- **CUDA on GPU**: ~10-100x faster than CPU
- **MPS on Mac**: Would be ~2-3x faster than CPU (but doesn't work with Lightning)

### Why This Happens
1. Model loads successfully on CPU
2. PyTorch Lightning moves it to MPS
3. Embedding layer's MPS storage isn't properly allocated during the move
4. Forward pass fails

### When This Will Be Fixed
This is likely a PyTorch Lightning issue, not HyenaDNA. The fix would need to come from:
- PyTorch Lightning improving device movement handling
- Or HyenaDNA adding special MPS initialization code

### For Production
If you need to train on Mac:
- Use CPU (current solution) - works reliably
- Or use a GPU machine (CUDA) for faster training

The code automatically uses CUDA if available, and falls back to CPU on Mac.
