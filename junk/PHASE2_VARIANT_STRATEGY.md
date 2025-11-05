# Phase 2: Variant Augmentation Strategy

## TL;DR - Your Questions Answered

### 1. Do we update transcript/protein when adding variants?

**YES for pathogenic, NO for benign** ✅

```
Reference:     ATCG... → Protein: MRAL... → NMD: False
                ↓
Benign Var:    ATGG... → Protein: MRAL... → NMD: False  (SAME!)
                ↓
Pathogenic:    ATGG... → Protein: MRXL... → NMD: True   (CHANGED!)
```

### 2. How do we use SpliceVar data?

**Three ways** ✅

1. **Ground truth**: `splice_effect_score` (0-1) is the target for variant effect prediction
2. **Label modification**: If score > 0.5, modify splice site labels
3. **Differential training**: Predict change in splicing (ref → alt)

### 3. Are we adding benign variants to keep model on its toes?

**YES - 33/33/33 strategy** ✅

```
Training batch composition:
├── 33% Reference (no variant)
│   └── Teaches: normal sequence patterns
├── 33% Benign variants (1000 Genomes)
│   └── Teaches: most variants are harmless
└── 33% Pathogenic/Splice variants
    └── Teaches: some variants break things
```

### 4. Do we modify predictions for benign variants?

**NO - That's the point!** ✅

```python
if variant_is_benign:
    seq = apply_variant(seq_ref)      # Sequence CHANGES
    labels = labels_ref               # Labels DON'T CHANGE
    # Model learns: "This change is OK"

if variant_is_pathogenic:
    seq = apply_variant(seq_ref)      # Sequence CHANGES
    labels = recompute_labels(seq)    # Labels CHANGE TOO
    # Model learns: "This change breaks things"
```

---

## Implementation Roadmap

### Current State (Phase 1)
```python
# We store but DON'T USE variants
example = {
    'seq': reference_sequence,
    'labels': reference_labels,
    'variants': [...],  # Metadata only!
}
```

### Phase 2A: Basic Augmentation (Recommended Next)
```python
def __getitem__(self, idx):
    seq_ref, labels_ref, variants = load_data(idx)
    
    # Random augmentation
    if random.random() < 0.33:
        return seq_ref, labels_ref  # Reference
    
    elif random.random() < 0.66:
        # Benign variant
        var = random.choice([v for v in variants if v['is_benign']])
        seq_alt = apply_variant(seq_ref, var)
        return seq_alt, labels_ref  # Labels unchanged!
    
    else:
        # Pathogenic variant
        var = random.choice([v for v in variants if v['is_pathogenic']])
        seq_alt = apply_variant(seq_ref, var)
        labels_alt = recompute_labels(seq_alt, var)
        return seq_alt, labels_alt  # Labels changed!
```

### Phase 2B: SpliceVar Integration
```python
# Add variant effect prediction head
model_outputs = {
    'splice_donor': [...],
    'splice_acceptor': [...],
    'variant_effect': 0.85,  # NEW: Predict SpliceVar score
}

# Loss includes variant effect
loss = (
    BCE(outputs['splice_donor'], labels['donor']) +
    MSE(outputs['variant_effect'], variant['splice_effect_score'])
)
```

### Phase 2C: Differential Prediction (Advanced)
```python
# Return both reference and variant
def __getitem__(self, idx):
    return {
        'seq_ref': seq_ref,
        'seq_alt': seq_alt,
        'labels_ref': labels_ref,
        'labels_alt': labels_alt,
        'delta': labels_alt - labels_ref  # What changed?
    }

# Model predicts change
outputs = model(seq_ref, seq_alt)
loss = MSE(outputs['delta'], true_delta)
```

---

## Critical Design Decisions

### Decision 1: Label Recomputation

| Task | Benign Variant | Pathogenic Variant |
|------|----------------|-------------------|
| Splice sites | Keep reference | Recompute if `splice_effect_score > 0.5` |
| Protein | Keep reference | Recompute (translate alt sequence) |
| NMD | Keep reference | Recompute (check alt protein) |
| Expression | Keep reference | Keep reference* |

*We don't have variant-specific expression data

### Decision 2: Recomputation Method

**Simple (Recommended for Phase 2A)**:
```python
def recompute_splice_labels(labels_ref, variant):
    if variant['splice_effect_score'] > 0.5:
        # Flip splice site at variant position
        labels_alt = labels_ref.copy()
        labels_alt[variant['pos']] = 1 - labels_ref[variant['pos']]
    return labels_alt
```

**Complex (Phase 3)**:
```python
def recompute_protein_labels(seq_alt, variant):
    # Re-translate the entire sequence
    protein_alt = translate(seq_alt)
    nmd_alt = check_nmd(protein_alt)
    return protein_alt, nmd_alt
```

### Decision 3: Training Balance

```
Epoch 1: [Ref, Ref, Benign, Path, Ref, Benign, Path, ...]
         └─ Random 33/33/33 mix

Why this works:
- Model sees normal patterns (reference)
- Learns benign variants don't change function
- Learns pathogenic variants do change function
- Prevents overfitting to reference sequence
```

---

## Recommended Next Steps

### Option A: Phase 1 First (Recommended)
1. ✅ Complete Phase 1 model integration
2. ✅ Train on reference data only
3. ✅ Verify all prediction heads work
4. ✅ Then add Phase 2A variant augmentation

**Pros**: 
- Simpler debugging
- Baseline model performance
- Incremental complexity

### Option B: Phase 1 + 2A Together
1. ✅ Integrate Phase 1 model
2. ✅ Add variant augmentation immediately
3. ✅ Train with variants from start

**Pros**:
- More realistic training
- Better final model
- Faster to full system

**Cons**:
- Harder to debug
- More moving parts

---

## My Recommendation

**Do Phase 1 first, then Phase 2A**

Here's why:
1. Phase 1 is nearly complete (just model integration)
2. We can verify the model works on reference data
3. Then add variant augmentation as a separate, testable feature
4. Easier to debug if something breaks

**Timeline**:
- **Today**: Complete Phase 1 model integration (2-3 hours)
- **Tomorrow**: Test Phase 1 on reference data
- **Next**: Add Phase 2A variant augmentation (1-2 hours)
- **Then**: Train full model with variants

---

## Code Preview: Phase 2A Addition

This is what we'll add AFTER Phase 1:

```python
class BetaDogmaDataset(Dataset):
    # ... existing Phase 1 code ...
    
    def _apply_variant_augmentation(self, seq, labels, variants):
        """Apply variant augmentation with 33/33/33 strategy."""
        
        if not variants or random.random() < 0.33:
            # 33%: No augmentation
            return seq, labels, None
        
        # Select variant type
        benign_vars = [v for v in variants if v.get('is_benign')]
        path_vars = [v for v in variants if v.get('is_pathogenic') or v.get('has_splice_effect')]
        
        if random.random() < 0.5 and benign_vars:
            # 33%: Benign variant
            variant = random.choice(benign_vars)
            seq_alt = self._apply_variant(seq, variant)
            labels_alt = labels  # UNCHANGED
            
        elif path_vars:
            # 33%: Pathogenic variant
            variant = random.choice(path_vars)
            seq_alt = self._apply_variant(seq, variant)
            labels_alt = self._recompute_labels(labels, variant)  # CHANGED
            
        else:
            return seq, labels, None
        
        return seq_alt, labels_alt, variant
    
    def _recompute_labels(self, labels, variant):
        """Recompute labels for pathogenic variant."""
        labels_alt = labels.copy()
        
        # Modify splice labels if SpliceVar score is high
        if variant.get('has_splice_effect') and variant.get('splice_effect_score', 0) > 0.5:
            pos = variant['pos']
            if 0 <= pos < len(labels['donor']):
                # Simple: flip the splice site label
                labels_alt['donor'][pos] = 1 - labels['donor'][pos]
        
        # TODO Phase 2B: Recompute protein
        # TODO Phase 2C: Recompute NMD
        
        return labels_alt
```

---

## Summary

✅ **Question 1**: YES, update labels for pathogenic, NO for benign
✅ **Question 2**: Use SpliceVar scores as ground truth and label modifier
✅ **Question 3**: YES, 33/33/33 augmentation strategy
✅ **Question 4**: NO, benign variants keep original labels

**Next**: Continue with Phase 1 model integration, then add Phase 2A.

Ready to proceed?
