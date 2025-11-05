# Variant Augmentation Design - Critical Questions & Answers

## Current State Analysis

### What We Store in Parquet Files

```python
example = {
    'seq': seq,              # REFERENCE sequence (unmodified)
    'isoforms': [            # Reference isoforms
        {
            'protein_seq': 'MRAL...',  # Reference protein
            'has_nmd': False,          # Reference NMD status
            # ... other reference features
        }
    ],
    'variants': [            # Variant METADATA (not applied)
        {
            'pos': 12345,
            'ref': 'A',
            'alt': 'G',
            'source': 'clinvar',
            'is_pathogenic': True,
            'has_splice_effect': True,
            'splice_effect_score': 0.85,  # SpliceVar prediction
            # ...
        }
    ]
}
```

**Key Insight**: We store REFERENCE sequence and REFERENCE labels, plus variant metadata for augmentation.

---

## Question 1: When we add variation to DNA, do we update transcript/protein?

### Current Answer: NO ❌

**Problem**: This is a critical gap! Here's what happens now:

```python
# Current (WRONG):
seq = "ATCG..."              # Apply variant → "ATGG..."
protein_label = "MRAL..."    # Still reference protein ❌
splice_labels = [0,1,0,...]  # Still reference splice sites ❌
```

### What SHOULD Happen: YES ✅

We need **differential prediction**:

```python
# Correct approach:
if variant_applied:
    # 1. Apply variant to sequence
    seq_alt = apply_variant(seq_ref, variant)
    
    # 2. Re-compute affected labels
    protein_alt = translate_with_variant(seq_alt, variant)
    splice_alt = predict_splice_sites(seq_alt, variant)
    
    # 3. Create differential labels
    labels = {
        'protein_ref': protein_ref,      # Reference
        'protein_alt': protein_alt,      # With variant
        'protein_changed': protein_ref != protein_alt,
        
        'splice_ref': splice_ref,
        'splice_alt': splice_alt,
        'splice_changed': splice_ref != splice_alt,
    }
```

### Implementation Strategy

**Option A: Pre-compute Variant Effects (Recommended)**
- During data generation, compute variant effects
- Store both reference AND variant-altered labels
- Training: randomly choose ref or alt

**Option B: On-the-fly Computation (Complex)**
- Apply variant during training
- Re-compute protein/splice on-the-fly
- Slower but more flexible

**Recommendation**: Option A for Phase 2

---

## Question 2: How are we using SpliceVar data?

### Current State: Metadata Only ❌

```python
variant = {
    'has_splice_effect': True,
    'splice_effect_score': 0.85,  # SpliceVar's prediction
}
```

**Problem**: We're storing SpliceVar's predictions but not using them for training!

### What We SHOULD Do: Differential Splice Prediction ✅

SpliceVar tells us "this variant changes splicing with score 0.85". We should:

1. **Use as ground truth for variant effect prediction**:
   ```python
   # Predict: Does this variant affect splicing?
   splice_effect_pred = model.predict_splice_effect(seq, variant_pos)
   loss = BCE(splice_effect_pred, variant['splice_effect_score'])
   ```

2. **Use to modify splice site labels**:
   ```python
   if variant['has_splice_effect']:
       # Variant creates/destroys splice site
       if variant['splice_effect_score'] > 0.5:
           # Strong effect: modify splice labels
           splice_labels_alt = modify_splice_sites(
               splice_labels_ref,
               variant_pos,
               effect_score
           )
   ```

3. **Train differential prediction**:
   ```python
   # Model learns: ref → alt changes
   outputs = model(seq_with_variant)
   
   # Compare to reference
   splice_change = outputs['splice'] - reference_splice
   
   # Loss: did we predict the change correctly?
   loss = MSE(splice_change, expected_change_from_splicevar)
   ```

### SpliceVar Integration Plan

```python
class VariantAugmentation:
    def apply_splice_variant(self, seq, labels, variant):
        """Apply variant and update splice labels based on SpliceVar."""
        
        # 1. Apply variant to sequence
        seq_alt = self._apply_variant(seq, variant)
        
        # 2. Get SpliceVar prediction
        effect_score = variant['splice_effect_score']  # 0-1
        
        # 3. Modify splice labels
        if effect_score > 0.5:  # Strong effect
            # Variant likely creates/destroys splice site
            labels_alt = self._modify_splice_labels(
                labels['donor'], 
                labels['acceptor'],
                variant['pos'],
                effect_score
            )
        else:
            # Weak effect: keep reference labels
            labels_alt = labels.copy()
        
        return seq_alt, labels_alt, effect_score
```

---

## Question 3: Benign Variation - Always Adding?

### Current State: Metadata Only ❌

We store 1000 Genomes variants but don't apply them during training.

### What We SHOULD Do: Augmentation Strategy ✅

**Purpose of Benign Variants**:
1. **Teach model what's normal**: Most variants don't affect function
2. **Prevent overfitting**: Model shouldn't predict every variant is pathogenic
3. **Calibration**: Balance pathogenic/benign examples

**Recommended Augmentation Strategy**:

```python
class VariantAugmentation:
    def __getitem__(self, idx):
        # Load reference example
        seq_ref, labels_ref = self._load_reference(idx)
        
        # Augmentation decision tree:
        if self.mode == 'train':
            p = random.random()
            
            if p < 0.33:
                # 33%: No variant (pure reference)
                return seq_ref, labels_ref, variant_type='reference'
            
            elif p < 0.66:
                # 33%: Benign variant (1000G)
                variant = random.choice(benign_variants)
                seq_alt = apply_variant(seq_ref, variant)
                
                # Labels should NOT change (benign)
                labels_alt = labels_ref.copy()
                labels_alt['is_pathogenic'] = 0.0
                
                return seq_alt, labels_alt, variant_type='benign'
            
            else:
                # 33%: Pathogenic/splice variant
                variant = random.choice(pathogenic_or_splice_variants)
                seq_alt = apply_variant(seq_ref, variant)
                
                # Labels SHOULD change
                labels_alt = recompute_labels(seq_alt, variant)
                labels_alt['is_pathogenic'] = 1.0
                
                return seq_alt, labels_alt, variant_type='pathogenic'
```

### Do We Modify Predictions for Benign Variants?

**Answer**: NO for benign, YES for pathogenic ✅

```python
if variant['is_benign']:
    # Benign: sequence changes but labels DON'T
    # This teaches model: "this change is OK"
    protein_alt = protein_ref  # Same
    splice_alt = splice_ref    # Same
    nmd_alt = nmd_ref          # Same

elif variant['is_pathogenic']:
    # Pathogenic: sequence changes AND labels change
    # This teaches model: "this change breaks things"
    protein_alt = recompute_protein(seq_alt)  # Different!
    splice_alt = recompute_splice(seq_alt)    # Different!
    nmd_alt = recompute_nmd(protein_alt)      # Different!
```

---

## Comprehensive Variant Augmentation Strategy

### Phase 2A: Basic Variant Augmentation

```python
def __getitem__(self, idx):
    # 1. Load reference
    seq_ref, labels_ref, variants = self._load_parquet(idx)
    
    # 2. Decide augmentation
    if random.random() < self.augment_prob:
        variant = self._select_variant(variants)
        
        # 3. Apply variant to sequence
        seq = self._apply_variant(seq_ref, variant)
        
        # 4. Modify labels based on variant type
        if variant['is_benign']:
            labels = labels_ref  # No change
            labels['variant_effect'] = 0.0
        else:
            labels = self._recompute_labels(seq, variant)
            labels['variant_effect'] = variant.get('splice_effect_score', 1.0)
    else:
        seq = seq_ref
        labels = labels_ref
        labels['variant_effect'] = 0.0
    
    return seq, labels
```

### Phase 2B: Differential Prediction (Advanced)

```python
def __getitem__(self, idx):
    # Always return BOTH reference and variant
    seq_ref, labels_ref, variants = self._load_parquet(idx)
    
    if variants and random.random() < 0.5:
        variant = random.choice(variants)
        seq_alt = self._apply_variant(seq_ref, variant)
        labels_alt = self._recompute_labels(seq_alt, variant)
        
        return {
            'seq_ref': seq_ref,
            'seq_alt': seq_alt,
            'labels_ref': labels_ref,
            'labels_alt': labels_alt,
            'variant': variant,
            'mode': 'differential'
        }
    else:
        return {
            'seq_ref': seq_ref,
            'seq_alt': seq_ref,  # Same
            'labels_ref': labels_ref,
            'labels_alt': labels_ref,  # Same
            'variant': None,
            'mode': 'reference'
        }
```

---

## Critical Implementation Decisions

### Decision 1: When to Recompute Labels?

| Variant Type | Sequence | Splice Labels | Protein Labels | NMD | Expression |
|--------------|----------|---------------|----------------|-----|------------|
| **None (Reference)** | Ref | Ref | Ref | Ref | Ref |
| **Benign (1000G)** | Alt | Ref ✓ | Ref ✓ | Ref | Ref |
| **Pathogenic (ClinVar)** | Alt | Recompute | Recompute | Recompute | Ref* |
| **Splice (SpliceVar)** | Alt | Recompute | Recompute | Recompute | Ref* |

*Expression might change but we don't have variant-specific expression data

### Decision 2: How to Recompute Labels?

**Option A: Rule-based (Simple)**
```python
def recompute_splice_labels(seq_alt, variant):
    if variant['has_splice_effect'] and variant['splice_effect_score'] > 0.5:
        # Strong effect: flip splice site at variant position
        labels_alt = labels_ref.copy()
        labels_alt[variant['pos']] = 1 - labels_ref[variant['pos']]
    return labels_alt
```

**Option B: Re-run prediction (Complex)**
```python
def recompute_splice_labels(seq_alt, variant):
    # Use SpliceAI or similar to predict new splice sites
    splice_pred = SpliceAI.predict(seq_alt)
    return splice_pred
```

**Recommendation**: Option A for Phase 2, Option B for Phase 3

### Decision 3: Training Objective

**Current (Phase 1)**: Predict reference labels from reference sequence

**Phase 2**: Predict variant effects
```python
# Model sees variant-altered sequence
# Predicts: what changed?

loss = (
    # Standard prediction on altered sequence
    BCE(model(seq_alt)['splice'], labels_alt['splice']) +
    
    # Variant effect prediction
    BCE(model.variant_effect(seq_alt, var_pos), variant['splice_effect_score'])
)
```

---

## Recommended Implementation Plan

### Phase 2A: Basic Variant Augmentation (Week 1)

1. ✅ Implement variant application to sequence
2. ✅ Add variant type labels (benign/pathogenic/splice)
3. ✅ Simple rule-based label modification
4. ✅ 33/33/33 augmentation strategy

### Phase 2B: SpliceVar Integration (Week 2)

1. ✅ Use `splice_effect_score` as training target
2. ✅ Add variant effect prediction head
3. ✅ Modify splice labels based on SpliceVar scores
4. ✅ Train differential prediction

### Phase 2C: Protein Effect Prediction (Week 3)

1. ✅ Recompute protein sequences for variants
2. ✅ Predict protein changes
3. ✅ NMD prediction for variant proteins
4. ✅ Frameshift detection

---

## Summary & Answers

### Q1: Update transcript/protein when adding variation?
**A**: YES - We must recompute for pathogenic variants, NO for benign variants

### Q2: How to use SpliceVar data?
**A**: Use `splice_effect_score` as:
1. Ground truth for variant effect prediction
2. Guide for modifying splice labels
3. Training signal for differential prediction

### Q3: Adding benign variation always?
**A**: Use 33/33/33 strategy:
- 33% reference (no variant)
- 33% benign (sequence changes, labels don't)
- 33% pathogenic (sequence AND labels change)

### Q4: Modify predictions for benign variants?
**A**: NO - This teaches model that not all variants are harmful

---

## Next Steps

Before continuing with model integration, we should decide:

1. **Phase 2A now or later?** 
   - Now: Add basic variant augmentation
   - Later: Focus on Phase 1 model first

2. **Recomputation strategy?**
   - Simple: Rule-based label modification
   - Complex: Re-run protein translation

3. **Training objective?**
   - Standard: Predict labels from sequence
   - Differential: Predict ref→alt changes

**Recommendation**: Complete Phase 1 model integration first, then add Phase 2A variant augmentation.
