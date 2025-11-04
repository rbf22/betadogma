# Variant Effects: Sequence vs Function - Critical Clarification

## The Key Distinction You Identified

You're absolutely right! There are **TWO different types of changes**:

1. **Sequence changes** (deterministic) - What IS the sequence?
2. **Functional changes** (learned) - Does it work differently?

## Example: Benign Missense Variant

```
Reference:  DNA: ...CTG...  →  Protein: ...L...  →  Function: ✓ Works
Variant:    DNA: ...ATC...  →  Protein: ...I...  →  Function: ✓ Still works!
                    ↑                      ↑
              CHANGED              CHANGED but BENIGN
```

### What Should Happen in Our Model?

```python
# Reference
seq_ref = "...CTG..."
protein_ref = "...L..."
is_pathogenic_ref = 0.0

# Benign missense variant (L→I)
seq_alt = "...ATC..."
protein_alt = "...I..."        # SEQUENCE CHANGED (deterministic)
is_pathogenic_alt = 0.0        # FUNCTION UNCHANGED (learned!)
```

---

## Two-Level Prediction Framework

### Level 1: Sequence Prediction (Deterministic)

**What we MUST predict correctly:**
- DNA sequence → Protein sequence (translation rules)
- Splice sites → Transcript structure
- CDS boundaries → Protein boundaries

```python
# These are DETERMINISTIC transformations
DNA "ATG" → Amino Acid "M"  (always!)
DNA "CTG" → Amino Acid "L"  (always!)
DNA "ATC" → Amino Acid "I"  (always!)

# Model MUST learn these rules
if variant changes DNA:
    protein_sequence WILL change (if in CDS)
    → This is deterministic, not optional
```

### Level 2: Functional Prediction (Learned)

**What the model LEARNS:**
- Does this sequence change affect function?
- Is this variant pathogenic or benign?
- Does splicing change?

```python
# These are LEARNED associations
Protein "...L..." → Pathogenic? (depends on context!)
Protein "...I..." → Pathogenic? (depends on context!)

# Model learns:
# - L→I in protein core = often benign
# - L→P in protein core = often pathogenic
# - Same mutation in different positions = different effects
```

---

## Corrected Training Strategy

### What We Store in Labels

```python
labels = {
    # LEVEL 1: Deterministic sequence predictions
    'protein_sequence': "MRAL...",      # What IS the protein?
    'splice_sites': [0,1,0,...],        # Where ARE the splice sites?
    'cds_boundaries': (100, 500),       # Where IS the CDS?
    
    # LEVEL 2: Learned functional predictions
    'is_pathogenic': 0.0,               # Does it WORK?
    'affects_splicing': 0.0,            # Does splicing CHANGE?
    'has_nmd': False,                   # Will it trigger NMD?
}
```

### Training Examples

#### Example 1: Benign Missense (L→I)

```python
# Reference
example_ref = {
    'seq': "...CTG...",
    'protein': "...L...",
    'is_pathogenic': 0.0,
    'function_score': 1.0
}

# Variant (benign missense)
example_alt = {
    'seq': "...ATC...",
    'protein': "...I...",           # CHANGED (deterministic)
    'is_pathogenic': 0.0,           # SAME (learned: still benign)
    'function_score': 0.95          # Slightly reduced but OK
}

# Model learns:
# - Input: "...ATC..." → Output protein: "...I..." ✓
# - Input: "...I..." → Output pathogenic: 0.0 ✓
```

#### Example 2: Pathogenic Missense (L→P)

```python
# Variant (pathogenic missense)
example_alt = {
    'seq': "...CCG...",
    'protein': "...P...",           # CHANGED (deterministic)
    'is_pathogenic': 1.0,           # CHANGED (learned: now pathogenic)
    'function_score': 0.1           # Severely reduced
}

# Model learns:
# - Input: "...CCG..." → Output protein: "...P..." ✓
# - Input: "...P..." → Output pathogenic: 1.0 ✓
```

#### Example 3: Synonymous (L→L)

```python
# Variant (synonymous)
example_alt = {
    'seq': "...CTA...",             # CHANGED
    'protein': "...L...",           # SAME (deterministic: same codon)
    'is_pathogenic': 0.0,           # SAME (learned: benign)
    'function_score': 1.0
}

# Model learns:
# - Input: "...CTA..." → Output protein: "...L..." ✓
# - Input: "...L..." → Output pathogenic: 0.0 ✓
```

---

## What is Deterministic vs Learned?

### Deterministic (Must Recompute)

| Change | Input | Output | Rule |
|--------|-------|--------|------|
| DNA→Protein | `CTG` | `L` | Genetic code |
| DNA→Protein | `ATC` | `I` | Genetic code |
| DNA→Protein | `CCG` | `P` | Genetic code |
| Splice site | `GT` at exon boundary | Donor site | Splice motif |
| Frameshift | Insert 1bp in CDS | Shifted protein | Translation frame |

**These MUST be recomputed when sequence changes!**

### Learned (Model Discovers)

| Change | Input | Output | Learned From |
|--------|-------|--------|--------------|
| Pathogenicity | Protein `...I...` | Benign | ClinVar labels |
| Pathogenicity | Protein `...P...` | Pathogenic | ClinVar labels |
| Splice effect | Variant near splice | Score 0.85 | SpliceVar data |
| NMD | Premature stop | Triggers NMD | Transcript structure |
| Expression | Promoter variant | Reduced TPM | GTEx data |

**These are learned associations, not deterministic rules!**

---

## Corrected Implementation Strategy

### Phase 1: Deterministic Predictions (Current)

```python
def __getitem__(self, idx):
    seq, isoforms, variants = load_data(idx)
    
    # ALWAYS recompute deterministic transformations
    protein = translate(seq, cds_start, cds_end)
    splice_sites = find_splice_motifs(seq)
    
    labels = {
        'protein': protein,           # Deterministic
        'splice_sites': splice_sites, # Deterministic
        # Functional labels from data
        'is_pathogenic': isoforms['is_pathogenic'],
        'expression': isoforms['tpm']
    }
    
    return seq, labels
```

### Phase 2A: Variant Augmentation (Corrected)

```python
def __getitem__(self, idx):
    seq_ref, isoforms_ref, variants = load_data(idx)
    
    if random.random() < 0.5 and variants:
        variant = random.choice(variants)
        
        # 1. Apply variant to sequence
        seq_alt = apply_variant(seq_ref, variant)
        
        # 2. ALWAYS recompute deterministic transformations
        protein_alt = translate(seq_alt, cds_start, cds_end)  # MUST recompute
        splice_alt = find_splice_motifs(seq_alt)              # MUST recompute
        
        # 3. Functional labels depend on variant type
        if variant['is_benign']:
            # Sequence changed, but function didn't
            is_pathogenic = 0.0
            function_score = 1.0
            
        elif variant['is_pathogenic']:
            # Sequence changed AND function changed
            is_pathogenic = 1.0
            function_score = 0.0
        
        labels = {
            'protein': protein_alt,        # CHANGED (deterministic)
            'splice_sites': splice_alt,    # CHANGED (deterministic)
            'is_pathogenic': is_pathogenic,# Depends on variant type
            'function_score': function_score
        }
    else:
        # Reference
        labels = {
            'protein': translate(seq_ref),
            'splice_sites': find_splice_motifs(seq_ref),
            'is_pathogenic': 0.0,
            'function_score': 1.0
        }
    
    return seq, labels
```

---

## Your Specific Example: L→I Mutation

```python
# Reference sequence
seq_ref = "...CTG..."  # Leucine codon
protein_ref = "...L..."
is_pathogenic_ref = 0.0

# Benign variant (L→I in protein core)
variant = {
    'pos': 12345,
    'ref': 'CTG',
    'alt': 'ATC',
    'is_benign': True,
    'is_pathogenic': False,
    'clinical_significance': 'benign'
}

# What happens during training:
seq_alt = "...ATC..."              # CHANGED
protein_alt = "...I..."            # CHANGED (deterministic!)
is_pathogenic_alt = 0.0            # SAME (learned: still benign)

# Model sees:
# Input:  DNA "...ATC..."
# Output: Protein "...I..." ✓ (learns genetic code)
#         Pathogenic: 0.0 ✓ (learns L→I is benign)
```

### What the Model Learns

```python
# The model learns TWO things:

# 1. DETERMINISTIC: Genetic code
model.translate("ATC") → "I"  # Always!

# 2. LEARNED: Context-dependent pathogenicity
model.predict_pathogenic("...I... in position 123") → 0.0  # Benign
model.predict_pathogenic("...I... in position 456") → 0.8  # Pathogenic
                                                            # (different context!)
```

---

## Summary: Deterministic vs Learned

### ✅ ALWAYS Recompute (Deterministic)

When sequence changes:
1. **Protein sequence** - Apply genetic code
2. **Splice sites** - Find GT/AG motifs
3. **CDS boundaries** - Track reading frame
4. **Transcript structure** - Follow exon junctions

### 🎓 Model Learns (Functional)

From training data:
1. **Pathogenicity** - Is this change harmful?
2. **Splice effects** - Does splicing change?
3. **NMD triggering** - Will this cause NMD?
4. **Expression changes** - Does expression change?

### 🎯 Your L→I Example

```
Sequence Level (Deterministic):
  CTG → ATC  ⟹  L → I  ✓ MUST recompute

Functional Level (Learned):
  L → I in core  ⟹  Benign  ✓ Model learns this
  L → I in active site  ⟹  Pathogenic  ✓ Model learns this too
```

---

## Implementation Answer

### Question: "Will that be deterministic or learned?"

**Answer: BOTH!**

1. **Sequence transformation is DETERMINISTIC**
   - We MUST recompute protein sequence when DNA changes
   - This follows genetic code rules
   - Not optional!

2. **Functional effect is LEARNED**
   - Model learns whether L→I is benign or pathogenic
   - Depends on context (position, structure, conservation)
   - Learned from ClinVar labels

### Corrected Training Strategy

```python
# For ALL variants (benign or pathogenic):
protein_alt = translate(seq_alt)  # ALWAYS recompute (deterministic)

# But functional labels differ:
if variant['is_benign']:
    labels['is_pathogenic'] = 0.0  # Sequence changed, function didn't
    
elif variant['is_pathogenic']:
    labels['is_pathogenic'] = 1.0  # Sequence AND function changed
```

---

## Key Insight

**Benign ≠ Synonymous**

- **Synonymous**: DNA changes, protein doesn't (CTG→CTA, both L)
- **Benign missense**: DNA changes, protein changes, but function doesn't (CTG→ATC, L→I, still works)
- **Pathogenic missense**: DNA changes, protein changes, function changes (CTG→CCG, L→P, breaks)

All three have different sequence changes, but different functional outcomes!

The model must learn:
- How to translate DNA→Protein (deterministic)
- Which protein changes are benign vs pathogenic (learned from data)

---

## Does This Make Sense?

The key is:
1. ✅ **Always recompute protein sequence** (it's deterministic)
2. ✅ **Let model learn pathogenicity** (it's context-dependent)
3. ✅ **Benign variants change sequence but not function** (that's the teaching signal)

Your L→I example is perfect: the protein sequence DOES change (deterministic), but the model learns it's still benign (learned from ClinVar).

Ready to implement this correctly in Phase 2?
