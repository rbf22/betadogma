# BetaDogma

> _“Revising the Central Dogma through data.”_  
> From DNA to RNA to (in)stability — a unified model for transcript structure, abundance, and NMD fate.

---

## 🔬 Overview

**BetaDogma** is a research framework that learns the **probabilistic central dogma** of molecular biology.  
It takes raw **genomic sequence** (± variants) and predicts:

- **Dominant mRNA isoform** — inferred from splice, TSS, and polyA patterns  
- **Relative isoform abundance (ψ)** — learned from RNA-seq junction data  
- **Nonsense-mediated decay (NMD) probability** — combining rule-based and learned features  
- **Variant effects** — Δψ and ΔNMD due to sequence edits

The project fine-tunes a long-context genomic language model (e.g. **GENERator**) with specialized biological heads and interpretable outputs.

---

## 🧱 Architecture

```
Genomic sequence (+ optional variants)
       │
       ▼
   BetaDogma backbone (GENERator)
       │
Base embeddings (nucleotide resolution)
       │
├── splice_head → donor/acceptor logits
├── tss_head → transcription start sites
├── polya_head → cleavage / 3′ ends
├── orf_head → CDS start/stop/frame
└── variant_channel → REF/ALT effects
       │
Isoform decoder + ψ head
       │
Dominant transcript selector
       │
NMD head (rule + learned features)
       ▼
Final outputs: mRNA structure + P(NMD)
```

---

## 🧩 Core Modules

| Module | Function |
|---------|-----------|
| `core/` | Backbone + per-base heads (splice, TSS, polyA, ORF) |
| `decoder/` | Isoform graph assembly and ψ scoring |
| `nmd/` | Rule-augmented classifier for transcript decay |
| `variant/` | Variant encoding, synthetic mutagenesis, Δ computations |
| `data/` | Data ingestion and preprocessing pipelines |
| `experiments/` | Training configurations, checkpoints |
| `notebooks/` | Analysis, visualization, evaluation tools |

---

## 📚 Data Layers

Detailed documentation: [`docs/DATASETS.md`](./docs/DATASETS.md)

| Purpose | Dataset |
|----------|----------|
| Gene structure | GENCODE, RefSeq |
| Isoform abundance | GTEx, ENCODE |
| Long-read truth | PacBio Iso-Seq, Nanopore |
| TSS / polyA sites | FANTOM5, PolyA-DB |
| Translation frame | Ribo-seq |
| NMD labels | UPF1/SMG6 knockdown RNA-seq, 4sU-seq |
| Variant effects | GTEx eQTL/sQTL, gnomAD, MPRA reporters |

---

## 🧠 Training Phases

1. **Structural fine-tuning** – teach splicing, TSS, and polyA recognition.  
2. **Isoform decoding** – learn exon chains and ψ distribution.  
3. **NMD prediction** – hybrid rule + learned classifier.  
4. **Variant adaptation** – train for Δψ and ΔNMD sensitivity.  
5. **Joint optimization** – multi-task fine-tuning end-to-end.

---

## ⚙️ Quickstart (conceptual)

> **Note:** This example is a conceptual guide. The API is under active development and this code is not yet runnable.

```python
from betadogma import BetaDogmaModel, preprocess_sequence, preprocess_variant

model = BetaDogmaModel.from_pretrained("betadogma/generator-base")

seq = preprocess_sequence(chrom="chr17", start=43044294, end=43099294)
variant = preprocess_variant("17:43051000A>T")

out_ref = model.predict(seq)
out_alt = model.predict(seq, variant=variant)

print(out_ref.dominant_isoform)
print("ΔNMD =", out_alt.P_NMD - out_ref.P_NMD)
```

---

## 🧪 Evaluation

| Metric | Description |
|---------|-------------|
| **Splice F1 / junction accuracy** | donor/acceptor prediction |
| **Isoform correctness** | exon chain match |
| **ψ correlation** | usage prediction vs. RNA-seq |
| **NMD AUROC / AUPRC** | decay classification |
| **Δψ / ΔNMD correlation** | variant effect prediction |

---

## 🧬 Philosophy

> The “central dogma” was never static — transcription and translation are dynamic systems.  
> **BetaDogma** re-learns these principles directly from data, modeling uncertainty, regulation, and decay as emergent behaviors.

---

## 📖 Docs

- [`docs/DATASETS.md`](./docs/DATASETS.md)
- [`docs/MODEL_CARD.md`](./docs/MODEL_CARD.md)
- [`docs/TASKS.md`](./docs/TASKS.md)
- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`LICENSE`](./LICENSE)

---

## 📜 License

MIT
