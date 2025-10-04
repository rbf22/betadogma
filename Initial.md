Here’s a **complete Markdown documentation suite** for your `betadogma` repository, written so agents, collaborators, or LLM-based code assistants can start working productively right away.  
It includes:

- `README.md` (main entry point — dense and conceptual)  
- `CONTRIBUTING.md` (how to extend and maintain modules)  
- `DATASETS.md` (data sources, licensing, and preprocessing notes)  
- `MODEL_CARD.md` (architecture summary, tasks, limitations, and ethics)  
- `TASKS.md` (for multi-agent orchestration / submodule goals)  

Each file is ready to drop into the repo.

---

## 🧭 1. `README.md`

```markdown
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

Detailed documentation: [`DATASETS.md`](./DATASETS.md)

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

## 📜 License

Open under Apache-2.0 (research use encouraged).

---

## 📖 Further Docs

- [`DATASETS.md`](./DATASETS.md)
- [`MODEL_CARD.md`](./MODEL_CARD.md)
- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`TASKS.md`](./TASKS.md)
```

---

## 🧩 2. `CONTRIBUTING.md`

```markdown
# Contributing to BetaDogma

We welcome contributions from computational biologists, ML researchers, and data engineers.

## 🧱 Repository Modules

| Path | Responsibility |
|------|----------------|
| `core/` | model architecture and training utilities |
| `nmd/` | transcript decay prediction module |
| `variant/` | variant encoding and simulation |
| `data/` | dataset loading, normalization, caching |
| `experiments/` | training configs, hyperparameters |
| `notebooks/` | analysis notebooks and evaluation scripts |

## 🧩 Contribution Workflow

1. **Fork and branch**  
   ```bash
   git clone https://github.com/yourname/betadogma
   git checkout -b feature/my-feature
   ```
2. **Add tests** in `tests/` where applicable.  
3. **Document** your function using NumPy-style docstrings.  
4. **Run pre-commit checks** (`ruff`, `pytest`).  
5. **Submit a PR** with clear summary and references.

## 🧠 Coding Standards

- Python ≥ 3.10  
- PEP8 / Ruff linting  
- Type hints required for all public functions  
- Use Hydra / YAML configs for experiments  
- Prefer pure PyTorch (no custom CUDA ops unless necessary)

## 🧪 Testing

- Unit tests for each module (`pytest -v`)  
- Integration tests on sample datasets (`tests/sample_gene.py`)  

## 🧬 Citing BetaDogma

If you use this codebase, please cite the BetaDogma preprint (forthcoming).
```

---

## 📚 3. `DATASETS.md`

```markdown
# BetaDogma Datasets

This document outlines data sources, licenses, and preprocessing used to train and evaluate BetaDogma.

---

## 📘 Core Reference

| Dataset | Source | License | Usage |
|----------|--------|----------|-------|
| **GENCODE v44** | [https://www.gencodegenes.org/](https://www.gencodegenes.org/) | Open | Exon/intron structure, CDS boundaries |
| **RefSeq** | NCBI | Public domain | Validation and redundancy checks |
| **GTEx v8** | Broad Institute | dbGaP | RNA-seq, eQTL/sQTL for ψ supervision |
| **ENCODE** | ENCODE Project | CC-BY 4.0 | Chromatin and expression data |
| **FANTOM5 CAGE** | RIKEN | CC-BY | TSS label training |
| **PolyA-DB / APASdb** | Various | CC-BY | PolyA site training |
| **Ribo-seq (GWIPS)** | Open | Research | ORF and translation signal supervision |
| **UPF1/SMG6 KD RNA-seq** | GEO Series (various) | CC-BY | NMD labels |
| **BRIC-seq / 4sU-seq** | GEO | CC-BY | mRNA half-life regression |
| **gnomAD + eQTL Catalog** | Broad / EBI | Open | Variant effect modeling |

---

## ⚙️ Preprocessing Pipelines

Each dataset is standardized into HDF5 or Parquet format with:

- `sequence` (DNA string)
- `features` (binary tracks for donor, acceptor, TSS, polyA)
- `isoforms` (list of exons)
- `psi` (relative usage)
- `nmd_label` (0/1)
- `variants` (VCF entries with effect sizes)

Scripts live in `data/prepare_*.py`.

---

## 🧬 Synthetic Data

To augment limited NMD data:
- Randomly introduce **PTCs** at variable distances from exon junctions.
- Generate **splice site deletions/insertions** to simulate exon skipping.
- Label synthetic samples using the 55-nt rule as weak supervision.
```

---

## 🧠 4. `MODEL_CARD.md`

```markdown
# BetaDogma Model Card

**Model Type:** Multi-task transformer over genomic sequence  
**Backbone:** GENERator (98 kb context)  
**Language:** DNA (A/C/G/T + variant tokens)

---

## 🎯 Tasks

| Task | Description | Output |
|------|--------------|--------|
| **Splicing** | Donor/acceptor detection | Binary logits per base |
| **TSS / PolyA** | Transcription initiation and termination | Binary logits per base |
| **Isoform assembly** | Predict exon chains | Exon graph and ψ distribution |
| **NMD prediction** | Transcript stability classification | P(NMD) ∈ [0, 1] |
| **Variant effect** | Predict Δψ, ΔNMD for allele changes | Regression / delta metrics |

---

## 📈 Performance Goals

| Metric | Target |
|---------|--------|
| Splice F1 | ≥ 0.95 |
| ψ correlation | ≥ 0.85 |
| NMD AUROC | ≥ 0.90 |
| Variant Δψ correlation | ≥ 0.70 |

---

## ⚠️ Limitations

- Limited to annotated human loci (GRCh38)  
- NMD labels derived from cell-line experiments (context-dependent)  
- Variant interpretation relies on eQTL datasets (biased toward common variants)

---

## 🧩 Ethical Considerations

- No personal genetic data; all sources are de-identified.  
- Predictions should not be used for clinical decision-making.  
- Encourage open, reproducible science; all training code and data processing scripts are transparent.
```

---

## 🧠 5. `TASKS.md` (for agents / automation)

```markdown
# BetaDogma: Task Index for Agents

Use this file as a coordination guide for automated or human contributors.

---

## 🧱 Core Development

| Task ID | Description | Dependencies |
|----------|--------------|---------------|
| `T1` | Implement GENERator backbone wrapper (`core/encoder.py`) | Base tokenizer |
| `T2` | Build splice / TSS / polyA heads | T1 |
| `T3` | Implement isoform decoder (pointer network) | T2 |
| `T4` | Create ψ head and abundance loss | T3 |
| `T5` | Add NMD rule + learned hybrid module | T3 |
| `T6` | Add variant encoder & delta predictor | T1 |
| `T7` | Data preprocessing pipeline (`data/prepare_*.py`) | Raw datasets |

---

## 🧪 Experimentation

| Task ID | Description | Notes |
|----------|--------------|-------|
| `E1` | Fine-tune splice heads on GENCODE + GTEx | baseline |
| `E2` | Train ψ head with long-read data | check correlation |
| `E3` | Train NMD head with UPF1 KD data | compare to rule baseline |
| `E4` | Variant fine-tuning using eQTL Catalog | compute Δψ correlation |
| `E5` | Joint multi-task training | final model |

---

## 📊 Evaluation & Reporting

| Task | Output |
|------|--------|
| `R1` | Benchmark report (`reports/eval_metrics.json`) |
| `R2` | Visualization notebooks (`notebooks/analysis.ipynb`) |
| `R3` | Documentation update (model card refresh) |

---

Agents or contributors should:
- Follow naming conventions: `feature/<taskID>_<shortname>`  
- Write modular code, minimal coupling between heads and backbone  
- Maintain reproducibility (`seed=42`, config in YAML)
```

---

Would you like me to generate the **initial file tree** (with placeholder `.py` files and docstrings for each module) so you can commit a ready-to-clone `betadogma` scaffold next?
