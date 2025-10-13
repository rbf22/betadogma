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

## 🏗️  Data Pipeline

BetaDogma provides a unified data pipeline to download, process, and generate datasets. The system supports both minimal (chromosome 22 only) and full dataset configurations through a single, streamlined interface.

### Key Features

- **Unified Interface**: Single entry point for all data operations
- **Modular Design**: Clear separation of fetching, processing, and generation steps
- **Resumable**: Skip already processed files with checksum verification
- **Configurable**: Control all aspects through YAML configuration files
- **Progress Tracking**: Real-time progress bars for downloads and processing

### Available Configurations

1. **Minimal Dataset** (`minimal_config.yaml`):
   - Chromosome 22 reference genome
   - GENCODE annotations for chr22
   - Small dataset size for testing and development
   - Faster processing time (minutes)
   - Minimal disk space requirements (~200MB)

2. **Full Dataset** (`full_config.yaml`):
   - Complete GRCh38 reference genome
   - Full GENCODE annotations
   - GTEx expression and junction data
   - Example variant data
   - Comprehensive for production use
   - Larger disk space requirements (~50GB)

### Installation

```bash
# Install core dependencies
pip install pyyaml tqdm requests pyfaidx numpy pandas

# For additional data processing (optional)
pip install pybigwig pybedtools
```

### Basic Usage

```bash
# Show help
python data/data_builder.py --help

# Build minimal dataset (recommended for testing)
python data/data_builder.py --config data/minimal_config.yaml

# Build full dataset
python data/data_builder.py --config data/full_config.yaml
```

### Advanced Usage

```bash
# Force rebuild all files (ignore cached/downloaded files)
python data/data_builder.py --config data/config.yaml --force

# Run specific steps only
python data/data_builder.py --config data/config.yaml --steps fetch process

# Custom output directory
python data/data_builder.py --config data/config.yaml --output-dir /path/to/output
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file (default: minimal_config.yaml) |
| `--force` | Force re-download and re-build of all files |
| `--steps` | Run specific steps: fetch, process, generate (default: all) |
| `--output-dir` | Custom output directory (overrides config) |
| `--skip-existing` | Skip existing files (default: true) |

### Pipeline Steps

The data pipeline is designed with a clear separation of concerns across three main steps:

1. **Fetch**
   - Downloads required data files from external sources
   - Verifies file integrity using checksums
   - Handles retries and resume for large downloads
   - Sources include:
     - GENCODE for gene annotations
     - UCSC for reference genomes
     - GTEx for expression and junction data
     - Custom variant data sources

2. **Process**
   - Decompresses downloaded files (e.g., .gz, .zip)
   - Filters and formats data as needed
   - Validates data consistency
   - Creates necessary indices for large files
   - Handles chromosome-specific filtering for minimal datasets

3. **Generate**
   - Creates training-ready datasets
   - Generates sequence and label files
   - Builds junction databases
   - Creates metadata and documentation
   - Validates final dataset integrity

Each step is designed to be idempotent and can be run independently using the `--steps` flag. The pipeline maintains a clean separation between raw, processed, and generated data in separate directories.

### Output Structure

```
data/
├── raw/                   # Raw downloaded files
│   ├── gencode/
│   │   ├── gencode.v44.annotation.gtf
│   │   └── GRCh38.primary_assembly.genome.fa
│   ├── gtex/
│   │   ├── GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct
│   │   ├── GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
│   │   └── GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct
│   └── variants/
│       └── example_variants.vcf
│
├── processed/             # Processed data files (full dataset)
│   ├── full_sequences.fa
│   ├── full_labels.npy
│   ├── full_junctions.gct
│   └── full_metadata.json
│
└── data_mini/             # Minimal dataset files
    ├── mini_sequences.fa
    ├── mini_labels.npy
    ├── mini_junctions.gct
    └── mini_metadata.json
```

### Configuration

The data pipeline is configured using YAML files (`minimal_config.yaml` and `full_config.yaml`). Key configuration options include:

- `output`: Output directory structure
- `gencode`: GENCODE data source configuration
- `gtex`: GTEx data source configuration
- `variants`: Variant data configuration
- `dataset`: Dataset generation parameters
  - `training`: Training data parameters (sequence length, number of samples, etc.)
  - `output_files`: Output file naming

Example configuration snippet:

```yaml
genome_build: GRCh38
release: 44
chromosome: chr22  # or 'all' for full dataset

output:
  raw: data/raw
  processed: data/processed
  mini: data/data_mini

dataset:
  training:
    num_sequences: 1000
    sequence_length: 1000
    num_junctions: 50
    num_samples: 5
  
  output_files:
    sequences: "mini_sequences.fa"
    labels: "mini_labels.npy"
    junctions: "mini_junctions.gct"
    metadata: "mini_metadata.json"
```

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

## 🛠️ Data Processing

### 1. Fetching Raw Data (`data/fetch_data.py`)

This script downloads and verifies the raw data needed for training.

#### Usage:
```bash
# Download all data (full dataset)
poetry run python data/fetch_data.py

# Check download status without downloading
poetry run python data/fetch_data.py --check

# Force re-download of specific files
poetry run python data/fetch_data.py --force gtex_junctions
```

#### Features:
- Downloads data from various sources (GTEx, GENCODE, etc.)
- Verifies file integrity with checksums
- Skips already downloaded files by default
- Supports resuming interrupted downloads

### 2. Preparing Training Data (`train/make_training_data.py`)

Processes raw data into training-ready format with PSI calculations and gene annotations.

#### Basic Usage:
```bash
# Full processing (all steps)
poetry run python train/make_training_data.py --config train/configs/data.base.yaml
```

#### Smoke Test Mode (for quick validation):
```bash
# Process only a small subset of data
poetry run python train/make_training_data.py --config train/configs/data.base.yaml --smoke
```

#### Checkpointing and Resuming:
```bash
# Resume from a specific step if interrupted
poetry run python train/make_training_data.py --config train/configs/data.base.yaml --from-step gtex

# Use a custom checkpoint directory
poetry run python train/make_training_data.py --config train/configs/data.base.yaml --checkpoint-dir my_checkpoints
```

#### Available Steps:
- `gencode`: Process GENCODE annotations
- `gtex`: Process GTEx junction data
- `variants`: Process variant data
- `data`: Final data aggregation

## 📜 License

MIT
