# Project Overview

### Detected Environment
- Root: `/workspaces/betadogma`
- Date: `2025-10-06 01:42:18 UTC`
- Python: `3.12.1`

## Structure (tree)
```text
betadogma
├── .github
│   └── workflows
│       └── ci.yml
├── docs
│   ├── DATASETS.md
│   ├── MODEL_CARD.md
│   └── TASKS.md
├── notebooks
│   └── README.md
├── src
│   └── betadogma
│       ├── core
│       │   ├── __init__.py
│       │   ├── encoder_nt.py
│       │   ├── heads.py
│       │   └── losses.py
│       ├── decoder
│       │   ├── __init__.py
│       │   ├── evaluate.py
│       │   ├── isoform_decoder.py
│       │   └── types.py
│       ├── experiments
│       │   ├── config
│       │   │   └── default.yaml
│       │   ├── __init__.py
│       │   ├── train.py
│       │   └── train_decoder.py
│       ├── nmd
│       │   ├── __init__.py
│       │   ├── nmd_model.py
│       │   └── nmd_rules.py
│       ├── variant
│       │   ├── __init__.py
│       │   ├── encode.py
│       │   └── simulate.py
│       ├── __init__.py
│       └── model.py
├── tests
│   ├── core
│   │   └── test_heads.py
│   ├── decoder
│   │   ├── test_decoder_logic.py
│   │   ├── test_isoform_decoder.py
│   │   ├── test_isoform_decoder_minus_strand.py
│   │   ├── test_isoform_enumerator.py
│   │   └── test_splice_graph_builder.py
│   └── test_import.py
├── train
│   ├── checkpoints
│   │   ├── betadogma
│   │   └── default
│   │       ├── epoch=18-val_score=0.0000.ckpt
│   │       ├── epoch=19-val_score=0.0000.ckpt
│   │       └── last.ckpt
│   ├── configs
│   │   ├── data.base.yaml
│   │   └── train.base.yaml
│   ├── runs
│   │   ├── betadogma
│   │   │   └── version_0
│   │   │       └── events.out.tfevents.1759711084.codespaces-7eda66.59460.0
│   │   └── default
│   │       └── version_0
│   │           ├── events.out.tfevents.1759707937.codespaces-7eda66.26280.0
│   │           └── hparams.yaml
│   ├── make_training_data.py
│   ├── prepare_model.py
│   ├── train.py
│   └── train_isoform_ranker.py
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── llm_project_brief.py
├── poetry.lock
├── project_brief.md
├── pyproject.toml
└── README.md
```
---

## Python API Map (classes & functions)
**File:** `llm_project_brief.py`
- **Classes**
  - MethodInfo
  - ClassInfo
  - FunctionInfo
  - FileRecord
- **Functions**
  - `human_path(p, root)`
  - `sha256sum(path)`
  - `is_text_file(path, max_probe_bytes)`
  - `walk_tree(root, exclude_dirs)`
  - `render_tree(root, exclude_dirs, exclude_files)`
  - `parse_api(py_path)`
  - `collect_files(root, exclude_dirs, exclude_files)`
  - `preview_file(path, max_lines, max_bytes)`
  - `main(argv)`

**File:** `src/betadogma/core/encoder_nt.py`
- **Classes**
  - NTEncoder
    - `__init__(self, model_id, device)`
    - `forward(self, seqs)`

**File:** `src/betadogma/core/heads.py`
- **Classes**
  - _ConvHead(nn.Module)
    - `__init__(self, d_in, d_hidden, out_ch, dropout, use_conv)`
    - `forward(self, x)`
  - SpliceHead(nn.Module)
    - `__init__(self, d_in, d_hidden, dropout, use_conv)`
    - `forward(self, embeddings)`
  - TSSHead(nn.Module)
    - `__init__(self, d_in, d_hidden, dropout, use_conv)`
    - `forward(self, embeddings)`
  - PolyAHead(nn.Module)
    - `__init__(self, d_in, d_hidden, dropout, use_conv)`
    - `forward(self, embeddings)`
  - ORFHead(nn.Module)
    - `__init__(self, d_in, d_hidden, dropout, use_conv)`
    - `forward(self, embeddings)`

**File:** `src/betadogma/core/losses.py`
- **Functions**
  - `bce_loss()`
  - `kl_loss()`

**File:** `src/betadogma/decoder/evaluate.py`
- **Functions**
  - `get_junctions(isoform)`
  - `exon_chain_match(predicted, ground_truth)`
  - `junction_f1(predicted, ground_truth)`
  - `top_k_recall(predicted_isoforms, ground_truth, k)`

**File:** `src/betadogma/decoder/isoform_decoder.py`
- **Classes**
  - SpliceGraph
    - `__init__(self)`
    - `add_exon(self, exon)`
    - `add_junction(self, from_exon, to_exon, score)`
    - `__repr__(self)`
  - SpliceGraphBuilder
    - `__init__(self, config)`
    - `_get_exons(self, starts, ends, start_scores, end_scores, strand)`
    - `build(self, head_outputs, strand)`
  - IsoformEnumerator
    - `__init__(self, config)`
    - `enumerate(self, graph, max_paths, strand)`
  - IsoformScorer(nn.Module)
    - `__init__(self, config)`
    - `forward(self, isoform, head_outputs, input_ids)`
  - IsoformDecoder(nn.Module)
    - `__init__(self, config)`
    - `forward(self, head_outputs, strand, input_ids)`
    - `decode(self, head_outputs, strand, input_ids)`
- **Functions**
  - `_find_peaks(logits, threshold, top_k)`
  - `_order_exons_by_transcription(exons, strand)`
  - `_log_sigmoid(x)`
  - `_get_peak_log_prob(logits, index, window)`
  - `_get_spliced_cDNA(isoform, input_ids, token_map)`
  - `_score_orf(isoform, head_outputs, scoring_config, input_ids)`
  - `_score_length(isoform, priors)`

**File:** `src/betadogma/decoder/types.py`
- **Classes**
  - Exon
  - Isoform
    - `start(self)`
    - `end(self)`

**File:** `src/betadogma/experiments/train.py`
- **Classes**
  - StructuralDataset(Dataset)
    - `__init__(self, parquet_paths, max_shards)`
    - `__len__(self)`
    - `__getitem__(self, idx)`
- **Functions**
  - `collate(batch)`
  - `train(cfg)`
  - `main()`

**File:** `src/betadogma/experiments/train_decoder.py`
- **Classes**
  - DummyEncoder(nn.Module)
    - `__init__(self, model_name)`
    - `forward(self)`
  - GroundTruthIsoformDataset(Dataset)
    - `__init__(self, num_samples)`
    - `__len__(self)`
    - `__getitem__(self, idx)`
- **Functions**
  - `are_isoforms_equal(iso1, iso2)`
  - `collate_decoder(batch)`
  - `listwise_hinge_loss(positive_score, negative_scores, margin)`
  - `train_decoder(cfg)`
  - `main()`

**File:** `src/betadogma/model.py`
- **Classes**
  - Prediction
    - `psi(self)`
    - `dominant_isoform(self)`
    - `p_nmd(self)`
  - BetaDogmaModel(nn.Module)
    - `__init__(self, d_in, config)`
    - `forward(self, embeddings, input_ids)`
    - `predict(self, head_outputs, strand)`
    - `from_config_file(cls, config_path)`
- **Functions**
  - `_isoform_to_key(iso)`
  - `preprocess_sequence(chrom, start, end)`
  - `preprocess_variant(vcf_like, window)`

**File:** `src/betadogma/nmd/nmd_model.py`
- **Classes**
  - NMDModel
    - `__init__(self, config)`
    - `predict(self, isoform)`

**File:** `src/betadogma/nmd/nmd_rules.py`
- **Functions**
  - `rule_ptc_before_last_junction(isoform, ptc_rule_threshold)`

**File:** `src/betadogma/variant/encode.py`
- **Functions**
  - `encode_variant(spec, window)`

**File:** `src/betadogma/variant/simulate.py`
- **Functions**
  - `make_ptc_variant()`

**File:** `tests/core/test_heads.py`
- **Functions**
  - `dummy_embeddings()`
  - `test_splice_head_shapes(dummy_embeddings)`
  - `test_tss_head_shapes(dummy_embeddings)`
  - `test_polya_head_shapes(dummy_embeddings)`
  - `test_orf_head_shapes(dummy_embeddings)`
  - `test_head_architectures(dummy_embeddings, use_conv)`
  - `test_betadogma_model_forward_pass_shapes()`

**File:** `tests/decoder/test_decoder_logic.py`
- **Functions**
  - `scoring_config()`
  - `mock_head_outputs()`
  - `decoder_config()`
  - `test_get_spliced_cDNA_positive_strand()`
  - `test_get_spliced_cDNA_negative_strand()`
  - `_dna_to_tensor(dna_str)`
  - `test_score_orf_sequence_valid(scoring_config, mock_head_outputs)`
  - `test_score_orf_ptc_penalty(scoring_config, mock_head_outputs)`
  - `test_score_orf_kozak_bonus(scoring_config, mock_head_outputs)`
  - `test_score_orf_no_valid_orf(scoring_config, mock_head_outputs)`
  - `test_splice_graph_builder_negative_strand(decoder_config)`

**File:** `tests/decoder/test_isoform_decoder.py`
- **Functions**
  - `decoder_config()`
  - `test_exon_creation()`
  - `test_isoform_creation()`
  - `test_splice_graph_builder_init(decoder_config)`
  - `test_isoform_enumerator_init(decoder_config)`
  - `test_isoform_scorer_init(decoder_config)`
  - `test_isoform_decoder_init(decoder_config)`

**File:** `tests/decoder/test_isoform_decoder_minus_strand.py`
- **Classes**
  - CdsWindow
- **Functions**
  - `toy_minus_gene()`
  - `mock_head_outputs(device)`
  - `test_minus_strand_splice_edge_directionality()`
  - `test_minus_strand_full_decoder_ordering()`
  - `test_minus_strand_orf_roles_via_scoring()`
  - `test_minus_strand_single_exon_cds()`
  - `test_minus_strand_utr_only()`

**File:** `tests/decoder/test_isoform_enumerator.py`
- **Functions**
  - `simple_splice_graph()`
  - `decoder_config()`
  - `test_isoform_enumerator_finds_best_path(simple_splice_graph, decoder_config)`
  - `test_isoform_enumerator_respects_max_paths(simple_splice_graph, decoder_config)`
  - `test_isoform_enumerator_empty_graph(decoder_config)`

**File:** `tests/decoder/test_splice_graph_builder.py`
- **Functions**
  - `test_find_peaks_basic()`
  - `test_find_peaks_top_k()`
  - `test_find_peaks_no_peaks()`
  - `sample_head_outputs()`
  - `decoder_config()`
  - `test_splice_graph_builder_positive_strand(sample_head_outputs, decoder_config)`
  - `test_splice_graph_builder_negative_strand(decoder_config)`
  - `test_splice_graph_builder_negative_strand_regression(decoder_config)`

**File:** `tests/test_import.py`
- **Functions**
  - `test_import()`

**File:** `train/make_training_data.py`
- **Functions**
  - `load_yaml(path)`
  - `resolve_path(p, base)`
  - `run_module_cli(module, args)`
  - `call_module_entrypoint(module, kwargs)`
  - `ensure_dir(p)`
  - `expect_files(paths)`
  - `step_prepare_gencode(cfg, cfg_dir)`
  - `step_prepare_gtex(cfg, cfg_dir)`
  - `step_prepare_data(cfg, cfg_dir)`
  - `main()`

**File:** `train/train.py`
- **Classes**
  - JsonlSeqDataset(Dataset)
    - `__init__(self, path, max_len, use_strand, revcomp_minus)`
    - `__len__(self)`
    - `__getitem__(self, idx)`
  - TinySeqModel(nn.Module)
    - `__init__(self, vocab_size, embed_dim, hidden, dropout)`
    - `forward(self, x_long)`
  - LitSeq(pl.LightningModule)
    - `__init__(self, model, lr, weight_decay)`
    - `forward(self, x_long)`
    - `_shared_step(self, batch, stage)`
    - `training_step(self, batch, _)`
    - `validation_step(self, batch, _)`
    - `on_validation_epoch_end(self)`
    - `configure_optimizers(self)`
  - SeqDataModule(pl.LightningDataModule)
    - `__init__(self, dcfg, cfg_dir)`
    - `_resolve(self, p)`
    - `setup(self, stage)`
    - `train_dataloader(self)`
    - `val_dataloader(self)`
    - `_toy_ds(n, L)`
- **Functions**
  - `load_config(path)`
  - `set_seed(seed)`
  - `revcomp(seq)`
  - `encode_seq(seq, max_len, pad_value)`
  - `collate_batch(batch)`
  - `_maybe_load_yaml_or_dict(value, cfg_dir)`
  - `build_model(mcfg, max_len, cfg_dir)`
  - `build_trainer(tcfg, cfg_dir)`
  - `main()`

**File:** `train/train_isoform_ranker.py`
- **Classes**
  - DummyEncoder(nn.Module)
    - `__init__(self, model_name)`
    - `forward(self, input_ids, attention_mask)`
  - PairwiseHingeLoss(nn.Module)
    - `__init__(self, margin)`
    - `forward(self, positive_scores, negative_scores)`
- **Functions**
  - `monkeypatch_encoder()`
  - `get_mock_batch(device)`
  - `select_candidates(candidates, true_isoform)`
  - `train_one_epoch(model, optimizer, loss_fn, device, mock_loader)`
  - `main()`


---

## File Inventory
| Path | Size (bytes) | SHA256 |
|------|---------------|--------|
| `.github/workflows/ci.yml` | 675 | `e5f33e88d6abeb388daf468a9eff90cea7b0e65341d3f181599e56c827d73e5e` |
| `.gitignore` | 124 | `36e08675036e95f7b6d1185200142b290e8e15908185bed865bfa9dd0161bec5` |
| `CONTRIBUTING.md` | 661 | `0b22d6461d5c012c96fb103930e25ea045c6770c565b13f855ffdd7a606cbddc` |
| `LICENSE` | 1062 | `5e56011466626b6928e9a878d9ea8e124ff63cde1996bc5924fbe38ffc549cdc` |
| `README.md` | 4209 | `312bf88e5d6a72385f6504507dcbf179750e3d8c81affc61ddfe7f072cf1841d` |
| `docs/DATASETS.md` | 924 | `e80d27337122ab880b44d0de91c5b1efb43565d84f039204ffbc8c570c282de5` |
| `docs/MODEL_CARD.md` | 565 | `38adee087894d7532372462dbea5c8e727c613f49d7cf4d8a14ce19b61f567df` |
| `docs/TASKS.md` | 488 | `ecc41147f5697bd54aed55230db0792e3c35d7f31c32eb32480526e93aac47ab` |
| `llm_project_brief.py` | 12650 | `09756373d6b95d70a15d68f0f985c11bd5f185e7425b42faaa187d72482bc31c` |
| `notebooks/README.md` | 64 | `70424a1d70c886292bbbb2dfdae2341838b6c5a74b81dcdde262d621ad917a37` |
| `poetry.lock` | 272650 | `4155c471a98b88d7e0cbbdfd90620b847a76c1e5598007886e890f14a2af4d31` |
| `project_brief.md` | 8161 | `9e588220232344e9e9ad732f5a86daee7ebe36c44625b90854969e54c4985597` |
| `pyproject.toml` | 697 | `fc1e9c9bf93b61fff356983fc12ecd5e046366178af3360fc6a4f270eb839a2d` |
| `src/betadogma/__init__.py` | 64 | `08f37e84b77e7e88d9c1e82b090be82c3ce66806598f6f9eaf8d44076099bf62` |
| `src/betadogma/core/__init__.py` | 48 | `d8dead261760c5dffaac5cd2baffad171a570f3740c7937208cefa8083ac9712` |
| `src/betadogma/core/encoder_nt.py` | 1062 | `a66a42879a9fb12216bfed285da60ab773766b1dde0af532c8bdc6a0a5550eee` |
| `src/betadogma/core/heads.py` | 3465 | `32e7972506edf377a717ce576938b616b7e3221d39b209d226b29ec4badff6ff` |
| `src/betadogma/core/losses.py` | 230 | `dbc2d16fb335497ee5dacfca7525e989d0867174e8311a41177f6ae394d97ef8` |
| `src/betadogma/decoder/__init__.py` | 299 | `393d3ace80b172d222ebe9e4deeb7f6e5820554230053d9fae381abea64ceaf4` |
| `src/betadogma/decoder/evaluate.py` | 2841 | `f86c2bcc80dd5f282aa16fb5ef38ef0aeab9d16a44d240b12a26ce99de0ca52e` |
| `src/betadogma/decoder/isoform_decoder.py` | 25609 | `23bab97b50e386e1d9d374ea30fa320b15c695ba0f2088327c0e4ee2013a9eba` |
| `src/betadogma/decoder/types.py` | 738 | `bbe05f8b39cf48c133e4efcdff0cfec42d238b615c072eb09795141cb04a931e` |
| `src/betadogma/experiments/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/betadogma/experiments/config/default.yaml` | 967 | `5b98d75fc62c833bc99b030182c2affd9529272d0d3a58e070637fdeecdae117` |
| `src/betadogma/experiments/train.py` | 6266 | `a68e6510ea7575ba5802870528c07219c4b7e0b3f3ecbcf2713cc7880f041ac3` |
| `src/betadogma/experiments/train_decoder.py` | 6828 | `339d13d87696e7ed1ba6df6582da1bbb9691280d1235f248239b2a5f43b224d4` |
| `src/betadogma/model.py` | 6557 | `221d596e9349d1d68c19cc2a50061f6a3875b770e43549a4fd872e7ca48b649c` |
| `src/betadogma/nmd/__init__.py` | 52 | `c44373b3cb50e8dda4aeda14c60312d4b85052b97ae69a53fbabe6081a368c20` |
| `src/betadogma/nmd/nmd_model.py` | 1165 | `af134b3705d00d5f5c79fa0c60b50c28f6a66c6841ea01bf2e5b54080318be43` |
| `src/betadogma/nmd/nmd_rules.py` | 3022 | `3aadead8e03ed5f98e297bfa051b25965b5120e7feeede5db16982f93c006166` |
| `src/betadogma/variant/__init__.py` | 50 | `d65280a5ca6ee9735158d43ce1740134873b8f0299ee1fea7987c6aec886b167` |
| `src/betadogma/variant/encode.py` | 304 | `057143235fec61d387091f4530ccfea67138ea75a510dfcd8fb7d418e9db60d0` |
| `src/betadogma/variant/simulate.py` | 192 | `2a350ffe4f7ab40df253add2a8336c85cf8f4701005395152ead50c90350ddb4` |
| `tests/core/test_heads.py` | 3819 | `ca08598240751532525fad05c02eb434e8dd2369b8690648d68fe08a14f2f882` |
| `tests/decoder/test_decoder_logic.py` | 7176 | `7430b56f049b39151334324bdd1fccaacad073e2a932909904869daed8144d50` |
| `tests/decoder/test_isoform_decoder.py` | 2365 | `63df80bd87f52aeda20ba1f4316dd5fda755dd230b3235b8bfcd4199ad3d676c` |
| `tests/decoder/test_isoform_decoder_minus_strand.py` | 9267 | `fd7d95d3c39f4d2bf299258b4b6b58203cd3b5781a49287412b3df3294cbe05e` |
| `tests/decoder/test_isoform_enumerator.py` | 3545 | `351ad39d334a1794b956bb15836fff6b6394432dd86906f64e91cc5b9e734543` |
| `tests/decoder/test_splice_graph_builder.py` | 9068 | `c2ab8634031d872789a3e83401477294f659e2c6a33823d5547ce3dd4d16197f` |
| `tests/test_import.py` | 39 | `1be7b0896026aeb8cbbd5bebefef1f2da083e093fb63fa76a44cea9c974258fb` |
| `train/checkpoints/default/epoch=18-val_score=0.0000.ckpt` | 6311789 | `2af617a5722df8675bb2825b5c9bdfc5a3e31a97db80f8a3884101a0bfcb4222` |
| `train/checkpoints/default/epoch=19-val_score=0.0000.ckpt` | 6311789 | `6acde2532f9f1fb71d8df79456022e677c11fd75eade7c4d262ba0e29f1fefd6` |
| `train/checkpoints/default/last.ckpt` | 6311789 | `6acde2532f9f1fb71d8df79456022e677c11fd75eade7c4d262ba0e29f1fefd6` |
| `train/configs/data.base.yaml` | 1971 | `3535a258d91a68024ba15924246d7929539ae167df55d8584c0c0f3c1b8833da` |
| `train/configs/train.base.yaml` | 1051 | `80cdc29e0c5f25b1f9db6c940946afded40ac17f575fc5e620b0cc1e60764003` |
| `train/make_training_data.py` | 7210 | `baab07bea7f1279b33e5d5596361dfaf9c7f16291537dd3d0d55bb14f179aded` |
| `train/prepare_model.py` | 2534 | `ecc2a5fe3ba088ff586cccccb8981dda416d1781e7a346ef8099126d90e58931` |
| `train/runs/betadogma/version_0/events.out.tfevents.1759711084.codespaces-7eda66.59460.0` | 88 | `c36e577bd4ac66fc8426c617da7dc856a68e253660d729f25e2b14f056db3b6d` |
| `train/runs/default/version_0/events.out.tfevents.1759707937.codespaces-7eda66.26280.0` | 8457 | `29e0da3ae74fc3fb78f3eae3a1c2e781c045e5cc14861ddacd0422485f57286d` |
| `train/runs/default/version_0/hparams.yaml` | 3 | `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356` |
| `train/train.py` | 16815 | `079e5a4c1fbace8800a403596c1166d939f6e7197bde89226f9f8c51c718e1ff` |
| `train/train_isoform_ranker.py` | 7033 | `5c8046a1a2113fb9fb7ba0d6747ed50c9ce816c728ad2a0015bd56f7f55a24c2` |

---

## Code Previews
_First 0 lines per file (capped at 0 bytes)._

<<<FILE:llm_project_brief.py>>>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_project_brief.py
Create an LLM-friendly Markdown brief of a Python project:
  1) Project overview + tree
  2) Classes & functions (parsed via AST)
  3) File inventory (path, size, sha256)
  4) Optional code previews (first N lines) in delimited blocks

Usage:
  python llm_project_brief.py [ROOT] > project_brief.md

Options:
  --preview                  include code previews (default on)
  --no-preview               disable code previews
  --preview-max-lines N      lines per file (default 60)
  --preview-max-bytes N      byte cap per file (default 40000)
  --preview-only-py          preview only .py files (default on)
  --no-preview-only-py       preview any text file
  --exclude-dir NAME ...     directory names to exclude (repeatable)
  --exclude-file GLOB ...    file glob patterns to exclude (repeatable)
  --json JSON_PATH           also write a machine-readable JSON with the same sections
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any
from datetime import datetime, UTC


# ------------------------ Configuration ------------------------
DEFAULT_EXCLUDE_DIRS = [
    ".git", "__pycache__", ".venv", "venv", "data", "build", "dist", ".tox",
    ".idea", ".vscode", ".ruff_cache", ".pytest_cache", ".mypy_cache",
]
DEFAULT_EXCLUDE_FILES = [
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib", "*.egg-info", "*.egg",
]

# --------------------------- Helpers ---------------------------

def human_path(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def is_text_file(path: Path, max_probe_bytes: int = 4096) -> bool:
    # Simple heuristic: if it decodes as utf-8 (with errors ignored) and has few NULs
    try:
        b = path.open("rb").read(max_probe_bytes)
    except Exception:
        return False
    if not b:
        return True
    nul = b.count(b"\x00")
    try:
        b.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return nul == 0

def walk_tree(root: Path, exclude_dirs: List[str]) -> Iterator[Tuple[Path, List[Path], List[Path]]]:
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # dirs-first, apply excludes by name
        dirnames[:] = sorted([d for d in dirnames if d not in exclude_dirs])
        yield p, [p / d for d in dirnames], sorted(p / f for f in filenames)

def render_tree(root: Path, exclude_dirs: List[str], exclude_files: List[str]) -> str:
    # Minimal tree printer (dirs first)
    lines: List[str] = []
    def _should_skip_file(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_files)

    prefix_map: Dict[Path, str] = {}

    def list_dir(base: Path, prefix: str = ""):
        entries_dirs: List[Path] = []
        entries_files: List[Path] = []
        try:
            for entry in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                if entry.is_dir():
                    if entry.name in exclude_dirs:
                        continue
                    entries_dirs.append(entry)
                else:
                    if _should_skip_file(entry.name):
                        continue
                    entries_files.append(entry)
        except PermissionError:
            return
        entries = entries_dirs + entries_files
        total = len(entries)
        for i, entry in enumerate(entries):
            last = (i == total - 1)
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if last else "│   "
                list_dir(entry, prefix + extension)
    lines.append(root.name)
    list_dir(root)
    return "\n".join(lines)

# ----------------------- AST extraction -----------------------

@dataclass
class MethodInfo:
    name: str
    args: List[str]

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[MethodInfo]

@dataclass
class FunctionInfo:
    name: str
    args: List[str]

def parse_api(py_path: Path) -> Tuple[List[ClassInfo], List[FunctionInfo]]:
    try:
        src = py_path.read_text(encoding="utf-8")
    except Exception:
        return [], []
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return [], []

    classes: List[ClassInfo] = []
    functions: List[FunctionInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: List[MethodInfo] = []
            for cnode in node.body:
                if isinstance(cnode, ast.FunctionDef):
                    args = [a.arg for a in cnode.args.args]  # no *args/**kwargs expansion here
                    methods.append(MethodInfo(cnode.name, args))
            bases: List[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    parts = []
                    cur: Any = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    bases.append(".".join(reversed(parts)))
                else:
                    bases.append(ast.dump(b, annotate_fields=False))
            classes.append(ClassInfo(node.name, bases, methods))
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            functions.append(FunctionInfo(node.name, args))
    return classes, functions

# ----------------------- Inventory & previews -----------------------

@dataclass
class FileRecord:
    path: str
    size: int
    sha256: str

def collect_files(root: Path, exclude_dirs: List[str], exclude_files: List[str]) -> List[Path]:
    out: List[Path] = []
    for dp, dnames, fnames in walk_tree(root, exclude_dirs):
        for f in fnames:
            if any(fnmatch.fnmatch(f.name, pat) for pat in exclude_files):
                continue
            out.append(f)
    return sorted(out)

def preview_file(path: Path, max_lines: int, max_bytes: int) -> str:
    """Preview file with optional caps. <=0 disables that cap."""
    no_line_cap = max_lines is None or max_lines <= 0
    no_byte_cap = max_bytes is None or max_bytes <= 0

    # Fast path: both caps disabled → read all
    if no_line_cap and no_byte_cap:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    buf = io.StringIO()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            total_bytes = 0
            for i, line in enumerate(f, 1):
                enc = line
                new_bytes = len(enc.encode("utf-8"))

                # only enforce a cap if it's enabled
                if (not no_line_cap and i > max_lines) or (not no_byte_cap and (total_bytes + new_bytes) > max_bytes):
                    break

                buf.write(line.rstrip("\n"))
                buf.write("\n")
                total_bytes += new_bytes
    except Exception:
        return ""
    return buf.getvalue()

# ----------------------------- Main -----------------------------

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("root", nargs="?", default=".", help="Project root")
    p.add_argument("--preview", dest="preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument("--preview-max-lines", type=int, default=60)
    p.add_argument("--preview-max-bytes", type=int, default=40000)
    p.add_argument("--preview-only-py", dest="preview_only_py", action="store_true", default=True)
    p.add_argument("--no-preview-only-py", dest="preview_only_py", action="store_false")
    p.add_argument("--exclude-dir", action="append", default=None)
    p.add_argument("--exclude-file", action="append", default=None)
    p.add_argument("--json", dest="json_out", default=None, help="Optional JSON output path")
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    exclude_dirs = args.exclude_dir or DEFAULT_EXCLUDE_DIRS[:]
    exclude_files = args.exclude_file or DEFAULT_EXCLUDE_FILES[:]

    # Header
    print("# Project Overview\n")
    print("### Detected Environment")
    print(f"- Root: `{root}`")
    print(f"- Date: `{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    print(f"- Python: `{sys.version.split()[0]}`\n")


    # Tree
    print("## Structure (tree)")
    print("```text")
    print(render_tree(root, exclude_dirs, exclude_files))
    print("```\n---\n")

    # API map
    print("## Python API Map (classes & functions)")
    api_map: Dict[str, Dict[str, Any]] = {}
    any_api = False
    for py in collect_files(root, exclude_dirs, exclude_files):
        if py.suffix != ".py":
            continue
        classes, functions = parse_api(py)
        if not classes and not functions:
            continue
        any_api = True
        print(f"**File:** `{human_path(py, root)}`")
        if classes:
            print("- **Classes**")
            for c in classes:
                base_str = f"({', '.join(c.bases)})" if c.bases else ""
                print(f"  - {c.name}{base_str}")
                for m in c.methods:
                    print(f"    - `{m.name}({', '.join(m.args)})`")
        if functions:
            print("- **Functions**")
            for f in functions:
                print(f"  - `{f.name}({', '.join(f.args)})`")
        print()
        api_map[str(py)] = {
            "classes": [asdict(c) for c in classes],
            "functions": [asdict(f) for f in functions],
        }
    if not any_api:
        print("_No top-level classes or functions found in .py files._")
    print("\n---\n")

    # File inventory
    print("## File Inventory")
    files = collect_files(root, exclude_dirs, exclude_files)
    print("| Path | Size (bytes) | SHA256 |")
    print("|------|---------------|--------|")
    inventory: List[FileRecord] = []
    for f in files:
        try:
            size = f.stat().st_size
            digest = sha256sum(f)
        except Exception:
            size, digest = -1, ""
        print(f"| `{human_path(f, root)}` | {size} | `{digest}` |")
        inventory.append(FileRecord(path=human_path(f, root), size=size, sha256=digest))
    print("\n---\n")

    # Optional previews
    preview_blocks: Dict[str, str] = {}
    if args.preview:
        print("## Code Previews")
        print(f"_First {args.preview_max_lines} lines per file (capped at {args.preview_max_bytes} bytes)._")
        print()
        for f in files:
            if args.preview_only_py and f.suffix != ".py":
                continue
            if not is_text_file(f):
                continue
            content = preview_file(f, args.preview_max_lines, args.preview_max_bytes)
            if not content.strip():
                continue
            rel = human_path(f, root)
            print(f"<<<FILE:{rel}>>>")
            # Don’t wrap in Markdown fences; use explicit markers for easier chunking
            sys.stdout.write(content)
            if not content.endswith("\n"):
                print()
            print(f"<<<END:{rel}>>>\n")
            preview_blocks[rel] = content

    # Optional JSON mirror
    if args.json_out:
        payload = {
            "root": str(root),
            "environment": {
                "python": sys.version.split()[0],
                "utc": __import__('datetime').datetime.utcnow().isoformat() + "Z",
            },
            "exclude_dirs": exclude_dirs,
            "exclude_files": exclude_files,
            "api_map": api_map,
            "inventory": [asdict(rec) for rec in inventory],
            "previews": preview_blocks,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
<<<END:llm_project_brief.py>>>

<<<FILE:src/betadogma/__init__.py>>>
# BetaDogma Package
# "Revising the Central Dogma through data."
<<<END:src/betadogma/__init__.py>>>

<<<FILE:src/betadogma/core/__init__.py>>>
"""Core modules: encoder and per-base heads."""
<<<END:src/betadogma/core/__init__.py>>>

<<<FILE:src/betadogma/core/encoder_nt.py>>>
# SPDX-License-Identifier: MIT
# src/betadogma/core/encoder_nt.py
from __future__ import annotations
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

class NTEncoder:
    """Transformers-native nucleotide encoder. Outputs [B, L, D] at base resolution (bin_size=1)."""
    def __init__(self, model_id: str = DEFAULT_NT, device: str = "auto"):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.bin_size = 1

    @torch.no_grad()
    def forward(self, seqs: list[str]) -> torch.Tensor:
        toks = self.tok(seqs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model(**toks)
        return out.last_hidden_state  # [B, L, D]
<<<END:src/betadogma/core/encoder_nt.py>>>

<<<FILE:src/betadogma/core/heads.py>>>
import torch
import torch.nn as nn
from typing import Dict

class _ConvHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, out_ch: int, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.norm = nn.LayerNorm(d_in)
            self.net = nn.Sequential(
                nn.Conv1d(d_in, d_in, kernel_size=11, groups=d_in, padding=5),      # depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_in, kernel_size=5, groups=d_in, padding=4, dilation=2),  # dilated depthwise
                nn.GELU(),
                nn.Conv1d(d_in, d_hidden, kernel_size=1),  # pointwise
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_hidden, out_ch, kernel_size=1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Lr, D]
        if self.use_conv:
            x = self.norm(x)              # Apply norm on [B, Lr, D]
            x = x.transpose(1, 2)         # [B, D, Lr]
            y = self.net(x)               # [B, out_ch, Lr]
            y = y.transpose(1, 2)         # [B, Lr, out_ch]
        else:
            y = self.net(x)               # [B, Lr, out_ch]
        return y

class SpliceHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=2, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.proj(embeddings)    # [B, Lr, 2]
        return {"donor": logits[..., 0:1], "acceptor": logits[..., 1:2]}

class TSSHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"tss": self.proj(embeddings)}  # [B, Lr, 1]

class PolyAHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.proj = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"polya": self.proj(embeddings)}  # [B, Lr, 1]

class ORFHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 768, dropout: float = 0.1, use_conv: bool = True):
        super().__init__()
        self.start = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.stop  = _ConvHead(d_in, d_hidden, out_ch=1, dropout=dropout, use_conv=use_conv)
        self.frame = _ConvHead(d_in, d_hidden, out_ch=3, dropout=dropout, use_conv=use_conv)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "start": self.start(embeddings),  # [B, Lr, 1]
            "stop":  self.stop(embeddings),   # [B, Lr, 1]
            "frame": self.frame(embeddings),  # [B, Lr, 3]
        }
<<<END:src/betadogma/core/heads.py>>>

<<<FILE:src/betadogma/core/losses.py>>>
"""
Loss functions for multi-task training:
- BCE for per-base classification
- KL divergence for ψ
- Consistency losses for NMD rule vs learned
"""
def bce_loss(*args, **kwargs):
    pass

def kl_loss(*args, **kwargs):
    pass
<<<END:src/betadogma/core/losses.py>>>

<<<FILE:src/betadogma/decoder/__init__.py>>>
"""Isoform graph construction and decoding."""

from .types import Isoform, Exon
from .isoform_decoder import (
    IsoformDecoder,
    IsoformScorer,
    IsoformEnumerator,
    SpliceGraphBuilder,
    SpliceGraph,
)
from .evaluate import (
    exon_chain_match,
    junction_f1,
    top_k_recall,
)
<<<END:src/betadogma/decoder/__init__.py>>>

<<<FILE:src/betadogma/decoder/evaluate.py>>>
"""
Evaluation utilities for the isoform decoder.

This module provides functions to compare predicted isoforms against a
ground truth, calculating metrics such as:
- Exact exon chain match rate
- Splice junction F1 score
- Top-K recall
"""

from typing import List, Set, Tuple

from .types import Isoform


def get_junctions(isoform: Isoform) -> Set[Tuple[int, int]]:
    """Extracts a set of splice junctions from an isoform."""
    junctions = set()
    if len(isoform.exons) > 1:
        for i in range(len(isoform.exons) - 1):
            # Junction is from the end of one exon to the start of the next
            junctions.add((isoform.exons[i].end, isoform.exons[i+1].start))
    return junctions


def exon_chain_match(predicted: Isoform, ground_truth: Isoform) -> bool:
    """
    Checks if the predicted isoform has the exact same exon chain as the
    ground truth.

    Returns:
        True if the exon start/end coordinates match exactly, False otherwise.
    """
    if len(predicted.exons) != len(ground_truth.exons):
        return False

    pred_exons = sorted([(e.start, e.end) for e in predicted.exons])
    true_exons = sorted([(e.start, e.end) for e in ground_truth.exons])

    return pred_exons == true_exons


def junction_f1(predicted: Isoform, ground_truth: Isoform) -> Tuple[float, float, float]:
    """
    Calculates the F1 score for splice junctions.

    Returns:
        A tuple containing (precision, recall, f1_score).
    """
    pred_junctions = get_junctions(predicted)
    true_junctions = get_junctions(ground_truth)

    if not true_junctions and not pred_junctions:
        return (1.0, 1.0, 1.0) # Both are single-exon, perfect match
    if not true_junctions or not pred_junctions:
        return (0.0, 0.0, 0.0) # One is single-exon, the other is not

    tp = len(pred_junctions.intersection(true_junctions))
    fp = len(pred_junctions.difference(true_junctions))
    fn = len(true_junctions.difference(pred_junctions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def top_k_recall(predicted_isoforms: List[Isoform], ground_truth: Isoform, k: int) -> bool:
    """
    Checks if the ground truth isoform is present in the top-K predicted isoforms.

    Args:
        predicted_isoforms: A list of predicted isoforms, sorted by score.
        ground_truth: The ground truth isoform.
        k: The number of top predictions to consider.

    Returns:
        True if a perfect match is found within the top K, False otherwise.
    """
    for i in range(min(k, len(predicted_isoforms))):
        if exon_chain_match(predicted_isoforms[i], ground_truth):
            return True
    return False
<<<END:src/betadogma/decoder/evaluate.py>>>

<<<FILE:src/betadogma/decoder/isoform_decoder.py>>>
"""
Isoform decoder for BetaDogma.

This module contains the components for turning the outputs of the structural
heads (splice, TSS, polyA) into transcript isoform structures.

The process follows these steps:
1.  **SpliceGraphBuilder**: Constructs a graph where nodes are potential exon
    boundaries and edges connect them to form valid exons and introns.
2.  **IsoformEnumerator**: Traverses the splice graph to find the most likely
    paths, which correspond to candidate isoforms.
3.  **IsoformScorer**: Scores each candidate isoform based on a combination of
    the structural head logits and other priors (e.g., ORF validity).
"""
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx

from .types import Isoform, Exon


def _find_peaks(logits: torch.Tensor, threshold: float, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds peaks in a logit tensor that are above a threshold."""
    probs = torch.sigmoid(logits)
    peak_indices = (probs > threshold).nonzero(as_tuple=False).squeeze()

    if peak_indices.numel() == 0:
        return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)

    # Ensure peak_indices is always iterable (1-D)
    if peak_indices.dim() == 0:
        peak_indices = peak_indices.unsqueeze(0)

    if top_k and peak_indices.numel() > top_k:
        # If there are too many peaks, keep the ones with the highest probability
        peak_probs = probs[peak_indices]
        top_k_indices = torch.topk(peak_probs, k=top_k).indices
        peak_indices = peak_indices[top_k_indices]

    peak_probs = probs[peak_indices]

    return peak_indices, peak_probs


def _order_exons_by_transcription(exons: List[Exon], strand: str) -> List[Exon]:
    """Sorts a list of exons in transcription order."""
    # key is a tuple to ensure deterministic sorting
    key = lambda e: (e.start, e.end)
    reverse = strand == '-'
    return sorted(exons, key=key, reverse=reverse)


class SpliceGraph:
    """
    A graph representation of potential splice sites and exons, using networkx.
    Nodes are (start, end) tuples representing exons.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_exon(self, exon: Exon):
        """Adds an exon as a node in the graph."""
        node_key = (exon.start, exon.end)
        self.graph.add_node(node_key, score=exon.score, type='exon')

    def add_junction(self, from_exon: Exon, to_exon: Exon, score: float):
        """Adds a directed edge representing a splice junction."""
        from_key = (from_exon.start, from_exon.end)
        to_key = (to_exon.start, to_exon.end)
        self.graph.add_edge(from_key, to_key, score=score, type='junction')

    def __repr__(self):
        return f"SpliceGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


class SpliceGraphBuilder:
    """
    Builds a splice graph from the outputs of the structural heads.
    This version anchors the graph to high-confidence TSS and polyA sites.
    """
    def __init__(self, config: Dict):
        if "decoder" in config:
            config = config["decoder"]

        self.config = config
        self.thresholds = self.config.get("thresholds", {})
        self.priors = self.config.get("priors", {})
        self.max_starts = self.config.get("max_starts", 8)
        self.max_ends = self.config.get("max_ends", 8)
        self.allow_unanchored = self.config.get("allow_unanchored", False)
        self.min_exon_len = self.priors.get("min_exon_len", 1)
        self.max_intron_len = self.priors.get("max_intron_len", 500000)

    def _get_exons(self, starts: List, ends: List, start_scores: List, end_scores: List, strand: str) -> List[Exon]:
        """Helper to generate exons from paired start/end coordinates. Strand-aware."""
        exons = []
        is_fwd = strand == '+'
        for i, s_idx in enumerate(starts):
            for j, e_idx in enumerate(ends):
                # On fwd strand, start < end. On rev strand, start > end.
                if (is_fwd and s_idx < e_idx) or (not is_fwd and s_idx > e_idx):
                    start_coord, end_coord = (int(s_idx), int(e_idx)) if is_fwd else (int(e_idx), int(s_idx))
                    exon_len = end_coord - start_coord
                    if exon_len >= self.min_exon_len:
                        score = (start_scores[i] + end_scores[j]) / 2.0
                        exons.append(Exon(start=start_coord, end=end_coord, score=float(score)))
        return exons

    def build(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> SpliceGraph:
        """Takes raw head outputs and constructs a splice graph."""
        # 1. Find all peaks for relevant signals
        donor_logits = head_outputs["splice"]["donor"].squeeze()
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze()
        tss_logits = head_outputs.get("tss", {}).get("tss", torch.tensor([])).squeeze()
        polya_logits = head_outputs.get("polya", {}).get("polya", torch.tensor([])).squeeze()

        donor_indices, donor_scores = _find_peaks(donor_logits, self.thresholds.get("donor", 0.6))
        acceptor_indices, acceptor_scores = _find_peaks(acceptor_logits, self.thresholds.get("acceptor", 0.6))
        tss_indices, tss_scores = _find_peaks(tss_logits, self.thresholds.get("tss", 0.5), top_k=self.max_starts)
        polya_indices, polya_scores = _find_peaks(polya_logits, self.thresholds.get("polya", 0.5), top_k=self.max_ends)

        # 2. Create candidate exons based on strand-aware roles.
        # Transcriptional start sites: Acceptor, TSS
        # Transcriptional end sites:   Donor, PolyA

        internal_exons = self._get_exons(acceptor_indices, donor_indices, acceptor_scores, donor_scores, strand)
        first_exons = self._get_exons(tss_indices, donor_indices, tss_scores, donor_scores, strand)
        last_exons = self._get_exons(acceptor_indices, polya_indices, acceptor_scores, polya_scores, strand)
        single_exons = self._get_exons(tss_indices, polya_indices, tss_scores, polya_scores, strand)

        # 3. Combine exons based on anchoring policy
        candidate_exons = []
        if self.allow_unanchored:
            candidate_exons.extend(internal_exons)
        candidate_exons.extend(first_exons)
        candidate_exons.extend(last_exons)
        candidate_exons.extend(single_exons)

        # 4. De-duplicate exons and add to graph
        unique_exons = { (e.start, e.end): e for e in sorted(candidate_exons, key=lambda x: x.score, reverse=True) }
        exons_list = list(unique_exons.values())

        graph = SpliceGraph()
        for exon in exons_list:
            graph.add_exon(exon)

        # 5. Add valid junctions between exons
        donor_set = set(donor_indices.tolist())
        acceptor_set = set(acceptor_indices.tolist())

        # Sort exons by transcriptional order to find junctions
        sorted_for_junctions = _order_exons_by_transcription(exons_list, strand)

        for i, up_exon in enumerate(sorted_for_junctions):
            for j in range(i + 1, len(sorted_for_junctions)):
                down_exon = sorted_for_junctions[j]

                # Check for valid junction based on strand
                if strand == '+':
                    # Junction: up_exon.end (donor) -> down_exon.start (acceptor)
                    intron_len = down_exon.start - up_exon.end
                    if 0 < intron_len <= self.max_intron_len:
                        if up_exon.end in donor_set and down_exon.start in acceptor_set:
                            score = (up_exon.score + down_exon.score) / 2.0
                            graph.add_junction(up_exon, down_exon, score=score)
                else:  # strand == '-'
                    # Junction: up_exon.start (donor) -> down_exon.end (acceptor)
                    intron_len = up_exon.start - down_exon.end
                    if 0 < intron_len <= self.max_intron_len:
                        if up_exon.start in donor_set and down_exon.end in acceptor_set:
                            score = (up_exon.score + down_exon.score) / 2.0
                            graph.add_junction(up_exon, down_exon, score=score)
        return graph


class IsoformEnumerator:
    """
    Enumerates candidate isoforms from a splice graph using beam search.
    Considers all visited paths as potential candidates.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.beam_size = self.config.get("beam_size", 16)

    def enumerate(self, graph: SpliceGraph, max_paths: int, strand: str = '+') -> List[Isoform]:
        """
        Finds the top-K paths through the splice graph using beam search.
        """
        if not graph.graph or graph.graph.number_of_nodes() == 0:
            return []

        source_nodes = [n for n, d in graph.graph.in_degree() if d == 0]
        if not source_nodes:
            return []

        # Sort source nodes by transcription order to initialize the beam correctly
        # The node key is a (start, end) tuple.
        source_exons = [Exon(start=n[0], end=n[1], score=graph.graph.nodes[n]['score']) for n in source_nodes]
        sorted_source_exons = _order_exons_by_transcription(source_exons, strand)
        sorted_source_nodes = [(e.start, e.end) for e in sorted_source_exons]

        # A beam is a list of (cumulative_score, path) tuples
        beam = [(graph.graph.nodes[n]['score'], [n]) for n in sorted_source_nodes]
        beam.sort(key=lambda x: x[0], reverse=True)

        all_candidate_paths = []

        while beam:
            # Add all paths in the current beam to our list of candidates
            all_candidate_paths.extend(beam)
            new_beam = []

            for score, path in beam:
                last_node = path[-1]

                # Expand to neighbors
                for neighbor in graph.graph.neighbors(last_node):
                    # Path score is the sum of its exon scores
                    node_score = graph.graph.nodes[neighbor]['score']
                    new_score = score + node_score
                    new_path = path + [neighbor]
                    new_beam.append((new_score, new_path))

            # Prune the new beam to keep only the top-k paths
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_size]

            if not beam:
                break

        # De-duplicate paths, keeping the one with the highest score
        unique_paths = {}
        for score, path in all_candidate_paths:
            path_tuple = tuple(path)
            if path_tuple not in unique_paths or score > unique_paths[path_tuple]:
                unique_paths[path_tuple] = score

        # Sort final candidates by normalized score
        sorted_paths = sorted(unique_paths.items(), key=lambda item: item[1] / len(item[0]), reverse=True)

        # Convert top paths to Isoform objects
        isoforms = []
        for path_tuple, score in sorted_paths[:max_paths]:
            exons = []
            for node_key in path_tuple:
                node_data = graph.graph.nodes[node_key]
                start, end = node_key
                exons.append(Exon(start=start, end=end, score=node_data['score']))

            normalized_score = score / len(path_tuple) if path_tuple else 0
            isoforms.append(Isoform(exons=exons, strand=strand, score=normalized_score))

        return isoforms


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(sigmoid(x))"""
    return torch.log(torch.sigmoid(x) + 1e-9)


def _get_peak_log_prob(logits: torch.Tensor, index: int, window: int) -> torch.Tensor:
    """Gets the max log probability in a window around a specified index."""
    if logits is None or index is None:
        return torch.tensor(0.0)
    start = max(0, index - window)
    end = min(len(logits), index + window + 1)
    if start >= end:
        return torch.tensor(0.0)
    return torch.max(_log_sigmoid(logits[start:end]))


def _get_spliced_cDNA(isoform: Isoform, input_ids: torch.Tensor, token_map: Dict[int, str]) -> Tuple[str, List[str]]:
    """
    Constructs the spliced cDNA sequence for an isoform from input token IDs.
    Handles reverse complement for the negative strand.
    Returns the cDNA sequence and a list of the exon sequences in transcription order.
    """
    sequence_str = "".join([token_map.get(token_id, "N") for token_id in input_ids.squeeze().tolist()])

    # To get the spliced sequence for cDNA, we must assemble in genomic order.
    genomically_sorted_exons = sorted(isoform.exons, key=lambda e: e.start)
    genomic_exon_seqs = [sequence_str[exon.start:exon.end] for exon in genomically_sorted_exons]
    spliced_in_genomic_order = "".join(genomic_exon_seqs)

    # For the PTC check, we need exon sequences in transcription order.
    # We assume isoform.exons is already in transcription order from the decoder.
    tx_ordered_exon_seqs = [sequence_str[exon.start:exon.end] for exon in isoform.exons]

    if isoform.strand == '-':
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        cDNA = "".join(complement.get(base, "N") for base in reversed(spliced_in_genomic_order))
        return cDNA, tx_ordered_exon_seqs
    else:
        return spliced_in_genomic_order, tx_ordered_exon_seqs


def _score_orf(
    isoform: Isoform,
    head_outputs: Dict[str, torch.Tensor],
    scoring_config: Dict,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scores the validity of the open reading frame using either head outputs or a sequence scan.
    This function implements a two-tier scoring system as requested.
    """
    use_head = scoring_config.get("use_orf_head", True)
    if "splice" not in head_outputs or "donor" not in head_outputs["splice"]:
        device = torch.device("cpu") # Fallback device
    else:
        device = head_outputs["splice"]["donor"].device

    # --- Tier A: Head-driven scoring (default) ---
    if use_head:
        if "orf" not in head_outputs:
            return torch.tensor(0.0, device=device)
        alpha = scoring_config.get("orf_alpha", 0.5)
        beta = scoring_config.get("orf_beta", 0.3)
        gamma = scoring_config.get("orf_gamma", 0.6)

        start_logits = head_outputs["orf"]["start"].squeeze()
        stop_logits = head_outputs["orf"]["stop"].squeeze()
        frame_logits = head_outputs["orf"]["frame"].squeeze()
        frame_probs = torch.softmax(frame_logits, dim=-1)
        stop_probs = torch.sigmoid(stop_logits)

        if not isoform.exons:
            return torch.tensor(0.0, device=device)

        # isoform.exons are in transcription order, so no sort needed here.
        tx_to_gen_map = [p for exon in isoform.exons for p in range(exon.start, exon.end)]
        if not tx_to_gen_map:
            return torch.tensor(0.0, device=device)

        exonic_indices = torch.tensor(tx_to_gen_map, device=device)
        exonic_start_logits = start_logits[exonic_indices]
        top_k = min(scoring_config.get("max_start_candidates", 5), len(exonic_indices))

        if top_k == 0: return torch.tensor(0.0, device=device)

        candidate_start_probs, relative_indices = torch.topk(torch.sigmoid(exonic_start_logits), k=top_k)
        best_overall_score = torch.tensor(-1.0, device=device)

        for s_prob, s_rel_idx in zip(candidate_start_probs, relative_indices):
            s_tx_idx = s_rel_idx.item()
            # Iterate through all three possible reading frames for the given start
            for frame_offset in range(3):
                cds_frame_probs = []
                # Scan downstream from start codon
                for tx_pos in range(s_tx_idx + frame_offset, len(tx_to_gen_map), 3):
                    gen_pos = tx_to_gen_map[tx_pos]
                    current_frame = (tx_pos - s_tx_idx) % 3
                    cds_frame_probs.append(frame_probs[gen_pos, current_frame])

                    # Check for stop codon
                    if stop_probs[gen_pos] > 0.5:
                        stop_prob = stop_probs[gen_pos]
                        mean_frame_prob = torch.mean(torch.stack(cds_frame_probs))
                        score = (alpha * s_prob + beta * mean_frame_prob + alpha * stop_prob)

                        # Apply PTC penalty if stop is premature
                        if len(isoform.exons) > 1:
                            # isoform.exons are in transcription order, so no sort needed here.
                            exon_lengths = [e.end - e.start for e in isoform.exons[:-1]]
                            last_junction_pos = sum(exon_lengths)
                            if tx_pos < last_junction_pos - 55:
                                score -= gamma

                        if score > best_overall_score:
                            best_overall_score = score
                        break # Found a stop, end this frame scan
                else: # No stop found
                    score = (alpha * s_prob) - gamma
                    if score > best_overall_score:
                        best_overall_score = score

        return torch.max(torch.tensor(0.0, device=device), best_overall_score)

    # --- Tier B: Sequence-based fallback ---
    else:
        if input_ids is None:
            return torch.tensor(0.0, device=device)

        token_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
        # cDNA is 5'->3', exon_seqs is in transcription order
        cDNA, exon_seqs = _get_spliced_cDNA(isoform, input_ids, token_map)

        best_score = -1.0

        def is_strong_kozak(seq, pos):
            if pos < 3 or pos + 4 >= len(seq): return False
            return seq[pos-3] in "AG" and seq[pos+3] == "G"

        for i in range(len(cDNA) - 2):
            if cDNA[i:i+3] == "ATG": # Found a start codon
                for j in range(i, len(cDNA) - 2, 3):
                    if cDNA[j:j+3] in {"TAA", "TAG", "TGA"}: # Found a stop codon
                        cds_len_aa = (j - i) // 3
                        score = 0.5
                        if scoring_config.get("min_cds_len_aa", 50) <= cds_len_aa <= scoring_config.get("max_cds_len_aa", 10000):
                            score += 0.3
                        if is_strong_kozak(cDNA, i):
                            score += scoring_config.get("kozak_bonus", 0.2)

                        # PTC penalty
                        if len(exon_seqs) > 1:
                            # exon_seqs is in transcription order, so this is correct now.
                            last_junction_pos = sum(len(s) for s in exon_seqs[:-1])
                            if j < last_junction_pos - 55:
                                score -= scoring_config.get("orf_gamma", 0.6)

                        if score > best_score:
                            best_score = score
                        break # Stop scanning for this ORF
        return torch.tensor(max(0.0, best_score), device=device)


def _score_length(isoform: Isoform, priors: Dict) -> torch.Tensor:
    """Applies a soft penalty for extreme lengths."""
    # Placeholder for a more sophisticated prior
    # For now, just a small penalty for having very few or many exons
    num_exons = len(isoform.exons)
    if num_exons < 2 or num_exons > 20:
        return torch.tensor(-0.5)
    return torch.tensor(0.0)


class IsoformScorer(nn.Module):
    """
    Scores candidate isoforms based on structural signals and priors.
    This is a learnable module.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.scoring_config = self.config.get("scoring", {})
        self.priors = self.config.get("priors", {})

        # Define learnable weights for different components of the score
        self.w_spl = nn.Parameter(torch.tensor(self.scoring_config.get("w_spl", 1.0)))
        self.w_tss = nn.Parameter(torch.tensor(self.scoring_config.get("w_tss", 0.4)))
        self.w_pa = nn.Parameter(torch.tensor(self.scoring_config.get("w_pa", 0.4)))
        self.w_orf = nn.Parameter(torch.tensor(self.scoring_config.get("w_orf", 0.8)))
        self.w_len = nn.Parameter(torch.tensor(self.scoring_config.get("w_len", 0.1)))

    def forward(self, isoform: Isoform, head_outputs: Dict[str, torch.Tensor], input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates a score for a given isoform.
        """
        device = self.w_spl.device

        # 1. Splice Score
        donor_logits = head_outputs["splice"]["donor"].squeeze()
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze()

        s_spl = 0
        if isoform.exons and len(isoform.exons) > 1:
            if isoform.strand == '+':
                donor_indices = [exon.end for exon in isoform.exons[:-1]]
                acceptor_indices = [exon.start for exon in isoform.exons[1:]]
            else: # strand == '-'
                donor_indices = [exon.start for exon in isoform.exons[:-1]]
                acceptor_indices = [exon.end for exon in isoform.exons[1:]]

            # By using the logits directly, we ensure that good splice sites (positive logits)
            # contribute positively to the score, out-competing single-exon transcripts (score=0).
            donor_scores = donor_logits[donor_indices]
            acceptor_scores = acceptor_logits[acceptor_indices]

            s_spl = (torch.sum(donor_scores) + torch.sum(acceptor_scores)) / (len(donor_indices) + len(acceptor_indices))
        else:
            s_spl = torch.tensor(0.0) # No junctions for single-exon transcripts

        # 2. TSS Score
        tss_logits = head_outputs.get("tss", {}).get("tss", None)
        s_tss = torch.tensor(0.0)
        if tss_logits is not None and isoform.exons:
            tss_logits = tss_logits.squeeze()
            first_exon = isoform.exons[0]
            # 5' end of transcript is start of first exon (+), end of first exon (-)
            start_bin = first_exon.end if isoform.strand == '-' else first_exon.start
            s_tss = _get_peak_log_prob(tss_logits, start_bin, self.config.get("tss_pa_window", 1))

        # 3. PolyA Score
        polya_logits = head_outputs.get("polya", {}).get("polya", None)
        s_pa = torch.tensor(0.0)
        if polya_logits is not None and isoform.exons:
            polya_logits = polya_logits.squeeze()
            last_exon = isoform.exons[-1]
            # 3' end of transcript is end of last exon (+), start of last exon (-)
            end_bin = last_exon.start if isoform.strand == '-' else last_exon.end
            s_pa = _get_peak_log_prob(polya_logits, end_bin, self.config.get("tss_pa_window", 1))

        # 4. ORF Score
        s_orf = _score_orf(
            isoform,
            head_outputs,
            self.scoring_config,
            input_ids=input_ids
        )

        # 5. Length Score
        s_len = _score_length(isoform, self.priors)

        # Weighted sum of scores
        total_score = (
            self.w_spl * s_spl +
            self.w_tss * s_tss +
            self.w_pa * s_pa +
            self.w_orf * s_orf +
            self.w_len * s_len
        ).to(device)

        return total_score


class IsoformDecoder(nn.Module):
    """
    The main class that orchestrates the isoform decoding process.
    Inherits from nn.Module to hold the learnable scorer.
    """
    def __init__(self, config: Dict):
        super().__init__()
        # Handle both full config and sub-config for backward compatibility in tests
        if "decoder" in config:
            decoder_config = config["decoder"]
        else:
            decoder_config = config

        self.config = {"decoder": decoder_config} # Store consistently
        self.graph_builder = SpliceGraphBuilder(decoder_config)
        self.enumerator = IsoformEnumerator(decoder_config)
        self.scorer = IsoformScorer(decoder_config)

    def forward(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.
        This is an alias for the `decode` method for nn.Module compatibility.
        """
        return self.decode(head_outputs, strand=strand, input_ids=input_ids)

    def decode(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.

        Args:
            head_outputs: The dictionary of outputs from the model's structural heads.
            strand: The genomic strand.
            input_ids: Optional tensor of input token IDs for sequence-based scoring.

        Returns:
            A list of decoded Isoform objects, ranked by score.
        """
        splice_graph = self.graph_builder.build(head_outputs, strand=strand)

        max_candidates = self.config.get("decoder", {}).get("max_candidates", 64)
        candidates = self.enumerator.enumerate(splice_graph, max_paths=max_candidates, strand=strand)

        # Re-score the final candidates with the learnable scorer.
        for isoform in candidates:
            # Ensure exons are always in transcription order before scoring.
            isoform.exons = _order_exons_by_transcription(isoform.exons, isoform.strand)
            isoform.score = self.scorer(isoform, head_outputs, input_ids=input_ids).item()

        # Sort candidates by score in descending order
        candidates.sort(key=lambda iso: iso.score, reverse=True)

        return candidates
<<<END:src/betadogma/decoder/isoform_decoder.py>>>

<<<FILE:src/betadogma/decoder/types.py>>>
"""
Data structures for the isoform decoder.
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Exon:
    """Represents a single exon in an isoform."""
    start: int
    end: int
    score: float = 0.0

@dataclass
class Isoform:
    """
    Represents a single transcript isoform, composed of a chain of exons.
    """
    exons: List[Exon]
    strand: str
    score: float = 0.0
    cds: Optional[Tuple[int, int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> int:
        return self.exons[0].start if self.exons else -1

    @property
    def end(self) -> int:
        return self.exons[-1].end if self.exons else -1
<<<END:src/betadogma/decoder/types.py>>>

<<<FILE:src/betadogma/experiments/train.py>>>
# betadogma/experiments/train.py
"""
Phase 1: Structural fine-tuning for splice/TSS/polyA heads.

Usage:
  python -m betadogma.experiments.train --config betadogma/experiments/config/default.yaml
"""
from __future__ import annotations
import argparse
import os
import yaml
from glob import glob
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from betadogma.core.encoder_nt import NTEncoder
from betadogma.model import BetaDogmaModel

# ---------------- Data ----------------

class StructuralDataset(Dataset):
    def __init__(self, parquet_paths: List[str], max_shards: int | None = None):
        self.paths = sorted(parquet_paths)[:max_shards] if max_shards else sorted(parquet_paths)
        self._rows = []
        for p in self.paths:
            df = pd.read_parquet(p)
            self._rows.extend(df.to_dict("records"))

    def __len__(self): return len(self._rows)

    def __getitem__(self, idx):
        r = self._rows[idx]
        seq = r["seq"]
        # Convert binned labels to tensors
        donor = torch.tensor(r["donor"], dtype=torch.float32)
        acceptor = torch.tensor(r["acceptor"], dtype=torch.float32)
        tss = torch.tensor(r["tss"], dtype=torch.float32)
        polya = torch.tensor(r["polya"], dtype=torch.float32)
        return {"seq": seq, "donor": donor, "acceptor": acceptor, "tss": tss, "polya": polya}

def collate(batch):
    # Pad to max label length (Lr) in batch
    max_label_len = max(len(x["donor"]) for x in batch)

    def pad1d(x, target_len):
        return torch.nn.functional.pad(x, (0, target_len - len(x)))

    seqs = [b["seq"] for b in batch]
    donor = torch.stack([pad1d(b["donor"], max_label_len) for b in batch])
    acceptor = torch.stack([pad1d(b["acceptor"], max_label_len) for b in batch])
    tss = torch.stack([pad1d(b["tss"], max_label_len) for b in batch])
    polya = torch.stack([pad1d(b["polya"], max_label_len) for b in batch])

    return {"seqs": seqs, "donor": donor, "acceptor": acceptor, "tss": tss, "polya": polya}


# --------------- Training ----------------

def train(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize Encoder ---
    enc_type = cfg["encoder"].get("type", "nt")
    if enc_type == "nt":
        encoder = NTEncoder(
            model_id=cfg["encoder"].get("model_id") or "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        )
    else:
        raise ValueError(f"Unsupported encoder type: {enc_type}")

    d_model = encoder.hidden_size
    bin_size = encoder.bin_size # pylint: disable=unused-variable

    # --- Initialize Model and Data ---
    model = BetaDogmaModel(d_in=d_model, config=cfg).to(device)

    shards_glob = os.path.join(cfg["data"]["out_cache"], "*.parquet")
    paths = glob(shards_glob)
    assert paths, f"No parquet shards found at {shards_glob}"

    ds = StructuralDataset(paths, max_shards=cfg["trainer"].get("max_shards"))
    dl = DataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True, collate_fn=collate)

    # --- Optimizer and Loss Function ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["loss"]["pos_weight"]).to(device))
    loss_weights = cfg["loss"]

    # --- Training Loop ---
    model.train()
    for epoch in range(cfg["trainer"]["epochs"]):
        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['trainer']['epochs']}")
        for batch in pbar:
            opt.zero_grad()

            # --- Encoder Forward Pass ---
            # The NTEncoder handles tokenization and embedding internally.
            embeddings = encoder.forward(batch["seqs"]) # [B, L, D]

            # --- Model Forward Pass ---
            outputs = model(embeddings=embeddings)

            # --- Calculate Multi-Task Loss ---
            # Splice loss
            splice_logits_donor = outputs['splice']['donor'].squeeze(-1)
            splice_logits_acceptor = outputs['splice']['acceptor'].squeeze(-1)
            labels_donor = batch['donor'].to(device)
            labels_acceptor = batch['acceptor'].to(device)

            min_len = min(splice_logits_donor.shape[1], labels_donor.shape[1])
            loss_d = criterion(splice_logits_donor[:, :min_len], labels_donor[:, :min_len])
            loss_a = criterion(splice_logits_acceptor[:, :min_len], labels_acceptor[:, :min_len])
            splice_loss = (loss_d + loss_a) * loss_weights['w_splice']

            # TSS loss
            tss_logits = outputs['tss']['tss'].squeeze(-1)
            labels_tss = batch['tss'].to(device)
            min_len_tss = min(tss_logits.shape[1], labels_tss.shape[1])
            tss_loss = criterion(tss_logits[:, :min_len_tss], labels_tss[:, :min_len_tss]) * loss_weights['w_tss']

            # PolyA loss
            polya_logits = outputs['polya']['polya'].squeeze(-1)
            labels_polya = batch['polya'].to(device)
            min_len_polya = min(polya_logits.shape[1], labels_polya.shape[1])
            polya_loss = criterion(polya_logits[:, :min_len_polya], labels_polya[:, :min_len_polya]) * loss_weights['w_polya']

            # Combine losses
            loss = splice_loss + tss_loss + polya_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"].get("grad_clip", 1.0))
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(dl))
        print(f"Epoch {epoch+1}/{cfg['trainer']['epochs']} average loss: {avg_loss:.4f}")

    # --- Save Checkpoint ---
    ckpt_dir = cfg["trainer"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "betadogma_structural.pt"))
    print("Saved checkpoint:", ckpt_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
<<<END:src/betadogma/experiments/train.py>>>

<<<FILE:src/betadogma/experiments/train_decoder.py>>>
"""
Phase 2a: Decoder fine-tuning for structural accuracy.

This script trains the IsoformScorer's parameters using a listwise
ranking loss. It freezes the underlying encoder and structural heads,
focusing only on teaching the decoder to rank ground-truth isoforms highly.

Usage:
  python -m betadogma.experiments.train_decoder --config betadogma/experiments/config/default.yaml
"""
import argparse
import yaml
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import betadogma.model # For monkeypatching
from betadogma.model import BetaDogmaModel
from betadogma.decoder.types import Exon, Isoform

# --- Dummy Encoder ---
class DummyEncoder(nn.Module):
    """A dummy encoder that avoids loading the real model from Hugging Face."""
    def __init__(self, model_name: str):
        super().__init__()
        print(f"Using DummyEncoder. Model '{model_name}' will not be loaded.")

    def forward(self, *args, **kwargs):
        # This should not be called during decoder training as we provide head_outputs directly.
        raise NotImplementedError("The dummy encoder's forward method should not be called.")

# --- Helper function to compare isoforms ---
def are_isoforms_equal(iso1: Isoform, iso2: Isoform) -> bool:
    """Checks if two isoforms have the same exon chain."""
    if iso1.strand != iso2.strand or len(iso1.exons) != len(iso2.exons):
        return False
    for ex1, ex2 in zip(iso1.exons, iso2.exons):
        if ex1.start != ex2.start or ex1.end != ex2.end:
            return False
    return True

# --- Dataset for Ground-Truth Isoforms ---
class GroundTruthIsoformDataset(Dataset):
    """
    A placeholder dataset that yields model inputs and ground-truth isoforms.
    """
    def __init__(self, num_samples: int = 20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict:
        # For this test, we create a dummy sequence and a ground-truth isoform.
        # The model will try to predict this isoform from the sequence.
        true_exons = [Exon(start=100, end=200, score=0.9), Exon(start=300, end=400, score=0.8)]
        true_isoform = Isoform(exons=true_exons, strand="+", score=0.85)

        # We also need to provide mock head outputs that would lead to this isoform.
        # This is a bit artificial but necessary for a self-contained training script.
        donor_logits = torch.full((1000,), -10.0)
        acceptor_logits = torch.full((1000,), -10.0)
        donor_logits[200] = 5.0
        acceptor_logits[100] = 5.0
        donor_logits[400] = 4.0
        acceptor_logits[300] = 4.0
        # Add some negative distractors
        acceptor_logits[150] = 3.0
        donor_logits[250] = 3.0

        head_outputs = {
            "splice": {
                "donor": donor_logits.unsqueeze(0).unsqueeze(-1),
                "acceptor": acceptor_logits.unsqueeze(0).unsqueeze(-1),
            }
        }
        return {"head_outputs": head_outputs, "true_isoform": true_isoform, "strand": "+"}

def collate_decoder(batch):
    # In this simple case, we process one sample at a time, so no collating needed.
    return batch[0]

# --- Ranking Loss ---
def listwise_hinge_loss(positive_score: torch.Tensor, negative_scores: List[torch.Tensor], margin: float = 0.1):
    """
    Calculates a simple hinge loss to rank the positive sample above negatives.
    Loss = sum(max(0, margin - (positive_score - negative_score)))
    """
    if not negative_scores:
        return torch.tensor(0.0)
    negative_scores_tensor = torch.stack(negative_scores)
    losses = torch.clamp(margin - (positive_score - negative_scores_tensor), min=0)
    return losses.mean()

# --- Training Loop ---
def train_decoder(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Monkeypatch the real encoder with our dummy version before creating the model
    betadogma.model.BetaDogmaEncoder = DummyEncoder

    # 2. Initialize Model and freeze all params
    model = BetaDogmaModel(config=cfg).to(device)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze only the scorer parameters
    for param in model.isoform_decoder.scorer.parameters():
        param.requires_grad = True

    print("Model loaded. Encoder and heads are FROZEN.")
    print("Training decoder scorer parameters...")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.isoform_decoder.scorer.parameters(), lr=cfg["optimizer"]["lr"])

    # 5. Data
    ds = GroundTruthIsoformDataset()
    # Batch size must be 1 for this simple implementation
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_decoder)

    # 6. Training loop
    model.isoform_decoder.scorer.train()
    for epoch in range(cfg["trainer"]["epochs"]):
        total_loss = 0
        for batch in dl:
            optimizer.zero_grad()

            # a. Get candidate isoforms from the decoder
            # We use the mock head_outputs directly from the dataset for this test
            head_outputs = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in batch["head_outputs"].items()}
            candidates = model.isoform_decoder.decode(head_outputs, strand=batch["strand"])

            # b. Find positive and negative samples
            true_isoform = batch["true_isoform"]
            positive_candidate = None
            negative_candidates = []
            for cand in candidates:
                if are_isoforms_equal(cand, true_isoform):
                    positive_candidate = cand
                else:
                    negative_candidates.append(cand)

            if not positive_candidate:
                # If the true isoform was not even found, we can't compute loss for this sample.
                continue

            # c. Calculate scores for all candidates
            positive_score = model.isoform_decoder.scorer(positive_candidate, head_outputs)
            negative_scores = [model.isoform_decoder.scorer(neg, head_outputs) for neg in negative_candidates]

            # d. Compute loss
            loss = listwise_hinge_loss(positive_score, negative_scores, margin=0.1)

            # e. Backpropagate
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dl) if dl else 0
        print(f"Epoch {epoch+1}/{cfg['trainer']['epochs']} | Avg Loss: {avg_loss:.4f}")

    print("Decoder training complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_decoder(cfg)

if __name__ == "__main__":
    main()
<<<END:src/betadogma/experiments/train_decoder.py>>>

<<<FILE:src/betadogma/model.py>>>
"""
High-level BetaDogma model API.

This module exposes a simple interface for inference:
- BetaDogmaModel: orchestrates encoder, heads, decoder, and NMD predictor.
- preprocess_sequence / preprocess_variant: utilities to create inputs.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead
from .decoder.isoform_decoder import IsoformDecoder
from .decoder.types import Isoform
from .nmd.nmd_model import NMDModel


def _isoform_to_key(iso: Isoform) -> str:
    """Creates a unique string identifier for an isoform."""
    exon_str = ",".join(f"{e.start}-{e.end}" for e in iso.exons)
    return f"{iso.strand}:{exon_str}"


@dataclass
class Prediction:
    """
    Encapsulates the full output of a BetaDogma prediction.

    This object holds the raw candidate isoforms and computes final values
    (PSI, dominant isoform, NMD status) on demand via properties.
    """
    isoforms: list[Isoform]
    _nmd_model: NMDModel # A reference to the NMD model for on-the-fly prediction

    @property
    def psi(self) -> Dict[str, float]:
        """Calculate Percent Spliced In (PSI) values for all isoforms."""
        if not self.isoforms:
            return {}

        scores = torch.tensor([iso.score for iso in self.isoforms])
        psi_values = F.softmax(scores, dim=0)

        return {
            _isoform_to_key(iso): psi.item()
            for iso, psi in zip(self.isoforms, psi_values)
        }

    @property
    def dominant_isoform(self) -> Optional[Isoform]:
        """Return the isoform with the highest PSI value."""
        if not self.isoforms:
            return None
        # The dominant isoform corresponds to the one with the maximum raw score
        return max(self.isoforms, key=lambda iso: iso.score)

    @property
    def p_nmd(self) -> float:
        """Predict the NMD fate of the dominant isoform."""
        dom_iso = self.dominant_isoform
        if dom_iso is None:
            return 0.0
        return self._nmd_model.predict(dom_iso)


class BetaDogmaModel(nn.Module):
    """
    Coordinates the backbone encoder, per-base heads, and isoform decoder.
    This model is instantiated with a configuration dictionary that defines the
    architecture of all its components.
    """

    def __init__(self, d_in: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # --- 2. Prediction Heads ---
        # These modules process the encoder embeddings to make per-base predictions.
        head_config = self.config["heads"]
        self.splice_head = SpliceHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.tss_head = TSSHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.polya_head = PolyAHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )
        self.orf_head = ORFHead(
            d_in=d_in,
            d_hidden=head_config["hidden"],
            dropout=head_config["dropout"],
            use_conv=head_config["use_conv"],
        )

        # --- 3. Isoform Decoder ---
        # This component assembles predictions into transcript structures.
        self.isoform_decoder = IsoformDecoder(config=self.config.get("decoder", {}))

        # --- 4. NMD Model ---
        # This component predicts NMD fate.
        self.nmd_model = NMDModel(config=self.config.get("nmd", {}))


    def forward(self, embeddings: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Runs all prediction heads on pre-computed embeddings.
        This is the main forward pass for training, returning logits from each head.
        """
        # Pass embeddings through each prediction head
        outputs = {
            "splice": self.splice_head(embeddings),
            "tss": self.tss_head(embeddings),
            "polya": self.polya_head(embeddings),
            "orf": self.orf_head(embeddings),
            "embeddings": embeddings,
            "input_ids": input_ids, # Pass through for downstream use (e.g., sequence-based ORF scoring)
        }
        return outputs

    def predict(self, head_outputs: Dict[str, Any], strand: str = '+') -> Prediction:
        """
        Decodes the raw outputs from the heads into a final, structured prediction
        object that can compute isoform abundances and NMD status.

        Args:
            head_outputs: A dictionary containing the raw tensor outputs from the
                          model's forward pass.
            strand: The strand of the gene, either '+' or '-'.

        Returns:
            A Prediction object containing the candidate isoforms.
        """
        input_ids = head_outputs.get("input_ids")

        # 1. Decode all candidate isoforms from the graph
        candidate_isoforms = self.isoform_decoder.decode(
            head_outputs,
            strand=strand,
            input_ids=input_ids,
        )

        # 2. Return the Prediction object, which handles all downstream calculations
        return Prediction(isoforms=candidate_isoforms, _nmd_model=self.nmd_model)

    @classmethod
    def from_config_file(cls, config_path: str):
        """Load model from a YAML config file."""
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        d_in = config.get("encoder", {}).get("hidden_size")
        if d_in is None:
            raise ValueError("`encoder.hidden_size` must be specified in the config.")

        return cls(d_in=d_in, config=config)

def preprocess_sequence(chrom: str, start: int, end: int) -> str:
    """
    Fetch and normalize a genomic window. Placeholder.
    Replace with FASTA-backed retrieval (pyfaidx/pysam) and uppercase normalization.
    """
    # TODO: implement FASTA retrieval
    return "N" * (end - start)

def preprocess_variant(vcf_like: str, window: Any = None) -> Dict[str, Any]:
    """
    Convert a simple variant spec (e.g., '17:43051000A>T') into a structured encoding
    suitable for the variant channel.
    """
    # TODO: parse and align to window coords
    return {"spec": vcf_like}
<<<END:src/betadogma/model.py>>>

<<<FILE:src/betadogma/nmd/__init__.py>>>
"""Hybrid NMD prediction: rules + learned model."""
<<<END:src/betadogma/nmd/__init__.py>>>

<<<FILE:src/betadogma/nmd/nmd_model.py>>>
"""
Learned NMD classifier using features derived from transcript structure.
This version uses a simple rule-based approach as a baseline.
"""
from typing import Dict, Any

from ..decoder.types import Isoform
from .nmd_rules import rule_ptc_before_last_junction


class NMDModel:
    """
    A simple rule-based model for predicting nonsense-mediated decay (NMD).
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Allow overriding the NMD rule threshold from the main model config
        self.ptc_rule_threshold = self.config.get("ptc_rule_threshold", 55)

    def predict(self, isoform: Isoform) -> float:
        """
        Predicts the probability of an isoform undergoing NMD.

        For this rule-based model, the probability is binary (0.0 or 1.0).

        Args:
            isoform: The isoform to be evaluated.

        Returns:
            A probability (float) indicating the likelihood of NMD.
        """
        is_nmd_target = rule_ptc_before_last_junction(
            isoform,
            ptc_rule_threshold=self.ptc_rule_threshold
        )
        return 1.0 if is_nmd_target else 0.0
<<<END:src/betadogma/nmd/nmd_model.py>>>

<<<FILE:src/betadogma/nmd/nmd_rules.py>>>
"""
Rule-based NMD heuristics (55-nt rule, last-exon exceptions, etc.).
"""
from ..decoder.types import Isoform


def rule_ptc_before_last_junction(isoform: Isoform, ptc_rule_threshold: int = 55) -> bool:
    """
    Return True if PTC > `ptc_rule_threshold` nt upstream of the last exon junction.

    This function implements the classic "55-nucleotide rule" for nonsense-mediated
    decay (NMD). An isoform is predicted to be an NMD target if its stop codon
    (premature termination codon or PTC) is located more than a certain distance
    upstream of the final exon-exon junction.

    Args:
        isoform: The isoform object to check. It must have its `cds` attribute
                 populated with the genomic coordinates of the coding sequence.
                 The `exons` attribute must be in transcription order.
        ptc_rule_threshold: The distance threshold in nucleotides.

    Returns:
        True if the isoform is predicted to be an NMD target, False otherwise.
    """
    # Rule only applies to spliced transcripts (more than one exon)
    if len(isoform.exons) < 2:
        return False

    # Rule requires a defined CDS to locate the PTC
    if isoform.cds is None:
        return False

    # 1. Find the transcript coordinate of the last exon-exon junction.
    # `isoform.exons` is in transcription order. The junction is after the second-to-last exon.
    last_junction_tx_coord = sum(e.end - e.start for e in isoform.exons[:-1])

    # 2. Find the transcript coordinate of the PTC (start of the stop codon).
    if isoform.strand == '+':
        # For plus strand, the stop codon is at the end of the CDS.
        # cds format is (start, end) of the entire CDS region.
        # The stop codon starts at end - 3.
        ptc_genomic_coord = isoform.cds[1] - 3
    else:  # strand == '-'
        # For minus strand, the stop codon is at the start of the CDS in genomic terms.
        ptc_genomic_coord = isoform.cds[0]

    ptc_tx_coord = -1
    len_before_exon = 0
    for exon in isoform.exons:  # Exons are in transcription order
        if exon.start <= ptc_genomic_coord < exon.end:
            # Found the exon containing the PTC
            if isoform.strand == '+':
                offset_in_exon = ptc_genomic_coord - exon.start
            else:  # strand == '-'
                # Transcription is right-to-left. The coordinate is relative to the exon's 3' end.
                offset_in_exon = exon.end - 1 - ptc_genomic_coord
            ptc_tx_coord = len_before_exon + offset_in_exon
            break
        len_before_exon += (exon.end - exon.start)

    # If PTC wasn't found in any exon, something is wrong.
    if ptc_tx_coord == -1:
        return False

    # 3. Calculate the distance and check the rule.
    # The distance is from the PTC to the junction.
    distance_from_ptc_to_junction = last_junction_tx_coord - ptc_tx_coord

    # A positive distance means the PTC is upstream of the junction.
    return distance_from_ptc_to_junction > ptc_rule_threshold
<<<END:src/betadogma/nmd/nmd_rules.py>>>

<<<FILE:src/betadogma/variant/__init__.py>>>
"""Variant encoding and synthetic mutagenesis."""
<<<END:src/betadogma/variant/__init__.py>>>

<<<FILE:src/betadogma/variant/encode.py>>>
"""
Encode variants (SNP/indel) into an auxiliary input channel aligned with the sequence window.
"""
from typing import Any, Dict

def encode_variant(spec: str, window=None) -> Dict[str, Any]:
    """Parse 'chr:posREF>ALT' into structured format and alignment. Placeholder."""
    return {"spec": spec}
<<<END:src/betadogma/variant/encode.py>>>

<<<FILE:src/betadogma/variant/simulate.py>>>
"""
Synthetic mutagenesis utilities for training variant sensitivity:
- Create PTCs
- Motif edits (splice sites, Kozak)
- PolyA disruptions
"""
def make_ptc_variant(*args, **kwargs):
    pass
<<<END:src/betadogma/variant/simulate.py>>>

<<<FILE:tests/core/test_heads.py>>>
import pytest
import torch
from betadogma.core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead
from betadogma.model import BetaDogmaModel

# Test parameters updated for Enformer-based model
BATCH_SIZE = 2
INPUT_SEQ_LEN = 196_608  # A typical Enformer input length
BINNED_SEQ_LEN = 896     # Enformer's binned output resolution
D_IN = 3072              # Enformer's embedding dimension
D_HIDDEN = 768

@pytest.fixture
def dummy_embeddings():
    """Create a dummy embeddings tensor with Enformer-like dimensions."""
    return torch.randn(BATCH_SIZE, BINNED_SEQ_LEN, D_IN)

def test_splice_head_shapes(dummy_embeddings):
    """Test the output shapes of the SpliceHead."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "donor" in output
    assert "acceptor" in output
    assert output["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["acceptor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_tss_head_shapes(dummy_embeddings):
    """Test the output shapes of the TSSHead."""
    head = TSSHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "tss" in output
    assert output["tss"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_polya_head_shapes(dummy_embeddings):
    """Test the output shapes of the PolyAHead."""
    head = PolyAHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "polya" in output
    assert output["polya"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_orf_head_shapes(dummy_embeddings):
    """Test the output shapes of the ORFHead."""
    head = ORFHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "start" in output
    assert "stop" in output
    assert "frame" in output
    assert output["start"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["stop"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["frame"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 3)

@pytest.mark.parametrize("use_conv", [True, False])
def test_head_architectures(dummy_embeddings, use_conv):
    """Test both convolutional and linear head architectures."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN, use_conv=use_conv)
    output = head(dummy_embeddings)
    assert output["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_betadogma_model_forward_pass_shapes():
    """Tests the end-to-end forward pass of the BetaDogmaModel with mock embeddings."""

    # 1. Instantiate the model using a dummy config.
    # The model is now decoupled from the encoder.
    config = {
        "heads": {"hidden": D_HIDDEN, "dropout": 0.1, "use_conv": False},
        "decoder": {}  # Not used in forward pass
    }
    model = BetaDogmaModel(d_in=D_IN, config=config)
    model.eval()

    # 2. Create dummy embeddings. The model is agnostic to sequence length.
    dummy_embeddings = torch.randn(BATCH_SIZE, BINNED_SEQ_LEN, D_IN)

    # 3. Run forward pass
    with torch.no_grad():
        output = model(embeddings=dummy_embeddings)

    # 4. Assert output shapes
    assert "splice" in output
    assert "tss" in output
    assert "polya" in output
    assert "orf" in output
    assert "embeddings" in output

    assert torch.equal(output["embeddings"], dummy_embeddings)
    assert output["splice"]["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["splice"]["acceptor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["tss"]["tss"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["polya"]["polya"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["start"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["stop"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["frame"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 3)
<<<END:tests/core/test_heads.py>>>

<<<FILE:tests/decoder/test_decoder_logic.py>>>
"""
Regression tests for the isoform decoder's internal logic, focusing on
strand-awareness, ORF scoring, and other complex behaviors.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import (
    _get_spliced_cDNA,
    _score_orf,
    SpliceGraphBuilder,
)
from betadogma.decoder.types import Exon, Isoform

# --- Test Fixtures ---

@pytest.fixture
def scoring_config():
    """Provides a standard configuration for the ORF scorer."""
    return {
        "use_orf_head": False,  # Force sequence-based fallback for these tests
        "min_cds_len_aa": 10,
        "max_cds_len_aa": 1000,
        "kozak_bonus": 0.2,
        "orf_gamma": 0.6,  # Penalty for PTC
    }

@pytest.fixture
def mock_head_outputs():
    """A mock output from the model's prediction heads."""
    return {
        "splice": {"donor": torch.randn(2000), "acceptor": torch.randn(2000)},
        "orf": {"start": torch.randn(2000), "stop": torch.randn(2000), "frame": torch.randn(2000, 3)},
    }

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components, shared across test files."""
    return {
        "decoder": {
            "max_candidates": 64, "beam_size": 16,
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10, "max_intron_len": 10000},
            "max_starts": 8, "max_ends": 8,
        }
    }

# A map for converting token indices to DNA characters
TOKEN_MAP = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


# --- Tests for _get_spliced_cDNA ---

def test_get_spliced_cDNA_positive_strand():
    """Tests cDNA construction for a simple positive-strand transcript."""
    # Sequence: ACGTACGTACGTACGTACGT
    # Exons:    [10:12], [16:18] -> "GT", "AC"
    # Expected: GTAC
    input_ids = torch.tensor([[
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    ]]) # "ACGTACGTACGTACGTACGT"
    isoform = Isoform(exons=[Exon(10, 12), Exon(16, 18)], strand="+")

    cDNA, _ = _get_spliced_cDNA(isoform, input_ids, TOKEN_MAP)
    assert cDNA == "GTAC"

def test_get_spliced_cDNA_negative_strand():
    """Tests that cDNA is correctly reverse-complemented for a negative-strand transcript."""
    # Sequence: ACGTACGTACGTACGTACGT
    # Exons:    [10:12], [16:18] -> "GT", "AC"
    # Spliced:  GTAC
    # RevComp:  GTAC
    input_ids = torch.tensor([[
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    ]]) # "ACGTACGTACGTACGTACGT"
    isoform = Isoform(exons=[Exon(10, 12), Exon(16, 18)], strand="-")

    cDNA, _ = _get_spliced_cDNA(isoform, input_ids, TOKEN_MAP)
    assert cDNA == "GTAC"


# --- Tests for _score_orf (Sequence-based Fallback) ---

def _dna_to_tensor(dna_str: str) -> torch.Tensor:
    """Helper to convert a DNA string to a batched tensor of token indices."""
    rev_token_map = {v: k for k, v in TOKEN_MAP.items()}
    return torch.tensor([[rev_token_map.get(c, 4) for c in dna_str]]).long()


def test_score_orf_sequence_valid(scoring_config, mock_head_outputs):
    """Tests scoring for a standard, valid ORF that meets length requirements."""
    # ORF is 12 codons long, should pass min_cds_len_aa=10 check
    orf = "ATG" + "GGCGGCGGCG" * 3 + "TAA"
    cDNA = "GCC" + orf + "GGC"
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score >= 0.8  # Should get base score (0.5) + length bonus (0.3)

def test_score_orf_ptc_penalty(scoring_config, mock_head_outputs):
    """Tests that a premature termination codon (PTC) is correctly penalized."""
    # Junction is at pos 62. Stop codon TAA at pos 6 is > 55bp upstream.
    exon1_seq = "GCCATGTAA" + "G" * 53 # len=62, stop at pos 6
    exon2_seq = "GCGGCTAGGGC"
    cDNA = exon1_seq + exon2_seq
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, 62), Exon(62, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score < 0.3 # Should be base score (0.5) - PTC penalty (0.6) -> 0

def test_score_orf_kozak_bonus(scoring_config, mock_head_outputs):
    """Tests that a strong Kozak sequence gets a bonus."""
    # An ORF long enough to pass the length filter (15 codons)
    long_orf_codons = "GGC" * 15

    # Strong Kozak: A at -3, G at +4. The G at +4 is the first base of the next codon.
    cDNA_strong = "GCCAGCATG" + long_orf_codons + "TAA"
    input_ids_strong = _dna_to_tensor(cDNA_strong)
    isoform_strong = Isoform(exons=[Exon(0, len(cDNA_strong))], strand="+")

    # Weak Kozak: T at -3
    cDNA_weak = "GCCTCCATG" + long_orf_codons + "TAA"
    input_ids_weak = _dna_to_tensor(cDNA_weak)
    isoform_weak = Isoform(exons=[Exon(0, len(cDNA_weak))], strand="+")

    score_strong = _score_orf(isoform_strong, mock_head_outputs, scoring_config, input_ids_strong)
    score_weak = _score_orf(isoform_weak, mock_head_outputs, scoring_config, input_ids_weak)

    # Strong score should be base (0.5) + length (0.3) + kozak (0.2) = 1.0
    # Weak score should be base (0.5) + length (0.3) = 0.8
    assert score_strong > score_weak
    assert score_strong.item() == pytest.approx(1.0)
    assert score_weak.item() == pytest.approx(0.8)

def test_score_orf_no_valid_orf(scoring_config, mock_head_outputs):
    """Tests that a sequence with no valid ORF gets a score of 0."""
    cDNA = "GCCTTTGGCCGGCGC" # No ATG or no stop
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score == 0.0

# --- Test for SpliceGraphBuilder ---

def test_splice_graph_builder_negative_strand(decoder_config):
    """
    Tests that the SpliceGraphBuilder correctly connects exons
    for a negative-strand transcript.
    """
    # Mock model outputs
    # Donor at 800, Acceptor at 500 (transcriptional order)
    # TSS at 900, PolyA at 400
    # Exon 1: 800-900 (Donor to TSS), Exon 2: 400-500 (PolyA to Acceptor)
    # Expected junction: from Exon 1 to Exon 2
    head_outputs = {
        "splice": {
            "donor": torch.full((1000,), -10.0),
            "acceptor": torch.full((1000,), -10.0),
        },
        "tss": {"tss": torch.full((1000,), -10.0)},
        "polya": {"polya": torch.full((1000,), -10.0)},
    }
    head_outputs["splice"]["donor"][800] = 5.0
    head_outputs["splice"]["acceptor"][500] = 5.0
    head_outputs["tss"]["tss"][900] = 5.0
    head_outputs["polya"]["polya"][400] = 5.0

    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(head_outputs, strand="-")

    # Check that the two exons were created as nodes
    # Note: on negative strand, a "first" exon is (donor, tss)
    # and a "last" exon is (polya, acceptor)
    assert (800, 900) in graph.graph.nodes
    assert (400, 500) in graph.graph.nodes

    # Check that a directed edge exists from the upstream to downstream exon
    assert graph.graph.has_edge((800, 900), (400, 500))
    # Verify no edge in the wrong direction
    assert not graph.graph.has_edge((400, 500), (800, 900))
<<<END:tests/decoder/test_decoder_logic.py>>>

<<<FILE:tests/decoder/test_isoform_decoder.py>>>
"""
Unit tests for the isoform decoder scaffolding.
"""
import pytest

from betadogma.decoder.isoform_decoder import (
    SpliceGraphBuilder,
    IsoformEnumerator,
    IsoformScorer,
    IsoformDecoder,
)
from betadogma.decoder.types import Exon, Isoform

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components."""
    return {
        "decoder": {
            "max_candidates": 64,
            "beam_size": 16,
            "thresholds": {
                "donor": 0.6,
                "acceptor": 0.6,
                "tss": 0.5,
                "polya": 0.5,
            },
        }
    }

def test_exon_creation():
    """Tests that the Exon data class can be created."""
    exon = Exon(start=100, end=200, score=0.9)
    assert exon.start == 100
    assert exon.end == 200
    assert exon.score == 0.9

def test_isoform_creation():
    """Tests that the Isoform data class can be created."""
    exon1 = Exon(start=100, end=200)
    exon2 = Exon(start=300, end=400)
    isoform = Isoform(exons=[exon1, exon2], strand="+", score=0.85)
    assert len(isoform.exons) == 2
    assert isoform.strand == "+"
    assert isoform.start == 100
    assert isoform.end == 400

def test_splice_graph_builder_init(decoder_config):
    """Tests that SpliceGraphBuilder can be instantiated."""
    builder = SpliceGraphBuilder(config=decoder_config["decoder"])
    assert builder is not None
    assert builder.config == decoder_config["decoder"]

def test_isoform_enumerator_init(decoder_config):
    """Tests that IsoformEnumerator can be instantiated."""
    enumerator = IsoformEnumerator(config=decoder_config["decoder"])
    assert enumerator is not None
    assert enumerator.config == decoder_config["decoder"]

def test_isoform_scorer_init(decoder_config):
    """Tests that IsoformScorer can be instantiated."""
    scorer = IsoformScorer(config=decoder_config["decoder"])
    assert scorer is not None
    assert scorer.config == decoder_config["decoder"]

def test_isoform_decoder_init(decoder_config):
    """Tests that the main IsoformDecoder can be instantiated."""
    decoder = IsoformDecoder(config=decoder_config)
    assert decoder is not None
    assert isinstance(decoder.graph_builder, SpliceGraphBuilder)
    assert isinstance(decoder.enumerator, IsoformEnumerator)
    assert isinstance(decoder.scorer, IsoformScorer)
<<<END:tests/decoder/test_isoform_decoder.py>>>

<<<FILE:tests/decoder/test_isoform_decoder_minus_strand.py>>>
import pytest
import torch
from dataclasses import dataclass

from betadogma.decoder.isoform_decoder import (
    IsoformDecoder,
    SpliceGraphBuilder,
    _score_orf,
    _get_spliced_cDNA,
)
from betadogma.decoder.types import Exon, Isoform

@dataclass
class CdsWindow:
    start: int
    end: int
    strand: str

def toy_minus_gene():
    # Genome coords: higher -> lower is transcription
    # Exon B (upstream in transcription) has higher coord than Exon A.
    exons = [
        Exon(start=100, end=120, score=0.9),  # A (3' in transcript)
        Exon(start=150, end=170, score=0.9),  # B (5' in transcript)
    ]
    strand = "-"
    # CDS covers from 155..115 on the genome (reverse-complement semantics)
    cds = CdsWindow(start=115, end=155, strand=strand)
    return exons, strand, cds

def mock_head_outputs(device="cpu"):
    """Mocks head outputs for a ~200bp sequence."""
    seq_len = 200
    outputs = {
        "splice": {
            "donor": torch.full((seq_len,), -10.0, device=device),
            "acceptor": torch.full((seq_len,), -10.0, device=device),
        },
        "tss": {"tss": torch.full((seq_len,), -10.0, device=device)},
        "polya": {"polya": torch.full((seq_len,), -10.0, device=device)},
        "orf": {
            "start": torch.full((seq_len,), -10.0, device=device),
            "stop": torch.full((seq_len,), -10.0, device=device),
            "frame": torch.full((seq_len, 3), -10.0, device=device),
        }
    }
    # From toy_minus_gene, transcription order is B -> A
    # Donor must be at start of B, Acceptor at end of A (genomic coordinates)
    # TSS at end of B, PolyA at start of A
    outputs["splice"]["donor"][150] = 5.0 # up_exon.start on minus strand
    outputs["splice"]["acceptor"][120] = 5.0 # down_exon.end on minus strand
    outputs["tss"]["tss"][170] = 5.0 # "end" of 5' exon
    outputs["polya"]["polya"][100] = 5.0 # "start" of 3' exon
    return outputs


def test_minus_strand_splice_edge_directionality():
    """
    Tests that splice graph edges on the minus strand are built
    in the direction of transcription (from higher to lower genomic coordinates).
    """
    _, strand, _ = toy_minus_gene()
    head_outputs = mock_head_outputs()

    # Use a dummy config for the builder
    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10},
            "allow_unanchored": True
        }
    }
    builder = SpliceGraphBuilder(config)
    graph = builder.build(head_outputs, strand=strand)

    # On '-', transcription flows high->low; edges must reflect that.
    # Exon B (150, 170) should connect to Exon A (100, 120)

    # The builder creates multiple candidate exons. Let's find the ones we expect.
    # First exon: donor -> tss = (150, 170)
    # Last exon: polya -> acceptor = (100, 120)
    # Internal exon: donor -> acceptor = (150, 120) -> this is wrong, should be donor -> acceptor

    # Correct internal exon on minus strand is (donor_pos, acceptor_pos) if donor_pos > acceptor_pos
    # The builder logic is `_get_exons(donor_indices, acceptor_indices, ...)`, which requires start < end.
    # This is the first bug. Let's check the junction logic given the exons it *does* find.

    # It will find a "first" exon (150, 170) and a "last" exon (100, 120)
    # The junction logic sorts by start coordinate (reversed for minus strand).
    # So it will check for a junction from (150, 170) to (100, 120)
    # Junction check on minus strand: `up_exon.start in donor_set and down_exon.end in acceptor_set`
    # Here, up_exon.start=150 (donor), down_exon.end=120 (acceptor). This should work.

    expected_edge = ((150, 170), (100, 120))

    assert expected_edge in graph.graph.edges(), "Edge from high-coordinate exon to low-coordinate exon not found for minus strand."
    assert len(graph.graph.edges()) == 1, "Incorrect number of edges found."


def test_minus_strand_full_decoder_ordering():
    """
    Tests that the full IsoformDecoder correctly orders exons
    for a minus-strand transcript.
    """
    _, strand, _ = toy_minus_gene()
    head_outputs = mock_head_outputs()

    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10},
            "beam_size": 2,
            "scoring": {
                "w_spl": 1.0, "w_tss": 0.0, "w_pa": 0.0, "w_orf": 0.0, "w_len": 0.0,
                "use_orf_head": False
            }
        }
    }
    decoder = IsoformDecoder(config)
    isoforms = decoder.decode(head_outputs, strand=strand)

    assert len(isoforms) > 0, "Decoder failed to produce any isoforms."

    # The top-scoring isoform should have exons ordered by transcription
    # For minus strand, this is high genomic coordinate to low.
    best_isoform = isoforms[0]
    ordered_coords = [(e.start, e.end) for e in best_isoform.exons]

    expected_order = [(150, 170), (100, 120)]

    assert ordered_coords == expected_order, f"Exons not in transcription order. Expected {expected_order}, got {ordered_coords}"


def test_minus_strand_orf_roles_via_scoring():
    """
    Tests that ORF roles (start/stop) are correctly identified on the minus strand,
    verified by checking the ORF score.
    """
    # Sequence: "AAA" (exon 2) "ATG" (intron) "TAG" (exon 1) "AAA"
    # RevComp:  "TTT" "CTA" (revcomp of TAG) "CAT" (revcomp of ATG) "TTT"
    # Exon 1 (150-153): TAG -> revcomp CTA (stop)
    # Exon 2 (100-103): AAA -> revcomp TTT
    # We expect the ORF to be read from a start codon on the revcomp sequence.
    # Let's try a different sequence.
    # Genomic:      ...CCC TGA ... ATG GGG...
    # Coords:           ^100    ^150
    # Exon 1 (150-156): ATG GGG -> revcomp CCC CAT
    # Exon 2 (100-103): TGA     -> revcomp TCA (stop)
    # Transcript:   CCC CAT TCA
    # This transcript has a start (CAT) and a stop (TCA).

    # Let's use a token map where A=0, C=1, G=2, T=3
    # Sequence: "AAACCC" (pos 100-105, exon 2) ... "ATGGGG" (pos 150-155, exon 1)
    input_ids = torch.tensor([[
        0]*100 + [3, 2, 0] + [0]*44 + [0, 3, 2, 2, 2, 2] + [0]*45
    ]) # Puts TGA at 100, ATGGGG at 150

    head_outputs = mock_head_outputs() # Uses peaks at 100,120,150,170
    # Let's adjust head outputs to match our sequence
    head_outputs["splice"]["donor"][150] = 5.0 # Exon 1 start
    head_outputs["splice"]["acceptor"][103] = 5.0 # Exon 2 end
    head_outputs["tss"]["tss"][156] = 5.0
    head_outputs["polya"]["polya"][100] = 5.0

    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 3},
            "beam_size": 2,
            "scoring": {
                "w_spl": 0.1, "w_tss": 0.1, "w_pa": 0.1, "w_orf": 1.0, "w_len": 0.0,
                "use_orf_head": False, # Use sequence-based scorer
                "min_cds_len_aa": 1,
                "orf_gamma": 0.0 # No PTC penalty for this test
            }
        }
    }
    decoder = IsoformDecoder(config)
    # We need to manually tell the builder about the exons we want it to find
    # because the builder logic is complex. We test the scorer here.
    exons = [Exon(start=150, end=156, score=0.9), Exon(start=100, end=103, score=0.9)]
    isoform = Isoform(exons=exons, strand="-")

    # The scorer needs the full head_outputs dict, even if it only uses a part of it.
    score = decoder.scorer(isoform, head_outputs, input_ids=input_ids)

    # A positive score indicates a valid ORF was found.
    assert score > 0, f"Valid ORF on minus strand not detected, score was {score}"


def test_minus_strand_single_exon_cds():
    """Tests scoring for a single-exon CDS on the minus strand."""
    # For a minus strand transcript to have an ATG, the genomic sequence must have a CAT at a
    # higher coordinate than the stop codon's reverse complement (e.g., TTA).
    # The distance between them must be a multiple of 3 for them to be in-frame.
    # Genomic: ... TTA ...... CAT ... (18bp between them)
    # RevComp: ... ATG ...... TAA ... -> Correct ORF
    input_ids = torch.tensor([[0]*50 + [3,3,0] + [0]*18 + [1,0,3] + [0]*27]) # TTA at 50, CAT at 71

    config = { "decoder": { "scoring": { "use_orf_head": False, "min_cds_len_aa": 5 } } }
    isoform = Isoform(exons=[Exon(start=40, end=80)], strand="-")

    head_outputs = {"splice": {"donor": torch.tensor([])}}
    score = _score_orf(isoform, head_outputs, config['decoder']['scoring'], input_ids=input_ids)

    assert score > 0.5, "Valid single-exon CDS on minus strand not scored correctly"


def test_minus_strand_utr_only():
    """Tests that a UTR-only transcript on the minus strand gets a low ORF score."""
    # Sequence with no ATG
    input_ids = torch.tensor([[1,2,3]*50])
    config = { "decoder": { "scoring": { "use_orf_head": False } } }
    isoform = Isoform(exons=[Exon(start=10, end=50), Exon(start=60, end=100)], strand="-")

    head_outputs = {"splice": {"donor": torch.tensor([])}}
    score = _score_orf(isoform, head_outputs, config['decoder']['scoring'], input_ids=input_ids)

    assert score <= 0.0, "UTR-only transcript should not have a positive ORF score"
<<<END:tests/decoder/test_isoform_decoder_minus_strand.py>>>

<<<FILE:tests/decoder/test_isoform_enumerator.py>>>
"""
Unit tests for the IsoformEnumerator and IsoformScorer.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import (
    SpliceGraph,
    IsoformEnumerator,
    IsoformScorer,
)
from betadogma.decoder.types import Exon, Isoform

@pytest.fixture
def simple_splice_graph():
    """
    Creates a simple splice graph for testing path enumeration.
    Graph structure:
    A -> B -> D (score: 0.9 + 0.8 + 0.9 = 2.6) -> path score ~0.87
    A -> C -> D (score: 0.9 + 0.7 + 0.9 = 2.5) -> path score ~0.83
    A is source, D is sink.
    """
    graph = SpliceGraph()
    exon_a = Exon(start=10, end=20, score=0.9)
    exon_b = Exon(start=30, end=40, score=0.8)
    exon_c = Exon(start=30, end=45, score=0.7) # Alternative exon
    exon_d = Exon(start=50, end=60, score=0.9)

    graph.add_exon(exon_a)
    graph.add_exon(exon_b)
    graph.add_exon(exon_c)
    graph.add_exon(exon_d)

    graph.add_junction(exon_a, exon_b, score=1.0)
    graph.add_junction(exon_a, exon_c, score=1.0)
    graph.add_junction(exon_b, exon_d, score=1.0)
    graph.add_junction(exon_c, exon_d, score=1.0)

    return graph

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components."""
    return {
        "decoder": {
            "beam_size": 4,
            "max_candidates": 10,
            "scoring": {
                "w_spl": 1.0,
                "orf_alpha": 0.5,
                "orf_beta": 0.3,
                "orf_gamma": 0.6,
                "max_start_candidates": 5,
            }
        }
    }

def test_isoform_enumerator_finds_best_path(simple_splice_graph, decoder_config):
    """
    Tests that the beam search enumerator finds the highest-scoring path.
    The new enumerator also returns partial paths, so we expect more candidates.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    isoforms = enumerator.enumerate(simple_splice_graph, max_paths=10)

    # The new enumerator finds all sub-paths as candidates.
    # We should find 4 single-exon paths and 2 two-exon paths.
    assert len(isoforms) >= 2

    # The best path is now based on score normalized by length.
    # Path A->B->D: (0.9+0.8+0.9)/3 = 0.867
    # Path A->C->D: (0.9+0.7+0.9)/3 = 0.833
    # Single exons A and D have score 0.9.
    # So the top-ranked isoforms should be the single exons.
    best_isoform = isoforms[0]
    assert len(best_isoform.exons) == 1
    assert best_isoform.exons[0].score == 0.9

    # Find the best multi-exon path to verify ranking among them
    multi_exon_isoforms = [iso for iso in isoforms if len(iso.exons) > 1]
    best_multi_exon = multi_exon_isoforms[0]
    assert best_multi_exon.exons[0].end == 20  # Exon A
    assert best_multi_exon.exons[1].end == 40  # Exon B

def test_isoform_enumerator_respects_max_paths(simple_splice_graph, decoder_config):
    """
    Tests that the enumerator returns the correct number of paths.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    isoforms = enumerator.enumerate(simple_splice_graph, max_paths=1)

    assert len(isoforms) == 1
    # Check it's the best one (a single-exon isoform with score 0.9)
    assert len(isoforms[0].exons) == 1
    assert isoforms[0].exons[0].score == 0.9


def test_isoform_enumerator_empty_graph(decoder_config):
    """
    Tests that the enumerator handles an empty graph gracefully.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    empty_graph = SpliceGraph()
    isoforms = enumerator.enumerate(empty_graph, max_paths=10)
    assert len(isoforms) == 0
<<<END:tests/decoder/test_isoform_enumerator.py>>>

<<<FILE:tests/decoder/test_splice_graph_builder.py>>>
"""
Unit tests for the SpliceGraphBuilder.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import _find_peaks, SpliceGraphBuilder

def test_find_peaks_basic():
    """Tests the _find_peaks helper function with a simple case."""
    # Probs: [0.1, 0.9, 0.2, 0.8, 0.4] -> sigmoid of below
    logits = torch.tensor([-2.2, 2.2, -1.4, 1.4, -0.4])
    threshold = 0.7

    peak_indices, peak_probs = _find_peaks(logits, threshold)

    assert torch.equal(peak_indices, torch.tensor([1, 3]))
    assert torch.allclose(peak_probs, torch.tensor([0.9, 0.8]), atol=1e-2)

def test_find_peaks_top_k():
    """Tests that _find_peaks correctly applies the top_k limit."""
    # Probs: [0.9, 0.8, 0.95, 0.75]
    logits = torch.tensor([2.2, 1.4, 2.9, 1.1])
    threshold = 0.7

    peak_indices, peak_probs = _find_peaks(logits, threshold, top_k=2)

    # Should return indices of 0.95 and 0.9
    assert set(peak_indices.tolist()) == {0, 2}
    assert torch.allclose(peak_probs.sort(descending=True).values, torch.tensor([0.95, 0.9]), atol=1e-2)

def test_find_peaks_no_peaks():
    """Tests that _find_peaks returns empty tensors when no peaks are found."""
    logits = torch.tensor([-2.0, -1.0, -3.0])
    threshold = 0.9

    peak_indices, peak_probs = _find_peaks(logits, threshold)

    assert peak_indices.numel() == 0
    assert peak_probs.numel() == 0

@pytest.fixture
def sample_head_outputs():
    """Provides a sample dictionary of head outputs for testing."""
    seq_len = 100
    donor_logits = torch.full((1, seq_len, 1), -10.0)
    acceptor_logits = torch.full((1, seq_len, 1), -10.0)
    tss_logits = torch.full((1, seq_len, 1), -10.0)
    polya_logits = torch.full((1, seq_len, 1), -10.0)

    # Donor peaks at 20, 40
    donor_logits[0, 20, 0] = 3.0
    donor_logits[0, 40, 0] = 2.0
    # Acceptor peaks at 10, 30
    acceptor_logits[0, 10, 0] = 3.0
    acceptor_logits[0, 30, 0] = 2.0
    # TSS peak at 5
    tss_logits[0, 5, 0] = 3.0
    # PolyA peak at 50
    polya_logits[0, 50, 0] = 3.0


    return {
        "splice": {
            "donor": donor_logits,
            "acceptor": acceptor_logits,
        },
        "tss": {"tss": tss_logits},
        "polya": {"polya": polya_logits},
    }

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components, allowing unanchored exons."""
    return {
        "decoder": {
            "max_starts": 8,
            "max_ends": 8,
            "allow_unanchored": True,
            "thresholds": {
                "donor": 0.8,
                "acceptor": 0.8,
                "tss": 0.8,
                "polya": 0.8,
            },
            "priors": {
                "min_exon_len": 5,
                "max_intron_len": 100,
            }
        }
    }

def test_splice_graph_builder_positive_strand(sample_head_outputs, decoder_config):
    """Tests that the SpliceGraphBuilder can build a graph on the positive strand."""
    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(sample_head_outputs, strand='+')

    assert graph is not None

    # Expected exons:
    # Internal: (10, 20), (10, 40), (30, 40) -> 3
    # First (TSS->donor): (5, 20), (5, 40) -> 2
    # Last (acceptor->polyA): (10, 50), (30, 50) -> 2
    # Single (TSS->polyA): (5, 50) -> 1
    # Total = 8
    assert graph.graph.number_of_nodes() == 8

    nodes = set(graph.graph.nodes)
    expected_nodes = {(10, 20), (10, 40), (30, 40), (5, 20), (5, 40), (10, 50), (30, 50), (5, 50)}
    assert nodes == expected_nodes

    # Expected junctions (intron len < 100):
    # (10,20) -> (30,40) [intron=10]
    # (5,20) -> (30,40) [intron=10]
    # (5,20) -> (30,50) [intron=10]
    # (10,20) -> (30,50) [intron=10]
    assert graph.graph.number_of_edges() == 4
    assert graph.graph.has_edge((10, 20), (30, 40))


def test_splice_graph_builder_negative_strand(decoder_config):
    """Tests graph construction on the negative strand with appropriate fixtures."""
    # On negative strand, donor < acceptor from a coordinate perspective.
    # A transcript flows from a high coordinate (TSS) to a low one (polyA).
    # A "first" exon is donor -> tss, "last" is polya -> acceptor.
    seq_len = 100
    donor_logits = torch.full((1, seq_len, 1), -10.0)
    acceptor_logits = torch.full((1, seq_len, 1), -10.0)
    tss_logits = torch.full((1, seq_len, 1), -10.0)
    polya_logits = torch.full((1, seq_len, 1), -10.0)

    # Donor peaks at 10, 30
    donor_logits[0, 10, 0] = 3.0
    donor_logits[0, 30, 0] = 2.0
    # Acceptor peaks at 20, 40
    acceptor_logits[0, 20, 0] = 3.0
    acceptor_logits[0, 40, 0] = 2.0
    # TSS at 50, polyA at 5
    tss_logits[0, 50, 0] = 3.0
    polya_logits[0, 5, 0] = 3.0

    head_outputs = {
        "splice": {"donor": donor_logits, "acceptor": acceptor_logits},
        "tss": {"tss": tss_logits},
        "polya": {"polya": polya_logits},
    }

    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(head_outputs, strand='-')

    # Expected exons:
    # Internal (donor->acceptor): (10,20), (10,40), (30,40) -> 3
    # First (donor->tss): (10,50), (30,50) -> 2
    # Last (polya->acceptor): (5,20), (5,40) -> 2
    # Single (polya->tss): (5,50) -> 1
    # Total = 8
    assert graph.graph.number_of_nodes() == 8, "Should find 8 unique exons"
    nodes = set(graph.graph.nodes)
    expected_nodes = {(10, 20), (10, 40), (30, 40), (10, 50), (30, 50), (5, 20), (5, 40), (5, 50)}
    assert nodes == expected_nodes

    # Expected junctions on negative strand (from upstream exon to downstream exon)
    # e.g. from (30,40) to (10,20). Intron len = 30-20 = 10.
    assert graph.graph.has_edge((30, 40), (10, 20)), "Junction from (30,40) to (10,20) missing"
    assert graph.graph.has_edge((30, 50), (10, 20)), "Junction from (30,50) to (10,20) missing"
    assert graph.graph.has_edge((30, 50), (5, 20)), "Junction from (30,50) to (5,20) missing"
    assert graph.graph.number_of_edges() > 0, "No junctions were found"


def test_splice_graph_builder_negative_strand_regression(decoder_config):
    """
    A specific regression test for the negative strand logic fix.
    It checks a simple two-exon case where sorting and junction-finding
    are critical.
    """
    # Locus (negative strand):
    # high coord <--- TSS at 150
    #              [Exon 1: 100-120] -> donor=100, acceptor=120
    #                 intron (len=30)
    #              [Exon 2: 50-70]   -> donor=50, acceptor=70
    # low coord  <--- polyA at 20
    #
    # Expected transcript order: Exon 1 (100,120) -> Exon 2 (50,70)
    # Expected junction: from donor at 100 to acceptor at 70.

    seq_len = 200
    donor_logits = torch.full((1, seq_len, 1), -10.0)
    acceptor_logits = torch.full((1, seq_len, 1), -10.0)
    tss_logits = torch.full((1, seq_len, 1), -10.0)
    polya_logits = torch.full((1, seq_len, 1), -10.0)

    # For negative strand, a donor is at a lower coordinate than an acceptor for an internal exon.
    # The builder pairs (donor, acceptor) where donor < acceptor.
    donor_logits[0, 100, 0] = 5.0
    donor_logits[0, 50, 0] = 5.0
    acceptor_logits[0, 120, 0] = 5.0
    acceptor_logits[0, 70, 0] = 5.0
    tss_logits[0, 150, 0] = 5.0
    polya_logits[0, 20, 0] = 5.0

    head_outputs = {
        "splice": {"donor": donor_logits, "acceptor": acceptor_logits},
        "tss": {"tss": tss_logits},
        "polya": {"polya": polya_logits},
    }

    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(head_outputs, strand='-')

    # Expected exons on negative strand:
    # type              start           end
    # ----------------------------------------
    # internal          (donor, acc) -> (100, 120), (50, 120), (50, 70)
    # first (5')        (donor, tss) -> (100, 150), (50, 150)
    # last (3')         (polya, acc) -> (20, 120), (20, 70)
    # single            (polya, tss) -> (20, 150)
    # Total unique exons = 8
    assert graph.graph.number_of_nodes() == 8, "Incorrect number of unique exons found"

    # Critical check: the junction must connect the upstream exon to the downstream one.
    # Upstream exon is (100, 120). Downstream exon is (50, 70).
    # Junction is from donor of upstream (at its start, 100) to acceptor of downstream (at its end, 70).
    assert graph.graph.has_edge((100, 120), (50, 70)), "Junction from internal to internal exon not found."

    # Check that there is NOT a reverse edge.
    assert not graph.graph.has_edge((50, 70), (100, 120)), "Incorrect reverse junction found."

    # Check first-exon to internal-exon junction
    # Upstream: (100, 150) [first exon], Downstream: (50, 70) [internal exon]
    # Donor at 100, acceptor at 70.
    assert graph.graph.has_edge((100, 150), (50, 70)), "Junction from first-exon to internal-exon not found."

    # Check internal-exon to last-exon junction
    # Upstream: (100, 120) [internal], Downstream: (20, 70) [last exon]
    # Donor at 100, acceptor at 70.
    assert graph.graph.has_edge((100, 120), (20, 70)), "Junction from internal-exon to last-exon not found."
<<<END:tests/decoder/test_splice_graph_builder.py>>>

<<<FILE:tests/test_import.py>>>
def test_import():
    import betadogma
<<<END:tests/test_import.py>>>

<<<FILE:train/make_training_data.py>>>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict data builder for Betadogma.
- Lives in train/
- Reads YAML config (default: train/configs/data.base.yaml, or DATA_CONFIG env)
- Resolves ALL paths relative to the YAML file location
- Runs your actual modules:
    - betadogma.data.prepare_gencode (optional)
    - betadogma.data.prepare_gtex    (optional)
    - betadogma.data.prepare_data    (REQUIRED step)
- No synthetic outputs. If expected files are missing after running, it raises.
"""

import os
import sys
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------- project paths --------
THIS = Path(__file__).resolve()
TRAIN_DIR = THIS.parent
PROJECT_ROOT = TRAIN_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # allow "betadogma.*" imports

DEFAULT_CFG = TRAIN_DIR / "configs" / "data.base.yaml"


# -------- utils --------
def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve_path(p: Optional[str], base: Path) -> Optional[Path]:
    if p in (None, "", False):
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def run_module_cli(module: str, args: List[str]) -> None:
    """Run a module as a CLI: python -m module [--k v ...]"""
    cmd = [sys.executable, "-m", module] + args
    print(f"[data] CLI: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def call_module_entrypoint(module: str, kwargs: Dict[str, Any]) -> None:
    """
    Import module and call a plausible entrypoint with filtered kwargs.
    Tried in order: main, run, prepare, build.
    """
    mod = importlib.import_module(module)
    for name in ("main", "run", "prepare", "build"):
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[data] Calling {module}.{name}(**kwargs)")
            # Filter kwargs to avoid unexpected-arg errors
            try:
                sig = inspect.signature(fn)
                filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
            except (ValueError, TypeError):
                filt = kwargs
            return fn(**filt)  # type: ignore
    raise AttributeError(f"No callable entrypoint (main/run/prepare/build) in {module}.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def expect_files(paths: List[Path]) -> Tuple[bool, List[Path]]:
    missing = [p for p in paths if not p.exists()]
    return (len(missing) == 0, missing)


# -------- steps --------
def step_prepare_gencode(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    section = cfg.get("gencode", {})
    if not section or not section.get("enabled", False):
        print("[data] gencode: skipped")
        return

    kwargs = dict(section.get("kwargs") or {})
    for key in ("gencode_gtf", "fasta", "out_dir"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_gencode"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


def step_prepare_gtex(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    section = cfg.get("gtex", {})
    if not section or not section.get("enabled", False):
        print("[data] gtex: skipped")
        return

    kwargs = dict(section.get("kwargs") or {})
    for key in ("gtex_expression", "sample_table", "out_dir"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_gtex"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


def step_prepare_data(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    REQUIRED: runs betadogma.data.prepare_data which should write train/val/(test).jsonl
    """
    section = cfg.get("aggregate", {})
    if not section or not section.get("enabled", True):
        raise RuntimeError("aggregate step is required; set aggregate.enabled: true")

    kwargs = dict(section.get("kwargs") or {})
    for key in ("input_dir", "output_dir"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_data"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


# -------- main driver --------
def main():
    cfg_env = os.environ.get("DATA_CONFIG", "")
    cfg_path = Path(cfg_env) if cfg_env else DEFAULT_CFG
    if not cfg_path.is_absolute():
        cfg_path = (TRAIN_DIR / cfg_path).resolve()
    cfg = load_yaml(cfg_path)
    cfg_dir = Path(cfg["_config_dir"]).resolve()

    # 1) optional raw preprocessing
    step_prepare_gencode(cfg, cfg_dir)

    # 2) optional expression preprocessing
    step_prepare_gtex(cfg, cfg_dir)

    # 3) required aggregation to JSONL
    step_prepare_data(cfg, cfg_dir)

    # 4) verify outputs exist
    out = cfg.get("outputs", {})
    out_dir = resolve_path(out.get("dir", "../../data/processed"), cfg_dir)
    if out_dir is None:
        raise ValueError("outputs.dir must be set")

    train_name = out.get("train_file", "train.jsonl")
    val_name   = out.get("val_file", "val.jsonl")
    test_name  = out.get("test_file")  # optional

    expected = [out_dir / train_name, out_dir / val_name]
    if test_name:
        expected.append(out_dir / test_name)

    ok, missing = expect_files(expected)
    if not ok:
        miss = "\n  - ".join(str(m) for m in missing)
        raise FileNotFoundError(
            "Expected output files were not produced by prepare_data:\n"
            f"  - {miss}\n"
            f"Check your config and the prepare_data implementation/paths."
        )

    print(f"[data] ✅ Data ready in {out_dir}")
    for p in expected:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
<<<END:train/make_training_data.py>>>

<<<FILE:train/prepare_model.py>>>
import torch
from pathlib import Path
from betadogma.model import BetaDogmaModel
from betadogma.core.encoder_nt import NTEncoder

print("--- Verifying Full Model Workflow ---")

# Define the path to the configuration file
config_path = str(Path(__file__).parent.parent / "src/betadogma/experiments/config/default.yaml")

# --- 1. Instantiate BetaDogmaModel from config ---
print("Loading BetaDogmaModel with NT backend...")
# Use the classmethod to load from config, which correctly sets d_in and attaches config
model = BetaDogmaModel.from_config_file(config_path)
model.eval()

# --- 2. Instantiate Encoder from config ---
print("Loading NTEncoder...")
encoder_config = model.config['encoder']
encoder = NTEncoder(model_id=encoder_config['model_id'])

# Verify that the embedding dimensions match
# The input dimension is stored in the LayerNorm layer of the head's projection module
head_d_in = model.splice_head.proj.norm.normalized_shape[0]
assert head_d_in == encoder.hidden_size, \
    f"Model d_in ({head_d_in}) does not match encoder hidden_size ({encoder.hidden_size})"

print("Model and encoder loaded successfully. Performing a forward pass...")

# --- 3. Create Dummy Data and Generate Embeddings ---
# The NT model has a max sequence length (e.g., 6k for some variants).
# Using a shorter sequence for this verification script is safer and faster.
dummy_sequence = "N" * 4096
print(f"Created a dummy sequence of length {len(dummy_sequence)}")

# Generate embeddings using the encoder. The NTEncoder expects a list of strings.
print("Generating embeddings from sequence...")
embeddings = encoder.forward([dummy_sequence])
print(f"Embeddings generated with shape: {embeddings.shape}")


# --- 4. Perform Forward Pass through BetaDogmaModel ---
# Perform a forward pass through the main model with the embeddings
with torch.no_grad():
    outputs = model(embeddings)

# --- 5. Verify Outputs ---
# Print the shapes of the outputs from each head to verify
print("\n--- Verification Complete ---")
print("Output shapes from prediction heads:")
print(f"Splice (donor):   {outputs['splice']['donor'].shape}")
print(f"Splice (acceptor):{outputs['splice']['acceptor'].shape}")
print(f"TSS:              {outputs['tss']['tss'].shape}")
print(f"PolyA:            {outputs['polya']['polya'].shape}")
print(f"ORF (start):      {outputs['orf']['start'].shape}")
print(f"ORF (stop):       {outputs['orf']['stop'].shape}")
print(f"ORF (frame):      {outputs['orf']['frame'].shape}")
print("\nFull model workflow verified successfully!")
<<<END:train/prepare_model.py>>>

<<<FILE:train/train.py>>>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config-only training entrypoint (lives inside train/).
- Defaults to configs under train/configs/.
- Resolves imports from project-root/src.
- Resolves paths in the config relative to the config file location.
- Works on CPU/GPU; logs AUROC + loss; checkpoints best by AUROC.
"""

import os
import sys
import json
import random
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Project paths ----------
THIS_FILE = Path(__file__).resolve()
TRAIN_DIR = THIS_FILE.parent                    # .../train
PROJECT_ROOT = TRAIN_DIR.parent                 # repo root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))           # make betadogma importable

DEFAULT_CFG = TRAIN_DIR / "configs" / "train.base.yaml"

# ---------- Torch / Lightning ----------
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC

# ---------- YAML loader ----------
def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_dir"] = str(path.parent)   # keep for relative path resolution
    return cfg

# ---------- Repro ----------
def set_seed(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- DNA utils ----------
DNA_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
DNA_COMP = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})

def revcomp(seq: str) -> str:
    return seq.upper().translate(DNA_COMP)[::-1]

def encode_seq(seq: str, max_len: int, pad_value: int = 0) -> torch.LongTensor:
    s = seq.upper()
    ids = [DNA_VOCAB.get(ch, DNA_VOCAB["N"]) for ch in s[:max_len]]
    if len(ids) < max_len:
        ids += [pad_value] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ---------- Dataset ----------
class JsonlSeqDataset(Dataset):
    def __init__(self, path: Path, max_len: int, use_strand: bool, revcomp_minus: bool):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.max_len = int(max_len)
        self.use_strand = bool(use_strand)
        self.revcomp_minus = bool(revcomp_minus)
        self._lines: List[str] = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if not self._lines:
            raise ValueError(f"No records found in JSONL: {path}")

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            ex = json.loads(self._lines[idx])
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {idx} is not valid JSON.") from e
        if "sequence" not in ex or "label" not in ex:
            raise KeyError("Each JSONL line must contain 'sequence' and 'label'.")
        seq = ex["sequence"]
        if not isinstance(seq, str) or not seq:
            raise ValueError("`sequence` must be a non-empty string.")
        if self.use_strand and ex.get("strand") == "-" and self.revcomp_minus:
            seq = revcomp(seq)
        x = encode_seq(seq, self.max_len)  # [L]
        y = ex["label"]
        if isinstance(y, bool):
            y = int(y)
        if y not in (0, 1):
            raise ValueError("`label` must be 0 or 1.")
        return {"x": x, "y": torch.tensor(float(y), dtype=torch.float32)}

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)         # [B, L]
    y = torch.stack([b["y"] for b in batch], dim=0).float() # [B]
    return {"x": x, "y": y}

# ---------- Tiny fallback model (used if model.toy: true) ----------
class TinySeqModel(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=64, hidden=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_long: torch.LongTensor) -> torch.Tensor:
        emb = self.embed(x_long).permute(0, 2, 1)  # [B,E,L]
        feat = self.encoder(emb)                    # [B,H,1]
        logits = self.head(feat).squeeze(1)         # [B]
        return logits

# ---------- LightningModule ----------
class LitSeq(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC()

    def forward(self, x_long: torch.LongTensor) -> torch.Tensor:
        return self.model(x_long)

    def _shared_step(self, batch, stage: str):
        x, y = batch["x"].long(), batch["y"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.auroc.update(probs, y.int())
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=(stage != "train"), batch_size=x.size(0))
        return loss

    def training_step(self, batch, _):  return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        try:
            au = self.auroc.compute()
        except Exception:
            au = torch.tensor(0.0)
        self.log("val/auroc", au, prog_bar=True)
        self.auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# ---------- DataModule ----------
class SeqDataModule(pl.LightningDataModule):
    def __init__(self, dcfg: Dict[str, Any], cfg_dir: Path):
        super().__init__()
        self.cfg = dcfg
        self.cfg_dir = cfg_dir  # for resolving relative paths in the config
        self.pin_memory = (torch.cuda.is_available() if dcfg.get("pin_memory", "auto") == "auto"
                           else bool(dcfg.get("pin_memory")))

    def _resolve(self, p: Optional[str]) -> Optional[Path]:
        if p in (None, "", False):
            return None
        # Paths in the YAML are resolved relative to the CONFIG FILE directory.
        pp = Path(p)
        return (self.cfg_dir / pp) if not pp.is_absolute() else pp

    def setup(self, stage: Optional[str] = None):
        if self.cfg.get("toy", False):
            self.train_ds = self._toy_ds(n=512, L=int(self.cfg["max_len"]))
            self.val_ds   = self._toy_ds(n=128, L=int(self.cfg["max_len"]))
        else:
            req = ("train", "val", "max_len")
            missing = [k for k in req if k not in self.cfg or self.cfg[k] in (None, "")]
            if missing:
                raise ValueError(f"Missing data config keys: {missing}")
            train_path = self._resolve(self.cfg["train"])
            val_path   = self._resolve(self.cfg["val"])
            self.train_ds = JsonlSeqDataset(
                train_path, self.cfg["max_len"], self.cfg.get("use_strand", False),
                self.cfg.get("reverse_complement_minus", True),
            )
            self.val_ds = JsonlSeqDataset(
                val_path, self.cfg["max_len"], self.cfg.get("use_strand", False),
                self.cfg.get("reverse_complement_minus", True),
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=int(self.cfg.get("batch_size", 32)), shuffle=True,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          drop_last=True, collate_fn=collate_batch)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=int(self.cfg.get("batch_size", 32)), shuffle=False,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          collate_fn=collate_batch)

    @staticmethod
    def _toy_ds(n: int, L: int):
        class _Toy(Dataset):
            def __len__(self): return n
            def __getitem__(self, i):
                x = torch.randint(low=0, high=6, size=(L,), dtype=torch.long)
                y = torch.tensor(float((x.sum() % 2)==0))
                return {"x": x, "y": y}
        return _Toy()

# ---------- Helpers ----------
def _maybe_load_yaml_or_dict(value: Union[Dict[str, Any], str, Path], cfg_dir: Path) -> Dict[str, Any]:
    """Accept dicts as-is; load strings/paths as YAML; resolve relative to CONFIG file."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        import yaml
        p = Path(value)
        if not p.is_absolute():
            p = (cfg_dir / p)
        if not p.exists():
            raise FileNotFoundError(f"Model config file not found: {p}")
        with p.open("r") as f:
            return yaml.safe_load(f) or {}
    raise TypeError("model.kwargs.config must be a dict or a path to a YAML file.")

# ---------- Model factory ----------
def build_model(mcfg: Dict[str, Any], max_len: int, cfg_dir: Path) -> nn.Module:
    """
    Accepts:
      - model.class_path: "pkg.mod.Class" (preferred), or "pkg.mod" + model.class_name
      - model.factory:    "pkg.mod:make_model" (callable -> nn.Module)
    Only forwards 'max_len' if the constructor/factory actually accepts it.
    Hydrates 'config' kwarg from dict or YAML path if present.
    """
    # Toy path
    if mcfg.get("toy", False):
        return TinySeqModel(
            vocab_size=6,
            embed_dim=int(mcfg.get("embed_dim", 64)),
            hidden=int(mcfg.get("hidden_dim", 128)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )

    # Optional factory path: "package.module:make_model"
    factory = mcfg.get("factory")
    if factory:
        mod_name, fn_name = factory.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        kwargs = dict(mcfg.get("kwargs") or {})
        # hydrate nested config
        if "config" in kwargs:
            kwargs["config"] = _maybe_load_yaml_or_dict(kwargs["config"], cfg_dir)
        elif "config_path" in kwargs:
            kwargs["config"] = _maybe_load_yaml_or_dict(kwargs.pop("config_path"), cfg_dir)
        # add max_len only if factory takes it
        try:
            params = inspect.signature(fn).parameters
            if "max_len" in params and "max_len" not in kwargs:
                kwargs["max_len"] = int(max_len)
        except (ValueError, TypeError):
            pass
        model = fn(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Factory '{factory}' must return a torch.nn.Module.")
        return model

    # Class path route
    class_path = mcfg.get("class_path")
    class_name = mcfg.get("class_name")
    if not class_path:
        raise ValueError("model.class_path (or model.factory) is required unless model.toy is true.")

    # Resolve module + class
    if "." in class_path and class_path.split(".")[-1][:1].isupper() and not class_name:
        module_name, cls_name = class_path.rsplit(".", 1)
    else:
        module_name, cls_name = class_path, (class_name or "")
    mod = importlib.import_module(module_name)

    obj = getattr(mod, cls_name) if cls_name else None
    if obj is None or isinstance(obj, type(importlib)):
        # fall back to a likely class in the module
        candidates = [(k, v) for k, v in vars(mod).items()
                      if isinstance(v, type) and issubclass(v, nn.Module) and k[:1].isupper()]
        if not candidates:
            raise ImportError(
                f"No nn.Module classes found in '{module_name}'. "
                f"Set model.class_path to 'pkg.mod.ClassName'."
            )
        # prefer names containing Decoder / Isoform
        candidates.sort(key=lambda kv: (("Decoder" not in kv[0], "Isoform" not in kv[0]), kv[0]))
        obj = candidates[0][1]

    kwargs = dict(mcfg.get("kwargs") or {})
    # hydrate nested config if provided
    if "config" in kwargs:
        kwargs["config"] = _maybe_load_yaml_or_dict(kwargs["config"], cfg_dir)
    elif "config_path" in kwargs:
        kwargs["config"] = _maybe_load_yaml_or_dict(kwargs.pop("config_path"), cfg_dir)

    # add max_len only if constructor takes it
    try:
        params = inspect.signature(obj.__init__).parameters
        if "max_len" in params and "max_len" not in kwargs:
            kwargs["max_len"] = int(max_len)
    except (ValueError, TypeError):
        pass

    return obj(**kwargs)

# ---------- Trainer from config ----------
def build_trainer(tcfg: Dict[str, Any], cfg_dir: Path) -> pl.Trainer:
    # Resolve log/ckpt dirs relative to CONFIG file (keeps test runs separate)
    logdir   = Path(tcfg.get("logdir", "runs/betadogma"))
    ckpt_dir = Path(tcfg.get("ckpt_dir", "checkpoints/betadogma"))
    if not logdir.is_absolute():   logdir   = cfg_dir / logdir
    if not ckpt_dir.is_absolute(): ckpt_dir = cfg_dir / ckpt_dir
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(logdir), name="", version=None, default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch:02d}-{val_auroc:.4f}",
                        monitor="val/auroc", mode="max",
                        save_top_k=int(tcfg.get("save_top_k", 2)), save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    pat = tcfg.get("early_stopping_patience", None)
    if pat is not None:
        callbacks.append(EarlyStopping(monitor="val/auroc", mode="max", patience=int(pat)))

    precision = tcfg.get("precision", "32-true")
    if not torch.cuda.is_available() and precision != "32-true":
        precision = "32-true"  # avoid AMP-on-CPU warnings

    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(tcfg.get("devices", 1)),
        max_epochs=int(tcfg.get("epochs", 20)),
        precision=precision,
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 25)),
        check_val_every_n_epoch=1,
        deterministic=True,
        enable_progress_bar=True,
        limit_train_batches=tcfg.get("limit_train_batches", None),
        limit_val_batches=tcfg.get("limit_val_batches", None),
    )

def main():
    # 1) Pick config (env or default inside train/configs/)
    cfg_env = os.environ.get("TRAIN_CONFIG", "")
    cfg_path = Path(cfg_env).resolve() if cfg_env else DEFAULT_CFG
    if not cfg_path.is_absolute():
        cfg_path = (TRAIN_DIR / cfg_path).resolve()
    cfg = load_config(cfg_path)
    cfg_dir = Path(cfg["_config_dir"]).resolve()

    # 2) Sections
    seed = int(cfg.get("seed", 42))
    dcfg = dict(cfg.get("data", {}))
    mcfg = dict(cfg.get("model", {}))
    ocfg = dict(cfg.get("optim", {"lr": 3e-4, "weight_decay": 0.01}))
    tcfg = dict(cfg.get("trainer", {}))

    # 3) Validate presence of data unless toy
    if not mcfg.get("toy", False):
        for key in ("train", "val", "max_len"):
            if key not in dcfg or dcfg[key] in (None, ""):
                raise ValueError(f"Non-toy training requires data.{key}. Missing in {cfg_path}.")

    # 4) Seed
    set_seed(seed)

    # 5) Build components (paths resolved vs. CONFIG location)
    dm = SeqDataModule(dcfg, cfg_dir=cfg_dir)
    _model_kwargs = mcfg.get("kwargs") or {}
    max_len = int(dcfg.get("max_len", _model_kwargs.get("max_len", 0)))
    model = build_model(mcfg, max_len=max_len, cfg_dir=cfg_dir)
    lit = LitSeq(model=model, lr=float(ocfg.get("lr", 3e-4)), weight_decay=float(ocfg.get("weight_decay", 0.01)))
    trainer = build_trainer(tcfg, cfg_dir=cfg_dir)

    # 6) Train
    trainer.fit(lit, datamodule=dm)

if __name__ == "__main__":
    main()
<<<END:train/train.py>>>

<<<FILE:train/train_isoform_ranker.py>>>
"""
Phase 2a Training: Isoform Structure Ranking

This script trains the IsoformScorer to rank candidate isoforms.
The goal is to teach the model to assign higher scores to the annotated/correct
isoform compared to other enumerated candidates.
"""

import argparse
import yaml
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Import the module so we can patch it
import betadogma.model
from betadogma.model import BetaDogmaModel
from betadogma.decoder import Isoform, Exon

# --- Monkeypatching for lightweight training ---
class DummyEncoder(nn.Module):
    """A dummy encoder that returns random tensors of the correct shape."""
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # This should match the encoder's hidden_size from the config
        return torch.randn(batch_size, seq_len, 1536)

def monkeypatch_encoder():
    """Replaces the real encoder class with a dummy one."""
    print("Monkeypatching BetaDogmaEncoder with DummyEncoder.")
    betadogma.model.BetaDogmaEncoder = DummyEncoder

# --- Loss Function ---
class PairwiseHingeLoss(nn.Module):
    """Pairwise hinge loss for ranking."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        loss = torch.relu(self.margin - (positive_scores - negative_scores))
        return loss.mean()

# --- Training Logic ---
def get_mock_batch(device: torch.device) -> Dict[str, Any]:
    """Creates a mock batch of data for demonstration purposes."""
    seq_len = 2048

    # Create logits with a few high-confidence peaks to prevent combinatorial explosion
    donor_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    acceptor_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    tss_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    polya_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_start_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_stop_logits = torch.full((1, 1, seq_len), -10.0, device=device)
    orf_frame_logits = torch.randn(1, 1, seq_len, 3, device=device)


    # Define sites for a true isoform and some distractors
    # True isoform path: [Exon(50, 200), Exon(300, 500)]
    # This requires: TSS@50, don@200, acc@300, polyA@500
    tss_logits[0, 0, 50] = 5.0
    donor_logits[0, 0, 200] = 5.0
    acceptor_logits[0, 0, 300] = 5.0
    polya_logits[0, 0, 500] = 5.0

    # Add some distractor sites
    acceptor_logits[0, 0, 100] = 2.0
    donor_logits[0, 0, 400] = 2.0

    head_outputs = {
        "splice": {"donor": donor_logits.transpose(1, 2), "acceptor": acceptor_logits.transpose(1, 2)},
        "tss": {"tss": tss_logits.transpose(1, 2)},
        "polya": {"polya": polya_logits.transpose(1, 2)},
        "orf": {
            "start": orf_start_logits.transpose(1, 2),
            "stop": orf_stop_logits.transpose(1, 2),
            "frame": orf_frame_logits.squeeze(0), # Remove channel dim
        }
    }

    # A "true" isoform for ranking against
    true_isoform = Isoform(exons=[Exon(50, 200), Exon(300, 500)], strand='+')

    # Mock input_ids for sequence-based scoring fallback
    mock_input_ids = torch.randint(0, 5, (1, seq_len), device=device)

    return {"head_outputs": head_outputs, "true_isoform": true_isoform, "strand": "+", "input_ids": mock_input_ids}

def select_candidates(candidates: List[Isoform], true_isoform: Isoform) -> (List[Isoform], List[Isoform]):
    """Separates candidate isoforms into positive (matching ground truth) and negative."""
    # This is a simplified matching logic. A real implementation would be more robust.
    positive, negative = [], []
    true_exons = tuple((e.start, e.end) for e in true_isoform.exons)
    for cand in candidates:
        cand_exons = tuple((e.start, e.end) for e in cand.exons)
        if cand_exons == true_exons:
            positive.append(cand)
        else:
            negative.append(cand)
    return positive, negative

def train_one_epoch(
    model: BetaDogmaModel,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mock_loader: List[Dict]
):
    model.train()
    total_loss = 0.0

    for batch in mock_loader:
        head_outputs = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in batch['head_outputs'].items()}
        input_ids = batch['input_ids'].to(device)

        # The decoder is used as a utility here, not trained directly.
        candidates = model.isoform_decoder.decode(head_outputs, strand=batch['strand'], input_ids=input_ids)

        pos_cands, neg_cands = select_candidates(candidates, batch['true_isoform'])

        if not pos_cands or not neg_cands:
            continue

        # Score candidates using the learnable scorer
        scorer = model.isoform_decoder.scorer
        pos_scores = torch.stack([scorer(iso, head_outputs, input_ids=input_ids) for iso in pos_cands])
        # For simplicity, we only use one negative example per positive
        neg_scores = torch.stack([scorer(iso, head_outputs, input_ids=input_ids) for iso in neg_cands[:len(pos_cands)]])

        loss = loss_fn(pos_scores, neg_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(mock_loader) if mock_loader else 0.0

def main():
    parser = argparse.ArgumentParser(description="Train the IsoformScorer for ranking.")
    parser.add_argument("--config", default="src/betadogma/experiments/config/default.yaml", help="Path to config.")
    parser.add_argument("--no-patch", action="store_true", help="Do not monkeypatch the encoder.")
    args = parser.parse_args()

    if not args.no_patch:
        monkeypatch_encoder()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The dummy encoder uses a hardcoded dimension of 1536
    model = BetaDogmaModel(d_in=1536, config=config)
    model.to(device)

    # We are only training the scorer's weights
    trainable_parameters = model.isoform_decoder.scorer.parameters()

    optimizer = optim.Adam(trainable_parameters, lr=config['optimizer']['lr'])
    loss_fn = PairwiseHingeLoss(margin=0.5)

    # Create a mock data loader for demonstration
    mock_loader = [get_mock_batch(device) for _ in range(10)]

    print("Starting training with mock data...")
    for epoch in range(config['trainer']['epochs']):
        epoch_loss = train_one_epoch(model, optimizer, loss_fn, device, mock_loader)
        print(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Mock Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()
<<<END:train/train_isoform_ranker.py>>>

