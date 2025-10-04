# BetaDogma: Task Index

## Core Development
- `core/encoder.py`: GENERator wrapper
- `core/heads.py`: splice/TSS/polyA/ORF heads
- `decoder/isoform_decoder.py`: exon chain + ψ
- `nmd/nmd_rules.py` & `nmd/nmd_model.py`
- `variant/encode.py` & `variant/simulate.py`
- `data/dataset.py` & `data/prepare_*`

## Experiments
- `experiments/train.py`: multi-task training loop
- `experiments/config/default.yaml`: hyperparameters

## Evaluation
- notebooks in `notebooks/` for metrics & plots
