# Contributing to BetaDogma

We welcome contributions from computational biologists, ML researchers, and data engineers.

## Modules

| Path | Responsibility |
|------|----------------|
| `core/` | model architecture and training utilities |
| `decoder/` | isoform graph + ψ |
| `nmd/` | transcript decay prediction |
| `variant/` | variant encoding and simulation |
| `data/` | dataset loading, normalization, caching |
| `experiments/` | training configs, hyperparameters |
| `notebooks/` | analysis notebooks |

## Workflow

1. Fork & branch  
2. Write tests (`pytest`)  
3. Lint & type-check (ruff/mypy)  
4. Open a PR with a clear summary and references.
