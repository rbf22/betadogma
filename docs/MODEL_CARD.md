# BetaDogma Model Card

**Type:** Multi-task genomic model  
**Backbone:** GENERator (98 kb context)  
**Language:** DNA (A/C/G/T + variant tokens)

## Tasks
- Splicing (donor/acceptor)
- TSS / PolyA
- Isoform assembly + ψ
- NMD probability
- Variant Δψ / ΔNMD

## Metrics (targets)
- Splice F1 ≥ 0.95
- ψ corr ≥ 0.85
- NMD AUROC ≥ 0.90
- Δψ corr ≥ 0.70

## Limitations
- Human GRCh38 focus
- Context-dependent NMD labels
- Variant supervision biased toward common variants

## Ethics
- No personally identifiable genetic data
- Not for clinical use
