# BetaDogma Datasets

## Core Reference

| Dataset | Source | Usage |
|---------|--------|-------|
| **GENCODE v44** | gencodegenes.org | Exon/intron structure, CDS boundaries |
| **RefSeq** | NCBI | Validation |
| **GTEx v8** | Broad | RNA-seq, eQTL/sQTL for ψ supervision |
| **ENCODE** | ENCODE Project | Chromatin/expression |
| **FANTOM5 CAGE** | RIKEN | TSS training |
| **PolyA-DB / APASdb** | Various | 3′ end training |
| **Ribo-seq (GWIPS)** | public | ORF evidence |
| **UPF1/SMG6 KD RNA-seq** | GEO | NMD labels |
| **BRIC / 4sU** | GEO | mRNA half-life |
| **gnomAD + eQTL Catalog** | Broad/EBI | Variant effect modeling |

## Preprocessing

All inputs are converted to HDF5/Parquet with fields:
- `sequence` (DNA)
- `features` (donor/acceptor/TSS/polyA)
- `isoforms` (exon chains)
- `psi` (relative usage)
- `nmd_label` (0/1 or continuous)
- `variants` (VCF-derived)

Pipelines live in `data/prepare_*.py`.
