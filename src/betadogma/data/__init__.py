"""Data loaders and preprocessing pipelines."""
"""
Data loaders and preprocessing pipelines for BetaDogma.
"""

from .dataset import (
    BetaDogmaDataset,
    StructuralParquetDataset,
    collate_structural_batch,
)
from .io import (
    read_fasta_window,
    read_parquet_shards,
    write_parquet,
    iter_vcf_records,
)

__all__ = [
    # datasets
    "BetaDogmaDataset",
    "StructuralParquetDataset",
    "collate_structural_batch",
    # IO
    "read_fasta_window",
    "read_parquet_shards",
    "write_parquet",
    "iter_vcf_records",
]