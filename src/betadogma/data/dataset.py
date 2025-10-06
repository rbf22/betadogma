"""
Unified dataset objects for training/eval.
"""


from __future__ import annotations
from typing import Any, Dict, List, Iterable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import pandas as pd


@dataclass
class BetaDogmaRecord:
    chrom: str
    start: int
    end: int
    seq: str
    donor: List[int]
    acceptor: List[int]
    tss: List[int]
    polya: List[int]
    bin_size: int = 1


class BetaDogmaDataset(Dataset):
    """
    In-memory dataset from a list of dicts or BetaDogmaRecord objects.
    Useful for small tests and unit fixtures.
    """
    def __init__(self, records: List[Dict[str, Any]] | List[BetaDogmaRecord]):
        self.records = [r.__dict__ if hasattr(r, "__dict__") else r for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class StructuralParquetDataset(Dataset):
    """
    Stream a set of Parquet shards produced by prepare_gencode.py.

    Each row must contain:
      - chrom, start, end, seq, bin_size
      - donor, acceptor, tss, polya  (array-like; length = (end-start)/bin_size)

    This dataset keeps rows in memory for simplicity. If shards are huge,
    replace with an on-demand reader.
    """
    def __init__(self, parquet_paths: List[str], max_shards: int | None = None):
        paths = sorted(parquet_paths)
        if max_shards:
            paths = paths[:max_shards]
        self.rows: List[Dict[str, Any]] = []
        for p in paths:
            df = pd.read_parquet(p)
            self.rows.extend(df.to_dict("records"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        # minimal sanity checks
        Lr = int((int(r["end"]) - int(r["start"])) // int(r.get("bin_size", 1)))
        for key in ("donor", "acceptor", "tss", "polya"):
            if len(r[key]) != Lr:
                # soft-fix by trimming/padding
                arr = list(r[key])[:Lr]
                if len(arr) < Lr:
                    arr += [0] * (Lr - len(arr))
                r[key] = arr
        return {
            "chrom": r.get("chrom", ""),
            "start": int(r.get("start", 0)),
            "end": int(r.get("end", 0)),
            "seq": r["seq"],
            "bin_size": int(r.get("bin_size", 1)),
            "donor": torch.tensor(r["donor"], dtype=torch.float32),
            "acceptor": torch.tensor(r["acceptor"], dtype=torch.float32),
            "tss": torch.tensor(r["tss"], dtype=torch.float32),
            "polya": torch.tensor(r["polya"], dtype=torch.float32),
        }


def collate_structural_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function compatible with experiments.train:
    - returns list[str] seqs for the encoder
    - pads binary label tracks to the max length in the mini-batch
    """
    seqs = [b["seq"] for b in batch]

    def pad_1d(x: torch.Tensor, T: int) -> torch.Tensor:
        L = x.numel()
        if L < T:
            return torch.nn.functional.pad(x, (0, T - L))
        return x[:T]

    T = max(b["donor"].numel() for b in batch)
    donor   = torch.stack([pad_1d(b["donor"],   T) for b in batch])
    accept  = torch.stack([pad_1d(b["acceptor"],T) for b in batch])
    tss     = torch.stack([pad_1d(b["tss"],     T) for b in batch])
    polya   = torch.stack([pad_1d(b["polya"],   T) for b in batch])

    return {"seqs": seqs, "donor": donor, "acceptor": accept, "tss": tss, "polya": polya}