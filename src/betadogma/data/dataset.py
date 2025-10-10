"""
Unified dataset objects for training/eval.
"""

from __future__ import annotations
from typing import Any, Dict, List, Iterable, Optional, Union
from dataclasses import dataclass
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd


# DNA utilities
DNA_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
DNA_COMP = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})


def revcomp(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.
    
    Args:
        seq: Input DNA sequence (case-insensitive)
        
    Returns:
        Reverse complement of the input sequence in uppercase
    """
    return seq.upper().translate(DNA_COMP)[::-1]


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
    def __init__(self, records: List[Union[Dict[str, Any], BetaDogmaRecord]]):
        """Initialize the dataset with a list of records.
        
        Args:
            records: List of records, where each record is either a dictionary
                    or a BetaDogmaRecord instance.
        """
        self.records: List[Dict[str, Any]] = [
            r if isinstance(r, dict) else r.__dict__ 
            for r in records
        ]

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a record by index.
        
        Args:
            idx: Index of the record to retrieve.
            
        Returns:
            A dictionary containing the record data.
            
        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self.records):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.records)}")
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
            "tss": torch.tensor(r["tss"], dtype=torch.float32),
            "polya": torch.tensor(r["polya"], dtype=torch.float32),
        }


class JsonlSeqDataset(Dataset):
    """Dataset for sequence data stored in JSONL format.
    
    Each line in the JSONL file should be a JSON object with at least a 'seq' field
    containing the DNA sequence. Optionally, it can include 'strand' and 'label' fields.
    
    Args:
        jsonl_path: Path to the JSONL file containing sequence data
        max_len: Maximum sequence length (longer sequences will be truncated)
        use_strand: If True, use strand information from the data
        reverse_complement_minus: If True, reverse complement sequences on the minus strand
                                 when use_strand is True
    """
    
    def __init__(
        self, 
        jsonl_path: Union[str, Path], 
        max_len: int,
        use_strand: bool = False,
        reverse_complement_minus: bool = True
    ):
        self.jsonl_path = Path(jsonl_path)
        self.max_len = max_len
        self.use_strand = use_strand
        self.reverse_complement_minus = reverse_complement_minus
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and validate samples from the JSONL file."""
        samples = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Validate required fields
                    if 'seq' not in item:
                        continue
                    # Ensure sequence is uppercase and contains only valid bases
                    seq = item['seq'].upper()
                    if not all(base in 'ACGTN' for base in seq):
                        continue
                    
                    # Process strand information if needed
                    if self.use_strand:
                        strand = item.get('strand', '+')
                        if strand not in ['+', '-']:
                            strand = '+'  # Default to '+' if invalid
                        item['strand'] = strand
                    
                    # Store the processed item
                    item['seq'] = seq
                    samples.append(item)
                except json.JSONDecodeError:
                    continue
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        item = self.samples[idx]
        seq = item['seq']
        
        # Handle sequence length
        if len(seq) > self.max_len:
            # Randomly sample a window of max_len
            start = random.randint(0, len(seq) - self.max_len)
            seq = seq[start:start + self.max_len]
        
        # Reverse complement if needed
        if self.use_strand and item.get('strand') == '-' and self.reverse_complement_minus:
            seq = revcomp(seq)
        
        # Convert sequence to tensor of indices
        seq_tensor = torch.tensor([DNA_VOCAB.get(base, 0) for base in seq], dtype=torch.long)
        
        # Prepare the output
        result = {
            'seq': seq,
            'seq_tensor': seq_tensor,
            'length': len(seq)
        }
        
        # Add label if available
        if 'label' in item:
            result['label'] = torch.tensor(float(item['label']), dtype=torch.float32)
        
        return result


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for JsonlSeqDataset.
    
    Pads sequences to the maximum length in the batch and returns a batch tensor.
    """
    from torch.nn.utils.rnn import pad_sequence
    
    # Extract sequences and labels
    seqs = [item['seq'] for item in batch]
    seq_tensors = [item['seq_tensor'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in seq_tensors], dtype=torch.long)
    
    # Pad sequences
    padded_seqs = pad_sequence(
        seq_tensors, 
        batch_first=True, 
        padding_value=0  # Assuming 0 is the padding index
    )
    
    # Prepare the batch
    batch_dict = {
        'seq': seqs,
        'seq_tensor': padded_seqs,
        'length': lengths
    }
    
    # Add labels if present
    if 'label' in batch[0]:
        batch_dict['label'] = torch.stack([item['label'] for item in batch])
    
    return batch_dict


def collate_structural_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function compatible with experiments.train:
    - returns list[str] seqs for the encoder
    - pads binary label tracks to the max length in the mini-batch
    """
    from torch.nn.utils.rnn import pad_sequence
    
    def pad_1d(x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad a 1D tensor to the target length with zeros."""
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.size(0) >= target_length:
            return x[:target_length]
        return torch.cat([x, torch.zeros(target_length - x.size(0), dtype=x.dtype, device=x.device)])
    
    # Extract sequences
    seqs = [item["seq"] for item in batch]
    
    # Find maximum sequence length in the batch
    T = max(item["donor"].numel() for item in batch)
    
    # Pad and stack all label tensors
    donor = torch.stack([pad_1d(item["donor"], T) for item in batch])
    acceptor = torch.stack([pad_1d(item["acceptor"], T) for item in batch])
    tss = torch.stack([pad_1d(item["tss"], T) for item in batch])
    polya = torch.stack([pad_1d(item["polya"], T) for item in batch])
    
    return {
        "seq": seqs,
        "donor": donor,
        "acceptor": acceptor,
        "tss": tss,
        "polya": polya
    }