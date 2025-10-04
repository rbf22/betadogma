"""
Unified dataset objects for training/eval.
"""
from typing import Any, Dict, List, Tuple

class BetaDogmaDataset:
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]
