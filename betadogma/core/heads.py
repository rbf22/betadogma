"""
Per-base biological heads:
- Splice donor/acceptor
- TSS
- PolyA
- ORF (start/stop/frame)

Each head should accept nucleotide embeddings [L, D] and return per-base logits or structured outputs.
"""
from typing import Any, Dict

class SpliceHead:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def forward(self, embeddings):
        # TODO: return donor/acceptor logits
        return {"donor": None, "acceptor": None}

class TSSHead:
    def forward(self, embeddings):
        return {"tss": None}

class PolyAHead:
    def forward(self, embeddings):
        return {"polya": None}

class ORFHead:
    def forward(self, embeddings):
        return {"start": None, "stop": None, "frame": None}
