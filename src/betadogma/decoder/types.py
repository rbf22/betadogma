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