"""
Data structures for the isoform decoder.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Exon:
    """
    Represents a single exon in an isoform.
    Coordinates are half-open [start, end).
    """
    start: int
    end: int
    score: float = 0.0

    def __post_init__(self):
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("Exon start and end must be integers")
        if self.end <= self.start:
            raise ValueError(f"Invalid exon coordinates: {self.start}–{self.end}")

    @property
    def length(self) -> int:
        """Return exon length (end - start)."""
        return self.end - self.start

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


@dataclass
class Isoform:
    """
    Represents a single transcript isoform, composed of a chain of exons.

    Fields:
        exons: list of Exon objects (in transcription order)
        strand: '+' or '-'
        score: isoform-level score (from IsoformScorer)
        cds: optional (start, end) tuple of CDS coordinates
        meta: dict for additional metadata
    """
    exons: List[Exon]
    strand: str
    score: float = 0.0
    cds: Optional[Tuple[int, int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        for e in self.exons:
            if not isinstance(e, Exon):
                raise TypeError("exons must be a list of Exon objects")

    @property
    def start(self) -> int:
        """Return genomic start coordinate (5' end for + strand)."""
        return self.exons[0].start if self.exons else -1

    @property
    def end(self) -> int:
        """Return genomic end coordinate (3' end for + strand)."""
        return self.exons[-1].end if self.exons else -1

    @property
    def exon_count(self) -> int:
        return len(self.exons)

    @property
    def transcript_length(self) -> int:
        """Total spliced cDNA length."""
        return sum(e.length for e in self.exons)

    def key(self) -> str:
        """Stable identifier for hashing / PSI dicts."""
        exons_str = ",".join(f"{e.start}-{e.end}" for e in self.exons)
        return f"{self.strand}:{exons_str}"