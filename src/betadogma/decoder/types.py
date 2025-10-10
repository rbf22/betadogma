"""
Data structures for the isoform decoder.

This module defines the core data structures used for representing exons and isoforms
in the transcript assembly process.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, TypeVar, Generic, Sequence, Union
from dataclasses import dataclass, field

# Type variable for generic types
T = TypeVar('T')


@dataclass
class Exon:
    """
    Represents a single exon in an isoform.
    
    Coordinates are half-open [start, end), following Python's 0-based, 
    half-open interval convention (like Python slices).
    
    Attributes:
        start: Start position of the exon (0-based, inclusive)
        end: End position of the exon (exclusive)
        score: Confidence score for the exon (default: 0.0)
    """
    start: int
    end: int
    score: float = 0.0

    def __post_init__(self) -> None:
        """Validate exon coordinates.
        
        Raises:
            TypeError: If start or end are not integers
            ValueError: If end is not greater than start
        """
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError(f"Exon start and end must be integers, got {type(self.start).__name__} and {type(self.end).__name__}")
        if self.end <= self.start:
            raise ValueError(f"Invalid exon coordinates: {self.start}–{self.end} (end must be > start)")

    @property
    def length(self) -> int:
        """Return the length of the exon.
        
        Returns:
            int: The length of the exon (end - start)
        """
        return self.end - self.start
        
    def overlaps(self, other: 'Exon') -> bool:
        """Check if this exon overlaps with another exon.
        
        Args:
            other: Another Exon to check for overlap with
            
        Returns:
            bool: True if the exons overlap, False otherwise
        """
        return (self.start < other.end) and (self.end > other.start)

    def as_tuple(self) -> Tuple[int, int]:
        """Return the exon coordinates as a tuple (start, end).
        
        Returns:
            Tuple[int, int]: A tuple containing the start and end positions
        """
        return (self.start, self.end)


@dataclass
class Isoform:
    """
    Represents a single transcript isoform, composed of a chain of exons.
    
    The exons are stored in transcription order (5' to 3' for the + strand,
    or 3' to 5' for the - strand).

    Attributes:
        exons: List of Exon objects in transcription order
        strand: Strand of the transcript, must be '+' or '-'
        score: Confidence score for the isoform (default: 0.0)
        cds: Optional tuple of (start, end) coordinates for the coding sequence
        meta: Dictionary for additional metadata (e.g., gene_id, transcript_id)
    """
    exons: List[Exon]
    strand: str
    score: float = 0.0
    cds: Optional[Tuple[int, int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the Isoform object.
        
        Raises:
            ValueError: If strand is not '+' or '-', or if exons are not in order
            TypeError: If exons contains non-Exon objects
        """
        if self.strand not in {"+", "-"}:
            raise ValueError(f"strand must be '+' or '-', got '{self.strand}'")
            
        if not isinstance(self.exons, list):
            raise TypeError(f"exons must be a list, got {type(self.exons).__name__}")
            
        for i, e in enumerate(self.exons):
            if not isinstance(e, Exon):
                raise TypeError(f"exons must be a list of Exon objects, got {type(e).__name__} at index {i}")
                
        # Check that exons are in the correct order and don't overlap
        for i in range(1, len(self.exons)):
            if self.exons[i].start <= self.exons[i-1].end:
                raise ValueError(
                    f"Exons must be non-overlapping and in order. "
                    f"Exon {i} starts at {self.exons[i].start} but previous exon ends at {self.exons[i-1].end}"
                )

    @property
    def start(self) -> int:
        """Return the genomic start coordinate (5' end for + strand).
        
        Returns:
            int: The start position of the first exon in the isoform,
                 or -1 if there are no exons
        """
        return self.exons[0].start if self.exons else -1

    @property
    def end(self) -> int:
        """Return the genomic end coordinate (3' end for + strand).
        
        Returns:
            int: The end position of the last exon in the isoform,
                 or -1 if there are no exons
        """
        return self.exons[-1].end if self.exons else -1

    @property
    def exon_count(self) -> int:
        """Return the number of exons in the isoform.
        
        Returns:
            int: The number of exons
        """
        return len(self.exons)

    @property
    def transcript_length(self) -> int:
        """Calculate the total length of the spliced transcript.
        
        This is the sum of all exon lengths, which represents the length
        of the mature mRNA after splicing.
        
        Returns:
            int: The total length of the spliced transcript in nucleotides
        """
        return sum(e.length for e in self.exons)

    def key(self) -> str:
        """Generate a stable string key for the isoform.
        
        This key is useful for hashing and for creating dictionary keys.
        The format is: "strand:start1-end1,start2-end2,..."
        
        Returns:
            str: A string representation of the isoform's structure
        """
        exons_str = ",".join(f"{e.start}-{e.end}" for e in self.exons)
        return f"{self.strand}:{exons_str}"
        
    def get_exon_overlaps(self, other: 'Isoform') -> List[Tuple[Exon, Exon]]:
        """Find all pairs of exons that overlap between two isoforms.
        
        Args:
            other: Another Isoform to compare with
            
        Returns:
            List of (exon1, exon2) tuples where exon1 is from this isoform
            and exon2 is from the other isoform, and they overlap
        """
        overlaps = []
        for e1 in self.exons:
            for e2 in other.exons:
                if e1.overlaps(e2):
                    overlaps.append((e1, e2))
        return overlaps