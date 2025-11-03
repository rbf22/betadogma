# BetaDogma

BetaDogma is a deep learning system for predicting protein sequences from genomic regions, with a focus on accurately modeling the effects of genetic variants on protein products.

## Core Data Structures

### 1. GenomicRegion
Represents a genomic interval with associated features and variants.

```python
@dataclass
class GenomicRegion:
    # Core features
    chrom: str           # Chromosome name (e.g., 'chr1')
    start: int           # 0-based start position
    end: int             # 1-based end position
    strand: str          # '+' or '-'
    sequence: str        # Reference DNA sequence (5'→3')
    
    # Variants in this region
    variants: List[Variant]  # All variants in this region
    
    # Splicing features
    junction_psi: Dict[str, float]      # junction_id -> PSI value (0-1)
    splice_sites: Dict[str, List[int]]  # 'donor'/'acceptor' -> positions
    tss_positions: List[int]            # Transcription start sites
    polya_positions: List[int]          # PolyA sites
```

### 2. Variant
Represents a genetic variant with functional annotations.

```python
@dataclass
class Variant:
    # Core variant information
    pos: int             # 0-based position
    ref: str             # Reference allele
    alt: str             # Alternate allele
    qual: float          # Quality score
    
    # Transcript context (if in coding region)
    transcript_id: Optional[str] = None
    gene_id: Optional[str] = None
    cds_position: Optional[int] = None  # 0-based position in CDS
    codon_change: Optional[Tuple[str, str]] = None  # (ref_codon, alt_codon)
    aa_change: Optional[Tuple[str, str]] = None     # (ref_aa, alt_aa)
    
    # Functional impact
    is_ptc: Optional[bool] = None          # Introduces premature stop
    is_nmd_sensitive: Optional[bool] = None # Triggers NMD
    impact: Optional[str] = None           # 'HIGH'|'MODERATE'|'LOW'|'MODIFIER'
    
    # Population genetics
    population_af: Optional[float] = None  # Allele frequency (0-1)
    variant_type: Optional[str] = None     # 'SNP'|'INS'|'DEL'|'COMPLEX'
    
    # Additional metadata
    info: Dict[str, Any] = field(default_factory=dict)
```

### 3. ProteinPrediction
Represents a predicted protein sequence with annotations.

```python
@dataclass
class ProteinPrediction:
    sequence: str                     # Predicted amino acid sequence
    coding_sequence: str              # Underlying coding DNA (CDS)
    transcript_id: str                # Source transcript
    is_canonical: bool                # Matches reference
    variant_effects: List[VariantEffect]  # Effects of variants
    confidence: float                 # Prediction confidence (0-1)
```

## Core Modules

### 1. DNATranslator
Handles DNA to protein translation with support for different genetic codes.

```python
class DNATranslator:
    def translate(self, dna: str, frame: int = 0) -> str:
        """Translate DNA sequence to protein."""
        pass
        
    def find_orfs(self, dna: str, min_aa: int = 30) -> List[Dict]:
        """Find all open reading frames in DNA."""
        pass
```

### 2. VariantEffectPredictor
Predicts the effect of variants on protein sequence.

```python
class VariantEffectPredictor:
    def predict_effect(
        self, 
        chrom: str, 
        pos: int, 
        ref: str, 
        alt: str
    ) -> Dict[str, Any]:
        """Predict effect of a variant on protein sequence."""
        pass
```

### 3. TranscriptAnnotator
Manages transcript annotations and coordinate transformations.

```python
class TranscriptAnnotator:
    def get_coding_sequence(self, transcript_id: str) -> str:
        """Get coding sequence for a transcript."""
        pass
        
    def genomic_to_cds(self, transcript_id: str, pos: int) -> Optional[int]:
        """Convert genomic position to CDS position."""
        pass
```

## Data Processing Pipeline

1. **Input Preparation**
   - Load reference genome
   - Annotate transcripts
   - Process variant calls

2. **Feature Extraction**
   - Extract sequence features
   - Calculate conservation scores
   - Annotate splice sites

3. **Prediction**
   - Predict transcript isoforms
   - Call variants
   - Predict protein sequences

4. **Output**
   - Annotated VCF
   - Protein sequences
   - Effect predictions

## Example Usage

```python
# Initialize components
translator = DNATranslator()
predictor = VariantEffectPredictor(reference_genome)
annotator = TranscriptAnnotator(gtf_path)

# Process a genomic region
region = GenomicRegion(
    chrom="chr1",
    start=10000,
    end=20000,
    strand="+",
    sequence="ATGCGTACGT...",
    variants=[variant1, variant2]
)

# Predict protein sequence
prediction = predict_protein_sequence(region)
print(f"Predicted protein: {prediction.sequence}")

# Get variant effects
for effect in prediction.variant_effects:
    print(f"{effect.variant_id}: {effect.effect_type}")
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/betadogma.git
cd betadogma

# Create and activate conda environment
conda create -n betadogma python=3.9
conda activate betadogma

# Install dependencies
pip install -r requirements.txt
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
