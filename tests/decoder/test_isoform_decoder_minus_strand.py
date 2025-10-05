import pytest
import torch
from dataclasses import dataclass

from betadogma.decoder.isoform_decoder import (
    IsoformDecoder,
    SpliceGraphBuilder,
    _score_orf,
    _get_spliced_cDNA,
)
from betadogma.decoder.types import Exon, Isoform

@dataclass
class CdsWindow:
    start: int
    end: int
    strand: str

def toy_minus_gene():
    # Genome coords: higher -> lower is transcription
    # Exon B (upstream in transcription) has higher coord than Exon A.
    exons = [
        Exon(start=100, end=120, score=0.9),  # A (3' in transcript)
        Exon(start=150, end=170, score=0.9),  # B (5' in transcript)
    ]
    strand = "-"
    # CDS covers from 155..115 on the genome (reverse-complement semantics)
    cds = CdsWindow(start=115, end=155, strand=strand)
    return exons, strand, cds

def mock_head_outputs(device="cpu"):
    """Mocks head outputs for a ~200bp sequence."""
    seq_len = 200
    outputs = {
        "splice": {
            "donor": torch.full((seq_len,), -10.0, device=device),
            "acceptor": torch.full((seq_len,), -10.0, device=device),
        },
        "tss": {"tss": torch.full((seq_len,), -10.0, device=device)},
        "polya": {"polya": torch.full((seq_len,), -10.0, device=device)},
        "orf": {
            "start": torch.full((seq_len,), -10.0, device=device),
            "stop": torch.full((seq_len,), -10.0, device=device),
            "frame": torch.full((seq_len, 3), -10.0, device=device),
        }
    }
    # From toy_minus_gene, transcription order is B -> A
    # Donor must be at start of B, Acceptor at end of A (genomic coordinates)
    # TSS at end of B, PolyA at start of A
    outputs["splice"]["donor"][150] = 5.0 # up_exon.start on minus strand
    outputs["splice"]["acceptor"][120] = 5.0 # down_exon.end on minus strand
    outputs["tss"]["tss"][170] = 5.0 # "end" of 5' exon
    outputs["polya"]["polya"][100] = 5.0 # "start" of 3' exon
    return outputs


def test_minus_strand_splice_edge_directionality():
    """
    Tests that splice graph edges on the minus strand are built
    in the direction of transcription (from higher to lower genomic coordinates).
    """
    _, strand, _ = toy_minus_gene()
    head_outputs = mock_head_outputs()

    # Use a dummy config for the builder
    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10},
            "allow_unanchored": True
        }
    }
    builder = SpliceGraphBuilder(config)
    graph = builder.build(head_outputs, strand=strand)

    # On '-', transcription flows high->low; edges must reflect that.
    # Exon B (150, 170) should connect to Exon A (100, 120)

    # The builder creates multiple candidate exons. Let's find the ones we expect.
    # First exon: donor -> tss = (150, 170)
    # Last exon: polya -> acceptor = (100, 120)
    # Internal exon: donor -> acceptor = (150, 120) -> this is wrong, should be donor -> acceptor

    # Correct internal exon on minus strand is (donor_pos, acceptor_pos) if donor_pos > acceptor_pos
    # The builder logic is `_get_exons(donor_indices, acceptor_indices, ...)`, which requires start < end.
    # This is the first bug. Let's check the junction logic given the exons it *does* find.

    # It will find a "first" exon (150, 170) and a "last" exon (100, 120)
    # The junction logic sorts by start coordinate (reversed for minus strand).
    # So it will check for a junction from (150, 170) to (100, 120)
    # Junction check on minus strand: `up_exon.start in donor_set and down_exon.end in acceptor_set`
    # Here, up_exon.start=150 (donor), down_exon.end=120 (acceptor). This should work.

    expected_edge = ((150, 170), (100, 120))

    assert expected_edge in graph.graph.edges(), "Edge from high-coordinate exon to low-coordinate exon not found for minus strand."
    assert len(graph.graph.edges()) == 1, "Incorrect number of edges found."


def test_minus_strand_full_decoder_ordering():
    """
    Tests that the full IsoformDecoder correctly orders exons
    for a minus-strand transcript.
    """
    _, strand, _ = toy_minus_gene()
    head_outputs = mock_head_outputs()

    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10},
            "beam_size": 2,
            "scoring": {
                "w_spl": 1.0, "w_tss": 0.0, "w_pa": 0.0, "w_orf": 0.0, "w_len": 0.0,
                "use_orf_head": False
            }
        }
    }
    decoder = IsoformDecoder(config)
    isoforms = decoder.decode(head_outputs, strand=strand)

    assert len(isoforms) > 0, "Decoder failed to produce any isoforms."

    # The top-scoring isoform should have exons ordered by transcription
    # For minus strand, this is high genomic coordinate to low.
    best_isoform = isoforms[0]
    ordered_coords = [(e.start, e.end) for e in best_isoform.exons]

    expected_order = [(150, 170), (100, 120)]

    assert ordered_coords == expected_order, f"Exons not in transcription order. Expected {expected_order}, got {ordered_coords}"


def test_minus_strand_orf_roles_via_scoring():
    """
    Tests that ORF roles (start/stop) are correctly identified on the minus strand,
    verified by checking the ORF score.
    """
    # Sequence: "AAA" (exon 2) "ATG" (intron) "TAG" (exon 1) "AAA"
    # RevComp:  "TTT" "CTA" (revcomp of TAG) "CAT" (revcomp of ATG) "TTT"
    # Exon 1 (150-153): TAG -> revcomp CTA (stop)
    # Exon 2 (100-103): AAA -> revcomp TTT
    # We expect the ORF to be read from a start codon on the revcomp sequence.
    # Let's try a different sequence.
    # Genomic:      ...CCC TGA ... ATG GGG...
    # Coords:           ^100    ^150
    # Exon 1 (150-156): ATG GGG -> revcomp CCC CAT
    # Exon 2 (100-103): TGA     -> revcomp TCA (stop)
    # Transcript:   CCC CAT TCA
    # This transcript has a start (CAT) and a stop (TCA).

    # Let's use a token map where A=0, C=1, G=2, T=3
    # Sequence: "AAACCC" (pos 100-105, exon 2) ... "ATGGGG" (pos 150-155, exon 1)
    input_ids = torch.tensor([[
        0]*100 + [3, 2, 0] + [0]*44 + [0, 3, 2, 2, 2, 2] + [0]*45
    ]) # Puts TGA at 100, ATGGGG at 150

    head_outputs = mock_head_outputs() # Uses peaks at 100,120,150,170
    # Let's adjust head outputs to match our sequence
    head_outputs["splice"]["donor"][150] = 5.0 # Exon 1 start
    head_outputs["splice"]["acceptor"][103] = 5.0 # Exon 2 end
    head_outputs["tss"]["tss"][156] = 5.0
    head_outputs["polya"]["polya"][100] = 5.0

    config = {
        "decoder": {
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 3},
            "beam_size": 2,
            "scoring": {
                "w_spl": 0.1, "w_tss": 0.1, "w_pa": 0.1, "w_orf": 1.0, "w_len": 0.0,
                "use_orf_head": False, # Use sequence-based scorer
                "min_cds_len_aa": 1,
                "orf_gamma": 0.0 # No PTC penalty for this test
            }
        }
    }
    decoder = IsoformDecoder(config)
    # We need to manually tell the builder about the exons we want it to find
    # because the builder logic is complex. We test the scorer here.
    exons = [Exon(start=150, end=156, score=0.9), Exon(start=100, end=103, score=0.9)]
    isoform = Isoform(exons=exons, strand="-")

    # The scorer needs the full head_outputs dict, even if it only uses a part of it.
    score = decoder.scorer(isoform, head_outputs, input_ids=input_ids)

    # A positive score indicates a valid ORF was found.
    assert score > 0, f"Valid ORF on minus strand not detected, score was {score}"


def test_minus_strand_single_exon_cds():
    """Tests scoring for a single-exon CDS on the minus strand."""
    # For a minus strand transcript to have an ATG, the genomic sequence must have a CAT at a
    # higher coordinate than the stop codon's reverse complement (e.g., TTA).
    # The distance between them must be a multiple of 3 for them to be in-frame.
    # Genomic: ... TTA ...... CAT ... (18bp between them)
    # RevComp: ... ATG ...... TAA ... -> Correct ORF
    input_ids = torch.tensor([[0]*50 + [3,3,0] + [0]*18 + [1,0,3] + [0]*27]) # TTA at 50, CAT at 71

    config = { "decoder": { "scoring": { "use_orf_head": False, "min_cds_len_aa": 5 } } }
    isoform = Isoform(exons=[Exon(start=40, end=80)], strand="-")

    head_outputs = {"splice": {"donor": torch.tensor([])}}
    score = _score_orf(isoform, head_outputs, config['decoder']['scoring'], input_ids=input_ids)

    assert score > 0.5, "Valid single-exon CDS on minus strand not scored correctly"


def test_minus_strand_utr_only():
    """Tests that a UTR-only transcript on the minus strand gets a low ORF score."""
    # Sequence with no ATG
    input_ids = torch.tensor([[1,2,3]*50])
    config = { "decoder": { "scoring": { "use_orf_head": False } } }
    isoform = Isoform(exons=[Exon(start=10, end=50), Exon(start=60, end=100)], strand="-")

    head_outputs = {"splice": {"donor": torch.tensor([])}}
    score = _score_orf(isoform, head_outputs, config['decoder']['scoring'], input_ids=input_ids)

    assert score <= 0.0, "UTR-only transcript should not have a positive ORF score"