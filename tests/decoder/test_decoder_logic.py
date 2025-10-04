"""
Regression tests for the isoform decoder's internal logic, focusing on
strand-awareness, ORF scoring, and other complex behaviors.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import (
    _get_spliced_cDNA,
    _score_orf,
    SpliceGraphBuilder,
)
from betadogma.decoder.types import Exon, Isoform

# --- Test Fixtures ---

@pytest.fixture
def scoring_config():
    """Provides a standard configuration for the ORF scorer."""
    return {
        "use_orf_head": False,  # Force sequence-based fallback for these tests
        "min_cds_len_aa": 10,
        "max_cds_len_aa": 1000,
        "kozak_bonus": 0.2,
        "orf_gamma": 0.6,  # Penalty for PTC
    }

@pytest.fixture
def mock_head_outputs():
    """A mock output from the model's prediction heads."""
    return {
        "splice": {"donor": torch.randn(2000), "acceptor": torch.randn(2000)},
        "orf": {"start": torch.randn(2000), "stop": torch.randn(2000), "frame": torch.randn(2000, 3)},
    }

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components, shared across test files."""
    return {
        "decoder": {
            "max_candidates": 64, "beam_size": 16,
            "thresholds": {"donor": 0.6, "acceptor": 0.6, "tss": 0.5, "polya": 0.5},
            "priors": {"min_exon_len": 10, "max_intron_len": 10000},
            "max_starts": 8, "max_ends": 8,
        }
    }

# A map for converting token indices to DNA characters
TOKEN_MAP = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


# --- Tests for _get_spliced_cDNA ---

def test_get_spliced_cDNA_positive_strand():
    """Tests cDNA construction for a simple positive-strand transcript."""
    # Sequence: ACGTACGTACGTACGTACGT
    # Exons:    [10:12], [16:18] -> "GT", "AC"
    # Expected: GTAC
    input_ids = torch.tensor([[
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    ]]) # "ACGTACGTACGTACGTACGT"
    isoform = Isoform(exons=[Exon(10, 12), Exon(16, 18)], strand="+")

    cDNA, _ = _get_spliced_cDNA(isoform, input_ids, TOKEN_MAP)
    assert cDNA == "GTAC"

def test_get_spliced_cDNA_negative_strand():
    """Tests that cDNA is correctly reverse-complemented for a negative-strand transcript."""
    # Sequence: ACGTACGTACGTACGTACGT
    # Exons:    [10:12], [16:18] -> "GT", "AC"
    # Spliced:  GTAC
    # RevComp:  GTAC
    input_ids = torch.tensor([[
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    ]]) # "ACGTACGTACGTACGTACGT"
    isoform = Isoform(exons=[Exon(10, 12), Exon(16, 18)], strand="-")

    cDNA, _ = _get_spliced_cDNA(isoform, input_ids, TOKEN_MAP)
    assert cDNA == "GTAC"


# --- Tests for _score_orf (Sequence-based Fallback) ---

def _dna_to_tensor(dna_str: str) -> torch.Tensor:
    """Helper to convert a DNA string to a batched tensor of token indices."""
    rev_token_map = {v: k for k, v in TOKEN_MAP.items()}
    return torch.tensor([[rev_token_map.get(c, 4) for c in dna_str]]).long()


def test_score_orf_sequence_valid(scoring_config, mock_head_outputs):
    """Tests scoring for a standard, valid ORF that meets length requirements."""
    # ORF is 12 codons long, should pass min_cds_len_aa=10 check
    orf = "ATG" + "GGCGGCGGCG" * 3 + "TAA"
    cDNA = "GCC" + orf + "GGC"
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score >= 0.8  # Should get base score (0.5) + length bonus (0.3)

def test_score_orf_ptc_penalty(scoring_config, mock_head_outputs):
    """Tests that a premature termination codon (PTC) is correctly penalized."""
    # Junction is at pos 62. Stop codon TAA at pos 6 is > 55bp upstream.
    exon1_seq = "GCCATGTAA" + "G" * 53 # len=62, stop at pos 6
    exon2_seq = "GCGGCTAGGGC"
    cDNA = exon1_seq + exon2_seq
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, 62), Exon(62, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score < 0.3 # Should be base score (0.5) - PTC penalty (0.6) -> 0

def test_score_orf_kozak_bonus(scoring_config, mock_head_outputs):
    """Tests that a strong Kozak sequence gets a bonus."""
    # An ORF long enough to pass the length filter (15 codons)
    long_orf_codons = "GGC" * 15

    # Strong Kozak: A at -3, G at +4. The G at +4 is the first base of the next codon.
    cDNA_strong = "GCCAGCATG" + long_orf_codons + "TAA"
    input_ids_strong = _dna_to_tensor(cDNA_strong)
    isoform_strong = Isoform(exons=[Exon(0, len(cDNA_strong))], strand="+")

    # Weak Kozak: T at -3
    cDNA_weak = "GCCTCCATG" + long_orf_codons + "TAA"
    input_ids_weak = _dna_to_tensor(cDNA_weak)
    isoform_weak = Isoform(exons=[Exon(0, len(cDNA_weak))], strand="+")

    score_strong = _score_orf(isoform_strong, mock_head_outputs, scoring_config, input_ids_strong)
    score_weak = _score_orf(isoform_weak, mock_head_outputs, scoring_config, input_ids_weak)

    # Strong score should be base (0.5) + length (0.3) + kozak (0.2) = 1.0
    # Weak score should be base (0.5) + length (0.3) = 0.8
    assert score_strong > score_weak
    assert score_strong.item() == pytest.approx(1.0)
    assert score_weak.item() == pytest.approx(0.8)

def test_score_orf_no_valid_orf(scoring_config, mock_head_outputs):
    """Tests that a sequence with no valid ORF gets a score of 0."""
    cDNA = "GCCTTTGGCCGGCGC" # No ATG or no stop
    input_ids = _dna_to_tensor(cDNA)
    isoform = Isoform(exons=[Exon(0, len(cDNA))], strand="+")

    score = _score_orf(isoform, mock_head_outputs, scoring_config, input_ids)
    assert score == 0.0

# --- Test for SpliceGraphBuilder ---

def test_splice_graph_builder_negative_strand(decoder_config):
    """
    Tests that the SpliceGraphBuilder correctly connects exons
    for a negative-strand transcript.
    """
    # Mock model outputs
    # Donor at 800, Acceptor at 500 (transcriptional order)
    # TSS at 900, PolyA at 400
    # Exon 1: 800-900 (Donor to TSS), Exon 2: 400-500 (PolyA to Acceptor)
    # Expected junction: from Exon 1 to Exon 2
    head_outputs = {
        "splice": {
            "donor": torch.full((1000,), -10.0),
            "acceptor": torch.full((1000,), -10.0),
        },
        "tss": {"tss": torch.full((1000,), -10.0)},
        "polya": {"polya": torch.full((1000,), -10.0)},
    }
    head_outputs["splice"]["donor"][800] = 5.0
    head_outputs["splice"]["acceptor"][500] = 5.0
    head_outputs["tss"]["tss"][900] = 5.0
    head_outputs["polya"]["polya"][400] = 5.0

    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(head_outputs, strand="-")

    # Check that the two exons were created as nodes
    # Note: on negative strand, a "first" exon is (donor, tss)
    # and a "last" exon is (polya, acceptor)
    assert (800, 900) in graph.graph.nodes
    assert (400, 500) in graph.graph.nodes

    # Check that a directed edge exists from the upstream to downstream exon
    assert graph.graph.has_edge((800, 900), (400, 500))
    # Verify no edge in the wrong direction
    assert not graph.graph.has_edge((400, 500), (800, 900))