"""
Unit tests for the SpliceGraphBuilder.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import _find_peaks, SpliceGraphBuilder

def test_find_peaks_basic():
    """Tests the _find_peaks helper function with a simple case."""
    # Probs: [0.1, 0.9, 0.2, 0.8, 0.4] -> sigmoid of below
    logits = torch.tensor([-2.2, 2.2, -1.4, 1.4, -0.4])
    threshold = 0.7

    peak_indices, peak_probs = _find_peaks(logits, threshold)

    assert torch.equal(peak_indices, torch.tensor([1, 3]))
    assert torch.allclose(peak_probs, torch.tensor([0.9, 0.8]), atol=1e-2)

def test_find_peaks_top_k():
    """Tests that _find_peaks correctly applies the top_k limit."""
    # Probs: [0.9, 0.8, 0.95, 0.75]
    logits = torch.tensor([2.2, 1.4, 2.9, 1.1])
    threshold = 0.7

    peak_indices, peak_probs = _find_peaks(logits, threshold, top_k=2)

    # Should return indices of 0.95 and 0.9
    assert set(peak_indices.tolist()) == {0, 2}
    assert torch.allclose(peak_probs.sort(descending=True).values, torch.tensor([0.95, 0.9]), atol=1e-2)

def test_find_peaks_no_peaks():
    """Tests that _find_peaks returns empty tensors when no peaks are found."""
    logits = torch.tensor([-2.0, -1.0, -3.0])
    threshold = 0.9

    peak_indices, peak_probs = _find_peaks(logits, threshold)

    assert peak_indices.numel() == 0
    assert peak_probs.numel() == 0

@pytest.fixture
def sample_head_outputs():
    """Provides a sample dictionary of head outputs for testing."""
    seq_len = 100
    donor_logits = torch.full((1, seq_len, 1), -10.0)
    acceptor_logits = torch.full((1, seq_len, 1), -10.0)
    tss_logits = torch.full((1, seq_len, 1), -10.0)
    polya_logits = torch.full((1, seq_len, 1), -10.0)

    # Donor peaks at 20, 40
    donor_logits[0, 20, 0] = 3.0
    donor_logits[0, 40, 0] = 2.0
    # Acceptor peaks at 10, 30
    acceptor_logits[0, 10, 0] = 3.0
    acceptor_logits[0, 30, 0] = 2.0
    # TSS peak at 5
    tss_logits[0, 5, 0] = 3.0
    # PolyA peak at 50
    polya_logits[0, 50, 0] = 3.0


    return {
        "splice": {
            "donor": donor_logits,
            "acceptor": acceptor_logits,
        },
        "tss": {"tss": tss_logits},
        "polya": {"polya": polya_logits},
    }

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components, allowing unanchored exons."""
    return {
        "decoder": {
            "max_starts": 8,
            "max_ends": 8,
            "allow_unanchored": True,
            "thresholds": {
                "donor": 0.8,
                "acceptor": 0.8,
                "tss": 0.8,
                "polya": 0.8,
            },
            "priors": {
                "min_exon_len": 5,
                "max_intron_len": 100,
            }
        }
    }

def test_splice_graph_builder_positive_strand(sample_head_outputs, decoder_config):
    """Tests that the SpliceGraphBuilder can build a graph on the positive strand."""
    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(sample_head_outputs, strand='+')

    assert graph is not None

    # Expected exons:
    # Internal: (10, 20), (10, 40), (30, 40) -> 3
    # First (TSS->donor): (5, 20), (5, 40) -> 2
    # Last (acceptor->polyA): (10, 50), (30, 50) -> 2
    # Single (TSS->polyA): (5, 50) -> 1
    # Total = 8
    assert graph.graph.number_of_nodes() == 8

    nodes = set(graph.graph.nodes)
    expected_nodes = {(10, 20), (10, 40), (30, 40), (5, 20), (5, 40), (10, 50), (30, 50), (5, 50)}
    assert nodes == expected_nodes

    # Expected junctions (intron len < 100):
    # (10,20) -> (30,40) [intron=10]
    # (5,20) -> (30,40) [intron=10]
    # (5,20) -> (30,50) [intron=10]
    # (10,20) -> (30,50) [intron=10]
    assert graph.graph.number_of_edges() == 4
    assert graph.graph.has_edge((10, 20), (30, 40))


def test_splice_graph_builder_negative_strand(decoder_config):
    """Tests graph construction on the negative strand with appropriate fixtures."""
    # On negative strand, donor < acceptor from a coordinate perspective.
    # A transcript flows from a high coordinate (TSS) to a low one (polyA).
    # A "first" exon is donor -> tss, "last" is polya -> acceptor.
    seq_len = 100
    donor_logits = torch.full((1, seq_len, 1), -10.0)
    acceptor_logits = torch.full((1, seq_len, 1), -10.0)
    tss_logits = torch.full((1, seq_len, 1), -10.0)
    polya_logits = torch.full((1, seq_len, 1), -10.0)

    # Donor peaks at 10, 30
    donor_logits[0, 10, 0] = 3.0
    donor_logits[0, 30, 0] = 2.0
    # Acceptor peaks at 20, 40
    acceptor_logits[0, 20, 0] = 3.0
    acceptor_logits[0, 40, 0] = 2.0
    # TSS at 50, polyA at 5
    tss_logits[0, 50, 0] = 3.0
    polya_logits[0, 5, 0] = 3.0

    head_outputs = {
        "splice": {"donor": donor_logits, "acceptor": acceptor_logits},
        "tss": {"tss": tss_logits},
        "polya": {"polya": polya_logits},
    }

    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(head_outputs, strand='-')

    # Expected exons:
    # Internal (donor->acceptor): (10,20), (10,40), (30,40) -> 3
    # First (donor->tss): (10,50), (30,50) -> 2
    # Last (polya->acceptor): (5,20), (5,40) -> 2
    # Single (polya->tss): (5,50) -> 1
    # Total = 8
    assert graph.graph.number_of_nodes() == 8
    nodes = set(graph.graph.nodes)
    expected_nodes = {(10, 20), (10, 40), (30, 40), (10, 50), (30, 50), (5, 20), (5, 40), (5, 50)}
    assert nodes == expected_nodes

    # Expected junctions on negative strand (from upstream exon to downstream exon)
    # e.g. from (30,40) to (10,20). Intron len = 30-20 = 10.
    assert graph.graph.has_edge((30, 40), (10, 20))
    assert graph.graph.has_edge((30, 50), (10, 20))
    assert graph.graph.has_edge((30, 50), (5, 20))
    # etc... check a few valid connections
    assert graph.graph.number_of_edges() > 0