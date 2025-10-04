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
    # Mock logits for a sequence of length 50
    donor_logits = torch.full((1, 50, 1), -5.0)
    acceptor_logits = torch.full((1, 50, 1), -5.0)

    # Donor peaks at 20 (prob ~0.95), 40 (prob ~0.88)
    donor_logits[0, 20, 0] = 3.0
    donor_logits[0, 40, 0] = 2.0
    # Acceptor peaks at 10 (prob ~0.95), 30 (prob ~0.88)
    acceptor_logits[0, 10, 0] = 3.0
    acceptor_logits[0, 30, 0] = 2.0

    return {
        "splice": {
            "donor": donor_logits,
            "acceptor": acceptor_logits,
        }
    }

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components."""
    return {
        "decoder": {
            "max_candidates": 64,
            "thresholds": {
                "donor": 0.8,
                "acceptor": 0.8,
            },
            "priors": {
                "min_exon_len": 5,
                "max_intron_len": 100,
            }
        }
    }

def test_splice_graph_builder_positive_strand(sample_head_outputs, decoder_config):
    """Tests that the SpliceGraphBuilder can build a basic graph on the positive strand."""
    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(sample_head_outputs, strand='+')

    assert graph is not None
    # Expected exons: (10, 20), (10, 40), (30, 40)
    assert graph.graph.number_of_nodes() == 3

    nodes = set(graph.graph.nodes)
    assert (10, 20) in nodes
    assert (10, 40) in nodes
    assert (30, 40) in nodes

    # Expected junctions: (10,20) -> (30,40). Intron length = 30-20=10. Valid.
    assert graph.graph.number_of_edges() == 1
    assert graph.graph.has_edge((10, 20), (30, 40))

def test_splice_graph_builder_negative_strand(sample_head_outputs, decoder_config):
    """Tests graph construction on the negative strand."""
    builder = SpliceGraphBuilder(config=decoder_config)
    graph = builder.build(sample_head_outputs, strand='-')

    # On negative strand, we pair donor -> acceptor (don_coord < acc_coord)
    # Donors at 20, 40. Acceptors at 10, 30.
    # Only valid pair is donor at 20, acceptor at 30. This is an error in my fixture.
    # The acceptor must be at a higher coordinate.
    # Let's adjust the fixture for a better negative strand test.
    donor_logits = torch.full((1, 50, 1), -5.0)
    acceptor_logits = torch.full((1, 50, 1), -5.0)
    donor_logits[0, 10, 0] = 3.0  # Donor at 10
    donor_logits[0, 30, 0] = 2.0  # Donor at 30
    acceptor_logits[0, 20, 0] = 3.0 # Acceptor at 20
    acceptor_logits[0, 40, 0] = 2.0 # Acceptor at 40
    head_outputs = {"splice": {"donor": donor_logits, "acceptor": acceptor_logits}}

    graph = builder.build(head_outputs, strand='-')

    # Expected exons: (10, 20), (10, 40), (30, 40)
    assert graph.graph.number_of_nodes() == 3
    nodes = set(graph.graph.nodes)
    assert (10, 20) in nodes
    assert (10, 40) in nodes
    assert (30, 40) in nodes

    # Expected junction: (10,20) -> (30,40). Intron length = 30-20=10. Valid.
    assert graph.graph.number_of_edges() == 1
    assert graph.graph.has_edge((10, 20), (30, 40))