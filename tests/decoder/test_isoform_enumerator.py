"""
Unit tests for the IsoformEnumerator and IsoformScorer.
"""
import pytest
import torch
from betadogma.decoder.isoform_decoder import (
    SpliceGraph,
    IsoformEnumerator,
    IsoformScorer,
)
from betadogma.decoder.types import Exon, Isoform

@pytest.fixture
def simple_splice_graph():
    """
    Creates a simple splice graph for testing path enumeration.
    Graph structure:
    A -> B -> D (score: 0.9 + 0.8 + 0.9 = 2.6) -> path score ~0.87
    A -> C -> D (score: 0.9 + 0.7 + 0.9 = 2.5) -> path score ~0.83
    A is source, D is sink.
    """
    graph = SpliceGraph()
    exon_a = Exon(start=10, end=20, score=0.9)
    exon_b = Exon(start=30, end=40, score=0.8)
    exon_c = Exon(start=30, end=45, score=0.7) # Alternative exon
    exon_d = Exon(start=50, end=60, score=0.9)

    graph.add_exon(exon_a)
    graph.add_exon(exon_b)
    graph.add_exon(exon_c)
    graph.add_exon(exon_d)

    graph.add_junction(exon_a, exon_b, score=1.0)
    graph.add_junction(exon_a, exon_c, score=1.0)
    graph.add_junction(exon_b, exon_d, score=1.0)
    graph.add_junction(exon_c, exon_d, score=1.0)

    return graph

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components."""
    return {
        "decoder": {
            "beam_size": 4,
            "max_candidates": 10,
            "scoring": {
                "w_spl": 1.0,
            }
        }
    }

def test_isoform_enumerator_finds_best_path(simple_splice_graph, decoder_config):
    """
    Tests that the beam search enumerator finds the highest-scoring path.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    isoforms = enumerator.enumerate(simple_splice_graph, max_paths=10)

    assert len(isoforms) == 2

    # The best path should be A -> B -> D
    best_isoform = isoforms[0]
    assert best_isoform.exons[0].start == 10
    assert best_isoform.exons[1].start == 30
    assert best_isoform.exons[1].end == 40 # This is exon B
    assert best_isoform.exons[2].start == 50

    # The second best path should be A -> C -> D
    second_isoform = isoforms[1]
    assert second_isoform.exons[1].end == 45 # This is exon C

def test_isoform_enumerator_respects_max_paths(simple_splice_graph, decoder_config):
    """
    Tests that the enumerator returns the correct number of paths.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    isoforms = enumerator.enumerate(simple_splice_graph, max_paths=1)

    assert len(isoforms) == 1
    # Check it's the best one
    assert isoforms[0].exons[1].end == 40

def test_isoform_scorer_basic(decoder_config):
    """
    Tests the IsoformScorer's basic functionality.
    """
    scorer = IsoformScorer(config=decoder_config)

    exon1 = Exon(start=100, end=200, score=0.9)
    exon2 = Exon(start=300, end=400, score=0.7)
    isoform = Isoform(exons=[exon1, exon2], strand="+")

    # The scorer should average the exon scores
    score = scorer(isoform, heads={}) # heads are not used in this version
    assert score.item() == pytest.approx(0.8)

def test_isoform_enumerator_empty_graph(decoder_config):
    """
    Tests that the enumerator handles an empty graph gracefully.
    """
    enumerator = IsoformEnumerator(config=decoder_config)
    empty_graph = SpliceGraph()
    isoforms = enumerator.enumerate(empty_graph, max_paths=10)
    assert len(isoforms) == 0