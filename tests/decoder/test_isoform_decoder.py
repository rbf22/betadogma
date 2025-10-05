"""
Unit tests for the isoform decoder scaffolding.
"""
import pytest

from betadogma.decoder.isoform_decoder import (
    SpliceGraphBuilder,
    IsoformEnumerator,
    IsoformScorer,
    IsoformDecoder,
)
from betadogma.decoder.types import Exon, Isoform

@pytest.fixture
def decoder_config():
    """A mock config for the decoder components."""
    return {
        "decoder": {
            "max_candidates": 64,
            "beam_size": 16,
            "thresholds": {
                "donor": 0.6,
                "acceptor": 0.6,
                "tss": 0.5,
                "polya": 0.5,
            },
        }
    }

def test_exon_creation():
    """Tests that the Exon data class can be created."""
    exon = Exon(start=100, end=200, score=0.9)
    assert exon.start == 100
    assert exon.end == 200
    assert exon.score == 0.9

def test_isoform_creation():
    """Tests that the Isoform data class can be created."""
    exon1 = Exon(start=100, end=200)
    exon2 = Exon(start=300, end=400)
    isoform = Isoform(exons=[exon1, exon2], strand="+", score=0.85)
    assert len(isoform.exons) == 2
    assert isoform.strand == "+"
    assert isoform.start == 100
    assert isoform.end == 400

def test_splice_graph_builder_init(decoder_config):
    """Tests that SpliceGraphBuilder can be instantiated."""
    builder = SpliceGraphBuilder(config=decoder_config["decoder"])
    assert builder is not None
    assert builder.config == decoder_config["decoder"]

def test_isoform_enumerator_init(decoder_config):
    """Tests that IsoformEnumerator can be instantiated."""
    enumerator = IsoformEnumerator(config=decoder_config["decoder"])
    assert enumerator is not None
    assert enumerator.config == decoder_config["decoder"]

def test_isoform_scorer_init(decoder_config):
    """Tests that IsoformScorer can be instantiated."""
    scorer = IsoformScorer(config=decoder_config["decoder"])
    assert scorer is not None
    assert scorer.config == decoder_config["decoder"]

def test_isoform_decoder_init(decoder_config):
    """Tests that the main IsoformDecoder can be instantiated."""
    decoder = IsoformDecoder(config=decoder_config)
    assert decoder is not None
    assert isinstance(decoder.graph_builder, SpliceGraphBuilder)
    assert isinstance(decoder.enumerator, IsoformEnumerator)
    assert isinstance(decoder.scorer, IsoformScorer)