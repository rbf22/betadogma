import pytest
import torch
from betadogma.core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead

# Test parameters
BATCH_SIZE = 2
SEQ_LEN = 100
D_IN = 1536
D_HIDDEN = 768

@pytest.fixture
def dummy_embeddings():
    """Create a dummy embeddings tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_IN)

def test_splice_head_shapes(dummy_embeddings):
    """Test the output shapes of the SpliceHead."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "donor" in output
    assert "acceptor" in output
    assert output["donor"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["acceptor"].shape == (BATCH_SIZE, SEQ_LEN, 1)

def test_tss_head_shapes(dummy_embeddings):
    """Test the output shapes of the TSSHead."""
    head = TSSHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "tss" in output
    assert output["tss"].shape == (BATCH_SIZE, SEQ_LEN, 1)

def test_polya_head_shapes(dummy_embeddings):
    """Test the output shapes of the PolyAHead."""
    head = PolyAHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "polya" in output
    assert output["polya"].shape == (BATCH_SIZE, SEQ_LEN, 1)

def test_orf_head_shapes(dummy_embeddings):
    """Test the output shapes of the ORFHead."""
    head = ORFHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "start" in output
    assert "stop" in output
    assert "frame" in output
    assert output["start"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["stop"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["frame"].shape == (BATCH_SIZE, SEQ_LEN, 3)

@pytest.mark.parametrize("use_conv", [True, False])
def test_head_architectures(dummy_embeddings, use_conv):
    """Test both convolutional and linear head architectures."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN, use_conv=use_conv)
    output = head(dummy_embeddings)
    assert output["donor"].shape == (BATCH_SIZE, SEQ_LEN, 1)