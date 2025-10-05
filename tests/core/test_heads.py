import pytest
import torch
from betadogma.core.heads import SpliceHead, TSSHead, PolyAHead, ORFHead
from betadogma.model import BetaDogmaModel

# Test parameters updated for Enformer-based model
BATCH_SIZE = 2
INPUT_SEQ_LEN = 196_608  # A typical Enformer input length
BINNED_SEQ_LEN = 896     # Enformer's binned output resolution
D_IN = 3072              # Enformer's embedding dimension
D_HIDDEN = 768

@pytest.fixture
def dummy_embeddings():
    """Create a dummy embeddings tensor with Enformer-like dimensions."""
    return torch.randn(BATCH_SIZE, BINNED_SEQ_LEN, D_IN)

def test_splice_head_shapes(dummy_embeddings):
    """Test the output shapes of the SpliceHead."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "donor" in output
    assert "acceptor" in output
    assert output["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["acceptor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_tss_head_shapes(dummy_embeddings):
    """Test the output shapes of the TSSHead."""
    head = TSSHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "tss" in output
    assert output["tss"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_polya_head_shapes(dummy_embeddings):
    """Test the output shapes of the PolyAHead."""
    head = PolyAHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "polya" in output
    assert output["polya"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_orf_head_shapes(dummy_embeddings):
    """Test the output shapes of the ORFHead."""
    head = ORFHead(d_in=D_IN, d_hidden=D_HIDDEN)
    output = head(dummy_embeddings)
    assert "start" in output
    assert "stop" in output
    assert "frame" in output
    assert output["start"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["stop"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["frame"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 3)

@pytest.mark.parametrize("use_conv", [True, False])
def test_head_architectures(dummy_embeddings, use_conv):
    """Test both convolutional and linear head architectures."""
    head = SpliceHead(d_in=D_IN, d_hidden=D_HIDDEN, use_conv=use_conv)
    output = head(dummy_embeddings)
    assert output["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)

def test_betadogma_model_forward_pass_shapes():
    """Tests the end-to-end forward pass of the BetaDogmaModel with mock embeddings."""

    # 1. Instantiate the model using a dummy config.
    # The model is now decoupled from the encoder.
    config = {
        "heads": {"hidden": D_HIDDEN, "dropout": 0.1, "use_conv": False},
        "decoder": {}  # Not used in forward pass
    }
    model = BetaDogmaModel(d_in=D_IN, config=config)
    model.eval()

    # 2. Create dummy embeddings. The model is agnostic to sequence length.
    dummy_embeddings = torch.randn(BATCH_SIZE, BINNED_SEQ_LEN, D_IN)

    # 3. Run forward pass
    with torch.no_grad():
        output = model(embeddings=dummy_embeddings)

    # 4. Assert output shapes
    assert "splice" in output
    assert "tss" in output
    assert "polya" in output
    assert "orf" in output
    assert "embeddings" in output

    assert torch.equal(output["embeddings"], dummy_embeddings)
    assert output["splice"]["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["splice"]["acceptor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["tss"]["tss"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["polya"]["polya"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["start"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["stop"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["frame"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 3)