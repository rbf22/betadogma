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

def test_betadogma_model_forward_pass_shapes(monkeypatch):
    """Tests the end-to-end forward pass of the BetaDogmaModel with a mocked Enformer backend."""

    # 1. Mock the enformer-pytorch model loading to avoid network calls
    class MockEnformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_layer = torch.nn.Linear(1, 1)

        def forward(self, one_hot_input, return_embeddings=False):
            batch_size, _, _ = one_hot_input.shape
            dummy_preds = {"human": torch.randn(batch_size, BINNED_SEQ_LEN, 5313)}
            embeddings = torch.randn(batch_size, BINNED_SEQ_LEN, D_IN)

            if return_embeddings:
                return dummy_preds, embeddings
            return dummy_preds

    # The encoder now calls `from_pretrained` from `enformer_pytorch`, imported into the encoder's namespace.
    monkeypatch.setattr("betadogma.core.encoder.from_pretrained", lambda *args, **kwargs: MockEnformer())

    # 2. Create model with a test config that matches the mocked encoder
    config = {
        "encoder": {"model_name": "mock_enformer", "hidden_size": D_IN},
        "heads": {"hidden": D_HIDDEN, "dropout": 0.1, "use_conv": False},
    }
    model = BetaDogmaModel(config)

    # 3. Create dummy input and run forward pass
    dummy_input_ids = torch.randint(0, 4, (BATCH_SIZE, INPUT_SEQ_LEN))
    output = model.forward(dummy_input_ids)

    # 4. Assert output shapes
    assert "splice" in output
    assert "tss" in output
    assert "polya" in output
    assert "orf" in output
    assert "embeddings" in output

    assert output["embeddings"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, D_IN)
    assert output["splice"]["donor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["splice"]["acceptor"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["tss"]["tss"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["polya"]["polya"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["start"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["stop"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 1)
    assert output["orf"]["frame"].shape == (BATCH_SIZE, BINNED_SEQ_LEN, 3)