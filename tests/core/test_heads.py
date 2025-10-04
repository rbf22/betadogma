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

from betadogma.model import BetaDogmaModel

def test_betadogma_model_forward_pass_shapes(monkeypatch):
    """Tests the end-to-end forward pass of the BetaDogmaModel for correct output shapes."""

    # 1. Mock the Hugging Face model loading to avoid network calls
    class MockHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_layer = torch.nn.Linear(1, 1) # Placeholder

        def forward(self, input_ids, attention_mask=None, output_hidden_states=None):
            batch_size, seq_len = input_ids.shape
            last_hidden_state = torch.randn(batch_size, seq_len, D_IN)

            # Mimic the output object of Hugging Face models
            class MockOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            return MockOutput(last_hidden_state)

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *args, **kwargs: MockHFModel())

    # 2. Create model with a test config
    config = {
        "encoder": {"model_name": "mock_model", "hidden_size": D_IN},
        "heads": {"hidden": D_HIDDEN, "dropout": 0.1, "use_conv": False},
    }
    model = BetaDogmaModel(config)

    # 3. Create dummy input and run forward pass
    dummy_input_ids = torch.randint(0, 4, (BATCH_SIZE, SEQ_LEN))
    output = model.forward(dummy_input_ids)

    # 4. Assert output shapes
    assert "splice" in output
    assert "tss" in output
    assert "polya" in output
    assert "orf" in output
    assert "embeddings" in output

    assert output["embeddings"].shape == (BATCH_SIZE, SEQ_LEN, D_IN)
    assert output["splice"]["donor"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["splice"]["acceptor"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["tss"]["tss"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["polya"]["polya"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["orf"]["start"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["orf"]["stop"].shape == (BATCH_SIZE, SEQ_LEN, 1)
    assert output["orf"]["frame"].shape == (BATCH_SIZE, SEQ_LEN, 3)