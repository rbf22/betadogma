"""Tests for the training module."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from hypothesis import given, strategies as st
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the train module
import train.train as train_module

# Import individual components from the train module
from train.train import (
    LitSeq,
    TinySeqModel,
    StructuralParquetDataset,
    encode_seq,
    load_config,
    main,
    revcomp,
    set_seed,
    structural_collate,
)

# Make the train module available as train for any direct attribute access
train = train_module


@pytest.mark.parametrize("config_data,expected", [
    ({"model": {"type": "another", "hidden_size": 128}}, "another"),
])
def test_load_config(config_data: Dict[str, Any], expected: str, tmp_path: Path):
    """Test loading configuration from YAML files with different configurations."""
    # Add test data to the config
    config_data["data"] = {"path": "/test/path"}  # Add required data config
    config_data["model"] = config_data.get("model", {"type": "test"})
    config_data["trainer"] = config_data.get("trainer", {})  # Add empty trainer config
    
    # Create a temporary config file
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    # Test loading the config
    loaded_config = load_config(str(config_path))
    
    # Verify the loaded config
    assert loaded_config["model"]["type"] == expected, \
        f"Expected model type '{expected}', got '{loaded_config['model']['type']}'"
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yaml"))

@pytest.mark.parametrize("seed", [42, 123, 0, 999])
def test_set_seed(seed: int):
    """Test that setting random seeds produces deterministic results."""
    set_seed(seed)
    
    # Generate some random numbers and check they're the same when seed is set
    torch_rand1 = torch.rand(10)
    np_rand1 = np.random.rand(10)
    
    # Set seed again and generate more random numbers
    set_seed(seed)
    torch_rand2 = torch.rand(10)
    np_rand2 = np.random.rand(10)
    
    # Should be the same with the same seed
    assert torch.allclose(torch_rand1, torch_rand2)
    assert np.allclose(np_rand1, np_rand2)


# DNA test cases
DNA_TEST_CASES = [
    ("ATCG", "CGAT"),
    ("ACGT", "ACGT"),  # Self-reverse complement
    ("GCTA", "TAGC"),
    ("NNNN", "NNNN"),
    ("", ""),
]

# Parametrized test for revcomp
@pytest.mark.parametrize("seq,expected", DNA_TEST_CASES)
def test_revcomp(seq: str, expected: str):
    """Test DNA reverse complementation."""
    assert revcomp(seq) == expected

# Property-based test for revcomp
@given(st.text(alphabet="ACGTNacgtn", min_size=0, max_size=100))
def test_revcomp_properties(seq: str):
    """Test properties of reverse complementation."""
    # Double reverse complement should return the original (uppercased)
    assert revcomp(revcomp(seq)) == seq.upper()
    
    # Length should be preserved
    assert len(revcomp(seq)) == len(seq)

# Test cases for sequence encoding
ENCODE_TEST_CASES = [
    ("ACGT", 10, [1, 2, 3, 4] + [0] * 6),  # ACGT encoding with padding
    ("ACGT", 2, [1, 2]),  # Truncation
    ("NNNN", 4, [5, 5, 5, 5]),  # N's should map to 5
    ("", 4, [0, 0, 0, 0]),  # Empty sequence
]

@pytest.mark.parametrize("seq,max_len,expected", ENCODE_TEST_CASES)
def test_encode_seq(seq: str, max_len: int, expected: List[int]):
    """Test DNA sequence encoding."""
    encoded = encode_seq(seq, max_len=max_len)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dtype == torch.long
    assert len(encoded) == max_len
    assert torch.all(encoded == torch.tensor(expected))


@pytest.mark.parametrize("batch_size,seq_len,embed_dim,hidden", [
    (1, 100, 16, 32),
    (4, 200, 32, 64),
    (8, 50, 64, 128),
])
def test_tiny_seq_model_forward(batch_size: int, seq_len: int, embed_dim: int, hidden: int):
    """Test TinySeqModel forward pass with different configurations."""
    model = TinySeqModel(vocab_size=6, embed_dim=embed_dim, hidden=hidden)
    x = torch.randint(0, 6, (batch_size, seq_len), dtype=torch.long)
    
    # Test forward pass
    out = model(x)
    assert out.shape == (batch_size, 1)  # Binary classification
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_tiny_seq_model_save_load(tmp_path: Path):
    """Test saving and loading TinySeqModel weights."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create and save model
    model1 = TinySeqModel(vocab_size=6, embed_dim=16, hidden=32)
    model_path = tmp_path / "model.pt"
    torch.save(model1.state_dict(), model_path)

    # Create a new model and load weights
    model2 = TinySeqModel(vocab_size=6, embed_dim=16, hidden=32)
    model2.load_state_dict(torch.load(model_path))

    # Test forward pass gives same results
    x = torch.randint(0, 6, (2, 100), dtype=torch.long)
    
    # Set models to eval mode to disable dropout
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    # Use a small tolerance for floating point comparison
    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-4), \
        f"Outputs differ by {torch.max(torch.abs(out1 - out2)):g}"


class TestLitSeq:
    @pytest.fixture
    def trainer(self) -> MagicMock:
        """Fixture to create a mock trainer for testing."""
        trainer = MagicMock()
        # Mock the logger and its log_metrics method
        trainer.logger = MagicMock()
        trainer.logger.log_metrics = MagicMock()
        return trainer
    
    @pytest.fixture
    def lit_model(self, trainer: MagicMock) -> LitSeq:
        """Fixture to create a LitSeq instance for testing with a mock trainer."""
        model = TinySeqModel()
        lit_model = LitSeq(model=model, lr=1e-3, weight_decay=1e-4)
        # Attach the mock trainer to the model
        lit_model.trainer = trainer
        return lit_model
    
    def test_init(self, lit_model: LitSeq):
        """Test LitSeq initialization."""
        assert lit_model.lr == 1e-3
        assert lit_model.weight_decay == 1e-4
        assert isinstance(lit_model.loss_fn, torch.nn.BCEWithLogitsLoss)
        assert hasattr(lit_model, 'auroc')
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_forward(self, lit_model: LitSeq, batch_size: int):
        """Test forward pass with different batch sizes."""
        x = torch.randint(0, 6, (batch_size, 100), dtype=torch.long)
        out = lit_model(x)
        assert out.shape == (batch_size, 1)
    
    def test_training_step(self, lit_model: LitSeq, mock_batch: Dict[str, torch.Tensor], trainer: MagicMock):
        """Test training step with mock batch."""
        # Patch the log method to track calls
        with patch.object(lit_model, 'log') as mock_log:
            # Test training step
            loss = lit_model.training_step(mock_batch, 0)
            
            # Verify the output
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar loss
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            
            # Verify logging was called via self.log()
            # Check that log was called at least once
            assert mock_log.call_count > 0
            
            # Check that it was called with the expected arguments
            # The actual arguments might vary based on the implementation
            found_loss_log = False
            for call in mock_log.call_args_list:
                args, kwargs = call
                if 'loss' in kwargs or any('loss' in str(arg) for arg in args):
                    found_loss_log = True
                    break
            assert found_loss_log, "Expected a log call with 'loss' in arguments"
    
    def test_validation_step(self, lit_model: LitSeq, mock_batch: Dict[str, torch.Tensor], trainer: MagicMock):
        """Test validation step with mock batch."""
        # Patch the log method to track calls
        with patch.object(lit_model, 'log') as mock_log:
            # Test validation step
            loss = lit_model.validation_step(mock_batch, 0)
            
            # Verify the output
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar loss
            
            # Verify logging was called via self.log()
            # Check that log was called at least once
            assert mock_log.call_count > 0
            
            # Check that it was called with the expected arguments
            # The actual arguments might vary based on the implementation
            found_loss_log = False
            for call in mock_log.call_args_list:
                args, kwargs = call
                if 'loss' in kwargs or any('loss' in str(arg) for arg in args):
                    found_loss_log = True
                    break
            assert found_loss_log, "Expected a log call with 'loss' in arguments"
    
    def test_configure_optimizers(self, lit_model: LitSeq):
        """Test optimizer configuration."""
        optimizer = lit_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        
        # Check learning rate and weight decay
        param_groups = optimizer.param_groups[0]
        assert param_groups['lr'] == pytest.approx(1e-3)
        assert param_groups['weight_decay'] == pytest.approx(1e-4)


class TestStructuralParquetDataset:
    def test_init(self, mock_parquet_file: Path):
        """Test dataset initialization with a mock parquet file."""
        dataset = StructuralParquetDataset([mock_parquet_file])
        assert len(dataset) == 10
        
        # Test __getitem__
        item = dataset[0]
        assert isinstance(item, dict)
        assert "seq" in item
        assert "donor" in item
        assert "acceptor" in item
        assert "tss" in item
        assert "polya" in item
    
    def test_missing_columns(self, tmp_path: Path):
        """Test error handling for missing required columns."""
        # Create a parquet file with missing columns
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Missing 'acceptor' column
        data = {
            "seq": ["ACGT" * 50] * 5,
            "donor": [0] * 5,
            "tss": [0] * 5,
            "polya": [0] * 5,
        }
        
        bad_path = tmp_path / "bad.parquet"
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, bad_path)
        
        with pytest.raises(ValueError) as excinfo:
            StructuralParquetDataset([bad_path])
        assert "Missing columns" in str(excinfo.value)
    
    def test_empty_file(self, tmp_path: Path):
        """Test behavior with an empty parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create an empty parquet file
        empty_path = tmp_path / "empty.parquet"
        table = pa.Table.from_arrays([], names=[])
        pq.write_table(table, empty_path)
        
        with pytest.raises(ValueError) as excinfo:
            StructuralParquetDataset([empty_path])
        assert "Missing columns" in str(excinfo.value)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_structural_collate(batch_size: int):
    """Test collate function with different batch sizes."""
    # Create a mock batch with sequence data as expected by structural_collate
    seq_length = 200  # Fixed sequence length for testing
    batch = []
    for i in range(batch_size):
        # Create sequence-like data with the expected structure
        batch.append({
            "seq": "ACGT" * 50,  # 200 bases
            "donor": torch.tensor([float(i % 2)] * seq_length, dtype=torch.float32),
            "acceptor": torch.tensor([float((i + 1) % 2)] * seq_length, dtype=torch.float32),
            "tss": torch.tensor([float(i % 3 == 0)] * seq_length, dtype=torch.float32),
            "polya": torch.tensor([float(i % 4 == 0)] * seq_length, dtype=torch.float32),
        })
    
    # Test collation
    collated = structural_collate(batch)
    
    # Check all required keys are present
    required_keys = {"seqs", "donor", "acceptor", "tss", "polya"}
    assert set(collated.keys()) >= required_keys
    
    # Check tensor shapes
    assert isinstance(collated["seqs"], list) and len(collated["seqs"]) == batch_size
    assert collated["donor"].shape == (batch_size, seq_length)
    assert collated["acceptor"].shape == (batch_size, seq_length)
    assert collated["tss"].shape == (batch_size, seq_length)
    assert collated["polya"].shape == (batch_size, seq_length)
    assert len(collated["seqs"]) == batch_size
    assert all(isinstance(s, str) for s in collated["seqs"])
    
    # Check values
    for i, item in enumerate(batch):
        assert collated["seqs"][i] == item["seq"]
        # Note: The original test was comparing tensors with scalars, which would fail.
        # We'll just check the first element of each tensor for simplicity.
        assert torch.allclose(collated["donor"][i], item["donor"][0].repeat(seq_length))
        assert torch.allclose(collated["acceptor"][i], item["acceptor"][0].repeat(seq_length))
        assert torch.allclose(collated["tss"][i], item["tss"][0].repeat(seq_length))
        assert torch.allclose(collated["polya"][i], item["polya"][0].repeat(seq_length))


class TestMain:
    """Tests for the main training function."""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a temporary config file for testing."""
        config = {
            "task": "jsonl",
            "data": {
                "train": "train.jsonl",
                "val": "val.jsonl",
                "batch_size": 32,
                "max_len": 1000,
            },
            "model": {
                "type": "tiny",
                "vocab_size": 6,
                "embed_dim": 64,
                "hidden": 128,
                "dropout": 0.1,
            },
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
            },
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config_path

    @patch("pytorch_lightning.Trainer.fit")
    @patch("train.train.build_trainer")
    @patch("train.train.build_model")
    @patch("train.train.SeqDataModule")
    def test_main_jsonl_task(
        self,
        mock_datamodule: MagicMock,
        mock_build_model: MagicMock,
        mock_build_trainer: MagicMock,
        mock_fit: MagicMock,
        mock_config_file: Path,
    ):
        """Test main function with JSONL task."""
        # Mock the command line arguments
        with patch("sys.argv", ["train.py", str(mock_config_file)]):
            # Mock the datamodule and trainer
            mock_dm = MagicMock()
            mock_trainer = MagicMock()
            
            # Create a real model for testing
            from train.train import TinySeqModel
            model = TinySeqModel(vocab_size=6, embed_dim=64, hidden=128, dropout=0.1)
            
            # The build_model function should return the model, not a LitSeq instance
            mock_build_model.return_value = model
            mock_datamodule.return_value = mock_dm
            mock_build_trainer.return_value = mock_trainer
            
            # Mock the config file loading
            with patch('train.train.load_config') as mock_load_config:
                with open(mock_config_file) as f:
                    config = yaml.safe_load(f)
                # Add the _config_dir that's expected by the code
                config['_config_dir'] = str(mock_config_file.parent)
                mock_load_config.return_value = config
                
                # Run main
                main()
            
            # Verify the mocks were called
            mock_datamodule.assert_called_once()
            mock_build_model.assert_called_once()
            mock_build_trainer.assert_called_once()
            
            # Get the actual model that was passed to fit
            args, kwargs = mock_trainer.fit.call_args
            lit_model_arg = args[0] if args else kwargs.get('model')
            
            # Verify the model is a LitSeq instance with the correct model inside
            assert lit_model_arg is not None
            assert hasattr(lit_model_arg, 'model')
            assert isinstance(lit_model_arg.model, type(model))
            
            # Verify the datamodule was passed correctly
            assert kwargs.get('datamodule') is mock_dm
    
    @patch("pytorch_lightning.Trainer.fit")
    @patch("train.train.build_trainer")
    @patch("train.train.StructuralDataModule")
    @patch("betadogma.model.BetaDogmaModel")
    @patch("betadogma.core.encoder_nt.NTEncoder")
    def test_main_structural_task(
        self,
        mock_nt_encoder: MagicMock,
        mock_betadogma_model: MagicMock,
        mock_datamodule: MagicMock,
        mock_build_trainer: MagicMock,
        mock_fit: MagicMock,
        tmp_path: Path,
    ):
        """Test main function with structural task."""
        # Create a config for structural task
        config = {
            "task": "structural",
            "data": {
                "train": ["train.parquet"],
                "val": ["val.parquet"],
                "batch_size": 32,
            },
            "model": {
                "encoder": {
                    "model_id": "local-model",  # This will use the mocked NTEncoder
                    "hidden_size": 64,
                },
                "heads": {
                    "splice_donor": {
                        "type": "binary",
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1
                    },
                    "splice_acceptor": {
                        "type": "binary",
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1
                    },
                    "tss": {
                        "type": "binary",
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1
                    },
                    "polya": {
                        "type": "binary",
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1
                    }
                },
                "loss": {
                    "pos_weight": 1.0,
                },
            },
            "trainer": {
                "max_epochs": 1,
                "devices": 1,
                "accelerator": "cpu",
            },
            "_config_dir": str(tmp_path),  # Add _config_dir to match expected structure
        }
        
        # Create a temporary config file
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create mock model and encoder instances
        mock_model = MagicMock()
        mock_betadogma_model.return_value = mock_model
        mock_nt_encoder.return_value = MagicMock()
        
        # Mock the datamodule and trainer
        mock_dm = MagicMock()
        mock_datamodule.return_value = mock_dm
        
        mock_trainer = MagicMock()
        mock_build_trainer.return_value = mock_trainer
        
        # Mock the command line arguments and config loading
        with patch("sys.argv", ["train.py", str(config_path)]):
            with patch('train.train.load_config') as mock_load_config:
                # Set up the mock to return our config
                mock_load_config.return_value = config
                
                # Run main
                main()
        
        # Verify the mocks were called
        mock_datamodule.assert_called_once()
        mock_build_trainer.assert_called_once()
        mock_betadogma_model.assert_called_once()
        
        # Verify the trainer's fit method was called with the correct arguments
        mock_trainer.fit.assert_called_once()
        
        # Get the arguments passed to the trainer's fit method
        args, kwargs = mock_trainer.fit.call_args
        
        # Verify the model and datamodule were passed to fit
        assert len(args) > 0
        assert hasattr(args[0], 'model')  # Should be a LitStructural instance
        assert kwargs.get('datamodule') is mock_dm
