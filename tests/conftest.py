"""Shared pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and return a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_parquet_file(temp_dir: Path) -> Path:
    """Create a mock parquet file for testing."""
    data = {
        "seq": ["ACGT" * 50] * 10,
        "donor": [0] * 10,
        "acceptor": [0] * 10,
        "tss": [0] * 10,
        "polya": [0] * 10,
    }
    df = pd.DataFrame(data)
    
    # Save to parquet
    parquet_path = temp_dir / "test.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    
    return parquet_path


@pytest.fixture
def mock_config_file(temp_dir: Path) -> Path:
    """Create a mock config file for testing."""
    config = {
        "task": "jsonl",
        "data": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "max_len": 100,
        },
        "model": {
            "type": "TinySeqModel",
            "hidden_size": 64,
        },
        "trainer": {
            "max_epochs": 1,
            "devices": 1,
            "accelerator": "cpu",
        },
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def mock_batch() -> dict[str, torch.Tensor]:
    """Create a mock batch for testing."""
    return {
        "x": torch.randint(0, 6, (4, 100), dtype=torch.long),
        "y": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
