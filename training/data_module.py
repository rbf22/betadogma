
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional
import logging

from config import Config
from dataset import BetaDogmaDataset
from tokenizer import CharacterTokenizer

# Setup logger
logger = logging.getLogger(__name__)

# ============================================================================
# Data Module
# ============================================================================

class BetaDogmaDataModule(pl.LightningDataModule):
    """Data module."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = CharacterTokenizer(max_length=config.max_seq_len)
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        data_dir = Path(self.config.data_dir)
        
        # Find all parquet files in the data directory subfolders
        train_files = list((data_dir / 'train').glob("*.parquet"))
        val_files = list((data_dir / 'val').glob("*.parquet"))
        test_files = list((data_dir / 'test').glob("*.parquet"))
        
        # Check if we have any data
        if not train_files and not val_files and not test_files:
            raise ValueError(
                f"No training files found in {self.config.data_dir}. "
                f"Expected to find parquet files in {data_dir}/{{train,val,test}}/"
            )
            
        if stage == "fit" or stage is None:
            if not train_files:
                raise ValueError(
                    f"No training files found in {data_dir}/train/. "
                    f"Expected parquet files in {data_dir}/train/"
                )
                
            logger.info(f"Found {len(train_files)} training files")
            self.data_train = BetaDogmaDataset(
                train_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="train",
                augment_prob=0.8,
            )
            
            if not val_files:
                logger.warning(f"No validation files found in {data_dir}/val/. Using training data for validation.")
                val_files = train_files
            
            logger.info(f"Using {len(val_files)} validation files")
            self.data_val = BetaDogmaDataset(
                val_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="val",
            )
            
        if stage == "test" or stage is None:
            if not test_files:
                logger.warning(f"No test files found in {data_dir}/test/. Using validation data for testing.")
                test_files = val_files if val_files else train_files
            
            logger.info(f"Using {len(test_files)} test files")
            self.data_test = BetaDogmaDataset(
                test_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                mode="test",
            )
    
    @staticmethod
    def _worker_init_fn(worker_id):
        """Worker init function for reproducibility."""
        # Get a unique seed for this worker
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        
        # Set seeds for reproducibility
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    def train_dataloader(self):
        if not hasattr(self, 'data_train'):
            self.setup(stage='fit')
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        if not hasattr(self, 'data_val'):
            self.setup(stage='validate')
        return DataLoader(
            self.data_val,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        if not hasattr(self, 'data_test'):
            self.setup(stage='test')
        return DataLoader(
            self.data_test,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
