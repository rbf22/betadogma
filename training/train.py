#!/usr/bin/env python3
"""train.py - Optimized for training 450k sequences on T4 (16GB)."""

import os
import json
import random
import time
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForMaskedLM
import warnings
import yaml
import logging

from config import Config, get_gpu_config, print_memory_breakdown

# Setup logger
logger = logging.getLogger(__name__)
from tokenizer import CharacterTokenizer
from dataset import BetaDogmaDataset
from model.model import BetaDogmaModel
from model.encoder import HyenaDNAEncoder
from model.heads import PredictionHead
from lightning_module import BetaDogmaLightning
from data_module import BetaDogmaDataModule


# Add parent directory to path for local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_level: str = 'INFO'):
    """Configure logging for the training script."""
    # Map string log levels to logging constants
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Suppress verbose loggers from dependencies
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# ============================================================================
# Training
# ============================================================================

def train(
    data_dir: str = None,
    output_dir: str = None,
    max_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    warmup_epochs: int = 1,
    num_workers: int = None,
    accelerator: str = "auto",
    devices: int = None,
    precision: str = None,
    monitor: str = None,
):
    """Train the model with the given configuration."""
    
    # Setup logging first
    setup_logging()
    
    # Load config
    config = Config()
    
    # Override with command line arguments
    if data_dir:
        config.data_dir = Path(data_dir)
    if output_dir:
        config.output_dir = Path(output_dir)
    if max_epochs is not None:
        config.max_epochs = max_epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if weight_decay is not None:
        config.weight_decay = weight_decay
    if num_workers is not None:
        config.num_workers = num_workers
    if precision is not None:
        config.precision = precision
    if monitor is not None:
        config.monitor = monitor
    
    # Auto-detect the best accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        logger.info("🚀 Using CUDA accelerator")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        logger.info("🍎 Using MPS (Apple Silicon) accelerator")
    else:
        accelerator = "cpu"
        logger.info("💻 Using CPU accelerator")
    
    # Log memory optimization strategy
    logger.info(f"Memory optimization: max_seq_len={config.max_seq_len:,}, batch_size={config.batch_size}, accumulate_grad_batches={config.accumulate_grad_batches}, effective_batch_size={config.batch_size * config.accumulate_grad_batches}")
    logger.debug(f"Gradient checkpointing: {config.use_gradient_checkpointing}, Mixed precision: {config.precision}")
    logger.debug(f"Encoder chunking threshold: {config.encoder_chunk_size:,} bp, Prediction chunking threshold: {config.prediction_chunk_size:,} bp")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize data module and model
    data_module = BetaDogmaDataModule(config)
    model = BetaDogmaLightning(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,} ({trainable_params/total_params*100:.1f}%), Frozen={total_params - trainable_params:,}")
    
    # Print memory breakdown (before optimizer)
    print_memory_breakdown(model, optimizer=None)
    
    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        monitor=monitor,
        mode='min',
        save_last=True,
        auto_insert_metric_name=False
    )
    
    # Initialize learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up callbacks
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Only enable early stopping if we have validation data
    if hasattr(data_module, 'val_dataset') and len(data_module.val_dataset) > 0:
        early_stop_callback = EarlyStopping(
            monitor=config.monitor,
            patience=config.patience,
            mode=config.mode,
            verbose=True,
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stop_callback)
    
    tb_logger = TensorBoardLogger(save_dir=config.output_dir, name="logs")
    
    # Configure trainer with proper accelerator selection
    # Auto-detect the best accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        logger.info("🚀 Using CUDA accelerator")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        logger.info("🍎 Using MPS (Apple Silicon) accelerator")
    else:
        accelerator = "cpu"
        logger.info("💻 Using CPU accelerator")
    
    
    # Configure trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,  # Use 1 device for MPS
        precision=config.precision,  # Use mixed precision
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=str(config.output_dir),
        num_sanity_val_steps=0,
        limit_val_batches=config.limit_val_batches,
        limit_train_batches=config.limit_train_batches,
        gradient_clip_algorithm="norm",
    )
    
    
    # PRE-FLIGHT CHECK
    logger.info(f"PRE-FLIGHT CHECK: max_seq_len={config.max_seq_len:,}, prediction_chunk_size={config.prediction_chunk_size:,}, will_chunk={config.max_seq_len > config.prediction_chunk_size}, accelerator={accelerator}, is_mps={accelerator == 'mps'}")
    if accelerator == 'mps' and config.max_seq_len >= 300000:
        logger.info(f"✅ MPS FORCED CHUNKING will be enabled for 300k sequences")
    
    logger.info("Starting training...")
    
    try:
        trainer.fit(model, data_module)
        
        logger.info("✅ Training complete!")
        logger.info(f"Best model: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            logger.info(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        else:
            logger.info("No validation metrics were recorded")
        
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak GPU memory: {peak:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OUT OF MEMORY! Current config: {config.max_seq_len:,} bp. Try reducing max_seq_len in Config")
        raise


if __name__ == "__main__":
    # Load configuration from YAML
    config = Config()
    
    # Start training with the loaded configuration
    train(
        data_dir=str(config.data_dir),
        output_dir=str(config.output_dir),
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_workers=config.num_workers,
        devices=config.devices,
        precision=config.precision,
        monitor=config.monitor
    )