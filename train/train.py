#!/usr/bin/env python
# -*- coding: utf-8 -
"""Config-only training entrypoint (lives inside train/).

Now supports two tasks:
  - task: "structural"  -> trains Betadogma heads on Parquet shard from prepare_gencode.py
  - task: "jsonl"       -> legacy JSONL binary classifier (Tiny model or user-supplied)
It resolves paths in the config relative to the config file, and uses PyTorch Lightning
for logging and checkpointing.
"""
from __future__ import annotations

# Standard library imports
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Optional, Union

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC

# Local imports
from betadogma.data.dataset import JsonlSeqDataset, collate_structural_batch as collate_batch

def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file (string or Path object)
        
    Returns:
        Dictionary containing the configuration
    """
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_dir"] = str(path.parent)   # keep for relative path resolution
    return cfg

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- DNA utils (JSONL/legacy path) ----------
DNA_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
DNA_COMP = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})

def revcomp(seq: str) -> str:
    return seq.upper().translate(DNA_COMP)[::-1]

def encode_seq(seq: str, max_len: int, pad_value: int = 0) -> torch.LongTensor:
    s = seq.upper()
    ids = [DNA_VOCAB.get(ch, DNA_VOCAB["N"]) for ch in s[:max_len]]
    if len(ids) < max_len:
        ids += [pad_value] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ---------- Tiny fallback model (legacy) ----------
class TinySeqModel(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=64, hidden=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_long: torch.LongTensor) -> torch.Tensor:
        emb = self.embed(x_long).permute(0, 2, 1)  # [B,E,L]
        feat = self.encoder(emb)                    # [B,H,1]
        logits = self.head(feat)                    # [B,1]
        return logits

# ---------- LightningModule: JSONL ----------
class LitSeq(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC()

    def forward(self, x_long: torch.LongTensor) -> torch.Tensor:
        return self.model(x_long)

    def _shared_step(self, batch, stage: str):
        x, y = batch["x"].long(), batch["y"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.auroc.update(probs, y.int())
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=(stage != "train"), batch_size=x.size(0))
        return loss

    def training_step(self, batch, _):  return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        try:
            au = self.auroc.compute()
        except Exception:
            au = torch.tensor(0.0)
        self.log("val/auroc", au, prog_bar=True)
        self.auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# ===================== NEW: STRUCTURAL TRAINING PATH =====================

# Data: structural shards (Parquet)
class StructuralParquetDataset(Dataset):
    """
    Each row has: chrom, start, end, seq, bin_size, donor, acceptor, tss, polya
    """
    def __init__(self, paths: list[Path]):
        import pandas as pd
        self._rows: list[dict[str, Any]] = []
        for p in paths:
            df = pd.read_parquet(p)
            req = {"seq", "donor", "acceptor", "tss", "polya"}
            miss = req - set(df.columns)
            if miss:
                raise ValueError(f"Missing columns in {p}: {miss}")
            self._rows.extend(df.to_dict("records"))

    def __len__(self): return len(self._rows)

    def __getitem__(self, idx):
        r = self._rows[idx]
        return {
            "seq": r["seq"],
            "donor": torch.tensor(r["donor"], dtype=torch.float32),
            "acceptor": torch.tensor(r["acceptor"], dtype=torch.float32),
            "tss": torch.tensor(r["tss"], dtype=torch.float32),
            "polya": torch.tensor(r["polya"], dtype=torch.float32),
        }

def structural_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch.nn.functional as F
    max_Lr = max(len(b["donor"]) for b in batch)

    def pad1d(x, L): return F.pad(x, (0, L - len(x)))

    seqs = [b["seq"] for b in batch]
    donor   = torch.stack([pad1d(b["donor"],   max_Lr) for b in batch])
    acceptor= torch.stack([pad1d(b["acceptor"],max_Lr) for b in batch])
    tss     = torch.stack([pad1d(b["tss"],     max_Lr) for b in batch])
    polya   = torch.stack([pad1d(b["polya"],   max_Lr) for b in batch])
    return {"seqs": seqs, "donor": donor, "acceptor": acceptor, "tss": tss, "polya": polya}

# LightningModule: Structural (NTEncoder + BetaDogmaModel)
class LitStructural(pl.LightningModule):
    def __init__(self, model_cfg: dict[str, Any], lr: float, weight_decay: float):
        super().__init__()
        from betadogma.model import BetaDogmaModel
        from betadogma.core.encoder_nt import NTEncoder

        self.cfg = model_cfg
        d_in = int(model_cfg["encoder"]["hidden_size"])
        
        # Keep encoder on CPU to avoid MPS device issues
        self.encoder = NTEncoder(model_id=model_cfg["encoder"].get("model_id") or
                                 "InstaDeepAI/nucleotide-transformer-500m-human-ref")
        
        # Ensure encoder stays on CPU
        if hasattr(self.encoder, 'model'):
            self.encoder.model = self.encoder.model.cpu()
            for param in self.encoder.model.parameters():
                param.requires_grad = False
            self.encoder.model.eval()
            print("[INFO] Encoder kept on CPU and frozen")
        
        self.model = BetaDogmaModel(d_in=d_in, config=model_cfg)

        pos_w = torch.tensor(model_cfg["loss"]["pos_weight"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='none')
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters({"lr": self.lr, "weight_decay": self.weight_decay})

    def _get_embeddings(self, seqs: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Extract embeddings tensor from encoder output (encoder runs on CPU).
        
        Args:
            seqs: List of DNA sequences to encode or a tensor of token IDs.
                If a tensor is provided, it should be of shape [batch_size, seq_len].
                
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size] containing the embeddings.
            
        Raises:
            KeyError: If encoder returns a dict with unexpected keys.
            TypeError: If encoder returns an unexpected type.
            RuntimeError: If the input tensor cannot be processed.
        """
        # Handle case where seqs is already a tensor
        if isinstance(seqs, torch.Tensor):
            if seqs.dim() == 1:  # [batch_size * seq_len]
                seqs = seqs.view(-1, seqs.size(0))  # Reshape to [1, seq_len]
            elif seqs.dim() > 2:
                raise ValueError(f"Expected 1D or 2D tensor, got {seqs.dim()}D")
            
            # Convert tensor to list of strings if needed
            # This is a simplified version - adjust based on your tokenization scheme
            try:
                seqs = [''.join([str(x.item()) for x in seq if x != 0]) for seq in seqs]
            except Exception as e:
                raise RuntimeError(f"Failed to convert tensor to sequences: {e}")
        
        # Run encoder on CPU
        with torch.no_grad():
            encoder_output = self.encoder.forward(seqs)
        
        # Handle different return types with type hints
        emb: torch.Tensor
        if isinstance(encoder_output, torch.Tensor):
            emb = encoder_output
        elif isinstance(encoder_output, dict):
            # Try common keys in order of preference
            if "embeddings" in encoder_output:
                emb = encoder_output["embeddings"]
            elif "last_hidden_state" in encoder_output:
                emb = encoder_output["last_hidden_state"]
            elif "hidden_states" in encoder_output:
                emb = encoder_output["hidden_states"]
                if isinstance(emb, (list, tuple)):
                    emb = emb[-1]  # Use last layer's hidden state
            else:
                raise KeyError(f"Encoder returned dict with unexpected keys: {list(encoder_output.keys())}")
        elif hasattr(encoder_output, "last_hidden_state"):
            emb = encoder_output.last_hidden_state
        elif hasattr(encoder_output, "embeddings"):
            emb = encoder_output.embeddings
        else:
            raise TypeError(f"Unexpected encoder output type: {type(encoder_output).__name__}")
            
        # Ensure we have a tensor
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"Expected tensor output, got {type(emb).__name__}")
        
        # Move embeddings from CPU to training device (MPS/GPU)
        return emb.to(self.device)

    def _compute_loss(
        self, 
        outputs: Dict[str, Dict[str, torch.Tensor]], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total loss and individual component losses.
        
        This method computes the loss for each task (donor, acceptor, TSS, polyA) and combines
        them into a total loss with task-specific weights.
        
        Args:
            outputs: Dictionary containing model outputs for each task.
                Expected structure:
                {
                    'splice': {'donor': Tensor[B, L, 1], 'acceptor': Tensor[B, L, 1]},
                    'tss': {'tss': Tensor[B, L, 1]},
                    'polya': {'polya': Tensor[B, L, 1]}
                }
            batch: Dictionary containing ground truth labels.
                Expected keys: 'donor', 'acceptor', 'tss', 'polya', each Tensor[B, L]
                
        Returns:
            A tuple of (total_loss, logs_dict) where:
            - total_loss: The weighted sum of all task losses
            - logs_dict: Dictionary containing individual loss components for logging
            
        Raises:
            KeyError: If required keys are missing from inputs
            RuntimeError: If there's a device mismatch between tensors
        """
        try:
            # Extract logits for each task and ensure correct shape [B, L]
            d_logits = outputs["splice"]["donor"].squeeze(-1)  # [B, L, 1] -> [B, L]
            a_logits = outputs["splice"]["acceptor"].squeeze(-1)
            t_logits = outputs["tss"]["tss"].squeeze(-1)
            p_logits = outputs["polya"]["polya"].squeeze(-1)

            # Move labels to the correct device and ensure float32 dtype for loss computation
            def prepare_label(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.to(device=self.device, dtype=torch.float32)

            donor = prepare_label(batch["donor"])
            acceptor = prepare_label(batch["acceptor"])
            tss = prepare_label(batch["tss"])
            polya = prepare_label(batch["polya"])

            def cut(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """Truncate tensors to the minimum length along dimension 1."""
                if x.dim() != 2 or y.dim() != 2:
                    raise ValueError(f"Expected 2D tensors, got shapes {x.shape} and {y.shape}")
                L = min(x.shape[1], y.shape[1])
                return x[:, :L].contiguous(), y[:, :L].contiguous()

            # Align logits and labels by minimum sequence length
            d_log, d_lab = cut(d_logits, donor)
            a_log, a_lab = cut(a_logits, acceptor)
            t_log, t_lab = cut(t_logits, tss)
            p_log, p_lab = cut(p_logits, polya)

            # Get loss weights from config
            w = self.cfg["loss"]
            w_splice = float(w["w_splice"])
            w_tss = float(w["w_tss"])
            w_polya = float(w["w_polya"])

            def _masked_loss(
                logits: torch.Tensor, 
                labels: torch.Tensor, 
                weight: float
            ) -> torch.Tensor:
                """Compute masked binary cross-entropy loss with NaN handling.
                
                Args:
                    logits: Model predictions of shape [B, L]
                    labels: Ground truth labels of shape [B, L] (may contain NaNs)
                    weight: Weight for this loss component
                    
                Returns:
                    Weighted loss value as a scalar tensor.
                    
                Note:
                    - NaN values in labels are treated as missing/masked out
                    - Loss is only computed over valid (non-NaN) positions
                    - Returns zero if no valid positions are found
                """
                if logits.device != labels.device:
                    raise RuntimeError(
                        f"Device mismatch: logits on {logits.device}, labels on {labels.device}"
                    )
                
                # Calculate per-element loss
                loss_elements = self.criterion(logits, labels)
                
                # Create mask for valid (non-NaN) labels
                mask = ~torch.isnan(labels)
                
                # Return zero loss if no valid labels
                if not mask.any():
                    return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                
                # Compute mean loss over valid positions
                loss = loss_elements[mask].mean()
                return loss * weight

            # Compute individual task losses
            loss_d = _masked_loss(d_log, d_lab, w_splice)
            loss_a = _masked_loss(a_log, a_lab, w_splice)
            loss_t = _masked_loss(t_log, t_lab, w_tss)
            loss_p = _masked_loss(p_log, p_lab, w_polya)

            # Combine losses
            total = loss_d + loss_a + loss_t + loss_p
            
            # Prepare logs with detached tensors to avoid memory leaks
            logs = {
                "loss/total": total.detach().clone(),
                "loss/donor": loss_d.detach().clone(),
                "loss/acceptor": loss_a.detach().clone(),
                "loss/tss": loss_t.detach().clone(),
                "loss/polya": loss_p.detach().clone(),
            }
            
            return total, logs
            
        except KeyError as e:
            raise KeyError(f"Missing required key in input: {e}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError("CUDA out of memory - try reducing batch size") from e
            raise
        except Exception as e:
            raise RuntimeError(f"Error in _compute_loss: {str(e)}") from e

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for the model.
        
        Args:
            batch: Dictionary containing the input batch
            batch_idx: Index of the current batch
            
        Returns:
            The computed loss for this batch
        """
        # Compute embeddings on CPU, then move to device
        emb = self._get_embeddings(batch["seqs"])
        
        outs = self.model(embeddings=emb)
        loss, logs = self._compute_loss(outs, batch)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()},
            on_epoch=True, 
            prog_bar=True, 
            batch_size=len(batch["seqs"])
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for the model.
        
        Args:
            batch: Dictionary containing the input batch
            batch_idx: Index of the current batch
            
        Returns:
            The computed loss for this batch
        """
        emb = self._get_embeddings(batch["seqs"])
        outs = self.model(embeddings=emb)
        loss, logs = self._compute_loss(outs, batch)
        self.log(
            "val/loss", 
            loss, 
            on_epoch=True, 
            prog_bar=True, 
            batch_size=len(batch["seqs"])
        )
        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()}, 
            on_epoch=True, 
            prog_bar=False
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.
        
        Note: Only optimizes the BetaDogma model parameters (not the encoder).
        
        Returns:
            Configured optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

# DataModule: Structural
class StructuralDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling structural data loading.
    
    This DataModule is responsible for loading and preparing structural data
    (e.g., splice sites, TSS, polyA) from Parquet files for training and validation.
    
    Args:
        dcfg: Data configuration dictionary containing paths and settings.
        cfg_dir: Directory containing the configuration file (for resolving relative paths).
    """
    
    def __init__(self, dcfg: dict[str, Any], cfg_dir: Path) -> None:
        """Initialize the StructuralDataModule with the given configuration.
        
        Args:
            dcfg: Data configuration dictionary.
            cfg_dir: Directory containing the configuration file.
        """
        super().__init__()
        self.cfg = dcfg
        self.cfg_dir = cfg_dir
        self.pin_memory = (
            torch.cuda.is_available() 
            if dcfg.get("pin_memory", "auto") == "auto"
            else bool(dcfg.get("pin_memory"))
        )
        self.num_workers = int(dcfg.get("num_workers", 2))
        # Enable persistent workers if using multiple workers
        self.persistent_workers = self.num_workers > 0
        
        # Initialize dataset attributes
        self.train_ds: Optional[StructuralParquetDataset] = None
        self.val_ds: Optional[StructuralParquetDataset] = None

    def _resolve_glob(self, pat: str) -> list[Path]:
        """Resolve a glob pattern to absolute paths.
        
        Args:
            pat: Glob pattern (can be relative to cfg_dir).
            
        Returns:
            List of sorted Path objects matching the pattern.
        """
        from glob import glob
        p = Path(pat)
        if not p.is_absolute():
            p = self.cfg_dir / p
        return [Path(x) for x in sorted(glob(str(p)))]

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module by loading and preparing the datasets.
        
        Args:
            stage: Optional stage ('fit', 'validate', 'test', or 'predict').
        """
        train_glob = self.cfg.get("train_parquet_glob")
        val_glob = self.cfg.get("val_parquet_glob")
        
        if not train_glob or not val_glob:
            raise ValueError(
                "For task=structural you must set data.train_parquet_glob and data.val_parquet_glob."
            )
        
        print(f"[DEBUG] Config dir: {self.cfg_dir}")
        print(f"[DEBUG] Train glob pattern (raw): {train_glob}")
        print(f"[DEBUG] Val glob pattern (raw): {val_glob}")
        
        train_paths = self._resolve_glob(train_glob)
        val_paths = self._resolve_glob(val_glob)
        
        print(f"[DEBUG] Train paths found: {len(train_paths)}")
        if train_paths:
            print(f"[DEBUG] First train path: {train_paths[0]}")
        print(f"[DEBUG] Val paths found: {len(val_paths)}")
        if val_paths:
            print(f"[DEBUG] First val path: {val_paths[0]}")
        
        if not train_paths or not val_paths:
            raise FileNotFoundError(
                f"No Parquet shards matched train/val globs.\n"
                f"  Config dir: {self.cfg_dir}\n"
                f"  Train glob: {train_glob} -> {len(train_paths)} files\n"
                f"  Val glob: {val_glob} -> {len(val_paths)} files\n"
                f"Check that:\n"
                f"  1. Your YAML has data.train_parquet_glob and data.val_parquet_glob set\n"
                f"  2. The glob patterns are correct (use wildcards like *.parquet)\n"
                f"  3. The parquet files exist at those locations"
            )
        
        self.train_ds = StructuralParquetDataset(train_paths)
        self.val_ds = StructuralParquetDataset(val_paths)

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.
        
        Returns:
            DataLoader configured for training data.
            
        Raises:
            RuntimeError: If called before setup().
        """
        if self.train_ds is None:
            raise RuntimeError("Call setup() before requesting a DataLoader")
            
        return DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.get("batch_size", 2)),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=structural_collate,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader.
        
        Returns:
            DataLoader configured for validation data.
            
        Raises:
            RuntimeError: If called before setup().
        """
        if self.val_ds is None:
            raise RuntimeError("Call setup() before requesting a DataLoader")
            
        return DataLoader(
            self.val_ds,
            batch_size=int(self.cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=structural_collate,
            persistent_workers=self.persistent_workers
        )


# ---------- Helpers ----------
def _maybe_load_yaml_or_dict(value: Union[dict[str, Any], str, Path], cfg_dir: Path) -> dict[str, Any]:
    """Accept dicts as-is; load strings/paths as YAML; resolve relative to CONFIG file."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        import yaml
        p = Path(value)
        if not p.is_absolute():
            p = (cfg_dir / p)
        if not p.exists():
            raise FileNotFoundError(f"Model config file not found: {p}")
        with p.open("r") as f:
            return yaml.safe_load(f) or {}
    raise TypeError("model.kwargs.config must be a dict or a path to a YAML file.")

# ---------- Model factory (legacy/jsonl path) ----------
def build_model(mcfg: dict[str, Any], max_len: int, cfg_dir: Path) -> nn.Module:
    """Build a model from configuration.
    
    Args:
        mcfg: Model configuration dictionary
        max_len: Maximum sequence length
        cfg_dir: Directory containing the configuration file
        
    Returns:
        Configured PyTorch model
    """
    # Toy path
    if mcfg.get("toy", False):
        return TinySeqModel(
            vocab_size=6,
            embed_dim=int(mcfg.get("embed_dim", 64)),
            hidden=int(mcfg.get("hidden_dim", 128)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )

    # Optional factory path: "package.module:make_model"
    factory = mcfg.get("factory")
    if factory:
        mod_name, fn_name = factory.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        kwargs = dict(mcfg.get("kwargs") or {})
        if "config" in kwargs:
            kwargs["config"] = _maybe_load_yaml_or_dict(kwargs["config"], cfg_dir)
        elif "config_path" in kwargs:
            kwargs["config"] = _maybe_load_yaml_or_dict(kwargs.pop("config_path"), cfg_dir)
        try:
            params = inspect.signature(fn).parameters
            if "max_len" in params and "max_len" not in kwargs:
                kwargs["max_len"] = int(max_len)
        except (ValueError, TypeError):
            pass
        model = fn(**kwargs)
        if not isinstance(model, nn.Module):
            err_msg = f"Factory '{factory}' must return a torch.nn.Module."
            raise TypeError(err_msg)
        return model

    # Class path route
    class_path = mcfg.get("class_path")
    class_name = mcfg.get("class_name")
    if not class_path:
        err_msg = "model.class_path (or model.factory) is required unless model.toy is true."
        raise ValueError(err_msg)
    if "." in class_path and class_path.split(".")[-1][:1].isupper() and not class_name:
        module_name, cls_name = class_path.rsplit(".", 1)
    else:
        module_name, cls_name = class_path, (class_name or "")
    mod = importlib.import_module(module_name)

    obj = getattr(mod, cls_name) if cls_name else None
    if obj is None or isinstance(obj, type(importlib)):
        candidates = [(k, v) for k, v in vars(mod).items()
                      if isinstance(v, type) and issubclass(v, nn.Module) and k[:1].isupper()]
        if not candidates:
            raise ImportError(
                f"No nn.Module classes found in '{module_name}'. "
                f"Set model.class_path to 'pkg.mod.ClassName'."
            )
        candidates.sort(key=lambda kv: (("Decoder" not in kv[0], "Isoform" not in kv[0]), kv[0]))
        obj = candidates[0][1]

    kwargs = dict(mcfg.get("kwargs") or {})
    if "config" in kwargs:
        kwargs["config"] = _maybe_load_yaml_or_dict(kwargs["config"], cfg_dir)
    elif "config_path" in kwargs:
        kwargs["config"] = _maybe_load_yaml_or_dict(kwargs.pop("config_path"), cfg_dir)
    try:
        params = inspect.signature(obj.__init__).parameters
        if "max_len" in params and "max_len" not in kwargs:
            kwargs["max_len"] = int(max_len)
    except (ValueError, TypeError):
        pass
    return obj(**kwargs)

# ---------- Trainer from config ----------
def build_trainer(tcfg: dict[str, Any], cfg_dir: Path) -> pl.Trainer:
    """Build and configure a PyTorch Lightning Trainer from a config.

    Args:
        tcfg: Trainer configuration dictionary
        cfg_dir: Directory containing the configuration file

    Returns:
        Configured PyTorch Lightning Trainer

    """
    logdir   = Path(tcfg.get("logdir", "runs/betadogma"))
    ckpt_dir = Path(tcfg.get("ckpt_dir", "checkpoints/betadogma"))
    if not logdir.is_absolute():
        logdir = cfg_dir / logdir
    if not ckpt_dir.is_absolute():
        ckpt_dir = cfg_dir / ckpt_dir
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=str(logdir), name="", version=None, default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch:02d}-{val_loss:.4f}",
                        monitor="val/loss", mode="min",
                        save_top_k=int(tcfg.get("save_top_k", 2)), save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    pat = tcfg.get("early_stopping_patience")
    if pat is not None:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=int(pat)))

    precision = tcfg.get("precision", "32-true")

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"  # Apple Silicon
        if precision != "32-true":
            precision = "32-true"  # MPS doesn't support mixed precision yet
    else:
        accelerator = "cpu"
        precision = "32-true"

    return pl.Trainer(
        accelerator=accelerator,
        devices=int(tcfg.get("devices", 1)),
        max_epochs=int(tcfg.get("epochs", 2)),
        precision=precision,
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 25)),
        check_val_every_n_epoch=1,
        deterministic=True,
        enable_progress_bar=True,
        limit_train_batches=tcfg.get("limit_train_batches"),
        limit_val_batches=tcfg.get("limit_val_batches"),
    )

# ---------- MAIN ----------

def main() -> None:
    """Run the main training function.
    
    Handles configuration, model setup, and training loop.
    """
    # 1) Config
    cfg_env = os.environ.get("TRAIN_CONFIG", "")
    configs_dir = Path(__file__).parent.parent / "configs"
    cfg_path = Path(cfg_env).resolve() if cfg_env else configs_dir / "train.base.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (Path(__file__).parent / cfg_path).resolve()
    cfg = load_config(cfg_path)
    cfg_dir = Path(cfg["_config_dir"]).resolve()

    # 2) Sections
    seed = int(cfg.get("seed", 42))
    task = cfg.get("task", "structural")  # "structural" or "jsonl"
    dcfg = dict(cfg.get("data", {}))
    mcfg = dict(cfg.get("model", {}))
    ocfg = dict(cfg.get("optim", {"lr": 2e-4, "weight_decay": 0.01}))
    tcfg = dict(cfg.get("trainer", {}))

    # 3) Seed
    set_seed(seed)

    # 4) Build
    trainer = build_trainer(tcfg, cfg_dir=cfg_dir)

    if task == "structural":
        # Data
        dm = StructuralDataModule(dcfg, cfg_dir=cfg_dir)
        # Model (uses BetaDogmaModel + NTEncoder, losses from cfg.model.loss)
        required = ["encoder", "heads", "loss"]
        missing = [f"model.{k}" for k in required if k not in mcfg]
        if missing:
            msg = f"Missing required config keys: {', '.join(missing)}"
            raise ValueError(msg)
        
        lr = float(ocfg.get("lr", 2e-4))
        weight_decay = float(ocfg.get("weight_decay", 0.01))
        lit = LitStructural(model_cfg=mcfg, lr=lr, weight_decay=weight_decay)
        # Train
        trainer.fit(lit, datamodule=dm)

    elif task == "jsonl":
        # Validate data presence unless toy
        if not mcfg.get("toy", False):
            for key in ("train", "val", "max_len"):
                if key not in dcfg or dcfg[key] in (None, ""):
                    msg = f"Non-toy training requires data.{key}. Missing in {cfg_path}."
                    raise ValueError(msg)
        # Data + model
        dm = SeqDataModule(dcfg, cfg_dir=cfg_dir)
        _model_kwargs = mcfg.get("kwargs") or {}
        max_len = int(dcfg.get("max_len", _model_kwargs.get("max_len", 0)))
        model = build_model(mcfg, max_len=max_len, cfg_dir=cfg_dir)
        lr = float(ocfg.get("lr", 3e-4))
        weight_decay = float(ocfg.get("weight_decay", 0.01))
        lit = LitSeq(model=model, lr=lr, weight_decay=weight_decay)
        # Train
        trainer.fit(lit, datamodule=dm)

    else:
        msg = f"Unknown task: {task}. Use 'structural' or 'jsonl'."
        raise ValueError(msg)

# ---------- DataModule (legacy/jsonl) ----------
class SeqDataModule(pl.LightningDataModule):
    """DataModule for sequence data.

    This handles loading and preparing sequence data for training and validation.

    Args:
        dcfg: Data configuration dictionary
        cfg_dir: Directory containing the configuration file

    """
    
    def __init__(self, dcfg: dict[str, Any], cfg_dir: Path) -> None:
        """Initialize the SeqDataModule.

        Args:
            dcfg: Data configuration dictionary
            cfg_dir: Directory containing the configuration file

        """
        super().__init__()
        self.cfg = dcfg
        self.cfg_dir = cfg_dir  # for resolving relative paths in the config
        self.pin_memory = (
            torch.cuda.is_available()
            if dcfg.get("pin_memory", "auto") == "auto"
            else bool(dcfg.get("pin_memory"))
        )

    def _resolve(self, p: str | None) -> Path | None:
        if p in (None, "", False):
            return None
        pp = Path(p)
        return (self.cfg_dir / pp) if not pp.is_absolute() else pp

    def setup(self, _stage: str | None = None) -> None:
        """Set up the data module.

        Args:
            stage: Optional; 'fit', 'validate', 'test', or 'predict'.

        """
        if self.cfg.get("toy", False):
            self.train_ds = self._toy_ds(n=512, seq_len=int(self.cfg["max_len"]))
            self.val_ds = self._toy_ds(n=128, seq_len=int(self.cfg["max_len"]))
        else:
            req = ("train", "val", "max_len")
            missing = [k for k in req if k not in self.cfg or self.cfg[k] in (None, "")]
            if missing:
                error_msg = f"Missing data config keys: {missing}"
                raise ValueError(error_msg)
            train_path = self._resolve(self.cfg["train"])
            val_path   = self._resolve(self.cfg["val"])
            self.train_ds = JsonlSeqDataset(
                train_path, self.cfg["max_len"], self.cfg.get("use_strand", False),
                self.cfg.get("reverse_complement_minus", True),
            )
            self.val_ds = JsonlSeqDataset(
                val_path, self.cfg["max_len"], self.cfg.get("use_strand", False),
                self.cfg.get("reverse_complement_minus", True),
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Returns:
            DataLoader: Configured DataLoader for training data

        """
        batch_size = int(self.cfg.get("batch_size", 32))
        num_workers = int(self.cfg.get("num_workers", 2))
        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader.

        Returns:
            DataLoader: Configured DataLoader for validation data

        """
        batch_size = int(self.cfg.get("batch_size", 32))
        num_workers = int(self.cfg.get("num_workers", 2))
        return DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_batch,
        )

    @staticmethod
    def _toy_ds(n: int, seq_len: int) -> Dataset:
        class _Toy(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                x = torch.randint(low=0, high=6, size=(seq_len,), dtype=torch.long)
                y = torch.tensor(float((x.sum() % 2)==0))
                return {"x": x, "y": y}

        return _Toy()

if __name__ == "__main__":
    main()