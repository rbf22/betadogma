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
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

# Configure logging to write to both console and file
def setup_logging():
    """Set up logging to write to both console and file."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("betadogma_training")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler (INFO level and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level and above)
    log_file = log_dir / "training_debug.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Set up logging at module level
logger = setup_logging()

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
            "seq": r["seq"],           # Reference sequence
            "seq_alt": r.get("seq_alt"),  # Variant sequence (if exists)
            "variant_type": r.get("variant_type"),  # SNP/INS/DEL
            "variant_af": r.get("variant_af", 1.0),  # Allele frequency
            "is_pathogenic": r.get("is_pathogenic", False),  # Label
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
    seqs_alt = [b.get("seq_alt") for b in batch]  # NEW
    has_variant = [s is not None for s in seqs_alt]  # NEW
    variant_afs = [b.get("variant_af", 1.0) for b in batch]  # NEW
    is_pathogenic = [b.get("is_pathogenic", False) for b in batch]  # NEW
    
    donor = torch.stack([pad1d(b["donor"], max_Lr) for b in batch])
    acceptor = torch.stack([pad1d(b["acceptor"], max_Lr) for b in batch])
    tss = torch.stack([pad1d(b["tss"], max_Lr) for b in batch])
    polya = torch.stack([pad1d(b["polya"], max_Lr) for b in batch])
    
    return {
        "seqs": seqs,
        "seqs_alt": seqs_alt,  # NEW
        "has_variant": torch.tensor(has_variant, dtype=torch.bool),  # NEW
        "variant_afs": torch.tensor(variant_afs, dtype=torch.float32),  # NEW
        "is_pathogenic": torch.tensor(is_pathogenic, dtype=torch.bool),  # NEW
        "donor": donor,
        "acceptor": acceptor,
        "tss": tss,
        "polya": polya,
    }

# LightningModule: Structural (NTEncoder + BetaDogmaModel)
class LitStructural(pl.LightningModule):
    def __init__(
        self, 
        model_cfg: dict[str, Any], 
        lr: float, 
        weight_decay: float,
        gradient_clip_val: float = 1.0  # Add gradient clip value with default
    ):
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
        
        # Ensure model is properly initialized and on the correct device
        self.model = self.model.to(self.device)
        print(f"[DEBUG] Model moved to device: {self.device}")
        
        # Verify model weights are not NaN after device transfer
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"[WARNING] NaN detected in model parameter: {name}")
        
        self.save_hyperparameters({
            "lr": lr, 
            "weight_decay": weight_decay,
            "gradient_clip_val": gradient_clip_val
        })

        pos_w = torch.tensor(model_cfg["loss"]["pos_weight"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='none')
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.gradient_clip_val = float(gradient_clip_val)  # Store gradient clip value

        # NEW: Variant effect prediction weights
        self.w_consistency = float(model_cfg.get("loss", {}).get("w_consistency", 0.1))
        self.w_disruption = float(model_cfg.get("loss", {}).get("w_disruption", 0.5))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Apply gradient clipping in the optimizer
        if self.gradient_clip_val > 0:
            from torch.nn.utils import clip_grad_norm_
            for group in optimizer.param_groups:
                if 'max_grad_norm' not in group:
                    group['max_grad_norm'] = self.gradient_clip_val
        
        return optimizer

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

            # Check for NaN in model outputs early
            if torch.isnan(d_logits).any() or torch.isnan(a_logits).any() or \
               torch.isnan(t_logits).any() or torch.isnan(p_logits).any():
                logger.warning("NaN detected in model outputs!")
                if torch.isnan(d_logits).any():
                    logger.warning(f"NaN in donor logits: {torch.isnan(d_logits).sum()} positions")
                if torch.isnan(a_logits).any():
                    logger.warning(f"NaN in acceptor logits: {torch.isnan(a_logits).sum()} positions")
                if torch.isnan(t_logits).any():
                    logger.warning(f"NaN in tss logits: {torch.isnan(t_logits).sum()} positions")
                if torch.isnan(p_logits).any():
                    logger.warning(f"NaN in polya logits: {torch.isnan(p_logits).sum()} positions")

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

                # Check for NaN in loss elements
                if torch.isnan(loss_elements).any():
                    logger.debug(f"NaN in loss_elements before masking: {torch.isnan(loss_elements).sum()} positions")
                    logger.debug(f"Loss elements range: [{loss_elements.min().item():.4f}, {loss_elements.max().item():.4f}]")

                # Create mask for valid (non-NaN) labels
                mask = ~torch.isnan(labels)

                # Return zero loss if no valid labels
                if not mask.any():
                    return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

                # Compute mean loss over valid positions
                loss = loss_elements[mask].mean()

                # Check for NaN in final loss
                if torch.isnan(loss):
                    logger.warning(f"NaN in final masked loss (weight={weight})")

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

    def _compute_variant_effect_loss(
        self,
        outputs_ref: Dict[str, Dict[str, torch.Tensor]],
        outputs_alt: Dict[str, Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss for variant effect prediction.
        
        Two components:
        1. Consistency: Common variants should have similar predictions (small Δ)
        2. Disruption: Pathogenic variants should have different predictions (large Δ)
        """
        # Extract predictions
        def extract_preds(outs):
            return {
                'donor': outs["splice"]["donor"].squeeze(-1),
                'acceptor': outs["splice"]["acceptor"].squeeze(-1),
                'tss': outs["tss"]["tss"].squeeze(-1),
                'polya': outs["polya"]["polya"].squeeze(-1),
            }
        
        preds_ref = extract_preds(outputs_ref)
        preds_alt = extract_preds(outputs_alt)
        
        has_variant = batch["has_variant"]  # [B]
        variant_afs = batch["variant_afs"]  # [B]
        is_pathogenic = batch["is_pathogenic"]  # [B]
        
        # Only compute for samples with variants
        if not has_variant.any():
            return torch.tensor(0.0, device=self.device), {}
        
        # Compute deltas for each task
        deltas = {}
        for task in ['donor', 'acceptor', 'tss', 'polya']:
            ref = preds_ref[task][has_variant]  # [N, L]
            alt = preds_alt[task][has_variant]  # [N, L]
            # Align lengths
            L = min(ref.shape[1], alt.shape[1])
            deltas[task] = torch.abs(ref[:, :L] - alt[:, :L])  # [N, L]
        
        # Stack all deltas: [N, L, 4]
        delta_stack = torch.stack([deltas[k] for k in ['donor', 'acceptor', 'tss', 'polya']], dim=-1)
        delta_magnitude = delta_stack.mean(dim=-1)  # [N, L] - average across tasks

        # Check for NaN in delta_magnitude early
        if torch.isnan(delta_magnitude).any():
            print(f"[DEBUG] NaN in delta_magnitude: {torch.isnan(delta_magnitude).sum()} positions")
            print(f"[DEBUG] delta_magnitude range: [{delta_magnitude.min().item():.4f}, {delta_magnitude.max().item():.4f}]")

        # 1. CONSISTENCY LOSS: Benign (common) variants should have small Δ
        #    Target: Δ ≈ 0 for high AF variants
        benign_mask = has_variant & ~is_pathogenic  # [B]
        if benign_mask.any():
            # Weight by AF: higher AF = stronger consistency requirement
            af_weights = variant_afs[benign_mask].unsqueeze(-1)  # [N_benign, 1]
            benign_deltas = delta_magnitude[benign_mask[has_variant]]  # [N_benign, L]

            # Loss: weighted L2 norm of deltas (we want them close to 0)
            consistency_loss = (af_weights * benign_deltas.pow(2)).mean()

            if torch.isnan(consistency_loss):
                print(f"[DEBUG] NaN in consistency_loss: af_weights={af_weights.mean().item():.4f}, benign_deltas={benign_deltas.mean().item():.4f}")
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)

        # 2. DISRUPTION LOSS: Pathogenic variants should have large Δ
        #    For splice variants, delta should be large specifically at splice sites
        pathogenic_mask = has_variant & is_pathogenic  # [B]
        if pathogenic_mask.any():
            path_deltas = delta_magnitude[pathogenic_mask[has_variant]]  # [N_path, L]

            # Check for NaN in path_deltas before log
            if torch.isnan(path_deltas).any():
                print(f"[DEBUG] NaN in path_deltas before log: {torch.isnan(path_deltas).sum()} positions")

            # We want LARGE deltas for pathogenic variants
            # Loss: negative log of delta magnitude (encourages large deltas)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-6
            disruption_loss = -torch.log(path_deltas + epsilon).mean()

            if torch.isnan(disruption_loss):
                print(f"[DEBUG] NaN in disruption_loss: path_deltas={path_deltas.mean().item():.4f}, epsilon={epsilon}")
        else:
            disruption_loss = torch.tensor(0.0, device=self.device)

        # Combine losses
        total_variant_loss = (
            self.w_consistency * consistency_loss +
            self.w_disruption * disruption_loss
        )

        logs = {
            "variant/consistency": consistency_loss.detach(),
            "variant/disruption": disruption_loss.detach(),
            "variant/total": total_variant_loss.detach(),
            "variant/mean_delta": delta_magnitude.mean().detach(),
        }

        return total_variant_loss, logs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 1. Primary task: predict splicing from reference sequence
        emb_ref = self._get_embeddings(batch["seqs"])
        
        # Check for NaN in embeddings immediately after encoder
        if torch.isnan(emb_ref).any():
            logger.warning(f"NaN in encoder embeddings: {torch.isnan(emb_ref).sum()} positions")
            logger.debug(f"Embeddings range: [{emb_ref.min().item():.4f}, {emb_ref.max().item():.4f}]")
            logger.debug(f"Embeddings shape: {emb_ref.shape}")
        
        # Check input data for potential issues
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    logger.warning(f"NaN in batch.{key}: {torch.isnan(value).sum()} positions")
                if torch.isinf(value).any():
                    logger.warning(f"Inf in batch.{key}: {torch.isinf(value).sum()} positions")
                # Check for extreme values that might cause numerical issues
                if value.numel() > 0:
                    val_range = (value.min().item(), value.max().item())
                    if abs(val_range[0]) > 1e6 or abs(val_range[1]) > 1e6:
                        logger.warning(f"Extreme values in batch.{key}: range {val_range}")
        
        outs_ref = self.model(embeddings=emb_ref)
        
        # Check if model parameters became NaN during forward pass
        nan_params_before = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params_before.append(name)
        
        if nan_params_before:
            logger.debug(f"NaN in parameters BEFORE backward: {nan_params_before}")
        
        # Debug model outputs before loss computation
        logger.debug("Model outputs shapes:")
        if isinstance(outs_ref, dict):
            for task, outputs in outs_ref.items():
                if isinstance(outputs, dict):
                    # Expected nested structure: task -> head -> tensor
                    logger.debug(f"  {task}:")
                    for head, logits in outputs.items():
                        if torch.is_tensor(logits):
                            logger.debug(f"    {head}: {logits.shape}")
                        else:
                            logger.debug(f"    {head}: {type(logits)}")
                elif torch.is_tensor(outputs):
                    # Direct tensor output (fallback)
                    logger.debug(f"  {task}: {outputs.shape}")
                else:
                    logger.debug(f"  {task}: {type(outputs)}")
        else:
            logger.debug(f"Unexpected outs_ref type: {type(outs_ref)}")

        # Safe iteration for NaN checking
        def check_tensor_nans(task_name, head_name, tensor):
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any():
                    logger.warning(f"NaN in {task_name}.{head_name} logits: {torch.isnan(tensor).sum()} positions")
                else:
                    logger.debug(f"{task_name}.{head_name} logits range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
                    logger.debug(f"{task_name}.{head_name} logits mean: {tensor.mean().item():.4f}")

        if isinstance(outs_ref, dict):
            for task, outputs in outs_ref.items():
                if isinstance(outputs, dict):
                    # Expected nested structure
                    for head, logits in outputs.items():
                        check_tensor_nans(task, head, logits)
                elif torch.is_tensor(outputs):
                    # Direct tensor output - treat as single head
                    check_tensor_nans(task, task, outputs)
        
        # Debug ground truth labels
        logger.debug("Ground truth label shapes:")
        for label_name in ['donor', 'acceptor', 'tss', 'polya']:
            if label_name in batch:
                label = batch[label_name]
                logger.debug(f"  {label_name}: {label.shape}")
                logger.debug(f"  {label_name} labels range: [{label.min().item():.4f}, {label.max().item():.4f}]")
                logger.debug(f"  {label_name} labels mean: {label.mean().item():.4f}")
                nan_count = torch.isnan(label).sum().item()
                if nan_count > 0:
                    logger.warning(f"{label_name} has {nan_count} NaN values")
        
        loss_primary, logs_primary = self._compute_loss(outs_ref, batch)
        
        # 2. Variant effect task (if variants present in batch)
        loss_variant = torch.tensor(0.0, device=self.device)
        logs_variant = {}
        
        if batch["has_variant"].any():
            # Get variant sequences (only for samples with variants)
            seqs_alt = [s for s, has in zip(batch["seqs_alt"], batch["has_variant"]) if has]
            
            if seqs_alt:
                # Compute predictions for variant sequences
                emb_alt = self._get_embeddings(batch["seqs_alt"])
                outs_alt = self.model(embeddings=emb_alt)
                
                # Compute variant effect loss
                loss_variant, logs_variant = self._compute_variant_effect_loss(
                    outs_ref, outs_alt, batch
                )
        
        # 3. Combine losses
        total_loss = loss_primary + loss_variant

        # 4. Track NaN occurrences for debugging
        if not hasattr(self, '_nan_debug_info'):
            self._nan_debug_info = {
                'total_nan_count': 0,
                'nan_steps': [],
                'nan_locations': [],
                'losses_at_nan': []
            }

        # Check for NaN in total loss
        if torch.isnan(total_loss):
            self._nan_debug_info['total_nan_count'] += 1
            self._nan_debug_info['nan_steps'].append(batch_idx)
            self._nan_debug_info['losses_at_nan'].append(total_loss.item() if not torch.isnan(total_loss) else 0.0)

            # Check where NaN is coming from
            nan_sources = []
            if torch.isnan(loss_primary):
                nan_sources.append("primary_loss")
            if torch.isnan(loss_variant):
                nan_sources.append("variant_loss")

            # Check individual loss components
            for key, value in {**logs_primary, **logs_variant}.items():
                if torch.isnan(value):
                    nan_sources.append(f"logs.{key}")

            logger.debug(f"NaN detected in training step {batch_idx} from: {nan_sources}")

        # 5. Logging
        self.log_dict(
            {f"train/{k}": v for k, v in {**logs_primary, **logs_variant}.items()},
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["seqs"])
        )
        self.log("train/loss_total", total_loss, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Same modifications as training_step
        emb_ref = self._get_embeddings(batch["seqs"])
        outs_ref = self.model(embeddings=emb_ref)
        loss_primary, logs_primary = self._compute_loss(outs_ref, batch)
        
        loss_variant = torch.tensor(0.0, device=self.device)
        logs_variant = {}
        
        if batch["has_variant"].any():
            seqs_alt = [s for s, has in zip(batch["seqs_alt"], batch["has_variant"]) if has]
            if seqs_alt:
                emb_alt = self._get_embeddings(batch["seqs_alt"])
                outs_alt = self.model(embeddings=emb_alt)
                loss_variant, logs_variant = self._compute_variant_effect_loss(
                    outs_ref, outs_alt, batch
                )
        
        total_loss = loss_primary + loss_variant

        # Track NaN occurrences for debugging in validation
        if not hasattr(self, '_val_nan_debug_info'):
            self._val_nan_debug_info = {
                'total_nan_count': 0,
                'nan_steps': [],
                'nan_locations': [],
                'losses_at_nan': []
            }

        # Check for NaN in total loss
        if torch.isnan(total_loss):
            self._val_nan_debug_info['total_nan_count'] += 1
            self._val_nan_debug_info['nan_steps'].append(batch_idx)
            self._val_nan_debug_info['losses_at_nan'].append(total_loss.item() if not torch.isnan(total_loss) else 0.0)

            # Check where NaN is coming from
            nan_sources = []
            if torch.isnan(loss_primary):
                nan_sources.append("primary_loss")
            if torch.isnan(loss_variant):
                nan_sources.append("variant_loss")

            # Check individual loss components
            for key, value in {**logs_primary, **logs_variant}.items():
                if torch.isnan(value):
                    nan_sources.append(f"logs.{key}")

            self._val_nan_debug_info['nan_locations'].extend(nan_sources)

            logger.debug(f"[DEBUG] NaN detected in validation step {batch_idx} from: {nan_sources}")

        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True, batch_size=len(batch["seqs"]))
        self.log_dict({f"val/{k}": v for k, v in {**logs_primary, **logs_variant}.items()}, on_epoch=True)
        
        return total_loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch for debugging."""
        if hasattr(self, '_nan_debug_info'):
            debug_info = self._nan_debug_info
            if debug_info['total_nan_count'] > 0:
                print("\n=== EPOCH END DEBUG ===")
                print(f"Total NaN occurrences this epoch: {debug_info['total_nan_count']}")
                print(f"Steps with NaN: {len(debug_info['nan_steps'])}")
                if debug_info['nan_steps']:
                    print(f"First NaN step: {debug_info['nan_steps'][0]}")
                    print(f"Last NaN step: {debug_info['nan_steps'][-1]}")
                print(f"Average loss when NaN occurred: {debug_info.get('avg_loss_at_nan', 'N/A')}")
                print(f"NaN locations: {set(debug_info.get('nan_locations', []))}")
                print("====================\n")

        # Reset counters for next epoch
        self._nan_debug_info = {
            'total_nan_count': 0,
            'nan_steps': [],
            'nan_locations': [],
            'losses_at_nan': []
        }

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch for debugging."""
        if hasattr(self, '_val_nan_debug_info'):
            debug_info = self._val_nan_debug_info
            if debug_info['total_nan_count'] > 0:
                print("\n=== VALIDATION EPOCH END DEBUG ===")
                print(f"Total NaN occurrences in validation: {debug_info['total_nan_count']}")
                print(f"Validation steps with NaN: {len(debug_info['nan_steps'])}")
                if debug_info['nan_steps']:
                    print(f"First NaN step: {debug_info['nan_steps'][0]}")
                    print(f"Last NaN step: {debug_info['nan_steps'][-1]}")
                print(f"Average loss when NaN occurred: {debug_info.get('avg_loss_at_nan', 'N/A')}")
                print("===============================\n")

        # Reset counters for next epoch
        self._val_nan_debug_info = {
            'total_nan_count': 0,
            'nan_steps': [],
            'nan_locations': [],
            'losses_at_nan': []
        }


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
    if cfg_env:
        cfg_path = Path(cfg_env).resolve()
    else:
        local_config = Path(__file__).parent / "configs" / "train.base.yaml"
        cfg_path = local_config if local_config.exists() else Path(__file__).parent.parent / "configs" / "train.base.yaml"
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

    # In train.py, around line 920-970, replace the relevant section with:

    # 4) Build trainer with updated configuration
    trainer = build_trainer(tcfg, cfg_dir=cfg_dir)

    if task == "structural":
        # Data
        dm = StructuralDataModule(dcfg, cfg_dir=cfg_dir)
        
        # Validate model configuration
        required = ["encoder", "heads", "loss"]
        missing = [f"model.{k}" for k in required if k not in mcfg]
        if missing:
            msg = f"Missing required config keys: {', '.join(missing)}"
            raise ValueError(msg)
        
        # Get training parameters
        lr = float(ocfg.get("lr", 1e-5))
        weight_decay = float(ocfg.get("weight_decay", 0.01))
        
        # Initialize model with gradient clipping
        gradient_clip_val = float(ocfg.get("gradient_clip_val", 1.0))
        lit = LitStructural(
            model_cfg=mcfg, 
            lr=lr, 
            weight_decay=weight_decay,
            gradient_clip_val=gradient_clip_val
        )
        
        # Configure learning rate scheduler if specified
        lr_scheduler_cfg = cfg.get("lr_scheduler", {})
        if lr_scheduler_cfg:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            # Get optimizer
            optimizer = lit.configure_optimizers()
            if isinstance(optimizer, tuple) and len(optimizer) > 0:
                optimizer = optimizer[0]
            
            # Create scheduler
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=lr_scheduler_cfg.get("mode", "min"),
                factor=float(lr_scheduler_cfg.get("factor", 0.5)),
                patience=int(lr_scheduler_cfg.get("patience", 2)),
                min_lr=float(lr_scheduler_cfg.get("min_lr", 1e-6)),
            )
            
            # Add scheduler to the trainer
            trainer.lr_schedulers = [{
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr"
            }]
        
        # Add data validation if enabled
        if dcfg.get("validate_numerical_stability", False):
            print("Enabling numerical stability validation...")
            torch.autograd.set_detect_anomaly(True)
            
            def check_tensor(name, tensor):
                if torch.is_tensor(tensor):
                    if torch.isnan(tensor).any():
                        print(f"Warning: NaN detected in {name}")
                    if torch.isinf(tensor).any():
                        print(f"Warning: Inf detected in {name}")
            
            # Patch the training step to check for numerical issues
            original_training_step = lit.training_step
            
            def patched_training_step(batch, batch_idx):
                # Check input data
                for k, v in batch.items():
                    check_tensor(f"batch.{k}", v)
                
                # Run original training step
                loss = original_training_step(batch, batch_idx)
                
                # Check output
                check_tensor("loss", loss)
                
                return loss
            
            lit.training_step = patched_training_step
        
        # Train the model
        trainer.fit(lit, datamodule=dm)

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