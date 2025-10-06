#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config-only training entrypoint (lives inside train/).

Now supports two tasks:
  - task: "structural"  -> trains Betadogma heads on Parquet shards from prepare_gencode.py
  - task: "jsonl"       -> legacy JSONL binary classifier (Tiny model or user-supplied)

It resolves paths in the config relative to the config file, and uses PyTorch Lightning
for logging and checkpointing.
"""

import os
import sys
import json
import random
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Project paths ----------
THIS_FILE = Path(__file__).resolve()
TRAIN_DIR = THIS_FILE.parent                    # .../train
PROJECT_ROOT = TRAIN_DIR.parent                 # repo root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))           # make betadogma importable

DEFAULT_CFG = TRAIN_DIR / "configs" / "train.base.yaml"

# ---------- Torch / Lightning ----------
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC

# ---------- YAML loader ----------
def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_dir"] = str(path.parent)   # keep for relative path resolution
    return cfg

# ---------- Repro ----------
def set_seed(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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

# ---------- Dataset: JSONL (legacy) ----------
class JsonlSeqDataset(Dataset):
    def __init__(self, path: Path, max_len: int, use_strand: bool, revcomp_minus: bool):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.max_len = int(max_len)
        self.use_strand = bool(use_strand)
        self.revcomp_minus = bool(revcomp_minus)
        self._lines: List[str] = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if not self._lines:
            raise ValueError(f"No records found in JSONL: {path}")

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            ex = json.loads(self._lines[idx])
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {idx} is not valid JSON.") from e
        if "sequence" not in ex or "label" not in ex:
            raise KeyError("Each JSONL line must contain 'sequence' and 'label'.")
        seq = ex["sequence"]
        if not isinstance(seq, str) or not seq:
            raise ValueError("`sequence` must be a non-empty string.")
        if self.use_strand and ex.get("strand") == "-" and self.revcomp_minus:
            seq = revcomp(seq)
        x = encode_seq(seq, self.max_len)  # [L]
        y = ex["label"]
        if isinstance(y, bool):
            y = int(y)
        if y not in (0, 1):
            raise ValueError("`label` must be 0 or 1.")
        return {"x": x, "y": torch.tensor(float(y), dtype=torch.float32)}

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)         # [B, L]
    y = torch.stack([b["y"] for b in batch], dim=0).float() # [B]
    return {"x": x, "y": y}

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
        logits = self.head(feat).squeeze(1)         # [B]
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
    def __init__(self, paths: List[Path]):
        import pandas as pd
        self._rows: List[Dict[str, Any]] = []
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

def structural_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    def __init__(self, model_cfg: Dict[str, Any], lr: float, weight_decay: float):
        super().__init__()
        from betadogma.model import BetaDogmaModel
        from betadogma.core.encoder_nt import NTEncoder

        self.cfg = model_cfg
        d_in = int(model_cfg["encoder"]["hidden_size"])
        self.encoder = NTEncoder(model_id=model_cfg["encoder"].get("model_id") or
                                 "InstaDeepAI/nucleotide-transformer-500m-human-ref")
        self.model = BetaDogmaModel(d_in=d_in, config=model_cfg)

        pos_w = torch.tensor(model_cfg["loss"]["pos_weight"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters({"lr": self.lr, "weight_decay": self.weight_decay})

    def _compute_loss(self, outputs, batch):
        # shapes: outputs['splice']['donor']: [B, L, 1] ; labels: [B, L]
        d_logits = outputs["splice"]["donor"].squeeze(-1)
        a_logits = outputs["splice"]["acceptor"].squeeze(-1)
        t_logits = outputs["tss"]["tss"].squeeze(-1)
        p_logits = outputs["polya"]["polya"].squeeze(-1)

        donor = batch["donor"].to(d_logits.device)
        acceptor = batch["acceptor"].to(d_logits.device)
        tss = batch["tss"].to(d_logits.device)
        polya = batch["polya"].to(d_logits.device)

        # align by min length
        def cut(x, y): 
            L = min(x.shape[1], y.shape[1])
            return x[:, :L], y[:, :L]

        d_log, d_lab = cut(d_logits, donor)
        a_log, a_lab = cut(a_logits, acceptor)
        t_log, t_lab = cut(t_logits, tss)
        p_log, p_lab = cut(p_logits, polya)

        w = self.cfg["loss"]
        loss_d = self.criterion(d_log, d_lab) * float(w["w_splice"])
        loss_a = self.criterion(a_log, a_lab) * float(w["w_splice"])
        loss_t = self.criterion(t_log, t_lab) * float(w["w_tss"])
        loss_p = self.criterion(p_log, p_lab) * float(w["w_polya"])
        total = loss_d + loss_a + loss_t + loss_p
        logs = {
            "loss/total": total,
            "loss/donor": loss_d.detach(),
            "loss/acceptor": loss_a.detach(),
            "loss/tss": loss_t.detach(),
            "loss/polya": loss_p.detach(),
        }
        return total, logs

    def training_step(self, batch, _):
        with torch.no_grad():
            emb = self.encoder.forward(batch["seqs"])   # [B, L, D]
        outs = self.model(embeddings=emb)
        loss, logs = self._compute_loss(outs, batch)
        self.log_dict({f"train/{k}": v for k, v in logs.items()},
                      on_epoch=True, prog_bar=True, batch_size=len(batch["seqs"]))
        return loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            emb = self.encoder.forward(batch["seqs"])
        outs = self.model(embeddings=emb)
        loss, logs = self._compute_loss(outs, batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch["seqs"]))
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# DataModule: Structural
class StructuralDataModule(pl.LightningDataModule):
    def __init__(self, dcfg: Dict[str, Any], cfg_dir: Path):
        super().__init__()
        self.cfg = dcfg
        self.cfg_dir = cfg_dir
        self.pin_memory = (torch.cuda.is_available() if dcfg.get("pin_memory", "auto") == "auto"
                           else bool(dcfg.get("pin_memory")))

    def _resolve_glob(self, pat: str) -> List[Path]:
        from glob import glob
        p = Path(pat)
        if not p.is_absolute():
            p = self.cfg_dir / p
        return [Path(x) for x in sorted(glob(str(p)))]

    def setup(self, stage: Optional[str] = None):
        train_glob = self.cfg.get("train_parquet_glob")
        val_glob   = self.cfg.get("val_parquet_glob")
        if not train_glob or not val_glob:
            raise ValueError("For task=structural you must set data.train_parquet_glob and data.val_parquet_glob.")
        train_paths = self._resolve_glob(train_glob)
        val_paths   = self._resolve_glob(val_glob)
        if not train_paths or not val_paths:
            raise FileNotFoundError("No Parquet shards matched train/val globs.")
        self.train_ds = StructuralParquetDataset(train_paths)
        self.val_ds   = StructuralParquetDataset(val_paths)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=int(self.cfg.get("batch_size", 2)), shuffle=True,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          drop_last=True, collate_fn=structural_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=int(self.cfg.get("batch_size", 2)), shuffle=False,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          collate_fn=structural_collate)

# ---------- Helpers ----------
def _maybe_load_yaml_or_dict(value: Union[Dict[str, Any], str, Path], cfg_dir: Path) -> Dict[str, Any]:
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
def build_model(mcfg: Dict[str, Any], max_len: int, cfg_dir: Path) -> nn.Module:
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
            raise TypeError(f"Factory '{factory}' must return a torch.nn.Module.")
        return model

    # Class path route
    class_path = mcfg.get("class_path")
    class_name = mcfg.get("class_name")
    if not class_path:
        raise ValueError("model.class_path (or model.factory) is required unless model.toy is true.")
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
def build_trainer(tcfg: Dict[str, Any], cfg_dir: Path) -> pl.Trainer:
    logdir   = Path(tcfg.get("logdir", "runs/betadogma"))
    ckpt_dir = Path(tcfg.get("ckpt_dir", "checkpoints/betadogma"))
    if not logdir.is_absolute():   logdir   = cfg_dir / logdir
    if not ckpt_dir.is_absolute(): ckpt_dir = cfg_dir / ckpt_dir
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(logdir), name="", version=None, default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch:02d}-{val_loss:.4f}",
                        monitor="val/loss", mode="min",
                        save_top_k=int(tcfg.get("save_top_k", 2)), save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    pat = tcfg.get("early_stopping_patience", None)
    if pat is not None:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=int(pat)))

    precision = tcfg.get("precision", "32-true")
    if not torch.cuda.is_available() and precision != "32-true":
        precision = "32-true"  # avoid AMP-on-CPU warnings

    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
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
        limit_train_batches=tcfg.get("limit_train_batches", None),
        limit_val_batches=tcfg.get("limit_val_batches", None),
    )

# ---------- MAIN ----------
def main():
    # 1) Config
    cfg_env = os.environ.get("TRAIN_CONFIG", "")
    cfg_path = Path(cfg_env).resolve() if cfg_env else DEFAULT_CFG
    if not cfg_path.is_absolute():
        cfg_path = (TRAIN_DIR / cfg_path).resolve()
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
        if "encoder" not in mcfg or "heads" not in mcfg or "loss" not in mcfg:
            raise ValueError("For task=structural, model.encoder, model.heads, and model.loss must be in config.")
        lit = LitStructural(model_cfg=mcfg, lr=float(ocfg.get("lr", 2e-4)), weight_decay=float(ocfg.get("weight_decay", 0.01)))
        # Train
        trainer.fit(lit, datamodule=dm)

    elif task == "jsonl":
        # Validate data presence unless toy
        if not mcfg.get("toy", False):
            for key in ("train", "val", "max_len"):
                if key not in dcfg or dcfg[key] in (None, ""):
                    raise ValueError(f"Non-toy training requires data.{key}. Missing in {cfg_path}.")
        # Data + model
        dm = SeqDataModule(dcfg, cfg_dir=cfg_dir)
        _model_kwargs = mcfg.get("kwargs") or {}
        max_len = int(dcfg.get("max_len", _model_kwargs.get("max_len", 0)))
        model = build_model(mcfg, max_len=max_len, cfg_dir=cfg_dir)
        lit = LitSeq(model=model, lr=float(ocfg.get("lr", 3e-4)), weight_decay=float(ocfg.get("weight_decay", 0.01)))
        # Train
        trainer.fit(lit, datamodule=dm)

    else:
        raise ValueError(f"Unknown task: {task}. Use 'structural' or 'jsonl'.")

# ---------- DataModule (legacy/jsonl) ----------
class SeqDataModule(pl.LightningDataModule):
    def __init__(self, dcfg: Dict[str, Any], cfg_dir: Path):
        super().__init__()
        self.cfg = dcfg
        self.cfg_dir = cfg_dir  # for resolving relative paths in the config
        self.pin_memory = (torch.cuda.is_available() if dcfg.get("pin_memory", "auto") == "auto"
                           else bool(dcfg.get("pin_memory")))

    def _resolve(self, p: Optional[str]) -> Optional[Path]:
        if p in (None, "", False):
            return None
        pp = Path(p)
        return (self.cfg_dir / pp) if not pp.is_absolute() else pp

    def setup(self, stage: Optional[str] = None):
        if self.cfg.get("toy", False):
            self.train_ds = self._toy_ds(n=512, L=int(self.cfg["max_len"]))
            self.val_ds   = self._toy_ds(n=128, L=int(self.cfg["max_len"]))
        else:
            req = ("train", "val", "max_len")
            missing = [k for k in req if k not in self.cfg or self.cfg[k] in (None, "")]
            if missing:
                raise ValueError(f"Missing data config keys: {missing}")
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

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=int(self.cfg.get("batch_size", 32)), shuffle=True,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          drop_last=True, collate_fn=collate_batch)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=int(self.cfg.get("batch_size", 32)), shuffle=False,
                          num_workers=int(self.cfg.get("num_workers", 2)), pin_memory=self.pin_memory,
                          collate_fn=collate_batch)

    @staticmethod
    def _toy_ds(n: int, L: int):
        class _Toy(Dataset):
            def __len__(self): return n
            def __getitem__(self, i):
                x = torch.randint(low=0, high=6, size=(L,), dtype=torch.long)
                y = torch.tensor(float((x.sum() % 2)==0))
                return {"x": x, "y": y}
        return _Toy()

if __name__ == "__main__":
    main()