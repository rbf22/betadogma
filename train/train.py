#!/usr/bin/env python
# train.py (repo root)

import argparse, os, math, random
from typing import Any, Dict, Optional

# --- config ---
def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 42):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Ensure src/ is importable when running from repo root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Prefer your real modules if present
try:
    from betadogma.decoder import DecoderModel  # adjust if your class is named differently
except Exception:
    DecoderModel = None

try:
    from betadogma.data import BetaDogmaDataset
except Exception:
    BetaDogmaDataset = None

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# --- fallbacks to keep things runnable even if data/model aren't wired yet ---
class _ToyDataset(Dataset):
    def __init__(self, n: int, dim: int):
        self.x = torch.randn(n, dim)
        self.y = (self.x.sum(dim=1) > 0).float()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {"x": self.x[i], "y": self.y[i]}

class LitDecoder(pl.LightningModule):
    def __init__(self, model_cfg: Dict[str, Any], optim_cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "optim_cfg"])
        if DecoderModel is not None:
            self.model = DecoderModel(**(model_cfg or {}))
        else:
            input_dim = model_cfg.get("input_dim", 1024)
            hidden = model_cfg.get("hidden_dim", 512)
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(model_cfg.get("dropout", 0.1)),
                torch.nn.Linear(hidden, 1),
                torch.nn.Sigmoid(),
            )
        self.lr = float(optim_cfg.get("lr", 3e-4))
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.01))
        self.warmup_steps = int(optim_cfg.get("warmup_steps", 0))
        self.max_steps = optim_cfg.get("max_steps")
        self.criterion = torch.nn.BCELoss()

    def forward(self, x): return self.model(x)

    def training_step(self, batch, _):
        x = (batch["x"] if isinstance(batch, dict) else batch[0]).float()
        y = (batch["y"] if isinstance(batch, dict) else batch[1]).float().view(-1, 1)
        y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, _):
        x = (batch["x"] if isinstance(batch, dict) else batch[0]).float()
        y = (batch["y"] if isinstance(batch, dict) else batch[1]).float().view(-1, 1)
        y_hat = self(x); loss = self.criterion(y_hat, y)
        # Generic “val/score”: replace with your ORF/decoder score if available
        pos = (y == 1).squeeze(1); neg = ~pos
        pos_m = y_hat[pos].mean().item() if pos.any() else 0.0
        neg_m = y_hat[neg].mean().item() if neg.any() else 0.0
        score = max(0.0, pos_m - neg_m)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val/score", score, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return {"val_loss": loss, "val_score": torch.tensor(score)}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if not self.max_steps and self.warmup_steps == 0:
            return opt
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step + 1) / max(1, self.warmup_steps)
            if self.max_steps:
                prog = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            return 1.0
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step", "name": "lr"}}

class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = data_cfg
        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = True
    def setup(self, stage: Optional[str] = None):
        if BetaDogmaDataset is not None:
            root = self.cfg["path"]
            self.train_ds = BetaDogmaDataset(root, split="train")
            self.val_ds   = BetaDogmaDataset(root, split="val")
        else:
            dim = int(self.cfg.get("input_dim", 1024))
            self.train_ds = _ToyDataset(2048, dim)
            self.val_ds   = _ToyDataset(256, dim)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

def build_trainer(cfg: Dict[str, Any]) -> pl.Trainer:
    logdir = cfg.get("logging", {}).get("logdir", "runs/default")
    logger = TensorBoardLogger(save_dir=logdir, name="", version=None, default_hp_metric=False)
    ckpt = cfg.get("ckpt", {})
    callbacks = [
        ModelCheckpoint(dirpath=ckpt.get("dir", "checkpoints/default"),
                        filename="{epoch:02d}-{val_score:.4f}",
                        monitor=ckpt.get("monitor", "val/score"),
                        mode=ckpt.get("mode", "max"),
                        save_top_k=int(ckpt.get("save_top_k", 2)),
                        save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    pat = cfg.get("train", {}).get("early_stopping_patience")
    if pat is not None:
        callbacks.append(EarlyStopping(monitor=ckpt.get("monitor", "val/score"),
                                       mode=ckpt.get("mode", "max"),
                                       patience=int(pat)))
    precision = cfg.get("train", {}).get("precision", "bf16")
    devices = int(cfg.get("train", {}).get("devices", 1))
    max_epochs = int(cfg.get("optim", {}).get("epochs", 20))
    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        max_epochs=max_epochs,
        precision=precision if isinstance(precision, int) else (16 if "16" in str(precision) else "bf16-mixed"),
        accumulate_grad_batches=int(cfg.get("train", {}).get("accumulate_grad_batches", 1)),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.get("train", {}).get("log_every_n_steps", 50)),
        val_check_interval=cfg.get("train", {}).get("val_every_n_steps", None),
        check_val_every_n_epoch=None if cfg.get("train", {}).get("val_every_n_steps") else 1,
        deterministic=True,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.base.yaml")
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--limit_train_batches", type=float, default=None)
    ap.add_argument("--limit_val_batches", type=float, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    seed_everything(cfg.get("seed", 42))

    dm = DataModule(cfg.get("data", {}))
    lit = LitDecoder(cfg.get("model", {}), cfg.get("optim", {}))
    trainer = build_trainer(cfg)

    fit_kwargs = {}
    if args.limit_train_batches is not None:
        fit_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        fit_kwargs["limit_val_batches"] = args.limit_val_batches
    if args.max_epochs is not None:
        trainer.max_epochs = args.max_epochs

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume_from or None)

if __name__ == "__main__":
    main()
