"""
Phase 1: Structural fine-tuning for splice / TSS / polyA heads.

Usage:
  python -m betadogma.experiments.train --config betadogma/experiments/config/default.yaml
"""
from __future__ import annotations
import argparse
import os
import random
from glob import glob
from typing import Dict, List, Optional, Any, Tuple

import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from betadogma.core.encoder_nt import NTEncoder
from betadogma.model import BetaDogmaModel
from betadogma.core.losses import structural_bce_ce_loss
from betadogma.data.variant_loader import CommonVariantLoader
from betadogma.data.variant_processor import OnTheFlyVariantProcessor
from betadogma.data.encode import apply_variants_to_sequence


# ---------------- Data ----------------

class StructuralDataset(Dataset):
    """
    Expects Parquet shards where each row has:
      - "seq": str DNA sequence (A/C/G/T/N)
      - "donor", "acceptor", "tss", "polya": lists/arrays of 0/1 labels (length Lr)
    """
    def __init__(self, parquet_paths: List[str], max_shards: int | None = None):
        self.paths = sorted(parquet_paths)[:max_shards] if max_shards else sorted(parquet_paths)
        self.rows = []
        for p in self.paths:
            df = pd.read_parquet(p)
            self.rows.extend(df.to_dict("records"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        item = {
            "seq": r["seq"],
            "donor": torch.as_tensor(r["donor"], dtype=torch.float32),
            "acceptor": torch.as_tensor(r["acceptor"], dtype=torch.float32),
            "tss": torch.as_tensor(r["tss"], dtype=torch.float32),
            "polya": torch.as_tensor(r["polya"], dtype=torch.float32),
        }
        
        # Add variant information if available
        if 'variant_pos' in r and r['variant_pos'] is not None:
            item['variant_pos'] = int(r['variant_pos'])  # 1-based position
            item['is_pathogenic'] = bool(r.get('is_pathogenic', False))
            
        return item


def log_variant_statistics(batch):
    """Log detailed statistics about variants in the batch."""
    if 'variant_pos' not in batch:
        logger.info("No variant information in batch")
        return
        
    total = len(batch['has_variant'])
    has_variants = batch['has_variant'].sum().item()
    pathogenic = batch.get('is_pathogenic', torch.zeros_like(batch['has_variant'])).sum().item()
    
    # Log basic counts
    logger.info(f"Batch statistics:")
    logger.info(f"  Total sequences: {total}")
    logger.info(f"  Sequences with variants: {has_variants} ({has_variants/max(1,total):.1%})")
    logger.info(f"  Pathogenic variants: {pathogenic} ({pathogenic/max(1,has_variants):.1%} of variants)")
    
    # Log variant types if available
    if 'variant_ref' in batch and 'variant_alt' in batch:
        var_types = []
        for ref, alt in zip(batch['variant_ref'], batch['variant_alt']):
            if ref is None or alt is None:
                continue
            if len(ref) == len(alt) == 1:
                var_types.append('SNP')
            elif len(ref) < len(alt):
                var_types.append('INS')
            elif len(ref) > len(alt):
                var_types.append('DEL')
            else:
                var_types.append('COMPLEX')
        
        if var_types:
            from collections import Counter
            type_counts = Counter(var_types)
            logger.info("  Variant types:")
            for var_type, count in type_counts.most_common():
                logger.info(f"    {var_type}: {count} ({count/len(var_types):.1%})")

def collate_structural(batch):
    # Variable-length sequences allowed; encoder will pad at token level.
    seqs = [b["seq"] for b in batch]

    # For labels we’ll pad to the max label length in this mini-batch
    def pad1d(x, T):
        L = x.numel()
        if L < T:
            return torch.nn.functional.pad(x, (0, T - L))
        return x[:T]

    max_Lr = max(b["donor"].numel() for b in batch)
    donor   = torch.stack([pad1d(b["donor"],   max_Lr) for b in batch])  # (B, Lr)
    accept  = torch.stack([pad1d(b["acceptor"],max_Lr) for b in batch])
    tss     = torch.stack([pad1d(b["tss"],     max_Lr) for b in batch])
    polya   = torch.stack([pad1d(b["polya"],   max_Lr) for b in batch])
    
    # Handle variant positions and pathogenicity
    variant_pos = []
    is_pathogenic = []
    has_variant = []
    
    for b in batch:
        if 'variant_pos' in b and b['variant_pos'] is not None:
            variant_pos.append(b['variant_pos'] - 1)  # Convert to 0-based
            is_pathogenic.append(b.get('is_pathogenic', False))
            has_variant.append(True)
        else:
            variant_pos.append(0)  # Dummy value, will be masked
            is_pathogenic.append(False)
            has_variant.append(False)
    
    result = {
        "seqs": seqs, 
        "donor": donor, 
        "acceptor": accept, 
        "tss": tss, 
        "polya": polya,
        "variant_pos": torch.tensor(variant_pos, dtype=torch.long),
        "is_pathogenic": torch.tensor(is_pathogenic, dtype=torch.float32),
        "has_variant": torch.tensor(has_variant, dtype=torch.bool)
    }
    
    return result


# ---------------- Training ----------------

def _align_labels_to_L(labels_B_Lr: torch.Tensor, L: int) -> torch.Tensor:
    """Pad/crop labels along length dim to match encoder/heads length L."""
    B, Lr = labels_B_Lr.shape
    if Lr == L:
        return labels_B_Lr
    if Lr < L:
        return torch.nn.functional.pad(labels_B_Lr, (0, L - Lr))
    return labels_B_Lr[:, :L]


def train(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    seed = cfg["trainer"].get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Variant Processor ---
    variant_loader = None
    variant_processor = None
    
    if cfg["trainer"].get("use_variants", False):
        # Initialize variant loader
        vcf_path = cfg["data"].get("vcf_path")
        if vcf_path and os.path.exists(vcf_path):
            variant_loader = CommonVariantLoader()
            variant_loader.load_vcf(vcf_path)
            
            # Initialize variant processor
            variant_processor = OnTheFlyVariantProcessor(
                variant_loader=variant_loader,
                max_variants=cfg["trainer"].get("max_variants", 5),
                balance_variants=cfg["trainer"].get("balance_variants", True),
                variant_prob=cfg["trainer"].get("variant_prob", 0.5),
                seed=seed
            )
            print(f"Initialized variant processor with {len(variant_loader.variants)} variants")
        else:
            print(f"Warning: VCF file not found at {vcf_path}, variant processing disabled")

    # --- Encoder ---
    enc = NTEncoder(
        model_id=cfg["encoder"].get("model_id", "InstaDeepAI/nucleotide-transformer-500m-human-ref"),
        device=cfg["encoder"].get("device", "auto"),
    )

    # --- Model ---
    d_in = enc.hidden_size
    model = BetaDogmaModel(d_in=d_in, config=cfg).to(device)
    model.train()

    # --- Data ---
    shard_glob = os.path.join(cfg["data"]["out_cache"], "*.parquet")
    paths = glob(shard_glob)
    assert paths, f"No parquet shards found at {shard_glob}"

    ds = StructuralDataset(paths, max_shards=cfg["trainer"].get("max_shards"))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["trainer"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["trainer"].get("num_workers", 0)),
        collate_fn=collate_structural,
    )

    # --- Optimizer ---
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg["optimizer"]["lr"]),
                            weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)))

    # Optional loss weights
    w = {
        "donor": cfg["loss"].get("w_splice", 1.0),
        "acceptor": cfg["loss"].get("w_splice", 1.0),
        "tss": cfg["loss"].get("w_tss", 1.0),
        "polya": cfg["loss"].get("w_polya", 1.0),
        "orf_start": cfg["loss"].get("w_orf_start", 0.0),
        "orf_stop": cfg["loss"].get("w_orf_stop", 0.0),
        "orf_frame": cfg["loss"].get("w_orf_frame", 0.0),
    }

    epochs = int(cfg["trainer"]["epochs"])
    for ep in range(epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[Structural] Epoch {ep+1}/{epochs}")
        
        # Log dataset statistics for the first batch of the first epoch
        if ep == 0 and not hasattr(dl, '_logged_stats'):
            logger.info("\n" + "="*50)
            logger.info("TRAINING DATASET STATISTICS")
            logger.info("="*50)
            
            # Count total examples and variants
            total_examples = 0
            total_variants = 0
            total_pathogenic = 0
            variant_types = Counter()
            
            for batch in dl:
                if 'has_variant' in batch:
                    total_examples += len(batch['has_variant'])
                    has_variants = batch['has_variant']
                    total_variants += has_variants.sum().item()
                    
                    if 'is_pathogenic' in batch:
                        total_pathogenic += batch['is_pathogenic'].sum().item()
                    
                    # Count variant types if available
                    if 'variant_ref' in batch and 'variant_alt' in batch:
                        for ref, alt in zip(batch['variant_ref'], batch['variant_alt']):
                            if ref is None or alt is None:
                                continue
                            if len(ref) == len(alt) == 1:
                                variant_types['SNP'] += 1
                            elif len(ref) < len(alt):
                                variant_types['INS'] += 1
                            elif len(ref) > len(alt):
                                variant_types['DEL'] += 1
                            else:
                                variant_types['COMPLEX'] += 1
            
            # Log statistics
            logger.info(f"Total examples: {total_examples}")
            logger.info(f"Examples with variants: {total_variants} ({total_variants/max(1,total_examples):.1%})")
            logger.info(f"Pathogenic variants: {total_pathogenic} ({total_pathogenic/max(1,total_variants):.1%} of variants)")
            
            if variant_types:
                logger.info("Variant type distribution:")
                for var_type, count in variant_types.most_common():
                    logger.info(f"  {var_type}: {count} ({count/sum(variant_types.values()):.1%})")
            
            logger.info("="*50 + "\n")
            
            # Mark as logged to avoid repeating
            dl._logged_stats = True
            
            # Reset the dataloader
            dl = DataLoader(ds, batch_size=int(cfg["trainer"]["batch_size"]), shuffle=True, collate_fn=collate_structural)
        
        # Set epoch for variant processor if available
        if variant_processor is not None:
            variant_processor.set_epoch(ep)
            
        for batch_idx, batch in enumerate(pbar):
            # Apply variants on-the-fly if enabled
            if variant_processor is not None:
                batch = variant_processor.process_batch(batch, device=device)
                
            # Log detailed variant statistics for the first few batches
            if batch_idx < 3:  # Log for first 3 batches of each epoch
                log_variant_statistics(batch)
                
            # Encode sequences -> embeddings + masks
            enc_out = enc.forward(batch["seqs"])
            embeddings = enc_out["embeddings"].to(device)   # (B, L, D)
            input_ids  = enc_out["input_ids"].to(device)    # (B, L)
            pad_mask   = enc_out["pad_mask"].to(device)     # (B, L)
            B, L, _ = embeddings.shape

            # Prepare model inputs
            model_kwargs = {
                "embeddings": embeddings,
                "input_ids": input_ids,
            }
            
            # Add variant positions if available
            if 'variant_pos' in batch and batch['has_variant'].any():
                model_kwargs['variant_positions'] = batch['variant_pos'].to(device)

            # Forward pass
            head_outs = model(**model_kwargs)

            # Prepare labels
            labels = {
                "splice": {
                    "donor": _align_labels_to_L(batch["donor"].to(device), L),
                    "acceptor": _align_labels_to_L(batch["acceptor"].to(device), L),
                },
                "tss": {"tss": _align_labels_to_L(batch["tss"].to(device), L)},
                "polya": {"polya": _align_labels_to_L(batch["polya"].to(device), L)},
            }
            
            # Add variant effect labels if available
            if 'is_pathogenic' in batch and batch['has_variant'].any():
                labels['variant_effect'] = batch['is_pathogenic'].to(device)

            # Compute loss
            loss = structural_bce_ce_loss(
                head_outs, 
                labels, 
                pad_mask=pad_mask,
                weights=w
            )
            
            # Add variant effect metrics to progress bar
            if 'variant_effect' in head_outs and 'is_pathogenic' in batch and batch['has_variant'].any():
                preds = (head_outs['variant_effect'] > 0.5).float()
                acc = (preds == batch['is_pathogenic'].to(device)).float().mean()
                pbar.set_postfix(loss=f"{loss.item():.4f}", var_acc=f"{acc.item():.4f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            loss = structural_bce_ce_loss(head_outs, labels, pad_mask=pad_mask, weights=w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"].get("grad_clip", 1.0)))
            opt.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / max(1, len(dl))
        print(f"Epoch {ep+1}/{epochs}  avg_loss={avg:.4f}")

    # --- Save ---
    ckpt_dir = cfg["trainer"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "betadogma_structural.pt"))
    print(f"Saved checkpoint to {ckpt_dir}/betadogma_structural.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()