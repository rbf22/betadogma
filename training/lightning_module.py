
import logging
from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from config import Config
from model.model import BetaDogmaModel
from torch import nn
from torchmetrics import AUROC

# Setup logger
logger = logging.getLogger(__name__)

# ============================================================================
# Lightning Module with Memory Management
# ============================================================================

class BetaDogmaLightning(pl.LightningModule):
    """Lightning module with aggressive memory management."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model = BetaDogmaModel(config)
        self.pos_weight = torch.tensor(config.pos_weight)

        # Add loss function for splice effect (regression)
        self.splice_effect_loss = nn.MSELoss()

        # Add AUROC metrics for binary classification tasks
        self.auroc_donor = AUROC(task="binary")
        self.auroc_acceptor = AUROC(task="binary")
        self.auroc_tss = AUROC(task="binary")
        self.auroc_polya = AUROC(task="binary")
        self.auroc_cds_start = AUROC(task="binary")
        self.auroc_cds_end = AUROC(task="binary")
        self.auroc_nmd = AUROC(task="binary")

        self.batch_count = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return cast(Dict[str, torch.Tensor], self.model(input_ids, attention_mask))

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch."""
        labels = batch["labels"]

        # Binary cross-entropy for splice site prediction
        loss_donor = F.binary_cross_entropy_with_logits(
            outputs["donor"],
            labels["donor"],
            pos_weight=self.pos_weight.to(self.device)
        )

        loss_acceptor = F.binary_cross_entropy_with_logits(
            outputs["acceptor"],
            labels["acceptor"],
            pos_weight=self.pos_weight.to(self.device)
        )

        # TSS and polyA site prediction
        loss_tss = F.binary_cross_entropy_with_logits(
            outputs["tss"],
            labels["tss"],
            pos_weight=self.pos_weight.to(self.device)
        )

        loss_polya = F.binary_cross_entropy_with_logits(
            outputs["polya"],
            labels["polya"],
            pos_weight=self.pos_weight.to(self.device)
        )

        # Splice effect prediction (regression)
        if "splice_effect" in outputs and "splice_effect" in labels:
            # Only compute loss on positions with non-zero effect
            mask = (labels["splice_effect"] > 0).float()
            if mask.sum() > 0:
                loss_splice = self.splice_effect_loss(
                    outputs["splice_effect"] * mask,
                    labels["splice_effect"] * mask
                )

                # Consistency loss between splice effect and donor/acceptor predictions
                if self.config.consistency_weight > 0:
                    # Get sigmoid of donor/acceptor logits
                    donor_probs = torch.sigmoid(outputs["donor"])
                    acceptor_probs = torch.sigmoid(outputs["acceptor"])

                    # Compute max probability of donor/acceptor at each position
                    max_probs = torch.max(donor_probs, acceptor_probs)

                    # Only compute consistency where we have splice effect labels
                    consistency_loss = F.mse_loss(
                        outputs["splice_effect"] * mask,
                        max_probs.detach() * mask
                    )

                    # Add to splice effect loss
                    loss_splice = loss_splice + self.config.consistency_weight * consistency_loss
            else:
                loss_splice = torch.tensor(0.0, device=self.device)
        else:
            loss_splice = torch.tensor(0.0, device=self.device)

        # Phase 1: Protein prediction with NaN handling
        # Check if we have any valid protein labels (not -1)
        protein_mask = (labels["protein"] != -1).view(-1)
        if protein_mask.sum() > 0:
            # Only compute loss on valid positions
            loss_protein = F.cross_entropy(
                outputs["protein"].view(-1, 21)[protein_mask],
                labels["protein"].view(-1)[protein_mask]
            )
        else:
            # No valid protein labels in this batch, use zero loss
            loss_protein = torch.tensor(0.0, device=self.device)

        loss_cds_start = F.binary_cross_entropy_with_logits(
            outputs["cds_start"],
            labels["cds_start"],
            pos_weight=self.pos_weight.to(self.device)
        )

        loss_cds_end = F.binary_cross_entropy_with_logits(
            outputs["cds_end"],
            labels["cds_end"],
            pos_weight=self.pos_weight.to(self.device)
        )

        loss_nmd = F.binary_cross_entropy_with_logits(
            outputs["nmd"],
            labels["nmd"]
        )

        loss_expression = F.mse_loss(
            outputs["expression"],
            labels["expression"]
        )

        # Phase 2B: Variant effect prediction loss
        loss_variant_effect = torch.tensor(0.0, device=self.device)
        if "variant_effect" in labels and labels["variant_effect"].sum() > 0:
            # Only compute loss on examples with variants
            mask = (labels["variant_effect"] > 0).float()
            if mask.sum() > 0:
                loss_variant_effect = F.mse_loss(
                    outputs.get("variant_effect", torch.zeros_like(labels["variant_effect"])) * mask,
                    labels["variant_effect"] * mask
                )

        # Combine losses with weights
        # Only include protein loss if it's valid (not zero from no labels)
        protein_weight = self.config.w_protein if protein_mask.sum() > 0 else 0.0

        loss = (
            self.config.w_splice_donor * loss_donor +
            self.config.w_splice_acceptor * loss_acceptor +
            self.config.w_tss * loss_tss +
            self.config.w_polya * loss_polya +
            self.config.w_splice_effect * loss_splice +
            protein_weight * loss_protein +
            self.config.w_cds_start * loss_cds_start +
            self.config.w_cds_end * loss_cds_end +
            self.config.w_nmd * loss_nmd +
            self.config.w_expression * loss_expression +
            0.5 * loss_variant_effect  # Phase 2B weight
        )

        # Additional safety check for NaN
        if torch.isnan(loss):
            logger.warning(f"Total loss is NaN! Individual losses: donor={loss_donor.item()}, acceptor={loss_acceptor.item()}, tss={loss_tss.item()}, polya={loss_polya.item()}, splice={loss_splice.item()}, protein={loss_protein.item()}, cds_start={loss_cds_start.item()}, cds_end={loss_cds_end.item()}, nmd={loss_nmd.item()}, expression={loss_expression.item()}, variant_effect={loss_variant_effect.item()}")

            # Replace NaN with a large value to continue training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        return {
            "loss": loss,
            "loss/donor": loss_donor,
            "loss/acceptor": loss_acceptor,
            "loss/tss": loss_tss,
            "loss/polya": loss_polya,
            "loss/splice_effect": loss_splice,
            "loss/protein": loss_protein,
            "loss/cds_start": loss_cds_start,
            "loss/cds_end": loss_cds_end,
            "loss/nmd": loss_nmd,
            "loss/expression": loss_expression,
            "loss/variant_effect": loss_variant_effect,
        }


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        try:
            # Log batch info at DEBUG level (only shown if verbose_batches=true)
            if self.config.verbose_batches:
                logger.debug(f"Training step {batch_idx}")
                logger.debug(f"Batch keys: {list(batch.keys())}")

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: shape={tuple(v.shape)}, device={v.device}, dtype={v.dtype}")

            # Forward pass
            logger.debug("Starting model forward...")
            outputs = self(batch["input_ids"],
                        attention_mask=batch.get("attention_mask"))
            logger.debug("Forward pass completed")

            # Compute loss
            logger.debug("Computing losses...")
            loss_dict = self._compute_loss(outputs, batch)

            if not isinstance(loss_dict, dict):
                raise ValueError(f"_compute_loss should return a dict, got {type(loss_dict)}")

            if "loss" not in loss_dict:
                raise ValueError("'loss' key not found in loss_dict")

            loss = loss_dict["loss"]
            if not isinstance(loss, torch.Tensor):
                raise ValueError(f"loss should be a tensor, got {type(loss)}")

            # Log loss breakdown at DEBUG level
            if self.config.verbose_batches:
                logger.debug("Loss breakdown:")
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: {v.item():.6f}")

            # Always log to tensorboard
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"train/{k}", v, prog_bar=True, on_step=True, on_epoch=True)

            # Compute and log AUROC metrics
            try:
                labels = batch["labels"]
                self.auroc_donor.update(torch.sigmoid(outputs["donor"]).detach(), labels["donor"].detach().long())
                self.log("train/auroc/donor", self.auroc_donor, on_step=False, on_epoch=True)
                self.auroc_acceptor.update(torch.sigmoid(outputs["acceptor"]).detach(), labels["acceptor"].detach().long())
                self.log("train/auroc/acceptor", self.auroc_acceptor, on_step=False, on_epoch=True)
                self.auroc_tss.update(torch.sigmoid(outputs["tss"]).detach(), labels["tss"].detach().long())
                self.log("train/auroc/tss", self.auroc_tss, on_step=False, on_epoch=True)
                self.auroc_polya.update(torch.sigmoid(outputs["polya"]).detach(), labels["polya"].detach().long())
                self.log("train/auroc/polya", self.auroc_polya, on_step=False, on_epoch=True)
                self.auroc_cds_start.update(torch.sigmoid(outputs["cds_start"]).detach(), labels["cds_start"].detach().long())
                self.log("train/auroc/cds_start", self.auroc_cds_start, on_step=False, on_epoch=True)
                self.auroc_cds_end.update(torch.sigmoid(outputs["cds_end"]).detach(), labels["cds_end"].detach().long())
                self.log("train/auroc/cds_end", self.auroc_cds_end, on_step=False, on_epoch=True)
                self.auroc_nmd.update(torch.sigmoid(outputs["nmd"]).detach(), labels["nmd"].detach().long())
                self.log("train/auroc/nmd", self.auroc_nmd, on_step=False, on_epoch=True)
            except Exception as e:
                logger.debug(f"Could not compute AUROC metrics: {e}")

            # Log learning rate
            try:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("train/learning_rate", current_lr, on_step=True, on_epoch=False)
            except Exception as e:
                logger.debug(f"Could not log learning rate: {e}")

            return loss

        except Exception as e:
            logger.error(f"Error in training step {batch_idx}: {type(e).__name__}: {e!s}", exc_info=True)
            raise

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss_dict = self._compute_loss(outputs, batch)
        labels = batch["labels"]

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True, on_step=False, on_epoch=True)

        # Compute and log validation AUROC metrics
        try:
            self.auroc_donor.update(torch.sigmoid(outputs["donor"]).detach(), labels["donor"].detach().long())
            self.log("val/auroc/donor", self.auroc_donor, on_step=False, on_epoch=True)
            self.auroc_acceptor.update(torch.sigmoid(outputs["acceptor"]).detach(), labels["acceptor"].detach().long())
            self.log("val/auroc/acceptor", self.auroc_acceptor, on_step=False, on_epoch=True)
            self.auroc_tss.update(torch.sigmoid(outputs["tss"]).detach(), labels["tss"].detach().long())
            self.log("val/auroc/tss", self.auroc_tss, on_step=False, on_epoch=True)
            self.auroc_polya.update(torch.sigmoid(outputs["polya"]).detach(), labels["polya"].detach().long())
            self.log("val/auroc/polya", self.auroc_polya, on_step=False, on_epoch=True)
            self.auroc_cds_start.update(torch.sigmoid(outputs["cds_start"]).detach(), labels["cds_start"].detach().long())
            self.log("val/auroc/cds_start", self.auroc_cds_start, on_step=False, on_epoch=True)
            self.auroc_cds_end.update(torch.sigmoid(outputs["cds_end"]).detach(), labels["cds_end"].detach().long())
            self.log("val/auroc/cds_end", self.auroc_cds_end, on_step=False, on_epoch=True)
            self.auroc_nmd.update(torch.sigmoid(outputs["nmd"]).detach(), labels["nmd"].detach().long())
            self.log("val/auroc/nmd", self.auroc_nmd, on_step=False, on_epoch=True)
        except Exception as e:
            logger.debug(f"Could not compute validation AUROC metrics: {e}")

        return loss_dict["loss"]

    def on_train_batch_end(self, outputs: Any, batch: Dict[str, Any], batch_idx: int) -> None:
        """Log memory usage, histograms, and clean up after each batch."""
        try:
            # Log memory usage EVERY batch
            if torch.backends.mps.is_available():
                try:
                    mps_allocated = torch.mps.current_allocated_memory() / (1024**3)
                    self.log("system/mps_memory_gb", mps_allocated, on_step=True, on_epoch=False)
                    if mps_allocated > 20.0:
                        logger.warning(f"MPS memory high: {mps_allocated:.2f} GB / 27.20 GB")
                except Exception as e:
                    logger.debug(f"Could not log MPS memory: {e}")

            if torch.cuda.is_available():
                try:
                    cuda_allocated = torch.cuda.memory_allocated() / (1024**3)
                    cuda_reserved = torch.cuda.memory_reserved() / (1024**3)
                    self.log("system/cuda_memory_allocated_gb", cuda_allocated, on_step=True, on_epoch=False)
                    self.log("system/cuda_memory_reserved_gb", cuda_reserved, on_step=True, on_epoch=False)
                except Exception as e:
                    logger.debug(f"Could not log CUDA memory: {e}")

            # Log prediction, weight, and gradient histograms every 100 batches
            if batch_idx % 100 == 0 and self.logger:
                try:
                    if isinstance(outputs, dict):
                        for name, pred in outputs.items():
                            if isinstance(pred, torch.Tensor) and pred.numel() > 0:
                                try:
                                    cast(Any, self.logger).experiment.add_histogram(f"predictions/{name}", pred.detach().cpu(), self.global_step)
                                except Exception as e:
                                    logger.debug(f"Could not log histogram for {name}: {e}")

                    for name, param in self.named_parameters():
                        if param.requires_grad and param.numel() > 0:
                            try:
                                cast(Any, self.logger).experiment.add_histogram(f"weights/{name}", param.detach().cpu(), self.global_step)
                                if param.grad is not None and param.grad.numel() > 0:
                                    cast(Any, self.logger).experiment.add_histogram(f"gradients/{name}", param.grad.detach().cpu(), self.global_step)
                            except Exception as e:
                                logger.debug(f"Could not log histogram for {name}: {e}")
                except Exception as e:
                    logger.debug(f"Could not log histograms: {e}")

            # Flush TensorBoard after every batch
            if self.logger and hasattr(self.logger, "experiment"):
                try:
                    cast(Any, self.logger).experiment.flush()
                except Exception as e:
                    logger.debug(f"Could not flush TensorBoard: {e}")
        finally:
            try:
                del batch, outputs
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    import gc
                    gc.collect()
            except Exception as e:
                logger.debug(f"Could not clean up memory: {e}")

    def on_train_epoch_end(self) -> None:
        """Log epoch-level summaries and flush TensorBoard."""
        try:
            metrics = self.trainer.callback_metrics
            self.log("epoch", float(self.current_epoch), on_epoch=True)
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {self.current_epoch} SUMMARY")
            logger.info(f"{'='*80}")
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.item():.6f}")
                elif isinstance(value, (int, float)):  # type: ignore[unreachable]
                    logger.info(f"  {key}: {value:.6f}")
            logger.info(f"{'='*80}\n")
            if self.logger and hasattr(self.logger, "experiment"):
                try:
                    cast(Any, self.logger).experiment.flush()
                except Exception as e:
                    logger.debug(f"Could not flush TensorBoard: {e}")
        except Exception as e:
            logger.debug(f"Could not log epoch summary: {e}")
        finally:
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception as e:
                logger.debug(f"Could not clean up at epoch end: {e}")

    def configure_optimizers(self) -> Any:
        """Configure optimizer with memory-efficient settings."""
        # Only trainable parameters (heads only)
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        # For MPS, use memory-efficient optimizer settings
        if self.device.type == "mps":
            # Try 8-bit AdamW first (saves ~4x memory)
            try:
                import bitsandbytes as bnb  # type: ignore
                optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
                logger.info("Using 8-bit AdamW optimizer (memory efficient)")
            except ImportError:
                # Fallback to regular AdamW with memory-efficient settings
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    foreach=False,  # Disable foreach for lower memory
                    capturable=False,  # Required for MPS
                    fused=False,  # Not supported on MPS
                )
                logger.info("Using regular AdamW (bitsandbytes not available on macOS/MPS)")
        else:
            # Standard optimizer for CUDA/CPU
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

