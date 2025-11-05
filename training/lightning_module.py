
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import logging

from config import Config
from model.model import BetaDogmaModel

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
        
        self.batch_count = 0
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def _compute_loss(self, outputs, batch):
        """Compute loss for a batch."""
        labels = batch['labels']
        
        # Binary cross-entropy for splice site prediction
        loss_donor = F.binary_cross_entropy_with_logits(
            outputs['donor'], 
            labels['donor'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_acceptor = F.binary_cross_entropy_with_logits(
            outputs['acceptor'], 
            labels['acceptor'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        # TSS and polyA site prediction
        loss_tss = F.binary_cross_entropy_with_logits(
            outputs['tss'], 
            labels['tss'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_polya = F.binary_cross_entropy_with_logits(
            outputs['polya'], 
            labels['polya'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        # Splice effect prediction (regression)
        if 'splice_effect' in outputs and 'splice_effect' in labels:
            # Only compute loss on positions with non-zero effect
            mask = (labels['splice_effect'] > 0).float()
            if mask.sum() > 0:
                loss_splice = self.splice_effect_loss(
                    outputs['splice_effect'] * mask,
                    labels['splice_effect'] * mask
                )
                
                # Consistency loss between splice effect and donor/acceptor predictions
                if self.config.consistency_weight > 0:
                    # Get sigmoid of donor/acceptor logits
                    donor_probs = torch.sigmoid(outputs['donor'])
                    acceptor_probs = torch.sigmoid(outputs['acceptor'])
                    
                    # Compute max probability of donor/acceptor at each position
                    max_probs = torch.max(donor_probs, acceptor_probs)
                    
                    # Only compute consistency where we have splice effect labels
                    consistency_loss = F.mse_loss(
                        outputs['splice_effect'] * mask,
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
        protein_mask = (labels['protein'] != -1).view(-1)
        if protein_mask.sum() > 0:
            # Only compute loss on valid positions
            loss_protein = F.cross_entropy(
                outputs['protein'].view(-1, 21)[protein_mask],
                labels['protein'].view(-1)[protein_mask]
            )
        else:
            # No valid protein labels in this batch, use zero loss
            loss_protein = torch.tensor(0.0, device=self.device)
        
        loss_cds_start = F.binary_cross_entropy_with_logits(
            outputs['cds_start'],
            labels['cds_start'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_cds_end = F.binary_cross_entropy_with_logits(
            outputs['cds_end'],
            labels['cds_end'],
            pos_weight=self.pos_weight.to(self.device)
        )
        
        loss_nmd = F.binary_cross_entropy_with_logits(
            outputs['nmd'],
            labels['nmd']
        )
        
        loss_expression = F.mse_loss(
            outputs['expression'],
            labels['expression']
        )
        
        # Phase 2B: Variant effect prediction loss
        loss_variant_effect = torch.tensor(0.0, device=self.device)
        if 'variant_effect' in labels and labels['variant_effect'].sum() > 0:
            # Only compute loss on examples with variants
            mask = (labels['variant_effect'] > 0).float()
            if mask.sum() > 0:
                loss_variant_effect = F.mse_loss(
                    outputs.get('variant_effect', torch.zeros_like(labels['variant_effect'])) * mask,
                    labels['variant_effect'] * mask
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
            'loss': loss,
            'loss/donor': loss_donor,
            'loss/acceptor': loss_acceptor,
            'loss/tss': loss_tss,
            'loss/polya': loss_polya,
            'loss/splice_effect': loss_splice,
            'loss/protein': loss_protein,
            'loss/cds_start': loss_cds_start,
            'loss/cds_end': loss_cds_end,
            'loss/nmd': loss_nmd,
            'loss/expression': loss_expression,
            'loss/variant_effect': loss_variant_effect,
        }

    
    def training_step(self, batch, batch_idx):
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
            outputs = self(batch['input_ids'], 
                        attention_mask=batch.get('attention_mask'))
            logger.debug("Forward pass completed")
            
            # Compute loss
            logger.debug("Computing losses...")
            loss_dict = self._compute_loss(outputs, batch)
            
            if not isinstance(loss_dict, dict):
                raise ValueError(f"_compute_loss should return a dict, got {type(loss_dict)}")
            
            if 'loss' not in loss_dict:
                raise ValueError("'loss' key not found in loss_dict")
            
            loss = loss_dict['loss']
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
                    self.log(f'train/{k}', v, prog_bar=True, on_step=True, on_epoch=True)
            
            return loss
                    
        except Exception as e:
            logger.error(f"Error in training step {batch_idx}: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss_dict = self._compute_loss(outputs, batch)
        
        for k, v in loss_dict.items():
            self.log(f'val/{k}', v, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss_dict['loss']
    
    def configure_optimizers(self):
        """Configure optimizer with memory-efficient settings."""
        # Only trainable parameters (heads only)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        # For MPS, use memory-efficient optimizer settings
        if self.device.type == 'mps':
            # Try 8-bit AdamW first (saves ~4x memory)
            try:
                import bitsandbytes as bnb
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
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
