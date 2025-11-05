
import torch
from pathlib import Path
import yaml 

# Load configuration from YAML
CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# ============================================================================
# GPU Detection with Memory-Optimized Configs
# ============================================================================

def get_gpu_config():
    """Auto-detect GPU and return memory-optimized config."""
    
    if not torch.cuda.is_available():
        return {
            'max_seq_len': 1000,
            'batch_size': 1,
            'accumulate_grad_batches': 64,
            'device_name': 'CPU',
            'use_gradient_checkpointing': False,
        }
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    gpu_memory_gb = gpu_props.total_memory / (1024**3)
    
    print(f"\n🔍 Detected GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb >= 75:  # A100-80GB
        config = {
            'max_seq_len': 300000,  # Full 450k!
            'batch_size': 2,
            'accumulate_grad_batches': 8,
            'device_name': 'A100-80GB',
            'use_gradient_checkpointing': False,
        }
        print("   ✅ A100-80GB: Full 450k sequences, batch_size=2")
        
    elif gpu_memory_gb >= 35:  # A100-40GB
        config = {
            'max_seq_len': 300000,  # Full 450k!
            'batch_size': 1,
            'accumulate_grad_batches': 16,
            'device_name': 'A100-40GB',
            'use_gradient_checkpointing': True,
        }
        print("   ✅ A100-40GB: Full 450k sequences with gradient checkpointing")
        
    elif gpu_memory_gb >= 14:  # T4, RTX 3080 (16GB)
        config = {
            'max_seq_len': 300000,  # YES! Full 450k!
            'batch_size': 1,
            'accumulate_grad_batches': 32,
            'device_name': 'T4/RTX3080',
            'use_gradient_checkpointing': True,  # Critical!
            'empty_cache_freq': 1,  # Clear cache every batch
        }
        print("   ✅ T4: Full 450k sequences (with optimizations)")
        print("      - Frozen encoder")
        print("      - Gradient checkpointing enabled")
        print("      - Aggressive memory management")
        
    else:  # <12GB
        config = {
            'max_seq_len': 100000,  # Reduced
            'batch_size': 1,
            'accumulate_grad_batches': 48,
            'device_name': 'Small GPU',
            'use_gradient_checkpointing': True,
        }
        print("   ⚠️  Limited memory: 100k sequences max")
    
    print(f"   Effective batch size: {config['batch_size'] * config['accumulate_grad_batches']}")
    return config


def print_memory_breakdown(model, optimizer=None):
    """Print detailed memory breakdown."""
    print("\n" + "="*80)
    print("MEMORY BREAKDOWN")
    print("="*80)
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Estimate memory (4 bytes per float32 parameter)
    model_memory_gb = (total_params * 4) / (1024**3)
    trainable_memory_gb = (trainable_params * 4) / (1024**3)
    frozen_memory_gb = (frozen_params * 4) / (1024**3)
    
    print(f"Model Parameters:")
    print(f"  Total params: {total_params:,} ({model_memory_gb:.2f} GB)")
    print(f"  Trainable params: {trainable_params:,} ({trainable_memory_gb:.2f} GB)")
    print(f"  Frozen params: {frozen_params:,} ({frozen_memory_gb:.2f} GB)")
    
    # Optimizer state (Adam stores 2 states per parameter)
    if optimizer is not None:
        optimizer_memory_gb = (trainable_params * 2 * 4) / (1024**3)
        print(f"\nOptimizer State (Adam):")
        print(f"  Memory: {optimizer_memory_gb:.2f} GB (2x trainable params)")
    
    # Gradients
    gradients_memory_gb = (trainable_params * 4) / (1024**3)
    print(f"\nGradients:")
    print(f"  Memory: {gradients_memory_gb:.2f} GB")
    
    # Total baseline
    baseline_gb = model_memory_gb
    if optimizer is not None:
        baseline_gb += optimizer_memory_gb
    baseline_gb += gradients_memory_gb
    
    print(f"\nTotal Baseline (before forward pass): {baseline_gb:.2f} GB")
    print(f"Available for forward pass: {27.2 - baseline_gb:.2f} GB")
    print("="*80 + "\n")


class Config:
    """Configuration class that loads settings from config.yaml."""
    def __init__(self):
        # GPU Configuration
        self.gpu_config = get_gpu_config()
        
        # Paths
        self.data_dir = Path(__file__).parent.parent / CONFIG['data']['data_dir']
        self.output_dir = Path(__file__).parent.parent / CONFIG['output']['output_dir']
        
        # Model architectures
        model_cfg = CONFIG['model']
        self.model_name = model_cfg['name']
        self.hidden_size = model_cfg.get('hidden_size', 768)
        self.num_layers = model_cfg.get('num_layers', 24)
        self.max_seq_len = model_cfg['max_seq_len']
        
        # MPS-specific memory optimizations: smaller prediction heads
        if torch.backends.mps.is_available():
            self.splice_hidden = model_cfg.get('splice_hidden', 64)   # Was 128
            self.splice_layers = 1
            self.tss_hidden = model_cfg.get('tss_hidden', 32)         # Was 64
            self.tss_layers = 1
            self.polya_hidden = model_cfg.get('polya_hidden', 32)     # Was 64
            self.polya_layers = 1
            print("⚠️  MPS: Using smaller prediction heads for memory efficiency")
        else:
            self.splice_hidden = model_cfg.get('splice_hidden', 128)
            self.splice_layers = 1
            self.tss_hidden = model_cfg.get('tss_hidden', 64)
            self.tss_layers = 1
            self.polya_hidden = model_cfg.get('polya_hidden', 64)
            self.polya_layers = 1
        
        self.dropout = 0.1
        
        # Memory optimization - UPDATED
        self.use_gradient_checkpointing = model_cfg.get('use_gradient_checkpointing', True)
        self.empty_cache_freq = self.gpu_config.get('empty_cache_freq', 0)
        
        # Memory optimization - NEW HYBRID APPROACH
        self.encoder_chunk_size = model_cfg.get('encoder_chunk_size', 300000)
        self.prediction_chunk_size = model_cfg.get('prediction_chunk_size', 50000)
        self.chunk_overlap = model_cfg.get('chunk_overlap', 2000)
        self.auto_chunk = model_cfg.get('auto_chunk', True)
        
        # DEBUG: Print config values
        print("\n" + "="*80)
        print("CHUNKING CONFIGURATION:")
        print(f"  encoder_chunk_size: {self.encoder_chunk_size:,}")
        print(f"  prediction_chunk_size: {self.prediction_chunk_size:,}")
        print(f"  auto_chunk: {self.auto_chunk}")
        print(f"  max_seq_len: {self.max_seq_len:,}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print("="*80 + "\n")
        
        # Loss weights
        self.w_splice_donor = 1.0
        self.w_splice_acceptor = 1.0
        self.w_tss = 0.5
        self.w_polya = 0.5
        self.w_splice_effect = 1.0
        self.pos_weight = 20.0  # For positive class in BCEWithLogitsLoss

        # Phase 1: Protein prediction
        if torch.backends.mps.is_available():
            self.protein_hidden = model_cfg.get('protein_hidden', 64)  # Was 256
            self.protein_layers = 1  # Was 2
        else:
            self.protein_hidden = model_cfg.get('protein_hidden', 256)
            self.protein_layers = 2
        
        # Phase 1: Loss weights
        self.w_protein = 2.0
        self.w_cds_start = 0.5
        self.w_cds_end = 0.5
        self.w_nmd = 1.0
        self.w_expression = 1.0
        
        # Coupling parameters
        self.coupling_strength = 0.1  # Controls strength of coupling between tasks
        self.consistency_weight = 0.1  # Weight for consistency loss
        
        # Training configuration
        train_cfg = CONFIG['training']
        self.limit_val_batches = train_cfg['limit_val_batches']
        self.limit_train_batches = train_cfg['limit_train_batches']
        self.batch_size = train_cfg.get('batch_size', 1)  # Keep at 1 for long sequences
        self.accumulate_grad_batches = train_cfg.get('accumulate_grad_batches', 8)  # Simulate larger batch
        self.num_workers = train_cfg['num_workers']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.max_epochs = train_cfg['max_epochs']
        self.gradient_clip_val = train_cfg['gradient_clip_val']
        self.precision = train_cfg.get('precision', '16-mixed')  # Use mixed precision by default
        self.devices = train_cfg.get('devices', 1)  # Use 1 device for MPS
        self.save_top_k = train_cfg.get('save_top_k', 1)
        self.monitor = train_cfg.get('monitor', 'val/loss/total')
        self.mode = train_cfg.get('mode', 'min')
        self.patience = train_cfg.get('patience', 5)
        self.freeze_encoder = train_cfg.get('freeze_encoder', True)
        
        # Performance optimization - NEW
        self.compile_model = train_cfg.get('compile_model', False)
        
        # Logging configuration
        self.log_level = train_cfg.get('log_level', 'INFO')
        self.verbose_batches = train_cfg.get('verbose_batches', False)
        
        # Set encoder dimension based on model name
        if 'medium' in self.model_name:
            self.encoder_dim = 768
        elif 'large' in self.model_name:
            self.encoder_dim = 1024
        else:  # small or base
            self.encoder_dim = 256
        
        # Print memory optimization strategy
        self._print_memory_strategy()
    
    def _print_memory_strategy(self):
        """Print the memory optimization strategy being used."""
        print("\n" + "="*80)
        print("MEMORY OPTIMIZATION CONFIGURATION")
        print("="*80)
        print(f"Max sequence length: {self.max_seq_len:,} bp")
        print(f"Encoder chunk size: {self.encoder_chunk_size:,} bp")
        print(f"Prediction chunk size: {self.prediction_chunk_size:,} bp")
        print(f"Chunk overlap: {self.chunk_overlap:,} bp")
        print(f"Auto chunking: {self.auto_chunk}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"Mixed precision: {self.precision}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.accumulate_grad_batches}")
        print(f"Effective batch size: {self.batch_size * self.accumulate_grad_batches}")
        print(f"Freeze encoder: {self.freeze_encoder}")
        
        # Calculate expected memory usage
        if self.max_seq_len <= self.encoder_chunk_size:
            print("\n✅ Using FULL CONTEXT processing (no encoder chunking)")
            print("   This provides best quality with full 300k context window")
        else:
            print("\n⚠️  Using CHUNKED ENCODER processing")
            print(f"   Sequences will be processed in {self.encoder_chunk_size:,} bp chunks")
        
        if self.max_seq_len > self.prediction_chunk_size:
            print(f"✅ Prediction heads will be chunked at {self.prediction_chunk_size:,} bp")
        else:
            print("✅ Prediction heads will process full sequence")
        
        print("="*80 + "\n")

