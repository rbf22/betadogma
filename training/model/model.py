
import torch
import torch.nn as nn
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model.encoder import CaduceusEncoder
from model.heads import PredictionHead

# Setup logger
logger = logging.getLogger(__name__)

class BetaDogmaModel(nn.Module):
    """BetaDogma model with memory optimizations and splice effect prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        logger.info(f"Initializing BetaDogmaModel")
        logger.debug(f"  Sequence length: {config.max_seq_len}")
        logger.debug(f"  Hidden size: {config.hidden_size}")
        logger.debug(f"  Number of layers: {config.num_layers}")
        logger.debug(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
        logger.debug(f"  Encoder chunk size: {config.encoder_chunk_size}")
        logger.debug(f"  Prediction chunk size: {config.prediction_chunk_size}")
        logger.debug(f"  Mixed precision: {config.precision}")
        
        # Smart device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"🚀 Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info(f"🍎 Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            logger.info(f"💻 Using CPU")
        
        logger.debug(f"Using device: {self.device}")
        
        # Initialize encoder with the determined device
        self.encoder = CaduceusEncoder(
            model_name=config.model_name,
            freeze=config.freeze_encoder,
            device=str(self.device),
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        logger.debug("Model initialized on device: %s", self.device)

        # Caduceus wrapper reports the actual representation width
        self.encoder_dim = self.encoder.hidden_size
        logger.debug("Using encoder dimension: %s", self.encoder_dim)
        
        # Dropout for hidden states
        self.hidden_dropout = nn.Dropout(config.dropout)
        
        # Initialize all prediction heads
        self._init_prediction_heads()
        
        # Loss weights
        self.pos_weight = torch.tensor([10.0])  # Weight for positive class in BCE
        
        # Splice effect specific loss
        self.splice_effect_loss = nn.MSELoss()
        
        # Move all components to device
        self.to(self.device)
        
        logger.info(f"✅ BetaDogmaModel initialized successfully on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _init_prediction_heads(self):
        """Initialize all prediction heads."""
        use_checkpointing = self.config.use_gradient_checkpointing
        
        # Initialize prediction heads with proper dimensions
        self.donor_head = PredictionHead(
            self.encoder_dim, 
            self.config.splice_hidden,
            self.config.splice_layers, 
            self.config.dropout, 
            use_checkpointing
        )
        
        self.acceptor_head = PredictionHead(
            self.encoder_dim, 
            self.config.splice_hidden,
            self.config.splice_layers, 
            self.config.dropout, 
            use_checkpointing
        )
        
        # Splice effect prediction head (regression)
        self.splice_effect_head = PredictionHead(
            self.encoder_dim, 
            self.config.splice_hidden,
            self.config.splice_layers, 
            self.config.dropout, 
            use_checkpointing
        )
        
        # Other prediction heads
        self.tss_head = PredictionHead(
            self.encoder_dim, 
            self.config.tss_hidden,
            self.config.tss_layers, 
            self.config.dropout, 
            use_checkpointing
        )
        
        self.polya_head = PredictionHead(
            self.encoder_dim, 
            self.config.polya_hidden,
            self.config.polya_layers, 
            self.config.dropout, 
            use_checkpointing
        )
        
        # Phase 1: Protein prediction heads
        self.protein_head = nn.ModuleList([
            nn.LSTM(
                self.encoder_dim, 
                self.config.protein_hidden, 
                num_layers=self.config.protein_layers,
                bidirectional=True, 
                batch_first=True, 
                dropout=self.config.dropout if self.config.protein_layers > 1 else 0.0
            ),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.protein_hidden * 2, 21)  # 20 AA + stop
        ])
        
        self.cds_start_head = PredictionHead(
            self.encoder_dim, 
            self.config.protein_hidden, 
            1, 
            self.config.dropout, 
            use_checkpointing
        )
        
        self.cds_end_head = PredictionHead(
            self.encoder_dim, 
            self.config.protein_hidden, 
            1, 
            self.config.dropout, 
            use_checkpointing
        )
        
        # NMD and expression prediction
        self.nmd_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.config.protein_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.protein_hidden, 1)
        )
        
        self.expression_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.config.protein_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.protein_hidden, 1)
        )
        
        # Initialize weights for all heads
        for head in [self.donor_head, self.acceptor_head, self.splice_effect_head, 
                    self.tss_head, self.polya_head, self.cds_start_head, self.cds_end_head]:
            self._init_weights(head)
        
        # Initialize protein head
        for module in self.protein_head.modules():
            self._init_weights(module)
        
        # Initialize other heads
        for head in [self.nmd_head, self.expression_head]:
            self._init_weights(head)
            
        logger.debug(f"✅ BetaDogmaModel initialized with encoder_dim={self.encoder_dim}")
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_memory_usage(self):
        """Get current memory usage."""
        if torch.backends.mps.is_available():
            try:
                import psutil
                process = psutil.Process()
                mem = process.memory_info()
                return {
                    'allocated_gb': mem.rss / 1e9,
                    'percent': psutil.virtual_memory().percent
                }
            except:
                return {'allocated_gb': 0, 'percent': 0}
        elif torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            }
        else:
            return {'allocated_gb': 0, 'percent': 0}

    def forward(self, input_ids, attention_mask=None):
        """Hybrid forward pass with adaptive chunking."""
        logger.debug("=== 🚀 BETA DOGMA FORWARD PASS ===")
        
        # Validate inputs
        logger.debug(f"Input shape: {tuple(input_ids.shape)}, device: {input_ids.device}, dtype: {input_ids.dtype}")
        logger.debug(f"Training mode: {self.training}")
        
        if attention_mask is not None:
            logger.debug(f"Attention mask shape: {tuple(attention_mask.shape)}")
        
        # Move to device
        logger.debug("Moving to device...")
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Ensure correct data types
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if attention_mask is not None and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()
        
        batch_size, seq_len = input_ids.shape
        logger.debug(f"Batch size: {batch_size}, Sequence length: {seq_len:,} bp")
        
        # Decide chunking strategy - FIXED LOGIC
        logger.debug(f"Processing strategy: auto_chunk={self.config.auto_chunk}, encoder_chunk_size={self.config.encoder_chunk_size:,}, prediction_chunk_size={self.config.prediction_chunk_size:,}")
        
        # ALWAYS chunk predictions on MPS for 300k sequences
        is_mps = self.device.type == 'mps'
        
        # Force chunking for long sequences on MPS
        if is_mps and seq_len >= 300000:
            use_encoder_chunking = False  # Full context for encoder
            use_prediction_chunking = True  # ALWAYS chunk predictions on MPS
            logger.debug(f"🍎 MPS FORCED CHUNKING for {seq_len:,} bp sequence")
        else:
            use_encoder_chunking = self.config.auto_chunk and seq_len > self.config.encoder_chunk_size
            use_prediction_chunking = self.config.auto_chunk and seq_len > self.config.prediction_chunk_size
        
        logger.debug(f"Device: {self.device} (MPS: {is_mps}), Encoder chunking: {use_encoder_chunking}, Prediction chunking: {use_prediction_chunking}")
        
        # Clear cache before processing
        self._clear_memory_cache()
        
        # ENCODER PROCESSING
        logger.debug("🧬 ENCODER FORWARD")
        
        if use_encoder_chunking:
            # Only use encoder chunking for extremely long sequences (>300k)
            hidden_states = self._forward_encoder_chunked(input_ids, attention_mask)
        else:
            # Standard full-context processing with gradient checkpointing
            logger.debug(f"Processing full sequence with gradient checkpointing...")
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            hidden_states = encoder_outputs.last_hidden_state

            logger.debug("Encoder output shape: %s", tuple(hidden_states.shape))
        
        # Apply dropout
        hidden_states = self.hidden_dropout(hidden_states)
        
        # Clear cache after encoder
        self._clear_memory_cache()
        
        # PREDICTION HEADS PROCESSING
        logger.debug(f"🎯 PREDICTION HEADS: use_prediction_chunking={use_prediction_chunking}, is_mps={is_mps}")
        
        if use_prediction_chunking:
            if is_mps:
                logger.debug("Using MPS-safe chunking...")
                outputs = self._forward_predictions_chunked_mps_safe(hidden_states)
            else:
                logger.debug("Using standard chunking...")
                outputs = self._forward_predictions_chunked(hidden_states)
        else:
            logger.debug("Using full processing (NO CHUNKING)...")
            outputs = self._forward_predictions_full(hidden_states)
        
        # Sequence-level predictions (always use full sequence pooling)
        logger.debug("📊 SEQUENCE-LEVEL PREDICTIONS")
        pooled = hidden_states.mean(dim=1)
        outputs['nmd'] = self.nmd_head(pooled).squeeze(-1)
        outputs['expression'] = self.expression_head(pooled).squeeze(-1)
        
        logger.debug("✅ FORWARD PASS COMPLETED SUCCESSFULLY!")
        
        return outputs

    def _forward_encoder_chunked(self, input_ids, attention_mask=None):
        """Process encoder in chunks (only for extremely long sequences >300k)."""
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.encoder_chunk_size
        overlap = self.config.chunk_overlap
        
        logger.debug(f"Using encoder chunking for {seq_len:,} bp sequence, chunk_size={chunk_size:,}, overlap={overlap:,}")
        
        hidden_states_chunks = []
        
        for i in range(0, seq_len, chunk_size - overlap):
            start_idx = i
            end_idx = min(i + chunk_size, seq_len)
            
            logger.debug(f"Processing encoder chunk {start_idx:,} - {end_idx:,}")
            
            chunk_input = input_ids[:, start_idx:end_idx]
            chunk_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
            
            chunk_output = self.encoder(
                input_ids=chunk_input,
                attention_mask=chunk_mask
            )
            
            if hasattr(chunk_output, 'last_hidden_state'):
                chunk_hidden = chunk_output.last_hidden_state
            else:
                chunk_hidden = chunk_output
            
            # Handle overlap
            if i > 0 and overlap > 0:
                chunk_hidden = chunk_hidden[:, overlap//2:]
            if end_idx < seq_len and overlap > 0:
                chunk_hidden = chunk_hidden[:, :-overlap//2]
            
            hidden_states_chunks.append(chunk_hidden)
            self._clear_memory_cache()
        
        hidden_states = torch.cat(hidden_states_chunks, dim=1)
        logger.debug(f"Concatenated {len(hidden_states_chunks)} chunks -> {tuple(hidden_states.shape)}")
        
        return hidden_states

    # In _forward_predictions_full:
    def _forward_predictions_full(self, hidden_states):
        """Process all prediction heads without chunking."""
        logger.debug("Processing all prediction heads (no chunking)...")
        
        outputs = {
            'donor': self.donor_head(hidden_states),
            'acceptor': self.acceptor_head(hidden_states),
            'splice_effect': self.splice_effect_head(hidden_states),
            'tss': self.tss_head(hidden_states),
            'polya': self.polya_head(hidden_states),
            'cds_start': self.cds_start_head(hidden_states),
            'cds_end': self.cds_end_head(hidden_states),
        }
        
        # Protein prediction with LSTM - FIX: use ModuleList indexing
        protein_lstm_out, _ = self.protein_head[0](hidden_states)
        protein_lstm_out = self.protein_head[1](protein_lstm_out)
        outputs['protein'] = self.protein_head[2](protein_lstm_out)
        
        logger.debug("All predictions completed")
        return outputs

    def _forward_predictions_chunked_mps_safe(self, hidden_states):
        """MPS-safe chunked prediction processing using CPU offloading.
        
        Key insight: MPS has bugs with tensor slicing. Solution: do ALL slicing on CPU,
        then move individual chunks to MPS for processing.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = self.config.prediction_chunk_size
        
        logger.debug(f"Processing prediction heads in MPS-safe chunks of {chunk_size:,} bp...")
        logger.debug(f"Strategy: Copy to CPU, slice on CPU, process chunks on MPS")
        
        # CRITICAL: Move entire hidden_states to CPU FIRST to avoid MPS slicing bugs
        logger.debug(f"Step 1: Moving hidden states to CPU ({seq_len:,} positions)...")
        hidden_states_cpu = hidden_states.cpu()
        logger.debug(f"Hidden states on CPU")
        
        # Free MPS memory
        del hidden_states
        self._clear_memory_cache()
        
        # Initialize output tensors on CPU
        outputs_cpu = {
            'donor': torch.zeros(batch_size, seq_len),
            'acceptor': torch.zeros(batch_size, seq_len),
            'splice_effect': torch.zeros(batch_size, seq_len),
            'tss': torch.zeros(batch_size, seq_len),
            'polya': torch.zeros(batch_size, seq_len),
            'protein': torch.zeros(batch_size, seq_len, 21),
            'cds_start': torch.zeros(batch_size, seq_len),
            'cds_end': torch.zeros(batch_size, seq_len),
        }
        
        # Process in chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        logger.debug(f"Step 2: Processing {num_chunks} chunks...")
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_idx = i // chunk_size + 1
            logger.debug(f"Chunk {chunk_idx}/{num_chunks}: positions {i:,} - {end_idx:,}")
            
            # Slice on CPU (no MPS involved!)
            chunk_hidden_cpu = hidden_states_cpu[:, i:end_idx, :]
            
            # Move chunk to MPS for processing
            chunk_hidden_mps = chunk_hidden_cpu.to(self.device)
            
            # Process all heads for this chunk
            outputs_cpu['donor'][:, i:end_idx] = self.donor_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['acceptor'][:, i:end_idx] = self.acceptor_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['splice_effect'][:, i:end_idx] = self.splice_effect_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['tss'][:, i:end_idx] = self.tss_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['polya'][:, i:end_idx] = self.polya_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['cds_start'][:, i:end_idx] = self.cds_start_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            outputs_cpu['cds_end'][:, i:end_idx] = self.cds_end_head(chunk_hidden_mps).cpu()
            self._clear_memory_cache()
            
            # Protein prediction (the memory-hungry one)
            protein_lstm_out, _ = self.protein_head[0](chunk_hidden_mps)
            protein_lstm_out = self.protein_head[1](protein_lstm_out)
            outputs_cpu['protein'][:, i:end_idx, :] = self.protein_head[2](protein_lstm_out).cpu()
            
            # Free chunk from MPS
            del chunk_hidden_mps, protein_lstm_out
            self._clear_memory_cache()
            
            logger.debug(f"Chunk {chunk_idx} completed")
        
        # Free CPU copy
        del hidden_states_cpu
        
        # Move final outputs back to MPS
        logger.debug(f"Step 3: Moving outputs back to {self.device}...")
        outputs = {k: v.to(self.device) for k, v in outputs_cpu.items()}
        
        logger.debug("All MPS-safe chunked predictions completed")
        return outputs

    # In _forward_predictions_chunked:
    def _forward_predictions_chunked(self, hidden_states):
        """Process prediction heads in chunks for memory efficiency."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = self.config.prediction_chunk_size
        
        logger.debug(f"Processing prediction heads in chunks of {chunk_size:,} bp...")
        
        # Initialize output tensors
        outputs = {
            'donor': torch.zeros(batch_size, seq_len, device=self.device),
            'acceptor': torch.zeros(batch_size, seq_len, device=self.device),
            'splice_effect': torch.zeros(batch_size, seq_len, device=self.device),
            'tss': torch.zeros(batch_size, seq_len, device=self.device),
            'polya': torch.zeros(batch_size, seq_len, device=self.device),
            'protein': torch.zeros(batch_size, seq_len, 21, device=self.device),
            'cds_start': torch.zeros(batch_size, seq_len, device=self.device),
            'cds_end': torch.zeros(batch_size, seq_len, device=self.device),
        }
        
        # Process in chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        logger.debug(f"Number of chunks: {num_chunks}")
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_idx = i // chunk_size + 1
            logger.debug(f"Chunk {chunk_idx}/{num_chunks}: {i:,} - {end_idx:,}")
            
            chunk_hidden = hidden_states[:, i:end_idx, :]
            
            # Simple heads (they return squeezed tensors)
            outputs['donor'][:, i:end_idx] = self.donor_head(chunk_hidden)
            outputs['acceptor'][:, i:end_idx] = self.acceptor_head(chunk_hidden)
            outputs['splice_effect'][:, i:end_idx] = self.splice_effect_head(chunk_hidden)
            outputs['tss'][:, i:end_idx] = self.tss_head(chunk_hidden)
            outputs['polya'][:, i:end_idx] = self.polya_head(chunk_hidden)
            outputs['cds_start'][:, i:end_idx] = self.cds_start_head(chunk_hidden)
            outputs['cds_end'][:, i:end_idx] = self.cds_end_head(chunk_hidden)
            
            # Protein prediction (LSTM) - FIX: use ModuleList indexing
            protein_lstm_out, _ = self.protein_head[0](chunk_hidden)
            protein_lstm_out = self.protein_head[1](protein_lstm_out)
            outputs['protein'][:, i:end_idx, :] = self.protein_head[2](protein_lstm_out)
            
            self._clear_memory_cache()
        
        logger.debug("All chunked predictions completed")
        return outputs


    def _clear_memory_cache(self):
        """Clear GPU memory cache."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()



