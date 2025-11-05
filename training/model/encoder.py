
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

# Setup logger
logger = logging.getLogger(__name__)

# ============================================================================
# Model with Gradient Checkpointing
# ============================================================================


class HyenaDNAEncoder(nn.Module):
    """Wrapper for the HyenaDNA model with memory optimizations for long sequences.
    
    This wrapper provides several key features:
    - Automatic device management (CPU/GPU)
    - Gradient checkpointing for memory efficiency
    - Detailed error reporting and recovery
    - Memory usage monitoring
    - Support for very long sequences (up to 300k tokens)
    """
    
    def __init__(self, model_name: str, device: str = None, use_gradient_checkpointing: bool = False, freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.frozen = freeze
        
        logger.info(f"Initializing HyenaDNA encoder: model={model_name}, device_type={device}, gradient_checkpointing={use_gradient_checkpointing}, freeze={freeze}")
        
        # Smart device selection
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        logger.debug(f"Selected device: {self.device}")
        
        try:
            # Initialize the model with memory-efficient settings
            logger.debug(f"Loading model weights from {model_name}...")
            
            # Clear any cached memory before loading the model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Import the correct Auto class - HyenaDNA needs AutoModel, not AutoModelForMaskedLM
            from transformers import AutoModel
            
            # Load model with memory-efficient settings
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                dtype=torch.float32,  # Use float32 for stability
                device_map=None  # Disable device_map to handle manually
            )
            
            # Move model to device
            logger.debug(f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing if requested
            if use_gradient_checkpointing:
                logger.debug(f"Enabling gradient checkpointing...")
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                else:
                    logger.warning(f"Model does not support gradient checkpointing")
            
            # Log model info
            logger.debug(f"Model class: {self.model.__class__.__name__}, device: {next(self.model.parameters()).device}, dtype: {next(self.model.parameters()).dtype}")
            
            # Get model config to determine hidden size
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'd_model'):
                    self.hidden_size = self.model.config.d_model
                    logger.debug(f"Hidden size (d_model): {self.hidden_size}")
                elif hasattr(self.model.config, 'hidden_size'):
                    self.hidden_size = self.model.config.hidden_size
                    logger.debug(f"Hidden size: {self.hidden_size}")
                else:
                    self.hidden_size = 768  # Default fallback
                    logger.warning(f"Could not determine hidden size, using default: {self.hidden_size}")
            else:
                self.hidden_size = 768
                logger.warning(f"No config found, using default hidden size: {self.hidden_size}")
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.debug(f"GPU memory: Allocated={torch.cuda.memory_allocated()/1e9:.2f} GB, Reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
            
            # Test with a small forward pass
            logger.debug(f"Testing forward pass...")
            with torch.no_grad():
                test_input = torch.zeros(1, 100, dtype=torch.long, device=self.device)
                test_output = self.model(test_input)
                
                # Check if output has the expected structure
                if hasattr(test_output, 'last_hidden_state'):
                    logger.debug(f"Output shape: {test_output.last_hidden_state.shape}")
                elif isinstance(test_output, tuple) and len(test_output) > 0:
                    logger.debug(f"Output shape (tuple): {test_output[0].shape}")
                    self.hidden_size = test_output[0].shape[-1]
                else:
                    logger.warning(f"Unexpected output type: {type(test_output)}")
            
            logger.info(f"Successfully loaded model: {model_name}, device: {next(self.model.parameters()).device}, hidden_size: {self.hidden_size}")
            
            # Freeze parameters if needed
            if freeze:
                logger.debug(f"Freezing encoder parameters")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            else:
                logger.debug(f"Keeping encoder parameters trainable")
            
        except Exception as e:
            logger.error(f"Failed to initialize HyenaDNA encoder: {type(e).__name__}: {str(e)}", exc_info=True)
            
            # Log memory stats if available
            if torch.cuda.is_available():
                try:
                    logger.error(f"GPU memory at failure: Allocated={torch.cuda.memory_allocated()/1e9:.2f} GB, Reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
                except Exception as me:
                    logger.error(f"Could not get GPU memory info: {str(me)}")
            
            # Re-raise the error with more context
            raise RuntimeError(f"Failed to initialize HyenaDNA model: {str(e)}") from e
    

    def forward(self, input_ids, attention_mask=None):
        logger.debug(f"HyenaDNAEncoder forward: input_shape={input_ids.shape}, device={input_ids.device}, dtype={input_ids.dtype}")
        
        # Ensure we're in eval mode if frozen
        if self.frozen:
            logger.debug(f"Setting model to eval mode (frozen)")
            self.model.eval()
        
        # Move inputs to the correct device
        logger.debug(f"Moving input to device...")
        input_ids = input_ids.to(self.device)
        
        # Ensure input is long type
        if input_ids.dtype != torch.long:
            logger.debug(f"Converting input dtype from {input_ids.dtype} to long")
            input_ids = input_ids.long()
        
        # Note: HyenaDNA doesn't use attention_mask, so we ignore it
        if attention_mask is not None:
            logger.debug(f"HyenaDNA doesn't use attention_mask (ignored)")
        
        try:
            logger.debug(f"Starting model forward...")
            with torch.no_grad() if self.frozen else torch.enable_grad():
                # Enable gradient checkpointing if not frozen
                if not self.frozen and hasattr(self.model, 'gradient_checkpointing_enable'):
                    logger.debug(f"Enabling gradient checkpointing")
                    self.model.gradient_checkpointing_enable()
                
                # HyenaDNA only takes input_ids, no attention_mask
                logger.debug(f"Running model forward pass...")
                outputs = self.model(input_ids)
                logger.debug(f"Forward pass completed")
                
                # Handle different output formats
                if hasattr(outputs, 'last_hidden_state'):
                    logger.debug(f"Using last_hidden_state from outputs")
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    logger.debug(f"Using first element from tuple output")
                    hidden_states = outputs[0]
                    # Create a simple object to hold the hidden states
                    class SimpleOutput:
                        def __init__(self, hidden_states):
                            self.last_hidden_state = hidden_states
                    outputs = SimpleOutput(hidden_states)
                elif isinstance(outputs, torch.Tensor):
                    logger.debug(f"Output is a tensor, wrapping it")
                    # Create a simple object to hold the hidden states
                    class SimpleOutput:
                        def __init__(self, hidden_states):
                            self.last_hidden_state = hidden_states
                    outputs = SimpleOutput(outputs)
                else:
                    raise ValueError(f"Unexpected output format: {type(outputs)}")
                
                logger.debug(f"Output shape: {outputs.last_hidden_state.shape}")
                return outputs
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.error(f"OUT OF MEMORY ERROR: input_shape={input_ids.shape}, batch_size={input_ids.size(0)}, seq_len={input_ids.size(1)}")
                
                # Clear cache and try again
                if torch.cuda.is_available():
                    logger.debug(f"Clearing CUDA cache...")
                    torch.cuda.empty_cache()
            
            raise

