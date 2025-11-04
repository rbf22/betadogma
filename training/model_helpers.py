"""
model_helpers.py - Prediction heads for Central Dogma modeling

Contains all prediction head classes to keep train.py clean.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PredictionHead(nn.Module):
    """Standard prediction head for binary classification (splice sites, TSS, polyA)."""
    
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int, 
                 dropout: float = 0.1, use_checkpointing: bool = False):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        
        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def _forward_impl(self, x):
        """Actual forward computation."""
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out).squeeze(-1)
        return logits
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class ProteinPredictionHead(nn.Module):
    """Protein sequence prediction head (21-way classification per position).
    
    Predicts amino acid at each codon position:
    - 20 amino acids + stop codon = 21 classes
    - Uses ignore_index=-1 for non-CDS regions
    """
    
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int,
                 num_classes: int = 21, dropout: float = 0.1, 
                 use_checkpointing: bool = False):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def _forward_impl(self, x):
        """Actual forward computation."""
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # [B, L, 21]
        return logits
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class ScalarPredictionHead(nn.Module):
    """Scalar prediction head for sequence-level predictions (NMD, expression).
    
    Uses adaptive pooling to aggregate sequence information into a single value.
    """
    
    def __init__(self, d_in: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.pool(x).squeeze(-1)  # [B, D]
        return self.fc(x).squeeze(-1)  # [B]


class HyenaDNAEncoder(nn.Module):
    """HyenaDNA encoder wrapper with improved error handling."""
    
    def __init__(self, model_name: str, freeze: bool = True, device: str = None):
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"\n🔧 Initializing HyenaDNA: {model_name}")
        print(f"   Device: {self.device}")
        
        try:
            from transformers import AutoModel, AutoConfig
            
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map='auto' if str(self.device) != 'cpu' else None,
            ).to(self.device)
            
            print(f"   ✅ Model loaded successfully")
            
            self.frozen = freeze
            if freeze:
                print(f"   ✅ Encoder frozen (training heads only)")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
            else:
                print(f"   ✅ Encoder trainable")
                
        except Exception as e:
            print(f"   ❌ Error loading model: {str(e)}")
            raise
    
    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1]
            else:
                raise ValueError("Could not extract embeddings from model output")
            
            return embeddings
