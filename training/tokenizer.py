

import torch

# ============================================================================
# Character Tokenizer
# ============================================================================

class CharacterTokenizer:
    """Character-level DNA tokenizer."""
    
    def __init__(self, max_length: int = 300000):
        self.max_length = max_length  # Hard cap at 450k
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.pad_token_id = 4
        
    def __call__(self, sequence: str, return_tensors: str = "pt", 
                 padding: str = "max_length", max_length: int = None,
                 truncation: bool = True):
        if max_length is None:
            max_length = self.max_length
        
        sequence = sequence.upper()
        
        if truncation and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        tokens = [self.vocab.get(char, 4) for char in sequence]
        attention_mask = [1] * len(tokens)
        
        if padding == "max_length":
            pad_length = max_length - len(tokens)
            if pad_length > 0:
                tokens = tokens + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        if return_tensors == "pt":
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
