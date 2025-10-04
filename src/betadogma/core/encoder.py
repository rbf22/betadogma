import torch
import torch.nn as nn
from transformers import AutoModel

class BetaDogmaEncoder(nn.Module):
    """
    A wrapper for a pre-trained genomic foundation model.

    This module loads a specified transformer model from the Hugging Face Hub
    and serves as the primary feature encoder for the BetaDogma framework.
    It is designed to be fine-tuned on downstream tasks.

    Args:
        model_name (str): The name of the pre-trained model to load from
                          the Hugging Face Hub (e.g., 'arm-genomics/enformer-finetuned-human-128k').
    """
    def __init__(self, model_name: str = "arm-genomics/enformer-finetuned-human-128k"):
        super().__init__()
        self.model_name = model_name
        try:
            self.transformer = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Please ensure you are connected to the internet and the model name is correct.")
            raise

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the transformer model.

        Args:
            input_ids (torch.Tensor): A tensor of token IDs of shape
                                      (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): A tensor indicating which
                                                     tokens to attend to.
                                                     Defaults to None.

        Returns:
            torch.Tensor: The last hidden state from the transformer model.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Return the last hidden state
        return outputs.last_hidden_state