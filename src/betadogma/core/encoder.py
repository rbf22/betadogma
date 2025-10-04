import torch
import torch.nn as nn
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot

class BetaDogmaEncoder(nn.Module):
    """
    Wrapper for the Enformer model, loaded using the enformer-pytorch library.
    This serves as the primary feature encoder for the BetaDogma framework.
    """
    def __init__(self, model_name: str = "EleutherAI/enformer-official-rough"):
        super().__init__()
        self.model_name = model_name
        try:
            # Load the model using the custom loader from enformer-pytorch
            # use_tf_gamma=False is needed for compatibility with the ported weights.
            self.transformer = from_pretrained(self.model_name, use_tf_gamma=False)
            self.transformer.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Please ensure you are connected to the internet and the model name is correct.")
            raise

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the Enformer model to get embeddings.
        The enformer-pytorch model expects one-hot encoded sequences.

        Args:
            input_ids (torch.Tensor): A tensor of token indices of shape
                                      (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): This argument is ignored
                                                     but kept for API consistency.

        Returns:
            torch.Tensor: The embeddings from the Enformer model.
        """
        # Convert token indices to one-hot encoding
        one_hot_input = seq_indices_to_one_hot(input_ids)

        # The model returns (predictions, embeddings) when return_embeddings=True
        _, embeddings = self.transformer(one_hot_input, return_embeddings=True)
        return embeddings