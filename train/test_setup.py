import torch
from betadogma.model import BetaDogmaModel
from betadogma.experiments.config import default as default_config
import yaml

# In a real scenario, you might load a specific config file.
# For this example, we'll use the default placeholder config.
# Note: The from_pretrained method is a placeholder and uses a default config.
model = BetaDogmaModel.from_pretrained("betadogma-base")
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor representing tokenized DNA
# (Batch size=1, Sequence length=1024)
dummy_input_ids = torch.randint(0, 5, (1, 1024))

# Perform a forward pass through the model
with torch.no_grad():
    outputs = model(dummy_input_ids)

# Print the shapes of the outputs from each head to verify
print("--- Model Head Output Shapes ---")
print(f"Splice (donor):   {outputs['splice']['donor'].shape}")
print(f"Splice (acceptor):{outputs['splice']['acceptor'].shape}")
print(f"TSS:              {outputs['tss']['tss'].shape}")
print(f"PolyA:            {outputs['polya']['polya'].shape}")
print(f"ORF (start):      {outputs['orf']['start'].shape}")
print(f"ORF (stop):       {outputs['orf']['stop'].shape}")
print(f"ORF (frame):      {outputs['orf']['frame'].shape}")