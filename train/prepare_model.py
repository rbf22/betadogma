import torch
from betadogma.model import BetaDogmaModel

print("--- Verifying Full Model Workflow ---")
print("Loading BetaDogmaModel with Enformer backend...")

# This will now instantiate the full model, including the corrected encoder
# and the prediction heads that now expect the correct embedding size.
model = BetaDogmaModel(d_in=1536, config=config)
model.eval()

print("Model loaded successfully. Performing a forward pass...")

# Create a dummy input tensor representing tokenized DNA
# (Batch size=1, Sequence length=196,608, as expected by Enformer)
dummy_input_ids = torch.randint(0, 5, (1, 196_608))

# Perform a forward pass through the entire model
with torch.no_grad():
    outputs = model(dummy_input_ids)

# Print the shapes of the outputs from each head to verify
print("\n--- Verification Complete ---")
print("Output shapes from prediction heads:")
print(f"Splice (donor):   {outputs['splice']['donor'].shape}")
print(f"Splice (acceptor):{outputs['splice']['acceptor'].shape}")
print(f"TSS:              {outputs['tss']['tss'].shape}")
print(f"PolyA:            {outputs['polya']['polya'].shape}")
print(f"ORF (start):      {outputs['orf']['start'].shape}")
print(f"ORF (stop):       {outputs['orf']['stop'].shape}")
print(f"ORF (frame):      {outputs['orf']['frame'].shape}")
print("\nFull model workflow verified successfully!")