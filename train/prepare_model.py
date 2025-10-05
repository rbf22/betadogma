import torch
from pathlib import Path
from betadogma.model import BetaDogmaModel
from betadogma.core.encoder_nt import NTEncoder

print("--- Verifying Full Model Workflow ---")

# Define the path to the configuration file
config_path = str(Path(__file__).parent.parent / "src/betadogma/experiments/config/default.yaml")

# --- 1. Instantiate BetaDogmaModel from config ---
print("Loading BetaDogmaModel with NT backend...")
# Use the classmethod to load from config, which correctly sets d_in and attaches config
model = BetaDogmaModel.from_config_file(config_path)
model.eval()

# --- 2. Instantiate Encoder from config ---
print("Loading NTEncoder...")
encoder_config = model.config['encoder']
encoder = NTEncoder(model_id=encoder_config['model_id'])

# Verify that the embedding dimensions match
# The input dimension is stored in the LayerNorm layer of the head's projection module
head_d_in = model.splice_head.proj.norm.normalized_shape[0]
assert head_d_in == encoder.hidden_size, \
    f"Model d_in ({head_d_in}) does not match encoder hidden_size ({encoder.hidden_size})"

print("Model and encoder loaded successfully. Performing a forward pass...")

# --- 3. Create Dummy Data and Generate Embeddings ---
# The NT model has a max sequence length (e.g., 6k for some variants).
# Using a shorter sequence for this verification script is safer and faster.
dummy_sequence = "N" * 4096
print(f"Created a dummy sequence of length {len(dummy_sequence)}")

# Generate embeddings using the encoder. The NTEncoder expects a list of strings.
print("Generating embeddings from sequence...")
embeddings = encoder.forward([dummy_sequence])
print(f"Embeddings generated with shape: {embeddings.shape}")


# --- 4. Perform Forward Pass through BetaDogmaModel ---
# Perform a forward pass through the main model with the embeddings
with torch.no_grad():
    outputs = model(embeddings)

# --- 5. Verify Outputs ---
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