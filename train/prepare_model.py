"""
Model Preparation and Verification Script

This script serves as a quick end-to-end check of the model pipeline.
It performs the following steps:
1. Loads the default model configuration from YAML.
2. Instantiates the BetaDogmaModel using the `from_config_file` classmethod.
3. Creates dummy embeddings with the correct shape.
4. Performs a forward pass to get head outputs.
5. Runs the isoform decoder on the head outputs to generate candidate isoforms.
6. Prints status messages to confirm each step was successful.
"""
import torch
import yaml
from betadogma.model import BetaDogmaModel

# --- Configuration ---
DEFAULT_CONFIG_PATH = "src/betadogma/experiments/config/default.yaml"
SEQ_LEN = 4096 # A reasonable sequence length for a local test

def main():
    print("--- Verifying Full Model Workflow ---")

    # 1. Load config and instantiate model
    print(f"Loading BetaDogmaModel from config: {DEFAULT_CONFIG_PATH}")
    try:
        model = BetaDogmaModel.from_config_file(DEFAULT_CONFIG_PATH)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from config. {e}")
        return

    # 2. Create dummy embeddings
    # The embedding dimension `d_in` is specified in the config file.
    d_in = model.config.get("encoder", {}).get("hidden_size")
    dummy_embeddings = torch.randn(1, SEQ_LEN, d_in)
    print(f"Created dummy embeddings of shape: {dummy_embeddings.shape}")

    # 3. Perform a forward pass to get head outputs
    print("Performing a forward pass through the model heads...")
    with torch.no_grad():
        head_outputs = model(dummy_embeddings)

    # 4. Verify head output shapes
    print("Head outputs generated. Verifying shapes...")
    for head_name, outputs in head_outputs.items():
        if isinstance(outputs, dict):
            for sub_name, tensor in outputs.items():
                print(f"  - {head_name}.{sub_name}: {tensor.shape}")
        elif isinstance(outputs, torch.Tensor):
             print(f"  - {head_name}: {outputs.shape}")


    # 5. Run the isoform decoder
    print("\nRunning the isoform decoder on head outputs...")
    with torch.no_grad():
        # Decoder expects a specific shape, let's adjust
        for head in ['tss', 'polya']:
            if head in head_outputs:
                head_outputs[head][head] = head_outputs[head][head].permute(0, 2, 1) # (B, C, L) -> (B, L, C)

        candidates = model.isoform_decoder.decode(head_outputs, strand='+')

    print(f"Isoform decoder ran successfully and found {len(candidates)} candidate isoforms.")

    print("\n--- Verification Complete ---")
    print("Full model workflow verified successfully!")


if __name__ == "__main__":
    main()