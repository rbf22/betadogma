#!/usr/bin/env python3
"""
create_mini_training_data.py - Create a minimal training dataset for BetaDogma.

This script generates a minimal training dataset using only chromosome 21
and a small window size to keep the dataset under 100MB.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Define paths
    root_dir = Path(__file__).parent.parent
    train_script = root_dir / "train" / "make_training_data.py"
    config_file = root_dir / "train" / "configs" / "data.mini.yaml"
    output_dir = root_dir / "data" / "data_mini" / "processed"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating minimal training dataset in: {output_dir}")
    
    # Run the training data preparation script
    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_file),
        "--force"  # Force regeneration if output exists
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=root_dir)
        print("\n✓ Minimal training dataset created successfully!")
        print(f"Location: {output_dir}")
        
        # Calculate and print the total size
        total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error creating training dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
