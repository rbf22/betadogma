#!/usr/bin/env python3
"""
create_mini_dataset.py - Create a minimal dataset for testing BetaDogma.

This script runs fetch_minimal_data.py to download a minimal dataset
and saves it in the data/data_mini directory.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    mini_data_dir = data_dir / "data_mini"
    
    # Create data_mini directory if it doesn't exist
    mini_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating minimal dataset in: {mini_data_dir}")
    
    # Run fetch_minimal_data.py with output directory set to data_mini
    cmd = [
        sys.executable, 
        str(script_dir / "fetch_minimal_data.py"),
        "--output-dir", str(mini_data_dir),
        "--chromosome", "chr21"  # Using chr21 as it's a smaller chromosome
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Minimal dataset created successfully!")
        print(f"Location: {mini_data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error creating minimal dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
