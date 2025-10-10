#!/usr/bin/env python3
"""
create_minimal_dataset_direct.py - Create a minimal training dataset for BetaDogma.

This script creates a minimal training dataset directly without using the full pipeline.
It generates a small number of sequences with random labels for testing purposes.
"""
import os
import sys
import random
import gzip
import json
from pathlib import Path
import pyfaidx

# Set random seed for reproducibility
random.seed(42)

def generate_minimal_dataset(fasta_path: Path, gtf_path: Path, output_dir: Path, num_sequences: int = 100, seq_length: int = 1000):
    """Generate a minimal dataset with random sequences and labels."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the reference genome
    print(f"Loading reference genome from {fasta_path}...")
    genome = pyfaidx.Fasta(str(fasta_path))
    
    # Get chromosome 21 sequence
    chrom = 'chr21'
    if chrom not in genome:
        raise ValueError(f"Chromosome {chrom} not found in the reference genome")
    
    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)
    
    print(f"Chromosome {chrom} length: {chrom_len:,} bp")
    
    # Generate random sequences
    sequences = []
    for i in range(num_sequences):
        # Choose a random start position
        start = random.randint(0, chrom_len - seq_length - 1)
        end = start + seq_length
        
        # Extract sequence
        seq = str(chrom_seq[start:end]).upper()
        
        # Generate random labels (binary for simplicity)
        labels = [random.randint(0, 1) for _ in range(seq_length)]
        
        sequences.append({
            'id': f"seq_{i+1}",
            'chrom': chrom,
            'start': start,
            'end': end,
            'sequence': seq,
            'labels': labels
        })
    
    # Save sequences to a JSON file
    output_file = output_dir / 'minimal_dataset.json.gz'
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(sequences, f)
    
    print(f"\n✓ Generated {len(sequences)} sequences")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")

def main():
    # Define paths
    data_dir = Path(__file__).parent / 'data_mini'
    fasta_path = data_dir / 'genome' / 'GRCh38.primary_assembly.chr21.genome.fa'
    gtf_path = data_dir / 'gencode' / 'gencode.v44.chr21.annotation.gtf.gz'
    output_dir = data_dir / 'processed'
    
    # Generate the minimal dataset
    generate_minimal_dataset(fasta_path, gtf_path, output_dir)

if __name__ == "__main__":
    main()
