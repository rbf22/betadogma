#!/usr/bin/env python3
"""
create_minimal_dataset_with_junctions.py - Create a minimal training dataset with junction data.

This script generates a minimal dataset including:
- Reference genome (chr21)
- GENCODE annotations (chr21)
- Simulated junction data for testing
"""
import os
import sys
import gzip
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Set random seed for reproducibility
random.seed(42)

def generate_junction_data(chrom: str, num_junctions: int = 50) -> List[Dict]:
    """Generate simulated junction data for testing."""
    junctions = []
    
    # Generate some realistic-looking junction positions
    for i in range(num_junctions):
        # Generate a random position on the chromosome
        # Using a large range that fits within chr21 (46.7Mbp)
        pos = random.randint(1_000_000, 40_000_000)
        
        # Generate donor and acceptor sites
        # Typical intron length is between 50-100,000 bp, but we'll use smaller for testing
        intron_length = random.randint(100, 10_000)
        
        # Randomly assign strand
        strand = random.choice(['+', '-'])
        
        if strand == '+':
            donor = pos
            acceptor = pos + intron_length
        else:
            donor = pos + intron_length
            acceptor = pos
        
        # Generate some sample counts (simulating multiple samples)
        num_samples = random.randint(3, 10)  # 3-10 samples per junction
        samples = [f"SAMPLE_{i+1}" for i in range(num_samples)]
        counts = [random.randint(5, 1000) for _ in range(num_samples)]
        
        junctions.append({
            'chrom': chrom,
            'start': min(donor, acceptor),
            'end': max(donor, acceptor),
            'donor': donor,
            'acceptor': acceptor,
            'strand': strand,
            'samples': samples,
            'counts': counts,
            'gene_id': f'ENSG{random.randint(10000000000, 99999999999)}.1',
            'transcript_id': f'ENST{random.randint(10000000000, 99999999999)}.1',
        })
    
    return junctions

def save_junction_gct(junctions: List[Dict], output_file: Path):
    """Save junction data in GCT format."""
    # Get unique sample names
    all_samples = sorted(list(set(s for j in junctions for s in j['samples'])))
    num_samples = len(all_samples)
    
    # Write GCT header
    with gzip.open(output_file, 'wt') as f:
        # GCT header
        f.write('#1.2\n')
        f.write(f'{len(junctions)}\t{num_samples}\n')
        
        # Sample names
        f.write('Name\tDescription' + '\t' + '\t'.join(all_samples) + '\n')
        
        # Junction data
        for i, j in enumerate(junctions):
            # Create a junction ID in the format: chr1:12345-23456:+
            jid = f"{j['chrom']}:{j['start']}-{j['end']}:{j['strand']}"
            
            # Initialize all sample counts to 0
            sample_counts = {s: 0 for s in all_samples}
            
            # Update with actual counts
            for s, c in zip(j['samples'], j['counts']):
                sample_counts[s] = c
            
            # Write the junction line
            f.write(f"{jid}\t{j['gene_id']}" + ''.join(f"\t{sample_counts[s]}" for s in all_samples) + '\n')
    
    print(f"Saved {len(junctions)} junctions to {output_file}")

def main():
    # Define paths
    data_dir = Path(__file__).parent / 'data_mini'
    
    # Create output directories
    gtex_dir = data_dir / 'raw' / 'gtex'
    gtex_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate junction data for chr21
    print("Generating junction data...")
    junctions = generate_junction_data('chr21', num_junctions=50)
    
    # Save in GCT format
    gct_file = gtex_dir / 'junctions' / 'GTEx_junctions.gct.gz'
    gct_file.parent.mkdir(exist_ok=True)
    save_junction_gct(junctions, gct_file)
    
    # Create a minimal dataset info file
    dataset_info = {
        'description': 'Minimal dataset for testing BetaDogma',
        'chromosomes': ['chr21'],
        'num_junctions': len(junctions),
        'files': {
            'genome': 'genome/GRCh38.primary_assembly.chr21.genome.fa',
            'gtf': 'gencode/gencode.v44.chr21.annotation.gtf.gz',
            'junctions': str(gct_file.relative_to(data_dir))
        },
        'size_mb': sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file()) / (1024 * 1024)
    }
    
    # Save dataset info
    with open(data_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n✓ Minimal dataset with junctions created successfully!")
    print(f"Location: {data_dir}")
    print(f"Total size: {dataset_info['size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
