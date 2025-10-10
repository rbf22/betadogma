#!/usr/bin/env python3
"""
fetch_minimal_data.py - Download and prepare a minimal dataset for BetaDogma.

This script handles downloading and processing of:
- GENCODE annotations (GTF) for a specific chromosome
- Reference genome (FASTA) for a specific chromosome

Usage:
    python fetch_minimal_data.py [--output-dir DIR] [--chromosome CHR]
"""

import os
import sys
import gzip
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd
from tqdm import tqdm

# Configuration
CONFIG = {
    "gencode": {
        "release": "44",
        "base_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}",
        "files": {
            "gtf": "gencode.v{release}.annotation.gtf.gz",
            "genome": "GRCh38.primary_assembly.genome.fa.gz"
        }
    }
}

class DataFetcher:
    def __init__(self, output_dir: str = "data/raw", chromosome: str = "chr21"):
        """Initialize the data fetcher with output directory and options.

        Args:
            output_dir: Directory to save downloaded files
            chromosome: The chromosome to filter for.
        """
        self.output_dir = Path(output_dir).resolve()
        self.chromosome = chromosome
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary subdirectories."""
        dirs = {
            "gencode_dir": "gencode",
            "genome_dir": "genome",
        }

        for attr, rel_path in dirs.items():
            path = self.output_dir / rel_path
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, path)

    def download_file(self, url: str, output_path: Path, description: str) -> bool:
        """
        Download a file with a progress bar.
        """
        if output_path.exists():
            print(f"✓ {description} already exists")
            return True

        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        try:
            print(f"• Downloading {description}...")
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=description) as pbar:
                def update_progress(count, block_size, total_size):
                    if pbar.total is None and total_size:
                        pbar.total = total_size
                    pbar.update(block_size)

                urlretrieve(url, filename=temp_path, reporthook=update_progress)

            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)

            print(f"✓ Downloaded {output_path.name}")
            return True

        except (URLError, HTTPError) as e:
            print(f"✗ Failed to download {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
        except Exception as e:
            print(f"✗ Error processing {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def filter_gtf(self, input_gtf_gz: Path, output_gtf_gz: Path):
        """Filter a GTF file for a specific chromosome."""
        if output_gtf_gz.exists():
            print(f"✓ Filtered GTF for {self.chromosome} already exists.")
            return

        print(f"Filtering GTF for {self.chromosome}...")
        with gzip.open(input_gtf_gz, 'rt') as f_in, gzip.open(output_gtf_gz, 'wt') as f_out:
            for line in f_in:
                if line.startswith('#'):
                    f_out.write(line)
                else:
                    fields = line.split('\t')
                    if fields[0] == self.chromosome:
                        f_out.write(line)
        print(f"✓ Created filtered GTF: {output_gtf_gz}")

    def filter_fasta(self, input_fa_gz: Path, output_fa: Path):
        """Filter a FASTA file for a specific chromosome."""
        if output_fa.exists():
            print(f"✓ Filtered FASTA for {self.chromosome} already exists.")
            return

        print(f"Filtering FASTA for {self.chromosome}...")
        with gzip.open(input_fa_gz, 'rt') as f_in, open(output_fa, 'w') as f_out:
            found_chr = False
            for line in f_in:
                if line.startswith('>'):
                    # Check if we are at the start of the desired chromosome
                    if line.startswith('>' + self.chromosome):
                        found_chr = True
                        f_out.write(line)
                    else:
                        # If we've already found our chromosome and hit a new one, stop.
                        if found_chr:
                            break
                elif found_chr:
                    f_out.write(line)
        print(f"✓ Created filtered FASTA: {output_fa}")


    def download_and_filter_gencode(self) -> None:
        """Download and filter GENCODE annotations and reference genome."""
        base_url = CONFIG["gencode"]["base_url"].format(release=CONFIG["gencode"]["release"])

        # Handle GTF
        gtf_url = f"{base_url}/{CONFIG['gencode']['files']['gtf'].format(release=CONFIG['gencode']['release'])}"
        gtf_filename = Path(CONFIG['gencode']['files']['gtf'].format(release=CONFIG['gencode']['release'])).name
        original_gtf_path = self.gencode_dir / gtf_filename
        filtered_gtf_path = self.gencode_dir / f"gencode.v{CONFIG['gencode']['release']}.{self.chromosome}.annotation.gtf.gz"

        if self.download_file(gtf_url, original_gtf_path, "GENCODE GTF"):
            self.filter_gtf(original_gtf_path, filtered_gtf_path)
            original_gtf_path.unlink() # Clean up large original file

        # Handle Genome
        genome_url = f"{base_url}/{CONFIG['gencode']['files']['genome'].format(release=CONFIG['gencode']['release'])}"
        genome_filename = Path(CONFIG['gencode']['files']['genome'].format(release=CONFIG['gencode']['release'])).name
        original_genome_path = self.genome_dir / genome_filename
        filtered_genome_path = self.genome_dir / f"GRCh38.primary_assembly.{self.chromosome}.genome.fa"

        if self.download_file(genome_url, original_genome_path, "Reference Genome"):
            self.filter_fasta(original_genome_path, filtered_genome_path)
            original_genome_path.unlink() # Clean up large original file


def main():
    parser = argparse.ArgumentParser(description="Download and prepare a minimal dataset for BetaDogma")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_minimal",
        help="Base directory to save downloaded files (default: data/raw_minimal)"
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr21",
        help="Chromosome to filter for (e.g., chr21)"
    )
    args = parser.parse_args()

    fetcher = DataFetcher(
        output_dir=args.output_dir,
        chromosome=args.chromosome
    )

    print("\n" + "="*80)
    print(f"Starting minimal data download to: {args.output_dir}")
    print(f"Filtering for chromosome: {args.chromosome}")
    print("="*80 + "\n")

    try:
        fetcher.download_and_filter_gencode()
        print("\n" + "="*80)
        print("✓ MINIMAL DATASET CREATED SUCCESSFULLY!")
        print("="*80)
    except Exception as e:
        print("\n" + "!"*60)
        print(f"ERROR: {str(e)}")
        print("!"*60)
        sys.exit(1)

    print(f"\nMinimal data for {args.chromosome} has been saved to: {fetcher.output_dir}")
    return 0

if __name__ == "__main__":
    main()