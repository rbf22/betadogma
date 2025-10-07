#!/usr/bin/env python3
"""
fetch_data.py - Download and prepare all required data for BetaDogma.

This script handles downloading and processing of:
- GENCODE annotations (GTF)
- Reference genome (FASTA)
- GTEx expression data
- GTEx junction data
- Example variant data

Usage:
    python fetch_data.py [--output-dir DIR] [--skip-existing]
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
    },
    "gtex": {
        "v8": {
            "base_url": "https://storage.googleapis.com/adult-gtex",
            "files": {
                "expression": "bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
                "samples": "annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
                "junctions": "bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz"
            }
        }
    },
    "variants": {
        "url": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr22.recalibrated_variants.annotated.vcf.gz",
        "index_url": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr22.recalibrated_variants.annotated.vcf.gz.tbi"
    }
}

class DataFetcher:
    def __init__(self, output_dir: str = "data/raw", skip_existing: bool = False, force: bool = False):
        """Initialize the data fetcher with output directory and options.
        
        Args:
            output_dir: Directory to save downloaded files
            skip_existing: If True, skip downloading files that already exist
            force: If True, always re-download files even if they exist
        """
        self.output_dir = Path(output_dir).resolve()
        self.skip_existing = skip_existing
        self.force = force
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Create necessary subdirectories."""
        dirs = {
            "gencode_dir": "gencode",
            "genome_dir": "genome",
            "gtex_dir": "gtex",
            "junctions_dir": "gtex/junctions",
            "variants_dir": "variants"
        }
        
        for attr, rel_path in dirs.items():
            path = self.output_dir / rel_path
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, path)
    
    def calculate_checksum(self, file_path: Path, algorithm: str = 'md5') -> str:
        """Calculate checksum of a file."""
        hash_func = getattr(hashlib, algorithm)()
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def get_remote_checksum(self, url: str) -> Optional[str]:
        """Try to get checksum from a .md5 or .sha256 file if it exists."""
        for ext in ['.md5', '.sha256']:
            try:
                checksum_url = f"{url}{ext}"
                with urlopen(checksum_url) as f:
                    checksum = f.read().decode('utf-8').split()[0]
                    print(f"  Found remote checksum at {checksum_url}")
                    return checksum
            except (URLError, HTTPError, IndexError):
                continue
        print(f"  No remote checksum found for {url}")
        return None

    def get_checksum_file_path(self, file_path: Path) -> Path:
        """Get the path for the checksum file."""
        return file_path.parent / f"{file_path.name}.md5"
        
    def verify_checksum(self, file_path: Path, expected_checksum: Optional[str] = None) -> bool:
        """Verify file checksum if expected_checksum is provided or local checksum exists."""
        if not file_path.exists():
            return False
            
        checksum_source = None
        # Try to read local checksum file if no checksum provided
        if not expected_checksum:
            checksum_file = self.get_checksum_file_path(file_path)
            if checksum_file.exists():
                try:
                    with open(checksum_file, 'r') as f:
                        expected_checksum = f.read().strip().split()[0]
                        checksum_source = "local"
                        print(f"  Using local checksum for {file_path.name}")
                except (IOError, IndexError) as e:
                    print(f"  Warning: Could not read checksum file {checksum_file}: {e}")
                    return True  # Skip verification if we can't read the checksum file
            else:
                return True  # No checksum to verify
        else:
            checksum_source = "remote"
                
        if not expected_checksum:
            return True  # Still no checksum to verify
            
        algorithm = 'md5' if len(expected_checksum) == 32 else ('sha256' if len(expected_checksum) == 64 else 'md5')
        actual_checksum = self.calculate_checksum(file_path, algorithm)
        
        print(f"  Verifying {file_path.name}:")
        print(f"    {checksum_source.capitalize()} checksum: {expected_checksum}")
        print(f"    Actual checksum:   {actual_checksum}")
        
        if actual_checksum != expected_checksum:
            print(f"✗ Checksum mismatch for {file_path.name}")
            return False
            
        print(f"✓ Checksum verified for {file_path.name}")
        return True

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing double slashes except after http(s):"""
        # Split the URL into protocol and the rest
        if '://' in url:
            protocol, rest = url.split('://', 1)
            # Remove double slashes in the path part
            rest = '/'.join(part for part in rest.split('/') if part)
            return f"{protocol}://{rest}"
        # If no protocol, just clean up the slashes
        return '/'.join(part for part in url.split('/') if part)

    def download_file(self, url: str, output_path: Path, description: str, expected_checksum: Optional[str] = None) -> bool:
        """
        Download a file with progress bar and checksum verification.
        
        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
            description: Description for progress bar
            expected_checksum: Optional expected checksum (MD5 or SHA256)
            
        Returns:
            bool: True if download and verification succeeded, False otherwise
        """
        """
        Download a file with progress bar and checksum verification.
        
        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
            description: Description for progress bar
            expected_checksum: Optional expected checksum (MD5 or SHA256)
            
        Returns:
            bool: True if download and verification succeeded, False otherwise
        """
        # Check if file exists and handle based on force/skip_existing flags
        if output_path.exists():
            if self.force:
                print(f"[FORCE] Re-downloading {output_path.name}")
                output_path.unlink()
            elif self.skip_existing:
                print(f"✓ {description} already exists")
                if not self.verify_checksum(output_path, expected_checksum):
                    print("  Warning: File exists but checksum verification failed")
                    if input("  Redownload? [y/N]: ").lower() != 'y':
                        return False
                    output_path.unlink()  # Remove the file to force redownload
                else:
                    return True
            elif self.verify_checksum(output_path, expected_checksum):
                print(f"✓ {description} exists and local checksum verified")
                return True
        
        # Normalize URL to remove any double slashes
        url = self.normalize_url(url)
        
        # Try to get checksum from remote if not provided
        if expected_checksum is None:
            expected_checksum = self.get_remote_checksum(url)
        
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
            
            # Verify checksum if available
            if expected_checksum and not self.verify_checksum(temp_path, expected_checksum):
                print(f"✗ Checksum verification failed for {output_path.name}")
                temp_path.unlink()
                return False
            
            # Move the temporary file to the final location
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            
            # Generate and store checksum if download was successful
            try:
                checksum = self.calculate_checksum(output_path, 'md5')
                checksum_file = self.get_checksum_file_path(output_path)
                with open(checksum_file, 'w') as f:
                    f.write(f"{checksum}  {output_path.name}\n")  # Add newline for consistency with md5sum
                print(f"✓ Generated and stored checksum for {output_path.name}:")
                print(f"  Checksum: {checksum}")
                print(f"  Stored in: {checksum_file.name}")
            except Exception as e:
                print(f"  Warning: Could not generate checksum for {output_path.name}: {e}")
            
            print(f"✓ Downloaded and verified {output_path.name}")
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
    
    def download_gencode(self) -> None:
        """Download GENCODE annotations and reference genome with checksum verification."""
        base_url = CONFIG["gencode"]["base_url"].format(release=CONFIG["gencode"]["release"])
        
        # Download checksum files first if they exist
        checksums = {}
        for file_type, filename in CONFIG["gencode"]["files"].items():
            checksum_url = f"{base_url}/{filename.format(release=CONFIG['gencode']['release'])}.md5"
            try:
                with urlopen(checksum_url) as f:
                    checksums[file_type] = f.read().decode('utf-8').split()[0]
            except (URLError, HTTPError):
                print(f"  Warning: No remote checksum available for {file_type}")
        
        # Download the actual files
        for file_type, filename in CONFIG["gencode"]["files"].items():
            url = f"{base_url}/{filename.format(release=CONFIG['gencode']['release'])}"
            output_file = self.gencode_dir / Path(filename).name
            checksum = checksums.get(file_type)
            
            if self.download_file(url, output_file, f"GENCODE {file_type}", checksum):
                # Uncompress if needed
                if output_file.suffix == '.gz':
                    uncompressed = output_file.with_suffix('')
                    self.gunzip_file(output_file, uncompressed)
                    
                    # Verify uncompressed file if we have a checksum
                    if checksum and uncompressed.exists():
                        print(f"  Verifying {uncompressed.name}...")
                        if not self.verify_checksum(uncompressed, checksum):
                            print(f"  Warning: Remote checksum verification failed for {uncompressed.name}")
                            uncompressed.unlink()
                            output_file.unlink()
                            print(f"  Removed corrupted files. Please try again.")
                            continue
                        print(f"  ✓ Verified {uncompressed.name}")
    
    def download_gtex(self) -> None:
        """Download GTEx data including expression, samples, and junctions with verification."""
        base_url = CONFIG["gtex"]["v8"]["base_url"]
        
        # Download expression and samples
        for file_type in ["expression", "samples"]:
            url = f"{base_url}/{CONFIG['gtex']['v8']['files'][file_type]}"
            output_file = self.gtex_dir / Path(CONFIG['gtex']['v8']['files'][file_type]).name
            
            # Try to get checksum if available
            checksum = None
            try:
                with urlopen(f"{url}.md5") as f:
                    checksum = f.read().decode('utf-8').split()[0]
            except (URLError, HTTPError):
                pass
            
            if self.download_file(url, output_file, f"GTEx {file_type}", checksum):
                if output_file.suffix == '.gz':
                    uncompressed = output_file.with_suffix('')
                    self.gunzip_file(output_file, uncompressed)
                    
                    # Verify uncompressed file if we have a checksum
                    if checksum and uncompressed.exists():
                        print(f"  Verifying {uncompressed.name}...")
                        if not self.verify_checksum(uncompressed, checksum):
                            print(f"  Warning: Checksum verification failed for {uncompressed.name}")
                            uncompressed.unlink()
                            output_file.unlink()
                            print(f"  Removed corrupted files. Please try again.")
                            continue
                        print(f"  ✓ Verified {uncompressed.name}")
        
        # Download junctions file
        jct_url = f"{base_url}/{CONFIG['gtex']['v8']['files']['junctions']}"
        jct_gz = self.junctions_dir / "GTEx_junctions.gct.gz"
        
        # Get checksum for junction file if available
        jct_checksum = None
        try:
            with urlopen(f"{jct_url}.md5") as f:
                jct_checksum = f.read().decode('utf-8').split()[0]
        except (URLError, HTTPError):
            pass
        
        self.download_file(jct_url, jct_gz, "GTEx junction data", jct_checksum)
    
    def download_variants(self) -> None:
        """Download example variant data with verification."""
        url = CONFIG["variants"]["url"]
        output_file = self.variants_dir / "ALL.chr22_GRCh38.genotypes.vcf.gz"
        
        # Get checksum if available
        checksum = None
        try:
            with urlopen(f"{url}.md5") as f:
                checksum = f.read().decode('utf-8').split()[0]
        except (URLError, HTTPError):
            pass
        
        if self.download_file(url, output_file, "Example variants", checksum):
            # Download tabix index if available
            try:
                index_url = f"{url}.tbi"
                index_file = f"{output_file}.tbi"
                print("  Downloading variant index file...")
                urlretrieve(index_url, index_file)
                
                # Verify index file
                if not os.path.exists(index_file) or os.path.getsize(index_file) == 0:
                    print("  ✗ Failed to download variant index file")
                    if os.path.exists(index_file):
                        os.remove(index_file)
                else:
                    print("  ✓ Downloaded variant index file")
            except Exception as e:
                print(f"  ✗ Could not download tabix index: {e}")

    @staticmethod
    @staticmethod
    def gunzip_file(input_path: Path, output_path: Path) -> None:
        """Decompress a gzipped file."""
        if output_path.exists():
            return
            
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✓ Decompressed {input_path.name}")
            input_path.unlink()  # Remove the gzipped file
        except Exception as e:
            print(f"✗ Failed to decompress {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare data for BetaDogma")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Base directory to save downloaded files (default: data/raw)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading files that already exist"
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all files, even if they exist"
    )
    args = parser.parse_args()
    
    # Create data fetcher instance
    fetcher = DataFetcher(
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        force=args.force
    )
    
    # Print header
    print("\n" + "="*80)
    print(f"Starting data download to: {args.output_dir}")
    if args.force:
        print("FORCE MODE: All files will be re-downloaded")
    elif args.skip_existing:
        print("SKIP-EXISTING MODE: Existing files will be skipped")
    print("="*80 + "\n")
    
    try:
        # Download data
        print("\n" + "-"*60)
        print("DOWNLOADING GENCODE ANNOTATIONS AND REFERENCE GENOME")
        print("-"*60)
        fetcher.download_gencode()
        
        print("\n" + "-"*60)
        print("DOWNLOADING GTEX DATA")
        print("-"*60)
        fetcher.download_gtex()
        
        print("\n" + "-"*60)
        print("DOWNLOADING EXAMPLE VARIANT DATA")
        print("-"*60)
        fetcher.download_variants()
        
        print("\n" + "="*80)
        print("✓ ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print("="*80)
    except Exception as e:
        print("\n" + "!"*60)
        print(f"ERROR: {str(e)}")
        print("!"*60)
        sys.exit(1)
    
    print(f"\nData has been saved to: {fetcher.output_dir}")
    return 0

if __name__ == "__main__":
    main()
