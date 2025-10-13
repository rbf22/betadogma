"""
DataFetcher - A modular data fetching utility for BetaDogma.

This module provides functionality to download and prepare various datasets
including GENCODE annotations, reference genomes, GTEx data, and variant data.
"""

import os
import gzip
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Optional, Callable, Any, Union
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd
from tqdm import tqdm

class DataFetcher:
    """A class to handle downloading and preparing various datasets."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Union[str, Path] = "data/raw", 
                 skip_existing: bool = False, force: bool = False):
        """Initialize the data fetcher with configuration and options.
        
        Args:
            config: Dictionary containing data source configurations
            output_dir: Directory to save downloaded files
            skip_existing: If True, skip downloading files that already exist
            force: If True, always re-download files even if they exist
        """
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.skip_existing = skip_existing
        self.force = force
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary subdirectories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_checksum(file_path: Path, algorithm: str = 'md5') -> str:
        """Calculate checksum of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use ('md5' or 'sha256')
            
        Returns:
            Hex digest of the file
        """
        hash_func = getattr(hashlib, algorithm)()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def verify_checksum(self, file_path: Path, expected_checksum: Optional[str] = None) -> bool:
        """Verify file checksum if expected_checksum is provided or local checksum exists.
        
        Args:
            file_path: Path to the file to verify
            expected_checksum: Optional expected checksum
            
        Returns:
            bool: True if checksum matches or no checksum to verify against, False otherwise
        """
        if not file_path.exists():
            return False
            
        if expected_checksum:
            algorithm = 'sha256' if len(expected_checksum) == 64 else 'md5'
            actual_checksum = self.calculate_checksum(file_path, algorithm)
            return actual_checksum == expected_checksum
            
        # Check for .md5 or .sha256 file
        for ext in ['.md5', '.sha256']:
            checksum_file = file_path.with_suffix(ext)
            if checksum_file.exists():
                with open(checksum_file) as f:
                    expected = f.read().split()[0]
                algorithm = 'sha256' if ext == '.sha256' else 'md5'
                actual_checksum = self.calculate_checksum(file_path, algorithm)
                return actual_checksum == expected
                
        return True  # No checksum to verify against

    def download_file(self, url: str, output_path: Path, description: str, 
                     expected_checksum: Optional[str] = None) -> bool:
        """Download a file with progress bar and checksum verification.
        
        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
            description: Description for progress bar
            expected_checksum: Optional expected checksum (MD5 or SHA256)
            
        Returns:
            bool: True if download and verification succeeded, False otherwise
        """
        if output_path.exists() and not self.force:
            # Always calculate and display checksum for existing files
            file_checksum = self.calculate_checksum(output_path, 'sha256')
            print(f"Found existing file: {output_path}")
            print(f"Checksum: {file_checksum}")
            
            if expected_checksum:
                if file_checksum == expected_checksum:
                    print("✅ Checksum matches expected value")
                else:
                    print("⚠️  Checksum does NOT match expected value")
                    print(f"   Expected: {expected_checksum}")
            
            if self.skip_existing:
                return True
                
            if self.verify_checksum(output_path, expected_checksum):
                return True
                
            print(f"Checksum verification failed for {output_path}, re-downloading...")

        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get the file size first for accurate progress reporting
            try:
                with urlopen(Request(url, method='HEAD')) as response:
                    file_size = int(response.headers.get('content-length', 0))
                    
                # Download with progress bar
                with tqdm(total=file_size, unit='B', unit_scale=True, 
                         unit_divisor=1024, miniters=1, desc=description) as t:
                    def update_progress(chunk_num, chunk_size, total_size):
                        t.update(chunk_num * chunk_size - t.n)
                    
                    urlretrieve(url, output_path, reporthook=update_progress)
                    
            except Exception as e:
                print(f"Error getting file size: {e}")
                # Fallback to simple download without progress bar
                urlretrieve(url, output_path)
                
            # Calculate and print checksum for verification
            file_checksum = self.calculate_checksum(output_path, 'sha256')
            print(f"Checksum for {output_path.name}: {file_checksum}")
            
            # Verify against expected checksum if provided
            if expected_checksum:
                if file_checksum != expected_checksum:
                    print(f"⚠️  Checksum verification failed for {output_path}")
                    print(f"   Expected: {expected_checksum}")
                    print(f"   Actual:   {file_checksum}")
                    return False
                print("✅ Checksum verified successfully")
                
            return True
            
        except (URLError, HTTPError) as e:
            print(f"Error downloading {url}: {e}")
            return False

    @staticmethod
    def gunzip_file(input_path: Path, output_path: Optional[Path] = None) -> bool:
        """Decompress a gzipped file.
        
        Args:
            input_path: Path to the gzipped file
            output_path: Path to save the decompressed file (default: input_path without .gz)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if output_path is None:
            output_path = input_path.with_suffix('')
            
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except (gzip.BadGzipFile, IOError) as e:
            print(f"Error decompressing {input_path}: {e}")
            return False

    def fetch(self) -> bool:
        """Main method to fetch all configured data sources.
        
        Returns:
            bool: True if all downloads were successful, False otherwise
        """
        success = True
        
        if 'gencode' in self.config:
            success &= self.fetch_gencode()
            
        if 'gtex' in self.config:
            success &= self.fetch_gtex()
            
        if 'variants' in self.config:
            success &= self.fetch_variants()
            
        return success

    def fetch_gencode(self) -> bool:
        """Fetch GENCODE annotations and reference genome."""
        if 'gencode' not in self.config:
            return True
            
        print("\n=== Fetching GENCODE data ===")
        config = self.config['gencode']
        base_url = config['base_url'].format(release=config['release'])
        success = True
        
        for file_type, file_name in config['files'].items():
            url = f"{base_url}/{file_name}"
            output_path = self.output_dir / file_name
            expected_checksum = config.get('checksums', {}).get(file_type)
            
            if not self.download_file(url, output_path, f"Downloading {file_type}", expected_checksum):
                success = False
                continue
                
            # Decompress if needed
            if output_path.suffix == '.gz':
                self.gunzip_file(output_path)
                
        return success

    def fetch_gtex(self) -> bool:
        """Fetch GTEx data including expression, samples, and junctions."""
        if 'gtex' not in self.config:
            return True
            
        print("\n=== Fetching GTEx data ===")
        config = self.config['gtex']
        base_url = config['base_url']
        success = True
        
        for file_type, rel_path in config['files'].items():
            url = f"{base_url}/{rel_path}"
            output_path = self.output_dir / 'gtex' / Path(rel_path).name
            expected_checksum = config.get('checksums', {}).get(file_type)
            
            if not self.download_file(url, output_path, f"Downloading GTEx {file_type}", expected_checksum):
                success = False
                continue
                
            # Decompress if needed and has .gz extension
            if output_path.suffix == '.gz':
                self.gunzip_file(output_path)
                
        return success

    def fetch_variants(self) -> bool:
        """Fetch example variant data."""
        if 'variants' not in self.config:
            return True
            
        print("\n=== Fetching variant data ===")
        config = self.config['variants']
        success = True
        
        for name, file_info in config['files'].items():
            url = file_info['url']
            output_path = self.output_dir / 'variants' / file_info.get('filename', url.split('/')[-1])
            expected_checksum = file_info.get('checksum')
            
            if not self.download_file(url, output_path, f"Downloading {name}", expected_checksum):
                success = False
                continue
                
            # Decompress if needed and has .gz extension
            if output_path.suffix == '.gz':
                self.gunzip_file(output_path)
                
        return success
