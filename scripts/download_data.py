#!/usr/bin/env python3
"""
download_data.py - Download and verify genomic data files for BetaDogma.

This script downloads reference genomes, annotations, expression data, and variants
from public repositories with checksum verification, automatic decompression, and
index generation.

Supports both CSI (modern) and TBI (legacy) index formats for VCF files.

Usage:
    python scripts/download_data.py --config configs/data.whole_genome.yaml
    python scripts/download_data.py --config configs/data.whole_genome.yaml --force
    python scripts/download_data.py --config configs/data.whole_genome.yaml --verify-only
"""

import sys
import yaml
import hashlib
import argparse
import requests
import gzip
import shutil
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urljoin
from tqdm import tqdm


class DataDownloader:
    """Downloads and verifies genomic data files."""
    
    def __init__(self, config_path: Path, force: bool = False, verify_only: bool = False):
        """Initialize the data downloader.
        
        Args:
            config_path: Path to YAML configuration file
            force: If True, re-download even if files exist
            verify_only: If True, only verify existing files without downloading
        """
        self.config_path = config_path.resolve()
        self.config = self._load_config()
        self.force = force
        self.verify_only = verify_only
        
        # Get output directory from config
        self.raw_data_dir = self._resolve_path(
            self.config.get('paths', {}).get('raw', 'data/raw')
        )
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download settings
        download_cfg = self.config.get('download', {})
        self.skip_existing = download_cfg.get('skip_existing', True) and not force
        self.max_retries = download_cfg.get('max_retries', 3)
        self.timeout = download_cfg.get('timeout', 300)
        self.chunk_size = download_cfg.get('chunk_size', 8388608)  # 8MB
        
        # Statistics
        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'verified': 0,
            'total_bytes': 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        if 'download' not in config:
            raise ValueError("Config missing 'download' section")
        if 'sources' not in config['download']:
            raise ValueError("Config missing 'download.sources' section")
        
        return config
    
    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to the config file, with template substitution.
        
        Args:
            path_str: Path string, possibly with {templates}
        
        Returns:
            Resolved absolute Path
        """
        # Resolve templates like {paths.raw}
        path_str = self._resolve_templates(path_str)
        
        # Convert to Path
        path = Path(path_str)
        
        # If relative, make it relative to config file
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        
        return path
    
    def _resolve_templates(self, value: str) -> str:
        """Replace {template} placeholders with values from config.
        
        Args:
            value: String potentially containing {key.subkey} templates
        
        Returns:
            String with templates replaced
        """
        if not isinstance(value, str):
            return value
        
        pattern = r'\{([^}]+)\}'
        
        def replacer(match):
            key_path = match.group(1).split('.')
            obj = self.config
            
            for key in key_path:
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                else:
                    # Template not found, return original
                    return match.group(0)
            
            return str(obj)
        
        return re.sub(pattern, replacer, value)
    
    def download_all(self) -> bool:
        """Download all configured data files.
        
        Returns:
            True if all downloads succeeded, False otherwise
        """
        print("=" * 80)
        print("🌐 BetaDogma Data Downloader")
        print("=" * 80)
        print(f"📝 Config: {self.config_path}")
        print(f"📂 Output: {self.raw_data_dir}")
        print(f"🔄 Force re-download: {self.force}")
        print(f"🔍 Verify only: {self.verify_only}")
        print("=" * 80)
        print()
        
        # Get all sources
        sources = self.config['download']['sources']
        
        # Process each source
        for source_name, source_config in sources.items():
            print(f"\n{'─' * 80}")
            print(f"📦 Processing source: {source_name}")
            print(f"{'─' * 80}")
            
            self._process_source(source_name, source_config)
        
        # Print summary
        self._print_summary()
        
        return self.stats['failed'] == 0
    
    def _process_source(self, source_name: str, source_config: Dict[str, Any]) -> None:
        """Process all files from a single source.
        
        Args:
            source_name: Name of the source (e.g., 'genome', 'gencode')
            source_config: Configuration for this source
        """
        base_url = source_config.get('base_url', '')
        files = source_config.get('files', [])
        
        for file_info in files:
            # Check if file should be processed
            if not self._should_process_file(file_info):
                continue
            
            # Get file details
            filename = file_info['filename']
            output_path = self.raw_data_dir / file_info['output_path']
            
            # Construct URL
            if 'url_path' in file_info:
                url = urljoin(base_url + '/', file_info['url_path'])
            else:
                url = urljoin(base_url + '/', filename)
            
            checksum = file_info.get('checksum')
            required = file_info.get('required', True)
            
            # Download the file
            success = self._download_file(
                url=url,
                output_path=output_path,
                checksum=checksum,
                required=required
            )
            
            if success:
                # Post-processing
                if file_info.get('decompress', False):
                    self._decompress_file(output_path)
                
                if file_info.get('create_index', False):
                    self._create_fasta_index(output_path)
                
                # Special handling for VCF files - create index if needed
                if (filename.endswith('.vcf.gz') and 
                    not filename.endswith('.tbi') and 
                    not filename.endswith('.csi')):
                    self._create_vcf_index(output_path)
    
    def _should_process_file(self, file_info: Dict[str, Any]) -> bool:
        """Check if a file should be processed based on config.
        
        Args:
            file_info: File configuration dict
        
        Returns:
            True if file should be processed
        """
        # Check if file is marked as required
        if not file_info.get('required', True):
            # Optional file - check if it's needed for this config
            enabled_for = file_info.get('enabled_for', None)
            if enabled_for is not None:
                chromosome = self.config.get('chromosome', '')
                if chromosome not in enabled_for:
                    print(f"  ⊘ Skipping {file_info['filename']} (not needed for chromosome={chromosome})")
                    return False
        
        return True
    
    def _download_file(
        self,
        url: str,
        output_path: Path,
        checksum: Optional[str] = None,
        required: bool = True
    ) -> bool:
        """Download a single file with checksum verification.
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            checksum: Expected SHA256 checksum (if None, skip verification)
            required: If False, don't fail if download fails
        
        Returns:
            True if download succeeded (or was skipped), False otherwise
        """
        filename = output_path.name
        
        # Check if file exists and should be skipped
        if output_path.exists() and self.skip_existing and not self.verify_only:
            if checksum and self._verify_checksum(output_path, checksum):
                print(f"  ✓ {filename} (already exists, checksum valid)")
                self.stats['skipped'] += 1
                return True
            elif not checksum:
                print(f"  ✓ {filename} (already exists, no checksum to verify)")
                self.stats['skipped'] += 1
                return True
        
        # Verify only mode
        if self.verify_only:
            if not output_path.exists():
                print(f"  ✗ {filename} (missing)")
                self.stats['failed'] += 1
                return False
            elif checksum and not self._verify_checksum(output_path, checksum):
                print(f"  ✗ {filename} (checksum mismatch)")
                self.stats['failed'] += 1
                return False
            else:
                print(f"  ✓ {filename} (verified)")
                self.stats['verified'] += 1
                return True
        
        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with retries
        for attempt in range(self.max_retries):
            try:
                print(f"  ⬇ Downloading {filename} (attempt {attempt + 1}/{self.max_retries})")
                print(f"    URL: {url}")
                
                # Download to temporary file
                temp_path = output_path.with_suffix('.tmp')
                
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                # Get file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                with open(temp_path, 'wb') as f:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"    {filename}"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify checksum if provided
                if checksum:
                    print(f"    Verifying checksum...")
                    if not self._verify_checksum(temp_path, checksum):
                        temp_path.unlink()
                        raise ValueError("Checksum verification failed")
                    print(f"    ✓ Checksum valid")
                
                # Move to final location
                temp_path.replace(output_path)
                
                print(f"  ✓ {filename} (downloaded successfully)")
                self.stats['downloaded'] += 1
                self.stats['total_bytes'] += total_size
                return True
                
            except Exception as e:
                print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"    Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    if required:
                        print(f"  ✗ {filename} (download failed after {self.max_retries} attempts)")
                        self.stats['failed'] += 1
                        return False
                    else:
                        print(f"  ⊘ {filename} (optional file, skipping)")
                        return True
        
        return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify SHA256 checksum of a file.
        
        Args:
            file_path: Path to file to check
            expected_checksum: Expected SHA256 hex digest
        
        Returns:
            True if checksum matches
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b''):
                sha256.update(chunk)
        
        actual_checksum = sha256.hexdigest()
        return actual_checksum == expected_checksum
    
    def _decompress_file(self, file_path: Path) -> bool:
        """Decompress a .gz file.
        
        Args:
            file_path: Path to .gz file
        
        Returns:
            True if decompression succeeded
        """
        if not file_path.suffix == '.gz':
            return True
        
        output_path = file_path.with_suffix('')
        
        # Skip if already decompressed
        if output_path.exists() and self.skip_existing:
            print(f"    ✓ {output_path.name} (already decompressed)")
            return True
        
        print(f"    📦 Decompressing {file_path.name}...")
        
        try:
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    # Get uncompressed size for progress bar
                    try:
                        f_in.seek(-4, 2)
                        uncompressed_size = int.from_bytes(f_in.read(4), 'little')
                        f_in.seek(0)
                    except:
                        # If we can't get size, use compressed size * 3 as estimate
                        uncompressed_size = file_path.stat().st_size * 3
                    
                    with tqdm(
                        total=uncompressed_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"    Decompressing"
                    ) as pbar:
                        while True:
                            chunk = f_in.read(self.chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"    ✓ Decompressed to {output_path.name}")
            return True
            
        except Exception as e:
            print(f"    ✗ Decompression failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _create_fasta_index(self, file_path: Path) -> bool:
        """Create index for FASTA file (.fai).
        
        Args:
            file_path: Path to FASTA file (possibly .gz)
        
        Returns:
            True if index creation succeeded
        """
        # Determine the actual fasta file (decompressed)
        if file_path.suffix == '.gz':
            fasta_path = file_path.with_suffix('')
        else:
            fasta_path = file_path
        
        # Check if file exists
        if not fasta_path.exists():
            print(f"    ⚠ Cannot create index: {fasta_path} does not exist")
            return False
        
        # Check if already indexed
        index_path = Path(str(fasta_path) + '.fai')
        if index_path.exists() and self.skip_existing:
            print(f"    ✓ {index_path.name} (index already exists)")
            return True
        
        print(f"    🔖 Creating FASTA index for {fasta_path.name}...")
        
        # Try using samtools first
        try:
            result = subprocess.run(
                ['samtools', 'faidx', str(fasta_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and index_path.exists():
                print(f"    ✓ Index created: {index_path.name}")
                return True
            else:
                raise Exception(f"samtools failed: {result.stderr}")
                
        except FileNotFoundError:
            print(f"    ⚠ samtools not found, trying pyfaidx...")
        except subprocess.TimeoutExpired:
            print(f"    ⚠ samtools timed out, trying pyfaidx...")
        except Exception as e:
            print(f"    ⚠ samtools failed ({e}), trying pyfaidx...")
        
        # Fall back to pyfaidx
        try:
            import pyfaidx
            pyfaidx.Faidx(str(fasta_path))
            if index_path.exists():
                print(f"    ✓ Index created: {index_path.name}")
                return True
        except Exception as e:
            print(f"    ✗ Index creation failed: {e}")
            print(f"    💡 Manual creation:")
            print(f"       samtools faidx {fasta_path}")
            print(f"       # or install pyfaidx:")
            print(f"       pip install pyfaidx")
            return False
    
    def _create_vcf_index(self, vcf_path: Path) -> bool:
        """Create an index for a VCF file.
        
        Tries bcftools first (creates .csi - better for large files),
        then falls back to tabix (creates .tbi).
        
        Args:
            vcf_path: Path to VCF file
        
        Returns:
            True if index was created successfully
        """
        # Check if index already exists (either format)
        csi_path = Path(str(vcf_path) + '.csi')
        tbi_path = Path(str(vcf_path) + '.tbi')
        
        if self.skip_existing:
            if csi_path.exists():
                print(f"    ✓ {csi_path.name} (CSI index already exists)")
                return True
            elif tbi_path.exists():
                print(f"    ✓ {tbi_path.name} (TBI index already exists)")
                return True
        
        print(f"    🔖 Creating VCF index for {vcf_path.name}...")
        
        # Try bcftools index first (creates .csi - better for large chromosomes)
        try:
            result = subprocess.run(
                ['bcftools', 'index', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for large files
            )
            
            if result.returncode == 0:
                # Check what was created
                if csi_path.exists():
                    print(f"    ✓ CSI index created: {csi_path.name}")
                    print(f"       (CSI format is recommended for large VCFs)")
                    return True
                elif tbi_path.exists():
                    print(f"    ✓ TBI index created: {tbi_path.name}")
                    return True
            else:
                raise Exception(f"bcftools failed: {result.stderr}")
        
        except FileNotFoundError:
            print(f"    ⚠ bcftools not found, trying tabix...")
        except subprocess.TimeoutExpired:
            print(f"    ⚠ bcftools timed out, trying tabix...")
        except Exception as e:
            print(f"    ⚠ bcftools failed ({e}), trying tabix...")
        
        # Fall back to tabix (creates .tbi)
        try:
            result = subprocess.run(
                ['tabix', '-p', 'vcf', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and tbi_path.exists():
                print(f"    ✓ TBI index created: {tbi_path.name}")
                return True
            else:
                raise Exception(f"tabix failed: {result.stderr}")
        
        except FileNotFoundError:
            print(f"    ⚠ tabix not found, trying pysam...")
        except subprocess.TimeoutExpired:
            print(f"    ⚠ tabix timed out, trying pysam...")
        except Exception as e:
            print(f"    ⚠ tabix failed ({e}), trying pysam...")
        
        # Last resort: pysam
        try:
            import pysam
            pysam.tabix_index(str(vcf_path), preset='vcf', force=True)
            
            # Check what was created
            if csi_path.exists():
                print(f"    ✓ CSI index created: {csi_path.name}")
                return True
            elif tbi_path.exists():
                print(f"    ✓ TBI index created: {tbi_path.name}")
                return True
            else:
                raise Exception("Index file not found after pysam.tabix_index")
        
        except ImportError:
            print(f"    ⚠ pysam not installed")
        except Exception as e:
            print(f"    ⚠ pysam failed: {e}")
        
        # All methods failed
        print(f"    ✗ VCF index creation failed")
        print(f"    💡 Manual creation (choose one):")
        print(f"       bcftools index {vcf_path}  # Creates .csi (recommended)")
        print(f"       tabix -p vcf {vcf_path}     # Creates .tbi")
        print(f"    💡 Or install tools:")
        print(f"       conda install -c bioconda bcftools")
        print(f"       conda install -c bioconda tabix")
        print(f"       pip install pysam")
        return False
    
    def _print_summary(self) -> None:
        """Print download summary statistics."""
        print("\n" + "=" * 80)
        print("📊 Download Summary")
        print("=" * 80)
        print(f"  Downloaded:  {self.stats['downloaded']} files")
        print(f"  Skipped:     {self.stats['skipped']} files (already exist)")
        print(f"  Verified:    {self.stats['verified']} files")
        print(f"  Failed:      {self.stats['failed']} files")
        print(f"  Total data:  {self.stats['total_bytes'] / (1024**3):.2f} GB")
        print("=" * 80)
        
        if self.stats['failed'] == 0:
            print("✅ All files downloaded and verified successfully!")
            print()
            print("📌 Next step:")
            print(f"   python scripts/prepare_training_data.py --config {self.config_path}")
            print()
        else:
            print("❌ Some downloads failed. Check the logs above for details.")
            print()
    
    def verify_prerequisites(self) -> bool:
        """Verify that all required files exist and are valid.
        
        Returns:
            True if all prerequisites are met
        """
        print("\n" + "=" * 80)
        print("🔍 Verifying Prerequisites")
        print("=" * 80)
        
        sources = self.config['download']['sources']
        all_valid = True
        
        for source_name, source_config in sources.items():
            print(f"\n📦 {source_name}:")
            
            files = source_config.get('files', [])
            for file_info in files:
                if not self._should_process_file(file_info):
                    continue
                
                output_path = self.raw_data_dir / file_info['output_path']
                checksum = file_info.get('checksum')
                required = file_info.get('required', True)
                
                # Check file existence
                if not output_path.exists():
                    status = "✗ MISSING"
                    if required:
                        all_valid = False
                elif checksum and not self._verify_checksum(output_path, checksum):
                    status = "✗ INVALID CHECKSUM"
                    if required:
                        all_valid = False
                else:
                    status = "✓ OK"
                
                print(f"  {status:20} {output_path.name}")
                
                # Check for VCF index (CSI or TBI)
                if output_path.name.endswith('.vcf.gz'):
                    csi_path = Path(str(output_path) + '.csi')
                    tbi_path = Path(str(output_path) + '.tbi')
                    
                    if csi_path.exists():
                        print(f"  {'✓ OK':20} {csi_path.name} (CSI index)")
                    elif tbi_path.exists():
                        print(f"  {'✓ OK':20} {tbi_path.name} (TBI index)")
                    else:
                        print(f"  {'⚠ WARNING':20} No index found (.csi or .tbi)")
                        print(f"     Processing will be slow without an index")
                
                # Check for FASTA index
                if output_path.name.endswith('.fa') or output_path.name.endswith('.fasta'):
                    fai_path = Path(str(output_path) + '.fai')
                    if fai_path.exists():
                        print(f"  {'✓ OK':20} {fai_path.name} (FASTA index)")
                    else:
                        print(f"  {'⚠ WARNING':20} No FASTA index found (.fai)")
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ All required files present and valid")
        else:
            print("❌ Some required files are missing or invalid")
            print("\n💡 Run download script to fetch missing files:")
            print(f"   python scripts/download_data.py --config {self.config_path}")
        print("=" * 80)
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify genomic data files for BetaDogma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files specified in config
  python scripts/download_data.py --config configs/data.whole_genome.yaml
  
  # Force re-download even if files exist
  python scripts/download_data.py --config configs/data.whole_genome.yaml --force
  
  # Only verify existing files without downloading
  python scripts/download_data.py --config configs/data.whole_genome.yaml --verify-only
  
  # Check prerequisites before processing
  python scripts/download_data.py --config configs/data.whole_genome.yaml --check
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, don't download"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites and exit (no download)"
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize downloader
        downloader = DataDownloader(
            args.config,
            force=args.force,
            verify_only=args.verify_only
        )
        
        # Run appropriate mode
        if args.check:
            success = downloader.verify_prerequisites()
        else:
            success = downloader.download_all()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Download interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()