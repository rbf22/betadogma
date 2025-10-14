#!/usr/bin/env python3
"""
prepare_training_data.py - Process raw genomic data into training format.

This script processes downloaded genomic data (GENCODE, GTEx, variants) through
multiple stages to create training-ready datasets for BetaDogma.

Processing steps:
    1. gencode    - Create genomic windows with structural annotations
    2. gtex       - Process GTEx junction data and calculate PSI values
    3. variants   - Add genetic variants to base windows
    4. overlapping - Create overlapping windows from base windows
    5. aggregate  - Merge all data sources into final training format

Usage:
    # Run full pipeline
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml
    
    # Resume from specific step
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step variants
    
    # Force re-run specific steps
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step gtex --force
"""

import sys
import yaml
import json
import argparse
import logging
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from glob import glob

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TrainingDataPreparer:
    """Prepares training data through multiple processing steps."""
    
    # Define processing steps in order
    STEPS = [
        "gencode",
        "gtex",
        "variants",
        "splice_variants",  # UPDATED: was "pathogenic_variants"
        "overlapping_windows",
        "aggregate"
    ]
    
    def __init__(
        self,
        config_path: Path,
        from_step: Optional[str] = None,
        force: bool = False,
        debug: bool = False,
        log_file: Optional[Path] = None
    ):
        """Initialize the training data preparer.
        
        Args:
            config_path: Path to YAML configuration file
            from_step: Start from this step (skip previous steps)
            force: Force re-run even if outputs exist
            debug: Enable debug logging
            log_file: Path to log file (if None, log to console only)
        """
        self.config_path = config_path.resolve()
        self.config = self._load_config()
        self.from_step = from_step
        self.force = force
        self.debug = debug
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # Resolve paths
        self.paths = self._resolve_paths()
        
        # Checkpoint directory
        checkpoints_cfg = self.config.get('checkpoints', {})
        if checkpoints_cfg.get('enabled', True):
            checkpoint_dir = self._resolve_path(
                checkpoints_cfg.get('dir', '{paths.cache}/checkpoints')
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = None
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'steps_skipped': [],
            'steps_failed': []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        if 'processing' not in config:
            raise ValueError("Config missing 'processing' section")
        
        return config
    
    def _setup_logging(self, log_file: Optional[Path]) -> logging.Logger:
        """Setup logging configuration.
        
        Args:
            log_file: Path to log file (if None, console only)
        
        Returns:
            Configured logger
        """
        # Get logging config
        log_cfg = self.config.get('logging', {})
        log_level = logging.DEBUG if self.debug else getattr(
            logging, log_cfg.get('level', 'INFO')
        )
        log_format = log_cfg.get(
            'format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create logger
        logger = logging.getLogger('prepare_training_data')
        logger.setLevel(log_level)
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file = self._resolve_path(str(log_file))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
        
        return logger
    
    def _resolve_paths(self) -> Dict[str, Path]:
        """Resolve all paths from config.
        
        Returns:
            Dictionary of resolved paths
        """
        paths_cfg = self.config.get('paths', {})
        paths = {}
        
        for key, value in paths_cfg.items():
            paths[key] = self._resolve_path(value)
        
        return paths
    
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
    
    def _resolve_templates(self, value: Any) -> Any:
        """Replace {template} placeholders with values from config.
        
        Args:
            value: Value potentially containing {key.subkey} templates
        
        Returns:
            Value with templates replaced
        """
        if isinstance(value, str):
            pattern = r'\{([^}]+)\}'
            
            def replacer(match):
                key_path = match.group(1).split('.')
                
                # Special handling for paths
                if key_path[0] == 'paths' and len(key_path) > 1:
                    if key_path[1] in self.paths:
                        return str(self.paths[key_path[1]])
                
                # General template resolution
                obj = self.config
                for key in key_path:
                    if isinstance(obj, dict) and key in obj:
                        obj = obj[key]
                    else:
                        # Template not found, return original
                        return match.group(0)
                
                return str(obj)
            
            return re.sub(pattern, replacer, value)
        
        elif isinstance(value, dict):
            return {k: self._resolve_templates(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self._resolve_templates(v) for v in value]
        
        else:
            return value
    
    def check_prerequisites(self) -> bool:
        """Check that all required raw data files exist.
        
        Returns:
            True if all prerequisites are met
        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 Checking prerequisites...")
        self.logger.info("=" * 80)
        
        missing = []
        processing_cfg = self.config.get('processing', {})
        
        for step_name in self.STEPS:
            if step_name not in processing_cfg:
                continue
            
            step_cfg = processing_cfg[step_name]
            if not step_cfg.get('enabled', True):
                continue
            
            # Check input files for this step
            step_missing = self._check_step_prerequisites(step_name, step_cfg)
            missing.extend(step_missing)
        
        if missing:
            self.logger.error("\n❌ Missing required files:")
            for m in missing:
                self.logger.error(f"   - {m}")
            self.logger.error("\n💡 Run the download script first:")
            self.logger.error(f"   python scripts/download_data.py --config {self.config_path}")
            return False
        
        self.logger.info("✅ All required files present")
        self.logger.info("=" * 80)
        return True
    
    def _check_step_prerequisites(
        self,
        step_name: str,
        step_cfg: Dict[str, Any]
    ) -> List[str]:
        """Check prerequisites for a specific step.
        
        Only checks for raw input files (downloaded data), not intermediate outputs.
        Intermediate outputs are checked automatically by _is_step_complete().
        
        Args:
            step_name: Name of the step
            step_cfg: Configuration for this step
        
        Returns:
            List of missing files
        """
        missing = []
        kwargs = step_cfg.get('kwargs', {})
        
        # Resolve all path templates
        kwargs = self._resolve_templates(kwargs)
        
        # Define which files are "raw inputs" vs "intermediate outputs"
        RAW_INPUT_KEYS = {
            'gencode': ['fasta', 'gtf'],
            'gtex': ['junctions', 'gtf'],
            'variants': ['vcf'],
            'splice_variants': ['splicevar_vcf', 'gtf'],  # UPDATED: renamed from pathogenic_variants
            'overlapping_windows': [],  # Only uses intermediate outputs
            'aggregate': []  # Only uses intermediate outputs
        }
        
        # Get raw input keys for this step
        raw_keys = RAW_INPUT_KEYS.get(step_name, [])
        
        # Check raw input files only
        for key in raw_keys:
            if key in kwargs:
                path = Path(kwargs[key])
                if not path.exists():
                    missing.append(f"{step_name}.{key}: {path}")
                else:
                    self.logger.info(f"  ✓ Found {key}: {path.name}")
                    
                    # Special handling for FASTA files - check/create index
                    if key == 'fasta':
                        fai_path = Path(str(path) + '.fai')
                        if not fai_path.exists():
                            self.logger.warning(f"  ⚠️  FASTA index not found, creating...")
                            if not self._create_fasta_index(path):
                                self.logger.warning(f"     Could not create index automatically")
                                self.logger.warning(f"     Create with: samtools faidx {path}")
                        else:
                            self.logger.info(f"  ✓ Found FASTA index: {fai_path.name}")
                    
                    # Special handling for VCF files (including splicevar_vcf) - check/create index
                    elif key in ['vcf', 'splicevar_vcf', 'clinvar_vcf']:  # UPDATED: added splicevar_vcf
                        tbi_path = Path(str(path) + '.tbi')
                        csi_path = Path(str(path) + '.csi')
                        
                        if csi_path.exists():
                            self.logger.info(f"  ✓ Found CSI index: {csi_path.name}")
                        elif tbi_path.exists():
                            self.logger.info(f"  ✓ Found TBI index: {tbi_path.name}")
                        else:
                            self.logger.warning(f"  ⚠️  VCF index not found, creating...")
                            if self._create_vcf_index(path):
                                self.logger.info(f"  ✓ Index created successfully")
                            else:
                                self.logger.warning(f"     Could not create index automatically")
                                self.logger.warning(f"     Processing may be slow without an index")
                                self.logger.warning(f"     Create with:")
                                self.logger.warning(f"       bcftools index {path}  # Creates .csi")
                                self.logger.warning(f"       tabix -p vcf {path}     # Creates .tbi")
        
        return missing
    
    def _create_fasta_index(self, fasta_path: Path) -> bool:
        """Create an index for a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
        
        Returns:
            True if index was created successfully
        """
        self.logger.info(f"    Creating FASTA index for {fasta_path.name}...")
        
        # Try samtools first
        try:
            result = subprocess.run(
                ['samtools', 'faidx', str(fasta_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                fai_path = Path(str(fasta_path) + '.fai')
                if fai_path.exists():
                    self.logger.info(f"    ✓ FASTA index created: {fai_path.name}")
                    return True
            else:
                self.logger.debug(f"    samtools failed: {result.stderr}")
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"    samtools not available or failed: {e}")
        
        # Fall back to pyfaidx
        try:
            import pyfaidx
            pyfaidx.Faidx(str(fasta_path))
            fai_path = Path(str(fasta_path) + '.fai')
            if fai_path.exists():
                self.logger.info(f"    ✓ FASTA index created: {fai_path.name}")
                return True
        except Exception as e:
            self.logger.debug(f"    pyfaidx failed: {e}")
        
        return False
    
    def _create_vcf_index(self, vcf_path: Path) -> bool:
        """Create an index for a VCF file.
        
        Tries bcftools first (for CSI), then falls back to tabix (for TBI).
        
        Args:
            vcf_path: Path to VCF file
        
        Returns:
            True if index was created successfully
        """
        self.logger.info(f"    Creating VCF index for {vcf_path.name}...")
        
        csi_path = Path(str(vcf_path) + '.csi')
        tbi_path = Path(str(vcf_path) + '.tbi')
        
        # Try bcftools index (creates .csi - better for large files)
        try:
            result = subprocess.run(
                ['bcftools', 'index', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            if result.returncode == 0:
                if csi_path.exists():
                    self.logger.info(f"    ✓ CSI index created: {csi_path.name}")
                    self.logger.info(f"       (CSI format is recommended for large VCFs)")
                    return True
                elif tbi_path.exists():
                    self.logger.info(f"    ✓ TBI index created: {tbi_path.name}")
                    return True
            else:
                self.logger.debug(f"    bcftools failed: {result.stderr}")
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"    bcftools not available or failed: {e}")
        
        # Fall back to tabix (creates .tbi - older format)
        try:
            result = subprocess.run(
                ['tabix', '-p', 'vcf', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and tbi_path.exists():
                self.logger.info(f"    ✓ TBI index created: {tbi_path.name}")
                return True
            else:
                self.logger.debug(f"    tabix failed: {result.stderr}")
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"    tabix not available or failed: {e}")
        
        # Try pysam as last resort
        try:
            import pysam
            pysam.tabix_index(str(vcf_path), preset='vcf', force=True)
            
            if csi_path.exists():
                self.logger.info(f"    ✓ CSI index created: {csi_path.name}")
                return True
            elif tbi_path.exists():
                self.logger.info(f"    ✓ TBI index created: {tbi_path.name}")
                return True
        
        except Exception as e:
            self.logger.debug(f"    pysam indexing failed: {e}")
        
        return False
    
    def prepare(self) -> bool:
        """Run the full data preparation pipeline.
        
        Returns:
            True if pipeline completed successfully
        """
        self.logger.info("=" * 80)
        self.logger.info("🚀 BetaDogma Training Data Preparation")
        self.logger.info("=" * 80)
        self.logger.info(f"📝 Config: {self.config_path}")
        self.logger.info(f"📂 Cache: {self.paths.get('cache', 'N/A')}")
        self.logger.info(f"📂 Output: {self.paths.get('output', 'N/A')}")
        self.logger.info(f"🔄 Force re-run: {self.force}")
        if self.from_step:
            self.logger.info(f"⏭️  Starting from: {self.from_step}")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Get processing configuration
        processing_cfg = self.config.get('processing', {})
        
        # Determine which steps to run
        if self.from_step:
            if self.from_step not in self.STEPS:
                self.logger.error(f"❌ Invalid step: {self.from_step}")
                self.logger.error(f"   Valid steps: {', '.join(self.STEPS)}")
                return False
            
            start_idx = self.STEPS.index(self.from_step)
            steps_to_run = self.STEPS[start_idx:]
        else:
            steps_to_run = self.STEPS
        
        # Run each step
        for step_name in steps_to_run:
            # Check if step is configured
            if step_name not in processing_cfg:
                self.logger.warning(f"⚠️  Step '{step_name}' not configured, skipping")
                self.stats['steps_skipped'].append(step_name)
                continue
            
            step_cfg = processing_cfg[step_name]
            
            # Check if step is enabled
            if not step_cfg.get('enabled', True):
                self.logger.info(f"⊘ Step '{step_name}' disabled, skipping")
                self.stats['steps_skipped'].append(step_name)
                continue
            
            # Check if step should run
            if not self.force and self._is_step_complete(step_name, step_cfg):
                self.logger.info(f"✓ Step '{step_name}' already complete, skipping")
                self.logger.info(f"  (use --force to re-run)")
                self.stats['steps_skipped'].append(step_name)
                continue
            
            # Run the step
            self.logger.info("")
            self.logger.info("─" * 80)
            self.logger.info(f"🔧 Running step: {step_name}")
            self.logger.info("─" * 80)
            
            step_start_time = time.time()
            
            try:
                success = self._run_step(step_name, step_cfg)
                
                if success:
                    step_duration = time.time() - step_start_time
                    self.logger.info(f"✅ Step '{step_name}' completed in {step_duration:.1f}s")
                    self.stats['steps_completed'].append(step_name)
                    
                    # Create checkpoint
                    if self.checkpoint_dir:
                        self._create_checkpoint(step_name)
                else:
                    self.logger.error(f"❌ Step '{step_name}' failed")
                    self.stats['steps_failed'].append(step_name)
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Step '{step_name}' failed with exception:")
                self.logger.error(f"   {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                self.stats['steps_failed'].append(step_name)
                return False
        
        # Print summary
        self._print_summary()
        
        return len(self.stats['steps_failed']) == 0
    
    def _is_step_complete(self, step_name: str, step_cfg: Dict[str, Any]) -> bool:
        """Check if a step has already been completed.
        
        Args:
            step_name: Name of the step
            step_cfg: Configuration for this step
        
        Returns:
            True if step outputs exist and checkpoint exists
        """
        # Check checkpoint file
        if self.checkpoint_dir:
            checkpoint_file = self.checkpoint_dir / f"{step_name}.done"
            if not checkpoint_file.exists():
                return False
        
        # Check expected outputs
        kwargs = self._resolve_templates(step_cfg.get('kwargs', {}))
        output_dir = kwargs.get('out')
        
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.exists():
                return False
            
            # Check if directory has any parquet files
            parquet_files = list(output_path.glob('*.parquet'))
            if not parquet_files:
                return False
        
        return True
    
    def _run_step(self, step_name: str, step_cfg: Dict[str, Any]) -> bool:
        """Run a single processing step.
        
        Args:
            step_name: Name of the step
            step_cfg: Configuration for this step
        
        Returns:
            True if step succeeded
        """
        kwargs = self._resolve_templates(step_cfg.get('kwargs', {}))
        
        # Map steps to their script paths
        STEP_SCRIPTS = {
            'gencode': 'src/betadogma/data/prepare_gencode.py',
            'gtex': 'src/betadogma/data/prepare_gtex.py',
            'variants': 'src/betadogma/data/prepare_variants.py',
            'splice_variants': 'src/betadogma/data/prepare_splice_variants.py',  # UPDATED: renamed
            'overlapping_windows': 'src/betadogma/data/prepare_overlapping.py',
            'aggregate': 'src/betadogma/data/aggregate_data.py',
        }
        
        if step_name not in STEP_SCRIPTS:
            self.logger.error(f"❌ Unknown step: {step_name}")
            return False
        
        # Build script path (relative to project root)
        script_path = ROOT / STEP_SCRIPTS[step_name]
        
        if not script_path.exists():
            self.logger.error(f"❌ Script not found: {script_path}")
            return False
        
        # Build command with arguments
        cmd = [sys.executable, str(script_path)]
        
        # Convert kwargs to command-line arguments
        for key, value in kwargs.items():
            # Convert underscores to hyphens for CLI (Python style -> CLI style)
            cli_key = key.replace('_', '-')
            
            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{cli_key}')
            elif isinstance(value, (list, tuple)):
                # Multiple values for same argument
                for item in value:
                    cmd.extend([f'--{cli_key}', str(item)])
            elif value is not None and value != '':
                # Skip empty strings and None
                cmd.extend([f'--{cli_key}', str(value)])
        
        self.logger.info(f"  Executing: {' '.join(cmd)}")
        
        try:
            # Run the script
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=ROOT,  # Run from project root
                timeout=None  # No timeout, let it run
            )
            
            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:  # Skip empty lines
                        self.logger.info(f"    {line}")
            
            if result.returncode != 0:
                self.logger.error(f"  ❌ Script failed with exit code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line:  # Skip empty lines
                            self.logger.error(f"    {line}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"  ❌ Script timed out")
            return False
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed to run script: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
            
    def _create_checkpoint(self, step_name: str) -> None:
        """Create a checkpoint file for a completed step.
        
        Args:
            step_name: Name of the step
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint_file = self.checkpoint_dir / f"{step_name}.done"
        
        checkpoint_data = {
            'step': step_name,
            'completed_at': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'config_hash': self._hash_config()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"📝 Checkpoint created: {checkpoint_file}")
    
    def _hash_config(self) -> str:
        """Create a hash of the config for change detection.
        
        Returns:
            SHA256 hash of config
        """
        import hashlib
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 Pipeline Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"  Total time:      {duration:.1f}s ({duration/3600:.2f} hours)")
        self.logger.info(f"  Steps completed: {len(self.stats['steps_completed'])}")
        self.logger.info(f"  Steps skipped:   {len(self.stats['steps_skipped'])}")
        self.logger.info(f"  Steps failed:    {len(self.stats['steps_failed'])}")
        
        if self.stats['steps_completed']:
            self.logger.info(f"\n  ✅ Completed steps:")
            for step in self.stats['steps_completed']:
                self.logger.info(f"     - {step}")
        
        if self.stats['steps_skipped']:
            self.logger.info(f"\n  ⊘ Skipped steps:")
            for step in self.stats['steps_skipped']:
                self.logger.info(f"     - {step}")
        
        if self.stats['steps_failed']:
            self.logger.info(f"\n  ❌ Failed steps:")
            for step in self.stats['steps_failed']:
                self.logger.info(f"     - {step}")
        
        self.logger.info("=" * 80)
        
        if len(self.stats['steps_failed']) == 0:
            self.logger.info("✅ Pipeline completed successfully!")
            self.logger.info("")
            self.logger.info("📌 Training data ready at:")
            self.logger.info(f"   {self.paths.get('output', 'N/A')}")
            self.logger.info("")
        else:
            self.logger.info("❌ Pipeline failed. Check logs for details.")
            self.logger.info("")
    
    def validate_outputs(self) -> bool:
        """Validate that all expected outputs were created.
        
        Returns:
            True if all expected outputs exist
        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 Validating outputs...")
        self.logger.info("=" * 80)
        
        validation_cfg = self.config.get('validation', {})
        expected_outputs = validation_cfg.get('expected_outputs', {})
        
        all_valid = True
        
        for step_name, patterns in expected_outputs.items():
            self.logger.info(f"\n📦 {step_name}:")
            
            for pattern in patterns:
                # Resolve template
                pattern = self._resolve_templates(pattern)
                
                # Check if pattern matches any files
                matches = glob(pattern)
                
                if matches:
                    self.logger.info(f"  ✓ {pattern} ({len(matches)} files)")
                else:
                    self.logger.error(f"  ✗ {pattern} (no matches)")
                    all_valid = False
        
        # Check minimum counts
        minimum_counts = validation_cfg.get('minimum_counts', {})
        if minimum_counts:
            self.logger.info(f"\n📊 Checking minimum counts:")
            all_valid &= self._validate_minimum_counts(minimum_counts)
        
        self.logger.info("\n" + "=" * 80)
        if all_valid:
            self.logger.info("✅ All validations passed")
        else:
            self.logger.error("❌ Some validations failed")
        self.logger.info("=" * 80)
        
        return all_valid
    
    def _validate_minimum_counts(self, minimum_counts: Dict[str, int]) -> bool:
        """Validate minimum counts for various data types.
        
        Args:
            minimum_counts: Dictionary of expected minimum counts
        
        Returns:
            True if all minimums are met
        """
        # This would need to be implemented based on your data format
        # For now, just return True
        self.logger.info("  (Minimum count validation not yet implemented)")
        return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare training data for BetaDogma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Steps (in order):
  1. gencode           - Create genomic windows with structural annotations
  2. gtex              - Process GTEx junction data and calculate PSI values
  3. variants          - Add genetic variants to base windows
  4. splice_variants   - Add experimentally validated splice variants (SpliceVarDB)
  5. overlapping_windows - Create overlapping windows from base windows
  6. aggregate         - Merge all data sources into final training format

Examples:
  # Run full pipeline
  python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml
  
  # Resume from splice variants step
  python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step splice_variants
  
  # Force re-run from specific step
  python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step gtex --force
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--from-step",
        choices=TrainingDataPreparer.STEPS,
        help="Start from this step (skip previous steps)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file (default: output_dir/pipeline.log)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate outputs, don't run pipeline"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check prerequisites, don't run pipeline"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize preparer
        preparer = TrainingDataPreparer(
            config_path=args.config,
            from_step=args.from_step,
            force=args.force,
            debug=args.debug,
            log_file=args.log_file
        )
        
        # Run appropriate mode
        if args.check:
            success = preparer.check_prerequisites()
        elif args.validate_only:
            success = preparer.validate_outputs()
        else:
            success = preparer.prepare()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()