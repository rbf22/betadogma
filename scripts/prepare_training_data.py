#!/usr/bin/env python3
"""
Data preparation pipeline for BetaDogma training data.

This module provides a comprehensive pipeline for processing raw genomic data into
training-ready datasets for the BetaDogma model. It handles multiple data sources
including GENCODE annotations, GTEx junction data, and genetic variants to create
richly annotated training examples for splice prediction.

Processing Steps:
    1. gencode             - Create genomic windows with structural annotations
    2. gtex_junctions      - Process GTEx junction data and calculate PSI values
    3. population_variants - Add common genetic variants from 1000 Genomes
    4. clinvar_variants    - Add pathogenic variants from ClinVar
    5. splice_variants     - Add experimentally validated splice variants (SpliceVarDB)
    6. overlapping_windows - Create overlapping windows from base windows
    7. aggregate           - Merge all data sources into final training format

Example Usage:
    # Run full pipeline
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml
    
    # Resume from specific step
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step splice_variants
    
    # Force re-run specific steps
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --from-step gtex --force
    
    # Validate existing outputs
    python scripts/prepare_training_data.py --config configs/data.whole_genome.yaml --validate-only
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
from typing import Dict, Any, Optional, List
from datetime import datetime
from glob import glob

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TrainingDataPreparer:
    """Orchestrates the end-to-end training data preparation pipeline.
    
    This class manages the complete workflow for processing raw genomic data into
    training examples. It handles data loading, processing, validation, and
    checkpointing across multiple processing steps.
    
    The pipeline is designed to be resumable and supports running specific steps
    independently. It also includes validation and logging capabilities.
    
    Attributes:
        config_path (Path): Path to the YAML configuration file
        config (dict): Loaded configuration parameters
        from_step (str, optional): Step to start from (if resuming)
        force (bool): Whether to force re-run of steps
        debug (bool): Whether to enable debug logging
        logger (logging.Logger): Configured logger instance
        paths (dict): Resolved file and directory paths
        checkpoint_dir (Path, optional): Directory for saving checkpoints
        stats (dict): Pipeline execution statistics
        
    Class Attributes:
        STEPS (list): Ordered list of processing step names
    """
    
    # Define processing steps in order
    STEPS = [
        "gencode",
        "gtex_junctions",
        "population_variants",
        "clinvar_variants",
        "splice_variants",
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
        """Resolve all paths from config."""
        paths_cfg = self.config.get('paths', {})
        paths = {}
        
        for key, value in paths_cfg.items():
            paths[key] = self._resolve_path(value)
        
        return paths
    
    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to the config file, with template substitution."""
        path_str = self._resolve_templates(path_str)
        path = Path(path_str)
        
        # If relative, make it relative to config file
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        
        return path
    
    def _resolve_templates(self, value: Any) -> Any:
        """Replace {template} placeholders with values from config."""
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
                        return match.group(0)
                
                return str(obj)
            
            return re.sub(pattern, replacer, value)
        
        elif isinstance(value, dict):
            return {k: self._resolve_templates(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self._resolve_templates(v) for v in value]
        
        else:
            return value
    
    def _log_step_summary(self, step_name: str, stdout: str, stderr: str) -> None:
        """Log a summary of the step's execution."""
        summary_patterns = {
            'gencode': [
                (r'Total genes processed: (\d+)', 'Genes processed'),
                (r'Total transcripts: (\d+)', 'Transcripts found'),
                (r'Total exons: (\d+)', 'Exons found'),
                (r'wrote .*?/shard_\d+\.parquet \((\d+) rows\)', 'Genomic windows'),
            ],
            'gtex_junctions': [
                (r'Total junctions: (\d+)', 'Junctions processed'),
                (r'Samples with expression data: (\d+)', 'Samples'),
            ],
            'population_variants': [
                (r'Total variants: (\d+)', 'Variants processed'),
            ],
            'aggregate': [
                (r'Total training examples: (\d+)', 'Training examples'),
            ]
        }
        
        patterns = summary_patterns.get(step_name, [])
        output = stdout + '\n' + stderr
        matches = []
        
        for pattern, label in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                matches.append(f"{label}: {match.group(1).strip()}")
        
        if matches:
            self.logger.info("📊 Step Summary:")
            for match in matches:
                self.logger.info(f"   • {match}")
    
    def check_prerequisites(self) -> bool:
        """Check that all required raw data files exist."""
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
        """Check prerequisites for a specific step."""
        missing = []
        kwargs = step_cfg.get('kwargs', {})
        kwargs = self._resolve_templates(kwargs)
        
        # Define raw input files (not intermediate outputs)
        RAW_INPUT_KEYS = {
            'gencode': [
                ('fasta', 'Reference genome FASTA file'),
                ('gtf', 'GENCODE GTF annotation file')
            ],
            'gtex_junctions': [
                ('junctions', 'GTEx junctions file'),
                ('gtf', 'GENCODE GTF annotation file')
            ],
            'population_variants': [
                ('vcf', '1000 Genomes VCF file')
            ],
            'clinvar_variants': [
                ('clinvar_vcf', 'ClinVar VCF file'),
                ('gtf', 'GENCODE GTF annotation file')
            ],
            'splice_variants': [
                ('splicevar_vcf', 'SpliceVarDB VCF file'),
                ('gtf', 'GENCODE GTF annotation file')
            ],
        }
        
        raw_keys = dict(RAW_INPUT_KEYS.get(step_name, []))
        
        for key, description in raw_keys.items():
            if key in kwargs:
                path = Path(kwargs[key])
                if not path.exists():
                    missing.append(f"{step_name}.{key}: {description} not found at: {path}")
                    self.logger.error(f"  ✗ Missing {key}: {path}")
                else:
                    self.logger.info(f"  ✓ Found {key}: {path.name}")
                    
                    # Check/create FASTA index
                    if key == 'fasta':
                        fai_path = Path(str(path) + '.fai')
                        if not fai_path.exists():
                            self.logger.warning(f"  ⚠️  Creating FASTA index...")
                            if not self._create_fasta_index(path):
                                missing.append(f"{step_name}.{key}: Could not create FASTA index")
                        else:
                            self.logger.info(f"  ✓ Found FASTA index: {fai_path.name}")
                    
                    # Check/create VCF index
                    elif key in ['vcf', 'splicevar_vcf', 'clinvar_vcf']:
                        if not self._check_vcf_index(path):
                            self.logger.warning(f"  ⚠️  Creating VCF index...")
                            if not self._create_vcf_index(path):
                                missing.append(f"{step_name}.{key}: Could not create VCF index")
                        else:
                            self.logger.info(f"  ✓ Found VCF index")
        
        return missing
    
    def _check_vcf_index(self, vcf_path: Path) -> bool:
        """Check if VCF index exists."""
        return (Path(str(vcf_path) + '.tbi').exists() or 
                Path(str(vcf_path) + '.csi').exists())
    
    def _create_fasta_index(self, fasta_path: Path) -> bool:
        """Create FASTA index using samtools or pyfaidx."""
        try:
            result = subprocess.run(
                ['samtools', 'faidx', str(fasta_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return Path(str(fasta_path) + '.fai').exists()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback to pyfaidx
        try:
            import pyfaidx
            pyfaidx.Faidx(str(fasta_path))
            return Path(str(fasta_path) + '.fai').exists()
        except Exception:
            pass
        
        return False
    
    def _create_vcf_index(self, vcf_path: Path) -> bool:
        """Create VCF index using bcftools or tabix."""
        # Try bcftools
        try:
            result = subprocess.run(
                ['bcftools', 'index', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0 and self._check_vcf_index(vcf_path):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try tabix
        try:
            result = subprocess.run(
                ['tabix', '-p', 'vcf', str(vcf_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0 and self._check_vcf_index(vcf_path):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return False
    
    def prepare(self) -> bool:
        """Run the full data preparation pipeline.
        
        Returns:
            True if pipeline completed successfully, False otherwise
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
                
                if not success:
                    self.logger.error(f"❌ Step '{step_name}' failed")
                    self.stats['steps_failed'].append(step_name)
                    return False
                
                step_duration = time.time() - step_start_time
                self.logger.info(f"✅ Step '{step_name}' completed in {step_duration:.1f}s")
                self.stats['steps_completed'].append(step_name)
                
                # Create checkpoint
                if self.checkpoint_dir:
                    self._create_checkpoint(step_name)
                    
            except Exception as e:
                self.logger.error(f"❌ Step '{step_name}' failed with exception: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                self.stats['steps_failed'].append(step_name)
                return False
        
        # Print summary
        self._print_summary()
        
        return len(self.stats['steps_failed']) == 0
    
    def _is_step_complete(self, step_name: str, step_cfg: Dict[str, Any]) -> bool:
        """Check if a step has already been completed."""
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
        
        Returns:
            True if step succeeded, False otherwise
        """
        kwargs = self._resolve_templates(step_cfg.get('kwargs', {}))
        
        # Map steps to their script paths
        STEP_SCRIPTS = {
            'gencode': 'src/betadogma/data/prepare_gencode.py',
            'gtex_junctions': 'src/betadogma/data/prepare_gtex.py',
            'population_variants': 'src/betadogma/data/prepare_population_variants.py',
            'clinvar_variants': 'src/betadogma/data/prepare_clinvar_variants.py',
            'splice_variants': 'src/betadogma/data/prepare_splice_variants.py',
            'overlapping_windows': 'src/betadogma/data/prepare_overlapping.py',
            'aggregate': 'src/betadogma/data/prepare_aggregate.py',
        }
        
        if step_name not in STEP_SCRIPTS:
            self.logger.error(f"❌ Unknown step: {step_name}")
            return False
        
        script_path = ROOT / STEP_SCRIPTS[step_name]
        
        if not script_path.exists():
            self.logger.error(f"❌ Script not found: {script_path}")
            return False
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        
        # Convert kwargs to command-line arguments
        for key, value in kwargs.items():
            cli_key = key.replace('_', '-')
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{cli_key}')
            elif isinstance(value, (list, tuple)):
                if step_name == 'splice_variants' and key == 'effects':
                    # For splice_variants, pass each effect as a separate --effect argument
                    for item in value:
                        cmd.extend(['--effect', str(item)])
                else:
                    # For nargs='+' style args, pass one flag followed by multiple values
                    if len(value) > 0:
                        cmd.append(f'--{cli_key}')
                        cmd.extend([str(item) for item in value])
            elif value is not None and value != '':
                cmd.extend([f'--{cli_key}', str(value)])
                
        try:
            start_time = time.time()
            
            # Run subprocess - this is the critical fix!
            # We capture output but immediately fail on non-zero return codes
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False  # We'll check manually for better error handling
            )
            
            exec_time = time.time() - start_time
            
            # Log output (filter out progress bars for cleaner logs)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line and not any(x in line for x in ['%|', 'it/s', 'ETA:']):
                        self.logger.info(f"    {line}")
            
            # Check return code BEFORE logging summary
            if result.returncode != 0:
                self.logger.error(f"❌ Script failed with exit code {result.returncode}")
                self.logger.error(f"   Execution time: {exec_time:.1f}s")
                
                # Log stderr for debugging
                if result.stderr:
                    self.logger.error("   Error output:")
                    for line in result.stderr.strip().split('\n')[:20]:  # First 20 lines
                        if line:
                            self.logger.error(f"     {line}")
                
                return False
            
            # Success - log summary
            self.logger.info(f"✅ Script completed in {exec_time:.1f}s")
            self._log_step_summary(step_name, result.stdout, result.stderr)
            
            return True
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"❌ Subprocess error: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _create_checkpoint(self, step_name: str) -> None:
        """Create a checkpoint file for a completed step."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_file = self.checkpoint_dir / f"{step_name}.done"
        checkpoint_data = {
            'step': step_name,
            'completed_at': datetime.now().isoformat(),
            'config_path': str(self.config_path),
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"📝 Checkpoint created: {checkpoint_file}")
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 Pipeline Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"  Total time:      {duration:.1f}s ({duration/60:.1f} min)")
        self.logger.info(f"  Steps completed: {len(self.stats['steps_completed'])}")
        self.logger.info(f"  Steps skipped:   {len(self.stats['steps_skipped'])}")
        self.logger.info(f"  Steps failed:    {len(self.stats['steps_failed'])}")
        
        if self.stats['steps_completed']:
            self.logger.info(f"\n  ✅ Completed:")
            for step in self.stats['steps_completed']:
                self.logger.info(f"     - {step}")
        
        if self.stats['steps_skipped']:
            self.logger.info(f"\n  ⊘ Skipped:")
            for step in self.stats['steps_skipped']:
                self.logger.info(f"     - {step}")
        
        if self.stats['steps_failed']:
            self.logger.info(f"\n  ❌ Failed:")
            for step in self.stats['steps_failed']:
                self.logger.info(f"     - {step}")
        
        self.logger.info("=" * 80)
        
        if not self.stats['steps_failed']:
            self.logger.info("✅ Pipeline completed successfully!")
            self.logger.info(f"\n📌 Training data ready at: {self.paths.get('output', 'N/A')}")
        else:
            self.logger.error("❌ Pipeline failed. Check logs above for details.")
        
        self.logger.info("")
    
    def validate_outputs(self) -> bool:
        """Validate that all expected outputs were created."""
        self.logger.info("=" * 80)
        self.logger.info("🔍 Validating outputs...")
        self.logger.info("=" * 80)
        
        validation_cfg = self.config.get('validation', {})
        expected_outputs = validation_cfg.get('expected_outputs', {})
        
        all_valid = True
        
        for step_name, patterns in expected_outputs.items():
            self.logger.info(f"\n📦 {step_name}:")
            
            for pattern in patterns:
                pattern = self._resolve_templates(pattern)
                matches = glob(pattern)
                
                if matches:
                    self.logger.info(f"  ✓ {pattern} ({len(matches)} files)")
                else:
                    self.logger.error(f"  ✗ {pattern} (no matches)")
                    all_valid = False
        
        self.logger.info("\n" + "=" * 80)
        if all_valid:
            self.logger.info("✅ All validations passed")
        else:
            self.logger.error("❌ Some validations failed")
        self.logger.info("=" * 80)
        
        return all_valid


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training data for BetaDogma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  %(prog)s --config configs/data.yaml
  
  # Resume from specific step
  %(prog)s --config configs/data.yaml --from-step splice_variants
  
  # Force re-run from specific step
  %(prog)s --config configs/data.yaml --from-step gtex --force
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
        help="Path to log file (default: console only)"
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
    
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        preparer = TrainingDataPreparer(
            config_path=args.config,
            from_step=args.from_step,
            force=args.force,
            debug=args.debug,
            log_file=args.log_file
        )
        
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