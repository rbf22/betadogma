#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict data builder for Betadogma.

- Resolves ALL paths relative to the YAML file location
- Steps (all optional except aggregate):
    - betadogma.data.prepare_gencode   (optional)
    - betadogma.data.prepare_gtex      (optional)
    - betadogma.data.prepare_variants  (optional - creates non-overlapping base windows)
    - betadogma.data.create_overlapping_windows (optional - creates overlapping windows from base)
    - betadogma.data.prepare_data      (REQUIRED aggregation)
"""

import logging
import os
import sys
import argparse
import subprocess
import importlib
import inspect
import yaml
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Project root (assumes this file is in train/)
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "train"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def setup_logging(log_file: str = "make_training_data.log", debug: bool = False):
    """
    Setup logging to both file and console.
    File is overwritten each time (mode='w').
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level
    level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (overwrite mode with 'w')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


DEFAULT_CFG = TRAIN_DIR / "configs" / "data.base.yaml"


def step_prepare_variants(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Step 1: Create NON-OVERLAPPING base windows with variants.
    These windows are abutting (stride = window) to avoid redundant work.
    """
    section = cfg.get("variants", {})
    if not section or not section.get("enabled", False):
        print("[data] variants: skipped")
        return

    # Get and process kwargs
    kwargs = dict(section.get("kwargs") or {})

    # Resolve all paths relative to the config file
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith(('.vcf', '.vcf.gz', '.parquet')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['vcf', 'windows', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_variants: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_variants import prepare_variants

        variant_limit = kwargs.get('max_per_window', 100)
        print(f"[data] Running prepare_variants on NON-OVERLAPPING windows with max_per_window={variant_limit}")
        
        prepare_variants(
            vcf=kwargs['vcf'],
            windows=kwargs['windows'],
            out=kwargs['out'],
            apply_alt=kwargs.get('apply_alt', True),
            max_per_window=variant_limit,
            shard_size=kwargs.get('shard_size', 50000),
            seed=kwargs.get('seed', 42),
            debug=kwargs.get('debug', False)
        )
        
        print(f"[data] Created base variant windows in: {kwargs['out']}")
        
    except Exception as e:
        print(f"[data] Error in prepare_variants: {e}")
        raise


def step_create_overlapping_windows(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Step 2: Create OVERLAPPING windows from the non-overlapping base windows.
    This reuses variants from base windows, only checking for conflicts at window boundaries.
    """
    section = cfg.get("overlapping_windows", {})
    if not section or not section.get("enabled", False):
        print("[data] overlapping_windows: skipped")
        return

    kwargs = dict(section.get("kwargs") or {})

    # Resolve paths
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith('.parquet') or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    required = ['base_windows', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for overlapping_windows: {', '.join(missing)}")

    try:
        from betadogma.data.create_overlapping_windows import create_overlapping_windows

        stride = kwargs.get('stride', 65536)
        print(f"[data] Creating overlapping windows with stride={stride}")
        
        create_overlapping_windows(
            base_windows_dir=kwargs['base_windows'],
            output_dir=kwargs['out'],
            stride=stride,
            seed=kwargs.get('seed', 42),
            debug=kwargs.get('debug', False)
        )
        
        print(f"[data] Created overlapping windows in: {kwargs['out']}")
        
    except Exception as e:
        print(f"[data] Error in create_overlapping_windows: {e}")
        raise


def step_prepare_data(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Step 3: Aggregate all data (gencode, gtex, variants) into final training format.
    """
    section = cfg.get("aggregate", {})
    if not section or not section.get("enabled", False):
        print("[data] aggregate: skipped")
        return

    # Get and process kwargs
    kwargs = dict(section.get("kwargs") or {})

    # Resolve all paths relative to the config file
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith(('.parquet', '.jsonl')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['input_dir', 'output_dir']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_data: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_data import prepare_data

        print(f"[data] Running prepare_data (final aggregation)")
        
        prepare_data(
            input_dir=kwargs['input_dir'],
            output_dir=kwargs['output_dir'],
            gtex_dir=kwargs.get('gtex_dir'),
            variant_dir=kwargs.get('variant_dir'),
            write_index=kwargs.get('write_index', True),
            split_by=kwargs.get('split_by'),
            keep_columns=kwargs.get('keep_columns'),
            max_variants_per_window=kwargs.get('max_variants_per_window', 0)
        )

    except Exception as e:
        print(f"[data] Error in prepare_data: {e}")
        raise


def resolve_path(p: Optional[str], base: Path) -> Optional[Path]:
    if p in (None, "", False):
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def run_module_cli(module: str, args: List[str]) -> None:
    """Run a module as a CLI: python -m module [--k v ...]"""
    cmd = [sys.executable, "-m", module] + args
    print(f"[data] CLI: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def call_module_entrypoint(module: str, kwargs: Dict[str, Any]) -> None:
    """
    Import module and call a plausible entrypoint with filtered kwargs.
    Tried in order: main, run, prepare, build.
    """
    mod = importlib.import_module(module)
    for name in ("main", "run", "prepare", "build"):
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[data] Calling {module}.{name}(**kwargs)")
            # Filter kwargs to avoid unexpected-arg errors
            try:
                sig = inspect.signature(fn)
                filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
            except (ValueError, TypeError):
                filt = kwargs
            return fn(**filt)  # type: ignore
    raise AttributeError(f"No callable entrypoint (main/run/prepare/build) in {module}.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def expect_files(paths: List[Path]) -> Tuple[bool, List[Path]]:
    missing = [p for p in paths if not p.exists()]
    return (len(missing) == 0, missing)


def expect_globs(patterns: List[Union[str, Path]]) -> Tuple[bool, List[str]]:
    """Return (ok, missing_patterns). ok=True if every pattern matched ≥1 file."""
    from glob import glob
    miss = []
    for pat in patterns:
        matches = glob(str(pat))
        if len(matches) == 0:
            miss.append(str(pat))
    return (len(miss) == 0, miss)


def step_prepare_gencode(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Step 0a: Prepare gencode windows (NON-OVERLAPPING for initial processing).
    """
    section = cfg.get("gencode", {})
    if not section or not section.get("enabled", False):
        print("[data] gencode: skipped")
        return

    # Get and process kwargs
    kwargs = dict(section.get("kwargs") or {})
    
    # Normalize common keys from earlier configs
    rename = {"gencode_gtf": "gtf", "out_dir": "out"}
    for old, new in rename.items():
        if old in kwargs and new not in kwargs:
            kwargs[new] = kwargs.pop(old)

    # Resolve all paths relative to the config file
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith(('.gtf', '.gz', '.fa', '.fasta')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['gtf', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_gencode: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_gencode import prepare_gencode

        # For base windows, use stride = window (non-overlapping)
        window = kwargs.get('window', 131072)
        stride = window  # Non-overlapping!
        
        print(f"[data] Running prepare_gencode with window={window}, stride={stride} (non-overlapping)")

        prepare_gencode(
            fasta_path=kwargs.get('fasta'),
            gtf_path=kwargs['gtf'],
            out_dir=kwargs['out'],
            window=window,
            stride=stride,
            bin_size=kwargs.get('bin_size', 1),
            chroms=kwargs.get('chroms', ''),
            max_shard_bases=kwargs.get('max_shard_bases', 50000000)
        )
    except Exception as e:
        print(f"[data] Error in prepare_gencode: {e}")
        raise


def step_prepare_gtex(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Step 0b: Prepare GTEx data.
    """
    section = cfg.get("gtex", {})
    if not section or not section.get("enabled", False):
        print("[data] gtex: skipped")
        return

    # Get and process kwargs
    kwargs = dict(section.get("kwargs") or {})
    
    # Resolve all paths relative to the config file
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith(('.gct', '.gz', '.parquet', '.tsv', '.csv')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['junctions', 'gtf', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_gtex: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_gtex import prepare_gtex

        print(f"[data] Running prepare_gtex")

        prepare_gtex(
            junctions=kwargs['junctions'],
            gtf=kwargs['gtf'],
            out=kwargs['out'],
            chroms=kwargs.get('chroms'),
            min_count=int(kwargs.get('min_count', 5)),
            min_samples=int(kwargs.get('min_samples', 3)),
            min_total=int(kwargs.get('min_total', 20)),
            smoke=kwargs.get('smoke', False)
        )
    except Exception as e:
        print(f"[data] Error in prepare_gtex: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Prepare training data for Betadogma")
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CFG,
        help=f"Path to YAML config (default: {DEFAULT_CFG.relative_to(ROOT)})",
    )
    ap.add_argument(
        "--from-step",
        choices=["gencode", "gtex", "variants", "overlapping", "aggregate"],
        help="Start from this step (inclusive)",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Directory to store completion checkpoints",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps, ignoring checkpoints",
    )
    ap.add_argument(
        "--log-file",
        type=str,
        default="make_training_data.log",
        help="Log file path (will be overwritten)"
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Force debugging",
    )
    return ap.parse_args()


def save_checkpoint(step_name: str, checkpoint_dir: Path) -> None:
    """Save a checkpoint file for the given step."""
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    (checkpoint_dir / f"{step_name}.done").touch()


def should_skip_step(step_name: str, checkpoint_dir: Path, from_step: Optional[str], force: bool = False) -> bool:
    """Check if we should skip this step based on checkpoints and --from-step.
    
    Args:
        step_name: Name of the current step
        checkpoint_dir: Directory containing checkpoint files
        from_step: If set, only run this step and after
        force: If True and step is at or after from_step, force it to run
    """
    # Define the order of steps
    step_order = ["gencode", "gtex", "variants", "overlapping", "aggregate"]
    
    # If --from-step is specified, check if we should run this step
    if from_step:
        try:
            from_idx = step_order.index(from_step)
            current_idx = step_order.index(step_name)
            
            # If this step is before the --from-step, skip it
            if current_idx < from_idx:
                print(f"[data] Step '{step_name}': skipped (before --from-step {from_step})")
                return True
                
            # If this is the --from-step or after, and --force is used, force run it
            if force:
                print(f"[data] Step '{step_name}': forced to run (--from-step {from_step} with --force)")
                return False
                
        except ValueError:
            # If step name not found in order, use default behavior
            pass
    
    # If --force is used without --from-step, force all steps
    if force:
        print(f"[data] Step '{step_name}': forced to run (--force)")
        return False
    
    # Check if this step has already been completed
    if (checkpoint_dir / f"{step_name}.done").exists():
        print(f"[data] Step '{step_name}': already completed, skipping")
        return True
        
    return False


def main() -> None:
    args = parse_args()
    
    # Setup logging FIRST, before any other code runs
    logger = setup_logging(log_file=args.log_file, debug=args.debug)
    
    # Load config
    cfg = load_yaml(args.config.resolve())
    cfg_dir = args.config.parent

    # Ensure checkpoint directory exists
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Log startup info
    logger.info("="*80)
    logger.info("Starting BetaDogma Training Data Preparation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    if args.from_step:
        logger.info(f"Starting from step: {args.from_step}")
    if args.force:
        logger.info("Force mode: ON")
    logger.info("="*80)

    # Run each step in order
    steps = [
        ("gencode", step_prepare_gencode),
        ("gtex", step_prepare_gtex),
        ("variants", step_prepare_variants),
        ("overlapping", step_create_overlapping_windows),
        ("aggregate", step_prepare_data),
    ]
    
    # Print header
    print("\n" + "="*80)
    print(f"[data] Starting data preparation with config: {args.config}")
    if args.force:
        print("[data] FORCE MODE: All steps will be re-run, ignoring checkpoints")
    if args.from_step:
        print(f"[data] Starting from step: {args.from_step}")
    print("="*80 + "\n")
    
    # Run each step
    for step_name, step_fn in steps:
        if should_skip_step(step_name, args.checkpoint_dir, args.from_step, args.force):
            continue
            
        print("\n" + "-"*60)
        print(f"[data] STARTING STEP: {step_name.upper()}")
        print("-"*60)
        
        try:
            start_time = time.time()
            step_fn(cfg, cfg_dir)
            save_checkpoint(step_name, args.checkpoint_dir)
            elapsed = time.time() - start_time
            print(f"[data] COMPLETED STEP: {step_name.upper()} in {elapsed:.1f} seconds")
            print("-"*60 + "\n")
        except Exception as e:
            logger.exception(f"ERROR in step {step_name}")
            print(f"[data] ERROR in step {step_name}: {e}")
            print("-"*60 + "\n")
            raise
    
    logger.info("="*80)
    logger.info("Finished BetaDogma Training Data Preparation")
    logger.info("="*80)


if __name__ == "__main__":
    main()