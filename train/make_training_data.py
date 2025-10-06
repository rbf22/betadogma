
def step_prepare_variants(cfg: Dict[str, Any], cfg_dir: Path) -> None:
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
    required = ['vcf', 'windows_glob', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_variants: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_variants import prepare_variants

        print(f"[data] Running prepare_variants with kwargs: {kwargs}")

        # Map the kwargs to the function parameters
        prepare_variants(
            vcf_path=kwargs['vcf'],
            windows_glob=kwargs['windows_glob'],
            out_dir=kwargs['out'],
            apply_alt=kwargs.get('apply_alt', True),
            max_per_window=kwargs.get('max_per_window', 64),
            shard_size=kwargs.get('shard_size', 50000)
        )
    except Exception as e:
        print(f"[data] Error in prepare_variants: {erm test.txt}")
        raise


def step_prepare_data(cfg: Dict[str, Any], cfg_dir: Path) -> None:
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

        print(f"[data] Running prepare_data with kwargs: {kwargs}")

        # Map the kwargs to the function parameters
        prepare_data(
            input_dir=kwargs['input_dir'],
            output_dir=kwargs['output_dir'],
            gtex_dir=kwargs.get('gtex_dir'),
            variant_dir=kwargs.get('variant_dir'),
            write_index=kwargs.get('write_index', True),
            split_by=kwargs.get('split_by'),
            keep_columns=kwargs.get('keep_columns')
        )
    except Exception as e:
        print(f"[data] Error in prepare_data: {erm test.txt}")
        raise


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict data builder for Betadogma.

- Resolves ALL paths relative to the YAML file location
- Steps (all optional except aggregate):
    - betadogma.data.prepare_gencode   (optional)
    - betadogma.data.prepare_gtex      (optional)
    - betadogma.data.prepare_variants  (optional)
    - betadogma.data.prepare_data      (REQUIRED aggregation)
- Validation:
    - Prefer `outputs.expect`: list of files or glob patterns to assert exist.
    - Falls back to legacy JSONL check (train/val[/test].jsonl) if `outputs.expect` not given.
"""

import os
import sys
import argparse
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------- project paths --------
THIS = Path(__file__).resolve()
TRAIN_DIR = THIS.parent
PROJECT_ROOT = TRAIN_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # allow "betadogma.*" imports

DEFAULT_CFG = TRAIN_DIR / "configs" / "data.base.yaml"


# -------- utils --------
def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_dir"] = str(path.parent)
    return cfg


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


def expect_globs(patterns: List[str | Path]) -> Tuple[bool, List[str]]:
    """Return (ok, missing_patterns). ok=True if every pattern matched ≥1 file."""
    from glob import glob
    miss = []
    for pat in patterns:
        matches = glob(str(pat))
        if len(matches) == 0:
            miss.append(str(pat))
    return (len(miss) == 0, miss)


# -------- steps --------
def step_prepare_gencode(cfg: Dict[str, Any], cfg_dir: Path) -> None:
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
        if isinstance(value, str) and (value.endswith(('.fa', '.fasta', '.gtf', '.gz', '.parquet')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['fasta', 'gtf', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_gencode: {', '.join(missing)}")

    try:
        # Import the module and call the function directly
        from betadogma.data.prepare_gencode import prepare_gencode
        
        print(f"[data] Running prepare_gencode with kwargs: {kwargs}")
        
        # Map the kwargs to the function parameters
        prepare_gencode(
            fasta_path=kwargs['fasta'],
            gtf_path=kwargs['gtf'],
            out_dir=kwargs['out'],
            window=kwargs.get('window', 131072),
            stride=kwargs.get('stride', 65536),
            bin_size=kwargs.get('bin_size', 1),
            chroms=kwargs.get('chroms', ''),
            max_shard_bases=kwargs.get('max_shard_bases', 50_000_000)
        )
    except Exception as e:
        print(f"[data] Error in prepare_gencode: {e!r}")
        raise


def step_prepare_gtex(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    section = cfg.get("gtex", {})
    if not section or not section.get("enabled", False):
        print("[data] gtex: skipped")
        return

    # Get and process kwargs
    kwargs = dict(section.get("kwargs") or {})

    # Resolve all paths relative to the config file
    for key, value in kwargs.items():
        if isinstance(value, str) and (value.endswith(('.gtf', '.parquet', '.csv', '.tsv')) or '/' in value):
            rp = resolve_path(value, cfg_dir)
            if rp is not None:
                kwargs[key] = str(rp)

    # Ensure required arguments are present
    required = ['junctions', 'gtf', 'out']
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise ValueError(f"Missing required arguments for prepare_gtex: {', '.join(missing)}")

    try:
        # Import the required functions
        from betadogma.data.prepare_gtex import (
            read_junction_tables,
            compute_junction_psi,
            build_gene_index,
            annotate_genes,
            summarize_gene_psi
        )
        import os

        print(f"[data] Running prepare_gtex with kwargs: {kwargs}")

        # 1) Create output directory
        os.makedirs(kwargs['out'], exist_ok=True)

        # 2) Load junctions
        df = read_junction_tables(kwargs['junctions'], smoke=kwargs.get('smoke', False), chroms=kwargs.get('chroms'))
        print(f"[data] Loaded {len(df)} junctions from {len(df['sample_id'].unique())} samples")
        print(f"[data] Chromosomes in data: {sorted(df['chrom'].unique())}")
        print(f"[data] Junctions per chromosome: {df['chrom'].value_counts().to_dict()}")

        # 3) Optional chromosome filter
        if 'chroms' in kwargs and kwargs['chroms']:
            keep = set([c.strip() for c in kwargs['chroms'].split(",") if c.strip()])
            print(f"[data] Filtering GTEx data to chromosomes: {sorted(keep)}")
            df = df[df["chrom"].isin(keep)].copy()
            print(f"[data] After chromosome filtering: {len(df)} junctions remain")

        # 4) Compute PSI
        min_count = kwargs.get('min_count', 5)
        min_total = kwargs.get('min_total', 20)
        print(f"[data] Computing PSI with min_count={min_count}, min_total={min_total}")
        print(f"[data] Input columns before PSI: {list(df.columns)}")
        df_psi = compute_junction_psi(df, min_count=min_count, min_total=min_total)
        print(f"[data] Columns after PSI computation: {list(df_psi.columns)}")

        # 5) Gene assignment
        chroms = sorted(df_psi["chrom"].unique())
        gene_index = build_gene_index(kwargs['gtf'], allowed_chroms=chroms)
        print(f"[data] Annotating genes for {len(chroms)} chromosomes")
        df_psi = annotate_genes(df_psi, gene_index)
        print(f"[data] Columns after gene annotation: {list(df_psi.columns)}")

        # 6) Write junction-level PSI
        junc_out = os.path.join(kwargs['out'], "junction_psi.parquet")
        df_psi.to_parquet(junc_out, index=False)
        print(f"[data] Wrote {junc_out} ({len(df_psi):,} rows)")

        # 7) Per-gene summary
        min_samples = kwargs.get('min_samples', 5)
        print(f"[data] Creating gene summary with min_samples={min_samples}")
        print(f"[data] Columns before gene summary: {list(df_psi.columns)}")
        gene_sum = summarize_gene_psi(df_psi, min_samples=min_samples)
        gene_out = os.path.join(kwargs['out'], "gene_psi_summary.parquet")
        gene_sum.to_parquet(gene_out, index=False)
        print(f"[data] Wrote {gene_out} ({len(gene_sum):,} genes)")

    except Exception as e:
        print(f"[data] Error in prepare_gtex: {e!r}")
        raise


# -------- main driver --------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare training data for Betadogma')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--smoke', action='store_true', help='Run in smoke test mode (process only a small subset of data)')
    parser.add_argument('--from-step', type=str, choices=['gencode', 'gtex', 'variants', 'data'], 
                       help='Start from a specific step')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                       help='Directory to save checkpoints')
    return parser.parse_args()


def save_checkpoint(step_name: str, checkpoint_dir: Path):
    """Save a checkpoint file for the given step."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{step_name}.done"
    checkpoint_file.touch()


def should_skip_step(step_name: str, checkpoint_dir: Path, from_step: Optional[str]) -> bool:
    """Check if we should skip this step based on checkpoints and --from-step."""
    if from_step:
        # If we're starting from a specific step, skip all previous steps
        step_order = ['gencode', 'gtex', 'variants', 'data']
        if step_name in step_order and step_order.index(step_name) < step_order.index(from_step):
            return True
    
    # Check if we already completed this step
    checkpoint_file = checkpoint_dir / f"{step_name}.done"
    return checkpoint_file.exists()


def main():
    import os
    from pathlib import Path
    import argparse
    
    # Parse command line arguments
    args = parse_args()
    
    # Set up paths
    TRAIN_DIR = Path(__file__).parent
    cfg_env = os.environ.get('DATA_CONFIG')
    
    # Handle config path resolution
    if args.config:
        cfg_path = Path(args.config)
    elif cfg_env:
        cfg_path = Path(cfg_env)
    else:
        # Default to configs/ directory in the project root (one level up from train/)
        cfg_path = TRAIN_DIR.parent / 'configs' / 'data.base.yaml'
    
    # Convert to absolute path if not already
    if not cfg_path.is_absolute():
        # Use the current working directory as the base for relative paths
        cfg_path = Path.cwd() / cfg_path
    
    cfg_path = cfg_path.resolve()
    cfg = load_yaml(cfg_path)
    cfg_dir = Path(cfg["_config_dir"]).resolve()

    # Set up checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir or 'checkpoints').resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply smoke test settings if enabled
    if args.smoke:
        print("\n[data] Running in SMOKE TEST mode - processing minimal data\n")
        # Update config for smoke test
        if 'gtex' in cfg and 'kwargs' in cfg['gtex']:
            cfg['gtex']['kwargs']['smoke'] = True
        if 'variants' in cfg and 'kwargs' in cfg['variants']:
            cfg['variants']['kwargs']['smoke'] = True
    
    # 1) optional raw preprocessing
    if not should_skip_step('gencode', checkpoint_dir, args.from_step):
        print("\n[data] Step 1/4: Running prepare_gencode\n")
        step_prepare_gencode(cfg, cfg_dir)
        save_checkpoint('gencode', checkpoint_dir)
    else:
        print("\n[data] Step 1/4: Skipping prepare_gencode (already completed)")

    # 2) optional expression preprocessing
    if not should_skip_step('gtex', checkpoint_dir, args.from_step):
        print("\n[data] Step 2/4: Running prepare_gtex\n")
        step_prepare_gtex(cfg, cfg_dir)
        save_checkpoint('gtex', checkpoint_dir)
    else:
        print("\n[data] Step 2/4: Skipping prepare_gtex (already completed)")

    # 3) optional variants preprocessing
    if not should_skip_step('variants', checkpoint_dir, args.from_step):
        print("\n[data] Step 3/4: Running prepare_variants\n")
        step_prepare_variants(cfg, cfg_dir)
        save_checkpoint('variants', checkpoint_dir)
    else:
        print("\n[data] Step 3/4: Skipping prepare_variants (already completed)")

    # 4) required aggregation
    if not should_skip_step('data', checkpoint_dir, args.from_step):
        print("\n[data] Step 4/4: Running prepare_data\n")
        step_prepare_data(cfg, cfg_dir)
        save_checkpoint('data', checkpoint_dir)
    else:
        print("\n[data] Step 4/4: Skipping prepare_data (already completed)")

    # 5) verify outputs exist
    out = cfg.get("outputs", {})
    # Preferred: explicit expectations — files or glob patterns
    expect = out.get("expect")  # list of file names or glob patterns
    out_dir = resolve_path(out.get("dir", ""), cfg_dir)

    if expect:
        patterns: List[str] = []
        for item in expect:
            p = Path(item)
            if not p.is_absolute():
                p = (out_dir / p) if out_dir else (cfg_dir / p)
            patterns.append(str(p))
        ok, missing_patterns = expect_globs(patterns)
        if not ok:
            miss = "\n  - ".join(missing_patterns)
            raise FileNotFoundError(
                "Expected outputs not found for patterns:\n"
                f"  - {miss}\n"
                f"Check your config and the prepare_* implementations/paths."
            )
        print(f"[data] ✅ Outputs matched:")
        for pat in patterns:
            print(f"  - {pat}")
        return

    # Legacy fallback: JSONL train/val(/test)
    if out_dir is None:
        raise ValueError("outputs.dir must be set when outputs.expect is not provided")

    train_name = out.get("train_file", "train.jsonl")
    val_name   = out.get("val_file", "val.jsonl")
    test_name  = out.get("test_file")  # optional

    expected = [out_dir / train_name, out_dir / val_name]
    if test_name:
        expected.append(out_dir / test_name)

    ok, missing = expect_files(expected)
    if not ok:
        miss = "\n  - ".join(str(m) for m in missing)
        raise FileNotFoundError(
            "Expected output files were not produced by prepare_data:\n"
            f"  - {miss}\n"
            f"Set explicit patterns under outputs.expect to validate Parquet/other formats."
        )

    print(f"[data] ✅ Data ready in {out_dir}")
    for p in expected:
        print(f"  - {p}")


if __name__ == "__main__":
    main()