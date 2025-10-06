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
        df = read_junction_tables(kwargs['junctions'])

        # 3) Optional chromosome filter
        if 'chroms' in kwargs and kwargs['chroms']:
            keep = set([c.strip() for c in kwargs['chroms'].split(",") if c.strip()])
            df = df[df["chrom"].isin(keep)].copy()

        # 4) Compute PSI
        min_count = kwargs.get('min_count', 5)
        min_total = kwargs.get('min_total', 20)
        df_psi = compute_junction_psi(df, min_count=min_count, min_total=min_total)

        # 5) Gene assignment
        chroms = sorted(df_psi["chrom"].unique())
        gene_index = build_gene_index(kwargs['gtf'], allowed_chroms=chroms)
        df_psi = annotate_genes(df_psi, gene_index)

        # 6) Write junction-level PSI
        junc_out = os.path.join(kwargs['out'], "junction_psi.parquet")
        df_psi.to_parquet(junc_out, index=False)
        print(f"[data] Wrote {junc_out} ({len(df_psi):,} rows)")

        # 7) Per-gene summary
        min_samples = kwargs.get('min_samples', 5)
        gene_sum = summarize_gene_psi(df_psi, min_samples=min_samples)
        gene_out = os.path.join(kwargs['out'], "gene_psi_summary.parquet")
        gene_sum.to_parquet(gene_out, index=False)
        print(f"[data] Wrote {gene_out} ({len(gene_sum):,} genes)")

    except Exception as e:
        print(f"[data] Error in prepare_gtex: {e!r}")
        raise


# -------- main driver --------
def main():
    import argparse
    import os
    from pathlib import Path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare training data for Betadogma')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
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

    # 1) optional raw preprocessing
    step_prepare_gencode(cfg, cfg_dir)

    # 2) optional expression preprocessing
    step_prepare_gtex(cfg, cfg_dir)

    # 3) optional variants preprocessing
    step_prepare_variants(cfg, cfg_dir)

    # 4) required aggregation
    step_prepare_data(cfg, cfg_dir)

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