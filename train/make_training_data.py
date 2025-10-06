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

    kwargs = dict(section.get("kwargs") or {})
    # normalize common keys from earlier configs
    rename = {"gencode_gtf": "gtf", "out_dir": "out"}
    for old, new in rename.items():
        if old in kwargs and new not in kwargs:
            kwargs[new] = kwargs.pop(old)

    for key in ("gtf", "fasta", "out"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_gencode"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


def step_prepare_gtex(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    section = cfg.get("gtex", {})
    if not section or not section.get("enabled", False):
        print("[data] gtex: skipped")
        return

    kwargs = dict(section.get("kwargs") or {})
    for key in ("gtex_expression", "sample_table", "out_dir", "out"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_gtex"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


def step_prepare_variants(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    Optional variant channel preparation. Expects a module:
        betadogma.data.prepare_variants
    with an entrypoint taking kwargs like:
        vcf, fasta, out, window, stride, bin_size, chroms
    """
    section = cfg.get("variants", {})
    if not section or not section.get("enabled", False):
        print("[data] variants: skipped")
        return

    kwargs = dict(section.get("kwargs") or {})
    for key in ("vcf", "fasta", "out", "out_dir"):
        if key in kwargs:
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]
    # normalize
    if "out_dir" in kwargs and "out" not in kwargs:
        kwargs["out"] = kwargs.pop("out_dir")

    module = "betadogma.data.prepare_variants"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


def step_prepare_data(cfg: Dict[str, Any], cfg_dir: Path) -> None:
    """
    REQUIRED: runs betadogma.data.prepare_data which should write final splits.
    Accepts flexible inputs (e.g., structural shards, GTEx tables, variant shards).
    """
    section = cfg.get("aggregate", {})
    if not section or not section.get("enabled", True):
        raise RuntimeError("aggregate step is required; set aggregate.enabled: true")

    kwargs = dict(section.get("kwargs") or {})
    # common keys to resolve: any *dir or *path option
    for key in list(kwargs.keys()):
        if key.endswith(("dir", "path", "glob", "file")):
            rp = resolve_path(kwargs[key], cfg_dir)
            kwargs[key] = str(rp) if rp is not None else kwargs[key]

    module = "betadogma.data.prepare_data"
    try:
        call_module_entrypoint(module, kwargs)
    except Exception as e:
        print(f"[data] import-call failed for {module}: {e!r} -> trying CLI")
        cli_args = []
        for k, v in (section.get("cli_args") or {}).items():
            rv = resolve_path(v, cfg_dir)
            cli_args += [f"--{k.replace('_','-')}", str(rv if rv is not None else v)]
        run_module_cli(module, cli_args)


# -------- main driver --------
def main():
    cfg_env = os.environ.get("DATA_CONFIG", "")
    cfg_path = Path(cfg_env) if cfg_env else DEFAULT_CFG
    if not cfg_path.is_absolute():
        cfg_path = (TRAIN_DIR / cfg_path).resolve()
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