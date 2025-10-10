"""Type stubs for prepare_data module."""

from typing import Optional, List, Any, Dict, Union, TextIO, BinaryIO, IO, TypeVar, Type, Tuple, Set, Callable, Iterable, Iterator, Generator, overload
from pathlib import Path
import pandas as pd
import numpy as np

# Type aliases
PathLike = Union[str, bytes, os.PathLike]
DataFrame = Any  # Placeholder for pandas.DataFrame
Series = Any     # Placeholder for pandas.Series

# Module-level functions
def parse_args() -> argparse.Namespace: ...

def setup_logging(debug: Optional[bool] = None) -> logging.Logger: ...

def load_parquet_files(glob_pattern: str) -> DataFrame: ...

def prepare_variants_wrapper(
    vcf_path: str, 
    windows_path: str, 
    out_path: str, 
    apply_alt: bool = ...,
    safety_limit: int = ...,
    shard_size: int = ...,
    seed: int = ...,
    debug: bool = ...
) -> None: ...

def prepare_data(
    input_dir: str,
    output_dir: str,
    gtex_dir: Optional[str] = ...,
    variant_dir: Optional[str] = ...,
    split_by: Optional[str] = ...,
    keep_columns: Optional[List[str]] = ...,
    safety_limit: int = ...,
    vcf_path: Optional[str] = ...,
    **kwargs: Any
) -> None: ...

def main() -> None: ...

# Module-level variables
logger: logging.Logger

# Import types that are used in the module
import argparse
import logging
import os

# Re-export public types and functions
__all__ = [
    'parse_args',
    'setup_logging',
    'load_parquet_files',
    'prepare_variants_wrapper',
    'prepare_data',
    'main',
    'logger',
]
