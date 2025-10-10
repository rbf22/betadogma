"""
I/O helpers for FASTA/VCF/Parquet/HDF5 (minimal, dependency-light).
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Generator, Optional, List, Union, cast, TypeVar, Type, TYPE_CHECKING
from pathlib import Path
import os

# Type variable for DataFrame to handle different DataFrame types
if TYPE_CHECKING:
    import pandas as pd
    from pyfaidx import Fasta

# FASTA (pyfaidx is a light dependency used elsewhere in the repo)
try:
    from pyfaidx import Fasta as _Fasta
except ImportError:  # pragma: no cover
    _Fasta = None


def read_fasta_window(
    fasta_path: Union[str, Path], 
    chrom: str, 
    start: int, 
    end: int, 
    uppercase: bool = True
) -> str:
    """
    Return FASTA slice [start, end) for a chromosome, padding with 'N' if out-of-range.
    
    Args:
        fasta_path: Path to the FASTA file
        chrom: Chromosome name
        start: Start position (0-based, inclusive)
        end: End position (0-based, exclusive)
        uppercase: If True, return sequence in uppercase
        
    Returns:
        DNA sequence with N-padding if the range extends beyond chromosome boundaries
        
    Raises:
        RuntimeError: If pyfaidx is not installed
        KeyError: If chromosome is not found in the FASTA file
    """
    if _Fasta is None:
        raise RuntimeError("pyfaidx is not installed; `pip install pyfaidx`.")
    
    fasta_path_str = str(fasta_path)
    fa = _Fasta(fasta_path_str, as_raw=True, sequence_always_upper=uppercase)
    
    if chrom not in fa:
        raise KeyError(f"Chromosome {chrom} not found in FASTA {fasta_path_str}")
        
    contig_len = len(fa[chrom])
    s = max(0, start)
    e = min(end, contig_len)
    
    # Get the sequence slice with proper bounds checking
    if s >= e:  # No valid range
        return "N" * (end - start)
        
    core = str(fa[chrom][s:e])
    left_pad = "N" * max(0, start - s)
    right_pad = "N" * max(0, end - e)
    
    return left_pad + core + right_pad


def read_parquet_shards(
    paths_glob_or_list: Union[str, Iterable[Union[str, Path]]]
) -> 'pd.DataFrame':
    """
    Read one or multiple Parquet files and return a concatenated DataFrame.
    
    Args:
        paths_glob_or_list: Either a glob pattern string or an iterable of file paths
        
    Returns:
        A single DataFrame containing all the data from the input files
        
    Raises:
        FileNotFoundError: If no files match the glob pattern or the input list is empty
        ImportError: If pandas is not installed
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for read_parquet_shards") from e
    
    if isinstance(paths_glob_or_list, str):
        from glob import glob
        paths = sorted(glob(paths_glob_or_list))
    else:
        paths = [str(p) for p in paths_glob_or_list]
        
    if not paths:
        raise FileNotFoundError("No Parquet files matched or provided list is empty.")
        
    # Check if files exist before attempting to read
    missing_files = [p for p in paths if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"Some files do not exist: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")
    
    try:
        dfs = [pd.read_parquet(p) for p in paths]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet files: {e}") from e


def write_parquet(
    df: 'pd.DataFrame', 
    out_path: Union[str, Path], 
    exist_ok: bool = True
) -> None:
    """
    Write a DataFrame to Parquet, creating parent dirs if needed.
    
    Args:
        df: The pandas DataFrame to write
        out_path: Output file path
        exist_ok: If False, raise an error if the output file already exists
        
    Raises:
        FileExistsError: If the output file exists and exist_ok is False
        ImportError: If pandas is not installed
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for write_parquet") from e
        
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists() and not exist_ok:
        raise FileExistsError(f"Output file exists and exist_ok=False: {out_path}")
        
    try:
        df.to_parquet(str(out_path), index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write parquet file to {out_path}: {e}") from e


def iter_vcf_records(vcf_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Minimal VCF reader (no external deps). Yields dicts with keys:
{{ ... }}

    Note: This does not parse INFO/FORMAT; it’s enough for simple variant sweeps.
    """
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, flt, info = parts[:8]
            yield {
                "chrom": chrom,
                "pos": int(pos),
                "id": vid if vid != "." else None,
                "ref": ref,
                "alt": alt.split(","),
                "qual": None if qual == "." else float(qual),
                "filter": flt,
                "info": info,
            }