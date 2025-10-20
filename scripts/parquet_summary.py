#!/usr/bin/env python3
"""
Parquet summary script with proper array handling.

Efficiently handles numpy array columns with exact equality checking.
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np


def _is_array_column(series: pd.Series) -> bool:
    """
    Determine if a pandas Series contains numpy arrays.
    
    Checks the first non-null value to see if it's a numpy array.
    """
    for val in series:
        if val is None:
            continue
        # For numpy arrays, don't use pd.isna() as it returns an array
        if isinstance(val, np.ndarray):
            return True
        # For scalar values, check if NaN
        if not (isinstance(val, float) and pd.isna(val)):
            return False
    return False


def _count_unique_arrays(series: pd.Series) -> int:
    """
    Count unique arrays in a Series using binary hash comparison.
    
    Provides exact equality: arrays match only if all values at all positions are identical.
    """
    unique_hashes = set()
    
    for val in series:
        if val is None:
            continue
        if isinstance(val, np.ndarray) and val.size > 0:
            try:
                # Fast hash of binary representation
                h = hash(val.tobytes())
                unique_hashes.add(h)
            except Exception:
                # Fallback for non-contiguous arrays
                try:
                    h = hash(tuple(val.flat))
                    unique_hashes.add(h)
                except Exception:
                    pass
    
    return len(unique_hashes)


def _analyze_array_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze a column containing numpy arrays.
    
    Returns shape, dtype, unique count, sparsity, and value range.
    """
    # Find first valid array for metadata
    first_array = None
    for val in series:
        if isinstance(val, np.ndarray):
            first_array = val
            break
    
    if first_array is None:
        return {
            "type": "array",
            "shape": None,
            "dtype": None,
            "unique_count": 0,
            "sparsity_pct": None,
            "value_range": None,
        }
    
    # Get array metadata
    shape = first_array.shape
    dtype = str(first_array.dtype)
    
    # Count unique arrays (exact matching)
    unique_count = _count_unique_arrays(series)
    
    # Sample arrays for statistics (avoid processing all data)
    sample_size = min(10, len(series))
    sample_indices = np.linspace(0, len(series) - 1, sample_size, dtype=int)
    
    all_mins = []
    all_maxs = []
    zero_count = 0
    total_count = 0
    
    for idx in sample_indices:
        arr = series.iloc[idx]
        if isinstance(arr, np.ndarray) and arr.size > 0:
            all_mins.append(arr.min())
            all_maxs.append(arr.max())
            zero_count += np.sum(arr == 0)
            total_count += arr.size
    
    sparsity_pct = (zero_count / total_count * 100) if total_count > 0 else None
    value_range = (min(all_mins), max(all_maxs)) if all_mins else None
    
    return {
        "type": "array",
        "shape": shape,
        "dtype": dtype,
        "unique_count": unique_count,
        "sparsity_pct": sparsity_pct,
        "value_range": value_range,
    }


def summarize_dataframe(df: pd.DataFrame, include_nan_in_unique: bool = False) -> Dict[str, Any]:
    """
    Generate a summary for a pandas DataFrame with proper array handling.
    
    Returns a dictionary with:
      - shape: {'rows': int, 'columns': int}
      - columns: [column names]
      - dtypes: {column: dtype string}
      - array_info: {column: detailed array stats} (for array columns only)
      - missing_per_column: {column: int}
      - non_missing_per_column: {column: int}
      - unique_per_column: {column: int}
      - total_missing: int
      - total_present: int
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    rows, cols = df.shape
    cols_list = df.columns.tolist()

    dtypes = {}
    array_info = {}
    missing_per_column = {}
    non_missing_per_column = {}
    unique_per_column = {}
    
    for col in df.columns:
        series = df[col]
        
        # Detect array columns
        is_array = _is_array_column(series)
        
        if is_array:
            # Full array analysis
            info = _analyze_array_column(series)
            array_info[col] = info
            
            # Format dtype to show it's an array
            if info['shape']:
                dtypes[col] = f"ndarray{info['shape']}"
            else:
                dtypes[col] = "ndarray"
            
            # For arrays, only None counts as missing (not empty arrays)
            missing_count = sum(1 for val in series if val is None)
            missing_per_column[col] = missing_count
            non_missing_per_column[col] = len(series) - missing_count
            unique_per_column[col] = info['unique_count']
            
        else:
            # Standard column handling
            dtypes[col] = str(series.dtype)
            missing_per_column[col] = int(series.isnull().sum())
            non_missing_per_column[col] = int(series.notnull().sum())
            unique_per_column[col] = int(series.nunique(dropna=not include_nan_in_unique))
    
    total_missing = sum(missing_per_column.values())

    return {
        "shape": {"rows": int(rows), "columns": int(cols)},
        "columns": cols_list,
        "dtypes": dtypes,
        "array_info": array_info,
        "missing_per_column": missing_per_column,
        "non_missing_per_column": non_missing_per_column,
        "unique_per_column": unique_per_column,
        "total_missing": total_missing,
        "total_present": int(rows * cols - total_missing),
    }


def summarize_parquet_file(
    path: str,
    engine: Optional[str] = "auto",
    include_nan_in_unique: bool = False,
    sample_frac: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Read a Parquet file and return a dataset summary.
    """
    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be in (0, 1].")

    read_engine = None if engine == "auto" else engine
    df = pd.read_parquet(path, engine=read_engine)

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    return summarize_dataframe(df, include_nan_in_unique=include_nan_in_unique)


def pretty_print_summary(summary: dict, max_columns_to_show: int = 24) -> None:
    """
    Print a compact, pretty summary with improved formatting.
    """
    shape = summary.get("shape", {})
    rows = shape.get("rows", 0)
    cols = shape.get("columns", 0)
    column_names = summary.get("columns", [])

    dtypes = summary.get("dtypes", {})
    array_info = summary.get("array_info", {})
    missing = summary.get("missing_per_column", {})
    non_missing_per_column = summary.get("non_missing_per_column", {})
    unique = summary.get("unique_per_column", {})

    total_missing = summary.get("total_missing", 0)
    total_present = summary.get("total_present", 0)

    # Header
    print(f"\nDataset Shape: {rows:,} rows × {cols} columns")
    print(f"Columns: {', '.join(column_names[:6])}{'...' if len(column_names) > 6 else ''}\n")
    
    # Determine if we have any array columns with details
    has_array_details = any(col in array_info for col in column_names[:max_columns_to_show])
    
    # Calculate optimal column name width
    cols_to_show = column_names[:max_columns_to_show]
    max_col_name_len = max(len(col) for col in cols_to_show) if cols_to_show else 10
    col_width = min(max(max_col_name_len, 12), 30)  # Between 12 and 30 chars
    
    # Build header
    if has_array_details:
        header = f"{'Column':<{col_width}} {'Type':<18} {'Unique':>12} {'Missing':>10} {'%Miss':>6}  {'Array Details'}"
        separator = "=" * (col_width + 18 + 12 + 10 + 6 + 2 + 30)
    else:
        header = f"{'Column':<{col_width}} {'Type':<18} {'Unique':>12} {'Missing':>10} {'%Miss':>6}"
        separator = "=" * (col_width + 18 + 12 + 10 + 6 + 4)
    
    print(header)
    print(separator)

    shown = 0
    for col in column_names:
        if shown >= max_columns_to_show:
            break
        
        # Format column name (truncate intelligently if needed)
        if len(col) > col_width:
            # Keep start and end of long names
            keep_chars = (col_width - 3) // 2
            col_display = col[:keep_chars] + "..." + col[-(col_width - keep_chars - 3):]
        else:
            col_display = col
        
        # Format dtype
        dtype = dtypes.get(col, "")
        if dtype.startswith("object"):
            dtype_display = "str"
        elif dtype.startswith("int64"):
            dtype_display = "int64"
        elif dtype.startswith("float64"):
            dtype_display = "float64"
        elif dtype.startswith("int") and not dtype.startswith("int64"):
            dtype_display = "int"
        elif dtype.startswith("float") and not dtype.startswith("float64"):
            dtype_display = "float"
        else:
            dtype_display = dtype[:18]
        
        # Get statistics
        uniq = unique.get(col, 0)
        miss = missing.get(col, 0)
        non_miss = non_missing_per_column.get(col, 0)
        total = non_miss + miss
        pct_miss = (miss / total * 100) if total > 0 else 0.0
        
        # Build details string for array columns
        details = ""
        if col in array_info:
            info = array_info[col]
            details_parts = []
            
            if info.get('sparsity_pct') is not None:
                details_parts.append(f"{info['sparsity_pct']:.1f}% zeros")
            
            if info.get('value_range'):
                vmin, vmax = info['value_range']
                details_parts.append(f"range=[{vmin}, {vmax}]")
            
            details = ", ".join(details_parts)
        
        # Print row
        if has_array_details:
            print(f"{col_display:<{col_width}} {dtype_display:<18} {uniq:>12,} {miss:>10,} {pct_miss:>6.1f}%  {details}")
        else:
            print(f"{col_display:<{col_width}} {dtype_display:<18} {uniq:>12,} {miss:>10,} {pct_miss:>6.1f}%")
        
        shown += 1

    if len(column_names) > max_columns_to_show:
        print(f"\n... and {len(column_names) - max_columns_to_show} more columns")

    # Summary statistics
    pct_total_missing = (total_missing / (rows * cols) * 100) if rows * cols > 0 else 0.0
    print(f"\n{separator}")
    print(f"Total: {total_present:,} values present, {total_missing:,} missing ({pct_total_missing:.2f}%)\n")

def main():
    parser = argparse.ArgumentParser(
        prog="summarize_parquet_cli",
        description="Generate a summary for Parquet files with proper numpy array handling."
    )
    parser.add_argument("path", help="Path to the Parquet file to summarize")
    parser.add_argument("--engine", choices=["pyarrow", "fastparquet"], default="pyarrow",
                        help="Parquet engine to use (default: pyarrow)")
    parser.add_argument("--sample-frac", type=float, default=None, dest="sample_frac",
                        help="Fraction of rows to sample before summarizing (0 < frac <= 1)")
    parser.add_argument("--include-nan-in-unique", action="store_true",
                        help="Count NaN as a unique value when computing uniques")
    parser.add_argument("--max-columns", type=int, default=24, dest="max_columns",
                        help="Maximum number of columns to display (default: 24)")
    parser.add_argument("--output-json", dest="output_json",
                        help="Path to write the JSON summary (optional)")
    parser.add_argument("--no-pretty", dest="pretty", action="store_false",
                        help="Disable pretty printing; only write JSON if requested")
    parser.set_defaults(pretty=True)

    args = parser.parse_args()

    try:
        summary = summarize_parquet_file(
            path=args.path,
            engine=args.engine,
            include_nan_in_unique=args.include_nan_in_unique,
            sample_frac=args.sample_frac,
        )
    except Exception as ex:
        print(f"Error generating summary: {ex}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.pretty:
        pretty_print_summary(summary, max_columns_to_show=args.max_columns)

    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as ex:
            print(f"Warning: failed to write JSON output: {ex}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()