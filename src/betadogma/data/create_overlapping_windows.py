"""
Create overlapping windows from non-overlapping base windows with variants.

This avoids redundant variant conflict checking by reusing variants from
base windows and only checking conflicts at window boundaries.

Usage:
  python -m betadogma.data.create_overlapping_windows \
    --base-windows data/cache/variants_base \
    --out data/cache/variants_overlapping \
    --stride 65536 \
    --seed 42
"""

from __future__ import annotations
import argparse
import os
import time
import gc
from glob import glob
from typing import Dict, Any, List, Set, Tuple
import logging
import pandas as pd
import random

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-windows", required=True, help="Directory with base (non-overlapping) variant windows")
    ap.add_argument("--out", required=True, help="Output directory for overlapping windows")
    ap.add_argument("--stride", type=int, default=65536, help="Stride for overlapping windows")
    ap.add_argument("--shard-size", type=int, default=1000, help="Rows per output shard (for memory management)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    return ap.parse_args()


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return 0


def get_variant_span(variant_row) -> tuple[int, int]:
    """Get 0-based span of a variant."""
    pos = variant_row['in_window_idx']  # 1-based in window
    window_start = variant_row['start']
    
    # Convert to absolute genomic position (0-based)
    abs_pos = window_start + pos - 1
    
    var_type = variant_row['var_type']
    if var_type == 'INS':
        return (abs_pos, abs_pos + 1)
    else:
        ref_len = variant_row['span_in_window'][1] - variant_row['span_in_window'][0]
        return (abs_pos, abs_pos + ref_len)


def variants_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    """Check if two variant spans overlap."""
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])


def write_shard(rows: List[Dict], shard_idx: int, output_dir: str, logger) -> int:
    """Write a shard and return the number of rows written."""
    if not rows:
        return 0
    
    df = pd.DataFrame(rows)
    outp = os.path.join(output_dir, f"shard_{shard_idx:04d}.parquet")
    df.to_parquet(outp, index=False, compression='snappy')
    
    ins_count = (df['var_type'] == 'INS').sum()
    num_rows = len(df)
    
    logger.info(f"Wrote shard {shard_idx:04d} with {num_rows:,} rows ({ins_count} INS)")
    
    del df
    del rows[:]
    gc.collect()
    
    return num_rows


def stream_base_shards(base_windows_dir: str, logger):
    """Stream base window shard files one at a time."""
    base_files = sorted(glob(os.path.join(base_windows_dir, "shard_*.parquet")))
    
    if not base_files:
        raise ValueError(f"No base window files found in {base_windows_dir}")
    
    logger.info(f"Found {len(base_files)} base window shards")
    
    for f in base_files:
        df = pd.read_parquet(f)
        yield df
        del df
        gc.collect()


class BaseWindowBuffer:
    """
    Maintains a sliding buffer of base windows, grouped by chromosome.
    Only keeps base windows needed for current overlapping window generation.
    """
    
    def __init__(self, window_size: int, stride: int, logger):
        self.window_size = window_size
        self.stride = stride
        self.logger = logger
        
        # Per-chromosome state
        self.chrom_data = {}  # chrom -> {'variants': [], 'next_overlap_start': int}
        
    def add_variants(self, df: pd.DataFrame):
        """Add variants from a base window shard."""
        for chrom in df['chrom'].unique():
            chrom_df = df[df['chrom'] == chrom]
            
            if chrom not in self.chrom_data:
                # Initialize for this chromosome
                first_window = chrom_df.iloc[0]
                first_start = first_window['start']
                
                self.chrom_data[chrom] = {
                    'variants': [],
                    'next_overlap_start': first_start,
                }
            
            # Add variants to buffer
            for _, row in chrom_df.iterrows():
                self.chrom_data[chrom]['variants'].append(row)
    
    def generate_overlapping_windows(self, chrom: str, force_all: bool = False) -> List[Dict]:
        """
        Generate overlapping windows for a chromosome.
        
        Args:
            chrom: Chromosome to process
            force_all: If True, generate all remaining windows (end of chromosome)
        
        Returns:
            List of variant rows for overlapping windows
        """
        if chrom not in self.chrom_data:
            return []
        
        data = self.chrom_data[chrom]
        variants = data['variants']
        
        if not variants:
            return []
        
        output_rows = []
        
        # Sort variants by absolute position
        variants.sort(key=lambda v: (v['start'], v['in_window_idx']))
        
        # Determine range we can safely process
        max_variant_end = max(v['end'] for v in variants)
        
        next_start = data['next_overlap_start']
        
        # Process overlapping windows
        while True:
            overlap_start = next_start
            overlap_end = overlap_start + self.window_size
            
            # Exit condition: if we've moved past all variants
            if overlap_start >= max_variant_end:
                break
            
            # Check if we can complete this window
            if not force_all and overlap_end > max_variant_end:
                # Need more data, stop here
                break
            
            # Find variants in this overlapping window
            window_variants = []
            for v in variants:
                variant_pos = v['start'] + v['in_window_idx'] - 1
                if overlap_start <= variant_pos < overlap_end:
                    window_variants.append(v)
            
            if window_variants:
                # Remove conflicts
                selected_variants = []
                selected_spans = []
                
                for v in window_variants:
                    v_span = get_variant_span(v)
                    
                    has_conflict = False
                    for sel_span in selected_spans:
                        if variants_overlap(v_span, sel_span):
                            has_conflict = True
                            break
                    
                    if not has_conflict:
                        selected_variants.append(v)
                        selected_spans.append(v_span)
                
                # Create output rows
                for v in selected_variants:
                    old_abs_pos = v['start'] + v['in_window_idx'] - 1
                    new_in_window_idx = old_abs_pos - overlap_start + 1
                    
                    new_row = v.to_dict() if hasattr(v, 'to_dict') else dict(v)
                    new_row['start'] = overlap_start
                    new_row['end'] = overlap_end
                    new_row['in_window_idx'] = new_in_window_idx
                    
                    span_len = v['span_in_window'][1] - v['span_in_window'][0]
                    new_row['span_in_window'] = [new_in_window_idx, new_in_window_idx + span_len]
                    
                    output_rows.append(new_row)
            
            # Move to next window
            next_start += self.stride
            
            # Safety check to prevent infinite loops
            if not force_all and next_start + self.window_size > max_variant_end:
                break
        
        # Update next start position
        data['next_overlap_start'] = next_start
        
        # Remove variants we no longer need
        if not force_all:
            min_needed_end = next_start
            data['variants'] = [v for v in variants if v['end'] >= min_needed_end]
        else:
            # When finalizing, clear all variants
            data['variants'] = []
        
        return output_rows
    
    def get_active_chromosomes(self) -> List[str]:
        """Get list of chromosomes currently in buffer."""
        return list(self.chrom_data.keys())


def create_overlapping_windows(
    base_windows_dir: str,
    output_dir: str,
    stride: int = 65536,
    shard_size: int = 1000,
    seed: int = 42,
    debug: bool = False
):
    """
    Create overlapping windows from non-overlapping base windows.
    
    Strategy:
    1. Stream base window shards one at a time
    2. Maintain a sliding buffer of base windows per chromosome
    3. Generate overlapping windows as soon as we have enough data
    4. Write output immediately to keep memory usage low
    """
    logger = setup_logging(debug)
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    
    start_time = time.time()
    initial_mem = get_memory_usage()
    logger.info(f"Initial memory: {initial_mem:.1f} MB")
    
    # Get window size from first shard
    logger.info(f"Loading base windows from: {base_windows_dir}")
    base_files = sorted(glob(os.path.join(base_windows_dir, "shard_*.parquet")))
    
    if not base_files:
        raise ValueError(f"No base window files found in {base_windows_dir}")
    
    first_df = pd.read_parquet(base_files[0])
    window_size = first_df.iloc[0]['end'] - first_df.iloc[0]['start']
    del first_df
    gc.collect()
    
    logger.info(f"Window size: {window_size:,}, Stride: {stride:,}")
    logger.info(f"Found {len(base_files)} base window shards")
    
    # Initialize buffer
    buffer = BaseWindowBuffer(window_size, stride, logger)
    
    # Output management
    output_rows = []
    shard_idx = 0
    
    stats = {
        'base_variants': 0,
        'overlapping_windows': 0,
        'total_variants': 0,
        'conflicts_removed': 0,
        'snp': 0, 'ins': 0, 'del': 0
    }
    
    last_log_time = time.time()
    seen_chromosomes = set()
    completed_chromosomes = set()
    
    def maybe_write_shard():
        """Write shard if buffer is full."""
        nonlocal output_rows, shard_idx
        if len(output_rows) >= shard_size:
            write_shard(output_rows, shard_idx, output_dir, logger)
            output_rows.clear()
            shard_idx += 1
            gc.collect()
            return True
        return False
    
    # Process each base shard
    for shard_num, base_df in enumerate(stream_base_shards(base_windows_dir, logger)):
        stats['base_variants'] += len(base_df)
        
        # Track chromosomes we've seen
        current_chroms = set(base_df['chrom'].unique())
        new_chroms = current_chroms - seen_chromosomes
        finished_chroms = seen_chromosomes - current_chroms
        
        # Finalize chromosomes that are no longer appearing
        for chrom in finished_chroms:
            if chrom not in completed_chromosomes:
                logger.info(f"Finalizing chromosome: {chrom}")
                final_rows = buffer.generate_overlapping_windows(chrom, force_all=True)
                
                for row in final_rows:
                    output_rows.append(row)
                    stats['total_variants'] += 1
                    stats[row['var_type'].lower()] += 1
                    maybe_write_shard()
                
                # Clear chromosome data
                if chrom in buffer.chrom_data:
                    del buffer.chrom_data[chrom]
                
                completed_chromosomes.add(chrom)
                logger.info(f"  {chrom}: finalized with {len(final_rows):,} final variants")
                gc.collect()
        
        seen_chromosomes.update(current_chroms)
        
        # Add variants to buffer
        buffer.add_variants(base_df)
        del base_df
        gc.collect()
        
        # Generate overlapping windows for active chromosomes
        for chrom in buffer.get_active_chromosomes():
            if chrom in completed_chromosomes:
                continue
            
            overlap_rows = buffer.generate_overlapping_windows(chrom, force_all=False)
            
            for row in overlap_rows:
                output_rows.append(row)
                stats['total_variants'] += 1
                stats[row['var_type'].lower()] += 1
                maybe_write_shard()
        
        # Progress logging
        if shard_num % 5 == 0 or time.time() - last_log_time >= 2.0:
            mem_str = ""
            if HAS_PSUTIL:
                mem_mb = get_memory_usage()
                mem_str = f"Mem: {mem_mb:.0f}MB | "
            
            active_chroms = ", ".join(sorted(buffer.get_active_chromosomes()))
            
            logger.info(
                f"Processed {shard_num+1}/{len(base_files)} base shards | "
                f"{mem_str}"
                f"Variants: {stats['total_variants']:,} | "
                f"Buffered: {len(output_rows):,} | "
                f"Active: {active_chroms}"
            )
            last_log_time = time.time()
    
    # Finalize all remaining chromosomes
    logger.info("Finalizing remaining chromosomes...")
    for chrom in list(buffer.get_active_chromosomes()):
        if chrom not in completed_chromosomes:
            logger.info(f"Finalizing chromosome: {chrom}")
            
            # Get chromosome data
            if chrom not in buffer.chrom_data:
                continue
                
            data = buffer.chrom_data[chrom]
            variants = data['variants']
            
            if not variants:
                logger.info(f"  {chrom}: no variants to finalize")
                continue
            
            variants.sort(key=lambda v: (v['start'], v['in_window_idx']))
            max_variant_end = max(v['end'] for v in variants)
            next_start = data['next_overlap_start']
            
            window_count = 0
            variant_count = 0
            last_progress_time = time.time()
            
            # Generate windows incrementally to avoid memory buildup
            while next_start < max_variant_end:
                overlap_start = next_start
                overlap_end = overlap_start + window_size
                
                # Find variants in this overlapping window
                window_variants = [v for v in variants 
                                 if overlap_start <= v['start'] + v['in_window_idx'] - 1 < overlap_end]
                
                if window_variants:
                    # Remove conflicts
                    selected_variants = []
                    selected_spans = []
                    
                    for v in window_variants:
                        v_span = get_variant_span(v)
                        
                        has_conflict = False
                        for sel_span in selected_spans:
                            if variants_overlap(v_span, sel_span):
                                has_conflict = True
                                break
                        
                        if not has_conflict:
                            selected_variants.append(v)
                            selected_spans.append(v_span)
                    
                    # Create output rows and write immediately
                    for v in selected_variants:
                        old_abs_pos = v['start'] + v['in_window_idx'] - 1
                        new_in_window_idx = old_abs_pos - overlap_start + 1
                        
                        new_row = v.to_dict() if hasattr(v, 'to_dict') else dict(v)
                        new_row['start'] = overlap_start
                        new_row['end'] = overlap_end
                        new_row['in_window_idx'] = new_in_window_idx
                        
                        span_len = v['span_in_window'][1] - v['span_in_window'][0]
                        new_row['span_in_window'] = [new_in_window_idx, new_in_window_idx + span_len]
                        
                        output_rows.append(new_row)
                        stats['total_variants'] += 1
                        stats[new_row['var_type'].lower()] += 1
                        variant_count += 1
                        maybe_write_shard()
                
                next_start += stride
                window_count += 1
                
                # Progress logging during finalization
                if window_count % 1000 == 0 or time.time() - last_progress_time >= 5.0:
                    mem_str = ""
                    if HAS_PSUTIL:
                        mem_mb = get_memory_usage()
                        mem_str = f"Mem: {mem_mb:.0f}MB | "
                    logger.info(f"  {chrom}: {window_count:,} windows, {variant_count:,} variants | {mem_str}Buffered: {len(output_rows):,}")
                    last_progress_time = time.time()
            
            logger.info(f"  {chrom}: completed with {window_count:,} windows, {variant_count:,} variants")
            
            # Clear chromosome data
            del buffer.chrom_data[chrom]
            del variants
            gc.collect()
    
    # Write final shard
    if output_rows:
        write_shard(output_rows, shard_idx, output_dir, logger)
        output_rows.clear()
    
    # Final stats
    elapsed = time.time() - start_time
    final_mem = get_memory_usage()
    
    logger.info("\n" + "="*80)
    logger.info("Overlapping Window Creation Complete!")
    logger.info("="*80)
    logger.info(f"\nBase variants:            {stats['base_variants']:,}")
    logger.info(f"Total output variants:    {stats['total_variants']:,}")
    logger.info(f"Conflicts removed:        {stats['conflicts_removed']:,}")
    logger.info(f"\nVariant types:")
    logger.info(f"  SNP: {stats['snp']:,}")
    logger.info(f"  INS: {stats['ins']:,}")
    logger.info(f"  DEL: {stats['del']:,}")
    logger.info(f"\nPerformance:")
    logger.info(f"  Total time:               {elapsed/60:.1f} minutes")
    logger.info(f"  Output shards:            {shard_idx + 1}")
    logger.info(f"  Output directory:         {os.path.abspath(output_dir)}")
    
    if HAS_PSUTIL:
        logger.info(f"  Initial memory:           {initial_mem:.1f} MB")
        logger.info(f"  Final memory:             {final_mem:.1f} MB")
        logger.info(f"  Peak delta:               {final_mem - initial_mem:.1f} MB")
    
    logger.info("="*80)


def main():
    args = parse_args()
    create_overlapping_windows(
        base_windows_dir=args.base_windows,
        output_dir=args.out,
        stride=args.stride,
        shard_size=args.shard_size,
        seed=args.seed,
        debug=args.debug
    )


if __name__ == "__main__":
    main()