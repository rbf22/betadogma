"""
Precompute variant-aligned channels for model windows.

Usage:
  python -m betadogma.data.prepare_variants \
    --vcf data/variants/cohort.vcf.gz \
    --windows "data/cache/gencode_v44_structural_base/*.parquet" \
    --out data/cache/variants_aligned \
    [--apply-alt] [--max-per-window 64] [--seed 42]

Outputs: Parquet shards in <out>/shard_*.parquet with rows:
  chrom, start, end, bin_size, seq (REF window), seq_alt (if --apply-alt),
  variant_spec, var_type, in_window_idx, span_in_window,
  ch_snp, ch_ins, ch_del, ch_any  (list[int], length = end-start)
"""

from __future__ import annotations
import argparse
import os
import random
import time
from glob import glob
from typing import Dict, Any, Iterable, List, Optional, TextIO, Union, Set
import logging

import pandas as pd

# local helpers
from .encode import encode_variant, apply_variants_to_sequence, build_variant_channels, rescue_insertion_variant



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="VCF path or glob; .vcf(.gz) or .txt with VCF columns.")
    ap.add_argument("--windows", required=True, help="Glob for window Parquet shards from prepare_gencode.py")
    ap.add_argument("--out", required=True, help="Output directory for variant-aligned Parquet shards.")
    ap.add_argument("--apply-alt", action="store_true", help="Also store seq_alt (sequence with variant applied).")
    ap.add_argument("--max-per-window", type=int, default=0, help="Optional cap of variants per window (0 = unlimited).")
    ap.add_argument("--shard-size", type=int, default=50_000, help="Rows per output shard.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for variant selection reproducibility.")
    ap.add_argument("--debug", action="store_true", help="Enable debug output.")
    return ap.parse_args()


def setup_logging(debug=False):
    """Configure logging based on debug flag"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def _iter_vcf(path: str, logger=None):
    """
    Minimal VCF parser that yields dicts:
      chrom, pos (1-based), id, ref, alt (list[str]), qual, filter, info, samples...
      
    Args:
        path: Path to the VCF file (can be gzipped)
        logger: Optional logger for debug output
    """
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    
    # Debug counters
    debug_count = 0
    debug_max = 5  # Number of variants to show in debug output
    
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        # Read and parse header
        header = []
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line[1:].rstrip("\n").split("\t")
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"VCF Header: {header}")
                    logger.debug("First few variants:")
                break
        
        # Process variant lines
        for line in f:
            if not line.strip():
                continue
                
            t = line.rstrip("\n").split("\t")
            if len(t) < 8: 
                continue
                
            # Parse basic fields
            chrom, pos, vid, ref, alt, qual, flt, info = t[:8]
            samples = t[8:]  # Get sample columns if they exist
            
            # Create variant dict
            variant = {
                "chrom": chrom,
                "pos": int(pos),
                "id": None if vid == "." else vid,
                "ref": ref,
                "alt": alt.split(","),
                "qual": None if qual == "." else float(qual),
                "filter": flt,
                "info": info,
                "samples": samples
            }
            
            # Debug output for first few variants
            if logger and logger.isEnabledFor(logging.DEBUG) and debug_count < debug_max:
                logger.debug(f"Variant {debug_count + 1}: {chrom}:{pos} {ref}>{alt}")
                debug_count += 1
            
            # Yield each alternate allele as a separate variant
            for alt_allele in variant['alt']:
                v = variant.copy()
                v['alt'] = [alt_allele]
                yield v


def _read_vcf_glob(vcf_glob: str, logger=None) -> List[Dict[str, Any]]:
    """Read VCF files and return a list of variant dictionaries.
    
    Args:
        vcf_glob: Path or glob pattern for VCF files
        logger: Optional logger for debug output
        
    Returns:
        List of variant dictionaries with keys: chrom, pos, ref, alt, spec
    """
    paths = sorted(glob(vcf_glob)) if not os.path.isfile(vcf_glob) else [vcf_glob]
    assert paths, f"No VCF files matched: {vcf_glob}"
    
    out: List[Dict[str, Any]] = []
    pos_counts = {}
    total_variants = 0
    
    logger.info(f"Processing VCF files: {paths}")
    
    for p in paths:
        logger.info(f"Reading variants from: {p}")
        file_variants = 0
        
        for r in _iter_vcf(p, logger):
            # Track unique positions (convert alt list to tuple for hashing)
            alt_tuple = tuple(r['alt'])
            pos_key = (r['chrom'], r['pos'], r['ref'], alt_tuple)
            pos_counts[pos_key] = pos_counts.get(pos_key, 0) + 1
            file_variants += 1
            
            # Create variant spec and add to output (join alts with comma for the spec)
            alt_str = ",".join(r['alt'])
            spec = f"{r['chrom']}:{r['pos']}{r['ref']}>{alt_str}"
            out.append({
                "chrom": r['chrom'], 
                "pos": r['pos'], 
                "ref": r['ref'], 
                "alt": r['alt'], 
                "spec": spec
            })
        
        logger.info(f"Read {file_variants:,} variants from {os.path.basename(p)}")
        total_variants += file_variants
    
    # Print statistics
    unique_variants = len(pos_counts)
    duplicate_count = total_variants - unique_variants
    
    logger.info(f"Total variants: {total_variants:,}")
    logger.info(f"Unique variants: {unique_variants:,}")
    logger.info(f"Duplicate variants: {duplicate_count:,} ({duplicate_count/max(1, total_variants)*100:.1f}%)")
    
    if duplicate_count > 0 and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Most common variant positions:")
        for (chrom, pos, ref, alt), count in sorted(pos_counts.items(), key=lambda x: -x[1])[:5]:
            logger.debug(f"  {chrom}:{pos} {ref}>{alt}: {count} occurrences")
    
    return out


def stream_windows(glob_pat: str):
    """Stream windows from Parquet files to reduce memory usage."""
    files = sorted(glob(glob_pat))
    assert files, f"No Parquet windows matched: {glob_pat}"
    
    for file in files:
        df = pd.read_parquet(file)
        need = {"chrom", "start", "end", "seq", "bin_size"}
        missing = need - set(df.columns)
        assert not missing, f"Missing window columns: {missing}"
        yield from df.itertuples(index=False)


def select_non_overlapping_variants(variants, max_count=0, random_seed=42):
    """
    Efficiently select non-overlapping variants using a reservoir sampling approach.
    
    Args:
        variants: List of variant dictionaries
        max_count: Maximum number of variants to select (0 for all)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of selected non-overlapping variants
    """
    if not variants:
        return []
    
    if max_count == 0 or max_count >= len(variants):
        # Check for overlaps and return all non-overlapping variants
        non_overlapping = []
        selected_spans = []
        
        # Process variants in order of position
        for v in sorted(variants, key=lambda x: x["pos"]):
            start = v["pos"] - 1  # Convert to 0-based
            end = start + len(v["ref"])
            
            # Check for overlap with any selected variant
            overlap = any(
                not (end <= sel_start or start >= sel_end)
                for sel_start, sel_end in selected_spans
            )
            
            if not overlap:
                non_overlapping.append(v)
                selected_spans.append((start, end))
                
        return non_overlapping
    
    # For limited selection, use a more efficient approach with random sampling
    local_random = random.Random(random_seed)
    
    # Index variants by their position for fast lookup
    variant_indices = list(range(len(variants)))
    local_random.shuffle(variant_indices)
    
    selected = []
    selected_spans = []
    
    # Try variants in random order, but maintain position order in final selection
    for idx in variant_indices:
        if len(selected) >= max_count:
            break
            
        v = variants[idx]
        start = v["pos"] - 1  # Convert to 0-based
        end = start + len(v["ref"])
        
        # Check for overlap with any selected variant
        overlap = any(
            not (end <= sel_start or start >= sel_end)
            for sel_start, sel_end in selected_spans
        )
        
        if not overlap:
            selected.append(v)
            selected_spans.append((start, end))
    
    # Sort selected variants by position before returning
    return sorted(selected, key=lambda x: x["pos"])


def run(vcf: str, windows: str, out: str, apply_alt: bool = False, 
        max_per_window: int = 0, shard_size: int = 50_000, seed: int = 42, 
        debug: bool = False, max_variants_per_window: int = 500,
        batch_size: int = 10):  # Process 10 windows before writing
    """Main function to prepare variant data.
    
    Args:
        vcf: Path to VCF file or glob pattern
        windows: Glob pattern for window files
        out: Output directory
        apply_alt: Whether to apply ALT alleles
        max_per_window: Maximum variants per window (0 for no limit)
        shard_size: Number of rows per output shard
        seed: Random seed
        debug: Enable debug logging
        max_variants_per_window: Maximum variants to process per window (for memory safety)
    """
    # Make psutil import optional
    try:
        import psutil
        import gc
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
        
    import time
    import pandas as pd
    from typing import Optional
    
    start_time = time.time()
    logger = setup_logging(debug)
    os.makedirs(out, exist_ok=True)
    random.seed(seed)
    
    def log_memory() -> Optional[float]:
        """Log memory usage if psutil is available."""
        if not HAS_PSUTIL:
            return None
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Current memory usage: {mem_mb:.1f}MB")
        return mem_mb
        
    def cleanup():
        """Force garbage collection and clean up large objects."""
        if 'gc' in globals():
            gc.collect()
        if 'pd' in globals():
            pd.DataFrame().empty  # Trigger pandas internal cleanup
            
    # Initial memory usage
    start_mem = log_memory()
    
    logger.info("="*80)
    logger.info("Starting variant preparation")
    logger.info(f"  VCF: {vcf}")
    logger.info(f"  Windows: {windows}")
    logger.info(f"  Output: {out}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Max variants/window: {max_per_window if max_per_window > 0 else 'unlimited'}")
    logger.info(f"  Apply ALT: {apply_alt}")
    logger.info("-"*80)

    # Read and process VCF files
    logger.info("Reading VCF files...")
    variants = _read_vcf_glob(vcf, logger)

    # Count variant types from the VCF
    logger.info("Counting variant types...")
    variant_type_counts = {"SNP": 0, "INS": 0, "DEL": 0, "OTHER": 0}

    for v in variants:
        ref = v["ref"]
        alt = v["alt"][0] if isinstance(v["alt"], list) else v["alt"]
        
        if len(ref) < len(alt):
            var_type = "INS"
        elif len(ref) > len(alt):
            var_type = "DEL"
        elif len(ref) == len(alt):
            var_type = "SNP"
        else:
            var_type = "OTHER"
            
        variant_type_counts[var_type] += 1
    
    # Group variants by chrom for faster filter
    variants_by_chr: Dict[str, List[Dict[str, Any]]] = {}
    for v in variants:
        variants_by_chr.setdefault(v["chrom"], []).append(v)
    
    # Log variant statistics
    logger.info("\nVariant Statistics:" + "-"*60)
    total_variants = sum(len(v) for v in variants_by_chr.values())
    logger.info(f"  Total variants: {total_variants:,}")
    logger.info("  Variants by chromosome:")
    for chrom in sorted(variants_by_chr.keys(), key=lambda x: (len(str(x).lstrip('chr')), x)):
        count = len(variants_by_chr[chrom])
        logger.info(f"    {chrom}: {count:>9,} variants ({count/max(1, total_variants)*100:.1f}%)")
    
    # Print variant type statistics
    logger.info("\n  Variant types:")
    for var_type, count in sorted(variant_type_counts.items()):
        if var_type != "OTHER" or variant_type_counts["OTHER"] > 0:  # Skip OTHER if zero
            logger.info(f"    {var_type}: {count:>9,} variants ({count/total_variants*100:.1f}%)")

    # Print summary of insertions vs deletions
    ins_count = variant_type_counts["INS"]
    del_count = variant_type_counts["DEL"]
    indel_ratio = ins_count / del_count if del_count > 0 else 0
    logger.info(f"\n  Insertion/Deletion ratio: {indel_ratio:.2f} ({ins_count:,}/{del_count:,})")

    rows: List[Dict[str, Any]] = []
    shard_idx = 0
    window_count = 0

    # Stream windows to reduce memory usage
    total_windows = sum(1 for _ in stream_windows(windows))
    logger.info(f"\nProcessing {total_windows:,} windows in batches of {batch_size}...")
    logger.info("-"*80)
    
    # Reset stream
    window_stream = stream_windows(windows)
    last_log_time = time.time()
    processed_variants = 0
    batch_count = 0
    
    # Track memory usage
    peak_mem = start_mem if start_mem else 0
    last_cleanup = time.time()
    cleanup_interval = 30  # seconds between cleanups
    
    for w in window_stream:
        window_count += 1
        chrom = w.chrom
        w0 = int(w.start)
        w1 = int(w.end)
        seq = w.seq
        L = len(seq)
        
        # Add memory optimization here - right after getting window properties
        if window_count % 20 == 0:  # Every 20 windows
            # Force garbage collection more frequently
            gc.collect()
            
            # Check memory usage
            if HAS_PSUTIL:
                current_mem = psutil.Process().memory_info().rss / 1024 / 1024
                if current_mem > 8000:  # If memory exceeds 8GB
                    logger.warning(f"Memory usage high ({current_mem:.1f}MB), forcing write")
                    # Force write current batch regardless of count
                    if rows:
                        df = pd.DataFrame(rows)
                        outp = os.path.join(out, f"shard_{shard_idx:04d}.parquet")
                        df.to_parquet(outp, index=False)
                        logger.info(f"Memory-triggered write: shard {shard_idx} with {len(df):,} rows")
                        del df
                        rows.clear()
                        shard_idx += 1
                        batch_count = 0
                        gc.collect()
        
        # Log progress every second or every 1000 windows, whichever comes first
        current_time = time.time()
        if window_count % 1000 == 0 or current_time - last_log_time >= 1.0:
            # Get memory usage if psutil is available
            mem_info = f"Mem: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB" if HAS_PSUTIL else ""
            
            elapsed = current_time - start_time
            windows_per_sec = window_count / elapsed if elapsed > 0 else 0
            remaining = (total_windows - window_count) / windows_per_sec if windows_per_sec > 0 else 0
            
            progress_parts = [
                f"Processed {window_count:,}/{total_windows:,} windows",
                f"({window_count/max(1, total_windows)*100:.1f}%)",
                f"{windows_per_sec:.1f} windows/sec",
                f"ETA: {remaining/60:.1f} min",
                mem_info,
                f"Variants: {processed_variants:,} ({processed_variants/max(1, window_count):.1f}/window)"
            ]
            
            logger.info(" | ".join(filter(None, progress_parts)))
            last_log_time = current_time

        # Get all variants in this window
        hits = [v for v in variants_by_chr.get(chrom, []) if w0 <= (v["pos"] - 1) < w1]
        total_hits = len(hits)
        processed_variants += total_hits
        
        # Apply max_variants_per_window limit
        if max_variants_per_window > 0 and total_hits > max_variants_per_window:
            if window_count <= 3 or logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Window {chrom}:{w0}-{w1} has {total_hits:,} variants, "
                    f"randomly selecting {min(max_variants_per_window, total_hits):,}"
                )
            hits = random.sample(hits, max_variants_per_window)
        
        if hits:
            # Log detailed variant info for the first few windows or when in debug mode
            if window_count <= 3 or logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"\nWindow {chrom}:{w0}-{w1} (length: {L:,} bp)")
                logger.debug(f"  Found {len(hits):,} variants in this window")
                if len(hits) > 0:
                    logger.debug("  First few variants:")
                    for v in hits[:3]:
                        logger.debug(f"    {v['chrom']}:{v['pos']} {v['ref']}>{v['alt'][0]} (len: {len(v['ref'])}->{len(v['alt'][0])})")
            
            # Select non-overlapping variants with memory limit
            try:
                non_overlapping_hits = select_non_overlapping_variants(
                    hits, max_per_window, seed + window_count
                )
                
                if len(hits) != len(non_overlapping_hits):
                    logger.debug(f"  Selected {len(non_overlapping_hits)} non-overlapping variants out of {len(hits)}")
                
                hits = non_overlapping_hits
                
            except MemoryError:
                logger.error(f"Memory error processing window {chrom}:{w0}-{w1}")
                logger.error(f"Skipping window with {len(hits):,} variants due to memory constraints")
                hits = []
                
            # Periodically log memory and clean up
            current_time = time.time()
            if current_time - last_cleanup > cleanup_interval:
                current_mem = log_memory()
                if current_mem and current_mem > peak_mem:
                    peak_mem = current_mem
                cleanup()
                last_cleanup = current_time

        # Process variants in this window
        variants_in_window = []
        for v in hits:
            # Convert to 0-based position within window
            pos_in_window = v["pos"] - w0  # 1-based position in window
            # Check if position is within window bounds (1-based to len(seq))
            if pos_in_window < 1 or pos_in_window > len(seq):
                continue
                
            # Encode variant and add window-specific info
            enc = encode_variant(v["ref"], v["alt"])
            enc.update({
                "spec": v["spec"],
                "in_window_idx": pos_in_window,
                "span_in_window": (pos_in_window, pos_in_window + enc["ref_len"])
            })
            
            # Create a record for this variant
            variant_tuple = (v["pos"], v["ref"], v["alt"])
            ch = build_variant_channels(seq, [variant_tuple], w0, w1)
            
            rec = {
                "chrom": chrom,
                "start": w0,
                "end": w1,
                "bin_size": getattr(w, "bin_size", 1),
                "seq": seq,
                "variant_spec": enc["spec"],
                "var_type": enc["type"],
                "in_window_idx": enc["in_window_idx"],
                "span_in_window": list(enc["span_in_window"]) if enc["span_in_window"] else None,
                "ch_snp": ch["snp"],
                "ch_ins": ch["ins"],
                "ch_del": ch["del_"],
                "ch_any": ch["any"],
            }
            
            # If we need to apply the variant, collect it for batch processing
            if apply_alt:
                variants_in_window.append({
                    'pos': pos_in_window,  # 1-based position within window
                    'ref': v['ref'],
                    'alt': v['alt'],
                    'rec': rec
                })
            else:
                rows.append(rec)
        
        # Apply all variants to the sequence in a single pass
        if apply_alt and variants_in_window:
            try:
                # Sort variants by position to ensure correct processing order
                variants_in_window.sort(key=lambda x: x['pos'])
                
                # Create a list of variant dicts for apply_variants_to_sequence
                variant_dicts = []
                ins_rescued = 0
                
                for v in variants_in_window:
                    pos = v['pos'] 
                    ref = v['ref']
                    alt = v['alt'][0] if isinstance(v['alt'], list) else v['alt']
                    
                    # Check for insertions that might need rescuing
                    if len(ref) < len(alt):  # This is an insertion
                        # Try to rescue the insertion
                        new_pos, new_ref, new_alt = rescue_insertion_variant(
                            seq, pos, ref, alt, w0
                        )
                        
                        # Update variant if we found a better position
                        if new_pos != pos or new_ref != ref or new_alt != alt:
                            v_dict = {
                                'pos': new_pos,
                                'ref': new_ref, 
                                'alt': [new_alt] if isinstance(v['alt'], list) else new_alt
                            }
                            ins_rescued += 1
                        else:
                            v_dict = {'pos': pos, 'ref': ref, 'alt': v['alt']}
                    else:
                        v_dict = {'pos': pos, 'ref': ref, 'alt': v['alt']}
                        
                    variant_dicts.append(v_dict)
                    
                if ins_rescued > 0:
                    print(f"Rescued {ins_rescued} insertion variants")
                    
                # Apply all variants to the window sequence with relaxed reference checking
                try:
                    alt_seq = apply_variants_to_sequence(seq, variant_dicts, strict_ref_check=False)
                    
                    # Update records with the modified sequence
                    for v in variants_in_window:
                        v['rec']['seq_alt'] = alt_seq
                        rows.append(v['rec'])
                        
                except ValueError as e:
                    logger.error(f"Error applying variants to window {chrom}:{w0}-{w1}: {str(e)}")
                    # Add records with placeholder seq_alt for consistent structure
                    for v in variants_in_window:
                        v['rec']['seq_alt'] = 'N' * len(seq)  # Placeholder
                        rows.append(v['rec'])
 

            except Exception as e:
                logger.error(f"Unexpected error processing window {chrom}:{w0}-{w1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # Add records with placeholder seq_alt for consistent structure
                for v in variants_in_window:
                    v['rec']['seq_alt'] = 'N' * len(seq)  # Placeholder
                    rows.append(v['rec'])

            # Add batch writing code HERE - right after all variants for this window are processed
            # Check if we've processed a full batch of windows or reached shard size
            batch_count += 1
            if batch_count >= batch_size or len(rows) >= shard_size:
                if rows:  # Only write if we have rows
                    try:
                        logger.debug(f"Writing batch of {len(rows)} rows to shard {shard_idx}")
                        df = pd.DataFrame(rows)
                        outp = os.path.join(out, f"shard_{shard_idx:04d}.parquet")
                        df.to_parquet(outp, index=False)
                        logger.info(f"Wrote shard {shard_idx} with {len(df):,} rows")
                        del df  # Explicitly delete the DataFrame
                        rows.clear()
                        shard_idx += 1
                        batch_count = 0  # Reset batch counter
                        
                        # Force garbage collection after each write
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error writing shard {shard_idx}: {str(e)}")
                        # Continue with the next batch even if one fails

    # Write any remaining rows
    if rows:
        try:
            df = pd.DataFrame(rows)
            outp = os.path.join(out, f"shard_{shard_idx:04d}.parquet")
            df.to_parquet(outp, index=False)
            logger.info(f"Wrote final shard {outp} ({len(df):,} rows, {batch_count} windows)")
            del df  # Explicitly delete the DataFrame
            rows.clear()  # Clear the rows list
            
            # Final garbage collection
            if 'gc' in globals():
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error writing final shard: {str(e)}")
    
    # Final cleanup
    cleanup()
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("Variant Processing Complete!")
    logger.info("="*80)
    logger.info(f"  Total windows processed:   {window_count:,}")
    logger.info(f"  Total variants processed:  {processed_variants:,}")
    logger.info(f"  Average variants/window:   {processed_variants/max(1, window_count):.2f}")
    logger.info(f"  Output shards written:     {shard_idx + 1}")
    logger.info(f"  Total time:               {total_time/60:.1f} minutes")
    logger.info(f"  Processing speed:         {window_count/max(1, total_time):.1f} windows/sec")
    
    if HAS_PSUTIL:
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"  Final memory usage:       {current_mem:.1f} MB")
        logger.info(f"  Peak memory usage:        {peak_mem:.1f} MB")
    
    logger.info(f"  Output directory:         {os.path.abspath(out)}")
    logger.info("="*80)


def main():
    args = parse_args()
    run(
        vcf=args.vcf,
        windows=args.windows,
        out=args.out,
        apply_alt=bool(args.apply_alt),
        max_per_window=int(args.max_per_window),
        batch_size=10,  # Process 10 windows before writing out
        shard_size=int(args.shard_size),
        seed=args.seed,
        debug=args.debug,
    )


def prepare_variants(vcf: str, windows: str, out: str, apply_alt: bool = False, 
                    max_per_window: int = 0, shard_size: int = 50_000, seed: int = 42) -> None:
    """
    Prepare variant data for Betadogma training.
    
    Args:
        vcf: Path to VCF file or glob pattern for VCF files
        windows: Glob pattern for window Parquet shards from prepare_gencode.py
        out: Output directory for variant-aligned Parquet shards
        apply_alt: Whether to store sequences with variants applied
        max_per_window: Maximum number of variants per window (0 for unlimited)
        shard_size: Number of rows per output shard
        seed: Random seed for reproducibility
    """
    run(
        vcf=vcf,
        windows=windows,
        out=out,
        apply_alt=apply_alt,
        max_per_window=max_per_window,
        shard_size=shard_size,
        seed=seed,
    )


if __name__ == "__main__":
    main()