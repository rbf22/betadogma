"""
Precompute variant-aligned channels for model windows.

Usage:
  python -m betadogma.data.prepare_variants \
    --vcf data/variants/cohort.vcf.gz \
    --windows "data/cache/gencode_v44_structural_base/*.parquet" \
    --out data/cache/variants_aligned \
    [--apply-alt] [--max-per-window 100] [--seed 42]

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
import gc
from glob import glob
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import numpy as np
import pandas as pd

# local helpers
from .encode import encode_variant, apply_variants_to_sequence, build_variant_channels, rescue_insertion_variant

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="VCF path or glob; .vcf(.gz) or .txt with VCF columns.")
    ap.add_argument("--windows", required=True, help="Glob for window Parquet shards from prepare_gencode.py")
    ap.add_argument("--out", required=True, help="Output directory for variant-aligned Parquet shards.")
    ap.add_argument("--apply-alt", action="store_true", help="Also store seq_alt (sequence with variant applied).")
    ap.add_argument("--max-per-window", type=int, default=100, 
                    help="Maximum variants per window (maintains type balance, 0 = unlimited)")
    ap.add_argument("--shard-size", type=int, default=1000,  # REDUCED for memory management
                    help="Rows per output shard.")
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


def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return 0


def _iter_vcf(path: str, logger=None):
    """
    Minimal VCF parser that yields dicts:
      chrom, pos (1-based), id, ref, alt (list[str]), qual, filter, info, samples...
    """
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        # Read and parse header
        header = []
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line[1:].rstrip("\n").split("\t")
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
            samples = t[8:]
            
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
            
            # Yield each alternate allele as a separate variant
            for alt_allele in variant['alt']:
                v = variant.copy()
                v['alt'] = [alt_allele]
                yield v


def _read_vcf_glob(vcf_glob: str, logger=None) -> List[Dict[str, Any]]:
    """Read VCF files and return a list of variant dictionaries."""
    paths = sorted(glob(vcf_glob)) if not os.path.isfile(vcf_glob) else [vcf_glob]
    assert paths, f"No VCF files matched: {vcf_glob}"
    
    out: List[Dict[str, Any]] = []
    variant_type_counts = {"SNP": 0, "INS": 0, "DEL": 0}
    
    logger.info(f"Processing VCF files: {paths}")
    
    for p in paths:
        logger.info(f"Reading variants from: {p}")
        file_variants = 0
        
        for r in _iter_vcf(p, logger):
            alt_str = ",".join(r['alt'])
            spec = f"{r['chrom']}:{r['pos']}{r['ref']}>{alt_str}"
            
            # Determine variant type
            ref = r['ref']
            alt = r['alt'][0]
            if len(ref) < len(alt):
                var_type = "INS"
            elif len(ref) > len(alt):
                var_type = "DEL"
            else:
                var_type = "SNP"
            
            variant_type_counts[var_type] += 1
            
            out.append({
                "chrom": r['chrom'], 
                "pos": r['pos'], 
                "ref": r['ref'], 
                "alt": r['alt'], 
                "spec": spec,
                "var_type": var_type
            })
            file_variants += 1
        
        logger.info(f"Read {file_variants:,} variants from {os.path.basename(p)}")
    
    # Print statistics
    total_variants = len(out)
    logger.info(f"\nTotal variants: {total_variants:,}")
    logger.info("Variant types:")
    for var_type, count in sorted(variant_type_counts.items()):
        pct = count / total_variants * 100 if total_variants > 0 else 0
        logger.info(f"  {var_type}: {count:>9,} ({pct:.1f}%)")
    
    return out


def get_variant_span(variant: Dict) -> Tuple[int, int]:
    """
    Get the 0-based span (start, end) that a variant affects.
    
    For SNPs and DELs: the span of the reference allele
    For INS: just the position (insertions don't consume reference bases beyond anchor)
    
    Returns:
        (start, end) in 0-based coordinates where end is exclusive
    """
    pos = variant['pos']  # 1-based
    ref = variant['ref']
    var_type = variant.get('var_type', 'SNP')
    
    start = pos - 1  # Convert to 0-based
    
    if var_type == 'INS':
        # Insertions only affect the anchor position
        end = start + 1
    else:
        # SNPs and DELs affect the full reference span
        end = start + len(ref)
    
    return (start, end)


def select_balanced_non_conflicting_variants(
    variants: List[Dict], 
    max_variants: int, 
    seed: int,
    logger=None,
    debug=False
) -> List[Dict]:
    """
    Select variants maintaining natural type distribution while avoiding conflicts.
    
    Strategy:
    1. Calculate natural proportions from all variants
    2. Shuffle variants within each type
    3. Place INS and DEL first (rarer, harder to place)
    4. Fill remaining slots with SNPs (abundant, easier to place)
    5. This maximizes our ability to hit target proportions
    
    Args:
        variants: List of variant dictionaries with 'var_type' field
        max_variants: Maximum number of variants to select
        seed: Random seed for reproducibility
        logger: Optional logger
        debug: Enable debug output
        
    Returns:
        List of selected non-conflicting variants maintaining type balance
    """
    if len(variants) <= max_variants:
        # Still need to remove conflicts even if under limit
        if len(variants) <= 1:
            return variants
        max_variants = len(variants)
    
    # Calculate natural proportions
    type_counts = {"SNP": 0, "INS": 0, "DEL": 0}
    for v in variants:
        var_type = v.get("var_type", "SNP")
        type_counts[var_type] += 1
    
    total = len(variants)
    target_proportions = {
        vtype: count / total 
        for vtype, count in type_counts.items()
    }
    
    # Calculate target counts
    target_counts = {
        vtype: int(max_variants * prop)
        for vtype, prop in target_proportions.items()
    }
    
    # Adjust for rounding - give extra slots to most common type
    slots_assigned = sum(target_counts.values())
    if slots_assigned < max_variants:
        # Find most common type and give it the extra slots
        most_common = max(target_proportions.items(), key=lambda x: x[1])[0]
        target_counts[most_common] += max_variants - slots_assigned
    
    if debug and logger:
        logger.debug(f"  Target counts: SNP={target_counts['SNP']}, "
                    f"INS={target_counts['INS']}, "
                    f"DEL={target_counts['DEL']}")
    
    # Group by type
    by_type = {"SNP": [], "INS": [], "DEL": []}
    for v in variants:
        var_type = v.get("var_type", "SNP")
        by_type[var_type].append(v)
    
    # Shuffle each type independently for randomness
    rng = random.Random(seed)
    for vtype in by_type:
        rng.shuffle(by_type[vtype])
    
    # OPTIMIZED ORDER: Process INS and DEL first, then SNPs
    selected = []
    selected_spans = []
    selected_counts = {"SNP": 0, "INS": 0, "DEL": 0}
    
    def try_add_variant(v, vtype):
        """Try to add a variant if it doesn't conflict and we haven't hit the limit."""
        if selected_counts[vtype] >= target_counts[vtype]:
            return False
        
        v_span = get_variant_span(v)
        
        # Check for conflicts
        for sel_span in selected_spans:
            if not (v_span[1] <= sel_span[0] or sel_span[1] <= v_span[0]):
                return False  # Conflict found
        
        # No conflict, add it
        selected.append(v)
        selected_spans.append(v_span)
        selected_counts[vtype] += 1
        return True
    
    # Phase 1: Alternate between INS and DEL (rarer types first)
    ins_idx = 0
    del_idx = 0
    
    while (selected_counts['INS'] < target_counts['INS'] or 
           selected_counts['DEL'] < target_counts['DEL']):
        
        added_any = False
        
        # Try to add an INS
        if selected_counts['INS'] < target_counts['INS']:
            while ins_idx < len(by_type['INS']):
                if try_add_variant(by_type['INS'][ins_idx], 'INS'):
                    added_any = True
                    ins_idx += 1
                    break
                ins_idx += 1
        
        # Try to add a DEL
        if selected_counts['DEL'] < target_counts['DEL']:
            while del_idx < len(by_type['DEL']):
                if try_add_variant(by_type['DEL'][del_idx], 'DEL'):
                    added_any = True
                    del_idx += 1
                    break
                del_idx += 1
        
        # If we couldn't add any INS or DEL, we're done with phase 1
        if not added_any:
            break
    
    # Phase 2: Fill remaining slots with SNPs
    snp_idx = 0
    while selected_counts['SNP'] < target_counts['SNP'] and snp_idx < len(by_type['SNP']):
        if try_add_variant(by_type['SNP'][snp_idx], 'SNP'):
            pass  # Successfully added
        snp_idx += 1
    
    # Phase 3: If we still have room and didn't hit targets, try to add more of any type
    remaining_slots = max_variants - len(selected)
    
    if remaining_slots > 0:
        # Try to add more variants in priority order: INS, DEL, SNP
        for vtype in ['INS', 'DEL', 'SNP']:
            if vtype == 'INS':
                idx = ins_idx
                variants_list = by_type['INS']
            elif vtype == 'DEL':
                idx = del_idx
                variants_list = by_type['DEL']
            else:
                idx = snp_idx
                variants_list = by_type['SNP']
            
            while idx < len(variants_list) and len(selected) < max_variants:
                if try_add_variant(variants_list[idx], vtype):
                    pass
                idx += 1
    
    # Sort by position for deterministic processing
    selected.sort(key=lambda x: x["pos"])
    
    if debug and logger:
        logger.debug(f"  Selected {len(selected)} non-conflicting variants: "
                    f"SNP={selected_counts['SNP']}, "
                    f"INS={selected_counts['INS']}, "
                    f"DEL={selected_counts['DEL']}")
        
        # Show how close we got to target proportions
        if len(selected) > 0:
            actual_props = {
                vtype: selected_counts[vtype] / len(selected)
                for vtype in ["SNP", "INS", "DEL"]
            }
            logger.debug(f"  Target proportions: SNP={target_proportions['SNP']:.2f}, "
                        f"INS={target_proportions['INS']:.2f}, "
                        f"DEL={target_proportions['DEL']:.2f}")
            logger.debug(f"  Actual proportions: SNP={actual_props['SNP']:.2f}, "
                        f"INS={actual_props['INS']:.2f}, "
                        f"DEL={actual_props['DEL']:.2f}")
    
    return selected


def remove_exact_duplicates(variants: List[Dict]) -> List[Dict]:
    """
    Remove variants at the exact same position with same ref/alt.
    Keeps first occurrence.
    """
    seen = set()
    unique = []
    
    for v in variants:
        alt = v['alt'][0] if isinstance(v['alt'], list) else v['alt']
        key = (v['chrom'], v['pos'], v['ref'], alt)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    
    return unique


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


def write_shard(rows: List[Dict], shard_idx: int, output_dir: str, logger) -> int:
    """Write a shard and return the number of rows written."""
    if not rows:
        return 0
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Write immediately
    outp = os.path.join(output_dir, f"shard_{shard_idx:04d}.parquet")
    df.to_parquet(outp, index=False, compression='snappy')
    
    ins_count = (df['var_type'] == 'INS').sum()
    num_rows = len(df)
    
    logger.info(f"Wrote shard {shard_idx:04d} with {num_rows:,} rows ({ins_count} INS)")
    
    # Aggressive cleanup
    del df
    del rows[:]  # Clear list in-place
    gc.collect()
    
    return num_rows


def run(vcf: str, windows: str, out: str, apply_alt: bool = False, 
        max_per_window: int = 100, shard_size: int = 1000, seed: int = 42, 
        debug: bool = False):
    """Main function to prepare variant data with natural distributions."""
    
    start_time = time.time()
    logger = setup_logging(debug)
    os.makedirs(out, exist_ok=True)
    random.seed(seed)
    
    # Track memory at start
    initial_mem = get_memory_usage()
    logger.info(f"Initial memory: {initial_mem:.1f} MB")
    
    # Tracking statistics
    stats = {
        "total_vcf_variants": 0,
        "total_windows": 0,
        "total_variants_in_windows": 0,
        "total_after_dedup": 0,
        "total_after_selection": 0,
        "total_processed": 0,
        "windows_limited": 0,
        "conflicts_removed": 0,
        # By type
        "vcf_snp": 0, "vcf_ins": 0, "vcf_del": 0,
        "window_snp": 0, "window_ins": 0, "window_del": 0,
        "selected_snp": 0, "selected_ins": 0, "selected_del": 0,
        "processed_snp": 0, "processed_ins": 0, "processed_del": 0,
    }
    
    logger.info("="*80)
    logger.info("Starting variant preparation")
    logger.info(f"  VCF: {vcf}")
    logger.info(f"  Windows: {windows}")
    logger.info(f"  Output: {out}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Max per window: {max_per_window if max_per_window > 0 else 'unlimited'}")
    logger.info(f"  Shard size: {shard_size}")
    logger.info(f"  Apply ALT: {apply_alt}")
    logger.info("-"*80)

    # Read VCF
    logger.info("Reading VCF files...")
    variants = _read_vcf_glob(vcf, logger)
    stats["total_vcf_variants"] = len(variants)
    
    vcf_mem = get_memory_usage()
    logger.info(f"Memory after loading VCF: {vcf_mem:.1f} MB (delta: {vcf_mem - initial_mem:.1f} MB)")
    
    # Count VCF types
    for v in variants:
        vtype = v.get("var_type", "SNP")
        stats[f"vcf_{vtype.lower()}"] += 1
    
    # Group by chromosome
    variants_by_chr: Dict[str, List[Dict[str, Any]]] = {}
    for v in variants:
        variants_by_chr.setdefault(v["chrom"], []).append(v)
    
    logger.info(f"\nVariants by chromosome:")
    for chrom in sorted(variants_by_chr.keys(), key=lambda x: (len(str(x).lstrip('chr')), x)):
        count = len(variants_by_chr[chrom])
        logger.info(f"  {chrom}: {count:>9,}")
    
    # Process windows with batch writing
    rows: List[Dict[str, Any]] = []
    shard_idx = 0
    
    total_windows = sum(1 for _ in stream_windows(windows))
    logger.info(f"\nProcessing {total_windows:,} windows...")
    logger.info("-"*80)
    
    window_stream = stream_windows(windows)
    last_log_time = time.time()
    
    def maybe_write_shard():
        """Write shard if buffer is full, return True if written."""
        nonlocal rows, shard_idx
        if len(rows) >= shard_size:
            mem_before = get_memory_usage()
            write_shard(rows, shard_idx, out, logger)
            # rows already cleared by write_shard
            shard_idx += 1
            mem_after = get_memory_usage()
            
            if debug:
                logger.debug(f"Memory before/after shard write: {mem_before:.1f} MB -> {mem_after:.1f} MB "
                           f"(freed: {mem_before - mem_after:.1f} MB)")
            
            return True
        return False
    
    for window_idx, w in enumerate(window_stream):
        stats["total_windows"] += 1
        chrom = w.chrom
        w0 = int(w.start)
        w1 = int(w.end)
        seq = w.seq
        
        # Progress logging
        if window_idx % 100 == 0 or time.time() - last_log_time >= 2.0:
            elapsed = time.time() - start_time
            rate = window_idx / elapsed if elapsed > 0 else 0
            eta = (total_windows - window_idx) / rate / 60 if rate > 0 else 0
            
            mem_str = ""
            if HAS_PSUTIL:
                mem_mb = get_memory_usage()
                mem_str = f"Mem: {mem_mb:.0f}MB | "
            
            logger.info(
                f"Processed {window_idx:,}/{total_windows:,} windows | "
                f"({window_idx/max(1,total_windows)*100:.1f}%) | "
                f"{rate:.1f} windows/sec | "
                f"ETA: {eta:.1f} min | "
                f"{mem_str}"
                f"Variants: {stats['total_processed']:,} | "
                f"INS: {stats['processed_ins']:,} | "
                f"Buffered: {len(rows):,}"
            )
            last_log_time = time.time()
        
        # Get variants in window
        hits = [
            v for v in variants_by_chr.get(chrom, []) 
            if w0 <= (v["pos"] - 1) < w1
        ]
        
        if not hits:
            continue
        
        stats["total_variants_in_windows"] += len(hits)
        
        # Count types in window
        for v in hits:
            vtype = v.get("var_type", "SNP")
            stats[f"window_{vtype.lower()}"] += 1
        
        # Remove exact duplicates
        hits = remove_exact_duplicates(hits)
        stats["total_after_dedup"] += len(hits)
        
        # Select balanced non-conflicting variants
        before_selection = len(hits)
        
        if max_per_window > 0 and len(hits) > max_per_window:
            stats["windows_limited"] += 1
            
            if debug:
                type_counts_before = {
                    "SNP": sum(1 for v in hits if v.get("var_type") == "SNP"),
                    "INS": sum(1 for v in hits if v.get("var_type") == "INS"),
                    "DEL": sum(1 for v in hits if v.get("var_type") == "DEL"),
                }
                logger.debug(
                    f"\nWindow {chrom}:{w0}-{w1} has {len(hits)} variants "
                    f"(SNP:{type_counts_before['SNP']}, "
                    f"INS:{type_counts_before['INS']}, "
                    f"DEL:{type_counts_before['DEL']})"
                )
            
            hits = select_balanced_non_conflicting_variants(
                hits, max_per_window, seed + window_idx, logger, debug
            )
        else:
            # Even if under limit, remove conflicts
            hits = select_balanced_non_conflicting_variants(
                hits, len(hits), seed + window_idx, logger, debug
            )
        
        conflicts_removed = before_selection - len(hits)
        stats["conflicts_removed"] += conflicts_removed
        stats["total_after_selection"] += len(hits)
        
        # Count selected types
        for v in hits:
            vtype = v.get("var_type", "SNP")
            stats[f"selected_{vtype.lower()}"] += 1
        
        # Process variants WITHOUT apply_alt first (simpler case)
        if not apply_alt:
            for v in hits:
                pos_in_window = v["pos"] - w0  # 1-based position in window
                
                if pos_in_window < 1 or pos_in_window > len(seq):
                    continue
                
                # Build variant channels
                variant_tuple = (v["pos"], v["ref"], v["alt"][0] if isinstance(v["alt"], list) else v["alt"])
                ch = build_variant_channels(seq, [variant_tuple], w0, w1)
                
                var_type = v.get("var_type", "SNP")
                
                rec = {
                    "chrom": chrom,
                    "start": w0,
                    "end": w1,
                    "bin_size": getattr(w, "bin_size", 1),
                    "seq": seq,
                    "variant_spec": v["spec"],
                    "var_type": var_type,
                    "in_window_idx": pos_in_window,
                    "span_in_window": [pos_in_window, pos_in_window + len(v["ref"])],
                    "ch_snp": ch["snp"],
                    "ch_ins": ch["ins"],
                    "ch_del": ch["del_"],
                    "ch_any": ch["any"],
                }
                
                rows.append(rec)
                stats["total_processed"] += 1
                stats[f"processed_{var_type.lower()}"] += 1
                
                # CHECK BUFFER SIZE IMMEDIATELY
                maybe_write_shard()
        
        # Process variants WITH apply_alt (more complex)
        else:
            # Collect variants to apply
            variants_to_apply = []
            
            for v in hits:
                pos_in_window = v["pos"] - w0  # 1-based position in window
                
                if pos_in_window < 1 or pos_in_window > len(seq):
                    continue
                
                # Build variant channels
                variant_tuple = (v["pos"], v["ref"], v["alt"][0] if isinstance(v["alt"], list) else v["alt"])
                ch = build_variant_channels(seq, [variant_tuple], w0, w1)
                
                var_type = v.get("var_type", "SNP")
                
                rec = {
                    "chrom": chrom,
                    "start": w0,
                    "end": w1,
                    "bin_size": getattr(w, "bin_size", 1),
                    "seq": seq,
                    "variant_spec": v["spec"],
                    "var_type": var_type,
                    "in_window_idx": pos_in_window,
                    "span_in_window": [pos_in_window, pos_in_window + len(v["ref"])],
                    "ch_snp": ch["snp"],
                    "ch_ins": ch["ins"],
                    "ch_del": ch["del_"],
                    "ch_any": ch["any"],
                }
                
                variants_to_apply.append({
                    'pos': pos_in_window,
                    'ref': v['ref'],
                    'alt': v['alt'],
                    'var_type': var_type,
                    'rec': rec
                })
            
            if variants_to_apply:
                # Sort by position
                variants_to_apply.sort(key=lambda x: x['pos'])
                
                # Rescue insertions
                variant_dicts = []
                for v in variants_to_apply:
                    pos = v['pos']
                    ref = v['ref']
                    alt = v['alt'][0] if isinstance(v['alt'], list) else v['alt']
                    var_type = v['var_type']
                    
                    if var_type == "INS":
                        new_pos, new_ref, new_alt = rescue_insertion_variant(
                            seq, pos, ref, alt, w0
                        )
                        variant_dicts.append({
                            'pos': new_pos,
                            'ref': new_ref,
                            'alt': [new_alt] if isinstance(v['alt'], list) else new_alt,
                            'var_type': 'INS',
                            'rec': v['rec']
                        })
                    else:
                        variant_dicts.append({
                            'pos': pos,
                            'ref': ref,
                            'alt': v['alt'],
                            'var_type': var_type,
                            'rec': v['rec']
                        })
                
                # Apply to sequence
                try:
                    alt_seq = apply_variants_to_sequence(seq, variant_dicts, strict_ref_check=False)
                    
                    for v in variant_dicts:
                        v['rec']['seq_alt'] = alt_seq
                        rows.append(v['rec'])
                        stats["total_processed"] += 1
                        stats[f"processed_{v['var_type'].lower()}"] += 1
                        
                        # CHECK BUFFER SIZE IMMEDIATELY
                        maybe_write_shard()
                        
                except Exception as e:
                    logger.error(f"Error applying variants to {chrom}:{w0}-{w1}: {e}")
                    # Add with placeholder
                    for v in variant_dicts:
                        v['rec']['seq_alt'] = 'N' * len(seq)
                        rows.append(v['rec'])
                        stats["total_processed"] += 1
                        stats[f"processed_{v['var_type'].lower()}"] += 1
                        
                        # CHECK BUFFER SIZE IMMEDIATELY
                        maybe_write_shard()
    
    # Write final shard
    if rows:
        write_shard(rows, shard_idx, out, logger)
        rows.clear()
    
    # Final statistics
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("Variant Processing Complete!")
    logger.info("="*80)
    
    logger.info(f"\nWindows:")
    logger.info(f"  Total processed:          {stats['total_windows']:,}")
    logger.info(f"  Windows limited:          {stats['windows_limited']:,}")
    
    logger.info(f"\nVariants:")
    logger.info(f"  In VCF:                   {stats['total_vcf_variants']:,}")
    logger.info(f"  In windows:               {stats['total_variants_in_windows']:,}")
    logger.info(f"  After deduplication:      {stats['total_after_dedup']:,}")
    logger.info(f"  Conflicts removed:        {stats['conflicts_removed']:,}")
    logger.info(f"  After selection:          {stats['total_after_selection']:,}")
    logger.info(f"  Successfully processed:   {stats['total_processed']:,}")
    
    # Type breakdown
    logger.info(f"\nVariant Type Distribution:")
    for stage, prefix in [("VCF", "vcf"), ("In Windows", "window"), 
                          ("Selected", "selected"), ("Processed", "processed")]:
        snp = stats[f"{prefix}_snp"]
        ins = stats[f"{prefix}_ins"]
        del_ = stats[f"{prefix}_del"]
        total = snp + ins + del_
        
        if total > 0:
            logger.info(f"  {stage}:")
            logger.info(f"    SNP: {snp:>9,} ({snp/total*100:.1f}%)")
            logger.info(f"    INS: {ins:>9,} ({ins/total*100:.1f}%)")
            logger.info(f"    DEL: {del_:>9,} ({del_/total*100:.1f}%)")
    
    logger.info(f"\nPerformance:")
    logger.info(f"  Total time:               {total_time/60:.1f} minutes")
    logger.info(f"  Processing speed:         {stats['total_windows']/max(1,total_time):.1f} windows/sec")
    logger.info(f"  Output shards:            {shard_idx + 1}")
    logger.info(f"  Output directory:         {os.path.abspath(out)}")
    
    if HAS_PSUTIL:
        mem_mb = get_memory_usage()
        logger.info(f"  Final memory usage:       {mem_mb:.1f} MB")
    
    logger.info("="*80)


def main():
    args = parse_args()
    run(
        vcf=args.vcf,
        windows=args.windows,
        out=args.out,
        apply_alt=bool(args.apply_alt),
        max_per_window=args.max_per_window,
        shard_size=int(args.shard_size),
        seed=args.seed,
        debug=args.debug,
    )


def prepare_variants(vcf: str, windows: str, out: str, apply_alt: bool = False, 
                    max_per_window: int = 100, shard_size: int = 1000, 
                    seed: int = 42, debug: bool = False) -> None:
    """
    Prepare variant data for Betadogma training.
    
    Args:
        vcf: Path to VCF file or glob pattern for VCF files
        windows: Glob pattern for window Parquet shards from prepare_gencode.py
        out: Output directory for variant-aligned Parquet shards
        apply_alt: Whether to store sequences with variants applied
        max_per_window: Maximum variants per window (maintains type balance, 0 = unlimited)
        shard_size: Number of rows per output shard (default 1000 for memory management)
        seed: Random seed for reproducibility
        debug: Enable debug logging
    """
    run(
        vcf=vcf,
        windows=windows,
        out=out,
        apply_alt=apply_alt,
        max_per_window=max_per_window,
        shard_size=shard_size,
        seed=seed,
        debug=debug,
    )


if __name__ == "__main__":
    main()