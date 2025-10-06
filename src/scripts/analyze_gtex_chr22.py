#!/usr/bin/env python3
"""
Standalone script to analyze GTEx chr22 junction data.
Extracts and counts high-quality junctions (≥2 reads in ≥N samples) for chromosome 22.
"""

import gzip
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set
from tqdm import tqdm


def parse_junction_id(junction_id: str) -> Tuple[str, int, int, str]:
    """Parse a junction ID string into its components.

    Expected format: 'chr1:12345-12456:strand' or 'chr1_12345_12456_strand' or 'chr1:12345-12456'

    Returns:
        Tuple of (chrom, start, end, strand)
    """
    # Try colon format first: chr1:12345-12456:+
    if ':' in junction_id and '-' in junction_id:
        parts = junction_id.split(':')
        if len(parts) >= 2:
            chrom = parts[0]
            positions = parts[1].split('-')
            if len(positions) == 2:
                start = int(positions[0])
                end = int(positions[1])
                strand = parts[2] if len(parts) > 2 else '+'
                return chrom, start, end, strand

    # Try underscore format: chr1_12228_12612 or chr1_12228_12612_+
    if '_' in junction_id:
        parts = junction_id.split('_')
        if len(parts) >= 3:
            # Check if last part is a strand indicator (+ or -)
            potential_strand = parts[-1] if len(parts) > 3 else None
            if potential_strand in ['+', '-']:
                # Has strand: chr1_12228_12612_+
                chrom = '_'.join(parts[:-3])
                start = int(parts[-3])
                end = int(parts[-2])
                strand = parts[-1]
            else:
                # No strand: chr1_12228_12612
                chrom = '_'.join(parts[:-2])
                start = int(parts[-2])
                end = int(parts[-1])
                strand = '+'
            return chrom, start, end, strand

    # If we get here, try to extract numbers in any format
    import re
    numbers = list(map(int, re.findall(r'\d+', junction_id)))
    if len(numbers) >= 2:
        start = min(numbers[0], numbers[1])
        end = max(numbers[0], numbers[1])
        chrom = re.sub(r'[^a-zA-Z0-9_]+', '', junction_id.split(str(numbers[0]))[0])
        strand = '+'  # Default strand if not specified
        return chrom, start, end, strand

    raise ValueError(f"Could not parse junction ID: {junction_id}")


def analyze_gtex_structure(gct_path: str) -> Dict:
    """
    Analyze the structure and content of the GTEx GCT file.

    Args:
        gct_path: Path to GTEx GCT file

    Returns:
        Dictionary with file structure information
    """
    print(f"Analyzing GTEx file structure: {gct_path}")

    # First, get the header and sample IDs
    with gzip.open(gct_path, 'rt') as f:
        # Read the version and dimension lines
        version = next(f).strip()
        dimensions_line = next(f).strip()
        n_rows_total, n_cols_total = map(int, dimensions_line.split('\t')[:2])

        # Get header line with sample IDs
        header_line = next(f).strip()
        header_fields = header_line.split('\t')
        sample_ids = [h for h in header_fields[2:] if h]  # Skip 'Name' and 'Description'

    print(f"\nFile Structure:")
    print(f"  Version: {version}")
    print(f"  Dimensions: {n_rows_total} rows × {n_cols_total} columns")
    print(f"  Header fields: {len(header_fields)}")
    print(f"  Sample IDs: {len(sample_ids)}")

    # Show header structure
    print(f"\nHeader structure:")
    print(f"  Field 1: '{header_fields[0]}' (Junction ID)")
    print(f"  Field 2: '{header_fields[1]}' (Gene ID)")
    print(f"  Fields 3-{len(header_fields)}: Sample count data ({len(sample_ids)} samples)")

    # Look at first few data rows
    print(f"\nFirst 5 data rows:")
    with gzip.open(gct_path, 'rt') as f:
        # Skip header lines (3 lines)
        for _ in range(3):
            next(f)

        for i, line in enumerate(f):
            if i >= 5:
                break

            fields = line.strip().split('\t')
            junction_id = fields[0]
            gene_id = fields[1]
            sample_counts = fields[2:7]  # Show first 5 sample counts

            print(f"  Row {i+1}:")
            print(f"    Junction ID: '{junction_id}'")
            print(f"    Gene ID: '{gene_id}'")
            print(f"    Sample counts (first 5): {sample_counts}")
            print(f"    Total samples: {len(fields) - 2}")

            # Parse junction to show chromosome
            try:
                chrom, start, end, strand = parse_junction_id(junction_id)
                print(f"    Parsed chromosome: '{chrom}' (positions {start}-{end}, strand {strand})")
            except:
                print(f"    Could not parse chromosome from: '{junction_id}'")

    # Count chromosomes present
    chromosome_counts = {}
    with gzip.open(gct_path, 'rt') as f:
        # Skip header
        for _ in range(3):
            next(f)

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 1:
                continue

            junction_id = fields[0]
            try:
                chrom, _, _, _ = parse_junction_id(junction_id)
                chromosome_counts[chrom] = chromosome_counts.get(chrom, 0) + 1
            except:
                chromosome_counts['unparseable'] = chromosome_counts.get('unparseable', 0) + 1

    print(f"\nChromosome distribution:")
    for chrom, count in sorted(chromosome_counts.items()):
        print(f"  {chrom}: {count:,}")

    # Check for chr22 specifically
    chr22_count = chromosome_counts.get('chr22', 0)
    print(f"\nchr22 specific:")
    print(f"  chr22 junctions found: {chr22_count:,}")
    if chr22_count == 0:
        print(f"  ❌ No chr22 data in this GTEx file")
    else:
        print(f"  ✅ Found chr22 data")

    return {
        'version': version,
        'total_rows': n_rows_total,
        'total_samples': len(sample_ids),
        'sample_ids': sample_ids[:5],  # First 5 for reference
        'chromosome_counts': chromosome_counts,
        'chr22_count': chr22_count
    }


def analyze_gtex_chr22_junctions(gct_path: str, min_count: int = 5) -> Dict:
    """
    Analyze GTEx data to count chr22 junctions by quality.

    Args:
        gct_path: Path to GTEx GCT file
        min_count: Minimum number of samples with ≥2 reads to consider "high quality"

    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing GTEx file: {gct_path}")
    print(f"Minimum count threshold: {min_count}")

    # First, get the header and sample IDs
    with gzip.open(gct_path, 'rt') as f:
        # Read the version and dimension lines
        version = next(f).strip()
        n_rows_total, n_cols_total = map(int, next(f).strip().split('\t')[:2])

        # Get header line with sample IDs
        header = next(f).strip().split('\t')
        all_sample_ids = [h for h in header[2:] if h]  # Skip 'Name' and 'Description'

    print(f"GCT version: {version}")
    print(f"Total rows: {n_rows_total:,}")
    print(f"Total samples: {len(all_sample_ids):,}")

    # Counters
    total_junctions = 0
    chr22_junctions = 0
    high_quality_chr22 = 0
    sample_count = len(all_sample_ids)

    # Process the GCT file
    with gzip.open(gct_path, 'rt') as f:
        # Skip header lines (3 lines)
        for _ in range(3):
            next(f)

        print(f"\nScanning for chr22 junctions with ≥{min_count} samples having ≥2 reads...")
        start_time = time.time()

        with tqdm(total=n_rows_total, desc="Analyzing junctions", unit="junctions") as pbar:
            # Process each junction
            for line_num, line in enumerate(f, 4):
                fields = line.strip().split('\t')

                # The format appears to be: Junction ID, Gene ID, then sample counts
                if len(fields) < 2:  # At least need junction ID and gene ID
                    pbar.update(1)
                    continue

                junction_id = fields[0]
                counts = fields[2:]  # The rest are sample counts

                # Handle case where we're missing one count (likely a trailing tab in the header)
                if len(counts) == sample_count - 1:
                    counts.append('0')  # Add a zero count for the missing sample
                elif len(counts) != sample_count:
                    pbar.update(1)
                    continue  # Skip malformed lines

                total_junctions += 1

                try:
                    # Parse junction ID
                    chrom, start, end, strand = parse_junction_id(junction_id)

                    if chrom == 'chr22':
                        chr22_junctions += 1

                        # Check if this is a high-quality junction (≥2 reads in ≥N samples)
                        try:
                            count_values = [int(float(c)) for c in counts]
                            # Count samples with ≥2 reads (two-read junction coverage)
                            samples_with_coverage = sum(1 for c in count_values if c >= 2)
                            if samples_with_coverage >= min_count:
                                high_quality_chr22 += 1
                        except (ValueError, IndexError):
                            pass  # Skip if counts can't be parsed

                except (ValueError, IndexError):
                    pass  # Skip malformed junction IDs

                pbar.update(1)

            elapsed = time.time() - start_time
            rate = line_num / elapsed
            remaining = (n_rows_total - line_num) / rate
            chr22_rate = chr22_junctions / elapsed if elapsed > 0 else 0

            print(f"  Found {chr22_junctions:,}","r22 junctions ({chr22_rate:.1f}/sec)")
            print(f"  High quality: {high_quality_chr22:,}","({high_quality_chr22/chr22_junctions*100:.1f}% of chr22)" if chr22_junctions > 0 else "")
            print(f"  Rate: {rate:.0f} junctions/sec | ETA: {remaining:.0f} sec")

    # Results
    results = {
        'total_junctions_analyzed': total_junctions,
        'chr22_junctions_found': chr22_junctions,
        'high_quality_chr22_junctions': high_quality_chr22,
        'high_quality_percentage': (high_quality_chr22 / chr22_junctions * 100) if chr22_junctions > 0 else 0,
        'min_count_threshold': min_count,
        'total_samples': sample_count
    }

    return results


def main():
    """Main function to run the analysis."""
    # Default path - adjust if needed
    default_gct_path = "/Users/robert_fenwick/SWE/betadogma/data/raw/gtex/junctions/GTEx_junctions.gct.gz"

    # Check if file exists
    gct_path = Path(default_gct_path)
    if not gct_path.exists():
        print(f"Error: GTEx file not found at {default_gct_path}")
        print("Please update the path in the script or ensure the file exists.")
        return

    print("=" * 80)
    print("GTEx File Structure Analysis")
    print("=" * 80)

    # First analyze the file structure
    structure = analyze_gtex_structure(str(gct_path))

    print("\n" + "=" * 80)
    print("GTEx chr22 Junction Quality Analysis")
    print("=" * 80)

    # Run quality analysis with different thresholds
    thresholds = [1, 5, 10, 20]

    for threshold in thresholds:
        print(f"\n--- Analysis with minimum samples ≥ {threshold} (≥2 reads each) ---")
        results = analyze_gtex_chr22_junctions(str(gct_path), min_count=threshold)

        print("Results:")
        print(f"  Total junctions analyzed: {results['total_junctions_analyzed']:,}")
        print(f"  chr22 junctions found: {results['chr22_junctions_found']:,}")
        print(f"  High-quality chr22 junctions: {results['high_quality_chr22_junctions']:,}")
        print(f"  High-quality percentage: {results['high_quality_percentage']:.1f}%")
        print(f"  Total samples: {results['total_samples']:,}")

        if results['high_quality_chr22_junctions'] > 0:
            print(f"  ✓ Found {results['high_quality_chr22_junctions']} chr22 junctions meeting threshold")
        else:
            print(f"  ✗ No chr22 junctions meet threshold ≥ {threshold}")

    print(f"\n{'=' * 80}")
    print("SUMMARY:")
    print(f"  File contains {structure['total_samples']:,} samples")
    print(f"  File contains {structure['total_rows']:,} total junctions")
    print(f"  Available chromosomes: {', '.join(sorted(structure['chromosome_counts'].keys()))}")
    if structure['chr22_count'] == 0:
        print(f"  ❌ chr22: No data found in this GTEx file")
    else:
        print(f"  ✅ chr22: {structure['chr22_count']:,} junctions found")


if __name__ == "__main__":
    main()
