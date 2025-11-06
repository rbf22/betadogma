#!/usr/bin/env python3
"""
Convert SpliceVarDB TSV to VCF format.

Usage:
    python convert_splicevar_to_vcf.py input.tsv output.vcf [--build hg38]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO, Tuple, Union

import pandas as pd  # type: ignore


def parse_variant_id(variant_id: str) -> Tuple[str, int, str, str]:
    """Parse variant ID in format 'chrom-pos-ref-alt'.
    
    Args:
        variant_id: String like "1-100573238-T-C"
    
    Returns:
        Tuple of (chrom, pos, ref, alt)
    """
    # Remove quotes if present
    variant_id = variant_id.strip('"')

    parts = variant_id.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid variant ID format: {variant_id}")

    chrom, pos, ref, alt = parts

    # Add chr prefix if not present
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"

    return chrom, int(pos), ref, alt


def chromosome_sort_key(chrom: str) -> Tuple[int, Union[int, str]]:
    """
    Generate sort key for chromosome names.
    
    Returns tuple for proper sorting: (type, number)
    - chr1, chr2, ..., chr9, chr10, ..., chr22, chrX, chrY, chrM
    """
    chrom = chrom.replace("chr", "")

    # Numeric chromosomes
    if chrom.isdigit():
        return (0, int(chrom))

    # Sex chromosomes
    if chrom == "X":
        return (1, 0)
    if chrom == "Y":
        return (1, 1)

    # Mitochondrial
    if chrom in ["M", "MT"]:
        return (2, 0)

    # Others (patches, alternate contigs)
    return (3, chrom)


def write_vcf_header(f: TextIO, source_file: str, build: str = "hg38", chromosomes: Optional[List[str]] = None) -> None:
    """Write VCF header with contig information."""
    date = datetime.now().strftime("%Y%m%d")

    f.write("##fileformat=VCFv4.2\n")
    f.write(f"##fileDate={date}\n")
    f.write(f"##source=SpliceVarDB_converted_from_{Path(source_file).name}\n")
    f.write(f"##reference={build}\n")

    # Add contig lines (required for sorted VCF)
    if chromosomes:
        for chrom in chromosomes:
            f.write(f"##contig=<ID={chrom}>\n")

    # INFO field definitions
    f.write('##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">\n')
    f.write('##INFO=<ID=HGVS,Number=1,Type=String,Description="HGVS notation">\n')
    f.write('##INFO=<ID=METHOD,Number=1,Type=String,Description="Experimental method used">\n')
    f.write('##INFO=<ID=CLASSIFICATION,Number=1,Type=String,Description="Splice classification: Splice-altering, Normal, or Low-frequency">\n')
    f.write('##INFO=<ID=LOCATION,Number=1,Type=String,Description="Variant location: Exonic or Intronic">\n')
    f.write('##INFO=<ID=DOI,Number=.,Type=String,Description="DOI references">\n')
    f.write('##INFO=<ID=SPLICE_EFFECT,Number=1,Type=String,Description="Splice effect strength: STRONG (high confidence altering), MILD (weak evidence altering), or NONE (normal splicing)">\n')
    f.write('##INFO=<ID=SPLICE_SCORE,Number=1,Type=Float,Description="Splice disruption score: 1.0=strong, 0.5=mild, 0.0=none">\n')

    # Column header
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")


def escape_vcf_string(s: str) -> str:
    """Escape special characters for VCF INFO field."""
    # Remove quotes
    s = s.strip('"')
    # Replace spaces and special chars
    s = s.replace(" ", "_")
    s = s.replace(";", ",")
    s = s.replace("=", ":")
    s = s.replace("\t", "_")
    s = s.replace("\n", "_")
    return s


def load_and_parse_tsv(input_tsv: str, build: str = "hg38") -> pd.DataFrame:
    """
    Load TSV and parse all variants into a DataFrame.
    
    Returns:
        DataFrame with columns: chrom, pos, ref, alt, info_fields...
    """
    print(f"Loading and parsing {input_tsv}...")

    # Determine which column to use for coordinates
    coord_col_idx = 1 if build == "hg19" else 2  # 0-based after split

    variants = []
    errors = 0

    with open(input_tsv, "r") as f:
        # Read header
        header = f.readline().strip().split("\t")
        print(f"TSV columns: {header}")

        # Read all variants
        for line_num, line in enumerate(f, start=2):
            try:
                fields = line.strip().split("\t")

                if len(fields) < 9:
                    print(f"Warning: Line {line_num} has only {len(fields)} fields, skipping")
                    continue

                # Extract fields
                variant_id = fields[0]
                hg19 = fields[1]
                hg38 = fields[2]
                gene = fields[3]
                hgvs = fields[4]
                method = fields[5]
                classification = fields[6]
                location = fields[7]
                doi = fields[8]

                # Classify splice effect strength
                class_clean = classification.strip('"')
                if class_clean == "Splice-altering":
                    effect = "STRONG"
                    score = 1.0
                elif class_clean == "Low-frequency":
                    effect = "MILD"
                    score = 0.5
                elif class_clean == "Normal":
                    effect = "NONE"
                    score = 0.0
                else:
                    print(f"Warning: Unknown classification '{class_clean}' on line {line_num}")
                    effect = "UNKNOWN"
                    score = 0.0

                # Parse coordinates
                coord_string = hg38 if build == "hg38" else hg19
                chrom, pos, ref, alt = parse_variant_id(coord_string)

                variants.append({
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "hgvs": hgvs,
                    "method": method,
                    "classification": classification,
                    "location": location,
                    "doi": doi,
                    "splice_effect": effect,
                    "splice_score": score
                })

            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                errors += 1
                if errors > 10:
                    print("Too many errors, stopping")
                    break

    print(f"Parsed {len(variants):,} variants ({errors} errors)")

    # Convert to DataFrame
    df = pd.DataFrame(variants)

    return df


def sort_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort variants by chromosome and position.
    
    This is REQUIRED for tabix indexing!
    """
    print("\nSorting variants...")

    # Add sort key column
    df["_sort_key"] = df["chrom"].apply(chromosome_sort_key)

    # Sort by chromosome (using sort key) and position
    df = df.sort_values(["_sort_key", "pos"])

    # Drop sort key
    df = df.drop(columns=["_sort_key"])

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Sorted {len(df):,} variants")

    # Show chromosome distribution
    print("\nVariants per chromosome:")
    chrom_counts = df["chrom"].value_counts()

    # Sort chromosomes naturally
    chrom_order = sorted(chrom_counts.index, key=chromosome_sort_key)
    for chrom in chrom_order:
        count = chrom_counts[chrom]
        print(f"  {chrom}: {count:,}")

    return df


def write_sorted_vcf(df: pd.DataFrame, output_vcf: str, source_file: str, build: str = "hg38") -> None:
    """Write sorted DataFrame to VCF file."""
    print(f"\nWriting sorted VCF to {output_vcf}...")

    # Get unique chromosomes in sorted order
    chromosomes = sorted(df["chrom"].unique(), key=chromosome_sort_key)

    with open(output_vcf, "w") as f:
        # Write header
        write_vcf_header(f, source_file, build, chromosomes)

        # Write variants
        for _, row in df.iterrows():
            # Create INFO field
            info_parts = []
            info_parts.append(f'GENE={escape_vcf_string(row["gene"])}')
            info_parts.append(f'HGVS={escape_vcf_string(row["hgvs"])}')
            info_parts.append(f'METHOD={escape_vcf_string(row["method"])}')
            info_parts.append(f'CLASSIFICATION={escape_vcf_string(row["classification"])}')
            info_parts.append(f'LOCATION={escape_vcf_string(row["location"])}')
            info_parts.append(f'DOI={escape_vcf_string(row["doi"])}')
            info_parts.append(f'SPLICE_EFFECT={row["splice_effect"]}')
            info_parts.append(f'SPLICE_SCORE={row["splice_score"]}')

            info = ";".join(info_parts)

            # Write VCF line
            vcf_line = f'{row["chrom"]}\t{row["pos"]}\t.\t{row["ref"]}\t{row["alt"]}\t.\tPASS\t{info}\n'
            f.write(vcf_line)

    print(f"Wrote {len(df):,} variants to VCF")


def convert_tsv_to_vcf(input_tsv: Path, output_vcf: Path, build: str = "hg38", skip_mild: bool = False, only_strong: bool = False) -> None:
    """
    Convert SpliceVarDB TSV to sorted VCF format.
    
    Args:
        input_tsv: Path to input TSV file
        output_vcf: Path to output VCF file
        build: Genome build ('hg19' or 'hg38')
        skip_mild: If True, skip "Low-frequency" (mild effect) variants
        only_strong: If True, only include "Splice-altering" (strong effect) variants
    """
    print(f"{'='*80}")
    print("SpliceVarDB TSV to VCF Conversion")
    print(f"{'='*80}")
    print(f"Input:  {input_tsv}")
    print(f"Output: {output_vcf}")
    print(f"Build:  {build}")
    print(f"{'='*80}\n")

    # Step 1: Load and parse TSV
    df = load_and_parse_tsv(str(input_tsv), build)

    if len(df) == 0:
        print("ERROR: No variants loaded!")
        return

    # Step 2: Apply filters
    original_count = len(df)

    if skip_mild:
        df = df[df["splice_effect"] != "MILD"]
        print(f"\nFiltered MILD variants: {original_count:,} → {len(df):,}")

    if only_strong:
        df = df[df["splice_effect"] == "STRONG"]
        print(f"\nFiltered to STRONG only: {original_count:,} → {len(df):,}")

    # Step 3: Sort variants (REQUIRED for tabix!)
    df = sort_variants(df)

    # Step 4: Write sorted VCF
    write_sorted_vcf(df, str(output_vcf), str(input_tsv), build)

    # Step 5: Statistics
    print(f"\n{'='*80}")
    print("Conversion Complete!")
    print(f"{'='*80}")

    print(f"\nTotal variants: {len(df):,}")

    print("\nBy splice effect:")
    for effect in ["STRONG", "MILD", "NONE"]:
        count = (df["splice_effect"] == effect).sum()
        pct = 100 * count / len(df) if len(df) > 0 else 0
        print(f"  {effect:10s}: {count:6,} ({pct:5.1f}%)")

    print("\n💡 Next steps:")
    print(f"   bgzip {output_vcf}")
    print(f"   tabix -p vcf {output_vcf}.gz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SpliceVarDB TSV to sorted VCF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Splice Effect Classifications:
  STRONG (1.0) = "Splice-altering" - High confidence splice disruption (~25%)
  MILD   (0.5) = "Low-frequency"   - Weak evidence of disruption (~50%)
  NONE   (0.0) = "Normal"          - No splice disruption (~25%)

Examples:
  # Convert with all variants (hg38 coordinates)
  python convert_splicevar_to_vcf.py splicevar.tsv splicevar_hg38.vcf
  
  # Use hg19 coordinates
  python convert_splicevar_to_vcf.py splicevar.tsv splicevar_hg19.vcf --build hg19
  
  # Only STRONG effects (high confidence only)
  python convert_splicevar_to_vcf.py splicevar.tsv splicevar_strong.vcf --only-strong
  
  # Skip MILD (keep STRONG + NONE)
  python convert_splicevar_to_vcf.py splicevar.tsv splicevar_binary.vcf --skip-mild
  
  # Compress and index (REQUIRED for use with pipeline)
  bgzip splicevar_hg38.vcf
  tabix -p vcf splicevar_hg38.vcf.gz
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input TSV file from SpliceVarDB"
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output VCF file"
    )

    parser.add_argument(
        "--build",
        choices=["hg19", "hg38"],
        default="hg38",
        help="Genome build to use (default: hg38)"
    )

    parser.add_argument(
        "--skip-mild",
        action="store_true",
        help='Skip "Low-frequency" (MILD effect) variants'
    )

    parser.add_argument(
        "--only-strong",
        action="store_true",
        help='Only include "Splice-altering" (STRONG effect) variants'
    )

    args = parser.parse_args()

    # Check input exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Convert
    try:
        convert_tsv_to_vcf(
            args.input,
            args.output,
            build=args.build,
            skip_mild=args.skip_mild,
            only_strong=args.only_strong
        )

        print("\n✅ Success!")
        print("\n⚠️  Don't forget to compress and index:")
        print(f"   bgzip {args.output}")
        print(f"   tabix -p vcf {args.output}.gz")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
