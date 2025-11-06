#!/usr/bin/env python3
"""
organize_data.py - Process and organize BetaDogma data files.

This script:
  1. Processes reference genome (adds chr prefix if needed)
  2. Processes GENCODE v26 annotations (GTEx-compatible)
  3. Processes GTEx v8 transcript TPM (critical for isoform training)
  4. Processes GTEx v8 sample metadata
  5. Processes variant databases (ClinVar, 1000G, SpliceVar)
  6. Creates summary files for quick loading

All files are standardized to use 'chr' prefix and indexed appropriately.
"""

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Setup paths
ROOT = Path.cwd()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Create target directories
DIRS = [
    "genome",
    "annotation/gencode_v26",
    "annotation/gencode_v44",
    "gtex/v8/expression",
    "gtex/v8/metadata",
    "gtex/v8/junctions",
    "variants/clinvar",
    "variants/1000genomes",
    "variants/splicevar",
]

for d in DIRS:
    (PROCESSED / d).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🧬 BetaDogma Data Organization")
print("=" * 80)
print()


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd: List[str], description: Optional[str] = None, check: bool = True, capture: bool = True) -> bool:
    """Run a shell command with error handling."""
    if description:
        print(f"  {description}...")
    try:
        if capture:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.returncode == 0
        else:
            result_bytes = subprocess.run(cmd, check=check)
            return result_bytes.returncode == 0
    except subprocess.CalledProcessError as e:
        if capture and hasattr(e, "stderr"):
            print(f"  ✗ Command failed: {' '.join(cmd)}")
            print(f"    Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  ✗ Command not found: {cmd[0]}")
        print(f"    Please install: conda install -c bioconda {cmd[0]}")
        return False


def has_chr_prefix(file_path: Path, file_type: str) -> Optional[bool]:
    """Check if file uses 'chr' prefix in chromosome names."""
    try:
        if file_type == "fasta":
            opener = gzip.open if str(file_path).endswith(".gz") else open
            with opener(file_path, "rt") as f:
                first_line = f.readline()
                return first_line.startswith(">chr")

        elif file_type == "gtf":
            opener = gzip.open if str(file_path).endswith(".gz") else open
            with opener(file_path, "rt") as f:
                for line in f:
                    if not line.startswith("#"):
                        return line.split("\t")[0].startswith("chr")

        elif file_type == "vcf":
            # Use bcftools to check first non-header line
            cmd = ["bcftools", "view", "-H", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            first_line = result.stdout.split("\n")[0] if result.stdout else ""
            if first_line:
                return first_line.split("\t")[0].startswith("chr")
            return None

        elif file_type == "gct":
            opener = gzip.open if str(file_path).endswith(".gz") else open
            with opener(file_path, "rt") as f:
                next(f)  # Skip #1.2
                next(f)  # Skip dimensions
                data_line = f.readline()
                return "\tchr" in data_line or data_line.startswith("chr")

    except Exception as e:
        print(f"  ⚠️  Error checking chr prefix: {e}")
        return None

    return None


def add_chr_prefix_vcf(input_file: Path, output_file: Path) -> bool:
    """Add 'chr' prefix to VCF using bcftools."""
    print("  Adding 'chr' prefix using bcftools...")

    try:
        # Create chromosome rename file
        chr_map = {str(i): f"chr{i}" for i in range(1, 23)}
        chr_map.update({"X": "chrX", "Y": "chrY", "MT": "chrM", "M": "chrM"})

        chr_map_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        for old, new in chr_map.items():
            chr_map_file.write(f"{old}\t{new}\n")
        chr_map_file.close()

        # Use bcftools to rename chromosomes
        cmd = [
            "bcftools", "annotate",
            "--rename-chrs", chr_map_file.name,
            "-O", "z",  # Output compressed VCF
            "-o", str(output_file),
            str(input_file)
        ]

        success = run_command(cmd, "Running bcftools annotate", capture=False)

        # Clean up temp file
        os.unlink(chr_map_file.name)

        return success

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def add_chr_prefix_text(input_file: Path, output_file: Path, file_type: str) -> bool:
    """Add 'chr' prefix to text-based files (FASTA, GTF, GCT)."""
    print(f"  Adding 'chr' prefix to {input_file.name}...")

    try:
        opener = gzip.open if str(input_file).endswith(".gz") else open
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tmp")

        with opener(input_file, "rt") as f_in, temp_file:
            if file_type == "fasta":
                for line in f_in:
                    if line.startswith(">"):
                        chrom = line[1:].split()[0]
                        if not chrom.startswith("chr"):
                            # Add chr prefix to standard chromosomes
                            if chrom in [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]:
                                line = ">chr" + line[1:].replace("MT", "M").replace(">chrM", ">chrM")
                    temp_file.write(line)

            elif file_type == "gtf":
                for line in f_in:
                    if line.startswith("#"):
                        temp_file.write(line)
                    else:
                        parts = line.split("\t")
                        chrom = parts[0]
                        if not chrom.startswith("chr"):
                            if chrom in [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]:
                                parts[0] = "chr" + chrom.replace("MT", "M")
                        temp_file.write("\t".join(parts))

            elif file_type == "gct":
                # Keep first two header lines
                temp_file.write(f_in.readline())  # #1.2
                temp_file.write(f_in.readline())  # dimensions

                # Process data lines
                for line in f_in:
                    parts = line.split("\t")
                    if ":" in parts[0]:  # Junction format: chrom:start:end:strand
                        coords = parts[0].split(":")
                        if len(coords) >= 3 and not coords[0].startswith("chr"):
                            chrom = coords[0]
                            if chrom in [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]:
                                coords[0] = "chr" + chrom
                                parts[0] = ":".join(coords)
                    temp_file.write("\t".join(parts))

        temp_file.close()

        # Compress output appropriately
        if str(output_file).endswith(".gz"):
            if file_type in ["fasta", "gtf"]:
                # Use bgzip for files that will be indexed
                print("    Compressing with bgzip...")
                subprocess.run(["bgzip", "-c", temp_file.name],
                             stdout=open(str(output_file), "wb"),
                             check=True)
            else:
                # Use regular gzip for GCT files
                with open(temp_file.name, "rb") as f_in:
                    with gzip.open(output_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
        else:
            shutil.move(temp_file.name, output_file)

        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def index_file(file_path: Path, file_type: str) -> bool:
    """Index a genomic file."""
    # Remove existing index if present
    for ext in [".fai", ".tbi", ".csi"]:
        idx_path = Path(str(file_path) + ext)
        if idx_path.exists():
            idx_path.unlink()

    if file_type == "fasta":
        return run_command(["samtools", "faidx", str(file_path)], "Indexing FASTA")
    elif file_type == "vcf":
        return run_command(["tabix", "-p", "vcf", str(file_path)], "Indexing VCF")
    elif file_type == "gtf":
        return run_command(["tabix", "-p", "gff", str(file_path)], "Indexing GTF")

    return True


def sort_gtf(input_file: Path, output_file: Path) -> bool:
    """Sort a GTF file by chromosome and position."""
    print("    Sorting GTF file (this may take a few minutes)...")

    try:
        temp_headers = tempfile.NamedTemporaryFile(delete=False, suffix=".header", mode="w")
        temp_data = tempfile.NamedTemporaryFile(delete=False, suffix=".data", mode="w")
        temp_sorted = tempfile.NamedTemporaryFile(delete=False, suffix=".sorted", mode="w")

        # Close file handles, we'll reopen them
        temp_headers.close()
        temp_data.close()
        temp_sorted.close()

        # Step 1: Split headers and data
        print("    Splitting headers and data...")
        with gzip.open(input_file, "rt") as f_in:
            with open(temp_headers.name, "w") as f_header, open(temp_data.name, "w") as f_data:
                for line in f_in:
                    if line.startswith("#"):
                        f_header.write(line)
                    else:
                        f_data.write(line)

        # Step 2: Sort data by chromosome (col 1) and start position (col 4, numeric)
        print("    Sorting by chromosome and position...")
        with open(temp_sorted.name, "w") as f_out:
            # First write headers
            with open(temp_headers.name, "r") as f_header:
                f_out.write(f_header.read())

            # Then write sorted data
            # sort -t $'\t' -k1,1 -k4,4n
            subprocess.run(
                ["sort", "-t", "\t", "-k1,1", "-k4,4n", temp_data.name],
                stdout=f_out,
                check=True
            )

        # Step 3: Compress with bgzip
        print("    Compressing with bgzip...")
        subprocess.run(
            ["bgzip", "-f", "-c", temp_sorted.name],
            stdout=open(str(output_file), "wb"),
            check=True
        )

        # Clean up
        for temp_file in [temp_headers.name, temp_data.name, temp_sorted.name]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        print("    ✓ Sorted and compressed")
        return True

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()

        # Clean up on error
        for temp_file in [temp_headers.name, temp_data.name, temp_sorted.name]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return False


def process_file(src: Path, dst: Path, file_type: str, index: bool = True) -> bool:
    """Process a file: check chr prefix, convert if needed, copy, and index."""
    if not src.exists():
        print(f"  ⚠️  Source not found: {src}")
        return False

    print(f"  Processing {src.name}...")

    # Check if destination already exists with index
    if dst.exists():
        has_index = False
        for ext in [".fai", ".tbi", ".csi"]:
            if Path(str(dst) + ext).exists():
                has_index = True
                break

        if has_index:
            print("    ✓ Already processed with index")
            return True

    # Check if chr prefix exists
    has_chr = has_chr_prefix(src, file_type)

    if has_chr is True:
        print("    ✓ Already has 'chr' prefix")

        # For GTF, need to sort before indexing
        if file_type == "gtf":
            if not sort_gtf(src, dst):
                return False
        # For FASTA, need to convert gzip to bgzip for indexing
        elif file_type == "fasta" and str(src).endswith(".gz"):
            print("    Converting gzip to bgzip for indexing...")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            temp_file.close()

            try:
                # Decompress gzip
                with gzip.open(src, "rb") as f_in:
                    with open(temp_file.name, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Compress with bgzip
                subprocess.run(["bgzip", "-c", temp_file.name],
                             stdout=open(str(dst), "wb"),
                             check=True)

                # Clean up
                os.unlink(temp_file.name)

            except Exception as e:
                print(f"    ✗ Error converting to bgzip: {e}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return False
        else:
            # Just copy for other files
            shutil.copy2(src, dst)

        # Copy existing index if present (though likely won't exist for unsorted GTF)
        for ext in [".tbi", ".csi", ".fai"]:
            idx_src = Path(str(src) + ext)
            if idx_src.exists():
                shutil.copy2(idx_src, Path(str(dst) + ext))
                print("    ✓ Copied existing index")
                index = False  # Don't re-index

    elif has_chr is False:
        print("    Converting chromosome names...")

        if file_type == "vcf":
            if not add_chr_prefix_vcf(src, dst):
                return False
        else:
            if not add_chr_prefix_text(src, dst, file_type):
                return False

            # Sort GTF after adding chr prefix
            if file_type == "gtf":
                print("    Sorting GTF after chr prefix conversion...")
                temp_dst = Path(str(dst) + ".unsorted")
                dst.rename(temp_dst)
                if not sort_gtf(temp_dst, dst):
                    return False
                temp_dst.unlink()

    else:
        print("    ⚠️  Could not determine chr prefix, copying as-is")
        shutil.copy2(src, dst)

    # Index the file
    if index:
        if not index_file(dst, file_type):
            print("    ⚠️  Indexing failed")
            # Don't fail completely, file is still usable without index

    print("    ✓ Done")
    return True


# ============================================================================
# Processing Functions
# ============================================================================

def process_genome() -> bool:
    """Process reference genome."""
    print("\n[1/8] Reference Genome")
    print("-" * 80)

    src = RAW / "genome" / "GRCh38.primary_assembly.genome.fa.gz"
    dst = PROCESSED / "genome" / "GRCh38.fa.gz"

    if dst.exists() and (Path(str(dst) + ".fai").exists() or Path(str(dst)[:-3] + ".fai").exists()):
        print("  ✓ Already processed")
        return True

    return process_file(src, dst, "fasta", index=True)


def process_gencode_v26() -> bool:
    """Process GENCODE v26 (GTEx-compatible)."""
    print("\n[2/8] GENCODE v26 Annotations (GTEx-compatible)")
    print("-" * 80)

    files = [
        ("gencode.v26.annotation.gtf.gz", "gtf"),
        ("gencode.v26.transcripts.fa.gz", "fasta"),
        ("gencode.v26.pc_translations.fa.gz", "fasta"),
    ]

    success = True
    for filename, file_type in files:
        src = RAW / "gencode" / "v26" / filename
        dst = PROCESSED / "annotation" / "gencode_v26" / filename

        if dst.exists():
            print(f"  ✓ {filename} already processed")
            continue

        if not src.exists():
            print(f"  ⚠️  {filename} not found (skipping)")
            continue

        success &= process_file(src, dst, file_type, index=True)

    return success


def process_gencode_v44() -> bool:
    """Process GENCODE v44 (latest, optional)."""
    print("\n[3/8] GENCODE v44 Annotations (latest, optional)")
    print("-" * 80)

    src = RAW / "gencode" / "v44" / "gencode.v44.annotation.gtf.gz"
    dst = PROCESSED / "annotation" / "gencode_v44" / "gencode.v44.annotation.gtf.gz"

    if dst.exists():
        print("  ✓ Already processed")
        return True

    if not src.exists():
        print("  ⊘ Not downloaded (optional)")
        return True

    return process_file(src, dst, "gtf", index=True)


def process_gtex_transcript_tpm() -> bool:
    """Process GTEx v8 transcript TPM - THE CRITICAL FILE."""
    print("\n[4/8] GTEx v8 Transcript TPM ⭐ CRITICAL")
    print("-" * 80)

    src = RAW / "gtex" / "v8" / "expression" / "GTEx_v8_transcript_tpm.gct.gz"
    dst = PROCESSED / "gtex" / "v8" / "expression" / "transcript_tpm_summary.parquet"

    if dst.exists():
        print("  ✓ Already processed")
        return True

    if not src.exists():
        print("  ✗ Source file not found!")
        print("    This file is CRITICAL for isoform training.")
        return False

    print(f"  Processing {src.name} (this may take several minutes)...")

    try:
        import numpy as np
        import pandas as pd  # type: ignore[import-untyped]
        from tqdm import tqdm  # type: ignore[import-untyped]

        # Load sample attributes first (optional)
        sample_attrs = RAW / "gtex" / "v8" / "metadata" / "GTEx_v8_sample_attributes.txt"
        if sample_attrs.exists():
            samples_df = pd.read_csv(sample_attrs, sep="\t")
            sample_to_tissue = dict(zip(samples_df["SAMPID"], samples_df["SMTSD"], strict=False))
            print(f"    Loaded {len(sample_to_tissue):,} sample mappings")
        else:
            print("    ⚠️  Sample attributes not found, will use all samples")
            sample_to_tissue = {}

        # Read TPM file header
        print("    Reading TPM data...")
        with gzip.open(src, "rt") as f:
            version = f.readline().strip()  # #1.2
            dims = f.readline().strip().split("\t")
            n_transcripts = int(dims[0])
            n_samples = int(dims[1])
            print(f"    Found {n_transcripts:,} transcripts × {n_samples:,} samples")

        # Process in chunks
        print("    Computing transcript statistics...")
        chunk_size = 10000
        results = []

        for chunk in tqdm(
            pd.read_csv(src, sep="\t", skiprows=2, chunksize=chunk_size, compression="gzip"),
            total=n_transcripts // chunk_size + 1,
            desc="    Progress"
        ):
            # Extract transcript info
            transcript_ids = chunk.iloc[:, 0]  # First column
            gene_info = chunk.iloc[:, 1]  # Second column

            # Get expression values
            sample_cols = chunk.columns[2:]
            expr_values = chunk[sample_cols].values

            # Compute statistics
            stats = pd.DataFrame({
                "transcript_id": transcript_ids,
                "gene_id": gene_info,
                "mean_tpm": np.mean(expr_values, axis=1),
                "median_tpm": np.median(expr_values, axis=1),
                "max_tpm": np.max(expr_values, axis=1),
                "std_tpm": np.std(expr_values, axis=1),
                "num_samples_expressed": np.sum(expr_values > 1.0, axis=1),
                "num_samples_detected": np.sum(expr_values > 0.1, axis=1),
            })

            results.append(stats)

        # Combine and save
        print("    Combining results...")
        final_df = pd.concat(results, ignore_index=True)

        print("    Saving to parquet...")
        final_df.to_parquet(dst, index=False, compression="snappy")

        print(f"    ✓ Processed {len(final_df):,} transcripts")
        print(f"    ✓ Mean TPM range: {final_df['mean_tpm'].min():.2f} - {final_df['mean_tpm'].max():.2f}")
        print(f"    ✓ {(final_df['mean_tpm'] > 1.0).sum():,} transcripts with mean TPM > 1.0")

        return True

    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("    Install with: pip install pandas numpy tqdm pyarrow")
        return False
    except Exception as e:
        print(f"  ✗ Error processing TPM data: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_gtex_metadata() -> bool:
    """Process GTEx v8 sample metadata."""
    print("\n[5/8] GTEx v8 Sample Metadata")
    print("-" * 80)

    src = RAW / "gtex" / "v8" / "metadata" / "GTEx_v8_sample_attributes.txt"
    dst = PROCESSED / "gtex" / "v8" / "metadata" / "sample_attributes.txt"

    if dst.exists():
        print("  ✓ Already processed")
        return True

    if not src.exists():
        print("  ⚠️  Source not found (optional)")
        return True

    print(f"  Copying {src.name}...")
    shutil.copy2(src, dst)

    # Create tissue summary
    try:
        import pandas as pd
        df = pd.read_csv(src, sep="\t")
        tissue_counts = df["SMTSD"].value_counts()

        summary = PROCESSED / "gtex" / "v8" / "metadata" / "tissue_summary.txt"
        with open(summary, "w") as f:
            f.write("Tissue\tSample Count\n")
            for tissue, count in tissue_counts.items():
                f.write(f"{tissue}\t{count}\n")

        print(f"    ✓ Found {len(tissue_counts)} tissues")
        print(f"    ✓ Total samples: {len(df):,}")
        return True

    except Exception as e:
        print(f"    ⚠️  Could not create summary: {e}")
        return True  # Non-critical


def process_variants() -> bool:
    """Process variant databases."""
    print("\n[6/8] Variant Databases")
    print("-" * 80)

    variants = [
        ("clinvar", "clinvar.vcf.gz", "ClinVar"),
        ("1000genomes", "1000GENOMES-phase_3.vcf.gz", "1000 Genomes"),
        ("splicevar", "splicevar_hg38.vcf.gz", "SpliceVarDB"),
    ]

    success = True
    for subdir, filename, name in variants:
        src = RAW / "variants" / subdir / filename
        dst = PROCESSED / "variants" / subdir / filename

        print(f"\n  {name}:")

        if dst.exists() and Path(str(dst) + ".tbi").exists():
            print("    ✓ Already processed")
            continue

        if not src.exists():
            print(f"    ⚠️  Source not found: {src}")
            continue

        if process_file(src, dst, "vcf", index=True):
            print(f"    ✓ {name} ready")
        else:
            print(f"    ✗ {name} failed")
            success = False

    return success


def process_gtex_junctions() -> bool:
    """Process GTEx junctions (optional)."""
    print("\n[7/8] GTEx v8 Junctions (optional)")
    print("-" * 80)

    src = RAW / "gtex" / "v8" / "junctions" / "GTEx_v8_junctions.gct.gz"
    dst = PROCESSED / "gtex" / "v8" / "junctions" / "junctions_filtered.parquet"

    if dst.exists():
        print("  ✓ Already processed")
        return True

    if not src.exists():
        print("  ⊘ Not downloaded (optional, large file)")
        return True

    print(f"  Processing {src.name}...")
    print("  ⚠️  This is a large file and may take 10-30 minutes...")

    try:
        import pandas as pd
        from tqdm import tqdm

        # Read in chunks
        print("    Reading junction data...")
        chunks = []
        chunk_size = 5000

        # Initialize counters
        total_chunks = 0
        passing_chunks = 0
        total_junctions = 0
        passing_junctions = 0

        # Get total chunks for progress bar
        with gzip.open(src, "rt") as f:
            total_rows = sum(1 for _ in f) - 2  # Subtract header rows
        total_chunks_estimate = (total_rows + chunk_size - 1) // chunk_size

        for chunk in tqdm(
            pd.read_csv(src, sep="\t", skiprows=2, chunksize=chunk_size, compression="gzip"),
            total=total_chunks_estimate,
            desc="    Processing chunks"
        ):
            total_chunks += 1
            total_junctions += len(chunk)

            # Parse junction coordinates from format: chr1_21885452_21887214
            # First check if we're using the old format with colons
            first_col = chunk.iloc[:, 0]

            # Handle the underscore-separated format
            coords = first_col.str.extract(r"([^_]+)_(\d+)_(\d+)")
            coords.columns = ["chrom", "start", "end"]
            # Add strand as unknown since it's not in this format
            coords["strand"] = "+"

            # Convert to numeric, coercing errors to NaN
            coords["start"] = pd.to_numeric(coords["start"], errors="coerce")
            coords["end"] = pd.to_numeric(coords["end"], errors="coerce")

            # Statistics
            sample_cols = chunk.columns[2:]

            coords["num_samples"] = (chunk[sample_cols] > 5).sum(axis=1)

            # Filter
            mask = (
                coords["start"].notna() &
                coords["end"].notna() &
                (coords["num_samples"] >= 5)
            )

            if mask.any():
                passing_junctions_in_chunk = mask.sum()
                passing_junctions += passing_junctions_in_chunk
                chunks.append(coords[mask])
                passing_chunks += 1

            else:
                tqdm.write("\nNo junctions passed filters in this chunk")

        # Print summary
        print("\n    === Filtering Summary ===")
        print(f"    Total chunks processed: {total_chunks}")
        print(f"    Chunks with passing junctions: {passing_chunks} ({passing_chunks/total_chunks*100:.1f}%)")
        print(f"    Total junctions processed: {total_junctions:,}")
        print(f"    Junctions passing filters: {passing_junctions:,} ({passing_junctions/total_junctions*100:.1f}%)")

        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            final_df.to_parquet(dst, index=False, compression="snappy")
            print(f"\n    ✓ Saved {len(final_df):,} high-confidence junctions")
            return True
        else:
            print("\n    ⚠️  No junctions passed filtering")
            return False

    except ImportError:
        print("  ⚠️  pandas/tqdm not installed, skipping")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def create_summary() -> bool:
    """Create a summary of all processed files."""
    print("\n[8/8] Creating Summary")
    print("-" * 80)

    summary_file = PROCESSED / "data_summary.txt"

    with open(summary_file, "w") as f:
        f.write("BetaDogma Processed Data Summary\n")
        f.write("=" * 80 + "\n\n")

        total_size: float = 0
        for path in sorted(PROCESSED.rglob("*")):
            if path.is_file():
                rel = path.relative_to(PROCESSED)
                size: float = path.stat().st_size / (1024**2)
                total_size += size
                f.write(f"{rel}\n  Size: {size:.1f} MB\n\n")

        f.write(f"\nTotal size: {total_size/1024:.2f} GB\n")

    print(f"  ✓ Summary written to {summary_file}")
    return True


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Run all processing steps."""

    # Check dependencies
    print("Checking dependencies...")
    required = {
        "samtools": "samtools",
        "tabix": "htslib",
        "bgzip": "htslib",
        "bcftools": "bcftools",
    }
    missing = []
    for cmd, package in required.items():
        if not shutil.which(cmd):
            missing.append(f"{cmd} (install: conda install -c bioconda {package})")

    if missing:
        print("✗ Missing required tools:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("✓ All required tools found\n")

    # Run processing steps
    steps = [
        ("Reference Genome", process_genome),
        ("GENCODE v26", process_gencode_v26),
        ("GENCODE v44", process_gencode_v44),
        ("GTEx Transcript TPM", process_gtex_transcript_tpm),
        ("GTEx Metadata", process_gtex_metadata),
        ("Variants", process_variants),
        ("GTEx Junctions", process_gtex_junctions),
        ("Summary", create_summary),
    ]

    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            return 1
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final summary
    print("\n" + "=" * 80)
    print("📊 Processing Summary")
    print("=" * 80)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\n" + "=" * 80)

    critical_failed = not results.get("GTEx Transcript TPM", False) or not results.get("GENCODE v26", False)

    if all(results.values()):
        print("✅ All processing steps completed successfully!")
    elif critical_failed:
        print("❌ CRITICAL files failed to process!")
        print("   Cannot proceed with isoform training without:")
        print("   - GENCODE v26 annotations")
        print("   - GTEx v8 transcript TPM")
    else:
        print("⚠️  Some optional steps failed (see above)")
        print("   You can proceed with core functionality")

    print("\n📁 Processed data location:")
    print(f"   {PROCESSED}/")

    print("\n📝 Next steps:")
    print("   1. Run preprocessing to merge GTEx + GENCODE data")
    print("   2. Generate isoform annotations for your training regions")
    print("   3. Update training data with isoform labels")

    return 0 if not critical_failed else 1


if __name__ == "__main__":
    sys.exit(main())
