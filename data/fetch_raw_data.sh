#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# fetch_raw_data.sh
# Run from anywhere. Designed to live in: <repo>/data/fetch_raw_data.sh
# It will download to: <repo>/data/raw/{gencode,genome,gtex}
#
# Adjust the VERSION variables if you need different releases.
# ------------------------------------------------------------------------------

# Figure out repo/data root based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"
RAW_DIR="${DATA_DIR}/raw"
GENCODE_DIR="${RAW_DIR}/gencode"
GENOME_DIR="${RAW_DIR}/genome"
GTEX_DIR="${RAW_DIR}/gtex"

mkdir -p "${GENCODE_DIR}" "${GENOME_DIR}" "${GTEX_DIR}"

# -----------------------------
# Versions / URLs (edit as needed)
# -----------------------------
GENCODE_RELEASE="44"   # gencode.v44 (GRCh38)
GENCODE_BASE="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}"

# GENCODE (annotation + genome)
GENCODE_GTF_URL="${GENCODE_BASE}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
GENOME_FASTA_URL="${GENCODE_BASE}/GRCh38.primary_assembly.genome.fa.gz"
GENCODE_GTF_MD5_URL="${GENCODE_GTF_URL}.md5"
GENOME_FASTA_MD5_URL="${GENOME_FASTA_URL}.md5"

# GTEx v8 (public, Google Cloud bucket)
GTEX_BASE="https://storage.googleapis.com/gtex_analysis_v8"
GTEX_EXPR_URL="${GTEX_BASE}/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
GTEX_SAMPLES_URL="${GTEX_BASE}/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"

# -----------------------------
# Output filenames (match your prepare scripts’ expectations)
# -----------------------------
GENCODE_GTF_OUT="${GENCODE_DIR}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
GENOME_FASTA_OUT="${GENOME_DIR}/GRCh38.primary_assembly.genome.fa.gz"

# For GTEx, we normalize to the names used in your configs:
GTEX_EXPR_OUT="${GTEX_DIR}/expression.tsv.gz"  # note: original is .gct.gz; we just keep gzipped text
GTEX_SAMPLES_OUT="${GTEX_DIR}/samples.tsv"

# -----------------------------
# Helpers
# -----------------------------
download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "✓ Exists: $(basename "$out")"
  else
    echo "↓ Downloading: $url"
    curl -fsSL "$url" -o "$out"
    echo "✓ Saved: $out"
  fi
}

check_gzip() {
  local file="$1"
  echo "• Validating gzip: $file"
  gunzip -t "$file"
}

check_md5_remote() {
  local file="$1" md5_url="$2"
  if curl -fsSL "$md5_url" -o "${file}.md5"; then
    echo "• Verifying MD5: $(basename "$file")"
    # The .md5 file from GENCODE is "<md5sum>  <filename>"
    # We need to compare the checksum against the downloaded file.
    local expected
    expected="$(awk '{print $1}' "${file}.md5")"
    local actual
    actual="$(md5sum "$file" | awk '{print $1}')"
    if [[ "$expected" != "$actual" ]]; then
      echo "✗ MD5 mismatch for $file"
      echo "  expected: $expected"
      echo "  actual:   $actual"
      exit 1
    fi
    rm -f "${file}.md5"
    echo "✓ MD5 OK: $(basename "$file")"
  else
    echo "• MD5 not available for $(basename "$file"); skipping checksum."
  fi
}

# -----------------------------
# Download GENCODE (GTF + genome FASTA)
# -----------------------------
download "$GENCODE_GTF_URL"   "$GENCODE_GTF_OUT"
download "$GENOME_FASTA_URL"  "$GENOME_FASTA_OUT"

check_md5_remote "$GENCODE_GTF_OUT"  "$GENCODE_GTF_MD5_URL"
check_md5_remote "$GENOME_FASTA_OUT" "$GENOME_FASTA_MD5_URL"

check_gzip "$GENCODE_GTF_OUT"
check_gzip "$GENOME_FASTA_OUT"

# -----------------------------
# Download GTEx v8
# -----------------------------
# Expression matrix (we keep original name temporarily, then copy to expected name)
GTEX_EXPR_TMP="${GTEX_DIR}/$(basename "$GTEX_EXPR_URL")"
download "$GTEX_EXPR_URL" "$GTEX_EXPR_TMP"
check_gzip "$GTEX_EXPR_TMP"

# Sample attributes
download "$GTEX_SAMPLES_URL" "$GTEX_SAMPLES_OUT"

# Normalize expression filename to what your pipeline expects (tsv.gz)
# (The GTEx file is a gzipped GCT; your prepare step can treat it as gzipped tabular text.)
if [[ "$GTEX_EXPR_TMP" != "$GTEX_EXPR_OUT" ]]; then
  cp -f "$GTEX_EXPR_TMP" "$GTEX_EXPR_OUT"
  echo "✓ Linked expression to expected name: $(basename "$GTEX_EXPR_OUT")"
fi

echo
echo "All downloads complete."
echo "Raw data layout:"
echo "  ${GENCODE_GTF_OUT}"
echo "  ${GENOME_FASTA_OUT}"
echo "  ${GTEX_EXPR_OUT}  (copied from $(basename "$GTEX_EXPR_TMP"))"
echo "  ${GTEX_SAMPLES_OUT}"
echo
echo "Next step: run your data preparation pipeline (e.g., train/make_training_data.py)."