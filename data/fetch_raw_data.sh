#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# fetch_raw_data.sh
# Downloads raw inputs for BetaDogma:
#   - GRCh38 primary assembly FASTA (unzipped .fa + .fai)
#   - GENCODE GTF (unzipped .gtf)
#   - GTEx v8 expression + sample table
#   - Small example VCF (1000G chr22) bgzipped + .tbi
#
# Output layout (relative to this script's directory):
#   data/raw/{genome,gencode,gtex,variants}
# ------------------------------------------------------------------------------

# Root dirs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"
RAW_DIR="${DATA_DIR}/raw"
GENCODE_DIR="${RAW_DIR}/gencode"
GENOME_DIR="${RAW_DIR}/genome"
GTEX_DIR="${RAW_DIR}/gtex"
VAR_DIR="${RAW_DIR}/variants"

mkdir -p "${GENCODE_DIR}" "${GENOME_DIR}" "${GTEX_DIR}" "${VAR_DIR}"

# -----------------------------
# Versions / URLs (adjust as needed)
# -----------------------------
GENCODE_RELEASE="44"
GENCODE_BASE="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}"

GENCODE_GTF_URL="${GENCODE_BASE}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
GENOME_FASTA_URL="${GENCODE_BASE}/GRCh38.primary_assembly.genome.fa.gz"
GENCODE_GTF_MD5_URL="${GENCODE_GTF_URL}.md5"
GENOME_FASTA_MD5_URL="${GENOME_FASTA_URL}.md5"

# GTEx v8
GTEX_BASE="https://storage.googleapis.com/gtex_analysis_v8"
GTEX_EXPR_URL="${GTEX_BASE}/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
GTEX_SAMPLES_URL="${GTEX_BASE}/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"

# Example Variants: 1000 Genomes chr22 (bgzipped + .tbi available)
# Example: 1000 Genomes Phase 3 lifted to GRCh38
VAR_VCF_BGZ_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502_supporting/grch38_positions/ALL.chr22_GRCh38.genotypes.20170504.vcf.gz"
VAR_VCF_TBI_URL="${VAR_VCF_BGZ_URL}.tbi"

# -----------------------------
# Output filenames
# -----------------------------
GENCODE_GTF_GZ="${GENCODE_DIR}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
GENCODE_GTF="${GENCODE_DIR}/gencode.v${GENCODE_RELEASE}.annotation.gtf"

GENOME_FASTA_GZ="${GENOME_DIR}/GRCh38.primary_assembly.genome.fa.gz"
GENOME_FASTA="${GENOME_DIR}/GRCh38.primary_assembly.genome.fa"
GENOME_FASTA_FAI="${GENOME_FASTA}.fai"

GTEX_EXPR_TMP="${GTEX_DIR}/$(basename "$GTEX_EXPR_URL")"
GTEX_EXPR_OUT="${GTEX_DIR}/expression.tsv.gz"   # just a normalized name
GTEX_SAMPLES_OUT="${GTEX_DIR}/samples.tsv"

VAR_VCF_BGZ="${VAR_DIR}/$(basename "$VAR_VCF_BGZ_URL")"   # keeps original bgz name
VAR_VCF_TBI="${VAR_VCF_BGZ}.tbi"

# -----------------------------
# Helpers
# -----------------------------
need_tool () {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "× Missing required tool: $name"
    return 1
  fi
  return 0
}

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "✓ Exists: $(basename "$out")"
  else
    echo "↓ Downloading: $url"
    curl -fL --retry 3 --retry-connrefused --retry-delay 2 -o "$out" "$url"
    echo "✓ Saved: $out"
  fi
}

check_gzip() {
  local file="$1"
  if [[ "${file}" != *.gz ]]; then
    return 0
  fi
  echo "• Validating gzip: $file"
  gunzip -t "$file"
}

check_md5_remote() {
  local file="$1" md5_url="$2"
  if curl -fsSL "$md5_url" -o "${file}.md5"; then
    echo "• Verifying MD5: $(basename "$file")"
    local expected actual
    expected="$(awk '{print $1}' "${file}.md5")"
    if command -v md5sum >/dev/null 2>&1; then
      actual="$(md5sum "$file" | awk '{print $1}')"
    elif command -v md5 >/dev/null 2>&1; then
      actual="$(md5 -q "$file")"
    else
      echo "  (md5 tools missing; skipping check)"
      rm -f "${file}.md5"
      return 0
    fi
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

gunzip_if_needed () {
  local gz="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "✓ Uncompressed exists: $(basename "$dest")"
    return 0
  fi
  echo "• Uncompressing $(basename "$gz") → $(basename "$dest")"
  gunzip -c "$gz" > "$dest"
  echo "✓ Wrote: $dest"
}

# -----------------------------
# Tool checks (optional but recommended)
# -----------------------------
HAVE_SAMTOOLS=0
if need_tool samtools; then HAVE_SAMTOOLS=1; fi

HAVE_TABIX=0
if need_tool tabix && need_tool bgzip; then HAVE_TABIX=1; fi

# -----------------------------
# Download GENCODE (GTF + genome FASTA)
# -----------------------------
download "$GENCODE_GTF_URL"   "$GENCODE_GTF_GZ"
download "$GENOME_FASTA_URL"  "$GENOME_FASTA_GZ"

check_md5_remote "$GENCODE_GTF_GZ"  "$GENCODE_GTF_MD5_URL"
check_md5_remote "$GENOME_FASTA_GZ" "$GENOME_FASTA_MD5_URL"

check_gzip "$GENCODE_GTF_GZ"
check_gzip "$GENOME_FASTA_GZ"

# Unzip to formats our pipeline expects
gunzip_if_needed "$GENCODE_GTF_GZ"  "$GENCODE_GTF"
gunzip_if_needed "$GENOME_FASTA_GZ" "$GENOME_FASTA"

# FASTA index (.fai) for pyfaidx/samtools
if [[ ! -f "$GENOME_FASTA_FAI" ]]; then
  if [[ $HAVE_SAMTOOLS -eq 1 ]]; then
    echo "• Indexing FASTA with samtools faidx"
    samtools faidx "$GENOME_FASTA"
  else
    echo "• samtools not found; pyfaidx will build index on first use."
  fi
else
  echo "✓ Exists: $(basename "$GENOME_FASTA_FAI")"
fi

# -----------------------------
# Download GTEx v8
# -----------------------------
download "$GTEX_EXPR_URL" "$GTEX_EXPR_TMP"
check_gzip "$GTEX_EXPR_TMP"
download "$GTEX_SAMPLES_URL" "$GTEX_SAMPLES_OUT"

# Normalize expression name to expression.tsv.gz
if [[ "$GTEX_EXPR_TMP" != "$GTEX_EXPR_OUT" ]]; then
  cp -f "$GTEX_EXPR_TMP" "$GTEX_EXPR_OUT"
  echo "✓ Linked expression to expected name: $(basename "$GTEX_EXPR_OUT")"
fi

# -----------------------------
# Download small example VCF (variants)
# -----------------------------
download "$VAR_VCF_BGZ_URL" "$VAR_VCF_BGZ"
check_gzip "$VAR_VCF_BGZ"

# Index (tabix). If .tbi fetch fails, build locally if tools exist.
if [[ -f "$VAR_VCF_TBI" ]]; then
  echo "✓ Exists: $(basename "$VAR_VCF_TBI")"
else
  echo "↓ Trying to fetch remote index: ${VAR_VCF_TBI_URL:-"(none)"}"
  if curl -fL --retry 3 --retry-connrefused --retry-delay 2 -o "$VAR_VCF_TBI" "$VAR_VCF_TBI_URL"; then
    echo "✓ Downloaded: $(basename "$VAR_VCF_TBI")"
  else
    if [[ $HAVE_TABIX -eq 1 ]]; then
      echo "• Remote .tbi unavailable; creating index locally (bgzip/tabix)"
      # Ensure bgzip format (some servers serve .gz that isn't bgzipped)
      if ! tabix -t "$VAR_VCF_BGZ" >/dev/null 2>&1; then
        echo "• Recompressing VCF as bgzip"
        gunzip -c "$VAR_VCF_BGZ" | bgzip -c > "${VAR_VCF_BGZ}.tmp"
        mv -f "${VAR_VCF_BGZ}.tmp" "$VAR_VCF_BGZ"
      fi
      tabix -p vcf "$VAR_VCF_BGZ"
    else
      echo "× Missing bgzip/tabix; cannot index VCF."
      echo "  Install htslib (tabix/bgzip) or provide a .tbi alongside the VCF."
    fi
  fi
fi

echo
echo "All downloads complete."
echo "Raw data layout:"
echo "  GENCODE GTF:    ${GENCODE_GTF}"
echo "  Genome FASTA:   ${GENOME_FASTA}"
echo "  Genome FAI:     ${GENOME_FASTA_FAI} $( [[ -f "$GENOME_FASTA_FAI" ]] && echo "" || echo "(will be created by pyfaidx)")"
echo "  GTEx expr:      ${GTEX_EXPR_OUT}"
echo "  GTEx samples:   ${GTEX_SAMPLES_OUT}"
echo "  Variants VCF:   ${VAR_VCF_BGZ}"
echo "  Variants index: ${VAR_VCF_TBI} $( [[ -f "$VAR_VCF_TBI" ]] && echo "" || echo "(missing)")"
echo
echo "Next:"
echo "  1) Build shards:  python train/build_data.py --config train/configs/data.full.yaml"
echo "  2) Train heads:   python train/train.py --config train/configs/train.structural.yaml"