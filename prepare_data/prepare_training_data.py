#!/usr/bin/env python3
"""prepare_training_data.py - Generate reference sequences with isoform annotations and variant metadata."""

import json
import logging
import os
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pyranges as pr  # type: ignore[import-untyped]
import pysam
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Global counter for tracking progress
processed_counter = 0
last_update_time = time.time()

# Standard genetic code (nuclear chromosomes)
CODON_TABLE_STANDARD = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}

# Mitochondrial genetic code (chrM)
# Key differences from standard code:
# - AGA, AGG: R -> * (stop codons)
# - ATA: I -> M (methionine)
# - TGA: * -> W (tryptophan)
CODON_TABLE_MITO = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "W", "TGG": "W",  # TGA -> W (not stop)
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "M", "ATG": "M",  # ATA -> M (not I)
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "*", "AGG": "*",  # AGA, AGG -> * (stop)
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}

@dataclass
class GenomicWindow:
    """Container for genomic window data."""
    chrom: str
    start: int
    end: int
    transcript_id: str
    gene_name: str
    strand: str = "+"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingDataGenerator:
    """Generate reference sequences + isoform annotations + variant metadata for training."""

    def __init__(
        self,
        reference_fasta: str,
        gencode_gtf: str,
        clinvar_vcf: str,
        splicevar_vcf: str,
        gtex_tpm: str,
        thousand_genomes_vcf: Optional[str] = None,
        gtex_junctions: Optional[str] = None,
        gtex_expression: Optional[str] = None,
        output_dir: Optional[str] = None,
        window_size: int = 300000,
        step_size: int = 150000,
        max_clinvar_variants: Optional[int] = None,
        max_benign_variants: Optional[int] = None,
        gtex_sample_min: int = 50,
    ):
        """Initialize the training data generator.
        
        Args:
            reference_fasta: Path to reference genome FASTA file
            gencode_gtf: Path to Gencode GTF file
            clinvar_vcf: Path to ClinVar VCF file (bgzipped with tabix index)
            splicevar_vcf: Path to splice-altering variants VCF
            gtex_tpm: Path to GTEx transcript TPM file (parquet)
            thousand_genomes_vcf: Path to 1000 Genomes VCF
            gtex_junctions: Path to GTEx junctions file
            gtex_expression: Path to GTEx expression summary file
            output_dir: Directory to save output files
            window_size: Size of the sliding window in base pairs
            step_size: Step size between windows
            max_clinvar_variants: Maximum number of ClinVar variants per chromosome
            max_benign_variants: Maximum number of benign variants per chromosome
        """
        print("="*80)
        print("INITIALIZING TRAINING DATA GENERATOR WITH ISOFORM ANNOTATION")
        print("="*80)

        self.genome = pysam.FastaFile(reference_fasta)
        self.window_size = window_size
        self.step_size = step_size
        self.gtex_sample_min = gtex_sample_min
        self.max_clinvar_variants = max_clinvar_variants
        self.max_benign_variants = max_benign_variants
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWindow size: {window_size:,} bp")
        print(f"Step size: {step_size:,} bp")
        print(f"Minimum GTEx samples: {gtex_sample_min}")
        if self.output_dir:
            print(f"Output directory: {self.output_dir}")

        # Store VCF paths
        self.clinvar_vcf_path = clinvar_vcf
        self.splicevar_vcf_path = splicevar_vcf
        self.thousand_genomes_vcf_path = thousand_genomes_vcf

        # Verify VCF files
        self._verify_vcf_files()

        # In-memory data structures (loaded per chromosome)
        self.clinvar: Dict[str, Any] = {}
        self.splice_variants: Dict[str, Any] = {}
        self.benign_variants: Dict[str, Any] = {}

        # Indexed variants for fast lookup (position -> variants)
        self.clinvar_index: Dict[str, Any] = {}
        self.splice_variants_index: Dict[str, Any] = {}
        self.benign_variants_index: Dict[str, Any] = {}

        # Caching for performance optimization
        self._isoform_cache: Dict[Any, List[Dict[str, Any]]] = {}  # Cache isoforms by (chrom, start, end)
        self._cds_seq_cache: Dict[Any, Any] = {}  # Cache CDS sequences by (chrom, tx_id)
        self._current_chrom: Optional[str] = None
        self._chrom_transcripts: Dict[str, Any] = {}
        self._cds_timings: Dict[str, List[float]] = {"fetch": [], "translate": [], "nmd": []}
        self._seq_cache: Dict[Any, Any] = {}  # Cache genome sequences by (chrom, start, end)
        self._variant_cache: Dict[Any, Any] = {}  # Cache variant lookups by (chrom, start, end)

        print("\nLoading static data files...")
        self.transcripts = self._load_gencode(gencode_gtf)

        # Load GTEx transcript TPM data
        self.transcript_tpm = {}
        if gtex_tpm and os.path.exists(gtex_tpm):
            print("\nLoading GTEx transcript TPM data...")
            self.transcript_tpm = self._load_gtex_tpm(gtex_tpm)
        else:
            print("\n⚠️  GTEx TPM file not provided, all isoforms will have TPM=0")

        # Load GTEx junction data
        self.junctions: Dict[str, Any] = {}
        if gtex_junctions and os.path.exists(gtex_junctions):
            print("\nLoading GTEx junction data...")
            self.junctions = self._load_gtex_junctions(gtex_junctions)
        else:
            print("\n⚠️  GTEx junctions file not provided")

        print("\n✅ Initialization complete")
        print(f"  Transcripts: {len(self.transcripts):,}")
        print(f"  Transcript TPM entries: {len(self.transcript_tpm):,}")
        print("  VCF files ready for loading")

    def _get_codon_table(self, chrom: str) -> Dict[str, str]:
        """Get the appropriate codon table for the chromosome.
        
        Args:
            chrom: Chromosome name
            
        Returns:
            Codon table dictionary (standard or mitochondrial)
        """
        if chrom in ["chrM", "MT", "M"]:
            return CODON_TABLE_MITO
        return CODON_TABLE_STANDARD

    def _load_gtex_tpm(self, tpm_path: str) -> Dict[str, float]:
        """Load GTEx transcript TPM data.
        
        Expected format: Parquet with columns ['transcript_id', 'mean_tpm']
        or CSV/TSV with transcript_id and TPM columns
        """
        print(f"  Loading transcript TPM from {tpm_path}")

        try:
            if tpm_path.endswith(".parquet"):
                df = pd.read_parquet(tpm_path)
            elif tpm_path.endswith(".csv"):
                df = pd.read_csv(tpm_path)
            elif tpm_path.endswith(".tsv") or tpm_path.endswith(".txt"):
                df = pd.read_csv(tpm_path, sep="\t")
            else:
                print(f"  ⚠️  Unknown file format: {tpm_path}")
                return {}

            # Find TPM column
            tpm_col = None
            for col in ["mean_tpm", "median_tpm", "tpm", "TPM"]:
                if col in df.columns:
                    tpm_col = col
                    break

            if tpm_col is None:
                print(f"  ⚠️  No TPM column found in {tpm_path}")
                print(f"  Available columns: {df.columns.tolist()}")
                return {}

            # Find transcript ID column
            id_col = None
            for col in ["transcript_id", "transcript", "ENST"]:
                if col in df.columns:
                    id_col = col
                    break

            if id_col is None:
                print(f"  ⚠️  No transcript ID column found in {tpm_path}")
                print(f"  Available columns: {df.columns.tolist()}")
                return {}

            # Create dictionary
            tpm_dict = {}
            for _, row in df.iterrows():
                tx_id = str(row[id_col]).split(".")[0]  # Remove version number
                tpm_dict[tx_id] = float(row[tpm_col])

            print(f"  ✅ Loaded TPM for {len(tpm_dict):,} transcripts")
            if tpm_dict:
                print(f"  TPM range: {min(tpm_dict.values()):.2f} - {max(tpm_dict.values()):.2f}")

            return tpm_dict

        except Exception as e:
            print(f"  ❌ Error loading TPM data: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _translate_cds(self, cds_seq: str, chrom: str = "chr1") -> Optional[str]:
        """Translate CDS sequence to protein using appropriate genetic code.
        
        Args:
            cds_seq: CDS sequence (must be in-frame)
            chrom: Chromosome name (to select genetic code)
            
        Returns:
            Protein sequence or None if translation fails
        """
        if len(cds_seq) < 3:
            return None

        # Ensure length is multiple of 3
        cds_seq = cds_seq[:len(cds_seq) - len(cds_seq) % 3]

        # Get appropriate codon table
        codon_table = self._get_codon_table(chrom)

        protein = []
        for i in range(0, len(cds_seq), 3):
            codon = cds_seq[i:i+3]
            if len(codon) == 3:
                aa = codon_table.get(codon, "X")
                if aa == "*":  # Stop codon
                    break
                protein.append(aa)

        return "".join(protein) if protein else None

    def _reverse_complement(self, seq: str) -> str:
        """Reverse complement a DNA sequence."""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        return "".join(complement.get(base, "N") for base in reversed(seq))

    def _check_nmd(self, exons: List[Tuple[int, int]], cds_start: int, cds_end: int, strand: str) -> bool:
        """Check if transcript is subject to NMD (nonsense-mediated decay).
        
        NMD rule: If stop codon is >50nt before the last exon-exon junction,
        transcript is degraded by NMD.
        
        Args:
            exons: List of (start, end) tuples for exons (sorted, absolute coords)
            cds_start: CDS start position (absolute)
            cds_end: CDS end position (stop codon, absolute)
            strand: '+' or '-'
            
        Returns:
            True if transcript triggers NMD
        """
        if len(exons) < 2:
            return False  # Single exon transcripts don't undergo NMD

        # Get last exon-exon junction position
        if strand == "+":
            last_junction = exons[-2][1]  # End of second-to-last exon
            stop_pos = cds_end
            distance = last_junction - stop_pos
        else:
            last_junction = exons[-2][0]  # Start of second-to-last exon
            stop_pos = cds_start
            distance = stop_pos - last_junction

        # NMD if stop codon is >50nt before last junction
        return distance > 50

    def _get_transcripts_in_region(self, chrom: str, start: int, end: int) -> List[Dict[str, Any]]:
        """Get all transcripts overlapping a genomic region with isoform details.
        
        Returns list of isoform dictionaries with:
        - transcript_id
        - gene_name
        - gene_id
        - exons: List of (start, end) relative to region start
        - cds_start, cds_end: CDS coordinates (relative)
        - expression_tpm: Expression level
        - protein_seq: Translated protein sequence
        - has_nmd: Whether transcript triggers NMD
        - is_canonical: Whether this is the main isoform
        """
        # OPTIMIZATION 1: Check isoform cache first
        cache_key = (chrom, start, end)
        if cache_key in self._isoform_cache:
            return self._isoform_cache[cache_key]

        if not self.transcripts or len(self.transcripts) == 0:
            return []

        isoforms = []

        try:
            # Get transcripts overlapping this region
            chrom_features = self.transcripts[self.transcripts.Chromosome == chrom]

            # OPTIMIZATION 5: Pre-load all transcripts for chromosome once
            if not hasattr(self, "_current_chrom") or self._current_chrom != chrom:
                self._current_chrom = chrom
                self._chrom_tx = chrom_features[chrom_features.Feature == "transcript"].df.copy()
                self._chrom_exons = chrom_features[chrom_features.Feature == "exon"].df.copy()
                self._chrom_cds = chrom_features[chrom_features.Feature == "CDS"].df.copy() if "CDS" in chrom_features.Feature.unique() else pd.DataFrame()

                # OPTIMIZATION 2: Pre-compute versionless IDs at load time
                if "transcript_id" in self._chrom_tx.columns:
                    self._chrom_tx["transcript_id"] = self._chrom_tx["transcript_id"].fillna("").astype(str)
                    self._chrom_tx["transcript_id_versionless"] = self._chrom_tx["transcript_id"].str.split(".").str[0]
                if "transcript_id" in self._chrom_exons.columns:
                    self._chrom_exons["transcript_id"] = self._chrom_exons["transcript_id"].fillna("").astype(str)
                    self._chrom_exons["transcript_id_versionless"] = self._chrom_exons["transcript_id"].str.split(".").str[0]
                if not self._chrom_cds.empty and "transcript_id" in self._chrom_cds.columns:
                    self._chrom_cds["transcript_id"] = self._chrom_cds["transcript_id"].fillna("").astype(str)
                    self._chrom_cds["transcript_id_versionless"] = self._chrom_cds["transcript_id"].str.split(".").str[0]

                # CRITICAL: Pre-compute 0-based coordinates ONCE per chromosome, not per window
                self._chrom_tx["Start0"] = self._chrom_tx["Start"] - 1
                self._chrom_tx["End0"] = self._chrom_tx["End"]

                # OPTIMIZATION 9: Pre-group exons and CDS by transcript_id for O(1) lookup
                self._exons_by_tx = {}
                if "transcript_id_versionless" in self._chrom_exons.columns:
                    for tx_id, group in self._chrom_exons.groupby("transcript_id_versionless"):
                        self._exons_by_tx[tx_id] = group

                self._cds_by_tx = {}
                if not self._chrom_cds.empty and "transcript_id_versionless" in self._chrom_cds.columns:
                    for tx_id, group in self._chrom_cds.groupby("transcript_id_versionless"):
                        self._cds_by_tx[tx_id] = group

            transcripts_df = self._chrom_tx
            exons_df = self._chrom_exons
            cds_df = self._chrom_cds

            overlapping_tx = transcripts_df[
                (transcripts_df["End0"] >= start) &
                (transcripts_df["Start0"] < end)
            ]

            if overlapping_tx.empty:
                self._isoform_cache[cache_key] = []
                return []

            # Process each transcript
            for _, tx in overlapping_tx.iterrows():
                try:
                    tx_id = tx.get("transcript_id_versionless")
                    if pd.isna(tx_id) or tx_id == "":
                        tx_id = str(tx.get("transcript_id", "")).split(".")[0]
                    gene_name = tx.get("gene_name", "UNKNOWN")
                    gene_id = tx.get("gene_id", "UNKNOWN").split(".")[0]
                    strand = tx.get("Strand", "+")

                    # Use transcript-level TPM from GTEx (fallback to 0.0 if missing)
                    tpm = float(self.transcript_tpm.get(tx_id, 0.0))

                    # Get exons for this transcript (use pre-grouped dict for O(1) lookup)
                    tx_exons = self._exons_by_tx.get(tx_id, pd.DataFrame())
                    if tx_exons.empty:
                        continue

                    # OPTIMIZATION 10: Vectorize exon coordinate conversion (3-5x faster)
                    # Convert to region-relative coordinates (use 0-based half-open)
                    starts = tx_exons["Start"].values.astype(int) - 1
                    ends = tx_exons["End"].values.astype(int)

                    # Vectorized clipping to window bounds
                    rel_starts = np.maximum(0, starts - start)
                    rel_ends = np.minimum(end - start, ends - start)

                    # Filter valid exons (rel_end > rel_start)
                    valid = rel_ends > rel_starts

                    if not np.any(valid):
                        continue

                    # Extract valid exons and sort
                    exons = list(zip(rel_starts[valid], rel_ends[valid], strict=False))
                    exons_abs = list(zip(starts[valid], ends[valid], strict=False))

                    # Sort by start position
                    exons_sorted = sorted(exons, key=lambda x: x[0])
                    exons_abs_sorted = sorted(exons_abs, key=lambda x: x[0])
                    exons = exons_sorted
                    exons_abs = exons_abs_sorted

                    # Get CDS if available (use pre-grouped dict for O(1) lookup)
                    tx_cds = self._cds_by_tx.get(tx_id, pd.DataFrame())

                    cds_start = None
                    cds_end = None
                    protein_seq = None
                    has_nmd = False

                    if not tx_cds.empty:
                        cds_start_abs0 = int(tx_cds["Start"].min()) - 1
                        cds_end_abs0 = int(tx_cds["End"].max())
                        cds_start = cds_start_abs0 - start
                        cds_end = cds_end_abs0 - start

                        try:
                            import time
                            t_cds_start = time.time()

                            # OPTIMIZATION 3: Cache CDS sequences by (chrom, tx_id)
                            cds_cache_key = (chrom, tx_id)
                            if cds_cache_key in self._cds_seq_cache:
                                cds_seq = self._cds_seq_cache[cds_cache_key]
                                t_fetch = 0.0
                            else:
                                # OPTIMIZATION 4: Batch genome fetches
                                t_fetch_start = time.time()
                                parts = []
                                cds_rows = tx_cds.sort_values("Start") if strand == "+" else tx_cds.sort_values("Start", ascending=False)

                                # Collect all regions to fetch
                                regions_to_fetch = []
                                for i, (_, row_cds) in enumerate(cds_rows.iterrows()):
                                    s0 = int(row_cds["Start"]) - 1
                                    e0 = int(row_cds["End"])
                                    frame = row_cds.get("Frame", 0)
                                    try:
                                        frame = int(frame) if frame != "." else 0
                                    except Exception:
                                        frame = 0
                                    regions_to_fetch.append((s0, e0, frame, i))

                                # Fetch all regions at once
                                for s0, e0, frame, i in regions_to_fetch:
                                    seq = self.genome.fetch(chrom, s0, e0).upper()
                                    if strand == "-":
                                        seq = self._reverse_complement(seq)
                                    if i == 0 and frame > 0 and len(seq) > frame:
                                        seq = seq[frame:]
                                    parts.append(seq)

                                cds_seq = "".join(parts)
                                self._cds_seq_cache[cds_cache_key] = cds_seq
                                t_fetch = time.time() - t_fetch_start

                            t_translate_start = time.time()
                            protein_seq = self._translate_cds(cds_seq, chrom)
                            t_translate = time.time() - t_translate_start

                            t_nmd_start = time.time()
                            if chrom not in ["chrM", "MT", "M"]:
                                has_nmd = self._check_nmd(
                                    exons_abs,
                                    cds_start_abs0,
                                    cds_end_abs0,
                                    strand
                                )
                            else:
                                has_nmd = False
                            t_nmd = time.time() - t_nmd_start

                            # Store timing for analysis
                            self._cds_timings["fetch"].append(t_fetch)
                            self._cds_timings["translate"].append(t_translate)
                            self._cds_timings["nmd"].append(t_nmd)
                        except Exception:
                            pass

                    # Determine if canonical (will be set later when we group by gene)
                    is_canonical = False

                    isoforms.append({
                        "transcript_id": tx_id,
                        "gene_name": gene_name,
                        "gene_id": gene_id,
                        "exons": exons,
                        "cds_start": cds_start,
                        "cds_end": cds_end,
                        "expression_tpm": tpm,
                        "protein_seq": protein_seq,
                        "has_nmd": has_nmd,
                        "is_canonical": is_canonical,
                        "strand": strand
                    })

                except Exception:
                    continue  # Skip problematic transcripts

            # Mark canonical isoforms (highest TPM per gene)
            if isoforms:
                isoforms_by_gene: Dict[str, List[Dict[str, Any]]] = {}
                for iso in isoforms:
                    gene = iso["gene_id"]
                    if gene not in isoforms_by_gene:
                        isoforms_by_gene[gene] = []
                    isoforms_by_gene[gene].append(iso)

                for gene, gene_isoforms in isoforms_by_gene.items():
                    if gene_isoforms:
                        canonical = max(gene_isoforms, key=lambda x: x["expression_tpm"])
                        canonical["is_canonical"] = True

        except Exception as e:
            print(f"  ⚠️  Error getting transcripts for {chrom}:{start}-{end}: {e}")

        # OPTIMIZATION 1: Cache the result
        self._isoform_cache[cache_key] = isoforms
        return isoforms

    def _collect_variants_in_region(self, chrom: str, start: int, end: int) -> List[Dict]:
        """Collect all variants in a region WITHOUT applying them.
        
        Returns list of variant metadata for on-the-fly augmentation.
        Uses indexed lookups for O(1) access instead of O(n) linear scans.
        """
        variants = {}  # Use dict keyed by (pos, ref, alt) for deduplication

        # 1. Add ClinVar variants (iterate only over positions with variants)
        if chrom in self.clinvar_index:
            for pos in self.clinvar_index[chrom]:
                if start <= pos <= end:
                    for var in self.clinvar_index[chrom][pos]:
                        clin_sig = var["info"].get("CLNSIG", "").lower()

                        is_pathogenic = any(x in clin_sig for x in
                            ["pathogenic", "likely_pathogenic", "risk_factor"])
                        is_benign = any(x in clin_sig for x in
                            ["benign", "likely_benign", "protective"])

                        key = (var["pos"] - start, var["ref"], var["alt"])
                        variants[key] = {
                            "pos": var["pos"] - start,
                            "ref": var["ref"],
                            "alt": var["alt"],
                            "source": "clinvar",
                            "is_pathogenic": is_pathogenic,
                            "is_benign": is_benign,
                            "has_splice_effect": False,
                            "splice_effect_score": 0.0,
                            "clinical_significance": clin_sig,
                            "allele_frequency": float(var["info"].get("AF", 0.0)),
                        }

        # 2. Add/update with SpliceVar data (iterate only over positions with variants)
        if chrom in self.splice_variants_index:
            for pos in self.splice_variants_index[chrom]:
                if start <= pos <= end:
                    for var in self.splice_variants_index[chrom][pos]:
                        key = (var["pos"] - start, var["ref"], var["alt"])

                        if key in variants:
                            # Update with splice effect info
                            variants[key]["has_splice_effect"] = True
                            variants[key]["splice_effect_score"] = var["effect_score"]
                            variants[key]["source"] = f"{variants[key]['source']}+splicevar"
                        else:
                            # Add new splice variant
                            variants[key] = {
                                "pos": var["pos"] - start,
                                "ref": var["ref"],
                                "alt": var["alt"],
                                "source": "splicevar",
                                "is_pathogenic": var["effect_score"] > 0.5,
                                "is_benign": False,
                                "has_splice_effect": True,
                                "splice_effect_score": var["effect_score"],
                                "gene": var.get("gene", "UNKNOWN"),
                            }

        # 3. Add 1000 Genomes variants (iterate only over positions with variants)
        if chrom in self.benign_variants_index:
            for pos in self.benign_variants_index[chrom]:
                if start <= pos <= end:
                    for var in self.benign_variants_index[chrom][pos]:
                        key = (var["pos"] - start, var["ref"], var["alt"])

                        if key not in variants:
                            variants[key] = {
                                "pos": var["pos"] - start,
                                "ref": var["ref"],
                                "alt": var["alt"],
                                "source": "1000g",
                                "is_pathogenic": False,
                                "is_benign": True,
                                "has_splice_effect": False,
                                "splice_effect_score": 0.0,
                                "allele_frequency": float(var["info"].get("AF", 0.0)),
                            }

        return list(variants.values())

    def _verify_vcf_files(self) -> None:
        """Verify that VCF files and their tabix indices exist and are valid."""
        print("\nVerifying VCF files and indices...")

        for vcf_path, name in [
            (self.clinvar_vcf_path, "ClinVar"),
            (self.splicevar_vcf_path, "SpliceVar"),
            (self.thousand_genomes_vcf_path, "1000 Genomes")
        ]:
            if not vcf_path:
                print(f"  ⚠️  {name} VCF path not provided")
                continue

            vcf_file = Path(vcf_path)
            tbi_path = Path(f"{vcf_file}.tbi")

            if not vcf_file.exists():
                if name != "1000 Genomes":
                    print(f"  ❌ {name} VCF not found: {vcf_file}")
                continue

            if not tbi_path.exists():
                print(f"  ⚠️  Tabix index not found for {name}: {tbi_path}")
                self._create_tabix_index(vcf_file, name)
                continue

            if tbi_path.stat().st_mtime < vcf_file.stat().st_mtime:
                print(f"  ⚠️  Tabix index is older than VCF for {name}, recreating...")
                self._create_tabix_index(vcf_file, name)
                continue

            try:
                with pysam.TabixFile(str(vcf_file)) as tbx:
                    try:
                        contigs = tbx.contigs
                        if contigs:
                            next(tbx.fetch(reference=contigs[0], start=0, end=1000), None)
                        print(f"  ✓ {name} VCF and index are valid")
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"  ⚠️  Corrupted tabix index for {name} ({e}), recreating...")
                        self._create_tabix_index(vcf_file, name)
            except Exception as e:
                print(f"  ⚠️  Error verifying {name} tabix index: {e}, recreating...")
                self._create_tabix_index(vcf_file, name)

    def _create_tabix_index(self, vcf_path: Path, name: str) -> None:
        """Helper method to create a tabix index for a VCF file."""
        print(f"    Creating tabix index for {name}...")
        try:
            if not str(vcf_path).endswith(".gz"):
                print(f"    Compressing {name} VCF with bgzip...")
                bgzip_path = f"{vcf_path}.gz"
                subprocess.run(["bgzip", "-c", str(vcf_path)],
                             stdout=open(bgzip_path, "wb"),
                             check=True)
                vcf_path = Path(bgzip_path)

            print(f"    Creating tabix index for {vcf_path}...")
            pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
            print("    ✅ Successfully created tabix index")

        except Exception as e:
            print(f"    ❌ Failed to create tabix index: {e}")
            raise

    def _load_chromosome_variants(self, chrom: str) -> None:
        """Load all variant data for a specific chromosome."""
        print(f"\n{'='*80}")
        print(f"LOADING VARIANTS FOR {chrom}")
        print(f"{'='*80}")

        self.clinvar.clear()
        self.splice_variants.clear()
        self.benign_variants.clear()

        print("\n  Loading ClinVar variants...")
        self.clinvar = self._load_clinvar_for_chromosome(chrom)

        print("\n  Loading splice variants...")
        self.splice_variants = self._load_splice_variants_for_chromosome(chrom)

        if self.thousand_genomes_vcf_path and os.path.exists(self.thousand_genomes_vcf_path):
            print("\n  Loading 1000 Genomes variants...")
            self.benign_variants = self._load_thousand_genomes_for_chromosome(chrom)

        print(f"\n✅ Loaded all variants for {chrom}")
        print(f"  ClinVar: {sum(len(v) for v in self.clinvar.values()):,}")
        print(f"  Splice: {sum(len(v) for v in self.splice_variants.values()):,}")
        print(f"  Benign: {sum(len(v) for v in self.benign_variants.values()):,}")

    def _load_clinvar_for_chromosome(self, chrom: str) -> Dict[str, List[Dict]]:
        """Load ClinVar variants for a specific chromosome using tabix."""
        variants: Dict[str, List[Dict]] = {chrom: []}
        index: Dict[int, List[Dict]] = {}  # Position-based index for fast lookup
        count = 0
        skipped = 0

        try:
            tbx = pysam.TabixFile(self.clinvar_vcf_path, encoding="latin-1")

            for record in tbx.fetch(chrom):
                try:
                    fields = record.split("\t")
                    if len(fields) < 8:
                        skipped += 1
                        continue

                    pos = int(fields[1])
                    ref = fields[3]
                    alt = fields[4]

                    # Parse INFO field
                    info = {}
                    for item in fields[7].split(";"):
                        try:
                            if "=" in item:
                                k, v = item.split("=", 1)
                                info[k] = v
                        except Exception:
                            continue

                    # Get clinical significance
                    clin_sig = info.get("CLNSIG", "").lower()
                    if not clin_sig:
                        skipped += 1
                        continue

                    is_benign = any(x in clin_sig for x in ["benign", "likely_benign", "protective"])
                    is_pathogenic = any(x in clin_sig for x in ["pathogenic", "likely_pathogenic", "risk_factor"])

                    if not (is_benign or is_pathogenic):
                        skipped += 1
                        continue

                    var_dict = {
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "info": info,
                        "is_benign": is_benign
                    }

                    variants[chrom].append(var_dict)

                    # Add to position-based index for O(1) lookup
                    if pos not in index:
                        index[pos] = []
                    index[pos].append(var_dict)

                    count += 1
                    if count % 1000 == 0:
                        print(f"\r    Loaded {count:,} variants...", end="", flush=True)

                    if self.max_clinvar_variants and count >= self.max_clinvar_variants:
                        break

                except (ValueError, IndexError, AttributeError):
                    skipped += 1
                    continue

            tbx.close()

        except Exception as e:
            print(f"  ⚠️  Error loading ClinVar for {chrom}: {e}")

        print(f"\r    Loaded {count:,} ClinVar variants for {chrom} (skipped {skipped:,})")

        # Store index for fast lookup
        if chrom not in self.clinvar_index:
            self.clinvar_index[chrom] = index

        return variants

    def _load_splice_variants_for_chromosome(self, chrom: str) -> Dict[str, List[Dict]]:
        """Load splice variants for a specific chromosome using tabix."""
        variants: Dict[str, List[Dict]] = {chrom: []}
        index: Dict[int, List[Dict]] = {}  # Position-based index for fast lookup
        count = 0

        try:
            tbx = pysam.TabixFile(self.splicevar_vcf_path)

            for record in tbx.fetch(chrom):
                fields = record.split("\t")
                if len(fields) < 8:
                    continue

                try:
                    pos = int(fields[1])
                    ref = fields[3]
                    alts = fields[4].split(",")

                    # Parse INFO field
                    info = {}
                    for item in fields[7].split(";"):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            info[k] = v

                    # Extract splice effect
                    effect = info.get("SPLICE_EFFECT", "NONE").upper()
                    if effect == "NONE":
                        effect_score = 0.0
                    elif effect == "MILD":
                        effect_score = 0.5
                    elif effect == "STRONG":
                        effect_score = 1.0
                    else:
                        effect_score = 0.0

                    gene = info.get("GENE", "UNKNOWN")

                    # Add all ALT alleles
                    for alt in alts:
                        var_dict = {
                            "pos": pos,
                            "ref": ref,
                            "alt": alt,
                            "effect": effect,
                            "effect_score": effect_score,
                            "gene": gene,
                            "info": info
                        }
                        variants[chrom].append(var_dict)

                        # Add to position-based index
                        if pos not in index:
                            index[pos] = []
                        index[pos].append(var_dict)

                        count += 1

                        if count % 10000 == 0:
                            print(f"\r    Loaded {count:,} variants...", end="", flush=True)

                except (ValueError, IndexError):
                    continue

            tbx.close()

        except Exception as e:
            print(f"  ⚠️  Error loading splice variants for {chrom}: {e}")

        print(f"\r    Loaded {count:,} splice variants for {chrom}")

        # Store index for fast lookup
        if chrom not in self.splice_variants_index:
            self.splice_variants_index[chrom] = index

        return variants

    def _load_thousand_genomes_for_chromosome(self, chrom: str) -> Dict[str, List[Dict]]:
        """Load 1000 Genomes variants for a specific chromosome using tabix."""
        variants: Dict[str, List[Dict]] = {chrom: []}
        index: Dict[int, List[Dict]] = {}  # Position-based index for fast lookup
        count = 0
        skipped = 0
        filtered = 0

        if not self.thousand_genomes_vcf_path or not os.path.exists(self.thousand_genomes_vcf_path):
            return variants

        try:
            tbx = pysam.TabixFile(self.thousand_genomes_vcf_path)

            if chrom not in tbx.contigs:
                print(f"  ⚠️  Chromosome {chrom} not found in VCF")
                tbx.close()
                return variants

            for record in tbx.fetch(chrom):
                fields = record.split("\t")
                if len(fields) < 8:
                    skipped += 1
                    continue

                try:
                    pos = int(fields[1])
                    ref = fields[3]
                    alts = fields[4].split(",")

                    if not alts or alts[0] == ".":
                        skipped += 1
                        continue

                    # Parse INFO field
                    info = {}
                    for item in fields[7].split(";"):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            info[k] = v

                    # Get allele frequency
                    af = 0.0
                    for pop in ["AFR", "AMR", "EAS", "EUR", "SAS", "MAF", "AF"]:
                        if pop in info:
                            try:
                                pop_afs = [float(x) for x in info[pop].split(",") if x]
                                if pop_afs:
                                    af = max(af, max(pop_afs))
                            except (ValueError, TypeError):
                                continue

                    # Only include variants with 1% < AF < 99%
                    if 0.01 <= af <= 0.99:
                        var_dict = {
                            "pos": pos,
                            "ref": ref,
                            "alt": alts[0],
                            "info": {"AF": af, **info},
                            "is_benign": True
                        }
                        variants[chrom].append(var_dict)

                        # Add to position-based index
                        if pos not in index:
                            index[pos] = []
                        index[pos].append(var_dict)

                        count += 1
                    else:
                        filtered += 1

                    if count % 10000 == 0:
                        print(f"\r    Loaded {count:,} variants (filtered {filtered:,})...", end="", flush=True)

                    if self.max_benign_variants and count >= self.max_benign_variants:
                        break

                except (ValueError, IndexError):
                    skipped += 1
                    continue

            tbx.close()

        except Exception as e:
            print(f"  ❌ Error loading 1000 Genomes for {chrom}: {e}")

        print(f"\r    Loaded {count:,} benign variants for {chrom} (skipped {skipped:,}, filtered {filtered:,})")

        # Store index for fast lookup
        if chrom not in self.benign_variants_index:
            self.benign_variants_index[chrom] = index
        return variants

    def _create_reference_example(self, region: Dict) -> Optional[Dict[str, Any]]:
        """Create a reference example with all metadata.
        
        This does NOT apply variants - it only stores them for later use.
        """
        try:
            import time
            t0 = time.time()

            chrom = region["Chromosome"]
            start = region["Start"]
            end = region["End"]

            # 1. Get reference sequence (with caching)
            t1 = time.time()
            seq_cache_key = (chrom, start, end)
            if seq_cache_key in self._seq_cache:
                seq = self._seq_cache[seq_cache_key]
            else:
                seq = self.genome.fetch(chrom, start, end).upper()
                self._seq_cache[seq_cache_key] = seq

            if not seq:
                return None
            t_seq = time.time() - t1

            # 2. Create site labels from GENCODE
            t2 = time.time()
            labels = self._create_labels(region, start, len(seq))
            if not labels:
                return None
            t_labels = time.time() - t2

            # 3. Get isoform annotations from GENCODE + GTEx
            t3 = time.time()
            isoforms = self._get_transcripts_in_region(chrom, start, end)
            t_isoforms = time.time() - t3

            # 4. Collect variant metadata (with caching)
            t4 = time.time()
            var_cache_key = (chrom, start, end)
            if var_cache_key in self._variant_cache:
                variants = self._variant_cache[var_cache_key]
            else:
                variants = self._collect_variants_in_region(chrom, start, end)
                self._variant_cache[var_cache_key] = variants
            t_variants = time.time() - t4

            # Print timing every 50 windows
            if not hasattr(self, "_window_count"):
                self._window_count = 0
            self._window_count += 1
            if self._window_count % 50 == 0:
                t_total = time.time() - t0
                t_json = t_total - (t_seq+t_labels+t_isoforms+t_variants)
                print(f"\n  ⏱️  W{self._window_count}: seq={t_seq*1000:.0f}ms, labels={t_labels*1000:.0f}ms, iso={t_isoforms*1000:.0f}ms, var={t_variants*1000:.0f}ms, json={t_json*1000:.0f}ms, total={t_total*1000:.0f}ms", flush=True)

                # Print CDS timing breakdown every 50 windows
                if hasattr(self, "_cds_timings") and self._cds_timings["fetch"]:
                    import numpy as np
                    fetch_times = np.array(self._cds_timings["fetch"])
                    translate_times = np.array(self._cds_timings["translate"])
                    nmd_times = np.array(self._cds_timings["nmd"])

                    print(f"      CDS breakdown: fetch={fetch_times.mean()*1000:.1f}ms, translate={translate_times.mean()*1000:.1f}ms, nmd={nmd_times.mean()*1000:.1f}ms", flush=True)

            # 5. Return everything as-is
            example = {
                "seq": seq,  # Reference sequence (unmodified)
                "chrom": chrom,
                "start": start,
                "end": end,
                "transcript_id": region.get("transcript_id", ""),
                "gene_name": region.get("gene_name", ""),
                "strand": region.get("Strand", "+"),
                "donor": labels.get("donor", []),
                "acceptor": labels.get("acceptor", []),
                "tss": labels.get("tss", []),
                "polya": labels.get("polya", []),
                "isoforms": isoforms,
                "variants": variants  # Metadata only, not applied
            }

            # Convert to JSON strings for Parquet
            for key in ["donor", "acceptor", "tss", "polya", "isoforms", "variants"]:
                if example[key] is not None:
                    if hasattr(example[key], "tolist"):
                        example[key] = example[key].tolist()
                    example[key] = json.dumps(example[key], default=str)

            return example

        except Exception as e:
            print(f"\n❌ Error creating example: {e}")
            return None

    def _create_labels(self, region: Dict, window_start: int, seq_len: int) -> Dict:
        """Create ground truth labels for a genomic region."""
        labels = {
            "donor": np.zeros(seq_len, dtype=np.float32),
            "acceptor": np.zeros(seq_len, dtype=np.float32),
            "tss": np.zeros(seq_len, dtype=np.float32),
            "polya": np.zeros(seq_len, dtype=np.float32)
        }

        if not hasattr(self, "transcripts") or self.transcripts is None:
            return labels

        chrom = region["Chromosome"]
        window_end = window_start + seq_len

        try:
            if chrom not in self._chrom_transcripts:
                chrom_features = self.transcripts[self.transcripts.Chromosome == chrom]
                transcripts = chrom_features[chrom_features.Feature == "transcript"]
                exons = chrom_features[chrom_features.Feature == "exon"]

                # Pre-group transcripts by ID for O(1) lookup
                transcripts_by_id = {}
                if "transcript_id" in transcripts.df.columns:
                    for tx_id, group in transcripts.df.groupby("transcript_id"):
                        transcripts_by_id[tx_id] = group.iloc[0]  # Get first row (should be unique)

                self._chrom_transcripts[chrom] = {
                    "exons": exons,
                    "transcripts": transcripts,
                    "transcripts_by_id": transcripts_by_id
                }

            chrom_data = self._chrom_transcripts[chrom]
            exons = chrom_data["exons"]

            # Filter exons for this window only
            overlapping_exons = exons[
                (exons.End >= window_start) &
                (exons.Start < window_end)
            ]

            if overlapping_exons.empty:
                return labels

            exons_df = overlapping_exons.df

            if "transcript_id" not in exons_df.columns:
                return labels

            tx_ids = exons_df["transcript_id"].values
            starts = exons_df["Start"].values
            ends = exons_df["End"].values
            unique_tx_ids = np.unique(tx_ids)
            transcripts_df = chrom_data["transcripts"].df

            for tx_id in unique_tx_ids:
                mask = tx_ids == tx_id
                tx_exon_starts = starts[mask]
                tx_exon_ends = ends[mask]

                if len(tx_exon_starts) < 2:
                    continue

                sort_idx = np.argsort(tx_exon_starts)
                tx_exon_starts = tx_exon_starts[sort_idx]
                tx_exon_ends = tx_exon_ends[sort_idx]

                # Use pre-grouped dict for O(1) lookup instead of boolean indexing
                if tx_id not in chrom_data["transcripts_by_id"]:
                    continue
                tx_info = chrom_data["transcripts_by_id"][tx_id]

                tx_start = tx_info["Start"]
                tx_end = tx_info["End"]
                strand = tx_info["Strand"] if "Strand" in tx_info.index else "+"

                # Label TSS and polyA sites
                tss_pos = tx_start - window_start if strand == "+" else tx_end - window_start
                if 0 <= tss_pos < seq_len:
                    labels["tss"][tss_pos] = 1.0

                polya_pos = tx_end - window_start if strand == "+" else tx_start - window_start
                if 0 <= polya_pos < seq_len:
                    labels["polya"][polya_pos] = 1.0

                # Label splice sites
                for i in range(len(tx_exon_ends) - 1):
                    if strand == "+":
                        donor_pos = tx_exon_ends[i] - window_start - 1
                        acceptor_pos = tx_exon_starts[i+1] - window_start
                    else:
                        donor_pos = tx_exon_starts[i] - window_start
                        acceptor_pos = tx_exon_ends[i+1] - window_start - 1

                    if 0 <= donor_pos < seq_len:
                        labels["donor"][donor_pos] = 1.0

                    if 0 <= acceptor_pos < seq_len:
                        labels["acceptor"][acceptor_pos] = 1.0

        except Exception as e:
            print(f"  ⚠️  Error creating labels: {e}")

        return labels

    def _load_gencode(self, gtf_path: str) -> pr.PyRanges:
        """Load Gencode GTF with proper filtering."""
        print(f"  Loading GTF from {gtf_path}...")

        try:
            # First try with pyranges
            try:
                gtf = pr.read_gtf(gtf_path)
                print(f"    Loaded {len(gtf):,} total features with pyranges")
            except Exception as e:
                print(f"    ⚠️  pyranges failed, falling back to pandas: {e}")
                # Fall back to pandas if pyranges fails
                gtf_columns = ["Chromosome", "Source", "Feature", "Start", "End",
                             "Score", "Strand", "Frame", "Attributes"]
                gtf_df = pd.read_csv(
                    gtf_path,
                    sep="\t",
                    comment="#",
                    header=None,
                    names=gtf_columns,
                    low_memory=False,
                    dtype={"Attributes": str}
                )
                gtf = pr.PyRanges(gtf_df)
                print(f"    Loaded {len(gtf):,} total features with pandas fallback")

            valid_features = ["transcript", "exon", "gene", "CDS"]
            gtf_filtered = gtf[gtf.Feature.isin(valid_features)]

            gtf_df = gtf.df
            if "transcript_type" in gtf_df.columns:
                original_count = len(gtf_filtered)
                gtf_filtered = gtf_filtered[gtf_filtered.transcript_type == "protein_coding"]
                print(f"    Filtered to {len(gtf_filtered):,} protein-coding features")

            print(f"    ✅ Final GTF contains {len(gtf_filtered):,} features")

            # One-time: parse Attributes and coerce numeric types
            def _parse_attr(s: str) -> Dict[str, str]:
                out = {}
                for item in str(s).split(";"):
                    item = item.strip()
                    if not item:
                        continue
                    if ' "' in item:
                        k, v = item.split(' "', 1)
                        v = v.rstrip('"')
                    elif "=" in item:
                        k, v = item.split("=", 1)
                    else:
                        parts = item.split(" ", 1)
                        if len(parts) == 2:
                            k, v = parts[0], parts[1].strip('"')
                        else:
                            continue
                    out[k.strip()] = v.strip()
                return out

            df = gtf_filtered.df.copy()
            if "transcript_id" not in df.columns and "Attributes" in df.columns:
                attrs = df["Attributes"].apply(_parse_attr)
                df["transcript_id"] = attrs.apply(lambda d: d.get("transcript_id", ""))
                df["gene_id"] = attrs.apply(lambda d: d.get("gene_id", ""))
                df["gene_name"] = attrs.apply(lambda d: d.get("gene_name", ""))
            if "transcript_id" in df.columns:
                df["transcript_id"] = df["transcript_id"].fillna("").astype(str)
            for col in ["Start","End"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Start","End"]).astype({"Start": int, "End": int})
            gtf_filtered = pr.PyRanges(df)
            return gtf_filtered

        except Exception as e:
            print(f"    ❌ Error loading GTF: {e}")
            raise

    def _load_gtex_junctions(self, junctions_path: str) -> Dict[str, List[Dict]]:
        """Load GTEx junction data from parquet file."""
        print(f"  Loading GTEx junctions from {junctions_path}")
        try:
            df = pd.read_parquet(junctions_path)
            print(f"  Available columns: {df.columns.tolist()}")

            junctions: Dict[str, List[Dict]] = {}
            for _, row in df.iterrows():
                chrom = row["chrom"]
                if chrom not in junctions:
                    junctions[chrom] = []

                junction_data = {
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "strand": row.get("strand", "+"),
                    "num_samples": int(row["num_samples"])
                }

                # Add total_reads if it exists
                if "total_reads" in df.columns:
                    junction_data["total_reads"] = int(row["total_reads"])

                junctions[chrom].append(junction_data)

            print(f"  Loaded {len(df):,} junctions for {len(junctions)} chromosomes")
            return junctions
        except Exception as e:
            print(f"  ⚠️  Error loading junctions: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_gtex_expression(self, expression_path: str) -> Dict[str, Dict[str, float]]:
        """Load GTEx expression data from parquet file."""
        print(f"  Loading GTEx expression data from {expression_path}")
        try:
            df = pd.read_parquet(expression_path)

            expression_data = {}
            for _, row in df.iterrows():
                expression_data[row["gene_id"]] = {
                    "mean_tpm": row["mean_tpm"],
                    "median_tpm": row["median_tpm"],
                    "max_tpm": row["max_tpm"],
                    "tissue_count": row["tissue_count"]
                }

            print(f"  Loaded expression data for {len(expression_data):,} genes")
            return expression_data
        except Exception as e:
            print(f"  ⚠️  Error loading expression: {e}")
            return {}

    def generate_datasets(
        self,
        train_chromosomes: List[str],
        val_chromosomes: List[str],
        test_chromosomes: List[str],
        batch_size: int = 100,
    ) -> None:
        """Generate train/val/test datasets using sliding windows.
        
        Args:
            train_chromosomes: Chromosomes for training
            val_chromosomes: Chromosomes for validation
            test_chromosomes: Chromosomes for testing
            batch_size: Number of examples per parquet file
        """
        print("\n" + "="*80)
        print("GENERATING DATASETS WITH SLIDING WINDOWS")
        print("="*80)
        print(f"\nTrain chromosomes: {', '.join(train_chromosomes)}")
        print(f"Val chromosomes: {', '.join(val_chromosomes)}")
        print(f"Test chromosomes: {', '.join(test_chromosomes)}")
        print(f"Window size: {self.window_size:,} bp")
        print(f"Step size: {self.step_size:,} bp")
        print(f"Batch size: {batch_size}")

        # Generate each split
        print("\n" + "="*80)
        print("PROCESSING TRAIN SPLIT")
        print("="*80)
        self._generate_sliding_windows(train_chromosomes, "train", batch_size)

        print("\n" + "="*80)
        print("PROCESSING VALIDATION SPLIT")
        print("="*80)
        self._generate_sliding_windows(val_chromosomes, "val", batch_size)

        print("\n" + "="*80)
        print("PROCESSING TEST SPLIT")
        print("="*80)
        self._generate_sliding_windows(test_chromosomes, "test", batch_size)

        # Print final summary
        self.print_summary()

    def _generate_windows_for_chromosome(self, chrom: str) -> List[GenomicWindow]:
        """Generate genomic windows for a single chromosome."""
        try:
            chrom_length = self.genome.get_reference_length(chrom)
            if chrom_length == 0:
                print(f"  ⚠️  Skipping {chrom} (length = 0)")
                return []

            windows: List[GenomicWindow] = []
            for start in range(0, chrom_length - self.window_size + 1, self.step_size):
                end = start + self.window_size
                windows.append(GenomicWindow(
                    chrom=chrom,
                    start=start,
                    end=end,
                    transcript_id=f"{chrom}:{start}-{end}",
                    gene_name=f"REGION_{len(windows)}",
                    strand="+"
                ))

            return windows
        except Exception as e:
            print(f"  ❌ Error generating windows for {chrom}: {e}")
            return []

    def _save_batch(self, examples: List[Dict], split_name: str, chrom: str, batch_idx: int) -> None:
        """Save a batch of examples to a Parquet file."""
        try:
            if not examples:
                return

            output_dir = Path(self.output_dir or "data/processed") / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{chrom}_batch_{batch_idx:04d}.parquet"

            df = pd.DataFrame(examples)
            df.to_parquet(
                output_file,
                engine="pyarrow",
                compression="snappy",
                index=False
            )

            if output_file.exists():
                print(f"  ✅ Saved {len(examples):,} examples to {output_file.name} ({output_file.stat().st_size / (1024*1024):.2f} MB)")

        except Exception as e:
            print(f"  ❌ Error saving batch {batch_idx}: {e}")

    def _process_window_batch(
        self,
        batch: List[GenomicWindow],
        split_name: str,
        chrom: str,
        batch_idx: int
    ) -> int:
        """Process a batch of windows and save to disk."""
        examples = []
        global processed_counter, last_update_time

        for window in batch:
            try:
                window_data = {
                    "Chromosome": window.chrom,
                    "Start": window.start,
                    "End": window.end,
                    "transcript_id": window.transcript_id,
                    "gene_name": window.gene_name,
                    "Strand": window.strand,
                    "Feature": "region",
                    "gene_id": f"REGION_{window.chrom}_{window.start}"
                }

                example = self._create_reference_example(window_data)

                if example:
                    examples.append(example)
                    processed_counter += 1

                    current_time = time.time()
                    if processed_counter % 10 == 0 or current_time - last_update_time > 1:
                        print(f"\r  Processed {processed_counter:,} windows...", end="", flush=True)
                        last_update_time = current_time

            except Exception:
                continue

        if examples:
            self._save_batch(examples, split_name, chrom, batch_idx)

        return len(examples)

    def _generate_sliding_windows(
        self,
        chromosomes: List[str],
        split_name: str,
        batch_size: int
    ) -> None:
        """Generate examples using sliding windows across chromosomes."""
        print(f"\n{'='*80}")
        print(f"GENERATING {split_name.upper()} SPLIT")
        print("="*80)

        global processed_counter, last_update_time
        processed_counter = 0
        last_update_time = time.time()
        start_time = time.time()

        output_dir = Path(self.output_dir or "data/processed") / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing files
        for f in output_dir.glob("*.parquet"):
            f.unlink()

        # Clear all caches at start of split to free memory
        self._isoform_cache.clear()
        self._cds_seq_cache.clear()
        self._seq_cache.clear()
        self._variant_cache.clear()

        total_examples = 0

        for chrom in chromosomes:
            try:
                print(f"\n{'='*80}")
                print(f"PROCESSING CHROMOSOME: {chrom}")
                print(f"{'='*80}")

                # Load variants for this chromosome
                self._load_chromosome_variants(chrom)

                # Generate windows
                print(f"\nGenerating windows for {chrom}...")
                windows = self._generate_windows_for_chromosome(chrom)
                if not windows:
                    continue

                print(f"  Generated {len(windows):,} windows for {chrom}")

                # Process windows in batches
                batch_idx = 0
                for i in range(0, len(windows), batch_size):
                    batch = windows[i:i + batch_size]
                    batch_examples = self._process_window_batch(
                        batch, split_name, chrom, batch_idx
                    )
                    total_examples += batch_examples
                    batch_idx += 1

                print(f"\n✅ Completed {chrom}: {batch_idx} batches, {total_examples:,} total examples")

                # Print CDS timing analysis
                if self._cds_timings["fetch"]:
                    import numpy as np
                    fetch_times = np.array(self._cds_timings["fetch"])
                    translate_times = np.array(self._cds_timings["translate"])
                    nmd_times = np.array(self._cds_timings["nmd"])

                    print(f"\n  CDS Processing Breakdown ({len(fetch_times)} transcripts with CDS):")
                    print(f"    Fetch:     {fetch_times.mean()*1000:.1f}ms avg (total: {fetch_times.sum()*1000:.0f}ms)")
                    print(f"    Translate: {translate_times.mean()*1000:.1f}ms avg (total: {translate_times.sum()*1000:.0f}ms)")
                    print(f"    NMD:       {nmd_times.mean()*1000:.1f}ms avg (total: {nmd_times.sum()*1000:.0f}ms)")
                else:
                    print("\n  ⚠️  No CDS timings collected (no transcripts with CDS?)")

                # Clear for next chromosome
                self._cds_timings = {"fetch": [], "translate": [], "nmd": []}

                # Clear variant data
                self.clinvar.clear()
                self.splice_variants.clear()
                self.benign_variants.clear()

            except Exception as e:
                print(f"  ❌ Error processing {chrom}: {e}")
                continue

        total_time = time.time() - start_time
        rate = total_examples / (total_time + 1e-6)
        print(f"\n✅ {split_name.upper()} split complete:")
        print(f"  - Examples: {total_examples:,}")
        print(f"  - Time: {total_time/60:.1f} minutes")
        print(f"  - Rate: {rate:.1f} examples/second")

    def print_summary(self) -> None:
        """Print summary of generated datasets."""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)

        total_examples: int = 0
        total_size_mb: float = 0

        for split in ["train", "val", "test"]:
            files = sorted((Path(self.output_dir or "data/processed") / split).glob("*.parquet"))

            if not files:
                print(f"\n{split.upper()}: ❌ No files found")
                continue

            split_examples: int = 0
            split_size_mb: float = 0

            for f in files:
                try:
                    file_size_mb = os.path.getsize(f) / (1024 * 1024)
                    split_size_mb += file_size_mb

                    with open(f, "rb") as pf:
                        parquet_file = pq.ParquetFile(pf)
                        num_rows = parquet_file.metadata.num_rows
                        split_examples += num_rows

                except Exception as e:
                    print(f"  ⚠️  Error reading {f.name}: {e}")

            total_examples += split_examples
            total_size_mb += split_size_mb

            print(f"\n{split.upper()}:")
            print(f"  Files: {len(files):,}")
            print(f"  Examples: {split_examples:,}")
            print(f"  Size: {split_size_mb:.1f} MB")

        print("\n" + "-"*50)
        print(f"TOTAL EXAMPLES: {total_examples:,}")
        print(f"TOTAL SIZE: {total_size_mb:.1f} MB")
        print("="*80)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    for section in ["data", "output", "training"]:
        if section not in config:
            config[section] = {}

    return config


def resolve_paths(config: Dict[str, Any]) -> None:
    """Resolve all paths in the configuration to absolute paths."""
    project_root = Path(__file__).parent.parent

    data = config["data"]
    for key in ["reference_fasta", "gencode_gtf", "clinvar_vcf",
               "splicevar_vcf", "thousand_genomes_vcf", "gtex_junctions",
               "gtex_expression"]:
        if data.get(key):
            data[key] = str(project_root / data[key])

    output = config["output"]
    for key in ["output_dir", "checkpoint_dir", "log_dir"]:
        if output.get(key):
            output[key] = str(project_root / output[key])
            Path(output[key]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Main function to generate training data."""
    try:
        config = load_config()
        resolve_paths(config)

        data_cfg = config["data"]
        train_cfg = config["training"]

        logger.info("Starting data preparation")

        # Initialize data generator
        generator = TrainingDataGenerator(
            reference_fasta=data_cfg["reference_fasta"],
            gencode_gtf=data_cfg["gencode_gtf"],
            clinvar_vcf=data_cfg["clinvar_vcf"],
            splicevar_vcf=data_cfg["splicevar_vcf"],
            gtex_tpm=data_cfg["gtex_expression"],  # Using this for TPM
            thousand_genomes_vcf=data_cfg.get("thousand_genomes_vcf"),
            gtex_junctions=data_cfg.get("gtex_junctions"),
            gtex_expression=data_cfg.get("gtex_expression"),
            output_dir=data_cfg["data_dir"],
            window_size=data_cfg.get("window_size", 300000),
            step_size=data_cfg.get("step_size", 150000),
            max_clinvar_variants=data_cfg.get("max_clinvar_variants"),
            max_benign_variants=data_cfg.get("max_benign_variants"),
            gtex_sample_min=int(data_cfg.get("gtex_sample_min", 50)),
        )

        # Generate datasets
        generator.generate_datasets(
            train_chromosomes=train_cfg.get("train_chromosomes", [f"chr{i}" for i in range(1, 20)]),
            val_chromosomes=train_cfg.get("val_chromosomes", ["chr20", "chr21", "chr22"]),
            test_chromosomes=train_cfg.get("test_chromosomes", ["chrX", "chrY", "chrM"]),
            batch_size=data_cfg.get("batch_size", 100),
        )

        print("\n" + "="*80)
        print("DATASET GENERATION COMPLETE!")
        print("="*80)

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Error in data preparation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
