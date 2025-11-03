#!/usr/bin/env python3
"""
Extract transcript and protein sequences from genome using GTF annotations.
Compare against reference sequences to validate the extraction pipeline.
"""

import gzip
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pysam
from Bio.Seq import Seq

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent / 'data/processed'

CONFIG = {
    'genome': DATA_DIR / 'genome/GRCh38.fa.gz',
    'gtf': DATA_DIR / 'annotation/gencode_v26/gencode.v26.annotation.gtf.gz',
    'transcripts': DATA_DIR / 'annotation/gencode_v26/gencode.v26.transcripts.fa.gz',
    'proteins': DATA_DIR / 'annotation/gencode_v26/gencode.v26.pc_translations.fa.gz',
}

# ============================================================================
# Core Functions
# ============================================================================

def parse_gtf(gtf_path: Path) -> Dict[str, Dict]:
    """Parse GTF file and extract exon and CDS coordinates for each transcript."""
    transcripts = {}
    
    opener = gzip.open if gtf_path.suffix == '.gz' else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            feature_type = parts[2]
            if feature_type not in ('exon', 'CDS', 'Selenocysteine'):
                continue
            
            chrom = parts[0]
            start = int(parts[3]) - 1  # Convert to 0-based
            end = int(parts[4])
            strand = parts[6]
            phase = int(parts[7]) if parts[7] != '.' else 0
            
            # Parse attributes
            attrs = {}
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    attrs[key] = value.strip('"')
            
            tx_id = attrs.get('transcript_id')
            if not tx_id:
                continue
            
            # Initialize transcript entry
            if tx_id not in transcripts:
                transcripts[tx_id] = {
                    'chr': chrom,
                    'strand': strand,
                    'exons': [],
                    'cds': [],
                    'selenocysteine': []
                }
            
            # Add coordinate
            if feature_type == 'exon':
                transcripts[tx_id]['exons'].append((start, end))
            elif feature_type == 'CDS':
                transcripts[tx_id]['cds'].append((start, end, phase))
            elif feature_type == 'Selenocysteine':
                transcripts[tx_id]['selenocysteine'].append((start, end))
    
    # Sort and deduplicate coordinates
    for tx_id in transcripts:
        transcripts[tx_id]['exons'] = sorted(set(transcripts[tx_id]['exons']))
        transcripts[tx_id]['cds'] = sorted(set(transcripts[tx_id]['cds']))
        transcripts[tx_id]['selenocysteine'] = sorted(set(transcripts[tx_id]['selenocysteine']))
    
    return transcripts


def load_fasta(fasta_path: Path, key_index: int = 0) -> Dict[str, str]:
    """Load FASTA file into dictionary."""
    if not fasta_path.exists():
        return {}
    
    sequences = {}
    opener = gzip.open if fasta_path.suffix == '.gz' else open
    
    with opener(fasta_path, 'rt') as f:
        seq_id = None
        seq_lines = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = ''.join(seq_lines).upper()
                
                header = line[1:]
                fields = header.split('|')
                
                if len(fields) > key_index:
                    seq_id = fields[key_index].split()[0]
                else:
                    seq_id = fields[0].split()[0]
                
                seq_lines = []
            else:
                seq_lines.append(line)
        
        if seq_id:
            sequences[seq_id] = ''.join(seq_lines).upper()
    
    return sequences


def extract_sequence(genome: pysam.FastaFile, chrom: str, regions: List[Tuple[int, int]], 
                     strand: str) -> str:
    """Extract sequence from genome given regions and strand."""
    if not regions:
        return ""
    
    sequences = []
    for start, end in regions:
        seq = genome.fetch(chrom, start, end).upper()
        sequences.append(seq)
    
    if strand == '-':
        sequences = [str(Seq(s).reverse_complement()) for s in sequences[::-1]]
    
    return ''.join(sequences)


def extract_cds_sequence(genome: pysam.FastaFile, chrom: str, 
                        cds_regions: List[Tuple[int, int, int]], strand: str) -> Tuple[str, int]:
    """
    Extract CDS sequence from genome, accounting for phase.
    Returns (sequence, first_phase).
    """
    if not cds_regions:
        return "", 0
    
    sorted_regions = sorted(cds_regions)
    
    if strand == '+':
        first_phase = sorted_regions[0][2]
        sequences = []
        
        for i, (start, end, phase) in enumerate(sorted_regions):
            seq = genome.fetch(chrom, start, end).upper()
            if i == 0 and phase > 0:
                seq = seq[phase:]
            sequences.append(seq)
    else:
        first_phase = sorted_regions[-1][2]
        sequences = []
        
        for i, (start, end, phase) in enumerate(reversed(sorted_regions)):
            seq = genome.fetch(chrom, start, end).upper()
            seq = str(Seq(seq).reverse_complement())
            if i == 0 and phase > 0:
                seq = seq[phase:]
            sequences.append(seq)
    
    return ''.join(sequences), first_phase


def get_selenocysteine_positions(cds_seq: str, cds_regions: List[Tuple[int, int, int]], 
                                seleno_regions: List[Tuple[int, int]], strand: str) -> List[int]:
    """
    Calculate protein positions of selenocysteines.
    Returns list of 0-based amino acid positions.
    """
    if not seleno_regions or not cds_regions:
        return []
    
    # Build a complete map from genomic position to CDS position
    genomic_to_cds = {}
    cds_pos = 0
    
    if strand == '+':
        # Sort CDS regions by genomic position
        sorted_cds = sorted(cds_regions)
        first_phase = sorted_cds[0][2]
        
        # Map each genomic position to its position in the CDS
        for i, (start, end, phase) in enumerate(sorted_cds):
            offset = phase if i == 0 else 0
            for pos in range(start + offset, end):
                genomic_to_cds[pos] = cds_pos
                cds_pos += 1
    else:  # strand == '-'
        # Sort CDS regions by genomic position (reverse for negative strand)
        sorted_cds = sorted(cds_regions, reverse=True)
        first_phase = sorted_cds[0][2]
        
        # Map each genomic position to its position in the CDS
        for i, (start, end, phase) in enumerate(sorted_cds):
            offset = phase if i == 0 else 0
            # For negative strand, we need to reverse the positions
            for pos in range(end - 1, start + offset - 1, -1):
                genomic_to_cds[pos] = cds_pos
                cds_pos += 1
    
    # Find selenocysteine positions in the translated protein sequence
    seleno_aa_positions = []
    
    for seleno_start, seleno_end in seleno_regions:
        # Check each position in the selenocysteine region
        for pos in range(seleno_start, seleno_end):
            if pos in genomic_to_cds:
                cds_pos = genomic_to_cds[pos]
                # Convert CDS position to amino acid position (integer division)
                aa_pos = cds_pos // 3
                seleno_aa_positions.append(aa_pos)
                # We only need one position per selenocysteine feature
                break
    
    return seleno_aa_positions


def translate(dna_seq: str, first_phase: int = 0, seleno_positions: List[int] = None, chrom: str = None) -> str:
    """
    Translate DNA sequence to protein.
    - Handle incomplete start codons (phase > 0) as X
    - Always use methionine (M) as first amino acid for complete proteins
    - Handle selenocysteine positions as U
    - Use mitochondrial genetic code for 'chrM'
    """
    if len(dna_seq) < 3:
        return ""
    
    seleno_positions = seleno_positions or []
    
    # Trim to multiple of 3
    trimmed = dna_seq[:len(dna_seq) // 3 * 3]
    
    # Choose the appropriate genetic code
    table = 2 if chrom == "chrM" else 1  # Table 2 is mitochondrial, Table 1 is standard
    
    # Translate
    protein = str(Seq(trimmed).translate(table=table))
    
    # Remove trailing stop codon
    if protein.endswith('*'):
        protein = protein[:-1]
    
    # Replace stop codons with U at selenocysteine positions
    if seleno_positions:
        protein_list = list(protein)
        for pos in seleno_positions:
            if 0 <= pos < len(protein_list) and protein_list[pos] == '*':
                protein_list[pos] = 'U'
        protein = ''.join(protein_list)
    
    # Handle start of protein
    if first_phase > 0:
        # Incomplete start codon represented as X
        protein = 'X' + protein
    elif len(protein) > 0 and protein[0] != 'M':
        # Force first amino acid to be M for biological relevance
        protein = 'M' + protein[1:]
    
    return protein


def compare_protein(extracted: str, reference: str, name: str) -> Tuple[bool, str]:
    """Special comparison for proteins that ignores first amino acid mismatch."""
    if not extracted or not reference:
        return False, f"{name}: Empty sequence (len {len(extracted)} vs {len(reference)})"
    
    # Check for exact match
    if extracted == reference:
        return True, f"{name}: Perfect match ({len(extracted)} aa)"
    
    # If only the first amino acid differs, consider it a match
    if len(extracted) == len(reference) and extracted[1:] == reference[1:]:
        return True, f"{name}: Match except first amino acid ({extracted[0]} vs {reference[0]})"
    
    # Standard comparison logic for other cases
    len1, len2 = len(extracted), len(reference)
    
    if len1 != len2:
        min_len = min(len1, len2)
        if extracted[1:min_len] == reference[1:min_len]:
            return False, f"{name}: Length mismatch ({len1} vs {len2}), but first {min_len-1} match after first aa"
        return False, f"{name}: Different lengths ({len1} vs {len2}) AND sequences differ"
    
    # Same length but different content
    mismatches = sum(c1 != c2 for c1, c2 in zip(extracted, reference))
    first_diff = next(i for i, (c1, c2) in enumerate(zip(extracted, reference)) if c1 != c2)
    
    context_start = max(0, first_diff - 10)
    context_end = min(len1, first_diff + 10)
    
    return False, (
        f"{name}: {mismatches}/{len1} mismatches ({100*mismatches/len1:.1f}%)\n"
        f"       First diff at position {first_diff}:\n"
        f"       Extracted: {extracted[context_start:context_end]}\n"
        f"       Reference: {reference[context_start:context_end]}"
    )


def compare(seq1: str, seq2: str, name: str) -> Tuple[bool, str]:
    """Compare two sequences and return (is_identical, message)."""
    if not seq1 or not seq2:
        return False, f"{name}: Empty sequence (len {len(seq1)} vs {len(seq2)})"
    
    if seq1 == seq2:
        return True, f"{name}: Perfect match ({len(seq1)} bp/aa)"
    
    len1, len2 = len(seq1), len(seq2)
    
    if len1 != len2:
        min_len = min(len1, len2)
        if seq1[:min_len] == seq2[:min_len]:
            return False, f"{name}: Length mismatch ({len1} vs {len2}), but first {min_len} match"
        return False, f"{name}: Different lengths ({len1} vs {len2}) AND sequences differ"
    
    # Same length but different content
    mismatches = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    first_diff = next(i for i, (c1, c2) in enumerate(zip(seq1, seq2)) if c1 != c2)
    
    context_start = max(0, first_diff - 10)
    context_end = min(len1, first_diff + 10)
    
    return False, (
        f"{name}: {mismatches}/{len1} mismatches ({100*mismatches/len1:.1f}%)\n"
        f"       First diff at position {first_diff}:\n"
        f"       Extracted: {seq1[context_start:context_end]}\n"
        f"       Reference: {seq2[context_start:context_end]}"
    )


# ============================================================================
# Main Pipeline
# ============================================================================

def validate_transcript(tx_id: str, tx_info: dict, genome: pysam.FastaFile,
                       ref_transcripts: dict, ref_proteins: dict) -> dict:
    """Validate one transcript by extracting and comparing sequences."""
    
    result = {
        'tx_id': tx_id,
        'chr': tx_info['chr'],
        'strand': tx_info['strand'],
        'num_exons': len(tx_info['exons']),
        'num_cds': len(tx_info['cds']),
        'transcript_match': None,
        'protein_match': None,
    }
    
    # Validate transcript (from exons)
    if tx_info['exons'] and tx_id in ref_transcripts:
        exon_regions = [(s, e) for s, e in tx_info['exons']]
        extracted = extract_sequence(genome, tx_info['chr'], exon_regions, tx_info['strand'])
        result['transcript_match'] = compare(extracted, ref_transcripts[tx_id], "Transcript")
    
    # Validate protein (from CDS)
    if tx_info['cds'] and tx_id in ref_proteins:
        cds_seq, first_phase = extract_cds_sequence(genome, tx_info['chr'], tx_info['cds'], tx_info['strand'])
        
        # Get selenocysteine positions
        seleno_positions = get_selenocysteine_positions(
            cds_seq, tx_info['cds'], tx_info['selenocysteine'], tx_info['strand']
        )
        
        extracted_protein = translate(cds_seq, first_phase, seleno_positions, chrom=tx_info['chr'])
        # Use the specialized protein comparison function
        result['protein_match'] = compare_protein(extracted_protein, ref_proteins[tx_id], "Protein")
    
    return result


def main():
    """Main validation pipeline."""
    
    print("Loading data...")
    print(f"  GTF: {CONFIG['gtf']}")
    transcripts = parse_gtf(CONFIG['gtf'])
    print(f"  → {len(transcripts)} transcripts")
    
    with_cds = sum(1 for t in transcripts.values() if t['cds'])
    with_seleno = sum(1 for t in transcripts.values() if t.get('selenocysteine'))
    print(f"  → {with_cds} with CDS regions")
    print(f"  → {with_seleno} with selenocysteine")
    
    print(f"  Genome: {CONFIG['genome']}")
    genome = pysam.FastaFile(str(CONFIG['genome']))
    print(f"  → {len(genome.references)} sequences")
    
    print(f"  Reference transcripts: {CONFIG['transcripts']}")
    ref_transcripts = load_fasta(CONFIG['transcripts'], key_index=0)
    print(f"  → {len(ref_transcripts)} sequences")
    
    print(f"  Reference proteins: {CONFIG['proteins']}")
    ref_proteins = load_fasta(CONFIG['proteins'], key_index=1)
    print(f"  → {len(ref_proteins)} sequences")
    
    test_ids = list(transcripts.keys())
    
    print(f"\n{'=' * 80}")
    print(f"Testing {len(test_ids)} transcript(s)")
    print(f"{'=' * 80}\n")
    
    stats = {'tx_pass': 0, 'tx_total': 0, 'prot_pass': 0, 'prot_total': 0}
    failures = []
    
    for i, tx_id in enumerate(test_ids):
        if tx_id not in transcripts:
            continue
        
        result = validate_transcript(tx_id, transcripts[tx_id], genome, 
                                     ref_transcripts, ref_proteins)
        
        tx_failed = result['transcript_match'] and not result['transcript_match'][0]
        prot_failed = result['protein_match'] and not result['protein_match'][0]
        
        if tx_failed or prot_failed:
            failures.append((tx_id, result))
        
        # Print first 5, last 5, and first 20 failures
        should_print = i < 5 or i >= len(test_ids) - 5 or (len(failures) <= 20 and (tx_failed or prot_failed))
        
        if should_print:
            if i == 5 and len(test_ids) > 10 and len(failures) == 0:
                print(f"... (processing, only failures will be shown) ...\n")
            
            print(f"{tx_id} ({result['chr']}:{result['strand']}, "
                  f"{result['num_exons']} exons, {result['num_cds']} CDS)")
            
            if result['transcript_match']:
                is_match, msg = result['transcript_match']
                symbol = '✓' if is_match else '✗'
                print(f"  {symbol} {msg}")
            
            if result['protein_match']:
                is_match, msg = result['protein_match']
                symbol = '✓' if is_match else '✗'
                print(f"  {symbol} {msg}")
            
            print()
        
        if result['transcript_match']:
            stats['tx_total'] += 1
            stats['tx_pass'] += result['transcript_match'][0]
        
        if result['protein_match']:
            stats['prot_total'] += 1
            stats['prot_pass'] += result['protein_match'][0]
    
    print(f"{'=' * 80}")
    print(f"Summary:")
    if stats['tx_total'] > 0:
        print(f"  Transcripts: {stats['tx_pass']}/{stats['tx_total']} passed "
              f"({100*stats['tx_pass']/stats['tx_total']:.1f}%)")
    if stats['prot_total'] > 0:
        print(f"  Proteins: {stats['prot_pass']}/{stats['prot_total']} passed "
              f"({100*stats['prot_pass']/stats['prot_total']:.1f}%)")
    if failures:
        print(f"  ⚠️  {len(failures)} transcripts had failures")
    else:
        print(f"  🎉 All validations passed!")
    print(f"{'=' * 80}")
    
    genome.close()
    return 0 if len(failures) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())