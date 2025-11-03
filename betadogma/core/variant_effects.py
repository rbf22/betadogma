"""Variant effect prediction - streamlined implementation."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import pysam
from Bio.Seq import Seq

class Effect(Enum):
    """Variant effects ordered by severity.
    
    Effects are ordered from most to least severe:
    1. Structural variants (deletions, duplications, etc.)
    2. Nonsense-mediated decay (NMD) triggers
    3. Splicing effects
    4. Protein-altering variants
    5. Non-coding variants
    """
    # Structural variants
    DELETION = auto()
    DUPLICATION = auto()
    INSERTION = auto()
    INVERSION = auto()
    TRANSLOCATION = auto()
    
    # Nonsense-mediated decay triggers
    STOP_GAIN = auto()
    FRAMESHIFT = auto()
    
    # Splicing effects
    SPLICE_SITE_DONOR = auto()
    SPLICE_SITE_ACCEPTOR = auto()
    SPLICE_REGION = auto()
    SPLICE_CONSENSUS = auto()
    
    # Protein-altering variants
    STOP_LOSS = auto()
    START_LOSS = auto()
    MISSENSE = auto()
    INFRAME_INSERTION = auto()
    INFRAME_DELETION = auto()
    
    # Non-coding variants
    SYNONYMOUS = auto()
    INTRONIC = auto()
    FIVE_PRIME_UTR = auto()
    THREE_PRIME_UTR = auto()
    UPSTREAM = auto()
    DOWNSTREAM = auto()
    INTERGENIC = auto()
    
    # Helper properties for easier checking
    @property
    def is_structural(self):
        return self in [
            Effect.DELETION,
            Effect.DUPLICATION,
            Effect.INSERTION,
            Effect.INVERSION,
            Effect.TRANSLOCATION
        ]
    
    @property
    def is_nmd_trigger(self):
        return self in [
            Effect.STOP_GAIN,
            Effect.FRAMESHIFT,
            Effect.SPLICE_SITE_DONOR,
            Effect.SPLICE_SITE_ACCEPTOR
        ]
    
    @property
    def is_protein_altering(self):
        return self in [
            Effect.STOP_LOSS,
            Effect.START_LOSS,
            Effect.MISSENSE,
            Effect.INFRAME_INSERTION,
            Effect.INFRAME_DELETION
        ]
    
    @property
    def is_splice_related(self):
        return self in [
            Effect.SPLICE_SITE_DONOR,
            Effect.SPLICE_SITE_ACCEPTOR,
            Effect.SPLICE_REGION,
            Effect.SPLICE_CONSENSUS
        ]

@dataclass
class ProteinChange:
    """HGVS-style protein change."""
    pos: int
    ref: str
    alt: str
    type: str
    
    def __str__(self):
        if self.type == 'missense':
            return f"p.{self.ref}{self.pos}{self.alt}"
        elif self.type == 'nonsense':
            return f"p.{self.ref}{self.pos}*"
        elif self.type == 'frameshift':
            return f"p.{self.ref}{self.pos}fs"
        return "p.?"

@dataclass
class VariantEffect:
    """Variant effect on a transcript."""
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    tx_id: str
    effects: List[Effect] = field(default_factory=list)
    changes: List[ProteinChange] = field(default_factory=list)
    nmd: bool = False
    nmd_confidence: float = 0.0
    
    @property
    def primary_effect(self) -> Effect:
        return min(self.effects) if self.effects else Effect.INTERGENIC

class VariantEffectPredictor:
    """Predicts variant effects on transcripts."""
    
    def __init__(self, ref_fasta: str, gtf: str):
        self.genome = pysam.FastaFile(ref_fasta)
        self.transcripts = self._load_transcripts(gtf)
        self.codon_table = {
            **{codon: 'L' for codon in ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG']},
            **{codon: 'S' for codon in ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC']},
            **{codon: 'R' for codon in ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG']},
            'TTT': 'F', 'TTC': 'F', 'TAT': 'Y', 'TAC': 'Y',
            'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C',
            'TGA': '*', 'TGG': 'W', 'CCT': 'P', 'CCC': 'P',
            'CCA': 'P', 'CCG': 'P', 'CAT': 'H', 'CAC': 'H',
            'CAA': 'Q', 'CAG': 'Q', 'ATT': 'I', 'ATC': 'I',
            'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T',
            'ACA': 'T', 'ACG': 'T', 'AAT': 'N', 'AAC': 'N',
            'AAA': 'K', 'AAG': 'K', 'GTT': 'V', 'GTC': 'V',
            'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A',
            'GCA': 'A', 'GCG': 'A', 'GAT': 'D', 'GAC': 'D',
            'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G',
            'GGA': 'G', 'GGG': 'G'
        }
    
    def predict(self, chrom: str, pos: int, ref: str, alt: str, tx_id: str = None) -> List[VariantEffect]:
        """Predict effects for a variant."""
        txs = [self.transcripts[tx_id]] if tx_id else self._find_txs(chrom, pos)
        return [self._predict_tx(chrom, pos, ref, alt, tx) for tx in txs if tx]
    
    def _predict_tx(self, chrom: str, pos: int, ref: str, alt: str, tx: dict) -> VariantEffect:
        """Predict effects on a single transcript."""
        effect = VariantEffect(chrom, pos, ref, alt, tx['gene'], tx['id'])
        
        # Get reference and variant CDS
        ref_cds = self._get_cds(tx)
        alt_cds = self._apply_variant(ref_cds, tx, pos, ref, alt)
        
        # Debug: Store CDS sequences
        effect.ref_cds = ref_cds
        effect.alt_cds = alt_cds
        
        # Translate to proteins
        ref_prot = self._translate(ref_cds)
        alt_prot = self._translate(alt_cds)
        
        # Debug: Store protein sequences
        effect.ref_prot = ref_prot
        effect.alt_prot = alt_prot
        
        # Debug: Print CDS and protein info
        print(f"\nTranscript: {tx['id']}")
        print(f"CDS length: {len(ref_cds)} bp")
        print(f"Ref CDS (first 50bp): {ref_cds[:50]}")
        print(f"Alt CDS (first 50bp): {alt_cds[:50]}")
        print(f"Ref protein (first 50aa): {ref_prot[:50]}")
        print(f"Alt protein (first 50aa): {alt_prot[:50]}")
        
        # Detect effects
        self._detect_effects(effect, ref_cds, alt_cds, ref_prot, alt_prot, tx)
        
        # Predict NMD
        effect.nmd = self._predict_nmd(alt_prot, tx)
        
        return effect
    
    def _detect_effects(self, effect: VariantEffect, ref_cds: str, alt_cds: str, 
                        ref_prot: str, alt_prot: str, tx: dict) -> None:
        """Detect all effects of the variant.
        
        Args:
            effect: VariantEffect object to populate with effects
            ref_cds: Reference CDS sequence
            alt_cds: Alternate CDS sequence (with variant)
            ref_prot: Reference protein sequence
            alt_prot: Alternate protein sequence
            tx: Transcript dictionary
        """
        # Debug information
        print(f"\nDetecting effects for {effect.chrom}:{effect.pos+1}{effect.ref}>{effect.alt}")
        print(f"Ref CDS len: {len(ref_cds)}, Alt CDS len: {len(alt_cds)}")
        print(f"Ref prot len: {len(ref_prot)}, Alt prot len: {len(alt_prot)}")
        print(f"Ref prot: {ref_prot[:50]}...")
        print(f"Alt prot: {alt_prot[:50]}...")
        
        # Debug: Print transcript structure
        print("\nTranscript structure:")
        print(f"Transcript ID: {tx.get('id')}")
        print(f"Gene: {tx.get('gene')}")
        print(f"Strand: {tx.get('strand')}")
        print(f"CDS regions: {tx.get('cds')}")
        
        # Check if the transcript has CDS regions
        cds_regions = tx.get('cds', [])
        if not cds_regions:
            effect.effects.append(Effect.INTERGENIC)
            print("No CDS regions found in transcript")
            return
            
        # Get the transcript's coding region boundaries
        cds_start = cds_regions[0]['start']  # Start of first CDS region
        cds_end = cds_regions[-1]['end']     # End of last CDS region
        print(f"CDS start: {cds_start}, CDS end: {cds_end}")
        
        # Convert CDS regions to list of tuples for compatibility with _genomic_to_cds_pos
        tx['cds_tuples'] = [(r['start'], r['end']) for r in cds_regions]
        
        # Check if the variant is in the coding region
        cds_pos, region_idx = self._genomic_to_cds_pos(tx, effect.pos)
        if cds_pos == -1:
            # Non-coding variant
            if effect.pos < cds_start:
                effect.effects.append(Effect.FIVE_PRIME_UTR)
                print(f"Variant at {effect.pos+1} is in 5' UTR (CDS starts at {cds_start+1})")
            else:
                effect.effects.append(Effect.THREE_PRIME_UTR)
                print(f"Variant at {effect.pos+1} is in 3' UTR (CDS ends at {cds_end})")
            return
            
        print(f"Variant at {effect.pos+1} is in CDS at position {cds_pos}")
        
        # Check for frameshift (indel that's not a multiple of 3)
        if len(ref_cds) != len(alt_cds):
            if (len(ref_cds) - len(alt_cds)) % 3 != 0:
                effect.effects.append(Effect.FRAMESHIFT)
                change = self._find_first_change(ref_prot, alt_prot, 'frameshift')
                effect.changes.append(change)
                print(f"Frameshift detected: {change}")
                return  # Frameshift is the most severe effect
            else:
                effect.effects.append(Effect.INFRAME_DELETION if len(ref_cds) > len(alt_cds) else Effect.INFRAME_INSERTION)
                change = self._find_first_change(ref_prot, alt_prot, 'inframe_indel')
                effect.changes.append(change)
                print(f"In-frame {'deletion' if len(ref_cds) > len(alt_cds) else 'insertion'} detected: {change}")
                return
        
        # Check for stop gain (premature stop codon in alt)
        if '*' in alt_prot and (not ref_prot or '*' not in ref_prot or alt_prot.index('*') < ref_prot.index('*')):
            effect.effects.append(Effect.STOP_GAIN)
            pos = alt_prot.index('*') + 1
            ref_aa = ref_prot[pos-1] if pos <= len(ref_prot) else ''
            change = ProteinChange(pos, ref_aa, '*', 'nonsense')
            effect.changes.append(change)
            print(f"Stop gain detected: {change}")
            return  # Stop gain is more severe than missense
            
        # Check for stop loss (loss of stop codon in alt)
        if '*' in ref_prot and (not alt_prot or '*' not in alt_prot or ref_prot.index('*') < alt_prot.index('*')):
            effect.effects.append(Effect.STOP_LOSS)
            pos = ref_prot.index('*') + 1
            alt_aa = alt_prot[pos-1] if pos <= len(alt_prot) else ''
            change = ProteinChange(pos, '*', alt_aa, 'stop_loss')
            effect.changes.append(change)
            print(f"Stop loss detected: {change}")
            return  # Stop loss is more severe than missense
        
        # Check for missense or synonymous
        min_len = min(len(ref_prot), len(alt_prot))
        if min_len > 0:  # Only check if we have protein sequences
            # Find the first position where the sequences differ
            for i in range(min_len):
                if ref_prot[i] != alt_prot[i]:
                    # Calculate the actual amino acid position in the protein
                    aa_pos = i + 1
                    ref_aa = ref_prot[i]
                    alt_aa = alt_prot[i]
                    
                    # Check if it's a start codon change (M to something else)
                    if i == 0 and ref_aa == 'M' and alt_aa != 'M':
                        effect.effects.append(Effect.START_LOSS)
                        change = ProteinChange(aa_pos, ref_aa, alt_aa, 'start_lost')
                        effect.changes.append(change)
                        print(f"Start loss detected: {change}")
                        return
                    
                    # Regular missense variant
                    effect.effects.append(Effect.MISSENSE)
                    change = ProteinChange(aa_pos, ref_aa, alt_aa, 'missense')
                    effect.changes.append(change)
                    print(f"Missense variant: {change} at position {aa_pos}")
                    
                    # Check if this is a known pathogenic mutation
                    if effect.gene == 'BRCA1' and aa_pos == 175 and alt_aa == 'R':
                        print("Note: This is the BRCA1 p.Cys175Arg pathogenic mutation")
                    elif effect.gene == 'TP53' and aa_pos == 175 and alt_aa == 'H':
                        print("Note: This is the TP53 p.Arg175His pathogenic mutation")
                        
                    return
            
            # If we get here and lengths are equal, it's a synonymous variant
            if len(ref_prot) == len(alt_prot):
                # Find the actual position of the change in the CDS
                aa_pos = (cds_pos // 3) + 1  # Convert 0-based CDS pos to 1-based AA pos
                if aa_pos <= len(ref_prot) and aa_pos <= len(alt_prot):
                    ref_aa = ref_prot[aa_pos-1]
                    alt_aa = alt_prot[aa_pos-1]
                    
                    # Only report as synonymous if the ref and alt match at this position
                    if ref_aa == alt_aa:
                        effect.effects.append(Effect.SYNONYMOUS)
                        change = ProteinChange(aa_pos, ref_aa, alt_aa, 'synonymous')
                        effect.changes.append(change)
                        print(f"Synonymous variant: {change}")
                    else:
                        effect.effects.append(Effect.MISSENSE)
                        change = ProteinChange(aa_pos, ref_aa, alt_aa, 'missense')
                        effect.changes.append(change)
                        print(f"Missense variant (position-based): {change}")
                else:
                    print(f"Warning: Position {aa_pos} is out of range for protein length {len(ref_prot)}")
        
        # If we get here and no effects were detected, it might be a silent variant in a non-coding region
        if not effect.effects:
            print(f"Warning: No effects detected for variant {effect.chrom}:{effect.pos+1}{effect.ref}>{effect.alt}")
            effect.effects.append(Effect.INTRONIC)
            effect.changes.append(change)
            print(f"Synonymous variant: {change}")
        else:
            # If lengths differ but no stop codons, it's a frameshift that wasn't caught earlier
            effect.effects.append(Effect.FRAMESHIFT)
            change = self._find_first_change(ref_prot, alt_prot, 'frameshift')
            effect.changes.append(change)
            print(f"Frameshift detected (length difference): {change}")
    
    def _predict_nmd(self, prot: str, tx: dict) -> bool:
        """Predict NMD based on PTC position.
        
        Args:
            prot: Protein sequence (may or may not contain stop codon)
            tx: Transcript dictionary
            
        Returns:
            bool: True if NMD is predicted, False otherwise
        """
        # If there's no stop codon, no NMD
        if '*' not in prot:
            return False
            
        # Find the first stop codon position (in amino acids)
        ptc_aa_pos = prot.index('*')
        
        # Convert to nucleotide position in CDS
        ptc_pos = ptc_aa_pos * 3
        
        # Get the position of the last exon start relative to CDS start
        last_exon_start = tx['exons'][-1]['start']
        cds_start = tx['cds_start']
        
        # For negative strand, we need to adjust the positions
        if tx['strand'] == '-':
            # For negative strand, positions are reversed
            cds_length = sum(c['end'] - c['start'] for c in tx['cds'])
            ptc_pos = cds_length - ptc_pos - 3  # Convert to position from end
            
            # Calculate distance from PTC to last exon start
            distance = (cds_start + cds_length - ptc_pos) - last_exon_start
        else:
            # For positive strand, just calculate the distance
            distance = (last_exon_start - cds_start) - ptc_pos
        
        # NMD is predicted if the PTC is more than 50-55 nt upstream of the last exon junction
        return distance > 50
    
    def _translate(self, cds: str) -> str:
        """Translate CDS to protein.
        
        Args:
            cds: Coding sequence (must be in frame)
            
        Returns:
            Protein sequence with '*' for stop codons
        """
        if not cds or len(cds) < 3:
            return ""
            
        protein = []
        for i in range(0, len(cds) - 2, 3):
            codon = cds[i:i+3]
            if len(codon) < 3:
                break  # Incomplete codon at the end
            aa = self.codon_table.get(codon, 'X')
            protein.append(aa)
            
        return ''.join(protein)
    
    def _load_transcripts(self, gtf: str) -> Dict[str, dict]:
        """Load transcript annotations from GTF file."""
        import gzip
        import re
        from collections import defaultdict
        
        transcripts = {}
        current_tx = None
        
        # Compile regex for parsing GTF attributes
        attr_pattern = re.compile(r'([^\s;]+)\s+"([^"]*)";')
        
        def parse_attrs(attr_str: str) -> dict:
            return {k: v for k, v in attr_pattern.findall(attr_str)}
        
        # First pass: collect all transcripts and their basic info
        with gzip.open(gtf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                chrom, source, feature, start, end, _, strand, _, attrs = fields
                if feature != 'transcript':
                    continue
                
                attrs = parse_attrs(attrs)
                tx_id = attrs.get('transcript_id')
                if not tx_id:
                    continue
                
                transcripts[tx_id] = {
                    'id': tx_id,
                    'gene': attrs.get('gene_name', tx_id.split('.')[0]),
                    'chrom': chrom,
                    'start': int(start) - 1,  # 0-based
                    'end': int(end),
                    'strand': strand,
                    'exons': [],
                    'cds': []
                }
        
        # Second pass: collect exons and CDS
        with gzip.open(gtf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                chrom, source, feature, start, end, _, strand, _, attrs = fields
                attrs = parse_attrs(attrs)
                
                if feature not in ('exon', 'CDS'):
                    continue
                
                tx_id = attrs.get('transcript_id')
                if not tx_id or tx_id not in transcripts:
                    continue
                
                tx = transcripts[tx_id]
                region = {
                    'start': int(start) - 1,  # 0-based
                    'end': int(end)
                }
                
                if feature == 'exon':
                    tx['exons'].append(region)
                elif feature == 'CDS':
                    tx['cds'].append(region)
        
        # Process transcripts
        for tx_id, tx in transcripts.items():
            # Sort exons and CDS by genomic position
            tx['exons'].sort(key=lambda x: x['start'])
            tx['cds'].sort(key=lambda x: x['start'])
            
            # Set CDS boundaries
            if tx['cds']:
                tx['cds_start'] = tx['cds'][0]['start']
                tx['cds_end'] = tx['cds'][-1]['end']
            else:
                tx['cds_start'] = tx['start']
                tx['cds_end'] = tx['end']
                
            # Add debugging info
            tx['num_exons'] = len(tx['exons'])
            tx['num_cds'] = len(tx['cds'])
            
            # Print debug info for our test transcripts
            if tx_id in ['ENST00000357654.9', 'ENST00000269305.9']:
                print(f"\nLoaded transcript: {tx_id}")
                print(f"  Gene: {tx['gene']}, Chrom: {tx['chrom']}, Strand: {tx['strand']}")
                print(f"  Exons: {len(tx['exons'])}")
                print(f"  CDS: {len(tx['cds'])} regions, {tx['cds_start']+1}-{tx['cds_end']}")
                for i, cds in enumerate(tx['cds'][:3]):
                    print(f"    CDS {i+1}: {cds['start']+1}-{cds['end']}")
                if len(tx['cds']) > 3:
                    print(f"    ... and {len(tx['cds'])-3} more CDS regions")
        
        return transcripts
    
    def _find_txs(self, chrom: str, pos: int) -> List[dict]:
        """Find transcripts overlapping position."""
        # Implementation would find overlapping transcripts
        return []
    
    def _get_cds(self, tx: dict) -> str:
        """Get CDS sequence for transcript."""
        if not tx.get('cds'):
            return ""
        return self._get_cds_sequence(tx)
    
    def _genomic_to_cds_pos(self, tx: dict, pos: int) -> Tuple[int, int]:
        """Convert genomic position to CDS position and check if it's in a CDS region.
        
        Returns:
            Tuple of (cds_position, region_index) if position is in CDS, (-1, -1) otherwise
        """
        # Debug information
        print(f"\nConverting genomic position {pos+1} to CDS position")
        print(f"Transcript: {tx.get('id')}, Strand: {tx.get('strand')}")
        print(f"CDS regions: {tx.get('cds')}")
        
        cds_pos = 0
        for i, region in enumerate(tx['cds']):
            print(f"  Checking region {i+1}: {region['start']+1}-{region['end']}")
            if region['start'] <= pos < region['end']:
                offset = pos - region['start']
                print(f"  Found in region {i+1}, offset {offset}, CDS pos {cds_pos + offset}")
                return (cds_pos + offset, i)
            cds_pos += (region['end'] - region['start'])
            
        print(f"  Position {pos+1} not found in any CDS region")
        return (-1, -1)
    
    def _get_cds_sequence(self, tx: dict, variant_pos: int = -1, ref: str = '', alt: str = '') -> str:
        """Get CDS sequence, optionally applying a variant.
        
        Args:
            tx: Transcript dictionary
            variant_pos: Genomic position of variant (0-based)
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            CDS sequence with variant applied (if applicable)
        """
        cds_seq = []
        
        for region in tx['cds']:
            # Get reference sequence for this CDS region
            region_seq = self.genome.fetch(tx['chrom'], region['start'], region['end']).upper()
            
            # If this region contains the variant, apply it
            if variant_pos >= 0 and region['start'] <= variant_pos < region['end']:
                offset = variant_pos - region['start']
                # Check reference matches
                if region_seq[offset:offset+len(ref)] != ref:
                    print(f"Warning: Reference mismatch at {tx['chrom']}:{variant_pos+1} (expected {ref}, found {region_seq[offset:offset+len(ref)]})")
                    return ''
                # Apply the variant
                region_seq = region_seq[:offset] + alt + region_seq[offset+len(ref):]
            
            cds_seq.append(region_seq)
        
        # Join all CDS regions
        full_cds = ''.join(cds_seq)
        
        # Reverse complement if on negative strand
        if tx['strand'] == '-':
            full_cds = str(Seq(full_cds).reverse_complement())
            
        return full_cds
    
    def _apply_variant(self, cds: str, tx: dict, pos: int, ref: str, alt: str) -> str:
        """Apply variant to CDS sequence."""
        # Get the full CDS sequence with the variant applied
        variant_cds = self._get_cds_sequence(tx, pos, ref, alt)
        
        # If we couldn't apply the variant, return the original CDS
        if not variant_cds:
            print(f"Warning: Could not apply variant at {tx['chrom']}:{pos+1}{ref}>{alt}")
            return cds
            
        return variant_cds
    
    def _find_first_change(self, ref: str, alt: str, change_type: str) -> ProteinChange:
        """Find first position where ref and alt differ."""
        for i, (r, a) in enumerate(zip(ref, alt)):
            if r != a:
                return ProteinChange(i+1, r, a, change_type)
        return ProteinChange(0, '', '', 'unknown')
