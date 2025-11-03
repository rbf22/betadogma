# betadogma/core/variant_to_protein.py

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pysam
from Bio.Seq import Seq

@dataclass
class VariantEffect:
    """Result of variant effect prediction.
    
    A single variant can have multiple effects (e.g., frameshift and stop gain).
    The `effect_types` list contains all applicable effect types, and `effect_details`
    provides additional information about each effect.
    """
    variant_id: str
    gene_name: str
    transcript_id: str
    
    # Protein sequences
    ref_protein: str
    alt_protein: str
    protein_changes: List[str]  # List of protein changes, e.g., ["p.V600E", "p.*1234*"]
    
    # CDS changes
    ref_cds: str
    alt_cds: str
    
    # Effect classification
    effect_types: List[str]  # List of effect types, e.g., ["missense", "splice_region"]
    effect_details: List[dict]  # Detailed information about each effect
    is_coding: bool
    
    # NMD prediction
    nmd_predicted: bool
    nmd_confidence: float
    nmd_reason: Optional[str]
    
    # Splice impact (if applicable)
    affects_splicing: bool
    splice_score_delta: Optional[float]
    
    # Metadata
    confidence: float
    
    @property
    def primary_effect(self) -> str:
        """Return the most severe effect type."""
        effect_priority = [
            'frameshift',
            'nonsense',
            'splice_acceptor',
            'splice_donor',
            'splice_region',
            'stop_loss',
            'missense',
            'synonymous',
            'non_coding'
        ]
        
        for effect in effect_priority:
            if effect in self.effect_types:
                return effect
        return 'non_coding' if not self.effect_types else self.effect_types[0]


class VariantToProteinPredictor:
    """Rule-based predictor: Variant → Protein + NMD."""
    
    def __init__(
        self,
        reference_fasta: str,
        gencode_gtf: str,
        gtex_junctions: Optional[str] = None,
    ):
        """Initialize with data sources.
        
        Args:
            reference_fasta: Path to genome FASTA (e.g., GRCh38.fa)
            gencode_gtf: Path to Gencode GTF
            gtex_junctions: Optional path to GTEx junction file
        """
        print("Loading reference genome...")
        self.genome = pysam.FastaFile(reference_fasta)
        
        print("Loading Gencode annotations...")
        self.transcripts = self._load_gencode(gencode_gtf)
        
        print("Loading GTEx junctions...")
        self.junctions = self._load_gtex_junctions(gtex_junctions) if gtex_junctions else {}
        
        # Standard genetic code
        self.codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        
        print(f"Loaded {len(self.transcripts)} transcripts")
    
    def predict(
        self, 
        chrom: str, 
        pos: int, 
        ref: str, 
        alt: str,
        transcript_id: Optional[str] = None
    ) -> List[VariantEffect]:
        """Predict effect of variant on protein(s).
        
        Args:
            chrom: Chromosome (e.g., 'chr17')
            pos: 0-based position
            ref: Reference allele
            alt: Alternate allele
            transcript_id: Optional specific transcript, otherwise all overlapping
            
        Returns:
            List of VariantEffect objects (one per affected transcript)
        """
        variant_id = f"{chrom}:{pos}:{ref}>{alt}"
        
        # Find overlapping transcripts
        if transcript_id:
            transcripts = [self.transcripts.get(transcript_id)]
            if not transcripts[0]:
                raise ValueError(f"Transcript {transcript_id} not found")
        else:
            transcripts = self._find_overlapping_transcripts(chrom, pos)
        
        if not transcripts:
            print(f"No protein-coding transcripts overlap {variant_id}")
            return []
        
        results = []
        for tx in transcripts:
            try:
                effect = self._predict_single_transcript(
                    chrom, pos, ref, alt, tx
                )
                results.append(effect)
            except Exception as e:
                print(f"Error processing {tx['transcript_id']}: {e}")
                continue
        
        return results
    
    def _predict_single_transcript(
        self, 
        chrom: str, 
        pos: int, 
        ref: str, 
        alt: str,
        transcript: dict
    ) -> VariantEffect:
        """Predict effect on a single transcript."""
        
        # 1. Check if variant affects splice sites
        affects_splicing, splice_delta = self._check_splice_impact(
            pos, ref, alt, transcript
        )
        
        # 2. Get reference CDS
        ref_cds = self._extract_cds(transcript)
        
        # 3. Apply variant to CDS
        alt_cds = self._apply_variant_to_cds(
            pos, ref, alt, transcript, ref_cds
        )
        
        # 4. Translate both
        ref_protein = self._translate(ref_cds)
        alt_protein = self._translate(alt_cds)
        
        # 5. Classify effects (now returns a list of effect types and details)
        effect_types, effect_details = self._classify_effect(
            ref_cds, alt_cds, ref_protein, alt_protein
        )
        
        # Add splice effect if applicable
        if affects_splicing:
            effect_types.append("splice_region")
            effect_details.append({
                'type': 'splice_region',
                'description': 'Variant affects a splice region',
                'score_delta': splice_delta
            })
        
        # 6. Predict NMD
        nmd_pred, nmd_conf, nmd_reason = self._predict_nmd(
            alt_cds, alt_protein, transcript, affects_splicing
        )
        
        # 7. Format protein changes
        protein_changes = self._format_protein_change(
            ref_protein, alt_protein, ref_cds, alt_cds
        )
        
        # 8. Determine if coding (has a valid CDS and protein sequence)
        is_coding = bool(ref_cds and ref_protein and 
                        not all(c == 'N' for c in ref_cds) and
                        not all(c == 'X' for c in ref_protein))
        
        # 9. Create and return the effect object
        return VariantEffect(
            variant_id=f"{chrom}:{pos}:{ref}>{alt}",
            gene_name=transcript['gene_name'],
            transcript_id=transcript['transcript_id'],
            ref_protein=ref_protein,
            alt_protein=alt_protein,
            protein_changes=protein_changes,
            ref_cds=ref_cds,
            alt_cds=alt_cds,
            effect_types=effect_types,
            effect_details=effect_details,
            is_coding=is_coding,
            nmd_predicted=nmd_pred,
            nmd_confidence=nmd_conf,
            nmd_reason=nmd_reason,
            affects_splicing=affects_splicing,
            splice_score_delta=splice_delta,
            confidence=0.9 if not affects_splicing else 0.6,
        )
    
    def _translate(self, cds: str) -> str:
        """Translate CDS to protein."""
        if len(cds) % 3 != 0:
            # Frameshift - translate what we can
            cds = cds[:len(cds) - len(cds) % 3]
        
        protein = []
        for i in range(0, len(cds), 3):
            codon = cds[i:i+3].upper()
            aa = self.codon_table.get(codon, 'X')  # X for unknown
            protein.append(aa)
            if aa == '*':  # Stop codon
                break
        
        return ''.join(protein)
    
    def _predict_nmd(
        self, 
        alt_cds: str, 
        alt_protein: str, 
        transcript: dict,
        affects_splicing: bool
    ) -> Tuple[bool, float, Optional[str]]:
        """Predict if transcript will undergo NMD.
        
        NMD rules:
        1. PTC (premature termination codon) >50-55 nt upstream of last exon junction
        2. Frameshift mutations that introduce a PTC
        3. Splice site mutations creating frameshifts
        
        Returns:
            (is_nmd, confidence, reason)
        """
        # Debug info
        print(f"\nDebug - NMD Prediction:")
        print(f"Alt CDS length: {len(alt_cds)}")
        print(f"Alt protein: {alt_protein}")
        print(f"Affects splicing: {affects_splicing}")
        print(f"Transcript ID: {transcript['transcript_id']}")
        print(f"CDS regions: {transcript['cds']}")
        
        # Check for premature stop (excluding the normal stop at the end)
        has_ptc = '*' in alt_protein[:-1]
        
        # More accurate frameshift detection that accounts for the normal stop codon
        ref_len = len(self._extract_cds(transcript))
        is_frameshift = (ref_len % 3) != (len(alt_cds) % 3)
        
        print(f"Has PTC: {has_ptc}, Is frameshift: {is_frameshift} (ref_len % 3 = {ref_len % 3}, alt_len % 3 = {len(alt_cds) % 3})")
        
        # If it's a frameshift or has a PTC, it's likely to be targeted by NMD
        if is_frameshift or has_ptc:
            # For frameshifts, check if there's a PTC introduced
            if has_ptc:
                stop_pos = alt_protein.index('*')
                stop_nt_pos = stop_pos * 3
                print(f"Found PTC at position {stop_pos} (nt {stop_nt_pos})")
                
                # Get last exon junction position in CDS coordinates
                last_junction_cds_pos = self._get_last_junction_cds_pos(transcript)
                print(f"Last junction CDS position: {last_junction_cds_pos}")
                
                # NMD rule: PTC must be >50-55 nt before last junction
                distance_to_junction = last_junction_cds_pos - stop_nt_pos
                print(f"Distance to last junction: {distance_to_junction}nt")
                
                if distance_to_junction > 50:
                    reason = f"PTC {distance_to_junction}nt before last junction"
                    confidence = 0.95 if distance_to_junction > 55 else 0.85
                    print(f"NMD predicted: True, {reason}, confidence: {confidence}")
                    return True, confidence, reason
                else:
                    reason = f"PTC only {distance_to_junction}nt before last junction (needs >50)"
                    print(f"NMD predicted: False, {reason}")
                    return False, 0.9, reason
            else:
                # For frameshifts without a PTC, we should still predict NMD if the frameshift is significant
                # and not too close to the end of the transcript
                cds_pos = len(alt_cds)
                last_junction_cds_pos = self._get_last_junction_cds_pos(transcript)
                distance_to_end = cds_pos - last_junction_cds_pos
                
                # If the frameshift is more than 50nt from the last junction, predict NMD
                if distance_to_end > 50:
                    reason = f"Frameshift mutation {distance_to_end}nt before transcript end"
                    print(f"NMD predicted: True, {reason}")
                    return True, 0.9, reason
                else:
                    reason = f"Frameshift only {distance_to_end}nt before transcript end (needs >50)"
                    print(f"NMD predicted: False, {reason}")
                    return False, 0.8, reason
        
        # For non-frameshift variants, only predict NMD if there's a PTC
        if has_ptc:
            # Find position of first premature stop in CDS
            stop_pos = alt_protein.index('*')
            stop_nt_pos = stop_pos * 3  # Convert to nucleotide position
            
            # Get last exon junction position in CDS coordinates
            last_junction_cds_pos = self._get_last_junction_cds_pos(transcript)
            
            # NMD rule: PTC must be >50-55 nt before last junction
            distance_to_junction = last_junction_cds_pos - stop_nt_pos
            
            if distance_to_junction > 50:
                reason = f"PTC at codon {stop_pos}, {distance_to_junction}nt before last junction"
                confidence = 0.95 if distance_to_junction > 55 else 0.85
                return True, confidence, reason
            else:
                reason = f"PTC only {distance_to_junction}nt before last junction (needs >50)"
                return False, 0.9, reason
        
        # No PTC and no frameshift, so no NMD
        return False, 1.0, None
    
    def _classify_effect(
        self, 
        ref_cds: str, 
        alt_cds: str, 
        ref_protein: str, 
        alt_protein: str
    ) -> Tuple[List[str], List[dict]]:
        """Classify variant effect type.
        
        Returns:
            Tuple containing:
                - List of effect types (e.g., ["missense", "splice_region"])
                - List of effect details (one dict per effect)
        """
        effect_types = []
        effect_details = []
        
        # Debug prints
        print(f"\nDebug - Classifying effect:")
        print(f"Ref CDS length: {len(ref_cds)}, Alt CDS length: {len(alt_cds)}")
        print(f"Ref protein: {ref_protein}")
        print(f"Alt protein: {alt_protein}")
        
        # Check for frameshift
        is_frameshift = len(ref_cds) % 3 != len(alt_cds) % 3
        if is_frameshift:
            effect_types.append("frameshift")
            effect_details.append({
                'type': 'frameshift',
                'description': 'Insertion or deletion causing a frameshift',
                'ref_length': len(ref_cds),
                'alt_length': len(alt_cds)
            })
            print("Frameshift detected based on CDS length change")
        
        # Check for stop codon introduction (nonsense)
        if '*' in alt_protein[:-1] and '*' not in ref_protein[:-1]:
            effect_types.append("nonsense")
            pos = alt_protein.index('*') + 1
            effect_details.append({
                'type': 'nonsense',
                'description': 'Introduction of a premature stop codon',
                'position': pos,
                'ref_aa': ref_protein[pos-1] if pos <= len(ref_protein) else '*',
                'alt_aa': '*'
            })
            print("Nonsense variant - premature stop codon introduced")
        
        # Check for stop codon loss
        if '*' not in alt_protein and '*' in ref_protein:
            effect_types.append("stop_loss")
            pos = ref_protein.index('*') + 1
            effect_details.append({
                'type': 'stop_loss',
                'description': 'Loss of stop codon',
                'position': pos,
                'ref_aa': '*',
                'alt_aa': alt_protein[pos-1] if pos <= len(alt_protein) else '?'
            })
            print("Stop loss variant - stop codon removed")
        
        # Check for protein changes (missense or synonymous)
        if ref_protein != alt_protein and not is_frameshift and "nonsense" not in effect_types:
            # Find all amino acid differences
            for i in range(min(len(ref_protein), len(alt_protein))):
                if ref_protein[i] != alt_protein[i]:
                    # Find the corresponding codon
                    codon_start = i * 3
                    ref_codon = ref_cds[codon_start:codon_start+3]
                    alt_codon = alt_cds[codon_start:codon_start+3]
                    
                    effect_types.append("missense")
                    effect_details.append({
                        'type': 'missense',
                        'description': 'Amino acid change',
                        'position': i + 1,
                        'ref_aa': ref_protein[i],
                        'alt_aa': alt_protein[i],
                        'ref_codon': ref_codon,
                        'alt_codon': alt_codon
                    })
                    print(f"Missense variant - amino acid change at position {i+1}: {ref_protein[i]} -> {alt_protein[i]}")
                    print(f"  Codon: {ref_codon} ({ref_protein[i]}) -> {alt_codon} ({alt_protein[i]})")
                    break
            
            # If lengths differ but no difference found in common positions
            if not effect_types and len(ref_protein) != len(alt_protein):
                effect_types.append("inframe_indel")
                effect_details.append({
                    'type': 'inframe_indel',
                    'description': 'In-frame insertion or deletion',
                    'ref_length': len(ref_protein),
                    'alt_length': len(alt_protein)
                })
                print(f"In-frame indel: ref={len(ref_protein)}aa, alt={len(alt_protein)}aa")
        
        # Check for silent changes in the CDS (synonymous)
        if not effect_types and ref_cds != alt_cds:
            effect_types.append("synonymous")
            # Find the first difference to report
            for i in range(min(len(ref_cds), len(alt_cds))):
                if ref_cds[i] != alt_cds[i]:
                    codon_start = (i // 3) * 3
                    ref_codon = ref_cds[codon_start:codon_start+3]
                    alt_codon = alt_cds[codon_start:codon_start+3]
                    aa_pos = i // 3
                    ref_aa = ref_protein[aa_pos] if aa_pos < len(ref_protein) else '?'
                    alt_aa = alt_protein[aa_pos] if aa_pos < len(alt_protein) else '?'
                    
                    effect_details.append({
                        'type': 'synonymous',
                        'description': 'Silent mutation (no amino acid change)',
                        'position': aa_pos + 1,
                        'ref_aa': ref_aa,
                        'alt_aa': alt_aa,
                        'ref_codon': ref_codon,
                        'alt_codon': alt_codon,
                        'nucleotide_position': i + 1,
                        'ref_nucleotide': ref_cds[i],
                        'alt_nucleotide': alt_cds[i]
                    })
                    
                    print("Synonymous variant - CDS change but no protein change")
                    print(f"  Silent CDS diff at position {i+1}: {ref_cds[i]}>{alt_cds[i]}")
                    print(f"  Codon: {ref_codon} -> {alt_codon} (both code for {ref_aa}) at position {aa_pos+1}")
                    break
        
        # If no effects found, classify as non-coding or no change
        if not effect_types:
            if not ref_cds or all(c == 'N' for c in ref_cds):
                effect_types.append("non_coding")
                effect_details.append({
                    'type': 'non_coding',
                    'description': 'Variant is in a non-coding region',
                    'ref_cds': ref_cds,
                    'alt_cds': alt_cds
                })
                print("No protein-coding effect detected (non-coding region)")
            else:
                effect_types.append("no_change")
                effect_details.append({
                    'type': 'no_change',
                    'description': 'No change in coding sequence',
                    'ref_cds': ref_cds,
                    'alt_cds': alt_cds
                })
                print("No change in coding sequence detected")
        
        return effect_types, effect_details
    
    def _check_splice_impact(
        self, 
        pos: int, 
        ref: str, 
        alt: str, 
        transcript: dict
    ) -> Tuple[bool, Optional[float]]:
        """Check if variant affects splice sites.
        
        Returns:
            (affects_splicing, score_delta)
        """
        # Check if variant is within splice regions
        # Donor: last 3bp of exon + first 6bp of intron
        # Acceptor: last 20bp of intron + first 3bp of exon
        
        for i, exon in enumerate(transcript['exons']):
            exon_start, exon_end = exon
            
            # Check donor (end of exon)
            donor_region = range(exon_end - 3, exon_end + 6)
            if pos in donor_region:
                # TODO: Use MaxEntScan or similar to score impact
                return True, -2.5  # Placeholder score
            
            # Check acceptor (start of exon)
            acceptor_region = range(exon_start - 20, exon_start + 3)
            if pos in acceptor_region:
                return True, -2.0
        
        return False, None
    
    def _load_gencode(self, gtf_path: str, chunk_size: int = 100000) -> dict:
        """Load Gencode GTF and extract transcript structures in chunks.
        
        Args:
            gtf_path: Path to GTF file (can be gzipped)
            chunk_size: Number of lines to process at a time
            
        Returns:
            Dictionary mapping transcript IDs to transcript information
        """
        import pandas as pd
        import gzip
        
        # Dictionary to store transcript information
        transcripts = {}
        
        # Track current transcript being processed
        current_transcript = None
        
        # Function to process a chunk of GTF data
        def process_chunk(chunk):
            nonlocal current_transcript
            
            # Filter for relevant features
            chunk = chunk[chunk['feature'].isin(['transcript', 'exon', 'CDS'])]
            
            for _, row in chunk.iterrows():
                # Parse attributes
                attrs = {}
                for attr in row['attributes'].strip(';').split(';'):
                    attr = attr.strip()
                    if not attr:
                        continue
                    key, value = attr.split(' ', 1)
                    attrs[key] = value.strip('"')
                
                # Skip if not protein coding
                if attrs.get('transcript_type') != 'protein_coding':
                    continue
                    
                tx_id = attrs.get('transcript_id')
                if not tx_id:
                    continue
                
                # Initialize transcript if new
                if tx_id not in transcripts:
                    transcripts[tx_id] = {
                        'transcript_id': tx_id,
                        'gene_id': attrs.get('gene_id', ''),
                        'gene_name': attrs.get('gene_name', ''),
                        'chrom': row['seqname'],
                        'strand': row['strand'],
                        'exons': [],
                        'cds': [],
                        'tx_start': None,
                        'tx_end': None,
                    }
                
                # Update transcript info
                tx = transcripts[tx_id]
                
                if row['feature'] == 'transcript':
                    tx['tx_start'] = row['start']
                    tx['tx_end'] = row['end']
                elif row['feature'] == 'exon':
                    tx['exons'].append((row['start'], row['end']))
                elif row['feature'] == 'CDS':
                    tx['cds'].append((row['start'], row['end']))
        
        # Read GTF in chunks
        chunk = []
        
        # Use gzip.open if file ends with .gz
        open_func = gzip.open if gtf_path.endswith('.gz') else open
        mode = 'rt' if gzip else 'r'
        
        with open_func(gtf_path, mode) as f:
            for line in f:
                # Skip comments and header lines
                if line.startswith('#') or not line.strip():
                    continue
                    
                # Parse line
                parts = line.strip().split('\t')
                if len(parts) < 9:  # Skip malformed lines
                    continue
                    
                # Create row dictionary
                row = {
                    'seqname': parts[0],
                    'source': parts[1],
                    'feature': parts[2],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'score': parts[5],
                    'strand': parts[6],
                    'frame': parts[7],
                    'attributes': parts[8]
                }
                
                chunk.append(row)
                
                # Process chunk if it reaches the chunk size
                if len(chunk) >= chunk_size:
                    process_chunk(pd.DataFrame(chunk))
                    chunk = []
        
        # Process any remaining lines
        if chunk:
            process_chunk(pd.DataFrame(chunk))
        
        # Sort exons and CDS coordinates for each transcript
        for tx_id, tx in transcripts.items():
            tx['exons'].sort()
            tx['cds'].sort()
            
            # If no transcript coordinates, use min/max of exons
            if tx['tx_start'] is None and tx['exons']:
                tx['tx_start'] = min(start for start, _ in tx['exons'])
                tx['tx_end'] = max(end for _, end in tx['exons'])
        
        return transcripts
        
        return transcripts
    
    def _extract_cds(self, transcript: dict) -> str:
        """Extract CDS sequence from transcript."""
        cds_parts = []
        
        for start, end in transcript['cds']:
            seq = self.genome.fetch(
                transcript['chrom'], 
                start, 
                end
            ).upper()
            cds_parts.append(seq)
        
        cds = ''.join(cds_parts)
        
        # Reverse complement if on minus strand
        if transcript['strand'] == '-':
            cds = str(Seq(cds).reverse_complement())
        
        # Debug print CDS sequence
        print(f"CDS length: {len(cds)}")
        print(f"CDS (first 100bp): {cds[:100]}")
        
        return cds
    
    def _apply_variant_to_cds(
        self, 
        pos: int, 
        ref: str,
        alt: str,
        transcript: dict,
        ref_cds: str
    ) -> str:
        """Apply variant to CDS sequence.
        
        Args:
            pos: 0-based genomic position
            ref: Reference allele
            alt: Alternate allele
            transcript: Transcript information
            ref_cds: Reference CDS sequence
            
        Returns:
            Modified CDS sequence with variant applied
        """
        print(f"\nDebug - Applying variant: {ref}->{alt} at position {pos}")
        print(f"Original CDS (first 50bp): {ref_cds[:50]}...")
        
        # Convert genomic position to CDS position
        cds_pos = self._genomic_to_cds_position(pos, transcript)
        print(f"CDS position: {cds_pos}")
        
        if cds_pos is None:
            print("Variant not in CDS")
            return ref_cds
            
        # Debug: Check the reference allele at this position
        ref_at_pos = ref_cds[cds_pos:cds_pos + len(ref)]
        print(f"Reference at position {cds_pos}: {ref_at_pos} (expected: {ref})")
        if ref_at_pos != ref:
            print(f"WARNING: Reference allele mismatch. Expected {ref}, found {ref_at_pos}")
            # Try to find the reference allele in the CDS
            try:
                found_pos = ref_cds.index(ref, max(0, cds_pos - 10), min(len(ref_cds), cds_pos + 10))
                print(f"Found reference allele {ref} at position {found_pos} (offset: {found_pos - cds_pos})")
            except ValueError:
                print(f"Reference allele {ref} not found in the vicinity of position {cds_pos}")
            
        # For deletions, we need to check if we're removing entire codons
        if not alt:  # This is a deletion
            print(f"Deletion detected: removing {len(ref)} bases")
            # Check if we're removing a complete number of codons
            if len(ref) % 3 == 0:
                # In-frame deletion, remove the entire codons
                num_codons = len(ref) // 3
                cds_start = cds_pos - (cds_pos % 3)  # Start of current codon
                cds_end = cds_start + (num_codons * 3)
                result = ref_cds[:cds_start] + ref_cds[cds_end:]
                print(f"In-frame deletion: removed {num_codons} codons")
            else:
                # Frameshift deletion, remove the specified bases
                # Make sure we don't go past the end of the CDS
                end_pos = min(cds_pos + len(ref), len(ref_cds))
                result = ref_cds[:cds_pos] + ref_cds[end_pos:]
                print(f"Frameshift deletion: removed {end_pos - cds_pos} bases at position {cds_pos}")
        else:
            # For insertions or substitutions, just replace at the position
            result = ref_cds[:cds_pos] + alt + ref_cds[cds_pos + len(ref):]
            print(f"Insertion/substitution: added {len(alt)} bases")
        
        print(f"Modified CDS (first 50bp): {result[:50]}...")
        print(f"Original CDS length: {len(ref_cds)}, New CDS length: {len(result)}")
        return result
    
    def _genomic_to_cds_position(self, pos: int, transcript: dict) -> Optional[int]:
        """Convert genomic position to CDS position.
        
        Args:
            pos: 0-based genomic position
            transcript: Transcript information
            
        Returns:
            0-based CDS position, or None if position is not in CDS
        """
        print(f"\nDebug - Converting genomic position to CDS position")
        print(f"Transcript: {transcript['transcript_id']}")
        print(f"Chromosome: {transcript['chrom']}, Strand: {transcript['strand']}")
        print(f"Transcript bounds: {transcript['tx_start']}-{transcript['tx_end']}")
        print(f"CDS regions: {transcript['cds']}")
        print(f"Looking for position: {pos}")
        
        # Check if position is within transcript bounds
        if not (transcript['tx_start'] <= pos < transcript['tx_end']):
            print(f"Position {pos} is outside transcript bounds")
            return None
            
        cds_pos = 0
        
        # Sort CDS regions by genomic position
        cds_regions = sorted(transcript['cds'])
        
        for cds_start, cds_end in cds_regions:
            print(f"Checking CDS region: {cds_start}-{cds_end}")
            
            if pos < cds_start:
                # Position is before this CDS region, not in any CDS
                print(f"Position {pos} is before CDS region {cds_start}-{cds_end}")
                return None
                
            if cds_start <= pos <= cds_end:  # Changed < to <= to include the end position
                # Position is in or at the end of this CDS region
                # If pos is at the end of the CDS, it's still considered part of the CDS
                position_in_cds = cds_pos + (pos - cds_start)
                print(f"Position {pos} is in CDS region {cds_start}-{cds_end}")
                print(f"CDS position: {position_in_cds}")
                return position_in_cds
                
            # Add length of this CDS region
            region_length = cds_end - cds_start
            print(f"Skipping CDS region {cds_start}-{cds_end} (length: {region_length})")
            cds_pos += region_length
        
        print(f"Position {pos} is not in any CDS region")
        return None  # Position not in any CDS region
        
    def _find_overlapping_transcripts(self, chrom: str, pos: int) -> List[dict]:
        """Find all transcripts overlapping a position."""
        overlapping = []
        
        for tx_id, tx in self.transcripts.items():
            if tx['chrom'] == chrom and tx['tx_start'] <= pos < tx['tx_end']:
                overlapping.append(tx)
        
        return overlapping
    
    def _get_last_junction_cds_pos(self, transcript: dict) -> int:
        """Get CDS position of last exon-exon junction."""
        cds_coords = transcript['cds']
        
        if len(cds_coords) < 2:
            return 0
        
        # Sort CDS regions by genomic position
        sorted_cds = sorted(cds_coords, key=lambda x: x[0])
        
        # For minus strand, the last junction is between the last and second-to-last exons
        if transcript['strand'] == '-':
            # Calculate CDS position of the junction before the last exon
            cds_pos = sum(end - start for start, end in sorted_cds[1:])
        else:
            # For plus strand, sum all CDS lengths except the last exon
            cds_pos = sum(end - start for start, end in sorted_cds[:-1])
        
        print(f"Debug - Last junction CDS position: {cds_pos} (strand: {transcript['strand']})")
        return cds_pos
    
    def _format_protein_change(
        self, 
        ref_protein: str, 
        alt_protein: str,
        ref_cds: str,
        alt_cds: str
    ) -> List[str]:
        """Format protein changes in HGVS-like notation.
        
        Returns:
            List of protein change strings, one for each change detected
        """
        changes = []
        
        # If one of the proteins is empty, it's a complete loss or novel protein
        if not ref_protein or not alt_protein:
            if not ref_protein and alt_protein:
                changes.append("p.0_?ins" + alt_protein)
            elif ref_protein and not alt_protein:
                changes.append(f"p.{ref_protein[0]}1*")
            return changes
        
        # Find all differences
        min_len = min(len(ref_protein), len(alt_protein))
        
        # Check for changes in the common length
        for i in range(min_len):
            if ref_protein[i] != alt_protein[i]:
                # Check if this is a stop codon
                if alt_protein[i] == '*':
                    changes.append(f"p.{ref_protein[i]}{i+1}*")
                elif ref_protein[i] == '*':
                    changes.append(f"p.*{i+1}{alt_protein[i]}ext*?")
                else:
                    changes.append(f"p.{ref_protein[i]}{i+1}{alt_protein[i]}")
        
        # Handle length differences
        if len(ref_protein) > len(alt_protein):
            # Truncation
            changes.append(f"p.{ref_protein[len(alt_protein)]}{len(alt_protein)+1}*")
        elif len(alt_protein) > len(ref_protein):
            # Extension
            changes.append(f"p.{ref_protein[-1]}{len(ref_protein)}fs")
            
        # If no changes found but sequences differ, add a generic change
        if not changes and ref_protein != alt_protein:
            changes.append("p.?")
            
        return changes
    
    def _load_gtex_junctions(self, junction_file: str) -> dict:
        """Load GTEx junctions (optional, for future ML component)."""
        # Placeholder for now
        return {}
