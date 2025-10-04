"""
Isoform decoder for BetaDogma.

This module contains the components for turning the outputs of the structural
heads (splice, TSS, polyA) into transcript isoform structures.

The process follows these steps:
1.  **SpliceGraphBuilder**: Constructs a graph where nodes are potential exon
    boundaries and edges connect them to form valid exons and introns.
2.  **IsoformEnumerator**: Traverses the splice graph to find the most likely
    paths, which correspond to candidate isoforms.
3.  **IsoformScorer**: Scores each candidate isoform based on a combination of
    the structural head logits and other priors (e.g., ORF validity).
"""
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx

from .types import Isoform, Exon


def _find_peaks(logits: torch.Tensor, threshold: float, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds peaks in a logit tensor that are above a threshold."""
    probs = torch.sigmoid(logits)
    peak_indices = (probs > threshold).nonzero(as_tuple=False).squeeze()

    if peak_indices.numel() == 0:
        return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)

    # Ensure peak_indices is always iterable (1-D)
    if peak_indices.dim() == 0:
        peak_indices = peak_indices.unsqueeze(0)

    if top_k and peak_indices.numel() > top_k:
        # If there are too many peaks, keep the ones with the highest probability
        peak_probs = probs[peak_indices]
        top_k_indices = torch.topk(peak_probs, k=top_k).indices
        peak_indices = peak_indices[top_k_indices]

    peak_probs = probs[peak_indices]

    return peak_indices, peak_probs


class SpliceGraph:
    """
    A graph representation of potential splice sites and exons, using networkx.
    Nodes are (start, end) tuples representing exons.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_exon(self, exon: Exon):
        """Adds an exon as a node in the graph."""
        node_key = (exon.start, exon.end)
        self.graph.add_node(node_key, score=exon.score, type='exon')

    def add_junction(self, from_exon: Exon, to_exon: Exon, score: float):
        """Adds a directed edge representing a splice junction."""
        from_key = (from_exon.start, from_exon.end)
        to_key = (to_exon.start, to_exon.end)
        self.graph.add_edge(from_key, to_key, score=score, type='junction')

    def __repr__(self):
        return f"SpliceGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


class SpliceGraphBuilder:
    """
    Builds a splice graph from the outputs of the structural heads.
    This version anchors the graph to high-confidence TSS and polyA sites.
    """
    def __init__(self, config: Dict):
        self.config = config.get("decoder", {})
        self.thresholds = self.config.get("thresholds", {})
        self.priors = self.config.get("priors", {})
        self.max_starts = self.config.get("max_starts", 8)
        self.max_ends = self.config.get("max_ends", 8)
        self.allow_unanchored = self.config.get("allow_unanchored", False)
        self.min_exon_len = self.priors.get("min_exon_len", 25)
        self.max_intron_len = self.priors.get("max_intron_len", 500000)

    def _get_exons(self, starts: List, ends: List, start_scores: List, end_scores: List) -> List[Exon]:
        """Helper to generate exons from paired start/end coordinates."""
        exons = []
        for i, s_idx in enumerate(starts):
            for j, e_idx in enumerate(ends):
                if s_idx < e_idx:
                    exon_len = e_idx - s_idx
                    if exon_len >= self.min_exon_len:
                        score = (start_scores[i] + end_scores[j]) / 2.0
                        exons.append(Exon(start=int(s_idx), end=int(e_idx), score=float(score)))
        return exons

    def build(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> SpliceGraph:
        """Takes raw head outputs and constructs a splice graph."""
        # 1. Find all peaks for relevant signals
        donor_logits = head_outputs["splice"]["donor"].squeeze()
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze()
        tss_logits = head_outputs.get("tss", {}).get("tss", torch.tensor([])).squeeze()
        polya_logits = head_outputs.get("polya", {}).get("polya", torch.tensor([])).squeeze()

        donor_indices, donor_scores = _find_peaks(donor_logits, self.thresholds.get("donor", 0.6))
        acceptor_indices, acceptor_scores = _find_peaks(acceptor_logits, self.thresholds.get("acceptor", 0.6))
        tss_indices, tss_scores = _find_peaks(tss_logits, self.thresholds.get("tss", 0.5), top_k=self.max_starts)
        polya_indices, polya_scores = _find_peaks(polya_logits, self.thresholds.get("polya", 0.5), top_k=self.max_ends)

        # 2. Create candidate exons based on strand-aware roles
        candidate_exons = []
        if strand == '+':
            internal_exons = self._get_exons(acceptor_indices, donor_indices, acceptor_scores, donor_scores)
            first_exons = self._get_exons(tss_indices, donor_indices, tss_scores, donor_scores)
            last_exons = self._get_exons(acceptor_indices, polya_indices, acceptor_scores, polya_scores)
            single_exons = self._get_exons(tss_indices, polya_indices, tss_scores, polya_scores)
        else:  # strand == '-'
            # On the minus strand, an exon (in genomic coordinates) starts at a donor and ends at an acceptor.
            # A "first" exon (5'-most) starts at a donor and ends at a TSS (higher coordinate).
            internal_exons = self._get_exons(donor_indices, acceptor_indices, donor_scores, acceptor_scores)
            first_exons = self._get_exons(donor_indices, tss_indices, donor_scores, tss_scores)
            last_exons = self._get_exons(polya_indices, acceptor_indices, polya_scores, acceptor_scores)
            single_exons = self._get_exons(polya_indices, tss_indices, polya_scores, tss_scores)

        # 3. Combine exons based on anchoring policy
        if self.allow_unanchored:
            candidate_exons.extend(internal_exons)
        candidate_exons.extend(first_exons)
        candidate_exons.extend(last_exons)
        candidate_exons.extend(single_exons)

        # 4. De-duplicate exons and add to graph
        unique_exons = { (e.start, e.end): e for e in sorted(candidate_exons, key=lambda x: x.score) }
        exons_list = list(unique_exons.values())

        graph = SpliceGraph()
        for exon in exons_list:
            graph.add_exon(exon)

        # 5. Add valid junctions between exons
        donor_set = set(donor_indices.tolist())
        acceptor_set = set(acceptor_indices.tolist())

        # Sort exons by transcriptional order to find junctions
        if strand == '+':
            sorted_for_junctions = sorted(exons_list, key=lambda e: e.start)
        else:  # strand == '-'
            sorted_for_junctions = sorted(exons_list, key=lambda e: e.start, reverse=True)

        for i, up_exon in enumerate(sorted_for_junctions):
            for j in range(i + 1, len(sorted_for_junctions)):
                down_exon = sorted_for_junctions[j]

                # Check for valid junction based on strand
                if strand == '+':
                    # Junction: up_exon.end (donor) -> down_exon.start (acceptor)
                    intron_len = down_exon.start - up_exon.end
                    if 0 < intron_len <= self.max_intron_len:
                        if up_exon.end in donor_set and down_exon.start in acceptor_set:
                            score = (up_exon.score + down_exon.score) / 2.0
                            graph.add_junction(up_exon, down_exon, score=score)
                else:  # strand == '-'
                    # Junction: up_exon.start (donor) -> down_exon.end (acceptor)
                    intron_len = up_exon.start - down_exon.end
                    if 0 < intron_len <= self.max_intron_len:
                        if up_exon.start in donor_set and down_exon.end in acceptor_set:
                            score = (up_exon.score + down_exon.score) / 2.0
                            graph.add_junction(up_exon, down_exon, score=score)
        return graph


class IsoformEnumerator:
    """
    Enumerates candidate isoforms from a splice graph using beam search.
    Considers all visited paths as potential candidates.
    """
    def __init__(self, config: Dict):
        self.config = config.get("decoder", {})
        self.beam_size = self.config.get("beam_size", 16)

    def enumerate(self, graph: SpliceGraph, max_paths: int, strand: str = '+') -> List[Isoform]:
        """
        Finds the top-K paths through the splice graph using beam search.
        """
        if not graph.graph or graph.graph.number_of_nodes() == 0:
            return []

        source_nodes = [n for n, d in graph.graph.in_degree() if d == 0]
        if not source_nodes:
            return []

        # A beam is a list of (cumulative_score, path) tuples
        beam = [(graph.graph.nodes[n]['score'], [n]) for n in source_nodes]
        beam.sort(key=lambda x: x[0], reverse=True)

        all_candidate_paths = []

        while beam:
            # Add all paths in the current beam to our list of candidates
            all_candidate_paths.extend(beam)
            new_beam = []

            for score, path in beam:
                last_node = path[-1]

                # Expand to neighbors
                for neighbor in graph.graph.neighbors(last_node):
                    # Path score is the sum of its exon scores
                    node_score = graph.graph.nodes[neighbor]['score']
                    new_score = score + node_score
                    new_path = path + [neighbor]
                    new_beam.append((new_score, new_path))

            # Prune the new beam to keep only the top-k paths
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_size]

            if not beam:
                break

        # De-duplicate paths, keeping the one with the highest score
        unique_paths = {}
        for score, path in all_candidate_paths:
            path_tuple = tuple(path)
            if path_tuple not in unique_paths or score > unique_paths[path_tuple]:
                unique_paths[path_tuple] = score

        # Sort final candidates by normalized score
        sorted_paths = sorted(unique_paths.items(), key=lambda item: item[1] / len(item[0]), reverse=True)

        # Convert top paths to Isoform objects
        isoforms = []
        for path_tuple, score in sorted_paths[:max_paths]:
            exons = []
            for node_key in path_tuple:
                node_data = graph.graph.nodes[node_key]
                start, end = node_key
                exons.append(Exon(start=start, end=end, score=node_data['score']))

            normalized_score = score / len(path_tuple) if path_tuple else 0
            isoforms.append(Isoform(exons=exons, strand=strand, score=normalized_score))

        return isoforms


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(sigmoid(x))"""
    return torch.log(torch.sigmoid(x) + 1e-9)


def _get_peak_log_prob(logits: torch.Tensor, index: int, window: int) -> torch.Tensor:
    """Gets the max log probability in a window around a specified index."""
    if logits is None or index is None:
        return torch.tensor(0.0)
    start = max(0, index - window)
    end = min(len(logits), index + window + 1)
    if start >= end:
        return torch.tensor(0.0)
    return torch.max(_log_sigmoid(logits[start:end]))


def _get_spliced_cDNA(isoform: Isoform, input_ids: torch.Tensor, token_map: Dict[int, str]) -> Tuple[str, List[str]]:
    """
    Constructs the spliced cDNA sequence for an isoform from input token IDs.
    Handles reverse complement for the negative strand.
    Returns the cDNA sequence and a list of the exon sequences.
    """
    # Squeeze to handle batch dimension of 1
    sequence_str = "".join([token_map.get(token_id, "N") for token_id in input_ids.squeeze().tolist()])

    # Sort exons by genomic coordinate to correctly assemble cDNA
    sorted_exons = sorted(isoform.exons, key=lambda e: e.start)
    exon_seqs = [sequence_str[exon.start:exon.end] for exon in sorted_exons]
    spliced_seq = "".join(exon_seqs)

    if isoform.strand == '-':
        # For negative strand, the transcript is read from the reverse complement
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        rev_comp = "".join(complement.get(base, "N") for base in reversed(spliced_seq))
        return rev_comp, exon_seqs

    return spliced_seq, exon_seqs


def _score_orf(
    isoform: Isoform,
    head_outputs: Dict[str, torch.Tensor],
    scoring_config: Dict,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scores the validity of the open reading frame using either head outputs or a sequence scan.
    This function implements a two-tier scoring system as requested.
    """
    use_head = scoring_config.get("use_orf_head", True)
    device = head_outputs["splice"]["donor"].device

    # --- Tier A: Head-driven scoring (default) ---
    if use_head:
        start_logits = head_outputs["orf"]["start"].squeeze()
        stop_logits = head_outputs["orf"]["stop"].squeeze()
        frame_logits = head_outputs["orf"]["frame"].squeeze()
        frame_probs = torch.softmax(frame_logits, dim=-1)
        stop_probs = torch.sigmoid(stop_logits)

        if not isoform.exons:
            return torch.tensor(0.0, device=device)

        tx_to_gen_map = [p for exon in sorted(isoform.exons, key=lambda e: e.start) for p in range(exon.start, exon.end)]
        if not tx_to_gen_map:
            return torch.tensor(0.0, device=device)

        exonic_indices = torch.tensor(tx_to_gen_map, device=device)
        exonic_start_logits = start_logits[exonic_indices]
        top_k = min(scoring_config.get("max_start_candidates", 5), len(exonic_indices))

        if top_k == 0: return torch.tensor(0.0, device=device)

        candidate_start_probs, relative_indices = torch.topk(torch.sigmoid(exonic_start_logits), k=top_k)
        best_overall_score = torch.tensor(-1.0, device=device)

        for s_prob, s_rel_idx in zip(candidate_start_probs, relative_indices):
            s_tx_idx = s_rel_idx.item()
            # Iterate through all three possible reading frames for the given start
            for frame_offset in range(3):
                cds_frame_probs = []
                # Scan downstream from start codon
                for tx_pos in range(s_tx_idx + frame_offset, len(tx_to_gen_map), 3):
                    gen_pos = tx_to_gen_map[tx_pos]
                    current_frame = (tx_pos - s_tx_idx) % 3
                    cds_frame_probs.append(frame_probs[gen_pos, current_frame])

                    # Check for stop codon
                    if stop_probs[gen_pos] > 0.5:
                        stop_prob = stop_probs[gen_pos]
                        mean_frame_prob = torch.mean(torch.stack(cds_frame_probs))
                        score = (scoring_config["orf_alpha"] * s_prob +
                                 scoring_config["orf_beta"] * mean_frame_prob +
                                 scoring_config["orf_alpha"] * stop_prob)

                        # Apply PTC penalty if stop is premature
                        if len(isoform.exons) > 1:
                            exon_lengths = [e.end - e.start for e in sorted(isoform.exons, key=lambda e: e.start)[:-1]]
                            last_junction_pos = sum(exon_lengths)
                            if tx_pos < last_junction_pos - 55:
                                score -= scoring_config["orf_gamma"]

                        if score > best_overall_score:
                            best_overall_score = score
                        break # Found a stop, end this frame scan
                else: # No stop found
                    score = (scoring_config["orf_alpha"] * s_prob) - scoring_config["orf_gamma"]
                    if score > best_overall_score:
                        best_overall_score = score

        return torch.max(torch.tensor(0.0, device=device), best_overall_score)

    # --- Tier B: Sequence-based fallback ---
    else:
        if input_ids is None:
            return torch.tensor(0.0, device=device)

        token_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
        cDNA, exon_seqs = _get_spliced_cDNA(isoform, input_ids, token_map)

        best_score = -1.0

        def is_strong_kozak(seq, pos):
            if pos < 3 or pos + 4 >= len(seq): return False
            return seq[pos-3] in "AG" and seq[pos+3] == "G"

        for i in range(len(cDNA) - 2):
            if cDNA[i:i+3] == "ATG": # Found a start codon
                for j in range(i, len(cDNA) - 2, 3):
                    if cDNA[j:j+3] in {"TAA", "TAG", "TGA"}: # Found a stop codon
                        cds_len_aa = (j - i) // 3
                        score = 0.5
                        if scoring_config.get("min_cds_len_aa", 50) <= cds_len_aa <= scoring_config.get("max_cds_len_aa", 10000):
                            score += 0.3
                        if is_strong_kozak(cDNA, i):
                            score += scoring_config.get("kozak_bonus", 0.2)

                        # PTC penalty
                        if len(exon_seqs) > 1:
                            last_junction_pos = sum(len(s) for s in exon_seqs[:-1])
                            if j < last_junction_pos - 55:
                                score -= scoring_config.get("orf_gamma", 0.6)

                        if score > best_score:
                            best_score = score
                        break # Stop scanning for this ORF
        return torch.tensor(max(0.0, best_score), device=device)


def _score_length(isoform: Isoform, priors: Dict) -> torch.Tensor:
    """Applies a soft penalty for extreme lengths."""
    # Placeholder for a more sophisticated prior
    # For now, just a small penalty for having very few or many exons
    num_exons = len(isoform.exons)
    if num_exons < 2 or num_exons > 20:
        return torch.tensor(-0.5)
    return torch.tensor(0.0)


class IsoformScorer(nn.Module):
    """
    Scores candidate isoforms based on structural signals and priors.
    This is a learnable module.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config.get("decoder", {})
        self.scoring_config = self.config.get("scoring", {})
        self.priors = self.config.get("priors", {})

        # Define learnable weights for different components of the score
        self.w_spl = nn.Parameter(torch.tensor(self.scoring_config.get("w_spl", 1.0)))
        self.w_tss = nn.Parameter(torch.tensor(self.scoring_config.get("w_tss", 0.4)))
        self.w_pa = nn.Parameter(torch.tensor(self.scoring_config.get("w_pa", 0.4)))
        self.w_orf = nn.Parameter(torch.tensor(self.scoring_config.get("w_orf", 0.8)))
        self.w_len = nn.Parameter(torch.tensor(self.scoring_config.get("w_len", 0.1)))

    def forward(self, isoform: Isoform, head_outputs: Dict[str, torch.Tensor], input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates a score for a given isoform.
        """
        device = self.w_spl.device

        # 1. Splice Score
        donor_logits = head_outputs["splice"]["donor"].squeeze()
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze()

        s_spl = 0
        if isoform.exons and len(isoform.exons) > 1:
            if isoform.strand == '+':
                donor_indices = [exon.end for exon in isoform.exons[:-1]]
                acceptor_indices = [exon.start for exon in isoform.exons[1:]]
            else: # strand == '-'
                donor_indices = [exon.start for exon in isoform.exons[:-1]]
                acceptor_indices = [exon.end for exon in isoform.exons[1:]]

            donor_log_probs = _log_sigmoid(donor_logits[donor_indices])
            acceptor_log_probs = _log_sigmoid(acceptor_logits[acceptor_indices])

            s_spl = (torch.sum(donor_log_probs) + torch.sum(acceptor_log_probs)) / (len(donor_indices) + len(acceptor_indices))
        else:
            s_spl = torch.tensor(0.0) # No junctions for single-exon transcripts

        # 2. TSS Score
        tss_logits = head_outputs.get("tss", {}).get("tss", None)
        if tss_logits is not None:
            tss_logits = tss_logits.squeeze()
            start_bin = isoform.exons[0].start if isoform.exons else None
            s_tss = _get_peak_log_prob(tss_logits, start_bin, self.config.get("tss_pa_window", 1))
        else:
            s_tss = torch.tensor(0.0)

        # 3. PolyA Score
        polya_logits = head_outputs.get("polya", {}).get("polya", None)
        if polya_logits is not None:
            polya_logits = polya_logits.squeeze()
            end_bin = isoform.exons[-1].end if isoform.exons else None
            s_pa = _get_peak_log_prob(polya_logits, end_bin, self.config.get("tss_pa_window", 1))
        else:
            s_pa = torch.tensor(0.0)

        # 4. ORF Score
        s_orf = _score_orf(
            isoform,
            head_outputs,
            self.scoring_config,
            input_ids=input_ids
        )

        # 5. Length Score
        s_len = _score_length(isoform, self.priors)

        # Weighted sum of scores
        total_score = (
            self.w_spl * s_spl +
            self.w_tss * s_tss +
            self.w_pa * s_pa +
            self.w_orf * s_orf +
            self.w_len * s_len
        ).to(device)

        return total_score


class IsoformDecoder(nn.Module):
    """
    The main class that orchestrates the isoform decoding process.
    Inherits from nn.Module to hold the learnable scorer.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.graph_builder = SpliceGraphBuilder(config)
        self.enumerator = IsoformEnumerator(config)
        self.scorer = IsoformScorer(config)

    def forward(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.
        This is an alias for the `decode` method for nn.Module compatibility.
        """
        return self.decode(head_outputs, strand=strand, input_ids=input_ids)

    def decode(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.

        Args:
            head_outputs: The dictionary of outputs from the model's structural heads.
            strand: The genomic strand.
            input_ids: Optional tensor of input token IDs for sequence-based scoring.

        Returns:
            A list of decoded Isoform objects, ranked by score.
        """
        splice_graph = self.graph_builder.build(head_outputs, strand=strand)

        max_candidates = self.config.get("decoder", {}).get("max_candidates", 64)
        candidates = self.enumerator.enumerate(splice_graph, max_paths=max_candidates, strand=strand)

        # Re-score the final candidates with the learnable scorer.
        for isoform in candidates:
            isoform.score = self.scorer(isoform, head_outputs, input_ids=input_ids).item()

        # Sort candidates by score in descending order
        candidates.sort(key=lambda iso: iso.score, reverse=True)

        return candidates