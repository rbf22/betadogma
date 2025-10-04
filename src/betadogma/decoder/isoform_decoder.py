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
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import networkx as nx

from .types import Isoform, Exon


def _find_peaks(logits: torch.Tensor, threshold: float, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds peaks in a logit tensor that are above a threshold."""
    probs = torch.sigmoid(logits)
    peak_indices = (probs > threshold).nonzero(as_tuple=False).squeeze()

    if peak_indices.numel() == 0:
        return torch.tensor([]), torch.tensor([])

    if top_k and peak_indices.numel() > top_k:
        # If there are too many peaks, keep the ones with the highest probability
        peak_probs = probs[peak_indices]
        top_k_indices = torch.topk(peak_probs, k=top_k).indices
        peak_indices = peak_indices[top_k_indices]

    return peak_indices, probs[peak_indices]


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
    """
    def __init__(self, config: Dict):
        self.config = config.get("decoder", {})
        self.thresholds = self.config.get("thresholds", {})
        self.priors = self.config.get("priors", {})

    def build(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> SpliceGraph:
        """
        Takes raw head outputs and constructs a splice graph.
        Assumes a single batch element for now.

        Args:
            head_outputs: A dictionary containing tensors for 'donor',
                          'acceptor', 'tss', and 'polya' logits.
            strand: The strand of the sequence ('+' or '-').

        Returns:
            A SpliceGraph object representing possible exon connections.
        """
        donor_logits = head_outputs["splice"]["donor"].squeeze(0).squeeze(-1)
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze(0).squeeze(-1)

        donor_indices, donor_scores = _find_peaks(donor_logits, self.thresholds.get("donor", 0.5))
        acceptor_indices, acceptor_scores = _find_peaks(acceptor_logits, self.thresholds.get("acceptor", 0.5))

        graph = SpliceGraph()
        candidate_exons = []

        # 2. Define candidate exons based on strand
        if strand == '+':
            # Positive strand: pair acceptor -> donor (acc_coord < don_coord)
            for i, acc_idx in enumerate(acceptor_indices):
                for j, don_idx in enumerate(donor_indices):
                    if acc_idx < don_idx:
                        exon_len = don_idx - acc_idx
                        if self.priors.get("min_exon_len", 0) <= exon_len:
                            score = (acceptor_scores[i] + donor_scores[j]) / 2.0
                            exon = Exon(start=int(acc_idx), end=int(don_idx), score=float(score))
                            graph.add_exon(exon)
                            candidate_exons.append(exon)
        else: # strand == '-'
            # Negative strand: pair donor -> acceptor (don_coord < acc_coord)
            for i, don_idx in enumerate(donor_indices):
                for j, acc_idx in enumerate(acceptor_indices):
                    if don_idx < acc_idx:
                        exon_len = acc_idx - don_idx
                        if self.priors.get("min_exon_len", 0) <= exon_len:
                            score = (donor_scores[i] + acceptor_scores[j]) / 2.0
                            exon = Exon(start=int(don_idx), end=int(acc_idx), score=float(score))
                            graph.add_exon(exon)
                            candidate_exons.append(exon)

        # 3. Add junctions (edges) between exons
        candidate_exons.sort(key=lambda e: e.start)
        for i in range(len(candidate_exons)):
            for j in range(i + 1, len(candidate_exons)):
                exon1 = candidate_exons[i]
                exon2 = candidate_exons[j]

                # Junctions are strand-independent here (always exon1.end -> exon2.start)
                intron_len = exon2.start - exon1.end
                if 0 < intron_len <= self.priors.get("max_intron_len", 500000):
                    junction_score = (exon1.score + exon2.score) / 2 # simplified
                    graph.add_junction(exon1, exon2, score=junction_score)

        return graph


class IsoformEnumerator:
    """
    Enumerates candidate isoforms from a splice graph using beam search.
    """
    def __init__(self, config: Dict):
        self.config = config.get("decoder", {})
        self.beam_size = self.config.get("beam_size", 16)

    def enumerate(self, graph: SpliceGraph, max_paths: int, strand: str = '+') -> List[Isoform]:
        """
        Finds the top-K paths through the splice graph using beam search.

        Args:
            graph: The SpliceGraph to search.
            max_paths: The maximum number of isoform candidates to return.
            strand: The strand, used to create the final Isoform object.

        Returns:
            A list of the most likely Isoform objects.
        """
        if not graph.graph or graph.graph.number_of_nodes() == 0:
            return []

        source_nodes = [n for n, d in graph.graph.in_degree() if d == 0]

        # A beam is a list of (score, path) tuples
        # Initialize the beam with the source nodes
        beam = [(-graph.graph.nodes[n]['score'], [n]) for n in source_nodes]
        beam.sort(key=lambda x: x[0]) # Sort by score (negative log prob)

        completed_paths = []

        while beam:
            new_beam = []
            for score, path in beam:
                last_node = path[-1]

                # If it's a terminal node, add to completed paths
                if graph.graph.out_degree(last_node) == 0:
                    completed_paths.append((score, path))
                    continue

                # Expand to neighbors
                for neighbor in graph.graph.neighbors(last_node):
                    edge_score = graph.graph.edges[last_node, neighbor].get('score', 0)
                    node_score = graph.graph.nodes[neighbor]['score']
                    new_score = score - (edge_score + node_score) # Additive log probs
                    new_path = path + [neighbor]
                    new_beam.append((new_score, new_path))

            # Prune the new beam to keep only the top-k paths
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:self.beam_size]

            # Break if beam is empty to prevent infinite loops
            if not beam:
                break

        # Sort completed paths by score and take the top `max_paths`
        completed_paths.sort(key=lambda x: x[0])

        # Convert paths to Isoform objects
        isoforms = []
        for score, path in completed_paths[:max_paths]:
            exons = []
            for node_key in path:
                node_data = graph.graph.nodes[node_key]
                start, end = node_key
                exons.append(Exon(start=start, end=end, score=node_data['score']))

            # Final score is the average log-likelihood
            isoform_score = -score / len(path) if path else 0
            isoforms.append(Isoform(exons=exons, strand=strand, score=isoform_score))

        return isoforms


class IsoformScorer(nn.Module):
    """
    Scores candidate isoforms based on structural signals and priors.
    This is a learnable module.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config.get("decoder", {})
        scoring_config = self.config.get("scoring", {})

        # Define learnable weights for different components of the score
        self.w_spl = nn.Parameter(torch.tensor(scoring_config.get("w_spl", 1.0)))
        self.w_tss = nn.Parameter(torch.tensor(scoring_config.get("w_tss", 0.3)))
        self.w_pa = nn.Parameter(torch.tensor(scoring_config.get("w_pa", 0.3)))
        self.w_len = nn.Parameter(torch.tensor(scoring_config.get("w_len", 0.1)))
        self.w_orf = nn.Parameter(torch.tensor(scoring_config.get("w_orf", 0.5)))

    def forward(self, isoform: Isoform, heads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates a score for a given isoform.
        """
        # 1. Splice Score (from graph construction)
        splice_score = 0
        if isoform.exons:
            splice_score = torch.tensor(sum(exon.score for exon in isoform.exons) / len(isoform.exons))

        # 2. Other scores (placeholders for now)
        tss_score = torch.tensor(0.0)
        polya_score = torch.tensor(0.0)
        len_score = torch.tensor(0.0)
        orf_score = torch.tensor(0.0)

        # Weighted sum of scores
        total_score = (
            self.w_spl * splice_score +
            self.w_tss * tss_score +
            self.w_pa * polya_score +
            self.w_len * len_score +
            self.w_orf * orf_score
        )

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

    def forward(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.
        This is an alias for the `decode` method for nn.Module compatibility.
        """
        return self.decode(head_outputs, strand)

    def decode(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> List[Isoform]:
        """
        Performs end-to-end isoform decoding.

        Args:
            head_outputs: The dictionary of outputs from the model's structural heads.
            strand: The genomic strand.

        Returns:
            A list of decoded Isoform objects, ranked by score.
        """
        splice_graph = self.graph_builder.build(head_outputs, strand=strand)

        max_candidates = self.config.get("decoder", {}).get("max_candidates", 64)
        candidates = self.enumerator.enumerate(splice_graph, max_paths=max_candidates, strand=strand)

        # Re-score the final candidates with the learnable scorer.
        for isoform in candidates:
            isoform.score = self.scorer(isoform, head_outputs).item()

        # Sort candidates by score in descending order
        candidates.sort(key=lambda iso: iso.score, reverse=True)

        return candidates