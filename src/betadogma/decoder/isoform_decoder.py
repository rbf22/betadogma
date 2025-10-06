# src/betadogma/decoder/isoform_decoder.py
"""
Isoform decoder for BetaDogma.

This module turns the outputs of the structural heads (splice, TSS, polyA, ORF)
into transcript isoform structures, scores them, and returns ranked candidates.

Exposed (tests import these):
- _get_spliced_cDNA
- _score_orf
- SpliceGraphBuilder
- IsoformEnumerator
- IsoformScorer
- IsoformDecoder
- SpliceGraph
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx

from .types import Isoform, Exon

__all__ = [
    "_get_spliced_cDNA",
    "_score_orf",
    "SpliceGraphBuilder",
    "IsoformEnumerator",
    "IsoformScorer",
    "IsoformDecoder",
    "SpliceGraph",
]

# -------------------------
# Shape normalizers (fix batch vs. per-sample bugs)
# -------------------------

def _as_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to shape [L] by selecting batch 0 and squeezing any trailing
    singleton 'channel' dim. Handles: [B,L,1], [B,L], [1,L], [L,1], [L].
    Returns contiguous [L].
    """
    t = x
    # [B,L,1] -> [B,L]
    if t.dim() == 3 and t.size(-1) == 1:
        t = t[..., 0]
    # [B,L] or [1,L] -> take batch 0
    if t.dim() == 2:
        if t.size(0) >= 1:
            t = t[0]
        elif t.size(-1) == 1:
            t = t[:, 0]
    # [L,1] -> [L]
    if t.dim() == 2 and t.size(-1) == 1:
        t = t[:, 0]
    # final safety: if still not 1D, try squeeze
    if t.dim() > 1:
        t = t.squeeze()
    return t.contiguous()

def _as_L3(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize ORF frame logits to shape [L,3].
    Accepts shapes: [B,L,3], [1,L,3], [L,3].
    Returns contiguous [L,3].
    """
    t = x
    if t.dim() == 4 and t.size(1) == 1:  # e.g., [B,1,L,3]
        t = t.squeeze(1)
    if t.dim() == 3 and t.size(0) > 1:   # [B,L,3] -> [L,3]
        t = t[0]
    if t.dim() == 3 and t.size(0) == 1:  # [1,L,3] -> [L,3]
        t = t[0]
    return t.contiguous()

# -------------------------
# Utilities
# -------------------------

def _find_peaks(logits: torch.Tensor, threshold: float, top_k: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds positions with sigmoid(logit) > threshold; keeps top_k by prob if set.
    Robust to logits shaped [1,L] by normalizing to [L] internally.
    """
    if logits.dim() != 1:
        logits = _as_1d(logits)

    probs = torch.sigmoid(logits)  # [L]
    peak_indices = (probs > threshold).nonzero(as_tuple=False).squeeze()

    if peak_indices.numel() == 0:
        device = logits.device
        return torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=logits.dtype)

    if peak_indices.dim() == 0:
        peak_indices = peak_indices.unsqueeze(0)

    if top_k and peak_indices.numel() > top_k:
        peak_probs = probs[peak_indices]
        top_k_indices = torch.topk(peak_probs, k=top_k).indices
        peak_indices = peak_indices[top_k_indices]

    peak_probs = probs[peak_indices]
    return peak_indices, peak_probs


def _order_exons_by_transcription(exons: List[Exon], strand: str) -> List[Exon]:
    """Sort exons in transcription order."""
    key = lambda e: (e.start, e.end)
    reverse = (strand == '-')
    return sorted(exons, key=key, reverse=reverse)


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.sigmoid(x) + 1e-9)


def _get_peak_log_prob(logits: Optional[torch.Tensor], index: Optional[int], window: int) -> torch.Tensor:
    """Max log(sigmoid) in a window around `index`."""
    if logits is None:
        return torch.tensor(0.0)
    if logits.dim() != 1:
        logits = _as_1d(logits)
    if index is None:
        return torch.tensor(0.0, device=logits.device)
    start = max(0, index - window)
    end = min(logits.shape[0], index + window + 1)
    if start >= end:
        return torch.tensor(0.0, device=logits.device)
    return torch.max(_log_sigmoid(logits[start:end]))


def _get_spliced_cDNA(isoform: Isoform, input_ids: torch.Tensor, token_map: Dict[int, str]) -> Tuple[str, List[str]]:
    """
    Build spliced cDNA string for an isoform from input token IDs.
    - Returns (cDNA_seq_5to3, exon_seqs_in_tx_order).
    - Handles reverse complement for '-' strand.
    """
    sequence_str = "".join([token_map.get(int(tid), "N") for tid in input_ids.squeeze().tolist()])

    # assemble exon sequences in genomic order first
    genomically_sorted_exons = sorted(isoform.exons, key=lambda e: e.start)
    genomic_exon_seqs = [sequence_str[e.start:e.end] for e in genomically_sorted_exons]
    spliced_in_genomic_order = "".join(genomic_exon_seqs)

    # exon sequences in transcription order (already ordered by decoder, but enforce)
    tx_exons = _order_exons_by_transcription(list(isoform.exons), isoform.strand)
    tx_ordered_exon_seqs = [sequence_str[e.start:e.end] for e in tx_exons]

    if isoform.strand == '-':
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        rc = "".join(comp.get(b, "N") for b in reversed(spliced_in_genomic_order))
        return rc, tx_ordered_exon_seqs
    else:
        return spliced_in_genomic_order, tx_ordered_exon_seqs


def _score_orf(
    isoform: Isoform,
    head_outputs: Dict[str, torch.Tensor],
    scoring_config: Dict,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Two-tier ORF scoring:
      Tier A: use ORF head (start/stop/frame) if available (default).
      Tier B: fallback to sequence scan (ATG...stop) with simple heuristics.
    Returns a scalar score tensor (>= 0).
    """
    use_head = scoring_config.get("use_orf_head", True)
    device = None
    try:
        device = head_outputs["splice"]["donor"].device
    except Exception:
        device = torch.device("cpu")

    # --- Tier A: head-driven ---
    if use_head and "orf" in head_outputs:
        alpha = scoring_config.get("orf_alpha", 0.5)
        beta  = scoring_config.get("orf_beta", 0.3)
        gamma = scoring_config.get("orf_gamma", 0.6)

        start_logits = _as_1d(head_outputs["orf"]["start"])
        stop_logits  = _as_1d(head_outputs["orf"]["stop"])
        frame_logits = _as_L3(head_outputs["orf"]["frame"])

        frame_probs = torch.softmax(frame_logits, dim=-1)  # [L,3]
        stop_probs  = torch.sigmoid(stop_logits)           # [L]

        if not isoform.exons:
            return torch.tensor(0.0, device=device)

        # tx coordinate map (list of genomic positions covered by exons in tx order)
        tx_exons = _order_exons_by_transcription(list(isoform.exons), isoform.strand)
        tx_to_gen = [p for ex in tx_exons for p in range(ex.start, ex.end)]
        if not tx_to_gen:
            return torch.tensor(0.0, device=device)

        exonic_indices = torch.tensor(tx_to_gen, device=device)
        exonic_start_logits = start_logits[exonic_indices]
        top_k = min(scoring_config.get("max_start_candidates", 5), exonic_start_logits.numel())
        if top_k <= 0:
            return torch.tensor(0.0, device=device)

        candidate_start_probs, relative_indices = torch.topk(torch.sigmoid(exonic_start_logits), k=top_k)
        best_overall = torch.tensor(0.0, device=device)

        for s_prob, s_rel in zip(candidate_start_probs, relative_indices):
            s_tx_idx = int(s_rel.item())
            # try three frames
            for frame_offset in range(3):
                cds_frame_probs = []
                stop_hit_score = None
                for tx_pos in range(s_tx_idx + frame_offset, len(tx_to_gen), 3):
                    gpos = tx_to_gen[tx_pos]
                    current_frame = (tx_pos - s_tx_idx) % 3
                    cds_frame_probs.append(frame_probs[gpos, current_frame])

                    if stop_probs[gpos] > 0.5:
                        stop_prob = stop_probs[gpos]
                        mean_frame = torch.mean(torch.stack(cds_frame_probs))
                        score = alpha * s_prob + beta * mean_frame + alpha * stop_prob

                        # PTC penalty: stop before last junction by >55 nt (tx coords)
                        if len(tx_exons) > 1:
                            last_junction_tx = sum((e.end - e.start) for e in tx_exons[:-1])
                            if tx_pos < last_junction_tx - 55:
                                score = score - gamma
                        stop_hit_score = score
                        break
                if stop_hit_score is None:
                    # no stop found; weak
                    score = (alpha * s_prob) - gamma
                    stop_hit_score = score
                best_overall = torch.maximum(best_overall, torch.maximum(stop_hit_score, torch.tensor(0.0, device=device)))

        return best_overall

    # --- Tier B: sequence-based fallback ---
    if input_ids is None:
        return torch.tensor(0.0, device=device)

    token_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
    cDNA, exon_seqs = _get_spliced_cDNA(isoform, input_ids, token_map)

    best = 0.0

    def is_strong_kozak(seq: str, pos: int) -> bool:
        # gccRccAUGG → we just check -3 in {A,G} and +3 == G
        if pos < 3 or pos + 4 >= len(seq):
            return False
        return seq[pos - 3] in "AG" and seq[pos + 3] == "G"

    min_aa = scoring_config.get("min_cds_len_aa", 50)
    max_aa = scoring_config.get("max_cds_len_aa", 10000)
    gamma = scoring_config.get("orf_gamma", 0.6)
    kozak_bonus = scoring_config.get("kozak_bonus", 0.2)

    for i in range(0, len(cDNA) - 2):
        if cDNA[i:i+3] == "ATG":
            for j in range(i, len(cDNA) - 2, 3):
                codon = cDNA[j:j+3]
                if codon in {"TAA", "TAG", "TGA"}:
                    cds_len_aa = (j - i) // 3
                    score = 0.5
                    if min_aa <= cds_len_aa <= max_aa:
                        score += 0.3
                    if is_strong_kozak(cDNA, i):
                        score += kozak_bonus
                    if len(exon_seqs) > 1:
                        last_junction_tx = sum(len(s) for s in exon_seqs[:-1])
                        if j < last_junction_tx - 55:
                            score -= gamma
                    best = max(best, score)
                    break
    return torch.tensor(max(0.0, best), device=device)

# -------------------------
# Graph construction
# -------------------------

class SpliceGraph:
    """Graph of exon nodes with junction edges."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_exon(self, exon: Exon):
        node_key = (exon.start, exon.end)
        self.graph.add_node(node_key, score=exon.score, type='exon')

    def add_junction(self, from_exon: Exon, to_exon: Exon, score: float):
        fk = (from_exon.start, from_exon.end)
        tk = (to_exon.start, to_exon.end)
        self.graph.add_edge(fk, tk, score=score, type='junction')

    def __repr__(self):
        return f"SpliceGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


class SpliceGraphBuilder:
    """
    Build a splice graph anchored by high-confidence TSS/polyA and splice peaks.
    """
    def __init__(self, config: Dict):
        if "decoder" in config:
            config = config["decoder"]
        self.config = config
        self.thresholds = self.config.get("thresholds", {})
        self.priors = self.config.get("priors", {})
        self.max_starts = self.config.get("max_starts", 8)
        self.max_ends = self.config.get("max_ends", 8)
        self.allow_unanchored = self.config.get("allow_unanchored", False)
        self.min_exon_len = self.priors.get("min_exon_len", 1)
        self.max_intron_len = self.priors.get("max_intron_len", 500000)

    def _get_exons(self, starts: List[int], ends: List[int],
                   start_scores: List[float], end_scores: List[float], strand: str) -> List[Exon]:
        exons: List[Exon] = []
        is_fwd = (strand == '+')
        for i, s_idx in enumerate(starts):
            for j, e_idx in enumerate(ends):
                if (is_fwd and s_idx < e_idx) or ((not is_fwd) and s_idx > e_idx):
                    start_coord, end_coord = (int(s_idx), int(e_idx)) if is_fwd else (int(e_idx), int(s_idx))
                    exon_len = end_coord - start_coord
                    if exon_len >= self.min_exon_len:
                        score = float((start_scores[i] + end_scores[j]) / 2.0)
                        exons.append(Exon(start=start_coord, end=end_coord, score=score))
        return exons

    def build(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> SpliceGraph:
        donor_logits    = _as_1d(head_outputs["splice"]["donor"])
        acceptor_logits = _as_1d(head_outputs["splice"]["acceptor"])

        # Optional heads may be absent; keep empty tensors on the same device
        device = donor_logits.device
        tss_raw   = head_outputs.get("tss", {}).get("tss", torch.empty(0, device=device))
        polya_raw = head_outputs.get("polya", {}).get("polya", torch.empty(0, device=device))
        tss_logits   = _as_1d(tss_raw)   if tss_raw.numel()   else tss_raw
        polya_logits = _as_1d(polya_raw) if polya_raw.numel() else polya_raw

        donor_idx, donor_scores   = _find_peaks(donor_logits,    self.thresholds.get("donor", 0.6))
        accept_idx, accept_scores = _find_peaks(acceptor_logits, self.thresholds.get("acceptor", 0.6))
        tss_idx, tss_scores       = _find_peaks(tss_logits,      self.thresholds.get("tss", 0.5),  top_k=self.max_starts) if tss_logits.numel()   else (torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device))
        pa_idx, pa_scores         = _find_peaks(polya_logits,    self.thresholds.get("polya", 0.5), top_k=self.max_ends)   if polya_logits.numel() else (torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device))

        # Internal exons are always formed by (acceptor -> donor) in genomic coords.
        # _get_exons enforces s<e for '+' and s>e for '-', so this pairing works on both strands.
        internal = self._get_exons(
            accept_idx.tolist(),
            donor_idx.tolist(),
            accept_scores.tolist(),
            donor_scores.tolist(),
            strand,
        )

        first   = self._get_exons(tss_idx.tolist(), donor_idx.tolist(),
                                  tss_scores.tolist(), donor_scores.tolist(), strand)
        last    = self._get_exons(accept_idx.tolist(), pa_idx.tolist(),
                                  accept_scores.tolist(), pa_scores.tolist(), strand)
        single  = self._get_exons(tss_idx.tolist(), pa_idx.tolist(),
                                  tss_scores.tolist(), pa_scores.tolist(), strand)

        candidates = []
        if self.allow_unanchored:
            candidates.extend(internal)
        candidates.extend(first); candidates.extend(last); candidates.extend(single)

        # dedupe by (start,end) keep highest score
        unique = {(e.start, e.end): e for e in sorted(candidates, key=lambda x: x.score, reverse=True)}
        exons_list = list(unique.values())

        graph = SpliceGraph()
        for ex in exons_list:
            graph.add_exon(ex)

        donor_set = set(donor_idx.tolist())
        accept_set = set(accept_idx.tolist())
        sorted_tx = _order_exons_by_transcription(exons_list, strand)

        for i, up in enumerate(sorted_tx):
            for j in range(i + 1, len(sorted_tx)):
                dn = sorted_tx[j]
                if strand == '+':
                    intron_len = dn.start - up.end
                    if 0 < intron_len <= self.max_intron_len:
                        if (up.end in donor_set) and (dn.start in accept_set):
                            score = (up.score + dn.score) / 2.0
                            graph.add_junction(up, dn, score=score)
                else:
                    intron_len = up.start - dn.end
                    if 0 < intron_len <= self.max_intron_len:
                        if (up.start in donor_set) and (dn.end in accept_set):
                            score = (up.score + dn.score) / 2.0
                            graph.add_junction(up, dn, score=score)
        return graph


class IsoformEnumerator:
    """Beam-search enumeration of candidate exon paths."""
    def __init__(self, config: Dict):
        self.config = config
        self.beam_size = self.config.get("beam_size", 16)

    def enumerate(self, graph: SpliceGraph, max_paths: int, strand: str = '+') -> List[Isoform]:
        if not graph.graph or graph.graph.number_of_nodes() == 0:
            return []

        source_nodes = [n for n, deg in graph.graph.in_degree() if deg == 0]
        if not source_nodes:
            return []

        src_exons = [Exon(start=n[0], end=n[1], score=graph.graph.nodes[n]['score']) for n in source_nodes]
        sorted_src = _order_exons_by_transcription(src_exons, strand)
        beam = [(graph.graph.nodes[(e.start, e.end)]['score'], [(e.start, e.end)]) for e in sorted_src]
        beam.sort(key=lambda x: x[0], reverse=True)

        all_paths: List[Tuple[float, List[Tuple[int,int]]]] = []

        while beam:
            all_paths.extend(beam)
            new_beam: List[Tuple[float, List[Tuple[int,int]]]] = []
            for score, path in beam:
                last = path[-1]
                for nb in graph.graph.neighbors(last):
                    node_score = graph.graph.nodes[nb]['score']
                    new_beam.append((score + node_score, path + [nb]))
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[: self.beam_size]
            if not beam:
                break

        best_by_path: Dict[Tuple[Tuple[int,int], ...], float] = {}
        for score, path in all_paths:
            pt = tuple(path)
            if pt not in best_by_path or score > best_by_path[pt]:
                best_by_path[pt] = score

        sorted_paths = sorted(best_by_path.items(), key=lambda kv: kv[1] / len(kv[0]), reverse=True)

        isoforms: List[Isoform] = []
        for path_tuple, score in sorted_paths[: max_paths]:
            exs = []
            for (s, e) in path_tuple:
                node_data = graph.graph.nodes[(s, e)]
                exs.append(Exon(start=s, end=e, score=float(node_data['score'])))
            norm_score = score / max(1, len(path_tuple))
            isoforms.append(Isoform(exons=exs, strand=strand, score=float(norm_score)))
        return isoforms


def _score_length(isoform: Isoform, priors: Dict) -> torch.Tensor:
    num_exons = len(isoform.exons)
    if num_exons < 2 or num_exons > 20:
        return torch.tensor(-0.5)
    return torch.tensor(0.0)


class IsoformScorer(nn.Module):
    """Learnable scorer over structural and ORF evidence."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.scoring_config = self.config.get("scoring", {})
        self.priors = self.config.get("priors", {})

        self.w_spl = nn.Parameter(torch.tensor(self.scoring_config.get("w_spl", 1.0)))
        self.w_tss = nn.Parameter(torch.tensor(self.scoring_config.get("w_tss", 0.4)))
        self.w_pa  = nn.Parameter(torch.tensor(self.scoring_config.get("w_pa", 0.4)))
        self.w_orf = nn.Parameter(torch.tensor(self.scoring_config.get("w_orf", 0.8)))
        self.w_len = nn.Parameter(torch.tensor(self.scoring_config.get("w_len", 0.1)))

    def forward(self, isoform: Isoform, head_outputs: Dict[str, torch.Tensor], input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.w_spl.device

        donor_logits    = _as_1d(head_outputs["splice"]["donor"])
        acceptor_logits = _as_1d(head_outputs["splice"]["acceptor"])

        # Splice score (average of junction site logits on the path)
        if len(isoform.exons) > 1:
            if isoform.strand == '+':
                donor_idx = [ex.end for ex in isoform.exons[:-1]]
                accept_idx = [ex.start for ex in isoform.exons[1:]]
            else:
                donor_idx = [ex.start for ex in isoform.exons[:-1]]
                accept_idx = [ex.end for ex in isoform.exons[1:]]
            s_spl = (donor_logits[donor_idx].sum() + acceptor_logits[accept_idx].sum()) / (len(donor_idx) + len(accept_idx))
        else:
            s_spl = torch.tensor(0.0, device=device)

        # TSS score: peak near transcript 5' end
        tss_raw = head_outputs.get("tss", {}).get("tss", None)
        s_tss = torch.tensor(0.0, device=device)
        if tss_raw is not None:
            tss_logits = _as_1d(tss_raw)
            if isoform.exons:
                first = isoform.exons[0]
                start_bin = first.end if isoform.strand == '-' else first.start
                s_tss = _get_peak_log_prob(tss_logits, start_bin, self.config.get("tss_pa_window", 1)).to(device)

        # polyA score: peak near transcript 3' end
        pa_raw = head_outputs.get("polya", {}).get("polya", None)
        s_pa = torch.tensor(0.0, device=device)
        if pa_raw is not None:
            pa_logits = _as_1d(pa_raw)
            if isoform.exons:
                last = isoform.exons[-1]
                end_bin = last.start if isoform.strand == '-' else last.end
                s_pa = _get_peak_log_prob(pa_logits, end_bin, self.config.get("tss_pa_window", 1)).to(device)

        # ORF score
        s_orf = _score_orf(isoform, head_outputs, self.scoring_config, input_ids=input_ids).to(device)

        # Length prior
        s_len = _score_length(isoform, self.priors).to(device)

        total = (
            self.w_spl * s_spl +
            self.w_tss * s_tss +
            self.w_pa  * s_pa  +
            self.w_orf * s_orf +
            self.w_len * s_len
        )
        return total


class IsoformDecoder(nn.Module):
    """
    End-to-end decoder orchestrating:
      - graph construction
      - beam enumeration
      - learnable scoring
    """
    def __init__(self, config: Dict):
        super().__init__()
        decoder_config = config["decoder"] if "decoder" in config else config
        self.config = {"decoder": decoder_config}
        self.graph_builder = SpliceGraphBuilder(decoder_config)
        self.enumerator    = IsoformEnumerator(decoder_config)
        self.scorer        = IsoformScorer(decoder_config)

    def forward(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        return self.decode(head_outputs, strand=strand, input_ids=input_ids)

    def decode(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        graph = self.graph_builder.build(head_outputs, strand=strand)
        max_candidates = self.config.get("decoder", {}).get("max_candidates", 64)
        candidates = self.enumerator.enumerate(graph, max_paths=max_candidates, strand=strand)

        # score & order
        for iso in candidates:
            iso.exons = _order_exons_by_transcription(iso.exons, iso.strand)
            iso.score = float(self.scorer(iso, head_outputs, input_ids=input_ids).item())
        candidates.sort(key=lambda z: z.score, reverse=True)
        return candidates