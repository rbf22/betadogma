"""
decoder/isoform_decoder.py
--------------------------
End-to-end isoform decoder compatible with BetaDogmaModel.

Pipeline:
  1) SpliceGraphBuilder: build a graph of candidate exons from head logits
  2) IsoformEnumerator:  beam-search paths (candidate isoforms)
  3) IsoformScorer:      score/rank each isoform (learnable)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from .types import Isoform, Exon  # assumes Exon has fields: start:int, end:int


# ---------- helpers

def _safe_squeeze(x: torch.Tensor) -> torch.Tensor:
    """Squeeze last dim if it's singleton; keep at least 1D."""
    if x.dim() == 0:
        return x.view(1)
    if x.size(-1) == 1:
        return x[..., 0]
    return x


def _find_peaks(logits: torch.Tensor, threshold: float, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns indices and probabilities of positions where sigmoid(logit) > threshold.
    Keeps top_k by prob if provided.
    """
    if logits is None or logits.numel() == 0:
        device = logits.device if isinstance(logits, torch.Tensor) else torch.device("cpu")
        z = torch.tensor([], device=device, dtype=torch.long)
        return z, z.to(dtype=torch.float32)

    probs = torch.sigmoid(_safe_squeeze(logits))  # (L,)
    mask = probs > threshold
    if not torch.any(mask):
        z = torch.tensor([], device=probs.device, dtype=torch.long)
        return z, z.to(dtype=torch.float32)

    peak_idx = torch.nonzero(mask, as_tuple=False).view(-1)   # (K,)
    peak_probs = probs[peak_idx]

    if top_k is not None and peak_idx.numel() > top_k:
        topk_vals, topk_idx = torch.topk(peak_probs, k=top_k)
        peak_idx = peak_idx[topk_idx]
        peak_probs = topk_vals

    return peak_idx, peak_probs


def _order_exons_by_transcription(exons: List[Exon], strand: str) -> List[Exon]:
    """Sorts a list of exons in transcription order."""
    exons_sorted = sorted(exons, key=lambda e: (e.start, e.end))
    return exons_sorted if strand == '+' else list(reversed(exons_sorted))


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.sigmoid(x) + 1e-9)


def _get_peak_log_prob(logits: torch.Tensor, index: int, window: int) -> torch.Tensor:
    """Max log-probability in a small window around index (inclusive)."""
    if logits is None or index is None:
        return torch.tensor(0.0, device=logits.device if isinstance(logits, torch.Tensor) else "cpu")
    x = _safe_squeeze(logits)  # (L,)
    start = max(0, int(index) - window)
    end = min(x.shape[0], int(index) + window + 1)
    if start >= end:
        return torch.tensor(0.0, device=x.device)
    return torch.max(_log_sigmoid(x[start:end]))


def _ids_to_seq(input_ids: Optional[torch.Tensor], token_map: Dict[int, str]) -> Optional[str]:
    if input_ids is None:
        return None
    ids = input_ids.detach().cpu().view(-1).tolist()
    return "".join(token_map.get(int(i), "N") for i in ids)


def _revcomp(seq: str) -> str:
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(comp)[::-1]


def _get_spliced_cDNA_tx_order(isoform: Isoform, genomic_seq: str) -> Tuple[str, List[str]]:
    """
    Build cDNA from exon sequences in *transcription order* (isoform.exons is assumed
    to be in tx order when called). Also returns the list of exon sequences in that order.
    """
    exon_seqs = [genomic_seq[e.start:e.end] for e in isoform.exons]
    cdna = "".join(exon_seqs)
    if isoform.strand == '-':
        cdna = _revcomp(cdna)
    return cdna, exon_seqs


# ---------- Graph objects

class SpliceGraph:
    """
    Graph over candidate exons. Nodes are (start, end) with attribute 'score'.
    Directed edges represent plausible splice junctions.
    """
    def __init__(self):
        self.g = nx.DiGraph()

    def add_exon(self, start: int, end: int, score: float):
        self.g.add_node((int(start), int(end)), score=float(score), type='exon')

    def add_junction(self, up: Tuple[int, int], down: Tuple[int, int], score: float):
        self.g.add_edge(up, down, score=float(score), type='junction')

    def num_nodes(self) -> int:
        return self.g.number_of_nodes()

    def neighbors(self, node):
        return self.g.neighbors(node)

    def node_score(self, node) -> float:
        return float(self.g.nodes[node]['score'])

    def nodes(self):
        return list(self.g.nodes)

    def in_degree_zero(self):
        return [n for n, d in self.g.in_degree() if d == 0]


# ---------- Builder

class SpliceGraphBuilder:
    """
    Builds a splice graph from head outputs. Anchors candidates by TSS/polyA.
    Config:
      thresholds: dict with keys donor/acceptor/tss/polya (sigmoid thresholds)
      priors:     dict with min_exon_len, max_intron_len
      max_starts / max_ends: top-k anchors
      allow_unanchored: include internal exons w/out TSS/polyA anchors
    """
    def __init__(self, config: Dict):
        if "decoder" in config:
            config = config["decoder"]
        self.config = config
        self.thresholds = self.config.get("thresholds", {})
        self.priors = self.config.get("priors", {})
        self.max_starts = int(self.config.get("max_starts", 8))
        self.max_ends = int(self.config.get("max_ends", 8))
        self.allow_unanchored = bool(self.config.get("allow_unanchored", False))
        self.min_exon_len = int(self.priors.get("min_exon_len", 1))
        self.max_intron_len = int(self.priors.get("max_intron_len", 500000))

    def _pair_to_exons(
        self,
        starts: torch.Tensor, start_scores: torch.Tensor,
        ends: torch.Tensor,   end_scores: torch.Tensor,
        strand: str
    ) -> List[Tuple[int, int, float]]:
        """
        Combine starts/ends into exon (start,end,score) tuples with strand-aware ordering.
        """
        is_fwd = strand == '+'
        exons: List[Tuple[int, int, float]] = []
        for i, s_idx in enumerate(starts.tolist()):
            for j, e_idx in enumerate(ends.tolist()):
                if (is_fwd and s_idx < e_idx) or ((not is_fwd) and s_idx > e_idx):
                    start_coord, end_coord = (int(s_idx), int(e_idx)) if is_fwd else (int(e_idx), int(s_idx))
                    exon_len = end_coord - start_coord
                    if exon_len >= self.min_exon_len:
                        score = float((start_scores[i] + end_scores[j]) / 2.0)
                        exons.append((start_coord, end_coord, score))
        return exons

    def build(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+') -> SpliceGraph:
        # Extract raw logits (expected shapes from heads.py)
        donor_logits    = head_outputs["splice"]["donor"]
        acceptor_logits = head_outputs["splice"]["acceptor"]
        tss_logits      = head_outputs.get("tss", {}).get("tss", None)
        polya_logits    = head_outputs.get("polya", {}).get("polya", None)

        donor_idx, donor_sc     = _find_peaks(donor_logits,    self.thresholds.get("donor",    0.6))
        acceptor_idx, accept_sc = _find_peaks(acceptor_logits, self.thresholds.get("acceptor", 0.6))
        tss_idx, tss_sc         = _find_peaks(tss_logits,      self.thresholds.get("tss",      0.5), top_k=self.max_starts) if tss_logits is not None else (torch.tensor([], dtype=torch.long, device=donor_idx.device), torch.tensor([], device=donor_idx.device))
        polya_idx, polya_sc     = _find_peaks(polya_logits,    self.thresholds.get("polya",    0.5), top_k=self.max_ends)   if polya_logits is not None else (torch.tensor([], dtype=torch.long, device=donor_idx.device), torch.tensor([], device=donor_idx.device))

        # Candidate exon types
        internal_exons = self._pair_to_exons(acceptor_idx, accept_sc, donor_idx,   donor_sc,   strand)
        first_exons    = self._pair_to_exons(tss_idx,      tss_sc,    donor_idx,   donor_sc,   strand)
        last_exons     = self._pair_to_exons(acceptor_idx, accept_sc, polya_idx,   polya_sc,   strand)
        single_exons   = self._pair_to_exons(tss_idx,      tss_sc,    polya_idx,   polya_sc,   strand)

        candidates: List[Tuple[int, int, float]] = []
        if self.allow_unanchored:
            candidates.extend(internal_exons)
        candidates.extend(first_exons)
        candidates.extend(last_exons)
        candidates.extend(single_exons)

        # Deduplicate by (start,end) keeping highest score
        best_by_span: Dict[Tuple[int, int], float] = {}
        for s, e, sc in candidates:
            key = (s, e)
            if key not in best_by_span or sc > best_by_span[key]:
                best_by_span[key] = sc

        g = SpliceGraph()
        for (s, e), sc in best_by_span.items():
            g.add_exon(s, e, sc)

        # Add junctions where donor at upstream boundary and acceptor at downstream boundary exist
        donor_set    = set(donor_idx.tolist())
        acceptor_set = set(acceptor_idx.tolist())

        # Sort exons in transcription order to wire edges in-order
        exons_sorted = _order_exons_by_transcription([Exon(start=s, end=e) for (s, e) in best_by_span.keys()], strand)
        exon_nodes = [(ex.start, ex.end) for ex in exons_sorted]

        for i, up in enumerate(exon_nodes):
            for j in range(i + 1, len(exon_nodes)):
                down = exon_nodes[j]
                if strand == '+':
                    intron_len = down[0] - up[1]
                    if 0 < intron_len <= self.max_intron_len and (up[1] in donor_set) and (down[0] in acceptor_set):
                        g.add_junction(up, down, score=(g.node_score(up) + g.node_score(down)) / 2.0)
                else:
                    intron_len = up[0] - down[1]
                    if 0 < intron_len <= self.max_intron_len and (up[0] in donor_set) and (down[1] in acceptor_set):
                        g.add_junction(up, down, score=(g.node_score(up) + g.node_score(down)) / 2.0)

        return g


# ---------- Enumerator

class IsoformEnumerator:
    """
    Enumerate paths through the splice graph with a simple beam search.
    Keeps all visited beams as candidates; deduplicates at the end.
    """
    def __init__(self, config: Dict):
        self.beam_size = int(config.get("beam_size", 16))

    def enumerate(self, graph: SpliceGraph, max_paths: int, strand: str = '+') -> List[Isoform]:
        if graph.num_nodes() == 0:
            return []

        sources = graph.in_degree_zero()
        if not sources:
            return []

        # order sources by tx direction
        src_exons = [Exon(start=s[0], end=s[1]) for s in sources]
        src_ordered = _order_exons_by_transcription(src_exons, strand)
        src_nodes = [(e.start, e.end) for e in src_ordered]

        # Beam: list[(score, path_nodes)]
        beam = [(graph.node_score(n), [n]) for n in src_nodes]
        beam.sort(key=lambda t: t[0], reverse=True)

        candidates: List[Tuple[float, List[Tuple[int, int]]]] = []

        while beam:
            candidates.extend(beam)
            new_beam: List[Tuple[float, List[Tuple[int, int]]]] = []
            for score, path in beam:
                last = path[-1]
                for nxt in graph.neighbors(last):
                    new_score = score + graph.node_score(nxt)
                    new_path = path + [nxt]
                    new_beam.append((new_score, new_path))
            new_beam.sort(key=lambda t: t[0], reverse=True)
            beam = new_beam[: self.beam_size]
            if not beam:
                break

        # Deduplicate paths; prefer higher score
        best_by_path: Dict[Tuple[Tuple[int, int], ...], float] = {}
        for sc, p in candidates:
            key = tuple(p)
            if key not in best_by_path or sc > best_by_path[key]:
                best_by_path[key] = sc

        # Normalize by length and take top max_paths
        ranked = sorted(
            ((p, sc / max(1, len(p))) for p, sc in best_by_path.items()),
            key=lambda t: t[1], reverse=True
        )[: max_paths]

        # Convert to Isoform objects (score filled later by scorer)
        out: List[Isoform] = []
        for path_nodes, _ in ranked:
            exons = [Exon(start=s, end=e) for (s, e) in path_nodes]
            out.append(Isoform(exons=exons, strand=strand, score=0.0))
        return out


# ---------- Scoring

def _score_length_prior(isoform: Isoform, priors: Dict) -> torch.Tensor:
    # Very light prior: n_exons outside [2, 20] gets a small penalty
    n = len(isoform.exons)
    return torch.tensor(0.0) if 2 <= n <= 20 else torch.tensor(-0.5)


def _sequence_orf_score(cdna: str, kozak_bonus: float, min_aa: int, max_aa: int, ptc_penalty: float, exon_seqs: List[str]) -> float:
    best = -1.0

    def strong_kozak(s: str, pos: int) -> bool:
        return (pos >= 3 and pos + 4 < len(s) and s[pos - 3] in "AG" and s[pos + 3] == "G")

    for i in range(len(cdna) - 2):
        if cdna[i:i+3] == "ATG":
            for j in range(i + 3, len(cdna) - 2, 3):
                trip = cdna[j:j+3]
                if trip in {"TAA", "TAG", "TGA"}:
                    aa_len = (j - i) // 3
                    s = 0.5
                    if min_aa <= aa_len <= max_aa:
                        s += 0.3
                    if strong_kozak(cdna, i):
                        s += kozak_bonus

                    if len(exon_seqs) > 1:
                        last_junction_tx = sum(len(x) for x in exon_seqs[:-1])
                        if j < last_junction_tx - 55:
                            s -= ptc_penalty

                    best = max(best, s)
                    break  # stop scanning this start
    return max(0.0, best)


class IsoformScorer(nn.Module):
    """
    Learnable mixture of:
      - splice evidence at junctions
      - TSS/polyA evidence at transcript ends
      - ORF evidence (head-driven or sequence fallback)
      - mild length prior
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.sc = self.cfg.get("scoring", {})
        self.priors = self.cfg.get("priors", {})

        # Learnable weights
        self.w_spl = nn.Parameter(torch.tensor(float(self.sc.get("w_spl", 1.0))))
        self.w_tss = nn.Parameter(torch.tensor(float(self.sc.get("w_tss", 0.4))))
        self.w_pa  = nn.Parameter(torch.tensor(float(self.sc.get("w_pa",  0.4))))
        self.w_orf = nn.Parameter(torch.tensor(float(self.sc.get("w_orf", 0.8))))
        self.w_len = nn.Parameter(torch.tensor(float(self.sc.get("w_len", 0.1))))

    def forward(self, isoform: Isoform, head_outputs: Dict[str, torch.Tensor], input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.w_spl.device

        # --- 1) splice evidence at junctions
        donor_logits    = head_outputs["splice"]["donor"].squeeze(-1)  # (B?,L) or (L,)
        acceptor_logits = head_outputs["splice"]["acceptor"].squeeze(-1)
        donor_logits = donor_logits[0] if donor_logits.dim() == 2 else donor_logits
        acceptor_logits = acceptor_logits[0] if acceptor_logits.dim() == 2 else acceptor_logits

        if len(isoform.exons) > 1:
            if isoform.strand == '+':
                donor_pos = [e.end for e in isoform.exons[:-1]]
                accept_pos = [e.start for e in isoform.exons[1:]]
            else:
                donor_pos = [e.start for e in isoform.exons[:-1]]
                accept_pos = [e.end for e in isoform.exons[1:]]
            s_spl = (donor_logits[donor_pos].mean() + acceptor_logits[accept_pos].mean()) / 2.0
        else:
            s_spl = torch.tensor(0.0, device=donor_logits.device)

        # --- 2) TSS/polyA evidence at transcript ends
        win = int(self.cfg.get("tss_pa_window", 1))

        s_tss = torch.tensor(0.0, device=device)
        tss_logits = head_outputs.get("tss", {}).get("tss", None)
        if tss_logits is not None and isoform.exons:
            tss_logits = tss_logits.squeeze(-1)
            tss_logits = tss_logits[0] if tss_logits.dim() == 2 else tss_logits
            first = isoform.exons[0]
            start_bin = first.end if isoform.strand == '-' else first.start
            s_tss = _get_peak_log_prob(tss_logits, start_bin, win).to(device)

        s_pa = torch.tensor(0.0, device=device)
        polya_logits = head_outputs.get("polya", {}).get("polya", None)
        if polya_logits is not None and isoform.exons:
            polya_logits = polya_logits.squeeze(-1)
            polya_logits = polya_logits[0] if polya_logits.dim() == 2 else polya_logits
            last = isoform.exons[-1]
            end_bin = last.start if isoform.strand == '-' else last.end
            s_pa = _get_peak_log_prob(polya_logits, end_bin, win).to(device)

        # --- 3) ORF evidence
        use_orf_head = bool(self.sc.get("use_orf_head", True))
        s_orf = torch.tensor(0.0, device=device)

        if use_orf_head and "orf" in head_outputs:
            # head-driven ORF mix
            start_logits = head_outputs["orf"]["start"].squeeze(-1)
            stop_logits  = head_outputs["orf"]["stop"].squeeze(-1)
            frame_logits = head_outputs["orf"]["frame"]            # (B?,L,3)
            start_logits = start_logits[0] if start_logits.dim() == 2 else start_logits
            stop_logits  = stop_logits[0]  if stop_logits.dim()  == 2 else stop_logits
            frame_logits = frame_logits[0] if frame_logits.dim() == 3 else frame_logits

            start_p = torch.sigmoid(start_logits)
            stop_p  = torch.sigmoid(stop_logits)
            frame_p = torch.softmax(frame_logits, dim=-1)          # (L,3)

            # Flatten exonic positions in transcription order
            tx_positions = [p for ex in isoform.exons for p in range(ex.start, ex.end)]
            if tx_positions:
                # start candidates within exons
                ex_start = start_p[tx_positions]
                k = min(int(self.sc.get("max_start_candidates", 5)), ex_start.numel())
                if k > 0:
                    cand_vals, cand_idx = torch.topk(ex_start, k=k)
                    best = -1e9
                    alpha = float(self.sc.get("orf_alpha", 0.5))
                    beta  = float(self.sc.get("orf_beta",  0.3))
                    gamma = float(self.sc.get("orf_gamma", 0.6))

                    last_junc_tx = sum((ex.end - ex.start) for ex in isoform.exons[:-1]) if len(isoform.exons) > 1 else None

                    for s_val, rel_i in zip(cand_vals, cand_idx):
                        s_tx = rel_i.item()
                        for frame_off in (0, 1, 2):
                            cds_probs = []
                            for tx_i in range(s_tx + frame_off, len(tx_positions), 3):
                                gpos = tx_positions[tx_i]
                                cds_probs.append(frame_p[gpos, (tx_i - s_tx) % 3])
                                if stop_p[gpos] > 0.5:
                                    mean_frame = torch.stack(cds_probs).mean()
                                    score = alpha * s_val + beta * mean_frame + alpha * stop_p[gpos]
                                    if last_junc_tx is not None and tx_i < last_junc_tx - 55:
                                        score = score - gamma
                                    best = max(best, float(score))
                                    break
                            else:
                                # no stop: mild penalty
                                best = max(best, float(alpha * s_val - gamma))
                    if best > -1e9:
                        s_orf = torch.tensor(best, device=device)
        else:
            # sequence fallback (requires input_ids)
            if input_ids is not None:
                token_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
                seq = _ids_to_seq(input_ids, token_map)
                if seq is not None:
                    cdna, exon_seqs = _get_spliced_cDNA_tx_order(isoform, seq)
                    s_orf = torch.tensor(
                        _sequence_orf_score(
                            cdna,
                            kozak_bonus=float(self.sc.get("kozak_bonus", 0.2)),
                            min_aa=int(self.sc.get("min_cds_len_aa", 50)),
                            max_aa=int(self.sc.get("max_cds_len_aa", 10000)),
                            ptc_penalty=float(self.sc.get("orf_gamma", 0.6)),
                            exon_seqs=exon_seqs,
                        ),
                        device=device,
                    )

        # --- 4) length prior
        s_len = _score_length_prior(isoform, self.priors).to(device)

        return (self.w_spl * s_spl) + (self.w_tss * s_tss) + (self.w_pa * s_pa) + (self.w_orf * s_orf) + (self.w_len * s_len)


# ---------- Orchestrator

class IsoformDecoder(nn.Module):
    """
    Orchestrates graph building, path enumeration, and scoring.
    """
    def __init__(self, config: Dict):
        super().__init__()
        # Support full config dict or decoder-only
        decoder_cfg = config.get("decoder", config)
        self.config = {"decoder": decoder_cfg}
        self.graph_builder = SpliceGraphBuilder(decoder_cfg)
        self.enumerator = IsoformEnumerator(decoder_cfg)
        self.scorer = IsoformScorer(decoder_cfg)

    def forward(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        return self.decode(head_outputs, strand=strand, input_ids=input_ids)

    def decode(self, head_outputs: Dict[str, torch.Tensor], strand: str = '+', input_ids: Optional[torch.Tensor] = None) -> List[Isoform]:
        graph = self.graph_builder.build(head_outputs, strand=strand)
        max_candidates = int(self.config.get("decoder", {}).get("max_candidates", 64))
        candidates = self.enumerator.enumerate(graph, max_paths=max_candidates, strand=strand)

        # Ensure exons are in tx order and compute final score
        for iso in candidates:
            iso.exons = _order_exons_by_transcription(iso.exons, iso.strand)
            iso.score = float(self.scorer(iso, head_outputs, input_ids=input_ids).item())

        # Rank by score desc
        candidates.sort(key=lambda z: z.score, reverse=True)
        return candidates