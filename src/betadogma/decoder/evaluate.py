# src/betadogma/decoder/evaluate.py
"""
Evaluation utilities for the isoform decoder.

Metrics provided:
- exon_chain_match(): exact exon chain match (optionally with slop)
- junction_f1(): splice junction precision/recall/F1 (optionally with slop)
- top_k_recall(): whether a GT isoform appears in top-K predictions
- evaluate_set(): aggregate metrics over a dataset
"""

from __future__ import annotations
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass

from .types import Isoform, Exon


# -----------------------
# Helpers
# -----------------------

def _tx_sorted_exons(iso: Isoform) -> List[Exon]:
    """Return exons sorted in transcription order for the isoform's strand."""
    exons = sorted(iso.exons, key=lambda e: (e.start, e.end))
    return exons if iso.strand == "+" else list(reversed(exons))


def _exon_pairs(iso: Isoform) -> List[Tuple[int, int]]:
    """Exons as (start,end) in transcription order."""
    return [(e.start, e.end) for e in _tx_sorted_exons(iso)]


def _near(a: int, b: int, slop_bp: int) -> bool:
    return abs(a - b) <= slop_bp


# -----------------------
# Core primitives
# -----------------------

def get_junctions(isoform: Isoform, slop_bp: int = 0) -> Set[Tuple[int, int]]:
    """
    Extract splice junctions as pairs (donor, acceptor) in genomic coordinates.
    Junctions are derived from transcription-ordered exons:
        junction between exon i and i+1 is (exon_i.end, exon_{i+1}.start)
    """
    juncs: Set[Tuple[int, int]] = set()
    exons = _tx_sorted_exons(isoform)
    for i in range(len(exons) - 1):
        donor = exons[i].end
        acceptor = exons[i + 1].start
        juncs.add((donor, acceptor))
    return juncs


def exon_chain_match(
    predicted: Isoform,
    ground_truth: Isoform,
    slop_bp: int = 0,
) -> bool:
    """
    Exact exon chain match in transcription order.
    If slop_bp > 0, consider exon boundaries matching if both start and end
    differ by <= slop_bp.
    """
    p = _exon_pairs(predicted)
    t = _exon_pairs(ground_truth)
    if len(p) != len(t):
        return False
    if slop_bp == 0:
        return p == t
    # slop-aware comparison
    for (ps, pe), (ts, te) in zip(p, t):
        if not (_near(ps, ts, slop_bp) and _near(pe, te, slop_bp)):
            return False
    return True


def junction_f1(
    predicted: Isoform,
    ground_truth: Isoform,
    slop_bp: int = 0,
) -> Tuple[float, float, float]:
    """
    Junction-level precision/recall/F1.
    With slop_bp > 0, a predicted junction matches a GT junction if both donor
    and acceptor positions are within slop_bp of a GT junction.
    """
    pj = list(get_junctions(predicted))
    tj = list(get_junctions(ground_truth))

    # Handle single-exon vs single-exon
    if not tj and not pj:
        return (1.0, 1.0, 1.0)
    # One has junctions, the other doesn’t
    if (not tj and pj) or (tj and not pj):
        return (0.0, 0.0, 0.0)

    # Exact or slop-aware matching
    matched_gt = set()
    tp = 0
    for d_pred, a_pred in pj:
        hit = False
        for idx, (d_true, a_true) in enumerate(tj):
            if idx in matched_gt:
                continue
            if slop_bp == 0:
                cond = (d_pred == d_true) and (a_pred == a_true)
            else:
                cond = _near(d_pred, d_true, slop_bp) and _near(a_pred, a_true, slop_bp)
            if cond:
                matched_gt.add(idx)
                tp += 1
                hit = True
                break
        # if no hit, it’s a FP (counted implicitly below)

    fp = max(0, len(pj) - tp)
    fn = max(0, len(tj) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def top_k_recall(
    predicted_isoforms: List[Isoform],
    ground_truth: Isoform,
    k: int,
    slop_bp: int = 0,
) -> bool:
    """
    True if an exon-chain match (optionally slop-aware) is found within top-K.
    Assumes predicted_isoforms are sorted by score descending.
    """
    K = min(k, len(predicted_isoforms))
    for i in range(K):
        if exon_chain_match(predicted_isoforms[i], ground_truth, slop_bp=slop_bp):
            return True
    return False


# -----------------------
# Batch evaluation
# -----------------------

@dataclass
class EvalSummary:
    n: int
    exact_chain_acc: float
    junc_precision: float
    junc_recall: float
    junc_f1: float
    topk_recall: float


def evaluate_set(
    preds: List[List[Isoform]],
    gts: List[Isoform],
    k: int = 5,
    slop_bp: int = 0,
) -> EvalSummary:
    """
    Evaluate a dataset:
      preds: list over examples, each a list of candidate isoforms (sorted by score)
      gts:   list of ground-truth isoforms (same length as preds)
      k:     top-K for recall
      slop_bp: coordinate tolerance for matches
    """
    assert len(preds) == len(gts), "preds and gts must be the same length"

    n = len(gts)
    chain_hits = 0
    p_sum = r_sum = f_sum = 0.0
    topk_hits = 0

    for p_list, gt in zip(preds, gts):
        best_pred = p_list[0] if p_list else Isoform(exons=[], strand=gt.strand)  # sentinel if empty

        if exon_chain_match(best_pred, gt, slop_bp=slop_bp):
            chain_hits += 1

        pr, rc, f1 = junction_f1(best_pred, gt, slop_bp=slop_bp)
        p_sum += pr; r_sum += rc; f_sum += f1

        if top_k_recall(p_list, gt, k=k, slop_bp=slop_bp):
            topk_hits += 1

    return EvalSummary(
        n=n,
        exact_chain_acc=chain_hits / n if n else 0.0,
        junc_precision=p_sum / n if n else 0.0,
        junc_recall=r_sum / n if n else 0.0,
        junc_f1=f_sum / n if n else 0.0,
        topk_recall=topk_hits / n if n else 0.0,
    )