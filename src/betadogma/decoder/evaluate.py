"""
Evaluation utilities for the isoform decoder.

This module provides functions to compare predicted isoforms against a
ground truth, calculating metrics such as:
- Exact exon chain match rate
- Splice junction F1 score
- Top-K recall
"""

from typing import List, Set, Tuple

from .types import Isoform


def get_junctions(isoform: Isoform) -> Set[Tuple[int, int]]:
    """Extracts a set of splice junctions from an isoform."""
    junctions = set()
    if len(isoform.exons) > 1:
        for i in range(len(isoform.exons) - 1):
            # Junction is from the end of one exon to the start of the next
            junctions.add((isoform.exons[i].end, isoform.exons[i+1].start))
    return junctions


def exon_chain_match(predicted: Isoform, ground_truth: Isoform) -> bool:
    """
    Checks if the predicted isoform has the exact same exon chain as the
    ground truth.

    Returns:
        True if the exon start/end coordinates match exactly, False otherwise.
    """
    if len(predicted.exons) != len(ground_truth.exons):
        return False

    pred_exons = sorted([(e.start, e.end) for e in predicted.exons])
    true_exons = sorted([(e.start, e.end) for e in ground_truth.exons])

    return pred_exons == true_exons


def junction_f1(predicted: Isoform, ground_truth: Isoform) -> Tuple[float, float, float]:
    """
    Calculates the F1 score for splice junctions.

    Returns:
        A tuple containing (precision, recall, f1_score).
    """
    pred_junctions = get_junctions(predicted)
    true_junctions = get_junctions(ground_truth)

    if not true_junctions and not pred_junctions:
        return (1.0, 1.0, 1.0) # Both are single-exon, perfect match
    if not true_junctions or not pred_junctions:
        return (0.0, 0.0, 0.0) # One is single-exon, the other is not

    tp = len(pred_junctions.intersection(true_junctions))
    fp = len(pred_junctions.difference(true_junctions))
    fn = len(true_junctions.difference(pred_junctions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def top_k_recall(predicted_isoforms: List[Isoform], ground_truth: Isoform, k: int) -> bool:
    """
    Checks if the ground truth isoform is present in the top-K predicted isoforms.

    Args:
        predicted_isoforms: A list of predicted isoforms, sorted by score.
        ground_truth: The ground truth isoform.
        k: The number of top predictions to consider.

    Returns:
        True if a perfect match is found within the top K, False otherwise.
    """
    for i in range(min(k, len(predicted_isoforms))):
        if exon_chain_match(predicted_isoforms[i], ground_truth):
            return True
    return False