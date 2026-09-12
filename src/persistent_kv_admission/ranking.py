"""Dependency-free binary ranking metrics with explicit tie handling."""

from __future__ import annotations

import math

import numpy as np


def _score_groups(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return descending order and inclusive score-group starts/ends."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    if len(order) == 0:
        return order, np.array([], dtype=int), np.array([], dtype=int)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    return order, starts, ends


def ranking_metrics(
    labels: np.ndarray, scores: np.ndarray, k: int = 100
) -> tuple[float, float, float]:
    """Compute ROC AUC, threshold AP, and expected precision@k.

    AP is integrated at distinct score thresholds. Precision@k assigns the
    expected positive fraction of a tied score group when k cuts through it.
    These definitions avoid arbitrary hash-ID ordering deciding tied results.
    """
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have equal shapes")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if len(labels) == 0:
        return math.nan, math.nan, math.nan

    order, starts, ends = _score_groups(scores)
    sorted_labels = labels[order]
    group_pos = np.add.reduceat(sorted_labels, starts).astype(float)
    group_size = (ends - starts).astype(float)

    if positives and negatives:
        # Pair counting from low to high score, with half credit for ties.
        neg_below = 0.0
        concordant = 0.0
        for pos_count, size in zip(group_pos[::-1], group_size[::-1]):
            neg_count = size - pos_count
            concordant += pos_count * neg_below + 0.5 * pos_count * neg_count
            neg_below += neg_count
        auc = concordant / (positives * negatives)
    else:
        auc = math.nan

    if positives:
        cumulative_pos = np.cumsum(group_pos)
        cumulative_total = np.cumsum(group_size)
        precision = cumulative_pos / cumulative_total
        recall_increment = group_pos / positives
        average_precision = float(np.sum(precision * recall_increment))
    else:
        average_precision = math.nan

    cutoff = min(k, len(labels))
    remaining = float(cutoff)
    expected_pos = 0.0
    for pos_count, size in zip(group_pos, group_size):
        take = min(remaining, size)
        expected_pos += take * (pos_count / size)
        remaining -= take
        if remaining <= 0:
            break
    precision_at_k = expected_pos / cutoff
    return float(auc), average_precision, float(precision_at_k)
