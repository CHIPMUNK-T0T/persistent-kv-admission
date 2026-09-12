"""Phase 0.5 — how much future reuse value lives in temporal reuse history.

The question is not which cache algorithm wins. It is which observable signals
carry future-reuse information, measured first as ranking quality and then, as
the decisive test, as avoided prefill tokens under a fixed byte budget.

Design points that the earlier phase got wrong and this module fixes:

* One snapshot set is shared by every horizon, so a horizon comparison is not
  confounded with position in the trace.
* Predictors are fitted on an earlier window with a full horizon embargo and
  are scored on a later window that the fit never saw.
* Every policy, baseline or predictor, is replayed over the whole trace but
  measured only on the evaluation window, so the cache is equally warm.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict

import numpy as np

from .predictors import LogisticRanker, case_control_sample
from .ranking import ranking_metrics
from .replay import POLICIES, _occurrence_groups, replay
from .temporal import (
    FEATURE_NAMES,
    GROUP_INDICES,
    MODEL_GROUPS,
    PopulationNormalizer,
    TemporalHistory,
    indices_excluding,
    model_indices,
    standardize_rows,
)
from .trace import Request, Trace


HORIZONS_SECONDS = (10, 60, 300, 600, 1800)
MODELS = ("base0", "base1", "base2")
ABLATION_GROUPS = ("rate", "interarrival", "window", "dynamics", "structural")
BASELINE_POLICIES = tuple(policy for policy in POLICIES if policy != "offline_next_use")


def split_design(
    trace: Trace,
    horizons=HORIZONS_SECONDS,
    train_fraction: float = 0.6,
    snapshot_count: int = 24,
    minimum_snapshots: int = 6,
):
    """Pick the evaluation window, the shared test snapshots, and per-horizon training snapshots.

    A horizon needs a label window on both sides of the split: `horizon` of
    lookahead before the split for training, and `horizon` of lookahead before
    the trace end for testing. A horizon longer than either side is unusable no
    matter how many snapshots are sampled, which is a property of the trace.

    Test snapshots are shared by every usable horizon so that a horizon
    comparison is not confounded with position in the trace. Training snapshots
    are per horizon because the embargo length is the horizon itself.
    """
    timestamps = [timestamp for timestamp, _ in trace.timestamp_groups()]
    split_ms = trace.start_ms + train_fraction * (trace.end_ms - trace.start_ms)

    usable = []
    for horizon in horizons:
        offset = horizon * 1000.0
        train_pool = [value for value in timestamps if value <= split_ms - offset]
        test_pool = [value for value in timestamps if split_ms <= value <= trace.end_ms - offset]
        if len(train_pool) >= minimum_snapshots and len(test_pool) >= minimum_snapshots:
            usable.append(horizon)
    if not usable:
        return [], [], {}, split_ms, timestamps

    max_horizon = max(usable)
    test_pool = [
        index
        for index, value in enumerate(timestamps)
        if split_ms <= value <= trace.end_ms - max_horizon * 1000.0
    ]
    test_snapshots = _even_sample(test_pool, snapshot_count)
    train_snapshots = {
        horizon: _even_sample(
            [
                index
                for index, value in enumerate(timestamps)
                if value <= split_ms - horizon * 1000.0
            ],
            snapshot_count,
        )
        for horizon in usable
    }
    return usable, test_snapshots, train_snapshots, split_ms, timestamps


def _even_sample(pool: list[int], count: int) -> list[int]:
    if not pool:
        return []
    positions = np.linspace(0, len(pool) - 1, min(count, len(pool)), dtype=int)
    return sorted({pool[int(position)] for position in positions})


def _labels(trace: Trace, state_ids: list[str], now_ms: float, horizon: int) -> np.ndarray:
    cutoff = now_ms + horizon * 1000.0
    out = np.empty(len(state_ids), dtype=np.int8)
    for position, state_id in enumerate(state_ids):
        occurrences = trace.occurrences_ms[state_id]
        index = bisect.bisect_right(occurrences, now_ms)
        out[position] = int(index < len(occurrences) and occurrences[index] <= cutoff)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ordered = values[order]
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_matrix(features: np.ndarray) -> np.ndarray:
    ranked = np.column_stack(
        [_average_ranks(features[:, index]) for index in range(features.shape[1])]
    )
    centred = ranked - ranked.mean(axis=0)
    deviation = np.sqrt((centred * centred).sum(axis=0))
    deviation[deviation < 1e-12] = 1.0
    return (centred.T @ centred) / np.outer(deviation, deviation)


class PredictorScorer:
    """Adapts a fitted ranker to the replay eviction path.

    The ranker was fitted on features standardised inside each decision point,
    so the same normalisation is reproduced here. Recomputing exact population
    statistics at every eviction would dominate the replay cost, so they are
    refreshed every `refresh_every` timestamp groups from a random sample of
    already-observed states. Sampling only touches past observations, so the
    scorer stays causal.
    """

    time_varying = True

    def __init__(
        self,
        trace: Trace,
        ranker: LogisticRanker,
        refresh_every: int = 16,
        sample_size: int = 512,
        seed: int = 0,
    ) -> None:
        self.history = TemporalHistory(trace)
        self.ranker = ranker
        self.normalizer = PopulationNormalizer(len(FEATURE_NAMES))
        self.refresh_every = refresh_every
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.groups_seen = 0

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)
        if self.groups_seen % self.refresh_every == 0:
            observed = self.history.observed_state_ids()
            if len(observed) >= 2:
                if len(observed) > self.sample_size:
                    picks = self.rng.choice(len(observed), size=self.sample_size, replace=False)
                    sample = [observed[int(position)] for position in picks]
                else:
                    sample = list(observed)
                self.normalizer.refresh(self.history.feature_matrix(sample, timestamp_ms))
        self.groups_seen += 1

    def score(self, state_id: str, timestamp_ms: float) -> float:
        row = self.history.feature_vector(state_id, timestamp_ms)
        return self.ranker.score_row(self.normalizer.apply_row(row))


def fast_ranking_metrics(labels: np.ndarray, scores: np.ndarray, k: int):
    """Vectorised equivalent of ranking.ranking_metrics with identical tie handling."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have equal shapes")
    if len(labels) == 0:
        return math.nan, math.nan, math.nan
    positives = float(labels.sum())
    negatives = float(len(labels)) - positives

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    group_positive = np.add.reduceat(labels[order], starts).astype(float)
    group_size = (ends - starts).astype(float)
    group_negative = group_size - group_positive

    if positives and negatives:
        negatives_below = np.cumsum(group_negative[::-1])[::-1] - group_negative
        concordant = float(
            np.sum(group_positive * negatives_below)
            + 0.5 * np.sum(group_positive * group_negative)
        )
        auc = concordant / (positives * negatives)
    else:
        auc = math.nan

    if positives:
        precision = np.cumsum(group_positive) / np.cumsum(group_size)
        average_precision = float(np.sum(precision * (group_positive / positives)))
    else:
        average_precision = math.nan

    cutoff = float(min(k, len(labels)))
    boundaries = np.cumsum(group_size)
    taken = np.clip(cutoff - np.r_[0.0, boundaries[:-1]], 0.0, group_size)
    expected = float(np.sum(taken * (group_positive / group_size)))
    return float(auc), average_precision, expected / cutoff


def variant_indices() -> dict[str, tuple[int, ...]]:
    variants: dict[str, tuple[int, ...]] = {model: model_indices(model) for model in MODELS}
    for group in ABLATION_GROUPS:
        variants[f"base2_minus_{group}"] = indices_excluding("base2", group)
        variants[f"base0_plus_{group}"] = tuple(
            sorted(set(model_indices("base0")) | set(GROUP_INDICES[group]))
        )
    return variants


def evaluate_temporal_prediction(
    trace: Trace,
    snapshot_count: int = 24,
    precision_k: int = 100,
    train_fraction: float = 0.6,
    candidate_cap: int = 40_000,
    max_train_rows: int = 150_000,
    negatives_per_positive: float = 10.0,
    l2: float = 1.0,
    seed: int = 0,
):
    """Fit and score every predictor variant in one causal pass over the trace."""
    rng = np.random.default_rng(seed)
    usable, test_snapshots, train_snapshots, split_ms, timestamps = split_design(
        trace, HORIZONS_SECONDS, train_fraction, snapshot_count
    )
    if not usable:
        raise ValueError(f"{trace.name}: no horizon supports an embargoed train/test split")

    variants = variant_indices()
    train_index_map: dict[int, set[int]] = {
        horizon: set(indices) for horizon, indices in train_snapshots.items()
    }
    all_train_indices = set().union(*train_index_map.values())
    test_index_set = set(test_snapshots)
    processed = all_train_indices | test_index_set

    quotas = {
        horizon: max(1, max_train_rows // max(len(indices), 1))
        for horizon, indices in train_snapshots.items()
    }
    train_features: dict[int, list[np.ndarray]] = defaultdict(list)
    train_labels: dict[int, list[np.ndarray]] = defaultdict(list)

    history = TemporalHistory(trace)
    models: dict[tuple[int, str], LogisticRanker] = {}
    fitted = False
    metric_samples: dict[tuple[int, str], list[tuple[float, float, float]]] = defaultdict(list)
    label_counts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    correlation_total: np.ndarray | None = None
    correlation_snapshots = 0

    for index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        history.observe_requests(requests, timestamp_ms)
        if index not in processed:
            continue
        state_ids = history.observed_state_ids()
        if len(state_ids) > candidate_cap:
            chosen = np.sort(rng.choice(len(state_ids), size=candidate_cap, replace=False))
            state_ids = [state_ids[int(position)] for position in chosen]
        features = history.feature_matrix(state_ids, timestamp_ms)
        normalized = standardize_rows(features)

        if index in all_train_indices:
            for horizon in usable:
                if index not in train_index_map[horizon]:
                    continue
                labels = _labels(trace, state_ids, timestamp_ms, horizon)
                sampled_features, sampled_labels = case_control_sample(
                    normalized, labels, negatives_per_positive, rng
                )
                quota = quotas[horizon]
                if len(sampled_labels) > quota:
                    keep = np.sort(rng.choice(len(sampled_labels), size=quota, replace=False))
                    sampled_features = sampled_features[keep]
                    sampled_labels = sampled_labels[keep]
                train_features[horizon].append(sampled_features)
                train_labels[horizon].append(sampled_labels)
            if index not in test_index_set:
                continue

        if not fitted:
            fitted = True
            for horizon in usable:
                if not train_labels[horizon]:
                    continue
                stacked_features = np.vstack(train_features[horizon])
                stacked_labels = np.concatenate(train_labels[horizon])
                total = int(stacked_labels.sum())
                if total == 0 or total == len(stacked_labels):
                    continue
                for name, indices in variants.items():
                    models[(horizon, name)] = LogisticRanker(indices=indices, l2=l2).fit(
                        stacked_features, stacked_labels
                    )
            train_features.clear()
            train_labels.clear()

        matrix = spearman_matrix(features)
        correlation_total = matrix if correlation_total is None else correlation_total + matrix
        correlation_snapshots += 1
        for horizon in usable:
            labels = _labels(trace, state_ids, timestamp_ms, horizon)
            label_counts[horizon].append((int(labels.sum()), len(labels)))
            for name in variants:
                model = models.get((horizon, name))
                if model is not None:
                    metric_samples[(horizon, name)].append(
                        fast_ranking_metrics(labels, model.score(normalized), precision_k)
                    )

    metric_rows = []
    for horizon in usable:
        counts = label_counts[horizon]
        positives = float(np.mean([value[0] for value in counts])) if counts else math.nan
        candidates = float(np.mean([value[1] for value in counts])) if counts else math.nan
        base_samples = metric_samples[(horizon, "base0")]
        for name in variants:
            samples = metric_samples[(horizon, name)]
            usable_samples = [value for value in samples if math.isfinite(value[0])]
            paired = [
                (value[0] - base[0], value[2] - base[2])
                for value, base in zip(samples, base_samples)
                if math.isfinite(value[0]) and math.isfinite(base[0])
            ]
            metric_rows.append(
                {
                    "trace": trace.name,
                    "horizon_seconds": horizon,
                    "variant": name,
                    "feature_count": len(variants[name]),
                    "auc": _mean([value[0] for value in usable_samples]),
                    "average_precision": _mean([value[1] for value in usable_samples]),
                    f"precision_at_{precision_k}": _mean([value[2] for value in usable_samples]),
                    "auc_std": _std([value[0] for value in usable_samples]),
                    "average_precision_std": _std([value[1] for value in usable_samples]),
                    f"precision_at_{precision_k}_std": _std(
                        [value[2] for value in usable_samples]
                    ),
                    "delta_auc_vs_base0": _mean([value[0] for value in paired]),
                    "delta_auc_vs_base0_std": _std([value[0] for value in paired]),
                    "delta_auc_vs_base0_win_rate": _mean(
                        [1.0 if value[0] > 0 else 0.0 for value in paired]
                    ),
                    f"delta_precision_at_{precision_k}_vs_base0": _mean(
                        [value[1] for value in paired]
                    ),
                    f"delta_precision_at_{precision_k}_vs_base0_win_rate": _mean(
                        [1.0 if value[1] > 0 else 0.0 for value in paired]
                    ),
                    "test_snapshots": len(usable_samples),
                    "train_snapshots": len(train_snapshots.get(horizon, ())),
                    "mean_positive_states": positives,
                    "mean_candidate_states": candidates,
                    "status": "ok" if usable_samples else "no_usable_test_windows",
                }
            )

    coefficient_rows = [
        {
            "trace": trace.name,
            "horizon_seconds": horizon,
            "variant": name,
            "feature": feature,
            "standardized_coefficient": value,
            "converged": model.converged,
            "newton_iterations": model.iterations,
        }
        for (horizon, name), model in models.items()
        if name in MODELS
        for feature, value in model.standardized_coefficients(FEATURE_NAMES)
    ]

    correlation_rows = []
    if correlation_total is not None and correlation_snapshots:
        mean_matrix = correlation_total / correlation_snapshots
        for row_index, row_name in enumerate(FEATURE_NAMES):
            for column_index in range(row_index + 1, len(FEATURE_NAMES)):
                correlation_rows.append(
                    {
                        "trace": trace.name,
                        "feature_a": row_name,
                        "feature_b": FEATURE_NAMES[column_index],
                        "mean_spearman": float(mean_matrix[row_index, column_index]),
                        "test_snapshots": correlation_snapshots,
                    }
                )

    meta = {
        "trace": trace.name,
        "usable_horizons": usable,
        "unusable_horizons": [h for h in HORIZONS_SECONDS if h not in usable],
        "max_shared_horizon_seconds": max(usable),
        "split_ms": split_ms,
        "split_seconds": split_ms / 1000.0,
        "test_snapshot_count": len(test_snapshots),
        "train_snapshot_count": {horizon: len(v) for horizon, v in train_snapshots.items()},
        "candidate_cap": candidate_cap,
        "train_fraction": train_fraction,
    }
    return metric_rows, coefficient_rows, correlation_rows, models, meta


def _mean(values) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def _std(values) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
