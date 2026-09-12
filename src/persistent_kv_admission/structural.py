"""Causal linear-ranking tests for structural feature incrementality."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict

import numpy as np

from .characterize import HORIZONS_SECONDS, _sample_indices
from .ranking import ranking_metrics
from .trace import Trace, unique_timestamps


FEATURES = ("frequency", "recency", "prefix_length", "fan_out", "branch_diversity")
BASE_INDICES = (0, 1, 2)
EXTENDED_INDICES = (0, 1, 2, 3, 4)


class _LinearMoments:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.sum_x = np.zeros(dimension)
        self.sum_xx = np.zeros((dimension, dimension))
        self.sum_xy = np.zeros(dimension)
        self.sum_y = 0.0

    def add(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.count += len(labels)
        self.sum_x += features.sum(axis=0)
        self.sum_xx += features.T @ features
        self.sum_xy += features.T @ labels
        self.sum_y += float(labels.sum())

    def fit(self, indices: tuple[int, ...], ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot fit without training samples")
        chosen = np.asarray(indices)
        mean = self.sum_x[chosen] / self.count
        centered_xx = self.sum_xx[np.ix_(chosen, chosen)] - self.count * np.outer(mean, mean)
        variance = np.maximum(np.diag(centered_xx) / self.count, 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        standardized_xx = centered_xx / np.outer(scale, scale)
        centered_xy = self.sum_xy[chosen] - mean * self.sum_y
        standardized_xy = centered_xy / scale
        coefficients = np.linalg.solve(
            standardized_xx + ridge * self.count * np.eye(len(indices)),
            standardized_xy,
        )
        return mean, scale, coefficients


def _features(
    trace: Trace,
    state_ids: list[str],
    timestamp: float,
    frequency: dict[str, int],
    last_seen: dict[str, float],
    children: dict[str, set[str]],
    terminals: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_frequency = np.asarray([frequency[state_id] for state_id in state_ids], dtype=float)
    age_seconds = np.asarray(
        [(timestamp - last_seen[state_id]) / 1000.0 for state_id in state_ids], dtype=float
    )
    matrix = np.column_stack(
        (
            np.log1p(raw_frequency),
            -np.log1p(age_seconds),
            np.log1p([trace.states[state_id].prefix_tokens for state_id in state_ids]),
            np.log1p([len(children[state_id]) for state_id in state_ids]),
            np.log1p([len(terminals[state_id]) for state_id in state_ids]),
        )
    )
    return matrix, raw_frequency, age_seconds


def _iter_snapshots(trace: Trace, horizon: int, selected: set[int]):
    frequency: dict[str, int] = defaultdict(int)
    last_seen: dict[str, float] = {}
    children: dict[str, set[str]] = defaultdict(set)
    terminals: dict[str, set[str]] = defaultdict(set)
    for timestamp_index, (timestamp, requests) in enumerate(trace.timestamp_groups()):
        for request in requests:
            terminal = request.hash_ids[-1]
            for index, state_id in enumerate(request.hash_ids):
                frequency[state_id] += 1
                last_seen[state_id] = timestamp
                terminals[state_id].add(terminal)
                if index + 1 < len(request.hash_ids):
                    children[state_id].add(request.hash_ids[index + 1])
        if timestamp_index not in selected:
            continue
        state_ids = list(last_seen)
        matrix, raw_frequency, age_seconds = _features(
            trace, state_ids, timestamp, frequency, last_seen, children, terminals
        )
        cutoff = timestamp + horizon * 1000.0
        labels = np.asarray(
            [
                int(
                    (position := bisect.bisect_right(trace.occurrences_ms[state_id], timestamp))
                    < len(trace.occurrences_ms[state_id])
                    and trace.occurrences_ms[state_id][position] <= cutoff
                )
                for state_id in state_ids
            ],
            dtype=np.int8,
        )
        yield timestamp_index, timestamp, matrix, raw_frequency, age_seconds, labels


def _macro(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _quantile_bins(values: np.ndarray, count: int = 5) -> list[np.ndarray]:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, count + 1)))
    if len(edges) < 2:
        return [np.ones(len(values), dtype=bool)]
    assignments = np.digitize(values, edges[1:-1], right=True)
    return [assignments == index for index in range(len(edges) - 1)]


def evaluate_structural_incrementality(
    trace: Trace,
    snapshot_count: int = 24,
    precision_k: int = 100,
    ridge: float = 1e-3,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Compare held-out Base and Extended linear rankings.

    The first 60% of sampled snapshots train a ridge linear probability ranker.
    Training snapshots whose label horizon overlaps the first test snapshot are
    embargoed. Metrics are macro-averaged over the final 40% of snapshots.
    """
    timestamps = unique_timestamps(trace)
    comparison_rows, stratified_rows, correlation_rows, coefficient_rows = [], [], [], []
    for horizon in HORIZONS_SECONDS:
        selected = sorted(
            _sample_indices(timestamps, trace.end_ms - horizon * 1000.0, snapshot_count)
        )
        if len(selected) < 4:
            for model in ("base", "extended"):
                comparison_rows.append(
                    _empty_comparison(horizon, model, "insufficient_snapshots", precision_k)
                )
            continue
        test_offset = max(1, int(math.ceil(0.6 * len(selected))))
        test_indices = set(selected[test_offset:])
        first_test_time = timestamps[selected[test_offset]]
        train_indices = {
            index
            for index in selected[:test_offset]
            if timestamps[index] + horizon * 1000.0 <= first_test_time
        }
        if not train_indices or not test_indices:
            for model in ("base", "extended"):
                comparison_rows.append(
                    _empty_comparison(
                        horizon, model, "insufficient_embargoed_split", precision_k
                    )
                )
            continue

        moments = _LinearMoments(len(FEATURES))
        fitted = None
        model_metrics = {model: [[], [], []] for model in ("base", "extended")}
        stratified: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
        correlations: dict[tuple[str, str], list[float]] = defaultdict(list)
        train_positive = train_samples = test_positive = test_samples = 0

        for index, _, matrix, raw_frequency, age_seconds, labels in _iter_snapshots(
            trace, horizon, train_indices | test_indices
        ):
            if index in train_indices:
                moments.add(matrix, labels)
                train_positive += int(labels.sum())
                train_samples += len(labels)
                continue
            if fitted is None:
                fitted = {
                    "base": (*moments.fit(BASE_INDICES, ridge), BASE_INDICES),
                    "extended": (*moments.fit(EXTENDED_INDICES, ridge), EXTENDED_INDICES),
                }
                for model, (_, _, coefficients, indices) in fitted.items():
                    for feature_index, coefficient in zip(indices, coefficients):
                        coefficient_rows.append(
                            {
                                "horizon_seconds": horizon,
                                "model": model,
                                "feature": FEATURES[feature_index],
                                "standardized_coefficient": float(coefficient),
                                "ridge": ridge,
                            }
                        )
            test_positive += int(labels.sum())
            test_samples += len(labels)
            for model, (mean, scale, coefficients, indices) in fitted.items():
                scores = ((matrix[:, indices] - mean) / scale) @ coefficients
                metrics = ranking_metrics(labels, scores, precision_k)
                for destination, value in zip(model_metrics[model], metrics):
                    destination.append(value)

            controls = {"frequency": raw_frequency, "recency": age_seconds}
            structures = {"fan_out": matrix[:, 3], "branch_diversity": matrix[:, 4]}
            for control_name, control_values in controls.items():
                for mask in _quantile_bins(control_values):
                    if int(mask.sum()) < 20 or labels[mask].min() == labels[mask].max():
                        continue
                    for signal, scores in structures.items():
                        stratified[(control_name, signal)].append(
                            ranking_metrics(labels[mask], scores[mask], precision_k)
                        )
            for base_index in range(3):
                for structural_index in (3, 4):
                    x, y = matrix[:, base_index], matrix[:, structural_index]
                    if np.std(x) > 0 and np.std(y) > 0:
                        correlations[(FEATURES[base_index], FEATURES[structural_index])].append(
                            float(np.corrcoef(x, y)[0, 1])
                        )

        base_values = model_metrics["base"]
        base_aggregates = tuple(_macro(values) for values in base_values)
        for model in ("base", "extended"):
            aggregates = tuple(_macro(values) for values in model_metrics[model])
            comparison_rows.append(
                {
                    "horizon_seconds": horizon,
                    "model": model,
                    "features": "+".join(FEATURES[index] for index in
                                         (BASE_INDICES if model == "base" else EXTENDED_INDICES)),
                    "auc": aggregates[0],
                    "average_precision": aggregates[1],
                    f"precision_at_{precision_k}": aggregates[2],
                    "delta_auc_vs_base": aggregates[0] - base_aggregates[0] if model == "extended" else 0.0,
                    "delta_ap_vs_base": aggregates[1] - base_aggregates[1] if model == "extended" else 0.0,
                    f"delta_precision_at_{precision_k}_vs_base": aggregates[2] - base_aggregates[2] if model == "extended" else 0.0,
                    "train_snapshots": len(train_indices),
                    "test_snapshots": len(test_indices),
                    "train_samples": train_samples,
                    "train_positive_rate": train_positive / max(train_samples, 1),
                    "test_samples": test_samples,
                    "test_positive_rate": test_positive / max(test_samples, 1),
                    "ridge": ridge,
                    "status": "ok" if math.isfinite(aggregates[1]) else "no_positive_test_windows",
                }
            )
        for (control, signal), metrics in stratified.items():
            stratified_rows.append(
                {
                    "horizon_seconds": horizon,
                    "control": control,
                    "structural_signal": signal,
                    "auc": _macro([value[0] for value in metrics]),
                    "average_precision": _macro([value[1] for value in metrics]),
                    f"precision_at_{precision_k}": _macro([value[2] for value in metrics]),
                    "valid_stratum_snapshots": len(metrics),
                }
            )
        for (base_feature, structural_feature), values in correlations.items():
            correlation_rows.append(
                {
                    "horizon_seconds": horizon,
                    "base_feature": base_feature,
                    "structural_feature": structural_feature,
                    "mean_pearson_correlation": _macro(values),
                    "test_snapshots": len(values),
                }
            )
    return comparison_rows, stratified_rows, correlation_rows, coefficient_rows


def _empty_comparison(
    horizon: int, model: str, status: str, precision_k: int = 100
) -> dict[str, object]:
    return {
        "horizon_seconds": horizon,
        "model": model,
        "features": "+".join(FEATURES[index] for index in
                             (BASE_INDICES if model == "base" else EXTENDED_INDICES)),
        "auc": math.nan,
        "average_precision": math.nan,
        f"precision_at_{precision_k}": math.nan,
        "delta_auc_vs_base": math.nan,
        "delta_ap_vs_base": math.nan,
        f"delta_precision_at_{precision_k}_vs_base": math.nan,
        "train_snapshots": 0,
        "test_snapshots": 0,
        "train_samples": 0,
        "train_positive_rate": math.nan,
        "test_samples": 0,
        "test_positive_rate": math.nan,
        "ridge": math.nan,
        "status": status,
    }
