"""Causal temporal-history features for persistent KV selection.

Every feature is computed from occurrences strictly at or before the decision
point. No future information enters a feature vector. The tracker is fed the
trace in arrival order, so the same object can serve offline snapshot
evaluation and online replay scoring without a second code path.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .trace import Request, Trace


WINDOW_SECONDS = (10.0, 60.0, 300.0, 600.0)
INTERVAL_MEMORY = 16
LAG_MEMORY = 4

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("log_frequency", "neg_log_recency", "log_prefix_tokens"),
    "rate": ("log_age", "log_reuse_rate"),
    "interarrival": (
        "single_occurrence",
        "log_mean_interval",
        "log_median_interval",
        "log_std_interval",
        "interval_cv",
        "log_interval_lag1",
        "log_interval_lag2",
        "log_interval_lag3",
        "log_interval_lag4",
    ),
    "window": (
        "log_freq_10s",
        "log_freq_60s",
        "log_freq_300s",
        "log_freq_600s",
        "recent_over_lifetime",
    ),
    "dynamics": ("interval_trend", "burstiness"),
    "structural": ("log_fan_out", "log_branch_diversity"),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in ("core", "rate", "interarrival", "window", "dynamics", "structural")
    for name in FEATURE_GROUPS[group]
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

GROUP_INDICES: dict[str, tuple[int, ...]] = {
    group: tuple(FEATURE_INDEX[name] for name in names)
    for group, names in FEATURE_GROUPS.items()
}

MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "base0": ("core",),
    "base1": ("core", "rate", "interarrival", "window", "dynamics"),
    "base2": ("core", "rate", "interarrival", "window", "dynamics", "structural"),
}


def model_indices(model: str) -> tuple[int, ...]:
    if model not in MODEL_GROUPS:
        raise ValueError(f"unknown model {model!r}")
    indices: list[int] = []
    for group in MODEL_GROUPS[model]:
        indices.extend(GROUP_INDICES[group])
    return tuple(sorted(indices))


def indices_excluding(model: str, dropped_group: str) -> tuple[int, ...]:
    if dropped_group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group {dropped_group!r}")
    dropped = set(GROUP_INDICES[dropped_group])
    return tuple(index for index in model_indices(model) if index not in dropped)


class TemporalHistory:
    """Incremental per-state occurrence history with causal feature readout."""

    def __init__(self, trace: Trace, max_window_seconds: float = max(WINDOW_SECONDS)) -> None:
        self.trace = trace
        self.max_window_ms = max_window_seconds * 1000.0
        self.count: dict[str, int] = defaultdict(int)
        self.first_ms: dict[str, float] = {}
        self.last_ms: dict[str, float] = {}
        self.interval_count: dict[str, int] = defaultdict(int)
        self.interval_sum: dict[str, float] = defaultdict(float)
        self.interval_square_sum: dict[str, float] = defaultdict(float)
        self.recent_intervals: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=INTERVAL_MEMORY)
        )
        self.recent_times: dict[str, deque[float]] = defaultdict(deque)
        self.children: dict[str, set[str]] = defaultdict(set)
        self.terminals: dict[str, set[str]] = defaultdict(set)
        self.order: list[str] = []

    def observe_requests(self, requests: list[Request], timestamp_ms: float) -> None:
        for request in requests:
            terminal = request.hash_ids[-1]
            for position, state_id in enumerate(request.hash_ids):
                self.terminals[state_id].add(terminal)
                if position + 1 < len(request.hash_ids):
                    self.children[state_id].add(request.hash_ids[position + 1])
                previous = self.last_ms.get(state_id)
                if previous is None:
                    self.first_ms[state_id] = timestamp_ms
                    self.order.append(state_id)
                else:
                    interval = (timestamp_ms - previous) / 1000.0
                    self.interval_count[state_id] += 1
                    self.interval_sum[state_id] += interval
                    self.interval_square_sum[state_id] += interval * interval
                    self.recent_intervals[state_id].append(interval)
                self.count[state_id] += 1
                self.last_ms[state_id] = timestamp_ms
                times = self.recent_times[state_id]
                times.append(timestamp_ms)
                cutoff = timestamp_ms - self.max_window_ms
                while times and times[0] < cutoff:
                    times.popleft()

    def observed_state_ids(self) -> list[str]:
        return self.order

    def feature_vector(self, state_id: str, now_ms: float) -> list[float]:
        count = self.count[state_id]
        last = self.last_ms[state_id]
        recency = max((now_ms - last) / 1000.0, 0.0)
        age = max((now_ms - self.first_ms[state_id]) / 1000.0, 0.0)
        reuse_rate = count / max(age, 1.0)

        interval_count = self.interval_count[state_id]
        if interval_count:
            mean_interval = self.interval_sum[state_id] / interval_count
            variance = max(
                self.interval_square_sum[state_id] / interval_count - mean_interval * mean_interval,
                0.0,
            )
            std_interval = math.sqrt(variance)
            window = self.recent_intervals[state_id]
            ordered = sorted(window)
            middle = len(ordered) // 2
            median_interval = (
                ordered[middle]
                if len(ordered) % 2
                else 0.5 * (ordered[middle - 1] + ordered[middle])
            )
            lags = list(window)[-LAG_MEMORY:][::-1]
            denominator = std_interval + mean_interval
            burstiness = (std_interval - mean_interval) / denominator if denominator > 0 else 0.0
            cv = std_interval / mean_interval if mean_interval > 0 else 0.0
            trend = math.log1p(lags[0]) - math.log1p(mean_interval)
        else:
            mean_interval = median_interval = std_interval = 0.0
            lags = []
            burstiness = cv = trend = 0.0

        lag_values = [lags[index] if index < len(lags) else 0.0 for index in range(LAG_MEMORY)]

        times = self.recent_times[state_id]
        window_counts = [0.0] * len(WINDOW_SECONDS)
        for index, seconds in enumerate(WINDOW_SECONDS):
            cutoff = now_ms - seconds * 1000.0
            # recent_times is ascending; count the suffix at or after the cutoff.
            total = 0
            for value in reversed(times):
                if value < cutoff:
                    break
                total += 1
            window_counts[index] = float(total)

        return [
            math.log1p(count),
            -math.log1p(recency),
            math.log1p(self.trace.states[state_id].prefix_tokens),
            math.log1p(age),
            math.log1p(reuse_rate),
            1.0 if count == 1 else 0.0,
            math.log1p(mean_interval),
            math.log1p(median_interval),
            math.log1p(std_interval),
            cv,
            math.log1p(lag_values[0]),
            math.log1p(lag_values[1]),
            math.log1p(lag_values[2]),
            math.log1p(lag_values[3]),
            math.log1p(window_counts[0]),
            math.log1p(window_counts[1]),
            math.log1p(window_counts[2]),
            math.log1p(window_counts[3]),
            window_counts[3] / count,
            trend,
            burstiness,
            math.log1p(len(self.children[state_id])),
            math.log1p(len(self.terminals[state_id])),
        ]

    def feature_matrix(self, state_ids: list[str], now_ms: float) -> np.ndarray:
        if not state_ids:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
        return np.asarray(
            [self.feature_vector(state_id, now_ms) for state_id in state_ids], dtype=float
        )


def standardize_rows(matrix: np.ndarray) -> np.ndarray:
    """Z-score each column against the population present at this decision point.

    Feature scales drift strongly with position in a trace: age is bounded by
    elapsed time, and counts accumulate. A model fitted on an early window and
    applied to a later one extrapolates badly without this. Standardising
    inside each decision point uses only states already observed, so it stays
    causal, and it is the same normalisation the replay scorer applies.
    """
    if len(matrix) == 0:
        return matrix
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (matrix - mean) / scale


class PopulationNormalizer:
    """Per-decision-point feature statistics, refreshed from a random sample."""

    def __init__(self, dimension: int) -> None:
        self.mean = np.zeros(dimension)
        self.scale = np.ones(dimension)
        self.refreshes = 0

    def refresh(self, matrix: np.ndarray) -> None:
        if len(matrix) < 2:
            return
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale = scale
        self.refreshes += 1

    def apply_row(self, row: list[float]) -> list[float]:
        mean = self.mean
        scale = self.scale
        return [(value - mean[index]) / scale[index] for index, value in enumerate(row)]
