"""Cross-workload evaluation of persistent KV retention.

The question is whether a serving runtime can decide what to keep by watching
its own reuse behaviour, without per-workload tuning. That makes the interesting
comparisons:

* `fixed_self`   — a model fitted on this workload's own earlier window. The
                   optimistic case, and not deployable on an unseen workload.
* `fixed_cross_*`— the same model fitted on a *different* workload. Tests
                   whether one learned weight vector transfers.
* `online`       — starts from zero weights and learns from its own matured
                   observations during the run. No workload-specific fitting.

One hyperparameter set is used for every workload. Nothing here is tuned per
trace; that restriction is the experiment.

All scored policies, baselines included, run through the same sampled-eviction
mechanism so that a difference between them is a difference in the scoring
function and not in the eviction machinery.
"""

from __future__ import annotations

import math

import numpy as np

from .online import OnlineAdaptiveScorer
from .predictors import LogisticRanker
from .replay import _occurrence_groups, replay
from .temporal import FEATURE_NAMES, PopulationNormalizer, TemporalHistory, model_indices
from .trace import Request, Trace


# Fixed for every workload. Changing any of these per trace would make the
# result a tuning exercise instead of an adaptivity result.
HYPERPARAMETERS = {
    "online_horizon_seconds": 600.0,
    "online_samples_per_group": 64,
    "online_l2": 1e-4,
    "online_learning_rate": 0.5,
    "normalizer_refresh_every": 16,
    "normalizer_sample": 512,
    "eviction_sample_width": 16,
    "offline_l2": 1e-2,
    "offline_model": "base2",
    "bytes_per_token": 2048,
    "size_model": "packed",
    "train_fraction": 0.6,
}

BASELINE_SCORERS = ("lru", "lfu", "longest_first", "frequency_x_prefix_length", "structural")


class BaselineScorer:
    """Reference policies expressed as scorers so every arm shares one mechanism."""

    time_varying = True

    def __init__(self, trace: Trace, kind: str) -> None:
        self.trace = trace
        self.kind = kind
        self.history = TemporalHistory(trace)

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> float:
        history = self.history
        if self.kind == "lru":
            return -(timestamp_ms - history.last_ms[state_id])
        if self.kind == "lfu":
            return float(history.count[state_id])
        if self.kind == "longest_first":
            return float(self.trace.states[state_id].prefix_tokens)
        if self.kind == "frequency_x_prefix_length":
            return float(history.count[state_id] * self.trace.states[state_id].prefix_tokens)
        if self.kind == "structural":
            return float(len(history.children[state_id]) + len(history.terminals[state_id]))
        raise ValueError(f"unknown baseline {self.kind!r}")


class FixedModelScorer:
    """A ranker fitted offline, applied unchanged during the run."""

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
        self.cache = None

    def attach(self, cache) -> None:
        self.cache = cache

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)
        if self.groups_seen % self.refresh_every == 0:
            population = self.history.observed_state_ids()
            if self.cache is not None and self.cache.cached:
                population = list(self.cache.cached)
            if len(population) >= 2:
                if len(population) > self.sample_size:
                    picks = self.rng.choice(len(population), size=self.sample_size, replace=False)
                    population = [population[int(p)] for p in picks]
                self.normalizer.refresh(self.history.feature_matrix(population, timestamp_ms))
        self.groups_seen += 1

    def score(self, state_id: str, timestamp_ms: float) -> float:
        row = self.history.feature_vector(state_id, timestamp_ms)
        return self.ranker.score_row(self.normalizer.apply_row(row))


def scorer_factories(trace: Trace, fixed_models: dict[str, LogisticRanker]):
    """Name -> zero-argument factory. Scorers are stateful, so each run needs a fresh one."""
    factories = {
        kind: (lambda kind=kind: BaselineScorer(trace, kind)) for kind in BASELINE_SCORERS
    }
    factories["online"] = lambda: OnlineAdaptiveScorer(
        trace,
        horizon_seconds=HYPERPARAMETERS["online_horizon_seconds"],
        samples_per_group=HYPERPARAMETERS["online_samples_per_group"],
        l2=HYPERPARAMETERS["online_l2"],
        learning_rate=HYPERPARAMETERS["online_learning_rate"],
        refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
        normalizer_sample=HYPERPARAMETERS["normalizer_sample"],
    )
    for name, ranker in fixed_models.items():
        factories[name] = lambda ranker=ranker: FixedModelScorer(
            trace,
            ranker,
            refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
            sample_size=HYPERPARAMETERS["normalizer_sample"],
        )
    return factories


def run_budget_sweep(
    trace: Trace,
    fixed_models: dict[str, LogisticRanker],
    budget_fractions: tuple[float, ...],
    split_ms: float,
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
) -> list[dict[str, object]]:
    groups = _occurrence_groups(trace)
    working_set_bytes = sum(
        (meta.block_tokens if size_model == "packed" else trace.block_size) * bytes_per_token
        for meta in trace.states.values()
    )
    factories = scorer_factories(trace, fixed_models)
    rows: list[dict[str, object]] = []
    for fraction in budget_fractions:
        capacity = max(1, round(working_set_bytes * fraction))
        shared = dict(
            occurrence_groups=groups,
            measure_from_ms=split_ms,
            size_model=size_model,
        )
        measurements: dict[str, int] = {}
        for name, factory in factories.items():
            result = replay(
                trace,
                name,
                capacity,
                fraction,
                bytes_per_token,
                scorer=factory(),
                eviction="sampled",
                sample_width=HYPERPARAMETERS["eviction_sample_width"],
                **shared,
            )
            rows.append(_row(trace, result, working_set_bytes, "sampled"))
            measurements[name] = result.avoided_prefill_tokens
        offline = replay(
            trace, "offline_next_use", capacity, fraction, bytes_per_token, **shared
        )
        rows.append(_row(trace, offline, working_set_bytes, "heap"))
        measurements["offline_next_use"] = offline.avoided_prefill_tokens

        reference = measurements["lru"]
        headroom = measurements["offline_next_use"] - reference
        for row in rows:
            if float(row["capacity_fraction"]) != fraction or row["trace"] != trace.name:
                continue
            gain = int(row["avoided_prefill_tokens"]) - reference
            row["gain_vs_lru_fraction"] = gain / max(reference, 1)
            row["headroom_closure"] = gain / headroom if headroom > 0 else math.nan
    return rows


def _row(trace: Trace, result, working_set_bytes: int, eviction: str) -> dict[str, object]:
    return {
        "trace": trace.name,
        "policy": result.policy,
        "eviction": eviction,
        "size_model": result.size_model,
        "capacity_fraction": result.capacity_fraction,
        "capacity_bytes": result.capacity_bytes,
        "working_set_bytes": working_set_bytes,
        "avoided_prefill_tokens": result.avoided_prefill_tokens,
        "reused_prefix_tokens": result.reused_prefix_tokens,
        "avoided_tokens_per_cache_byte": result.avoided_tokens_per_cache_byte,
        "block_hit_rate": result.block_hit_rate,
        "request_hit_rate": result.request_hit_rate,
        "measured_requests": result.measured_requests,
    }


def fit_fixed_model(
    trace: Trace,
    horizon_seconds: float,
    model_name: str = HYPERPARAMETERS["offline_model"],
    l2: float = HYPERPARAMETERS["offline_l2"],
    snapshot_count: int = 24,
    candidate_cap: int = 40_000,
    max_train_rows: int = 150_000,
    negatives_per_positive: float = 10.0,
    train_fraction: float = HYPERPARAMETERS["train_fraction"],
    seed: int = 0,
):
    """Fit one ranker on the trace's training window only.

    Returns the ranker, the horizon actually used, and the split timestamp. The
    requested horizon is clipped to what the trace can support on both sides of
    the split; that is a property of the trace, not a tuned choice.
    """
    from .phase05 import split_design, _labels
    from .predictors import case_control_sample
    from .temporal import standardize_rows

    usable, _, train_snapshots, split_ms, _ = split_design(
        trace, (10, 60, 300, 600, 1800), train_fraction, snapshot_count
    )
    if not usable:
        raise ValueError(f"{trace.name}: no horizon supports an embargoed split")
    horizon = max(value for value in usable if value <= horizon_seconds) if any(
        value <= horizon_seconds for value in usable
    ) else min(usable)
    indices = set(train_snapshots[horizon])
    rng = np.random.default_rng(seed)
    quota = max(1, max_train_rows // max(len(indices), 1))

    history = TemporalHistory(trace)
    features_buffer, labels_buffer = [], []
    for index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        history.observe_requests(requests, timestamp_ms)
        if index not in indices:
            continue
        state_ids = history.observed_state_ids()
        if len(state_ids) > candidate_cap:
            picks = np.sort(rng.choice(len(state_ids), size=candidate_cap, replace=False))
            state_ids = [state_ids[int(position)] for position in picks]
        normalized = standardize_rows(history.feature_matrix(state_ids, timestamp_ms))
        labels = _labels(trace, state_ids, timestamp_ms, horizon)
        sampled_features, sampled_labels = case_control_sample(
            normalized, labels, negatives_per_positive, rng
        )
        if len(sampled_labels) > quota:
            keep = np.sort(rng.choice(len(sampled_labels), size=quota, replace=False))
            sampled_features, sampled_labels = sampled_features[keep], sampled_labels[keep]
        features_buffer.append(sampled_features)
        labels_buffer.append(sampled_labels)

    stacked_features = np.vstack(features_buffer)
    stacked_labels = np.concatenate(labels_buffer)
    ranker = LogisticRanker(indices=model_indices(model_name), l2=l2).fit(
        stacked_features, stacked_labels
    )
    return ranker, horizon, split_ms, int(stacked_labels.sum()), len(stacked_labels)


def transfer_metrics(
    trace: Trace,
    rankers: dict[str, LogisticRanker],
    horizons: tuple[int, ...],
    snapshot_count: int = 24,
    candidate_cap: int = 40_000,
    precision_k: int = 100,
    train_fraction: float = HYPERPARAMETERS["train_fraction"],
    seed: int = 0,
) -> list[dict[str, object]]:
    """Score externally fitted rankers on this trace's held-out test snapshots."""
    from collections import defaultdict

    from .phase05 import _labels, fast_ranking_metrics, split_design
    from .temporal import standardize_rows

    usable, test_snapshots, _, _, _ = split_design(
        trace, tuple(horizons), train_fraction, snapshot_count
    )
    if not usable or not test_snapshots:
        return []
    rng = np.random.default_rng(seed)
    test_set = set(test_snapshots)
    samples: dict[tuple[int, str], list[tuple[float, float, float]]] = defaultdict(list)
    counts: dict[int, list[tuple[int, int]]] = defaultdict(list)

    history = TemporalHistory(trace)
    for index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        history.observe_requests(requests, timestamp_ms)
        if index not in test_set:
            continue
        state_ids = history.observed_state_ids()
        if len(state_ids) > candidate_cap:
            picks = np.sort(rng.choice(len(state_ids), size=candidate_cap, replace=False))
            state_ids = [state_ids[int(position)] for position in picks]
        normalized = standardize_rows(history.feature_matrix(state_ids, timestamp_ms))
        for horizon in usable:
            labels = _labels(trace, state_ids, timestamp_ms, horizon)
            counts[horizon].append((int(labels.sum()), len(labels)))
            for name, ranker in rankers.items():
                samples[(horizon, name)].append(
                    fast_ranking_metrics(labels, ranker.score(normalized), precision_k)
                )

    rows = []
    for horizon in usable:
        pairs = counts[horizon]
        for name in rankers:
            values = [v for v in samples[(horizon, name)] if math.isfinite(v[0])]
            rows.append(
                {
                    "evaluated_on": trace.name,
                    "fitted_on": name,
                    "horizon_seconds": horizon,
                    "auc": float(np.mean([v[0] for v in values])) if values else math.nan,
                    "auc_std": float(np.std([v[0] for v in values], ddof=1))
                    if len(values) > 1
                    else math.nan,
                    "average_precision": float(np.mean([v[1] for v in values]))
                    if values
                    else math.nan,
                    f"precision_at_{precision_k}": float(np.mean([v[2] for v in values]))
                    if values
                    else math.nan,
                    "test_snapshots": len(values),
                    "mean_positive_states": float(np.mean([p[0] for p in pairs])) if pairs else math.nan,
                    "mean_candidate_states": float(np.mean([p[1] for p in pairs])) if pairs else math.nan,
                }
            )
    return rows
