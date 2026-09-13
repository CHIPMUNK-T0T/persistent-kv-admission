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
from .predictors import LogisticRanker, RidgeRanker
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
    """A ranker fitted offline, applied unchanged during the run.

    `normalize_on` picks the population whose feature statistics standardise
    the row before the fitted coefficients are applied. "cached" uses the
    retained set (the population the eviction ranks); "observed" uses every
    state seen so far, which is the population the ranker was fitted and
    evaluated on. The two are not ranking-equivalent for a linear model: the
    effective weight of feature i is w_i / scale_i, so a different scale
    vector is a different ranking. Both are measured.
    """

    time_varying = True

    def __init__(
        self,
        trace: Trace,
        ranker: LogisticRanker,
        refresh_every: int = 16,
        sample_size: int = 512,
        seed: int = 0,
        normalize_on: str = "cached",
    ) -> None:
        if normalize_on not in {"cached", "observed"}:
            raise ValueError(f"unknown normalisation population {normalize_on!r}")
        self.history = TemporalHistory(trace)
        self.ranker = ranker
        self.normalizer = PopulationNormalizer(len(FEATURE_NAMES))
        self.refresh_every = refresh_every
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.groups_seen = 0
        self.cache = None
        self.normalize_on = normalize_on

    def attach(self, cache) -> None:
        self.cache = cache

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)
        if self.groups_seen % self.refresh_every == 0:
            population = self.history.observed_state_ids()
            # A population of one has no scale, so `refresh` would decline it
            # and the normaliser would keep statistics from an older, unrelated
            # moment. Fall back to the observed population instead, as an empty
            # retained set already does. Only reachable while the ranked set is
            # still filling up; with two or more members nothing changes.
            if self.normalize_on == "cached" and self.cache is not None and len(self.cache.cached) >= 2:
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


def scorer_factories(trace: Trace, fixed_models: dict[str, LogisticRanker], seed: int = 0):
    """Name -> zero-argument factory. Scorers are stateful, so each run needs a fresh one.

    `seed` reaches every random draw a scorer makes (normaliser samples, online
    training samples), so one seed fixes one complete replay.
    """
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
        seed=seed,
    )
    for name, ranker in fixed_models.items():
        factories[name] = lambda ranker=ranker: FixedModelScorer(
            trace,
            ranker,
            refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
            sample_size=HYPERPARAMETERS["normalizer_sample"],
            seed=seed,
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


STANDARDIZATIONS = ("per_decision", "train_global", "none", "ranker")

# What the fitted ranker predicts. `binary` is the Phase 0.5 target. The other
# two are the graded alternatives the gap decomposition pointed at; changing
# the target changes nothing else (features, standardisation, penalty).
TARGETS = ("binary", "count", "next_use")


def target_values(trace: Trace, state_ids: list[str], now_ms: float, horizon_seconds: float, target: str) -> np.ndarray:
    """Per-state label for one decision point.

    * `binary`: 1 if the state is used again within the horizon.
    * `count`: log1p(number of uses within the horizon).
    * `next_use`: -log1p(seconds to the next use), floored at the horizon, so a
      state not used within the horizon gets the floor. Everything is censored
      at the horizon, which is what the embargo guarantees is observable.
    """
    import bisect

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    horizon_ms = horizon_seconds * 1000.0
    out = np.empty(len(state_ids), dtype=float)
    for position, state_id in enumerate(state_ids):
        occurrences = trace.occurrences_ms[state_id]
        start = bisect.bisect_right(occurrences, now_ms)
        if target == "binary":
            out[position] = float(start < len(occurrences) and occurrences[start] <= now_ms + horizon_ms)
        elif target == "count":
            end = bisect.bisect_right(occurrences, now_ms + horizon_ms)
            out[position] = math.log1p(end - start)
        else:
            delta = (occurrences[start] - now_ms) / 1000.0 if start < len(occurrences) else math.inf
            out[position] = -math.log1p(min(delta, horizon_seconds))
    return out


def _normalize(matrix: np.ndarray, standardization: str) -> np.ndarray:
    """Apply the decision-point normalisation a standardisation variant asks for.

    * `per_decision`: z-score inside the decision point (the default protocol).
    * `train_global`: no per-point normalisation; the ranker standardises with
      its own training-set statistics, so a transferred model carries the
      *source* workload's feature scales onto the target.
    * `none`: no normalisation anywhere; the ranker is fitted on raw features
      with the L2 penalty acting on raw-unit coefficients.
    * `ranker`: no per-point normalisation either, but the ranker keeps its own
      training mean and scale, so the penalty acts on standardised
      coefficients exactly as under `per_decision`. The difference from
      `per_decision` is then the *population* the scales come from (the whole
      training set rather than each decision point), which is what a scorer
      that applies no population normaliser at replay time reproduces.
    """
    from .temporal import standardize_rows

    if standardization not in STANDARDIZATIONS:
        raise ValueError(f"unknown standardization {standardization!r}")
    return standardize_rows(matrix) if standardization == "per_decision" else matrix


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
    standardization: str = "per_decision",
    target: str = "binary",
):
    """Fit one ranker on the trace's training window only.

    Returns the ranker, the horizon actually used, and the split timestamp. The
    requested horizon is clipped to what the trace can support on both sides of
    the split; that is a property of the trace, not a tuned choice. `target`
    picks what is predicted (see `target_values`); a graded target is fitted
    by ridge regression with the same features, standardisation, and penalty
    scale as the logistic ranker.
    """
    from .phase05 import split_design
    from .predictors import case_control_sample

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")

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
        normalized = _normalize(
            history.feature_matrix(state_ids, timestamp_ms), standardization
        )
        labels = target_values(trace, state_ids, timestamp_ms, horizon, target)
        if target == "binary":
            sampled_features, sampled_labels = case_control_sample(
                normalized, labels.astype(np.int8), negatives_per_positive, rng
            )
        else:
            # Graded targets keep every row; the quota below bounds the size.
            sampled_features, sampled_labels = normalized, labels
        if len(sampled_labels) > quota:
            keep = np.sort(rng.choice(len(sampled_labels), size=quota, replace=False))
            sampled_features, sampled_labels = sampled_features[keep], sampled_labels[keep]
        features_buffer.append(sampled_features)
        labels_buffer.append(sampled_labels)

    stacked_features = np.vstack(features_buffer)
    stacked_labels = np.concatenate(labels_buffer)
    # Only `none` asks the ranker to skip its own standardisation; `ranker`
    # feeds it raw rows and lets it keep the training scales.
    standardize = standardization != "none"
    if target == "binary":
        ranker = LogisticRanker(
            indices=model_indices(model_name), l2=l2, standardize=standardize
        ).fit(stacked_features, stacked_labels)
        positives = int(stacked_labels.sum())
    else:
        ranker = RidgeRanker(
            indices=model_indices(model_name), l2=l2, standardize=standardize
        ).fit(stacked_features, stacked_labels)
        positives = int((stacked_labels > stacked_labels.min()).sum())
    return ranker, horizon, split_ms, positives, len(stacked_labels)


def transfer_metrics(
    trace: Trace,
    rankers: dict[str, LogisticRanker],
    horizons: tuple[int, ...],
    snapshot_count: int = 24,
    candidate_cap: int = 40_000,
    precision_k: int = 100,
    train_fraction: float = HYPERPARAMETERS["train_fraction"],
    seed: int = 0,
    standardization: str = "per_decision",
) -> list[dict[str, object]]:
    """Score externally fitted rankers on this trace's held-out test snapshots."""
    from collections import defaultdict

    from .phase05 import _labels, fast_ranking_metrics, split_design

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
        normalized = _normalize(
            history.feature_matrix(state_ids, timestamp_ms), standardization
        )
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
                    "standardization": standardization,
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
