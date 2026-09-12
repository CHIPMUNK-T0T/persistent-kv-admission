"""Predictability–retention gap: oracle arms, candidate-set logging, seeds.

Phase 0.5 found that future reuse is predictable from history (AUC 0.86–0.94 on
the real traces) but that no causal policy recovers more than about a quarter
of the LRU-to-offline headroom. This module separates the reasons without
inventing a policy.

Every arm here runs through the same sampled-leaf eviction machinery as the
causal arms, so a difference between two arms is a difference in what they
know or what they optimise, never a difference in how they evict:

    signal gap            = oracle_binary            − learned_history
    objective gap         = oracle_next_use_sampled  − oracle_binary
                            (and oracle_binary_per_byte − oracle_binary,
                             oracle_count − oracle_binary)
    candidate-search gap  = offline_next_use (heap)  − oracle_next_use_sampled

`oracle_binary` knows exactly the label the learned arms were trained on. If a
perfect predictor of that label still leaves most of the headroom on the table,
the label is the wrong target and no better predictor of it can help.

The second measurement is the population the eviction decision actually ranks.
Prediction metrics were computed over every observed state; the decision only
ever ranks a handful of retained leaves. `DecisionLogger` records those exact
candidate sets so ranking quality can be measured where the decision is made,
and `PopulationLadder` measures the same scorer on nested populations
(observed ⊃ cached ⊃ leaves) at the shared test snapshots.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .crossworkload import HYPERPARAMETERS, FixedModelScorer, scorer_factories
from .phase05 import _labels, fast_ranking_metrics, split_design
from .predictors import LogisticRanker
from .replay import _occurrence_groups, replay
from .temporal import standardize_rows
from .trace import Request, Trace


ORACLE_KINDS = ("oracle_binary", "oracle_binary_per_byte", "oracle_count", "oracle_next_use")

# Arms whose eviction decisions are logged and whose scorer is measured on the
# population ladder. Oracle arms are logged too, as a check that the logger's
# labels agree with the oracle (its within-decision AUC must be 1).
LOGGED_ARMS = ("lru", "lfu", "learned_history", "learned_history_observed_norm", "online", "oracle_binary")
LADDER_ARMS = ("lru", "lfu", "learned_history", "learned_history_observed_norm", "online")

# Two-sided 97.5% Student-t quantiles by degrees of freedom; 1.96 beyond.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


class OracleScorer:
    """Future-knowledge scorers that share the causal arms' eviction machinery.

    Each returns the quantity a causal predictor would have to know perfectly.
    `oracle_binary` is exactly the label the learned arms were trained on:
    whether the state is used again within `horizon_seconds`. The cache breaks
    ties by recency, the same secondary key every scored arm gets, so a binary
    oracle behaves as "LRU among states the predictor cannot separate". That is
    what a deployed binary predictor would do, and it is the point: the binary
    label carries no information about *which* of two positives, or two
    negatives, to keep.
    """

    time_varying = True

    def __init__(
        self,
        trace: Trace,
        kind: str,
        horizon_seconds: float,
        bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
        size_model: str = HYPERPARAMETERS["size_model"],
    ) -> None:
        if kind not in ORACLE_KINDS:
            raise ValueError(f"unknown oracle {kind!r}")
        self.trace = trace
        self.kind = kind
        self.horizon_ms = horizon_seconds * 1000.0
        self.bytes_per_token = bytes_per_token
        self.size_model = size_model

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        return None

    def _state_bytes(self, state_id: str) -> int:
        tokens = (
            self.trace.states[state_id].block_tokens
            if self.size_model == "packed"
            else self.trace.block_size
        )
        return tokens * self.bytes_per_token

    def next_use_ms(self, state_id: str, now_ms: float) -> float:
        """Absolute time of the first occurrence strictly after `now_ms`."""
        occurrences = self.trace.occurrences_ms[state_id]
        index = bisect.bisect_right(occurrences, now_ms)
        return occurrences[index] if index < len(occurrences) else math.inf

    def reuse_count(self, state_id: str, now_ms: float) -> int:
        occurrences = self.trace.occurrences_ms[state_id]
        start = bisect.bisect_right(occurrences, now_ms)
        end = bisect.bisect_right(occurrences, now_ms + self.horizon_ms)
        return end - start

    def score(self, state_id: str, timestamp_ms: float) -> float:
        if self.kind == "oracle_next_use":
            return -(self.next_use_ms(state_id, timestamp_ms) - timestamp_ms)
        if self.kind == "oracle_count":
            return float(self.reuse_count(state_id, timestamp_ms))
        reused = self.next_use_ms(state_id, timestamp_ms) - timestamp_ms <= self.horizon_ms
        binary = 1.0 if reused else 0.0
        if self.kind == "oracle_binary_per_byte":
            return binary / self._state_bytes(state_id)
        return binary


@dataclass
class _Decision:
    timestamp_ms: float
    candidates: tuple[str, ...]
    scores: tuple[float, ...]
    victim_index: int


class DecisionLogger:
    """Keeps a uniform random sample of eviction decisions from the measured window.

    Memory is bounded by reservoir sampling over decisions, not over candidate
    rows, so every kept decision is complete and within-decision ranking can be
    measured. Labels are derived afterwards from the trace, so one log serves
    every horizon.
    """

    def __init__(
        self,
        trace: Trace,
        measure_from_ms: float | None,
        max_decisions: int = 20_000,
        seed: int = 0,
    ) -> None:
        self.trace = trace
        self.measure_from_ms = measure_from_ms
        self.max_decisions = max_decisions
        self.rng = random.Random(seed)
        self.decisions_seen = 0
        self.kept: list[_Decision] = []

    def __call__(
        self,
        candidates: list[str],
        scores: list[tuple[float, ...]],
        victim_index: int,
        timestamp_ms: float,
        group_index: int,
    ) -> None:
        if self.measure_from_ms is not None and timestamp_ms < self.measure_from_ms:
            return
        record = _Decision(
            timestamp_ms,
            tuple(candidates),
            tuple(float(score[0]) for score in scores),
            victim_index,
        )
        position = self.decisions_seen
        self.decisions_seen += 1
        if len(self.kept) < self.max_decisions:
            self.kept.append(record)
            return
        slot = self.rng.randrange(position + 1)
        if slot < self.max_decisions:
            self.kept[slot] = record

    def _next_use_delta_ms(self, state_id: str, now_ms: float) -> float:
        occurrences = self.trace.occurrences_ms[state_id]
        index = bisect.bisect_right(occurrences, now_ms)
        return occurrences[index] - now_ms if index < len(occurrences) else math.inf

    def rows(
        self,
        bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
        size_model: str = HYPERPARAMETERS["size_model"],
        max_decisions: int | None = None,
    ) -> list[dict[str, object]]:
        """Flat candidate-level rows for inspection; one row per candidate."""
        out = []
        decisions = self.kept if max_decisions is None else self.kept[:max_decisions]
        for decision_id, decision in enumerate(decisions):
            for position, (state_id, score) in enumerate(
                zip(decision.candidates, decision.scores)
            ):
                meta = self.trace.states[state_id]
                tokens = meta.block_tokens if size_model == "packed" else self.trace.block_size
                out.append(
                    {
                        "decision_id": decision_id,
                        "timestamp_ms": decision.timestamp_ms,
                        "state_id": state_id,
                        "score": score,
                        "is_victim": int(position == decision.victim_index),
                        "next_use_delta_ms": self._next_use_delta_ms(
                            state_id, decision.timestamp_ms
                        ),
                        "state_bytes": tokens * bytes_per_token,
                        "prefix_tokens": meta.prefix_tokens,
                        "depth": meta.depth,
                    }
                )
        return out

    def metrics(self, horizons_seconds: tuple[int, ...], precision_k: int = 100) -> list[dict]:
        """Ranking quality measured on the logged candidate sets, per horizon.

        Two views are reported. The pooled view stacks every logged candidate
        row and scores it like the global protocol does. The within-decision
        view only compares candidates that were ranked against each other in
        one decision, which is the comparison the eviction actually makes.
        A decision is uncensored at horizon H only if H seconds of trace remain
        after it, so the label is not truncated by the end of the trace.
        """
        rows = []
        for horizon in horizons_seconds:
            horizon_ms = horizon * 1000.0
            pooled_labels: list[int] = []
            pooled_scores: list[float] = []
            concordant = 0.0
            pairs = 0.0
            per_decision_auc: list[float] = []
            decisions = 0
            both_classes = 0
            all_positive = 0
            all_negative = 0
            victim_positive = 0
            victim_positive_avoidable = 0
            avoidable = 0
            victim_never_reused = 0
            best_never_reused_available = 0
            for decision in self.kept:
                if decision.timestamp_ms + horizon_ms > self.trace.end_ms:
                    continue
                decisions += 1
                deltas = [
                    self._next_use_delta_ms(state_id, decision.timestamp_ms)
                    for state_id in decision.candidates
                ]
                labels = np.fromiter(
                    (1 if delta <= horizon_ms else 0 for delta in deltas), dtype=np.int8,
                    count=len(deltas),
                )
                scores = np.asarray(decision.scores, dtype=float)
                pooled_labels.extend(labels.tolist())
                pooled_scores.extend(scores.tolist())
                positives = int(labels.sum())
                negatives = len(labels) - positives
                victim_label = int(labels[decision.victim_index])
                victim_positive += victim_label
                if negatives > 0:
                    avoidable += 1
                    victim_positive_avoidable += victim_label
                if positives and negatives:
                    both_classes += 1
                    auc, _, _ = fast_ranking_metrics(labels, scores, precision_k)
                    per_decision_auc.append(auc)
                    concordant += auc * positives * negatives
                    pairs += positives * negatives
                elif positives:
                    all_positive += 1
                else:
                    all_negative += 1
                never = [math.isinf(delta) for delta in deltas]
                if any(never):
                    best_never_reused_available += 1
                    victim_never_reused += int(never[decision.victim_index])
            if decisions == 0:
                rows.append(
                    {"horizon_seconds": horizon, "decisions": 0, "status": "no_uncensored_decisions"}
                )
                continue
            pooled = fast_ranking_metrics(
                np.asarray(pooled_labels, dtype=np.int8),
                np.asarray(pooled_scores, dtype=float),
                precision_k,
            )
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "decisions": decisions,
                    "decisions_seen": self.decisions_seen,
                    "candidate_rows": len(pooled_labels),
                    "prevalence": float(np.mean(pooled_labels)),
                    "pooled_auc": pooled[0],
                    "pooled_average_precision": pooled[1],
                    f"pooled_precision_at_{precision_k}": pooled[2],
                    "within_decision_auc_micro": concordant / pairs if pairs else math.nan,
                    "within_decision_auc_macro": (
                        float(np.mean(per_decision_auc)) if per_decision_auc else math.nan
                    ),
                    "decisions_with_both_classes": both_classes / decisions,
                    "decisions_all_positive": all_positive / decisions,
                    "decisions_all_negative": all_negative / decisions,
                    # The regret measures: how often the evicted leaf would have
                    # been reused within the horizon, overall and when a
                    # not-reused candidate was available instead.
                    "victim_positive_rate": victim_positive / decisions,
                    "victim_positive_rate_when_avoidable": (
                        victim_positive_avoidable / avoidable if avoidable else math.nan
                    ),
                    "decisions_avoidable": avoidable / decisions,
                    "victim_never_reused_rate_when_available": (
                        victim_never_reused / best_never_reused_available
                        if best_never_reused_available
                        else math.nan
                    ),
                    "decisions_with_never_reused_candidate": (
                        best_never_reused_available / decisions
                    ),
                    "status": "ok",
                }
            )
        return rows


class PopulationLadder:
    """Score one arm's own ranking on nested populations at shared snapshots.

    observed ⊃ cached ⊃ leaves. `observed` is every state seen so far, the
    population the prediction protocol used. `cached` is the retained set,
    `leaves` the evictable subset. Each is scored with the cache's own score at
    that instant, so the drop between populations isolates the population and
    not the scorer. For a fitted model the protocol score (exact per-snapshot
    standardisation) is also computed on `observed`, to tie back to the
    transfer-matrix numbers.
    """

    def __init__(
        self,
        trace: Trace,
        snapshot_indices: set[int],
        horizons_seconds: tuple[int, ...],
        precision_k: int = 100,
        candidate_cap: int = 40_000,
        seed: int = 0,
    ) -> None:
        self.trace = trace
        self.snapshots = snapshot_indices
        self.horizons = horizons_seconds
        self.precision_k = precision_k
        self.candidate_cap = candidate_cap
        self.rng = np.random.default_rng(seed)
        self.samples: dict[tuple[str, int], list[tuple[float, float, float, float, int]]] = (
            defaultdict(list)
        )

    def __call__(self, cache, group_index: int, timestamp_ms: float) -> None:
        if group_index not in self.snapshots:
            return
        observed = list(cache.frequency)
        if len(observed) > self.candidate_cap:
            picks = np.sort(self.rng.choice(len(observed), size=self.candidate_cap, replace=False))
            observed = [observed[int(position)] for position in picks]
        populations = {
            "observed": observed,
            "cached": list(cache.cached),
            "leaves": list(cache.leaves),
        }
        for name, state_ids in populations.items():
            if len(state_ids) < 2:
                continue
            scores = np.asarray(
                [cache._score(state_id)[0] for state_id in state_ids], dtype=float
            )
            self._record(name, state_ids, scores, timestamp_ms)
        scorer = cache.scorer
        ranker = getattr(scorer, "ranker", None)
        history = getattr(scorer, "history", None)
        if isinstance(ranker, LogisticRanker) and history is not None and len(observed) >= 2:
            protocol = ranker.score(
                standardize_rows(history.feature_matrix(observed, timestamp_ms))
            )
            self._record("observed_protocol", observed, protocol, timestamp_ms)

    def _record(self, name: str, state_ids: list[str], scores: np.ndarray, now_ms: float) -> None:
        for horizon in self.horizons:
            if now_ms + horizon * 1000.0 > self.trace.end_ms:
                continue
            labels = _labels(self.trace, state_ids, now_ms, horizon)
            auc, ap, precision = fast_ranking_metrics(labels, scores, self.precision_k)
            self.samples[(name, horizon)].append(
                (auc, ap, precision, float(labels.mean()), len(labels))
            )

    def rows(self) -> list[dict[str, object]]:
        out = []
        for (name, horizon), values in sorted(self.samples.items()):
            usable = [value for value in values if math.isfinite(value[0])]
            out.append(
                {
                    "population": name,
                    "horizon_seconds": horizon,
                    "auc": _mean([v[0] for v in usable]),
                    "auc_std": _std([v[0] for v in usable]),
                    "average_precision": _mean([v[1] for v in usable]),
                    f"precision_at_{self.precision_k}": _mean([v[2] for v in usable]),
                    "prevalence": _mean([v[3] for v in values]),
                    "population_size": _mean([float(v[4]) for v in values]),
                    "snapshots": len(values),
                    "snapshots_with_both_classes": len(usable),
                }
            )
        return out


@dataclass(frozen=True)
class ArmSpec:
    """One replay arm: how it scores and how it evicts."""

    name: str
    eviction: str  # "sampled" or "heap"
    policy: str | None = None  # built-in policy name when no scorer is used
    scorer: str | None = None  # "baseline:<kind>", "fixed_self", "online", "oracle:<kind>"
    horizon_seconds: float | None = None
    deterministic: bool = False


def arm_specs(fit_horizon_seconds: float, extra_horizons: tuple[int, ...] = (60, 300)) -> list[ArmSpec]:
    """The arm set for one trace. Oracle arms use the trace's own fit horizon."""
    arms = [
        ArmSpec("lru", "sampled", scorer="baseline:lru"),
        ArmSpec("lfu", "sampled", scorer="baseline:lfu"),
        ArmSpec("learned_history", "sampled", scorer="fixed_self"),
        # Same fitted model, standardised against the observed population it
        # was fitted on instead of the retained set. Controls for the
        # train/deploy normalisation mismatch of the arm above.
        ArmSpec("learned_history_observed_norm", "sampled", scorer="fixed_self_observed_norm"),
        ArmSpec("online", "sampled", scorer="online"),
        ArmSpec("oracle_binary", "sampled", scorer="oracle:oracle_binary", horizon_seconds=fit_horizon_seconds),
        ArmSpec("oracle_binary_per_byte", "sampled", scorer="oracle:oracle_binary_per_byte", horizon_seconds=fit_horizon_seconds),
        ArmSpec("oracle_count", "sampled", scorer="oracle:oracle_count", horizon_seconds=fit_horizon_seconds),
        # Same score function as the heap comparator, same eviction machinery
        # as every causal arm: the only difference from offline_next_use is the
        # 16-candidate search.
        ArmSpec("oracle_next_use_sampled", "sampled", policy="offline_next_use"),
        # Deterministic reference arms, run once per budget.
        ArmSpec("lru_heap", "heap", policy="lru", deterministic=True),
        ArmSpec("lfu_heap", "heap", policy="lfu", deterministic=True),
        ArmSpec("offline_next_use", "heap", policy="offline_next_use", deterministic=True),
    ]
    for horizon in extra_horizons:
        if horizon != fit_horizon_seconds:
            arms.append(
                ArmSpec(
                    f"oracle_binary_h{horizon}", "sampled",
                    scorer="oracle:oracle_binary", horizon_seconds=float(horizon),
                )
            )
    return arms


class HorizonMatchedScorer:
    """One fitted ranker per label horizon; the horizon follows the cache's residence time.

    The gap decomposition showed the best label horizon grows with the budget:
    the cache needs to know about reuse within roughly the time a state can
    survive in it. That time is observable online without any parameter, by
    Little's law: residence ≈ capacity / byte-insertion rate, with the rate
    taken as the cumulative average since the start of the run. At each
    decision the ranker whose horizon is nearest (in log space) to the current
    residence estimate is used. Nothing else differs from `FixedModelScorer`:
    same history, same population normalisation, same features.
    """

    time_varying = True

    def __init__(
        self,
        trace: Trace,
        rankers: dict[float, LogisticRanker],
        refresh_every: int = 16,
        sample_size: int = 512,
        seed: int = 0,
        bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
        size_model: str = HYPERPARAMETERS["size_model"],
    ) -> None:
        if not rankers:
            raise ValueError("at least one horizon ranker is required")
        self.base = FixedModelScorer(
            trace, next(iter(rankers.values())), refresh_every, sample_size, seed, normalize_on="cached"
        )
        self.trace = trace
        self.rankers = dict(sorted(rankers.items()))
        self.horizons = np.log(np.array(list(self.rankers), dtype=float))
        self.bytes_per_token = bytes_per_token
        self.size_model = size_model
        self.cache = None
        self.start_ms: float | None = None
        self.inserted_bytes = 0
        self.current_horizon = float(next(iter(self.rankers)))
        self.horizon_groups: dict[float, int] = defaultdict(int)

    def attach(self, cache) -> None:
        self.cache = cache
        self.base.attach(cache)

    def _state_bytes(self, state_id: str) -> int:
        meta = self.trace.states[state_id]
        tokens = meta.block_tokens if self.size_model == "packed" else self.trace.block_size
        return tokens * self.bytes_per_token

    def residence_seconds(self, timestamp_ms: float) -> float:
        if self.cache is None or self.start_ms is None or self.inserted_bytes == 0:
            return math.inf
        elapsed = max((timestamp_ms - self.start_ms) / 1000.0, 1.0)
        rate = self.inserted_bytes / elapsed
        return self.cache.capacity_bytes / rate if rate > 0 else math.inf

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        if self.start_ms is None:
            self.start_ms = timestamp_ms
        if self.cache is not None:
            # observe() runs before insertion, so a state absent from the cache
            # now is one this group will insert.
            new_states = {
                state_id
                for request in requests
                for state_id in request.hash_ids
                if state_id not in self.cache.cached
            }
            self.inserted_bytes += sum(self._state_bytes(state_id) for state_id in new_states)
        self.base.observe(requests, timestamp_ms)
        residence = self.residence_seconds(timestamp_ms)
        if math.isfinite(residence) and residence > 0:
            position = int(np.argmin(np.abs(self.horizons - math.log(residence))))
            self.current_horizon = float(list(self.rankers)[position])
        self.horizon_groups[self.current_horizon] += 1

    def score(self, state_id: str, timestamp_ms: float) -> float:
        row = self.base.history.feature_vector(state_id, timestamp_ms)
        return self.rankers[self.current_horizon].score_row(self.base.normalizer.apply_row(row))


def target_arm_specs(available: dict[str, LogisticRanker]) -> list[ArmSpec]:
    """Arms for the target-change replay: one per fitted target, plus references."""
    arms = [
        ArmSpec("lru", "sampled", scorer="baseline:lru"),
        ArmSpec("lfu", "sampled", scorer="baseline:lfu"),
        ArmSpec("offline_next_use", "heap", policy="offline_next_use", deterministic=True),
    ]
    for key in available:
        arms.append(ArmSpec(f"learned_{key}", "sampled", scorer=f"fixed:{key}"))
    binary_keys = [key for key in available if key.startswith("binary_h")]
    if len(binary_keys) >= 2:
        arms.append(ArmSpec("learned_matched", "sampled", scorer="matched"))
    return arms


def build_scorer(
    spec: ArmSpec,
    trace: Trace,
    ranker: LogisticRanker | None,
    seed: int,
    rankers: dict[str, LogisticRanker] | None = None,
):
    if spec.scorer is None:
        return None
    if spec.scorer.startswith("oracle:"):
        return OracleScorer(trace, spec.scorer.split(":", 1)[1], spec.horizon_seconds)
    if spec.scorer.startswith("fixed:"):
        key = spec.scorer.split(":", 1)[1]
        if rankers is None or key not in rankers:
            raise ValueError(f"no fitted ranker for {key!r}")
        return FixedModelScorer(
            trace,
            rankers[key],
            refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
            sample_size=HYPERPARAMETERS["normalizer_sample"],
            seed=seed,
        )
    if spec.scorer == "matched":
        if rankers is None:
            raise ValueError("matched scorer needs per-horizon rankers")
        by_horizon = {
            float(key.split("_h", 1)[1]): value
            for key, value in rankers.items()
            if key.startswith("binary_h")
        }
        return HorizonMatchedScorer(
            trace,
            by_horizon,
            refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
            sample_size=HYPERPARAMETERS["normalizer_sample"],
            seed=seed,
        )
    factories = scorer_factories(
        trace, {"fixed_self": ranker} if ranker is not None else {}, seed=seed
    )
    if spec.scorer.startswith("baseline:"):
        return factories[spec.scorer.split(":", 1)[1]]()
    if spec.scorer == "fixed_self":
        if ranker is None:
            raise ValueError("learned_history needs a fitted ranker")
        return factories["fixed_self"]()
    if spec.scorer == "fixed_self_observed_norm":
        if ranker is None:
            raise ValueError("learned_history_observed_norm needs a fitted ranker")
        return FixedModelScorer(
            trace,
            ranker,
            refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
            sample_size=HYPERPARAMETERS["normalizer_sample"],
            seed=seed,
            normalize_on="observed",
        )
    if spec.scorer == "online":
        return factories["online"]()
    raise ValueError(f"unknown scorer spec {spec.scorer!r}")


def working_set_bytes(
    trace: Trace,
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
) -> int:
    return sum(
        (meta.block_tokens if size_model == "packed" else trace.block_size) * bytes_per_token
        for meta in trace.states.values()
    )


def run_arm(
    trace: Trace,
    spec: ArmSpec,
    fraction: float,
    seed: int,
    ranker: LogisticRanker | None,
    split_ms: float,
    occurrence_groups: dict[str, list[int]] | None = None,
    log_horizons: tuple[int, ...] = (60, 300, 600),
    ladder_snapshots: set[int] | None = None,
    precision_k: int = 100,
    max_logged_decisions: int = 20_000,
    dump_decisions: int = 0,
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
    sample_width: int = HYPERPARAMETERS["eviction_sample_width"],
    rankers: dict[str, LogisticRanker] | None = None,
) -> dict[str, object]:
    """Replay one arm at one budget and seed, with logging where it applies."""
    groups = occurrence_groups or _occurrence_groups(trace)
    total_bytes = working_set_bytes(trace, bytes_per_token, size_model)
    capacity = max(1, round(total_bytes * fraction))
    scorer = build_scorer(spec, trace, ranker, seed, rankers)
    logger = None
    ladder = None
    if spec.eviction == "sampled" and spec.name in LOGGED_ARMS:
        logger = DecisionLogger(trace, split_ms, max_logged_decisions, seed)
    if ladder_snapshots and spec.name in LADDER_ARMS:
        ladder = PopulationLadder(trace, ladder_snapshots, log_horizons, precision_k, seed=seed)
    result = replay(
        trace,
        spec.policy or spec.name,
        capacity,
        fraction,
        bytes_per_token,
        size_model=size_model,
        occurrence_groups=groups,
        scorer=scorer,
        measure_from_ms=split_ms,
        eviction=spec.eviction,
        sample_width=sample_width,
        seed=seed,
        decision_hook=logger,
        group_hook=ladder,
    )
    base = {
        "trace": trace.name,
        "policy": spec.name,
        "eviction": spec.eviction,
        "capacity_fraction": fraction,
        "seed": seed,
    }
    replay_row = dict(
        base,
        size_model=result.size_model,
        capacity_bytes=result.capacity_bytes,
        working_set_bytes=total_bytes,
        avoided_prefill_tokens=result.avoided_prefill_tokens,
        block_hit_rate=result.block_hit_rate,
        request_hit_rate=result.request_hit_rate,
        measured_requests=result.measured_requests,
        oracle_horizon_seconds=spec.horizon_seconds if spec.horizon_seconds else "",
    )
    if isinstance(scorer, HorizonMatchedScorer):
        total_groups = max(sum(scorer.horizon_groups.values()), 1)
        replay_row["matched_horizon_shares"] = json.dumps(
            {str(int(h)): round(c / total_groups, 3) for h, c in sorted(scorer.horizon_groups.items())}
        )
        replay_row["matched_final_residence_seconds"] = scorer.residence_seconds(trace.end_ms)
    usable_horizons = tuple(
        horizon for horizon in log_horizons if split_ms + horizon * 1000.0 <= trace.end_ms
    )
    candidate_rows = (
        [dict(base, **row) for row in logger.metrics(usable_horizons, precision_k)]
        if logger is not None
        else []
    )
    ladder_rows = [dict(base, **row) for row in ladder.rows()] if ladder is not None else []
    dump = (
        [dict(base, **row) for row in logger.rows(bytes_per_token, size_model, dump_decisions)]
        if logger is not None and dump_decisions > 0
        else []
    )
    return {
        "replay": replay_row,
        "candidate": candidate_rows,
        "ladder": ladder_rows,
        "dump": dump,
    }


def ladder_snapshot_indices(
    trace: Trace, horizons=(10, 60, 300, 600, 1800), train_fraction: float = 0.6, snapshot_count: int = 24
) -> set[int]:
    """The shared held-out test snapshots the prediction protocol scored on."""
    _, test_snapshots, _, _, _ = split_design(trace, horizons, train_fraction, snapshot_count)
    return set(test_snapshots)


# --- multi-seed aggregation ----------------------------------------------------


def summarize(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "ci95_half": math.nan, "n": 0}
    mean = float(np.mean(clean))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95_half": math.nan, "n": 1}
    std = float(np.std(clean, ddof=1))
    t = _T_975.get(n - 1, 1.96)
    return {"mean": mean, "std": std, "ci95_half": t * std / math.sqrt(n), "n": n}


def add_closure(replay_rows: list[dict[str, object]]) -> None:
    """Per-seed HeadroomClosure against the same seed's sampled LRU.

    Reference and comparator match the Phase 0.5 definition: sampled LRU as the
    floor, heap offline-next-use as the headroom comparator. Deterministic arms
    are compared against the seed's LRU too, one row per seed.
    """
    by_key: dict[tuple[str, float, int, str], dict] = {}
    deterministic: dict[tuple[str, float, str], dict] = {}
    for row in replay_rows:
        key = (row["trace"], float(row["capacity_fraction"]), int(row["seed"]), row["policy"])
        by_key[key] = row
        if row["eviction"] == "heap":
            deterministic[(row["trace"], float(row["capacity_fraction"]), row["policy"])] = row
    for row in replay_rows:
        trace, fraction, seed = row["trace"], float(row["capacity_fraction"]), int(row["seed"])
        lru = by_key.get((trace, fraction, seed, "lru"))
        offline = deterministic.get((trace, fraction, "offline_next_use"))
        if lru is None or offline is None:
            row["gain_vs_lru_fraction"] = math.nan
            row["headroom_closure"] = math.nan
            continue
        reference = int(lru["avoided_prefill_tokens"])
        headroom = int(offline["avoided_prefill_tokens"]) - reference
        gain = int(row["avoided_prefill_tokens"]) - reference
        row["gain_vs_lru_fraction"] = gain / max(reference, 1)
        row["headroom_closure"] = gain / headroom if headroom > 0 else math.nan


def aggregate_replay(replay_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, float], list[dict]] = defaultdict(list)
    for row in replay_rows:
        groups[(row["trace"], row["policy"], row["eviction"], float(row["capacity_fraction"]))].append(row)
    out = []
    for (trace, policy, eviction, fraction), rows in sorted(groups.items()):
        entry: dict[str, object] = {
            "trace": trace,
            "policy": policy,
            "eviction": eviction,
            "capacity_fraction": fraction,
            "seeds": len(rows),
        }
        for metric in ("avoided_prefill_tokens", "headroom_closure", "gain_vs_lru_fraction", "block_hit_rate"):
            stats = summarize([row[metric] for row in rows])
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


DECOMPOSITION = (
    # name, minuend, subtrahend
    ("signal_gap", "oracle_binary", "learned_history"),
    ("signal_gap_observed_norm", "oracle_binary", "learned_history_observed_norm"),
    ("signal_gap_vs_lfu", "oracle_binary", "lfu"),
    ("objective_gap_next_use", "oracle_next_use_sampled", "oracle_binary"),
    ("objective_gap_per_byte", "oracle_binary_per_byte", "oracle_binary"),
    ("objective_gap_count", "oracle_count", "oracle_binary"),
    ("candidate_search_gap", "offline_next_use", "oracle_next_use_sampled"),
)


def decomposition_rows(replay_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Seed-paired gap decomposition in closure units and in avoided tokens.

    Each gap is computed within a seed first (so the LRU floor is shared) and
    then summarised across seeds. Deterministic arms contribute the same value
    to every seed.
    """
    index: dict[tuple[str, float, int, str], dict] = {}
    deterministic: dict[tuple[str, float, str], dict] = {}
    seeds_by_pair: dict[tuple[str, float], set[int]] = defaultdict(set)
    for row in replay_rows:
        key = (row["trace"], float(row["capacity_fraction"]))
        index[(*key, int(row["seed"]), row["policy"])] = row
        seeds_by_pair[key].add(int(row["seed"]))
        if row["eviction"] == "heap":
            deterministic[(*key, row["policy"])] = row

    def lookup(trace, fraction, seed, policy):
        row = index.get((trace, fraction, seed, policy))
        if row is None:
            row = deterministic.get((trace, fraction, policy))
        return row

    out = []
    for (trace, fraction), seeds in sorted(seeds_by_pair.items()):
        entry: dict[str, object] = {"trace": trace, "capacity_fraction": fraction, "seeds": len(seeds)}
        for name, minuend, subtrahend in DECOMPOSITION:
            closures, tokens = [], []
            for seed in sorted(seeds):
                left = lookup(trace, fraction, seed, minuend)
                right = lookup(trace, fraction, seed, subtrahend)
                if left is None or right is None:
                    continue
                closures.append(float(left["headroom_closure"]) - float(right["headroom_closure"]))
                tokens.append(
                    int(left["avoided_prefill_tokens"]) - int(right["avoided_prefill_tokens"])
                )
            closure_stats = summarize(closures)
            token_stats = summarize(tokens)
            entry[f"{name}_closure_mean"] = closure_stats["mean"]
            entry[f"{name}_closure_ci95_half"] = closure_stats["ci95_half"]
            entry[f"{name}_tokens_mean"] = token_stats["mean"]
        for policy in ("learned_history", "learned_history_observed_norm", "online", "lfu",
                       "oracle_binary", "oracle_binary_per_byte", "oracle_count",
                       "oracle_next_use_sampled"):
            values = [
                float(row["headroom_closure"])
                for seed in seeds
                if (row := lookup(trace, fraction, seed, policy)) is not None
            ]
            stats = summarize(values)
            entry[f"{policy}_closure_mean"] = stats["mean"]
            entry[f"{policy}_closure_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


def aggregate_candidate(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "prevalence", "pooled_auc", "pooled_average_precision", "pooled_precision_at_100",
        "within_decision_auc_micro", "within_decision_auc_macro",
        "decisions_with_both_classes", "decisions_all_positive", "decisions_all_negative",
        "victim_positive_rate", "victim_positive_rate_when_avoidable", "decisions_avoidable",
        "victim_never_reused_rate_when_available", "decisions_with_never_reused_candidate",
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("status") != "ok":
            continue
        groups[(row["trace"], row["policy"], float(row["capacity_fraction"]), int(row["horizon_seconds"]))].append(row)
    out = []
    for (trace, policy, fraction, horizon), rows in sorted(groups.items()):
        entry: dict[str, object] = {
            "trace": trace,
            "policy": policy,
            "capacity_fraction": fraction,
            "horizon_seconds": horizon,
            "seeds": len(rows),
            "decisions_mean": float(np.mean([row["decisions"] for row in rows])),
            "decisions_seen_mean": float(np.mean([row["decisions_seen"] for row in rows])),
        }
        for metric in metrics:
            if metric not in rows[0]:
                continue
            stats = summarize([row[metric] for row in rows])
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


def aggregate_ladder(ladder_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in ladder_rows:
        groups[(row["trace"], row["policy"], float(row["capacity_fraction"]), row["population"], int(row["horizon_seconds"]))].append(row)
    out = []
    for (trace, policy, fraction, population, horizon), rows in sorted(groups.items()):
        entry: dict[str, object] = {
            "trace": trace,
            "policy": policy,
            "capacity_fraction": fraction,
            "population": population,
            "horizon_seconds": horizon,
            "seeds": len(rows),
            "population_size_mean": float(np.mean([row["population_size"] for row in rows])),
            "snapshots": int(rows[0]["snapshots"]),
        }
        for metric in ("auc", "average_precision", "precision_at_100", "prevalence"):
            stats = summarize([row[metric] for row in rows])
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


def coefficient_cosines(rankers: dict[str, LogisticRanker]) -> list[dict[str, object]]:
    names = sorted(rankers)
    out = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = np.asarray(rankers[left].coefficients, dtype=float)
            b = np.asarray(rankers[right].coefficients, dtype=float)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            out.append(
                {
                    "fitted_on_a": left,
                    "fitted_on_b": right,
                    "cosine": float(a @ b / denominator) if denominator > 0 else math.nan,
                }
            )
    return out


def _mean(values) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(np.mean(clean)) if clean else math.nan


def _std(values) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else math.nan
