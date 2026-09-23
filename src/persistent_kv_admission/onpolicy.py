"""On-policy decision populations, frozen rankers, and audit metrics.

This module implements the data boundary registered in
``docs/onpolicy-learning-plan.md``.  Training and held-out populations are
collected in separate cold replays.  The collector reservoirs only eligible
complete decisions, keeps float64 causal features for exact replay-score
checks, and exposes an explicit float32 conversion only at the existing
``fit_population_ranker`` boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .decisionpop import (
    DECISION_CAP,
    FEATURE_COUNT,
    MIN_DISTINCT_LABELS,
    Rows,
    _Labeller,
    population_metrics,
    spearman,
    state_indices,
)
from .phase05 import _average_ranks, fast_ranking_metrics
from .predictors import LogisticRanker, RidgeRanker
from .temporal import FEATURE_NAMES, TemporalHistory
from .trace import Request, Trace

Window = Literal["train", "test"]


def sha256_path(path: str | Path) -> str:
    """Return the SHA-256 of one artifact without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _LoggedDecision:
    timestamp_ms: float
    trace_group_index: int
    decision_ordinal: int
    states: np.ndarray
    features: np.ndarray
    next_use_delta_ms: np.ndarray
    count_within_h: np.ndarray
    victim_index: int
    arriving_index: int
    arm_score: np.ndarray
    arm_tiebreak: np.ndarray


@dataclass
class DecisionPopulation:
    """A decision population stored as flat rows and complete groups.

    Features are float64.  ``to_fit_rows`` is the only method that rounds them
    to the float32 input precision used by the Phase 0.97 candidate logs.
    ``trace_group_index`` and ``decision_ordinal`` form the stable event key;
    ``group`` is a local dense identifier used for grouped metrics.
    """

    timestamp_ms: np.ndarray
    trace_group_index: np.ndarray
    decision_ordinal: np.ndarray
    state_index: np.ndarray
    features: np.ndarray
    next_use_delta_ms: np.ndarray
    count_within_h: np.ndarray
    group: np.ndarray
    victim: np.ndarray
    arriving: np.ndarray
    arm_score: np.ndarray
    arm_tiebreak: np.ndarray
    horizon_seconds: float
    window: str
    seed: int
    decisions_offered: int
    decisions_eligible: int
    max_decisions: int

    def __post_init__(self) -> None:
        arrays = (
            self.trace_group_index,
            self.decision_ordinal,
            self.state_index,
            self.next_use_delta_ms,
            self.count_within_h,
            self.group,
            self.victim,
            self.arriving,
            self.arm_score,
            self.arm_tiebreak,
        )
        size = len(self.timestamp_ms)
        if any(len(value) != size for value in arrays):
            raise ValueError("decision-population row arrays have different lengths")
        if np.asarray(self.features).shape != (size, FEATURE_COUNT):
            raise ValueError(
                f"features have shape {np.asarray(self.features).shape}, "
                f"expected {(size, FEATURE_COUNT)}"
            )
        if np.asarray(self.features).dtype != np.float64:
            raise ValueError("DecisionPopulation features must be float64")
        if self.window not in ("train", "test"):
            raise ValueError(f"unknown population window {self.window!r}")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        if size:
            if not np.isin(self.victim, (0, 1)).all():
                raise ValueError("victim flags must be binary")
            if not np.isin(self.arriving, (0, 1)).all():
                raise ValueError("arrival flags must be binary")
            if not np.isfinite(self.timestamp_ms).all():
                raise ValueError("decision timestamps must be finite")
            if not np.isfinite(self.features).all():
                raise ValueError("decision features must be finite")
            if not np.isfinite(self.arm_score).all():
                raise ValueError("recorded primary scores must be finite")
            if not np.isfinite(self.arm_tiebreak).all():
                raise ValueError("recorded tie-break scores must be finite")
            unique, starts, counts = np.unique(
                self.group, return_index=True, return_counts=True
            )
            if not np.array_equal(unique, np.arange(len(unique), dtype=unique.dtype)):
                raise ValueError("population groups must be dense from zero")
            if not np.array_equal(starts, np.cumsum(np.r_[0, counts[:-1]])):
                raise ValueError("population groups must be contiguous")
            for start, count in zip(starts, counts):
                block = slice(int(start), int(start + count))
                if int(np.asarray(self.victim)[block].sum()) != 1:
                    raise ValueError("each decision must record exactly one actual victim")
                if int(np.asarray(self.arriving)[block].sum()) not in (0, 1):
                    raise ValueError("each decision may record at most one arrival")
                if len(np.unique(np.asarray(self.timestamp_ms)[block])) != 1:
                    raise ValueError("timestamp varies inside one decision")
                if len(np.unique(np.asarray(self.state_index)[block])) != count:
                    raise ValueError("candidate state repeats inside one decision")
                if len(np.unique(np.asarray(self.trace_group_index)[block])) != 1:
                    raise ValueError("trace group varies inside one decision")
                if len(np.unique(np.asarray(self.decision_ordinal)[block])) != 1:
                    raise ValueError("decision ordinal varies inside one decision")
        if self.decisions_eligible < self.decisions:
            raise ValueError("eligible-decision count is smaller than kept decisions")
        if self.decisions_offered < self.decisions_eligible:
            raise ValueError("offered-decision count is smaller than eligible decisions")

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    @property
    def decisions(self) -> int:
        return int(len(np.unique(self.group))) if len(self) else 0

    @property
    def cap_bound(self) -> bool:
        return self.decisions_eligible > self.max_decisions

    def to_rows(self) -> Rows:
        """Return the existing metric container without changing precision."""

        return Rows(
            np.asarray(self.timestamp_ms, dtype=float),
            np.asarray(self.state_index, dtype=np.int32),
            np.asarray(self.features, dtype=np.float64),
            np.asarray(self.next_use_delta_ms, dtype=float),
            np.asarray(self.count_within_h, dtype=float),
            np.asarray(self.group, dtype=np.int64),
            np.asarray(self.victim, dtype=np.int8),
            np.asarray(self.arriving, dtype=np.int8),
            float(self.horizon_seconds),
            np.asarray(self.arm_score, dtype=float),
            np.asarray(self.arm_tiebreak, dtype=float),
        )

    def to_fit_rows(self) -> Rows:
        """Return the Phase 0.97 fitting boundary with float32 features."""

        rows = self.to_rows()
        rows.features = np.asarray(self.features, dtype=np.float32)
        return rows

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez_compressed(
                handle,
                timestamp_ms=self.timestamp_ms,
                trace_group_index=self.trace_group_index,
                decision_ordinal=self.decision_ordinal,
                state_index=self.state_index,
                features=self.features,
                next_use_delta_ms=self.next_use_delta_ms,
                count_within_h=self.count_within_h,
                group=self.group,
                victim=self.victim,
                arriving=self.arriving,
                arm_score=self.arm_score,
                arm_tiebreak=self.arm_tiebreak,
                horizon_seconds=np.asarray(self.horizon_seconds),
                window=np.asarray(self.window),
                seed=np.asarray(self.seed),
                decisions_offered=np.asarray(self.decisions_offered),
                decisions_eligible=np.asarray(self.decisions_eligible),
                max_decisions=np.asarray(self.max_decisions),
            )
        return sha256_path(target)

    @staticmethod
    def load(path: str | Path) -> "DecisionPopulation":
        with np.load(path, allow_pickle=False) as data:
            return DecisionPopulation(
                timestamp_ms=data["timestamp_ms"],
                trace_group_index=data["trace_group_index"],
                decision_ordinal=data["decision_ordinal"],
                state_index=data["state_index"],
                features=data["features"],
                next_use_delta_ms=data["next_use_delta_ms"],
                count_within_h=data["count_within_h"],
                group=data["group"],
                victim=data["victim"],
                arriving=data["arriving"],
                arm_score=data["arm_score"],
                arm_tiebreak=data["arm_tiebreak"],
                horizon_seconds=float(data["horizon_seconds"]),
                window=str(data["window"]),
                seed=int(data["seed"]),
                decisions_offered=int(data["decisions_offered"]),
                decisions_eligible=int(data["decisions_eligible"]),
                max_decisions=int(data["max_decisions"]),
            )

    @staticmethod
    def from_rows(
        rows: Rows,
        window: Window,
        seed: int = 0,
        source_policy: str | None = None,
    ) -> "DecisionPopulation":
        """Adapt an existing grouped Phase 0.97 log for auxiliary scoring.

        Existing candidate artifacts contain float32 features and no trace-group
        event key.  The values are promoted without inventing precision, and
        their original group id is retained as a local-only trace-group key.

        ``source_policy`` restores the candidate's live ``last_group`` for use
        as the learned policy's secondary key.  The old sampled-LRU logger put
        recency in ``arm_score`` and wrote a zero ``arm_tiebreak``; sampled LFU
        put frequency in ``arm_score`` and recency in ``arm_tiebreak``.  A
        learned rescoring must therefore use the former for LRU/LRU-2hit and
        the latter for LFU.  Omitting the policy preserves the recorded tuple
        and is appropriate only when evaluating that recorded source policy.
        """

        if not rows.grouped:
            raise ValueError("DecisionPopulation.from_rows needs grouped rows")
        if window not in ("train", "test"):
            raise ValueError(f"unknown population window {window!r}")
        if source_policy not in (None, "lru", "lru_2hit", "lfu"):
            raise ValueError(f"unsupported source policy {source_policy!r}")
        starts = np.r_[0, np.flatnonzero(rows.group[1:] != rows.group[:-1]) + 1]
        ends = np.r_[starts[1:], len(rows)]
        dense_group = np.empty(len(rows), dtype=np.int64)
        trace_group = np.empty(len(rows), dtype=np.int64)
        ordinal = np.zeros(len(rows), dtype=np.int64)
        timestamps_seen: dict[float, int] = {}
        for dense, (start, end) in enumerate(zip(starts, ends)):
            block = slice(int(start), int(end))
            timestamp = float(rows.timestamp_ms[start])
            dense_group[block] = dense
            trace_group[block] = int(rows.group[start])
            ordinal[block] = timestamps_seen.get(timestamp, 0)
            timestamps_seen[timestamp] = int(ordinal[start]) + 1
        decisions = len(starts)
        if source_policy in ("lru", "lru_2hit"):
            learned_tiebreak = np.asarray(rows.arm_score, dtype=float)
        else:
            learned_tiebreak = np.asarray(rows.arm_tiebreak, dtype=float)
        return DecisionPopulation(
            timestamp_ms=np.asarray(rows.timestamp_ms, dtype=float),
            trace_group_index=trace_group,
            decision_ordinal=ordinal,
            state_index=np.asarray(rows.state_index, dtype=np.int32),
            features=np.asarray(rows.features, dtype=np.float64),
            next_use_delta_ms=np.asarray(rows.next_use_delta_ms, dtype=float),
            count_within_h=np.asarray(rows.count_within_h, dtype=float),
            group=dense_group,
            victim=np.asarray(rows.victim, dtype=np.int8),
            arriving=np.asarray(rows.arriving, dtype=np.int8),
            arm_score=np.asarray(rows.arm_score, dtype=float),
            arm_tiebreak=learned_tiebreak,
            horizon_seconds=float(rows.horizon_seconds),
            window=window,
            seed=int(seed),
            decisions_offered=decisions,
            decisions_eligible=decisions,
            max_decisions=max(decisions, 1),
        )

    def group_blocks(self):
        """Yield ``(group, row_indices)`` in stored decision order."""

        if not len(self):
            return
        starts = np.r_[0, np.flatnonzero(self.group[1:] != self.group[:-1]) + 1]
        ends = np.r_[starts[1:], len(self)]
        for start, end in zip(starts, ends):
            yield int(self.group[start]), np.arange(start, end, dtype=np.int64)


class DecisionPopulationLogger:
    """Reservoir eligible train or held-out decisions before any split filtering.

    The logger is passed both as ``run_two_tier``'s observer and decision hook.
    Reservoir positions count only eligible decisions, so held-out decisions can
    never displace training decisions.  Complete decisions are the sampling
    unit.  The full trace remains the source of state identities and labels.
    """

    def __init__(
        self,
        trace: Trace,
        horizon_seconds: float,
        split_ms: float,
        end_ms: float,
        window: Window,
        max_decisions: int = DECISION_CAP,
        seed: int = 0,
        state_index: dict[str, int] | None = None,
    ) -> None:
        if window not in ("train", "test"):
            raise ValueError(f"unknown population window {window!r}")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self.trace = trace
        self.history = TemporalHistory(trace)
        self.horizon_seconds = float(horizon_seconds)
        self.split_ms = float(split_ms)
        self.end_ms = float(end_ms)
        self.window = window
        self.max_decisions = int(max_decisions)
        self.seed = int(seed)
        self.label = _Labeller(trace, self.horizon_seconds)
        self.state_index = state_index if state_index is not None else state_indices(trace)
        self.rng = random.Random(seed)
        self.decisions_offered = 0
        self.decisions_eligible = 0
        self._ordinals: dict[int, int] = {}
        self.kept: list[_LoggedDecision] = []

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)

    def _eligible(self, timestamp_ms: float) -> bool:
        label_end = timestamp_ms + self.horizon_seconds * 1000.0
        if self.window == "train":
            return label_end <= self.split_ms
        return timestamp_ms >= self.split_ms and label_end <= self.end_ms

    def __call__(
        self,
        candidates,
        scores,
        victim_index,
        timestamp_ms,
        group_index,
        arriving_index,
    ) -> None:
        ordinal = self._ordinals.get(int(group_index), 0)
        self._ordinals[int(group_index)] = ordinal + 1
        self.decisions_offered += 1
        if not self._eligible(float(timestamp_ms)):
            return
        position = self.decisions_eligible
        self.decisions_eligible += 1
        if len(self.kept) >= self.max_decisions:
            slot = self.rng.randrange(position + 1)
            if slot >= self.max_decisions:
                return
        else:
            slot = None
        width = len(candidates)
        if width == 0:
            raise AssertionError("sampled eviction exposed an empty decision")
        if not (0 <= int(victim_index) < width):
            raise AssertionError("victim index lies outside the candidate set")
        if int(arriving_index) >= width:
            raise AssertionError("arrival index lies outside the candidate set")
        labels = [self.label(state_id, timestamp_ms) for state_id in candidates]
        primary = np.fromiter((float(score[0]) for score in scores), float, width)
        secondary = np.fromiter(
            (float(score[1]) if len(score) > 1 else 0.0 for score in scores),
            float,
            width,
        )
        record = _LoggedDecision(
            timestamp_ms=float(timestamp_ms),
            trace_group_index=int(group_index),
            decision_ordinal=int(ordinal),
            states=np.fromiter(
                (self.state_index[state_id] for state_id in candidates),
                dtype=np.int32,
                count=width,
            ),
            features=np.asarray(
                self.history.feature_matrix(list(candidates), timestamp_ms),
                dtype=np.float64,
            ),
            next_use_delta_ms=np.fromiter(
                (value[0] for value in labels), dtype=float, count=width
            ),
            count_within_h=np.fromiter(
                (value[1] for value in labels), dtype=float, count=width
            ),
            victim_index=int(victim_index),
            arriving_index=int(arriving_index),
            arm_score=primary,
            arm_tiebreak=secondary,
        )
        if slot is None:
            self.kept.append(record)
        else:
            self.kept[slot] = record

    def rows(self) -> DecisionPopulation:
        if not self.kept:
            return DecisionPopulation(
                timestamp_ms=np.zeros(0, dtype=float),
                trace_group_index=np.zeros(0, dtype=np.int64),
                decision_ordinal=np.zeros(0, dtype=np.int64),
                state_index=np.zeros(0, dtype=np.int32),
                features=np.zeros((0, FEATURE_COUNT), dtype=np.float64),
                next_use_delta_ms=np.zeros(0, dtype=float),
                count_within_h=np.zeros(0, dtype=float),
                group=np.zeros(0, dtype=np.int64),
                victim=np.zeros(0, dtype=np.int8),
                arriving=np.zeros(0, dtype=np.int8),
                arm_score=np.zeros(0, dtype=float),
                arm_tiebreak=np.zeros(0, dtype=float),
                horizon_seconds=self.horizon_seconds,
                window=self.window,
                seed=self.seed,
                decisions_offered=self.decisions_offered,
                decisions_eligible=self.decisions_eligible,
                max_decisions=self.max_decisions,
            )
        timestamp, trace_group, ordinal, states = [], [], [], []
        features, deltas, counts = [], [], []
        groups, victims, arrivals, primary, secondary = [], [], [], [], []
        for dense_group, record in enumerate(self.kept):
            width = len(record.states)
            timestamp.append(np.full(width, record.timestamp_ms, dtype=float))
            trace_group.append(
                np.full(width, record.trace_group_index, dtype=np.int64)
            )
            ordinal.append(np.full(width, record.decision_ordinal, dtype=np.int64))
            states.append(record.states)
            features.append(record.features)
            deltas.append(record.next_use_delta_ms)
            counts.append(record.count_within_h)
            groups.append(np.full(width, dense_group, dtype=np.int64))
            victim = np.zeros(width, dtype=np.int8)
            victim[record.victim_index] = 1
            victims.append(victim)
            arriving = np.zeros(width, dtype=np.int8)
            if record.arriving_index >= 0:
                arriving[record.arriving_index] = 1
            arrivals.append(arriving)
            primary.append(record.arm_score)
            secondary.append(record.arm_tiebreak)
        return DecisionPopulation(
            timestamp_ms=np.concatenate(timestamp),
            trace_group_index=np.concatenate(trace_group),
            decision_ordinal=np.concatenate(ordinal),
            state_index=np.concatenate(states),
            features=np.vstack(features).astype(np.float64, copy=False),
            next_use_delta_ms=np.concatenate(deltas),
            count_within_h=np.concatenate(counts),
            group=np.concatenate(groups),
            victim=np.concatenate(victims),
            arriving=np.concatenate(arrivals),
            arm_score=np.concatenate(primary),
            arm_tiebreak=np.concatenate(secondary),
            horizon_seconds=self.horizon_seconds,
            window=self.window,
            seed=self.seed,
            decisions_offered=self.decisions_offered,
            decisions_eligible=self.decisions_eligible,
            max_decisions=self.max_decisions,
        )


def sequential_ranker_score(ranker, features: np.ndarray) -> np.ndarray:
    """Score float64 rows in exactly ``score_row``'s arithmetic order.

    The multiplication occurs before division for every term:
    ``(coefficient * (raw - mean)) / scale``.  Terms are added in the stored
    ``ranker.indices`` order.  This deliberately avoids a BLAS reduction.
    """

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if ranker.coefficients is None or ranker.mean is None or ranker.scale is None:
        raise ValueError("ranker is not fitted")
    out = np.full(len(matrix), float(ranker.intercept), dtype=np.float64)
    for position, index in enumerate(ranker.indices):
        difference = matrix[:, int(index)] - float(ranker.mean[position])
        term = (float(ranker.coefficients[position]) * difference) / float(
            ranker.scale[position]
        )
        out += term
    return out


def _first_tuple_argmin(primary: np.ndarray, secondary: np.ndarray) -> int:
    if not len(primary):
        raise ValueError("cannot choose from an empty decision")
    return min(
        range(len(primary)),
        key=lambda index: (float(primary[index]), float(secondary[index])),
    )


def _dense_tuple_order(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    pairs = np.column_stack((np.asarray(primary, float), np.asarray(secondary, float)))
    _, order = np.unique(pairs, axis=0, return_inverse=True)
    return order.astype(float)


def verify_recorded_argmin(
    population: DecisionPopulation, raise_on_mismatch: bool = True
) -> dict[str, float]:
    """Verify that every recorded victim is the first full-tuple argmin."""

    mismatches = 0
    for _, block in population.group_blocks() or ():
        actual = int(np.flatnonzero(population.victim[block])[0])
        expected = _first_tuple_argmin(
            population.arm_score[block], population.arm_tiebreak[block]
        )
        mismatches += int(actual != expected)
    groups = population.decisions
    result = {
        "recorded_argmin_groups": float(groups),
        "recorded_argmin_mismatches": float(mismatches),
        "recorded_argmin_match_rate": (
            (groups - mismatches) / groups if groups else math.nan
        ),
    }
    if mismatches and raise_on_mismatch:
        raise AssertionError(
            f"{mismatches} of {groups} recorded victims are not full-tuple argmins"
        )
    return result


def _scored_population(
    ranker, population: DecisionPopulation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary = sequential_ranker_score(ranker, population.features)
    # Global dense ranks make the pooled metric meaningful.  Restricting this
    # order to one decision induces the same lexicographic order as ranking the
    # decision alone, so within-decision metrics are unchanged.
    order = _dense_tuple_order(primary, population.arm_tiebreak)
    proposed = np.full(len(population), False, dtype=bool)
    for _, block in population.group_blocks() or ():
        chosen = _first_tuple_argmin(primary[block], population.arm_tiebreak[block])
        proposed[int(block[chosen])] = True
    return primary, order, proposed


def _victim_diagnostics(
    population: DecisionPopulation,
    labels: np.ndarray,
    scores: np.ndarray,
    proposed: np.ndarray,
    target: str,
) -> dict[str, float]:
    actual_lowest = proposed_lowest = agreements = 0
    actual_ranks: list[float] = []
    proposed_ranks: list[float] = []
    tuple_ties = 0
    nonconstant = metric_nan_nonconstant = 0
    rejection = {"decisions": 0, "reused": 0, "avoidable": 0}
    eviction = {"decisions": 0, "reused": 0, "avoidable": 0}
    horizon_ms = population.horizon_seconds * 1000.0
    for _, block in population.group_blocks() or ():
        block_labels = labels[block]
        block_scores = scores[block]
        actual = int(np.flatnonzero(population.victim[block])[0])
        predicted = int(np.flatnonzero(proposed[block])[0])
        actual_lowest += int(block_labels[actual] == block_labels.min())
        proposed_lowest += int(block_labels[predicted] == block_labels.min())
        agreements += int(actual == predicted)
        if len(block) > 1:
            label_ranks = _average_ranks(block_labels) / (len(block) - 1)
            actual_ranks.append(float(label_ranks[actual]))
            proposed_ranks.append(float(label_ranks[predicted]))
        minimum = block_scores.min()
        tuple_ties += int(int((block_scores == minimum).sum()) > 1)
        if len(np.unique(block_labels)) >= MIN_DISTINCT_LABELS:
            nonconstant += 1
            if target == "binary":
                value = fast_ranking_metrics(
                    block_labels.astype(np.int8), block_scores, 100
                )[0]
            else:
                value = spearman(block_scores, block_labels)
            metric_nan_nonconstant += int(not math.isfinite(value))
        arriving_positions = np.flatnonzero(population.arriving[block])
        is_rejection = bool(len(arriving_positions) and actual == int(arriving_positions[0]))
        category = rejection if is_rejection else eviction
        category["decisions"] += 1
        reused = population.next_use_delta_ms[block][actual] <= horizon_ms
        category["reused"] += int(reused)
        if reused:
            category["avoidable"] += int(
                (population.next_use_delta_ms[block] > horizon_ms).any()
            )
    groups = max(population.decisions, 1)
    out = {
        "actual_victim_lowest_label_rate": actual_lowest / groups,
        "proposed_victim_lowest_label_rate": proposed_lowest / groups,
        "actual_victim_label_rank_fraction_mean": (
            float(np.mean(actual_ranks)) if actual_ranks else math.nan
        ),
        "proposed_victim_label_rank_fraction_mean": (
            float(np.mean(proposed_ranks)) if proposed_ranks else math.nan
        ),
        "actual_proposed_victim_agreement_rate": agreements / groups,
        "proposed_minimum_tuple_tie_decisions": float(tuple_ties),
        "nonconstant_label_decisions": float(nonconstant),
        "metric_nan_nonconstant_decisions": float(metric_nan_nonconstant),
    }
    for name, values in (("rejected", rejection), ("evicted", eviction)):
        out[f"actual_{name}_decisions"] = float(values["decisions"])
        out[f"actual_{name}_reused_within_h_share"] = (
            values["reused"] / values["decisions"]
            if values["decisions"]
            else math.nan
        )
        out[f"actual_{name}_avoidable_share"] = (
            values["avoidable"] / values["reused"]
            if values["reused"]
            else math.nan
        )
    return out


def score_population(
    ranker,
    population: DecisionPopulation,
    target: str,
    model_iteration: int | None = None,
) -> dict[str, float | int]:
    """Score one fixed population and separate actual/proposed victim metrics."""

    if not len(population):
        raise ValueError("cannot score an empty decision population")
    primary, order, proposed = _scored_population(ranker, population)
    rows = population.to_rows()
    metrics: dict[str, float | int] = dict(
        population_metrics(order, rows, target, decision_sets=False)
    )
    labels = rows.labels(target)
    metrics.update(
        _victim_diagnostics(population, labels, order, proposed, target)
    )
    metrics.update(_score_precision_diagnostics(primary, population.arm_tiebreak, population))
    metrics.update(verify_recorded_argmin(population, raise_on_mismatch=True))
    recorded_primary_delta = np.abs(primary - population.arm_score)
    metrics["recomputed_recorded_primary_exact"] = int(
        np.array_equal(primary, population.arm_score)
    )
    metrics["recomputed_recorded_primary_max_abs"] = float(
        recorded_primary_delta.max(initial=0.0)
    )
    if model_iteration is not None:
        metrics["model_iteration"] = int(model_iteration)
    return metrics


def score_recorded_population(
    population: DecisionPopulation, target: str
) -> dict[str, float | int]:
    """Score the authoritative tuple used by the logging policy itself."""

    if not len(population):
        raise ValueError("cannot score an empty decision population")
    verify_recorded_argmin(population, raise_on_mismatch=True)
    order = _dense_tuple_order(population.arm_score, population.arm_tiebreak)
    actual = np.asarray(population.victim, dtype=bool)
    rows = population.to_rows()
    metrics: dict[str, float | int] = dict(
        population_metrics(order, rows, target, decision_sets=False)
    )
    metrics.update(
        _victim_diagnostics(population, rows.labels(target), order, actual, target)
    )
    metrics.update(
        _score_precision_diagnostics(
            population.arm_score, population.arm_tiebreak, population
        )
    )
    metrics.update(verify_recorded_argmin(population, raise_on_mismatch=True))
    metrics["scoring_recorded_tuple"] = 1
    return metrics


def _score_precision_diagnostics(
    primary: np.ndarray,
    secondary: np.ndarray,
    population: DecisionPopulation,
) -> dict[str, float]:
    primary_tie_pairs = full_tie_pairs = 0
    positive_margins: list[float] = []
    top_two_margins: list[float] = []
    for _, block in population.group_blocks() or ():
        block_primary = np.asarray(primary[block], dtype=float)
        block_secondary = np.asarray(secondary[block], dtype=float)
        _, primary_counts = np.unique(block_primary, return_counts=True)
        primary_tie_pairs += int(sum(n * (n - 1) // 2 for n in primary_counts))
        pairs = np.column_stack((block_primary, block_secondary))
        _, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        full_tie_pairs += int(sum(n * (n - 1) // 2 for n in pair_counts))
        ordered_primary = np.sort(np.unique(block_primary))
        if len(ordered_primary) > 1:
            positive_margins.extend(np.diff(ordered_primary).tolist())
        if len(block) > 1:
            ordering = sorted(
                range(len(block)),
                key=lambda index: (
                    float(block_primary[index]), float(block_secondary[index])
                ),
            )
            top_two_margins.append(
                float(block_primary[ordering[1]] - block_primary[ordering[0]])
            )
    return {
        "primary_exact_tie_pairs": float(primary_tie_pairs),
        "full_tuple_exact_tie_pairs": float(full_tie_pairs),
        "minimum_positive_primary_pair_margin": (
            min(positive_margins) if positive_margins else math.nan
        ),
        "minimum_top_two_primary_margin": (
            min(top_two_margins) if top_two_margins else math.nan
        ),
    }


def raw_coefficient_vector(ranker, width: int = FEATURE_COUNT) -> np.ndarray:
    """Return fitted slopes in the common raw 23-feature coordinate system."""

    if ranker.coefficients is None or ranker.scale is None:
        raise ValueError("ranker is not fitted")
    vector = np.zeros(width, dtype=float)
    for position, index in enumerate(ranker.indices):
        vector[int(index)] = (
            float(ranker.coefficients[position]) / float(ranker.scale[position])
        )
    return vector


def coefficient_cosine(left, right) -> float:
    left_vector = raw_coefficient_vector(left)
    right_vector = raw_coefficient_vector(right)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0.0:
        return math.nan
    return float(left_vector @ right_vector / denominator)


def _population_index(population: DecisionPopulation):
    index = {}
    for _, block in population.group_blocks() or ():
        key = (
            float(population.timestamp_ms[block[0]]),
            int(population.trace_group_index[block[0]]),
            int(population.decision_ordinal[block[0]]),
        )
        if key in index:
            raise AssertionError(f"duplicate decision event key {key}")
        index[key] = block
    return index


def align_populations(
    left: DecisionPopulation,
    right: DecisionPopulation,
    left_ranker=None,
    right_ranker=None,
) -> dict[str, float]:
    """Align only stable event keys and never force unmatched decisions."""

    left_index = _population_index(left)
    right_index = _population_index(right)
    left_keys, right_keys = set(left_index), set(right_index)
    shared = sorted(left_keys & right_keys)
    exact = 0
    jaccards: list[float] = []
    agreements = 0
    comparable = 0
    left_proposed = right_proposed = None
    if left_ranker is not None:
        left_proposed = _scored_population(left_ranker, left)[2]
    if right_ranker is not None:
        right_proposed = _scored_population(right_ranker, right)[2]
    for key in shared:
        lb, rb = left_index[key], right_index[key]
        left_states = tuple(int(value) for value in left.state_index[lb])
        right_states = tuple(int(value) for value in right.state_index[rb])
        union = set(left_states) | set(right_states)
        intersection = set(left_states) & set(right_states)
        jaccards.append(len(intersection) / len(union) if union else 1.0)
        same = left_states == right_states
        exact += int(same)
        if same and left_proposed is not None and right_proposed is not None:
            left_victim = int(left.state_index[lb][np.flatnonzero(left_proposed[lb])[0]])
            right_victim = int(right.state_index[rb][np.flatnonzero(right_proposed[rb])[0]])
            agreements += int(left_victim == right_victim)
            comparable += 1
    union_keys = left_keys | right_keys
    return {
        "left_decisions": float(len(left_keys)),
        "right_decisions": float(len(right_keys)),
        "shared_event_keys": float(len(shared)),
        "left_only_event_keys": float(len(left_keys - right_keys)),
        "right_only_event_keys": float(len(right_keys - left_keys)),
        "exact_ordered_candidate_sets": float(exact),
        "different_candidate_sets_on_shared_keys": float(len(shared) - exact),
        "exact_match_share_of_shared": exact / len(shared) if shared else math.nan,
        "exact_match_share_of_key_union": exact / len(union_keys) if union_keys else math.nan,
        "candidate_set_jaccard_mean": float(np.mean(jaccards)) if jaccards else math.nan,
        "candidate_set_jaccard_min": float(np.min(jaccards)) if jaccards else math.nan,
        "prediction_agreement_exact_sets": float(agreements),
        "prediction_comparable_exact_sets": float(comparable),
        "prediction_agreement_rate": agreements / comparable if comparable else math.nan,
    }


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def ranker_detail(ranker) -> dict[str, object]:
    """Canonical, sufficient model state plus non-scoring diagnostics."""

    if not isinstance(ranker, (LogisticRanker, RidgeRanker)):
        raise TypeError(f"unsupported ranker class {type(ranker).__name__}")
    if ranker.mean is None or ranker.scale is None or ranker.coefficients is None:
        raise ValueError("cannot serialize an unfitted ranker")
    return {
        "format": "persistent-kv-admission-ranker-v1",
        "class": type(ranker).__name__,
        "indices": [int(value) for value in ranker.indices],
        "l2": float(ranker.l2),
        "standardize": bool(ranker.standardize),
        "mean": [float(value) for value in ranker.mean],
        "scale": [float(value) for value in ranker.scale],
        "coefficients": [float(value) for value in ranker.coefficients],
        "intercept": float(ranker.intercept),
        "converged": bool(getattr(ranker, "converged", False)),
        "iterations": int(getattr(ranker, "iterations", -1)),
        "condition_number": _json_float(
            getattr(ranker, "condition_number", math.nan)
        ),
        "feature_names": [FEATURE_NAMES[index] for index in ranker.indices],
    }


def _ranker_bytes(ranker) -> bytes:
    return (
        json.dumps(
            ranker_detail(ranker),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def serialize_ranker(ranker, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _ranker_bytes(ranker)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def deserialize_ranker(
    path: str | Path, expected_sha256: str | None = None
):
    source = Path(path)
    payload = source.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"ranker hash mismatch for {source}: {actual} != {expected_sha256}"
        )
    data = json.loads(payload)
    if data.get("format") != "persistent-kv-admission-ranker-v1":
        raise ValueError(f"unknown ranker format {data.get('format')!r}")
    common = dict(
        indices=tuple(int(value) for value in data["indices"]),
        l2=float(data["l2"]),
        standardize=bool(data["standardize"]),
    )
    if data["class"] == "LogisticRanker":
        ranker = LogisticRanker(**common)
    elif data["class"] == "RidgeRanker":
        ranker = RidgeRanker(**common)
    else:
        raise ValueError(f"unknown ranker class {data['class']!r}")
    ranker.mean = np.asarray(data["mean"], dtype=float)
    ranker.scale = np.asarray(data["scale"], dtype=float)
    ranker.coefficients = np.asarray(data["coefficients"], dtype=float)
    ranker.intercept = float(data["intercept"])
    ranker.converged = bool(data["converged"])
    ranker.iterations = int(data["iterations"])
    if isinstance(ranker, RidgeRanker):
        condition = data.get("condition_number")
        ranker.condition_number = float(condition) if condition is not None else math.nan
    expected_features = [FEATURE_NAMES[index] for index in ranker.indices]
    if data.get("feature_names") != expected_features:
        raise ValueError("serialized ranker feature names do not match current indices")
    return ranker


class LabelWindowUtilityCollector:
    """Utility counters on the label-observable held-out subwindow."""

    def __init__(self, trace: Trace, start_ms: float, end_ms: float) -> None:
        if end_ms < start_ms:
            raise ValueError("utility window ends before it starts")
        self.trace = trace
        self.start_ms = float(start_ms)
        self.end_ms = float(end_ms)
        self.measured_requests = 0
        self.requested_tokens = 0
        self.requested_blocks = 0
        self.l1_avoided_tokens = 0
        self.l2_avoided_tokens = 0
        self.l2_hit_blocks = 0
        self.present_unusable_tokens = 0
        self.present_unusable_blocks = 0

    def on_request(
        self,
        ids,
        prefix,
        l2_hits,
        present,
        timestamp_ms,
        group_index,
        measured,
    ) -> None:
        del group_index, measured
        if timestamp_ms < self.start_ms or timestamp_ms > self.end_ms:
            return
        states = self.trace.states
        self.measured_requests += 1
        self.requested_tokens += sum(states[state_id].block_tokens for state_id in ids)
        self.requested_blocks += len(ids)
        self.l1_avoided_tokens += sum(
            states[ids[index]].block_tokens for index in range(prefix)
        )
        self.l2_avoided_tokens += sum(
            states[ids[index]].block_tokens for index in l2_hits
        )
        self.l2_hit_blocks += len(l2_hits)
        hit_set = set(l2_hits)
        for index in present:
            if index not in hit_set:
                self.present_unusable_blocks += 1
                self.present_unusable_tokens += states[ids[index]].block_tokens

    def row(self) -> dict[str, int]:
        return {
            "label_window_measured_requests": self.measured_requests,
            "label_window_requested_tokens": self.requested_tokens,
            "label_window_requested_blocks": self.requested_blocks,
            "label_window_l1_avoided_tokens": self.l1_avoided_tokens,
            "label_window_l2_avoided_tokens": self.l2_avoided_tokens,
            "label_window_avoided_prefill_tokens": (
                self.l1_avoided_tokens + self.l2_avoided_tokens
            ),
            "label_window_l2_hit_blocks": self.l2_hit_blocks,
            "label_window_present_unusable_tokens": self.present_unusable_tokens,
            "label_window_present_unusable_blocks": self.present_unusable_blocks,
        }
