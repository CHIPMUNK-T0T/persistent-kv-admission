"""Phase 0.97 — training populations for retention, and how to score them.

Phases 0.5–0.95 fitted one causal history ranker on the *observed* population
(every state seen so far, sampled at shared snapshots) and then asked it to
order the states an eviction actually chooses between. This module supplies the
two other training populations the phase compares against that control, with
the features, target, model, split, and embargo held fixed:

* **victims** — every L1 eviction of the training window, with the 23 causal
  features read at the instant of the eviction;
* **candidates** — the decision sets a *causal* behaviour policy (sampled
  L2-LRU, sampled L2-LFU) puts in front of the lower tier, logged while it runs.
  No future-aware policy generates training data and the behaviour policy's
  identity is never a feature.

Both loggers are shown the request stream through `run_two_tier`'s `observer`
slot, at the same instant `replay.py` shows a deployed scorer its observations,
so a logged feature vector is one a scorer could have computed at that decision.

Standardisation. The population-matched fits take raw features and let the
ranker keep its own training mean and scale, and the replay scorer applies no
population normaliser (`RawFeatureScorer`). The Phase 0.9 control keeps its own
convention (per-decision standardisation at fit time, `FixedModelScorer`
normalising on the ranked population at replay time). For the *predictive*
comparison the two conventions must be honoured at scoring time too, or the
comparison measures the normaliser and not the training population; a control
ranker is therefore scored on the standardised feature matrix of the decision
population it is ranking — the snapshot for `observed`, the logged decision set
for `candidates` — and on the pooled test matrix for `victims`, which has no
decision grouping to standardise inside.

Nothing here re-derives a feature, a target, a fit, or the replay: they come
from `temporal`, `crossworkload`, `predictors`, and `twotier` unchanged.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from .crossworkload import HYPERPARAMETERS, TARGETS
from .phase05 import _average_ranks, fast_ranking_metrics, split_design
from .predictors import LogisticRanker, RidgeRanker, case_control_sample
from .temporal import FEATURE_NAMES, TemporalHistory, model_indices, standardize_rows
from .trace import Request, Trace

FEATURE_COUNT = len(FEATURE_NAMES)
# The label horizons `split_design` offers; `fit_fixed_model` picks from these.
HORIZONS_SECONDS = (10, 60, 300, 600, 1800)
REQUESTED_HORIZON_SECONDS = 600.0
# Populations a ranker can be trained on (A_* are the Phase 0.9 control).
TRAIN_POPULATIONS = ("A_pd", "A_none", "B", "C_lru", "C_lfu", "C_union")
# Behaviour policies whose logged decision sets become a training population.
BEHAVIOUR_POLICIES = ("lru", "lfu")

SNAPSHOT_COUNT = 24
CANDIDATE_CAP = 40_000       # states scored at one observed snapshot
DECISION_CAP = 40_000        # decisions kept by one candidate log
MAX_TRAIN_ROWS = 150_000     # same cap as `fit_fixed_model`
NEGATIVES_PER_POSITIVE = 10.0
SAMPLE_WIDTH = HYPERPARAMETERS["eviction_sample_width"]

# How a ranker's scores are produced on an evaluation population. "raw" feeds
# the feature vector straight in, which is what `RawFeatureScorer` does at
# replay; "per_decision" and "pooled" reproduce the control's own protocol.
SCORINGS = ("raw", "per_decision", "pooled")


def horizon_for(trace: Trace, requested_seconds: float = REQUESTED_HORIZON_SECONDS,
                train_fraction: float = HYPERPARAMETERS["train_fraction"],
                snapshot_count: int = SNAPSHOT_COUNT):
    """The label horizon and split `fit_fixed_model` would use for this trace.

    Same rule, same call: the requested horizon clipped to what the trace can
    label on both sides of the split. Returned separately so the loggers, which
    run before any fit, can store counts over exactly that horizon.
    """
    usable, test_snapshots, _, split_ms, _ = split_design(
        trace, HORIZONS_SECONDS, train_fraction, snapshot_count
    )
    if not usable:
        raise ValueError(f"{trace.name}: no horizon supports an embargoed split")
    horizon = (
        max(value for value in usable if value <= requested_seconds)
        if any(value <= requested_seconds for value in usable)
        else min(usable)
    )
    return float(horizon), float(split_ms), list(test_snapshots)


class RawFeatureScorer:
    """A ranker fitted on raw features, applied to raw features.

    The counterpart of `crossworkload.FixedModelScorer` for the population-
    matched fits: no population normaliser stands between the history and the
    coefficients, because none stood there at fit time either.
    """

    time_varying = True

    def __init__(self, trace: Trace, ranker) -> None:
        self.history = TemporalHistory(trace)
        self.ranker = ranker

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> float:
        return self.ranker.score_row(self.history.feature_vector(state_id, timestamp_ms))


# --- labels ------------------------------------------------------------------


def target_column(
    next_use_delta_ms: np.ndarray, count_within_h: np.ndarray, target: str, horizon_seconds: float
) -> np.ndarray:
    """The three Phase 0.9 targets from the two stored label primitives.

    Identical to `crossworkload.target_values` by construction; the equality is
    asserted in the tests rather than assumed here.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    delta = np.asarray(next_use_delta_ms, dtype=float)
    if target == "binary":
        return (delta <= horizon_seconds * 1000.0).astype(float)
    if target == "count":
        return np.log1p(np.asarray(count_within_h, dtype=float))
    return -np.log1p(np.minimum(delta / 1000.0, horizon_seconds))


class _Labeller:
    """Next-use delta and within-horizon reuse count at a decision point."""

    def __init__(self, trace: Trace, horizon_seconds: float) -> None:
        self.occurrences = trace.occurrences_ms
        self.horizon_ms = horizon_seconds * 1000.0

    def __call__(self, state_id: str, now_ms: float) -> tuple[float, int]:
        occurrences = self.occurrences[state_id]
        start = bisect.bisect_right(occurrences, now_ms)
        delta = occurrences[start] - now_ms if start < len(occurrences) else math.inf
        end = bisect.bisect_right(occurrences, now_ms + self.horizon_ms)
        return delta, end - start


# --- row containers ----------------------------------------------------------


class _FeatureBuffer:
    """Grows an (n, 23) float32 array in chunks.

    A whole-trace victim log is a few hundred thousand rows; holding that many
    Python lists of floats before one big conversion costs an order of magnitude
    more memory than the array itself.
    """

    def __init__(self, width: int = FEATURE_COUNT, chunk: int = 8192) -> None:
        self.width = width
        self.chunk = chunk
        self._chunks: list[np.ndarray] = []
        self._pending: list[list[float]] = []

    def append(self, row: list[float]) -> None:
        self._pending.append(row)
        if len(self._pending) >= self.chunk:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            self._chunks.append(np.asarray(self._pending, dtype=np.float32))
            self._pending = []

    def value(self) -> np.ndarray:
        self._flush()
        if not self._chunks:
            return np.zeros((0, self.width), dtype=np.float32)
        if len(self._chunks) > 1:
            self._chunks = [np.vstack(self._chunks)]
        return self._chunks[0]


@dataclass
class Rows:
    """One population as flat rows: (decision point, state) is the unit.

    `group` numbers the decision point a row belongs to (-1 when the population
    has no decision structure, as the victim stream has). `victim` marks the
    candidate the logging arm evicted and `arriving` the L1 victim that was
    being offered; both are 0 outside a logged decision set. `arm_score` and
    `arm_tiebreak` are the score tuple the logging arm actually ranked this
    candidate by, recorded rather than recomputed, and nan outside a logged
    decision set.
    """

    timestamp_ms: np.ndarray
    state_index: np.ndarray
    features: np.ndarray
    next_use_delta_ms: np.ndarray
    count_within_h: np.ndarray
    group: np.ndarray
    victim: np.ndarray
    arriving: np.ndarray
    horizon_seconds: float
    arm_score: np.ndarray | None = None
    arm_tiebreak: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = len(self.timestamp_ms)
        if self.arm_score is None:
            self.arm_score = np.full(size, math.nan)
        if self.arm_tiebreak is None:
            self.arm_tiebreak = np.full(size, math.nan)

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    @property
    def grouped(self) -> bool:
        return len(self) > 0 and bool(self.group[0] >= 0)

    def select(self, mask: np.ndarray) -> "Rows":
        mask = np.asarray(mask)
        return Rows(
            self.timestamp_ms[mask], self.state_index[mask], self.features[mask],
            self.next_use_delta_ms[mask], self.count_within_h[mask], self.group[mask],
            self.victim[mask], self.arriving[mask], self.horizon_seconds,
            self.arm_score[mask], self.arm_tiebreak[mask],
        )

    def labels(self, target: str) -> np.ndarray:
        return target_column(self.next_use_delta_ms, self.count_within_h, target,
                             self.horizon_seconds)

    def train_mask(self, split_ms: float) -> np.ndarray:
        """Decision points whose whole label horizon lies before the split."""
        return self.timestamp_ms + self.horizon_seconds * 1000.0 <= split_ms

    def test_mask(self, split_ms: float, end_ms: float) -> np.ndarray:
        """Decision points in the evaluation window with an observable label."""
        return (self.timestamp_ms >= split_ms) & (
            self.timestamp_ms + self.horizon_seconds * 1000.0 <= end_ms
        )

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            timestamp_ms=self.timestamp_ms, state_index=self.state_index,
            features=self.features, next_use_delta_ms=self.next_use_delta_ms,
            count_within_h=self.count_within_h, group=self.group, victim=self.victim,
            arriving=self.arriving, horizon_seconds=np.asarray(self.horizon_seconds),
            arm_score=self.arm_score, arm_tiebreak=self.arm_tiebreak,
        )

    @staticmethod
    def load(path) -> "Rows":
        with np.load(path) as data:
            return Rows(
                data["timestamp_ms"], data["state_index"], data["features"],
                data["next_use_delta_ms"], data["count_within_h"], data["group"],
                data["victim"], data["arriving"], float(data["horizon_seconds"]),
                data["arm_score"], data["arm_tiebreak"],
            )


def concatenate(parts: list[Rows]) -> Rows:
    """Stack populations, renumbering groups so they stay distinct."""
    parts = [part for part in parts if len(part) > 0]
    if not parts:
        raise ValueError("nothing to concatenate")
    horizon = parts[0].horizon_seconds
    if any(part.horizon_seconds != horizon for part in parts):
        raise ValueError("cannot concatenate populations labelled at different horizons")
    groups, offset = [], 0
    for part in parts:
        shifted = np.where(part.group >= 0, part.group + offset, -1)
        groups.append(shifted)
        offset += int(part.group.max()) + 1 if len(part) and part.group.max() >= 0 else 0
    return Rows(
        np.concatenate([p.timestamp_ms for p in parts]),
        np.concatenate([p.state_index for p in parts]),
        np.vstack([p.features for p in parts]),
        np.concatenate([p.next_use_delta_ms for p in parts]),
        np.concatenate([p.count_within_h for p in parts]),
        np.concatenate(groups),
        np.concatenate([p.victim for p in parts]),
        np.concatenate([p.arriving for p in parts]),
        horizon,
        np.concatenate([p.arm_score for p in parts]),
        np.concatenate([p.arm_tiebreak for p in parts]),
    )


def deduplicate(rows: Rows) -> tuple[Rows, int]:
    """Drop every repeat of a (timestamp, state), keeping the first.

    Diagnostic only. A single log legitimately repeats a (t, state) pair: the
    same resident can be sampled into two admissions of one timestamp and
    really did face both decisions. This counts those; `union_across_logs` is
    what the union population is built with.
    """
    keys = np.stack((rows.timestamp_ms, rows.state_index.astype(float)))
    _, first = np.unique(keys, axis=1, return_index=True)
    keep = np.sort(first)
    return rows.select(keep), len(rows) - len(keep)


def union_across_logs(parts: list[Rows]) -> tuple[Rows, int]:
    """Concatenate logs, dropping only what an earlier log already exposed.

    Repeats inside one log are kept: they are separate decisions the state
    really faced, and collapsing them would reweight the population. What is
    dropped is the second arm's copy of a (t, state) the first arm already
    supplies, which is the same row twice — features and labels are both
    functions of (state, time) — and would count one exposure twice only
    because two behaviour policies happened to look at it. The union is
    therefore a superset of its first component.
    """
    kept: list[Rows] = []
    seen: set[tuple[float, int]] = set()
    dropped = 0
    for part in parts:
        keys = list(zip(part.timestamp_ms.tolist(), part.state_index.tolist()))
        mask = np.fromiter((key not in seen for key in keys), dtype=bool, count=len(keys))
        dropped += len(keys) - int(mask.sum())
        seen.update(keys)
        kept.append(part.select(mask))
    return concatenate(kept), dropped


# --- loggers -----------------------------------------------------------------


class VictimLogger:
    """Observer and L1-eviction hook: one row per L1 eviction, whole trace.

    Handed to `run_two_tier` as both `observer` and `victim_hook`, with an L2 of
    zero bytes, so the only thing the replay does differently is record.
    """

    def __init__(self, trace: Trace, horizon_seconds: float,
                 state_index: dict[str, int] | None = None) -> None:
        self.trace = trace
        self.history = TemporalHistory(trace)
        self.horizon_seconds = horizon_seconds
        self.label = _Labeller(trace, horizon_seconds)
        self.state_index = state_index if state_index is not None else state_indices(trace)
        self._features = _FeatureBuffer()
        self._timestamp: list[float] = []
        self._state: list[int] = []
        self._delta: list[float] = []
        self._count: list[int] = []

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)

    def __call__(self, state_id: str, timestamp_ms: float, group_index: int) -> None:
        delta, count = self.label(state_id, timestamp_ms)
        self._features.append(self.history.feature_vector(state_id, timestamp_ms))
        self._timestamp.append(timestamp_ms)
        self._state.append(self.state_index[state_id])
        self._delta.append(delta)
        self._count.append(count)

    def rows(self) -> Rows:
        size = len(self._timestamp)
        return Rows(
            np.asarray(self._timestamp, dtype=float),
            np.asarray(self._state, dtype=np.int32),
            self._features.value(),
            np.asarray(self._delta, dtype=float),
            np.asarray(self._count, dtype=float),
            np.full(size, -1, dtype=np.int64),
            np.zeros(size, dtype=np.int8),
            np.zeros(size, dtype=np.int8),
            self.horizon_seconds,
        )


@dataclass
class _Decision:
    timestamp_ms: float
    group_index: int
    states: tuple[int, ...]
    features: np.ndarray
    next_use_delta_ms: tuple[float, ...]
    count_within_h: tuple[int, ...]
    victim_index: int
    arriving_index: int
    # The score tuple the store ranked by, as the hook received it: never
    # recomputed, so the reference rows measure the ordering that really ran.
    arm_score: tuple[float, ...]
    arm_tiebreak: tuple[float, ...]


class CandidateLogger:
    """Reservoir of sampled-L2 decisions with raw features and both labels.

    Modelled on `candidate_models.FeatureLogger`: the whole replay is sampled,
    warm-up included, so one log carries training decisions (before the split,
    with the horizon embargo) and test decisions (after it). Handed to
    `run_two_tier` as `observer` and `l2_decision_hook`.
    """

    def __init__(self, trace: Trace, horizon_seconds: float, max_decisions: int = DECISION_CAP,
                 seed: int = 0, state_index: dict[str, int] | None = None) -> None:
        self.trace = trace
        self.history = TemporalHistory(trace)
        self.horizon_seconds = horizon_seconds
        self.label = _Labeller(trace, horizon_seconds)
        self.state_index = state_index if state_index is not None else state_indices(trace)
        self.max_decisions = max_decisions
        self.rng = random.Random(seed)
        self.decisions_seen = 0
        self.kept: list[_Decision] = []

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.history.observe_requests(requests, timestamp_ms)

    def __call__(self, candidates, scores, victim_index, timestamp_ms, group_index,
                 arriving_index) -> None:
        position = self.decisions_seen
        self.decisions_seen += 1
        if len(self.kept) >= self.max_decisions:
            slot = self.rng.randrange(position + 1)
            if slot >= self.max_decisions:
                return
        else:
            slot = None
        labels = [self.label(state_id, timestamp_ms) for state_id in candidates]
        record = _Decision(
            timestamp_ms,
            group_index,
            tuple(self.state_index[state_id] for state_id in candidates),
            self.history.feature_matrix(list(candidates), timestamp_ms).astype(np.float32),
            tuple(value[0] for value in labels),
            tuple(value[1] for value in labels),
            victim_index,
            arriving_index,
            tuple(float(score[0]) for score in scores),
            tuple(float(score[1]) if len(score) > 1 else 0.0 for score in scores),
        )
        if slot is None:
            self.kept.append(record)
        else:
            self.kept[slot] = record

    def rows(self) -> Rows:
        if not self.kept:
            empty = Rows(np.zeros(0), np.zeros(0, np.int32),
                         np.zeros((0, FEATURE_COUNT), np.float32), np.zeros(0), np.zeros(0),
                         np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, np.int8),
                         self.horizon_seconds)
            return empty
        timestamps, states, deltas, counts, groups, victims, arrivings = [], [], [], [], [], [], []
        blocks, primary, secondary = [], [], []
        for index, record in enumerate(self.kept):
            width = len(record.states)
            blocks.append(record.features)
            primary.append(np.asarray(record.arm_score, dtype=float))
            secondary.append(np.asarray(record.arm_tiebreak, dtype=float))
            timestamps.append(np.full(width, record.timestamp_ms))
            states.append(np.asarray(record.states, dtype=np.int32))
            deltas.append(np.asarray(record.next_use_delta_ms, dtype=float))
            counts.append(np.asarray(record.count_within_h, dtype=float))
            groups.append(np.full(width, index, dtype=np.int64))
            victim = np.zeros(width, dtype=np.int8)
            victim[record.victim_index] = 1
            victims.append(victim)
            arriving = np.zeros(width, dtype=np.int8)
            if record.arriving_index >= 0:
                arriving[record.arriving_index] = 1
            arrivings.append(arriving)
        return Rows(
            np.concatenate(timestamps), np.concatenate(states), np.vstack(blocks),
            np.concatenate(deltas), np.concatenate(counts), np.concatenate(groups),
            np.concatenate(victims), np.concatenate(arrivings), self.horizon_seconds,
            np.concatenate(primary), np.concatenate(secondary),
        )



def lexicographic_score(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """A scalar that induces exactly the order of the (primary, secondary) key.

    The store compares score *tuples*, so a generic policy's ordering is
    lexicographic and cannot be read off its first element alone: LFU ranks by
    frequency and breaks ties by recency. Dense-ranking the pairs reproduces
    that order exactly and, because equal pairs get equal ranks, keeps the tie
    handling the ranking metrics apply.
    """
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if len(primary) == 0:
        return np.zeros(0)
    pairs = np.column_stack((primary, np.nan_to_num(secondary, nan=0.0)))
    _, dense = np.unique(pairs, axis=0, return_inverse=True)
    return dense.astype(float).ravel()


@dataclass
class _OnPolicyDecision:
    timestamp_ms: float
    group_index: int
    states: np.ndarray
    # The store's score tuple, split into its two positions: `tiebreak` is 0.0
    # when the policy's key has one element. `order` is the lexicographic rank
    # of those tuples inside this decision, which is the ordering the store
    # actually compared, computed once here rather than per label target.
    scores: np.ndarray
    tiebreaks: np.ndarray
    order: np.ndarray
    next_use_delta_ms: np.ndarray
    count_within_h: np.ndarray
    victim_index: int
    arriving_index: int


class OnPolicyDecisionLogger:
    """The decisions one arm actually faced, with the scores it actually used.

    The off-policy candidate populations are logged under fixed behaviour
    policies, so every arm is measured on the same decisions — the right
    common condition for comparing rankers. But a learned arm changes what L2
    retains, so the decisions it meets are not those decisions, and its
    accuracy on its own stream is a different quantity. This records that
    stream: no features (the ranker has already spoken), the whole score tuple
    exactly as the store compared it, and the candidate the store actually
    evicted. The tuple matters: the store ranks lexicographically, so LFU's
    recency tie-break and a learned score's own tie-break are part of the
    ordering, and reading only the first element would credit the arm with ties
    it never had. Only decisions inside the evaluation window whose label
    horizon fits before the trace ends are kept.
    """

    def __init__(self, trace: Trace, horizon_seconds: float, split_ms: float, end_ms: float,
                 max_decisions: int = DECISION_CAP, seed: int = 0,
                 state_index: dict[str, int] | None = None) -> None:
        self.horizon_seconds = horizon_seconds
        self.split_ms = split_ms
        self.end_ms = end_ms
        self.label = _Labeller(trace, horizon_seconds)
        self.state_index = state_index if state_index is not None else state_indices(trace)
        self.max_decisions = max_decisions
        self.rng = random.Random(seed)
        self.decisions_seen = 0       # decisions inside the window
        self.decisions_offered = 0    # every decision the store made
        self.kept: list[_OnPolicyDecision] = []

    def __call__(self, candidates, scores, victim_index, timestamp_ms, group_index,
                 arriving_index) -> None:
        self.decisions_offered += 1
        if timestamp_ms < self.split_ms:
            return
        if timestamp_ms + self.horizon_seconds * 1000.0 > self.end_ms:
            return
        position = self.decisions_seen
        self.decisions_seen += 1
        if len(self.kept) >= self.max_decisions:
            slot = self.rng.randrange(position + 1)
            if slot >= self.max_decisions:
                return
        else:
            slot = None
        width = len(candidates)
        labels = [self.label(state_id, timestamp_ms) for state_id in candidates]
        primary = np.fromiter((float(score[0]) for score in scores), float, width)
        secondary = np.fromiter(
            (float(score[1]) if len(score) > 1 else 0.0 for score in scores), float, width
        )
        record = _OnPolicyDecision(
            timestamp_ms,
            group_index,
            np.fromiter((self.state_index[s] for s in candidates), np.int32, width),
            primary,
            secondary,
            lexicographic_score(primary, secondary),
            np.fromiter((value[0] for value in labels), float, width),
            np.fromiter((value[1] for value in labels), float, width),
            victim_index,
            arriving_index,
        )
        if slot is None:
            self.kept.append(record)
        else:
            self.kept[slot] = record

    def metrics(self, target: str, precision_k: int = 100) -> dict[str, float]:
        """Within-decision quality of the ordering the arm really used.

        Candidates are ranked by the lexicographic order of their score tuples,
        which is what the store compared, so a policy's tie-break counts as part
        of its ordering. `evicted_lowest_label_rate` and
        `evicted_positive_rate_when_avoidable` read the candidate the store
        actually removed; `victim_matches_argmin_rate` checks that this is the
        first minimum of the recorded order, which it must be unless the
        recording lost something.
        """
        out: dict[str, float] = {
            "decisions_offered": float(self.decisions_offered),
            "decisions_in_window": float(self.decisions_seen),
            "decisions_kept": float(len(self.kept)),
            "rows": float(sum(len(r.states) for r in self.kept)),
        }
        if not self.kept:
            for key in ("prevalence", "within_decision_micro", "within_decision_macro",
                        "decisions_scored", "decisions_constant_label",
                        "decisions_constant_label_share", "evicted_lowest_label_rate",
                        "evicted_positive_rate_when_avoidable", "victim_matches_argmin_rate"):
                out[key] = math.nan
            return out
        concordant = pairs = 0.0
        per_decision: list[float] = []
        constant = lowest = matches = 0
        evicted_positive = avoidable = 0
        label_sum = label_count = 0.0
        for record in self.kept:
            labels = target_column(record.next_use_delta_ms, record.count_within_h, target,
                                   self.horizon_seconds)
            scores = record.order
            matches += int(int(np.argmin(scores)) == record.victim_index)
            label_sum += float(labels.sum() if target == "binary"
                               else (labels > labels.min()).sum())
            label_count += len(labels)
            lowest += int(labels[record.victim_index] == labels.min())
            if target == "binary":
                positives = float(labels.sum())
                negatives = float(len(labels)) - positives
                if negatives > 0:
                    avoidable += 1
                    evicted_positive += int(labels[record.victim_index] > 0)
            distinct = len(np.unique(labels))
            if distinct < MIN_DISTINCT_LABELS:
                constant += 1
                continue
            if target == "binary":
                positives = float(labels.sum())
                negatives = float(len(labels)) - positives
                value = fast_ranking_metrics(labels.astype(np.int8), scores, precision_k)[0]
                per_decision.append(value)
                concordant += value * positives * negatives
                pairs += positives * negatives
            else:
                value = spearman(scores, labels)
                if math.isfinite(value):
                    per_decision.append(value)
        out["prevalence"] = label_sum / max(label_count, 1.0)
        out["within_decision_micro"] = (concordant / pairs if pairs else math.nan) if target == "binary" else math.nan
        out["within_decision_macro"] = float(np.mean(per_decision)) if per_decision else math.nan
        out["decisions_scored"] = float(len(per_decision))
        out["decisions_constant_label"] = float(constant)
        out["decisions_constant_label_share"] = constant / len(self.kept)
        out["evicted_lowest_label_rate"] = lowest / len(self.kept)
        out["evicted_positive_rate_when_avoidable"] = (
            evicted_positive / avoidable if target == "binary" and avoidable else math.nan
        )
        out["victim_matches_argmin_rate"] = matches / len(self.kept)
        return out


def state_indices(trace: Trace) -> dict[str, int]:
    """Compact integer id per state, so a log stores ints instead of strings."""
    return {state_id: index for index, state_id in enumerate(trace.states)}


def observed_rows(
    trace: Trace,
    horizon_seconds: float,
    test_snapshots: list[int],
    candidate_cap: int = CANDIDATE_CAP,
    seed: int = 0,
    state_index: dict[str, int] | None = None,
) -> Rows:
    """The Phase 0.5 shared test snapshots, as one grouped population.

    One group per snapshot, every state observed so far (capped by a uniform
    draw), features and labels read exactly as the prediction protocol reads
    them.
    """
    index_of = state_index if state_index is not None else state_indices(trace)
    wanted = set(test_snapshots)
    rng = np.random.default_rng(seed)
    label = _Labeller(trace, horizon_seconds)
    history = TemporalHistory(trace)
    timestamps, states, deltas, counts, groups = [], [], [], [], []
    blocks = []
    group = 0
    for index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        history.observe_requests(requests, timestamp_ms)
        if index not in wanted:
            continue
        state_ids = history.observed_state_ids()
        if len(state_ids) > candidate_cap:
            picks = np.sort(rng.choice(len(state_ids), size=candidate_cap, replace=False))
            state_ids = [state_ids[int(position)] for position in picks]
        width = len(state_ids)
        blocks.append(history.feature_matrix(state_ids, timestamp_ms).astype(np.float32))
        pairs = [label(state_id, timestamp_ms) for state_id in state_ids]
        timestamps.append(np.full(width, timestamp_ms))
        states.append(np.fromiter((index_of[s] for s in state_ids), np.int32, width))
        deltas.append(np.asarray([p[0] for p in pairs], dtype=float))
        counts.append(np.asarray([p[1] for p in pairs], dtype=float))
        groups.append(np.full(width, group, dtype=np.int64))
        group += 1
    if not blocks:
        return Rows(np.zeros(0), np.zeros(0, np.int32), np.zeros((0, FEATURE_COUNT), np.float32),
                    np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int8),
                    np.zeros(0, np.int8), horizon_seconds)
    size = sum(len(b) for b in blocks)
    return Rows(
        np.concatenate(timestamps), np.concatenate(states), np.vstack(blocks),
        np.concatenate(deltas), np.concatenate(counts), np.concatenate(groups),
        np.zeros(size, dtype=np.int8), np.zeros(size, dtype=np.int8), horizon_seconds,
    )


# --- fitting -----------------------------------------------------------------


def fit_population_ranker(
    rows: Rows,
    target: str,
    l2: float = HYPERPARAMETERS["offline_l2"],
    model_name: str = HYPERPARAMETERS["offline_model"],
    max_rows: int = MAX_TRAIN_ROWS,
    negatives_per_positive: float = NEGATIVES_PER_POSITIVE,
    seed: int = 0,
):
    """Fit one ranker on raw features of a decision population.

    Same model, penalty, feature set, case-control rule and row cap as
    `crossworkload.fit_fixed_model`; only the rows come from somewhere else.
    Returns (ranker, positives, rows used).
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    if len(rows) == 0:
        raise ValueError("cannot fit on an empty population")
    features = rows.features.astype(float)
    labels = rows.labels(target)
    rng = np.random.default_rng(seed)
    if target == "binary":
        features, labels = case_control_sample(
            features, labels.astype(np.int8), negatives_per_positive, rng
        )
        labels = labels.astype(float)
    if len(labels) > max_rows:
        keep = np.sort(rng.choice(len(labels), size=max_rows, replace=False))
        features, labels = features[keep], labels[keep]
    indices = model_indices(model_name)
    if target == "binary":
        ranker = LogisticRanker(indices=indices, l2=l2, standardize=True).fit(features, labels)
        positives = int(labels.sum())
    else:
        ranker = RidgeRanker(indices=indices, l2=l2, standardize=True).fit(features, labels)
        positives = int((labels > labels.min()).sum())
    return ranker, positives, len(labels)


# --- metrics -----------------------------------------------------------------


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Rank correlation with average ranks for ties; nan if either side is flat."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("spearman needs equal shapes")
    if len(left) < 2:
        return math.nan
    ranked_left = _average_ranks(left) - 0.5 * (len(left) - 1)
    ranked_right = _average_ranks(right) - 0.5 * (len(right) - 1)
    denominator = math.sqrt(
        float((ranked_left * ranked_left).sum()) * float((ranked_right * ranked_right).sum())
    )
    if denominator <= 0.0:
        return math.nan
    return float((ranked_left * ranked_right).sum() / denominator)


def _group_bounds(group: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(group, kind="stable")
    ordered = group[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1] if len(ordered) else np.zeros(0, int)
    ends = np.r_[starts[1:], len(ordered)] if len(starts) else np.zeros(0, int)
    return order, starts, ends


def score_rows(ranker, rows: Rows, scoring: str) -> np.ndarray:
    """Scores under one standardisation convention.

    * `raw` — the population-matched convention: the ranker sees the feature
      vector a `RawFeatureScorer` would hand it at replay.
    * `per_decision` — the Phase 0.9 control's fit protocol: z-score inside the
      decision point before applying the coefficients.
    * `pooled` — the same z-score taken over the whole evaluated population,
      for a population that has no decision grouping.
    """
    if scoring not in SCORINGS:
        raise ValueError(f"unknown scoring convention {scoring!r}")
    features = rows.features.astype(float)
    if len(features) == 0:
        return np.zeros(0)
    if scoring == "raw":
        return ranker.score(features)
    if scoring == "pooled":
        return ranker.score(standardize_rows(features))
    if not rows.grouped:
        raise ValueError("per_decision scoring needs grouped rows")
    scores = np.empty(len(features))
    order, starts, ends = _group_bounds(rows.group)
    for start, end in zip(starts, ends):
        block = order[start:end]
        scores[block] = ranker.score(standardize_rows(features[block]))
    return scores


# A decision separates candidates only when its labels are not all equal. Two
# distinct labels are enough to rank: {no reuse, one reuse} is a real ranking
# problem, so decisions are dropped only when the label carries no order at all.
MIN_DISTINCT_LABELS = 2


def population_metrics(
    scores: np.ndarray,
    rows: Rows,
    target: str,
    decision_sets: bool = False,
    precision_k: int = 100,
) -> dict[str, float]:
    """Pooled and within-decision ranking quality of one score vector.

    Binary targets are measured by AUC, graded ones by Spearman; the
    within-decision statistics are only defined on a grouped population, and
    over decisions with at least `MIN_DISTINCT_LABELS` distinct labels — the
    rest are counted in `decisions_constant_label`, which is the share of the
    population a ranker could not be wrong about. `evicted_lowest_label_rate` —
    the share of decisions whose lowest-scored candidate carries the lowest
    label of its set, ties counting as lowest — is reported only for a
    population whose groups really are eviction decisions.
    """
    labels = rows.labels(target)
    if len(labels) == 0:
        prevalence = math.nan
    elif target == "binary":
        prevalence = float(labels.mean())
    else:
        prevalence = float((labels > labels.min()).mean())
    out: dict[str, float] = {
        "rows": float(len(labels)),
        "groups": float(len(np.unique(rows.group))) if rows.grouped else 0.0,
        "prevalence": prevalence,
    }
    if target == "binary":
        integer = labels.astype(np.int8)
        out["pooled_metric"] = fast_ranking_metrics(integer, scores, precision_k)[0]
    else:
        out["pooled_metric"] = spearman(scores, labels)
    out["within_decision_micro"] = math.nan
    out["within_decision_macro"] = math.nan
    out["decisions_scored"] = math.nan
    out["decisions_constant_label"] = math.nan
    out["decisions_constant_label_share"] = math.nan
    out["evicted_lowest_label_rate"] = math.nan
    if not rows.grouped or len(labels) == 0:
        return out
    order, starts, ends = _group_bounds(rows.group)
    ordered_scores, ordered_labels = scores[order], labels[order]
    concordant = pairs = 0.0
    per_decision: list[float] = []
    lowest = decisions = constant = 0
    for start, end in zip(starts, ends):
        block_scores = ordered_scores[start:end]
        block_labels = ordered_labels[start:end]
        distinct = len(np.unique(block_labels))
        constant += int(distinct < MIN_DISTINCT_LABELS)
        if decision_sets:
            decisions += 1
            lowest += int(block_labels[int(np.argmin(block_scores))] == block_labels.min())
        if distinct < MIN_DISTINCT_LABELS:
            continue
        if target == "binary":
            positives = float(block_labels.sum())
            negatives = float(len(block_labels)) - positives
            value = fast_ranking_metrics(block_labels.astype(np.int8), block_scores, precision_k)[0]
            per_decision.append(value)
            concordant += value * positives * negatives
            pairs += positives * negatives
        else:
            value = spearman(block_scores, block_labels)
            if math.isfinite(value):
                per_decision.append(value)
    if target == "binary":
        out["within_decision_micro"] = concordant / pairs if pairs else math.nan
    out["within_decision_macro"] = float(np.mean(per_decision)) if per_decision else math.nan
    out["decisions_scored"] = float(len(per_decision))
    out["decisions_constant_label"] = float(constant)
    out["decisions_constant_label_share"] = constant / max(len(starts), 1)
    if decision_sets and decisions:
        out["evicted_lowest_label_rate"] = lowest / decisions
    return out


def evaluate(ranker, rows: Rows, target: str, scoring: str, decision_sets: bool = False,
             precision_k: int = 100) -> dict[str, float]:
    """Score a population under one convention and measure the ranking."""
    return population_metrics(
        score_rows(ranker, rows, scoring), rows, target, decision_sets, precision_k
    )


def scoring_for(population_fit: str, rows: Rows) -> str:
    """The convention a ranker fitted on `population_fit` is scored with.

    The control (`A_pd`) was fitted on per-decision standardised features, so it
    is scored that way wherever the population has decision points and against
    the pooled statistics of the population where it has none. Every other fit
    took raw features and is scored raw.
    """
    if population_fit != "A_pd":
        return "raw"
    return "per_decision" if rows.grouped else "pooled"


def coefficient_row(ranker) -> dict[str, float]:
    """Standardised coefficients keyed by feature name."""
    return dict(ranker.standardized_coefficients(FEATURE_NAMES))


def max_coefficient_deviation(ranker, reference: dict[str, float]) -> float:
    """Largest absolute difference against a stored coefficient dict."""
    mine = coefficient_row(ranker)
    if set(mine) != set(reference):
        return math.inf
    return max(abs(mine[name] - float(reference[name])) for name in mine)
