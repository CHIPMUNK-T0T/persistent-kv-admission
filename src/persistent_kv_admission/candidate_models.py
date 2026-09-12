"""Can a stronger history model rank the eviction candidates?

Phase 0.75 measured that the linear history ranker reaches AUC 0.86–0.94 on
observed states but only 0.57–0.63 on the leaves an eviction decision actually
ranks. Concluding from one linear model that "history carries no information
among live states" is weak. This module fits models of increasing capacity on
the *same* 23 causal features over the *same* logged candidate sets, with a
time split that mirrors the prediction protocol (train before the Phase 0.5
split with a horizon embargo, test in the evaluation window), and reports
within-decision ranking quality for each.

If a gradient-boosted model does not move within-decision AUC materially above
the linear model, the missing information is not a modelling-capacity problem
in the history features. If it does, Research 2 is premature.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .crossworkload import HYPERPARAMETERS, FixedModelScorer
from .phase05 import fast_ranking_metrics
from .predictors import LogisticRanker
from .replay import _occurrence_groups, replay
from .temporal import FEATURE_INDEX, FEATURE_NAMES
from .trace import Trace


@dataclass
class _FeatureDecision:
    timestamp_ms: float
    candidates: tuple[str, ...]
    features: np.ndarray  # (k, n_features) float32, raw causal features
    arm_scores: tuple[float, ...]
    next_use_delta_ms: tuple[float, ...]
    state_bytes: tuple[int, ...]
    victim_index: int


class FeatureLogger:
    """Reservoir of eviction decisions with raw feature vectors, whole replay.

    Unlike `gap.DecisionLogger` this keeps decisions from the warm-up too, so a
    model can be fitted on decisions before the split and tested after it.
    """

    def __init__(
        self,
        trace: Trace,
        history,
        max_decisions: int = 40_000,
        seed: int = 0,
        bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
        size_model: str = HYPERPARAMETERS["size_model"],
    ) -> None:
        self.trace = trace
        self.history = history
        self.max_decisions = max_decisions
        self.rng = random.Random(seed)
        self.bytes_per_token = bytes_per_token
        self.size_model = size_model
        self.decisions_seen = 0
        self.kept: list[_FeatureDecision] = []

    def _next_use_delta(self, state_id: str, now_ms: float) -> float:
        occurrences = self.trace.occurrences_ms[state_id]
        index = bisect.bisect_right(occurrences, now_ms)
        return occurrences[index] - now_ms if index < len(occurrences) else math.inf

    def _bytes(self, state_id: str) -> int:
        meta = self.trace.states[state_id]
        tokens = meta.block_tokens if self.size_model == "packed" else self.trace.block_size
        return tokens * self.bytes_per_token

    def __call__(self, candidates, scores, victim_index, timestamp_ms, group_index) -> None:
        position = self.decisions_seen
        self.decisions_seen += 1
        if len(self.kept) >= self.max_decisions:
            slot = self.rng.randrange(position + 1)
            if slot >= self.max_decisions:
                return
        else:
            slot = None
        record = _FeatureDecision(
            timestamp_ms,
            tuple(candidates),
            self.history.feature_matrix(list(candidates), timestamp_ms).astype(np.float32),
            tuple(float(score[0]) for score in scores),
            tuple(self._next_use_delta(state_id, timestamp_ms) for state_id in candidates),
            tuple(self._bytes(state_id) for state_id in candidates),
            victim_index,
        )
        if slot is None:
            self.kept.append(record)
        else:
            self.kept[slot] = record


def log_candidate_features(
    trace: Trace,
    ranker: LogisticRanker,
    fraction: float,
    seed: int = 0,
    max_decisions: int = 40_000,
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
    sample_width: int = HYPERPARAMETERS["eviction_sample_width"],
    occurrence_groups: dict[str, list[int]] | None = None,
) -> FeatureLogger:
    """Replay the `learned_history` arm and log its decisions with features."""
    from .gap import working_set_bytes

    scorer = FixedModelScorer(
        trace,
        ranker,
        refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
        sample_size=HYPERPARAMETERS["normalizer_sample"],
        seed=seed,
    )
    logger = FeatureLogger(trace, scorer.history, max_decisions, seed, bytes_per_token, size_model)
    capacity = max(1, round(working_set_bytes(trace, bytes_per_token, size_model) * fraction))
    replay(
        trace,
        "learned_history",
        capacity,
        fraction,
        bytes_per_token,
        size_model=size_model,
        occurrence_groups=occurrence_groups or _occurrence_groups(trace),
        scorer=scorer,
        eviction="sampled",
        sample_width=sample_width,
        seed=seed,
        decision_hook=logger,
    )
    return logger


@dataclass
class Dataset:
    features: np.ndarray  # (n, d)
    labels: np.ndarray  # (n,) int8
    decision: np.ndarray  # (n,) decision index
    timestamp_ms: np.ndarray  # (n,)
    arm_score: np.ndarray  # (n,)
    is_victim: np.ndarray  # (n,) int8
    decisions: int


def build_dataset(logger: FeatureLogger, horizon_seconds: float, start_ms: float, end_ms: float) -> Dataset:
    """Rows from decisions with start_ms ≤ t and t + H ≤ end_ms (uncensored)."""
    horizon_ms = horizon_seconds * 1000.0
    blocks, labels, decision, stamps, arm, victim = [], [], [], [], [], []
    count = 0
    for index, record in enumerate(logger.kept):
        t = record.timestamp_ms
        if t < start_ms or t + horizon_ms > end_ms:
            continue
        k = len(record.candidates)
        blocks.append(record.features)
        labels.append(np.fromiter((1 if d <= horizon_ms else 0 for d in record.next_use_delta_ms), np.int8, k))
        decision.append(np.full(k, index, dtype=np.int64))
        stamps.append(np.full(k, t))
        arm.append(np.asarray(record.arm_scores, dtype=float))
        v = np.zeros(k, dtype=np.int8)
        v[record.victim_index] = 1
        victim.append(v)
        count += 1
    if not blocks:
        d = len(FEATURE_NAMES)
        return Dataset(np.zeros((0, d)), np.zeros(0, np.int8), np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int8), 0)
    return Dataset(
        np.vstack(blocks).astype(float),
        np.concatenate(labels),
        np.concatenate(decision),
        np.concatenate(stamps),
        np.concatenate(arm),
        np.concatenate(victim),
        count,
    )


def within_decision_metrics(scores: np.ndarray, data: Dataset, precision_k: int = 16) -> dict[str, float]:
    """Within-decision AUC (micro and macro), pooled AUC, and regret if evicting the minimum."""
    order = np.argsort(data.decision, kind="stable")
    scores, labels, decision = scores[order], data.labels[order], data.decision[order]
    starts = np.r_[0, np.flatnonzero(decision[1:] != decision[:-1]) + 1]
    ends = np.r_[starts[1:], len(decision)]
    concordant = pairs = 0.0
    per_decision = []
    regret = avoidable = 0
    both = 0
    for s, e in zip(starts, ends):
        y = labels[s:e]
        z = scores[s:e]
        positives = int(y.sum())
        negatives = len(y) - positives
        if negatives > 0:
            avoidable += 1
            victim = int(np.argmin(z))
            regret += int(y[victim])
        if positives and negatives:
            both += 1
            auc, _, _ = fast_ranking_metrics(y, z, precision_k)
            per_decision.append(auc)
            concordant += auc * positives * negatives
            pairs += positives * negatives
    pooled = fast_ranking_metrics(labels, scores, 100)
    return {
        "within_decision_auc_micro": concordant / pairs if pairs else math.nan,
        "within_decision_auc_macro": float(np.mean(per_decision)) if per_decision else math.nan,
        "pooled_auc": pooled[0],
        "regret_when_avoidable": regret / avoidable if avoidable else math.nan,
        "decisions": int(len(starts)),
        "decisions_with_both_classes": both / max(len(starts), 1),
        "prevalence": float(labels.mean()) if len(labels) else math.nan,
    }


# One hyperparameter set for every workload, budget, and horizon. Not tuned.
GBM_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}


def fit_and_score_models(train: Dataset, test: Dataset, l2: float = HYPERPARAMETERS["offline_l2"]) -> dict[str, np.ndarray]:
    """Return test-set scores for each model, keyed by model name.

    * `recency`, `frequency`: single features, the LRU and LFU orderings.
    * `arm`: the deployed learned_history score as logged (no refit).
    * `linear`: L2 logistic regression refitted on the candidate population.
    * `gbm`: histogram gradient-boosted trees on the same features.
    * `gbm_plus_context`: the same, with each feature also expressed relative
      to the other candidates in its decision (rank within decision), which a
      pointwise model cannot see otherwise. Same feature source, more context.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    out: dict[str, np.ndarray] = {
        "recency": test.features[:, FEATURE_INDEX["neg_log_recency"]],
        "frequency": test.features[:, FEATURE_INDEX["log_frequency"]],
        "arm": test.arm_score,
    }
    if train.decisions == 0 or train.labels.sum() == 0 or train.labels.sum() == len(train.labels):
        return out
    linear = LogisticRanker(indices=tuple(range(train.features.shape[1])), l2=l2).fit(train.features, train.labels)
    out["linear"] = linear.score(test.features)

    gbm = HistGradientBoostingClassifier(**GBM_PARAMS).fit(train.features, train.labels)
    out["gbm"] = gbm.predict_proba(test.features)[:, 1]

    train_context = np.hstack((train.features, _within_decision_ranks(train)))
    test_context = np.hstack((test.features, _within_decision_ranks(test)))
    gbm_context = HistGradientBoostingClassifier(**GBM_PARAMS).fit(train_context, train.labels)
    out["gbm_plus_context"] = gbm_context.predict_proba(test_context)[:, 1]
    return out


def _within_decision_ranks(data: Dataset) -> np.ndarray:
    """Each feature's fractional dense rank among the candidates of its own decision.

    Ties share a rank and a constant column ranks 0 everywhere, so the draw
    order of the candidates never leaks into the context features.
    """
    ranks = np.zeros_like(data.features)
    order = np.argsort(data.decision, kind="stable")
    decision = data.decision[order]
    starts = np.r_[0, np.flatnonzero(decision[1:] != decision[:-1]) + 1]
    ends = np.r_[starts[1:], len(decision)]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        if len(rows) == 1:
            continue
        block = data.features[rows]
        for column in range(block.shape[1]):
            _, dense = np.unique(block[:, column], return_inverse=True)
            distinct = int(dense.max())
            if distinct > 0:
                ranks[rows, column] = dense / distinct
    return ranks


def evaluate_candidate_models(
    logger: FeatureLogger,
    trace: Trace,
    split_ms: float,
    horizons_seconds: tuple[int, ...],
) -> list[dict[str, object]]:
    """Time-split evaluation: train before the split with embargo, test after."""
    rows = []
    for horizon in horizons_seconds:
        train = build_dataset(logger, horizon, 0.0, split_ms)
        test = build_dataset(logger, horizon, split_ms, trace.end_ms)
        if test.decisions == 0 or train.decisions == 0:
            rows.append({"horizon_seconds": horizon, "status": "no_data", "train_decisions": train.decisions, "test_decisions": test.decisions})
            continue
        scores = fit_and_score_models(train, test)
        for model, values in scores.items():
            metrics = within_decision_metrics(values, test)
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "model": model,
                    "train_decisions": train.decisions,
                    "train_rows": len(train.labels),
                    "train_prevalence": float(train.labels.mean()),
                    "test_decisions": test.decisions,
                    "test_rows": len(test.labels),
                    **metrics,
                    "status": "ok",
                }
            )
    return rows
