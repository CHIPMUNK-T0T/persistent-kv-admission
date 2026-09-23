"""Boundary and numeric-contract tests for on-policy population learning."""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import itertools
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from persistent_kv_admission.decisionpop import Rows, fit_population_ranker
from persistent_kv_admission.onpolicy import (
    DecisionPopulation,
    DecisionPopulationLogger,
    LabelWindowUtilityCollector,
    align_populations,
    deserialize_ranker,
    score_population,
    sequential_ranker_score,
    serialize_ranker,
    verify_recorded_argmin,
)
from persistent_kv_admission.onpolicy_reporting import (
    BarrierValidation,
    PI0_REPLAY_EXACT_FIELDS,
    PI0_REPLAY_TEXT_FIELDS,
    ReferenceMismatchError,
    classify_registered_outcome,
    validate_attribution_partition,
    validate_4x4_matrix,
    validate_training_model_barrier,
    verify_candidate_log_metadata,
    verify_published_pi0_replay,
)
from persistent_kv_admission.predictors import LogisticRanker, RidgeRanker
from persistent_kv_admission.temporal import FEATURE_NAMES, model_indices
from persistent_kv_admission.trace import Request, StateMeta, Trace
from persistent_kv_admission.twotier import run_two_tier

from scripts import run_onpolicy_learning as runner


def _request_trace(chains: list[tuple[str, ...]], step_ms: float = 1_000.0) -> Trace:
    """Build a small causal prefix-tree trace without filesystem fixtures."""
    requests: list[Request] = []
    states: dict[str, StateMeta] = {}
    occurrences: dict[str, list[float]] = defaultdict(list)
    children: dict[str, set[str]] = defaultdict(set)
    terminal_branches: dict[str, set[str]] = defaultdict(set)
    for order, chain in enumerate(chains):
        timestamp_ms = order * step_ms
        requests.append(Request(order, timestamp_ms, len(chain), 0, chain))
        terminal = chain[-1]
        for index, state_id in enumerate(chain):
            parent_id = chain[index - 1] if index else None
            meta = StateMeta(state_id, parent_id, index + 1, index + 1, 1)
            if state_id in states:
                assert states[state_id] == meta
            states[state_id] = meta
            occurrences[state_id].append(timestamp_ms)
            terminal_branches[state_id].add(terminal)
            if parent_id is not None:
                children[parent_id].add(state_id)
    return Trace(
        name="onpolicy-toy",
        requests=requests,
        states=states,
        occurrences_ms=dict(occurrences),
        children=dict(children),
        terminal_branches=dict(terminal_branches),
        block_size=1,
    )


def _branching_trace(steps: int = 30) -> Trace:
    chains: list[tuple[str, ...]] = []
    for step in range(steps):
        if step % 5 == 0:
            chains.append(("root", "hot"))
        elif step % 7 == 0:
            chains.append(("root", "hot", f"branch-{step % 3}"))
        else:
            chains.append(("root", f"cold-{step}"))
    return _request_trace(chains)


class _Observer:
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.timestamps.append(timestamp_ms)


class _StableScorer(_Observer):
    time_varying = True

    def score(self, state_id: str, timestamp_ms: float) -> float:
        # Stable across processes and independent of Python's randomized hash.
        return float(sum((index + 1) * ord(value) for index, value in enumerate(state_id)))


def _record_decisions(target: list[tuple]) -> object:
    def record(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
        target.append(
            (
                tuple(candidates),
                tuple(tuple(score) for score in scores),
                victim_index,
                timestamp_ms,
                group_index,
                arriving_index,
            )
        )

    return record


def _population(
    *,
    features=None,
    primary=None,
    secondary=None,
    victim=None,
    deltas=None,
    counts=None,
) -> DecisionPopulation:
    """Construct two complete two-candidate decisions for focused tests."""
    size = 4
    return DecisionPopulation(
        timestamp_ms=np.asarray([10.0, 10.0, 11.0, 11.0]),
        trace_group_index=np.asarray([10, 10, 11, 11], dtype=np.int64),
        decision_ordinal=np.zeros(size, dtype=np.int64),
        state_index=np.arange(size, dtype=np.int32),
        features=(
            np.zeros((size, len(FEATURE_NAMES)), dtype=np.float64)
            if features is None
            else np.asarray(features, dtype=np.float64)
        ),
        next_use_delta_ms=(
            np.asarray([1_000.0, math.inf, math.inf, 2_000.0])
            if deltas is None
            else np.asarray(deltas, dtype=float)
        ),
        count_within_h=(
            np.asarray([1.0, 0.0, 0.0, 1.0])
            if counts is None
            else np.asarray(counts, dtype=float)
        ),
        group=np.asarray([0, 0, 1, 1], dtype=np.int64),
        victim=(
            np.asarray([0, 1, 1, 0], dtype=np.int8)
            if victim is None
            else np.asarray(victim, dtype=np.int8)
        ),
        arriving=np.asarray([1, 0, 1, 0], dtype=np.int8),
        arm_score=(
            np.asarray([1.0, 1.0, 0.0, 2.0])
            if primary is None
            else np.asarray(primary, dtype=float)
        ),
        arm_tiebreak=(
            np.asarray([5.0, 3.0, 0.0, 0.0])
            if secondary is None
            else np.asarray(secondary, dtype=float)
        ),
        horizon_seconds=30.0,
        window="test",
        seed=3,
        decisions_offered=2,
        decisions_eligible=2,
        max_decisions=10,
    )


class ReplayCutoffTests(unittest.TestCase):
    def test_cutoff_group_is_not_observed_or_counted(self):
        trace = _branching_trace(8)
        observer = _Observer()
        result = run_two_tier(
            trace,
            "lru",
            2,
            0,
            bytes_per_token=1,
            observer=observer,
            stop_before_ms=3_000.0,
        )

        self.assertEqual(observer.timestamps, [0.0, 1_000.0, 2_000.0])
        self.assertEqual(result.measured_requests, 3)
        self.assertEqual(result.requested_tokens, 6)

    def test_cutoff_does_not_integrate_the_interval_ending_at_cutoff(self):
        trace = _branching_trace(10)
        prefix = run_two_tier(
            trace,
            "lru",
            2,
            3,
            "lru",
            bytes_per_token=1,
            l2_eviction="sampled",
            l2_sample_width=2,
            l2_seed=4,
            stop_before_ms=5_000.0,
        )
        shorter = Trace(
            name=trace.name,
            requests=[request for request in trace.requests if request.timestamp_ms < 5_000.0],
            # Keep the full-trace metadata intentionally: the experiment fixes
            # capacities, state identities, and labels from the full trace.
            states=trace.states,
            occurrences_ms=trace.occurrences_ms,
            children=trace.children,
            terminal_branches=trace.terminal_branches,
            block_size=trace.block_size,
        )
        explicit = run_two_tier(
            shorter,
            "lru",
            2,
            3,
            "lru",
            bytes_per_token=1,
            l2_eviction="sampled",
            l2_sample_width=2,
            l2_seed=4,
        )

        self.assertEqual(prefix.as_row(), explicit.as_row())

    def test_prefix_and_full_replay_have_identical_pre_cutoff_decisions(self):
        trace = _branching_trace()
        cutoff = 18_000.0
        prefix_decisions: list[tuple] = []
        full_decisions: list[tuple] = []

        def replay(stop_before_ms, decisions):
            scorer = _StableScorer()
            result = run_two_tier(
                trace,
                "lru",
                2,
                4,
                "learned",
                bytes_per_token=1,
                l2_eviction="sampled",
                l2_sample_width=3,
                l2_seed=3,
                l2_scorer=scorer,
                l2_decision_hook=_record_decisions(decisions),
                stop_before_ms=stop_before_ms,
            )
            return scorer, result

        prefix_scorer, prefix_result = replay(cutoff, prefix_decisions)
        full_scorer, _ = replay(None, full_decisions)
        full_prefix = [decision for decision in full_decisions if decision[3] < cutoff]

        self.assertGreater(len(prefix_decisions), 0)
        self.assertEqual(prefix_decisions, full_prefix)
        self.assertEqual(prefix_result.l2_decisions, len(prefix_decisions))
        self.assertEqual(prefix_scorer.timestamps, full_scorer.timestamps[:18])
        self.assertNotIn(cutoff, prefix_scorer.timestamps)


class DecisionPopulationLoggerTests(unittest.TestCase):
    @staticmethod
    def _offer(logger, requests, timestamp_ms, group_index):
        logger.observe(requests, timestamp_ms)
        candidates = list(requests[0].hash_ids)
        scores = [(float(index), float(group_index)) for index in range(len(candidates))]
        logger(candidates, scores, 0, timestamp_ms, group_index, 0)

    def test_future_decisions_never_compete_for_training_reservoir_slots(self):
        trace = _branching_trace(10)

        def collect(offer_future):
            logger = DecisionPopulationLogger(
                trace,
                horizon_seconds=2.0,
                split_ms=5_000.0,
                end_ms=trace.end_ms,
                window="train",
                max_decisions=2,
                seed=7,
            )
            for group_index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
                if timestamp_ms <= 3_000.0 or offer_future:
                    self._offer(logger, requests, timestamp_ms, group_index)
                else:
                    # Future history may be observed in a full replay, but its
                    # decisions must never enter or perturb the train reservoir.
                    logger.observe(requests, timestamp_ms)
            return logger.rows()

        prefix_only = collect(False)
        with_future = collect(True)
        self.assertEqual(prefix_only.decisions_eligible, 4)
        self.assertEqual(with_future.decisions_eligible, 4)
        self.assertEqual(prefix_only.decisions, 2)
        self.assertEqual(with_future.decisions, 2)
        self.assertEqual(prefix_only.decisions_offered, 4)
        self.assertEqual(with_future.decisions_offered, 10)
        for name in (
            "timestamp_ms",
            "trace_group_index",
            "decision_ordinal",
            "state_index",
            "features",
            "next_use_delta_ms",
            "count_within_h",
            "group",
            "victim",
            "arriving",
            "arm_score",
            "arm_tiebreak",
        ):
            np.testing.assert_array_equal(getattr(prefix_only, name), getattr(with_future, name))
        for _, block in with_future.group_blocks():
            self.assertEqual(int(with_future.victim[block].sum()), 1)

    def test_train_and_test_embargo_boundaries_are_inclusive_only_at_registered_edges(self):
        trace = _branching_trace(10)
        first_timestamp, first_requests = next(trace.timestamp_groups())
        candidates = list(first_requests[0].hash_ids)
        scores = [(0.0, 0.0), (1.0, 0.0)]

        train = DecisionPopulationLogger(trace, 2.0, 5_000.0, 9_000.0, "train")
        train.observe(first_requests, first_timestamp)
        for group_index, timestamp_ms in enumerate((3_000.0, 3_000.0001)):
            train(candidates, scores, 0, timestamp_ms, group_index, 0)
        np.testing.assert_array_equal(np.unique(train.rows().timestamp_ms), [3_000.0])

        test = DecisionPopulationLogger(trace, 2.0, 5_000.0, 9_000.0, "test")
        test.observe(first_requests, first_timestamp)
        for group_index, timestamp_ms in enumerate(
            (4_999.9999, 5_000.0, 7_000.0, 7_000.0001)
        ):
            test(candidates, scores, 0, timestamp_ms, group_index, 0)
        np.testing.assert_array_equal(
            np.unique(test.rows().timestamp_ms), [5_000.0, 7_000.0]
        )

    def test_reservoir_cap_keeps_complete_variable_width_decisions(self):
        trace = _request_trace(
            [
                ("root", "a", "a-leaf"),
                ("root", "b"),
                ("root", "c", "c-leaf"),
                ("root", "d"),
                ("root", "e", "e-leaf"),
            ]
        )
        logger = DecisionPopulationLogger(
            trace,
            horizon_seconds=1.0,
            split_ms=10_000.0,
            end_ms=trace.end_ms,
            window="train",
            max_decisions=2,
            seed=11,
        )
        expected_width = {}
        for group_index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
            logger.observe(requests, timestamp_ms)
            candidates = list(requests[0].hash_ids)
            expected_width[timestamp_ms] = len(candidates)
            logger(
                candidates,
                [(float(index), float(index)) for index in range(len(candidates))],
                0,
                timestamp_ms,
                group_index,
                len(candidates) - 1,
            )

        population = logger.rows()
        self.assertEqual(population.decisions_eligible, 5)
        self.assertEqual(population.decisions, 2)
        self.assertTrue(population.cap_bound)
        for _, block in population.group_blocks():
            self.assertEqual(len(block), expected_width[population.timestamp_ms[block[0]]])
            self.assertEqual(int(population.victim[block].sum()), 1)
            self.assertEqual(int(population.arriving[block].sum()), 1)

    def test_labels_skip_all_same_timestamp_occurrences_and_include_exact_horizon(self):
        requests = [
            Request(0, 1_000.0, 2, 0, ("root", "same-only")),
            Request(1, 1_000.0, 2, 0, ("root", "at-horizon")),
            Request(2, 3_000.0, 2, 0, ("root", "at-horizon")),
        ]
        states = {
            "root": StateMeta("root", None, 1, 1, 1),
            "same-only": StateMeta("same-only", "root", 2, 2, 1),
            "at-horizon": StateMeta("at-horizon", "root", 2, 2, 1),
        }
        trace = Trace(
            name="same-timestamp",
            requests=requests,
            states=states,
            occurrences_ms={
                "root": [1_000.0, 1_000.0, 3_000.0],
                "same-only": [1_000.0],
                "at-horizon": [1_000.0, 3_000.0],
            },
            children={"root": {"same-only", "at-horizon"}},
            terminal_branches={
                "root": {"same-only", "at-horizon"},
                "same-only": {"same-only"},
                "at-horizon": {"at-horizon"},
            },
            block_size=1,
        )
        logger = DecisionPopulationLogger(
            trace,
            horizon_seconds=2.0,
            split_ms=3_000.0,
            end_ms=trace.end_ms,
            window="train",
        )
        logger.observe(requests[:2], 1_000.0)
        logger(
            ["same-only", "at-horizon"],
            [(0.0, 0.0), (1.0, 0.0)],
            0,
            1_000.0,
            0,
            1,
        )
        population = logger.rows()

        self.assertTrue(math.isinf(population.next_use_delta_ms[0]))
        self.assertEqual(population.count_within_h[0], 0.0)
        self.assertEqual(population.next_use_delta_ms[1], 2_000.0)
        self.assertEqual(population.count_within_h[1], 1.0)

    def test_float64_population_roundtrip_and_explicit_float32_fit_boundary(self):
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
        features[:, 0] = np.asarray(
            [1.0 + 2.0**-30, 2.0 + 2.0**-29, 3.0 + 2.0**-28, 4.0 + 2.0**-27]
        )
        population = _population(features=features)
        self.assertEqual(population.features.dtype, np.float64)
        fit_rows = population.to_fit_rows()
        self.assertEqual(fit_rows.features.dtype, np.float32)
        np.testing.assert_array_equal(fit_rows.features, features.astype(np.float32))
        self.assertNotEqual(population.features[0, 0], float(fit_rows.features[0, 0]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.npz"
            digest = population.save(path)
            loaded = DecisionPopulation.load(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(loaded.features.dtype, np.float64)
            np.testing.assert_array_equal(loaded.features, population.features)
            np.testing.assert_array_equal(loaded.arm_score, population.arm_score)

    def test_fit_normalizer_comes_only_from_current_float32_population(self):
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
        features[:, 0] = [1.0 + 2.0**-30, 2.0, 4.0, 8.0]
        first = _population(features=features).to_fit_rows()
        first_ranker, _, _ = fit_population_ranker(first, "next_use", seed=0)
        expected_first = first.features[:, model_indices("base2")].astype(float).mean(axis=0)
        np.testing.assert_array_equal(first_ranker.mean, expected_first)

        shifted = features.copy()
        shifted[:, 0] += 100.0
        second = _population(features=shifted).to_fit_rows()
        second_ranker, _, _ = fit_population_ranker(second, "next_use", seed=0)
        expected_second = second.features[:, model_indices("base2")].astype(float).mean(axis=0)
        np.testing.assert_array_equal(second_ranker.mean, expected_second)
        self.assertNotEqual(first_ranker.mean[0], second_ranker.mean[0])


class SequentialScoreAndTupleTests(unittest.TestCase):
    def test_sequential_score_matches_score_row_and_detects_changed_parenthesization(self):
        coefficient = 1.5736064017304683e50
        raw = 5.719280665634883e-243
        scale = 2.9070959581918375e-91
        correct = 3.09583749493227e-102
        wrong = 3.0958374949322706e-102
        ranker = LogisticRanker(indices=(0,))
        ranker.mean = np.asarray([0.0])
        ranker.scale = np.asarray([scale])
        ranker.coefficients = np.asarray([coefficient])
        ranker.intercept = 0.0
        features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
        features[0, 0] = raw

        scored = sequential_ranker_score(ranker, features)[0]
        self.assertEqual(scored, ranker.score_row(features[0].tolist()))
        self.assertEqual(scored, correct)
        self.assertNotEqual(scored, wrong)

    def test_recorded_argmin_uses_secondary_and_first_equal_tuple(self):
        population = _population()
        metrics = verify_recorded_argmin(population)
        self.assertEqual(metrics["recorded_argmin_match_rate"], 1.0)

        invalid = replace(population, victim=np.asarray([1, 0, 0, 1], dtype=np.int8))
        with self.assertRaises(AssertionError):
            verify_recorded_argmin(invalid)
        diagnostics = verify_recorded_argmin(invalid, raise_on_mismatch=False)
        self.assertEqual(diagnostics["recorded_argmin_mismatches"], 2.0)

    def test_nonconstant_labels_with_constant_scores_are_counted_as_metric_nan(self):
        ranker = RidgeRanker(indices=(0,))
        ranker.mean = np.asarray([0.0])
        ranker.scale = np.asarray([1.0])
        ranker.coefficients = np.asarray([0.0])
        ranker.intercept = 0.0
        population = _population(
            primary=np.zeros(4),
            secondary=np.zeros(4),
            victim=np.asarray([1, 0, 1, 0], dtype=np.int8),
        )
        metrics = score_population(ranker, population, "next_use", model_iteration=3)
        self.assertTrue(math.isnan(metrics["within_decision_macro"]))
        self.assertEqual(metrics["nonconstant_label_decisions"], 2.0)
        self.assertEqual(metrics["metric_nan_nonconstant_decisions"], 2.0)
        self.assertEqual(metrics["model_iteration"], 3)

    def test_actual_and_counterfactual_proposed_victims_remain_separate(self):
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
        features[:, 0] = [0.0, 1.0, 1.0, 0.0]
        population = _population(features=features)
        ranker = RidgeRanker(indices=(0,))
        ranker.mean = np.asarray([0.0])
        ranker.scale = np.asarray([1.0])
        ranker.coefficients = np.asarray([1.0])
        ranker.intercept = 0.0

        metrics = score_population(ranker, population, "next_use")

        self.assertEqual(metrics["actual_victim_lowest_label_rate"], 1.0)
        self.assertEqual(metrics["proposed_victim_lowest_label_rate"], 0.0)
        self.assertEqual(metrics["actual_proposed_victim_agreement_rate"], 0.0)
        self.assertEqual(metrics["actual_rejected_decisions"], 1.0)
        self.assertEqual(metrics["actual_evicted_decisions"], 1.0)

    def test_exact_primary_ties_and_full_tuple_ties_are_distinguished(self):
        ranker = RidgeRanker(indices=(0,))
        ranker.mean = np.asarray([0.0])
        ranker.scale = np.asarray([1.0])
        ranker.coefficients = np.asarray([0.0])
        ranker.intercept = 0.0
        metrics = score_population(ranker, _population(), "next_use")

        self.assertEqual(metrics["primary_exact_tie_pairs"], 2.0)
        self.assertEqual(metrics["full_tuple_exact_tie_pairs"], 1.0)
        self.assertTrue(math.isnan(metrics["minimum_positive_primary_pair_margin"]))
        self.assertEqual(metrics["minimum_top_two_primary_margin"], 0.0)


class ExistingLogAndAlignmentTests(unittest.TestCase):
    @staticmethod
    def _legacy_rows() -> Rows:
        return Rows(
            timestamp_ms=np.asarray([10.0, 10.0, 10.0, 10.0]),
            state_index=np.asarray([0, 1, 2, 3], dtype=np.int32),
            features=np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32),
            next_use_delta_ms=np.asarray([1.0, 2.0, 3.0, 4.0]),
            count_within_h=np.asarray([1.0, 1.0, 0.0, 0.0]),
            group=np.asarray([7, 7, 9, 9], dtype=np.int64),
            victim=np.asarray([1, 0, 1, 0], dtype=np.int8),
            arriving=np.asarray([0, 1, 0, 1], dtype=np.int8),
            horizon_seconds=30.0,
            arm_score=np.asarray([9.0, 7.0, 5.0, 3.0]),
            arm_tiebreak=np.asarray([90.0, 70.0, 50.0, 30.0]),
        )

    def test_existing_lru_and_lfu_logs_restore_live_learned_tiebreak(self):
        rows = self._legacy_rows()
        lru = DecisionPopulation.from_rows(rows, "test", source_policy="lru")
        lru_2hit = DecisionPopulation.from_rows(
            rows, "test", source_policy="lru_2hit"
        )
        lfu = DecisionPopulation.from_rows(rows, "test", source_policy="lfu")
        recorded = DecisionPopulation.from_rows(rows, "test")

        np.testing.assert_array_equal(lru.arm_tiebreak, rows.arm_score)
        np.testing.assert_array_equal(lru_2hit.arm_tiebreak, rows.arm_score)
        np.testing.assert_array_equal(lfu.arm_tiebreak, rows.arm_tiebreak)
        np.testing.assert_array_equal(recorded.arm_tiebreak, rows.arm_tiebreak)
        np.testing.assert_array_equal(lru.decision_ordinal, [0, 0, 1, 1])
        self.assertEqual(lru.features.dtype, np.float64)
        with self.assertRaises(ValueError):
            DecisionPopulation.from_rows(rows, "test", source_policy="random")

    def test_population_alignment_counts_unmatched_and_changed_candidate_sets(self):
        left = _population()
        unmatched = replace(
            left,
            timestamp_ms=np.asarray([10.0, 10.0, 12.0, 12.0]),
            trace_group_index=np.asarray([10, 10, 12, 12], dtype=np.int64),
        )
        unmatched_metrics = align_populations(left, unmatched)
        self.assertEqual(unmatched_metrics["shared_event_keys"], 1.0)
        self.assertEqual(unmatched_metrics["left_only_event_keys"], 1.0)
        self.assertEqual(unmatched_metrics["right_only_event_keys"], 1.0)
        self.assertEqual(unmatched_metrics["exact_ordered_candidate_sets"], 1.0)

        changed = replace(
            left,
            state_index=np.asarray([0, 1, 2, 9], dtype=np.int32),
        )
        changed_metrics = align_populations(left, changed)
        self.assertEqual(changed_metrics["shared_event_keys"], 2.0)
        self.assertEqual(changed_metrics["exact_ordered_candidate_sets"], 1.0)
        self.assertEqual(
            changed_metrics["different_candidate_sets_on_shared_keys"], 1.0
        )
        self.assertAlmostEqual(
            changed_metrics["candidate_set_jaccard_mean"], 2.0 / 3.0
        )


class CompletenessBarrierTests(unittest.TestCase):
    TRACES = ("conversation_trace", "toolagent_trace")
    CELLS = ("0.0025x1", "0.0025x4", "0.01x1", "0.01x4", "0.02x1", "0.02x4")
    TARGETS = ("next_use", "binary")
    SEEDS = (0, 1, 2, 3, 4)
    ITERATIONS = (0, 1, 2, 3)

    @classmethod
    def _full_barrier_rows(cls):
        population_rows = [
            {
                "trace": trace,
                "cell": cell,
                "target": target,
                "seed": seed,
                "iteration": iteration,
                "artifact_sha256": "1" * 64,
            }
            for trace, cell, target, seed, iteration in itertools.product(
                cls.TRACES, cls.CELLS, cls.TARGETS, cls.SEEDS, cls.ITERATIONS
            )
        ]
        model_rows = [
            {
                "trace": trace,
                "cell": "",
                "target": target,
                "seed": "",
                "iteration": 0,
                "model_sha256": "2" * 64,
            }
            for trace, target in itertools.product(cls.TRACES, cls.TARGETS)
        ]
        model_rows.extend(
            {
                "trace": trace,
                "cell": cell,
                "target": target,
                "seed": seed,
                "iteration": iteration,
                "model_sha256": "3" * 64,
            }
            for trace, cell, target, seed, iteration in itertools.product(
                cls.TRACES,
                cls.CELLS,
                cls.TARGETS,
                cls.SEEDS,
                cls.ITERATIONS[1:],
            )
        )
        return population_rows, model_rows

    def test_full_480_training_and_364_model_identity_barrier(self):
        populations, models = self._full_barrier_rows()
        validation = validate_training_model_barrier(
            populations,
            models,
            traces=self.TRACES,
            cells=self.CELLS,
            targets=self.TARGETS,
            seeds=self.SEEDS,
        )

        self.assertEqual(validation.training_identities, 480)
        self.assertEqual(validation.expected_training_identities, 480)
        self.assertEqual(validation.model_identities, 364)
        self.assertEqual(validation.expected_model_identities, 364)
        self.assertEqual(validation.pi0_models, 4)
        self.assertEqual(validation.updated_models, 360)

        with self.subTest("missing training population"):
            with self.assertRaises(ReferenceMismatchError):
                validate_training_model_barrier(
                    populations[:-1],
                    models,
                    traces=self.TRACES,
                    cells=self.CELLS,
                    targets=self.TARGETS,
                    seeds=self.SEEDS,
                )
        with self.subTest("duplicate shared pi0"):
            with self.assertRaises(ReferenceMismatchError):
                validate_training_model_barrier(
                    populations,
                    models + [dict(models[0])],
                    traces=self.TRACES,
                    cells=self.CELLS,
                    targets=self.TARGETS,
                    seeds=self.SEEDS,
                )
        with self.subTest("invalid population hash"):
            invalid = [dict(row) for row in populations]
            invalid[0]["artifact_sha256"] = "not-a-hash"
            with self.assertRaises(ReferenceMismatchError):
                validate_training_model_barrier(
                    invalid,
                    models,
                    traces=self.TRACES,
                    cells=self.CELLS,
                    targets=self.TARGETS,
                    seeds=self.SEEDS,
                )

    def test_barrier_verifies_serialized_artifact_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population_paths = []
            for iteration in self.ITERATIONS:
                artifact = root / f"population-{iteration}.npz"
                artifact.write_bytes(f"population-{iteration}".encode())
                population_paths.append(artifact)
            model_paths = []
            for iteration in self.ITERATIONS:
                artifact = root / f"model-{iteration}.json"
                artifact.write_bytes(f"model-{iteration}".encode())
                model_paths.append(artifact)

            populations = [
                {
                    "trace": "trace",
                    "cell": "cell",
                    "target": "next_use",
                    "seed": 0,
                    "iteration": iteration,
                    "population_path": str(population_paths[iteration]),
                    "population_sha256": hashlib.sha256(
                        population_paths[iteration].read_bytes()
                    ).hexdigest(),
                }
                for iteration in self.ITERATIONS
            ]
            models = [
                {
                    "trace": "trace",
                    "cell": "cell" if iteration else "",
                    "target": "next_use",
                    "seed": 0 if iteration else "",
                    "iteration": iteration,
                    "model_path": str(model_paths[iteration]),
                    "model_sha256": hashlib.sha256(
                        model_paths[iteration].read_bytes()
                    ).hexdigest(),
                }
                for iteration in self.ITERATIONS
            ]
            validation = validate_training_model_barrier(
                populations,
                models,
                traces=("trace",),
                cells=("cell",),
                targets=("next_use",),
                seeds=(0,),
                population_hash_field="population_sha256",
                population_path_field="population_path",
                model_path_field="model_path",
            )
            self.assertEqual(validation.training_identities, 4)
            self.assertEqual(validation.model_identities, 4)

            population_paths[2].write_bytes(b"changed after manifest")
            with self.assertRaises(ReferenceMismatchError):
                validate_training_model_barrier(
                    populations,
                    models,
                    traces=("trace",),
                    cells=("cell",),
                    targets=("next_use",),
                    seeds=(0,),
                    population_hash_field="population_sha256",
                    population_path_field="population_path",
                    model_path_field="model_path",
                )

    def test_each_lineage_window_requires_the_complete_4x4_matrix(self):
        groups = [
            ("trace", "cell", "next_use", 0, "train"),
            ("trace", "cell", "next_use", 0, "test"),
        ]
        rows = [
            {
                "trace": trace,
                "cell": cell,
                "target": target,
                "seed": seed,
                "window": window,
                "population_iteration": population_iteration,
                "scoring_iteration": scoring_iteration,
            }
            for (trace, cell, target, seed, window), population_iteration, scoring_iteration
            in itertools.product(groups, self.ITERATIONS, self.ITERATIONS)
        ]
        kwargs = {
            "group_fields": ("trace", "cell", "target", "seed", "window"),
            "model_iteration_field": "scoring_iteration",
            "expected_groups": groups,
            "expected_group_count": 2,
        }
        validation = validate_4x4_matrix(rows, **kwargs)
        self.assertEqual(validation.groups, 2)
        self.assertEqual(validation.rows, 32)
        self.assertEqual(validation.iterations, self.ITERATIONS)

        for name, invalid in (
            ("missing", rows[:-1]),
            ("duplicate", rows + [dict(rows[0])]),
            (
                "out of range",
                [
                    {
                        **row,
                        "scoring_iteration": 4,
                    }
                    if index == 0
                    else row
                    for index, row in enumerate(rows)
                ],
            ),
        ):
            with self.subTest(name):
                with self.assertRaises(ReferenceMismatchError):
                    validate_4x4_matrix(invalid, **kwargs)


class RankerSerializationTests(unittest.TestCase):
    @staticmethod
    def _rankers():
        logistic = LogisticRanker(indices=(0, 2), l2=0.01, standardize=True)
        logistic.mean = np.asarray([1.5, -2.0])
        logistic.scale = np.asarray([0.5, 4.0])
        logistic.coefficients = np.asarray([0.25, -0.75])
        logistic.intercept = 0.125
        logistic.converged = True
        logistic.iterations = 7
        ridge = RidgeRanker(indices=(0, 2), l2=0.01, standardize=True)
        ridge.mean = logistic.mean.copy()
        ridge.scale = logistic.scale.copy()
        ridge.coefficients = logistic.coefficients.copy()
        ridge.intercept = logistic.intercept
        ridge.condition_number = 12.5
        return logistic, ridge

    def test_serialization_is_stable_and_roundtrips_every_scoring_field(self):
        features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float64)
        features[:, 0] = [0.0, 1.0, 2.0]
        features[:, 2] = [8.0, 4.0, 2.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, ranker in enumerate(self._rankers()):
                with self.subTest(ranker=type(ranker).__name__):
                    first = root / f"ranker-{index}-a.json"
                    second = root / f"ranker-{index}-b.json"
                    first_hash = serialize_ranker(ranker, first)
                    second_hash = serialize_ranker(ranker, second)
                    self.assertEqual(first_hash, second_hash)
                    self.assertEqual(first.read_bytes(), second.read_bytes())
                    restored = deserialize_ranker(first, expected_sha256=first_hash)
                    self.assertIs(type(restored), type(ranker))
                    self.assertEqual(restored.indices, ranker.indices)
                    self.assertEqual(restored.l2, ranker.l2)
                    self.assertEqual(restored.standardize, ranker.standardize)
                    np.testing.assert_array_equal(restored.mean, ranker.mean)
                    np.testing.assert_array_equal(restored.scale, ranker.scale)
                    np.testing.assert_array_equal(restored.coefficients, ranker.coefficients)
                    self.assertEqual(restored.intercept, ranker.intercept)
                    np.testing.assert_array_equal(
                        sequential_ranker_score(restored, features),
                        sequential_ranker_score(ranker, features),
                    )
                    with self.assertRaises(ValueError):
                        deserialize_ranker(first, expected_sha256="0" * 64)


class WorkerLineageTests(unittest.TestCase):
    @staticmethod
    def _ranker(intercept: float) -> RidgeRanker:
        ranker = RidgeRanker(indices=(0,), l2=0.01, standardize=True)
        ranker.mean = np.asarray([0.0])
        ranker.scale = np.asarray([1.0])
        ranker.coefficients = np.asarray([1.0])
        ranker.intercept = intercept
        ranker.condition_number = 1.0
        return ranker

    def test_training_worker_refits_only_latest_population_with_fixed_fit_seed(self):
        trace = _branching_trace(8)
        seen_replays = []
        seen_fits = []

        class FakeLogger:
            made = 0

            def __init__(self, *args, **kwargs):
                self.iteration = FakeLogger.made
                FakeLogger.made += 1
                self.seed = kwargs["seed"]
                self.window = args[4]

            def rows(self):
                features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
                features[:, 0] = self.iteration
                return _population(features=features)

        def fake_replay(trace_arg, ranker, l1, l2, seed, split, groups, logger, **kwargs):
            self.assertIs(trace_arg, trace)
            self.assertEqual(seed, 3)
            self.assertEqual(kwargs["stop_before_ms"], split)
            self.assertEqual(logger.window, "train")
            self.assertEqual(logger.seed, seed)
            seen_replays.append((ranker.intercept, logger, logger.iteration))
            return SimpleNamespace(l2_decisions=2)

        def fake_fit(rows, target, *, seed):
            self.assertEqual(target, "next_use")
            self.assertEqual(seed, 0)
            self.assertEqual(rows.features.dtype, np.float32)
            marker = float(rows.features[0, 0])
            self.assertTrue(np.all(rows.features[:, 0] == marker))
            seen_fits.append(marker)
            return self._ranker(marker + 1), 2, len(rows.features)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pi0_path = root / "pi0.json"
            pi0_hash = serialize_ranker(self._ranker(0.0), pi0_path)
            shared = {
                "traces": {trace.name: trace},
                "splits": {trace.name: 5_000.0},
                "horizons": {trace.name: 1.0},
                "working_sets": {trace.name: 1_000},
                "groups": {trace.name: ()},
                "state_indices": {trace.name: {state: i for i, state in enumerate(trace.states)}},
                "output_dir": str(root),
                "pi0_models": {(trace.name, "next_use"): {"path": str(pi0_path), "sha256": pi0_hash}},
            }
            with mock.patch.object(runner, "_SHARED", shared), mock.patch.object(
                runner, "DecisionPopulationLogger", FakeLogger
            ), mock.patch.object(runner, "_learned_replay", fake_replay), mock.patch.object(
                runner, "fit_population_ranker", fake_fit
            ):
                output = runner._training_worker((trace.name, 0.01, 4.0, "next_use", 3))

            self.assertEqual([item[0] for item in seen_replays], [0.0, 1.0, 2.0, 3.0])
            self.assertEqual([item[2] for item in seen_replays], [0, 1, 2, 3])
            self.assertEqual(len({id(item[1]) for item in seen_replays}), 4)
            self.assertEqual(seen_fits, [0.0, 1.0, 2.0])
            self.assertEqual(len(output["populations"]), 4)
            self.assertEqual(len(output["models"]), 4)
            self.assertEqual(
                [row["fitted_from_population_iteration"] for row in output["models"]],
                ["", 0, 1, 2],
            )
            self.assertEqual(
                [row.get("fit_seed") for row in output["populations"]],
                [0, 0, 0, None],
            )

    def test_real_workers_join_four_training_and_heldout_populations(self):
        """Exercise the live replay, fitting, serialization and score handoffs."""
        trace = _branching_trace(50)
        split_ms = 30_000.0
        task = (trace.name, 0.01, 4.0, "next_use", 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pi0_path = root / "pi0.json"
            pi0_hash = serialize_ranker(self._ranker(0.0), pi0_path)
            shared = {
                "traces": {trace.name: trace},
                "splits": {trace.name: split_ms},
                "horizons": {trace.name: 1.0},
                # Two packed blocks in L1 and eight in L2 create live choices.
                "working_sets": {trace.name: 409_600},
                "groups": {trace.name: None},
                "state_indices": {
                    trace.name: {state: index for index, state in enumerate(trace.states)}
                },
                "output_dir": str(root),
                "pi0_models": {
                    (trace.name, "next_use"): {"path": str(pi0_path), "sha256": pi0_hash}
                },
            }
            with mock.patch.object(runner, "_SHARED", shared):
                train = runner._training_worker(task)
                self.assertEqual(len(train["models"]), 4)
                self.assertEqual(len(train["populations"]), 4)
                self.assertTrue(all(row["decisions_kept"] > 0 for row in train["populations"]))
                self.assertTrue(all(row["recorded_argmin_mismatches"] == 0
                                    for row in train["populations"]))
                for row in train["populations"]:
                    population = runner._load_population(row)
                    self.assertTrue(np.all(population.timestamp_ms + 1_000.0 <= split_ms))
                    self.assertEqual(population.features.dtype, np.float64)
                shared["model_index"] = runner.model_index(train["models"])
                heldout = runner._heldout_worker(task)
                self.assertEqual(len(heldout["populations"]), 4)
                self.assertEqual(len(heldout["utilities"]), 4)
                self.assertEqual(len(heldout["losses"]), 4)
                for row in heldout["populations"]:
                    population = runner._load_population(row)
                    self.assertTrue(np.all(population.timestamp_ms >= split_ms))
                    self.assertTrue(np.all(population.timestamp_ms + 1_000.0 <= trace.end_ms))
                    self.assertEqual(row["recorded_argmin_mismatches"], 0)
                for loss in heldout["losses"]:
                    validate_attribution_partition(loss)
                shared["population_index"] = runner.population_index(
                    train["populations"] + heldout["populations"]
                )
                scored = runner._crossscore_worker(task)
                integrity = runner.validate_crossscore_integrity(
                    scored["matrix"], scored["recorded"], [task]
                )
                self.assertEqual((integrity["matrix_rows"], integrity["recorded_rows"]),
                                 (32, 8))
                self.assertTrue(integrity["numerical_agreement"])

    def test_canonical_models_reject_broken_links_or_reused_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_path = root / "pi0.json"
            shared_hash = serialize_ranker(self._ranker(0.0), shared_path)
            pi0 = {("trace", "next_use"): {"path": str(shared_path), "sha256": shared_hash}}
            rows = []
            for fraction, seed, iteration in itertools.product((0.01, 0.02), (0, 1), range(4)):
                if iteration == 0:
                    path, digest = shared_path, shared_hash
                else:
                    path = root / f"{fraction}-{seed}-{iteration}.json"
                    digest = serialize_ranker(self._ranker(float(iteration)), path)
                rows.append({
                    **runner._identity("trace", fraction, 4.0, "next_use", seed, iteration),
                    "model_path": str(path),
                    "model_sha256": digest,
                    "shared_pi0": iteration == 0,
                    "fitted_from_population_iteration": "" if iteration == 0 else iteration - 1,
                })
            kwargs = dict(
                traces=("trace",), cells=((0.01, 4.0), (0.02, 4.0)),
                targets=("next_use",), seeds=(0, 1), pi0_models=pi0,
            )
            canonical = runner.canonical_models_for_barrier(rows, **kwargs)
            self.assertEqual(len(canonical), 13)
            self.assertEqual(sum(row["iteration"] == 0 for row in canonical), 1)
            for change in (
                {"fitted_from_population_iteration": 2},
                {"model_path": rows[1]["model_path"], "model_sha256": rows[1]["model_sha256"]},
                {"shared_pi0": True},
            ):
                broken = [dict(row) for row in rows]
                broken[5].update(change)  # pi1 of the second seed
                with self.subTest(change=change), self.assertRaises(AssertionError):
                    runner.canonical_models_for_barrier(broken, **kwargs)
            broken = [dict(row) for row in rows]
            broken[0]["model_sha256"] = "0" * 64
            with self.assertRaises(AssertionError):
                runner.canonical_models_for_barrier(broken, **kwargs)


class PublicationBoundaryTests(unittest.TestCase):
    def test_published_reference_csv_schema_matches_registered_grid_read_only(self):
        import json

        phase1 = json.loads(runner.PHASE1_CONFIG.read_text(encoding="utf-8"))
        names = ("conversation_trace", "toolagent_trace")
        trace_files = {name: {"sha256": phase1["trace_files"][name]["sha256"]}
                       for name in names}
        rows, hashes, _ = runner._reference_context(
            names, runner.CELLS, runner.SEEDS, trace_files,
            phase1["measure_from_ms"], phase1["horizons_seconds"],
            phase1["working_set_bytes"],
        )
        self.assertEqual(len(rows), 1_500)
        self.assertEqual(len(hashes), 5)
        generic = [row for row in rows if row["kind"] in ("heap", "sampled")]
        learned = [row for row in rows if row["arm"] == "A_none"]
        self.assertTrue(generic)
        self.assertTrue(all(row["target"] == "" for row in generic))
        self.assertTrue(set(runner.TARGETS) <= {row["target"] for row in learned})
        self.assertEqual({int(row["seed"]) for row in generic}, set(runner.SEEDS))

    def test_runner_completes_barrier_before_heldout_and_gates_publication(self):
        trace = _request_trace([("root", "a")])
        trace.name = "conversation_trace"
        base = runner._identity(trace.name, *runner.SMOKE_CELL, "next_use", 0)
        models = [{**base, "iteration": iteration} for iteration in range(4)]
        train = [{**base, "window": "train", "iteration": iteration} for iteration in range(4)]
        test = [{**base, "window": "test", "iteration": iteration} for iteration in range(4)]
        utilities = [{**base, "iteration": iteration} for iteration in range(4)]
        terminal = [{**base, "terminal_common_population_ranking_delta": 0.1,
                     "fit_population_ranking_delta": 0.1,
                     "own_policy_test_ranking_delta": 0.1}]
        barrier = BarrierValidation(4, 4, 4, 4, 1, 3)

        def exercise(*, fail_barrier=False, change_source=False):
            events = []
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace_path = root / "toy.jsonl"
                trace_path.write_text("toy", encoding="utf-8")
                candidate_path = root / "candidate.npz"
                candidate_path.write_bytes(b"candidate")
                metadata_path = root / "candidate.csv"
                metadata_path.write_text("toy", encoding="utf-8")
                model_path = root / "model.json"
                model_path.write_text("{}", encoding="utf-8")
                population_path = root / "population.npz"
                population_path.write_bytes(b"toy population")
                model_rows = [{**row, "model_path": str(model_path)} for row in models]
                train_rows = [{**row, "population_path": str(population_path)} for row in train]
                test_rows = [{**row, "population_path": str(population_path)} for row in test]
                candidate = [{"trace": trace.name, "l1_fraction": runner.SMOKE_CELL[0],
                              "l2_multiplier": runner.SMOKE_CELL[1],
                              "behaviour_policy": policy,
                              "artifact_path": str(candidate_path),
                              "current_sha256": hashlib.sha256(b"candidate").hexdigest()}
                             for policy in ("lru", "lfu")]
                args = SimpleNamespace(traces=[trace_path], output_dir=root / "out",
                                       paper_dir=root / "paper", workers=1, smoke=True)

                def stage(worker, tasks, workers):
                    self.assertEqual(len(tasks), 1)
                    self.assertEqual(workers, 1)
                    events.append(worker.__name__)
                    if worker is runner._training_worker:
                        yield {"models": model_rows, "populations": train_rows,
                               "peak_worker_rss_mib": 7.0,
                               "training_replay_seconds": 1.0,
                               "fit_seconds": 1.0,
                               "serialization_seconds": 1.0}
                    elif worker is runner._heldout_worker:
                        yield {"populations": test_rows, "utilities": utilities,
                               "losses": utilities, "orphaning": utilities,
                               "heldout_replay_seconds": 1.0,
                               "peak_worker_rss_mib": 8.0}
                    elif worker is runner._crossscore_worker:
                        yield {"matrix": [], "recorded": [],
                               "coefficient_similarity": [{} for _ in range(16)],
                               "crossscore_seconds": 1.0,
                               "peak_worker_rss_mib": 9.0}
                    elif worker is runner._alignment_with_rss:
                        yield {"rows": [{} for _ in range(12)],
                               "peak_worker_rss_mib": 10.0}
                    elif worker is runner._auxiliary_with_rss:
                        yield {"rows": [{} for _ in range(16)],
                               "peak_worker_rss_mib": 11.0}
                    else:
                        self.fail(f"unexpected worker {worker}")

                def check_barrier(*args, **kwargs):
                    events.append("barrier")
                    if fail_barrier:
                        raise AssertionError("barrier failed")
                    return barrier

                def publish(*args, **kwargs):
                    events.append("publish")
                    return root / "paper"

                source = {"files": {"src/x.py": "1" * 64}, "sha256": "1" * 64}
                source_end = ({"files": {"src/x.py": "2" * 64}, "sha256": "2" * 64}
                              if change_source else source)
                patches = (
                    mock.patch.object(runner, "ensure_execution_tree_committed"),
                    mock.patch.object(runner, "_git", return_value="commit"),
                    mock.patch.object(runner, "source_manifest", side_effect=[source, source_end]),
                    mock.patch.object(runner, "dirty_doc_hashes", return_value={}),
                    mock.patch.object(runner, "load_mooncake_trace", return_value=trace),
                    mock.patch.object(runner, "horizon_for", return_value=(600.0, 0.0, None)),
                    mock.patch.object(runner, "_occurrence_groups", return_value=()),
                    mock.patch.object(runner, "working_set_bytes", return_value=1_000),
                    mock.patch.object(runner, "state_indices", return_value={"root": 0, "a": 1}),
                    mock.patch.object(runner, "verify_phase097_config"),
                    mock.patch.object(runner, "_reference_context", return_value=([], {}, {})),
                    mock.patch.object(runner, "_verify_candidate_references", return_value=candidate),
                    mock.patch.object(runner, "CANDIDATE_META", metadata_path),
                    mock.patch.object(runner, "rebuild_pi0", return_value=(
                        {(trace.name, "next_use"): {"path": "pi0", "sha256": "0" * 64}}, [])),
                    mock.patch.object(runner, "_run_stage", side_effect=stage),
                    mock.patch.object(runner, "canonical_models_for_barrier", return_value=model_rows),
                    mock.patch.object(runner, "validate_training_model_barrier", side_effect=check_barrier),
                    mock.patch.object(runner, "write_csv_rows"),
                    mock.patch.object(runner, "verify_published_pi0_replay",
                                      return_value=SimpleNamespace(row_count=1)),
                    mock.patch.object(runner, "_pi0_loss_reference_check", return_value={
                        "matched_rows": 1, "reference_unavailable_cells": []}),
                    mock.patch.object(runner, "validate_crossscore_integrity", return_value={
                        "numerical_agreement": True}),
                    mock.patch.object(runner, "validate_4x4_matrix"),
                    mock.patch.object(runner, "select_generic_winners", return_value={}),
                    mock.patch.object(runner, "derive_seed_utilities", return_value=utilities),
                    mock.patch.object(runner, "merge_seed_attribution", return_value=[]),
                    mock.patch.object(runner, "aggregate_seed_metrics", return_value=[]),
                    mock.patch.object(runner, "_ranking_and_verdicts", return_value=(
                        [{} for _ in range(8)], terminal, [{}])),
                    mock.patch.object(runner, "_publish_outputs", side_effect=publish),
                    mock.patch.object(runner, "_SHARED", {}),
                )
                with ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    if fail_barrier or change_source:
                        with self.assertRaises(AssertionError):
                            runner.run_experiment(args)
                    else:
                        config = runner.run_experiment(args)
                        self.assertEqual(config["training_prefix_replays"], 4)
                        self.assertEqual(config["heldout_replays"], 4)
            return events

        success = exercise()
        self.assertLess(success.index("_training_worker"), success.index("barrier"))
        self.assertLess(success.index("barrier"), success.index("_heldout_worker"))
        self.assertEqual(success[-1], "publish")
        self.assertEqual(exercise(fail_barrier=True), ["_training_worker", "barrier"])
        self.assertNotIn("publish", exercise(change_source=True))

    def test_reference_context_accepts_targetless_generic_rows_but_requires_both_learned_targets(self):
        name = "conversation_trace"
        fraction, multiplier, working_set = 0.01, 4.0, 1_000
        base = {
            "trace": name, "l1_fraction": str(fraction),
            "l2_multiplier": str(multiplier), "seed": "0",
            "l1_capacity_bytes": "10", "l2_capacity_bytes": "40",
            "l1_policy": runner.L1_POLICY, "hit_model": runner.HIT_MODEL,
            "closure": runner.CLOSURE, "l2_sample_width": str(runner.SAMPLE_WIDTH),
        }
        references = [
            {**base, "arm": arm, "kind": kind, "target": target,
             "l2_seed": "0", "l2_decisions": "3"}
            for arm, kind, target in (
                ("A_none", "learned", "next_use"),
                ("A_none", "learned", "binary"),
                ("lru_s", "sampled", ""),
                ("lfu_s", "sampled", ""),
                ("offline_next_use", "heap", ""),
            )
        ]
        phase1 = {
            "trace_files": {name: {"sha256": "a" * 64}},
            "horizons_seconds": {name: 600.0},
            "measure_from_ms": {name: 6_000.0},
            "working_set_bytes": {name: working_set},
            "l1_policy": runner.L1_POLICY,
            "hit_model": runner.HIT_MODEL,
            "closure": runner.CLOSURE,
            "sample_width": runner.SAMPLE_WIDTH,
            "size_model": runner.HYPERPARAMETERS["size_model"],
            "bytes_per_token": runner.HYPERPARAMETERS["bytes_per_token"],
        }
        with tempfile.TemporaryDirectory(dir=runner.REPOSITORY) as directory:
            config_path = Path(directory) / "phase1.json"
            import json
            config_path.write_text(json.dumps(phase1), encoding="utf-8")
            with mock.patch.object(runner, "PHASE1_CONFIG", config_path), mock.patch.object(
                runner, "_published_sha256", return_value="b" * 64
            ), mock.patch.object(runner, "load_csv", return_value=references):
                selected, _, _ = runner._reference_context(
                    (name,), ((fraction, multiplier),), (0,),
                    {name: {"sha256": "a" * 64}}, {name: 6_000.0},
                    {name: 600.0}, {name: working_set},
                )
                self.assertEqual(len(selected), 5)
                missing_binary = [row for row in references
                                  if not (row["arm"] == "A_none" and row["target"] == "binary")]
                with mock.patch.object(runner, "load_csv", return_value=missing_binary):
                    with self.assertRaises(AssertionError):
                        runner._reference_context(
                            (name,), ((fraction, multiplier),), (0,),
                            {name: {"sha256": "a" * 64}}, {name: 6_000.0},
                            {name: 600.0}, {name: working_set},
                        )

    def test_candidate_decision_counts_use_sampled_generic_reference_seed(self):
        name = "conversation_trace"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual, metadata, references = [], [], []
            for policy, sampled_arm in (("lru", "lru_s"), ("lfu", "lfu_s")):
                path = root / f"{policy}.npz"
                path.write_bytes(policy.encode())
                identity = {"trace": name, "l1_fraction": 0.01,
                            "l2_multiplier": 4.0, "behaviour_policy": policy}
                actual.append({**identity, "decisions_kept": 2, "rows": 4,
                               "artifact_path": str(path),
                               "current_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                metadata.append({**identity, "decisions_kept": "2", "rows": "4",
                                 "decisions_seen": "5", "l2_decisions": "5",
                                 "l2_admissions": "3", "l2_rejections": "1",
                                 "l2_evictions": "1"})
                references.append({"trace": name, "l1_fraction": 0.01,
                                   "l2_multiplier": 4.0, "arm": sampled_arm,
                                   "kind": "sampled", "target": "", "seed": "0",
                                   "l2_decisions": "5", "l2_admissions": "3",
                                   "l2_rejections": "1", "l2_evictions": "1"})
                # Heap rows have zero sampled decisions and must not be used.
                references.append({"trace": name, "l1_fraction": 0.01,
                                   "l2_multiplier": 4.0, "arm": policy,
                                   "kind": "heap", "target": "", "seed": "0",
                                   "l2_decisions": "0"})
            with mock.patch.object(runner, "_actual_candidate_metadata", return_value=actual), mock.patch.object(
                runner, "load_csv", return_value=metadata
            ):
                checked = runner._verify_candidate_references(
                    (name,), ((0.01, 4.0),), references)
                self.assertEqual(len(checked), 2)
                self.assertTrue(all(row["l2_decisions"] == "5" for row in checked))
                invalid = [dict(row) for row in references]
                invalid[0]["seed"] = "1"
                with self.assertRaises(AssertionError):
                    runner._verify_candidate_references(
                        (name,), ((0.01, 4.0),), invalid)

    def test_phase098b_pi0_reference_reports_only_covered_cells(self):
        name = "conversation_trace"
        common = {"trace": name, "target": "next_use", "seed": 0,
                  "measured_requests": 1, "requested_tokens": 2,
                  "l1_avoided_tokens": 1, "l2_avoided_tokens": 0,
                  "avoided_prefill_tokens": 1, "root_loss_tokens": 1,
                  "unusable_tokens": 0, "decision_loss_tokens": 1,
                  "absent_loss_tokens": 1, "perblock_decision_loss_tokens": 1,
                  "downstream_absent_tokens": 0}
        covered = {**common, "l1_fraction": 0.01, "l2_multiplier": 4.0,
                   "cell": runner.cell_label(0.01, 4.0)}
        uncovered = {**common, "l1_fraction": 0.02, "l2_multiplier": 4.0,
                     "cell": runner.cell_label(0.02, 4.0)}
        reference = {**covered, "arm": "A_none"}
        with mock.patch.object(runner, "load_csv", return_value=[reference]):
            result = runner._pi0_loss_reference_check(
                [{**covered, "iteration": 0}, {**uncovered, "iteration": 0}],
                (name,), ((0.01, 4.0), (0.02, 4.0)), ("next_use",), (0,),
            )
            self.assertEqual(result["matched_rows"], 1)
            self.assertEqual(result["reference_unavailable_cells"],
                             [runner.cell_label(0.02, 4.0)])
            changed = {**covered, "iteration": 0, "perblock_decision_loss_tokens": 2}
            with self.assertRaises(ReferenceMismatchError):
                runner._pi0_loss_reference_check(
                    [changed, {**uncovered, "iteration": 0}],
                    (name,), ((0.01, 4.0), (0.02, 4.0)), ("next_use",), (0,),
                )

    def test_pi0_reference_requires_exact_integer_replay_and_all_identities(self):
        common = {field: "0" for field in PI0_REPLAY_EXACT_FIELDS}
        common.update({field: "fixed" for field in PI0_REPLAY_TEXT_FIELDS})
        published = [
            {**common, "trace": "trace", "l1_fraction": "0.01",
             "l2_multiplier": "4", "target": "next_use", "seed": str(seed),
             "arm": "A_none"}
            for seed in (0, 1)
        ]
        replayed = [{**row, "iteration": 0, "arm": "pi0"} for row in published]
        self.assertEqual(verify_published_pi0_replay(
            replayed, published, expected_count=2).row_count, 2)
        with self.assertRaises(ReferenceMismatchError):
            verify_published_pi0_replay(replayed[:1], published, expected_count=2)
        changed = [dict(row) for row in replayed]
        changed[0]["l2_evictions"] = "1"
        with self.assertRaises(ReferenceMismatchError):
            verify_published_pi0_replay(changed, published, expected_count=2)

    def test_crossscore_integrity_requires_every_lineage_diagonal_and_recorded_row(self):
        task = ("trace", 0.01, 4.0, "next_use", 0)
        base = dict(trace="trace", l1_fraction=0.01, l2_multiplier=4.0,
                    target="next_use", seed=0)
        matrix = [
            {**base, "window": window, "population_iteration": i,
             "scoring_iteration": j, "recorded_argmin_mismatches": 0,
             "recomputed_recorded_primary_exact": 1,
             "actual_proposed_victim_agreement_rate": 1.0}
            for window in ("train", "test") for i in range(4) for j in range(4)
        ]
        recorded = [
            {**base, "window": window, "population_iteration": i}
            for window in ("train", "test") for i in range(4)
        ]
        result = runner.validate_crossscore_integrity(matrix, recorded, [task])
        self.assertEqual((result["matrix_rows"], result["recorded_rows"]), (32, 8))
        self.assertTrue(result["numerical_agreement"])
        with self.assertRaises(AssertionError):
            runner.validate_crossscore_integrity(matrix, recorded[:-1], [task])
        with self.assertRaises(AssertionError):
            runner.validate_crossscore_integrity(matrix[:-1], recorded, [task])
        broken = [dict(row) for row in matrix]
        broken[0]["recomputed_recorded_primary_exact"] = 0
        result = runner.validate_crossscore_integrity(broken, recorded, [task])
        self.assertFalse(result["numerical_agreement"])

    def test_complete_matrix_rejects_an_entire_missing_group(self):
        groups = (("trace", "cell", "next_use", 0, "train"),
                  ("trace", "cell", "next_use", 0, "test"))
        rows = [
            dict(zip(("trace", "cell", "target", "seed", "window"), group))
            | {"population_iteration": i, "model_iteration": j}
            for group in groups for i in range(4) for j in range(4)
        ]
        kwargs = dict(group_fields=("trace", "cell", "target", "seed", "window"),
                      expected_groups=groups, expected_group_count=2)
        self.assertEqual(validate_4x4_matrix(rows, **kwargs).rows, 32)
        with self.assertRaises(ReferenceMismatchError):
            validate_4x4_matrix(rows[:16], **kwargs)

    def test_contemporaneous_candidate_hash_cannot_satisfy_historical_hash_gate(self):
        metadata = dict(trace="trace", l1_fraction="0.01", l2_multiplier="4",
                        behaviour_policy="lru", decisions_kept="2", rows="4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.npz"
            path.write_bytes(b"toy candidate log")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            kwargs = dict(count_fields=("decisions_kept", "rows"), artifact_path=path)
            checked = verify_candidate_log_metadata([metadata], [metadata], **kwargs)
            self.assertFalse(checked.historical_hash_verified)
            self.assertEqual(checked.hash_status, "contemporaneous_only")
            with self.assertRaises(ReferenceMismatchError):
                verify_candidate_log_metadata([metadata], [metadata],
                                              require_historical_hash=True, **kwargs)
            with self.assertRaises(ReferenceMismatchError):
                verify_candidate_log_metadata([metadata], [metadata],
                                              published_historical_sha256="0" * 64, **kwargs)
            verified = verify_candidate_log_metadata([metadata], [metadata],
                                                     published_historical_sha256=digest,
                                                     require_historical_hash=True, **kwargs)
            self.assertTrue(verified.historical_hash_verified)
            with self.assertRaises(ReferenceMismatchError):
                verify_candidate_log_metadata([metadata], [{**metadata, "rows": "5"}], **kwargs)

    def test_nonfinite_registered_metric_stays_unresolved(self):
        finite = dict(
            target="next_use", utility_point_deltas=[0.2] * 5,
            heldout_ranking_deltas=[0.1] * 5,
            fit_population_ranking_deltas=[0.1] * 5,
            pi3_fit_population_scores=[0.4] * 5,
            pi3_terminal_train_scores=[0.4] * 5,
            pi3_test_scores=[0.4] * 5,
        )
        self.assertEqual(classify_registered_outcome(**finite).label, "A")
        invalid = {**finite, "heldout_ranking_deltas": [0.1, 0.1, math.nan, 0.1, 0.1]}
        outcome = classify_registered_outcome(**invalid)
        self.assertEqual(outcome.label, "unresolved")
        self.assertFalse(outcome.heldout_ranking_pass)
        self.assertTrue(any("non-finite" in reason for reason in outcome.reasons))

    def test_attribution_enforces_token_and_block_partition_identities(self):
        from persistent_kv_admission.onpolicy_reporting import ATTRIBUTION_CATEGORIES

        row = {}
        for category in ATTRIBUTION_CATEGORIES:
            for unit in ("tokens", "blocks"):
                row[f"absent_{category}_{unit}"] = 0
                row[f"root_{category}_{unit}"] = 0
                row[f"downstream_{category}_{unit}"] = 0
            row[f"unusable_after_{category}_tokens"] = 0
            row[f"unusable_after_{category}_blocks"] = 0
        row.update(absent_rejected_tokens=3, root_rejected_tokens=2,
                   downstream_rejected_tokens=1, absent_rejected_blocks=1,
                   root_rejected_blocks=1, unusable_after_evicted_tokens=1,
                   unusable_after_evicted_blocks=1, root_loss_tokens=2,
                   unusable_tokens=1, absent_loss_tokens=3,
                   decision_loss_tokens=3, perblock_decision_loss_tokens=4,
                   downstream_absent_tokens=1, requested_tokens=10,
                   l1_avoided_tokens=1, beyond_prefix_tokens=9,
                   l2_hit_tokens=5, l2_avoided_tokens=5,
                   l2_present_unusable_tokens=1, beyond_prefix_blocks=5,
                   l2_hit_blocks_checked=3)
        validate_attribution_partition(row)
        for field, value in (("perblock_decision_loss_tokens", 3),
                             ("beyond_prefix_tokens", 8),
                             ("beyond_prefix_blocks", 4),
                             ("root_rejected_tokens", 1)):
            with self.subTest(field=field), self.assertRaises(ReferenceMismatchError):
                validate_attribution_partition({**row, field: value})

    def test_label_window_and_composite_request_hook_use_same_request(self):
        trace = _request_trace([("root", "a")])
        collector = LabelWindowUtilityCollector(trace, 1_000.0, 2_000.0)
        attribution = SimpleNamespace(on_request=mock.Mock())
        hook = runner._request_hook(attribution, collector)
        requests = [
            (("root", "a"), 1, {1}, {1}, 999.0, 0, True),
            (("root", "a"), 1, {1}, {1}, 1_000.0, 1, True),
            (("root", "a"), 1, set(), {1}, 2_000.0, 2, True),
            (("root", "a"), 1, {1}, {1}, 2_001.0, 3, True),
        ]
        for request in requests:
            hook(*request)
        self.assertEqual(attribution.on_request.call_count, 4)
        self.assertEqual(collector.row(), {
            "label_window_measured_requests": 2,
            "label_window_requested_tokens": 4,
            "label_window_requested_blocks": 4,
            "label_window_l1_avoided_tokens": 2,
            "label_window_l2_avoided_tokens": 1,
            "label_window_avoided_prefill_tokens": 3,
            "label_window_l2_hit_blocks": 1,
            "label_window_present_unusable_tokens": 1,
            "label_window_present_unusable_blocks": 1,
        })


if __name__ == "__main__":
    unittest.main()
