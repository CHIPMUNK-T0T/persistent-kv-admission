"""Tests for the matched class-order control (docs/matched-class-order-plan.md).

No real trace is replayed: the plan forbids a real replay before the
implementation is reviewed and committed. Everything here runs on constructed
traces, hand-built decisions and synthetic rows; the published result tables
are only read. What has to hold before a real replay means anything: the grid
and the matched horizons are the plan's; at h = 600 s each new arm is the
published arm decision by decision, counter digest included; at another h the
store's key orders the candidates as (reusable within h, score, last_group),
which differs from the 600-second class on a constructed decision; with
admission by the exact h-second reuse label the recency arm is the
`label_binary_h` rung decision by decision; the class statistic is the
error-location hook's own rule constructed at h and is read-only; the reading
helpers do what the plan fixes at their boundaries; and the runner's checks,
refusals, derivations (also against the published horizon-control tables), its
end-to-end run with git mocked and the read-only tabulation behave as stated,
including that a reproduction difference publishes nothing.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from persistent_kv_admission import errorloc, horizonctl, horizonfill, matchedorder
from persistent_kv_admission.decisionpop import RawFeatureScorer, horizon_for
from persistent_kv_admission.errorloc import (
    DecisionStatistics,
    HybridOverride,
    HybridScorer,
    RecordingOverride,
    first_minimum,
)
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import ReuseClassScorer
from persistent_kv_admission.matchedorder import (
    ARMS,
    MATCHED_HORIZON_SECONDS,
    MATCHED_READINGS,
    PUBLISHED_ARMS,
    ClassStatistics,
    cell_arms,
    matched_arm,
)
from persistent_kv_admission.mechanism import ExactLabelScorer
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The horizon control's fixtures (constructed traces at one step per 5 s, the
# traced replay helper, the rule checker of its class-order arms), reused as
# they are; only plain functions and classes are read from them.
horizon_tests = _load("_matched_horizon_control_fixtures",
                      REPOSITORY / "tests/test_horizon_control.py")
runner = _load("_run_matched_class_order_under_test",
               REPOSITORY / "scripts/run_matched_class_order.py")
tabulate = _load("_tabulate_matched_class_order_under_test",
                 REPOSITORY / "scripts/tabulate_matched_class_order.py")

HORIZON = horizon_tests.HORIZON
MEASURE_FROM_MS = horizon_tests.MEASURE_FROM_MS
L1_BYTES = horizon_tests.L1_BYTES
L2_BYTES = horizon_tests.L2_BYTES
partial_trace = horizon_tests.partial_trace
replay = horizon_tests.replay
_row = horizon_tests._row
_binary = horizon_tests._binary
_Fixed = horizon_tests._Fixed
_RuleCheck = horizon_tests._RuleCheck
_Tracer = horizon_tests._Tracer
_LinearRanker = horizon_tests._LinearRanker
csv_rows = horizon_tests.csv_rows
SEEDS = (0, 1, 2, 3, 4)
# Columns that differ between two replays of the same arm on the same input,
# and those that name the arm.
VOLATILE = ("seconds", "worker_peak_rss_mib", "worker_pss_mib_end")
NAMING = ("arm", "family", "arm_parameter")
NEW_CELLS = ((0.0025, 1.0), (0.0025, 4.0), (0.01, 1.0), (0.02, 1.0))
REPRODUCED_CELLS = ((0.01, 4.0), (0.02, 4.0))


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _setup(arm, trace):
    return matchedorder.arm_setup(arm, trace, learned_ranker=_LinearRanker())


def _published_setup(published_arm, trace, seed=1):
    return horizonctl.arm_setup(published_arm, trace, HORIZON, seed,
                                learned_ranker=_LinearRanker())


def _replay(trace, arm, setup, seed=1):
    """One traced replay under all16 with the error-location statistics."""
    return replay(trace, arm, seed=seed, setup=setup, eligibility="all")


def class_replay(trace, setup, h, seed=1, wrapped=True):
    """A traced all16 replay with the error-location statistics *and* the class
    statistics constructed at `h`, nested as the runner nests them (or neither,
    `wrapped=False`). Returns (result, statistics, class statistics, tracer)."""
    statistics = DecisionStatistics(trace, h, MEASURE_FROM_MS) if wrapped else None
    classes = ClassStatistics(trace, h, MEASURE_FROM_MS) if wrapped else None
    override = (RecordingOverride(statistics, RecordingOverride(classes, setup.override))
                if wrapped else setup.override)
    tracer = _Tracer(override)
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                          l2_sample_width=4, l2_seed=seed, l2_eligibility="all",
                          l2_override_hook=tracer, **setup.replay_arguments())
    return result, statistics, classes, tracer


# --- the grid --------------------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_matched_horizons_and_arms_are_the_plans(self):
        self.assertEqual(MATCHED_HORIZON_SECONDS, {
            (0.0025, 1.0): 60.0, (0.0025, 4.0): 150.0, (0.01, 1.0): 150.0,
            (0.01, 4.0): 600.0, (0.02, 1.0): 300.0, (0.02, 4.0): 600.0})
        self.assertEqual(set(MATCHED_HORIZON_SECONDS), set(runner.CELLS))
        self.assertEqual(matchedorder.MATCHED_HORIZONS_SECONDS, (60.0, 150.0, 300.0, 600.0))
        self.assertEqual(ARMS, ("evict_binary_60_learned", "evict_binary_60_recency",
                                "evict_binary_150_learned", "evict_binary_150_recency",
                                "evict_binary_300_learned", "evict_binary_300_recency",
                                "evict_binary_600_learned", "evict_binary_600_recency"))
        self.assertEqual(PUBLISHED_ARMS, {"evict_binary_600_learned": "evict_binary_learned",
                                          "evict_binary_600_recency": "evict_binary_recency"})
        for published, order in horizonctl.CLASS_ORDER.items():
            self.assertEqual(matchedorder.ORDER_OF_ARM[matched_arm(600, order)], order)
            self.assertEqual(PUBLISHED_ARMS[matched_arm(600, order)], published)
        self.assertEqual(cell_arms(0.0025, 1.0), ("evict_binary_60_learned",
                                                  "evict_binary_60_recency"))
        self.assertEqual(cell_arms(0.02, 1.0), ("evict_binary_300_learned",
                                                "evict_binary_300_recency"))
        self.assertEqual(matchedorder.RUNG_OF_HORIZON,
                         {60.0: "label_binary_60", 150.0: "label_binary_150",
                          300.0: "label_binary_300", 600.0: "label_binary_600"})
        with self.assertRaises(ValueError):
            matchedorder.matched_horizon(0.05, 1.0)
        with self.assertRaises(ValueError):
            matched_arm(60, "frequency")

    def test_the_roles_follow_the_table_eight_new_and_four_reproduced(self):
        # The plan's text says six and six; its own table of h* gives eight
        # trace x cell below 600 s and four at 600 s. The roles follow the table.
        roles = {cell: matchedorder.role_of_horizon(h) for cell, h in MATCHED_HORIZON_SECONDS.items()}
        self.assertEqual({cell for cell, role in roles.items() if role == "new"}, set(NEW_CELLS))
        self.assertEqual({cell for cell, role in roles.items() if role == "reproduced"},
                         set(REPRODUCED_CELLS))
        self.assertEqual(len(runner.TRACES) * len(NEW_CELLS), 8)
        with self.assertRaises(ValueError):
            matchedorder.role_of_horizon(90.0)

    def test_the_tasks(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS)
        self.assertEqual(len(tasks), 120)
        self.assertEqual(len(set(tasks)), 120)
        self.assertEqual(sum(1 for task in tasks if task[3] in PUBLISHED_ARMS), 40)
        by_cell = defaultdict(set)
        for name, fraction, multiplier, arm, seed in tasks:
            by_cell[(name, fraction, multiplier)].add(arm)
        self.assertEqual(len(by_cell), 12)
        for (name, fraction, multiplier), arms in by_cell.items():
            self.assertEqual(arms, set(cell_arms(fraction, multiplier)))
        costs = [runner.arm_cost(task[3]) for task in tasks]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH, runner.MECHANISM), ("all", 16, "all16"))

    def test_the_references_are_the_plans(self):
        self.assertEqual(runner.REFERENCES, {
            ("all16", "lru"): "error_location_001", ("all16", "learned"): "error_location_001",
            ("all16", "label"): "error_location_001",
            ("all16", "evict_label"): "error_location_001",
            ("all16", "label_binary_60"): "horizon_control_001",
            ("all16", "label_binary_150"): "horizon_fill_001",
            ("all16", "label_binary_300"): "horizon_control_001",
            ("all16", "label_binary_600"): "horizon_control_001",
            ("all16", "evict_binary_learned"): "horizon_control_001",
            ("all16", "evict_binary_recency"): "horizon_control_001"})
        self.assertEqual(runner.REFERENCE_SOURCES, {
            "error_location_001": REPOSITORY / "results/paper/error_location_001/replay_seeds.csv",
            "horizon_control_001": REPOSITORY / "results/paper/horizon_control_001/replay_seeds.csv",
            "horizon_fill_001": REPOSITORY / "results/paper/horizon_fill_001/replay_seeds.csv"})
        self.assertEqual(runner.reference_cells("all16", "label_binary_150"),
                         ((0.0025, 4.0), (0.01, 1.0)))
        self.assertEqual(runner.reference_cells("all16", "label_binary_60"), ((0.0025, 1.0),))
        self.assertEqual(runner.reference_cells("all16", "label_binary_300"), ((0.02, 1.0),))
        self.assertEqual(runner.reference_cells("all16", "label_binary_600"),
                         ((0.01, 4.0), (0.02, 4.0)))
        self.assertEqual(runner.reference_cells("all16", "evict_binary_learned"), runner.CELLS)
        with self.assertRaises(ValueError):
            runner.reference_cells("leaf16", "learned")

    def test_the_tabulation_uses_the_same_vocabulary(self):
        self.assertEqual(tabulate.MATCHED_READINGS, MATCHED_READINGS)
        self.assertEqual(tabulate.ORDERS, matchedorder.ORDERS)
        self.assertEqual(tabulate.ROLES, matchedorder.ROLES)
        self.assertEqual(tabulate.SIGN_READINGS, matchedorder.SIGN_READINGS)
        self.assertEqual(tabulate.TRACES, runner.TRACES)


# --- the arms --------------------------------------------------------------------------------------


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_arms_are_part_2s_construction_with_the_reuse_label_at_h(self):
        for h in matchedorder.MATCHED_HORIZONS_SECONDS:
            with self.subTest(h=h):
                learned = _setup(matched_arm(h, "learned"), self.trace)
                self.assertIsInstance(learned.scorer, ReuseClassScorer)
                self.assertIsInstance(learned.scorer.reuse, ExactLabelScorer)
                self.assertEqual((learned.scorer.reuse.target,
                                  learned.scorer.reuse.horizon_seconds), ("binary", h))
                self.assertIsInstance(learned.scorer.within, RawFeatureScorer)
                self.assertIsInstance(learned.override, HybridOverride)
                # The ranker inside the composite is the one admission consults.
                self.assertIs(learned.override.admission, learned.scorer.within)
                self.assertEqual(learned.l2_policy, "learned")
                recency = _setup(matched_arm(h, "recency"), self.trace)
                self.assertIsInstance(recency.scorer, HybridScorer)
                self.assertIsInstance(recency.scorer.admission, RawFeatureScorer)
                self.assertEqual((recency.scorer.eviction.target,
                                  recency.scorer.eviction.horizon_seconds), ("binary", h))
                self.assertIs(recency.override.admission, recency.scorer.admission)
                identity = matchedorder.identity_setup(self.trace, h)
                self.assertIsNot(identity.scorer.admission, identity.scorer.eviction)
                for scorer in (identity.scorer.admission, identity.scorer.eviction):
                    self.assertEqual((scorer.target, scorer.horizon_seconds), ("binary", h))
        # At 600 s the reuse label is horizonctl's own.
        mine = matchedorder.reuse_label(self.trace, 600.0)
        theirs = horizonctl.reuse_label(self.trace)
        self.assertEqual((mine.target, mine.horizon_seconds),
                         (theirs.target, theirs.horizon_seconds))

    def test_unknown_arms_and_missing_inputs_are_refused(self):
        for arm in ("evict_binary_learned", "evict_binary_90_learned", "label_binary_60"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                _setup(arm, self.trace)
        with self.assertRaises(ValueError):
            matchedorder.arm_setup("evict_binary_60_learned", self.trace, learned_ranker=None)
        with self.assertRaises(ValueError):
            matchedorder.class_order_setup("x", self.trace, _Fixed({}), "frequency", 60.0)

    def test_at_600_seconds_each_arm_is_the_published_arm_decision_by_decision(self):
        for arm, published in PUBLISHED_ARMS.items():
            for seed in (1, 3):
                with self.subTest(arm=arm, seed=seed):
                    mine, mine_stats, mine_trace, _ = _replay(
                        self.trace, arm, _setup(arm, self.trace), seed=seed)
                    theirs, theirs_stats, theirs_trace, _ = _replay(
                        self.trace, published, _published_setup(published, self.trace, seed),
                        seed=seed)
                    self.assertEqual(mine_trace.decisions, theirs_trace.decisions)
                    self.assertGreater(len(mine_trace.decisions), 100)
                    self.assertGreater(sum(1 for d in mine_trace.decisions if d[5] != d[6]), 0)
                    self.assertEqual(_row(mine), _row(theirs))
                    self.assertEqual(mine_stats.row(), theirs_stats.row())

    def test_at_another_horizon_the_rule_holds_with_the_class_at_h(self):
        # The horizon control's own rule checker, given the exact reuse label
        # at h and an independent copy of the ranker: at every decision the
        # store's key is ((reusable within h, score), last_group) or
        # (reusable within h, last_group), the arrival is rejected iff it is
        # the ranker's first minimum, and otherwise the victim is the
        # composite's first minimum among the others.
        for h in (60.0, 150.0, 300.0):
            for order in matchedorder.ORDERS:
                arm = matched_arm(h, order)
                with self.subTest(arm=arm):
                    setup = _setup(arm, self.trace)
                    independent = RawFeatureScorer(self.trace, _LinearRanker())
                    check = _RuleCheck(self, setup.override, independent,
                                       ExactLabelScorer(self.trace, h, "binary"), order)
                    checked = errorloc.ArmSetup(arm, "learned", scorer=setup.scorer,
                                                override=check)
                    result, _, tracer, _ = replay(self.trace, arm, setup=checked,
                                                  eligibility="all", observer=independent)
                    counts = check.counts
                    self.assertEqual(sum(counts[k] for k in ("rejected", "admitted", "later")),
                                     result.l2_decisions)
                    for kind in ("rejected", "admitted", "later", "differs_from_ranker"):
                        self.assertGreater(counts[kind], 0, kind)
                    # The class at h is not the 600-second class on this replay.
                    self.assertTrue(any(
                        _binary(self.trace, state_id, timestamp, h)
                        != _binary(self.trace, state_id, timestamp, 600.0)
                        for timestamp, _, _, candidates, _, _, _ in tracer.decisions
                        for state_id in candidates))

    def test_the_horizons_change_the_decisions(self):
        for order in matchedorder.ORDERS:
            streams = [_replay(self.trace, matched_arm(h, order),
                               _setup(matched_arm(h, order), self.trace))[2].decisions
                       for h in matchedorder.MATCHED_HORIZONS_SECONDS]
            for left in range(len(streams)):
                for right in range(left + 1, len(streams)):
                    self.assertNotEqual([d[:4] + d[5:] for d in streams[left]],
                                        [d[:4] + d[5:] for d in streams[right]], (order, left, right))

    def test_with_admission_by_the_h_second_label_the_recency_arm_is_the_label_binary_h_rung(self):
        for h in matchedorder.MATCHED_HORIZONS_SECONDS:
            rung_arm = horizonctl.horizon_arm(h)
            rungs = {"horizonfill": horizonfill.arm_setup(rung_arm, self.trace)}
            if rung_arm in horizonctl.HORIZON_ARMS:
                rungs["horizonctl"] = horizonctl.arm_setup(rung_arm, self.trace, HORIZON, 1)
            for source, rung_setup in rungs.items():
                with self.subTest(h=h, rung=source):
                    rung, rung_stats, rung_trace, _ = _replay(self.trace, rung_arm, rung_setup)
                    arm, arm_stats, arm_trace, _ = _replay(
                        self.trace, matchedorder.identity_arm(h),
                        matchedorder.identity_setup(self.trace, h))
                    self.assertEqual(arm_trace.decisions, rung_trace.decisions)
                    self.assertEqual(_row(arm), _row(rung))
                    self.assertEqual(arm_stats.row(), rung_stats.row())
                    # The identity is not vacuous: admission decisions are
                    # answered, arrivals are rejected and later rounds occur.
                    admissions = sum(1 for d in arm_trace.decisions if d[2] >= 0)
                    self.assertGreater(admissions, 0)
                    self.assertEqual(arm_trace.returned, admissions)
                    self.assertGreater(rung.l2_rejections, 0)
                    self.assertGreater(sum(1 for d in arm_trace.decisions if d[2] < 0), 0)


class MatchedKeyOnAHandBuiltDecisionTests(unittest.TestCase):
    """One decision on a constructed trace, the arrival first: r2's next use is
    300 s away, so it is reusable within 600 s and not within 60 s."""

    CANDIDATES = ["int:4", "int:1", "int:2", "int:3"]     # arrival, r1, r2, r3

    def setUp(self):
        records = [{"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [k]}
                   for k in (1, 2, 3, 4)]
        for timestamp, k in ((30_000, 1), (30_000, 4), (300_000, 2), (1_000_000, 3)):
            records.append({"timestamp": timestamp, "input_length": 512, "output_length": 1,
                            "hash_ids": [k]})
        temporary, self.trace, _ = horizon_tests.build(records)
        self.addCleanup(temporary.cleanup)
        # The ranker keeps the arrival and would itself evict r1 (reusable at
        # both horizons); r2 ranks below r3. r2 is older than r3.
        self.ranker = {"int:4": 9.0, "int:1": 0.5, "int:2": 1.0, "int:3": 2.0}
        self.last_group = {"int:4": 9, "int:1": 7, "int:2": 1, "int:3": 5}

    def _decide(self, h, order):
        setup = matchedorder.class_order_setup("arm", self.trace, _Fixed(self.ranker), order, h)
        keys = [(setup.scorer.score(state_id, 0.0), float(self.last_group[state_id]))
                for state_id in self.CANDIDATES]
        victim = first_minimum(keys)
        answer = setup.override(list(self.CANDIDATES), keys, victim, 0.0, 0, 0)
        return keys, victim, answer

    def test_the_classes_at_60_and_600_seconds_differ_here(self):
        for h, expected in ((60.0, [1.0, 1.0, 0.0, 0.0]), (600.0, [1.0, 1.0, 1.0, 0.0])):
            self.assertEqual([_binary(self.trace, state_id, 0.0, h)
                              for state_id in self.CANDIDATES], expected)

    def test_the_stores_key_orders_as_reusable_within_h_score_last_group(self):
        for h in (60.0, 600.0):
            for order in matchedorder.ORDERS:
                with self.subTest(h=h, order=order):
                    keys, _, _ = self._decide(h, order)
                    for index, state_id in enumerate(self.CANDIDATES):
                        reusable = _binary(self.trace, state_id, 0.0, h)
                        expected = ((reusable, self.ranker[state_id]) if order == "learned"
                                    else reusable)
                        self.assertEqual(keys[index], (expected, float(self.last_group[state_id])))
                    flat = [((_binary(self.trace, s, 0.0, h), self.ranker[s],
                              float(self.last_group[s])) if order == "learned"
                             else (_binary(self.trace, s, 0.0, h), float(self.last_group[s])))
                            for s in self.CANDIDATES]
                    self.assertEqual(sorted(range(4), key=keys.__getitem__),
                                     sorted(range(4), key=flat.__getitem__))

    def test_the_victim_moves_with_the_horizon(self):
        # At 60 s r2 joins the non-reusable class: the learned order evicts it
        # (ranker 1.0 < 2.0) and so does recency (last_group 1 < 5). At 600 s
        # r3 is the only non-reusable resident and leaves under both orders.
        expected = {(60.0, "learned"): 2, (60.0, "recency"): 2,
                    (600.0, "learned"): 3, (600.0, "recency"): 3}
        for (h, order), victim in expected.items():
            with self.subTest(h=h, order=order):
                _, _, answer = self._decide(h, order)
                self.assertEqual(answer, victim)
        # Recency splits from the learned order when r3 is the older one.
        self.last_group.update({"int:2": 6, "int:3": 1})
        self.assertEqual(self._decide(60.0, "learned")[2], 2)
        self.assertEqual(self._decide(60.0, "recency")[2], 3)


# --- the class statistic ---------------------------------------------------------------------------


class ClassStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_hand_built_decisions(self):
        records = [{"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [k]}
                   for k in (1, 2, 3, 4)]
        for timestamp, k in ((30_000, 1), (30_000, 4), (300_000, 2)):
            records.append({"timestamp": timestamp, "input_length": 512, "output_length": 1,
                            "hash_ids": [k]})
        temporary, trace, _ = horizon_tests.build(records)
        self.addCleanup(temporary.cleanup)
        # Reusable within 60 s: int:1, int:4; not: int:2 (300 s), int:3 (never).
        classes = ClassStatistics(trace, 60.0, measure_from_ms=10.0)
        keys = [(0.0, 0.0)] * 4
        # (candidates, original, final, timestamp, arriving_index)
        decisions = [
            (["int:4", "int:1", "int:2"], 1, 1, 0.0, 0),    # evicts reusable r1, r2 is not: violation
            (["int:2", "int:1", "int:4"], 1, 1, 0.0, 0),    # only the arrival is not reusable: none
            (["int:1", "int:2"], 0, 0, 0.0, 0),             # rejection: no resident eviction
            (["int:4", "int:3", "int:2"], 1, 1, 0.0, 0),    # evicts non-reusable r3: none
            (["int:1", "int:2"], 1, 0, 20.0, -1),           # later round, overridden, violation
        ]
        for candidates, original, final, timestamp, arriving in decisions:
            classes.record(candidates, keys[:len(candidates)], original, final, timestamp, 0,
                           arriving)
        row = classes.row()
        self.assertEqual(row["class_horizon_seconds"], 60.0)
        self.assertEqual((row["class_decisions_seen"], row["class_rejections_seen"],
                          row["class_resident_evictions_seen"], row["class_violations_seen"],
                          row["class_overridden_seen"],
                          row["class_overridden_outside_admission_seen"]), (5, 1, 4, 2, 1, 1))
        # Only the last decision is at or after the window start.
        self.assertEqual((row["class_decisions"], row["class_violations"],
                          row["class_overridden_outside_admission"],
                          row["class_m4_count_resident"]), (1, 1, 1, 1))
        self.assertEqual(set(row), set(matchedorder.CLASS_COLUMNS))

    def test_the_class_rule_is_the_error_location_hooks_constructed_at_h(self):
        # On the learned rung (which does evict reusable states), the class
        # statistic at h counts the binary target at h decision by decision,
        # and on resident-only decisions of the window it is the error-location
        # hook's m4_count_resident constructed at the same h.
        nonzero = 0
        for h in matchedorder.MATCHED_HORIZONS_SECONDS:
            with self.subTest(h=h):
                setup = errorloc.arm_setup("learned", self.trace, HORIZON, 1,
                                           learned_ranker=_LinearRanker())
                result, statistics, classes, tracer = class_replay(self.trace, setup, h)
                row, stats = classes.row(), statistics.row()
                self.assertEqual(row["class_m4_count_resident"], stats["m4_count_resident"])
                self.assertEqual(row["class_decisions"], stats["stat_decisions"])
                self.assertEqual((row["class_decisions_seen"], row["class_rejections_seen"],
                                  row["class_resident_evictions_seen"]),
                                 (result.l2_decisions, result.l2_rejections, result.l2_evictions))
                violations = 0
                for timestamp, _, arriving, candidates, _, _, final in tracer.decisions:
                    if arriving >= 0 and final == arriving:
                        continue
                    reusable = [_binary(self.trace, s, timestamp, h) == 1.0 for s in candidates]
                    residents = [i for i in range(len(candidates)) if i != arriving]
                    violations += reusable[final] and not all(reusable[i] for i in residents)
                self.assertEqual(row["class_violations_seen"], violations)
                nonzero += violations > 0
        self.assertGreater(nonzero, 0)

    def test_the_matched_arms_never_violate_their_class_and_override_only_admissions(self):
        for arm in ARMS:
            h = matchedorder.HORIZON_OF_ARM[arm]
            with self.subTest(arm=arm):
                result, statistics, classes, tracer = class_replay(self.trace,
                                                                   _setup(arm, self.trace), h)
                row = classes.row()
                self.assertEqual(row["class_violations_seen"], 0)
                self.assertEqual(row["class_overridden_outside_admission_seen"], 0)
                overridden = sum(1 for d in tracer.decisions if d[5] != d[6])
                self.assertGreater(overridden, 0)
                self.assertEqual(row["class_overridden_seen"], overridden)
                self.assertEqual(statistics.row()["overridden_decisions_seen"], overridden)
                self.assertGreater(row["class_resident_evictions_seen"], 0)

    def test_the_check_has_teeth_at_the_wrong_horizon(self):
        # The 60-second arm, measured against the 600-second class, does evict
        # states reusable within 600 s while a sampled resident is not.
        result, _, classes, _ = class_replay(
            self.trace, _setup("evict_binary_60_learned", self.trace), 600.0)
        self.assertGreater(classes.row()["class_violations_seen"], 0)

    def test_the_class_statistics_are_read_only(self):
        for arm in ("evict_binary_60_learned", "evict_binary_150_recency",
                    "evict_binary_600_learned"):
            with self.subTest(arm=arm):
                h = matchedorder.HORIZON_OF_ARM[arm]
                with_hooks, _, _, traced_with = class_replay(self.trace, _setup(arm, self.trace), h)
                without, _, _, traced_without = class_replay(self.trace, _setup(arm, self.trace),
                                                             h, wrapped=False)
                self.assertEqual(with_hooks.as_row(), without.as_row())
                self.assertEqual(traced_with.decisions, traced_without.decisions)


# --- the reading helpers ---------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_recovery_and_the_09_rule_at_its_boundary(self):
        self.assertEqual(matchedorder.matched_recovery(29.0, 2.0, 32.0), 0.9)
        self.assertEqual(matchedorder.matched_recovery(29.0, 2.0, 32.0),
                         horizonctl.recovery(29.0, 2.0, 32.0))
        self.assertTrue(math.isnan(matchedorder.matched_recovery(3.0, 2.0, 2.0)))
        reading = matchedorder.matched_reading
        suffices, costs = MATCHED_READINGS
        self.assertEqual((suffices, costs), ("reuse_identification_at_matched_horizon_suffices",
                                             "ranker_order_costs_at_matched_horizon"))
        self.assertEqual(reading(0.9), suffices)
        self.assertEqual(reading(matchedorder.matched_recovery(29.0, 2.0, 32.0)), suffices)
        self.assertEqual(reading(1.4), suffices)
        self.assertEqual(reading(0.8999999), costs)
        self.assertEqual(reading(-2.0), costs)
        self.assertEqual(reading(math.nan), costs)

    def test_counts_split_new_and_reproduced_and_are_never_merged(self):
        suffices, costs = MATCHED_READINGS
        counts = matchedorder.reading_counts([("new", suffices), ("new", costs),
                                              ("reproduced", suffices), ("new", suffices)])
        self.assertEqual(counts, {"new": {"cells": 3, suffices: 2, costs: 1},
                                  "reproduced": {"cells": 1, suffices: 1, costs: 0}})
        self.assertNotIn("all", counts)
        self.assertFalse(matchedorder.prediction_holds(counts["new"]))
        self.assertTrue(matchedorder.prediction_holds({"cells": 8, suffices: 8, costs: 0}))
        self.assertFalse(matchedorder.prediction_holds({"cells": 0, suffices: 0, costs: 0}))
        with self.assertRaises(ValueError):
            matchedorder.reading_counts([("all", suffices)])
        with self.assertRaises(ValueError):
            matchedorder.reading_counts([("new", "reuse_identification_suffices")])

    def test_seed_sign_readings(self):
        self.assertEqual(matchedorder.seed_signs([0.1, 2.0, 0.3, 0.01, 5.0]),
                         (5, 0, 0, "consistent_gain"))
        self.assertEqual(matchedorder.seed_signs([-0.1, -2.0, -0.3, -0.01, -5.0]),
                         (0, 0, 5, "consistent_loss"))
        self.assertEqual(matchedorder.seed_signs([0.1, 0.0, 0.3, 0.01, 5.0]),
                         (4, 1, 0, "mixed"))
        self.assertEqual(matchedorder.seed_signs([0.1, -0.2, 0.3, 0.01, 5.0])[3], "mixed")
        self.assertEqual(matchedorder.seed_signs([0.1, math.nan, 0.3, 0.01, 5.0])[3], "mixed")
        self.assertEqual(matchedorder.sign_reading_counts(["mixed", "consistent_gain", "mixed"]),
                         {"consistent_gain": 1, "consistent_loss": 0, "mixed": 2})
        with self.assertRaises(ValueError):
            matchedorder.sign_reading_counts(["gain"])


# --- the runner's derivations on synthetic rows ----------------------------------------------------

CELL_N60 = ("conversation_trace", 0.0025, 1.0)
CELL_N150 = ("toolagent_trace", 0.01, 1.0)
CELL_R600 = ("conversation_trace", 0.01, 4.0)
CELL_N300 = ("toolagent_trace", 0.02, 1.0)
IDENTIFIERS = {"l1_capacity_bytes": 1, "l2_capacity_bytes": 4, "requested_tokens": 1000,
               "l1_avoided_tokens": 50, "absent_compulsory_tokens": 77}
# U in tokens over 1000 requested tokens (points = tokens / 10). Published:
PUBLISHED_TOKENS = {"lru": 100, "learned": 200, "label": 600, "evict_label": 500,
                    "rung": 520, "evict_binary_learned": 300, "evict_binary_recency": 250}
# Matched arms: N60 R = 0.933 / 0.9; N150 0.8 / 0.9 (costs, learned < recency);
# R600 the published rows; N300 0.967 with recency mixed against it.
MATCHED_TOKENS = {CELL_N60: (480, 470), CELL_N150: (440, 470), CELL_R600: (300, 250),
                  CELL_N300: (490, 490)}
RECENCY_OFFSETS = {CELL_N300: (5, -5, 0, 5, -5)}


def synthetic_published():
    published = {}
    for cell in MATCHED_TOKENS:
        rung = matchedorder.RUNG_OF_HORIZON[MATCHED_HORIZON_SECONDS[cell[1:]]]
        for name, tokens in PUBLISHED_TOKENS.items():
            arm = rung if name == "rung" else name
            for seed in SEEDS:
                published[("all16", arm) + cell + (seed,)] = {
                    "source": runner.REFERENCES[("all16", arm)], "mechanism": "all16",
                    "arm": arm, "trace": cell[0], "l1_fraction": cell[1],
                    "l2_multiplier": cell[2], "cell": runner.rdp.cell_label(*cell[1:]),
                    "seed": seed, "avoided_prefill_tokens": tokens + seed + 50,
                    "extra_avoided_tokens": tokens + seed, **IDENTIFIERS}
    return published


def synthetic_rows():
    rows = []
    for cell, (learned, recency) in MATCHED_TOKENS.items():
        arms = cell_arms(*cell[1:])
        for seed in SEEDS:
            offset = RECENCY_OFFSETS.get(cell, (0,) * 5)[seed]
            for arm, tokens in zip(arms, (learned + seed, recency + seed + offset)):
                row = {metric: 0.0 for metric in runner.REPLAY_METRICS + runner.CLASS_METRICS}
                row.update(trace=cell[0], l1_fraction=cell[1], l2_multiplier=cell[2],
                           cell=runner.rdp.cell_label(*cell[1:]), arm=arm, seed=seed,
                           mechanism="all16", variant="main", extra_avoided_tokens=tokens,
                           avoided_prefill_tokens=tokens + 50, extra_points=tokens / 10.0,
                           **IDENTIFIERS)
                rows.append(row)
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.published = synthetic_published()
        self.tables = runner.matched_tables(self.rows, self.published)

    def _by(self, name):
        return {runner._cell_key(entry): entry for entry in self.tables[name]}

    def test_class_order(self):
        by = self._by("class_order")
        suffices, costs = MATCHED_READINGS
        expected = {CELL_N60: (60.0, "new", 280 / 300, 270 / 300, suffices),
                    CELL_N150: (150.0, "new", 0.8, 0.9, costs),
                    CELL_R600: (600.0, "reproduced", 1 / 3, 50 / 300, costs),
                    CELL_N300: (300.0, "new", 290 / 300, 290 / 300, suffices)}
        for cell, (h, role, r_learned, r_recency, reading) in expected.items():
            with self.subTest(cell=cell):
                entry = by[cell]
                self.assertEqual((entry["h_star"], entry["role"]), (h, role))
                self.assertAlmostEqual(entry["R_matched_learned"], r_learned, places=9)
                self.assertAlmostEqual(entry["R_matched_recency"], r_recency, places=9)
                self.assertEqual(entry["reading"], reading)
                self.assertEqual(entry["predicted_reading"], suffices if role == "new" else "")
                self.assertAlmostEqual(entry["R_600_published_learned"], 1 / 3, places=9)
                self.assertAlmostEqual(entry["R_600_published_recency"], 50 / 300, places=9)
                self.assertEqual(entry["published_600_reading"], "ranker_order_costs")
                self.assertEqual((entry["matched_learned_arm"], entry["matched_recency_arm"]),
                                 cell_arms(*cell[1:]))
                self.assertEqual(entry["rung_arm"], horizonctl.horizon_arm(h))
        self.assertEqual(by[CELL_N150]["rung_source"], "horizon_fill_001")
        self.assertEqual(by[CELL_N60]["rung_source"], "horizon_control_001")

    def test_order_admission_and_versus_600(self):
        order = self._by("order")
        self.assertEqual({cell: entry["learned_minus_recency_reading"]
                          for cell, entry in order.items()},
                         {CELL_N60: "consistent_gain", CELL_N150: "consistent_loss",
                          CELL_R600: "consistent_gain", CELL_N300: "mixed"})
        self.assertEqual(order[CELL_N300]["learned_minus_recency_seed_signs"], "-+0-+")
        self.assertAlmostEqual(order[CELL_N150]["learned_minus_recency_points_mean"], -3.0)
        self.assertAlmostEqual(order[CELL_N150]["R_matched_recency"], 0.9, places=9)
        self.assertTrue(all(entry["published_600_learned_minus_recency_reading"]
                            == "consistent_gain" for entry in order.values()))
        admission = self._by("admission")
        self.assertAlmostEqual(admission[CELL_R600]["label_binary_minus_recency_points_mean"], 27.0)
        self.assertAlmostEqual(admission[CELL_N60]["label_binary_minus_recency_points_mean"], 5.0)
        self.assertEqual(admission[CELL_N300]["label_binary_minus_recency_seed_signs"], "+++++")
        versus = self.tables["versus600"]
        self.assertEqual(len(versus), 6)
        self.assertNotIn(CELL_R600, {runner._cell_key(entry) for entry in versus})
        index = {(runner._cell_key(entry), entry["order"]): entry for entry in versus}
        self.assertAlmostEqual(index[(CELL_N60, "learned")]["matched_minus_600_points_mean"], 18.0)
        self.assertAlmostEqual(index[(CELL_N150, "recency")]["matched_minus_600_points_mean"],
                               22.0)
        self.assertEqual(index[(CELL_N300, "recency")]["published_arm"], "evict_binary_recency")
        self.assertEqual(index[(CELL_N300, "learned")]["matched_arm"], "evict_binary_300_learned")
        seeds = self.tables["class_order_seeds"]
        self.assertEqual(len(seeds), 4 * 5)
        first = next(entry for entry in seeds if runner._cell_key(entry) == CELL_N60
                     and entry["seed"] == 0)
        self.assertEqual(first["learned_minus_recency_tokens"], 10)
        self.assertEqual(first["matched_minus_600_recency_tokens"], 220)
        self.assertAlmostEqual(first["U_label_binary_hstar_points"], 52.0)

    def test_the_reading_counts(self):
        summary = runner.reading_summary(self.tables)
        by = {(entry["reading"], entry["scope"]): entry for entry in summary}
        suffices, costs = MATCHED_READINGS
        new = by[("1_class_order_at_matched_horizon", "new")]
        self.assertEqual((new["cells"], new[suffices], new[costs]), (3, 2, 1))
        self.assertEqual(new["predicted_" + suffices], 3)
        self.assertIs(new["prediction_holds"], False)
        reproduced = by[("1_class_order_at_matched_horizon", "reproduced")]
        self.assertEqual((reproduced["cells"], reproduced[suffices], reproduced[costs]), (1, 0, 1))
        self.assertEqual(reproduced["prediction_holds"], "")
        self.assertNotIn(("1_class_order_at_matched_horizon", "all"), by)
        order = by[("2_order_within_matched_class", "all")]
        self.assertEqual((order["consistent_gain"], order["consistent_loss"], order["mixed"],
                          order["cells"]), (2, 1, 1, 4))
        self.assertEqual(by[("3_admission_at_matched_horizon", "all")]["consistent_gain"], 4)
        for name in matchedorder.ORDERS:
            entry = by[(f"4_matched_minus_600_{name}", "new")]
            self.assertEqual((entry["cells"], entry["consistent_gain"]), (3, 3))
        self.assertEqual(len(summary), 7)

    def test_aggregation_and_reference_rows(self):
        summary = runner.aggregate_replays(self.rows)
        self.assertEqual(len(summary), 8)
        entry = next(e for e in summary if runner._cell_key(e) == CELL_N60
                     and e["arm"] == "evict_binary_60_learned")
        self.assertEqual((entry["arm_parameter"], entry["order"], entry["role"],
                          entry["published_arm"]), (60.0, "learned", "new", ""))
        self.assertAlmostEqual(entry["extra_points_mean"], 48.2)
        reproduced = next(e for e in summary if e["arm"] == "evict_binary_600_recency")
        self.assertEqual(reproduced["published_arm"], "evict_binary_recency")
        references = runner.reference_rows(self.published, self.rows)
        # Four base references, the cell's rung and the two published arms.
        self.assertEqual(len(references), 4 * 5 * 7)
        self.assertEqual({entry["arm"] for entry in references
                          if (entry["trace"], entry["l1_fraction"], entry["l2_multiplier"])
                          == CELL_N150} - {"lru", "learned", "label", "evict_label",
                                           "evict_binary_learned", "evict_binary_recency"},
                         {"label_binary_150"})

    def test_the_derivation_reproduces_the_published_class_order_on_its_own_rows(self):
        # Feed the published 600-second class-order rows in as if they were the
        # matched arms of every cell: R_h* must then be the horizon control's
        # published R, and its order reading the published one, cell by cell.
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        rows = []
        for name in runner.TRACES:
            for fraction, multiplier in runner.CELLS:
                for seed in runner.SEEDS:
                    for arm in cell_arms(fraction, multiplier):
                        source = published[("all16", PUBLISHED_ARMS[matched_arm(
                            600, matchedorder.ORDER_OF_ARM[arm])], name, fraction, multiplier,
                            seed)]
                        rows.append(dict(source, arm=arm))
        tables = runner.matched_tables(rows, published)
        expected = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])): row
                    for row in csv_rows(REPOSITORY / "results/paper/horizon_control_001/"
                                                     "class_order.csv")}
        self.assertEqual(len(tables["class_order"]), 12)
        for entry in tables["class_order"]:
            reference = expected[runner._cell_key(entry)]
            for mine, theirs in (("R_matched_learned", "R_evict_binary_learned"),
                                 ("R_matched_recency", "R_evict_binary_recency"),
                                 ("R_600_published_learned", "R_evict_binary_learned"),
                                 ("R_600_published_recency", "R_evict_binary_recency")):
                self.assertAlmostEqual(entry[mine], float(reference[theirs]), places=12)
            self.assertEqual(entry["published_600_reading"], reference["reading"])
        for entry in tables["order"]:
            reference = expected[runner._cell_key(entry)]
            self.assertEqual(entry["learned_minus_recency_seed_signs"],
                             reference["learned_minus_recency_seed_signs"])
            self.assertEqual(entry["learned_minus_recency_reading"],
                             reference["learned_minus_recency_reading"])
            self.assertAlmostEqual(entry["learned_minus_recency_points_mean"],
                                   float(reference["learned_minus_recency_points_mean"]),
                                   places=12)
        for entry in tables["versus600"]:
            self.assertEqual(entry["matched_minus_600_points_mean"], 0.0)


# --- the runner's checks and refusals --------------------------------------------------------------


def _class_row(arm="evict_binary_60_learned", cell=(0.0025, 1.0), **changes):
    """A replay row as `check_class` and `check_statistics` read it."""
    h = MATCHED_HORIZON_SECONDS[cell]
    row = {"trace": "conversation_trace", "l1_fraction": cell[0], "l2_multiplier": cell[1],
           "cell": runner.rdp.cell_label(*cell), "seed": 0, "arm": arm, "variant": "main",
           "arm_parameter": h, "class_horizon_seconds": h,
           "l2_decisions": 10, "l2_rejections": 3, "l2_evictions": 7,
           "stat_decisions_seen": 10, "stat_rejections_seen": 3, "stat_evictions_seen": 7,
           "overridden_decisions_seen": 4, "stat_decisions": 6, "overridden_decisions": 2,
           "overridden_decisions_resident": 0, "m4_count_resident": 0,
           "class_decisions_seen": 10, "class_rejections_seen": 3,
           "class_resident_evictions_seen": 7, "class_violations_seen": 0,
           "class_overridden_seen": 4, "class_overridden_outside_admission_seen": 0,
           "class_decisions": 6, "class_rejections": 2, "class_resident_evictions": 4,
           "class_violations": 0, "class_overridden": 2, "class_overridden_outside_admission": 0,
           "class_m4_count_resident": 0}
    row.update(changes)
    return row


def _reproduction_row(cell=(0.01, 4.0), seed=0, arm="evict_binary_600_learned", **changes):
    row = {"trace": "conversation_trace", "l1_fraction": cell[0], "l2_multiplier": cell[1],
           "cell": runner.rdp.cell_label(*cell), "seed": seed, "arm": arm,
           "avoided_prefill_tokens": 1000 + seed, "counters_sha256": f"c{seed}",
           "decision_sha256": f"d{seed}"}
    row.update(changes)
    return row


def _reproduction_published(rows):
    return {("all16", PUBLISHED_ARMS[row["arm"]], row["trace"], row["l1_fraction"],
             row["l2_multiplier"], row["seed"]): {
                "source": "horizon_control_001",
                "avoided_prefill_tokens": row["avoided_prefill_tokens"],
                "counters_sha256": row["counters_sha256"],
                "decision_sha256": row["decision_sha256"]} for row in rows}


class RunnerCheckTests(unittest.TestCase):
    def test_reproduction(self):
        rows = [_reproduction_row(seed=seed, arm=arm) for seed in SEEDS for arm in PUBLISHED_ARMS]
        rows.append(_reproduction_row(cell=(0.0025, 1.0), arm="evict_binary_60_learned"))
        published = _reproduction_published(rows[:-1])
        report, table = runner.check_reproduction(rows, published, 10)
        self.assertEqual((report["matched"], report["replays"], report["mismatched"],
                          report["missing"], report["decision_digests_compared"]),
                         (10, 10, 0, 0, 10))
        self.assertTrue(runner.reproduction_passes(report))
        self.assertEqual(len(table), 10)
        self.assertTrue(all(entry["reproduces"] for entry in table))
        self.assertEqual(table[0]["published_arm"], "evict_binary_learned")
        self.assertNotIn("reproduces_reference", rows[-1])
        for column, value in (("avoided_prefill_tokens", 7), ("counters_sha256", "x"),
                              ("decision_sha256", "y")):
            with self.subTest(column=column):
                changed = [dict(row) for row in rows]
                changed[3][column] = value
                report, table = runner.check_reproduction(changed, published, 10)
                self.assertEqual((report["matched"], report["mismatched"]), (9, 1))
                self.assertFalse(runner.reproduction_passes(report))
                self.assertIn(column, report["mismatches"][0])
                self.assertIs(changed[3]["reproduces_reference"], False)
        # A published row without a counter digest cannot reproduce.
        bare = dict(published)
        key = next(iter(bare))
        bare[key] = {k: v for k, v in bare[key].items() if k != "counters_sha256"}
        report, _ = runner.check_reproduction(rows, bare, 10)
        self.assertEqual(report["mismatched"], 1)
        # A missing published row and a wrong expectation fail.
        del bare[key]
        report, _ = runner.check_reproduction(rows, bare, 10)
        self.assertEqual((report["missing"], report["matched"]), (1, 9))
        self.assertFalse(runner.reproduction_passes(report))
        report, _ = runner.check_reproduction(rows, published, 12)
        self.assertFalse(runner.reproduction_passes(report))

    def test_identifiers_against_every_reference_of_the_cell(self):
        published = synthetic_published()
        rows = synthetic_rows()
        compared, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])
        self.assertEqual(compared, len(rows) * 7)
        rows[0]["absent_compulsory_tokens"] = 78
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 7)
        self.assertTrue(all("absent_compulsory_tokens 78 != 77" in line for line in problems))
        del published[("all16", "label_binary_60") + CELL_N60 + (1,)]
        _, problems = runner.check_identifiers(rows, published)
        self.assertTrue(any("no published label_binary_60/all16" in line for line in problems))

    def test_class(self):
        self.assertEqual(runner.check_class([_class_row()]), [])
        self.assertEqual(runner.check_class([_class_row(
            "evict_binary_600_recency", (0.02, 4.0))]), [])
        cases = (
            (_class_row(class_violations_seen=2), "2 resident evictions discard a state "
                                                  "reusable within 60 s"),
            (_class_row(class_overridden_outside_admission_seen=1),
             "the arrival is not a candidate"),
            (_class_row(overridden_decisions_resident=1), "overridden_decisions_resident 1"),
            (_class_row(class_horizon_seconds=600.0), "the cell's h* is 60 s"),
            (_class_row(arm_parameter=150.0), "the cell's h* is 60 s"),
            (_class_row(class_decisions_seen=9), "class_decisions_seen 9 != l2_decisions 10"),
            (_class_row(class_rejections_seen=2), "class_rejections_seen 2 != l2_rejections"),
            (_class_row(class_resident_evictions_seen=6), "class_resident_evictions_seen 6"),
            (_class_row(class_overridden_seen=3), "class_overridden_seen 3 != "
                                                  "overridden_decisions_seen 4"),
            (_class_row(class_decisions=5), "class_decisions 5 != stat_decisions 6"),
            (_class_row(class_overridden=1), "class_overridden 1 != overridden_decisions 2"),
            (_class_row("evict_binary_600_learned", (0.01, 4.0), class_m4_count_resident=1),
             "class_m4_count_resident 1 != m4_count_resident 0 at 600 s"),
        )
        for row, text in cases:
            with self.subTest(case=text):
                problems = runner.check_class([row])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])
        # Below 600 s the 600-second m4 is not the class at h*: no comparison.
        self.assertEqual(runner.check_class([_class_row(m4_count_resident=3)]), [])
        # The statistics check of the horizon control applies as it is.
        self.assertEqual(runner.check_statistics([_class_row()]), [])
        self.assertEqual(len(runner.check_statistics([_class_row(stat_decisions_seen=9)])), 1)

    def test_the_published_references_load_as_the_plan_names_them(self):
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        self.assertEqual(len(published), 4 * 60 + 60 + 2 * 60)
        self.assertEqual(published[("all16", "label_binary_150", "toolagent_trace", 0.01, 1.0, 3)]
                         ["source"], "horizon_fill_001")
        self.assertEqual(published[("all16", "label_binary_60", "conversation_trace", 0.0025, 1.0,
                                    0)]["source"], "horizon_control_001")
        self.assertNotIn(("all16", "label_binary_60", "conversation_trace", 0.01, 1.0, 0),
                         published)
        for key, entry in published.items():
            self.assertEqual(set(runner.REFERENCE_IDENTIFIERS) - set(entry), set())
            if key[1] in PUBLISHED_ARMS.values():
                self.assertEqual(len(entry["counters_sha256"]), 64)
                self.assertEqual(len(entry["decision_sha256"]), 64)
        reproduced = [key for key in published if key[1] in PUBLISHED_ARMS.values()
                      and key[3:5] in REPRODUCED_CELLS]
        self.assertEqual(len(reproduced), 40)

    def test_reference_loading_keeps_each_reference_to_its_source(self):
        header = ("trace,l1_fraction,l2_multiplier,cell,eligibility,width,mechanism,arm,seed,"
                  "variant,avoided_prefill_tokens,extra_avoided_tokens,l1_capacity_bytes,"
                  "l2_capacity_bytes,requested_tokens,l1_avoided_tokens,counters_sha256\n")
        line = "conversation_trace,{},{},c,{},16,all16,{},0,{},10,5,1,4,100,5,abc\n"
        with tempfile.TemporaryDirectory() as directory:
            paths = {source: Path(directory) / f"{source}.csv" for source in runner.REFERENCE_SOURCES}
            paths["error_location_001"].write_text(
                header + line.format(0.01, 4.0, "all", "lru", "main")
                + line.format(0.01, 4.0, "all", "lru", "main")            # duplicate
                + line.format(0.01, 4.0, "leaf", "learned", "main")       # wrong eligibility
                + line.format(0.01, 4.0, "all", "label", "nostats")       # not main
                + line.format(0.01, 4.0, "all", "label_binary_600", "main"))  # wrong source
            paths["horizon_control_001"].write_text(
                header + line.format(0.01, 4.0, "all", "label_binary_600", "main")
                + line.format(0.01, 1.0, "all", "label_binary_150", "main")   # wrong source
                + line.format(0.01, 1.0, "all", "label_binary_600", "main")   # not its cell
                + line.format(0.01, 4.0, "all", "evict_binary_learned", "main"))
            paths["horizon_fill_001"].write_text(
                header + line.format(0.01, 1.0, "all", "label_binary_150", "main")
                + line.format(0.01, 4.0, "all", "label_binary_600", "main"))  # wrong source
            published, problems = runner.load_references(paths)
        cell = ("conversation_trace", 0.01, 4.0, 0)
        self.assertEqual(published[("all16", "lru") + cell]["source"], "error_location_001")
        self.assertEqual(published[("all16", "label_binary_600") + cell]["source"],
                         "horizon_control_001")
        self.assertEqual(published[("all16", "label_binary_150", "conversation_trace", 0.01, 1.0,
                                    0)]["source"], "horizon_fill_001")
        self.assertEqual(published[("all16", "evict_binary_learned") + cell]["counters_sha256"],
                         "abc")
        self.assertNotIn(("all16", "label") + cell, published)
        self.assertNotIn(("all16", "label_binary_600", "conversation_trace", 0.01, 1.0, 0),
                         published)
        self.assertTrue(any(line.startswith("duplicate") for line in problems))
        self.assertTrue(any("has leaf/16" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))
        self.assertFalse(any("label_binary_600" in line and "duplicate" in line
                             for line in problems))

    def test_argument_refusals_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "new"
            paper = Path(directory) / "paper"
            existing = Path(directory)
            base = ["a.jsonl", "b.jsonl", "--output-dir", str(fresh), "--paper-dir", str(paper)]
            cases = {"positive": base + ["--workers", "0"],
                     "hard cap of 12": base + ["--workers", "13"],
                     "expected 2 trace files": base[1:],
                     "must differ": ["a.jsonl", "b.jsonl", "--output-dir", str(fresh),
                                     "--paper-dir", str(fresh)]}
            for text, argv in cases.items():
                with self.subTest(case=text), self.assertRaises(SystemExit) as caught:
                    runner.validate_arguments(runner.parse_args(argv))
                self.assertIn(text, str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(
                    ["a.jsonl", "b.jsonl", "--output-dir", str(existing), "--paper-dir",
                     str(paper)]))
            self.assertIn("output directory", str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(
                    ["a.jsonl", "b.jsonl", "--output-dir", str(fresh), "--paper-dir",
                     str(existing)]))
            self.assertIn("paper directory", str(caught.exception))
            runner.validate_arguments(runner.parse_args(base + ["--workers", "12"]))
        args = runner.parse_args(["a.jsonl", "b.jsonl", "--output-dir", "x"])
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.paper_dir,
                         REPOSITORY / "results/paper/matched_class_order_001")

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output, paper = Path(directory) / "run", Path(directory) / "paper"
            argv = ["run_matched_class_order.py", "a.jsonl", "b.jsonl", "--output-dir",
                    str(output), "--paper-dir", str(paper)]

            def dirty(*arguments):
                return " M src/x.py" if arguments[0] == "status" else "0" * 40

            for git, differing, text in ((dirty, [], "not clean"),
                                         (lambda *a: "" if a[0] == "status" else "0" * 40,
                                          ["src/x.py"], "differs from HEAD")):
                with self.subTest(case=text), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(runner.rmc, "_git", git), \
                        mock.patch.object(runner, "sources_differing_from_head",
                                          lambda: differing), \
                        self.assertRaises(SystemExit) as caught:
                    runner.main()
                self.assertIn(text, str(caught.exception))
                self.assertFalse(output.exists())
                self.assertFalse(paper.exists())

    def test_the_execution_sources_cover_the_imported_runners(self):
        sources = {str(path.relative_to(REPOSITORY)) for path in runner.execution_sources()}
        for name in ("scripts/run_matched_class_order.py", "scripts/run_horizon_control.py",
                     "scripts/run_error_location.py", "scripts/run_mechanism_control.py",
                     "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/matchedorder.py",
                     "src/persistent_kv_admission/horizonctl.py",
                     "src/persistent_kv_admission/errorloc.py"):
            self.assertIn(name, sources)

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "class_order_seeds",
                     "class_order", "order", "admission", "versus600", "reproduction",
                     "readings"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        self.assertIn("`run_config.json`", runner.README_TEXT)
        self.assertIn("never merged", runner.README_TEXT)


# --- the replay worker -----------------------------------------------------------------------------


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker` on a constructed trace, with the module state the
    runner's main would set, against the horizon control's own worker."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        name = self.trace.name
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON},
                      rankers={name: _LinearRanker()})
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.hc.SHARED, shared),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def test_at_600_seconds_the_worker_is_the_horizon_controls_digest_for_digest(self):
        for fraction, multiplier in REPRODUCED_CELLS:
            for arm, published in PUBLISHED_ARMS.items():
                with self.subTest(cell=(fraction, multiplier), arm=arm):
                    mine = runner._replay_worker((self.name, fraction, multiplier, arm, 1))
                    theirs = runner.hc._replay_worker((self.name, fraction, multiplier, published,
                                                       1, "main"))
                    self.assertEqual(mine["counters_sha256"], theirs["counters_sha256"])
                    self.assertEqual(mine["decision_sha256"], theirs["decision_sha256"])
                    self.assertEqual(mine["avoided_prefill_tokens"],
                                     theirs["avoided_prefill_tokens"])
                    differing = [key for key in theirs if key not in VOLATILE + NAMING
                                 and not _same(mine[key], theirs[key])]
                    self.assertEqual(differing, [])
                    self.assertEqual(set(theirs) - set(mine), set())
                    self.assertEqual(set(mine) - set(theirs),
                                     {"order", "role", "published_arm"}
                                     | set(matchedorder.CLASS_COLUMNS))
                    self.assertEqual((mine["arm"], mine["family"], mine["arm_parameter"],
                                      mine["published_arm"], mine["role"]),
                                     (arm, "class_order", 600.0, published, "reproduced"))
                    self.assertGreater(mine["l2_decisions"], 0)

    def test_worker_rows_pass_the_checks(self):
        rows = [runner._replay_worker((self.name, fraction, multiplier, arm, 0))
                for fraction, multiplier in runner.CELLS for arm in cell_arms(fraction, multiplier)]
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_class(rows), [])
        groups, problems = runner.rmc.check_invariants(rows, 2)
        self.assertEqual((groups, problems), (6, []))
        for row in rows:
            h = MATCHED_HORIZON_SECONDS[(row["l1_fraction"], row["l2_multiplier"])]
            self.assertEqual((row["mechanism"], row["eligibility"], row["width"], row["family"],
                              row["variant"], row["arm_parameter"], row["class_horizon_seconds"]),
                             ("all16", "all", 16, "class_order", "main", h, h))
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertGreater(row["stat_decisions"], 0)
            self.assertGreater(row["overridden_decisions_seen"], 0)
            self.assertEqual(row["class_violations_seen"], 0)
        # A publication of these very rows passes the identifier check and,
        # at h* = 600, the reproduction.
        published = {}
        for row in rows:
            for mechanism, arm in runner.REFERENCES:
                if (row["l1_fraction"], row["l2_multiplier"]) in runner.reference_cells(mechanism,
                                                                                       arm):
                    published[(mechanism, arm) + runner._cell_seed(row)] = {
                        "source": "s", **{field: row[field] for field in runner.REFERENCE_IDENTIFIERS}}
        for row in rows:
            if row["arm"] in PUBLISHED_ARMS:
                published[("all16", PUBLISHED_ARMS[row["arm"]]) + runner._cell_seed(row)].update(
                    avoided_prefill_tokens=row["avoided_prefill_tokens"],
                    counters_sha256=row["counters_sha256"], decision_sha256=row["decision_sha256"])
        self.assertEqual(runner.check_identifiers(rows, published)[1], [])
        report, _ = runner.check_reproduction(rows, published, 4)
        self.assertTrue(runner.reproduction_passes(report))

    def test_the_worker_refuses_an_arm_that_is_not_its_cells(self):
        with self.assertRaises(ValueError):
            runner._replay_worker((self.name, 0.0025, 1.0, "evict_binary_600_learned", 0))


# --- the whole run on constructed traces -----------------------------------------------------------


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


def _fake_models(names):
    return ({name: _LinearRanker() for name in names},
            {name: {"path": f"models/pi0/{name}__next_use.json", "sha256": "a" * 64}
             for name in names})


class MainTests(unittest.TestCase):
    """`main` end to end on two constructed traces named as the grid's, with a
    constructed publication of every reference (the published class-order rows
    replayed by the horizon control's own worker, the other references
    synthesised on the same identifiers), git answered as for a clean tree and
    the ranker loader replaced by the fixed linear ranker."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.root = root
        traces = {}
        for name, seed in (("conversation_trace", 9), ("toolagent_trace", 5)):
            path = root / f"{name}.jsonl"
            records = horizon_tests.scaled(horizon_tests.error_tests.partial_records(seed=seed))
            path.write_text("".join(json.dumps(record) + "\n" for record in records),
                            encoding="utf-8")
            traces[name] = load_mooncake_trace(path, 512)
        cls.trace_paths = [str(root / f"{name}.jsonl") for name in traces]
        shared = {"traces": traces, "groups": {}, "splits": {}, "horizons": {},
                  "rankers": {name: _LinearRanker() for name in traces}}
        working_set = {}
        for name, trace in traces.items():
            horizon, split_ms, _ = horizon_for(trace)
            assert horizon == 600.0, (name, horizon)
            shared["groups"][name] = _occurrence_groups(trace)
            shared["splits"][name] = split_ms
            shared["horizons"][name] = horizon
            working_set[name] = working_set_bytes(trace)
        with mock.patch.dict(runner.hc.SHARED, shared), \
                mock.patch.dict(runner.rdp._SHARED, {"working_set": working_set}):
            published_rows = [runner.hc._replay_worker((name, fraction, multiplier, arm, seed,
                                                        "main"))
                              for name in sorted(traces) for fraction, multiplier in runner.CELLS
                              for arm in horizonctl.CLASS_ORDER_ARMS for seed in SEEDS]
        sources = {source: [] for source in runner.REFERENCE_SOURCES}
        for row in published_rows:
            sources["horizon_control_001"].append(row)
            if row["arm"] != "evict_binary_learned":
                continue
            extra = row["extra_avoided_tokens"]
            cell = (row["l1_fraction"], row["l2_multiplier"])
            rung = matchedorder.RUNG_OF_HORIZON[MATCHED_HORIZON_SECONDS[cell]]
            synthesised = [("error_location_001", "lru", extra // 4),
                           ("error_location_001", "learned", extra // 3),
                           ("error_location_001", "label", extra + 500),
                           ("error_location_001", "evict_label", extra + 400),
                           (runner.REFERENCES[("all16", rung)], rung, extra + 450),
                           # Rows the loader must skip: a rung at a cell that is
                           # not its own, and the fill-in's reruns of the
                           # horizon control's rungs.
                           ("horizon_control_001", "label_binary_300" if rung != "label_binary_300"
                            else "label_binary_60", extra)]
            if rung == "label_binary_150":
                synthesised.append(("horizon_fill_001", "label_binary_600", extra))
            for source, arm, tokens in synthesised:
                sources[source].append(dict(row, arm=arm, family="reference", arm_parameter="",
                                            extra_avoided_tokens=tokens,
                                            avoided_prefill_tokens=row["l1_avoided_tokens"]
                                            + tokens))
        cls.sources = {}
        for source, rows in sources.items():
            path = root / "published" / source / "replay_seeds.csv"
            path.parent.mkdir(parents=True)
            runner._write(path, rows)
            cls.sources[source] = path
        cls.published_rows = published_rows
        cls.manifests = (root / "manifest_a.csv", root / "manifest_b.csv")
        for path in cls.manifests:
            path.write_text("policy,target,trace,model_sha256\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _tampered(self, name, change) -> dict:
        """The fixture sources with one horizon-control row changed by `change`
        (a function of the rows that edits them in place)."""
        rows = list(csv_rows(self.sources["horizon_control_001"]))
        change(rows)
        path = self.root / name / "replay_seeds.csv"
        path.parent.mkdir(parents=True)
        runner._write(path, rows)
        return dict(self.sources, horizon_control_001=path)

    def _main(self, argv, sources=None, cells=None, seeds=None) -> str:
        patches = [mock.patch.object(sys, "argv", ["run_matched_class_order.py"]
                                     + self.trace_paths + argv),
                   mock.patch.object(runner.rmc, "_git", _fake_git),
                   mock.patch.object(runner, "sources_differing_from_head", lambda: []),
                   mock.patch.dict(runner.REFERENCE_SOURCES, sources or self.sources),
                   mock.patch.object(runner.rmc, "load_models", _fake_models),
                   mock.patch.object(runner.rmc, "MODEL_MANIFESTS", self.manifests),
                   mock.patch.dict(runner.SHARED), mock.patch.dict(runner.rdp._SHARED)]
        if cells is not None:
            patches.append(mock.patch.object(runner, "CELLS", cells))
        if seeds is not None:
            patches.append(mock.patch.object(runner, "SEEDS", seeds))
        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            try:
                runner.main()
            finally:
                self.stdout = stdout.getvalue()
        return self.stdout

    def test_a_run_publishes_every_table_after_every_check(self):
        output, paper = self.root / "run_ok", self.root / "paper_ok"
        text = self._main(["--output-dir", str(output), "--paper-dir", str(paper),
                           "--workers", "2"])
        tables = {"replay_seeds.csv", "replay.csv", "references_seeds.csv",
                  "class_order_seeds.csv", "class_order.csv", "order.csv", "admission.csv",
                  "versus600.csv", "reproduction.csv", "readings.csv", "README.md",
                  "run_config.json"}
        self.assertEqual(set(os.listdir(paper)), tables)
        self.assertEqual(set(os.listdir(output)), tables | {"raw_replays.jsonl"})
        self.assertEqual(len((output / "raw_replays.jsonl").read_text().splitlines()), 120)
        self.assertIn("[120/120]", text)
        self.assertIn("reading 1, class order at the matched horizon", text)
        replays = list(csv_rows(paper / "replay_seeds.csv"))
        self.assertEqual(len(replays), 120)
        self.assertEqual(list(replays[0])[:12],
                         ["trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width",
                          "mechanism", "arm", "family", "arm_parameter", "seed", "variant"])
        # The 40 replays at h* = 600 are the horizon control's own rows, digest
        # for digest; every replay keeps its class.
        index = {(row["arm"],) + runner._cell_seed(row): row for row in replays}
        compared = 0
        for row in self.published_rows:
            if (row["l1_fraction"], row["l2_multiplier"]) not in REPRODUCED_CELLS:
                continue
            arm = matched_arm(600, horizonctl.CLASS_ORDER[row["arm"]])
            mine = index[(arm,) + runner._cell_seed(row)]
            self.assertEqual((mine["counters_sha256"], mine["decision_sha256"],
                              int(mine["avoided_prefill_tokens"])),
                             (row["counters_sha256"], row["decision_sha256"],
                              row["avoided_prefill_tokens"]))
            self.assertEqual(mine["reproduces_reference"], "True")
            compared += 1
        self.assertEqual(compared, 40)
        self.assertTrue(all(int(row["class_violations_seen"]) == 0 for row in replays))
        reproduction = list(csv_rows(paper / "reproduction.csv"))
        self.assertEqual(len(reproduction), 40)
        self.assertTrue(all(row["reproduces"] == "True" for row in reproduction))
        counts = {name: len(list(csv_rows(paper / f"{name}.csv")))
                  for name in ("class_order", "order", "admission", "versus600",
                               "class_order_seeds", "references_seeds", "readings", "replay")}
        self.assertEqual(counts, {"class_order": 12, "order": 12, "admission": 12,
                                  "versus600": 16, "class_order_seeds": 60,
                                  "references_seeds": 60 * 7, "readings": 7, "replay": 24})
        readings = {(row["reading"], row["scope"]): row
                    for row in csv_rows(paper / "readings.csv")}
        self.assertEqual(readings[("1_class_order_at_matched_horizon", "new")]["cells"], "8")
        self.assertEqual(readings[("1_class_order_at_matched_horizon", "reproduced")]["cells"],
                         "4")
        self.assertEqual(readings[("2_order_within_matched_class", "all")]["cells"], "12")
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["phase"], "matched_class_order")
        self.assertEqual(config["plan"], "docs/matched-class-order-plan.md")
        self.assertEqual((config["plan_commit"], config["code_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(set(config["references"]),
                         {str(path.resolve()) for path in self.sources.values()})
        for path in self.sources.values():
            self.assertEqual(config["references"][str(path.resolve())], sha256_path(path))
        self.assertEqual(set(config["model_manifests"]),
                         {str(path.resolve()) for path in self.manifests})
        self.assertEqual(config["models"]["pi0_next_use"]["conversation_trace"]["sha256"],
                         "a" * 64)
        self.assertEqual(set(config["trace_files"]), {"conversation_trace", "toolagent_trace"})
        self.assertEqual({(entry["l1_fraction"], entry["l2_multiplier"]): entry["h_star"]
                          for entry in config["matched_horizon_seconds"]},
                         MATCHED_HORIZON_SECONDS)
        checks = config["checks"]
        self.assertEqual((checks["reproduction"]["matched"], checks["reproduction"]["expected"],
                          checks["reproduction"]["decision_digests_compared"]), (40, 40, 40))
        self.assertEqual((checks["identifier_problems"], checks["statistics_problems"],
                          checks["class_problems"], checks["class_violations"],
                          checks["class_overridden_outside_admission"],
                          checks["invariant_violations"], checks["unexplained_absent_tokens"]),
                         (0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(checks["identifier_pairs_compared"], 120 * 7)
        self.assertEqual((config["replays"], config["workers"]), (120, 2))
        self.assertIn("never merged", config["note"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(tabulate.main([str(paper)]), 0)
        printed = stdout.getvalue()
        for section in ("M0", "M1", "M2", "M3", "M4"):
            self.assertIn(f"### {section} ", printed)
        self.assertEqual(sorted(os.listdir(paper)), sorted(tables))

    def test_a_reproduction_difference_publishes_nothing(self):
        def counters(rows):
            target = next(row for row in rows if row["arm"] == "evict_binary_recency"
                          and float(row["l1_fraction"]) == 0.01 and row["seed"] == "0"
                          and float(row["l2_multiplier"]) == 4.0
                          and row["trace"] == "conversation_trace")
            target["counters_sha256"] = "0" * 64

        def tokens(rows):
            target = next(row for row in rows if row["arm"] == "evict_binary_learned"
                          and float(row["l1_fraction"]) == 0.01 and row["seed"] == "0"
                          and float(row["l2_multiplier"]) == 4.0
                          and row["trace"] == "toolagent_trace")
            target["avoided_prefill_tokens"] = str(int(target["avoided_prefill_tokens"]) + 1)

        for name, change, column in (("counters", counters, "counters_sha256"),
                                     ("tokens", tokens, "avoided_prefill_tokens")):
            with self.subTest(case=name):
                sources = self._tampered(f"tampered_{name}", change)
                output, paper = self.root / f"run_bad_{name}", self.root / f"paper_bad_{name}"
                with self.assertRaises(SystemExit) as caught:
                    self._main(["--output-dir", str(output), "--paper-dir", str(paper),
                                "--workers", "2"], sources=sources, cells=((0.01, 4.0),),
                               seeds=(0,))
                self.assertIn("do not reproduce", str(caught.exception))
                self.assertIn("nothing derived or published", str(caught.exception))
                self.assertIn("MISMATCH", self.stdout)
                self.assertIn(column, self.stdout)
                self.assertFalse(paper.exists())
                self.assertEqual(os.listdir(output), ["raw_replays.jsonl"])

    def test_a_cell_subset_runs_and_publishes(self):
        output, paper = self.root / "run_subset", self.root / "paper_subset"
        self._main(["--output-dir", str(output), "--paper-dir", str(paper), "--workers", "2"],
                   cells=((0.0025, 4.0), (0.02, 4.0)), seeds=(0, 1))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["replays"], 2 * 2 * 2 * 2)
        self.assertEqual(config["checks"]["reproduction"]["matched"], 2 * 2 * 2)
        self.assertEqual(len(list(csv_rows(paper / "versus600.csv"))), 2 * 2)

    def test_missing_references_are_refused_before_any_replay(self):
        def drop(rows):
            rows[:] = [row for row in rows if not (row["arm"] == "evict_binary_learned"
                                                   and float(row["l1_fraction"]) == 0.0025
                                                   and row["seed"] == "4")]

        sources = self._tampered("missing", drop)
        output, paper = self.root / "run_refused", self.root / "paper_refused"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper)], sources=sources)
        self.assertIn("nothing run", str(caught.exception))
        self.assertIn("REFERENCE", self.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(paper.exists())


# --- the tabulation --------------------------------------------------------------------------------


class TabulationTests(unittest.TestCase):
    def test_tabulation_reads_and_prints_without_writing(self):
        rows = synthetic_rows()
        published = synthetic_published()
        tables = runner.matched_tables(rows, published)
        summary = runner.reading_summary(tables)
        reproduction_rows = [_reproduction_row(seed=seed, arm=arm)
                             for seed in SEEDS for arm in PUBLISHED_ARMS]
        _, reproduction = runner.check_reproduction(
            reproduction_rows, _reproduction_published(reproduction_rows), 10)
        written = {"replay": runner.aggregate_replays(rows), "reproduction": reproduction,
                   "readings": summary, **{name: tables[name] for name in
                                           ("class_order", "order", "admission", "versus600")}}
        with tempfile.TemporaryDirectory() as directory:
            for name, table in written.items():
                runner._write(Path(directory) / f"{name}.csv", table)
            before = sorted(os.listdir(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(tabulate.main([directory]), 0)
            self.assertEqual(sorted(os.listdir(directory)), before)
            parsed = {name: tabulate.read_csv(Path(directory) / f"{name}.csv")
                      for name in written}
        text = stdout.getvalue()
        for section in ("M0", "M1", "M2", "M3", "M4"):
            self.assertIn(f"### {section} ", text)
        suffices, costs = MATCHED_READINGS
        self.assertEqual(tabulate.class_order_counts(parsed["readings"]), [
            {"scope": "new", "cells": 3, suffices: 2, costs: 1, "predicted": "3",
             "holds": False},
            {"scope": "reproduced", "cells": 1, suffices: 0, costs: 1, "predicted": "",
             "holds": ""}])
        self.assertEqual(tabulate.reproduction_counts(parsed["reproduction"]), [
            {"published_arm": "evict_binary_learned", "replays": 5,
             "same_avoided_prefill_tokens": 5, "same_counters_sha256": 5,
             "same_decision_sha256": 5, "reproduces": 5},
            {"published_arm": "evict_binary_recency", "replays": 5,
             "same_avoided_prefill_tokens": 5, "same_counters_sha256": 5,
             "same_decision_sha256": 5, "reproduces": 5}])
        table = tabulate.class_order_table(parsed["class_order"])
        self.assertEqual([(row["trace"], row["l1_fraction"]) for row in table],
                         [("conversation_trace", 0.0025), ("conversation_trace", 0.01),
                          ("toolagent_trace", 0.01), ("toolagent_trace", 0.02)])
        self.assertAlmostEqual(table[0]["R_learned"], 280 / 300, places=9)
        order = tabulate.order_table(parsed["order"])
        self.assertEqual(order[-1]["signs"], (2, 1, 2))
        self.assertEqual([row["order"] for row in tabulate.versus600_table(parsed["versus600"])],
                         ["learned", "recency"] * 3)
        signs = {(row["reading"], row["scope"]): row
                 for row in tabulate.sign_counts_table(parsed["readings"])}
        self.assertEqual(signs[("2_order_within_matched_class", "all")]["mixed"], 1)
        self.assertEqual(len(tabulate.admission_table(parsed["admission"])), 4)
        self.assertEqual([row["arm"] for row in tabulate.replay_counters(parsed["replay"])][:2],
                         ["evict_binary_60_learned", "evict_binary_60_recency"])


if __name__ == "__main__":
    unittest.main()
