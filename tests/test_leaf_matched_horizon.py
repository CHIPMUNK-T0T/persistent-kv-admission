"""Tests for the leaf-matched horizon control (docs/leaf-matched-horizon-plan.md).

No real trace is replayed: the plan forbids a real replay before the
implementation is reviewed and committed. Everything here runs on constructed
traces, hand-built rows and synthetic publications; the published result
tables are only read. What has to hold before a real replay means anything:
every arm is its parent's construction, unchanged, and only the store's
eligibility differs, so on a trace where no state has a cached child the
`leaf16` replay is the `all16` one decision by decision, and with the
eligibility set back to `all` the runner's worker is the parents' worker
digest for digest; under `leaf16`, with admission by the exact `h*` label in
the ranker's place, the recency arm is the `label_binary_h*` rung decision by
decision (counter and decision digests included); the `label` anchor is the
error-location `label` rung and the horizon control's and mechanism control's
`leaf16` `label` replay; no block is ever present but unusable; the class
statistic is zero in every `h*` arm; the reading helpers do what the plan
fixes at their boundaries; the `all16` values the run prints are the
published ones; and the runner's checks, refusals, derivations, its
end-to-end run with git mocked (the `leaf16` reference rows produced by the
parents' own workers) and the read-only tabulation behave as stated,
including that a reproduction difference publishes nothing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from persistent_kv_admission import errorloc, horizonctl, horizonfill, leafmatched, matchedorder
from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.decisionpop import RawFeatureScorer, horizon_for
from persistent_kv_admission.errorloc import (
    DecisionStatistics,
    HybridOverride,
    HybridScorer,
    RecordingOverride,
)
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import ReuseClassScorer
from persistent_kv_admission.leafmatched import (
    ANCHOR_ARM,
    CLASS_READINGS,
    HORIZON_READINGS,
    RUNG_ARMS,
    cell_arms,
)
from persistent_kv_admission.matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    ClassStatistics,
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
# traced replay helper, the legality recorder of its leaf hybrids), reused as
# they are; only plain functions and classes are read from them.
horizon_tests = _load("_leaf_matched_horizon_control_fixtures",
                      REPOSITORY / "tests/test_horizon_control.py")
runner = _load("_run_leaf_matched_horizon_under_test",
               REPOSITORY / "scripts/run_leaf_matched_horizon.py")
tabulate = _load("_tabulate_leaf_matched_horizon_under_test",
                 REPOSITORY / "scripts/tabulate_leaf_matched_horizon.py")

HORIZON = horizon_tests.HORIZON
MEASURE_FROM_MS = horizon_tests.MEASURE_FROM_MS
L1_BYTES = horizon_tests.L1_BYTES
L2_BYTES = horizon_tests.L2_BYTES
partial_trace = horizon_tests.partial_trace
replay = horizon_tests.replay
_row = horizon_tests._row
_binary = horizon_tests._binary
_Tracer = horizon_tests._Tracer
_LegalityRecorder = horizon_tests._LegalityRecorder
_LinearRanker = horizon_tests._LinearRanker
csv_rows = horizon_tests.csv_rows
SEEDS = (0, 1, 2, 3, 4)
# Columns that differ between two replays of the same arm on the same input,
# and those that name the arm or the run.
VOLATILE = ("seconds", "worker_peak_rss_mib", "worker_pss_mib_end")
NAMING = ("arm", "family", "arm_parameter", "mechanism", "eligibility", "rung")


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _setup(arm, trace, seed=1):
    return leafmatched.arm_setup(arm, trace, HORIZON, seed, learned_ranker=_LinearRanker())


def hooked_replay(trace, arm, setup, eligibility="leaf", seed=1, class_horizon=None):
    """One traced replay with the worker's hooks (attribution, the
    error-location statistics and, given a horizon, the class statistics, nested
    as the worker nests them). Returns (result, statistics, class statistics,
    tracer, counter digest, collector)."""
    collector = AttributionCollector(trace, measure_from_ms=MEASURE_FROM_MS, bytes_per_token=1)
    statistics = DecisionStatistics(trace, HORIZON, MEASURE_FROM_MS)
    classes = None
    inner = setup.override
    if class_horizon is not None:
        classes = ClassStatistics(trace, class_horizon, MEASURE_FROM_MS)
        inner = RecordingOverride(classes, setup.override)
    tracer = _Tracer(RecordingOverride(statistics, inner))
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                          l2_sample_width=4, l2_seed=seed, l2_eligibility=eligibility, l2_arm=arm,
                          l2_request_hook=collector.on_request, l2_removal_hook=collector,
                          l2_override_hook=tracer, **setup.replay_arguments())
    collector.check_against(result)
    return (result, statistics, classes, tracer, runner.rel._counters_digest(result, collector),
            collector)


def childless_trace(test, steps=480, keys=60, seed=3):
    """Single-block requests only: no state has a child, so no arrival and no
    resident ever has a cached child. Blocks of three sizes (fixed per key), so
    an offer can need several rounds."""
    rng = random.Random(seed)
    records = []
    for step in range(steps):
        key = rng.randrange(keys)
        records.append({"timestamp": step * 5000, "input_length": (100, 300, 512)[key % 3],
                        "output_length": 1, "hash_ids": [key]})
    temporary, trace, _ = horizon_tests.build(records)
    test.addCleanup(temporary.cleanup)
    return trace


# --- the grid and the arms -------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_mechanism_and_the_arms_are_the_plans(self):
        self.assertEqual(leafmatched.MECHANISM, "leaf16")
        self.assertEqual((leafmatched.ELIGIBILITY, leafmatched.WIDTH), ("leaf", 16))
        self.assertEqual(horizonctl.MECHANISMS["leaf16"], ("leaf", 16))
        self.assertEqual(RUNG_ARMS, ("label_binary_60", "label_binary_150", "label_binary_300",
                                     "label_binary_600"))
        self.assertEqual(leafmatched.CLASS_STATISTIC_ARMS, RUNG_ARMS + matchedorder.ARMS)
        expected = {(0.0025, 1.0): 60, (0.0025, 4.0): 150, (0.01, 1.0): 150, (0.01, 4.0): 600,
                    (0.02, 1.0): 300, (0.02, 4.0): 600}
        for cell, h in expected.items():
            with self.subTest(cell=cell):
                self.assertEqual(cell_arms(*cell), (f"label_binary_{h}",
                                                    f"evict_binary_{h}_learned",
                                                    f"evict_binary_{h}_recency", "label"))
        self.assertEqual(set(runner.CELLS), set(MATCHED_HORIZON_SECONDS))
        with self.assertRaises(ValueError):
            cell_arms(0.05, 1.0)

    def test_the_tasks(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS)
        self.assertEqual(len(tasks), 240)
        self.assertEqual(len(set(tasks)), 240)
        self.assertEqual(sum(1 for task in tasks if task[3] == ANCHOR_ARM), 60)
        by_cell = defaultdict(set)
        for name, fraction, multiplier, arm, seed in tasks:
            by_cell[(name, fraction, multiplier)].add(arm)
        self.assertEqual(len(by_cell), 12)
        for (_, fraction, multiplier), arms in by_cell.items():
            self.assertEqual(arms, set(cell_arms(fraction, multiplier)))
        costs = [runner.arm_cost(task[3]) for task in tasks]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH, runner.MECHANISM),
                         ("leaf", 16, "leaf16"))

    def test_the_references_are_the_plans(self):
        leaf = {key: source for key, source in runner.REFERENCES.items() if key[0] == "leaf16"}
        self.assertEqual(leaf, {("leaf16", "lru"): "mechanism_control_001",
                                ("leaf16", "learned"): "mechanism_control_001",
                                ("leaf16", "label"): "mechanism_control_001",
                                ("leaf16", "evict_label"): "horizon_control_001"})
        all16 = {key[1]: source for key, source in runner.REFERENCES.items() if key[0] == "all16"}
        self.assertEqual(all16, {
            "lru": "error_location_001", "learned": "error_location_001",
            "label": "error_location_001", "evict_label": "error_location_001",
            "label_binary_60": "horizon_control_001", "label_binary_150": "horizon_fill_001",
            "label_binary_300": "horizon_control_001", "label_binary_600": "horizon_control_001",
            **{arm: "matched_class_order_001" for arm in matchedorder.ARMS}})
        self.assertEqual(runner.REFERENCE_SOURCES["mechanism_control_001"],
                         REPOSITORY / "results/paper/mechanism_control_001/replay_seeds.csv")
        self.assertEqual(runner.REFERENCE_SOURCES["matched_class_order_001"],
                         REPOSITORY / "results/paper/matched_class_order_001/replay_seeds.csv")
        self.assertEqual(runner.REPRODUCTION_REFERENCE, ("leaf16", "label"))
        self.assertEqual(runner.reference_cells("all16", "label_binary_150"),
                         ((0.0025, 4.0), (0.01, 1.0)))
        self.assertEqual(runner.reference_cells("all16", "evict_binary_300_recency"),
                         ((0.02, 1.0),))
        self.assertEqual(runner.reference_cells("leaf16", "label"), runner.CELLS)
        with self.assertRaises(ValueError):
            runner.reference_cells("leaf16", "label_binary_60")

    def test_the_tabulation_uses_the_same_vocabulary(self):
        self.assertEqual(tabulate.HORIZON_READINGS, HORIZON_READINGS)
        self.assertEqual(tabulate.CLASS_READINGS, CLASS_READINGS)
        self.assertEqual(tabulate.SIGN_READINGS, leafmatched.SIGN_READINGS)
        self.assertEqual(tabulate.READINGS, tuple(leafmatched.PREDICTIONS)
                         + ("4_admission_descriptive",))
        self.assertEqual(tabulate.TRACES, runner.TRACES)


class ArmTests(unittest.TestCase):
    """Every arm is its parent's construction, unchanged."""

    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_rungs_are_horizonfills(self):
        for h, arm in zip(MATCHED_HORIZONS_SECONDS, RUNG_ARMS):
            with self.subTest(arm=arm):
                mine, theirs = _setup(arm, self.trace), horizonfill.arm_setup(arm, self.trace)
                for setup in (mine, theirs):
                    self.assertIsInstance(setup.scorer, ExactLabelScorer)
                    self.assertEqual((setup.scorer.target, setup.scorer.horizon_seconds),
                                     ("binary", h))
                    self.assertIsNone(setup.override)
                self.assertEqual(mine.replay_arguments().keys(), theirs.replay_arguments().keys())
                self.assertEqual((mine.arm, mine.l2_policy, mine.sampled_offline),
                                 (theirs.arm, theirs.l2_policy, theirs.sampled_offline))

    def test_the_class_order_arms_are_matchedorders(self):
        for arm in matchedorder.ARMS:
            h, order = matchedorder.HORIZON_OF_ARM[arm], matchedorder.ORDER_OF_ARM[arm]
            with self.subTest(arm=arm):
                mine = _setup(arm, self.trace)
                theirs = matchedorder.arm_setup(arm, self.trace, learned_ranker=_LinearRanker())
                self.assertIs(type(mine.scorer), type(theirs.scorer))
                self.assertIs(type(mine.override), type(theirs.override))
                self.assertIsInstance(mine.override, HybridOverride)
                if order == "learned":
                    self.assertIsInstance(mine.scorer, ReuseClassScorer)
                    reuse = mine.scorer.reuse
                    self.assertIs(mine.override.admission, mine.scorer.within)
                else:
                    self.assertIsInstance(mine.scorer, HybridScorer)
                    reuse = mine.scorer.eviction
                    self.assertIs(mine.override.admission, mine.scorer.admission)
                self.assertIsInstance(mine.override.admission, RawFeatureScorer)
                self.assertEqual((reuse.target, reuse.horizon_seconds), ("binary", h))
                self.assertEqual(mine.l2_policy, theirs.l2_policy)

    def test_the_anchor_is_the_error_location_label_rung(self):
        mine = _setup(ANCHOR_ARM, self.trace)
        theirs = errorloc.arm_setup("label", self.trace, HORIZON, 1)
        for setup in (mine, theirs):
            self.assertIsInstance(setup.scorer, ExactLabelScorer)
            self.assertEqual((setup.scorer.target, setup.scorer.horizon_seconds),
                             ("next_use", HORIZON))
            self.assertIsNone(setup.override)
        self.assertEqual(mine.replay_arguments().keys(), theirs.replay_arguments().keys())

    def test_unknown_arms_and_missing_inputs_are_refused(self):
        for arm in ("label_binary_90", "evict_binary_learned", "evict_label", "lru", "learned"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                _setup(arm, self.trace)
        with self.assertRaises(ValueError):
            leafmatched.arm_setup("evict_binary_60_recency", self.trace, HORIZON, 0)


# --- eligibility is the only difference ------------------------------------------------------------


class EligibilityOnlyTests(unittest.TestCase):
    def test_without_cached_children_leaf_and_all_replay_identically(self):
        trace = childless_trace(self)
        self.assertEqual(sum(len(children) for children in trace.children.values()), 0)
        for arm in cell_arms(0.0025, 1.0):
            with self.subTest(arm=arm):
                h = leafmatched.HORIZON_OF_ARM.get(arm)
                outcomes = {}
                for eligibility in ("all", "leaf"):
                    result, statistics, classes, tracer, digest, _ = hooked_replay(
                        trace, arm, _setup(arm, trace), eligibility=eligibility, class_horizon=h)
                    outcomes[eligibility] = (tracer.decisions, _row(result), statistics.row(),
                                             classes.row() if classes else None, digest)
                self.assertEqual(outcomes["all"], outcomes["leaf"])
                decisions = outcomes["leaf"][0]
                self.assertGreater(len(decisions), 100)
                # Every first round offers the arrival (a leaf), so admission
                # decisions are answered by the arms that have an override.
                self.assertGreater(sum(1 for d in decisions if d[2] == 0), 50)
                if arm in matchedorder.ARMS:
                    self.assertGreater(sum(1 for d in decisions if d[5] != d[6]), 0)

    def test_with_non_leaf_arrivals_the_eligibility_changes_the_decisions(self):
        trace, _ = partial_trace(self)
        for arm in cell_arms(0.0025, 4.0):
            with self.subTest(arm=arm):
                streams = {eligibility: hooked_replay(trace, arm, _setup(arm, trace),
                                                      eligibility=eligibility)[3].decisions
                           for eligibility in ("all", "leaf")}
                self.assertNotEqual(streams["all"], streams["leaf"])

    def test_the_hybrid_override_judges_leaf_arrivals_only(self):
        # The horizon control's legality recorder: every answer is None or a
        # candidate index, and None whenever the arrival is not a candidate of a
        # first round; first rounds with a non-leaf arrival do occur.
        trace, _ = partial_trace(self)
        for arm in matchedorder.cell_arms(0.0025, 4.0):
            with self.subTest(arm=arm):
                setup = _setup(arm, trace)
                recorder = _LegalityRecorder(self, setup.override)
                checked = errorloc.ArmSetup(arm, "learned", scorer=setup.scorer, override=recorder)
                result, _, _, _ = replay(trace, arm, setup=checked, eligibility="leaf",
                                         victim_hook=recorder.on_victim)
                kinds = defaultdict(int)
                for kind, *_ in recorder.records:
                    kinds[kind] += 1
                self.assertEqual(sum(kinds.values()), result.l2_decisions)
                for kind in ("first_round_leaf_arrival", "first_round_arrival_not_a_leaf",
                             "later_round"):
                    self.assertGreater(kinds[kind], 0, kind)
                answered = sum(1 for record in recorder.records if record[4] is not None)
                self.assertEqual(answered, kinds["first_round_leaf_arrival"])


class LeafIdentityTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_with_admission_by_the_exact_h_label_the_recency_arm_is_the_rung_under_leaf16(self):
        for h, rung_arm in zip(MATCHED_HORIZONS_SECONDS, RUNG_ARMS):
            with self.subTest(h=h):
                rung = hooked_replay(self.trace, rung_arm, _setup(rung_arm, self.trace),
                                     class_horizon=h)
                identity = hooked_replay(self.trace, matchedorder.identity_arm(h),
                                         matchedorder.identity_setup(self.trace, h),
                                         class_horizon=h)
                self.assertEqual(identity[3].decisions, rung[3].decisions)
                self.assertEqual(_row(identity[0]), _row(rung[0]))
                self.assertEqual(identity[4], rung[4])                     # counter digest
                self.assertEqual(identity[1].row(), rung[1].row())       # decision digest included
                self.assertEqual(identity[1].row()["decision_sha256"],
                                 rung[1].row()["decision_sha256"])
                self.assertEqual(identity[2].row(), rung[2].row())
                # Not vacuous: leaf arrivals are judged by X, rejections and
                # first rounds without the arrival (non-leaf) and later rounds occur.
                decisions = identity[3].decisions
                admissions = sum(1 for d in decisions if d[2] == 0)
                self.assertGreater(admissions, 0)
                self.assertEqual(identity[3].returned, admissions)
                self.assertGreater(rung[0].l2_rejections, 0)
                self.assertGreater(sum(1 for d in decisions if d[2] < 0), 0)
                self.assertEqual(rung[0].l2_present_unusable_tokens, 0)

    def test_the_recency_arm_is_not_the_rung(self):
        # The ranker's admission makes a difference under leaf16 too.
        for arm in matchedorder.cell_arms(0.0025, 1.0)[1:]:
            mine = hooked_replay(self.trace, arm, _setup(arm, self.trace))[3].decisions
            rung = hooked_replay(self.trace, "label_binary_60",
                                 _setup("label_binary_60", self.trace))[3].decisions
            self.assertNotEqual([d[:4] + d[5:] for d in mine], [d[:4] + d[5:] for d in rung])

    def test_the_label_anchor_is_the_error_location_label_rung_under_leaf16(self):
        mine = hooked_replay(self.trace, ANCHOR_ARM, _setup(ANCHOR_ARM, self.trace))
        theirs = hooked_replay(self.trace, "label",
                               errorloc.arm_setup("label", self.trace, HORIZON, 1))
        self.assertEqual(mine[3].decisions, theirs[3].decisions)
        self.assertEqual(_row(mine[0]), _row(theirs[0]))
        self.assertEqual(mine[4], theirs[4])
        self.assertEqual(mine[1].row(), theirs[1].row())
        self.assertGreater(mine[0].l2_rejections, 0)
        # And the horizon control's own leaf16 label replay.
        hc_result, hc_stats, hc_trace, _ = replay(self.trace, "label")
        self.assertEqual(horizonctl.arm_mechanism("label")[1], "leaf")
        self.assertEqual(hc_trace.decisions, mine[3].decisions)
        self.assertEqual(hc_stats.row(), mine[1].row())


class ClosureAndClassTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_no_block_is_ever_present_but_unusable_under_leaf16(self):
        unusable_under_all = 0
        for arm in cell_arms(0.0025, 4.0):
            with self.subTest(arm=arm):
                result, _, _, _, _, collector = hooked_replay(self.trace, arm,
                                                              _setup(arm, self.trace))
                self.assertEqual((result.l2_present_unusable_tokens,
                                  result.l2_present_unusable_blocks), (0, 0))
                self.assertEqual(collector.orphaned_blocks, 0)
                self.assertGreater(result.l2_decisions, 0)
                under_all = hooked_replay(self.trace, arm, _setup(arm, self.trace),
                                          eligibility="all")[0]
                unusable_under_all += under_all.l2_present_unusable_tokens > 0
        # The check has teeth: under all16 the same arms do leave blocks unusable.
        self.assertGreater(unusable_under_all, 0)

    def test_the_class_statistic_is_zero_in_every_h_star_arm(self):
        for arm in leafmatched.CLASS_STATISTIC_ARMS:
            h = leafmatched.HORIZON_OF_ARM[arm]
            with self.subTest(arm=arm):
                result, statistics, classes, tracer, _, _ = hooked_replay(
                    self.trace, arm, _setup(arm, self.trace), class_horizon=h)
                row = classes.row()
                self.assertEqual(row["class_violations_seen"], 0)
                self.assertEqual(row["class_overridden_outside_admission_seen"], 0)
                self.assertGreater(row["class_resident_evictions_seen"], 0)
                overridden = sum(1 for d in tracer.decisions if d[5] != d[6])
                self.assertEqual(row["class_overridden_seen"], overridden)
                self.assertEqual(statistics.row()["overridden_decisions_seen"], overridden)
                if arm in matchedorder.ARMS:
                    self.assertGreater(overridden, 0)
                else:
                    self.assertEqual(overridden, 0)
                if h == 600.0:
                    self.assertEqual(row["class_m4_count_resident"],
                                     statistics.row()["m4_count_resident"])

    def test_the_class_check_has_teeth_at_the_wrong_horizon(self):
        _, _, classes, _, _, _ = hooked_replay(
            self.trace, "evict_binary_60_learned", _setup("evict_binary_60_learned", self.trace),
            class_horizon=600.0)
        self.assertGreater(classes.row()["class_violations_seen"], 0)


# --- the reading helpers ---------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_shortfall_and_its_010_rule_at_the_boundary(self):
        self.assertEqual(leafmatched.hstar_shortfall(10.0, 9.0, 0.0), 0.1)
        self.assertEqual(leafmatched.hstar_shortfall(10.0, 9.0, 0.0),
                         horizonctl.shortfall(10.0, 9.0, 0.0))
        self.assertTrue(math.isnan(leafmatched.hstar_shortfall(3.0, 1.0, 3.0)))
        suffices, short = HORIZON_READINGS
        self.assertEqual(leafmatched.hstar_reading(0.1), suffices)
        self.assertEqual(leafmatched.hstar_reading(leafmatched.hstar_shortfall(10.0, 9.0, 0.0)),
                         suffices)
        self.assertEqual(leafmatched.hstar_reading(-0.5), suffices)
        self.assertEqual(leafmatched.hstar_reading(0.1000001), short)
        self.assertEqual(leafmatched.hstar_reading(math.nan), short)

    def test_the_recovery_and_its_09_rule_at_the_boundary(self):
        self.assertEqual(leafmatched.class_recovery(29.0, 2.0, 32.0), 0.9)
        self.assertTrue(math.isnan(leafmatched.class_recovery(3.0, 2.0, 2.0)))
        suffices, costs = CLASS_READINGS
        self.assertEqual(CLASS_READINGS, matchedorder.MATCHED_READINGS)
        self.assertEqual(leafmatched.class_reading(0.9), suffices)
        self.assertEqual(leafmatched.class_reading(0.8999999), costs)
        self.assertEqual(leafmatched.class_reading(math.nan), costs)

    def test_the_predictions_at_their_boundaries(self):
        holds = leafmatched.prediction_holds
        self.assertIs(holds("1_horizon_under_leaf_eligibility", 12, 12), True)
        self.assertIs(holds("1_horizon_under_leaf_eligibility", 11, 12), False)
        self.assertIs(holds("2_class_order_under_leaf_eligibility", 10, 12), True)
        self.assertIs(holds("2_class_order_under_leaf_eligibility", 9, 12), False)
        self.assertIs(holds("3_order_within_matched_class", 8, 12), True)
        self.assertIs(holds("3_order_within_matched_class", 7, 12), False)
        # A prediction about 12 trace x cell is not evaluated on another number.
        self.assertIsNone(holds("1_horizon_under_leaf_eligibility", 3, 3))
        with self.assertRaises(ValueError):
            holds("4_admission_descriptive", 1, 12)
        self.assertEqual({reading: outcome for reading, (outcome, _)
                          in leafmatched.PREDICTIONS.items()},
                         {"1_horizon_under_leaf_eligibility": HORIZON_READINGS[0],
                          "2_class_order_under_leaf_eligibility": CLASS_READINGS[0],
                          "3_order_within_matched_class": "consistent_loss"})
        self.assertFalse(leafmatched.predicted("2_class_order_under_leaf_eligibility", 0.0025, 1.0))
        self.assertTrue(leafmatched.predicted("2_class_order_under_leaf_eligibility", 0.0025, 4.0))
        self.assertTrue(leafmatched.predicted("1_horizon_under_leaf_eligibility", 0.0025, 1.0))

    def test_seed_signs_counts_and_the_arrival_share(self):
        self.assertEqual(leafmatched.seed_signs([-0.1, -2.0, -0.3, -0.01, -5.0]),
                         (0, 0, 5, "consistent_loss"))
        self.assertEqual(leafmatched.seed_signs([-0.1, 0.0, -0.3, -0.01, -5.0])[3], "mixed")
        self.assertEqual(leafmatched.outcome_counts(["mixed", "consistent_loss"],
                                                    leafmatched.SIGN_READINGS),
                         {"consistent_gain": 0, "consistent_loss": 1, "mixed": 1})
        with self.assertRaises(ValueError):
            leafmatched.outcome_counts(["gain"], leafmatched.SIGN_READINGS)
        self.assertEqual(leafmatched.arrival_candidate_share(25, 100), 0.25)
        self.assertTrue(math.isnan(leafmatched.arrival_candidate_share(0, 0)))
        self.assertEqual(leafmatched.row_arrival_candidate_share(
            {"stat_decisions_admission": "3", "stat_decisions": "4"}), 0.75)


# --- the runner's derivations on synthetic rows ----------------------------------------------------

CELL_A = ("conversation_trace", 0.0025, 1.0)     # h* = 60, reading 2 not predicted
CELL_B = ("toolagent_trace", 0.01, 1.0)          # h* = 150
CELL_C = ("conversation_trace", 0.02, 4.0)       # h* = 600
IDENTIFIERS = {"l1_capacity_bytes": 1, "l2_capacity_bytes": 4, "requested_tokens": 1000,
               "l1_avoided_tokens": 50, "absent_compulsory_tokens": 77}
# U in tokens over 1000 requested tokens (points = tokens / 10), per mechanism.
LEAF_REFERENCES = {"label": 600, "lru": 100, "learned": 200, "evict_label": 500}
ALL16_REFERENCES = {"label": 700, "lru": 100, "learned": 300, "evict_label": 600}
# (label_binary_h*, learned order, recency) per cell and mechanism.
LEAF_ARMS = {CELL_A: (560, 440, 470), CELL_B: (500, 485, 485), CELL_C: (600, 490, 500)}
ALL16_ARMS = {CELL_A: (650, 590, 560), CELL_B: (690, 560, 600), CELL_C: (700, 595, 590)}
RECENCY_OFFSETS = {CELL_B: (5, -5, 0, 5, -5)}
SHARES = {"leaf16": (20, 25), "all16": (95, 96)}       # admission decisions of 100


def _synthetic_row(cell, arm, mechanism, tokens, seed, admission=None):
    row = {"trace": cell[0], "l1_fraction": cell[1], "l2_multiplier": cell[2],
           "cell": runner.rdp.cell_label(*cell[1:]), "arm": arm, "seed": seed,
           "mechanism": mechanism, "variant": "main", "extra_avoided_tokens": tokens,
           "avoided_prefill_tokens": tokens + 50, "stat_decisions": 100,
           **IDENTIFIERS}
    if admission is not None:
        row["stat_decisions_admission"] = admission
    return row


def synthetic_published():
    published = {}
    for cell, arms in ALL16_ARMS.items():
        rung, learned, recency, _ = cell_arms(*cell[1:])
        for seed in SEEDS:
            offset = RECENCY_OFFSETS.get(cell, (0,) * 5)[seed]
            for arm, tokens in LEAF_REFERENCES.items():
                published[("leaf16", arm) + cell + (seed,)] = dict(
                    _synthetic_row(cell, arm, "leaf16", tokens, seed), source="s")
            for arm, tokens in ALL16_REFERENCES.items():
                published[("all16", arm) + cell + (seed,)] = dict(
                    _synthetic_row(cell, arm, "all16", tokens, seed), source="s")
            for arm, tokens, admission in ((rung, arms[0], SHARES["all16"][0]),
                                           (learned, arms[1], None),
                                           (recency, arms[2] + offset, SHARES["all16"][1])):
                published[("all16", arm) + cell + (seed,)] = dict(
                    _synthetic_row(cell, arm, "all16", tokens, seed,
                                   admission if admission is not None else 50), source="s")
    return published


def synthetic_rows():
    rows = []
    for cell, arms in LEAF_ARMS.items():
        rung, learned, recency, anchor = cell_arms(*cell[1:])
        for seed in SEEDS:
            offset = RECENCY_OFFSETS.get(cell, (0,) * 5)[seed]
            for arm, tokens, admission in ((rung, arms[0], SHARES["leaf16"][0]),
                                           (learned, arms[1], 30),
                                           (recency, arms[2] + offset, SHARES["leaf16"][1]),
                                           (anchor, LEAF_REFERENCES["label"], 40)):
                row = {metric: 0.0 for metric in runner.REPLAY_METRICS}
                row.update(_synthetic_row(cell, arm, "leaf16", tokens, seed, admission),
                           extra_points=tokens / 10.0)
                if arm in leafmatched.CLASS_STATISTIC_ARMS:
                    row.update({metric: 0 for metric in runner.CLASS_METRICS})
                rows.append(row)
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.published = synthetic_published()
        self.tables = runner.readings_tables(self.rows, self.published)

    def _by(self, name):
        return {runner._cell_key(entry): entry for entry in self.tables[name]}

    def test_reading_1(self):
        by = self._by("horizon")
        suffices, short = HORIZON_READINGS
        expected = {CELL_A: (60.0, 4 / 50, suffices, 5 / 60, suffices),
                    CELL_B: (150.0, 10 / 50, short, 1 / 60, suffices),
                    CELL_C: (600.0, 0.0, suffices, 0.0, suffices)}
        for cell, (h, s, reading, s16, reading16) in expected.items():
            with self.subTest(cell=cell):
                entry = by[cell]
                self.assertEqual((entry["h_star"], entry["rung_arm"]), (h, f"label_binary_{h:g}"))
                self.assertAlmostEqual(entry["S_hstar"], s, places=12)
                self.assertEqual(entry["reading"], reading)
                self.assertAlmostEqual(entry["all16_S_hstar"], s16, places=12)
                self.assertEqual(entry["all16_reading"], reading16)
                self.assertEqual(entry["predicted_reading"], suffices)
                self.assertAlmostEqual(entry["U_label_points_mean"], 60.0)
                self.assertAlmostEqual(entry["all16_U_label_points_mean"], 70.0)
        self.assertEqual(by[CELL_A]["label_minus_rung_seed_signs"], "+++++")
        self.assertEqual(by[CELL_B]["all16_rung_source"], "horizon_fill_001")

    def test_reading_2(self):
        by = self._by("class_order")
        suffices, costs = CLASS_READINGS
        expected = {CELL_A: (24 / 30, 27 / 30, costs, False, 290 / 300, 260 / 300, suffices),
                    CELL_B: (28.5 / 30, 28.5 / 30, suffices, True, 260 / 300, 300 / 300, costs),
                    CELL_C: (29 / 30, 30 / 30, suffices, True, 295 / 300, 290 / 300, suffices)}
        for cell, (r, r_recency, reading, covered, r16, r16_recency, reading16) in expected.items():
            with self.subTest(cell=cell):
                entry = by[cell]
                self.assertAlmostEqual(entry["R_matched_learned"], r, places=12)
                self.assertAlmostEqual(entry["R_matched_recency"], r_recency, places=12)
                self.assertEqual(entry["reading"], reading)
                self.assertIs(entry["predicted"], covered)
                self.assertEqual(entry["predicted_reading"], suffices if covered else "")
                self.assertAlmostEqual(entry["all16_R_matched_learned"], r16, places=12)
                self.assertAlmostEqual(entry["all16_R_matched_recency"], r16_recency, places=12)
                self.assertEqual(entry["all16_reading"], reading16)

    def test_readings_3_and_4(self):
        order = self._by("order")
        self.assertEqual({cell: entry["learned_minus_recency_reading"]
                          for cell, entry in order.items()},
                         {CELL_A: "consistent_loss", CELL_B: "mixed", CELL_C: "consistent_loss"})
        self.assertEqual(order[CELL_B]["learned_minus_recency_seed_signs"], "-+0-+")
        self.assertAlmostEqual(order[CELL_A]["learned_minus_recency_points_mean"], -3.0)
        self.assertEqual(order[CELL_A]["all16_learned_minus_recency_reading"], "consistent_gain")
        self.assertEqual(order[CELL_B]["all16_learned_minus_recency_seed_signs"], "-----")
        self.assertEqual(order[CELL_A]["predicted_reading"], "consistent_loss")
        admission = self._by("admission")
        self.assertAlmostEqual(admission[CELL_C]["label_binary_minus_recency_points_mean"], 10.0)
        self.assertEqual(admission[CELL_B]["label_binary_minus_recency_seed_signs"], "+++++")
        self.assertAlmostEqual(admission[CELL_A]["all16_label_binary_minus_recency_points_mean"],
                               9.0)
        for cell in LEAF_ARMS:
            entry = admission[cell]
            self.assertAlmostEqual(entry["arrival_candidate_share_label_binary_hstar_mean"], 0.20)
            self.assertAlmostEqual(entry["arrival_candidate_share_matched_recency_mean"], 0.25)
            self.assertAlmostEqual(entry["all16_arrival_candidate_share_label_binary_hstar_mean"],
                                   0.95)
            self.assertAlmostEqual(entry["all16_arrival_candidate_share_matched_recency_min"], 0.96)

    def test_the_seed_table(self):
        seeds = self.tables["readings_seeds"]
        self.assertEqual(len(seeds), 3 * 5 * 2)
        first = next(entry for entry in seeds if runner._cell_key(entry) == CELL_B
                     and entry["seed"] == 1 and entry["mechanism"] == "leaf16")
        self.assertEqual(first["learned_minus_recency_tokens"], 5)
        self.assertEqual(first["label_binary_minus_recency_tokens"], 20)
        self.assertAlmostEqual(first["U_matched_recency_points"], 48.0)
        self.assertAlmostEqual(first["S_seed"], 0.2)
        self.assertEqual(first["arrival_candidate_share_matched_recency"], 0.25)
        other = next(entry for entry in seeds if runner._cell_key(entry) == CELL_B
                     and entry["seed"] == 1 and entry["mechanism"] == "all16")
        self.assertAlmostEqual(other["U_label_binary_hstar_points"], 69.0)

    def test_the_counts(self):
        summary = runner.reading_summary(self.tables)
        by = {entry["reading"]: entry for entry in summary}
        self.assertEqual(len(summary), 4)
        one = by["1_horizon_under_leaf_eligibility"]
        self.assertEqual((one["cells"], one["count"], one["required"], one["of"],
                          one["all16_count"]), (3, 2, 12, 12, 3))
        self.assertEqual(one["prediction_holds"], "")         # 3 trace x cell, not the 12
        two = by["2_class_order_under_leaf_eligibility"]
        self.assertEqual((two["count"], two["required"], two["all16_count"],
                          two[CLASS_READINGS[1]]), (2, 10, 2, 1))
        three = by["3_order_within_matched_class"]
        self.assertEqual((three["counted"], three["count"], three["mixed"],
                          three["all16_consistent_gain"]), ("consistent_loss", 2, 1, 2))
        four = by["4_admission_descriptive"]
        self.assertEqual((four["counted"], four["prediction_holds"], four["consistent_gain"]),
                         ("", "", 3))
        # Over the 12 trace x cell the prediction is evaluated.
        tables = {name: [dict(entry) for entry in rows for _ in range(4)]
                  for name, rows in self.tables.items()}
        by = {entry["reading"]: entry for entry in runner.reading_summary(tables)}
        self.assertIs(by["1_horizon_under_leaf_eligibility"]["prediction_holds"], False)
        self.assertIs(by["2_class_order_under_leaf_eligibility"]["prediction_holds"], False)
        self.assertIs(by["3_order_within_matched_class"]["prediction_holds"], True)

    def test_aggregation_and_reference_rows(self):
        summary = runner.aggregate_replays(self.rows)
        self.assertEqual(len(summary), 12)
        entry = next(e for e in summary if runner._cell_key(e) == CELL_A
                     and e["arm"] == "evict_binary_60_recency")
        self.assertEqual((entry["arm_parameter"], entry["order"], entry["family"],
                          entry["h_star"], entry["mechanism"], entry["eligibility"]),
                         (60.0, "recency", "class_order", 60.0, "leaf16", "leaf"))
        self.assertAlmostEqual(entry["arrival_candidate_share_mean"], 0.25)
        self.assertIn("class_violations_seen_mean", entry)
        anchor = next(e for e in summary if e["arm"] == ANCHOR_ARM)
        self.assertNotIn("class_violations_seen_mean", anchor)
        self.assertEqual((anchor["family"], anchor["arm_parameter"]), ("reproduction_anchor", ""))
        references = runner.reference_rows(self.published, self.rows)
        # Four leaf16 and four all16 references, the rung and two matched arms.
        self.assertEqual(len(references), 3 * 5 * 11)
        self.assertFalse(any("counters" in entry for entry in references))

    def test_the_derivation_needs_every_replay(self):
        rows = [row for row in self.rows if not (row["arm"] == "evict_binary_150_learned"
                                                 and row["seed"] == 2)]
        with self.assertRaises(KeyError):
            runner.readings_tables(rows, self.published)


class PublishedTablesTests(unittest.TestCase):
    """Reads only: the published reference rows and tables load as the plan
    names them, and the all16 values printed beside every leaf16 value are the
    published ones."""

    @classmethod
    def setUpClass(cls):
        cls.published, cls.problems = runner.load_references()

    def test_the_published_references_load_as_the_plan_names_them(self):
        self.assertEqual(self.problems, [])
        # Eight references at every trace x cell x seed, the rung and two
        # matched arms at their own cells.
        self.assertEqual(len(self.published), 8 * 60 + 3 * 60)
        entry = self.published[("leaf16", "label", "toolagent_trace", 0.02, 1.0, 4)]
        self.assertEqual(entry["source"], "mechanism_control_001")
        self.assertEqual(self.published[("leaf16", "evict_label", "conversation_trace", 0.01, 4.0,
                                         0)]["source"], "horizon_control_001")
        self.assertEqual(self.published[("all16", "label_binary_150", "toolagent_trace", 0.01,
                                         1.0, 3)]["source"], "horizon_fill_001")
        self.assertEqual(self.published[("all16", "evict_binary_60_recency", "conversation_trace",
                                         0.0025, 1.0, 2)]["source"], "matched_class_order_001")
        for key, entry in self.published.items():
            self.assertEqual(set(runner.REFERENCE_IDENTIFIERS) - set(entry), set())
            if key[1] in RUNG_ARMS + matchedorder.ARMS:
                self.assertEqual(set(runner.SHARE_COLUMNS) - set(entry), set())

    def test_the_published_leaf16_label_rows_carry_no_digest_but_every_counter(self):
        # The plan's counter digest is not published for these rows: the
        # reproduction compares every counter column they carry instead.
        anchors = {key: entry for key, entry in self.published.items()
                   if key[:2] == runner.REPRODUCTION_REFERENCE}
        self.assertEqual(len(anchors), 60)
        for entry in anchors.values():
            self.assertNotIn("counters_sha256", entry)
            self.assertNotIn("decision_sha256", entry)
            self.assertEqual(set(runner.REQUIRED_COUNTER_COLUMNS) - set(entry["counters"]), set())
            self.assertEqual(len(entry["counters"]), 87)
            self.assertFalse(set(entry["counters"]) & runner.REPRODUCTION_SKIP)

    def test_the_identifiers_do_not_depend_on_the_mechanism(self):
        # Every reference row of a trace x cell x seed, leaf16 and all16, has
        # the same identifiers: the identifier check can be met by every replay.
        groups = defaultdict(set)
        for key, entry in self.published.items():
            groups[key[2:]].add(tuple(entry[field] for field in runner.REFERENCE_IDENTIFIERS))
        self.assertEqual(len(groups), 60)
        self.assertTrue(all(len(values) == 1 for values in groups.values()))

    def test_the_all16_values_are_the_published_ones(self):
        published_all16, problems = runner.load_published_all16()
        self.assertEqual(problems, [])
        self.assertEqual(len(published_all16), 12)
        values = runner.all16_values(self.published)
        self.assertEqual(runner.check_all16(values, published_all16), [])
        # Spot values of the published tables (matched-class-order plan's table).
        conversation = values[("conversation_trace", 0.0025, 1.0)]
        self.assertAlmostEqual(conversation["S_hstar"], 0.028, places=3)
        self.assertAlmostEqual(conversation["R_matched_learned"], 0.7343162086645255, places=12)
        # A changed published value is caught, a float within the tolerance is not.
        cell = ("toolagent_trace", 0.01, 4.0)
        changed = {key: dict(entry) for key, entry in published_all16.items()}
        changed[cell]["R_matched_recency"] = repr(float(changed[cell]["R_matched_recency"]) + 1e-9)
        changed[cell]["learned_minus_recency_reading"] = "mixed"
        problems = runner.check_all16(values, changed)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("R_matched_recency" in line for line in problems))
        nearly = {key: dict(entry) for key, entry in published_all16.items()}
        nearly[cell]["S_hstar"] = repr(float(nearly[cell]["S_hstar"]) + 1e-15)
        self.assertEqual(runner.check_all16(values, nearly), [])


# --- the runner's checks ---------------------------------------------------------------------------


def _anchor_row(cell=(0.01, 4.0), seed=0, **changes):
    row = {"trace": "conversation_trace", "l1_fraction": cell[0], "l2_multiplier": cell[1],
           "cell": runner.rdp.cell_label(*cell), "seed": seed, "arm": ANCHOR_ARM,
           "avoided_prefill_tokens": 1000 + seed, "counters_sha256": f"c{seed}",
           "decision_sha256": f"d{seed}",
           **{column: 7 for column in runner.REQUIRED_COUNTER_COLUMNS
              if column != "avoided_prefill_tokens"},
           "extra_points": 0.5, "absent_rejected_share_of_decision_absent": math.nan}
    row.update(changes)
    return row


def _anchor_published(rows, digests=False):
    published = {}
    for row in rows:
        counters = {column: str(row[column]) for column in runner.REQUIRED_COUNTER_COLUMNS}
        counters.update(extra_points="0.5", absent_rejected_share_of_decision_absent="nan")
        entry = {"source": "mechanism_control_001",
                 "avoided_prefill_tokens": row["avoided_prefill_tokens"], "counters": counters}
        if digests:
            entry.update(counters_sha256=row["counters_sha256"],
                         decision_sha256=row["decision_sha256"])
        published[("leaf16", "label", row["trace"], row["l1_fraction"], row["l2_multiplier"],
                   row["seed"])] = entry
    return published


def _check_row(arm="evict_binary_60_learned", cell=(0.0025, 1.0), **changes):
    """A replay row as the mechanism, statistics and class checks read it."""
    h = MATCHED_HORIZON_SECONDS[cell]
    row = {"trace": "conversation_trace", "l1_fraction": cell[0], "l2_multiplier": cell[1],
           "cell": runner.rdp.cell_label(*cell), "seed": 0, "arm": arm, "variant": "main",
           "mechanism": "leaf16", "eligibility": "leaf", "width": 16, "h_star": h,
           "arm_parameter": leafmatched.HORIZON_OF_ARM.get(arm, ""),
           "l2_decisions": 10, "l2_rejections": 3, "l2_evictions": 7,
           "l2_present_unusable_tokens": 0, "l2_present_unusable_blocks": 0,
           "stat_decisions_seen": 10, "stat_rejections_seen": 3, "stat_evictions_seen": 7,
           "overridden_decisions_seen": 4 if arm in matchedorder.ARMS else 0,
           "stat_decisions": 6, "overridden_decisions": 2 if arm in matchedorder.ARMS else 0,
           "overridden_decisions_resident": 0, "m4_count_resident": 0, "m1": 1.0}
    for suffix in ("", "_admission", "_resident"):
        row.update({f"stat_decisions{suffix}": 6 if not suffix else 3, f"m2{suffix}": 1.0,
                    f"m3{suffix}": 0.0, f"m4_count{suffix}": 0,
                    f"m4_victim_at_horizon{suffix}": 0})
    if arm in leafmatched.CLASS_STATISTIC_ARMS:
        row.update({"class_horizon_seconds": h, "class_decisions_seen": 10,
                    "class_rejections_seen": 3, "class_resident_evictions_seen": 7,
                    "class_violations_seen": 0,
                    "class_overridden_seen": row["overridden_decisions_seen"],
                    "class_overridden_outside_admission_seen": 0, "class_decisions": 6,
                    "class_rejections": 2, "class_resident_evictions": 4, "class_violations": 0,
                    "class_overridden": row["overridden_decisions"],
                    "class_overridden_outside_admission": 0, "class_m4_count_resident": 0})
    row.update(changes)
    return row


class RunnerCheckTests(unittest.TestCase):
    def test_reproduction_on_every_published_counter(self):
        rows = [_anchor_row(seed=seed) for seed in SEEDS]
        rows.append(_check_row())                       # not an anchor: ignored
        published = _anchor_published(rows[:-1])
        report, table = runner.check_reproduction(rows, published, 5)
        self.assertEqual((report["matched"], report["replays"], report["mismatched"],
                          report["missing"], report["counter_digests_compared"],
                          report["decision_digests_compared"],
                          report["counter_columns_compared"]),
                         (5, 5, 0, 0, 0, 0, [len(runner.REQUIRED_COUNTER_COLUMNS) + 2]))
        self.assertTrue(runner.reproduction_passes(report))
        self.assertTrue(all(entry["reproduces"] for entry in table))
        self.assertEqual(table[0]["same_counters_sha256"], "")
        self.assertEqual(table[0]["published_counters_sha256"], "")
        self.assertEqual(table[0]["counters_sha256"], "c0")
        self.assertNotIn("reproduces_reference", rows[-1])
        for column, value in (("avoided_prefill_tokens", 3), ("l2_evictions", 8),
                              ("extra_points", 0.5000001),
                              ("absent_rejected_share_of_decision_absent", 0.0)):
            with self.subTest(column=column):
                changed = [dict(row) for row in rows]
                changed[2][column] = value
                report, table = runner.check_reproduction(changed, published, 5)
                self.assertEqual((report["matched"], report["mismatched"]), (4, 1))
                self.assertFalse(runner.reproduction_passes(report))
                self.assertIn(column, report["mismatches"][0])
                self.assertIs(changed[2]["reproduces_reference"], False)
        # A counter missing from the replay row cannot be equal.
        changed = [dict(row) for row in rows]
        del changed[0]["l1_evictions"]
        report, table = runner.check_reproduction(changed, published, 5)
        self.assertEqual(report["mismatched"], 1)
        self.assertEqual(table[0]["counter_columns_differing"], "l1_evictions")
        # Without a published digest a required counter must be published.
        bare = {key: dict(entry, counters={k: v for k, v in entry["counters"].items()
                                           if k != "l2_rejections"})
                for key, entry in published.items()}
        report, table = runner.check_reproduction(rows, bare, 5)
        self.assertEqual(report["mismatched"], 5)
        self.assertEqual(table[0]["required_counter_columns_unpublished"], "l2_rejections")
        # With published digests they are compared too, and then a required
        # counter may be unpublished.
        digested = _anchor_published(rows[:-1], digests=True)
        report, _ = runner.check_reproduction(rows, digested, 5)
        self.assertEqual((report["matched"], report["counter_digests_compared"],
                          report["decision_digests_compared"]), (5, 5, 5))
        for key in digested:
            digested[key]["counters"].pop("l2_rejections")
        self.assertEqual(runner.check_reproduction(rows, digested, 5)[0]["matched"], 5)
        changed = [dict(row) for row in rows]
        changed[1]["decision_sha256"] = "x"
        report, _ = runner.check_reproduction(changed, digested, 5)
        self.assertEqual(report["mismatched"], 1)
        self.assertIn("decision_sha256", report["mismatches"][0])
        # A missing published row and a wrong expectation fail.
        del published[next(iter(published))]
        report, _ = runner.check_reproduction(rows, published, 5)
        self.assertEqual((report["missing"], report["matched"]), (1, 4))
        self.assertFalse(runner.reproduction_passes(report))
        report, _ = runner.check_reproduction(rows, _anchor_published(rows[:-1]), 6)
        self.assertFalse(runner.reproduction_passes(report))
        report, _ = runner.check_reproduction([], {}, 0)
        self.assertFalse(runner.reproduction_passes(report))

    def test_identifiers_against_every_reference_of_the_cell(self):
        published = synthetic_published()
        rows = synthetic_rows()
        compared, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])
        self.assertEqual(compared, len(rows) * 11)
        rows[0]["absent_compulsory_tokens"] = 78
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 11)
        self.assertTrue(all("absent_compulsory_tokens 78 != 77" in line for line in problems))
        del published[("leaf16", "evict_label") + CELL_A + (1,)]
        _, problems = runner.check_identifiers(rows, published)
        self.assertTrue(any("no published evict_label/leaf16" in line for line in problems))

    def test_mechanism(self):
        rows = [_check_row(arm, cell) for cell in MATCHED_HORIZON_SECONDS
                for arm in cell_arms(*cell)]
        self.assertEqual(runner.check_mechanism(rows), [])
        cases = ((_check_row(eligibility="all"), "leaf16/all/16"),
                 (_check_row(width=64), "leaf16/leaf/64"),
                 (_check_row(mechanism="all16"), "all16/leaf/16"),
                 (_check_row("evict_binary_150_learned"), "not an arm of its cell"),
                 (_check_row(h_star=150.0), "the cell's h* is 60 s"),
                 (_check_row("label_binary_60", arm_parameter=600.0), "the cell's h* is 60 s"))
        for row, text in cases:
            with self.subTest(case=text):
                problems = runner.check_mechanism([row])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])

    def test_statistics_and_class(self):
        rows = [_check_row(arm, cell) for cell in MATCHED_HORIZON_SECONDS
                for arm in cell_arms(*cell)]
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_class(rows), [])
        keep = "overridden decisions for an arm that must keep the store's choice"
        for arm in ("label_binary_60", ANCHOR_ARM):
            problems = runner.check_statistics([_check_row(arm, overridden_decisions_seen=1)])
            self.assertTrue(any(keep in line for line in problems), problems)
        problems = runner.check_statistics([_check_row("label_binary_150", (0.01, 1.0),
                                                       overridden_decisions_seen=1)])
        self.assertEqual(len(problems), 1)
        self.assertIn(keep, problems[0])
        # The label anchor is held to the label identities.
        self.assertTrue(runner.check_statistics([_check_row(ANCHOR_ARM, m3=0.5)]))
        self.assertEqual(runner.check_statistics([_check_row(m3=0.5)]), [])
        cases = ((_check_row(class_violations_seen=1), "1 resident evictions discard"),
                 (_check_row("label_binary_60", class_violations_seen=2), "2 resident evictions"),
                 (_check_row(class_overridden_outside_admission_seen=1),
                  "the arrival is not a candidate"),
                 (_check_row(class_horizon_seconds=600.0), "the cell's h* is 60 s"),
                 (_check_row("evict_binary_600_recency", (0.01, 4.0), class_m4_count_resident=1),
                  "class_m4_count_resident 1 != m4_count_resident 0 at 600 s"))
        for row, text in cases:
            with self.subTest(case=text):
                problems = runner.check_class([row])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])
        bare = _check_row()
        del bare["class_violations"]
        self.assertEqual(runner.check_class([bare]),
                         ["conversation_trace/l1=0.0025,l2x1/evict_binary_60_learned/s0: "
                          "no class statistics"])
        self.assertEqual(runner.check_class([_check_row(ANCHOR_ARM)]), [])

    def test_leaf_closure(self):
        problems = runner.check_leaf_closure([_check_row(), _check_row(
            "label_binary_60", l2_present_unusable_tokens=512)])
        self.assertEqual(len(problems), 1)
        self.assertIn("label_binary_60", problems[0])

    def test_reference_loading_keeps_each_reference_to_its_source(self):
        header = ("trace,l1_fraction,l2_multiplier,cell,eligibility,width,mechanism,{},seed,"
                  "{}avoided_prefill_tokens,extra_avoided_tokens,l1_capacity_bytes,"
                  "l2_capacity_bytes,requested_tokens,l1_avoided_tokens,l2_evictions\n")
        line = "conversation_trace,0.01,4.0,c,{},{},{},{},0,{}10,5,1,4,100,5,9\n"
        with tempfile.TemporaryDirectory() as directory:
            paths = {source: Path(directory) / f"{source}.csv"
                     for source in runner.REFERENCE_SOURCES}
            paths["mechanism_control_001"].write_text(
                header.format("rung", "")
                + line.format("leaf", 16, "leaf16", "label", "")
                + line.format("leaf", 16, "leaf16", "label", "")          # duplicate
                + line.format("all", 16, "all16", "lru", "")               # wrong source
                + line.format("leaf", 64, "leaf64", "learned", "")         # not a reference
                + line.format("all", 16, "leaf16", "lru", ""))             # wrong eligibility
            paths["horizon_control_001"].write_text(
                header.format("arm", "variant,")
                + line.format("leaf", 16, "leaf16", "evict_label", "main,")
                + line.format("leaf", 16, "leaf16", "adm_label", "main,")  # not a reference
                + line.format("all", 16, "all16", "label_binary_600", "nostats,"))   # not main
            for source in ("horizon_fill_001", "error_location_001", "matched_class_order_001"):
                paths[source].write_text(header.format("arm", "variant,")
                                         + line.format("all", 16, "all16", "lru", "main,"))
            published, problems = runner.load_references(paths)
        cell = ("conversation_trace", 0.01, 4.0, 0)
        self.assertEqual(published[("leaf16", "label") + cell]["counters"],
                         {"avoided_prefill_tokens": "10", "extra_avoided_tokens": "5",
                          "l1_capacity_bytes": "1", "l2_capacity_bytes": "4",
                          "requested_tokens": "100", "l1_avoided_tokens": "5",
                          "l2_evictions": "9"})
        self.assertEqual(published[("leaf16", "evict_label") + cell]["source"],
                         "horizon_control_001")
        self.assertNotIn("counters", published[("leaf16", "evict_label") + cell])
        self.assertEqual(published[("all16", "lru") + cell]["source"], "error_location_001")
        self.assertNotIn(("all16", "label_binary_600") + cell, published)
        self.assertTrue(any(line.startswith("duplicate") for line in problems))
        self.assertTrue(any("has all/16" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))

    def test_argument_refusals_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh, paper = Path(directory) / "new", Path(directory) / "paper"
            base = ["a.jsonl", "b.jsonl", "--output-dir", str(fresh), "--paper-dir", str(paper)]
            cases = {"positive": base + ["--workers", "0"],
                     "hard cap of 12": base + ["--workers", "13"],
                     "expected 2 trace files": base[1:],
                     "must differ": ["a.jsonl", "b.jsonl", "--output-dir", str(fresh),
                                     "--paper-dir", str(fresh)],
                     "output directory": ["a.jsonl", "b.jsonl", "--output-dir", directory,
                                          "--paper-dir", str(paper)],
                     "paper directory": ["a.jsonl", "b.jsonl", "--output-dir", str(fresh),
                                         "--paper-dir", directory]}
            for text, argv in cases.items():
                with self.subTest(case=text), self.assertRaises(SystemExit) as caught:
                    runner.validate_arguments(runner.parse_args(argv))
                self.assertIn(text, str(caught.exception))
            runner.validate_arguments(runner.parse_args(base + ["--workers", "12"]))
        args = runner.parse_args(["a.jsonl", "b.jsonl", "--output-dir", "x"])
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.paper_dir, REPOSITORY / "results/paper/leaf_matched_horizon_001")

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output, paper = Path(directory) / "run", Path(directory) / "paper"
            argv = ["run_leaf_matched_horizon.py", "a.jsonl", "b.jsonl", "--output-dir",
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
        for name in ("scripts/run_leaf_matched_horizon.py", "scripts/run_matched_class_order.py",
                     "scripts/run_horizon_control.py", "scripts/run_error_location.py",
                     "scripts/run_mechanism_control.py", "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/leafmatched.py",
                     "src/persistent_kv_admission/matchedorder.py",
                     "src/persistent_kv_admission/horizonfill.py",
                     "src/persistent_kv_admission/errorloc.py"):
            self.assertIn(name, sources)

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "readings_seeds", "horizon",
                     "class_order", "order", "admission", "readings", "reproduction"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        self.assertIn("`run_config.json`", runner.README_TEXT)
        self.assertIn("carry no counter digest", runner.README_TEXT)


# --- the replay worker -----------------------------------------------------------------------------


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker` on a constructed trace, with the module state the
    runner's main would set, against the parents' own workers."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        name = self.trace.name
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON},
                      rankers={name: _LinearRanker()})
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.mco.SHARED, shared),
                   mock.patch.dict(runner.hc.SHARED, shared),
                   mock.patch.dict(runner.rmc.SHARED, shared),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def _compare(self, mine, theirs, extra_mine, extra_theirs=()):
        """Every column the two rows share is equal (timing, memory and the
        arm's names aside); each has only the named columns of its own."""
        differing = [key for key in theirs if key in mine and key not in VOLATILE + NAMING
                     and not _same(mine[key], theirs[key])]
        self.assertEqual(differing, [])
        self.assertEqual(set(theirs) - set(mine) - set(NAMING), set(extra_theirs))
        self.assertEqual(set(mine) - set(theirs), set(extra_mine))

    def test_the_anchor_is_the_parents_leaf16_label_replay(self):
        for fraction, multiplier in ((0.0025, 4.0), (0.02, 1.0)):
            with self.subTest(cell=(fraction, multiplier)):
                mine = runner._replay_worker((self.name, fraction, multiplier, ANCHOR_ARM, 1))
                horizon = runner.hc._replay_worker((self.name, fraction, multiplier, "label", 1,
                                                    "main"))
                self.assertEqual((horizon["mechanism"], horizon["eligibility"]), ("leaf16", "leaf"))
                self.assertEqual((mine["counters_sha256"], mine["decision_sha256"]),
                                 (horizon["counters_sha256"], horizon["decision_sha256"]))
                self._compare(mine, horizon, {"h_star", "order"})
                mechanism = runner.rmc._replay_worker((self.name, fraction, multiplier, "leaf", 16,
                                                       "label", 1))
                differing = [key for key in mechanism if key not in VOLATILE + NAMING
                             and not _same(mine[key], mechanism[key])]
                self.assertEqual(differing, [])
                self.assertEqual(set(mechanism) - set(mine), {"rung"})
                self.assertGreater(mine["l2_decisions"], 0)

    def test_with_all_eligibility_the_worker_is_the_parents_digest_for_digest(self):
        # The only change from the parents is the store's eligibility: set back
        # to "all", the worker reproduces the matched class-order worker and the
        # horizon control's rungs digest for digest.
        with mock.patch.object(runner, "ELIGIBILITY", "all"):
            for fraction, multiplier in ((0.0025, 1.0), (0.02, 1.0), (0.01, 4.0)):
                rung, learned, recency, _ = cell_arms(fraction, multiplier)
                for arm in (learned, recency):
                    with self.subTest(arm=arm):
                        mine = runner._replay_worker((self.name, fraction, multiplier, arm, 2))
                        theirs = runner.mco._replay_worker((self.name, fraction, multiplier, arm,
                                                            2))
                        self.assertEqual(theirs["eligibility"], "all")
                        self._compare(mine, theirs, {"h_star"}, {"role", "published_arm"})
                        self.assertEqual(mine["counters_sha256"], theirs["counters_sha256"])
                        self.assertEqual(mine["decision_sha256"], theirs["decision_sha256"])
                with self.subTest(arm=rung):
                    mine = runner._replay_worker((self.name, fraction, multiplier, rung, 2))
                    theirs = runner.hc._replay_worker((self.name, fraction, multiplier, rung, 2,
                                                       "main"))
                    self._compare(mine, theirs, {"h_star", "order"}
                                  | set(matchedorder.CLASS_COLUMNS))
        # Under leaf16 the same tasks differ.
        mine = runner._replay_worker((self.name, 0.0025, 1.0, "evict_binary_60_learned", 2))
        theirs = runner.mco._replay_worker((self.name, 0.0025, 1.0, "evict_binary_60_learned", 2))
        self.assertNotEqual(mine["decision_sha256"], theirs["decision_sha256"])
        self.assertEqual(mine["eligibility"], "leaf")

    def test_worker_rows_pass_the_checks(self):
        rows = [runner._replay_worker((self.name, fraction, multiplier, arm, 0))
                for fraction, multiplier in runner.CELLS for arm in cell_arms(fraction, multiplier)]
        self.assertEqual(runner.check_mechanism(rows), [])
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_class(rows), [])
        self.assertEqual(runner.check_leaf_closure(rows), [])
        groups, problems = runner.rmc.check_invariants(rows, 4)
        self.assertEqual((groups, problems), (6, []))
        for row in rows:
            h = MATCHED_HORIZON_SECONDS[(row["l1_fraction"], row["l2_multiplier"])]
            self.assertEqual((row["mechanism"], row["eligibility"], row["width"], row["variant"],
                              row["h_star"]), ("leaf16", "leaf", 16, "main", h))
            self.assertEqual((row["l2_present_unusable_tokens"], row["orphaned_blocks"]), (0, 0))
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertGreater(row["stat_decisions"], 0)
            if row["arm"] in leafmatched.CLASS_STATISTIC_ARMS:
                self.assertEqual((row["class_horizon_seconds"], row["class_violations_seen"]),
                                 (h, 0))
            else:
                self.assertNotIn("class_violations_seen", row)
            if row["arm"] in matchedorder.ARMS:
                self.assertGreater(row["overridden_decisions_seen"], 0)
        # A publication of the mechanism control's own label rows passes the
        # reproduction column by column.
        published = {}
        for row in rows:
            if row["arm"] != ANCHOR_ARM:
                continue
            theirs = runner.rmc._replay_worker((self.name, row["l1_fraction"],
                                                row["l2_multiplier"], "leaf", 16, "label", 0))
            text = {key: str(value) for key, value in theirs.items()}
            published[("leaf16", "label") + runner._cell_seed(row)] = runner._reference_entry(
                text, "leaf16", "label", "mechanism_control_001")
        report, table = runner.check_reproduction(rows, published, 6)
        self.assertTrue(runner.reproduction_passes(report), report["mismatches"])
        self.assertEqual(report["counter_digests_compared"], 0)
        self.assertEqual(len(report["counter_columns_compared"]), 1)
        self.assertGreater(report["counter_columns_compared"][0], 80)

    def test_the_worker_refuses_an_arm_that_is_not_its_cells(self):
        with self.assertRaises(ValueError):
            runner._replay_worker((self.name, 0.0025, 1.0, "label_binary_600", 0))
        with self.assertRaises(ValueError):
            runner._replay_worker((self.name, 0.0025, 1.0, "learned", 0))


# --- the whole run on constructed traces -----------------------------------------------------------


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


def _fake_models(names):
    return ({name: _LinearRanker() for name in names},
            {name: {"path": f"models/pi0/{name}__next_use.json", "sha256": "a" * 64}
             for name in names})


def _sign_text(values) -> str:
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)


def _sign_reading(values) -> str:
    if all(value > 0 for value in values):
        return "consistent_gain"
    if all(value < 0 for value in values):
        return "consistent_loss"
    return "mixed"


def fixture_all16_tables(rows_by_key, traces, cells, seeds):
    """The published all16 tables of a constructed publication, computed here
    by hand from its rows (independently of the runner): one row per trace x
    cell at its h*, plus a decoy row the loader must skip."""
    tables = {name: [] for name in runner.PUBLISHED_TABLES}
    for name in traces:
        for cell in cells:
            h = MATCHED_HORIZON_SECONDS[cell]
            rung, learned, recency, _ = cell_arms(*cell)

            def u(arm, seed):
                row = rows_by_key[(arm, name) + cell + (seed,)]
                return 100.0 * row["extra_avoided_tokens"] / row["requested_tokens"]

            def mean(arm):
                return sum(u(arm, seed) for seed in seeds) / len(seeds)

            def difference(later, earlier):
                values = []
                for seed in seeds:
                    left = rows_by_key[(later, name) + cell + (seed,)]
                    right = rows_by_key[(earlier, name) + cell + (seed,)]
                    values.append(100.0 * (left["extra_avoided_tokens"]
                                           - right["extra_avoided_tokens"])
                                  / left["requested_tokens"])
                return values

            identity = {"trace": name, "l1_fraction": cell[0], "l2_multiplier": cell[1]}
            s = (mean("label") - mean(rung)) / (mean("label") - mean("lru"))
            shortfall_table = runner.SHORTFALL_TABLES[h]
            tables[shortfall_table].append({**identity, "h": h, "S": repr(s)})
            tables[shortfall_table].append({**identity, "h": 6.0, "S": "99.0"})      # decoy
            r_learned = (mean(learned) - mean("learned")) / (mean("evict_label") - mean("learned"))
            r_recency = (mean(recency) - mean("learned")) / (mean("evict_label") - mean("learned"))
            tables["matched_class_order_001/class_order.csv"].append(
                {**identity, "h_star": h, "R_matched_learned": repr(r_learned),
                 "R_matched_recency": repr(r_recency),
                 "reading": CLASS_READINGS[0] if r_learned >= 0.9 else CLASS_READINGS[1]})
            for table, prefix, later, earlier in (
                    ("matched_class_order_001/order.csv", "learned_minus_recency", learned,
                     recency),
                    ("matched_class_order_001/admission.csv", "label_binary_minus_recency", rung,
                     recency)):
                values = difference(later, earlier)
                tables[table].append({**identity, "h_star": h,
                                      f"{prefix}_points_mean": repr(sum(values) / len(values)),
                                      f"{prefix}_seed_signs": _sign_text(values),
                                      f"{prefix}_reading": _sign_reading(values)})
    return tables


class MainTests(unittest.TestCase):
    """`main` end to end on two constructed traces named as the grid's. The
    leaf16 references are produced by the parents' own workers (`lru`,
    `learned`, `label` by the mechanism control's, `evict_label` by the horizon
    control's); the all16 rows are synthesised on the same identifiers, with the
    published all16 tables computed from them by hand; git is answered as for a
    clean tree and the ranker loader is replaced by the fixed linear ranker."""

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
        grid = [(name, fraction, multiplier, seed) for name in sorted(traces)
                for fraction, multiplier in runner.CELLS for seed in SEEDS]
        with mock.patch.dict(runner.hc.SHARED, shared), \
                mock.patch.dict(runner.rmc.SHARED, shared), \
                mock.patch.dict(runner.rdp._SHARED, {"working_set": working_set}):
            mechanism_rows = [runner.rmc._replay_worker((name, fraction, multiplier, "leaf", 16,
                                                         rung, seed))
                              for name, fraction, multiplier, seed in grid
                              for rung in ("lru", "learned", "label")]
            evict_rows = [runner.hc._replay_worker((name, fraction, multiplier, "evict_label",
                                                    seed, "main"))
                          for name, fraction, multiplier, seed in grid]
        sources = {source: [] for source in runner.REFERENCE_SOURCES}
        sources["mechanism_control_001"] = list(mechanism_rows)
        # Rows the loader must skip: another mechanism's rung.
        sources["mechanism_control_001"].append(dict(mechanism_rows[0], mechanism="leaf64",
                                                     width=64))
        sources["horizon_control_001"] = list(evict_rows)
        rows_by_key = {}
        for row in evict_rows:
            cell = (row["l1_fraction"], row["l2_multiplier"])
            rung, learned, recency, _ = cell_arms(*cell)
            base = row["extra_avoided_tokens"]
            synthesised = [("error_location_001", "lru", base // 4),
                           ("error_location_001", "learned", base // 3),
                           ("error_location_001", "label", base + 500),
                           ("error_location_001", "evict_label", base + 400),
                           (runner.REFERENCES[("all16", rung)], rung, base + 450 + row["seed"]),
                           ("matched_class_order_001", learned, base + 380 + 7 * row["seed"]),
                           ("matched_class_order_001", recency, base + 390)]
            for source, arm, tokens in synthesised:
                entry = dict(row, arm=arm, mechanism="all16", eligibility="all", family="reference",
                             arm_parameter="", extra_avoided_tokens=tokens,
                             avoided_prefill_tokens=row["l1_avoided_tokens"] + tokens,
                             stat_decisions_admission=row["stat_decisions"] - 1)
                sources[source].append(entry)
                rows_by_key[(arm, row["trace"], cell[0], cell[1], row["seed"])] = entry
            # The fill-in's rerun of a horizon-control rung, which is skipped.
            if rung == "label_binary_150":
                sources["horizon_fill_001"].append(dict(row, arm="label_binary_600",
                                                        mechanism="all16", eligibility="all"))
        cls.sources = {}
        for source, rows in sources.items():
            path = root / "published" / source / "replay_seeds.csv"
            path.parent.mkdir(parents=True)
            runner._write(path, rows)
            cls.sources[source] = path
        cls.mechanism_rows = mechanism_rows
        cls.all16_rows = rows_by_key
        cls.trace_names = sorted(traces)
        cls.tables = cls._tables_for(SEEDS)
        cls.manifests = (root / "manifest_a.csv", root / "manifest_b.csv")
        for path in cls.manifests:
            path.write_text("policy,target,trace,model_sha256\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _tables_for(cls, seeds) -> dict:
        """The published all16 tables of the fixture over `seeds` (the seeds
        of the run, as for the real tables)."""
        directory = cls.root / "published" / ("tables_" + "_".join(str(seed) for seed in seeds))
        paths = {}
        for table, rows in fixture_all16_tables(cls.all16_rows, cls.trace_names, runner.CELLS,
                                                seeds).items():
            path = directory / table
            path.parent.mkdir(parents=True, exist_ok=True)
            runner._write(path, rows)
            paths[table] = path
        return paths

    def _tampered(self, name, source, change, table=False) -> dict:
        """The fixture sources (or tables) with one file's rows changed by
        `change` (a function of the rows that edits them in place)."""
        original = (self.tables if table else self.sources)[source]
        rows = list(csv_rows(original))
        change(rows)
        path = self.root / name / Path(source).name
        path.parent.mkdir(parents=True)
        runner._write(path, rows)
        return dict(self.tables if table else self.sources, **{source: path})

    def _main(self, argv, sources=None, tables=None, cells=None, seeds=None) -> str:
        patches = [mock.patch.object(sys, "argv", ["run_leaf_matched_horizon.py"]
                                     + self.trace_paths + argv),
                   mock.patch.object(runner.rmc, "_git", _fake_git),
                   mock.patch.object(runner, "sources_differing_from_head", lambda: []),
                   mock.patch.dict(runner.REFERENCE_SOURCES, sources or self.sources),
                   mock.patch.dict(runner.PUBLISHED_TABLES,
                                   tables or self._tables_for(seeds or SEEDS)),
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
        tables = {"replay_seeds.csv", "replay.csv", "references_seeds.csv", "readings_seeds.csv",
                  "horizon.csv", "class_order.csv", "order.csv", "admission.csv", "readings.csv",
                  "reproduction.csv", "README.md", "run_config.json"}
        self.assertEqual(set(os.listdir(paper)), tables)
        self.assertEqual(set(os.listdir(output)), tables | {"raw_replays.jsonl"})
        self.assertEqual(len((output / "raw_replays.jsonl").read_text().splitlines()), 240)
        self.assertIn("[240/240]", text)
        self.assertIn("all16 values recomputed from the published rows against the published "
                      "all16 tables (12 trace x cell, 10 values each): equal", text)
        self.assertIn("reading 1, horizon under leaf eligibility", text)
        replays = list(csv_rows(paper / "replay_seeds.csv"))
        self.assertEqual(len(replays), 240)
        self.assertEqual({(row["mechanism"], row["eligibility"], row["width"]) for row in replays},
                         {("leaf16", "leaf", "16")})
        self.assertTrue(all(row["l2_present_unusable_tokens"] == "0" for row in replays))
        # The 60 label replays are the mechanism control's own rows, column by column.
        reproduction = list(csv_rows(paper / "reproduction.csv"))
        self.assertEqual(len(reproduction), 60)
        self.assertTrue(all(row["reproduces"] == "True" for row in reproduction))
        self.assertTrue(all(row["counter_columns_differing"] == "" for row in reproduction))
        self.assertTrue(all(row["same_counters_sha256"] == "" for row in reproduction))
        published_label = {(row["trace"], row["l1_fraction"], row["l2_multiplier"],
                            row["seed"]): row for row in self.mechanism_rows
                           if row["rung"] == "label"}
        for row in replays:
            if row["arm"] != ANCHOR_ARM:
                continue
            theirs = published_label[runner._cell_seed(row)]
            self.assertEqual(int(row["avoided_prefill_tokens"]), theirs["avoided_prefill_tokens"])
            self.assertEqual(row["reproduces_reference"], "True")
        counts = {name: len(list(csv_rows(paper / f"{name}.csv")))
                  for name in ("horizon", "class_order", "order", "admission", "readings_seeds",
                               "references_seeds", "readings", "replay")}
        self.assertEqual(counts, {"horizon": 12, "class_order": 12, "order": 12, "admission": 12,
                                  "readings_seeds": 120, "references_seeds": 60 * 11,
                                  "readings": 4, "replay": 48})
        # The all16 columns are the published (fixture) values.
        published_r = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])):
                       float(row["R_matched_learned"])
                       for row in csv_rows(self.tables["matched_class_order_001/class_order.csv"])}
        for row in csv_rows(paper / "class_order.csv"):
            self.assertAlmostEqual(float(row["all16_R_matched_learned"]),
                                   published_r[tabulate.cell_key(row)], places=12)
        readings = {row["reading"]: row for row in csv_rows(paper / "readings.csv")}
        for reading in leafmatched.PREDICTIONS:
            self.assertEqual(readings[reading]["cells"], "12")
            self.assertIn(readings[reading]["prediction_holds"], ("True", "False"))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["phase"], "leaf_matched_horizon")
        self.assertEqual(config["plan"], "docs/leaf-matched-horizon-plan.md")
        self.assertEqual((config["plan_commit"], config["code_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(config["mechanism"]["eligibility"], "leaf")
        self.assertEqual(config["mechanism"]["width"], 16)
        hashed = set(config["references"])
        for path in list(self.sources.values()) + list(self.tables.values()):
            self.assertIn(str(path.resolve()), hashed)
            self.assertEqual(config["references"][str(path.resolve())], sha256_path(path))
        self.assertEqual(set(config["trace_files"]), {"conversation_trace", "toolagent_trace"})
        checks = config["checks"]
        self.assertEqual((checks["reproduction"]["matched"], checks["reproduction"]["expected"],
                          checks["reproduction"]["counter_digests_compared"]), (60, 60, 0))
        self.assertEqual((checks["identifier_problems"], checks["mechanism_problems"],
                          checks["leaf16_closure_problems"], checks["statistics_problems"],
                          checks["class_problems"], checks["class_violations"],
                          checks["class_overridden_outside_admission"],
                          checks["invariant_violations"], checks["unexplained_absent_tokens"],
                          checks["all16_mismatches"]), (0,) * 10)
        self.assertEqual(checks["class_replays"], 180)
        self.assertEqual(checks["identifier_pairs_compared"], 240 * 11)
        self.assertEqual(checks["all16_values_checked"], 120)
        self.assertEqual((config["replays"], config["workers"]), (240, 2))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(tabulate.main([str(paper)]), 0)
        printed = stdout.getvalue()
        for section in ("L0", "L1", "L2", "L3", "L4", "L5"):
            self.assertIn(f"### {section} ", printed)
        self.assertEqual(sorted(os.listdir(paper)), sorted(tables))

    def test_a_reproduction_difference_publishes_nothing(self):
        def target(rows, trace):
            return next(row for row in rows if row["rung"] == "label" and row["seed"] == "0"
                        and float(row["l1_fraction"]) == 0.01
                        and float(row["l2_multiplier"]) == 4.0 and row["trace"] == trace)

        def tokens(rows):
            row = target(rows, "conversation_trace")
            row["avoided_prefill_tokens"] = str(int(row["avoided_prefill_tokens"]) + 1)

        def counter(rows):
            row = target(rows, "toolagent_trace")
            row["l2_rejections"] = str(int(row["l2_rejections"]) + 1)

        for name, change, column in (("tokens", tokens, "avoided_prefill_tokens"),
                                     ("counter", counter, "l2_rejections")):
            with self.subTest(case=name):
                sources = self._tampered(f"tampered_{name}", "mechanism_control_001", change)
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
        self.assertEqual(config["replays"], 2 * 2 * 4 * 2)
        self.assertEqual(config["checks"]["reproduction"]["matched"], 2 * 2 * 2)
        readings = {row["reading"]: row for row in csv_rows(paper / "readings.csv")}
        # Four trace x cell: the predictions about 12 are not evaluated.
        self.assertEqual(readings["1_horizon_under_leaf_eligibility"]["prediction_holds"], "")

    def test_an_all16_table_off_the_published_rows_is_refused_before_any_replay(self):
        def change(rows):
            rows[3]["R_matched_learned"] = repr(float(rows[3]["R_matched_learned"]) + 0.01)

        tables = self._tampered("tampered_table", "matched_class_order_001/class_order.csv",
                                change, table=True)
        output, paper = self.root / "run_table", self.root / "paper_table"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper)], tables=tables)
        self.assertIn("not the published ones; nothing run", str(caught.exception))
        self.assertIn("ALL16", self.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(paper.exists())

    def test_missing_references_are_refused_before_any_replay(self):
        def drop(rows):
            rows[:] = [row for row in rows if not (row["rung"] == "learned"
                                                   and float(row["l1_fraction"]) == 0.0025
                                                   and row["seed"] == "4")]

        sources = self._tampered("missing", "mechanism_control_001", drop)
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
        tables = runner.readings_tables(rows, published)
        summary = runner.reading_summary(tables)
        anchors = [_anchor_row(seed=seed) for seed in SEEDS]
        _, reproduction = runner.check_reproduction(anchors, _anchor_published(anchors), 5)
        written = {"replay": runner.aggregate_replays(rows), "reproduction": reproduction,
                   "readings": summary, **{name: tables[name] for name in
                                           ("horizon", "class_order", "order", "admission")}}
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
        for section in ("L0", "L1", "L2", "L3", "L4", "L5"):
            self.assertIn(f"### {section} ", text)
        self.assertEqual(tabulate.reproduction_counts(parsed["reproduction"]), [{
            "replays": 5, "same_tokens": 5,
            "counter_columns": str(len(runner.REQUIRED_COUNTER_COLUMNS) + 2),
            "all_counter_columns_equal": 5, "counter_digests_published": 0,
            "same_counter_digest": 0, "decision_digests_published": 0,
            "same_decision_digest": 0, "reproduces": 5}])
        horizon = tabulate.horizon_table(parsed["horizon"])
        self.assertEqual([(row["trace"], row["l1_fraction"]) for row in horizon],
                         [("conversation_trace", 0.0025), ("conversation_trace", 0.02),
                          ("toolagent_trace", 0.01)])
        self.assertAlmostEqual(horizon[0]["S"], 0.08, places=12)
        self.assertEqual(horizon[2]["reading"], HORIZON_READINGS[1])
        class_order = tabulate.class_order_table(parsed["class_order"])
        self.assertEqual([row["predicted"] for row in class_order], [False, True, True])
        order = tabulate.order_table(parsed["order"])
        self.assertEqual(order[2]["signs"], (2, 1, 2))
        admission = tabulate.admission_table(parsed["admission"])
        self.assertAlmostEqual(admission[0]["share_recency"], 0.25)
        self.assertAlmostEqual(admission[0]["all16_share_rung"], 0.95)
        counts = {row["reading"]: row for row in tabulate.counts_table(parsed["readings"])}
        self.assertEqual(counts["1_horizon_under_leaf_eligibility"]["count"], 2)
        self.assertEqual(counts["1_horizon_under_leaf_eligibility"]["required"], "12/12")
        self.assertEqual(counts["1_horizon_under_leaf_eligibility"]["holds"], "")
        self.assertEqual(counts["4_admission_descriptive"]["count"], "")
        self.assertIn("consistent_gain 3", counts["4_admission_descriptive"]["outcomes"])
        counters = tabulate.replay_counters(parsed["replay"])
        self.assertEqual([row["arm"] for row in counters][:4],
                         ["label_binary_60", "evict_binary_60_learned", "evict_binary_60_recency",
                          "label"])
        # Every printed counter is one the aggregation writes (the anchor has no
        # class statistics).
        for row in counters:
            for column in tabulate.REPLAY_COLUMNS:
                if column != "class_violations_seen_mean" or row["arm"] != ANCHOR_ARM:
                    self.assertFalse(math.isnan(row[column]), (row["arm"], column))


if __name__ == "__main__":
    unittest.main()
