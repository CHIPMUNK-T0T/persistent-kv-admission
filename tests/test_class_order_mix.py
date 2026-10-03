"""Tests for the class-order mix (docs/class-order-mix-plan.md).

No real trace is replayed: the plan forbids a real replay before the
implementation is reviewed and committed. Everything here runs on constructed
traces, hand-built decisions and synthetic rows; the published result tables
are only read. What has to hold before a real replay means anything: the arms,
grid, references and named cells are the plan's; the mixed construction with
the ranker on both classes is the matched learned arm and with the ranker on
neither is the matched recency arm, decision by decision with digests;
`mix_out` and `mix_in` order the right class by the right key on a
constructed decision where the classes differ; the random arm draws one
uniform per candidate per decision from a stream of its own, is identical
under the same seed and differs under another, has the recency arm's
candidate sets on a replay where both must make the same choices, and never
reads or advances the store's sampling stream (and the test that says so has
teeth); the class statistic is zero for every arm; the ranker is observed once
per observation; the reading helpers do what the plan fixes at their
boundaries; and the runner's checks, refusals, derivations (also against the
published matched class-order tables), its end-to-end run with git mocked and
the read-only tabulation behave as stated, including that a tampered
reference publishes nothing.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from persistent_kv_admission import classmix, errorloc, horizonctl, matchedorder
from persistent_kv_admission.classmix import (
    ARMS,
    BOTH_CLASSES,
    KINDS,
    MIX_IN,
    MIX_OUT,
    NEITHER_CLASS,
    PREDICTION_ONE_CELLS,
    RANDOM,
    MixedClassScorer,
    RandomClassScorer,
    cell_arms,
    mix_arm,
)
from persistent_kv_admission.decisionpop import RawFeatureScorer, horizon_for
from persistent_kv_admission.errorloc import (
    DecisionStatistics,
    HybridOverride,
    HybridScorer,
    RecordingOverride,
    first_minimum,
)
from persistent_kv_admission.gap import summarize, working_set_bytes
from persistent_kv_admission.horizonctl import ReuseClassScorer
from persistent_kv_admission.matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    ClassStatistics,
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
# tracer, the fixed and linear rankers), reused as they are; only plain
# functions and classes are read from them.
horizon_tests = _load("_class_order_mix_horizon_fixtures",
                      REPOSITORY / "tests/test_horizon_control.py")
runner = _load("_run_class_order_mix_under_test", REPOSITORY / "scripts/run_class_order_mix.py")
tabulate = _load("_tabulate_class_order_mix_under_test",
                 REPOSITORY / "scripts/tabulate_class_order_mix.py")

HORIZON = horizon_tests.HORIZON
MEASURE_FROM_MS = horizon_tests.MEASURE_FROM_MS
L1_BYTES = horizon_tests.L1_BYTES
L2_BYTES = horizon_tests.L2_BYTES
partial_trace = horizon_tests.partial_trace
_row = horizon_tests._row
_binary = horizon_tests._binary
_Fixed = horizon_tests._Fixed
_Tracer = horizon_tests._Tracer
_LinearRanker = horizon_tests._LinearRanker
csv_rows = horizon_tests.csv_rows
SEEDS = (0, 1, 2, 3, 4)
Replay = collections.namedtuple("Replay", "result statistics classes tracer spy")


def _without_keys(decisions):
    """Each traced decision without the store's keys: (timestamp, group,
    arrival index, candidates, store's victim, final victim)."""
    return [decision[:4] + decision[5:] for decision in decisions]


def _candidates_and_final(decisions):
    """Each traced decision's candidate set and final victim (and where it
    was): what two arms making the same choices must share."""
    return [decision[:4] + (decision[6],) for decision in decisions]


class _RngSpy:
    """Stands in for the store's sampling stream: forwards every call to the
    real `random.Random` and records the method and the calling function."""

    def __init__(self, rng):
        self.rng = rng
        self.calls = []

    def __getattr__(self, name):
        attribute = getattr(self.rng, name)
        if not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            self.calls.append((name, sys._getframe(1).f_code.co_name))
            return attribute(*args, **kwargs)
        return call


class _StoreSpy:
    """Override-slot wrapper: on attach, replaces the store's sampling stream
    by an `_RngSpy`; at every decision records a hash of that stream's state
    (after the decision's draw); passes the inner override through."""

    def __init__(self, inner):
        self.inner = inner
        self.rng = None
        self.states = []

    def attach(self, l1, l2, counters):
        self.rng = _RngSpy(l2.rng)
        l2.rng = self.rng
        if hasattr(self.inner, "attach"):
            self.inner.attach(l1, l2, counters)

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index, arriving_index):
        self.states.append(hash(self.rng.rng.getstate()))
        return self.inner(candidates, keys, victim_index, timestamp_ms, group_index,
                          arriving_index)


def traced_replay(trace, setup, h, seed=1, width=4, wrapped=True, spy=False):
    """One traced all16-style replay (`width` residents sampled) with the
    error-location statistics at the trace's label horizon and the class
    statistics at `h`, nested as the runner nests them (or neither,
    `wrapped=False`); with `spy`, the store's sampling stream is watched."""
    statistics = DecisionStatistics(trace, HORIZON, MEASURE_FROM_MS) if wrapped else None
    classes = ClassStatistics(trace, h, MEASURE_FROM_MS) if wrapped else None
    override = (RecordingOverride(statistics, RecordingOverride(classes, setup.override))
                if wrapped else setup.override)
    watcher = _StoreSpy(override) if spy else None
    tracer = _Tracer(watcher if spy else override)
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                          l2_sample_width=width, l2_seed=seed, l2_eligibility="all",
                          l2_override_hook=tracer, **setup.replay_arguments())
    return Replay(result, statistics, classes, tracer, watcher)


def _setup(arm, trace, seed=1):
    return classmix.arm_setup(arm, trace, learned_ranker=_LinearRanker(), seed=seed)


def _matched_setup(h, order, trace):
    return matchedorder.arm_setup(matched_arm(h, order), trace, learned_ranker=_LinearRanker())


def _ranker(trace):
    return RawFeatureScorer(trace, _LinearRanker())


# --- the grid --------------------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_arms_are_the_plans(self):
        self.assertEqual(KINDS, ("mix_out_learned", "mix_in_learned", "random_within_class"))
        self.assertEqual(ARMS, tuple(f"{kind}_{h}" for h in ("60", "150", "300", "600")
                                     for kind in KINDS))
        self.assertEqual(classmix.LEARNED_CLASS, {MIX_OUT: 0, MIX_IN: 1})
        self.assertEqual((classmix.NON_REUSABLE, classmix.REUSABLE), (0, 1))
        for cell, h in MATCHED_HORIZON_SECONDS.items():
            with self.subTest(cell=cell):
                self.assertEqual(cell_arms(*cell), (f"mix_out_learned_{h:g}",
                                                    f"mix_in_learned_{h:g}",
                                                    f"random_within_class_{h:g}"))
                for arm in cell_arms(*cell):
                    self.assertEqual(classmix.HORIZON_OF_ARM[arm], h)
                self.assertEqual(classmix.matched_reference_arms(*cell), {
                    "matched_learned": f"evict_binary_{h:g}_learned",
                    "matched_recency": f"evict_binary_{h:g}_recency",
                    "label_binary_hstar": f"label_binary_{h:g}"})
        self.assertEqual(mix_arm(MIX_OUT, 150), "mix_out_learned_150")
        with self.assertRaises(ValueError):
            mix_arm("mix_both", 60.0)
        with self.assertRaises(ValueError):
            cell_arms(0.05, 1.0)

    def test_the_tasks(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(len(set(tasks)), 180)
        by_cell = collections.defaultdict(set)
        for name, fraction, multiplier, arm, seed in tasks:
            by_cell[(name, fraction, multiplier)].add(arm)
        self.assertEqual(len(by_cell), 12)
        for (name, fraction, multiplier), arms in by_cell.items():
            self.assertEqual(arms, set(cell_arms(fraction, multiplier)))
        kinds = [KINDS.index(classmix.KIND_OF_ARM[task[3]]) for task in tasks]
        self.assertEqual(kinds, sorted(kinds))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH, runner.MECHANISM), ("all", 16, "all16"))
        self.assertEqual(runner.ARMS_PER_CELL, 3)

    def test_the_references_are_the_plans(self):
        expected = {("all16", "learned"): "error_location_001",
                    ("all16", "evict_label"): "error_location_001",
                    ("all16", "label_binary_60"): "horizon_control_001",
                    ("all16", "label_binary_150"): "horizon_fill_001",
                    ("all16", "label_binary_300"): "horizon_control_001",
                    ("all16", "label_binary_600"): "horizon_control_001"}
        for h in ("60", "150", "300", "600"):
            for order in ("learned", "recency"):
                expected[("all16", f"evict_binary_{h}_{order}")] = "matched_class_order_001"
        self.assertEqual(runner.REFERENCES, expected)
        self.assertEqual(runner.REFERENCE_SOURCES, {
            "error_location_001": REPOSITORY / "results/paper/error_location_001/replay_seeds.csv",
            "matched_class_order_001":
                REPOSITORY / "results/paper/matched_class_order_001/replay_seeds.csv",
            "horizon_control_001": REPOSITORY / "results/paper/horizon_control_001/replay_seeds.csv",
            "horizon_fill_001": REPOSITORY / "results/paper/horizon_fill_001/replay_seeds.csv"})
        self.assertEqual(runner.reference_cells("all16", "evict_binary_150_recency"),
                         ((0.0025, 4.0), (0.01, 1.0)))
        self.assertEqual(runner.reference_cells("all16", "label_binary_60"), ((0.0025, 1.0),))
        self.assertEqual(runner.reference_cells("all16", "evict_binary_600_learned"),
                         ((0.01, 4.0), (0.02, 4.0)))
        self.assertEqual(runner.reference_cells("all16", "learned"), runner.CELLS)
        for name in ("lru", "label", "evict_binary_learned"):
            with self.assertRaises(ValueError):
                runner.reference_cells("all16", name)
        self.assertEqual(runner.DEFAULT_PAPER_DIR, REPOSITORY / "results/paper/class_order_mix_001")

    def test_the_named_cells_are_the_plans_and_the_published_consistent_losses(self):
        self.assertEqual(PREDICTION_ONE_CELLS, (
            ("conversation_trace", 0.0025, 1.0), ("conversation_trace", 0.0025, 4.0),
            ("conversation_trace", 0.01, 1.0), ("conversation_trace", 0.02, 1.0),
            ("toolagent_trace", 0.0025, 1.0), ("toolagent_trace", 0.01, 1.0),
            ("toolagent_trace", 0.01, 4.0), ("toolagent_trace", 0.02, 1.0)))
        # The plan names them as the consistent losses of the matched control's
        # reading 2; the published table agrees (read only).
        losses = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))
                  for row in csv_rows(REPOSITORY / "results/paper/matched_class_order_001/order.csv")
                  if row["learned_minus_recency_reading"] == "consistent_loss"}
        self.assertEqual(losses, set(PREDICTION_ONE_CELLS))
        self.assertEqual((classmix.PREDICTION_TWO_MINIMUM, classmix.PREDICTION_TWO_CELLS), (8, 12))

    def test_the_tabulation_uses_the_same_vocabulary(self):
        self.assertEqual(tabulate.KINDS, KINDS)
        self.assertEqual(tabulate.TRACES, runner.TRACES)
        self.assertEqual(tabulate.SIGN_READINGS, matchedorder.SIGN_READINGS)
        self.assertEqual(tabulate.U_NAMES, runner.U_NAMES)
        self.assertEqual(tabulate.R_NAMES, runner.R_NAMES)
        self.assertEqual(tabulate.READING_COLUMNS, runner.READING_COLUMNS)


# --- the arms --------------------------------------------------------------------------------------


class ArmConstructionTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_mixed_arms(self):
        for h in MATCHED_HORIZONS_SECONDS:
            for kind, learned_class in ((MIX_OUT, 0), (MIX_IN, 1)):
                with self.subTest(h=h, kind=kind):
                    setup = _setup(mix_arm(kind, h), self.trace)
                    self.assertEqual((setup.arm, setup.l2_policy), (mix_arm(kind, h), "learned"))
                    self.assertIsInstance(setup.scorer, MixedClassScorer)
                    self.assertIsInstance(setup.scorer.reuse, ExactLabelScorer)
                    self.assertEqual((setup.scorer.reuse.target, setup.scorer.reuse.horizon_seconds),
                                     ("binary", h))
                    self.assertIsInstance(setup.scorer.within, RawFeatureScorer)
                    self.assertEqual(setup.scorer.learned_classes, frozenset({learned_class}))
                    self.assertIsInstance(setup.override, HybridOverride)
                    # The ranker the store orders by is the one admission consults.
                    self.assertIs(setup.override.admission, setup.scorer.within)
                    self.assertIsNone(classmix.random_scorer(setup))
                    self.assertFalse(hasattr(setup.scorer, "attach"))

    def test_the_random_arm_is_built_as_the_recency_arm_with_its_own_stream(self):
        for h in MATCHED_HORIZONS_SECONDS:
            with self.subTest(h=h):
                arm = mix_arm(RANDOM, h)
                setup = _setup(arm, self.trace, seed=3)
                recency = _matched_setup(h, "recency", self.trace)
                self.assertIsInstance(setup.scorer, HybridScorer)
                self.assertIsInstance(setup.scorer.admission, RawFeatureScorer)
                self.assertIs(setup.override.admission, setup.scorer.admission)
                stream = classmix.random_scorer(setup)
                self.assertIs(stream, setup.scorer.eviction)
                self.assertIsInstance(stream, RandomClassScorer)
                self.assertEqual((stream.reuse.target, stream.reuse.horizon_seconds), ("binary", h))
                self.assertEqual((stream.seed, stream.arm), (3, arm))
                self.assertEqual(stream.stream_seed, classmix.random_stream_seed(3, arm))
                self.assertEqual(stream.draws, 0)
                # The ranker is shown the history in the recency arm's order:
                # the admission ranker first, then the eviction side (the reuse
                # label there, the stream forwarding to it here).
                self.assertEqual(len(setup.scorer.observed), len(recency.scorer.observed))
                self.assertIs(setup.scorer.observed[0], setup.scorer.admission)
                self.assertIs(setup.scorer.observed[1], stream)
                self.assertIs(recency.scorer.observed[0], recency.scorer.admission)
                self.assertFalse(hasattr(stream, "attach"))
                self.assertFalse(hasattr(setup.scorer, "attach"))

    def test_unknown_arms_and_bad_inputs_are_refused(self):
        for arm in ("evict_binary_60_learned", "mix_out_learned_90", "random_within_class",
                    "mix_out_learned_600.5"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                _setup(arm, self.trace)
        with self.assertRaises(ValueError):
            classmix.arm_setup("mix_in_learned_60", self.trace, learned_ranker=None, seed=0)
        ranker = _ranker(self.trace)
        with self.assertRaises(ValueError):
            MixedClassScorer(ranker, ranker, 0)
        reuse = ExactLabelScorer(self.trace, 60.0, "binary")
        for bad in (2, -1, True, "1", 0.0, (0, 2), (1.0,), [0, 1], None):
            with self.subTest(learned_class=bad), self.assertRaises(ValueError):
                MixedClassScorer(reuse, ranker, bad)
        self.assertEqual(MixedClassScorer(reuse, ranker, BOTH_CLASSES).learned_classes,
                         frozenset({0, 1}))
        self.assertEqual(MixedClassScorer(reuse, ranker, NEITHER_CLASS).learned_classes,
                         frozenset())

    def test_a_reuse_label_that_is_not_a_bit_is_refused(self):
        not_a_bit = _Fixed({"s": 0.5})
        for scorer in (MixedClassScorer(not_a_bit, _Fixed({"s": 1.0}), 0),
                       RandomClassScorer(not_a_bit, 0, "random_within_class_60")):
            with self.subTest(scorer=type(scorer).__name__), self.assertRaises(ValueError):
                scorer.score("s", 0.0)


# --- the null settings: the matched control's two arms, decision by decision ----------------------


class NullSettingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_ranker_on_both_classes_is_the_matched_learned_arm(self):
        for h in MATCHED_HORIZONS_SECONDS:
            for seed in (1, 3):
                with self.subTest(h=h, seed=seed):
                    mine = traced_replay(self.trace, classmix.mixed_setup(
                        "both", self.trace, _ranker(self.trace), BOTH_CLASSES, h), h, seed=seed)
                    theirs = traced_replay(self.trace, _matched_setup(h, "learned", self.trace), h,
                                           seed=seed)
                    # Keys included: the store's score is ReuseClassScorer's.
                    self.assertEqual(mine.tracer.decisions, theirs.tracer.decisions)
                    self.assertGreater(len(mine.tracer.decisions), 100)
                    self.assertEqual(_row(mine.result), _row(theirs.result))
                    self.assertEqual(mine.statistics.row(), theirs.statistics.row())
                    self.assertEqual(mine.classes.row(), theirs.classes.row())
                    self.assertGreater(mine.statistics.row()["overridden_decisions_seen"], 0)

    def test_the_ranker_on_neither_class_is_the_matched_recency_arm(self):
        for h in MATCHED_HORIZONS_SECONDS:
            for seed in (1, 3):
                with self.subTest(h=h, seed=seed):
                    mine = traced_replay(self.trace, classmix.mixed_setup(
                        "neither", self.trace, _ranker(self.trace), NEITHER_CLASS, h), h,
                        seed=seed)
                    theirs = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h,
                                           seed=seed)
                    self.assertEqual(_without_keys(mine.tracer.decisions),
                                     _without_keys(theirs.tracer.decisions))
                    # The keys differ only in form: ((bit, constant), last_group)
                    # where the recency arm has (bit, last_group).
                    for left, right in zip(mine.tracer.decisions, theirs.tracer.decisions):
                        self.assertEqual(left[4], tuple(((key[0], classmix.WITHIN_CLASS_CONSTANT),
                                                         key[1]) for key in right[4]))
                    self.assertEqual(_row(mine.result), _row(theirs.result))
                    # decision_sha256 and m1 included: the key orders identically.
                    self.assertEqual(mine.statistics.row(), theirs.statistics.row())
                    self.assertEqual(mine.classes.row(), theirs.classes.row())

    def test_the_scores_of_the_null_settings(self):
        # Candidate by candidate, at every decision instant of a replay, the
        # store's score is ReuseClassScorer's over the same ranker object (both
        # classes) or (bit, constant) (neither).
        h = 150.0
        for classes in (BOTH_CLASSES, NEITHER_CLASS):
            with self.subTest(classes=classes):
                ranker, reuse = _ranker(self.trace), ExactLabelScorer(self.trace, h, "binary")
                setup = classmix.mixed_setup("null", self.trace, ranker, classes, h)
                # Not observed itself: it reads the ranker the store's scorer observes.
                composite = ReuseClassScorer(reuse, ranker)
                compared = []

                def check(candidates, keys, victim_index, timestamp_ms, group_index,
                          arriving_index, inner=setup.override, classes=classes,
                          composite=composite, reuse=reuse, compared=compared):
                    for index, state_id in enumerate(candidates):
                        expected = (composite.score(state_id, timestamp_ms) if classes
                                    else (reuse.score(state_id, timestamp_ms),
                                          classmix.WITHIN_CLASS_CONSTANT))
                        self.assertEqual(keys[index][0], expected)
                        compared.append(state_id)
                    return inner(candidates, keys, victim_index, timestamp_ms, group_index,
                                 arriving_index)

                run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                             measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                             l2_sample_width=4, l2_seed=1, l2_eligibility="all",
                             l2_override_hook=check, **setup.replay_arguments())
                self.assertGreater(len(compared), 500)

    def test_the_mixed_and_random_arms_are_neither_null_setting(self):
        for h in MATCHED_HORIZONS_SECONDS:
            learned = traced_replay(self.trace, _matched_setup(h, "learned", self.trace), h)
            recency = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h)
            for kind in KINDS:
                with self.subTest(h=h, kind=kind):
                    mine = traced_replay(self.trace, _setup(mix_arm(kind, h), self.trace), h)
                    for other in (learned, recency):
                        self.assertNotEqual(_candidates_and_final(mine.tracer.decisions),
                                            _candidates_and_final(other.tracer.decisions))

    def test_the_ranker_is_observed_once_per_observation_in_every_arm(self):
        class Counting:
            time_varying = True

            def __init__(self, inner):
                self.inner, self.observed = inner, 0

            def observe(self, requests, timestamp_ms):
                self.observed += 1
                self.inner.observe(requests, timestamp_ms)

            def score(self, state_id, timestamp_ms):
                return self.inner.score(state_id, timestamp_ms)

        counts = {}
        h = 150.0
        builders = {
            "matched_learned": lambda r: matchedorder.class_order_setup("l", self.trace, r,
                                                                        "learned", h),
            "matched_recency": lambda r: matchedorder.class_order_setup("r", self.trace, r,
                                                                        "recency", h),
            MIX_OUT: lambda r: classmix.mixed_setup("o", self.trace, r, 0, h),
            MIX_IN: lambda r: classmix.mixed_setup("i", self.trace, r, 1, h),
            RANDOM: lambda r: classmix.random_setup(mix_arm(RANDOM, h), self.trace, r, h, 1),
        }
        for name, build in builders.items():
            ranker = Counting(_ranker(self.trace))
            traced_replay(self.trace, build(ranker), h)
            counts[name] = ranker.observed
        self.assertEqual(len(set(counts.values())), 1, counts)
        self.assertEqual(counts[MIX_OUT], len(list(self.trace.timestamp_groups())))


# --- the mixed key on a hand-built decision --------------------------------------------------------


class MixedKeyOnAHandBuiltDecisionTests(unittest.TestCase):
    """Decisions at t = 0 on a constructed trace, the arrival first. Within
    60 s, r1 and r2 are reusable (next use 30 s and 40 s away) and r3, r4 are
    not (300 s away, never). The ranker keeps the arrival; within each class
    its order and recency's disagree: the ranker evicts r3 (2.0 < 3.0) and r1
    (0.5 < 1.0), recency evicts r4 (last_group 1 < 6) and r2 (2 < 7)."""

    ARRIVAL, R1, R2, R3, R4 = "int:5", "int:1", "int:2", "int:3", "int:4"

    def setUp(self):
        records = [{"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [k]}
                   for k in (1, 2, 3, 4, 5)]
        for timestamp, k in ((30_000, 1), (30_000, 5), (40_000, 2), (300_000, 3)):
            records.append({"timestamp": timestamp, "input_length": 512, "output_length": 1,
                            "hash_ids": [k]})
        temporary, self.trace, _ = horizon_tests.build(records)
        self.addCleanup(temporary.cleanup)
        self.ranker = {self.ARRIVAL: 9.0, self.R1: 0.5, self.R2: 1.0, self.R3: 2.0, self.R4: 3.0}
        self.last_group = {self.ARRIVAL: 9, self.R1: 7, self.R2: 2, self.R3: 6, self.R4: 1}

    def _keys(self, setup, candidates):
        return [(setup.scorer.score(state_id, 0.0), float(self.last_group[state_id]))
                for state_id in candidates]

    def _decide(self, setup, candidates, arriving_index):
        keys = self._keys(setup, candidates)
        victim = first_minimum(keys)
        answer = setup.override(list(candidates), keys, victim, 0.0, 0, arriving_index)
        return keys, victim, (victim if answer is None else answer)

    def _mixed(self, learned_class):
        return classmix.mixed_setup("arm", self.trace, _Fixed(self.ranker), learned_class, 60.0)

    def test_the_classes_within_60_seconds(self):
        self.assertEqual([_binary(self.trace, s, 0.0, 60.0)
                          for s in (self.ARRIVAL, self.R1, self.R2, self.R3, self.R4)],
                         [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_the_stores_key_orders_each_class_by_its_key(self):
        candidates = [self.ARRIVAL, self.R1, self.R2, self.R3, self.R4]
        for learned_class in (0, 1, BOTH_CLASSES, NEITHER_CLASS):
            with self.subTest(learned_class=learned_class):
                classes = {learned_class} if isinstance(learned_class, int) else set(learned_class)
                keys = self._keys(self._mixed(learned_class), candidates)
                flat = []
                for index, state_id in enumerate(candidates):
                    bit = _binary(self.trace, state_id, 0.0, 60.0)
                    within = self.ranker[state_id] if bit in classes else 0.0
                    self.assertEqual(keys[index], ((bit, within), float(self.last_group[state_id])))
                    flat.append((bit, within, float(self.last_group[state_id])))
                self.assertEqual(sorted(range(5), key=keys.__getitem__),
                                 sorted(range(5), key=flat.__getitem__))

    def test_mix_out_orders_the_non_reusable_class_by_the_ranker(self):
        candidates = [self.ARRIVAL, self.R1, self.R2, self.R3, self.R4]
        # Admission decision: the ranker keeps the arrival; the victim is the
        # first minimum among the others, in the non-reusable class {r3, r4}.
        expected = {0: 3, 1: 4, BOTH_CLASSES: 3, NEITHER_CLASS: 4}
        for learned_class, victim in expected.items():
            with self.subTest(learned_class=learned_class):
                self.assertEqual(self._decide(self._mixed(learned_class), candidates, 0)[2], victim)
        # A later round (no arrival among the candidates): the store's own choice.
        for learned_class, victim in expected.items():
            with self.subTest(later=learned_class):
                keys, store_victim, final = self._decide(self._mixed(learned_class),
                                                         candidates[1:], -1)
                self.assertEqual((store_victim, final), (victim - 1, victim - 1))

    def test_mix_in_orders_the_reusable_class_by_the_ranker(self):
        # Only reusable residents: mix_in evicts the ranker's minimum r1, mix_out
        # recency's r2.
        candidates = [self.ARRIVAL, self.R1, self.R2]
        expected = {0: 2, 1: 1, BOTH_CLASSES: 1, NEITHER_CLASS: 2}
        for learned_class, victim in expected.items():
            with self.subTest(learned_class=learned_class):
                self.assertEqual(self._decide(self._mixed(learned_class), candidates, 0)[2], victim)

    def test_the_ranker_alone_decides_rejection_in_every_arm(self):
        self.ranker[self.ARRIVAL] = 0.1
        candidates = [self.ARRIVAL, self.R1, self.R2, self.R3, self.R4]
        for learned_class in (0, 1):
            self.assertEqual(self._decide(self._mixed(learned_class), candidates, 0)[2], 0)
        setup = classmix.random_setup(mix_arm(RANDOM, 60), self.trace, _Fixed(self.ranker), 60.0, 0)
        self.assertEqual(self._decide(setup, candidates, 0)[2], 0)

    def test_the_random_key_is_a_fresh_uniform_per_candidate(self):
        arm = mix_arm(RANDOM, 60)
        candidates = [self.ARRIVAL, self.R1, self.R2, self.R3, self.R4]
        setup = classmix.random_setup(arm, self.trace, _Fixed(self.ranker), 60.0, 7)
        stream = random.Random(classmix.random_stream_seed(7, arm))
        for decision in range(3):
            keys, _, final = self._decide(setup, candidates, 0)
            draws = [stream.random() for _ in candidates]
            for index, state_id in enumerate(candidates):
                self.assertEqual(keys[index][0], (_binary(self.trace, state_id, 0.0, 60.0),
                                                  draws[index]))
                self.assertTrue(0.0 <= keys[index][0][1] < 1.0)
            # The victim is the non-reusable resident with the smaller draw.
            self.assertEqual(final, 3 if draws[3] < draws[4] else 4)
        self.assertEqual(classmix.random_scorer(setup).draws, 3 * len(candidates))


# --- the random arm on replays ---------------------------------------------------------------------


class _LeakyRandom:
    """A deliberately wrong random arm (tests only): its within-class key is
    drawn from the store's own sampling stream (`store.rng`, read at every
    draw), the store being what `run_two_tier` hands a scorer through
    `attach`."""

    time_varying = True

    def __init__(self, admission, reuse):
        self.admission, self.reuse, self.store = admission, reuse, None

    def attach(self, store):
        self.store = store

    def observe(self, requests, timestamp_ms):
        self.admission.observe(requests, timestamp_ms)
        self.reuse.observe(requests, timestamp_ms)

    def score(self, state_id, timestamp_ms):
        return (self.reuse.score(state_id, timestamp_ms), self.store.rng.random())


class RandomArmTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_same_seed_replays_identically_and_another_seed_differs(self):
        for h in (60.0, 600.0):
            arm = mix_arm(RANDOM, h)
            with self.subTest(h=h):
                first = traced_replay(self.trace, _setup(arm, self.trace, seed=1), h, seed=1)
                second = traced_replay(self.trace, _setup(arm, self.trace, seed=1), h, seed=1)
                self.assertEqual(first.tracer.decisions, second.tracer.decisions)
                self.assertEqual(_row(first.result), _row(second.result))
                self.assertEqual(first.statistics.row(), second.statistics.row())
                other = traced_replay(self.trace, _setup(arm, self.trace, seed=3), h, seed=3)
                self.assertNotEqual(first.statistics.row()["decision_sha256"],
                                    other.statistics.row()["decision_sha256"])
                # The stream alone changes the decisions: same sampling seed,
                # another stream seed.
                restreamed = traced_replay(self.trace, classmix.random_setup(
                    arm, self.trace, _ranker(self.trace), h, 2), h, seed=1)
                self.assertNotEqual(_candidates_and_final(first.tracer.decisions),
                                    _candidates_and_final(restreamed.tracer.decisions))

    def test_one_draw_per_candidate_per_decision_in_the_unit_interval(self):
        for h in (60.0, 300.0):
            with self.subTest(h=h):
                setup = _setup(mix_arm(RANDOM, h), self.trace)
                replayed = traced_replay(self.trace, setup, h)
                decisions = replayed.tracer.decisions
                stream = classmix.random_scorer(setup)
                self.assertEqual(stream.draws, sum(len(d[3]) for d in decisions))
                self.assertGreater(stream.draws, len(decisions))
                values = [key[0][1] for d in decisions for key in d[4]]
                self.assertTrue(all(0.0 <= value < 1.0 for value in values))
                # In the store's scoring order, they are the stream's draws.
                expected = random.Random(stream.stream_seed)
                self.assertEqual(values, [expected.random() for _ in values])
                # Each class is ordered by its draws: the store's victim is the
                # first minimum of (bit, u).
                for d in decisions:
                    self.assertEqual(d[5], first_minimum([key[0] for key in d[4]]))

    def test_the_candidate_sets_are_the_recency_arms_when_both_make_the_same_choices(self):
        # With one sampled resident a decision has at most one candidate other
        # than the arrival, so every arm whose admission is the ranker's makes
        # the same choice: the candidate sets must then be the recency arm's,
        # draw for draw of the store's sampling stream.
        for h in (60.0, 600.0):
            with self.subTest(h=h):
                setup = _setup(mix_arm(RANDOM, h), self.trace)
                mine = traced_replay(self.trace, setup, h, width=1, spy=True)
                theirs = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h,
                                       width=1, spy=True)
                self.assertEqual(_candidates_and_final(mine.tracer.decisions),
                                 _candidates_and_final(theirs.tracer.decisions))
                self.assertEqual(mine.spy.states, theirs.spy.states)
                self.assertEqual(mine.spy.rng.calls, theirs.spy.rng.calls)
                self.assertEqual(_row(mine.result), _row(theirs.result))
                self.assertEqual(mine.statistics.row()["decision_sha256"],
                                 theirs.statistics.row()["decision_sha256"])
                self.assertGreater(len(mine.spy.rng.calls), 100)
                self.assertGreater(classmix.random_scorer(setup).draws, len(mine.tracer.decisions))
                # The store's own victim does differ (its key orders within the
                # class by the draw): the same choices come from the override.
                self.assertNotEqual(mine.statistics.row()["overridden_decisions_seen"],
                                    theirs.statistics.row()["overridden_decisions_seen"])

    def test_the_check_has_teeth_a_stream_shared_with_the_store_moves_the_candidates(self):
        h = 60.0
        admission = _ranker(self.trace)
        leaky = errorloc.ArmSetup("leaky", "learned",
                                  scorer=_LeakyRandom(admission, matchedorder.reuse_label(self.trace,
                                                                                          h)),
                                  override=HybridOverride(admission))
        mine = traced_replay(self.trace, leaky, h, width=1, spy=True)
        theirs = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h, width=1,
                               spy=True)
        self.assertNotEqual(_candidates_and_final(mine.tracer.decisions),
                            _candidates_and_final(theirs.tracer.decisions))
        self.assertNotEqual(mine.spy.states, theirs.spy.states)
        self.assertIn(("random", "score"), mine.spy.rng.calls)

    def test_the_sampling_stream_is_the_stores_alone(self):
        for h in (60.0, 600.0):
            with self.subTest(h=h):
                state = random.getstate()
                setup = _setup(mix_arm(RANDOM, h), self.trace)
                mine = traced_replay(self.trace, setup, h, spy=True)
                # Python's global stream is not touched either.
                self.assertEqual(random.getstate(), state)
                calls = mine.spy.rng.calls
                self.assertGreater(len(calls), 100)
                self.assertEqual({caller for _, caller in calls}, {"_admit_sampled"})
                self.assertEqual({method for method, _ in calls}, {"sample"})
                self.assertGreater(classmix.random_scorer(setup).draws, 0)
                self.assertIsNot(classmix.random_scorer(setup).rng, mine.spy.rng.rng)

    def test_the_recency_arm_replayed_beside_the_random_arm_is_unchanged(self):
        h = 150.0
        before = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h, seed=2)
        random_arm = traced_replay(self.trace, _setup(mix_arm(RANDOM, h), self.trace, seed=2), h,
                                   seed=2)
        after = traced_replay(self.trace, _matched_setup(h, "recency", self.trace), h, seed=2)
        self.assertEqual(before.tracer.decisions, after.tracer.decisions)
        self.assertEqual(_row(before.result), _row(after.result))
        self.assertEqual(before.statistics.row(), after.statistics.row())
        self.assertNotEqual(_candidates_and_final(random_arm.tracer.decisions),
                            _candidates_and_final(before.tracer.decisions))

    def test_the_stream_seed(self):
        arm = mix_arm(RANDOM, 60)
        self.assertEqual(classmix.random_stream_seed(0, arm),
                         errorloc.hash_bits("class_order_mix/random_within_class", 0, arm))
        self.assertEqual(classmix.RANDOM_STREAM_TAG, "class_order_mix/random_within_class")
        # The rule as the README and run_config state it, recomputed here.
        text = "\x1f".join(("class_order_mix/random_within_class", "3", arm)).encode("utf-8")
        self.assertEqual(classmix.random_stream_seed(3, arm), int.from_bytes(
            hashlib.blake2b(text, digest_size=8).digest(), "big"))
        seeds = {classmix.random_stream_seed(seed, mix_arm(RANDOM, h))
                 for seed in SEEDS for h in MATCHED_HORIZONS_SECONDS}
        self.assertEqual(len(seeds), 20)
        self.assertTrue(all(0 <= value < 2 ** 64 for value in seeds))
        scorer = RandomClassScorer(_Fixed({"s": 1.0}), 4, arm)
        expected = random.Random(classmix.random_stream_seed(4, arm))
        self.assertEqual([scorer.score("s", 0.0) for _ in range(5)],
                         [(1.0, expected.random()) for _ in range(5)])
        self.assertEqual(scorer.draws, 5)


# --- the class statistic ---------------------------------------------------------------------------


class ClassStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_no_arm_violates_its_class_or_overrides_outside_admission(self):
        for arm in ARMS:
            h = classmix.HORIZON_OF_ARM[arm]
            with self.subTest(arm=arm):
                replayed = traced_replay(self.trace, _setup(arm, self.trace), h)
                row = replayed.classes.row()
                self.assertEqual(row["class_violations_seen"], 0)
                self.assertEqual(row["class_overridden_outside_admission_seen"], 0)
                overridden = sum(1 for d in replayed.tracer.decisions if d[5] != d[6])
                self.assertGreater(overridden, 0)
                self.assertEqual(row["class_overridden_seen"], overridden)
                self.assertGreater(row["class_resident_evictions_seen"], 0)
                self.assertEqual(row["class_horizon_seconds"], h)

    def test_the_hooks_are_read_only(self):
        for arm in (mix_arm(MIX_OUT, 60), mix_arm(MIX_IN, 300), mix_arm(RANDOM, 150)):
            with self.subTest(arm=arm):
                h = classmix.HORIZON_OF_ARM[arm]
                hooked = traced_replay(self.trace, _setup(arm, self.trace), h)
                bare = traced_replay(self.trace, _setup(arm, self.trace), h, wrapped=False)
                self.assertEqual(hooked.result.as_row(), bare.result.as_row())
                self.assertEqual(hooked.tracer.decisions, bare.tracer.decisions)


# --- the reading helpers ---------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_class_conditions_at_their_boundaries(self):
        conditions = classmix.class_conditions
        self.assertEqual(conditions(-1.0, -0.5), {"D_in_negative": True,
                                                  "D_out_above_D_in": True, "both": True})
        self.assertEqual(conditions(0.0, 1.0)["D_in_negative"], False)
        self.assertEqual(conditions(0.0, 1.0)["both"], False)
        self.assertEqual(conditions(-1e-12, -1e-12), {"D_in_negative": True,
                                                      "D_out_above_D_in": False, "both": False})
        self.assertEqual(conditions(-0.3, -0.3 + 1e-12)["both"], True)
        self.assertEqual(conditions(math.nan, 1.0), {"D_in_negative": False,
                                                     "D_out_above_D_in": False, "both": False})
        self.assertEqual(conditions(-1.0, math.nan)["both"], False)

    def test_prediction_one(self):
        every = {cell: True for cell in PREDICTION_ONE_CELLS}
        self.assertTrue(classmix.prediction_one_holds(every))
        self.assertTrue(classmix.prediction_one_holds(
            {**every, ("toolagent_trace", 0.02, 4.0): False}))
        self.assertTrue(classmix.prediction_one_holds(
            {(cell[0], str(cell[1]), str(cell[2])): True for cell in PREDICTION_ONE_CELLS}))
        for cell in PREDICTION_ONE_CELLS:
            with self.subTest(cell=cell):
                self.assertFalse(classmix.prediction_one_holds({**every, cell: False}))
                self.assertFalse(classmix.prediction_one_holds(
                    {other: True for other in PREDICTION_ONE_CELLS if other != cell}))
        self.assertFalse(classmix.prediction_one_holds({}))
        self.assertTrue(classmix.in_prediction_one(("conversation_trace", "0.0025", "4.0")))
        self.assertFalse(classmix.in_prediction_one(("conversation_trace", 0.01, 4.0)))

    def test_prediction_two_at_eight(self):
        self.assertFalse(classmix.prediction_two_holds(7))
        self.assertTrue(classmix.prediction_two_holds(8))
        self.assertTrue(classmix.prediction_two_holds(12))

    def test_the_seed_interval_and_additivity_at_their_bounds(self):
        values = [-1.0, -0.5, -0.8, -1.2, -0.6]
        mean, low, high = classmix.seed_interval(values)
        stats = summarize(values)
        self.assertEqual((mean, low, high), (stats["mean"], stats["mean"] - stats["ci95_half"],
                                             stats["mean"] + stats["ci95_half"]))
        self.assertEqual(runner._stats(values)["ci95_half"], stats["ci95_half"])
        self.assertTrue(classmix.additive(low, 0.0, low, high))
        self.assertTrue(classmix.additive(-0.5, high + 0.5, low, high))
        self.assertTrue(classmix.additive(-2.0, 2.0 + mean, low, high))
        self.assertFalse(classmix.additive(low - 1e-9, 0.0, low, high))
        self.assertFalse(classmix.additive(high + 1e-9, 0.0, low, high))
        self.assertFalse(classmix.additive(math.nan, 0.0, low, high))
        single = classmix.seed_interval([-1.0])
        self.assertEqual(single[0], -1.0)
        self.assertTrue(math.isnan(single[1]) and math.isnan(single[2]))
        self.assertFalse(classmix.additive(-0.5, -0.5, single[1], single[2]))
        # A zero-width interval contains its own mean.
        self.assertTrue(classmix.additive(-3.0, -2.0, -5.0, -5.0))

    def test_the_reused_readings(self):
        self.assertIs(classmix.recovery, horizonctl.recovery)
        self.assertIs(classmix.seed_signs, matchedorder.seed_signs)
        self.assertEqual(classmix.recovery(29.0, 2.0, 32.0), 0.9)
        self.assertTrue(math.isnan(classmix.recovery(3.0, 2.0, 2.0)))
        self.assertEqual(classmix.seed_signs([-0.1, -2.0, -0.3, -0.01, -5.0])[3],
                         "consistent_loss")
        self.assertEqual(classmix.seed_signs([-0.1, 0.0, -0.3, -0.01, -5.0])[3], "mixed")
        self.assertEqual(classmix.DIFFERENCES, {
            "D_out": ("mix_out_learned", "matched_recency"),
            "D_in": ("mix_in_learned", "matched_recency"),
            "D_rand": ("random_within_class", "matched_recency"),
            "D_learned": ("matched_learned", "matched_recency"),
            "random_minus_learned": ("random_within_class", "matched_learned")})


# --- the runner's derivations on synthetic rows ----------------------------------------------------

CELL_A = ("conversation_trace", 0.0025, 1.0)    # named, h* = 60
CELL_B = ("toolagent_trace", 0.0025, 4.0)       # not named, h* = 150
CELL_C = ("toolagent_trace", 0.02, 4.0)         # not named, h* = 600
IDENTIFIERS = {"l1_capacity_bytes": 1, "l2_capacity_bytes": 4, "requested_tokens": 1000,
               "l1_avoided_tokens": 50, "absent_compulsory_tokens": 77}
# U in tokens over 1000 requested tokens (points = tokens / 10), plus the seed.
PUBLISHED_TOKENS = {"learned": 200, "evict_label": 500, "label_binary_hstar": 520,
                    "matched_learned": 400, "matched_recency": 450}
# (mix_out, mix_in, random): A: D_out 2, D_in -3, D_rand -1 (not additive);
# B: D_out -2, D_in -3 (additive at both bounds of a zero-width interval);
# C: D_in 0, D_rand mixed around 0.
REPLAY_TOKENS = {CELL_A: (470, 420, 440), CELL_B: (430, 420, 440), CELL_C: (460, 450, 450)}
RANDOM_OFFSETS = {CELL_C: (5, -5, 0, 5, -5)}


def synthetic_published():
    published = {}
    for cell in REPLAY_TOKENS:
        names = {"learned": "learned", "evict_label": "evict_label",
                 **classmix.matched_reference_arms(*cell[1:])}
        for name, arm in names.items():
            for seed in SEEDS:
                tokens = PUBLISHED_TOKENS[name] + seed
                published[("all16", arm) + cell + (seed,)] = {
                    "source": runner.REFERENCES[("all16", arm)], "mechanism": "all16",
                    "arm": arm, "trace": cell[0], "l1_fraction": cell[1],
                    "l2_multiplier": cell[2], "cell": runner.rdp.cell_label(*cell[1:]),
                    "seed": seed, "avoided_prefill_tokens": tokens + 50,
                    "extra_avoided_tokens": tokens, **IDENTIFIERS}
    return published


def _replay_row(cell, arm, seed, tokens, **changes):
    row = {metric: 0.0 for metric in runner.REPLAY_METRICS + runner.CLASS_METRICS}
    kind = classmix.KIND_OF_ARM[arm]
    row.update(trace=cell[0], l1_fraction=cell[1], l2_multiplier=cell[2],
               cell=runner.rdp.cell_label(*cell[1:]), arm=arm, seed=seed, mechanism="all16",
               variant="main", kind=kind, extra_avoided_tokens=tokens,
               avoided_prefill_tokens=tokens + 50, extra_points=tokens / 10.0,
               random_stream_seed=(classmix.random_stream_seed(seed, arm) if kind == RANDOM
                                   else ""),
               random_draws=40 if kind == RANDOM else "", l2_decisions=10, **IDENTIFIERS)
    row.update(changes)
    return row


def synthetic_rows():
    rows = []
    for cell, tokens in REPLAY_TOKENS.items():
        for seed in SEEDS:
            for kind, arm, value in zip(KINDS, cell_arms(*cell[1:]), tokens):
                offset = RANDOM_OFFSETS.get(cell, (0,) * 5)[seed] if kind == RANDOM else 0
                rows.append(_replay_row(cell, arm, seed, value + seed + offset))
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.published = synthetic_published()
        self.tables = runner.mix_tables(self.rows, self.published)
        self.by = {runner._cell_key(entry): entry for entry in self.tables["differences"]}

    def test_the_differences(self):
        expected = {CELL_A: {"D_out": 2.0, "D_in": -3.0, "D_rand": -1.0, "D_learned": -5.0,
                             "random_minus_learned": 4.0},
                    CELL_B: {"D_out": -2.0, "D_in": -3.0, "D_rand": -1.0, "D_learned": -5.0,
                             "random_minus_learned": 4.0},
                    CELL_C: {"D_out": 1.0, "D_in": 0.0, "D_rand": 0.0, "D_learned": -5.0,
                             "random_minus_learned": 5.0}}
        for cell, values in expected.items():
            entry = self.by[cell]
            for name, value in values.items():
                with self.subTest(cell=cell, name=name):
                    self.assertAlmostEqual(entry[f"{name}_points_mean"], value, places=12)
        readings = {cell: {name: self.by[cell][f"{name}_reading"] for name in classmix.DIFFERENCES}
                    for cell in expected}
        self.assertEqual(readings[CELL_A], {"D_out": "consistent_gain", "D_in": "consistent_loss",
                                            "D_rand": "consistent_loss",
                                            "D_learned": "consistent_loss",
                                            "random_minus_learned": "consistent_gain"})
        self.assertEqual(readings[CELL_C]["D_in"], "mixed")
        self.assertEqual(self.by[CELL_C]["D_in_n_zero"], 5)
        self.assertEqual(self.by[CELL_C]["D_rand_seed_signs"], "+-0+-")
        self.assertEqual(readings[CELL_C]["D_rand"], "mixed")
        self.assertEqual(readings[CELL_C]["random_minus_learned"], "consistent_gain")
        seeds = [entry for entry in self.tables["differences_seeds"]
                 if runner._cell_key(entry) == CELL_C]
        self.assertEqual([entry["D_rand_tokens"] for entry in seeds], [5, -5, 0, 5, -5])
        self.assertEqual(len(self.tables["differences_seeds"]), 3 * 5)
        first = next(entry for entry in self.tables["differences_seeds"]
                     if runner._cell_key(entry) == CELL_A and entry["seed"] == 0)
        self.assertEqual((first["D_out_tokens"], first["D_in_tokens"],
                          first["random_minus_learned_tokens"]), (20, -30, 40))
        self.assertAlmostEqual(first["D_in_plus_D_out_points"], -1.0)
        self.assertAlmostEqual(first["U_label_binary_hstar_points"], 52.0)

    def test_the_conditions_additivity_and_recovery(self):
        a, b, c = self.by[CELL_A], self.by[CELL_B], self.by[CELL_C]
        self.assertEqual((a["D_in_negative"], a["D_out_above_D_in"], a["which_class_condition"]),
                         (True, True, True))
        self.assertEqual((b["which_class_condition"], c["D_in_negative"],
                          c["D_out_above_D_in"], c["which_class_condition"]),
                         (True, False, True, False))
        self.assertEqual((a["prediction_1_cell"], b["prediction_1_cell"], c["prediction_1_cell"]),
                         (True, False, False))
        # D_learned is -5 points in every seed: a zero-width interval, which
        # B's D_in + D_out = -5 meets exactly and A's -1 does not.
        self.assertEqual((b["D_learned_interval_low"], b["D_learned_interval_high"]), (-5.0, -5.0))
        self.assertEqual((a["additive"], b["additive"], c["additive"]), (False, True, False))
        self.assertAlmostEqual(a["D_in_plus_D_out_points"], -1.0)
        # R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned)), in tokens / 300.
        for name, tokens in (("mix_out_learned", 470), ("mix_in_learned", 420),
                             ("random_within_class", 440), ("matched_learned", 400),
                             ("matched_recency", 450), ("label_binary_hstar", 520)):
            with self.subTest(name=name):
                self.assertAlmostEqual(a[f"R_{name}"], (tokens - 200) / 300, places=9)
        self.assertNotIn("R_learned", a)
        self.assertEqual((a["h_star"], b["h_star"], c["h_star"]), (60.0, 150.0, 600.0))
        self.assertEqual((b["rung_arm"], b["rung_source"]), ("label_binary_150", "horizon_fill_001"))
        self.assertEqual(c["random_within_class_arm"], "random_within_class_600")

    def test_the_reading_counts(self):
        summary = runner.reading_summary(self.tables)
        by = {(entry["reading"], entry["scope"]): entry for entry in summary}
        named = by[("1_which_class", "prediction_cells")]
        self.assertEqual((named["cells"], named["D_in_negative"], named["both"]), (1, 1, 1))
        # Seven of the eight named cells are absent: the prediction cannot hold.
        self.assertIs(named["prediction_holds"], False)
        self.assertIn("each of the 8 named", named["prediction"])
        every = by[("1_which_class", "all")]
        self.assertEqual((every["cells"], every["D_in_negative"], every["D_out_above_D_in"],
                          every["both"], every["prediction"], every["prediction_holds"]),
                         (3, 2, 3, 2, "", ""))
        self.assertEqual(by[("1_additivity", "all")]["additive"], 1)
        rand = by[("2_D_rand_seed_signs", "all")]
        self.assertEqual((rand["consistent_loss"], rand["mixed"], rand["consistent_gain"]),
                         (2, 1, 0))
        self.assertIs(rand["prediction_holds"], False)
        self.assertEqual(by[("2_random_minus_learned_seed_signs", "all")]["consistent_gain"], 3)
        self.assertEqual(by[("context_D_learned_seed_signs", "all")]["consistent_loss"], 3)
        self.assertEqual(by[("1_D_in_seed_signs", "prediction_cells")]["consistent_loss"], 1)
        self.assertEqual(len(summary), 11)
        self.assertTrue(all(set(entry) == set(summary[0]) for entry in summary))
        self.assertFalse(runner.named_cells_are_published_losses(self.tables))

    def test_prediction_two_on_the_full_grid(self):
        cells = [{"D_rand_reading": "consistent_loss" if index < count else "mixed",
                  "prediction_1_cell": False, "D_in_negative": False, "D_out_above_D_in": False,
                  "which_class_condition": False, "additive": False,
                  **{f"{name}_reading": "mixed" for name in classmix.DIFFERENCES
                     if name != "D_rand"}}
                 for count in (8,) for index in range(12)]
        for count, holds in ((8, True), (7, False)):
            for index, entry in enumerate(cells):
                entry["D_rand_reading"] = "consistent_loss" if index < count else "mixed"
            entry = next(e for e in runner.reading_summary({"differences": cells})
                         if e["reading"] == "2_D_rand_seed_signs")
            self.assertIs(entry["prediction_holds"], holds)
        entry = next(e for e in runner.reading_summary({"differences": cells[:11]})
                     if e["reading"] == "2_D_rand_seed_signs")
        self.assertIs(entry["prediction_holds"], False)

    def test_aggregation_and_reference_rows(self):
        summary = runner.aggregate_replays(self.rows)
        self.assertEqual(len(summary), 9)
        entry = next(e for e in summary if runner._cell_key(e) == CELL_A
                     and e["arm"] == "mix_out_learned_60")
        self.assertEqual((entry["arm_parameter"], entry["kind"], entry["learned_class"]),
                         (60.0, "mix_out_learned", 0))
        self.assertAlmostEqual(entry["extra_points_mean"], 47.2)
        self.assertNotIn("random_draws_mean", entry)
        rand = next(e for e in summary if e["arm"] == "random_within_class_150")
        self.assertEqual((rand["random_draws_mean"], rand["learned_class"]), (40.0, ""))
        references = runner.reference_rows(self.published, self.rows)
        self.assertEqual(len(references), 3 * 5 * 5)
        self.assertEqual({entry["arm"] for entry in references
                          if runner._cell_key(entry) == CELL_B},
                         {"learned", "evict_label", "label_binary_150", "evict_binary_150_learned",
                          "evict_binary_150_recency"})

    def test_the_derivation_on_the_published_matched_rows(self):
        # Feed the published matched rows in as if they were this run's arms:
        # mix_out := the learned order, mix_in and random := recency. Then
        # D_out is the published D_learned seed for seed, D_in and D_rand are
        # zero, and the published matched tables come back cell by cell.
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        rows = []
        for name in runner.TRACES:
            for fraction, multiplier in runner.CELLS:
                references = classmix.matched_reference_arms(fraction, multiplier)
                for seed in runner.SEEDS:
                    for kind, arm in zip(KINDS, cell_arms(fraction, multiplier)):
                        source = references["matched_learned" if kind == MIX_OUT
                                            else "matched_recency"]
                        rows.append(dict(published[("all16", source, name, fraction, multiplier,
                                                    seed)], arm=arm))
        tables = runner.mix_tables(rows, published)
        self.assertEqual(len(tables["differences"]), 12)
        directory = REPOSITORY / "results/paper/matched_class_order_001"
        order = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])): row
                 for row in csv_rows(directory / "order.csv")}
        class_order = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])): row
                       for row in csv_rows(directory / "class_order.csv")}
        for entry in tables["differences"]:
            cell = runner._cell_key(entry)
            with self.subTest(cell=cell):
                reference = order[cell]
                for name in ("D_learned", "D_out"):
                    self.assertAlmostEqual(entry[f"{name}_points_mean"],
                                           float(reference["learned_minus_recency_points_mean"]),
                                           places=12)
                    self.assertEqual(entry[f"{name}_seed_signs"],
                                     reference["learned_minus_recency_seed_signs"])
                    self.assertEqual(entry[f"{name}_reading"],
                                     reference["learned_minus_recency_reading"])
                self.assertEqual((entry["D_in_points_mean"], entry["D_rand_points_mean"]),
                                 (0.0, 0.0))
                self.assertEqual(entry["D_in_reading"], "mixed")
                self.assertAlmostEqual(entry["R_matched_learned"],
                                       float(class_order[cell]["R_matched_learned"]), places=12)
                self.assertAlmostEqual(entry["R_matched_recency"],
                                       float(class_order[cell]["R_matched_recency"]), places=12)
                self.assertEqual(entry["R_mix_out_learned"], entry["R_matched_learned"])
                # With D_in = 0, additivity is D_out = D_learned: inside.
                self.assertIs(entry["additive"], True)
        self.assertTrue(runner.named_cells_are_published_losses(tables))


# --- the runner's checks and refusals --------------------------------------------------------------


class RunnerCheckTests(unittest.TestCase):
    def test_identifiers_against_every_reference_of_the_cell(self):
        published = synthetic_published()
        rows = synthetic_rows()
        compared, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])
        self.assertEqual(compared, len(rows) * 5)
        rows[0]["absent_compulsory_tokens"] = 78
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 5)
        self.assertTrue(all("absent_compulsory_tokens 78 != 77" in line for line in problems))
        rows[0]["absent_compulsory_tokens"] = 77
        del published[("all16", "evict_binary_60_recency") + CELL_A + (1,)]
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 3)
        self.assertTrue(all("no published evict_binary_60_recency/all16" in line
                            for line in problems))

    def test_the_random_check(self):
        rows = synthetic_rows()
        self.assertEqual(runner.check_random(rows), [])
        cases = (
            (dict(kind=RANDOM, random_stream_seed=1), "random stream seed 1 != "),
            (dict(kind=RANDOM, random_draws=9), "9 random draws for 10 decisions"),
            (dict(kind=RANDOM, random_draws=0, l2_decisions=0), "0 random draws for 0"),
        )
        rand = next(row for row in rows if row["kind"] == RANDOM)
        mixed = next(row for row in rows if row["kind"] == MIX_IN)
        for changes, text in cases:
            with self.subTest(case=text):
                problems = runner.check_random([dict(rand, **changes)])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])
        problems = runner.check_random([dict(mixed, random_draws=3)])
        self.assertEqual(len(problems), 1)
        self.assertIn("a mixed arm carries a random stream", problems[0])

    def test_the_statistics_and_class_checks_are_the_matched_runners(self):
        self.assertIs(runner.check_class, runner.mco.check_class)
        self.assertIs(runner.check_statistics, runner.hc.check_statistics)
        row = {"trace": "conversation_trace", "l1_fraction": 0.0025, "l2_multiplier": 1.0,
               "cell": "l1=0.0025,l2x1", "seed": 0, "arm": "random_within_class_60",
               "variant": "main", "arm_parameter": 60.0, "class_horizon_seconds": 60.0,
               "l2_decisions": 10, "l2_rejections": 3, "l2_evictions": 7,
               "stat_decisions_seen": 10, "stat_rejections_seen": 3, "stat_evictions_seen": 7,
               "overridden_decisions_seen": 4, "stat_decisions": 6, "overridden_decisions": 2,
               "overridden_decisions_resident": 0, "m4_count_resident": 0,
               "class_decisions_seen": 10, "class_rejections_seen": 3,
               "class_resident_evictions_seen": 7, "class_violations_seen": 0,
               "class_overridden_seen": 4, "class_overridden_outside_admission_seen": 0,
               "class_decisions": 6, "class_overridden": 2, "class_m4_count_resident": 0}
        self.assertEqual(runner.check_class([row]), [])
        self.assertEqual(runner.check_statistics([row]), [])
        for changes, text in ((dict(class_violations_seen=1), "reusable within 60 s"),
                              (dict(class_overridden_outside_admission_seen=2),
                               "the arrival is not a candidate"),
                              (dict(arm_parameter=150.0), "the cell's h* is 60 s")):
            with self.subTest(case=text):
                problems = runner.check_class([dict(row, **changes)])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])

    def test_the_published_references_load_as_the_plan_names_them(self):
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        self.assertEqual(len(published), 2 * 60 + 120 + 60)
        self.assertEqual(published[("all16", "evict_binary_150_recency", "toolagent_trace", 0.01,
                                    1.0, 3)]["source"], "matched_class_order_001")
        self.assertEqual(published[("all16", "label_binary_150", "toolagent_trace", 0.01, 1.0,
                                    3)]["source"], "horizon_fill_001")
        self.assertNotIn(("all16", "evict_binary_60_learned", "conversation_trace", 0.01, 1.0, 0),
                         published)
        for key, entry in published.items():
            self.assertEqual(set(runner.REFERENCE_IDENTIFIERS) - set(entry), set())
            if key[1].startswith("evict_binary_"):
                self.assertEqual(len(entry["counters_sha256"]), 64)
                self.assertEqual(len(entry["decision_sha256"]), 64)

    def test_reference_loading_keeps_each_reference_to_its_source(self):
        header = ("trace,l1_fraction,l2_multiplier,cell,eligibility,width,mechanism,arm,seed,"
                  "variant,avoided_prefill_tokens,extra_avoided_tokens,l1_capacity_bytes,"
                  "l2_capacity_bytes,requested_tokens,l1_avoided_tokens,counters_sha256\n")
        line = "conversation_trace,{},{},c,{},16,all16,{},0,{},10,5,1,4,100,5,abc\n"
        with tempfile.TemporaryDirectory() as directory:
            paths = {source: Path(directory) / f"{source}.csv" for source in runner.REFERENCE_SOURCES}
            paths["error_location_001"].write_text(
                header + line.format(0.01, 4.0, "all", "learned", "main")
                + line.format(0.01, 4.0, "all", "learned", "main")          # duplicate
                + line.format(0.01, 4.0, "leaf", "evict_label", "main")     # wrong eligibility
                + line.format(0.01, 4.0, "all", "lru", "main")              # not a reference
                + line.format(0.01, 4.0, "all", "evict_binary_600_learned", "main"))  # wrong source
            paths["matched_class_order_001"].write_text(
                header + line.format(0.01, 4.0, "all", "evict_binary_600_learned", "main")
                + line.format(0.01, 1.0, "all", "evict_binary_600_recency", "main")  # not its cell
                + line.format(0.01, 4.0, "all", "evict_binary_600_recency", "nostats"))
            paths["horizon_control_001"].write_text(
                header + line.format(0.01, 4.0, "all", "label_binary_600", "main")
                + line.format(0.01, 4.0, "all", "evict_binary_learned", "main"))  # not a reference
            paths["horizon_fill_001"].write_text(
                header + line.format(0.01, 1.0, "all", "label_binary_150", "main"))
            published, problems = runner.load_references(paths)
        cell = ("conversation_trace", 0.01, 4.0, 0)
        self.assertEqual(published[("all16", "learned") + cell]["source"], "error_location_001")
        self.assertEqual(published[("all16", "evict_binary_600_learned") + cell]["source"],
                         "matched_class_order_001")
        self.assertEqual(published[("all16", "evict_binary_600_learned") + cell]["counters_sha256"],
                         "abc")
        self.assertEqual(published[("all16", "label_binary_600") + cell]["source"],
                         "horizon_control_001")
        self.assertNotIn(("all16", "evict_binary_600_recency") + cell, published)
        self.assertNotIn(("all16", "lru") + cell, published)
        self.assertTrue(any(line.startswith("duplicate") for line in problems))
        self.assertTrue(any("has leaf/16" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))

    def test_argument_refusals_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh, paper = Path(directory) / "new", Path(directory) / "paper"
            existing = Path(directory)
            base = ["a.jsonl", "b.jsonl", "--output-dir", str(fresh), "--paper-dir", str(paper)]
            cases = {"positive": base + ["--workers", "0"],
                     "hard cap of 12": base + ["--workers", "13"],
                     "expected 2 trace files": base[1:],
                     "must differ": ["a.jsonl", "b.jsonl", "--output-dir", str(fresh),
                                     "--paper-dir", str(fresh)],
                     "output directory": ["a.jsonl", "b.jsonl", "--output-dir", str(existing),
                                          "--paper-dir", str(paper)],
                     "paper directory": ["a.jsonl", "b.jsonl", "--output-dir", str(fresh),
                                         "--paper-dir", str(existing)]}
            for text, argv in cases.items():
                with self.subTest(case=text), self.assertRaises(SystemExit) as caught:
                    runner.validate_arguments(runner.parse_args(argv))
                self.assertIn(text, str(caught.exception))
            runner.validate_arguments(runner.parse_args(base + ["--workers", "12"]))
        args = runner.parse_args(["a.jsonl", "b.jsonl", "--output-dir", "x"])
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.paper_dir, REPOSITORY / "results/paper/class_order_mix_001")

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output, paper = Path(directory) / "run", Path(directory) / "paper"
            argv = ["run_class_order_mix.py", "a.jsonl", "b.jsonl", "--output-dir", str(output),
                    "--paper-dir", str(paper)]

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
        for name in ("scripts/run_class_order_mix.py", "scripts/run_matched_class_order.py",
                     "scripts/run_horizon_control.py", "scripts/run_error_location.py",
                     "scripts/run_mechanism_control.py", "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/classmix.py",
                     "src/persistent_kv_admission/matchedorder.py",
                     "src/persistent_kv_admission/horizonctl.py",
                     "src/persistent_kv_admission/errorloc.py"):
            self.assertIn(name, sources)
        self.assertNotIn("scripts/tabulate_class_order_mix.py", sources)

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "differences_seeds",
                     "differences", "readings"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        self.assertIn("`run_config.json`", runner.README_TEXT)
        self.assertIn(classmix.RANDOM_STREAM_TAG, runner.README_TEXT)
        for column in runner.READING_COLUMNS + ("which_class_condition", "prediction_1_cell",
                                                "random_stream_seed", "random_draws"):
            self.assertIn(f"`{column}`", runner.README_TEXT)


# --- the replay worker -----------------------------------------------------------------------------


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker` on a constructed trace, with the module state the
    runner's main would set, against the matched runner's own worker."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        name = self.trace.name
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON},
                      rankers={name: _LinearRanker()})
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.mco.SHARED, shared),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def test_the_worker_is_the_matched_runners_at_the_null_settings(self):
        # The worker's replay call and hooks, given the null settings of the
        # mixed construction in place of an arm, are the matched runner's
        # replay of its two arms, digest for digest.
        for fraction, multiplier in ((0.0025, 1.0), (0.01, 4.0)):
            h = MATCHED_HORIZON_SECONDS[(fraction, multiplier)]
            for classes, order, arm in ((BOTH_CLASSES, "learned", mix_arm(MIX_OUT, h)),
                                        (NEITHER_CLASS, "recency", mix_arm(MIX_IN, h))):
                with self.subTest(cell=(fraction, multiplier), order=order):
                    def null_setup(arm, trace, learned_ranker, seed, classes=classes):
                        return classmix.mixed_setup(arm, trace, RawFeatureScorer(trace,
                                                                                 learned_ranker),
                                                    classes, h)
                    with mock.patch.object(runner, "arm_setup", null_setup):
                        mine = runner._replay_worker((self.name, fraction, multiplier, arm, 1))
                    theirs = runner.mco._replay_worker((self.name, fraction, multiplier,
                                                        matched_arm(h, order), 1))
                    for column in ("counters_sha256", "decision_sha256", "avoided_prefill_tokens",
                                   "m1", "m2", "m4_count") + matchedorder.CLASS_COLUMNS:
                        self.assertEqual(mine[column], theirs[column], column)
                    self.assertGreater(mine["l2_decisions"], 0)

    def test_worker_rows_pass_the_checks(self):
        rows = [runner._replay_worker((self.name, fraction, multiplier, arm, 0))
                for fraction, multiplier in runner.CELLS for arm in cell_arms(fraction, multiplier)]
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_class(rows), [])
        self.assertEqual(runner.check_random(rows), [])
        groups, problems = runner.rmc.check_invariants(rows, 3)
        self.assertEqual((groups, problems), (6, []))
        for row in rows:
            h = MATCHED_HORIZON_SECONDS[(row["l1_fraction"], row["l2_multiplier"])]
            self.assertEqual((row["mechanism"], row["eligibility"], row["width"], row["family"],
                              row["variant"], row["arm_parameter"], row["class_horizon_seconds"]),
                             ("all16", "all", 16, "class_order_mix", "main", h, h))
            self.assertEqual(row["kind"], classmix.KIND_OF_ARM[row["arm"]])
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertGreater(row["stat_decisions"], 0)
            self.assertGreater(row["overridden_decisions_seen"], 0)
            self.assertEqual(row["class_violations_seen"], 0)
            if row["kind"] == RANDOM:
                self.assertEqual(row["random_stream_seed"],
                                 classmix.random_stream_seed(0, row["arm"]))
                self.assertGreater(row["random_draws"], row["l2_decisions"])
                self.assertEqual(row["learned_class"], "")
            else:
                self.assertEqual((row["random_stream_seed"], row["random_draws"]), ("", ""))
                self.assertEqual(row["learned_class"], classmix.LEARNED_CLASS[row["kind"]])
        # The same replay twice is the same row, digest for digest.
        again = runner._replay_worker((self.name, 0.0025, 1.0, "random_within_class_60", 0))
        first = next(row for row in rows if row["arm"] == "random_within_class_60")
        self.assertEqual((again["counters_sha256"], again["decision_sha256"]),
                         (first["counters_sha256"], first["decision_sha256"]))

    def test_the_worker_refuses_an_arm_that_is_not_its_cells(self):
        with self.assertRaises(ValueError):
            runner._replay_worker((self.name, 0.0025, 1.0, "mix_out_learned_600", 0))


# --- the whole run on constructed traces -----------------------------------------------------------


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


def _fake_models(names):
    return ({name: _LinearRanker() for name in names},
            {name: {"path": f"models/pi0/{name}__next_use.json", "sha256": "a" * 64}
             for name in names})


class MainTests(unittest.TestCase):
    """`main` end to end on two constructed traces named as the grid's, with a
    constructed publication of every reference (the matched class-order rows
    replayed by the matched runner's own worker, the other references
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
        with mock.patch.dict(runner.mco.SHARED, shared), \
                mock.patch.dict(runner.rdp._SHARED, {"working_set": working_set}):
            matched_rows = [runner.mco._replay_worker((name, fraction, multiplier, arm, seed))
                            for name in sorted(traces) for fraction, multiplier in runner.CELLS
                            for arm in matchedorder.cell_arms(fraction, multiplier)
                            for seed in SEEDS]
        sources = {source: [] for source in runner.REFERENCE_SOURCES}
        for row in matched_rows:
            sources["matched_class_order_001"].append(row)
            if matchedorder.ORDER_OF_ARM[row["arm"]] != "learned":
                continue
            extra = row["extra_avoided_tokens"]
            cell = (row["l1_fraction"], row["l2_multiplier"])
            rung = matchedorder.RUNG_OF_HORIZON[MATCHED_HORIZON_SECONDS[cell]]
            synthesised = [("error_location_001", "learned", extra // 3),
                           ("error_location_001", "evict_label", extra + 400),
                           (runner.REFERENCES[("all16", rung)], rung, extra + 450),
                           # Rows the loader must skip: a reference this plan
                           # does not read, and a rung at a cell not its own.
                           ("error_location_001", "lru", extra // 4),
                           ("horizon_control_001", "label_binary_300" if rung != "label_binary_300"
                            else "label_binary_60", extra)]
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
        cls.matched_rows = matched_rows
        cls.manifests = (root / "manifest_a.csv", root / "manifest_b.csv")
        for path in cls.manifests:
            path.write_text("policy,target,trace,model_sha256\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _tampered(self, name, source, change) -> dict:
        """The fixture sources with `source`'s rows changed by `change` (a
        function of the rows that edits them in place)."""
        rows = list(csv_rows(self.sources[source]))
        change(rows)
        path = self.root / name / "replay_seeds.csv"
        path.parent.mkdir(parents=True)
        runner._write(path, rows)
        return dict(self.sources, **{source: path})

    def _main(self, argv, sources=None, cells=None, seeds=None) -> str:
        patches = [mock.patch.object(sys, "argv", ["run_class_order_mix.py"]
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
                  "differences_seeds.csv", "differences.csv", "readings.csv", "README.md",
                  "run_config.json"}
        self.assertEqual(set(os.listdir(paper)), tables)
        self.assertEqual(set(os.listdir(output)), tables | {"raw_replays.jsonl"})
        self.assertEqual(len((output / "raw_replays.jsonl").read_text().splitlines()), 180)
        self.assertIn("[180/180]", text)
        self.assertIn("reading 1, which class", text)
        self.assertIn("reading 2, recency beyond the bit", text)
        replays = list(csv_rows(paper / "replay_seeds.csv"))
        self.assertEqual(len(replays), 180)
        self.assertEqual(list(replays[0])[:16],
                         ["trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width",
                          "mechanism", "arm", "family", "arm_parameter", "seed", "variant",
                          "kind", "learned_class", "random_stream_seed", "random_draws"])
        self.assertTrue(all(int(row["class_violations_seen"]) == 0 for row in replays))
        for row in replays:
            if row["kind"] == RANDOM:
                self.assertEqual(int(row["random_stream_seed"]),
                                 classmix.random_stream_seed(int(row["seed"]), row["arm"]))
        counts = {name: len(list(csv_rows(paper / f"{name}.csv")))
                  for name in ("differences", "differences_seeds", "references_seeds",
                               "readings", "replay")}
        self.assertEqual(counts, {"differences": 12, "differences_seeds": 60,
                                  "references_seeds": 60 * 5, "readings": 11, "replay": 36})
        # D_learned is the fixture's matched rows, seed-paired.
        matched = {(row["arm"],) + runner._cell_seed(row): row for row in self.matched_rows}
        for entry in csv_rows(paper / "differences_seeds.csv"):
            key = runner._cell_seed(entry)
            later = matched[(entry["matched_learned_arm"],) + key]["extra_avoided_tokens"]
            earlier = matched[(entry["matched_recency_arm"],) + key]["extra_avoided_tokens"]
            self.assertEqual(int(entry["D_learned_tokens"]), later - earlier)
        readings = {(row["reading"], row["scope"]): row
                    for row in csv_rows(paper / "readings.csv")}
        self.assertEqual(readings[("1_which_class", "prediction_cells")]["cells"], "8")
        self.assertEqual(readings[("1_which_class", "all")]["cells"], "12")
        self.assertEqual(readings[("2_D_rand_seed_signs", "all")]["cells"], "12")
        self.assertIn(readings[("2_D_rand_seed_signs", "all")]["prediction_holds"],
                      ("True", "False"))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["phase"], "class_order_mix")
        self.assertEqual(config["plan"], "docs/class-order-mix-plan.md")
        self.assertEqual((config["plan_commit"], config["code_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(set(config["references"]),
                         {str(path.resolve()) for path in self.sources.values()})
        for path in self.sources.values():
            self.assertEqual(config["references"][str(path.resolve())], sha256_path(path))
        self.assertEqual(set(config["trace_files"]), {"conversation_trace", "toolagent_trace"})
        for name, path in zip(("conversation_trace", "toolagent_trace"), self.trace_paths):
            self.assertEqual(config["trace_files"][name]["sha256"], sha256_path(Path(path)))
        stream = config["random_stream"]
        self.assertEqual(stream["tag"], classmix.RANDOM_STREAM_TAG)
        self.assertEqual(set(stream["stream_seeds"]),
                         {mix_arm(RANDOM, h) for h in MATCHED_HORIZONS_SECONDS})
        self.assertEqual(stream["stream_seeds"]["random_within_class_150"]["3"],
                         classmix.random_stream_seed(3, "random_within_class_150"))
        self.assertEqual(len(config["arms"]), 12)
        checks = config["checks"]
        self.assertEqual((checks["identifier_problems"], checks["statistics_problems"],
                          checks["class_problems"], checks["class_violations"],
                          checks["class_overridden_outside_admission"], checks["random_problems"],
                          checks["invariant_violations"], checks["unexplained_absent_tokens"]),
                         (0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(checks["identifier_pairs_compared"], 180 * 5)
        self.assertGreater(checks["random_draws"], 0)
        self.assertEqual((config["replays"], config["workers"]), (180, 2))
        self.assertIn("named_cells_equal_published_consistent_losses", config)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(tabulate.main([str(paper)]), 0)
        printed = stdout.getvalue()
        for section in ("X0", "X1", "X2", "X3"):
            self.assertIn(f"### {section} ", printed)
        self.assertEqual(sorted(os.listdir(paper)), sorted(tables))

    def test_a_tampered_reference_publishes_nothing(self):
        def matched_identifier(rows):
            target = next(row for row in rows if row["arm"] == "evict_binary_600_recency"
                          and float(row["l1_fraction"]) == 0.01 and row["seed"] == "0"
                          and row["trace"] == "conversation_trace")
            target["absent_compulsory_tokens"] = str(int(target["absent_compulsory_tokens"]) + 1)

        def learned_capacity(rows):
            target = next(row for row in rows if row["arm"] == "learned"
                          and float(row["l1_fraction"]) == 0.01
                          and float(row["l2_multiplier"]) == 4.0 and row["seed"] == "0"
                          and row["trace"] == "toolagent_trace")
            target["l2_capacity_bytes"] = str(int(target["l2_capacity_bytes"]) + 1)

        for name, source, change, column in (
                ("matched", "matched_class_order_001", matched_identifier,
                 "absent_compulsory_tokens"),
                ("learned", "error_location_001", learned_capacity, "l2_capacity_bytes")):
            with self.subTest(case=name):
                sources = self._tampered(f"tampered_{name}", source, change)
                output, paper = self.root / f"run_bad_{name}", self.root / f"paper_bad_{name}"
                with self.assertRaises(SystemExit) as caught:
                    self._main(["--output-dir", str(output), "--paper-dir", str(paper),
                                "--workers", "2"], sources=sources, cells=((0.01, 4.0),),
                               seeds=(0,))
                self.assertIn("an identifier differs", str(caught.exception))
                self.assertIn("nothing derived or published", str(caught.exception))
                self.assertIn("IDENTIFIER", self.stdout)
                self.assertIn(column, self.stdout)
                self.assertFalse(paper.exists())
                self.assertEqual(os.listdir(output), ["raw_replays.jsonl"])

    def test_missing_references_are_refused_before_any_replay(self):
        def drop(rows):
            rows[:] = [row for row in rows if not (row["arm"] == "evict_binary_60_learned"
                                                   and row["seed"] == "4")]

        sources = self._tampered("missing", "matched_class_order_001", drop)
        output, paper = self.root / "run_refused", self.root / "paper_refused"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper)], sources=sources)
        self.assertIn("nothing run", str(caught.exception))
        self.assertIn("REFERENCE", self.stdout)
        self.assertIn("evict_binary_60_learned", self.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(paper.exists())

    def test_a_cell_subset_runs_and_publishes(self):
        output, paper = self.root / "run_subset", self.root / "paper_subset"
        self._main(["--output-dir", str(output), "--paper-dir", str(paper), "--workers", "2"],
                   cells=((0.0025, 4.0), (0.02, 4.0)), seeds=(0, 1))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["replays"], 2 * 2 * 3 * 2)
        self.assertEqual(set(config["random_stream"]["stream_seeds"]),
                         {"random_within_class_150", "random_within_class_600"})
        readings = {(row["reading"], row["scope"]): row
                    for row in csv_rows(paper / "readings.csv")}
        # Not the full grid: neither prediction can hold.
        self.assertEqual(readings[("1_which_class", "prediction_cells")]["prediction_holds"],
                         "False")
        self.assertEqual(readings[("2_D_rand_seed_signs", "all")]["prediction_holds"], "False")


# --- the tabulation --------------------------------------------------------------------------------


class TabulationTests(unittest.TestCase):
    def test_tabulation_reads_and_prints_without_writing(self):
        rows = synthetic_rows()
        published = synthetic_published()
        tables = runner.mix_tables(rows, published)
        written = {"replay": runner.aggregate_replays(rows),
                   "differences": tables["differences"],
                   "readings": runner.reading_summary(tables)}
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
        for section in ("X0", "X1", "X2", "X3"):
            self.assertIn(f"### {section} ", text)
        which = tabulate.which_class_table(parsed["differences"])
        self.assertEqual([(row["trace"], row["l1_fraction"]) for row in which],
                         [("conversation_trace", 0.0025), ("toolagent_trace", 0.0025),
                          ("toolagent_trace", 0.02)])
        self.assertEqual((which[0]["named"], which[0]["condition"], which[0]["additive"]),
                         (True, True, False))
        self.assertEqual((which[1]["additive"], which[1]["low"], which[1]["high"]),
                         (True, -5.0, -5.0))
        self.assertAlmostEqual(which[0]["D_in"], -3.0)
        self.assertEqual(which[0]["D_in_signs"], (0, 0, 5))
        recency = tabulate.recency_table(parsed["differences"])
        self.assertEqual(recency[2]["D_rand_signs"], (2, 1, 2))
        self.assertEqual(recency[2]["D_rand_reading"], "mixed")
        arms = tabulate.arms_table(parsed["differences"])
        self.assertAlmostEqual(arms[0]["R_mix_out_learned"], 0.9, places=9)
        self.assertAlmostEqual(arms[0]["U_matched_recency"], 45.2)
        counts = {(row["reading"], row["scope"]): row
                  for row in tabulate.reading_counts(parsed["readings"], ("1_", "2_", "context_"))}
        self.assertEqual(counts[("1_which_class", "all")]["both"], 2)
        self.assertEqual(counts[("1_which_class", "all")]["consistent_loss"], "")
        self.assertIs(counts[("1_which_class", "prediction_cells")]["holds"], False)
        self.assertEqual(counts[("1_which_class", "all")]["holds"], "")
        self.assertEqual(counts[("2_D_rand_seed_signs", "all")]["consistent_loss"], 2)
        replay = tabulate.replay_counters(parsed["replay"])
        self.assertEqual([row["arm"] for row in replay][:3],
                         ["mix_out_learned_60", "mix_in_learned_60", "random_within_class_60"])
        self.assertTrue(math.isnan(replay[0]["random_draws_mean"]))
        self.assertEqual(replay[2]["random_draws_mean"], 40.0)


if __name__ == "__main__":
    unittest.main()
