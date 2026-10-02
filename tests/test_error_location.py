"""Tests for the error-location control (docs/error-location-plan.md).

No real trace is replayed: the plan forbids a real replay of the new arms
before the implementation is reviewed. Everything here runs on constructed
traces and hand-made rows. What has to hold before a real replay means
anything: every arm is the score the plan names (the reference rungs as the
mechanism control ran them, the exact `binary` label, the hybrids' admission
rule, label + s·z, the two swaps); the null settings of each construction are
the label or rung itself decision by decision; the noise and swap draws are
keyed hashes that never touch the store's sampling RNG; the statistics hook is
read-only and records the final victim of an overridden decision; m1-m4 are
what the plan defines, on hand-built candidate sets; and the reading helpers
apply the pre-registered rules at their boundaries. The runner's derivation,
checks and the read-only tabulation are run on synthetic rows.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import persistent_kv_admission
from persistent_kv_admission import errorloc
from persistent_kv_admission.decisionpop import RawFeatureScorer, _Labeller, target_column
from persistent_kv_admission.errorloc import (
    ARMS,
    IDENTITY_ARMS,
    PRIMARY_ARMS,
    SWAP_ARMS,
    DecisionStatistics,
    HybridOverride,
    HybridScorer,
    NoisyScorer,
    RecordingOverride,
    SwapOverride,
    arm_setup,
    noise_z,
    swap_pick,
    swap_uniform,
)
from persistent_kv_admission.mechanism import ExactLabelScorer, sign_reading
from persistent_kv_admission.trace import StateMeta, Trace
from persistent_kv_admission.twotier import run_two_tier

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(Path(list(persistent_kv_admission.__path__)[0]).resolve().parent)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The mechanism control's fixtures (real-shaped trace, learned-like ranker),
# reused as they are; only plain functions and classes are read from it.
mechanism_tests = _load("_mechanism_control_fixtures", REPOSITORY / "tests/test_mechanism_control.py")
runner = _load("_run_error_location_under_test", REPOSITORY / "scripts/run_error_location.py")
tabulate = _load("_tabulate_error_location_under_test",
                 REPOSITORY / "scripts/tabulate_error_location.py")

L1_BYTES = mechanism_tests.L1_BYTES
L2_BYTES = mechanism_tests.L2_BYTES
MEASURE_FROM_MS = mechanism_tests.MEASURE_FROM_MS
HORIZON = mechanism_tests.LABEL_HORIZON
_LinearRanker = mechanism_tests._LinearRanker


class _OtherRanker(_LinearRanker):
    """A second fixed linear score, for the real-ranker arms."""

    COEFFICIENTS = tuple(-0.5 * c if i % 2 else 1.5 * c
                         for i, c in enumerate(_LinearRanker.COEFFICIENTS))


# --- fixtures -------------------------------------------------------------------------------


def jittered(records, seed=3):
    """The same records with every distinct timestamp moved by a random
    sub-second offset (equal timestamps stay equal, the order is kept): no
    next use then lies exactly at the horizon of a decision."""
    rng = random.Random(seed)
    offsets: dict[float, float] = {}
    out = []
    for record in records:
        base = record["timestamp"]
        offset = offsets.setdefault(base, rng.uniform(1.0, 999.0))
        out.append(dict(record, timestamp=base + offset))
    return out


def partial_records(steps=360, seed=5):
    """Sessions of full blocks, each request ending in a fresh partial block;
    three in ten requests repeat an earlier request exactly. Blocks of many
    sizes share the store, so an arriving full block can need more than one
    eviction round: resident-only decisions occur."""
    rng = random.Random(seed)
    sessions, made, out = [], [], []
    next_id = 5000
    for step in range(steps):
        if made and rng.random() < 0.3:
            ids, length = made[rng.randrange(len(made))]
        else:
            if sessions and rng.random() < 0.6:
                chain = sessions[rng.randrange(len(sessions))]
                chain.append(next_id)
                next_id += 1
                if len(chain) > 8:
                    sessions.remove(chain)
            else:
                chain = [1, 2 if rng.random() < 0.5 else 3, next_id]
                next_id += 1
                sessions.append(chain)
            ids = list(chain) + [next_id]
            next_id += 1
            length = 512 * len(chain) + rng.randint(32, 480)
            made.append((ids, length))
        timestamp = (step - (1 if step % 7 == 6 else 0)) * 1000
        out.append({"timestamp": timestamp, "input_length": length, "output_length": 1,
                    "hash_ids": list(ids)})
    return out


def build(records):
    temporary, trace, path = mechanism_tests.build_trace(records)
    return temporary, trace, path


class _Tracer:
    """Override-slot wrapper: records every decision with the store's original
    and the final victim, and passes the inner override's answer through."""

    def __init__(self, inner=None):
        self.inner = inner
        self.decisions = []
        self.returned = 0
        self.l1 = None

    def attach(self, l1, l2, counters):
        self.l1 = l1
        if self.inner is not None and hasattr(self.inner, "attach"):
            self.inner.attach(l1, l2, counters)

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index,
                 arriving_index):
        choice = None if self.inner is None else self.inner(
            candidates, keys, victim_index, timestamp_ms, group_index, arriving_index)
        self.returned += choice is not None
        final = victim_index if choice is None else choice
        self.decisions.append((timestamp_ms, group_index, arriving_index, tuple(candidates),
                               tuple(tuple(key) for key in keys), victim_index, final))
        return choice


def replay(trace, arm, seed=1, width=4, statistics=True, traced=True,
           measure_from_ms=MEASURE_FROM_MS, setup=None):
    """One replay of `arm` on a test trace; returns (result, statistics, tracer, setup)."""
    setup = setup or arm_setup(arm, trace, HORIZON, seed, learned_ranker=_LinearRanker(),
                               real_ranker=_OtherRanker())
    stats = DecisionStatistics(trace, HORIZON, measure_from_ms) if statistics else None
    override = RecordingOverride(stats, setup.override) if statistics else setup.override
    tracer = _Tracer(override) if traced else None
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=measure_from_ms, l2_eviction="sampled",
                          l2_sample_width=width, l2_seed=seed, l2_arm=arm,
                          l2_override_hook=tracer if traced else override,
                          **setup.replay_arguments())
    return result, stats, tracer, setup


def _row(result):
    row = result.as_row()
    row.pop("l2_arm")
    return row


def _label(trace, state_id, timestamp_ms, target="next_use"):
    delta, count = _Labeller(trace, HORIZON)(state_id, timestamp_ms)
    return float(target_column(np.asarray([delta]), np.asarray([count], dtype=float), target,
                               HORIZON)[0])


# --- the arms -------------------------------------------------------------------------------


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, _ = build(mechanism_tests.conversation_records(60))
        self.addCleanup(self.temporary.cleanup)

    def _setup(self, arm, seed=0):
        return arm_setup(arm, self.trace, HORIZON, seed, learned_ranker=_LinearRanker(),
                         real_ranker=_OtherRanker())

    def test_the_eighteen_arms_and_the_primary_set(self):
        self.assertEqual(ARMS, (
            "lru", "learned", "label", "offline", "pi3_next_use", "pi0_binary", "pi3_binary",
            "label_binary", "adm_label", "evict_label", "noise_0.5", "noise_1", "noise_2",
            "noise_4", "swap_uniform_0.25", "swap_uniform_0.5", "swap_runnerup_0.25",
            "swap_runnerup_0.5"))
        self.assertEqual(tabulate.ARMS, ARMS)
        # The primary set is exactly the arms whose victim is the store's first
        # minimum of one key: no override.
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(self._setup(arm).override is None, arm in PRIMARY_ARMS)
        self.assertEqual(len(PRIMARY_ARMS), 12)
        self.assertEqual(set(IDENTITY_ARMS.values()), {"learned", "label"})

    def test_reference_rungs_are_the_mechanism_controls(self):
        for rung in ("lru", "learned", "label", "offline"):
            with self.subTest(rung=rung):
                theirs = mechanism_tests.rung_arguments(self.trace, rung)
                mine = self._setup(rung).replay_arguments()
                self.assertEqual(set(mine), set(theirs))
                self.assertEqual(mine["l2_policy"], theirs["l2_policy"])
                if "l2_scorer" in theirs:
                    self.assertIs(type(mine["l2_scorer"]), type(theirs["l2_scorer"]))
        self.assertEqual(self._setup("label").scorer.target, "next_use")
        self.assertEqual(self._setup("label_binary").scorer.target, "binary")
        self.assertIsInstance(self._setup("pi3_binary").scorer, RawFeatureScorer)
        self.assertIsInstance(self._setup("pi3_binary").scorer.ranker, _OtherRanker)

    def test_arm_parameters_are_parsed(self):
        self.assertEqual(self._setup("noise_0.5").scorer.scale, 0.5)
        self.assertEqual(self._setup("noise_4", seed=3).scorer.seed, 3)
        swap = self._setup("swap_runnerup_0.25", seed=2).override
        self.assertEqual((swap.kind, swap.probability, swap.seed), ("runnerup", 0.25, 2))
        hybrid = self._setup("adm_label")
        self.assertIsInstance(hybrid.scorer.admission, ExactLabelScorer)
        self.assertIsInstance(hybrid.scorer.eviction, RawFeatureScorer)
        self.assertIs(hybrid.override.admission, hybrid.scorer.admission)
        evict = self._setup("evict_label")
        self.assertIsInstance(evict.scorer.admission, RawFeatureScorer)
        self.assertIsInstance(evict.scorer.eviction, ExactLabelScorer)
        same = self._setup("hybrid_learned_learned")
        self.assertIs(same.scorer.admission, same.scorer.eviction)

    def test_unknown_arms_and_missing_models_are_refused(self):
        for arm in ("noise_x", "swap_sideways_0.5", "label_count", "pi2_next_use"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self._setup(arm)
        with self.assertRaises(ValueError):
            arm_setup("learned", self.trace, HORIZON, 0)
        with self.assertRaises(ValueError):
            arm_setup("pi0_binary", self.trace, HORIZON, 0, learned_ranker=_LinearRanker())
        with self.assertRaises(ValueError):
            SwapOverride("uniform", 1.5, 0)
        with self.assertRaises(ValueError):
            NoisyScorer(ExactLabelScorer(self.trace, HORIZON), -1.0, 0)


class LabelBinaryTests(unittest.TestCase):
    HORIZON = 600.0

    def setUp(self):
        stamps = (0, 1000, 1000, 5000, 605000, 1205001)
        records = [{"timestamp": t, "input_length": 512, "output_length": 1, "hash_ids": [1]}
                   for t in stamps]
        self.temporary, self.trace, _ = build(records)
        self.addCleanup(self.temporary.cleanup)
        self.scorer = ExactLabelScorer(self.trace, self.HORIZON, target="binary")

    def test_it_is_the_binary_training_target(self):
        labeller = _Labeller(self.trace, self.HORIZON)
        rng = np.random.default_rng(4)
        instants = np.concatenate([rng.uniform(-1000.0, 1_300_000.0, 400),
                                   [0.0, 1000.0, 5000.0, 605000.0, 1205001.0]])
        pairs = [labeller("int:1", float(t)) for t in instants]
        expected = target_column(np.asarray([p[0] for p in pairs]),
                                 np.asarray([p[1] for p in pairs], dtype=float), "binary",
                                 self.HORIZON)
        mine = np.asarray([self.scorer.score("int:1", float(t)) for t in instants])
        np.testing.assert_array_equal(mine, expected)

    def test_requested_within_the_horizon_includes_exactly_h(self):
        self.assertEqual(self.scorer.score("int:1", 5000.0), 1.0)        # next use 600 s later
        self.assertEqual(self.scorer.score("int:1", 605000.0), 0.0)      # 600.001 s later
        self.assertEqual(self.scorer.score("int:1", 1205001.0), 0.0)     # no further use


# --- keyed randomness -----------------------------------------------------------------------


class KeyedRandomnessTests(unittest.TestCase):
    def test_noise_is_a_function_of_seed_state_and_timestamp(self):
        before = (random.getstate(), np.random.get_state()[1].copy())
        z = noise_z(3, "int:7", 1500.0)
        self.assertEqual(noise_z(3, "int:7", 1500.0), z)
        self.assertEqual(noise_z(3, "int:7", 1500), z)          # timestamps as floats
        self.assertNotEqual(noise_z(3, "int:7", 1501.0), z)     # fresh at every timestamp
        self.assertNotEqual(noise_z(4, "int:7", 1500.0), z)
        self.assertNotEqual(noise_z(3, "int:8", 1500.0), z)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1])

    def test_noise_is_standard_normal(self):
        draws = np.asarray([noise_z(0, f"int:{i}", float(i % 97) * 1000.0)
                            for i in range(20000)])
        self.assertLess(abs(draws.mean()), 0.03)
        self.assertLess(abs(draws.std() - 1.0), 0.03)
        self.assertTrue(np.all(np.isfinite(draws)))

    def test_the_swap_coin_and_pick(self):
        before = random.getstate()
        coins = np.asarray([swap_uniform(1, index) for index in range(20000)])
        self.assertTrue(np.all((coins >= 0.0) & (coins < 1.0)))
        self.assertAlmostEqual(float((coins < 0.25).mean()), 0.25, delta=0.015)
        self.assertAlmostEqual(float((coins < 0.5).mean()), 0.5, delta=0.015)
        self.assertEqual(swap_uniform(1, 17), swap_uniform(1, 17))
        self.assertNotEqual(swap_uniform(1, 17), swap_uniform(2, 17))
        picks = [swap_pick(1, index, 5) for index in range(20000)]
        self.assertEqual(set(picks), set(range(5)))
        counts = np.bincount(picks, minlength=5)
        self.assertLess(counts.max() - counts.min(), 400)
        with self.assertRaises(ValueError):
            swap_pick(1, 0, 0)
        self.assertEqual(random.getstate(), before)


# --- the scorers and overrides on hand-built decisions ----------------------------------------


class _Recording:
    """A scorer that returns fixed values and counts what it was shown."""

    time_varying = True

    def __init__(self, scores):
        self.scores = scores
        self.observed = 0

    def observe(self, requests, timestamp_ms):
        self.observed += 1

    def score(self, state_id, timestamp_ms):
        return float(self.scores[state_id])


class ScorerTests(unittest.TestCase):
    def test_noise_adds_s_times_z_and_forwards_history(self):
        base = _Recording({"a": -2.0, "b": -6.0})
        scorer = NoisyScorer(base, 2.0, seed=5)
        self.assertEqual(scorer.score("a", 7000.0), -2.0 + 2.0 * noise_z(5, "a", 7000.0))
        self.assertEqual(NoisyScorer(base, 0.0, 5).score("b", 7000.0), -6.0)
        scorer.observe([], 0.0)
        self.assertEqual(base.observed, 1)

    def test_the_hybrid_scorer_is_y_and_shows_both_their_history(self):
        x, y = _Recording({"a": 1.0}), _Recording({"a": 9.0})
        scorer = HybridScorer(x, y)
        self.assertEqual(scorer.score("a", 0.0), 9.0)
        scorer.observe([], 0.0)
        self.assertEqual((x.observed, y.observed), (1, 1))
        same = HybridScorer(x, x)
        same.observe([], 0.0)
        self.assertEqual(x.observed, 2)          # one scorer in both roles: observed once


class HybridOverrideTests(unittest.TestCase):
    def test_admission_is_x_and_eviction_is_y(self):
        # Y's keys (the store's): (score, last_group). Arrival first.
        keys = [(5.0, 3.0), (1.0, 2.0), (1.0, 1.0), (4.0, 0.0)]
        candidates = ["arr", "r1", "r2", "r3"]
        rejecting = HybridOverride(_Recording({"arr": 0.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}))
        self.assertEqual(rejecting(candidates, keys, 2, 0.0, 0, 0), 0)
        keeping = HybridOverride(_Recording({"arr": 9.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}))
        # Y's first minimum over the residents: r2, (1.0, 1.0) < (1.0, 2.0).
        self.assertEqual(keeping(candidates, keys, 2, 0.0, 0, 0), 2)
        # Y ties broken in draw order.
        tied = [(5.0, 3.0), (1.0, 1.0), (1.0, 1.0), (4.0, 0.0)]
        self.assertEqual(keeping(candidates, tied, 1, 0.0, 0, 0), 1)
        # Later rounds keep the store's (Y's) choice.
        self.assertIsNone(rejecting(candidates, keys, 2, 0.0, 0, -1))
        # A lone arrival is its own first minimum.
        self.assertEqual(keeping(["arr"], [(5.0, 3.0)], 0, 0.0, 0, 0), 0)

    def test_x_ties_are_broken_by_last_group_from_y_keys_then_draw_order(self):
        x = _Recording({"arr": 1.0, "r1": 1.0})
        override = HybridOverride(x)
        # Equal X scores; the arrival's last_group is larger, so r1 is X's minimum.
        self.assertEqual(override(["arr", "r1"], [(0.0, 5.0), (9.0, 2.0)], 0, 0.0, 0, 0), 1)
        # Equal X keys altogether: the arrival, first in draw order, is rejected.
        self.assertEqual(override(["arr", "r1"], [(0.0, 2.0), (9.0, 2.0)], 0, 0.0, 0, 0), 0)
        with self.assertRaises(ValueError):
            override(["arr", "r1"], [(0.0,), (9.0,)], 0, 0.0, 0, 0)


class SwapOverrideTests(unittest.TestCase):
    KEYS = [(-6.4, 3.0), (-1.0, 2.0), (-6.4, 3.0), (-6.4, 1.0), (-2.0, 9.0)]

    def test_the_runner_up_is_second_by_kstar_in_draw_order(self):
        swap = SwapOverride("runnerup", 1.0, 0)
        victim = errorloc.first_minimum(self.KEYS)
        self.assertEqual(victim, 3)
        # (-6.4, 3.0) at 0 and 2 tie: the earlier draw is the runner-up.
        self.assertEqual(swap(list("abcde"), self.KEYS, victim, 0.0, 0, 0), 0)
        self.assertEqual(errorloc.reference_order(self.KEYS), [3, 0, 2, 4, 1])

    def test_the_uniform_swap_never_keeps_the_victim_and_covers_the_others(self):
        for victim in range(5):
            swap = SwapOverride("uniform", 1.0, seed=victim)
            picks = [swap(list("abcde"), self.KEYS, victim, 0.0, 0, -1) for _ in range(4000)]
            self.assertNotIn(victim, picks)
            counts = np.bincount(picks, minlength=5)
            others = [counts[index] for index in range(5) if index != victim]
            self.assertTrue(all(800 <= count <= 1200 for count in others), counts)

    def test_the_rate_the_null_and_the_single_candidate(self):
        swap = SwapOverride("uniform", 0.25, 7)
        answers = [swap(list("abcde"), self.KEYS, 3, 0.0, 0, 0) for _ in range(8000)]
        rate = sum(answer is not None for answer in answers) / len(answers)
        self.assertAlmostEqual(rate, 0.25, delta=0.02)
        self.assertEqual(swap.swaps, sum(answer is not None for answer in answers))
        # The swapped decision indices are the coin's, and nested across p.
        quarter = [index for index, answer in enumerate(answers) if answer is not None]
        self.assertEqual(quarter, [index for index in range(8000) if swap_uniform(7, index) < 0.25])
        half = SwapOverride("runnerup", 0.5, 7)
        halves = [half(list("abcde"), self.KEYS, 3, 0.0, 0, 0) for _ in range(8000)]
        self.assertTrue(set(quarter) <= {i for i, answer in enumerate(halves) if answer is not None})
        never = SwapOverride("uniform", 0.0, 7)
        self.assertTrue(all(never(list("ab"), self.KEYS[:2], 0, 0.0, 0, 0) is None
                            for _ in range(500)))
        lone = SwapOverride("uniform", 1.0, 7)
        self.assertIsNone(lone(["a"], [(-1.0, 0.0)], 0, 0.0, 0, 0))
        self.assertEqual(lone.decisions, 1)      # counted, never swapped

    def test_the_decision_index_is_checked_against_the_store(self):
        swap = SwapOverride("uniform", 0.5, 0)
        swap.attach(None, SimpleNamespace(decisions=1), None)
        self.assertIsNone(swap(["a"], [(0.0, 0.0)], 0, 0.0, 0, 0))
        swap.store.decisions = 5                 # the store moved on without the hook
        with self.assertRaises(AssertionError):
            swap(["a", "b"], [(0.0, 0.0), (1.0, 0.0)], 0, 0.0, 0, 0)


# --- the statistics on hand-built candidate sets ------------------------------------------------


class StatisticsDefinitionTests(unittest.TestCase):
    """Labels at t = 1000 ms with H = 30 s: a next 10 s on (reusable), b 20 s on
    (reusable), h exactly 30 s on (clipped label, but reusable), c and d never
    again (clipped, not reusable)."""

    T = 1000.0

    def setUp(self):
        occurrences = {"a": [0.0, 11000.0], "b": [0.0, 21000.0], "h": [0.0, 31000.0],
                       "c": [0.0], "d": [0.0]}
        states = {name: StateMeta(state_id=name, parent_id=None, depth=1, prefix_tokens=1,
                                  block_tokens=1) for name in occurrences}
        self.trace = Trace(name="hand", requests=[], states=states, occurrences_ms=occurrences,
                           children={}, terminal_branches={}, block_size=1)
        self.last_group = {"a": 5, "b": 4, "c": 3, "d": 3, "h": 2}
        self.stats = DecisionStatistics(self.trace, HORIZON, measure_from_ms=900.0)
        self.stats.attach(SimpleNamespace(last_group=self.last_group))
        self.labels = {name: _label(self.trace, name, self.T) for name in occurrences}

    def record(self, candidates, keys, original, final, arriving, timestamp=None):
        keys = [tuple(key) for key in keys]
        snapshot = (list(candidates), list(keys))
        self.stats.record(candidates, keys, original, final, self.T if timestamp is None
                          else timestamp, 0, arriving)
        self.assertEqual((candidates, keys), snapshot)          # read-only

    def test_the_labels_of_the_fixture(self):
        clipped = -math.log1p(HORIZON)
        self.assertEqual(self.labels["c"], clipped)
        self.assertEqual(self.labels["h"], clipped)             # exactly H: clipped
        self.assertEqual(self.labels["a"], -math.log1p(10.0))

    def test_m1_to_m4_on_six_decisions(self):
        # 1. Admission. K*: c = d < b < a; (c, d) excluded from m1. Arm keys
        #    c 1, a 3, d 0.5, b 3: four concordant pairs, (a, b) tied. Victim d,
        #    the arm's first minimum; the reference victim is c (first in draw
        #    order of the K* tie), so m2 misses; label(d) is the minimum.
        self.record(["c", "a", "d", "b"], [(1.0,), (3.0,), (0.5,), (3.0,)], 2, 2, 0)
        # 2. Resident-only. K*: h < c < b (h's last_group is smaller). Arm keys
        #    b 2, h 1, c 1: (h, c) tied. Victim h = reference; h is reusable
        #    (next use exactly at H) while c is not: m4, at the horizon.
        self.record(["b", "h", "c"], [(2.0,), (1.0,), (1.0,)], 1, 1, -1)
        # 3. Admission with a single candidate, rejected.
        self.record(["a"], [(5.0,)], 0, 0, 0)
        # 4. Before the window: seen, digested, counted as a rejection, no statistic.
        self.record(["c", "a"], [(0.0,), (1.0,)], 0, 0, 0, timestamp=500.0)
        # 5. Admission overridden from a to b: the final victim b is the reference.
        #    K* a > b while the arm key a < b: discordant.
        self.record(["a", "b"], [(1.0,), (2.0,)], 0, 1, 0)
        # 6. Resident-only, victim a while c (not reusable) has the minimum
        #    label: discordant, m2 misses, m3 = label(a) - label(c), m4 counts.
        self.record(["c", "a"], [(9.0,), (1.0,)], 1, 1, -1)
        row = self.stats.row()
        excess = self.labels["a"] - self.labels["c"]
        self.assertGreater(excess, 0.0)
        expected = {
            "stat_decisions_seen": 6, "stat_rejections_seen": 2, "stat_evictions_seen": 4,
            "overridden_decisions_seen": 1,
            "m1_pairs": 10, "m1_concordant": 6, "m1_tied": 2, "m1": 0.7,
            "stat_decisions": 5, "stat_decisions_admission": 3, "stat_decisions_resident": 2,
            "single_candidate_decisions": 1, "single_candidate_decisions_admission": 1,
            "single_candidate_decisions_resident": 0,
            "overridden_decisions": 1, "overridden_decisions_admission": 1,
            "overridden_decisions_resident": 0,
            "m2_agree": 3, "m2": 0.6, "m2_agree_admission": 2, "m2_admission": 2 / 3,
            "m2_agree_resident": 1, "m2_resident": 0.5,
            "m3_excess_sum": excess, "m3": excess / 5, "m3_excess_sum_admission": 0.0,
            "m3_admission": 0.0, "m3_excess_sum_resident": excess, "m3_resident": excess / 2,
            "m4_count": 2, "m4": 0.4, "m4_count_admission": 0, "m4_admission": 0.0,
            "m4_count_resident": 2, "m4_resident": 1.0,
            "m4_victim_at_horizon": 1, "m4_victim_at_horizon_admission": 0,
            "m4_victim_at_horizon_resident": 1,
        }
        for column, value in expected.items():
            with self.subTest(column=column):
                if isinstance(value, float):
                    self.assertAlmostEqual(row[column], value, places=12)
                else:
                    self.assertEqual(row[column], value)
        self.assertEqual(len(row["decision_sha256"]), 64)

    def test_an_empty_window_gives_nan_ratios(self):
        self.record(["c", "a"], [(0.0,), (1.0,)], 0, 0, 0, timestamp=500.0)
        row = self.stats.row()
        self.assertEqual(row["stat_decisions"], 0)
        self.assertTrue(all(math.isnan(row[c]) for c in ("m1", "m2", "m3", "m4",
                                                         "m2_resident")))

    def test_the_digest_is_of_the_final_victim(self):
        other = DecisionStatistics(self.trace, HORIZON, measure_from_ms=900.0)
        other.attach(SimpleNamespace(last_group=self.last_group))
        self.stats.record(["a", "b"], [(1.0,), (2.0,)], 0, 1, self.T, 0, 0)
        other.record(["a", "b"], [(1.0,), (2.0,)], 0, 0, self.T, 0, 0)
        self.assertNotEqual(self.stats.row()["decision_sha256"], other.row()["decision_sha256"])

    def test_a_candidate_without_last_group_is_refused(self):
        unattached = DecisionStatistics(self.trace, HORIZON, measure_from_ms=None)
        with self.assertRaises(RuntimeError):
            unattached.record(["a"], [(0.0,)], 0, 0, self.T, 0, 0)
        self.stats.attach(SimpleNamespace(last_group={"a": 1}))
        with self.assertRaises(RuntimeError):
            self.stats.record(["a", "b"], [(0.0,), (1.0,)], 0, 0, self.T, 0, 0)

    def test_as_a_plain_decision_hook_the_victim_is_final(self):
        self.stats(["c", "a"], [(1.0,), (0.0,)], 1, self.T, 0, -1)
        row = self.stats.row()
        self.assertEqual((row["stat_decisions_resident"], row["overridden_decisions"]), (1, 0))


# --- the statistics on replays --------------------------------------------------------------------


class StatisticsReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, _ = build(jittered(partial_records()))
        self.addCleanup(self.temporary.cleanup)

    def test_resident_only_decisions_occur_on_this_fixture(self):
        result, stats, tracer, _ = replay(self.trace, "lru")
        row = stats.row()
        self.assertGreater(row["stat_decisions_resident"], 10)
        self.assertGreater(row["stat_decisions_admission"], 10)
        self.assertEqual(row["stat_decisions_admission"] + row["stat_decisions_resident"],
                         row["stat_decisions"])
        self.assertLess(row["stat_decisions"], row["stat_decisions_seen"])   # window filter
        window = [d for d in tracer.decisions if d[0] >= MEASURE_FROM_MS]
        self.assertEqual(row["stat_decisions"], len(window))
        self.assertEqual(row["stat_decisions_resident"], sum(1 for d in window if d[2] < 0))

    def test_the_label_identities_hold_without_horizon_ties(self):
        for arm in ("label", "hybrid_label_label", "noise_0", "swap_uniform_0",
                    "swap_runnerup_0"):
            with self.subTest(arm=arm):
                _, stats, _, _ = replay(self.trace, arm)
                row = stats.row()
                self.assertGreater(row["m1_pairs"], 0)
                self.assertEqual(row["m1"], 1.0)
                for suffix in ("", "_admission", "_resident"):
                    self.assertEqual(row[f"m2{suffix}"], 1.0)
                    self.assertEqual(row[f"m3{suffix}"], 0.0)
                    self.assertEqual(row[f"m4{suffix}"], 0.0)
                self.assertEqual(runner.check_statistics([self._runner_row(arm, row)]), [])

    def test_with_horizon_ties_the_label_has_m4_victims_only_there(self):
        # Whole-second timestamps put next uses exactly H after a decision: the
        # label clips them with the non-reused states, recency picks among
        # those, and m4 (next use <= H, the binary target) counts the victim.
        # The checked identity is that every m4 victim of the label is one of
        # these, and it holds here with m4 > 0.
        temporary, trace, _ = build(mechanism_tests.conversation_records())
        self.addCleanup(temporary.cleanup)
        for arm in ("label", "hybrid_label_label", "noise_0", "swap_uniform_0",
                    "swap_runnerup_0"):
            with self.subTest(arm=arm):
                _, stats, _, _ = replay(trace, arm)
                row = stats.row()
                self.assertGreater(row["m4_count"], 0)
                for suffix in ("", "_admission", "_resident"):
                    self.assertEqual(row[f"m4_count{suffix}"],
                                     row[f"m4_victim_at_horizon{suffix}"])
                self.assertEqual((row["m1"], row["m2"], row["m3"]), (1.0, 1.0, 0.0))
                self.assertEqual(runner.check_statistics([self._runner_row(arm, row)]), [])

    def test_a_label_m4_victim_off_the_horizon_is_reported(self):
        _, stats, _, _ = replay(self.trace, "label")
        row = self._runner_row("label", stats.row())
        self.assertEqual(runner.check_statistics([row]), [])
        for suffix in ("", "_admission", "_resident"):
            with self.subTest(scope=suffix or "all"):
                broken = dict(row)
                broken[f"m4_count{suffix}"] = int(row[f"m4_victim_at_horizon{suffix}"]) + 1
                problems = runner.check_statistics([broken])
                self.assertEqual(len(problems), 1)
                self.assertIn(f"m4_count{suffix} = {broken[f'm4_count{suffix}']}", problems[0])
                self.assertIn(f"m4_victim_at_horizon{suffix} = "
                              f"{row[f'm4_victim_at_horizon{suffix}']}", problems[0])
        # The other identities stay exact.
        for column, value in (("m1", 0.999), ("m2_resident", 0.5), ("m3", 1e-12)):
            with self.subTest(column=column):
                problems = runner.check_statistics([dict(row, **{column: value})])
                self.assertEqual(len(problems), 1)
                self.assertIn(f"label identity {column}", problems[0])
        # An arm that is not built to equal the label is not held to it.
        self.assertEqual(runner.check_statistics([dict(row, arm="lru", m4_count=5)]), [])

    def _runner_row(self, arm, stats_row):
        """A replay row as the runner's statistics check reads it."""
        return {"trace": "t", "cell": "c", "seed": 0, "arm": arm, "variant": "main",
                "l2_decisions": stats_row["stat_decisions_seen"],
                "l2_rejections": stats_row["stat_rejections_seen"],
                "l2_evictions": stats_row["stat_evictions_seen"], **stats_row}

    def test_kstar_reads_the_stores_last_group(self):
        seen = []

        class Checked(DecisionStatistics):
            def record(inner, candidates, keys, *rest):
                for index, state_id in enumerate(candidates):
                    seen.append(inner._last_group(state_id) == keys[index][-1])
                super().record(candidates, keys, *rest)

        for arm in ("lru", "label", "learned"):
            setup = arm_setup(arm, self.trace, HORIZON, 1, learned_ranker=_LinearRanker())
            stats = Checked(self.trace, HORIZON, MEASURE_FROM_MS)
            run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                         measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                         l2_sample_width=4, l2_seed=1,
                         l2_override_hook=RecordingOverride(stats, setup.override),
                         **setup.replay_arguments())
        self.assertGreater(len(seen), 100)
        self.assertTrue(all(seen))

    def test_every_decision_is_seen_with_its_final_victim(self):
        for arm in ARMS + tuple(IDENTITY_ARMS):
            with self.subTest(arm=arm):
                result, stats, tracer, setup = replay(self.trace, arm)
                row = stats.row()
                self.assertEqual(row["stat_decisions_seen"], result.l2_decisions)
                self.assertEqual(row["stat_rejections_seen"], result.l2_rejections)
                self.assertEqual(row["stat_evictions_seen"], result.l2_evictions)
                changed = sum(1 for d in tracer.decisions if d[5] != d[6])
                self.assertEqual(row["overridden_decisions_seen"], changed)
                if arm in ("adm_label", "evict_label") or arm in SWAP_ARMS:
                    self.assertGreater(changed, 0)

    def test_a_swap_at_p_one_records_the_swapped_victim(self):
        for kind in ("uniform", "runnerup"):
            with self.subTest(kind=kind):
                setup = arm_setup("label", self.trace, HORIZON, 1)
                setup.override = SwapOverride(kind, 1.0, 1)
                result, stats, tracer, _ = replay(self.trace, "label", setup=setup)
                row = stats.row()
                # Every decision with two candidates leaves the K* victim, so the
                # final victims never agree with it; the store's originals always do.
                self.assertEqual(row["single_candidate_decisions"], 0)
                self.assertEqual(row["m2"], 0.0)
                self.assertEqual(row["overridden_decisions"], row["stat_decisions"])
                self.assertEqual(setup.override.swaps, result.l2_decisions)
                plain = DecisionStatistics(self.trace, HORIZON, MEASURE_FROM_MS)
                for timestamp, group, arriving, candidates, keys, original, _ in tracer.decisions:
                    # last_group as it stood at the decision: the second key element.
                    plain.attach(SimpleNamespace(last_group={
                        state_id: keys[index][1] for index, state_id in enumerate(candidates)}))
                    plain(list(candidates), list(keys), original, timestamp, group, arriving)
                self.assertEqual(plain.row()["m2"], 1.0)


# --- the hook is read-only, and the null settings are the label or the rung --------------------


class ReadOnlyAndIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, self.path = build(partial_records())
        self.addCleanup(self.temporary.cleanup)

    def test_every_arm_gives_identical_counters_with_and_without_the_statistics(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                with_stats, _, traced_with, _ = replay(self.trace, arm, statistics=True)
                without, _, traced_without, _ = replay(self.trace, arm, statistics=False)
                self.assertEqual(with_stats.as_row(), without.as_row())
                self.assertEqual(traced_with.decisions, traced_without.decisions)
                # Reading `last_group` inserted nothing into L1's history.
                self.assertEqual(len(traced_with.l1.last_group), len(traced_without.l1.last_group))
                if arm in PRIMARY_ARMS:
                    bare, _, _, _ = replay(self.trace, arm, statistics=False, traced=False)
                    self.assertEqual(bare.as_row(), with_stats.as_row())

    def test_admission_by_x_eviction_by_x_is_rung_x_decision_by_decision(self):
        for x in ("learned", "label"):
            with self.subTest(x=x):
                rung, rung_stats, rung_trace, _ = replay(self.trace, x)
                hybrid, hybrid_stats, hybrid_trace, setup = replay(self.trace, f"hybrid_{x}_{x}")
                self.assertEqual(hybrid_trace.decisions, rung_trace.decisions)
                self.assertEqual(_row(hybrid), _row(rung))
                self.assertEqual(hybrid_stats.row(), rung_stats.row())
                # The hybrid rule ran: it answered every admission decision.
                admissions = sum(1 for d in hybrid_trace.decisions if d[2] >= 0)
                self.assertGreater(admissions, 0)
                self.assertEqual(hybrid_trace.returned, admissions)
        # Two scorer objects in the two roles, each shown the history: the same.
        x, y = RawFeatureScorer(self.trace, _LinearRanker()), RawFeatureScorer(self.trace,
                                                                               _LinearRanker())
        scorer = HybridScorer(x, y)
        setup = errorloc.ArmSetup("hybrid", "learned", scorer=scorer,
                                  override=HybridOverride(scorer.admission))
        _, _, separate, _ = replay(self.trace, "hybrid", setup=setup)
        _, _, rung_trace, _ = replay(self.trace, "learned")
        self.assertEqual(separate.decisions, rung_trace.decisions)

    def test_swap_at_p_zero_and_noise_at_s_zero_are_the_label(self):
        label, label_stats, label_trace, _ = replay(self.trace, "label")
        for arm in ("noise_0", "swap_uniform_0", "swap_runnerup_0"):
            with self.subTest(arm=arm):
                result, stats, tracer, setup = replay(self.trace, arm)
                self.assertEqual(tracer.decisions, label_trace.decisions)
                self.assertEqual(_row(result), _row(label))
                self.assertEqual(stats.row(), label_stats.row())
                if isinstance(setup.override, SwapOverride):
                    self.assertEqual(setup.override.decisions, result.l2_decisions)
                    self.assertEqual(setup.override.swaps, 0)

    def test_the_constructed_arms_differ_from_the_rungs(self):
        _, _, learned, _ = replay(self.trace, "learned")
        _, _, label, _ = replay(self.trace, "label")
        for arm in ("adm_label", "evict_label", "noise_1", "swap_uniform_0.25",
                    "swap_runnerup_0.5"):
            with self.subTest(arm=arm):
                _, _, tracer, _ = replay(self.trace, arm)
                self.assertNotEqual(tracer.decisions, label.decisions)
                self.assertNotEqual(tracer.decisions, learned.decisions)

    def test_admission_by_label_follows_the_rule_on_a_replay(self):
        _, _, tracer, _ = replay(self.trace, "adm_label")
        label = ExactLabelScorer(self.trace, HORIZON)
        admissions = rejections = 0
        for timestamp, _, arriving, candidates, keys, original, final in tracer.decisions:
            # The store's keys are the learned ranker's (Y), its victim their first minimum.
            self.assertEqual(original, errorloc.first_minimum(keys))
            if arriving < 0:
                self.assertEqual(final, original)
                continue
            admissions += 1
            x_keys = [(label.score(state_id, timestamp), keys[index][1])
                      for index, state_id in enumerate(candidates)]
            if errorloc.first_minimum(x_keys) == arriving:
                rejections += 1
                self.assertEqual(final, arriving)
            else:
                others = [index for index in range(len(candidates)) if index != arriving]
                self.assertEqual(final, min(others, key=keys.__getitem__))
        self.assertGreater(admissions, rejections)
        self.assertGreater(rejections, 0)

    def test_the_noise_arms_key_is_label_plus_s_z(self):
        _, _, tracer, _ = replay(self.trace, "noise_2", seed=3)
        checked = 0
        for timestamp, _, _, candidates, keys, original, final in tracer.decisions:
            for index, state_id in enumerate(candidates):
                expected = (_label(self.trace, state_id, timestamp)
                            + 2.0 * noise_z(3, state_id, timestamp))
                self.assertEqual(keys[index][0], expected)
                checked += 1
            self.assertEqual(final, original)
        self.assertGreater(checked, 100)

    def test_noise_and_swap_replays_do_not_depend_on_the_string_hash_seed(self):
        child = (
            "import json, sys\n"
            "from persistent_kv_admission.trace import load_mooncake_trace\n"
            "from persistent_kv_admission.twotier import run_two_tier\n"
            "from persistent_kv_admission import errorloc\n"
            "trace = load_mooncake_trace(sys.argv[1])\n"
            "out = {}\n"
            "for arm in ('noise_1', 'swap_uniform_0.5', 'swap_runnerup_0.25',\n"
            "            'hybrid_label_label'):\n"
            "    setup = errorloc.arm_setup(arm, trace, 30.0, 2)\n"
            "    stats = errorloc.DecisionStatistics(trace, 30.0, 120000.0)\n"
            "    r = run_two_tier(trace, 'lru', 4 * 512, 8 * 512, bytes_per_token=1,\n"
            "                     measure_from_ms=120000.0, l2_eviction='sampled',\n"
            "                     l2_sample_width=4, l2_seed=2,\n"
            "                     l2_override_hook=errorloc.RecordingOverride(stats, setup.override),\n"
            "                     **setup.replay_arguments())\n"
            "    out[arm] = [r.as_row(), stats.row()]\n"
            "print(json.dumps(out, sort_keys=True, default=str))\n"
        )
        outputs = []
        for hash_seed in ("0", "4242"):
            environment = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=SOURCE_ROOT)
            finished = subprocess.run([sys.executable, "-c", child, str(self.path)],
                                      capture_output=True, text=True, env=environment)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            outputs.append(finished.stdout)
        self.assertEqual(outputs[0], outputs[1])


# --- the reading helpers ---------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_location_by_the_half_rule(self):
        label = errorloc.location_label
        self.assertEqual(label(2.0, 1.0, 0.99), "admission_located")    # exactly half reaches
        self.assertEqual(label(2.0, 0.99, 1.0), "eviction_located")
        self.assertEqual(label(2.0, 1.0, 1.0), "both")
        self.assertEqual(label(2.0, 0.99, 0.99), "neither")
        self.assertEqual(label(2.0, 3.0, -1.0), "admission_located")
        # The rule is applied as written for G <= 0, and a nan reaches nothing.
        self.assertEqual(label(-2.0, -1.0, -1.5), "admission_located")
        self.assertEqual(label(0.0, 0.0, -0.1), "admission_located")
        self.assertEqual(label(math.nan, 1.0, 1.0), "neither")
        self.assertEqual(set(errorloc.LOCATIONS),
                         {"admission_located", "eviction_located", "both", "neither"})

    def test_dose_response_monotonicity_and_crossing(self):
        levels = errorloc.NOISE_LEVELS
        self.assertTrue(errorloc.non_increasing([5.0, 4.0, 4.0, 1.0]))
        self.assertFalse(errorloc.non_increasing([5.0, 4.0, 4.5, 1.0]))
        cross = errorloc.crossing_bracket
        self.assertEqual(cross(levels, [5.0, 4.0, 2.0, 1.0], 3.0), "1-2")
        self.assertEqual(cross(levels, [5.0, 3.0, 2.0, 1.0], 3.0), "1-2")   # at it: upper side
        self.assertEqual(cross(levels, [5.0, 4.0, 3.5, 3.1], 3.0), "above_all")
        self.assertEqual(cross(levels, [2.0, 1.0, 0.5, 0.1], 3.0), "below_all")
        self.assertEqual(cross(levels, [2.0, 4.0, 1.0, 1.0], 3.0), "0.5-1;1-2")
        with self.assertRaises(ValueError):
            cross(levels, [1.0], 0.0)

    def test_seed_sign_readings_with_zeros_and_mixed(self):
        self.assertEqual(sign_reading([1, 2, 3, 4, 5]), "consistent_gain")
        self.assertEqual(sign_reading([-1, -2, -3, -4, -5]), "consistent_loss")
        self.assertEqual(sign_reading([1, 2, 0, 4, 5]), "mixed")
        self.assertEqual(sign_reading([1, -2, 3, 4, 5]), "mixed")
        self.assertEqual(sign_reading([0, 0, 0, 0, 0]), "mixed")

    def test_spearman_ties_constant_sides_and_the_boundary(self):
        rho = errorloc.rank_correlation
        # Average ranks for ties: x ranks (0, 1.5, 1.5, 3), y ranks (0, 1, 2, 3).
        self.assertAlmostEqual(rho([1, 2, 2, 3], [1, 2, 3, 4]), 4.5 / math.sqrt(4.5 * 5.0))
        self.assertTrue(math.isnan(rho([1, 1, 1, 1], [1, 2, 3, 4])))
        self.assertTrue(math.isnan(rho([1, 2, 3, 4], [7, 7, 7, 7])))
        self.assertTrue(math.isnan(rho([1], [1])))
        boundary = rho([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])           # sum d^2 = 2, n = 5
        self.assertEqual(boundary, 0.9)
        self.assertTrue(errorloc.orders_utility(boundary))
        self.assertFalse(errorloc.orders_utility(0.8999999))
        self.assertFalse(errorloc.orders_utility(math.nan))
        self.assertEqual(rho([3, 1, 2], [30, 10, 20]), 1.0)

    def test_orientation(self):
        self.assertEqual([errorloc.oriented(s, 0.25) for s in errorloc.STATISTICS],
                         [0.25, 0.25, -0.25, -0.25])
        self.assertEqual([errorloc.oriented_name(s) for s in errorloc.STATISTICS],
                         ["m1", "m2", "-m3", "-m4"])

    def test_agreement_and_contradiction(self):
        agrees, contradicts = errorloc.agrees, errorloc.contradicts
        self.assertTrue(agrees(0.1, 0.2))
        self.assertTrue(agrees(-0.1, -3.0))
        self.assertFalse(agrees(-0.1, 0.2))
        self.assertTrue(agrees(0.0, 0.0))
        self.assertFalse(agrees(0.0, 0.1))
        self.assertFalse(agrees(math.nan, 0.1))
        self.assertTrue(contradicts("consistent_gain", -0.01))
        self.assertFalse(contradicts("consistent_gain", 0.01))
        self.assertFalse(contradicts("consistent_gain", 0.0))
        self.assertTrue(contradicts("consistent_loss", 0.01))
        self.assertFalse(contradicts("mixed", -5.0))

    def test_the_binary_ceiling_ratio(self):
        self.assertAlmostEqual(errorloc.ceiling_ratio(22.0, 50.0, 10.0), 0.3)
        self.assertTrue(math.isnan(errorloc.ceiling_ratio(22.0, 10.0, 10.0)))


# --- the runner on synthetic rows, and on a constructed trace -----------------------------------

# U of each arm in tokens over 1000 requested tokens (so points = tokens / 10),
# chosen so that every reading has a known answer.
SYNTHETIC_U = {
    "lru": 100, "learned": 200, "label": 600, "offline": 700, "pi3_next_use": 250,
    "pi0_binary": 220, "pi3_binary": 210, "label_binary": 500, "adm_label": 450,
    "evict_label": 300, "noise_0.5": 550, "noise_1": 450, "noise_2": 300, "noise_4": 150,
    "swap_uniform_0.25": 400, "swap_runnerup_0.25": 500, "swap_uniform_0.5": 300,
    "swap_runnerup_0.5": 300,
}
RUNNERUP_HALF_OFFSETS = (-20, 20, -20, -20, -20)     # a mixed placement reading at p = 0.5


def synthetic_rows(traces=("conversation_trace", "toolagent_trace"), cell=(0.01, 4.0)):
    rows = []
    for trace in traces:
        for seed in range(5):
            for index, arm in enumerate(ARMS):
                tokens = SYNTHETIC_U[arm] + seed
                if arm == "swap_runnerup_0.5":
                    tokens += RUNNERUP_HALF_OFFSETS[seed]
                row = {metric: 0.0 for metric in runner.REPLAY_METRICS}
                row.update(trace=trace, l1_fraction=cell[0], l2_multiplier=cell[1],
                           cell=runner.rdp.cell_label(*cell), arm=arm, seed=seed,
                           variant="main", requested_tokens=1000, l1_avoided_tokens=50,
                           l1_capacity_bytes=1, l2_capacity_bytes=4,
                           extra_avoided_tokens=tokens, extra_points=tokens / 10.0,
                           swaps=7 if arm.startswith("swap_") else "")
                # m1 rises with U, m2 is constant, m3 follows the arm order, m4
                # falls with U.
                row.update(m1=tokens / 1000.0, m2=0.5, m3=0.1 * index, m4=1.0 - tokens / 1000.0)
                rows.append(row)
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.summary = runner.aggregate_replays(self.rows)

    def test_location(self):
        seeds, table = runner.location_tables(self.rows)
        self.assertEqual(len(seeds), 10)
        for entry in table:
            self.assertAlmostEqual(entry["G_points_mean"], 40.0)
            self.assertAlmostEqual(entry["A_points_mean"], 25.0)
            self.assertAlmostEqual(entry["E_points_mean"], 10.0)
            self.assertAlmostEqual(entry["interaction_points_mean"], -5.0)
            self.assertEqual(entry["location"], "admission_located")
            self.assertEqual(entry["location_both_traces"], "admission_located")
            self.assertEqual(entry["interaction_reading"], "consistent_loss")
            self.assertEqual(entry["G_seed_signs"], "+++++")
        _, single = runner.location_tables(synthetic_rows(traces=("conversation_trace",)))
        self.assertEqual(single[0]["location_both_traces"], "single_trace")

    def test_dose_response(self):
        _, table = runner.dose_response_tables(self.rows)
        for entry in table:
            self.assertTrue(entry["non_increasing_means"])
            self.assertEqual(entry["seeds_non_increasing"], 5)
            self.assertEqual(entry["crossing"], "2-4")

    def test_placement(self):
        seeds, table = runner.placement_tables(self.rows)
        readings = {(entry["trace"], entry["p"]): entry["reading"] for entry in table}
        for trace in ("conversation_trace", "toolagent_trace"):
            self.assertEqual(readings[(trace, 0.25)], "consistent_gain")
            self.assertEqual(readings[(trace, 0.5)], "mixed")
        self.assertEqual(len(seeds), 20)
        self.assertTrue(all(entry["m2_runnerup_mean"] == 0.5 for entry in table))

    def test_metric_order(self):
        table = runner.metric_order_table(self.summary)
        rho = {(entry["trace"], entry["set"], entry["statistic"]): entry for entry in table}
        for trace in ("conversation_trace", "toolagent_trace"):
            for set_name, arms in (("primary", 12), ("secondary", 18)):
                self.assertEqual(rho[(trace, set_name, "m1")]["arms"], arms)
                self.assertAlmostEqual(rho[(trace, set_name, "m1")]["rho"], 1.0)
                self.assertTrue(rho[(trace, set_name, "m1")]["orders_utility"])
                self.assertAlmostEqual(rho[(trace, set_name, "m4")]["rho"], 1.0)   # -m4 oriented
                self.assertTrue(math.isnan(rho[(trace, set_name, "m2")]["rho"]))
                self.assertFalse(rho[(trace, set_name, "m2")]["orders_utility"])

    def test_ranker_pairs(self):
        _, table = runner.ranker_pair_tables(self.rows)
        by = {(entry["trace"], entry["target"]): entry for entry in table}
        nu = by[("conversation_trace", "next_use")]
        self.assertAlmostEqual(nu["dU_points_mean"], 5.0)
        self.assertEqual(nu["dU_reading"], "consistent_gain")
        self.assertTrue(nu["m1_agrees"] and nu["m4_agrees"])
        self.assertFalse(nu["m2_agrees"])                         # no change against a gain
        self.assertAlmostEqual(nu["d_m3_mean"], -0.3)            # -m3: 0.1 -> 0.4
        self.assertTrue(nu["m3_contradicts"])
        self.assertFalse(nu["m1_contradicts"])
        binary = by[("conversation_trace", "binary")]
        self.assertAlmostEqual(binary["dU_points_mean"], -1.0)
        self.assertEqual(binary["dU_reading"], "consistent_loss")
        self.assertTrue(binary["m1_agrees"])
        # pi3_binary has the larger arm index, so -m3 falls with the utility: it agrees.
        self.assertAlmostEqual(binary["d_m3_mean"], -0.1)
        self.assertTrue(binary["m3_agrees"])
        self.assertFalse(any(binary[f"{s}_contradicts"] for s in errorloc.STATISTICS))

    def test_binary_ceiling(self):
        seeds, table = runner.binary_ceiling_tables(self.rows)
        for entry in table:
            self.assertAlmostEqual(entry["label_minus_label_binary_points_mean"], 10.0)
            self.assertEqual(entry["label_minus_label_binary_reading"], "consistent_gain")
            self.assertAlmostEqual(entry["ceiling_ratio"], 0.3)
        self.assertAlmostEqual(seeds[0]["ceiling_ratio"], 0.3)

    def test_reproduction_check(self):
        rows = [dict(row) for row in self.rows if row["arm"] in ("lru", "pi0_binary")]
        for row in rows:
            row["avoided_prefill_tokens"] = row["extra_avoided_tokens"] + 50
        published = {}
        for row in rows:
            published[(row["arm"],) + runner._cell_seed(row)] = {
                "source": "s", "avoided_prefill_tokens": row["avoided_prefill_tokens"],
                **{field: row[field] for field in runner.REFERENCE_IDENTIFIERS}}
        key = ("lru", "toolagent_trace", 0.01, 4.0, 3)
        published[key] = dict(published[key], avoided_prefill_tokens=1)
        del published[("pi0_binary", "conversation_trace", 0.01, 4.0, 0)]
        report = runner.check_reproduction(rows, published, 10)
        self.assertEqual((report["lru"]["matched"], report["lru"]["mismatched"],
                          report["lru"]["missing"]), (9, 1, 0))
        self.assertEqual((report["pi0_binary"]["matched"], report["pi0_binary"]["missing"]),
                         (9, 1))
        self.assertEqual(report["label"]["matched"], 0)

    def test_the_grid_and_the_task_order(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS, smoke=False)
        self.assertEqual(len(tasks), 1080)
        self.assertEqual(len(set(tasks)), 1080)
        costs = [runner.arm_cost(task[3]) for task in tasks]
        self.assertEqual(costs, sorted(costs))
        self.assertIn(tasks[0][3], ("adm_label", "evict_label"))
        smoke = runner.build_tasks(("conversation_trace",), (runner.SMOKE_CELL,),
                                   runner.SMOKE_SEEDS, smoke=True)
        self.assertEqual(len(smoke), 18 + 18 + 5)
        self.assertEqual({task[5] for task in smoke}, {"main", "nostats"})
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH, runner.MECHANISM), ("all", 16, "all16"))

    def test_tabulation_reads_and_prints_without_writing(self):
        _, location = runner.location_tables(self.rows)
        _, dose = runner.dose_response_tables(self.rows)
        _, placement = runner.placement_tables(self.rows)
        _, pairs = runner.ranker_pair_tables(self.rows)
        _, ceiling = runner.binary_ceiling_tables(self.rows)
        tables = {"replay": self.summary, "location": location, "dose_response": dose,
                  "placement": placement, "metric_order": runner.metric_order_table(self.summary),
                  "ranker_pairs": pairs, "binary_ceiling": ceiling}
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in tables.items():
                runner._write(Path(directory) / f"{name}.csv", rows)
            before = sorted(os.listdir(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(tabulate.main([directory]), 0)
            self.assertEqual(sorted(os.listdir(directory)), before)
            text = stdout.getvalue()
            parsed = {name: tabulate.read_csv(Path(directory) / f"{name}.csv")
                      for name in tables}
        for section in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
            self.assertIn(f"### {section} ", text)
        counts = tabulate.location_counts(parsed["location"])
        self.assertEqual(counts[-1]["admission_located"], 1)
        self.assertEqual(counts[-1]["cells"], 1)
        self.assertEqual([entry["consistent_gain"] for entry in
                          tabulate.placement_counts(parsed["placement"])], [2, 0])
        orders = {(entry["set"], entry["statistic"]): entry["orders"]
                  for entry in tabulate.metric_order_counts(parsed["metric_order"])}
        self.assertEqual((orders[("primary", "m1")], orders[("primary", "m2")]), (2, 0))
        listed = tabulate.contradictions(parsed["ranker_pairs"])
        self.assertEqual({(entry["target"], entry["statistic"]) for entry in listed},
                         {("next_use", "-m3")})
        agree = {entry["target"]: entry for entry in tabulate.ranker_pair_counts(parsed["ranker_pairs"])}
        self.assertEqual((agree["next_use"]["m1_agrees"], agree["next_use"]["m2_agrees"]), (2, 0))


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker`, the checks and the smoke comparisons on a constructed
    trace, with the module state the runner's main would set."""

    def setUp(self):
        self.temporary, self.trace, _ = build(jittered(partial_records(), seed=9))
        self.addCleanup(self.temporary.cleanup)
        name = self.trace.name
        from persistent_kv_admission.replay import _occurrence_groups

        real = {(name, 0.01, 4.0, arm, 0): _OtherRanker() for arm in errorloc.REAL_RANKERS}
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON},
                      rankers={name: _LinearRanker()}, real_rankers=real)
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def test_worker_rows_pass_the_checks_and_the_smoke_comparisons(self):
        tasks = [(self.name, 0.01, 4.0, arm, 0, variant)
                 for arm in ("lru", "learned", "label", "adm_label", "swap_uniform_0.5",
                             "pi0_binary")
                 for variant in ("main", "nostats")]
        tasks += [(self.name, 0.01, 4.0, arm, 0, "main") for arm in IDENTITY_ARMS]
        rows = [runner._replay_worker(task) for task in tasks]
        self.assertEqual(runner.check_statistics(rows), [])
        for row in rows:
            self.assertEqual(row["mechanism"], "all16")
            self.assertGreater(row["l2_decisions"], 0)
            if row["variant"] == "main":
                self.assertEqual(row["stat_decisions_seen"], row["l2_decisions"])
            else:
                self.assertNotIn("m1", row)
        swap = next(row for row in rows if row["arm"] == "swap_uniform_0.5")
        self.assertGreater(swap["swaps"], 0)
        table, problems = runner.check_smoke_identities(rows)
        identity = [entry for entry in table if entry["check"] == "identity"]
        hook = [entry for entry in table if entry["check"] == "hook_read_only"]
        self.assertEqual(len(identity), len(IDENTITY_ARMS))
        self.assertTrue(all(entry["holds"] for entry in identity + hook
                            if "replay missing" not in str(entry)))
        # Arms not replayed here are reported missing, not silently passed.
        self.assertTrue(all("missing" in line for line in problems))
        self.assertEqual(len(problems), len(ARMS) - 6)
        # A broken identity is caught.
        broken = [dict(row) for row in rows]
        next(row for row in broken if row["arm"] == "noise_0")["decision_sha256"] = "0"
        _, problems = runner.check_smoke_identities(broken)
        self.assertTrue(any("noise_0" in line and "decision_sha256" in line for line in problems))


if __name__ == "__main__":
    unittest.main()
