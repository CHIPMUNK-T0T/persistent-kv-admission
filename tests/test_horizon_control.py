"""Tests for the horizon and class-order control (docs/horizon-control-plan.md).

No real trace is replayed: the plan forbids a real replay of the new arms
before the implementation is reviewed. Everything here runs on constructed
traces, hand-built decisions and synthetic rows; the published reference CSVs
are only read. What has to hold before a real replay means anything: each
`label_binary_h` key is the exact `binary` target at h (a next use at exactly h
counts as reuse) and h = 600 is the published `label_binary` arm; the
class-order arms reject the arrival exactly when the frozen ranker does, evict
the first minimum of their composite key among the other candidates, use that
key in every later round, read both components at the decision instant, and
show the ranker exactly the history the store's own scorer would see; the
error-location hybrids return a legal index, or None, in every round type
under leaf eligibility; the identities of the plan hold decision by decision;
the statistics hook is read-only; and the reading helpers, the runner's checks
and refusals, its derivations and the read-only tabulation do what the plan
fixes, at their boundaries.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

from persistent_kv_admission import errorloc, horizonctl
from persistent_kv_admission import twotier
from persistent_kv_admission.decisionpop import RawFeatureScorer, _Labeller, target_column
from persistent_kv_admission.errorloc import (
    DecisionStatistics,
    HybridOverride,
    HybridScorer,
    RecordingOverride,
    first_minimum,
)
from persistent_kv_admission.horizonctl import (
    ARM_MECHANISMS,
    ARMS,
    HORIZON_ARMS,
    HORIZONS_SECONDS,
    IDENTITY_ARMS,
    IDENTITY_REFERENCE_ARMS,
    ReuseClassScorer,
    arm_mechanism,
    arm_setup,
)
from persistent_kv_admission.mechanism import ExactLabelScorer
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.twotier import run_two_tier

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The mechanism-control and error-location fixtures, reused as they are; only
# plain functions and classes are read from them.
mechanism_tests = _load("_horizon_mechanism_fixtures", REPOSITORY / "tests/test_mechanism_control.py")
error_tests = _load("_horizon_error_location_fixtures", REPOSITORY / "tests/test_error_location.py")
runner = _load("_run_horizon_control_under_test", REPOSITORY / "scripts/run_horizon_control.py")
tabulate = _load("_tabulate_horizon_control_under_test",
                 REPOSITORY / "scripts/tabulate_horizon_control.py")

L1_BYTES = mechanism_tests.L1_BYTES
L2_BYTES = mechanism_tests.L2_BYTES
_LinearRanker = mechanism_tests._LinearRanker
_Tracer = error_tests._Tracer
_row = error_tests._row
# The constructed traces run one step per 5 seconds, so every horizon of the
# grid (6 to 600 s) separates states, and the plan's 600-second label horizon
# is the trace's own, as on the real traces.
SCALE = 5
HORIZON = 600.0
MEASURE_FROM_MS = 120_000.0 * SCALE


# --- fixtures -------------------------------------------------------------------------------------


def scaled(records, factor=SCALE):
    """The same records with every timestamp multiplied by `factor`."""
    return [dict(record, timestamp=record["timestamp"] * factor) for record in records]


def build(records):
    return mechanism_tests.build_trace(records)


def partial_trace(test, seed=5):
    """Sessions with partial blocks of many sizes (several rounds per offer;
    non-leaf arrivals under leaf eligibility), at one step per 5 s."""
    temporary, trace, path = build(scaled(error_tests.partial_records(seed=seed)))
    test.addCleanup(temporary.cleanup)
    return trace, path


def replay(trace, arm, seed=1, width=4, statistics=True, traced=True, setup=None,
           eligibility=None, **extra):
    """One replay of `arm` under its mechanism (or `eligibility`, for the all16
    rungs, which this control does not replay); returns (result, statistics,
    tracer, setup)."""
    setup = setup or arm_setup(arm, trace, HORIZON, seed, learned_ranker=_LinearRanker())
    eligibility = eligibility or arm_mechanism(arm)[1]
    stats = DecisionStatistics(trace, HORIZON, MEASURE_FROM_MS) if statistics else None
    override = RecordingOverride(stats, setup.override) if statistics else setup.override
    tracer = _Tracer(override) if traced else None
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                          l2_sample_width=width, l2_seed=seed, l2_eligibility=eligibility,
                          l2_arm=arm, l2_override_hook=tracer if traced else override,
                          **setup.replay_arguments(), **extra)
    return result, stats, tracer, setup


def _binary(trace, state_id, timestamp_ms, horizon_seconds):
    delta, count = _Labeller(trace, horizon_seconds)(state_id, timestamp_ms)
    return float(target_column(np.asarray([delta]), np.asarray([count], dtype=float), "binary",
                               horizon_seconds)[0])


class _Fixed:
    """A scorer with fixed values that counts its observations."""

    time_varying = True

    def __init__(self, scores):
        self.scores = scores
        self.observed = 0

    def observe(self, requests, timestamp_ms):
        self.observed += 1

    def score(self, state_id, timestamp_ms):
        return float(self.scores[state_id])


# --- the arms -------------------------------------------------------------------------------------


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def _setup(self, arm, seed=0):
        return arm_setup(arm, self.trace, HORIZON, seed, learned_ranker=_LinearRanker())

    def test_the_nine_arms_their_mechanisms_and_the_identities(self):
        self.assertEqual(ARMS, (
            "label_binary_6", "label_binary_15", "label_binary_60", "label_binary_300",
            "label_binary_600", "evict_binary_learned", "evict_binary_recency", "adm_label",
            "evict_label"))
        self.assertEqual(tabulate.ARMS, ARMS)
        self.assertEqual(tabulate.HORIZONS, HORIZONS_SECONDS)
        for arm in ARMS[:7]:
            self.assertEqual(arm_mechanism(arm), ("all16", "all", 16))
        for arm in ("adm_label", "evict_label", "hybrid_learned_learned", "hybrid_label_label",
                    "learned", "label"):
            self.assertEqual(arm_mechanism(arm), ("leaf16", "leaf", 16))
        self.assertEqual(IDENTITY_ARMS, {"adm_label_binary_evict_binary_recency": "label_binary_600",
                                         "hybrid_learned_learned": "learned",
                                         "hybrid_label_label": "label"})
        self.assertEqual(arm_mechanism("adm_label_binary_evict_binary_recency")[0], "all16")
        with self.assertRaises(ValueError):
            arm_mechanism("lru")

    def test_the_horizon_arms_are_the_binary_target_at_h(self):
        for h, arm in zip(HORIZONS_SECONDS, HORIZON_ARMS):
            with self.subTest(arm=arm):
                setup = self._setup(arm)
                self.assertIsInstance(setup.scorer, ExactLabelScorer)
                self.assertEqual((setup.scorer.target, setup.scorer.horizon_seconds),
                                 ("binary", h))
                self.assertIsNone(setup.override)
                self.assertEqual(setup.l2_policy, "learned")
        # h = 600 is the error-location control's label_binary at its horizon.
        mine = self._setup("label_binary_600")
        theirs = errorloc.arm_setup("label_binary", self.trace, 600.0, 0)
        self.assertEqual(mine.replay_arguments().keys(), theirs.replay_arguments().keys())
        self.assertEqual((mine.scorer.target, mine.scorer.horizon_seconds),
                         (theirs.scorer.target, theirs.scorer.horizon_seconds))

    def test_the_class_order_arms_are_hybrids_on_the_frozen_ranker(self):
        learned = self._setup("evict_binary_learned")
        self.assertIsInstance(learned.scorer, ReuseClassScorer)
        self.assertIsInstance(learned.scorer.within, RawFeatureScorer)
        self.assertIsInstance(learned.scorer.within.ranker, _LinearRanker)
        self.assertEqual((learned.scorer.reuse.target, learned.scorer.reuse.horizon_seconds),
                         ("binary", 600.0))
        self.assertIsInstance(learned.override, HybridOverride)
        # X is the ranker inside the composite: one object, observed once.
        self.assertIs(learned.override.admission, learned.scorer.within)
        recency = self._setup("evict_binary_recency")
        self.assertIsInstance(recency.scorer, HybridScorer)
        self.assertIsInstance(recency.scorer.admission, RawFeatureScorer)
        self.assertEqual((recency.scorer.eviction.target, recency.scorer.eviction.horizon_seconds),
                         ("binary", 600.0))
        self.assertIs(recency.override.admission, recency.scorer.admission)
        identity = self._setup("adm_label_binary_evict_binary_recency")
        self.assertIsInstance(identity.scorer, HybridScorer)
        self.assertIsNot(identity.scorer.admission, identity.scorer.eviction)
        for scorer in (identity.scorer.admission, identity.scorer.eviction):
            self.assertEqual((scorer.target, scorer.horizon_seconds), ("binary", 600.0))
        self.assertIs(identity.override.admission, identity.scorer.admission)

    def test_the_leaf_arms_are_the_error_location_construction(self):
        for arm in ("adm_label", "evict_label", "hybrid_learned_learned", "hybrid_label_label",
                    "learned", "label"):
            with self.subTest(arm=arm):
                mine = self._setup(arm)
                theirs = errorloc.arm_setup(arm, self.trace, HORIZON, 0,
                                            learned_ranker=_LinearRanker())
                self.assertIs(type(mine.scorer), type(theirs.scorer))
                self.assertIs(type(mine.override), type(theirs.override))
                self.assertEqual(mine.l2_policy, theirs.l2_policy)
        adm = self._setup("adm_label")
        self.assertEqual(adm.scorer.admission.target, "next_use")
        self.assertIsInstance(adm.scorer.eviction, RawFeatureScorer)

    def test_unknown_arms_and_missing_inputs_are_refused(self):
        for arm in ("label_binary_7", "label_binary", "evict_binary_lru", "noise_1", "lru"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self._setup(arm)
        for arm in ("evict_binary_learned", "evict_binary_recency", "adm_label"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                arm_setup(arm, self.trace, HORIZON, 0)
        scorer = _Fixed({})
        with self.assertRaises(ValueError):
            ReuseClassScorer(scorer, scorer)
        with self.assertRaises(ValueError):
            horizonctl.class_order_setup("x", self.trace, scorer, "frequency")


# --- Part 1: the horizon label -------------------------------------------------------------------


class HorizonLabelTests(unittest.TestCase):
    def test_a_next_use_at_exactly_h_is_reuse_and_one_millisecond_later_is_not(self):
        for h in HORIZONS_SECONDS:
            with self.subTest(h=h):
                # State int:1 occurs at 0, exactly h later, and h + 1 ms after that.
                h_ms = int(h * 1000)
                stamps = (0, h_ms, 2 * h_ms + 1)
                records = [{"timestamp": t, "input_length": 512, "output_length": 1,
                            "hash_ids": [1]} for t in stamps]
                temporary, trace, _ = build(records)
                self.addCleanup(temporary.cleanup)
                scorer = arm_setup(horizonctl.horizon_arm(h), trace, HORIZON, 0).scorer
                self.assertEqual(scorer.score("int:1", 0.0), 1.0)              # exactly h
                self.assertEqual(scorer.score("int:1", 1.0), 1.0)              # h - 1 ms
                self.assertEqual(scorer.score("int:1", float(h_ms)), 0.0)      # h + 1 ms
                self.assertEqual(scorer.score("int:1", float(h_ms) + 1.0), 1.0)  # exactly h
                self.assertEqual(scorer.score("int:1", float(stamps[-1])), 0.0)  # no further use
                # Everywhere: the training target itself.
                instants = np.linspace(-500.0, stamps[-1] + 500.0, 301)
                expected = [_binary(trace, "int:1", float(t), h) for t in instants]
                self.assertEqual([scorer.score("int:1", float(t)) for t in instants], expected)

    def test_on_a_replay_the_key_is_the_binary_target_at_h(self):
        trace, _ = partial_trace(self)
        for h, arm in zip(HORIZONS_SECONDS, HORIZON_ARMS):
            with self.subTest(arm=arm):
                result, _, tracer, _ = replay(trace, arm)
                checked = 0
                for timestamp, _, _, candidates, keys, original, final in tracer.decisions:
                    for index, state_id in enumerate(candidates):
                        self.assertEqual(keys[index][0], _binary(trace, state_id, timestamp, h))
                        self.assertEqual(len(keys[index]), 2)
                        checked += 1
                    self.assertEqual(final, original)
                    self.assertEqual(original, first_minimum(keys))
                self.assertGreater(checked, 1000)

    def test_label_binary_600_is_the_published_label_binary_arm(self):
        trace, _ = partial_trace(self)
        mine, mine_stats, mine_trace, _ = replay(trace, "label_binary_600")
        theirs_setup = errorloc.arm_setup("label_binary", trace, 600.0, 1)
        theirs, theirs_stats, theirs_trace, _ = replay(trace, "label_binary_600",
                                                       setup=theirs_setup)
        self.assertEqual(mine_trace.decisions, theirs_trace.decisions)
        self.assertEqual(_row(mine), _row(theirs))
        self.assertEqual(mine_stats.row(), theirs_stats.row())

    def test_the_horizons_change_the_decisions(self):
        trace, _ = partial_trace(self)
        streams = [replay(trace, arm)[2].decisions for arm in HORIZON_ARMS]
        for left in range(len(streams)):
            for right in range(left + 1, len(streams)):
                self.assertNotEqual(streams[left], streams[right])


# --- Part 2: the composite key on hand-built decisions ------------------------------------------


class CompositeOverrideTests(unittest.TestCase):
    """Candidates in draw order, the arrival first; the store's key is
    `(scorer.score, last_group)` and its victim the first minimum."""

    CANDIDATES = ["arr", "r1", "r2", "r3"]

    def _decide(self, order, ranker, reuse, last_group, arriving=0, candidates=None):
        candidates = candidates or self.CANDIDATES
        ranker_scorer, reuse_scorer = _Fixed(ranker), _Fixed(reuse)
        with mock.patch.object(horizonctl, "reuse_label", lambda trace: reuse_scorer):
            setup = horizonctl.class_order_setup("arm", None, ranker_scorer, order)
        keys = [(setup.scorer.score(state_id, 0.0), float(last_group[state_id]))
                for state_id in candidates]
        victim = first_minimum(keys)
        answer = setup.override(list(candidates), keys, victim, 0.0, 0, arriving)
        return answer, victim, keys

    def test_the_arrival_is_rejected_iff_it_is_the_rankers_first_minimum(self):
        last_group = {"arr": 9, "r1": 1, "r2": 4, "r3": 2}
        reuse = {"arr": 1, "r1": 0, "r2": 1, "r3": 0}
        for order, kept in (("learned", 3), ("recency", 1)):
            with self.subTest(order=order):
                # The ranker's minimum is the arrival: rejected, although it is
                # reusable and the composite would evict a resident.
                answer, _, _ = self._decide(order, {"arr": 0.5, "r1": 3.0, "r2": 1.0, "r3": 2.0},
                                            reuse, last_group)
                self.assertEqual(answer, 0)
                # The ranker's minimum is r2 (reusable): the arrival is kept and
                # the victim is the composite's minimum among the others, a
                # non-reusable resident: r3 by the ranker's order, r1 by recency.
                answer, victim, _ = self._decide(
                    order, {"arr": 5.0, "r1": 3.0, "r2": 1.0, "r3": 2.0}, reuse, last_group)
                self.assertEqual(answer, kept)
                self.assertEqual(victim, kept)
                # Equal ranker keys: the arrival, first in draw order, is rejected.
                answer, _, _ = self._decide(order, {"arr": 1.0, "r1": 1.0, "r2": 1.0, "r3": 1.0},
                                            reuse, {"arr": 1, "r1": 1, "r2": 1, "r3": 1})
                self.assertEqual(answer, 0)

    def test_the_arrival_is_excluded_from_the_composite_choice(self):
        # The arrival is not reusable and the composite's minimum over all four
        # candidates, so the store alone would reject it; the ranker keeps it
        # (r2 is the ranker's minimum), so the victim is the composite's
        # minimum over the residents.
        ranker = {"arr": 1.5, "r1": 3.0, "r2": 1.0, "r3": 2.0}
        reuse = {"arr": 0, "r1": 0, "r2": 1, "r3": 0}
        last_group = {"arr": 0, "r1": 1, "r2": 4, "r3": 2}
        for order, expected in (("learned", 3), ("recency", 1)):
            with self.subTest(order=order):
                answer, victim, _ = self._decide(order, ranker, reuse, last_group)
                self.assertEqual(victim, 0)
                self.assertEqual(answer, expected)

    def test_the_order_within_a_reuse_class(self):
        reuse = {"arr": 1, "r1": 0, "r2": 0, "r3": 0}
        keep = {"arr": 9.0}
        # Learned order: the lowest ranker score among the non-reusable leaves
        # (r1), whatever its recency; recency order: the oldest (r3).
        ranker = dict(keep, r1=0.1, r2=5.0, r3=7.0)
        last_group = {"arr": 9, "r1": 8, "r2": 5, "r3": 2}
        self.assertEqual(self._decide("learned", ranker, reuse, last_group)[0], 1)
        self.assertEqual(self._decide("recency", ranker, reuse, last_group)[0], 3)
        # Equal ranker scores: recency decides, then draw order.
        tied = dict(keep, r1=1.0, r2=1.0, r3=1.0)
        self.assertEqual(self._decide("learned", tied, reuse, last_group)[0], 3)
        flat = {"arr": 9, "r1": 2, "r2": 2, "r3": 2}
        self.assertEqual(self._decide("learned", tied, reuse, flat)[0], 1)
        self.assertEqual(self._decide("recency", ranker, reuse, flat)[0], 1)
        # A non-reusable resident leaves before any reusable one, whatever the
        # ranker says.
        reuse_mixed = {"arr": 1, "r1": 1, "r2": 1, "r3": 0}
        ranker_low = dict(keep, r1=-5.0, r2=-6.0, r3=8.0)
        self.assertEqual(self._decide("learned", ranker_low, reuse_mixed, last_group)[0], 3)

    def test_later_rounds_keep_the_composite_store_choice(self):
        ranker = {"arr": -9.0, "r1": 3.0, "r2": 1.0, "r3": 2.0}
        reuse = {"arr": 1, "r1": 0, "r2": 1, "r3": 0}
        last_group = {"arr": 0, "r1": 4, "r2": 1, "r3": 2}
        for order, expected in (("learned", 3), ("recency", 3)):
            with self.subTest(order=order):
                # arriving_index -1: the hook keeps the store's choice, the first
                # minimum of the composite over every candidate (the arrival
                # included as an ordinary resident, never the ranker's pick).
                answer, victim, keys = self._decide(order, ranker, reuse, last_group, arriving=-1)
                self.assertIsNone(answer)
                self.assertEqual(victim, expected)
        # The composite of evict_binary_learned orders as (reusable, score, last_group).
        _, _, keys = self._decide("learned", ranker, reuse, last_group, arriving=-1)
        self.assertEqual(keys[1], ((0.0, 3.0), 4.0))
        _, _, keys = self._decide("recency", ranker, reuse, last_group, arriving=-1)
        self.assertEqual(keys[1], (0.0, 4.0))

    def test_the_reuse_class_scorer(self):
        reuse, within = _Fixed({"a": 1}), _Fixed({"a": -2.5})
        scorer = ReuseClassScorer(reuse, within)
        self.assertEqual(scorer.score("a", 0.0), (1.0, -2.5))
        scorer.observe([], 0.0)
        self.assertEqual((reuse.observed, within.observed), (1, 1))
        # Nested and flat composites order alike.
        nested = sorted([((0.0, 2.0), 1.0), ((0.0, 2.0), 0.0), ((1.0, -9.0), 0.0), ((0.0, 1.0), 9.0)])
        flat = sorted([(0.0, 2.0, 1.0), (0.0, 2.0, 0.0), (1.0, -9.0, 0.0), (0.0, 1.0, 9.0)])
        self.assertEqual([(a, b, c) for (a, b), c in nested], flat)


# --- Part 2 on replays ----------------------------------------------------------------------------


class _RuleCheck:
    """Override-slot wrapper for a class-order arm: at every decision, before
    the arm's own override runs, reads the ranker score and the reuse label of
    every candidate from independent scorers (the ranker shown the same
    history through `run_two_tier`'s observer) and checks the store's keys and
    the arm's answer against the plan's rule."""

    def __init__(self, test, inner, independent, reuse, order):
        self.test, self.inner, self.independent, self.reuse = test, inner, independent, reuse
        self.order = order
        self.counts = defaultdict(int)

    def attach(self, l1, l2, counters):
        if hasattr(self.inner, "attach"):
            self.inner.attach(l1, l2, counters)

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index, arriving_index):
        test = self.test
        ranker = [self.independent.score(state_id, timestamp_ms) for state_id in candidates]
        reuse = [self.reuse.score(state_id, timestamp_ms) for state_id in candidates]
        last = [key[1] for key in keys]
        for index in range(len(candidates)):
            expected = ((reuse[index], ranker[index]) if self.order == "learned"
                        else reuse[index])
            test.assertEqual(keys[index][0], expected)
        composite = [((reuse[i], ranker[i], last[i]) if self.order == "learned"
                      else (reuse[i], last[i])) for i in range(len(candidates))]
        answer = self.inner(candidates, keys, victim_index, timestamp_ms, group_index,
                            arriving_index)
        final = victim_index if answer is None else answer
        if arriving_index >= 0:
            ranker_keys = [(ranker[i], last[i]) for i in range(len(candidates))]
            if first_minimum(ranker_keys) == arriving_index:
                test.assertEqual(final, arriving_index)
                self.counts["rejected"] += 1
            else:
                others = [i for i in range(len(candidates)) if i != arriving_index]
                test.assertEqual(final, min(others, key=composite.__getitem__))
                self.counts["admitted"] += 1
                self.counts["differs_from_ranker"] += final != first_minimum(ranker_keys)
        else:
            test.assertEqual(final, first_minimum(composite))
            self.counts["later"] += 1
        return answer


class ClassOrderReplayTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_the_rule_on_a_replay_with_both_components_read_at_the_decision(self):
        for arm, order in horizonctl.CLASS_ORDER.items():
            with self.subTest(arm=arm):
                setup = arm_setup(arm, self.trace, HORIZON, 1, learned_ranker=_LinearRanker())
                independent = RawFeatureScorer(self.trace, _LinearRanker())
                check = _RuleCheck(self, setup.override, independent,
                                   ExactLabelScorer(self.trace, 600.0, "binary"), order)
                checked = errorloc.ArmSetup(arm, "learned", scorer=setup.scorer, override=check)
                result, _, _, _ = replay(self.trace, arm, setup=checked, observer=independent)
                counts = check.counts
                self.assertEqual(sum(counts[k] for k in ("rejected", "admitted", "later")),
                                 result.l2_decisions)
                self.assertEqual(counts["rejected"], result.l2_rejections)
                for kind in ("rejected", "admitted", "later", "differs_from_ranker"):
                    self.assertGreater(counts[kind], 0, kind)

    def test_the_ranker_is_observed_once_per_group_as_the_stores_own_scorer(self):
        groups = [timestamp for timestamp, _ in self.trace.timestamp_groups()]
        for arm in horizonctl.CLASS_ORDER_ARMS:
            with self.subTest(arm=arm):
                setup = arm_setup(arm, self.trace, HORIZON, 1, learned_ranker=_LinearRanker())
                ranker = setup.override.admission
                seen = []
                original = ranker.observe

                def counted(requests, timestamp_ms, original=original, seen=seen):
                    seen.append(timestamp_ms)
                    original(requests, timestamp_ms)

                ranker.observe = counted
                replay(self.trace, arm, setup=setup)
                self.assertEqual(seen, groups)

    def _learned_all16(self):
        return replay(self.trace, "learned", eligibility="all", setup=errorloc.arm_setup(
            "learned", self.trace, HORIZON, 1, learned_ranker=_LinearRanker()))

    def test_with_a_constant_reuse_label_evict_binary_learned_is_the_learned_rung(self):
        # The composite with one reuse class is the ranker's key: the replay is
        # the all16 learned rung decision by decision only if the ranker inside
        # it is shown exactly the store's own history.
        _, learned_stats, learned, _ = self._learned_all16()
        for value in (0.0, 1.0):
            with self.subTest(reuse=value):
                constant = _Fixed(defaultdict(lambda value=value: value))
                with mock.patch.object(horizonctl, "reuse_label", lambda trace: constant):
                    setup = arm_setup("evict_binary_learned", self.trace, HORIZON, 1,
                                      learned_ranker=_LinearRanker())
                result, stats, tracer, _ = replay(self.trace, "evict_binary_learned",
                                                  setup=setup)
                self.assertEqual([d[:3] + d[3:4] + d[5:] for d in tracer.decisions],
                                 [d[:3] + d[3:4] + d[5:] for d in learned.decisions])
                self.assertEqual(result.l2_rejections, learned_stats.row()["stat_rejections_seen"])
                self.assertEqual(stats.row()["overridden_decisions_seen"], 0)

    def test_the_class_order_arms_differ_from_the_rungs_and_each_other(self):
        streams = {arm: [d[:4] + d[5:] for d in replay(self.trace, arm)[2].decisions]
                   for arm in ("label_binary_600",) + horizonctl.CLASS_ORDER_ARMS}
        learned_stream = [d[:4] + d[5:] for d in self._learned_all16()[2].decisions]
        for arm in horizonctl.CLASS_ORDER_ARMS:
            self.assertNotEqual(streams[arm], streams["label_binary_600"])
            self.assertNotEqual(streams[arm], learned_stream)
        self.assertNotEqual(streams["evict_binary_learned"], streams["evict_binary_recency"])


# --- Part 3: the hybrids under leaf eligibility ----------------------------------------------------


class _LegalityRecorder:
    """Override slot around an override: records every decision with its round
    type and checks that the answer is None or a candidate index, and None
    whenever the arrival is not a candidate of a first round."""

    def __init__(self, test, inner):
        self.test, self.inner = test, inner
        self.pending = None
        self.records = []

    def attach(self, l1, l2, counters):
        if hasattr(self.inner, "attach"):
            self.inner.attach(l1, l2, counters)

    def on_victim(self, state_id, timestamp_ms, group_index):
        self.pending = state_id

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index, arriving_index):
        arrival, self.pending = self.pending, None
        answer = self.inner(candidates, keys, victim_index, timestamp_ms, group_index,
                            arriving_index)
        if answer is not None:
            self.test.assertIs(type(answer), int)
            self.test.assertTrue(0 <= answer < len(candidates))
        if arrival is not None and arriving_index == 0:
            kind = "first_round_leaf_arrival"
            self.test.assertEqual(candidates[0], arrival)
        elif arrival is not None:
            kind = "first_round_arrival_not_a_leaf"
            self.test.assertEqual(arriving_index, -1)
            self.test.assertNotIn(arrival, candidates)
        else:
            kind = "later_round"
            self.test.assertEqual(arriving_index, -1)
        if arriving_index < 0:
            self.test.assertIsNone(answer)
        self.records.append((kind, tuple(candidates), arriving_index, victim_index, answer))
        return answer


class LeafHybridTests(unittest.TestCase):
    def test_every_round_type_on_a_hand_built_leaf_store(self):
        trace = mechanism_tests._tree_trace({"p": None, "c": "p", "x": None, "y": None,
                                             "a": None, "b": None, "d": None})
        sizes = {"p": 2, "c": 1, "x": 1, "y": 1, "a": 1, "b": 1, "d": 1}
        x = _Fixed({"p": -100.0, "c": 5.0, "x": 5.0, "y": 5.0, "a": -50.0, "b": 50.0,
                    "d": -1000.0})
        y = _Fixed({"p": 1.0, "c": 0.0, "x": 5.0, "y": 6.0, "a": 9.0, "b": 9.0, "d": 100.0})
        last_group = {"p": 4, "c": 3, "x": 1, "y": 2, "a": 5, "b": 6, "d": 7}
        recorder = _LegalityRecorder(self, HybridOverride(x))
        removals = []
        store = twotier._VictimStore(
            trace, 3, "learned", {}, sizes.__getitem__, eviction="sampled", sample_width=16,
            seed=0, scorer=HybridScorer(x, y), frequency=defaultdict(int), last_group=last_group,
            removal_hook=lambda state_id, kind, *rest: removals.append((state_id, kind)),
            override_hook=recorder, eligibility="leaf")
        for state_id in ("c", "x", "y"):
            recorder.on_victim(state_id, 0.0, 0)
            self.assertTrue(store.admit(state_id, 1, 0, 0))
        # p arrives with its child c resident: not a candidate of its first
        # round, so X (which would reject it) is never asked; Y evicts c.
        # p is then a leaf and, in the second round, an ordinary candidate
        # that Y evicts: an eviction, not a rejection.
        recorder.on_victim("p", 0.0, 0)
        self.assertTrue(store.admit("p", 1, 0, 0))
        # b, a leaf, arrives: X keeps it (a is X's minimum), Y evicts x among
        # the others.
        for state_id in ("a", "b"):
            recorder.on_victim(state_id, 0.0, 0)
            self.assertTrue(store.admit(state_id, 1, 0, 0))
        # d, a leaf that Y would keep, is rejected by X.
        recorder.on_victim("d", 0.0, 0)
        self.assertFalse(store.admit("d", 1, 0, 0))
        self.assertEqual(recorder.records, [
            ("first_round_arrival_not_a_leaf", ("c", "x", "y"), -1, 0, None),
            ("later_round", ("x", "y", "p"), -1, 2, None),
            ("first_round_leaf_arrival", ("b", "x", "y", "a"), 0, 1, 1),
            ("first_round_leaf_arrival", ("d", "y", "a", "b"), 0, 1, 0),
        ])
        self.assertEqual(removals, [("c", "evicted"), ("p", "evicted"), ("x", "evicted"),
                                    ("d", "rejected")])
        self.assertEqual((store.rejections, store.evictions), (1, 3))

    def test_every_answer_is_legal_on_a_leaf_replay_and_every_round_type_occurs(self):
        trace, _ = partial_trace(self)
        for arm in ("adm_label", "evict_label", "hybrid_learned_learned", "hybrid_label_label"):
            with self.subTest(arm=arm):
                setup = arm_setup(arm, trace, HORIZON, 1, learned_ranker=_LinearRanker())
                recorder = _LegalityRecorder(self, setup.override)
                checked = errorloc.ArmSetup(arm, "learned", scorer=setup.scorer,
                                            override=recorder)
                collector = runner.AttributionCollector(trace, measure_from_ms=MEASURE_FROM_MS,
                                                        bytes_per_token=1)
                result, _, _, _ = replay(trace, arm, setup=checked, victim_hook=recorder.on_victim,
                                         l2_request_hook=collector.on_request,
                                         l2_removal_hook=collector)
                collector.check_against(result)
                kinds = defaultdict(int)
                for kind, *_ in recorder.records:
                    kinds[kind] += 1
                self.assertEqual(sum(kinds.values()), result.l2_decisions)
                for kind in ("first_round_leaf_arrival", "first_round_arrival_not_a_leaf",
                             "later_round"):
                    self.assertGreater(kinds[kind], 0, kind)
                self.assertEqual(result.l2_present_unusable_tokens, 0)
                self.assertEqual(collector.orphaned_blocks, 0)
                # Only first rounds with a leaf arrival are answered.
                answered = sum(1 for record in recorder.records if record[4] is not None)
                self.assertEqual(answered, kinds["first_round_leaf_arrival"])

    def test_admission_by_x_eviction_by_x_is_rung_x_under_leaf16(self):
        trace, _ = partial_trace(self)
        for x in ("learned", "label"):
            with self.subTest(x=x):
                rung, rung_stats, rung_trace, _ = replay(trace, x)
                hybrid, hybrid_stats, hybrid_trace, _ = replay(trace, f"hybrid_{x}_{x}")
                self.assertEqual(hybrid_trace.decisions, rung_trace.decisions)
                self.assertEqual(_row(hybrid), _row(rung))
                self.assertEqual(hybrid_stats.row(), rung_stats.row())
                admissions = sum(1 for d in hybrid_trace.decisions if d[2] >= 0)
                self.assertGreater(admissions, 0)
                self.assertEqual(hybrid_trace.returned, admissions)
                self.assertGreater(sum(1 for d in hybrid_trace.decisions if d[2] < 0), 0)

    def test_the_leaf_hybrids_differ_from_the_leaf_rungs(self):
        trace, _ = partial_trace(self)
        rungs = {x: replay(trace, x)[2].decisions for x in ("learned", "label")}
        for arm in horizonctl.LEAF_ARMS:
            decisions = replay(trace, arm)[2].decisions
            for x, stream in rungs.items():
                self.assertNotEqual([d[:4] + d[5:] for d in decisions],
                                    [d[:4] + d[5:] for d in stream], (arm, x))


# --- identities and the read-only hook ------------------------------------------------------------


class IdentityAndReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_admission_by_the_reuse_label_with_recency_eviction_is_label_binary_600(self):
        rung, rung_stats, rung_trace, _ = replay(self.trace, "label_binary_600")
        arm, arm_stats, arm_trace, _ = replay(self.trace, "adm_label_binary_evict_binary_recency")
        self.assertEqual(arm_trace.decisions, rung_trace.decisions)
        self.assertEqual(_row(arm), _row(rung))
        self.assertEqual(arm_stats.row(), rung_stats.row())
        admissions = sum(1 for d in arm_trace.decisions if d[2] >= 0)
        self.assertEqual(arm_trace.returned, admissions)
        self.assertGreater(rung.l2_rejections, 0)
        self.assertGreater(sum(1 for d in arm_trace.decisions if d[2] < 0), 0)

    def test_every_arm_gives_identical_counters_with_and_without_the_statistics(self):
        for arm in ARMS + tuple(IDENTITY_ARMS) + IDENTITY_REFERENCE_ARMS:
            with self.subTest(arm=arm):
                with_stats, stats, traced_with, setup = replay(self.trace, arm, statistics=True)
                without, _, traced_without, _ = replay(self.trace, arm, statistics=False)
                self.assertEqual(with_stats.as_row(), without.as_row())
                self.assertEqual(traced_with.decisions, traced_without.decisions)
                self.assertEqual(len(traced_with.l1.last_group), len(traced_without.l1.last_group))
                if setup.override is None:
                    bare, _, _, _ = replay(self.trace, arm, statistics=False, traced=False)
                    self.assertEqual(bare.as_row(), with_stats.as_row())
                # The statistics saw every decision with its final victim.
                row = stats.row()
                self.assertEqual(row["stat_decisions_seen"], with_stats.l2_decisions)
                self.assertEqual(row["stat_rejections_seen"], with_stats.l2_rejections)
                changed = sum(1 for d in traced_with.decisions if d[5] != d[6])
                self.assertEqual(row["overridden_decisions_seen"], changed)
                if arm in ARMS[5:]:
                    self.assertGreater(changed, 0)
                elif arm in IDENTITY_ARMS or arm in HORIZON_ARMS:
                    self.assertEqual(changed, 0)

    def test_the_new_arms_do_not_depend_on_the_string_hash_seed(self):
        _, path = partial_trace(self)
        child = (
            "import json, sys\n"
            "from persistent_kv_admission import horizonctl\n"
            "from persistent_kv_admission.trace import load_mooncake_trace\n"
            "from persistent_kv_admission.twotier import run_two_tier\n"
            "class Ranker:\n"
            "    def score_row(self, row):\n"
            "        return sum((0.3 - 0.1 * (i % 5)) * v for i, v in enumerate(row))\n"
            "trace = load_mooncake_trace(sys.argv[1])\n"
            "out = {}\n"
            "for arm in ('label_binary_6', 'evict_binary_learned', 'evict_binary_recency',\n"
            "            'adm_label', 'evict_label'):\n"
            "    _, eligibility, _ = horizonctl.arm_mechanism(arm)\n"
            "    setup = horizonctl.arm_setup(arm, trace, 600.0, 2, learned_ranker=Ranker())\n"
            "    r = run_two_tier(trace, 'lru', 4 * 512, 8 * 512, bytes_per_token=1,\n"
            "                     l2_eviction='sampled', l2_sample_width=4, l2_seed=2,\n"
            "                     l2_eligibility=eligibility, l2_override_hook=setup.override,\n"
            "                     **setup.replay_arguments())\n"
            "    out[arm] = r.as_row()\n"
            "print(json.dumps(out, sort_keys=True, default=str))\n"
        )
        outputs = []
        for hash_seed in ("0", "4242"):
            environment = dict(os.environ, PYTHONHASHSEED=hash_seed,
                               PYTHONPATH=error_tests.SOURCE_ROOT)
            finished = subprocess.run([sys.executable, "-c", child, str(path)],
                                      capture_output=True, text=True, env=environment)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            outputs.append(finished.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_the_composite_keys_reach_m1(self):
        # The statistics' arm key is the store's key, Y's: for
        # evict_binary_learned the nested composite, compared element by element.
        _, stats, tracer, _ = replay(self.trace, "evict_binary_learned")
        self.assertTrue(all(isinstance(d[4][0][0], tuple) for d in tracer.decisions))
        self.assertGreater(stats.row()["m1_pairs"], 0)
        self.assertTrue(0.0 <= stats.row()["m1"] <= 1.0)


# --- the reading helpers ------------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_shortfall(self):
        self.assertEqual(horizonctl.shortfall(10.0, 9.0, 0.0), 0.1)
        self.assertEqual(horizonctl.shortfall(10.0, 10.0, 2.0), 0.0)
        self.assertEqual(horizonctl.shortfall(10.0, 2.0, 2.0), 1.0)
        self.assertEqual(horizonctl.shortfall(10.0, 12.0, 2.0), -0.25)
        self.assertTrue(math.isnan(horizonctl.shortfall(5.0, 3.0, 5.0)))

    def test_the_horizon_reading_at_its_boundary_ties_and_nan(self):
        reading = horizonctl.horizon_reading
        self.assertEqual(reading({6.0: 0.4, 15.0: 0.10, 600.0: 0.3}),
                         ("reuse_label_suffices", 0.10, (15.0,)))
        self.assertEqual(reading({6.0: 0.4, 15.0: 0.1000001, 600.0: 0.3})[0], "order_needed")
        self.assertEqual(reading({6.0: 0.4, 300.0: 0.2, 600.0: 0.2}),
                         ("order_needed", 0.2, (300.0, 600.0)))
        self.assertEqual(reading({600.0: -0.5, 6.0: 0.0}),
                         ("reuse_label_suffices", -0.5, (600.0,)))
        label, smallest, attaining = reading({6.0: math.nan, 600.0: math.nan})
        self.assertEqual((label, attaining), ("order_needed", ()))
        self.assertTrue(math.isnan(smallest))
        self.assertEqual(reading({6.0: math.nan, 600.0: 0.05})[2], (600.0,))
        self.assertEqual(set(horizonctl.HORIZON_READINGS), {"reuse_label_suffices", "order_needed"})

    def test_the_published_subset_is_strictly_above_five_percent(self):
        self.assertFalse(horizonctl.in_published_subset(0.05))
        self.assertTrue(horizonctl.in_published_subset(0.0500001))
        self.assertFalse(horizonctl.in_published_subset(-0.1))
        self.assertFalse(horizonctl.in_published_subset(math.nan))

    def test_the_recovery_share_and_the_class_order_reading(self):
        self.assertEqual(horizonctl.recovery(29.0, 2.0, 32.0), 0.9)
        self.assertTrue(math.isnan(horizonctl.recovery(3.0, 2.0, 2.0)))
        reading = horizonctl.class_order_reading
        self.assertEqual(reading(0.9), "reuse_identification_suffices")
        self.assertEqual(reading(1.4), "reuse_identification_suffices")
        self.assertEqual(reading(0.8999999), "ranker_order_costs")
        self.assertEqual(reading(-2.0), "ranker_order_costs")
        self.assertEqual(reading(math.nan), "ranker_order_costs")

    def test_the_leaf_location_and_its_comparison(self):
        self.assertEqual(horizonctl.leaf_location(2.0, 0.5, 1.0, "eviction_located"),
                         ("eviction_located", True))
        self.assertEqual(horizonctl.leaf_location(2.0, 1.0, 0.5, "eviction_located"),
                         ("admission_located", False))
        self.assertEqual(horizonctl.leaf_location(2.0, 1.0, 1.0, "both"), ("both", True))
        self.assertEqual(horizonctl.leaf_location(math.nan, 1.0, 1.0, "neither"),
                         ("neither", True))
        with self.assertRaises(ValueError):
            horizonctl.leaf_location(2.0, 1.0, 1.0, "located")


# --- the runner on synthetic rows ---------------------------------------------------------------------

CELL_A = ("conversation_trace", 0.01, 4.0)
CELL_B = ("toolagent_trace", 0.01, 4.0)
CELL_C = ("toolagent_trace", 0.02, 4.0)
# U in tokens over 1000 requested tokens (points = tokens / 10), per trace x
# cell, chosen so that every reading has a known answer. Published:
PUBLISHED_U = {
    CELL_A: {("all16", "lru"): 100, ("all16", "learned"): 200, ("all16", "label"): 600,
             ("all16", "evict_label"): 500, ("all16", "label_binary"): 520,
             ("leaf16", "learned"): 210, ("leaf16", "label"): 610},
    CELL_B: {("all16", "lru"): 100, ("all16", "learned"): 200, ("all16", "label"): 600,
             ("all16", "evict_label"): 500, ("all16", "label_binary"): 535,
             ("leaf16", "learned"): 210, ("leaf16", "label"): 610},
    CELL_C: {("all16", "lru"): 100, ("all16", "learned"): 200, ("all16", "label"): 600,
             ("all16", "evict_label"): 500, ("all16", "label_binary"): 598,
             ("leaf16", "learned"): 210, ("leaf16", "label"): 610},
}
# New arms. A: S = 0.6, 0.3, 0.08, 0.11, 0.16 (suffices at 60, subset);
# B: 0.4, 0.2, 0.16, 0.12, 0.13 (order needed, smallest at 300, subset);
# C: 0.02, 0.01, 0.006, 0.002, 0.004 (suffices at 300; published S_600 0.004,
# not in the subset). R(evict_binary_learned): A 0.933, B 0.8, C 0.95. Leaf16:
# A eviction_located, B admission_located, C both.
NEW_U = {
    CELL_A: {"label_binary_6": 300, "label_binary_15": 450, "label_binary_60": 560,
             "label_binary_300": 545, "label_binary_600": 520, "evict_binary_learned": 480,
             "evict_binary_recency": 400, "adm_label": 220, "evict_label": 590},
    CELL_B: {"label_binary_6": 400, "label_binary_15": 500, "label_binary_60": 520,
             "label_binary_300": 540, "label_binary_600": 535, "evict_binary_learned": 440,
             "evict_binary_recency": 470, "adm_label": 450, "evict_label": 300},
    CELL_C: {"label_binary_6": 590, "label_binary_15": 595, "label_binary_60": 597,
             "label_binary_300": 599, "label_binary_600": 598, "evict_binary_learned": 485,
             "evict_binary_recency": 485, "adm_label": 420, "evict_label": 420},
}
# Seed offsets with zero mean: B's label_binary_6 mixes the seed signs of
# U(label) - U(label_binary_6); C's recency arm mixes learned minus recency.
OFFSETS = {(CELL_B, "label_binary_6"): (250, -50, -50, -50, -100),
           (CELL_C, "evict_binary_recency"): (5, -5, 0, 5, -5)}
PUBLISHED_LOCATIONS = {CELL_A: "eviction_located", CELL_B: "eviction_located", CELL_C: "both"}
IDENTIFIERS = {"l1_capacity_bytes": 1, "l2_capacity_bytes": 4, "requested_tokens": 1000,
               "l1_avoided_tokens": 50, "absent_compulsory_tokens": 77}


def synthetic_published():
    published = {}
    for cell, arms in PUBLISHED_U.items():
        for (mechanism, arm), tokens in arms.items():
            for seed in range(5):
                published[(mechanism, arm) + cell + (seed,)] = {
                    "source": "s", "mechanism": mechanism, "arm": arm, "trace": cell[0],
                    "l1_fraction": cell[1], "l2_multiplier": cell[2],
                    "cell": runner.rdp.cell_label(*cell[1:]), "seed": seed,
                    "avoided_prefill_tokens": tokens + seed + 50,
                    "extra_avoided_tokens": tokens + seed, **IDENTIFIERS}
    return published


def synthetic_locations():
    out = {}
    for cell, label in PUBLISHED_LOCATIONS.items():
        out[cell] = {"location": label, "G": 40.0, "A": 1.0, "E": 30.0, "interaction": -9.0}
    return out


def synthetic_rows():
    rows = []
    for cell, arms in NEW_U.items():
        for seed in range(5):
            for arm, tokens in arms.items():
                tokens += seed + OFFSETS.get((cell, arm), (0,) * 5)[seed]
                mechanism, eligibility, width = arm_mechanism(arm)
                row = {metric: 0.0 for metric in runner.REPLAY_METRICS}
                row.update(trace=cell[0], l1_fraction=cell[1], l2_multiplier=cell[2],
                           cell=runner.rdp.cell_label(*cell[1:]), arm=arm, seed=seed,
                           mechanism=mechanism, eligibility=eligibility, width=width,
                           variant="main", extra_avoided_tokens=tokens,
                           avoided_prefill_tokens=tokens + 50, extra_points=tokens / 10.0,
                           **IDENTIFIERS)
                rows.append(row)
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.published = synthetic_published()

    def test_horizon_tables(self):
        seeds, per_h, readings = runner.horizon_tables(self.rows, self.published)
        self.assertEqual(len(seeds), 3 * 5 * 5)
        self.assertEqual(len(per_h), 3 * 5)
        by = {runner._cell_key(entry): entry for entry in readings}
        expected = {CELL_A: ((0.6, 0.3, 0.08, 0.11, 0.16), "reuse_label_suffices", "60", 0.16),
                    CELL_B: ((0.4, 0.2, 0.16, 0.12, 0.13), "order_needed", "300", 0.13),
                    CELL_C: ((0.02, 0.01, 0.006, 0.002, 0.004), "reuse_label_suffices", "300",
                             0.004)}
        for cell, (values, reading, horizon, published_s600) in expected.items():
            entry = by[cell]
            for h, value in zip(HORIZONS_SECONDS, values):
                self.assertAlmostEqual(entry[f"S_{h:g}"], value, places=9)
            self.assertEqual(entry["reading"], reading)
            self.assertEqual(entry["S_min_horizons"], horizon)
            self.assertAlmostEqual(entry["published_S_600"], published_s600, places=9)
            self.assertAlmostEqual(entry["S_learned"], 0.8, places=9)
        self.assertEqual([by[cell]["in_published_subset"] for cell in (CELL_A, CELL_B, CELL_C)],
                         [True, True, False])
        signs = {(runner._cell_key(entry), entry["h"]): entry for entry in per_h}
        self.assertEqual(signs[(CELL_B, 6.0)]["label_minus_arm_reading"], "mixed")
        self.assertEqual(signs[(CELL_B, 6.0)]["label_minus_arm_seed_signs"], "-++++")
        self.assertEqual(signs[(CELL_A, 6.0)]["label_minus_arm_reading"], "consistent_gain")
        self.assertEqual(signs[(CELL_C, 300.0)]["label_minus_arm_reading"], "consistent_gain")

    def test_class_order_tables(self):
        seeds, table = runner.class_order_tables(self.rows, self.published)
        self.assertEqual(len(seeds), 15)
        by = {runner._cell_key(entry): entry for entry in table}
        expected = {CELL_A: (280 / 300, 200 / 300, "reuse_identification_suffices",
                             "consistent_gain"),
                    CELL_B: (0.8, 0.9, "ranker_order_costs", "consistent_loss"),
                    CELL_C: (0.95, 0.95, "reuse_identification_suffices", "mixed")}
        for cell, (r_learned, r_recency, reading, difference) in expected.items():
            entry = by[cell]
            self.assertAlmostEqual(entry["R_evict_binary_learned"], r_learned, places=9)
            self.assertAlmostEqual(entry["R_evict_binary_recency"], r_recency, places=9)
            self.assertEqual(entry["reading"], reading)
            self.assertEqual(entry["learned_minus_recency_reading"], difference)
        self.assertEqual(by[CELL_C]["learned_minus_recency_seed_signs"], "-+0-+")

    def test_leaf_location_tables(self):
        seeds, table = runner.leaf_location_tables(self.rows, self.published,
                                                   synthetic_locations())
        self.assertEqual(len(seeds), 15)
        by = {runner._cell_key(entry): entry for entry in table}
        expected = {CELL_A: (40.0, 1.0, 38.0, "eviction_located", True),
                    CELL_B: (40.0, 24.0, 9.0, "admission_located", False),
                    CELL_C: (40.0, 21.0, 21.0, "both", True)}
        for cell, (g, a, e, label, same) in expected.items():
            entry = by[cell]
            self.assertAlmostEqual(entry["G_points_mean"], g, places=9)
            self.assertAlmostEqual(entry["A_points_mean"], a, places=9)
            self.assertAlmostEqual(entry["E_points_mean"], e, places=9)
            self.assertAlmostEqual(entry["interaction_points_mean"], a + e - g, places=9)
            self.assertEqual((entry["location"], entry["same_as_all16"]), (label, same))
            self.assertEqual(entry["all16_location"], PUBLISHED_LOCATIONS[cell])
            self.assertEqual(entry["mechanism"], "leaf16")

    def test_aggregation_and_reference_rows(self):
        summary = runner.aggregate_replays(self.rows)
        self.assertEqual(len(summary), 3 * 9)
        entry = next(e for e in summary if runner._cell_key(e) == CELL_A
                     and e["arm"] == "label_binary_60")
        self.assertEqual((entry["mechanism"], entry["family"], entry["arm_parameter"]),
                         ("all16", "horizon", 60.0))
        self.assertAlmostEqual(entry["extra_points_mean"], 56.2)
        self.assertEqual([e["arm"] for e in summary[:9]], list(ARMS))
        references = runner.reference_rows(self.published, self.rows)
        self.assertEqual(len(references), 3 * 5 * len(runner.REFERENCES))
        self.assertAlmostEqual(references[0]["extra_points"],
                               references[0]["extra_avoided_tokens"] / 10.0)

    def test_the_figure_is_written(self):
        seeds, _, readings = runner.horizon_tables(self.rows, self.published)
        with tempfile.TemporaryDirectory() as directory:
            runner.horizon_figure(Path(directory), readings, seeds)
            self.assertGreater((Path(directory) / "horizon.png").stat().st_size, 10_000)

    def test_tabulation_reads_and_prints_without_writing(self):
        _, per_h, readings = runner.horizon_tables(self.rows, self.published)
        _, class_order = runner.class_order_tables(self.rows, self.published)
        _, leaf = runner.leaf_location_tables(self.rows, self.published, synthetic_locations())
        tables = {"replay": runner.aggregate_replays(self.rows), "horizon": per_h,
                  "horizon_reading": readings, "class_order": class_order, "location_leaf": leaf}
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in tables.items():
                runner._write(Path(directory) / f"{name}.csv", rows)
            before = sorted(os.listdir(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(tabulate.main([directory]), 0)
            self.assertEqual(sorted(os.listdir(directory)), before)
            text = stdout.getvalue()
            parsed = {name: tabulate.read_csv(Path(directory) / f"{name}.csv") for name in tables}
        for section in ("R0", "R1", "R2", "R3"):
            self.assertIn(f"### {section} ", text)
        counts = tabulate.horizon_counts(parsed["horizon_reading"])
        self.assertEqual(counts[0], {"scope": "all", "reuse_label_suffices": 2,
                                     "order_needed": 1, "cells": 3})
        self.assertEqual((counts[1]["reuse_label_suffices"], counts[1]["order_needed"],
                          counts[1]["cells"]), (1, 1, 2))
        self.assertEqual(tabulate.class_order_counts(parsed["class_order"]), [{
            "reuse_identification_suffices": 2, "ranker_order_costs": 1, "consistent_gain": 1,
            "consistent_loss": 1, "mixed": 1, "cells": 3}])
        self.assertEqual(tabulate.location_leaf_counts(parsed["location_leaf"]), [{
            "admission_located": 1, "eviction_located": 1, "both": 1, "neither": 0,
            "same_as_all16": 2, "cells": 3}])
        reading_rows = tabulate.horizon_reading_table(parsed["horizon_reading"])
        self.assertEqual([row["S_min_horizons"] for row in reading_rows], ["60", "300", "300"])
        self.assertEqual([row["arm"] for row in tabulate.replay_counters(parsed["replay"])[:9]],
                         list(ARMS))

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "horizon_seeds", "horizon",
                     "horizon_reading", "class_order_seeds", "class_order",
                     "location_leaf_seeds", "location_leaf"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        self.assertIn("`horizon.png`", runner.README_TEXT)
        self.assertIn("`run_config.json`", runner.README_TEXT)
        self.assertIn("`smoke_identities.csv`", runner.SMOKE_README)


# --- the runner's checks and refusals -------------------------------------------------------------------


def _stats_row(arm, variant="main", **changes):
    """A replay row as the statistics, closure and identity checks read it."""
    mechanism, eligibility, width = arm_mechanism(arm)
    row = {"trace": "conversation_trace", "l1_fraction": 0.01, "l2_multiplier": 4.0,
           "cell": "l1=0.01,l2x4", "seed": 0, "arm": arm, "variant": variant,
           "mechanism": mechanism, "eligibility": eligibility, "width": width,
           "l2_decisions": 10, "l2_rejections": 3, "l2_evictions": 7,
           "l2_present_unusable_tokens": 0, "l2_present_unusable_blocks": 0,
           "stat_decisions_seen": 10, "stat_rejections_seen": 3, "stat_evictions_seen": 7,
           "overridden_decisions_seen": 0, "stat_decisions": 6, "m1": 1.0,
           "counters_sha256": "c", "decision_sha256": "d", "seconds": 1.0, **IDENTIFIERS}
    for suffix in ("", "_admission", "_resident"):
        row.update({f"stat_decisions{suffix}": 6 if not suffix else 3, f"m2{suffix}": 1.0,
                    f"m3{suffix}": 0.0, f"m4_count{suffix}": 1,
                    f"m4_victim_at_horizon{suffix}": 1})
    row.update(changes)
    return row


class RunnerCheckTests(unittest.TestCase):
    def test_the_grid_and_the_task_order(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS, smoke=False)
        self.assertEqual(len(tasks), 540)
        self.assertEqual(len(set(tasks)), 540)
        self.assertEqual({task[3] for task in tasks}, set(ARMS))
        self.assertEqual({task[5] for task in tasks}, {"main"})
        costs = [runner.arm_cost(task[3]) for task in tasks]
        self.assertEqual(costs, sorted(costs))
        smoke = runner.build_tasks(("conversation_trace",), (runner.SMOKE_CELL,),
                                   runner.SMOKE_SEEDS, smoke=True)
        self.assertEqual(len(smoke), 9 + 9 + 3 + 2)
        self.assertEqual({task[3] for task in smoke if task[5] == "nostats"}, set(ARMS))
        self.assertEqual({task[3] for task in smoke},
                         set(ARMS) | set(IDENTITY_ARMS) | set(IDENTITY_REFERENCE_ARMS))
        self.assertEqual((runner.SMOKE_TRACE, runner.SMOKE_CELL, runner.SMOKE_SEEDS),
                         ("conversation_trace", (0.01, 4.0), (0,)))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))

    def test_reproduction(self):
        published = synthetic_published()
        rows = [row for row in synthetic_rows() if row["arm"] == "label_binary_600"]
        report = runner.check_reproduction(rows, published, {"label_binary_600": 15})
        self.assertEqual((report["label_binary_600"]["matched"],
                          report["label_binary_600"]["mismatched"]), (15, 0))
        self.assertTrue(all(row["reproduces_reference"] for row in rows))
        rows[0]["avoided_prefill_tokens"] += 1
        del published[("all16", "label_binary") + CELL_C + (4,)]
        report = runner.check_reproduction(rows, published, {"label_binary_600": 15})["label_binary_600"]
        self.assertEqual((report["matched"], report["mismatched"], report["missing"]), (13, 1, 1))
        self.assertEqual(report["reference"], "label_binary/all16")
        # The smoke's leaf16 rungs are held to their own published rows.
        leaf = [dict(rows[1], arm="learned", mechanism="leaf16",
                     avoided_prefill_tokens=published[("leaf16", "learned") + CELL_A + (1,)]
                     ["avoided_prefill_tokens"])]
        report = runner.check_reproduction(leaf, published, {"learned": 1})["learned"]
        self.assertEqual((report["matched"], report["reference"]), (1, "learned/leaf16"))

    def test_identifiers_against_every_published_reference(self):
        published = synthetic_published()
        rows = synthetic_rows()
        compared, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])
        self.assertEqual(compared, len(rows) * len(runner.REFERENCES))
        rows[0]["absent_compulsory_tokens"] = 78
        rows[1]["l2_capacity_bytes"] = 5
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 2 * len(runner.REFERENCES))
        self.assertTrue(any("absent_compulsory_tokens 78 != 77" in line for line in problems))
        # A reference that does not publish an identifier is not compared on it;
        # a missing reference row is a problem.
        for key in published:
            published[key].pop("absent_compulsory_tokens")
        del published[("leaf16", "label") + CELL_A + (0,)]
        _, problems = runner.check_identifiers(rows, published)
        self.assertTrue(any("l2_capacity_bytes 5" in line for line in problems))
        self.assertFalse(any("absent_compulsory" in line for line in problems))
        self.assertTrue(any("no published label/leaf16 reference row" in line
                            for line in problems))

    def test_leaf_closure(self):
        rows = [_stats_row("adm_label"), _stats_row("label_binary_6",
                                                    l2_present_unusable_tokens=512),
                _stats_row("evict_label", l2_present_unusable_blocks=1)]
        problems = runner.check_leaf_closure(rows)
        self.assertEqual(len(problems), 1)
        self.assertIn("evict_label", problems[0])

    def test_statistics(self):
        self.assertEqual(runner.check_statistics([_stats_row(arm) for arm in ARMS]), [])
        keep = "overridden decisions for an arm that must keep the store's choice"
        cases = (
            (_stats_row("evict_label", stat_decisions_seen=9),
             "stat_decisions_seen 9 != l2_decisions 10"),
            (_stats_row("evict_binary_recency", stat_rejections_seen=4),
             "stat_rejections_seen 4 != l2_rejections 3"),
            (_stats_row("adm_label", stat_decisions=0), "no decision inside"),
            (_stats_row("label_binary_60", overridden_decisions_seen=2), keep),
            (_stats_row("learned", overridden_decisions_seen=2), keep),
            (_stats_row("hybrid_learned_learned", overridden_decisions_seen=1), keep),
            (_stats_row("adm_label_binary_evict_binary_recency", overridden_decisions_seen=1),
             keep),
            (_stats_row("hybrid_label_label", m2_resident=0.5), "label identity m2_resident"),
            (_stats_row("label", m3=0.01), "label identity m3 "),
            (_stats_row("label", m4_count_admission=2),
             "m4_count_admission = 2 != m4_victim_at_horizon_admission = 1"),
        )
        for row, text in cases:
            with self.subTest(case=text, arm=row["arm"]):
                problems = runner.check_statistics([row])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])
        # An unused decision type is not held to the identity.
        self.assertEqual(runner.check_statistics([_stats_row(
            "label", stat_decisions_resident=0, m2_resident=math.nan, m4_count_resident=5)]), [])
        # Override arms may override; the label identity binds only the leaf16 label.
        self.assertEqual(runner.check_statistics([
            _stats_row("evict_binary_learned", overridden_decisions_seen=4),
            _stats_row("learned", m2=0.3), _stats_row("label_binary_600", m1=0.4)]), [])
        self.assertEqual(runner.check_statistics([_stats_row("adm_label", variant="nostats",
                                                             stat_decisions=0)]), [])

    def test_smoke_identities(self):
        rows = [_stats_row(arm) for arm in ARMS + tuple(IDENTITY_ARMS) + IDENTITY_REFERENCE_ARMS]
        rows += [_stats_row(arm, variant="nostats") for arm in ARMS]
        table, problems = runner.check_smoke_identities(rows)
        self.assertEqual(problems, [])
        self.assertEqual(len(table), len(IDENTITY_ARMS) + len(ARMS))
        broken = [dict(row) for row in rows]
        next(row for row in broken if row["arm"] == "hybrid_label_label")["decision_sha256"] = "x"
        next(row for row in broken if row["arm"] == "evict_label"
             and row["variant"] == "nostats")["counters_sha256"] = "x"
        _, problems = runner.check_smoke_identities(broken)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("hybrid_label_label vs label" in line and "decision_sha256" in line
                            for line in problems))
        _, problems = runner.check_smoke_identities([row for row in rows if row["arm"] != "label"])
        self.assertTrue(any("missing" in line for line in problems))

    def test_the_published_references_load_as_the_plan_names_them(self):
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        self.assertEqual(len(published), len(runner.REFERENCES) * 60)
        entry = published[("all16", "learned", "conversation_trace", 0.01, 4.0, 0)]
        self.assertEqual(entry["source"], "error_location_001")
        self.assertEqual(published[("leaf16", "label", "toolagent_trace", 0.02, 1.0, 4)]["source"],
                         "mechanism_control_001")
        for key, entry in published.items():
            self.assertEqual(set(runner.REFERENCE_IDENTIFIERS) - set(entry), set())
        locations, problems = runner.load_published_locations()
        self.assertEqual(problems, [])
        self.assertEqual(len(locations), 12)

    def test_reference_loading_refuses_what_the_plan_does_not_name(self):
        header = ("trace,l1_fraction,l2_multiplier,cell,eligibility,width,mechanism,{},seed,"
                  "variant,avoided_prefill_tokens,extra_avoided_tokens,l1_capacity_bytes,"
                  "l2_capacity_bytes,requested_tokens,l1_avoided_tokens\n")
        line = "conversation_trace,0.01,4.0,c,{},16,{},{},0,main,10,5,1,4,100,5\n"
        with tempfile.TemporaryDirectory() as directory:
            errors = Path(directory) / "e.csv"
            errors.write_text(header.format("arm") + line.format("all", "all16", "lru")
                              + line.format("all", "all16", "lru")
                              + line.format("leaf", "all16", "learned")
                              + line.format("leaf", "leaf16", "learned"))
            mechanism = Path(directory) / "m.csv"
            mechanism.write_text(header.format("rung").replace("variant,", "")
                                 + line.format("leaf", "leaf16", "label").replace("main,", "")
                                 + line.format("all", "all16", "learned").replace("main,", ""))
            published, problems = runner.load_references(
                {"error_location_001": errors, "mechanism_control_001": mechanism})
        self.assertIn(("all16", "lru", "conversation_trace", 0.01, 4.0, 0), published)
        self.assertIn(("leaf16", "label", "conversation_trace", 0.01, 4.0, 0), published)
        # all16 learned comes from the error-location rows only, leaf16 learned
        # from the mechanism control's only.
        self.assertEqual(published[("all16", "learned", "conversation_trace", 0.01, 4.0, 0)]
                         ["source"], "error_location_001")
        self.assertNotIn(("leaf16", "learned", "conversation_trace", 0.01, 4.0, 0), published)
        self.assertTrue(any(line.startswith("duplicate") for line in problems))
        self.assertTrue(any("has leaf/16" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))
        self.assertNotIn("absent_compulsory_tokens",
                         published[("all16", "lru", "conversation_trace", 0.01, 4.0, 0)])

    def test_a_published_location_off_its_rule_is_refused(self):
        rows = list(csv_rows(REPOSITORY / "results/paper/error_location_001/location.csv"))
        rows[0]["location"] = "admission_located"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "location.csv"
            runner._write(path, rows[:-1])
            _, problems = runner.load_published_locations(path)
        self.assertTrue(any("not the rule" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))

    def test_argument_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "new"
            existing = Path(directory)
            cases = {
                "positive": ["--workers", "0"],
                "hard cap of 12": ["--workers", "13"],
                "--allow-dirty is for --smoke": ["--allow-dirty"],
                "never writes a paper": ["--smoke", "--paper-dir", str(fresh / "p")],
                "pre-registered grid": ["--seeds", "4"],
            }
            for text, extra in cases.items():
                with self.subTest(case=text), self.assertRaises(SystemExit) as caught:
                    runner.validate_arguments(runner.parse_args(
                        ["t.jsonl", "--output-dir", str(fresh)] + extra))
                self.assertIn(text, str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(
                    ["t.jsonl", "--output-dir", str(existing)]))
            self.assertIn("exists", str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(
                    ["t.jsonl", "--output-dir", str(fresh), "--paper-dir", str(existing)]))
            self.assertIn("paper directory", str(caught.exception))
            inside = REPOSITORY / "results/paper/horizon_smoke_never"
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(
                    ["t.jsonl", "--smoke", "--output-dir", str(inside)]))
            self.assertIn("never writes under results/paper", str(caught.exception))
            self.assertFalse(inside.exists())
            runner.validate_arguments(runner.parse_args(
                ["t.jsonl", "--smoke", "--allow-dirty", "--workers", "12", "--output-dir",
                 str(fresh)]))
            self.assertEqual(runner.parse_args(["t.jsonl", "--output-dir", "x"]).workers, 10)

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["run_horizon_control.py", "missing.jsonl", "--output-dir", str(output)]

            def git(*arguments):
                return " M src/x.py" if arguments[0] == "status" else "0" * 40

            for status, differing, text in ((git, [], "not clean"),
                                            (lambda *a: "" if a[0] == "status" else "0" * 40,
                                             ["src/x.py"], "differs from HEAD")):
                with self.subTest(case=text), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(runner.rmc, "_git", status), \
                        mock.patch.object(runner, "sources_differing_from_head",
                                          lambda: differing), \
                        self.assertRaises(SystemExit) as caught:
                    runner.main()
                self.assertIn(text, str(caught.exception))
                self.assertFalse(output.exists())

    def test_the_execution_sources_cover_the_imported_runners(self):
        sources = {str(path.relative_to(REPOSITORY)) for path in runner.execution_sources()}
        for name in ("scripts/run_horizon_control.py", "scripts/run_error_location.py",
                     "scripts/run_mechanism_control.py", "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/horizonctl.py",
                     "src/persistent_kv_admission/errorloc.py"):
            self.assertIn(name, sources)


def csv_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker`, the checks and the smoke comparisons on a constructed
    trace, with the module state the runner's main would set."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        name = self.trace.name
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON},
                      rankers={name: _LinearRanker()})
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def test_worker_rows_pass_the_checks_and_the_smoke_comparisons(self):
        arms = ("label_binary_15", "label_binary_600", "evict_binary_learned",
                "evict_binary_recency", "adm_label", "evict_label")
        tasks = [(self.name, 0.01, 4.0, arm, 0, variant) for arm in arms
                 for variant in ("main", "nostats")]
        tasks += [(self.name, 0.01, 4.0, arm, 0, "main")
                  for arm in tuple(IDENTITY_ARMS) + IDENTITY_REFERENCE_ARMS]
        rows = [runner._replay_worker(task) for task in tasks]
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_leaf_closure(rows), [])
        for row in rows:
            self.assertEqual(row["mechanism"], ARM_MECHANISMS[row["arm"]])
            self.assertEqual(row["eligibility"], "leaf" if row["mechanism"] == "leaf16" else "all")
            self.assertGreater(row["l2_decisions"], 0)
            if row["variant"] == "main":
                self.assertEqual(row["stat_decisions_seen"], row["l2_decisions"])
            else:
                self.assertNotIn("m1", row)
        main_rows = [row for row in rows if row["variant"] == "main" and row["arm"] in arms]
        _, problems = runner.rmc.check_invariants(main_rows, len(arms))
        self.assertEqual(problems, [])
        self.assertEqual(rows[0]["arm_parameter"], 15.0)
        table, problems = runner.check_smoke_identities(rows)
        held = [entry for entry in table if entry["holds"]]
        self.assertEqual(len([e for e in held if e["check"] == "identity"]), len(IDENTITY_ARMS))
        self.assertEqual(len([e for e in held if e["check"] == "hook_read_only"]), len(arms))
        # The grid arms not replayed here are reported missing, not passed.
        self.assertEqual(len(problems), len(ARMS) - len(arms))
        self.assertTrue(all("missing" in line for line in problems))
        # A synthetic publication of these very rows passes the identifier check
        # and holds label_binary_600 and the leaf16 rungs to it.
        published = {}
        for row in rows:
            if row["variant"] != "main":
                continue
            for (mechanism, arm) in runner.REFERENCES:
                published[(mechanism, arm) + runner._cell_seed(row)] = {
                    "source": "s", "avoided_prefill_tokens": row["avoided_prefill_tokens"],
                    **{field: row[field] for field in runner.REFERENCE_IDENTIFIERS}}
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
