"""Tests for the horizon fill-in (docs/horizon-fill-plan.md).

No real trace is replayed and no real results directory is read: the plan
forbids a real replay before the implementation is reviewed. Everything here
runs on constructed traces, hand-written shortfalls and synthetic rows. What
has to hold before a real replay means anything: the grid is the plan's; each
`label_binary_h` arm is the exact `binary` target at h (a next use at exactly
h counts as reuse) built as the horizon control builds its own horizon arms,
so that at h = 60, 300 and 600 the fill-in's replay is the horizon control's,
decision by decision and column by column; the reading helpers do what the
plan fixes at their boundaries; and the runner's checks, refusals,
derivations, end-to-end run and read-only tabulation behave as stated,
including that no ranker is loaded and that a failed check publishes nothing.
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
from pathlib import Path
from unittest import mock

from persistent_kv_admission import errorloc, horizonctl, horizonfill
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import horizon_arm
from persistent_kv_admission.horizonfill import (
    ARMS,
    CHECK_ARMS,
    CHECK_HORIZONS_SECONDS,
    CURVE_HORIZONS_SECONDS,
    FILL_CELLS,
    FILL_HORIZONS_SECONDS,
    FILL_READINGS,
    HORIZON_OF_ARM,
    HORIZONS_SECONDS,
)
from persistent_kv_admission.mechanism import ExactLabelScorer
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The horizon control's fixtures (constructed traces at one step per 5 s, the
# traced replay helper), reused as they are; only plain functions are read.
horizon_tests = _load("_fill_horizon_control_fixtures", REPOSITORY / "tests/test_horizon_control.py")
runner = _load("_run_horizon_fill_under_test", REPOSITORY / "scripts/run_horizon_fill.py")
tabulate = _load("_tabulate_horizon_fill_under_test",
                 REPOSITORY / "scripts/tabulate_horizon_fill.py")

HORIZON = horizon_tests.HORIZON
MEASURE_FROM_MS = horizon_tests.MEASURE_FROM_MS
partial_trace = horizon_tests.partial_trace
replay = horizon_tests.replay
_row = horizon_tests._row
# Columns that differ between two replays of the same arm on the same input.
VOLATILE = ("seconds", "worker_peak_rss_mib", "worker_pss_mib_end")


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


# --- the grid ---------------------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_horizons_cells_and_arms_are_the_plans(self):
        self.assertEqual(FILL_HORIZONS_SECONDS, (90.0, 120.0, 150.0, 180.0, 240.0))
        self.assertEqual(CHECK_HORIZONS_SECONDS, (60.0, 300.0, 600.0))
        self.assertEqual(HORIZONS_SECONDS, (60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 600.0))
        self.assertEqual(CURVE_HORIZONS_SECONDS, (60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0))
        self.assertEqual(ARMS, ("label_binary_60", "label_binary_90", "label_binary_120",
                                "label_binary_150", "label_binary_180", "label_binary_240",
                                "label_binary_300", "label_binary_600"))
        self.assertEqual(CHECK_ARMS, ("label_binary_60", "label_binary_300", "label_binary_600"))
        self.assertEqual(horizonfill.PUBLISHED_HORIZON_ARM, horizonctl.PUBLISHED_HORIZON_ARM)
        self.assertEqual(FILL_CELLS, ((0.0025, 4.0), (0.01, 1.0)))
        self.assertEqual(horizonfill.MECHANISM, "all16")
        self.assertEqual({h: horizonfill.ROLE_OF_HORIZON[h] for h in HORIZONS_SECONDS},
                         {60.0: "check", 90.0: "fill", 120.0: "fill", 150.0: "fill",
                          180.0: "fill", 240.0: "fill", 300.0: "check", 600.0: "check"})
        # Multiples of 30 s on the 3-second grid; the check horizons are the
        # horizon control's and the fill horizons are new.
        for h in FILL_HORIZONS_SECONDS:
            self.assertEqual(h % 30.0, 0.0)
            self.assertNotIn(h, horizonctl.HORIZONS_SECONDS)
        self.assertTrue(set(CHECK_HORIZONS_SECONDS) <= set(horizonctl.HORIZONS_SECONDS))
        self.assertEqual(FILL_READINGS, ("reuse_label_suffices_at_filled_horizon",
                                         "order_needed_stands"))
        self.assertEqual(horizonfill.PREDICTED_SUFFICING, 4)

    def test_the_runner_and_the_tabulation_use_the_same_grid(self):
        self.assertEqual(runner.CELLS, FILL_CELLS)
        self.assertEqual(runner.TRACES, ("conversation_trace", "toolagent_trace"))
        self.assertEqual(runner.SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH), ("all", 16))
        self.assertEqual(set(runner.REFERENCES), {("all16", "label"), ("all16", "lru"),
                                                  ("all16", "label_binary")})
        self.assertEqual(set(runner.REFERENCES.values()), {"error_location_001"})
        self.assertEqual(runner.ERROR_LOCATION_REFERENCE,
                         REPOSITORY / "results/paper/error_location_001/replay_seeds.csv")
        self.assertEqual(tabulate.HORIZONS, HORIZONS_SECONDS)
        self.assertEqual(tabulate.FILL_HORIZONS, FILL_HORIZONS_SECONDS)
        self.assertEqual(tabulate.FILL_READINGS, FILL_READINGS)
        self.assertEqual(tabulate.TRACES, runner.TRACES)


# --- the arms ---------------------------------------------------------------------------------------


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self)

    def test_every_arm_is_the_binary_label_at_its_horizon_without_override(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                setup = horizonfill.arm_setup(arm, self.trace)
                self.assertIs(type(setup.scorer), ExactLabelScorer)
                self.assertEqual((setup.scorer.target, setup.scorer.horizon_seconds),
                                 ("binary", HORIZON_OF_ARM[arm]))
                self.assertEqual(setup.scorer.label.horizon_ms, HORIZON_OF_ARM[arm] * 1000.0)
                self.assertIsNone(setup.override)
                self.assertEqual(setup.l2_policy, "learned")
                self.assertFalse(setup.sampled_offline)
                # A fresh scorer per call.
                self.assertIsNot(setup.scorer, horizonfill.arm_setup(arm, self.trace).scorer)

    def test_the_check_arms_are_built_as_the_horizon_control_builds_them(self):
        for arm in CHECK_ARMS:
            with self.subTest(arm=arm):
                mine = horizonfill.arm_setup(arm, self.trace)
                theirs = horizonctl.arm_setup(arm, self.trace, HORIZON, 0)
                self.assertIs(type(mine), type(theirs))
                self.assertIs(type(mine.scorer), type(theirs.scorer))
                self.assertIs(type(mine.scorer.label), type(theirs.scorer.label))
                for attribute in ("target", "horizon_seconds"):
                    self.assertEqual(getattr(mine.scorer, attribute),
                                     getattr(theirs.scorer, attribute))
                self.assertEqual(mine.scorer.label.horizon_ms, theirs.scorer.label.horizon_ms)
                self.assertIs(mine.scorer.label.occurrences, theirs.scorer.label.occurrences)
                self.assertEqual((mine.arm, mine.l2_policy, mine.override, mine.sampled_offline),
                                 (theirs.arm, theirs.l2_policy, theirs.override,
                                  theirs.sampled_offline))
                self.assertEqual(mine.replay_arguments().keys(), theirs.replay_arguments().keys())

    def test_unknown_arms_are_refused(self):
        for arm in ("label_binary_6", "label_binary_15", "label_binary_100", "label_binary",
                    "evict_binary_learned", "lru", "label"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                horizonfill.arm_setup(arm, self.trace)

    def test_a_next_use_at_exactly_h_is_reuse_and_one_millisecond_later_is_not(self):
        for h in FILL_HORIZONS_SECONDS:
            with self.subTest(h=h):
                h_ms = int(h * 1000)
                stamps = (0, h_ms, 2 * h_ms + 1)
                records = [{"timestamp": t, "input_length": 512, "output_length": 1,
                            "hash_ids": [1]} for t in stamps]
                temporary, trace, _ = horizon_tests.build(records)
                self.addCleanup(temporary.cleanup)
                scorer = horizonfill.arm_setup(horizon_arm(h), trace).scorer
                self.assertEqual(scorer.score("int:1", 0.0), 1.0)              # exactly h
                self.assertEqual(scorer.score("int:1", float(h_ms)), 0.0)      # h + 1 ms
                self.assertEqual(scorer.score("int:1", float(h_ms) + 1.0), 1.0)  # exactly h
                self.assertEqual(scorer.score("int:1", float(stamps[-1])), 0.0)  # no further use
                for t in range(-500, stamps[-1] + 500, 997):
                    self.assertEqual(scorer.score("int:1", float(t)),
                                     horizon_tests._binary(trace, "int:1", float(t), h))

    def test_the_check_arms_replay_as_the_horizon_controls_decision_by_decision(self):
        for arm in CHECK_ARMS:
            with self.subTest(arm=arm):
                mine, mine_stats, mine_trace, _ = replay(
                    self.trace, arm, setup=horizonfill.arm_setup(arm, self.trace),
                    eligibility="all")
                theirs, theirs_stats, theirs_trace, _ = replay(self.trace, arm)
                self.assertEqual(mine_trace.decisions, theirs_trace.decisions)
                self.assertGreater(len(mine_trace.decisions), 100)
                self.assertEqual(_row(mine), _row(theirs))
                self.assertEqual(mine_stats.row(), theirs_stats.row())
        # h = 600 is also the error-location control's label_binary.
        mine, _, mine_trace, _ = replay(self.trace, "label_binary_600",
                                        setup=horizonfill.arm_setup("label_binary_600", self.trace))
        published, _, published_trace, _ = replay(
            self.trace, "label_binary_600",
            setup=errorloc.arm_setup("label_binary", self.trace, 600.0, 1))
        self.assertEqual(mine_trace.decisions, published_trace.decisions)
        self.assertEqual(_row(mine), _row(published))

    def test_every_horizon_changes_the_decisions(self):
        streams = [replay(self.trace, arm, setup=horizonfill.arm_setup(arm, self.trace),
                          eligibility="all")[2].decisions for arm in ARMS]
        for left in range(len(streams)):
            for right in range(left + 1, len(streams)):
                self.assertNotEqual(streams[left], streams[right], (ARMS[left], ARMS[right]))
        # No arm overrides the store: the victim is always the store's first minimum.
        for decisions in streams:
            self.assertTrue(all(d[5] == d[6] for d in decisions))


# --- the reading helpers ----------------------------------------------------------------------------


def _fill(values):
    return dict(zip(FILL_HORIZONS_SECONDS, values))


def _curve(values):
    return dict(zip(CURVE_HORIZONS_SECONDS, values))


class ReadingTests(unittest.TestCase):
    def test_the_fill_reading_at_its_boundary(self):
        suffices, stands = FILL_READINGS
        self.assertEqual(horizonfill.fill_reading(_fill((0.4, 0.10, 0.3, 0.5, 0.6))),
                         (suffices, 0.10, (120.0,)))
        self.assertEqual(horizonfill.fill_reading(_fill((0.4, 0.1000001, 0.3, 0.5, 0.6)))[0],
                         stands)
        self.assertEqual(horizonfill.fill_reading(_fill((0.4, 0.3, 0.2, 0.5, -0.5))),
                         (suffices, -0.5, (240.0,)))
        # Integer horizons are the same horizons.
        self.assertEqual(horizonfill.fill_reading({90: 0.2, 120: 0.2, 150: 0.2, 180: 0.2,
                                                   240: 0.05})[2], (240.0,))

    def test_ties_list_every_fill_horizon_attaining_the_minimum(self):
        self.assertEqual(horizonfill.fill_reading(_fill((0.3, 0.2, 0.25, 0.2, 0.4))),
                         (FILL_READINGS[1], 0.2, (120.0, 180.0)))
        self.assertEqual(horizonfill.fill_reading(_fill((0.05,) * 5))[2], FILL_HORIZONS_SECONDS)

    def test_nan_attains_nothing_and_all_nan_stands(self):
        nan = math.nan
        self.assertEqual(horizonfill.fill_reading(_fill((nan, 0.07, nan, 0.3, nan))),
                         (FILL_READINGS[0], 0.07, (120.0,)))
        label, smallest, attaining = horizonfill.fill_reading(_fill((nan,) * 5))
        self.assertEqual((label, attaining), (FILL_READINGS[1], ()))
        self.assertTrue(math.isnan(smallest))

    def test_wrong_or_missing_horizons_are_refused(self):
        cases = (
            {**_fill((0.1,) * 5), 60.0: 0.0},                               # a check horizon
            {h: 0.1 for h in FILL_HORIZONS_SECONDS[:4]},                    # 240 missing
            {**{h: 0.1 for h in FILL_HORIZONS_SECONDS[:4]}, 210.0: 0.1},    # 210 for 240
            {},
        )
        for shortfalls in cases:
            with self.subTest(shortfalls=sorted(shortfalls)), self.assertRaises(ValueError):
                horizonfill.fill_reading(shortfalls)
        for function in (horizonfill.curve_minimisers, horizonfill.curve_values,
                         horizonfill.sufficing_set, horizonfill.sufficing_width_seconds):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(_fill((0.1,) * 5))
                with self.assertRaises(ValueError):
                    function({**_curve((0.1,) * 7), 600.0: 0.0})

    def test_unimodal(self):
        unimodal = horizonfill.unimodal
        self.assertTrue(unimodal([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]))       # decreasing
        self.assertTrue(unimodal([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))        # increasing
        self.assertTrue(unimodal([0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4]))        # V
        self.assertTrue(unimodal([0.4, 0.3, 0.1, 0.1, 0.1, 0.3, 0.4]))        # flat bottom
        self.assertTrue(unimodal([0.4, 0.4, 0.2, 0.2, 0.3, 0.3, 0.5]))        # steps
        self.assertTrue(unimodal([0.2] * 7))                                  # flat
        self.assertTrue(unimodal([0.3]))
        self.assertFalse(unimodal([0.4, 0.2, 0.3, 0.1, 0.3, 0.4, 0.5]))       # W
        self.assertFalse(unimodal([0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.5]))       # W, second dip
        self.assertFalse(unimodal([0.1, 0.3, 0.2]))                           # a peak
        self.assertFalse(unimodal([0.2, 0.2, 0.3, 0.2, 0.2]))                 # flat peak
        self.assertFalse(unimodal([0.4, 0.3, math.nan, 0.1, 0.2, 0.3, 0.4]))  # nan
        self.assertFalse(unimodal([math.nan] * 7))
        self.assertFalse(unimodal([]))

    def test_curve_minimisers_and_whether_two_cells_share_them(self):
        curve = _curve((0.4, 0.3, 0.2, 0.2, 0.25, 0.3, 0.35))
        self.assertEqual(horizonfill.curve_values(curve), [0.4, 0.3, 0.2, 0.2, 0.25, 0.3, 0.35])
        self.assertEqual(horizonfill.curve_minimisers(curve), (120.0, 150.0))
        self.assertEqual(horizonfill.curve_minimisers(_curve((0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                              0.7))), (60.0,))
        self.assertEqual(horizonfill.curve_minimisers(_curve((math.nan,) * 7)), ())
        same = horizonfill.same_minimiser
        self.assertTrue(same((150.0,), (150.0,)))
        self.assertTrue(same((120.0, 150.0), (120.0, 150.0)))
        self.assertFalse(same((120.0, 150.0), (150.0,)))
        self.assertFalse(same((120.0,), (300.0,)))
        self.assertFalse(same((), ()))

    def test_the_sufficing_set_and_its_width(self):
        curve = _curve((0.3, 0.10, 0.05, 0.1000001, 0.09, 0.2, 0.4))
        self.assertEqual(horizonfill.sufficing_set(curve), (90.0, 120.0, 180.0))
        self.assertEqual(horizonfill.sufficing_width_seconds(curve), 90.0)
        single = _curve((0.3, 0.2, 0.2, 0.05, 0.2, 0.2, 0.3))
        self.assertEqual(horizonfill.sufficing_set(single), (150.0,))
        self.assertEqual(horizonfill.sufficing_width_seconds(single), 0.0)
        everything = _curve((0.0,) * 7)
        self.assertEqual(horizonfill.sufficing_width_seconds(everything), 240.0)
        none = _curve((0.3, 0.2, math.nan, 0.15, 0.2, 0.2, 0.3))
        self.assertEqual(horizonfill.sufficing_set(none), ())
        self.assertTrue(math.isnan(horizonfill.sufficing_width_seconds(none)))
        self.assertEqual(horizonfill.sufficing_set(_curve((math.nan, 0.05) + (0.5,) * 5)),
                         (90.0,))


# --- the runner on synthetic rows -------------------------------------------------------------------

CELL_A = ("conversation_trace", 0.0025, 4.0)
CELL_B = ("conversation_trace", 0.01, 1.0)
CELL_C = ("toolagent_trace", 0.0025, 4.0)
CELL_D = ("toolagent_trace", 0.01, 1.0)
SEEDS = (0, 1, 2, 3, 4)
LABEL_TOKENS, LRU_TOKENS = 600, 100
# S_h on the five-seed means at h = 60, 90, 120, 150, 180, 240, 300, 600 (U in
# tokens over 1000 requested tokens; U(arm) = 600 - 500 S). A: suffices at
# 150 and 180 (tie), V-shaped, sufficing {150, 180}; B: stands at 150 and 180,
# W-shaped, same minimisers as A; C: suffices at 90, increasing, minimiser 60,
# sufficing {60 .. 150}; D: stands at 240, decreasing to a tie at 240 and 300.
S_TABLE = {
    CELL_A: (0.40, 0.30, 0.20, 0.08, 0.08, 0.25, 0.30, 0.46),
    CELL_B: (0.36, 0.30, 0.28, 0.20, 0.20, 0.24, 0.22, 0.38),
    CELL_C: (0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, 0.45),
    CELL_D: (0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.25, 0.37),
}
NEW_U = {cell: {arm: round(LABEL_TOKENS - 500 * s) for arm, s in zip(ARMS, values)}
         for cell, values in S_TABLE.items()}
# Seed offsets with zero mean: B's label_binary_90 mixes the seed signs of
# U(label) - U(label_binary_90); C's lru moves the per-seed denominators, so
# the mean of the seed S is not the S of the means.
OFFSETS = {(CELL_B, "label_binary_90"): (250, -50, -50, -50, -100)}
LRU_OFFSETS = {CELL_C: (0, 40, -40, 80, -80)}
IDENTIFIERS = {"l1_capacity_bytes": 1, "l2_capacity_bytes": 4, "requested_tokens": 1000,
               "l1_avoided_tokens": 50, "absent_compulsory_tokens": 77}


def _published_tokens(cell, arm, seed):
    if arm == "label":
        return LABEL_TOKENS + seed
    if arm == "lru":
        return LRU_TOKENS + seed + LRU_OFFSETS.get(cell, (0,) * 5)[seed]
    return NEW_U[cell]["label_binary_600"] + seed


def synthetic_published():
    published = {}
    for cell in S_TABLE:
        for arm in ("label", "lru", "label_binary"):
            for seed in SEEDS:
                tokens = _published_tokens(cell, arm, seed)
                published[("all16", arm) + cell + (seed,)] = {
                    "source": "error_location_001", "mechanism": "all16", "arm": arm,
                    "trace": cell[0], "l1_fraction": cell[1], "l2_multiplier": cell[2],
                    "cell": runner.rdp.cell_label(*cell[1:]), "seed": seed,
                    "avoided_prefill_tokens": tokens + 50, "extra_avoided_tokens": tokens,
                    **IDENTIFIERS}
    return published


def synthetic_rows():
    rows = []
    for cell, arms in NEW_U.items():
        for seed in SEEDS:
            for arm, tokens in arms.items():
                tokens += seed + OFFSETS.get((cell, arm), (0,) * 5)[seed]
                row = {metric: 0.0 for metric in runner.REPLAY_METRICS}
                row.update(trace=cell[0], l1_fraction=cell[1], l2_multiplier=cell[2],
                           cell=runner.rdp.cell_label(*cell[1:]), eligibility="all", width=16,
                           mechanism="all16", arm=arm, family="horizon",
                           arm_parameter=HORIZON_OF_ARM[arm], seed=seed, variant="main",
                           extra_avoided_tokens=tokens, avoided_prefill_tokens=tokens + 50,
                           extra_points=tokens / 10.0,
                           counters_sha256=f"c/{arm}/{cell}/{seed}",
                           decision_sha256=f"d/{arm}/{cell}/{seed}", **IDENTIFIERS)
                rows.append(row)
    return rows


class RunnerDerivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.published = synthetic_published()

    def test_the_fill_tables(self):
        seeds, per_h, readings = runner.fill_tables(self.rows, self.published)
        self.assertEqual(len(seeds), 4 * 8 * 5)
        self.assertEqual(len(per_h), 4 * 8)
        self.assertEqual(len(readings), 4)
        by = {runner._cell_key(entry): entry for entry in readings}
        suffices, stands = FILL_READINGS
        expected = {
            CELL_A: (suffices, 0.08, "150;180", True, "150;180", "150;180", 2, 30.0),
            CELL_B: (stands, 0.20, "150;180", False, "150;180", "", 0, math.nan),
            CELL_C: (suffices, 0.05, "90", True, "60", "60;90;120;150", 4, 90.0),
            CELL_D: (stands, 0.25, "240", True, "240;300", "", 0, math.nan),
        }
        for cell, (reading, smallest, at, shape, minimisers, members, count, width) \
                in expected.items():
            with self.subTest(cell=cell):
                entry = by[cell]
                for h, value in zip(HORIZONS_SECONDS, S_TABLE[cell]):
                    self.assertAlmostEqual(entry[f"S_{h:g}"], value, places=9)
                self.assertEqual(entry["reading"], reading)
                self.assertAlmostEqual(entry["S_fill_min"], smallest, places=9)
                self.assertEqual(entry["S_fill_min_horizons"], at)
                self.assertIs(entry["unimodal_60_300"], shape)
                self.assertEqual(entry["curve_minimiser_horizons"], minimisers)
                self.assertEqual(entry["sufficing_horizons"], members)
                self.assertEqual(entry["sufficing_count"], count)
                self.assertTrue(_same(entry["sufficing_width_seconds"], width))
                self.assertAlmostEqual(entry["U_label_points_mean"], 60.2, places=9)
                self.assertAlmostEqual(entry["U_lru_points_mean"], 10.2, places=9)
                self.assertEqual(entry["seeds"], 5)
        self.assertEqual(list(by[CELL_A])[:13],
                         ["trace", "l1_fraction", "l2_multiplier", "cell", "seeds"]
                         + [f"S_{h:g}" for h in HORIZONS_SECONDS])

    def test_s_is_on_the_five_seed_means_with_seed_signs_and_roles(self):
        seeds, per_h, _ = runner.fill_tables(self.rows, self.published)
        per = {(runner._cell_key(entry), entry["h"]): entry for entry in per_h}
        for entry in per_h:
            self.assertEqual(entry["role"], horizonfill.ROLE_OF_HORIZON[entry["h"]])
            self.assertEqual(entry["arm"], horizon_arm(entry["h"]))
        for row in seeds:
            self.assertEqual(row["role"], "fill" if row["h"] in FILL_HORIZONS_SECONDS else "check")
        # C's per-seed denominators differ: S is the S of the means, not the
        # mean of the seed values.
        entry = per[(CELL_C, 120.0)]
        seed_values = [row["S_seed"] for row in seeds
                       if runner._cell_key(row) == CELL_C and row["h"] == 120.0]
        self.assertEqual(len(seed_values), 5)
        self.assertAlmostEqual(entry["S"], horizonctl.shortfall(
            entry["U_label_points_mean"], entry["U_arm_points_mean"], entry["U_lru_points_mean"]),
            places=12)
        self.assertAlmostEqual(entry["S"], 0.07, places=9)
        self.assertGreater(abs(sum(seed_values) / 5 - entry["S"]), 1e-4)
        # Seed signs of U(label) - U(label_binary_h).
        self.assertEqual(per[(CELL_B, 90.0)]["label_minus_arm_seed_signs"], "-++++")
        self.assertEqual(per[(CELL_B, 90.0)]["label_minus_arm_reading"], "mixed")
        self.assertEqual((per[(CELL_B, 90.0)]["label_minus_arm_n_pos"],
                          per[(CELL_B, 90.0)]["label_minus_arm_n_neg"]), (4, 1))
        self.assertEqual(per[(CELL_A, 60.0)]["label_minus_arm_reading"], "consistent_gain")
        self.assertAlmostEqual(per[(CELL_A, 60.0)]["label_minus_arm_points_mean"], 20.0, places=9)
        first = next(row for row in seeds if runner._cell_key(row) == CELL_B
                     and row["h"] == 90.0 and row["seed"] == 0)
        self.assertEqual(first["label_minus_arm_tokens"], 600 - (450 + 250))
        self.assertAlmostEqual(first["label_minus_arm_points"], -10.0, places=9)
        self.assertEqual(first["arm"], "label_binary_90")

    def test_the_summary_against_the_prediction(self):
        _, _, readings = runner.fill_tables(self.rows, self.published)
        summary = runner.fill_summary(readings)
        self.assertEqual([entry["scope"] for entry in summary],
                         ["all", "conversation_trace", "toolagent_trace"])
        total, conversation, toolagent = summary
        self.assertEqual((total[FILL_READINGS[0]], total[FILL_READINGS[1]], total["cells"]),
                         (2, 2, 4))
        self.assertEqual(total["predicted_" + FILL_READINGS[0]], 4)
        self.assertIs(total["prediction_holds"], False)
        self.assertEqual(total["unimodal_60_300"], 3)
        self.assertEqual(conversation["cell_labels"], "l1=0.0025,l2x4|l1=0.01,l2x1")
        self.assertEqual(conversation["curve_minimiser_horizons"], "150;180|150;180")
        self.assertIs(conversation["same_curve_minimiser"], True)
        self.assertEqual(toolagent["curve_minimiser_horizons"], "60|240;300")
        self.assertIs(toolagent["same_curve_minimiser"], False)
        self.assertEqual((conversation[FILL_READINGS[0]], conversation["predicted_"
                                                                       + FILL_READINGS[0]]), (1, 2))
        # Every trace x cell sufficing is the predicted 4/4.
        held = runner.fill_summary([dict(entry, reading=FILL_READINGS[0]) for entry in readings])
        self.assertTrue(all(entry["prediction_holds"] for entry in held))

    def test_aggregation_and_reference_rows(self):
        summary = runner.aggregate_replays(self.rows)
        self.assertEqual(len(summary), 4 * 8)
        self.assertEqual([entry["arm"] for entry in summary[:8]], list(ARMS))
        entry = next(e for e in summary if runner._cell_key(e) == CELL_A
                     and e["arm"] == "label_binary_150")
        self.assertEqual((entry["mechanism"], entry["family"], entry["arm_parameter"],
                          entry["seeds"]), ("all16", "horizon", 150.0, 5))
        self.assertAlmostEqual(entry["extra_points_mean"], 56.2)
        references = runner.reference_rows(self.published, self.rows)
        self.assertEqual(len(references), 4 * 5 * 3)
        self.assertEqual({entry["arm"] for entry in references}, {"label", "lru", "label_binary"})
        self.assertAlmostEqual(references[0]["extra_points"],
                               references[0]["extra_avoided_tokens"] / 10.0)

    def test_the_figure_is_written(self):
        seeds, _, readings = runner.fill_tables(self.rows, self.published)
        with tempfile.TemporaryDirectory() as directory:
            runner.fill_figure(Path(directory), readings, seeds)
            self.assertGreater((Path(directory) / "fill.png").stat().st_size, 10_000)

    def test_tabulation_reads_and_prints_without_writing(self):
        _, per_h, readings = runner.fill_tables(self.rows, self.published)
        tables = {"fill": per_h, "fill_reading": readings,
                  "fill_summary": runner.fill_summary(readings)}
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
        for section in ("F1", "F2", "F3"):
            self.assertIn(f"### {section} ", text)
        self.assertIn("S_90*", text)
        self.assertEqual(tabulate.fill_counts(parsed["fill_summary"])[0],
                         {"scope": "all", FILL_READINGS[0]: 2, FILL_READINGS[1]: 2,
                          "predicted": 4, "holds": False, "cells": 4})
        self.assertEqual([(row["trace"], row["minimisers"], row["same"])
                          for row in tabulate.shape_by_trace(parsed["fill_summary"])],
                         [("conversation_trace", "150;180 / 150;180", True),
                          ("toolagent_trace", "60 / 240;300", False)])
        # No published "|" reaches a Markdown table cell.
        for line in text.splitlines():
            if line.startswith("| conversation_trace") or line.startswith("| toolagent_trace"):
                self.assertNotIn("x4|", line)
        reading_rows = tabulate.fill_reading_table(parsed["fill_reading"])
        self.assertEqual([row["S_fill_min_horizons"] for row in reading_rows],
                         ["150;180", "150;180", "90", "240"])
        widths = [row["width"] for row in tabulate.sufficing_table(parsed["fill_reading"])]
        self.assertEqual(widths[0], 30.0)
        self.assertTrue(math.isnan(widths[1]))
        self.assertEqual([row["unimodal"] for row in tabulate.shape_table(parsed["fill_reading"])],
                         [True, False, True, True])
        signs = tabulate.fill_signs_table(parsed["fill"])
        self.assertEqual(len(signs), 32)
        b90 = next(row for row in signs if (row["trace"], row["l1_fraction"], row["h"])
                   == ("conversation_trace", 0.01, 90.0))
        self.assertEqual((b90["signs"], b90["reading"], b90["role"]), ((4, 0, 1), "mixed", "fill"))

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "fill_seeds", "fill",
                     "fill_reading", "fill_summary"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        for name in ("`fill.png`", "`run_config.json`", "8/12", "never merged",
                     "none is a policy", "sampling-seed variability only"):
            self.assertIn(name, runner.README_TEXT)


# --- the runner's checks and refusals ---------------------------------------------------------------


def _stats_row(arm="label_binary_120", **changes):
    row = {"trace": "conversation_trace", "l1_fraction": 0.01, "l2_multiplier": 1.0,
           "cell": "l1=0.01,l2x1", "seed": 0, "arm": arm, "variant": "main",
           "l2_decisions": 10, "l2_rejections": 3, "l2_evictions": 7,
           "stat_decisions_seen": 10, "stat_rejections_seen": 3, "stat_evictions_seen": 7,
           "overridden_decisions_seen": 0, "stat_decisions": 6}
    row.update(changes)
    return row


def _as_csv_rows(rows):
    """Rows as a table written by the runner and read back: every value a string."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.csv"
        runner._write(path, rows)
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class RunnerCheckTests(unittest.TestCase):
    def test_the_grid_and_the_task_order(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS)
        self.assertEqual(len(tasks), 160)
        self.assertEqual(len(set(tasks)), 160)
        self.assertEqual({task[3] for task in tasks}, set(ARMS))
        self.assertEqual({task[1:3] for task in tasks}, set(FILL_CELLS))
        self.assertEqual({task[0] for task in tasks}, set(runner.TRACES))
        self.assertEqual({task[4] for task in tasks}, set(SEEDS))
        self.assertEqual([task[1] for task in tasks], sorted((task[1] for task in tasks),
                                                             reverse=True))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))

    def test_reproduction_of_the_published_label_binary(self):
        published = synthetic_published()
        rows = [row for row in synthetic_rows() if row["arm"] == "label_binary_600"]
        report = runner.check_reproduction(rows, published, {"label_binary_600": 20})
        report = report["label_binary_600"]
        self.assertEqual((report["matched"], report["mismatched"], report["missing"],
                          report["reference"]), (20, 0, 0, "label_binary/all16"))
        self.assertTrue(all(row["reproduces_reference"] for row in rows))
        rows[0]["avoided_prefill_tokens"] += 1
        del published[("all16", "label_binary") + CELL_D + (4,)]
        report = runner.check_reproduction(rows, published, {"label_binary_600": 20})
        report = report["label_binary_600"]
        self.assertEqual((report["matched"], report["mismatched"], report["missing"]), (18, 1, 1))

    def test_identifiers_against_every_published_reference(self):
        published = synthetic_published()
        rows = synthetic_rows()
        compared, problems = runner.check_identifiers(rows, published)
        self.assertEqual(problems, [])
        self.assertEqual(compared, len(rows) * 3)
        rows[0]["absent_compulsory_tokens"] = 78
        rows[1]["l2_capacity_bytes"] = 5
        _, problems = runner.check_identifiers(rows, published)
        self.assertEqual(len(problems), 2 * 3)
        self.assertTrue(any("absent_compulsory_tokens 78 != 77" in line for line in problems))
        del published[("all16", "lru") + CELL_A + (0,)]
        _, problems = runner.check_identifiers(rows, published)
        self.assertTrue(any("no published lru/all16 reference row" in line for line in problems))

    def test_statistics(self):
        self.assertEqual(runner.check_statistics([_stats_row(arm) for arm in ARMS]), [])
        cases = (
            (_stats_row(stat_decisions_seen=9), "stat_decisions_seen 9 != l2_decisions 10"),
            (_stats_row(stat_rejections_seen=4), "stat_rejections_seen 4 != l2_rejections 3"),
            (_stats_row(stat_evictions_seen=6), "stat_evictions_seen 6 != l2_evictions 7"),
            (_stats_row(stat_decisions=0), "no decision inside"),
            (_stats_row("label_binary_600", overridden_decisions_seen=1),
             "overridden decisions for an arm without an override"),
        )
        for row, text in cases:
            with self.subTest(case=text):
                problems = runner.check_statistics([row])
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(text, problems[0])

    def test_the_first_run_comparison(self):
        rows = synthetic_rows()
        checks = [row for row in rows if row["arm"] in CHECK_ARMS]
        self.assertEqual(len(checks), 60)
        # The first run's table, as written and read back, with rows that are
        # not compared: other arms, and a variant other than main.
        extra = [dict(row, arm="label_binary_6") for row in checks[:3]]
        extra += [dict(checks[0], variant="nostats", counters_sha256="other")]
        first = _as_csv_rows(checks + extra)
        report = runner.check_first_run(rows, first, 60)
        self.assertEqual((report["matched"], report["missing"], report["mismatched"],
                          report["duplicates"], report["replays"],
                          report["decision_digests_compared"]), (60, 0, 0, 0, 60, 60))
        self.assertTrue(runner.first_run_passes(report))
        self.assertEqual(runner.first_run_coverage(first, runner.TRACES, FILL_CELLS, SEEDS), [])

        def changed(position, **values):
            table = [dict(row) for row in first]
            table[position].update(values)
            return table

        for column, value in (("avoided_prefill_tokens", "1"), ("counters_sha256", "x"),
                              ("decision_sha256", "x")):
            with self.subTest(column=column):
                report = runner.check_first_run(rows, changed(7, **{column: value}), 60)
                self.assertEqual((report["matched"], report["mismatched"]), (59, 1))
                self.assertIn(column, report["mismatches"][0])
                self.assertFalse(runner.first_run_passes(report))
        # A decision digest is compared only where both runs carry one.
        without = [{key: value for key, value in row.items() if key != "decision_sha256"}
                   for row in changed(7, decision_sha256="x")]
        report = runner.check_first_run(rows, without, 60)
        self.assertEqual((report["matched"], report["decision_digests_compared"]), (60, 0))
        self.assertTrue(runner.first_run_passes(report))
        blank = [dict(row, decision_sha256="") for row in rows]
        self.assertEqual(runner.check_first_run(blank, first, 60)["decision_digests_compared"], 0)
        # A missing row and a duplicated row each stop publication.
        missing = [row for row in first if not (row["arm"] == "label_binary_300"
                                                and row["trace"] == "toolagent_trace"
                                                and row["seed"] == "3"
                                                and row["l1_fraction"] == "0.01")]
        report = runner.check_first_run(rows, missing, 60)
        self.assertEqual((report["matched"], report["missing"]), (59, 1))
        self.assertFalse(runner.first_run_passes(report))
        problems = runner.first_run_coverage(missing, runner.TRACES, FILL_CELLS, SEEDS)
        self.assertEqual(problems, ["first-run row ('label_binary_300', 'toolagent_trace', 0.01, "
                                    "1.0, 3) missing"])
        duplicated = first + [dict(first[0])]
        report = runner.check_first_run(rows, duplicated, 60)
        self.assertEqual((report["matched"], report["duplicates"]), (60, 1))
        self.assertFalse(runner.first_run_passes(report))
        self.assertTrue(any(line.startswith("duplicate") for line in
                            runner.first_run_coverage(duplicated, runner.TRACES, FILL_CELLS,
                                                      SEEDS)))
        # Fewer replays than expected is a failure too.
        report = runner.check_first_run(rows[:-8], first, 60)
        self.assertFalse(runner.first_run_passes(report))

    def test_the_published_references_load_from_a_constructed_table(self):
        published = synthetic_published()
        table = [dict(entry, eligibility="all", width=16, variant="main",
                      **{"family": "x", "arm_parameter": ""}) for entry in published.values()]
        # Rows the loader must skip: another cell, another arm, another variant.
        table += [dict(table[0], l1_fraction=0.02), dict(table[0], arm="learned"),
                  dict(table[0], variant="nostats", avoided_prefill_tokens=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_seeds.csv"
            runner._write(path, table)
            loaded, problems = runner.load_references(path)
            self.assertEqual(problems, [])
            self.assertEqual(len(loaded), 60)
            key = ("all16", "label_binary") + CELL_C + (2,)
            self.assertEqual(loaded[key]["avoided_prefill_tokens"],
                             published[key]["avoided_prefill_tokens"])
            self.assertEqual(set(runner.REFERENCE_IDENTIFIERS) - set(loaded[key]), set())
            broken = table[1:] + [dict(table[2]), dict(table[3], eligibility="leaf")]
            runner._write(path, broken)
            _, problems = runner.load_references(path)
        self.assertTrue(any(line.startswith("duplicate") for line in problems))
        self.assertTrue(any("has leaf/16" in line for line in problems))
        self.assertTrue(any("missing" in line for line in problems))

    def test_argument_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "new"
            existing = Path(directory)
            empty = Path(directory) / "empty"
            empty.mkdir()
            first = Path(directory) / "first"
            first.mkdir()
            (first / "replay_seeds.csv").write_text("trace\n", encoding="utf-8")
            two = ["c.jsonl", "t.jsonl"]

            def validate(argv):
                runner.validate_arguments(runner.parse_args(argv))

            cases = {
                "positive": two + ["--output-dir", str(fresh), "--workers", "0"],
                "hard cap of 12": two + ["--output-dir", str(fresh), "--workers", "13"],
                "expected 2 trace files": ["c.jsonl", "--output-dir", str(fresh)],
                "output directory": two + ["--output-dir", str(existing)],
                "paper directory": two + ["--output-dir", str(fresh), "--paper-dir", str(existing)],
                "has no replay_seeds.csv": two + ["--output-dir", str(fresh), "--first-run-dir",
                                                  str(empty)],
            }
            for text, argv in cases.items():
                with self.subTest(case=text), self.assertRaises(SystemExit) as caught:
                    validate(argv)
                self.assertIn(text, str(caught.exception))
            validate(two + ["--output-dir", str(fresh), "--paper-dir", str(fresh / "p"),
                            "--workers", "12", "--first-run-dir", str(first)])
            self.assertFalse(fresh.exists())
            self.assertEqual(runner.parse_args(two + ["--output-dir", "x"]).workers, 10)
            self.assertIsNone(runner.parse_args(two + ["--output-dir", "x"]).first_run_dir)
            # No smoke, dirty-tree or seed option exists.
            for option in (["--smoke"], ["--allow-dirty"], ["--seeds", "4"]):
                with self.subTest(option=option), self.assertRaises(SystemExit), \
                        contextlib.redirect_stderr(io.StringIO()):
                    runner.parse_args(two + ["--output-dir", "x"] + option)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                runner.parse_args(two)

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["run_horizon_fill.py", "c.jsonl", "t.jsonl", "--output-dir", str(output)]
            dirty = lambda *a: " M src/x.py" if a[0] == "status" else "0" * 40
            clean = lambda *a: "" if a[0] == "status" else "0" * 40
            for git, differing, text in ((dirty, [], "not clean"),
                                         (clean, ["src/x.py"], "differs from HEAD")):
                with self.subTest(case=text), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(runner.rmc, "_git", git), \
                        mock.patch.object(runner, "sources_differing_from_head",
                                          lambda: differing), \
                        self.assertRaises(SystemExit) as caught:
                    runner.main()
                self.assertIn(text, str(caught.exception))
                self.assertFalse(output.exists())

    def test_the_execution_sources_cover_the_imported_runners(self):
        sources = {str(path.relative_to(REPOSITORY)) for path in runner.execution_sources()}
        for name in ("scripts/run_horizon_fill.py", "scripts/run_horizon_control.py",
                     "scripts/run_error_location.py", "scripts/run_mechanism_control.py",
                     "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/horizonfill.py",
                     "src/persistent_kv_admission/horizonctl.py",
                     "src/persistent_kv_admission/errorloc.py"):
            self.assertIn(name, sources)
        self.assertNotIn("scripts/tabulate_horizon_fill.py", sources)


# --- the worker on a constructed trace ----------------------------------------------------------------


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker` on a constructed trace, with the module state the
    runner's main would set, against the horizon control's own worker."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        name = self.trace.name
        shared = dict(traces={name: self.trace}, groups={name: _occurrence_groups(self.trace)},
                      splits={name: MEASURE_FROM_MS}, horizons={name: HORIZON})
        patches = (mock.patch.dict(runner.SHARED, shared),
                   mock.patch.dict(runner.hc.SHARED, dict(shared, rankers={name: None})),
                   mock.patch.dict(runner.rdp._SHARED, {"working_set": {name: 400 * 2**20}}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.name = name

    def test_at_the_check_horizons_the_worker_is_the_horizon_controls(self):
        for arm in CHECK_ARMS:
            for fraction, multiplier in FILL_CELLS:
                with self.subTest(arm=arm, cell=(fraction, multiplier)):
                    mine = runner._replay_worker((self.name, fraction, multiplier, arm, 1))
                    theirs = runner.hc._replay_worker((self.name, fraction, multiplier, arm, 1,
                                                       "main"))
                    self.assertEqual(list(mine), list(theirs))
                    differing = [key for key in mine if key not in VOLATILE
                                 and not _same(mine[key], theirs[key])]
                    self.assertEqual(differing, [])
                    self.assertEqual(mine["decision_sha256"], theirs["decision_sha256"])
                    self.assertGreater(mine["l2_decisions"], 0)

    def test_worker_rows_pass_the_checks(self):
        rows = [runner._replay_worker((self.name, fraction, multiplier, arm, 0))
                for fraction, multiplier in FILL_CELLS for arm in ARMS]
        self.assertEqual(runner.check_statistics(rows), [])
        groups, problems = runner.rmc.check_invariants(rows, len(ARMS))
        self.assertEqual((groups, problems), (2, []))
        for row in rows:
            self.assertEqual((row["mechanism"], row["eligibility"], row["width"], row["family"],
                              row["variant"]), ("all16", "all", 16, "horizon", "main"))
            self.assertEqual(row["arm_parameter"], HORIZON_OF_ARM[row["arm"]])
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertGreater(row["stat_decisions"], 0)
            self.assertEqual(row["overridden_decisions_seen"], 0)
        for fraction, _ in FILL_CELLS:
            digests = {row["decision_sha256"] for row in rows if row["l1_fraction"] == fraction}
            self.assertEqual(len(digests), len(ARMS))
        # A publication of these very rows passes the identifier check and
        # holds label_binary_600 to it.
        published = {}
        for row in rows:
            for mechanism, arm in runner.REFERENCES:
                published[(mechanism, arm) + runner._cell_seed(row)] = {
                    "source": "s", "avoided_prefill_tokens": row["avoided_prefill_tokens"],
                    **{field: row[field] for field in runner.REFERENCE_IDENTIFIERS}}
        self.assertEqual(runner.check_identifiers(rows, published)[1], [])
        report = runner.check_reproduction(rows, published, {"label_binary_600": 2})
        self.assertEqual(report["label_binary_600"]["matched"], 2)


# --- the whole run on constructed traces ---------------------------------------------------------------


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


class MainTests(unittest.TestCase):
    """`main` end to end on two constructed traces named as the grid's, with a
    constructed publication and a constructed first run, git answered as for
    a clean tree, and the ranker loaders replaced by failures: the run must
    not need them."""

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
        # The check arms replayed as main will replay them (same split,
        # horizon and working set), for the publication and the first run.
        shared = {"traces": traces, "groups": {}, "splits": {}, "horizons": {}}
        working_set = {}
        for name, trace in traces.items():
            horizon, split_ms, _ = horizon_for(trace)
            assert horizon == 600.0, (name, horizon)
            shared["groups"][name] = _occurrence_groups(trace)
            shared["splits"][name] = split_ms
            shared["horizons"][name] = horizon
            working_set[name] = working_set_bytes(trace)
        with mock.patch.dict(runner.SHARED, shared), \
                mock.patch.dict(runner.rdp._SHARED, {"working_set": working_set}):
            checks = [runner._replay_worker((name, fraction, multiplier, arm, seed))
                      for name in sorted(traces) for fraction, multiplier in FILL_CELLS
                      for arm in CHECK_ARMS for seed in SEEDS]
        reference = []
        for row in checks:
            if row["arm"] != "label_binary_600":
                continue
            extra = row["extra_avoided_tokens"]
            for arm, tokens in (("label_binary", extra), ("label", extra + 500),
                                ("lru", extra // 2)):
                reference.append(dict(row, arm=arm, family="reference", arm_parameter="",
                                      extra_avoided_tokens=tokens,
                                      avoided_prefill_tokens=row["l1_avoided_tokens"] + tokens))
        cls.reference = root / "published" / "replay_seeds.csv"
        cls.reference.parent.mkdir()
        runner._write(cls.reference, reference)
        cls.first = root / "first"
        cls.first.mkdir()
        runner._write(cls.first / "replay_seeds.csv", checks)
        cls.checks = checks

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _main(self, argv, cells=None, seeds=None) -> str:
        refuse = AssertionError("no ranker may be loaded")
        patches = [mock.patch.object(sys, "argv", ["run_horizon_fill.py"] + self.trace_paths
                                     + argv),
                   mock.patch.object(runner.rmc, "_git", _fake_git),
                   mock.patch.object(runner, "sources_differing_from_head", lambda: []),
                   mock.patch.object(runner, "ERROR_LOCATION_REFERENCE", self.reference),
                   mock.patch.object(runner.rmc, "load_models", side_effect=refuse),
                   mock.patch.object(runner.rmc, "deserialize_ranker", side_effect=refuse),
                   mock.patch.object(runner.rel, "deserialize_ranker", side_effect=refuse),
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
                           "--workers", "2", "--first-run-dir", str(self.first)])
        tables = {"replay_seeds.csv", "replay.csv", "references_seeds.csv", "fill_seeds.csv",
                  "fill.csv", "fill_reading.csv", "fill_summary.csv", "fill.png", "README.md",
                  "run_config.json"}
        self.assertEqual(set(os.listdir(paper)), tables)
        self.assertEqual(set(os.listdir(output)), tables | {"raw_replays.jsonl"})
        self.assertEqual(len((output / "raw_replays.jsonl").read_text().splitlines()), 160)
        self.assertIn("[20/160]", text)
        self.assertIn("[160/160]", text)
        self.assertIn("reading 1, fill", text)
        replays = horizon_tests.csv_rows(paper / "replay_seeds.csv")
        replays = list(replays)
        self.assertEqual(len(replays), 160)
        self.assertEqual(list(replays[0])[:12],
                         ["trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width",
                          "mechanism", "arm", "family", "arm_parameter", "seed", "variant"])
        # The check-arm rows are the constructed first run's, digest for digest.
        index = {(row["arm"],) + runner._cell_seed(row): row for row in replays}
        for row in self.checks:
            mine = index[(row["arm"],) + runner._cell_seed(row)]
            self.assertEqual((mine["counters_sha256"], mine["decision_sha256"]),
                             (row["counters_sha256"], row["decision_sha256"]))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["phase"], "horizon_fill")
        self.assertEqual(config["models"], {})
        self.assertEqual((config["plan_commit"], config["code_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(config["plan"], "docs/horizon-fill-plan.md")
        self.assertEqual(config["first_run"]["checked"], True)
        self.assertEqual(config["first_run"]["sha256"], sha256_path(self.first / "replay_seeds.csv"))
        self.assertEqual(config["references"], {str(self.reference.resolve()):
                                                sha256_path(self.reference)})
        checks = config["checks"]
        self.assertEqual(checks["reproduction"]["label_binary_600"]["matched"], 20)
        self.assertEqual((checks["first_run"]["matched"], checks["first_run"]["expected"],
                          checks["first_run"]["decision_digests_compared"]), (60, 60, 60))
        self.assertEqual((checks["identifier_problems"], checks["statistics_problems"],
                          checks["invariant_violations"], checks["unexplained_absent_tokens"]),
                         (0, 0, 0, 0))
        self.assertEqual(checks["identifier_pairs_compared"], 160 * 3)
        self.assertEqual(config["replays"], 160)
        self.assertEqual(set(config["arms"]), set(ARMS))
        self.assertEqual(set(config["trace_files"]), {"conversation_trace", "toolagent_trace"})
        self.assertIn("never merged", config["note"])
        readings = list(horizon_tests.csv_rows(paper / "fill_reading.csv"))
        self.assertEqual(len(readings), 4)
        self.assertTrue({row["reading"] for row in readings} <= set(FILL_READINGS))
        self.assertEqual(len(list(horizon_tests.csv_rows(paper / "fill.csv"))), 32)
        self.assertEqual(len(list(horizon_tests.csv_rows(paper / "fill_seeds.csv"))), 160)
        self.assertEqual(len(list(horizon_tests.csv_rows(paper / "fill_summary.csv"))), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tabulate.main([str(paper)]), 0)

    def test_a_first_run_difference_publishes_nothing(self):
        tampered = self.root / "first_tampered"
        tampered.mkdir()
        rows = [dict(row) for row in self.checks]
        target = next(row for row in rows if row["arm"] == "label_binary_300"
                      and row["l1_fraction"] == 0.01 and row["seed"] == 0)
        target["counters_sha256"] = "0" * 64
        runner._write(tampered / "replay_seeds.csv", rows)
        output, paper = self.root / "run_bad", self.root / "paper_bad"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper), "--workers", "2",
                        "--first-run-dir", str(tampered)], cells=((0.01, 1.0),), seeds=(0,))
        self.assertIn("differ from the first run", str(caught.exception))
        self.assertIn("nothing derived or published", str(caught.exception))
        self.assertIn("FIRST-RUN", self.stdout)
        self.assertIn("counters_sha256", self.stdout)
        self.assertFalse(paper.exists())
        self.assertEqual(os.listdir(output), ["raw_replays.jsonl"])

    def test_without_a_first_run_the_comparison_is_recorded_as_not_run(self):
        output = self.root / "run_alone"
        self._main(["--output-dir", str(output), "--workers", "2"], cells=((0.0025, 4.0),),
                   seeds=(0,))
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["first_run"]["checked"], False)
        self.assertEqual(config["checks"]["first_run"], "not run")
        self.assertEqual(config["checks"]["reproduction"]["label_binary_600"]["matched"], 2)
        self.assertEqual(config["replays"], 16)
        self.assertIn("not given", self.stdout)

    def test_a_first_run_without_every_row_is_refused_before_any_replay(self):
        partial = self.root / "first_partial"
        partial.mkdir()
        runner._write(partial / "replay_seeds.csv",
                      [row for row in self.checks if row["arm"] != "label_binary_60"])
        output = self.root / "run_refused"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--first-run-dir", str(partial)])
        self.assertIn("nothing run", str(caught.exception))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
