"""Tests for the matched-horizon diagnosis (docs/matched-horizon-diagnosis-plan.md).

No saved log is read: everything runs on constructed populations, built as
the ranker-error diagnosis's tests build them (its test module is loaded by
path and its builders used as they are). What has to hold before the logs are
read: the horizons and the matched map are the plan's; the decisions at a
horizon are the diagnosis's decisions with the reuse boundary moved, the
population untouched, and at 600 s every statistic is the diagnosis's own
number; the error kinds partition the decisions at every horizon, and a
candidate reusable at 600 s but not at 60 s moves between kinds and between
the mixed and all-reusable sets; the within-decision concordance is 1, 0 and
0.5 for a key that orders every non-reusable candidate first, last or not at
all, counts an exact tie one half, weighs every decision one, and equals a
literal pairwise loop; a uniform victim is 0.5 exactly; Spearman's rank
correlation on hand-computed examples with ties; the reading rules at their
boundaries. The runner is run end to end on a constructed published run whose
diagnosis tables are written by the diagnosis's own runner: every table
written with exactly its documented columns, one worker equal to two, no
`pi3` population read, the h = 600 rows equal to the published tables value
for value, the bridge and concordance against independent computations; and a
one-ulp difference in a published value, a published row removed, a published
column this run does not have, a missing, duplicated or inconsistent replay
row, a changed population, the stop rule and the refusals publish nothing
(or, for the stop rule, only the checks).
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import itertools
import json
import math
import random
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from persistent_kv_admission import errordiag, matcheddiag
from persistent_kv_admission.matcheddiag import HORIZONS_SECONDS, MATCHED_HORIZON_SECONDS

REPOSITORY = Path(__file__).resolve().parents[1]
# The name under which the diagnosis's own tests import its runner.
_PARENT_RUNNER_UNDER_TEST = "_ranker_error_diagnosis_runner_under_test"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixtures():
    """The diagnosis's test module, for its population and published-run
    builders. Loading it imports the diagnosis runner under the name the
    diagnosis's own tests use; that registration is put back as it was, so the
    two test modules never replace each other's runner (a fork pool pickles a
    worker function by its module's registered name)."""
    saved = sys.modules.get(_PARENT_RUNNER_UNDER_TEST)
    try:
        return _load("_matched_horizon_ranker_error_fixtures",
                     REPOSITORY / "tests/test_ranker_error_diagnosis.py")
    finally:
        if saved is None:
            sys.modules.pop(_PARENT_RUNNER_UNDER_TEST, None)
        else:
            sys.modules[_PARENT_RUNNER_UNDER_TEST] = saved


fixtures = _load_fixtures()
runner = _load("_run_matched_horizon_diagnosis_under_test",
               REPOSITORY / "scripts/run_matched_horizon_diagnosis.py")
tabulate = _load("_tabulate_matched_horizon_diagnosis_under_test",
                 REPOSITORY / "scripts/tabulate_matched_horizon_diagnosis.py")
red = runner.red
# The fixtures build their published run on, and run, the diagnosis runner
# this runner imported, so one patch of a grid constant or a path reaches both.
fixtures.runner = red

NEVER = fixtures.NEVER
build_population = fixtures.build_population
random_decisions = fixtures.random_decisions
random_ranker = fixtures.random_ranker
linear_ranker = fixtures.linear_ranker
# Next uses on both sides of every horizon's boundary, exactly at it, and never.
BOUNDARY_DELTAS = (NEVER, NEVER, 0.0, 1.0, 30.0, 59.0, 60.0, 61.0, 120.0, 150.0, 151.0, 299.0,
                   300.0, 301.0, 599.0, 600.0, 601.0)


def boundary_decisions(rng: random.Random, count: int, **kwargs) -> list[dict]:
    with mock.patch.object(fixtures, "DELTA_CHOICES", BOUNDARY_DELTAS):
        return random_decisions(rng, count, **kwargs)


def logged_at(population, h: float):
    decisions = matcheddiag.decisions_at(population, h)
    return decisions, errordiag.logged_victims(population.victim, decisions)


def literal_concordance(keys: list, reusable: list[bool]):
    """One decision's concordance by a literal loop over the (reusable,
    non-reusable) pairs, with Python tuple comparison and exact fractions;
    None when one class is absent."""
    kept = [index for index, flag in enumerate(reusable) if flag]
    other = [index for index, flag in enumerate(reusable) if not flag]
    if not kept or not other:
        return None
    total = Fraction(0)
    for r in kept:
        for n in other:
            if keys[n] < keys[r]:
                total += 1
            elif keys[n] == keys[r]:
                total += Fraction(1, 2)
    return float(total / (len(kept) * len(other)))


# --- the grid -------------------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_horizons_and_the_matched_map_are_the_plans(self):
        self.assertEqual(HORIZONS_SECONDS, (60.0, 150.0, 300.0, 600.0))
        self.assertEqual(matcheddiag.DIAGNOSIS_HORIZON_SECONDS, 600.0)
        self.assertEqual(MATCHED_HORIZON_SECONDS, {
            (0.0025, 1.0): 60.0, (0.0025, 4.0): 150.0, (0.01, 1.0): 150.0,
            (0.01, 4.0): 600.0, (0.02, 1.0): 300.0, (0.02, 4.0): 600.0})
        self.assertEqual(matcheddiag.COMPOSITION_SHARE, 0.99)
        self.assertEqual(matcheddiag.UNIFORM_CONCORDANCE, 0.5)
        self.assertEqual(matcheddiag.PREDICTION_TARGET, "next_use")

    def test_the_matched_map_holds_the_six_cells_of_the_grid(self):
        self.assertEqual(set(MATCHED_HORIZON_SECONDS), set(red.CELLS))
        self.assertEqual(len(red.CELLS), 6)
        below = [cell for cell in red.CELLS if matcheddiag.matched_horizon(*cell) < 600.0]
        self.assertEqual(len(below) * len(red.TRACES), 8)
        for (fraction, multiplier), h in MATCHED_HORIZON_SECONDS.items():
            self.assertTrue(matcheddiag.is_matched(fraction, multiplier, h))
            for other in set(HORIZONS_SECONDS) - {h}:
                self.assertFalse(matcheddiag.is_matched(fraction, multiplier, other))
        with self.assertRaises(KeyError):
            matcheddiag.matched_horizon(0.05, 1.0)

    def test_the_runner_reads_the_diagnosis_grid(self):
        self.assertEqual(red.TRACES, ("conversation_trace", "toolagent_trace"))
        self.assertEqual(red.TARGETS, ("next_use", "binary"))
        self.assertEqual(red.SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(runner.FIRST, 0)
        self.assertEqual(len(runner.first_keys()), 120)
        self.assertTrue(all(key[5] == 0 for key in runner.first_keys()))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))
        self.assertEqual(runner.LABEL_BINARY_ARMS, {
            60.0: ("horizon_control_001", "label_binary_60"),
            150.0: ("horizon_fill_001", "label_binary_150"),
            300.0: ("horizon_control_001", "label_binary_300"),
            600.0: ("horizon_control_001", "label_binary_600")})
        self.assertEqual(set(runner.REPRODUCED), {"classes", "rates", "rates_seeds"})

    def test_the_execution_sources_include_the_imported_runner(self):
        sources = runner.execution_sources()
        self.assertIn(runner.PARENT_SCRIPT.resolve(), sources)
        self.assertIn(REPOSITORY / "scripts/run_matched_horizon_diagnosis.py", sources)
        self.assertIn(REPOSITORY / "src/persistent_kv_admission/matcheddiag.py", sources)
        self.assertIn(REPOSITORY / "src/persistent_kv_admission/errordiag.py", sources)

    def test_inputs_outside_the_repository_differ_from_head(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner, "execution_sources", return_value=[]):
            differing = runner.sources_differing_from_head(Path(directory))
        for path in runner.reference_paths(Path(directory)).values():
            self.assertIn(str(path.resolve()), differing)


# --- the decisions at a horizon -------------------------------------------------------------------


class DecisionsAtTests(unittest.TestCase):
    def test_at_600_they_are_the_diagnosis_decisions(self):
        rng = random.Random(2)
        for _ in range(10):
            population = build_population(boundary_decisions(rng, 50))
            ours = matcheddiag.decisions_at(population, 600.0)
            theirs = errordiag.decisions_of(population)
            for field in errordiag.Decisions.__dataclass_fields__:
                self.assertTrue(np.array_equal(getattr(ours, field), getattr(theirs, field)), field)

    def test_the_population_is_not_changed(self):
        population = build_population(boundary_decisions(random.Random(4), 40))
        before = {name: np.array(value, copy=True) for name, value in vars(population).items()
                  if isinstance(value, np.ndarray)}
        for h in HORIZONS_SECONDS:
            matcheddiag.decisions_at(population, h)
        self.assertEqual(population.horizon_seconds, 600.0)
        for name, value in before.items():
            self.assertTrue(np.array_equal(getattr(population, name), value), name)

    def test_label_and_reuse_at_each_horizon(self):
        deltas = [59.999, 60.0, 60.001, 150.0, 300.0, 600.0, 600.001, NEVER]
        population = build_population([{"delta_s": deltas, "victim": 0}])
        expected_reusable = {60.0: 2, 150.0: 4, 300.0: 5, 600.0: 6}
        for h in HORIZONS_SECONDS:
            decisions = matcheddiag.decisions_at(population, h)
            self.assertEqual(decisions.reusable.tolist(),
                             [index < expected_reusable[h] for index in range(len(deltas))])
            expected = [-math.log1p(min(delta, h)) for delta in deltas]
            self.assertTrue(np.allclose(decisions.label, expected, rtol=0, atol=1e-15))
            self.assertEqual(float(decisions.min_label[0]), -math.log1p(h))


# --- the error kinds at every horizon -------------------------------------------------------------


class ClassTests(unittest.TestCase):
    def test_the_error_kinds_partition_the_decisions_at_every_horizon(self):
        rng = random.Random(5)
        for _ in range(20):
            population = build_population(boundary_decisions(rng, 80), scorer=random_ranker(rng))
            for h in HORIZONS_SECONDS:
                decisions, victims = logged_at(population, h)
                memberships = errordiag.class_memberships(decisions, victims)
                counts = memberships.sum(axis=0)
                self.assertTrue((counts >= 1).all())
                # The one overlap: a victim at exactly h with a non-reusable candidate.
                at_h = ((population.next_use_delta_ms[victims] == h * 1000.0)
                        & ~decisions.all_reusable)
                self.assertTrue((counts[~at_h] == 1).all())
                self.assertTrue((counts[at_h] == 2).all())
                classes = errordiag.classify(decisions, victims)
                value = errordiag.excess(decisions, victims)
                self.assertTrue((value[classes <= 1] == 0).all())
                self.assertTrue((value[classes >= 2] > 0).all())
                rows = errordiag.class_rows(decisions, victims)
                for subset in errordiag.SUBSETS:
                    members = [row for row in rows if row["subset"] == subset]
                    self.assertEqual(sum(row["decisions"] for row in members),
                                     members[0]["subset_decisions"])
                self.assertEqual(sum(row["decisions"] for row in rows if row["subset"] == "all"),
                                 len(decisions))

    def test_a_candidate_reusable_at_600_but_not_at_60_changes_class(self):
        """100 s and 200 s are reusable at 600 s and not at 60 s. (A later next
        use has the lower label: evicting it is no excess.)"""
        population = build_population([
            # 600 s: the victim is reusable, the other not. 60 s: both clipped,
            # the victim is the first minimum of (label, tie-break).
            {"delta_s": [100.0, NEVER], "tiebreak": [0, 1], "victim": 0},
            # 600 s: both reusable, the victim above the minimum. 60 s: the
            # victim reusable, the other not.
            {"delta_s": [10.0, 100.0], "tiebreak": [0, 1], "victim": 0},
            # 600 s: both reusable, the victim above the minimum. 60 s: both
            # clipped, the victim not the first minimum of (label, tie-break).
            {"delta_s": [100.0, 200.0], "tiebreak": [1, 0], "victim": 0},
        ])
        names = {}
        for h in (60.0, 600.0):
            decisions, victims = logged_at(population, h)
            names[h] = [errordiag.CLASSES[index]
                        for index in errordiag.classify(decisions, victims)]
            names[h, "mixed"] = decisions.mixed.tolist()
            names[h, "all_reusable"] = decisions.all_reusable.tolist()
            names[h, "m4"] = errordiag.statistics(decisions, victims)["avoidable_count"]
            names[h, "excess"] = (errordiag.excess(decisions, victims) > 0).tolist()
        self.assertEqual(names[600.0], ["avoidable_reusable", "order_error", "order_error"])
        self.assertEqual(names[60.0], ["reference_victim", "avoidable_reusable", "other_tiebreak"])
        self.assertEqual(names[600.0, "mixed"], [True, False, False])
        self.assertEqual(names[60.0, "mixed"], [False, True, False])
        self.assertEqual(names[600.0, "all_reusable"], [False, True, True])
        self.assertEqual(names[60.0, "all_reusable"], [False, False, False])
        self.assertEqual(names[600.0, "excess"], [True, True, True])
        self.assertEqual(names[60.0, "excess"], [False, True, False])
        self.assertEqual((names[600.0, "m4"], names[60.0, "m4"]), (1, 1))

    def test_the_first_decision_is_a_tie_at_60_and_the_victim_is_the_reference(self):
        """At 60 s both candidates of the first decision have the clipped label:
        the reference is the first minimum of (label, tie-break), the victim."""
        population = build_population([{"delta_s": [100.0, NEVER], "tiebreak": [1, 0],
                                        "victim": 0}])
        decisions, victims = logged_at(population, 60.0)
        self.assertEqual(decisions.reference.tolist(), [1])
        self.assertEqual(errordiag.CLASSES[int(errordiag.classify(decisions, victims)[0])],
                         "other_tiebreak")


# --- concordance ----------------------------------------------------------------------------------


def _single(deltas, h=600.0, **extra):
    population = build_population([{"delta_s": deltas, "victim": 0, **extra}])
    return population, matcheddiag.decisions_at(population, h)


class ConcordanceTests(unittest.TestCase):
    # Reusable at 600: rows 0 and 2; not: rows 1 and 3.
    DELTAS = [10.0, NEVER, 20.0, 700.0]

    def _value(self, key, deltas=None, h=600.0, tiebreak=None):
        _, decisions = _single(deltas or self.DELTAS, h)
        ranks = (matcheddiag.key_ranks(key) if tiebreak is None
                 else matcheddiag.key_ranks(key, tiebreak))
        return matcheddiag.concordance(ranks, decisions)["concordance"]

    def test_every_non_reusable_first_is_one_the_reverse_zero_a_constant_one_half(self):
        self.assertEqual(self._value([5.0, 1.0, 6.0, 2.0]), 1.0)
        self.assertEqual(self._value([1.0, 5.0, 2.0, 6.0]), 0.0)
        self.assertEqual(self._value([3.0, 3.0, 3.0, 3.0]), 0.5)

    def test_an_exact_tie_counts_one_half(self):
        # R = {0, 1}, N = {2}: (0, 2) the non-reusable is above; (1, 2) a tie.
        self.assertEqual(self._value([1.0, 2.0, 2.0], deltas=[10.0, 20.0, NEVER]), 0.25)
        # -0.0 and 0.0 are the same key.
        self.assertEqual(self._value([0.0, -0.0], deltas=[10.0, NEVER]), 0.5)

    def test_the_ranker_key_is_lexicographic(self):
        deltas = [10.0, NEVER]
        # Equal scores: the tie-break decides (the non-reusable's is lower: first).
        self.assertEqual(self._value([1.0, 1.0], deltas=deltas, tiebreak=[5.0, 4.0]), 1.0)
        self.assertEqual(self._value([1.0, 1.0], deltas=deltas, tiebreak=[4.0, 5.0]), 0.0)
        self.assertEqual(self._value([1.0, 1.0], deltas=deltas, tiebreak=[4.0, 4.0]), 0.5)
        # A lower score wins whatever the tie-break.
        self.assertEqual(self._value([1.0, 0.5], deltas=deltas, tiebreak=[0.0, 9.0]), 1.0)

    def test_the_reuse_bit_is_the_horizons(self):
        # 100 s: reusable at 600, not at 60. Key: the 100-s candidate lowest.
        deltas = [10.0, 100.0, NEVER]
        key = [3.0, 1.0, 2.0]
        # At 600 R = {0, 1}, N = {2}: (0, 2) first, (1, 2) not: 1/2.
        self.assertEqual(self._value(key, deltas=deltas, h=600.0), 0.5)
        # At 60 R = {0}, N = {1, 2}: both first: 1.
        self.assertEqual(self._value(key, deltas=deltas, h=60.0), 1.0)

    def test_each_decision_weighs_one_and_only_mixed_decisions_count(self):
        population = build_population([
            {"delta_s": [10.0, NEVER], "victim": 0},                        # 1 pair, concordant
            {"delta_s": [10.0, 20.0, NEVER, NEVER, NEVER], "victim": 0},     # 6 pairs, none
            {"delta_s": [10.0, 20.0], "victim": 0},                          # all reusable
            {"delta_s": [NEVER, NEVER], "victim": 0},                        # none reusable
        ])
        decisions = matcheddiag.decisions_at(population, 600.0)
        key = [2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 1.0, 2.0]
        values = matcheddiag.concordance_by_decision(matcheddiag.key_ranks(key), decisions)
        self.assertEqual(values[:2].tolist(), [1.0, 0.0])
        self.assertTrue(np.isnan(values[2:]).all())
        result = matcheddiag.concordance(matcheddiag.key_ranks(key), decisions)
        # The mean of 1 and 0, not the pooled 1 / 7.
        self.assertEqual(result, {"concordance_decisions": 2, "concordance": 0.5})
        none = build_population([{"delta_s": [10.0, 20.0], "victim": 0}])
        empty = matcheddiag.concordance(np.zeros(2, dtype=np.int64),
                                        matcheddiag.decisions_at(none, 600.0))
        self.assertEqual(empty["concordance_decisions"], 0)
        self.assertTrue(math.isnan(empty["concordance"]))

    def test_against_a_literal_pairwise_loop(self):
        rng = random.Random(9)
        for _ in range(15):
            ranker = random_ranker(rng)
            population = build_population(boundary_decisions(rng, 60, max_width=9), scorer=ranker)
            scores = errordiag.scorer_scores(ranker, population)
            base = matcheddiag.decisions_at(population, 600.0)
            keys = {"ranker": (matcheddiag.ranker_key_ranks(scores, base),
                               list(zip(scores.tolist(), base.tiebreak.tolist()))),
                    "recency": (matcheddiag.recency_key_ranks(base),
                                [(value,) for value in base.tiebreak.tolist()])}
            for h in HORIZONS_SECONDS:
                decisions = matcheddiag.decisions_at(population, h)
                for name, (ranks, tuples) in keys.items():
                    got = matcheddiag.concordance_by_decision(ranks, decisions)
                    expected = []
                    for start, width in zip(decisions.starts.tolist(), decisions.widths.tolist()):
                        block = slice(start, start + width)
                        expected.append(literal_concordance(
                            tuples[block], decisions.reusable[block].tolist()))
                    for value, want in zip(got.tolist(), expected):
                        if want is None:
                            self.assertTrue(math.isnan(value))
                        else:
                            self.assertEqual(value, want, (name, h))
                    kept = [value for value in expected if value is not None]
                    mean = matcheddiag.concordance(ranks, decisions)
                    self.assertEqual(mean["concordance_decisions"], len(kept))
                    if kept:
                        self.assertAlmostEqual(mean["concordance"], sum(kept) / len(kept),
                                               places=13)

    def test_blocks_do_not_change_the_values(self):
        rng = random.Random(10)
        population = build_population(boundary_decisions(rng, 200, max_width=7),
                                      scorer=random_ranker(rng))
        decisions = matcheddiag.decisions_at(population, 150.0)
        ranks = matcheddiag.recency_key_ranks(decisions)
        whole = matcheddiag.concordance_by_decision(ranks, decisions)
        with mock.patch.object(matcheddiag, "_CONCORDANCE_CHUNK", 3):
            blocked = matcheddiag.concordance_by_decision(ranks, decisions)
        self.assertTrue(np.array_equal(whole, blocked, equal_nan=True))

    def test_a_uniform_victim_is_one_half_exactly(self):
        """Over every ordering of distinct keys, each decision's mean
        concordance is one half; a constant key is one half; and the chooser
        table carries 0.5."""
        deltas = [10.0, NEVER, 20.0, 700.0, NEVER]
        _, decisions = _single(deltas)
        values = []
        for order in itertools.permutations(range(len(deltas))):
            ranks = np.asarray(order, dtype=np.int64)
            values.append(Fraction(float(matcheddiag.concordance_by_decision(ranks, decisions)[0])))
        # Each value is k/6 rounded once; their mean is one half up to that rounding.
        self.assertAlmostEqual(float(sum(values) / len(values)), 0.5, places=15)
        self.assertEqual(sorted(set(round(float(value) * 6) for value in values)),
                         [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(matcheddiag.concordance(np.zeros(len(deltas), dtype=np.int64),
                                                 decisions)["concordance"], 0.5)
        population = build_population(boundary_decisions(random.Random(1), 20))
        decisions, victims = logged_at(population, 60.0)
        ranks = matcheddiag.recency_key_ranks(decisions)
        entry = matcheddiag.horizon_statistics(decisions, victims, victims, victims, ranks, ranks)
        self.assertEqual(entry["concordance"]["uniform"], 0.5)
        self.assertEqual(entry["concordance"]["ranker"], entry["concordance"]["recency"])

    def test_key_ranks_order_like_tuples(self):
        rng = random.Random(12)
        primary = [rng.choice((-0.0, 0.0, 1.0, 2.5, -3.0)) for _ in range(300)]
        secondary = [float(rng.randint(0, 3)) for _ in range(300)]
        ranks = matcheddiag.key_ranks(primary, secondary).tolist()
        alone = matcheddiag.key_ranks(primary).tolist()
        for _ in range(3000):
            i, j = rng.randrange(300), rng.randrange(300)
            pair = (primary[i], secondary[i]), (primary[j], secondary[j])
            self.assertEqual(ranks[i] < ranks[j], pair[0] < pair[1])
            self.assertEqual(ranks[i] == ranks[j], pair[0] == pair[1])
            self.assertEqual(alone[i] < alone[j], primary[i] < primary[j])
            self.assertEqual(alone[i] == alone[j], primary[i] == primary[j])
        with self.assertRaises(ValueError):
            matcheddiag.key_ranks([1.0, math.nan])
        with self.assertRaises(ValueError):
            matcheddiag.key_ranks([1.0, 2.0], [1.0])


# --- the diagnosis's numbers at 600 s -------------------------------------------------------------


class StatisticsAt600Tests(unittest.TestCase):
    def test_the_module_statistics_are_the_diagnosis_functions_at_600(self):
        rng = random.Random(13)
        for _ in range(10):
            population = build_population(boundary_decisions(rng, 70), scorer=random_ranker(rng))
            decisions, victims = logged_at(population, 600.0)
            theirs = errordiag.decisions_of(population)
            logged = errordiag.logged_victims(population.victim, theirs)
            recency = errordiag.recency_victims(theirs)
            ranks = matcheddiag.recency_key_ranks(decisions)
            entry = matcheddiag.horizon_statistics(
                decisions, victims, victims, errordiag.recency_victims(decisions), ranks, ranks)
            self.assertEqual(entry["classes"], errordiag.class_rows(theirs, logged))
            self.assertEqual(entry["rates"], {
                "ranker": errordiag.conditional_rates(theirs, logged),
                "recency": errordiag.conditional_rates(theirs, recency),
                "uniform": errordiag.uniform_rates(theirs)})
            self.assertEqual(entry["statistics"]["logged"]["all"],
                             errordiag.statistics(theirs, logged))

    def test_the_runner_equals_the_diagnosis_runner_at_600(self):
        rng = random.Random(14)
        for _ in range(5):
            ranker = random_ranker(rng)
            population = build_population(boundary_decisions(rng, 60), scorer=ranker)
            ours = runner.population_statistics(population, ranker)
            theirs = red.population_statistics(population, {0: ranker, 3: random_ranker(rng)}, 0,
                                               readings=True)
            at_600 = ours["horizons"][600.0]
            self.assertEqual(at_600["classes"], theirs["classes"])
            self.assertEqual(at_600["rates"], theirs["rates"])
            self.assertEqual(at_600["statistics"]["logged"]["all"], theirs["matrix"]["logged"])
            self.assertEqual(at_600["statistics"]["scorer_pi0"]["all"],
                             theirs["matrix"]["scorer_pi0"])
            for field in ("rows", "decisions", "decisions_eligible", "cap_bound", "window",
                          "horizon_seconds", "population_seed", "recorded_argmin_mismatches",
                          "own_victim_mismatches", "own_score_exact", "own_score_max_abs_diff"):
                self.assertEqual(ours[field], theirs[field], field)
            self.assertEqual(set(ours["horizons"]), set(HORIZONS_SECONDS))

    def test_subsets_add_up_and_follow_the_victim(self):
        rng = random.Random(15)
        for _ in range(10):
            population = build_population(boundary_decisions(rng, 50), scorer=random_ranker(rng))
            for h in HORIZONS_SECONDS:
                decisions, victims = logged_at(population, h)
                result = matcheddiag.subset_statistics(decisions, victims)
                for field in ("decisions", "avoidable_count"):
                    self.assertEqual(result["arrival"][field] + result["resident"][field],
                                     result["all"][field])
                self.assertAlmostEqual(result["arrival"]["excess_sum"]
                                       + result["resident"]["excess_sum"],
                                       result["all"]["excess_sum"], places=12)
                resident = ~decisions.arriving[victims]
                self.assertEqual(result["resident"]["decisions"], int(resident.sum()))

    def test_m4_by_subset_by_hand(self):
        population = build_population([
            {"delta_s": [10.0, NEVER], "victim": 0, "arrival": 0},   # arrival, avoidable
            {"delta_s": [10.0, NEVER], "victim": 0, "arrival": 1},   # resident, avoidable
            {"delta_s": [NEVER, 10.0], "victim": 0, "arrival": 1},   # resident, no excess
            {"delta_s": [100.0, NEVER], "victim": 0},                # resident; at 600 only
        ])
        for h, (all_m4, resident_m4) in ((600.0, (0.75, 2 / 3)), (60.0, (0.5, 1 / 3))):
            decisions, victims = logged_at(population, h)
            result = matcheddiag.subset_statistics(decisions, victims)
            self.assertEqual(result["all"]["m4"], all_m4)
            self.assertEqual(result["resident"]["m4"], resident_m4)
            self.assertEqual(result["arrival"]["m4"], 1.0)


# --- the reading rules ----------------------------------------------------------------------------


class ReadingRuleTests(unittest.TestCase):
    def test_composition(self):
        self.assertTrue(matcheddiag.composition_holds(0.99))
        self.assertTrue(matcheddiag.composition_holds(1.0))
        self.assertFalse(matcheddiag.composition_holds(math.nextafter(0.99, 0.0)))
        self.assertFalse(matcheddiag.composition_holds(math.nan))

    def test_the_concordance_prediction(self):
        self.assertTrue(matcheddiag.concordance_prediction_holds(0.5, 0.6))
        self.assertFalse(matcheddiag.concordance_prediction_holds(0.6, 0.6))
        self.assertFalse(matcheddiag.concordance_prediction_holds(0.7, 0.6))
        self.assertFalse(matcheddiag.concordance_prediction_holds(math.nan, 0.6))
        registered = [(trace, cell) for trace in red.TRACES for cell in red.CELLS
                      if matcheddiag.prediction_registered("next_use", *cell)]
        self.assertEqual(len(registered), 8)
        self.assertFalse(any(matcheddiag.prediction_registered("binary", *cell)
                             for cell in red.CELLS))
        self.assertFalse(matcheddiag.prediction_registered("next_use", 0.01, 4.0))

    def test_the_baseline_reading_is_the_diagnosis(self):
        self.assertEqual(matcheddiag.baseline_reading([0.1] * 5, [0.2] * 5), "better")
        self.assertEqual(matcheddiag.baseline_reading([0.3] * 5, [0.2] * 5), "worse")
        self.assertEqual(matcheddiag.baseline_reading([0.1, 0.3, 0.1, 0.1, 0.1], [0.2] * 5),
                         "mixed")


class SpearmanTests(unittest.TestCase):
    def test_average_ranks(self):
        self.assertEqual(matcheddiag.average_ranks([10.0, 20.0, 20.0, 30.0]).tolist(),
                         [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(matcheddiag.average_ranks([3.0, 1.0, 3.0, 3.0, 2.0]).tolist(),
                         [4.0, 1.0, 4.0, 4.0, 2.0])
        self.assertEqual(matcheddiag.average_ranks([0.0, -0.0]).tolist(), [1.5, 1.5])

    def test_by_hand(self):
        self.assertEqual(matcheddiag.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(matcheddiag.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        # Monotone but not linear: still one.
        self.assertEqual(matcheddiag.spearman([1, 2, 3, 4], [1, 8, 27, 1000]), 1.0)
        # Ties: ranks (1, 2.5, 2.5, 4) against (1, 2, 3, 4): 4.5 / sqrt(4.5 * 5).
        self.assertAlmostEqual(matcheddiag.spearman([1, 2, 2, 3], [1, 2, 3, 4]),
                               4.5 / math.sqrt(4.5 * 5.0), places=15)
        # Ties on y: (1, 2, 3, 4, 5) against (1, 2, 3.5, 5, 3.5): 8 / sqrt(10 * 9.5).
        self.assertAlmostEqual(matcheddiag.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]),
                               8.0 / math.sqrt(95.0), places=15)
        # Ties on both sides: (1.5, 1.5, 3) against (1, 2.5, 2.5): 0.75 / sqrt(1.5 * 1.5).
        self.assertAlmostEqual(matcheddiag.spearman([0, 0, 1], [0, 1, 1]), 0.5, places=15)

    def test_undefined_cases_are_nan(self):
        self.assertTrue(math.isnan(matcheddiag.spearman([1, 1, 1], [1, 2, 3])))
        self.assertTrue(math.isnan(matcheddiag.spearman([1, 2, math.nan], [1, 2, 3])))
        self.assertTrue(math.isnan(matcheddiag.spearman([1], [1])))
        with self.assertRaises(ValueError):
            matcheddiag.spearman([1, 2], [1, 2, 3])


# --- the runner on a constructed published run ----------------------------------------------------

REPLAY_COLUMNS = ("trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width",
                  "mechanism", "arm", "family", "seed", "variant", "l1_capacity_bytes",
                  "l2_capacity_bytes", "requested_tokens", "l1_avoided_tokens", "l2_avoided_tokens",
                  "avoided_prefill_tokens", "extra_avoided_tokens")
# The arms each constructed replay table carries; the bridge reads only some.
CONSTRUCTED_ARMS = {
    "error_location_001": [("all16", "lru"), ("all16", "label"), ("all16", "learned"),
                           ("all16", "pi0_binary"), ("all16", "label_binary")],
    "horizon_control_001": [("all16", f"label_binary_{h}") for h in (6, 15, 60, 300, 600)]
                           + [("leaf16", "adm_label")],
    "horizon_fill_001": [("all16", f"label_binary_{h}") for h in (60, 90, 150, 300, 600)],
}


def replay_rows(seed: int = 3) -> dict[str, list[dict]]:
    """Published-looking replay rows of the current grid: arm-independent
    identifiers per trace x cell x seed, random L2-avoided tokens per arm. The
    fill-in's table covers only the cells whose h* is 150 s. Each table also
    holds rows the bridge must skip (other arms, leaf16, a hook-off variant)."""
    rng = random.Random(seed)
    out = {source: [] for source in CONSTRUCTED_ARMS}
    for trace in red.TRACES:
        for fraction, multiplier in red.CELLS:
            for seed_value in red.SEEDS:
                identity = {"trace": trace, "l1_fraction": str(fraction),
                            "l2_multiplier": str(multiplier),
                            "cell": red.cell_label(fraction, multiplier), "family": "x",
                            "seed": str(seed_value), "variant": "main"}
                numbers = {"l1_capacity_bytes": rng.randint(10**6, 10**7),
                           "l2_capacity_bytes": rng.randint(10**6, 10**8),
                           "requested_tokens": rng.randint(10**6, 10**7),
                           "l1_avoided_tokens": rng.randint(0, 10**5)}
                for source, arms in CONSTRUCTED_ARMS.items():
                    if (source == "horizon_fill_001"
                            and matcheddiag.matched_horizon(fraction, multiplier) != 150.0):
                        continue
                    for mechanism, arm in arms:
                        l2 = rng.randint(0, 10**6)
                        eligibility = "all" if mechanism == "all16" else "leaf"
                        out[source].append({
                            **identity, "eligibility": eligibility, "width": "16",
                            "mechanism": mechanism, "arm": arm,
                            **{key: str(value) for key, value in numbers.items()},
                            "l2_avoided_tokens": str(l2),
                            "avoided_prefill_tokens": str(numbers["l1_avoided_tokens"] + l2),
                            "extra_avoided_tokens": str(l2)})
                learned = next(row for row in out["error_location_001"]
                               if row["arm"] == "learned" and row["trace"] == trace
                               and row["cell"] == identity["cell"]
                               and row["seed"] == str(seed_value))
                out["error_location_001"].append({**learned, "variant": "nostats",
                                                  "extra_avoided_tokens": "1",
                                                  "l2_avoided_tokens": "1"})
    return out


def write_replays(paper: Path, rows: dict[str, list[dict]]) -> None:
    for source, entries in rows.items():
        path = paper / source / "replay_seeds.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPLAY_COLUMNS)
            writer.writeheader()
            writer.writerows(entries)


def expected_gap(rows: dict[str, list[dict]], source: str, arm: str, cell: tuple,
                 base_arm: str = "learned") -> list[float]:
    """`U(arm) - U(base_arm)` per seed from the constructed rows, by hand."""
    trace, fraction, multiplier = cell

    def find(table, name, seed):
        return next(row for row in rows[table] if row["arm"] == name and row["trace"] == trace
                    and float(row["l1_fraction"]) == fraction
                    and float(row["l2_multiplier"]) == multiplier
                    and row["seed"] == str(seed) and row["variant"] == "main"
                    and row["mechanism"] == "all16")

    out = []
    for seed in red.SEEDS:
        top, base = find(source, arm, seed), find("error_location_001", base_arm, seed)
        out.append(100.0 * (int(top["extra_avoided_tokens"]) - int(base["extra_avoided_tokens"]))
                   / int(base["requested_tokens"]))
    return out


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_paper(root: Path, decisions: int = 14):
    """A constructed published on-policy run (populations with next uses on
    both sides of every horizon), the diagnosis's own tables computed from it
    by the diagnosis runner, and constructed replay tables, under the current
    grid. Returns the run, the paper directory and the replay rows."""
    with mock.patch.object(fixtures, "DELTA_CHOICES", BOUNDARY_DELTAS):
        published = fixtures.ConstructedRun(root / "published", decisions=decisions)
    paper = root / "paper"
    published.main(paper / runner.DIAGNOSIS_SOURCE, workers=1)
    rows = replay_rows()
    write_replays(paper, rows)
    return published, paper, rows


@contextlib.contextmanager
def patched_run(published, arguments: list[str]):
    """Git answers a clean tree, HEAD `0...0` and `1...1` as the plan's last
    commit; the diagnosis runner reads the constructed on-policy tables."""
    def git(*command):
        if command[0] == "status":
            return ""
        return "1" * 40 if command[0] == "log" else "0" * 40

    paths = published.paths
    with mock.patch.object(red, "PUBLISHED_CONFIG", paths["config"]), \
            mock.patch.object(red, "TEST_POPULATIONS", paths["populations"]), \
            mock.patch.object(red, "CANONICAL_MODELS", paths["canonical"]), \
            mock.patch.object(red, "MODEL_LINKS", paths["links"]), \
            mock.patch.object(red, "SEED_UTILITY", paths["utility"]), \
            mock.patch.object(red, "RUN_MODEL_MANIFEST", paths["manifest"]), \
            mock.patch.object(runner, "_git", side_effect=git), \
            mock.patch.object(runner, "sources_differing_from_head", return_value=[]), \
            mock.patch.object(sys, "argv", ["run_matched_horizon_diagnosis.py", *arguments]):
        yield


class Run:
    """`runner.main` on a constructed run; what it printed is kept."""

    printed = ""

    @classmethod
    def main(cls, published, paper: Path, target: Path, workers: int = 1) -> str:
        buffer = io.StringIO()
        try:
            with patched_run(published, ["--output-dir", str(target), "--workers", str(workers),
                                         "--paper-dir", str(paper)]), \
                    contextlib.redirect_stdout(buffer):
                runner.main()
        finally:
            cls.printed = buffer.getvalue()
        return cls.printed


def _float(text: str) -> float:
    return math.nan if text in ("", None) else float(text)


def _same(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


class EndToEndTests(unittest.TestCase):
    """The full grid: 120 lineages, the diagnosis's tables written by its own
    runner, then this runner with one worker and with two, after every pi3
    population file was deleted."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.published, cls.paper, cls.replays = build_paper(root)
        for row in cls.published.population_rows:
            if row["iteration"] == "3":
                Path(row["population_path"]).unlink()
        cls.outputs = {}
        for workers in (1, 2):
            cls.outputs[workers] = root / f"out_{workers}"
            Run.main(cls.published, cls.paper, cls.outputs[workers], workers)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def table(self, name: str, workers: int = 1) -> list[dict]:
        return _read_rows(self.outputs[workers] / f"{name}.csv")

    def test_every_table_has_exactly_its_documented_columns(self):
        target = self.outputs[1]
        expected = {f"{name}.csv" for name in runner.TABLES} | {"README.md", "run_config.json"}
        self.assertEqual({path.name for path in target.iterdir()}, expected)
        readme = (target / "README.md").read_text(encoding="utf-8")
        for name, (_, columns) in runner.TABLES.items():
            with (target / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, [column for column, _ in columns], name)
            self.assertEqual(len(header), len(set(header)), name)
            self.assertIn(f"## `{name}.csv`", readme)
            for column in header:
                self.assertIn(f"- `{column}`:", readme)

    def test_row_counts(self):
        lineages, cells, horizons = 120, 24, 4
        expected = {"checks": lineages, "classes_seeds": lineages * horizons * 3 * 4,
                    "classes": cells * horizons * 3 * 4, "rates_seeds": lineages * horizons * 2,
                    "rates": cells * horizons * 2, "rates_counts": 5 * 2 * 2,
                    "statistics_seeds": lineages * horizons * 2 * 3,
                    "concordance_seeds": lineages * horizons, "concordance": cells * horizons,
                    "concordance_reading": cells, "concordance_counts": 2,
                    "composition_reading": cells, "composition_counts": 3,
                    "bridge": cells * horizons, "bridge_spearman": 2 * 5 * 2 * 2,
                    "reproduction": 3}
        self.assertEqual(set(expected), set(runner.TABLES))
        for name, count in expected.items():
            self.assertEqual(len(self.table(name)), count, name)

    def test_one_worker_and_two_agree(self):
        for name in runner.TABLES:
            self.assertEqual((self.outputs[1] / f"{name}.csv").read_bytes(),
                             (self.outputs[2] / f"{name}.csv").read_bytes(), name)

    def test_the_matched_column_marks_h_star(self):
        for name, (_, columns) in runner.TABLES.items():
            names = [column for column, _ in columns]
            if "h" not in names:
                continue
            for row in self.table(name):
                star = MATCHED_HORIZON_SECONDS[(float(row["l1_fraction"]),
                                                float(row["l2_multiplier"]))]
                self.assertEqual(float(row["h_star"]), star)
                self.assertEqual(row["matched"], str(float(row["h"]) == star), name)
        horizons = {(row["horizon"], row["matched"]) for row in self.table("rates_counts")}
        self.assertEqual(horizons, {("60", "False"), ("150", "False"), ("300", "False"),
                                    ("600", "False"), ("matched", "True")})

    def test_checks_and_config(self):
        checks = self.table("checks")
        self.assertTrue(all(row["policy"] == "pi0" for row in checks))
        self.assertTrue(all(row["sha256_matches"] == "True" for row in checks))
        self.assertTrue(all(row["own_victim_mismatches"] == "0" for row in checks))
        self.assertTrue(all(row["exceeds_stop_rule"] == "False" for row in checks))
        self.assertTrue(all(row["own_score_exact"] == "True" for row in checks))
        config = json.loads((self.outputs[1] / "run_config.json").read_text())
        self.assertEqual(config["status"], "complete")
        self.assertEqual(config["plan"], "docs/matched-horizon-diagnosis-plan.md")
        self.assertEqual(config["plan_commit"], "1" * 40)
        self.assertEqual(config["code_commit"], "0" * 40)
        self.assertEqual(len(config["populations"]), 120)
        self.assertTrue(all(name.endswith("/pi0") for name in config["populations"]))
        self.assertEqual(len(config["models"]), 4)
        self.assertEqual(config["iterations"], [0])
        self.assertEqual(config["horizons_seconds"], [60.0, 150.0, 300.0, 600.0])
        self.assertEqual({(fraction, multiplier): h for fraction, multiplier, h
                          in config["matched_horizon_seconds"]}, MATCHED_HORIZON_SECONDS)
        # Six on-policy inputs, three diagnosis tables, three replay tables.
        self.assertEqual(len(config["published_inputs"]), 12)
        for name in runner.REPRODUCED:
            self.assertEqual(config["checks"]["reproduction"][name]["differing_values"], 0)
        self.assertEqual(config["checks"]["populations_matching_published_sha256"], 120)
        self.assertEqual(config["workers"], 1)
        self.assertIn("readings", config)
        self.assertIn("source_manifest", config)

    def test_the_600_second_rows_reproduce_the_published_tables_exactly(self):
        for row in self.table("reproduction"):
            self.assertEqual(row["equal"], "True", row)
            self.assertEqual(row["differing_values"], "0")
            self.assertEqual(row["published_columns_absent"], "")
            self.assertEqual(int(row["rows_at_600"]), int(row["published_rows"]))
            self.assertEqual(int(row["compared_values"]),
                             int(row["rows_at_600"]) * int(row["common_columns"]))
        # And independently: every published value, as written, is the one here.
        for name, keys in runner.REPRODUCED.items():
            published = _read_rows(self.paper / runner.DIAGNOSIS_SOURCE / f"{name}.csv")
            ours = {tuple(row[key] for key in keys): row for row in self.table(name)
                    if row["h"] == "600.0"}
            self.assertEqual(len(ours), len(published))
            for row in published:
                mine = ours[tuple(row[key] for key in keys)]
                for column, value in row.items():
                    self.assertEqual(mine[column], value, (name, column))

    def test_the_tables_change_with_the_horizon(self):
        shares = {}
        for row in self.table("classes_seeds"):
            if row["subset"] == "all" and row["class"] == "avoidable_reusable":
                shares.setdefault(row["h"], []).append(int(row["decisions"]))
        self.assertEqual(len(shares), 4)
        self.assertGreater(len({tuple(values) for values in shares.values()}), 1)

    def test_concordance_against_a_literal_loop_on_the_populations(self):
        rows = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                 row["target"], int(row["seed"]), float(row["h"])): row
                for row in self.table("concordance_seeds")}
        for lineage in random.Random(2).sample(red.lineages(), 8):
            population = self.published.population(lineage, 0)
            ranker = self.published.models[lineage + (0,)]
            scores = [float(ranker.score_row(row.tolist())) for row in population.features]
            tiebreak = population.arm_tiebreak.tolist()
            for h in HORIZONS_SECONDS:
                values = {"ranker": [], "recency": []}
                for _, block in population.group_blocks():
                    reusable = [population.next_use_delta_ms[row] <= h * 1000.0 for row in block]
                    for name, keys in (("ranker", [(scores[row], tiebreak[row]) for row in block]),
                                       ("recency", [(tiebreak[row],) for row in block])):
                        value = literal_concordance(keys, reusable)
                        if value is not None:
                            values[name].append(value)
                row = rows[lineage + (h,)]
                self.assertEqual(int(row["decisions"]), len(values["ranker"]))
                self.assertEqual(row["uniform"], "0.5")
                for name in values:
                    if values[name]:
                        self.assertAlmostEqual(float(row[name]),
                                               sum(values[name]) / len(values[name]), places=12)
                    else:
                        self.assertEqual(row[name], "nan")

    def test_the_bridge_against_the_constructed_replays(self):
        statistics = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                       row["target"], int(row["seed"]), float(row["h"]), row["subset"]): row
                      for row in self.table("statistics_seeds") if row["victim_source"] == "logged"}
        for row in self.table("bridge"):
            cell = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))
            h = float(row["h"])
            for subset in runner.M4_SUBSETS:
                values = [float(statistics[cell + (row["target"], seed, h, subset)]["m4"])
                          for seed in red.SEEDS]
                expected = (float(np.mean(values)) if all(map(math.isfinite, values))
                            else math.nan)
                self.assertTrue(_same(float(row[f"m4_{subset}_mean"]), expected))
            # Each target's gaps are against its own ranker's replay.
            own = {"next_use": "learned", "binary": "pi0_binary"}[row["target"]]
            self.assertEqual(row["ranker_arm"], own)
            label = expected_gap(self.replays, "error_location_001", "label", cell, own)
            self.assertEqual(float(row["u_label_minus_ranker_mean"]), float(np.mean(label)))
            self.assertEqual(float(row["u_label_minus_ranker_min"]), min(label))
            other = expected_gap(self.replays, "error_location_001", "label", cell,
                                 "pi0_binary" if own == "learned" else "learned")
            self.assertNotEqual(float(row["u_label_minus_ranker_mean"]), float(np.mean(other)))
            source, arm = runner.LABEL_BINARY_ARMS[h]
            if h == 150.0 and MATCHED_HORIZON_SECONDS[cell[1:]] != 150.0:
                self.assertEqual(row["u_label_binary_minus_ranker_mean"], "nan")
                self.assertEqual((row["label_binary_arm"], row["label_binary_source"]), ("", ""))
            else:
                binary = expected_gap(self.replays, source, arm, cell, own)
                self.assertEqual(float(row["u_label_binary_minus_ranker_mean"]),
                                 float(np.mean(binary)))
                self.assertEqual((row["label_binary_arm"], row["label_binary_source"]),
                                 (arm, source))

    def test_spearman_is_over_the_bridge_rows(self):
        bridge = self.table("bridge")
        for row in self.table("bridge_spearman"):
            members = [entry for entry in bridge if entry["target"] == row["target"]
                       and (entry["matched"] == "True" if row["horizon"] == "matched"
                            else entry["h"] == f"{float(row['horizon'])}")]
            self.assertEqual(int(row["cells"]), 12)
            self.assertEqual(len(members), 12)
            self.assertEqual(row["ranker_arm"],
                             {"next_use": "learned", "binary": "pi0_binary"}[row["target"]])
            self.assertEqual({entry["ranker_arm"] for entry in members}, {row["ranker_arm"]})
            x = [float(entry[f"m4_{row['m4_subset']}_mean"]) for entry in members]
            y = [float(entry[f"u_{row['u_gap']}_minus_ranker_mean"]) for entry in members]
            points = sum(math.isfinite(a) and math.isfinite(b) for a, b in zip(x, y))
            self.assertEqual(int(row["points"]), points)
            expected = matcheddiag.spearman(x, y) if points == 12 else math.nan
            self.assertTrue(_same(float(row["spearman"]), expected), row)
        at_150 = [row for row in self.table("bridge_spearman")
                  if row["horizon"] == "150" and row["u_gap"] == "label_binary"]
        self.assertTrue(all(row["points"] == "4" and row["spearman"] == "nan" for row in at_150))

    def test_the_readings(self):
        concordance = {(row["trace"], row["cell"], row["target"], row["h"]): row
                       for row in self.table("concordance")}
        readings = self.table("concordance_reading")
        for row in readings:
            star = float(row["h_star"])
            self.assertEqual(row["registered"],
                             str(row["target"] == "next_use" and star < 600.0))
            self.assertEqual(row["ranker_at_h_star_mean"],
                             concordance[(row["trace"], row["cell"], row["target"],
                                          str(star))]["ranker_mean"])
            below = float(row["ranker_at_h_star_mean"]) < float(row["ranker_at_600_mean"])
            self.assertEqual(row["below_600"], str(below))
            expected = (("holds" if below else "fails") if row["registered"] == "True"
                        else "not_registered")
            self.assertEqual(row["prediction"], expected)
        for count in self.table("concordance_counts"):
            members = [row for row in readings if row["target"] == count["target"]]
            self.assertEqual(int(count["registered"]),
                             8 if count["target"] == "next_use" else 0)
            self.assertEqual(int(count["holds"]) + int(count["fails"]), int(count["registered"]))
            self.assertEqual(int(count["holds"]),
                             sum(row["prediction"] == "holds" for row in members))
        composition = self.table("composition_reading")
        for row in composition:
            share = float(row["excess_share_mean"])
            self.assertEqual(row["holds"], str(math.isfinite(share) and share >= 0.99))
        totals = {row["target"]: row for row in self.table("composition_counts")}
        self.assertEqual(int(totals["all"]["cells"]), 24)
        self.assertEqual(int(totals["all"]["holds"]),
                         sum(row["holds"] == "True" for row in composition))
        rates = [row for row in self.table("rates") if row["matched"] == "True"]
        for count in self.table("rates_counts"):
            if count["horizon"] != "matched":
                continue
            members = [row for row in rates if row["target"] == count["target"]
                       and row["rate"] == count["rate"]]
            self.assertEqual(int(count["cells"]), 12)
            for reading in ("better", "worse"):
                self.assertEqual(int(count[f"{reading}_than_recency"]),
                                 sum(row["vs_recency"] == reading for row in members))

    def test_the_tabulation_reads_and_prints(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(tabulate.main([str(self.outputs[1])]), 0)
        text = buffer.getvalue()
        for section in ("M0", "M1", "M2", "M3", "M4"):
            self.assertIn(f"### {section}", text)
        tables = {name: _read_rows(self.outputs[1] / f"{name}.csv") for name in tabulate.INPUTS}
        self.assertEqual(len(tabulate.composition_table(tables["composition_reading"])), 24)
        self.assertEqual(len(tabulate.matched_rate_table(tables["rates"])), 24)
        self.assertEqual(
            len(tabulate.concordance_prediction_table(tables["concordance_reading"])), 8)
        matrix = tabulate.concordance_matrix(tables["concordance"])
        self.assertEqual(len(matrix), 24)
        self.assertTrue(all(f"ranker_{h:g}" in row for row in matrix for h in HORIZONS_SECONDS))
        bridge = tabulate.bridge_table(tables["bridge"])
        self.assertEqual(len(bridge), 24)
        self.assertTrue(all(row["h_star"] == MATCHED_HORIZON_SECONDS[
            next((f, m) for f, m in red.CELLS if red.cell_label(f, m) == row["cell"])]
            for row in bridge))
        self.assertEqual(len(tabulate.spearman_table(tables["bridge_spearman"])), 40)
        self.assertEqual(len(tabulate.rate_counts(tables["rates_counts"])), 20)
        self.assertEqual(sorted(path.name for path in self.outputs[1].iterdir()),
                         sorted([f"{name}.csv" for name in runner.TABLES]
                                + ["README.md", "run_config.json"]))


SMALL_GRID = {"TRACES": ("conversation_trace",), "CELLS": ((0.0025, 1.0),),
              "TARGETS": ("next_use",), "SEEDS": (0, 1, 2, 3, 4)}


@contextlib.contextmanager
def small_grid():
    with contextlib.ExitStack() as stack:
        for name, value in SMALL_GRID.items():
            stack.enter_context(mock.patch.object(red, name, value))
        yield


class FailurePathTests(unittest.TestCase):
    """One trace x cell x target, five seeds (h* = 60 s), for the failure paths."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(small_grid())
        self.root = Path(temporary.name)
        self.published, self.paper, self.replays = build_paper(self.root, decisions=12)
        self.target = self.root / "out"

    def _main(self):
        return Run.main(self.published, self.paper, self.target)

    def _refused(self, fragment: str):
        with self.assertRaises(SystemExit) as raised:
            self._main()
        self.assertFalse(self.target.exists())
        text = str(raised.exception) + "\n" + Run.printed
        self.assertIn(fragment, text)
        return text

    def _published(self, name: str) -> Path:
        return self.paper / runner.DIAGNOSIS_SOURCE / f"{name}.csv"

    def test_the_small_grid_completes(self):
        self._main()
        rows = _read_rows(self.target / "reproduction.csv")
        self.assertTrue(all(row["equal"] == "True" for row in rows))
        self.assertEqual(len(_read_rows(self.target / "concordance_reading.csv")), 1)
        reading = _read_rows(self.target / "concordance_reading.csv")[0]
        self.assertEqual((reading["h_star"], reading["registered"]), ("60.0", "True"))

    def test_a_one_ulp_difference_publishes_nothing(self):
        path = self._published("classes")
        with path.open(encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle))
        rows = _read_rows(path)
        row = next(row for row in rows if math.isfinite(_float(row["decision_share_mean"]))
                   and float(row["decision_share_mean"]) > 0)
        row["decision_share_mean"] = repr(math.nextafter(float(row["decision_share_mean"]),
                                                         math.inf))
        _rewrite(path, rows, fields)
        text = self._refused("do not reproduce")
        self.assertIn("decision_share_mean", text)

    def test_a_one_ulp_difference_in_a_seed_rate_publishes_nothing(self):
        path = self._published("rates_seeds")
        rows = _read_rows(path)
        row = next(row for row in rows if math.isfinite(_float(row["recency"])))
        row["recency"] = repr(math.nextafter(float(row["recency"]), -math.inf))
        _rewrite(path, rows)
        self._refused("recency")

    def test_a_missing_published_row_publishes_nothing(self):
        path = self._published("rates")
        rows = _read_rows(path)
        _rewrite(path, rows[1:], list(rows[0]))
        with self.assertRaises(SystemExit):
            self._main()
        self.assertFalse(self.target.exists())
        self.assertIn("reproduction at 600 s, rates: ", Run.printed)
        self.assertIn("1 extra rows", Run.printed)

    def test_a_published_column_absent_here_publishes_nothing(self):
        path = self._published("rates")
        rows = _read_rows(path)
        for row in rows:
            row["another_column"] = "1"
        _rewrite(path, rows)
        self._refused("do not reproduce")

    def test_a_missing_replay_row_is_refused_before_any_population_is_read(self):
        rows = self.replays
        rows["error_location_001"] = [row for row in rows["error_location_001"]
                                      if not (row["arm"] == "learned" and row["seed"] == "2"
                                              and row["variant"] == "main")]
        write_replays(self.paper, rows)
        with mock.patch.object(runner, "lineage_worker",
                               side_effect=AssertionError("a population was read")):
            text = self._refused("checks failed")
        self.assertIn("error_location_001/learned", text)

    def test_replay_rows_that_disagree_are_refused(self):
        rows = self.replays
        changed = next(row for row in rows["horizon_control_001"]
                       if row["arm"] == "label_binary_60")
        changed["requested_tokens"] = str(int(changed["requested_tokens"]) + 1)
        wrong = next(row for row in rows["horizon_control_001"]
                     if row["arm"] == "label_binary_300")
        wrong["width"] = "8"
        extra = next(row for row in rows["horizon_control_001"]
                     if row["arm"] == "label_binary_600")
        extra["extra_avoided_tokens"] = str(int(extra["extra_avoided_tokens"]) + 1)
        rows["horizon_control_001"].append(dict(next(
            row for row in rows["horizon_control_001"]
            if row["arm"] == "label_binary_60" and row["seed"] == "4")))
        write_replays(self.paper, rows)
        text = self._refused("checks failed")
        for fragment in ("differ in", "eligibility/width", "extra != avoided", "duplicate row"):
            self.assertIn(fragment, text)

    def test_a_partial_label_binary_150_cell_is_refused(self):
        rows = self.replays
        template = next(row for row in rows["horizon_control_001"]
                        if row["arm"] == "label_binary_60" and row["seed"] == "0")
        rows["horizon_fill_001"] = [{**template, "arm": "label_binary_150"}]
        write_replays(self.paper, rows)
        self._refused("horizon_fill_001/label_binary_150")

    def test_a_changed_population_is_refused(self):
        lineage = red.lineages()[0]
        population = self.published.population(lineage, 0)
        population.decisions_offered += 1
        row = next(row for row in self.published.population_rows
                   if red._key(row) == lineage + (0,))
        population.save(row["population_path"])
        self._refused("population SHA-256")

    def test_the_stop_rule_writes_only_the_checks(self):
        lineage = red.lineages()[2]
        population = self.published.population(lineage, 0)
        block = next(rows for _, rows in population.group_blocks() if len(rows) >= 2)
        victim = population.victim.copy()
        position = int(np.flatnonzero(victim[block])[0])
        victim[block] = 0
        victim[block[(position + 1) % len(block)]] = 1
        population.victim = victim
        row = next(row for row in self.published.population_rows
                   if red._key(row) == lineage + (0,))
        row["population_sha256"] = population.save(row["population_path"])
        self.published.write()
        with self.assertRaises(SystemExit) as raised:
            self._main()
        self.assertIn("unresolved", str(raised.exception))
        self.assertEqual({path.name for path in self.target.iterdir()},
                         {"checks.csv", "README.md", "run_config.json"})
        checks = _read_rows(self.target / "checks.csv")
        self.assertEqual([row["seed"] for row in checks if row["exceeds_stop_rule"] == "True"],
                         [str(lineage[4])])
        self.assertEqual(json.loads((self.target / "run_config.json").read_text())["status"],
                         "unresolved")
        self.assertIn("UNRESOLVED", (self.target / "README.md").read_text())


class ReplayLoadingTests(unittest.TestCase):
    def test_the_gaps_are_token_differences_in_points(self):
        with small_grid(), tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            rows = replay_rows(seed=8)
            write_replays(paper, rows)
            replays, problems = runner.load_replays(paper)
            self.assertEqual(problems, [])
            # label, learned, pi0_binary, three label_binary_h; 150 s not
            # replayed for h* = 60.
            self.assertEqual(len(replays), 6 * 5)
            gaps = runner.utility_gaps(replays)
            cell = ("conversation_trace", 0.0025, 1.0)
            self.assertEqual(set(gaps), {cell + ("learned",), cell + ("pi0_binary",)})
            for base in ("learned", "pi0_binary"):
                self.assertEqual(gaps[cell + (base,)]["label"],
                                 expected_gap(rows, "error_location_001", "label", cell, base))
                self.assertIsNone(gaps[cell + (base,)]["label_binary"][150.0])
                for h in (60.0, 300.0, 600.0):
                    self.assertEqual(gaps[cell + (base,)]["label_binary"][h],
                                     expected_gap(rows, "horizon_control_001",
                                                  f"label_binary_{h:g}", cell, base))
            self.assertEqual(runner.RANKER_ARMS, {"next_use": "learned", "binary": "pi0_binary"})

    def test_pi0_binary_has_the_checks_of_every_reference_row(self):
        with small_grid(), tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            rows = replay_rows(seed=10)
            missing = [row for row in rows["error_location_001"]
                       if not (row["arm"] == "pi0_binary" and row["seed"] == "3")]
            write_replays(paper, {**rows, "error_location_001": missing})
            _, problems = runner.load_replays(paper)
            self.assertEqual(problems, ["published error_location_001/pi0_binary/"
                                        "conversation_trace/l1=0.0025,l2x1: seeds [3] missing"])
            changed = [dict(row) for row in rows["error_location_001"]]
            target = next(row for row in changed if row["arm"] == "pi0_binary"
                          and row["seed"] == "1" and row["variant"] == "main")
            target["l1_capacity_bytes"] = str(int(target["l1_capacity_bytes"]) + 1)
            target["width"] = "8"
            changed.append(dict(next(row for row in changed if row["arm"] == "pi0_binary"
                                     and row["seed"] == "4" and row["variant"] == "main")))
            write_replays(paper, {**rows, "error_location_001": changed})
            _, problems = runner.load_replays(paper)
            self.assertEqual(len(problems), 3, problems)
            for fragment in ("pi0_binary/conversation_trace/l1=0.0025,l2x1/s1: eligibility/width",
                             "pi0_binary/conversation_trace/l1=0.0025,l2x1/s4: duplicate row",
                             "l1=0.0025,l2x1/s1: the replay rows differ in"):
                self.assertTrue(any(fragment in problem for problem in problems), fragment)

    def test_label_binary_150_is_required_where_h_star_is_150(self):
        with mock.patch.object(red, "TRACES", ("toolagent_trace",)), \
                mock.patch.object(red, "CELLS", ((0.01, 1.0),)), \
                tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            rows = replay_rows(seed=9)
            write_replays(paper, rows)
            replays, problems = runner.load_replays(paper)
            self.assertEqual(problems, [])
            gaps = runner.utility_gaps(replays)
            cell = ("toolagent_trace", 0.01, 1.0)
            self.assertEqual(gaps[cell + ("pi0_binary",)]["label_binary"][150.0],
                             expected_gap(rows, "horizon_fill_001", "label_binary_150", cell,
                                          "pi0_binary"))
            # The fill-in's own 60 s rows are not the ones read.
            self.assertEqual(gaps[cell + ("learned",)]["label_binary"][60.0],
                             expected_gap(rows, "horizon_control_001", "label_binary_60", cell))
            rows["horizon_fill_001"] = [row for row in rows["horizon_fill_001"]
                                        if row["arm"] != "label_binary_150"]
            write_replays(paper, rows)
            _, problems = runner.load_replays(paper)
            self.assertEqual(len(problems), 1)
            self.assertIn("horizon_fill_001/label_binary_150", problems[0])

    def test_a_missing_table_is_a_problem(self):
        with small_grid(), tempfile.TemporaryDirectory() as directory:
            _, problems = runner.load_replays(Path(directory))
            self.assertTrue(any("is missing" in problem for problem in problems))
            _, problems = runner.load_reproduced(Path(directory))
            self.assertEqual(len(problems), 3)


class RefusalTests(unittest.TestCase):
    def _main(self, arguments):
        with mock.patch.object(sys, "argv", ["run_matched_horizon_diagnosis.py", *arguments]), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def test_workers_above_the_cap_or_below_one_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "new")
            for workers, fragment in (("13", "cap"), ("0", "positive")):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", target, "--workers", workers])
                self.assertIn(fragment, str(raised.exception))
            self.assertFalse(Path(target).exists())

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                self._main(["--output-dir", directory])
            self.assertIn("exists", str(raised.exception))

    def test_an_unclean_or_differing_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new"
            with mock.patch.object(runner, "_git", return_value="?? scripts/new.py"):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", str(target)])
            self.assertIn("not clean", str(raised.exception))
            with mock.patch.object(runner, "_git", return_value=""), \
                    mock.patch.object(runner, "sources_differing_from_head",
                                      return_value=["src/persistent_kv_admission/matcheddiag.py"]):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", str(target)])
            self.assertIn("differs from HEAD", str(raised.exception))
            self.assertFalse(target.exists())

    def test_an_undocumented_column_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                runner._write(Path(directory) / "x.csv", [{"target": "next_use"}],
                              "concordance_counts")


if __name__ == "__main__":
    unittest.main()
