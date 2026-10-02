"""Tests for the ranker-error diagnosis (docs/ranker-error-diagnosis-plan.md).

The saved logs are not read before this is reviewed, so everything runs on
constructed populations: a scorer's victim (the first minimum of its score and
the stored tie-break, in stored candidate order) against a literal loop, and
its reproduction of the logged victim on a population logged with the
replay's own `score_row` arithmetic; the four-way matrix of m3 and m4 on two
hand-built populations and two scorers, through the runner's seed-paired
changes; the four classes of reading 3 as a partition of the decisions (with
the one overlap the definitions allow, a victim exactly at the horizon); the
conditional rates by hand and the uniform expectation against an exhaustive
enumeration of the candidates; the concentration numbers; the reading rules,
zeros, nans and mixed included. The runner is run end to end on a constructed
published run written to a temporary directory: every table against its
documented columns and against an independent brute-force computation, one
worker against two, the refusals, changed inputs, a missing utility and the
0.1% stop rule.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from persistent_kv_admission import errordiag
from persistent_kv_admission.decisionpop import FEATURE_COUNT
from persistent_kv_admission.onpolicy import DecisionPopulation, serialize_ranker
from persistent_kv_admission.predictors import LogisticRanker, RidgeRanker

REPOSITORY = Path(__file__).resolve().parents[1]
H = 600.0
NEVER = math.inf
CLIP = -math.log1p(H)


def _runner():
    path = REPOSITORY / "scripts" / "run_ranker_error_diagnosis.py"
    spec = importlib.util.spec_from_file_location("_ranker_error_diagnosis_runner_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _runner()


# --- constructed scorers and populations ---------------------------------------------------


def linear_ranker(coefficients: dict[int, float], intercept: float = 0.0) -> RidgeRanker:
    """`intercept + sum(c * feature)`: no standardisation, so scores are plain."""
    indices = tuple(sorted(coefficients))
    ranker = RidgeRanker(indices=indices, l2=1.0, standardize=False)
    ranker.mean = np.zeros(len(indices))
    ranker.scale = np.ones(len(indices))
    ranker.coefficients = np.asarray([coefficients[index] for index in indices], dtype=float)
    ranker.intercept = float(intercept)
    return ranker


def random_ranker(rng: random.Random):
    """A fitted-looking ranker with non-trivial floats in every term."""
    indices = tuple(sorted(rng.sample(range(FEATURE_COUNT), 3)))
    cls = LogisticRanker if rng.random() < 0.5 else RidgeRanker
    ranker = cls(indices=indices, l2=1.0, standardize=True)
    ranker.mean = np.asarray([rng.uniform(-2, 2) for _ in indices])
    ranker.scale = np.asarray([rng.uniform(0.3, 3.0) for _ in indices])
    ranker.coefficients = np.asarray([rng.uniform(-2, 2) for _ in indices])
    ranker.intercept = rng.uniform(-1, 1)
    return ranker


def replay_victim(scores: list[float], tiebreak: list[float]) -> int:
    """The store's victim, written as a literal loop: the first candidate whose
    (score, tie-break) is strictly below every earlier one's minimum."""
    best = 0
    for index in range(1, len(scores)):
        if (scores[index], tiebreak[index]) < (scores[best], tiebreak[best]):
            best = index
    return best


def build_population(decisions: list[dict], *, scorer=None, seed: int = 0,
                     horizon: float = H, extra_eligible: int = 0) -> DecisionPopulation:
    """A population of hand-written decisions.

    Each decision gives `delta_s` (next use in seconds, `NEVER` for none) and
    optionally `tiebreak`, `features` (rows of 23, or `feature` for column 0
    alone), `arrival` (candidate index or -1), `states` and `victim`. Without a
    `victim` the logged victim is the replay's: the first minimum of the
    scorer's `score_row` and the tie-break. `arm_score` is the scorer's
    `score_row` (zeros without a scorer).
    """
    columns = {name: [] for name in ("timestamp", "trace_group", "ordinal", "state", "delta",
                                      "group", "victim", "arriving", "score", "tiebreak")}
    features = []
    for number, decision in enumerate(decisions):
        width = len(decision["delta_s"])
        if "features" in decision:
            rows = [list(map(float, row)) for row in decision["features"]]
        else:
            values = decision.get("feature", [0.0] * width)
            rows = [[float(value)] + [0.0] * (FEATURE_COUNT - 1) for value in values]
        tiebreak = [float(value) for value in decision.get("tiebreak", range(width))]
        scores = ([float(scorer.score_row(row)) for row in rows] if scorer is not None
                  else [0.0] * width)
        victim = decision.get("victim")
        if victim is None:
            victim = replay_victim(scores, tiebreak)
        arrival = decision.get("arrival", -1)
        states = decision.get("states", [1000 * number + index for index in range(width)])
        for index in range(width):
            columns["timestamp"].append(1000.0 * (number + 1))
            columns["trace_group"].append(number)
            columns["ordinal"].append(0)
            columns["state"].append(states[index])
            columns["delta"].append(decision["delta_s"][index] * 1000.0)
            columns["group"].append(number)
            columns["victim"].append(int(index == victim))
            columns["arriving"].append(int(index == arrival))
            columns["score"].append(scores[index])
            columns["tiebreak"].append(tiebreak[index])
        features.extend(rows)
    count = len(decisions)
    return DecisionPopulation(
        timestamp_ms=np.asarray(columns["timestamp"], dtype=float),
        trace_group_index=np.asarray(columns["trace_group"], dtype=np.int64),
        decision_ordinal=np.asarray(columns["ordinal"], dtype=np.int64),
        state_index=np.asarray(columns["state"], dtype=np.int32),
        features=np.asarray(features, dtype=np.float64).reshape(-1, FEATURE_COUNT),
        next_use_delta_ms=np.asarray(columns["delta"], dtype=float),
        count_within_h=np.zeros(len(columns["delta"])),
        group=np.asarray(columns["group"], dtype=np.int64),
        victim=np.asarray(columns["victim"], dtype=np.int8),
        arriving=np.asarray(columns["arriving"], dtype=np.int8),
        arm_score=np.asarray(columns["score"], dtype=float),
        arm_tiebreak=np.asarray(columns["tiebreak"], dtype=float),
        horizon_seconds=horizon, window="test", seed=seed,
        decisions_offered=count + extra_eligible + 2,
        decisions_eligible=count + extra_eligible, max_decisions=count,
    )


DELTA_CHOICES = (NEVER, NEVER, H, 1.0, 10.0, 50.0, 300.0, 599.0, 601.0, 0.0)


def random_decisions(rng: random.Random, count: int, *, max_width: int = 6,
                     feature_values: int = 4, state_pool: int = 30) -> list[dict]:
    """Decisions with ties everywhere: next uses from a short list (with
    exactly H and never), small-integer tie-breaks and feature rows drawn
    from a few values, so equal scores and equal keys occur."""
    out = []
    for _ in range(count):
        width = rng.randint(1, max_width)
        out.append({
            "delta_s": [rng.choice(DELTA_CHOICES) for _ in range(width)],
            "tiebreak": [float(rng.randint(0, 3)) for _ in range(width)],
            "features": [[float(rng.randint(0, feature_values)) * 0.37 + 0.11 * column
                          for column in range(FEATURE_COUNT)] for _ in range(width)],
            "arrival": 0 if rng.random() < 0.5 else -1,
            "states": rng.sample(range(state_pool), width),
        })
    return out


# --- victims -----------------------------------------------------------------------------------


class VictimTests(unittest.TestCase):
    def test_first_minimum_equals_a_literal_loop_under_ties(self):
        rng = random.Random(3)
        for _ in range(200):
            widths = [rng.randint(1, 7) for _ in range(rng.randint(1, 6))]
            primary = [float(rng.choice((-0.0, 0.0, 1.0, 2.0))) for _ in range(sum(widths))]
            secondary = [float(rng.randint(0, 2)) for _ in range(sum(widths))]
            starts = np.cumsum([0] + widths[:-1]).astype(np.int64)
            got = errordiag.first_minimum(np.asarray(primary), np.asarray(secondary), starts,
                                          np.asarray(widths, dtype=np.int64))
            for start, width, row in zip(starts, widths, got):
                block = slice(int(start), int(start + width))
                self.assertEqual(int(row), int(start) + replay_victim(primary[block], secondary[block]))

    def test_tie_break_then_stored_order(self):
        population = build_population([
            {"delta_s": [NEVER] * 3, "feature": [1.0, 1.0, 2.0], "tiebreak": [5.0, 3.0, 0.0],
             "victim": 0},
            {"delta_s": [NEVER] * 3, "feature": [4.0, 4.0, 4.0], "tiebreak": [2.0, 1.0, 1.0],
             "victim": 0},
        ])
        decisions = errordiag.decisions_of(population)
        scores = errordiag.scorer_scores(linear_ranker({0: 1.0}), population)
        # Equal scores at rows 0 and 1, tie-break 3 < 5; then a full tie at 4 and 5.
        self.assertEqual(errordiag.scorer_victims(scores, decisions).tolist(), [1, 4])
        self.assertEqual(errordiag.recency_victims(decisions).tolist(), [2, 4])

    def test_own_scorer_reproduces_victims_logged_with_score_row(self):
        """The population is logged with the replay's per-row `score_row`; the
        diagnosis recomputes with the vectorised `sequential_ranker_score`. The
        two must give identical floats and the identical victim."""
        rng = random.Random(11)
        for _ in range(5):
            ranker = random_ranker(rng)
            population = build_population(random_decisions(rng, 60), scorer=ranker)
            decisions = errordiag.decisions_of(population)
            scores = errordiag.scorer_scores(ranker, population)
            self.assertTrue(np.array_equal(scores, population.arm_score))
            self.assertEqual(errordiag.scorer_victims(scores, decisions).tolist(),
                             errordiag.logged_victims(population.victim, decisions).tolist())

    def test_logged_victims_one_per_decision(self):
        population = build_population([{"delta_s": [1.0, 2.0], "victim": 1},
                                       {"delta_s": [3.0], "victim": 0}])
        decisions = errordiag.decisions_of(population)
        self.assertEqual(errordiag.logged_victims(population.victim, decisions).tolist(), [1, 2])
        with self.assertRaises(ValueError):
            errordiag.logged_victims(np.asarray([0, 1, 0]), decisions)

    def test_a_non_finite_score_is_refused(self):
        population = build_population([{"delta_s": [1.0, 2.0], "feature": [1e308, 1e308]}])
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            errordiag.scorer_scores(linear_ranker({0: 10.0}), population)


class LabelTests(unittest.TestCase):
    def test_label_and_reuse_at_the_horizon(self):
        population = build_population([{"delta_s": [0.0, 10.0, H, H + 0.001, NEVER],
                                        "victim": 0}])
        decisions = errordiag.decisions_of(population)
        expected = [0.0, -math.log1p(10.0), CLIP, CLIP, CLIP]
        self.assertTrue(np.allclose(decisions.label, expected, rtol=0, atol=1e-15))
        self.assertEqual(len(set(decisions.label[2:].tolist())), 1)
        self.assertEqual(decisions.reusable.tolist(), [True, True, True, False, False])
        self.assertAlmostEqual(float(decisions.min_label[0]), CLIP, places=14)
        self.assertEqual(decisions.reusable_count.tolist(), [3])
        self.assertEqual(decisions.above_min_count.tolist(), [2])
        # Three candidates share the clipped label; the tie-break 2 < 3 < 4 picks row 2.
        self.assertEqual(decisions.reference.tolist(), [2])


# --- the four-way matrix -------------------------------------------------------------------------

# S0 evicts the lowest feature, S3 the highest.
S0 = linear_ranker({0: 1.0})
S3 = linear_ranker({0: -1.0})
P0_DECISIONS = [
    {"delta_s": [10.0, NEVER, 100.0], "feature": [1.0, 2.0, 3.0], "arrival": 0},
    {"delta_s": [NEVER, 5.0], "feature": [5.0, 4.0], "arrival": -1},
]
P3_DECISIONS = [
    {"delta_s": [30.0, 20.0, NEVER], "feature": [1.0, 3.0, 2.0], "arrival": 1},
    {"delta_s": [1.0, 2.0, 3.0], "feature": [3.0, 2.0, 1.0], "arrival": -1},
]


def excess_of(delta_victim: float, delta_minimum: float) -> float:
    return -math.log1p(min(delta_victim, H)) + math.log1p(min(delta_minimum, H))


MATRIX = {
    # (population, scorer): (m3, m4)
    (0, 0): ((excess_of(10.0, NEVER) + excess_of(5.0, NEVER)) / 2, 1.0),
    (0, 3): ((excess_of(100.0, NEVER) + 0.0) / 2, 0.5),
    (3, 0): ((excess_of(30.0, NEVER) + 0.0) / 2, 0.5),
    (3, 3): ((excess_of(20.0, NEVER) + excess_of(1.0, 3.0)) / 2, 0.5),
}


class FourWayMatrixTests(unittest.TestCase):
    def setUp(self):
        self.populations = {0: build_population(P0_DECISIONS, scorer=S0),
                            3: build_population(P3_DECISIONS, scorer=S3)}
        self.statistics = {
            iteration: runner.population_statistics(population, {0: S0, 3: S3}, iteration,
                                                     readings=iteration == 0)
            for iteration, population in self.populations.items()}

    def test_matrix_by_hand(self):
        for (population, scorer), (m3, m4) in MATRIX.items():
            values = self.statistics[population]["matrix"][f"scorer_pi{scorer}"]
            self.assertAlmostEqual(values["m3"], m3, places=12)
            self.assertEqual(values["m4"], m4)
            self.assertEqual(values["decisions"], 2)
        for iteration in (0, 3):
            entry = self.statistics[iteration]
            self.assertEqual(entry["matrix"]["logged"], entry["matrix"][f"scorer_pi{iteration}"])
            self.assertEqual(entry["own_victim_mismatches"], 0)
            self.assertTrue(entry["own_score_exact"])
            self.assertEqual(entry["recorded_argmin_mismatches"], 0)
        self.assertIn("classes", self.statistics[0])
        self.assertNotIn("classes", self.statistics[3])

    def test_seed_paired_changes(self):
        lineage = ("trace", 0.01, 1.0, "next_use", 0)
        results = {lineage: {"populations": self.statistics}}
        utility = {"delta_u_points": 0.25, "delta_u_label_points": -0.5}
        with mock.patch.object(runner, "TRACES", ("trace",)), \
                mock.patch.object(runner, "CELLS", ((0.01, 1.0),)), \
                mock.patch.object(runner, "TARGETS", ("next_use",)), \
                mock.patch.object(runner, "SEEDS", (0,)):
            row = runner.selection_seed_rows(results, {lineage: utility})[0]
        for statistic, position in (("m3", 0), ("m4", 1)):
            value = {key: MATRIX[key][position] for key in MATRIX}
            self.assertAlmostEqual(row[f"delta_sel_{statistic}_pi0pop"],
                                   value[(0, 3)] - value[(0, 0)], places=12)
            self.assertAlmostEqual(row[f"delta_sel_{statistic}_pi3pop"],
                                   value[(3, 3)] - value[(3, 0)], places=12)
            self.assertAlmostEqual(row[f"delta_own_{statistic}"],
                                   value[(3, 3)] - value[(0, 0)], places=12)
            self.assertAlmostEqual(row[f"delta_own_{statistic}_logged"],
                                   value[(3, 3)] - value[(0, 0)], places=12)
        self.assertEqual(row["delta_u_points"], 0.25)

    def test_a_victim_other_than_the_scorers_is_a_mismatch(self):
        decisions = [dict(decision) for decision in P0_DECISIONS]
        decisions[1]["victim"] = 0
        population = build_population(decisions, scorer=S0)
        entry = runner.population_statistics(population, {0: S0, 3: S3}, 0, readings=True)
        self.assertEqual(entry["own_victim_mismatches"], 1)
        self.assertEqual(entry["recorded_argmin_mismatches"], 1)
        # Logged statistics follow the logged victim, the scorer's its own.
        self.assertEqual(entry["matrix"]["logged"]["m4"], 0.5)
        self.assertEqual(entry["matrix"]["scorer_pi0"]["m4"], 1.0)


# --- reading 3 ------------------------------------------------------------------------------------


class ClassTests(unittest.TestCase):
    def test_each_definition_by_hand(self):
        population = build_population([
            {"delta_s": [NEVER, NEVER], "tiebreak": [2, 1], "victim": 1},     # reference
            {"delta_s": [NEVER, NEVER], "tiebreak": [2, 1], "victim": 0},     # other tie-break
            {"delta_s": [10.0, NEVER], "victim": 0, "arrival": 0},           # avoidable
            {"delta_s": [10.0, 20.0], "victim": 0, "arrival": 0},            # order error
            {"delta_s": [10.0, 20.0], "victim": 1},                          # reference
            {"delta_s": [5.0], "victim": 0, "arrival": 0},                   # reference
        ])
        decisions = errordiag.decisions_of(population)
        victims = errordiag.logged_victims(population.victim, decisions)
        classes = [errordiag.CLASSES[index] for index in errordiag.classify(decisions, victims)]
        self.assertEqual(classes, ["reference_victim", "other_tiebreak", "avoidable_reusable",
                                   "order_error", "reference_victim", "reference_victim"])
        self.assertTrue((errordiag.class_memberships(decisions, victims).sum(axis=0) == 1).all())

    def test_a_victim_at_the_horizon_is_the_only_overlap(self):
        population = build_population([
            {"delta_s": [H, NEVER], "tiebreak": [0, 1], "victim": 0},
            {"delta_s": [H, NEVER], "tiebreak": [1, 0], "victim": 0},
            {"delta_s": [H, 10.0], "tiebreak": [1, 0], "victim": 0},
        ])
        decisions = errordiag.decisions_of(population)
        victims = errordiag.logged_victims(population.victim, decisions)
        memberships = errordiag.class_memberships(decisions, victims)
        self.assertEqual(memberships.sum(axis=0).tolist(), [2, 2, 1])
        self.assertEqual(errordiag.classify(decisions, victims).tolist(), [0, 1, 0])
        self.assertTrue(memberships[errordiag.AVOIDABLE_CLASS, :2].all())
        rows = {(row["subset"], row["class"]): row for row in errordiag.class_rows(decisions, victims)}
        self.assertEqual(rows[("all", "reference_victim")]["overlap_decisions"], 1)
        self.assertEqual(rows[("all", "other_tiebreak")]["overlap_decisions"], 1)
        self.assertEqual(rows[("all", "avoidable_reusable")]["decisions"], 0)
        # m4 still counts both: it is the plan's definition, not a class.
        self.assertEqual(errordiag.statistics(decisions, victims)["avoidable_count"], 2)

    def test_the_classes_partition_random_populations(self):
        rng = random.Random(5)
        for trial in range(30):
            population = build_population(random_decisions(rng, 80), scorer=random_ranker(rng))
            decisions = errordiag.decisions_of(population)
            victims = errordiag.logged_victims(population.victim, decisions)
            memberships = errordiag.class_memberships(decisions, victims)
            counts = memberships.sum(axis=0)
            self.assertTrue((counts >= 1).all())
            delta = population.next_use_delta_ms[victims]
            horizon = (delta == H * 1000.0) & ~decisions.all_reusable
            self.assertTrue((counts[~horizon] == 1).all())
            self.assertTrue((counts[horizon] == 2).all())
            classes = errordiag.classify(decisions, victims)
            self.assertTrue(np.isin(classes, range(4)).all())
            value = errordiag.excess(decisions, victims)
            self.assertTrue((value[classes <= 1] == 0).all())
            self.assertTrue((value[classes >= 2] > 0).all())
            rows = errordiag.class_rows(decisions, victims)
            self.assertEqual(len(rows), len(errordiag.SUBSETS) * len(errordiag.CLASSES))
            for subset in errordiag.SUBSETS:
                members = [row for row in rows if row["subset"] == subset]
                self.assertEqual(sum(row["decisions"] for row in members),
                                 members[0]["subset_decisions"])
                if members[0]["subset_decisions"]:
                    self.assertAlmostEqual(sum(row["decision_share"] for row in members), 1.0)
                if members[0]["subset_excess_sum"] > 0:
                    self.assertAlmostEqual(sum(row["excess_share"] for row in members), 1.0)
            total = float(value.sum())
            for name in errordiag.CLASSES:
                pick = {row["subset"]: row for row in rows if row["class"] == name}
                self.assertEqual(pick["arrival"]["decisions"] + pick["resident"]["decisions"],
                                 pick["all"]["decisions"])
                self.assertAlmostEqual(pick["arrival"]["excess_sum"] + pick["resident"]["excess_sum"],
                                       pick["all"]["excess_sum"])
                if total > 0:
                    self.assertAlmostEqual(pick["arrival"]["excess_share_of_population"]
                                           + pick["resident"]["excess_share_of_population"],
                                           pick["all"]["excess_share_of_population"])

    def test_shares_by_subset(self):
        population = build_population([
            {"delta_s": [10.0, NEVER], "victim": 0, "arrival": 0},   # avoidable, arrival
            {"delta_s": [10.0, 20.0], "victim": 0, "arrival": 1},    # order error, resident
            {"delta_s": [NEVER, 30.0], "victim": 0, "arrival": 1},   # reference, resident
            {"delta_s": [5.0, NEVER], "victim": 0},                  # avoidable, resident
        ])
        decisions = errordiag.decisions_of(population)
        victims = errordiag.logged_victims(population.victim, decisions)
        rows = {(row["subset"], row["class"]): row for row in errordiag.class_rows(decisions, victims)}
        avoidable = [excess_of(10.0, NEVER), excess_of(5.0, NEVER)]
        order = excess_of(10.0, 20.0)
        total = sum(avoidable) + order
        self.assertEqual(rows[("all", "avoidable_reusable")]["decision_share"], 0.5)
        self.assertAlmostEqual(rows[("all", "avoidable_reusable")]["excess_share"],
                               sum(avoidable) / total)
        self.assertEqual(rows[("arrival", "avoidable_reusable")]["decision_share"], 1.0)
        self.assertEqual(rows[("arrival", "avoidable_reusable")]["decision_share_of_population"], 0.25)
        self.assertAlmostEqual(rows[("arrival", "avoidable_reusable")]["excess_share_of_population"],
                               avoidable[0] / total)
        self.assertAlmostEqual(rows[("resident", "order_error")]["excess_share"],
                               order / (order + avoidable[1]))
        self.assertAlmostEqual(rows[("resident", "reference_victim")]["decision_share"], 1 / 3)
        self.assertTrue(math.isnan(errordiag.class_rows(
            *self._no_excess())[0]["excess_share"]))

    @staticmethod
    def _no_excess():
        population = build_population([{"delta_s": [NEVER, NEVER], "victim": 0}])
        decisions = errordiag.decisions_of(population)
        return decisions, errordiag.logged_victims(population.victim, decisions)


# --- reading 4 ------------------------------------------------------------------------------------


class RateTests(unittest.TestCase):
    def test_rates_by_hand(self):
        population = build_population([
            {"delta_s": [10.0, NEVER, NEVER], "tiebreak": [0, 1, 2], "victim": 0},  # mixed
            {"delta_s": [10.0, NEVER], "tiebreak": [1, 0], "victim": 1},            # mixed
            {"delta_s": [10.0, 20.0, 30.0], "tiebreak": [0, 2, 1], "victim": 0},    # order
            {"delta_s": [10.0, 10.0], "tiebreak": [0, 1], "victim": 0},             # one label
            {"delta_s": [NEVER, NEVER], "victim": 0},                               # none
        ])
        decisions = errordiag.decisions_of(population)
        victims = errordiag.logged_victims(population.victim, decisions)
        self.assertEqual(errordiag.conditional_rates(decisions, victims), {
            "avoidable_decisions": 2, "avoidable_rate": 0.5,
            "order_decisions": 1, "order_rate": 1.0})
        self.assertEqual(errordiag.conditional_rates(decisions, errordiag.recency_victims(decisions)), {
            "avoidable_decisions": 2, "avoidable_rate": 0.5,
            "order_decisions": 1, "order_rate": 1.0})
        uniform = errordiag.uniform_rates(decisions)
        self.assertAlmostEqual(uniform["avoidable_rate"], (1 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(uniform["order_rate"], 2 / 3)
        self.assertEqual((uniform["avoidable_decisions"], uniform["order_decisions"]), (2, 1))

    def test_the_uniform_expectation_is_the_mean_over_every_candidate(self):
        """With L the least common multiple of the widths, victim k mod width
        visits every candidate of every decision L / width times, so the mean
        of the realised rates over k = 0..L-1 is the uniform expectation."""
        rng = random.Random(17)
        for _ in range(10):
            population = build_population(random_decisions(rng, 40))
            decisions = errordiag.decisions_of(population)
            period = math.lcm(*decisions.widths.tolist())
            realised = [errordiag.conditional_rates(
                decisions, decisions.starts + np.mod(k, decisions.widths)) for k in range(period)]
            uniform = errordiag.uniform_rates(decisions)
            for rate in errordiag.RATES:
                values = [entry[f"{rate}_rate"] for entry in realised]
                if math.isnan(uniform[f"{rate}_rate"]):
                    self.assertTrue(all(math.isnan(value) for value in values))
                else:
                    self.assertAlmostEqual(uniform[f"{rate}_rate"], float(np.mean(values)), places=12)

    def test_no_eligible_decisions_is_nan(self):
        population = build_population([{"delta_s": [NEVER, NEVER], "victim": 0}])
        decisions = errordiag.decisions_of(population)
        victims = errordiag.logged_victims(population.victim, decisions)
        for rates in (errordiag.conditional_rates(decisions, victims),
                      errordiag.uniform_rates(decisions)):
            self.assertEqual((rates["avoidable_decisions"], rates["order_decisions"]), (0, 0))
            self.assertTrue(math.isnan(rates["avoidable_rate"]))
            self.assertTrue(math.isnan(rates["order_rate"]))


# --- reading 5 ------------------------------------------------------------------------------------


class ConcentrationTests(unittest.TestCase):
    def test_by_hand(self):
        states = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(errordiag.concentration(np.asarray(states)), {
            "evictions": 12, "distinct_states": 9, "repeated_states": 2,
            "repeat_share": 5 / 12, "top_states": 1, "top_share": 3 / 12})

    def test_top_count_is_ten_percent_rounded_up_in_integers(self):
        for distinct, top in ((1, 1), (9, 1), (10, 1), (11, 2), (30, 3), (31, 4), (70, 7)):
            result = errordiag.concentration(np.arange(distinct))
            self.assertEqual(result["top_states"], top)
            self.assertEqual(result["repeat_share"], 0.0)
            self.assertAlmostEqual(result["top_share"], top / distinct)

    def test_empty(self):
        result = errordiag.concentration(np.zeros(0, dtype=np.int64))
        self.assertEqual((result["evictions"], result["distinct_states"], result["top_states"]),
                         (0, 0, 0))
        self.assertTrue(math.isnan(result["repeat_share"]) and math.isnan(result["top_share"]))

    def test_population_evictions_are_the_avoidable_class_by_subset(self):
        population = build_population([
            {"delta_s": [10.0, NEVER], "states": [7, 8], "victim": 0, "arrival": 0},
            {"delta_s": [20.0, NEVER], "states": [7, 9], "victim": 0},
            {"delta_s": [30.0, NEVER], "states": [3, 9], "victim": 0, "arrival": 1},
            {"delta_s": [5.0, NEVER], "states": [2, 9], "victim": 0, "arrival": 0},
            {"delta_s": [H, NEVER], "states": [5, 6], "victim": 0},     # horizon: no excess
            {"delta_s": [10.0, 20.0], "states": [5, 4], "victim": 0},   # order error
        ])
        entry = runner.population_statistics(population, {0: S0, 3: S3}, 0, readings=True)
        rows = {row["subset"]: row for row in entry["concentration"]}
        self.assertEqual([row["subset"] for row in entry["concentration"]],
                         list(errordiag.SUBSETS))
        # all: states 7, 7, 3, 2; arrival victims: 7, 2; resident victims: 7, 3.
        self.assertEqual(rows["all"], {"subset": "all", "evictions": 4, "distinct_states": 3,
                                       "repeated_states": 1, "repeat_share": 0.5,
                                       "top_states": 1, "top_share": 0.5})
        self.assertEqual(rows["arrival"], {"subset": "arrival", "evictions": 2,
                                           "distinct_states": 2, "repeated_states": 0,
                                           "repeat_share": 0.0, "top_states": 1,
                                           "top_share": 0.5})
        self.assertEqual((rows["resident"]["evictions"], rows["resident"]["distinct_states"],
                          rows["resident"]["repeat_share"]), (2, 2, 0.0))

    def test_subsets_split_the_evictions_on_random_populations(self):
        rng = random.Random(23)
        for _ in range(20):
            population = build_population(random_decisions(rng, 80, state_pool=12),
                                          scorer=random_ranker(rng))
            decisions = errordiag.decisions_of(population)
            victims = errordiag.logged_victims(population.victim, decisions)
            rows = {row["subset"]: row for row in errordiag.concentration_rows(decisions, victims)}
            classes = {row["subset"]: row for row in errordiag.class_rows(decisions, victims)
                       if row["class"] == "avoidable_reusable"}
            for subset in errordiag.SUBSETS:
                self.assertEqual(rows[subset]["evictions"], classes[subset]["decisions"])
            self.assertEqual(rows["arrival"]["evictions"] + rows["resident"]["evictions"],
                             rows["all"]["evictions"])
            self.assertLessEqual(max(rows["arrival"]["distinct_states"],
                                     rows["resident"]["distinct_states"]),
                                 rows["all"]["distinct_states"])


# --- the reading rules ----------------------------------------------------------------------------


class ReadingRuleTests(unittest.TestCase):
    def test_stop_rule(self):
        for mismatches, decisions, expected in ((0, 40000, False), (40, 40000, False),
                                                (41, 40000, True), (1, 1000, False),
                                                (2, 1000, True), (1, 999, True), (0, 1, False)):
            self.assertEqual(errordiag.exceeds_stop_rule(mismatches, decisions), expected,
                             (mismatches, decisions))

    def test_has_sign_of(self):
        self.assertTrue(errordiag.has_sign_of(0.2, 1.0))
        self.assertTrue(errordiag.has_sign_of(-0.2, -1.0))
        self.assertFalse(errordiag.has_sign_of(-0.2, 1.0))
        self.assertTrue(errordiag.has_sign_of(0.0, 0.0))
        self.assertTrue(errordiag.has_sign_of(-0.0, 0.0))
        self.assertFalse(errordiag.has_sign_of(0.0, 1.0))
        self.assertFalse(errordiag.has_sign_of(1.0, 0.0))
        self.assertFalse(errordiag.has_sign_of(math.nan, 1.0))
        self.assertFalse(errordiag.has_sign_of(1.0, math.nan))

    def test_selection_reading(self):
        self.assertEqual(errordiag.selection_reading(True, True), "carried_by_selection")
        self.assertEqual(errordiag.selection_reading(True, False), "population_dependent")
        self.assertEqual(errordiag.selection_reading(False, True), "population_dependent")
        self.assertEqual(errordiag.selection_reading(False, False), "not_carried_by_selection")

    def test_baseline_reading(self):
        base = [0.5] * 5
        self.assertEqual(errordiag.baseline_reading([0.1, 0.2, 0.3, 0.4, 0.49], base), "better")
        self.assertEqual(errordiag.baseline_reading([0.6] * 5, base), "worse")
        self.assertEqual(errordiag.baseline_reading([0.1, 0.2, 0.3, 0.4, 0.5], base), "mixed")
        self.assertEqual(errordiag.baseline_reading([0.1, 0.2, 0.3, 0.4, 0.6], base), "mixed")
        self.assertEqual(errordiag.baseline_reading([0.1, 0.2, math.nan, 0.4, 0.4], base), "mixed")
        self.assertEqual(errordiag.baseline_reading([], []), "mixed")
        with self.assertRaises(ValueError):
            errordiag.baseline_reading([0.1], base)

    def test_window_contradicts(self):
        self.assertTrue(errordiag.window_contradicts("consistent_gain", -0.1))
        self.assertFalse(errordiag.window_contradicts("consistent_gain", 0.0))
        self.assertFalse(errordiag.window_contradicts("consistent_gain", 0.1))
        self.assertTrue(errordiag.window_contradicts("consistent_loss", 0.1))
        self.assertFalse(errordiag.window_contradicts("mixed", -5.0))

    def test_seed_signs(self):
        self.assertEqual(errordiag.seed_signs([1.0, -2.0, 0.0, -0.0, math.nan]), "+-00n")

    def test_selection_rows_apply_the_rule_to_the_means(self):
        """Reading 1 compares the sign of the mean of -Δsel with the mean ΔU,
        not seed by seed; a zero mean ΔU needs a zero mean of -Δsel."""
        def seed_row(seed, delta_u, pi0pop, pi3pop):
            row = {"trace": "trace", "l1_fraction": 0.01, "l2_multiplier": 1.0,
                   "target": "next_use", "seed": seed, "delta_u_points": delta_u}
            for statistic in ("m3", "m4"):
                row.update({f"delta_sel_{statistic}_pi0pop": pi0pop,
                            f"delta_sel_{statistic}_pi3pop": pi3pop,
                            f"delta_own_{statistic}": pi0pop, f"delta_own_{statistic}_logged": pi0pop})
            return row

        cases = (
            # (ΔU per seed, Δsel(pi0) per seed, Δsel(pi3) per seed, expected reading)
            ([1, 1, 1, 1, -1], [-1, -1, -1, -1, 2], [-1] * 5, "carried_by_selection"),
            ([1] * 5, [-1, -1, -1, -1, 5], [-1] * 5, "population_dependent"),
            ([-1] * 5, [-1] * 5, [-1] * 5, "not_carried_by_selection"),
            ([0] * 5, [0] * 5, [0] * 5, "carried_by_selection"),
            ([0] * 5, [0] * 5, [1, 0, 0, 0, 0], "population_dependent"),
        )
        with mock.patch.object(runner, "TRACES", ("trace",)), \
                mock.patch.object(runner, "CELLS", ((0.01, 1.0),)), \
                mock.patch.object(runner, "TARGETS", ("next_use",)):
            for delta_u, pi0pop, pi3pop, expected in cases:
                rows = [seed_row(seed, *values) for seed, values in
                        enumerate(zip(delta_u, pi0pop, pi3pop))]
                selection = runner.selection_rows(rows)
                self.assertEqual([row["selection_reading"] for row in selection], [expected] * 2)
                counts = runner.selection_count_rows(selection)
                self.assertEqual(sum(row[expected] for row in counts), 2)


# --- the runner on a constructed published run --------------------------------------------------


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConstructedRun:
    """A published on-policy run of the runner's current grid in `root`: the
    four tables, the config recording their SHA-256, the run's model manifest,
    a model per lineage at pi0 (shared per trace x target) and pi3, and a small
    population per lineage at pi0 and pi3 logged by its own model. pi1 and pi2
    are listed with files that do not exist: the run must not read them."""

    def __init__(self, root: Path, decisions: int = 14, seed: int = 7) -> None:
        self.root = root
        rng = random.Random(seed)
        self.paths = {
            "config": root / "onpolicy_learning_config.json",
            "populations": root / "onpolicy_test_populations.csv",
            "canonical": root / "onpolicy_canonical_models.csv",
            "links": root / "onpolicy_model_links.csv",
            "utility": root / "onpolicy_seed_utility.csv",
            "manifest": root / "run" / "canonical_model_manifest.csv",
        }
        (root / "run").mkdir(parents=True)
        self.models, self.population_rows, canonical, links, utility = {}, [], [], [], []
        shared = {}
        for trace in runner.TRACES:
            for target in runner.TARGETS:
                path = root / "run" / "models" / "pi0" / f"{trace}__{target}.json"
                ranker = random_ranker(rng)
                shared[(trace, target)] = (ranker, str(path), serialize_ranker(ranker, path))
        for lineage in runner.lineages():
            trace, fraction, multiplier, target, seed_value = lineage
            slug = f"{trace}__l1_{fraction:g}__l2x_{multiplier:g}__{target}__seed_{seed_value}"
            identity = {"trace": trace, "l1_fraction": str(fraction),
                        "l2_multiplier": str(multiplier),
                        "cell": runner.cell_label(fraction, multiplier), "target": target,
                        "seed": str(seed_value)}
            requested, window = rng.randint(10_000, 20_000), rng.randint(5_000, 9_000)
            l1, window_l1 = rng.randint(0, 2_000), rng.randint(0, 1_000)
            pi0_avoided = None
            for iteration in range(4):
                policy = {**identity, "iteration": str(iteration), "policy": f"pi{iteration}"}
                if iteration == 0:
                    ranker, model_path, model_sha = shared[(trace, target)]
                else:
                    ranker = random_ranker(rng)
                    model_path = root / "run" / "models" / f"pi{iteration}" / f"{slug}.json"
                    model_sha = serialize_ranker(ranker, model_path)
                    model_path = str(model_path)
                self.models[lineage + (iteration,)] = ranker
                model_row = {**policy, "model_path": model_path, "model_sha256": model_sha,
                             "shared_pi0": str(iteration == 0),
                             "fitted_from_population_iteration": "" if iteration == 0
                             else str(iteration - 1)}
                links.append(model_row)
                if iteration > 0 or not any(row["iteration"] == "0" and row["trace"] == trace
                                            and row["target"] == target for row in canonical):
                    canonical.append(dict(model_row))
                population_path = root / "run" / "populations" / slug / f"pi{iteration}.npz"
                if iteration in runner.ITERATIONS:
                    population = build_population(random_decisions(rng, decisions), scorer=ranker,
                                                  seed=seed_value, extra_eligible=3)
                    digest = population.save(population_path)
                    counts = {"rows": str(len(population)), "decisions_kept": str(decisions),
                              "decisions_eligible": str(population.decisions_eligible)}
                else:
                    digest = "0" * 64
                    counts = {"rows": "1", "decisions_kept": "1", "decisions_eligible": "1"}
                self.population_rows.append({**policy, "window": "test",
                                             "population_path": str(population_path),
                                             "population_sha256": digest,
                                             "feature_dtype": "float64", **counts,
                                             "recorded_argmin_mismatches": "0.0"})
                l2 = rng.randint(0, 5_000)
                window_l2 = rng.randint(0, 3_000)
                avoided = l1 + l2
                if iteration == 0:
                    pi0_avoided = avoided
                utility.append({
                    **policy, "requested_tokens": str(requested),
                    "avoided_prefill_tokens": str(avoided), "l1_avoided_tokens": str(l1),
                    "l2_avoided_tokens": str(l2),
                    "label_window_requested_tokens": str(window),
                    "label_window_avoided_prefill_tokens": str(window_l1 + window_l2),
                    "label_window_l1_avoided_tokens": str(window_l1),
                    "label_window_l2_avoided_tokens": str(window_l2),
                    "input_token_points_vs_pi0": repr(100.0 * (avoided - pi0_avoided) / requested),
                })
        self.tables = {"populations": self.population_rows, "canonical": canonical,
                       "links": links, "utility": utility}
        self.write()

    def write(self) -> None:
        """(Re)write the four tables, the run manifest and the config."""
        for name, rows in self.tables.items():
            _write_rows(self.paths[name], rows)
        _write_rows(self.paths["manifest"], self.tables["canonical"])
        artifacts = {self.paths[name].name: {"sha256": _sha(self.paths[name])}
                     for name in ("populations", "canonical", "links", "utility")}
        self.paths["config"].write_text(json.dumps({"output_artifacts": artifacts}),
                                        encoding="utf-8")

    def population(self, lineage: tuple, iteration: int) -> DecisionPopulation:
        row = next(row for row in self.population_rows
                   if runner._key(row) == lineage + (iteration,))
        return DecisionPopulation.load(row["population_path"])

    @contextlib.contextmanager
    def patched(self, arguments: list[str]):
        """Git answers a clean tree, HEAD `0...0` and `1...1` as the last commit
        touching the plan (an addendum may follow the code's parent commit)."""
        def git(*command):
            if command[0] == "status":
                return ""
            return "1" * 40 if command[0] == "log" else "0" * 40

        with mock.patch.object(runner, "PUBLISHED_CONFIG", self.paths["config"]), \
                mock.patch.object(runner, "TEST_POPULATIONS", self.paths["populations"]), \
                mock.patch.object(runner, "CANONICAL_MODELS", self.paths["canonical"]), \
                mock.patch.object(runner, "MODEL_LINKS", self.paths["links"]), \
                mock.patch.object(runner, "SEED_UTILITY", self.paths["utility"]), \
                mock.patch.object(runner, "RUN_MODEL_MANIFEST", self.paths["manifest"]), \
                mock.patch.object(runner, "_git", side_effect=git), \
                mock.patch.object(runner, "sources_differing_from_head", return_value=[]), \
                mock.patch.object(sys, "argv", ["run_ranker_error_diagnosis.py", *arguments]):
            yield

    def main(self, target: Path, workers: int = 1) -> None:
        """`runner.main`; what it printed is kept in `self.printed`."""
        buffer = io.StringIO()
        try:
            with self.patched(["--output-dir", str(target), "--workers", str(workers)]), \
                    contextlib.redirect_stdout(buffer):
                runner.main()
        finally:
            self.printed = buffer.getvalue()


SMALL_GRID = {"TRACES": ("conversation_trace",), "CELLS": ((0.01, 4.0),),
              "TARGETS": ("next_use",), "SEEDS": (0, 1, 2, 3, 4)}


@contextlib.contextmanager
def small_grid():
    with contextlib.ExitStack() as stack:
        for name, value in SMALL_GRID.items():
            stack.enter_context(mock.patch.object(runner, name, value))
        yield


def brute_force_m3(population: DecisionPopulation, ranker) -> float:
    """m3 of a scorer on a population, without `errordiag`: per decision, the
    replay victim of `score_row` and the tie-break, and labels from the delta."""
    total, count = 0.0, 0
    for _, block in population.group_blocks():
        rows = [population.features[row].tolist() for row in block]
        victim = replay_victim([float(ranker.score_row(row)) for row in rows],
                               population.arm_tiebreak[block].tolist())
        labels = [-math.log1p(min(delta / 1000.0, H)) for delta in
                  population.next_use_delta_ms[block].tolist()]
        total += labels[victim] - min(labels)
        count += 1
    return total / count


class RunnerEndToEndTests(unittest.TestCase):
    """The full grid, 120 lineages and 240 small populations, run once with one
    worker and once with two."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.published = ConstructedRun(root / "published")
        cls.outputs = {}
        for workers in (1, 2):
            cls.outputs[workers] = root / f"out_{workers}"
            cls.published.main(cls.outputs[workers], workers=workers)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

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
        target = self.outputs[1]
        lineages, cells = 120, 24
        expected = {"checks": lineages * 2, "matrix_seeds": lineages * 2 * 3,
                    "selection_seeds": lineages, "selection": cells * 2, "selection_counts": 4,
                    "window": cells, "window_counts": 2, "classes_seeds": lineages * 3 * 4,
                    "classes": cells * 3 * 4, "rates_seeds": lineages * 2, "rates": cells * 2,
                    "rates_counts": 4, "concentration_seeds": lineages * 3,
                    "concentration": cells * 3}
        for name, count in expected.items():
            self.assertEqual(len(_read_rows(target / f"{name}.csv")), count, name)

    def test_one_worker_and_two_agree(self):
        for name in runner.TABLES:
            self.assertEqual((self.outputs[1] / f"{name}.csv").read_bytes(),
                             (self.outputs[2] / f"{name}.csv").read_bytes(), name)

    def test_checks_and_config(self):
        checks = _read_rows(self.outputs[1] / "checks.csv")
        self.assertTrue(all(row["sha256_matches"] == "True" for row in checks))
        self.assertTrue(all(row["own_victim_mismatches"] == "0" for row in checks))
        self.assertTrue(all(row["exceeds_stop_rule"] == "False" for row in checks))
        self.assertTrue(all(row["own_score_exact"] == "True" for row in checks))
        self.assertEqual({row["policy"] for row in checks}, {"pi0", "pi3"})
        config = json.loads((self.outputs[1] / "run_config.json").read_text())
        self.assertEqual(config["status"], "complete")
        # The plan's last commit is recorded, not required to be a fixed one.
        self.assertEqual(config["plan_commit"], "1" * 40)
        self.assertEqual(config["code_commit"], "0" * 40)
        self.assertNotIn("preregistration_commit", config)
        self.assertEqual(len(config["populations"]), 240)
        self.assertEqual(len(config["models"]), 4 + 120)
        self.assertEqual(len(config["published_inputs"]), 6)
        self.assertEqual(config["checks"]["utility_pairs_matched"], 120)
        self.assertEqual(config["checks"]["populations_matching_published_sha256"], 240)
        self.assertEqual(config["checks"]["own_victim_mismatches_total"], 0)
        self.assertEqual(config["workers"], 1)
        self.assertEqual(json.loads((self.outputs[2] / "run_config.json").read_text())["workers"], 2)

    def test_matrix_against_a_brute_force_computation(self):
        rows = _read_rows(self.outputs[1] / "matrix_seeds.csv")
        index = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                  row["target"], int(row["seed"]), row["population"], row["victim_source"]): row
                 for row in rows}
        for lineage in random.Random(1).sample(runner.lineages(), 6):
            for population_iteration in runner.ITERATIONS:
                population = self.published.population(lineage, population_iteration)
                for scorer in runner.ITERATIONS:
                    row = index[lineage + (f"pi{population_iteration}", f"scorer_pi{scorer}")]
                    self.assertAlmostEqual(float(row["m3"]),
                                           brute_force_m3(population, self.published.models[lineage + (scorer,)]),
                                           places=12)
                own = index[lineage + (f"pi{population_iteration}", f"scorer_pi{population_iteration}")]
                logged = index[lineage + (f"pi{population_iteration}", "logged")]
                self.assertEqual(own["m3"], logged["m3"])
                self.assertEqual(own["m4"], logged["m4"])

    def test_utilities_are_the_published_ones(self):
        rows = _read_rows(self.outputs[1] / "selection_seeds.csv")
        utility = {runner._key(row): row for row in self.published.tables["utility"]}
        for row in rows:
            lineage = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                       row["target"], int(row["seed"]))
            first, last = utility[lineage + (0,)], utility[lineage + (3,)]
            self.assertEqual(float(row["delta_u_points"]), float(last["input_token_points_vs_pi0"]))
            window = int(first["label_window_requested_tokens"])
            extra = [int(r["label_window_avoided_prefill_tokens"]) - int(r["label_window_l1_avoided_tokens"])
                     for r in (first, last)]
            self.assertAlmostEqual(float(row["delta_u_label_points"]),
                                   100.0 * (extra[1] - extra[0]) / window, places=12)

    def test_counts_cover_every_cell(self):
        for name, fields in (("selection_counts", ("carried_by_selection", "population_dependent",
                                                    "not_carried_by_selection")),
                             ("rates_counts", ("better_than_recency", "worse_than_recency",
                                               "mixed_vs_recency"))):
            for row in _read_rows(self.outputs[1] / f"{name}.csv"):
                self.assertEqual(row["cells"], "12")
                self.assertEqual(sum(int(row[field]) for field in fields), 12)
        for row in _read_rows(self.outputs[1] / "window_counts.csv"):
            self.assertEqual(row["cells"], "12")
            listed = [entry for entry in row["contradiction_list"].split("; ") if entry]
            self.assertEqual(len(listed), int(row["contradictions"]))

    def test_class_shares_partition_each_population(self):
        rows = _read_rows(self.outputs[1] / "classes_seeds.csv")
        groups = {}
        for row in rows:
            key = (row["trace"], row["cell"], row["target"], row["seed"], row["subset"])
            groups.setdefault(key, []).append(row)
        for members in groups.values():
            self.assertEqual(len(members), 4)
            self.assertEqual(sum(int(row["decisions"]) for row in members),
                             int(members[0]["subset_decisions"]))

    def test_concentration_subsets_are_the_avoidable_class_by_subset(self):
        classes = {(row["trace"], row["cell"], row["target"], row["seed"], row["subset"]): row
                   for row in _read_rows(self.outputs[1] / "classes_seeds.csv")
                   if row["class"] == "avoidable_reusable"}
        rows = _read_rows(self.outputs[1] / "concentration_seeds.csv")
        evictions = {}
        for row in rows:
            key = (row["trace"], row["cell"], row["target"], row["seed"], row["subset"])
            self.assertEqual(row["evictions"], classes[key]["decisions"])
            evictions[key] = int(row["evictions"])
        for key, total in evictions.items():
            if key[4] == "all":
                self.assertEqual(evictions[key[:4] + ("arrival",)]
                                 + evictions[key[:4] + ("resident",)], total)
        summary = _read_rows(self.outputs[1] / "concentration.csv")
        self.assertEqual({row["subset"] for row in summary}, set(errordiag.SUBSETS))


class RunnerStopAndCheckTests(unittest.TestCase):
    """A small grid (one trace x cell x target, five seeds) for the failure paths."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(small_grid())
        self.root = Path(temporary.name)
        self.published = ConstructedRun(self.root / "published", decisions=12)
        self.target = self.root / "out"

    def _resave(self, lineage, iteration, population, update_hash=True):
        row = next(row for row in self.published.population_rows
                   if runner._key(row) == lineage + (iteration,))
        digest = population.save(row["population_path"])
        if update_hash:
            row["population_sha256"] = digest
            self.published.write()

    def test_the_small_grid_completes(self):
        self.published.main(self.target)
        self.assertEqual(json.loads((self.target / "run_config.json").read_text())["status"],
                         "complete")
        self.assertEqual(len(_read_rows(self.target / "selection.csv")), 2)

    def test_the_stop_rule_writes_only_the_checks(self):
        lineage = runner.lineages()[2]
        population = self.published.population(lineage, 3)
        block = next(rows for _, rows in population.group_blocks() if len(rows) >= 2)
        victim = population.victim.copy()
        position = int(np.flatnonzero(victim[block])[0])
        victim[block] = 0
        victim[block[(position + 1) % len(block)]] = 1
        population.victim = victim
        self._resave(lineage, 3, population)
        with self.assertRaises(SystemExit) as raised:
            self.published.main(self.target)
        self.assertIn("unresolved", str(raised.exception))
        self.assertEqual({path.name for path in self.target.iterdir()},
                         {"checks.csv", "README.md", "run_config.json"})
        checks = _read_rows(self.target / "checks.csv")
        flagged = [(row["seed"], row["policy"]) for row in checks if row["exceeds_stop_rule"] == "True"]
        self.assertEqual(flagged, [(str(lineage[4]), "pi3")])
        flagged_row = next(row for row in checks if row["exceeds_stop_rule"] == "True")
        self.assertEqual(flagged_row["own_victim_mismatches"], "1")
        config = json.loads((self.target / "run_config.json").read_text())
        self.assertEqual(config["status"], "unresolved")
        self.assertEqual(config["checks"]["populations_exceeding_stop_rule"], 1)
        self.assertIn("UNRESOLVED", (self.target / "README.md").read_text())

    def _assert_refused(self, fragment: str, problems: int = 1):
        """The run exits before writing anything, naming `fragment` in its
        CHECK lines, with exactly `problems` failed checks."""
        with self.assertRaises(SystemExit) as raised:
            self.published.main(self.target)
        self.assertIn(f"{problems} checks failed", str(raised.exception))
        lines = [line for line in self.published.printed.splitlines() if "CHECK" in line]
        self.assertEqual(len(lines), problems, lines)
        self.assertTrue(any(fragment in line for line in lines), lines)
        self.assertFalse(self.target.exists())

    def test_a_changed_population_is_refused(self):
        lineage = runner.lineages()[0]
        population = self.published.population(lineage, 0)
        population.decisions_offered += 1
        self._resave(lineage, 0, population, update_hash=False)
        self._assert_refused("population SHA-256")

    def test_a_missing_population_file_is_refused(self):
        Path(self.published.population_rows[0]["population_path"]).unlink()
        self._assert_refused("pi0.npz missing")

    def test_a_missing_utility_is_refused(self):
        lineage = runner.lineages()[1]
        self.published.tables["utility"] = [row for row in self.published.tables["utility"]
                                            if runner._key(row) != lineage + (3,)]
        self.published.write()
        self._assert_refused(f"seed utility: missing {runner._name(lineage + (3,))}")

    def test_a_utility_identity_failure_is_refused(self):
        row = next(row for row in self.published.tables["utility"] if row["iteration"] == "3")
        row["label_window_requested_tokens"] = str(int(row["label_window_requested_tokens"]) + 1)
        self.published.write()
        self._assert_refused("label_window_requested_tokens differs between pi0 and pi3")

    def test_a_changed_model_is_refused(self):
        row = next(row for row in self.published.tables["canonical"] if row["iteration"] == "3")
        path = Path(row["model_path"])
        changed = path.read_bytes().replace(b'"l2":1.0', b'"l2":2.0')
        self.assertNotEqual(changed, path.read_bytes())
        path.write_bytes(changed)
        self._assert_refused("ranker hash mismatch")

    def test_a_table_unlike_its_recorded_hash_is_refused(self):
        config = json.loads(self.published.paths["config"].read_text())
        config["output_artifacts"]["onpolicy_seed_utility.csv"]["sha256"] = "0" * 64
        self.published.paths["config"].write_text(json.dumps(config))
        self._assert_refused("onpolicy_seed_utility.csv: SHA-256")

    def test_a_link_naming_another_model_is_refused(self):
        rows = self.published.tables["links"]
        pi3 = [row for row in rows if row["iteration"] == "3"]
        pi3[0]["model_path"], pi3[1]["model_path"] = pi3[1]["model_path"], pi3[0]["model_path"]
        self.published.write()
        self._assert_refused("model links:", problems=2)

    def test_a_run_manifest_unlike_the_canonical_table_is_refused(self):
        rows = self.published.tables["canonical"][:-1]
        _write_rows(self.published.paths["manifest"], rows)
        self._assert_refused("the run's model manifest differs")


class RunnerRefusalTests(unittest.TestCase):
    def _main(self, arguments):
        with mock.patch.object(sys, "argv", ["run_ranker_error_diagnosis.py", *arguments]), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def test_workers_above_the_cap_or_below_one_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "new")
            for workers, fragment in (("13", "cap"), ("0", "positive")):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", target, "--workers", workers])
                self.assertIn(fragment, str(raised.exception))
        self.assertEqual(runner.MAX_WORKERS, 12)
        self.assertEqual(runner.DEFAULT_WORKERS, 10)

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                self._main(["--output-dir", directory])
            self.assertIn("exists", str(raised.exception))

    def test_an_unclean_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new"
            with mock.patch.object(runner, "_git", return_value="?? scripts/new.py"):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", str(target)])
            self.assertIn("not clean", str(raised.exception))
            with mock.patch.object(runner, "_git", return_value=""), \
                    mock.patch.object(runner, "sources_differing_from_head",
                                      return_value=["src/persistent_kv_admission/errordiag.py"]):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["--output-dir", str(target)])
            self.assertIn("differs from HEAD", str(raised.exception))
            self.assertFalse(target.exists())

    def test_an_undocumented_column_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                runner._write(Path(directory) / "x.csv", [{"target": "next_use"}],
                              "selection_counts")


if __name__ == "__main__":
    unittest.main()
