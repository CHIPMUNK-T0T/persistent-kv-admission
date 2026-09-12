"""Tests for the predictability–retention gap machinery.

Covers the oracle scorers, the eviction decision logger, the population ladder,
the multi-seed aggregation, and the standardisation variants of the ranker.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.crossworkload import FixedModelScorer, fit_fixed_model
from persistent_kv_admission.gap import (
    ArmSpec,
    DecisionLogger,
    OracleScorer,
    PopulationLadder,
    add_closure,
    aggregate_replay,
    arm_specs,
    coefficient_cosines,
    decomposition_rows,
    run_arm,
    summarize,
)
from persistent_kv_admission.phase05 import _labels
from persistent_kv_admission.predictors import LogisticRanker
from persistent_kv_admission.replay import _occurrence_groups, replay
from persistent_kv_admission.temporal import FEATURE_NAMES, model_indices
from persistent_kv_admission.trace import load_mooncake_trace


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def branching_records(steps=60):
    """Two hot roots, one of which is reused every 2 s, plus one-off leaves."""
    records = []
    for step in range(steps):
        hot = [1, 2] if step % 2 == 0 else [1, 3]
        records.append(
            {"timestamp": step * 1000, "input_length": 1536, "output_length": 1,
             "hash_ids": hot + [100 + step]}
        )
    return records


class OracleScorerTests(unittest.TestCase):
    def test_binary_oracle_matches_protocol_labels(self):
        temporary, trace = build_trace(branching_records())
        self.addCleanup(temporary.cleanup)
        oracle = OracleScorer(trace, "oracle_binary", horizon_seconds=3.0)
        state_ids = list(trace.states)
        for now_ms in (0.0, 5_000.0, 17_500.0, 58_000.0, 59_000.0):
            expected = _labels(trace, state_ids, now_ms, 3)
            got = [oracle.score(state_id, now_ms) for state_id in state_ids]
            self.assertEqual([int(v) for v in got], expected.tolist(), msg=f"at {now_ms}")

    def test_next_use_and_count_oracles(self):
        temporary, trace = build_trace(branching_records())
        self.addCleanup(temporary.cleanup)
        next_use = OracleScorer(trace, "oracle_next_use", horizon_seconds=10.0)
        count = OracleScorer(trace, "oracle_count", horizon_seconds=10.0)
        # int:2 appears at even seconds, int:3 at odd seconds; from t=0 the next
        # use of int:2 is 2 s away and of int:3 is 1 s away.
        self.assertAlmostEqual(next_use.score("int:2", 0.0), -2000.0)
        self.assertAlmostEqual(next_use.score("int:3", 0.0), -1000.0)
        self.assertEqual(next_use.score("int:100", 0.0), -math.inf)
        # Within 10 s after t=0, int:2 is used at 2,4,6,8,10 and int:3 at 1,3,5,7,9.
        self.assertEqual(count.score("int:2", 0.0), 5.0)
        self.assertEqual(count.score("int:3", 0.0), 5.0)
        self.assertEqual(count.score("int:100", 0.0), 0.0)

    def test_per_byte_oracle_prefers_the_smaller_positive(self):
        records = [
            {"timestamp": 0, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]},
            {"timestamp": 0, "input_length": 600, "output_length": 1, "hash_ids": [1, 3]},
            {"timestamp": 5000, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]},
            {"timestamp": 5000, "input_length": 600, "output_length": 1, "hash_ids": [1, 3]},
        ]
        temporary, trace = build_trace(records)
        self.addCleanup(temporary.cleanup)
        oracle = OracleScorer(trace, "oracle_binary_per_byte", horizon_seconds=10.0, bytes_per_token=1)
        # Both are reused; int:3 holds 88 tokens, int:2 holds 512, so int:3
        # scores higher per byte and int:2 is the one a per-byte rule evicts.
        self.assertGreater(oracle.score("int:3", 0.0), oracle.score("int:2", 0.0))
        self.assertEqual(oracle.score("int:2", 6000.0), 0.0)


def leaf_reuse_records(steps=60):
    """A reused two-block chain whose tip is a leaf, interleaved with one-offs.

    The tip `int:2` is reused every 4 s and sits among the evictable leaves, so
    eviction decisions contain both a positive and negative candidates. With a
    three-block budget LRU evicts the tip before its reuse; a 3 s binary oracle
    does not.
    """
    records = []
    for step in range(steps):
        chain = [1, 2] if step % 4 == 0 else [1, 100 + step]
        records.append(
            {"timestamp": step * 1000, "input_length": 1024, "output_length": 1,
             "hash_ids": chain}
        )
    return records


class ReplayWithOraclesTests(unittest.TestCase):
    def test_binary_oracle_never_evicts_a_positive_when_a_negative_is_available(self):
        temporary, trace = build_trace(leaf_reuse_records())
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        logger = DecisionLogger(trace, measure_from_ms=None, max_decisions=10_000, seed=0)
        replay(
            trace, "oracle_binary", 3 * 512, 0.1, 1,
            occurrence_groups=groups,
            scorer=OracleScorer(trace, "oracle_binary", horizon_seconds=3.0),
            eviction="sampled", sample_width=16, decision_hook=logger,
        )
        rows = {row["horizon_seconds"]: row for row in logger.metrics((3,), precision_k=10)}
        self.assertGreater(rows[3]["decisions"], 0)
        self.assertGreater(rows[3]["decisions_with_both_classes"], 0.0)
        self.assertEqual(rows[3]["victim_positive_rate_when_avoidable"], 0.0)
        self.assertAlmostEqual(rows[3]["within_decision_auc_micro"], 1.0)

    def test_lru_evicts_the_reused_leaf_that_the_binary_oracle_protects(self):
        temporary, trace = build_trace(leaf_reuse_records())
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        lru = replay(trace, "lru", 3 * 512, 0.1, 1, occurrence_groups=groups,
                     eviction="sampled", sample_width=16)
        oracle = replay(
            trace, "oracle_binary", 3 * 512, 0.1, 1, occurrence_groups=groups,
            scorer=OracleScorer(trace, "oracle_binary", horizon_seconds=3.0),
            eviction="sampled", sample_width=16,
        )
        self.assertGreater(oracle.avoided_prefill_tokens, lru.avoided_prefill_tokens)

    def test_sampled_next_use_oracle_equals_heap_comparator_when_every_leaf_is_a_candidate(self):
        temporary, trace = build_trace(branching_records())
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        heap = replay(trace, "offline_next_use", 6 * 512, 0.1, 1, occurrence_groups=groups)
        sampled = replay(
            trace, "offline_next_use", 6 * 512, 0.1, 1,
            occurrence_groups=groups, eviction="sampled", sample_width=10_000,
        )
        self.assertEqual(heap.avoided_prefill_tokens, sampled.avoided_prefill_tokens)

    def test_seeded_sampled_replay_is_repeatable(self):
        temporary, trace = build_trace(branching_records(120))
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        results = [
            replay(
                trace, "offline_next_use", 4 * 512, 0.1, 1,
                occurrence_groups=groups, eviction="sampled", sample_width=2, seed=7,
            ).avoided_prefill_tokens
            for _ in range(3)
        ]
        self.assertEqual(len(set(results)), 1)

    def test_run_arm_returns_replay_candidate_and_ladder_rows(self):
        temporary, trace = build_trace(branching_records(120))
        self.addCleanup(temporary.cleanup)
        spec = ArmSpec("oracle_binary", "sampled", scorer="oracle:oracle_binary", horizon_seconds=3.0)
        out = run_arm(
            trace, spec, 0.1, seed=0, ranker=None, split_ms=60_000.0,
            log_horizons=(3,), ladder_snapshots=None, bytes_per_token=1, sample_width=4,
        )
        self.assertEqual(out["replay"]["policy"], "oracle_binary")
        self.assertEqual(out["replay"]["seed"], 0)
        self.assertTrue(out["candidate"])
        self.assertEqual(out["ladder"], [])
        self.assertEqual(out["candidate"][0]["horizon_seconds"], 3)

    def test_arm_specs_use_the_fit_horizon_and_add_extra_horizons(self):
        names = [spec.name for spec in arm_specs(600, extra_horizons=(60, 300, 600))]
        self.assertIn("oracle_binary_h60", names)
        self.assertIn("oracle_binary_h300", names)
        self.assertNotIn("oracle_binary_h600", names)
        primary = next(spec for spec in arm_specs(600) if spec.name == "oracle_binary")
        self.assertEqual(primary.horizon_seconds, 600)


class DecisionLoggerTests(unittest.TestCase):
    def test_reservoir_is_bounded_and_counts_every_decision(self):
        temporary, trace = build_trace(branching_records(10))
        self.addCleanup(temporary.cleanup)
        logger = DecisionLogger(trace, measure_from_ms=None, max_decisions=5, seed=1)
        candidates = ["int:1", "int:2"]
        for step in range(50):
            logger(candidates, [(0.0, 0.0), (1.0, 0.0)], 0, float(step * 1000), step)
        self.assertEqual(logger.decisions_seen, 50)
        self.assertEqual(len(logger.kept), 5)

    def test_decisions_before_the_measurement_window_are_ignored(self):
        temporary, trace = build_trace(branching_records(10))
        self.addCleanup(temporary.cleanup)
        logger = DecisionLogger(trace, measure_from_ms=5_000.0, max_decisions=100, seed=1)
        logger(["int:1"], [(0.0, 0.0)], 0, 1_000.0, 1)
        logger(["int:1"], [(0.0, 0.0)], 0, 6_000.0, 6)
        self.assertEqual(logger.decisions_seen, 1)

    def test_censored_decisions_are_excluded_per_horizon(self):
        temporary, trace = build_trace(branching_records(10))  # ends at 9 s
        self.addCleanup(temporary.cleanup)
        logger = DecisionLogger(trace, measure_from_ms=None, max_decisions=100, seed=1)
        logger(["int:2", "int:100"], [(1.0, 0.0), (0.0, 0.0)], 1, 0.0, 0)
        logger(["int:2", "int:100"], [(1.0, 0.0), (0.0, 0.0)], 1, 8_000.0, 8)
        rows = {row["horizon_seconds"]: row for row in logger.metrics((1, 5), precision_k=10)}
        self.assertEqual(rows[1]["decisions"], 2)
        self.assertEqual(rows[5]["decisions"], 1)

    def test_candidate_rows_carry_next_use_and_bytes(self):
        temporary, trace = build_trace(branching_records(10))
        self.addCleanup(temporary.cleanup)
        logger = DecisionLogger(trace, measure_from_ms=None, max_decisions=100, seed=1)
        logger(["int:2", "int:100"], [(1.0, 0.0), (0.0, 0.0)], 1, 0.0, 0)
        rows = logger.rows(bytes_per_token=1)
        by_state = {row["state_id"]: row for row in rows}
        self.assertEqual(by_state["int:2"]["next_use_delta_ms"], 2000.0)
        self.assertEqual(by_state["int:100"]["next_use_delta_ms"], math.inf)
        self.assertEqual(by_state["int:100"]["is_victim"], 1)
        self.assertEqual(by_state["int:2"]["state_bytes"], 512)


class PopulationLadderTests(unittest.TestCase):
    def test_ladder_records_nested_populations(self):
        temporary, trace = build_trace(branching_records(120))
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        ladder = PopulationLadder(trace, {60, 80, 100}, (3,), precision_k=10)
        replay(
            trace, "lfu", 6 * 512, 0.1, 1, occurrence_groups=groups,
            eviction="heap", group_hook=ladder,
        )
        rows = {row["population"]: row for row in ladder.rows()}
        self.assertEqual(set(rows), {"observed", "cached", "leaves"})
        self.assertEqual(rows["observed"]["snapshots"], 3)
        self.assertGreater(rows["observed"]["population_size"], rows["cached"]["population_size"])
        self.assertGreaterEqual(rows["cached"]["population_size"], rows["leaves"]["population_size"])


class AggregationTests(unittest.TestCase):
    def test_summarize_reports_t_based_interval(self):
        stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["std"], math.sqrt(2.5))
        self.assertAlmostEqual(stats["ci95_half"], 2.776 * math.sqrt(2.5) / math.sqrt(5), places=6)
        self.assertEqual(summarize([])["n"], 0)
        self.assertTrue(math.isnan(summarize([1.0])["ci95_half"]))

    def _rows(self):
        rows = []
        for seed in (0, 1):
            lru = 100 + seed
            rows.append({"trace": "t", "policy": "lru", "eviction": "sampled", "capacity_fraction": 0.01,
                         "seed": seed, "avoided_prefill_tokens": lru, "block_hit_rate": 0.1})
            rows.append({"trace": "t", "policy": "learned_history", "eviction": "sampled", "capacity_fraction": 0.01,
                         "seed": seed, "avoided_prefill_tokens": lru + 20, "block_hit_rate": 0.1})
            rows.append({"trace": "t", "policy": "oracle_binary", "eviction": "sampled", "capacity_fraction": 0.01,
                         "seed": seed, "avoided_prefill_tokens": lru + 50, "block_hit_rate": 0.1})
            rows.append({"trace": "t", "policy": "oracle_next_use_sampled", "eviction": "sampled", "capacity_fraction": 0.01,
                         "seed": seed, "avoided_prefill_tokens": lru + 90, "block_hit_rate": 0.1})
            rows.append({"trace": "t", "policy": "offline_next_use", "eviction": "heap", "capacity_fraction": 0.01,
                         "seed": seed, "avoided_prefill_tokens": 200, "block_hit_rate": 0.1})
        return rows

    def test_closure_uses_the_same_seed_lru_and_the_heap_comparator(self):
        rows = self._rows()
        add_closure(rows)
        by = {(r["policy"], r["seed"]): r for r in rows}
        self.assertAlmostEqual(by[("oracle_binary", 0)]["headroom_closure"], 0.5)
        self.assertAlmostEqual(by[("oracle_binary", 1)]["headroom_closure"], 50 / 99)
        self.assertAlmostEqual(by[("offline_next_use", 0)]["headroom_closure"], 1.0)
        summary = {r["policy"]: r for r in aggregate_replay(rows)}
        self.assertEqual(summary["oracle_binary"]["seeds"], 2)
        self.assertAlmostEqual(summary["lru"]["headroom_closure_mean"], 0.0)

    def test_decomposition_is_seed_paired(self):
        rows = self._rows()
        add_closure(rows)
        entry = decomposition_rows(rows)[0]
        self.assertAlmostEqual(entry["signal_gap_closure_mean"], 0.5 * (30 / 100 + 30 / 99))
        self.assertAlmostEqual(entry["objective_gap_next_use_closure_mean"], 0.5 * (40 / 100 + 40 / 99))
        self.assertAlmostEqual(entry["candidate_search_gap_closure_mean"], 0.5 * (10 / 100 + 9 / 99))
        self.assertAlmostEqual(entry["signal_gap_tokens_mean"], 30.0)


class StandardizationVariantTests(unittest.TestCase):
    def test_unstandardized_ranker_scores_consistently(self):
        generator = np.random.default_rng(3)
        features = generator.normal(size=(400, len(FEATURE_NAMES))) * 3.0 + 1.0
        labels = (generator.uniform(size=400) < 0.3).astype(float)
        ranker = LogisticRanker(indices=model_indices("base0"), l2=1e-2, standardize=False).fit(features, labels)
        self.assertTrue(np.allclose(ranker.mean, 0.0))
        self.assertTrue(np.allclose(ranker.scale, 1.0))
        batch = ranker.score(features)
        for index in range(10):
            self.assertAlmostEqual(ranker.score_row(list(features[index])), batch[index], places=9)

    def test_fit_fixed_model_accepts_every_standardization(self):
        temporary, trace = build_trace(branching_records(200))
        self.addCleanup(temporary.cleanup)
        rankers = {}
        for variant in ("per_decision", "train_global", "none"):
            ranker, horizon, _, positives, total = fit_fixed_model(
                trace, 10.0, snapshot_count=8, standardization=variant, candidate_cap=500,
            )
            self.assertGreater(positives, 0)
            self.assertLessEqual(positives, total)
            rankers[variant] = ranker
        self.assertTrue(rankers["per_decision"].standardize)
        self.assertFalse(rankers["none"].standardize)
        with self.assertRaises(ValueError):
            fit_fixed_model(trace, 10.0, snapshot_count=8, standardization="bogus")
        cosines = coefficient_cosines({"a": rankers["per_decision"], "b": rankers["train_global"]})
        self.assertEqual(len(cosines), 1)
        self.assertLessEqual(abs(cosines[0]["cosine"]), 1.0 + 1e-9)

    def test_fixed_model_scorer_normalisation_population_is_selectable(self):
        temporary, trace = build_trace(branching_records(200))
        self.addCleanup(temporary.cleanup)
        ranker, _, _, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500)
        groups = _occurrence_groups(trace)
        for population in ("cached", "observed"):
            result = replay(
                trace, "learned", 4 * 512, 0.1, 1, occurrence_groups=groups,
                scorer=FixedModelScorer(trace, ranker, normalize_on=population),
                eviction="sampled", sample_width=4,
            )
            self.assertGreaterEqual(result.avoided_prefill_tokens, 0)
        with self.assertRaises(ValueError):
            FixedModelScorer(trace, ranker, normalize_on="bogus")


if __name__ == "__main__":
    unittest.main()
