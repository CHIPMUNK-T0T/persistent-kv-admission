"""Tests for the target-change machinery: graded targets, ridge ranker, matched horizon."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.crossworkload import fit_fixed_model, target_values
from persistent_kv_admission.gap import HorizonMatchedScorer, run_arm, target_arm_specs
from persistent_kv_admission.predictors import LogisticRanker, RidgeRanker
from persistent_kv_admission.replay import _occurrence_groups, replay
from persistent_kv_admission.temporal import FEATURE_NAMES, model_indices
from persistent_kv_admission.trace import load_mooncake_trace


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def records(steps=300):
    out = []
    for step in range(steps):
        chain = [1, 2] if step % 4 == 0 else [1, 100 + step]
        out.append({"timestamp": step * 1000, "input_length": 1024, "output_length": 1, "hash_ids": chain})
    return out


class TargetValueTests(unittest.TestCase):
    def test_targets_have_the_documented_semantics(self):
        temporary, trace = build_trace(records(40))
        self.addCleanup(temporary.cleanup)
        ids = ["int:2", "int:101"]
        binary = target_values(trace, ids, 0.0, 10, "binary")
        count = target_values(trace, ids, 0.0, 10, "count")
        next_use = target_values(trace, ids, 0.0, 10, "next_use")
        # int:2 recurs at 4, 8 s (two uses within 10 s); int:101 is used once at 1 s.
        self.assertEqual(binary.tolist(), [1.0, 1.0])
        self.assertAlmostEqual(count[0], math.log1p(2))
        self.assertAlmostEqual(count[1], math.log1p(1))
        self.assertAlmostEqual(next_use[0], -math.log1p(4.0))
        self.assertAlmostEqual(next_use[1], -math.log1p(1.0))
        # A state never used again sits at the floor.
        never = target_values(trace, ["int:101"], 2_000.0, 10, "next_use")
        self.assertAlmostEqual(never[0], -math.log1p(10.0))
        self.assertEqual(target_values(trace, ["int:101"], 2_000.0, 10, "count")[0], 0.0)
        with self.assertRaises(ValueError):
            target_values(trace, ids, 0.0, 10, "bogus")


class RidgeRankerTests(unittest.TestCase):
    def test_ridge_recovers_a_linear_target_and_scores_rows_consistently(self):
        generator = np.random.default_rng(2)
        features = generator.normal(size=(2000, len(FEATURE_NAMES))) * 2.0 + 1.0
        weights = np.zeros(len(FEATURE_NAMES))
        weights[0], weights[3] = 1.5, -0.7
        targets = features @ weights + 0.1 * generator.normal(size=2000)
        ranker = RidgeRanker(indices=tuple(range(len(FEATURE_NAMES))), l2=1e-4).fit(features, targets)
        predictions = ranker.score(features)
        self.assertGreater(np.corrcoef(predictions, targets)[0, 1], 0.99)
        for index in range(10):
            self.assertAlmostEqual(ranker.score_row(list(features[index])), predictions[index], places=9)
        self.assertEqual(len(ranker.standardized_coefficients(FEATURE_NAMES)), len(FEATURE_NAMES))

    def test_fit_fixed_model_returns_the_right_ranker_per_target(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        binary, _, _, positives, total = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500, target="binary")
        count, _, _, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500, target="count")
        next_use, _, _, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500, target="next_use")
        self.assertIsInstance(binary, LogisticRanker)
        self.assertIsInstance(count, RidgeRanker)
        self.assertIsInstance(next_use, RidgeRanker)
        self.assertGreater(positives, 0)
        self.assertLessEqual(positives, total)
        with self.assertRaises(ValueError):
            fit_fixed_model(trace, 10.0, snapshot_count=8, target="bogus")


class MatchedHorizonTests(unittest.TestCase):
    def test_matched_scorer_tracks_residence_time_and_replays(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        rankers = {}
        # Requested horizons are clipped to the usable set (10, 60, ...), so
        # ask for the two smallest usable ones directly.
        for horizon in (10, 60):
            ranker, used, _, _, _ = fit_fixed_model(trace, float(horizon), snapshot_count=8, candidate_cap=500)
            rankers[float(used)] = ranker
        self.assertEqual(set(rankers), {10.0, 60.0})
        groups = _occurrence_groups(trace)
        scorer = HorizonMatchedScorer(trace, rankers, bytes_per_token=1)
        result = replay(trace, "matched", 3 * 512, 0.1, 1, occurrence_groups=groups,
                        scorer=scorer, eviction="sampled", sample_width=4)
        self.assertGreaterEqual(result.avoided_prefill_tokens, 0)
        residence = scorer.residence_seconds(trace.end_ms)
        self.assertTrue(math.isfinite(residence))
        # Three blocks of capacity against roughly one new block per second:
        # residence is a few seconds, nearer 10 s than 60 s in log space.
        self.assertLess(residence, 10.0)
        self.assertGreater(scorer.horizon_groups[10.0], scorer.horizon_groups[60.0])
        with self.assertRaises(ValueError):
            HorizonMatchedScorer(trace, {})

    def test_target_arm_specs_and_run_arm_with_named_rankers(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        ranker, used, split_ms, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500)
        rankers = {f"binary_h{used}": ranker, "count_h10": ranker}
        names = [spec.name for spec in target_arm_specs(rankers)]
        self.assertIn(f"learned_binary_h{used}", names)
        self.assertIn("learned_count_h10", names)
        self.assertNotIn("learned_matched", names)  # one binary horizon only
        spec = next(s for s in target_arm_specs(rankers) if s.name == "learned_count_h10")
        out = run_arm(trace, spec, 0.1, 0, None, split_ms, bytes_per_token=1, sample_width=4, rankers=rankers)
        self.assertEqual(out["replay"]["policy"], "learned_count_h10")
        with self.assertRaises(ValueError):
            run_arm(trace, spec, 0.1, 0, None, split_ms, bytes_per_token=1, sample_width=4, rankers={})


if __name__ == "__main__":
    unittest.main()
