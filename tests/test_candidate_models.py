"""Tests for the candidate-population model check."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.candidate_models import (
    Dataset,
    FeatureLogger,
    _within_decision_ranks,
    build_dataset,
    log_candidate_features,
    within_decision_metrics,
)
from persistent_kv_admission.crossworkload import fit_fixed_model
from persistent_kv_admission.temporal import FEATURE_NAMES
from persistent_kv_admission.trace import load_mooncake_trace


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def records(steps=200):
    out = []
    for step in range(steps):
        chain = [1, 2] if step % 4 == 0 else [1, 100 + step]
        out.append({"timestamp": step * 1000, "input_length": 1024, "output_length": 1, "hash_ids": chain})
    return out


class FeatureLoggingTests(unittest.TestCase):
    def test_logger_keeps_features_labels_and_a_bounded_reservoir(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        ranker, _, split_ms, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500)
        logger = log_candidate_features(trace, ranker, 0.05, max_decisions=25, bytes_per_token=1, sample_width=4)
        self.assertGreater(logger.decisions_seen, 25)
        self.assertEqual(len(logger.kept), 25)
        record = logger.kept[0]
        self.assertEqual(record.features.shape, (len(record.candidates), len(FEATURE_NAMES)))
        self.assertEqual(len(record.next_use_delta_ms), len(record.candidates))
        self.assertTrue(all(b > 0 for b in record.state_bytes))

    def test_dataset_split_respects_window_and_embargo(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        ranker, _, split_ms, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500)
        logger = log_candidate_features(trace, ranker, 0.05, max_decisions=10_000, bytes_per_token=1, sample_width=4)
        horizon = 10
        train = build_dataset(logger, horizon, 0.0, split_ms)
        test = build_dataset(logger, horizon, split_ms, trace.end_ms)
        self.assertTrue(np.all(train.timestamp_ms + horizon * 1000.0 <= split_ms))
        self.assertTrue(np.all(test.timestamp_ms >= split_ms))
        self.assertTrue(np.all(test.timestamp_ms + horizon * 1000.0 <= trace.end_ms))
        self.assertEqual(int(test.is_victim.sum()), test.decisions)


class MetricTests(unittest.TestCase):
    def _dataset(self):
        # Two decisions of three candidates each; feature 0 is informative.
        features = np.array([[3.0, 0], [1.0, 0], [2.0, 0], [0.5, 0], [0.7, 0], [0.9, 0]])
        labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
        decision = np.array([0, 0, 0, 1, 1, 1])
        return Dataset(features, labels, decision, np.zeros(6), np.zeros(6), np.zeros(6, np.int8), 2)

    def test_within_decision_metrics_are_computed_per_decision(self):
        data = self._dataset()
        perfect = within_decision_metrics(data.features[:, 0], data)
        self.assertAlmostEqual(perfect["within_decision_auc_micro"], 1.0)
        self.assertAlmostEqual(perfect["regret_when_avoidable"], 0.0)
        inverted = within_decision_metrics(-data.features[:, 0], data)
        self.assertAlmostEqual(inverted["within_decision_auc_micro"], 0.0)
        self.assertAlmostEqual(inverted["regret_when_avoidable"], 1.0)
        self.assertEqual(perfect["decisions"], 2)

    def test_within_decision_ranks_are_fractional_per_decision(self):
        data = self._dataset()
        ranks = _within_decision_ranks(data)
        self.assertEqual(ranks.shape, data.features.shape)
        self.assertEqual(sorted(ranks[:3, 0].tolist()), [0.0, 0.5, 1.0])
        self.assertEqual(sorted(ranks[3:, 0].tolist()), [0.0, 0.5, 1.0])
        self.assertTrue(np.all(ranks[:, 1] == ranks[0, 1]))


class ModelTests(unittest.TestCase):
    def test_models_score_the_test_set_when_sklearn_is_available(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        from persistent_kv_admission.candidate_models import fit_and_score_models

        temporary, trace = build_trace(records(400))
        self.addCleanup(temporary.cleanup)
        ranker, _, split_ms, _, _ = fit_fixed_model(trace, 10.0, snapshot_count=8, candidate_cap=500)
        logger = log_candidate_features(trace, ranker, 0.05, max_decisions=10_000, bytes_per_token=1, sample_width=4)
        train = build_dataset(logger, 10, 0.0, split_ms)
        test = build_dataset(logger, 10, split_ms, trace.end_ms)
        scores = fit_and_score_models(train, test)
        for name in ("recency", "frequency", "arm", "linear", "gbm", "gbm_plus_context"):
            self.assertIn(name, scores)
            self.assertEqual(len(scores[name]), len(test.labels))
        metrics = within_decision_metrics(scores["gbm"], test)
        self.assertTrue(math.isfinite(metrics["regret_when_avoidable"]))


if __name__ == "__main__":
    unittest.main()
