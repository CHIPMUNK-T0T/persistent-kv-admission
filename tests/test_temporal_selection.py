"""Tests for temporal features, online adaptation, and the scoring hook."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.online import OnlineAdaptiveScorer
from persistent_kv_admission.phase05 import fast_ranking_metrics, split_design
from persistent_kv_admission.predictors import LogisticRanker, case_control_sample
from persistent_kv_admission.ranking import ranking_metrics
from persistent_kv_admission.replay import replay, _occurrence_groups
from persistent_kv_admission.temporal import (
    FEATURE_NAMES,
    TemporalHistory,
    model_indices,
    indices_excluding,
    standardize_rows,
)
from persistent_kv_admission.trace import load_mooncake_trace


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


class RecencyScorer:
    """Reproduces LRU exactly, which pins the scorer hook to a known policy."""

    time_varying = True

    def __init__(self, trace):
        self.history = TemporalHistory(trace)

    def observe(self, requests, timestamp_ms):
        self.history.observe_requests(requests, timestamp_ms)

    def score(self, state_id, timestamp_ms):
        return -(timestamp_ms - self.history.last_ms[state_id])


class FrequencyScorer:
    time_varying = False

    def __init__(self, trace):
        self.history = TemporalHistory(trace)

    def observe(self, requests, timestamp_ms):
        self.history.observe_requests(requests, timestamp_ms)

    def score(self, state_id, timestamp_ms):
        return float(self.history.count[state_id])


class TemporalFeatureTests(unittest.TestCase):
    def test_features_are_causal_and_use_only_past_occurrences(self):
        temporary, trace = build_trace(
            [
                {"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1]},
                {"timestamp": 2000, "input_length": 512, "output_length": 1, "hash_ids": [1]},
                {"timestamp": 9000, "input_length": 512, "output_length": 1, "hash_ids": [1]},
            ]
        )
        self.addCleanup(temporary.cleanup)
        history = TemporalHistory(trace)
        groups = list(trace.timestamp_groups())
        history.observe_requests(groups[0][1], groups[0][0])
        first = dict(zip(FEATURE_NAMES, history.feature_vector("int:1", 0.0)))
        self.assertEqual(first["single_occurrence"], 1.0)
        self.assertEqual(first["log_mean_interval"], 0.0)

        history.observe_requests(groups[1][1], groups[1][0])
        second = dict(zip(FEATURE_NAMES, history.feature_vector("int:1", 2000.0)))
        self.assertEqual(second["single_occurrence"], 0.0)
        self.assertAlmostEqual(second["log_mean_interval"], math.log1p(2.0))
        self.assertAlmostEqual(second["log_frequency"], math.log1p(2.0))
        # The third occurrence is in the future and must not be visible yet.
        self.assertAlmostEqual(second["log_interval_lag1"], math.log1p(2.0))

    def test_recency_feature_decays_without_new_observations(self):
        temporary, trace = build_trace(
            [{"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1]}]
        )
        self.addCleanup(temporary.cleanup)
        history = TemporalHistory(trace)
        timestamp, requests = next(iter(trace.timestamp_groups()))
        history.observe_requests(requests, timestamp)
        index = FEATURE_NAMES.index("neg_log_recency")
        near = history.feature_vector("int:1", 1_000.0)[index]
        far = history.feature_vector("int:1", 100_000.0)[index]
        self.assertGreater(near, far)

    def test_standardize_rows_handles_constant_columns(self):
        matrix = np.array([[1.0, 5.0], [3.0, 5.0]])
        standardized = standardize_rows(matrix)
        self.assertTrue(np.allclose(standardized[:, 1], 0.0))
        self.assertAlmostEqual(float(standardized[:, 0].std()), 1.0)

    def test_model_feature_sets_are_nested(self):
        base0, base1, base2 = (set(model_indices(name)) for name in ("base0", "base1", "base2"))
        self.assertTrue(base0 < base1 < base2)
        self.assertEqual(len(base2), len(FEATURE_NAMES))
        self.assertEqual(
            set(indices_excluding("base2", "structural")), base1
        )


class RankingAndFittingTests(unittest.TestCase):
    def test_fast_ranking_metrics_matches_reference_with_ties(self):
        generator = np.random.default_rng(11)
        for _ in range(5):
            size = int(generator.integers(40, 300))
            labels = (generator.uniform(size=size) < 0.25).astype(np.int8)
            scores = generator.integers(0, 4, size=size).astype(float)
            reference = ranking_metrics(labels, scores, k=20)
            fast = fast_ranking_metrics(labels, scores, k=20)
            for left, right in zip(reference, fast):
                self.assertAlmostEqual(left, right, places=9)

    def test_logistic_row_scoring_matches_batch_scoring(self):
        generator = np.random.default_rng(5)
        features = generator.normal(size=(500, len(FEATURE_NAMES)))
        labels = (generator.uniform(size=500) < 0.3).astype(float)
        ranker = LogisticRanker(indices=model_indices("base1"), l2=1e-2).fit(features, labels)
        batch = ranker.score(features)
        for index in range(25):
            self.assertAlmostEqual(ranker.score_row(list(features[index])), batch[index], places=9)

    def test_case_control_sampling_keeps_every_positive(self):
        generator = np.random.default_rng(7)
        features = generator.normal(size=(1000, 3))
        labels = (generator.uniform(size=1000) < 0.05).astype(np.int8)
        sampled_features, sampled_labels = case_control_sample(features, labels, 3.0, generator)
        self.assertEqual(int(sampled_labels.sum()), int(labels.sum()))
        self.assertLess(len(sampled_labels), len(labels))


class ReplayHookTests(unittest.TestCase):
    def make_trace(self):
        records = []
        for step in range(40):
            shared = [1, 2]
            records.append(
                {
                    "timestamp": step * 1000,
                    "input_length": 1536,
                    "output_length": 1,
                    "hash_ids": shared + [100 + step],
                }
            )
        return build_trace(records)

    def test_recency_scorer_reproduces_lru_exactly(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        capacity = 8 * 512
        baseline = replay(trace, "lru", capacity, 0.1, 1, occurrence_groups=groups)
        hooked = replay(
            trace, "recency", capacity, 0.1, 1,
            occurrence_groups=groups, scorer=RecencyScorer(trace),
        )
        self.assertEqual(baseline.avoided_prefill_tokens, hooked.avoided_prefill_tokens)
        self.assertEqual(baseline.hit_blocks, hooked.hit_blocks)

    def test_frequency_scorer_reproduces_lfu_exactly(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        capacity = 8 * 512
        baseline = replay(trace, "lfu", capacity, 0.1, 1, occurrence_groups=groups)
        hooked = replay(
            trace, "frequency", capacity, 0.1, 1,
            occurrence_groups=groups, scorer=FrequencyScorer(trace),
        )
        self.assertEqual(baseline.avoided_prefill_tokens, hooked.avoided_prefill_tokens)

    def test_sampled_eviction_keeps_the_retained_set_prefix_closed(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        result = replay(
            trace, "online", 6 * 512, 0.1, 1,
            occurrence_groups=groups,
            scorer=OnlineAdaptiveScorer(trace, horizon_seconds=5.0, samples_per_group=8),
            eviction="sampled",
        )
        self.assertGreaterEqual(result.avoided_prefill_tokens, 0)
        self.assertLessEqual(result.hit_blocks, result.requested_blocks)

    def test_measurement_window_excludes_the_warmup(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        groups = _occurrence_groups(trace)
        whole = replay(trace, "lru", 8 * 512, 0.1, 1, occurrence_groups=groups)
        tail = replay(
            trace, "lru", 8 * 512, 0.1, 1,
            occurrence_groups=groups, measure_from_ms=20_000,
        )
        self.assertEqual(whole.measured_requests, len(trace.requests))
        self.assertEqual(tail.measured_requests, 20)
        self.assertLess(tail.avoided_prefill_tokens, whole.avoided_prefill_tokens)


class OnlineAdaptationTests(unittest.TestCase):
    def test_online_scorer_starts_neutral_and_then_learns(self):
        temporary, trace = build_trace(
            [
                {"timestamp": step * 1000, "input_length": 1024, "output_length": 1,
                 "hash_ids": [1, 2 if step % 2 == 0 else 900 + step]}
                for step in range(60)
            ]
        )
        self.addCleanup(temporary.cleanup)
        scorer = OnlineAdaptiveScorer(trace, horizon_seconds=5.0, samples_per_group=8)
        self.assertTrue(np.allclose(scorer.weights, 0.0))
        for timestamp, requests in trace.timestamp_groups():
            scorer.observe(requests, timestamp)
        self.assertGreater(scorer.updates, 0)
        self.assertGreater(scorer.positive_examples, 0)
        self.assertFalse(np.allclose(scorer.weights, 0.0))

    def test_online_scorer_prefers_the_repeatedly_reused_state(self):
        temporary, trace = build_trace(
            [
                {"timestamp": step * 1000, "input_length": 1024, "output_length": 1,
                 "hash_ids": [1, 2 if step % 2 == 0 else 900 + step]}
                for step in range(80)
            ]
        )
        self.addCleanup(temporary.cleanup)
        scorer = OnlineAdaptiveScorer(trace, horizon_seconds=4.0, samples_per_group=16)
        last = 0.0
        for timestamp, requests in trace.timestamp_groups():
            scorer.observe(requests, timestamp)
            last = timestamp
        repeated = scorer.score("int:2", last)
        one_off = scorer.score("int:979", last)
        self.assertGreater(repeated, one_off)


class SplitDesignTests(unittest.TestCase):
    def test_horizon_longer_than_either_window_is_rejected(self):
        temporary, trace = build_trace(
            [
                {"timestamp": step * 1000, "input_length": 512, "output_length": 1,
                 "hash_ids": [step]}
                for step in range(100)
            ]
        )
        self.addCleanup(temporary.cleanup)
        usable, test_snapshots, train_snapshots, split_ms, _ = split_design(
            trace, (10, 60, 300), train_fraction=0.6, snapshot_count=12
        )
        # The trace spans 99 s, so only the 10 s horizon leaves room on both sides.
        self.assertEqual(usable, [10])
        self.assertTrue(test_snapshots)
        self.assertTrue(train_snapshots[10])
        self.assertAlmostEqual(split_ms, 0.6 * 99_000, places=3)


if __name__ == "__main__":
    unittest.main()
