"""Tests for the Phase 0.97 training populations, their logging, and their scoring."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.crossworkload import (
    FixedModelScorer,
    _normalize,
    fit_fixed_model,
    target_values,
)
from persistent_kv_admission.decisionpop import (
    CandidateLogger,
    OnPolicyDecisionLogger,
    RawFeatureScorer,
    Rows,
    VictimLogger,
    _Labeller,
    concatenate,
    deduplicate,
    evaluate,
    fit_population_ranker,
    horizon_for,
    lexicographic_score,
    population_metrics,
    score_rows,
    spearman,
    state_indices,
    target_column,
    union_across_logs,
)
from persistent_kv_admission.decisionpop import _OnPolicyDecision
from persistent_kv_admission.predictors import LogisticRanker, RidgeRanker
from persistent_kv_admission.temporal import (
    FEATURE_NAMES,
    TemporalHistory,
    model_indices,
    standardize_rows,
)
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def records(steps=300):
    """A hot chain, a small branching subtree, and cold leaves, one per second."""
    out = []
    for step in range(steps):
        if step % 4 == 0:
            chain = [1, 2]
        elif step % 7 == 0:
            chain = [1, 2, 3 + (step % 5)]
        else:
            chain = [1, 100 + step]
        out.append({"timestamp": step * 1000, "input_length": 512 * len(chain),
                    "output_length": 1, "hash_ids": chain})
    return out


class LabelTests(unittest.TestCase):
    def test_stored_primitives_reproduce_target_values_exactly(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        horizon = 30.0
        label = _Labeller(trace, horizon)
        ids = ["int:1", "int:2", "int:101", "int:299", "int:3"]
        for now_ms in (0.0, 12_345.0, 100_000.0, 299_000.0):
            pairs = [label(state_id, now_ms) for state_id in ids]
            deltas = np.asarray([p[0] for p in pairs], dtype=float)
            counts = np.asarray([p[1] for p in pairs], dtype=float)
            for target in ("binary", "count", "next_use"):
                expected = target_values(trace, ids, now_ms, horizon, target)
                got = target_column(deltas, counts, target, horizon)
                np.testing.assert_allclose(got, expected, rtol=0, atol=0,
                                           err_msg=f"{target} at {now_ms}")

    def test_a_state_never_used_again_gets_the_floor_and_a_zero_count(self):
        temporary, trace = build_trace(records(20))
        self.addCleanup(temporary.cleanup)
        label = _Labeller(trace, 10.0)
        delta, count = label("int:101", 19_000.0)
        self.assertEqual(count, 0)
        self.assertTrue(math.isinf(delta))
        self.assertEqual(target_column(np.array([delta]), np.array([count]), "binary", 10.0)[0], 0.0)
        self.assertAlmostEqual(
            target_column(np.array([delta]), np.array([count]), "next_use", 10.0)[0],
            -math.log1p(10.0),
        )


class SpearmanTests(unittest.TestCase):
    def test_matches_hand_computed_values_without_and_with_ties(self):
        # d = (0, 0, 0, 0) except one swapped pair each side: sum d^2 = 4,
        # rho = 1 - 6*4 / (5 * 24) = 0.8.
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)
        # Average ranks x = (0.5, 0.5, 2, 3), y = (0, 1, 2, 3):
        # covariance 4.5, variances 4.5 and 5.0, rho = 4.5 / sqrt(22.5).
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 3, 4]), 4.5 / math.sqrt(22.5))
        self.assertTrue(math.isnan(spearman([1, 1, 1], [1, 2, 3])))
        self.assertTrue(math.isnan(spearman([1.0], [2.0])))


class VictimLoggerTests(unittest.TestCase):
    def test_one_row_per_eviction_with_the_history_of_that_moment(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        logger = VictimLogger(trace, 30.0)
        seen = []
        result = run_two_tier(
            trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=logger,
            victim_hook=lambda s, t, g: (logger(s, t, g), seen.append((s, t, g)))[0],
        )
        rows = logger.rows()
        self.assertEqual(len(rows), result.l1_evictions)
        self.assertGreater(len(rows), 10)
        self.assertFalse(rows.grouped)
        self.assertEqual(rows.features.shape, (len(rows), len(FEATURE_NAMES)))
        # Replay the same history independently: a victim's features must be
        # the ones a scorer would compute after its own group was observed.
        history = TemporalHistory(trace)
        by_group: dict[int, list[tuple[str, float]]] = {}
        for state_id, timestamp_ms, group_index in seen:
            by_group.setdefault(group_index, []).append((state_id, timestamp_ms))
        expected = []
        for index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
            history.observe_requests(requests, timestamp_ms)
            for state_id, stamp in by_group.get(index, []):
                expected.append(history.feature_vector(state_id, stamp))
        np.testing.assert_allclose(rows.features, np.asarray(expected, dtype=np.float32))

    def test_labels_agree_with_the_trace(self):
        temporary, trace = build_trace(records())
        self.addCleanup(temporary.cleanup)
        logger = VictimLogger(trace, 30.0)
        run_two_tier(trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=logger,
                     victim_hook=logger)
        rows = logger.rows()
        names = list(trace.states)
        for position in (0, len(rows) // 2, len(rows) - 1):
            state_id = names[int(rows.state_index[position])]
            now = float(rows.timestamp_ms[position])
            expected = target_values(trace, [state_id], now, rows.horizon_seconds, "binary")[0]
            self.assertEqual(rows.labels("binary")[position], expected)


class CandidateLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)

    def _log(self, max_decisions=10_000, width=4):
        logger = CandidateLogger(self.trace, 30.0, max_decisions=max_decisions, seed=0)
        result = run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lru", bytes_per_token=1,
                              l2_eviction="sampled", l2_sample_width=width, l2_seed=0,
                              observer=logger, l2_decision_hook=logger)
        return logger, result

    def test_rows_line_up_with_the_logged_decisions(self):
        logger, result = self._log()
        self.assertEqual(logger.decisions_seen, result.l2_decisions)
        self.assertEqual(len(logger.kept), result.l2_decisions)
        rows = logger.rows()
        self.assertTrue(rows.grouped)
        self.assertEqual(len(np.unique(rows.group)), len(logger.kept))
        self.assertEqual(int(rows.victim.sum()), len(logger.kept))
        first_rounds = sum(1 for record in logger.kept if record.arriving_index >= 0)
        self.assertEqual(int(rows.arriving.sum()), first_rounds)
        self.assertGreater(first_rounds, 0)
        for index, record in enumerate(logger.kept):
            block = rows.group == index
            self.assertEqual(int(block.sum()), len(record.states))
            np.testing.assert_allclose(rows.features[block], record.features)
            self.assertTrue(np.all(rows.timestamp_ms[block] == record.timestamp_ms))
        # Every candidate row carries the history of its own decision instant.
        record = logger.kept[len(logger.kept) // 2]
        names = list(self.trace.states)
        history = TemporalHistory(self.trace)
        for timestamp_ms, requests in self.trace.timestamp_groups():
            history.observe_requests(requests, timestamp_ms)
            if timestamp_ms >= record.timestamp_ms:
                break
        expected = history.feature_matrix([names[i] for i in record.states], record.timestamp_ms)
        np.testing.assert_allclose(record.features, expected.astype(np.float32))

    def test_the_reservoir_is_bounded_and_keeps_both_sides_of_the_split(self):
        logger, _ = self._log(max_decisions=40)
        self.assertGreater(logger.decisions_seen, 40)
        self.assertEqual(len(logger.kept), 40)
        rows = logger.rows()
        split_ms = self.trace.start_ms + 0.6 * (self.trace.end_ms - self.trace.start_ms)
        self.assertGreater(int(rows.train_mask(split_ms).sum()), 0)
        self.assertGreater(int(rows.test_mask(split_ms, self.trace.end_ms).sum()), 0)

    def test_masks_respect_the_split_and_the_horizon_embargo(self):
        logger, _ = self._log()
        rows = logger.rows()
        split_ms = self.trace.start_ms + 0.6 * (self.trace.end_ms - self.trace.start_ms)
        train = rows.select(rows.train_mask(split_ms))
        test = rows.select(rows.test_mask(split_ms, self.trace.end_ms))
        self.assertTrue(np.all(train.timestamp_ms + 30_000.0 <= split_ms))
        self.assertTrue(np.all(test.timestamp_ms >= split_ms))
        self.assertTrue(np.all(test.timestamp_ms + 30_000.0 <= self.trace.end_ms))
        self.assertEqual(len(set(np.unique(train.timestamp_ms)) & set(np.unique(test.timestamp_ms))), 0)

    def _second_log(self):
        second = CandidateLogger(self.trace, 30.0, max_decisions=10_000, seed=0)
        run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lfu", bytes_per_token=1,
                     l2_eviction="sampled", l2_sample_width=4, l2_seed=0,
                     observer=second, l2_decision_hook=second)
        return second

    def test_the_union_drops_only_what_the_earlier_log_already_exposed(self):
        first = self._log()[0].rows()
        second = self._second_log().rows()
        union, dropped = union_across_logs([first, second])
        self.assertGreater(dropped, 0)
        self.assertEqual(len(union), len(first) + len(second) - dropped)
        # A superset of its first component: nothing of that log is collapsed,
        # repeats inside it included.
        self.assertGreaterEqual(len(union), len(first))
        self.assertGreaterEqual(len(union), len(second))
        keys = lambda rows: list(zip(rows.timestamp_ms.tolist(), rows.state_index.tolist()))
        first_keys = keys(first)
        self.assertEqual(first_keys, keys(union)[:len(first_keys)])
        seen = set(first_keys)
        self.assertEqual(dropped, sum(1 for key in keys(second) if key in seen))
        # What was dropped was a copy: same (t, state), same features, same labels.
        table = {key: position for position, key in enumerate(first_keys)}
        checked = 0
        for position, key in enumerate(keys(second)):
            if key in table:
                np.testing.assert_allclose(first.features[table[key]], second.features[position])
                self.assertEqual(first.next_use_delta_ms[table[key]],
                                 second.next_use_delta_ms[position])
                self.assertEqual(first.count_within_h[table[key]], second.count_within_h[position])
                checked += 1
        self.assertEqual(checked, dropped)
        # Groups stay distinct across the two logs: the first log's decisions
        # keep their numbers and the second's are shifted past them. A decision
        # of the second log whose every candidate was already exposed loses all
        # its rows, so the union can hold fewer groups than the two logs sum to.
        self.assertEqual(union.group[:len(first)].tolist(), first.group.tolist())
        self.assertTrue(np.all(union.group[len(first):] > first.group.max()))
        self.assertLessEqual(len(np.unique(union.group)),
                             len(np.unique(first.group)) + len(np.unique(second.group)))


    def test_a_repeat_inside_one_log_survives_the_union(self):
        # Two decisions at one instant that both sampled state 7: separate
        # decisions the state really faced, so the union keeps both. The second
        # log's copy of that same (t, state) is the only thing dropped.
        def rows(timestamps, states, group):
            size = len(states)
            return Rows(np.asarray(timestamps, dtype=float),
                        np.asarray(states, dtype=np.int32),
                        np.tile(np.asarray(states, dtype=np.float32)[:, None],
                                (1, len(FEATURE_NAMES))),
                        np.zeros(size), np.zeros(size),
                        np.asarray(group, dtype=np.int64),
                        np.zeros(size, dtype=np.int8), np.zeros(size, dtype=np.int8), 30.0)

        left = rows([5.0, 5.0, 5.0, 5.0], [7, 8, 7, 9], [0, 0, 1, 1])
        right = rows([5.0, 5.0, 6.0], [7, 11, 7], [0, 0, 1])
        union, dropped = union_across_logs([left, right])
        self.assertEqual(dropped, 1)
        self.assertEqual(union.state_index.tolist(), [7, 8, 7, 9, 11, 7])
        self.assertEqual(union.timestamp_ms.tolist(), [5.0, 5.0, 5.0, 5.0, 5.0, 6.0])
        # Groups stay distinct: the right log's 0 and 1 are renumbered.
        self.assertEqual(union.group.tolist(), [0, 0, 1, 1, 2, 3])
        self.assertEqual(deduplicate(left)[1], 1)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)
        self.ranker, _, self.split_ms, _, _ = fit_fixed_model(
            self.trace, 30.0, snapshot_count=8, candidate_cap=500
        )

    def test_raw_feature_scorer_matches_batch_scoring_of_the_same_row(self):
        scorer = RawFeatureScorer(self.trace, self.ranker)
        for timestamp_ms, requests in self.trace.timestamp_groups():
            scorer.observe(requests, timestamp_ms)
            if timestamp_ms >= 20_000.0:
                break
        state_id = "int:2"
        row = scorer.history.feature_vector(state_id, timestamp_ms)
        self.assertAlmostEqual(scorer.score(state_id, timestamp_ms),
                               float(self.ranker.score(np.asarray([row]))[0]), places=10)

    def test_per_decision_scoring_standardises_inside_each_group(self):
        logger = CandidateLogger(self.trace, 30.0, max_decisions=200, seed=0)
        run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lru", bytes_per_token=1,
                     l2_eviction="sampled", l2_sample_width=4, l2_seed=0,
                     observer=logger, l2_decision_hook=logger)
        rows = logger.rows()
        scores = score_rows(self.ranker, rows, "per_decision")
        for index in np.unique(rows.group):
            block = rows.group == index
            expected = self.ranker.score(standardize_rows(rows.features[block].astype(float)))
            np.testing.assert_allclose(scores[block], expected)
        raw = score_rows(self.ranker, rows, "raw")
        np.testing.assert_allclose(raw, self.ranker.score(rows.features.astype(float)))
        pooled = score_rows(self.ranker, rows, "pooled")
        np.testing.assert_allclose(
            pooled, self.ranker.score(standardize_rows(rows.features.astype(float)))
        )
        with self.assertRaises(ValueError):
            score_rows(self.ranker, rows, "per_snapshot")

    def test_evicted_lowest_label_rate_counts_ties_as_lowest(self):
        # Three decisions of two candidates. Scores rank the first candidate
        # lowest everywhere. Labels: decision 0 the argmin also has the lowest
        # label; decision 1 it has the higher label; decision 2 they tie.
        features = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
        rows = Rows(
            timestamp_ms=np.zeros(6),
            state_index=np.arange(6, dtype=np.int32),
            features=features,
            next_use_delta_ms=np.array([1e9, 1.0, 1.0, 1e9, 1.0, 1.0]),
            count_within_h=np.zeros(6),
            group=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            victim=np.zeros(6, dtype=np.int8),
            arriving=np.zeros(6, dtype=np.int8),
            horizon_seconds=30.0,
        )
        scores = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        metrics = population_metrics(scores, rows, "binary", decision_sets=True)
        self.assertAlmostEqual(metrics["evicted_lowest_label_rate"], 2 / 3)
        self.assertEqual(metrics["groups"], 3.0)
        ungrouped = rows.select(rows.group == 0)
        ungrouped.group[:] = -1
        self.assertTrue(math.isnan(population_metrics(scores[:2], ungrouped, "binary")["within_decision_macro"]))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)

    def _victim_rows(self, horizon=30.0):
        logger = VictimLogger(self.trace, horizon)
        run_two_tier(self.trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=logger,
                     victim_hook=logger)
        return logger.rows()

    def test_fit_uses_the_shared_model_penalty_and_feature_set(self):
        rows = self._victim_rows()
        ranker, positives, used = fit_population_ranker(rows, "binary", seed=0)
        self.assertEqual(ranker.indices, model_indices("base2"))
        self.assertEqual(ranker.l2, 0.01)
        self.assertTrue(ranker.standardize)
        self.assertEqual(len(ranker.coefficients), len(FEATURE_NAMES))
        self.assertGreater(positives, 0)
        self.assertLessEqual(used, len(rows))
        # No subsampling happens here, so the fit is reproducible by hand.
        labels = rows.labels("binary")
        if int(labels.sum()) * 10 >= len(labels) - int(labels.sum()):
            manual = LogisticRanker(indices=model_indices("base2"), l2=0.01, standardize=True).fit(
                rows.features.astype(float), labels
            )
            np.testing.assert_allclose(ranker.coefficients, manual.coefficients)

    def test_graded_targets_use_the_ridge_ranker_and_keep_every_row(self):
        rows = self._victim_rows()
        for target in ("count", "next_use"):
            ranker, positives, used = fit_population_ranker(rows, target, seed=0)
            self.assertEqual(type(ranker).__name__, "RidgeRanker")
            self.assertEqual(used, min(len(rows), 150_000))
            self.assertGreaterEqual(positives, 0)
        with self.assertRaises(ValueError):
            fit_population_ranker(rows, "nonsense")

    def test_evaluate_returns_the_expected_metric_per_target(self):
        rows = self._victim_rows()
        ranker, _, _ = fit_population_ranker(rows, "binary", seed=0)
        metrics = evaluate(ranker, rows, "binary", "raw")
        self.assertTrue(0.0 <= metrics["pooled_metric"] <= 1.0)
        graded, _, _ = fit_population_ranker(rows, "next_use", seed=0)
        spearman_metrics = evaluate(graded, rows, "next_use", "raw")
        self.assertTrue(-1.0 <= spearman_metrics["pooled_metric"] <= 1.0)


class HorizonAndStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)

    def test_horizon_for_agrees_with_fit_fixed_model(self):
        horizon, split_ms, snapshots = horizon_for(self.trace, 600.0, snapshot_count=8)
        ranker, used, fitted_split, _, _ = fit_fixed_model(
            self.trace, 600.0, snapshot_count=8, candidate_cap=500
        )
        self.assertEqual(horizon, float(used))
        self.assertEqual(split_ms, fitted_split)
        self.assertGreater(len(snapshots), 0)

    def test_rows_round_trip_through_npz(self):
        logger = VictimLogger(self.trace, 30.0)
        run_two_tier(self.trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=logger,
                     victim_hook=logger)
        rows = logger.rows()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "rows.npz"
        rows.save(path)
        back = Rows.load(path)
        self.assertEqual(len(back), len(rows))
        self.assertEqual(back.horizon_seconds, rows.horizon_seconds)
        np.testing.assert_allclose(back.features, rows.features)
        np.testing.assert_allclose(back.next_use_delta_ms, rows.next_use_delta_ms)
        self.assertEqual(back.state_index.tolist(), rows.state_index.tolist())

    def test_state_indices_cover_every_state_once(self):
        index = state_indices(self.trace)
        self.assertEqual(len(index), len(self.trace.states))
        self.assertEqual(sorted(index.values()), list(range(len(self.trace.states))))


class NormaliserFallbackTests(unittest.TestCase):
    class _Cache:
        def __init__(self, cached):
            self.cached = cached

    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)
        self.ranker, _, _, _, _ = fit_fixed_model(self.trace, 30.0, snapshot_count=8,
                                                  candidate_cap=500)
        self.groups = list(self.trace.timestamp_groups())

    def _scorer(self, cached):
        scorer = FixedModelScorer(self.trace, self.ranker, refresh_every=1, seed=0,
                                  normalize_on="cached")
        scorer.attach(self._Cache(cached))
        return scorer

    def test_a_ranked_set_of_one_falls_back_to_the_observed_population(self):
        scorer = self._scorer({"int:2": None})
        for timestamp_ms, requests in self.groups[:5]:
            scorer.observe(requests, timestamp_ms)
        self.assertEqual(scorer.normalizer.refreshes, 5)
        observed = scorer.history.observed_state_ids()
        expected = scorer.history.feature_matrix(observed, timestamp_ms)
        np.testing.assert_allclose(scorer.normalizer.mean, expected.mean(axis=0))

    def test_a_ranked_set_of_two_or_more_still_normalises_on_it(self):
        scorer = self._scorer({"int:1": None, "int:2": None})
        for timestamp_ms, requests in self.groups[:5]:
            scorer.observe(requests, timestamp_ms)
        expected = scorer.history.feature_matrix(["int:1", "int:2"], timestamp_ms)
        np.testing.assert_allclose(scorer.normalizer.mean, expected.mean(axis=0))



class StandardisationConventionTests(unittest.TestCase):
    """A_none, B and every C fit take raw rows and standardise inside the ranker."""

    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)

    def test_the_ranker_variant_leaves_rows_raw_and_none_keeps_its_meaning(self):
        matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
        np.testing.assert_allclose(_normalize(matrix, "ranker"), matrix)
        np.testing.assert_allclose(_normalize(matrix, "none"), matrix)
        with self.assertRaises(ValueError):
            _normalize(matrix, "ranker_internal")
        # "none" still means no standardisation anywhere, ranker included.
        control, _, _, _, _ = fit_fixed_model(self.trace, 30.0, snapshot_count=8,
                                              candidate_cap=500, standardization="none")
        self.assertFalse(control.standardize)
        np.testing.assert_allclose(control.mean, np.zeros_like(control.mean))
        np.testing.assert_allclose(control.scale, np.ones_like(control.scale))

    def test_a_pd_standardises_per_decision_and_the_others_inside_the_ranker(self):
        a_pd, _, _, _, _ = fit_fixed_model(self.trace, 30.0, snapshot_count=8,
                                           candidate_cap=500, standardization="per_decision")
        a_none, _, _, _, _ = fit_fixed_model(self.trace, 30.0, snapshot_count=8,
                                             candidate_cap=500, standardization="ranker")
        self.assertTrue(a_pd.standardize)
        self.assertTrue(a_none.standardize)
        # Raw rows in means the stored scales are the training population's own.
        self.assertFalse(np.allclose(a_none.scale, np.ones_like(a_none.scale)))
        victims = VictimLogger(self.trace, 30.0)
        run_two_tier(self.trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=victims,
                     victim_hook=victims)
        candidates = CandidateLogger(self.trace, 30.0, max_decisions=500, seed=0)
        run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lru", bytes_per_token=1,
                     l2_eviction="sampled", l2_sample_width=4, l2_seed=0,
                     observer=candidates, l2_decision_hook=candidates)
        for name, rows in (("B", victims.rows()), ("C", candidates.rows())):
            selected = rows.features[:, list(model_indices("base2"))].astype(float)
            for target in ("binary", "count", "next_use"):
                ranker, _, _ = fit_population_ranker(rows, target, seed=0)
                self.assertTrue(ranker.standardize, f"{name}/{target}")
                self.assertFalse(np.allclose(ranker.scale, np.ones_like(ranker.scale)),
                                 f"{name}/{target}")
                if target != "binary":
                    # Graded targets keep every row, so the stored mean is the
                    # raw population mean: the rows really did go in unscaled.
                    np.testing.assert_allclose(ranker.mean, selected.mean(axis=0), rtol=1e-10)



class LexicographicScoreTests(unittest.TestCase):
    def test_it_reproduces_the_order_of_the_score_tuple_including_ties(self):
        # LFU ranks by (frequency, last_group): the pairs must order exactly as
        # tuple comparison does, and equal pairs must score equal.
        primary = np.array([2.0, 1.0, 2.0, 1.0, 2.0])
        secondary = np.array([5.0, 9.0, 3.0, 9.0, 5.0])
        scores = lexicographic_score(primary, secondary)
        pairs = list(zip(primary.tolist(), secondary.tolist()))
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                self.assertEqual(pairs[i] < pairs[j], scores[i] < scores[j], (i, j))
                self.assertEqual(pairs[i] == pairs[j], scores[i] == scores[j], (i, j))
        # A one-element key (LRU) arrives with a constant or absent tie-break.
        lru = lexicographic_score(np.array([3.0, 1.0, 2.0]), np.full(3, np.nan))
        self.assertEqual(lru.tolist(), [2.0, 0.0, 1.0])
        self.assertEqual(lexicographic_score(np.zeros(0), np.zeros(0)).tolist(), [])


class OnPolicyLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)
        self.split_ms = self.trace.start_ms + 0.6 * (self.trace.end_ms - self.trace.start_ms)

    def test_it_records_the_stores_own_scores_and_victims_inside_the_window(self):
        seen = []
        logger = OnPolicyDecisionLogger(self.trace, 30.0, self.split_ms, self.trace.end_ms,
                                        max_decisions=10_000, seed=0)

        def hook(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
            seen.append((timestamp_ms, tuple(candidates), tuple(tuple(s) for s in scores),
                         victim_index, arriving_index))
            logger(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index)

        result = run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lfu", bytes_per_token=1,
                              l2_eviction="sampled", l2_sample_width=4, l2_seed=0,
                              l2_decision_hook=hook)
        self.assertEqual(logger.decisions_offered, result.l2_decisions)
        eligible = [s for s in seen
                    if s[0] >= self.split_ms and s[0] + 30_000.0 <= self.trace.end_ms]
        self.assertGreater(len(eligible), 5)
        self.assertLess(len(eligible), len(seen))
        self.assertEqual(logger.decisions_seen, len(eligible))
        self.assertEqual(len(logger.kept), len(eligible))
        names = list(self.trace.states)
        widths = set()
        for record, (stamp, candidates, scores, victim, arriving) in zip(logger.kept, eligible):
            self.assertEqual(record.timestamp_ms, stamp)
            self.assertEqual([names[index] for index in record.states], list(candidates))
            # Recorded, never recomputed: the store's own tuples, both positions.
            widths.update(len(score) for score in scores)
            self.assertEqual(record.scores.tolist(), [float(s[0]) for s in scores])
            self.assertEqual(record.tiebreaks.tolist(),
                             [float(s[1]) if len(s) > 1 else 0.0 for s in scores])
            # The order the store compared, and the victim it chose, agree.
            self.assertEqual(record.order.tolist(),
                             lexicographic_score(record.scores, record.tiebreaks).tolist())
            self.assertEqual(int(np.argmin(record.order)), record.victim_index)
            self.assertEqual(record.victim_index, victim)
            self.assertEqual(record.arriving_index, arriving)
        # LFU ranks by a two-element key, so the tie-break really was recorded.
        self.assertEqual(widths, {2})
        metrics = logger.metrics("binary")
        self.assertEqual(int(metrics["decisions_kept"]), len(eligible))
        self.assertEqual(metrics["victim_matches_argmin_rate"], 1.0)

    def test_the_reservoir_is_bounded(self):
        logger = OnPolicyDecisionLogger(self.trace, 30.0, self.split_ms, self.trace.end_ms,
                                        max_decisions=7, seed=0)
        run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lfu", bytes_per_token=1,
                     l2_eviction="sampled", l2_sample_width=4, l2_seed=0, l2_decision_hook=logger)
        self.assertEqual(len(logger.kept), 7)
        self.assertGreater(logger.decisions_seen, 7)
        self.assertGreater(logger.decisions_offered, logger.decisions_seen)

    def _hand_built(self):
        logger = OnPolicyDecisionLogger(self.trace, 30.0, 0.0, self.trace.end_ms, seed=0)
        far = math.inf
        near = 1_000.0
        # d0: the evicted candidate is the only one reused; d1: it is the only
        # one not reused; d2: both reused, so nothing to get right.
        decisions = [((near, far), (1.0, 0.0), 0), ((far, near), (1.0, 0.0), 0),
                     ((near, near), (1.0, 1.0), 0)]
        for deltas, counts, victim in decisions:
            primary, secondary = np.array([0.0, 1.0]), np.zeros(2)
            logger.kept.append(_OnPolicyDecision(
                0.0, 0, np.array([0, 1], dtype=np.int32), primary, secondary,
                lexicographic_score(primary, secondary),
                np.asarray(deltas, dtype=float), np.asarray(counts, dtype=float), victim, 0))
        logger.decisions_seen = len(logger.kept)
        logger.decisions_offered = len(logger.kept)
        return logger

    def test_metrics_use_the_recorded_victim_and_the_two_distinct_label_rule(self):
        metrics = self._hand_built().metrics("binary")
        # d0 ranks the positive lowest (AUC 0), d1 ranks it highest (AUC 1),
        # d2 has one distinct label and is not scored.
        self.assertEqual(metrics["decisions_kept"], 3.0)
        self.assertEqual(metrics["decisions_scored"], 2.0)
        self.assertEqual(metrics["decisions_constant_label"], 1.0)
        self.assertAlmostEqual(metrics["decisions_constant_label_share"], 1 / 3)
        self.assertAlmostEqual(metrics["within_decision_macro"], 0.5)
        self.assertAlmostEqual(metrics["within_decision_micro"], 0.5)
        # Evicted candidate carries the lowest label in d1 and d2, not in d0.
        self.assertAlmostEqual(metrics["evicted_lowest_label_rate"], 2 / 3)
        # A negative was available in d0 and d1; the evicted one was positive
        # in d0 only. d2 has no negative, so it is not counted.
        self.assertAlmostEqual(metrics["evicted_positive_rate_when_avoidable"], 0.5)
        graded = self._hand_built().metrics("count")
        # log1p(1) vs log1p(0) is two distinct labels: a real ranking problem.
        self.assertEqual(graded["decisions_scored"], 2.0)
        self.assertEqual(graded["decisions_constant_label"], 1.0)
        self.assertTrue(math.isnan(graded["within_decision_micro"]))
        self.assertTrue(math.isnan(graded["evicted_positive_rate_when_avoidable"]))


    def test_an_lfu_tie_is_broken_by_recency_the_way_the_store_breaks_it(self):
        # Two candidates with the same frequency: LFU's key is (frequency,
        # last_group), so the older one loses. Reading only the first element
        # would call this a tie and score the decision 0.5; the tuple order
        # ranks it correctly and the store evicted accordingly.
        logger = OnPolicyDecisionLogger(self.trace, 30.0, 0.0, self.trace.end_ms, seed=0)
        primary = np.array([3.0, 3.0])       # equal frequency
        secondary = np.array([10.0, 4.0])    # the second was used longer ago
        order = lexicographic_score(primary, secondary)
        self.assertEqual(order.tolist(), [1.0, 0.0])
        logger.kept.append(_OnPolicyDecision(
            0.0, 0, np.array([0, 1], dtype=np.int32), primary, secondary, order,
            np.array([1_000.0, math.inf]), np.array([1.0, 0.0]), 1, 0))
        logger.decisions_seen = logger.decisions_offered = 1
        metrics = logger.metrics("binary")
        self.assertAlmostEqual(metrics["within_decision_macro"], 1.0)
        self.assertAlmostEqual(metrics["within_decision_micro"], 1.0)
        self.assertEqual(metrics["victim_matches_argmin_rate"], 1.0)
        self.assertEqual(metrics["evicted_lowest_label_rate"], 1.0)
        self.assertEqual(metrics["evicted_positive_rate_when_avoidable"], 0.0)
        # Ignoring the tie-break would have made the pair indistinguishable.
        flat = lexicographic_score(primary, np.zeros(2))
        self.assertEqual(flat.tolist(), [0.0, 0.0])


class WithinDecisionRuleTests(unittest.TestCase):
    def _rows(self, deltas, counts, group):
        size = len(deltas)
        return Rows(np.zeros(size), np.arange(size, dtype=np.int32),
                    np.zeros((size, len(FEATURE_NAMES)), dtype=np.float32),
                    np.asarray(deltas, dtype=float), np.asarray(counts, dtype=float),
                    np.asarray(group, dtype=np.int64), np.zeros(size, dtype=np.int8),
                    np.zeros(size, dtype=np.int8), 30.0)

    def test_two_distinct_graded_labels_are_scored_and_constants_are_counted(self):
        # Decision 0 has counts {1, 0} — two distinct labels, so it is ranked.
        # Decision 1 has counts {2, 2} — constant, so it is only counted.
        rows = self._rows([1_000.0, math.inf, 1_000.0, 1_000.0], [1.0, 0.0, 2.0, 2.0],
                          [0, 0, 1, 1])
        scores = np.array([1.0, 0.0, 0.0, 1.0])
        metrics = population_metrics(scores, rows, "count", decision_sets=True)
        self.assertEqual(metrics["decisions_scored"], 1.0)
        self.assertEqual(metrics["decisions_constant_label"], 1.0)
        self.assertAlmostEqual(metrics["decisions_constant_label_share"], 0.5)
        self.assertAlmostEqual(metrics["within_decision_macro"], 1.0)
        binary = population_metrics(scores, rows, "binary", decision_sets=True)
        self.assertEqual(binary["decisions_scored"], 1.0)
        self.assertEqual(binary["decisions_constant_label"], 1.0)


class ConvergenceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(records())
        self.addCleanup(self.temporary.cleanup)
        logger = VictimLogger(self.trace, 30.0)
        run_two_tier(self.trace, "lru", 3 * 512, 0, bytes_per_token=1, observer=logger,
                     victim_hook=logger)
        self.rows = logger.rows()

    def test_every_fit_reports_whether_it_solved(self):
        binary, _, _ = fit_population_ranker(self.rows, "binary", seed=0)
        self.assertIsInstance(binary, LogisticRanker)
        self.assertTrue(binary.converged)
        self.assertGreater(binary.iterations, 0)
        self.assertLessEqual(binary.iterations, binary.max_iterations)
        for target in ("count", "next_use"):
            graded, _, _ = fit_population_ranker(self.rows, target, seed=0)
            self.assertIsInstance(graded, RidgeRanker)
            self.assertTrue(math.isfinite(graded.condition_number))
            self.assertGreaterEqual(graded.condition_number, 1.0)
        # The logistic ranker carries no condition number and the ridge one no
        # iteration count worth reading; the CSV reads both through getattr.
        self.assertTrue(math.isnan(getattr(binary, "condition_number", math.nan)))


if __name__ == "__main__":
    unittest.main()
