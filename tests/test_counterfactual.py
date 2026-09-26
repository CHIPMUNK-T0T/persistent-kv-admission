"""Constructed-trace checks for one-step sampled-L2 counterfactuals."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from persistent_kv_admission.counterfactual import (
    ForkController,
    WindowReward,
    continuation_seed,
    decision_hash,
    label_tie_sets,
    load_population_verified,
    select_decisions,
    selectors,
)
from persistent_kv_admission.onpolicy import DecisionPopulation, deserialize_ranker
from persistent_kv_admission.temporal import FEATURE_NAMES
from persistent_kv_admission.temporal import TemporalHistory


def _runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_counterfactual.py"
    spec = importlib.util.spec_from_file_location("_counterfactual_runner_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
from persistent_kv_admission.trace import Request, StateMeta, Trace
from persistent_kv_admission.twotier import _VictimStore, run_two_tier


def _trace(groups: list[tuple[float, list[str]]], *, tokens: dict[str, int] | None = None) -> Trace:
    """Independent root blocks keep the expected hit value unambiguous."""
    tokens = tokens or {}
    requests = []
    occurrences = defaultdict(list)
    states = {}
    order = 0
    for timestamp_ms, state_ids in groups:
        for state_id in state_ids:
            block_tokens = tokens.get(state_id, 512)
            states[state_id] = StateMeta(state_id, None, 1, block_tokens, block_tokens)
            requests.append(Request(order, timestamp_ms, block_tokens, 1, (state_id,)))
            occurrences[state_id].append(timestamp_ms)
            order += 1
    return Trace("counterfactual-synthetic", requests, states, dict(occurrences), {}, {}, 512)


def _run(trace: Trace, **kwargs):
    return run_two_tier(
        trace, "lru", 512, 512, "lru", bytes_per_token=1,
        l2_eviction="sampled", l2_sample_width=16, l2_seed=7, **kwargs,
    )


class _SyntheticScorer:
    time_varying = True

    def __init__(self, trace: Trace, *, fail_in_child_at: float | None = None):
        self.history = TemporalHistory(trace)
        self.parent_pid = os.getpid()
        self.fail_in_child_at = fail_in_child_at

    def observe(self, requests, timestamp_ms):
        if (self.fail_in_child_at is not None and os.getpid() != self.parent_pid
                and timestamp_ms >= self.fail_in_child_at):
            raise RuntimeError("injected synthetic child failure")
        self.history.observe_requests(requests, timestamp_ms)

    def score(self, state_id, timestamp_ms):
        return float(ord(state_id) - ord("A"))


def _fork_fixture_selected(
    trace: Trace, state_index: dict[str, int], *, group_to_capture=2,
    l2_capacity=512, sample_width=16,
) -> dict:
    """Capture one complete decision independently before enabling the fork."""
    from persistent_kv_admission.decisionpop import _Labeller

    scorer = _SyntheticScorer(trace)
    captured = []
    labeller = _Labeller(trace, 600.0)

    def record(candidates, scores, victim, timestamp, group, arriving):
        if group != group_to_capture:
            return
        labels = [labeller(state_id, timestamp) for state_id in candidates]
        captured.append({
            "lineage": "synthetic/seed7",
            "hash_rank": 0,
            "selection_hash": decision_hash("synthetic/seed7", group, 0),
            "group_index": group,
            "ordinal": 0,
            "timestamp_ms": timestamp,
            "candidate_indices": [state_index[state_id] for state_id in candidates],
            "scores": [list(score) for score in scores],
            "victim_index": victim,
            "arriving_index": arriving,
            "features": [scorer.history.feature_vector(state_id, timestamp)
                         for state_id in candidates],
            "next_use_delta_ms": [None if np.isinf(delta) else delta for delta, _ in labels],
            "count_within_h": [count for _, count in labels],
        })

    run_two_tier(
        trace, "lru", 512, l2_capacity, "learned", bytes_per_token=1,
        l2_eviction="sampled", l2_sample_width=sample_width, l2_seed=7,
        l2_scorer=scorer, l2_decision_hook=record,
    )
    assert len(captured) == 1
    return captured[0]


def _fork_replay(
    trace, selected, branch_dir, *, fail_in_child_at=None,
    l2_capacity=512, sample_width=16,
):
    scorer = _SyntheticScorer(trace, fail_in_child_at=fail_in_child_at)
    state_index = {state_id: index for index, state_id in enumerate(sorted(trace.states))}
    controller = ForkController(
        trace, scorer, state_index, [selected], branch_dir, "synthetic-code-and-input-v1"
    )
    try:
        result = run_two_tier(
            trace, "lru", 512, l2_capacity, "learned", bytes_per_token=1,
            l2_eviction="sampled", l2_sample_width=sample_width, l2_seed=7,
            l2_scorer=scorer, l2_override_hook=controller,
            l2_request_hook=controller.on_request,
        )
    except BaseException as error:
        if controller.is_child:
            controller.fail_child(error)
            os._exit(2)
        raise
    if controller.is_child:
        try:
            controller.finish_child(result)
        except BaseException as error:
            controller.fail_child(error)
            os._exit(2)
        os._exit(0)
    return result, controller.validate_parent(result)


class SampledOverrideTests(unittest.TestCase):
    def setUp(self):
        # At t=2, offering B finds A in L2. A's removal keeps B; rejecting B
        # keeps A, which is needed again at t=3.
        self.trace = _trace([(0, ["A"]), (1000, ["B"]), (2000, ["C"]),
                             (3000, ["A"])])

    def test_default_and_explicit_noop_are_exactly_equal(self):
        default_decisions = []
        noop_decisions = []

        def record(target):
            def hook(candidates, scores, victim, timestamp, group, arriving):
                target.append((tuple(candidates), tuple(scores), victim, timestamp, group, arriving))
            return hook

        baseline = _run(self.trace, l2_decision_hook=record(default_decisions))
        noop = _run(self.trace, l2_decision_hook=record(noop_decisions),
                    l2_override_hook=lambda *args: None)
        self.assertGreater(baseline.l2_decisions, 0)
        self.assertEqual(default_decisions, noop_decisions)
        self.assertEqual(baseline.as_row(), noop.as_row())

    def test_every_exposed_candidate_is_legal_and_invalid_indices_fail(self):
        seen = []

        def choose_first(candidates, scores, original, timestamp, group, arriving):
            seen.append((tuple(candidates), original, 0))
            return 0

        result = _run(self.trace, l2_override_hook=choose_first)
        self.assertEqual(result.l2_decisions, len(seen))
        self.assertTrue(any(choice != original for _, original, choice in seen))
        self.assertTrue(all(0 <= choice < len(candidates) for candidates, _, choice in seen))
        for bad in (-1, 99, True, 0.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _run(self.trace, l2_override_hook=lambda *args, value=bad: value)

    def test_rejecting_arrival_and_evicting_resident_change_future_hits(self):
        snapshots = []

        def decide(reject):
            def hook(candidates, scores, original, timestamp, group, arriving):
                if group == 2:
                    snapshots.append((tuple(candidates), original, arriving))
                    self.assertEqual(tuple(candidates), ("B", "A"))
                    self.assertEqual(arriving, 0)
                    return 0 if reject else 1
                return None
            return hook

        removals_reject = []
        removals_keep = []
        rejected = _run(self.trace, l2_override_hook=decide(True),
                        l2_removal_hook=lambda *args: removals_reject.append(args))
        kept = _run(self.trace, l2_override_hook=decide(False),
                    l2_removal_hook=lambda *args: removals_keep.append(args))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(rejected.l1_avoided_tokens, kept.l1_avoided_tokens)
        self.assertEqual(rejected.l2_avoided_tokens - kept.l2_avoided_tokens, 512)
        self.assertIn(("B", "rejected", 2000, 2, 0), removals_reject)
        self.assertIn(("A", "evicted", 2000, 2, 0), removals_keep)

    def test_partial_block_removal_continues_normal_policy_in_later_rounds(self):
        trace = _trace([(0, ["small", "large", "other", "arrival"])],
                       tokens={"small": 1, "large": 3, "other": 1, "arrival": 4})
        scores = {"small": 2.0, "large": 0.0, "other": 1.0, "arrival": 3.0}

        class Scorer:
            time_varying = True

            def observe(self, requests, timestamp_ms):
                pass

            def score(self, state_id, timestamp_ms):
                return scores[state_id]

        decisions = []
        removals = []

        def override(candidates, score_tuples, original, timestamp, group, arriving):
            decisions.append((tuple(candidates), original, arriving))
            # First round removes only one byte, leaving more than capacity.
            return candidates.index("small") if len(decisions) == 1 else None

        store = _VictimStore(
            trace, 5, "learned", {}, lambda s: trace.states[s].block_tokens,
            eviction="sampled", sample_width=16, seed=0, scorer=Scorer(),
            frequency=defaultdict(int), last_group=defaultdict(int),
            override_hook=override,
            removal_hook=lambda state_id, kind, *_: removals.append((state_id, kind)),
        )
        for state_id in ("small", "large", "other"):
            self.assertTrue(store.admit(state_id, 1, 0, 0))
        self.assertTrue(store.admit("arrival", 1, 1, 1, 1000.0))
        self.assertEqual(decisions[0][0][0], "arrival")
        self.assertEqual(decisions[0][2], 0)
        self.assertEqual(decisions[0][0][decisions[0][1]], "large")
        self.assertGreater(len(decisions), 1)
        self.assertTrue(all(arriving == -1 for _, _, arriving in decisions[1:]))
        self.assertEqual(removals[0], ("small", "evicted"))
        self.assertEqual(removals[1], ("large", "evicted"))
        self.assertLessEqual(store.current_bytes, store.capacity_bytes)
        self.assertIn("arrival", store.cached)

    def test_nonleaf_resident_remains_a_legal_override_action(self):
        states = {
            "root": StateMeta("root", None, 1, 1, 1),
            "child": StateMeta("child", "root", 2, 2, 1),
            "arrival": StateMeta("arrival", None, 1, 1, 1),
        }
        trace = Trace("nonleaf", [], states, {}, {"root": {"child"}}, {}, 1)
        choices = []

        def override(candidates, scores, original, timestamp, group, arriving):
            choices.append(tuple(candidates))
            return candidates.index("root")

        store = _VictimStore(
            trace, 2, "lru", {}, lambda state_id: 1,
            eviction="sampled", sample_width=16, seed=0,
            frequency=defaultdict(int), last_group=defaultdict(int),
            override_hook=override,
        )
        self.assertTrue(store.admit("root", 1, 0, 0))
        self.assertTrue(store.admit("child", 1, 0, 0))
        self.assertTrue(store.admit("arrival", 1, 1, 1))
        self.assertEqual(choices, [("arrival", "root", "child")])
        self.assertEqual(list(store.cached), ["child", "arrival"])

    def test_same_timestamp_requests_are_served_before_any_override(self):
        trace = _trace([(0, ["A"]), (1000, ["B"]),
                        (2000, ["C", "C"]), (3000, ["A"])])
        at_group_two = []

        def run_with(choice):
            requests = []

            def decision(candidates, scores, original, timestamp, group, arriving):
                if group == 2 and not at_group_two:
                    at_group_two.append(tuple(candidates))
                if group == 2 and candidates[0] == "B":
                    return candidates.index(choice)
                return None

            result = _run(trace, l2_override_hook=decision,
                          l2_request_hook=lambda ids, prefix, hits, present, timestamp, group, measured:
                          requests.append((group, ids[0], tuple(hits))))
            return result, requests

        keep_a, first = run_with("B")
        drop_a, second = run_with("A")
        self.assertEqual(first[:4], second[:4])
        self.assertEqual([(g, s) for g, s, _ in first[:4]],
                         [(0, "A"), (1, "B"), (2, "C"), (2, "C")])
        self.assertEqual(first[2:4], [(2, "C", ()), (2, "C", ())])
        self.assertEqual(keep_a.l1_avoided_tokens, drop_a.l1_avoided_tokens)
        self.assertNotEqual(keep_a.l2_avoided_tokens, drop_a.l2_avoided_tokens)


class SelectionAndRewardTests(unittest.TestCase):
    def test_window_excludes_the_current_group_and_includes_exact_600_seconds(self):
        trace = _trace([(2000, ["A"]), (602000, ["A"]),
                        (602001, ["A"])])
        reward = WindowReward(trace, 2000)
        for group, timestamp in enumerate((2000, 602000, 602001)):
            reward.on_request(("A",), 0, (0,), (0,), timestamp, group, True)
        self.assertEqual((reward.q600, reward.qend), (512, 1024))
        self.assertEqual((reward.l2_q600, reward.l2_qend), (512, 1024))
        self.assertEqual((reward.requests600, reward.requests_end), (1, 2))
        self.assertEqual((reward.requested600, reward.requested_end), (512, 1024))

    def test_exact_h_label_is_clipped_but_count_includes_it(self):
        from persistent_kv_admission.decisionpop import _Labeller

        trace = _trace([(0, ["A"]), (600000, ["A"]), (600001, ["B"])])
        label = _Labeller(trace, 600.0)
        self.assertEqual(label("A", 0), (600000, 1))
        self.assertEqual(label("B", 0), (600001, 0))
        scores = [(1.0, 5.0), (1.0, 2.0), (0.0, 9.0)]
        picks = selectors(scores, [5, 2, 9], [3, 2, 1],
                          [600000, 600001, 1000], [1, 0, 3])
        self.assertEqual(picks, {
            "learned": 2, "next_use": 1, "count": 1, "lru": 1, "lfu": 2,
        })
        self.assertEqual(label_tie_sets([600000, 600001, 1000], [1, 0, 3]),
                         {"next_use": [0, 1], "count": [1]})
        self.assertEqual(selectors([(1, 0), (1, 0)], [0, 0], [0, 0],
                                   [600000, 600000], [0, 0]),
                         dict.fromkeys(("learned", "next_use", "count", "lru", "lfu"), 0))

    def test_hash_selection_uses_complete_identity_and_ignores_labels_and_scores(self):
        decisions = 12
        size = decisions * 2
        blocks = np.repeat(np.arange(decisions), 2)
        population = DecisionPopulation(
            timestamp_ms=np.repeat(np.arange(decisions, dtype=float) * 1000, 2),
            trace_group_index=blocks.copy(),
            decision_ordinal=np.zeros(size, dtype=np.int64),
            state_index=np.arange(size, dtype=np.int32),
            features=np.zeros((size, len(FEATURE_NAMES)), dtype=np.float64),
            next_use_delta_ms=np.full(size, 10.0),
            count_within_h=np.ones(size, dtype=np.int64),
            group=blocks.copy(),
            victim=np.tile([1, 0], decisions),
            arriving=np.tile([1, 0], decisions),
            arm_score=np.zeros(size),
            arm_tiebreak=np.zeros(size),
            horizon_seconds=600.0, window="test", seed=7,
            decisions_offered=decisions, decisions_eligible=decisions,
            max_decisions=40000,
        )
        lineage = "synthetic/0.0025x1/pi0/seed0"
        chosen = select_decisions(population, lineage, count=8)
        expected = sorted(range(decisions), key=lambda g: decision_hash(lineage, g, 0))[:8]
        self.assertEqual([row["group_index"] for row in chosen], expected)
        self.assertEqual([row["hash_rank"] for row in chosen], list(range(8)))
        population.next_use_delta_ms[:] = np.arange(size) * 1000.0
        population.count_within_h[:] = np.arange(size)[::-1]
        population.arm_score[:] = np.arange(size)[::-1]
        population.victim[:] = np.tile([0, 1], decisions)
        changed = select_decisions(population, lineage, count=8)
        self.assertEqual([(r["group_index"], r["selection_hash"]) for r in chosen],
                         [(r["group_index"], r["selection_hash"]) for r in changed])
        self.assertNotEqual(chosen[0]["victim_index"], changed[0]["victim_index"])
        self.assertNotEqual(continuation_seed(lineage, 3, 0, 1),
                            continuation_seed(lineage, 3, 0, 2))
        self.assertEqual(continuation_seed(lineage, 3, 0, 1),
                         continuation_seed(lineage, 3, 0, 1))
        with self.assertRaises(ValueError):
            continuation_seed(lineage, 3, 0, 0)

    def test_model_hash_mismatch_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.json"
            path.write_bytes(b"untrusted changed model")
            with self.assertRaisesRegex(ValueError, "ranker hash mismatch"):
                deserialize_ranker(path, expected_sha256="0" * 64)

    def test_population_source_hash_mismatch_rejects_even_a_readable_archive(self):
        population = DecisionPopulation(
            timestamp_ms=np.array([1000.0, 1000.0]),
            trace_group_index=np.array([1, 1]),
            decision_ordinal=np.array([0, 0]),
            state_index=np.array([0, 1]),
            features=np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64),
            next_use_delta_ms=np.array([10.0, 20.0]),
            count_within_h=np.array([1, 0]),
            group=np.array([0, 0]), victim=np.array([1, 0]),
            arriving=np.array([1, 0]),
            arm_score=np.array([0.0, 1.0]),
            arm_tiebreak=np.array([0.0, 0.0]),
            horizon_seconds=600.0, window="test", seed=0,
            decisions_offered=1, decisions_eligible=1, max_decisions=40000,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.npz"
            expected = population.save(path)
            self.assertEqual(load_population_verified(path, expected).decisions, 1)
            with path.open("ab") as handle:
                handle.write(b"changed-source")
            with self.assertRaisesRegex(ValueError, "population hash mismatch"):
                load_population_verified(path, expected)

    def test_runner_rejects_changed_source_before_replay(self):
        runner = _runner_module()
        self.assertIn("src/persistent_kv_admission/onpolicy.py", runner._source_tree())
        self.assertIn("src/persistent_kv_admission/temporal.py", runner._source_tree())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "persistent_kv_admission" / "marker.py"
            source.parent.mkdir(parents=True)
            source.write_text("original source\n")
            plan = root / "plan.md"
            plan.write_text("frozen plan\n")
            config = root / "config.json"
            config.write_text("{}\n")
            output = root / "run"
            output.mkdir()
            with (patch.object(runner, "REPOSITORY", root),
                  patch.object(runner, "SOURCE_FILES", ()),
                  patch.object(runner, "PLAN", plan),
                  patch.object(runner, "ONPOLICY_CONFIG", config),
                  patch.object(runner, "require_committed_source", return_value="fixed-commit"),
                  patch.object(runner, "protected_diff_hashes", return_value={})):
                manifest = {
                    "code_commit": "fixed-commit",
                    "source_manifest": runner.source_manifest(),
                    "plan_sha256": runner.sha256_path(plan),
                    "onpolicy_config_sha256": runner.sha256_path(config),
                    "protected_diff_sha256": {},
                    "lineages": [{} for _ in range(40)],
                    "selected_count": 320,
                }
                manifest_path = output / "selection_manifest.json"
                runner.atomic_json(manifest_path, manifest)
                runner.atomic_json(output / "prepare_status.json", {
                    "complete": True,
                    "manifest_sha256": runner.sha256_path(manifest_path),
                })
                runner._load_manifest(output)
                source.write_text("tampered source\n")
                with self.assertRaisesRegex(AssertionError, "source hashes changed"):
                    runner._load_manifest(output)
                source.write_text("original source\n")
                (source.parent / "new_dependency.py").write_text("new source\n")
                with self.assertRaisesRegex(AssertionError, "source hashes changed"):
                    runner._load_manifest(output)

    def test_reported_rank_and_regret_summary_use_eviction_value_orientation(self):
        runner = _runner_module()
        self.assertAlmostEqual(runner._spearman([0, 1, 2], [3, 2, 1]), 1.0)
        self.assertAlmostEqual(runner._spearman([0, 1, 2], [1, 2, 3]), -1.0)
        self.assertIsNone(runner._spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(runner._spearman([0, 1, 2], [3, 3, 3]))
        rows = [
            {"selector": "next_use", "regret_tokens": value,
             "constant_q": value == 0, "q_range_tokens": 1024,
             "spearman": None if value == 0 else 1.0}
            for value in (0, 512, 1024)
        ]
        summary = runner._summarize_regrets(rows, ("selector",))[0]
        self.assertEqual(summary["decisions"], 3)
        self.assertEqual(summary["mean_regret_tokens"], 512)
        self.assertEqual(summary["median_regret_tokens"], 512)
        self.assertAlmostEqual(summary["p90_regret_tokens"], 921.6)
        self.assertEqual(summary["max_regret_tokens"], 1024)
        self.assertAlmostEqual(summary["zero_regret_rate"], 1 / 3)
        self.assertAlmostEqual(summary["regret_ge_512_rate"], 2 / 3)


@unittest.skipUnless(hasattr(os, "fork"), "POSIX fork is required")
class ForkReplayTests(unittest.TestCase):
    def setUp(self):
        self.trace = _trace([(0, ["A"]), (1000, ["B"]), (2000, ["C"]),
                             (3000, ["A"]), (4000, ["D"])])
        state_index = {state_id: index for index, state_id in enumerate(sorted(self.trace.states))}
        self.selected = _fork_fixture_selected(self.trace, state_index)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def test_actual_fork_matches_parent_and_alternate_changes_suffix(self):
        baseline = run_two_tier(
            self.trace, "lru", 512, 512, "learned", bytes_per_token=1,
            l2_eviction="sampled", l2_sample_width=16, l2_seed=7,
            l2_scorer=_SyntheticScorer(self.trace),
        )
        parent, snapshots = _fork_replay(self.trace, self.selected, self.temporary.name)
        self.assertEqual(parent.as_row(), baseline.as_row())
        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual(snapshot["metadata"]["candidate_count"], 2)
        self.assertEqual(len(snapshot["branches"]), 6)  # two actions, three paired streams
        actual = self.selected["victim_index"]
        for replicate in (0, 1, 2):
            pair = {row["action_index"]: row for row in snapshot["branches"]
                    if row["replicate"] == replicate}
            self.assertEqual(set(pair), {0, 1})
            self.assertEqual(pair[actual]["reward"]["requests_end"], 2)
            self.assertEqual(pair[1 - actual]["reward"]["q600"]
                             - pair[actual]["reward"]["q600"], 512)
            self.assertEqual(pair[1 - actual]["reward"]["l2_q600"]
                             - pair[actual]["reward"]["l2_q600"], 512)
        captured = next(row for row in snapshot["branches"]
                        if row["replicate"] == 0 and row["action_index"] == actual)
        self.assertEqual(captured["reward"], snapshot["parent_reward"].row())
        self.assertEqual(captured["end_result"], parent.as_row())
        self.assertEqual(len(captured["terminal_state_digest"]), 64)

    def test_fork_preserves_rng_stream_through_later_sampled_draws(self):
        trace = _trace([(index * 1000, [state_id]) for index, state_id in
                        enumerate("ABCDEFGHA")])
        state_index = {state_id: index for index, state_id in enumerate(sorted(trace.states))}
        selected = _fork_fixture_selected(
            trace, state_index, group_to_capture=3,
            l2_capacity=1024, sample_width=1,
        )

        class Probe:
            def __init__(self):
                self.store = None
                self.states = []

            def attach(self, l1, l2, counters):
                self.store = l2

            def __call__(self, candidates, scores, victim, timestamp, group, arriving):
                if group >= 3:
                    self.states.append((group, self.store.rng.getstate()))
                return None

        probe = Probe()
        baseline = run_two_tier(
            trace, "lru", 512, 1024, "learned", bytes_per_token=1,
            l2_eviction="sampled", l2_sample_width=1, l2_seed=7,
            l2_scorer=_SyntheticScorer(trace), l2_override_hook=probe,
        )
        self.assertGreater(len(probe.states), 1)
        self.assertNotEqual(probe.states[0][1], probe.states[1][1])
        parent, snapshots = _fork_replay(
            trace, selected, self.temporary.name,
            l2_capacity=1024, sample_width=1,
        )
        self.assertEqual(parent.as_row(), baseline.as_row())
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(snapshots[0]["branches"]), 6)
        actual = selected["victim_index"]
        captured = next(row for row in snapshots[0]["branches"]
                        if row["replicate"] == 0 and row["action_index"] == actual)
        self.assertEqual(captured["reward"], snapshots[0]["parent_reward"].row())

    def test_snapshot_mismatch_is_rejected_before_fork(self):
        broken = dict(self.selected, candidate_indices=[999, 998])
        with self.assertRaisesRegex(AssertionError, "candidate identities/order mismatch"):
            _fork_replay(self.trace, broken, self.temporary.name)
        self.assertEqual(list(Path(self.temporary.name).glob("*.json")), [])

    def test_completed_branches_resume_without_reforking_or_changing_outputs(self):
        first_result, first_snapshots = _fork_replay(
            self.trace, self.selected, self.temporary.name
        )
        branch_paths = sorted(Path(self.temporary.name).glob("*.json"))
        self.assertEqual(len(branch_paths), 6)
        original_bytes = {path.name: path.read_bytes() for path in branch_paths}
        with patch("persistent_kv_admission.counterfactual.os.fork",
                   side_effect=AssertionError("completed branch forked again")):
            second_result, second_snapshots = _fork_replay(
                self.trace, self.selected, self.temporary.name
            )
        self.assertEqual(first_result.as_row(), second_result.as_row())
        self.assertEqual(first_snapshots[0]["branches"], second_snapshots[0]["branches"])
        self.assertEqual(first_snapshots[0]["parent_reward"].row(),
                         second_snapshots[0]["parent_reward"].row())
        self.assertEqual(original_bytes,
                         {path.name: path.read_bytes() for path in branch_paths})

    def test_tampered_completed_branch_is_rejected(self):
        _fork_replay(self.trace, self.selected, self.temporary.name)
        branch = sorted(Path(self.temporary.name).glob("*.json"))[0]
        payload = json.loads(branch.read_text())
        payload["reward"]["q600"] += 1
        branch.write_text(json.dumps(payload))
        with self.assertRaisesRegex(AssertionError, "branch output content hash mismatch"):
            _fork_replay(self.trace, self.selected, self.temporary.name)

    def test_child_failure_reaches_parent_without_hanging(self):
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "branch child failed"):
            _fork_replay(self.trace, self.selected, self.temporary.name,
                         fail_in_child_at=3000)
        self.assertLess(time.monotonic() - started, 10.0)
        details = list(Path(self.temporary.name).glob("*.error.json"))
        self.assertEqual(len(details), 1)
        self.assertIn("injected synthetic child failure", details[0].read_text())


if __name__ == "__main__":
    unittest.main()
