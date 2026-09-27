"""Constructed checks for the fixed E/Z residence diagnostic."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from persistent_kv_admission.counterfactual_randomness import derive_stream_seed
from persistent_kv_admission.counterfactual_residence import (
    ResidenceForkController, ResidenceTelemetry, fixed_pair, pair_analysis,
)
from persistent_kv_admission.trace import Request, StateMeta, Trace
from persistent_kv_admission.twotier import run_two_tier
from test_counterfactual import _SyntheticScorer, _fork_fixture_selected, _trace


def _runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_counterfactual_residence.py"
    spec = importlib.util.spec_from_file_location("_residence_runner_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _branch(rows, q600, qend, first=None, censored=True):
    return {"reward": {"q600": q600, "qend": qend},
            "telemetry": {"requests": rows, "removals": [],
                          "summary": {"censored_600": censored,
                                      "first_survivor_removal": first}}}


class SelectionAndPairTests(unittest.TestCase):
    def test_opposite_tie_breaks_use_indices_and_no_q(self):
        meta = {"candidate_count": 4, "label_ties": {"count": [0, 1, 2, 3],
                "next_use": [0, 1, 2, 3]}, "count_within_h": [0] * 4,
                "last_group": [1, 1, 8, 8], "selectors": {"next_use": 0}}
        self.assertEqual(fixed_pair(meta), (0, 3))
        changed = dict(meta, count_within_h=[0, 1, 0, 0])
        with self.assertRaisesRegex(AssertionError, "zero own reuse"):
            fixed_pair(changed)
        changed = dict(meta, label_ties={"count": [0, 1], "next_use": [0, 2]})
        with self.assertRaisesRegex(AssertionError, "tie sets"):
            fixed_pair(changed)

    def test_first_hit_identity_can_precede_first_reward_difference(self):
        # First request exchanges equal-token block identities; second changes Q.
        e = [[10, 3, 1000.0, 0, [1], [1], 0, 512, 0, 1],
             [11, 4, 2000.0, 0, [], [], 0, 0, 0, 0]]
        z = [[10, 3, 1000.0, 0, [2], [2], 0, 512, 0, 2],
             [11, 4, 2000.0, 0, [2], [2], 0, 256, 0, 2]]
        result, events = pair_analysis(_branch(e, 512, 512), _branch(z, 768, 768),
                                       start_ms=0, start_group=2)
        self.assertEqual(result["first_hit"]["order"], 10)
        self.assertEqual(result["first_reward"]["order"], 11)
        self.assertEqual(result["delta_q600"], 256)
        self.assertEqual([event["cumulative_delta_q600"] for event in events], [0, 256])
        self.assertEqual(result["rejoining"], "unmeasured")

    def test_pair_alignment_l1_invariant_and_trace_end_own_reuse(self):
        e = [[1, 2, 1000.0, 0, [], [], 0, 0, 0, 1],
             [2, 3, 700000.0, 0, [1], [1], 0, 512, 1, 0]]
        z = [[1, 2, 1000.0, 0, [], [], 0, 0, 0, 2],
             [2, 3, 700000.0, 0, [], [], 0, 0, 1, 0]]
        result, _ = pair_analysis(_branch(e, 0, 512), _branch(z, 0, 0),
                                  start_ms=0, start_group=1)
        self.assertEqual(result["delta_q600"], 0)
        self.assertEqual(result["delta_qend"], -512)
        self.assertIsNone(result["first_hit600"])
        bad = [row[:] for row in z]
        bad[0][0] = 9
        with self.assertRaisesRegex(AssertionError, "identity"):
            pair_analysis(_branch(e, 0, 512), _branch(bad, 0, 0), start_ms=0, start_group=1)
        bad = [row[:] for row in z]
        bad[0][3] = 1
        with self.assertRaisesRegex(AssertionError, "L1"):
            pair_analysis(_branch(e, 0, 512), _branch(bad, 0, 0), start_ms=0, start_group=1)


class TelemetryTests(unittest.TestCase):
    def test_same_timestamp_excluded_and_first_residence_not_readmission(self):
        trace = _trace([(1000, ["A", "B"]), (2000, ["C"]), (700000, ["A"])])
        index = {s: i for i, s in enumerate(sorted(trace.states))}
        telemetry = ResidenceTelemetry(trace, 1000, 0, ("A", "B"), "B", index, 1)
        cache = type("Cache", (), {"cached": {"A": None, "B": None}})()
        telemetry.on_request(cache, 0, ("A",), 0, [0], [0], 1000, 0, True)
        telemetry.on_request(cache, 2, ("C",), 0, [], [], 2000, 1, True)
        telemetry.on_removal("B", "evicted", 3000, 2, 7)
        telemetry.on_removal("B", "promoted", 8000, 3, -1)
        self.assertEqual(len(telemetry.requests), 1)
        reward = {"q600": 0, "qend": 0, "l2_q600": 0, "l2_qend": 0,
                  "requests600": 1, "requests_end": 1}
        summary = telemetry.summary(reward)
        self.assertEqual(summary["first_residence_ms"], 2000)
        self.assertEqual(summary["first_residence_groups"], 2)
        self.assertEqual(summary["resident_byte_seconds_600"], 1024)
        self.assertEqual(summary["first_survivor_removal"][3], "evicted")
        self.assertFalse(summary["censored_600"])

    def test_initial_overflow_rounds_stop_at_next_offer_same_timestamp(self):
        trace = _trace([(1000, ["A", "B", "C"])])
        index = {state: i for i, state in enumerate(sorted(trace.states))}
        t = ResidenceTelemetry(trace, 1000, 0, ("A", "B"), "B", index, 1)
        t.on_decision(["A", "B"], [], 0, 1000, 0, 0)
        t.on_victim("C", 1000, 0)
        t.on_decision(["B", "C"], [], 0, 1000, 0, 0)
        t.on_removal("B", "evicted", 1000, 0, 2)
        self.assertEqual(t.immediate_overflow_rounds, 2)
        summary = t.summary({"q600": 0, "qend": 0, "l2_q600": 0,
                             "l2_qend": 0, "requests600": 0, "requests_end": 0})
        self.assertTrue(summary["survived_immediate_overflow"])
        self.assertEqual(summary["first_residence_ms"], 0)

    def test_focal_request_inside_horizon_fails_but_after_horizon_is_allowed(self):
        trace = _trace([(0, ["B"]), (1000, ["A"]), (700000, ["A"])])
        cache = type("Cache", (), {"cached": {"A": None}})()
        t = ResidenceTelemetry(trace, 0, 0, ("A", "B"), "B", {"A": 0, "B": 1}, 1)
        with self.assertRaisesRegex(AssertionError, "zero-own-reuse"):
            t.on_request(cache, 1, ("A",), 0, [], [], 1000, 1, True)
        t = ResidenceTelemetry(trace, 0, 0, ("A", "B"), "B", {"A": 0, "B": 1}, 1)
        t.on_request(cache, 2, ("A",), 0, [], [], 700000, 2, True)
        self.assertEqual(t.requests_end, 1)
        self.assertEqual(t.requests600, 0)


class RunnerIntegrityTests(unittest.TestCase):
    def test_pinned_old_paper_checksum_rejects_mutated_source(self):
        runner = _runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results/paper/counterfactual_randomness_001"
            path.mkdir(parents=True)
            (path / "run_config.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "paper config SHA256"):
                runner._old_context(Path(directory))

    def test_old_branch_checksum_failure_precedes_payload_trust(self):
        runner = _runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"slug": "one", "selected": [{"selection_hash": "abc"}]}
            path = root / runner.OLD_RAW / "branches/one/abc__r1__a0.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"selection_hash": "abc", "replicate": 1, "action_index": 0}))
            with self.assertRaisesRegex(AssertionError, "SHA256"):
                runner._old_branch(root, row, 1, 0, "0" * 64)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_constructed_fork_two_actions_same_rng_and_reward_accounting(self):
        # A/B are never requested within H after the C offer; both E/Z forks
        # continue the same fixed future trace, including a trace-end A reuse.
        trace = _trace([(0, ["A"]), (1000, ["B"]), (2000, ["C"]),
                        (3000, ["D"]), (700000, ["A"])])
        state_index = {state_id: i for i, state_id in enumerate(sorted(trace.states))}
        selected = _fork_fixture_selected(trace, state_index, group_to_capture=2)
        candidates = [sorted(trace.states)[index] for index in selected["candidate_indices"]]
        self.assertEqual(set(candidates), {"A", "B"})
        pair = (candidates.index("A"), candidates.index("B"))
        seeds = {s: derive_stream_seed(selected["lineage"], selected["group_index"],
                                       selected["ordinal"], s) for s in range(1, 17)}
        with tempfile.TemporaryDirectory() as directory:
            scorer = _SyntheticScorer(trace)
            controller = ResidenceForkController(trace, scorer, state_index, selected,
                                                 Path(directory), "constructed-residence-v1",
                                                 pair, seeds, smoke=True, bytes_per_token=1)
            try:
                result = run_two_tier(trace, "lru", 512, 512, "learned", bytes_per_token=1,
                                      l2_eviction="sampled", l2_sample_width=16, l2_seed=7,
                                      l2_scorer=scorer, l2_override_hook=controller,
                                      l2_request_hook=controller.on_request,
                                      l2_decision_hook=controller.on_decision,
                                      l2_removal_hook=controller.on_removal,
                                      victim_hook=controller.on_victim)
                if controller.is_child:
                    controller.finish_child(result)
                    os._exit(0)
            except BaseException as error:
                if controller.is_child:
                    controller.fail_child(error)
                    os._exit(2)
                raise
            self.assertEqual(len(controller.snapshots), 1)
            branches = controller.snapshots[0]["branches"]
            self.assertEqual({b["replicate"] for b in branches}, {1})
            self.assertEqual({b["action_index"] for b in branches}, set(pair))
            for branch in branches:
                self.assertTrue(branch["telemetry"]["summary"]["immediate_overflow_rounds"] >= 1)
                self.assertEqual(branch["reward"]["requests_end"],
                                 len(branch["telemetry"]["requests"]))
            pair_analysis(branches[0], branches[1], start_ms=2000, start_group=2)


if __name__ == "__main__":
    unittest.main()
