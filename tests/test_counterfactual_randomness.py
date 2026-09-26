"""Adversarial constructed checks for fresh-stream counterfactual cross-fitting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from persistent_kv_admission.counterfactual import (
    _canonical_bytes,
    continuation_seed,
    label_tie_sets,
)
from persistent_kv_admission.counterfactual_randomness import (
    NAMESPACE,
    RandomnessForkController,
    crossfit_analysis,
    derive_stream_seed,
)
from persistent_kv_admission.twotier import run_two_tier
from persistent_kv_admission.onpolicy import sha256_path

from test_counterfactual import (
    _SyntheticScorer,
    _fork_fixture_selected,
    _fork_replay,
    _trace,
)


def _matrix(a_rows, b_rows):
    """Eight independent stream vectors in each preassigned fold."""
    if len(a_rows) != 8 or len(b_rows) != 8:
        raise ValueError("test fixture requires eight rows per fold")
    return {stream_id: list(row) for stream_id, row in enumerate(a_rows + b_rows, 1)}


def _repeated(a, b):
    return _matrix([a] * 8, [b] * 8)


def _analyse(q600, *, qend=None, actual=0, last_group=None, ties=None, fixed=None):
    width = len(q600[1])
    if qend is None:
        qend = q600
    if last_group is None:
        last_group = list(range(width))
    if ties is None:
        ties = {"next_use": list(range(width)), "count": list(range(width))}
    if fixed is None:
        fixed = {"next_use": ties["next_use"][0] if ties["next_use"] else 0,
                 "count": ties["count"][0] if ties["count"] else 0,
                 "lru": 0, "lfu": 0}
    return crossfit_analysis(q600, qend, actual, last_group, ties, fixed)


def _direction(result, selector, direction):
    return next(row for row in result["directions"]
                if row["selector"] == selector and row["direction"] == direction)


def _state(result, selector):
    return next(row for row in result["state"] if row["selector"] == selector)


def _runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_counterfactual_randomness.py"
    spec = importlib.util.spec_from_file_location("_randomness_runner_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamAndCrossfitTests(unittest.TestCase):
    def test_seed_uses_fresh_namespace_full_digest_and_complete_identity(self):
        lineage = "synthetic/seed7"
        expected = int.from_bytes(hashlib.sha256(_canonical_bytes(
            [NAMESPACE, lineage, 3, 0, 1]
        )).digest(), "big")
        self.assertEqual(derive_stream_seed(lineage, 3, 0, 1), expected)
        self.assertEqual(derive_stream_seed(lineage, 3, 0, 1), expected)
        self.assertGreater(expected.bit_length(), 64)
        seeds = {derive_stream_seed(lineage, 3, 0, r) for r in range(1, 17)}
        self.assertEqual(len(seeds), 16)
        self.assertNotIn(continuation_seed(lineage, 3, 0, 1), seeds)
        self.assertNotIn(continuation_seed(lineage, 3, 0, 2), seeds)
        self.assertNotEqual(expected, derive_stream_seed(lineage, 3, 1, 1))
        self.assertNotEqual(expected, derive_stream_seed(lineage, 4, 0, 1))
        for bad in (0, 17, True, 1.0):
            with self.subTest(stream_id=bad), self.assertRaises(ValueError):
                derive_stream_seed(lineage, 3, 0, bad)

    def test_winner_curse_keeps_negative_heldout_gain_and_no_holdout_max(self):
        # The training winner is worse than the original action on every B stream.
        result = _analyse(_repeated([10, 11], [10, 9]))
        forward = _direction(result, "unrestricted", "A_to_B")
        reverse = _direction(result, "unrestricted", "B_to_A")
        self.assertEqual(forward["selected_index"], 1)
        self.assertEqual(forward["train_mean_gain_tokens"], 1)
        self.assertEqual(forward["heldout_mean_gain_tokens"], -1)
        self.assertEqual((forward["heldout_sd_tokens"], forward["heldout_se_tokens"]), (0, 0))
        self.assertEqual((forward["heldout_ci95_low_tokens"],
                          forward["heldout_ci95_high_tokens"]), (-1, -1))
        self.assertEqual(reverse["selected_index"], 0)
        self.assertEqual(reverse["heldout_mean_gain_tokens"], 0)
        self.assertEqual(_state(result, "unrestricted")["symmetric_heldout_mean_gain_tokens"], -0.5)
        heldout = [row for row in result["per_stream"]
                   if row["selector"] == "unrestricted" and row["direction"] == "A_to_B"]
        self.assertEqual(len(heldout), 8)
        self.assertEqual({row["stream_id"] for row in heldout}, set(range(9, 17)))
        self.assertEqual({row["gain_tokens"] for row in heldout}, {-1})
        self.assertEqual({row["heldout_hindsight_max_gain_tokens"] for row in heldout}, {0})

    def test_maximizes_mean_q_not_streamwise_wins_or_maximum(self):
        a = [[0, 10, 0] for _ in range(7)] + [[0, 10, 100]]
        result = _analyse(_matrix(a, [[0, 10, 0]] * 8))
        forward = _direction(result, "unrestricted", "A_to_B")
        self.assertEqual(forward["selected_index"], 2)
        self.assertEqual(forward["train_mean_gain_tokens"], 12.5)
        self.assertEqual(forward["heldout_mean_gain_tokens"], 0)
        self.assertEqual(_direction(result, "unrestricted", "B_to_A")["selected_index"], 1)
        # A single spectacular stream must not beat a larger eight-stream mean.
        a = [[0, 30, 0] for _ in range(7)] + [[0, 30, 100]]
        result = _analyse(_matrix(a, [[0, 30, 0]] * 8))
        self.assertEqual(_direction(result, "unrestricted", "A_to_B")["selected_index"], 1)
        self.assertEqual(_direction(result, "unrestricted", "A_to_B")["train_mean_gain_tokens"], 30)

    def test_streamwise_hindsight_max_exceeds_best_actionwise_mean(self):
        result = _analyse(_repeated([10, 0], [0, 10]))
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["mean_stream_hindsight_max_q600_tokens"], 10)
        self.assertEqual(diagnostics["max_actionwise_mean_q600_tokens"], 5)
        self.assertEqual(diagnostics["actual_mean_q600_tokens"], 5)
        self.assertEqual(diagnostics["hindsight_jensen_gap_tokens"], 5)
        self.assertEqual(diagnostics["mean_stream_hindsight_regret_vs_actual_tokens"], 5)
        self.assertEqual(diagnostics["fitted_all_stream_regret_vs_actual_tokens"], 0)

    def test_evaluation_fold_cannot_change_that_directions_selected_action(self):
        original = _repeated([10, 11, 12], [10, 8, 9])
        changed = dict(original)
        for stream_id in range(9, 17):
            changed[stream_id] = [0, 1000, 1]
        first = _analyse(original)
        second = _analyse(changed)
        self.assertEqual(_direction(first, "unrestricted", "A_to_B")["selected_index"], 2)
        self.assertEqual(_direction(second, "unrestricted", "A_to_B")["selected_index"], 2)
        self.assertNotEqual(_direction(first, "unrestricted", "B_to_A")["selected_index"],
                            _direction(second, "unrestricted", "B_to_A")["selected_index"])

    def test_fold_swap_exchanges_directions_and_preserves_symmetric_mean(self):
        q = _repeated([10, 11], [10, 9])
        swapped = {r: q[r + 8] if r <= 8 else q[r - 8] for r in range(1, 17)}
        first = _analyse(q)
        second = _analyse(swapped)
        self.assertEqual(_direction(first, "unrestricted", "A_to_B")["heldout_mean_gain_tokens"],
                         _direction(second, "unrestricted", "B_to_A")["heldout_mean_gain_tokens"])
        self.assertEqual(_state(first, "unrestricted")["symmetric_heldout_mean_gain_tokens"],
                         _state(second, "unrestricted")["symmetric_heldout_mean_gain_tokens"])

    def test_tie_break_uses_last_group_then_draw_index_and_constant_q_remains(self):
        q = _repeated([5, 5, 5], [5, 5, 5])
        result = _analyse(q, actual=0, last_group=[9, 1, 1])
        for direction in ("A_to_B", "B_to_A"):
            row = _direction(result, "unrestricted", direction)
            self.assertEqual(row["selected_index"], 1)
            self.assertEqual(row["training_tie_count"], 3)
            self.assertEqual(row["heldout_mean_gain_tokens"], 0)
        self.assertEqual(len(result["hindsight"]), 16)
        self.assertTrue(all(row["constant_q"] for row in result["hindsight"]))

    def test_exact_h_label_ties_can_select_different_actions(self):
        # At exactly H, next-use is clipped while count still includes reuse.
        ties = label_tie_sets([600000, 600001, 1000], [1, 0, 3])
        self.assertEqual(ties, {"next_use": [0, 1], "count": [1]})
        q = _repeated([100, 50, 0], [100, 50, 0])
        result = _analyse(q, ties=ties, fixed={"next_use": 0, "count": 1,
                                                "lru": 0, "lfu": 0})
        self.assertEqual(_direction(result, "next_use_tie", "A_to_B")["selected_index"], 0)
        self.assertEqual(_direction(result, "count_tie", "A_to_B")["selected_index"], 1)
        self.assertEqual(_direction(result, "count_tie", "A_to_B")["heldout_mean_gain_tokens"], -50)

    def test_restricted_tie_and_trace_end_evaluate_q600_selected_action(self):
        q600 = _repeated([0, 1, 100], [0, 0, 2])
        # Qend strongly prefers the actual action; Q600 must still select 2.
        qend = _repeated([1000, 1001, 100], [1000, 1005, 3])
        ties = {"next_use": [0, 1], "count": [0, 1]}
        result = _analyse(q600, qend=qend, ties=ties)
        full = _direction(result, "unrestricted", "A_to_B")
        tied = _direction(result, "next_use_tie", "A_to_B")
        self.assertEqual((full["selected_index"], tied["selected_index"]), (2, 1))
        self.assertEqual((full["heldout_mean_gain_tokens"], tied["heldout_mean_gain_tokens"]), (2, 0))
        self.assertEqual((full["heldout_end_mean_gain_tokens"], tied["heldout_end_mean_gain_tokens"]), (-997, 5))
        modified_end = _repeated([1000, 5000, 10000], [1000, 5000, 10000])
        result2 = _analyse(q600, qend=modified_end, ties=ties)
        self.assertEqual(_direction(result2, "unrestricted", "A_to_B")["selected_index"], 2)
        self.assertEqual(_direction(result2, "next_use_tie", "A_to_B")["selected_index"], 1)

    def test_missing_stream_wrong_width_invalid_q_or_ties_fail_closed(self):
        q = _repeated([1, 2], [2, 1])
        bad_cases = []
        missing = dict(q)
        missing.pop(16)
        bad_cases.append((missing, {}, None, None, None))
        extra = dict(q)
        extra[17] = [1, 2]
        bad_cases.append((extra, {}, None, None, None))
        captured = dict(q)
        captured[0] = [1, 2]
        bad_cases.append((captured, {}, None, None, None))
        wrong_width = dict(q)
        wrong_width[2] = [1]
        bad_cases.append((wrong_width, {}, None, None, None))
        for value in (-1, float("nan"), 1.5, True):
            changed = dict(q)
            changed[1] = [value, 2]
            bad_cases.append((changed, {}, None, None, None))
        bad_cases.extend([
            (q, {"actual": 2}, None, None, None),
            (q, {}, {"next_use": [], "count": [0]}, None, None),
            (q, {}, {"next_use": [0, 0], "count": [0]}, None, None),
            (q, {}, {"next_use": [2], "count": [0]}, None, None),
            (q, {}, {"next_use": [0], "count": [0]},
             {"next_use": 1, "count": 0, "lru": 0, "lfu": 0}, None),
        ])
        for index, (primary, opts, ties, fixed, _) in enumerate(bad_cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                _analyse(primary, actual=opts.get("actual", 0), ties=ties, fixed=fixed)
        bad_end = dict(q)
        bad_end[8] = [1, -1]
        with self.assertRaises(ValueError):
            _analyse(q, qend=bad_end)


class RunnerIntegrityTests(unittest.TestCase):
    def test_source_commit_gate_rejects_changed_code_and_plan_mismatch(self):
        runner = _runner_module()
        with patch.object(runner, "_git", return_value="scripts/run_counterfactual_randomness.py"):
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                runner.require_committed_source()
        with patch.object(runner, "_git", side_effect=["", "", "different"]):
            with self.assertRaisesRegex(RuntimeError, "plan commit mismatch"):
                runner.require_committed_source()

    def test_smoke_pairs_are_subset_of_full_and_marker_rejects_missing_or_tampered_branch(self):
        runner = _runner_module()
        row = {"slug": "synthetic", "selected": [{
            "candidate_indices": [3, 4], "victim_index": 1,
            "selection_hash": "synthetic-hash",
        }], "model_sha256": "model", "population_sha256": "population"}
        smoke = runner._expected_pairs(row, smoke=True)
        full = runner._expected_pairs(row, smoke=False)
        self.assertEqual(smoke, {(0, 1), (1, 0), (1, 1)})
        self.assertEqual(len(full), 33)
        self.assertTrue(smoke < full)
        self.assertEqual({pair for pair in full if pair[0] == 0}, {(0, 1)})
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            manifest_hash = "manifest"
            hashes = {}
            for (replicate, action), relative in runner._expected_relative_paths(row, smoke=True).items():
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = {"run_fingerprint": manifest_hash,
                           "selection_hash": "synthetic-hash",
                           "replicate": replicate, "action_index": action}
                payload = content | {"content_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest()}
                path.write_text(json.dumps(payload), encoding="utf-8")
                hashes[str(relative)] = sha256_path(path)
            marker = {"complete": True, "smoke": True, "slug": "synthetic",
                      "manifest_sha256": manifest_hash, "selection_hash": "synthetic-hash",
                      "candidate_indices": [3, 4], "model_sha256": "model",
                      "population_sha256": "population", "published_replay_checked": True,
                      "captured_parent_and_original_checked": True,
                      "expected_branches": 3, "actual_branches": 3,
                      "branch_sha256": hashes}
            marker_path = output_dir / "smoke_marker.json"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertEqual(runner._verify_marker(output_dir, row, manifest_hash, smoke=True), marker)
            one_path = output_dir / next(iter(hashes))
            original_bytes = one_path.read_bytes()
            one_path.unlink()
            with self.assertRaises((AssertionError, FileNotFoundError)):
                runner._verify_marker(output_dir, row, manifest_hash, smoke=True)
            one_path.write_bytes(original_bytes + b" ")
            with self.assertRaisesRegex(AssertionError, "branch hash mismatch"):
                runner._verify_marker(output_dir, row, manifest_hash, smoke=True)


@unittest.skipUnless(hasattr(os, "fork"), "POSIX fork is required")
class FreshForkTests(unittest.TestCase):
    def setUp(self):
        self.trace = _trace([(0, ["A"]), (1000, ["B"]), (2000, ["C"]),
                             (3000, ["A"]), (4000, ["D"])])
        self.state_index = {state_id: index for index, state_id in enumerate(sorted(self.trace.states))}
        self.selected = _fork_fixture_selected(self.trace, self.state_index)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def _replay(self, branch_dir, *, stream_seeds=None):
        scorer = _SyntheticScorer(self.trace)
        controller = RandomnessForkController(
            self.trace, scorer, self.state_index, [self.selected], branch_dir,
            "synthetic-randomness-v1", smoke=True, stream_seeds=stream_seeds,
        )
        try:
            result = run_two_tier(
                self.trace, "lru", 512, 512, "learned", bytes_per_token=1,
                l2_eviction="sampled", l2_sample_width=16, l2_seed=7,
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

    def test_captured_control_matches_original_parent_and_fresh_streams_pair_actions(self):
        old_dir = Path(self.temporary.name) / "old"
        new_dir = Path(self.temporary.name) / "fresh"
        old_dir.mkdir()
        new_dir.mkdir()
        old_parent, old_snapshots = _fork_replay(self.trace, self.selected, old_dir)
        seed_log = Path(self.temporary.name) / "reseed.log"
        expected_seed = derive_stream_seed(
            self.selected["lineage"], self.selected["group_index"],
            self.selected["ordinal"], 1,
        )
        original_seed = random.Random.seed

        def recording_seed(rng, value=None, version=2):
            if value == expected_seed:
                with seed_log.open("a", encoding="ascii") as handle:
                    handle.write(f"{value}\n")
            return original_seed(rng, value, version)

        with patch.object(random.Random, "seed", recording_seed):
            new_parent, snapshots = self._replay(new_dir)
        self.assertEqual(seed_log.read_text().splitlines(), [str(expected_seed)] * 2)
        self.assertEqual(new_parent.as_row(), old_parent.as_row())
        old_actual = next(row for row in old_snapshots[0]["branches"]
                          if row["replicate"] == 0 and row["action_index"] == self.selected["victim_index"])
        captured = next(row for row in snapshots[0]["branches"] if row["replicate"] == 0)
        self.assertEqual(captured["reward"], old_actual["reward"])
        self.assertEqual(captured["terminal_state_digest"], old_actual["terminal_state_digest"])
        self.assertEqual(captured["end_result"], old_actual["end_result"])
        self.assertEqual({(row["replicate"], row["action_index"])
                          for row in snapshots[0]["branches"]}, {(0, self.selected["victim_index"]), (1, 0), (1, 1)})
        self.assertEqual(len(list(new_dir.glob("*.json"))), 3)

    def test_resume_uses_verified_branch_bytes_and_tamper_fails(self):
        branch_dir = Path(self.temporary.name) / "resume"
        branch_dir.mkdir()
        _, first = self._replay(branch_dir)
        original = {path.name: path.read_bytes() for path in branch_dir.glob("*.json")}
        with patch("persistent_kv_admission.counterfactual_randomness.os.fork",
                   side_effect=AssertionError("completed branch reforked")):
            _, second = self._replay(branch_dir)
        self.assertEqual(first[0]["branches"], second[0]["branches"])
        self.assertEqual(original, {path.name: path.read_bytes() for path in branch_dir.glob("*.json")})
        path = next(branch_dir.glob("*.json"))
        payload = json.loads(path.read_text())
        payload["reward"]["q600"] += 1
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(AssertionError, "branch output content hash mismatch"):
            self._replay(branch_dir)

    def test_frozen_seed_mismatch_fails_before_branch(self):
        branch_dir = Path(self.temporary.name) / "wrong-seed"
        branch_dir.mkdir()
        wrong = {r: derive_stream_seed(self.selected["lineage"],
                                      self.selected["group_index"],
                                      self.selected["ordinal"], r)
                 for r in range(1, 17)}
        wrong[1] += 1
        with self.assertRaisesRegex(AssertionError, "frozen stream seeds"):
            self._replay(branch_dir, stream_seeds=wrong)
        self.assertEqual(list(branch_dir.glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
