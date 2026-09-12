import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.characterize import (
    longest_online_comparison_rows,
    replay_comparison_rows,
    replay_rows,
    trace_summary,
    write_csv,
)
from persistent_kv_admission.ranking import ranking_metrics
from persistent_kv_admission.replay import POLICIES, replay
from persistent_kv_admission.trace import load_mooncake_trace


class CharacterizationTests(unittest.TestCase):
    def make_trace(self):
        records = [
            {"timestamp": 0, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]},
            {"timestamp": 0, "input_length": 1024, "output_length": 1, "hash_ids": [1, 3]},
            {"timestamp": 1000, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]},
        ]
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "tiny.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return temporary, load_mooncake_trace(path)

    def test_trace_reconstructs_cumulative_prefix_tree(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(len(trace.states), 3)
        root = trace.requests[0].hash_ids[0]
        self.assertEqual(len(trace.children[root]), 2)
        self.assertEqual(len(trace.terminal_branches[root]), 2)
        self.assertTrue(all(meta.block_tokens == 512 for meta in trace.states.values()))

    def test_equal_timestamp_requests_do_not_hit_each_other(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        result = replay(trace, "lru", 3 * 512, 1.0, bytes_per_token=1)
        self.assertEqual(result.avoided_prefill_tokens, 1024)
        self.assertEqual(result.hit_blocks, 2)

    def test_all_replay_policies_respect_small_prefix_budget(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        for policy in POLICIES:
            result = replay(trace, policy, 2 * 512, 2 / 3, bytes_per_token=1)
            self.assertLessEqual(result.hit_blocks, result.requested_blocks)

    def test_two_hit_delays_admission_until_after_second_observation(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        lru = replay(trace, "lru", 3 * 512, 1.0, bytes_per_token=1)
        delayed = replay(trace, "lru_2hit", 3 * 512, 1.0, bytes_per_token=1)
        self.assertEqual(lru.avoided_prefill_tokens, 1024)
        self.assertEqual(delayed.avoided_prefill_tokens, 512)

    def test_fixed_block_charges_partial_node_at_full_block_size(self):
        records = [
            {"timestamp": 0, "input_length": 513, "output_length": 1, "hash_ids": [1, 2]},
            {"timestamp": 1000, "input_length": 513, "output_length": 1, "hash_ids": [1, 2]},
        ]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "partial.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        trace = load_mooncake_trace(path)
        packed = replay(trace, "lru", 513, 1.0, bytes_per_token=1, size_model="packed")
        fixed = replay(trace, "lru", 513, 1.0, bytes_per_token=1, size_model="fixed_block")
        self.assertEqual(packed.avoided_prefill_tokens, 513)
        self.assertEqual(fixed.avoided_prefill_tokens, 512)

    def test_online_comparison_excludes_offline_next_use(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        replayed = replay_rows(trace, (1.0,), bytes_per_token=1)
        online, headroom, sensitivity = replay_comparison_rows(replayed)
        self.assertNotIn("offline_next_use", {row["policy"] for row in online})
        self.assertEqual(len(headroom), 2)
        self.assertTrue(sensitivity)
        longest = longest_online_comparison_rows(online)
        self.assertEqual(len(longest), 2)
        self.assertTrue(all("longest_minus_lfu_fraction" in row for row in longest))

    def test_two_hit_loss_is_idealized_first_reuse(self):
        temporary, trace = self.make_trace()
        self.addCleanup(temporary.cleanup)
        summary = trace_summary(trace, bytes_per_token=1)
        self.assertEqual(summary["potential_reuse_events"], 3)
        self.assertEqual(summary["two_hit_lost_reuse_events"], 2)
        self.assertAlmostEqual(summary["two_hit_lost_avoided_prefill_fraction"], 2 / 3)

    def test_ranking_metrics_handle_ties_without_id_order(self):
        auc, ap, precision = ranking_metrics(
            np.asarray([1, 0, 1, 0]), np.asarray([2.0, 1.0, 1.0, 0.0]), k=2
        )
        self.assertAlmostEqual(auc, 0.875)
        self.assertAlmostEqual(ap, 5 / 6)
        self.assertAlmostEqual(precision, 0.75)
        self.assertTrue(math.isfinite(auc))

    def test_csv_writer_uses_lf_line_endings(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "rows.csv"
        write_csv(path, [{"value": 1}, {"value": 2}])
        content = path.read_bytes()
        self.assertNotIn(b"\r", content)
        self.assertEqual(content, b"value\n1\n2\n")


if __name__ == "__main__":
    unittest.main()
