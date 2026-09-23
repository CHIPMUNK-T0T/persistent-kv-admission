"""Boundaries for exploratory saved-decision postprocessing."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.decisionpop import state_indices, target_column
from persistent_kv_admission.trace import load_mooncake_trace
from scripts.analyze_retention_decisions import (
    candidate_measure,
    concentration,
    first_tuple_argmin,
    group_boundaries,
    summarize_decisions,
    utility_rows,
)


class RetentionDecisionTests(unittest.TestCase):
    def test_first_full_tuple_minimum_uses_tie_break_and_first_index(self):
        starts, widths = group_boundaries(np.array([0, 0, 0, 1, 1]), 2)
        chosen = first_tuple_argmin(
            np.array([1.0, 1.0, 2.0, -1.0, -1.0]),
            np.array([3.0, 2.0, 0.0, 4.0, 4.0]), starts, widths,
        )
        np.testing.assert_array_equal(chosen, [1, 3])

    def test_tied_minimum_and_constant_label_denominators(self):
        starts, widths = group_boundaries(np.array([0, 0, 0, 1, 1]), 2)
        chosen = np.array([1, 4])
        native = candidate_measure(np.array([0.0, 0.0, 2.0, 1.0, 1.0]), chosen, starts, widths)
        counts = candidate_measure(np.array([0.0, 0.0, 2.0, 1.0, 1.0]), chosen, starts, widths)
        proxy = candidate_measure(np.array([0.0, 0.0, 4.0, 3.0, 3.0]), chosen, starts, widths)
        result = summarize_decisions(native, counts, proxy, np.array(["resident_eviction"] * 2), "all")
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["discriminative_decisions"], 1)
        self.assertEqual(result["native_min_correct_rate_all"], 1.0)
        self.assertEqual(result["native_uniform_expected_min_correct_rate_all"], (2 / 3 + 1) / 2)
        self.assertEqual(result["native_min_correct_rate_discriminative"], 1.0)
        self.assertEqual(result["native_uniform_expected_min_correct_rate_discriminative"], 2 / 3)
        self.assertEqual(result["proxy_min_positive_rate"], 0.5)

    def test_horizon_boundary_preserves_target_conventions(self):
        delta = np.array([600_000.0, 600_001.0])
        counts = np.array([1.0, 0.0])
        next_use = target_column(delta, counts, "next_use", 600.0)
        binary = target_column(delta, counts, "binary", 600.0)
        np.testing.assert_array_equal(next_use, [next_use[0], next_use[0]])
        np.testing.assert_array_equal(binary, [1.0, 0.0])
        np.testing.assert_array_equal(target_column(delta, counts, "count", 600.0), [math.log(2), 0.0])

    def test_partial_block_proxy_uses_incremental_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.jsonl"
            path.write_text('{"timestamp":0,"input_length":600,"output_length":0,"hash_ids":["root","tail"]}\n')
            trace = load_mooncake_trace(path, block_size=512)
        mapping = state_indices(trace)
        tokens = np.array([trace.states[state].block_tokens for state in mapping])
        self.assertEqual(tokens[mapping["str:tail"]], 88)
        self.assertEqual(trace.states["str:tail"].prefix_tokens, 600)
        self.assertEqual(float((np.array([2.0, 2.0]) * tokens)[mapping["str:tail"]]), 176.0)

    def test_zero_total_concentration_is_undefined_and_all_decisions_define_top_n(self):
        self.assertTrue(math.isnan(concentration(np.zeros(100), 0.01)))
        gaps = np.zeros(100)
        gaps[0] = 9.0
        gaps[1] = 1.0
        self.assertAlmostEqual(concentration(gaps, 0.01), 0.9)
        self.assertEqual(concentration(gaps, 0.10), 1.0)

    def test_utility_is_paired_and_normalized_inside_each_window(self):
        utility, terminal = [], []
        for index in range(120):
            key = {"trace": "tiny", "cell": f"cell{index // 5}",
                   "target": "next_use", "seed": index % 5}
            terminal.append({**key, "terminal_common_population_ranking_delta": 0.2})
            for iteration in range(4):
                full = 100 + 10 * (iteration == 3)
                label = 20 + 5 * (iteration == 3)
                utility.append({**key, "iteration": iteration,
                                "requested_tokens": 1000, "avoided_prefill_tokens": full,
                                "label_window_requested_tokens": 100,
                                "label_window_avoided_prefill_tokens": label,
                                "input_token_points_vs_pi0": 100 * (full - 100) / 1000})
        seeds, aggregate = utility_rows(utility, terminal)
        selected = [row for row in seeds if row["cell"] == "cell0" and
                    row["seed"] == 0 and row["iteration"] == 3]
        by_window = {row["window"]: row for row in selected}
        self.assertEqual(by_window["full"]["delta_tokens_vs_pi0"], 10)
        self.assertEqual(by_window["full"]["delta_input_token_points_vs_pi0"], 1)
        self.assertEqual(by_window["label"]["delta_tokens_vs_pi0"], 5)
        self.assertEqual(by_window["label"]["delta_input_token_points_vs_pi0"], 5)
        self.assertEqual(len(aggregate), 192)
        self.assertTrue(all(row["ranking_utility_sign_agreement"] == "not_applicable"
                            for row in seeds if row["iteration"] != 3))


if __name__ == "__main__":
    unittest.main()
