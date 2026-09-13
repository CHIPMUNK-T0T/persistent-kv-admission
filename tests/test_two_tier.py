"""Tests for the two-tier victim replay and the event log."""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import persistent_kv_admission
from persistent_kv_admission.replay import replay
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import (
    _VictimStore,
    annotate_events,
    depth_bin,
    run_two_tier,
    summarize_events,
)

# A namespace package: __file__ is None, so locate it through __path__.
SOURCE_ROOT = str(Path(list(persistent_kv_admission.__path__)[0]).resolve().parent)

# Run in a subprocess so that PYTHONHASHSEED actually differs: set iteration
# order is fixed for the life of a process.
_DETERMINISM_CHILD = """
import json, sys
from persistent_kv_admission.replay import replay
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

trace = load_mooncake_trace(sys.argv[1])
out = {}
for policy in ("lru", "lfu"):
    result = replay(trace, policy, 12 * 512, 0.0, 1, eviction="heap")
    out["replay_" + policy] = [result.avoided_prefill_tokens, result.hit_blocks, result.block_hit_rate]
    for l2_policy in ("lru", "lfu"):
        two = run_two_tier(trace, policy, 12 * 512, 24 * 512, l2_policy, bytes_per_token=1)
        out["two_" + policy + "_" + l2_policy] = [
            two.avoided_prefill_tokens, two.l2_avoided_tokens, two.l1_evictions,
            two.l2_admissions, two.l2_evictions, two.l2_byte_seconds,
            sorted(two.l2_avoided_tokens_by_depth.items()),
            sorted(two.l2_byte_seconds_by_depth.items()),
            sorted(two.l2_admitted_bytes_by_depth.items()),
        ]
print(json.dumps(out, sort_keys=True))
"""


# Heap-path L2 numbers produced by the code as it stood before the sampled
# mechanism was added (Phase 0.95, commit 6793e08), captured once and frozen
# here as literals. Key: records/L1 blocks/L2 blocks/L1 policy/L2 policy; value:
# (avoided_prefill_tokens, l1_avoided_tokens, l2_avoided_tokens, l1_hit_blocks,
#  l2_hit_blocks, l2_present_unusable_blocks, l2_present_unusable_tokens,
#  l1_evictions, l2_admissions, l2_rejections, l2_evictions, l2_byte_seconds).
HEAP_REFERENCE = {
    'mixed/2/2/lfu/lfu': (152064, 152064, 0, 297, 0, 0, 0, 181, 181, 0, 179, 243200.0),
    'mixed/2/2/lfu/lru': (152064, 152064, 0, 297, 0, 0, 0, 181, 181, 0, 179, 243200.0),
    'mixed/2/2/lfu/lru_2hit': (152064, 152064, 0, 297, 0, 0, 0, 181, 0, 181, 0, 0),
    'mixed/2/2/lfu/offline_next_use': (152576, 152064, 512, 297, 1, 0, 0, 181, 181, 0, 178, 243200.0),
    'mixed/2/2/lru/lfu': (152064, 122368, 29696, 239, 58, 0, 0, 239, 239, 0, 179, 243200.0),
    'mixed/2/2/lru/lru': (122368, 122368, 0, 239, 0, 0, 0, 239, 239, 0, 237, 243200.0),
    'mixed/2/2/lru/lru_2hit': (152064, 122368, 29696, 239, 58, 0, 0, 239, 59, 180, 0, 90112.0),
    'mixed/2/2/lru/offline_next_use': (152576, 122368, 30208, 239, 59, 0, 0, 239, 239, 0, 178, 243200.0),
    'mixed/3/6/lfu/lfu': (152576, 152064, 512, 297, 1, 0, 0, 180, 180, 0, 173, 717824.0),
    'mixed/3/6/lfu/lru': (152576, 152064, 512, 297, 1, 0, 0, 180, 180, 0, 173, 717824.0),
    'mixed/3/6/lfu/lru_2hit': (152064, 152064, 0, 297, 0, 0, 0, 180, 0, 180, 0, 0),
    'mixed/3/6/lfu/offline_next_use': (152576, 152064, 512, 297, 1, 0, 0, 180, 180, 0, 173, 717824.0),
    'mixed/3/6/lru/lfu': (152576, 122368, 30208, 239, 59, 0, 0, 238, 238, 0, 173, 717824.0),
    'mixed/3/6/lru/lru': (152576, 122368, 30208, 239, 59, 0, 0, 238, 238, 0, 173, 717824.0),
    'mixed/3/6/lru/lru_2hit': (152064, 122368, 29696, 239, 58, 0, 0, 238, 59, 179, 0, 59904.0),
    'mixed/3/6/lru/offline_next_use': (152576, 122368, 30208, 239, 59, 0, 0, 238, 238, 0, 173, 717824.0),
    'mixed/4/3/lfu/lfu': (152576, 152064, 512, 297, 1, 0, 0, 179, 179, 0, 175, 359936.0),
    'mixed/4/3/lfu/lru': (152576, 152064, 512, 297, 1, 0, 0, 179, 179, 0, 175, 359936.0),
    'mixed/4/3/lfu/lru_2hit': (152064, 152064, 0, 297, 0, 0, 0, 179, 0, 179, 0, 0),
    'mixed/4/3/lfu/offline_next_use': (152576, 152064, 512, 297, 1, 0, 0, 179, 179, 0, 175, 359936.0),
    'mixed/4/3/lru/lfu': (152576, 122368, 30208, 239, 59, 0, 0, 237, 237, 0, 175, 359936.0),
    'mixed/4/3/lru/lru': (152576, 122368, 30208, 239, 59, 0, 0, 237, 237, 0, 175, 359936.0),
    'mixed/4/3/lru/lru_2hit': (152064, 122368, 29696, 239, 58, 0, 0, 237, 59, 178, 0, 29696.0),
    'mixed/4/3/lru/offline_next_use': (152576, 122368, 30208, 239, 59, 0, 0, 237, 237, 0, 175, 359936.0),
    'tie_heavy/2/2/lfu/lfu': (323584, 201216, 122368, 393, 239, 0, 0, 1086, 1086, 0, 979, 75776.0),
    'tie_heavy/2/2/lfu/lru': (321536, 201216, 120320, 393, 235, 0, 0, 1086, 1086, 0, 982, 75776.0),
    'tie_heavy/2/2/lfu/lru_2hit': (321536, 201216, 120320, 393, 235, 0, 0, 1086, 1017, 69, 913, 75776.0),
    'tie_heavy/2/2/lfu/offline_next_use': (257024, 201216, 55808, 393, 109, 81, 41472, 1086, 1086, 0, 937, 75776.0),
    'tie_heavy/2/2/lru/lfu': (323584, 204288, 119296, 399, 233, 0, 0, 1086, 1086, 0, 979, 75776.0),
    'tie_heavy/2/2/lru/lru': (322048, 204288, 117760, 399, 230, 0, 0, 1086, 1086, 0, 978, 75776.0),
    'tie_heavy/2/2/lru/lru_2hit': (322048, 204288, 117760, 399, 230, 0, 0, 1086, 1017, 69, 909, 75776.0),
    'tie_heavy/2/2/lru/offline_next_use': (272896, 204288, 68608, 399, 134, 60, 30720, 1086, 1086, 0, 937, 75776.0),
    'tie_heavy/3/6/lfu/lfu': (439808, 301056, 138752, 588, 271, 0, 0, 1013, 1013, 0, 804, 227328.0),
    'tie_heavy/3/6/lfu/lru': (440832, 301056, 139776, 588, 273, 0, 0, 1013, 1013, 0, 802, 227328.0),
    'tie_heavy/3/6/lfu/lru_2hit': (439808, 301056, 138752, 588, 271, 0, 0, 1013, 944, 69, 735, 225792.0),
    'tie_heavy/3/6/lfu/offline_next_use': (506368, 301056, 205312, 588, 401, 24, 12288, 1013, 1013, 0, 671, 227328.0),
    'tie_heavy/3/6/lru/lfu': (441344, 297472, 143872, 581, 281, 0, 0, 1016, 1016, 0, 803, 227328.0),
    'tie_heavy/3/6/lru/lru': (435712, 297472, 138240, 581, 270, 0, 0, 1016, 1016, 0, 806, 227328.0),
    'tie_heavy/3/6/lru/lru_2hit': (436736, 297472, 139264, 581, 272, 0, 0, 1016, 947, 69, 736, 225792.0),
    'tie_heavy/3/6/lru/offline_next_use': (507904, 297472, 210432, 581, 411, 23, 11776, 1016, 1016, 0, 671, 227328.0),
    'tie_heavy/4/3/lfu/lfu': (390144, 323584, 66560, 632, 130, 0, 0, 979, 979, 0, 881, 113664.0),
    'tie_heavy/4/3/lfu/lru': (395264, 323584, 71680, 632, 140, 0, 0, 979, 979, 0, 870, 113664.0),
    'tie_heavy/4/3/lfu/lru_2hit': (394752, 323584, 71168, 632, 139, 0, 0, 979, 910, 69, 802, 113152.0),
    'tie_heavy/4/3/lfu/offline_next_use': (434688, 323584, 111104, 632, 217, 33, 16896, 979, 979, 0, 779, 113664.0),
    'tie_heavy/4/3/lru/lfu': (393728, 322048, 71680, 629, 140, 0, 0, 978, 978, 0, 874, 113664.0),
    'tie_heavy/4/3/lru/lru': (398336, 322048, 76288, 629, 149, 0, 0, 978, 978, 0, 865, 113664.0),
    'tie_heavy/4/3/lru/lru_2hit': (398336, 322048, 76288, 629, 149, 0, 0, 978, 910, 68, 797, 113664.0),
    'tie_heavy/4/3/lru/offline_next_use': (438272, 322048, 116224, 629, 227, 31, 15872, 978, 978, 0, 773, 113664.0),
}

# Run in a subprocess so that PYTHONHASHSEED actually differs: dict and set
# iteration order over strings is fixed for the life of a process, and the
# sampled L2 draw reads the resident set as a list.
_SAMPLED_DETERMINISM_CHILD = """
import json, sys
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

trace = load_mooncake_trace(sys.argv[1])
out = {}
for l2_policy in ("lru", "lfu", "lru_2hit"):
    for seed in (0, 3):
        two = run_two_tier(trace, "lru", 12 * 512, 24 * 512, l2_policy, bytes_per_token=1,
                           l2_eviction="sampled", l2_sample_width=4, l2_seed=seed)
        out[l2_policy + "/" + str(seed)] = [
            two.avoided_prefill_tokens, two.l2_avoided_tokens, two.l1_evictions,
            two.l2_admissions, two.l2_rejections, two.l2_evictions, two.l2_decisions,
            two.l2_byte_seconds, sorted(two.l2_avoided_tokens_by_depth.items()),
        ]
print(json.dumps(out, sort_keys=True))
"""


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def record(step, chain, tokens=None):
    return {"timestamp": step * 1000, "input_length": tokens or 512 * len(chain),
            "output_length": 1, "hash_ids": chain}


def tie_heavy_records(groups=75, per_group=8, seed=0):
    """A shallow shared tree with several requests per timestamp group.

    Many states are touched in the same group, so LRU and LFU scores tie
    constantly and the victim is decided by the tie-breaking heap serials.
    """
    rng = random.Random(seed)
    records = []
    for step in range(groups):
        for _ in range(per_group):
            root, middle, leaf = rng.randrange(3), rng.randrange(4), rng.randrange(6)
            branch = root * 4 + middle
            records.append(record(step, [root, 10 + branch, 100 + branch * 6 + leaf]))
    return records


def mixed_records(steps=240):
    """A hot chain [1, 2] every fourth step, cold leaves [1, 100+step] otherwise."""
    return [record(step, [1, 2] if step % 4 == 0 else [1, 100 + step]) for step in range(steps)]


class EventLogTests(unittest.TestCase):
    def test_every_l1_eviction_is_one_event_and_states_leave_l1(self):
        temporary, trace = build_trace(mixed_records())
        self.addCleanup(temporary.cleanup)
        events = []
        result = run_two_tier(trace, "lru", 3 * 512, 0, bytes_per_token=1, event_sink=events)
        self.assertEqual(result.l1_evictions, len(events))
        self.assertGreater(len(events), 0)
        self.assertEqual(result.l2_policy, "none")
        self.assertEqual(result.avoided_prefill_tokens, result.l1_avoided_tokens)
        # Matches the single-tier engine exactly when there is no L2.
        single = replay(trace, "lru", 3 * 512, 0.0, 1, eviction="heap")
        self.assertEqual(result.avoided_prefill_tokens, single.avoided_prefill_tokens)
        # Causal fields are set at eviction; a state evicted twice has prior_evictions 1.
        self.assertTrue(any(e.prior_evictions > 0 for e in events))
        self.assertTrue(all(e.frequency >= 1 for e in events))

    def test_reuse_and_censoring_fields(self):
        # Three blocks of L1. Step 0 fills [1, 2]; steps 1-3 insert cold leaves
        # that push 2 out; step 4 reuses 2 while 1 is still in L1.
        records = [record(0, [1, 2]), record(1, [1, 101]), record(2, [1, 102]), record(3, [1, 103]),
                   record(4, [1, 2]), record(5, [1, 104]), record(6, [1, 105]), record(7, [1, 106])]
        temporary, trace = build_trace(records)
        self.addCleanup(temporary.cleanup)
        events = []
        run_two_tier(trace, "lru", 3 * 512, 0, bytes_per_token=1, event_sink=events)
        annotate_events(trace, events, bytes_per_token=1)
        first = next(e for e in events if e.state_id == "int:2")
        self.assertFalse(first.censored)
        self.assertEqual(first.reuse_order, 4)
        self.assertEqual(first.missing_ancestor_blocks, 0)
        self.assertEqual(first.rescue_tokens_self, 512)
        self.assertEqual(first.rescue_tokens_with_ancestors, 512)
        # Evicted at step 2 (after inserting 102, three blocks 1/101/102 remain);
        # competing requests are steps 3 (1, 103): one request, 1024 arriving
        # bytes, distinct bytes 1024 (states 1 and 103 each once).
        self.assertEqual(first.group_index, 2)
        self.assertEqual(first.requests_to_reuse, 1)
        self.assertEqual(first.arriving_bytes_to_reuse, 1024)
        self.assertEqual(first.distinct_bytes_to_reuse, 1024)
        self.assertAlmostEqual(first.seconds_to_reuse, 2.0)
        # 2 is evicted a second time and never comes back: censored.
        second = [e for e in events if e.state_id == "int:2"][-1]
        self.assertTrue(second.censored)
        self.assertEqual(second.prior_evictions, 1)
        self.assertTrue(math.isnan(second.seconds_to_reuse))
        self.assertGreater(second.requests_to_end, 0)
        self.assertEqual(second.future_uses, 0)
        summary, by_depth = summarize_events(events, 3 * 512, 8 * 1024)
        self.assertEqual(summary["events_total"], len(events))
        self.assertEqual(len(by_depth), 7)
        self.assertAlmostEqual(sum(row["share_of_events"] for row in by_depth), 1.0)

    def test_simultaneous_requests_count_every_rescue(self):
        # Two requests at the same timestamp share the evicted state 2; both
        # would be served from L2, as both are served from L1 in replay.
        records = [record(0, [1, 2]), record(1, [1, 101]), record(2, [1, 102]), record(3, [1, 103]),
                   record(4, [1, 2]), record(4, [1, 2, 5])]
        temporary, trace = build_trace(records)
        self.addCleanup(temporary.cleanup)
        events = []
        run_two_tier(trace, "lru", 3 * 512, 0, bytes_per_token=1, event_sink=events)
        event = next(e for e in events if e.state_id == "int:2")
        self.assertEqual(event.reuse_requests_in_group, 2)
        self.assertEqual(event.rescue_tokens_with_ancestors, 1024)
        summary, _ = summarize_events(events, 3 * 512, 6 * 1024)
        huge = run_two_tier(trace, "lru", 3 * 512, 10**9, "lru", bytes_per_token=1)
        self.assertEqual(huge.l2_avoided_tokens, summary["rescue_tokens_with_ancestors"])

    def test_distances_ignore_row_order_inside_the_reuse_group(self):
        # Steps 0-3 push 2 out of a three-block L1; at step 4 two requests share
        # the timestamp and only one of them touches 2. Both observe the same
        # pre-batch cache, so neither may count as competing traffic for the
        # other, whatever their order in the file.
        head = [record(0, [1, 2]), record(1, [1, 101]), record(2, [1, 102]), record(3, [1, 103])]
        other, reuse = record(4, [1, 104]), record(4, [1, 2])
        distances = []
        for group in ([other, reuse], [reuse, other]):
            temporary, trace = build_trace(head + group)
            self.addCleanup(temporary.cleanup)
            events = []
            run_two_tier(trace, "lru", 3 * 512, 0, bytes_per_token=1, event_sink=events)
            annotate_events(trace, events, bytes_per_token=1)
            event = next(e for e in events if e.state_id == "int:2" and not e.censored)
            distances.append((event.requests_to_reuse, event.arriving_bytes_to_reuse,
                              event.distinct_bytes_to_reuse))
        self.assertEqual(distances[0], distances[1])
        # 2 is evicted in group 2, reused in group 4, so the competing traffic
        # is group 3 alone: one request, 1024 arriving bytes, distinct states
        # 1 and 103 (512 bytes each).
        self.assertEqual(distances[0], (1, 1024, 1024))

    def test_depth_bins(self):
        self.assertEqual([depth_bin(d) for d in (1, 2, 3, 4, 7, 8, 15, 16, 40, 64, 200)],
                         ["1", "2-3", "2-3", "4-7", "4-7", "8-15", "8-15", "16-31", "32-63", "64+", "64+"])


class TwoTierReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(mixed_records())
        self.addCleanup(self.temporary.cleanup)

    def test_infinite_union_l2_equals_an_infinite_single_cache(self):
        huge = 10**9
        two = run_two_tier(self.trace, "lru", 3 * 512, huge, "lru", bytes_per_token=1)
        one = replay(self.trace, "lru", huge, 0.0, 1, eviction="heap")
        self.assertEqual(two.avoided_prefill_tokens, one.avoided_prefill_tokens)
        self.assertGreater(two.l2_avoided_tokens, 0)
        self.assertEqual(two.l2_evictions, 0)

    def test_independent_is_an_upper_bound_and_standalone_charges_ancestors(self):
        tree = run_two_tier(self.trace, "lru", 2 * 512, 2 * 512, "lru", hit_model="tree", bytes_per_token=1)
        independent = run_two_tier(self.trace, "lru", 2 * 512, 2 * 512, "lru", hit_model="independent", bytes_per_token=1)
        self.assertGreaterEqual(independent.avoided_prefill_tokens, tree.avoided_prefill_tokens)
        self.assertEqual(independent.l2_present_unusable_blocks, 0)
        standalone = run_two_tier(self.trace, "lru", 2 * 512, 2 * 512, "lru", closure="standalone", bytes_per_token=1)
        self.assertGreater(standalone.l2_ancestor_copy_bytes, 0)
        self.assertGreaterEqual(standalone.l1_avoided_tokens, 0)
        with self.assertRaises(ValueError):
            run_two_tier(self.trace, "lru", 512, 512, "lru", closure="standalone", hit_model="independent", bytes_per_token=1)
        with self.assertRaises(ValueError):
            run_two_tier(self.trace, "fifo", 512, 512, "lru", bytes_per_token=1)

    def test_two_hit_rejects_first_time_victims(self):
        result = run_two_tier(self.trace, "lru", 2 * 512, 4 * 512, "lru_2hit", bytes_per_token=1)
        self.assertGreater(result.l2_rejections, 0)
        # The exclusive union store never sees a victim it already holds.
        self.assertEqual(result.l2_already_held, 0)
        self.assertEqual(result.l2_admissions + result.l2_rejections + result.l2_already_held,
                         result.l1_evictions)
        # The inclusive standalone tier does: a block reused from L2 stays
        # there and can be evicted from L1 again. That is neither an admission
        # nor a rejection, and the three still account for every L1 eviction.
        standalone = run_two_tier(self.trace, "lru", 2 * 512, 4 * 512, "lru", closure="standalone",
                                  bytes_per_token=1)
        self.assertGreater(standalone.l2_already_held, 0)
        self.assertEqual(standalone.l2_admissions + standalone.l2_rejections + standalone.l2_already_held,
                         standalone.l1_evictions)
        self.assertIn("l2_already_held", standalone.as_row())

    def test_byte_seconds_and_admitted_bytes_follow_the_measured_window(self):
        # Five one-block roots, one per second. L1 holds one block, so every
        # step evicts the previous block into an L2 that never fills up:
        # L2 holds 512 B over [1s, 2s], 1024 over [2s, 3s], 1536 over [3s, 4s].
        temporary, trace = build_trace([record(step, [step + 1]) for step in range(5)])
        self.addCleanup(temporary.cleanup)
        for closure in ("union", "standalone"):
            whole = run_two_tier(trace, "lru", 512, 10 * 512, "lru", closure=closure, bytes_per_token=1)
            self.assertEqual((whole.l1_evictions, whole.l2_admissions), (4, 4), closure)
            self.assertAlmostEqual(whole.l2_byte_seconds, 3072.0)
            self.assertEqual(whole.l2_byte_seconds_by_depth, {"1": 3072.0})
            self.assertEqual(whole.l2_admitted_bytes_by_depth, {"1": 4 * 512})
            # The window opens inside [2s, 3s]: only the interval starting at
            # 3s is integrated, and only the evictions at 3s and 4s admit.
            windowed = run_two_tier(trace, "lru", 512, 10 * 512, "lru", closure=closure,
                                    bytes_per_token=1, measure_from_ms=2500)
            self.assertAlmostEqual(windowed.l2_byte_seconds, 1536.0, msg=closure)
            self.assertEqual(windowed.l2_byte_seconds_by_depth, {"1": 1536.0}, closure)
            self.assertEqual(windowed.l2_admitted_bytes_by_depth, {"1": 2 * 512}, closure)
            # Whole-trace counters are deliberately not windowed.
            self.assertEqual((windowed.l1_evictions, windowed.l2_admissions), (4, 4), closure)

    def test_lfu_l1_runs_and_depth_breakdown_adds_up(self):
        result = run_two_tier(self.trace, "lfu", 2 * 512, 4 * 512, "offline_next_use", bytes_per_token=1)
        self.assertEqual(sum(result.l2_avoided_tokens_by_depth.values()), result.l2_avoided_tokens)
        self.assertAlmostEqual(sum(result.l2_byte_seconds_by_depth.values()), result.l2_byte_seconds)
        row = result.as_row()
        self.assertIn("l2_avoided_tokens_depth_2-3", row)


class DeterminismTests(unittest.TestCase):
    def test_results_do_not_depend_on_hash_randomisation(self):
        # Heap serials break ties between equal scores, so anything derived
        # from iterating a set of state ids would make the result depend on
        # the process's string-hash seed.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trace.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in tie_heavy_records()))
        outputs = []
        for seed in ("0", "12345"):
            environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=SOURCE_ROOT)
            finished = subprocess.run([sys.executable, "-c", _DETERMINISM_CHILD, str(path)],
                                      capture_output=True, text=True, env=environment)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            outputs.append(finished.stdout)
        self.assertTrue(json.loads(outputs[0]))
        self.assertEqual(outputs[0], outputs[1])


class OfflineTiebreakTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace([record(0, [1, 2, 3])])
        self.addCleanup(self.temporary.cleanup)
        self.groups = {"int:1": [0], "int:2": [0], "int:3": [0]}

    def test_tiebreak_decides_which_of_two_equal_next_uses_is_evicted(self):
        # Neither block is used again, so their next use ties at infinity and
        # only the prefix length separates them.
        for tiebreak, evicted in (("prefix_first", "int:2"), ("deeper_first", "int:3")):
            store = _VictimStore(self.trace, 512, "offline_next_use", self.groups, lambda s: 512,
                                 offline_tiebreak=tiebreak)
            store.admit("int:2", 1, 0, 0)
            store.admit("int:3", 1, 0, 0)
            self.assertNotIn(evicted, store.cached, tiebreak)
            self.assertEqual(len(store.cached), 1, tiebreak)

    def test_unknown_tiebreak_raises_and_the_default_is_unchanged(self):
        with self.assertRaises(ValueError):
            _VictimStore(self.trace, 512, "offline_next_use", self.groups, lambda s: 512,
                         offline_tiebreak="shallower_first")
        temporary, trace = build_trace(mixed_records())
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            run_two_tier(trace, "lru", 2 * 512, 4 * 512, "offline_next_use", bytes_per_token=1,
                         offline_tiebreak="shallower_first")
        default = run_two_tier(trace, "lru", 2 * 512, 4 * 512, "offline_next_use", bytes_per_token=1)
        explicit = run_two_tier(trace, "lru", 2 * 512, 4 * 512, "offline_next_use", bytes_per_token=1,
                                offline_tiebreak="prefix_first")
        deeper = run_two_tier(trace, "lru", 2 * 512, 4 * 512, "offline_next_use", bytes_per_token=1,
                              offline_tiebreak="deeper_first")
        self.assertEqual(default.offline_tiebreak, "prefix_first")
        self.assertEqual(default.avoided_prefill_tokens, explicit.avoided_prefill_tokens)
        self.assertEqual(deeper.offline_tiebreak, "deeper_first")
        self.assertIn("offline_tiebreak", default.as_row())


class VictimStoreTests(unittest.TestCase):
    def test_offline_store_evicts_the_farthest_next_use(self):
        temporary, trace = build_trace([record(0, [1, 2]), record(1, [1, 3]), record(2, [1, 4]),
                                        record(3, [1, 3]), record(4, [1, 2])])
        self.addCleanup(temporary.cleanup)
        groups = {"int:2": [0, 4], "int:3": [1, 3], "int:4": [2]}
        store = _VictimStore(trace, 2 * 512, "offline_next_use", groups, lambda s: 512)
        store.admit("int:2", 1, 0, 0)
        store.admit("int:3", 1, 1, 1)
        store.admit("int:4", 1, 2, 2)  # never used again: evicted first
        self.assertEqual(set(store.cached), {"int:2", "int:3"})
        store.admit("int:4", 1, 2, 2)  # re-admission is allowed after eviction
        self.assertNotIn("int:4", store.cached)  # still the farthest
        store.promote("int:3")
        self.assertNotIn("int:3", store.cached)
        with self.assertRaises(RuntimeError):
            store.admit("int:2", 1, 0, 0)



class _IdScorer:
    """Scores a state by the integer in its canonical id, recomputed every call.

    Deterministic and injective on these traces, so a decision's scores can be
    checked against its candidate list position by position.
    """

    time_varying = True

    def __init__(self):
        self.observations = 0

    def observe(self, requests, timestamp_ms):
        self.observations += 1

    def score(self, state_id, timestamp_ms):
        return float(int(state_id.split(":")[1]))


class _ConstantLowScorer:
    """Scores one named state below everything else."""

    time_varying = True

    def __init__(self, low):
        self.low = low

    def observe(self, requests, timestamp_ms):
        pass

    def score(self, state_id, timestamp_ms):
        return 0.0 if state_id == self.low else 1.0


class SampledVictimStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(mixed_records())
        self.addCleanup(self.temporary.cleanup)

    def _run(self, **kwargs):
        return run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, bytes_per_token=1, **kwargs)

    def test_a_full_width_sample_without_ties_reproduces_the_heap_exactly(self):
        # With a sample wider than the store the decision set is every resident,
        # so the only thing that could separate the two mechanisms is the
        # tie-break. The hook asserts that no decision ever ties, which makes
        # the equality a statement about the mechanism and not about luck.
        ties = []

        def hook(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
            if len(set(scores)) != len(scores):
                ties.append((timestamp_ms, tuple(candidates), tuple(scores)))

        heap = self._run(l2_policy="lru", l2_eviction="heap")
        sampled = self._run(l2_policy="lru", l2_eviction="sampled", l2_sample_width=10 ** 6,
                            l2_decision_hook=hook)
        self.assertEqual(ties, [])
        self.assertGreater(sampled.l2_evictions, 0)
        for field in ("avoided_prefill_tokens", "l1_avoided_tokens", "l2_avoided_tokens",
                      "l2_admissions", "l2_evictions", "l2_rejections", "l1_evictions",
                      "l2_hit_blocks", "l2_byte_seconds"):
            self.assertEqual(getattr(heap, field), getattr(sampled, field), field)
        self.assertEqual(heap.l2_avoided_tokens_by_depth, sampled.l2_avoided_tokens_by_depth)
        self.assertEqual(sampled.l2_eviction, "sampled")
        self.assertEqual(heap.l2_eviction, "heap")

    def test_b_a_scorer_that_ranks_the_arrival_lowest_turns_admission_into_rejection(self):
        temporary, trace = build_trace([record(0, [1, 2, 3])])
        self.addCleanup(temporary.cleanup)
        store = _VictimStore(trace, 512, "learned", {}, lambda s: 512, eviction="sampled",
                             scorer=_ConstantLowScorer("int:3"), frequency={"int:2": 1, "int:3": 1},
                             last_group={"int:2": 0, "int:3": 1})
        self.assertTrue(store.admit("int:2", 1, 0, 0, 0.0))
        self.assertEqual((store.admissions, store.rejections, store.evictions), (1, 0, 0))
        # The store is full and the arrival scores below the resident: declined.
        self.assertFalse(store.admit("int:3", 1, 1, 1, 1000.0))
        self.assertEqual((store.admissions, store.rejections, store.evictions), (1, 1, 0))
        self.assertEqual(list(store.cached), ["int:2"])
        self.assertEqual(store.current_bytes, 512)
        # The other way round the arrival wins and the resident is evicted.
        store.scorer = _ConstantLowScorer("int:2")
        self.assertTrue(store.admit("int:3", 1, 1, 1, 1000.0))
        self.assertEqual((store.admissions, store.rejections, store.evictions), (2, 1, 1))
        self.assertEqual(list(store.cached), ["int:3"])

    def test_b_rejections_and_admissions_still_account_for_every_l1_eviction(self):
        result = self._run(l2_policy="learned", l2_eviction="sampled", l2_sample_width=4,
                           l2_scorer=_ConstantLowScorer("int:2"))
        self.assertEqual(result.l2_admissions + result.l2_rejections + result.l2_already_held,
                         result.l1_evictions)
        self.assertGreater(result.l2_rejections, 0)
        self.assertGreater(result.l2_decisions, 0)

    def test_c_the_hook_sees_the_arrival_first_and_scores_aligned_with_candidates(self):
        victims = []
        decisions = []

        def victim_hook(state_id, timestamp_ms, group_index):
            victims.append(state_id)
            decisions.append([])

        def hook(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
            decisions[-1].append((list(candidates), list(scores), victim_index, arriving_index))

        result = self._run(l2_policy="learned", l2_eviction="sampled", l2_sample_width=4,
                           l2_scorer=_IdScorer(), l2_decision_hook=hook, victim_hook=victim_hook)
        rounds = [r for per_victim in decisions for r in per_victim]
        self.assertGreater(len(rounds), 0)
        self.assertEqual(sum(len(per_victim) for per_victim in decisions), result.l2_decisions)
        for victim, per_victim in zip(victims, decisions):
            for position, (candidates, scores, victim_index, arriving_index) in enumerate(per_victim):
                if position == 0:
                    self.assertEqual(arriving_index, 0)
                    self.assertEqual(candidates[0], victim)
                else:
                    self.assertEqual(arriving_index, -1)
                self.assertEqual(len(scores), len(candidates))
                self.assertLessEqual(len(candidates), 5 if position == 0 else 4)
                self.assertEqual(len(set(candidates)), len(candidates))
                for state_id, score in zip(candidates, scores):
                    self.assertEqual(score[0], float(int(state_id.split(":")[1])))
                self.assertEqual(victim_index, scores.index(min(scores)))

    def test_d_the_heap_path_still_reproduces_the_pre_sampling_numbers(self):
        for records_name, records in (("mixed", mixed_records()), ("tie_heavy", tie_heavy_records())):
            temporary, trace = build_trace(records)
            self.addCleanup(temporary.cleanup)
            for l1_blocks, l2_blocks in ((2, 2), (3, 6), (4, 3)):
                for policy in ("lru", "lfu", "lru_2hit", "offline_next_use"):
                    for l1_policy in ("lru", "lfu"):
                        key = f"{records_name}/{l1_blocks}/{l2_blocks}/{l1_policy}/{policy}"
                        r = run_two_tier(trace, l1_policy, l1_blocks * 512, l2_blocks * 512, policy,
                                         bytes_per_token=1)
                        got = (r.avoided_prefill_tokens, r.l1_avoided_tokens, r.l2_avoided_tokens,
                               r.l1_hit_blocks, r.l2_hit_blocks, r.l2_present_unusable_blocks,
                               r.l2_present_unusable_tokens, r.l1_evictions, r.l2_admissions,
                               r.l2_rejections, r.l2_evictions, round(r.l2_byte_seconds, 6))
                        self.assertEqual(got, HEAP_REFERENCE[key], key)
        self.assertEqual(len(HEAP_REFERENCE), 48)

    def test_sampled_eviction_rejects_the_settings_it_is_not_defined_for(self):
        with self.assertRaises(ValueError):
            self._run(l2_policy="lru", l2_eviction="sampled", closure="standalone")
        with self.assertRaises(ValueError):
            self._run(l2_policy="offline_next_use", l2_eviction="sampled")
        with self.assertRaises(ValueError):
            self._run(l2_policy="learned", l2_eviction="heap", l2_scorer=_IdScorer())
        with self.assertRaises(ValueError):
            self._run(l2_policy="learned", l2_eviction="sampled")
        with self.assertRaises(ValueError):
            self._run(l2_policy="lru", l2_eviction="sampled", l2_scorer=_IdScorer())
        with self.assertRaises(ValueError):
            self._run(l2_policy="lru", l2_eviction="rr")

    def test_the_scorer_is_shown_every_group_once_and_is_attached_to_the_store(self):
        attached = []

        class Attaching(_IdScorer):
            def attach(self, store):
                attached.append(store)

        scorer = Attaching()
        self._run(l2_policy="learned", l2_eviction="sampled", l2_scorer=scorer)
        self.assertEqual(scorer.observations, sum(1 for _ in self.trace.timestamp_groups()))
        self.assertEqual(len(attached), 1)
        self.assertIs(attached[0].cached, attached[0].cached)
        self.assertTrue(hasattr(attached[0], "cached"))

    def test_an_observer_without_an_l2_sees_the_same_groups_as_a_scorer_would(self):
        seen, victims = [], []

        class Observer:
            def observe(self, requests, timestamp_ms):
                seen.append((timestamp_ms, len(requests)))

        result = run_two_tier(self.trace, "lru", 3 * 512, 0, bytes_per_token=1,
                              observer=Observer(),
                              victim_hook=lambda s, t, g: victims.append((s, t, g)))
        self.assertEqual(seen, [(t, len(r)) for t, r in self.trace.timestamp_groups()])
        self.assertEqual(len(victims), result.l1_evictions)
        self.assertGreater(len(victims), 0)

    def test_seeds_change_the_sampled_result_and_the_row_records_them(self):
        temporary, trace = build_trace(tie_heavy_records())
        self.addCleanup(temporary.cleanup)

        def run(seed):
            return run_two_tier(trace, "lru", 6 * 512, 12 * 512, "lru", bytes_per_token=1,
                                l2_eviction="sampled", l2_sample_width=2, l2_seed=seed,
                                l2_arm="lru_s")

        rows = {}
        for seed in (0, 1, 2):
            result = run(seed)
            row = result.as_row()
            self.assertEqual((row["l2_eviction"], row["l2_sample_width"], row["l2_seed"],
                              row["l2_arm"]), ("sampled", 2, seed, "lru_s"))
            rows[seed] = result.avoided_prefill_tokens
        self.assertGreater(len(set(rows.values())), 1)
        # Same seed, same answer.
        self.assertEqual(run(1).avoided_prefill_tokens, rows[1])


class SampledDeterminismTests(unittest.TestCase):
    def test_e_the_sampled_path_does_not_depend_on_hash_randomisation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trace.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in tie_heavy_records()))
        outputs = []
        for seed in ("0", "12345"):
            environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=SOURCE_ROOT)
            finished = subprocess.run([sys.executable, "-c", _SAMPLED_DETERMINISM_CHILD, str(path)],
                                      capture_output=True, text=True, env=environment)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            outputs.append(finished.stdout)
        self.assertTrue(json.loads(outputs[0]))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
