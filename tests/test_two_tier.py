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


if __name__ == "__main__":
    unittest.main()
