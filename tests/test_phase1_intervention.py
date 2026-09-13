"""Core API regressions for the Phase 1 arrival-protection intervention."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest

from persistent_kv_admission.intervention import Phase1Collector, paired_request_summary
from persistent_kv_admission.trace import Request, StateMeta, Trace
from persistent_kv_admission.twotier import (
    ARRIVAL_PROTECTIONS,
    _VictimStore,
    run_two_tier,
)


class _MappingScorer:
    time_varying = True

    def __init__(self, scores):
        self.scores = scores

    def observe(self, requests, timestamp_ms):
        pass

    def score(self, state_id, timestamp_ms):
        return float(self.scores[state_id])


class _NumericScorer:
    time_varying = True

    def observe(self, requests, timestamp_ms):
        pass

    def score(self, state_id, timestamp_ms):
        return float(int(state_id[1:]))


class _NewestLowScorer(_NumericScorer):
    def score(self, state_id, timestamp_ms):
        return -super().score(state_id, timestamp_ms)


def _tree_trace(parents, sizes=None):
    """Build the smallest Trace needed by _VictimStore."""
    sizes = sizes or {}
    states = {}
    children = defaultdict(set)

    def depth(state_id):
        value = 1
        parent = parents[state_id]
        while parent is not None:
            value += 1
            parent = parents[parent]
        return value

    for state_id, parent in parents.items():
        state_depth = depth(state_id)
        states[state_id] = StateMeta(
            state_id=state_id,
            parent_id=parent,
            depth=state_depth,
            prefix_tokens=state_depth,
            block_tokens=sizes.get(state_id, 1),
        )
        if parent is not None:
            children[parent].add(state_id)
    return Trace(
        name="phase1-store",
        requests=[],
        states=states,
        occurrences_ms={},
        children=dict(children),
        terminal_branches={},
        block_size=1,
    )


def _request_trace(chains):
    """Build a regular 512-token trace without filesystem fixtures."""
    requests = []
    states = {}
    occurrences = defaultdict(list)
    children = defaultdict(set)
    terminal_branches = defaultdict(set)
    for order, chain in enumerate(chains):
        timestamp_ms = float(order * 1000)
        chain = tuple(chain)
        requests.append(
            Request(
                order=order,
                timestamp_ms=timestamp_ms,
                input_length=512 * len(chain),
                output_length=1,
                hash_ids=chain,
            )
        )
        terminal = chain[-1]
        for index, state_id in enumerate(chain):
            parent = chain[index - 1] if index else None
            meta = StateMeta(
                state_id=state_id,
                parent_id=parent,
                depth=index + 1,
                prefix_tokens=512 * (index + 1),
                block_tokens=512,
            )
            if state_id in states:
                assert states[state_id] == meta
            states[state_id] = meta
            occurrences[state_id].append(timestamp_ms)
            terminal_branches[state_id].add(terminal)
            if parent is not None:
                children[parent].add(state_id)
    return Trace(
        name="phase1-public-api",
        requests=requests,
        states=states,
        occurrences_ms=dict(occurrences),
        children=dict(children),
        terminal_branches=dict(terminal_branches),
        block_size=512,
    )


def _decision_recorder(target):
    def record(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
        target.append(
            (
                tuple(candidates),
                tuple(scores),
                victim_index,
                timestamp_ms,
                group_index,
                arriving_index,
            )
        )

    return record


def _removal_recorder(target):
    def record(state_id, kind, timestamp_ms, group_index, decision_index):
        target.append((state_id, kind, timestamp_ms, group_index, decision_index))

    return record


def _protection_recorder(target):
    def record(event, timestamp_ms, group_index):
        target.append((event, timestamp_ms, group_index))

    return record


def _make_store(
    trace,
    capacity,
    scores,
    *,
    sizes=None,
    protection="none",
    sample_width=16,
    seed=0,
    decisions=None,
    removals=None,
    events=None,
):
    sizes = sizes or {state_id: 1 for state_id in trace.states}
    return _VictimStore(
        trace,
        capacity,
        "learned",
        {},
        sizes.__getitem__,
        eviction="sampled",
        sample_width=sample_width,
        seed=seed,
        scorer=_MappingScorer(scores),
        decision_hook=_decision_recorder(decisions) if decisions is not None else None,
        frequency=defaultdict(int),
        last_group=defaultdict(int),
        removal_hook=_removal_recorder(removals) if removals is not None else None,
        arrival_protection=protection,
        protection_hook=_protection_recorder(events) if events is not None else None,
    )


class ArrivalProtectionStoreTests(unittest.TestCase):
    def test_public_modes_are_fixed(self):
        self.assertEqual(ARRIVAL_PROTECTIONS, ("none", "direct_child", "all"))

    def test_direct_child_does_not_protect_an_arrival_without_a_child(self):
        trace = _tree_trace({"resident": None, "arrival": None})
        decisions, events = [], []
        store = _make_store(
            trace,
            1,
            {"resident": 1, "arrival": 0},
            protection="direct_child",
            decisions=decisions,
            events=events,
        )
        self.assertTrue(store.admit("resident", 1, 0, 0))
        self.assertFalse(store.admit("arrival", 1, 1, 1, 1000.0))
        self.assertEqual(list(store.cached), ["resident"])
        self.assertEqual(events, [])
        self.assertEqual(decisions[0][0], ("arrival", "resident"))
        self.assertEqual(decisions[0][-1], 0)

    def test_direct_child_protects_and_hook_sees_only_eligible_residents(self):
        trace = _tree_trace(
            {"arrival": None, "child": "arrival", "other": None}
        )
        decisions, events = [], []
        store = _make_store(
            trace,
            2,
            {"arrival": 0, "other": 1, "child": 2},
            protection="direct_child",
            decisions=decisions,
            events=events,
        )
        store.admit("child", 1, 0, 0)
        store.admit("other", 1, 0, 0)
        self.assertTrue(store.admit("arrival", 1, 1, 1, 1000.0))

        self.assertEqual(list(store.cached), ["child", "arrival"])
        self.assertNotIn("arrival", decisions[0][0])
        self.assertEqual(decisions[0][0], ("child", "other"))
        self.assertEqual(decisions[0][1], ((2.0, 0.0), (1.0, 0.0)))
        self.assertEqual(decisions[0][2], 1)
        self.assertEqual(decisions[0][-1], -1)
        self.assertEqual(
            Counter(event for event, _, _ in events),
            Counter(
                {
                    "protected_offer": 1,
                    "protected_overflow_offer": 1,
                    "first_round_override": 1,
                }
            ),
        )

    def test_deep_descendant_behind_missing_child_does_not_trigger(self):
        trace = _tree_trace(
            {"arrival": None, "missing": "arrival", "deep": "missing"}
        )
        decisions, events = [], []
        store = _make_store(
            trace,
            1,
            {"arrival": 0, "missing": 2, "deep": 1},
            protection="direct_child",
            decisions=decisions,
            events=events,
        )
        store.admit("deep", 1, 0, 0)
        self.assertFalse(store.admit("arrival", 1, 1, 1))
        self.assertEqual(list(store.cached), ["deep"])
        self.assertEqual(events, [])
        self.assertEqual(decisions[0][0], ("arrival", "deep"))
        self.assertEqual(decisions[0][-1], 0)

    def test_all_protects_a_childless_arrival(self):
        trace = _tree_trace({"resident": None, "arrival": None})
        decisions, events = [], []
        store = _make_store(
            trace,
            1,
            {"resident": 1, "arrival": 0},
            protection="all",
            decisions=decisions,
            events=events,
        )
        store.admit("resident", 1, 0, 0)
        events.clear()
        self.assertTrue(store.admit("arrival", 1, 1, 1))
        self.assertEqual(list(store.cached), ["arrival"])
        self.assertEqual(decisions[0][0], ("resident",))
        self.assertEqual(decisions[0][-1], -1)
        self.assertEqual(
            Counter(event for event, _, _ in events),
            Counter(
                {
                    "protected_offer": 1,
                    "protected_overflow_offer": 1,
                    "first_round_override": 1,
                }
            ),
        )

    def test_direct_child_snapshot_survives_eviction_of_the_triggering_child(self):
        trace = _tree_trace(
            {"arrival": None, "child": "arrival", "other": None},
            {"arrival": 3, "child": 1, "other": 1},
        )
        sizes = {"arrival": 3, "child": 1, "other": 1}
        decisions, removals = [], []
        store = _make_store(
            trace,
            3,
            {"arrival": 0, "child": 1, "other": 2},
            sizes=sizes,
            protection="direct_child",
            decisions=decisions,
            removals=removals,
        )
        store.admit("child", 1, 0, 0)
        store.admit("other", 1, 0, 0)
        removals.clear()
        self.assertTrue(store.admit("arrival", 1, 1, 1))

        self.assertEqual(list(store.cached), ["arrival"])
        self.assertEqual([decision[0] for decision in decisions], [("child", "other"), ("other",)])
        self.assertTrue(all("arrival" not in decision[0] for decision in decisions))
        self.assertTrue(all(decision[-1] == -1 for decision in decisions))
        self.assertEqual([state_id for state_id, kind, *_ in removals], ["child", "other"])
        self.assertTrue(all(kind == "evicted" for _, kind, *_ in removals))

    def test_multiple_rounds_exclude_arrival_and_preserve_first_resident_draw(self):
        parents = {"r1": None, "r2": None, "r3": None, "arrival": None}
        sizes = {"r1": 1, "r2": 1, "r3": 1, "arrival": 4}
        scores = {"r1": 1, "r2": 2, "r3": 3, "arrival": 0}
        trace = _tree_trace(parents, sizes)
        baseline_decisions, protected_decisions, events = [], [], []
        baseline = _make_store(
            trace,
            4,
            scores,
            sizes=sizes,
            sample_width=1,
            seed=11,
            decisions=baseline_decisions,
        )
        protected = _make_store(
            trace,
            4,
            scores,
            sizes=sizes,
            protection="all",
            sample_width=1,
            seed=11,
            decisions=protected_decisions,
            events=events,
        )
        for store in (baseline, protected):
            for state_id in ("r1", "r2", "r3"):
                store.admit(state_id, 1, 0, 0)
        events.clear()

        self.assertFalse(baseline.admit("arrival", 1, 1, 1))
        self.assertTrue(protected.admit("arrival", 1, 1, 1))
        self.assertEqual(
            protected_decisions[0][0], baseline_decisions[0][0][1:]
        )
        self.assertEqual(len(protected_decisions), 3)
        for candidates, scores_seen, victim_index, *_, arriving_index in protected_decisions:
            self.assertNotIn("arrival", candidates)
            self.assertEqual(arriving_index, -1)
            self.assertEqual(victim_index, scores_seen.index(min(scores_seen)))
        self.assertEqual(list(protected.cached), ["arrival"])
        self.assertEqual(
            Counter(event for event, _, _ in events),
            Counter(
                {
                    "protected_offer": 1,
                    "protected_overflow_offer": 1,
                    "first_round_override": 1,
                }
            ),
        )

    def test_oversized_arrival_uses_the_complete_original_path(self):
        parents = {"r1": None, "r2": None, "arrival": None}
        sizes = {"r1": 1, "r2": 1, "arrival": 3}
        scores = {"r1": 0, "r2": 1, "arrival": 10}
        trace = _tree_trace(parents, sizes)
        stores = []
        recordings = []
        for protection in ("none", "all"):
            decisions, removals, events = [], [], []
            store = _make_store(
                trace,
                2,
                scores,
                sizes=sizes,
                protection=protection,
                sample_width=16,
                seed=7,
                decisions=decisions,
                removals=removals,
                events=events,
            )
            store.admit("r1", 1, 0, 0)
            store.admit("r2", 1, 0, 0)
            events.clear()
            decisions.clear()
            removals.clear()
            store.admit("arrival", 1, 1, 1, 1000.0)
            stores.append(store)
            recordings.append((decisions, removals, events))

        baseline, protected = stores
        self.assertEqual(list(protected.cached), list(baseline.cached))
        self.assertEqual(protected.current_bytes, baseline.current_bytes)
        self.assertEqual(
            (protected.admissions, protected.rejections, protected.evictions, protected.decisions),
            (baseline.admissions, baseline.rejections, baseline.evictions, baseline.decisions),
        )
        self.assertEqual(protected.rng.getstate(), baseline.rng.getstate())
        self.assertEqual(recordings[1][0], recordings[0][0])
        self.assertEqual(recordings[1][1], recordings[0][1])
        self.assertEqual(recordings[1][2], [("oversized_ineligible", 1000.0, 1)])
        self.assertGreater(len(recordings[1][0]), 1)

    def test_feasible_protected_offer_with_no_resident_needs_no_decision(self):
        trace = _tree_trace({"arrival": None})
        decisions, events = [], []
        store = _make_store(
            trace,
            1,
            {"arrival": 0},
            protection="all",
            decisions=decisions,
            events=events,
        )
        self.assertTrue(store.admit("arrival", 1, 0, 0, 250.0))
        self.assertEqual(list(store.cached), ["arrival"])
        self.assertEqual(decisions, [])
        self.assertEqual(events, [("protected_offer", 250.0, 0)])

    def test_empty_eligible_set_during_feasible_overflow_is_an_invariant_error(self):
        trace = _tree_trace({"arrival": None})
        store = _make_store(
            trace,
            1,
            {"arrival": 0},
            protection="all",
        )
        # Simulate corrupt accounting: a feasible arrival cannot legitimately
        # overflow when it is the store's only cached state.
        store.current_bytes = 1
        with self.assertRaisesRegex(AssertionError, "without an eligible resident"):
            store.admit("arrival", 1, 0, 0)

    def test_positive_sample_width_is_required_only_for_active_protection(self):
        trace = _tree_trace({"arrival": None})
        for protection in ("direct_child", "all"):
            with self.subTest(protection=protection):
                with self.assertRaises(ValueError):
                    _make_store(
                        trace,
                        1,
                        {"arrival": 0},
                        protection=protection,
                        sample_width=0,
                    )
        # The pre-existing none path keeps accepting this constructor setting.
        _make_store(trace, 1, {"arrival": 0}, sample_width=0)


    def test_protection_does_not_carry_to_the_next_offer(self):
        trace = _tree_trace(
            {"arrival": None, "child": "arrival", "later": None}
        )
        decisions, events = [], []
        store = _make_store(
            trace,
            1,
            {"arrival": 0, "child": 2, "later": -1},
            protection="direct_child",
            decisions=decisions,
            events=events,
        )
        store.admit("child", 1, 0, 0)
        self.assertTrue(store.admit("arrival", 1, 1, 1))
        self.assertEqual(list(store.cached), ["arrival"])
        decisions.clear()
        events.clear()
        self.assertFalse(store.admit("later", 1, 2, 2))
        self.assertEqual(list(store.cached), ["arrival"])
        self.assertEqual(decisions[0][0], ("later", "arrival"))
        self.assertEqual(decisions[0][-1], 0)
        self.assertEqual(events, [])


class ArrivalProtectionPublicApiTests(unittest.TestCase):
    @staticmethod
    def _mixed_trace():
        chains = []
        for step in range(48):
            chains.append(("s1", "s2") if step % 4 == 0 else ("s1", f"s{100 + step}"))
        return _request_trace(chains)

    def _run_none(self, policy, explicit):
        decisions, removals, protection_events = [], [], []
        kwargs = dict(
            l2_eviction="sampled",
            l2_sample_width=3,
            l2_seed=19,
            l2_decision_hook=_decision_recorder(decisions),
            l2_removal_hook=_removal_recorder(removals),
        )
        if policy == "learned":
            kwargs["l2_scorer"] = _NumericScorer()
        if explicit:
            kwargs["l2_arrival_protection"] = "none"
            kwargs["l2_protection_hook"] = _protection_recorder(protection_events)
        result = run_two_tier(
            self._mixed_trace(),
            "lru",
            3 * 512,
            6 * 512,
            policy,
            bytes_per_token=1,
            **kwargs,
        )
        return result, decisions, removals, protection_events

    def test_omitted_and_explicit_none_are_bit_identical_for_generic_and_learned(self):
        for policy in ("lru", "learned"):
            with self.subTest(policy=policy):
                omitted = self._run_none(policy, explicit=False)
                explicit = self._run_none(policy, explicit=True)
                self.assertGreater(omitted[0].l2_decisions, 0)
                self.assertEqual(asdict(explicit[0]), asdict(omitted[0]))
                self.assertEqual(explicit[1], omitted[1])
                self.assertEqual(explicit[2], omitted[2])
                self.assertEqual(explicit[3], [])

    def test_public_validation_and_disabled_l2_keep_their_boundaries(self):
        trace = _request_trace((("s1",), ("s2",)))
        with self.assertRaises(ValueError):
            run_two_tier(
                trace,
                "lru",
                512,
                512,
                "lru",
                bytes_per_token=1,
                l2_eviction="sampled",
                l2_sample_width=0,
                l2_arrival_protection="all",
            )
        with self.assertRaises(ValueError):
            run_two_tier(
                trace,
                "lru",
                512,
                512,
                "lru",
                bytes_per_token=1,
                l2_eviction="sampled",
                l2_arrival_protection="future",
            )
        # A disabled L2 follows the old path even if active-mode-only settings
        # would be invalid for a live store.
        result = run_two_tier(
            trace,
            "lru",
            512,
            0,
            "lru",
            bytes_per_token=1,
            l2_eviction="sampled",
            l2_sample_width=0,
            l2_arrival_protection="all",
        )
        self.assertEqual(result.l2_admissions, 0)

    def test_protection_hook_counts_the_whole_trace_and_preserves_context(self):
        trace = _request_trace((("s1",), ("s2",), ("s3",), ("s4",)))
        events = []
        result = run_two_tier(
            trace,
            "lru",
            512,
            512,
            "learned",
            bytes_per_token=1,
            measure_from_ms=2500.0,
            l2_eviction="sampled",
            l2_sample_width=1,
            l2_scorer=_NewestLowScorer(),
            l2_arrival_protection="all",
            l2_protection_hook=_protection_recorder(events),
        )
        self.assertEqual(result.measured_requests, 1)
        self.assertEqual(result.l1_evictions, 3)
        self.assertEqual(
            Counter(event for event, _, _ in events),
            Counter(
                {
                    "protected_offer": 3,
                    "protected_overflow_offer": 2,
                    "first_round_override": 2,
                }
            ),
        )
        protected_context = [
            (timestamp_ms, group_index)
            for event, timestamp_ms, group_index in events
            if event == "protected_offer"
        ]
        self.assertEqual(protected_context, [(1000.0, 1), (2000.0, 2), (3000.0, 3)])
        self.assertTrue(any(timestamp_ms < 2500.0 for _, timestamp_ms, _ in events))


class Phase1CollectorTests(unittest.TestCase):
    def test_paired_summary_uses_fixed_left_minus_right_direction(self):
        left = [
            (1, 1000.0, 20, 2, "a", 1, 4, 10),
            (2, 2000.0, 30, 3, "b", 1, 4, 3),
        ]
        right = [
            (1, 1000.0, 20, 2, "a", 1, 4, 5),
            (2, 2000.0, 30, 3, "b", 1, 4, 7),
        ]
        row = paired_request_summary(left, right)
        self.assertEqual(row["saved_tokens"], 5)
        self.assertEqual(row["lost_tokens"], 4)
        self.assertEqual(row["net_avoided_tokens"], 1)
        self.assertEqual((row["saved_requests"], row["lost_requests"]), (1, 1))
        self.assertEqual(row["net_avoided_tokens"], row["saved_tokens"] - row["lost_tokens"])

        changed_identity = list(right)
        changed_identity[1] = (*changed_identity[1][:4], "other", *changed_identity[1][5:])
        with self.assertRaisesRegex(AssertionError, "identity or L1 prefix"):
            paired_request_summary(left, changed_identity)

    def test_composite_collector_checks_replay_and_windows_protection_events(self):
        trace = _request_trace(tuple((f"s{index}",) for index in range(1, 7)))
        collector = Phase1Collector(trace, measure_from_ms=3000.0)
        block_bytes = 512 * 2048
        result = run_two_tier(
            trace,
            "lru",
            block_bytes,
            block_bytes,
            "learned",
            l2_eviction="sampled",
            l2_sample_width=1,
            l2_scorer=_NewestLowScorer(),
            measure_from_ms=3000.0,
            l2_arrival_protection="all",
            l2_request_hook=collector.on_request,
            l2_removal_hook=collector,
            l2_protection_hook=collector.on_protection,
        )
        collector.finish(result)
        counts = collector.protection_row()
        self.assertEqual(result.measured_requests, 3)
        self.assertEqual(counts["protected_offer_count"], 5)
        self.assertEqual(counts["protected_offer_window_count"], 3)
        self.assertEqual(counts["protected_overflow_offer_count"], 4)
        self.assertEqual(counts["protected_overflow_offer_window_count"], 3)
        self.assertEqual(counts["first_round_override_count"], 4)
        self.assertEqual(counts["first_round_override_window_count"], 3)

    def test_seed_varying_sampled_tokens_are_aggregated_as_a_metric(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_phase1_intervention.py"
        spec = importlib.util.spec_from_file_location("phase1_runner_test", path)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        rows = [
            {"group": "cell", "best_sampled_arm": "lru_s", "tokens": 10},
            {"group": "cell", "best_sampled_arm": "lru_s", "tokens": 20},
        ]
        out = runner.aggregate(
            rows,
            ("group",),
            ("tokens",),
            carry=("best_sampled_arm",),
        )
        self.assertEqual(out[0]["best_sampled_arm"], "lru_s")
        self.assertEqual(out[0]["tokens_mean"], 15.0)
        self.assertEqual(out[0]["tokens_seed_min"], 10.0)
        self.assertEqual(out[0]["tokens_seed_max"], 20.0)


if __name__ == "__main__":
    unittest.main()
