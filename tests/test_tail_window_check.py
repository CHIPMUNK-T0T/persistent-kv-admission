"""Tests for the evaluation-window check (docs/tail-window-check-plan.md).

No real trace is replayed: the plan forbids a real replay before the
implementation is reviewed and committed. Everything here runs on constructed
traces, hand-built rows and the published result tables, which are only read.
What has to hold before a real replay means anything: the arms, windows and
sources are the plan's; the request hook that calls both collectors is
read-only (a replay with and without it gives identical counters and digests,
and the attribution identities hold); the head counters are
`LabelWindowUtilityCollector`'s arithmetic, equal the full counters when no
request falls after the cut, and equal a replay of the trace cut at the window
end where the cut cannot change decisions; tail = full - head; every arm is
built exactly as its parent built it (decision by decision, digests equal,
and the runner's worker equals each parent's worker digest for digest); the
reading helpers do what the plan fixes at their boundaries; on the published
rows the full-window derivation is the published reading value for value; and
the runner's checks, refusals, its end-to-end run with git mocked (references
produced by the parents' own workers) and the read-only tabulation behave as
stated, including that a tampered reference or published reading publishes
nothing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

from persistent_kv_admission import errorloc, horizonctl, horizonfill, matchedorder, tailwindow
from persistent_kv_admission.attribution import AttributionCollector, LossPartitionError
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import horizon_arm
from persistent_kv_admission.matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_READINGS,
    PUBLISHED_ARMS,
    ClassStatistics,
    matched_arm,
)
from persistent_kv_admission.onpolicy import LabelWindowUtilityCollector, sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.tailwindow import (
    ARMS,
    COUNTERS,
    WINDOWS,
    ChainedRequestHook,
    cell_arms,
    full_row,
    head_row,
    tail_row,
    window_columns,
)
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The horizon control's fixtures (constructed traces at one step per 5 s, the
# 600-second label horizon, the fixed linear ranker, the decision tracer),
# reused as they are; only plain functions and classes are read from them.
horizon_tests = _load("_tail_horizon_control_fixtures", REPOSITORY / "tests/test_horizon_control.py")
runner = _load("_run_tail_window_check_under_test", REPOSITORY / "scripts/run_tail_window_check.py")
# The fill-in's runner: the parent of `label_binary_150`, whose own worker
# produces that arm's reference rows below. It brings its own chain of runners.
fill_runner = _load("_tail_horizon_fill_parent", REPOSITORY / "scripts/run_horizon_fill.py")
tabulate = _load("_tabulate_tail_window_check_under_test",
                 REPOSITORY / "scripts/tabulate_tail_window_check.py")

HORIZON = horizon_tests.HORIZON                    # 600 s
MEASURE_FROM_MS = horizon_tests.MEASURE_FROM_MS    # 600,000 ms
L1_BYTES = horizon_tests.L1_BYTES
L2_BYTES = horizon_tests.L2_BYTES
partial_trace = horizon_tests.partial_trace
_LinearRanker = horizon_tests._LinearRanker
_Tracer = horizon_tests._Tracer
csv_rows = horizon_tests.csv_rows
SEEDS = (0, 1, 2, 3, 4)
VOLATILE = ("seconds", "worker_peak_rss_mib", "worker_pss_mib_end")
# Columns of a parent's row that name the arm within that run's design: the
# error-location runner's `swaps`, the matched class-order runner's `role`
# (new / reproduced) and `published_arm`.
PARENT_NAMING = ("swaps", "role", "published_arm")
ALL_CELLS = tuple(MATCHED_HORIZON_SECONDS)


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def head_end(trace) -> float:
    return trace.end_ms - 600_000.0


def parent_setup(arm, trace, seed=1):
    """The arm as its published run built it, by the parent's own builder."""
    ranker = _LinearRanker()
    if arm in tailwindow.ERROR_LOCATION_ARMS:
        return errorloc.arm_setup(arm, trace, HORIZON, seed, learned_ranker=ranker)
    if arm == "label_binary_150":
        return horizonfill.arm_setup(arm, trace)
    if arm in horizonctl.HORIZON_OF_ARM:
        return horizonctl.arm_setup(arm, trace, HORIZON, seed, learned_ranker=ranker)
    return matchedorder.arm_setup(arm, trace, ranker)


def replay(trace, arm, setup, seed=1, chained=True, cut_ms=None, extra_hook=None,
           measure_from_ms=MEASURE_FROM_MS):
    """One traced all16 replay with the attribution collector, the
    error-location statistics (and, for a class-order arm, the class
    statistics at its h), as the runner nests them; with `chained`, the head
    collector (cut at `cut_ms`, default trace end - 600 s) beside the
    attribution collector through `ChainedRequestHook`; `extra_hook` is chained
    last. Returns a dict of the result, collectors, digests and decisions."""
    collector = AttributionCollector(trace, measure_from_ms=measure_from_ms, bytes_per_token=1)
    statistics = DecisionStatistics(trace, HORIZON, measure_from_ms)
    classes = None
    if tailwindow.is_class_order_arm(arm):
        classes = ClassStatistics(trace, tailwindow.arm_horizon(arm), measure_from_ms)
        override = RecordingOverride(statistics, RecordingOverride(classes, setup.override))
    else:
        override = RecordingOverride(statistics, setup.override)
    tracer = _Tracer(override)
    window = None
    hooks = [collector.on_request]
    if chained:
        window = LabelWindowUtilityCollector(trace, measure_from_ms,
                                             head_end(trace) if cut_ms is None else cut_ms)
        hooks.append(window.on_request)
    if extra_hook is not None:
        hooks.append(extra_hook)
    request_hook = ChainedRequestHook(*hooks) if len(hooks) > 1 else hooks[0]
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                          measure_from_ms=measure_from_ms, l2_eviction="sampled",
                          l2_sample_width=4, l2_seed=seed, l2_eligibility="all", l2_arm=arm,
                          l2_request_hook=request_hook, l2_removal_hook=collector,
                          l2_override_hook=tracer, **setup.replay_arguments())
    collector.check_against(result)
    return {"result": result, "collector": collector, "window": window, "tracer": tracer,
            "counters": runner.rel._counters_digest(result, collector),
            "decisions": statistics.row()["decision_sha256"],
            "classes": classes.row() if classes is not None else None}


class _Recorder:
    """A request hook that keeps a copy of every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, ids, prefix, l2_hits, present, timestamp_ms, group_index, measured):
        self.calls.append((tuple(ids), prefix, tuple(l2_hits), tuple(present), timestamp_ms,
                           group_index, measured))


def recount(trace, calls, start_ms, end_ms):
    """The window's counters recomputed from recorded request-hook calls with
    the replay's own expressions (`run_two_tier`'s request loop)."""
    states = trace.states
    out = dict.fromkeys(COUNTERS, 0)
    for ids, prefix, l2_hits, present, timestamp_ms, _, _ in calls:
        if not start_ms <= timestamp_ms <= end_ms:
            continue
        out["measured_requests"] += 1
        out["requested_tokens"] += states[ids[-1]].prefix_tokens
        out["requested_blocks"] += len(ids)
        out["l1_avoided_tokens"] += states[ids[prefix - 1]].prefix_tokens if prefix else 0
        out["l2_avoided_tokens"] += sum(states[ids[index]].block_tokens for index in l2_hits)
        out["l2_hit_blocks"] += len(l2_hits)
        for index in present:
            if index not in l2_hits:
                out["l2_present_unusable_blocks"] += 1
                out["l2_present_unusable_tokens"] += states[ids[index]].block_tokens
    out["avoided_prefill_tokens"] = out["l1_avoided_tokens"] + out["l2_avoided_tokens"]
    return out


# --- the grid --------------------------------------------------------------------------------------


class GridTests(unittest.TestCase):
    def test_the_arms_windows_and_sources_are_the_plans(self):
        self.assertEqual(tailwindow.TAIL_SECONDS, 600.0)
        self.assertEqual(tailwindow.TAIL_MS, 600_000.0)
        self.assertEqual(WINDOWS, ("full", "head", "tail"))
        self.assertEqual(ARMS, ("lru", "learned", "label", "evict_label", "label_binary_h*",
                                "evict_binary_h*_learned", "evict_binary_h*_recency"))
        expected = {
            (0.0025, 1.0): 60, (0.0025, 4.0): 150, (0.01, 1.0): 150, (0.01, 4.0): 600,
            (0.02, 1.0): 300, (0.02, 4.0): 600}
        self.assertEqual(set(expected), set(runner.CELLS))
        for cell, h in expected.items():
            self.assertEqual(cell_arms(*cell),
                             ("lru", "learned", "label", "evict_label", f"label_binary_{h}",
                              f"evict_binary_{h}_learned", f"evict_binary_{h}_recency"))
        sources = {"lru": "error_location_001", "learned": "error_location_001",
                   "label": "error_location_001", "evict_label": "error_location_001",
                   "label_binary_60": "horizon_control_001", "label_binary_150": "horizon_fill_001",
                   "label_binary_300": "horizon_control_001",
                   "label_binary_600": "horizon_control_001",
                   **{matched_arm(h, order): "matched_class_order_001"
                      for h in (60, 150, 300, 600) for order in ("learned", "recency")}}
        self.assertEqual({arm: source for (_, arm), source in runner.REFERENCES.items()}, sources)
        self.assertEqual({mechanism for mechanism, _ in runner.REFERENCES}, {"all16"})
        self.assertEqual({name: str(path.relative_to(REPOSITORY))
                          for name, path in runner.REFERENCE_SOURCES.items()},
                         {name: f"results/paper/{name}/replay_seeds.csv"
                          for name in ("error_location_001", "horizon_control_001",
                                       "horizon_fill_001", "matched_class_order_001")})
        self.assertEqual({name: str(path.relative_to(REPOSITORY))
                          for name, path in runner.PUBLISHED_READING_SOURCES.items()},
                         {"horizon": "results/paper/horizon_control_001/horizon.csv",
                          "fill": "results/paper/horizon_fill_001/fill.csv",
                          "class_order": "results/paper/matched_class_order_001/class_order.csv",
                          "order": "results/paper/matched_class_order_001/order.csv",
                          "admission": "results/paper/matched_class_order_001/admission.csv"})
        self.assertEqual(runner.DEFAULT_PAPER_DIR,
                         REPOSITORY / "results/paper/tail_window_check_001")
        self.assertEqual(runner.PLAN_PATH, REPOSITORY / "docs/tail-window-check-plan.md")
        self.assertEqual(runner.reference_cells("all16", "label_binary_150"),
                         ((0.0025, 4.0), (0.01, 1.0)))
        self.assertEqual(runner.reference_cells("all16", "lru"), tuple(runner.CELLS))
        self.assertEqual(tailwindow.PREDICTED_BELOW_CELL, (0.0025, 1.0))

    def test_roles_resolve_both_ways(self):
        for cell in ALL_CELLS:
            for role, arm in zip(ARMS, cell_arms(*cell)):
                self.assertEqual(tailwindow.arm_role(arm), role)
                self.assertEqual(tailwindow.resolve_arm(role, MATCHED_HORIZON_SECONDS[cell]), arm)
        self.assertIsNone(tailwindow.arm_horizon("evict_label"))
        self.assertEqual(tailwindow.arm_horizon("label_binary_150"), 150.0)
        self.assertEqual(tailwindow.arm_horizon("evict_binary_300_recency"), 300.0)
        for bad in ("label_binary_90", "evict_binary_learned", "pi3_binary"):
            with self.assertRaises(ValueError):
                tailwindow.arm_role(bad)
        with self.assertRaises(ValueError):
            tailwindow.resolve_arm("offline", 60.0)

    def test_the_tasks(self):
        tasks = runner.build_tasks(runner.TRACES, runner.CELLS, runner.SEEDS)
        self.assertEqual(len(tasks), 420)
        self.assertEqual(len(set(tasks)), 420)
        by_cell = defaultdict(set)
        for name, fraction, multiplier, arm, seed in tasks:
            by_cell[(name, fraction, multiplier)].add(arm)
        self.assertEqual(len(by_cell), 12)
        for (name, fraction, multiplier), arms in by_cell.items():
            self.assertEqual(arms, set(cell_arms(fraction, multiplier)))
        costs = [runner.arm_cost(task[3]) for task in tasks]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual((runner.DEFAULT_WORKERS, runner.MAX_WORKERS), (10, 12))
        self.assertEqual((runner.ELIGIBILITY, runner.WIDTH, runner.MECHANISM), ("all", 16, "all16"))


# --- the chained hook ------------------------------------------------------------------------------


class ChainedRequestHookTests(unittest.TestCase):
    def test_every_hook_is_called_in_order_with_the_same_objects(self):
        calls = []
        first = lambda *args, **kwargs: calls.append(("first", args, kwargs)) or "ignored"
        second = lambda *args, **kwargs: calls.append(("second", args, kwargs))
        hook = ChainedRequestHook(first, second)
        ids, hits, present = ["a", "b"], [1], [1]
        self.assertIsNone(hook(ids, 1, hits, present, 5.0, 3, True))
        self.assertEqual([name for name, _, _ in calls], ["first", "second"])
        for _, args, kwargs in calls:
            self.assertIs(args[0], ids)
            self.assertIs(args[2], hits)
            self.assertIs(args[3], present)
            self.assertEqual(args[4:], (5.0, 3, True))
            self.assertEqual(kwargs, {})
        hook(ids, 1, hits, present, 5.0, 3, measured=False)
        self.assertEqual(calls[-1][2], {"measured": False})
        with self.assertRaises(ValueError):
            ChainedRequestHook()
        with self.assertRaises(TypeError):
            ChainedRequestHook(first, "not callable")

    def test_the_chained_hook_is_read_only(self):
        """The plan's check: a replay with and without the hook that calls both
        collectors gives identical counters, digests and decisions, for the
        seven arms of every cell, and the attribution identities hold in both
        (`check_against` inside `replay`)."""
        trace, _ = partial_trace(self, seed=9)
        compared = 0
        for cell in ALL_CELLS:
            for arm in cell_arms(*cell):
                with self.subTest(arm=arm):
                    alone = replay(trace, arm, tailwindow.arm_setup(arm, trace, HORIZON, 1,
                                                                    _LinearRanker()),
                                   chained=False)
                    both = replay(trace, arm, tailwindow.arm_setup(arm, trace, HORIZON, 1,
                                                                   _LinearRanker()))
                    self.assertEqual(both["result"].as_row(), alone["result"].as_row())
                    self.assertEqual(both["counters"], alone["counters"])
                    self.assertEqual(both["decisions"], alone["decisions"])
                    self.assertEqual(both["tracer"].decisions, alone["tracer"].decisions)
                    self.assertEqual(both["collector"].loss_row(), alone["collector"].loss_row())
                    self.assertEqual(both["classes"], alone["classes"])
                    self.assertGreater(both["result"].l2_decisions, 0)
                    self.assertGreater(both["window"].measured_requests, 0)
                    compared += 1
        self.assertEqual(compared, 6 * 7)

    def test_the_read_only_comparison_has_teeth(self):
        """A hook in the chain that changes its arguments (drops the L2 hits
        the replay is about to count) changes the counters, so the comparison
        above would see it; placed after the attribution collector, the Phase
        0.98b identities fail as well."""
        trace, _ = partial_trace(self, seed=9)

        def mutating(ids, prefix, l2_hits, present, timestamp_ms, group_index, measured):
            del l2_hits[:]

        setup = lambda: tailwindow.arm_setup("label", trace, HORIZON, 1, _LinearRanker())
        alone = replay(trace, "label", setup(), chained=False)
        with self.assertRaises(LossPartitionError):
            replay(trace, "label", setup(), extra_hook=mutating)
        collector = AttributionCollector(trace, measure_from_ms=MEASURE_FROM_MS,
                                         bytes_per_token=1)
        changed = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                               measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled",
                               l2_sample_width=4, l2_seed=1, l2_eligibility="all",
                               l2_request_hook=ChainedRequestHook(collector.on_request, mutating),
                               l2_removal_hook=collector, **setup().replay_arguments())
        self.assertNotEqual(changed.as_row(), alone["result"].as_row())
        self.assertLess(changed.l2_avoided_tokens, alone["result"].l2_avoided_tokens)

    def test_the_hooks_see_unchanged_arguments(self):
        """Neither collector changes what the next hook sees: a recorder
        before both collectors and one after them, in the same chain, record
        the same arguments at every request (each copies them when called)."""
        trace, _ = partial_trace(self, seed=9)
        for arm in ("evict_label", "evict_binary_60_learned"):
            with self.subTest(arm=arm):
                setup = tailwindow.arm_setup(arm, trace, HORIZON, 1, _LinearRanker())
                first, last = _Recorder(), _Recorder()
                collector = AttributionCollector(trace, measure_from_ms=MEASURE_FROM_MS,
                                                 bytes_per_token=1)
                window = tailwindow.head_collector(trace, MEASURE_FROM_MS)
                result = run_two_tier(
                    trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                    measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled", l2_sample_width=4,
                    l2_seed=1, l2_eligibility="all", l2_arm=arm,
                    l2_request_hook=ChainedRequestHook(first, collector.on_request,
                                                       window.on_request, last),
                    l2_removal_hook=collector, l2_override_hook=setup.override,
                    **setup.replay_arguments())
                collector.check_against(result)
                self.assertEqual(first.calls, last.calls)
                self.assertEqual(len(first.calls), len(trace.requests))
                self.assertTrue(any(call[2] for call in first.calls))      # some L2 hits


# --- the windows -----------------------------------------------------------------------------------


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.trace, self.path = partial_trace(self, seed=9)

    def test_the_bounds(self):
        bounds = tailwindow.window_bounds(self.trace, MEASURE_FROM_MS)
        self.assertEqual(bounds["end_ms"], self.trace.end_ms)
        self.assertEqual(bounds["head_end_ms"], self.trace.end_ms - 600_000.0)
        self.assertEqual(bounds["full_seconds"], (self.trace.end_ms - MEASURE_FROM_MS) / 1000.0)
        self.assertEqual(bounds["head_seconds"] + 600.0, bounds["full_seconds"])
        collector = tailwindow.head_collector(self.trace, MEASURE_FROM_MS)
        self.assertEqual((collector.start_ms, collector.end_ms),
                         (MEASURE_FROM_MS, self.trace.end_ms - 600_000.0))

    def test_head_counters_are_the_collectors_arithmetic(self):
        """The head counters are the collector's row under the replay's own
        expressions: recomputed from the recorded request-hook calls over
        split <= t <= end - 600 s they are equal, and over t >= split the same
        recount is the replay's full counters."""
        for arm in ("lru", "evict_label", "label_binary_150", "evict_binary_300_learned"):
            with self.subTest(arm=arm):
                recorder = _Recorder()
                out = replay(self.trace, arm, tailwindow.arm_setup(arm, self.trace, HORIZON, 1,
                                                                   _LinearRanker()),
                             extra_hook=recorder)
                head = head_row(out["window"])
                self.assertEqual(head, recount(self.trace, recorder.calls, MEASURE_FROM_MS,
                                               head_end(self.trace)))
                row = out["window"].row()
                self.assertEqual(head, {counter: row[tailwindow.COLLECTOR_COLUMNS[counter]]
                                        for counter in COUNTERS})
                self.assertEqual(full_row(out["result"]),
                                 recount(self.trace, recorder.calls, MEASURE_FROM_MS, math.inf))
                self.assertGreater(head["measured_requests"], 0)
                self.assertGreater(head["l2_avoided_tokens"], 0)

    def test_tail_is_full_minus_head_and_the_requests_after_the_cut(self):
        for arm in ("lru", "label", "evict_binary_150_recency"):
            with self.subTest(arm=arm):
                after_cut = LabelWindowUtilityCollector(
                    self.trace, math.nextafter(head_end(self.trace), math.inf), self.trace.end_ms)
                out = replay(self.trace, arm, tailwindow.arm_setup(arm, self.trace, HORIZON, 1,
                                                                   _LinearRanker()),
                             extra_hook=after_cut.on_request)
                full, head = full_row(out["result"]), head_row(out["window"])
                tail = tail_row(full, head)
                self.assertEqual(tail, {counter: full[counter] - head[counter]
                                        for counter in COUNTERS})
                self.assertEqual(tail, head_row(after_cut))
                self.assertTrue(all(value >= 0 for value in tail.values()))
                self.assertGreater(tail["measured_requests"], 0)
                columns = window_columns(full, head, tail)
                for window, counters in zip(WINDOWS, (full, head, tail)):
                    self.assertEqual(tailwindow.counters_of(columns, window), counters)
                    self.assertEqual(columns[f"U_{window}_points"],
                                     100.0 * (counters["avoided_prefill_tokens"]
                                              - counters["l1_avoided_tokens"])
                                     / counters["requested_tokens"])

    def test_with_no_request_after_the_cut_head_is_full(self):
        """Cut at or after the last request, the head collector counts the
        full window, for the seven arms of every cell."""
        for cell in ALL_CELLS:
            for arm in cell_arms(*cell):
                with self.subTest(arm=arm):
                    out = replay(self.trace, arm,
                                 tailwindow.arm_setup(arm, self.trace, HORIZON, 1, _LinearRanker()),
                                 cut_ms=self.trace.end_ms)
                    full, head = full_row(out["result"]), head_row(out["window"])
                    self.assertEqual(head, full)
                    self.assertEqual(set(tail_row(full, head).values()), {0})

    def test_head_is_the_replay_of_the_cut_trace_where_the_cut_cannot_change_decisions(self):
        """The plan's unit test: replay the trace cut at the window end (no
        request after the cut). Sampled LRU reads no future, so the cut cannot
        change a decision: every decision up to the cut is the same, and the
        cut trace's full counters are the head counters of the whole trace."""
        cut = head_end(self.trace)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        temporary, cut_trace, _ = horizon_tests.build([record for record in records
                                                       if record["timestamp"] <= cut])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(cut_trace.end_ms, max(t for t in (r.timestamp_ms for r in
                                                           self.trace.requests) if t <= cut))
        self.assertLess(len(cut_trace.requests), len(self.trace.requests))
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                whole = replay(self.trace, "lru",
                               tailwindow.arm_setup("lru", self.trace, HORIZON, seed, None),
                               seed=seed)
                cut_out = replay(cut_trace, "lru",
                                 tailwindow.arm_setup("lru", cut_trace, HORIZON, seed, None),
                                 seed=seed, cut_ms=cut_trace.end_ms)
                self.assertEqual(head_row(whole["window"]), full_row(cut_out["result"]))
                self.assertEqual(head_row(cut_out["window"]), full_row(cut_out["result"]))
                before = [decision for decision in whole["tracer"].decisions
                          if decision[0] <= cut]
                self.assertEqual(before, cut_out["tracer"].decisions)
                self.assertGreater(len(before), 0)

    def test_utility_and_difference_arithmetic(self):
        counters = dict.fromkeys(COUNTERS, 0)
        counters.update(requested_tokens=400, l1_avoided_tokens=100, avoided_prefill_tokens=130)
        self.assertEqual(tailwindow.extra_tokens(counters), 30)
        self.assertEqual(tailwindow.utility_points(counters), 7.5)
        self.assertEqual(tailwindow.points(5, 0), 500.0)          # the parents' max(., 1) guard
        later = window_columns(counters, counters, tail_row(counters, counters))
        earlier_counters = dict(counters, avoided_prefill_tokens=110)
        earlier = window_columns(earlier_counters, earlier_counters,
                                 tail_row(earlier_counters, earlier_counters))
        self.assertEqual(tailwindow.window_difference(later, earlier, "full"), (20, 5.0))
        self.assertEqual(tailwindow.window_difference(later, earlier, "tail"), (0, 0.0))
        later.update(extra_avoided_tokens=30, requested_tokens=400)
        earlier.update(extra_avoided_tokens=10, requested_tokens=400)
        self.assertEqual(tailwindow.window_difference(later, earlier, "full"),
                         runner.hc._difference_points(later, earlier))
        self.assertEqual(tailwindow.window_utility(later, "full"), runner.hc._u(later))
        with self.assertRaises(ValueError):
            tailwindow.counters_of(later, "middle")


# --- the arms --------------------------------------------------------------------------------------


class ArmSetupTests(unittest.TestCase):
    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)

    def test_each_arm_is_built_as_its_parent_decision_by_decision(self):
        for cell in ALL_CELLS:
            for arm in cell_arms(*cell):
                with self.subTest(arm=arm):
                    mine = tailwindow.arm_setup(arm, self.trace, HORIZON, 1, _LinearRanker())
                    theirs = parent_setup(arm, self.trace)
                    self.assertEqual(type(mine.scorer), type(theirs.scorer))
                    self.assertEqual(type(mine.override), type(theirs.override))
                    self.assertEqual(mine.replay_arguments().keys(),
                                     theirs.replay_arguments().keys())
                    self.assertEqual(mine.l2_policy, theirs.l2_policy)
                    left = replay(self.trace, arm, mine)
                    right = replay(self.trace, arm, theirs)
                    self.assertEqual(left["tracer"].decisions, right["tracer"].decisions)
                    self.assertEqual(left["counters"], right["counters"])
                    self.assertEqual(left["decisions"], right["decisions"])
                    self.assertEqual(left["classes"], right["classes"])
                    self.assertGreater(len(left["tracer"].decisions), 0)

    def test_the_rung_builders_agree_where_both_apply(self):
        """`label_binary_60/300/600` come from the horizon control and
        `label_binary_150` from the fill-in; where both builders apply they are
        the same arm, so the dispatch cannot move a decision."""
        for h in (60.0, 300.0, 600.0):
            arm = horizon_arm(h)
            control = horizonctl.arm_setup(arm, self.trace, HORIZON, 1)
            fill = horizonfill.arm_setup(arm, self.trace)
            self.assertEqual(replay(self.trace, arm, control)["tracer"].decisions,
                             replay(self.trace, arm, fill)["tracer"].decisions)
        with self.assertRaises(ValueError):
            horizonctl.arm_setup("label_binary_150", self.trace, HORIZON, 1)

    def test_unknown_arms_and_missing_rankers_are_refused(self):
        for arm in ("label_binary_90", "evict_binary_learned", "offline"):
            with self.assertRaises(ValueError):
                tailwindow.arm_setup(arm, self.trace, HORIZON, 1, _LinearRanker())
        for arm in ("learned", "evict_label", "evict_binary_60_recency"):
            with self.assertRaises(ValueError):
                tailwindow.arm_setup(arm, self.trace, HORIZON, 1, None)
        self.assertIsNone(tailwindow.arm_setup("lru", self.trace, HORIZON, 1, None).scorer)


# --- the worker against each parent's worker ------------------------------------------------------


def _shared(traces):
    """The module state each runner's main sets for these traces."""
    shared = {"traces": traces, "groups": {}, "splits": {}, "horizons": {},
              "rankers": {name: _LinearRanker() for name in traces}, "real_rankers": {}}
    working_set = {}
    for name, trace in traces.items():
        horizon, split_ms, _ = horizon_for(trace)
        assert horizon == 600.0, (name, horizon)
        shared["groups"][name] = _occurrence_groups(trace)
        shared["splits"][name] = split_ms
        shared["horizons"][name] = horizon
        working_set[name] = working_set_bytes(trace)
    return shared, working_set


def _parent_patches(shared, working_set):
    """Every runner's module state, the parents' and this run's."""
    return [mock.patch.dict(runner.SHARED, shared), mock.patch.dict(runner.mco.SHARED, shared),
            mock.patch.dict(runner.hc.SHARED, shared), mock.patch.dict(runner.rel.SHARED, shared),
            mock.patch.dict(runner.rdp._SHARED, {"working_set": working_set}),
            mock.patch.dict(fill_runner.SHARED, shared),
            mock.patch.dict(fill_runner.rdp._SHARED, {"working_set": working_set})]


def parent_worker(name, fraction, multiplier, arm, seed) -> dict:
    """The replay of `arm` by the worker of the run that published it."""
    source = tailwindow.reference_source(arm)
    if source == "error_location_001":
        return runner.rel._replay_worker((name, fraction, multiplier, arm, seed, "main"))
    if source == "horizon_control_001":
        return runner.hc._replay_worker((name, fraction, multiplier, arm, seed, "main"))
    if source == "horizon_fill_001":
        return fill_runner._replay_worker((name, fraction, multiplier, arm, seed))
    return runner.mco._replay_worker((name, fraction, multiplier, arm, seed))


class RunnerWorkerTests(unittest.TestCase):
    """`_replay_worker` on a constructed trace, with the module state the
    runner's main would set, against the worker of each arm's parent run."""

    def setUp(self):
        self.trace, _ = partial_trace(self, seed=9)
        self.name = self.trace.name
        shared, working_set = _shared({self.name: self.trace})
        for patch in _parent_patches(shared, working_set):
            patch.start()
            self.addCleanup(patch.stop)

    def test_every_arm_is_its_parents_worker_digest_for_digest(self):
        for fraction, multiplier in ALL_CELLS:
            for arm in cell_arms(fraction, multiplier):
                with self.subTest(cell=(fraction, multiplier), arm=arm):
                    mine = runner._replay_worker((self.name, fraction, multiplier, arm, 1))
                    theirs = parent_worker(self.name, fraction, multiplier, arm, 1)
                    for column in ("counters_sha256", "decision_sha256", "avoided_prefill_tokens"):
                        self.assertEqual(mine[column], theirs[column])
                    differing = [key for key in theirs if key not in VOLATILE + PARENT_NAMING
                                 and not _same(mine.get(key), theirs[key])]
                    self.assertEqual(differing, [])
                    added = set(mine) - set(theirs)
                    self.assertTrue({"arm_role", "h_star", "U_head_points",
                                     "tail_requested_tokens"} <= added)
                    for counter in ("requested_tokens", "l1_avoided_tokens",
                                    "avoided_prefill_tokens", "measured_requests"):
                        self.assertEqual(mine[f"full_{counter}"], theirs[counter])
                    self.assertEqual(mine["U_full_points"], theirs["extra_points"])

    def test_worker_rows_pass_the_checks(self):
        rows = [runner._replay_worker((self.name, fraction, multiplier, arm, 0))
                for fraction, multiplier in runner.CELLS for arm in cell_arms(fraction, multiplier)]
        self.assertEqual(runner.check_windows(rows), (6, []))
        self.assertEqual(runner.check_statistics(rows), [])
        self.assertEqual(runner.check_class(rows), (12, []))
        self.assertEqual(runner.rmc.check_invariants(rows, 7), (6, []))
        for row in rows:
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertLess(row["head_requested_tokens"], row["full_requested_tokens"])
            self.assertGreater(row["head_requested_tokens"], 0)
            self.assertEqual(row["head_requested_tokens"] + row["tail_requested_tokens"],
                             row["requested_tokens"])
            self.assertEqual(row["h_star"],
                             MATCHED_HORIZON_SECONDS[(row["l1_fraction"], row["l2_multiplier"])])
            carries = "class_horizon_seconds" in row
            self.assertEqual(carries, tailwindow.is_class_order_arm(row["arm"]))
        # A publication of these very rows passes the identifier check and the
        # reproduction.
        published = {("all16", row["arm"]) + runner._cell_seed(row):
                     runner._reference_entry(row, "all16", row["arm"], "s") for row in rows}
        self.assertEqual(runner.check_identifiers(rows, published), (42 * 7, []))
        report, table = runner.check_reproduction(rows, published, 42)
        self.assertTrue(runner.reproduction_passes(report))
        self.assertEqual((report["decision_digests_compared"], len(table)), (42, 42))

    def test_the_worker_refuses_an_arm_that_is_not_its_cells(self):
        for arm in ("evict_binary_600_learned", "label_binary_150"):
            with self.assertRaises(ValueError):
                runner._replay_worker((self.name, 0.0025, 1.0, arm, 0))


# --- the readings ----------------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_010_rule_at_its_boundary(self):
        self.assertTrue(tailwindow.suffices(0.10))
        self.assertTrue(tailwindow.suffices(-0.4))
        self.assertFalse(tailwindow.suffices(math.nextafter(0.10, 1.0)))
        self.assertFalse(tailwindow.suffices(math.nan))
        self.assertEqual(tailwindow.shortfall_reading(0.10), "reuse_label_suffices")
        self.assertEqual(tailwindow.shortfall_reading(math.nan), "order_needed")
        self.assertEqual(tailwindow.window_shortfall(50.0, 47.0, 10.0), 3.0 / 40.0)
        self.assertTrue(math.isnan(tailwindow.window_shortfall(5.0, 4.0, 5.0)))

    def test_the_09_rule_at_its_boundary(self):
        suffices, costs = MATCHED_READINGS
        self.assertEqual(tailwindow.class_order_reading(0.9), suffices)
        self.assertEqual(tailwindow.class_order_reading(math.nextafter(0.9, 0.0)), costs)
        self.assertEqual(tailwindow.class_order_reading(math.nan), costs)
        self.assertTrue(tailwindow.recovers(0.9))
        self.assertEqual(tailwindow.window_recovery(38.0, 20.0, 40.0), 0.9)
        self.assertTrue(math.isnan(tailwindow.window_recovery(38.0, 20.0, 20.0)))

    def test_sign_agreement_with_zeros_and_nans(self):
        agrees = tailwindow.sign_agrees
        self.assertTrue(agrees(2.0, 3.0))
        self.assertTrue(agrees(-1e-9, -4.0))
        self.assertFalse(agrees(-1.0, 1.0))
        self.assertTrue(agrees(0.0, 0.0))
        self.assertTrue(agrees(-0.0, 0.0))
        self.assertFalse(agrees(0.0, 1e-300))
        self.assertFalse(agrees(1e-300, 0.0))
        for left, right in ((math.nan, math.nan), (math.nan, 0.0), (0.0, math.nan),
                            (1.0, math.nan)):
            self.assertFalse(agrees(left, right))

    def test_tail_share(self):
        self.assertTrue(math.isnan(tailwindow.tail_share(0, 5)))
        self.assertEqual(tailwindow.tail_share(10, 4), 0.4)
        self.assertEqual(tailwindow.tail_share(10, -4), -0.4)
        self.assertEqual(tailwindow.tail_share(-10, 15), -1.5)
        self.assertEqual(tailwindow.tail_share(7, 0), 0.0)

    def test_the_predictions(self):
        suffices, costs = MATCHED_READINGS
        self.assertTrue(tailwindow.horizon_prediction_holds([0.1, -0.2, 0.0]))
        self.assertFalse(tailwindow.horizon_prediction_holds([0.1, math.nan]))
        self.assertFalse(tailwindow.horizon_prediction_holds([]))
        self.assertEqual(tailwindow.predicted_head_class_reading((0.0025, 1.0)), costs)
        self.assertEqual(tailwindow.predicted_head_class_reading((0.0025, 4.0)), suffices)
        holds = tailwindow.class_order_prediction_holds
        self.assertTrue(holds([((0.0025, 1.0), costs, costs), ((0.01, 4.0), suffices, suffices)]))
        # The head reading differs from the full one.
        self.assertFalse(holds([((0.01, 4.0), costs, suffices)]))
        # Equal to the full reading but not the predicted cell pattern.
        self.assertFalse(holds([((0.01, 4.0), costs, costs)]))
        self.assertFalse(holds([((0.0025, 1.0), suffices, suffices)]))
        self.assertFalse(holds([]))
        self.assertTrue(tailwindow.sign_prediction_holds([True, True]))
        self.assertFalse(tailwindow.sign_prediction_holds([True, False]))
        self.assertFalse(tailwindow.sign_prediction_holds([]))


# --- derivation ------------------------------------------------------------------------------------


def _counters(requested, l1, extra, requests, blocks):
    out = dict.fromkeys(COUNTERS, 0)
    out.update(measured_requests=requests, requested_tokens=requested, requested_blocks=blocks,
               l1_avoided_tokens=l1, l2_avoided_tokens=extra, avoided_prefill_tokens=l1 + extra,
               l2_hit_blocks=extra // 64)
    return out


# Hand-built extras of one trace x cell (0.25% x 1, h* = 60 s), the same in
# every seed: full window 1000 input tokens, head 600 (U = extra / 10 and
# extra / 6 points), tail the difference (400).
SYNTHETIC_EXTRA = {
    "lru": (100, 60), "learned": (200, 120), "label": (500, 300), "evict_label": (400, 240),
    "label_binary_h*": (470, 234), "evict_binary_h*_learned": (380, 216),
    "evict_binary_h*_recency": (410, 252)}


def synthetic_rows(trace="conversation_trace", cell=(0.0025, 1.0), seeds=SEEDS):
    rows = []
    for seed in seeds:
        for role, arm in zip(ARMS, cell_arms(*cell)):
            full_extra, head_extra = SYNTHETIC_EXTRA[role]
            full = _counters(1000, 100, full_extra, 10, 20)
            head = _counters(600, 60, head_extra, 6, 12)
            row = {"trace": trace, "l1_fraction": cell[0], "l2_multiplier": cell[1],
                   "cell": runner.rdp.cell_label(*cell), "seed": seed, "arm": arm,
                   "requested_tokens": 1000, "l1_avoided_tokens": 100,
                   "extra_avoided_tokens": full_extra}
            row.update(window_columns(full, head, tail_row(full, head)))
            rows.append(row)
    return rows


def published_replay_rows():
    """The 420 published rows of the seven arms, as replay rows whose full
    window is the published counters (the head is half of each, a
    placeholder: only the full window is compared here)."""
    rows = []
    for source, path in runner.REFERENCE_SOURCES.items():
        for row in csv_rows(path):
            key = (row["mechanism"], row["arm"])
            if runner.REFERENCES.get(key) != source or row.get("variant", "main") != "main":
                continue
            cell_seed = runner._cell_seed(row)
            if cell_seed[1:3] not in runner.reference_cells(*key):
                continue
            full = {counter: int(row.get(counter) or 0) for counter in COUNTERS}
            half = {counter: value // 2 for counter, value in full.items()}
            entry = {"trace": cell_seed[0], "l1_fraction": cell_seed[1],
                     "l2_multiplier": cell_seed[2], "seed": cell_seed[3], "arm": row["arm"],
                     "cell": row["cell"]}
            entry.update(window_columns(full, half, tail_row(full, half)))
            rows.append(entry)
    return rows


class DerivationTests(unittest.TestCase):
    def test_on_the_published_rows_the_full_window_is_the_published_reading(self):
        """The plan's "full (recomputed from the reproduced rows, equal to the
        published values)": from the published rows themselves, S_h*, both R,
        the order and admission means, seed signs and readings equal the
        published tables value for value (exact float equality)."""
        rows = published_replay_rows()
        self.assertEqual(len(rows), 420)
        tables = runner.window_tables(rows)
        readings, problems = runner.load_published_readings()
        self.assertEqual(problems, [])
        compared, problems = runner.compare_with_published(tables, readings)
        self.assertEqual((compared, problems), (120, []))
        self.assertTrue(all(entry["full_equals_published"] for name in
                            ("horizon", "class_order", "order", "admission")
                            for entry in tables[name]))
        summary = {(entry["reading"], entry["window"]): entry
                   for entry in runner.reading_summary(tables)}
        self.assertEqual(summary[("1_horizon_S_at_most_0.10", "full")]["count"], 12)
        self.assertEqual(summary[("2_class_order_R_learned_order_at_least_0.9", "full")]["count"],
                         10)
        below = {(entry["trace"], entry["cell"]) for entry in tables["class_order"]
                 if entry["reading_full"] == MATCHED_READINGS[1]}
        self.assertEqual(below, {("conversation_trace", "l1=0.0025,l2x1"),
                                 ("toolagent_trace", "l1=0.0025,l2x1")})
        signs = summary[("3_learned_minus_recency_seed_signs", "full")]
        self.assertEqual((signs["consistent_gain"], signs["consistent_loss"], signs["mixed"]),
                         (2, 8, 2))
        self.assertLessEqual(max(entry["S_full"] for entry in tables["horizon"]), 0.0282)

    def test_one_value_off_by_one_ulp_is_a_difference(self):
        rows = published_replay_rows()
        tables = runner.window_tables(rows)
        readings, _ = runner.load_published_readings()
        cell = ("toolagent_trace", 0.02, 1.0)
        changed = {key: dict(value) for key, value in readings.items()}
        changed[cell]["S"] = math.nextafter(changed[cell]["S"], math.inf)
        changed[cell]["learned_minus_recency_seed_signs"] = "+----"
        del changed[("conversation_trace", 0.01, 4.0)]["R_matched_recency"]
        compared, problems = runner.compare_with_published(tables, changed)
        self.assertEqual(compared, 119)
        self.assertEqual(len(problems), 3)
        flags = {(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"]):
                 entry["full_equals_published"] for entry in tables["horizon"]}
        self.assertFalse(flags[cell])
        self.assertEqual(sum(1 for value in flags.values() if not value), 1)
        self.assertTrue(runner._same(math.nan, math.nan))
        self.assertFalse(runner._same(0.1, math.nan))

    def test_hand_built_windows(self):
        rows = synthetic_rows()
        tables = runner.window_tables(rows)
        suffices, costs = MATCHED_READINGS
        (horizon,) = tables["horizon"]
        self.assertEqual(horizon["S_full"], (50.0 - 47.0) / (50.0 - 10.0))
        self.assertEqual(horizon["S_head"], (50.0 - 39.0) / (50.0 - 10.0))
        self.assertEqual(horizon["S_tail"], (50.0 - 59.0) / (50.0 - 10.0))
        self.assertEqual((horizon["reading_full"], horizon["reading_head"], horizon["reading_tail"]),
                         ("reuse_label_suffices", "order_needed", "reuse_label_suffices"))
        self.assertEqual((horizon["reading_head_agrees_with_full"],
                          horizon["reading_tail_agrees_with_full"]), (False, True))
        (class_order,) = tables["class_order"]
        self.assertEqual(class_order["R_matched_learned_full"], 0.9)        # exactly at the rule
        self.assertEqual(class_order["R_matched_learned_head"], 0.8)
        self.assertEqual(class_order["R_matched_learned_tail"], 21.0 / 20.0)
        self.assertEqual(class_order["R_matched_recency_full"], 21.0 / 20.0)
        self.assertEqual((class_order["reading_full"], class_order["reading_head"]),
                         (suffices, costs))
        self.assertEqual(class_order["predicted_reading_head"], costs)
        (order,) = tables["order"]
        self.assertEqual([order[f"learned_minus_recency_{window}_points_mean"]
                          for window in WINDOWS], [-3.0, -6.0, 1.5])
        self.assertEqual([order[f"learned_minus_recency_{window}_reading"] for window in WINDOWS],
                         ["consistent_loss", "consistent_loss", "consistent_gain"])
        self.assertEqual((order["learned_minus_recency_head_mean_sign_agrees_with_full"],
                          order["learned_minus_recency_tail_mean_sign_agrees_with_full"]),
                         (True, False))
        (admission,) = tables["admission"]
        self.assertEqual([admission[f"label_binary_minus_recency_{window}_points_mean"]
                          for window in WINDOWS], [6.0, -3.0, 19.5])
        self.assertFalse(admission["label_binary_minus_recency_head_mean_sign_agrees_with_full"])
        windows = {entry["arm_role"]: entry for entry in tables["windows"]}
        self.assertEqual(len(windows), 7)
        self.assertEqual((windows["label"]["U_full_points_mean"],
                          windows["label"]["U_head_points_mean"],
                          windows["label"]["U_tail_points_mean"]), (50.0, 50.0, 50.0))
        self.assertEqual((windows["lru"]["full_requested_tokens"],
                          windows["lru"]["head_requested_tokens"],
                          windows["lru"]["tail_requested_tokens"]), (1000, 600, 400))
        shares = {entry["difference"]: entry for entry in tables["tail_share"]}
        self.assertEqual(set(shares), set(tailwindow.DIFFERENCES))
        g = shares["G"]
        self.assertEqual((g["full_tokens"], g["head_tokens"], g["tail_tokens"]),
                         (1500, 900, 600))
        self.assertEqual((g["tail_share_of_difference"], g["tail_share_of_input"]), (0.4, 0.4))
        self.assertEqual(shares["learned_minus_recency"]["tail_share_of_difference"],
                         (5 * 6) / (5 * -30))
        summary = {(entry["reading"], entry["window"]): entry
                   for entry in runner.reading_summary(tables)}
        p1 = summary[("1_horizon_S_at_most_0.10", "head")]
        self.assertEqual((p1["kind"], p1["count"], p1["cells"], p1["prediction_holds"]),
                         ("prediction", 0, 1, False))
        p2 = summary[("2_class_order_R_learned_order_at_least_0.9", "head")]
        self.assertEqual((p2["count"], p2["agrees_with_full"], p2["predicted"],
                          p2["prediction_holds"]), (0, 0, 0, False))
        p3 = summary[("3_learned_minus_recency_sign_of_mean_agrees_with_full", "head")]
        self.assertEqual((p3["count"], p3["prediction_holds"]), (1, True))
        p4 = summary[("4_label_binary_minus_recency_sign_of_mean_agrees_with_full", "head")]
        self.assertEqual((p4["count"], p4["prediction_holds"]), (0, False))
        tail = summary[("3_learned_minus_recency_sign_of_mean_agrees_with_full", "tail")]
        self.assertEqual((tail["kind"], tail["count"], tail["prediction_holds"]),
                         ("descriptive", 0, ""))
        self.assertEqual(summary[("3_learned_minus_recency_seed_signs", "tail")]
                         ["consistent_gain"], 1)
        self.assertEqual(set(entry["reading"] for entry in summary.values()),
                         {"1_horizon_S_at_most_0.10", "2_class_order_R_learned_order_at_least_0.9",
                          "2_context_R_recency_at_least_0.9",
                          "3_learned_minus_recency_sign_of_mean_agrees_with_full",
                          "3_learned_minus_recency_seed_signs",
                          "4_label_binary_minus_recency_sign_of_mean_agrees_with_full",
                          "4_label_binary_minus_recency_seed_signs"})

    def test_aggregation_and_reference_rows(self):
        rows = synthetic_rows()
        for row in rows:
            for metric in runner.REPLAY_METRICS:
                row.setdefault(metric, 1.0)
            if tailwindow.is_class_order_arm(row["arm"]):
                for metric in runner.CLASS_METRICS:
                    row.setdefault(metric, 0)
            for column in ("l1_capacity_bytes", "l2_capacity_bytes"):
                row[column] = 7
        summary = runner.aggregate_replays(rows)
        self.assertEqual([entry["arm_role"] for entry in summary], list(ARMS))
        self.assertEqual(summary[2]["U_tail_points_mean"], 50.0)
        self.assertIn("class_violations_seen_mean", summary[-1])
        self.assertNotIn("class_violations_seen_mean", summary[0])
        published = {("all16", row["arm"]) + runner._cell_seed(row): {
            "source": tailwindow.reference_source(row["arm"]), "arm": row["arm"],
            "extra_avoided_tokens": row["extra_avoided_tokens"], "requested_tokens": 1000}
            for row in rows}
        references = runner.reference_rows(published, rows)
        self.assertEqual(len(references), 35)
        self.assertEqual({entry["extra_points"] for entry in references
                          if entry["arm"] == "label"}, {50.0})


# --- checks ----------------------------------------------------------------------------------------


def _check_rows():
    """Synthetic rows of one cell and seed that pass the window check."""
    return synthetic_rows(seeds=(0,))


class RunnerCheckTests(unittest.TestCase):
    def test_reproduction(self):
        rows = synthetic_rows(seeds=(0, 1))
        for index, row in enumerate(rows):
            row.update(avoided_prefill_tokens=1000 + index, counters_sha256=f"c{index}",
                       decision_sha256=f"d{index}")
        published = {("all16", row["arm"]) + runner._cell_seed(row): {
            "source": tailwindow.reference_source(row["arm"]),
            "avoided_prefill_tokens": row["avoided_prefill_tokens"],
            "counters_sha256": row["counters_sha256"], "decision_sha256": row["decision_sha256"]}
            for row in rows}
        report, table = runner.check_reproduction(rows, published, 14)
        self.assertTrue(runner.reproduction_passes(report))
        self.assertEqual((report["matched"], report["decision_digests_compared"]), (14, 14))
        self.assertTrue(all(entry["reproduces"] for entry in table))
        self.assertEqual({entry["reference_source"] for entry in table},
                         {"error_location_001", "horizon_control_001",
                          "matched_class_order_001"})
        for column, value in (("avoided_prefill_tokens", 1), ("counters_sha256", "x"),
                              ("decision_sha256", "y")):
            with self.subTest(column=column):
                changed = {key: dict(entry) for key, entry in published.items()}
                key = ("all16", rows[3]["arm"]) + runner._cell_seed(rows[3])
                changed[key][column] = value
                report, table = runner.check_reproduction(rows, changed, 14)
                self.assertFalse(runner.reproduction_passes(report))
                self.assertEqual(report["mismatched"], 1)
                self.assertIn(column, report["mismatches"][0])
        missing = dict(published)
        del missing[("all16", rows[0]["arm"]) + runner._cell_seed(rows[0])]
        report, _ = runner.check_reproduction(rows, missing, 14)
        self.assertEqual((report["missing"], report["matched"]), (1, 13))
        self.assertFalse(runner.reproduction_passes(report))
        report, _ = runner.check_reproduction(rows, published, 15)
        self.assertFalse(runner.reproduction_passes(report))

    def test_the_window_check(self):
        rows = _check_rows()
        for row in rows:
            for counter in ("measured_requests", "requested_tokens", "l1_avoided_tokens",
                            "avoided_prefill_tokens", "l2_present_unusable_tokens",
                            "l2_present_unusable_blocks"):
                row[counter] = row[f"full_{counter}"]
        self.assertEqual(runner.check_windows(rows), (1, []))

        def problems(change):
            changed = [dict(row) for row in rows]
            change(changed)
            return runner.check_windows(changed)[1]

        def above_full(changed):
            changed[2]["head_l2_avoided_tokens"] = changed[2]["full_l2_avoided_tokens"] + 1
            changed[2]["tail_l2_avoided_tokens"] = -1

        found = problems(above_full)
        self.assertTrue(any("head l2_avoided_tokens" in line and "> full" in line
                            for line in found))

        def head_requests_vary(changed):
            changed[4]["head_measured_requests"] -= 1
            changed[4]["tail_measured_requests"] += 1

        self.assertTrue(any("head measured_requests takes" in line
                            for line in problems(head_requests_vary)))

        def bad_tail(changed):
            changed[1]["tail_requested_tokens"] += 1

        self.assertTrue(any("tail requested_tokens" in line for line in problems(bad_tail)))

        def bad_full(changed):
            changed[5]["requested_tokens"] += 1

        self.assertTrue(any("full requested_tokens" in line for line in problems(bad_full)))
        self.assertTrue(any("6 arms, expected 7" in line
                            for line in problems(lambda changed: changed.pop())))

    def test_the_statistics_check_covers_every_arm_without_an_override(self):
        def row(arm, overridden):
            return {"trace": "t", "cell": "c", "arm": arm, "seed": 0, "variant": "main",
                    "stat_decisions_seen": 5, "l2_decisions": 5, "stat_rejections_seen": 1,
                    "l2_rejections": 1, "stat_evictions_seen": 4, "l2_evictions": 4,
                    "stat_decisions": 3, "overridden_decisions_seen": overridden}

        self.assertEqual(runner.check_statistics([row("evict_label", 3),
                                                  row("evict_binary_150_learned", 2)]), [])
        for arm in ("lru", "label_binary_150", "learned", "label_binary_60"):
            with self.subTest(arm=arm):
                found = runner.check_statistics([row(arm, 1)])
                self.assertTrue(any("must keep the store's choice" in line for line in found))

    def test_the_class_check(self):
        good = {"trace": "t", "cell": "l1=0.0025,l2x1", "arm": "evict_binary_60_learned",
                "seed": 0, "l1_fraction": 0.0025, "l2_multiplier": 1.0, "arm_parameter": 60.0,
                "class_horizon_seconds": 60.0, "class_decisions_seen": 5, "l2_decisions": 5,
                "class_rejections_seen": 1, "l2_rejections": 1,
                "class_resident_evictions_seen": 4, "l2_evictions": 4,
                "class_overridden_seen": 2, "overridden_decisions_seen": 2,
                "class_decisions": 3, "stat_decisions": 3, "class_overridden": 1,
                "overridden_decisions": 1, "class_violations_seen": 0,
                "class_overridden_outside_admission_seen": 0, "overridden_decisions_resident": 0,
                "class_m4_count_resident": 0, "m4_count_resident": 0}
        plain = {"trace": "t", "cell": "c", "arm": "lru", "seed": 0}
        self.assertEqual(runner.check_class([good, plain]), (1, []))
        self.assertIn("class statistics on an arm",
                      runner.check_class([dict(plain, class_horizon_seconds=60.0)])[1][0])
        self.assertIn("no class statistics",
                      runner.check_class([dict(good, class_horizon_seconds="")])[1][0])
        self.assertTrue(runner.check_class([dict(good, class_violations_seen=1)])[1])

    def test_the_published_references_load_as_the_plan_names_them(self):
        published, problems = runner.load_references()
        self.assertEqual(problems, [])
        self.assertEqual(len(published), 420)
        by_source = defaultdict(int)
        for entry in published.values():
            by_source[entry["source"]] += 1
            self.assertIn("counters_sha256", entry)
            self.assertIn("decision_sha256", entry)
        self.assertEqual(dict(by_source), {"error_location_001": 240, "horizon_control_001": 40,
                                           "horizon_fill_001": 20, "matched_class_order_001": 120})
        readings, problems = runner.load_published_readings()
        self.assertEqual((len(readings), problems), (12, []))
        self.assertEqual(readings[("conversation_trace", 0.0025, 1.0)]["class_order_reading"],
                         MATCHED_READINGS[1])

    def test_reference_loading_keeps_each_reference_to_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def copy(source, change):
                rows = list(csv_rows(runner.REFERENCE_SOURCES[source]))
                change(rows)
                path = root / f"{source}_{len(os.listdir(root))}.csv"
                runner._write(path, rows)
                return path

            # The fill-in's reruns of label_binary_60/300/600 are not read; a
            # label_binary_60 row only there would be missing.
            def drop_60(rows):
                rows[:] = [row for row in rows if not (row["arm"] == "label_binary_60"
                                                       and row["seed"] == "2")]

            paths = {"horizon_control_001": copy("horizon_control_001", drop_60)}
            _, problems = runner.load_references(paths)
            self.assertEqual(len(problems), 2)
            self.assertTrue(all("label_binary_60" in line and "missing" in line
                                for line in problems))

            def duplicate(rows):
                rows.append(next(row for row in rows if row["arm"] == "label_binary_150"))

            _, problems = runner.load_references({"horizon_fill_001": copy("horizon_fill_001",
                                                                           duplicate)})
            self.assertEqual(len(problems), 1)
            self.assertIn("duplicate", problems[0])

            def no_digest(rows):
                for row in rows:
                    if row["arm"] == "evict_binary_150_recency" and row["seed"] == "0":
                        row["counters_sha256"] = ""

            _, problems = runner.load_references({"matched_class_order_001": copy(
                "matched_class_order_001", no_digest)})
            self.assertEqual(len(problems), 4)
            self.assertTrue(all("has no counters_sha256" in line for line in problems))

    def test_published_reading_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = list(csv_rows(runner.PUBLISHED_READING_SOURCES["fill"]))
            rows = [row for row in rows if not (row["h"] == "150.0"
                                                and row["trace"] == "toolagent_trace"
                                                and row["l1_fraction"] == "0.01")]
            path = Path(directory) / "fill.csv"
            runner._write(path, rows)
            _, problems = runner.load_published_readings({"fill": path})
            self.assertEqual(problems, ["published S of ('toolagent_trace', 0.01, 1.0) missing"])
            rows = list(csv_rows(runner.PUBLISHED_READING_SOURCES["order"]))
            rows[0]["h_star"] = "90.0"
            path = Path(directory) / "order.csv"
            runner._write(path, rows)
            _, problems = runner.load_published_readings({"order": path})
            self.assertEqual(len(problems), 1)
            self.assertIn("h* 90.0", problems[0])

    def test_argument_refusals_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["a.jsonl", "b.jsonl", "--output-dir", str(root / "out")]
            args = runner.parse_args(base)
            self.assertEqual((args.workers, args.paper_dir),
                             (10, REPOSITORY / "results/paper/tail_window_check_001"))
            for argv, message in (
                    (base + ["--workers", "13"], "hard cap"),
                    (base + ["--workers", "0"], "positive"),
                    (["a.jsonl", "--output-dir", str(root / "out")], "expected 2 trace files"),
                    (base + ["--paper-dir", str(root)], "exists"),
                    (base + ["--paper-dir", str(root / "out")], "must differ")):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as caught:
                        runner.validate_arguments(runner.parse_args(argv))
                    self.assertIn(message, str(caught.exception))
            (root / "out").mkdir()
            with self.assertRaises(SystemExit) as caught:
                runner.validate_arguments(runner.parse_args(base + ["--paper-dir",
                                                                    str(root / "p")]))
            self.assertIn("exists", str(caught.exception))

    def test_an_unclean_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["run_tail_window_check.py", "a.jsonl", "b.jsonl", "--output-dir",
                    str(root / "out"), "--paper-dir", str(root / "paper")]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runner.rmc, "_git", lambda *a: " M src/x.py"):
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
            self.assertIn("not clean", str(caught.exception))
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runner.rmc, "_git", lambda *a: ""), \
                    mock.patch.object(runner, "sources_differing_from_head",
                                      lambda: ["src/persistent_kv_admission/tailwindow.py"]):
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
            self.assertIn("differs from HEAD", str(caught.exception))
            self.assertEqual(os.listdir(root), [])

    def test_the_execution_sources_cover_the_imported_runners(self):
        sources = {str(path.relative_to(REPOSITORY)) for path in runner.execution_sources()}
        for path in ("scripts/run_tail_window_check.py", "scripts/run_matched_class_order.py",
                     "scripts/run_horizon_control.py", "scripts/run_error_location.py",
                     "scripts/run_mechanism_control.py", "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/tailwindow.py",
                     "src/persistent_kv_admission/horizonfill.py",
                     "src/persistent_kv_admission/onpolicy.py"):
            self.assertIn(path, sources)

    def test_every_written_table_is_described_in_the_readme(self):
        for name in ("replay_seeds", "replay", "references_seeds", "windows", "horizon",
                     "class_order", "order", "admission", "tail_share", "reproduction",
                     "readings"):
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        self.assertIn("`run_config.json`", runner.README_TEXT)


# --- the whole run on constructed traces -----------------------------------------------------------


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


def _fake_models(names):
    return ({name: _LinearRanker() for name in names},
            {name: {"path": f"models/pi0/{name}__next_use.json", "sha256": "a" * 64}
             for name in names})


def _published_readings(index, matched_rows):
    """The published reading tables of the constructed publication: S at h*
    by the parents' formula on the reference rows' five-seed means, and the
    matched class-order run's own `matched_tables` for R, order and
    admission."""
    def u_mean(arm, cell):
        return float(np.mean([runner.hc._u(index[(arm,) + cell + (seed,)]) for seed in SEEDS]))

    horizon_rows, fill_rows = [], []
    for name in runner.TRACES:
        for fraction, multiplier in ALL_CELLS:
            cell = (name, fraction, multiplier)
            h = MATCHED_HORIZON_SECONDS[(fraction, multiplier)]
            rung = horizon_arm(h)
            entry = {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                     "cell": runner.rdp.cell_label(fraction, multiplier), "h": h, "arm": rung,
                     "S": horizonctl.shortfall(u_mean("label", cell), u_mean(rung, cell),
                                               u_mean("lru", cell))}
            (fill_rows if h == 150.0 else horizon_rows).append(entry)
            # Rows of other horizons, which the loader must skip.
            horizon_rows.append(dict(entry, h=6.0, arm="label_binary_6", S=99.0))
    published = {}
    for (arm, *cell_seed), row in index.items():
        published[("all16", arm) + tuple(cell_seed)] = runner.mco._reference_entry(
            row, "all16", arm, tailwindow.reference_source(arm))
    for row in matched_rows:
        published_arm = PUBLISHED_ARMS[matched_arm(600, row["order"])]
        published[("all16", published_arm) + runner._cell_seed(row)] = \
            runner.mco._reference_entry(row, "all16", published_arm, "horizon_control_001")
    tables = runner.mco.matched_tables(matched_rows, published)
    return {"horizon": horizon_rows, "fill": fill_rows, "class_order": tables["class_order"],
            "order": tables["order"], "admission": tables["admission"]}


class MainTests(unittest.TestCase):
    """`main` end to end on two constructed traces named as the grid's, with a
    constructed publication of every reference produced by the worker of the
    run that published it (the error-location, horizon-control, fill-in and
    matched class-order runners), git answered as for a clean tree and the
    ranker loader replaced by the fixed linear ranker."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.root = root
        traces = {}
        for name, seed in (("conversation_trace", 9), ("toolagent_trace", 5)):
            path = root / f"{name}.jsonl"
            records = horizon_tests.scaled(horizon_tests.error_tests.partial_records(seed=seed))
            path.write_text("".join(json.dumps(record) + "\n" for record in records),
                            encoding="utf-8")
            traces[name] = load_mooncake_trace(path, 512)
        cls.trace_paths = [str(root / f"{name}.jsonl") for name in traces]
        shared, working_set = _shared(traces)
        sources = {source: [] for source in runner.REFERENCE_SOURCES}
        index, matched_rows = {}, []
        with contextlib.ExitStack() as stack:
            for patch in _parent_patches(shared, working_set):
                stack.enter_context(patch)
            for name in sorted(traces):
                for fraction, multiplier in ALL_CELLS:
                    for arm in cell_arms(fraction, multiplier):
                        for seed in SEEDS:
                            row = parent_worker(name, fraction, multiplier, arm, seed)
                            source = tailwindow.reference_source(arm)
                            sources[source].append(row)
                            index[(arm, name, fraction, multiplier, seed)] = row
                            if source == "matched_class_order_001":
                                matched_rows.append(row)
                    # Rows the loader must skip: the horizon control's rung of
                    # another horizon, and the fill-in's rerun of label_binary_60.
                    sources["horizon_control_001"].append(
                        runner.hc._replay_worker((name, fraction, multiplier, "label_binary_6", 0,
                                                  "main")))
                    if MATCHED_HORIZON_SECONDS[(fraction, multiplier)] == 150.0:
                        sources["horizon_fill_001"].append(fill_runner._replay_worker(
                            (name, fraction, multiplier, "label_binary_60", 0)))
        cls.index = index
        cls.sources = {}
        for source, rows in sources.items():
            path = root / "published" / source / "replay_seeds.csv"
            path.parent.mkdir(parents=True)
            runner._write(path, rows)
            cls.sources[source] = path
        cls.readings = {}
        for name, rows in _published_readings(index, matched_rows).items():
            path = root / "published" / "readings" / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            runner._write(path, rows)
            cls.readings[name] = path
        cls.manifests = (root / "manifest_a.csv", root / "manifest_b.csv")
        for path in cls.manifests:
            path.write_text("policy,target,trace,model_sha256\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _tampered(self, name, kind, table, change) -> dict:
        """The fixture sources (`kind` "sources" or "readings") with `table`
        changed by `change` (a function editing its rows in place)."""
        originals = self.sources if kind == "sources" else self.readings
        rows = list(csv_rows(originals[table]))
        change(rows)
        path = self.root / name / f"{table}.csv"
        path.parent.mkdir(parents=True)
        runner._write(path, rows)
        return dict(originals, **{table: path})

    def _main(self, argv, sources=None, readings=None, cells=None, seeds=None) -> str:
        patches = [mock.patch.object(sys, "argv", ["run_tail_window_check.py"]
                                     + self.trace_paths + argv),
                   mock.patch.object(runner.rmc, "_git", _fake_git),
                   mock.patch.object(runner, "sources_differing_from_head", lambda: []),
                   mock.patch.dict(runner.REFERENCE_SOURCES, sources or self.sources),
                   mock.patch.dict(runner.PUBLISHED_READING_SOURCES, readings or self.readings),
                   mock.patch.object(runner.rmc, "load_models", _fake_models),
                   mock.patch.object(runner.rmc, "MODEL_MANIFESTS", self.manifests),
                   mock.patch.dict(runner.SHARED), mock.patch.dict(runner.rdp._SHARED)]
        if cells is not None:
            patches.append(mock.patch.object(runner, "CELLS", cells))
        if seeds is not None:
            patches.append(mock.patch.object(runner, "SEEDS", seeds))
        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            try:
                runner.main()
            finally:
                self.stdout = stdout.getvalue()
        return self.stdout

    def test_a_run_publishes_every_table_after_every_check(self):
        output, paper = self.root / "run_ok", self.root / "paper_ok"
        text = self._main(["--output-dir", str(output), "--paper-dir", str(paper),
                           "--workers", "2"])
        tables = {"replay_seeds.csv", "replay.csv", "references_seeds.csv", "windows.csv",
                  "horizon.csv", "class_order.csv", "order.csv", "admission.csv",
                  "tail_share.csv", "reproduction.csv", "readings.csv", "README.md",
                  "run_config.json"}
        self.assertEqual(set(os.listdir(paper)), tables)
        self.assertEqual(set(os.listdir(output)), tables | {"raw_replays.jsonl"})
        self.assertEqual(len((output / "raw_replays.jsonl").read_text().splitlines()), 420)
        self.assertIn("[420/420]", text)
        self.assertIn("420 matched, 0 missing, 0 mismatched", text)
        self.assertIn("120 values compared, equal", text)
        replays = list(csv_rows(paper / "replay_seeds.csv"))
        self.assertEqual(len(replays), 420)
        self.assertEqual(list(replays[0])[:12],
                         ["trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width",
                          "mechanism", "arm", "family", "arm_parameter", "seed", "variant"])
        # Every replay is its parent's published row, digest for digest, and
        # its windows add up.
        for row in replays:
            parent = self.index[(row["arm"],) + runner._cell_seed(row)]
            self.assertEqual((row["counters_sha256"], row["decision_sha256"],
                              int(row["avoided_prefill_tokens"]), row["family"]),
                             (parent["counters_sha256"], parent["decision_sha256"],
                              parent["avoided_prefill_tokens"], parent["family"]))
            self.assertEqual(row["reproduces_reference"], "True")
            for counter in COUNTERS:
                self.assertEqual(int(row[f"head_{counter}"]) + int(row[f"tail_{counter}"]),
                                 int(row[f"full_{counter}"]))
        reproduction = list(csv_rows(paper / "reproduction.csv"))
        self.assertEqual(len(reproduction), 420)
        self.assertTrue(all(row["reproduces"] == "True" and row["same_decision_sha256"] == "True"
                            for row in reproduction))
        counts = {name: len(list(csv_rows(paper / f"{name}.csv")))
                  for name in ("replay", "references_seeds", "windows", "horizon", "class_order",
                               "order", "admission", "tail_share", "readings")}
        self.assertEqual(counts, {"replay": 84, "references_seeds": 420, "windows": 84,
                                  "horizon": 12, "class_order": 12, "order": 12, "admission": 12,
                                  "tail_share": 96, "readings": 19})
        self.assertTrue(all(row["full_equals_published"] == "True"
                            for name in ("horizon", "class_order", "order", "admission")
                            for row in csv_rows(paper / f"{name}.csv")))
        readings = {(row["reading"], row["window"]): row
                    for row in csv_rows(paper / "readings.csv")}
        for reading in ("1_horizon_S_at_most_0.10", "2_class_order_R_learned_order_at_least_0.9",
                        "3_learned_minus_recency_sign_of_mean_agrees_with_full",
                        "4_label_binary_minus_recency_sign_of_mean_agrees_with_full"):
            self.assertEqual(readings[(reading, "head")]["kind"], "prediction")
            self.assertIn(readings[(reading, "head")]["prediction_holds"], ("True", "False"))
            self.assertEqual(readings[(reading, "head")]["cells"], "12")
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["phase"], "tail_window_check")
        self.assertEqual(config["plan"], "docs/tail-window-check-plan.md")
        self.assertEqual((config["plan_commit"], config["code_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(config["references"],
                         {str(path.resolve()): sha256_path(path)
                          for path in self.sources.values()})
        self.assertEqual(config["published_readings"],
                         {str(path.resolve()): sha256_path(path)
                          for path in self.readings.values()})
        for name, trace_path in zip(("conversation_trace", "toolagent_trace"), self.trace_paths):
            bounds = config["windows"]["bounds_ms"][name]
            trace = load_mooncake_trace(trace_path, 512)
            self.assertEqual((bounds["split_ms"], bounds["head_end_ms"], bounds["end_ms"]),
                             (horizon_for(trace)[1], trace.end_ms - 600_000.0, trace.end_ms))
            self.assertEqual(config["trace_files"][name]["sha256"], sha256_path(trace_path))
        checks = config["checks"]
        self.assertEqual((checks["reproduction"]["matched"], checks["reproduction"]["expected"],
                          checks["reproduction"]["decision_digests_compared"]), (420, 420, 420))
        self.assertEqual((checks["identifier_pairs_compared"], checks["window_groups"],
                          checks["class_replays_checked"], checks["published_readings_compared"]),
                         (420 * 7, 60, 120, 120))
        self.assertEqual((checks["identifier_problems"], checks["window_problems"],
                          checks["statistics_problems"], checks["class_problems"],
                          checks["class_violations"], checks["invariant_violations"],
                          checks["unexplained_absent_tokens"],
                          checks["published_reading_problems"]), (0,) * 8)
        self.assertEqual((config["replays"], config["workers"]), (420, 2))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(tabulate.main([str(paper)]), 0)
        printed = stdout.getvalue()
        for section in ("W0", "W1", "W2", "W3", "W4", "W5"):
            self.assertIn(f"### {section} ", printed)
        self.assertEqual(sorted(os.listdir(paper)), sorted(tables))
        parsed = {name: tabulate.read_csv(paper / f"{name}.csv") for name in tabulate.INPUTS}
        self.assertEqual(sum(entry["reproduces"]
                             for entry in tabulate.reproduction_counts(parsed["reproduction"])),
                         420)
        self.assertEqual(len(tabulate.horizon_table(parsed["horizon"])), 12)
        self.assertTrue(all(entry["equal"] for entry in
                            tabulate.difference_table(parsed["order"], "learned_minus_recency")))
        sizes = tabulate.window_sizes(parsed["windows"])
        self.assertEqual(len(sizes), 12)
        self.assertTrue(all(entry["head_tokens"] + entry["tail_tokens"] == entry["full_tokens"]
                            for entry in sizes))

    def test_a_reproduction_difference_publishes_nothing(self):
        def counters(rows):
            target = next(row for row in rows if row["arm"] == "lru" and row["seed"] == "0"
                          and float(row["l1_fraction"]) == 0.01
                          and float(row["l2_multiplier"]) == 1.0)
            target["counters_sha256"] = "0" * 64

        def tokens(rows):
            target = next(row for row in rows if row["arm"] == "label_binary_150"
                          and row["seed"] == "0" and row["trace"] == "toolagent_trace"
                          and float(row["l1_fraction"]) == 0.01)
            target["avoided_prefill_tokens"] = str(int(target["avoided_prefill_tokens"]) + 1)

        def decisions(rows):
            target = next(row for row in rows if row["arm"] == "evict_binary_150_recency"
                          and row["seed"] == "0" and float(row["l1_fraction"]) == 0.01)
            target["decision_sha256"] = "1" * 64

        for name, source, change, column in (
                ("counters", "error_location_001", counters, "counters_sha256"),
                ("tokens", "horizon_fill_001", tokens, "avoided_prefill_tokens"),
                ("decisions", "matched_class_order_001", decisions, "decision_sha256")):
            with self.subTest(case=name):
                sources = self._tampered(f"tampered_{name}", "sources", source, change)
                output, paper = self.root / f"run_bad_{name}", self.root / f"paper_bad_{name}"
                with self.assertRaises(SystemExit) as caught:
                    self._main(["--output-dir", str(output), "--paper-dir", str(paper),
                                "--workers", "2"], sources=sources, cells=((0.01, 1.0),),
                               seeds=(0,))
                self.assertIn("do not reproduce", str(caught.exception))
                self.assertIn("nothing derived or published", str(caught.exception))
                self.assertIn("MISMATCH", self.stdout)
                self.assertIn(column, self.stdout)
                self.assertFalse(paper.exists())
                self.assertEqual(os.listdir(output), ["raw_replays.jsonl"])

    def test_a_published_reading_difference_publishes_nothing(self):
        # S at 2% x 1 (h* = 300 s) is finite on the constructed traces; at
        # 0.25% x 1 it is nan there (U(label) = U(lru)), and nan equals nan.
        def shift(rows):
            target = next(row for row in rows if row["trace"] == "conversation_trace"
                          and row["h"] == "300.0")
            self.assertTrue(math.isfinite(float(target["S"])))
            target["S"] = repr(math.nextafter(float(target["S"]), math.inf))

        readings = self._tampered("tampered_reading", "readings", "horizon", shift)
        output, paper = self.root / "run_bad_reading", self.root / "paper_bad_reading"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper), "--workers", "2"],
                       readings=readings, cells=((0.02, 1.0),))
        self.assertIn("differ from the published values", str(caught.exception))
        self.assertIn("70 matched", self.stdout)
        self.assertIn("PUBLISHED", self.stdout)
        self.assertFalse(paper.exists())
        self.assertEqual(os.listdir(output), ["raw_replays.jsonl"])

    def test_a_cell_subset_runs_and_publishes(self):
        output, paper = self.root / "run_subset", self.root / "paper_subset"
        self._main(["--output-dir", str(output), "--paper-dir", str(paper), "--workers", "2"],
                   cells=((0.0025, 4.0), (0.02, 4.0)))
        config = json.loads((paper / "run_config.json").read_text())
        self.assertEqual(config["replays"], 2 * 2 * 7 * 5)
        self.assertEqual(config["checks"]["reproduction"]["matched"], 140)
        self.assertEqual(config["checks"]["published_readings_compared"], 4 * 10)
        self.assertEqual(len(list(csv_rows(paper / "tail_share.csv"))), 4 * 8)

    def test_missing_inputs_are_refused_before_any_replay(self):
        def drop(rows):
            rows[:] = [row for row in rows if not (row["arm"] == "evict_label"
                                                   and row["seed"] == "4")]

        sources = self._tampered("missing", "sources", "error_location_001", drop)
        output, paper = self.root / "run_refused", self.root / "paper_refused"
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper)], sources=sources)
        self.assertIn("nothing run", str(caught.exception))
        self.assertIn("REFERENCE", self.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(paper.exists())

        def drop_reading(rows):
            rows[:] = rows[1:]

        readings = self._tampered("missing_reading", "readings", "admission", drop_reading)
        with self.assertRaises(SystemExit) as caught:
            self._main(["--output-dir", str(output), "--paper-dir", str(paper)],
                       readings=readings)
        self.assertIn("published readings are not what the plan names", str(caught.exception))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
