"""Tests for the Bailian external-workload check's fixed surface and readings
(`persistent_kv_admission.bailiancheck`, docs/bailian-external-check-plan.md).

No real trace is replayed: everything runs on constructed traces. What has to
hold before a real replay means anything: the grid, the h* table, the cells,
the arms and the counts are the plan's; every arm is built by the module that
built its published rows (the same policy, scorer type, horizon and target, no
override) and replays decision for decision as that module's build; the
random arm with its draw replaced by a constant is `label_binary_h*` decision
by decision, and draws once per candidate from its own stream; the three
window collectors partition the primary window (a request exactly at `mid` is
in W1 only) and stay within the full window; the 16-token path goes through
the unmodified converter and loader; and every reading function does what the
plan fixes at its boundaries (S never clipped and not computed below 1.0 point
of headroom, the evaluable rule, h_best ties, the classification order, the
direction of failure with its 0/0 case, monotonicity with the merged 1% level
and "not assessable", D_rand readings, the calibration half, the label's share
of T, the 1,200-second flag, absolute thresholds, and reading 7's separation,
reversal and both-mechanisms rule).
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

from persistent_kv_admission import bailiancheck as bc
from persistent_kv_admission import classmix, errorloc, horizonctl, horizonfill, matchedorder
from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import ALL16, LEAF16, MECHANISMS, horizon_arm
from persistent_kv_admission.mechanism import ExactLabelScorer
from persistent_kv_admission.tailwindow import ChainedRequestHook
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


error_tests = _load("_bailian_error_location_fixtures", REPOSITORY / "tests/test_error_location.py")
convert = _load("_bailian_convert_under_test", REPOSITORY / "scripts/convert_bailian_trace.py")

# The constructed traces run one step per 20 s (7,200 s, as the Bailian traces),
# so the split (60%), the label horizon (600 s) and the 1,200-second exclusion
# leave a primary window of about 1,700 s.
SCALE = 20
L1_BYTES = 6 * 2**20
L2_BYTES = 12 * 2**20


# --- fixtures -------------------------------------------------------------------------------------


def mooncake_records(seed=9, steps=360):
    """The error-location fixture's sessions of full blocks ending in partial
    blocks, at one step per 20 s."""
    return [dict(record, timestamp=record["timestamp"] * SCALE)
            for record in error_tests.partial_records(steps=steps, seed=seed)]


def write_trace(directory: Path, name: str, records) -> Path:
    path = Path(directory) / f"{name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def raw_bailian_records(seed=1, steps=240, span_seconds=7176.0):
    """Upstream-shaped 16-token records: two shared system prompts of 96 ids
    (1,536 tokens), sessions that grow by 8-40 ids, a partial final 16-token
    block, timestamps in seconds with 3 decimals."""
    rng = random.Random(seed)
    prompts = [list(range(base, base + 96)) for base in (1_000_000, 2_000_000)]
    sessions, out = [], []
    next_id = 10_000_000
    for step in range(steps):
        if sessions and rng.random() < 0.6:
            chain = sessions[rng.randrange(len(sessions))]
        else:
            chain = list(prompts[rng.randrange(2)])
            sessions.append(chain)
        grow = rng.randint(8, 40)
        chain.extend(range(next_id, next_id + grow))
        next_id += grow
        if len(chain) > 400:
            sessions.remove(chain)
        length = 16 * len(chain) - rng.randint(0, 15)
        out.append({"chat_id": step, "parent_chat_id": -1,
                    "timestamp": round(step * span_seconds / steps, 3),
                    "input_length": length, "output_length": 5, "type": "text", "turn": 1,
                    "hash_ids": list(chain)})
    return out


def convert_raw(directory: Path, name: str, records, block_tokens: int) -> Path:
    """The unmodified converter on a constructed upstream file named as the
    trace's upstream source; returns the converted file (with its manifest)."""
    directory = Path(directory)
    source = directory / "raw" / bc.UPSTREAM_SOURCES[name]
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("".join(json.dumps(record) + "\n" for record in records),
                          encoding="utf-8")
    output = directory / f"qwen_bailian_{block_tokens}" / f"{name}.jsonl"
    convert.convert_file(source, output, block_tokens)
    return output


class _Tracer:
    """Override-slot wrapper recording every decision's candidates and final victim."""

    def __init__(self, inner):
        self.inner = inner
        self.decisions = []

    def attach(self, l1, l2, counters):
        if hasattr(self.inner, "attach"):
            self.inner.attach(l1, l2, counters)

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index, arriving_index):
        choice = self.inner(candidates, keys, victim_index, timestamp_ms, group_index,
                            arriving_index)
        final = victim_index if choice is None else choice
        self.decisions.append((timestamp_ms, arriving_index, tuple(candidates), final))
        return choice


class _Constant:
    """A stand-in for the random arm's generator whose every draw is one constant."""

    def __init__(self, value=0.5):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def replay(trace, setup, mechanism=ALL16, seed=1, split_ms=None, windows=False):
    """One sampled replay as the runner nests its hooks, with a tracer; returns
    the result, digests, decisions, candidate counter and window counters."""
    split_ms = horizon_for(trace)[1] if split_ms is None else split_ms
    eligibility, width = MECHANISMS[mechanism]
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    statistics = DecisionStatistics(trace, 600.0, split_ms)
    candidates = bc.CandidateCount()
    tracer = _Tracer(RecordingOverride(statistics, RecordingOverride(candidates, setup.override)))
    hooks = [collector.on_request]
    collectors = None
    if windows:
        collectors = bc.window_collectors(trace, split_ms)
        hooks += [collectors[window].on_request for window in bc.COLLECTED_WINDOWS]
    result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, measure_from_ms=split_ms,
                          l2_eviction="sampled", l2_sample_width=width, l2_seed=seed,
                          l2_eligibility=eligibility, l2_arm=setup.arm,
                          l2_request_hook=ChainedRequestHook(*hooks), l2_removal_hook=collector,
                          l2_override_hook=tracer, **setup.replay_arguments())
    collector.check_against(result)
    return {"result": result, "decisions": tracer.decisions, "candidates": candidates,
            "decision_sha256": statistics.row()["decision_sha256"],
            "overridden": statistics.row()["overridden_decisions_seen"],
            "counters": (result.avoided_prefill_tokens, result.l2_admissions,
                         result.l2_rejections, result.l2_evictions, result.l2_decisions),
            "windows": bc.window_counters(result, collectors) if windows else None}


class _TraceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.trace = load_mooncake_trace(write_trace(cls.directory.name, "conversation_trace",
                                                    mooncake_records()), 512)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()


# --- the fixed surface ----------------------------------------------------------------------------


class SurfaceTests(unittest.TestCase):
    def test_the_grid_the_table_and_the_cells_are_the_plans(self):
        self.assertEqual(bc.GRID_SECONDS, (6.0, 15.0, 60.0, 150.0, 300.0, 600.0, 1200.0))
        self.assertEqual(bc.HSTAR_SECONDS, {(0.0025, 1.0): 60.0, (0.0025, 4.0): 150.0,
                                            (0.01, 1.0): 150.0, (0.01, 4.0): 600.0,
                                            (0.02, 1.0): 300.0, (0.02, 4.0): 600.0})
        self.assertEqual(bc.HSTAR_SECONDS, matchedorder.MATCHED_HORIZON_SECONDS)
        self.assertEqual(bc.CELLS, ((0.0025, 1.0), (0.0025, 4.0), (0.01, 1.0), (0.01, 4.0),
                                    (0.02, 1.0), (0.02, 4.0)))
        self.assertEqual(bc.MECHANISM_NAMES, ("all16", "leaf16"))
        self.assertEqual((bc.SEEDS, bc.REDUCED_SEEDS), ((0, 1, 2, 3, 4), (0, 1, 2)))
        self.assertEqual(bc.TRACES, ("bailian_toc_trace", "bailian_tob_trace",
                                     "bailian_thinking_trace", "bailian_coder_trace"))
        self.assertEqual((bc.TAIL_EXCLUSION_MS, bc.LABEL_HORIZON_SECONDS), (1_200_000.0, 600.0))
        self.assertEqual((bc.SUFFICES_SHORTFALL, bc.EVALUABLE_POINTS, bc.MIN_DENOMINATOR_POINTS),
                         (0.10, 1.0, 1.0))
        self.assertEqual(bc.L2_LEVELS, (0.0025, 0.01, 0.02, 0.04, 0.08))

    def test_the_arms_of_a_cell(self):
        for cell, h in bc.HSTAR_SECONDS.items():
            arms = bc.cell_arms(*cell)
            self.assertEqual(len(arms), 10)
            self.assertEqual(arms[:2], ("lru", "label"))
            self.assertEqual(arms[2:9], tuple(horizon_arm(g) for g in bc.GRID_SECONDS))
            self.assertEqual(arms[9], f"label_binary_random_{h:g}")
            self.assertEqual(bc.a_arms(*cell), ("lru", "label", horizon_arm(h),
                                                f"label_binary_random_{h:g}"))
            self.assertEqual(len(bc.b_arms(*cell)), 6)
            self.assertNotIn(horizon_arm(h), bc.b_arms(*cell))
            self.assertEqual({bc.part(arm, *cell) for arm in bc.a_arms(*cell)}, {"A"})
            self.assertEqual({bc.part(arm, *cell) for arm in bc.b_arms(*cell)}, {"B"})
            self.assertEqual(bc.granularity_arms(*cell), ("lru", "label", horizon_arm(h)))
        self.assertEqual(horizon_arm(1200.0), "label_binary_1200")
        self.assertEqual(bc.random_arm(150.0), "label_binary_random_150")
        with self.assertRaises(ValueError):
            bc.part("label_binary_random_60", 0.01, 4.0)
        with self.assertRaises(ValueError):
            bc.hstar(0.05, 1.0)

    def test_the_counts(self):
        self.assertEqual((bc.COUNT_A, bc.COUNT_B, bc.COUNT_SAMPLED, bc.COUNT_HEAP),
                         (960, 1440, 2400, 48))
        tasks = bc.main_replays()
        self.assertEqual(len(tasks), 2400)
        self.assertEqual(len(set(tasks)), 2400)
        self.assertEqual(sum(1 for task in tasks if bc.part(task[4], task[1], task[2]) == "A"),
                         960)
        self.assertEqual(len(bc.heap_replays()), 48)
        anchor = bc.anchor_replays()
        self.assertEqual(len(anchor), 70)
        self.assertEqual(sum(1 for task in anchor if task[3] == ALL16), 40)
        self.assertEqual(sum(1 for task in anchor if task[3] == LEAF16), 30)
        self.assertEqual({task[4] for task in anchor if task[3] == LEAF16 and task[1] == 0.0025},
                         {"lru", "label", "label_binary_60"})
        self.assertEqual({task[4] for task in anchor if task[3] == LEAF16 and task[1] == 0.01},
                         {"lru", "label", "label_binary_600"})
        self.assertEqual((len(bc.smoke_replays()), len(bc.smoke_fine_replays())), (16, 3))
        self.assertTrue(all(task[1:4] == (0.01, 1.0, ALL16) and task[5] == 0
                            for task in bc.smoke_replays() + bc.smoke_fine_replays()))
        self.assertEqual((len(bc.granularity_replays()), len(bc.granularity_replays(
            bc.REDUCED_SEEDS))), (90, 54))

    def test_all16_and_leaf16_tasks_differ_only_in_the_mechanism(self):
        tasks = bc.main_replays()
        all16 = {task[:3] + task[4:] for task in tasks if task[3] == ALL16}
        leaf16 = {task[:3] + task[4:] for task in tasks if task[3] == LEAF16}
        self.assertEqual(all16, leaf16)
        self.assertEqual(MECHANISMS[ALL16][1], MECHANISMS[LEAF16][1])
        self.assertNotEqual(MECHANISMS[ALL16][0], MECHANISMS[LEAF16][0])

    def test_the_predictions_are_absolute_counts(self):
        self.assertEqual({name: spec["at_least"] for name, spec in bc.PREDICTIONS.items()},
                         {"1_transplant_all16": 12, "2_direction_of_failure_all16": None,
                          "3_grid_all16": 20, "4_monotonicity_all16": 3,
                          "5_random_order_all16": 16, "6_grid_leaf16": 16,
                          "7_order_beyond_bit_separated": 10})
        # An absolute count: 12 of 18 evaluable holds; 11 of 11 evaluable does not.
        self.assertTrue(bc.prediction_holds("1_transplant_all16", 12))
        self.assertFalse(bc.prediction_holds("1_transplant_all16", 11))
        self.assertTrue(bc.prediction_holds("3_grid_all16", 20))
        self.assertFalse(bc.prediction_holds("3_grid_all16", 19))
        self.assertTrue(bc.prediction_holds("4_monotonicity_all16", 3))
        self.assertFalse(bc.prediction_holds("5_random_order_all16", 15))
        self.assertTrue(bc.prediction_holds("6_grid_leaf16", 16))
        self.assertTrue(bc.prediction_holds("7_order_beyond_bit_separated", 10))
        self.assertFalse(bc.prediction_holds("7_order_beyond_bit_separated", 9))
        with self.assertRaises(ValueError):
            bc.prediction_holds("2_direction_of_failure_all16", 3)


# --- the arms -------------------------------------------------------------------------------------


class ArmSetupTests(_TraceCase):
    def test_published_horizons_are_built_by_the_published_builders(self):
        trace = self.trace
        expected = {ALL16: {6.0: "horizonctl", 15.0: "horizonctl", 60.0: "horizonctl",
                            150.0: "horizonfill", 300.0: "horizonctl", 600.0: "horizonctl"},
                    LEAF16: {6.0: "horizonctl", 15.0: "horizonctl", 60.0: "horizonfill",
                             150.0: "horizonfill", 300.0: "horizonfill", 600.0: "horizonfill"}}
        for mechanism, builders in expected.items():
            for h, module in builders.items():
                arm = horizon_arm(h)
                with self.subTest(mechanism=mechanism, arm=arm):
                    self.assertEqual(bc.arm_builder(arm, mechanism), f"{module}.arm_setup")
                    mine = bc.arm_setup(arm, trace, 600.0, 3, mechanism)
                    theirs = (horizonfill.arm_setup(arm, trace) if module == "horizonfill"
                              else horizonctl.arm_setup(arm, trace, 600.0, 3))
                    self._same_setup(mine, theirs, h, "binary")
            for arm, target in (("label", "next_use"),):
                mine = bc.arm_setup(arm, trace, 600.0, 3, mechanism)
                theirs = errorloc.arm_setup(arm, trace, 600.0, 3)
                self._same_setup(mine, theirs, 600.0, target)
            mine, theirs = (bc.arm_setup("lru", trace, 600.0, 3, mechanism),
                            errorloc.arm_setup("lru", trace, 600.0, 3))
            self.assertEqual((mine.arm, mine.l2_policy, mine.scorer, mine.override),
                             (theirs.arm, theirs.l2_policy, None, None))

    def _same_setup(self, mine, theirs, horizon, target):
        self.assertEqual((mine.arm, mine.l2_policy, mine.override, mine.sampled_offline),
                         (theirs.arm, theirs.l2_policy, None, False))
        self.assertEqual(type(mine.scorer), type(theirs.scorer))
        self.assertIsInstance(mine.scorer, ExactLabelScorer)
        self.assertEqual((mine.scorer.horizon_seconds, mine.scorer.target),
                         (theirs.scorer.horizon_seconds, theirs.scorer.target))
        self.assertEqual((mine.scorer.horizon_seconds, mine.scorer.target), (horizon, target))
        self.assertIsNot(mine.scorer, theirs.scorer)

    def test_the_1200_second_arm_is_the_identical_construction(self):
        for mechanism in bc.MECHANISM_NAMES:
            setup = bc.arm_setup("label_binary_1200", self.trace, 600.0, 0, mechanism)
            self.assertEqual((setup.l2_policy, setup.override), ("learned", None))
            self.assertIsInstance(setup.scorer, ExactLabelScorer)
            self.assertEqual((setup.scorer.horizon_seconds, setup.scorer.target),
                             (1200.0, "binary"))
            self.assertEqual(bc.arm_builder("label_binary_1200", mechanism),
                             "ExactLabelScorer(trace, h, target='binary'), l2_policy='learned'")

    def test_the_random_arm(self):
        for h in bc.HSTAR_HORIZONS:
            arm = bc.random_arm(h)
            setup = bc.arm_setup(arm, self.trace, 600.0, 4, LEAF16)
            stream = bc.random_stream(setup)
            self.assertIsInstance(stream, classmix.RandomClassScorer)
            self.assertIs(setup.scorer, stream)
            self.assertEqual((setup.l2_policy, setup.override), ("learned", None))
            self.assertIsInstance(stream.reuse, ExactLabelScorer)
            self.assertEqual((stream.reuse.horizon_seconds, stream.reuse.target), (h, "binary"))
            self.assertEqual(stream.stream_seed, classmix.random_stream_seed(4, arm))
            self.assertEqual(stream.draws, 0)
        self.assertIsNone(bc.random_stream(bc.arm_setup("label", self.trace, 600.0, 0)))
        with self.assertRaises(ValueError):
            bc.arm_setup("evict_label", self.trace, 600.0, 0)
        with self.assertRaises(ValueError):
            bc.arm_builder("label", "all64")

    def test_the_builds_replay_as_their_parents_decision_by_decision(self):
        for mechanism in bc.MECHANISM_NAMES:
            for arm in ("lru", "label", "label_binary_150", "label_binary_600"):
                with self.subTest(mechanism=mechanism, arm=arm):
                    module = bc.arm_builder(arm, mechanism)
                    parent = (errorloc.arm_setup(arm, self.trace, 600.0, 2)
                              if module.startswith("errorloc")
                              else horizonfill.arm_setup(arm, self.trace)
                              if module.startswith("horizonfill")
                              else horizonctl.arm_setup(arm, self.trace, 600.0, 2))
                    mine = replay(self.trace, bc.arm_setup(arm, self.trace, 600.0, 2, mechanism),
                                  mechanism, seed=2)
                    theirs = replay(self.trace, parent, mechanism, seed=2)
                    self.assertEqual(mine["decisions"], theirs["decisions"])
                    self.assertEqual(mine["decision_sha256"], theirs["decision_sha256"])
                    self.assertEqual(mine["counters"], theirs["counters"])
                    self.assertGreater(len(mine["decisions"]), 0)
                    self.assertEqual(mine["overridden"], 0)


class RandomArmTests(_TraceCase):
    """Required unit test (a): with its draw replaced by a constant, the random
    arm is `label_binary_h*` decision by decision; with real draws it draws
    once per candidate from its own stream."""

    def test_with_a_constant_draw_the_random_arm_is_the_recency_arm(self):
        for mechanism in bc.MECHANISM_NAMES:
            for h in bc.HSTAR_HORIZONS:
                with self.subTest(mechanism=mechanism, h=h):
                    setup = bc.arm_setup(bc.random_arm(h), self.trace, 600.0, 3, mechanism)
                    constant = _Constant(0.25)
                    bc.random_stream(setup).rng = constant
                    mine = replay(self.trace, setup, mechanism, seed=3)
                    theirs = replay(self.trace, bc.arm_setup(horizon_arm(h), self.trace, 600.0,
                                                             3, mechanism), mechanism, seed=3)
                    self.assertGreater(len(mine["decisions"]), 10)
                    self.assertEqual(mine["decisions"], theirs["decisions"])
                    self.assertEqual(mine["decision_sha256"], theirs["decision_sha256"])
                    self.assertEqual(mine["counters"], theirs["counters"])
                    self.assertEqual(constant.calls, mine["candidates"].candidates)
                    self.assertEqual(bc.random_stream(setup).draws, constant.calls)

    def test_real_draws_are_one_per_candidate_from_the_arms_own_stream(self):
        differs = False
        for h in bc.HSTAR_HORIZONS:
            setup = bc.arm_setup(bc.random_arm(h), self.trace, 600.0, 3, ALL16)
            mine = replay(self.trace, setup, ALL16, seed=3)
            stream = bc.random_stream(setup)
            self.assertEqual(stream.draws, mine["candidates"].candidates)
            self.assertEqual(mine["candidates"].decisions, mine["result"].l2_decisions)
            self.assertTrue(bc.stream_is_separate(stream, mine["candidates"]))
            self.assertEqual(mine["overridden"], 0)
            rung = replay(self.trace, bc.arm_setup(horizon_arm(h), self.trace, 600.0, 3, ALL16),
                          ALL16, seed=3)
            differs |= mine["decisions"] != rung["decisions"]
        # The draw does change decisions somewhere (the test has teeth).
        self.assertTrue(differs)

    def test_stream_separation_is_checked_against_the_store(self):
        counter = bc.CandidateCount()
        stream = classmix.RandomClassScorer(ExactLabelScorer(self.trace, 60.0, "binary"), 0, "x")
        self.assertFalse(bc.stream_is_separate(stream, counter))     # never attached

        class _Store:
            rng = stream.rng

        counter.attach(_Store())
        self.assertFalse(bc.stream_is_separate(stream, counter))
        counter.attach(type("_Other", (), {"rng": random.Random(0)})())
        self.assertTrue(bc.stream_is_separate(stream, counter))


# --- the windows ----------------------------------------------------------------------------------


def window_records():
    """A trace whose requests fall exactly on the window boundaries: split
    3,000,000 ms, W end 3,800,000 (end 5,000,000 - 1,200,000), mid 3,400,000."""
    base = error_tests.partial_records(steps=200, seed=4)
    stamps = sorted({int(record["timestamp"]) * 14 for record in base} - {3_000_000})
    special = [3_000_000, 3_400_000, 3_400_000, 3_400_001, 3_800_000, 3_800_001, 5_000_000]
    times = sorted([value for value in stamps if value < 5_000_000] + special)
    out = []
    for index, timestamp in enumerate(times):
        record = dict(base[index % len(base)])
        record["timestamp"] = timestamp
        out.append(record)
    return out


class WindowTests(unittest.TestCase):
    """Required unit test (b): three window collectors on one replay sum
    correctly (W1 + W2 = W, W <= full), with requests exactly at mid and at
    the window ends."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.trace = load_mooncake_trace(write_trace(cls.directory.name, "window_trace",
                                                    window_records()), 512)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_bounds(self):
        bounds = bc.window_bounds(5_000_000.0, 3_000_000.0)
        self.assertEqual((bounds["split_ms"], bounds["primary_end_ms"], bounds["mid_ms"],
                          bounds["end_ms"]), (3_000_000.0, 3_800_000.0, 3_400_000.0, 5_000_000.0))
        self.assertEqual(bounds["second_start_ms"], math.nextafter(3_400_000.0, math.inf))
        self.assertEqual((bounds["primary_seconds"], bounds["first_half_seconds"],
                          bounds["second_half_seconds"], bounds["full_seconds"]),
                         (800.0, 400.0, 400.0, 2000.0))
        self.assertEqual(bc.window_of(3_000_000.0, bounds), ("full", "W", "W1"))
        self.assertEqual(bc.window_of(3_400_000.0, bounds), ("full", "W", "W1"))
        self.assertEqual(bc.window_of(3_400_001.0, bounds), ("full", "W", "W2"))
        self.assertEqual(bc.window_of(3_800_000.0, bounds), ("full", "W", "W2"))
        self.assertEqual(bc.window_of(3_800_001.0, bounds), ("full",))
        self.assertEqual(bc.window_of(5_000_000.0, bounds), ("full",))
        self.assertEqual(bc.window_of(2_999_999.0, bounds), ())
        with self.assertRaises(ValueError):
            bc.window_bounds(4_200_000.0, 3_000_000.0)       # W would be empty
        # The real traces' shape: 7,200 s with the split at 60%.
        bounds = bc.window_bounds(7_200_000.0, 4_320_000.0)
        self.assertAlmostEqual(bounds["primary_seconds"], 1680.0)

    def test_three_collectors_on_one_replay_partition_the_primary_window(self):
        trace = self.trace
        self.assertEqual(trace.end_ms, 5_000_000.0)
        split = 3_000_000.0
        recorder_calls = []
        collectors = bc.window_collectors(trace, split)
        collector = AttributionCollector(trace, measure_from_ms=split)
        hook = ChainedRequestHook(collector.on_request,
                                  *(collectors[w].on_request for w in bc.COLLECTED_WINDOWS),
                                  lambda *args: recorder_calls.append(args))
        setup = bc.arm_setup("label_binary_60", trace, 600.0, 1)
        result = run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, measure_from_ms=split,
                              l2_eviction="sampled", l2_sample_width=16, l2_seed=1,
                              l2_request_hook=hook, l2_removal_hook=collector,
                              **setup.replay_arguments())
        collector.check_against(result)
        counters = bc.window_counters(result, collectors)
        self.assertEqual(bc.window_problems(counters), [])
        for counter in bc.COUNTERS:
            self.assertEqual(counters["W1"][counter] + counters["W2"][counter],
                             counters["W"][counter])
            self.assertLessEqual(counters["W"][counter], counters["full"][counter])
        bounds = bc.window_bounds(trace.end_ms, split)
        # Request by request, from the recorded hook calls: the window counts.
        expected = {window: 0 for window in bc.WINDOWS}
        tokens = {window: 0 for window in bc.WINDOWS}
        for ids, prefix, l2_hits, present, timestamp_ms, group_index, measured in recorder_calls:
            for window in bc.window_of(timestamp_ms, bounds):
                expected[window] += 1
                tokens[window] += sum(trace.states[state].block_tokens for state in ids)
        for window in bc.WINDOWS:
            self.assertEqual(counters[window]["measured_requests"], expected[window])
            self.assertEqual(counters[window]["requested_tokens"], tokens[window])
        # Both requests at exactly mid are in W1 and not in W2.
        at_mid = sum(1 for call in recorder_calls if call[4] == 3_400_000.0)
        self.assertEqual(at_mid, 2)
        self.assertEqual(counters["W1"]["measured_requests"] - sum(
            1 for call in recorder_calls if split <= call[4] < 3_400_000.0), 2)
        self.assertEqual(counters["full"]["measured_requests"], result.measured_requests)
        self.assertGreater(counters["W2"]["l1_avoided_tokens"] + counters["W2"]["l2_avoided_tokens"],
                           0)

    def test_window_columns_and_differences(self):
        counters = {"full": dict.fromkeys(bc.COUNTERS, 10), "W": dict.fromkeys(bc.COUNTERS, 6),
                    "W1": dict.fromkeys(bc.COUNTERS, 2), "W2": dict.fromkeys(bc.COUNTERS, 4)}
        counters["W"].update(requested_tokens=200, avoided_prefill_tokens=50, l1_avoided_tokens=20)
        counters["W1"].update(requested_tokens=80, avoided_prefill_tokens=20, l1_avoided_tokens=8)
        counters["W2"].update(requested_tokens=120, avoided_prefill_tokens=30,
                              l1_avoided_tokens=12)
        counters["full"].update(requested_tokens=300, avoided_prefill_tokens=90,
                                l1_avoided_tokens=30)
        self.assertEqual(bc.window_problems(counters), [])
        row = bc.window_columns(counters)
        self.assertEqual(row["W_extra_avoided_tokens"], 30)
        self.assertEqual(row["U_W_points"], 15.0)
        self.assertEqual(bc.window_utility(row, "full"), (60, 20.0))
        later = dict(row, W_avoided_prefill_tokens=60)
        self.assertEqual(bc.window_difference(later, row, "W"), (10, 5.0))
        broken = {window: dict(values) for window, values in counters.items()}
        broken["W2"]["l2_hit_blocks"] = 5
        broken["W"]["measured_requests"] = 11
        problems = bc.window_problems(broken)
        self.assertTrue(any("W1 + W2 l2_hit_blocks" in problem for problem in problems))
        self.assertTrue(any("W measured_requests 11 > full 10" in problem for problem in problems))
        with self.assertRaises(ValueError):
            bc.counters_of(row, "head")


# --- the 16-token path ----------------------------------------------------------------------------


class FineTokenPathTests(unittest.TestCase):
    """Required unit test (c): the 16-token path through the unmodified
    converter and loader on a constructed file."""

    def test_the_converter_and_loader_at_16_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            records = raw_bailian_records(seed=2, steps=200)
            fine = convert_raw(directory, "bailian_tob_trace", records, 16)
            coarse = convert_raw(directory, "bailian_tob_trace", records, 512)
            manifest = json.loads(fine.with_name("bailian_tob_trace.manifest.json").read_text())
            self.assertEqual((manifest["block_tokens"], manifest["source"],
                              manifest["source_block_tokens"]),
                             (16, "qwen_traceB_blksz_16.jsonl", 16))
            trace16 = load_mooncake_trace(fine, block_size=16)
            trace512 = load_mooncake_trace(coarse, block_size=512)
            self.assertEqual(trace16.name, trace512.name)
            self.assertEqual([request.timestamp_ms for request in trace16.requests],
                             [request.timestamp_ms for request in trace512.requests])
            self.assertEqual([request.input_length for request in trace16.requests],
                             [request.input_length for request in trace512.requests])
            self.assertEqual([len(request.hash_ids) for request in trace16.requests],
                             [len(record["hash_ids"]) for record in records])
            self.assertEqual(horizon_for(trace16)[:2], horizon_for(trace512)[:2])
            self.assertEqual(horizon_for(trace16)[0], 600.0)
            self.assertGreater(len(trace16.states), len(trace512.states))
            # Two requests share a 16-token state iff they share every source id
            # up to it and its token count: the shared system prompt is one chain.
            prompt_states = {request.hash_ids[95] for request in trace16.requests}
            self.assertLessEqual(len(prompt_states), 2)
            self.assertEqual({trace16.states[state].prefix_tokens for state in prompt_states},
                             {96 * 16})
            # The packed working set at 16 tokens is at most the 512-token one.
            self.assertLessEqual(working_set_bytes(trace16), working_set_bytes(trace512))
            # A replay at 16 tokens on the same windows and the same bytes.
            split = horizon_for(trace16)[1]
            setup = bc.arm_setup("label_binary_150", trace16, 600.0, 0)
            fine_run = replay(trace16, setup, ALL16, seed=0, split_ms=split, windows=True)
            coarse_run = replay(trace512, bc.arm_setup("label_binary_150", trace512, 600.0, 0),
                                ALL16, seed=0, split_ms=split, windows=True)
            for window in bc.WINDOWS:
                for counter in ("measured_requests", "requested_tokens"):
                    self.assertEqual(fine_run["windows"][window][counter],
                                     coarse_run["windows"][window][counter])
            self.assertEqual(bc.window_problems(fine_run["windows"]), [])
            self.assertGreater(fine_run["result"].l2_decisions, 0)


# --- the readings ---------------------------------------------------------------------------------


class ReadingTests(unittest.TestCase):
    def test_the_shortfall_is_never_clipped_and_not_computed_without_headroom(self):
        self.assertEqual(bc.shortfall(10.0, 9.0, 0.0), 0.1)
        self.assertEqual(bc.shortfall(10.0, 12.0, 0.0), -0.2)      # below 0, as is
        self.assertEqual(bc.shortfall(10.0, -5.0, 0.0), 1.5)       # above 1, as is
        self.assertEqual(bc.shortfall(3.0, 2.5, 2.0), 0.5)         # denominator exactly 1.0
        self.assertTrue(math.isnan(bc.shortfall(3.0, 2.5, 2.001)))  # below 1.0 point
        self.assertTrue(math.isnan(bc.shortfall(2.0, 2.5, 3.0)))   # negative denominator
        self.assertTrue(math.isnan(bc.shortfall(math.nan, 1.0, 0.0)))
        self.assertTrue(math.isnan(bc.ratio(1.0, 0.5)))
        self.assertEqual(bc.ratio(3.0, 1.5), 2.0)

    def test_the_evaluable_rule(self):
        self.assertTrue(bc.evaluable(3.0, 2.0))       # exactly 1.0 point
        self.assertFalse(bc.evaluable(2.999, 2.0))
        self.assertFalse(bc.evaluable(math.nan, 2.0))
        self.assertFalse(bc.evaluable(1.0, 2.0))
        self.assertTrue(bc.suffices(0.10))
        self.assertFalse(bc.suffices(0.1000001))
        self.assertFalse(bc.suffices(math.nan))
        self.assertTrue(bc.suffices(-0.3))

    def test_h_best_ties_go_to_the_smaller_horizon(self):
        self.assertEqual(bc.best_horizon({6.0: 0.3, 60.0: 0.05, 150.0: 0.05, 600.0: 0.2}),
                         (60.0, 0.05))
        self.assertEqual(bc.best_horizon({1200.0: -0.1, 600.0: 0.0}), (1200.0, -0.1))
        self.assertEqual(bc.best_horizon({6.0: math.nan, 15.0: 0.4}), (15.0, 0.4))
        best, smallest = bc.best_horizon({6.0: math.nan})
        self.assertIsNone(best)
        self.assertTrue(math.isnan(smallest))
        self.assertEqual(bc.sufficing_horizons({6.0: 0.5, 600.0: 0.1, 300.0: 0.02}),
                         (300.0, 600.0))

    def test_the_classification_in_the_plans_order(self):
        self.assertEqual(bc.classify(True, 0.05, 0.01), "transplant_suffices")
        self.assertEqual(bc.classify(True, 0.10, 0.10), "transplant_suffices")
        self.assertEqual(bc.classify(True, 0.2, 0.05), "retuned_grid_point_suffices")
        self.assertEqual(bc.classify(True, 0.2, 0.15), "none_in_grid")
        self.assertEqual(bc.classify(False, 0.05, 0.01), "no_headroom")
        self.assertEqual(bc.classify(False, math.nan, math.nan), "no_headroom")
        self.assertEqual(bc.CLASSES, ("transplant_suffices", "retuned_grid_point_suffices",
                                      "none_in_grid", "no_headroom"))

    def test_the_direction_of_failure(self):
        entries = [("transplant_suffices", 60.0, 60.0), ("no_headroom", 60.0, 600.0),
                   ("retuned_grid_point_suffices", 60.0, 300.0),
                   ("retuned_grid_point_suffices", 600.0, 150.0),
                   ("none_in_grid", 150.0, 1200.0), ("none_in_grid", 300.0, 300.0)]
        counts = bc.direction_counts(entries)
        self.assertEqual(counts, {"failing": 4, "longer": 2, "shorter": 1, "equal": 1})
        self.assertTrue(bc.direction_holds(counts))
        tie = bc.direction_counts([("none_in_grid", 60.0, 300.0),
                                   ("retuned_grid_point_suffices", 600.0, 60.0)])
        self.assertFalse(bc.direction_holds(tie))
        # 0/0: no evaluable cell fails the transplant.
        none = bc.direction_counts([("transplant_suffices", 60.0, 15.0), ("no_headroom", 60.0, 6.0)])
        self.assertEqual((none["failing"], none["longer"], none["shorter"]), (0, 0, 0))
        self.assertIsNone(bc.direction_holds(none))
        with self.assertRaises(ValueError):
            bc.direction_counts([("suffices", 60.0, 60.0)])
        self.assertEqual(bc.direction(150.0, None), "")

    def test_monotonicity_with_the_merged_1_percent_level(self):
        bests = {(0.0025, 1.0): 60.0, (0.0025, 4.0): 300.0, (0.01, 1.0): 150.0,
                 (0.02, 1.0): 150.0, (0.01, 4.0): 600.0, (0.02, 4.0): 600.0}
        levels = bc.level_bests(bests)
        self.assertEqual(levels, {0.0025: 60.0, 0.01: 150.0, 0.02: 150.0, 0.04: 600.0,
                                  0.08: 600.0})
        reading = bc.monotonicity(levels)
        self.assertEqual((reading["evaluable_levels"], reading["assessable"],
                          reading["nondecreasing"], reading["rests_on_1200"]),
                         (5, True, True, False))
        # A decrease breaks it.
        broken = bc.monotonicity(bc.level_bests({**bests, (0.02, 4.0): 300.0}))
        self.assertFalse(broken["nondecreasing"])
        # A non-evaluable cell is left out; its level merges from the other cell.
        partial = bc.level_bests({**bests, (0.01, 1.0): None})
        self.assertEqual(partial[0.01], 300.0)
        # Fewer than three evaluable levels: not assessable.
        sparse = bc.monotonicity(bc.level_bests({(0.0025, 1.0): 600.0, (0.0025, 4.0): None,
                                                 (0.01, 1.0): None, (0.02, 1.0): 60.0,
                                                 (0.01, 4.0): None, (0.02, 4.0): None}))
        self.assertEqual((sparse["evaluable_levels"], sparse["assessable"],
                          sparse["nondecreasing"]), (2, False, None))
        three = bc.monotonicity(bc.level_bests({(0.0025, 1.0): 60.0, (0.02, 1.0): 60.0,
                                                (0.02, 4.0): 1200.0}))
        self.assertEqual((three["assessable"], three["nondecreasing"], three["rests_on_1200"]),
                         (True, True, True))

    def test_the_1200_second_flag(self):
        self.assertTrue(bc.rests_on_1200(1200.0, (600.0, 1200.0)))
        self.assertTrue(bc.rests_on_1200(600.0, (1200.0,)))
        self.assertFalse(bc.rests_on_1200(600.0, (600.0, 1200.0)))
        self.assertFalse(bc.rests_on_1200(None, ()))

    def test_d_rand_readings(self):
        self.assertEqual(bc.seed_reading([-0.1, -0.2, -0.01, -0.3, -0.5]), "consistent_loss")
        self.assertEqual(bc.seed_reading([-0.1, 0.0, -0.01, -0.3, -0.5]), "mixed")
        self.assertEqual(bc.seed_reading([0.1, 0.2, 0.3, 0.4, 0.5]), "consistent_gain")
        self.assertEqual(bc.seed_signs([-1.0, 0.0, 2.0]), (1, 1, 1))

    def test_the_calibration_half(self):
        first = {h: 0.5 for h in bc.GRID_SECONDS}
        first.update({150.0: 0.05, 300.0: 0.05})
        second = {h: 0.4 for h in bc.GRID_SECONDS}
        second.update({150.0: 0.12, 600.0: 0.03})
        reading = bc.calibration(first, second)
        self.assertEqual((reading["h_cal"], reading["S_hcal_W1"], reading["S_hcal_W2"]),
                         (150.0, 0.05, 0.12))
        self.assertFalse(reading["hcal_suffices_W2"])
        self.assertEqual((reading["h_best_W2"], reading["min_S_W2"]), (600.0, 0.03))
        self.assertTrue(reading["min_S_W2_suffices"])
        self.assertFalse(reading["rests_on_1200"])
        none = bc.calibration({h: math.nan for h in bc.GRID_SECONDS}, second)
        self.assertIsNone(none["h_cal"])
        self.assertTrue(math.isnan(none["S_hcal_W2"]))
        self.assertFalse(none["hcal_suffices_W2"])
        flagged = bc.calibration({**first, 1200.0: 0.0}, second)
        self.assertEqual(flagged["h_cal"], 1200.0)
        self.assertTrue(flagged["rests_on_1200"])

    def test_the_labels_share_of_t(self):
        self.assertEqual(bc.label_share_of_t(6.0, 2.0, 10.0), 0.5)
        self.assertEqual(bc.label_share_of_t(6.0, 2.0, 3.0), 4.0)        # not clipped
        self.assertTrue(math.isnan(bc.label_share_of_t(6.0, 2.0, 2.5)))  # T below 1.0 point

    def test_cell_window_values(self):
        u = {"lru": [1.0, 1.0], "label": [5.0, 7.0]}
        for h in bc.GRID_SECONDS:
            u[horizon_arm(h)] = [2.0, 2.0]
        u["label_binary_150"] = [5.6, 5.8]
        u["label_binary_random_150"] = [5.0, 5.9]
        values = bc.cell_window_values(u, 150.0, u_hoff=11.0)
        self.assertEqual(values["headroom"], 5.0)
        self.assertTrue(values["evaluable"])
        self.assertAlmostEqual(values["S_hstar"], 0.06)
        self.assertEqual((values["h_best"], values["sufficing"]), (150.0, (150.0,)))
        self.assertAlmostEqual(values["min_S"], 0.06)
        self.assertEqual([round(v, 6) for v in values["D_rand"]], [-0.6, 0.1])
        self.assertEqual(values["D_rand_reading"], "mixed")
        self.assertEqual(values["T"], 10.0)
        self.assertEqual(values["label_share_of_T"], 0.5)
        self.assertEqual([round(s, 6) for s in values["S_seeds"][150.0]], [-0.15, 0.2])


class OrderBeyondBitTests(unittest.TestCase):
    """Reading 7 and prediction 7 (Addendum 1)."""

    def test_separated_is_strict(self):
        self.assertTrue(bc.separated([1.0, 1.2], [0.5, 0.99]))
        self.assertFalse(bc.separated([1.0, 1.2], [0.5, 1.0]))      # a tie is not separated
        self.assertFalse(bc.separated([0.0, 2.0], [-1.0, 0.5]))
        self.assertTrue(bc.separated([-0.1, -0.05], [-0.3, -0.2]))  # both negative, still above
        self.assertFalse(bc.separated([], [0.0]))
        self.assertFalse(bc.separated([1.0, math.nan], [0.0]))

    def test_reversed_needs_both_consistent_and_opposite(self):
        self.assertTrue(bc.reversed_signs("consistent_gain", "consistent_loss"))
        self.assertTrue(bc.reversed_signs("consistent_loss", "consistent_gain"))
        self.assertFalse(bc.reversed_signs("consistent_gain", "consistent_gain"))
        self.assertFalse(bc.reversed_signs("consistent_gain", "mixed"))
        self.assertFalse(bc.reversed_signs("mixed", "consistent_loss"))
        with self.assertRaises(ValueError):
            bc.reversed_signs("gain", "consistent_loss")

    def test_the_both_mechanisms_rule_and_the_prediction_count(self):
        self.assertEqual(bc.order_status(True, True), "evaluable")
        self.assertEqual(bc.order_status(True, False), "no_headroom")
        self.assertEqual(bc.order_status(False, True), "no_headroom")
        self.assertEqual(bc.PREDICTION_7_CELLS, ((0.0025, 1.0), (0.0025, 4.0), (0.01, 1.0),
                                                 (0.02, 1.0)))
        self.assertEqual(bc.PREDICTION_7_TRACE_CELLS, 16)
        entries = [((0.0025, 1.0), "evaluable", True), ((0.0025, 4.0), "evaluable", False),
                   ((0.01, 1.0), "no_headroom", True), ((0.01, 4.0), "evaluable", True),
                   ((0.02, 4.0), "evaluable", True), ((0.02, 1.0), "evaluable", True)]
        self.assertEqual(bc.prediction_7_count(entries),
                         {"cells": 4, "evaluable": 3, "separated": 2})
        # Ten separated of 16 holds whatever the number evaluable; nine does not.
        many = [(cell, "evaluable", index < 10) for index, cell in
                enumerate(list(bc.PREDICTION_7_CELLS) * 4)]
        count = bc.prediction_7_count(many)
        self.assertEqual(count, {"cells": 16, "evaluable": 16, "separated": 10})
        self.assertTrue(bc.prediction_holds("7_order_beyond_bit_separated", count["separated"]))
        self.assertEqual(bc.order_horizon("h_star", 0.02, 1.0), 300.0)
        self.assertEqual(bc.order_horizon("600", 0.02, 1.0), 600.0)
        self.assertEqual(bc.order_deltas([3.0, 4.0], [1.0, 5.0]), [2.0, -1.0])
        with self.assertRaises(ValueError):
            bc.order_deltas([1.0], [1.0, 2.0])


class SmokeVerdictTests(unittest.TestCase):
    def test_the_512_token_stop_conditions(self):
        self.assertTrue(bc.smoke_verdict([10.0, 900.0], [100.0, 3072.0], True)["start"])
        self.assertFalse(bc.smoke_verdict([900.1], [100.0], True)["start"])
        self.assertFalse(bc.smoke_verdict([10.0], [3072.1], True)["start"])
        self.assertFalse(bc.smoke_verdict([10.0], [100.0], False)["start"])
        self.assertFalse(bc.smoke_verdict([], [], True)["start"])

    def test_the_16_token_reductions(self):
        full = bc.fine_smoke_verdict([1200.0], [6144.0], True)
        self.assertEqual((full["seeds"], full["max_workers"], full["replays"], full["run"]),
                         ([0, 1, 2, 3, 4], 4, 90, True))
        self.assertAlmostEqual(full["projected_hours"], 23 * 1200.0 / 3600.0)
        reduced = bc.fine_smoke_verdict([1500.0], [6000.0], True)
        self.assertEqual((reduced["seeds"], reduced["max_workers"], reduced["replays"]),
                         ([0, 1, 2], 4, 54))
        self.assertAlmostEqual(reduced["projected_hours"], 14 * 1500.0 / 3600.0)
        self.assertTrue(reduced["run"])
        halved = bc.fine_smoke_verdict([1000.0], [7000.0], True)
        self.assertEqual((halved["seeds"], halved["max_workers"]), ([0, 1, 2, 3, 4], 2))
        self.assertAlmostEqual(halved["projected_hours"], 45 * 1000.0 / 3600.0)
        self.assertFalse(halved["run"])                     # 12.5 h > 12 h
        both = bc.fine_smoke_verdict([1600.0], [7000.0], True)
        self.assertEqual((both["seeds"], both["max_workers"], both["replays"]), ([0, 1, 2], 2, 54))
        self.assertAlmostEqual(both["projected_hours"], 27 * 1600.0 / 3600.0)
        self.assertTrue(both["run"])                        # exactly 12.0 h
        self.assertFalse(bc.fine_smoke_verdict([1700.0], [7000.0], True)["run"])
        self.assertFalse(bc.fine_smoke_verdict([10.0], [10.0], False)["run"])


if __name__ == "__main__":
    unittest.main()
