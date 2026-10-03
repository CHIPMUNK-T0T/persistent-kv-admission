"""Tests for the Bailian external-check runner (scripts/run_bailian_external_check.py).

No real trace is replayed: the plan forbids a replay before the implementation
is reviewed and committed. Everything runs on constructed traces (a Mooncake-
shaped conversation trace, and upstream-shaped 16-token Bailian records turned
into 512- and 16-token traces by the unmodified converter) and on the
published result tables, which are only read. What is checked: the argument
refusals and the files each mode owns; the anchor's comparison with published
rows with and without digests, failing where it must; the worker's all16 and
leaf16 replays differ only in the store's eligibility; the smoke writes no
utility-bearing column; `main` refuses without a passing anchor and smoke
from the same sources; the Mooncake counterpart of reading 7 recomputed from
the published rows is the one the plan's Addendum 1 states; and the whole
pipeline on constructed traces with a reduced design injected by the test
(anchor against references produced by the parents' own workers, smoke, main,
granularity, tabulation), including that a tampered reference publishes
nothing.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from persistent_kv_admission import bailiancheck as bc
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.horizonctl import ALL16, LEAF16, horizon_arm
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("_run_bailian_external_check_under_test",
               REPOSITORY / "scripts/run_bailian_external_check.py")
tabulate = _load("_tabulate_bailian_external_check_under_test",
                 REPOSITORY / "scripts/tabulate_bailian_external_check.py")
fixtures = _load("_bailiancheck_fixtures", REPOSITORY / "tests/test_bailiancheck.py")
mechanism_tests = sys.modules.get("_bailian_mechanism_fixtures") or _load(
    "_bailian_mechanism_fixtures", REPOSITORY / "tests/test_mechanism_control.py")
_LinearRanker = mechanism_tests._LinearRanker

# The capacity base of the constructed traces (their own working set is a few
# hundred MiB, so 0.25% would hold one block): injected by the tests only.
CAPACITY_BASE = 2 * 2**30
DESIGN = runner.Design(traces=("bailian_toc_trace", "bailian_tob_trace"),
                       cells=((0.0025, 1.0), (0.01, 4.0)), seeds=(0, 1), reduced_seeds=(0,),
                       anchor_seeds=(0, 1))
VOLATILE = ("seconds", "worker_peak_rss_mib", "worker_pss_mib_end", "trace_load_seconds")


def _fake_git(*arguments):
    return "" if arguments[0] == "status" else "f" * 40


def _csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_working_set(trace, *args, **kwargs):
    return CAPACITY_BASE


@contextlib.contextmanager
def _patched(**extra):
    """Git answered as for a clean tree at HEAD, the capacity base injected,
    and the runner's module state restored afterwards."""
    patches = [mock.patch.object(runner.rmc, "_git", _fake_git),
               mock.patch.object(runner, "sources_differing_from_head", lambda: []),
               mock.patch.object(runner, "working_set_bytes", _base_working_set),
               mock.patch.dict(runner.SHARED), mock.patch.dict(runner.rdp._SHARED)]
    for name, value in extra.items():
        patches.append(mock.patch.dict(getattr(runner, name), value))
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield


def _quiet(function, *args, **kwargs):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            result = function(*args, **kwargs)
        finally:
            _quiet.last = stdout.getvalue()
    return result


# --- arguments --------------------------------------------------------------------------------------


class ArgumentTests(unittest.TestCase):
    def test_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exists").mkdir()
            paper = root / "paper"
            paper.mkdir()
            (paper / "anchor.csv").write_text("x\n")
            cases = (
                (["main", "a", "--run-dir", str(root / "r"), "--workers", "13"], "hard cap"),
                (["main", "a", "--run-dir", str(root / "r"), "--workers", "0"], "positive"),
                (["main", "a", "--run-dir", str(root / "exists")], "exists"),
                (["anchor", "t", "--run-dir", str(root / "r"), "--paper-dir", str(paper)],
                 "already holds"),
                (["main", "a", "a", "--run-dir", str(root / "r")], "twice"),
                (["smoke", "a", "--run-dir", str(root / "r"), "--workers16", "5"], "workers16"),
                (["granularity", "--trace16", "x", "--trace512", "y", "--run-dir",
                  str(root / "r"), "--workers", "5"], "at most 4"),
                (["granularity", "--trace16", "x", "--trace512", "y", "--run-dir",
                  str(root / "r"), "--paper-dir", str(paper), "--main-dir", str(paper)],
                 "own directory"),
                (["main", "a", "--run-dir", str(paper), "--paper-dir", str(paper)], "exists"),
            )
            for argv, message in cases:
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as caught:
                        runner.validate_arguments(runner.parse_args(argv))
                    self.assertIn(message, str(caught.exception))
            # Another mode's files do not block a mode.
            runner.validate_arguments(runner.parse_args(
                ["smoke", "a", "--run-dir", str(root / "r"), "--paper-dir", str(paper)]))
            args = runner.parse_args(["granularity", "--trace16", "x", "--trace512", "y",
                                      "--run-dir", str(root / "r")])
            self.assertEqual((args.workers, args.paper_dir, args.main_dir, args.reduced_seeds),
                             (4, runner.DEFAULT_GRANULARITY_DIR, runner.DEFAULT_PAPER_DIR, False))
            self.assertEqual(runner.parse_args(["main", "a", "--run-dir", "r"]).workers, 10)

    def test_an_unclean_tree_is_refused_before_anything_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("anchor", "main"):
                argv = [mode, "missing.jsonl", "--run-dir", str(root / "run"), "--paper-dir",
                        str(root / "paper")]
                with mock.patch.object(runner.rmc, "_git", lambda *a: " M src/x.py"):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(argv)
                self.assertIn("not clean", str(caught.exception))
                with mock.patch.object(runner.rmc, "_git", lambda *a: ""), \
                        mock.patch.object(runner, "sources_differing_from_head",
                                          lambda: ["src/persistent_kv_admission/bailiancheck.py"]):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(argv)
                self.assertIn("differs from HEAD", str(caught.exception))
            self.assertEqual(os.listdir(root), [])

    def test_the_files_of_each_mode_are_disjoint_and_described(self):
        groups = (runner.ANCHOR_FILES, runner.SMOKE_FILES, runner.MAIN_FILES)
        names = [name for group in groups for name in group]
        self.assertEqual(len(names), len(set(names)))
        for name in runner.MAIN_TABLES:
            self.assertIn(f"`{name}.csv`", runner.README_TEXT)
        for name in ("anchor.csv", "smoke.csv", "smoke_config.json", "run_config.json"):
            self.assertIn(f"`{name}`", runner.README_TEXT)
        for name in runner.GRANULARITY_TABLES:
            self.assertIn(f"`{name}.csv`", runner.GRANULARITY_README)

    def test_the_registered_design_is_the_plans(self):
        self.assertEqual(runner.REGISTERED, runner.Design(bc.TRACES, bc.CELLS, bc.SEEDS))
        self.assertTrue(runner._registered(runner.Design()))
        self.assertFalse(runner._registered(DESIGN))
        tasks = (bc.main_replays(runner.REGISTERED.traces, runner.REGISTERED.cells)
                 + bc.heap_replays())
        self.assertEqual(len(tasks), 2448)

    def test_the_execution_sources_cover_the_chain(self):
        sources = {str(path.relative_to(REPOSITORY)) for path in runner.execution_sources()}
        for path in ("scripts/run_bailian_external_check.py",
                     "scripts/run_leaf_matched_horizon.py", "scripts/run_matched_class_order.py",
                     "scripts/run_horizon_control.py", "scripts/run_error_location.py",
                     "scripts/run_mechanism_control.py", "scripts/run_decision_population.py",
                     "src/persistent_kv_admission/bailiancheck.py",
                     "src/persistent_kv_admission/classmix.py"):
            self.assertIn(path, sources)


# --- the anchor's comparison ---------------------------------------------------------------------


def _replay_row(**overrides) -> dict:
    row = {"trace": "conversation_trace", "l1_fraction": 0.0025, "l2_multiplier": 1.0,
           "cell": "l1=0.0025,l2x1", "mechanism": "all16", "arm": "lru", "seed": 0,
           "avoided_prefill_tokens": 1000, "counters_sha256": "c" * 64,
           "decision_sha256": "d" * 64, "window_problems": "", "seconds": 1.0,
           "worker_peak_rss_mib": 10.0}
    for column in runner.rlm.REQUIRED_COUNTER_COLUMNS:
        row.setdefault(column, 7)
    for column in runner.IDENTIFIERS:
        row[column] = 7
    row.update(overrides)
    return row


def _published(row, digests=True, **overrides) -> dict:
    published = {column: str(value) for column, value in row.items()}
    published.update({"rung": row["arm"], "variant": "main", "eligibility": "all", "width": "16"})
    if not digests:
        published.pop("counters_sha256")
        published.pop("decision_sha256")
    for column in ("window_problems",):
        published.pop(column)
    published.update(overrides)
    return published


class AnchorComparisonTests(unittest.TestCase):
    def test_with_digests(self):
        row = _replay_row()
        entry = runner.compare_with_published(row, {"error_location_001": _published(row)})
        self.assertTrue(entry["reproduces"])
        self.assertEqual((entry["same_counters_sha256"], entry["same_decision_sha256"],
                          entry["counter_columns_compared"]), (True, True, 0))
        for column, value in (("counters_sha256", "0" * 64), ("decision_sha256", "1" * 64),
                              ("avoided_prefill_tokens", "1001"), ("l1_capacity_bytes", "8")):
            with self.subTest(column=column):
                entry = runner.compare_with_published(
                    row, {"error_location_001": _published(row, **{column: value})})
                self.assertFalse(entry["reproduces"])

    def test_without_digests_every_counter_column_is_compared(self):
        row = _replay_row(extra_points=1.25, share_evictions_with_orphans=math.nan)
        published = _published(row, digests=False)
        entry = runner.compare_with_published(row, {"mechanism_control_001": published})
        self.assertTrue(entry["reproduces"], entry)
        self.assertEqual(entry["same_counters_sha256"], "")
        self.assertEqual(entry["counter_columns_source"], "mechanism_control_001")
        self.assertGreaterEqual(entry["counter_columns_compared"],
                                len(runner.rlm.REQUIRED_COUNTER_COLUMNS))
        for column, value in (("extra_points", "1.2500000001"), ("l2_evictions", "8")):
            with self.subTest(column=column):
                entry = runner.compare_with_published(
                    row, {"mechanism_control_001": dict(published, **{column: value})})
                self.assertFalse(entry["reproduces"])
                self.assertIn(column, entry["counter_columns_differing"])
        missing = dict(published)
        missing.pop("l2_rejections")
        entry = runner.compare_with_published(row, {"mechanism_control_001": missing})
        self.assertFalse(entry["reproduces"])
        self.assertIn("l2_rejections", entry["required_counter_columns_unpublished"])

    def test_two_sources_must_both_agree(self):
        row = _replay_row(mechanism="leaf16", arm="label")
        sources = {"leaf_matched_horizon_001": _published(row),
                   "mechanism_control_001": _published(row, digests=False)}
        entry = runner.compare_with_published(row, sources)
        self.assertTrue(entry["reproduces"])
        self.assertEqual(entry["reference_sources"],
                         "leaf_matched_horizon_001;mechanism_control_001")
        sources["mechanism_control_001"]["l2_admissions"] = "9"
        self.assertFalse(runner.compare_with_published(row, sources)["reproduces"])

    def test_the_published_references_load_as_the_plan_names_them(self):
        published, problems = runner.load_anchor_references()
        self.assertEqual(problems, [])
        self.assertEqual(len(published), 70)
        for key, sources in published.items():
            self.assertEqual(tuple(sources), bc.ANCHOR_SOURCES[key[:2]])
        lru = published[(LEAF16, "lru", "conversation_trace", 0.0025, 1.0, 0)]
        self.assertNotIn("counters_sha256", lru["mechanism_control_001"])
        label = published[(LEAF16, "label", "conversation_trace", 0.01, 4.0, 3)]
        self.assertEqual(label["leaf_matched_horizon_001"]["avoided_prefill_tokens"],
                         label["mechanism_control_001"]["avoided_prefill_tokens"])
        self.assertTrue(label["leaf_matched_horizon_001"]["counters_sha256"])

    def test_the_anchor_gate(self):
        manifest = {"sha256": "m" * 64}
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            design = runner.Design(anchor_seeds=(0,))
            entries, problems = runner.anchor_gate(paper, manifest, design)
            self.assertIn("no anchor", problems[0])
            rows = [{"mechanism": task[3], "arm": task[4], "l1_fraction": task[1],
                     "l2_multiplier": task[2], "seed": task[5], "reproduces": True}
                    for task in bc.anchor_replays((0,))]
            runner.rdp._write(paper / "anchor.csv", rows)
            config = {"mode": "anchor", "passed": True, "source_manifest": manifest}
            (paper / "anchor_config.json").write_text(json.dumps(config))
            self.assertEqual(runner.anchor_gate(paper, manifest, design)[1], [])
            self.assertIn("other execution sources",
                          runner.anchor_gate(paper, {"sha256": "x"}, design)[1][0])
            self.assertIn("not the plan's", runner.anchor_gate(paper, manifest)[1][0])
            rows[3]["reproduces"] = False
            runner.rdp._write(paper / "anchor.csv", rows)
            self.assertIn("do not reproduce", runner.anchor_gate(paper, manifest, design)[1][0])


# --- the worker -------------------------------------------------------------------------------------


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = fixtures.write_trace(cls.directory.name, "bailian_coder_trace",
                                        fixtures.mooncake_records(seed=5))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _state(self):
        info = _quiet(runner.load_input, self.path, 512)
        runner.rdp._SHARED.update(working_set={info["name"]: CAPACITY_BASE})
        runner.SHARED.update(inputs={info["name"]: runner._worker_spec(info)})
        return info

    def test_all16_and_leaf16_differ_only_in_the_eligibility(self):
        calls = []
        real = runner.run_two_tier

        def capture(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        with _patched(), mock.patch.object(runner, "run_two_tier", capture):
            info = self._state()
            for mechanism in bc.MECHANISM_NAMES:
                runner._replay_worker((info["name"], 0.01, 4.0, mechanism,
                                       "label_binary_random_600", 2))
        (args_all, kwargs_all), (args_leaf, kwargs_leaf) = calls
        self.assertEqual(args_all, args_leaf)
        differing = {key for key in kwargs_all
                     if not callable(kwargs_all[key]) and kwargs_all[key] != kwargs_leaf[key]
                     and key not in ("l2_scorer",)}
        self.assertEqual(differing, {"l2_eligibility"})
        self.assertEqual((kwargs_all["l2_eligibility"], kwargs_leaf["l2_eligibility"]),
                         ("all", "leaf"))
        self.assertEqual({type(kwargs_all[key]) for key in ("l2_scorer", "l2_request_hook",
                                                            "l2_override_hook")},
                         {type(kwargs_leaf[key]) for key in ("l2_scorer", "l2_request_hook",
                                                             "l2_override_hook")})
        self.assertEqual(kwargs_all["l2_sample_width"], 16)

    def test_worker_rows_pass_every_check(self):
        with _patched():
            info = self._state()
            tasks = [(info["name"], fraction, multiplier, mechanism, arm, 1)
                     for fraction, multiplier in ((0.0025, 4.0), (0.02, 1.0))
                     for mechanism in bc.MECHANISM_NAMES
                     for arm in bc.cell_arms(fraction, multiplier)]
            tasks += [(info["name"], fraction, multiplier, "heap", arm, None)
                      for fraction, multiplier in ((0.0025, 4.0), (0.02, 1.0))
                      for arm in bc.HEAP_ARMS]
            rows = [runner._replay_worker(task) for task in tasks]
        checks = runner.run_checks(rows, tasks)
        self.assertTrue(all(check["passes"] for check in checks), checks)
        heap = [row for row in rows if row["mechanism"] == "heap"]
        self.assertEqual({row["heap_policy"] for row in heap}, {"lru", "offline_next_use"})
        for row in rows:
            self.assertEqual(row["window_problems"], "")
            if row["mechanism"] == "heap":
                continue
            self.assertEqual(row["overridden_decisions_seen"], 0)
            self.assertEqual(row["candidate_decisions_seen"], row["l2_decisions"])
            if row["family"] == "random":
                self.assertEqual(row["random_draws"], row["candidates_seen"])
                self.assertIs(row["random_stream_separate"], True)
        # The checks have teeth.
        broken = [dict(row) for row in rows]
        random_row = next(row for row in broken if row["family"] == "random")
        random_row["random_draws"] += 1
        broken[0]["W_requested_tokens"] += 1
        broken[1]["window_problems"] = "W1 + W2 differs"
        leaf = next(row for row in broken if row["mechanism"] == LEAF16)
        leaf["full_l2_present_unusable_tokens"] = 3
        failing = {check["check"] for check in runner.run_checks(broken, tasks)
                   if not check["passes"]}
        self.assertEqual(failing, {"identifiers", "windows", "leaf16_closure", "random_stream"})
        self.assertIn("missing", runner.check_complete(rows[1:], tasks)[0])

    def test_the_worker_refuses_a_trace_that_is_not_the_parents(self):
        with _patched():
            info = self._state()
            spec = dict(runner.SHARED["inputs"][info["name"]], split_ms=1.0)
            runner.SHARED["inputs"] = {info["name"]: spec}
            runner._CACHE.clear()
            with self.assertRaises(RuntimeError):
                runner._replay_worker((info["name"], 0.01, 1.0, ALL16, "lru", 0))
            with self.assertRaises(ValueError):
                runner._replay_worker((info["name"], 0.01, 1.0, ALL16, "label_binary_random_60",
                                       0))

    def test_the_smoke_row_carries_no_utility(self):
        markers = ("avoided", "hit", "present", "extra", "points", "U_", "unusable", "loss",
                   "absent_", "utility", "counters_sha256", "orphan", "requested_tokens")
        for column in runner.SMOKE_COLUMNS:
            if column == "check_no_unexplained_absent":
                continue
            self.assertFalse(any(marker in column for marker in markers), column)
        with _patched():
            info = self._state()
            row = runner._replay_worker((info["name"], 0.01, 1.0, ALL16,
                                         "label_binary_random_150", 0))
        smoke = runner.smoke_projection(row, 123.0)
        self.assertEqual(tuple(smoke), runner.SMOKE_COLUMNS)
        self.assertTrue(smoke["check_replay"])
        self.assertEqual(smoke["random_draws"], smoke["candidates_seen"])
        self.assertEqual(smoke["parent_rss_mib_at_fork"], 123.0)
        values = {str(value) for value in smoke.values()}
        for column in ("avoided_prefill_tokens", "extra_avoided_tokens", "l2_avoided_tokens"):
            self.assertNotIn(column, smoke)
        self.assertNotIn(str(row["U_W_points"]), values)


# --- running tasks ------------------------------------------------------------------------------------


def _toy_worker(task):
    """A stand-in replay: dies on task 2 (`os._exit`), raises on task 3."""
    if task == 2:
        os._exit(3)
    if task == 3:
        raise ValueError("a replay failed")
    return {"trace": "t", "mechanism": "all16", "cell": "c", "arm": "a", "seed": task,
            "seconds": 0.0, "worker_peak_rss_mib": 0.0}


class RunTaskTests(unittest.TestCase):
    def test_rows_are_streamed_and_failures_make_the_run_incomplete(self):
        for isolated in (True, False):
            with self.subTest(isolated=isolated):
                raw = io.StringIO()
                rows = _quiet(runner.run_tasks, [0, 1, 4], 2, _toy_worker, raw, isolated)
                self.assertEqual(sorted(row["seed"] for row in rows), [0, 1, 4])
                self.assertEqual(len(raw.getvalue().splitlines()), 3)
                for task, message in ((2, "died"), (3, "failed")):
                    with self.assertRaises(runner.IncompleteRun) as caught:
                        _quiet(runner.run_tasks, [0, task, 1], 2, _toy_worker, io.StringIO(),
                               isolated)
                    self.assertIn(message, str(caught.exception))


# --- the Mooncake counterpart -------------------------------------------------------------------------


class MooncakeTests(unittest.TestCase):
    def test_the_published_counterpart_is_the_addendums(self):
        side, order, problems = runner.load_mooncake()
        self.assertEqual(problems, [])
        self.assertEqual(len(side), 24)
        self.assertEqual(len(order), 24)
        hstar = [row for row in order if row["role"] == "h_star"]
        self.assertEqual(len(hstar), 12)

        def count(mechanism, reading):
            return sum(1 for row in hstar if row[f"Delta_{mechanism}_reading_full"] == reading)

        self.assertEqual((count(ALL16, "consistent_loss"), count(ALL16, "consistent_gain"),
                          count(ALL16, "mixed")), (6, 1, 5))
        self.assertEqual((count(LEAF16, "consistent_gain"), count(LEAF16, "mixed")), (8, 4))
        self.assertEqual(sum(1 for row in hstar if row["separated_full"] is True), 8)
        self.assertEqual(sum(1 for row in hstar if row["reversed_full"] is True), 6)
        # Every cell with h* < 600 s is a consistent leaf16 gain; the four
        # 600-second cells are mixed.
        for row in hstar:
            expected = "mixed" if row["h"] == 600.0 else "consistent_gain"
            self.assertEqual(row[f"Delta_{LEAF16}_reading_full"], expected)
            if row["h"] < 600.0:
                self.assertGreaterEqual(row[f"Delta_{LEAF16}_points_min_full"], 0.85)
                self.assertLessEqual(row[f"Delta_{LEAF16}_points_max_full"], 1.95)
        # Delta(600) under leaf16 exists only at the 600-second cells.
        at600 = [row for row in order if row["role"] == "600"]
        self.assertEqual(sum(1 for row in at600 if row["separated_full"] != ""), 4)
        # Reading 6: the published S at h* and the published D_rand.
        conversation = next(row for row in side if row["trace"] == "conversation_trace"
                            and row["mechanism"] == ALL16 and row["l1_fraction"] == 0.01
                            and row["l2_multiplier"] == 4.0)
        published = next(row for row in _csv(REPOSITORY / "results/paper/horizon_control_001/"
                                                           "horizon.csv")
                         if row["trace"] == "conversation_trace" and float(row["h"]) == 600.0
                         and float(row["l1_fraction"]) == 0.01
                         and float(row["l2_multiplier"]) == 4.0)
        self.assertEqual(conversation["S_hstar_full"], float(published["S"]))
        self.assertEqual(conversation["grid_points_full"], "6;15;60;300;600")
        fill_cell = next(row for row in side if row["mechanism"] == ALL16
                         and row["l1_fraction"] == 0.01 and row["l2_multiplier"] == 1.0)
        self.assertEqual(fill_cell["grid_points_full"], "6;15;60;150;300;600")
        self.assertIn("learned ranker", conversation["D_rand_note"])

    def test_a_missing_published_row_is_a_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = _csv(runner.MOONCAKE_SOURCES["leaf_matched_horizon_001/replay_seeds.csv"])
            rows = [row for row in rows if not (row["arm"] == "label" and row["seed"] == "4")]
            path = Path(directory) / "seeds.csv"
            runner.rdp._write(path, rows)
            _, _, problems = runner.load_mooncake(
                {"leaf_matched_horizon_001/replay_seeds.csv": path})
            self.assertTrue(any("leaf16 label" in problem for problem in problems))


# --- the whole pipeline on constructed traces ---------------------------------------------------------


def _parent_references(trace, directory: Path, design) -> dict[str, Path]:
    """The anchor's published sources, produced by the workers of the runs
    that published them (error location, horizon control, mechanism control,
    leaf-matched horizon) on a constructed conversation trace, plus rows the
    loader must skip."""
    name = trace.name
    horizon, split, _ = horizon_for(trace)
    shared = dict(traces={name: trace}, groups={name: _occurrence_groups(trace)},
                  splits={name: split}, horizons={name: horizon},
                  rankers={name: _LinearRanker()}, real_rankers={})
    rows = {source: [] for source in runner.ANCHOR_REFERENCE_SOURCES}
    with contextlib.ExitStack() as stack:
        for module in (runner.rel, runner.hc, runner.rmc, runner.rlm):
            stack.enter_context(mock.patch.dict(module.SHARED, shared))
        stack.enter_context(mock.patch.dict(runner.rdp._SHARED,
                                            {"working_set": {name: CAPACITY_BASE}}))
        for fraction, multiplier in design.anchor_cells:
            h = bc.hstar(fraction, multiplier)
            for seed in design.anchor_seeds:
                for arm in ("lru", "label"):
                    rows["error_location_001"].append(
                        runner.rel._replay_worker((name, fraction, multiplier, arm, seed, "main")))
                    rows["mechanism_control_001"].append(
                        runner.rmc._replay_worker((name, fraction, multiplier, "leaf", 16, arm,
                                                   seed)))
                for arm in ("label_binary_60", "label_binary_600"):
                    rows["horizon_control_001"].append(
                        runner.hc._replay_worker((name, fraction, multiplier, arm, seed, "main")))
                for arm in ("label", horizon_arm(h)):
                    rows["leaf_matched_horizon_001"].append(
                        runner.rlm._replay_worker((name, fraction, multiplier, arm, seed)))
            # Rows the loader must skip: another horizon, another mechanism.
            rows["horizon_control_001"].append(
                runner.hc._replay_worker((name, fraction, multiplier, "label_binary_6", 0, "main")))
            rows["mechanism_control_001"].append(
                runner.rmc._replay_worker((name, fraction, multiplier, "all", 16, "lru", 0)))
    paths = {}
    for source, source_rows in rows.items():
        path = directory / "published" / source / "replay_seeds.csv"
        path.parent.mkdir(parents=True)
        runner.rdp._write(path, source_rows)
        paths[source] = path
    return paths


class PipelineTests(unittest.TestCase):
    """anchor -> smoke -> main -> granularity -> tabulation on constructed
    traces, with the reduced design injected by the test."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.root = root
        (root / "mooncake").mkdir()
        conversation = fixtures.write_trace(root / "mooncake", "conversation_trace",
                                            fixtures.mooncake_records(seed=9))
        cls.conversation = conversation
        cls.references = _parent_references(load_mooncake_trace(conversation, 512), root, DESIGN)
        cls.bailian = {}
        for name, seed in (("bailian_toc_trace", 3), ("bailian_tob_trace", 4)):
            records = fixtures.raw_bailian_records(seed=seed, steps=240)
            cls.bailian[name] = fixtures.convert_raw(root / "data", name, records, 512)
        cls.tob16 = fixtures.convert_raw(root / "data", "bailian_tob_trace",
                                         fixtures.raw_bailian_records(seed=4, steps=240), 16)
        cls.paper = root / "paper"
        cls.fine_paper = root / "paper_fine"
        traces = [str(cls.bailian[name]) for name in DESIGN.traces]
        with _patched(ANCHOR_REFERENCE_SOURCES=cls.references):
            # Nothing runs before the anchor.
            with cls._refused("the anchor and the smoke do not let the run start"):
                _quiet(runner.run_main, runner.parse_args(
                    ["main", *traces, "--run-dir", str(root / "run_early"),
                     "--paper-dir", str(cls.paper)]), DESIGN)
            _quiet(runner.run_anchor, runner.parse_args(
                ["anchor", str(conversation), "--run-dir", str(root / "run_anchor"),
                 "--paper-dir", str(cls.paper), "--workers", "2"]), DESIGN)
            cls.anchor_output = _quiet.last
            with cls._refused("the anchor and the smoke do not let the run start"):
                _quiet(runner.run_main, runner.parse_args(
                    ["main", *traces, "--run-dir", str(root / "run_early2"),
                     "--paper-dir", str(cls.paper)]), DESIGN)
            _quiet(runner.run_smoke, runner.parse_args(
                ["smoke", *traces, "--trace16", str(cls.tob16), "--run-dir",
                 str(root / "run_smoke"), "--paper-dir", str(cls.paper), "--workers", "3",
                 "--workers16", "2"]), DESIGN)
            cls.smoke_output = _quiet.last
            _quiet(runner.run_main, runner.parse_args(
                ["main", *traces, "--run-dir", str(root / "run_main"), "--paper-dir",
                 str(cls.paper), "--workers", "2"]), DESIGN)
            cls.main_output = _quiet.last
            _quiet(runner.run_granularity, runner.parse_args(
                ["granularity", "--trace16", str(cls.tob16), "--trace512",
                 str(cls.bailian["bailian_tob_trace"]), "--main-dir", str(cls.paper),
                 "--run-dir", str(root / "run_fine"), "--paper-dir", str(cls.fine_paper),
                 "--workers", "2"]), DESIGN)
            cls.fine_output = _quiet.last

    @classmethod
    @contextlib.contextmanager
    def _refused(cls, message):
        try:
            yield
        except SystemExit as error:
            if message not in str(error):
                raise AssertionError(f"refused with {error}, expected {message!r}") from error
        else:
            raise AssertionError(f"not refused; expected {message!r}")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_the_anchor_reproduces_the_parents_rows(self):
        table = _csv(self.paper / "anchor.csv")
        self.assertEqual(len(table), 28)
        self.assertTrue(all(row["reproduces"] == "True" for row in table))
        by_source = {}
        for row in table:
            by_source.setdefault(row["reference_sources"], []).append(row)
        self.assertEqual(set(by_source), {"error_location_001", "horizon_control_001",
                                          "mechanism_control_001",
                                          "leaf_matched_horizon_001;mechanism_control_001",
                                          "leaf_matched_horizon_001"})
        for row in by_source["mechanism_control_001"]:
            self.assertEqual(row["same_counters_sha256"], "")
            self.assertGreaterEqual(int(row["counter_columns_compared"]), 80)
        for row in by_source["error_location_001"] + by_source["horizon_control_001"]:
            self.assertEqual((row["same_counters_sha256"], row["same_decision_sha256"]),
                             ("True", "True"))
        config = json.loads((self.paper / "anchor_config.json").read_text())
        self.assertTrue(config["passed"])
        self.assertEqual((config["code_commit"], config["plan_commit"]), ("f" * 40, "f" * 40))
        self.assertEqual(config["source_manifest"]["sha256"],
                         runner.source_manifest()["sha256"])
        self.assertIn("28 of 28", self.anchor_output)

    def test_a_tampered_reference_publishes_no_anchor(self):
        rows = _csv(self.references["horizon_control_001"])
        target = next(row for row in rows if row["arm"] == "label_binary_600"
                      and row["seed"] == "1")
        target["decision_sha256"] = "0" * 64
        path = self.root / "tampered" / "replay_seeds.csv"
        path.parent.mkdir()
        runner.rdp._write(path, rows)
        paper = self.root / "paper_tampered"
        with _patched(ANCHOR_REFERENCE_SOURCES=dict(self.references,
                                                    horizon_control_001=path)):
            with self.assertRaises(SystemExit) as caught:
                _quiet(runner.run_anchor, runner.parse_args(
                    ["anchor", str(self.conversation), "--run-dir",
                     str(self.root / "run_tampered"), "--paper-dir", str(paper),
                     "--workers", "2"]), DESIGN)
        self.assertIn("does not reproduce", str(caught.exception))
        self.assertFalse(paper.exists())
        table = _csv(self.root / "run_tampered" / "anchor.csv")
        failing = [row for row in table if row["reproduces"] != "True"]
        self.assertEqual(len(failing), 1)
        self.assertEqual((failing[0]["arm"], failing[0]["seed"]), ("label_binary_600", "1"))
        self.assertTrue(all(row["same_decision_sha256"] == "False" for row in failing))

    def test_the_smoke(self):
        rows = _csv(self.paper / "smoke.csv")
        self.assertEqual(len(rows), 8 + 3)
        self.assertEqual(list(rows[0]), list(runner.SMOKE_COLUMNS)
                         + ["check_identifiers_across_arms"])
        for row in rows:
            self.assertTrue(all(row[column] == "True" for column in row
                                if column.startswith("check_")), row)
        self.assertEqual({row["block_tokens"] for row in rows}, {"512", "16"})
        self.assertEqual({row["arm"] for row in rows if row["block_tokens"] == "16"},
                         {"lru", "label", "label_binary_150"})
        config = json.loads((self.paper / "smoke_config.json").read_text())
        self.assertTrue(config["verdict_512"]["start"])
        self.assertTrue(config["verdict_16"]["run"])
        self.assertEqual(config["verdict_16"]["seeds"], [0, 1, 2, 3, 4])
        self.assertIn("START the run", self.smoke_output)
        text = (self.paper / "smoke.csv").read_text()
        for word in ("avoided", "points", "extra"):
            self.assertNotIn(word, text)

    def test_main_publishes_every_table_after_every_check(self):
        self.assertEqual(set(os.listdir(self.paper)),
                         set(runner.MAIN_FILES) | set(runner.ANCHOR_FILES)
                         | set(runner.SMOKE_FILES))
        self.assertEqual(set(os.listdir(self.root / "run_main")),
                         set(runner.MAIN_FILES) | {"raw_replays.jsonl"})
        replays = _csv(self.paper / "replay_seeds.csv")
        self.assertEqual(len(replays), 160 + 8)
        checks = _csv(self.paper / "checks.csv")
        self.assertTrue(all(row["passes"] == "True" for row in checks), checks)
        self.assertEqual({row["check"] for row in checks},
                         {"complete", "identifiers", "windows", "mechanism", "statistics",
                          "leaf16_closure", "phase098b", "random_stream"})
        counts = {name: len(_csv(self.paper / f"{name}.csv"))
                  for name in ("transplant", "grid", "grid_summary", "monotonicity", "random",
                               "calibration", "order_beyond_bit", "heap_references",
                               "mooncake_side_by_side", "mooncake_order_beyond_bit", "replay")}
        self.assertEqual(counts, {"transplant": 8, "grid": 56, "grid_summary": 8,
                                  "monotonicity": 8, "random": 8, "calibration": 8,
                                  "order_beyond_bit": 8, "heap_references": 8,
                                  "mooncake_side_by_side": 8 + 24,
                                  "mooncake_order_beyond_bit": 24,
                                  "replay": (2 * 2 * 2 * 10 + 2 * 2 * 2) * 4})
        for row in replays:
            for counter in bc.COUNTERS:
                self.assertEqual(int(row[f"W1_{counter}"]) + int(row[f"W2_{counter}"]),
                                 int(row[f"W_{counter}"]))
            self.assertEqual(row["window_problems"], "")
        config = json.loads((self.paper / "run_config.json").read_text())
        self.assertFalse(config["registered_design"])
        self.assertIn(sha256_path(self.paper / "anchor.csv"), config["anchor"]["paths"].values())
        self.assertTrue(config["smoke"]["verdict_512"]["start"])
        for name in DESIGN.traces:
            self.assertEqual(config["trace_files"][name]["sha256"],
                             sha256_path(self.bailian[name]))
            self.assertEqual(config["trace_files"][name]["manifest"]["contents"]["block_tokens"],
                             512)
        self.assertIn("leaf_matched_horizon_001/replay_seeds.csv",
                      "".join(config["mooncake_tables"]))

    def test_the_readings_are_arithmetic_on_the_replays(self):
        replays = _csv(self.paper / "replay_seeds.csv")
        index = {(row["trace"], row["mechanism"], float(row["l1_fraction"]),
                  float(row["l2_multiplier"]), row["arm"], row["seed"]): row for row in replays}

        def u(name, mechanism, cell, arm, window="W"):
            return bc.mean(bc.window_utility(index[(name, mechanism) + cell + (arm, str(seed))],
                                             window)[1] for seed in DESIGN.seeds)

        transplant = _csv(self.paper / "transplant.csv")
        for row in transplant:
            cell = (float(row["l1_fraction"]), float(row["l2_multiplier"]))
            h = bc.hstar(*cell)
            label, lru, rung = (u(row["trace"], row["mechanism"], cell, arm)
                                for arm in ("label", "lru", horizon_arm(h)))
            expected = bc.shortfall(label, rung, lru)
            published = float(row["S_hstar_W"]) if row["S_hstar_W"] != "nan" else math.nan
            if math.isnan(expected):
                self.assertTrue(math.isnan(published))
            else:
                self.assertAlmostEqual(published, expected, places=12)
            self.assertEqual(row["evaluable"] == "True", bc.evaluable(label, lru))
            heap = index[(row["trace"], "heap") + cell + ("H_off", "")]
            self.assertAlmostEqual(float(row["H_off_points_W"]),
                                   bc.window_utility(heap, "W")[1], places=12)
        summary = {(row["trace"], row["mechanism"], row["cell"]): row
                   for row in _csv(self.paper / "grid_summary.csv")}
        for row in summary.values():
            self.assertIn(row["classification"], bc.CLASSES)
        order = _csv(self.paper / "order_beyond_bit.csv")
        for row in order:
            cell = (float(row["l1_fraction"]), float(row["l2_multiplier"]))
            for mechanism in bc.MECHANISM_NAMES:
                deltas = [bc.window_difference(
                    index[(row["trace"], mechanism) + cell + ("label", str(seed))],
                    index[(row["trace"], mechanism) + cell + (row["rung_arm"], str(seed))],
                    "W")[1] for seed in DESIGN.seeds]
                self.assertAlmostEqual(float(row[f"Delta_{mechanism}_points_mean_W"]),
                                       bc.mean(deltas), places=12)
                self.assertEqual(row[f"Delta_{mechanism}_reading_W"], bc.seed_reading(deltas))
        readings = _csv(self.paper / "readings.csv")
        names = {row["reading"] for row in readings if row["kind"] == "prediction"}
        self.assertEqual(names, {"1_transplant_all16", "2_direction_of_failure_all16",
                                 "3_grid_all16", "4_monotonicity_all16",
                                 "5_random_order_all16", "6_grid_leaf16",
                                 "7_order_beyond_bit_separated"})
        seven = next(row for row in readings if row["reading"] == "7_order_beyond_bit_separated"
                     and row["window"] == "W")
        predicted = [row for row in order if row["predicted"] == "True"]
        self.assertEqual(int(seven["cells"]), len(predicted))
        self.assertEqual(int(seven["evaluable"]),
                         sum(1 for row in predicted if row["status"] == "evaluable"))
        self.assertEqual(int(seven["count"]),
                         sum(1 for row in predicted if row["status"] == "evaluable"
                             and row["separated_W"] == "True"))
        self.assertEqual(seven["threshold"], "10")
        mooncake = [row for row in readings if row["workload"] == "mooncake"
                    and row["reading"] == "reading7_order_beyond_bit_h_star"]
        self.assertEqual(mooncake[0]["count"], "8/6")

    def test_granularity(self):
        self.assertEqual(set(os.listdir(self.fine_paper)), set(runner.GRANULARITY_FILES))
        replays = _csv(self.fine_paper / "replay_seeds.csv")
        self.assertEqual(len(replays), 2 * 3 * 2)
        self.assertEqual({row["block_tokens"] for row in replays}, {"16"})
        checks = _csv(self.fine_paper / "checks.csv")
        self.assertTrue(all(row["passes"] == "True" for row in checks), checks)
        self.assertIn("same_bytes_and_windows", {row["check"] for row in checks})
        main_rows = {(row["l1_fraction"], row["l2_multiplier"], row["arm"], row["seed"]): row
                     for row in _csv(self.paper / "replay_seeds.csv")
                     if row["trace"] == "bailian_tob_trace" and row["mechanism"] == ALL16}
        for row in replays:
            other = main_rows[(row["l1_fraction"], row["l2_multiplier"], row["arm"], row["seed"])]
            for column in ("l1_capacity_bytes", "l2_capacity_bytes", "W_requested_tokens",
                           "full_requested_tokens"):
                self.assertEqual(row[column], other[column])
        table = _csv(self.fine_paper / "granularity.csv")
        self.assertEqual(len(table), 2 * 2)
        for row in table:
            self.assertIn(row["agree"], ("True", "False", ""))
        config = json.loads((self.fine_paper / "run_config.json").read_text())
        self.assertEqual((config["seeds"], config["reduced_seeds"]), ([0, 1], False))
        self.assertEqual(config["trace16"]["manifest"]["contents"]["block_tokens"], 16)
        self.assertEqual(config["capacity_base_bytes"], CAPACITY_BASE)
        self.assertEqual(config["capacities"]["l1=0.0025,l2x1"]["l1_capacity_bytes"],
                         round(CAPACITY_BASE * 0.0025))

    def test_granularity_follows_the_smoke_verdict(self):
        with _patched():
            for argv, message in ((["--reduced-seeds"], "--reduced-seeds is True"),):
                with self.assertRaises(SystemExit) as caught:
                    _quiet(runner.run_granularity, runner.parse_args(
                        ["granularity", "--trace16", str(self.tob16), "--trace512",
                         str(self.bailian["bailian_tob_trace"]), "--main-dir", str(self.paper),
                         "--run-dir", str(self.root / "run_fine_refused"), "--paper-dir",
                         str(self.root / "paper_fine_refused")] + argv), DESIGN)
                self.assertIn("cannot start", str(caught.exception))
                self.assertIn(message, _quiet.last)
                self.assertFalse((self.root / "run_fine_refused").exists())

    def test_the_tabulation_reads_and_prints_without_writing(self):
        before = sorted(os.listdir(self.paper))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(tabulate.main([str(self.paper), "--granularity-dir",
                                            str(self.fine_paper)]), 0)
        printed = stdout.getvalue()
        for section in ("X0 anchor", "X0 smoke", "X1 ", "X2 ", "X3 ", "X4 ", "X5 ", "X6 ",
                        "X7 ", "X8 ", "X9 ", "C1 granularity"):
            self.assertIn(f"### {section}", printed)
        self.assertEqual(sorted(os.listdir(self.paper)), before)
        anchor = tabulate.anchor_counts(tabulate.read_csv(self.paper / "anchor.csv"))
        self.assertEqual(sum(entry["reproduce"] for entry in anchor), 28)
        grid = tabulate.grid_table(tabulate.read_csv(self.paper / "grid.csv"),
                                   tabulate.read_csv(self.paper / "grid_summary.csv"))
        self.assertEqual(len(grid), 8)
        self.assertTrue(all(f"S_{h:g}" in grid[0] for h in bc.GRID_SECONDS))
        order = tabulate.order_table(tabulate.read_csv(self.paper / "mooncake_order_beyond_bit.csv"),
                                     "full")
        self.assertEqual(sum(1 for entry in order if entry["role"] == "h_star"
                             and entry["separated"] is True), 8)
        self.assertEqual(tabulate.TRACES[:4], bc.TRACES)


if __name__ == "__main__":
    unittest.main()
