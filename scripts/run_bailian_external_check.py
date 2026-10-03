#!/usr/bin/env python3
"""External-workload check on the Qwen-Bailian traces: the matched horizons transplanted, and a sufficing horizon within a fixed grid.

The pre-registration is `docs/bailian-external-check-plan.md` (commit
`553b026`, Addendum 1 in `2ac7c9d`); nothing here may change it. The fixed
surface and the arithmetic of the readings are
`persistent_kv_admission.bailiancheck`. One runner, four modes, each run once
into its own files, never overwriting another mode's:

* `anchor`: the Mooncake reproduction anchor, 70 replays on the conversation
  trace at 0.25% x 1 and 1% x 4 (all16: lru, label, label_binary_60,
  label_binary_600; leaf16: lru, label, the cell's label_binary_h*), with the
  three window collectors attached exactly as in `main`, each held to its
  published row: `avoided_prefill_tokens` and both digests where the row
  carries them, every published counter column where it does not
  (mechanism_control_001). Writes `anchor.csv` and `anchor_config.json` to the
  paper directory only when all 70 reproduce and every check passes.
* `smoke`: 16 replays at 512 tokens (seed 0, 1% x 1, all16, the four A arms on
  each Bailian trace) and, given the 16-token To-B file, the three C arms at
  that budget (3 replays), each in a fresh process. Writes seconds, peak RSS,
  counts and integrity-check outcomes only (`smoke.csv`, `smoke_config.json`):
  no utility, no avoided-token column. Evaluates the plan's stop conditions.
* `main`: A+B, 2,400 sampled replays (4 traces x 2 mechanisms x 6 cells x 5
  seeds x 10 arms) and the 48 heap references, refused unless the paper
  directory holds a passing anchor and a smoke verdict that lets the run
  start, both from the same execution sources. Every required check of the
  plan is a hard failure; a dead worker makes the run incomplete and nothing
  is published.
* `granularity`: control C on To-B at 16-token blocks at the same absolute
  bytes as the 512-token cells (90 replays, or 54 with `--reduced-seeds` when
  the smoke's verdict says so), compared with the main run's 512-token rows.

Replays: `run_two_tier` with L1 `lru`, the published hit rule, union closure,
sampled-16 eviction under the mechanism's eligibility, `measure_from_ms` the
trace's split, the Phase 0.98b attribution collector, the error-location
decision statistics at the trace's 600-second label horizon around the arm's
(absent) override, a read-only candidate counter, and three window collectors
(W, W1, W2) chained beside the attribution collector. Heap references are
`run_decision_population`'s heap call (`l2_eviction="heap"`, policies `lru`
and `offline_next_use`) with the window collectors as the only hook.

Workers are forked; each loads the trace its task names into a per-process
cache (one trace at a time; tasks are grouped by trace), and checks that the
trace's split, label horizon, end, requests and states are the ones the parent
recorded. Nothing is fitted; every arm reads the trace's future on purpose and
none is a policy.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import multiprocessing.connection
import resource
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from persistent_kv_admission import bailiancheck as bc
from persistent_kv_admission import classmix
from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import ALL16, LEAF16, MECHANISMS, horizon_arm
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.tailwindow import ChainedRequestHook
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

LEAF_MATCHED_SCRIPT = REPOSITORY / "scripts/run_leaf_matched_horizon.py"
PLAN_PATH = REPOSITORY / "docs/bailian-external-check-plan.md"


def _load_leaf_matched():
    """Import the leaf-matched horizon runner by path, unchanged, for the
    replay constants, provenance and reproduction helpers it and the runners
    it chains define (matched class order, horizon control, error location,
    mechanism control, Phase 0.97; one instance of each, so the capacity rule
    reads the working set this run sets). `main` is behind the usual guard, so
    nothing runs, and nothing is read at import."""
    spec = importlib.util.spec_from_file_location("run_leaf_matched_horizon", LEAF_MATCHED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rlm = _load_leaf_matched()
mco = rlm.mco
hc = rlm.hc
rel = rlm.rel
rmc = rlm.rmc
rdp = rlm.rdp

# --- the fixed surface ------------------------------------------------------------------------------

MODES = ("anchor", "smoke", "main", "granularity")
VARIANT = "main"
DEFAULT_WORKERS = 10
MAX_WORKERS = 12
DEFAULT_FINE_SMOKE_WORKERS = 1
PAPER_ROOT = REPOSITORY / "results/paper"
DEFAULT_PAPER_DIR = PAPER_ROOT / "bailian_external_check_001"
DEFAULT_GRANULARITY_DIR = PAPER_ROOT / "bailian_granularity_control_001"
assert rdp.L1_POLICY == "lru" and rdp.HIT_MODEL == "tree" and rdp.CLOSURE == "union"
# The files each mode writes into its paper directory; a mode refuses to start
# when any of its own exists and never touches another mode's.
ANCHOR_FILES = ("anchor.csv", "anchor_config.json")
SMOKE_FILES = ("smoke.csv", "smoke_config.json")
MAIN_TABLES = ("replay_seeds", "replay", "heap_references", "transplant", "grid", "grid_summary",
               "monotonicity", "random", "calibration", "order_beyond_bit",
               "mooncake_side_by_side", "mooncake_order_beyond_bit", "readings", "checks")
MAIN_FILES = tuple(f"{name}.csv" for name in MAIN_TABLES) + ("run_config.json", "README.md")
GRANULARITY_TABLES = ("replay_seeds", "replay", "granularity", "checks")
GRANULARITY_FILES = (tuple(f"{name}.csv" for name in GRANULARITY_TABLES)
                     + ("run_config.json", "README.md"))
# The anchor's published sources, anchored at the repository.
ANCHOR_REFERENCE_SOURCES = {
    "error_location_001": PAPER_ROOT / "error_location_001/replay_seeds.csv",
    "horizon_control_001": PAPER_ROOT / "horizon_control_001/replay_seeds.csv",
    "mechanism_control_001": PAPER_ROOT / "mechanism_control_001/replay_seeds.csv",
    "leaf_matched_horizon_001": PAPER_ROOT / "leaf_matched_horizon_001/replay_seeds.csv",
}
assert {source for sources in bc.ANCHOR_SOURCES.values() for source in sources} == set(
    ANCHOR_REFERENCE_SOURCES)
# Reading 6 and the Mooncake counterpart of reading 7: published tables, read only.
MOONCAKE_TRACES = ("conversation_trace", "toolagent_trace")
MOONCAKE_SOURCES = {
    "horizon_control_001/horizon.csv": PAPER_ROOT / "horizon_control_001/horizon.csv",
    "horizon_fill_001/fill.csv": PAPER_ROOT / "horizon_fill_001/fill.csv",
    "leaf_matched_horizon_001/horizon.csv": PAPER_ROOT / "leaf_matched_horizon_001/horizon.csv",
    "class_order_mix_001/differences.csv": PAPER_ROOT / "class_order_mix_001/differences.csv",
    "error_location_001/replay_seeds.csv": PAPER_ROOT / "error_location_001/replay_seeds.csv",
    "horizon_control_001/replay_seeds.csv": PAPER_ROOT / "horizon_control_001/replay_seeds.csv",
    "horizon_fill_001/replay_seeds.csv": PAPER_ROOT / "horizon_fill_001/replay_seeds.csv",
    "leaf_matched_horizon_001/replay_seeds.csv":
        PAPER_ROOT / "leaf_matched_horizon_001/replay_seeds.csv",
}
MOONCAKE_RANDOM_ARM = ("random_within_class_{h*} - evict_binary_{h*}_recency "
                       "(class_order_mix_001): admission by the frozen learned ranker")
BAILIAN_RANDOM_ARM = ("label_binary_random_{h*} - label_binary_{h*}: admission and eviction by "
                      "the exact bit, random vs recency order within the class")
# Identifiers every replay of a trace x cell x seed shares on the full window
# (`hc.REFERENCE_IDENTIFIERS`) and on every collected window.
IDENTIFIERS = hc.REFERENCE_IDENTIFIERS
HEAP_IDENTIFIERS = ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                    "l1_avoided_tokens", "measured_requests")
SUMMARY_WINDOWS = ("W", "full")
SHARED: dict[str, object] = {}
_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class Design:
    """What a run replays. `REGISTERED` is the plan's and the only one the
    command line can reach; a test passes a reduced one to the `run_*`
    functions directly."""

    traces: tuple = bc.TRACES
    cells: tuple = bc.CELLS
    seeds: tuple = bc.SEEDS
    reduced_seeds: tuple = bc.REDUCED_SEEDS
    mechanisms: tuple = bc.MECHANISM_NAMES
    anchor_cells: tuple = bc.ANCHOR_CELLS
    anchor_seeds: tuple = bc.SEEDS


REGISTERED = Design()


def _registered(design: Design) -> bool:
    return design == REGISTERED


# --- arguments ------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)

    def common(sub, default_paper: Path, workers: int):
        sub.add_argument("--run-dir", type=Path, required=True,
                         help="new run directory (raw replay stream and copies); refused if it "
                              "exists")
        sub.add_argument("--paper-dir", type=Path, default=default_paper,
                         help=f"publication directory (default "
                              f"{default_paper.relative_to(REPOSITORY)})")
        sub.add_argument("--workers", type=int, default=workers,
                         help=f"replay processes (default {workers}, at most {MAX_WORKERS})")

    anchor = modes.add_parser("anchor", help="the 70-replay Mooncake reproduction anchor")
    anchor.add_argument("trace", type=Path, help="the Mooncake conversation trace")
    common(anchor, DEFAULT_PAPER_DIR, DEFAULT_WORKERS)
    smoke = modes.add_parser("smoke", help="the 16 + 3 replay smoke (no utility written)")
    smoke.add_argument("traces", nargs="+", type=Path, help="the four 512-token Bailian traces")
    smoke.add_argument("--trace16", type=Path, default=None,
                       help="the 16-token To-B trace (the three C smoke replays)")
    smoke.add_argument("--workers16", type=int, default=DEFAULT_FINE_SMOKE_WORKERS,
                       help=f"concurrent 16-token smoke replays (default "
                            f"{DEFAULT_FINE_SMOKE_WORKERS}, at most {bc.FINE_WORKERS})")
    common(smoke, DEFAULT_PAPER_DIR, DEFAULT_WORKERS)
    main = modes.add_parser("main", help="A+B: 2,400 sampled replays and 48 heap references")
    main.add_argument("traces", nargs="+", type=Path, help="the four 512-token Bailian traces")
    common(main, DEFAULT_PAPER_DIR, DEFAULT_WORKERS)
    fine = modes.add_parser("granularity", help="control C on To-B at 16-token blocks")
    fine.add_argument("--trace16", type=Path, required=True, help="the 16-token To-B trace")
    fine.add_argument("--trace512", type=Path, required=True,
                      help="the 512-token To-B trace of the main run (the bytes of the cells)")
    fine.add_argument("--main-dir", type=Path, default=DEFAULT_PAPER_DIR,
                      help="the main run's paper directory (default "
                           f"{DEFAULT_PAPER_DIR.relative_to(REPOSITORY)})")
    fine.add_argument("--reduced-seeds", action="store_true",
                      help="seeds 0-2 (54 replays); required exactly when the smoke's verdict "
                           "reduces the control")
    common(fine, DEFAULT_GRANULARITY_DIR, bc.FINE_WORKERS)
    return parser.parse_args(argv)


def _existing(directory: Path, names) -> list[str]:
    return [name for name in names if (Path(directory) / name).exists()]


def validate_arguments(args) -> None:
    """Every refusal that needs no trace, reference or git call."""
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} exceeds the hard cap of {MAX_WORKERS}")
    if args.run_dir.exists():
        raise SystemExit(f"run directory {args.run_dir} exists; choose a new one")
    if args.run_dir.resolve() == args.paper_dir.resolve():
        raise SystemExit("--run-dir and --paper-dir must differ")
    own = {"anchor": ANCHOR_FILES, "smoke": SMOKE_FILES, "main": MAIN_FILES,
           "granularity": GRANULARITY_FILES}[args.mode]
    present = _existing(args.paper_dir, own)
    if present:
        raise SystemExit(f"{args.paper_dir} already holds this mode's {present}; nothing is "
                         "overwritten")
    if args.mode in ("smoke", "main") and len(args.traces) != len(set(args.traces)):
        raise SystemExit("a trace is given twice")
    if args.mode == "smoke" and not 0 < args.workers16 <= bc.FINE_WORKERS:
        raise SystemExit(f"--workers16 must be between 1 and {bc.FINE_WORKERS}")
    if args.mode == "granularity":
        if args.workers > bc.FINE_WORKERS:
            raise SystemExit(f"control C runs on at most {bc.FINE_WORKERS} workers")
        if args.paper_dir.resolve() == args.main_dir.resolve():
            raise SystemExit("control C is published in its own directory")


# --- provenance -------------------------------------------------------------------------------------


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository: the leaf-matched
    runner's (every `src` module and the runners it chains) and this script."""
    paths = set(rlm.execution_sources())
    paths.add(Path(__file__).resolve())
    return sorted(paths)


def source_manifest() -> dict[str, object]:
    files = {str(path.relative_to(REPOSITORY)): sha256_path(path) for path in execution_sources()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def sources_differing_from_head() -> list[str]:
    """Execution files whose working copy is not byte-identical to HEAD's."""
    differing = []
    for path in execution_sources():
        relative = str(path.relative_to(REPOSITORY))
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPOSITORY,
                                   capture_output=True)
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            differing.append(relative)
    return differing


_repository_path = mco._repository_path


def provenance() -> dict[str, object]:
    """A clean tree whose execution sources are HEAD's, or a refusal."""
    status = rmc._git("status", "--porcelain")
    if status:
        raise SystemExit(f"the working tree is not clean:\n{status}")
    differing = sources_differing_from_head()
    if differing:
        raise SystemExit(f"execution source differs from HEAD: {differing}")
    return {"code_commit": rmc._git("rev-parse", "HEAD"),
            "plan_commit": rmc._git("log", "-1", "--format=%H", "--",
                                    str(PLAN_PATH.relative_to(REPOSITORY))),
            "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
            "git_dirty_at_start": False, "source_manifest": source_manifest()}


def check_unmoved(start: dict) -> None:
    """The execution sources and HEAD did not move while the replays ran."""
    if source_manifest() != start["source_manifest"]:
        raise SystemExit("execution source changed during the run; nothing published")
    if rmc._git("rev-parse", "HEAD") != start["code_commit"]:
        raise SystemExit("HEAD changed during the run; nothing published")


def _hashes(paths) -> dict[str, str]:
    return {_repository_path(path): sha256_path(path) for path in paths}


# --- inputs -----------------------------------------------------------------------------------------


def manifest_path(path: Path) -> Path:
    """The converter's manifest beside a converted trace."""
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def read_manifest(path: Path, block_tokens: int) -> tuple[dict, list[str]]:
    """The converter's manifest of a Bailian trace, checked against the file:
    its own name, block size, output hash and upstream source."""
    path = Path(path)
    problems = []
    manifest_file = manifest_path(path)
    if not manifest_file.is_file():
        return {}, [f"{path.name}: no manifest {manifest_file.name}"]
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    digest = sha256_path(path)
    expected = {"output": path.name, "block_tokens": int(block_tokens), "output_sha256": digest,
                "source_block_tokens": bc.FINE_BLOCK_TOKENS, "timestamp_unit": "ms",
                "source": bc.UPSTREAM_SOURCES.get(path.stem)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            problems.append(f"{manifest_file.name}: {key} {manifest.get(key)!r} != {value!r}")
    return {"path": _repository_path(manifest_file), "sha256": sha256_path(manifest_file),
            "contents": manifest}, problems


def load_input(path: Path, block_tokens: int) -> dict[str, object]:
    """One trace read once in the parent: its identity, split, label horizon,
    windows and working set. The trace itself is not kept (workers load their
    own)."""
    path = Path(path)
    started = time.time()
    trace = load_mooncake_trace(path, block_tokens)
    horizon, split_ms, _ = horizon_for(trace)
    info = {"name": trace.name, "path": str(path.resolve()), "block_size": int(block_tokens),
            "sha256": sha256_path(path), "horizon_seconds": float(horizon),
            "split_ms": float(split_ms), "start_ms": float(trace.start_ms),
            "end_ms": float(trace.end_ms), "requests": len(trace.requests),
            "states": len(trace.states), "working_set_bytes": int(working_set_bytes(trace)),
            "load_seconds": time.time() - started}
    info["windows"] = bc.window_bounds(info["end_ms"], info["split_ms"])
    del trace
    gc.collect()
    return info


def load_inputs(paths, block_tokens: int, expected: tuple, bailian: bool
                ) -> tuple[dict[str, dict], list[str]]:
    """Every trace, by name, with its manifest when `bailian`; refusals for a
    name off the design, a label horizon other than 600 s or a bad manifest."""
    inputs, problems = {}, []
    for path in paths:
        info = load_input(path, block_tokens)
        name = info["name"]
        if name in inputs:
            problems.append(f"duplicate trace {name}")
        if name not in expected:
            problems.append(f"trace {name} is not one of {list(expected)}")
        if info["horizon_seconds"] != bc.LABEL_HORIZON_SECONDS:
            problems.append(f"{name}: label horizon {info['horizon_seconds']} s, the plan fixes "
                            f"{bc.LABEL_HORIZON_SECONDS} s")
        if bailian:
            info["manifest"], manifest_problems = read_manifest(path, block_tokens)
            problems += manifest_problems
        inputs[name] = info
        print(f"loaded {name} ({block_tokens}-token blocks): requests={info['requests']} "
              f"states={info['states']} working set={info['working_set_bytes'] / 2**30:.2f} GiB "
              f"split={info['split_ms']:.1f} ms primary window [split, "
              f"{info['windows']['primary_end_ms']:.1f}] ms "
              f"({info['windows']['primary_seconds']:.0f} s) sha256={info['sha256'][:12]} "
              f"({info['load_seconds']:.1f} s)", flush=True)
    if set(inputs) != set(expected):
        problems.append(f"traces {sorted(inputs)} do not match the design {sorted(expected)}")
    return inputs, problems


def _worker_spec(info: dict, path=None, block_size=None) -> dict:
    return {"path": str(path or info["path"]), "block_size": int(block_size or info["block_size"]),
            "split_ms": info["split_ms"], "horizon_seconds": info["horizon_seconds"],
            "end_ms": info["end_ms"], "requests": info["requests"],
            "states": None if block_size is not None else info["states"]}


def _trace_state(name: str) -> dict:
    """The worker's cache: the trace a task names, loaded once per process
    (one trace at a time), with its occurrence groups, checked against what
    the parent recorded."""
    spec = SHARED["inputs"][name]
    key = (spec["path"], int(spec["block_size"]))
    if _CACHE.get("key") != key:
        _CACHE.clear()
        gc.collect()
        started = time.time()
        trace = load_mooncake_trace(spec["path"], int(spec["block_size"]))
        horizon, split_ms, _ = horizon_for(trace)
        observed = (trace.name, float(horizon), float(split_ms), float(trace.end_ms),
                    len(trace.requests))
        expected = (name, spec["horizon_seconds"], spec["split_ms"], spec["end_ms"],
                    spec["requests"])
        if observed != expected:
            raise RuntimeError(f"worker trace {spec['path']}: {observed} != the parent's {expected}")
        if spec.get("states") is not None and len(trace.states) != spec["states"]:
            raise RuntimeError(f"worker trace {spec['path']}: {len(trace.states)} states != "
                               f"{spec['states']}")
        _CACHE.update(key=key, trace=trace, groups=_occurrence_groups(trace),
                      load_seconds=time.time() - started, states=len(trace.states),
                      working_set=None)
    return _CACHE


# --- the replays ------------------------------------------------------------------------------------


def _identity(name, fraction, multiplier, mechanism, arm, seed, block_size) -> dict:
    if mechanism == bc.HEAP:
        eligibility, width, family, parameter, h_star, part = "", "", "heap", "", "", ""
    else:
        eligibility, width = MECHANISMS[mechanism]
        family = bc.arm_family(arm)
        horizon = bc.arm_horizon(arm)
        parameter = "" if horizon is None else horizon
        h_star = bc.hstar(fraction, multiplier)
        part = bc.part(arm, fraction, multiplier)
    return {"trace": name, "block_tokens": block_size, "l1_fraction": fraction,
            "l2_multiplier": multiplier, "cell": rdp.cell_label(fraction, multiplier),
            "eligibility": eligibility, "width": width, "mechanism": mechanism, "arm": arm,
            "family": family, "part": part, "arm_parameter": parameter, "h_star": h_star,
            "seed": "" if seed is None else seed, "variant": VARIANT}


def replay_sampled(task) -> dict:
    """One sampled replay: the published replay call (attribution hooks, the
    error-location statistics at the label horizon around the arm's override,
    which no arm here has), a read-only candidate counter inside them, and the
    three window collectors chained beside the attribution collector."""
    name, fraction, multiplier, mechanism, arm, seed = task
    if mechanism not in bc.MECHANISM_NAMES:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if arm not in bc.cell_arms(fraction, multiplier):
        raise ValueError(f"{arm} is not an arm of cell {(fraction, multiplier)}")
    state = _trace_state(name)
    spec = SHARED["inputs"][name]
    trace, groups = state["trace"], state["groups"]
    split_ms, horizon = spec["split_ms"], spec["horizon_seconds"]
    eligibility, width = MECHANISMS[mechanism]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = bc.arm_setup(arm, trace, horizon, seed, mechanism)
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    windows = bc.window_collectors(trace, split_ms)
    statistics = DecisionStatistics(trace, horizon, split_ms)
    candidates = bc.CandidateCount()
    # Every recorder sees every decision with its final victim; the arm's own
    # answer (None: no arm here overrides) passes through unchanged.
    override = RecordingOverride(statistics, RecordingOverride(candidates, setup.override))
    request_hook = ChainedRequestHook(collector.on_request,
                                      *(windows[window].on_request
                                        for window in bc.COLLECTED_WINDOWS))
    started = time.time()
    result = run_two_tier(
        trace, rdp.L1_POLICY, l1_bytes, l2_bytes, hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=groups, measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=width, l2_seed=seed,
        l2_eligibility=eligibility, l2_arm=arm,
        l2_request_hook=request_hook, l2_removal_hook=collector,
        l2_override_hook=override, **setup.replay_arguments(),
    )
    seconds = time.time() - started
    # The partition and per-block identities of Phase 0.98b, and every counter
    # the replay keeps for itself, must agree with the attribution built beside it.
    collector.check_against(result)
    requested = max(result.requested_tokens, 1)
    row = _identity(name, fraction, multiplier, mechanism, arm, seed, trace.block_size)
    row.update({
        "l1_capacity_bytes": result.l1_capacity_bytes,
        "l2_capacity_bytes": result.l2_capacity_bytes,
        "requested_tokens": result.requested_tokens,
        "measured_requests": result.measured_requests,
        "l1_avoided_tokens": result.l1_avoided_tokens,
        "l2_avoided_tokens": result.l2_avoided_tokens,
        "avoided_prefill_tokens": result.avoided_prefill_tokens,
        "extra_avoided_tokens": result.avoided_prefill_tokens - result.l1_avoided_tokens,
        "l2_present_unusable_tokens": result.l2_present_unusable_tokens,
        "l2_present_unusable_blocks": result.l2_present_unusable_blocks,
        "l2_admissions": result.l2_admissions,
        "l2_rejections": result.l2_rejections,
        "l2_evictions": result.l2_evictions,
        "l2_decisions": result.l2_decisions,
        "l1_evictions": result.l1_evictions,
        "seconds": seconds,
        "worker_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF),
        "worker_pss_mib_end": rmc._proc_mib("/proc/self/smaps_rollup", "Pss"),
        "trace_load_seconds": state["load_seconds"],
        "counters_sha256": rel._counters_digest(result, collector),
        **collector.loss_row(),
        **collector.orphaning_row(),
    })
    row["extra_fraction_of_input"] = row["extra_avoided_tokens"] / requested
    for column, points in rmc.POINT_COLUMNS:
        row[points] = 100.0 * row[column] / requested
    charged = row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
    for category in ("rejected", "evicted"):
        row[f"absent_{category}_share_of_decision_absent"] = (
            row[f"absent_{category}_tokens"] / charged if charged else math.nan
        )
    row.update(statistics.row())
    stream = bc.random_stream(setup)
    row.update({"arm_builder": bc.arm_builder(arm, mechanism),
                "candidate_decisions_seen": candidates.decisions,
                "candidates_seen": candidates.candidates,
                "random_stream_seed": stream.stream_seed if stream is not None else "",
                "random_draws": stream.draws if stream is not None else "",
                "random_stream_separate": (bc.stream_is_separate(stream, candidates)
                                           if stream is not None else "")})
    counters = bc.window_counters(result, windows)
    row.update(bc.window_columns(counters))
    row["window_problems"] = "; ".join(bc.window_problems(counters))
    return row


def replay_heap(task) -> dict:
    """One heap reference: `run_decision_population`'s heap call, with the
    three window collectors as the only hook."""
    name, fraction, multiplier, mechanism, arm, seed = task
    if mechanism != bc.HEAP or arm not in bc.HEAP_POLICIES or seed is not None:
        raise ValueError(f"not a heap reference task: {task!r}")
    policy = bc.HEAP_POLICIES[arm]
    state = _trace_state(name)
    spec = SHARED["inputs"][name]
    trace, groups = state["trace"], state["groups"]
    split_ms = spec["split_ms"]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    windows = bc.window_collectors(trace, split_ms)
    started = time.time()
    result = run_two_tier(
        trace, rdp.L1_POLICY, l1_bytes, l2_bytes, policy, l2_eviction="heap", l2_arm=policy,
        hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE, occurrence_groups=groups,
        measure_from_ms=split_ms,
        l2_request_hook=ChainedRequestHook(*(windows[window].on_request
                                             for window in bc.COLLECTED_WINDOWS)),
    )
    seconds = time.time() - started
    row = _identity(name, fraction, multiplier, bc.HEAP, arm, None, trace.block_size)
    row.update({
        "heap_policy": policy, "offline_tiebreak": result.offline_tiebreak,
        "l2_eviction": result.l2_eviction,
        "l1_capacity_bytes": result.l1_capacity_bytes,
        "l2_capacity_bytes": result.l2_capacity_bytes,
        "requested_tokens": result.requested_tokens,
        "measured_requests": result.measured_requests,
        "l1_avoided_tokens": result.l1_avoided_tokens,
        "l2_avoided_tokens": result.l2_avoided_tokens,
        "avoided_prefill_tokens": result.avoided_prefill_tokens,
        "extra_avoided_tokens": result.avoided_prefill_tokens - result.l1_avoided_tokens,
        "l2_present_unusable_tokens": result.l2_present_unusable_tokens,
        "l2_present_unusable_blocks": result.l2_present_unusable_blocks,
        "l2_admissions": result.l2_admissions,
        "l2_rejections": result.l2_rejections,
        "l2_evictions": result.l2_evictions,
        "l1_evictions": result.l1_evictions,
        "seconds": seconds,
        "worker_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF),
        "worker_pss_mib_end": rmc._proc_mib("/proc/self/smaps_rollup", "Pss"),
        "trace_load_seconds": state["load_seconds"],
    })
    counters = bc.window_counters(result, windows)
    row.update(bc.window_columns(counters))
    row["window_problems"] = "; ".join(bc.window_problems(counters))
    return row


def _replay_worker(task) -> dict:
    return replay_heap(task) if task[3] == bc.HEAP else replay_sampled(task)


# --- the smoke row: seconds, memory, counts and integrity only ----------------------------------------

# Every column a smoke row may carry. Nothing from which a utility can be
# read: no avoided, hit, present, absent, extra or point column, no window
# counter, no digest of counters.
SMOKE_COLUMNS = (
    "trace", "block_tokens", "l1_fraction", "l2_multiplier", "cell", "mechanism", "arm", "seed",
    "l1_capacity_bytes", "l2_capacity_bytes", "measured_requests", "l2_decisions",
    "l2_admissions", "l2_rejections", "l2_evictions", "l1_evictions", "candidates_seen",
    "random_draws", "seconds", "worker_peak_rss_mib", "worker_pss_mib_end",
    "trace_load_seconds", "parent_rss_mib_at_fork", "identifiers_sha256",
    "check_attribution_identities", "check_no_unexplained_absent", "check_statistics",
    "check_no_override", "check_windows", "check_random_stream", "check_replay")
SMOKE_CHECKS = tuple(column for column in SMOKE_COLUMNS if column.startswith("check_"))
_UTILITY_MARKERS = ("avoided", "hit", "present", "absent_", "extra", "points", "unusable",
                    "loss", "utility", "U_", "counters_sha256", "orphan")
assert not any(marker in column for column in SMOKE_COLUMNS for marker in _UTILITY_MARKERS
               if column != "check_no_unexplained_absent")


def _identifier_values(row) -> dict:
    values = {field: int(row[field]) for field in IDENTIFIERS}
    for window in bc.COLLECTED_WINDOWS:
        for counter in bc.WINDOW_IDENTIFIERS:
            values[f"{window}_{counter}"] = int(row[f"{window}_{counter}"])
    return values


def random_problems(row) -> list[str]:
    """The random arm drew once per candidate of every decision, from its own
    stream seeded by the replay seed and the arm name; no other arm carries a
    stream."""
    where = _where(row)
    if row["family"] != "random":
        if any(row.get(column, "") not in ("", None)
               for column in ("random_stream_seed", "random_draws", "random_stream_separate")):
            return [f"{where}: an arm without a random stream carries one"]
        return []
    problems = []
    if int(row["random_draws"]) != int(row["candidates_seen"]):
        problems.append(f"{where}: {row['random_draws']} random draws != "
                        f"{row['candidates_seen']} candidates")
    if int(row["candidates_seen"]) <= 0:
        problems.append(f"{where}: no candidate scored")
    if int(row["random_stream_seed"]) != classmix.random_stream_seed(int(row["seed"]), row["arm"]):
        problems.append(f"{where}: random stream seed {row['random_stream_seed']} is not "
                        "random_stream_seed(seed, arm)")
    if str(row["random_stream_separate"]) != "True":
        problems.append(f"{where}: the random stream is not its own generator")
    return problems


def smoke_projection(row, parent_rss_mib: float = math.nan) -> dict:
    """The smoke's row: `SMOKE_COLUMNS` of a replay row, the per-replay
    integrity checks evaluated here, and a digest of the identifiers for the
    across-arm check. Nothing utility-bearing leaves the worker."""
    out = {column: row.get(column, "") for column in SMOKE_COLUMNS if column in row}
    out["parent_rss_mib_at_fork"] = parent_rss_mib
    out["identifiers_sha256"] = hashlib.sha256(json.dumps(
        _identifier_values(row), sort_keys=True).encode()).hexdigest()
    out["check_attribution_identities"] = True     # `check_against` raised otherwise
    out["check_no_unexplained_absent"] = int(row["absent_unexplained_tokens"]) == 0
    out["check_statistics"] = not hc.check_statistics([row])
    out["check_no_override"] = int(row["overridden_decisions_seen"]) == 0
    out["check_windows"] = row["window_problems"] == ""
    out["check_random_stream"] = not random_problems(row)
    out["check_replay"] = all(out[column] for column in SMOKE_CHECKS if column != "check_replay")
    return {column: out.get(column, "") for column in SMOKE_COLUMNS}


def _smoke_worker(task) -> dict:
    return smoke_projection(replay_sampled(task), SHARED.get("parent_rss_mib", math.nan))


# --- running tasks ----------------------------------------------------------------------------------


class IncompleteRun(RuntimeError):
    """A replay failed or a worker died: the run is incomplete."""


def _progress(done: int, total: int, started: float, row: dict) -> None:
    if done % 10 == 0 or done == total:
        print(f"  [{done}/{total}] {time.time() - started:.0f}s last={row['trace']}/"
              f"{row['mechanism']}/{row['cell']}/{row['arm']}/s{row['seed']} "
              f"({float(row['seconds']):.0f}s, worker peak "
              f"{float(row['worker_peak_rss_mib']):.0f} MiB)", flush=True)


def _pool_rows(tasks, workers: int, worker, raw) -> list[dict]:
    """Long-lived forked workers; a raised replay or a dead worker aborts."""
    rows: list[dict] = []
    started = time.time()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"))
    try:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                row = future.result()
            except BrokenProcessPool as error:
                raise IncompleteRun(f"a worker died ({error}); the run is incomplete") from error
            except Exception as error:
                raise IncompleteRun(f"replay {futures[future]} failed: {error!r}") from error
            rows.append(row)
            raw.write(json.dumps(row, default=str) + "\n")
            raw.flush()
            _progress(len(rows), len(tasks), started, row)
    except BaseException:
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    executor.shutdown(wait=True)
    return rows


def _isolated_child(connection, worker, task) -> None:
    try:
        connection.send(("ok", worker(task)))
    except BaseException as error:  # reported to the parent, never swallowed
        connection.send(("error", f"{error!r}\n{traceback.format_exc()}"))
    finally:
        connection.close()


def _isolated_rows(tasks, workers: int, worker, raw) -> list[dict]:
    """One fresh forked process per task, at most `workers` at once, so that
    each replay's peak RSS is its own."""
    context = mp.get_context("fork")
    pending = list(tasks)
    running: dict = {}
    rows: list[dict] = []
    started = time.time()
    try:
        while pending or running:
            while pending and len(running) < workers:
                task = pending.pop(0)
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_isolated_child, args=(send, worker, task))
                process.start()
                send.close()
                running[receive] = (process, task)
            for receive in multiprocessing.connection.wait(list(running)):
                process, task = running.pop(receive)
                try:
                    status, payload = receive.recv()
                except EOFError:
                    process.join()
                    raise IncompleteRun(f"the process of {task} died (exit code "
                                        f"{process.exitcode}); the run is incomplete") from None
                finally:
                    receive.close()
                process.join()
                if status != "ok":
                    raise IncompleteRun(f"replay {task} failed: {payload}")
                rows.append(payload)
                raw.write(json.dumps(payload, default=str) + "\n")
                raw.flush()
                _progress(len(rows), len(tasks), started, payload)
    except BaseException:
        for process, _ in running.values():
            if process.is_alive():
                process.terminate()
            process.join()
        raise
    return rows


def run_tasks(tasks, workers: int, worker, raw, isolated: bool = False) -> list[dict]:
    workers = max(1, min(int(workers), len(tasks)))
    return (_isolated_rows if isolated else _pool_rows)(tasks, workers, worker, raw)


def _ordered(tasks, inputs_order) -> list[tuple]:
    """Tasks grouped by trace (each worker then loads each trace once), the
    sampled replays before the heap references, the larger L1 first. The order
    only schedules work; every table is sorted before it is written."""
    position = {name: index for index, name in enumerate(inputs_order)}
    return sorted(tasks, key=lambda task: (position[task[0]], task[3] == bc.HEAP, -task[1],
                                           -task[2], str(task[3]), task[4],
                                           -1 if task[5] is None else task[5]))


# --- checks -----------------------------------------------------------------------------------------


def _where(row) -> str:
    return (f"{row['trace']}/{row.get('block_tokens', '')}/{row['mechanism']}/{row['cell']}/"
            f"{row['arm']}/s{row['seed']}")


def _task_key(row) -> tuple:
    seed = None if row["seed"] in ("", None) else int(row["seed"])
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
            row["mechanism"], row["arm"], seed)


def check_complete(rows, tasks) -> list[str]:
    """Exactly one row per task, and no other."""
    seen: dict[tuple, int] = defaultdict(int)
    for row in rows:
        seen[_task_key(row)] += 1
    expected = {(task[0], float(task[1]), float(task[2]), task[3], task[4], task[5])
                for task in tasks}
    problems = [f"{key}: {count} rows" for key, count in sorted(seen.items(), key=str)
                if count != 1]
    problems += [f"{key}: missing" for key in sorted(expected - set(seen), key=str)]
    problems += [f"{key}: not a task of the run" for key in sorted(set(seen) - expected, key=str)]
    return problems


def check_identifiers(rows) -> tuple[int, list[str]]:
    """Every arm of a trace x cell x seed (both mechanisms) shares the
    identifiers on the full window and the window identifiers on W, W1 and W2;
    each heap reference shares them (those it carries) with every seed of its
    cell. Returns the groups and the problems."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    heap: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        cell = (row["trace"], row.get("block_tokens", ""), float(row["l1_fraction"]),
                float(row["l2_multiplier"]))
        if row["mechanism"] == bc.HEAP:
            heap[cell].append(row)
        else:
            groups[cell + (int(row["seed"]),)].append(row)
    problems = []
    for key, members in sorted(groups.items(), key=str):
        for field_name, value in _identifier_values(members[0]).items():
            values = sorted({_identifier_values(member)[field_name] for member in members})
            if len(values) != 1:
                problems.append(f"{key}: {field_name} takes {values} across the arms")
    for cell, references in sorted(heap.items(), key=str):
        seeds = [members for key, members in groups.items() if key[:4] == cell]
        if not seeds:
            problems.append(f"{cell}: heap reference without sampled replays")
        for reference in references:
            for members in seeds:
                for name in HEAP_IDENTIFIERS + tuple(
                        f"{window}_{counter}" for window in bc.COLLECTED_WINDOWS
                        for counter in bc.WINDOW_IDENTIFIERS):
                    if int(reference[name]) != int(members[0][name]):
                        problems.append(f"{_where(reference)}: {name} {reference[name]} != "
                                        f"{members[0][name]} of {_where(members[0])}")
    return len(groups), problems


def check_windows(rows) -> list[str]:
    """W at most full and W1 + W2 = W in every replay (computed in the worker)."""
    return [f"{_where(row)}: {row['window_problems']}" for row in rows if row["window_problems"]]


def check_mechanism(rows) -> list[str]:
    """Every sampled replay ran under its mechanism's eligibility and width,
    as an arm of its cell at its h*, built by the plan's builder."""
    problems = []
    for row in rows:
        if row["mechanism"] == bc.HEAP:
            if row["arm"] not in bc.HEAP_POLICIES or row["heap_policy"] != bc.HEAP_POLICIES[
                    row["arm"]] or row["l2_eviction"] != "heap":
                problems.append(f"{_where(row)}: not a heap reference")
            continue
        cell = (float(row["l1_fraction"]), float(row["l2_multiplier"]))
        if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
            problems.append(f"{_where(row)}: {row['eligibility']}/{row['width']}")
        if row["arm"] not in bc.cell_arms(*cell) or float(row["h_star"]) != bc.hstar(*cell):
            problems.append(f"{_where(row)}: not an arm of its cell at h* {bc.hstar(*cell):g} s")
        if row["arm_builder"] != bc.arm_builder(row["arm"], row["mechanism"]):
            problems.append(f"{_where(row)}: built by {row['arm_builder']}")
    return problems


def check_statistics(rows) -> list[str]:
    """The horizon control's statistics check (every decision seen with its
    final victim, a decision in the window, `label`'s identities) and no
    override anywhere: no arm of this plan has one."""
    sampled = [row for row in rows if row["mechanism"] != bc.HEAP]
    problems = hc.check_statistics(sampled)
    for row in sampled:
        if int(row["overridden_decisions_seen"]) != 0:
            problems.append(f"{_where(row)}: {row['overridden_decisions_seen']} overridden "
                            "decisions; no arm of the plan overrides")
        if int(row["candidate_decisions_seen"]) != int(row["l2_decisions"]):
            problems.append(f"{_where(row)}: the candidate counter saw "
                            f"{row['candidate_decisions_seen']} of {row['l2_decisions']} decisions")
    return problems


def check_leaf_closure(rows) -> list[str]:
    """Under leaf16 no replay has a present-but-unusable token, on any window."""
    problems = []
    for row in rows:
        if row["mechanism"] != LEAF16:
            continue
        for window in bc.WINDOWS:
            if int(row[f"{window}_l2_present_unusable_tokens"]) or int(
                    row[f"{window}_l2_present_unusable_blocks"]):
                problems.append(f"{_where(row)}: present-but-unusable on {window}")
    return problems


def check_unexplained(rows) -> list[str]:
    """No absent token is unexplained (the attribution's own identities were
    checked per replay by `check_against`)."""
    return [f"{_where(row)}: {row['absent_unexplained_tokens']} unexplained absent tokens"
            for row in rows if row["mechanism"] != bc.HEAP
            and int(row["absent_unexplained_tokens"])]


def check_random(rows) -> list[str]:
    """`random_problems` of every sampled replay, and the random arm's
    identifiers equal to its recency arm's (`label_binary_h*`) on every window."""
    sampled = [row for row in rows if row["mechanism"] != bc.HEAP]
    problems = [problem for row in sampled for problem in random_problems(row)]
    index = {_task_key(row): row for row in sampled}
    for row in sampled:
        if row["family"] != "random":
            continue
        key = _task_key(row)
        rung = index.get(key[:4] + (horizon_arm(float(row["arm_parameter"])),) + key[5:])
        if rung is None:
            continue
        if _identifier_values(row) != _identifier_values(rung):
            problems.append(f"{_where(row)}: identifiers differ from {_where(rung)}")
    return problems


def run_checks(rows, tasks) -> list[dict]:
    """Every required check, as rows of `checks.csv` with their problems."""
    out = []

    def add(name, scope, checked, problems):
        out.append({"check": name, "scope": scope, "checked": checked,
                    "problems": len(problems), "passes": not problems,
                    "detail": "; ".join(problems[:20])})

    sampled = [row for row in rows if row["mechanism"] != bc.HEAP]
    add("complete", "one row per task", len(tasks), check_complete(rows, tasks))
    groups, problems = check_identifiers(rows)
    add("identifiers", "every arm of a trace x cell x seed, every window; heap with its cell",
        groups, problems)
    add("windows", "W <= full and W1 + W2 = W, every replay", len(rows), check_windows(rows))
    add("mechanism", "eligibility, width, arm, h*, builder", len(rows), check_mechanism(rows))
    add("statistics", "every decision seen with its final victim; label identities; no override",
        len(sampled), check_statistics(rows))
    add("leaf16_closure", "no present-but-unusable token under leaf16",
        sum(1 for row in rows if row["mechanism"] == LEAF16), check_leaf_closure(rows))
    add("phase098b", "attribution identities (per replay) and no unexplained absent token",
        len(sampled), check_unexplained(rows))
    add("random_stream", "draws = candidates; own stream; recency-arm identifiers",
        sum(1 for row in sampled if row["family"] == "random"), check_random(rows))
    return out


def _report(kind: str, lines, limit: int = 20) -> None:
    for line in list(lines)[:limit]:
        print(f"    {kind} {line}", flush=True)


def _print_checks(checks) -> list[str]:
    failures = []
    for check in checks:
        print(f"  check {check['check']} ({check['scope']}; {check['checked']} checked): "
              f"{'pass' if check['passes'] else 'FAIL'}", flush=True)
        if not check["passes"]:
            _report(check["check"].upper(), check["detail"].split("; "))
            failures.append(check["check"])
    return failures


# --- the anchor -------------------------------------------------------------------------------------


def load_anchor_references(design: Design = REGISTERED, paths=None
                           ) -> tuple[dict[tuple, dict[str, dict]], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> {source: the
    published row as written}` for every anchor replay, each from its own
    source(s) only (`bc.ANCHOR_SOURCES`; the mechanism control's arm column is
    `rung`), with its mechanism's eligibility and width, exactly once. `paths`
    overrides the source files (tests)."""
    paths = dict(ANCHOR_REFERENCE_SOURCES, **(paths or {}))
    wanted = {(task[3], task[4], task[0], float(task[1]), float(task[2]), task[5])
              for task in bc.anchor_replays(design.anchor_seeds, design.anchor_cells)}
    published: dict[tuple, dict[str, dict]] = defaultdict(dict)
    problems = []
    for source, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                arm = row["arm"] if "arm" in row else row["rung"]
                if source not in bc.ANCHOR_SOURCES.get((row["mechanism"], arm), ()):
                    continue
                if row.get("variant", VARIANT) != VARIANT:
                    continue
                key = (row["mechanism"], arm, row["trace"], float(row["l1_fraction"]),
                       float(row["l2_multiplier"]), int(row["seed"]))
                if key not in wanted:
                    continue
                if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
                    problems.append(f"{source} {key}: {row['eligibility']}/{row['width']}")
                if source in published[key]:
                    problems.append(f"duplicate published {source} row {key}")
                    continue
                published[key][source] = dict(row)
    for key in sorted(wanted, key=str):
        for source in bc.ANCHOR_SOURCES[key[:2]]:
            if source not in published.get(key, {}):
                problems.append(f"published {source} row {key} missing")
    # The plan's order of the sources, the first the primary, whatever the
    # order the files were read in.
    ordered = {key: {source: sources[source] for source in bc.ANCHOR_SOURCES[key[:2]]
                     if source in sources}
               for key, sources in published.items()}
    return ordered, problems


def compare_with_published(row, sources: dict[str, dict]) -> dict:
    """One anchor replay against its published row(s): `avoided_prefill_tokens`
    exactly; each digest a published row carries; every published counter
    column of a row that carries no counter digest (then every column of
    `REQUIRED_COUNTER_COLUMNS` must be among them); the identifiers the row
    publishes. Returns the anchor table's row."""
    names = list(sources)
    primary = sources[names[0]]
    same_tokens = all(int(row["avoided_prefill_tokens"]) == int(published["avoided_prefill_tokens"])
                      for published in sources.values())
    entry = {"trace": row["trace"], "l1_fraction": row["l1_fraction"],
             "l2_multiplier": row["l2_multiplier"], "cell": row["cell"],
             "mechanism": row["mechanism"], "arm": row["arm"], "seed": row["seed"],
             "reference_sources": ";".join(names),
             "avoided_prefill_tokens": row["avoided_prefill_tokens"],
             "published_avoided_prefill_tokens": primary["avoided_prefill_tokens"],
             "same_avoided_prefill_tokens": same_tokens}
    compared = []
    digest_same = []
    for column in rlm.DIGEST_COLUMNS:
        published = [(name, values[column]) for name, values in sources.items()
                     if values.get(column, "") not in ("", None)]
        entry[column] = row.get(column, "")
        entry[f"published_{column}"] = ";".join(value for _, value in published)
        if published:
            same = all(str(row.get(column, "")) == value for _, value in published)
            entry[f"same_{column}"] = same
            digest_same.append(same)
            compared.append(f"{column} ({', '.join(name for name, _ in published)})")
        else:
            entry[f"same_{column}"] = ""
    differing, unpublished, column_sources, columns_compared = [], [], [], 0
    for name, published in sources.items():
        if published.get("counters_sha256", "") not in ("", None):
            continue
        counters = {column: value for column, value in published.items()
                    if column is not None and column not in rlm.REPRODUCTION_SKIP
                    and column not in rlm.DIGEST_COLUMNS}
        columns_compared += len(counters)
        column_sources.append(name)
        differing += [f"{name}:{column}" for column, value in sorted(counters.items())
                      if column not in row or not rlm._same_published_value(row[column], value)]
        unpublished += [f"{name}:{column}" for column in rlm.REQUIRED_COUNTER_COLUMNS
                        if column not in counters]
        compared.append(f"{len(counters)} counter columns ({name})")
    identifiers_differing = []
    identifiers_compared = 0
    for name, published in sources.items():
        for column in IDENTIFIERS:
            if published.get(column, "") in ("", None):
                continue
            identifiers_compared += 1
            if int(row[column]) != int(published[column]):
                identifiers_differing.append(f"{name}:{column}")
    entry.update({"counter_columns_source": ";".join(column_sources),
                  "counter_columns_compared": columns_compared,
                  "counter_columns_differing": ";".join(differing),
                  "required_counter_columns_unpublished": ";".join(unpublished),
                  "identifiers_compared": identifiers_compared,
                  "identifiers_differing": ";".join(identifiers_differing),
                  "compared": "avoided_prefill_tokens; " + "; ".join(compared),
                  "windows_consistent": row["window_problems"] == ""})
    entry["reproduces"] = bool(same_tokens and all(digest_same) and (digest_same or column_sources)
                               and not differing and not unpublished
                               and not identifiers_differing)
    entry["seconds"] = row["seconds"]
    entry["worker_peak_rss_mib"] = row["worker_peak_rss_mib"]
    return entry


def check_anchor(rows, published) -> tuple[list[dict], list[str]]:
    """Every anchor replay against its published row(s); one table row each."""
    table, problems = [], []
    for row in rows:
        key = (row["mechanism"], row["arm"], row["trace"], float(row["l1_fraction"]),
               float(row["l2_multiplier"]), int(row["seed"]))
        sources = published.get(key)
        if not sources:
            problems.append(f"{_where(row)}: no published row")
            continue
        entry = compare_with_published(row, sources)
        table.append(entry)
        if not entry["reproduces"]:
            reasons = [column for column in ("same_avoided_prefill_tokens", "same_counters_sha256",
                                             "same_decision_sha256") if entry[column] is False]
            for column in ("counter_columns_differing", "required_counter_columns_unpublished",
                           "identifiers_differing"):
                if entry[column]:
                    reasons.append(f"{column} {entry[column]}")
            problems.append(f"{_where(row)} vs {entry['reference_sources']}: differs in "
                            + "; ".join(reasons))
    table.sort(key=lambda entry: (entry["l1_fraction"], entry["l2_multiplier"],
                                  bc.MECHANISM_NAMES.index(entry["mechanism"]), entry["arm"],
                                  entry["seed"]))
    return table, problems


def anchor_gate(paper_dir: Path, manifest: dict, design: Design = REGISTERED
                ) -> tuple[dict, list[str]]:
    """The anchor a Bailian replay needs: `anchor.csv` in the paper directory
    with every planned replay reproduced, written by the same execution
    sources (`anchor_config.json`)."""
    table_path, config_path = Path(paper_dir) / "anchor.csv", Path(paper_dir) / "anchor_config.json"
    if not table_path.is_file() or not config_path.is_file():
        return {}, [f"no anchor in {paper_dir} (run the anchor mode first)"]
    with table_path.open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    problems = []
    expected = {(task[3], task[4], float(task[1]), float(task[2]), task[5])
                for task in bc.anchor_replays(design.anchor_seeds, design.anchor_cells)}
    found = [(row["mechanism"], row["arm"], float(row["l1_fraction"]),
              float(row["l2_multiplier"]), int(row["seed"])) for row in table]
    if len(found) != len(expected) or set(found) != expected:
        problems.append(f"anchor.csv covers {len(found)} replays, not the plan's {len(expected)}")
    failing = [row for row in table if row["reproduces"] != "True"]
    if failing:
        problems.append(f"{len(failing)} anchor replays do not reproduce")
    if config.get("mode") != "anchor" or config.get("passed") is not True:
        problems.append("anchor_config.json does not record a passing anchor")
    if config.get("source_manifest", {}).get("sha256") != manifest["sha256"]:
        problems.append("the anchor ran from other execution sources")
    return {"paths": _hashes([table_path, config_path]), "replays": len(table),
            "reproduced": len(table) - len(failing),
            "code_commit": config.get("code_commit")}, problems


def run_anchor(args, design: Design = REGISTERED) -> None:
    clock = time.time()
    validate_arguments(args)
    start = provenance()
    inputs, problems = load_inputs([args.trace], bc.BLOCK_TOKENS, (bc.ANCHOR_TRACE,),
                                   bailian=False)
    published, reference_problems = load_anchor_references(design)
    if problems or reference_problems:
        _report("REFERENCE", problems + reference_problems)
        raise SystemExit("the anchor's inputs are not what the plan names; nothing run")
    references = _hashes(ANCHOR_REFERENCE_SOURCES.values())
    tasks = _ordered(bc.anchor_replays(design.anchor_seeds, design.anchor_cells), [bc.ANCHOR_TRACE])
    if _registered(design):
        assert len(tasks) == 70
    rdp._SHARED.update(working_set={name: info["working_set_bytes"]
                                    for name, info in inputs.items()})
    SHARED.update(inputs={name: _worker_spec(info) for name, info in inputs.items()})
    args.run_dir.mkdir(parents=True)
    print(f"anchor: {len(tasks)} replays on {min(args.workers, len(tasks))} workers", flush=True)
    with (args.run_dir / "raw_replays.jsonl").open("w", encoding="utf-8") as raw:
        try:
            rows = run_tasks(tasks, args.workers, _replay_worker, raw)
        except IncompleteRun as error:
            raise SystemExit(f"anchor incomplete, nothing published: {error}") from error
    table, anchor_problems = check_anchor(rows, published)
    checks = run_checks(rows, tasks)
    checks.insert(0, {"check": "reproduction", "scope": "every anchor replay vs its published row",
                      "checked": len(rows), "problems": len(anchor_problems),
                      "passes": not anchor_problems and len(table) == len(tasks),
                      "detail": "; ".join(anchor_problems[:20])})
    reproduced = sum(1 for entry in table if entry["reproduces"])
    print(f"  reproduction: {reproduced} of {len(tasks)} replays reproduce their published rows",
          flush=True)
    failures = _print_checks(checks)
    rdp._write(args.run_dir / "anchor.csv", table)
    check_unmoved(start)
    config = {"mode": "anchor", **start, "passed": not failures,
              "trace_files": {name: {"path": info["path"], "sha256": info["sha256"]}
                              for name, info in inputs.items()},
              "inputs": inputs, "references": references,
              "anchor_sources": {f"{m}/{a}": list(s) for (m, a), s in bc.ANCHOR_SOURCES.items()},
              "cells": [list(cell) for cell in design.anchor_cells],
              "seeds": list(design.anchor_seeds), "replays": len(rows),
              "reproduced": reproduced, "checks": checks,
              "compared": "avoided_prefill_tokens and each digest the published row carries; "
                          "every published counter column where it carries no counter digest "
                          "(skipped: " + ", ".join(sorted(rlm.REPRODUCTION_SKIP)) + ")",
              "windows": "the three window collectors attached exactly as in main",
              "workers": args.workers, "wall_seconds": time.time() - clock}
    (args.run_dir / "anchor_config.json").write_text(json.dumps(config, indent=2, default=str)
                                                     + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("the anchor does not reproduce the published rows (" + ", ".join(failures)
                         + "); nothing published, no Bailian replay may run")
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    present = _existing(args.paper_dir, ANCHOR_FILES)
    if present:
        raise SystemExit(f"{present} appeared in {args.paper_dir} during the run")
    rdp._write(args.paper_dir / "anchor.csv", table)
    (args.paper_dir / "anchor_config.json").write_text(json.dumps(config, indent=2, default=str)
                                                       + "\n", encoding="utf-8")
    print(f"anchor passed: {reproduced}/{len(tasks)} reproduced; written to {args.paper_dir} "
          f"({time.time() - clock:.0f}s)", flush=True)


# --- the smoke --------------------------------------------------------------------------------------


def fine_input(path16: Path, info512: dict) -> tuple[dict, list[str]]:
    """The 16-token To-B file, read by the parent only for its identity: the
    converter's manifest (16-token blocks, the To-B upstream source with the
    512-token file's source hash). Split, horizon and end are the 512-token
    file's (the same timestamps); each worker checks its loaded trace has them."""
    path16 = Path(path16)
    problems = []
    if path16.stem != bc.GRANULARITY_TRACE:
        problems.append(f"{path16.name}: the 16-token file must be {bc.GRANULARITY_TRACE}.jsonl")
    manifest, manifest_problems = read_manifest(path16, bc.FINE_BLOCK_TOKENS)
    problems += manifest_problems
    source512 = info512.get("manifest", {}).get("contents", {}).get("source_sha256")
    if manifest and manifest["contents"].get("source_sha256") != source512:
        problems.append("the 16-token file is not converted from the 512-token file's upstream "
                        "source")
    info = {"name": bc.GRANULARITY_TRACE, "path": str(path16.resolve()),
            "block_size": bc.FINE_BLOCK_TOKENS, "sha256": sha256_path(path16),
            "manifest": manifest, "horizon_seconds": info512["horizon_seconds"],
            "split_ms": info512["split_ms"], "end_ms": info512["end_ms"],
            "start_ms": info512["start_ms"], "requests": info512["requests"], "states": None,
            "windows": info512["windows"],
            "capacity_base_bytes": info512["working_set_bytes"],
            "capacity_base": "the 512-token To-B working set (same absolute bytes)"}
    return info, problems


def smoke_gate(paper_dir: Path, manifest: dict) -> tuple[dict, list[str]]:
    """The smoke verdict `main` needs: the 512-token smoke let the run start,
    from the same execution sources."""
    config_path = Path(paper_dir) / "smoke_config.json"
    table_path = Path(paper_dir) / "smoke.csv"
    if not config_path.is_file() or not table_path.is_file():
        return {}, [f"no smoke in {paper_dir} (run the smoke mode after the anchor)"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    problems = []
    if config.get("mode") != "smoke" or config.get("verdict_512", {}).get("start") is not True:
        problems.append("the smoke's verdict does not let the run start (stop condition or "
                        "integrity failure)")
    if config.get("source_manifest", {}).get("sha256") != manifest["sha256"]:
        problems.append("the smoke ran from other execution sources")
    return {"paths": _hashes([table_path, config_path]), "config": config}, problems


def _smoke_identifier_digests(rows) -> dict[tuple, set]:
    groups: dict[tuple, set] = defaultdict(set)
    for row in rows:
        groups[(row["trace"], row["block_tokens"])].add(row["identifiers_sha256"])
    return groups


def smoke_identifier_groups_differing(rows) -> set[tuple]:
    """(trace, block tokens) whose arms do not share one identifier digest."""
    return {key for key, values in _smoke_identifier_digests(rows).items() if len(values) != 1}


def check_smoke_identifiers(rows) -> list[str]:
    """The arms of each trace (and granularity) share the identifier digest."""
    return [f"{key}: {len(values)} identifier digests across the arms"
            for key, values in sorted(_smoke_identifier_digests(rows).items(), key=str)
            if len(values) != 1]


def run_smoke(args, design: Design = REGISTERED) -> None:
    clock = time.time()
    validate_arguments(args)
    start = provenance()
    anchor, gate_problems = anchor_gate(args.paper_dir, start["source_manifest"], design)
    if gate_problems:
        _report("ANCHOR", gate_problems)
        raise SystemExit("no passing anchor from these sources; no Bailian replay may run")
    inputs, problems = load_inputs(args.traces, bc.BLOCK_TOKENS, design.traces, bailian=True)
    fine = None
    if args.trace16 is not None and bc.GRANULARITY_TRACE in inputs:
        fine, fine_problems = fine_input(args.trace16, inputs[bc.GRANULARITY_TRACE])
        problems += fine_problems
    if problems:
        _report("INPUT", problems)
        raise SystemExit("the smoke's inputs are not what the plan names; nothing run")
    args.run_dir.mkdir(parents=True)
    rows: list[dict] = []
    raw_path = args.run_dir / "raw_smoke.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        try:
            rdp._SHARED.update(working_set={name: info["working_set_bytes"]
                                            for name, info in inputs.items()})
            SHARED.update(inputs={name: _worker_spec(info) for name, info in inputs.items()},
                          parent_rss_mib=rmc._proc_mib("/proc/self/status", "VmRSS"))
            tasks = bc.smoke_replays(design.traces)
            print(f"smoke at 512 tokens: {len(tasks)} replays, one fresh process each, at most "
                  f"{args.workers} at once", flush=True)
            rows += run_tasks(tasks, args.workers, _smoke_worker, raw, isolated=True)
            if fine is not None:
                gc.collect()
                rdp._SHARED.update(working_set={bc.GRANULARITY_TRACE: fine["capacity_base_bytes"]})
                SHARED.update(inputs={bc.GRANULARITY_TRACE: _worker_spec(
                    fine, fine["path"], bc.FINE_BLOCK_TOKENS)},
                    parent_rss_mib=rmc._proc_mib("/proc/self/status", "VmRSS"))
                tasks = bc.smoke_fine_replays()
                print(f"smoke at 16 tokens: {len(tasks)} replays at the 512-token bytes, one "
                      f"fresh process each, at most {args.workers16} at once", flush=True)
                rows += run_tasks(tasks, args.workers16, _smoke_worker, raw, isolated=True)
        except IncompleteRun as error:
            raise SystemExit(f"smoke incomplete: {error}") from error
    identifier_problems = check_smoke_identifiers(rows)
    differing = smoke_identifier_groups_differing(rows)
    for row in rows:
        row["check_identifiers_across_arms"] = (row["trace"], row["block_tokens"]) not in differing
    coarse = [row for row in rows if int(row["block_tokens"]) == bc.BLOCK_TOKENS]
    fine_rows = [row for row in rows if int(row["block_tokens"]) == bc.FINE_BLOCK_TOKENS]

    def integrity(selection):
        return bool(selection) and all(row["check_replay"] and row["check_identifiers_across_arms"]
                                       for row in selection)

    verdict_512 = bc.smoke_verdict([row["seconds"] for row in coarse],
                                   [row["worker_peak_rss_mib"] for row in coarse],
                                   integrity(coarse))
    verdict_16 = (bc.fine_smoke_verdict([row["seconds"] for row in fine_rows],
                                        [row["worker_peak_rss_mib"] for row in fine_rows],
                                        integrity(fine_rows)) if fine is not None else None)
    rows.sort(key=lambda row: (-int(row["block_tokens"]), row["trace"], row["arm"]))
    for row in rows:
        print(f"  {row['trace']:24s} {row['block_tokens']:>3} {row['arm']:26s} "
              f"{float(row['seconds']):8.1f} s {float(row['worker_peak_rss_mib']):8.0f} MiB "
              f"decisions={row['l2_decisions']} checks={'pass' if row['check_replay'] and row['check_identifiers_across_arms'] else 'FAIL'}",
              flush=True)
    print(f"  512-token verdict: max {verdict_512['max_seconds']:.1f} s (limit "
          f"{bc.SMOKE_MAX_SECONDS:.0f}), max {verdict_512['max_rss_mib']:.0f} MiB (limit "
          f"{bc.SMOKE_MAX_RSS_MIB:.0f}), integrity {'ok' if verdict_512['integrity_ok'] else 'FAILED'}"
          f" -> {'START the run' if verdict_512['start'] else 'DO NOT START; the plan is re-issued'}",
          flush=True)
    if verdict_16 is not None:
        print(f"  16-token verdict: max {verdict_16['max_seconds']:.1f} s, max "
              f"{verdict_16['max_rss_mib']:.0f} MiB -> seeds {verdict_16['seeds']}, at most "
              f"{verdict_16['max_workers']} workers, {verdict_16['replays']} replays, projected "
              f"{verdict_16['projected_hours']:.1f} h -> "
              f"{'C runs' if verdict_16['run'] else 'C IS NOT RUN'}", flush=True)
    else:
        print("  16-token smoke not run (no --trace16): control C has no verdict", flush=True)
    check_unmoved(start)
    config = {"mode": "smoke", **start, "anchor": anchor,
              "trace_files": {name: {"path": info["path"], "sha256": info["sha256"],
                                     "manifest": info.get("manifest")}
                              for name, info in inputs.items()},
              "inputs": inputs, "fine_input": fine,
              "replays": len(rows), "workers": args.workers, "workers16": args.workers16,
              "isolation": "one fresh forked process per replay; peak RSS includes the parent's "
                           "resident pages at fork (parent_rss_mib_at_fork)",
              "stop_conditions": {"max_seconds_512": bc.SMOKE_MAX_SECONDS,
                                  "max_rss_mib_512": bc.SMOKE_MAX_RSS_MIB,
                                  "max_seconds_16": bc.FINE_MAX_SECONDS,
                                  "max_rss_mib_16": bc.FINE_MAX_RSS_MIB,
                                  "max_projected_hours_16": bc.FINE_MAX_PROJECTED_HOURS},
              "verdict_512": verdict_512, "verdict_16": verdict_16,
              "identifier_problems": identifier_problems,
              "columns": list(SMOKE_COLUMNS) + ["check_identifiers_across_arms"],
              "note": "No utility of a smoke replay is written or read: the rows carry seconds, "
                      "memory, counts and integrity-check outcomes only.",
              "wall_seconds": time.time() - clock}
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    for directory in (args.run_dir, args.paper_dir):
        rdp._write(directory / "smoke.csv", rows)
        (directory / "smoke_config.json").write_text(json.dumps(config, indent=2, default=str)
                                                     + "\n", encoding="utf-8")
    print(f"smoke written to {args.paper_dir} ({time.time() - clock:.0f}s)", flush=True)


# --- derivation -------------------------------------------------------------------------------------

LEVEL_FIELDS = ("mean", "min", "max")
SPREAD_FIELDS = ("mean", "std", "min", "max")
_stats = rmc._stats


def _put(entry: dict, name: str, values, suffix: str, fields=LEVEL_FIELDS) -> None:
    stats = _stats([value for value in values if not _nan(value)])
    for field_name in fields:
        entry[f"{name}_{field_name}_{suffix}"] = stats[field_name]


def _nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _finite(values) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def index_sampled(rows) -> dict[tuple, dict[str, dict[int, dict]]]:
    """(trace, mechanism, fraction, multiplier) -> arm -> seed -> row."""
    out: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["mechanism"] == bc.HEAP:
            continue
        out[(row["trace"], row["mechanism"], float(row["l1_fraction"]),
             float(row["l2_multiplier"]))][row["arm"]][int(row["seed"])] = row
    return out


def index_heap(rows) -> dict[tuple, dict[str, dict]]:
    """(trace, fraction, multiplier) -> H_lru / H_off -> row."""
    out: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        if row["mechanism"] == bc.HEAP:
            out[(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))][
                row["arm"]] = row
    return out


def _cell_identity(name, mechanism, fraction, multiplier) -> dict:
    return {"trace": name, "mechanism": mechanism, "l1_fraction": fraction,
            "l2_multiplier": multiplier, "cell": rdp.cell_label(fraction, multiplier),
            "l2_level": bc.L2_LEVEL_OF_CELL[(fraction, multiplier)],
            "h_star": bc.hstar(fraction, multiplier)}


def cell_values(arms: dict[str, dict[int, dict]], heap: dict[str, dict] | None, h_star: float,
                window: str, seeds) -> dict:
    """`bc.cell_window_values` on one window from the replay rows, with the
    per-seed tokens beside the points and the heap references."""
    u_points, u_tokens = {}, {}
    for arm, by_seed in arms.items():
        missing = [seed for seed in seeds if seed not in by_seed]
        if missing:
            raise KeyError(f"{arm}: seeds {missing} missing")
        pairs = [bc.window_utility(by_seed[seed], window) for seed in seeds]
        u_tokens[arm] = [tokens for tokens, _ in pairs]
        u_points[arm] = [value for _, value in pairs]
    heap_values = {}
    for name in bc.HEAP_ARMS:
        if heap and name in heap:
            heap_values[name] = bc.window_utility(heap[name], window)
        else:
            heap_values[name] = (math.nan, math.nan)
    values = bc.cell_window_values(u_points, h_star, heap_values["H_off"][1])
    any_row = next(iter(arms[bc.LRU].values()))
    values.update(u_points=u_points, u_tokens=u_tokens, heap=heap_values,
                  requested=int(any_row[f"{window}_requested_tokens"]),
                  token_means={arm: bc.mean(tokens) for arm, tokens in u_tokens.items()})
    return values


def derive_values(rows, design: Design) -> dict[tuple, dict[str, dict]]:
    """(trace, mechanism, fraction, multiplier) -> window -> `cell_values`."""
    sampled, heap = index_sampled(rows), index_heap(rows)
    out = {}
    for key, arms in sorted(sampled.items()):
        name, mechanism, fraction, multiplier = key
        h_star = bc.hstar(fraction, multiplier)
        out[key] = {window: cell_values(arms, heap.get((name, fraction, multiplier)), h_star,
                                        window, design.seeds)
                    for window in bc.WINDOWS}
    return out


def counted(by_window: dict, window: str) -> bool:
    """Whether a trace x cell enters the counts on `window`: evaluable by the
    plan's rule (on W) and, on another window, with that window's own headroom
    at least 1.0 point (its S is not computed otherwise)."""
    return bool(by_window["W"]["evaluable"]) and (window == "W"
                                                  or bool(by_window[window]["evaluable"]))


def _alias(arm: str, h_star: float) -> str:
    return "hstar" if arm == horizon_arm(h_star) else arm


def transplant_table(values) -> list[dict]:
    """Reading 1 per trace x mechanism x cell: S_h* on W beside full, with
    the absolute U of lru, label and label_binary_h*, T and the label's share."""
    out = []
    for (name, mechanism, fraction, multiplier), by_window in values.items():
        h_star = bc.hstar(fraction, multiplier)
        entry = _cell_identity(name, mechanism, fraction, multiplier)
        entry.update(transplant_arm=horizon_arm(h_star),
                     seeds=len(by_window["W"]["u_points"][bc.LRU]),
                     evaluable=by_window["W"]["evaluable"])
        for window in SUMMARY_WINDOWS:
            v = by_window[window]
            entry[f"requested_tokens_{window}"] = v["requested"]
            for arm in (bc.LRU, bc.LABEL, horizon_arm(h_star)):
                alias = _alias(arm, h_star)
                _put(entry, f"U_{alias}_points", v["u_points"][arm], window)
                _put(entry, f"U_{alias}_tokens", v["u_tokens"][arm], window)
            entry[f"headroom_points_{window}"] = v["headroom"]
            entry[f"S_hstar_{window}"] = v["S_hstar"]
            seeds = _finite(v["S_seeds"][h_star])
            entry[f"S_hstar_seed_min_{window}"] = min(seeds) if seeds else math.nan
            entry[f"S_hstar_seed_max_{window}"] = max(seeds) if seeds else math.nan
            entry[f"suffices_{window}"] = bc.suffices(v["S_hstar"])
            for heap_name in bc.HEAP_ARMS:
                tokens, value = v["heap"][heap_name]
                entry[f"{heap_name}_tokens_{window}"] = tokens
                entry[f"{heap_name}_points_{window}"] = value
            entry[f"T_points_{window}"] = v["T"]
            entry[f"label_share_of_T_{window}"] = v["label_share_of_T"]
        entry["reading"] = (bc.NO_HEADROOM if not entry["evaluable"]
                            else "transplant_suffices" if entry["suffices_W"]
                            else "transplant_falls_short")
        out.append(entry)
    return out


def grid_table(values) -> list[dict]:
    """Reading 2 per trace x mechanism x cell x h: U and S_h on W beside full."""
    out = []
    for (name, mechanism, fraction, multiplier), by_window in values.items():
        h_star = bc.hstar(fraction, multiplier)
        for h in bc.GRID_SECONDS:
            arm = horizon_arm(h)
            entry = _cell_identity(name, mechanism, fraction, multiplier)
            entry.update(h=h, arm=arm, is_h_star=h == h_star,
                         part=bc.part(arm, fraction, multiplier),
                         evaluable=by_window["W"]["evaluable"])
            for window in SUMMARY_WINDOWS:
                v = by_window[window]
                _put(entry, "U_points", v["u_points"][arm], window, SPREAD_FIELDS)
                _put(entry, "U_tokens", v["u_tokens"][arm], window)
                entry[f"S_{window}"] = v["S"][h]
                seeds = _finite(v["S_seeds"][h])
                entry[f"S_seed_min_{window}"] = min(seeds) if seeds else math.nan
                entry[f"S_seed_max_{window}"] = max(seeds) if seeds else math.nan
                entry[f"suffices_{window}"] = bc.suffices(v["S"][h])
            out.append(entry)
    return out


def _horizons_text(horizons) -> str:
    return ";".join(f"{h:g}" for h in horizons)


def grid_summary_table(values) -> list[dict]:
    """Reading 2's per-cell summary: h_best, min_h S_h, the classification
    in the plan's order and the direction of failure, on W beside full."""
    out = []
    for (name, mechanism, fraction, multiplier), by_window in values.items():
        h_star = bc.hstar(fraction, multiplier)
        entry = _cell_identity(name, mechanism, fraction, multiplier)
        entry["evaluable"] = by_window["W"]["evaluable"]
        for window in SUMMARY_WINDOWS:
            v = by_window[window]
            # The rule is the plan's on W; on full, as context, a cell counts
            # when it is evaluable on W and its full-window S is computed.
            classification = bc.classify(counted(by_window, window), v["S_hstar"], v["min_S"])
            entry.update({f"headroom_points_{window}": v["headroom"],
                          f"S_hstar_{window}": v["S_hstar"],
                          f"h_best_{window}": "" if v["h_best"] is None else v["h_best"],
                          f"min_S_{window}": v["min_S"],
                          f"sufficing_horizons_{window}": _horizons_text(v["sufficing"]),
                          f"classification_{window}": classification,
                          f"direction_{window}": (bc.direction(h_star, v["h_best"])
                                                  if classification in bc.CLASSES[1:3] else ""),
                          f"rests_on_1200_{window}": v["rests_on_1200"],
                          f"T_points_{window}": v["T"],
                          f"label_share_of_T_{window}": v["label_share_of_T"]})
        entry["classification"] = entry["classification_W"]
        out.append(entry)
    return out


def monotonicity_table(values, design: Design) -> list[dict]:
    """Reading 2's monotonicity per trace x mechanism x window: the best grid
    horizon per L2 level (non-evaluable cells left out; the 1% level the
    smaller best of its two cells), assessable with at least three levels."""
    out = []
    names = sorted({key[0] for key in values})
    for name in names:
        for mechanism in design.mechanisms:
            for window in SUMMARY_WINDOWS:
                bests = {}
                for cell in design.cells:
                    v = values.get((name, mechanism) + tuple(cell))
                    if v is None:
                        continue
                    bests[tuple(cell)] = v[window]["h_best"] if counted(v, window) else None
                levels = bc.level_bests(bests)
                reading = bc.monotonicity(levels)
                entry = {"trace": name, "mechanism": mechanism, "window": window,
                         **{f"h_best_L2_{level * 100:g}pct": ("" if levels[level] is None
                                                              else levels[level])
                            for level in bc.L2_LEVELS},
                         "evaluable_levels": reading["evaluable_levels"],
                         "assessable": reading["assessable"],
                         "nondecreasing": ("" if reading["nondecreasing"] is None
                                           else reading["nondecreasing"]),
                         "rests_on_1200": reading["rests_on_1200"],
                         "sequence": ";".join(f"{level * 100:g}%:{best:g}"
                                              for level, best in reading["sequence"])}
                out.append(entry)
    return out


def _signs(entry: dict, prefix: str, values, suffix: str) -> None:
    positive, zero, negative = bc.seed_signs(values)
    entry.update({f"{prefix}_n_pos_{suffix}": positive, f"{prefix}_n_zero_{suffix}": zero,
                  f"{prefix}_n_neg_{suffix}": negative,
                  f"{prefix}_seed_signs_{suffix}": rmc._sign_string(values),
                  f"{prefix}_reading_{suffix}": bc.seed_reading(values)})


def _paired(arms, later: str, earlier: str, window: str, seeds) -> tuple[list[int], list[float]]:
    pairs = [bc.window_difference(arms[later][seed], arms[earlier][seed], window)
             for seed in seeds]
    return [tokens for tokens, _ in pairs], [value for _, value in pairs]


def random_table(rows, values, design: Design) -> list[dict]:
    """Reading 3 per trace x mechanism x cell: D_rand = U(label_binary_random_h*)
    - U(label_binary_h*), seed-paired, on W beside full."""
    sampled = index_sampled(rows)
    out = []
    for key, by_window in values.items():
        name, mechanism, fraction, multiplier = key
        h_star = bc.hstar(fraction, multiplier)
        rand, rung = bc.random_arm(h_star), horizon_arm(h_star)
        arms = sampled[key]
        entry = _cell_identity(name, mechanism, fraction, multiplier)
        entry.update(random_arm=rand, recency_arm=rung, evaluable=by_window["W"]["evaluable"],
                     random_draws=sum(int(arms[rand][seed]["random_draws"])
                                      for seed in design.seeds),
                     candidates_seen=sum(int(arms[rand][seed]["candidates_seen"])
                                         for seed in design.seeds))
        for window in SUMMARY_WINDOWS:
            v = by_window[window]
            _put(entry, "U_random_points", v["u_points"][rand], window)
            _put(entry, "U_recency_points", v["u_points"][rung], window)
            tokens, points = _paired(arms, rand, rung, window, design.seeds)
            _put(entry, "D_rand_points", points, window, ("mean", "ci95_half", "min", "max"))
            _put(entry, "D_rand_tokens", tokens, window)
            _signs(entry, "D_rand", points, window)
        out.append(entry)
    return out


def calibration_table(values) -> list[dict]:
    """Reading 4 (descriptive) per trace x mechanism x cell: h_cal on W1,
    S_{h_cal} on W2 beside min_h S_h on W2."""
    out = []
    for (name, mechanism, fraction, multiplier), by_window in values.items():
        first, second = by_window["W1"], by_window["W2"]
        reading = bc.calibration(first["S"], second["S"])
        entry = _cell_identity(name, mechanism, fraction, multiplier)
        entry.update(evaluable=by_window["W"]["evaluable"],
                     headroom_points_W1=first["headroom"], headroom_points_W2=second["headroom"],
                     **{f"S_W1_{h:g}": first["S"][h] for h in bc.GRID_SECONDS},
                     **{f"S_W2_{h:g}": second["S"][h] for h in bc.GRID_SECONDS},
                     h_cal="" if reading["h_cal"] is None else reading["h_cal"],
                     S_hcal_W1=reading["S_hcal_W1"], S_hcal_W2=reading["S_hcal_W2"],
                     hcal_suffices_W2=reading["hcal_suffices_W2"],
                     h_best_W2="" if reading["h_best_W2"] is None else reading["h_best_W2"],
                     min_S_W2=reading["min_S_W2"],
                     min_S_W2_suffices=reading["min_S_W2_suffices"],
                     rests_on_1200=reading["rests_on_1200"])
        out.append(entry)
    return out


def _order_entry(entry: dict, deltas: dict[str, tuple[list | None, list]], suffix: str) -> None:
    """Reading 7's columns of one window: per mechanism Delta in points (and
    tokens when given) with seed signs; separated and reversed when both
    mechanisms are there, empty otherwise."""
    for mechanism, (tokens, points) in deltas.items():
        _put(entry, f"Delta_{mechanism}_points", points, suffix)
        if tokens is not None:
            _put(entry, f"Delta_{mechanism}_tokens", tokens, suffix, ("mean",))
        _signs(entry, f"Delta_{mechanism}", points, suffix)
    if set(deltas) == {ALL16, LEAF16}:
        leaf, base = deltas[LEAF16][1], deltas[ALL16][1]
        entry[f"separated_{suffix}"] = bc.separated(leaf, base)
        entry[f"reversed_{suffix}"] = bc.reversed_signs(bc.seed_reading(leaf),
                                                        bc.seed_reading(base))
    else:
        entry[f"separated_{suffix}"] = ""
        entry[f"reversed_{suffix}"] = ""


def order_table(rows, values, design: Design) -> list[dict]:
    """Reading 7 (Addendum 1) per trace x cell x horizon role (h*, 600 s):
    Delta_m(h) = U_m(label) - U_m(label_binary_h), seed-paired within each
    mechanism, on W beside full; separated / reversed across the mechanisms;
    the status is "evaluable" when the cell is evaluable under both."""
    sampled = index_sampled(rows)
    out = []
    cells = sorted({(key[0], key[2], key[3]) for key in values})
    for name, fraction, multiplier in cells:
        if not all((name, mechanism, fraction, multiplier) in values
                   for mechanism in (ALL16, LEAF16)):
            continue
        evaluable = {mechanism: values[(name, mechanism, fraction, multiplier)]["W"]["evaluable"]
                     for mechanism in (ALL16, LEAF16)}
        status = bc.order_status(evaluable[ALL16], evaluable[LEAF16])
        for role in bc.ORDER_HORIZON_ROLES:
            h = bc.order_horizon(role, fraction, multiplier)
            rung = horizon_arm(h)
            entry = {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                     "cell": rdp.cell_label(fraction, multiplier),
                     "h_star": bc.hstar(fraction, multiplier), "role": role, "h": h,
                     "rung_arm": rung,
                     "predicted": role == "h_star" and (fraction, multiplier) in
                     bc.PREDICTION_7_CELLS,
                     "evaluable_all16": evaluable[ALL16], "evaluable_leaf16": evaluable[LEAF16],
                     "status": status}
            for window in SUMMARY_WINDOWS:
                deltas = {mechanism: _paired(sampled[(name, mechanism, fraction, multiplier)],
                                             bc.LABEL, rung, window, design.seeds)
                          for mechanism in (ALL16, LEAF16)}
                _order_entry(entry, deltas, window)
            out.append(entry)
    return out


def _sort(rows, keys) -> list[dict]:
    return sorted(rows, key=lambda row: tuple(
        (bc.MECHANISM_NAMES.index(row[key]) if key == "mechanism" and row[key] in
         bc.MECHANISM_NAMES else row[key]) for key in keys))


def aggregate_replays(rows) -> list[dict]:
    """Per trace x mechanism x cell x arm x window: five-seed mean / std / min
    / max of U in tokens and points (heap references: the single value), with
    the window's input tokens, and the replay's timing and decisions."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["trace"], row["mechanism"], float(row["l1_fraction"]),
                float(row["l2_multiplier"]), row["arm"])].append(row)
    out = []
    for (name, mechanism, fraction, multiplier, arm), members in groups.items():
        for window in bc.WINDOWS:
            utilities = [bc.window_utility(member, window) for member in members]
            entry = {"trace": name, "mechanism": mechanism, "l1_fraction": fraction,
                     "l2_multiplier": multiplier, "cell": rdp.cell_label(fraction, multiplier),
                     "arm": arm, "family": members[0]["family"], "part": members[0]["part"],
                     "arm_parameter": members[0]["arm_parameter"], "window": window,
                     "seeds": len(members),
                     "requested_tokens": int(members[0][f"{window}_requested_tokens"]),
                     "l1_avoided_tokens": int(members[0][f"{window}_l1_avoided_tokens"]),
                     "l1_capacity_bytes": members[0]["l1_capacity_bytes"],
                     "l2_capacity_bytes": members[0]["l2_capacity_bytes"]}
            for field_name in SPREAD_FIELDS:
                entry[f"U_tokens_{field_name}"] = _stats([t for t, _ in utilities])[field_name]
            for field_name in SPREAD_FIELDS:
                entry[f"U_points_{field_name}"] = _stats([p for _, p in utilities])[field_name]
            for metric in ("seconds", "worker_peak_rss_mib", "l2_decisions", "l2_rejections",
                           "l2_evictions"):
                stats = _stats([float(member.get(metric) or 0) for member in members])
                entry[f"{metric}_mean"] = stats["mean"]
                entry[f"{metric}_max"] = stats["max"]
            out.append(entry)
    order = {window: index for index, window in enumerate(bc.WINDOWS)}
    return sorted(out, key=lambda entry: (entry["trace"], entry["l1_fraction"],
                                          entry["l2_multiplier"], str(entry["mechanism"]),
                                          bc.ARMS.index(entry["arm"]) if entry["arm"] in bc.ARMS
                                          else 100 + bc.HEAP_ARMS.index(entry["arm"]),
                                          order[entry["window"]]))


# --- the Mooncake counterpart (reading 6 and reading 7's counterpart) -------------------------------


def _read(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _u_full(row) -> float:
    """U of a published row on its (full) window, in points."""
    return bc.points(int(row["extra_avoided_tokens"]), int(row["requested_tokens"]))


def _float(value) -> float:
    return math.nan if value in ("", None) else float(value)


def load_mooncake(paths=None) -> tuple[list[dict], list[dict], list[str]]:
    """The published Mooncake values of readings 1-3 and reading 7, full
    window, read from the published tables only: per trace x cell x mechanism
    (reading 6) and per trace x cell x horizon role (reading 7). `paths`
    overrides the files (tests). Every value the tables name must be there
    exactly once."""
    paths = dict(MOONCAKE_SOURCES, **(paths or {}))
    tables = {name: _read(path) for name, path in paths.items()}
    problems: list[str] = []

    def one(table, predicate, what):
        matches = [row for row in tables[table] if predicate(row)]
        if len(matches) != 1:
            problems.append(f"{table}: {len(matches)} rows for {what}, expected 1")
            return None
        return matches[0]

    def at(row, name, cell):
        return (row["trace"] == name and float(row["l1_fraction"]) == cell[0]
                and float(row["l2_multiplier"]) == cell[1])

    seeds_index: dict[tuple, dict[str, float]] = {}
    for table, mechanism in (("error_location_001/replay_seeds.csv", ALL16),
                             ("horizon_control_001/replay_seeds.csv", ALL16),
                             ("horizon_fill_001/replay_seeds.csv", ALL16),
                             ("leaf_matched_horizon_001/replay_seeds.csv", LEAF16)):
        for row in tables[table]:
            if row["mechanism"] != mechanism or row.get("variant", VARIANT) != VARIANT:
                continue
            key = (mechanism, row["arm"], row["trace"], float(row["l1_fraction"]),
                   float(row["l2_multiplier"]), int(row["seed"]))
            seeds_index.setdefault(key, {})[table] = _u_full(row)

    def seeds_of(mechanism, arm, name, cell, table):
        values = []
        for seed in bc.SEEDS:
            entry = seeds_index.get((mechanism, arm, name) + tuple(cell) + (seed,), {})
            if table not in entry:
                return None
            values.append(entry[table])
        return values

    side, order = [], []
    for name in MOONCAKE_TRACES:
        for cell in bc.CELLS:
            h_star = bc.hstar(*cell)
            shortfalls = {}
            for h in bc.GRID_SECONDS:
                if h == 150.0:
                    matches = [row for row in tables["horizon_fill_001/fill.csv"]
                               if at(row, name, cell) and float(row["h"]) == h]
                else:
                    matches = [row for row in tables["horizon_control_001/horizon.csv"]
                               if at(row, name, cell) and float(row["h"]) == h]
                if len(matches) > 1:
                    problems.append(f"{name}/{cell}: {len(matches)} published S at {h:g} s")
                elif matches:
                    shortfalls[h] = _float(matches[0]["S"])
            if h_star not in shortfalls:
                problems.append(f"{name}/{cell}: no published all16 S at h* {h_star:g} s")
            h_best, min_s = bc.best_horizon(shortfalls)
            mix = one("class_order_mix_001/differences.csv",
                      lambda row: at(row, name, cell) and float(row["h_star"]) == h_star,
                      f"{name}/{cell}")
            entry = {"workload": "mooncake", "trace": name, "l1_fraction": cell[0],
                     "l2_multiplier": cell[1], "cell": rdp.cell_label(*cell),
                     "mechanism": ALL16, "h_star": h_star,
                     "S_hstar_full": shortfalls.get(h_star, math.nan),
                     "suffices_full": bc.suffices(shortfalls.get(h_star, math.nan)),
                     "grid_points_full": _horizons_text(sorted(shortfalls)),
                     "h_best_full": "" if h_best is None else h_best, "min_S_full": min_s,
                     "min_S_suffices_full": bc.suffices(min_s),
                     "D_rand_arm": MOONCAKE_RANDOM_ARM,
                     "D_rand_note": "context, not the same arm: Mooncake's random arm kept the "
                                    "learned ranker's admission",
                     "source": "horizon_control_001/horizon.csv, horizon_fill_001/fill.csv (150 s), "
                               "class_order_mix_001/differences.csv"}
            if mix is not None:
                entry.update(D_rand_points_mean_full=_float(mix["D_rand_points_mean"]),
                             D_rand_points_min_full=_float(mix["D_rand_points_min"]),
                             D_rand_points_max_full=_float(mix["D_rand_points_max"]),
                             D_rand_seed_signs_full=mix["D_rand_seed_signs"],
                             D_rand_reading_full=mix["D_rand_reading"])
            side.append(entry)
            leaf = one("leaf_matched_horizon_001/horizon.csv",
                       lambda row: at(row, name, cell) and float(row["h_star"]) == h_star,
                       f"{name}/{cell}")
            entry = {"workload": "mooncake", "trace": name, "l1_fraction": cell[0],
                     "l2_multiplier": cell[1], "cell": rdp.cell_label(*cell),
                     "mechanism": LEAF16, "h_star": h_star,
                     "S_hstar_full": _float(leaf["S_hstar"]) if leaf else math.nan,
                     "suffices_full": bc.suffices(_float(leaf["S_hstar"])) if leaf else False,
                     "grid_points_full": "", "h_best_full": "", "min_S_full": math.nan,
                     "min_S_suffices_full": "", "D_rand_arm": "", "D_rand_note":
                         "no leaf16 grid or random arm was run on Mooncake",
                     "source": "leaf_matched_horizon_001/horizon.csv"}
            side.append(entry)
            for role in bc.ORDER_HORIZON_ROLES:
                h = bc.order_horizon(role, *cell)
                rung = horizon_arm(h)
                rung_table = ("horizon_fill_001/replay_seeds.csv" if h == 150.0
                              else "horizon_control_001/replay_seeds.csv")
                deltas = {}
                base_label = seeds_of(ALL16, bc.LABEL, name, cell,
                                      "error_location_001/replay_seeds.csv")
                base_rung = seeds_of(ALL16, rung, name, cell, rung_table)
                if base_label is None or base_rung is None:
                    problems.append(f"{name}/{cell}: published all16 label / {rung} seeds missing")
                else:
                    deltas[ALL16] = (None, bc.order_deltas(base_label, base_rung))
                leaf_label = seeds_of(LEAF16, bc.LABEL, name, cell,
                                      "leaf_matched_horizon_001/replay_seeds.csv")
                leaf_rung = seeds_of(LEAF16, rung, name, cell,
                                     "leaf_matched_horizon_001/replay_seeds.csv")
                if leaf_label is not None and leaf_rung is not None:
                    deltas[LEAF16] = (None, bc.order_deltas(leaf_label, leaf_rung))
                elif role == "h_star":
                    problems.append(f"{name}/{cell}: published leaf16 label / {rung} seeds missing")
                entry = {"workload": "mooncake", "trace": name, "l1_fraction": cell[0],
                         "l2_multiplier": cell[1], "cell": rdp.cell_label(*cell),
                         "h_star": h_star, "role": role, "h": h, "rung_arm": rung,
                         "predicted": "", "status": "not_applied",
                         "sources": "error_location_001 (all16 label), "
                                    + rung_table.split("/")[0] + f" (all16 {rung}), "
                                    "leaf_matched_horizon_001 (leaf16 label, leaf16 rung)"}
                for mechanism in (ALL16, LEAF16):
                    if mechanism not in deltas:
                        for column in ("points_mean", "points_min", "points_max", "n_pos", "n_zero",
                                       "n_neg", "seed_signs", "reading"):
                            entry[f"Delta_{mechanism}_{column}_full"] = ""
                _order_entry(entry, deltas, "full")
                order.append(entry)
    return side, order, problems


def side_by_side_table(values, mooncake_side) -> list[dict]:
    """Reading 6: the Bailian values of readings 1-3 (W beside full) and the
    published Mooncake values (full), one row per workload x trace x cell x
    mechanism, never pooled."""
    out = []
    for (name, mechanism, fraction, multiplier), by_window in values.items():
        h_star = bc.hstar(fraction, multiplier)
        entry = {"workload": "bailian", "trace": name, "l1_fraction": fraction,
                 "l2_multiplier": multiplier, "cell": rdp.cell_label(fraction, multiplier),
                 "mechanism": mechanism, "h_star": h_star,
                 "evaluable_W": by_window["W"]["evaluable"]}
        for window in SUMMARY_WINDOWS:
            v = by_window[window]
            entry.update({f"S_hstar_{window}": v["S_hstar"],
                          f"suffices_{window}": bc.suffices(v["S_hstar"]),
                          f"grid_points_{window}": _horizons_text(bc.GRID_SECONDS),
                          f"h_best_{window}": "" if v["h_best"] is None else v["h_best"],
                          f"min_S_{window}": v["min_S"],
                          f"min_S_suffices_{window}": bc.suffices(v["min_S"]),
                          f"D_rand_points_mean_{window}": bc.mean(v["D_rand"]),
                          f"D_rand_seed_signs_{window}": rmc._sign_string(v["D_rand"]),
                          f"D_rand_reading_{window}": v["D_rand_reading"]})
        entry.update(D_rand_arm=BAILIAN_RANDOM_ARM, D_rand_note="", source="this run")
        out.append(entry)
    out.sort(key=lambda entry: (entry["trace"], entry["l1_fraction"], entry["l2_multiplier"],
                                bc.MECHANISM_NAMES.index(entry["mechanism"])))
    return out + list(mooncake_side)


# --- the readings table -----------------------------------------------------------------------------


def _count_row(reading, workload, mechanism, kind, window, counted, count, evaluable, cells,
               threshold, holds, resting=0, detail="") -> dict:
    return {"reading": reading, "workload": workload, "mechanism": mechanism, "kind": kind,
            "window": window, "counted": counted, "count": count, "evaluable": evaluable,
            "cells": cells, "threshold": "" if threshold is None else threshold,
            "holds": "" if holds is None else holds, "count_resting_on_1200": resting,
            "detail": detail}


def readings_table(values, monotonicity_rows, order_rows, mooncake_order, design: Design
                   ) -> list[dict]:
    """The seven predictions (W; the full window's counts beside as context)
    and the descriptive counts the plan reports without a prediction."""
    out = []
    cells_total = len(design.traces) * len(design.cells)

    def per_cell(mechanism, window):
        return [(key, by_window[window], counted(by_window, window))
                for key, by_window in values.items() if key[1] == mechanism]

    for mechanism in design.mechanisms:
        for window in SUMMARY_WINDOWS:
            kind = "prediction" if window == "W" else "context"
            entries = per_cell(mechanism, window)
            evaluable = [(key, v) for key, v, ok in entries if ok]
            transplant = sum(1 for _, v in evaluable if bc.suffices(v["S_hstar"]))
            grid = [(key, v) for key, v in evaluable if bc.suffices(v["min_S"])]
            name1 = "1_transplant_all16" if mechanism == ALL16 else "reading1_transplant_leaf16"
            out.append(_count_row(name1, "bailian", mechanism,
                                  kind if mechanism == ALL16 else "descriptive", window,
                                  "S_hstar_at_most_0.10", transplant, len(evaluable), cells_total,
                                  bc.PREDICTIONS["1_transplant_all16"]["at_least"]
                                  if mechanism == ALL16 else None,
                                  (bc.prediction_holds("1_transplant_all16", transplant)
                                   if mechanism == ALL16 and window == "W" else None)))
            name3 = "3_grid_all16" if mechanism == ALL16 else "6_grid_leaf16"
            out.append(_count_row(name3, "bailian", mechanism, kind, window,
                                  "min_h_S_h_at_most_0.10", len(grid), len(evaluable),
                                  cells_total, bc.PREDICTIONS[name3]["at_least"],
                                  bc.prediction_holds(name3, len(grid)) if window == "W" else None,
                                  sum(1 for _, v in grid if v["rests_on_1200"])))
            classes = defaultdict(int)
            direction_entries = []
            for key, v, ok in entries:
                label = bc.classify(ok, v["S_hstar"], v["min_S"])
                classes[label] += 1
                direction_entries.append((label, bc.hstar(key[2], key[3]), v["h_best"]))
            directions = bc.direction_counts(direction_entries)
            holds2 = bc.direction_holds(directions)
            out.append(_count_row(
                "2_direction_of_failure_all16" if mechanism == ALL16
                else "reading2_direction_of_failure_leaf16", "bailian", mechanism,
                kind if mechanism == ALL16 else "descriptive", window,
                "longer/shorter", f"{directions['longer']}/{directions['shorter']}",
                directions["failing"], cells_total, "longer > shorter",
                holds2 if (mechanism == ALL16 and window == "W") else None,
                detail=("0/0: no evaluable cell where the transplant fails"
                        if holds2 is None else
                        f"over the {directions['failing']} evaluable cells with S_h* > 0.10; "
                        f"h_best equal to h* in {directions['equal']}")))
            out.append(_count_row("reading2_classification", "bailian", mechanism,
                                  "descriptive", window,
                                  "cells per class", sum(classes.values()), len(evaluable),
                                  cells_total, None, None,
                                  detail="; ".join(f"{label} {classes[label]}"
                                                   for label in bc.CLASSES)))
            mono = [row for row in monotonicity_rows
                    if row["mechanism"] == mechanism and row["window"] == window]
            assessable = [row for row in mono if row["assessable"]]
            nondecreasing = sum(1 for row in assessable if row["nondecreasing"] is True)
            name4 = ("4_monotonicity_all16" if mechanism == ALL16
                     else "reading2_monotonicity_leaf16")
            out.append(_count_row(name4, "bailian", mechanism,
                                  kind if mechanism == ALL16 else "descriptive", window,
                                  "traces with h_best nondecreasing in L2", nondecreasing,
                                  len(assessable), len(mono),
                                  bc.PREDICTIONS["4_monotonicity_all16"]["at_least"]
                                  if mechanism == ALL16 else None,
                                  (bc.prediction_holds("4_monotonicity_all16", nondecreasing)
                                   if mechanism == ALL16 and window == "W" else None),
                                  sum(1 for row in assessable if row["rests_on_1200"]),
                                  detail="not assessable: " + ", ".join(
                                      row["trace"] for row in mono if not row["assessable"])))
            losses = sum(1 for _, v in evaluable if v["D_rand_reading"] == bc.CONSISTENT_LOSS)
            name5 = ("5_random_order_all16" if mechanism == ALL16
                     else "reading3_random_order_leaf16")
            out.append(_count_row(name5, "bailian", mechanism,
                                  kind if mechanism == ALL16 else "descriptive", window,
                                  "D_rand_consistent_loss", losses, len(evaluable), cells_total,
                                  bc.PREDICTIONS["5_random_order_all16"]["at_least"]
                                  if mechanism == ALL16 else None,
                                  (bc.prediction_holds("5_random_order_all16", losses)
                                   if mechanism == ALL16 and window == "W" else None)))
            calibration = [bc.calibration(values[key]["W1"]["S"], values[key]["W2"]["S"])
                           for key, _, ok in entries if ok]
            if window == "W":
                out.append(_count_row("reading4_calibration_half", "bailian", mechanism,
                                      "descriptive",
                                      "W1->W2", "S_hcal_on_W2_at_most_0.10",
                                      sum(1 for c in calibration if c["hcal_suffices_W2"]),
                                      len(calibration), cells_total, None, None,
                                      sum(1 for c in calibration if c["rests_on_1200"]),
                                      detail="min_S_W2 at most 0.10: " + str(sum(
                                          1 for c in calibration if c["min_S_W2_suffices"]))))
    for window in SUMMARY_WINDOWS:
        kind = "prediction" if window == "W" else "context"
        hstar_rows = [row for row in order_rows if row["role"] == "h_star"]
        count7 = bc.prediction_7_count(((row["l1_fraction"], row["l2_multiplier"]),
                                        row["status"], row[f"separated_{window}"] is True)
                                       for row in hstar_rows)
        out.append(_count_row("7_order_beyond_bit_separated", "bailian",
                              f"{LEAF16}_vs_{ALL16}", kind, window,
                              "Delta_hstar_separated (h* < 600 s)", count7["separated"],
                              count7["evaluable"], count7["cells"],
                              bc.PREDICTIONS["7_order_beyond_bit_separated"]["at_least"],
                              (bc.prediction_holds("7_order_beyond_bit_separated",
                                                   count7["separated"])
                               if window == "W" else None)))
        for role in bc.ORDER_HORIZON_ROLES:
            selected = [row for row in order_rows if row["role"] == role]
            evaluable = [row for row in selected if row["status"] == "evaluable"]
            readings = {mechanism: defaultdict(int) for mechanism in (ALL16, LEAF16)}
            for row in evaluable:
                for mechanism in (ALL16, LEAF16):
                    readings[mechanism][row[f"Delta_{mechanism}_reading_{window}"]] += 1
            out.append(_count_row(
                f"reading7_order_beyond_bit_{role}", "bailian", f"{LEAF16}_vs_{ALL16}",
                "descriptive",
                window, "separated / reversed",
                f"{sum(1 for row in evaluable if row[f'separated_{window}'] is True)}/"
                f"{sum(1 for row in evaluable if row[f'reversed_{window}'] is True)}",
                len(evaluable), len(selected), None, None,
                detail="; ".join(f"{mechanism} " + ", ".join(
                    f"{label} {readings[mechanism][label]}" for label in bc.SIGN_READINGS)
                    for mechanism in (ALL16, LEAF16))))
    for role in bc.ORDER_HORIZON_ROLES:
        selected = [row for row in mooncake_order if row["role"] == role]
        both = [row for row in selected if row["separated_full"] != ""]
        readings = {mechanism: defaultdict(int) for mechanism in (ALL16, LEAF16)}
        for row in selected:
            for mechanism in (ALL16, LEAF16):
                if row[f"Delta_{mechanism}_reading_full"] != "":
                    readings[mechanism][row[f"Delta_{mechanism}_reading_full"]] += 1
        out.append(_count_row(
            f"reading7_order_beyond_bit_{role}", "mooncake", f"{LEAF16}_vs_{ALL16}", "context",
            "full",
            "separated / reversed",
            f"{sum(1 for row in both if row['separated_full'] is True)}/"
            f"{sum(1 for row in both if row['reversed_full'] is True)}", len(both), len(selected),
            None, None, detail="; ".join(f"{mechanism} " + ", ".join(
                f"{label} {readings[mechanism][label]}" for label in bc.SIGN_READINGS)
                for mechanism in (ALL16, LEAF16)) + "; published rows, no evaluable rule applied"))
    return out


# --- README ----------------------------------------------------------------------------------------

README_TEXT = """# External-workload check on the Qwen-Bailian traces

Pre-registration: `docs/bailian-external-check-plan.md` (with Addendum 1). Four converted Qwen-Bailian traces at 512-token blocks (`bailian_toc_trace`, `bailian_tob_trace`, `bailian_thinking_trace`, `bailian_coder_trace`; a second workload family of one provider, not four deployments and not pooled with Mooncake). Mechanisms `all16` (the arrival plus up to 16 uniformly sampled residents) and `leaf16` (leaf residents only). Cells: L1 = round(f x W), L2 = round(f x k x W) bytes of each trace's own packed working set W, f in {0.25%, 1%, 2%}, k in {1, 4}. Seeds 0-4. Split at 60% of the span, label horizon 600 s, L1 `lru`, union closure, tree hit rule. Arms per trace x mechanism x cell x seed, each the store's first minimum of its key with no override: `lru`, `label` (exact next use, `-log1p(min(delta, 600 s))`), `label_binary_h` for h in G = {6, 15, 60, 150, 300, 600, 1200} s (key `(next use within h, last_group)`), and `label_binary_random_{h*}` (key `((next use within h*, u), last_group)` with u a uniform draw per candidate per decision from `random.Random(classmix.random_stream_seed(seed, arm))`, for admission and eviction alike). h* is the Mooncake table carried over unchanged: 0.25% x 1: 60 s; 0.25% x 4: 150 s; 1% x 1: 150 s; 1% x 4: 600 s; 2% x 1: 300 s; 2% x 4: 600 s. 2,400 sampled replays (A = 960, B = 1,440) and 48 heap references (`H_lru`, `H_off`: run_decision_population's heap path, deterministic).

Windows: `full` = [split, end] (the replay's own counters); `W` = [split, end - 1,200 s] (the primary window: every label of every grid horizon is observed inside it); `W1` = [split, mid], `W2` = (mid, end - 1,200 s] with mid the midpoint of W. W, W1 and W2 are counted by read-only `LabelWindowUtilityCollector`s with inclusive bounds (W2 from `nextafter(mid)`). U is extra avoided prefill tokens over L1 alone, in tokens and in points (100 x tokens / the window's input tokens). Readings are on five-seed means on W, the full window beside them. A trace x cell is evaluable under a mechanism when the five-seed mean of U(label) - U(lru) on W is at least 1.0 point; otherwise it is "no headroom" and left out of every count. A ratio whose denominator is below 1.0 point is not computed (nan); no ratio is clipped. Every arm reads the trace's future on purpose and none is a policy; the comparator is sampled-16 greedy, not an optimum. At h = 1,200 s the bit is not a coarsening of the label (its cap is 600 s) and S can be negative; every reading that rests on the 1,200-second point is flagged (`rests_on_1200`).

Files of the anchor and the smoke (written before this run, by their own modes): `anchor.csv` (one row per Mooncake anchor replay: the published sources, what was compared, the values and whether each is equal, `reproduces`), `anchor_config.json`; `smoke.csv` (seconds, peak RSS, counts and integrity checks only; no utility), `smoke_config.json` (the stop-condition verdicts).

Common columns: `trace`, `mechanism`, `l1_fraction`, `l2_multiplier`, `cell`, `l2_level` (f x k), `h_star`. A suffix `_W` or `_full` names the window. Seed aggregates `_mean`, `_std`, `_ci95_half`, `_min`, `_max`. Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (+ 0 - in seed order), `<d>_reading` (consistent_gain / consistent_loss / mixed).

- `replay_seeds.csv`: one row per replay (sampled and heap): identity (`block_tokens`, `eligibility`, `width`, `arm`, `family` reference / grid / random / heap, `part` A / B, `arm_parameter` h, `seed`), the replay counters, the Phase 0.98b attribution columns and point columns, the error-location decision statistics at the 600-second label horizon (`stat_*`, `overridden_*`, `decision_sha256`, `m1` .. `m4_*`), `counters_sha256`, `arm_builder`, `candidate_decisions_seen`, `candidates_seen` (candidates over every decision), `random_stream_seed`, `random_draws`, `random_stream_separate` (random arm only), `<window>_<counter>` / `<window>_extra_avoided_tokens` / `U_<window>_points` for full, W, W1, W2, `window_problems` (empty), `seconds`, `worker_peak_rss_mib`, `worker_pss_mib_end`, `trace_load_seconds`; heap rows add `heap_policy`, `offline_tiebreak`, `l2_eviction`.
- `replay.csv`: per trace x mechanism x cell x arm x window: the window's `requested_tokens` and `l1_avoided_tokens`, capacities, `U_tokens_*` and `U_points_*` (mean, std, min, max over seeds; a heap reference has one value), and the mean / max of `seconds`, `worker_peak_rss_mib`, `l2_decisions`, `l2_rejections`, `l2_evictions`.
- `heap_references.csv`: the 48 heap rows (`H_lru`, `H_off`) with their window counters.
- `transplant.csv` (reading 1): per trace x mechanism x cell: `transplant_arm`, `evaluable`, per window the input tokens, `U_<lru|label|hstar>_<points|tokens>_<mean|min|max>`, `headroom_points` = U(label) - U(lru), `S_hstar` = (U(label) - U(label_binary_h*)) / (U(label) - U(lru)) on the five-seed means with its seed min / max (each seed's own S), `suffices` (S_h* <= 0.10), `H_off_*`, `H_lru_*` (tokens, points), `T_points` = H_off - U(lru) and `label_share_of_T` = (U(label) - U(lru)) / T; `reading` transplant_suffices / transplant_falls_short / no_headroom on W.
- `grid.csv` (reading 2): per trace x mechanism x cell x h in G: `arm`, `is_h_star`, `part`, `evaluable`, per window `U_points_*`, `U_tokens_*`, `S`, `S_seed_min`, `S_seed_max`, `suffices`.
- `grid_summary.csv` (reading 2): per trace x mechanism x cell: per window `headroom_points`, `S_hstar`, `h_best` (argmin of S_h, ties to the smaller h), `min_S`, `sufficing_horizons`, `classification` (transplant_suffices, retuned_grid_point_suffices, none_in_grid, no_headroom, in that order; on full, as context, a cell counts when it is evaluable on W and its full-window headroom is at least 1.0 point), `direction` of h_best relative to h* where the transplant fails, `rests_on_1200`, `T_points`, `label_share_of_T`; `classification` = the W classification.
- `monotonicity.csv` (reading 2, prediction 4): per trace x mechanism x window: `h_best_L2_<0.25|1|2|4|8>pct` (the 1% level the smaller best of its two cells; empty when no cell of the level is evaluable), `evaluable_levels`, `assessable` (at least three), `nondecreasing`, `rests_on_1200`, `sequence`.
- `random.csv` (reading 3): per trace x mechanism x cell: `random_arm`, `recency_arm`, `evaluable`, `random_draws` and `candidates_seen` (over seeds), per window U of both arms and D_rand = U(label_binary_random_h*) - U(label_binary_h*), seed-paired (`D_rand_points_*`, `D_rand_tokens_*`, seed signs).
- `calibration.csv` (reading 4, descriptive): per trace x mechanism x cell: `headroom_points_W1`, `headroom_points_W2`, `S_W1_<h>` and `S_W2_<h>` for every h in G, `h_cal` (argmin on W1, ties to the smaller h), `S_hcal_W1`, `S_hcal_W2`, `hcal_suffices_W2`, `h_best_W2`, `min_S_W2`, `min_S_W2_suffices`, `rests_on_1200`.
- `order_beyond_bit.csv` (reading 7, Addendum 1): per trace x cell x `role` (h_star / 600) at horizon `h`: `rung_arm`, `predicted` (role h_star at a cell with h* < 600 s), `evaluable_all16`, `evaluable_leaf16`, `status` (evaluable under both / no_headroom), per window and mechanism `Delta_<mechanism>_points_*` and `Delta_<mechanism>_tokens_mean` with seed signs, where Delta_m(h) = U_m(label) - U_m(label_binary_h) seed-paired within the mechanism; `separated` (the smallest leaf16 seed value above the largest all16 seed value, strictly) and `reversed` (both seed readings consistent and opposite).
- `mooncake_side_by_side.csv` (reading 6): one row per workload x trace x cell x mechanism: the Bailian values (W and full) of readings 1-3 beside the published Mooncake values (full): `S_hstar`, `suffices`, `grid_points` (Mooncake: the published horizons of G only: 6, 15, 60, 300, 600 s everywhere and 150 s at the two fill cells; no 1,200 s point), `h_best`, `min_S`, `D_rand_*`; `D_rand_arm` and `D_rand_note` say that Mooncake's random arm kept the learned ranker's admission, so that column is context, not the same arm. No Mooncake leaf16 grid or random arm exists.
- `mooncake_order_beyond_bit.csv` (reading 7's counterpart): the columns of `order_beyond_bit.csv` on the full window, from the published per-seed rows (error_location_001, horizon_control_001, horizon_fill_001, leaf_matched_horizon_001; U = 100 x extra avoided tokens / requested tokens); no evaluable rule applied; under leaf16 Delta(600) exists only at the 600-second cells.
- `readings.csv`: one row per reading x mechanism x window: `kind` (prediction on W; context on full; descriptive), `counted`, `count`, `evaluable` (the cells counted over), `cells`, `threshold` (an absolute count), `holds` (W only), `count_resting_on_1200`, `detail`. Predictions: 1 (S_h* <= 0.10 in at least 12, all16), 2 (longer > shorter among failing cells; 0/0 reported as such), 3 (min_h S_h <= 0.10 in at least 20, all16), 4 (nondecreasing in at least 3 of 4 traces, all16), 5 (D_rand a consistent loss in at least 16, all16), 6 (min_h S_h <= 0.10 in at least 16, leaf16), 7 (Delta(h*) separated in at least 10 of the 16 trace x cell with h* < 600 s, over those evaluable under both mechanisms). Descriptive rows are named `reading<k>_...` after the plan's reading: the leaf16 counterparts of predictions 1, 2, 4 and 5, the classification counts, the calibration half (reading 4), reading 7 at h* and at 600 s (separated / reversed and the Delta readings per mechanism) and its Mooncake counterpart.
- `checks.csv`: every required check: `check`, `scope`, `checked`, `problems`, `passes`, `detail`.
- `run_config.json`: plan and code commits, source manifest, traces with hashes and manifests, the anchor and smoke files used (hashes), the Mooncake tables read (hashes), the design, windows, capacities, checks, timing and memory.
"""

GRANULARITY_README = """# Granularity control (C) on To-B at 16-token blocks

Pre-registration: `docs/bailian-external-check-plan.md`, "Granularity control (C)". A sensitivity control, not a ground truth at 16 tokens and not pooled with A or B. To-B converted by the unmodified converter with `--block-tokens 16` and read by the unmodified loader at `block_size=16`; mechanism `all16`; arms `lru`, `label`, `label_binary_h*`; seeds 0-4 (or 0-2 when the smoke's per-replay time at 16 tokens exceeded 20 minutes; recorded in `run_config.json`); at the same absolute bytes as To-B's six 512-token cells: L1 = round(f x W512), L2 = round(f x k x W512) with W512 the 512-token To-B working set. The windows (split, W, W1, W2, full) are the 512-token trace's: the timestamps are the same. The 512-token side is read from the main run's `replay_seeds.csv` (hash recorded), on the same seeds; nothing is rerun at 512.

- `replay_seeds.csv`: one row per 16-token replay, the main run's columns.
- `replay.csv`: per cell x arm x window, U in tokens and points (mean, std, min, max).
- `granularity.csv`: per cell and window (W, full): `U_<arm>_<points|tokens>_<mean|min|max>_<16|512>`, `headroom_points_<16|512>`, `S_hstar_<16|512>` (nan when the headroom is below 1.0 point), `suffices_<16|512>`, `agree` (both computed and the same answer; empty when either is not computed).
- `checks.csv`: the main run's replay checks on the 16-token replays, and the cross-granularity checks (the same bytes, the same window input tokens and requests as the 512-token rows).
- `run_config.json`: commits, source manifest, the 16-token file and manifest hashes, the 512-token file hash, the main run's tables read (hashes), the smoke verdict, seeds and workers, checks, timing and memory.
"""


# --- main -------------------------------------------------------------------------------------------


def _bailian_gates(args, start: dict, design: Design) -> dict:
    anchor, problems = anchor_gate(args.paper_dir, start["source_manifest"], design)
    smoke, smoke_problems = smoke_gate(args.paper_dir, start["source_manifest"])
    problems += smoke_problems
    if problems:
        _report("GATE", problems)
        raise SystemExit("the anchor and the smoke do not let the run start; nothing run")
    return {"anchor": anchor, "smoke": {"paths": smoke["paths"],
                                        "verdict_512": smoke["config"]["verdict_512"]}}


def _write_tables(directory: Path, tables: dict[str, list[dict]]) -> None:
    for name, rows in tables.items():
        rdp._write(directory / f"{name}.csv", rows)


def run_main(args, design: Design = REGISTERED) -> None:
    clock = time.time()
    validate_arguments(args)
    start = provenance()
    gates = _bailian_gates(args, start, design)
    inputs, problems = load_inputs(args.traces, bc.BLOCK_TOKENS, design.traces, bailian=True)
    mooncake_side, mooncake_order, mooncake_problems = load_mooncake()
    if problems or mooncake_problems:
        _report("INPUT", problems + mooncake_problems)
        raise SystemExit("the inputs are not what the plan names; nothing run")
    names = tuple(name for name in design.traces)
    tasks = (bc.main_replays(names, design.cells, design.seeds, design.mechanisms)
             + bc.heap_replays(names, design.cells))
    if _registered(design):
        assert len(tasks) == bc.COUNT_SAMPLED + bc.COUNT_HEAP == 2448
        assert sum(1 for task in tasks if task[3] != bc.HEAP
                   and bc.part(task[4], task[1], task[2]) == "A") == bc.COUNT_A
    tasks = _ordered(tasks, names)
    rdp._SHARED.update(working_set={name: info["working_set_bytes"]
                                    for name, info in inputs.items()})
    SHARED.update(inputs={name: _worker_spec(info) for name, info in inputs.items()})
    args.run_dir.mkdir(parents=True)
    print(f"main: {len(tasks)} replays ({len(tasks) - len(names) * len(design.cells) * 2} sampled "
          f"+ {len(names) * len(design.cells) * 2} heap) on {min(args.workers, len(tasks))} "
          "workers", flush=True)
    available = rmc._proc_mib("/proc/meminfo", "MemAvailable")
    started = time.time()
    with (args.run_dir / "raw_replays.jsonl").open("w", encoding="utf-8") as raw:
        try:
            rows = run_tasks(tasks, args.workers, _replay_worker, raw)
        except IncompleteRun as error:
            raise SystemExit(f"the run is incomplete and nothing is published: {error}") from error
    replay_seconds = time.time() - started
    checks = run_checks(rows, tasks)
    if _registered(design):
        sampled = sum(1 for row in rows if row["mechanism"] != bc.HEAP)
        problems = [] if (sampled, len(rows) - sampled) == (bc.COUNT_SAMPLED, bc.COUNT_HEAP) else [
            f"{sampled} sampled and {len(rows) - sampled} heap replays"]
        checks.insert(0, {"check": "counts", "scope": "2,400 sampled + 48 heap",
                          "checked": len(rows), "problems": len(problems),
                          "passes": not problems, "detail": "; ".join(problems)})
    failures = _print_checks(checks)
    if failures:
        rdp._write(args.run_dir / "checks.csv", checks)
        raise SystemExit("checks failed, nothing derived or published: " + ", ".join(failures))

    rows.sort(key=lambda row: (names.index(row["trace"]), row["l1_fraction"],
                               row["l2_multiplier"], str(row["mechanism"]),
                               bc.ARMS.index(row["arm"]) if row["arm"] in bc.ARMS
                               else 100 + bc.HEAP_ARMS.index(row["arm"]),
                               -1 if row["seed"] == "" else row["seed"]))
    values = derive_values(rows, design)
    monotonicity_rows = monotonicity_table(values, design)
    order_rows = order_table(rows, values, design)
    tables = {
        "replay_seeds": rows,
        "replay": aggregate_replays(rows),
        "heap_references": [row for row in rows if row["mechanism"] == bc.HEAP],
        "transplant": transplant_table(values),
        "grid": grid_table(values),
        "grid_summary": grid_summary_table(values),
        "monotonicity": monotonicity_rows,
        "random": random_table(rows, values, design),
        "calibration": calibration_table(values),
        "order_beyond_bit": order_rows,
        "mooncake_side_by_side": side_by_side_table(values, mooncake_side),
        "mooncake_order_beyond_bit": mooncake_order,
        "readings": readings_table(values, monotonicity_rows, order_rows, mooncake_order, design),
        "checks": checks,
    }
    assert set(tables) == set(MAIN_TABLES)
    for entry in tables["readings"]:
        if entry["kind"] == "prediction":
            print(f"  {entry['reading']} ({entry['window']}): {entry['counted']} {entry['count']} "
                  f"over {entry['evaluable']} evaluable of {entry['cells']}; threshold "
                  f"{entry['threshold']}: {entry['holds']} {entry['detail']}", flush=True)
    check_unmoved(start)
    worker_peak = max(float(row["worker_peak_rss_mib"]) for row in rows)
    config = {
        "phase": "bailian_external_check", "mode": "main", **start, **gates,
        "registered_design": _registered(design),
        "design": {"traces": list(design.traces), "cells": [list(c) for c in design.cells],
                   "seeds": list(design.seeds), "mechanisms": list(design.mechanisms),
                   "grid_seconds": list(bc.GRID_SECONDS),
                   "h_star": {f"{f:g}x{k:g}": h for (f, k), h in bc.HSTAR_SECONDS.items()}},
        "trace_files": {name: {"path": info["path"], "sha256": info["sha256"],
                               "manifest": info["manifest"]} for name, info in inputs.items()},
        "inputs": inputs,
        "capacities": {name: {rdp.cell_label(*cell): {
            "l1_capacity_bytes": rdp._capacity(name, cell[0]),
            "l2_capacity_bytes": rdp._capacity(name, cell[0] * cell[1])}
            for cell in design.cells} for name in inputs},
        "mooncake_tables": _hashes(MOONCAKE_SOURCES.values()),
        "windows": {"tail_exclusion_seconds": bc.TAIL_EXCLUSION_SECONDS,
                    "collector": "onpolicy.LabelWindowUtilityCollector, inclusive bounds; W2 from "
                                 "math.nextafter(mid, inf)",
                    "bounds_ms": {name: info["windows"] for name, info in inputs.items()}},
        "arm_builders": {f"{arm}/{mechanism}": bc.arm_builder(arm, mechanism)
                         for arm in bc.ARMS for mechanism in bc.MECHANISM_NAMES},
        "heap_references": {name: policy for name, policy in bc.HEAP_POLICIES.items()},
        "random_stream": "random.Random(classmix.random_stream_seed(seed, arm)); the store's "
                         "sampling stream random.Random(l2_seed) is separate",
        "readings": {"suffices_shortfall": bc.SUFFICES_SHORTFALL,
                     "evaluable_points": bc.EVALUABLE_POINTS,
                     "min_denominator_points": bc.MIN_DENOMINATOR_POINTS,
                     "predictions": bc.PREDICTIONS, "classes": list(bc.CLASSES),
                     "l2_levels": list(bc.L2_LEVELS)},
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "checks": checks, "replays": len(rows), "workers": args.workers,
        "replay_seconds": replay_seconds, "wall_seconds": time.time() - clock,
        "memory": {"rss_unit": "MiB (Linux ru_maxrss KiB / 1024)",
                   "peak_worker_rss_mib": worker_peak,
                   "parent_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF),
                   "children_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_CHILDREN),
                   "mem_available_mib_at_start": available},
    }
    present = _existing(args.paper_dir, MAIN_FILES)
    if present:
        raise SystemExit(f"{present} appeared in {args.paper_dir} during the run")
    for directory in (args.run_dir, args.paper_dir):
        _write_tables(directory, tables)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(json.dumps(config, indent=2, default=str)
                                                   + "\n", encoding="utf-8")
    print(f"written to {args.run_dir}, {args.paper_dir}; done in {time.time() - clock:.0f}s",
          flush=True)


# --- granularity ------------------------------------------------------------------------------------


def granularity_gate(args, start: dict, design: Design) -> tuple[dict, dict, list[str]]:
    """What control C needs: a passing anchor and smoke from these sources,
    a 16-token smoke verdict that lets C run with these seeds and workers, and
    a complete main run whose checks passed."""
    problems = []
    anchor, anchor_problems = anchor_gate(args.main_dir, start["source_manifest"], design)
    problems += anchor_problems
    smoke, smoke_problems = smoke_gate(args.main_dir, start["source_manifest"])
    problems += smoke_problems
    verdict = (smoke.get("config") or {}).get("verdict_16")
    if verdict is None:
        problems.append("the smoke has no 16-token verdict (run the smoke with --trace16)")
    else:
        if verdict["run"] is not True:
            problems.append("the 16-token smoke verdict is that C is not run")
        if bool(args.reduced_seeds) != bool(verdict["reduce_seeds"]):
            problems.append(f"--reduced-seeds is {bool(args.reduced_seeds)} but the smoke's "
                            f"verdict reduces the seeds: {verdict['reduce_seeds']}")
        if args.workers > int(verdict["max_workers"]):
            problems.append(f"--workers {args.workers} exceeds the verdict's {verdict['max_workers']}")
    config_path = Path(args.main_dir) / "run_config.json"
    seeds_path = Path(args.main_dir) / "replay_seeds.csv"
    main_config = {}
    if not config_path.is_file() or not seeds_path.is_file():
        problems.append(f"no complete main run in {args.main_dir}")
    else:
        main_config = json.loads(config_path.read_text(encoding="utf-8"))
        if main_config.get("mode") != "main" or not all(
                check["passes"] for check in main_config.get("checks", [])):
            problems.append("the main run did not pass its checks")
    gate = {"anchor": anchor, "smoke": {"paths": smoke.get("paths"), "verdict_16": verdict},
            "main": (_hashes([config_path, seeds_path])
                     if config_path.is_file() and seeds_path.is_file() else {})}
    return gate, main_config, problems


def granularity_table(rows16, rows512, design: Design, seeds) -> list[dict]:
    """Per cell and window: the three arms' U, the headroom and S_h* at 16
    and at 512 tokens under the same bytes, and whether "suffices" agrees."""
    index = {16: defaultdict(dict), 512: defaultdict(dict)}
    for granularity, rows in ((16, rows16), (512, rows512)):
        for row in rows:
            index[granularity][(float(row["l1_fraction"]), float(row["l2_multiplier"]),
                                row["arm"])][int(row["seed"])] = row
    out = []
    for fraction, multiplier in design.cells:
        h_star = bc.hstar(fraction, multiplier)
        arms = bc.granularity_arms(fraction, multiplier)
        for window in SUMMARY_WINDOWS:
            entry = {"trace": bc.GRANULARITY_TRACE, "l1_fraction": fraction,
                     "l2_multiplier": multiplier, "cell": rdp.cell_label(fraction, multiplier),
                     "h_star": h_star, "window": window, "seeds": len(seeds)}
            answers = {}
            for granularity in (16, 512):
                means = {}
                for arm in arms:
                    by_seed = index[granularity][(fraction, multiplier, arm)]
                    pairs = [bc.window_utility(by_seed[seed], window) for seed in seeds]
                    alias = _alias(arm, h_star)
                    _put(entry, f"U_{alias}_points", [p for _, p in pairs],
                         str(granularity))
                    _put(entry, f"U_{alias}_tokens", [t for t, _ in pairs], str(granularity))
                    means[arm] = bc.mean([p for _, p in pairs])
                    entry[f"requested_tokens_{granularity}"] = int(
                        by_seed[seeds[0]][f"{window}_requested_tokens"])
                s = bc.shortfall(means[bc.LABEL], means[horizon_arm(h_star)], means[bc.LRU])
                entry[f"headroom_points_{granularity}"] = means[bc.LABEL] - means[bc.LRU]
                entry[f"S_hstar_{granularity}"] = s
                entry[f"suffices_{granularity}"] = bc.suffices(s)
                answers[granularity] = s
            computed = all(math.isfinite(s) for s in answers.values())
            entry["agree"] = (bc.suffices(answers[16]) == bc.suffices(answers[512])
                              if computed else "")
            out.append(entry)
    return out


def run_granularity(args, design: Design = REGISTERED) -> None:
    clock = time.time()
    validate_arguments(args)
    start = provenance()
    gate, main_config, problems = granularity_gate(args, start, design)
    seeds = tuple(design.reduced_seeds if args.reduced_seeds else design.seeds)
    inputs, input_problems = load_inputs([args.trace512], bc.BLOCK_TOKENS,
                                         (bc.GRANULARITY_TRACE,), bailian=True)
    problems += input_problems
    info512 = inputs.get(bc.GRANULARITY_TRACE)
    fine = None
    if info512 is not None:
        recorded = (main_config.get("trace_files", {}).get(bc.GRANULARITY_TRACE) or {})
        if recorded.get("sha256") != info512["sha256"]:
            problems.append("the 512-token To-B file is not the main run's")
        fine, fine_problems = fine_input(args.trace16, info512)
        problems += fine_problems
    rows512 = []
    if not problems:
        with (Path(args.main_dir) / "replay_seeds.csv").open(encoding="utf-8") as handle:
            rows512 = [row for row in csv.DictReader(handle)
                       if row["trace"] == bc.GRANULARITY_TRACE and row["mechanism"] == ALL16
                       and (float(row["l1_fraction"]), float(row["l2_multiplier"])) in
                       {tuple(cell) for cell in design.cells}
                       and row["arm"] in bc.granularity_arms(float(row["l1_fraction"]),
                                                             float(row["l2_multiplier"]))
                       and int(row["seed"]) in seeds]
        expected = len(bc.granularity_replays(seeds, design.cells))
        if len(rows512) != expected:
            problems.append(f"the main run has {len(rows512)} matching 512-token rows, not "
                            f"{expected}")
    if problems:
        _report("GATE", problems)
        raise SystemExit("control C cannot start; nothing run")
    tasks = bc.granularity_replays(seeds, design.cells)
    if _registered(design):
        assert len(tasks) == (54 if args.reduced_seeds else 90)
    rdp._SHARED.update(working_set={bc.GRANULARITY_TRACE: info512["working_set_bytes"]})
    SHARED.update(inputs={bc.GRANULARITY_TRACE: _worker_spec(fine, fine["path"],
                                                             bc.FINE_BLOCK_TOKENS)})
    args.run_dir.mkdir(parents=True)
    print(f"granularity: {len(tasks)} replays at 16-token blocks, seeds {list(seeds)}, on "
          f"{min(args.workers, len(tasks))} workers; capacities from the 512-token working set "
          f"{info512['working_set_bytes']} bytes", flush=True)
    started = time.time()
    with (args.run_dir / "raw_replays.jsonl").open("w", encoding="utf-8") as raw:
        try:
            rows = run_tasks(tasks, args.workers, _replay_worker, raw)
        except IncompleteRun as error:
            raise SystemExit(f"control C is incomplete and nothing is published: {error}") from error
    replay_seconds = time.time() - started
    checks = run_checks(rows, tasks)
    # Across the granularities: the same bytes, and the same requests and input
    # tokens on every window (the timestamps and input lengths are the same).
    cross = []
    by_key = {(float(r["l1_fraction"]), float(r["l2_multiplier"]), r["arm"], int(r["seed"])): r
              for r in rows512}
    for row in rows:
        other = by_key.get((float(row["l1_fraction"]), float(row["l2_multiplier"]), row["arm"],
                            int(row["seed"])))
        if other is None:
            cross.append(f"{_where(row)}: no 512-token row")
            continue
        for name in ("l1_capacity_bytes", "l2_capacity_bytes") + tuple(
                f"{window}_{counter}" for window in bc.WINDOWS
                for counter in ("measured_requests", "requested_tokens")):
            if int(row[name]) != int(other[name]):
                cross.append(f"{_where(row)}: {name} {row[name]} != {other[name]} at 512 tokens")
    checks.append({"check": "same_bytes_and_windows", "scope": "16-token rows vs the main run's "
                   "512-token rows", "checked": len(rows), "problems": len(cross),
                   "passes": not cross, "detail": "; ".join(cross[:20])})
    failures = _print_checks(checks)
    if failures:
        rdp._write(args.run_dir / "checks.csv", checks)
        raise SystemExit("checks failed, nothing derived or published: " + ", ".join(failures))
    rows.sort(key=lambda row: (row["l1_fraction"], row["l2_multiplier"],
                               bc.ARMS.index(row["arm"]), row["seed"]))
    tables = {"replay_seeds": rows, "replay": aggregate_replays(rows),
              "granularity": granularity_table(rows, rows512, design, seeds), "checks": checks}
    for entry in tables["granularity"]:
        print(f"  {entry['cell']:16s} {entry['window']:4s} S_h*: 16 {entry['S_hstar_16']:.3f} "
              f"512 {entry['S_hstar_512']:.3f} agree={entry['agree']}", flush=True)
    check_unmoved(start)
    config = {
        "phase": "bailian_granularity_control", "mode": "granularity", **start, **gate,
        "registered_design": _registered(design), "seeds": list(seeds),
        "reduced_seeds": bool(args.reduced_seeds), "workers": args.workers,
        "trace16": fine, "trace512": info512,
        "capacity_base_bytes": info512["working_set_bytes"],
        "capacities": {rdp.cell_label(*cell): {
            "l1_capacity_bytes": rdp._capacity(bc.GRANULARITY_TRACE, cell[0]),
            "l2_capacity_bytes": rdp._capacity(bc.GRANULARITY_TRACE, cell[0] * cell[1])}
            for cell in design.cells},
        "checks": checks, "replays": len(rows), "replay_seconds": replay_seconds,
        "wall_seconds": time.time() - clock,
        "memory": {"peak_worker_rss_mib": max(float(row["worker_peak_rss_mib"]) for row in rows),
                   "parent_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF)},
    }
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    present = _existing(args.paper_dir, GRANULARITY_FILES)
    if present:
        raise SystemExit(f"{present} appeared in {args.paper_dir} during the run")
    for directory in (args.run_dir, args.paper_dir):
        _write_tables(directory, tables)
        (directory / "README.md").write_text(GRANULARITY_README, encoding="utf-8")
        (directory / "run_config.json").write_text(json.dumps(config, indent=2, default=str)
                                                   + "\n", encoding="utf-8")
    print(f"written to {args.run_dir}, {args.paper_dir}; done in {time.time() - clock:.0f}s",
          flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    {"anchor": run_anchor, "smoke": run_smoke, "main": run_main,
     "granularity": run_granularity}[args.mode](args)


if __name__ == "__main__":
    main()
