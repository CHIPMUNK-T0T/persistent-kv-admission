#!/usr/bin/env python3
"""Frozen one-step counterfactual sampled-L2 action-value experiment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import resource
import signal
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# A fork inside a replay worker is safe only with one interpreter/native thread.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np

from persistent_kv_admission.counterfactual import (
    ForkController, NAMESPACE, atomic_json, label_tie_sets,
    load_population_verified, select_decisions,
)
from persistent_kv_admission.decisionpop import RawFeatureScorer, state_indices
from persistent_kv_admission.onpolicy import (
    LabelWindowUtilityCollector, deserialize_ranker, sha256_path,
)
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

PLAN_COMMIT = "75add46"
PLAN = REPOSITORY / "docs/counterfactual-action-value-plan.md"
ONPOLICY_CONFIG = REPOSITORY / "results/paper/onpolicy_learning/onpolicy_learning_config.json"
PAPER_SOURCE = REPOSITORY / "results/paper/onpolicy_learning"
SOURCE_FILES = (
    "src/persistent_kv_admission/twotier.py",
    "src/persistent_kv_admission/counterfactual.py",
    "scripts/run_counterfactual.py",
    "tests/test_counterfactual.py",
    "docs/counterfactual-action-value-plan.md",
    "docs/counterfactual-implementation.md",
)
def _source_tree() -> tuple[str, ...]:
    return tuple(sorted(
        str(path.relative_to(REPOSITORY))
        for path in (REPOSITORY / "src" / "persistent_kv_admission").glob("*.py")
    ))
PROTECTED_DOCS = (
    "README.md", "docs/decision-population-findings.md",
    "docs/experiment-plan.md",
)
CELLS = ((0.0025, 1), (0.02, 4))
POLICIES = ("pi0", "pi3")
SEEDS = tuple(range(5))
TRACES = ("conversation_trace", "toolagent_trace")
SELECTED_PER_LINEAGE = 8
SMOKE_IDENTITY = ("conversation_trace", 0.0025, 1, "pi0", 0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, check=True,
        capture_output=True, text=True
    ).stdout.strip()


@contextmanager
def _run_lock(output_dir: Path):
    if not output_dir.is_dir():
        raise FileNotFoundError(f"prepared output directory is missing: {output_dir}")
    with (output_dir / "run.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another smoke/full/aggregate run owns {output_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_manifest() -> dict:
    files = {
        name: sha256_path(REPOSITORY / name)
        for name in sorted(set(SOURCE_FILES) | set(_source_tree()))
    }
    return {
        "files": files,
        "sha256": _sha_bytes(json.dumps(
            files, sort_keys=True, separators=(",", ":")
        ).encode()),
    }


def protected_diff_hashes() -> dict[str, str]:
    return {
        name: _sha_bytes(subprocess.run(
            ["git", "diff", "HEAD", "--no-ext-diff", "--", name],
            cwd=REPOSITORY, check=True, capture_output=True
        ).stdout)
        for name in PROTECTED_DOCS
    }


def require_committed_source() -> str:
    paths = tuple(sorted(set(SOURCE_FILES) | set(_source_tree())))
    changed = _git("diff", "HEAD", "--name-only", "--", *paths)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    if changed or untracked:
        raise RuntimeError(f"counterfactual execution source is uncommitted: {changed} {untracked}")
    plan_commit = _git("log", "-1", "--format=%h", "--", str(PLAN.relative_to(REPOSITORY)))
    if plan_commit != PLAN_COMMIT:
        raise RuntimeError(f"plan commit mismatch: {plan_commit} != {PLAN_COMMIT}")
    return _git("rev-parse", "HEAD")


def lineage_key(name: str, fraction: float, multiplier: int, policy: str, seed: int) -> str:
    return f"{name}|l1={fraction:g},l2x{multiplier:g}|{policy}|s={seed}"


def lineage_slug(name: str, fraction: float, multiplier: int, policy: str, seed: int) -> str:
    return f"{name}__l1_{fraction:g}__l2x_{multiplier:g}__{policy}__seed_{seed}"


def identity(row: dict) -> tuple:
    return (
        row["trace"], float(row["l1_fraction"]), int(float(row["l2_multiplier"])),
        row["policy"], int(row["seed"])
    )


def _published_context() -> tuple[dict, dict, dict, dict]:
    config = json.loads(ONPOLICY_CONFIG.read_text(encoding="utf-8"))
    old_source = config["source_manifest_end"]["files"]
    for name, expected in old_source.items():
        if name.startswith("src/") and name != "src/persistent_kv_admission/twotier.py":
            if sha256_path(REPOSITORY / name) != expected:
                raise ValueError(f"published replay dependency changed: {name}")
    tables = {}
    for name in (
        "onpolicy_model_links.csv", "onpolicy_test_populations.csv",
        "onpolicy_seed_utility.csv"
    ):
        path = PAPER_SOURCE / name
        expected = config["output_artifacts"][name]["sha256"]
        if sha256_path(path) != expected:
            raise ValueError(f"published CSV hash mismatch: {path}")
        tables[name] = _read_csv(path)
    models = {
        identity(row): row for row in tables["onpolicy_model_links.csv"]
        if row["target"] == "next_use" and row["policy"] in POLICIES
        and (float(row["l1_fraction"]), int(float(row["l2_multiplier"]))) in CELLS
        and row["trace"] in TRACES
    }
    populations = {
        identity(row): row for row in tables["onpolicy_test_populations.csv"]
        if row["target"] == "next_use" and row["policy"] in POLICIES
        and (float(row["l1_fraction"]), int(float(row["l2_multiplier"]))) in CELLS
        and row["trace"] in TRACES
    }
    utilities = {
        identity(row): row for row in tables["onpolicy_seed_utility.csv"]
        if row["target"] == "next_use" and row["policy"] in POLICIES
        and (float(row["l1_fraction"]), int(float(row["l2_multiplier"]))) in CELLS
        and row["trace"] in TRACES
    }
    if any(len(mapping) != 40 for mapping in (models, populations, utilities)):
        raise AssertionError("published source index does not contain exactly 40 lineages")
    return config, models, populations, utilities


def prepare(output_dir: Path) -> dict:
    head = require_committed_source()
    if output_dir.exists():
        raise FileExistsError(f"new output directory required: {output_dir}")
    config, models, populations, _ = _published_context()
    source = source_manifest()
    trace_files = config["trace_files"]
    for name in TRACES:
        path = Path(trace_files[name]["path"])
        if sha256_path(path) != trace_files[name]["sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")
    lineages = []
    selected_count = 0
    for name in TRACES:
        for fraction, multiplier in CELLS:
            for policy in POLICIES:
                for seed in SEEDS:
                    key = (name, fraction, multiplier, policy, seed)
                    model, population_row = models[key], populations[key]
                    model_path = Path(model["model_path"])
                    deserialize_ranker(model_path, model["model_sha256"])
                    population_path = Path(population_row["population_path"])
                    population = load_population_verified(
                        population_path, population_row["population_sha256"]
                    )
                    if (
                        population.window != "test" or population.seed != seed
                        or population.decisions != 40_000
                        or population.horizon_seconds != 600.0
                    ):
                        raise AssertionError(f"source population identity mismatch: {key}")
                    lineage = lineage_key(*key)
                    chosen = select_decisions(
                        population, lineage, SELECTED_PER_LINEAGE
                    )
                    if any(
                        row["timestamp_ms"] < config["measure_from_ms"][name]
                        or row["timestamp_ms"] + 600_000 > config["trace_end_ms"][name]
                        for row in chosen
                    ):
                        raise AssertionError("selected decision crosses observable window")
                    selected_count += len(chosen)
                    lineages.append({
                        "trace": name, "l1_fraction": fraction,
                        "l2_multiplier": multiplier, "policy": policy, "seed": seed,
                        "lineage": lineage,
                        "slug": lineage_slug(*key),
                        "model_path": str(model_path.resolve()),
                        "model_sha256": model["model_sha256"],
                        "population_path": str(population_path.resolve()),
                        "population_sha256": population_row["population_sha256"],
                        "selected": chosen,
                    })
                    del population
    if len(lineages) != 40 or selected_count != 320:
        raise AssertionError("selection size mismatch")
    manifest = {
        "phase": NAMESPACE, "code_commit": head, "source_manifest": source,
        "plan_commit": PLAN_COMMIT, "plan_sha256": sha256_path(PLAN),
        "onpolicy_config_sha256": sha256_path(ONPOLICY_CONFIG),
        "published_csv_sha256": {
            name: config["output_artifacts"][name]["sha256"] for name in (
                "onpolicy_model_links.csv", "onpolicy_test_populations.csv",
                "onpolicy_seed_utility.csv"
            )
        },
        "trace_files": trace_files,
        "protected_diff_sha256": protected_diff_hashes(),
        "lineages": lineages,
        "selected_count": selected_count,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "selection_manifest.json", manifest)
    atomic_json(output_dir / "prepare_status.json", {
        "complete": True, "selected_count": selected_count,
        "manifest_sha256": sha256_path(output_dir / "selection_manifest.json")
    })
    return manifest


def _load_manifest(output_dir: Path) -> tuple[dict, str]:
    path = output_dir / "selection_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_hash = sha256_path(path)
    status = json.loads((output_dir / "prepare_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or status.get("manifest_sha256") != manifest_hash:
        raise AssertionError("selection manifest is not frozen")
    if manifest["code_commit"] != require_committed_source():
        raise AssertionError("selection manifest code commit mismatch")
    if manifest["source_manifest"] != source_manifest():
        raise AssertionError("selection manifest source hashes changed")
    if manifest["plan_sha256"] != sha256_path(PLAN):
        raise AssertionError("selection manifest plan hash changed")
    if manifest["onpolicy_config_sha256"] != sha256_path(ONPOLICY_CONFIG):
        raise AssertionError("selection manifest on-policy config changed")
    if manifest["protected_diff_sha256"] != protected_diff_hashes():
        raise AssertionError("protected pre-existing documentation changed")
    if len(manifest["lineages"]) != 40 or manifest["selected_count"] != 320:
        raise AssertionError("selection manifest cardinality mismatch")
    return manifest, manifest_hash


def _compare_reference(actual: dict, reference: dict) -> None:
    for name, value in actual.items():
        if name not in reference:
            raise AssertionError(f"missing published replay field: {name}")
        expected = reference[name]
        if isinstance(value, bool):
            equal = str(value).lower() == expected.lower()
        elif isinstance(value, int):
            equal = value == int(expected)
        elif isinstance(value, float):
            equal = value == float(expected)
        else:
            equal = str(value) == str(expected)
        if not equal:
            raise AssertionError(f"published replay mismatch {name}: {value} != {expected}")


def _worker_one(output_dir: Path, slug: str, smoke: bool) -> dict:
    manifest, manifest_hash = _load_manifest(output_dir)
    _, _, _, utilities = _published_context()
    rows = [row for row in manifest["lineages"] if row["slug"] == slug]
    if len(rows) != 1:
        raise ValueError(f"unknown lineage slug: {slug}")
    row = rows[0]
    name = row["trace"]
    trace_ref = manifest["trace_files"][name]
    trace_path = Path(trace_ref["path"])
    if sha256_path(trace_path) != trace_ref["sha256"]:
        raise ValueError(f"trace hash mismatch: {trace_path}")
    trace = load_mooncake_trace(trace_path)
    if trace.name != name:
        raise AssertionError("trace name mismatch")
    config = json.loads(ONPOLICY_CONFIG.read_text(encoding="utf-8"))
    fraction = float(row["l1_fraction"])
    multiplier = int(row["l2_multiplier"])
    seed = int(row["seed"])
    policy = row["policy"]
    key = (name, fraction, multiplier, policy, seed)
    selected = row["selected"][:1] if smoke else row["selected"]
    ranker = deserialize_ranker(row["model_path"], row["model_sha256"])
    scorer = RawFeatureScorer(trace, ranker)
    index = state_indices(trace)
    branch_dir = output_dir / "branches" / slug
    controller = ForkController(
        trace, scorer, index, selected, branch_dir, manifest_hash
    )
    split_ms = float(config["measure_from_ms"][name])
    label_window = LabelWindowUtilityCollector(
        trace, split_ms, trace.end_ms - 600_000.0
    )

    def on_request(*args):
        controller.on_request(*args)
        if not controller.is_child:
            label_window.on_request(*args)

    working_set = int(config["working_set_bytes"][name])
    l1_bytes = max(1, round(working_set * fraction))
    l2_bytes = max(1, round(working_set * fraction * multiplier))
    started = time.monotonic()
    try:
        result = run_two_tier(
            trace, "lru", l1_bytes, l2_bytes, "learned",
            hit_model="tree", closure="union", bytes_per_token=2048,
            size_model="packed", measure_from_ms=split_ms,
            occurrence_groups=_occurrence_groups(trace),
            l2_eviction="sampled", l2_sample_width=16, l2_seed=seed,
            l2_scorer=scorer, l2_arm="onpolicy", l2_request_hook=on_request,
            l2_arrival_protection="none", l2_override_hook=controller,
        )
        if controller.is_child:
            controller.finish_child(result)
            os._exit(0)
    except BaseException as error:
        if controller.is_child:
            controller.fail_child(error)
            os._exit(1)
        raise
    validated = controller.validate_parent(result)
    reference = utilities[key]
    _compare_reference(result.as_row(), reference)
    _compare_reference(label_window.row(), reference)
    if len(validated) != len(selected):
        raise AssertionError("lineage selected-snapshot count mismatch")
    expected_branches = sum(
        len(snapshot["metadata"]["candidate_indices"])
        * (3 if snapshot["selected"]["hash_rank"] < 2 else 1)
        for snapshot in validated
    )
    actual_branches = sum(len(snapshot["branches"]) for snapshot in validated)
    if expected_branches != actual_branches:
        raise AssertionError("lineage branch-count mismatch")
    detail = []
    branch_hashes = {}
    for snapshot in validated:
        saved = snapshot["selected"]
        for branch in snapshot["branches"]:
            path = controller._branch_path(
                saved, branch["replicate"], branch["action_index"]
            )
            branch_hashes[str(path.relative_to(output_dir))] = sha256_path(path)
        detail.append({
            "selection_hash": saved["selection_hash"],
            "hash_rank": saved["hash_rank"],
            "group_index": saved["group_index"],
            "ordinal": saved["ordinal"],
            "timestamp_ms": saved["timestamp_ms"],
            "metadata": snapshot["metadata"],
            "parent_reward": snapshot["parent_reward"].row(),
            "branch_count": len(snapshot["branches"]),
        })
    marker = {
        "complete": True, "smoke": smoke, "slug": slug,
        "manifest_sha256": manifest_hash,
        "model_sha256": row["model_sha256"],
        "population_sha256": row["population_sha256"],
        "published_replay_checked": True,
        "actual_branch_parent_checks": len(validated),
        "expected_branches": expected_branches,
        "actual_branches": actual_branches,
        "branch_sha256": branch_hashes,
        "decisions": detail,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    marker_path = (
        output_dir / "smoke_marker.json" if smoke
        else output_dir / "lineages" / f"{slug}.json"
    )
    if marker_path.exists():
        raise FileExistsError(f"completed marker already exists: {marker_path}")
    atomic_json(marker_path, marker)
    return marker


def _verify_marker(output_dir: Path, slug: str, manifest_hash: str) -> dict | None:
    path = output_dir / "lineages" / f"{slug}.json"
    if not path.exists():
        return None
    marker = json.loads(path.read_text(encoding="utf-8"))
    if (
        marker.get("complete") is not True
        or marker.get("slug") != slug
        or marker.get("manifest_sha256") != manifest_hash
        or marker.get("actual_branches") != marker.get("expected_branches")
        or marker.get("actual_branch_parent_checks") != 8
    ):
        raise AssertionError(f"completed lineage marker invalid: {path}")
    for relative, expected in marker["branch_sha256"].items():
        if sha256_path(output_dir / relative) != expected:
            raise AssertionError(f"completed branch hash mismatch: {relative}")
    return marker


def _spawn_worker(output_dir: Path, slug: str, smoke: bool, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    command = [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--output-dir", str(output_dir), "--slug", slug,
    ]
    if smoke:
        command.append("--smoke-worker")
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    process = subprocess.Popen(
        command, cwd=REPOSITORY, env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process


def _stop_process_groups(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run_smoke(output_dir: Path) -> dict:
    manifest, manifest_hash = _load_manifest(output_dir)
    row = next(
        row for row in manifest["lineages"]
        if (
            row["trace"], row["l1_fraction"], row["l2_multiplier"],
            row["policy"], row["seed"]
        ) == SMOKE_IDENTITY
    )
    marker_path = output_dir / "smoke_marker.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("manifest_sha256") != manifest_hash
            or marker.get("slug") != row["slug"]
            or marker.get("actual_branch_parent_checks") != 1
        ):
            raise AssertionError("existing smoke marker mismatch")
        return marker
    process = _spawn_worker(
        output_dir, row["slug"], True, output_dir / "logs" / "smoke.log"
    )
    try:
        exit_code = process.wait()
    except BaseException:
        _stop_process_groups([process])
        raise
    if exit_code:
        _stop_process_groups([process])
        raise RuntimeError(f"smoke worker failed ({exit_code}); see logs/smoke.log")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != manifest_hash:
        raise AssertionError("smoke marker manifest mismatch")
    return marker


def _available_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("Linux MemAvailable is unavailable")


def _run_full(output_dir: Path, requested_workers: int) -> dict:
    if requested_workers <= 0:
        raise ValueError("worker count must be positive")
    manifest, manifest_hash = _load_manifest(output_dir)
    smoke_path = output_dir / "smoke_marker.json"
    if not smoke_path.exists():
        raise RuntimeError("completed smoke run is required before full run")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("manifest_sha256") != manifest_hash
        or smoke.get("actual_branch_parent_checks") != 1
        or smoke.get("actual_branches") != smoke.get("expected_branches")
    ):
        raise AssertionError("smoke marker failed integrity check")
    smoke_branch_rows = []
    for relative, expected in smoke["branch_sha256"].items():
        path = output_dir / relative
        if sha256_path(path) != expected:
            raise AssertionError(f"smoke branch hash changed: {relative}")
        smoke_branch_rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not smoke_branch_rows:
        raise AssertionError("smoke has no completed branches")
    parent_rss = float(smoke["peak_rss_mib"])
    child_rss = max(float(row["peak_rss_mib"]) for row in smoke_branch_rows)
    available = _available_mib()
    safe_workers = max(1, math.floor(0.7 * available / max(parent_rss + child_rss, 1)))
    if requested_workers > safe_workers:
        raise ValueError(
            f"requested {requested_workers} workers exceeds measured memory limit "
            f"{safe_workers} (smoke parent {parent_rss:.1f} MiB, "
            f"child {child_rss:.1f} MiB, available {available:.1f} MiB)"
        )
    expected_branches = sum(
        sum(len(event["candidate_indices"]) * (3 if event["hash_rank"] < 2 else 1)
            for event in row["selected"])
        for row in manifest["lineages"]
    )
    smoke_cpu = sum(float(row["elapsed_seconds"]) for row in smoke_branch_rows)
    projected_cpu_h = expected_branches * smoke_cpu / len(smoke_branch_rows) / 3600.0
    current = {
        "phase": "full", "manifest_sha256": manifest_hash,
        "requested_workers": requested_workers, "safe_workers": safe_workers,
        "smoke_parent_peak_rss_mib": parent_rss,
        "smoke_child_peak_rss_mib": child_rss,
        "mem_available_mib_at_start": available,
        "expected_branch_count": expected_branches,
        "smoke_projected_branch_cpu_hours": projected_cpu_h,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": False, "completed_lineages": 0, "failed_lineages": [],
    }
    status_path = output_dir / "full_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") != manifest_hash:
            raise AssertionError("existing full status manifest mismatch")
        if old.get("complete"):
            raise RuntimeError("full run already complete")
    atomic_json(status_path, current)
    pending = [
        row for row in manifest["lineages"]
        if _verify_marker(output_dir, row["slug"], manifest_hash) is None
    ]
    current["completed_lineages"] = 40 - len(pending)
    running = {}
    launched = []
    try:
        while pending or running:
            while pending and len(running) < requested_workers:
                row = pending.pop(0)
                slug = row["slug"]
                process = _spawn_worker(
                    output_dir, slug, False, output_dir / "logs" / f"{slug}.log"
                )
                running[process.pid] = (process, slug)
                launched.append(process)
            finished = []
            for pid, (process, slug) in running.items():
                code = process.poll()
                if code is None:
                    continue
                finished.append(pid)
                if code:
                    current["failed_lineages"].append({
                        "slug": slug, "exit_code": code,
                        "log": str((output_dir / "logs" / f"{slug}.log").resolve())
                    })
                else:
                    _verify_marker(output_dir, slug, manifest_hash)
                    current["completed_lineages"] += 1
            for pid in finished:
                process, _ = running[pid]
                if process.returncode == 0:
                    launched.remove(process)
                del running[pid]
            atomic_json(status_path, current)
            if current["failed_lineages"]:
                raise RuntimeError(
                    f"{len(current['failed_lineages'])} lineage worker(s) failed; "
                    "inspect full_status.json and logs"
                )
            if running and not finished:
                time.sleep(0.5)
    except BaseException as error:
        _stop_process_groups(launched)
        current["complete"] = False
        current["error"] = repr(error)
        atomic_json(status_path, current)
        raise
    if current["completed_lineages"] != 40:
        raise AssertionError("full lineage count incomplete")
    current["complete"] = True
    current["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current["protected_diff_sha256_end"] = protected_diff_hashes()
    if current["protected_diff_sha256_end"] != manifest["protected_diff_sha256"]:
        raise AssertionError("protected documentation changed during full run")
    current["source_manifest_end"] = source_manifest()
    if current["source_manifest_end"] != manifest["source_manifest"]:
        raise AssertionError("execution source changed during full run")
    if sha256_path(PLAN) != manifest["plan_sha256"]:
        raise AssertionError("plan changed during full run")
    if sha256_path(ONPOLICY_CONFIG) != manifest["onpolicy_config_sha256"]:
        raise AssertionError("on-policy config changed during full run")
    atomic_json(status_path, current)
    return current


def _average_ranks(values: list[float]) -> np.ndarray:
    numbers = np.asarray(values, dtype=float)
    order = np.argsort(numbers, kind="stable")
    ranks = np.empty(len(numbers), dtype=float)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and numbers[order[end]] == numbers[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _spearman(values: list[float], q: list[int]) -> float | None:
    left, right = _average_ranks(values), _average_ranks([-value for value in q])
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float | None:
    numbers = [float(value) for value in values]
    return statistics.mean(numbers) if numbers else None


def _quantile90(values) -> float:
    numbers = sorted(float(value) for value in values)
    position = 0.9 * (len(numbers) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)


def _summarize_regrets(rows: list[dict], grouping: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        regrets = [int(row["regret_tokens"]) for row in subset]
        output.append({
            **dict(zip(grouping, key)),
            "decisions": len(subset),
            "mean_regret_tokens": _mean(regrets),
            "median_regret_tokens": statistics.median(regrets),
            "p90_regret_tokens": _quantile90(regrets),
            "max_regret_tokens": max(regrets),
            "zero_regret_rate": sum(value == 0 for value in regrets) / len(regrets),
            "regret_ge_512_rate": sum(value >= 512 for value in regrets) / len(regrets),
            "constant_q_decisions": sum(bool(row["constant_q"]) for row in subset),
            "mean_q_range_tokens": _mean(row["q_range_tokens"] for row in subset),
            "mean_spearman": _mean(
                row["spearman"] for row in subset if row["spearman"] is not None
            ),
        })
    return output


def aggregate(output_dir: Path, paper_dir: Path | None = None) -> dict:
    manifest, manifest_hash = _load_manifest(output_dir)
    status = json.loads((output_dir / "full_status.json").read_text(encoding="utf-8"))
    if (
        not status.get("complete") or status.get("manifest_sha256") != manifest_hash
        or status.get("completed_lineages") != 40
    ):
        raise AssertionError("cannot aggregate an incomplete full run")
    published = json.loads(ONPOLICY_CONFIG.read_text(encoding="utf-8"))
    selection_rows, action_rows, regret_rows = [], [], []
    total_branches = 0
    for lineage in manifest["lineages"]:
        marker = _verify_marker(output_dir, lineage["slug"], manifest_hash)
        if marker is None or marker["actual_branch_parent_checks"] != 8:
            raise AssertionError(f"missing baseline checks: {lineage['slug']}")
        selected = {row["selection_hash"]: row for row in lineage["selected"]}
        if len(marker["decisions"]) != 8:
            raise AssertionError("completed lineage does not contain eight decisions")
        for detail in marker["decisions"]:
            saved = selected[detail["selection_hash"]]
            metadata = detail["metadata"]
            actual = int(metadata["actual_index"])
            common = {
                "trace": lineage["trace"], "l1_fraction": lineage["l1_fraction"],
                "l2_multiplier": lineage["l2_multiplier"],
                "policy": lineage["policy"], "seed": lineage["seed"],
                "lineage": lineage["lineage"], "selection_hash": saved["selection_hash"],
                "hash_rank": saved["hash_rank"], "group_index": saved["group_index"],
                "ordinal": saved["ordinal"], "timestamp_ms": saved["timestamp_ms"],
                "candidate_count": metadata["candidate_count"],
                "arrival_rejection": metadata["arriving_index"] == actual,
                "actual_index": actual, "arriving_index": metadata["arriving_index"],
                "secondary_horizon_seconds": (
                    published["trace_end_ms"][lineage["trace"]] - saved["timestamp_ms"]
                ) / 1000.0,
            }
            selection_rows.append({
                **common,
                "population_sha256": lineage["population_sha256"],
                "model_sha256": lineage["model_sha256"],
            })
            replicates = (0, 1, 2) if saved["hash_rank"] < 2 else (0,)
            for replicate in replicates:
                branches = []
                for action in range(metadata["candidate_count"]):
                    relative = (
                        Path("branches") / lineage["slug"]
                        / f"{saved['selection_hash']}__r{replicate}__a{action}.json"
                    )
                    path = output_dir / relative
                    if sha256_path(path) != marker["branch_sha256"][str(relative)]:
                        raise AssertionError(f"branch hash changed: {relative}")
                    branch = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        branch["run_fingerprint"] != manifest_hash
                        or branch["selection_hash"] != saved["selection_hash"]
                        or branch["replicate"] != replicate
                        or branch["action_index"] != action
                    ):
                        raise AssertionError(f"branch identity changed: {relative}")
                    branches.append(branch)
                total_branches += len(branches)
                baseline = branches[actual]
                for action, branch in enumerate(branches):
                    reward = branch["reward"]
                    for window in ("600", "end"):
                        action_rows.append({
                            **common, "replicate": replicate, "window": window,
                            "action_index": action,
                            "candidate_state_index": metadata["candidate_indices"][action],
                            "is_actual": action == actual,
                            "is_arrival": action == metadata["arriving_index"],
                            "learned_score": metadata["scores"][action][0],
                            "last_group": metadata["last_group"][action],
                            "frequency": metadata["frequency"][action],
                            "next_use_delta_ms": metadata["next_use_delta_ms"][action],
                            "count_within_h": metadata["count_within_h"][action],
                            "q_tokens": reward[f"q{window}"],
                            "l2_q_tokens": reward[f"l2_q{window}"],
                            "delta_q_vs_actual_tokens": (
                                reward[f"q{window}"]
                                - baseline["reward"][f"q{window}"]
                            ),
                            "requested_tokens": reward[
                                "requested600" if window == "600" else "requested_end"
                            ],
                            "future_requests": reward[
                                "requests600" if window == "600" else "requests_end"
                            ],
                            "branch_content_sha256": branch["content_sha256"],
                        })
                for window in ("600", "end"):
                    q = [int(branch["reward"][f"q{window}"]) for branch in branches]
                    best, worst = max(q), min(q)
                    score_vectors = {
                        "learned": [pair[0] for pair in metadata["scores"]],
                        "next_use": [
                            -math.log1p(min(
                                (float(value) / 1000.0 if value is not None else math.inf),
                                600.0
                            )) for value in metadata["next_use_delta_ms"]
                        ],
                        "count": [
                            math.log1p(value) for value in metadata["count_within_h"]
                        ],
                        "lru": metadata["last_group"],
                        "lfu": metadata["frequency"],
                    }
                    learned_regret = best - q[metadata["selectors"]["learned"]]
                    for selector, choice in metadata["selectors"].items():
                        ties = metadata["label_ties"].get(selector)
                        regret_rows.append({
                            **common, "replicate": replicate, "window": window,
                            "selector": selector, "chosen_index": choice,
                            "chosen_is_actual": choice == actual,
                            "q_selected_tokens": q[choice],
                            "q_actual_tokens": q[actual],
                            "q_best_tokens": best, "q_worst_tokens": worst,
                            "q_range_tokens": best - worst,
                            "regret_tokens": best - q[choice],
                            "regret_minus_learned_tokens": q[
                                metadata["selectors"]["learned"]
                            ] - q[choice],
                            "actual_advantage_over_worst_tokens": q[actual] - worst,
                            "constant_q": best == worst,
                            "label_min_tie_count": len(ties) if ties else None,
                            "label_tie_best_regret_tokens": (
                                best - max(q[index] for index in ties) if ties else None
                            ),
                            "label_tie_worst_regret_tokens": (
                                best - min(q[index] for index in ties) if ties else None
                            ),
                            "spearman": _spearman(score_vectors[selector], q),
                        })
                    if learned_regret != best - q[actual]:
                        raise AssertionError("recorded learned action differs from actual")
    if len(selection_rows) != 320 or total_branches != status["expected_branch_count"]:
        raise AssertionError("aggregate selected/branch cardinality mismatch")
    seed_rows = _summarize_regrets(
        regret_rows,
        ("trace", "l1_fraction", "l2_multiplier", "policy", "seed",
         "replicate", "window", "selector")
    )
    type_rows = _summarize_regrets(
        [row for row in regret_rows if row["replicate"] == 0],
        ("trace", "l1_fraction", "l2_multiplier", "policy", "seed",
         "window", "selector", "arrival_rejection")
    )
    groups = defaultdict(list)
    for row in seed_rows:
        if row["replicate"] != 0:
            continue
        key = tuple(row[name] for name in (
            "trace", "l1_fraction", "l2_multiplier", "policy", "window", "selector"
        ))
        groups[key].append(row)
    stratum_rows = []
    for key, subset in sorted(groups.items()):
        if len(subset) != 5:
            raise AssertionError("stratum lacks five lineage seeds")
        stratum_rows.append({
            **dict(zip(
                ("trace", "l1_fraction", "l2_multiplier", "policy", "window", "selector"),
                key
            )),
            "lineage_seeds": 5,
            "decisions": sum(row["decisions"] for row in subset),
            "mean_of_seed_mean_regret_tokens": _mean(
                row["mean_regret_tokens"] for row in subset
            ),
            "mean_of_seed_zero_regret_rate": _mean(
                row["zero_regret_rate"] for row in subset
            ),
            "mean_of_seed_regret_ge_512_rate": _mean(
                row["regret_ge_512_rate"] for row in subset
            ),
        })
    sensitivity_rows = []
    sensitivity_groups = defaultdict(list)
    for row in regret_rows:
        if row["hash_rank"] >= 2:
            continue
        key = (
            row["lineage"], row["selection_hash"], row["window"], row["selector"]
        )
        sensitivity_groups[key].append(row)
    for key, subset in sorted(sensitivity_groups.items()):
        if sorted(row["replicate"] for row in subset) != [0, 1, 2]:
            raise AssertionError("sensitivity decision lacks three paired streams")
        captured = next(row for row in subset if row["replicate"] == 0)
        sensitivity_rows.append({
            "lineage": key[0], "selection_hash": key[1],
            "window": key[2], "selector": key[3],
            "captured_regret_tokens": captured["regret_tokens"],
            "three_stream_mean_regret_tokens": _mean(
                row["regret_tokens"] for row in subset
            ),
            "three_stream_mean_paired_diff_vs_learned_tokens": _mean(
                row["regret_minus_learned_tokens"] for row in subset
            ),
        })
    target = paper_dir if paper_dir is not None else output_dir / "tables"
    if target.exists():
        raise FileExistsError(f"new publication directory required: {target}")
    if (
        protected_diff_hashes() != manifest["protected_diff_sha256"]
        or source_manifest() != manifest["source_manifest"]
    ):
        raise AssertionError("source or protected documentation changed before publication")
    staging = target.with_name(target.name + f".staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"publication staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for name, rows in (
            ("selected_decisions.csv", selection_rows),
            ("branch_actions.csv", action_rows),
            ("decision_regrets.csv", regret_rows),
            ("seed_regrets.csv", seed_rows),
            ("decision_type_regrets.csv", type_rows),
            ("stratified_regrets.csv", stratum_rows),
            ("sensitivity.csv", sensitivity_rows),
        ):
            _write_csv(staging / name, rows)
        _plot_regrets(stratum_rows, staging)
        (staging / "README.md").write_text(
            "# One-step counterfactual action values\n\n"
            "This directory reports avoided-prefill-token Q for every legal candidate "
            "at 320 sampled L2 decisions. Each branch changes one victim and then "
            "resumes the frozen source policy. The 600-second window is primary; "
            "trace end is secondary.\n\n"
            "selected_decisions.csv identifies the fixed snapshots. "
            "branch_actions.csv gives every candidate and stream outcome. "
            "decision_regrets.csv compares selectors with the best Q in that same "
            "candidate set; label_tie_best/worst_regret show the range among "
            "candidates tied on the primary exact label. seed_regrets.csv and "
            "stratified_regrets.csv summarize primary and sensitivity streams "
            "without treating candidates as independent observations. "
            "sensitivity.csv averages paired regrets over each snapshot's three "
            "continuation streams.\n\n"
            "The two figures show mean candidate regret by selector and source "
            "policy for each trace/cell. pi0 and pi3 bars come from different "
            "policy-created states, so their difference is descriptive, not a "
            "causal comparison of the policies. The figures do not establish a "
            "trace-wide recoverable loss or a deployment-general effect. "
            "Use the run config, frozen selection manifest, and retained branch "
            "JSON files for reproduction and audit.\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(staging)
        raise
    summary = {
        "phase": NAMESPACE, "manifest_sha256": manifest_hash,
        "code_commit": manifest["code_commit"],
        "plan_commit": manifest["plan_commit"],
        "plan_sha256": manifest["plan_sha256"],
        "source_manifest_start": manifest["source_manifest"],
        "onpolicy_config_sha256": manifest["onpolicy_config_sha256"],
        "published_csv_sha256": manifest["published_csv_sha256"],
        "trace_files": manifest["trace_files"],
        "lineage_references": [{
            "lineage": row["lineage"],
            "model_sha256": row["model_sha256"],
            "population_sha256": row["population_sha256"],
        } for row in manifest["lineages"]],
        "selection_rule": "eight lowest SHA256 namespace/lineage/group/ordinal per published reservoir",
        "horizon_primary_seconds": 600,
        "horizon_secondary": "through trace end",
        "seed_derivation": "SHA256(namespace, continuation, lineage, group, ordinal, replicate) first 8 bytes big-endian",
        "full_status": status,
        "selected_decisions": len(selection_rows),
        "branch_continuations": total_branches,
        "actual_branch_parent_checks": 320,
        "files_sha256": {
            path.name: sha256_path(path)
            for path in sorted(staging.iterdir()) if path.suffix in (".csv", ".png")
        },
        "protected_diff_sha256_end": protected_diff_hashes(),
        "source_manifest_end": source_manifest(),
    }
    if (
        summary["protected_diff_sha256_end"] != manifest["protected_diff_sha256"]
        or summary["source_manifest_end"] != manifest["source_manifest"]
    ):
        shutil.rmtree(staging)
        raise AssertionError("source or protected documentation changed before publication")
    try:
        atomic_json(staging / "run_config.json", summary)
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _plot_regrets(stratum_rows: list[dict], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = ("learned", "next_use", "count", "lru", "lfu")
    colors = {"pi0": "#4169a8", "pi3": "#ca694b"}
    for window, filename in (
        ("600", "primary_600s_regret.png"),
        ("end", "secondary_trace_end_regret.png"),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
        for i, trace in enumerate(TRACES):
            for j, (fraction, multiplier) in enumerate(CELLS):
                axis = axes[i, j]
                for policy, offset in (("pi0", -0.18), ("pi3", 0.18)):
                    means = []
                    for selector in selectors:
                        matched = [
                            row for row in stratum_rows
                            if (
                                row["trace"] == trace
                                and row["l1_fraction"] == fraction
                                and row["l2_multiplier"] == multiplier
                                and row["policy"] == policy
                                and row["window"] == window
                                and row["selector"] == selector
                            )
                        ]
                        if len(matched) != 1:
                            raise AssertionError("missing regret plot stratum")
                        means.append(matched[0]["mean_of_seed_mean_regret_tokens"])
                    axis.bar(
                        np.arange(len(selectors)) + offset, means, width=0.36,
                        label=policy, color=colors[policy]
                    )
                axis.set_xticks(range(len(selectors)), selectors, rotation=30)
                axis.set_title(f"{trace.replace('_trace', '')}: L1 {fraction:g}, L2/L1 {multiplier}")
                axis.set_ylabel("Mean candidate regret (tokens)")
                axis.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2)
        figure.suptitle(
            "One forced victim, frozen policy continuation; "
            + ("future 600 s" if window == "600" else "through trace end")
        )
        figure.text(
            0.5, 0.025,
            "pi0 and pi3 bars use different sampled states; compare selectors within each bar group.",
            ha="center", fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.05, 1, 0.92))
        figure.savefig(target / filename, dpi=160)
        plt.close(figure)


def _detach(args) -> dict:
    if args.command not in ("smoke", "full", "aggregate"):
        raise ValueError("detach is supported for smoke, full, or aggregate")
    output_dir = args.output_dir.resolve()
    log_path = output_dir / "logs" / f"{args.command}_launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(Path(__file__).resolve()), args.command,
        "--output-dir", str(output_dir),
    ]
    if args.command == "full":
        command.extend(("--workers", str(args.workers)))
    if args.command == "aggregate" and args.paper_dir is not None:
        command.extend(("--paper-dir", str(args.paper_dir.resolve())))
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command, cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    info = {
        "command": args.command, "pid": process.pid,
        "log": str(log_path), "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    atomic_json(output_dir / f"{args.command}_launch_{process.pid}.json", info)
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "full", "aggregate", "worker", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name in ("smoke", "full", "aggregate"):
            command.add_argument("--detach", action="store_true")
        if name == "full":
            command.add_argument("--workers", type=int, default=1)
        if name == "aggregate":
            command.add_argument("--paper-dir", type=Path)
        if name == "worker":
            command.add_argument("--slug", required=True)
            command.add_argument("--smoke-worker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "detach", False):
        output = _detach(args)
    elif args.command == "prepare":
        output = prepare(args.output_dir)
        output = {
            "selected_count": output["selected_count"],
            "code_commit": output["code_commit"],
            "manifest": str(args.output_dir / "selection_manifest.json"),
        }
    elif args.command == "smoke":
        with _run_lock(args.output_dir):
            output = _run_smoke(args.output_dir)
    elif args.command == "full":
        with _run_lock(args.output_dir):
            output = _run_full(args.output_dir, args.workers)
    elif args.command == "aggregate":
        with _run_lock(args.output_dir):
            output = aggregate(args.output_dir, args.paper_dir)
    elif args.command == "worker":
        output = _worker_one(args.output_dir, args.slug, args.smoke_worker)
    else:
        paths = (
            args.output_dir / "prepare_status.json",
            args.output_dir / "smoke_marker.json",
            args.output_dir / "full_status.json",
        )
        output = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in paths if path.exists()
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
