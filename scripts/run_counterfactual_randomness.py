#!/usr/bin/env python3
"""Frozen fresh-RNG cross-fit replay of 40 counterfactual snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

import run_counterfactual as base
from persistent_kv_admission.counterfactual import (
    atomic_json, continuation_seed, load_population_verified, select_decisions,
)
from persistent_kv_admission.counterfactual_randomness import (
    FOLD_A, FOLD_B, NAMESPACE, STREAM_IDS, RandomnessForkController,
    crossfit_analysis, derive_stream_seed,
)
from persistent_kv_admission.decisionpop import RawFeatureScorer, state_indices
from persistent_kv_admission.onpolicy import (
    LabelWindowUtilityCollector, deserialize_ranker, sha256_path,
)
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

PLAN = REPOSITORY / "docs/counterfactual-randomness-plan.md"
PLAN_COMMIT = "c821254"
OLD_AUDIT = REPOSITORY / "results/counterfactual_action_001"
OLD_PAPER_CONFIG = REPOSITORY / "results/paper/counterfactual_action_001/run_config.json"
OLD_PAPER_CONFIG_SHA256 = "175970ef039c76e930398745c633598b3463109b8205191a6c90ce0f66ebbb7a"
OLD_MANIFEST_SHA256 = "5db02a59c058a5934d308b8f893cec855021a62fe1284071ade9d08f50ba80be"
SOURCE_FILES = tuple(sorted(set(base.SOURCE_FILES) | {
    "src/persistent_kv_admission/counterfactual_randomness.py",
    "scripts/run_counterfactual_randomness.py",
    "tests/test_counterfactual_randomness.py",
    "docs/counterfactual-randomness-plan.md",
    "docs/counterfactual-randomness-implementation.md",
}))
SMOKE_IDENTITY = base.SMOKE_IDENTITY


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> str:
    return base._git(*arguments)


def source_manifest() -> dict:
    files = {name: sha256_path(REPOSITORY / name)
             for name in sorted(set(SOURCE_FILES) | set(base._source_tree()))}
    return {"files": files,
            "sha256": _hash_bytes(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())}


def require_committed_source() -> str:
    paths = tuple(sorted(set(SOURCE_FILES) | set(base._source_tree())))
    changed = _git("diff", "HEAD", "--name-only", "--", *paths)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    if changed or untracked:
        raise RuntimeError(f"randomness execution source is uncommitted: {changed} {untracked}")
    plan_commit = _git("log", "-1", "--format=%h", "--", str(PLAN.relative_to(REPOSITORY)))
    if plan_commit != PLAN_COMMIT:
        raise RuntimeError(f"plan commit mismatch: {plan_commit} != {PLAN_COMMIT}")
    return _git("rev-parse", "HEAD")


def _old_context():
    if sha256_path(OLD_PAPER_CONFIG) != OLD_PAPER_CONFIG_SHA256:
        raise AssertionError("frozen original paper config hash mismatch")
    config = json.loads(OLD_PAPER_CONFIG.read_text(encoding="utf-8"))
    if config["manifest_sha256"] != OLD_MANIFEST_SHA256:
        raise AssertionError("paper config does not identify original selection")
    path = OLD_AUDIT / "selection_manifest.json"
    if sha256_path(path) != OLD_MANIFEST_SHA256:
        raise AssertionError("frozen original selection manifest hash mismatch")
    old_manifest = json.loads(path.read_text(encoding="utf-8"))
    if old_manifest["selected_count"] != 320 or len(old_manifest["lineages"]) != 40:
        raise AssertionError("original selection cardinality mismatch")
    return config, old_manifest


def _old_captured_reference(lineage: dict, selected: dict) -> dict:
    marker_path = OLD_AUDIT / "lineages" / f"{lineage['slug']}.json"
    marker_sha = sha256_path(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (not marker.get("complete") or marker["manifest_sha256"] != OLD_MANIFEST_SHA256
            or marker["actual_branch_parent_checks"] != 8):
        raise AssertionError("original lineage marker is invalid")
    actual = int(selected["victim_index"])
    relative = Path("branches") / lineage["slug"] / f"{selected['selection_hash']}__r0__a{actual}.json"
    branch_path = OLD_AUDIT / relative
    branch_sha = sha256_path(branch_path)
    if marker["branch_sha256"][str(relative)] != branch_sha:
        raise AssertionError("original captured branch hash mismatch")
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    if (branch["run_fingerprint"] != OLD_MANIFEST_SHA256
            or branch["selection_hash"] != selected["selection_hash"]
            or branch["replicate"] != 0 or branch["action_index"] != actual):
        raise AssertionError("original captured branch identity mismatch")
    if branch.get("content_sha256") != _hash_bytes(base.json.dumps(
            {k: v for k, v in branch.items() if k != "content_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()):
        raise AssertionError("original captured branch content hash mismatch")
    return {"marker_path": str(marker_path.resolve()), "marker_sha256": marker_sha,
            "branch_path": str(branch_path.resolve()), "branch_sha256": branch_sha}


def prepare(output_dir: Path) -> dict:
    head = require_committed_source()
    if output_dir.exists():
        raise FileExistsError(f"new output directory required: {output_dir}")
    old_config, old_manifest = _old_context()
    published, models, populations, _ = base._published_context()
    source = source_manifest()
    lineages, seen_seeds = [], set()
    for lineage in old_manifest["lineages"]:
        selected = lineage["selected"]
        if len(selected) != 8 or selected[0]["hash_rank"] != 0:
            raise AssertionError("original hash-ranked selection is malformed")
        selected = selected[0]
        key = (lineage["trace"], float(lineage["l1_fraction"]),
               int(lineage["l2_multiplier"]), lineage["policy"], int(lineage["seed"]))
        model, population_row = models[key], populations[key]
        if (str(Path(model["model_path"]).resolve()) != lineage["model_path"]
                or model["model_sha256"] != lineage["model_sha256"]
                or str(Path(population_row["population_path"]).resolve()) != lineage["population_path"]
                or population_row["population_sha256"] != lineage["population_sha256"]):
            raise AssertionError("old selection source differs from published source")
        deserialize_ranker(lineage["model_path"], lineage["model_sha256"])
        population = load_population_verified(lineage["population_path"], lineage["population_sha256"])
        if (population.window != "test" or population.seed != key[4]
                or population.decisions != 40_000 or population.horizon_seconds != 600.0):
            raise AssertionError("published population properties changed")
        recomputed = select_decisions(population, lineage["lineage"], count=1)[0]
        if recomputed != selected:
            raise AssertionError("first selected decision differs from published population")
        trace_ref = old_manifest["trace_files"][lineage["trace"]]
        if sha256_path(Path(trace_ref["path"])) != trace_ref["sha256"]:
            raise AssertionError("frozen trace hash mismatch")
        seeds = {r: derive_stream_seed(lineage["lineage"], selected["group_index"],
                                        selected["ordinal"], r) for r in STREAM_IDS}
        if len(set(seeds.values())) != 16 or seen_seeds.intersection(seeds.values()):
            raise AssertionError("fresh continuation seed collision")
        old_seeds = {continuation_seed(lineage["lineage"], selected["group_index"],
                                        selected["ordinal"], r) for r in (1, 2)}
        if set(seeds.values()).intersection(old_seeds):
            raise AssertionError("fresh seed collides with old sensitivity seed")
        seen_seeds.update(seeds.values())
        captured = _old_captured_reference(lineage, selected)
        lineages.append({k: v for k, v in lineage.items() if k != "selected"} |
                        {"selected": [selected], "stream_seeds": {str(k): str(v) for k, v in seeds.items()},
                         "old_captured": captured})
    if len(lineages) != 40 or sum(len(row["selected"][0]["candidate_indices"]) for row in lineages) != 678:
        raise AssertionError("fixed snapshot/action cardinality mismatch")
    manifest = {
        "phase": NAMESPACE, "code_commit": head, "source_manifest": source,
        "plan_commit": PLAN_COMMIT, "plan_sha256": sha256_path(PLAN),
        "old_paper_config_sha256": OLD_PAPER_CONFIG_SHA256,
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "onpolicy_config_sha256": sha256_path(base.ONPOLICY_CONFIG),
        "published_csv_sha256": old_manifest["published_csv_sha256"],
        "trace_files": old_manifest["trace_files"],
        "protected_diff_sha256": base.protected_diff_hashes(),
        "fold_A": list(FOLD_A), "fold_B": list(FOLD_B),
        "stream_count": 16, "snapshot_count": 40, "action_count": 678,
        "expected_fresh_branches": 10_848, "expected_validation_branches": 40,
        "lineages": lineages,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "selection_manifest.json", manifest)
    atomic_json(output_dir / "prepare_status.json", {
        "complete": True, "selected_count": 40,
        "manifest_sha256": sha256_path(output_dir / "selection_manifest.json")})
    return manifest


def _load_manifest(output_dir: Path) -> tuple[dict, str]:
    path = output_dir / "selection_manifest.json"
    digest = sha256_path(path)
    status = json.loads((output_dir / "prepare_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or status["manifest_sha256"] != digest:
        raise AssertionError("new selection manifest is not frozen")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["phase"] != NAMESPACE or manifest["code_commit"] != require_committed_source()
            or manifest["source_manifest"] != source_manifest()
            or manifest["plan_sha256"] != sha256_path(PLAN)
            or manifest["onpolicy_config_sha256"] != sha256_path(base.ONPOLICY_CONFIG)
            or manifest["protected_diff_sha256"] != base.protected_diff_hashes()
            or manifest["fold_A"] != list(FOLD_A) or manifest["fold_B"] != list(FOLD_B)
            or len(manifest["lineages"]) != 40 or manifest["action_count"] != 678):
        raise AssertionError("new frozen run configuration changed")
    _old_context()
    for row in manifest["lineages"]:
        selected = row["selected"][0]
        seeds = {str(r): str(derive_stream_seed(row["lineage"], selected["group_index"],
                                                selected["ordinal"], r)) for r in STREAM_IDS}
        if row["stream_seeds"] != seeds:
            raise AssertionError("frozen stream seeds changed")
        captured = row["old_captured"]
        if (sha256_path(Path(captured["marker_path"])) != captured["marker_sha256"]
                or sha256_path(Path(captured["branch_path"])) != captured["branch_sha256"]):
            raise AssertionError("frozen old captured reference changed")
    return manifest, digest


def _expected_pairs(row: dict, *, smoke: bool) -> set[tuple[int, int]]:
    selected = row["selected"][0]
    count = len(selected["candidate_indices"])
    streams = (1,) if smoke else STREAM_IDS
    return {(0, int(selected["victim_index"]))} | {
        (stream_id, action) for stream_id in streams for action in range(count)
    }


def _expected_relative_paths(row: dict, *, smoke: bool) -> dict[tuple[int, int], Path]:
    selected = row["selected"][0]
    root = Path("branches") / row["slug"]
    return {(r, a): root / f"{selected['selection_hash']}__r{r}__a{a}.json"
            for r, a in _expected_pairs(row, smoke=smoke)}


def _worker_one(output_dir: Path, slug: str, smoke: bool) -> dict:
    manifest, manifest_hash = _load_manifest(output_dir)
    _, _, _, utilities = base._published_context()
    matched = [row for row in manifest["lineages"] if row["slug"] == slug]
    if len(matched) != 1:
        raise ValueError(f"unknown lineage slug: {slug}")
    row = matched[0]
    if smoke and tuple((row["trace"], float(row["l1_fraction"]),
                        int(row["l2_multiplier"]), row["policy"], int(row["seed"]))) != SMOKE_IDENTITY:
        raise AssertionError("wrong smoke lineage")
    name = row["trace"]
    trace_ref = manifest["trace_files"][name]
    if sha256_path(Path(trace_ref["path"])) != trace_ref["sha256"]:
        raise AssertionError("frozen trace changed")
    trace = load_mooncake_trace(Path(trace_ref["path"]))
    if trace.name != name:
        raise AssertionError("trace name mismatch")
    config = json.loads(base.ONPOLICY_CONFIG.read_text(encoding="utf-8"))
    fraction, multiplier = float(row["l1_fraction"]), int(row["l2_multiplier"])
    seed, policy = int(row["seed"]), row["policy"]
    ranker = deserialize_ranker(row["model_path"], row["model_sha256"])
    scorer = RawFeatureScorer(trace, ranker)
    selected = row["selected"]
    controller = RandomnessForkController(
        trace, scorer, state_indices(trace), selected,
        output_dir / "branches" / slug, manifest_hash, smoke=smoke,
        stream_seeds={int(k): int(v) for k, v in row["stream_seeds"].items()},
    )
    split_ms = float(config["measure_from_ms"][name])
    label_window = LabelWindowUtilityCollector(trace, split_ms, trace.end_ms - 600_000.0)

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
            trace, "lru", l1_bytes, l2_bytes, "learned", hit_model="tree",
            closure="union", bytes_per_token=2048, size_model="packed",
            measure_from_ms=split_ms, occurrence_groups=_occurrence_groups(trace),
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
    key = (name, fraction, multiplier, policy, seed)
    base._compare_reference(result.as_row(), utilities[key])
    base._compare_reference(label_window.row(), utilities[key])
    if len(validated) != 1:
        raise AssertionError("expected exactly one frozen snapshot")
    snapshot = validated[0]
    saved = snapshot["selected"]
    expected = _expected_pairs(row, smoke=smoke)
    branch_rows = {(b["replicate"], b["action_index"]): b for b in snapshot["branches"]}
    if len(branch_rows) != len(snapshot["branches"]) or set(branch_rows) != expected:
        raise AssertionError("completed branch action/stream set mismatch")
    actual = saved["victim_index"]
    new_captured = branch_rows[(0, actual)]
    old_ref = row["old_captured"]
    old_path = Path(old_ref["branch_path"])
    if sha256_path(old_path) != old_ref["branch_sha256"]:
        raise AssertionError("old captured branch changed")
    old_captured = json.loads(old_path.read_text(encoding="utf-8"))
    for field in ("reward", "terminal_state_digest", "end_result"):
        if new_captured[field] != old_captured[field]:
            raise AssertionError(f"new captured actual branch differs from original: {field}")
    relative = _expected_relative_paths(row, smoke=smoke)
    branch_hashes = {str(path): sha256_path(output_dir / path) for path in relative.values()}
    marker = {
        "complete": True, "smoke": smoke, "slug": slug,
        "manifest_sha256": manifest_hash,
        "selection_hash": saved["selection_hash"],
        "candidate_indices": saved["candidate_indices"],
        "model_sha256": row["model_sha256"],
        "population_sha256": row["population_sha256"],
        "published_replay_checked": True,
        "captured_parent_and_original_checked": True,
        "expected_branches": len(expected), "actual_branches": len(branch_rows),
        "branch_sha256": branch_hashes,
        "metadata": snapshot["metadata"],
        "parent_reward": snapshot["parent_reward"].row(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    marker_path = output_dir / ("smoke_marker.json" if smoke else f"lineages/{slug}.json")
    if marker_path.exists():
        raise FileExistsError(f"completed marker already exists: {marker_path}")
    atomic_json(marker_path, marker)
    return marker


def _verify_marker(output_dir: Path, row: dict, manifest_hash: str, *, smoke: bool) -> dict | None:
    path = output_dir / ("smoke_marker.json" if smoke else f"lineages/{row['slug']}.json")
    if not path.exists():
        return None
    marker = json.loads(path.read_text(encoding="utf-8"))
    expected = _expected_relative_paths(row, smoke=smoke)
    if (not marker.get("complete") or marker.get("smoke") is not smoke
            or marker.get("slug") != row["slug"]
            or marker.get("manifest_sha256") != manifest_hash
            or marker.get("selection_hash") != row["selected"][0]["selection_hash"]
            or marker.get("candidate_indices") != row["selected"][0]["candidate_indices"]
            or marker.get("model_sha256") != row["model_sha256"]
            or marker.get("population_sha256") != row["population_sha256"]
            or not marker.get("published_replay_checked")
            or not marker.get("captured_parent_and_original_checked")
            or marker.get("expected_branches") != len(expected)
            or marker.get("actual_branches") != len(expected)
            or set(marker.get("branch_sha256", {})) != {str(path) for path in expected.values()}):
        raise AssertionError(f"completed marker identity/action set mismatch: {path}")
    for (replicate, action), relative in expected.items():
        branch_path = output_dir / relative
        if sha256_path(branch_path) != marker["branch_sha256"][str(relative)]:
            raise AssertionError(f"completed branch hash mismatch: {relative}")
        branch = json.loads(branch_path.read_text(encoding="utf-8"))
        claimed = branch.get("content_sha256")
        content = {k: v for k, v in branch.items() if k != "content_sha256"}
        if (claimed != _hash_bytes(json.dumps(content, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode())
                or branch.get("run_fingerprint") != manifest_hash
                or branch.get("selection_hash") != row["selected"][0]["selection_hash"]
                or branch.get("replicate") != replicate or branch.get("action_index") != action):
            raise AssertionError(f"completed branch identity/content mismatch: {relative}")
    return marker


def _spawn_worker(output_dir: Path, slug: str, smoke: bool, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker",
             "--output-dir", str(output_dir), "--slug", slug] + (["--smoke"] if smoke else []),
            cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def smoke(output_dir: Path) -> dict:
    manifest, digest = _load_manifest(output_dir)
    row = next(row for row in manifest["lineages"] if (
        row["trace"], float(row["l1_fraction"]), int(row["l2_multiplier"]),
        row["policy"], int(row["seed"])) == SMOKE_IDENTITY)
    existing = _verify_marker(output_dir, row, digest, smoke=True)
    if existing is not None:
        return existing
    process = _spawn_worker(output_dir, row["slug"], True, output_dir / "logs/smoke.log")
    try:
        if process.wait():
            raise RuntimeError("smoke worker failed; inspect logs/smoke.log")
    except BaseException:
        base._stop_process_groups([process])
        raise
    marker = _verify_marker(output_dir, row, digest, smoke=True)
    if marker is None:
        raise AssertionError("smoke worker produced no marker")
    return marker


def _available_mib() -> float:
    host = base._available_mib()
    memory_max = Path("/sys/fs/cgroup/memory.max")
    memory_current = Path("/sys/fs/cgroup/memory.current")
    if memory_max.exists() and memory_current.exists():
        raw = memory_max.read_text().strip()
        if raw != "max":
            host = min(host, max(0, (int(raw) - int(memory_current.read_text().strip())) / 1024**2))
    return host


def full(output_dir: Path, requested_workers: int) -> dict:
    if not 1 <= requested_workers <= 6:
        raise ValueError("workers must be in 1..6")
    manifest, digest = _load_manifest(output_dir)
    smoke_row = next(row for row in manifest["lineages"] if (
        row["trace"], float(row["l1_fraction"]), int(row["l2_multiplier"]),
        row["policy"], int(row["seed"])) == SMOKE_IDENTITY)
    smoke_marker = _verify_marker(output_dir, smoke_row, digest, smoke=True)
    if smoke_marker is None:
        raise RuntimeError("completed smoke is required")
    child_rss = max(json.loads((output_dir / rel).read_text(encoding="utf-8"))["peak_rss_mib"]
                    for rel in smoke_marker["branch_sha256"])
    parent_rss = float(smoke_marker["peak_rss_mib"])
    available = _available_mib()
    safe_workers = max(0, math.floor(0.7 * available / max(parent_rss + child_rss, 1)))
    if requested_workers > safe_workers:
        raise ValueError(f"requested {requested_workers} workers exceeds smoke memory guard {safe_workers}")
    status_path = output_dir / "full_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") != digest:
            raise AssertionError("existing full status manifest mismatch")
        if old.get("complete"):
            raise RuntimeError("full run already complete")
    current = {"phase": "full", "manifest_sha256": digest,
               "requested_workers": requested_workers, "safe_workers": safe_workers,
               "smoke_parent_peak_rss_mib": parent_rss,
               "smoke_child_peak_rss_mib": child_rss,
               "mem_available_mib_at_start": available,
               "expected_branch_count": 10_888,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "complete": False, "completed_lineages": 0, "failed_lineages": []}
    pending = []
    for row in manifest["lineages"]:
        if _verify_marker(output_dir, row, digest, smoke=False) is None:
            pending.append(row)
    current["completed_lineages"] = 40 - len(pending)
    atomic_json(status_path, current)
    running = {}
    launched = []
    try:
        while pending or running:
            while pending and len(running) < requested_workers:
                row = pending.pop(0)
                process = _spawn_worker(output_dir, row["slug"], False,
                                        output_dir / "logs" / f"{row['slug']}.log")
                running[process.pid] = (process, row)
                launched.append(process)
            finished = []
            for pid, (process, row) in running.items():
                code = process.poll()
                if code is None:
                    continue
                finished.append(pid)
                if code:
                    current["failed_lineages"].append({"slug": row["slug"], "exit_code": code,
                        "log": str((output_dir / "logs" / f"{row['slug']}.log").resolve())})
                else:
                    if _verify_marker(output_dir, row, digest, smoke=False) is None:
                        raise AssertionError(f"worker exited without completed marker: {row['slug']}")
                    current["completed_lineages"] += 1
            for pid in finished:
                process, _ = running.pop(pid)
                if process.returncode == 0:
                    launched.remove(process)
            atomic_json(status_path, current)
            if current["failed_lineages"]:
                raise RuntimeError("lineage worker failed; inspect full_status.json and logs")
            if running and not finished:
                time.sleep(0.5)
    except BaseException as error:
        base._stop_process_groups(launched)
        current["complete"] = False
        current["error"] = repr(error)
        atomic_json(status_path, current)
        raise
    if current["completed_lineages"] != 40:
        raise AssertionError("full lineage count incomplete")
    _load_manifest(output_dir)
    current["source_manifest_end"] = source_manifest()
    current["protected_diff_sha256_end"] = base.protected_diff_hashes()
    if (current["source_manifest_end"] != manifest["source_manifest"]
            or current["protected_diff_sha256_end"] != manifest["protected_diff_sha256"]):
        raise AssertionError("source or protected documentation changed during full run")
    current["complete"] = True
    current["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(status_path, current)
    return current


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"empty publication table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _plot_crossfit(state_rows: list[dict], direction_rows: list[dict], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selector_names = ("unrestricted", "next_use_tie", "count_tie")
    strata = sorted({(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["policy"])
                     for row in state_rows})
    if len(strata) != 8:
        raise AssertionError("expected eight trace/cell/source-policy strata")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    for axis, key in zip(axes.flat, strata):
        selected = [r for r in state_rows if (
            r["trace"], r["l1_fraction"], r["l2_multiplier"], r["policy"]) == key]
        trained = [r for r in direction_rows if (
            r["trace"], r["l1_fraction"], r["l2_multiplier"], r["policy"]) == key]
        x = np.arange(3)
        train_means = [np.mean([r["train_mean_gain_tokens"] for r in trained if r["selector"] == name])
                       for name in selector_names]
        heldout_means = [np.mean([r["symmetric_heldout_mean_gain_tokens"] for r in selected
                                 if r["selector"] == name]) for name in selector_names]
        axis.bar(x - .18, train_means, width=.36, label="training", color="#a5a9b3")
        axis.bar(x + .18, heldout_means, width=.36, label="held-out", color="#2865a8")
        axis.axhline(0, color="black", lw=.7)
        axis.set_xticks(x, ["All actions", "Next-use tie", "Count tie"], rotation=18, ha="right")
        axis.set_title(f"{key[0].replace('_trace', '')} | L1 {float(key[1]):g}, L2×{key[2]} | {key[3]}", fontsize=10)
        axis.set_ylabel("Mean paired gain vs actual (avoided tokens)")
        axis.legend(fontsize=8)
    fig.suptitle("Selection on eight streams; evaluation on the other eight (40 fixed states)", fontsize=13)
    fig.tight_layout()
    fig.savefig(target / "crossfit_train_heldout.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    for axis, key in zip(axes.flat, strata):
        trained = [r for r in direction_rows if (
            r["trace"], r["l1_fraction"], r["l2_multiplier"], r["policy"]) == key]
        optimism = [np.mean([r["train_mean_gain_tokens"] - r["heldout_mean_gain_tokens"]
                             for r in trained if r["selector"] == name]) for name in selector_names]
        axis.bar(range(3), optimism, color="#a45328")
        axis.axhline(0, color="black", lw=.7)
        axis.set_xticks(range(3), ["All actions", "Next-use tie", "Count tie"], rotation=18, ha="right")
        axis.set_title(f"{key[0].replace('_trace', '')} | L1 {float(key[1]):g}, L2×{key[2]} | {key[3]}", fontsize=10)
        axis.set_ylabel("Training minus held-out gain (avoided tokens)")
    fig.suptitle("Selection optimism diagnostic; the signed held-out gain is the primary result", fontsize=13)
    fig.tight_layout()
    fig.savefig(target / "selection_optimism.png", dpi=180)
    plt.close(fig)


def aggregate(output_dir: Path, paper_dir: Path | None = None) -> dict:
    manifest, digest = _load_manifest(output_dir)
    status = json.loads((output_dir / "full_status.json").read_text(encoding="utf-8"))
    if (not status.get("complete") or status["manifest_sha256"] != digest
            or status["completed_lineages"] != 40
            or status["expected_branch_count"] != 10_888):
        raise AssertionError("cannot aggregate incomplete full run")
    tables = {name: [] for name in (
        "selected_decisions", "action_stream", "directional", "heldout_stream",
        "state_crossfit", "actionwise", "stream_hindsight", "fixed_selectors",
        "directional_contrasts", "state_contrasts", "state_diagnostics", "stratum_summary",
    )}
    branch_total = 0
    for row in manifest["lineages"]:
        marker = _verify_marker(output_dir, row, digest, smoke=False)
        if marker is None:
            raise AssertionError(f"missing completed lineage: {row['slug']}")
        selected = row["selected"][0]
        metadata = marker["metadata"]
        if (metadata["candidate_indices"] != selected["candidate_indices"]
                or metadata["scores"] != selected["scores"]
                or metadata["actual_index"] != selected["victim_index"]
                or metadata["next_use_delta_ms"] != selected["next_use_delta_ms"]
                or metadata["count_within_h"] != selected["count_within_h"]):
            raise AssertionError("completed marker snapshot metadata mismatch")
        actual = int(metadata["actual_index"])
        common = {
            "trace": row["trace"], "l1_fraction": row["l1_fraction"],
            "l2_multiplier": row["l2_multiplier"], "policy": row["policy"],
            "seed": row["seed"], "lineage": row["lineage"],
            "selection_hash": selected["selection_hash"], "hash_rank": 0,
            "group_index": selected["group_index"], "ordinal": selected["ordinal"],
            "timestamp_ms": selected["timestamp_ms"],
            "candidate_count": metadata["candidate_count"],
            "actual_index": actual,
            "arrival_rejection": metadata["arriving_index"] == actual,
        }
        tables["selected_decisions"].append({**common,
            "population_sha256": row["population_sha256"],
            "model_sha256": row["model_sha256"],
            "old_captured_branch_sha256": row["old_captured"]["branch_sha256"],
            "next_use_min_tie_count": len(metadata["label_ties"]["next_use"]),
            "count_min_tie_count": len(metadata["label_ties"]["count"]),
            "label_tie_sets_equal": metadata["label_ties"]["next_use"] == metadata["label_ties"]["count"],
        })
        paths = _expected_relative_paths(row, smoke=False)
        branch_total += len(paths)
        branches = {(r, a): json.loads((output_dir / path).read_text(encoding="utf-8"))
                    for (r, a), path in paths.items()}
        q600 = {r: [branches[(r, a)]["reward"]["q600"] for a in range(metadata["candidate_count"])]
                for r in STREAM_IDS}
        qend = {r: [branches[(r, a)]["reward"]["qend"] for a in range(metadata["candidate_count"])]
                for r in STREAM_IDS}
        analysis = crossfit_analysis(
            q600, qend, actual, metadata["last_group"], metadata["label_ties"],
            {name: metadata["selectors"][name] for name in ("next_use", "count", "lru", "lfu")},
        )
        tables["state_diagnostics"].append({**common, **analysis["diagnostics"]})
        for r in STREAM_IDS:
            for a in range(metadata["candidate_count"]):
                branch = branches[(r, a)]
                reward = branch["reward"]
                tables["action_stream"].append({**common,
                    "stream_id": r, "fold": "A" if r in FOLD_A else "B",
                    "stream_seed": row["stream_seeds"][str(r)], "action_index": a,
                    "candidate_state_index": metadata["candidate_indices"][a],
                    "is_actual": a == actual,
                    "is_arrival": a == metadata["arriving_index"],
                    "last_group": metadata["last_group"][a],
                    "frequency": metadata["frequency"][a],
                    "next_use_delta_ms": metadata["next_use_delta_ms"][a],
                    "count_within_h": metadata["count_within_h"][a],
                    "q600_tokens": reward["q600"], "qend_tokens": reward["qend"],
                    "l2_q600_tokens": reward["l2_q600"],
                    "l2_qend_tokens": reward["l2_qend"],
                    "paired_gain_vs_actual_tokens": reward["q600"] - q600[r][actual],
                    "paired_end_gain_vs_actual_tokens": reward["qend"] - qend[r][actual],
                    "branch_content_sha256": branch["content_sha256"],
                })
        for name, key in (("directional", "directions"), ("heldout_stream", "per_stream"),
                          ("state_crossfit", "state"), ("actionwise", "actionwise"),
                          ("stream_hindsight", "hindsight"), ("fixed_selectors", "fixed")):
            tables[name].extend({**common, **item} for item in analysis[key])

        directions = {(d["selector"], d["direction"]): d for d in analysis["directions"]}
        states = {d["selector"]: d for d in analysis["state"]}
        for direction in ("A_to_B", "B_to_A"):
            eval_ids = FOLD_B if direction == "A_to_B" else FOLD_A
            unrestricted = directions[("unrestricted", direction)]
            for comparator in ("next_use_tie", "count_tie"):
                restricted = directions[(comparator, direction)]
                tables["directional_contrasts"].append({**common, "direction": direction,
                    "contrast": f"unrestricted_minus_{comparator}",
                    "heldout_mean_paired_difference_tokens": unrestricted["heldout_mean_gain_tokens"]
                                                      - restricted["heldout_mean_gain_tokens"],
                    "heldout_end_mean_paired_difference_tokens": unrestricted["heldout_end_mean_gain_tokens"]
                                                          - restricted["heldout_end_mean_gain_tokens"],
                    "choices_identical": unrestricted["selected_index"] == restricted["selected_index"]})
            for source_name in ("unrestricted", "next_use_tie", "count_tie"):
                selected_action = directions[(source_name, direction)]["selected_index"]
                for fixed_name in ("next_use", "count", "lru", "lfu"):
                    fixed_action = metadata["selectors"][fixed_name]
                    tables["directional_contrasts"].append({**common, "direction": direction,
                        "contrast": f"{source_name}_minus_fixed_{fixed_name}",
                        "heldout_mean_paired_difference_tokens": sum(
                            q600[r][selected_action] - q600[r][fixed_action] for r in eval_ids) / 8,
                        "heldout_end_mean_paired_difference_tokens": sum(
                            qend[r][selected_action] - qend[r][fixed_action] for r in eval_ids) / 8,
                        "choices_identical": selected_action == fixed_action})
        for comparator in ("next_use_tie", "count_tie"):
            tables["state_contrasts"].append({**common,
                "contrast": f"unrestricted_minus_{comparator}",
                "symmetric_heldout_mean_paired_difference_tokens":
                    states["unrestricted"]["symmetric_heldout_mean_gain_tokens"]
                    - states[comparator]["symmetric_heldout_mean_gain_tokens"],
                "symmetric_heldout_end_mean_paired_difference_tokens":
                    states["unrestricted"]["symmetric_heldout_end_mean_gain_tokens"]
                    - states[comparator]["symmetric_heldout_end_mean_gain_tokens"],
                "choices_identical_both_folds":
                    states["unrestricted"]["selected_A_index"] == states[comparator]["selected_A_index"]
                    and states["unrestricted"]["selected_B_index"] == states[comparator]["selected_B_index"],
            })
    if (len(tables["selected_decisions"]) != 40 or branch_total != 10_888
            or len(tables["action_stream"]) != 10_848
            or len(tables["directional"]) != 240
            or len(tables["heldout_stream"]) != 1920):
        raise AssertionError("aggregate table cardinality mismatch")
    groups = defaultdict(list)
    for item in tables["state_crossfit"]:
        key = (item["trace"], item["l1_fraction"], item["l2_multiplier"],
               item["policy"], item["selector"])
        groups[key].append(item)
    if len(groups) != 24 or any(len(items) != 5 for items in groups.values()):
        raise AssertionError("stratum/selector must contain five fixed states")
    for key, items in sorted(groups.items()):
        values = [item["symmetric_heldout_mean_gain_tokens"] for item in items]
        tables["stratum_summary"].append({
            **dict(zip(("trace", "l1_fraction", "l2_multiplier", "policy", "selector"), key)),
            "states": len(items), "mean_symmetric_heldout_gain_tokens": sum(values) / len(values),
            "positive_states": sum(value > 0 for value in values),
            "zero_states": sum(value == 0 for value in values),
            "negative_states": sum(value < 0 for value in values),
            "mean_symmetric_heldout_end_gain_tokens": sum(
                item["symmetric_heldout_end_mean_gain_tokens"] for item in items) / len(items),
            "fold_choice_agreement_count": sum(item["fold_choices_agree"] for item in items),
        })
    target = paper_dir if paper_dir is not None else output_dir / "tables"
    if target.exists():
        raise FileExistsError(f"new publication directory required: {target}")
    staging = target.with_name(target.name + f".staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"publication staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for name, rows in tables.items():
            _write_csv(staging / f"{name}.csv", rows)
        _plot_crossfit(tables["state_crossfit"], tables["directional"], staging)
        (staging / "README.md").write_text(
            "# Fresh continuation RNG cross-fit\n\n"
            "Forty frozen counterfactual snapshots were each evaluated under sixteen new "
            "post-draw L2 sampling seeds. The primary statistic is the symmetric mean of "
            "A-to-B and B-to-A held-out paired gain versus the original action, in avoided "
            "prefill tokens within 600 seconds. Negative gains are retained. Trace-end values "
            "evaluate the same 600-second-selected actions. Each direction's interval uses "
            "only eight evaluation streams and is conditional; the symmetric statistic has "
            "no naive pooled confidence interval.\n\n"
            "selected_decisions.csv identifies the fixed states. action_stream.csv holds all "
            "10,848 fresh action/stream outcomes with candidate IDs. directional.csv and "
            "heldout_stream.csv contain all cross-fit selections and paired evaluations. "
            "state_crossfit.csv and stratum_summary.csv summarize the primary result. "
            "directional_contrasts.csv and state_contrasts.csv compare unrestricted, exact-tie "
            "restricted, and fixed selectors on identical held-out streams. "
            "actionwise.csv reports both Q variation and paired-action variation. "
            "stream_hindsight.csv records stream-specific hindsight maxima separately. "
            "state_diagnostics.csv distinguishes mean of stream-wise maxima from the "
            "maximum action-wise mean, which is fitted using all sixteen streams. "
            "crossfit_train_heldout.png plots training and held-out means; "
            "selection_optimism.png isolates their difference. These states and traces do "
            "not provide a deployment-wide or workload-general improvement estimate.\n",
            encoding="utf-8",
        )
        _load_manifest(output_dir)
        if (source_manifest() != manifest["source_manifest"]
                or base.protected_diff_hashes() != manifest["protected_diff_sha256"]):
            raise AssertionError("source or protected documentation changed before publication")
        summary = {
            "phase": NAMESPACE, "manifest_sha256": digest,
            "code_commit": manifest["code_commit"],
            "plan_commit": manifest["plan_commit"],
            "plan_sha256": manifest["plan_sha256"],
            "source_manifest_start": manifest["source_manifest"],
            "source_manifest_end": source_manifest(),
            "protected_diff_sha256_start": manifest["protected_diff_sha256"],
            "protected_diff_sha256_end": base.protected_diff_hashes(),
            "old_paper_config_sha256": OLD_PAPER_CONFIG_SHA256,
            "old_manifest_sha256": OLD_MANIFEST_SHA256,
            "fold_A": list(FOLD_A), "fold_B": list(FOLD_B),
            "horizon_primary_seconds": 600,
            "horizon_secondary": "trace end; actions selected using Q600",
            "selected_states": 40, "candidate_actions": 678,
            "fresh_branches": 10_848, "captured_actual_validation_branches": 40,
            "full_status": status,
            "files_sha256": {path.name: sha256_path(path) for path in sorted(staging.iterdir())
                             if path.is_file()},
        }
        atomic_json(staging / "run_config.json", summary)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return summary


def status(output_dir: Path) -> dict:
    manifest, digest = _load_manifest(output_dir)
    completed = sum(_verify_marker(output_dir, row, digest, smoke=False) is not None
                    for row in manifest["lineages"])
    full_path = output_dir / "full_status.json"
    full_status = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else None
    return {"manifest_sha256": digest, "completed_lineages": completed,
            "remaining_lineages": 40 - completed, "full_status": full_status}


def _detach_full(output_dir: Path, workers: int) -> dict:
    manifest, digest = _load_manifest(output_dir)
    smoke_row = next(row for row in manifest["lineages"] if (
        row["trace"], float(row["l1_fraction"]), int(row["l2_multiplier"]),
        row["policy"], int(row["seed"])) == SMOKE_IDENTITY)
    if _verify_marker(output_dir, smoke_row, digest, smoke=True) is None:
        raise RuntimeError("completed smoke is required before detached full run")
    record_path = output_dir / "launcher.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        pid = int(previous["pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"existing launcher process is still running: {pid}")
    log_path = output_dir / "logs/full_launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "full",
             "--output-dir", str(output_dir), "--workers", str(workers)],
            cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {"pid": process.pid, "manifest_sha256": digest,
              "log": str(log_path.resolve()),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(record_path, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "full", "status", "aggregate", "_worker"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--slug")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output_dir)
        display = {"selected_count": result["snapshot_count"],
                   "candidate_actions": result["action_count"],
                   "manifest": str(output_dir / "selection_manifest.json")}
    elif args.command == "_worker":
        if not args.slug:
            raise ValueError("internal worker requires --slug")
        display = _worker_one(output_dir, args.slug, args.smoke)
    elif args.command == "status":
        display = status(output_dir)
    elif args.command == "full" and args.detach:
        if not 1 <= args.workers <= 6:
            raise ValueError("workers must be in 1..6")
        display = _detach_full(output_dir, args.workers)
    else:
        with base._run_lock(output_dir):
            if args.command == "smoke":
                display = smoke(output_dir)
            elif args.command == "full":
                display = full(output_dir, args.workers)
            elif args.command == "aggregate":
                display = aggregate(output_dir, args.paper_dir.resolve() if args.paper_dir else None)
            else:
                raise AssertionError("unsupported command")
    print(json.dumps(display, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
