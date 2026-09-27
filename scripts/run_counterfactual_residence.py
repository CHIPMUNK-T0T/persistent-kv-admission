#!/usr/bin/env python3
"""Frozen E/Z residence diagnostic. Execute only after source review and commit."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from persistent_kv_admission.counterfactual import atomic_json, terminal_state_digest
from persistent_kv_admission.counterfactual_randomness import derive_stream_seed
from persistent_kv_admission.counterfactual_residence import (
    HORIZON_MS, STREAMS, ResidenceForkController, fixed_pair, verified_content,
)
from persistent_kv_admission.decisionpop import RawFeatureScorer, state_indices
from persistent_kv_admission.onpolicy import deserialize_ranker, sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

PLAN = REPOSITORY / "docs/counterfactual-residence-plan.md"
PLAN_COMMIT = "780f301"
PHASE = "persistent-kv-counterfactual-residence-v1"
OLD_PAPER_SHA256 = "3b9f48cf33132eea87f7fa6d937ac1537058b99cf8b8f074ac14afa1f488b63e"
OLD_MANIFEST_SHA256 = "5eea63ccac9ecd09648aca61b77ad9b43c8aa1f38b9edfd5cabc387cdcac0667"
OLD_RAW = Path("results/counterfactual_randomness_001")
OLD_PAPER = Path("results/paper/counterfactual_randomness_001")
EXECUTION_FILES = (
    "docs/counterfactual-residence-plan.md",
    "scripts/run_counterfactual_residence.py",
    "src/persistent_kv_admission/counterfactual_residence.py",
    "tests/test_counterfactual_residence.py",
)
SMOKE_SLUG = "conversation_trace__l1_0.0025__l2x_1__pi0__seed_0"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_files() -> tuple[str, ...]:
    tracked = _git("ls-files", "src", "scripts", "tests").splitlines()
    return tuple(sorted(set(tracked) | set(EXECUTION_FILES)))


def source_manifest() -> dict:
    files = {name: sha256_path(REPOSITORY / name) for name in _source_files()}
    return {"files": files, "sha256": _hash_json(files)}


def require_committed_source() -> str:
    paths = _source_files()
    if _git("diff", "HEAD", "--name-only", "--", *paths) or _git(
        "ls-files", "--others", "--exclude-standard", "--", *paths
    ):
        raise RuntimeError("residence execution source is uncommitted")
    if _git("log", "-1", "--format=%h", "--", str(PLAN.relative_to(REPOSITORY))) != PLAN_COMMIT:
        raise RuntimeError("residence plan commit mismatch")
    return _git("rev-parse", "HEAD")


def _source_path(root: Path, path: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise AssertionError(f"frozen input escapes --source-root: {path}")
    return resolved


def _read_verified_json(path: Path, digest: str) -> dict:
    if sha256_path(path) != digest:
        raise AssertionError(f"frozen SHA256 mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _old_branch_rel(row: dict, stream: int, action: int) -> Path:
    selected = row["selected"][0]
    return Path("branches") / row["slug"] / f"{selected['selection_hash']}__r{stream}__a{action}.json"


def _old_branch(root: Path, row: dict, stream: int, action: int, digest: str) -> dict:
    path = root / OLD_RAW / _old_branch_rel(row, stream, action)
    branch = _read_verified_json(path, digest)
    if (not verified_content(branch) or branch.get("selection_hash") != row["selected"][0]["selection_hash"]
            or branch.get("replicate") != stream or branch.get("action_index") != action):
        raise AssertionError(f"frozen branch identity/content mismatch: {path}")
    return branch


def _old_context(root: Path) -> tuple[dict, dict, str, str]:
    paper_path = root / OLD_PAPER / "run_config.json"
    paper_sha = sha256_path(paper_path)
    if paper_sha != OLD_PAPER_SHA256:
        raise AssertionError("frozen old paper config SHA256 mismatch")
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    if paper.get("phase") != "persistent-kv-counterfactual-rng-crossfit-v1":
        raise AssertionError("wrong frozen paper phase")
    for name, digest in paper["files_sha256"].items():
        if sha256_path(root / OLD_PAPER / name) != digest:
            raise AssertionError(f"frozen published file changed: {name}")
    raw_path = root / OLD_RAW / "selection_manifest.json"
    raw_sha = sha256_path(raw_path)
    if raw_sha != OLD_MANIFEST_SHA256 or raw_sha != paper["manifest_sha256"]:
        raise AssertionError("raw selection manifest differs from published run")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    full = json.loads((root / OLD_RAW / "full_status.json").read_text(encoding="utf-8"))
    if (raw["phase"] != paper["phase"] or raw["code_commit"] != paper["code_commit"]
            or raw["snapshot_count"] != 40 or raw["stream_count"] != 16
            or len(raw["lineages"]) != 40 or not full.get("complete")
            or full["manifest_sha256"] != raw_sha or full["completed_lineages"] != 40
            or full["expected_branch_count"] != 10_888):
        raise AssertionError("frozen full run is incomplete or inconsistent")
    return paper, raw, paper_sha, raw_sha


def prepare(output_dir: Path, source_root: Path) -> dict:
    head = require_committed_source()
    if output_dir.exists():
        raise FileExistsError(f"new output directory required: {output_dir}")
    root = source_root.resolve(strict=True)
    paper, raw, paper_sha, raw_sha = _old_context(root)
    with (root / OLD_PAPER / "action_stream.csv").open(newline="", encoding="utf-8") as handle:
        published_actions = {}
        for published_row in csv.DictReader(handle):
            key = (published_row["selection_hash"], int(published_row["stream_id"]),
                   int(published_row["action_index"]))
            if key in published_actions:
                raise AssertionError("duplicate published action-stream row")
            published_actions[key] = published_row
    if len(published_actions) != 10_848:
        raise AssertionError("published action-stream cardinality changed")
    rows = []
    for old in raw["lineages"]:
        selected = old["selected"][0]
        marker_path = root / OLD_RAW / "lineages" / f"{old['slug']}.json"
        marker_sha = sha256_path(marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (not marker.get("complete") or marker.get("manifest_sha256") != raw_sha
                or marker.get("selection_hash") != selected["selection_hash"]
                or marker.get("candidate_indices") != selected["candidate_indices"]
                or marker.get("expected_branches") != 1 + len(selected["candidate_indices"]) * 16):
            raise AssertionError(f"frozen lineage marker changed: {old['slug']}")
        metadata = marker["metadata"]
        if (metadata["candidate_indices"] != selected["candidate_indices"]
                or metadata["count_within_h"] != selected["count_within_h"]
                or metadata["next_use_delta_ms"] != selected["next_use_delta_ms"]):
            raise AssertionError("frozen snapshot identity/labels changed")
        e, z = fixed_pair(metadata)
        for action in range(metadata["candidate_count"]):
            public = published_actions.get((selected["selection_hash"], 1, action))
            if (public is None or int(public["candidate_state_index"]) != selected["candidate_indices"][action]
                    or int(public["last_group"]) != metadata["last_group"][action]
                    or int(public["count_within_h"]) != metadata["count_within_h"][action]):
                raise AssertionError("frozen candidate identity/labels differ from published table")
        # Choose E/Z before any old branch reward is loaded.
        for path_key in ("model_path", "population_path"):
            _source_path(root, old[path_key])
        for path_key, digest_key in (("model_path", "model_sha256"),
                                     ("population_path", "population_sha256")):
            if sha256_path(Path(old[path_key])) != old[digest_key]:
                raise AssertionError(f"frozen {path_key} changed")
        trace_ref = raw["trace_files"][old["trace"]]
        if sha256_path(_source_path(root, trace_ref["path"])) != trace_ref["sha256"]:
            raise AssertionError("frozen trace changed")
        seeds = {str(s): str(derive_stream_seed(old["lineage"], selected["group_index"],
                                                  selected["ordinal"], s)) for s in STREAMS}
        if old["stream_seeds"] != seeds:
            raise AssertionError("frozen fresh stream seeds changed")
        refs = {}
        for stream in STREAMS:
            for action in (e, z):
                relative = _old_branch_rel(old, stream, action)
                digest = marker["branch_sha256"].get(str(relative))
                if digest is None:
                    raise AssertionError("frozen old branch reference missing")
                old_branch = _old_branch(root, old, stream, action, digest)
                public = published_actions.get((selected["selection_hash"], stream, action))
                if (public is None or old_branch["content_sha256"] != public["branch_content_sha256"]
                        or any(old_branch["reward"][old_name] != int(public[public_name])
                               for old_name, public_name in (("q600", "q600_tokens"),
                                                             ("qend", "qend_tokens"),
                                                             ("l2_q600", "l2_q600_tokens"),
                                                             ("l2_qend", "l2_qend_tokens")))):
                    raise AssertionError("old raw branch differs from hash-pinned published action table")
                refs[f"{stream}:{action}"] = digest
        actual = int(selected["victim_index"])
        captured_rel = _old_branch_rel(old, 0, actual)
        captured_sha = marker["branch_sha256"].get(str(captured_rel))
        if captured_sha is None:
            raise AssertionError("frozen captured reference missing")
        captured = _old_branch(root, old, 0, actual, captured_sha)
        if captured["reward"] != marker["parent_reward"]:
            raise AssertionError("frozen parent reward mismatch")
        original_ref = old["old_captured"]
        original_path = _source_path(root, original_ref["branch_path"])
        original = _read_verified_json(original_path, original_ref["branch_sha256"])
        if not verified_content(original) or any(
            captured[field] != original[field]
            for field in ("reward", "terminal_state_digest", "end_result")
        ):
            raise AssertionError("fresh captured control differs from original captured branch")
        rows.append({"slug": old["slug"], "trace": old["trace"], "lineage": old["lineage"],
                     "l1_fraction": old["l1_fraction"], "l2_multiplier": old["l2_multiplier"],
                     "policy": old["policy"], "seed": old["seed"],
                     "selected": old["selected"], "pair": {"E": e, "Z": z},
                     "last_group": [metadata["last_group"][e], metadata["last_group"][z]],
                     "focal_indices": [selected["candidate_indices"][e], selected["candidate_indices"][z]],
                     "model_path": old["model_path"], "model_sha256": old["model_sha256"],
                     "population_path": old["population_path"],
                     "population_sha256": old["population_sha256"],
                     "stream_seeds": seeds, "old_marker_sha256": marker_sha,
                     "old_captured_sha256": captured_sha, "old_branch_sha256": refs})
    if len(rows) != 40 or sum(len(row["old_branch_sha256"]) for row in rows) != 1280:
        raise AssertionError("fixed residence grid cardinality mismatch")
    manifest = {"phase": PHASE, "code_commit": head, "source_manifest": source_manifest(),
                "plan_commit": PLAN_COMMIT, "plan_sha256": sha256_path(PLAN),
                "source_root": str(root), "old_paper_config_sha256": paper_sha,
                "old_manifest_sha256": raw_sha, "old_code_commit": paper["code_commit"],
                "trace_files": raw["trace_files"], "onpolicy_config_sha256": raw["onpolicy_config_sha256"],
                "lineages": rows, "snapshot_count": 40, "stream_count": 16,
                "expected_branches": 1280,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "selection_manifest.json", manifest)
    atomic_json(output_dir / "prepare_status.json", {
        "complete": True, "manifest_sha256": sha256_path(output_dir / "selection_manifest.json"),
        "selected_count": 40, "expected_branches": 1280})
    return manifest


def _load_manifest(output_dir: Path, source_root: Path) -> tuple[dict, str]:
    path = output_dir / "selection_manifest.json"
    digest = sha256_path(path)
    status = json.loads((output_dir / "prepare_status.json").read_text(encoding="utf-8"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not status.get("complete") or status["manifest_sha256"] != digest
            or manifest["phase"] != PHASE or manifest["code_commit"] != require_committed_source()
            or manifest["source_manifest"] != source_manifest()
            or manifest["plan_sha256"] != sha256_path(PLAN)
            or manifest["source_root"] != str(source_root.resolve())
            or len(manifest["lineages"]) != 40 or manifest["expected_branches"] != 1280):
        raise AssertionError("residence run manifest/source changed")
    paper, raw, paper_sha, raw_sha = _old_context(source_root)
    if (paper_sha != manifest["old_paper_config_sha256"]
            or raw_sha != manifest["old_manifest_sha256"]
            or raw["trace_files"] != manifest["trace_files"]
            or paper["code_commit"] != manifest["old_code_commit"]):
        raise AssertionError("frozen old context changed")
    return manifest, digest


def _new_paths(row: dict, smoke: bool) -> dict[tuple[int, int], Path]:
    selected = row["selected"][0]
    streams = (1,) if smoke else STREAMS
    return {(s, a): Path("branches") / row["slug"] /
            f"{selected['selection_hash']}__r{s}__a{a}.json"
            for s in streams for a in (row["pair"]["E"], row["pair"]["Z"])}


def _verified_new_branch(path: Path, row: dict, digest: str, stream: int, action: int) -> dict:
    branch = json.loads(path.read_text(encoding="utf-8"))
    if (not verified_content(branch) or branch.get("run_fingerprint") != digest
            or branch.get("selection_hash") != row["selected"][0]["selection_hash"]
            or branch.get("replicate") != stream or branch.get("action_index") != action
            or not isinstance(branch.get("telemetry"), dict)):
        raise AssertionError(f"new branch identity/content mismatch: {path}")
    return branch


def _worker_one(output_dir: Path, source_root: Path, slug: str, smoke: bool) -> dict:
    manifest, digest = _load_manifest(output_dir, source_root)
    matches = [row for row in manifest["lineages"] if row["slug"] == slug]
    if len(matches) != 1 or smoke and slug != SMOKE_SLUG:
        raise ValueError("unknown or wrong smoke lineage")
    row = matches[0]
    trace_ref = manifest["trace_files"][row["trace"]]
    trace_path = _source_path(source_root, trace_ref["path"])
    if sha256_path(trace_path) != trace_ref["sha256"]:
        raise AssertionError("frozen trace changed")
    trace = load_mooncake_trace(trace_path)
    if trace.name != row["trace"]:
        raise AssertionError("trace identity mismatch")
    config_path = source_root / "results/paper/onpolicy_learning/onpolicy_learning_config.json"
    if sha256_path(config_path) != manifest["onpolicy_config_sha256"]:
        raise AssertionError("frozen onpolicy config changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ranker = deserialize_ranker(row["model_path"], row["model_sha256"])
    scorer = RawFeatureScorer(trace, ranker)
    selected = row["selected"][0]
    pair = (row["pair"]["E"], row["pair"]["Z"])
    controller = ResidenceForkController(
        trace, scorer, state_indices(trace), selected,
        output_dir / "branches" / slug, digest, pair,
        {int(k): int(v) for k, v in row["stream_seeds"].items()}, smoke=smoke)
    fraction = float(row["l1_fraction"])
    multiplier = int(row["l2_multiplier"])
    working_set = int(config["working_set_bytes"][row["trace"]])
    l1_bytes = max(1, round(working_set * fraction))
    l2_bytes = max(1, round(working_set * fraction * multiplier))
    started = time.monotonic()
    try:
        result = run_two_tier(
            trace, "lru", l1_bytes, l2_bytes, "learned", hit_model="tree",
            closure="union", bytes_per_token=2048, size_model="packed",
            measure_from_ms=float(config["measure_from_ms"][row["trace"]]),
            occurrence_groups=_occurrence_groups(trace),
            l2_eviction="sampled", l2_sample_width=16, l2_seed=int(row["seed"]),
            l2_scorer=scorer, l2_arm="onpolicy", l2_request_hook=controller.on_request,
            l2_decision_hook=controller.on_decision, l2_removal_hook=controller.on_removal,
            victim_hook=controller.on_victim, l2_arrival_protection="none",
            l2_override_hook=controller)
        if controller.is_child:
            controller.finish_child(result)
            os._exit(0)
    except BaseException as error:
        if controller.is_child:
            controller.fail_child(error)
            os._exit(1)
        raise
    if len(controller.snapshots) != 1:
        raise AssertionError("selected state not replayed exactly once")
    snapshot = controller.snapshots[0]
    if fixed_pair(snapshot["metadata"]) != pair:
        raise AssertionError("live fixed pair changed")
    old_marker_path = source_root / OLD_RAW / "lineages" / f"{slug}.json"
    old_marker = _read_verified_json(old_marker_path, row["old_marker_sha256"])
    old_raw_row = {"slug": slug, "selected": row["selected"]}
    old_actual = int(selected["victim_index"])
    old_captured = _old_branch(source_root, old_raw_row, 0, old_actual,
                               row["old_captured_sha256"])
    if (snapshot["parent_reward"].row() != old_marker["parent_reward"]
            or snapshot["parent_reward"].row() != old_captured["reward"]
            or terminal_state_digest(controller.l1, controller.l2, scorer, controller.counters)
               != old_captured["terminal_state_digest"]
            or result.as_row() != old_captured["end_result"]):
        raise AssertionError("uninterrupted parent differs from frozen old run")
    paths = _new_paths(row, smoke)
    for (stream, action), relative in paths.items():
        branch = _verified_new_branch(output_dir / relative, row, digest, stream, action)
        old = _old_branch(source_root, old_raw_row, stream, action,
                          row["old_branch_sha256"][f"{stream}:{action}"])
        if branch["reward"] != old["reward"]:
            raise AssertionError(f"instrumented branch reward/digest changed: {relative}")
    branch_hashes = {str(rel): sha256_path(output_dir / rel) for rel in paths.values()}
    marker = {"complete": True, "smoke": smoke, "slug": slug,
              "manifest_sha256": digest, "selection_hash": selected["selection_hash"],
              "pair": row["pair"], "expected_branches": len(paths),
              "branch_sha256": branch_hashes, "old_reward_digest_checked": True,
              "parent_checked": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    marker_path = output_dir / ("smoke_marker.json" if smoke else f"lineages/{slug}.json")
    if marker_path.exists():
        raise FileExistsError(f"completed marker already exists: {marker_path}")
    atomic_json(marker_path, marker)
    return marker


def _verify_marker(output_dir: Path, row: dict, digest: str, smoke: bool) -> dict | None:
    marker_path = output_dir / ("smoke_marker.json" if smoke else f"lineages/{row['slug']}.json")
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    paths = _new_paths(row, smoke)
    if (not marker.get("complete") or marker.get("smoke") is not smoke
            or marker.get("slug") != row["slug"] or marker.get("manifest_sha256") != digest
            or marker.get("selection_hash") != row["selected"][0]["selection_hash"]
            or marker.get("pair") != row["pair"] or marker.get("expected_branches") != len(paths)
            or not marker.get("old_reward_digest_checked") or not marker.get("parent_checked")
            or set(marker.get("branch_sha256", {})) != {str(p) for p in paths.values()}):
        raise AssertionError(f"completed marker mismatch: {marker_path}")
    for (stream, action), relative in paths.items():
        path = output_dir / relative
        if sha256_path(path) != marker["branch_sha256"][str(relative)]:
            raise AssertionError(f"completed branch SHA mismatch: {path}")
        _verified_new_branch(path, row, digest, stream, action)
    return marker


@contextlib.contextmanager
def run_lock(output_dir: Path):
    with (output_dir / "run.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _spawn_worker(output_dir: Path, source_root: Path, slug: str, smoke: bool):
    log_path = output_dir / "logs" / ("smoke.log" if smoke else f"{slug}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker",
             "--output-dir", str(output_dir), "--source-root", str(source_root),
             "--slug", slug] + (["--smoke"] if smoke else []),
            cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def smoke(output_dir: Path, source_root: Path) -> dict:
    with run_lock(output_dir):
        manifest, digest = _load_manifest(output_dir, source_root)
        row = next(row for row in manifest["lineages"] if row["slug"] == SMOKE_SLUG)
        existing = _verify_marker(output_dir, row, digest, True)
        if existing is not None:
            return existing
        process = _spawn_worker(output_dir, source_root, row["slug"], True)
        if process.wait():
            raise RuntimeError("smoke worker failed; inspect logs/smoke.log")
        result = _verify_marker(output_dir, row, digest, True)
        if result is None:
            raise AssertionError("smoke worker produced no marker")
        return result


def _available_mib() -> float:
    meminfo = Path("/proc/meminfo").read_text()
    host = int(next(line.split()[1] for line in meminfo.splitlines()
                    if line.startswith("MemAvailable:"))) / 1024.0
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if maximum.exists() and current.exists() and maximum.read_text().strip() != "max":
        host = min(host, max(0.0, (int(maximum.read_text()) - int(current.read_text())) / 1024**2))
    return host


def full(output_dir: Path, source_root: Path, workers: int) -> dict:
    if not 1 <= workers <= 6:
        raise ValueError("workers must be 1..6")
    with run_lock(output_dir):
        manifest, digest = _load_manifest(output_dir, source_root)
        smoke_row = next(row for row in manifest["lineages"] if row["slug"] == SMOKE_SLUG)
        smoke_marker = _verify_marker(output_dir, smoke_row, digest, True)
        if smoke_marker is None:
            raise RuntimeError("completed smoke required")
        child_rss = max(_verified_new_branch(output_dir / rel, smoke_row, digest,
                                             int(Path(rel).stem.split("__r")[1].split("__a")[0]),
                                             int(Path(rel).stem.split("__a")[1]))["peak_rss_mib"]
                        for rel in smoke_marker["branch_sha256"])
        parent_rss = smoke_marker["peak_rss_mib"]
        safe_workers = math.floor(0.7 * _available_mib() / max(1, parent_rss + child_rss))
        if workers > safe_workers:
            raise ValueError(f"worker count exceeds smoke memory guard: {safe_workers}")
        status_path = output_dir / "full_status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("manifest_sha256") != digest or previous.get("complete"):
                raise AssertionError("existing full status mismatch or complete")
        pending = [row for row in manifest["lineages"]
                   if _verify_marker(output_dir, row, digest, False) is None]
        status = {"phase": "full", "manifest_sha256": digest,
                  "expected_branch_count": 1280, "requested_workers": workers,
                  "safe_workers": safe_workers, "complete": False,
                  "completed_lineages": 40 - len(pending), "failed_lineages": [],
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(status_path, status)
        running = {}
        try:
            while pending or running:
                while pending and len(running) < workers:
                    row = pending.pop(0)
                    running[row["slug"]] = (_spawn_worker(output_dir, source_root, row["slug"], False), row)
                progressed = False
                for slug, (process, row) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    progressed = True
                    del running[slug]
                    if code or _verify_marker(output_dir, row, digest, False) is None:
                        status["failed_lineages"].append({"slug": slug, "exit_code": code})
                    else:
                        status["completed_lineages"] += 1
                atomic_json(status_path, status)
                if status["failed_lineages"]:
                    raise RuntimeError("lineage worker failed; inspect logs and full_status.json")
                if running and not progressed:
                    time.sleep(0.5)
        except BaseException as error:
            for process, _ in running.values():
                try:
                    os.killpg(process.pid, 15)
                except ProcessLookupError:
                    pass
            status["error"] = repr(error)
            atomic_json(status_path, status)
            raise
        if status["completed_lineages"] != 40:
            raise AssertionError("incomplete full grid")
        _load_manifest(output_dir, source_root)
        status["complete"] = True
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["source_manifest_end"] = source_manifest()
        atomic_json(status_path, status)
        return status


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_first(prefix: str, value: dict | None) -> dict:
    names = ("order", "group_index", "timestamp_ms", "lag_requests", "lag_groups", "lag_ms")
    return {f"{prefix}_{name}": None if value is None else value[name] for name in names}


def aggregate(output_dir: Path, source_root: Path, paper_dir: Path | None) -> dict:
    from persistent_kv_admission.counterfactual_residence import pair_analysis
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with run_lock(output_dir):
        manifest, digest = _load_manifest(output_dir, source_root)
        status = json.loads((output_dir / "full_status.json").read_text(encoding="utf-8"))
        if (not status.get("complete") or status.get("manifest_sha256") != digest
                or status.get("completed_lineages") != 40
                or status.get("expected_branch_count") != 1280):
            raise AssertionError("cannot aggregate incomplete full run")
        if paper_dir is not None and paper_dir.exists():
            raise FileExistsError(f"publication directory already exists: {paper_dir}")
        pair_rows = []
        divergent_rows = []
        representative = {}
        for row in manifest["lineages"]:
            _verify_marker(output_dir, row, digest, False)
            selected = row["selected"][0]
            e, z = row["pair"]["E"], row["pair"]["Z"]
            start_ms = float(selected["timestamp_ms"])
            start_group = int(selected["group_index"])
            paths = _new_paths(row, False)
            for stream in STREAMS:
                old_row = {"slug": row["slug"], "selected": row["selected"]}
                for action in (e, z):
                    _old_branch(source_root, old_row, stream, action,
                                row["old_branch_sha256"][f"{stream}:{action}"])
                e_branch = _verified_new_branch(output_dir / paths[(stream, e)], row, digest, stream, e)
                z_branch = _verified_new_branch(output_dir / paths[(stream, z)], row, digest, stream, z)
                result, events = pair_analysis(e_branch, z_branch,
                                               start_ms=start_ms, start_group=start_group)
                es = e_branch["telemetry"]["summary"]
                zs = z_branch["telemetry"]["summary"]
                common = {"trace": row["trace"], "l1_fraction": row["l1_fraction"],
                          "l2_multiplier": row["l2_multiplier"], "policy": row["policy"],
                          "seed": row["seed"], "lineage": row["lineage"],
                          "selection_hash": selected["selection_hash"],
                          "stream_id": stream, "timestamp_ms": start_ms,
                          "E_action": e, "Z_action": z,
                          "E_state_index": row["focal_indices"][0],
                          "Z_state_index": row["focal_indices"][1],
                          "E_last_group": row["last_group"][0],
                          "Z_last_group": row["last_group"][1],
                          "E_block_bytes": zs["focal_survivor_bytes"],
                          "Z_block_bytes": es["focal_survivor_bytes"],
                          "Z_minus_E_block_bytes": zs["focal_survivor_bytes"] - es["focal_survivor_bytes"],
                          "E_branch_survivor_residence_ms": es["first_residence_ms"],
                          "Z_branch_survivor_residence_ms": zs["first_residence_ms"],
                          "E_branch_survivor_residence_groups": es["first_residence_groups"],
                          "Z_branch_survivor_residence_groups": zs["first_residence_groups"],
                          "E_branch_survivor_removal_kind":
                              None if es["first_survivor_removal"] is None else es["first_survivor_removal"][3],
                          "Z_branch_survivor_removal_kind":
                              None if zs["first_survivor_removal"] is None else zs["first_survivor_removal"][3],
                          "E_branch_survivor_removal_timestamp_ms":
                              None if es["first_survivor_removal"] is None else es["first_survivor_removal"][1],
                          "Z_branch_survivor_removal_timestamp_ms":
                              None if zs["first_survivor_removal"] is None else zs["first_survivor_removal"][1],
                          "E_branch_survivor_byte_seconds600": es["resident_byte_seconds_600"],
                          "Z_branch_survivor_byte_seconds600": zs["resident_byte_seconds_600"],
                          "Z_minus_E_branch_survivor_byte_seconds600":
                              zs["resident_byte_seconds_600"] - es["resident_byte_seconds_600"],
                          "E_branch_survivor_censored600": es["censored_600"],
                          "Z_branch_survivor_censored600": zs["censored_600"],
                          "E_branch_survivor_censored_end": es["censored_end"],
                          "Z_branch_survivor_censored_end": zs["censored_end"],
                          "E_branch_survivor_immediate_removal": not es["survived_immediate_overflow"],
                          "Z_branch_survivor_immediate_removal": not zs["survived_immediate_overflow"],
                          "E_branch_initial_rounds": es["immediate_overflow_rounds"],
                          "Z_branch_initial_rounds": zs["immediate_overflow_rounds"],
                          "E_q600": e_branch["reward"]["q600"],
                          "Z_q600": z_branch["reward"]["q600"],
                          "E_qend": e_branch["reward"]["qend"],
                          "Z_qend": z_branch["reward"]["qend"],
                          "E_l2_q600": e_branch["reward"]["l2_q600"],
                          "Z_l2_q600": z_branch["reward"]["l2_q600"],
                          "E_l2_qend": e_branch["reward"]["l2_qend"],
                          "Z_l2_qend": z_branch["reward"]["l2_qend"],
                          "E_reward_digest": e_branch["reward"]["reward_digest"],
                          "Z_reward_digest": z_branch["reward"]["reward_digest"]}
                first_fields = {name: result.pop(name) for name in (
                    "first_hit", "first_reward", "first_hit600", "first_reward600")}
                flat = {**common, **result,
                        **_flatten_first("first_hit", first_fields["first_hit"]),
                        **_flatten_first("first_reward", first_fields["first_reward"]),
                        **_flatten_first("first_hit600", first_fields["first_hit600"]),
                        **_flatten_first("first_reward600", first_fields["first_reward600"])}
                pair_rows.append(flat)
                for event in events:
                    divergent_rows.append({"selection_hash": selected["selection_hash"],
                                           "stream_id": stream, **event})
                if stream == 1:
                    regime = (float(row["l1_fraction"]), int(row["l2_multiplier"]))
                    current = representative.get(regime)
                    if current is None or selected["selection_hash"] < current[0]:
                        representative[regime] = (selected["selection_hash"], events)
        if len(pair_rows) != 640:
            raise AssertionError("aggregate pair cardinality mismatch")
        raw_tables = output_dir / "aggregate"
        if raw_tables.exists():
            raise FileExistsError(f"aggregate already exists: {raw_tables}")
        raw_tables.mkdir()
        _write_csv(raw_tables / "pairs.csv", pair_rows)
        if divergent_rows:
            # Lists of state indices are JSON-encoded by csv.DictWriter; detailed branch
            # telemetry remains in ignored results for exact reconstruction.
            _write_csv(raw_tables / "divergent_requests.csv", divergent_rows)
        state_rows = []
        for row in manifest["lineages"]:
            selected = row["selected"][0]
            values = [item for item in pair_rows if item["selection_hash"] == selected["selection_hash"]]
            if len(values) != 16:
                raise AssertionError("state missing paired streams")
            state_rows.append({"trace": row["trace"], "l1_fraction": row["l1_fraction"],
                               "l2_multiplier": row["l2_multiplier"], "policy": row["policy"],
                               "lineage": row["lineage"], "selection_hash": selected["selection_hash"],
                               "mean_delta_q600": sum(v["delta_q600"] for v in values) / 16,
                               "mean_delta_qend": sum(v["delta_qend"] for v in values) / 16,
                               "positive_streams": sum(v["delta_q600"] > 0 for v in values),
                               "zero_streams": sum(v["delta_q600"] == 0 for v in values),
                               "negative_streams": sum(v["delta_q600"] < 0 for v in values),
                               "mean_differential_byte_seconds600":
                                   sum(v["Z_minus_E_branch_survivor_byte_seconds600"] for v in values) / 16})
        _write_csv(raw_tables / "states.csv", state_rows)
        strata = {}
        for value in state_rows:
            key = (value["trace"], value["l1_fraction"], value["l2_multiplier"], value["policy"])
            strata.setdefault(key, []).append(value)
        stratum_rows = []
        for key, values in sorted(strata.items()):
            if len(values) != 5:
                raise AssertionError("stratum must have five states")
            stratum_rows.append({"trace": key[0], "l1_fraction": key[1],
                                 "l2_multiplier": key[2], "policy": key[3],
                                 "states": 5, "streams_per_state": 16,
                                 "mean_state_delta_q600": sum(v["mean_delta_q600"] for v in values) / 5,
                                 "mean_state_delta_qend": sum(v["mean_delta_qend"] for v in values) / 5,
                                 "positive_states": sum(v["mean_delta_q600"] > 0 for v in values),
                                 "zero_states": sum(v["mean_delta_q600"] == 0 for v in values),
                                 "negative_states": sum(v["mean_delta_q600"] < 0 for v in values)})
        _write_csv(raw_tables / "strata.csv", stratum_rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter([v["Z_minus_E_branch_survivor_byte_seconds600"] for v in pair_rows],
                        [v["delta_q600"] for v in pair_rows], s=9, alpha=0.4)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].axvline(0, color="black", lw=0.8)
        axes[0].set_xlabel("Z branch minus E branch focal-survivor byte-seconds")
        axes[0].set_ylabel("Q600(Z) - Q600(E), tokens")
        axes[0].set_title("Descriptive paired streams; shared states/requests")
        axes[1].hist([v["mean_delta_q600"] for v in state_rows], bins=16)
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_xlabel("State mean ΔQ600 across 16 streams, tokens")
        axes[1].set_ylabel("States")
        fig.tight_layout()
        fig.savefig(raw_tables / "residence_delta.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(len(representative), 1, figsize=(9, 3.6 * len(representative)),
                                 squeeze=False)
        for axis, (regime, (selection_hash, events)) in zip(axes.flat, sorted(representative.items())):
            within = [event for event in events if event["within_600"]]
            axis.step([0] + [event["lag_ms"] / 1000 for event in within],
                      [0] + [event["cumulative_delta_q600"] for event in within], where="post")
            axis.axhline(0, color="black", lw=0.8)
            axis.set_title(f"Lowest hash in L1 {regime[0]:g}, L2×{regime[1]}: {selection_hash[:12]}, stream 1")
            axis.set_xlabel("Seconds after forced action")
            axis.set_ylabel("Cumulative ΔQ600, tokens")
        fig.tight_layout()
        fig.savefig(raw_tables / "representative_cumulative.png", dpi=180)
        plt.close(fig)
        summary = {"phase": PHASE, "manifest_sha256": digest, "complete": True,
                   "states": 40, "paired_streams": 640, "instrumented_branches": 1280,
                   "branch_rewards_and_digests_matched_old": True,
                   "rejoining": "unmeasured",
                   "overall_mean_state_delta_q600":
                       sum(v["mean_delta_q600"] for v in state_rows) / 40,
                   "overall_mean_state_delta_qend":
                       sum(v["mean_delta_qend"] for v in state_rows) / 40,
                   "representative_selection_hashes":
                       {f"{key[0]}x{key[1]}": value[0] for key, value in representative.items()}}
        atomic_json(raw_tables / "summary.json", summary)
        if paper_dir is not None:
            paper_dir.mkdir(parents=True)
            for name in ("pairs.csv", "states.csv", "strata.csv", "residence_delta.png",
                         "representative_cumulative.png"):
                (paper_dir / name).write_bytes((raw_tables / name).read_bytes())
            (paper_dir / "README.md").write_text(
                "# Fixed E/Z residence diagnostic\n\n"
                "Pairs use zero-own-reuse exact-label ties and the same frozen 16 RNG streams. "
                "ΔQ is Z minus E. `pairs.csv` contains all 640 paired streams; `states.csv` "
                "and `strata.csv` retain state-level descriptive summaries. "
                "The scatter compares post-action residence with ΔQ observationally; "
                "it does not estimate mediation. The cumulative examples use the lowest "
                "selection hash per capacity regime and stream 1. Rejoining is unmeasured. "
                "The 40 states share two fixed traces and are not independent workloads.\n",
                encoding="utf-8")
            files = {p.name: sha256_path(p) for p in paper_dir.iterdir() if p.is_file()}
            atomic_json(paper_dir / "run_config.json", {**summary, "files_sha256": files,
                        "code_commit": manifest["code_commit"],
                        "source_manifest": manifest["source_manifest"],
                        "plan_sha256": manifest["plan_sha256"],
                        "old_paper_config_sha256": manifest["old_paper_config_sha256"],
                        "old_manifest_sha256": manifest["old_manifest_sha256"],
                        "source_root": manifest["source_root"]})
        return summary


def status(output_dir: Path, source_root: Path) -> dict:
    manifest, digest = _load_manifest(output_dir, source_root)
    completed = sum(_verify_marker(output_dir, row, digest, False) is not None
                    for row in manifest["lineages"])
    return {"manifest_sha256": digest, "completed_lineages": completed,
            "expected_lineages": 40, "expected_branches": 1280,
            "smoke_complete": (output_dir / "smoke_marker.json").exists()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "full", "aggregate", "status", "_worker"))
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original checkout containing frozen ignored raw input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--slug")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if args.phase == "prepare":
        value = prepare(output_dir, source_root)
        print(json.dumps({"phase": value["phase"], "lineages": len(value["lineages"])}, sort_keys=True))
    elif args.phase == "smoke":
        print(json.dumps(smoke(output_dir, source_root), sort_keys=True))
    elif args.phase == "full":
        print(json.dumps(full(output_dir, source_root, args.workers), sort_keys=True))
    elif args.phase == "aggregate":
        print(json.dumps(aggregate(output_dir, source_root, args.paper_dir), sort_keys=True))
    elif args.phase == "status":
        print(json.dumps(status(output_dir, source_root), sort_keys=True))
    else:
        if not args.slug:
            raise ValueError("_worker requires --slug")
        print(json.dumps(_worker_one(output_dir, source_root, args.slug, args.smoke), sort_keys=True))


if __name__ == "__main__":
    main()
