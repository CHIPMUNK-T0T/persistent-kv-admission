"""Exact one-action counterfactuals for sampled L2 eviction decisions.

Selection reads only complete published decision identities. A fork made inside
the sampled eviction hook preserves the pending L1 eviction and group iterator.
Children change one victim index and then run the unchanged frozen policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import resource
import struct
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .decisionpop import _Labeller, target_column
from .onpolicy import DecisionPopulation, sha256_path

NAMESPACE = "persistent-kv-counterfactual-action-v1"
HORIZON_MS = 600_000.0


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decision_hash(lineage: str, group_index: int, ordinal: int) -> str:
    """Versioned, outcome-independent order for complete decision identities."""
    return hashlib.sha256(_canonical_bytes(
        [NAMESPACE, lineage, int(group_index), int(ordinal)]
    )).hexdigest()


def load_population_verified(path: str | Path, expected_sha256: str) -> DecisionPopulation:
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ValueError(f"population hash mismatch: {actual} != {expected_sha256}")
    return DecisionPopulation.load(path)


def select_decisions(population, lineage: str, count: int = 8) -> list[dict]:
    """Return the lowest-hash complete decisions, still ordered by hash rank."""
    if count <= 0:
        raise ValueError("selection count must be positive")
    selected = []
    seen = set()
    for _, rows in population.group_blocks():
        first = int(rows[0])
        key = (
            int(population.trace_group_index[first]),
            int(population.decision_ordinal[first]),
        )
        if key in seen:
            raise ValueError(f"duplicate decision identity {key}")
        seen.add(key)
        selected.append((decision_hash(lineage, *key), key, rows))
    if len(selected) < count:
        raise ValueError("population contains fewer complete decisions than requested")
    selected.sort(key=lambda item: (item[0], item[1]))
    result = []
    for rank, (digest, (group_index, ordinal), rows) in enumerate(selected[:count]):
        first = int(rows[0])
        victim = np.flatnonzero(population.victim[rows])
        arrival = np.flatnonzero(population.arriving[rows])
        if len(victim) != 1 or len(arrival) > 1:
            raise ValueError("selected decision lacks one victim or has multiple arrivals")
        result.append({
            "lineage": lineage,
            "hash_rank": rank,
            "selection_hash": digest,
            "group_index": group_index,
            "ordinal": ordinal,
            "timestamp_ms": float(population.timestamp_ms[first]),
            "candidate_indices": [int(value) for value in population.state_index[rows]],
            "scores": [
                [float(population.arm_score[index]), float(population.arm_tiebreak[index])]
                for index in rows
            ],
            "victim_index": int(victim[0]),
            "arriving_index": int(arrival[0]) if len(arrival) else -1,
            "features": np.asarray(population.features[rows], dtype=float).tolist(),
            "next_use_delta_ms": [
                float(value) if math.isfinite(float(value)) else None
                for value in population.next_use_delta_ms[rows]
            ],
            "count_within_h": [int(value) for value in population.count_within_h[rows]],
        })
    return result


def continuation_seed(
    lineage: str, group_index: int, ordinal: int, replicate: int
) -> int:
    if replicate not in (1, 2):
        raise ValueError("sensitivity replicate must be 1 or 2")
    payload = _canonical_bytes(
        [NAMESPACE, "continuation", lineage, int(group_index), int(ordinal), replicate]
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class WindowReward:
    """Per-request suffix reward with inclusive 600-second and trace endpoints."""

    def __init__(self, trace, start_ms: float, horizon_ms: float = HORIZON_MS) -> None:
        self.trace = trace
        self.start_ms = float(start_ms)
        self.end_ms = self.start_ms + float(horizon_ms)
        self.q600 = 0
        self.qend = 0
        self.l2_q600 = 0
        self.l2_qend = 0
        self.requested600 = 0
        self.requested_end = 0
        self.requests600 = 0
        self.requests_end = 0
        self._digest = hashlib.sha256()

    def on_request(
        self, ids, prefix, l2_hits, present, timestamp_ms, group_index, measured
    ) -> None:
        del present, measured
        if timestamp_ms <= self.start_ms:
            return
        states = self.trace.states
        l1_tokens = states[ids[prefix - 1]].prefix_tokens if prefix else 0
        l2_tokens = sum(states[ids[index]].block_tokens for index in l2_hits)
        requested = sum(states[state_id].block_tokens for state_id in ids)
        value = l1_tokens + l2_tokens
        self.qend += value
        self.l2_qend += l2_tokens
        self.requested_end += requested
        self.requests_end += 1
        self._digest.update(struct.pack(
            "<qqqqqq", int(group_index), int(self.requests_end), int(requested),
            int(l1_tokens), int(l2_tokens), int(prefix)
        ))
        if timestamp_ms <= self.end_ms:
            self.q600 += value
            self.l2_q600 += l2_tokens
            self.requested600 += requested
            self.requests600 += 1

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()

    def row(self) -> dict:
        return {
            "q600": self.q600,
            "qend": self.qend,
            "l2_q600": self.l2_q600,
            "l2_qend": self.l2_qend,
            "requested600": self.requested600,
            "requested_end": self.requested_end,
            "requests600": self.requests600,
            "requests_end": self.requests_end,
            "reward_digest": self.digest,
        }


def _normal(value):
    if isinstance(value, dict):
        return tuple((key, _normal(item)) for key, item in value.items())
    if isinstance(value, (tuple, list, deque)):
        return tuple(_normal(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_normal(item) for item in value))
    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (str, int, float, bool, type(None), bytes)):
        return value
    raise TypeError(f"unsupported terminal audit value: {type(value).__name__}")


def terminal_state_digest(l1, l2, scorer, counters) -> str:
    """Hash all live mutable replay state used by future decisions and accounting.

    The explicit exclusion lists contain only immutable input references,
    callbacks, and shared aliases. New mutable fields fail closed.
    """
    l1_exclude = {
        "trace", "occurrence_groups", "scorer", "decision_hook", "evict_hook"
    }
    l2_exclude = {
        "trace", "groups", "state_bytes", "scorer", "decision_hook",
        "override_hook", "removal_hook", "protection_hook", "frequency", "last_group"
    }
    history_exclude = {"trace"}
    l1_state = {}
    for key, value in vars(l1).items():
        if key in l1_exclude:
            continue
        l1_state[key] = l1.rng.getstate() if key == "rng" else value
    l2_state = {}
    for key, value in vars(l2).items():
        if key in l2_exclude:
            continue
        l2_state[key] = l2.rng.getstate() if key == "rng" else value
    history_state = {
        key: value for key, value in vars(scorer.history).items()
        if key not in history_exclude
    }
    payload = (
        _normal(l1_state), _normal(l2_state), _normal(history_state),
        _normal(counters)
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def selectors(
    scores, last_groups, frequencies, deltas, counts
) -> dict[str, int]:
    """One predeclared victim per selector, with draw-order final ties."""
    width = len(scores)
    if not all(len(values) == width for values in (
        last_groups, frequencies, deltas, counts
    )):
        raise ValueError("candidate vector lengths differ")
    next_values = target_column(
        np.asarray(deltas), np.asarray(counts), "next_use", HORIZON_MS / 1000.0
    )
    count_values = target_column(
        np.asarray(deltas), np.asarray(counts), "count", HORIZON_MS / 1000.0
    )
    pick = lambda key: min(range(width), key=key)
    return {
        "learned": pick(lambda j: tuple(scores[j])),
        "next_use": pick(lambda j: (float(next_values[j]), last_groups[j])),
        "count": pick(lambda j: (float(count_values[j]), last_groups[j])),
        "lru": pick(lambda j: (last_groups[j],)),
        "lfu": pick(lambda j: (frequencies[j], last_groups[j])),
    }


def label_tie_sets(deltas, counts) -> dict[str, list[int]]:
    next_values = target_column(
        np.asarray(deltas), np.asarray(counts), "next_use", HORIZON_MS / 1000.0
    )
    count_values = target_column(
        np.asarray(deltas), np.asarray(counts), "count", HORIZON_MS / 1000.0
    )
    return {
        "next_use": np.flatnonzero(next_values == min(next_values)).astype(int).tolist(),
        "count": np.flatnonzero(count_values == min(count_values)).astype(int).tolist(),
    }


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(value) + b"\n"
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


class ForkController:
    """Fork every saved action at selected sampled-L2 decisions.

    The caller must invoke finish_child(result) and then os._exit(0) when
    is_child is true after run_two_tier returns. The parent invokes
    validate_parent(result), which compares each captured-stream actual-action
    child against its uninterrupted suffix and terminal state.
    """

    def __init__(
        self, trace, scorer, state_index: dict[str, int],
        selected: list[dict], branch_dir: str | Path, run_fingerprint: str
    ) -> None:
        self.trace = trace
        self.scorer = scorer
        self.state_index = state_index
        self.selected = {
            (row["group_index"], row["ordinal"]): row for row in selected
        }
        if len(self.selected) != len(selected):
            raise ValueError("selected decision keys are not unique")
        self.branch_dir = Path(branch_dir)
        self.run_fingerprint = run_fingerprint
        self.labeller = _Labeller(trace, HORIZON_MS / 1000.0)
        self.ordinals: dict[int, int] = defaultdict(int)
        self.active: list[WindowReward] = []
        self.snapshots: list[dict] = []
        self.is_child = False
        self.branch: dict | None = None
        self.l1 = None
        self.l2 = None
        self.counters = None

    def attach(self, l1, l2, counters) -> None:
        self.l1, self.l2, self.counters = l1, l2, counters

    def on_request(
        self, ids, prefix, l2_hits, present, timestamp_ms, group_index, measured
    ) -> None:
        for collector in self.active:
            collector.on_request(
                ids, prefix, l2_hits, present, timestamp_ms, group_index, measured
            )

    def _verify_snapshot(
        self, saved: dict, candidates, scores, victim_index,
        timestamp_ms, group_index, arriving_index
    ) -> dict:
        if float(saved["timestamp_ms"]) != float(timestamp_ms):
            raise AssertionError("selected timestamp mismatch")
        if int(saved["group_index"]) != int(group_index):
            raise AssertionError("selected group mismatch")
        indices = [self.state_index[state_id] for state_id in candidates]
        if indices != saved["candidate_indices"]:
            raise AssertionError("selected candidate identities/order mismatch")
        tuples = [[float(value) for value in score] for score in scores]
        if tuples != saved["scores"]:
            raise AssertionError("selected online score tuples mismatch")
        if (int(victim_index) != saved["victim_index"]
                or int(arriving_index) != saved["arriving_index"]):
            raise AssertionError("selected actual victim/arrival mismatch")
        features = np.asarray([
            self.scorer.history.feature_vector(state_id, timestamp_ms)
            for state_id in candidates
        ], dtype=np.float64)
        if not np.array_equal(features, np.asarray(saved["features"], dtype=np.float64)):
            raise AssertionError("selected causal feature mismatch")
        labels = [self.labeller(state_id, timestamp_ms) for state_id in candidates]
        deltas = [float(value[0]) for value in labels]
        counts = [int(value[1]) for value in labels]
        encoded_deltas = [
            value if math.isfinite(value) else None for value in deltas
        ]
        if (encoded_deltas != saved["next_use_delta_ms"]
                or counts != saved["count_within_h"]):
            raise AssertionError("selected future label primitives mismatch")
        last_groups = [int(self.l1.last_group[state_id]) for state_id in candidates]
        frequencies = [int(self.l1.frequency[state_id]) for state_id in candidates]
        choices = selectors(tuples, last_groups, frequencies, deltas, counts)
        if choices["learned"] != victim_index:
            raise AssertionError("learned selector differs from online victim")
        return {
            "candidate_indices": indices,
            "scores": tuples,
            "last_group": last_groups,
            "frequency": frequencies,
            "next_use_delta_ms": encoded_deltas,
            "count_within_h": counts,
            "selectors": choices,
            "label_ties": label_tie_sets(deltas, counts),
            "candidate_count": len(candidates),
            "arriving_index": int(arriving_index),
            "actual_index": int(victim_index),
        }

    def _branch_path(self, saved: dict, replicate: int, action: int) -> Path:
        return self.branch_dir / (
            f"{saved['selection_hash']}__r{replicate}__a{action}.json"
        )

    def _load_branch(self, path: Path, saved: dict, replicate: int, action: int) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed_hash = payload.get("content_sha256")
        content = {key: value for key, value in payload.items() if key != "content_sha256"}
        if claimed_hash != hashlib.sha256(_canonical_bytes(content)).hexdigest():
            raise AssertionError(f"branch output content hash mismatch: {path}")
        if (
            payload.get("run_fingerprint") != self.run_fingerprint
            or payload.get("selection_hash") != saved["selection_hash"]
            or payload.get("replicate") != replicate
            or payload.get("action_index") != action
        ):
            raise AssertionError(f"branch output identity mismatch: {path}")
        if payload.get("error"):
            raise RuntimeError(f"branch failure {path}: {payload['error']}")
        reward = payload.get("reward")
        if not isinstance(reward, dict) or any(
            type(reward.get(name)) is not int or reward[name] < 0
            for name in (
                "q600", "qend", "l2_q600", "l2_qend",
                "requested600", "requested_end", "requests600", "requests_end"
            )
        ):
            raise AssertionError(f"malformed branch reward: {path}")
        if not isinstance(reward.get("reward_digest"), str) or len(reward["reward_digest"]) != 64:
            raise AssertionError(f"malformed branch reward digest: {path}")
        if any(
            not isinstance(payload.get(name), (int, float))
            or not math.isfinite(payload[name]) or payload[name] < 0
            for name in ("elapsed_seconds", "peak_rss_mib")
        ):
            raise AssertionError(f"malformed branch resource field: {path}")
        return payload

    def __call__(
        self, candidates, scores, victim_index,
        timestamp_ms, group_index, arriving_index
    ) -> int | None:
        if self.is_child:
            return None
        ordinal = self.ordinals[group_index]
        self.ordinals[group_index] = ordinal + 1
        saved = self.selected.get((int(group_index), int(ordinal)))
        if saved is None:
            return None
        if self.l1 is None or self.l2 is None:
            raise AssertionError("fork controller was not attached")
        metadata = self._verify_snapshot(
            saved, candidates, scores, victim_index,
            timestamp_ms, group_index, arriving_index
        )
        parent_reward = WindowReward(self.trace, timestamp_ms)
        self.active.append(parent_reward)
        snapshot = {
            "selected": saved, "metadata": metadata,
            "parent_reward": parent_reward, "branches": [],
        }
        self.snapshots.append(snapshot)
        replicates = (0, 1, 2) if saved["hash_rank"] < 2 else (0,)
        if os.path.isdir("/proc/self/task") and len(os.listdir("/proc/self/task")) != 1:
            raise RuntimeError("fork worker has multiple native threads")
        for replicate in replicates:
            for action in range(len(candidates)):
                path = self._branch_path(saved, replicate, action)
                if path.exists():
                    branch_row = self._load_branch(path, saved, replicate, action)
                else:
                    pid = os.fork()
                    if pid == 0:
                        self.is_child = True
                        self.selected = {}
                        self.active = [WindowReward(self.trace, timestamp_ms)]
                        self.branch = {
                            "path": path,
                            "selection_hash": saved["selection_hash"],
                            "replicate": replicate,
                            "action_index": action,
                            "actual_index": int(victim_index),
                            "started_monotonic": time.monotonic(),
                        }
                        if replicate:
                            self.l2.rng.seed(continuation_seed(
                                saved["lineage"], group_index, ordinal, replicate
                            ))
                        return int(action)
                    waited, status = os.waitpid(pid, 0)
                    if waited != pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                        detail = path.with_suffix(".error.json")
                        error = detail.read_text(encoding="utf-8") if detail.exists() else ""
                        raise RuntimeError(f"branch child failed: {path}: status={status} {error}")
                    branch_row = self._load_branch(path, saved, replicate, action)
                snapshot["branches"].append(branch_row)
        return None

    def finish_child(self, result) -> None:
        if not self.is_child or self.branch is None or len(self.active) != 1:
            raise AssertionError("finish_child called outside one branch")
        payload = {
            "run_fingerprint": self.run_fingerprint,
            "selection_hash": self.branch["selection_hash"],
            "replicate": self.branch["replicate"],
            "action_index": self.branch["action_index"],
            "reward": self.active[0].row(),
            "terminal_state_digest": (
                terminal_state_digest(self.l1, self.l2, self.scorer, self.counters)
                if self.branch["replicate"] == 0
                and self.branch["action_index"] == self.branch["actual_index"]
                else None
            ),
            "end_result": result.as_row() if (
                self.branch["replicate"] == 0
                and self.branch["action_index"] == self.branch["actual_index"]
            ) else None,
            "elapsed_seconds": time.monotonic() - self.branch["started_monotonic"],
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        payload["content_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        atomic_json(self.branch["path"], payload)

    def fail_child(self, error: BaseException) -> None:
        if self.is_child and self.branch is not None:
            path = self.branch["path"].with_suffix(".error.json")
            atomic_json(path, {
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })

    def validate_parent(self, result) -> list[dict]:
        if self.is_child:
            raise AssertionError("child cannot validate parent replay")
        if len(self.snapshots) != len(self.selected):
            raise AssertionError("not all selected decisions were replayed")
        terminal = terminal_state_digest(self.l1, self.l2, self.scorer, self.counters)
        rows = []
        for snapshot in self.snapshots:
            saved = snapshot["selected"]
            actual = saved["victim_index"]
            baseline = next(
                row for row in snapshot["branches"]
                if row["replicate"] == 0 and row["action_index"] == actual
            )
            if baseline["reward"] != snapshot["parent_reward"].row():
                raise AssertionError("actual-action child differs from parent reward suffix")
            if baseline["terminal_state_digest"] != terminal:
                raise AssertionError("actual-action child differs from parent terminal state")
            if baseline["end_result"] != result.as_row():
                raise AssertionError("actual-action child differs from parent replay counters")
            for branch in snapshot["branches"]:
                control = next(
                    row for row in snapshot["branches"]
                    if row["replicate"] == branch["replicate"]
                    and row["action_index"] == actual
                )
                for window in ("600", "end"):
                    total_delta = (
                        branch["reward"][f"q{window}"] - control["reward"][f"q{window}"]
                    )
                    l2_delta = (
                        branch["reward"][f"l2_q{window}"]
                        - control["reward"][f"l2_q{window}"]
                    )
                    if total_delta != l2_delta:
                        raise AssertionError("total and L2 action differences disagree")
            rows.append(snapshot)
        return rows
