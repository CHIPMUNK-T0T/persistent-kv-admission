"""Read-only residence telemetry for two frozen one-action counterfactual branches."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

from .counterfactual import ForkController, WindowReward, _canonical_bytes, atomic_json

HORIZON_MS = 600_000.0
STREAMS = tuple(range(1, 17))


def fixed_pair(metadata: dict) -> tuple[int, int]:
    """Choose the two pre-registered zero-own-reuse actions using labels only."""
    ties = metadata["label_ties"]
    if ties["count"] != ties["next_use"] or len(ties["count"]) < 2:
        raise AssertionError("the exact zero-count tie sets must coincide and contain two actions")
    allowed = ties["count"]
    if len(set(allowed)) != len(allowed) or any(
        type(a) is not int or a < 0 or a >= metadata["candidate_count"] for a in allowed
    ):
        raise AssertionError("invalid exact tie set")
    if any(metadata["count_within_h"][a] != 0 for a in allowed):
        raise AssertionError("focal candidates must have zero own reuse within H")
    last = metadata["last_group"]
    e = min(allowed, key=lambda a: (last[a], a))
    z = max(allowed, key=lambda a: (last[a], a))
    if e == z or metadata["selectors"]["next_use"] != e:
        raise AssertionError("fixed E/Z action contract failed")
    return e, z


def verified_content(payload: dict) -> bool:
    claimed = payload.get("content_sha256")
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == hashlib.sha256(_canonical_bytes(content)).hexdigest()


class ResidenceTelemetry:
    """One fork child's observations. All callbacks leave replay state untouched."""

    def __init__(self, trace, start_ms: float, start_group: int, focus: tuple[str, str], survivor: str,
                 state_index: dict[str, int], bytes_per_token: int = 2048) -> None:
        self.trace = trace
        self.start_ms = float(start_ms)
        self.start_group = int(start_group)
        self.focus = focus
        self.survivor = survivor
        self.state_index = state_index
        self.survivor_bytes = trace.states[survivor].block_tokens * bytes_per_token
        self.requests: list[list] = []
        self.decisions: list[list] = []
        self.removals: list[list] = []
        self.first_survivor_removal: list | None = None
        self.first_survivor_removal_initial = False
        self.in_initial_admission = True
        self.immediate_overflow_rounds = 1
        self.q600 = self.qend = self.l2_q600 = self.l2_qend = 0
        self.requests600 = self.requests_end = 0

    def on_request(self, cache, order: int, ids, prefix, l2_hits, present,
                   timestamp_ms, group_index, measured) -> None:
        del measured
        if timestamp_ms <= self.start_ms:
            return
        self.in_initial_admission = False
        if self.trace.requests[order].hash_ids != ids or self.trace.requests[order].timestamp_ms != timestamp_ms:
            raise AssertionError("request order no longer matches frozen trace")
        states = self.trace.states
        l1_tokens = states[ids[prefix - 1]].prefix_tokens if prefix else 0
        l2_tokens = sum(states[ids[index]].block_tokens for index in l2_hits)
        hit_ids = [self.state_index[ids[index]] for index in l2_hits]
        present_ids = [self.state_index[ids[index]] for index in present]
        requested_mask = sum(1 << i for i, state_id in enumerate(self.focus) if state_id in ids)
        resident_mask = sum(1 << i for i, state_id in enumerate(self.focus) if state_id in cache.cached)
        # The two-bit masks are keyed by the frozen E,Z candidate order.
        self.requests.append([order, group_index, timestamp_ms, prefix, hit_ids,
                              present_ids, l1_tokens, l2_tokens, requested_mask, resident_mask])
        self.qend += l1_tokens + l2_tokens
        self.l2_qend += l2_tokens
        self.requests_end += 1
        if timestamp_ms <= self.start_ms + HORIZON_MS:
            self.q600 += l1_tokens + l2_tokens
            self.l2_q600 += l2_tokens
            self.requests600 += 1
            if requested_mask:
                raise AssertionError("zero-own-reuse focal candidate requested within H")

    def on_decision(self, candidates, scores, victim_index, timestamp_ms,
                    group_index, arriving_index) -> None:
        del scores
        if timestamp_ms > self.start_ms:
            self.in_initial_admission = False
        if self.in_initial_admission and timestamp_ms == self.start_ms:
            self.immediate_overflow_rounds += 1
        self.decisions.append([group_index, timestamp_ms,
                               [self.state_index[state_id] for state_id in candidates],
                               self.state_index[candidates[victim_index]], arriving_index])

    def on_victim(self, state_id, timestamp_ms, group_index) -> None:
        del state_id, timestamp_ms, group_index
        self.in_initial_admission = False

    def on_removal(self, state_id, kind, timestamp_ms, group_index, decision_index) -> None:
        if timestamp_ms < self.start_ms:
            return
        if timestamp_ms > self.start_ms:
            self.in_initial_admission = False
        row = [group_index, timestamp_ms, self.state_index[state_id], kind, decision_index]
        self.removals.append(row)
        if state_id == self.survivor and self.first_survivor_removal is None:
            self.first_survivor_removal = row
            self.first_survivor_removal_initial = self.in_initial_admission

    def summary(self, reward: dict) -> dict:
        if (self.q600 != reward["q600"] or self.qend != reward["qend"]
                or self.l2_q600 != reward["l2_q600"] or self.l2_qend != reward["l2_qend"]
                or self.requests600 != reward["requests600"]
                or self.requests_end != reward["requests_end"]):
            raise AssertionError("per-request telemetry does not sum to branch reward")
        end_ms = min(self.trace.end_ms, self.start_ms + HORIZON_MS)
        first = self.first_survivor_removal
        leave_ms = float(first[1]) if first is not None else None
        exposure_ms = max(0.0, min(end_ms, leave_ms if leave_ms is not None else end_ms) - self.start_ms)
        return {
            "focal_survivor_index": self.state_index[self.survivor],
            "focal_survivor_bytes": self.survivor_bytes,
            "first_survivor_removal": first,
            "survived_immediate_overflow": not self.first_survivor_removal_initial,
            "first_residence_ms": None if first is None else max(0.0, leave_ms - self.start_ms),
            "first_residence_groups": None if first is None else int(first[0] - self.start_group),
            "censored_600": first is None or leave_ms > end_ms,
            "censored_end": first is None,
            "resident_byte_seconds_600": self.survivor_bytes * exposure_ms / 1000.0,
            "immediate_overflow_rounds": self.immediate_overflow_rounds,
        }


class ResidenceForkController(ForkController):
    """Fork only E/Z under each frozen fresh stream, with read-only telemetry."""

    def __init__(self, trace, scorer, state_index, selected, branch_dir, run_fingerprint,
                 pair: tuple[int, int], stream_seeds: dict[int, int], *, smoke=False,
                 bytes_per_token: int = 2048):
        super().__init__(trace, scorer, state_index, [selected], branch_dir, run_fingerprint)
        self.pair = pair
        self.stream_seeds = stream_seeds
        self.smoke = smoke
        self.bytes_per_token = bytes_per_token
        self.request_cursor = 0
        self.telemetry: ResidenceTelemetry | None = None

    def on_request(self, ids, prefix, l2_hits, present, timestamp_ms, group_index, measured):
        order = self.request_cursor
        self.request_cursor += 1
        super().on_request(ids, prefix, l2_hits, present, timestamp_ms, group_index, measured)
        if self.is_child and self.telemetry is not None:
            self.telemetry.on_request(self.l2, order, ids, prefix, l2_hits, present,
                                      timestamp_ms, group_index, measured)

    def on_decision(self, *args):
        if self.is_child and self.telemetry is not None:
            self.telemetry.on_decision(*args)

    def on_victim(self, *args):
        if self.is_child and self.telemetry is not None:
            self.telemetry.on_victim(*args)

    def on_removal(self, *args):
        if self.is_child and self.telemetry is not None:
            self.telemetry.on_removal(*args)

    def __call__(self, candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
        if self.is_child:
            return None
        ordinal = self.ordinals[group_index]
        self.ordinals[group_index] = ordinal + 1
        saved = self.selected.get((int(group_index), int(ordinal)))
        if saved is None:
            return None
        metadata = self._verify_snapshot(saved, candidates, scores, victim_index,
                                         timestamp_ms, group_index, arriving_index)
        if fixed_pair(metadata) != self.pair:
            raise AssertionError("frozen fixed pair changed at live snapshot")
        if os.path.isdir("/proc/self/task") and len(os.listdir("/proc/self/task")) != 1:
            raise RuntimeError("fork worker has multiple native threads")
        snapshot = {"selected": saved, "metadata": metadata,
                    "parent_reward": WindowReward(self.trace, timestamp_ms), "branches": []}
        self.active.append(snapshot["parent_reward"])
        self.snapshots.append(snapshot)
        for replicate in ((1,) if self.smoke else STREAMS):
            for action in self.pair:
                path = self._branch_path(saved, replicate, action)
                if path.exists():
                    branch = self._load_branch(path, saved, replicate, action)
                else:
                    pid = os.fork()
                    if pid == 0:
                        self.is_child = True
                        self.selected = {}
                        self.active = [WindowReward(self.trace, timestamp_ms)]
                        survivor = candidates[self.pair[1] if action == self.pair[0] else self.pair[0]]
                        self.telemetry = ResidenceTelemetry(
                            self.trace, timestamp_ms, group_index,
                            (candidates[self.pair[0]], candidates[self.pair[1]]),
                            survivor, self.state_index, self.bytes_per_token)
                        self.branch = {"path": path, "selection_hash": saved["selection_hash"],
                                       "replicate": replicate, "action_index": action,
                                       "started_monotonic": time.monotonic()}
                        self.l2.rng.seed(self.stream_seeds[replicate])
                        return int(action)
                    waited, status = os.waitpid(pid, 0)
                    if waited != pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                        error_path = path.with_suffix(".error.json")
                        detail = error_path.read_text() if error_path.exists() else ""
                        raise RuntimeError(f"residence branch failed: {path}: status={status} {detail}")
                    branch = self._load_branch(path, saved, replicate, action)
                snapshot["branches"].append(branch)
        return None

    def finish_child(self, result) -> None:
        if not self.is_child or self.branch is None or self.telemetry is None or len(self.active) != 1:
            raise AssertionError("finish_child outside residence branch")
        reward = self.active[0].row()
        summary = self.telemetry.summary(reward)
        payload = {"run_fingerprint": self.run_fingerprint,
                   "selection_hash": self.branch["selection_hash"],
                   "replicate": self.branch["replicate"],
                   "action_index": self.branch["action_index"],
                   "reward": reward,
                   "telemetry": {"requests": self.telemetry.requests,
                                 "decisions": self.telemetry.decisions,
                                 "removals": self.telemetry.removals,
                                 "summary": summary},
                   "terminal_state_digest": None, "end_result": None,
                   "elapsed_seconds": time.monotonic() - self.branch["started_monotonic"],
                   "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
        payload["content_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        atomic_json(self.branch["path"], payload)


def pair_analysis(e_branch: dict, z_branch: dict, *, start_ms: float,
                  start_group: int) -> tuple[dict, list[dict]]:
    """Align identical future requests and retain signed cumulative Z minus E."""
    e_rows = e_branch["telemetry"]["requests"]
    z_rows = z_branch["telemetry"]["requests"]
    if len(e_rows) != len(z_rows):
        raise AssertionError("paired request counts differ")
    cumulative600 = cumulative_end = 0
    differing_hit600 = differing_reward600 = 0
    first_hit = first_reward = first_hit600 = first_reward600 = None
    events = []
    request_number = 0
    before = after = abs_before = abs_after = 0
    e_summary = e_branch["telemetry"]["summary"]
    z_summary = z_branch["telemetry"]["summary"]
    split_ms = None
    if not e_summary["censored_600"] and not z_summary["censored_600"]:
        split_ms = max(e_summary["first_survivor_removal"][1],
                       z_summary["first_survivor_removal"][1])
    for e, z in zip(e_rows, z_rows):
        request_number += 1
        if e[:3] != z[:3]:
            raise AssertionError("paired request identity/order changed")
        if e[3] != z[3] or e[6] != z[6]:
            raise AssertionError("L1 prefix/reward differs between fixed branches")
        order, group, timestamp_ms = e[:3]
        if timestamp_ms <= start_ms:
            raise AssertionError("same-timestamp request entered future telemetry")
        hit_diff = e[4] != z[4]
        delta = int(z[6] + z[7] - e[6] - e[7])
        cumulative_end += delta
        in600 = timestamp_ms <= start_ms + HORIZON_MS
        if in600:
            if e[8] or z[8]:
                raise AssertionError("focal own request inside zero-reuse horizon")
            cumulative600 += delta
            differing_hit600 += hit_diff
            differing_reward600 += delta != 0
            if split_ms is not None:
                if timestamp_ms <= split_ms:
                    before += delta
                    abs_before += abs(delta)
                else:
                    after += delta
                    abs_after += abs(delta)
        identity = {"order": order, "group_index": group, "timestamp_ms": timestamp_ms,
                    "lag_requests": request_number, "lag_groups": group - start_group,
                    "lag_ms": timestamp_ms - start_ms}
        if hit_diff:
            if first_hit is None:
                first_hit = identity
            if in600 and first_hit600 is None:
                first_hit600 = identity
        if delta != 0:
            if first_reward is None:
                first_reward = identity
            if in600 and first_reward600 is None:
                first_reward600 = identity
        if hit_diff or delta:
            events.append({**identity, "within_600": in600,
                           "e_l2_hit_ids": e[4], "z_l2_hit_ids": z[4],
                           "e_l2_present_ids": e[5], "z_l2_present_ids": z[5],
                           "e_focal_resident_mask": e[9], "z_focal_resident_mask": z[9],
                           "delta_q_request": delta,
                           "cumulative_delta_q600": cumulative600,
                           "cumulative_delta_qend": cumulative_end})
    if (cumulative600 != z_branch["reward"]["q600"] - e_branch["reward"]["q600"]
            or cumulative_end != z_branch["reward"]["qend"] - e_branch["reward"]["qend"]):
        raise AssertionError("paired request increments do not sum to archived Q difference")

    def victim_groups(branch):
        groups = {}
        for group, timestamp_ms, state_index, kind, _ in branch["telemetry"]["removals"]:
            if timestamp_ms > start_ms and kind in {"evicted", "rejected"}:
                groups.setdefault(group, []).append([state_index, kind])
        return groups

    e_events, z_events = victim_groups(e_branch), victim_groups(z_branch)
    first_different_victim_group = next((group for group in sorted(set(e_events) | set(z_events))
                                         if e_events.get(group, []) != z_events.get(group, [])), None)
    return {
        "delta_q600": cumulative600, "delta_qend": cumulative_end,
        "first_hit": first_hit, "first_reward": first_reward,
        "first_hit600": first_hit600, "first_reward600": first_reward600,
        "first_different_victim_group": first_different_victim_group,
        "different_hit_requests600": differing_hit600,
        "different_reward_requests600": differing_reward600,
        "residence_split_ms": split_ms,
        "delta_q_before_focal_end": before if split_ms is not None else None,
        "delta_q_after_focal_end": after if split_ms is not None else None,
        "absolute_delta_q_before_focal_end": abs_before if split_ms is not None else None,
        "absolute_delta_q_after_focal_end": abs_after if split_ms is not None else None,
        "rejoining": "unmeasured",
    }, events
