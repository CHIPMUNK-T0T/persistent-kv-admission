"""Fresh-stream cross-fit diagnostics for frozen counterfactual snapshots."""

from __future__ import annotations

import hashlib
import math
import os
import time
from statistics import mean, stdev

from .counterfactual import ForkController, WindowReward, _canonical_bytes

NAMESPACE = "persistent-kv-counterfactual-rng-crossfit-v1"
FOLD_A = tuple(range(1, 9))
FOLD_B = tuple(range(9, 17))
STREAM_IDS = FOLD_A + FOLD_B
T7_975 = 2.3646242510102993


def derive_stream_seed(lineage: str, group_index: int, ordinal: int, stream_id: int) -> int:
    if not isinstance(lineage, str) or not lineage or type(stream_id) is not int or stream_id not in STREAM_IDS:
        raise ValueError("invalid fresh stream identity")
    if type(group_index) is not int or type(ordinal) is not int or group_index < 0 or ordinal < 0:
        raise ValueError("invalid decision identity")
    payload = _canonical_bytes([NAMESPACE, lineage, group_index, ordinal, stream_id])
    return int.from_bytes(hashlib.sha256(payload).digest(), "big", signed=False)


def _validated_values(q600, qend, actual_index, last_group, label_ties, fixed_selectors):
    if (set(q600) != set(STREAM_IDS) or set(qend) != set(STREAM_IDS)
            or any(type(key) is not int for source in (q600, qend) for key in source)):
        raise ValueError("all sixteen fresh streams are required")
    count = len(last_group)
    if count < 2 or type(actual_index) is not int or not 0 <= actual_index < count:
        raise ValueError("invalid candidate count or actual index")
    if any(type(value) is not int for value in last_group):
        raise ValueError("last_group values must be integers")
    for source in (q600, qend):
        for stream_id in STREAM_IDS:
            values = source[stream_id]
            if len(values) != count or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Q vector must have one nonnegative integer per action")
    if set(label_ties) != {"next_use", "count"}:
        raise ValueError("both exact label tie sets are required")
    for values in label_ties.values():
        if not values or len(set(values)) != len(values) or any(type(index) is not int or not 0 <= index < count for index in values):
            raise ValueError("invalid exact label tie set")
    if set(fixed_selectors) != {"next_use", "count", "lru", "lfu"}:
        raise ValueError("fixed exact/count/LRU/LFU selectors are required")
    if any(type(index) is not int or not 0 <= index < count for index in fixed_selectors.values()):
        raise ValueError("invalid fixed selector")
    for name in ("next_use", "count"):
        if fixed_selectors[name] not in label_ties[name]:
            raise ValueError("fixed exact selector is outside its minimum-label tie")
    return count


def _choose(q600, train_ids, allowed, last_group):
    totals = {action: sum(q600[stream_id][action] for stream_id in train_ids) for action in allowed}
    best = max(totals.values())
    tied = [action for action, total in totals.items() if total == best]
    return min(tied, key=lambda action: (last_group[action], action)), len(tied)


def _sample_stats(values):
    center = mean(values)
    sd = stdev(values)
    se = sd / math.sqrt(len(values))
    return center, sd, se, center - T7_975 * se, center + T7_975 * se


def crossfit_analysis(
    q600_by_stream: dict[int, list[int]], qend_by_stream: dict[int, list[int]],
    actual_index: int, last_group: list[int], label_ties: dict[str, list[int]],
    fixed_selectors: dict[str, int],
) -> dict:
    """Select on one fold and evaluate paired gains on the other fold only."""
    count = _validated_values(
        q600_by_stream, qend_by_stream, actual_index, last_group,
        label_ties, fixed_selectors,
    )
    choices = {"unrestricted": tuple(range(count)),
               "next_use_tie": tuple(label_ties["next_use"]),
               "count_tie": tuple(label_ties["count"])}
    directions, per_stream, state = [], [], []
    for name, allowed in choices.items():
        subset = []
        for direction, train_ids, evaluation_ids in (
            ("A_to_B", FOLD_A, FOLD_B), ("B_to_A", FOLD_B, FOLD_A)
        ):
            selected, training_tie_count = _choose(q600_by_stream, train_ids, allowed, last_group)
            train_gain = mean(q600_by_stream[r][selected] - q600_by_stream[r][actual_index] for r in train_ids)
            gains = [q600_by_stream[r][selected] - q600_by_stream[r][actual_index] for r in evaluation_ids]
            end_gains = [qend_by_stream[r][selected] - qend_by_stream[r][actual_index] for r in evaluation_ids]
            center, sd, se, lower, upper = _sample_stats(gains)
            record = {
                "selector": name, "direction": direction, "selected_index": selected,
                "training_tie_count": training_tie_count, "label_tie_count": len(allowed),
                "train_mean_gain_tokens": train_gain,
                "heldout_mean_gain_tokens": center, "heldout_sd_tokens": sd,
                "heldout_se_tokens": se, "heldout_ci95_low_tokens": lower,
                "heldout_ci95_high_tokens": upper,
                "heldout_end_mean_gain_tokens": mean(end_gains),
            }
            directions.append(record)
            subset.append(record)
            for stream_id, gain, end_gain in zip(evaluation_ids, gains, end_gains):
                q = q600_by_stream[stream_id]
                per_stream.append({
                    "selector": name, "direction": direction,
                    "stream_id": stream_id, "selected_index": selected,
                    "actual_index": actual_index, "gain_tokens": gain,
                    "end_gain_tokens": end_gain,
                    "heldout_hindsight_regret_tokens": max(q) - q[selected],
                    "heldout_hindsight_max_gain_tokens": max(q) - q[actual_index],
                })
        state.append({
            "selector": name,
            "symmetric_heldout_mean_gain_tokens": mean(record["heldout_mean_gain_tokens"] for record in subset),
            "symmetric_heldout_end_mean_gain_tokens": mean(record["heldout_end_mean_gain_tokens"] for record in subset),
            "selected_A_index": subset[0]["selected_index"],
            "selected_B_index": subset[1]["selected_index"],
            "fold_choices_agree": subset[0]["selected_index"] == subset[1]["selected_index"],
            "training_ties_A": subset[0]["training_tie_count"],
            "training_ties_B": subset[1]["training_tie_count"],
            "label_tie_count": len(allowed),
        })
    actionwise = []
    for action in range(count):
        values = [q600_by_stream[r][action] for r in STREAM_IDS]
        end_values = [qend_by_stream[r][action] for r in STREAM_IDS]
        paired = [q600_by_stream[r][action] - q600_by_stream[r][actual_index] for r in STREAM_IDS]
        end_paired = [qend_by_stream[r][action] - qend_by_stream[r][actual_index] for r in STREAM_IDS]
        actionwise.append({"action_index": action, "mean_q600_tokens": mean(values),
                           "sd_q600_tokens": stdev(values),
                           "mean_paired_gain_vs_actual_tokens": mean(paired),
                           "sd_paired_gain_vs_actual_tokens": stdev(paired),
                           "mean_qend_tokens": mean(end_values),
                           "sd_qend_tokens": stdev(end_values),
                           "mean_paired_end_gain_vs_actual_tokens": mean(end_paired),
                           "sd_paired_end_gain_vs_actual_tokens": stdev(end_paired)})
    hindsight, fixed = [], []
    for stream_id in STREAM_IDS:
        q, qe = q600_by_stream[stream_id], qend_by_stream[stream_id]
        hindsight.append({"stream_id": stream_id, "max_q600_tokens": max(q),
                          "actual_q600_tokens": q[actual_index],
                          "actual_hindsight_regret_tokens": max(q) - q[actual_index],
                          "constant_q": min(q) == max(q)})
        for name, action in fixed_selectors.items():
            fixed.append({"stream_id": stream_id, "selector": name,
                          "selected_index": action,
                          "paired_gain_vs_actual_tokens": q[action] - q[actual_index],
                          "paired_end_gain_vs_actual_tokens": qe[action] - qe[actual_index],
                          "hindsight_regret_tokens": max(q) - q[action]})
    mean_stream_max = mean(row["max_q600_tokens"] for row in hindsight)
    max_action_mean = max(row["mean_q600_tokens"] for row in actionwise)
    actual_mean = actionwise[actual_index]["mean_q600_tokens"]
    diagnostics = {
        "mean_stream_hindsight_max_q600_tokens": mean_stream_max,
        "max_actionwise_mean_q600_tokens": max_action_mean,
        "actual_mean_q600_tokens": actual_mean,
        "hindsight_jensen_gap_tokens": mean_stream_max - max_action_mean,
        "mean_stream_hindsight_regret_vs_actual_tokens": mean_stream_max - actual_mean,
        "fitted_all_stream_regret_vs_actual_tokens": max_action_mean - actual_mean,
    }
    if diagnostics["hindsight_jensen_gap_tokens"] < -1e-8:
        raise AssertionError("mean of stream maxima is smaller than maximum actionwise mean")
    return {"directions": directions, "per_stream": per_stream, "state": state,
            "actionwise": actionwise, "hindsight": hindsight, "fixed": fixed,
            "diagnostics": diagnostics}


class RandomnessForkController(ForkController):
    """Reuse the validated fork boundary; r0 is an actual-action check only."""

    def __init__(self, trace, scorer, state_index, selected, branch_dir, run_fingerprint,
                 *, smoke: bool = False, stream_seeds: dict[int, int] | None = None):
        super().__init__(trace, scorer, state_index, selected, branch_dir, run_fingerprint)
        self.smoke = smoke
        self.stream_seeds = stream_seeds

    def __call__(self, candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
        if self.is_child:
            return None
        ordinal = self.ordinals[group_index]
        self.ordinals[group_index] = ordinal + 1
        saved = self.selected.get((int(group_index), int(ordinal)))
        if saved is None:
            return None
        if self.l1 is None or self.l2 is None:
            raise AssertionError("fork controller was not attached")
        metadata = self._verify_snapshot(saved, candidates, scores, victim_index,
                                         timestamp_ms, group_index, arriving_index)
        parent_reward = WindowReward(self.trace, timestamp_ms)
        self.active.append(parent_reward)
        snapshot = {"selected": saved, "metadata": metadata,
                    "parent_reward": parent_reward, "branches": []}
        self.snapshots.append(snapshot)
        if os.path.isdir("/proc/self/task") and len(os.listdir("/proc/self/task")) != 1:
            raise RuntimeError("fork worker has multiple native threads")
        seed_map = self.stream_seeds if self.stream_seeds is not None else {
            r: derive_stream_seed(saved["lineage"], int(group_index), int(ordinal), r)
            for r in STREAM_IDS
        }
        expected = {r: derive_stream_seed(saved["lineage"], int(group_index), int(ordinal), r)
                    for r in STREAM_IDS}
        if seed_map != expected:
            raise AssertionError("frozen stream seeds do not match derivation")
        replicate_actions = [(0, (int(victim_index),))]
        replicate_actions += [(r, range(len(candidates))) for r in (1,) if self.smoke]
        if not self.smoke:
            replicate_actions += [(r, range(len(candidates))) for r in STREAM_IDS]
        for replicate, actions in replicate_actions:
            for action in actions:
                path = self._branch_path(saved, replicate, action)
                if path.exists():
                    branch_row = self._load_branch(path, saved, replicate, action)
                else:
                    pid = os.fork()
                    if pid == 0:
                        self.is_child = True
                        self.selected = {}
                        self.active = [WindowReward(self.trace, timestamp_ms)]
                        self.branch = {"path": path, "selection_hash": saved["selection_hash"],
                                       "replicate": replicate, "action_index": action,
                                       "actual_index": int(victim_index),
                                       "started_monotonic": time.monotonic()}
                        if replicate:
                            self.l2.rng.seed(seed_map[replicate])
                        return int(action)
                    waited, status = os.waitpid(pid, 0)
                    if waited != pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                        detail = path.with_suffix(".error.json")
                        error = detail.read_text(encoding="utf-8") if detail.exists() else ""
                        raise RuntimeError(f"branch child failed: {path}: status={status} {error}")
                    branch_row = self._load_branch(path, saved, replicate, action)
                snapshot["branches"].append(branch_row)
        expected_pairs = {(r, a) for r, actions in replicate_actions for a in actions}
        actual_pairs = {(b["replicate"], b["action_index"]) for b in snapshot["branches"]}
        if expected_pairs != actual_pairs or len(actual_pairs) != len(snapshot["branches"]):
            raise AssertionError("branch action/stream set mismatch")
        return None
