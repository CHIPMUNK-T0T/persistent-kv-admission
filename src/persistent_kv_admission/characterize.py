"""Trace characterization, causal horizon evaluation, and result writers."""

from __future__ import annotations

import bisect
import csv
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ranking import ranking_metrics
from .replay import POLICIES, replay
from .trace import Trace, unique_timestamps


HORIZONS_SECONDS = (1, 10, 60, 600, 3600, 21600, 86400)
SIGNALS = ("recency", "frequency", "prefix_length", "fan_out", "branch_diversity")
INTER_ARRIVAL_BINS_SECONDS = (
    0.0,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    60.0,
    600.0,
    3600.0,
    21600.0,
    86400.0,
    math.inf,
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def state_feature_rows(trace: Trace, bytes_per_token: int) -> list[dict[str, object]]:
    rows = []
    for state_id, meta in trace.states.items():
        occurrence_count = len(trace.occurrences_ms[state_id])
        potential_reuses = occurrence_count - 1
        rows.append(
            {
                "state_id": state_id,
                "parent_id": meta.parent_id or "",
                "depth_blocks": meta.depth,
                "prefix_tokens": meta.prefix_tokens,
                "block_tokens": meta.block_tokens,
                "state_size_bytes": meta.block_tokens * bytes_per_token,
                "occurrence_count": occurrence_count,
                "potential_reuses": potential_reuses,
                "fan_out": len(trace.children.get(state_id, ())),
                "branch_diversity": len(trace.terminal_branches.get(state_id, ())),
                "reuse_count_x_prefix_length": potential_reuses * meta.prefix_tokens,
                "estimated_avoided_prefill_tokens": potential_reuses * meta.block_tokens,
            }
        )
    return rows


def reuse_mass_rows(trace: Trace) -> list[dict[str, object]]:
    histogram: dict[int, int] = defaultdict(int)
    for occurrences in trace.occurrences_ms.values():
        histogram[len(occurrences)] += 1
    total_states = len(trace.states)
    total_occurrences = sum(count * states for count, states in histogram.items())
    cumulative = 0
    rows = []
    for occurrence_count in sorted(histogram):
        state_count = histogram[occurrence_count]
        cumulative += state_count
        states_ge_n = sum(value for n, value in histogram.items() if n >= occurrence_count)
        rows.append(
            {
                "occurrence_count_n": occurrence_count,
                "state_count": state_count,
                "state_mass": state_count / total_states,
                "occurrence_mass": occurrence_count * state_count / total_occurrences,
                "cdf_p_N_le_n": cumulative / total_states,
                "ccdf_p_N_ge_n": states_ge_n / total_states,
            }
        )
    return rows


def inter_arrival_rows(trace: Trace) -> tuple[list[dict[str, object]], dict[str, float]]:
    intervals = np.asarray(
        [
            (right - left) / 1000.0
            for times in trace.occurrences_ms.values()
            for left, right in zip(times, times[1:])
        ],
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    if len(intervals):
        counts, _ = np.histogram(intervals, bins=np.asarray(INTER_ARRIVAL_BINS_SECONDS))
        cumulative = 0
        total = len(intervals)
        for index, count in enumerate(counts):
            cumulative += int(count)
            rows.append(
                {
                    "lower_seconds_inclusive": INTER_ARRIVAL_BINS_SECONDS[index],
                    "upper_seconds_exclusive": INTER_ARRIVAL_BINS_SECONDS[index + 1],
                    "interval_count": int(count),
                    "mass": int(count) / total,
                    "cdf": cumulative / total,
                    "ccdf_at_lower": int(counts[index:].sum()) / total,
                }
            )
        quantiles = {
            "inter_arrival_count": float(total),
            "inter_arrival_p50_seconds": float(np.quantile(intervals, 0.5)),
            "inter_arrival_p90_seconds": float(np.quantile(intervals, 0.9)),
            "inter_arrival_p99_seconds": float(np.quantile(intervals, 0.99)),
        }
    else:
        quantiles = {
            "inter_arrival_count": 0.0,
            "inter_arrival_p50_seconds": math.nan,
            "inter_arrival_p90_seconds": math.nan,
            "inter_arrival_p99_seconds": math.nan,
        }
    return rows, quantiles


def trace_summary(trace: Trace, bytes_per_token: int) -> dict[str, object]:
    occurrence_counts = np.asarray(
        [len(times) for times in trace.occurrences_ms.values()], dtype=int
    )
    potential_reuses = occurrence_counts - 1
    prefix_lengths = np.asarray(
        [meta.prefix_tokens for meta in trace.states.values()], dtype=np.int64
    )
    block_tokens = np.asarray(
        [meta.block_tokens for meta in trace.states.values()], dtype=np.int64
    )
    reusable = occurrence_counts >= 2
    potential_reuse_events = int(potential_reuses.sum())
    two_hit_lost_events = int(reusable.sum())
    potential_token_value = int(np.sum(potential_reuses * prefix_lengths))
    two_hit_lost_prefix_value = int(np.sum(prefix_lengths[reusable]))
    incremental_avoided = int(np.sum(potential_reuses * block_tokens))
    two_hit_lost_incremental = int(np.sum(block_tokens[reusable]))
    total_unique_bytes = int(block_tokens.sum() * bytes_per_token)
    return {
        "trace": trace.name,
        "request_count": len(trace.requests),
        "unique_timestamp_count": len(unique_timestamps(trace)),
        "duration_seconds": trace.duration_ms / 1000.0,
        "state_count": len(trace.states),
        "block_size_tokens": trace.block_size,
        "bytes_per_token": bytes_per_token,
        "total_unique_state_bytes": total_unique_bytes,
        "potential_reuse_events": potential_reuse_events,
        "two_hit_lost_reuse_events": two_hit_lost_events,
        "two_hit_lost_reuse_fraction": two_hit_lost_events / max(potential_reuse_events, 1),
        "potential_reuse_x_prefix_tokens": potential_token_value,
        "two_hit_lost_prefix_value": two_hit_lost_prefix_value,
        "two_hit_lost_prefix_value_fraction": two_hit_lost_prefix_value
        / max(potential_token_value, 1),
        "estimated_avoided_prefill_tokens_unbounded": incremental_avoided,
        "two_hit_lost_avoided_prefill_tokens": two_hit_lost_incremental,
        "two_hit_lost_avoided_prefill_fraction": two_hit_lost_incremental
        / max(incremental_avoided, 1),
    }


def _sample_indices(values: list[float], latest: float, count: int) -> set[int]:
    eligible = [index for index, value in enumerate(values) if value <= latest]
    if not eligible:
        return set()
    positions = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    return {eligible[int(position)] for position in positions}


def predictive_power_rows(
    trace: Trace, snapshot_count: int = 24, precision_k: int = 100
) -> list[dict[str, object]]:
    timestamps = unique_timestamps(trace)
    selected_by_horizon: dict[int, set[int]] = {}
    for horizon in HORIZONS_SECONDS:
        selected_by_horizon[horizon] = _sample_indices(
            timestamps, trace.end_ms - horizon * 1000.0, snapshot_count
        )

    frequency: dict[str, int] = defaultdict(int)
    last_seen: dict[str, float] = {}
    observed_children: dict[str, set[str]] = defaultdict(set)
    observed_terminals: dict[str, set[str]] = defaultdict(set)
    metric_values: dict[tuple[int, str], list[tuple[float, float, float, int, int]]] = defaultdict(list)

    for timestamp_index, (timestamp, requests) in enumerate(trace.timestamp_groups()):
        for request in requests:
            terminal = request.hash_ids[-1]
            for index, state_id in enumerate(request.hash_ids):
                frequency[state_id] += 1
                last_seen[state_id] = timestamp
                observed_terminals[state_id].add(terminal)
                if index + 1 < len(request.hash_ids):
                    observed_children[state_id].add(request.hash_ids[index + 1])

        horizons_here = [
            horizon
            for horizon, indices in selected_by_horizon.items()
            if timestamp_index in indices
        ]
        if not horizons_here:
            continue
        state_ids = list(last_seen)
        scores = {
            "recency": -np.asarray(
                [(timestamp - last_seen[state_id]) / 1000.0 for state_id in state_ids]
            ),
            "frequency": np.asarray([frequency[state_id] for state_id in state_ids], dtype=float),
            "prefix_length": np.asarray(
                [trace.states[state_id].prefix_tokens for state_id in state_ids], dtype=float
            ),
            "fan_out": np.asarray(
                [len(observed_children[state_id]) for state_id in state_ids], dtype=float
            ),
            "branch_diversity": np.asarray(
                [len(observed_terminals[state_id]) for state_id in state_ids], dtype=float
            ),
        }
        for horizon in horizons_here:
            cutoff = timestamp + horizon * 1000.0
            labels = np.asarray(
                [
                    int(
                        (position := bisect.bisect_right(trace.occurrences_ms[state_id], timestamp))
                        < len(trace.occurrences_ms[state_id])
                        and trace.occurrences_ms[state_id][position] <= cutoff
                    )
                    for state_id in state_ids
                ],
                dtype=np.int8,
            )
            positives = int(labels.sum())
            for signal, signal_scores in scores.items():
                auc, average_precision, precision_at_k = ranking_metrics(
                    labels, signal_scores, precision_k
                )
                metric_values[(horizon, signal)].append(
                    (auc, average_precision, precision_at_k, positives, len(labels))
                )

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS_SECONDS:
        for signal in SIGNALS:
            values = metric_values[(horizon, signal)]
            valid_auc = [value[0] for value in values if not math.isnan(value[0])]
            valid_ap = [value[1] for value in values if not math.isnan(value[1])]
            valid_pk = [value[2] for value in values if not math.isnan(value[2])]
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "signal": signal,
                    "auc": float(np.mean(valid_auc)) if valid_auc else math.nan,
                    "average_precision": float(np.mean(valid_ap)) if valid_ap else math.nan,
                    f"precision_at_{precision_k}": float(np.mean(valid_pk)) if valid_pk else math.nan,
                    "snapshots": len(values),
                    "valid_auc_snapshots": len(valid_auc),
                    "mean_positive_states": float(np.mean([value[3] for value in values]))
                    if values
                    else math.nan,
                    "mean_candidate_states": float(np.mean([value[4] for value in values]))
                    if values
                    else math.nan,
                    "status": (
                        "ok"
                        if valid_ap
                        else "no_positive_windows"
                        if values
                        else "insufficient_uncensored_trace_duration"
                    ),
                }
            )
    return rows


def replay_rows(
    trace: Trace, budget_fractions: tuple[float, ...], bytes_per_token: int
) -> list[dict[str, object]]:
    working_set_bytes = sum(
        meta.block_tokens * bytes_per_token for meta in trace.states.values()
    )
    rows = []
    for fraction in budget_fractions:
        capacity = max(1, round(working_set_bytes * fraction))
        for policy in POLICIES:
            result = replay(trace, policy, capacity, fraction, bytes_per_token)
            row = asdict(result)
            row["trace"] = trace.name
            row["working_set_bytes"] = working_set_bytes
            rows.append(row)
    return rows
