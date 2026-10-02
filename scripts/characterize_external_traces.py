#!/usr/bin/env python3
"""Trace-only characterization of Mooncake-schema traces (no cache policy is run).

For each trace the unmodified `load_mooncake_trace(path, 512)` is called in a
fresh subprocess, so its wall time and peak RSS are measured per file. The
child then times the policy-free preprocessing the two-tier / decision-
population runners do once per trace (`replay._occurrence_groups`,
`gap.working_set_bytes`, `decisionpop.horizon_for`, `decisionpop.state_indices`)
and computes:

* requests, time span (minutes), total input tokens, mean / median / p95
  input_length (numpy linear-interpolation percentile), mean blocks per
  request, unique prefix states;
* working_set_bytes: `gap.working_set_bytes(trace)` itself, the base the
  runners multiply by a capacity fraction (`run_decision_population._capacity`,
  packed size model, 2048 bytes per token);
* infinite-capacity repeat share: input tokens in blocks whose prefix state
  occurred in an earlier request, over all input tokens. "Earlier" is a
  strictly earlier timestamp, because the replay lets every request of one
  timestamp see the pre-batch cache; the variant that also counts earlier
  requests of the same timestamp (file order) is reported next to it. It is
  given for the whole trace and for the requests at or after
  start + 0.6 * span (the last 40 %, the runners' split), with the whole
  trace before them as history.

For traces carrying the Bailian metadata kept by
`scripts/convert_bailian_trace.py` it also reports the `type` and `turn`
distributions and the share of requests with parent_chat_id != -1.

  .venv/bin/python scripts/characterize_external_traces.py \
      data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl \
      data/raw/qwen_bailian_512/*_trace.jsonl

Writes characterization.json and characterization.csv to --output-dir
(default results/external_trace_characterization, untracked) and prints a
Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

BLOCK_TOKENS = 512
TEST_FRACTION_START = 0.6


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def repeat_shares(trace, window_start_ms: float) -> dict[str, float]:
    """Infinite-capacity repeat share, strict-timestamp and file-order variants."""
    states = trace.states
    seen_strict: set[str] = set()
    seen_order: set[str] = set()
    totals = Counter()
    for timestamp_ms, requests in trace.timestamp_groups():
        in_window = timestamp_ms >= window_start_ms
        for request in requests:
            tokens = [states[state_id].block_tokens for state_id in request.hash_ids]
            strict = [state_id in seen_strict for state_id in request.hash_ids]
            order = [state_id in seen_order for state_id in request.hash_ids]
            # A prefix-chained state is seen only with all its ancestors, so the
            # repeated blocks are a leading run; the replay counts that run.
            for flags in (strict, order):
                run = flags.index(False) if False in flags else len(flags)
                if any(flags[run:]):
                    raise AssertionError(f"{trace.name}: repeated blocks are not a prefix")
            hit_strict = sum(t for t, flag in zip(tokens, strict) if flag)
            hit_order = sum(t for t, flag in zip(tokens, order) if flag)
            assert sum(tokens) == request.input_length
            for scope, use in (("all", True), ("last40", in_window)):
                if use:
                    totals[f"{scope}_tokens"] += request.input_length
                    totals[f"{scope}_strict"] += hit_strict
                    totals[f"{scope}_order"] += hit_order
                    totals[f"{scope}_requests"] += 1
            seen_order.update(request.hash_ids)
        for request in requests:
            seen_strict.update(request.hash_ids)
    return {
        "repeat_share": totals["all_strict"] / totals["all_tokens"],
        "repeat_share_file_order": totals["all_order"] / totals["all_tokens"],
        "repeat_share_last40": totals["last40_strict"] / totals["last40_tokens"],
        "repeat_share_last40_file_order": totals["last40_order"] / totals["last40_tokens"],
        "repeated_tokens": totals["all_strict"],
        "repeated_tokens_file_order": totals["all_order"],
        "last40_requests": totals["last40_requests"],
        "last40_input_tokens": totals["last40_tokens"],
    }


def metadata(path: Path) -> dict | None:
    types, turns, with_parent, records = Counter(), Counter(), 0, 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "type" not in record:
                return None
            records += 1
            types[record["type"]] += 1
            turns[record["turn"]] += 1
            with_parent += record["parent_chat_id"] != -1
    buckets = Counter()
    for turn, count in turns.items():
        buckets[str(turn) if turn <= 5 else ("6-10" if turn <= 10 else ">10")] += count
    return {
        "type_counts": dict(types.most_common()),
        "turn_buckets": {key: buckets[key] for key in ("1", "2", "3", "4", "5", "6-10", ">10")
                         if buckets[key]},
        "turn_max": max(turns),
        "parent_share": with_parent / records,
    }


def characterize_one(path: Path) -> dict:
    import numpy as np

    from persistent_kv_admission.decisionpop import horizon_for, state_indices
    from persistent_kv_admission.gap import working_set_bytes
    from persistent_kv_admission.replay import _occurrence_groups
    from persistent_kv_admission.trace import load_mooncake_trace

    row: dict = {"path": str(path), "rss_before_load_mib": _rss_mib()}
    clock = time.perf_counter()
    trace = load_mooncake_trace(path, BLOCK_TOKENS)
    row["load_seconds"] = time.perf_counter() - clock
    row["peak_rss_after_load_mib"] = _rss_mib()
    clock = time.perf_counter()
    groups = _occurrence_groups(trace)
    row["occurrence_groups_seconds"] = time.perf_counter() - clock
    clock = time.perf_counter()
    working_set = working_set_bytes(trace)
    row["working_set_seconds"] = time.perf_counter() - clock
    clock = time.perf_counter()
    horizon, split_ms, snapshots = horizon_for(trace)
    row["horizon_for_seconds"] = time.perf_counter() - clock
    clock = time.perf_counter()
    state_indices(trace)
    row["state_indices_seconds"] = time.perf_counter() - clock
    row["peak_rss_after_preprocessing_mib"] = _rss_mib()

    lengths = np.asarray([request.input_length for request in trace.requests], dtype=np.int64)
    blocks = np.asarray([len(request.hash_ids) for request in trace.requests], dtype=np.int64)
    row.update(
        trace=trace.name,
        requests=len(trace.requests),
        timestamp_groups=sum(1 for _ in trace.timestamp_groups()),
        span_minutes=trace.duration_ms / 60000.0,
        total_input_tokens=int(lengths.sum()),
        input_mean=float(lengths.mean()),
        input_median=float(np.median(lengths)),
        input_p95=float(np.percentile(lengths, 95)),
        blocks_per_request=float(blocks.mean()),
        total_blocks=int(blocks.sum()),
        unique_states=len(trace.states),
        working_set_bytes=int(working_set),
        state_occurrences=sum(len(value) for value in groups.values()),
        horizon_seconds=horizon,
        split_ms=split_ms,
        test_snapshots=len(snapshots),
    )
    window_start = trace.start_ms + TEST_FRACTION_START * trace.duration_ms
    row.update(repeat_shares(trace, window_start))
    row["window_start_ms"] = window_start
    row["metadata"] = metadata(path)
    return row


def _run_child(path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, __file__, "--one", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


COLUMNS = (
    ("requests", "requests", "{:,}"),
    ("span (min)", "span_minutes", "{:.1f}"),
    ("input tokens", "total_input_tokens", "{:,}"),
    ("input mean", "input_mean", "{:,.0f}"),
    ("input median", "input_median", "{:,.0f}"),
    ("input p95", "input_p95", "{:,.0f}"),
    ("blocks/request", "blocks_per_request", "{:.2f}"),
    ("unique states", "unique_states", "{:,}"),
    ("working set (GiB)", "working_set_gib", "{:.1f}"),
    ("repeat share", "repeat_share", "{:.3f}"),
    ("repeat share, last 40%", "repeat_share_last40", "{:.3f}"),
    ("repeat share (file order)", "repeat_share_file_order", "{:.3f}"),
    ("last 40%, file order", "repeat_share_last40_file_order", "{:.3f}"),
    ("load (s)", "load_seconds", "{:.2f}"),
    ("peak RSS after load (MiB)", "peak_rss_after_load_mib", "{:,.0f}"),
    ("occurrence groups (s)", "occurrence_groups_seconds", "{:.2f}"),
    ("horizon_for (s)", "horizon_for_seconds", "{:.2f}"),
    ("peak RSS after preprocessing (MiB)", "peak_rss_after_preprocessing_mib", "{:,.0f}"),
    ("H (s) / test snapshots", None, None),
)


def _table(rows: list[dict]) -> str:
    lines = ["| metric | " + " | ".join(row["trace"] for row in rows) + " |",
             "|---|" + "---:|" * len(rows)]
    for label, key, form in COLUMNS:
        if key is None:
            cells = [f"{row['horizon_seconds']:g} / {row['test_snapshots']}" for row in rows]
        else:
            cells = [form.format(row[key]) for row in rows]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", nargs="*", type=Path)
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY / "results/external_trace_characterization")
    args = parser.parse_args(argv)
    if args.one is not None:
        print(json.dumps(characterize_one(args.one)))
        return
    if not args.traces:
        parser.error("give trace files")
    rows = []
    for path in args.traces:
        row = _run_child(path)
        row["working_set_gib"] = row["working_set_bytes"] / 2**30
        rows.append(row)
        print(f"done {row['trace']}", file=sys.stderr, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "characterization.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat = [{key: value for key, value in row.items() if key != "metadata"} for row in rows]
    with (args.output_dir / "characterization.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(_table(rows))
    for row in rows:
        if row["metadata"]:
            meta = row["metadata"]
            print(f"\n{row['trace']}: type {meta['type_counts']}; turn {meta['turn_buckets']} "
                  f"(max {meta['turn_max']}); parent_chat_id != -1: {meta['parent_share']:.2%}")


if __name__ == "__main__":
    main()
