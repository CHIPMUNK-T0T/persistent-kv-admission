#!/usr/bin/env python3
"""Phase 0.95: L1 victim stream and tree-allocation characterization.

The persistent tier of a two-tier KV cache decides about the states the
upper tier evicts, not about every state ever seen. This script fixes the
upper tier (L1: prefix-closed, heap LRU or LFU, one of several capacities),
logs every L1 eviction as one event with causal and evaluation-only fields,
and replays generic lower-tier (L2) policies and an offline comparator on the
same victim stream under three fixed models: union closure with the tree hit
rule (primary), the independent-block hit rule (control that removes prefix
dependency), and a standalone prefix-closed L2 (sensitivity). Single-tier
references at L1 + L2 capacity are added for context.

Outputs, written to --output-dir and copied to --paper-dir:
victim_events_summary.csv, victim_events_by_depth.csv, two_tier_replay.csv,
two_tier_gate.csv, two_tier_single_tier_reference.csv,
two_tier_offline_tiebreak.csv (sensitivity of the offline comparator to its
tie-break on a few cells) and the run configuration JSON.

No policy is introduced. Everything is deterministic (one seed).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.characterize import write_csv
from persistent_kv_admission.crossworkload import HYPERPARAMETERS
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import (
    DEPTH_LABELS,
    annotate_events,
    run_two_tier,
    single_tier_reference,
    summarize_events,
)

DEFAULT_L1_BUDGETS = (0.0025, 0.01, 0.02)
DEFAULT_MULTIPLIERS = (1, 2, 4, 8, 16)
DEFAULT_MAX_TOTAL = 0.10
L1_POLICIES = ("lru", "lfu")
# (closure, hit_model, l2_policy)
ARMS = (
    [("union", "tree", p) for p in ("lru", "lfu", "lru_2hit", "offline_next_use")]
    + [("union", "independent", p) for p in ("lru", "lfu", "lru_2hit", "offline_next_use")]
    + [("standalone", "tree", p) for p in ("lru", "lfu", "offline_next_use")]
)
GENERIC = ("lru", "lfu", "lru_2hit")
# Go / stop thresholds, fixed before the run (see docs/two-tier-victim-findings.md).
GATE_ABSOLUTE = 0.05      # offline L2 extra avoided tokens / evaluation-window input tokens
GATE_DEPENDENCY = 0.10    # (independent − tree) / independent, for at least one L2 policy
GATE_ROOM = 0.20          # (offline − best generic) / offline, tree model
# Cells (L1 fraction, L2 multiplier) replayed a second time with the offline
# comparator's other tie-break, to show the comparator's sensitivity to it.
TIEBREAK_DIAGNOSTIC_CELLS = ((0.0025, 1), (0.0025, 2), (0.01, 1))
TIEBREAKS = ("prefix_first", "deeper_first")
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", type=Path, help="trace files (required unless --figure-only)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/two_tier"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--l1-budgets", default=",".join(str(v) for v in DEFAULT_L1_BUDGETS))
    parser.add_argument("--l2-multipliers", default=",".join(str(v) for v in DEFAULT_MULTIPLIERS))
    parser.add_argument("--max-total", type=float, default=DEFAULT_MAX_TOTAL,
                        help="skip cells whose L1 + L2 capacity exceeds this fraction of the working set")
    parser.add_argument("--l1-policies", default=",".join(L1_POLICIES))
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figures from the CSVs in <output-dir> without replaying")
    return parser.parse_args()


def _write(path: Path, rows) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fields} for row in rows], fields)


def _capacity(name: str, fraction: float) -> int:
    return max(1, round(_SHARED["working_set"][name] * fraction))


def _event_worker(task):
    name, l1_policy, fraction = task
    trace = _SHARED["traces"][name]
    started = time.time()
    events = []
    base = run_two_tier(
        trace, l1_policy, _capacity(name, fraction), 0, occurrence_groups=_SHARED["groups"][name],
        measure_from_ms=_SHARED["splits"][name], event_sink=events,
    )
    annotate_events(trace, events)
    summary, by_depth = summarize_events(events, base.l1_capacity_bytes, base.requested_tokens,
                                         tuple(_SHARED["multipliers"]))
    key = dict(trace=name, l1_policy=l1_policy, l1_fraction=fraction, l1_capacity_bytes=base.l1_capacity_bytes,
               requested_tokens=base.requested_tokens, l1_avoided_tokens=base.l1_avoided_tokens)
    path = _SHARED["events_dir"] / f"{name}_{l1_policy}_{fraction}.jsonl"
    with path.open("w", encoding="utf-8") as sink:
        for event in events:
            sink.write(json.dumps(asdict(event)) + "\n")
    return {
        "summary": dict(key, **summary),
        "by_depth": [dict(key, **row) for row in by_depth],
        "baseline": dict(key, avoided_prefill_tokens=base.avoided_prefill_tokens, l1_evictions=base.l1_evictions),
        "seconds": time.time() - started,
        "task": task,
    }


def _replay_worker(task):
    name, l1_policy, fraction, multiplier, closure, hit_model, policy = task
    trace = _SHARED["traces"][name]
    started = time.time()
    result = run_two_tier(
        trace, l1_policy, _capacity(name, fraction), _capacity(name, fraction * multiplier), policy,
        hit_model=hit_model, closure=closure, occurrence_groups=_SHARED["groups"][name],
        measure_from_ms=_SHARED["splits"][name],
    )
    row = result.as_row()
    row.update(l1_fraction=fraction, l2_multiplier=multiplier, l2_fraction=fraction * multiplier,
               total_fraction=fraction * (1 + multiplier), seconds=time.time() - started)
    return row


def _reference_worker(task):
    name, policy, fraction, multiplier = task
    trace = _SHARED["traces"][name]
    # Exactly the bytes the two-tier cell holds: the two tier capacities are
    # rounded separately there, so rounding the total fraction once instead
    # can differ by a byte.
    capacity = _capacity(name, fraction) + _capacity(name, fraction * multiplier)
    avoided = single_tier_reference(trace, policy, capacity, _SHARED["splits"][name],
                                    occurrence_groups=_SHARED["groups"][name])
    return {"trace": name, "policy": policy, "total_fraction": round(fraction * (1 + multiplier), 6),
            "l1_fraction": fraction, "l2_multiplier": multiplier, "avoided_prefill_tokens": avoided}


def _tiebreak_worker(task):
    """One offline-L2 replay under an explicit tie-break (diagnostic only)."""
    name, l1_policy, fraction, multiplier, hit_model, tiebreak = task
    trace = _SHARED["traces"][name]
    result = run_two_tier(
        trace, l1_policy, _capacity(name, fraction), _capacity(name, fraction * multiplier),
        "offline_next_use", hit_model=hit_model, closure="union",
        occurrence_groups=_SHARED["groups"][name], measure_from_ms=_SHARED["splits"][name],
        offline_tiebreak=tiebreak,
    )
    return {"trace": name, "l1_policy": l1_policy, "l1_fraction": fraction, "l2_multiplier": multiplier,
            "hit_model": hit_model, "tiebreak": tiebreak,
            "avoided_prefill_tokens": result.avoided_prefill_tokens,
            "requested_tokens": result.requested_tokens}


def tiebreak_rows(cells, outputs, replays, baselines) -> list[dict]:
    """One row per (cell, tie-break): tree, independent, and their difference.

    The `prefix_first` numbers are the grid's own offline rows (the grid runs
    the default tie-break), so they are not replayed a second time.
    """
    base = {(b["trace"], b["l1_policy"], float(b["l1_fraction"])): int(b["avoided_prefill_tokens"])
            for b in baselines}
    grid = {(r["trace"], r["l1_policy"], float(r["l1_fraction"]), float(r["l2_multiplier"]), r["hit_model"]): r
            for r in replays if r["closure"] == "union" and r["l2_policy"] == "offline_next_use"}
    replayed = {(o["trace"], o["l1_policy"], float(o["l1_fraction"]), float(o["l2_multiplier"]),
                 o["hit_model"], o["tiebreak"]): o for o in outputs}
    rows = []
    for name, l1_policy, fraction, multiplier in cells:
        l1_only = base[(name, l1_policy, fraction)]
        requested = int(grid[(name, l1_policy, fraction, multiplier, "tree")]["requested_tokens"])
        for tiebreak in TIEBREAKS:
            avoided = {}
            for hit_model in ("tree", "independent"):
                if tiebreak == "prefix_first":
                    avoided[hit_model] = int(grid[(name, l1_policy, fraction, multiplier, hit_model)]
                                             ["avoided_prefill_tokens"])
                else:
                    avoided[hit_model] = int(replayed[(name, l1_policy, fraction, multiplier, hit_model,
                                                       tiebreak)]["avoided_prefill_tokens"])
            tree, independent = avoided["tree"], avoided["independent"]
            rows.append({
                "trace": name,
                "l1_policy": l1_policy,
                "l1_fraction": fraction,
                "l2_multiplier": multiplier,
                "tiebreak": tiebreak,
                "tree_avoided_tokens": tree,
                "independent_avoided_tokens": independent,
                "l1_only_avoided_tokens": l1_only,
                "dependency_cost_tokens": independent - tree,
                "dependency_cost_fraction": ((independent - tree) / (independent - l1_only)
                                             if independent > l1_only else math.nan),
                "tree_extra_fraction_of_input": (tree - l1_only) / max(requested, 1),
            })
    return rows


def derive(replays: list[dict], baselines: list[dict], summaries: list[dict], references: list[dict]) -> list[dict]:
    """Add per-row gains, closures, dependency costs, and the per-cell gate table."""
    base = {(b["trace"], b["l1_policy"], float(b["l1_fraction"])): b for b in baselines}
    ceiling = {(s["trace"], s["l1_policy"], float(s["l1_fraction"])): s for s in summaries}
    reference = {(r["trace"], r["policy"], round(float(r["total_fraction"]), 6)): int(r["avoided_prefill_tokens"])
                 for r in references}
    for row in replays:
        key = (row["trace"], row["l1_policy"], float(row["l1_fraction"]))
        baseline = int(base[key]["avoided_prefill_tokens"])
        row["l1_only_avoided_tokens"] = baseline
        row["extra_avoided_tokens"] = int(row["avoided_prefill_tokens"]) - baseline
        row["extra_fraction_of_input"] = row["extra_avoided_tokens"] / max(int(row["requested_tokens"]), 1)
        row["rescue_ceiling_tokens"] = int(ceiling[key]["rescue_tokens_with_ancestors"])
        row["extra_over_rescue_ceiling"] = row["extra_avoided_tokens"] / max(row["rescue_ceiling_tokens"], 1)
        total = round(float(row["total_fraction"]), 6)
        for policy in ("lru", "lfu", "offline_next_use"):
            value = reference.get((row["trace"], policy, total))
            row[f"single_tier_{policy}_avoided_tokens"] = value if value is not None else ""
    cells: dict[tuple, dict[tuple[str, str, str], dict]] = defaultdict(dict)
    for row in replays:
        cells[(row["trace"], row["l1_policy"], float(row["l1_fraction"]), float(row["l2_multiplier"]))][
            (row["closure"], row["hit_model"], row["l2_policy"])] = row
    gate_rows = []
    for cell, arms in sorted(cells.items()):
        trace, l1_policy, fraction, multiplier = cell

        def extra(closure, hit_model, policy):
            row = arms.get((closure, hit_model, policy))
            return int(row["extra_avoided_tokens"]) if row is not None else math.nan

        for (closure, hit_model, policy), row in arms.items():
            lru = extra(closure, hit_model, "lru")
            offline = extra(closure, hit_model, "offline_next_use")
            gain = int(row["extra_avoided_tokens"])
            row["l2_closure_vs_offline"] = (gain - lru) / (offline - lru) if offline > lru else math.nan
            if closure == "union":
                independent = extra("union", "independent", policy)
                tree = extra("union", "tree", policy)
                row["dependency_cost_tokens"] = independent - tree
                row["dependency_cost_fraction"] = (independent - tree) / independent if independent > 0 else math.nan
            else:
                union = extra("union", "tree", policy)
                row["standalone_cost_tokens"] = union - gain
                row["standalone_cost_fraction"] = (union - gain) / union if union > 0 else math.nan
        requested = int(next(iter(arms.values()))["requested_tokens"])
        # The gate row is only defined when every union/tree arm ran.
        missing = [arm for arm in (("union", "tree", p) for p in GENERIC + ("offline_next_use",))
                   if arm not in arms]
        if missing:
            raise KeyError(f"cell {cell} is missing arm(s) {missing}; cannot build its gate row")
        offline_tree = extra("union", "tree", "offline_next_use")
        generic = {p: extra("union", "tree", p) for p in GENERIC}
        # A non-finite gain means the arm did not produce a number; it must not
        # win the comparison by accident.
        finite_generic = {p: v for p, v in generic.items() if math.isfinite(v)}
        best_generic = max(finite_generic, key=finite_generic.get) if finite_generic else None
        best_generic_extra = generic[best_generic] if best_generic is not None else math.nan
        dependency = {p: arms[("union", "tree", p)]["dependency_cost_fraction"] for p in GENERIC + ("offline_next_use",)}
        standalone = arms.get(("standalone", "tree", "offline_next_use"))
        summary = ceiling[(trace, l1_policy, fraction)]
        gate = {
            "trace": trace,
            "l1_policy": l1_policy,
            "l1_fraction": fraction,
            "l2_multiplier": multiplier,
            "total_fraction": fraction * (1 + multiplier),
            "requested_tokens": requested,
            "l1_only_avoided_tokens": base[(trace, l1_policy, fraction)]["avoided_prefill_tokens"],
            "rescue_ceiling_fraction_of_input": summary["rescue_fraction_with_ancestors"],
            "offline_tree_extra_tokens": offline_tree,
            "offline_tree_extra_fraction_of_input": offline_tree / requested,
            "offline_tree_over_rescue_ceiling": offline_tree / max(int(summary["rescue_tokens_with_ancestors"]), 1),
            "offline_independent_extra_tokens": extra("union", "independent", "offline_next_use"),
            "best_generic_policy": best_generic if best_generic is not None else "",
            "best_generic_extra_tokens": best_generic_extra,
            "best_generic_extra_fraction_of_input": best_generic_extra / requested,
            "room_over_generic_fraction": ((offline_tree - best_generic_extra) / offline_tree
                                           if best_generic is not None and offline_tree > 0 else math.nan),
            **{f"dependency_cost_fraction_{p}": dependency[p] for p in dependency},
            "dependency_cost_fraction_max": max(v for v in dependency.values() if not math.isnan(v)) if any(
                not math.isnan(v) for v in dependency.values()) else math.nan,
            "standalone_offline_cost_fraction": standalone["standalone_cost_fraction"] if standalone else math.nan,
            "single_tier_offline_avoided_tokens": arms[("union", "tree", "lru")]["single_tier_offline_next_use_avoided_tokens"],
            "single_tier_lru_avoided_tokens": arms[("union", "tree", "lru")]["single_tier_lru_avoided_tokens"],
        }
        gate["pass_absolute"] = gate["offline_tree_extra_fraction_of_input"] >= GATE_ABSOLUTE
        gate["pass_dependency"] = gate["dependency_cost_fraction_max"] >= GATE_DEPENDENCY
        gate["pass_room"] = gate["room_over_generic_fraction"] >= GATE_ROOM
        gate["pass_all"] = gate["pass_absolute"] and gate["pass_dependency"] and gate["pass_room"]
        gate_rows.append(gate)
    return gate_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        load = lambda name: list(csv.DictReader((args.output_dir / name).open(encoding="utf-8")))
        make_figures(args.paper_dir, load("victim_events_summary.csv"), load("victim_events_by_depth.csv"),
                     load("two_tier_replay.csv"))
        print("figures redrawn", flush=True)
        return
    if not args.traces:
        raise SystemExit("at least one trace file is required unless --figure-only is given")
    l1_budgets = tuple(float(v) for v in args.l1_budgets.split(","))
    multipliers = tuple(float(v) for v in args.l2_multipliers.split(","))
    l1_policies = tuple(args.l1_policies.split(","))
    context = mp.get_context("fork")
    events_dir = args.output_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    traces, groups, splits, working_set = {}, {}, {}, {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = trace.start_ms + HYPERPARAMETERS["train_fraction"] * (trace.end_ms - trace.start_ms)
        working_set[trace.name] = working_set_bytes(trace)
        print(f"loaded {trace.name}", flush=True)
    _SHARED.update(traces=traces, groups=groups, splits=splits, working_set=working_set,
                   multipliers=multipliers, events_dir=events_dir)

    # 1. Victim streams (L1 only), one per trace × L1 policy × L1 budget.
    event_tasks = [(name, policy, fraction) for name in traces for policy in l1_policies for fraction in l1_budgets]
    started = time.time()
    with context.Pool(min(args.workers, len(event_tasks))) as pool:
        event_outputs = pool.map(_event_worker, event_tasks)
    summaries = [o["summary"] for o in event_outputs]
    by_depth = [row for o in event_outputs for row in o["by_depth"]]
    baselines = [o["baseline"] for o in event_outputs]
    print(f"  {len(event_tasks)} victim streams in {time.time()-started:.0f}s", flush=True)

    # 2. Two-tier replays on the same streams.
    cells = [(name, policy, fraction, multiplier)
             for name in traces for policy in l1_policies for fraction in l1_budgets for multiplier in multipliers
             if fraction * (1 + multiplier) <= args.max_total + 1e-12]
    tasks = [(name, policy, fraction, multiplier, closure, hit_model, l2_policy)
             for (name, policy, fraction, multiplier) in cells for (closure, hit_model, l2_policy) in ARMS]
    print(f"  {len(tasks)} two-tier replays on {args.workers} workers", flush=True)
    replays = []
    started = time.time()
    raw_path = args.output_dir / "raw_results.jsonl"
    with context.Pool(args.workers) as pool, raw_path.open("w", encoding="utf-8") as raw:
        for done, row in enumerate(pool.imap_unordered(_replay_worker, tasks, chunksize=2), 1):
            replays.append(row)
            raw.write(json.dumps(row, default=str) + "\n")
            if done % 100 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {time.time()-started:.0f}s", flush=True)

    # 3. Single-tier references at L1 + L2 capacity. The capacity is the exact
    # sum of the cell's two tier capacities, so the pair identifies the task;
    # the total fraction is kept as the join key and is unique per pair here.
    totals = sorted({(name, round(fraction * (1 + multiplier), 6), fraction, multiplier)
                     for (name, _, fraction, multiplier) in cells})
    reference_tasks = [(name, policy, fraction, multiplier) for name, _total, fraction, multiplier in totals
                       for policy in ("lru", "lfu", "offline_next_use")]
    with context.Pool(min(args.workers, len(reference_tasks))) as pool:
        references = pool.map(_reference_worker, reference_tasks)

    # 4. Offline tie-break diagnostic on a few cells of the grid.
    grid_cells = set(cells)
    diagnostic_cells = [(name, policy, float(fraction), float(multiplier))
                        for name in traces for policy in l1_policies
                        for (fraction, multiplier) in TIEBREAK_DIAGNOSTIC_CELLS
                        if (name, policy, float(fraction), float(multiplier)) in grid_cells]
    diagnostic_tasks = [(name, policy, fraction, multiplier, hit_model, "deeper_first")
                        for (name, policy, fraction, multiplier) in diagnostic_cells
                        for hit_model in ("tree", "independent")]
    print(f"  {len(diagnostic_tasks)} offline tie-break replays", flush=True)
    diagnostic_outputs = []
    if diagnostic_tasks:
        with context.Pool(min(args.workers, len(diagnostic_tasks))) as pool:
            diagnostic_outputs = pool.map(_tiebreak_worker, diagnostic_tasks)
    diagnostics = tiebreak_rows(diagnostic_cells, diagnostic_outputs, replays, baselines)

    gate_rows = derive(replays, baselines, summaries, references)
    replays.sort(key=lambda r: (r["trace"], r["l1_policy"], float(r["l1_fraction"]), float(r["l2_multiplier"]),
                                r["closure"], r["hit_model"], r["l2_policy"]))
    summaries.sort(key=lambda r: (r["trace"], r["l1_policy"], float(r["l1_fraction"])))
    by_depth.sort(key=lambda r: (r["trace"], r["l1_policy"], float(r["l1_fraction"]), DEPTH_LABELS.index(r["depth_bin"])))
    for directory in (args.output_dir, args.paper_dir):
        _write(directory / "victim_events_summary.csv", summaries)
        _write(directory / "victim_events_by_depth.csv", by_depth)
        _write(directory / "two_tier_replay.csv", replays)
        _write(directory / "two_tier_gate.csv", gate_rows)
        _write(directory / "two_tier_single_tier_reference.csv", references)
        _write(directory / "two_tier_offline_tiebreak.csv", diagnostics)
    config = {
        "l1_policies": l1_policies, "l1_budgets": l1_budgets, "l2_multipliers": multipliers,
        "max_total_fraction": args.max_total, "arms": ARMS, "hyperparameters": HYPERPARAMETERS,
        "gates": {"absolute": GATE_ABSOLUTE, "dependency": GATE_DEPENDENCY, "room": GATE_ROOM},
        "measure_from_ms": splits, "working_set_bytes": working_set,
        "note": "L1 fixed (heap LRU/LFU); every arriving block enters L1, so the victim stream is "
                "identical for every L2 arm. Exclusive L2 under union closure; independent-block hit "
                "rule as the control; standalone prefix-closed L2 as the sensitivity. One seed: "
                "everything is deterministic.",
    }
    text = json.dumps(config, indent=2, default=str) + "\n"
    (args.output_dir / "run_config.json").write_text(text, encoding="utf-8")
    (args.paper_dir / "two_tier_config.json").write_text(text, encoding="utf-8")
    make_figures(args.paper_dir, summaries, by_depth, replays)
    print("done", flush=True)


def _f(row, key) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def make_figures(paper_dir: Path, summaries, by_depth, replays) -> None:
    traces = sorted({r["trace"] for r in summaries})
    # Figure 11: what the victim stream looks like.
    figure, axes = plt.subplots(2, len(traces), figsize=(5.2 * len(traces), 7.6), squeeze=False)
    for column, trace in enumerate(traces):
        top, bottom = axes[0][column], axes[1][column]
        for l1_policy, style in (("lru", "-"), ("lfu", "--")):
            rows = sorted((r for r in summaries if r["trace"] == trace and r["l1_policy"] == l1_policy),
                          key=lambda r: _f(r, "l1_fraction"))
            if not rows:
                continue
            x = [100 * _f(r, "l1_fraction") for r in rows]
            top.plot(x, [_f(r, "share_reused_before_end") for r in rows], style, marker="o", color="tab:blue",
                     label=f"L1 {l1_policy.upper()}: victims requested again before trace end")
            top.plot(x, [_f(r, "share_reused_with_l1_ancestors") for r in rows], style, marker="s", color="tab:red",
                     label=f"L1 {l1_policy.upper()}: of those, all ancestors still in L1 at reuse")
            bottom.plot(x, [100 * _f(r, "rescue_fraction_with_ancestors") for r in rows], style, marker="o",
                        color="tab:green", label=f"L1 {l1_policy.upper()}: ceiling, L2 holds block + missing ancestors")
            bottom.plot(x, [100 * _f(r, "rescue_fraction_self") for r in rows], style, marker="s", color="tab:orange",
                        label=f"L1 {l1_policy.upper()}: L2 holds the block only")
        ticks = sorted({100 * _f(r, "l1_fraction") for r in summaries if r["trace"] == trace})
        top.set(xscale="log", ylim=(0, 1.02), title=trace, ylabel="share of measured victim events")
        bottom.set(xscale="log", xlabel="L1 capacity (% of unique-state bytes)",
                   ylabel="avoided prefill by an infinite L2 (% of input tokens)")
        for axis in (top, bottom):
            axis.set_xticks(ticks)
            axis.set_xticklabels([f"{t:g}" for t in ticks])
            axis.minorticks_off()
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=6)
    figure.suptitle("L1 victim stream: reuse after eviction and what a lower tier could rescue")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig11_victim_stream.png", dpi=170)
    plt.close(figure)

    # Figure 12: two-tier gain by L2 size, L1 = LRU.
    l1_fractions = sorted({_f(r, "l1_fraction") for r in replays})
    figure, axes = plt.subplots(len(l1_fractions), len(traces), figsize=(5.2 * len(traces), 3.9 * len(l1_fractions)),
                                squeeze=False)
    arms = [
        ("union", "tree", "lru", "L2 LRU (tree)", "tab:blue", "-"),
        ("union", "tree", "lfu", "L2 LFU (tree)", "tab:orange", "-"),
        ("union", "tree", "lru_2hit", "L2 2-hit LRU (tree)", "tab:green", "-"),
        ("union", "tree", "offline_next_use", "L2 offline next use (tree)", "black", "-"),
        ("union", "independent", "offline_next_use", "L2 offline, independent blocks (control)", "black", "--"),
        ("union", "independent", "lru", "L2 LRU, independent blocks (control)", "tab:blue", "--"),
        ("standalone", "tree", "offline_next_use", "L2 offline, standalone closure", "black", ":"),
    ]
    for row_index, fraction in enumerate(l1_fractions):
        for column, trace in enumerate(traces):
            axis = axes[row_index][column]
            sel = [r for r in replays if r["trace"] == trace and r["l1_policy"] == "lru" and _f(r, "l1_fraction") == fraction]
            if not sel:
                axis.set_visible(False)
                continue
            requested = _f(sel[0], "requested_tokens")
            for closure, hit_model, policy, label, color, style in arms:
                rows = sorted((r for r in sel if r["closure"] == closure and r["hit_model"] == hit_model
                               and r["l2_policy"] == policy), key=lambda r: _f(r, "l2_multiplier"))
                if rows:
                    axis.plot([_f(r, "l2_multiplier") for r in rows], [100 * _f(r, "extra_fraction_of_input") for r in rows],
                              style, marker="o", ms=3.5, color=color, label=label)
            rows = sorted((r for r in sel if r["closure"] == "union" and r["hit_model"] == "tree" and r["l2_policy"] == "lru"),
                          key=lambda r: _f(r, "l2_multiplier"))
            baseline = _f(rows[0], "l1_only_avoided_tokens")
            for key, label, marker in (("single_tier_offline_next_use_avoided_tokens", "one cache of L1+L2 bytes: offline", "x"),
                                       ("single_tier_lru_avoided_tokens", "one cache of L1+L2 bytes: LRU", "+")):
                axis.plot([_f(r, "l2_multiplier") for r in rows], [100 * (_f(r, key) - baseline) / requested for r in rows],
                          linestyle="none", marker=marker, color="0.5", label=label)
            axis.axhline(100 * _f(rows[0], "rescue_ceiling_tokens") / requested, color="tab:green", linewidth=0.8,
                         label="infinite L2 ceiling")
            axis.set(xscale="log", xticks=[1, 2, 4, 8, 16], xticklabels=["1", "2", "4", "8", "16"],
                     title=f"{trace}, L1 = LRU {100 * fraction:g}%", xlabel="L2 capacity / L1 capacity",
                     ylabel="extra avoided prefill (% of input tokens)")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=5.5)
    figure.suptitle("Lower-tier gain on the same L1 victim stream")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig12_two_tier_gain.png", dpi=170)
    plt.close(figure)

    # Figure 13: where L2 bytes sit vs where L2 benefit comes from, by depth.
    target_fraction = 0.01 if 0.01 in l1_fractions else l1_fractions[len(l1_fractions) // 2]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.4 * len(traces), 4.2), squeeze=False)
    width = 0.2
    positions = np.arange(len(DEPTH_LABELS))
    for column, trace in enumerate(traces):
        axis = axes[0][column]
        sel = [r for r in replays if r["trace"] == trace and r["l1_policy"] == "lru"
               and _f(r, "l1_fraction") == target_fraction and r["closure"] == "union" and r["hit_model"] == "tree"]
        multipliers = sorted({_f(r, "l2_multiplier") for r in sel})
        if not multipliers:
            axis.set_visible(False)
            continue
        multiplier = 4.0 if 4.0 in multipliers else multipliers[-1]
        events = [r for r in by_depth if r["trace"] == trace and r["l1_policy"] == "lru" and _f(r, "l1_fraction") == target_fraction]
        events.sort(key=lambda r: DEPTH_LABELS.index(r["depth_bin"]))
        series = [("victim events (share)", [_f(r, "share_of_events") for r in events], "0.6"),
                  ("rescue ceiling (share of tokens)", [_f(r, "rescue_share_with_ancestors") for r in events], "tab:green")]
        for policy, color in (("lru", "tab:blue"), ("offline_next_use", "black")):
            row = next((r for r in sel if r["l2_policy"] == policy and _f(r, "l2_multiplier") == multiplier), None)
            if row is None:
                continue
            held = np.array([_f(row, f"l2_byte_seconds_depth_{label}") for label in DEPTH_LABELS])
            gained = np.array([_f(row, f"l2_avoided_tokens_depth_{label}") for label in DEPTH_LABELS])
            series.append((f"L2 {policy}: bytes held (share)", list(held / max(held.sum(), 1e-9)), color))
            series.append((f"L2 {policy}: avoided tokens (share)", list(gained / max(gained.sum(), 1e-9)), color))
        for index, (label, values, color) in enumerate(series):
            hatch = "//" if "avoided" in label else None
            axis.bar(positions + (index - len(series) / 2 + 0.5) * width, values, width, label=label, color=color,
                     hatch=hatch, edgecolor="white" if hatch else color, alpha=0.9)
        axis.set(xticks=positions, xticklabels=DEPTH_LABELS, xlabel="block depth", ylabel="share",
                 title=f"{trace}, L1 = LRU {100 * target_fraction:g}%, L2 = {multiplier:g}× L1")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle("Depth allocation: where victims come from, where L2 bytes sit, where L2 gain comes from")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig13_depth_allocation.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
