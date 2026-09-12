#!/usr/bin/env python3
"""Research 1, target change: same ranker, same eviction, different label.

The gap decomposition showed that a perfect predictor of the 600 s binary
label recovers little headroom at small budgets, that the best label horizon
grows with the budget, and that count and next-use oracles do better. This
script asks whether the *causal* ranker inherits that: it fits the Phase 0.5
linear ranker to each alternative target (binary at 60 / 300 / fit horizon,
log reuse count, negative log next-use time) with the same features,
standardisation, and hyperparameters, and replays every fit through the
sampled-leaf eviction over several seeds and budgets. A horizon-matched arm
switches between the binary rankers by the cache's Little's-law residence
time, with no parameter.

No policy is introduced. Nothing is tuned per trace.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.characterize import write_csv
from persistent_kv_admission.crossworkload import HYPERPARAMETERS, fit_fixed_model
from persistent_kv_admission.gap import add_closure, aggregate_replay, run_arm, target_arm_specs
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.temporal import FEATURE_NAMES
from persistent_kv_admission.trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.001, 0.0025, 0.01, 0.02, 0.05)
BINARY_HORIZONS = (60, 300, 600)
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", type=Path, help="trace files (required unless --figure-only)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/target_change"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--budgets", default=",".join(str(v) for v in DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--oracle-csv", type=Path, default=Path("results/paper/oracle_replay.csv"),
                        help="Phase 0.75 summary, drawn as reference lines in the figure")
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figure from <output-dir>/target_change.csv without replaying")
    return parser.parse_args()


def _write(path: Path, rows) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fields} for row in rows], fields)


def _fit_worker(task):
    name, target, horizon = task
    trace = _SHARED["traces"][name]
    started = time.time()
    ranker, used, split_ms, positives, total = fit_fixed_model(
        trace, float(horizon), snapshot_count=_SHARED["snapshots"],
        train_fraction=HYPERPARAMETERS["train_fraction"], target=target,
    )
    return {"trace": name, "target": target, "requested_horizon": horizon, "horizon": used,
            "ranker": ranker, "split_ms": split_ms, "train_rows": total, "train_positives": positives,
            "seconds": time.time() - started}


def _replay_worker(task):
    name, spec, fraction, seed = task
    trace = _SHARED["traces"][name]
    started = time.time()
    out = run_arm(
        trace, spec, fraction, seed, None, _SHARED["splits"][name], _SHARED["groups"][name],
        ladder_snapshots=None, rankers=_SHARED["rankers"][name],
    )
    out["seconds"] = time.time() - started
    out["task"] = (name, spec.name, fraction, seed)
    return out


def main() -> None:
    args = parse_args()
    budgets = tuple(float(v) for v in args.budgets.split(","))
    seeds = tuple(range(args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        summary = list(csv.DictReader((args.output_dir / "target_change.csv").open(encoding="utf-8")))
        make_figure(args.paper_dir, summary, args.oracle_csv)
        print("figure redrawn", flush=True)
        return
    if not args.traces:
        raise SystemExit("at least one trace file is required unless --figure-only is given")
    context = mp.get_context("fork")

    traces = {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        traces[trace.name] = trace
        print(f"loaded {trace.name}", flush=True)
    _SHARED.update(traces=traces, snapshots=args.snapshots)

    # 1. Fit every target. The fit horizon is clipped by the trace; duplicates
    #    (e.g. 600 s on synthetic, which clips to 300 s) are dropped.
    fit_tasks = []
    for name in traces:
        for horizon in BINARY_HORIZONS:
            fit_tasks.append((name, "binary", horizon))
        fit_tasks.append((name, "count", 60))
        fit_tasks.append((name, "count", 600))
        fit_tasks.append((name, "next_use", 600))
    with context.Pool(min(args.workers, len(fit_tasks))) as pool:
        fits = pool.map(_fit_worker, fit_tasks)
    rankers: dict[str, dict] = {name: {} for name in traces}
    splits, fit_meta = {}, {name: [] for name in traces}
    for fit in fits:
        key = f"{fit['target']}_h{int(fit['horizon'])}"
        if key in rankers[fit["trace"]]:
            continue
        rankers[fit["trace"]][key] = fit["ranker"]
        splits[fit["trace"]] = fit["split_ms"]
        meta = {k: v for k, v in fit.items() if k != "ranker"}
        meta["coefficients"] = dict(fit["ranker"].standardized_coefficients(FEATURE_NAMES))
        fit_meta[fit["trace"]].append(meta)
        print(f"  fitted {fit['trace']} {key} (requested {fit['requested_horizon']}s) in {fit['seconds']:.0f}s", flush=True)
    _SHARED.update(rankers=rankers, splits=splits,
                   groups={name: _occurrence_groups(trace) for name, trace in traces.items()})

    # 2. Replay.
    tasks = []
    for name in traces:
        for spec in target_arm_specs(rankers[name]):
            for fraction in budgets:
                for seed in (seeds[:1] if spec.deterministic else seeds):
                    tasks.append((name, spec, fraction, seed))
    tasks.sort(key=lambda t: (t[1].name.startswith("learned"), t[2]), reverse=True)
    print(f"  {len(tasks)} replay tasks on {args.workers} workers", flush=True)
    rows = []
    started = time.time()
    raw_path = args.output_dir / "raw_results.jsonl"
    with context.Pool(args.workers) as pool, raw_path.open("w", encoding="utf-8") as raw:
        for done, out in enumerate(pool.imap_unordered(_replay_worker, tasks, chunksize=1), 1):
            rows.append(out["replay"])
            raw.write(json.dumps(out["replay"], default=str) + "\n")
            raw.flush()
            if done % 25 == 0 or done == len(tasks):
                name, arm, fraction, seed = out["task"]
                print(f"  [{done}/{len(tasks)}] {time.time()-started:.0f}s last={name}/{arm}/{fraction}/s{seed} ({out['seconds']:.0f}s)", flush=True)
    for row in [r for r in rows if r["eviction"] == "heap"]:
        for seed in seeds[1:]:
            rows.append(dict(row, seed=seed))
    add_closure(rows)
    rows.sort(key=lambda r: (r["trace"], r["policy"], float(r["capacity_fraction"]), int(r["seed"])))
    summary = aggregate_replay(rows)
    _write(args.output_dir / "target_change_seeds.csv", rows)
    _write(args.output_dir / "target_change.csv", summary)
    _write(args.paper_dir / "target_change.csv", summary)
    config = json.dumps({"hyperparameters": HYPERPARAMETERS, "budget_fractions": budgets, "seeds": list(seeds),
                         "fits": fit_meta, "note": "Same ranker, features, and hyperparameters as Phase 0.5; only the target changes."},
                        indent=2, default=str) + "\n"
    (args.output_dir / "run_config.json").write_text(config, encoding="utf-8")
    (args.paper_dir / "target_change_config.json").write_text(config, encoding="utf-8")
    make_figure(args.paper_dir, summary, args.oracle_csv)
    print("done", flush=True)


def make_figure(paper_dir: Path, summary, oracle_csv: Path) -> None:
    traces = sorted({r["trace"] for r in summary})
    oracle = list(csv.DictReader(oracle_csv.open())) if oracle_csv.exists() else []
    # Fit horizons are clipped per trace (600 s becomes 300 s on the synthetic
    # trace), so every key that a trace can produce is listed; absent ones are
    # skipped.
    arms = [("lfu", "LFU", "0.5", "-")] + [
        (f"learned_{key}", label, color, "-") for key, label, color in (
            ("binary_h600", "learned: binary 600 s (Phase 0.5 target)", "tab:blue"),
            ("binary_h300", "learned: binary 300 s", "tab:green"),
            ("binary_h60", "learned: binary 60 s", "tab:red"),
            ("count_h60", "learned: count 60 s", "tab:orange"),
            ("count_h600", "learned: count 600 s", "tab:brown"),
            ("count_h300", "learned: count 300 s (600 s clipped)", "tab:brown"),
            ("next_use_h600", "learned: next use 600 s", "tab:purple"),
            ("next_use_h300", "learned: next use 300 s (600 s clipped)", "tab:purple"),
            ("matched", "learned: horizon matched (Little's law)", "black"),
        )
    ]
    oracle_arms = [("oracle_binary_h60", "oracle 60 s", "tab:red"), ("oracle_binary", "oracle fit horizon", "tab:blue"),
                   ("oracle_next_use_sampled", "oracle next use", "black")]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.6 * len(traces), 4.4), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        for policy, label, color, style in arms:
            sel = sorted((r for r in summary if r["trace"] == trace and r["policy"] == policy), key=lambda r: float(r["capacity_fraction"]))
            sel = [r for r in sel if r["headroom_closure_mean"] not in ("", "nan") and np.isfinite(float(r["headroom_closure_mean"]))]
            if sel:
                err = [float(r["headroom_closure_ci95_half"]) if str(r["headroom_closure_ci95_half"]) not in ("", "nan") else 0.0 for r in sel]
                axis.errorbar([100 * float(r["capacity_fraction"]) for r in sel], [float(r["headroom_closure_mean"]) for r in sel],
                              yerr=err, marker="o", ms=3.5, color=color, linestyle=style, label=label, capsize=2)
        for policy, label, color in oracle_arms:
            sel = sorted((r for r in oracle if r["trace"] == trace and r["policy"] == policy), key=lambda r: float(r["capacity_fraction"]))
            if sel:
                axis.plot([100 * float(r["capacity_fraction"]) for r in sel], [float(r["headroom_closure_mean"]) for r in sel],
                          color=color, linestyle=":", linewidth=1.2, label=label)
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axhline(1.0, color="0.5", linewidth=0.8)
        axis.set(xscale="log", xlabel="cache budget (% of unique-state bytes)", ylabel="headroom closure vs sampled LRU", title=trace)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle("Same causal ranker, different target (dotted: the corresponding oracles)")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig10_target_change.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
