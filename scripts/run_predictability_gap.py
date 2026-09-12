#!/usr/bin/env python3
"""Where does the predictability–retention gap come from?

Runs three measurements over every trace with one fixed hyperparameter set:

1. Oracle replay. Arms that know the future label perfectly run through the
   same sampled-leaf eviction as the causal arms, over several seeds, so the
   unrecovered headroom can be split into a signal gap, an objective gap, and
   a candidate-search gap.
2. Candidate-set prediction. Ranking quality measured on the exact leaf sets
   each eviction decision ranked, and on nested populations at the shared test
   snapshots, against the global observed-state numbers.
3. Standardisation robustness of the cross-workload transfer matrix.

No new policy is introduced. Nothing is tuned per trace.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
from persistent_kv_admission.crossworkload import (
    HYPERPARAMETERS,
    STANDARDIZATIONS,
    fit_fixed_model,
    transfer_metrics,
)
from persistent_kv_admission.gap import (
    add_closure,
    aggregate_candidate,
    aggregate_ladder,
    aggregate_replay,
    arm_specs,
    coefficient_cosines,
    decomposition_rows,
    ladder_snapshot_indices,
    run_arm,
)
from persistent_kv_admission.phase05 import HORIZONS_SECONDS
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
LOG_HORIZONS = (60, 300, 600)

# Shared with forked workers. Set once in the parent before the pool starts.
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/predictability_gap"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--budgets", default=",".join(str(v) for v in DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--precision-k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-logged-decisions", type=int, default=20_000)
    parser.add_argument("--dump-decisions", type=int, default=2_000,
                        help="candidate-level rows kept per (trace, budget) for seed 0 of learned_history")
    parser.add_argument("--skip-standardization", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    return parser.parse_args()


def _write(path: Path, rows, fieldnames=None) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("" if fieldnames is None else ",".join(fieldnames) + "\n", encoding="utf-8")
        return
    # Rows can carry different keys (e.g. status rows); union them in first-seen order.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fields} for row in rows], fields)


# --- workers ------------------------------------------------------------------

def _fit_worker(task):
    name, standardization = task
    trace = _SHARED["traces"][name]
    started = time.time()
    ranker, horizon, split_ms, positives, total = fit_fixed_model(
        trace,
        HYPERPARAMETERS["online_horizon_seconds"],
        snapshot_count=_SHARED["snapshots"],
        train_fraction=HYPERPARAMETERS["train_fraction"],
        standardization=standardization,
    )
    return {
        "trace": name,
        "standardization": standardization,
        "ranker": ranker,
        "fit_horizon_seconds": horizon,
        "split_ms": split_ms,
        "train_rows": total,
        "train_positives": positives,
        "seconds": time.time() - started,
    }


def _replay_worker(task):
    name, spec, fraction, seed = task
    trace = _SHARED["traces"][name]
    started = time.time()
    out = run_arm(
        trace,
        spec,
        fraction,
        seed,
        _SHARED["rankers"][name],
        _SHARED["splits"][name],
        _SHARED["groups"][name],
        log_horizons=LOG_HORIZONS,
        ladder_snapshots=_SHARED["snapshot_indices"][name],
        precision_k=_SHARED["precision_k"],
        max_logged_decisions=_SHARED["max_logged_decisions"],
        dump_decisions=(
            _SHARED["dump_decisions"] if (seed == 0 and spec.name == "learned_history") else 0
        ),
    )
    out["seconds"] = time.time() - started
    out["task"] = (name, spec.name, fraction, seed)
    return out


def _transfer_worker(task):
    name, standardization = task
    trace = _SHARED["traces"][name]
    rankers = _SHARED["variant_rankers"][standardization]
    return transfer_metrics(
        trace,
        rankers,
        HORIZONS_SECONDS,
        snapshot_count=_SHARED["snapshots"],
        precision_k=_SHARED["precision_k"],
        train_fraction=HYPERPARAMETERS["train_fraction"],
        standardization=standardization,
    )


# --- main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    budgets = tuple(float(value) for value in args.budgets.split(","))
    seeds = tuple(range(args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("fork")

    traces = {}
    for path in args.traces:
        trace = load_mooncake_trace(path, args.block_size)
        traces[trace.name] = trace
        print(f"loaded {trace.name}: {len(trace.requests)} requests, {len(trace.states)} states", flush=True)
    _SHARED.update(
        traces=traces,
        snapshots=args.snapshots,
        precision_k=args.precision_k,
        max_logged_decisions=args.max_logged_decisions,
        dump_decisions=args.dump_decisions,
    )

    # 1. One fitted model per trace and standardisation variant.
    variants = ("per_decision",) if args.skip_standardization else STANDARDIZATIONS
    fit_tasks = [(name, variant) for variant in variants for name in traces]
    with context.Pool(min(args.workers, len(fit_tasks))) as pool:
        fits = pool.map(_fit_worker, fit_tasks)
    variant_rankers = {variant: {} for variant in variants}
    fit_meta = {}
    for fit in fits:
        variant_rankers[fit["standardization"]][fit["trace"]] = fit["ranker"]
        if fit["standardization"] == "per_decision":
            fit_meta[fit["trace"]] = {k: v for k, v in fit.items() if k != "ranker"}
        print(f"  fitted {fit['trace']} [{fit['standardization']}] at {fit['fit_horizon_seconds']}s "
              f"in {fit['seconds']:.0f}s", flush=True)

    rankers = variant_rankers["per_decision"]
    splits = {name: fit_meta[name]["split_ms"] for name in traces}
    _SHARED.update(
        rankers=rankers,
        splits=splits,
        groups={name: _occurrence_groups(trace) for name, trace in traces.items()},
        snapshot_indices={
            name: ladder_snapshot_indices(trace, HORIZONS_SECONDS, HYPERPARAMETERS["train_fraction"], args.snapshots)
            for name, trace in traces.items()
        },
        variant_rankers=variant_rankers,
    )

    replay_rows, candidate_rows, ladder_rows, dump_rows = [], [], [], []
    if not args.skip_replay:
        # 2. Oracle replay and candidate-set logging, every seed.
        tasks = []
        for name in traces:
            for spec in arm_specs(fit_meta[name]["fit_horizon_seconds"]):
                for fraction in budgets:
                    for seed in (seeds[:1] if spec.deterministic else seeds):
                        tasks.append((name, spec, fraction, seed))
        # Slow arms first so the pool tail is short.
        cost = {"learned_history": 3, "learned_history_observed_norm": 3, "online": 3}
        tasks.sort(key=lambda t: (cost.get(t[1].name, 1), t[2]), reverse=True)
        print(f"  {len(tasks)} replay tasks on {args.workers} workers", flush=True)
        raw_path = args.output_dir / "raw_results.jsonl"
        started = time.time()
        done = 0
        with context.Pool(args.workers) as pool, raw_path.open("w", encoding="utf-8") as raw:
            for out in pool.imap_unordered(_replay_worker, tasks, chunksize=1):
                done += 1
                replay_rows.append(out["replay"])
                candidate_rows.extend(out["candidate"])
                ladder_rows.extend(out["ladder"])
                dump_rows.extend(out["dump"])
                raw.write(json.dumps({k: v for k, v in out.items() if k != "dump"}, default=str) + "\n")
                raw.flush()
                if done % 25 == 0 or done == len(tasks):
                    name, arm, fraction, seed = out["task"]
                    print(f"  [{done}/{len(tasks)}] {time.time()-started:.0f}s  last={name}/{arm}/{fraction}/s{seed} "
                          f"({out['seconds']:.0f}s)", flush=True)

        # Deterministic arms: replicate the one row across seeds so every
        # per-seed comparison has a partner.
        deterministic = [row for row in replay_rows if row["eviction"] == "heap"]
        for row in deterministic:
            for seed in seeds[1:]:
                replay_rows.append(dict(row, seed=seed))

        add_closure(replay_rows)
        replay_rows.sort(key=lambda r: (r["trace"], r["policy"], float(r["capacity_fraction"]), int(r["seed"])))
        summary = aggregate_replay(replay_rows)
        decomposition = decomposition_rows(replay_rows)
        candidate_summary = aggregate_candidate(candidate_rows)
        ladder_summary = aggregate_ladder(ladder_rows)

        _write(args.output_dir / "oracle_replay_seeds.csv", replay_rows)
        _write(args.output_dir / "candidate_set_prediction_seeds.csv", candidate_rows)
        _write(args.output_dir / "population_ladder_seeds.csv", ladder_rows)
        _write(args.output_dir / "eviction_decisions_sample.csv", dump_rows)
        for name, rows in (
            ("oracle_replay", summary),
            ("oracle_decomposition", decomposition),
            ("candidate_set_prediction", candidate_summary),
            ("population_ladder", ladder_summary),
        ):
            _write(args.output_dir / f"{name}.csv", rows)
            _write(args.paper_dir / f"{name}.csv", rows)
        make_figures(args.paper_dir, summary, decomposition, candidate_summary, ladder_summary, fit_meta)

    # 3. Standardisation robustness of the transfer matrix.
    transfer_rows, cosine_rows = [], []
    if not args.skip_standardization:
        transfer_tasks = [(name, variant) for variant in variants for name in traces]
        with context.Pool(min(args.workers, len(transfer_tasks))) as pool:
            for rows in pool.imap_unordered(_transfer_worker, transfer_tasks):
                transfer_rows.extend(rows)
        for variant in variants:
            for row in coefficient_cosines(variant_rankers[variant]):
                cosine_rows.append(dict(standardization=variant, **row))
        transfer_rows.sort(key=lambda r: (r["standardization"], r["evaluated_on"], r["fitted_on"], int(r["horizon_seconds"])))
        for name, rows in (
            ("standardization_transfer", transfer_rows),
            ("coefficient_similarity", cosine_rows),
        ):
            _write(args.output_dir / f"{name}.csv", rows)
            _write(args.paper_dir / f"{name}.csv", rows)
        print("  standardisation check done", flush=True)

    run_config = json.dumps(
        {
            "hyperparameters": HYPERPARAMETERS,
            "budget_fractions": budgets,
            "seeds": list(seeds),
            "snapshots": args.snapshots,
            "precision_k": args.precision_k,
            "log_horizons_seconds": LOG_HORIZONS,
            "max_logged_decisions": args.max_logged_decisions,
            "fits": fit_meta,
            "standardizations": list(variants),
            "note": (
                "One hyperparameter set for every workload; no per-trace tuning. Oracle "
                "arms use each trace's own fit horizon. Closure uses the same seed's "
                "sampled LRU as floor and the heap offline-next-use comparator as ceiling."
            ),
        },
        indent=2,
        default=str,
    ) + "\n"
    (args.output_dir / "run_config.json").write_text(run_config, encoding="utf-8")
    (args.paper_dir / "gap_run_config.json").write_text(run_config, encoding="utf-8")
    print("done", flush=True)


# --- figures ------------------------------------------------------------------

def _series(rows, trace, policy, key, filters=None):
    selected = [
        r for r in rows
        if r["trace"] == trace and r["policy"] == policy
        and all(r.get(k) == v for k, v in (filters or {}).items())
    ]
    selected.sort(key=lambda r: float(r["capacity_fraction"]))
    x = [100 * float(r["capacity_fraction"]) for r in selected]
    y = [float(r[f"{key}_mean"]) for r in selected]
    e = [float(r[f"{key}_ci95_half"]) if np.isfinite(float(r[f"{key}_ci95_half"])) else 0.0 for r in selected]
    return x, y, e


def make_figures(paper_dir, summary, decomposition, candidate_summary, ladder_summary, fit_meta) -> None:
    traces = sorted({row["trace"] for row in summary})

    # Figure 6 — headroom closure of oracle and causal arms, with 95% CI.
    arms = [
        ("lfu", "LFU", "tab:gray", "-"),
        ("learned_history", "learned (history)", "tab:blue", "-"),
        ("learned_history_observed_norm", "learned (history, observed-norm)", "tab:green", "-"),
        ("online", "online", "tab:cyan", "-"),
        ("oracle_binary", "oracle: binary label", "tab:red", "--"),
        ("oracle_binary_per_byte", "oracle: binary / bytes", "tab:orange", "--"),
        ("oracle_count", "oracle: reuse count", "tab:purple", "--"),
        ("oracle_next_use_sampled", "oracle: next use (sampled)", "black", "--"),
    ]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.4 * len(traces), 4.2), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        for policy, label, color, style in arms:
            x, y, e = _series(summary, trace, policy, "headroom_closure")
            if x:
                axis.errorbar(x, y, yerr=e, marker="o", ms=3.5, color=color, linestyle=style, label=label, capsize=2)
        axis.axhline(1.0, color="0.5", linewidth=0.8)
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axhspan(0.3, 0.7, color="0.9", zorder=0)
        horizon = fit_meta[trace]["fit_horizon_seconds"]
        axis.set(xscale="log", xlabel="cache budget (% of unique-state bytes)",
                 ylabel="headroom closure vs sampled LRU", title=f"{trace} (oracle label: {horizon}s)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6.5)
    figure.suptitle("Perfect knowledge of the training label recovers only part of the headroom")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig6_oracle_closure.png", dpi=170)
    plt.close(figure)

    # Figure 7 — gap decomposition, stacked per budget.
    segments = [
        ("learned_history_closure_mean", "recovered by learned history", "tab:blue"),
        ("signal_gap_closure_mean", "signal gap (binary oracle − learned)", "tab:red"),
        ("objective_gap_next_use_closure_mean", "objective gap (next-use − binary)", "tab:purple"),
        ("candidate_search_gap_closure_mean", "candidate-search gap (heap − sampled)", "0.6"),
    ]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.4 * len(traces), 4.2), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        rows = sorted((r for r in decomposition if r["trace"] == trace), key=lambda r: float(r["capacity_fraction"]))
        labels = [f"{100*float(r['capacity_fraction']):g}%" for r in rows]
        positions = np.arange(len(rows))
        positive_base = np.zeros(len(rows))
        negative_base = np.zeros(len(rows))
        for key, label, color in segments:
            values = np.array([float(r[key]) for r in rows])
            bottoms = np.where(values >= 0, positive_base, negative_base)
            axis.bar(positions, values, 0.7, bottom=bottoms, color=color, label=label)
            positive_base += np.where(values >= 0, values, 0.0)
            negative_base += np.where(values < 0, values, 0.0)
        axis.axhline(1.0, color="0.3", linewidth=0.8)
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.set_xticks(positions, labels)
        axis.set(xlabel="cache budget", ylabel="share of LRU→offline headroom", title=trace)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=6.5)
    figure.suptitle("Decomposition of the unrecovered headroom (seed means)")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig7_gap_decomposition.png", dpi=170)
    plt.close(figure)

    # Figure 8 — population ladder: AUC from observed states down to the decision.
    stages = ["observed_protocol", "observed", "cached", "leaves", "decision"]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.4 * len(traces), 4.2), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        horizon = fit_meta[trace]["fit_horizon_seconds"]
        budgets = sorted({float(r["capacity_fraction"]) for r in ladder_summary if r["trace"] == trace})
        shown = [b for b in budgets if b in (0.0025, 0.01, 0.05)] or budgets[:3]
        for policy, style in (("learned_history", "-"), ("learned_history_observed_norm", "-."), ("lfu", "--")):
            for budget in shown:
                values = []
                for stage in stages:
                    if stage == "decision":
                        match = [r for r in candidate_summary if r["trace"] == trace and r["policy"] == policy
                                 and float(r["capacity_fraction"]) == budget and int(r["horizon_seconds"]) == horizon]
                        values.append(float(match[0]["within_decision_auc_micro_mean"]) if match else np.nan)
                    else:
                        match = [r for r in ladder_summary if r["trace"] == trace and r["policy"] == policy
                                 and float(r["capacity_fraction"]) == budget and r["population"] == stage
                                 and int(r["horizon_seconds"]) == horizon]
                        values.append(float(match[0]["auc_mean"]) if match else np.nan)
                axis.plot(range(len(stages)), values, marker="o", linestyle=style,
                          label=f"{policy} @ {100*budget:g}%")
        axis.axhline(0.5, color="0.5", linewidth=0.8)
        axis.set_xticks(range(len(stages)), ["observed\n(protocol)", "observed\n(replay score)", "cached", "leaves", "decision\ncandidates"], fontsize=7)
        axis.set(ylabel=f"AUC, {horizon}s reuse label", title=trace, ylim=(0.0, 1.0))
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle("Ranking skill by population: where the prediction skill goes")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig8_population_ladder.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
