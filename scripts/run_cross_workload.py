#!/usr/bin/env python3
"""Phase 0.5 — can retention adapt to a workload without per-workload tuning?

Runs three arms over every trace under one fixed hyperparameter set:
a model fitted on the same workload, the same model fitted on a different
workload, and an online learner that starts from nothing.
"""
from __future__ import annotations

import argparse
import json
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
    fit_fixed_model,
    run_budget_sweep,
    transfer_metrics,
)
from persistent_kv_admission.phase05 import HORIZONS_SECONDS, evaluate_temporal_prediction
from persistent_kv_admission.trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.0025, 0.01, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/crossworkload"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--budgets", default=",".join(str(v) for v in DEFAULT_BUDGETS))
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--precision-k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def _write(path: Path, rows, fieldnames=None) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("" if fieldnames is None else ",".join(fieldnames) + "\n", encoding="utf-8")
        return
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    budgets = tuple(float(value) for value in args.budgets.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    traces = {}
    for path in args.traces:
        trace = load_mooncake_trace(path, args.block_size)
        traces[trace.name] = trace
        print(f"loaded {trace.name}: {len(trace.requests)} requests, {len(trace.states)} states", flush=True)

    # 1. Prediction-side ablation on each workload, and one fitted model per workload.
    prediction_rows, coefficient_rows, correlation_rows = [], [], []
    rankers, fit_meta = {}, {}
    for name, trace in traces.items():
        started = time.time()
        rows, coefficients, correlations, _, meta = evaluate_temporal_prediction(
            trace,
            snapshot_count=args.snapshots,
            precision_k=args.precision_k,
            l2=HYPERPARAMETERS["offline_l2"],
            train_fraction=HYPERPARAMETERS["train_fraction"],
        )
        prediction_rows.extend(rows)
        coefficient_rows.extend(coefficients)
        correlation_rows.extend(correlations)
        ranker, horizon, split_ms, positives, total = fit_fixed_model(
            trace,
            HYPERPARAMETERS["online_horizon_seconds"],
            snapshot_count=args.snapshots,
            train_fraction=HYPERPARAMETERS["train_fraction"],
        )
        rankers[name] = ranker
        fit_meta[name] = {
            "trace": name,
            "fit_horizon_seconds": horizon,
            "split_ms": split_ms,
            "split_seconds": split_ms / 1000.0,
            "train_rows": total,
            "train_positives": positives,
            "converged": ranker.converged,
            "usable_horizons": meta["usable_horizons"],
            "unusable_horizons": meta["unusable_horizons"],
        }
        print(f"  fitted {name} at horizon {horizon}s in {time.time()-started:.0f}s", flush=True)

    # 2. Cross-workload prediction transfer.
    transfer_rows = []
    for name, trace in traces.items():
        transfer_rows.extend(
            transfer_metrics(
                trace,
                rankers,
                HORIZONS_SECONDS,
                snapshot_count=args.snapshots,
                precision_k=args.precision_k,
                train_fraction=HYPERPARAMETERS["train_fraction"],
            )
        )
        print(f"  transfer metrics done for {name}", flush=True)

    # 3. Fixed-budget replay with self-fitted, cross-fitted, and online arms.
    replay_rows = []
    for name, trace in traces.items():
        fixed_models = {"fixed_self": rankers[name]}
        for other in traces:
            if other != name:
                fixed_models[f"fixed_cross_{other}"] = rankers[other]
        started = time.time()
        replay_rows.extend(
            run_budget_sweep(trace, fixed_models, budgets, fit_meta[name]["split_ms"])
        )
        print(f"  replay sweep done for {name} in {time.time()-started:.0f}s", flush=True)

    _write(args.output_dir / "prediction_ablation.csv", prediction_rows)
    _write(args.output_dir / "model_coefficients.csv", coefficient_rows)
    _write(args.output_dir / "feature_correlations_spearman.csv", correlation_rows)
    _write(args.output_dir / "cross_workload_transfer.csv", transfer_rows)
    _write(args.output_dir / "policy_replay.csv", replay_rows)
    for name, rows in (
        ("prediction_ablation", prediction_rows),
        ("cross_workload_transfer", transfer_rows),
        ("policy_replay", replay_rows),
        # The coefficient vectors are the evidence for the cross-workload claim,
        # so they belong with the review artifacts and not only in the full dump.
        ("model_coefficients", coefficient_rows),
    ):
        _write(args.paper_dir / f"{name}.csv", rows)
    run_config = (
        json.dumps(
            {
                "hyperparameters": HYPERPARAMETERS,
                "budget_fractions": budgets,
                "snapshots": args.snapshots,
                "precision_k": args.precision_k,
                "fits": fit_meta,
                "note": "One hyperparameter set is used for every workload. No per-trace tuning.",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    # Written to both places: the no-per-workload-tuning claim has to be checkable
    # from the review artifacts alone.
    (args.output_dir / "run_config.json").write_text(run_config, encoding="utf-8")
    (args.paper_dir / "run_config.json").write_text(run_config, encoding="utf-8")
    make_figures(args.paper_dir, prediction_rows, transfer_rows, replay_rows)
    print("done", flush=True)


def make_figures(paper_dir: Path, prediction_rows, transfer_rows, replay_rows) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    traces = sorted({row["trace"] for row in replay_rows})

    # Figure 1 — prediction ablation by horizon.
    figure, axes = plt.subplots(1, len(traces), figsize=(5.2 * len(traces), 4.0), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        for variant in ("base0", "base1", "base2"):
            rows = sorted(
                (r for r in prediction_rows if r["trace"] == trace and r["variant"] == variant),
                key=lambda r: int(r["horizon_seconds"]),
            )
            if rows:
                axis.plot([int(r["horizon_seconds"]) for r in rows],
                          [float(r["average_precision"]) for r in rows], marker="o", label=variant)
        axis.set(xscale="log", xlabel="horizon (s)", ylabel="mean average precision", title=trace)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Base-0 / Base-1 / Base-2 future-reuse prediction")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig1_prediction_ablation.png", dpi=170)
    plt.close(figure)

    # Figure 2 — avoided prefill tokens by budget.
    figure, axes = plt.subplots(1, len(traces), figsize=(5.2 * len(traces), 4.0), squeeze=False)
    policies = ["lru", "lfu", "online", "fixed_self", "offline_next_use"]
    for axis, trace in zip(axes[0], traces):
        for policy in policies:
            rows = sorted(
                (r for r in replay_rows if r["trace"] == trace and r["policy"] == policy),
                key=lambda r: float(r["capacity_fraction"]),
            )
            if rows:
                axis.plot([100 * float(r["capacity_fraction"]) for r in rows],
                          [int(r["avoided_prefill_tokens"]) for r in rows], marker="o", label=policy)
        axis.set(xlabel="cache budget (% of unique-state bytes)",
                 ylabel="avoided prefill tokens", title=trace)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Fixed-budget replay, evaluation window only")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig2_budget_avoided_tokens.png", dpi=170)
    plt.close(figure)

    # Figure 3 — headroom closure.
    figure, axes = plt.subplots(1, len(traces), figsize=(5.2 * len(traces), 4.0), squeeze=False)
    closure_policies = ["lfu", "structural", "fixed_self", "online"]
    for axis, trace in zip(axes[0], traces):
        for policy in closure_policies:
            rows = sorted(
                (r for r in replay_rows if r["trace"] == trace and r["policy"] == policy),
                key=lambda r: float(r["capacity_fraction"]),
            )
            rows = [r for r in rows if r.get("headroom_closure") is not None]
            if rows:
                axis.plot([100 * float(r["capacity_fraction"]) for r in rows],
                          [float(r["headroom_closure"]) for r in rows], marker="o", label=policy)
        for policy in [p for p in {r["policy"] for r in replay_rows} if p.startswith("fixed_cross")]:
            rows = sorted(
                (r for r in replay_rows if r["trace"] == trace and r["policy"] == policy),
                key=lambda r: float(r["capacity_fraction"]),
            )
            if rows:
                axis.plot([100 * float(r["capacity_fraction"]) for r in rows],
                          [float(r["headroom_closure"]) for r in rows], marker="x",
                          linestyle="--", label=policy.replace("fixed_cross_", "cross:"))
        axis.axhline(0.0, color="0.4", linewidth=1)
        axis.set(xlabel="cache budget (%)", ylabel="headroom closure vs LRU", title=trace)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle("Share of the LRU-to-offline gap that a causal policy recovers")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig3_headroom_closure.png", dpi=170)
    plt.close(figure)

    # Figure 4 — feature-group ablation, delta AUC against base0.
    groups = ["rate", "interarrival", "window", "dynamics", "structural"]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.2 * len(traces), 4.0), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        horizons = sorted({int(r["horizon_seconds"]) for r in prediction_rows if r["trace"] == trace})
        width = 0.8 / max(len(groups), 1)
        for offset, group in enumerate(groups):
            values = []
            for horizon in horizons:
                match = [r for r in prediction_rows
                         if r["trace"] == trace and int(r["horizon_seconds"]) == horizon
                         and r["variant"] == f"base0_plus_{group}"]
                values.append(float(match[0]["delta_auc_vs_base0"]) if match else np.nan)
            axis.bar(np.arange(len(horizons)) + offset * width, values, width, label=group)
        axis.axhline(0.0, color="0.3", linewidth=1)
        axis.set_xticks(np.arange(len(horizons)) + 0.4, [f"{h}s" for h in horizons])
        axis.set(ylabel="ΔAUC vs base0", title=trace)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle("Incremental ranking value of each feature group")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig4_feature_group_ablation.png", dpi=170)
    plt.close(figure)

    # Figure 5 — cross-workload transfer matrix.
    if transfer_rows:
        sources = sorted({r["fitted_on"] for r in transfer_rows})
        targets = sorted({r["evaluated_on"] for r in transfer_rows})
        # Use the longest horizon every evaluated trace can support, otherwise a
        # short trace contributes an empty column and the matrix looks broken.
        per_target = {
            name: {int(r["horizon_seconds"]) for r in transfer_rows if r["evaluated_on"] == name}
            for name in targets
        }
        shared = set.intersection(*per_target.values()) if per_target else set()
        target = max(shared) if shared else max(
            int(r["horizon_seconds"]) for r in transfer_rows
        )
        matrix = np.full((len(sources), len(targets)), np.nan)
        for row in transfer_rows:
            if int(row["horizon_seconds"]) != target:
                continue
            matrix[sources.index(row["fitted_on"]), targets.index(row["evaluated_on"])] = float(row["auc"])
        figure, axis = plt.subplots(figsize=(1.6 * len(targets) + 3.2, 1.2 * len(sources) + 2.6))
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(targets)), targets, rotation=20, ha="right", fontsize=8)
        axis.set_yticks(range(len(sources)), sources, fontsize=8)
        for i in range(len(sources)):
            for j in range(len(targets)):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                              color="white", fontsize=9)
        axis.set(xlabel="evaluated on", ylabel="fitted on",
                 title=f"Cross-workload transfer AUC at {target}s")
        figure.colorbar(image, ax=axis, shrink=0.85)
        figure.tight_layout()
        figure.savefig(paper_dir / "fig5_cross_workload_transfer.png", dpi=170)
        plt.close(figure)


if __name__ == "__main__":
    main()
