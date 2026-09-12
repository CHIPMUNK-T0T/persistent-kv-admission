#!/usr/bin/env python3
"""Does a stronger history model rank the eviction candidates better?

Replays the learned_history arm once per trace and budget, logging every
sampled eviction decision with the raw 23 causal features of its candidates.
Fits, on the same candidate sets, single-feature orderings (recency, frequency),
the linear ranker, a gradient-boosted model, and a gradient-boosted model with
within-decision rank context; trains on decisions before the Phase 0.5 split
with a horizon embargo and tests on the evaluation window. One hyperparameter
set for everything. Nothing is tuned per trace.

Requires scikit-learn (see README: `.venv/bin/python scripts/run_candidate_models.py ...`).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

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

from persistent_kv_admission.candidate_models import (
    GBM_PARAMS,
    evaluate_candidate_models,
    log_candidate_features,
)
from persistent_kv_admission.characterize import write_csv
from persistent_kv_admission.crossworkload import HYPERPARAMETERS, fit_fixed_model
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.0025, 0.01, 0.02, 0.05, 0.1)
HORIZONS = (60, 300, 600)
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/candidate_models"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--budgets", default=",".join(str(v) for v in DEFAULT_BUDGETS))
    parser.add_argument("--max-decisions", type=int, default=40_000)
    parser.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _write(path: Path, rows) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fields} for row in rows], fields)


def _worker(task):
    name, fraction = task
    trace = _SHARED["traces"][name]
    started = time.time()
    logger = log_candidate_features(
        trace,
        _SHARED["rankers"][name],
        fraction,
        seed=_SHARED["seed"],
        max_decisions=_SHARED["max_decisions"],
        occurrence_groups=_SHARED["groups"][name],
    )
    logged = time.time() - started
    horizons = tuple(h for h in HORIZONS if _SHARED["splits"][name] + h * 1000.0 <= trace.end_ms)
    rows = evaluate_candidate_models(logger, trace, _SHARED["splits"][name], horizons)
    for row in rows:
        row.update(trace=name, capacity_fraction=fraction, seed=_SHARED["seed"],
                   decisions_seen=logger.decisions_seen, decisions_kept=len(logger.kept))
    return {"task": task, "rows": rows, "seconds": time.time() - started, "log_seconds": logged}


def main() -> None:
    args = parse_args()
    budgets = tuple(float(v) for v in args.budgets.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("fork")

    traces, rankers, splits, groups, fit_meta = {}, {}, {}, {}, {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        traces[trace.name] = trace
        ranker, horizon, split_ms, positives, total = fit_fixed_model(
            trace, HYPERPARAMETERS["online_horizon_seconds"], snapshot_count=args.snapshots,
            train_fraction=HYPERPARAMETERS["train_fraction"],
        )
        rankers[trace.name], splits[trace.name] = ranker, split_ms
        groups[trace.name] = _occurrence_groups(trace)
        fit_meta[trace.name] = {"fit_horizon_seconds": horizon, "split_ms": split_ms,
                                "train_rows": total, "train_positives": positives}
        print(f"loaded and fitted {trace.name} (horizon {horizon}s)", flush=True)
    _SHARED.update(traces=traces, rankers=rankers, splits=splits, groups=groups,
                   seed=args.seed, max_decisions=args.max_decisions)

    tasks = [(name, fraction) for name in traces for fraction in budgets]
    tasks.sort(key=lambda t: t[1], reverse=True)
    rows = []
    started = time.time()
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for out in pool.imap_unordered(_worker, tasks):
            rows.extend(out["rows"])
            name, fraction = out["task"]
            print(f"  {name}/{fraction}: {out['seconds']:.0f}s (logging {out['log_seconds']:.0f}s), "
                  f"{time.time()-started:.0f}s elapsed", flush=True)
    rows.sort(key=lambda r: (r["trace"], float(r["capacity_fraction"]), int(r["horizon_seconds"]), r.get("model", "")))
    _write(args.output_dir / "candidate_model_check.csv", rows)
    _write(args.paper_dir / "candidate_model_check.csv", rows)
    config = json.dumps({"gbm_params": GBM_PARAMS, "budgets": budgets, "horizons": HORIZONS,
                         "max_decisions": args.max_decisions, "seed": args.seed, "fits": fit_meta,
                         "note": "One hyperparameter set for every trace, budget, and horizon."},
                        indent=2, default=str) + "\n"
    (args.output_dir / "run_config.json").write_text(config, encoding="utf-8")
    (args.paper_dir / "candidate_model_run_config.json").write_text(config, encoding="utf-8")
    make_figure(args.paper_dir, rows, fit_meta)
    print("done", flush=True)


def make_figure(paper_dir: Path, rows, fit_meta) -> None:
    traces = sorted({r["trace"] for r in rows})
    models = [("recency", "recency (LRU order)", "0.6", "--"), ("frequency", "frequency (LFU order)", "0.3", "--"),
              ("arm", "deployed linear arm", "tab:cyan", "-"), ("linear", "linear, refit on candidates", "tab:blue", "-"),
              ("gbm", "gradient boosting", "tab:red", "-"), ("gbm_plus_context", "gradient boosting + decision context", "tab:purple", "-")]
    figure, axes = plt.subplots(1, len(traces), figsize=(5.4 * len(traces), 4.2), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        horizon = fit_meta[trace]["fit_horizon_seconds"]
        for model, label, color, style in models:
            selected = sorted((r for r in rows if r["trace"] == trace and r.get("model") == model
                               and int(r["horizon_seconds"]) == horizon), key=lambda r: float(r["capacity_fraction"]))
            if selected:
                axis.plot([100 * float(r["capacity_fraction"]) for r in selected],
                          [float(r["within_decision_auc_micro"]) for r in selected],
                          marker="o", ms=3.5, color=color, linestyle=style, label=label)
        axis.axhline(0.5, color="0.5", linewidth=0.8)
        axis.set(xscale="log", xlabel="cache budget (% of unique-state bytes)",
                 ylabel=f"within-decision AUC, {horizon}s label", title=trace, ylim=(0.0, 1.0))
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6.5)
    figure.suptitle("Ranking the eviction candidates: model capacity on the same 23 history features")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig9_candidate_model_check.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
