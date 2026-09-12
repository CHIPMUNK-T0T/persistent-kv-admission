"""Plots and concise interpretation notes for characterization outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "longest_first": "Longest-first",
    "frequency_x_prefix_length": "frequency × prefix length",
    "structural": "fan-out + branch diversity",
    "offline_next_use": "offline next-use (approx.)",
}


def make_plots(output_dir, trace_name, reuse_rows, state_rows, predictive_rows, replay_rows):
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.loglog([int(r["occurrence_count_n"]) for r in reuse_rows],
                [float(r["ccdf_p_N_ge_n"]) for r in reuse_rows], marker="o", markersize=3)
    axis.set(xlabel="Occurrences per cumulative-prefix state (N)", ylabel="P(N ≥ n)")
    axis.set_title(f"Reuse count CCDF — {trace_name}")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout(); figure.savefig(plot_dir / "reuse_count_ccdf.png", dpi=180); plt.close(figure)

    plotted = state_rows[::max(1, len(state_rows) // 100_000)]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.scatter([int(r["prefix_tokens"]) for r in plotted],
                 [int(r["occurrence_count"]) for r in plotted], s=5, alpha=0.18, linewidths=0)
    axis.set(xlabel="Cumulative prefix length (tokens)", ylabel="Occurrence count", yscale="log")
    axis.set_title(f"Prefix length vs reuse — {trace_name}")
    axis.grid(True, alpha=0.25)
    figure.tight_layout(); figure.savefig(plot_dir / "prefix_length_vs_reuse_count.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    horizons = sorted({int(r["horizon_seconds"]) for r in predictive_rows})
    for signal in ("recency", "frequency", "prefix_length", "fan_out", "branch_diversity"):
        selected = [r for r in predictive_rows if r["signal"] == signal]
        axis.plot(range(len(horizons)), [float(r["average_precision"]) for r in selected],
                  marker="o", label=signal.replace("_", " "))
    axis.set_xticks(range(len(horizons)), [_format_horizon(v) for v in horizons])
    axis.set(xlabel="Future horizon", ylabel="Mean average precision")
    axis.set_title(f"Single-signal future-reuse prediction — {trace_name}")
    axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(plot_dir / "horizon_predictive_power.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for policy in LABELS:
        selected = sorted((r for r in replay_rows if r["policy"] == policy),
                          key=lambda r: float(r["capacity_fraction"]))
        axis.plot([100 * float(r["capacity_fraction"]) for r in selected],
                  [int(r["avoided_prefill_tokens"]) for r in selected], marker="o", label=LABELS[policy])
    axis.set(xlabel="Cache budget (% of unique-state bytes)", ylabel="Avoided prefill tokens")
    axis.set_title(f"Fixed-budget prefix-dependent replay — {trace_name}")
    axis.grid(True, alpha=0.25); axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(plot_dir / "budget_vs_avoided_prefill_tokens.png", dpi=180); plt.close(figure)


def _format_horizon(seconds):
    if seconds < 60: return f"{seconds}s"
    if seconds < 3600: return f"{seconds // 60}m"
    if seconds < 86400: return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def write_interpretation(output_dir, trace_name, summary, predictive_rows, replay_rows):
    by_budget = {}
    for row in replay_rows:
        by_budget.setdefault(float(row["capacity_fraction"]), {})[str(row["policy"])] = row
    headrooms, longest_gaps = [], []
    for fraction, values in sorted(by_budget.items()):
        lru = int(values["lru"]["avoided_prefill_tokens"])
        offline = int(values["offline_next_use"]["avoided_prefill_tokens"])
        longest = int(values["longest_first"]["avoided_prefill_tokens"])
        best = max(int(v["avoided_prefill_tokens"]) for v in values.values())
        headrooms.append((fraction, 100.0 * (offline - lru) / max(lru, 1)))
        longest_gaps.append((fraction, 100.0 * (best - longest) / max(best, 1)))
    evaluated = [r for r in predictive_rows if r["status"] == "ok" and not math.isnan(float(r["average_precision"]))]
    signal_notes = []
    for horizon in sorted({int(r["horizon_seconds"]) for r in evaluated}):
        rows = [r for r in evaluated if int(r["horizon_seconds"]) == horizon]
        best = max(rows, key=lambda r: float(r["average_precision"]))
        recency = next(r for r in rows if r["signal"] == "recency")
        structural = max((r for r in rows if r["signal"] in {"fan_out", "branch_diversity"}),
                         key=lambda r: float(r["average_precision"]))
        signal_notes.append(f"- {_format_horizon(horizon)}: best={best['signal']} (AP={float(best['average_precision']):.4f}); "
                            f"recency={float(recency['average_precision']):.4f}; best structural={structural['signal']} "
                            f"({float(structural['average_precision']):.4f})")
    unavailable = [_format_horizon(int(r["horizon_seconds"])) for r in predictive_rows
                   if r["signal"] == "recency" and r["status"] != "ok"]
    all_small = all(value < 5.0 for _, value in headrooms)
    lines = [f"# Interpretation — {trace_name}", "", "## What the figures support", "",
        f"- The idealized 2-hit gate loses {100 * float(summary['two_hit_lost_avoided_prefill_fraction']):.2f}% of incremental reusable-block token opportunities before capacity eviction is considered.",
        "- LRU-to-offline-next-use headroom by budget: " + ", ".join(f"{100*f:g}%={v:.2f}%" for f,v in headrooms) + ".",
        "- Longest-first gap to the best measured comparator by budget: " + ", ".join(f"{100*f:g}%={v:.2f}%" for f,v in longest_gaps) + ".",
        f"- Provisional stop rule (all measured offline-vs-LRU gaps <5%): **{'triggered' if all_small else 'not triggered'}**.",
        "", "Single-signal future-reuse ranking:", "", *signal_notes, "", "## What the figures do not establish", "",
        "- The offline-next-use comparator has future knowledge but uses greedy leaf eviction; it is an approximate comparator, not a proven optimum for weighted, prefix-dependent caching.",
        "- Avoided prefill tokens are inferred from exact cumulative-prefix block hits. The trace has no measured GPU time, FLOPs, restore cost, or KV byte layout.",
        "- fan-out and branch diversity are request-level structural observations because this trace contains no session ID.",
        "- Requests sharing one timestamp are treated as simultaneous. They cannot create hits for one another within that timestamp bucket.",
        "- Predictive metrics use only fully observable horizons; right-censored windows are excluded." + (f" Unavailable horizons: {', '.join(unavailable)}." if unavailable else ""),
        "- Results characterize this trace and these cache budgets; they do not yet validate a deployable selection policy.", ""]
    (Path(output_dir) / "INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")
