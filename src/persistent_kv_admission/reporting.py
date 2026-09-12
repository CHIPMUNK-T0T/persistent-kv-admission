"""Plots and evidence-bounded interpretation for characterization outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "lru": "LRU",
    "lru_2hit": "2-hit + LRU",
    "lfu": "LFU",
    "longest_first": "Longest-first",
    "frequency_x_prefix_length": "frequency × prefix length",
    "structural": "fan-out + branch diversity",
    "offline_next_use": "offline next-use (approx.)",
}
ONLINE_POLICIES = tuple(policy for policy in LABELS if policy != "offline_next_use")


def _format_horizon(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def make_plots(output_dir, trace_name, reuse_rows, state_rows, predictive_rows,
               replay_rows, incremental_rows):
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.loglog([int(row["occurrence_count_n"]) for row in reuse_rows],
                [float(row["ccdf_p_N_ge_n"]) for row in reuse_rows], marker="o", markersize=3)
    axis.set(xlabel="Occurrences per cumulative-prefix state (N)", ylabel="P(N ≥ n)")
    axis.set_title(f"Reuse count CCDF — {trace_name}")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "reuse_count_ccdf.png", dpi=180)
    plt.close(figure)

    plotted = state_rows[::max(1, len(state_rows) // 100_000)]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.scatter([int(row["prefix_tokens"]) for row in plotted],
                 [int(row["occurrence_count"]) for row in plotted],
                 s=5, alpha=0.18, linewidths=0)
    axis.set(xlabel="Cumulative prefix length (tokens)", ylabel="Occurrence count", yscale="log")
    axis.set_title(f"Prefix length vs reuse — {trace_name}")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "prefix_length_vs_reuse_count.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    horizons = sorted({int(row["horizon_seconds"]) for row in predictive_rows})
    for signal in ("recency", "frequency", "prefix_length", "fan_out", "branch_diversity"):
        selected = [row for row in predictive_rows if row["signal"] == signal]
        axis.plot(range(len(horizons)), [float(row["average_precision"]) for row in selected],
                  marker="o", label=signal.replace("_", " "))
    axis.set_xticks(range(len(horizons)), [_format_horizon(value) for value in horizons])
    axis.set(xlabel="Future horizon", ylabel="Mean average precision")
    axis.set_title(f"Single-signal future-reuse prediction — {trace_name}")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "horizon_predictive_power.png", dpi=180)
    plt.close(figure)

    packed = [row for row in replay_rows if row["size_model"] == "packed"]
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for policy in ONLINE_POLICIES:
        selected = sorted((row for row in packed if row["policy"] == policy),
                          key=lambda row: float(row["capacity_fraction"]))
        axis.plot([100 * float(row["capacity_fraction"]) for row in selected],
                  [int(row["avoided_prefill_tokens"]) for row in selected],
                  marker="o", label=LABELS[policy])
    axis.set(xlabel="Cache budget (% of packed unique-state bytes)", ylabel="Avoided prefill tokens")
    axis.set_title(f"Causal online policy comparison — {trace_name}")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plot_dir / "budget_vs_avoided_prefill_tokens.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for policy in ("lru", "offline_next_use"):
        selected = sorted((row for row in packed if row["policy"] == policy),
                          key=lambda row: float(row["capacity_fraction"]))
        axis.plot([100 * float(row["capacity_fraction"]) for row in selected],
                  [int(row["avoided_prefill_tokens"]) for row in selected],
                  marker="o", label=LABELS[policy])
    axis.set(xlabel="Cache budget (% of packed unique-state bytes)", ylabel="Avoided prefill tokens")
    axis.set_title(f"Offline headroom comparator — {trace_name}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "offline_next_use_headroom.png", dpi=180)
    plt.close(figure)

    extended = [row for row in incremental_rows if row["model"] == "extended"]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    valid = [row for row in extended if row["status"] == "ok"]
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot([_format_horizon(int(row["horizon_seconds"])) for row in valid],
              [float(row["delta_ap_vs_base"]) for row in valid], marker="o", label="Δ AP")
    axis.plot([_format_horizon(int(row["horizon_seconds"])) for row in valid],
              [float(row["delta_auc_vs_base"]) for row in valid], marker="s", label="Δ AUC")
    axis.set(xlabel="Held-out future horizon", ylabel="Extended − Base")
    axis.set_title(f"Incremental structural value — {trace_name}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "structural_incremental_value.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for policy in ONLINE_POLICIES:
        values = []
        fractions = []
        for fraction in sorted({float(row["capacity_fraction"]) for row in replay_rows}):
            packed_value = next(int(row["avoided_prefill_tokens"]) for row in replay_rows
                                if row["size_model"] == "packed" and row["policy"] == policy
                                and float(row["capacity_fraction"]) == fraction)
            fixed_value = next(int(row["avoided_prefill_tokens"]) for row in replay_rows
                               if row["size_model"] == "fixed_block" and row["policy"] == policy
                               and float(row["capacity_fraction"]) == fraction)
            fractions.append(100 * fraction)
            values.append((fixed_value - packed_value) / max(packed_value, 1))
        axis.plot(fractions, values, marker="o", label=LABELS[policy])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Cache budget (% of packed unique-state bytes)",
             ylabel="Fixed-block vs packed avoided-token change")
    axis.set_title(f"Capacity size-model sensitivity — {trace_name}")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plot_dir / "capacity_size_model_sensitivity.png", dpi=180)
    plt.close(figure)


def write_interpretation(output_dir, trace_name, summary, online_rows, headroom_rows,
                         sensitivity_rows, predictive_rows, incremental_rows,
                         stratified_rows, correlation_rows):
    lines = [f"# Interpretation — {trace_name}", "", "## What the figures support", ""]
    lines.append(
        f"- The unbounded 2-hit accounting loses {100 * float(summary['two_hit_lost_avoided_prefill_fraction']):.2f}% "
        "of incremental reusable-block token opportunities; the capacity-aware comparison below includes pollution avoidance."
    )
    for size_model in ("packed", "fixed_block"):
        rows = [row for row in online_rows if row["size_model"] == size_model]
        budgets = sorted({float(row["capacity_fraction"]) for row in rows})
        two_hit_gains, longest_comparisons = [], defaultdict(list)
        for fraction in budgets:
            values = {str(row["policy"]): int(row["avoided_prefill_tokens"])
                      for row in rows if float(row["capacity_fraction"]) == fraction}
            two_hit_gains.append(100 * (values["lru_2hit"] - values["lru"]) / max(values["lru"], 1))
            for policy in ("lru", "lfu", "frequency_x_prefix_length", "structural"):
                longest_comparisons[policy].append(
                    100 * (values["longest_first"] - values[policy]) / max(values[policy], 1)
                )
        lines.append(
            f"- {size_model}: capacity-aware 2-hit minus LRU ranges from "
            f"{min(two_hit_gains):.2f}% to {max(two_hit_gains):.2f}% across budgets."
        )
        lines.append(
            f"- {size_model}: Longest-first minus causal comparators: "
            + "; ".join(f"{LABELS[policy]} {min(values):.2f}% to {max(values):.2f}%"
                        for policy, values in longest_comparisons.items()) + "."
        )
    for size_model in ("packed", "fixed_block"):
        values = [100 * float(row["headroom_vs_lru_fraction"])
                  for row in headroom_rows if row["size_model"] == size_model]
        lines.append(
            f"- {size_model}: approximate offline-next-use headroom over LRU is "
            f"{min(values):.2f}% to {max(values):.2f}%; this is used only as a headroom indicator."
        )
    valid_extended = [row for row in incremental_rows if row["model"] == "extended"
                      and row["status"] == "ok"]
    lines.extend(["", "Held-out Base vs Extended linear ranking:", ""])
    for row in valid_extended:
        precision_gain_key = next(
            key for key in row if key.startswith("delta_precision_at_") and key.endswith("_vs_base")
        )
        precision_k = precision_gain_key.removeprefix("delta_precision_at_").removesuffix(
            "_vs_base"
        )
        lines.append(
            f"- {_format_horizon(int(row['horizon_seconds']))}: ΔAUC={float(row['delta_auc_vs_base']):+.4f}, "
            f"ΔAP={float(row['delta_ap_vs_base']):+.4f}, ΔP@{precision_k}={float(row[precision_gain_key]):+.4f}."
        )
    positive = sum(float(row["delta_auc_vs_base"]) > 0 and float(row["delta_ap_vs_base"]) > 0
                   for row in valid_extended)
    lines.append(
        f"- Structural incrementality is positive in both AUC and AP for {positive}/{len(valid_extended)} "
        "observable held-out horizons in this trace. Cross-trace consistency determines the research conclusion."
    )
    if stratified_rows:
        for control in ("frequency", "recency"):
            values = [float(row["auc"]) for row in stratified_rows if row["control"] == control]
            if values:
                lines.append(
                    f"- Within approximate {control} strata, structural-signal AUC ranges from "
                    f"{min(values):.4f} to {max(values):.4f}."
                )
    if correlation_rows:
        values = [abs(float(row["mean_pearson_correlation"])) for row in correlation_rows]
        lines.append(
            f"- Absolute transformed-feature correlation between base and structural signals ranges from "
            f"{min(values):.4f} to {max(values):.4f}; see `feature_correlations.csv` for redundancy details."
        )
    max_sensitivity = max(abs(float(row["fixed_vs_packed_change_fraction"]))
                          for row in sensitivity_rows)
    lines.extend(["", "## What the figures do not establish", "",
        "- The approximate offline-next-use comparator has future knowledge and greedy leaf eviction. It is not included in causal online-policy rankings and is not a proven optimum.",
        "- The Base/Extended models are ridge linear rankers trained on earlier snapshots with a horizon embargo. A gain shows incremental predictive information under this specification, not causality or deployment benefit.",
        "- Stratification uses within-snapshot quantile bins. It reduces frequency or recency variation but does not make matched causal pairs.",
        f"- The largest fixed-block vs packed change in avoided tokens is {100 * max_sensitivity:.2f}%; model-specific details are in `capacity_sensitivity.csv`.",
        "- Avoided prefill tokens are inferred from exact cumulative-prefix hits. The trace has no measured GPU time, FLOPs, restore cost, or physical KV layout.",
        "- fan-out and branch diversity are request-level structural observations because this trace contains no session ID.",
        "- Equal timestamps are treated as simultaneous. Fully observable horizon windows only are used.",
        "- Structural signal can predict reuse, but whether its relative advantage increases at persistent-cache timescales remains unresolved.", ""])
    (Path(output_dir) / "INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")


def make_paper_figures(paper_dir: Path, online_rows, headroom_rows, incremental_rows):
    paper_dir.mkdir(parents=True, exist_ok=True)
    traces = sorted({str(row["trace"]) for row in online_rows})
    figure, axes = plt.subplots(1, len(traces), figsize=(6 * len(traces), 4.5), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        for policy in ONLINE_POLICIES:
            selected = sorted((row for row in online_rows if row["trace"] == trace
                               and row["size_model"] == "packed" and row["policy"] == policy),
                              key=lambda row: float(row["capacity_fraction"]))
            axis.plot([100 * float(row["capacity_fraction"]) for row in selected],
                      [int(row["avoided_prefill_tokens"]) for row in selected], marker="o", label=LABELS[policy])
        axis.set_title(trace)
        axis.set_xlabel("Packed cache budget (%)")
        axis.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Avoided prefill tokens")
    axes[0][-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(paper_dir / "online_policy_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for trace in traces:
        selected = sorted((row for row in incremental_rows if row["trace"] == trace
                           and row["model"] == "extended" and row["status"] == "ok"),
                          key=lambda row: int(row["horizon_seconds"]))
        axis.plot([_format_horizon(int(row["horizon_seconds"])) for row in selected],
                  [float(row["delta_ap_vs_base"]) for row in selected], marker="o", label=trace)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Held-out future horizon", ylabel="Extended − Base average precision")
    axis.set_title("Incremental value of fan-out and branch diversity")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paper_dir / "structural_incremental_ap.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(traces), figsize=(5.5 * len(traces), 4.2), squeeze=False)
    for axis, trace in zip(axes[0], traces):
        for size_model in ("packed", "fixed_block"):
            selected = sorted((row for row in headroom_rows if row["trace"] == trace
                               and row["size_model"] == size_model),
                              key=lambda row: float(row["capacity_fraction"]))
            axis.plot([100 * float(row["capacity_fraction"]) for row in selected],
                      [100 * float(row["headroom_vs_lru_fraction"]) for row in selected],
                      marker="o", label=size_model)
        axis.axhline(5.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(trace)
        axis.set_xlabel("Cache budget (%)")
        axis.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Offline-next-use headroom over LRU (%)")
    axes[0][-1].legend()
    figure.tight_layout()
    figure.savefig(paper_dir / "offline_headroom_sensitivity.png", dpi=180)
    plt.close(figure)
