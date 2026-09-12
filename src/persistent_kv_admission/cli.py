"""Command-line entry point for reproducible trace characterization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .characterize import (
    inter_arrival_rows,
    longest_online_comparison_rows,
    predictive_power_rows,
    replay_comparison_rows,
    replay_rows,
    reuse_mass_rows,
    state_feature_rows,
    trace_summary,
    write_csv,
)
from .reporting import make_paper_figures, make_plots, write_interpretation
from .structural import evaluate_structural_incrementality
from .trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/characterization"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument(
        "--bytes-per-token",
        type=int,
        default=2048,
        help="KV size proxy; does not affect token-hit rankings",
    )
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--precision-k", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = tuple(float(value) for value in args.budgets.split(","))
    if any(value <= 0 or value > 1 for value in budgets):
        raise SystemExit("all budget fractions must be in (0, 1]")

    summaries = []
    all_online, all_longest, all_headroom, all_sensitivity = [], [], [], []
    all_incremental, all_stratified, all_correlations, all_coefficients = [], [], [], []
    for trace_path in args.traces:
        trace = load_mooncake_trace(trace_path, args.block_size)
        output_dir = args.output_dir / trace.name
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = trace_summary(trace, args.bytes_per_token)
        inter_arrivals, inter_summary = inter_arrival_rows(trace)
        summary.update(inter_summary)
        states = state_feature_rows(trace, args.bytes_per_token)
        reuse = reuse_mass_rows(trace)
        predictive = predictive_power_rows(trace, args.snapshots, args.precision_k)
        incremental, stratified, correlations, coefficients = evaluate_structural_incrementality(
            trace, args.snapshots, args.precision_k, args.ridge
        )
        for collection in (incremental, stratified, correlations, coefficients):
            for row in collection:
                row["trace"] = trace.name
        replayed = replay_rows(trace, budgets, args.bytes_per_token)
        online, headroom, sensitivity = replay_comparison_rows(replayed)
        longest = longest_online_comparison_rows(online)

        write_csv(output_dir / "trace_summary.csv", [summary])
        write_csv(output_dir / "reuse_count_distribution.csv", reuse)
        write_csv(output_dir / "inter_arrival_distribution.csv", inter_arrivals)
        write_csv(output_dir / "state_features.csv", states)
        write_csv(output_dir / "predictive_power.csv", predictive)
        write_csv(output_dir / "structural_incremental.csv", incremental)
        write_csv(output_dir / "structural_stratified.csv", stratified)
        write_csv(output_dir / "feature_correlations.csv", correlations)
        write_csv(output_dir / "structural_coefficients.csv", coefficients)
        write_csv(output_dir / "cache_replay.csv", replayed)
        write_csv(output_dir / "online_policy_comparison.csv", online)
        write_csv(output_dir / "longest_online_comparison.csv", longest)
        write_csv(output_dir / "offline_headroom.csv", headroom)
        write_csv(output_dir / "capacity_sensitivity.csv", sensitivity)
        make_plots(output_dir, trace.name, reuse, states, predictive, replayed, incremental)
        write_interpretation(
            output_dir,
            trace.name,
            summary,
            online,
            headroom,
            sensitivity,
            predictive,
            incremental,
            stratified,
            correlations,
        )
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "trace": str(trace_path),
                    "block_size": args.block_size,
                    "bytes_per_token": args.bytes_per_token,
                    "size_models": ["packed", "fixed_block"],
                    "snapshots": args.snapshots,
                    "precision_k": args.precision_k,
                    "ridge": args.ridge,
                    "budget_fractions": budgets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
        all_online.extend(online)
        all_longest.extend(longest)
        all_headroom.extend(headroom)
        all_sensitivity.extend(sensitivity)
        all_incremental.extend(incremental)
        all_stratified.extend(stratified)
        all_correlations.extend(correlations)
        all_coefficients.extend(coefficients)
        print(f"completed {trace.name}: {len(trace.requests)} requests, {len(trace.states)} states")

    write_csv(args.output_dir / "trace_summary.csv", summaries)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.paper_dir / "trace_summary.csv", summaries)
    write_csv(args.paper_dir / "online_policy_comparison.csv", all_online)
    write_csv(args.paper_dir / "longest_online_comparison.csv", all_longest)
    write_csv(args.paper_dir / "offline_headroom.csv", all_headroom)
    write_csv(args.paper_dir / "capacity_sensitivity.csv", all_sensitivity)
    write_csv(args.paper_dir / "structural_incremental.csv", all_incremental)
    write_csv(args.paper_dir / "structural_stratified.csv", all_stratified)
    write_csv(args.paper_dir / "feature_correlations.csv", all_correlations)
    write_csv(args.paper_dir / "structural_coefficients.csv", all_coefficients)
    make_paper_figures(args.paper_dir, all_online, all_headroom, all_incremental)


if __name__ == "__main__":
    main()
