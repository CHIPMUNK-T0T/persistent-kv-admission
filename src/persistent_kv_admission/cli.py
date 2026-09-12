"""Command-line entry point for reproducible trace characterization."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .characterize import (inter_arrival_rows, predictive_power_rows, replay_rows,
    reuse_mass_rows, state_feature_rows, trace_summary, write_csv)
from .reporting import make_plots, write_interpretation
from .trace import load_mooncake_trace

DEFAULT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/characterization"))
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--bytes-per-token", type=int, default=2048,
                        help="KV size proxy; does not affect token-hit rankings")
    parser.add_argument("--snapshots", type=int, default=24)
    parser.add_argument("--precision-k", type=int, default=100)
    parser.add_argument("--budgets", default=",".join(str(v) for v in DEFAULT_BUDGETS))
    return parser.parse_args()

def main():
    args = parse_args()
    budgets = tuple(float(v) for v in args.budgets.split(","))
    if any(v <= 0 or v > 1 for v in budgets):
        raise SystemExit("all budget fractions must be in (0, 1]")
    summaries = []
    for trace_path in args.traces:
        trace = load_mooncake_trace(trace_path, args.block_size)
        out = args.output_dir / trace.name
        out.mkdir(parents=True, exist_ok=True)
        summary = trace_summary(trace, args.bytes_per_token)
        inter, inter_summary = inter_arrival_rows(trace); summary.update(inter_summary)
        states = state_feature_rows(trace, args.bytes_per_token)
        reuse = reuse_mass_rows(trace)
        predictive = predictive_power_rows(trace, args.snapshots, args.precision_k)
        replayed = replay_rows(trace, budgets, args.bytes_per_token)
        write_csv(out / "trace_summary.csv", [summary])
        write_csv(out / "reuse_count_distribution.csv", reuse)
        write_csv(out / "inter_arrival_distribution.csv", inter)
        write_csv(out / "state_features.csv", states)
        write_csv(out / "predictive_power.csv", predictive)
        write_csv(out / "cache_replay.csv", replayed)
        make_plots(out, trace.name, reuse, states, predictive, replayed)
        write_interpretation(out, trace.name, summary, predictive, replayed)
        (out / "run_config.json").write_text(json.dumps({"trace": str(trace_path),
            "block_size": args.block_size, "bytes_per_token": args.bytes_per_token,
            "snapshots": args.snapshots, "precision_k": args.precision_k,
            "budget_fractions": budgets}, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(f"completed {trace.name}: {len(trace.requests)} requests, {len(trace.states)} states")
    write_csv(args.output_dir / "trace_summary.csv", summaries)

if __name__ == "__main__":
    main()
