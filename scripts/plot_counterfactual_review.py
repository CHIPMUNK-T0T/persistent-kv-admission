#!/usr/bin/env python3
"""Review plots from frozen counterfactual CSVs; no replay or refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACES = ("conversation_trace", "toolagent_trace")
CELLS = (("0.0025", "1"), ("0.02", "4"))
POLICIES = ("pi0", "pi3")
SELECTORS = ("learned", "next_use", "count", "lru", "lfu")
COLORS = {"pi0": "#4169a8", "pi3": "#ca694b"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _panel(axis, trace: str, fraction: str, multiplier: str) -> None:
    axis.set_title(f"{trace.replace('_trace', '')}: L1 {fraction}, L2/L1 {multiplier}")
    axis.set_ylabel("Mean candidate regret (tokens)")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def selector_figure(rows: list[dict], window: str, target: Path) -> None:
    values = {
        (r["trace"], r["l1_fraction"], r["l2_multiplier"], r["policy"],
         r["window"], r["selector"]): float(r["mean_of_seed_mean_regret_tokens"])
        for r in rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for i, trace in enumerate(TRACES):
        for j, (fraction, multiplier) in enumerate(CELLS):
            axis = axes[i, j]
            _panel(axis, trace, fraction, multiplier)
            for policy, offset in (("pi0", -0.18), ("pi3", 0.18)):
                heights = [
                    values[(trace, fraction, multiplier, policy, window, selector)]
                    for selector in SELECTORS
                ]
                axis.bar(
                    np.arange(len(SELECTORS)) + offset, heights, width=0.36,
                    label=policy, color=COLORS[policy]
                )
            axis.set_xticks(range(len(SELECTORS)), SELECTORS, rotation=25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "One forced victim; frozen policy continuation; "
        + ("future 600 s" if window == "600" else "through trace end"),
        y=0.985,
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2)
    figure.text(
        0.5, 0.015,
        "pi0 and pi3 use different sampled states; compare selectors within each policy.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.88))
    figure.savefig(target, dpi=160)
    plt.close(figure)


def tie_figure(rows: list[dict], target: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["window"] == "600" and row["replicate"] == "0":
            groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"],
                    row["policy"], row["selector"])].append(row)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    labels = ("Learned", "Exact label, fixed tie", "Best within label tie\n(hindsight)")
    colors = ("#4169a8", "#ca694b", "#6b9270")
    for i, trace in enumerate(TRACES):
        for j, (fraction, multiplier) in enumerate(CELLS):
            axis = axes[i, j]
            _panel(axis, trace, fraction, multiplier)
            for policy_index, policy in enumerate(POLICIES):
                common = (trace, fraction, multiplier, policy)
                learned = groups[(*common, "learned")]
                exact = groups[(*common, "next_use")]
                if len(learned) != 40 or len(exact) != 40:
                    raise AssertionError(f"unexpected decision count for {common}")
                heights = (
                    sum(int(r["regret_tokens"]) for r in learned) / 40,
                    sum(int(r["regret_tokens"]) for r in exact) / 40,
                    sum(int(r["label_tie_best_regret_tokens"]) for r in exact) / 40,
                )
                for selector_index, (height, color) in enumerate(zip(heights, colors)):
                    axis.bar(
                        policy_index + (selector_index - 1) * 0.23, height,
                        width=0.22, color=color,
                        hatch="///" if selector_index == 2 else None,
                        label=labels[selector_index] if i == j == policy_index == 0 else None,
                    )
            axis.set_xticks(range(len(POLICIES)), POLICIES)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Future 600 s: fixed selectors and exact-label tie envelope", y=0.985)
    figure.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3
    )
    figure.text(
        0.5, 0.015,
        "The green bar uses future Q to choose within a label tie; it is a diagnostic bound, not a deployable selector.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.86))
    figure.savefig(target, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", required=True, type=Path)
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    config_path = paper / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = ("stratified_regrets.csv", "decision_regrets.csv")
    for name in inputs:
        if sha256(paper / name) != config["files_sha256"][name]:
            raise AssertionError(f"frozen input hash mismatch: {name}")
    strata = csv_rows(paper / inputs[0])
    decisions = csv_rows(paper / inputs[1])
    outputs = (
        "primary_600s_regret_reviewed.png",
        "secondary_trace_end_regret_reviewed.png",
        "primary_600s_label_tie_reviewed.png",
    )
    if any((paper / name).exists() for name in (*outputs, "plot_review_provenance.json")):
        raise FileExistsError("review plot outputs already exist")
    selector_figure(strata, "600", paper / outputs[0])
    selector_figure(strata, "end", paper / outputs[1])
    tie_figure(decisions, paper / outputs[2])
    provenance = {
        "purpose": "layout correction and preregistered label-tie diagnostic from frozen CSVs",
        "source_manifest_sha256": config["manifest_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {"run_config.json": sha256(config_path), **{
            name: sha256(paper / name) for name in inputs
        }},
        "output_sha256": {name: sha256(paper / name) for name in outputs},
        "tie_plot_window": "(t,t+600 seconds]",
        "tie_plot_replicate": 0,
        "tie_plot_decisions_per_stratum": 40,
        "tie_best_is_hindsight": True,
    }
    (paper / "plot_review_provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outputs": outputs, "provenance": "plot_review_provenance.json"}))


if __name__ == "__main__":
    main()
