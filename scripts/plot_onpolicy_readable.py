"""Render a spaced version of the published next-use cross-score heatmap.

This is report-only postprocessing: it reads the frozen cross-score CSV and
does not run a replay, fit a model, or change the original paper figure.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


DEFAULT_CSV = Path("results/paper/onpolicy_learning/onpolicy_cross_score_matrix.csv")
DEFAULT_OUTPUT = Path("results/paper/onpolicy_learning/onpolicy_cross_score_next_use_readable.png")
TRACES = ("conversation_trace", "toolagent_trace")
WINDOWS = ("train", "test")
CELLS = (
    "l1=0.0025,l2x1", "l1=0.0025,l2x4", "l1=0.01,l2x1",
    "l1=0.01,l2x4", "l1=0.02,l2x1", "l1=0.02,l2x4",
)
SEEDS = frozenset(range(5))


def matrix_means(path: Path) -> dict[tuple[str, str, str, int, int], float]:
    values: dict[tuple[str, str, str, int, int], dict[int, float]] = defaultdict(dict)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["target"] != "next_use":
                continue
            key = (
                row["trace"], row["cell"], row["window"],
                int(row["population_iteration"]), int(row["scoring_iteration"]),
            )
            seed = int(row["seed"])
            if seed in values[key]:
                raise ValueError(f"duplicate seed for {key}: {seed}")
            score = float(row["within_decision_macro"])
            if not math.isfinite(score):
                raise ValueError(f"nonfinite score for {key}, seed {seed}")
            values[key][seed] = score
    expected = {
        (trace, cell, window, population, scorer)
        for trace in TRACES for cell in CELLS for window in WINDOWS
        for population in range(4) for scorer in range(4)
    }
    if set(values) != expected:
        raise ValueError("cross-score CSV does not contain the fixed next-use grid")
    if any(set(seed_values) != SEEDS for seed_values in values.values()):
        raise ValueError("each cross-score entry must contain seeds 0..4")
    return {key: fmean(seed_values.values()) for key, seed_values in values.items()}


def render(csv_path: Path, output_path: Path) -> None:
    means = matrix_means(csv_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lower, upper = min(means.values()), max(means.values())
    fig, axes = plt.subplots(4, 6, figsize=(25, 17), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.89, top=0.92, bottom=0.065,
                        wspace=0.12, hspace=0.24)
    for column, cell in enumerate(CELLS):
        axes[0, column].set_title(cell.replace("l1=", "L1 ").replace(",l2x", "%, L2 ×")
                                  .replace("0.0025%", "0.25%")
                                  .replace("0.01%", "1%")
                                  .replace("0.02%", "2%"), fontsize=13, pad=15)

    for row_number, (trace, window) in enumerate(
        (trace, window) for trace in TRACES for window in WINDOWS
    ):
        row_label = f"{trace.replace('_trace', '')} / {window}"
        axes[row_number, 0].set_ylabel(f"{row_label}\nlogged population", fontsize=12,
                                       labelpad=12)
        for column, cell in enumerate(CELLS):
            axis = axes[row_number, column]
            matrix = np.array([
                [means[(trace, cell, window, population, scorer)]
                 for scorer in range(4)]
                for population in range(4)
            ])
            image = axis.imshow(matrix, vmin=lower, vmax=upper, cmap="viridis",
                                aspect="auto")
            axis.set_xticks(range(4), [f"π{i}" for i in range(4)], fontsize=10)
            axis.set_yticks(range(4), [f"D(π{i})" for i in range(4)], fontsize=10)
            axis.tick_params(axis="x", labelbottom=True, pad=5)
            axis.tick_params(axis="y", labelleft=(column == 0), pad=5)
            for population in range(4):
                for scorer in range(4):
                    score = matrix[population, scorer]
                    axis.text(scorer, population, f"{score:.2f}", ha="center",
                              va="center", fontsize=10,
                              color="white" if score < (lower + upper) / 2 else "black")
        axes[row_number, 2].set_xlabel("scoring model", fontsize=11, labelpad=8)
    colorbar_axis = fig.add_axes((0.915, 0.23, 0.018, 0.54))
    fig.colorbar(image, cax=colorbar_axis, label="within-decision Spearman")
    fig.suptitle("Next-use cross-scoring on each saved decision population\n"
                 "Five-seed mean; rows are logged populations and columns are scoring models",
                 fontsize=17, y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.csv, args.output)
