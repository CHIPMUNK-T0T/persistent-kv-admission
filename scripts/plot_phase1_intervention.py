#!/usr/bin/env python3
"""Plot the fixed Phase 1 arrival-protection intervention summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TRACES = ("conversation_trace", "toolagent_trace")
TARGETS = ("next_use", "binary")
CELLS = ((0.0025, 1.0), (0.01, 4.0), (0.02, 4.0))
PRIMARY_CELL = (0.02, 4.0)
VARIANTS = ("none", "direct_child", "all")
COMPARISONS = (
    ("direct_child-none", "direct child − original B", "#0072B2", "o"),
    ("all-none", "all arrivals − original B", "#D55E00", "s"),
    ("direct_child-all", "direct child − all arrivals", "#009E73", "^"),
)
PAIR_FILE = "phase1_intervention_pairs.csv"
REPLAY_FILE = "phase1_intervention_replay.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir",
        type=Path,
        help="directory containing the Phase 1 summary CSVs",
    )
    source.add_argument(
        "--paper-dir",
        type=Path,
        help="alias for --input-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper"),
        help="directory for fig19 and fig20 (default: results/paper)",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as error:
        raise SystemExit(f"missing Phase 1 input: {path}") from error
    if not rows:
        raise SystemExit(f"empty Phase 1 input: {path}")
    return rows


def require_columns(
    rows: list[dict[str, str]], required: set[str], path: Path
) -> None:
    present = set(rows[0])
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"{path} lacks required columns: {', '.join(missing)}")


def number(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid numeric value in {source}/{field}: {row.get(field)!r}") from error
    if not math.isfinite(value):
        raise SystemExit(f"non-finite value in {source}/{field}: {value!r}")
    return value


def integer(row: dict[str, str], field: str, source: str) -> int:
    value = number(row, field, source)
    if not value.is_integer():
        raise SystemExit(f"non-integer value in {source}/{field}: {value!r}")
    return int(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def cell_of(row: dict[str, str], source: str) -> tuple[float, float]:
    return number(row, "l1_fraction", source), number(row, "l2_multiplier", source)


def pair_key(
    row: dict[str, str], source: str
) -> tuple[str, tuple[float, float], str, str]:
    return row["trace"], cell_of(row, source), row["target"], row["comparison"]


def replay_key(
    row: dict[str, str], source: str
) -> tuple[str, tuple[float, float], str, str]:
    return row["trace"], cell_of(row, source), row["target"], row["variant"]


def index_unique(rows, key_function, source: str) -> dict:
    indexed = {}
    for row in rows:
        key = key_function(row, source)
        if key in indexed:
            raise SystemExit(f"duplicate row in {source}: {key}")
        indexed[key] = row
    return indexed


def load_and_validate(input_dir: Path) -> tuple[dict, dict]:
    pair_path = input_dir / PAIR_FILE
    replay_path = input_dir / REPLAY_FILE
    pairs = read_csv(pair_path)
    replay = read_csv(replay_path)
    require_columns(
        pairs,
        {
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "target",
            "left_variant",
            "right_variant",
            "comparison",
            "seeds",
            "requested_tokens",
            "left_avoided_tokens_mean",
            "right_avoided_tokens_mean",
            "saved_tokens_mean",
            "lost_tokens_mean",
            "net_avoided_tokens_mean",
            "net_input_percentage_points_mean",
            "net_input_percentage_points_ci95_half",
        },
        pair_path,
    )
    require_columns(
        replay,
        {
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "target",
            "variant",
            "seeds",
            "requested_tokens",
            "avoided_prefill_tokens_mean",
        },
        replay_path,
    )
    pair_index = index_unique(pairs, pair_key, PAIR_FILE)
    replay_index = index_unique(replay, replay_key, REPLAY_FILE)

    comparison_directions = {
        "direct_child-none": ("direct_child", "none"),
        "all-none": ("all", "none"),
        "direct_child-all": ("direct_child", "all"),
    }
    expected_pairs = {
        (trace, cell, target, comparison)
        for trace in TRACES
        for cell in CELLS
        for target in TARGETS
        for comparison in comparison_directions
    }
    expected_replay = {
        (trace, cell, target, variant)
        for trace in TRACES
        for cell in CELLS
        for target in TARGETS
        for variant in VARIANTS
    }
    for name, actual, expected in (
        (PAIR_FILE, set(pair_index), expected_pairs),
        (REPLAY_FILE, set(replay_index), expected_replay),
    ):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {len(missing)} rows (first: {missing[0]})")
            if unexpected:
                details.append(
                    f"unexpected {len(unexpected)} rows (first: {unexpected[0]})"
                )
            raise SystemExit(f"{name} is not the fixed Phase 1 grid: {'; '.join(details)}")

    for key, row in replay_index.items():
        if integer(row, "seeds", REPLAY_FILE) != 5:
            raise SystemExit(f"{REPLAY_FILE} does not contain five seeds at {key}")
        if number(row, "requested_tokens", REPLAY_FILE) <= 0:
            raise SystemExit(f"non-positive requested_tokens in {REPLAY_FILE} at {key}")

    for key, row in pair_index.items():
        trace, cell, target, comparison = key
        left_variant, right_variant = comparison_directions[comparison]
        if (row["left_variant"], row["right_variant"]) != (
            left_variant,
            right_variant,
        ):
            raise SystemExit(f"comparison direction mismatch in {PAIR_FILE} at {key}")
        if integer(row, "seeds", PAIR_FILE) != 5:
            raise SystemExit(f"{PAIR_FILE} does not contain five seeds at {key}")
        requested = number(row, "requested_tokens", PAIR_FILE)
        saved = number(row, "saved_tokens_mean", PAIR_FILE)
        lost = number(row, "lost_tokens_mean", PAIR_FILE)
        net = number(row, "net_avoided_tokens_mean", PAIR_FILE)
        net_points = number(row, "net_input_percentage_points_mean", PAIR_FILE)
        net_ci = number(row, "net_input_percentage_points_ci95_half", PAIR_FILE)
        if requested <= 0 or saved < 0 or lost < 0 or net_ci < 0:
            raise SystemExit(f"invalid paired totals in {PAIR_FILE} at {key}")
        if not close(net, saved - lost):
            raise SystemExit(f"saved-minus-lost identity failed in {PAIR_FILE} at {key}")
        if not close(net_points, 100.0 * net / requested):
            raise SystemExit(f"net input-share identity failed in {PAIR_FILE} at {key}")

        left = replay_index[(trace, cell, target, left_variant)]
        right = replay_index[(trace, cell, target, right_variant)]
        left_mean = number(left, "avoided_prefill_tokens_mean", REPLAY_FILE)
        right_mean = number(right, "avoided_prefill_tokens_mean", REPLAY_FILE)
        pair_left = number(row, "left_avoided_tokens_mean", PAIR_FILE)
        pair_right = number(row, "right_avoided_tokens_mean", PAIR_FILE)
        replay_requested = number(left, "requested_tokens", REPLAY_FILE)
        if not all(
            (
                close(pair_left, left_mean),
                close(pair_right, right_mean),
                close(net, left_mean - right_mean),
                close(requested, replay_requested),
                close(
                    requested,
                    number(right, "requested_tokens", REPLAY_FILE),
                ),
            )
        ):
            raise SystemExit(f"pair/replay consistency check failed at {key}")
    return pair_index, replay_index


def cell_labels() -> list[str]:
    return [f"{100.0 * fraction:g}% × {multiplier:g}" for fraction, multiplier in CELLS]


def panel_title(trace: str, target: str) -> str:
    return f"{trace.removesuffix('_trace')} — target: {target}"


def mark_primary(axis: plt.Axes, annotate: bool) -> None:
    position = CELLS.index(PRIMARY_CELL)
    axis.axvspan(position - 0.45, position + 0.45, color="#F0E442", alpha=0.13, zorder=0)
    if annotate:
        axis.text(
            position,
            0.98,
            "primary",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="0.25",
        )


def make_net_figure(output_dir: Path, pairs: dict) -> Path:
    positions = np.arange(len(CELLS), dtype=float)
    figure, axes = plt.subplots(
        len(TARGETS),
        len(TRACES),
        figsize=(10.8, 7.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, target in enumerate(TARGETS):
        for column, trace in enumerate(TRACES):
            axis = axes[row_index][column]
            mark_primary(axis, annotate=target == "next_use")
            for comparison, label, color, marker in COMPARISONS:
                selected = [pairs[(trace, cell, target, comparison)] for cell in CELLS]
                means = [
                    number(row, "net_input_percentage_points_mean", PAIR_FILE)
                    for row in selected
                ]
                intervals = [
                    number(row, "net_input_percentage_points_ci95_half", PAIR_FILE)
                    for row in selected
                ]
                axis.errorbar(
                    positions,
                    means,
                    yerr=intervals,
                    color=color,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.35,
                    capsize=2.5,
                    label=label,
                    zorder=3,
                )
            axis.axhline(0.0, color="0.25", linewidth=0.9, zorder=1)
            axis.set(
                xticks=positions,
                xticklabels=cell_labels(),
                title=panel_title(trace, target),
                ylabel="net avoided tokens\n(input percentage points)" if column == 0 else "",
            )
            axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#F0E442", alpha=0.25, edgecolor="none"))
    labels.append("primary cell: 2% × 4")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.supxlabel("L1 working-set fraction × L2/L1 multiplier", fontsize=9)
    figure.suptitle(
        "Phase 1 arrival protection: paired net avoided input",
        fontsize=11,
        y=0.99,
    )
    figure.text(
        0.5,
        0.95,
        "points are five-seed means; error bars are 95% t intervals over seeds",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.85))
    path = output_dir / "fig19_phase1_intervention.png"
    figure.savefig(path, dpi=170, metadata={"Software": "matplotlib"})
    plt.close(figure)
    return path


def make_saved_lost_figure(output_dir: Path, pairs: dict) -> Path:
    positions = np.arange(len(CELLS), dtype=float)
    offsets = (-0.24, 0.0, 0.24)
    width = 0.20
    figure, axes = plt.subplots(
        len(TARGETS),
        len(TRACES),
        figsize=(10.8, 7.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, target in enumerate(TARGETS):
        for column, trace in enumerate(TRACES):
            axis = axes[row_index][column]
            mark_primary(axis, annotate=target == "next_use")
            for comparison_index, (comparison, _label, color, _marker) in enumerate(
                COMPARISONS
            ):
                selected = [pairs[(trace, cell, target, comparison)] for cell in CELLS]
                requested = np.array(
                    [number(row, "requested_tokens", PAIR_FILE) for row in selected]
                )
                saved = 100.0 * np.array(
                    [number(row, "saved_tokens_mean", PAIR_FILE) for row in selected]
                ) / requested
                lost = -100.0 * np.array(
                    [number(row, "lost_tokens_mean", PAIR_FILE) for row in selected]
                ) / requested
                net = np.array(
                    [
                        number(row, "net_input_percentage_points_mean", PAIR_FILE)
                        for row in selected
                    ]
                )
                intervals = np.array(
                    [
                        number(row, "net_input_percentage_points_ci95_half", PAIR_FILE)
                        for row in selected
                    ]
                )
                x = positions + offsets[comparison_index]
                axis.bar(x, saved, width=width, color=color, alpha=0.82, zorder=2)
                axis.bar(
                    x,
                    lost,
                    width=width,
                    color=color,
                    alpha=0.28,
                    hatch="///",
                    zorder=2,
                )
                axis.errorbar(
                    x,
                    net,
                    yerr=intervals,
                    linestyle="none",
                    marker="D",
                    markersize=3.8,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2,
                    zorder=4,
                )
            axis.axhline(0.0, color="0.2", linewidth=0.9, zorder=1)
            axis.set(
                xticks=positions,
                xticklabels=cell_labels(),
                title=panel_title(trace, target),
                ylabel="tokens, share of input (%)\n(+ saved; − lost)" if column == 0 else "",
            )
            axis.grid(True, axis="y", alpha=0.25)

    comparison_handles = [
        Patch(facecolor=color, label=label) for _name, label, color, _marker in COMPARISONS
    ]
    outcome_handles = [
        Patch(facecolor="0.35", alpha=0.82, label="saved tokens"),
        Patch(facecolor="0.35", alpha=0.28, hatch="///", label="lost tokens"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="D",
            markerfacecolor="white",
            linestyle="none",
            markersize=4,
            label="net (95% seed interval)",
        ),
    ]
    figure.legend(
        comparison_handles + outcome_handles,
        [handle.get_label() for handle in comparison_handles + outcome_handles],
        loc="upper center",
        ncol=3,
        fontsize=7.3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.supxlabel("L1 working-set fraction × L2/L1 multiplier", fontsize=9)
    figure.suptitle(
        "Phase 1 request outcomes: saved tokens, lost tokens, and net",
        fontsize=11,
        y=0.99,
    )
    figure.text(
        0.5,
        0.95,
        "bars and diamonds are five-seed means; diamond intervals show seed variability",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.81))
    path = output_dir / "fig20_phase1_saved_lost.png"
    figure.savefig(path, dpi=170, metadata={"Software": "matplotlib"})
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or args.paper_dir or Path("results/paper")
    pairs, _replay = load_and_validate(input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        make_net_figure(args.output_dir, pairs),
        make_saved_lost_figure(args.output_dir, pairs),
    )
    for path in outputs:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
