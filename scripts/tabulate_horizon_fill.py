#!/usr/bin/env python3
"""Tabulate the published horizon fill-in tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_horizon_fill.py`
published in PAPER_DIR and prints every number a findings document would
quote, as Markdown tables on stdout, one section per pre-registered reading.
Nothing is written, fitted, rerun or re-derived from the replays: each section
is arithmetic on the published columns, so a number in the document can be
checked against this output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_horizon_fill.py`) and a small printer. Shares are
printed to 3 decimals, points to 3, counts of seeds as +/0/-. Traces are in
grid order, cells by (l1_fraction, l2_multiplier) ascending, horizons
ascending.

  F1 fill: S_h at eight horizons, the fill minimum, reading   fill_reading.csv, fill.csv
     and its count against the prediction                     fill_summary.csv
  F2 shape over 60-300 s: unimodal, minimisers, same per trace fill_reading.csv, fill_summary.csv
  F3 horizons from 60 to 300 s with S_h <= 0.10                fill_reading.csv

The count here is the fill-in's own ("n/4 of the remaining cells at a horizon
added afterwards"); it is never merged with the horizon control's count on its
registered grid.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --- the published grid and vocabularies (docs/horizon-fill-plan.md) ----------------------------

TRACES = ("conversation_trace", "toolagent_trace")
HORIZONS = (60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 600.0)
FILL_HORIZONS = (90.0, 120.0, 150.0, 180.0, 240.0)
FILL_READINGS = ("reuse_label_suffices_at_filled_horizon", "order_needed_stands")
PREDICTED_COLUMN = "predicted_reuse_label_suffices_at_filled_horizon"
INPUTS = ("fill", "fill_reading", "fill_summary")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published horizon fill-in directory (read only)")
    return parser.parse_args(argv)


# --- parsing and ordering -------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value) -> bool:
    """A boolean column as written by `csv` from a Python bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _float(value) -> float:
    return math.nan if value in ("", None) else float(value)


def _rank(preferred, value) -> tuple:
    return (preferred.index(value), "") if value in preferred else (len(preferred), value)


def cell_key(row) -> tuple:
    """(trace, l1_fraction, l2_multiplier), the fractions compared as floats."""
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def _row_order(row) -> tuple:
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "h" in row:
        order += (float(row["h"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


# --- F1: fill -------------------------------------------------------------------------------------


def fill_reading_table(rows) -> list[dict]:
    """F1: per trace x cell, S_h of the eight horizons on the five-seed means,
    the smallest over the five fill horizons, the fill horizons attaining it
    and the reading."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for h in HORIZONS:
            entry[f"S_{h:g}"] = _float(row[f"S_{h:g}"])
        entry.update(S_fill_min=_float(row["S_fill_min"]),
                     S_fill_min_horizons=row["S_fill_min_horizons"], reading=row["reading"])
        out.append(entry)
    return out


def fill_signs_table(rows) -> list[dict]:
    """F1, per h: `U(label) - U(label_binary_h)` on the five-seed means with
    its seed signs and reading, the horizon's role, and S_h."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h=float(row["h"]), role=row["role"],
                     arm_mean=float(row["U_arm_points_mean"]),
                     diff=float(row["label_minus_arm_points_mean"]),
                     signs=_signs(row, "label_minus_arm"),
                     reading=row["label_minus_arm_reading"], S=_float(row["S"]))
        out.append(entry)
    return out


def fill_counts(summary_rows) -> list[dict]:
    """F1 counts as published: per scope, the trace x cells of each reading,
    the predicted count and whether the prediction holds."""
    out = []
    for row in summary_rows:
        out.append({"scope": row["scope"], **{reading: int(row[reading])
                                              for reading in FILL_READINGS},
                    "predicted": int(row[PREDICTED_COLUMN]),
                    "holds": _bool(row["prediction_holds"]), "cells": int(row["cells"])})
    return out


# --- F2: shape ------------------------------------------------------------------------------------


def shape_table(rows) -> list[dict]:
    """F2: per trace x cell, unimodality over 60-300 s and the minimisers."""
    return [{**_identity(row), "unimodal": _bool(row["unimodal_60_300"]),
             "minimisers": row["curve_minimiser_horizons"]}
            for row in sorted(rows, key=_row_order)]


def shape_by_trace(summary_rows) -> list[dict]:
    """F2, per trace: the two cells' minimisers and whether they are the same.
    The published "|" between the cells is shown as " / ", which a Markdown
    table cell can hold."""
    return [{"trace": row["scope"], "cells": row["cell_labels"].replace("|", " / "),
             "minimisers": row["curve_minimiser_horizons"].replace("|", " / "),
             "same": _bool(row["same_curve_minimiser"])}
            for row in sorted((row for row in summary_rows if row["scope"] != "all"),
                              key=lambda row: _rank(TRACES, row["scope"]))]


# --- F3: sufficing set ----------------------------------------------------------------------------


def sufficing_table(rows) -> list[dict]:
    """F3: per trace x cell, the curve horizons with S_h <= 0.10, how many,
    and the width from the smallest to the largest (nan when none)."""
    return [{**_identity(row), "members": row["sufficing_horizons"],
             "count": int(row["sufficing_count"]),
             "width": _float(row["sufficing_width_seconds"])}
            for row in sorted(rows, key=_row_order)]


# --- printing -------------------------------------------------------------------------------------


def _number(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _seconds(value) -> str:
    return "empty" if math.isnan(value) else f"{value:g}"


def _sign_text(signs) -> str:
    return "/".join(str(n) for n in signs)


def _column(header, key, formatter=str):
    return (header, lambda row: formatter(row[key]))


def print_table(title: str, columns, rows, note: str = "") -> None:
    header = [name for name, _ in columns]
    body = [[formatter(row) for _, formatter in columns] for row in rows]
    widths = [max(len(text) for text in column) for column in zip(header, *body)]
    print(f"### {title}\n")
    if note:
        print(f"{note}\n")
    print("| " + " | ".join(text.ljust(width) for text, width in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for cells in body:
        print("| " + " | ".join(text.ljust(width) for text, width in zip(cells, widths)) + " |")
    print()


IDENTITY = [_column("trace", "trace"), _column("cell", "cell")]


def print_fill(reading_rows, per_h_rows, summary_rows) -> None:
    print_table("F1 fill: S_h of label_binary_h (five-seed means)",
                IDENTITY + [_column(f"S_{h:g}" + ("*" if h in FILL_HORIZONS else ""), f"S_{h:g}",
                                    _number) for h in HORIZONS]
                + [_column("S fill min", "S_fill_min", _number),
                   _column("at h", "S_fill_min_horizons"), _column("reading", "reading")],
                fill_reading_table(reading_rows),
                note="S_h = (U(label) - U(label_binary_h)) / (U(label) - U(lru)); * marks a fill "
                     "horizon. reuse_label_suffices_at_filled_horizon when the smallest S_h over "
                     "the five fill horizons is <= 0.10.")
    print_table("F1 U(label) - U(label_binary_h) per horizon (five-seed means, points)",
                IDENTITY + [_column("h", "h", lambda value: f"{value:g}"), _column("role", "role"),
                            _column("U(arm)", "arm_mean", _number),
                            _column("diff", "diff", _number), _column("+/0/-", "signs", _sign_text),
                            _column("reading", "reading"), _column("S_h", "S", _number)],
                fill_signs_table(per_h_rows))
    print_table("F1 reading counts against the prediction",
                [_column("scope", "scope")]
                + [_column(reading, reading) for reading in FILL_READINGS]
                + [_column("predicted", "predicted"), _column("holds", "holds"),
                   _column("of", "cells")], fill_counts(summary_rows),
                note="A second look at four trace x cell chosen because they failed the horizon "
                     "control; never merged with its count on the registered grid.")


def print_shape(reading_rows, summary_rows) -> None:
    print_table("F2 shape over 60-300 s (descriptive)",
                IDENTITY + [_column("unimodal", "unimodal"), _column("minimiser h", "minimisers")],
                shape_table(reading_rows))
    print_table("F2 same minimiser in the two cells of a trace (descriptive)",
                [_column("trace", "trace"), _column("cells", "cells"),
                 _column("minimiser h", "minimisers"), _column("same", "same")],
                shape_by_trace(summary_rows))


def print_sufficing(reading_rows) -> None:
    print_table("F3 horizons from 60 to 300 s with S_h <= 0.10 (descriptive)",
                IDENTITY + [_column("members", "members"), _column("count", "count"),
                            _column("width, s", "width", _seconds)],
                sufficing_table(reading_rows))


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Horizon fill-in tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_fill(tables["fill_reading"], tables["fill"], tables["fill_summary"])
    print_shape(tables["fill_reading"], tables["fill_summary"])
    print_sufficing(tables["fill_reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
