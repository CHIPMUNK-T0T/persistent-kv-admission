#!/usr/bin/env python3
"""Tabulate the published class-order mix tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_class_order_mix.py`
published in PAPER_DIR and prints every number a findings document would
quote, as Markdown tables on stdout, one section per pre-registered reading
plus the replay counters and the level of every arm. Nothing is written,
fitted, rerun or re-derived from the replays: each section is arithmetic on
the published columns (stdlib `csv` only), so a number in the document can be
checked against this output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_class_order_mix.py`) and a small printer. Points and
shares are printed to 3 decimals, counts of seeds as +/0/-. Traces are in
grid order, cells by (l1_fraction, l2_multiplier) ascending, arms as
(mix_out_learned, mix_in_learned, random_within_class).

  X0 replay counters                                  replay.csv
  X1 which class: D_in, D_out, the two conditions,    differences.csv, readings.csv
     additivity against D_learned (prediction 1)
  X2 recency beyond the bit: D_rand and               differences.csv, readings.csv
     random - learned (prediction 2)
  X3 every arm: five-seed U and R_h*                  differences.csv

D_* are seed-paired differences against the published matched recency arm
`evict_binary_{h*}_recency`, in points; D_learned is the published matched
learned arm's.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --- the published grid and vocabularies (docs/class-order-mix-plan.md) ---------------------------

TRACES = ("conversation_trace", "toolagent_trace")
KINDS = ("mix_out_learned", "mix_in_learned", "random_within_class")
SIGN_READINGS = ("consistent_gain", "consistent_loss", "mixed")
READING_COLUMNS = ("D_in_negative", "D_out_above_D_in", "both", "additive")
# U of every arm in `differences.csv`, in table order, and those with an R_h*.
U_NAMES = ("learned", "evict_label", "label_binary_hstar", "matched_learned",
           "matched_recency") + KINDS
R_NAMES = U_NAMES[2:]
INPUTS = ("replay", "differences", "readings")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published class-order mix directory (read only)")
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


def _int_or_blank(value):
    return "" if value in ("", None) else int(value)


def _rank(preferred, value) -> tuple:
    return (preferred.index(value), "") if value in preferred else (len(preferred), value)


def cell_key(row) -> tuple:
    """(trace, l1_fraction, l2_multiplier), the fractions compared as floats."""
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def _row_order(row) -> tuple:
    """Trace x cell, then arm kind and seed where a table has them."""
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "kind" in row:
        order += (_rank(KINDS, row["kind"]),)
    if "seed" in row:
        order += (int(row["seed"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"]),
            "h_star": float(row["h_star"] if "h_star" in row else row["arm_parameter"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


def _difference(row, name: str) -> dict:
    return {name: float(row[f"{name}_points_mean"]),
            f"{name}_ci95": _float(row[f"{name}_points_ci95_half"]),
            f"{name}_signs": _signs(row, name), f"{name}_reading": row[f"{name}_reading"]}


def _readings(summary_rows, prefixes) -> list[dict]:
    return [row for row in summary_rows if row["reading"].startswith(prefixes)]


# --- X0: replay counters --------------------------------------------------------------------------

REPLAY_COLUMNS = ("extra_points_mean", "extra_points_ci95_half", "l2_rejections_mean",
                  "l2_evictions_mean", "overridden_decisions_mean",
                  "class_violations_seen_mean", "class_overridden_outside_admission_seen_mean",
                  "random_draws_mean")


def replay_counters(rows) -> list[dict]:
    """X0: five-seed means per trace x cell x arm, with h* and kind."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], kind=row["kind"], seeds=int(row["seeds"]))
        for column in REPLAY_COLUMNS:
            entry[column] = _float(row.get(column, ""))
        out.append(entry)
    return out


# --- X1: which class ------------------------------------------------------------------------------


def which_class_table(rows) -> list[dict]:
    """X1: per trace x cell, D_in and D_out with their seed signs and readings,
    the two conditions and their conjunction on the means, and D_in + D_out
    beside the seed interval of the published D_learned."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry["named"] = _bool(row["prediction_1_cell"])
        for name in ("D_in", "D_out", "D_learned"):
            entry.update(_difference(row, name))
        entry.update(D_in_negative=_bool(row["D_in_negative"]),
                     D_out_above_D_in=_bool(row["D_out_above_D_in"]),
                     condition=_bool(row["which_class_condition"]),
                     sum=float(row["D_in_plus_D_out_points"]),
                     low=_float(row["D_learned_interval_low"]),
                     high=_float(row["D_learned_interval_high"]),
                     additive=_bool(row["additive"]))
        out.append(entry)
    return out


def reading_counts(summary_rows, prefixes) -> list[dict]:
    """The published counts of the readings whose name starts with one of
    `prefixes`, with the prediction and whether it holds where one is set."""
    out = []
    for row in _readings(summary_rows, prefixes):
        entry = {"reading": row["reading"], "scope": row["scope"], "cells": int(row["cells"])}
        for column in READING_COLUMNS + SIGN_READINGS:
            entry[column] = _int_or_blank(row[column])
        entry["prediction"] = row["prediction"]
        entry["holds"] = _bool(row["prediction_holds"]) if row["prediction_holds"] != "" else ""
        out.append(entry)
    return out


# --- X2: recency beyond the bit -------------------------------------------------------------------


def recency_table(rows) -> list[dict]:
    """X2: per trace x cell, D_rand (random within class minus recency) and
    random minus the learned order, seed-paired, with seed signs and readings,
    and D_learned beside them."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for name in ("D_rand", "random_minus_learned", "D_learned"):
            entry.update(_difference(row, name))
        out.append(entry)
    return out


# --- X3: every arm --------------------------------------------------------------------------------


def arms_table(rows) -> list[dict]:
    """X3: per trace x cell, the five-seed mean U of every arm (published and
    replayed) and R_h* of every arm but learned and evict_label."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for name in U_NAMES:
            entry[f"U_{name}"] = float(row[f"U_{name}_points_mean"])
        for name in R_NAMES:
            entry[f"R_{name}"] = _float(row[f"R_{name}"])
        out.append(entry)
    return out


# --- printing -------------------------------------------------------------------------------------


def _number(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _count_mean(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.1f}"


def _sign_text(signs) -> str:
    return "/".join(str(n) for n in signs)


def _seconds(value) -> str:
    return f"{value:g}"


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


IDENTITY = [_column("trace", "trace"), _column("cell", "cell"), _column("h*", "h_star", _seconds)]


def _difference_columns(name: str, label: str) -> list:
    return [_column(label, name, _number), _column("ci95", f"{name}_ci95", _number),
            _column("+/0/-", f"{name}_signs", _sign_text),
            _column("reading", f"{name}_reading")]


def print_replay(rows) -> None:
    formats = {column: _count_mean for column in REPLAY_COLUMNS
               if not column.startswith("extra_points")}
    print_table("X0 replay counters (five-seed means; U in points)",
                IDENTITY + [_column("arm", "arm"), _column("seeds", "seeds")]
                + [_column(column.removesuffix("_mean"), column, formats.get(column, _number))
                   for column in REPLAY_COLUMNS], replay_counters(rows))


def _print_counts(title: str, summary_rows, prefixes) -> None:
    print_table(title,
                [_column("reading", "reading"), _column("scope", "scope"),
                 _column("of", "cells")]
                + [_column(column, column) for column in READING_COLUMNS + SIGN_READINGS]
                + [_column("prediction", "prediction"), _column("holds", "holds")],
                reading_counts(summary_rows, prefixes))


def print_which_class(rows, summary_rows) -> None:
    print_table("X1 which class: D_in, D_out against the matched recency arm (points)",
                IDENTITY + [_column("named", "named")]
                + _difference_columns("D_in", "D_in") + _difference_columns("D_out", "D_out")
                + [_column("D_in < 0", "D_in_negative"),
                   _column("D_out > D_in", "D_out_above_D_in"),
                   _column("both", "condition"),
                   _column("D_learned", "D_learned", _number),
                   _column("D_in + D_out", "sum", _number),
                   _column("D_learned low", "low", _number),
                   _column("D_learned high", "high", _number),
                   _column("additive", "additive")],
                which_class_table(rows),
                note="named: one of the eight trace x cell of prediction 1. The conditions and "
                     "additivity are on five-seed means; the interval is D_learned's mean -/+ "
                     "its 95% t half-width.")
    _print_counts("X1 counts (prediction 1: both conditions in each named trace x cell)",
                  summary_rows, ("1_", "context_"))


def print_recency(rows, summary_rows) -> None:
    print_table("X2 recency beyond the bit: D_rand and random - learned (points)",
                IDENTITY + _difference_columns("D_rand", "D_rand")
                + _difference_columns("random_minus_learned", "random - learned")
                + [_column("D_learned", "D_learned", _number)],
                recency_table(rows),
                note="D_rand = U(random_within_class) - U(evict_binary_h*_recency); random - "
                     "learned = U(random_within_class) - U(evict_binary_h*_learned), "
                     "descriptive.")
    _print_counts("X2 counts (prediction 2: D_rand consistent_loss in at least 8 of 12)",
                  summary_rows, ("2_",))


def print_arms(rows) -> None:
    print_table("X3 every arm: five-seed U (points) and R_h*",
                IDENTITY + [_column(f"U({name})", f"U_{name}", _number) for name in U_NAMES]
                + [_column(f"R({name})", f"R_{name}", _number) for name in R_NAMES],
                arms_table(rows),
                note="R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned)) on the "
                     "five-seed means.")


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Class-order mix tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_replay(tables["replay"])
    print_which_class(tables["differences"], tables["readings"])
    print_recency(tables["differences"], tables["readings"])
    print_arms(tables["differences"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
