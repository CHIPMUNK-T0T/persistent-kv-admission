#!/usr/bin/env python3
"""Tabulate the published evaluation-window check tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_tail_window_check.py`
published in PAPER_DIR and prints every number a findings document would
quote, as Markdown tables on stdout: the reproduction check, the windows, one
section per pre-registered reading on the full, head and tail windows, and
the reading counts. Nothing is written, fitted, rerun or re-derived from the
replays: each section is arithmetic on the published columns (stdlib `csv`
only), so a number in the document can be checked against this output and
this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_tail_window_check.py`) and a small printer. Points and
shares are printed to 3 decimals, seed signs as +/0/-. Traces are in grid
order, cells by (l1_fraction, l2_multiplier) ascending, arms in the plan's
order.

  W0 reproduction and windows                  reproduction.csv, windows.csv
  W1 horizon: S_h* per window                  horizon.csv, readings.csv
  W2 class order: R_h* per window              class_order.csv, readings.csv
  W3 order within the matched class            order.csv, readings.csv
  W4 admission at the matched horizon          admission.csv, readings.csv
  W5 tail share of each difference             tail_share.csv

The predictions are read on the head window; full and tail are printed beside
it, never merged with it.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --- the published grid and vocabularies (docs/tail-window-check-plan.md) -------------------------

TRACES = ("conversation_trace", "toolagent_trace")
WINDOWS = ("full", "head", "tail")
ARM_ROLES = ("lru", "learned", "label", "evict_label", "label_binary_h*",
             "evict_binary_h*_learned", "evict_binary_h*_recency")
SOURCES = ("error_location_001", "horizon_control_001", "horizon_fill_001",
           "matched_class_order_001")
DIFFERENCES = ("G", "label_minus_label_binary", "label_minus_lru",
               "matched_learned_minus_learned", "matched_recency_minus_learned",
               "evict_label_minus_learned", "learned_minus_recency",
               "label_binary_minus_recency")
SIGN_READINGS = ("consistent_gain", "consistent_loss", "mixed")
INPUTS = ("reproduction", "windows", "horizon", "class_order", "order", "admission",
          "tail_share", "readings")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published evaluation-window check directory (read only)")
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
    """Trace x cell, then arm role or difference where a table has one."""
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "arm_role" in row:
        order += (_rank(ARM_ROLES, row["arm_role"]),)
    if "difference" in row:
        order += (_rank(DIFFERENCES, row["difference"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"]), "h_star": float(row["h_star"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


# --- W0: reproduction and windows -----------------------------------------------------------------


def reproduction_counts(rows) -> list[dict]:
    """W0: per published source and arm role, the replays compared and how
    many equal the published row in tokens, counter digest, decision digest
    and all three."""
    out = []
    keys = sorted({(row["reference_source"], row["arm_role"]) for row in rows},
                  key=lambda key: (_rank(SOURCES, key[0]), _rank(ARM_ROLES, key[1])))
    for source, role in keys:
        members = [row for row in rows
                   if row["reference_source"] == source and row["arm_role"] == role]
        out.append({"reference_source": source, "arm_role": role, "replays": len(members),
                    **{f"same_{column}": sum(1 for row in members if _bool(row[f"same_{column}"]))
                       for column in ("avoided_prefill_tokens", "counters_sha256",
                                      "decision_sha256")},
                    "reproduces": sum(1 for row in members if _bool(row["reproduces"]))})
    return out


def window_sizes(rows) -> list[dict]:
    """W0: per trace x cell, the requests and input tokens of each window (the
    first arm's row; they are arm-independent) and the tail's share of input."""
    out, seen = [], set()
    for row in sorted(rows, key=_row_order):
        if cell_key(row) in seen:
            continue
        seen.add(cell_key(row))
        entry = _identity(row)
        for window in WINDOWS:
            entry[f"{window}_requests"] = int(row[f"{window}_measured_requests"])
            entry[f"{window}_tokens"] = int(row[f"{window}_requested_tokens"])
        entry["tail_input_share"] = (entry["tail_tokens"] / entry["full_tokens"]
                                     if entry["full_tokens"] else math.nan)
        out.append(entry)
    return out


def utility_table(rows) -> list[dict]:
    """W0: per trace x cell x arm, five-seed mean U on each window."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], arm_role=row["arm_role"],
                     **{window: _float(row[f"U_{window}_points_mean"]) for window in WINDOWS})
        out.append(entry)
    return out


# --- W1-W4: the readings per window ---------------------------------------------------------------


def horizon_table(rows) -> list[dict]:
    """W1: per trace x cell, S_h* on each window, the head reading and its
    agreement with full, and the published full-window S."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(rung=row["rung_arm"], **{window: _float(row[f"S_{window}"])
                                              for window in WINDOWS},
                     reading_head=row["reading_head"],
                     head_agrees=_bool(row["reading_head_agrees_with_full"]),
                     tail_agrees=_bool(row["reading_tail_agrees_with_full"]),
                     published=_float(row["S_full_published"]),
                     equal=_bool(row["full_equals_published"]))
        out.append(entry)
    return out


def class_order_table(rows) -> list[dict]:
    """W2: per trace x cell, R_h* of both class-order arms on each window, the
    head reading against the predicted one, and agreement with full."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for window in WINDOWS:
            entry[f"learned_{window}"] = _float(row[f"R_matched_learned_{window}"])
            entry[f"recency_{window}"] = _float(row[f"R_matched_recency_{window}"])
        entry.update(reading_full=row["reading_full"], reading_head=row["reading_head"],
                     predicted=row["predicted_reading_head"],
                     head_agrees=_bool(row["reading_head_agrees_with_full"]),
                     tail_agrees=_bool(row["reading_tail_agrees_with_full"]),
                     equal=_bool(row["full_equals_published"]))
        out.append(entry)
    return out


def difference_table(rows, prefix: str) -> list[dict]:
    """W3 / W4: per trace x cell, the five-seed mean of a seed-paired
    difference on each window with its seed signs and reading, and the
    agreement of the head and tail signs of the mean with full."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for window in WINDOWS:
            entry[f"{window}_mean"] = _float(row[f"{prefix}_{window}_points_mean"])
            entry[f"{window}_signs"] = _signs(row, f"{prefix}_{window}")
            entry[f"{window}_reading"] = row[f"{prefix}_{window}_reading"]
        for window in ("head", "tail"):
            entry[f"{window}_agrees"] = _bool(row[f"{prefix}_{window}_mean_sign_agrees_with_full"])
        entry["equal"] = _bool(row["full_equals_published"])
        out.append(entry)
    return out


def tail_share_table(rows) -> list[dict]:
    """W5: per trace x cell x difference, the full and tail token differences
    (five seeds summed), the tail's share of the difference and of input."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(difference=row["difference"], reading=row["reading"],
                     full_tokens=int(row["full_tokens"]), tail_tokens=int(row["tail_tokens"]),
                     share=_float(row["tail_share_of_difference"]),
                     input_share=_float(row["tail_share_of_input"]))
        out.append(entry)
    return out


def reading_counts(summary_rows) -> list[dict]:
    """The counts as published, one row per reading x window."""
    out = []
    for row in summary_rows:
        entry = {"reading": row["reading"], "window": row["window"], "kind": row["kind"],
                 "cells": int(row["cells"])}
        for column in ("count", "agrees_with_full", "predicted"):
            entry[column] = int(row[column]) if row[column] != "" else ""
        entry["holds"] = _bool(row["prediction_holds"]) if row["prediction_holds"] != "" else ""
        for label in SIGN_READINGS:
            entry[label] = int(row[label]) if row[label] != "" else ""
        out.append(entry)
    return out


# --- printing -------------------------------------------------------------------------------------


def _number(value) -> str:
    return "nan" if isinstance(value, float) and math.isnan(value) else f"{value:.3f}"


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


def print_reproduction(reproduction_rows, window_rows) -> None:
    print_table("W0 reproduction of the published rows",
                [_column("source", "reference_source"), _column("arm", "arm_role"),
                 _column("replays", "replays"),
                 _column("same tokens", "same_avoided_prefill_tokens"),
                 _column("same counters", "same_counters_sha256"),
                 _column("same decisions", "same_decision_sha256"),
                 _column("reproduces", "reproduces")], reproduction_counts(reproduction_rows))
    print_table("W0 windows (requests and input tokens; arm-independent)",
                IDENTITY + [_column(f"{window} {kind}", f"{window}_{kind}")
                            for window in WINDOWS for kind in ("requests", "tokens")]
                + [_column("tail share of input", "tail_input_share", _number)],
                window_sizes(window_rows))
    print_table("W0 U per window (five-seed means, points of the window's input tokens)",
                IDENTITY + [_column("arm", "arm")]
                + [_column(window, window, _number) for window in WINDOWS],
                utility_table(window_rows))


def print_readings(tables) -> None:
    print_table("W1 horizon: S_h* = (U(label) - U(label_binary_h*)) / (U(label) - U(lru))",
                IDENTITY + [_column("rung", "rung")]
                + [_column(f"S {window}", window, _number) for window in WINDOWS]
                + [_column("head reading", "reading_head"),
                   _column("head = full", "head_agrees"), _column("tail = full", "tail_agrees"),
                   _column("S full, published", "published", _number),
                   _column("full = published", "equal")], horizon_table(tables["horizon"]),
                note="reuse_label_suffices when S_h* <= 0.10. Prediction 1: 12/12 on head.")
    print_table("W2 class order: R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))",
                IDENTITY + [_column(f"R learned order {window}", f"learned_{window}", _number)
                            for window in WINDOWS]
                + [_column(f"R recency {window}", f"recency_{window}", _number)
                   for window in WINDOWS]
                + [_column("full reading", "reading_full"), _column("head reading", "reading_head"),
                   _column("predicted (head)", "predicted"), _column("head = full", "head_agrees"),
                   _column("tail = full", "tail_agrees"), _column("full = published", "equal")],
                class_order_table(tables["class_order"]),
                note="suffices when R_h*(learned order) >= 0.9. Prediction 2: the full reading "
                     "on head in every trace x cell, below 0.9 at 0.25% x 1 on both traces.")
    for title, name, prefix in (
            ("W3 order within the matched class: U(learned order) - U(recency), points",
             "order", "learned_minus_recency"),
            ("W4 admission at the matched horizon: U(label_binary_h*) - U(recency arm), points",
             "admission", "label_binary_minus_recency")):
        columns = list(IDENTITY)
        for window in WINDOWS:
            columns += [_column(f"{window} mean", f"{window}_mean", _number),
                        _column(f"{window} +/0/-", f"{window}_signs", _sign_text),
                        _column(f"{window} reading", f"{window}_reading")]
        columns += [_column("head sign = full", "head_agrees"),
                    _column("tail sign = full", "tail_agrees"),
                    _column("full = published", "equal")]
        print_table(title, columns, difference_table(tables[name], prefix),
                    note="Predictions 3 and 4: the sign of the head mean is the full window's "
                         "in 12/12 (zero matches zero only).")
    print_table("W5 tail share of each full-window token difference (five seeds summed)",
                IDENTITY + [_column("difference", "difference"), _column("reading", "reading"),
                            _column("full tokens", "full_tokens"),
                            _column("tail tokens", "tail_tokens"),
                            _column("tail share", "share", _number),
                            _column("tail share of input", "input_share", _number)],
                tail_share_table(tables["tail_share"]), note="Descriptive.")
    print_table("Reading counts",
                [_column(column, column) for column in
                 ("reading", "window", "kind", "count", "cells", "agrees_with_full", "predicted",
                  "holds") + SIGN_READINGS], reading_counts(tables["readings"]))


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Evaluation-window check tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_reproduction(tables["reproduction"], tables["windows"])
    print_readings(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
