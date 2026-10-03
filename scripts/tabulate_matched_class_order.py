#!/usr/bin/env python3
"""Tabulate the published matched class-order tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_matched_class_order.py`
published in PAPER_DIR and prints every number a findings document would
quote, as Markdown tables on stdout, one section per pre-registered reading
plus the reproduction check and the replay counters. Nothing is written,
fitted, rerun or re-derived from the replays: each section is arithmetic on
the published columns (stdlib `csv` only), so a number in the document can be
checked against this output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_matched_class_order.py`) and a small printer. Points
and shares are printed to 3 decimals, counts of seeds as +/0/-. Traces are in
grid order, cells by (l1_fraction, l2_multiplier) ascending, orders as
(learned, recency).

  M0 reproduction at h* = 600 s and replay counters   reproduction.csv, replay.csv
  M1 class order at the matched horizon: R_h*, reading class_order.csv, readings.csv
     (new and reproduced counted separately)
  M2 order within the matched class                   order.csv, readings.csv
  M3 admission at the matched horizon                 admission.csv, readings.csv
  M4 matched versus 600 s (new trace x cell)          versus600.csv, readings.csv

The counts of reading 1 for h* < 600 s ("new") and h* = 600 s ("reproduced")
are printed side by side and never merged.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --- the published grid and vocabularies (docs/matched-class-order-plan.md) -----------------------

TRACES = ("conversation_trace", "toolagent_trace")
ORDERS = ("learned", "recency")
ROLES = ("new", "reproduced")
MATCHED_READINGS = ("reuse_identification_at_matched_horizon_suffices",
                    "ranker_order_costs_at_matched_horizon")
SIGN_READINGS = ("consistent_gain", "consistent_loss", "mixed")
PREDICTED_COLUMN = "predicted_reuse_identification_at_matched_horizon_suffices"
INPUTS = ("replay", "reproduction", "class_order", "order", "admission", "versus600", "readings")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published matched class-order directory (read only)")
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
    """Trace x cell, then order and seed where a table has them."""
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "order" in row:
        order += (_rank(ORDERS, row["order"]),)
    elif "arm" in row:
        order += (_rank(ORDERS, row["arm"].rsplit("_", 1)[-1]),)
    if "seed" in row:
        order += (int(row["seed"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


def _readings(summary_rows, prefix: str) -> list[dict]:
    return [row for row in summary_rows if row["reading"].startswith(prefix)]


# --- M0: reproduction and replay counters ---------------------------------------------------------


def reproduction_counts(rows) -> list[dict]:
    """M0: per published arm, the replays at h* = 600 s compared, and how many
    equal the published row in tokens, counter digest, decision digest and all
    three."""
    out = []
    for arm in sorted({row["published_arm"] for row in rows},
                      key=lambda name: _rank(ORDERS, name.rsplit("_", 1)[-1])):
        members = [row for row in rows if row["published_arm"] == arm]
        out.append({"published_arm": arm, "replays": len(members),
                    **{f"same_{column}": sum(1 for row in members if _bool(row[f"same_{column}"]))
                       for column in ("avoided_prefill_tokens", "counters_sha256",
                                      "decision_sha256")},
                    "reproduces": sum(1 for row in members if _bool(row["reproduces"]))})
    return out


REPLAY_COLUMNS = ("extra_points_mean", "extra_points_ci95_half", "l2_rejections_mean",
                  "l2_evictions_mean", "overridden_decisions_mean",
                  "class_violations_seen_mean", "class_overridden_outside_admission_seen_mean")


def replay_counters(rows) -> list[dict]:
    """M0: five-seed means per trace x cell x arm, with h* and role."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], h_star=float(row["arm_parameter"]), role=row["role"],
                     seeds=int(row["seeds"]))
        for column in REPLAY_COLUMNS:
            entry[column] = _float(row.get(column, ""))
        out.append(entry)
    return out


# --- M1: class order at the matched horizon -------------------------------------------------------


def class_order_table(rows) -> list[dict]:
    """M1: per trace x cell, h*, role, the five-seed mean U of the published
    learned and evict_label rows and of both matched arms, R_h* of both, the
    reading, and the published 600-second R and reading beside them."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h_star=float(row["h_star"]), role=row["role"],
                     learned=float(row["U_learned_points_mean"]),
                     evict_label=float(row["U_evict_label_points_mean"]),
                     matched_learned=float(row["U_matched_learned_points_mean"]),
                     matched_recency=float(row["U_matched_recency_points_mean"]),
                     R_learned=_float(row["R_matched_learned"]),
                     R_recency=_float(row["R_matched_recency"]), reading=row["reading"],
                     R_600_learned=_float(row["R_600_published_learned"]),
                     R_600_recency=_float(row["R_600_published_recency"]),
                     reading_600=row["published_600_reading"])
        out.append(entry)
    return out


def class_order_counts(summary_rows) -> list[dict]:
    """M1 counts as published, one row per role (never merged)."""
    out = []
    for row in sorted(_readings(summary_rows, "1_"), key=lambda row: _rank(ROLES, row["scope"])):
        out.append({"scope": row["scope"], "cells": int(row["cells"]),
                    **{label: int(row[label]) for label in MATCHED_READINGS},
                    "predicted": row[PREDICTED_COLUMN],
                    "holds": (_bool(row["prediction_holds"]) if row["prediction_holds"] != ""
                              else "")})
    return out


# --- M2-M4: seed-paired differences ---------------------------------------------------------------


def order_table(rows) -> list[dict]:
    """M2: per trace x cell, U(learned order) - U(recency) at h*, seed-paired,
    with R_h* of the recency arm beside it and the published 600-second
    difference as context."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h_star=float(row["h_star"]), role=row["role"],
                     diff=float(row["learned_minus_recency_points_mean"]),
                     signs=_signs(row, "learned_minus_recency"),
                     reading=row["learned_minus_recency_reading"],
                     R_recency=_float(row["R_matched_recency"]),
                     diff_600=float(row["published_600_learned_minus_recency_points_mean"]),
                     reading_600=row["published_600_learned_minus_recency_reading"])
        out.append(entry)
    return out


def admission_table(rows) -> list[dict]:
    """M3: per trace x cell, U(label_binary_h*) - U(evict_binary_h*_recency),
    seed-paired."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h_star=float(row["h_star"]), role=row["role"], rung=row["rung_arm"],
                     rung_mean=float(row["U_label_binary_hstar_points_mean"]),
                     recency_mean=float(row["U_matched_recency_points_mean"]),
                     diff=float(row["label_binary_minus_recency_points_mean"]),
                     signs=_signs(row, "label_binary_minus_recency"),
                     reading=row["label_binary_minus_recency_reading"])
        out.append(entry)
    return out


def versus600_table(rows) -> list[dict]:
    """M4: per new trace x cell and order, the matched arm minus the published
    600-second arm, seed-paired, with both R."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h_star=float(row["h_star"]), order=row["order"],
                     matched=float(row["U_matched_points_mean"]),
                     published=float(row["U_published_600_points_mean"]),
                     diff=float(row["matched_minus_600_points_mean"]),
                     signs=_signs(row, "matched_minus_600"),
                     reading=row["matched_minus_600_reading"],
                     R_matched=_float(row["R_matched"]),
                     R_600=_float(row["R_600_published"]))
        out.append(entry)
    return out


def sign_counts_table(summary_rows) -> list[dict]:
    """M2-M4 counts as published: consistent gain / loss / mixed per reading."""
    return [{"reading": row["reading"], "scope": row["scope"], "cells": int(row["cells"]),
             **{label: int(row[label]) for label in SIGN_READINGS}}
            for row in summary_rows if not row["reading"].startswith("1_")]


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


IDENTITY = [_column("trace", "trace"), _column("cell", "cell")]
MATCHED = [_column("h*", "h_star", _seconds), _column("role", "role")]


def print_reproduction(reproduction_rows, replay_rows) -> None:
    print_table("M0 reproduction at h* = 600 s",
                [_column("published arm", "published_arm"), _column("replays", "replays"),
                 _column("same tokens", "same_avoided_prefill_tokens"),
                 _column("same counters", "same_counters_sha256"),
                 _column("same decisions", "same_decision_sha256"),
                 _column("reproduces", "reproduces")], reproduction_counts(reproduction_rows),
                note="evict_binary_600_learned / _recency against the published "
                     "evict_binary_learned / _recency rows of the same trace x cell x seed.")
    formats = {column: _count_mean for column in REPLAY_COLUMNS
               if not column.startswith("extra_points")}
    print_table("M0 replay counters (five-seed means; U in points)",
                IDENTITY + [_column("arm", "arm")] + MATCHED + [_column("seeds", "seeds")]
                + [_column(column.removesuffix("_mean"), column, formats.get(column, _number))
                   for column in REPLAY_COLUMNS], replay_counters(replay_rows))


def print_class_order(rows, summary_rows) -> None:
    print_table("M1 class order at the matched horizon (five-seed means; U in points)",
                IDENTITY + MATCHED
                + [_column("U(learned)", "learned", _number),
                   _column("U(evict_label)", "evict_label", _number),
                   _column("U(matched learned)", "matched_learned", _number),
                   _column("U(matched recency)", "matched_recency", _number),
                   _column("R_h*(learned order)", "R_learned", _number),
                   _column("R_h*(recency)", "R_recency", _number), _column("reading", "reading"),
                   _column("R_600(learned order), published", "R_600_learned", _number),
                   _column("R_600(recency), published", "R_600_recency", _number),
                   _column("reading at 600 s", "reading_600")],
                class_order_table(rows),
                note="R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned)); "
                     "reuse_identification_at_matched_horizon_suffices when R_h*(learned order) "
                     ">= 0.9.")
    print_table("M1 reading counts, new (h* < 600 s) and reproduced (h* = 600 s), never merged",
                [_column("scope", "scope")]
                + [_column(label, label) for label in MATCHED_READINGS]
                + [_column("of", "cells"), _column("predicted", "predicted"),
                   _column("holds", "holds")], class_order_counts(summary_rows))


def print_differences(order_rows, admission_rows, versus_rows, summary_rows) -> None:
    print_table("M2 order within the matched class: U(learned order) - U(recency), points",
                IDENTITY + MATCHED
                + [_column("diff", "diff", _number), _column("+/0/-", "signs", _sign_text),
                   _column("reading", "reading"), _column("R_h*(recency)", "R_recency", _number),
                   _column("at 600 s, published", "diff_600", _number),
                   _column("reading at 600 s", "reading_600")], order_table(order_rows))
    print_table("M3 admission at the matched horizon: U(label_binary_h*) - U(recency arm), "
                "points (descriptive)",
                IDENTITY + MATCHED
                + [_column("rung", "rung"), _column("U(rung)", "rung_mean", _number),
                   _column("U(recency arm)", "recency_mean", _number),
                   _column("diff", "diff", _number), _column("+/0/-", "signs", _sign_text),
                   _column("reading", "reading")], admission_table(admission_rows))
    print_table("M4 matched minus 600 s, new trace x cell, points (descriptive)",
                IDENTITY + [_column("h*", "h_star", _seconds), _column("order", "order"),
                            _column("U(matched)", "matched", _number),
                            _column("U(600), published", "published", _number),
                            _column("diff", "diff", _number),
                            _column("+/0/-", "signs", _sign_text), _column("reading", "reading"),
                            _column("R_h*", "R_matched", _number),
                            _column("R_600", "R_600", _number)], versus600_table(versus_rows))
    print_table("M2-M4 reading counts",
                [_column("reading", "reading"), _column("scope", "scope")]
                + [_column(label, label) for label in SIGN_READINGS]
                + [_column("of", "cells")], sign_counts_table(summary_rows),
                note="consistent_gain / consistent_loss / mixed: seed-paired sign in all five "
                     "seeds.")


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Matched class-order tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_reproduction(tables["reproduction"], tables["replay"])
    print_class_order(tables["class_order"], tables["readings"])
    print_differences(tables["order"], tables["admission"], tables["versus600"],
                      tables["readings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
