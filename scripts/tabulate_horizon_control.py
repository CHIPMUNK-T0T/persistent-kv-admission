#!/usr/bin/env python3
"""Tabulate the published horizon and class-order control tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_horizon_control.py`
published in PAPER_DIR and prints every number the findings document quotes,
as Markdown tables on stdout, one section per pre-registered reading plus the
replay counters. Nothing is written, fitted, rerun or re-derived from the
replays: each section is arithmetic on the published columns, so a number in
the document can be checked against this output and this output against the
CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_horizon_control.py`) and a small printer. Points are
printed to 3 decimals, shares to 3, counts of seeds as +/0/-. Traces are in
grid order, cells by (l1_fraction, l2_multiplier) ascending, arms and
horizons in the plan's order.

  R0 replay counters per trace x cell x arm                    replay.csv
  R1 horizon: S_h, the smallest, its horizon, reading, counts  horizon_reading.csv, horizon.csv
  R2 class order: R(a), reading, learned minus recency, counts class_order.csv
  R3 location under leaf16 next to the published all16 label   location_leaf.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

# --- the published grid and vocabularies (docs/horizon-control-plan.md) -------------------------

TRACES = ("conversation_trace", "toolagent_trace")
HORIZONS = (6.0, 15.0, 60.0, 300.0, 600.0)
ARMS = ("label_binary_6", "label_binary_15", "label_binary_60", "label_binary_300",
        "label_binary_600", "evict_binary_learned", "evict_binary_recency", "adm_label",
        "evict_label")
HORIZON_READINGS = ("reuse_label_suffices", "order_needed")
CLASS_ORDER_READINGS = ("reuse_identification_suffices", "ranker_order_costs")
READINGS = ("consistent_gain", "consistent_loss", "mixed")
LOCATIONS = ("admission_located", "eviction_located", "both", "neither")
LOCATION_TERMS = ("G", "A", "E", "interaction")
SUBSET_SHORTFALL = 0.05
INPUTS = ("replay", "horizon", "horizon_reading", "class_order", "location_leaf")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published horizon-control directory (read only)")
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
    """Trace x cell, then arm and horizon where a table has them."""
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "arm" in row:
        order += (_rank(ARMS, row["arm"]),)
    if "h" in row:
        order += (float(row["h"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


# --- R0: replay counters ---------------------------------------------------------------------------

REPLAY_COLUMNS = ("extra_points_mean", "extra_points_ci95_half", "l2_rejections_mean",
                  "l2_evictions_mean", "overridden_decisions_mean",
                  "l2_present_unusable_tokens_mean", "m2_mean", "m4_mean")


def replay_counters(rows) -> list[dict]:
    """R0: five-seed means per trace x cell x arm, with the arm's mechanism."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], mechanism=row["mechanism"], seeds=int(row["seeds"]))
        for column in REPLAY_COLUMNS:
            entry[column] = _float(row.get(column, ""))
        out.append(entry)
    return out


# --- R1: horizon ------------------------------------------------------------------------------------


def horizon_reading_table(rows) -> list[dict]:
    """R1: per trace x cell, S_h of the five horizons on the five-seed means,
    the smallest, the horizons attaining it, the reading, the learned level and
    the published S_600 with the subset flag."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for h in HORIZONS:
            entry[f"S_{h:g}"] = _float(row[f"S_{h:g}"])
        entry.update(S_min=_float(row["S_min"]), S_min_horizons=row["S_min_horizons"],
                     reading=row["reading"], S_learned=_float(row["S_learned"]),
                     published_S_600=_float(row["published_S_600"]),
                     subset=_bool(row["in_published_subset"]))
        out.append(entry)
    return out


def horizon_signs_table(rows) -> list[dict]:
    """R1, per h: `U(label) - U(label_binary_h)` on the five-seed means with its
    seed signs and reading, and S_h."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(h=float(row["h"]), diff=float(row["label_minus_arm_points_mean"]),
                     signs=_signs(row, "label_minus_arm"),
                     reading=row["label_minus_arm_reading"], S=_float(row["S"]))
        out.append(entry)
    return out


def horizon_counts(rows) -> list[dict]:
    """R1 counts: trace x cells per reading, over all of them and over those
    whose published S_600 exceeds 0.05."""
    out = []
    for scope, members in (("all", rows),
                           (f"published S_600 > {SUBSET_SHORTFALL:g}",
                            [row for row in rows if _bool(row["in_published_subset"])])):
        counts = Counter(row["reading"] for row in members)
        out.append({"scope": scope, **{reading: counts.get(reading, 0)
                                       for reading in HORIZON_READINGS},
                    "cells": len(members)})
    return out


# --- R2: class order --------------------------------------------------------------------------------


def class_order_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(learned=float(row["U_learned_points_mean"]),
                     evict_label=float(row["U_evict_label_points_mean"]),
                     evict_binary_learned=float(row["U_evict_binary_learned_points_mean"]),
                     evict_binary_recency=float(row["U_evict_binary_recency_points_mean"]),
                     R_learned=_float(row["R_evict_binary_learned"]),
                     R_recency=_float(row["R_evict_binary_recency"]), reading=row["reading"],
                     diff=float(row["learned_minus_recency_points_mean"]),
                     signs=_signs(row, "learned_minus_recency"),
                     diff_reading=row["learned_minus_recency_reading"])
        out.append(entry)
    return out


def class_order_counts(rows) -> list[dict]:
    """R2 counts: trace x cells per reading, and per reading of the
    seed-paired learned-minus-recency difference."""
    readings = Counter(row["reading"] for row in rows)
    differences = Counter(row["learned_minus_recency_reading"] for row in rows)
    return [{**{reading: readings.get(reading, 0) for reading in CLASS_ORDER_READINGS},
             **{reading: differences.get(reading, 0) for reading in READINGS},
             "cells": len(rows)}]


# --- R3: location under leaf16 ----------------------------------------------------------------------


def location_leaf_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for term in LOCATION_TERMS:
            entry[term] = float(row[f"{term}_points_mean"])
            entry[f"{term}_signs"] = _signs(row, term)
            entry[f"all16_{term}"] = float(row[f"all16_{term}_points_mean"])
        entry.update(A_share_of_G=_float(row["A_share_of_G"]),
                     E_share_of_G=_float(row["E_share_of_G"]), location=row["location"],
                     all16_location=row["all16_location"], same=_bool(row["same_as_all16"]),
                     seed_locations=row["seed_locations"])
        out.append(entry)
    return out


def location_leaf_counts(rows) -> list[dict]:
    """R3 counts: trace x cells per leaf16 label, and with the same label as
    the published all16 one."""
    labels = Counter(row["location"] for row in rows)
    return [{**{label: labels.get(label, 0) for label in LOCATIONS},
             "same_as_all16": sum(1 for row in rows if _bool(row["same_as_all16"])),
             "cells": len(rows)}]


# --- printing ---------------------------------------------------------------------------------------


def _number(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _count_mean(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.1f}"


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


def print_replay(rows) -> None:
    formats = {"l2_rejections_mean": _count_mean, "l2_evictions_mean": _count_mean,
               "overridden_decisions_mean": _count_mean,
               "l2_present_unusable_tokens_mean": _count_mean}
    print_table("R0 replay counters (five-seed means; U in points)",
                IDENTITY + [_column("arm", "arm"), _column("mechanism", "mechanism"),
                            _column("seeds", "seeds")]
                + [_column(column.removesuffix("_mean"), column, formats.get(column, _number))
                   for column in REPLAY_COLUMNS], replay_counters(rows))


def print_horizon(reading_rows, per_h_rows) -> None:
    print_table("R1 horizon: S_h of label_binary_h (five-seed means)",
                IDENTITY + [_column(f"S_{h:g}", f"S_{h:g}", _number) for h in HORIZONS]
                + [_column("S_min", "S_min", _number), _column("at h", "S_min_horizons"),
                   _column("reading", "reading"), _column("S(learned)", "S_learned", _number),
                   _column("published S_600", "published_S_600", _number),
                   _column("subset", "subset")],
                horizon_reading_table(reading_rows),
                note="S_h = (U(label) - U(label_binary_h)) / (U(label) - U(lru)); "
                     "reuse_label_suffices when the smallest S_h is <= 0.10. subset: published "
                     "S_600 > 0.05.")
    print_table("R1 U(label) - U(label_binary_h) per horizon (five-seed means, points)",
                IDENTITY + [_column("h", "h", lambda value: f"{value:g}"),
                            _column("diff", "diff", _number), _column("+/0/-", "signs", _sign_text),
                            _column("reading", "reading"), _column("S_h", "S", _number)],
                horizon_signs_table(per_h_rows))
    print_table("R1 reading counts", [_column("scope", "scope")]
                + [_column(reading, reading) for reading in HORIZON_READINGS]
                + [_column("of", "cells")], horizon_counts(reading_rows))


def print_class_order(rows) -> None:
    print_table("R2 class order (five-seed means; U in points)",
                IDENTITY + [_column("U(learned)", "learned", _number),
                            _column("U(evict_label)", "evict_label", _number),
                            _column("U(evict_binary_learned)", "evict_binary_learned", _number),
                            _column("U(evict_binary_recency)", "evict_binary_recency", _number),
                            _column("R(learned order)", "R_learned", _number),
                            _column("R(recency)", "R_recency", _number),
                            _column("reading", "reading"),
                            _column("learned - recency", "diff", _number),
                            _column("+/0/-", "signs", _sign_text),
                            _column("diff reading", "diff_reading")],
                class_order_table(rows),
                note="R(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned)); "
                     "reuse_identification_suffices when R(evict_binary_learned) >= 0.9.")
    print_table("R2 reading counts",
                [_column(reading, reading) for reading in CLASS_ORDER_READINGS + READINGS]
                + [_column("of", "cells")], class_order_counts(rows),
                note="consistent_gain / consistent_loss / mixed: U(evict_binary_learned) - "
                     "U(evict_binary_recency), seed-paired.")


def print_location_leaf(rows) -> None:
    columns = list(IDENTITY)
    for term in LOCATION_TERMS:
        columns += [_column(term, term, _number), _column(f"{term} +/0/-", f"{term}_signs",
                                                          _sign_text)]
    columns += [_column("A/G", "A_share_of_G", _number), _column("E/G", "E_share_of_G", _number),
                _column("leaf16", "location"), _column("all16 (published)", "all16_location"),
                _column("same", "same"), _column("all16 G", "all16_G", _number),
                _column("all16 A", "all16_A", _number), _column("all16 E", "all16_E", _number),
                _column("seed labels", "seed_locations")]
    print_table("R3 location by decision type under leaf16 (five-seed means, points)", columns,
                location_leaf_table(rows),
                note="G = U(label) - U(learned), A = U(adm_label) - U(learned), E = "
                     "U(evict_label) - U(learned), interaction = A + E - G, all under leaf16; a "
                     "term reaches half when it is >= 0.5 G. +/0/-: seed signs.")
    print_table("R3 label counts", [_column(label, label) for label in LOCATIONS]
                + [_column("same as all16", "same_as_all16"), _column("of", "cells")],
                location_leaf_counts(rows))


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Horizon and class-order control tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_replay(tables["replay"])
    print_horizon(tables["horizon_reading"], tables["horizon"])
    print_class_order(tables["class_order"])
    print_location_leaf(tables["location_leaf"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
