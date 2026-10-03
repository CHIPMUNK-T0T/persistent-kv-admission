#!/usr/bin/env python3
"""Tabulate the published leaf-matched horizon tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_leaf_matched_horizon.py`
published in PAPER_DIR and prints every number a findings document would
quote, as Markdown tables on stdout, one section per pre-registered reading
plus the reproduction check and the replay counters. Nothing is written,
fitted, rerun or re-derived from the replays: each section is arithmetic on
the published columns (stdlib `csv` only), so a number in the document can be
checked against this output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_leaf_matched_horizon.py`) and a small printer. Points,
shares and ratios are printed to 3 decimals, counts of seeds as +/0/-. Traces
are in grid order, cells by (l1_fraction, l2_multiplier) ascending, arms as
(label_binary_h*, learned order, recency, label). Every leaf16 value is
printed beside the published all16 value of the same arm.

  L0 reproduction of the leaf16 label rows and replay counters   reproduction.csv, replay.csv
  L1 horizon under leaf eligibility: S_h*, reading               horizon.csv
  L2 class order under leaf eligibility: R_h*, reading           class_order.csv
  L3 order within the matched class                              order.csv
  L4 admission (descriptive), arrival-candidate share            admission.csv
  L5 the counts against the three predictions                    readings.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --- the published grid and vocabularies (docs/leaf-matched-horizon-plan.md) ----------------------

TRACES = ("conversation_trace", "toolagent_trace")
ARM_KINDS = ("label_binary", "learned", "recency", "label")
HORIZON_READINGS = ("reuse_label_at_matched_horizon_suffices",
                    "reuse_label_at_matched_horizon_falls_short")
CLASS_READINGS = ("reuse_identification_at_matched_horizon_suffices",
                  "ranker_order_costs_at_matched_horizon")
SIGN_READINGS = ("consistent_gain", "consistent_loss", "mixed")
READINGS = ("1_horizon_under_leaf_eligibility", "2_class_order_under_leaf_eligibility",
            "3_order_within_matched_class", "4_admission_descriptive")
INPUTS = ("replay", "reproduction", "horizon", "class_order", "order", "admission", "readings")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published leaf-matched horizon directory (read only)")
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


def arm_kind(arm: str) -> str:
    """label_binary / learned / recency / label, for ordering."""
    if arm.startswith("label_binary_"):
        return "label_binary"
    if arm.startswith("evict_binary_"):
        return arm.rsplit("_", 1)[-1]
    return arm


def cell_key(row) -> tuple:
    """(trace, l1_fraction, l2_multiplier), the fractions compared as floats."""
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def _row_order(row) -> tuple:
    """Trace x cell, then arm and seed where a table has them."""
    key = cell_key(row)
    order: tuple = (_rank(TRACES, key[0]), key[1], key[2])
    if "arm" in row:
        order += (_rank(ARM_KINDS, arm_kind(row["arm"])),)
    if "seed" in row:
        order += (int(row["seed"]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"]),
            "h_star": float(row["h_star"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


# --- L0: reproduction and replay counters ---------------------------------------------------------


def reproduction_counts(rows) -> list[dict]:
    """L0: the `label` replays compared with the published leaf16 rows: how
    many equal them in tokens, in every published counter column, in each
    digest the published rows carry (and how many carry one), and in all."""
    if not rows:
        return []
    columns = sorted({int(row["counter_columns_compared"]) for row in rows})

    def published(column):
        return sum(1 for row in rows if row[f"published_{column}"] != "")

    def same(column):
        return sum(1 for row in rows
                   if row[f"same_{column}"] != "" and _bool(row[f"same_{column}"]))

    return [{"replays": len(rows),
             "same_tokens": same("avoided_prefill_tokens"),
             "counter_columns": "/".join(str(n) for n in columns),
             "all_counter_columns_equal": sum(1 for row in rows
                                              if row["counter_columns_differing"] == ""),
             "counter_digests_published": published("counters_sha256"),
             "same_counter_digest": same("counters_sha256"),
             "decision_digests_published": published("decision_sha256"),
             "same_decision_digest": same("decision_sha256"),
             "reproduces": sum(1 for row in rows if _bool(row["reproduces"]))}]


REPLAY_COLUMNS = ("extra_points_mean", "extra_points_ci95_half", "arrival_candidate_share_mean",
                  "l2_rejections_mean", "l2_evictions_mean", "overridden_decisions_mean",
                  "class_violations_seen_mean", "l2_present_unusable_tokens_mean")


def replay_counters(rows) -> list[dict]:
    """L0: five-seed means per trace x cell x arm."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], seeds=int(row["seeds"]))
        for column in REPLAY_COLUMNS:
            entry[column] = _float(row.get(column, ""))
        out.append(entry)
    return out


# --- L1-L4: the readings --------------------------------------------------------------------------


def horizon_table(rows) -> list[dict]:
    """L1: per trace x cell, the leaf16 rungs' U, the rung's U, S_h* and its
    reading, beside the published all16 S_h* and reading."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(rung=row["rung_arm"], label=float(row["U_label_points_mean"]),
                     lru=float(row["U_lru_points_mean"]),
                     rung_mean=float(row["U_label_binary_hstar_points_mean"]),
                     diff=float(row["label_minus_rung_points_mean"]),
                     signs=_signs(row, "label_minus_rung"),
                     S=_float(row["S_hstar"]), reading=row["reading"],
                     all16_S=_float(row["all16_S_hstar"]), all16_reading=row["all16_reading"])
        out.append(entry)
    return out


def class_order_table(rows) -> list[dict]:
    """L2: per trace x cell, the leaf16 references' U, both matched arms' U,
    R_h* of both and the reading (with whether it is predicted), beside the
    published all16 R_h* and reading."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(learned=float(row["U_learned_points_mean"]),
                     evict_label=float(row["U_evict_label_points_mean"]),
                     matched_learned=float(row["U_matched_learned_points_mean"]),
                     matched_recency=float(row["U_matched_recency_points_mean"]),
                     R_learned=_float(row["R_matched_learned"]),
                     R_recency=_float(row["R_matched_recency"]), reading=row["reading"],
                     predicted=_bool(row["predicted"]),
                     all16_R_learned=_float(row["all16_R_matched_learned"]),
                     all16_R_recency=_float(row["all16_R_matched_recency"]),
                     all16_reading=row["all16_reading"])
        out.append(entry)
    return out


def order_table(rows) -> list[dict]:
    """L3: per trace x cell, U(learned order) - U(recency), seed-paired, with
    R_h* of both arms, beside the published all16 difference."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(diff=float(row["learned_minus_recency_points_mean"]),
                     signs=_signs(row, "learned_minus_recency"),
                     reading=row["learned_minus_recency_reading"],
                     R_learned=_float(row["R_matched_learned"]),
                     R_recency=_float(row["R_matched_recency"]),
                     all16_diff=float(row["all16_learned_minus_recency_points_mean"]),
                     all16_signs=_signs(row, "all16_learned_minus_recency"),
                     all16_reading=row["all16_learned_minus_recency_reading"])
        out.append(entry)
    return out


def admission_table(rows) -> list[dict]:
    """L4: per trace x cell, U(label_binary_h*) - U(recency arm), seed-paired,
    and the five-seed mean share of decisions in which the arrival is a
    candidate in both arms, beside the published all16 values."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(diff=float(row["label_binary_minus_recency_points_mean"]),
                     signs=_signs(row, "label_binary_minus_recency"),
                     reading=row["label_binary_minus_recency_reading"],
                     share_rung=_float(row["arrival_candidate_share_label_binary_hstar_mean"]),
                     share_recency=_float(row["arrival_candidate_share_matched_recency_mean"]),
                     all16_diff=float(row["all16_label_binary_minus_recency_points_mean"]),
                     all16_signs=_signs(row, "all16_label_binary_minus_recency"),
                     all16_reading=row["all16_label_binary_minus_recency_reading"],
                     all16_share_rung=_float(
                         row["all16_arrival_candidate_share_label_binary_hstar_mean"]),
                     all16_share_recency=_float(
                         row["all16_arrival_candidate_share_matched_recency_mean"]))
        out.append(entry)
    return out


def counts_table(summary_rows) -> list[dict]:
    """L5: per reading, the count of the predicted outcome against the
    prediction, beside all16, and the count of every outcome."""
    out = []
    for row in sorted(summary_rows, key=lambda row: _rank(READINGS, row["reading"])):
        labels = (HORIZON_READINGS if row["reading"].startswith("1_")
                  else CLASS_READINGS if row["reading"].startswith("2_") else SIGN_READINGS)
        holds = row["prediction_holds"]
        out.append({"reading": row["reading"], "cells": int(row["cells"]),
                    "counted": row["counted"],
                    "count": int(row["count"]) if row["count"] != "" else "",
                    "required": f"{row['required']}/{row['of']}" if row["required"] != "" else "",
                    "holds": _bool(holds) if holds != "" else "",
                    "all16_count": int(row["all16_count"]) if row["all16_count"] != "" else "",
                    "outcomes": "; ".join(f"{label} {row[label]}" for label in labels),
                    "all16_outcomes": "; ".join(f"{label} {row['all16_' + label]}"
                                                for label in labels)})
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


IDENTITY = [_column("trace", "trace"), _column("cell", "cell"),
            _column("h*", "h_star", _seconds)]


def print_reproduction(reproduction_rows, replay_rows) -> None:
    print_table("L0 reproduction of the published leaf16 label rows",
                [_column(name, name) for name in ("replays", "same_tokens", "counter_columns",
                                                  "all_counter_columns_equal",
                                                  "counter_digests_published",
                                                  "same_counter_digest",
                                                  "decision_digests_published",
                                                  "same_decision_digest", "reproduces")],
                reproduction_counts(reproduction_rows),
                note="label (leaf16) against mechanism_control_001; the published rows carry "
                     "no digest, so every counter column they publish is compared.")
    formats = {column: _count_mean for column in REPLAY_COLUMNS
               if column.startswith(("l2_", "overridden", "class_"))}
    print_table("L0 replay counters (five-seed means; U in points)",
                IDENTITY + [_column("arm", "arm"), _column("seeds", "seeds")]
                + [_column(column.removesuffix("_mean"), column, formats.get(column, _number))
                   for column in REPLAY_COLUMNS], replay_counters(replay_rows))


def print_readings(tables) -> None:
    print_table("L1 horizon under leaf eligibility (five-seed means; U in points)",
                IDENTITY + [_column("rung", "rung"), _column("U(label)", "label", _number),
                            _column("U(lru)", "lru", _number),
                            _column("U(rung)", "rung_mean", _number),
                            _column("label - rung", "diff", _number),
                            _column("+/0/-", "signs", _sign_text), _column("S_h*", "S", _number),
                            _column("reading", "reading"),
                            _column("S_h*, all16", "all16_S", _number),
                            _column("reading, all16", "all16_reading")],
                horizon_table(tables["horizon"]),
                note="S_h* = (U(label) - U(label_binary_h*)) / (U(label) - U(lru)) with the "
                     "leaf16 rungs; suffices when S_h* <= 0.10.")
    print_table("L2 class order under leaf eligibility (five-seed means; U in points)",
                IDENTITY + [_column("U(learned)", "learned", _number),
                            _column("U(evict_label)", "evict_label", _number),
                            _column("U(learned order)", "matched_learned", _number),
                            _column("U(recency)", "matched_recency", _number),
                            _column("R_h*(learned order)", "R_learned", _number),
                            _column("R_h*(recency)", "R_recency", _number),
                            _column("reading", "reading"), _column("predicted", "predicted"),
                            _column("R_h*(learned order), all16", "all16_R_learned", _number),
                            _column("R_h*(recency), all16", "all16_R_recency", _number),
                            _column("reading, all16", "all16_reading")],
                class_order_table(tables["class_order"]),
                note="R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned)) with the "
                     "leaf16 references; suffices when R_h*(learned order) >= 0.9; the reading "
                     "at 0.25% x 1 is not predicted.")
    print_table("L3 order within the matched class: U(learned order) - U(recency), points",
                IDENTITY + [_column("diff", "diff", _number),
                            _column("+/0/-", "signs", _sign_text), _column("reading", "reading"),
                            _column("R_h*(learned order)", "R_learned", _number),
                            _column("R_h*(recency)", "R_recency", _number),
                            _column("diff, all16", "all16_diff", _number),
                            _column("+/0/-, all16", "all16_signs", _sign_text),
                            _column("reading, all16", "all16_reading")],
                order_table(tables["order"]))
    print_table("L4 admission: U(label_binary_h*) - U(recency arm), points (descriptive)",
                IDENTITY + [_column("diff", "diff", _number),
                            _column("+/0/-", "signs", _sign_text), _column("reading", "reading"),
                            _column("arrival a candidate, rung", "share_rung", _number),
                            _column("arrival a candidate, recency arm", "share_recency", _number),
                            _column("diff, all16", "all16_diff", _number),
                            _column("+/0/-, all16", "all16_signs", _sign_text),
                            _column("reading, all16", "all16_reading"),
                            _column("arrival a candidate, rung, all16", "all16_share_rung",
                                    _number),
                            _column("arrival a candidate, recency arm, all16",
                                    "all16_share_recency", _number)],
                admission_table(tables["admission"]),
                note="Share: evaluation-window decisions in which the arrival is a candidate "
                     "(stat_decisions_admission / stat_decisions), five-seed mean.")
    print_table("L5 counts against the predictions fixed before the run",
                [_column("reading", "reading"), _column("counted", "counted"),
                 _column("count", "count"), _column("of", "cells"),
                 _column("predicted", "required"), _column("holds", "holds"),
                 _column("all16", "all16_count"), _column("outcomes", "outcomes"),
                 _column("outcomes, all16", "all16_outcomes")],
                counts_table(tables["readings"]))


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Leaf-matched horizon tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_reproduction(tables["reproduction"], tables["replay"])
    print_readings(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
