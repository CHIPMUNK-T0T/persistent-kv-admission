#!/usr/bin/env python3
"""Tabulate the matched-horizon diagnosis tables for the findings document.

Read-only. The script reads the CSVs that
`scripts/run_matched_horizon_diagnosis.py` wrote into a run directory and
prints every number a findings document would quote, as Markdown tables on
stdout, one section per pre-registered reading
(`docs/matched-horizon-diagnosis-plan.md`). Nothing is written, recomputed
from the logs or re-derived: each section selects and orders published
columns, so a number in the document can be checked against this output and
this output against the CSVs. Standard library only.

Shares, rates and concordances are printed to 3 decimals, m4 to 4, points to
3. Traces are in grid order, cells by (l1_fraction, l2_multiplier) ascending,
targets `next_use` then `binary`, horizons ascending.

  M0 reproduction of the diagnosis at 600 s                   reproduction.csv
  M1 composition at h*, against 0.99, and its counts           composition_reading.csv,
                                                               composition_counts.csv
  M2 avoidable-eviction rate at h* against recency and uniform rates.csv, rates_counts.csv
     and the counts at every horizon and at h*
  M3 concordance at every horizon, the prediction and count    concordance.csv,
                                                               concordance_reading.csv,
                                                               concordance_counts.csv
  M4 bridge at h* and Spearman over the trace x cells          bridge.csv, bridge_spearman.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

TRACES = ("conversation_trace", "toolagent_trace")
TARGETS = ("next_use", "binary")
HORIZONS = (60.0, 150.0, 300.0, 600.0)
HORIZON_SELECTIONS = ("60", "150", "300", "600", "matched")
INPUTS = ("reproduction", "composition_reading", "composition_counts", "rates", "rates_counts",
          "concordance", "concordance_reading", "concordance_counts", "bridge", "bridge_spearman")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="matched-horizon diagnosis run directory (read only)")
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


def _row_order(row) -> tuple:
    order: tuple = (_rank(TRACES, row["trace"]), float(row["l1_fraction"]),
                    float(row["l2_multiplier"]))
    if "target" in row:
        order += (_rank(TARGETS, row["target"]),)
    if "h" in row:
        order += (float(row["h"]),)
    return order


def _identity(row) -> dict:
    out = {"trace": row["trace"], "cell": row["cell"]}
    if "target" in row:
        out["target"] = row["target"]
    return out


# --- M0: reproduction -----------------------------------------------------------------------------


def reproduction_table(rows) -> list[dict]:
    return [{"table": row["table"], "rows": int(row["rows_at_600"]),
             "compared": int(row["compared_values"]), "differing": int(row["differing_values"]),
             "equal": _bool(row["equal"])} for row in rows]


# --- M1: composition ------------------------------------------------------------------------------


def composition_table(rows) -> list[dict]:
    """M1: per trace x cell x target, the avoidable reusable eviction's share
    of the label excess at h* (five-seed mean and minimum), its share of
    decisions, the share at 600 s and whether the prediction holds."""
    return [{**_identity(row), "h_star": float(row["h_star"]),
             "share": float(row["excess_share_mean"]), "share_min": _float(row["excess_share_min"]),
             "decisions": float(row["decision_share_mean"]),
             "share_600": float(row["excess_share_at_600_mean"]), "holds": _bool(row["holds"]),
             "every_seed": _bool(row["holds_every_seed"])}
            for row in sorted(rows, key=_row_order)]


def composition_counts(rows) -> list[dict]:
    return [{"target": row["target"], "holds": int(row["holds"]),
             "every_seed": int(row["holds_every_seed"]), "cells": int(row["cells"])}
            for row in rows]


# --- M2: rates ------------------------------------------------------------------------------------


def matched_rate_table(rows, rate: str = "avoidable") -> list[dict]:
    """M2: per trace x cell x target at h*, the ranker's rate, recency's and
    the uniform expectation, with the seed reading against each."""
    return [{**_identity(row), "h_star": float(row["h_star"]),
             "ranker": _float(row["ranker_mean"]), "recency": _float(row["recency_mean"]),
             "uniform": _float(row["uniform_mean"]),
             "signs": row["ranker_minus_recency_seed_signs"], "vs_recency": row["vs_recency"],
             "vs_uniform": row["vs_uniform"]}
            for row in sorted(rows, key=_row_order)
            if row["rate"] == rate and _bool(row["matched"])]


def rate_counts(rows) -> list[dict]:
    """M2 counts per horizon selection x target x rate, out of the trace x cells."""
    out = [{"horizon": row["horizon"], "target": row["target"], "rate": row["rate"],
            "better": int(row["better_than_recency"]), "worse": int(row["worse_than_recency"]),
            "mixed": int(row["mixed_vs_recency"]),
            "better_uniform": int(row["better_than_uniform"]), "cells": int(row["cells"])}
           for row in rows]
    return sorted(out, key=lambda row: (_rank(("avoidable", "order"), row["rate"]),
                                        _rank(TARGETS, row["target"]),
                                        _rank(HORIZON_SELECTIONS, row["horizon"])))


# --- M3: concordance ------------------------------------------------------------------------------


def concordance_matrix(rows) -> list[dict]:
    """M3: per trace x cell x target, the ranker's and recency's five-seed
    mean concordance at every horizon, with h*."""
    by_cell: dict[tuple, dict] = {}
    for row in sorted(rows, key=_row_order):
        key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]), row["target"])
        entry = by_cell.setdefault(key, {**_identity(row), "h_star": float(row["h_star"])})
        h = float(row["h"])
        entry[f"ranker_{h:g}"] = _float(row["ranker_mean"])
        entry[f"recency_{h:g}"] = _float(row["recency_mean"])
    return list(by_cell.values())


def concordance_prediction_table(rows) -> list[dict]:
    """M3's registered prediction: the rows where it is registered."""
    return [{**_identity(row), "h_star": float(row["h_star"]),
             "at_h_star": float(row["ranker_at_h_star_mean"]),
             "at_600": float(row["ranker_at_600_mean"]),
             "difference": float(row["h_star_minus_600_mean"]),
             "signs": row["h_star_minus_600_seed_signs"], "prediction": row["prediction"]}
            for row in sorted(rows, key=_row_order) if _bool(row["registered"])]


def concordance_counts(rows) -> list[dict]:
    return [{"target": row["target"], "holds": int(row["holds"]), "fails": int(row["fails"]),
             "registered": int(row["registered"]), "below_600": int(row["below_600"]),
             "shorter": int(row["cells_h_star_below_600"])} for row in rows]


# --- M4: bridge -----------------------------------------------------------------------------------


def bridge_table(rows) -> list[dict]:
    """M4 at h*: per trace x cell x target, m4 over every decision and over
    the resident-victim decisions, and the two published U gaps."""
    return [{**_identity(row), "h_star": float(row["h_star"]),
             "m4_all": float(row["m4_all_mean"]), "m4_resident": float(row["m4_resident_mean"]),
             "ranker_arm": row["ranker_arm"],
             "gap_label": _float(row["u_label_minus_ranker_mean"]),
             "gap_binary": _float(row["u_label_binary_minus_ranker_mean"]),
             "arm": row["label_binary_arm"]}
            for row in sorted(rows, key=_row_order) if _bool(row["matched"])]


def spearman_table(rows) -> list[dict]:
    out = [{"target": row["target"], "ranker_arm": row["ranker_arm"],
            "horizon": row["horizon"], "m4": row["m4_subset"],
            "gap": row["u_gap"], "points": int(row["points"]), "cells": int(row["cells"]),
            "spearman": _float(row["spearman"])} for row in rows]
    return sorted(out, key=lambda row: (_rank(TARGETS, row["target"]),
                                        _rank(HORIZON_SELECTIONS[::-1], row["horizon"]),
                                        row["m4"], row["gap"]))


# --- printing -------------------------------------------------------------------------------------


def _number(value, digits: int = 3) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


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


IDENTITY = [_column("trace", "trace"), _column("cell", "cell"), _column("target", "target")]
SECONDS = lambda value: f"{value:g}"  # noqa: E731


def print_all(tables: dict[str, list[dict]]) -> None:
    print_table("M0 reproduction of the ranker-error diagnosis at h = 600 s",
                [_column("table", "table"), _column("rows", "rows"),
                 _column("values compared", "compared"), _column("differing", "differing"),
                 _column("equal", "equal")], reproduction_table(tables["reproduction"]))
    print_table("M1 composition at h*: the avoidable reusable eviction's share of the label_h* "
                "excess (every decision)",
                IDENTITY + [_column("h*", "h_star", SECONDS),
                            _column("excess share", "share", _number),
                            _column("min over seeds", "share_min", _number),
                            _column("decision share", "decisions", _number),
                            _column("excess share at 600 s", "share_600", _number),
                            _column(">= 0.99", "holds"), _column("every seed", "every_seed")],
                composition_table(tables["composition_reading"]),
                note="Five-seed means. Registered: the mean is at least 0.99.")
    print_table("M1 counts", [_column("target", "target"), _column("holds", "holds"),
                              _column("every seed", "every_seed"), _column("of", "cells")],
                composition_counts(tables["composition_counts"]))
    print_table("M2 avoidable-eviction rate at h* (decisions with both classes present)",
                IDENTITY + [_column("h*", "h_star", SECONDS),
                            _column("ranker", "ranker", _number),
                            _column("recency", "recency", _number),
                            _column("uniform", "uniform", _number),
                            _column("ranker - recency signs", "signs"),
                            _column("vs recency", "vs_recency"),
                            _column("vs uniform", "vs_uniform")],
                matched_rate_table(tables["rates"]),
                note="Five-seed means; better / worse = lower / higher in all five seeds.")
    print_table("M2 counts against recency (and better than uniform), out of the trace x cells",
                [_column("rate", "rate"), _column("target", "target"),
                 _column("horizon", "horizon"), _column("better", "better"),
                 _column("worse", "worse"), _column("mixed", "mixed"),
                 _column("better than uniform", "better_uniform"), _column("of", "cells")],
                rate_counts(tables["rates_counts"]),
                note="`matched` counts every cell at its own h*.")
    print_table("M3 within-decision concordance for the reuse bit at h (five-seed means)",
                IDENTITY + [_column("h*", "h_star", SECONDS)]
                + [_column(f"ranker {h:g}", f"ranker_{h:g}", _number) for h in HORIZONS]
                + [_column(f"recency {h:g}", f"recency_{h:g}", _number) for h in HORIZONS],
                concordance_matrix(tables["concordance"]),
                note="Share of (reusable, non-reusable) pairs whose key puts the non-reusable "
                     "candidate first, ties one half; uniform = 0.5.")
    print_table("M3 registered prediction: concordance for the h*-bit below the 600-second bit",
                IDENTITY + [_column("h*", "h_star", SECONDS),
                            _column("at h*", "at_h_star", _number),
                            _column("at 600 s", "at_600", _number),
                            _column("h* - 600", "difference", lambda v: _number(v, 4)),
                            _column("seed signs", "signs"), _column("prediction", "prediction")],
                concordance_prediction_table(tables["concordance_reading"]))
    print_table("M3 counts", [_column("target", "target"), _column("holds", "holds"),
                              _column("fails", "fails"), _column("of registered", "registered"),
                              _column("below 600 s (descriptive)", "below_600"),
                              _column("of cells with h* < 600 s", "shorter")],
                concordance_counts(tables["concordance_counts"]))
    print_table("M4 bridge at h* (descriptive)",
                IDENTITY + [_column("h*", "h_star", SECONDS),
                            _column("m4 all", "m4_all", lambda v: _number(v, 4)),
                            _column("m4 resident", "m4_resident", lambda v: _number(v, 4)),
                            _column("ranker arm", "ranker_arm"),
                            _column("U(label) - U(ranker arm)", "gap_label", _number),
                            _column("U(label_binary_h*) - U(ranker arm)", "gap_binary", _number),
                            _column("arm", "arm")],
                bridge_table(tables["bridge"]),
                note="m4 by the logged victim, five-seed means; U gaps in points, five-seed "
                     "means of the published all16 replays.")
    print_table("M4 Spearman over the trace x cells of each target (a description, not a test)",
                [_column("target", "target"), _column("ranker arm", "ranker_arm"),
                 _column("horizon", "horizon"),
                 _column("m4", "m4"), _column("U gap", "gap"), _column("points", "points"),
                 _column("of", "cells"), _column("Spearman", "spearman", _number)],
                spearman_table(tables["bridge_spearman"]),
                note="`matched`: every cell at its own h* (the registered description).")


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.run_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Matched-horizon diagnosis tables\n\nSource: {args.run_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_all(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
