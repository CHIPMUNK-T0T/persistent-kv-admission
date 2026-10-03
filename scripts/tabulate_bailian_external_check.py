#!/usr/bin/env python3
"""Tabulate the published Bailian external-check tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_bailian_external_check.py`
published in PAPER_DIR (the anchor, the smoke and the main run) and, when given,
in GRANULARITY_DIR (control C), and prints every number a findings document
would quote, as Markdown tables on stdout. Nothing is written, fitted, rerun or
re-derived from the replays: each section is a selection of published columns
(stdlib `csv` only), so a number in the document can be checked against this
output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_run_bailian_external_check.py`) and a printer. Points,
shares and ratios are printed to 3 decimals. Traces are in the plan's order,
cells by (l1_fraction, l2_multiplier), mechanisms all16 then leaf16.

  X0 anchor reproduction and the smoke                    anchor.csv, smoke.csv, smoke_config.json
  X1 transplant (reading 1)                               transplant.csv
  X2 grid (reading 2): S_h, h_best, classification        grid.csv, grid_summary.csv
  X3 monotonicity of h_best in L2 (prediction 4)          monotonicity.csv
  X4 recency beyond the bit, D_rand (reading 3)           random.csv
  X5 calibration half (reading 4)                         calibration.csv
  X6 Mooncake side by side (reading 6)                    mooncake_side_by_side.csv
  X7 order beyond the bit by mechanism (reading 7)        order_beyond_bit.csv,
                                                          mooncake_order_beyond_bit.csv
  X8 the predictions and the descriptive counts           readings.csv
  X9 checks                                               checks.csv
  C1 granularity control (C)                              GRANULARITY_DIR/granularity.csv, checks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TRACES = ("bailian_toc_trace", "bailian_tob_trace", "bailian_thinking_trace",
          "bailian_coder_trace", "conversation_trace", "toolagent_trace")
MECHANISMS = ("all16", "leaf16")
GRID_SECONDS = (6.0, 15.0, 60.0, 150.0, 300.0, 600.0, 1200.0)
INPUTS = ("anchor", "smoke", "transplant", "grid", "grid_summary", "monotonicity", "random",
          "calibration", "mooncake_side_by_side", "order_beyond_bit",
          "mooncake_order_beyond_bit", "readings", "checks")
GRANULARITY_INPUTS = ("granularity", "checks")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published bailian_external_check directory (read only)")
    parser.add_argument("--granularity-dir", type=Path, default=None,
                        help="published bailian_granularity_control directory (read only)")
    return parser.parse_args(argv)


# --- parsing and ordering -------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float:
    return math.nan if value in ("", None) else float(value)


def _bool(value) -> bool | str:
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return ""


def _rank(preferred, value) -> tuple:
    return (preferred.index(value), "") if value in preferred else (len(preferred), str(value))


def _order(row) -> tuple:
    key: tuple = (_rank(("bailian", "mooncake"), row.get("workload", "bailian")),
                  _rank(TRACES, row["trace"]))
    if "mechanism" in row:
        key += (_rank(MECHANISMS, row["mechanism"]),)
    if "l1_fraction" in row:
        key += (float(row["l1_fraction"]), float(row["l2_multiplier"]))
    if "role" in row:
        key += (_rank(("h_star", "600"), row["role"]),)
    if "window" in row:
        key += (_rank(("W", "full", "W1->W2"), row["window"]),)
    return key


# --- X0 the anchor and the smoke --------------------------------------------------------------------


def anchor_counts(rows) -> list[dict]:
    """Per mechanism x arm: replays, how many reproduce, digests compared,
    counter columns compared."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mechanism"], row["arm"]), []).append(row)
    out = []
    for (mechanism, arm), members in sorted(groups.items(),
                                            key=lambda item: (_rank(MECHANISMS, item[0][0]),
                                                              item[0][1])):
        out.append({"mechanism": mechanism, "arm": arm, "replays": len(members),
                    "sources": members[0]["reference_sources"],
                    "same_tokens": sum(1 for row in members
                                       if _bool(row["same_avoided_prefill_tokens"]) is True),
                    "counter_digests": sum(1 for row in members
                                           if _bool(row["same_counters_sha256"]) is True),
                    "decision_digests": sum(1 for row in members
                                            if _bool(row["same_decision_sha256"]) is True),
                    "counter_columns": sorted({int(row["counter_columns_compared"])
                                               for row in members}),
                    "reproduce": sum(1 for row in members if _bool(row["reproduces"]) is True)})
    return out


def smoke_rows(rows) -> list[dict]:
    """Per smoke replay: seconds, peak RSS, decisions and the checks."""
    return [{"trace": row["trace"], "block_tokens": int(row["block_tokens"]), "arm": row["arm"],
             "seconds": float(row["seconds"]), "rss_mib": float(row["worker_peak_rss_mib"]),
             "parent_rss_mib": _float(row.get("parent_rss_mib_at_fork", "")),
             "decisions": int(row["l2_decisions"]),
             "checks": all(_bool(row[column]) is True for column in row
                           if column.startswith("check_"))}
            for row in sorted(rows, key=lambda row: (-int(row["block_tokens"]),
                                                     _rank(TRACES, row["trace"]), row["arm"]))]


# --- X1-X7 the readings ---------------------------------------------------------------------------


def transplant_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_order):
        out.append({"mechanism": row["mechanism"], "trace": row["trace"], "cell": row["cell"],
                    "h_star": float(row["h_star"]), "evaluable": _bool(row["evaluable"]),
                    "lru": _float(row["U_lru_points_mean_W"]),
                    "label": _float(row["U_label_points_mean_W"]),
                    "hstar": _float(row["U_hstar_points_mean_W"]),
                    "S": _float(row["S_hstar_W"]), "S_min": _float(row["S_hstar_seed_min_W"]),
                    "S_max": _float(row["S_hstar_seed_max_W"]),
                    "S_full": _float(row["S_hstar_full"]), "reading": row["reading"],
                    "H_off": _float(row["H_off_points_W"]), "T": _float(row["T_points_W"]),
                    "share": _float(row["label_share_of_T_W"])})
    return out


def grid_table(grid_rows, summary_rows) -> list[dict]:
    """Per mechanism x trace x cell: S_h on W at every grid point, h_best,
    min_h S_h, the classification, the direction and the 1,200-second flag."""
    shortfalls: dict[tuple, dict[float, float]] = {}
    for row in grid_rows:
        key = (row["mechanism"], row["trace"], float(row["l1_fraction"]),
               float(row["l2_multiplier"]))
        shortfalls.setdefault(key, {})[float(row["h"])] = _float(row["S_W"])
    out = []
    for row in sorted(summary_rows, key=_order):
        key = (row["mechanism"], row["trace"], float(row["l1_fraction"]),
               float(row["l2_multiplier"]))
        entry = {"mechanism": row["mechanism"], "trace": row["trace"], "cell": row["cell"],
                 "h_star": float(row["h_star"])}
        for h in GRID_SECONDS:
            entry[f"S_{h:g}"] = shortfalls.get(key, {}).get(h, math.nan)
        entry.update(h_best=row["h_best_W"], min_S=_float(row["min_S_W"]),
                     classification=row["classification_W"], direction=row["direction_W"],
                     flag=_bool(row["rests_on_1200_W"]), h_best_full=row["h_best_full"],
                     min_S_full=_float(row["min_S_full"]),
                     classification_full=row["classification_full"])
        out.append(entry)
    return out


def monotonicity_table(rows) -> list[dict]:
    return [{"mechanism": row["mechanism"], "trace": row["trace"], "window": row["window"],
             "sequence": row["sequence"], "levels": int(row["evaluable_levels"]),
             "assessable": _bool(row["assessable"]), "nondecreasing": _bool(row["nondecreasing"]),
             "flag": _bool(row["rests_on_1200"])} for row in sorted(rows, key=_order)]


def random_table(rows) -> list[dict]:
    return [{"mechanism": row["mechanism"], "trace": row["trace"], "cell": row["cell"],
             "h_star": float(row["h_star"]), "evaluable": _bool(row["evaluable"]),
             "D": _float(row["D_rand_points_mean_W"]), "D_min": _float(row["D_rand_points_min_W"]),
             "D_max": _float(row["D_rand_points_max_W"]), "signs": row["D_rand_seed_signs_W"],
             "reading": row["D_rand_reading_W"], "D_full": _float(row["D_rand_points_mean_full"]),
             "reading_full": row["D_rand_reading_full"]} for row in sorted(rows, key=_order)]


def calibration_table(rows) -> list[dict]:
    return [{"mechanism": row["mechanism"], "trace": row["trace"], "cell": row["cell"],
             "evaluable": _bool(row["evaluable"]), "h_cal": row["h_cal"],
             "S_W1": _float(row["S_hcal_W1"]), "S_W2": _float(row["S_hcal_W2"]),
             "suffices_W2": _bool(row["hcal_suffices_W2"]), "h_best_W2": row["h_best_W2"],
             "min_S_W2": _float(row["min_S_W2"]), "flag": _bool(row["rests_on_1200"])}
            for row in sorted(rows, key=_order)]


def side_by_side_table(rows) -> list[dict]:
    return [{"workload": row["workload"], "mechanism": row["mechanism"], "trace": row["trace"],
             "cell": row["cell"], "h_star": float(row["h_star"]),
             "S_W": _float(row.get("S_hstar_W", "")), "S_full": _float(row["S_hstar_full"]),
             "min_S_W": _float(row.get("min_S_W", "")), "min_S_full": _float(row["min_S_full"]),
             "h_best_full": row["h_best_full"], "grid_full": row["grid_points_full"],
             "D_W": _float(row.get("D_rand_points_mean_W", "")),
             "D_full": _float(row.get("D_rand_points_mean_full", "")),
             "D_reading_full": row.get("D_rand_reading_full", ""),
             "note": row.get("D_rand_note", "")} for row in sorted(rows, key=_order)]


def order_table(rows, window: str) -> list[dict]:
    """Reading 7 per trace x cell x role on one window: Delta per mechanism,
    its seed range and reading, separated, reversed."""
    out = []
    for row in sorted(rows, key=_order):
        entry = {"workload": row.get("workload", "bailian"), "trace": row["trace"],
                 "cell": row["cell"], "role": row["role"], "h": float(row["h"]),
                 "status": row["status"], "predicted": _bool(row["predicted"])}
        for mechanism in MECHANISMS:
            entry[f"{mechanism}_mean"] = _float(row.get(f"Delta_{mechanism}_points_mean_{window}",
                                                        ""))
            entry[f"{mechanism}_min"] = _float(row.get(f"Delta_{mechanism}_points_min_{window}", ""))
            entry[f"{mechanism}_max"] = _float(row.get(f"Delta_{mechanism}_points_max_{window}", ""))
            entry[f"{mechanism}_reading"] = row.get(f"Delta_{mechanism}_reading_{window}", "")
        entry["separated"] = _bool(row.get(f"separated_{window}", ""))
        entry["reversed"] = _bool(row.get(f"reversed_{window}", ""))
        out.append(entry)
    return out


def readings_table(rows) -> list[dict]:
    return [{"reading": row["reading"], "workload": row["workload"],
             "mechanism": row["mechanism"], "kind": row["kind"], "window": row["window"],
             "counted": row["counted"], "count": row["count"], "evaluable": row["evaluable"],
             "cells": row["cells"], "threshold": row["threshold"], "holds": row["holds"],
             "on_1200": row["count_resting_on_1200"], "detail": row["detail"]}
            for row in rows]


def checks_table(rows) -> list[dict]:
    return [{"check": row["check"], "checked": row["checked"], "problems": row["problems"],
             "passes": row["passes"], "scope": row["scope"]} for row in rows]


def granularity_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_order):
        entry = {"cell": row["cell"], "h_star": float(row["h_star"]), "window": row["window"],
                 "seeds": int(row["seeds"])}
        for granularity in ("16", "512"):
            for arm in ("lru", "label", "hstar"):
                entry[f"{arm}_{granularity}"] = _float(row[f"U_{arm}_points_mean_{granularity}"])
            entry[f"S_{granularity}"] = _float(row[f"S_hstar_{granularity}"])
        entry["agree"] = row["agree"]
        out.append(entry)
    return out


# --- printing -------------------------------------------------------------------------------------


def _number(value) -> str:
    if isinstance(value, str):
        return value
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _text(value) -> str:
    return str(value)


def print_table(title: str, rows, columns=None, note: str = "") -> None:
    print(f"### {title}\n")
    if note:
        print(f"{note}\n")
    if not rows:
        print("(no rows)\n")
        return
    columns = columns or list(rows[0])
    body = [[_number(row[column]) if isinstance(row[column], float) else _text(row[column])
             for column in columns] for row in rows]
    widths = [max(len(text) for text in column) for column in zip(columns, *body)]
    print("| " + " | ".join(text.ljust(width) for text, width in zip(columns, widths)) + " |")
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for cells in body:
        print("| " + " | ".join(text.ljust(width) for text, width in zip(cells, widths)) + " |")
    print()


def print_main(tables: dict, smoke_config: dict | None) -> None:
    print_table("X0 anchor: reproduction of the published Mooncake rows",
                anchor_counts(tables["anchor"]),
                note="Each replay against its published row(s): tokens, each digest the row "
                     "carries, every counter column where it carries no digest.")
    print_table("X0 smoke: seconds, peak RSS and checks per replay (no utility)",
                smoke_rows(tables["smoke"]))
    if smoke_config:
        verdicts = [{"verdict": name, **{key: value for key, value in verdict.items()
                                         if not isinstance(value, (list, dict))}}
                    for name, verdict in (("512", smoke_config.get("verdict_512")),
                                          ("16", smoke_config.get("verdict_16"))) if verdict]
        for verdict in verdicts:
            print_table(f"X0 smoke verdict at {verdict['verdict']} tokens", [verdict])
    print_table("X1 transplant: S_h* on W (five-seed means; U in points)",
                transplant_table(tables["transplant"]),
                note="S_h* = (U(label) - U(label_binary_h*)) / (U(label) - U(lru)); suffices at "
                     "<= 0.10; not computed below 1.0 point of headroom; T = H_off - U(lru).")
    print_table("X2 grid: S_h on W, h_best, classification",
                grid_table(tables["grid"], tables["grid_summary"]),
                note="h_best ties to the smaller h; flag: the reading rests on 1,200 s.")
    print_table("X3 monotonicity of h_best in L2 bytes", monotonicity_table(tables["monotonicity"]))
    print_table("X4 recency beyond the bit: D_rand = U(random_h*) - U(label_binary_h*), points",
                random_table(tables["random"]))
    print_table("X5 calibration half: h_cal on W1, S_h_cal on W2",
                calibration_table(tables["calibration"]))
    print_table("X6 Mooncake side by side (Mooncake on its full window; context, not pooled)",
                side_by_side_table(tables["mooncake_side_by_side"]))
    for window in ("W", "full"):
        print_table(f"X7 order beyond the bit: Delta_m(h) = U_m(label) - U_m(label_binary_h), "
                    f"{window}", order_table(tables["order_beyond_bit"], window))
    print_table("X7 order beyond the bit: Mooncake counterpart (published rows, full window)",
                order_table(tables["mooncake_order_beyond_bit"], "full"))
    print_table("X8 predictions and descriptive counts", readings_table(tables["readings"]))
    print_table("X9 checks", checks_table(tables["checks"]))


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    config_path = args.paper_dir / "smoke_config.json"
    smoke_config = (json.loads(config_path.read_text(encoding="utf-8"))
                    if config_path.is_file() else None)
    print(f"# Bailian external-check tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_main(tables, smoke_config)
    if args.granularity_dir is not None:
        fine = {}
        for name in GRANULARITY_INPUTS:
            path = args.granularity_dir / f"{name}.csv"
            if not path.is_file():
                raise SystemExit(f"{path} not found")
            fine[name] = read_csv(path)
        print(f"# Granularity control\n\nSource: {args.granularity_dir}.\n")
        print_table("C1 granularity: S_h* and U at 16 and 512 tokens under the same bytes",
                    granularity_table(fine["granularity"]))
        print_table("C1 checks", checks_table(fine["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
