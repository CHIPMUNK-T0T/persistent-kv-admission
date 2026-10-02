#!/usr/bin/env python3
"""Tabulate the published error-location tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_error_location.py`
published in PAPER_DIR and prints every number the findings document quotes,
as Markdown tables on stdout, one section per pre-registered reading plus the
replay counters. Nothing is written, fitted, rerun or re-derived from the
replays: each section is arithmetic on the published columns, so a number in
the document can be checked against this output and this output against the
CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_error_location.py`) and a small printer. Points are
printed to 3 decimals, shares and correlations to 3, percents to 1. Traces are
in grid order, cells by (l1_fraction, l2_multiplier) ascending, arms in the
plan's order.

  R0 replay counters and statistics per trace x cell x arm   replay.csv
  R1 location by decision type, and label counts              location.csv
  R2 dose-response of the noise arms                          dose_response.csv
  R3 placement at an equal error rate, and counts per p       placement.csv
  R4 which statistic orders utility, and counts               metric_order.csv
  R5 real ranker pairs, agreement counts, contradictions      ranker_pairs.csv
  R6 the binary target's ceiling                              binary_ceiling.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

# --- the published grid and vocabularies (docs/error-location-plan.md) -------------------

TRACES = ("conversation_trace", "toolagent_trace")
ARMS = ("lru", "learned", "label", "offline", "pi3_next_use", "pi0_binary", "pi3_binary",
        "label_binary", "adm_label", "evict_label", "noise_0.5", "noise_1", "noise_2",
        "noise_4", "swap_uniform_0.25", "swap_uniform_0.5", "swap_runnerup_0.25",
        "swap_runnerup_0.5")
NOISE_ARMS = ("noise_0.5", "noise_1", "noise_2", "noise_4")
STATISTICS = ("m1", "m2", "m3", "m4")
ORIENTED = {"m1": "m1", "m2": "m2", "m3": "-m3", "m4": "-m4"}
SETS = ("primary", "secondary")
TARGETS = ("next_use", "binary")
LOCATIONS = ("admission_located", "eviction_located", "both", "neither")
READINGS = ("consistent_gain", "consistent_loss", "mixed")
LOCATION_TERMS = ("G", "A", "E", "interaction")
INPUTS = ("replay", "location", "dose_response", "placement", "metric_order", "ranker_pairs",
          "binary_ceiling")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published error-location directory (read only)")
    return parser.parse_args(argv)


# --- parsing and ordering -------------------------------------------------------------------


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
    """Target and p first where a table has them, then trace x cell, then arm,
    set and statistic."""
    order: tuple = ()
    if "target" in row:
        order += (_rank(TARGETS, row["target"]),)
    if "p" in row:
        order += (float(row["p"]),)
    key = cell_key(row)
    order += (_rank(TRACES, key[0]), key[1], key[2])
    for column, preferred in (("arm", ARMS), ("set", SETS), ("statistic", STATISTICS)):
        if column in row:
            order += (_rank(preferred, row[column]),)
    return order


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"])}


def _signs(row, prefix: str) -> tuple[int, int, int]:
    return tuple(int(row[f"{prefix}_n_{sign}"]) for sign in ("pos", "zero", "neg"))


def _cells_of(rows) -> int:
    return len({cell_key(row) for row in rows})


# --- R0: replay counters -----------------------------------------------------------------------

REPLAY_COLUMNS = ("extra_points_mean", "extra_points_ci95_half", "l2_rejections_mean",
                  "l2_evictions_mean", "m1_mean", "m2_mean", "m3_mean", "m4_mean",
                  "m2_admission_mean", "m2_resident_mean", "stat_decisions_resident_mean")


def replay_counters(rows) -> list[dict]:
    """R0: five-seed means per trace x cell x arm."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(arm=row["arm"], seeds=int(row["seeds"]))
        for column in REPLAY_COLUMNS:
            entry[column] = _float(row.get(column, ""))
        out.append(entry)
    return out


# --- R1: location ------------------------------------------------------------------------------


def location_table(rows) -> list[dict]:
    """R1: G, A, E and the interaction on the five-seed means with their seed
    signs, the shares of G, the label and the label on both traces."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for term in LOCATION_TERMS:
            entry[term] = float(row[f"{term}_points_mean"])
            entry[f"{term}_signs"] = _signs(row, term)
        entry["A_share_of_G"] = _float(row["A_share_of_G"])
        entry["E_share_of_G"] = _float(row["E_share_of_G"])
        entry["location"] = row["location"]
        entry["both_traces"] = row["location_both_traces"]
        entry["seed_locations"] = row["seed_locations"]
        out.append(entry)
    return out


def location_counts(rows) -> list[dict]:
    """R1 counts: per trace, how many cells carry each label, and how many
    cells carry it on both traces (each cell once; `traces_differ` and
    `single_trace` are the cells that carry no label on both)."""
    out = []
    traces = sorted({row["trace"] for row in rows}, key=lambda name: _rank(TRACES, name))
    for trace in traces:
        counts = Counter(row["location"] for row in rows if row["trace"] == trace)
        out.append({"scope": trace, **{label: counts.get(label, 0) for label in LOCATIONS},
                    "traces_differ": 0, "single_trace": 0, "cells": sum(counts.values())})
    cells = {}
    for row in rows:
        cells[(float(row["l1_fraction"]), float(row["l2_multiplier"]))] = row["location_both_traces"]
    counts = Counter(cells.values())
    out.append({"scope": "both traces",
                **{label: counts.get(label, 0) for label in LOCATIONS + ("traces_differ",
                                                                        "single_trace")},
                "cells": len(cells)})
    return out


# --- R2: dose-response -------------------------------------------------------------------------


def dose_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        for arm in NOISE_ARMS:
            entry[arm] = float(row[f"U_{arm}_points_mean"])
        for arm in ("lru", "learned", "label"):
            entry[arm] = float(row[f"U_{arm}_points_mean"])
        entry["non_increasing"] = _bool(row["non_increasing_means"])
        entry["seeds_non_increasing"] = int(row["seeds_non_increasing"])
        entry["seeds"] = int(row["seeds"])
        entry["crossing"] = row["crossing"]
        out.append(entry)
    return out


# --- R3: placement -----------------------------------------------------------------------------


def placement_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(p=float(row["p"]), diff=float(row["diff_points_mean"]),
                     signs=_signs(row, "diff"), reading=row["reading"],
                     m2_runnerup=float(row["m2_runnerup_mean"]),
                     m2_uniform=float(row["m2_uniform_mean"]))
        out.append(entry)
    return out


def placement_counts(rows) -> list[dict]:
    """R3 counts: per p, trace x cells per reading."""
    out = []
    for p in sorted({float(row["p"]) for row in rows}):
        members = [row for row in rows if float(row["p"]) == p]
        counts = Counter(row["reading"] for row in members)
        out.append({"p": p, **{reading: counts.get(reading, 0) for reading in READINGS},
                    "cells": len(members)})
    return out


# --- R4: metric order --------------------------------------------------------------------------


def metric_order_wide(rows) -> list[dict]:
    """R4: one row per trace x cell, rho of every statistic in both sets."""
    out: dict[tuple, dict] = {}
    for row in sorted(rows, key=_row_order):
        entry = out.setdefault(cell_key(row), _identity(row))
        entry[f"{row['set']}_{row['statistic']}"] = _float(row["rho"])
        entry[f"{row['set']}_{row['statistic']}_orders"] = _bool(row["orders_utility"])
        entry[f"{row['set']}_arms"] = int(row["arms"])
    return list(out.values())


def metric_order_counts(rows) -> list[dict]:
    """R4 counts: per set and statistic, trace x cells where rho >= 0.9."""
    out = []
    for set_name in SETS:
        for statistic in STATISTICS:
            members = [row for row in rows
                       if row["set"] == set_name and row["statistic"] == statistic]
            if not members:
                continue
            out.append({"set": set_name, "statistic": ORIENTED[statistic],
                        "orders": sum(1 for row in members if _bool(row["orders_utility"])),
                        "nan": sum(1 for row in members if math.isnan(_float(row["rho"]))),
                        "cells": len(members)})
    return out


# --- R5: ranker pairs --------------------------------------------------------------------------


def ranker_pair_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(target=row["target"], dU=float(row["dU_points_mean"]),
                     dU_signs=_signs(row, "dU"), reading=row["dU_reading"])
        for statistic in STATISTICS:
            entry[f"d_{statistic}"] = float(row[f"d_{statistic}_mean"])
            entry[f"{statistic}_agrees"] = _bool(row[f"{statistic}_agrees"])
            entry[f"{statistic}_contradicts"] = _bool(row[f"{statistic}_contradicts"])
        out.append(entry)
    return out


def ranker_pair_counts(rows) -> list[dict]:
    """R5 counts: per target and statistic, the cells where the statistic agrees."""
    out = []
    for target in TARGETS:
        members = [row for row in rows if row["target"] == target]
        if not members:
            continue
        readings = Counter(row["dU_reading"] for row in members)
        entry = {"target": target, "cells": len(members),
                 **{reading: readings.get(reading, 0) for reading in READINGS}}
        for statistic in STATISTICS:
            entry[f"{statistic}_agrees"] = sum(1 for row in members
                                               if _bool(row[f"{statistic}_agrees"]))
        out.append(entry)
    return out


def contradictions(rows) -> list[dict]:
    """R5 list: every cell where utility changes consistently while a statistic
    moves the other way."""
    out = []
    for row in sorted(rows, key=_row_order):
        for statistic in STATISTICS:
            if _bool(row[f"{statistic}_contradicts"]):
                out.append({**_identity(row), "target": row["target"],
                            "statistic": ORIENTED[statistic], "reading": row["dU_reading"],
                            "dU": float(row["dU_points_mean"]),
                            "d_statistic": float(row[f"d_{statistic}_mean"])})
    return out


# --- R6: binary ceiling ------------------------------------------------------------------------


def ceiling_table(rows) -> list[dict]:
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(diff=float(row["label_minus_label_binary_points_mean"]),
                     signs=_signs(row, "label_minus_label_binary"),
                     reading=row["label_minus_label_binary_reading"],
                     lru=float(row["U_lru_points_mean"]),
                     pi0_binary=float(row["U_pi0_binary_points_mean"]),
                     label_binary=float(row["U_label_binary_points_mean"]),
                     ratio=_float(row["ceiling_ratio"]))
        out.append(entry)
    return out


# --- printing ----------------------------------------------------------------------------------


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
               "stat_decisions_resident_mean": _count_mean}
    print_table("R0 replay counters and statistics (five-seed means; U in points)",
                IDENTITY + [_column("arm", "arm"), _column("seeds", "seeds")]
                + [_column(column.removesuffix("_mean"), column, formats.get(column, _number))
                   for column in REPLAY_COLUMNS], replay_counters(rows))


def print_location(rows) -> None:
    columns = list(IDENTITY)
    for term in LOCATION_TERMS:
        columns += [_column(term, term, _number), _column(f"{term} +/0/-", f"{term}_signs",
                                                          _sign_text)]
    columns += [_column("A/G", "A_share_of_G", _number), _column("E/G", "E_share_of_G", _number),
                _column("location", "location"), _column("both traces", "both_traces"),
                _column("seed labels", "seed_locations")]
    print_table("R1 location by decision type (five-seed means, points)", columns,
                location_table(rows),
                note="G = U(label) - U(learned), A = U(adm_label) - U(learned), E = "
                     "U(evict_label) - U(learned), interaction = A + E - G; a term reaches half "
                     "when it is >= 0.5 G. +/0/-: seed signs.")
    print_table("R1 location label counts", [_column("scope", "scope")]
                + [_column(label, label) for label in LOCATIONS + ("traces_differ",
                                                                   "single_trace")]
                + [_column("of", "cells")], location_counts(rows))


def print_dose(rows) -> None:
    print_table("R2 dose-response (five-seed means, points)",
                IDENTITY + [_column(arm, arm, _number) for arm in NOISE_ARMS]
                + [_column(f"U({arm})", arm, _number) for arm in ("learned", "label", "lru")]
                + [_column("non-increasing", "non_increasing"),
                   ("seeds non-increasing",
                    lambda row: f"{row['seeds_non_increasing']}/{row['seeds']}"),
                   _column("crosses U(learned)", "crossing")],
                dose_table(rows),
                note="Crossing: the adjacent noise levels between which the mean crosses "
                     "U(learned) (a level at or above it is on the upper side).")


def print_placement(rows) -> None:
    print_table("R3 placement at an equal error rate (runner-up minus uniform swap, points)",
                [_column("p", "p", lambda value: f"{value:g}")] + IDENTITY
                + [_column("diff", "diff", _number), _column("+/0/-", "signs", _sign_text),
                   _column("reading", "reading"),
                   _column("m2 runner-up", "m2_runnerup", _number),
                   _column("m2 uniform", "m2_uniform", _number)],
                placement_table(rows))
    print_table("R3 placement reading counts per p",
                [_column("p", "p", lambda value: f"{value:g}")]
                + [_column(reading, reading) for reading in READINGS] + [_column("of", "cells")],
                placement_counts(rows))


def print_metric_order(rows) -> None:
    columns = list(IDENTITY)
    for set_name in SETS:
        columns += [_column(f"{set_name} {ORIENTED[s]}", f"{set_name}_{s}", _number)
                    for s in STATISTICS]
    print_table("R4 Spearman rho over arms of the oriented statistic with U (five-seed means)",
                columns, metric_order_wide(rows),
                note="primary: the twelve single-key arms; secondary: all 18. nan: a constant side.")
    print_table("R4 cells where the statistic orders utility (rho >= 0.9)",
                [_column("set", "set"), _column("statistic", "statistic"),
                 _column("orders", "orders"), _column("nan", "nan"), _column("of", "cells")],
                metric_order_counts(rows))


def print_ranker_pairs(rows) -> None:
    columns = [_column("target", "target")] + IDENTITY + [
        _column("dU", "dU", _number), _column("+/0/-", "dU_signs", _sign_text),
        _column("reading", "reading")]
    for statistic in STATISTICS:
        columns += [_column(f"d {ORIENTED[statistic]}", f"d_{statistic}", _number),
                    _column("agrees", f"{statistic}_agrees")]
    print_table("R5 real ranker pairs, pi3 minus pi0 (seed-paired means; U in points)",
                columns, ranker_pair_table(rows),
                note="next_use: learned -> pi3_next_use; binary: pi0_binary -> pi3_binary. "
                     "d: the seed-paired mean change of the oriented statistic.")
    counts = ranker_pair_counts(rows)
    print_table("R5 agreement counts", [_column("target", "target"), _column("of", "cells")]
                + [_column(reading, reading) for reading in READINGS]
                + [_column(f"{ORIENTED[s]} agrees", f"{s}_agrees") for s in STATISTICS], counts)
    listed = contradictions(rows)
    if listed:
        print_table("R5 consistent utility change, statistic moving the other way",
                    [_column("target", "target")] + IDENTITY
                    + [_column("statistic", "statistic"), _column("reading", "reading"),
                       _column("dU", "dU", _number), _column("d statistic", "d_statistic",
                                                            _number)], listed)
    else:
        print("### R5 consistent utility change, statistic moving the other way\n\nnone\n")


def print_ceiling(rows) -> None:
    print_table("R6 the binary target's ceiling (five-seed means, points)",
                IDENTITY + [_column("U(label)-U(label_binary)", "diff", _number),
                            _column("+/0/-", "signs", _sign_text), _column("reading", "reading"),
                            _column("U(lru)", "lru", _number),
                            _column("U(pi0_binary)", "pi0_binary", _number),
                            _column("U(label_binary)", "label_binary", _number),
                            _column("ratio", "ratio", _number)],
                ceiling_table(rows),
                note="ratio = (U(pi0_binary) - U(lru)) / (U(label_binary) - U(lru)) on the "
                     "five-seed means; descriptive.")


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    print(f"# Error-location control tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_replay(tables["replay"])
    print_location(tables["location"])
    print_dose(tables["dose_response"])
    print_placement(tables["placement"])
    print_metric_order(tables["metric_order"])
    print_ranker_pairs(tables["ranker_pairs"])
    print_ceiling(tables["binary_ceiling"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
