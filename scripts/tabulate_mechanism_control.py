#!/usr/bin/env python3
"""Tabulate the published mechanism-control tables for the findings document.

Read-only. The script reads the CSVs that `scripts/run_mechanism_control.py`
published in PAPER_DIR (and, for T9 only, the published Phase 0.97 per-seed
replay table given with --references) and prints every number the findings
document quotes, as Markdown tables on stdout. Nothing is written, fitted,
rerun or re-derived from the replays: each section is arithmetic on the
published columns, so a number in the document can be checked against this
output and this output against the CSVs.

Each section is one pure function from parsed CSV rows to a list of dict rows
(tested in `tests/test_tabulate_mechanism_control.py`) and a small printer.
Points are printed to 3 decimals and percents to 1. Traces are in grid order,
cells by (l1_fraction, l2_multiplier) ascending, mechanisms all16, leaf16,
all64, leaf64 and rungs lru, learned, label, offline.

  T1 decomposition per trace x cell x mechanism       decomposition.csv
  T2 dominant-label counts per mechanism              decomposition.csv
  T3 recovery ratios on the five-seed means           decomposition.csv
  T4 mechanism effects per contrast x rung            mechanism_effects.csv
  T5 seed-paired term changes against all16           decomposition_seeds.csv
  T6 ladder order, inversions and the pre-registered  ladder.csv,
     inversion reading, applied literally             ladder_inversions.csv
  T7 capacity-pattern reading counts                  capacity_pattern.csv
  T8 replay counters                                  replay.csv
  T9 leaf64 against the heap offline reference, next  decomposition_seeds.csv,
     to the heap comparator's present-unusable tokens --references
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# --- the published grid and vocabularies (docs/mechanism-control-plan.md) ------------

TRACES = ("conversation_trace", "toolagent_trace")
MECHANISMS = ("all16", "leaf16", "all64", "leaf64")
BASE_MECHANISM = "all16"
LEAF_MECHANISMS = ("leaf16", "leaf64")
# The mechanism of reading 4: the learned arm once the mechanism is controlled.
CONTROLLED_MECHANISM = "leaf64"
RUNGS = ("lru", "learned", "label", "offline")
GAPS = ("candidate_search", "objective", "signal", "achieved")
# Levels of T1 and T3, every one a `*_points_mean` column of decomposition.csv.
LEVELS = ("H_off", "H_lru", "U_lru", "U_learned", "U_label", "U_offline", "T")
PAIRS = ("lru_le_learned", "learned_le_label", "label_le_offline")
CONTRASTS = ("leaf16_minus_all16", "all64_minus_all16", "leaf64_minus_leaf16",
             "leaf64_minus_all16")
READINGS = ("consistent_gain", "consistent_loss", "mixed")
DOMINANT_VALUES = ("mechanism_bound", "objective_bound", "signal_bound", "achieved", "mixed")
# The heap offline comparator of the published Phase 0.97 table.
HEAP_KIND = "heap"
HEAP_OFFLINE_POLICY = "offline_next_use"
INPUTS = ("decomposition", "decomposition_seeds", "mechanism_effects", "ladder",
          "ladder_inversions", "capacity_pattern", "replay")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published mechanism-control directory (read only)")
    parser.add_argument("--references", type=Path, default=None,
                        help="published Phase 0.97 per-seed replay table, for the heap "
                             "present-unusable column of T9; without it that column is "
                             "reported unavailable")
    return parser.parse_args(argv)


# --- parsing and ordering -----------------------------------------------------------------


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


def _rank(preferred, value) -> tuple:
    return (preferred.index(value), "") if value in preferred else (len(preferred), value)


def _ordered(values, preferred) -> list:
    """The distinct values, the known ones in their fixed order, any other after them."""
    return sorted(set(values), key=lambda value: _rank(preferred, value))


def cell_key(row) -> tuple:
    """(trace, l1_fraction, l2_multiplier), the fractions compared as floats."""
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def _cell_order(key) -> tuple:
    return (_rank(TRACES, key[0]), key[1], key[2])


def _row_order(row) -> tuple:
    order = _cell_order(cell_key(row))
    if "mechanism" in row:
        order += (_rank(MECHANISMS, row["mechanism"]),)
    if "rung" in row:
        order += (_rank(RUNGS, row["rung"]),)
    if "seed" in row:
        order += (int(row["seed"]),)
    return order


def _cells(rows) -> list[tuple]:
    """Distinct (trace, l1_fraction, l2_multiplier, cell label) in table order."""
    labels = {}
    for row in rows:
        labels.setdefault(cell_key(row), row["cell"])
    return [key + (labels[key],) for key in sorted(labels, key=_cell_order)]


def _identity(row) -> dict:
    return {"trace": row["trace"], "cell": row["cell"],
            "l1_fraction": float(row["l1_fraction"]),
            "l2_multiplier": float(row["l2_multiplier"])}


def _mean(values) -> float:
    return statistics.fmean(values) if values else math.nan


def _signs(values) -> tuple[int, int, int]:
    return (sum(1 for value in values if value > 0), sum(1 for value in values if value == 0),
            sum(1 for value in values if value < 0))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else math.nan


# --- T1-T3: the decomposition ------------------------------------------------------------


def decomposition_table(rows) -> list[dict]:
    """T1: levels, the four terms with their share of mean T and seed signs, the
    dominant label and whether the five per-seed labels agree."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry["mechanism"] = row["mechanism"]
        for level in LEVELS:
            entry[level] = float(row[f"{level}_points_mean"])
        for gap in GAPS:
            entry[gap] = float(row[f"{gap}_points_mean"])
            entry[f"{gap}_share"] = float(row[f"{gap}_share_of_mean_T"])
            entry[f"{gap}_signs"] = tuple(int(row[f"{gap}_seeds_{sign}"])
                                          for sign in ("pos", "zero", "neg"))
        labels = row["seed_dominant"].split("|")
        entry["dominant"] = row["dominant"]
        entry["seed_labels"] = len(labels)
        entry["seed_unanimous"] = len(set(labels)) == 1
        entry["seed_label"] = labels[0] if entry["seed_unanimous"] else "split"
        entry["identity_exact_all_seeds"] = _bool(row["identity_exact_all_seeds"])
        out.append(entry)
    return out


def dominant_counts(rows) -> list[dict]:
    """T2: per mechanism, the dominant labels over its trace x cell rows and how
    many rows have one label in every seed (and that label the mean's)."""
    table = decomposition_table(rows)
    out = []
    for mechanism in _ordered((entry["mechanism"] for entry in table), MECHANISMS):
        members = [entry for entry in table if entry["mechanism"] == mechanism]
        counts = Counter(entry["dominant"] for entry in members)
        entry = {"mechanism": mechanism, "rows": len(members)}
        for label in DOMINANT_VALUES:
            entry[label] = counts.get(label, 0)
        entry["other"] = sum(n for label, n in counts.items() if label not in DOMINANT_VALUES)
        entry["seed_unanimous"] = sum(1 for member in members if member["seed_unanimous"])
        entry["seed_unanimous_as_mean"] = sum(
            1 for member in members
            if member["seed_unanimous"] and member["seed_label"] == member["dominant"])
        out.append(entry)
    return out


def recovery_ratios(rows) -> list[dict]:
    """T3: ratios and differences of the five-seed means."""
    out = []
    for entry in decomposition_table(rows):
        out.append({
            **{key: entry[key] for key in ("trace", "cell", "l1_fraction", "l2_multiplier",
                                           "mechanism")},
            "label_over_H_off": _ratio(entry["U_label"], entry["H_off"]),
            "offline_over_H_off": _ratio(entry["U_offline"], entry["H_off"]),
            "learned_share_of_label": _ratio(entry["U_learned"] - entry["U_lru"],
                                             entry["U_label"] - entry["U_lru"]),
            "U_lru_minus_H_lru": entry["U_lru"] - entry["H_lru"],
            "U_learned_minus_H_lru": entry["U_learned"] - entry["H_lru"],
        })
    return out


# --- T4: mechanism effects ---------------------------------------------------------------


def mechanism_effects_summary(rows) -> list[dict]:
    """T4: per contrast x rung, the reading counts, the range of the cell means
    and every cell mean in the fixed trace x cell order."""
    cells = _cells(rows)
    groups: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for row in rows:
        groups[(row["contrast"], row["rung"])][cell_key(row)] = row
    out = []
    for contrast in _ordered((row["contrast"] for row in rows), CONTRASTS):
        for rung in _ordered((row["rung"] for row in rows), RUNGS):
            members = groups.get((contrast, rung))
            if not members:
                continue
            readings = Counter(row["reading"] for row in members.values())
            means = [float(row["diff_points_mean"]) for row in members.values()]
            entry = {"contrast": contrast, "rung": rung, "rows": len(members)}
            for reading in READINGS:
                entry[reading] = readings.get(reading, 0)
            entry["other"] = sum(n for reading, n in readings.items() if reading not in READINGS)
            entry["min"] = min(means)
            entry["max"] = max(means)
            entry["cells"] = []
            for cell in cells:
                row = members.get(cell[:3])
                entry["cells"].append({
                    "trace": cell[0], "cell": cell[3],
                    "mean": float(row["diff_points_mean"]) if row else math.nan,
                    "reading": row["reading"] if row else ""})
            out.append(entry)
    return out


# --- T5: term changes across mechanisms --------------------------------------------------


def term_changes(seed_rows, base: str = BASE_MECHANISM,
                 others=tuple(m for m in MECHANISMS if m != BASE_MECHANISM)) -> list[dict]:
    """T5: per term, mechanism and trace x cell, the seed-paired difference
    `term_points(m) - term_points(base)` over the seeds both have, and the base
    mechanism's mean of the term over its own seeds."""
    index: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in seed_rows:
        index[cell_key(row) + (row["mechanism"],)][int(row["seed"])] = row
    cells = _cells(seed_rows)
    out = []
    for term in GAPS:
        for mechanism in others:
            for cell in cells:
                before = index.get(cell[:3] + (base,), {})
                after = index.get(cell[:3] + (mechanism,), {})
                seeds = sorted(set(before) & set(after))
                differences = [float(after[seed][f"{term}_points"])
                               - float(before[seed][f"{term}_points"]) for seed in seeds]
                out.append({
                    "term": term, "mechanism": mechanism, "trace": cell[0], "cell": cell[3],
                    "l1_fraction": cell[1], "l2_multiplier": cell[2],
                    "base": _mean([float(row[f"{term}_points"]) for row in before.values()]),
                    "base_seeds": len(before), "seeds": len(seeds),
                    "diff_mean": _mean(differences), "signs": _signs(differences)})
    return out


# --- T6: the ladder ------------------------------------------------------------------------


def ladder_order(ladder_rows) -> list[dict]:
    """T6a: per mechanism, how many trace x cell are ordered in every seed, and which."""
    out = []
    for mechanism in _ordered((row["mechanism"] for row in ladder_rows), MECHANISMS):
        members = sorted((row for row in ladder_rows if row["mechanism"] == mechanism),
                         key=_row_order)
        ordered = [f"{row['trace']}/{row['cell']}" for row in members if _bool(row["ordered"])]
        out.append({"mechanism": mechanism, "ordered": len(ordered), "rows": len(members),
                    "ordered_cells": ordered})
    return out


def inversion_counts(inversion_rows) -> list[dict]:
    """T6b: inversion rows (pairs failing in at least one seed) per pair and mechanism."""
    counts = Counter((row["pair"], row["mechanism"]) for row in inversion_rows)
    out = []
    for pair in _ordered(list(PAIRS) + [row["pair"] for row in inversion_rows], PAIRS):
        entry = {"pair": pair}
        for mechanism in _ordered(list(MECHANISMS) + [row["mechanism"] for row in inversion_rows],
                                  MECHANISMS):
            entry[mechanism] = counts.get((pair, mechanism), 0)
        entry["total"] = sum(n for (name, _), n in counts.items() if name == pair)
        out.append(entry)
    return out


def inversion_presence(ladder_rows, inversion_rows) -> list[dict]:
    """T6c: for every cell x pair x trace, the inversion row of each mechanism
    (None where the pair holds in every seed)."""
    rows = list(ladder_rows) + list(inversion_rows)
    traces = _ordered((row["trace"] for row in rows), TRACES)
    cells: dict[tuple, str] = {}
    for row in rows:
        cells.setdefault((float(row["l1_fraction"]), float(row["l2_multiplier"])), row["cell"])
    index = {cell_key(row) + (row["mechanism"], row["pair"]): row for row in inversion_rows}
    out = []
    for fraction, multiplier in sorted(cells):
        for pair in _ordered(list(PAIRS) + [row["pair"] for row in inversion_rows], PAIRS):
            for trace in traces:
                entry = {"trace": trace, "cell": cells[(fraction, multiplier)],
                         "l1_fraction": fraction, "l2_multiplier": multiplier, "pair": pair}
                for mechanism in MECHANISMS:
                    row = index.get((trace, fraction, multiplier, mechanism, pair))
                    entry[mechanism] = None if row is None else {
                        "seeds_inverted": int(row["seeds_inverted"]), "seeds": int(row["seeds"]),
                        "mean_diff_points": float(row["mean_diff_points"])}
                entry["present"] = tuple(m for m in MECHANISMS if entry[m] is not None)
                out.append(entry)
    return out


def classify_inversions(presence_rows) -> list[dict]:
    """T6d: the pre-registered reading 3, applied literally per cell x pair. An
    inversion is present under a mechanism when it fails in at least one seed.
    `mechanism_borne`: on every trace (both) present under all16 and absent
    under at least one of leaf16 / leaf64. `score_property`: on every trace
    present under all four mechanisms. Anything else is `unclassified`."""
    traces = _ordered((row["trace"] for row in presence_rows), TRACES)
    groups: dict[tuple, dict[str, set]] = defaultdict(dict)
    labels: dict[tuple, str] = {}
    for row in presence_rows:
        key = (row["l1_fraction"], row["l2_multiplier"], row["pair"])
        groups[key][row["trace"]] = set(row["present"])
        labels.setdefault(key, row["cell"])
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], _rank(PAIRS, k[2]))):
        present = {trace: groups[key].get(trace, set()) for trace in traces}
        borne = bool(traces) and all(
            BASE_MECHANISM in present[trace]
            and any(m not in present[trace] for m in LEAF_MECHANISMS) for trace in traces)
        score = bool(traces) and all(set(MECHANISMS) <= present[trace] for trace in traces)
        out.append({
            "cell": labels[key], "l1_fraction": key[0], "l2_multiplier": key[1], "pair": key[2],
            "classification": ("mechanism_borne" if borne else
                               "score_property" if score else "unclassified"),
            "traces_inverted": sum(1 for trace in traces if present[trace]),
            "present": {trace: tuple(m for m in MECHANISMS if m in present[trace])
                        for trace in traces},
            "absent": {trace: tuple(m for m in MECHANISMS if m not in present[trace])
                       for trace in traces}})
    return out


# --- T7, T8: capacity pattern and replay counters ----------------------------------------


def capacity_summary(rows) -> list[dict]:
    """T7: per mechanism, the reading counts of both differences and their mean ranges."""
    out = []
    for mechanism in _ordered((row["mechanism"] for row in rows), MECHANISMS):
        members = [row for row in rows if row["mechanism"] == mechanism]
        entry = {"mechanism": mechanism, "rows": len(members)}
        for name in ("learned_minus_lru", "label_minus_lru"):
            readings = Counter(row[f"{name}_reading"] for row in members)
            for reading in READINGS:
                entry[f"{name}_{reading}"] = readings.get(reading, 0)
            entry[f"{name}_other"] = sum(n for reading, n in readings.items()
                                         if reading not in READINGS)
            means = [float(row[f"{name}_mean_points"]) for row in members]
            entry[f"{name}_min"] = min(means) if means else math.nan
            entry[f"{name}_max"] = max(means) if means else math.nan
        out.append(entry)
    return out


REPLAY_COUNTERS = ("extra_points_mean", "extra_points_ci95_half", "present_unusable_points_mean",
                   "l2_rejections_mean", "l2_evictions_mean", "absent_rejected_points_mean",
                   "absent_evicted_points_mean")


def replay_counters(rows) -> list[dict]:
    """T8: the five-seed means of the replay counters per trace x cell x mechanism x rung."""
    out = []
    for row in sorted(rows, key=_row_order):
        entry = _identity(row)
        entry.update(mechanism=row["mechanism"], rung=row["rung"], seeds=int(row["seeds"]))
        for column in REPLAY_COUNTERS:
            entry[column] = float(row[column])
        out.append(entry)
    return out


# --- T9: the controlled mechanism against the heap reference -----------------------------


def heap_present_unusable(reference_rows) -> dict[tuple, list[float]]:
    """(trace, l1_fraction, l2_multiplier) -> present-but-unusable points of each
    published heap offline row (100 x tokens / requested tokens)."""
    out: dict[tuple, list[float]] = defaultdict(list)
    for row in reference_rows:
        if row["kind"] == HEAP_KIND and row["l2_policy"] == HEAP_OFFLINE_POLICY:
            out[cell_key(row)].append(_ratio(100.0 * float(row["l2_present_unusable_tokens"]),
                                             float(row["requested_tokens"])))
    return dict(out)


def controlled_versus_heap(seed_rows, reference_rows=None,
                           mechanism: str = CONTROLLED_MECHANISM) -> list[dict]:
    """T9: per trace x cell, `U_offline - H_off` and `U_label - H_off` of one
    mechanism, paired within each seed row, next to the heap comparator's
    present-but-unusable points (None for `reference_rows` = unavailable)."""
    members_of: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        if row["mechanism"] == mechanism:
            members_of[cell_key(row)].append(row)
    heap = None if reference_rows is None else heap_present_unusable(reference_rows)
    out = []
    for cell in _cells(row for row in seed_rows if row["mechanism"] == mechanism):
        members = sorted(members_of[cell[:3]], key=lambda row: int(row["seed"]))
        entry = {"trace": cell[0], "cell": cell[3], "l1_fraction": cell[1],
                 "l2_multiplier": cell[2], "mechanism": mechanism, "seeds": len(members)}
        for rung in ("offline", "label"):
            differences = [float(row[f"U_{rung}_points"]) - float(row["H_off_points"])
                           for row in members]
            entry[f"{rung}_minus_H_off"] = _mean(differences)
            entry[f"{rung}_minus_H_off_signs"] = _signs(differences)
        entry["heap_available"] = heap is not None
        values = [] if heap is None else heap.get(cell[:3], [])
        entry["heap_rows"] = len(values)
        entry["heap_present_unusable_mean"] = _mean(values)
        entry["heap_present_unusable_min"] = min(values) if values else math.nan
        entry["heap_present_unusable_max"] = max(values) if values else math.nan
        out.append(entry)
    return out


# --- printing ----------------------------------------------------------------------------


def _points(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _percent(value) -> str:
    """A share (fraction) as a percent."""
    return "nan" if math.isnan(value) else f"{100.0 * value:.1f}"


def _count_mean(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.1f}"


def _sign_text(signs) -> str:
    return "/".join(str(n) for n in signs)


def _short(trace: str) -> str:
    return trace.removesuffix("_trace")


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


def print_decomposition(rows) -> None:
    table = decomposition_table(rows)
    print_table("T1a decomposition: levels (five-seed means, points)",
                IDENTITY + [_column("mech", "mechanism")]
                + [_column(level, level, _points) for level in LEVELS], table)
    columns = IDENTITY + [_column("mech", "mechanism")]
    for gap in GAPS:
        columns += [_column(gap, gap, _points), _column(f"{gap} %T", f"{gap}_share", _percent),
                    _column(f"{gap} +/0/-", f"{gap}_signs", _sign_text)]
    columns += [_column("dominant", "dominant"), _column("seeds unanimous", "seed_unanimous"),
                _column("seed label", "seed_label"),
                _column("identity exact", "identity_exact_all_seeds")]
    print_table("T1b decomposition: terms of T (points, % of mean T, seed signs +/0/-)",
                columns, table)


def print_dominant_counts(rows) -> None:
    print_table("T2 dominant label counts",
                [_column("mech", "mechanism"), _column("rows", "rows")]
                + [_column(label, label) for label in DOMINANT_VALUES]
                + [_column("other", "other"), _column("seed-unanimous", "seed_unanimous"),
                   _column("seed-unanimous = mean label", "seed_unanimous_as_mean")],
                dominant_counts(rows))


def print_recovery_ratios(rows) -> None:
    print_table("T3 recovery ratios (five-seed means; ratios in %, differences in points)",
                IDENTITY + [_column("mech", "mechanism"),
                            _column("U_label/H_off %", "label_over_H_off", _percent),
                            _column("U_offline/H_off %", "offline_over_H_off", _percent),
                            _column("(U_learned-U_lru)/(U_label-U_lru) %",
                                    "learned_share_of_label", _percent),
                            _column("U_lru-H_lru", "U_lru_minus_H_lru", _points),
                            _column("U_learned-H_lru", "U_learned_minus_H_lru", _points)],
                recovery_ratios(rows))


def _marked(value: dict) -> str:
    text = _points(value["mean"])
    return text if value["reading"] in ("consistent_gain", "consistent_loss") else "~" + text


def print_mechanism_effects(rows) -> None:
    table = mechanism_effects_summary(rows)
    columns = [_column("contrast", "contrast"), _column("rung", "rung"), _column("rows", "rows")]
    columns += [_column(reading, reading) for reading in READINGS]
    columns += [_column("other", "other"), _column("min", "min", _points),
                _column("max", "max", _points)]
    for index, cell in enumerate(_cells(rows)):
        header = f"{_short(cell[0])[:4]} {100 * cell[1]:g}%x{cell[2]:g}"
        columns.append((header, lambda row, i=index: _marked(row["cells"][i])))
    print_table("T4 mechanism effects (seed-paired U differences, points)", columns, table,
                note="Per-cell columns: five-seed mean of the difference; `~` marks a mixed "
                     "(not seed-consistent) reading. Cell header: trace, L1 fraction, L2 multiplier.")


def term_changes_wide(rows) -> list[dict]:
    """T5 rows pivoted to one row per term x trace x cell, one column per mechanism."""
    out: dict[tuple, dict] = {}
    for row in rows:
        key = (row["term"], row["trace"], row["l1_fraction"], row["l2_multiplier"])
        entry = out.setdefault(key, {"term": row["term"], "trace": row["trace"],
                                     "cell": row["cell"], "base": row["base"], "mechanisms": {}})
        entry["mechanisms"][row["mechanism"]] = row
    return list(out.values())


def print_term_changes(seed_rows) -> None:
    rows = term_changes(seed_rows)
    mechanisms = _ordered((row["mechanism"] for row in rows), MECHANISMS)
    columns = [_column("term", "term")] + IDENTITY + [
        _column(f"base {BASE_MECHANISM}", "base", _points)]
    for mechanism in mechanisms:
        columns.append((f"{mechanism}-{BASE_MECHANISM}",
                        lambda row, m=mechanism: _points(row["mechanisms"][m]["diff_mean"])
                        if m in row["mechanisms"] else "-"))
        columns.append((f"{mechanism} +/0/-",
                        lambda row, m=mechanism: _sign_text(row["mechanisms"][m]["signs"])
                        if m in row["mechanisms"] else "-"))
    print_table("T5 term changes across mechanisms (seed-paired, points)", columns,
                term_changes_wide(rows),
                note=f"`m-{BASE_MECHANISM}`: five-seed mean of term_points(m) - "
                     f"term_points({BASE_MECHANISM}) paired by seed; +/0/- its seed signs; "
                     f"base: the {BASE_MECHANISM} five-seed mean of the term.")


def _inversion_text(value) -> str:
    if value is None:
        return "-"
    return f"{value['seeds_inverted']}/{value['seeds']} ({_points(value['mean_diff_points'])})"


def _mechanism_list(mechanisms) -> str:
    return ",".join(mechanisms) if mechanisms else "none"


def print_ladder(ladder_rows, inversion_rows) -> None:
    print_table("T6a ladder order (lru <= learned <= label <= offline in every seed)",
                [_column("mech", "mechanism"), _column("ordered", "ordered"),
                 _column("of", "rows"),
                 ("ordered trace/cell", lambda row: "; ".join(row["ordered_cells"]) or "none")],
                ladder_order(ladder_rows))
    counts = inversion_counts(inversion_rows)
    mechanisms = [key for key in counts[0] if key not in ("pair", "total")] if counts else []
    print_table("T6b inversion rows by pair (a pair failing in >= 1 seed)",
                [_column("pair", "pair")] + [_column(m, m) for m in mechanisms]
                + [_column("total", "total")], counts)
    presence = inversion_presence(ladder_rows, inversion_rows)
    print_table("T6c inversion presence per cell x pair x trace",
                [_column("cell", "cell"), _column("pair", "pair"), _column("trace", "trace")]
                + [(m, lambda row, m=m: _inversion_text(row[m])) for m in MECHANISMS],
                presence,
                note="Entry: seeds inverted / seeds (mean diff points, upper minus lower rung); "
                     "`-` = no inversion row (the pair holds in every seed).")
    classified = classify_inversions(presence)
    traces = _ordered((row["trace"] for row in presence), TRACES)
    columns = [_column("cell", "cell"), _column("pair", "pair"),
               _column("classification", "classification"),
               _column("traces inverted", "traces_inverted")]
    for trace in traces:
        columns.append((f"present under ({_short(trace)})",
                        lambda row, t=trace: _mechanism_list(row["present"][t])))
    columns.append(("absent under, mechanism_borne only",
                    lambda row: "; ".join(f"{_short(t)}: {_mechanism_list(row['absent'][t])}"
                                          for t in traces)
                    if row["classification"] == "mechanism_borne" else "-"))
    print_table("T6d inversion reading (pre-registered rule, applied literally)", columns,
                classified,
                note="Present = an inversion row exists (fails in >= 1 seed). mechanism_borne: "
                     "on both traces present under all16 and absent under leaf16 or leaf64; "
                     "score_property: on both traces present under all four mechanisms; "
                     "otherwise unclassified. `traces inverted`: traces with any inversion row.")


def print_capacity(rows) -> None:
    columns = [_column("mech", "mechanism"), _column("rows", "rows")]
    for name in ("learned_minus_lru", "label_minus_lru"):
        columns += [_column(f"{name} {reading}", f"{name}_{reading}")
                    for reading in READINGS + ("other",)]
        columns += [_column(f"{name} min", f"{name}_min", _points),
                    _column(f"{name} max", f"{name}_max", _points)]
    print_table("T7 capacity pattern (reading counts; min/max of the cell means, points)",
                columns, capacity_summary(rows))


def print_replay(rows) -> None:
    formats = {"l2_rejections_mean": _count_mean, "l2_evictions_mean": _count_mean}
    print_table("T8 replay counters (five-seed means; points unless a count)",
                IDENTITY + [_column("mech", "mechanism"), _column("rung", "rung"),
                            _column("seeds", "seeds")]
                + [_column(column, column, formats.get(column, _points))
                   for column in REPLAY_COUNTERS],
                replay_counters(rows))


def print_controlled(seed_rows, reference_rows, reference_note: str) -> None:
    rows = controlled_versus_heap(seed_rows, reference_rows)
    columns = IDENTITY + [_column("seeds", "seeds")]
    for rung in ("offline", "label"):
        columns += [_column(f"U_{rung}({CONTROLLED_MECHANISM})-H_off", f"{rung}_minus_H_off",
                            _points),
                    _column("+/0/-", f"{rung}_minus_H_off_signs", _sign_text)]
    if reference_rows is None:
        columns.append(("heap offline present-unusable", lambda row: "unavailable"))
    else:
        columns += [_column("heap offline present-unusable", "heap_present_unusable_mean",
                            _points),
                    _column("heap rows", "heap_rows"),
                    _column("heap min", "heap_present_unusable_min", _points),
                    _column("heap max", "heap_present_unusable_max", _points)]
    print_table(f"T9 {CONTROLLED_MECHANISM} versus the heap reference (points)", columns, rows,
                note="Differences paired within each seed row of decomposition_seeds.csv. Heap "
                     "present-unusable: 100 x l2_present_unusable_tokens / requested_tokens of "
                     f"the published kind={HEAP_KIND}, l2_policy={HEAP_OFFLINE_POLICY} rows, mean "
                     f"over the matched rows. References: {reference_note}")


def load_references(path) -> tuple[list[dict] | None, str]:
    if path is None:
        return None, "not given (--references); heap column unavailable"
    if not Path(path).is_file():
        return None, f"{path} does not exist; heap column unavailable"
    return read_csv(Path(path)), str(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = {}
    for name in INPUTS:
        path = args.paper_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        tables[name] = read_csv(path)
    reference_rows, reference_note = load_references(args.references)
    print(f"# Mechanism control tables\n\nSource: {args.paper_dir} "
          f"({', '.join(f'{name}.csv: {len(rows)} rows' for name, rows in tables.items())}).\n")
    print_decomposition(tables["decomposition"])
    print_dominant_counts(tables["decomposition"])
    print_recovery_ratios(tables["decomposition"])
    print_mechanism_effects(tables["mechanism_effects"])
    print_term_changes(tables["decomposition_seeds"])
    print_ladder(tables["ladder"], tables["ladder_inversions"])
    print_capacity(tables["capacity_pattern"])
    print_replay(tables["replay"])
    print_controlled(tables["decomposition_seeds"], reference_rows, reference_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
