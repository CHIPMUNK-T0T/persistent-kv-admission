#!/usr/bin/env python3
"""Tabulate what the all16 closure costs each arm in the Bailian external check, beside Mooncake.

Read-only, post hoc and descriptive: nothing here is a reading of the plan
(`docs/bailian-external-check-plan.md`), and nothing is written, fitted or
rerun. The script reads the published per-replay rows and prints Markdown
tables on stdout (stdlib `csv`; the plan's surface, h* table, arm names and
evaluable rule come from `persistent_kv_admission.bailiancheck`).

Bailian: PAPER_DIR/replay_seeds.csv on the primary window W. Mooncake (with
`--mooncake-root`): the published per-seed rows the runner reads for
`mooncake_order_beyond_bit.csv` (error_location_001, horizon_control_001,
horizon_fill_001, leaf_matched_horizon_001, variant main) on their full window;
the same replay in two directories must carry the same values.

"Points" are 100 x tokens / the window's requested tokens; every value is a
five-seed mean. Per trace x cell, with bit = label_binary_{h*} and (Bailian)
rand = label_binary_random_{h*}:

  pu_m(arm)        present-but-unusable points (Bailian W_l2_present_unusable_tokens,
                   Mooncake l2_present_unusable_tokens)
  U_m(arm)         Bailian U_W_points; Mooncake extra_avoided_tokens in points
  P_m(arm)         requested tokens present in L2 at request time, in points:
                   (l2_avoided_tokens + l2_present_unusable_tokens) per replay
  gain(arm)        U_leaf16(arm) - U_all16(arm), for label and bit
  Delta_m          U_m(label) - U_m(bit) (reading 7's Delta at h*)
  separation       Delta_leaf16 - Delta_all16 (= gain(label) - gain(bit))
  pu_gap           pu_all16(label) - pu_all16(bit)
  residual         separation - pu_gap
  D_rand           U_all16(rand) - U_all16(bit) (Bailian)
  pu_gap_rand      pu_all16(rand) - pu_all16(bit) (Bailian; printed negated)
  residual_rand    D_rand + pu_gap_rand (Bailian)
  evaluable_m      `bailiancheck.evaluable` on U_m(label) and U_m(lru) (Bailian only)

pu counts requested tokens at request time: blocks of the request held in L2
but past the first block found in neither tier (the tree hit rule stops
there; an ancestor is absent), in points of the window's input tokens. It is
not stranded capacity and not bytes x residence time.

The accounting identity. Per replay, the requested tokens present in L2 split
into usable and unusable, P = U + pu (U is exactly the L2-avoided tokens in
every published row). Hence, exactly, on the five-seed means:

  residual      = [P_leaf16(label) - P_all16(label)] - [P_leaf16(bit) - P_all16(bit)]
                  - [pu_leaf16(label) - pu_leaf16(bit)]
  residual_rand = P_all16(rand) - P_all16(bit)

The last term of the first line is the measured leaf16 pu gap (zero when
leaf16 leaves no token unusable, as in every published leaf16 replay).
Part of corr(separation, pu_gap) is therefore this accounting relation: the
residual is what the shift in tokens present in L2 adds beyond pu_gap.

Before anything is printed the computed values are checked against the
published tables, and the script exits non-zero naming every mismatch:
Delta_all16 and Delta_leaf16 at role h_star and both evaluable flags against
order_beyond_bit.csv (W), D_rand against random.csv (all16, W) and, with
Mooncake, both Deltas against mooncake_order_beyond_bit.csv (full); values to
within 1e-9 absolute, flags exactly, and the same trace x cell on both sides.
Both identities above are checked the same way, with P read from the
l2_avoided_tokens column rather than built from U. A missing file, column,
arm or seed, or conflicting Mooncake rows, is an error.

Each computation is a pure function of parsed rows (tested in
`tests/test_tabulate_bailian_closure_cost.py`).

  B1 Bailian per trace x cell (W)        replay_seeds.csv
  B2 Bailian summary                     the B1 rows
  B3 Bailian per trace                   the B1 rows
  M1 Mooncake per trace x cell (full)    MOONCAKE_ROOT/<four directories>/replay_seeds.csv
  M2 Mooncake summary                    the M1 rows
  M3 Mooncake per trace                  the M1 rows
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from persistent_kv_admission import bailiancheck as bc  # noqa: E402
from persistent_kv_admission.horizonctl import ALL16, LEAF16  # noqa: E402

MECHANISMS = (ALL16, LEAF16)
MOONCAKE_TRACES = ("conversation_trace", "toolagent_trace")
# The runner's published Mooncake per-seed sources (`MOONCAKE_SOURCES` of
# scripts/run_bailian_external_check.py), by directory under MOONCAKE_ROOT.
MOONCAKE_DIRS = ("error_location_001", "horizon_control_001", "horizon_fill_001",
                 "leaf_matched_horizon_001")
VARIANT = "main"
TOLERANCE = 1e-9
# The ratio separation / pu_gap is reported over the cells with at least this pu_gap.
MIN_PU_GAP_POINTS = 0.5

IDENTITY_COLUMNS = ("trace", "l1_fraction", "l2_multiplier", "mechanism", "arm", "seed")
BAILIAN_COLUMNS = IDENTITY_COLUMNS + ("W_l2_present_unusable_tokens", "W_l2_avoided_tokens",
                                      "W_requested_tokens", "U_W_points")
MOONCAKE_COLUMNS = IDENTITY_COLUMNS + ("l2_present_unusable_tokens", "l2_avoided_tokens",
                                       "requested_tokens", "extra_avoided_tokens")
CELL_COLUMNS = ("trace", "l1_fraction", "l2_multiplier")
RANDOM_COLUMNS = CELL_COLUMNS + ("mechanism", "D_rand_points_mean_W")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", type=Path,
                        help="published bailian_external_check directory (read only)")
    parser.add_argument("--mooncake-root", type=Path, default=None,
                        help="directory holding the published Mooncake result directories "
                             "(results/paper; read only); without it no Mooncake table is printed")
    return parser.parse_args(argv)


# --- parsing --------------------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV; a ValueError naming the path when it is not a file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} not found")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows, columns: Sequence[str], source: str) -> None:
    """A ValueError when `rows` is empty or lacks any of `columns`."""
    if not rows:
        raise ValueError(f"{source}: no rows")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{source}: missing column(s) {', '.join(missing)}")


def _int(row, column: str, source: str) -> int:
    try:
        return int(row[column])
    except (TypeError, ValueError):
        raise ValueError(f"{source}: {column} = {row[column]!r} is not an integer") from None


def _finite(row, column: str, source: str) -> float:
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{source}: {column} = {row[column]!r} is not a finite number")
    return value


def _bool(value) -> bool | None:
    text = str(value).strip().lower()
    return True if text == "true" else False if text == "false" else None


def _cell_key(row) -> tuple[str, float, float]:
    return row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])


def _key(row) -> tuple[str, float, float, str, str]:
    return _cell_key(row) + (row["mechanism"], row["arm"])


def cell_label(fraction: float, multiplier: float) -> str:
    """The published `cell` text, `l1=<f>,l2x<k>`."""
    return f"l1={fraction:g},l2x{multiplier:g}"


def _where(key) -> str:
    trace, fraction, multiplier, mechanism, arm = key
    return f"{trace} {cell_label(fraction, multiplier)} {mechanism} {arm}"


# --- the replays ----------------------------------------------------------------------------------


def _present(row, prefix: str, source: str) -> tuple[float, float]:
    """(pu, P) of one replay in points: present-but-unusable tokens, and every
    requested token present in L2 at request time (L2-avoided + unusable)."""
    requested = _int(row, f"{prefix}requested_tokens", source)
    unusable = _int(row, f"{prefix}l2_present_unusable_tokens", source)
    return (bc.points(unusable, requested),
            bc.points(_int(row, f"{prefix}l2_avoided_tokens", source) + unusable, requested))


def bailian_replays(rows, source: str = "replay_seeds.csv") -> dict[tuple, dict[int, tuple]]:
    """(trace, fraction, multiplier, mechanism, arm) -> seed -> (pu, U, P) in
    points on W, from the Bailian replay_seeds.csv rows. Heap references are
    skipped; a replay listed twice is an error."""
    require_columns(rows, BAILIAN_COLUMNS, source)
    out: dict[tuple, dict[int, tuple]] = {}
    for row in rows:
        if row["mechanism"] == bc.HEAP:
            continue
        key, seed = _key(row), _int(row, "seed", source)
        by_seed = out.setdefault(key, {})
        if seed in by_seed:
            raise ValueError(f"{source}: {_where(key)} seed {seed} listed twice")
        unusable, present = _present(row, "W_", source)
        by_seed[seed] = (unusable, _finite(row, "U_W_points", source), present)
    return out


def mooncake_replays(sources: Mapping[str, list[dict]]) -> dict[tuple, dict[int, tuple]]:
    """(trace, fraction, multiplier, mechanism, arm) -> seed -> (pu, U, P) in
    points on the full window, from the published per-seed rows of each
    source (name -> rows; variant main only, like the runner). The same
    replay found twice must carry the same pu, U and P (within TOLERANCE);
    every conflict is named in one ValueError."""
    out: dict[tuple, dict[int, tuple]] = {}
    origin: dict[tuple, str] = {}
    conflicts = []
    for source, rows in sources.items():
        require_columns(rows, MOONCAKE_COLUMNS, source)
        for row in rows:
            if row.get("variant", VARIANT) != VARIANT:
                continue
            key, seed = _key(row), _int(row, "seed", source)
            unusable, present = _present(row, "", source)
            value = (unusable, bc.points(_int(row, "extra_avoided_tokens", source),
                                         _int(row, "requested_tokens", source)), present)
            by_seed = out.setdefault(key, {})
            if seed not in by_seed:
                by_seed[seed] = value
                origin[key + (seed,)] = source
            elif any(abs(old - new) > TOLERANCE for old, new in zip(by_seed[seed], value)):
                conflicts.append(f"{_where(key)} seed {seed}: (pu, U, P) {by_seed[seed]} in "
                                 f"{origin[key + (seed,)]}, {value} in {source}")
    if conflicts:
        raise ValueError("conflicting Mooncake rows:\n  " + "\n  ".join(conflicts))
    return out


def seed_means(replays, key: tuple, seeds: Sequence[int]) -> tuple[float, float, float]:
    """The means of (pu, U, P) of one arm over `seeds` (`bailiancheck.mean`);
    an error when the arm is missing or its seeds are not exactly `seeds`."""
    by_seed = replays.get(key)
    if not by_seed:
        raise ValueError(f"missing arm: {_where(key)}")
    if set(by_seed) != set(seeds):
        raise ValueError(f"{_where(key)}: seeds {sorted(by_seed)}, expected {sorted(seeds)}")
    return tuple(bc.mean(by_seed[seed][index] for seed in seeds) for index in range(3))


# --- per trace x cell -----------------------------------------------------------------------------


def cell_quantities(replays, trace: str, fraction: float, multiplier: float,
                    seeds: Sequence[int], bailian: bool = True) -> dict:
    """The closure-cost quantities of one trace x cell (module docstring).
    `bailian` adds lru (the evaluable flags) and the random arm (D_rand);
    Mooncake has neither in this comparison. `residual_from_P` and
    `residual_rand_from_P` are the right-hand sides of the accounting identity,
    built from P (`check_identities` compares them with the residuals)."""
    fraction, multiplier = float(fraction), float(multiplier)
    arms = {"label": bc.LABEL, "bit": bc.transplant_arm(fraction, multiplier)}
    if bailian:
        arms.update(lru=bc.LRU, rand=bc.cell_random_arm(fraction, multiplier))
    pu, u, p = {}, {}, {}
    for mechanism in MECHANISMS:
        for role, arm in arms.items():
            if role == "rand" and mechanism != ALL16:
                continue
            pu[mechanism, role], u[mechanism, role], p[mechanism, role] = seed_means(
                replays, (trace, fraction, multiplier, mechanism, arm), seeds)
    leaf16 = [value[0] for key, by_seed in replays.items()
              if key[:4] == (trace, fraction, multiplier, LEAF16) for value in by_seed.values()]
    delta = {mechanism: u[mechanism, "label"] - u[mechanism, "bit"] for mechanism in MECHANISMS}
    row: dict = {"trace": trace, "l1_fraction": fraction, "l2_multiplier": multiplier,
                 "cell": cell_label(fraction, multiplier),
                 "h_star": bc.hstar(fraction, multiplier)}
    if bailian:
        row.update({f"evaluable_{mechanism}": bc.evaluable(u[mechanism, "label"],
                                                           u[mechanism, "lru"])
                    for mechanism in MECHANISMS})
    row.update(pu_all16_label=pu[ALL16, "label"], pu_all16_bit=pu[ALL16, "bit"])
    if bailian:
        row["pu_all16_rand"] = pu[ALL16, "rand"]
    row.update(pu_leaf16_max=max(leaf16),
               gain_label=u[LEAF16, "label"] - u[ALL16, "label"],
               gain_bit=u[LEAF16, "bit"] - u[ALL16, "bit"],
               Delta_all16=delta[ALL16], Delta_leaf16=delta[LEAF16],
               separation=delta[LEAF16] - delta[ALL16],
               pu_gap=pu[ALL16, "label"] - pu[ALL16, "bit"],
               pu_gap_leaf16=pu[LEAF16, "label"] - pu[LEAF16, "bit"])
    row["residual"] = row["separation"] - row["pu_gap"]
    row["residual_from_P"] = ((p[LEAF16, "label"] - p[ALL16, "label"])
                              - (p[LEAF16, "bit"] - p[ALL16, "bit"]) - row["pu_gap_leaf16"])
    if bailian:
        row.update(D_rand=u[ALL16, "rand"] - u[ALL16, "bit"],
                   pu_gap_rand=pu[ALL16, "rand"] - pu[ALL16, "bit"])
        row["residual_rand"] = row["D_rand"] + row["pu_gap_rand"]
        row["residual_rand_from_P"] = p[ALL16, "rand"] - p[ALL16, "bit"]
    return row


def closure_table(replays, traces: Sequence[str], cells: Sequence[tuple[float, float]],
                  seeds: Sequence[int], bailian: bool = True) -> list[dict]:
    """`cell_quantities` for every trace x cell, traces then cells in the given order."""
    return [cell_quantities(replays, trace, fraction, multiplier, seeds, bailian)
            for trace in traces for fraction, multiplier in cells]


# --- summary --------------------------------------------------------------------------------------


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Pearson's r; nan with fewer than two pairs or a constant side."""
    xs, ys = [float(x) for x in xs], [float(y) for y in ys]
    if len(xs) != len(ys):
        raise ValueError(f"pearson needs pairs: {len(xs)} x vs {len(ys)} y")
    if len(xs) < 2 or min(xs) == max(xs) or min(ys) == max(ys):
        return math.nan
    mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    syy = math.fsum((y - mean_y) ** 2 for y in ys)
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0.0 or syy == 0.0:
        return math.nan
    return sxy / math.sqrt(sxx * syy)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.fsum(values) / len(values) if values else math.nan


def _at(row) -> str:
    return "" if row is None else f"{row['trace']} {row['cell']}"


def population(rows, bailian: bool = True) -> tuple[list[dict], str]:
    """The cells corr(separation, pu_gap) is taken over, and their description:
    those evaluable under both mechanisms (Bailian), every cell (Mooncake)."""
    if bailian:
        return ([row for row in rows if row["evaluable_all16"] and row["evaluable_leaf16"]],
                "evaluable under all16 and leaf16")
    return list(rows), "every cell (no evaluable rule)"


def residual_lines(rows, column: str, over: str) -> list[dict]:
    """Summary lines of `column` over `rows`: mean, mean |.| and max |.|
    (the last naming its trace x cell)."""
    values = [row[column] for row in rows]
    top = max(rows, key=lambda row: abs(row[column]), default=None)
    return [{"quantity": f"mean {column}", "over": over, "cells": len(rows),
             "value": _mean(values), "at": ""},
            {"quantity": f"mean |{column}|", "over": over, "cells": len(rows),
             "value": _mean(abs(value) for value in values), "at": ""},
            {"quantity": f"max |{column}|", "over": over, "cells": len(rows),
             "value": math.nan if top is None else abs(top[column]), "at": _at(top)}]


def summary(rows, bailian: bool = True) -> list[dict]:
    """The summary lines of one workload's `closure_table`: the cell counts,
    corr(separation, pu_gap) over the cells evaluable under both mechanisms
    (Mooncake: every cell), the min and max of separation / pu_gap over those
    with pu_gap >= 0.5 point, the residual's mean, mean |.| and max |.| over
    the same cells, corr(D_rand, -pu_gap_rand) and the same three lines of
    residual_rand over the cells evaluable under all16 (Bailian), and the
    largest leaf16 pu."""
    pool, over = population(rows, bailian)
    gapped = [row for row in pool if row["pu_gap"] >= MIN_PU_GAP_POINTS]
    ratios = {id(row): row["separation"] / row["pu_gap"] for row in gapped}
    low = min(gapped, key=lambda row: ratios[id(row)], default=None)
    high = max(gapped, key=lambda row: ratios[id(row)], default=None)
    gapped_over = f"{over}, pu_gap >= {MIN_PU_GAP_POINTS:g}"
    out = [{"quantity": "trace x cell", "over": "all", "cells": len(rows), "value": "", "at": ""},
           {"quantity": "corr(separation, pu_gap)", "over": over, "cells": len(pool),
            "value": pearson([row["separation"] for row in pool], [row["pu_gap"] for row in pool]),
            "at": ""},
           {"quantity": "min separation / pu_gap", "over": gapped_over, "cells": len(gapped),
            "value": math.nan if low is None else ratios[id(low)], "at": _at(low)},
           {"quantity": "max separation / pu_gap", "over": gapped_over, "cells": len(gapped),
            "value": math.nan if high is None else ratios[id(high)], "at": _at(high)}]
    out += residual_lines(pool, "residual", over)
    if bailian:
        random_pool = [row for row in rows if row["evaluable_all16"]]
        out.append({"quantity": "corr(D_rand, -pu_gap_rand)", "over": "evaluable under all16",
                    "cells": len(random_pool),
                    "value": pearson([row["D_rand"] for row in random_pool],
                                     [-row["pu_gap_rand"] for row in random_pool]), "at": ""})
        out += residual_lines(random_pool, "residual_rand", "evaluable under all16")
    out.append({"quantity": "max pu over every leaf16 replay", "over": "all", "cells": len(rows),
                "value": max((row["pu_leaf16_max"] for row in rows), default=math.nan),
                "at": ""})
    return out


def per_trace(rows, bailian: bool = True) -> list[dict]:
    """Per trace, over its cells in the correlation's population: their count,
    corr(separation, pu_gap) (nan with fewer than two cells or a constant
    side), the residual's mean, mean |.| and max |.|, and the range of pu_gap
    (a narrow range leaves a correlation without meaning)."""
    pool, _ = population(rows, bailian)
    out = []
    for trace in dict.fromkeys(row["trace"] for row in rows):
        members = [row for row in pool if row["trace"] == trace]
        residuals = [row["residual"] for row in members]
        gaps = [row["pu_gap"] for row in members]
        out.append({"trace": trace, "cells": len(members),
                    "corr_separation_pu_gap": pearson([row["separation"] for row in members],
                                                      gaps),
                    "mean_residual": _mean(residuals),
                    "mean_abs_residual": _mean(abs(value) for value in residuals),
                    "max_abs_residual": max((abs(value) for value in residuals),
                                            default=math.nan),
                    "pu_gap_min": min(gaps, default=math.nan),
                    "pu_gap_max": max(gaps, default=math.nan)})
    return out


# --- the accounting identity ----------------------------------------------------------------------

IDENTITIES = (("residual", "residual_from_P"), ("residual_rand", "residual_rand_from_P"))


def check_identities(rows) -> None:
    """The accounting identity on every row of a `closure_table`: residual
    equals residual_from_P and (Bailian) residual_rand equals
    residual_rand_from_P, within TOLERANCE. P is read from the l2_avoided
    column, so this also checks that U is the L2-avoided tokens. One
    ValueError names every trace x cell that is off."""
    problems = []
    for row in rows:
        for left, right in IDENTITIES:
            if left in row and not abs(row[left] - row[right]) <= TOLERANCE:
                problems.append(f"{_at(row)} {left} {row[left]!r}, from P {row[right]!r}")
    if problems:
        raise ValueError("the accounting identity (P = U + pu) does not hold:\n  "
                         + "\n  ".join(problems))


# --- consistency with the published tables --------------------------------------------------------


def _matched(rows, published, source: str) -> tuple[list[tuple], list[str]]:
    """Pairs (computed row, published row) by trace x cell, and a problem for
    every trace x cell on one side only or repeated in `published`."""
    by_cell: dict[tuple, list[dict]] = {}
    for entry in published:
        by_cell.setdefault(_cell_key(entry), []).append(entry)
    pairs, problems = [], []
    computed = {(row["trace"], row["l1_fraction"], row["l2_multiplier"]): row for row in rows}
    for cell, row in computed.items():
        entries = by_cell.get(cell, [])
        if len(entries) != 1:
            problems.append(f"{source}: {len(entries)} rows for {cell[0]} "
                            f"{cell_label(*cell[1:])}, expected 1")
        else:
            pairs.append((row, entries[0]))
    for cell in sorted(set(by_cell) - set(computed)):
        problems.append(f"{source}: {cell[0]} {cell_label(*cell[1:])} is not in the computed table")
    return pairs, problems


def _compare(problems: list[str], source: str, row, name: str, computed: float,
             published_text: str) -> None:
    try:
        published = float(published_text)
    except (TypeError, ValueError):
        published = math.nan
    if not (math.isfinite(published) and abs(computed - published) <= TOLERANCE):
        problems.append(f"{source}: {row['trace']} {row['cell']} {name} computed {computed!r}, "
                        f"published {published_text!r}")


def check_order(rows, order_rows, window: str, flags: bool = True,
                source: str = "order_beyond_bit.csv") -> None:
    """Delta_all16 and Delta_leaf16 of every computed row against the
    published role-h_star five-seed means on `window` (within TOLERANCE),
    and, with `flags`, evaluable_all16 / evaluable_leaf16 exactly. One
    ValueError names every problem, including a trace x cell on one side only."""
    columns = [f"Delta_{mechanism}_points_mean_{window}" for mechanism in MECHANISMS]
    if flags:
        columns += [f"evaluable_{mechanism}" for mechanism in MECHANISMS]
    require_columns(order_rows, CELL_COLUMNS + ("role",) + tuple(columns), source)
    pairs, problems = _matched(rows, [entry for entry in order_rows if entry["role"] == "h_star"],
                               source)
    for row, entry in pairs:
        for mechanism in MECHANISMS:
            _compare(problems, source, row, f"Delta_{mechanism}", row[f"Delta_{mechanism}"],
                     entry[f"Delta_{mechanism}_points_mean_{window}"])
            if flags and _bool(entry[f"evaluable_{mechanism}"]) != row[f"evaluable_{mechanism}"]:
                problems.append(f"{source}: {row['trace']} {row['cell']} evaluable_{mechanism} "
                                f"computed {row[f'evaluable_{mechanism}']}, published "
                                f"{entry[f'evaluable_{mechanism}']!r}")
    if problems:
        raise ValueError("the computed values do not match the published table:\n  "
                         + "\n  ".join(problems))


def check_random(rows, random_rows, source: str = "random.csv") -> None:
    """D_rand of every computed row against the published all16 five-seed
    mean on W (within TOLERANCE); one ValueError names every problem."""
    require_columns(random_rows, RANDOM_COLUMNS, source)
    pairs, problems = _matched(rows, [entry for entry in random_rows
                                      if entry["mechanism"] == ALL16], source)
    for row, entry in pairs:
        _compare(problems, source, row, "D_rand", row["D_rand"], entry["D_rand_points_mean_W"])
    if problems:
        raise ValueError("the computed values do not match the published table:\n  "
                         + "\n  ".join(problems))


# --- printing -------------------------------------------------------------------------------------


def _number(value) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def print_table(title: str, rows, columns=None, note: str = "") -> None:
    print(f"### {title}\n")
    if note:
        print(f"{note}\n")
    if not rows:
        print("(no rows)\n")
        return
    columns = columns or list(rows[0])
    body = [[_number(row[column]) if isinstance(row[column], float) else str(row[column])
             for column in columns] for row in rows]
    widths = [max(len(text) for text in column) for column in zip(columns, *body)]
    print("| " + " | ".join(text.ljust(width) for text, width in zip(columns, widths)) + " |")
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for cells in body:
        print("| " + " | ".join(text.ljust(width) for text, width in zip(cells, widths)) + " |")
    print()


# Kept in the rows (checked, tested) but not printed: the identity's
# right-hand sides equal the residuals, and the leaf16 pu gap is zero wherever
# pu_leaf16_max is.
HIDDEN = ("l1_fraction", "l2_multiplier", "pu_gap_leaf16", "residual_from_P",
          "residual_rand_from_P")


def display(rows) -> list[dict]:
    """The printed columns of a `closure_table`: h* in seconds, and
    pu_gap_rand negated in place (the sign it is correlated with)."""
    out = []
    for row in rows:
        entry: dict = {}
        for key, value in row.items():
            if key in HIDDEN:
                continue
            if key == "pu_gap_rand":
                entry["-pu_gap_rand"] = -value
            else:
                entry[key] = f"{value:g}" if key == "h_star" else value
        out.append(entry)
    return out


PU_NOTE = ("pu counts requested tokens at request time that are present in L2 but unusable "
           "because an ancestor is absent (blocks past the first block found in neither tier; "
           "the tree hit rule stops there), in points of the window's input tokens; it is not "
           "stranded capacity and not bytes x residence time. Per replay the requested tokens "
           "present in L2, P, split as P = U + pu, so residual = separation - pu_gap = "
           "[P_leaf16(label) - P_all16(label)] - [P_leaf16(bit) - P_all16(bit)] - "
           "[pu_leaf16(label) - pu_leaf16(bit)] (the last term is zero wherever pu_leaf16_max "
           "is), checked before printing with P read from the l2_avoided column.")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bailian = closure_table(bailian_replays(read_csv(args.paper_dir / "replay_seeds.csv")),
                                bc.TRACES, bc.CELLS, bc.SEEDS)
        check_identities(bailian)
        check_order(bailian, read_csv(args.paper_dir / "order_beyond_bit.csv"), "W")
        check_random(bailian, read_csv(args.paper_dir / "random.csv"))
        mooncake = None
        if args.mooncake_root is not None:
            sources = {f"{name}/replay_seeds.csv":
                       read_csv(args.mooncake_root / name / "replay_seeds.csv")
                       for name in MOONCAKE_DIRS}
            mooncake = closure_table(mooncake_replays(sources), MOONCAKE_TRACES, bc.CELLS,
                                     bc.SEEDS, bailian=False)
            check_identities(mooncake)
            check_order(mooncake, read_csv(args.paper_dir / "mooncake_order_beyond_bit.csv"),
                        "full", flags=False, source="mooncake_order_beyond_bit.csv")
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None
    print(f"# Closure cost of all16, post hoc (descriptive)\n\nSource: {args.paper_dir}"
          f" (replay_seeds.csv); Mooncake: "
          + (", ".join(f"{args.mooncake_root / name}/replay_seeds.csv" for name in MOONCAKE_DIRS)
             if mooncake is not None else "not read (no --mooncake-root)") + ".\n")
    print("Checked before printing: Delta_all16 and Delta_leaf16 at h* and both evaluable flags "
          "equal order_beyond_bit.csv (W), D_rand equals random.csv (all16, W)"
          + (", and the Mooncake Deltas equal mooncake_order_beyond_bit.csv (full)"
             if mooncake is not None else "")
          + "; residual and residual_rand equal their right-hand sides built from P (the "
            "accounting identity)"
          + f"; to within {TOLERANCE:g} absolute, at every trace x cell.\n")
    print_table("B1 Bailian, primary window W: closure cost per trace x cell (five-seed means, "
                "points)", display(bailian),
                note="bit = label_binary_{h*}, rand = label_binary_random_{h*}. pu = present-but-"
                     "unusable points under all16; pu_leaf16_max over every leaf16 replay of the "
                     "cell. gain = U_leaf16 - U_all16; Delta_m = U_m(label) - U_m(bit); "
                     "separation = Delta_leaf16 - Delta_all16; pu_gap = pu(label) - pu(bit); "
                     "residual = separation - pu_gap; D_rand = U_all16(rand) - U_all16(bit); "
                     "-pu_gap_rand = pu(bit) - pu(rand); residual_rand = D_rand + pu_gap_rand = "
                     "P_all16(rand) - P_all16(bit). evaluable: five-seed mean U(label) - U(lru) "
                     ">= 1.0 point on W.\n\n" + PU_NOTE)
    print_table("B2 Bailian summary", summary(bailian))
    print_table("B3 Bailian per trace (cells evaluable under all16 and leaf16)",
                per_trace(bailian),
                note="corr is nan with fewer than two cells or a constant side; pu_gap_min and "
                     "pu_gap_max show the range a correlation would rest on.")
    if mooncake is not None:
        print_table("M1 Mooncake, full window: closure cost per trace x cell (five-seed means, "
                    "points)", display(mooncake),
                    note="Published rows, full window; no evaluable rule is applied (all 12 "
                         "trace x cell are listed and counted); no random arm. Columns and the "
                         "identity as in B1.")
        print_table("M2 Mooncake summary", summary(mooncake, bailian=False))
        print_table("M3 Mooncake per trace (every cell)", per_trace(mooncake, bailian=False),
                    note="As B3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
