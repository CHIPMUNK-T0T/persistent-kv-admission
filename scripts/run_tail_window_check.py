#!/usr/bin/env python3
"""Evaluation-window check: do the matched-horizon readings hold when the last 600 seconds are excluded.

The pre-registration is `docs/tail-window-check-plan.md`; nothing here may
change it. With traces, capacities, seeds, mechanism (`all16`) and the matched
horizon `h*` of each cell unchanged, the seven arms behind the matched-horizon
readings are rerun, each built by the module that built its published rows
(`persistent_kv_admission.tailwindow.arm_setup`):

* `lru`, `learned`, `label`, `evict_label` (`errorloc`), published in
  `results/paper/error_location_001/`;
* `label_binary_{h*}` (`horizonctl` at 60, 300 and 600 s, published in
  `results/paper/horizon_control_001/`; `horizonfill` at 150 s, published in
  `results/paper/horizon_fill_001/`);
* `evict_binary_{h*}_learned`, `evict_binary_{h*}_recency` (`matchedorder`),
  published in `results/paper/matched_class_order_001/`.

2 traces x 6 cells x 7 arms x 5 seeds = 420 replays, each with the Phase
0.98b attribution hooks and the error-location decision statistics attached
read-only, and the class statistics at `h*` for the two class-order arms, as
their parent runs attached them. Each replay also carries
`onpolicy.LabelWindowUtilityCollector(trace, split_ms, end_ms - 600,000)`
beside the attribution collector, through `tailwindow.ChainedRequestHook`
(new code; no existing replay path changes). The head window's counters are
the collector's; the tail's are the full window's minus the head's.

The replay call and row, the provenance rules and the table helpers are the
matched class-order runner's (`scripts/run_matched_class_order.py`, imported
by path, unchanged, with the horizon-control, error-location,
mechanism-control and Phase 0.97 runners it brings); the frozen ranker is
loaded exactly as there.

Before anything is derived or published the run checks that every one of the
420 replays reproduces its published row in `avoided_prefill_tokens`,
`counters_sha256` and `decision_sha256`; that the identifiers of every replay
equal those of every published reference row of its trace x cell x seed; that
the head counters are at most the full ones in every counter and that the head
`measured_requests`, `requested_tokens` and `l1_avoided_tokens` are equal
across the seven arms of a trace x cell x seed; that the statistics saw every
decision with its final victim and the arms without an override never
overrode; that no class-order replay evicts a state reusable within `h*` while
a sampled resident is not; that the arm-independent counters are
arm-independent; that no absent token is unexplained; and (per replay) the
attribution identities. After the derivation it checks that the full-window
readings, recomputed from the reproduced rows, equal the published values
(`S_h*` in `horizon_control_001/horizon.csv` or `horizon_fill_001/fill.csv`;
`R_h*`, the order and the admission differences in
`matched_class_order_001/`). Any failure exits without writing a derived
table, and nothing reaches the paper directory.

Outputs (in --output-dir and in --paper-dir): per-seed and aggregated replay
tables with full, head and tail counters, the published reference rows used,
U per window, the five readings per window, the reproduction table, the
reading counts, a README describing every table, and the run configuration.
There is no smoke mode and no dirty-tree mode: the plan runs once, from a
clean tree at the code commit.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np

from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.crossworkload import HYPERPARAMETERS
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import ALL16, HORIZON_READINGS, MECHANISMS
from persistent_kv_admission.matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_READINGS,
    ORDER_OF_ARM,
    SIGN_READINGS,
    ClassStatistics,
    matched_horizon,
    sign_reading_counts,
)
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.tailwindow import (
    ARMS,
    CLASS_ORDER_SHARE,
    ERROR_LOCATION_ARMS,
    COUNTERS,
    DIFFERENCES,
    HEAD_INVARIANT_COUNTERS,
    MECHANISM,
    PREDICTED_BELOW_CELL,
    RUNG_ROLE,
    RUNG_SOURCES,
    SLUGS,
    SOURCES,
    SUFFICES_SHORTFALL,
    TAIL_SECONDS,
    WINDOWS,
    ChainedRequestHook,
    arm_horizon,
    arm_role,
    arm_setup,
    cell_arms,
    class_order_prediction_holds,
    class_order_reading,
    counters_of,
    full_row,
    head_collector,
    head_row,
    horizon_prediction_holds,
    is_class_order_arm,
    predicted_head_class_reading,
    recovers,
    reference_source,
    shortfall_reading,
    sign_agrees,
    sign_prediction_holds,
    suffices,
    tail_row,
    tail_share,
    window_bounds,
    window_columns,
    window_difference,
    window_recovery,
    window_shortfall,
    window_utility,
)
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

MATCHED_SCRIPT = REPOSITORY / "scripts/run_matched_class_order.py"
PLAN_PATH = REPOSITORY / "docs/tail-window-check-plan.md"


def _load_matched_class_order():
    """Import the matched class-order runner by path, unchanged, for its
    replay constants, provenance, reference and check helpers; it brings the
    horizon-control, error-location, mechanism-control and Phase 0.97 runners
    with it (one instance of each, so the capacity rule reads the working set
    this run sets, and the ranker loader is the one it uses). `main` is behind
    the usual guard, so nothing runs, and nothing is read at import."""
    spec = importlib.util.spec_from_file_location("run_matched_class_order", MATCHED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mco = _load_matched_class_order()
hc = mco.hc
rel = mco.rel
rmc = mco.rmc
rdp = mco.rdp

# --- the fixed grid (docs/tail-window-check-plan.md) ----------------------------------------------

TRACES = mco.TRACES
CELLS = mco.CELLS
SEEDS = mco.SEEDS
assert set(CELLS) == set(MATCHED_HORIZON_SECONDS)
LABEL_HORIZON_SECONDS = mco.LABEL_HORIZON_SECONDS
assert LABEL_HORIZON_SECONDS == 600.0 and MECHANISM == ALL16 == mco.MECHANISM
ELIGIBILITY, WIDTH = MECHANISMS[MECHANISM]
# Every replay carries the statistics hooks; there is no hook-off variant here.
VARIANT = "main"
DEFAULT_WORKERS = mco.DEFAULT_WORKERS
MAX_WORKERS = mco.MAX_WORKERS
assert (DEFAULT_WORKERS, MAX_WORKERS) == (10, 12)
# The plan's publication directory (a new one; refused if it exists).
DEFAULT_PAPER_DIR = REPOSITORY / "results/paper/tail_window_check_001"
# Published references, anchored at the repository (tracked files): every arm's
# rows, each from the run that published them.
MATCHED_REFERENCE = REPOSITORY / "results/paper/matched_class_order_001/replay_seeds.csv"
REFERENCE_SOURCES = {"error_location_001": mco.ERROR_LOCATION_REFERENCE,
                     "horizon_control_001": mco.HORIZON_CONTROL_REFERENCE,
                     "horizon_fill_001": mco.HORIZON_FILL_REFERENCE,
                     "matched_class_order_001": MATCHED_REFERENCE}
assert tuple(REFERENCE_SOURCES) == SOURCES
assert RUNG_SOURCES == mco.RUNG_SOURCES
# Every arm replayed, by (mechanism, arm), and the source of its published row.
REFERENCES = {(MECHANISM, arm): reference_source(arm)
              for cell in CELLS for arm in cell_arms(*cell)}
REFERENCE_IDENTIFIERS = hc.REFERENCE_IDENTIFIERS
# The published readings the full window must equal (read, never rerun).
PUBLISHED_READING_SOURCES = {
    "horizon": REPOSITORY / "results/paper/horizon_control_001/horizon.csv",
    "fill": REPOSITORY / "results/paper/horizon_fill_001/fill.csv",
    "class_order": REPOSITORY / "results/paper/matched_class_order_001/class_order.csv",
    "order": REPOSITORY / "results/paper/matched_class_order_001/order.csv",
    "admission": REPOSITORY / "results/paper/matched_class_order_001/admission.csv",
}
# The table that holds S_h* of each label-rung source.
SHORTFALL_TABLES = {"horizon_control_001": "horizon", "horizon_fill_001": "fill"}
# The reproduction compares these exactly; `decision_sha256` where the
# published row carries one (all four sources do).
REPRODUCTION_COLUMNS = ("avoided_prefill_tokens", "counters_sha256")
REPRODUCTION_OPTIONAL_COLUMNS = ("decision_sha256",)
REPLAY_METRICS = hc.REPLAY_METRICS
CLASS_METRICS = mco.CLASS_METRICS
WINDOW_METRICS = (tuple(f"U_{window}_points" for window in WINDOWS)
                  + tuple(f"{window}_extra_avoided_tokens" for window in WINDOWS))
# Arms whose victim is the store's first minimum: they must never override.
NO_OVERRIDE_ROLES = ("lru", "learned", "label", RUNG_ROLE)
ARMS_PER_CELL = len(ARMS)
POINT_FIELDS = ("mean", "ci95_half", "min", "max")
SHARED: dict[str, object] = {}


def reference_cells(mechanism: str, arm: str) -> tuple[tuple[float, float], ...]:
    """The cells of the grid at which a reference is read: those whose seven
    arms include it (a horizon-specific arm at the cells whose h* is its
    horizon, the four error-location arms at every cell)."""
    if (mechanism, arm) not in REFERENCES:
        raise ValueError(f"unknown reference {arm}/{mechanism}")
    return tuple(cell for cell in CELLS if arm in cell_arms(*cell))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path,
                        help="the two Mooncake trace files (conversation, tool-agent)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new run directory; refused if it exists")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR,
                        help="new publication directory; refused if it exists; written only "
                             "when every check passes (default "
                             f"{DEFAULT_PAPER_DIR.relative_to(REPOSITORY)})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"replay processes (default {DEFAULT_WORKERS}, at most {MAX_WORKERS})")
    return parser.parse_args(argv)


validate_arguments = mco.validate_arguments


# --- provenance ------------------------------------------------------------------------------------


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository: the matched
    class-order runner's (every `src` module and the five runners it chains)
    and this script."""
    paths = set(mco.execution_sources())
    paths.add(Path(__file__).resolve())
    return sorted(paths)


def source_manifest() -> dict[str, object]:
    files = {str(path.relative_to(REPOSITORY)): sha256_path(path) for path in execution_sources()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def sources_differing_from_head() -> list[str]:
    """Execution files whose working copy is not byte-identical to HEAD's."""
    differing = []
    for path in execution_sources():
        relative = str(path.relative_to(REPOSITORY))
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPOSITORY,
                                   capture_output=True)
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            differing.append(relative)
    return differing


_repository_path = mco._repository_path


# --- inputs ----------------------------------------------------------------------------------------

_cell_seed = hc._cell_seed
_reference_entry = mco._reference_entry


def load_references(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> published row`
    for every arm of `REFERENCES`, each read from its own source only, at the
    cells `reference_cells` names, with the eligibility and width of `all16`
    and the digests the row carries; each must be there exactly once for every
    trace and seed of the grid. Rows of other arms, sources, cells or seeds
    are skipped unchecked (the matched class-order runner's loader with this
    run's references). `paths` overrides the source files (tests)."""
    paths = dict(REFERENCE_SOURCES, **(paths or {}))
    wanted = {key: set(reference_cells(*key)) for key in REFERENCES}
    published: dict[tuple, dict] = {}
    problems: list[str] = []
    for source, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["mechanism"], row["arm"])
                if REFERENCES.get(key) != source or row.get("variant", "main") != VARIANT:
                    continue
                cell_seed = _cell_seed(row)
                if (cell_seed[0] not in TRACES or cell_seed[1:3] not in wanted[key]
                        or cell_seed[3] not in SEEDS):
                    continue
                if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
                    problems.append(f"{source} row {key[1]}/{key[0]} has "
                                    f"{row['eligibility']}/{row['width']}")
                full = key + cell_seed
                if full in published:
                    problems.append(f"duplicate published reference {full}")
                    continue
                published[full] = _reference_entry(row, key[0], key[1], source)
    for key, source in REFERENCES.items():
        for name in TRACES:
            for fraction, multiplier in reference_cells(*key):
                for seed in SEEDS:
                    full = key + (name, fraction, multiplier, seed)
                    if full not in published:
                        problems.append(f"published {source} reference {full} missing")
                    elif any(column not in published[full] for column in REPRODUCTION_COLUMNS):
                        problems.append(f"published {source} reference {full} has no "
                                        + " / ".join(column for column in REPRODUCTION_COLUMNS
                                                     if column not in published[full]))
    return published, problems


def _cell(row) -> tuple:
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def _published_float(value) -> float:
    return math.nan if value in ("", None) else float(value)


def load_published_readings(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(trace, fraction, multiplier) -> published full-window readings` of
    every trace x cell of the grid: `S` at the cell's h* (from the horizon
    control's `horizon.csv` at 60, 300 and 600 s, the fill-in's `fill.csv` at
    150 s), and from the matched class-order run `R_matched_learned`,
    `R_matched_recency` and the reading (`class_order.csv`), and the mean,
    seed signs and reading of the order (`order.csv`) and admission
    (`admission.csv`) differences. Each must be there exactly once; rows of
    other horizons or cells are skipped. `paths` overrides the files (tests)."""
    paths = dict(PUBLISHED_READING_SOURCES, **(paths or {}))
    grid = {(name, fraction, multiplier) for name in TRACES for fraction, multiplier in CELLS}
    readings: dict[tuple, dict] = defaultdict(dict)
    problems: list[str] = []

    def put(cell, column, value, table):
        if column in readings[cell]:
            problems.append(f"duplicate published {column} of {cell} in {table}")
            return
        readings[cell][column] = value

    for table, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cell = _cell(row)
                if cell not in grid:
                    continue
                h = matched_horizon(cell[1], cell[2])
                if table in ("horizon", "fill"):
                    if (float(row["h"]) != h
                            or SHORTFALL_TABLES[RUNG_SOURCES[h]] != table):
                        continue
                    put(cell, "S", _published_float(row["S"]), table)
                    continue
                if float(row["h_star"]) != h:
                    problems.append(f"{table} row {cell} has h* {row['h_star']}, the plan's is "
                                    f"{h:g}")
                if table == "class_order":
                    put(cell, "R_matched_learned", _published_float(row["R_matched_learned"]),
                        table)
                    put(cell, "R_matched_recency", _published_float(row["R_matched_recency"]),
                        table)
                    put(cell, "class_order_reading", row["reading"], table)
                elif table in ("order", "admission"):
                    prefix = ("learned_minus_recency" if table == "order"
                              else "label_binary_minus_recency")
                    put(cell, f"{prefix}_points_mean",
                        _published_float(row[f"{prefix}_points_mean"]), table)
                    put(cell, f"{prefix}_seed_signs", row[f"{prefix}_seed_signs"], table)
                    put(cell, f"{prefix}_reading", row[f"{prefix}_reading"], table)
                else:
                    raise ValueError(f"unknown published reading table {table!r}")
    for cell in sorted(grid):
        for column in PUBLISHED_COLUMNS:
            if column not in readings.get(cell, {}):
                problems.append(f"published {column} of {cell} missing")
    return dict(readings), problems


# Full-window table column -> (table, published column).
PUBLISHED_COMPARISONS = (
    ("horizon", "S_full", "S"),
    ("class_order", "R_matched_learned_full", "R_matched_learned"),
    ("class_order", "R_matched_recency_full", "R_matched_recency"),
    ("class_order", "reading_full", "class_order_reading"),
    ("order", "learned_minus_recency_full_points_mean", "learned_minus_recency_points_mean"),
    ("order", "learned_minus_recency_full_seed_signs", "learned_minus_recency_seed_signs"),
    ("order", "learned_minus_recency_full_reading", "learned_minus_recency_reading"),
    ("admission", "label_binary_minus_recency_full_points_mean",
     "label_binary_minus_recency_points_mean"),
    ("admission", "label_binary_minus_recency_full_seed_signs",
     "label_binary_minus_recency_seed_signs"),
    ("admission", "label_binary_minus_recency_full_reading", "label_binary_minus_recency_reading"),
)
PUBLISHED_COLUMNS = tuple(published for _, _, published in PUBLISHED_COMPARISONS)


# --- the replay worker -----------------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one arm under `all16`: the matched class-order runner's
    replay call and row (the error-location statistics at the trace's label
    horizon around the arm's override; for the two class-order arms the class
    statistics at h* inside them, as there), with the head collector chained
    beside the attribution collector, and the full, head and tail counters."""
    name, fraction, multiplier, arm, seed = task
    if arm not in cell_arms(fraction, multiplier):
        raise ValueError(f"{arm} is not an arm of cell {(fraction, multiplier)}")
    h = matched_horizon(fraction, multiplier)
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace, horizon, seed, learned_ranker=SHARED["rankers"][name])
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    window = head_collector(trace, split_ms)
    statistics = DecisionStatistics(trace, horizon, split_ms)
    class_statistics = None
    if is_class_order_arm(arm):
        class_statistics = ClassStatistics(trace, h, split_ms)
        # Both recorders see every decision with its final victim; the arm's
        # own answer passes through both unchanged (the matched runner's nesting).
        override = RecordingOverride(statistics,
                                     RecordingOverride(class_statistics, setup.override))
    else:
        override = RecordingOverride(statistics, setup.override)
    started = time.time()
    result = run_two_tier(
        trace, rdp.L1_POLICY, l1_bytes, l2_bytes, hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=SHARED["groups"][name], measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=WIDTH, l2_seed=seed,
        l2_eligibility=ELIGIBILITY, l2_arm=arm,
        l2_request_hook=ChainedRequestHook(collector.on_request, window.on_request),
        l2_removal_hook=collector, l2_override_hook=override, **setup.replay_arguments(),
    )
    seconds = time.time() - started
    # The partition and per-block identities of Phase 0.98b, and every counter
    # the replay keeps for itself, must agree with the attribution built beside it.
    collector.check_against(result)
    requested = max(result.requested_tokens, 1)
    role = arm_role(arm)
    parameter = arm_horizon(arm)
    row = {
        "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
        "cell": rdp.cell_label(fraction, multiplier), "eligibility": ELIGIBILITY,
        "width": WIDTH, "mechanism": MECHANISM, "arm": arm, "family": arm_family(arm),
        "arm_parameter": "" if parameter is None else parameter, "seed": seed,
        "variant": VARIANT, "arm_role": role, "h_star": h,
        "l1_capacity_bytes": result.l1_capacity_bytes,
        "l2_capacity_bytes": result.l2_capacity_bytes,
        "requested_tokens": result.requested_tokens,
        "measured_requests": result.measured_requests,
        "l1_avoided_tokens": result.l1_avoided_tokens,
        "l2_avoided_tokens": result.l2_avoided_tokens,
        "avoided_prefill_tokens": result.avoided_prefill_tokens,
        "extra_avoided_tokens": result.avoided_prefill_tokens - result.l1_avoided_tokens,
        "l2_present_unusable_tokens": result.l2_present_unusable_tokens,
        "l2_present_unusable_blocks": result.l2_present_unusable_blocks,
        "l2_admissions": result.l2_admissions,
        "l2_rejections": result.l2_rejections,
        "l2_evictions": result.l2_evictions,
        "l2_decisions": result.l2_decisions,
        "l1_evictions": result.l1_evictions,
        "seconds": seconds,
        "worker_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF),
        "worker_pss_mib_end": rmc._proc_mib("/proc/self/smaps_rollup", "Pss"),
        "counters_sha256": rel._counters_digest(result, collector),
        **collector.loss_row(),
        **collector.orphaning_row(),
    }
    row["extra_fraction_of_input"] = row["extra_avoided_tokens"] / requested
    for column, points_column in rmc.POINT_COLUMNS:
        row[points_column] = 100.0 * row[column] / requested
    charged = row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
    for category in ("rejected", "evicted"):
        row[f"absent_{category}_share_of_decision_absent"] = (
            row[f"absent_{category}_tokens"] / charged if charged else math.nan
        )
    row.update(statistics.row())
    if class_statistics is not None:
        row["order"] = ORDER_OF_ARM[arm]
        row.update(class_statistics.row())
    full = full_row(result)
    head = head_row(window)
    row.update(window_columns(full, head, tail_row(full, head)))
    return row


def arm_family(arm: str) -> str:
    """The `family` column of the arm's published row."""
    if arm in ERROR_LOCATION_ARMS:
        return rel.arm_family(arm)
    if arm_role(arm) == RUNG_ROLE:
        return "horizon"
    return mco.FAMILY


# Scheduling class, most expensive first: the ranker scored by the store and at
# admission decisions; the ranker in one role (with the label beside it); the
# label family alone; LRU.
_ARM_COST = {"evict_binary_h*_learned": 0, "evict_binary_h*_recency": 1, "evict_label": 1,
             "learned": 1, "label": 2, RUNG_ROLE: 2, "lru": 3}


def arm_cost(arm: str) -> int:
    return _ARM_COST[arm_role(arm)]


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay: the seven arms of each cell, the most expensive and the
    larger L1 first. The order only schedules work; every table is sorted
    before it is written."""
    tasks = [(name, fraction, multiplier, arm, seed)
             for name in names for fraction, multiplier in cells
             for arm in cell_arms(fraction, multiplier) for seed in seeds]
    tasks.sort(key=lambda task: (arm_cost(task[3]), -task[1], -task[2], task[0], task[3],
                                 task[4]))
    return tasks


# --- checks ----------------------------------------------------------------------------------------


def _where(row) -> str:
    return f"{row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}"


def check_reproduction(rows, published, expected: int) -> tuple[dict, list[dict]]:
    """Every replay against the published row of its arm and trace x cell x
    seed: `avoided_prefill_tokens` and `counters_sha256` exactly, and
    `decision_sha256` where both rows carry one. Annotates each row and
    returns the report and one table row per replay. Every one of `expected`
    must be found and equal."""
    matched = seen = decisions_compared = 0
    mismatches: list[str] = []
    missing: list[str] = []
    table: list[dict] = []
    for row in rows:
        seen += 1
        reference = published.get((MECHANISM, row["arm"]) + _cell_seed(row))
        if reference is None:
            row["reference_source"] = ""
            row["reference_avoided_prefill_tokens"] = ""
            row["reproduces_reference"] = ""
            missing.append(f"{_where(row)}: no published {row['arm']}/{MECHANISM} row")
            continue
        same = {"avoided_prefill_tokens":
                int(row["avoided_prefill_tokens"]) == int(reference["avoided_prefill_tokens"]),
                "counters_sha256": str(row["counters_sha256"]) == reference.get("counters_sha256")}
        for column in REPRODUCTION_OPTIONAL_COLUMNS:
            if row.get(column, "") not in ("", None) and column in reference:
                decisions_compared += 1
                same[column] = str(row[column]) == reference[column]
        holds = all(same.values())
        row["reference_source"] = reference["source"]
        row["reference_avoided_prefill_tokens"] = reference["avoided_prefill_tokens"]
        row["reproduces_reference"] = holds
        entry = {"trace": row["trace"], "l1_fraction": row["l1_fraction"],
                 "l2_multiplier": row["l2_multiplier"], "cell": row["cell"],
                 "seed": row["seed"], "arm_role": arm_role(row["arm"]), "arm": row["arm"],
                 "reference_source": reference["source"]}
        for column in REPRODUCTION_COLUMNS + REPRODUCTION_OPTIONAL_COLUMNS:
            entry[column] = row.get(column, "")
            entry[f"published_{column}"] = reference.get(column, "")
            entry[f"same_{column}"] = same.get(column, "")
        entry["reproduces"] = holds
        table.append(entry)
        if holds:
            matched += 1
        else:
            mismatches.append(f"{_where(row)} vs published {reference['source']}: differs in "
                              + ", ".join(column for column, value in same.items() if not value))
    report = {"reference": "the published row of every arm (" + ", ".join(SOURCES) + ")",
              "expected": expected, "replays": seen, "matched": matched,
              "missing": len(missing), "mismatched": len(mismatches),
              "decision_digests_compared": decisions_compared,
              "mismatches": missing + mismatches}
    return report, table


reproduction_passes = mco.reproduction_passes


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed (the seven arms of its cell), on each identifier the reference
    publishes. Returns the pairs compared and the problems."""
    compared, problems = 0, []
    for row in rows:
        cell = (float(row["l1_fraction"]), float(row["l2_multiplier"]))
        for mechanism, arm in REFERENCES:
            if cell not in reference_cells(mechanism, arm):
                continue
            reference = published.get((mechanism, arm) + _cell_seed(row))
            if reference is None:
                problems.append(f"{_where(row)}: no published {arm}/{mechanism} reference row")
                continue
            compared += 1
            for field in REFERENCE_IDENTIFIERS:
                if field in reference and int(row[field]) != reference[field]:
                    problems.append(f"{_where(row)}: {field} {row[field]} != {reference[field]} "
                                    f"of the published {arm}/{mechanism} row")
    return compared, problems


def check_windows(rows) -> tuple[int, list[str]]:
    """The plan's head-window check: in every replay the head counters are at
    most the full ones (and the tail counters are full minus head, the full
    counters the replay's own); in every trace x cell x seed the seven arms
    are there and share the head `measured_requests`, `requested_tokens` and
    `l1_avoided_tokens`. Returns the groups and the problems."""
    problems: list[str] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_seed(row)].append(row)
        full, head, tail = (counters_of(row, window) for window in WINDOWS)
        for counter in COUNTERS:
            if head[counter] > full[counter]:
                problems.append(f"{_where(row)}: head {counter} {head[counter]} > full "
                                f"{full[counter]}")
            if head[counter] < 0:
                problems.append(f"{_where(row)}: head {counter} {head[counter]} < 0")
            if tail[counter] != full[counter] - head[counter]:
                problems.append(f"{_where(row)}: tail {counter} {tail[counter]} != full - head")
        for counter in ("measured_requests", "requested_tokens", "l1_avoided_tokens",
                        "avoided_prefill_tokens", "l2_present_unusable_tokens",
                        "l2_present_unusable_blocks"):
            if full[counter] != int(row[counter]):
                problems.append(f"{_where(row)}: full {counter} {full[counter]} != the replay's "
                                f"{row[counter]}")
    for key, members in sorted(groups.items()):
        if len(members) != ARMS_PER_CELL:
            problems.append(f"{key}: {len(members)} arms, expected {ARMS_PER_CELL}")
        for counter in HEAD_INVARIANT_COUNTERS:
            values = sorted({int(member[f"head_{counter}"]) for member in members})
            if len(values) != 1:
                problems.append(f"{key}: head {counter} takes {values} across the arms")
    return len(groups), problems


def check_statistics(rows) -> list[str]:
    """The horizon control's statistics check (every decision seen with its
    final victim, a decision in the window, `label`'s identities, no override
    by `learned`, `label` or a horizon-control rung), and no override by any
    arm of this run without one (`lru`, `learned`, `label`, `label_binary_h*`,
    150 s included)."""
    problems = hc.check_statistics(rows)
    for row in rows:
        if arm_role(row["arm"]) in NO_OVERRIDE_ROLES and int(row["overridden_decisions_seen"]):
            problems.append(f"{_where(row)}: {row['overridden_decisions_seen']} overridden "
                            "decisions for an arm that must keep the store's choice")
    return problems


def check_class(rows) -> tuple[int, list[str]]:
    """The matched class-order runner's class check on the two class-order
    arms of every cell, and their class statistics present; no other arm
    carries them. Returns the class-order rows checked and the problems."""
    class_rows, problems = [], []
    for row in rows:
        carries = row.get("class_horizon_seconds", "") not in ("", None)
        if is_class_order_arm(row["arm"]):
            if not carries:
                problems.append(f"{_where(row)}: no class statistics")
                continue
            class_rows.append(row)
        elif carries:
            problems.append(f"{_where(row)}: class statistics on an arm that is not a "
                            "class-order arm")
    return len(class_rows), problems + mco.check_class(class_rows)


# --- derivation ------------------------------------------------------------------------------------

_put_stats = hc._put_stats
_stats = rmc._stats
_put_signs = hc._put_signs
_identity = hc._identity
_is_nan = hc._is_nan
_cell_key = hc._cell_key
_u = hc._u
index_arms = hc.index_arms


def _role_index(arm: str) -> int:
    return ARMS.index(arm_role(arm))


def _cell_context(cell) -> dict:
    """h* and the arms of one trace x cell."""
    h = matched_horizon(cell[1], cell[2])
    arms = dict(zip(ARMS, cell_arms(cell[1], cell[2])))
    return {"h_star": h, "rung_arm": arms[RUNG_ROLE],
            "rung_source": reference_source(arms[RUNG_ROLE]),
            "matched_learned_arm": arms["evict_binary_h*_learned"],
            "matched_recency_arm": arms["evict_binary_h*_recency"]}


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x
    arm, of every replay metric, U and extra tokens per window, and (for the
    class-order arms) every class count."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _role_index(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "arm": arm, "arm_role": arm_role(arm),
                 "family": arm_family(arm), "h_star": matched_horizon(key[1], key[2]),
                 "reference_source": reference_source(arm), "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes", "head_requested_tokens", "tail_requested_tokens"):
            entry[column] = members[0][column]
        metrics = REPLAY_METRICS + WINDOW_METRICS
        if is_class_order_arm(arm):
            metrics += CLASS_METRICS
        for metric in metrics:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        out.append(entry)
    return out


def reference_rows(published, rows) -> list[dict]:
    """The published reference rows of every trace x cell x seed of the run,
    with their U in points: the rows every replay was held to, as read."""
    out = []
    for cell_seed in sorted({_cell_seed(row) for row in rows}):
        for (mechanism, arm), source in REFERENCES.items():
            if cell_seed[1:3] not in reference_cells(mechanism, arm):
                continue
            entry = dict(published[(mechanism, arm) + cell_seed])
            entry["arm_role"] = arm_role(arm)
            entry["extra_points"] = _u(entry)
            out.append(entry)
    return out


def window_tables(rows) -> dict[str, list[dict]]:
    """The readings on every window, from the replays alone.

    Per trace x cell x arm (`windows`): U on full, head and tail (five-seed
    mean, 95% half-width, min, max) and each window's input tokens and
    requests. Per trace x cell: `horizon` (reading 1: S_h* and its reading per
    window, the head and tail readings' agreement with full), `class_order`
    (reading 2: R_h* of both class-order arms and the reading per window,
    agreement, the predicted head reading), `order` and `admission`
    (readings 3 and 4: the seed-paired differences per window with seed
    signs and categorical reading, and the agreement of the sign of the head
    and tail means with the full mean), and per trace x cell x difference
    `tail_share` (reading 5). Five-seed means are `_stats`' (the parents'), so
    that the full window is the published arithmetic on the reproduced rows.
    """
    windows, horizon, class_order, order, admission, shares = [], [], [], [], [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        context = _cell_context(cell)
        arms = dict(zip(ARMS, cell_arms(cell[1], cell[2])))
        per_seed = []
        for seed in seeds:
            missing = [arm for arm in arms.values() if arm not in by_seed[seed]]
            if missing:
                raise KeyError(f"{cell} seed {seed}: replays of {missing} missing")
            per_seed.append({role: by_seed[seed][arm] for role, arm in arms.items()})
        u = {window: {role: [window_utility(entry[role], window) for entry in per_seed]
                      for role in ARMS} for window in WINDOWS}
        means = {window: {role: _stats(values)["mean"] for role, values in u[window].items()}
                 for window in WINDOWS}
        base = {**identity, **context, "seeds": len(seeds)}

        for role, arm in arms.items():
            first = per_seed[0][role]
            entry = {**identity, "h_star": context["h_star"], "arm_role": role, "arm": arm,
                     "reference_source": reference_source(arm), "seeds": len(seeds)}
            for window in WINDOWS:
                entry[f"{window}_measured_requests"] = int(first[f"{window}_measured_requests"])
                entry[f"{window}_requested_tokens"] = int(first[f"{window}_requested_tokens"])
            for window in WINDOWS:
                _put_stats(entry, f"U_{window}_points", u[window][role], fields=POINT_FIELDS)
            windows.append(entry)

        entry = dict(base)
        for window in WINDOWS:
            for role in ("label", "lru", RUNG_ROLE):
                entry[f"U_{SLUGS[role]}_{window}_points_mean"] = means[window][role]
        for window in WINDOWS:
            value = window_shortfall(means[window]["label"], means[window][RUNG_ROLE],
                                     means[window]["lru"])
            entry[f"S_{window}"] = value
            entry[f"reading_{window}"] = shortfall_reading(value)
        entry["predicted_reading_head"] = HORIZON_READINGS[0]
        for window in ("head", "tail"):
            entry[f"reading_{window}_agrees_with_full"] = (entry[f"reading_{window}"]
                                                           == entry["reading_full"])
        horizon.append(entry)

        entry = dict(base)
        for window in WINDOWS:
            for role in ("learned", "evict_label") + tuple(ARMS[-2:]):
                entry[f"U_{SLUGS[role]}_{window}_points_mean"] = means[window][role]
        for window in WINDOWS:
            for role in ARMS[-2:]:
                entry[f"R_{SLUGS[role]}_{window}"] = window_recovery(
                    means[window][role], means[window]["learned"], means[window]["evict_label"])
            entry[f"reading_{window}"] = class_order_reading(entry[f"R_matched_learned_{window}"])
        entry["predicted_reading_head"] = predicted_head_class_reading(cell[1:])
        for window in ("head", "tail"):
            entry[f"reading_{window}_agrees_with_full"] = (entry[f"reading_{window}"]
                                                           == entry["reading_full"])
        class_order.append(entry)

        for table, name in ((order, "learned_minus_recency"),
                            (admission, "label_binary_minus_recency")):
            later, earlier, _ = DIFFERENCES[name]
            entry = dict(base)
            for window in WINDOWS:
                for role in (later, earlier):
                    entry[f"U_{SLUGS[role]}_{window}_points_mean"] = means[window][role]
            for window in WINDOWS:
                values = [window_difference(seed_arms[later], seed_arms[earlier], window)[1]
                          for seed_arms in per_seed]
                _put_stats(entry, f"{name}_{window}_points", values, fields=POINT_FIELDS)
                _put_signs(entry, f"{name}_{window}", values)
            for window in ("head", "tail"):
                entry[f"{name}_{window}_mean_sign_agrees_with_full"] = sign_agrees(
                    entry[f"{name}_{window}_points_mean"], entry[f"{name}_full_points_mean"])
            table.append(entry)

        for name, (later, earlier, reading) in DIFFERENCES.items():
            tokens = {window: [window_difference(seed_arms[later], seed_arms[earlier], window)
                               for seed_arms in per_seed] for window in WINDOWS}
            totals = {window: sum(token for token, _ in tokens[window]) for window in WINDOWS}
            inputs = {window: sum(int(seed_arms[later][f"{window}_requested_tokens"])
                                  for seed_arms in per_seed) for window in WINDOWS}
            entry = {**identity, "h_star": context["h_star"], "difference": name,
                     "reading": reading, "later_arm": arms[later], "earlier_arm": arms[earlier],
                     "seeds": len(seeds)}
            for window in WINDOWS:
                entry[f"{window}_tokens"] = totals[window]
            entry["tail_share_of_difference"] = tail_share(totals["full"], totals["tail"])
            for window in WINDOWS:
                entry[f"{window}_input_tokens"] = inputs[window]
            entry["tail_share_of_input"] = tail_share(inputs["full"], inputs["tail"])
            for window in WINDOWS:
                entry[f"{window}_points_mean"] = _stats(
                    [value for _, value in tokens[window]])["mean"]
            shares.append(entry)
    return {"windows": windows, "horizon": horizon, "class_order": class_order, "order": order,
            "admission": admission, "tail_share": shares}


def _same(left, right) -> bool:
    """Exact equality; two nans are the same value."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def compare_with_published(tables, published_readings) -> tuple[int, list[str]]:
    """The full-window readings against the published values of the same
    trace x cell, exactly (`PUBLISHED_COMPARISONS`). Annotates each compared
    entry with `<column>_published` and `full_equals_published`. Returns the
    values compared and the problems (a missing published value is one)."""
    compared, problems = 0, []
    for name in ("horizon", "class_order", "order", "admission"):
        for entry in tables[name]:
            cell = (entry["trace"], entry["l1_fraction"], entry["l2_multiplier"])
            published = published_readings.get(cell, {})
            equal = True
            for table, column, source in PUBLISHED_COMPARISONS:
                if table != name:
                    continue
                if source not in published:
                    problems.append(f"{name} {cell}: no published {source}")
                    entry[f"{column}_published"] = ""
                    equal = False
                    continue
                entry[f"{column}_published"] = published[source]
                compared += 1
                if not _same(entry[column], published[source]):
                    equal = False
                    problems.append(f"{name} {cell}: {column} {entry[column]!r} != published "
                                    f"{source} {published[source]!r}")
            entry["full_equals_published"] = equal
    return compared, problems


READING_COLUMNS = ("reading", "window", "kind", "cells", "count", "agrees_with_full",
                   "predicted", "prediction_holds") + SIGN_READINGS


def _summary_row(reading, window, kind, cells, **values) -> dict:
    entry = dict.fromkeys(READING_COLUMNS, "")
    entry.update(reading=reading, window=window, kind=kind, cells=cells, **values)
    return entry


def reading_summary(tables: dict[str, list[dict]]) -> list[dict]:
    """The four prediction counts (head window) and the descriptive counts
    (full and tail, and the categorical seed-sign readings per window)."""
    horizon, class_order = tables["horizon"], tables["class_order"]
    out = []
    for window in WINDOWS:
        head = window == "head"
        agreements = ("" if window == "full" else
                      sum(1 for entry in horizon if entry[f"reading_{window}_agrees_with_full"]))
        out.append(_summary_row(
            "1_horizon_S_at_most_0.10", window, "prediction" if head else "descriptive",
            len(horizon), count=sum(1 for entry in horizon if suffices(entry[f"S_{window}"])),
            agrees_with_full=agreements, predicted=len(horizon) if head else "",
            prediction_holds=(horizon_prediction_holds(entry["S_head"] for entry in horizon)
                              if head else "")))
    for window in WINDOWS:
        head = window == "head"
        agreements = ("" if window == "full" else
                      sum(1 for entry in class_order
                          if entry[f"reading_{window}_agrees_with_full"]))
        out.append(_summary_row(
            "2_class_order_R_learned_order_at_least_0.9", window,
            "prediction" if head else "descriptive", len(class_order),
            count=sum(1 for entry in class_order if recovers(entry[f"R_matched_learned_{window}"])),
            agrees_with_full=agreements,
            predicted=(sum(1 for entry in class_order
                           if entry["predicted_reading_head"] == MATCHED_READINGS[0])
                       if head else ""),
            prediction_holds=(class_order_prediction_holds(
                ((entry["l1_fraction"], entry["l2_multiplier"]), entry["reading_full"],
                 entry["reading_head"]) for entry in class_order) if head else "")))
    for window in WINDOWS:
        out.append(_summary_row(
            "2_context_R_recency_at_least_0.9", window, "descriptive", len(class_order),
            count=sum(1 for entry in class_order
                      if recovers(entry[f"R_matched_recency_{window}"]))))
    for number, name, table in (("3", "learned_minus_recency", tables["order"]),
                                ("4", "label_binary_minus_recency", tables["admission"])):
        for window in ("head", "tail"):
            head = window == "head"
            agreements = [entry[f"{name}_{window}_mean_sign_agrees_with_full"] for entry in table]
            out.append(_summary_row(
                f"{number}_{name}_sign_of_mean_agrees_with_full", window,
                "prediction" if head else "descriptive", len(table),
                count=sum(1 for value in agreements if value),
                agrees_with_full=sum(1 for value in agreements if value),
                predicted=len(table) if head else "",
                prediction_holds=sign_prediction_holds(agreements) if head else ""))
        for window in WINDOWS:
            out.append(_summary_row(f"{number}_{name}_seed_signs", window, "descriptive",
                                    len(table), **sign_reading_counts(
                                        entry[f"{name}_{window}_reading"] for entry in table)))
    return out


# --- README ----------------------------------------------------------------------------------------

README_TEXT = """# Evaluation-window check: do the matched-horizon readings hold when the last 600 seconds of the trace are excluded?

Pre-registration: `docs/tail-window-check-plan.md`. Every exact-label arm treats a state with no later occurrence in the trace as "not reused", which at the end of a finite trace is knowledge of the end. With traces, capacities (six cells), seeds 0-4, L1, L2 store, hit rule, sizes, mechanism `all16` (the arrival plus up to 16 uniformly sampled residents, any of which may leave), hooks and decision statistics unchanged, the seven arms behind the matched-horizon readings are rerun, each built exactly as its published run built it, and each replay is counted on three windows. `h*` per cell, the same on both traces, kept and not re-chosen: 0.25% x 1: 60 s; 0.25% x 4: 150 s; 1% x 1: 150 s; 1% x 4: 600 s; 2% x 1: 300 s; 2% x 4: 600 s. 2 traces x 6 cells x 7 arms x 5 seeds = 420 replays.

Arms (role: built by, published in): `lru`, `learned` (the frozen pi0 `next_use` ranker), `label` (the exact `next_use` label at the trace's 600-second label horizon), `evict_label` (admission by the ranker, eviction by the label): `errorloc`, `results/paper/error_location_001/`; `label_binary_h*` (key `(1 if the next use is at most h* s away else 0, last_group)`): `horizonctl` at 60, 300 and 600 s, `results/paper/horizon_control_001/`, and `horizonfill` at 150 s, `results/paper/horizon_fill_001/`; `evict_binary_h*_learned` / `evict_binary_h*_recency` (admission by the ranker; eviction by `(reusable within h*, ranker score, last_group)` / `(reusable within h*, last_group)`): `matchedorder`, `results/paper/matched_class_order_001/`. Every replay carries the Phase 0.98b attribution hooks and the error-location decision statistics at the trace's 600-second label horizon; the two class-order arms also carry the class statistics at `h*`, as their run did.

Windows. `full`: the published evaluation window, from the split to the trace end, the replay's own counters. `head`: from the split to `trace end - 600,000 ms`, inclusive, counted by `onpolicy.LabelWindowUtilityCollector(trace, split_ms, end_ms - 600,000)` as the on-policy run counted its label window, attached beside the attribution collector through `tailwindow.ChainedRequestHook` (a request hook that calls both with the same arguments and returns nothing). `tail`: `full - head`, by subtraction of the integer counters. `run_config.json` records the bounds of every trace in ms. `U` on a window is extra avoided prefill tokens (avoided minus L1-avoided) in points (100 x share) of that window's input tokens. No window depends on `h`.

Readings (each on every window; the predictions are on `head`; `full` is recomputed from the reproduced rows and checked equal to the published values): 1. `S_h* = (U(label) - U(label_binary_h*)) / (U(label) - U(lru))` on five-seed means, "reuse_label_suffices" when `S_h*` <= 0.10 (exactly 0.10 sufficing), "order_needed" otherwise (a nan included). 2. `R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))` on five-seed means for both class-order arms, "reuse_identification_at_matched_horizon_suffices" when `R_h*(evict_binary_h*_learned)` >= 0.9 (exactly 0.9 sufficing), "ranker_order_costs_at_matched_horizon" otherwise (a nan included). 3. `U(evict_binary_h*_learned) - U(evict_binary_h*_recency)` and 4. `U(label_binary_h*) - U(evict_binary_h*_recency)`, seed-paired: the agreement of the sign of the five-seed mean with the full window's (zero matches zero only, a nan matches nothing), and the categorical reading beside it. 5. The tail's share of the full-window token difference of `G = U(label) - U(learned)` and of every difference of readings 1-4, beside the tail's share of input tokens (descriptive).

Predictions, fixed before the run, on the head window: 1. `S_h*` <= 0.10 in 12/12 trace x cell. 2. `R_h*(evict_binary_h*_learned)` >= 0.9 in the same 10/12 trace x cell as on the full window, and below it at 0.25% x 1 on both traces. 3. The five-seed mean of reading 3 has its full-window sign in 12/12. 4. The same for reading 4.

Boundaries. A difference between `head` and `tail` mixes the exact labels' knowledge of the end with any change of the workload over time; nothing here attributes it to either. On the real traces the head window is 814.8 of the 1,414.8 seconds of the full window (the plan's figures; `run_config.json` records each trace's bounds). `h*` was read from full-window replays of the same traces and is kept, not re-chosen: the head readings are a sensitivity check of published readings, not a new selection. Every arm reads the trace's future and none is a policy. Seed intervals describe sampling-seed variability only.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated), `h_star` (seconds), `rung_arm` (`label_binary_{h*}`), `rung_source`, `matched_learned_arm`, `matched_recency_arm`; `arm` is the replayed (and published) arm name and `arm_role` its role (`lru`, `learned`, `label`, `evict_label`, `label_binary_h*`, `evict_binary_h*_learned`, `evict_binary_h*_recency`). In column names a role's `h*` is written `hstar`: `label_binary_hstar`, `matched_learned`, `matched_recency`. A window is `full`, `head` or `tail`. Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise).

- `replay_seeds.csv`: one row per replay: the columns of the matched class-order runner's replay rows (`eligibility`, `width`, `mechanism`, `arm`, `family` and `arm_parameter` as in the arm's published row, `variant` main, the replay counters, timing and worker memory, `counters_sha256`, the Phase 0.98b attribution columns, the point columns, the error-location decision statistics `stat_*`, `overridden_*`, `decision_sha256`, `m1` .. `m4_victim_at_horizon`; for the two class-order arms `order` and the class statistics `class_*`), plus `arm_role`, `h_star`; per window the counters `<window>_<counter>` for `measured_requests`, `requested_tokens`, `requested_blocks`, `l1_avoided_tokens`, `l2_avoided_tokens`, `avoided_prefill_tokens`, `l2_hit_blocks`, `l2_present_unusable_tokens`, `l2_present_unusable_blocks` (full: the replay's; head: the collector's; tail: full - head), `<window>_extra_avoided_tokens` and `U_<window>_points`; and `reference_source`, `reference_avoided_prefill_tokens`, `reproduces_reference`.
- `replay.csv`: per trace x cell x arm, `mechanism`, `arm_role`, `family`, `h_star`, `reference_source`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, `head_requested_tokens`, `tail_requested_tokens`, and mean / std / ci95_half / min / max of every replay metric and statistic, of `U_<window>_points` and `<window>_extra_avoided_tokens`, and of every class count of the class-order arms.
- `references_seeds.csv`: the published row every replay was held to, one per trace x cell x seed x arm: `source`, `mechanism`, `arm`, `arm_role`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`), the digests (`counters_sha256`, `decision_sha256`) and `extra_points` (U on the full window, as published).
- `windows.csv`: per trace x cell x arm, `reference_source`, `<window>_measured_requests` and `<window>_requested_tokens` (the window's requests and input tokens; arm- and seed-independent), and `U_<window>_points_mean` / `_ci95_half` / `_min` / `_max`.
- `horizon.csv` (reading 1): per trace x cell, `U_label_<window>_points_mean`, `U_lru_<window>_points_mean`, `U_label_binary_hstar_<window>_points_mean`; `S_<window>` and `reading_<window>`; `predicted_reading_head`; `reading_head_agrees_with_full`, `reading_tail_agrees_with_full`; `S_full_published` (the published S at h*: `horizon_control_001/horizon.csv`, or `horizon_fill_001/fill.csv` at 150 s) and `full_equals_published`.
- `class_order.csv` (reading 2): per trace x cell, `U_<role>_<window>_points_mean` for learned, evict_label, matched_learned, matched_recency; `R_matched_learned_<window>`, `R_matched_recency_<window>`, `reading_<window>` (on R of the learned-order arm); `predicted_reading_head` (the costs reading at 0.25% x 1, suffices elsewhere); `reading_head_agrees_with_full`, `reading_tail_agrees_with_full`; the published `R_matched_learned_full_published`, `R_matched_recency_full_published`, `reading_full_published` (`matched_class_order_001/class_order.csv`) and `full_equals_published`.
- `order.csv` (reading 3): per trace x cell, `U_matched_learned_<window>_points_mean`, `U_matched_recency_<window>_points_mean`, `learned_minus_recency_<window>_points_*` (U(evict_binary_h*_learned) - U(evict_binary_h*_recency), seed-paired; mean, ci95_half, min, max) with its seed signs (`learned_minus_recency_<window>_*`); `learned_minus_recency_head_mean_sign_agrees_with_full`, `..._tail_...`; the published full-window mean, seed signs and reading (`..._full_points_mean_published`, `..._full_seed_signs_published`, `..._full_reading_published`, `matched_class_order_001/order.csv`) and `full_equals_published`.
- `admission.csv` (reading 4): the same for `label_binary_minus_recency` (U(label_binary_h*) - U(evict_binary_h*_recency), seed-paired), with `U_label_binary_hstar_<window>_points_mean` and `U_matched_recency_<window>_points_mean`, against `matched_class_order_001/admission.csv`.
- `tail_share.csv` (reading 5, descriptive): per trace x cell x `difference` (`G` = label - learned; `label_minus_label_binary` and `label_minus_lru`, reading 1; `matched_learned_minus_learned`, `matched_recency_minus_learned`, `evict_label_minus_learned`, reading 2; `learned_minus_recency`, reading 3; `label_binary_minus_recency`, reading 4), `reading`, `later_arm`, `earlier_arm`; `<window>_tokens` (the difference in extra avoided tokens summed over the seeds); `tail_share_of_difference` (tail / full, nan when the full difference is 0; outside [0, 1] when head and tail differ in sign); `<window>_input_tokens` (summed over the seeds), `tail_share_of_input`; and `<window>_points_mean` (the seed-paired difference in points of the window's input tokens).
- `reproduction.csv`: one row per replay: `arm_role`, `arm`, `reference_source`, and for `avoided_prefill_tokens`, `counters_sha256`, `decision_sha256` this run's value, the published value (`published_<column>`) and whether they are equal (`same_<column>`); `reproduces` (all equal).
- `readings.csv`: the counts, one row per reading x window: `kind` (prediction on head; descriptive otherwise), `cells`, `count` (1: S_h* <= 0.10; 2: R_h*(learned order) >= 0.9; 2_context: R_h*(recency) >= 0.9; 3 and 4: sign of the mean agrees with full), `agrees_with_full` (cells whose reading or sign equals the full window's), `predicted` and `prediction_holds` (head only; prediction 2 holds when the head reading equals the full one in every trace x cell and is below 0.9 exactly at 0.25% x 1), and for `<n>_<difference>_seed_signs` the counts of consistent_gain / consistent_loss / mixed per window.
- `run_config.json`: plan and code commits, source manifest, trace, model, model-manifest and reference hashes (the four `replay_seeds.csv` and the five published reading tables), the window bounds of every trace in ms, the matched horizons, grid, arms, check results, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


_report_lines = mco._report_lines


def _sign_text(entry, prefix: str) -> str:
    return (f"{entry[f'{prefix}_points_mean']:7.3f} ({entry[f'{prefix}_seed_signs']}, "
            f"{entry[f'{prefix}_reading']})")


def main() -> None:
    args = parse_args()
    clock = time.time()
    validate_arguments(args)
    seeds = tuple(SEEDS)
    cells = tuple(CELLS)

    status = rmc._git("status", "--porcelain")
    if status:
        raise SystemExit(f"the working tree is not clean:\n{status}")
    differing = sources_differing_from_head()
    if differing:
        raise SystemExit(f"execution source differs from HEAD: {differing}")
    head_start = rmc._git("rev-parse", "HEAD")
    plan_commit = rmc._git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()

    traces, groups, splits, horizons, working_set, trace_files = {}, {}, {}, {}, {}, {}
    bounds = {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace {trace.name}")
        horizon, split_ms, _ = horizon_for(trace)
        if horizon != LABEL_HORIZON_SECONDS:
            raise SystemExit(f"{trace.name}: horizon {horizon} s, the plan fixes "
                             f"{LABEL_HORIZON_SECONDS} s")
        bounds[trace.name] = window_bounds(trace, split_ms)
        if not bounds[trace.name]["head_end_ms"] > split_ms:
            raise SystemExit(f"{trace.name}: the head window ends at "
                             f"{bounds[trace.name]['head_end_ms']} ms, not after the split at "
                             f"{split_ms} ms")
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = split_ms
        horizons[trace.name] = horizon
        working_set[trace.name] = working_set_bytes(trace)
        trace_files[trace.name] = {"path": str(path.resolve()), "sha256": sha256_path(path)}
        print(f"loaded {trace.name}: requests={len(trace.requests)} H={horizon:g}s "
              f"split={split_ms:.1f}ms head end={bounds[trace.name]['head_end_ms']:.1f}ms "
              f"end={trace.end_ms:.1f}ms (head {bounds[trace.name]['head_seconds']:.1f}s of "
              f"{bounds[trace.name]['full_seconds']:.1f}s)", flush=True)
    if set(traces) != set(TRACES):
        raise SystemExit(f"traces {sorted(traces)} do not match the grid {sorted(TRACES)}")
    names = tuple(sorted(traces))
    rankers, model_files = rmc.load_models(names)
    for name in names:
        print(f"  pi0 next_use {name}: {model_files[name]['path']} "
              f"sha256={model_files[name]['sha256'][:12]}", flush=True)
    published, reference_problems = load_references()
    reference_hashes = {_repository_path(path): sha256_path(path)
                        for path in REFERENCE_SOURCES.values()}
    print(f"  published references: {len(published)} rows of {len(REFERENCES)} arms from "
          f"{', '.join(REFERENCE_SOURCES)}", flush=True)
    for path, digest in reference_hashes.items():
        print(f"    {path} sha256={digest[:12]}", flush=True)
    if reference_problems:
        _report_lines("REFERENCE", reference_problems)
        raise SystemExit("published references are not what the plan names; nothing run")
    published_readings, reading_problems = load_published_readings()
    reading_hashes = {_repository_path(path): sha256_path(path)
                      for path in PUBLISHED_READING_SOURCES.values()}
    for path, digest in reading_hashes.items():
        print(f"    {path} sha256={digest[:12]}", flush=True)
    if reading_problems:
        _report_lines("PUBLISHED", reading_problems)
        raise SystemExit("published readings are not what the plan names; nothing run")

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                  rankers=rankers)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds)
    workers = min(args.workers, len(tasks))
    print(f"{len(tasks)} replays on {workers} workers ({len(names)} traces x {len(cells)} cells x "
          f"{ARMS_PER_CELL} arms x {len(seeds)} seeds); matched horizons "
          + ", ".join(f"{rdp.cell_label(*cell)}: {matched_horizon(*cell):g}s" for cell in cells),
          flush=True)
    available_start = rmc._proc_mib("/proc/meminfo", "MemAvailable")
    rows: list[dict] = []
    started = time.time()
    context = mp.get_context("fork")
    raw_path = args.output_dir / "raw_replays.jsonl"
    pool = context.Pool(workers)
    try:
        with raw_path.open("w", encoding="utf-8") as raw:
            for done, row in enumerate(pool.imap_unordered(_replay_worker, tasks, chunksize=1), 1):
                rows.append(row)
                raw.write(json.dumps(row, default=str) + "\n")
                raw.flush()
                if done % 10 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {time.time() - started:.0f}s "
                          f"last={row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']} "
                          f"({row['seconds']:.0f}s, worker peak "
                          f"{row['worker_peak_rss_mib']:.0f} MiB)", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        # Joined, so every worker is reaped and counted in RUSAGE_CHILDREN.
        pool.join()
    replay_seconds = time.time() - started
    children_peak = rmc._peak_rss_mib(resource.RUSAGE_CHILDREN)
    worker_peak = max(float(row["worker_peak_rss_mib"]) for row in rows)
    worker_pss = max((float(row["worker_pss_mib_end"]) for row in rows
                      if math.isfinite(float(row["worker_pss_mib_end"]))), default=math.nan)
    print(f"  replays in {replay_seconds:.0f}s; peak worker RSS {worker_peak:.0f} MiB "
          f"(children max {children_peak:.0f} MiB), max worker PSS at replay end "
          f"{worker_pss:.0f} MiB, parent peak {rmc._peak_rss_mib(resource.RUSAGE_SELF):.0f} MiB, "
          f"MemAvailable at start {available_start:.0f} MiB", flush=True)

    timing: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        timing[(_role_index(row["arm"]), row["arm"])].append(float(row["seconds"]))
    print("  seconds per replay, by arm:", flush=True)
    timing_rows = []
    for (_, arm), values in sorted(timing.items()):
        print(f"    {arm:26s} {MECHANISM:6s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"arm": arm, "mechanism": MECHANISM, "variant": VARIANT,
                            "n": len(values), "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    reproduction, reproduction_table = check_reproduction(rows, published, len(tasks))
    print(f"  reproduction (every replay vs its published row; "
          f"{', '.join(REPRODUCTION_COLUMNS)}, and {', '.join(REPRODUCTION_OPTIONAL_COLUMNS)} "
          f"where the published row carries it): {reproduction['matched']} matched, "
          f"{reproduction['missing']} missing, {reproduction['mismatched']} mismatched "
          f"(expected {reproduction['expected']}; {reproduction['decision_digests_compared']} "
          "decision digests compared)", flush=True)
    _report_lines("MISMATCH", reproduction["mismatches"], limit=len(reproduction["mismatches"]))
    if not reproduction_passes(reproduction):
        failures.append("the replays do not reproduce their published rows")
    compared, identifier_problems = check_identifiers(rows, published)
    print(f"  identifiers ({', '.join(REFERENCE_IDENTIFIERS)}) of {len(rows)} replays against "
          f"{compared} published reference rows of their trace x cell x seed: "
          f"{'equal' if not identifier_problems else 'DIFFER'}", flush=True)
    _report_lines("IDENTIFIER", identifier_problems)
    if identifier_problems:
        failures.append("an identifier differs from a published reference row")
    window_groups, window_problems = check_windows(rows)
    print(f"  head window over {window_groups} trace x cell x seed groups (head <= full in every "
          f"counter; head {', '.join(HEAD_INVARIANT_COUNTERS)} equal across the {ARMS_PER_CELL} "
          f"arms): {'hold' if not window_problems else 'BROKEN'}", flush=True)
    _report_lines("WINDOW", window_problems)
    if window_problems:
        failures.append("a head-window check failed")
    statistics_problems = check_statistics(rows)
    print(f"  statistics: every decision seen with its final victim, no override without one: "
          f"{'hold' if not statistics_problems else 'BROKEN'}", flush=True)
    _report_lines("STATISTICS", statistics_problems)
    if statistics_problems:
        failures.append("a statistics check failed")
    class_checked, class_problems = check_class(rows)
    violations = sum(int(row["class_violations_seen"]) for row in rows
                     if is_class_order_arm(row["arm"]))
    outside = sum(int(row["class_overridden_outside_admission_seen"]) for row in rows
                  if is_class_order_arm(row["arm"]))
    print(f"  class at h* ({class_checked} class-order replays): resident evictions discarding a "
          f"state reusable within h* while a sampled resident is not: {violations}; overridden "
          f"decisions outside a first round with the arrival a candidate: {outside}; "
          f"{'hold' if not class_problems else 'BROKEN'}", flush=True)
    _report_lines("CLASS", class_problems)
    if class_problems:
        failures.append("a class check failed")
    cell_seeds, invariant_problems = rmc.check_invariants(rows, ARMS_PER_CELL)
    print(f"  invariants over {cell_seeds} trace x cell x seed groups "
          f"({', '.join(rmc.INVARIANT_COLUMNS)}): "
          f"{'hold' if not invariant_problems else 'BROKEN'}", flush=True)
    _report_lines("VARIES", invariant_problems)
    if invariant_problems:
        failures.append("an arm-independent counter varies across arms")
    unexplained = sum(int(row["absent_unexplained_tokens"]) for row in rows)
    print(f"  unexplained per-block absent tokens over the run: {unexplained}", flush=True)
    if unexplained:
        _report_lines("UNEXPLAINED", [f"{_where(row)}: {row['absent_unexplained_tokens']} tokens"
                                      for row in rows if int(row["absent_unexplained_tokens"])])
        failures.append("an absent token is not explained")
    if failures:
        raise SystemExit("checks failed, nothing derived or published: " + "; ".join(failures))

    # --- derived tables ---
    rows.sort(key=lambda row: (row["trace"], row["l1_fraction"], row["l2_multiplier"],
                               _role_index(row["arm"]), row["seed"]))
    reproduction_table.sort(key=lambda entry: (entry["trace"], entry["l1_fraction"],
                                               entry["l2_multiplier"],
                                               _role_index(entry["arm"]), entry["seed"]))
    replay_summary = aggregate_replays(rows)
    references = reference_rows(published, rows)
    tables = window_tables(rows)
    published_compared, published_problems = compare_with_published(tables, published_readings)
    print(f"  full-window readings against the published values: {published_compared} values "
          f"compared, {'equal' if not published_problems else 'DIFFER'}", flush=True)
    _report_lines("PUBLISHED", published_problems)
    if published_problems:
        raise SystemExit("the full-window readings differ from the published values; nothing "
                         "published")
    summary = reading_summary(tables)

    print("  reading 1, S_h* per window (five-seed means):", flush=True)
    for entry in tables["horizon"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              + " ".join(f"{window}={entry[f'S_{window}']:7.3f}" for window in WINDOWS)
              + f" head {entry['reading_head']}", flush=True)
    print("  reading 2, R_h*(learned order) / R_h*(recency) per window:", flush=True)
    for entry in tables["class_order"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              + " ".join(f"{window}={entry[f'R_matched_learned_{window}']:6.3f}/"
                         f"{entry[f'R_matched_recency_{window}']:6.3f}" for window in WINDOWS)
              + f" head {entry['reading_head']}", flush=True)
    for title, table, prefix in (
            ("reading 3, order within the matched class (learned - recency)", tables["order"],
             "learned_minus_recency"),
            ("reading 4, admission at the matched horizon (label_binary_h* - recency arm)",
             tables["admission"], "label_binary_minus_recency")):
        print(f"  {title}, points, full | head | tail:", flush=True)
        for entry in table:
            print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
                  + " | ".join(_sign_text(entry, f"{prefix}_{window}") for window in WINDOWS),
                  flush=True)
    for entry in summary:
        if entry["kind"] == "prediction":
            print(f"    prediction {entry['reading']} ({entry['window']}): {entry['count']}/"
                  f"{entry['cells']} (predicted {entry['predicted']}; "
                  f"{'holds' if entry['prediction_holds'] else 'FAILS'})", flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "tail_window_check",
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "git_dirty_at_start": bool(status),
        "execution_sources_differing_from_head": differing,
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "models": {"pi0_next_use": model_files},
        "model_manifests": {_repository_path(path): sha256_path(path)
                            for path in rmc.MODEL_MANIFESTS},
        "references": reference_hashes,
        "published_readings": reading_hashes,
        "reference_rows": {f"{arm}/{mechanism}": {
            "source": source,
            "cells": [list(cell) for cell in reference_cells(mechanism, arm)]}
            for (mechanism, arm), source in REFERENCES.items()},
        "reference_identifiers": list(REFERENCE_IDENTIFIERS),
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "mechanism": {"name": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH},
        "matched_horizon_seconds": [{"l1_fraction": cell[0], "l2_multiplier": cell[1],
                                     "h_star": matched_horizon(*cell),
                                     "arms": list(cell_arms(*cell))} for cell in cells],
        "arm_roles": list(ARMS),
        "windows": {"names": list(WINDOWS), "tail_seconds": TAIL_SECONDS,
                    "head": "split_ms <= t <= end_ms - 600000 "
                            "(onpolicy.LabelWindowUtilityCollector)",
                    "tail": "full - head (integer counters)", "bounds_ms": bounds},
        "readings": {"suffices_shortfall": SUFFICES_SHORTFALL,
                     "class_order_share": CLASS_ORDER_SHARE,
                     "predicted_below_cell": list(PREDICTED_BELOW_CELL),
                     "differences": {name: list(value) for name, value in DIFFERENCES.items()},
                     "sign_readings": list(SIGN_READINGS)},
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "statistics_horizon_seconds": {"decision_statistics": "trace label horizon (600 s)",
                                       "class_statistics": "h* of the cell (class-order arms)"},
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {k: v for k, v in reproduction.items() if k != "mismatches"},
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "window_groups": window_groups,
            "window_problems": len(window_problems),
            "head_invariant_counters": list(HEAD_INVARIANT_COUNTERS),
            "statistics_problems": len(statistics_problems),
            "class_replays_checked": class_checked,
            "class_problems": len(class_problems),
            "class_violations": violations,
            "class_overridden_outside_admission": outside,
            "invariant_groups": cell_seeds, "invariant_violations": len(invariant_problems),
            "invariant_columns": list(rmc.INVARIANT_COLUMNS),
            "unexplained_absent_tokens": unexplained,
            "published_readings_compared": published_compared,
            "published_reading_problems": len(published_problems),
        },
        "replays": len(rows), "workers": workers, "replay_seconds": replay_seconds,
        "wall_seconds": time.time() - clock,
        "seconds_by_arm": timing_rows,
        "memory": {
            "rss_unit": "MiB (Linux ru_maxrss KiB / 1024)",
            "peak_worker_rss_mib": worker_peak,
            "children_peak_rss_mib": children_peak,
            "max_worker_pss_mib_at_replay_end": worker_pss,
            "parent_peak_rss_mib": rmc._peak_rss_mib(resource.RUSAGE_SELF),
            "mem_available_mib_at_start": available_start,
        },
        "note": "Sensitivity check of published readings: the seven arms behind the "
                "matched-horizon readings rerun (each reproducing its published row), counted on "
                "the full window, on the head window ending 600 s before the trace end, and on "
                "the tail (full - head). h* is kept, not re-chosen. A head-tail difference mixes "
                "the exact labels' knowledge of the end with any change of the workload over "
                "time. Every arm reads the trace's future and none is a policy. Seed intervals "
                "describe sampling-seed variability only.",
    }
    for directory in (args.output_dir, args.paper_dir):
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        for name in ("windows", "horizon", "class_order", "order", "admission", "tail_share"):
            _write(directory / f"{name}.csv", tables[name])
        _write(directory / "reproduction.csv", reproduction_table)
        _write(directory / "readings.csv", summary)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {args.output_dir}, {args.paper_dir}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
