#!/usr/bin/env python3
"""Leaf-matched horizon: the matched-horizon bit and the class order under leaf eligibility.

The pre-registration is `docs/leaf-matched-horizon-plan.md`; nothing here may
change it. Four arms per trace x cell are replayed under the horizon control's
Part 3 mechanism `leaf16` (`l2_eligibility="leaf"`, width 16: every decision
set is restricted to leaf residents, and the arrival is offered in its own
first round only when it is a leaf), with the cell's matched horizon `h*`
carried over from `all16` unchanged (`persistent_kv_admission.leafmatched`):

* `label_binary_{h*}`: the label rung's scorer at `h*` (`horizonfill.arm_setup`);
* `evict_binary_{h*}_learned` and `evict_binary_{h*}_recency`: the matched
  class-order arms (`matchedorder.arm_setup`);
* `label`: the error-location `label` rung (`errorloc.arm_setup`), the
  reproduction anchor.

2 traces x 6 cells x 4 arms x 5 seeds = 240 replays, each with the Phase
0.98b attribution hooks and the error-location decision statistics (at the
trace's 600-second label horizon) attached read-only, and
`matchedorder.ClassStatistics` at `h*` around the three `h*` arms. The replay
call, the row schema, the provenance rules and the table helpers are the
matched class-order runner's (`scripts/run_matched_class_order.py`, imported by
path, unchanged), with `l2_eligibility="leaf"` in place of `"all"`; the frozen
ranker is loaded exactly as there.

References are published rows, not reruns, with the hash of every table read:
`lru`, `learned`, `label` under `leaf16` from
`results/paper/mechanism_control_001/replay_seeds.csv` (arm column `rung`) and
`evict_label` under `leaf16` from `results/paper/horizon_control_001/`; and,
for the `all16` values every reading is reported beside, `lru`, `learned`,
`label`, `evict_label` from `results/paper/error_location_001/`,
`label_binary_{h*}` from `results/paper/horizon_control_001/` (60, 300, 600 s)
or `results/paper/horizon_fill_001/` (150 s), and the matched arms from
`results/paper/matched_class_order_001/`. Before any replay the `all16` values
recomputed from those rows must equal the published `all16` tables they come
from (`S` of `horizon.csv` / `fill.csv`; `R_matched_*`, the readings, the
seed-paired means and seed signs of `class_order.csv`, `order.csv`,
`admission.csv`), so that what is printed beside a `leaf16` value is the
published `all16` value.

Nothing is fitted and no arm is a proposed policy: every arm reads the trace's
future on purpose; `leaf16` is a control on eligibility. Before anything is
derived or published the run checks that the 60 `label` replays reproduce the
published `leaf16` `label` rows exactly (see "Reproduction" below); that the
identifiers of every replay equal those of every published reference row of
its trace x cell x seed; that every replay ran under `leaf`/16 and has zero
present-but-unusable tokens; that the statistics saw every decision with its
final victim, no arm without an override overrode, and the `label` replays
carry the label identities; that the class statistic is zero in every replay
that carries it (the three `h*` arms); that the arm-independent counters are
arm-independent and no absent token is unexplained; and (per replay, in the
worker) the attribution identities of Phase 0.98b. Any failure exits without
writing a derived table, and nothing reaches the paper directory.

Reproduction. The plan holds the `label` replays to the published rows in
`avoided_prefill_tokens` and the counter digest. The published `leaf16`
`label` rows (mechanism_control_001) carry no counter digest and no decision
digest. Each `label` replay is therefore compared, exactly, on
`avoided_prefill_tokens` and on every counter column the published row
carries (the replay counters, the attribution and orphaning counters and the
point columns derived from them: every column except identity, timing,
memory and annotation columns), and on a digest wherever the published row
carries one; the replay's own `counters_sha256` and `decision_sha256` are
recorded. Without a published counter digest every replay counter of
`REQUIRED_COUNTER_COLUMNS` must be among the compared columns.

Outputs (in --output-dir and in --paper-dir): per-seed and aggregated replay
tables, the published reference rows used, the four readings per trace x cell
with the published `all16` value beside each, the per-seed values they are
computed from, the prediction counts, the reproduction table, a README
describing every table, and the run configuration. There is no smoke mode and
no dirty-tree mode: the plan runs once, from a clean tree at the code commit.
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
from persistent_kv_admission.horizonctl import (
    ALL16,
    CLASS_ORDER_SHARE,
    LEAF16,
    MECHANISMS,
    SUFFICES_SHORTFALL,
)
from persistent_kv_admission.leafmatched import (
    ANCHOR_ARM,
    ARMS_PER_CELL,
    CLASS_ORDER_ARMS,
    CLASS_READINGS,
    CLASS_STATISTIC_ARMS,
    ELIGIBILITY,
    FAMILY_OF_ARM,
    GRID_TRACE_CELLS,
    HORIZON_OF_ARM,
    HORIZON_READINGS,
    MECHANISM,
    NOT_PREDICTED_CELLS,
    ORDER_OF_ARM,
    PREDICTIONS,
    RUNG_ARMS,
    RUNG_OF_HORIZON,
    SIGN_READINGS,
    WIDTH,
    arm_setup,
    cell_arms,
    class_reading,
    class_recovery,
    hstar_reading,
    hstar_shortfall,
    outcome_counts,
    predicted,
    prediction_holds,
    row_arrival_candidate_share,
)
from persistent_kv_admission.matchedorder import (
    CLASS_COLUMNS,
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    ClassStatistics,
    matched_horizon,
)
from persistent_kv_admission.mechanism import sign_reading
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

MATCHED_CLASS_ORDER_SCRIPT = REPOSITORY / "scripts/run_matched_class_order.py"
PLAN_PATH = REPOSITORY / "docs/leaf-matched-horizon-plan.md"


def _load_matched_class_order():
    """Import the matched class-order runner by path, unchanged, for its
    replay constants, provenance, check and table helpers; it brings the
    horizon-control, error-location, mechanism-control and Phase 0.97 runners
    with it (one instance of each, so the capacity rule reads the working set
    this run sets, and the ranker loader is the one they use). `main` is
    behind the usual guard, so nothing runs, and nothing is read at import."""
    spec = importlib.util.spec_from_file_location("run_matched_class_order",
                                                  MATCHED_CLASS_ORDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mco = _load_matched_class_order()
hc = mco.hc
rel = mco.rel
rmc = mco.rmc
rdp = mco.rdp

# --- the fixed grid (docs/leaf-matched-horizon-plan.md) -------------------------------------------

TRACES = mco.TRACES
CELLS = mco.CELLS
SEEDS = mco.SEEDS
assert set(CELLS) == set(MATCHED_HORIZON_SECONDS)
assert len(TRACES) * len(CELLS) == GRID_TRACE_CELLS
LABEL_HORIZON_SECONDS = mco.LABEL_HORIZON_SECONDS
assert LABEL_HORIZON_SECONDS == 600.0
assert MECHANISM == LEAF16 and (ELIGIBILITY, WIDTH) == MECHANISMS[LEAF16] == ("leaf", 16)
VARIANT = "main"
DEFAULT_WORKERS = mco.DEFAULT_WORKERS
MAX_WORKERS = mco.MAX_WORKERS
assert (DEFAULT_WORKERS, MAX_WORKERS) == (10, 12)
# The plan's publication directory (a new one; refused if it exists).
PAPER_ROOT = REPOSITORY / "results/paper"
DEFAULT_PAPER_DIR = PAPER_ROOT / "leaf_matched_horizon_001"
# Published per-seed rows, anchored at the repository (tracked files).
MECHANISM_CONTROL_REFERENCE = PAPER_ROOT / "mechanism_control_001/replay_seeds.csv"
MATCHED_CLASS_ORDER_REFERENCE = PAPER_ROOT / "matched_class_order_001/replay_seeds.csv"
REFERENCE_SOURCES = {"mechanism_control_001": MECHANISM_CONTROL_REFERENCE,
                     "horizon_control_001": mco.HORIZON_CONTROL_REFERENCE,
                     "horizon_fill_001": mco.HORIZON_FILL_REFERENCE,
                     "error_location_001": mco.ERROR_LOCATION_REFERENCE,
                     "matched_class_order_001": MATCHED_CLASS_ORDER_REFERENCE}
RUNG_SOURCES = mco.RUNG_SOURCES
assert set(RUNG_SOURCES) == set(MATCHED_HORIZONS_SECONDS)
REFERENCE_ARMS = ("lru", "learned", "label", "evict_label")
# Published reference rows, by (mechanism, arm): the source each is read from.
# The leaf16 references are the plan's; the all16 rows are the published
# values of the same arms beside which every reading is reported. A rung or a
# matched arm is read at the cells whose h* is its horizon only.
REFERENCES = {
    (LEAF16, "lru"): "mechanism_control_001",
    (LEAF16, "learned"): "mechanism_control_001",
    (LEAF16, "label"): "mechanism_control_001",
    (LEAF16, "evict_label"): "horizon_control_001",
    **{(ALL16, arm): "error_location_001" for arm in REFERENCE_ARMS},
    **{(ALL16, RUNG_OF_HORIZON[h]): RUNG_SOURCES[h] for h in MATCHED_HORIZONS_SECONDS},
    **{(ALL16, arm): "matched_class_order_001" for arm in CLASS_ORDER_ARMS},
}
REFERENCE_IDENTIFIERS = mco.REFERENCE_IDENTIFIERS
# The published row each `label` replay must reproduce.
REPRODUCTION_REFERENCE = (LEAF16, ANCHOR_ARM)
# Columns of a published row that are not counters: identity, timing, memory,
# and annotations of the run that published it.
REPRODUCTION_SKIP = frozenset((
    "trace", "l1_fraction", "l2_multiplier", "cell", "eligibility", "width", "mechanism", "rung",
    "arm", "family", "arm_parameter", "seed", "variant", "seconds", "worker_peak_rss_mib",
    "worker_pss_mib_end", "reference_source", "reference_avoided_prefill_tokens",
    "reproduces_reference"))
DIGEST_COLUMNS = ("counters_sha256", "decision_sha256")
# Without a published counter digest, each of these must be published and equal.
REQUIRED_COUNTER_COLUMNS = (
    "l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens", "measured_requests",
    "l1_avoided_tokens", "l2_avoided_tokens", "avoided_prefill_tokens", "extra_avoided_tokens",
    "l2_present_unusable_tokens", "l2_present_unusable_blocks", "l2_admissions", "l2_rejections",
    "l2_evictions", "l2_decisions", "l1_evictions")
# The error-location statistics a published row keeps for the arrival share.
SHARE_COLUMNS = ("stat_decisions", "stat_decisions_admission")
# Published all16 tables the recomputed all16 values must equal before any replay.
PUBLISHED_TABLES = {
    "horizon_control_001/horizon.csv": PAPER_ROOT / "horizon_control_001/horizon.csv",
    "horizon_fill_001/fill.csv": PAPER_ROOT / "horizon_fill_001/fill.csv",
    "matched_class_order_001/class_order.csv":
        PAPER_ROOT / "matched_class_order_001/class_order.csv",
    "matched_class_order_001/order.csv": PAPER_ROOT / "matched_class_order_001/order.csv",
    "matched_class_order_001/admission.csv": PAPER_ROOT / "matched_class_order_001/admission.csv",
}
# The table that publishes S at each matched horizon (the rung's own run).
SHORTFALL_TABLES = {60.0: "horizon_control_001/horizon.csv", 150.0: "horizon_fill_001/fill.csv",
                    300.0: "horizon_control_001/horizon.csv",
                    600.0: "horizon_control_001/horizon.csv"}
assert set(SHORTFALL_TABLES) == set(MATCHED_HORIZONS_SECONDS)
# all16 value of this run -> (published table, column). Floats compare within
# ALL16_TOLERANCE (the same arithmetic on the same rows), text exactly.
ALL16_CHECKS = (
    ("S_hstar", "shortfall", "S"),
    ("R_matched_learned", "matched_class_order_001/class_order.csv", "R_matched_learned"),
    ("R_matched_recency", "matched_class_order_001/class_order.csv", "R_matched_recency"),
    ("class_reading", "matched_class_order_001/class_order.csv", "reading"),
    ("learned_minus_recency_points_mean", "matched_class_order_001/order.csv",
     "learned_minus_recency_points_mean"),
    ("learned_minus_recency_seed_signs", "matched_class_order_001/order.csv",
     "learned_minus_recency_seed_signs"),
    ("learned_minus_recency_reading", "matched_class_order_001/order.csv",
     "learned_minus_recency_reading"),
    ("label_binary_minus_recency_points_mean", "matched_class_order_001/admission.csv",
     "label_binary_minus_recency_points_mean"),
    ("label_binary_minus_recency_seed_signs", "matched_class_order_001/admission.csv",
     "label_binary_minus_recency_seed_signs"),
    ("label_binary_minus_recency_reading", "matched_class_order_001/admission.csv",
     "label_binary_minus_recency_reading"),
)
ALL16_TOLERANCE = 1e-12
REPLAY_METRICS = mco.REPLAY_METRICS
CLASS_METRICS = mco.CLASS_METRICS
SHARED: dict[str, object] = {}


def reference_cells(mechanism: str, arm: str) -> tuple[tuple[float, float], ...]:
    """The cells of the grid at which a reference is read: a rung or a
    matched arm at the cells whose h* is its horizon, every other reference at
    every cell."""
    if (mechanism, arm) not in REFERENCES:
        raise ValueError(f"unknown reference {arm}/{mechanism}")
    if arm in HORIZON_OF_ARM:
        return tuple(cell for cell in CELLS if MATCHED_HORIZON_SECONDS[cell] == HORIZON_OF_ARM[arm])
    return tuple(CELLS)


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


def validate_arguments(args) -> None:
    """Every refusal that needs no trace, model, reference or git call."""
    mco.validate_arguments(args)


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


def _reference_entry(row, mechanism: str, arm: str, source: str) -> dict:
    """The matched class-order runner's reference entry (identifiers and
    digests), with the error-location decision counts where the row carries
    them (the arrival share) and, for the reproduction target, every counter
    column it publishes, as published (`counters`)."""
    entry = mco._reference_entry(row, mechanism, arm, source)
    for column in SHARE_COLUMNS:
        if row.get(column, "") not in ("", None):
            entry[column] = int(row[column])
    if (mechanism, arm) == REPRODUCTION_REFERENCE:
        entry["counters"] = {column: value for column, value in row.items()
                             if column not in REPRODUCTION_SKIP and column not in DIGEST_COLUMNS
                             and column is not None}
    return entry


def load_references(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> published row`
    for every reference of `REFERENCES`, each read from its own source only
    (the arm from `arm`, or `rung` in the mechanism control's table), at the
    cells `reference_cells` names, with the eligibility and width of its
    mechanism; each must be there exactly once for every trace and seed of the
    grid. Rows of other arms, mechanisms, sources, cells, seeds or variants
    are skipped unchecked. `paths` overrides the source files (tests)."""
    paths = dict(REFERENCE_SOURCES, **(paths or {}))
    wanted = {key: set(reference_cells(*key)) for key in REFERENCES}
    published: dict[tuple, dict] = {}
    problems: list[str] = []
    for source, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                arm = row["arm"] if "arm" in row else row["rung"]
                key = (row["mechanism"], arm)
                if REFERENCES.get(key) != source or row.get("variant", VARIANT) != VARIANT:
                    continue
                cell_seed = _cell_seed(row)
                if (cell_seed[0] not in TRACES or cell_seed[1:3] not in wanted[key]
                        or cell_seed[3] not in SEEDS):
                    continue
                if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
                    problems.append(f"{source} row {arm}/{key[0]} has "
                                    f"{row['eligibility']}/{row['width']}")
                full = key + cell_seed
                if full in published:
                    problems.append(f"duplicate published reference {full}")
                    continue
                published[full] = _reference_entry(row, key[0], arm, source)
    for key, source in REFERENCES.items():
        for name in TRACES:
            for fraction, multiplier in reference_cells(*key):
                for seed in SEEDS:
                    full = key + (name, fraction, multiplier, seed)
                    if full not in published:
                        problems.append(f"published {source} reference {full} missing")
    return published, problems


def load_published_all16(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(trace, fraction, multiplier) -> {name: published value}` for every
    trace x cell of the grid, `name` as in `ALL16_CHECKS`, each read from its
    table (S from the table of the cell's h*, at that h). `paths` overrides
    the table files (tests)."""
    paths = dict(PUBLISHED_TABLES, **(paths or {}))
    rows = {}
    for table, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            rows[table] = list(csv.DictReader(handle))
    published: dict[tuple, dict] = {}
    problems: list[str] = []
    for name in TRACES:
        for fraction, multiplier in CELLS:
            cell = (name, fraction, multiplier)
            h = MATCHED_HORIZON_SECONDS[(fraction, multiplier)]
            values = {}
            for mine, table, column in ALL16_CHECKS:
                if table == "shortfall":
                    table = SHORTFALL_TABLES[h]
                matches = [row for row in rows[table]
                           if (row["trace"], float(row["l1_fraction"]),
                               float(row["l2_multiplier"])) == cell
                           and ("h" not in row or float(row["h"]) == h)
                           and ("h_star" not in row or float(row["h_star"]) == h)]
                if len(matches) != 1:
                    problems.append(f"{table}: {len(matches)} rows for {name}/{fraction}/"
                                    f"{multiplier} at h* = {h:g} s, expected 1")
                    continue
                values[mine] = matches[0][column]
            published[cell] = values
    return published, problems


def _same_published_value(value, published: str) -> bool:
    """A value of this run against its published text: integers exactly,
    floats exactly (nan equal to nan), anything else as text."""
    if isinstance(value, bool) or value is None:
        return str(value) == published
    if isinstance(value, (int, np.integer)):
        try:
            return int(published) == int(value)
        except ValueError:
            return False
    if isinstance(value, (float, np.floating)):
        try:
            other = float(published)
        except ValueError:
            return False
        return (math.isnan(value) and math.isnan(other)) or float(value) == other
    return str(value) == published


def _close(value: float, published: str) -> bool:
    try:
        other = float(published)
    except ValueError:
        return False
    if math.isnan(value) or math.isnan(other):
        return math.isnan(value) and math.isnan(other)
    return math.isclose(value, other, rel_tol=ALL16_TOLERANCE, abs_tol=ALL16_TOLERANCE)


def check_all16(values_by_cell: dict[tuple, dict], published: dict[tuple, dict]) -> list[str]:
    """The all16 values this run recomputed from the published rows against
    the published all16 tables, cell by cell: floats within
    `ALL16_TOLERANCE`, readings and seed signs exactly."""
    problems = []
    for cell, published_values in sorted(published.items()):
        mine = values_by_cell.get(cell)
        if mine is None:
            problems.append(f"{cell}: no all16 values recomputed")
            continue
        for name, table, column in ALL16_CHECKS:
            if name not in published_values:
                continue
            value, theirs = mine[name], published_values[name]
            same = _close(value, theirs) if isinstance(value, float) else str(value) == theirs
            if not same:
                source = SHORTFALL_TABLES[MATCHED_HORIZON_SECONDS[cell[1:]]] if (
                    table == "shortfall") else table
                problems.append(f"{cell}: all16 {name} {value!r} != published {theirs!r} "
                                f"({source}, column {column})")
    return problems


# --- the replay worker -----------------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one arm under `leaf16`, with the attribution hooks, the
    error-location statistics at the trace's label horizon and, for the three
    h* arms, the class statistics at the cell's h*, both around the arm's
    override: the matched class-order runner's replay call and row, with leaf
    eligibility."""
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
    statistics = DecisionStatistics(trace, horizon, split_ms)
    class_statistics = None
    inner = setup.override
    if arm in CLASS_STATISTIC_ARMS:
        class_statistics = ClassStatistics(trace, h, split_ms)
        inner = RecordingOverride(class_statistics, setup.override)
    # Every recorder sees every decision with its final victim; the arm's own
    # answer passes through unchanged.
    override = RecordingOverride(statistics, inner)
    started = time.time()
    result = run_two_tier(
        trace, rdp.L1_POLICY, l1_bytes, l2_bytes, hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=SHARED["groups"][name], measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=WIDTH, l2_seed=seed,
        l2_eligibility=ELIGIBILITY, l2_arm=arm,
        l2_request_hook=collector.on_request, l2_removal_hook=collector,
        l2_override_hook=override, **setup.replay_arguments(),
    )
    seconds = time.time() - started
    # The partition and per-block identities of Phase 0.98b, and every counter
    # the replay keeps for itself, must agree with the attribution built beside it.
    collector.check_against(result)
    requested = max(result.requested_tokens, 1)
    row = {
        "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
        "cell": rdp.cell_label(fraction, multiplier), "eligibility": ELIGIBILITY,
        "width": WIDTH, "mechanism": MECHANISM, "arm": arm, "family": FAMILY_OF_ARM[arm],
        "arm_parameter": HORIZON_OF_ARM.get(arm, ""), "seed": seed, "variant": VARIANT,
        "h_star": h, "order": ORDER_OF_ARM.get(arm, ""),
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
    for column, points in rmc.POINT_COLUMNS:
        row[points] = 100.0 * row[column] / requested
    charged = row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
    for category in ("rejected", "evicted"):
        row[f"absent_{category}_share_of_decision_absent"] = (
            row[f"absent_{category}_tokens"] / charged if charged else math.nan
        )
    row.update(statistics.row())
    if class_statistics is not None:
        row.update(class_statistics.row())
    return row


def arm_cost(arm: str) -> int:
    """Scheduling class: the ranker scored by the store and at admission
    decisions first, the ranker at admission decisions second, the labels
    last."""
    order = ORDER_OF_ARM.get(arm)
    return {"learned": 0, "recency": 1}.get(order, 2)


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay: the four arms of each cell, the costlier arm and the
    larger L1 first. The order only schedules work; every table is sorted
    before it is written."""
    tasks = [(name, fraction, multiplier, arm, seed)
             for name in names for fraction, multiplier in cells
             for arm in cell_arms(fraction, multiplier) for seed in seeds]
    tasks.sort(key=lambda task: (arm_cost(task[3]), -task[1], -task[2], task[0], task[3],
                                 task[4]))
    return tasks


# --- checks ----------------------------------------------------------------------------------------

_where = mco._where


def check_reproduction(rows, published, expected: int) -> tuple[dict, list[dict]]:
    """The `label` replays against the published `leaf16` `label` row of
    their trace x cell x seed: `avoided_prefill_tokens` exactly, every counter
    column the published row carries exactly, and each digest of
    `DIGEST_COLUMNS` the published row carries; without a published counter
    digest, every column of `REQUIRED_COUNTER_COLUMNS` must have been
    compared. Annotates each such row and returns the report and one table
    row per replay. Every one of `expected` must be found and equal."""
    matched = seen = 0
    digests_compared = dict.fromkeys(DIGEST_COLUMNS, 0)
    columns_compared: set[int] = set()
    mismatches: list[str] = []
    missing: list[str] = []
    table: list[dict] = []
    for row in rows:
        if row["arm"] != ANCHOR_ARM:
            continue
        seen += 1
        reference = published.get(REPRODUCTION_REFERENCE + _cell_seed(row))
        if reference is None:
            row["reference_source"] = ""
            row["reference_avoided_prefill_tokens"] = ""
            row["reproduces_reference"] = ""
            missing.append(f"{_where(row)}: no published {ANCHOR_ARM}/{LEAF16} row")
            continue
        same = {"avoided_prefill_tokens":
                int(row["avoided_prefill_tokens"]) == int(reference["avoided_prefill_tokens"])}
        for column in DIGEST_COLUMNS:
            if reference.get(column, "") not in ("", None):
                digests_compared[column] += 1
                same[column] = str(row.get(column, "")) == reference[column]
        counters = reference.get("counters", {})
        differing = sorted(column for column, value in counters.items()
                           if column not in row or not _same_published_value(row[column], value))
        unpublished = ([] if "counters_sha256" in same
                       else [column for column in REQUIRED_COUNTER_COLUMNS
                             if column not in counters])
        columns_compared.add(len(counters))
        holds = all(same.values()) and not differing and not unpublished
        row["reference_source"] = reference["source"]
        row["reference_avoided_prefill_tokens"] = reference["avoided_prefill_tokens"]
        row["reproduces_reference"] = holds
        entry = {"trace": row["trace"], "l1_fraction": row["l1_fraction"],
                 "l2_multiplier": row["l2_multiplier"], "cell": row["cell"], "seed": row["seed"],
                 "arm": row["arm"], "published_arm": ANCHOR_ARM, "published_mechanism": LEAF16,
                 "reference_source": reference["source"],
                 "avoided_prefill_tokens": row["avoided_prefill_tokens"],
                 "published_avoided_prefill_tokens": reference["avoided_prefill_tokens"],
                 "same_avoided_prefill_tokens": same["avoided_prefill_tokens"],
                 "counter_columns_compared": len(counters),
                 "counter_columns_differing": ";".join(differing),
                 "required_counter_columns_unpublished": ";".join(unpublished)}
        for column in DIGEST_COLUMNS:
            entry[column] = row.get(column, "")
            entry[f"published_{column}"] = reference.get(column, "")
            entry[f"same_{column}"] = same.get(column, "")
        entry["reproduces"] = holds
        table.append(entry)
        if holds:
            matched += 1
        else:
            reasons = [column for column, value in same.items() if not value]
            if differing:
                reasons.append("counter columns " + ", ".join(differing))
            if unpublished:
                reasons.append("unpublished required counters " + ", ".join(unpublished))
            mismatches.append(f"{_where(row)} vs published {ANCHOR_ARM}/{LEAF16}: differs in "
                              + "; ".join(reasons))
    report = {"reference": f"{ANCHOR_ARM} ({LEAF16}, {REFERENCES[REPRODUCTION_REFERENCE]})",
              "expected": expected, "replays": seen, "matched": matched,
              "missing": len(missing), "mismatched": len(mismatches),
              "counter_digests_compared": digests_compared["counters_sha256"],
              "decision_digests_compared": digests_compared["decision_sha256"],
              "counter_columns_compared": sorted(columns_compared),
              "mismatches": missing + mismatches}
    return report, table


def reproduction_passes(report: dict) -> bool:
    return (report["expected"] > 0
            and report["matched"] == report["expected"] == report["replays"]
            and not report["mismatched"] and not report["missing"])


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed (the leaf16 references and the all16 rows beside them), on each
    identifier the reference publishes. Returns the pairs compared and the
    problems."""
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


def check_mechanism(rows) -> list[str]:
    """Every replay ran under leaf16 (`leaf`, width 16) as an arm of its cell,
    with the cell's h* (and as its parameter for the h* arms)."""
    problems = []
    for row in rows:
        where = _where(row)
        if (row["mechanism"], row["eligibility"], int(row["width"])) != (MECHANISM, ELIGIBILITY,
                                                                          WIDTH):
            problems.append(f"{where}: {row['mechanism']}/{row['eligibility']}/{row['width']}, "
                            f"the plan's mechanism is {MECHANISM}/{ELIGIBILITY}/{WIDTH}")
        cell = (float(row["l1_fraction"]), float(row["l2_multiplier"]))
        if row["arm"] not in cell_arms(*cell):
            problems.append(f"{where}: not an arm of its cell")
            continue
        h = matched_horizon(*cell)
        parameter = HORIZON_OF_ARM.get(row["arm"], "")
        if float(row["h_star"]) != h or (parameter != "" and float(row["arm_parameter"]) != h):
            problems.append(f"{where}: h* {row['h_star']} / parameter {row['arm_parameter']}, "
                            f"the cell's h* is {h:g} s")
    return problems


def check_statistics(rows) -> list[str]:
    """The horizon control's statistics check (every decision seen with its
    final victim; the `label` replays carry the label identities), and no
    override in an arm that has none (`label_binary_h*` at every h*, and
    `label`)."""
    problems = hc.check_statistics(rows)
    for row in rows:
        if (row["arm"] in RUNG_ARMS or row["arm"] == ANCHOR_ARM) and int(
                row["overridden_decisions_seen"]) != 0:
            problems.append(f"{_where(row)}: {row['overridden_decisions_seen']} overridden "
                            "decisions for an arm that must keep the store's choice")
    return problems


def check_class(rows) -> list[str]:
    """The matched class-order runner's class check on every replay of the
    three h* arms (class statistics at h*, every decision seen, no violation,
    no override outside a first round with the arrival a candidate, and at
    h* = 600 s the identity with `m4_count_resident`), which must carry the
    class columns."""
    carrying = []
    problems = []
    for row in rows:
        if row["arm"] not in CLASS_STATISTIC_ARMS:
            continue
        if any(column not in row or row[column] in ("", None) for column in CLASS_COLUMNS):
            problems.append(f"{_where(row)}: no class statistics")
            continue
        carrying.append(row)
    return problems + mco.check_class(carrying)


check_leaf_closure = hc.check_leaf_closure


# --- derivation ------------------------------------------------------------------------------------

_put_stats = hc._put_stats
_stats = rmc._stats
_put_signs = hc._put_signs
_difference_points = hc._difference_points
_identity = hc._identity
_is_nan = hc._is_nan
_u = hc._u
_reference = hc._reference
_cell_key = hc._cell_key
index_arms = hc.index_arms
POINT_FIELDS = mco.POINT_FIELDS
LEVEL_FIELDS = mco.LEVEL_FIELDS
# The seven inputs of every reading under one mechanism: four references and
# the three h* arms (replayed under leaf16, published under all16).
INPUTS = ("label", "lru", "learned", "evict_label", "label_binary_hstar", "matched_learned",
          "matched_recency")
# Seed-paired differences: name -> (later, earlier).
DIFFERENCES = {"label_minus_rung": ("label", "label_binary_hstar"),
               "learned_minus_recency": ("matched_learned", "matched_recency"),
               "label_binary_minus_recency": ("label_binary_hstar", "matched_recency")}
# The arms whose arrival-candidate share reading 4 reports.
SHARE_INPUTS = ("label_binary_hstar", "matched_recency")
MECHANISM_ORDER = (LEAF16, ALL16)


def _arm_index(arm: str) -> int:
    return (RUNG_ARMS + CLASS_ORDER_ARMS + (ANCHOR_ARM,)).index(arm)


def _cell_context(cell) -> dict:
    """h* and the arm names of one trace x cell, with the source of its
    published all16 rung."""
    h = matched_horizon(cell[1], cell[2])
    rung, learned_arm, recency_arm, _ = cell_arms(cell[1], cell[2])
    return {"h_star": h, "rung_arm": rung, "matched_learned_arm": learned_arm,
            "matched_recency_arm": recency_arm, "all16_rung_source": RUNG_SOURCES[h]}


def cell_inputs(cell, seeds, mechanism: str, published, by_seed=None) -> dict[str, list[dict]]:
    """The per-seed rows of the seven inputs of one trace x cell, in seed
    order: the four references published under `mechanism`; the three h* arms
    from this run's replays (`by_seed`, seed -> arm -> row) under leaf16 and
    from the published rows under all16."""
    context = _cell_context(cell)
    arms = {"label_binary_hstar": context["rung_arm"],
            "matched_learned": context["matched_learned_arm"],
            "matched_recency": context["matched_recency_arm"]}
    inputs = {name: [_reference(published, cell, seed, mechanism, name) for seed in seeds]
              for name in REFERENCE_ARMS}
    for name, arm in arms.items():
        if mechanism == LEAF16:
            missing = [seed for seed in seeds if arm not in by_seed[seed]]
            if missing:
                raise KeyError(f"{cell}: replays of {arm} missing for seeds {missing}")
            inputs[name] = [by_seed[seed][arm] for seed in seeds]
        elif mechanism == ALL16:
            inputs[name] = [_reference(published, cell, seed, ALL16, arm) for seed in seeds]
        else:
            raise ValueError(f"unknown mechanism {mechanism!r}")
    return inputs


def cell_values(inputs: dict[str, list[dict]]) -> dict:
    """Every reading's numbers for one trace x cell under one mechanism, from
    its per-seed inputs: U per seed and on five-seed means (`_stats`, as the
    parents' tables), S_h*, R_h* of both matched arms, the seed-paired
    differences in tokens and points with their seed signs and readings, and
    the arrival-candidate share of the rung and the recency arm per seed."""
    u = {name: [_u(entry) for entry in inputs[name]] for name in INPUTS}
    means = {name: _stats(values)["mean"] for name, values in u.items()}
    values: dict[str, object] = {"u": u, "means": means, "tokens": {}, "points": {}}
    for name, (later, earlier) in DIFFERENCES.items():
        pairs = [_difference_points(a, b) for a, b in zip(inputs[later], inputs[earlier])]
        values["tokens"][name] = [tokens for tokens, _ in pairs]
        values["points"][name] = [points for _, points in pairs]
        values[f"{name}_points_mean"] = _stats(values["points"][name])["mean"]
        values[f"{name}_seed_signs"] = rmc._sign_string(values["points"][name])
        values[f"{name}_reading"] = sign_reading(values["points"][name])
    values["shares"] = {name: [row_arrival_candidate_share(entry) for entry in inputs[name]]
                        for name in SHARE_INPUTS}
    values["S_hstar"] = hstar_shortfall(means["label"], means["label_binary_hstar"], means["lru"])
    values["horizon_reading"] = hstar_reading(values["S_hstar"])
    for name in ("matched_learned", "matched_recency"):
        values[f"R_{name}"] = class_recovery(means[name], means["learned"], means["evict_label"])
    values["class_reading"] = class_reading(values["R_matched_learned"])
    return values


def all16_values(published, cells=None, seeds=None) -> dict[tuple, dict]:
    """`cell_values` under all16 for every trace x cell of the grid (or the
    cells and seeds given), from the published rows alone."""
    cells = tuple(CELLS) if cells is None else tuple(cells)
    seeds = tuple(SEEDS) if seeds is None else tuple(seeds)
    return {(name,) + tuple(cell): cell_values(cell_inputs((name,) + tuple(cell), seeds, ALL16,
                                                           published))
            for name in TRACES for cell in cells}


def _put_mechanism(entry: dict, prefix: str, values: dict, names: tuple[str, ...],
                   differences: tuple[str, ...], scalars: tuple[str, ...], fields) -> None:
    """One mechanism's columns of a table row: U levels, differences with
    seed signs and the named scalars, each under `prefix`."""
    for name in names:
        _put_stats(entry, f"{prefix}U_{name}_points", values["u"][name], fields=LEVEL_FIELDS)
    for name in differences:
        _put_stats(entry, f"{prefix}{name}_points", values["points"][name], fields=fields)
        _put_signs(entry, f"{prefix}{name}", values["points"][name])
    for name in scalars:
        entry[f"{prefix}{name}"] = values[name]


def _put_shares(entry: dict, prefix: str, values: dict) -> None:
    for name in SHARE_INPUTS:
        shares = [share for share in values["shares"][name] if not _is_nan(share)]
        _put_stats(entry, f"{prefix}arrival_candidate_share_{name}", shares, fields=LEVEL_FIELDS)


def readings_tables(rows, published) -> dict[str, list[dict]]:
    """The four readings, from the replays and the published references, each
    beside the published all16 value of the same arm.

    Per seed and mechanism (`readings_seeds`): U of the seven inputs, the
    seed's own S and R (descriptive), the three seed-paired differences in
    tokens and points, and the arrival-candidate shares.

    Per trace x cell: `horizon` (reading 1: S_h* and its reading, the
    label-minus-rung difference with seed signs), `class_order` (reading 2:
    R_h* of both matched arms and the reading of the learned order),
    `order` (reading 3: learned order minus recency, seed-paired) and
    `admission` (reading 4: label_binary_h* minus the recency arm,
    seed-paired, with the share of decisions in which the arrival is a
    candidate); every leaf16 column has its all16 counterpart prefixed
    `all16_`.
    """
    seed_rows, horizon, class_order, order, admission = [], [], [], [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        context = _cell_context(cell)
        values = {LEAF16: cell_values(cell_inputs(cell, seeds, LEAF16, published, by_seed)),
                  ALL16: cell_values(cell_inputs(cell, seeds, ALL16, published))}
        for mechanism in MECHANISM_ORDER:
            mine = values[mechanism]
            for position, seed in enumerate(seeds):
                u = {name: mine["u"][name][position] for name in INPUTS}
                entry = {**identity, **context, "mechanism": mechanism, "seed": seed,
                         **{f"U_{name}_points": u[name] for name in INPUTS},
                         "S_seed": hstar_shortfall(u["label"], u["label_binary_hstar"], u["lru"]),
                         **{f"R_{name}_seed": class_recovery(u[name], u["learned"],
                                                             u["evict_label"])
                            for name in ("matched_learned", "matched_recency")}}
                for name in DIFFERENCES:
                    entry[f"{name}_tokens"] = mine["tokens"][name][position]
                    entry[f"{name}_points"] = mine["points"][name][position]
                for name in SHARE_INPUTS:
                    entry[f"arrival_candidate_share_{name}"] = mine["shares"][name][position]
                seed_rows.append(entry)
        leaf, base = values[LEAF16], values[ALL16]

        entry = {**identity, **context, "seeds": len(seeds)}
        for prefix, mine in (("", leaf), ("all16_", base)):
            entry[f"{prefix}U_label_points_mean"] = mine["means"]["label"]
            entry[f"{prefix}U_lru_points_mean"] = mine["means"]["lru"]
            _put_mechanism(entry, prefix, mine, ("label_binary_hstar",), ("label_minus_rung",),
                           ("S_hstar",), POINT_FIELDS)
            entry[f"{prefix}reading"] = mine["horizon_reading"]
        entry["predicted_reading"] = PREDICTIONS["1_horizon_under_leaf_eligibility"][0]
        horizon.append(entry)

        entry = {**identity, **context, "seeds": len(seeds)}
        for prefix, mine in (("", leaf), ("all16_", base)):
            _put_mechanism(entry, prefix, mine,
                           ("learned", "evict_label", "matched_learned", "matched_recency"), (),
                           ("R_matched_learned", "R_matched_recency"), POINT_FIELDS)
            entry[f"{prefix}reading"] = mine["class_reading"]
        covered = predicted("2_class_order_under_leaf_eligibility", cell[1], cell[2])
        entry["predicted"] = covered
        entry["predicted_reading"] = (PREDICTIONS["2_class_order_under_leaf_eligibility"][0]
                                      if covered else "")
        class_order.append(entry)

        entry = {**identity, **context, "seeds": len(seeds)}
        for prefix, mine in (("", leaf), ("all16_", base)):
            _put_mechanism(entry, prefix, mine, (), ("learned_minus_recency",),
                           ("R_matched_learned", "R_matched_recency"), POINT_FIELDS)
        entry["predicted_reading"] = PREDICTIONS["3_order_within_matched_class"][0]
        order.append(entry)

        entry = {**identity, **context, "seeds": len(seeds)}
        for prefix, mine in (("", leaf), ("all16_", base)):
            _put_mechanism(entry, prefix, mine, ("label_binary_hstar", "matched_recency"),
                           ("label_binary_minus_recency",), (), POINT_FIELDS)
            _put_shares(entry, prefix, mine)
        admission.append(entry)
    return {"readings_seeds": seed_rows, "horizon": horizon, "class_order": class_order,
            "order": order, "admission": admission}


def reading_summary(tables: dict[str, list[dict]]) -> list[dict]:
    """The counts out of the trace x cell of the run, leaf16 beside all16:
    reading 1 (S_h* <= 0.10), reading 2 (R_h* of the learned order >= 0.9)
    and reading 3 (consistent loss of the learned order) against the
    predictions fixed before the run; reading 4 (descriptive) with no
    prediction."""
    out = []
    specifications = (
        ("1_horizon_under_leaf_eligibility", "horizon", "reading", HORIZON_READINGS),
        ("2_class_order_under_leaf_eligibility", "class_order", "reading", CLASS_READINGS),
        ("3_order_within_matched_class", "order", "learned_minus_recency_reading", SIGN_READINGS),
        ("4_admission_descriptive", "admission", "label_binary_minus_recency_reading",
         SIGN_READINGS),
    )
    for reading, table, column, labels in specifications:
        entries = tables[table]
        counts = outcome_counts((entry[column] for entry in entries), labels)
        base = outcome_counts((entry[f"all16_{column}"] for entry in entries), labels)
        entry = {"reading": reading, "cells": len(entries)}
        if reading in PREDICTIONS:
            outcome, required = PREDICTIONS[reading]
            holds = prediction_holds(reading, counts[outcome], len(entries))
            entry.update(counted=outcome, count=counts[outcome], required=required,
                         of=GRID_TRACE_CELLS, prediction_holds="" if holds is None else holds,
                         all16_count=base[outcome])
        else:
            entry.update(counted="", count="", required="", of="", prediction_holds="",
                         all16_count="")
        entry.update({label: counts[label] for label in labels})
        entry.update({f"all16_{label}": base[label] for label in labels})
        out.append(entry)
    return out


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x
    arm, of every replay metric and (for the h* arms) class count."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _arm_index(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH, "arm": arm,
                 "family": FAMILY_OF_ARM[arm], "arm_parameter": HORIZON_OF_ARM.get(arm, ""),
                 "h_star": matched_horizon(key[1], key[2]), "order": ORDER_OF_ARM.get(arm, ""),
                 "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        metrics = REPLAY_METRICS + (CLASS_METRICS if arm in CLASS_STATISTIC_ARMS else ())
        for metric in metrics:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        shares = [row_arrival_candidate_share(member) for member in members]
        _put_stats(entry, "arrival_candidate_share", [share for share in shares
                                                      if not _is_nan(share)])
        out.append(entry)
    return out


def reference_rows(published, rows) -> list[dict]:
    """The published reference rows of every trace x cell x seed of the run,
    with their U in points: the inputs of the readings, as they were read
    (the reproduction target's counter columns are in `reproduction.csv`)."""
    out = []
    for cell_seed in sorted({_cell_seed(row) for row in rows}):
        for (mechanism, arm), source in REFERENCES.items():
            if cell_seed[1:3] not in reference_cells(mechanism, arm):
                continue
            entry = {key: value for key, value in published[(mechanism, arm) + cell_seed].items()
                     if key != "counters"}
            entry["extra_points"] = _u(entry)
            out.append(entry)
    return out


# --- README ----------------------------------------------------------------------------------------

README_TEXT = """# The matched-horizon bit and the class order under leaf eligibility

Pre-registration: `docs/leaf-matched-horizon-plan.md`. A control on eligibility: the arms that the horizon fill-in (`results/paper/horizon_fill_001/`) and the matched class-order control (`results/paper/matched_class_order_001/`) ran under `all16` at the cell's matched horizon `h*`, rerun under the horizon control's Part 3 mechanism `leaf16` (the arrival, if it is a leaf, plus up to 16 uniformly sampled leaf residents; a resident with a cached child is never a candidate), with `h*` carried over from `all16` unchanged: 0.25% x 1: 60 s; 0.25% x 4: 150 s; 1% x 1: 150 s; 1% x 4: 600 s; 2% x 1: 300 s; 2% x 4: 600 s. Arms per trace x cell: `label_binary_{h*}` (key `(1 if the next use is at most h* s away else 0, last_group)`, built as the horizon fill-in builds it), `evict_binary_{h*}_learned` and `evict_binary_{h*}_recency` (admission by the frozen pi0 `next_use` ranker: in a first round in which the arrival is a candidate it is rejected iff it is the first minimum of the ranker's keys; otherwise the victim is the first minimum over the candidates other than the arrival of `(reusable within h* s, ranker score, last_group)` or `(reusable within h* s, last_group)`; every other round uses that key; built as the matched class-order control builds them), and `label` (the exact `next_use` label rung, the reproduction anchor). Under `leaf16` a non-leaf arrival is not a candidate of its own first round, so the ranker's admission rule judges leaf arrivals only. 2 traces x 6 cells x 4 arms x 5 seeds = 240 replays, each with the Phase 0.98b attribution hooks, the error-location decision statistics at the trace's 600-second label horizon and, for the three `h*` arms, the class statistics at `h*` (read-only).

References are published rows, not reruns: `lru`, `learned`, `label` under `leaf16` from `results/paper/mechanism_control_001/replay_seeds.csv` (arm column `rung`) and `evict_label` under `leaf16` from `results/paper/horizon_control_001/replay_seeds.csv`. Every value is reported beside the published `all16` value of the same arm, computed from published rows: `lru`, `learned`, `label`, `evict_label` from `results/paper/error_location_001/`, `label_binary_{h*}` from `results/paper/horizon_control_001/` (60, 300, 600 s) or `results/paper/horizon_fill_001/` (150 s), and the matched arms from `results/paper/matched_class_order_001/`; before any replay those `all16` values were checked against the published `all16` tables (`horizon.csv` / `fill.csv` S at h*; `class_order.csv`, `order.csv`, `admission.csv` of the matched class-order control). U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens.

Reproduction. The plan holds the 60 `label` replays to the published `leaf16` `label` rows in `avoided_prefill_tokens` and the counter digest. Those published rows (mechanism_control_001) carry no counter digest and no decision digest; each `label` replay was compared exactly on `avoided_prefill_tokens` and on every counter column the published row carries (every column except identity, timing, memory and annotation columns: the replay counters, the attribution and orphaning counters and the point columns derived from them), and on any digest the published row carries; this run's own `counters_sha256` and `decision_sha256` are recorded in `reproduction.csv`.

Boundaries. `h*` is the `all16` best of grids run on these traces; under `leaf16` the best horizon may differ, and nothing here looks for it. `leaf16` is a control on eligibility, not a proposed mechanism. Every arm reads the trace's future on purpose and none is a policy. `evict_label` is the exact `next_use` label applied greedily on eviction, the comparison the parents used, not an optimum. Seed intervals describe sampling-seed variability only.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated), `h_star` (seconds), `rung_arm` (`label_binary_{h*}`), `matched_learned_arm`, `matched_recency_arm`, `all16_rung_source` (the published run of the all16 rung). Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise). A column prefixed `all16_` is the same quantity under `all16`, from the published rows.

- `replay_seeds.csv`: one row per replay, the horizon control's columns (`eligibility` leaf, `width` 16, `mechanism` leaf16, `arm`, `family` (horizon / class_order / reproduction_anchor), `arm_parameter` (h* for the three h* arms), `variant` main, the replay counters, timing and worker memory, `counters_sha256`, the Phase 0.98b attribution columns, the point columns, the error-location decision statistics at the 600-second label horizon: `stat_*`, `overridden_*`, `decision_sha256`, `m1` .. `m4_victim_at_horizon`, with `stat_decisions_admission` the window decisions in which the arrival is a candidate), plus `h_star`, `order` (learned / recency for the class-order arms), the class statistics at h* for the three h* arms (`class_horizon_seconds`, `class_decisions`, `class_rejections`, `class_resident_evictions`, `class_violations` (resident evictions whose victim is reusable within h* while some sampled resident is not), `class_overridden`, `class_overridden_outside_admission`, each over the evaluation window and with `_seen` over every decision; `class_m4_count_resident`), and, for `label`, `reference_source`, `reference_avoided_prefill_tokens`, `reproduces_reference`.
- `replay.csv`: per trace x cell x arm, `mechanism`, `eligibility`, `width`, `family`, `arm_parameter`, `h_star`, `order`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, mean / std / ci95_half / min / max of every replay metric and statistic (class counts for the h* arms), and `arrival_candidate_share_*` (`stat_decisions_admission / stat_decisions` per replay).
- `references_seeds.csv`: the published reference rows the readings use, one per trace x cell x seed x reference: `source`, `mechanism`, `arm`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`), the digests and `stat_decisions` / `stat_decisions_admission` where published, and `extra_points` (U).
- `readings_seeds.csv`: per trace x cell x seed x `mechanism` (leaf16 / all16): `U_<input>_points` for label, lru, learned, evict_label, label_binary_hstar, matched_learned, matched_recency; `S_seed`, `R_matched_learned_seed`, `R_matched_recency_seed` (the seed's own, descriptive); `_tokens` / `_points` of `label_minus_rung` (U(label) - U(label_binary_h*)), `learned_minus_recency` (U(evict_binary_h*_learned) - U(evict_binary_h*_recency)) and `label_binary_minus_recency` (U(label_binary_h*) - U(evict_binary_h*_recency)); `arrival_candidate_share_label_binary_hstar` and `_matched_recency`.
- `horizon.csv` (reading 1): per trace x cell, `U_label_points_mean`, `U_lru_points_mean` (leaf16 rungs), `U_label_binary_hstar_points_*`, `label_minus_rung_points_*` with its seed signs, `S_hstar` = (U(label) - U(label_binary_h*)) / (U(label) - U(lru)) on the five-seed means (nan when the denominator is zero), `reading` (reuse_label_at_matched_horizon_suffices when S_hstar <= 0.10, exactly 0.10 sufficing; reuse_label_at_matched_horizon_falls_short otherwise, a nan included), `predicted_reading`, and the `all16_` counterparts (all16 S_hstar is the published S of the cell's h*).
- `class_order.csv` (reading 2): per trace x cell, `U_<name>_points_mean` / `_min` / `_max` for learned, evict_label (leaf16 references), matched_learned, matched_recency; `R_matched_learned`, `R_matched_recency` = (U(a) - U(learned)) / (U(evict_label) - U(learned)) on the five-seed means with the leaf16 references; `reading` (reuse_identification_at_matched_horizon_suffices when R_matched_learned >= 0.9, exactly 0.9 sufficing; ranker_order_costs_at_matched_horizon otherwise, a nan included); `predicted` (False at 0.25% x 1, whose reading the plan does not predict) and `predicted_reading`; and the `all16_` counterparts (the published R and reading of the matched class-order control).
- `order.csv` (reading 3): per trace x cell, `learned_minus_recency_points_*` (seed-paired; mean, ci95_half, min, max) with its seed signs (`learned_minus_recency_*`), `R_matched_learned`, `R_matched_recency` beside it, `predicted_reading` (consistent_loss), and the `all16_` counterparts.
- `admission.csv` (reading 4, descriptive): per trace x cell, `U_label_binary_hstar_points_*`, `U_matched_recency_points_*`, `label_binary_minus_recency_points_*` with its seed signs (the two arms evict by the same key and differ in who decides rejection), `arrival_candidate_share_label_binary_hstar_*` and `arrival_candidate_share_matched_recency_*` (mean / min / max over seeds of `stat_decisions_admission / stat_decisions`, the share of evaluation-window decisions in which the arrival is a candidate), and the `all16_` counterparts.
- `readings.csv`: one row per reading: `cells` (trace x cell of the run), `counted` (the predicted outcome), `count`, `required` and `of` (the prediction: 12 of 12 for reading 1, at least 10 of 12 for reading 2, at least 8 of 12 for reading 3), `prediction_holds` (empty unless the run covers the 12 trace x cell), `all16_count`, and the count of every outcome under leaf16 and (`all16_`) all16. Reading 4 has no prediction.
- `reproduction.csv`: one row per `label` replay: `published_arm` label, `published_mechanism` leaf16, `reference_source`, `avoided_prefill_tokens` with `published_` and `same_`, `counter_columns_compared` (the counter columns the published row carries), `counter_columns_differing`, `required_counter_columns_unpublished`, this run's `counters_sha256` and `decision_sha256` with the published value and `same_` (empty: not published), and `reproduces`.
- `run_config.json`: plan and code commits, source manifest, trace, model, model-manifest and reference hashes (every table read, the published all16 tables included), the mechanism, the matched horizons, grid, arms, check results, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


_report_lines = mco._report_lines


def _number(value) -> str:
    return f"{value:7.3f}" if isinstance(value, float) else str(value)


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
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace {trace.name}")
        horizon, split_ms, _ = horizon_for(trace)
        if horizon != LABEL_HORIZON_SECONDS:
            raise SystemExit(f"{trace.name}: horizon {horizon} s, the plan fixes "
                             f"{LABEL_HORIZON_SECONDS} s")
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = split_ms
        horizons[trace.name] = horizon
        working_set[trace.name] = working_set_bytes(trace)
        trace_files[trace.name] = {"path": str(path.resolve()), "sha256": sha256_path(path)}
        print(f"loaded {trace.name}: requests={len(trace.requests)} H={horizon:g}s "
              f"split={split_ms:.1f}ms", flush=True)
    if set(traces) != set(TRACES):
        raise SystemExit(f"traces {sorted(traces)} do not match the grid {sorted(TRACES)}")
    names = tuple(sorted(traces))
    rankers, model_files = rmc.load_models(names)
    for name in names:
        print(f"  pi0 next_use {name}: {model_files[name]['path']} "
              f"sha256={model_files[name]['sha256'][:12]}", flush=True)
    published, reference_problems = load_references()
    published_all16, all16_problems = load_published_all16()
    reference_hashes = {_repository_path(path): sha256_path(path)
                        for path in list(REFERENCE_SOURCES.values())
                        + list(PUBLISHED_TABLES.values())}
    print(f"  published references: {len(published)} rows of "
          f"{', '.join(f'{arm}/{mechanism}' for mechanism, arm in REFERENCES)}", flush=True)
    for path, digest in reference_hashes.items():
        print(f"    {path} sha256={digest[:12]}", flush=True)
    if reference_problems or all16_problems:
        _report_lines("REFERENCE", reference_problems + all16_problems)
        raise SystemExit("published references are not what the plan names; nothing run")
    # The all16 values printed beside every leaf16 value must be the published ones.
    recomputed_all16 = all16_values(published, cells, seeds)
    all16_mismatches = check_all16(recomputed_all16, published_all16)
    print(f"  all16 values recomputed from the published rows against the published all16 "
          f"tables ({len(published_all16)} trace x cell, {len(ALL16_CHECKS)} values each): "
          f"{'equal' if not all16_mismatches else 'DIFFER'}", flush=True)
    if all16_mismatches:
        _report_lines("ALL16", all16_mismatches)
        raise SystemExit("the all16 values are not the published ones; nothing run")

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                  rankers=rankers)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds)
    workers = min(args.workers, len(tasks))
    print(f"{len(tasks)} replays on {workers} workers ({len(names)} traces x {len(cells)} cells x "
          f"{ARMS_PER_CELL} arms x {len(seeds)} seeds) under {MECHANISM} "
          f"(l2_eligibility={ELIGIBILITY}, width {WIDTH}); matched horizons "
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
        timing[(_arm_index(row["arm"]), row["arm"])].append(float(row["seconds"]))
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
    expected_reproductions = sum(1 for task in tasks if task[3] == ANCHOR_ARM)
    reproduction, reproduction_table = check_reproduction(rows, published, expected_reproductions)
    print(f"  reproduction ({ANCHOR_ARM} vs published {reproduction['reference']}; "
          f"avoided_prefill_tokens and every published counter column "
          f"({'/'.join(str(n) for n in reproduction['counter_columns_compared']) or 'none'} "
          f"columns per row); {reproduction['counter_digests_compared']} counter and "
          f"{reproduction['decision_digests_compared']} decision digests published): "
          f"{reproduction['matched']} matched, {reproduction['missing']} missing, "
          f"{reproduction['mismatched']} mismatched (expected {reproduction['expected']})",
          flush=True)
    _report_lines("MISMATCH", reproduction["mismatches"], limit=len(reproduction["mismatches"]))
    if not reproduction_passes(reproduction):
        failures.append(f"the {ANCHOR_ARM} replays do not reproduce the published {LEAF16} "
                        f"{ANCHOR_ARM} rows")
    compared, identifier_problems = check_identifiers(rows, published)
    print(f"  identifiers ({', '.join(REFERENCE_IDENTIFIERS)}) of {len(rows)} replays against "
          f"{compared} published reference rows of their trace x cell x seed: "
          f"{'equal' if not identifier_problems else 'DIFFER'}", flush=True)
    _report_lines("IDENTIFIER", identifier_problems)
    if identifier_problems:
        failures.append("an identifier differs from a published reference row")
    mechanism_problems = check_mechanism(rows)
    print(f"  mechanism: every replay under {MECHANISM} ({ELIGIBILITY}, width {WIDTH}) as an arm "
          f"of its cell at its h*: {'yes' if not mechanism_problems else 'NO'}", flush=True)
    _report_lines("MECHANISM", mechanism_problems)
    if mechanism_problems:
        failures.append("a replay is not under the plan's mechanism or arm")
    closure_problems = check_leaf_closure(rows)
    print(f"  leaf16 closure: present-but-unusable tokens in {len(rows)} replays: "
          f"{'none' if not closure_problems else 'FOUND'}", flush=True)
    _report_lines("CLOSURE", closure_problems)
    if closure_problems:
        failures.append("a leaf16 replay left a block present but unusable")
    statistics_problems = check_statistics(rows)
    label_rows = sum(1 for row in rows if row["arm"] == ANCHOR_ARM)
    print(f"  statistics: every decision seen with its final victim, no override where none is "
          f"allowed, label identities in {label_rows} {ANCHOR_ARM} replays: "
          f"{'hold' if not statistics_problems else 'BROKEN'}", flush=True)
    _report_lines("STATISTICS", statistics_problems)
    if statistics_problems:
        failures.append("a statistics check failed")
    class_problems = check_class(rows)
    class_rows = [row for row in rows if row["arm"] in CLASS_STATISTIC_ARMS]
    violations = sum(int(row.get("class_violations_seen", 0) or 0) for row in class_rows)
    outside = sum(int(row.get("class_overridden_outside_admission_seen", 0) or 0)
                  for row in class_rows)
    print(f"  class at h* ({len(class_rows)} replays of the three h* arms): resident evictions "
          f"discarding a state reusable within h* while a sampled resident is not: {violations}; "
          f"overridden decisions outside a first round with the arrival a candidate: {outside}; "
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
                               _arm_index(row["arm"]), row["seed"]))
    reproduction_table.sort(key=lambda entry: (entry["trace"], entry["l1_fraction"],
                                               entry["l2_multiplier"], entry["seed"]))
    replay_summary = aggregate_replays(rows)
    references = reference_rows(published, rows)
    tables = readings_tables(rows, published)
    summary = reading_summary(tables)

    print("  reading 1, horizon under leaf eligibility (five-seed means):", flush=True)
    for entry in tables["horizon"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"S_h*={_number(entry['S_hstar'])} -> {entry['reading']} "
              f"(all16: {_number(entry['all16_S_hstar'])}, {entry['all16_reading']})", flush=True)
    print("  reading 2, class order under leaf eligibility:", flush=True)
    for entry in tables["class_order"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"R(learned order)={_number(entry['R_matched_learned'])} "
              f"R(recency)={_number(entry['R_matched_recency'])} -> {entry['reading']}"
              f"{'' if entry['predicted'] else ' (not predicted)'} "
              f"(all16: {_number(entry['all16_R_matched_learned'])}, "
              f"{_number(entry['all16_R_matched_recency'])}, {entry['all16_reading']})",
              flush=True)
    for title, table, prefix in (
            ("reading 3, order within the matched class (learned - recency)", tables["order"],
             "learned_minus_recency"),
            ("reading 4, admission (label_binary_h* - recency arm), descriptive",
             tables["admission"], "label_binary_minus_recency")):
        print(f"  {title}, points:", flush=True)
        for entry in table:
            share = ""
            if prefix == "label_binary_minus_recency":
                share = (f"; arrival a candidate in "
                         f"{entry['arrival_candidate_share_matched_recency_mean']:.3f} of the "
                         f"recency arm's decisions (all16 "
                         f"{entry['all16_arrival_candidate_share_matched_recency_mean']:.3f})")
            print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
                  f"{entry[f'{prefix}_points_mean']:7.3f} ({entry[f'{prefix}_seed_signs']}, "
                  f"{entry[f'{prefix}_reading']}; all16 "
                  f"{entry[f'all16_{prefix}_points_mean']:7.3f} "
                  f"{entry[f'all16_{prefix}_seed_signs']}, {entry[f'all16_{prefix}_reading']})"
                  f"{share}", flush=True)
    for entry in summary:
        if entry["counted"]:
            holds = entry["prediction_holds"]
            print(f"    {entry['reading']}: {entry['counted']} {entry['count']}/{entry['cells']} "
                  f"(predicted at least {entry['required']}/{entry['of']}: "
                  f"{'holds' if holds is True else 'FAILS' if holds is False else 'not evaluated'}"
                  f"; all16 {entry['all16_count']}/{entry['cells']})", flush=True)
        else:
            print(f"    {entry['reading']}: " + ", ".join(
                f"{label} {entry[label]}/{entry['cells']}" for label in SIGN_READINGS)
                + "; all16 " + ", ".join(f"{label} {entry[f'all16_{label}']}"
                                         for label in SIGN_READINGS), flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "leaf_matched_horizon",
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
        "reference_rows": {f"{arm}/{mechanism}": {
            "source": source,
            "cells": [list(cell) for cell in reference_cells(mechanism, arm)]}
            for (mechanism, arm), source in REFERENCES.items()},
        "reference_identifiers": list(REFERENCE_IDENTIFIERS),
        "published_all16_tables": {"tables": list(PUBLISHED_TABLES),
                                   "shortfall_table_of_h_star": {f"{h:g}": table for h, table
                                                                 in SHORTFALL_TABLES.items()},
                                   "values": [list(check) for check in ALL16_CHECKS],
                                   "tolerance": ALL16_TOLERANCE},
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "mechanism": {"name": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH,
                      "applied_as": "run_two_tier(l2_eligibility='leaf', l2_sample_width=16); "
                                    "every arm is its parent's construction, unchanged"},
        "matched_horizon_seconds": [{"l1_fraction": cell[0], "l2_multiplier": cell[1],
                                     "h_star": matched_horizon(*cell),
                                     "arms": list(cell_arms(*cell)),
                                     "all16_rung_source": RUNG_SOURCES[matched_horizon(*cell)]}
                                    for cell in cells],
        "arm_builders": {"label_binary_h": "horizonfill.arm_setup",
                         "evict_binary_h_learned/recency": "matchedorder.arm_setup",
                         ANCHOR_ARM: "errorloc.arm_setup"},
        "class_statistic_arms": "label_binary_{h*}, evict_binary_{h*}_learned, "
                                "evict_binary_{h*}_recency",
        "reproduction": {"arm": ANCHOR_ARM, "against": f"{ANCHOR_ARM}/{LEAF16}",
                         "source": REFERENCES[REPRODUCTION_REFERENCE],
                         "compared": "avoided_prefill_tokens, every counter column the published "
                                     "row carries, and each published digest",
                         "skipped_columns": sorted(REPRODUCTION_SKIP),
                         "required_counter_columns_without_digest": list(REQUIRED_COUNTER_COLUMNS)},
        "readings": {"suffices_shortfall": SUFFICES_SHORTFALL,
                     "class_order_share": CLASS_ORDER_SHARE,
                     "horizon_labels": list(HORIZON_READINGS),
                     "class_labels": list(CLASS_READINGS), "sign_readings": list(SIGN_READINGS),
                     "predictions": {reading: {"counted": outcome, "at_least": required,
                                               "of": GRID_TRACE_CELLS}
                                     for reading, (outcome, required) in PREDICTIONS.items()},
                     "not_predicted_cells_of_reading_2": [list(cell)
                                                          for cell in NOT_PREDICTED_CELLS]},
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "statistics_horizon_seconds": {"decision_statistics": "trace label horizon (600 s)",
                                       "class_statistics": "h* of the cell"},
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {k: v for k, v in reproduction.items() if k != "mismatches"},
            "all16_values_checked": len(published_all16) * len(ALL16_CHECKS),
            "all16_mismatches": len(all16_mismatches),
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "mechanism_problems": len(mechanism_problems),
            "leaf16_replays": len(rows), "leaf16_closure_problems": len(closure_problems),
            "statistics_problems": len(statistics_problems),
            "label_identity_replays": label_rows,
            "class_replays": len(class_rows),
            "class_problems": len(class_problems),
            "class_violations": violations,
            "class_overridden_outside_admission": outside,
            "invariant_groups": cell_seeds, "invariant_violations": len(invariant_problems),
            "invariant_columns": list(rmc.INVARIANT_COLUMNS),
            "unexplained_absent_tokens": unexplained,
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
        "note": "A control on eligibility: the matched-horizon arms of the horizon fill-in and "
                "the matched class-order control rerun under leaf16 with h* carried over from "
                "all16 unchanged (the all16 best of grids run on these traces, read after the "
                "fact; the leaf16 best may differ and is not looked for). Every value is "
                "reported beside the published all16 value of the same arm. The published "
                "leaf16 label rows carry no counter digest; the label replays were compared on "
                "every counter column they publish. Nothing is fitted; every arm reads the "
                "trace's future on purpose and none is a policy. References are published rows, "
                "not reruns. Seed intervals describe sampling-seed variability only.",
    }
    for directory in (args.output_dir, args.paper_dir):
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        for name in ("readings_seeds", "horizon", "class_order", "order", "admission"):
            _write(directory / f"{name}.csv", tables[name])
        _write(directory / "readings.csv", summary)
        _write(directory / "reproduction.csv", reproduction_table)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {args.output_dir}, {args.paper_dir}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
