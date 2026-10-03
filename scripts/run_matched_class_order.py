#!/usr/bin/env python3
"""Matched class order: the frozen ranker given the exact reuse class at the horizon that works for the cell.

The pre-registration is `docs/matched-class-order-plan.md`; nothing here may
change it. Part 2 of the horizon and class-order control is replayed with one
parameter changed (`persistent_kv_admission.matchedorder`): at every trace x
cell, the two class-order arms are built exactly as `horizonctl` builds
`evict_binary_learned` and `evict_binary_recency`, with the exact reuse label
at the cell's matched horizon `h*` (60, 150, 300 or 600 s, the plan's table)
in place of the 600-second one:

* `evict_binary_{h*}_learned`: admission by the frozen pi0 `next_use` ranker
  (the arrival is rejected iff it is the first minimum of the ranker's keys);
  otherwise, and in every later round, eviction by
  `(reusable within h* s, ranker score, last_group)`;
* `evict_binary_{h*}_recency`: the same admission; eviction by
  `(reusable within h* s, last_group)`.

Mechanism `all16`, 2 traces x 6 cells x 2 arms x 5 seeds = 120 replays, each
with the Phase 0.98b attribution hooks and the error-location decision
statistics (at the trace's 600-second label horizon, as in Part 2) attached
read-only, plus `matchedorder.ClassStatistics` at `h*`. At the cells with
`h*` = 600 s the arms are the published ones under another name and their
replays are a reproduction check.

The replay call, the row schema, the provenance rules and the table helpers are
the horizon control's (`scripts/run_horizon_control.py`, imported by path,
unchanged); the frozen ranker is loaded exactly as there. References are
published rows, not reruns: `lru`, `learned`, `label` and `evict_label` under
`all16` from `results/paper/error_location_001/replay_seeds.csv`; the label
rung of the cell's `h*`, `label_binary_{h*}`, from
`results/paper/horizon_control_001/replay_seeds.csv` (60, 300, 600 s) or
`results/paper/horizon_fill_001/replay_seeds.csv` (150 s); and the published
`evict_binary_learned` / `evict_binary_recency` of every trace x cell from
`results/paper/horizon_control_001/replay_seeds.csv`.

Nothing is fitted and no arm is a proposed policy: every arm reads the trace's
future on purpose. Before anything is derived or published the run checks that
the replays at `h*` = 600 s equal the published class-order rows in
`avoided_prefill_tokens` and `counters_sha256` (and `decision_sha256`, which
both carry); that the identifiers of every replay equal those of every
published reference row of its trace x cell x seed; that the statistics saw
every decision with its final victim; that in every replay no resident
eviction discards a state reusable within `h*` while a sampled resident is
not, and every overridden decision is a first-round decision in which the
arrival is a candidate; that the arm-independent counters are arm-independent;
that no absent token is unexplained; and (per replay) the attribution
identities. Any failure exits without writing a derived table, and nothing
reaches the paper directory.

Outputs (in --output-dir and in --paper-dir): per-seed and aggregated replay
tables, the published reference rows used, the four readings per seed and per
trace x cell, the reproduction table, the reading counts, a README describing
every table, and the run configuration. There is no smoke mode and no
dirty-tree mode: the plan runs once, from a clean tree at the code commit.
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
    MECHANISMS,
    class_order_reading,
)
from persistent_kv_admission.matchedorder import (
    CLASS_COLUMNS,
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    MATCHED_READINGS,
    MECHANISM,
    ORDER_OF_ARM,
    ORDERS,
    PUBLISHED_ARMS,
    PUBLISHED_HORIZON_SECONDS,
    ROLES,
    RUNG_OF_HORIZON,
    SIGN_READINGS,
    ClassStatistics,
    arm_setup,
    cell_arms,
    matched_arm,
    matched_horizon,
    matched_reading,
    matched_recovery,
    prediction_holds,
    reading_counts,
    role_of_horizon,
    sign_reading_counts,
)
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

HORIZON_CONTROL_SCRIPT = REPOSITORY / "scripts/run_horizon_control.py"
PLAN_PATH = REPOSITORY / "docs/matched-class-order-plan.md"


def _load_horizon_control():
    """Import the horizon-control runner by path, unchanged, for its replay
    constants, provenance, check and table helpers; it brings the
    error-location, mechanism-control and Phase 0.97 runners with it (one
    instance of each, so the capacity rule reads the working set this run
    sets, and the ranker loader is the one it uses). `main` is behind the
    usual guard, so nothing runs, and nothing is read at import."""
    spec = importlib.util.spec_from_file_location("run_horizon_control", HORIZON_CONTROL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hc = _load_horizon_control()
rel = hc.rel
rmc = hc.rmc
rdp = hc.rdp

# --- the fixed grid (docs/matched-class-order-plan.md) --------------------------------------------

TRACES = hc.TRACES
CELLS = hc.CELLS
SEEDS = hc.SEEDS
assert set(CELLS) == set(MATCHED_HORIZON_SECONDS)
LABEL_HORIZON_SECONDS = hc.LABEL_HORIZON_SECONDS
assert LABEL_HORIZON_SECONDS == PUBLISHED_HORIZON_SECONDS == 600.0
assert MECHANISM == ALL16
ELIGIBILITY, WIDTH = MECHANISMS[MECHANISM]
# Every replay carries the statistics hooks; there is no hook-off variant here.
VARIANT = "main"
FAMILY = "class_order"
DEFAULT_WORKERS = hc.DEFAULT_WORKERS
MAX_WORKERS = hc.MAX_WORKERS
assert (DEFAULT_WORKERS, MAX_WORKERS) == (10, 12)
# The plan's publication directory (a new one; refused if it exists).
DEFAULT_PAPER_DIR = REPOSITORY / "results/paper/matched_class_order_001"
# Published references, anchored at the repository (tracked files).
ERROR_LOCATION_REFERENCE = hc.ERROR_LOCATION_REFERENCE
HORIZON_CONTROL_REFERENCE = REPOSITORY / "results/paper/horizon_control_001/replay_seeds.csv"
HORIZON_FILL_REFERENCE = REPOSITORY / "results/paper/horizon_fill_001/replay_seeds.csv"
REFERENCE_SOURCES = {"error_location_001": ERROR_LOCATION_REFERENCE,
                     "horizon_control_001": HORIZON_CONTROL_REFERENCE,
                     "horizon_fill_001": HORIZON_FILL_REFERENCE}
# The label rung of each matched horizon and the only source it is read from.
RUNG_SOURCES = {60.0: "horizon_control_001", 150.0: "horizon_fill_001",
                300.0: "horizon_control_001", 600.0: "horizon_control_001"}
assert set(RUNG_SOURCES) == set(MATCHED_HORIZONS_SECONDS)
RUNG_ARMS = {RUNG_OF_HORIZON[h]: h for h in MATCHED_HORIZONS_SECONDS}
# Published reference rows, by (mechanism, arm): the source each is read from.
# A rung is read at the cells whose h* is its horizon only; the others at
# every cell of the grid.
REFERENCES = {
    (MECHANISM, "lru"): "error_location_001",
    (MECHANISM, "learned"): "error_location_001",
    (MECHANISM, "label"): "error_location_001",
    (MECHANISM, "evict_label"): "error_location_001",
    **{(MECHANISM, RUNG_OF_HORIZON[h]): RUNG_SOURCES[h] for h in MATCHED_HORIZONS_SECONDS},
    **{(MECHANISM, published): "horizon_control_001" for published in PUBLISHED_ARMS.values()},
}
REFERENCE_IDENTIFIERS = hc.REFERENCE_IDENTIFIERS
# The reproduction compares these exactly; `decision_sha256` where both rows
# carry one (the published class-order rows do).
REPRODUCTION_COLUMNS = ("avoided_prefill_tokens", "counters_sha256")
REPRODUCTION_OPTIONAL_COLUMNS = ("decision_sha256",)
DIGEST_COLUMNS = ("counters_sha256", "decision_sha256")
REPLAY_METRICS = hc.REPLAY_METRICS
CLASS_METRICS = tuple(column for column in CLASS_COLUMNS if column != "class_horizon_seconds")
ARMS_PER_CELL = len(ORDERS)
SHARED: dict[str, object] = {}


def reference_cells(mechanism: str, arm: str) -> tuple[tuple[float, float], ...]:
    """The cells of the grid at which a reference is read: a label rung at the
    cells whose h* is its horizon, every other reference at every cell."""
    if (mechanism, arm) not in REFERENCES:
        raise ValueError(f"unknown reference {arm}/{mechanism}")
    if arm in RUNG_ARMS:
        return tuple(cell for cell in CELLS if MATCHED_HORIZON_SECONDS[cell] == RUNG_ARMS[arm])
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
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} exceeds the hard cap of {MAX_WORKERS}")
    if len(args.traces) != len(TRACES):
        raise SystemExit(f"expected {len(TRACES)} trace files ({', '.join(TRACES)}); "
                         f"got {len(args.traces)}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    if args.paper_dir.exists():
        raise SystemExit(f"paper directory {args.paper_dir} exists; choose a new one")
    if args.output_dir.resolve() == args.paper_dir.resolve():
        raise SystemExit("--output-dir and --paper-dir must differ")


# --- provenance ------------------------------------------------------------------------------------


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository: the horizon
    control's (every `src` module and the four runners it chains) and this
    script."""
    paths = set(hc.execution_sources())
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


def _repository_path(path: Path) -> str:
    """A path as recorded: relative to the repository when it is inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)


# --- inputs ----------------------------------------------------------------------------------------

_cell_seed = hc._cell_seed


def _reference_entry(row, mechanism: str, arm: str, source: str) -> dict:
    """The horizon control's reference entry, with the replay digests the row
    carries (the class-order rows are held to them)."""
    entry = hc._reference_entry(row, mechanism, arm, source)
    for column in DIGEST_COLUMNS:
        if row.get(column, "") not in ("", None):
            entry[column] = str(row[column])
    return entry


def load_references(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> published row`
    for every reference of `REFERENCES`, each read from its own source only,
    at the cells `reference_cells` names, with the eligibility and width of
    `all16`; each must be there exactly once for every trace and seed of the
    grid. Rows of other arms, sources, cells or seeds are skipped unchecked.
    `paths` overrides the source files (tests)."""
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
    return published, problems


# --- the replay worker -----------------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one matched arm under `all16`, with the attribution hooks,
    the error-location statistics at the trace's label horizon (Part 2's) and
    the class statistics at the arm's h*, both around the arm's override: the
    horizon control's replay call and row, for these arms."""
    name, fraction, multiplier, arm, seed = task
    if arm not in cell_arms(fraction, multiplier):
        raise ValueError(f"{arm} is not an arm of cell {(fraction, multiplier)}")
    h = matched_horizon(fraction, multiplier)
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace, learned_ranker=SHARED["rankers"][name])
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    statistics = DecisionStatistics(trace, horizon, split_ms)
    class_statistics = ClassStatistics(trace, h, split_ms)
    # Both recorders see every decision with its final victim; the arm's own
    # answer passes through both unchanged.
    override = RecordingOverride(statistics, RecordingOverride(class_statistics, setup.override))
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
        "width": WIDTH, "mechanism": MECHANISM, "arm": arm, "family": FAMILY,
        "arm_parameter": h, "seed": seed, "variant": VARIANT,
        "order": ORDER_OF_ARM[arm], "role": role_of_horizon(h),
        "published_arm": PUBLISHED_ARMS.get(arm, ""),
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
    row.update(class_statistics.row())
    return row


def arm_cost(arm: str) -> int:
    """Scheduling class: the ranker scored by the store and at admission
    decisions first, the ranker at admission decisions only second."""
    return ORDERS.index(ORDER_OF_ARM[arm])


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay: the two matched arms of each cell, the learned order and
    the larger L1 first. The order only schedules work; every table is sorted
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
    """The replays at h* = 600 s against the published class-order row of
    their trace x cell x seed (`PUBLISHED_ARMS`): `avoided_prefill_tokens` and
    `counters_sha256` exactly, and `decision_sha256` where both rows carry
    one. Annotates each such row and returns the report and one table row per
    replay. Every one of `expected` must be found and equal."""
    matched = seen = decisions_compared = 0
    mismatches: list[str] = []
    missing: list[str] = []
    table: list[dict] = []
    for row in rows:
        if row["arm"] not in PUBLISHED_ARMS:
            continue
        seen += 1
        published_arm = PUBLISHED_ARMS[row["arm"]]
        reference = published.get((MECHANISM, published_arm) + _cell_seed(row))
        if reference is None:
            row["reference_source"] = ""
            row["reference_avoided_prefill_tokens"] = ""
            row["reproduces_reference"] = ""
            missing.append(f"{_where(row)}: no published {published_arm}/{MECHANISM} row")
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
                 "seed": row["seed"], "arm": row["arm"], "published_arm": published_arm,
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
            mismatches.append(f"{_where(row)} vs published {published_arm}: differs in "
                              + ", ".join(column for column, value in same.items() if not value))
    report = {"reference": f"{'/'.join(PUBLISHED_ARMS.values())} ({MECHANISM})",
              "expected": expected, "replays": seen, "matched": matched,
              "missing": len(missing), "mismatched": len(mismatches),
              "decision_digests_compared": decisions_compared,
              "mismatches": missing + mismatches}
    return report, table


def reproduction_passes(report: dict) -> bool:
    return (report["matched"] == report["expected"] == report["replays"]
            and not report["mismatched"] and not report["missing"])


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed (the rung of its cell's h* among them), on each identifier the
    reference publishes. Returns the pairs compared and the problems."""
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


check_statistics = hc.check_statistics


def check_class(rows) -> list[str]:
    """The plan's class check, per replay: the class statistics are at the
    cell's h* and saw every decision (with the replay's rejections, evictions
    and the error-location hook's overrides and window); no resident eviction
    discards a state reusable within h* while a sampled resident is not; every
    overridden decision is a first-round decision in which the arrival is a
    candidate (none with `arriving_index < 0`, in the whole replay and, by the
    error-location hook, in the window); and at h* = 600 s the class count on
    resident-only decisions is the error-location hook's `m4_count_resident`."""
    problems = []
    for row in rows:
        where = _where(row)
        h = matched_horizon(row["l1_fraction"], row["l2_multiplier"])
        if float(row["class_horizon_seconds"]) != h or float(row["arm_parameter"]) != h:
            problems.append(f"{where}: class horizon {row['class_horizon_seconds']} s / arm "
                            f"parameter {row['arm_parameter']} s, the cell's h* is {h:g} s")
        for mine, theirs in (("class_decisions_seen", "l2_decisions"),
                             ("class_rejections_seen", "l2_rejections"),
                             ("class_resident_evictions_seen", "l2_evictions"),
                             ("class_overridden_seen", "overridden_decisions_seen"),
                             ("class_decisions", "stat_decisions"),
                             ("class_overridden", "overridden_decisions")):
            if int(row[mine]) != int(row[theirs]):
                problems.append(f"{where}: {mine} {row[mine]} != {theirs} {row[theirs]}")
        if int(row["class_violations_seen"]) != 0:
            problems.append(f"{where}: {row['class_violations_seen']} resident evictions discard "
                            f"a state reusable within {h:g} s while a sampled resident is not")
        if int(row["class_overridden_outside_admission_seen"]) != 0:
            problems.append(f"{where}: {row['class_overridden_outside_admission_seen']} overridden "
                            "decisions in which the arrival is not a candidate")
        if int(row["overridden_decisions_resident"]) != 0:
            problems.append(f"{where}: overridden_decisions_resident "
                            f"{row['overridden_decisions_resident']} != 0")
        if h == PUBLISHED_HORIZON_SECONDS and (int(row["class_m4_count_resident"])
                                               != int(row["m4_count_resident"])):
            problems.append(f"{where}: class_m4_count_resident {row['class_m4_count_resident']} "
                            f"!= m4_count_resident {row['m4_count_resident']} at 600 s")
    return problems


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
POINT_FIELDS = ("mean", "ci95_half", "min", "max")
LEVEL_FIELDS = ("mean", "min", "max")


def _order_index(arm: str) -> int:
    return ORDERS.index(ORDER_OF_ARM[arm])


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x
    arm, of every replay metric and class count."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _order_index(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        h = matched_horizon(key[1], key[2])
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "arm": arm, "family": FAMILY, "arm_parameter": h,
                 "order": ORDER_OF_ARM[arm], "role": role_of_horizon(h),
                 "published_arm": PUBLISHED_ARMS.get(arm, ""), "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        for metric in REPLAY_METRICS + CLASS_METRICS:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        out.append(entry)
    return out


def reference_rows(published, rows) -> list[dict]:
    """The published reference rows of every trace x cell x seed of the run,
    with their U in points: the inputs of the readings, as they were read."""
    out = []
    for cell_seed in sorted({_cell_seed(row) for row in rows}):
        for (mechanism, arm), source in REFERENCES.items():
            if cell_seed[1:3] not in reference_cells(mechanism, arm):
                continue
            entry = dict(published[(mechanism, arm) + cell_seed])
            entry["extra_points"] = _u(entry)
            out.append(entry)
    return out


def _cell_context(cell) -> dict:
    """h*, role and the arm names of one trace x cell."""
    h = matched_horizon(cell[1], cell[2])
    learned_arm, recency_arm = cell_arms(cell[1], cell[2])
    rung = RUNG_OF_HORIZON[h]
    return {"h_star": h, "role": role_of_horizon(h), "matched_learned_arm": learned_arm,
            "matched_recency_arm": recency_arm, "rung_arm": rung,
            "rung_source": REFERENCES[(MECHANISM, rung)]}


def matched_tables(rows, published) -> dict[str, list[dict]]:
    """The four readings, from the replays and the published references.

    Per seed (`class_order_seeds`): U of learned, evict_label, both matched
    arms, the cell's label rung and the two published 600-second arms; the
    seed's R of both matched arms; and the seed-paired differences of readings
    2-4 (learned minus recency order at h*; label_binary_h* minus the recency
    arm; each matched arm minus its published 600-second arm) and, as
    context, the published 600-second order difference.

    Per trace x cell: `class_order` (reading 1: R_h* of both arms on the
    five-seed means, the reading and role, beside R of the published arms at
    600 s and the published reading), `order` (reading 2), `admission`
    (reading 3), and `versus600` (reading 4, the trace x cell with h* < 600 s
    only, one row per order).
    """
    seed_rows, class_order, order, admission, versus600 = [], [], [], [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        context = _cell_context(cell)
        learned_arm, recency_arm = context["matched_learned_arm"], context["matched_recency_arm"]
        published_learned, published_recency = (
            PUBLISHED_ARMS[matched_arm(PUBLISHED_HORIZON_SECONDS, name)] for name in ORDERS)
        u: dict[str, list[float]] = defaultdict(list)
        differences: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            arms = by_seed[seed]
            missing = [arm for arm in (learned_arm, recency_arm) if arm not in arms]
            if missing:
                raise KeyError(f"{cell} seed {seed}: replays of {missing} missing")
            refs = {name: _reference(published, cell, seed, MECHANISM, arm)
                    for name, arm in (("learned", "learned"), ("evict_label", "evict_label"),
                                      ("rung", context["rung_arm"]),
                                      ("published_learned", published_learned),
                                      ("published_recency", published_recency))}
            values = {"learned": _u(refs["learned"]), "evict_label": _u(refs["evict_label"]),
                      "matched_learned": _u(arms[learned_arm]),
                      "matched_recency": _u(arms[recency_arm]),
                      "label_binary_hstar": _u(refs["rung"]),
                      "published_learned_600": _u(refs["published_learned"]),
                      "published_recency_600": _u(refs["published_recency"])}
            for name, value in values.items():
                u[name].append(value)
            pairs = {"learned_minus_recency": (arms[learned_arm], arms[recency_arm]),
                     "label_binary_minus_recency": (refs["rung"], arms[recency_arm]),
                     "matched_minus_600_learned": (arms[learned_arm], refs["published_learned"]),
                     "matched_minus_600_recency": (arms[recency_arm], refs["published_recency"]),
                     "published_600_learned_minus_recency": (refs["published_learned"],
                                                             refs["published_recency"])}
            entry = {**identity, **context, "seed": seed,
                     **{f"U_{name}_points": value for name, value in values.items()}}
            for name in ("matched_learned", "matched_recency"):
                entry[f"R_{name}_seed"] = matched_recovery(values[name], values["learned"],
                                                           values["evict_label"])
            for name, (later, earlier) in pairs.items():
                tokens, points = _difference_points(later, earlier)
                differences[name].append(points)
                entry[f"{name}_tokens"] = tokens
                entry[f"{name}_points"] = points
            seed_rows.append(entry)
        # The five-seed means the tables print (`_put_stats`'s), so that R is
        # arithmetic on published columns.
        means = {name: _stats(values)["mean"] for name, values in u.items()}

        def r(name: str) -> float:
            return matched_recovery(means[name], means["learned"], means["evict_label"])

        entry = {**identity, **context, "seeds": len(seeds)}
        for name in ("learned", "evict_label", "matched_learned", "matched_recency",
                     "published_learned_600", "published_recency_600"):
            _put_stats(entry, f"U_{name}_points", u[name], fields=LEVEL_FIELDS)
        entry["R_matched_learned"] = r("matched_learned")
        entry["R_matched_recency"] = r("matched_recency")
        entry["reading"] = matched_reading(entry["R_matched_learned"])
        entry["predicted_reading"] = MATCHED_READINGS[0] if context["role"] == "new" else ""
        entry["R_600_published_learned"] = r("published_learned_600")
        entry["R_600_published_recency"] = r("published_recency_600")
        entry["published_600_reading"] = class_order_reading(entry["R_600_published_learned"])
        class_order.append(entry)

        entry = {**identity, **context, "seeds": len(seeds)}
        _put_stats(entry, "learned_minus_recency_points", differences["learned_minus_recency"],
                   fields=POINT_FIELDS)
        _put_signs(entry, "learned_minus_recency", differences["learned_minus_recency"])
        entry["R_matched_recency"] = r("matched_recency")
        entry["R_matched_learned"] = r("matched_learned")
        _put_stats(entry, "published_600_learned_minus_recency_points",
                   differences["published_600_learned_minus_recency"], fields=POINT_FIELDS)
        _put_signs(entry, "published_600_learned_minus_recency",
                   differences["published_600_learned_minus_recency"])
        order.append(entry)

        entry = {**identity, **context, "seeds": len(seeds)}
        _put_stats(entry, "U_label_binary_hstar_points", u["label_binary_hstar"],
                   fields=LEVEL_FIELDS)
        _put_stats(entry, "U_matched_recency_points", u["matched_recency"], fields=LEVEL_FIELDS)
        _put_stats(entry, "label_binary_minus_recency_points",
                   differences["label_binary_minus_recency"], fields=POINT_FIELDS)
        _put_signs(entry, "label_binary_minus_recency", differences["label_binary_minus_recency"])
        admission.append(entry)

        if context["role"] != "new":
            continue
        for name, matched, published_arm in (("learned", learned_arm, published_learned),
                                             ("recency", recency_arm, published_recency)):
            entry = {**identity, "h_star": context["h_star"], "role": context["role"],
                     "order": name, "matched_arm": matched, "published_arm": published_arm,
                     "seeds": len(seeds)}
            _put_stats(entry, "U_matched_points", u[f"matched_{name}"], fields=LEVEL_FIELDS)
            _put_stats(entry, "U_published_600_points", u[f"published_{name}_600"],
                       fields=LEVEL_FIELDS)
            _put_stats(entry, "matched_minus_600_points", differences[f"matched_minus_600_{name}"],
                       fields=POINT_FIELDS)
            _put_signs(entry, "matched_minus_600", differences[f"matched_minus_600_{name}"])
            entry["R_matched"] = r(f"matched_{name}")
            entry["R_600_published"] = r(f"published_{name}_600")
            versus600.append(entry)
    return {"class_order_seeds": seed_rows, "class_order": class_order, "order": order,
            "admission": admission, "versus600": versus600}


def reading_summary(tables: dict[str, list[dict]]) -> list[dict]:
    """The counts: reading 1 per role (new: h* < 600 s, against the
    prediction; reproduced: h* = 600 s), never merged; reading 2 and its
    published 600-second context, and reading 3, out of every trace x cell;
    reading 4 per order out of the new trace x cell."""
    class_order = tables["class_order"]
    counts = reading_counts((entry["role"], entry["reading"]) for entry in class_order)
    out = []
    for role in ROLES:
        role_counts = counts[role]
        out.append({"reading": "1_class_order_at_matched_horizon", "scope": role,
                    "cells": role_counts["cells"],
                    **{label: role_counts[label] for label in MATCHED_READINGS},
                    **dict.fromkeys(SIGN_READINGS, ""),
                    "predicted_" + MATCHED_READINGS[0]: (role_counts["cells"]
                                                         if role == "new" else ""),
                    "prediction_holds": prediction_holds(role_counts) if role == "new" else ""})

    def signs(reading: str, scope: str, entries, column: str) -> dict:
        return {"reading": reading, "scope": scope, "cells": len(entries),
                **dict.fromkeys(MATCHED_READINGS, ""),
                **sign_reading_counts(entry[column] for entry in entries),
                "predicted_" + MATCHED_READINGS[0]: "", "prediction_holds": ""}

    out.append(signs("2_order_within_matched_class", "all", tables["order"],
                     "learned_minus_recency_reading"))
    out.append(signs("2_context_order_within_600_class_published", "all", tables["order"],
                     "published_600_learned_minus_recency_reading"))
    out.append(signs("3_admission_at_matched_horizon", "all", tables["admission"],
                     "label_binary_minus_recency_reading"))
    for name in ORDERS:
        out.append(signs(f"4_matched_minus_600_{name}", "new",
                         [entry for entry in tables["versus600"] if entry["order"] == name],
                         "matched_minus_600_reading"))
    return out


# --- README ----------------------------------------------------------------------------------------

README_TEXT = """# Class order at the matched horizon: the frozen ranker given the exact reuse class at the horizon that works for the cell

Pre-registration: `docs/matched-class-order-plan.md`, a follow-up of Part 2 of the horizon and class-order control (`docs/horizon-control-plan.md`, `results/paper/horizon_control_001/`) and its fill-in (`results/paper/horizon_fill_001/`), written after both were seen. Two arms per trace x cell, mechanism `all16` (the arrival plus up to 16 uniformly sampled residents, any of which may leave), built exactly as the horizon control built `evict_binary_learned` / `evict_binary_recency` with the exact reuse label at the cell's matched horizon `h*` in place of 600 s: admission by the frozen pi0 `next_use` ranker (the arrival is rejected iff it is the first minimum of the ranker's keys `(score, last_group)` over the candidates); otherwise the victim is the first minimum, over the candidates other than the arrival, of `(1 if the next use is at most h* s away else 0, ranker score, last_group)` for `evict_binary_{h*}_learned` and `(1 if ... else 0, last_group)` for `evict_binary_{h*}_recency`; every later round uses that key. `h*` (the horizon of the grids run that minimises `S_h`, read after those runs, the same on both traces): 0.25% x 1: 60 s; 0.25% x 4: 150 s; 1% x 1: 150 s; 1% x 4: 600 s; 2% x 1: 300 s; 2% x 4: 600 s. At `h*` = 600 s the arm is the published one under another name (`evict_binary_600_learned` = `evict_binary_learned`, `evict_binary_600_recency` = `evict_binary_recency`) and its replays are the reproduction check. 2 traces x 6 cells x 2 arms x 5 seeds = 120 replays, each with the Phase 0.98b attribution hooks, the error-location decision statistics at the trace's 600-second label horizon (as in Part 2), and the class statistics at `h*` (read-only).

Roles. `new`: the trace x cell with `h*` < 600 s (0.25% x 1, 0.25% x 4, 1% x 1 and 2% x 1 on both traces: 8); `reproduced`: those with `h*` = 600 s (1% x 4 and 2% x 4 on both traces: 4, 40 replays). The plan's text counts six and six (60 replays); by its own table of `h*` the counts are eight and four, and every count here is computed from that table and reported with its denominator. The two roles are never merged.

References are published rows, not reruns: `lru`, `learned`, `label`, `evict_label` (all16) from `results/paper/error_location_001/replay_seeds.csv`; the label rung `label_binary_{h*}` of each cell from `results/paper/horizon_control_001/replay_seeds.csv` (60, 300, 600 s) or `results/paper/horizon_fill_001/replay_seeds.csv` (150 s); the published `evict_binary_learned` / `evict_binary_recency` of every trace x cell from `results/paper/horizon_control_001/replay_seeds.csv`. U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. `R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))` on the five-seed means (nan when the denominator is zero).

Boundaries. `h*` is the best of grids run on these traces, read after the fact: the reading says what an exact class at that horizon does for this ranker, not that the horizon can be chosen before the fact or that the class can be learned. `evict_label` is the exact `next_use` label applied greedily to a sampled decision, the comparison the horizon control used, not an optimum. The ranker's admission is the published relative rule on the arrival against sampled residents. Every arm reads the trace's future on purpose and none is a policy. Seed intervals describe sampling-seed variability only.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated), `h_star` (seconds), `role` (new / reproduced), `matched_learned_arm`, `matched_recency_arm`, `rung_arm` (`label_binary_{h*}`) and `rung_source`. Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise).

- `replay_seeds.csv`: one row per replay, the horizon control's columns (`eligibility`, `width`, `mechanism`, `arm`, `family` class_order, `arm_parameter` = h*, `variant` main, the replay counters, timing and worker memory, `counters_sha256`, the Phase 0.98b attribution columns, the point columns, the error-location decision statistics `stat_*`, `overridden_*`, `decision_sha256`, `m1` .. `m4_victim_at_horizon`, all at the 600-second label horizon), plus `order` (learned / recency), `role`, `published_arm` (h* = 600 only), the class statistics at h* (`class_horizon_seconds`; `class_decisions`, `class_rejections`, `class_resident_evictions`, `class_violations` (resident evictions whose victim is reusable within h* while some sampled resident is not), `class_overridden`, `class_overridden_outside_admission` (overridden decisions whose arrival is not a candidate), each over the evaluation window and with `_seen` over every decision of the replay; `class_m4_count_resident`, the violations of the window's resident-only decisions), and, at h* = 600, `reference_source`, `reference_avoided_prefill_tokens`, `reproduces_reference`.
- `replay.csv`: per trace x cell x arm, `mechanism`, `family`, `arm_parameter`, `order`, `role`, `published_arm`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, and mean / std / ci95_half / min / max of every replay metric, statistic and class count.
- `references_seeds.csv`: the published reference rows the readings use, one per trace x cell x seed x reference: `source`, `mechanism`, `arm`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`), the digests where published (`counters_sha256`, `decision_sha256`) and `extra_points` (U).
- `class_order_seeds.csv` (per seed, readings 1-4): `U_learned_points`, `U_evict_label_points`, `U_label_binary_hstar_points`, `U_published_learned_600_points`, `U_published_recency_600_points` (published), `U_matched_learned_points`, `U_matched_recency_points` (replays); `R_matched_learned_seed`, `R_matched_recency_seed` (the seed's own R, descriptive); and `_tokens` / `_points` of `learned_minus_recency` (reading 2), `label_binary_minus_recency` (reading 3), `matched_minus_600_learned` and `matched_minus_600_recency` (reading 4) and `published_600_learned_minus_recency` (context).
- `class_order.csv` (reading 1): per trace x cell, `U_<name>_points_mean` / `_min` / `_max` for learned, evict_label, matched_learned, matched_recency, published_learned_600, published_recency_600; `R_matched_learned` and `R_matched_recency` (R_h* on the five-seed means); `reading` (reuse_identification_at_matched_horizon_suffices when `R_matched_learned` >= 0.9, exactly 0.9 sufficing; ranker_order_costs_at_matched_horizon otherwise, a nan R included); `predicted_reading` (the prediction for the new role; empty for reproduced); and, as context, `R_600_published_learned`, `R_600_published_recency` and `published_600_reading` (the horizon control's reading on the published rows).
- `order.csv` (reading 2): per trace x cell, `learned_minus_recency_points_*` (U(evict_binary_h*_learned) - U(evict_binary_h*_recency), seed-paired; mean, ci95_half, min, max) with its seed signs (`learned_minus_recency_*`), `R_matched_recency` and `R_matched_learned` beside it, and the published 600-second difference with its seed signs (`published_600_learned_minus_recency_*`) as context.
- `admission.csv` (reading 3, descriptive): per trace x cell, `U_label_binary_hstar_points_*`, `U_matched_recency_points_*` and `label_binary_minus_recency_points_*` (U(label_binary_h*) - U(evict_binary_h*_recency), seed-paired) with its seed signs: the two arms evict residents by the same key and differ in who decides rejection.
- `versus600.csv` (reading 4, descriptive): per new trace x cell and `order`, `matched_arm`, `published_arm`, `U_matched_points_*`, `U_published_600_points_*`, `matched_minus_600_points_*` (the matched arm minus the published 600-second arm of the same order, seed-paired) with its seed signs, `R_matched` and `R_600_published`.
- `reproduction.csv`: one row per replay at h* = 600 s: `arm`, `published_arm`, `reference_source`, and for `avoided_prefill_tokens`, `counters_sha256`, `decision_sha256` this run's value, the published value (`published_<column>`) and whether they are equal (`same_<column>`); `reproduces` (all equal).
- `readings.csv`: the counts. `1_class_order_at_matched_horizon` per `scope` new and reproduced (never merged): `cells` and the count of each reading, with `predicted_reuse_identification_at_matched_horizon_suffices` (every new cell) and `prediction_holds` for new; `2_order_within_matched_class`, `2_context_order_within_600_class_published` and `3_admission_at_matched_horizon` over every trace x cell: consistent_gain / consistent_loss / mixed; `4_matched_minus_600_learned` / `_recency` over the new trace x cell.
- `run_config.json`: plan and code commits, source manifest, trace, model, model-manifest and reference hashes, the matched horizons, grid, arms, check results, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


def _report_lines(kind: str, lines, limit: int = 20) -> None:
    for line in list(lines)[:limit]:
        print(f"    {kind} {line}", flush=True)


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
    reference_hashes = {_repository_path(path): sha256_path(path)
                        for path in REFERENCE_SOURCES.values()}
    print(f"  published references: {len(published)} rows of "
          f"{', '.join(f'{arm}/{mechanism}' for mechanism, arm in REFERENCES)}", flush=True)
    for path, digest in reference_hashes.items():
        print(f"    {path} sha256={digest[:12]}", flush=True)
    if reference_problems:
        _report_lines("REFERENCE", reference_problems)
        raise SystemExit("published references are not what the plan names; nothing run")

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
        timing[(row["arm_parameter"], _order_index(row["arm"]), row["arm"])].append(
            float(row["seconds"]))
    print("  seconds per replay, by arm:", flush=True)
    timing_rows = []
    for (_, _, arm), values in sorted(timing.items()):
        print(f"    {arm:26s} {MECHANISM:6s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"arm": arm, "mechanism": MECHANISM, "variant": VARIANT,
                            "n": len(values), "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    expected_reproductions = sum(1 for task in tasks if task[3] in PUBLISHED_ARMS)
    reproduction, reproduction_table = check_reproduction(rows, published, expected_reproductions)
    print(f"  reproduction (h* = 600 s vs published {reproduction['reference']}; "
          f"{', '.join(REPRODUCTION_COLUMNS)}, and {', '.join(REPRODUCTION_OPTIONAL_COLUMNS)} "
          f"where both carry it): {reproduction['matched']} matched, "
          f"{reproduction['missing']} missing, {reproduction['mismatched']} mismatched "
          f"(expected {reproduction['expected']}; {reproduction['decision_digests_compared']} "
          "decision digests compared)", flush=True)
    _report_lines("MISMATCH", reproduction["mismatches"], limit=len(reproduction["mismatches"]))
    if not reproduction_passes(reproduction):
        failures.append("the replays at h* = 600 s do not reproduce the published class-order "
                        "rows")
    compared, identifier_problems = check_identifiers(rows, published)
    print(f"  identifiers ({', '.join(REFERENCE_IDENTIFIERS)}) of {len(rows)} replays against "
          f"{compared} published reference rows of their trace x cell x seed: "
          f"{'equal' if not identifier_problems else 'DIFFER'}", flush=True)
    _report_lines("IDENTIFIER", identifier_problems)
    if identifier_problems:
        failures.append("an identifier differs from a published reference row")
    statistics_problems = check_statistics(rows)
    print(f"  statistics: every decision seen with its final victim: "
          f"{'hold' if not statistics_problems else 'BROKEN'}", flush=True)
    _report_lines("STATISTICS", statistics_problems)
    if statistics_problems:
        failures.append("a statistics check failed")
    class_problems = check_class(rows)
    violations = sum(int(row["class_violations_seen"]) for row in rows)
    outside = sum(int(row["class_overridden_outside_admission_seen"]) for row in rows)
    print(f"  class at h*: resident evictions discarding a state reusable within h* while a "
          f"sampled resident is not: {violations}; overridden decisions outside a first round "
          f"with the arrival a candidate: {outside}; "
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
                               _order_index(row["arm"]), row["seed"]))
    reproduction_table.sort(key=lambda entry: (entry["trace"], entry["l1_fraction"],
                                               entry["l2_multiplier"],
                                               _order_index(entry["arm"]), entry["seed"]))
    replay_summary = aggregate_replays(rows)
    references = reference_rows(published, rows)
    tables = matched_tables(rows, published)
    summary = reading_summary(tables)

    print("  reading 1, class order at the matched horizon (five-seed means):", flush=True)
    for entry in tables["class_order"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"{entry['role']:10s} R(learned order)={entry['R_matched_learned']:7.3f} "
              f"R(recency)={entry['R_matched_recency']:7.3f} -> {entry['reading']} "
              f"(published at 600 s: {entry['R_600_published_learned']:7.3f}, "
              f"{entry['published_600_reading']})", flush=True)
    for entry in summary:
        if entry["reading"].startswith("1_"):
            print(f"    {entry['scope']}: " + ", ".join(
                f"{label} {entry[label]}/{entry['cells']}" for label in MATCHED_READINGS)
                + (f" (predicted {entry['predicted_' + MATCHED_READINGS[0]]}/{entry['cells']}; "
                   f"prediction {'holds' if entry['prediction_holds'] else 'FAILS'})"
                   if entry["scope"] == "new" else ""), flush=True)
    print("    the two roles are never merged", flush=True)
    for title, table, prefix in (
            ("reading 2, order within the matched class (learned - recency)", tables["order"],
             "learned_minus_recency"),
            ("reading 3, admission at the matched horizon (label_binary_h* - recency arm)",
             tables["admission"], "label_binary_minus_recency")):
        print(f"  {title}, points:", flush=True)
        for entry in table:
            print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
                  f"{entry[f'{prefix}_points_mean']:7.3f} ({entry[f'{prefix}_seed_signs']}, "
                  f"{entry[f'{prefix}_reading']})", flush=True)
    print("  reading 4, matched minus 600 s (new trace x cell), points:", flush=True)
    for entry in tables["versus600"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"{entry['order']:8s} {entry['matched_minus_600_points_mean']:7.3f} "
              f"({entry['matched_minus_600_seed_signs']}, {entry['matched_minus_600_reading']})",
              flush=True)
    for entry in summary:
        if not entry["reading"].startswith("1_"):
            print(f"    {entry['reading']} ({entry['scope']}): " + ", ".join(
                f"{label} {entry[label]}/{entry['cells']}" for label in SIGN_READINGS),
                flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "matched_class_order",
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
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "mechanism": {"name": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH},
        "matched_horizon_seconds": [{"l1_fraction": cell[0], "l2_multiplier": cell[1],
                                     "h_star": matched_horizon(*cell),
                                     "role": role_of_horizon(matched_horizon(*cell)),
                                     "arms": list(cell_arms(*cell)),
                                     "rung": RUNG_OF_HORIZON[matched_horizon(*cell)]}
                                    for cell in cells],
        "published_arms": dict(PUBLISHED_ARMS),
        "readings": {"class_order_share": CLASS_ORDER_SHARE,
                     "labels": list(MATCHED_READINGS), "roles": list(ROLES),
                     "sign_readings": list(SIGN_READINGS)},
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "statistics_horizon_seconds": {"decision_statistics": "trace label horizon (600 s)",
                                       "class_statistics": "h* of the cell"},
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {k: v for k, v in reproduction.items() if k != "mismatches"},
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "statistics_problems": len(statistics_problems),
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
        "note": "Descriptive follow-up of Part 2 of the horizon control with the exact reuse "
                "class at the horizon that works for the cell (h*, the best of grids run on "
                "these traces, read after the fact). Reading 1 is counted separately for the "
                "trace x cell with h* < 600 s (new) and h* = 600 s (reproduced: the published "
                "arms rerun) and never merged. Nothing is fitted; every arm reads the trace's "
                "future on purpose and none is a policy. References are published rows, not "
                "reruns. Seed intervals describe sampling-seed variability only.",
    }
    for directory in (args.output_dir, args.paper_dir):
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        for name in ("class_order_seeds", "class_order", "order", "admission", "versus600"):
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
