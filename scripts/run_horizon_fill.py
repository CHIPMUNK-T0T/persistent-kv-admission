#!/usr/bin/env python3
"""Horizon fill-in: is there a reuse horizon between 60 and 300 seconds where none of five sufficed.

The pre-registration is `docs/horizon-fill-plan.md`; nothing here may change
it. The horizon control's Part 1 arm, the exact `binary` label at horizon h
with recency ties (`label_binary_h`, mechanism `all16`), is replayed at eight
horizons on the four trace x cell that read "order needed" there
({conversation, tool-agent} x {0.25% x 4, 1% x 1}), five seeds, each with the
Phase 0.98b attribution hooks and the error-location decision statistics
attached read-only (`persistent_kv_admission.horizonfill`):

* fill horizons 90, 120, 150, 180 and 240 s;
* check horizons 60, 300 and 600 s, rerun for a self-contained curve and as
  reproduction checks (600 s is the published `label_binary`).

The replay call, the row schema, the provenance rules and the table helpers are
the horizon control's (`scripts/run_horizon_control.py`, imported by path,
unchanged); no ranker is loaded, because no arm needs one. References are
published rows, not reruns: `label`, `lru` and `label_binary` under `all16`
from `results/paper/error_location_001/replay_seeds.csv`.

Nothing is fitted and no arm is a proposed policy: every arm reads the trace's
future on purpose. Before anything is derived or published the run checks that
`label_binary_600` reproduces the 20 published `label_binary` rows exactly;
when `--first-run-dir` names the horizon control's run directory, that the 60
replays at h = 60, 300 and 600 equal its rows in `avoided_prefill_tokens` and
in the counter (and decision) digests; that the identifiers of every replay
equal those of the published reference rows of its trace x cell x seed; that
the statistics saw every decision with its final victim and no decision was
overridden; that the arm-independent counters are arm-independent; that no
absent token is unexplained; and (per replay) the attribution identities. Any
failure exits without writing a derived table, and nothing reaches
--paper-dir.

Outputs (in --output-dir, and in --paper-dir when given): per-seed and
aggregated replay tables, the published reference rows used, the three
readings per seed, per horizon and per trace x cell, a summary against the
prediction, one figure, a README describing every table, and the run
configuration. There is no smoke mode and no dirty-tree mode.
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

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import (
    ALL16,
    MECHANISMS,
    PUBLISHED_LABEL_BINARY_ARM,
    SUFFICES_SHORTFALL,
    shortfall,
)
from persistent_kv_admission.horizonfill import (
    ARMS,
    CHECK_ARMS,
    CHECK_HORIZONS_SECONDS,
    CURVE_HORIZONS_SECONDS,
    FILL_CELLS,
    FILL_HORIZONS_SECONDS,
    FILL_READINGS,
    HORIZON_OF_ARM,
    HORIZONS_SECONDS,
    MECHANISM,
    PREDICTED_SUFFICING,
    PUBLISHED_HORIZON_ARM,
    ROLE_OF_HORIZON,
    arm_setup,
    curve_minimisers,
    curve_values,
    fill_reading,
    same_minimiser,
    sufficing_set,
    sufficing_width_seconds,
    unimodal,
)
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

HORIZON_CONTROL_SCRIPT = REPOSITORY / "scripts/run_horizon_control.py"
PLAN_PATH = REPOSITORY / "docs/horizon-fill-plan.md"


def _load_horizon_control():
    """Import the horizon-control runner by path, unchanged, for its replay
    constants, provenance, check and table helpers; it brings the
    error-location, mechanism-control and Phase 0.97 runners with it (one
    instance of each, so the capacity rule reads the working set this run
    sets). `main` is behind the usual guard, so nothing runs, and nothing is
    read at import: its model loading is a function this runner never calls."""
    spec = importlib.util.spec_from_file_location("run_horizon_control", HORIZON_CONTROL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hc = _load_horizon_control()
rel = hc.rel
rmc = hc.rmc
rdp = hc.rdp

# --- the fixed grid (docs/horizon-fill-plan.md) ---------------------------------------------------

TRACES = hc.TRACES
CELLS = FILL_CELLS
assert set(CELLS) <= set(hc.CELLS)
SEEDS = hc.SEEDS
LABEL_HORIZON_SECONDS = hc.LABEL_HORIZON_SECONDS
ELIGIBILITY, WIDTH = MECHANISMS[MECHANISM]
# Every replay carries the statistics hook; there is no hook-off variant here.
VARIANT = "main"
FAMILY = "horizon"
DEFAULT_WORKERS = hc.DEFAULT_WORKERS
MAX_WORKERS = hc.MAX_WORKERS
assert (DEFAULT_WORKERS, MAX_WORKERS) == (10, 12)
# Published references, anchored at the repository (a tracked file).
ERROR_LOCATION_REFERENCE = hc.ERROR_LOCATION_REFERENCE
REFERENCE_SOURCE = "error_location_001"
# Published reference rows, by (mechanism, arm). `label_binary` is the
# reproduction target of `label_binary_600`.
REFERENCES = {(MECHANISM, arm): REFERENCE_SOURCE
              for arm in ("label", "lru", PUBLISHED_LABEL_BINARY_ARM)}
assert hc.REPRODUCTIONS[PUBLISHED_HORIZON_ARM] == (MECHANISM, PUBLISHED_LABEL_BINARY_ARM)
REFERENCE_IDENTIFIERS = hc.REFERENCE_IDENTIFIERS
REPLAY_METRICS = hc.REPLAY_METRICS
# The horizon control's run directory: the table compared, the columns that
# must be equal, and the column compared when both runs carry it.
FIRST_RUN_TABLE = "replay_seeds.csv"
FIRST_RUN_COLUMNS = ("avoided_prefill_tokens", "counters_sha256")
FIRST_RUN_OPTIONAL_COLUMNS = ("decision_sha256",)
SHARED: dict[str, object] = {}


def arm_parameter(arm: str) -> float:
    """The horizon h of a `label_binary_h` arm, in seconds."""
    return HORIZON_OF_ARM[arm]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path,
                        help="the two Mooncake trace files (conversation, tool-agent)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new run directory; refused if it exists")
    parser.add_argument("--paper-dir", type=Path, default=None,
                        help="new publication directory; refused if it exists; written only "
                             "when every check passes")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"replay processes (default {DEFAULT_WORKERS}, at most {MAX_WORKERS})")
    parser.add_argument("--first-run-dir", type=Path, default=None,
                        help=f"the horizon control's run directory (holding {FIRST_RUN_TABLE}); "
                             "its rows at h = 60, 300, 600 must equal this run's")
    return parser.parse_args(argv)


def validate_arguments(args) -> None:
    """Every refusal that needs no trace, reference or git call, in one place."""
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} exceeds the hard cap of {MAX_WORKERS}")
    if len(args.traces) != len(TRACES):
        raise SystemExit(f"expected {len(TRACES)} trace files ({', '.join(TRACES)}); "
                         f"got {len(args.traces)}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    if args.paper_dir is not None and args.paper_dir.exists():
        raise SystemExit(f"paper directory {args.paper_dir} exists; choose a new one")
    if args.first_run_dir is not None and not (args.first_run_dir / FIRST_RUN_TABLE).is_file():
        raise SystemExit(f"--first-run-dir {args.first_run_dir} has no {FIRST_RUN_TABLE}")


# --- provenance ----------------------------------------------------------------------------------


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


# --- inputs ---------------------------------------------------------------------------------------

_cell_seed = hc._cell_seed


def load_references(path=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> published row`
    for every reference of `REFERENCES` on the grid's 20 trace x cell x seed,
    with the eligibility and width its mechanism names; each of the 60 must be
    there exactly once. Rows of other cells are skipped unchecked. `path`
    overrides the source file (tests)."""
    path = Path(path or ERROR_LOCATION_REFERENCE)
    grid = {(name, fraction, multiplier, seed) for name in TRACES
            for fraction, multiplier in CELLS for seed in SEEDS}
    published: dict[tuple, dict] = {}
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            if (row["mechanism"], arm) not in REFERENCES or row.get("variant", "main") != "main":
                continue
            cell_seed = _cell_seed(row)
            if cell_seed not in grid:
                continue
            if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
                problems.append(f"{REFERENCE_SOURCE} row {row['mechanism']}/{arm} has "
                                f"{row['eligibility']}/{row['width']}")
            key = (row["mechanism"], arm) + cell_seed
            if key in published:
                problems.append(f"duplicate published reference {key}")
                continue
            published[key] = hc._reference_entry(row, row["mechanism"], arm, REFERENCE_SOURCE)
    for mechanism, arm in REFERENCES:
        for cell_seed in sorted(grid):
            key = (mechanism, arm) + cell_seed
            if key not in published:
                problems.append(f"published {REFERENCE_SOURCE} reference {key} missing")
    return published, problems


def load_first_run(directory: Path) -> tuple[list[dict], str]:
    """The horizon control's per-replay rows and the SHA-256 of their table."""
    path = Path(directory) / FIRST_RUN_TABLE
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, sha256_path(path)


# --- the replay worker ----------------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one `label_binary_h` arm under `all16`, with the
    attribution hooks and the statistics hook around the arm's (absent)
    override: the horizon control's replay call and row, for these horizons."""
    name, fraction, multiplier, arm, seed = task
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace)
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    statistics = DecisionStatistics(trace, horizon, split_ms)
    override = RecordingOverride(statistics, setup.override)
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
        "arm_parameter": arm_parameter(arm), "seed": seed, "variant": VARIANT,
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
    return row


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay, the larger L1 first (the arms cost alike). The order only
    schedules work; every table is sorted before it is written."""
    tasks = [(name, fraction, multiplier, arm, seed)
             for name in names for fraction, multiplier in cells for arm in ARMS for seed in seeds]
    tasks.sort(key=lambda task: (-task[1], -task[2], task[0], HORIZON_OF_ARM[task[3]], task[4]))
    return tasks


# --- checks ---------------------------------------------------------------------------------------

check_reproduction = hc.check_reproduction


def _where(row) -> str:
    return f"{row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}"


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed, on each identifier the reference publishes. Returns the number of
    (replay, reference) pairs compared and the problems."""
    compared, problems = 0, []
    for row in rows:
        for mechanism, arm in REFERENCES:
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


def check_statistics(rows) -> list[str]:
    """The statistics saw every decision with its final victim, at least one
    decision fell inside the evaluation window, and no decision was
    overridden: no arm here has an override."""
    problems = []
    for row in rows:
        for mine, theirs in (("stat_decisions_seen", "l2_decisions"),
                             ("stat_rejections_seen", "l2_rejections"),
                             ("stat_evictions_seen", "l2_evictions")):
            if int(row[mine]) != int(row[theirs]):
                problems.append(f"{_where(row)}: {mine} {row[mine]} != {theirs} {row[theirs]}")
        if int(row["stat_decisions"]) <= 0:
            problems.append(f"{_where(row)}: no decision inside the evaluation window")
        if int(row["overridden_decisions_seen"]) != 0:
            problems.append(f"{_where(row)}: {row['overridden_decisions_seen']} overridden "
                            "decisions for an arm without an override")
    return problems


def index_first_run(first_rows) -> tuple[dict[tuple, dict], list[str]]:
    """`(arm, trace, fraction, multiplier, seed) -> row` of the first run's
    check-arm rows (variant main), and a problem per duplicate."""
    index: dict[tuple, dict] = {}
    problems: list[str] = []
    for row in first_rows:
        if row.get("variant", "main") != VARIANT or row["arm"] not in CHECK_ARMS:
            continue
        key = (row["arm"],) + _cell_seed(row)
        if key in index:
            problems.append(f"duplicate first-run row {key}")
            continue
        index[key] = row
    return index, problems


def first_run_coverage(first_rows, names, cells, seeds) -> list[str]:
    """Before any replay: the first run must hold every check-arm row of the
    grid exactly once, so that a wrong directory is refused with nothing run."""
    index, problems = index_first_run(first_rows)
    for arm in CHECK_ARMS:
        for name in names:
            for fraction, multiplier in cells:
                for seed in seeds:
                    key = (arm, name, fraction, multiplier, seed)
                    if key not in index:
                        problems.append(f"first-run row {key} missing")
    return problems


def check_first_run(rows, first_rows, expected: int) -> dict[str, object]:
    """The replays of the check arms (h = 60, 300, 600) against the first
    run's rows of the same trace x cell x arm x seed (variant main): equal in
    `avoided_prefill_tokens` and `counters_sha256`, and in `decision_sha256`
    where both rows carry one. Every one of `expected` must be found, once."""
    index, problems = index_first_run(first_rows)
    matched = missing = decisions_compared = 0
    mismatches: list[str] = []
    seen = 0
    for row in rows:
        if row["variant"] != VARIANT or row["arm"] not in CHECK_ARMS:
            continue
        seen += 1
        first = index.get((row["arm"],) + _cell_seed(row))
        if first is None:
            missing += 1
            problems.append(f"{_where(row)}: no first-run row")
            continue
        differs = []
        if int(row["avoided_prefill_tokens"]) != int(first["avoided_prefill_tokens"]):
            differs.append(f"avoided_prefill_tokens {row['avoided_prefill_tokens']} != "
                           f"{first['avoided_prefill_tokens']}")
        if str(row["counters_sha256"]) != str(first["counters_sha256"]):
            differs.append("counters_sha256")
        for column in FIRST_RUN_OPTIONAL_COLUMNS:
            if row.get(column, "") not in ("", None) and first.get(column, "") not in ("", None):
                decisions_compared += 1
                if str(row[column]) != str(first[column]):
                    differs.append(column)
        if differs:
            mismatches.append(f"{_where(row)}: " + ", ".join(differs))
        else:
            matched += 1
    return {"expected": expected, "replays": seen, "matched": matched, "missing": missing,
            "mismatched": len(mismatches), "decision_digests_compared": decisions_compared,
            "duplicates": sum(1 for line in problems if line.startswith("duplicate")),
            "mismatches": mismatches, "problems": problems}


def first_run_passes(report: dict) -> bool:
    return (report["matched"] == report["expected"] == report["replays"]
            and not report["mismatched"] and not report["missing"] and not report["duplicates"])


# --- derivation -----------------------------------------------------------------------------------

_put_stats = hc._put_stats
_put_signs = hc._put_signs
_difference_points = hc._difference_points
_identity = hc._identity
_is_nan = hc._is_nan
_u = hc._u
_reference = hc._reference
_cell_key = hc._cell_key
index_arms = hc.index_arms


def _horizons_text(horizons) -> str:
    return ";".join(f"{h:g}" for h in horizons)


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x arm."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], ARMS.index(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "arm": arm, "family": FAMILY,
                 "arm_parameter": arm_parameter(arm), "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        for metric in REPLAY_METRICS:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        out.append(entry)
    return out


def reference_rows(published, rows) -> list[dict]:
    """The published reference rows of every trace x cell x seed of the run,
    with their U in points: the inputs of the readings, as they were read."""
    out = []
    for cell_seed in sorted({_cell_seed(row) for row in rows}):
        for mechanism, arm in REFERENCES:
            entry = dict(published[(mechanism, arm) + cell_seed])
            entry["extra_points"] = _u(entry)
            out.append(entry)
    return out


def fill_tables(rows, published) -> tuple[list[dict], list[dict], list[dict]]:
    """The three readings. Per seed and h: U(label), U(lru), U(label_binary_h),
    `U(label) - U(label_binary_h)` and the seed's own S; per trace x cell x h:
    the five-seed means, the difference with its seed signs and S_h on the
    means; per trace x cell: S_h of all eight horizons, reading 1 over the five
    fill horizons (the smallest S_h, every fill horizon attaining it, the
    label), reading 2 over the seven from 60 to 300 s (unimodal, the
    minimisers) and reading 3 over the same seven (the sufficing set, its size
    and width)."""
    seed_rows, table, readings = [], [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        references = {arm: [_reference(published, cell, seed, MECHANISM, arm) for seed in seeds]
                      for arm in ("label", "lru")}
        u_ref = {arm: [_u(entry) for entry in entries] for arm, entries in references.items()}
        mean_ref = {arm: float(np.mean(values)) for arm, values in u_ref.items()}
        shortfalls = {}
        for h, arm in zip(HORIZONS_SECONDS, ARMS):
            role = ROLE_OF_HORIZON[h]
            differences, values = [], []
            for position, seed in enumerate(seeds):
                row = by_seed[seed][arm]
                tokens, difference = _difference_points(references["label"][position], row)
                differences.append(difference)
                values.append(_u(row))
                seed_rows.append({**identity, "h": h, "arm": arm, "role": role, "seed": seed,
                                  "U_label_points": u_ref["label"][position],
                                  "U_lru_points": u_ref["lru"][position],
                                  "U_arm_points": values[-1],
                                  "label_minus_arm_tokens": tokens,
                                  "label_minus_arm_points": difference,
                                  "S_seed": shortfall(u_ref["label"][position], values[-1],
                                                      u_ref["lru"][position])})
            entry = {**identity, "h": h, "arm": arm, "role": role, "seeds": len(seeds),
                     "U_label_points_mean": mean_ref["label"], "U_lru_points_mean": mean_ref["lru"]}
            _put_stats(entry, "U_arm_points", values, fields=("mean", "ci95_half", "min", "max"))
            _put_stats(entry, "label_minus_arm_points", differences,
                       fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, "label_minus_arm", differences)
            entry["S"] = shortfall(mean_ref["label"], entry["U_arm_points_mean"], mean_ref["lru"])
            shortfalls[h] = entry["S"]
            table.append(entry)
        reading, smallest, attaining = fill_reading({h: shortfalls[h]
                                                     for h in FILL_HORIZONS_SECONDS})
        curve = {h: shortfalls[h] for h in CURVE_HORIZONS_SECONDS}
        members = sufficing_set(curve)
        entry = {**identity, "seeds": len(seeds)}
        for h in HORIZONS_SECONDS:
            entry[f"S_{h:g}"] = shortfalls[h]
        entry.update(S_fill_min=smallest, S_fill_min_horizons=_horizons_text(attaining),
                     reading=reading, unimodal_60_300=unimodal(curve_values(curve)),
                     curve_minimiser_horizons=_horizons_text(curve_minimisers(curve)),
                     sufficing_horizons=_horizons_text(members), sufficing_count=len(members),
                     sufficing_width_seconds=sufficing_width_seconds(curve),
                     U_label_points_mean=mean_ref["label"], U_lru_points_mean=mean_ref["lru"])
        readings.append(entry)
    return seed_rows, table, readings


def _count(entries, column, value) -> int:
    return sum(1 for entry in entries if entry[column] == value)


def fill_summary(readings) -> list[dict]:
    """Reading 1 counted against the prediction (a reuse label suffices at a
    filled horizon in every trace x cell: 4 of 4 overall, every cell of a
    trace), and per trace whether its two cells, which share an L2 capacity,
    have the same minimising horizon over 60 to 300 s (reading 2)."""
    def counts(members) -> dict:
        return {reading: _count(members, "reading", reading) for reading in FILL_READINGS}

    total = counts(readings)
    out = [{"scope": "all", "cells": len(readings), **total,
            "predicted_" + FILL_READINGS[0]: PREDICTED_SUFFICING,
            "prediction_holds": total[FILL_READINGS[0]] == PREDICTED_SUFFICING,
            "unimodal_60_300": _count(readings, "unimodal_60_300", True),
            "cell_labels": "", "curve_minimiser_horizons": "", "same_curve_minimiser": ""}]
    cell_order = {cell: position for position, cell in enumerate(CELLS)}
    for name in sorted({entry["trace"] for entry in readings}):
        members = sorted((entry for entry in readings if entry["trace"] == name),
                         key=lambda entry: cell_order.get(
                             (entry["l1_fraction"], entry["l2_multiplier"]), len(CELLS)))
        minimisers = [tuple(float(h) for h in entry["curve_minimiser_horizons"].split(";") if h)
                      for entry in members]
        share = len(members) == 2 and same_minimiser(minimisers[0], minimisers[1])
        trace_counts = counts(members)
        out.append({"scope": name, "cells": len(members), **trace_counts,
                    "predicted_" + FILL_READINGS[0]: len(members),
                    "prediction_holds": trace_counts[FILL_READINGS[0]] == len(members),
                    "unimodal_60_300": _count(members, "unimodal_60_300", True),
                    "cell_labels": "|".join(entry["cell"] for entry in members),
                    "curve_minimiser_horizons": "|".join(entry["curve_minimiser_horizons"]
                                                         for entry in members),
                    "same_curve_minimiser": share})
    return out


# --- figure ---------------------------------------------------------------------------------------

BLUE, ORANGE, AQUA = hc.BLUE, hc.ORANGE, hc.AQUA
INK, MUTED = hc.INK, hc.MUTED
_style_axis = hc._style_axis


def fill_figure(directory: Path, readings: list[dict], seed_rows: list[dict]) -> None:
    """S_h against h per trace x cell on a log h axis (S on the five-seed
    means; band = seed min-max of the per-seed S), the fill horizons as filled
    circles and the check horizons as open squares, with the label (S = 0) and
    sampled LRU (S = 1) levels and the 0.10 reading threshold."""
    traces, cells = rmc._panel_grid(readings)
    index = {_cell_key(entry): entry for entry in readings}
    seeds_of: dict[tuple, list[float]] = defaultdict(list)
    for row in seed_rows:
        seeds_of[_cell_key(row) + (row["h"],)].append(row["S_seed"])
    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(7.0, 4.0 * len(cells) + 0.6), 2.9 * len(traces) + 1.2))
    fill_x = list(FILL_HORIZONS_SECONDS)
    check_x = list(CHECK_HORIZONS_SECONDS)
    for row_index, name in enumerate(traces):
        for column, (fraction, multiplier) in enumerate(cells):
            axis = axes[row_index][column]
            entry = index.get((name, fraction, multiplier))
            if entry is None:
                axis.set_visible(False)
                continue
            values = [entry[f"S_{h:g}"] for h in HORIZONS_SECONDS]
            lows = [min(seeds_of[(name, fraction, multiplier, h)], default=math.nan)
                    for h in HORIZONS_SECONDS]
            highs = [max(seeds_of[(name, fraction, multiplier, h)], default=math.nan)
                     for h in HORIZONS_SECONDS]
            axis.fill_between(HORIZONS_SECONDS, lows, highs, color=BLUE, alpha=0.14, linewidth=0)
            axis.plot(HORIZONS_SECONDS, values, color=BLUE, linewidth=2.0)
            axis.plot(fill_x, [entry[f"S_{h:g}"] for h in fill_x], linestyle="none", marker="o",
                      markersize=6, color=BLUE)
            axis.plot(check_x, [entry[f"S_{h:g}"] for h in check_x], linestyle="none",
                      marker="s", markersize=6, markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=1.4)
            axis.axhline(0.0, color=INK, linestyle=":", linewidth=1.3)
            axis.axhline(1.0, color=MUTED, linestyle="-.", linewidth=1.3)
            axis.axhline(SUFFICES_SHORTFALL, color=AQUA, linestyle=(0, (1, 2)), linewidth=1.2)
            axis.set_xscale("log")
            axis.set_xticks(HORIZONS_SECONDS, [f"{h:g}" for h in HORIZONS_SECONDS])
            axis.minorticks_off()
            axis.set_title(f"{rel._panel_title(name, fraction, multiplier)}\n{entry['reading']}",
                           fontsize=8, color=INK)
            axis.set_xlabel("h, seconds", fontsize=7.5, color=INK)
            if column == 0:
                axis.set_ylabel("S_h (share of label - lru gap not reached)", fontsize=7.5,
                                color=INK)
            _style_axis(axis)
    handles = [
        Line2D([], [], color=BLUE, marker="o", markersize=6, linewidth=2.0,
               label="S_h, fill horizon (90-240 s)"),
        Line2D([], [], color=BLUE, marker="s", markersize=6, markerfacecolor="white",
               markeredgecolor=INK, markeredgewidth=1.4, linewidth=2.0,
               label="S_h, check horizon (60, 300, 600 s; rerun)"),
        Line2D([], [], color=INK, linestyle=":", linewidth=1.3, label="label (S = 0)"),
        Line2D([], [], color=MUTED, linestyle="-.", linewidth=1.3, label="sampled LRU (S = 1)"),
        Line2D([], [], color=AQUA, linestyle=(0, (1, 2)), linewidth=1.2,
               label=f"reading threshold S = {SUFFICES_SHORTFALL:g}"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    figure.suptitle("Exact reuse label at horizon h (all16), filled between 60 and 300 s: "
                    "shortfall from the next_use label (S on five-seed means; band = seed "
                    "min-max)", fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.10 if len(traces) > 1 else 0.17, 1.0, 0.95))
    figure.savefig(directory / "fill.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


README_TEXT = """# Horizon fill-in: is there a reuse horizon between 60 and 300 seconds where none of five sufficed

Pre-registration: `docs/horizon-fill-plan.md`, a follow-up of the horizon and class-order control (`docs/horizon-control-plan.md`, `results/paper/horizon_control_001/`), written after that control's results were seen and because of them. One arm family, `label_binary_h` (mechanism `all16`: the arrival plus up to 16 uniformly sampled residents, any of which may leave; key `(1 if the next use is at most h s away else 0, last_group)`, the exact `binary` target at h, built and replayed exactly as in the horizon control), at eight horizons: the fill horizons 90, 120, 150, 180, 240 s and the check horizons 60, 300, 600 s (rerun; 600 s is the published `label_binary`). Four trace x cell: {conversation, tool-agent} x {0.25% x 4, 1% x 1}, the four that read "order needed" in the horizon control; seeds 0-4; 160 replays, each with the Phase 0.98b attribution hooks and the error-location decision statistics (read-only). References are published rows, not reruns: `label`, `lru` and `label_binary` under `all16` from `results/paper/error_location_001/replay_seeds.csv`. U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. `S_h = (U(label) - U(label_binary_h)) / (U(label) - U(lru))` on the five-seed means (nan when the denominator is zero).

Boundaries. This is a second look at four trace x cell chosen because they failed the first one, judged on ten horizons where the other eight were judged on five: its count ("n/4 of the remaining cells at a horizon added afterwards") is reported beside the horizon control's ("8/12 on the registered grid") and never merged with it. A horizon found by filling a grid after the run describes that grid; it does not show that the horizon can be chosen before the fact, predicted from capacity or learned from features. Every arm reads the trace's future on purpose and none is a policy. Seed intervals describe sampling-seed variability only.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated), `h` (seconds), `role` (fill: 90-240 s; check: 60, 300, 600 s). Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise).

- `replay_seeds.csv`: one row per replay, the horizon control's columns: `eligibility`, `width`, `mechanism`, `arm`, `family` (horizon), `arm_parameter` (h), `variant` (main: with the statistics hook), the replay counters, timing and worker memory, `counters_sha256`, the Phase 0.98b attribution columns, the point columns, and the error-location decision statistics (`stat_*`, `overridden_*`, `decision_sha256`, `m1` .. `m4_victim_at_horizon`). `label_binary_600` rows also carry `reference_source`, `reference_avoided_prefill_tokens` and `reproduces_reference`.
- `replay.csv`: per trace x cell x arm, `mechanism`, `family`, `arm_parameter`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, and mean / std / ci95_half / min / max of every replay metric and statistic.
- `references_seeds.csv`: the published reference rows the readings use, one per trace x cell x seed x reference: `source`, `mechanism`, `arm`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`) and `extra_points` (U).
- `fill_seeds.csv` (per seed): per trace x cell x `h` x seed, `arm`, `role`, `U_label_points`, `U_lru_points` (published, all16), `U_arm_points` (`label_binary_h`), `label_minus_arm_tokens` / `_points` (U(label) - U(label_binary_h)) and `S_seed` (the seed's own shortfall, descriptive).
- `fill.csv` (per h): per trace x cell x `h`, `arm`, `role`, `U_label_points_mean`, `U_lru_points_mean`, `U_arm_points_*` (mean, ci95_half, min, max), `label_minus_arm_points_*` with its seed signs (`label_minus_arm_*`), and `S` on the five-seed means.
- `fill_reading.csv` (the three readings): per trace x cell, `S_60` .. `S_600` (all eight horizons); reading 1 over the five fill horizons: `S_fill_min` (the smallest), `S_fill_min_horizons` (every fill horizon attaining it, ";"-joined) and `reading` (reuse_label_suffices_at_filled_horizon when S_fill_min <= 0.10, else order_needed_stands); reading 2 over the seven horizons from 60 to 300 s: `unimodal_60_300` (S_h non-increasing up to a minimiser and non-decreasing after it) and `curve_minimiser_horizons` (every horizon attaining the smallest of the seven); reading 3 over the same seven: `sufficing_horizons` (every h with S_h <= 0.10, ";"-joined, empty when none), `sufficing_count` and `sufficing_width_seconds` (largest minus smallest member; nan when empty); and the five-seed means `U_label_points_mean`, `U_lru_points_mean`.
- `fill_summary.csv`: `scope` all: the count of each reading out of `cells` (4) next to `predicted_reuse_label_suffices_at_filled_horizon` (4, the prediction fixed before the run) and `prediction_holds`, and the count with `unimodal_60_300`; `scope` per trace: the same over its two cells, `cell_labels` and `curve_minimiser_horizons` (in that cell order, "|"-separated) and `same_curve_minimiser` (the two cells, which share an L2 capacity and differ in L1, have the same non-empty set of minimising horizons).
- `fill.png`: S_h against h on a log axis per trace x cell (line: S on five-seed means; band: seed min-max of S_seed; filled circles: fill horizons; open squares: check horizons), with the label (S = 0) and sampled-LRU (S = 1) levels and the 0.10 threshold.
- `run_config.json`: plan and code commits, source manifest, trace and reference hashes, the first run's table hash (or that the comparison was not run), grid, arms, check counts, timing and memory.
"""


# --- main -----------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


def _repository_path(path: Path) -> str:
    """A path as recorded: relative to the repository when it is inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)


def _report_lines(kind: str, lines, limit: int = 20) -> None:
    for line in list(lines)[:limit]:
        print(f"    {kind} {line}", flush=True)


def main() -> None:
    args = parse_args()
    clock = time.time()
    validate_arguments(args)
    seeds = tuple(SEEDS)

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
    cells = CELLS
    published, reference_problems = load_references()
    reference_sha256 = sha256_path(ERROR_LOCATION_REFERENCE)
    print(f"  published references: {len(published)} rows of "
          f"{', '.join(f'{arm}/{mechanism}' for mechanism, arm in REFERENCES)} "
          f"({_repository_path(ERROR_LOCATION_REFERENCE)} sha256={reference_sha256[:12]})",
          flush=True)
    if reference_problems:
        _report_lines("REFERENCE", reference_problems)
        raise SystemExit("published references are not what the plan names; nothing run")
    first_rows, first_sha256 = None, None
    if args.first_run_dir is not None:
        first_rows, first_sha256 = load_first_run(args.first_run_dir)
        print(f"  first run: {args.first_run_dir / FIRST_RUN_TABLE} ({len(first_rows)} rows, "
              f"sha256={first_sha256[:12]})", flush=True)
        coverage_problems = first_run_coverage(first_rows, names, cells, seeds)
        if coverage_problems:
            _report_lines("FIRST-RUN", coverage_problems)
            raise SystemExit("the first run does not hold every row to compare; nothing run")
    else:
        print("  first run: not given; the comparison at h = 60, 300, 600 is not run", flush=True)

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds)
    workers = min(args.workers, len(tasks))
    print(f"{len(tasks)} replays on {workers} workers ({len(names)} traces x {len(cells)} cells x "
          f"{len(ARMS)} horizons x {len(seeds)} seeds)", flush=True)
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
                if done % 20 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {time.time() - started:.0f}s "
                          f"last={row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}"
                          f"/{row['variant']} ({row['seconds']:.0f}s, "
                          f"worker peak {row['worker_peak_rss_mib']:.0f} MiB)", flush=True)
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
        timing[(ARMS.index(row["arm"]), row["arm"])].append(float(row["seconds"]))
    print("  seconds per replay, by arm:", flush=True)
    timing_rows = []
    for (_, arm), values in sorted(timing.items()):
        print(f"    {arm:20s} {MECHANISM:6s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"arm": arm, "mechanism": MECHANISM, "variant": VARIANT,
                            "n": len(values), "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    per_arm = len(names) * len(cells) * len(seeds)
    reproduction = check_reproduction(rows, published, {PUBLISHED_HORIZON_ARM: per_arm})
    for arm, report in reproduction.items():
        print(f"  reproduction ({arm}/{MECHANISM} vs published {report['reference']}): "
              f"{report['matched']} matched, {report['missing']} missing, "
              f"{report['mismatched']} mismatched (expected {report['expected']})", flush=True)
        _report_lines("MISMATCH", report["mismatches"], limit=len(report["mismatches"]))
        if report["mismatched"] or report["missing"] or report["matched"] != report["expected"]:
            failures.append(f"reproduction of {report['reference']} by {arm} failed")
    first_run = None
    if first_rows is not None:
        first_run = check_first_run(rows, first_rows, len(CHECK_ARMS) * per_arm)
        print(f"  first run ({', '.join(CHECK_ARMS)} vs {args.first_run_dir / FIRST_RUN_TABLE}; "
              f"{', '.join(FIRST_RUN_COLUMNS)}, and {', '.join(FIRST_RUN_OPTIONAL_COLUMNS)} where "
              f"both carry it): {first_run['matched']} matched, {first_run['missing']} missing, "
              f"{first_run['mismatched']} mismatched, {first_run['duplicates']} duplicate "
              f"first-run rows (expected {first_run['expected']}; "
              f"{first_run['decision_digests_compared']} decision digests compared)", flush=True)
        _report_lines("FIRST-RUN", first_run["mismatches"], limit=len(first_run["mismatches"]))
        _report_lines("FIRST-RUN", first_run["problems"])
        if not first_run_passes(first_run):
            failures.append("the replays at h = 60, 300, 600 differ from the first run's")
    compared, identifier_problems = check_identifiers(rows, published)
    print(f"  identifiers ({', '.join(REFERENCE_IDENTIFIERS)}) of {len(rows)} replays against "
          f"{compared} published reference rows of their trace x cell x seed: "
          f"{'equal' if not identifier_problems else 'DIFFER'}", flush=True)
    _report_lines("IDENTIFIER", identifier_problems)
    if identifier_problems:
        failures.append("an identifier differs from a published reference row")
    statistics_problems = check_statistics(rows)
    print(f"  statistics: every decision seen with its final victim, no overridden decision: "
          f"{'hold' if not statistics_problems else 'BROKEN'}", flush=True)
    _report_lines("STATISTICS", statistics_problems)
    if statistics_problems:
        failures.append("a statistics check failed")
    cell_seeds, invariant_problems = rmc.check_invariants(rows, len(ARMS))
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
                               ARMS.index(row["arm"]), row["seed"]))
    replay_summary = aggregate_replays(rows)
    references = reference_rows(published, rows)
    fill_seeds, fill, readings = fill_tables(rows, published)
    summary = fill_summary(readings)

    total = summary[0]
    print("  reading 1, fill (S_h on five-seed means; fill horizons 90-240 s):", flush=True)
    for entry in readings:
        values = " ".join(f"{entry[f'S_{h:g}']:7.3f}" for h in HORIZONS_SECONDS)
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} "
              f"S({','.join(f'{h:g}' for h in HORIZONS_SECONDS)})={values} "
              f"fill min={entry['S_fill_min']:.3f} at h={entry['S_fill_min_horizons'] or '-'} -> "
              f"{entry['reading']}", flush=True)
    print("    " + ", ".join(f"{reading} {total[reading]}/{total['cells']}"
                             for reading in FILL_READINGS)
          + f" (predicted {FILL_READINGS[0]} {PREDICTED_SUFFICING}/{total['cells']}; "
          f"prediction {'holds' if total['prediction_holds'] else 'FAILS'}); not merged with the "
          "horizon control's count on its registered grid", flush=True)
    print("  reading 2, shape over 60-300 s:", flush=True)
    for entry in readings:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} "
              f"unimodal={entry['unimodal_60_300']} "
              f"minimiser h={entry['curve_minimiser_horizons'] or '-'}", flush=True)
    for entry in summary[1:]:
        print(f"    {entry['scope'][:12]:12s} minimisers {entry['curve_minimiser_horizons']} "
              f"({entry['cell_labels']}): same={entry['same_curve_minimiser']}", flush=True)
    print("  reading 3, horizons from 60 to 300 s with S_h <= "
          f"{SUFFICES_SHORTFALL:g}:", flush=True)
    for entry in readings:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} "
              f"{{{entry['sufficing_horizons']}}} count={entry['sufficing_count']} width="
              + ("empty" if math.isnan(entry["sufficing_width_seconds"])
                 else f"{entry['sufficing_width_seconds']:g}s"), flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "horizon_fill",
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "git_dirty_at_start": bool(status),
        "execution_sources_differing_from_head": differing,
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "models": {},
        "references": {_repository_path(ERROR_LOCATION_REFERENCE): reference_sha256},
        "reference_rows": {f"{arm}/{mechanism}": source
                           for (mechanism, arm), source in REFERENCES.items()},
        "reference_identifiers": list(REFERENCE_IDENTIFIERS),
        "first_run": ({"checked": True, "directory": str(args.first_run_dir.resolve()),
                       "table": FIRST_RUN_TABLE, "sha256": first_sha256,
                       "columns": list(FIRST_RUN_COLUMNS),
                       "columns_where_both_carry_them": list(FIRST_RUN_OPTIONAL_COLUMNS)}
                      if first_rows is not None else
                      {"checked": False, "note": "--first-run-dir not given; the replays at "
                                                 "h = 60, 300, 600 were not compared with the "
                                                 "horizon control's run"}),
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "mechanism": {"name": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH},
        "arms": {arm: {"h": HORIZON_OF_ARM[arm], "role": ROLE_OF_HORIZON[HORIZON_OF_ARM[arm]]}
                 for arm in ARMS},
        "fill_horizons_seconds": list(FILL_HORIZONS_SECONDS),
        "check_horizons_seconds": list(CHECK_HORIZONS_SECONDS),
        "curve_horizons_seconds": list(CURVE_HORIZONS_SECONDS),
        "readings": {"suffices_shortfall": SUFFICES_SHORTFALL,
                     "fill_labels": list(FILL_READINGS),
                     "predicted_" + FILL_READINGS[0]: PREDICTED_SUFFICING},
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none",
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {arm: {k: v for k, v in report.items() if k != "mismatches"}
                             for arm, report in reproduction.items()},
            "first_run": ({k: v for k, v in first_run.items()
                           if k not in ("mismatches", "problems")}
                          if first_run is not None else "not run"),
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "statistics_problems": len(statistics_problems),
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
        "note": "Descriptive follow-up of the horizon control on the four trace x cell that read "
                "order needed there, chosen because they failed it and judged on ten horizons "
                "where the other eight were judged on five; its count is reported beside the "
                "horizon control's 8/12 and never merged with it. A horizon found by filling the "
                "grid after the run describes that grid: it does not show that the horizon can "
                "be chosen before the fact, predicted from capacity or learned from features. "
                "Nothing is fitted; every arm reads the trace's future on purpose and none is a "
                "policy. References are published rows, not reruns; label_binary_600 reproduces "
                "the published label_binary rows. Seed intervals describe sampling-seed "
                "variability only.",
    }
    destinations = [args.output_dir]
    if args.paper_dir is not None:
        destinations.append(args.paper_dir)
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        _write(directory / "fill_seeds.csv", fill_seeds)
        _write(directory / "fill.csv", fill)
        _write(directory / "fill_reading.csv", readings)
        _write(directory / "fill_summary.csv", summary)
        fill_figure(directory, readings, fill_seeds)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {', '.join(str(d) for d in destinations)}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
