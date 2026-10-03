#!/usr/bin/env python3
"""Class-order mix: which reuse class carries the learned order's loss at the matched horizon, and does recency carry information beyond the bit.

The pre-registration is `docs/class-order-mix-plan.md`; nothing here may
change it. At every trace x cell three arms are replayed at the cell's matched
horizon `h*` (`persistent_kv_admission.classmix`), each with admission by the
frozen pi0 `next_use` ranker through `errorloc.HybridOverride` unchanged and
the store's key `((reusable within h* s, within-class key), last_group)`:

* `mix_out_learned_{h*}`: the ranker's score within the non-reusable class, a
  constant (so recency) within the reusable one;
* `mix_in_learned_{h*}`: the ranker within the reusable class, recency within
  the non-reusable one;
* `random_within_class_{h*}`: a uniform draw in [0, 1) per candidate per
  decision within each class, from a stream seeded by the replay seed and the
  arm name (`classmix.random_stream_seed`), separate from the store's sampling
  stream (`random.Random(l2_seed)` inside the store, untouched).

Mechanism `all16`, 2 traces x 6 cells x 3 arms x 5 seeds = 180 replays. The
replay call, the hooks and the row are the matched class-order runner's
(`scripts/run_matched_class_order.py`, imported by path, unchanged): the
Phase 0.98b attribution hooks, the error-location decision statistics at the
trace's 600-second label horizon and `matchedorder.ClassStatistics` at `h*`,
all read-only, around the arm's own override.

References are published rows, not reruns: `learned` and `evict_label` under
`all16` from `results/paper/error_location_001/replay_seeds.csv`;
`evict_binary_{h*}_learned` and `evict_binary_{h*}_recency` from
`results/paper/matched_class_order_001/replay_seeds.csv`; and the label rung of
the cell's `h*`, `label_binary_{h*}`, from
`results/paper/horizon_control_001/replay_seeds.csv` (60, 300, 600 s) or
`results/paper/horizon_fill_001/replay_seeds.csv` (150 s). The differences
`D_out`, `D_in`, `D_rand` (each arm minus `evict_binary_{h*}_recency`),
`D_learned` (published, `evict_binary_{h*}_learned` minus the recency arm) and
`random - learned` are seed-paired, in points.

Nothing is fitted and no arm is a proposed policy: every arm reads the trace's
future on purpose. Before anything is derived or published the run checks
that the identifiers of every replay equal those of every published reference
row of its trace x cell x seed; that the statistics saw every decision with
its final victim; that in every replay no resident eviction discards a state
reusable within `h*` while a sampled resident is not, and every overridden
decision is a first-round decision in which the arrival is a candidate; that
every random replay drew from its own stream; that the arm-independent
counters are arm-independent; that no absent token is unexplained; and (per
replay) the Phase 0.98b attribution identities. Any failure exits without
writing a derived table, and nothing reaches the paper directory.

Outputs (in --output-dir and in --paper-dir): per-seed and aggregated replay
tables, the published reference rows used, the differences per seed and per
trace x cell with their seed signs and readings and `R_h*` of every arm, the
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
from persistent_kv_admission.classmix import (
    CONSISTENT_LOSS,
    DIFFERENCES,
    HORIZON_OF_ARM,
    KIND_OF_ARM,
    KINDS,
    LEARNED_CLASS,
    PREDICTION_ONE_CELLS,
    PREDICTION_TWO_CELLS,
    PREDICTION_TWO_MINIMUM,
    RANDOM,
    RANDOM_STREAM_TAG,
    additive,
    arm_setup,
    cell_arms,
    class_conditions,
    in_prediction_one,
    matched_reference_arms,
    prediction_one_holds,
    prediction_two_holds,
    random_scorer,
    random_stream_seed,
    recovery,
    seed_interval,
    sign_reading_counts,
)
from persistent_kv_admission.crossworkload import HYPERPARAMETERS
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import DecisionStatistics, RecordingOverride
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    MECHANISM,
    ORDERS,
    RUNG_OF_HORIZON,
    SIGN_READINGS,
    ClassStatistics,
    matched_arm,
    matched_horizon,
)
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

MATCHED_CLASS_ORDER_SCRIPT = REPOSITORY / "scripts/run_matched_class_order.py"
PLAN_PATH = REPOSITORY / "docs/class-order-mix-plan.md"


def _load_matched_class_order():
    """Import the matched class-order runner by path, unchanged, for its replay
    constants, provenance, reference-entry, check and table helpers; it brings
    the horizon-control, error-location, mechanism-control and Phase 0.97
    runners with it (one instance of each, so the capacity rule reads the
    working set this run sets, and the ranker loader is the one it uses).
    `main` is behind the usual guard, so nothing runs, and nothing is read at
    import."""
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

# --- the fixed grid (docs/class-order-mix-plan.md) ------------------------------------------------

TRACES = mco.TRACES
CELLS = mco.CELLS
SEEDS = mco.SEEDS
assert set(CELLS) == set(MATCHED_HORIZON_SECONDS)
assert all(cell[0] in TRACES and cell[1:] in CELLS for cell in PREDICTION_ONE_CELLS)
assert PREDICTION_TWO_CELLS == len(TRACES) * len(CELLS)
LABEL_HORIZON_SECONDS = mco.LABEL_HORIZON_SECONDS
assert LABEL_HORIZON_SECONDS == 600.0
assert MECHANISM == mco.MECHANISM == "all16"
ELIGIBILITY, WIDTH = mco.ELIGIBILITY, mco.WIDTH
assert (ELIGIBILITY, WIDTH) == ("all", 16)
# Every replay carries the statistics hooks; there is no hook-off variant here.
VARIANT = "main"
FAMILY = "class_order_mix"
DEFAULT_WORKERS = mco.DEFAULT_WORKERS
MAX_WORKERS = mco.MAX_WORKERS
assert (DEFAULT_WORKERS, MAX_WORKERS) == (10, 12)
# The plan's publication directory (a new one; refused if it exists).
DEFAULT_PAPER_DIR = REPOSITORY / "results/paper/class_order_mix_001"
# Published references, anchored at the repository (tracked files).
MATCHED_SOURCE = "matched_class_order_001"
REFERENCE_SOURCES = {"error_location_001": mco.ERROR_LOCATION_REFERENCE,
                     MATCHED_SOURCE: REPOSITORY / "results/paper/matched_class_order_001/"
                                                  "replay_seeds.csv",
                     "horizon_control_001": mco.HORIZON_CONTROL_REFERENCE,
                     "horizon_fill_001": mco.HORIZON_FILL_REFERENCE}
# The label rung of each matched horizon and the only source it is read from.
RUNG_SOURCES = mco.RUNG_SOURCES
assert RUNG_SOURCES == {60.0: "horizon_control_001", 150.0: "horizon_fill_001",
                        300.0: "horizon_control_001", 600.0: "horizon_control_001"}
# Published reference rows, by (mechanism, arm): the source each is read from.
REFERENCES = {
    (MECHANISM, "learned"): "error_location_001",
    (MECHANISM, "evict_label"): "error_location_001",
    **{(MECHANISM, matched_arm(h, order)): MATCHED_SOURCE
       for h in MATCHED_HORIZONS_SECONDS for order in ORDERS},
    **{(MECHANISM, RUNG_OF_HORIZON[h]): RUNG_SOURCES[h] for h in MATCHED_HORIZONS_SECONDS},
}
# A matched arm or a rung is read at the cells whose h* is its horizon only;
# `learned` and `evict_label` at every cell of the grid.
REFERENCE_HORIZON = {**{matched_arm(h, order): h for h in MATCHED_HORIZONS_SECONDS
                        for order in ORDERS},
                     **{RUNG_OF_HORIZON[h]: h for h in MATCHED_HORIZONS_SECONDS}}
REFERENCE_IDENTIFIERS = hc.REFERENCE_IDENTIFIERS
assert REFERENCE_IDENTIFIERS == ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                                 "l1_avoided_tokens", "absent_compulsory_tokens")
REPLAY_METRICS = hc.REPLAY_METRICS
CLASS_METRICS = mco.CLASS_METRICS
ARMS_PER_CELL = len(KINDS)
SHARED: dict[str, object] = {}


def reference_cells(mechanism: str, arm: str) -> tuple[tuple[float, float], ...]:
    """The cells of the grid at which a reference is read: a matched arm or a
    label rung at the cells whose h* is its horizon, `learned` and
    `evict_label` at every cell."""
    if (mechanism, arm) not in REFERENCES:
        raise ValueError(f"unknown reference {arm}/{mechanism}")
    if arm in REFERENCE_HORIZON:
        return tuple(cell for cell in CELLS
                     if MATCHED_HORIZON_SECONDS[cell] == REFERENCE_HORIZON[arm])
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
    for every reference of `REFERENCES`, each read from its own source only,
    at the cells `reference_cells` names, with the eligibility and width of
    `all16`; each must be there exactly once for every trace and seed of the
    grid. Rows of other arms, sources, cells or seeds are skipped unchecked.
    The entries are the matched runner's (`_reference_entry`: tokens, the
    identifiers and the digests the row carries). `paths` overrides the
    source files (tests)."""
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
                if (row["eligibility"], int(row["width"])) != (ELIGIBILITY, WIDTH):
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
    """One replay of one arm under `all16`, with the attribution hooks, the
    error-location statistics at the trace's label horizon and the class
    statistics at the arm's h*, both around the arm's override: the matched
    class-order runner's replay call and row, for these arms. The random arm's
    stream seed and the number of draws it made are added to the row."""
    name, fraction, multiplier, arm, seed = task
    if arm not in cell_arms(fraction, multiplier):
        raise ValueError(f"{arm} is not an arm of cell {(fraction, multiplier)}")
    h = matched_horizon(fraction, multiplier)
    kind = KIND_OF_ARM[arm]
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace, learned_ranker=SHARED["rankers"][name], seed=seed)
    stream = random_scorer(setup)
    if (stream is None) != (kind != RANDOM):
        raise AssertionError(f"{arm}: the random stream is {'missing' if stream is None else 'present'}")
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
        "kind": kind, "learned_class": LEARNED_CLASS.get(kind, ""),
        "random_stream_seed": stream.stream_seed if stream is not None else "",
        "random_draws": stream.draws if stream is not None else "",
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


def _kind_index(arm: str) -> int:
    return KINDS.index(KIND_OF_ARM[arm])


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay: the three arms of each cell at its h*, the arms that
    score the ranker in the store first and the larger L1 first. The order
    only schedules work; every table is sorted before it is written."""
    tasks = [(name, fraction, multiplier, arm, seed)
             for name in names for fraction, multiplier in cells
             for arm in cell_arms(fraction, multiplier) for seed in seeds]
    tasks.sort(key=lambda task: (_kind_index(task[3]), -task[1], -task[2], task[0], task[3],
                                 task[4]))
    return tasks


# --- checks ----------------------------------------------------------------------------------------

_where = mco._where


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed (the matched arms and the rung of its cell's h* among them), on
    each identifier the reference publishes. Returns the pairs compared and
    the problems."""
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


# The horizon control's statistics check and the matched runner's class check,
# as they are: the class statistics are at the cell's h* and saw every
# decision, no resident eviction discards a state reusable within h* while a
# sampled resident is not, and every overridden decision is a first-round
# decision in which the arrival is a candidate.
check_statistics = hc.check_statistics
check_class = mco.check_class


def check_random(rows) -> list[str]:
    """Every random replay drew from its own stream: the row carries the
    stream seed of `random_stream_seed(seed, arm)` and at least one draw per
    decision (every decision scores at least one candidate); no mixed replay
    carries a stream."""
    problems = []
    for row in rows:
        where = _where(row)
        if row["kind"] != RANDOM:
            if row["random_stream_seed"] != "" or row["random_draws"] != "":
                problems.append(f"{where}: a mixed arm carries a random stream")
            continue
        expected = random_stream_seed(int(row["seed"]), row["arm"])
        if int(row["random_stream_seed"]) != expected:
            problems.append(f"{where}: random stream seed {row['random_stream_seed']} != "
                            f"{expected}")
        draws, decisions = int(row["random_draws"]), int(row["l2_decisions"])
        if decisions <= 0 or draws < decisions:
            problems.append(f"{where}: {draws} random draws for {decisions} decisions")
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
POINT_FIELDS = mco.POINT_FIELDS
LEVEL_FIELDS = mco.LEVEL_FIELDS
# U of every arm of a trace x cell, in table order: the published references,
# then the three replayed arms. R_h* is reported for every arm but the two
# that define it (0 for learned and 1 for evict_label by construction).
U_NAMES = ("learned", "evict_label", "label_binary_hstar", "matched_learned",
           "matched_recency") + KINDS
R_NAMES = U_NAMES[2:]
READING_COLUMNS = ("D_in_negative", "D_out_above_D_in", "both", "additive")


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x
    arm, of every replay metric and class count (and the random arm's draws)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _kind_index(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        kind = KIND_OF_ARM[arm]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "arm": arm, "family": FAMILY,
                 "arm_parameter": matched_horizon(key[1], key[2]), "kind": kind,
                 "learned_class": LEARNED_CLASS.get(kind, ""), "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        metrics = REPLAY_METRICS + CLASS_METRICS + (("random_draws",) if kind == RANDOM else ())
        for metric in metrics:
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
    """h*, the arm names and the reference arms of one trace x cell."""
    h = matched_horizon(cell[1], cell[2])
    references = matched_reference_arms(cell[1], cell[2])
    context = {"h_star": h, **{f"{kind}_arm": arm for kind, arm in zip(KINDS, cell_arms(*cell[1:]))},
               "matched_learned_arm": references["matched_learned"],
               "matched_recency_arm": references["matched_recency"],
               "rung_arm": references["label_binary_hstar"],
               "rung_source": REFERENCES[(MECHANISM, references["label_binary_hstar"])],
               "prediction_1_cell": in_prediction_one(cell)}
    return context


def mix_tables(rows, published) -> dict[str, list[dict]]:
    """The differences, from the replays and the published references.

    Per seed (`differences_seeds`): U of every arm (published: learned,
    evict_label, the rung and the matched control's two arms at h*; replayed:
    the three arms), the seed's own R of every arm but learned and
    evict_label (descriptive), and `_tokens` / `_points` of every difference
    of `classmix.DIFFERENCES`, plus `D_in_plus_D_out_points`.

    Per trace x cell (`differences`): the five-seed U of every arm and R_h* of
    every arm on the five-seed means; every difference's mean, 95% t
    half-width, min, max and seed signs; reading 1's two conditions on the
    means (`D_in < 0`, `D_out > D_in`) and their conjunction; `D_in + D_out`
    beside the seed interval of `D_learned` and whether it lies within it
    (additivity, descriptive).
    """
    seed_rows, cell_rows = [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        context = _cell_context(cell)
        arms_of = {kind: context[f"{kind}_arm"] for kind in KINDS}
        references = {"learned": "learned", "evict_label": "evict_label",
                      "label_binary_hstar": context["rung_arm"],
                      "matched_learned": context["matched_learned_arm"],
                      "matched_recency": context["matched_recency_arm"]}
        u: dict[str, list[float]] = defaultdict(list)
        differences: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            arms = by_seed[seed]
            missing = [arm for arm in arms_of.values() if arm not in arms]
            if missing:
                raise KeyError(f"{cell} seed {seed}: replays of {missing} missing")
            source = {name: _reference(published, cell, seed, MECHANISM, arm)
                      for name, arm in references.items()}
            source.update({kind: arms[arm] for kind, arm in arms_of.items()})
            values = {name: _u(source[name]) for name in U_NAMES}
            for name, value in values.items():
                u[name].append(value)
            entry = {**identity, **context, "seed": seed,
                     **{f"U_{name}_points": values[name] for name in U_NAMES}}
            for name in R_NAMES:
                entry[f"R_{name}_seed"] = recovery(values[name], values["learned"],
                                                   values["evict_label"])
            for name, (later, earlier) in DIFFERENCES.items():
                tokens, points = _difference_points(source[later], source[earlier])
                differences[name].append(points)
                entry[f"{name}_tokens"] = tokens
                entry[f"{name}_points"] = points
            entry["D_in_plus_D_out_points"] = entry["D_in_points"] + entry["D_out_points"]
            seed_rows.append(entry)
        # The five-seed means the tables print (`_put_stats`'s), so that every
        # reading is arithmetic on published columns.
        means = {name: _stats(values)["mean"] for name, values in u.items()}
        entry = {**identity, **context, "seeds": len(seeds)}
        for name in U_NAMES:
            _put_stats(entry, f"U_{name}_points", u[name], fields=LEVEL_FIELDS)
        for name in R_NAMES:
            entry[f"R_{name}"] = recovery(means[name], means["learned"], means["evict_label"])
        for name in DIFFERENCES:
            _put_stats(entry, f"{name}_points", differences[name], fields=POINT_FIELDS)
            _put_signs(entry, name, differences[name])
        d_in, d_out = entry["D_in_points_mean"], entry["D_out_points_mean"]
        conditions = class_conditions(d_in, d_out)
        entry["D_in_negative"] = conditions["D_in_negative"]
        entry["D_out_above_D_in"] = conditions["D_out_above_D_in"]
        entry["which_class_condition"] = conditions["both"]
        entry["D_in_plus_D_out_points"] = d_in + d_out
        _, low, high = seed_interval(differences["D_learned"])
        entry["D_learned_interval_low"] = low
        entry["D_learned_interval_high"] = high
        entry["additive"] = additive(d_in, d_out, low, high)
        cell_rows.append(entry)
    return {"differences_seeds": seed_rows, "differences": cell_rows}


PREDICTION_ONE_TEXT = (f"D_in < 0 and D_out > D_in in each of the {len(PREDICTION_ONE_CELLS)} "
                       "named trace x cell")
PREDICTION_TWO_TEXT = (f"D_rand consistent_loss in at least {PREDICTION_TWO_MINIMUM} of "
                       f"{PREDICTION_TWO_CELLS} trace x cell")


def reading_summary(tables: dict[str, list[dict]]) -> list[dict]:
    """The counts. Reading 1: the two conditions and their conjunction over
    the eight named trace x cell (prediction 1) and over every trace x cell
    (descriptive), additivity, and the seed signs of D_in and D_out; reading
    2: the seed signs of D_rand over every trace x cell (prediction 2) and of
    random - learned (descriptive); and, as context, the seed signs of the
    published D_learned."""
    cells = tables["differences"]
    named = [entry for entry in cells if entry["prediction_1_cell"]]
    out = []

    def row(reading: str, scope: str, entries, **values) -> dict:
        return {"reading": reading, "scope": scope, "cells": len(entries),
                **dict.fromkeys(READING_COLUMNS, ""), **dict.fromkeys(SIGN_READINGS, ""),
                "prediction": "", "prediction_holds": "", **values}

    def conditions(entries) -> dict:
        return {"D_in_negative": sum(bool(entry["D_in_negative"]) for entry in entries),
                "D_out_above_D_in": sum(bool(entry["D_out_above_D_in"]) for entry in entries),
                "both": sum(bool(entry["which_class_condition"]) for entry in entries)}

    def signs(entries, name: str) -> dict:
        return sign_reading_counts(entry[f"{name}_reading"] for entry in entries)

    holds_one = prediction_one_holds({_cell_key(entry): entry["which_class_condition"]
                                      for entry in named})
    out.append(row("1_which_class", "prediction_cells", named, **conditions(named),
                   prediction=PREDICTION_ONE_TEXT, prediction_holds=holds_one))
    out.append(row("1_which_class", "all", cells, **conditions(cells)))
    out.append(row("1_additivity", "all", cells,
                   additive=sum(bool(entry["additive"]) for entry in cells)))
    for name in ("D_in", "D_out"):
        for scope, entries in (("prediction_cells", named), ("all", cells)):
            out.append(row(f"1_{name}_seed_signs", scope, entries, **signs(entries, name)))
    rand = signs(cells, "D_rand")
    holds_two = (len(cells) == PREDICTION_TWO_CELLS
                 and prediction_two_holds(rand[CONSISTENT_LOSS]))
    out.append(row("2_D_rand_seed_signs", "all", cells, **rand, prediction=PREDICTION_TWO_TEXT,
                   prediction_holds=holds_two))
    out.append(row("2_random_minus_learned_seed_signs", "all", cells,
                   **signs(cells, "random_minus_learned")))
    for scope, entries in (("prediction_cells", named), ("all", cells)):
        out.append(row("context_D_learned_seed_signs", scope, entries,
                       **signs(entries, "D_learned")))
    return out


def named_cells_are_published_losses(tables: dict[str, list[dict]]) -> bool:
    """Whether, over the trace x cell of the run, the cells the plan names for
    prediction 1 are exactly those whose published D_learned is a consistent
    loss (descriptive: the plan's list is used as written either way)."""
    cells = tables["differences"]
    named = {_cell_key(entry) for entry in cells if entry["prediction_1_cell"]}
    losses = {_cell_key(entry) for entry in cells
              if entry["D_learned_reading"] == CONSISTENT_LOSS}
    return named == losses


# --- README ----------------------------------------------------------------------------------------

README_TEXT = """# Class-order mix: which reuse class carries the learned order's loss at the matched horizon, and does recency carry information beyond the bit

Pre-registration: `docs/class-order-mix-plan.md`, a follow-up of the matched class-order control (`docs/matched-class-order-plan.md`, `results/paper/matched_class_order_001/`), written after it was seen. Three arms per trace x cell, mechanism `all16` (the arrival plus up to 16 uniformly sampled residents, any of which may leave), at the cell's matched horizon `h*` (0.25% x 1: 60 s; 0.25% x 4: 150 s; 1% x 1: 150 s; 1% x 4: 600 s; 2% x 1: 300 s; 2% x 4: 600 s; the same on both traces). Every arm has admission by the frozen pi0 `next_use` ranker (the arrival is rejected iff it is the first minimum of the ranker's keys `(score, last_group)` over the candidates); otherwise the victim is the first minimum, over the candidates other than the arrival, of the store's key `((1 if the next use is at most h* s away else 0, within-class key), last_group)`, and every later round uses that key:

- `mix_out_learned_{h*}`: the within-class key is the ranker's score for a candidate not reusable within h* and a constant for a reusable one: the non-reusable class ordered by the ranker, the reusable class by recency (`last_group`);
- `mix_in_learned_{h*}`: the reverse, the reusable class by the ranker and the non-reusable class by recency;
- `random_within_class_{h*}`: the within-class key is a uniform draw in [0, 1) per candidate per decision, from a stream of its own: `random.Random(s)` with `s` the 8-byte BLAKE2b digest (`digest_size=8`), read big-endian, of `"class_order_mix/random_within_class" + "\\x1f" + <replay seed> + "\\x1f" + <arm name>` (UTF-8) (`classmix.random_stream_seed`, `errorloc.hash_bits`), one `.random()` per candidate in the order the store scores them. It never reads or advances the store's sampling stream (`random.Random(l2_seed)` inside the store), so given the same earlier choices the candidate sets are the recency arm's. The stream depends on the seed and the arm name only, so both traces share it at a cell and seed.

The ranker inside the mixed arms is the one object the admission rule consults, observed once. 2 traces x 6 cells x 3 arms x 5 seeds = 180 replays, each with the Phase 0.98b attribution hooks, the error-location decision statistics at the trace's 600-second label horizon and the class statistics at `h*` (read-only).

References are published rows, not reruns: `learned`, `evict_label` (all16) from `results/paper/error_location_001/replay_seeds.csv`; `evict_binary_{h*}_learned` and `evict_binary_{h*}_recency` from `results/paper/matched_class_order_001/replay_seeds.csv`; the label rung `label_binary_{h*}` from `results/paper/horizon_control_001/replay_seeds.csv` (60, 300, 600 s) or `results/paper/horizon_fill_001/replay_seeds.csv` (150 s). U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. Seed-paired differences, in points: `D_out = U(mix_out_learned) - U(evict_binary_h*_recency)`, `D_in = U(mix_in_learned) - U(evict_binary_h*_recency)`, `D_rand = U(random_within_class) - U(evict_binary_h*_recency)`, `D_learned = U(evict_binary_h*_learned) - U(evict_binary_h*_recency)` (published rows only), and `random_minus_learned = U(random_within_class) - U(evict_binary_h*_learned)`. `R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))` on the five-seed means (nan when the denominator is zero).

Readings fixed before the run (five-seed means; "consistent" means the same sign in all five seed-paired values). Prediction 1: in each of the eight trace x cell named by the plan (conversation 0.25% x 1, 0.25% x 4, 1% x 1, 2% x 1; tool-agent 0.25% x 1, 1% x 1, 1% x 4, 2% x 1), `D_in < 0` and `D_out > D_in`. Prediction 2: `D_rand` a consistent loss in at least 8 of the 12 trace x cell. Descriptive: the two conditions over all 12; whether `D_in + D_out` lies within the seed interval of `D_learned`; the seed signs of `random_minus_learned`.

Boundaries. `h*` is the best of grids run on these traces, read after the fact; the reuse bit is exact; the comparison is sampled-16 greedy, not an optimum. "Not reusable within h*" includes states reused after h*; nothing here says they are worthless. The random arm describes one random stream per seed; seed intervals describe sampling-seed and random-stream variability together. Every arm reads the trace's future on purpose and none is a policy.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated), `h_star` (seconds), `mix_out_learned_arm`, `mix_in_learned_arm`, `random_within_class_arm`, `matched_learned_arm`, `matched_recency_arm`, `rung_arm` (`label_binary_{h*}`), `rung_source`, and `prediction_1_cell` (one of the eight named trace x cell). Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise).

- `replay_seeds.csv`: one row per replay, the matched class-order runner's columns (`eligibility`, `width`, `mechanism`, `arm`, `family` class_order_mix, `arm_parameter` = h*, `variant` main, the replay counters, timing and worker memory, `counters_sha256`, the Phase 0.98b attribution columns, the point columns, the error-location decision statistics at the 600-second label horizon with `decision_sha256`, and the class statistics at h*: `class_horizon_seconds`, `class_decisions`, `class_rejections`, `class_resident_evictions`, `class_violations` (resident evictions whose victim is reusable within h* while some sampled resident is not), `class_overridden`, `class_overridden_outside_admission`, each over the evaluation window and with `_seen` over every decision, and `class_m4_count_resident`), plus `kind` (mix_out_learned / mix_in_learned / random_within_class), `learned_class` (0: the ranker orders the non-reusable class; 1: the reusable class; empty for the random arm), `random_stream_seed` (the random arm's `s`) and `random_draws` (the uniforms it drew, one per candidate per decision).
- `replay.csv`: per trace x cell x arm, `mechanism`, `family`, `arm_parameter`, `kind`, `learned_class`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, and mean / std / ci95_half / min / max of every replay metric, statistic and class count (and of `random_draws` for the random arm).
- `references_seeds.csv`: the published reference rows the readings use, one per trace x cell x seed x reference: `source`, `mechanism`, `arm`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`), the digests where published (`counters_sha256`, `decision_sha256`) and `extra_points` (U).
- `differences_seeds.csv` (per seed): `U_<arm>_points` for learned, evict_label, label_binary_hstar, matched_learned, matched_recency (published) and mix_out_learned, mix_in_learned, random_within_class (replays); `R_<arm>_seed` (the seed's own R, descriptive) of every arm but learned and evict_label; `_tokens` / `_points` of `D_out`, `D_in`, `D_rand`, `D_learned`, `random_minus_learned`; and `D_in_plus_D_out_points`.
- `differences.csv` (per trace x cell, readings 1 and 2): `U_<arm>_points_mean` / `_min` / `_max` for every arm above; `R_<arm>` (R_h* on the five-seed means) for label_binary_hstar, matched_learned, matched_recency, mix_out_learned, mix_in_learned, random_within_class; `<d>_points_mean` / `_ci95_half` / `_min` / `_max` and the seed signs of `D_out`, `D_in`, `D_rand`, `D_learned`, `random_minus_learned`; `D_in_negative` (`D_in_points_mean` < 0), `D_out_above_D_in` (`D_out_points_mean` > `D_in_points_mean`), `which_class_condition` (both); `D_in_plus_D_out_points` (the sum of the two means), `D_learned_interval_low` / `_high` (`D_learned_points_mean` -/+ `D_learned_points_ci95_half`) and `additive` (the sum within that interval, bounds included; False when a bound is nan).
- `readings.csv`: the counts. `1_which_class` per `scope` prediction_cells (the eight named trace x cell; `prediction` and `prediction_holds` are prediction 1: `both` in each of the eight) and all (descriptive): `cells`, `D_in_negative`, `D_out_above_D_in`, `both`; `1_additivity` (all): `additive`; `1_D_in_seed_signs` and `1_D_out_seed_signs` per scope: consistent_gain / consistent_loss / mixed; `2_D_rand_seed_signs` (all; prediction 2: consistent_loss in at least 8 of 12, judged only on the full grid); `2_random_minus_learned_seed_signs` (all, descriptive); `context_D_learned_seed_signs` per scope (the published matched control's reading).
- `run_config.json`: plan and code commits, source manifest, trace, model, model-manifest and reference-table hashes, the matched horizons, grid, arms, the random stream's seeding rule and every stream seed used, check results, whether the eight named cells are exactly the published consistent losses of D_learned, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------------

_write = mco._write
_report_lines = mco._report_lines


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
        timing[(row["arm_parameter"], _kind_index(row["arm"]), row["arm"])].append(
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
    random_problems = check_random(rows)
    draws = sum(int(row["random_draws"]) for row in rows if row["kind"] == RANDOM)
    print(f"  random stream: {draws} draws over "
          f"{sum(1 for row in rows if row['kind'] == RANDOM)} random replays, each from its own "
          f"stream: {'hold' if not random_problems else 'BROKEN'}", flush=True)
    _report_lines("RANDOM", random_problems)
    if random_problems:
        failures.append("a random-stream check failed")
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
                               _kind_index(row["arm"]), row["seed"]))
    replay_summary = aggregate_replays(rows)
    references = reference_rows(published, rows)
    tables = mix_tables(rows, published)
    summary = reading_summary(tables)
    named_are_losses = named_cells_are_published_losses(tables)

    print("  reading 1, which class (five-seed means, points; D vs the matched recency arm):",
          flush=True)
    for entry in tables["differences"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"{'named' if entry['prediction_1_cell'] else '     '} "
              f"D_in={entry['D_in_points_mean']:7.3f} ({entry['D_in_seed_signs']}) "
              f"D_out={entry['D_out_points_mean']:7.3f} ({entry['D_out_seed_signs']}) "
              f"condition={entry['which_class_condition']} "
              f"D_learned={entry['D_learned_points_mean']:7.3f} ({entry['D_learned_seed_signs']}) "
              f"additive={entry['additive']}", flush=True)
    print("  reading 2, recency beyond the bit (five-seed means, points):", flush=True)
    for entry in tables["differences"]:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} h*={entry['h_star']:g}s "
              f"D_rand={entry['D_rand_points_mean']:7.3f} ({entry['D_rand_seed_signs']}, "
              f"{entry['D_rand_reading']}) random-learned="
              f"{entry['random_minus_learned_points_mean']:7.3f} "
              f"({entry['random_minus_learned_seed_signs']}, "
              f"{entry['random_minus_learned_reading']})", flush=True)
    for entry in summary:
        counts = ", ".join(f"{column} {entry[column]}" for column in
                           READING_COLUMNS + SIGN_READINGS if entry[column] != "")
        print(f"    {entry['reading']} ({entry['scope']}, {entry['cells']} cells): {counts}"
              + (f"; prediction: {entry['prediction']}: "
                 f"{'holds' if entry['prediction_holds'] else 'FAILS'}"
                 if entry["prediction"] else ""), flush=True)
    print(f"    the named cells are exactly the published consistent losses of D_learned: "
          f"{named_are_losses}", flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    run_arms = sorted({task[3] for task in tasks},
                      key=lambda arm: (HORIZON_OF_ARM[arm], _kind_index(arm)))
    random_arms = [arm for arm in run_arms if KIND_OF_ARM[arm] == RANDOM]
    config = {
        "phase": "class_order_mix",
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
                                     "arms": list(cell_arms(*cell)),
                                     "references": matched_reference_arms(*cell)}
                                    for cell in cells],
        "arms": {arm: {"h_star": HORIZON_OF_ARM[arm], "kind": KIND_OF_ARM[arm],
                       "learned_class": LEARNED_CLASS.get(KIND_OF_ARM[arm], "")}
                 for arm in run_arms},
        "random_stream": {
            "tag": RANDOM_STREAM_TAG,
            "rule": "random.Random(s).random() once per candidate per decision, in the order "
                    "the store scores the candidates; s = int.from_bytes(blake2b((tag, str(seed), "
                    "arm) joined by '\\x1f', utf-8; digest_size=8), 'big') "
                    "(classmix.random_stream_seed = errorloc.hash_bits(tag, seed, arm))",
            "store_sampling_stream": "random.Random(l2_seed) inside the store, unchanged and "
                                     "never read by the random arm's scorer",
            "stream_seeds": {arm: {str(seed): random_stream_seed(seed, arm) for seed in seeds}
                             for arm in random_arms},
        },
        "differences": {name: {"later": later, "earlier": earlier}
                        for name, (later, earlier) in DIFFERENCES.items()},
        "readings": {"prediction_1": PREDICTION_ONE_TEXT,
                     "prediction_1_cells": [list(cell) for cell in PREDICTION_ONE_CELLS],
                     "prediction_2": PREDICTION_TWO_TEXT,
                     "prediction_2_minimum": PREDICTION_TWO_MINIMUM,
                     "additivity": "D_in + D_out (five-seed means) within D_learned mean -/+ "
                                   "its 95% t half-width, bounds included",
                     "sign_readings": list(SIGN_READINGS)},
        "named_cells_equal_published_consistent_losses": named_are_losses,
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "statistics_horizon_seconds": {"decision_statistics": "trace label horizon (600 s)",
                                       "class_statistics": "h* of the cell"},
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "statistics_problems": len(statistics_problems),
            "class_problems": len(class_problems),
            "class_violations": violations,
            "class_overridden_outside_admission": outside,
            "random_problems": len(random_problems),
            "random_draws": draws,
            "invariant_groups": cell_seeds, "invariant_violations": len(invariant_problems),
            "invariant_columns": list(rmc.INVARIANT_COLUMNS),
            "unexplained_absent_tokens": unexplained,
            "attribution_identities": "checked per replay (AttributionCollector.check_against)",
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
        "note": "Descriptive follow-up of the matched class-order control: which reuse class "
                "at h* carries the learned order's loss, and whether recency within a class "
                "carries information beyond the exact bit. h* is the best of grids run on "
                "these traces, read after the fact. Nothing is fitted; every arm reads the "
                "trace's future on purpose and none is a policy. References are published "
                "rows, not reruns. Seed intervals describe sampling-seed and random-stream "
                "variability together.",
    }
    for directory in (args.output_dir, args.paper_dir):
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        for name in ("differences_seeds", "differences"):
            _write(directory / f"{name}.csv", tables[name])
        _write(directory / "readings.csv", summary)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {args.output_dir}, {args.paper_dir}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
