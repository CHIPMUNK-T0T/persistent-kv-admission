#!/usr/bin/env python3
"""Horizon and class-order control: is a reuse label enough, at which horizon, and with whose order.

The pre-registration is `docs/horizon-control-plan.md`; nothing here may change
it. Nine arms are replayed on the Phase 0.97 grid, five seeds, each with the
Phase 0.98b attribution hooks and the decision statistics of the
error-location control attached read-only (`persistent_kv_admission.horizonctl`):

* `all16` (the arrival plus up to 16 uniformly sampled residents; any may
  leave): the exact `binary` label at five horizons, `label_binary_h` for h in
  {6, 15, 60, 300, 600} s, and the two class-order arms `evict_binary_learned`
  and `evict_binary_recency` (the published admission rule on the frozen
  `next_use` ranker's keys; eviction by `(reusable within 600 s, ranker score,
  last_group)` or `(reusable within 600 s, last_group)`);
* `leaf16` (the arrival, if it is a leaf, plus up to 16 sampled leaf
  residents): the error-location hybrids `adm_label` and `evict_label`.

References are published rows, not reruns: `lru`, `learned`, `label` and
`evict_label` under `all16` from `results/paper/error_location_001/`, and
`learned`, `label` under `leaf16` from `results/paper/mechanism_control_001/`,
matched on trace, cell and seed and checked on the arm-independent identifiers
(capacities, requested and L1-avoided tokens, the compulsory-absent charge).

Nothing is fitted and no arm is a proposed policy: every constructed arm reads
the trace's future on purpose. Before anything is derived or published the run
checks that `label_binary_600` reproduces the 60 published `label_binary` rows
exactly, that the identifiers of every replay equal those of every published
reference row of its trace x cell x seed, that no `leaf16` replay has a
present-but-unusable token, that the statistics saw every decision with its
final victim, that the arm-independent counters are arm-independent, and (per
replay) the attribution identities; any failure exits without writing a
derived table.

Outputs (in --output-dir, and in --paper-dir for the full run): per-seed and
aggregated replay tables, the published reference rows used, the three
readings per seed and aggregated, one figure, a README describing every
column, and the run configuration. `--smoke` runs the conversation trace, 1% x
4, seed 0: the nine arms, each once more without the statistics hook, the
identity arms of the plan (admission by the exact 600-second reuse label with
`evict_binary_recency`'s eviction; admission by X, eviction by X under
`leaf16` for X in {learned, label}) and the two `leaf16` rungs they are
compared with; it checks those identities, integrity, runtime and memory,
writes only its own directory (never one under results/paper), and is not to
be interpreted.
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
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.crossworkload import HYPERPARAMETERS
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import (
    DECISION_TYPES,
    LOCATIONS,
    DecisionStatistics,
    RecordingOverride,
    location_label,
)
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.horizonctl import (
    ALL16,
    ARM_MECHANISMS,
    ARMS,
    CLASS_ORDER,
    CLASS_ORDER_ARMS,
    CLASS_ORDER_READINGS,
    CLASS_ORDER_SHARE,
    HORIZON_ARMS,
    HORIZON_OF_ARM,
    HORIZON_READINGS,
    HORIZONS_SECONDS,
    IDENTITY_ARMS,
    IDENTITY_REFERENCE_ARMS,
    LEAF16,
    LEAF_ARMS,
    MECHANISMS,
    PUBLISHED_HORIZON_ARM,
    PUBLISHED_LABEL_BINARY_ARM,
    PUBLISHED_SUBSET_SHORTFALL,
    REUSE_HORIZON_SECONDS,
    SUFFICES_SHORTFALL,
    arm_mechanism,
    arm_setup,
    class_order_reading,
    horizon_reading,
    in_published_subset,
    leaf_location,
    recovery,
    shortfall,
)
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

ERROR_LOCATION_SCRIPT = REPOSITORY / "scripts/run_error_location.py"
PLAN_PATH = REPOSITORY / "docs/horizon-control-plan.md"
PAPER_ROOT = REPOSITORY / "results/paper"


def _load_error_location():
    """Import the error-location runner by path, unchanged, for its replay,
    provenance and table helpers; it brings the mechanism-control runner and
    the Phase 0.97 runner with it (one instance of each, so the capacity rule
    reads the working set this run sets). `main` is behind the usual guard,
    so nothing runs."""
    spec = importlib.util.spec_from_file_location("run_error_location", ERROR_LOCATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rel = _load_error_location()
rmc = rel.rmc
rdp = rel.rdp

# --- the fixed grid (docs/horizon-control-plan.md) ---------------------------------------------

TRACES = rmc.TRACES
CELLS = rmc.CELLS
SEEDS = rmc.SEEDS
LABEL_HORIZON_SECONDS = rmc.LABEL_HORIZON_SECONDS
assert LABEL_HORIZON_SECONDS == REUSE_HORIZON_SECONDS
SMOKE_TRACE = rmc.SMOKE_TRACE
SMOKE_CELL = rmc.SMOKE_CELL
SMOKE_SEEDS = rmc.SMOKE_SEEDS
# Replay variants: every arm runs with the statistics hook ("main"); the smoke
# also runs every grid arm without it ("nostats") to show the hook read-only.
VARIANTS = ("main", "nostats")
# The machine overheats above 12 replay processes.
DEFAULT_WORKERS = 10
MAX_WORKERS = 12
# Published references, anchored at the repository.
ERROR_LOCATION_REFERENCE = REPOSITORY / "results/paper/error_location_001/replay_seeds.csv"
ERROR_LOCATION_LOCATIONS = REPOSITORY / "results/paper/error_location_001/location.csv"
MECHANISM_REFERENCE = REPOSITORY / "results/paper/mechanism_control_001/replay_seeds.csv"
REFERENCE_SOURCES = {"error_location_001": ERROR_LOCATION_REFERENCE,
                     "mechanism_control_001": MECHANISM_REFERENCE}
# Published reference rows, by (mechanism, arm): the source each is read from.
# `label_binary` is the reproduction target of `label_binary_600`.
REFERENCES = {
    (ALL16, "lru"): "error_location_001",
    (ALL16, "learned"): "error_location_001",
    (ALL16, "label"): "error_location_001",
    (ALL16, "evict_label"): "error_location_001",
    (ALL16, PUBLISHED_LABEL_BINARY_ARM): "error_location_001",
    (LEAF16, "learned"): "mechanism_control_001",
    (LEAF16, "label"): "mechanism_control_001",
}
# Arm-independent identifiers every replay must share with every published
# reference row of its trace x cell x seed (a reference that does not publish
# one is not compared on it; both sources publish all five).
REFERENCE_IDENTIFIERS = ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                         "l1_avoided_tokens", "absent_compulsory_tokens")
# Replayed arm -> the published row it must reproduce (`avoided_prefill_tokens`
# exactly). The grid has label_binary_600; the smoke's leaf16 rungs, replayed
# for the identities, are held to their published rows too.
REPRODUCTIONS = {PUBLISHED_HORIZON_ARM: (ALL16, PUBLISHED_LABEL_BINARY_ARM),
                 "learned": (LEAF16, "learned"), "label": (LEAF16, "label")}
STAT_METRICS = rel.STAT_METRICS
REPLAY_METRICS = rel.REPLAY_METRICS
# Arms whose victim is the store's first minimum (no override at all).
NO_OVERRIDE_ARMS = HORIZON_ARMS + IDENTITY_REFERENCE_ARMS
# leaf16 replays whose statistics must be the label identities (the K* key is
# the label rung's key whatever the eligibility).
LABEL_IDENTITY_ARMS = ("label", "hybrid_label_label")
SHARED: dict[str, object] = {}


def arm_family(arm: str) -> str:
    if arm in HORIZON_ARMS:
        return "horizon"
    if arm in CLASS_ORDER_ARMS:
        return "class_order"
    if arm in LEAF_ARMS:
        return "leaf_location"
    if arm in IDENTITY_ARMS:
        return "identity"
    if arm in IDENTITY_REFERENCE_ARMS:
        return "identity_reference"
    raise ValueError(f"unknown arm {arm!r}")


def arm_parameter(arm: str):
    """The horizon h of a `label_binary_h` arm, in seconds; "" otherwise."""
    return HORIZON_OF_ARM.get(arm, "")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path, help="Mooncake trace files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new run directory; refused if it exists")
    parser.add_argument("--paper-dir", type=Path, default=None,
                        help="new publication directory (full run only); refused if it exists")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"replay processes (default {DEFAULT_WORKERS}, at most {MAX_WORKERS})")
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help="number of seeds; the pre-registered grid is 5 (smoke: seed 0)")
    parser.add_argument("--smoke", action="store_true",
                        help="conversation trace, 1%% x 4, seed 0, the nine arms plus the "
                             "identity, leaf16-rung and hook-off replays; never a paper dir; "
                             "not to be interpreted")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="smoke only: run from an uncommitted execution tree")
    return parser.parse_args(argv)


def validate_arguments(args) -> None:
    """Every refusal that needs no trace, model or git call, in one place."""
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} exceeds the hard cap of {MAX_WORKERS}")
    if args.allow_dirty and not args.smoke:
        raise SystemExit("--allow-dirty is for --smoke only; the full run needs a committed tree")
    if args.smoke and args.paper_dir is not None:
        raise SystemExit("--smoke never writes a paper directory; drop --paper-dir")
    if args.smoke and args.output_dir.resolve().is_relative_to(PAPER_ROOT.resolve()):
        raise SystemExit(f"--smoke never writes under {PAPER_ROOT.relative_to(REPOSITORY)}; "
                         "choose another --output-dir")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    if args.paper_dir is not None and args.paper_dir.exists():
        raise SystemExit(f"paper directory {args.paper_dir} exists; choose a new one")
    if not args.smoke and args.seeds != len(SEEDS):
        raise SystemExit(f"the pre-registered grid is seeds {list(SEEDS)}; got {args.seeds}")


# --- provenance ----------------------------------------------------------------------------------


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.update((Path(__file__).resolve(), ERROR_LOCATION_SCRIPT, rel.MECHANISM_SCRIPT,
                  rmc.PHASE097_SCRIPT))
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


# --- inputs ----------------------------------------------------------------------------------------

_cell_seed = rmc._cell_seed


def _reference_entry(row, mechanism: str, arm: str, source: str) -> dict:
    entry = {"source": source, "mechanism": mechanism, "arm": arm, "trace": row["trace"],
             "l1_fraction": float(row["l1_fraction"]),
             "l2_multiplier": float(row["l2_multiplier"]), "cell": row["cell"],
             "seed": int(row["seed"]),
             "avoided_prefill_tokens": int(row["avoided_prefill_tokens"]),
             "extra_avoided_tokens": int(row["extra_avoided_tokens"])}
    for field in REFERENCE_IDENTIFIERS:
        if row.get(field, "") not in ("", None):
            entry[field] = int(row[field])
    return entry


def load_references(paths=None) -> tuple[dict[tuple, dict], list[str]]:
    """`(mechanism, arm, trace, fraction, multiplier, seed) -> published row`
    for every reference of `REFERENCES`, each read from its own source only,
    with the eligibility and width its mechanism names; every reference must
    cover the 60 trace x cell x seed of the grid. `paths` overrides the source
    files (tests)."""
    paths = dict(REFERENCE_SOURCES, **(paths or {}))
    published: dict[tuple, dict] = {}
    problems: list[str] = []
    for source, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                arm = row["arm"] if "arm" in row else row["rung"]
                if REFERENCES.get((row["mechanism"], arm)) != source:
                    continue
                if row.get("variant", "main") != "main":
                    continue
                if (row["eligibility"], int(row["width"])) != MECHANISMS[row["mechanism"]]:
                    problems.append(f"{source} row {row['mechanism']}/{arm} has "
                                    f"{row['eligibility']}/{row['width']}")
                key = (row["mechanism"], arm) + _cell_seed(row)
                if key in published:
                    problems.append(f"duplicate published reference {key}")
                    continue
                published[key] = _reference_entry(row, row["mechanism"], arm, source)
    for (mechanism, arm), source in REFERENCES.items():
        for name in TRACES:
            for fraction, multiplier in CELLS:
                for seed in SEEDS:
                    key = (mechanism, arm, name, fraction, multiplier, seed)
                    if key not in published:
                        problems.append(f"published {source} reference {key} missing")
    return published, problems


def load_published_locations(path=ERROR_LOCATION_LOCATIONS) -> tuple[dict[tuple, dict], list[str]]:
    """The published `all16` location label of every trace x cell (reading 1
    of the error-location control) with its G, A, E and interaction means; the
    label must be what the rule gives on those means."""
    locations: dict[tuple, dict] = {}
    problems: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))
            if key in locations:
                problems.append(f"duplicate published location {key}")
                continue
            entry = {"location": row["location"]}
            for term in rel.LOCATION_TERMS:
                entry[term] = float(row[f"{term}_points_mean"])
            if entry["location"] not in LOCATIONS:
                problems.append(f"{key}: unknown published location {entry['location']!r}")
            elif location_label(entry["G"], entry["A"], entry["E"]) != entry["location"]:
                problems.append(f"{key}: published location {entry['location']} is not the "
                                "rule on its published means")
            locations[key] = entry
    for name in TRACES:
        for fraction, multiplier in CELLS:
            if (name, fraction, multiplier) not in locations:
                problems.append(f"published all16 location of {name}/{fraction}/{multiplier} missing")
    return locations, problems


# --- the replay worker -------------------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one arm under its mechanism, with the attribution hooks
    and (variant "main") the statistics hook around the arm's own override."""
    name, fraction, multiplier, arm, seed, variant = task
    mechanism, eligibility, width = arm_mechanism(arm)
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace, horizon, seed, learned_ranker=SHARED["rankers"][name])
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    statistics = None
    override = setup.override
    if variant == "main":
        statistics = DecisionStatistics(trace, horizon, split_ms)
        override = RecordingOverride(statistics, setup.override)
    elif variant != "nostats":
        raise ValueError(f"unknown variant {variant!r}")
    started = time.time()
    result = run_two_tier(
        trace, rdp.L1_POLICY, l1_bytes, l2_bytes, hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=SHARED["groups"][name], measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=width, l2_seed=seed,
        l2_eligibility=eligibility, l2_arm=arm,
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
        "cell": rdp.cell_label(fraction, multiplier), "eligibility": eligibility,
        "width": width, "mechanism": mechanism, "arm": arm, "family": arm_family(arm),
        "arm_parameter": arm_parameter(arm), "seed": seed, "variant": variant,
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
    if statistics is not None:
        row.update(statistics.row())
    return row


def arm_cost(arm: str) -> int:
    """Scheduling class, most expensive first: the frozen ranker scored by the
    store and again at admission decisions, or with the label beside it; the
    ranker in one role; the label family alone."""
    if arm in ("evict_binary_learned", "adm_label", "hybrid_learned_learned"):
        return 0
    if arm in ("evict_binary_recency", "evict_label", "learned"):
        return 1
    return 2


def build_tasks(names, cells, seeds, smoke: bool) -> list[tuple]:
    """Every replay, most expensive first so the pool's tail is short. The order
    only schedules work; every table is sorted before it is written."""
    tasks = [(name, fraction, multiplier, arm, seed, "main")
             for name in names for fraction, multiplier in cells for arm in ARMS
             for seed in seeds]
    if smoke:
        tasks += [(name, fraction, multiplier, arm, seed, "nostats")
                  for name in names for fraction, multiplier in cells for arm in ARMS
                  for seed in seeds]
        tasks += [(name, fraction, multiplier, arm, seed, "main")
                  for name in names for fraction, multiplier in cells
                  for arm in tuple(IDENTITY_ARMS) + IDENTITY_REFERENCE_ARMS for seed in seeds]
    tasks.sort(key=lambda task: (arm_cost(task[3]), -task[1], -task[2], task[0], task[3],
                                 task[4], task[5]))
    return tasks


# --- checks -------------------------------------------------------------------------------------------


def grid_rows(rows) -> list[dict]:
    """The nine grid arms with the statistics hook: the rows every reading uses."""
    return [row for row in rows if row["variant"] == "main" and row["arm"] in ARMS]


def check_reproduction(rows, published, expected: dict[str, int]) -> dict[str, dict]:
    """Each replayed arm of `expected` (arm -> rows expected) against the
    published row of `REPRODUCTIONS`: `avoided_prefill_tokens` exactly. The
    identifiers are checked for every row by `check_identifiers`."""
    report = {}
    for arm, count in expected.items():
        mechanism, reference_arm = REPRODUCTIONS[arm]
        matched = missing = 0
        mismatches: list[str] = []
        for row in rows:
            if row["arm"] != arm or row["variant"] != "main":
                continue
            reference = published.get((mechanism, reference_arm) + _cell_seed(row))
            if reference is None:
                missing += 1
                row["reference_avoided_prefill_tokens"] = ""
                row["reproduces_reference"] = ""
                continue
            same = int(row["avoided_prefill_tokens"]) == reference["avoided_prefill_tokens"]
            row["reference_source"] = reference["source"]
            row["reference_avoided_prefill_tokens"] = reference["avoided_prefill_tokens"]
            row["reproduces_reference"] = same
            if same:
                matched += 1
            else:
                mismatches.append(f"{row['trace']}/{row['cell']}/s{row['seed']}: "
                                  f"{row['avoided_prefill_tokens']} != "
                                  f"{reference['avoided_prefill_tokens']}")
        report[arm] = {"reference": f"{reference_arm}/{mechanism}", "matched": matched,
                       "missing": missing, "mismatched": len(mismatches), "expected": count,
                       "mismatches": mismatches}
    return report


def check_identifiers(rows, published) -> tuple[int, list[str]]:
    """Every replay against every published reference row of its trace x cell
    x seed, on each identifier the reference publishes. Returns the number of
    (replay, reference) pairs compared and the problems."""
    compared, problems = 0, []
    for row in rows:
        where = f"{row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}/{row['variant']}"
        for mechanism, arm in REFERENCES:
            reference = published.get((mechanism, arm) + _cell_seed(row))
            if reference is None:
                problems.append(f"{where}: no published {arm}/{mechanism} reference row")
                continue
            compared += 1
            for field in REFERENCE_IDENTIFIERS:
                if field in reference and int(row[field]) != reference[field]:
                    problems.append(f"{where}: {field} {row[field]} != {reference[field]} of "
                                    f"the published {arm}/{mechanism} row")
    return compared, problems


def check_leaf_closure(rows) -> list[str]:
    """Under leaf eligibility no block may be present but unusable."""
    return [f"{row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}/{row['variant']}: "
            f"{row['l2_present_unusable_tokens']} tokens, "
            f"{row['l2_present_unusable_blocks']} blocks present but unusable"
            for row in rows
            if row["eligibility"] == "leaf"
            and (int(row["l2_present_unusable_tokens"]) or int(row["l2_present_unusable_blocks"]))]


def check_statistics(rows) -> list[str]:
    """The statistics saw every decision with its final victim; an arm without
    an override never overrode, nor did an identity arm (X = Y, so its rule is
    the store's own); and the leaf16 label replays carry the label identities
    of the error-location control (m1 = m2 = 1, m3 = 0, every m4 victim's next
    use exactly at the horizon), overall and per decision type that occurred."""
    problems = []
    for row in rows:
        if row["variant"] != "main":
            continue
        where = f"{row['trace']}/{row['cell']}/{row['arm']}/s{row['seed']}"
        for mine, theirs in (("stat_decisions_seen", "l2_decisions"),
                             ("stat_rejections_seen", "l2_rejections"),
                             ("stat_evictions_seen", "l2_evictions")):
            if int(row[mine]) != int(row[theirs]):
                problems.append(f"{where}: {mine} {row[mine]} != {theirs} {row[theirs]}")
        if int(row["stat_decisions"]) <= 0:
            problems.append(f"{where}: no decision inside the evaluation window")
        if ((row["arm"] in NO_OVERRIDE_ARMS or row["arm"] in IDENTITY_ARMS)
                and int(row["overridden_decisions_seen"]) != 0):
            problems.append(f"{where}: {row['overridden_decisions_seen']} overridden decisions "
                            "for an arm that must keep the store's choice")
        if row["arm"] not in LABEL_IDENTITY_ARMS:
            continue
        identities = [("m1", 1.0)]
        m4_suffixes = []
        for kind in DECISION_TYPES:
            suffix = f"_{kind}" if kind else ""
            if not kind or int(row[f"stat_decisions{suffix}"]) > 0:
                identities += [(f"m2{suffix}", 1.0), (f"m3{suffix}", 0.0)]
                m4_suffixes.append(suffix)
        for column, value in identities:
            if not float(row[column]) == value:
                problems.append(f"{where}: label identity {column} = {row[column]} != {value}")
        for suffix in m4_suffixes:
            count = int(row[f"m4_count{suffix}"])
            at_horizon = int(row[f"m4_victim_at_horizon{suffix}"])
            if count != at_horizon:
                problems.append(f"{where}: label m4 victims not at the horizon: "
                                f"m4_count{suffix} = {count} != "
                                f"m4_victim_at_horizon{suffix} = {at_horizon}")
    return problems


def check_smoke_identities(rows) -> tuple[list[dict], list[str]]:
    """Smoke only: each identity arm equals its rung counter for counter and
    decision for decision, under the same mechanism, and every grid arm's
    counters are the same with and without the statistics hook."""
    index = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["seed"], row["arm"],
              row["variant"]): row for row in rows}
    table, problems = [], []

    def compare(kind, arm, reference, left, right, digests):
        if left is None or right is None:
            problems.append(f"{kind} {arm} vs {reference}: replay missing")
            return
        same = {name: left[name] == right[name] for name in digests}
        entry = {"trace": left["trace"], "cell": left["cell"], "seed": left["seed"],
                 "mechanism": left["mechanism"], "check": kind, "arm": arm, "against": reference,
                 "seconds": left["seconds"], "against_seconds": right["seconds"],
                 **{f"same_{name}": value for name, value in same.items()}}
        entry["holds"] = all(same.values()) and left["mechanism"] == right["mechanism"]
        table.append(entry)
        if not entry["holds"]:
            problems.append(f"{kind} {arm} vs {reference} on {left['trace']}/{left['cell']}/"
                            f"s{left['seed']}: differs in "
                            + ", ".join([name for name, value in same.items() if not value]
                                        + (["mechanism"] if left["mechanism"]
                                           != right["mechanism"] else [])))

    for cell_seed in sorted({key[:4] for key in index}):
        for arm, reference in IDENTITY_ARMS.items():
            compare("identity", arm, reference, index.get(cell_seed + (arm, "main")),
                    index.get(cell_seed + (reference, "main")),
                    ("counters_sha256", "decision_sha256"))
        for arm in ARMS:
            compare("hook_read_only", arm, f"{arm} without statistics",
                    index.get(cell_seed + (arm, "main")), index.get(cell_seed + (arm, "nostats")),
                    ("counters_sha256",))
    return table, problems


# --- derivation ---------------------------------------------------------------------------------------

_put_stats = rmc._put_stats
_points = rmc._points
_is_nan = rmc._is_nan
_put_signs = rel._put_signs
_difference_points = rel._difference_points
_identity = rel._identity
index_arms = rel.index_arms


def _cell_key(row) -> tuple:
    return (row["trace"], row["l1_fraction"], row["l2_multiplier"])


def _arm_order(arm: str) -> int:
    order = ARMS + tuple(IDENTITY_ARMS) + IDENTITY_REFERENCE_ARMS
    return order.index(arm) if arm in order else len(order)


def _u(entry) -> float:
    """U of one replay or published row: extra avoided tokens over L1 alone,
    in points of window input tokens."""
    return _points(int(entry["extra_avoided_tokens"]), int(entry["requested_tokens"]))


def _reference(published, cell: tuple, seed: int, mechanism: str, arm: str) -> dict:
    key = (mechanism, arm) + tuple(cell) + (seed,)
    if key not in published:
        raise KeyError(f"published reference {key} missing")
    return published[key]


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x arm."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _arm_order(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        arm = key[5]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": ARM_MECHANISMS[arm], "arm": arm, "family": arm_family(arm),
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
        for (mechanism, arm), source in REFERENCES.items():
            entry = dict(published[(mechanism, arm) + cell_seed])
            entry["extra_points"] = _u(entry)
            out.append(entry)
    return out


def published_shortfall(published, cell: tuple) -> float:
    """S_600 of the published rows, `(U(label) - U(label_binary)) / (U(label) -
    U(lru))` on the five-seed means of the published grid (whatever seeds
    this run replays)."""
    means = {}
    for arm in ("label", PUBLISHED_LABEL_BINARY_ARM, "lru"):
        means[arm] = float(np.mean([_u(_reference(published, cell, seed, ALL16, arm))
                                    for seed in SEEDS]))
    return shortfall(means["label"], means[PUBLISHED_LABEL_BINARY_ARM], means["lru"])


def horizon_tables(rows, published) -> tuple[list[dict], list[dict], list[dict]]:
    """Reading 1. Per seed and h: U(label), U(lru), U(label_binary_h) and
    `U(label) - U(label_binary_h)`; per trace x cell x h: the five-seed means,
    the difference with its seed signs and S_h on the means; per trace x cell:
    S_h of every h, the smallest, the horizons attaining it, the reading, the
    learned level S(learned) and the published S_600 with the subset flag."""
    seed_rows, table, readings = [], [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        references = {arm: [_reference(published, cell, seed, ALL16, arm) for seed in seeds]
                      for arm in ("label", "lru", "learned")}
        u_ref = {arm: [_u(entry) for entry in entries] for arm, entries in references.items()}
        mean_ref = {arm: float(np.mean(values)) for arm, values in u_ref.items()}
        shortfalls = {}
        for h, arm in zip(HORIZONS_SECONDS, HORIZON_ARMS):
            differences, values = [], []
            for position, seed in enumerate(seeds):
                row = by_seed[seed][arm]
                tokens, difference = _difference_points(references["label"][position], row)
                differences.append(difference)
                values.append(_u(row))
                seed_rows.append({**identity, "h": h, "arm": arm, "seed": seed,
                                  "U_label_points": u_ref["label"][position],
                                  "U_lru_points": u_ref["lru"][position],
                                  "U_learned_points": u_ref["learned"][position],
                                  "U_arm_points": values[-1],
                                  "label_minus_arm_tokens": tokens,
                                  "label_minus_arm_points": difference,
                                  "S_seed": shortfall(u_ref["label"][position], values[-1],
                                                      u_ref["lru"][position])})
            entry = {**identity, "h": h, "arm": arm, "seeds": len(seeds),
                     "U_label_points_mean": mean_ref["label"], "U_lru_points_mean": mean_ref["lru"]}
            _put_stats(entry, "U_arm_points", values, fields=("mean", "ci95_half", "min", "max"))
            _put_stats(entry, "label_minus_arm_points", differences,
                       fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, "label_minus_arm", differences)
            entry["S"] = shortfall(mean_ref["label"], entry["U_arm_points_mean"], mean_ref["lru"])
            shortfalls[h] = entry["S"]
            table.append(entry)
        reading, smallest, attaining = horizon_reading(shortfalls)
        published_s600 = published_shortfall(published, cell)
        entry = {**identity, "seeds": len(seeds)}
        for h in HORIZONS_SECONDS:
            entry[f"S_{h:g}"] = shortfalls[h]
        entry.update(S_min=smallest, S_min_horizons=";".join(f"{h:g}" for h in attaining),
                     reading=reading,
                     U_label_points_mean=mean_ref["label"], U_lru_points_mean=mean_ref["lru"],
                     U_learned_points_mean=mean_ref["learned"],
                     S_learned=shortfall(mean_ref["label"], mean_ref["learned"], mean_ref["lru"]),
                     published_S_600=published_s600,
                     in_published_subset=in_published_subset(published_s600))
        readings.append(entry)
    return seed_rows, table, readings


def class_order_tables(rows, published) -> tuple[list[dict], list[dict]]:
    """Reading 2. Per seed: U(learned), U(evict_label) (published, all16),
    U(evict_binary_learned), U(evict_binary_recency), their seed-paired
    difference and the seed R values; per trace x cell: the five-seed means,
    R(a) of both arms on the means, the reading, and the difference with its
    seed signs and consistent-gain / consistent-loss / mixed reading."""
    seed_rows, table = [], []
    learned_arm, recency_arm = CLASS_ORDER_ARMS
    for cell, by_seed in sorted(index_arms(rows).items()):
        seeds = sorted(by_seed)
        identity = _identity(*cell)
        u = {arm: [] for arm in ("learned", "evict_label") + CLASS_ORDER_ARMS}
        differences = []
        for seed in seeds:
            learned = _reference(published, cell, seed, ALL16, "learned")
            evict = _reference(published, cell, seed, ALL16, "evict_label")
            arms = by_seed[seed]
            values = {"learned": _u(learned), "evict_label": _u(evict),
                      learned_arm: _u(arms[learned_arm]), recency_arm: _u(arms[recency_arm])}
            for arm, value in values.items():
                u[arm].append(value)
            tokens, difference = _difference_points(arms[learned_arm], arms[recency_arm])
            differences.append(difference)
            seed_rows.append({**identity, "seed": seed,
                              **{f"U_{arm}_points": value for arm, value in values.items()},
                              "learned_minus_recency_tokens": tokens,
                              "learned_minus_recency_points": difference,
                              **{f"R_{arm}_seed": recovery(values[arm], values["learned"],
                                                           values["evict_label"])
                                 for arm in CLASS_ORDER_ARMS}})
        entry = {**identity, "seeds": len(seeds)}
        for arm, values in u.items():
            _put_stats(entry, f"U_{arm}_points", values, fields=("mean", "min", "max"))
        for arm in CLASS_ORDER_ARMS:
            entry[f"R_{arm}"] = recovery(entry[f"U_{arm}_points_mean"],
                                         entry["U_learned_points_mean"],
                                         entry["U_evict_label_points_mean"])
        entry["reading"] = class_order_reading(entry[f"R_{learned_arm}"])
        _put_stats(entry, "learned_minus_recency_points", differences,
                   fields=("mean", "ci95_half", "min", "max"))
        _put_signs(entry, "learned_minus_recency", differences)
        table.append(entry)
    return seed_rows, table


LOCATION_TERMS = rel.LOCATION_TERMS


def leaf_location_tables(rows, published, locations) -> tuple[list[dict], list[dict]]:
    """Reading 3. Under leaf16, `G = U(label) - U(learned)`, `A = U(adm_label)
    - U(learned)`, `E = U(evict_label) - U(learned)` and `A + E - G` per seed
    and on the five-seed means with seed signs, the location label of the
    means, next to the published all16 label (and its G, A, E) of the same
    trace x cell, and whether the two labels are the same."""
    seed_rows, table = [], []
    for cell, by_seed in sorted(index_arms(rows).items()):
        identity = _identity(*cell)
        terms: dict[str, list[float]] = {term: [] for term in LOCATION_TERMS}
        seed_labels = []
        for seed in sorted(by_seed):
            arms = by_seed[seed]
            learned = _reference(published, cell, seed, LEAF16, "learned")
            label = _reference(published, cell, seed, LEAF16, "label")
            g_tokens, g = _difference_points(label, learned)
            a_tokens, a = _difference_points(arms["adm_label"], learned)
            e_tokens, e = _difference_points(arms["evict_label"], learned)
            i_tokens = a_tokens + e_tokens - g_tokens
            interaction = _points(i_tokens, int(learned["requested_tokens"]))
            seed_rows.append({**identity, "mechanism": LEAF16, "seed": seed,
                              "G_tokens": g_tokens, "A_tokens": a_tokens, "E_tokens": e_tokens,
                              "interaction_tokens": i_tokens, "G_points": g, "A_points": a,
                              "E_points": e, "interaction_points": interaction,
                              "seed_location": location_label(g, a, e)})
            seed_labels.append(seed_rows[-1]["seed_location"])
            for term, value in zip(LOCATION_TERMS, (g, a, e, interaction)):
                terms[term].append(value)
        entry = {**identity, "mechanism": LEAF16, "seeds": len(terms["G"])}
        for term in LOCATION_TERMS:
            _put_stats(entry, f"{term}_points", terms[term],
                       fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, term, terms[term])
        g, a, e = (entry[f"{term}_points_mean"] for term in ("G", "A", "E"))
        entry["A_share_of_G"] = a / g if g else math.nan
        entry["E_share_of_G"] = e / g if g else math.nan
        published_all16 = locations[cell]
        entry["location"], entry["same_as_all16"] = leaf_location(g, a, e,
                                                                  published_all16["location"])
        entry["seed_locations"] = "|".join(seed_labels)
        entry["all16_location"] = published_all16["location"]
        for term in LOCATION_TERMS:
            entry[f"all16_{term}_points_mean"] = published_all16[term]
        table.append(entry)
    return seed_rows, table


# --- figure ------------------------------------------------------------------------------------------

BLUE, ORANGE, AQUA = rel.BLUE, rel.ORANGE, rel.AQUA
INK, MUTED, GRID = rmc.INK, rmc.MUTED, rmc.GRID
_style_axis = rmc._style_axis


def horizon_figure(directory: Path, readings: list[dict], seed_rows: list[dict]) -> None:
    """S_h against h per trace x cell (S on the five-seed means; band = seed
    min-max of the per-seed S), with the label (S = 0), sampled LRU (S = 1) and
    frozen-ranker (S of `learned`) levels as horizontal lines and the 0.10
    reading threshold."""
    traces, cells = rmc._panel_grid(readings)
    index = {_cell_key(entry): entry for entry in readings}
    seeds_of: dict[tuple, list[float]] = defaultdict(list)
    for row in seed_rows:
        seeds_of[_cell_key(row) + (row["h"],)].append(row["S_seed"])
    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(7.0, 3.0 * len(cells) + 0.6), 2.9 * len(traces) + 1.2))
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
            axis.plot(HORIZONS_SECONDS, values, color=BLUE, marker="o", markersize=4.5,
                      linewidth=2.0, label="S_h of label_binary_h")
            for level, color, style, label in ((0.0, INK, ":", "label (S = 0)"),
                                               (entry["S_learned"], ORANGE, "--",
                                                "learned (frozen pi0)"),
                                               (1.0, MUTED, "-.", "sampled LRU (S = 1)")):
                axis.axhline(level, color=color, linestyle=style, linewidth=1.3, label=label)
            axis.axhline(SUFFICES_SHORTFALL, color=AQUA, linestyle=(0, (1, 2)), linewidth=1.2,
                         label=f"reading threshold S = {SUFFICES_SHORTFALL:g}")
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
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    figure.suptitle("Exact reuse label at horizon h (all16): shortfall from the next_use label "
                    "(S on five-seed means; band = seed min-max)", fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.10 if len(traces) > 1 else 0.17, 1.0, 0.95))
    figure.savefig(directory / "horizon.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


README_TEXT = """# Horizon and class-order control: is a reuse label enough, at which horizon, and with whose order

Pre-registration: `docs/horizon-control-plan.md`. Builds on the error-location control (`docs/error-location-plan.md`, `results/paper/error_location_001/`). Nine arms, 2 traces x 6 cells x 5 seeds, each replay with the Phase 0.98b attribution hooks and the error-location decision statistics (read-only). Mechanisms: `all16` (the arrival plus up to 16 uniformly sampled residents; any candidate may leave) and `leaf16` (the arrival if it is a leaf, plus up to 16 sampled leaf residents; a candidate with a cached child may not leave). Arms: `label_binary_h` for h in 6, 15, 60, 300, 600 s (`all16`; key `(1 if the next use is at most h s away else 0, last_group)`, the exact `binary` target at h; h = 600 is the published `label_binary`); `evict_binary_learned` and `evict_binary_recency` (`all16`; the arrival is rejected iff it is the first minimum of the frozen `next_use` ranker's keys `(score, last_group)` over the candidates; otherwise the victim is the first minimum, over the candidates other than the arrival, of `(reusable within 600 s, ranker score, last_group)` or `(reusable within 600 s, last_group)`; every later round uses that key); `adm_label` and `evict_label` (`leaf16`; the error-location hybrids, admission by the label and eviction by the ranker, or the reverse; an admission decision is a first round in which the arrival is a candidate, so a non-leaf arrival is never judged by X). References are published rows, not reruns: `lru`, `learned`, `label`, `evict_label` (`all16`, error_location_001) and `learned`, `label` (`leaf16`, mechanism_control_001). U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. Every constructed arm reads the future on purpose and none is a policy. Seed intervals describe sampling-seed variability only.

Common identifier columns: `trace`, `l1_fraction`, `l2_multiplier`, `cell` (`l1=<fraction>,l2x<multiplier>`), `seed`, `seeds` (seeds aggregated). Aggregates carry `<metric>_mean`, `_std`, `_ci95_half` (95% t half-width over seeds), `_min`, `_max` (a subset where stated). Seed signs of a seed-paired difference `<d>`: `<d>_n_pos`, `<d>_n_zero`, `<d>_n_neg`, `<d>_seed_signs` (one of + 0 - per seed, in seed order) and `<d>_reading` (consistent_gain: every seed > 0; consistent_loss: every seed < 0; mixed otherwise).

- `replay_seeds.csv`: one row per replay. Identity: `eligibility`, `width`, `mechanism`, `arm`, `family` (horizon / class_order / leaf_location; smoke also identity / identity_reference), `arm_parameter` (h in seconds for `label_binary_h`), `variant` (main: with the statistics hook; smoke also nostats). Replay counters (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `measured_requests`, `l1_avoided_tokens`, `l2_avoided_tokens`, `avoided_prefill_tokens`, `extra_avoided_tokens`, `l2_present_unusable_tokens`/`_blocks`, `l2_admissions`, `l2_rejections`, `l2_evictions`, `l2_decisions`, `l1_evictions`), timing and worker memory (`seconds`, `worker_peak_rss_mib`, `worker_pss_mib_end`), `counters_sha256` (every replay and attribution counter), the Phase 0.98b attribution columns (`attribution_*` .. `orphaned_blocks_per_eviction`; `root_*`, `unusable_after_*`, `absent_*`, `downstream_*` per loss category rejected / evicted / compulsory / unexplained), the same token counters in points (`extra_points`, `*_points`), `absent_*_share_of_decision_absent`, and the error-location decision statistics of the arm's own decisions in the evaluation window against `K* = (next_use label, last_group)` (`stat_*`, `overridden_*`, `decision_sha256`, `m1_pairs`, `m1_concordant`, `m1_tied`, `m1`, `m2_agree`, `m2`, `m3_excess_sum`, `m3`, `m4_count`, `m4`, `m4_victim_at_horizon`, each of m2-m4 also `_admission` and `_resident`; m1 uses the store's key, Y's for the hybrids and the class-order arms; `overridden_*` counts admission decisions where X's choice differs from Y's). `label_binary_600` rows (and in the smoke the leaf16 `learned` / `label` rows) also carry `reference_source`, `reference_avoided_prefill_tokens` and `reproduces_reference`.
- `replay.csv`: per trace x cell x arm, `mechanism`, `family`, `arm_parameter`, the cell's `requested_tokens`, `l1_avoided_tokens`, capacities, and mean / std / ci95_half / min / max of every replay metric and statistic.
- `references_seeds.csv`: the published reference rows the readings use, one per trace x cell x seed x reference: `source`, `mechanism`, `arm`, `avoided_prefill_tokens`, `extra_avoided_tokens`, the identifiers checked against every replay (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`) and `extra_points` (U).
- `horizon_seeds.csv` (reading 1, per seed): per trace x cell x `h` x seed, `arm`, `U_label_points`, `U_lru_points`, `U_learned_points` (published, all16), `U_arm_points` (`label_binary_h`), `label_minus_arm_tokens` / `_points` (U(label) - U(label_binary_h)) and `S_seed` (the seed's own shortfall, descriptive).
- `horizon.csv` (reading 1, per h): per trace x cell x `h`, `arm`, `U_label_points_mean`, `U_lru_points_mean`, `U_arm_points_*` (mean, ci95_half, min, max), `label_minus_arm_points_*` with its seed signs (`label_minus_arm_*`), and `S` = (U(label) - U(label_binary_h)) / (U(label) - U(lru)) on the five-seed means (nan when the denominator is zero).
- `horizon_reading.csv` (reading 1): per trace x cell, `S_6` .. `S_600`, `S_min` (the smallest), `S_min_horizons` (every h attaining it, ";"-joined), `reading` (reuse_label_suffices when S_min <= 0.10, else order_needed), the five-seed means `U_label_points_mean`, `U_lru_points_mean`, `U_learned_points_mean`, `S_learned` (the frozen ranker's own shortfall, the figure's learned level), `published_S_600` (from the published label_binary, label and lru rows, five seeds) and `in_published_subset` (published_S_600 > 0.05; the reading is counted out of 12 and separately over this subset).
- `class_order_seeds.csv` (reading 2, per seed): `U_learned_points`, `U_evict_label_points` (published, all16), `U_evict_binary_learned_points`, `U_evict_binary_recency_points`, `learned_minus_recency_tokens` / `_points` (U(evict_binary_learned) - U(evict_binary_recency)), and the seed's own `R_<arm>_seed` (descriptive).
- `class_order.csv` (reading 2): per trace x cell, `U_<arm>_points_mean` / `_min` / `_max` of the four, `R_evict_binary_learned` and `R_evict_binary_recency` = (U(a) - U(learned)) / (U(evict_label) - U(learned)) on the five-seed means (nan when the denominator is zero), `reading` (reuse_identification_suffices when R(evict_binary_learned) >= 0.9, else ranker_order_costs), `learned_minus_recency_points_*` and its seed signs (`learned_minus_recency_*`, read consistent_gain / consistent_loss / mixed).
- `location_leaf_seeds.csv` (reading 3, per seed): under leaf16, `G_tokens` / `G_points` = U(label) - U(learned), `A_*` = U(adm_label) - U(learned), `E_*` = U(evict_label) - U(learned) (label and learned published, leaf16), `interaction_*` = A + E - G, and `seed_location`.
- `location_leaf.csv` (reading 3): per trace x cell, `G`, `A`, `E`, `interaction` `_points_*` with seed signs, `A_share_of_G`, `E_share_of_G`, `location` (admission_located: A >= G/2 and E < G/2; eviction_located: the reverse; both; neither), `seed_locations`, `all16_location` (published, error_location_001 `location.csv`) with `all16_G` .. `all16_interaction_points_mean`, and `same_as_all16`.
- `horizon.png`: S_h against h per trace x cell (line: S on five-seed means; band: seed min-max of S_seed), with the label (S = 0), learned and sampled-LRU (S = 1) levels and the 0.10 threshold.
- `run_config.json`: plan and code commits, source manifest, trace, model and reference hashes, grid, arms, mechanisms, check counts, timing and memory.
"""

SMOKE_README = """# SMOKE RUN — NOT TO BE INTERPRETED

This directory is the horizon-control smoke (conversation trace, L1 1% x L2 4, seed 0). It checks integrity, the identities of the plan (admission by the exact 600-second reuse label with `evict_binary_recency`'s eviction equals `label_binary_600`; under leaf16, admission by X, eviction by X equals rung X for X in learned and label, the leaf16 rungs replayed here and held to their published rows), that the statistics hook is read-only (every grid arm replayed with and without it), runtime and memory. It is excluded from every confirmatory table; one cell and one seed cannot carry a reading. `smoke_identities.csv` lists each identity and hook check; `replay_seeds.csv` holds the identity, leaf16-rung and hook-off replays too (`variant`, `arm`, `family`).

"""


# --- main ---------------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


def _count(entries, column, value) -> int:
    return sum(1 for entry in entries if entry[column] == value)


def main() -> None:
    args = parse_args()
    clock = time.time()
    validate_arguments(args)
    seeds = SMOKE_SEEDS if args.smoke else tuple(range(args.seeds))

    status = rmc._git("status", "--porcelain")
    differing = sources_differing_from_head()
    if not args.allow_dirty:
        if status:
            raise SystemExit(f"the working tree is not clean:\n{status}")
        if differing:
            raise SystemExit(f"execution source differs from HEAD: {differing}")
    head_start = rmc._git("rev-parse", "HEAD")
    plan_commit = rmc._git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()
    if args.allow_dirty and (status or differing):
        print(f"WARNING smoke from a dirty tree: {len(status.splitlines())} status lines, "
              f"{len(differing)} execution files differ from HEAD", flush=True)
    if args.smoke:
        print("SMOKE RUN: integrity, identities, runtime and memory only; not to be interpreted",
              flush=True)

    traces, groups, splits, horizons, working_set, trace_files = {}, {}, {}, {}, {}, {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace {trace.name}")
        if args.smoke and trace.name != SMOKE_TRACE:
            print(f"smoke: skipping {trace.name}", flush=True)
            continue
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
    expected_names = {SMOKE_TRACE} if args.smoke else set(TRACES)
    if set(traces) != expected_names:
        raise SystemExit(f"traces {sorted(traces)} do not match the grid {sorted(expected_names)}")
    names = tuple(sorted(traces))
    cells = (SMOKE_CELL,) if args.smoke else CELLS
    rankers, model_files = rmc.load_models(names)
    published, reference_problems = load_references()
    locations, location_problems = load_published_locations()
    for name in names:
        print(f"  pi0 next_use {name}: {model_files[name]['path']} "
              f"sha256={model_files[name]['sha256'][:12]}", flush=True)
    print(f"  published references: {len(published)} rows of "
          f"{', '.join(f'{arm}/{mechanism}' for mechanism, arm in REFERENCES)}; "
          f"{len(locations)} published all16 locations", flush=True)
    if reference_problems or location_problems:
        for line in (reference_problems + location_problems)[:20]:
            print(f"    REFERENCE {line}", flush=True)
        raise SystemExit("published references are not what the plan names; nothing run")

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                  rankers=rankers)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds, args.smoke)
    workers = min(args.workers, len(tasks))
    grid_count = len(names) * len(cells) * len(ARMS) * len(seeds)
    print(f"{len(tasks)} replays on {workers} workers "
          f"({len(names)} traces x {len(cells)} cells x {len(ARMS)} arms x {len(seeds)} seeds"
          + (f", plus {len(tasks) - grid_count} smoke identity, leaf16-rung and hook-off replays"
             if args.smoke else "") + ")", flush=True)
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
        timing[(_arm_order(row["arm"]), row["arm"], row["variant"])].append(float(row["seconds"]))
    print("  seconds per replay, by arm and variant:", flush=True)
    timing_rows = []
    for (_, arm, variant), values in sorted(timing.items()):
        print(f"    {arm:40s} {ARM_MECHANISMS[arm]:6s} {variant:8s} n={len(values):4d} "
              f"mean={np.mean(values):7.1f} max={np.max(values):7.1f} "
              f"total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"arm": arm, "mechanism": ARM_MECHANISMS[arm], "variant": variant,
                            "n": len(values), "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    per_arm = len(names) * len(cells) * len(seeds)
    expected = {PUBLISHED_HORIZON_ARM: per_arm}
    if args.smoke:
        expected.update({arm: per_arm for arm in IDENTITY_REFERENCE_ARMS})
    reproduction = check_reproduction(rows, published, expected)
    for arm, report in reproduction.items():
        print(f"  reproduction ({arm}/{ARM_MECHANISMS[arm]} vs published {report['reference']}): "
              f"{report['matched']} matched, {report['missing']} missing, "
              f"{report['mismatched']} mismatched (expected {report['expected']})", flush=True)
        for line in report["mismatches"]:
            print(f"    MISMATCH {line}", flush=True)
        if report["mismatched"] or report["missing"] or report["matched"] != report["expected"]:
            failures.append(f"reproduction of {report['reference']} by {arm} failed")
    compared, identifier_problems = check_identifiers(rows, published)
    print(f"  identifiers ({', '.join(REFERENCE_IDENTIFIERS)}) of {len(rows)} replays against "
          f"{compared} published reference rows of their trace x cell x seed: "
          f"{'equal' if not identifier_problems else 'DIFFER'}", flush=True)
    for line in identifier_problems[:20]:
        print(f"    IDENTIFIER {line}", flush=True)
    if identifier_problems:
        failures.append("an identifier differs from a published reference row")
    closure_problems = check_leaf_closure(rows)
    leaf_rows = sum(1 for row in rows if row["eligibility"] == "leaf")
    print(f"  leaf16 closure: present-but-unusable tokens in {leaf_rows} leaf16 replays: "
          f"{'none' if not closure_problems else 'FOUND'}", flush=True)
    for line in closure_problems[:20]:
        print(f"    CLOSURE {line}", flush=True)
    if closure_problems:
        failures.append("a leaf16 replay left a block present but unusable")
    statistics_problems = check_statistics(rows)
    label_rows = sum(1 for row in rows if row["variant"] == "main"
                     and row["arm"] in LABEL_IDENTITY_ARMS)
    print(f"  statistics: every decision seen with its final victim, no override where none is "
          f"allowed, label identities in {label_rows} leaf16 label replays: "
          f"{'hold' if not statistics_problems else 'BROKEN'}", flush=True)
    for line in statistics_problems[:20]:
        print(f"    STATISTICS {line}", flush=True)
    if statistics_problems:
        failures.append("a statistics check failed")
    main_rows = grid_rows(rows)
    cell_seeds, invariant_problems = rmc.check_invariants(main_rows, len(ARMS))
    print(f"  invariants over {cell_seeds} trace x cell x seed groups "
          f"({', '.join(rmc.INVARIANT_COLUMNS)}): "
          f"{'hold' if not invariant_problems else 'BROKEN'}", flush=True)
    for line in invariant_problems[:20]:
        print(f"    VARIES {line}", flush=True)
    if invariant_problems:
        failures.append("an arm-independent counter varies across arms")
    unexplained = sum(int(row["absent_unexplained_tokens"]) for row in rows)
    print(f"  unexplained per-block absent tokens over the run: {unexplained}", flush=True)
    identity_table, identity_problems = [], []
    if args.smoke:
        identity_table, identity_problems = check_smoke_identities(rows)
        held = sum(1 for entry in identity_table if entry["holds"])
        print(f"  smoke identities and hook-off replays: {held}/{len(identity_table)} hold",
              flush=True)
        for line in identity_problems:
            print(f"    IDENTITY {line}", flush=True)
        if identity_problems:
            failures.append("a smoke identity or the hook-off comparison failed")
        for entry in identity_table:
            if entry["check"] == "hook_read_only":
                print(f"    {entry['arm']:24s} seconds with statistics {entry['seconds']:6.1f}, "
                      f"without {entry['against_seconds']:6.1f}", flush=True)
    if failures:
        raise SystemExit("checks failed, nothing derived or published: " + "; ".join(failures))

    # --- derived tables (grid arms with the statistics hook only) ---
    key = lambda row: (row["trace"], row["l1_fraction"], row["l2_multiplier"],
                       _arm_order(row["arm"]), row["arm"], VARIANTS.index(row["variant"]),
                       row["seed"])
    rows.sort(key=key)
    main_rows = grid_rows(rows)
    replay_summary = aggregate_replays(main_rows)
    references = reference_rows(published, main_rows)
    horizon_seeds, horizon, horizon_readings = horizon_tables(main_rows, published)
    class_seeds, class_order = class_order_tables(main_rows, published)
    leaf_seeds, leaf = leaf_location_tables(main_rows, published, locations)

    total_cells = len(names) * len(cells)
    subset = [entry for entry in horizon_readings if entry["in_published_subset"]]
    print("  reading 1, horizon (S_h on five-seed means):", flush=True)
    for entry in horizon_readings:
        values = " ".join(f"{entry[f'S_{h:g}']:7.3f}" for h in HORIZONS_SECONDS)
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} S(6,15,60,300,600)={values} "
              f"min={entry['S_min']:.3f} at h={entry['S_min_horizons'] or '-'} -> "
              f"{entry['reading']} (published S_600 {entry['published_S_600']:.3f}"
              f"{', subset' if entry['in_published_subset'] else ''})", flush=True)
    print("    " + ", ".join(f"{reading} {_count(horizon_readings, 'reading', reading)}/{total_cells}"
                             for reading in HORIZON_READINGS)
          + f"; over the {len(subset)} trace x cell with published S_600 > "
          f"{PUBLISHED_SUBSET_SHORTFALL:g}: "
          + ", ".join(f"{reading} {_count(subset, 'reading', reading)}/{len(subset)}"
                      for reading in HORIZON_READINGS), flush=True)
    print("  reading 2, class order:", flush=True)
    for entry in class_order:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} "
              f"R(evict_binary_learned)={entry['R_evict_binary_learned']:7.3f} "
              f"R(evict_binary_recency)={entry['R_evict_binary_recency']:7.3f} -> "
              f"{entry['reading']}; U(learned order) - U(recency) = "
              f"{entry['learned_minus_recency_points_mean']:7.3f} "
              f"({entry['learned_minus_recency_seed_signs']}, "
              f"{entry['learned_minus_recency_reading']})", flush=True)
    print("    " + ", ".join(f"{reading} {_count(class_order, 'reading', reading)}/{total_cells}"
                             for reading in CLASS_ORDER_READINGS)
          + "; learned minus recency: " + ", ".join(
              f"{reading} {_count(class_order, 'learned_minus_recency_reading', reading)}/"
              f"{total_cells}" for reading in ("consistent_gain", "consistent_loss", "mixed")),
          flush=True)
    print("  reading 3, location under leaf16:", flush=True)
    for entry in leaf:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} G={entry['G_points_mean']:7.3f} "
              f"A={entry['A_points_mean']:7.3f} E={entry['E_points_mean']:7.3f} "
              f"A+E-G={entry['interaction_points_mean']:7.3f} -> {entry['location']} "
              f"(all16: {entry['all16_location']}, same: {entry['same_as_all16']})", flush=True)
    print(f"    same label as all16: {_count(leaf, 'same_as_all16', True)}/{total_cells}",
          flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "horizon_control",
        "smoke": bool(args.smoke),
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "git_dirty_at_start": bool(status),
        "execution_sources_differing_from_head": differing,
        "allow_dirty": bool(args.allow_dirty),
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "models": {"pi0_next_use": model_files},
        "model_manifests": {str(path.relative_to(REPOSITORY)): sha256_path(path)
                            for path in rmc.MODEL_MANIFESTS},
        "references": {str(path.relative_to(REPOSITORY)): sha256_path(path)
                       for path in (ERROR_LOCATION_REFERENCE, ERROR_LOCATION_LOCATIONS,
                                    MECHANISM_REFERENCE)},
        "reference_rows": {f"{arm}/{mechanism}": source
                           for (mechanism, arm), source in REFERENCES.items()},
        "reference_identifiers": list(REFERENCE_IDENTIFIERS),
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "arms": {arm: ARM_MECHANISMS[arm] for arm in ARMS},
        "mechanisms": {name: {"eligibility": eligibility, "width": width}
                       for name, (eligibility, width) in MECHANISMS.items()},
        "horizons_seconds_of_label_binary_h": list(HORIZONS_SECONDS),
        "reuse_horizon_seconds": REUSE_HORIZON_SECONDS,
        "class_order": dict(CLASS_ORDER),
        "identity_arms": ({arm: {"against": reference, "mechanism": ARM_MECHANISMS[arm]}
                           for arm, reference in IDENTITY_ARMS.items()} if args.smoke else {}),
        "identity_reference_arms": ({arm: ARM_MECHANISMS[arm] for arm in IDENTITY_REFERENCE_ARMS}
                                    if args.smoke else {}),
        "readings": {"suffices_shortfall": SUFFICES_SHORTFALL,
                     "published_subset_shortfall": PUBLISHED_SUBSET_SHORTFALL,
                     "class_order_share": CLASS_ORDER_SHARE,
                     "horizon_labels": list(HORIZON_READINGS),
                     "class_order_labels": list(CLASS_ORDER_READINGS),
                     "location_labels": list(LOCATIONS)},
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {arm: {k: v for k, v in report.items() if k != "mismatches"}
                             for arm, report in reproduction.items()},
            "identifier_pairs_compared": compared,
            "identifier_problems": len(identifier_problems),
            "leaf16_replays": leaf_rows, "leaf16_closure_problems": len(closure_problems),
            "statistics_problems": len(statistics_problems),
            "label_identity_replays": label_rows,
            "invariant_groups": cell_seeds, "invariant_violations": len(invariant_problems),
            "invariant_columns": list(rmc.INVARIANT_COLUMNS),
            "unexplained_absent_tokens": unexplained,
            "smoke_identity_checks": len(identity_table),
            "smoke_identity_failures": len(identity_problems),
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
        "note": ("SMOKE RUN, NOT TO BE INTERPRETED. " if args.smoke else "")
                + "Descriptive control. Nothing is fitted; every constructed arm reads the trace's "
                  "future on purpose and none is a policy. References are published rows, not "
                  "reruns; label_binary_600 reproduces the published label_binary rows. A reuse "
                  "label that suffices at a horizon chosen from five after the run describes "
                  "that grid only. Seed intervals describe sampling-seed variability only.",
    }
    destinations = [args.output_dir]
    if args.paper_dir is not None and not args.smoke:
        destinations.append(args.paper_dir)
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows if args.smoke else main_rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "references_seeds.csv", references)
        _write(directory / "horizon_seeds.csv", horizon_seeds)
        _write(directory / "horizon.csv", horizon)
        _write(directory / "horizon_reading.csv", horizon_readings)
        _write(directory / "class_order_seeds.csv", class_seeds)
        _write(directory / "class_order.csv", class_order)
        _write(directory / "location_leaf_seeds.csv", leaf_seeds)
        _write(directory / "location_leaf.csv", leaf)
        if args.smoke:
            _write(directory / "smoke_identities.csv", identity_table)
        horizon_figure(directory, horizon_readings, horizon_seeds)
        (directory / "README.md").write_text((SMOKE_README if args.smoke else "") + README_TEXT,
                                             encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {', '.join(str(d) for d in destinations)}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
