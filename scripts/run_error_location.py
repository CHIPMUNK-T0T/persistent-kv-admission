#!/usr/bin/env python3
"""Error-location control: which decisions and which errors separate a ranker from its label.

The pre-registration is `docs/error-location-plan.md`; nothing here may change
it. Through the published sampled mechanism only (`all16`: the arrival plus up
to 16 uniformly sampled residents, any of which may leave), 18 arms are
replayed on the Phase 0.97 grid, five seeds, with the Phase 0.98b attribution
hooks attached read-only: the four reference rungs of the mechanism control,
the real rankers it did not include (`pi3_next_use`, `pi0_binary`,
`pi3_binary`, loaded by their published SHA-256), the exact `binary` label,
the two decision-type hybrids of label and frozen ranker, the label with
Gaussian noise at four levels, and the label rung with its victim swapped
uniformly or to the runner-up at two rates (`persistent_kv_admission.errorloc`).
Every arm's own decisions inside the evaluation window are compared with the
label rung's key `K*` (statistics m1-m4), recorded with the final victim of
every decision.

Nothing is fitted, no feature or target is added, and no arm is a proposed
policy: every constructed arm reads the trace's future on purpose. Before
anything is published the run checks that the four reference rungs reproduce
the 240 `all16` rows of the mechanism control and the three real rankers their
180 published on-policy rows exactly, that the label's statistics are the
identities m1 = m2 = 1, m3 = 0 in every replay and that every m4 victim of the
label has its next use exactly at the horizon (the binary target counts a reuse
at exactly H, the clipped label ties it with no reuse; the pre-smoke addendum
of the plan), that the statistics saw
every decision with its final victim, that the arm-independent counters are
arm-independent, and (per replay) the attribution identities; any failure
exits without writing a derived table.

Outputs (in --output-dir, and in --paper-dir for the full run): per-seed and
aggregated replay tables with the statistics, the six readings with per-seed
tables where a reading is seed-paired, two figures, a README, and the run
configuration. `--smoke` runs one trace, one cell, one seed and all 18 arms,
plus each arm once more without the statistics hook and the five identity arms
of the plan (admission by X, eviction by X for X in {learned, label}; swaps at
p = 0; noise at s = 0), checks those identities, integrity, runtime and memory,
writes only its own directory, and is not to be interpreted.
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
from matplotlib.lines import Line2D
import numpy as np

from persistent_kv_admission.attribution import AttributionCollector
from persistent_kv_admission.crossworkload import HYPERPARAMETERS
from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.errorloc import (
    ARMS,
    DECISION_TYPES,
    HYBRID_ARMS,
    HYBRIDS,
    IDENTITY_ARMS,
    LABEL_BINARY_ARM,
    LOCATIONS,
    METRIC_SETS,
    NOISE_LEVELS,
    ORDERS_THRESHOLD,
    PRIMARY_ARMS,
    REAL_RANKERS,
    REFERENCE_ARMS,
    STATISTICS,
    SWAP_KINDS,
    SWAP_PROBABILITIES,
    DecisionStatistics,
    RecordingOverride,
    SwapOverride,
    agrees,
    arm_setup,
    ceiling_ratio,
    contradicts,
    crossing_bracket,
    location_label,
    noise_arm,
    non_increasing,
    orders_utility,
    oriented,
    oriented_name,
    rank_correlation,
    swap_arm,
)
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.mechanism import sign_counts, sign_reading
from persistent_kv_admission.onpolicy import deserialize_ranker, sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

MECHANISM_SCRIPT = REPOSITORY / "scripts/run_mechanism_control.py"
PLAN_PATH = REPOSITORY / "docs/error-location-plan.md"


def _load_mechanism_control():
    """Import the mechanism-control runner by path, unchanged, for its grid,
    provenance, check and table helpers (it imports the Phase 0.97 runner the
    same way); `main` is behind the usual guard, so nothing runs."""
    spec = importlib.util.spec_from_file_location("run_mechanism_control", MECHANISM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rmc = _load_mechanism_control()
rdp = rmc.rdp

# --- the fixed grid (docs/error-location-plan.md) ----------------------------------------

TRACES = rmc.TRACES
CELLS = rmc.CELLS
SEEDS = rmc.SEEDS
# The published mechanism, and only it.
ELIGIBILITY = "all"
WIDTH = 16
MECHANISM = rmc.mechanism_label(ELIGIBILITY, WIDTH)
assert (ELIGIBILITY, WIDTH) == rmc.BASE_MECHANISM
LABEL_HORIZON_SECONDS = rmc.LABEL_HORIZON_SECONDS
SMOKE_TRACE = rmc.SMOKE_TRACE
SMOKE_CELL = rmc.SMOKE_CELL
SMOKE_SEEDS = rmc.SMOKE_SEEDS
# Replay variants: every grid arm runs with the statistics hook ("main"); the
# smoke also runs every arm without it ("nostats") to show the hook read-only.
VARIANTS = ("main", "nostats")
# Published reproduction targets, anchored at the repository.
MECHANISM_REFERENCE = REPOSITORY / "results/paper/mechanism_control_001/replay_seeds.csv"
ONPOLICY_REFERENCE = REPOSITORY / "results/paper/onpolicy_learning/onpolicy_seed_utility.csv"
# The on-policy models: pi0 is shared per trace x target, later iterations are
# per trace x cell x target x seed; both manifests name them and must agree.
MODEL_MANIFESTS = rmc.MODEL_MANIFESTS
MODEL_ROOT = REPOSITORY / "results/onpolicy_full_feb30eb_001/models"
MODEL_MARKER = "results/onpolicy_full_feb30eb_001/models/"
# Identifiers a published row must share with this run's replay, beyond the
# reproduced `avoided_prefill_tokens` itself.
REFERENCE_IDENTIFIERS = ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                         "l1_avoided_tokens")
# Reading 5: target -> (pi0 arm, pi3 arm).
RANKER_PAIRS = {"next_use": ("learned", "pi3_next_use"), "binary": ("pi0_binary", "pi3_binary")}
STAT_METRICS = (
    "m1", "m2", "m2_admission", "m2_resident", "m3", "m3_admission", "m3_resident",
    "m4", "m4_admission", "m4_resident", "stat_decisions", "stat_decisions_admission",
    "stat_decisions_resident", "overridden_decisions", "m4_victim_at_horizon",
)
REPLAY_METRICS = rmc.REPLAY_METRICS + STAT_METRICS
# The arms whose statistics must be the label identities (the label itself,
# and in the smoke the identity arms built to equal it).
LABEL_IDENTITY_ARMS = ("label",) + tuple(arm for arm, ref in IDENTITY_ARMS.items() if ref == "label")
SHARED: dict[str, object] = {}


def arm_family(arm: str) -> str:
    if arm in REFERENCE_ARMS:
        return "reference"
    if arm in REAL_RANKERS:
        return "real_ranker"
    if arm == LABEL_BINARY_ARM:
        return "label_binary"
    if arm in HYBRIDS:
        return "hybrid"
    if arm.startswith("noise_"):
        return "noise"
    if arm.startswith("swap_"):
        return "swap"
    raise ValueError(f"unknown arm {arm!r}")


def arm_parameter(arm: str):
    """The noise level s or the swap probability p of an arm, "" otherwise."""
    if arm.startswith("noise_"):
        return float(arm[len("noise_"):])
    for kind in SWAP_KINDS:
        if arm.startswith(f"swap_{kind}_"):
            return float(arm[len(f"swap_{kind}_"):])
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path, help="Mooncake trace files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new run directory; refused if it exists")
    parser.add_argument("--paper-dir", type=Path, default=None,
                        help="new publication directory (full run only); refused if it exists")
    parser.add_argument("--workers", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help="number of seeds; the pre-registered grid is 5 (smoke: seed 0)")
    parser.add_argument("--smoke", action="store_true",
                        help="conversation trace, 1%% x 4, seed 0, all 18 arms plus the identity "
                             "and hook-off replays; no paper dir; not to be interpreted")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="smoke only: run from an uncommitted execution tree")
    return parser.parse_args()


# --- provenance ---------------------------------------------------------------------------


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.update((Path(__file__).resolve(), MECHANISM_SCRIPT, rmc.PHASE097_SCRIPT))
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


# --- inputs ----------------------------------------------------------------------------------


def _model_path(manifest_path: str) -> Path:
    """A manifest's model path, re-anchored at this repository."""
    text = manifest_path.replace("\\", "/")
    position = text.find(MODEL_MARKER)
    if position < 0:
        raise SystemExit(f"model path {manifest_path} is not under {MODEL_MARKER}")
    return MODEL_ROOT / text[position + len(MODEL_MARKER):]


def load_real_rankers(names, cells, seeds) -> tuple[dict, dict]:
    """The published pi0 / pi3 models of the real-ranker arms, by their hash.

    pi0 is one model per trace x target; pi3 one per trace x cell x target x
    seed. Each model must be named with one (path, SHA-256) by both manifests.
    Returns `(name, fraction, multiplier, arm, seed) -> ranker` and provenance.
    """
    wanted = {(policy, target) for policy, target in REAL_RANKERS.values()}
    named: dict[tuple, set] = defaultdict(set)
    for manifest in MODEL_MANIFESTS:
        with manifest.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row["policy"], row["target"]) not in wanted:
                    continue
                if row["policy"] == "pi0":
                    key = (row["trace"], row["policy"], row["target"])
                else:
                    key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                           int(row["seed"]), row["policy"], row["target"])
                named[key].add((str(_model_path(row["model_path"]).relative_to(REPOSITORY)),
                                row["model_sha256"]))
    loaded: dict[tuple, object] = {}
    rankers, provenance = {}, {}
    for name in names:
        for arm, (policy, target) in REAL_RANKERS.items():
            for fraction, multiplier in cells:
                for seed in seeds:
                    key = ((name, policy, target) if policy == "pi0"
                           else (name, fraction, multiplier, seed, policy, target))
                    entries = named.get(key, set())
                    if len(entries) != 1:
                        raise SystemExit(f"{policy} {target} model of {key} is not named uniquely "
                                         f"by the manifests: {sorted(entries)}")
                    relative, digest = next(iter(entries))
                    if key not in loaded:
                        # raises on a hash mismatch
                        loaded[key] = deserialize_ranker(REPOSITORY / relative, digest)
                    rankers[(name, fraction, multiplier, arm, seed)] = loaded[key]
                    provenance[f"{name}/{rdp.cell_label(fraction, multiplier)}/{arm}/s{seed}"] = {
                        "path": relative, "sha256": digest}
    return rankers, provenance


def _cell_seed(row) -> tuple:
    return rmc._cell_seed(row)


def load_references() -> tuple[dict[tuple, dict], list[str]]:
    """`(arm, trace, fraction, multiplier, seed) -> published row` for the
    reference rungs (mechanism control, `all16`) and the real rankers
    (on-policy held-out utility), with the settings the rows must carry."""
    published: dict[tuple, dict] = {}
    problems: list[str] = []

    def put(key, row, source):
        if key in published:
            problems.append(f"duplicate published reference {key}")
            return
        entry = {"source": source, "avoided_prefill_tokens": int(row["avoided_prefill_tokens"])}
        for field in REFERENCE_IDENTIFIERS:
            entry[field] = int(row[field])
        published[key] = entry

    with MECHANISM_REFERENCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mechanism"] != MECHANISM or row["rung"] not in REFERENCE_ARMS:
                continue
            if row["eligibility"] != ELIGIBILITY or int(row["width"]) != WIDTH:
                problems.append(f"mechanism-control row {row['mechanism']} has "
                                f"{row['eligibility']}/{row['width']}")
            put((row["rung"],) + _cell_seed(row), row, "mechanism_control_001")
    by_policy = {value: arm for arm, value in REAL_RANKERS.items()}
    with ONPOLICY_REFERENCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = by_policy.get((row["policy"], row["target"]))
            if arm is None:
                continue
            for field, value in (("l1_policy", rdp.L1_POLICY), ("hit_model", rdp.HIT_MODEL),
                                 ("closure", rdp.CLOSURE), ("l2_eviction", "sampled"),
                                 ("l2_policy", "learned")):
                if row[field] != value:
                    problems.append(f"on-policy {arm} row: {field} {row[field]!r} != {value!r}")
            if int(row["l2_sample_width"]) != WIDTH or int(row["l2_seed"]) != int(row["seed"]):
                problems.append(f"on-policy {arm} row: width {row['l2_sample_width']}, "
                                f"l2_seed {row['l2_seed']} for seed {row['seed']}")
            put((arm,) + _cell_seed(row), row, "onpolicy_learning")
    return published, problems


# --- the replay worker -------------------------------------------------------------------------


def _counters_digest(result, collector) -> str:
    """SHA-256 of every counter the replay and the attribution keep (the arm
    label aside), so two replays can be compared counter for counter."""
    counters = {key: value for key, value in result.as_row().items() if key != "l2_arm"}
    counters.update(collector.loss_row())
    counters.update(collector.orphaning_row())
    encoded = json.dumps(counters, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _replay_worker(task):
    """One replay of one arm, with the attribution hooks and (variant "main")
    the statistics hook around the arm's own override."""
    name, fraction, multiplier, arm, seed, variant = task
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    horizon = SHARED["horizons"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    setup = arm_setup(arm, trace, horizon, seed, learned_ranker=SHARED["rankers"][name],
                      real_ranker=SHARED["real_rankers"].get((name, fraction, multiplier, arm, seed)))
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
        "width": WIDTH, "mechanism": MECHANISM, "arm": arm, "family": arm_family(arm),
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
        "counters_sha256": _counters_digest(result, collector),
        "swaps": setup.override.swaps if isinstance(setup.override, SwapOverride) else "",
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
    """Scheduling class, most expensive first: a hybrid with a history ranker
    scores twice; a history ranker computes features; the label family reads
    the future; LRU and the offline key are table lookups."""
    if arm in HYBRIDS and "learned" in HYBRIDS[arm]:
        return 0
    if arm == "learned" or arm in REAL_RANKERS:
        return 1
    if arm in ("lru", "offline"):
        return 3
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
                  for name in names for fraction, multiplier in cells for arm in IDENTITY_ARMS
                  for seed in seeds]
    tasks.sort(key=lambda task: (arm_cost(task[3]), -task[1], -task[2], task[0], task[3],
                                 task[4], task[5]))
    return tasks


# --- checks ----------------------------------------------------------------------------------


def grid_rows(rows) -> list[dict]:
    """The 18 grid arms with the statistics hook: the rows every reading uses."""
    return [row for row in rows if row["variant"] == "main" and row["arm"] in ARMS]


def check_reproduction(rows, published, expected_per_arm: int) -> dict[str, dict]:
    """Reference rungs against the mechanism control's `all16` rows, real
    rankers against their published on-policy rows: `avoided_prefill_tokens`
    exactly, and the identifiers of the cell-seed."""
    report = {}
    arms = REFERENCE_ARMS + tuple(REAL_RANKERS)
    for arm in arms:
        matched = missing = 0
        mismatches: list[str] = []
        for row in grid_rows(rows):
            if row["arm"] != arm:
                continue
            reference = published.get((arm,) + _cell_seed(row))
            if reference is None:
                missing += 1
                row["reference_avoided_prefill_tokens"] = ""
                row["reproduces_reference"] = ""
                continue
            differing = [field for field in ("avoided_prefill_tokens",) + REFERENCE_IDENTIFIERS
                         if int(row[field]) != reference[field]]
            row["reference_source"] = reference["source"]
            row["reference_avoided_prefill_tokens"] = reference["avoided_prefill_tokens"]
            row["reproduces_reference"] = not differing
            if differing:
                mismatches.append(f"{row['trace']}/{row['cell']}/s{row['seed']}: " + ", ".join(
                    f"{field} {row[field]} != {reference[field]}" for field in differing))
            else:
                matched += 1
        source = "mechanism_control_001/all16" if arm in REFERENCE_ARMS else "onpolicy_learning"
        report[arm] = {"reference": f"{source}/{arm}", "matched": matched, "missing": missing,
                       "mismatched": len(mismatches), "expected": expected_per_arm,
                       "mismatches": mismatches}
    return report


def check_statistics(rows) -> list[str]:
    """The statistics saw every decision with its final victim, an arm without
    an override never overrode, and the label's statistics are the identities
    m1 = m2 = 1, m3 = 0 with every m4 victim's next use exactly at the horizon
    (`m4_count == m4_victim_at_horizon`), overall and per decision type that
    occurred. m4 counts a next use <= H (the binary target), while the label
    clips a next use at exactly H to the value of no reuse, so the label can
    evict such a state on a recency tie; anywhere else an m4 victim of the
    label is a fault (pre-smoke addendum of the plan)."""
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
        if (row["arm"] not in HYBRIDS and not row["arm"].startswith("swap_")
                and int(row["overridden_decisions_seen"]) != 0):
            problems.append(f"{where}: {row['overridden_decisions_seen']} overridden decisions "
                            "for an arm without an override")
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
    """Smoke only: each identity arm equals its reference arm counter for
    counter and decision for decision, and every arm's counters are the same
    with and without the statistics hook."""
    index = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["seed"], row["arm"],
              row["variant"]): row for row in rows}
    table, problems = [], []

    def compare(kind, arm, reference, left, right, digests):
        if left is None or right is None:
            problems.append(f"{kind} {arm} vs {reference}: replay missing")
            return
        same = {name: left[name] == right[name] for name in digests}
        entry = {"trace": left["trace"], "cell": left["cell"], "seed": left["seed"],
                 "check": kind, "arm": arm, "against": reference,
                 "seconds": left["seconds"], "against_seconds": right["seconds"],
                 **{f"same_{name}": value for name, value in same.items()}}
        entry["holds"] = all(same.values())
        table.append(entry)
        if not entry["holds"]:
            problems.append(f"{kind} {arm} vs {reference} on {left['trace']}/{left['cell']}/"
                            f"s{left['seed']}: differs in "
                            + ", ".join(name for name, value in same.items() if not value))

    cell_seeds = sorted({key[:4] for key in index})
    for cell_seed in cell_seeds:
        for arm, reference in IDENTITY_ARMS.items():
            compare("identity", arm, reference, index.get(cell_seed + (arm, "main")),
                    index.get(cell_seed + (reference, "main")),
                    ("counters_sha256", "decision_sha256"))
        for arm in ARMS:
            compare("hook_read_only", arm, f"{arm} without statistics",
                    index.get(cell_seed + (arm, "main")), index.get(cell_seed + (arm, "nostats")),
                    ("counters_sha256",))
    return table, problems


# --- derivation -------------------------------------------------------------------------------

_stats = rmc._stats
_put_stats = rmc._put_stats
_points = rmc._points
_sign_string = rmc._sign_string
_is_nan = rmc._is_nan


def _cell_key(row) -> tuple:
    return (row["trace"], row["l1_fraction"], row["l2_multiplier"])


def _arm_order(arm: str) -> int:
    return ARMS.index(arm) if arm in ARMS else len(ARMS)


def aggregate_replays(rows) -> list[dict]:
    """Five-seed mean, std, 95% t half-width, min and max per trace x cell x arm."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_key(row) + (row["cell"], _arm_order(row["arm"]), row["arm"])].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2], "cell": key[3],
                 "mechanism": MECHANISM, "arm": key[5], "family": arm_family(key[5]),
                 "arm_parameter": arm_parameter(key[5]), "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        for metric in REPLAY_METRICS:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        out.append(entry)
    return out


def index_arms(rows) -> dict[tuple, dict[int, dict[str, dict]]]:
    """(trace, fraction, multiplier) -> seed -> arm -> row."""
    out: dict[tuple, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        out[_cell_key(row)][row["seed"]][row["arm"]] = row
    return out


def _difference_points(later: dict, earlier: dict) -> tuple[int, float]:
    """`U(later) - U(earlier)` of one seed: integer tokens and input-token points."""
    tokens = int(later["extra_avoided_tokens"]) - int(earlier["extra_avoided_tokens"])
    return tokens, _points(tokens, int(later["requested_tokens"]))


def _put_signs(entry: dict, prefix: str, values) -> None:
    positive, zero, negative = sign_counts(values)
    entry.update({f"{prefix}_n_pos": positive, f"{prefix}_n_zero": zero,
                  f"{prefix}_n_neg": negative, f"{prefix}_seed_signs": _sign_string(values),
                  f"{prefix}_reading": sign_reading(values)})


def _identity(name, fraction, multiplier) -> dict:
    return {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
            "cell": rdp.cell_label(fraction, multiplier)}


def _cell_agreement(table: list[dict], column: str, target: str) -> None:
    """Writes `target`: the value of `column` when every trace of the cell
    carries the same one (and both traces of the grid are present), else
    "traces_differ" / "single_trace"."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for entry in table:
        by_cell[(entry["l1_fraction"], entry["l2_multiplier"])].append(entry)
    for members in by_cell.values():
        values = {member[column] for member in members}
        if len(members) < len(TRACES):
            value = "single_trace"
        elif len(values) == 1:
            value = next(iter(values))
        else:
            value = "traces_differ"
        for member in members:
            member[target] = value


LOCATION_TERMS = ("G", "A", "E", "interaction")


def location_tables(rows) -> tuple[list[dict], list[dict]]:
    """Reading 1: G, A, E and the interaction A + E - G per seed and on the
    five-seed means, with the location label of the means."""
    seed_rows, table = [], []
    for (name, fraction, multiplier), by_seed in sorted(index_arms(rows).items()):
        terms: dict[str, list[float]] = {term: [] for term in LOCATION_TERMS}
        seed_labels = []
        for seed in sorted(by_seed):
            arms = by_seed[seed]
            g_tokens, g = _difference_points(arms["label"], arms["learned"])
            a_tokens, a = _difference_points(arms["adm_label"], arms["learned"])
            e_tokens, e = _difference_points(arms["evict_label"], arms["learned"])
            i_tokens = a_tokens + e_tokens - g_tokens
            interaction = _points(i_tokens, int(arms["label"]["requested_tokens"]))
            seed_rows.append({**_identity(name, fraction, multiplier), "seed": seed,
                              "G_tokens": g_tokens, "A_tokens": a_tokens, "E_tokens": e_tokens,
                              "interaction_tokens": i_tokens, "G_points": g, "A_points": a,
                              "E_points": e, "interaction_points": interaction,
                              "seed_location": location_label(g, a, e)})
            seed_labels.append(seed_rows[-1]["seed_location"])
            for term, value in zip(LOCATION_TERMS, (g, a, e, interaction)):
                terms[term].append(value)
        entry = {**_identity(name, fraction, multiplier), "seeds": len(terms["G"])}
        for term in LOCATION_TERMS:
            _put_stats(entry, f"{term}_points", terms[term], fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, term, terms[term])
        g, a, e = (entry[f"{term}_points_mean"] for term in ("G", "A", "E"))
        entry["A_share_of_G"] = a / g if g else math.nan
        entry["E_share_of_G"] = e / g if g else math.nan
        entry["location"] = location_label(g, a, e)
        entry["seed_locations"] = "|".join(seed_labels)
        table.append(entry)
    _cell_agreement(table, "location", "location_both_traces")
    return seed_rows, table


def dose_response_tables(rows) -> tuple[list[dict], list[dict]]:
    """Reading 2: U(noise_s) per level against U(learned), per seed and on the
    five-seed means; non-increasing in s and the crossing bracket."""
    seed_rows, table = [], []
    for (name, fraction, multiplier), by_seed in sorted(index_arms(rows).items()):
        per_level: dict[float, list[float]] = {level: [] for level in NOISE_LEVELS}
        references: dict[str, list[float]] = {arm: [] for arm in ("lru", "learned", "label")}
        seeds_non_increasing = 0
        for seed in sorted(by_seed):
            arms = by_seed[seed]
            values = [float(arms[noise_arm(level)]["extra_points"]) for level in NOISE_LEVELS]
            learned = float(arms["learned"]["extra_points"])
            monotone = non_increasing(values)
            seeds_non_increasing += monotone
            entry = {**_identity(name, fraction, multiplier), "seed": seed}
            for level, value in zip(NOISE_LEVELS, values):
                entry[f"U_{noise_arm(level)}_points"] = value
                per_level[level].append(value)
            for arm in references:
                entry[f"U_{arm}_points"] = float(arms[arm]["extra_points"])
                references[arm].append(float(arms[arm]["extra_points"]))
            entry["non_increasing"] = monotone
            entry["crossing"] = crossing_bracket(NOISE_LEVELS, values, learned)
            seed_rows.append(entry)
        entry = {**_identity(name, fraction, multiplier), "seeds": len(references["learned"])}
        means = []
        for level in NOISE_LEVELS:
            _put_stats(entry, f"U_{noise_arm(level)}_points", per_level[level],
                       fields=("mean", "ci95_half", "min", "max"))
            means.append(entry[f"U_{noise_arm(level)}_points_mean"])
        for arm, values in references.items():
            _put_stats(entry, f"U_{arm}_points", values, fields=("mean", "min", "max"))
        entry["non_increasing_means"] = non_increasing(means)
        entry["seeds_non_increasing"] = seeds_non_increasing
        entry["crossing"] = crossing_bracket(NOISE_LEVELS, means, entry["U_learned_points_mean"])
        table.append(entry)
    return seed_rows, table


def placement_tables(rows) -> tuple[list[dict], list[dict]]:
    """Reading 3: `U(swap_runnerup_p) - U(swap_uniform_p)` seed-paired per p,
    with both arms' realised m2."""
    seed_rows, table = [], []
    for (name, fraction, multiplier), by_seed in sorted(index_arms(rows).items()):
        for probability in SWAP_PROBABILITIES:
            runnerup, uniform = swap_arm("runnerup", probability), swap_arm("uniform", probability)
            differences, m2 = [], {runnerup: [], uniform: []}
            for seed in sorted(by_seed):
                arms = by_seed[seed]
                tokens, difference = _difference_points(arms[runnerup], arms[uniform])
                differences.append(difference)
                for arm in m2:
                    m2[arm].append(float(arms[arm]["m2"]))
                seed_rows.append({**_identity(name, fraction, multiplier), "p": probability,
                                  "seed": seed, "diff_tokens": tokens, "diff_points": difference,
                                  "m2_runnerup": float(arms[runnerup]["m2"]),
                                  "m2_uniform": float(arms[uniform]["m2"]),
                                  "swaps_runnerup": arms[runnerup]["swaps"],
                                  "swaps_uniform": arms[uniform]["swaps"]})
            entry = {**_identity(name, fraction, multiplier), "p": probability,
                     "seeds": len(differences)}
            _put_stats(entry, "diff_points", differences, fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, "diff", differences)
            entry["reading"] = entry.pop("diff_reading")
            entry["m2_runnerup_mean"] = float(np.mean(m2[runnerup]))
            entry["m2_uniform_mean"] = float(np.mean(m2[uniform]))
            table.append(entry)
    return seed_rows, table


def metric_order_table(summary) -> list[dict]:
    """Reading 4: per trace x cell, set and statistic, the Spearman correlation
    over arms of the oriented five-seed mean statistic with the five-seed mean U."""
    by_cell: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for entry in summary:
        by_cell[_cell_key(entry)][entry["arm"]] = entry
    out = []
    for (name, fraction, multiplier), arms in sorted(by_cell.items()):
        for set_name, members in METRIC_SETS.items():
            present = [arm for arm in members if arm in arms]
            utilities = [arms[arm]["extra_points_mean"] for arm in present]
            for statistic in STATISTICS:
                values = [oriented(statistic, arms[arm][f"{statistic}_mean"]) for arm in present]
                rho = rank_correlation(values, utilities)
                out.append({**_identity(name, fraction, multiplier), "set": set_name,
                            "statistic": statistic, "oriented": oriented_name(statistic),
                            "arms": len(present), "rho": rho, "orders_utility": orders_utility(rho)})
    return out


def ranker_pair_tables(rows) -> tuple[list[dict], list[dict]]:
    """Reading 5: per target, `U(pi3) - U(pi0)` seed-paired with its reading,
    and the seed-paired mean change of each oriented statistic, whether it
    agrees in sign with the mean utility change, and whether it contradicts a
    consistent utility change."""
    seed_rows, table = [], []
    for (name, fraction, multiplier), by_seed in sorted(index_arms(rows).items()):
        for target, (pi0, pi3) in RANKER_PAIRS.items():
            utility = []
            changes: dict[str, list[float]] = {statistic: [] for statistic in STATISTICS}
            for seed in sorted(by_seed):
                arms = by_seed[seed]
                tokens, difference = _difference_points(arms[pi3], arms[pi0])
                utility.append(difference)
                entry = {**_identity(name, fraction, multiplier), "target": target, "pi0": pi0,
                         "pi3": pi3, "seed": seed, "dU_tokens": tokens, "dU_points": difference}
                for statistic in STATISTICS:
                    change = (oriented(statistic, float(arms[pi3][statistic]))
                              - oriented(statistic, float(arms[pi0][statistic])))
                    changes[statistic].append(change)
                    entry[f"d_{statistic}"] = change
                seed_rows.append(entry)
            entry = {**_identity(name, fraction, multiplier), "target": target, "pi0": pi0,
                     "pi3": pi3, "seeds": len(utility)}
            _put_stats(entry, "dU_points", utility, fields=("mean", "ci95_half", "min", "max"))
            _put_signs(entry, "dU", utility)
            reading = entry["dU_reading"]
            for statistic in STATISTICS:
                mean_change = float(np.mean(changes[statistic]))
                entry[f"d_{statistic}_mean"] = mean_change
                _put_signs(entry, f"d_{statistic}", changes[statistic])
                entry[f"{statistic}_agrees"] = agrees(mean_change, entry["dU_points_mean"])
                entry[f"{statistic}_contradicts"] = contradicts(reading, mean_change)
            table.append(entry)
    return seed_rows, table


def binary_ceiling_tables(rows) -> tuple[list[dict], list[dict]]:
    """Reading 6: `U(label) - U(label_binary)` with seed signs, and
    `(U(pi0_binary) - U(lru)) / (U(label_binary) - U(lru))` per seed and on the
    five-seed means."""
    seed_rows, table = [], []
    for (name, fraction, multiplier), by_seed in sorted(index_arms(rows).items()):
        differences, levels = [], {arm: [] for arm in ("lru", "pi0_binary", LABEL_BINARY_ARM,
                                                       "label")}
        for seed in sorted(by_seed):
            arms = by_seed[seed]
            tokens, difference = _difference_points(arms["label"], arms[LABEL_BINARY_ARM])
            differences.append(difference)
            values = {arm: float(arms[arm]["extra_points"]) for arm in levels}
            for arm, value in values.items():
                levels[arm].append(value)
            seed_rows.append({**_identity(name, fraction, multiplier), "seed": seed,
                              "label_minus_label_binary_tokens": tokens,
                              "label_minus_label_binary_points": difference,
                              **{f"U_{arm}_points": value for arm, value in values.items()},
                              "ceiling_ratio": ceiling_ratio(values["pi0_binary"],
                                                             values[LABEL_BINARY_ARM],
                                                             values["lru"])})
        entry = {**_identity(name, fraction, multiplier), "seeds": len(differences)}
        _put_stats(entry, "label_minus_label_binary_points", differences,
                   fields=("mean", "ci95_half", "min", "max"))
        _put_signs(entry, "label_minus_label_binary", differences)
        for arm, values in levels.items():
            entry[f"U_{arm}_points_mean"] = float(np.mean(values))
        entry["ceiling_ratio"] = ceiling_ratio(entry["U_pi0_binary_points_mean"],
                                               entry[f"U_{LABEL_BINARY_ARM}_points_mean"],
                                               entry["U_lru_points_mean"])
        table.append(entry)
    return seed_rows, table


# --- figures ------------------------------------------------------------------------------------

# The validated reference palette's first three slots (they pass the all-pairs
# CVD and normal-vision floors; aqua sits below 3:1 on white, so every series
# also differs in marker or line style and the CSVs are the table view).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = rmc.INK, rmc.MUTED, rmc.GRID
_style_axis = rmc._style_axis


def _panel_title(name: str, fraction: float, multiplier: float) -> str:
    return f"{name.replace('_trace', '')} · L1 {100 * fraction:g}% · L2 x{multiplier:g}"


def dose_response_figure(directory: Path, summary: list[dict]) -> None:
    """U against the noise level per trace x cell (mean, seed min-max band),
    the lru / learned / label levels as horizontal lines, and both swap
    families at p = 0.25 and 0.5 to the right of the noise axis."""
    traces, cells = rmc._panel_grid(summary)
    index = {(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"], entry["arm"]): entry
             for entry in summary}
    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(7.0, 3.0 * len(cells) + 0.6), 2.9 * len(traces) + 1.2))
    noise_x = np.arange(len(NOISE_LEVELS), dtype=float)
    swap_x = {probability: len(NOISE_LEVELS) + 0.6 + index_p
              for index_p, probability in enumerate(SWAP_PROBABILITIES)}
    swap_style = {"uniform": ("v", "none", -0.13, "swap to a uniform other candidate"),
                  "runnerup": ("^", AQUA, 0.13, "swap to the K* runner-up")}
    for row_index, name in enumerate(traces):
        for column, (fraction, multiplier) in enumerate(cells):
            axis = axes[row_index][column]

            def get(arm):
                return index.get((name, fraction, multiplier, arm))

            noise = [get(noise_arm(level)) for level in NOISE_LEVELS]
            if all(entry is not None for entry in noise):
                means = [entry["extra_points_mean"] for entry in noise]
                lows = [entry["extra_points_min"] for entry in noise]
                highs = [entry["extra_points_max"] for entry in noise]
                axis.fill_between(noise_x, lows, highs, color=BLUE, alpha=0.14, linewidth=0)
                axis.plot(noise_x, means, color=BLUE, marker="o", markersize=4.5, linewidth=2.0,
                          label="label + s·z (noise arms)")
            for arm, color, style, label in (("label", INK, ":", "label"),
                                             ("learned", ORANGE, "--", "learned (frozen pi0)"),
                                             ("lru", MUTED, "-.", "sampled LRU")):
                entry = get(arm)
                if entry is not None:
                    axis.axhline(entry["extra_points_mean"], color=color, linestyle=style,
                                 linewidth=1.3, label=label)
            for kind, (marker, face, offset, label) in swap_style.items():
                for probability in SWAP_PROBABILITIES:
                    entry = get(swap_arm(kind, probability))
                    if entry is None:
                        continue
                    x = swap_x[probability] + offset
                    mean = entry["extra_points_mean"]
                    axis.errorbar([x], [mean], yerr=[[mean - entry["extra_points_min"]],
                                                     [entry["extra_points_max"] - mean]],
                                  fmt=marker, color=AQUA, markerfacecolor=face, markersize=7,
                                  elinewidth=1.0, capsize=2,
                                  label=label if probability == SWAP_PROBABILITIES[0] else None)
            # Separates the noise axis from the swap arms, which have no s.
            axis.axvline(0.5 * (noise_x[-1] + swap_x[SWAP_PROBABILITIES[0]]), color=GRID,
                         linewidth=1.0)
            ticks = list(noise_x) + [swap_x[p] for p in SWAP_PROBABILITIES]
            labels = [f"s={level:g}" for level in NOISE_LEVELS] + [
                f"p={p:g}" for p in SWAP_PROBABILITIES]
            axis.set_xticks(ticks, labels)
            axis.set_title(_panel_title(name, fraction, multiplier), fontsize=8, color=INK)
            if column == 0:
                axis.set_ylabel("extra avoided, input-token points", fontsize=7.5, color=INK)
            _style_axis(axis)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    figure.suptitle("Dose-response of Gaussian label noise, and victim swaps on the label rung "
                    "(mean over seeds; band / whisker = seed min-max)", fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.10 if len(traces) > 1 else 0.17, 1.0, 0.95))
    figure.savefig(directory / "dose_response.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


# Composite encoding of the arm families: three colours, one marker per family,
# hollow markers for the override arms (secondary set only).
FAMILY_STYLE = {
    "reference": (BLUE, "o", "full", "reference rungs (lru, learned, label, offline)"),
    "real_ranker": (BLUE, "s", "full", "real rankers (pi3_next_use, pi0/pi3_binary)"),
    "label_binary": (BLUE, "D", "full", "label_binary"),
    "noise": (ORANGE, "o", "full", "noise arms"),
    "hybrid": (AQUA, "^", "none", "hybrids (override; secondary set)"),
    "swap": (AQUA, "v", "none", "swap arms (override; secondary set)"),
}


def statistic_figure(directory: Path, summary: list[dict], metric_order: list[dict]) -> None:
    """Per-cell panels: one row per trace x oriented statistic, one column per
    cell, the 18 arms as points (five-seed means), x the oriented statistic and
    y the utility; the primary-set Spearman rho of the panel in its title.
    Per-cell rather than pooled: reading 4 ranks arms inside a cell and U's
    scale differs several-fold between cells, so a pooled panel would mostly
    show the between-cell spread."""
    traces, cells = rmc._panel_grid(summary)
    rho = {(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"], entry["statistic"]):
           entry["rho"] for entry in metric_order if entry["set"] == "primary"}
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for entry in summary:
        by_cell[_cell_key(entry)].append(entry)
    rows_of = [(name, statistic) for name in traces for statistic in STATISTICS]
    figure, axes = plt.subplots(len(rows_of), len(cells), squeeze=False,
                                figsize=(max(7.0, 2.6 * len(cells) + 0.8), 2.1 * len(rows_of) + 1.2))
    for row_index, (name, statistic) in enumerate(rows_of):
        for column, (fraction, multiplier) in enumerate(cells):
            axis = axes[row_index][column]
            for family, (color, marker, fill, label) in FAMILY_STYLE.items():
                members = [entry for entry in by_cell.get((name, fraction, multiplier), [])
                           if entry["family"] == family]
                if not members:
                    continue
                xs = [oriented(statistic, entry[f"{statistic}_mean"]) for entry in members]
                ys = [entry["extra_points_mean"] for entry in members]
                axis.scatter(xs, ys, s=26, marker=marker, linewidths=1.2,
                             facecolors=color if fill == "full" else "none", edgecolors=color,
                             label=label)
            value = rho.get((name, fraction, multiplier, statistic), math.nan)
            axis.set_title(f"{name.replace('_trace', '')} · L1 {100 * fraction:g}% · "
                           f"x{multiplier:g} · rho(primary) {value:.2f}", fontsize=7, color=INK)
            axis.set_xlabel(oriented_name(statistic), fontsize=7, color=INK)
            if column == 0:
                axis.set_ylabel("U, points", fontsize=7, color=INK)
            _style_axis(axis)
            axis.grid(True, axis="x", color=GRID, linewidth=0.6)
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=6, color=color,
                      markerfacecolor=color if fill == "full" else "none", label=label)
               for color, marker, fill, label in FAMILY_STYLE.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    figure.suptitle("Decision statistics against utility, per trace x cell (five-seed means; "
                    "statistics oriented so that larger is better)", fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.04 if len(rows_of) > 4 else 0.12, 1.0, 0.97))
    figure.savefig(directory / "statistic_utility.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


README_TEXT = """# Error-location control: which decisions and which errors separate a ranker from its label

Pre-registration: `docs/error-location-plan.md`. Mechanism: the published sampled one only (`all16`: the arrival plus up to 16 uniformly sampled residents; any candidate may leave). Arms (18): the reference rungs `lru`, `learned` (frozen pi0 `next_use`), `label` (exact `next_use` training target), `offline` (heap comparator's key on the sampled path); the real rankers `pi3_next_use`, `pi0_binary`, `pi3_binary` (published on-policy models, loaded by SHA-256); `label_binary` (exact `binary` target); the hybrids `adm_label` (admission by label, eviction by learned) and `evict_label` (admission by learned, eviction by label); `noise_s` (label + s·z, s in 0.5, 1, 2, 4); `swap_uniform_p` and `swap_runnerup_p` (label rung, victim swapped with probability p in 0.25, 0.5). U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. Statistics are of each arm's own decisions in the evaluation window against `K* = (label, last_group)`: m1 pairwise concordance (pooled over the pairs with different K*; an arm-key tie counts one half), m2 victim agreement with the first minimum of K*, m3 mean `label(victim) - min label`, m4 share of decisions whose victim is requested within the horizon while some candidate is not; m2-m4 also for admission decisions (the arrival is a candidate) and resident-only decisions. They are not the registered on-policy ranking statistic. Every constructed arm reads the future on purpose and none is a policy. Seed intervals describe sampling-seed variability only.

- `replay_seeds.csv`: one row per replay (trace x cell x arm x seed) with the replay counters, the Phase 0.98b attribution columns, the statistics as sums, counts and ratios (`m*`, `*_admission`, `*_resident`; `m4_count` and `m4_victim_at_horizon` count m4 victims and those whose next use is exactly at the horizon, which the binary target counts as reused and the clipped label ties with no reuse; for the label every m4 victim must be one of these, with m1 = m2 = 1 and m3 = 0, checked before publication), the decision-stream and counter digests, the realised swap count, the reproduction mark of the reference rungs and real rankers, timing and worker memory.
- `replay.csv`: five-seed mean, std, 95% t half-width, min and max of the replay metrics and the statistics per trace x cell x arm.
- `location.csv` / `location_seeds.csv` (reading 1): `G = U(label) - U(learned)`, `A = U(adm_label) - U(learned)`, `E = U(evict_label) - U(learned)` and the interaction `A + E - G`, per seed and on the five-seed means with seed signs, and the location label of the means (admission_located / eviction_located / both / neither, each term reaching half of G or not); `location_both_traces` is the label when both traces carry it.
- `dose_response.csv` / `dose_response_seeds.csv` (reading 2): U of the four noise arms with U(lru), U(learned), U(label); whether the means are non-increasing in s and in how many seeds; between which adjacent levels the mean crosses U(learned) (a level at or above it is on the upper side), or above_all / below_all.
- `placement.csv` / `placement_seeds.csv` (reading 3): `U(swap_runnerup_p) - U(swap_uniform_p)` seed-paired per p, its seed signs and consistent-gain / consistent-loss / mixed reading, and each arm's realised m2.
- `metric_order.csv` (reading 4): per trace x cell, set (primary: the twelve single-key arms; secondary: all 18) and statistic, the Spearman correlation over arms of the oriented five-seed mean statistic (m1, m2, -m3, -m4) with the five-seed mean U (average ranks for ties; nan when either side is constant), and whether it is at least 0.9.
- `ranker_pairs.csv` / `ranker_pairs_seeds.csv` (reading 5): for `next_use` (learned -> pi3_next_use) and `binary` (pi0_binary -> pi3_binary), `U(pi3) - U(pi0)` seed-paired with its reading, and the seed-paired change of each oriented statistic, whether its mean change has the sign of the mean utility change (`*_agrees`), and whether it moves against a consistent utility change (`*_contradicts`).
- `binary_ceiling.csv` / `binary_ceiling_seeds.csv` (reading 6, descriptive): `U(label) - U(label_binary)` with seed signs, and `(U(pi0_binary) - U(lru)) / (U(label_binary) - U(lru))` per seed and on the five-seed means.
- `dose_response.png`: U against the noise level per trace x cell (mean, seed min-max band), the lru / learned / label levels as horizontal lines, both swap families at p = 0.25 and 0.5 (whisker = seed min-max).
- `statistic_utility.png`: per trace x statistic x cell, the 18 arms as points (oriented statistic against U, five-seed means), with the primary-set Spearman rho of the panel.
- `run_config.json`: plan and code commits, source manifest, trace, model and reference hashes, grid, check counts, timing and memory.
"""

SMOKE_README = """# SMOKE RUN — NOT TO BE INTERPRETED

This directory is the error-location smoke (conversation trace, L1 1% x L2 4, seed 0). It checks integrity, the identities of the plan (admission by X, eviction by X equals rung X for X in learned and label; swaps at p = 0 and noise at s = 0 equal the label), that the statistics hook is read-only (every arm replayed with and without it), runtime and memory. It is excluded from every confirmatory table; one cell and one seed cannot carry a reading. `smoke_identities.csv` lists each identity and hook check; `replay_seeds.csv` holds the identity and hook-off replays too (`variant`, `arm`).

"""


# --- main ----------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


def main() -> None:
    args = parse_args()
    clock = time.time()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.allow_dirty and not args.smoke:
        raise SystemExit("--allow-dirty is for --smoke only; the full run needs a committed tree")
    if args.smoke and args.paper_dir is not None:
        raise SystemExit("--smoke never writes a paper directory; drop --paper-dir")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    if args.paper_dir is not None and args.paper_dir.exists():
        raise SystemExit(f"paper directory {args.paper_dir} exists; choose a new one")
    seeds = SMOKE_SEEDS if args.smoke else tuple(range(args.seeds))
    if not args.smoke and seeds != SEEDS:
        raise SystemExit(f"the pre-registered grid is seeds {list(SEEDS)}; got {list(seeds)}")

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
    real_rankers, real_model_files = load_real_rankers(names, cells, seeds)
    published, reference_problems = load_references()
    for name in names:
        print(f"  pi0 next_use {name}: {model_files[name]['path']} "
              f"sha256={model_files[name]['sha256'][:12]}", flush=True)
    print(f"  real-ranker models: {len(real_model_files)} (arm x cell x seed) entries, "
          f"{len({id(ranker) for ranker in real_rankers.values()})} distinct models", flush=True)
    if reference_problems:
        for line in reference_problems[:20]:
            print(f"    REFERENCE {line}", flush=True)
        raise SystemExit("published references are not what the plan names; nothing run")

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                  rankers=rankers, real_rankers=real_rankers)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds, args.smoke)
    workers = min(args.workers, len(tasks))
    print(f"{len(tasks)} replays on {workers} workers "
          f"({len(names)} traces x {len(cells)} cells x {len(ARMS)} arms x {len(seeds)} seeds"
          + (f", plus {len(tasks) - len(names) * len(cells) * len(ARMS) * len(seeds)} smoke "
             "identity and hook-off replays" if args.smoke else "") + ")", flush=True)
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
        print(f"    {arm:24s} {variant:8s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"arm": arm, "variant": variant, "n": len(values),
                            "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    expected_per_arm = len(names) * len(cells) * len(seeds)
    reproduction = check_reproduction(rows, published, expected_per_arm)
    for arm, report in reproduction.items():
        print(f"  reproduction ({arm} vs published {report['reference']}): "
              f"{report['matched']} matched, {report['missing']} missing, "
              f"{report['mismatched']} mismatched (expected {report['expected']})", flush=True)
        for line in report["mismatches"]:
            print(f"    MISMATCH {line}", flush=True)
        if report["mismatched"] or report["missing"] or report["matched"] != report["expected"]:
            failures.append(f"reproduction of {report['reference']} failed")
    statistics_problems = check_statistics(rows)
    label_rows = sum(1 for row in rows if row["variant"] == "main"
                     and row["arm"] in LABEL_IDENTITY_ARMS)
    print(f"  statistics: every decision seen with its final victim, label identities "
          f"(m1 = m2 = 1, m3 = 0, m4 victims only at the horizon) in "
          f"{label_rows} label replays: {'hold' if not statistics_problems else 'BROKEN'}",
          flush=True)
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
    location_seeds, location = location_tables(main_rows)
    dose_seeds, dose = dose_response_tables(main_rows)
    placement_seeds, placement = placement_tables(main_rows)
    metric_order = metric_order_table(replay_summary)
    pair_seeds, pairs = ranker_pair_tables(main_rows)
    ceiling_seeds, ceiling = binary_ceiling_tables(main_rows)

    total_cells = len(names) * len(cells)
    print("  reading 1, location (five-seed means, points):", flush=True)
    for entry in location:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} G={entry['G_points_mean']:7.3f} "
              f"A={entry['A_points_mean']:7.3f} E={entry['E_points_mean']:7.3f} "
              f"A+E-G={entry['interaction_points_mean']:7.3f} -> {entry['location']} "
              f"(both traces: {entry['location_both_traces']})", flush=True)
    print("  reading 2, dose-response:", flush=True)
    for entry in dose:
        levels = " ".join(f"{entry[f'U_{noise_arm(level)}_points_mean']:7.3f}"
                          for level in NOISE_LEVELS)
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} U(noise)={levels} "
              f"U(learned)={entry['U_learned_points_mean']:7.3f} non-increasing="
              f"{entry['non_increasing_means']} ({entry['seeds_non_increasing']}/{entry['seeds']} "
              f"seeds) crossing={entry['crossing']}", flush=True)
    print("  reading 3, placement (runner-up minus uniform swap):", flush=True)
    for probability in SWAP_PROBABILITIES:
        readings = [entry["reading"] for entry in placement if entry["p"] == probability]
        print(f"    p={probability:g}: " + ", ".join(
            f"{reading} {readings.count(reading)}/{total_cells}"
            for reading in ("consistent_gain", "consistent_loss", "mixed")), flush=True)
    print(f"  reading 4, statistic orders utility (rho >= {ORDERS_THRESHOLD:g}):", flush=True)
    for set_name in METRIC_SETS:
        orders = {statistic: sum(1 for e in metric_order if e["set"] == set_name
                                 and e["statistic"] == statistic and e["orders_utility"])
                  for statistic in STATISTICS}
        print(f"    {set_name:9s} " + ", ".join(
            f"{oriented_name(statistic)} {orders[statistic]}/{total_cells}"
            for statistic in STATISTICS), flush=True)
    print("  reading 5, ranker pairs (statistic agrees with the utility change):", flush=True)
    for target in RANKER_PAIRS:
        members = [entry for entry in pairs if entry["target"] == target]
        print(f"    {target:9s} " + ", ".join(
            f"{oriented_name(statistic)} "
            f"{sum(1 for e in members if e[f'{statistic}_agrees'])}/{len(members)}"
            for statistic in STATISTICS), flush=True)
        for entry in members:
            for statistic in STATISTICS:
                if entry[f"{statistic}_contradicts"]:
                    print(f"      CONTRADICTS {entry['trace']}/{entry['cell']}: U "
                          f"{entry['dU_reading']} ({entry['dU_points_mean']:.3f}) while "
                          f"{oriented_name(statistic)} moves {entry[f'd_{statistic}_mean']:+.4f}",
                          flush=True)
    print("  reading 6, binary ceiling:", flush=True)
    for entry in ceiling:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} U(label)-U(label_binary)="
              f"{entry['label_minus_label_binary_points_mean']:7.3f} "
              f"({entry['label_minus_label_binary_seed_signs']}) ratio={entry['ceiling_ratio']:.3f}",
              flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = rmc._git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "error_location",
        "smoke": bool(args.smoke),
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "git_dirty_at_start": bool(status),
        "execution_sources_differing_from_head": differing,
        "allow_dirty": bool(args.allow_dirty),
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "models": {"pi0_next_use": model_files, "real_rankers": real_model_files},
        "model_manifests": {str(path.relative_to(REPOSITORY)): sha256_path(path)
                            for path in MODEL_MANIFESTS},
        "references": {str(path.relative_to(REPOSITORY)): sha256_path(path)
                       for path in (MECHANISM_REFERENCE, ONPOLICY_REFERENCE)},
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "mechanism": MECHANISM, "eligibility": ELIGIBILITY, "width": WIDTH,
        "arms": list(ARMS), "primary_arms": list(PRIMARY_ARMS),
        "hybrids": {arm: {"admission": HYBRIDS[arm][0], "eviction": HYBRIDS[arm][1]}
                    for arm in HYBRID_ARMS},
        "noise_levels": list(NOISE_LEVELS), "swap_kinds": list(SWAP_KINDS),
        "swap_probabilities": list(SWAP_PROBABILITIES),
        "identity_arms": dict(IDENTITY_ARMS) if args.smoke else {},
        "statistics": list(STATISTICS), "orders_threshold": ORDERS_THRESHOLD,
        "location_labels": list(LOCATIONS),
        "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {arm: {k: v for k, v in report.items() if k != "mismatches"}
                             for arm, report in reproduction.items()},
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
                  "future on purpose and none is a policy. The reference rungs reproduce the "
                  "mechanism control's all16 rows and the real rankers the published on-policy "
                  "rows. Seed intervals describe sampling-seed variability only.",
    }
    destinations = [args.output_dir]
    if args.paper_dir is not None and not args.smoke:
        destinations.append(args.paper_dir)
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows if args.smoke else main_rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "location_seeds.csv", location_seeds)
        _write(directory / "location.csv", location)
        _write(directory / "dose_response_seeds.csv", dose_seeds)
        _write(directory / "dose_response.csv", dose)
        _write(directory / "placement_seeds.csv", placement_seeds)
        _write(directory / "placement.csv", placement)
        _write(directory / "metric_order.csv", metric_order)
        _write(directory / "ranker_pairs_seeds.csv", pair_seeds)
        _write(directory / "ranker_pairs.csv", pairs)
        _write(directory / "binary_ceiling_seeds.csv", ceiling_seeds)
        _write(directory / "binary_ceiling.csv", ceiling)
        if args.smoke:
            _write(directory / "smoke_identities.csv", identity_table)
        dose_response_figure(directory, replay_summary)
        statistic_figure(directory, replay_summary, metric_order)
        (directory / "README.md").write_text((SMOKE_README if args.smoke else "") + README_TEXT,
                                             encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {', '.join(str(d) for d in destinations)}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
