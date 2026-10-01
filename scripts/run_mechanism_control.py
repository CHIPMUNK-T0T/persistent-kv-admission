#!/usr/bin/env python3
"""Mechanism control: a fixed score ladder under four L2 eviction mechanisms.

The pre-registration is `docs/mechanism-control-plan.md`; nothing here may
change it. Four rungs of score — the sampled LRU key, the frozen `next_use`
pi0 ranker of each trace (the Phase 0.97 A_none fit, loaded by its published
SHA-256), the exact training label at the decision instant, and the heap
offline comparator's own key — are each replayed under four mechanisms
(eligibility `all` | `leaf` x sample width 16 | 64) on the complete Phase 0.97
grid, five seeds, with the Phase 0.98b attribution hooks attached read-only.
The published headroom `T_m = H_off - U_m(lru)` then splits exactly into the
candidate-search, objective and signal gaps and the achieved part
(`persistent_kv_admission.mechanism`).

Nothing is fitted, no feature is added and no policy is promoted. The heap
references `H_off` and `H_lru` are not rerun: they are read from
`results/paper/decision_population_replay_seeds.csv`, and only after the
trace, cell, capacities, requested tokens and L1 tokens of every cell-seed
match this run's. Before anything is published the run checks that the base
mechanism `(all, 16)` reproduces the published sampled LRU and A_none rows
exactly, that no `leaf` replay leaves a block present but unusable, and that
the arm-independent counters are arm-independent; any failure exits without
writing a derived table.

Outputs (in --output-dir, and in --paper-dir for the full run): per-seed and
aggregated replay tables, the decomposition per seed and aggregated, the
paired mechanism effects, the ladder-order table and its inversions, the
capacity pattern, two figures, a README, and the run configuration.
`--smoke` runs one trace, one cell, one seed and all sixteen arms, checks
integrity, runtime and memory only, and never writes a paper directory.
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
from persistent_kv_admission.decisionpop import RawFeatureScorer, horizon_for
from persistent_kv_admission.gap import summarize, working_set_bytes
from persistent_kv_admission.mechanism import (
    DOMINANT_LABELS,
    GAPS,
    LADDER_PAIRS,
    RUNGS,
    ExactLabelScorer,
    decompose,
    dominant_label,
    ladder_holds,
    pair_name,
    sign_counts,
    sign_reading,
)
from persistent_kv_admission.onpolicy import deserialize_ranker, sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import L2_ELIGIBILITIES, run_two_tier

PHASE097_SCRIPT = REPOSITORY / "scripts/run_decision_population.py"
PLAN_PATH = REPOSITORY / "docs/mechanism-control-plan.md"
# The published per-seed Phase 0.97 replay table: the reproduction target of the
# base mechanism and the source of the heap references. Anchored at the
# repository, never at an output directory.
REFERENCE_SEEDS = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"
MODEL_DIR = REPOSITORY / "results/onpolicy_full_feb30eb_001/models/pi0"
# Both published manifests name the pi0 hash; they must agree with each other.
MODEL_MANIFESTS = (
    REPOSITORY / "results/onpolicy_full_feb30eb_001/canonical_model_manifest.csv",
    REPOSITORY / "results/paper/onpolicy_learning/onpolicy_canonical_models.csv",
)


def _load_phase097():
    """Import the Phase 0.97 runner by path, unchanged, for its capacity rule
    and its constants; `main` is behind the usual guard, so nothing runs."""
    spec = importlib.util.spec_from_file_location("run_decision_population", PHASE097_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rdp = _load_phase097()

# --- the fixed grid (docs/mechanism-control-plan.md) -----------------------------

TRACES = ("conversation_trace", "toolagent_trace")
CELLS = ((0.0025, 1.0), (0.0025, 4.0), (0.01, 1.0), (0.01, 4.0), (0.02, 1.0), (0.02, 4.0))
ELIGIBILITIES = ("all", "leaf")
WIDTHS = (16, 64)
BASE_MECHANISM = ("all", 16)
# Table and figure order: the published mechanism first, then the eligibility
# change at the published width, then the two wide mechanisms.
MECHANISMS = (("all", 16), ("leaf", 16), ("all", 64), ("leaf", 64))
assert set(ELIGIBILITIES) == set(L2_ELIGIBILITIES)
assert {mechanism for mechanism in MECHANISMS} == {(e, k) for e in ELIGIBILITIES for k in WIDTHS}
SEEDS = (0, 1, 2, 3, 4)
LEARNED_TARGET = "next_use"
# The plan writes the label as -log1p(min(next_use_delta_s, 600)); the run
# refuses a trace whose fitted horizon is anything else.
LABEL_HORIZON_SECONDS = 600.0
SMOKE_TRACE = "conversation_trace"
SMOKE_CELL = (0.01, 4.0)
SMOKE_SEEDS = (0,)
# Published arms the base mechanism must reproduce, by rung: (kind, arm, target).
REPRODUCED_ARMS = {"lru": ("sampled", "lru_s", ""), "learned": ("learned", "A_none", "next_use")}
# Published heap references, by name: the published arm.
HEAP_REFERENCES = {"H_off": "offline_next_use", "H_lru": "lru"}
# Mechanism contrasts of reading 2, later minus earlier, seed-paired.
CONTRASTS = (("leaf16", "all16"), ("all64", "all16"), ("leaf64", "leaf16"), ("leaf64", "all16"))
# Per trace x cell x seed these must not depend on the arm: L1 never consults L2,
# and a block L2 never held is compulsory whatever L2 decided.
INVARIANT_COLUMNS = ("l1_avoided_tokens", "requested_tokens", "absent_compulsory_tokens")
# Token counters also reported in input-token points (100 x share of input).
POINT_COLUMNS = (
    ("extra_avoided_tokens", "extra_points"),
    ("l2_present_unusable_tokens", "present_unusable_points"),
    ("absent_rejected_tokens", "absent_rejected_points"),
    ("absent_evicted_tokens", "absent_evicted_points"),
    ("absent_compulsory_tokens", "absent_compulsory_points"),
    ("unusable_after_rejected_tokens", "unusable_after_rejected_points"),
    ("unusable_after_evicted_tokens", "unusable_after_evicted_points"),
    ("perblock_decision_loss_tokens", "perblock_decision_loss_points"),
)
REPLAY_METRICS = (
    "extra_points", "avoided_prefill_tokens", "extra_avoided_tokens",
    "present_unusable_points", "l2_present_unusable_tokens", "l2_present_unusable_blocks",
    "absent_rejected_points", "absent_evicted_points", "absent_compulsory_points",
    "unusable_after_rejected_points", "unusable_after_evicted_points",
    "perblock_decision_loss_points",
    "absent_rejected_share_of_decision_absent", "absent_evicted_share_of_decision_absent",
    "l2_admissions", "l2_rejections", "l2_evictions", "l2_decisions",
    "orphaned_blocks_window", "evictions_with_orphans_window", "seconds",
)
SHARED: dict[str, object] = {}


def mechanism_label(eligibility: str, width: int) -> str:
    return f"{eligibility}{width}"


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
                        help="conversation trace, 1%% x 4, seed 0, all 16 arms; no paper dir")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="smoke only: run from an uncommitted execution tree")
    return parser.parse_args()


# --- provenance -----------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, check=True,
                          capture_output=True, text=True).stdout.strip()


def execution_sources() -> list[Path]:
    """Every file the replays execute from this repository."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.update((Path(__file__).resolve(), PHASE097_SCRIPT))
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


def _peak_rss_mib(who: int) -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(who).ru_maxrss) / 1024.0


def _proc_mib(path: str, field: str) -> float:
    """One kB field of a /proc file in MiB, nan where it is not available."""
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return math.nan


# --- inputs -----------------------------------------------------------------------------


def load_models(names) -> tuple[dict, dict]:
    """The frozen pi0 `next_use` ranker of each trace, by its published hash."""
    expected: dict[str, set[str]] = defaultdict(set)
    for manifest in MODEL_MANIFESTS:
        with manifest.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["policy"] == "pi0" and row["target"] == LEARNED_TARGET:
                    expected[row["trace"]].add(row["model_sha256"])
    rankers, provenance = {}, {}
    for name in names:
        hashes = expected.get(name, set())
        if len(hashes) != 1:
            raise SystemExit(f"pi0 {LEARNED_TARGET} hash for {name} is not unique across "
                             f"the manifests: {sorted(hashes)}")
        digest = next(iter(hashes))
        path = MODEL_DIR / f"{name}__{LEARNED_TARGET}.json"
        rankers[name] = deserialize_ranker(path, digest)     # raises on a hash mismatch
        provenance[name] = {"path": str(path.relative_to(REPOSITORY)), "sha256": digest}
    return rankers, provenance


def load_reference_rows() -> list[dict[str, str]]:
    with REFERENCE_SEEDS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell_seed(row) -> tuple:
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]), int(row["seed"]))


# --- the replay worker --------------------------------------------------------------------


def _replay_worker(task):
    """One replay of one rung under one mechanism, with the attribution hooks."""
    name, fraction, multiplier, eligibility, width, rung, seed = task
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    common = dict(
        hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=SHARED["groups"][name], measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=width, l2_seed=seed,
        l2_eligibility=eligibility, l2_arm=rung,
        l2_request_hook=collector.on_request, l2_removal_hook=collector,
    )
    started = time.time()
    if rung == "lru":
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, "lru", **common)
    elif rung == "learned":
        scorer = RawFeatureScorer(trace, SHARED["rankers"][name])
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, "learned",
                              l2_scorer=scorer, **common)
    elif rung == "label":
        scorer = ExactLabelScorer(trace, SHARED["horizons"][name], target=LEARNED_TARGET)
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, "learned",
                              l2_scorer=scorer, **common)
    elif rung == "offline":
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, "offline_next_use",
                              l2_sampled_offline=True, **common)
    else:
        raise ValueError(f"unknown rung {rung!r}")
    seconds = time.time() - started
    # The partition and per-block identities of Phase 0.98b, and every counter
    # the replay keeps for itself, must agree with the attribution built beside it.
    collector.check_against(result)
    requested = max(result.requested_tokens, 1)
    row = {
        "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
        "cell": rdp.cell_label(fraction, multiplier), "eligibility": eligibility,
        "width": width, "mechanism": mechanism_label(eligibility, width), "rung": rung,
        "seed": seed,
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
        "worker_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
        "worker_pss_mib_end": _proc_mib("/proc/self/smaps_rollup", "Pss"),
        **collector.loss_row(),
        **collector.orphaning_row(),
    }
    row["extra_fraction_of_input"] = row["extra_avoided_tokens"] / requested
    for column, points in POINT_COLUMNS:
        row[points] = 100.0 * row[column] / requested
    # Which decision the per-block absent charge falls on, as shares of the two
    # decision categories together (nan when no block was charged to either).
    charged = row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
    for category in ("rejected", "evicted"):
        row[f"absent_{category}_share_of_decision_absent"] = (
            row[f"absent_{category}_tokens"] / charged if charged else math.nan
        )
    return row


def build_tasks(names, cells, seeds) -> list[tuple]:
    """Every replay, most expensive first so the pool's tail is short. The order
    only schedules work; every table is sorted before it is written."""
    cost = {"learned": 0, "label": 1, "lru": 2, "offline": 3}
    tasks = [(name, fraction, multiplier, eligibility, width, rung, seed)
             for name in names for fraction, multiplier in cells
             for eligibility, width in MECHANISMS for rung in RUNGS for seed in seeds]
    tasks.sort(key=lambda task: (cost[task[5]], -task[4], task[0], task[1], task[2],
                                 task[3], task[6]))
    return tasks


# --- checks ---------------------------------------------------------------------------------


def check_reproduction(rows, reference_rows, expected_per_rung: int) -> dict[str, dict]:
    """(all, 16) lru and learned against the published lru_s and A_none rows."""
    published: dict[tuple, int] = {}
    for row in reference_rows:
        for rung, (kind, arm, target) in REPRODUCED_ARMS.items():
            if row["kind"] == kind and row["arm"] == arm and row["target"] == target:
                key = (rung,) + _cell_seed(row)
                if key in published:
                    raise SystemExit(f"duplicate published reference {key}")
                published[key] = int(row["avoided_prefill_tokens"])
    report = {}
    base = mechanism_label(*BASE_MECHANISM)
    for rung, (kind, arm, target) in REPRODUCED_ARMS.items():
        matched = missing = 0
        mismatches: list[str] = []
        for row in rows:
            if row["mechanism"] != base or row["rung"] != rung:
                continue
            expected = published.get((rung,) + _cell_seed(row))
            if expected is None:
                missing += 1
                row["reference_avoided_prefill_tokens"] = ""
                row["reproduces_reference"] = ""
                continue
            same = int(row["avoided_prefill_tokens"]) == expected
            row["reference_avoided_prefill_tokens"] = expected
            row["reproduces_reference"] = same
            if same:
                matched += 1
            else:
                mismatches.append(f"{row['trace']}/{row['cell']}/s{row['seed']}: "
                                  f"{row['avoided_prefill_tokens']} != {expected}")
        report[rung] = {"reference": f"{kind}/{arm}/{target or '-'}", "matched": matched,
                        "missing": missing, "mismatched": len(mismatches),
                        "expected": expected_per_rung, "mismatches": mismatches}
    return report


def check_leaf_closure(rows) -> list[str]:
    return [f"{row['trace']}/{row['cell']}/{row['mechanism']}/{row['rung']}/s{row['seed']}: "
            f"{row['l2_present_unusable_tokens']} tokens, {row['l2_present_unusable_blocks']} blocks"
            for row in rows
            if row["eligibility"] == "leaf"
            and (int(row["l2_present_unusable_tokens"]) or int(row["l2_present_unusable_blocks"]))]


def check_invariants(rows, arms_per_group: int) -> tuple[int, list[str]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_cell_seed(row)].append(row)
    problems = []
    for key, members in sorted(groups.items()):
        if len(members) != arms_per_group:
            problems.append(f"{key}: {len(members)} arms, expected {arms_per_group}")
        for column in INVARIANT_COLUMNS:
            values = sorted({int(member[column]) for member in members})
            if len(values) != 1:
                problems.append(f"{key}: {column} takes {values}")
    return len(groups), problems


def select_heap_references(rows, reference_rows) -> tuple[dict[tuple, dict], list[str]]:
    """Published heap offline and heap LRU rows of every cell-seed of this run,
    accepted only when the identifiers this run measured match them."""
    ours: dict[tuple, dict] = {}
    for row in rows:
        ours.setdefault(_cell_seed(row), row)
    wanted = {arm: name for name, arm in HEAP_REFERENCES.items()}
    found: dict[tuple, dict] = defaultdict(dict)
    problems = []
    for row in reference_rows:
        if row["kind"] != "heap" or row["arm"] not in wanted or row["target"] != "":
            continue
        key = _cell_seed(row)
        if key not in ours:
            continue
        name = wanted[row["arm"]]
        if name in found[key]:
            problems.append(f"{key}: duplicate published {row['arm']} row")
            continue
        mine = ours[key]
        for field in ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                      "l1_avoided_tokens"):
            if int(row[field]) != int(mine[field]):
                problems.append(f"{key}/{row['arm']}: {field} {row[field]} != {mine[field]}")
        for field, value in (("l1_policy", rdp.L1_POLICY), ("hit_model", rdp.HIT_MODEL),
                             ("closure", rdp.CLOSURE), ("l2_eviction", "heap"),
                             ("offline_tiebreak", "prefix_first")):
            if row[field] != value:
                problems.append(f"{key}/{row['arm']}: {field} {row[field]!r} != {value!r}")
        found[key][name] = {"avoided_prefill_tokens": int(row["avoided_prefill_tokens"]),
                            "l1_avoided_tokens": int(row["l1_avoided_tokens"]),
                            "requested_tokens": int(row["requested_tokens"])}
    for key in sorted(ours):
        for name in HEAP_REFERENCES:
            if name not in found.get(key, {}):
                problems.append(f"{key}: published heap {HEAP_REFERENCES[name]} row missing")
    return dict(found), problems


# --- derivation ---------------------------------------------------------------------------------


def _stats(values) -> dict[str, float]:
    values = [float(value) for value in values]
    stats = summarize(values)
    return {"mean": stats["mean"], "std": stats["std"], "ci95_half": stats["ci95_half"],
            "min": min(values) if values else math.nan,
            "max": max(values) if values else math.nan}


def _put_stats(entry: dict, metric: str, values, fields=("mean", "std", "ci95_half", "min", "max")):
    stats = _stats(values)
    for field in fields:
        entry[f"{metric}_{field}"] = stats[field]


def _points(tokens, requested: int) -> float:
    return 100.0 * tokens / max(requested, 1)


def _sign_string(values) -> str:
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)


def aggregate_replays(rows) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["cell"],
                MECHANISMS.index((row["eligibility"], row["width"])),
                RUNGS.index(row["rung"]))].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        eligibility, width = MECHANISMS[key[4]]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                 "cell": key[3], "eligibility": eligibility, "width": width,
                 "mechanism": mechanism_label(eligibility, width), "rung": RUNGS[key[5]],
                 "seeds": len(members)}
        for column in ("requested_tokens", "l1_avoided_tokens", "l1_capacity_bytes",
                       "l2_capacity_bytes"):
            entry[column] = members[0][column]
        for metric in REPLAY_METRICS:
            _put_stats(entry, metric, [member[metric] for member in members
                                       if not _is_nan(member[metric])])
        out.append(entry)
    return out


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def index_utilities(rows) -> dict[tuple, dict]:
    """(trace, fraction, multiplier, mechanism, seed) -> rung -> seed row."""
    out: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        out[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["mechanism"],
             row["seed"])][row["rung"]] = row
    return out


def decomposition_seeds(rows, heap: dict[tuple, dict]) -> list[dict]:
    """One telescoping split per trace x cell x mechanism x seed, in integer tokens
    (exact) and in input-token points."""
    out = []
    for (name, fraction, multiplier, mechanism, seed), arms in sorted(
            index_utilities(rows).items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2],
                              [mechanism_label(*m) for m in MECHANISMS].index(item[0][3]),
                              item[0][4])):
        any_row = arms[RUNGS[0]]
        requested = int(any_row["requested_tokens"])
        l1_tokens = int(any_row["l1_avoided_tokens"])
        references = heap[(name, fraction, multiplier, seed)]
        extra = {rung: int(arms[rung]["extra_avoided_tokens"]) for rung in RUNGS}
        h_off = references["H_off"]["avoided_prefill_tokens"] - l1_tokens
        h_lru = references["H_lru"]["avoided_prefill_tokens"] - l1_tokens
        split = decompose(extra, h_off)
        entry = {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                 "cell": any_row["cell"], "eligibility": any_row["eligibility"],
                 "width": any_row["width"], "mechanism": mechanism, "seed": seed,
                 "requested_tokens": requested, "l1_avoided_tokens": l1_tokens,
                 "H_off_tokens": h_off, "H_lru_tokens": h_lru,
                 "H_off_points": _points(h_off, requested),
                 "H_lru_points": _points(h_lru, requested)}
        for rung in RUNGS:
            entry[f"U_{rung}_tokens"] = extra[rung]
            entry[f"U_{rung}_points"] = _points(extra[rung], requested)
        for gap in GAPS:
            entry[f"{gap}_tokens"] = split["gaps"][gap]
            entry[f"{gap}_points"] = _points(split["gaps"][gap], requested)
            entry[f"{gap}_share"] = split["shares"][gap]
        entry["T_tokens"] = split["total"]
        entry["T_points"] = _points(split["total"], requested)
        entry["identity_exact"] = sum(split["gaps"].values()) == split["total"]
        entry["dominant"] = split["dominant"]
        for pair, holds in ladder_holds(extra).items():
            entry[pair] = holds
        out.append(entry)
    return out


def decomposition_summary(seed_rows) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["cell"],
                MECHANISMS.index((row["eligibility"], row["width"])))].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        eligibility, width = MECHANISMS[key[4]]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                 "cell": key[3], "eligibility": eligibility, "width": width,
                 "mechanism": mechanism_label(eligibility, width), "seeds": len(members)}
        for metric in (["H_off_points", "H_lru_points"] + [f"U_{rung}_points" for rung in RUNGS]
                       + [f"{gap}_points" for gap in GAPS] + ["T_points"]):
            _put_stats(entry, metric, [member[metric] for member in members],
                       fields=("mean", "ci95_half", "min", "max"))
        means = {gap: entry[f"{gap}_points_mean"] for gap in GAPS}
        total = entry["T_points_mean"]
        for gap in GAPS:
            entry[f"{gap}_share_of_mean_T"] = means[gap] / total if total != 0 else math.nan
            positive, zero, negative = sign_counts([member[f"{gap}_tokens"] for member in members])
            entry[f"{gap}_seeds_pos"] = positive
            entry[f"{gap}_seeds_zero"] = zero
            entry[f"{gap}_seeds_neg"] = negative
        entry["dominant"] = dominant_label(means, total)
        if entry["dominant"] == "mixed":
            entry["dominant_term"] = ""
            entry["dominant_sign_consistent"] = ""
        else:
            term = next(gap for gap in GAPS if DOMINANT_LABELS[gap] == entry["dominant"])
            entry["dominant_term"] = term
            entry["dominant_sign_consistent"] = (
                sign_reading([member[f"{term}_tokens"] for member in members]) != "mixed"
            )
        entry["seed_dominant"] = "|".join(member["dominant"] for member in members)
        entry["identity_exact_all_seeds"] = all(member["identity_exact"] for member in members)
        out.append(entry)
    return out


def mechanism_effects(rows) -> list[dict]:
    """Seed-paired differences of U between mechanisms, for every rung (reading 2)."""
    utilities = index_utilities(rows)
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for (name, fraction, multiplier, mechanism, seed), arms in utilities.items():
        groups[(name, fraction, multiplier)].setdefault(seed, {})[mechanism] = arms
    out = []
    for (name, fraction, multiplier), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        for rung in RUNGS:
            for later, earlier in CONTRASTS:
                differences, tokens = [], []
                for seed in seeds:
                    arms = by_seed[seed]
                    if later not in arms or earlier not in arms:
                        continue
                    after, before = arms[later][rung], arms[earlier][rung]
                    delta = int(after["extra_avoided_tokens"]) - int(before["extra_avoided_tokens"])
                    tokens.append(delta)
                    differences.append(_points(delta, int(after["requested_tokens"])))
                if not differences:
                    continue
                entry = {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                         "cell": rdp.cell_label(fraction, multiplier), "rung": rung,
                         "contrast": f"{later}_minus_{earlier}", "later": later,
                         "earlier": earlier, "seeds": len(differences)}
                _put_stats(entry, "diff_points", differences,
                           fields=("mean", "ci95_half", "min", "max"))
                entry["diff_tokens_mean"] = float(np.mean(tokens))
                positive, zero, negative = sign_counts(differences)
                entry.update(n_pos=positive, n_zero=zero, n_neg=negative,
                             seed_signs=_sign_string(differences),
                             reading=sign_reading(differences))
                out.append(entry)
    return out


def ladder_table(seed_rows) -> tuple[list[dict], list[dict]]:
    """Ladder order per trace x cell x mechanism (reading 3), and every inversion."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["cell"],
                MECHANISMS.index((row["eligibility"], row["width"])))].append(row)
    table, inversions = [], []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        eligibility, width = MECHANISMS[key[4]]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                 "cell": key[3], "eligibility": eligibility, "width": width,
                 "mechanism": mechanism_label(eligibility, width), "seeds": len(members)}
        ordered = bool(members)
        for lower, upper in LADDER_PAIRS:
            pair = pair_name(lower, upper)
            holds = sum(1 for member in members if member[pair])
            differences = [member[f"U_{upper}_points"] - member[f"U_{lower}_points"]
                           for member in members]
            entry[f"{pair}_seeds"] = holds
            entry[f"{pair}_mean_diff_points"] = float(np.mean(differences))
            entry[f"{pair}_min_diff_points"] = float(np.min(differences))
            ordered = ordered and holds == len(members)
            if holds < len(members):
                inversions.append({
                    "trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                    "cell": key[3], "mechanism": entry["mechanism"], "pair": pair,
                    "seeds_inverted": len(members) - holds, "seeds": len(members),
                    "mean_diff_points": float(np.mean(differences)),
                    "seed_signs": _sign_string(differences),
                })
        entry["ordered"] = ordered
        table.append(entry)
    return table, inversions


def capacity_pattern(seed_rows) -> list[dict]:
    """Signs of U(learned) - U(lru) and U(label) - U(lru) by cell (reading 5)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["cell"],
                MECHANISMS.index((row["eligibility"], row["width"])))].append(row)
    out = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        eligibility, width = MECHANISMS[key[4]]
        entry = {"trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                 "cell": key[3], "eligibility": eligibility, "width": width,
                 "mechanism": mechanism_label(eligibility, width), "seeds": len(members)}
        for rung in ("learned", "label"):
            differences = [member[f"U_{rung}_points"] - member["U_lru_points"]
                           for member in members]
            positive, zero, negative = sign_counts(differences)
            entry[f"{rung}_minus_lru_mean_points"] = float(np.mean(differences))
            entry[f"{rung}_minus_lru_seed_signs"] = _sign_string(differences)
            entry[f"{rung}_minus_lru_n_pos"] = positive
            entry[f"{rung}_minus_lru_n_zero"] = zero
            entry[f"{rung}_minus_lru_n_neg"] = negative
            entry[f"{rung}_minus_lru_reading"] = sign_reading(differences)
        out.append(entry)
    return out


# --- figures ---------------------------------------------------------------------------------

# Fixed categorical order (validated: adjacent CVD and normal-vision separation
# pass; two slots sit below 3:1 on white, so every series also differs in line
# style or marker and the CSVs are the table view).
MECHANISM_STYLE = {
    "all16": ("#2a78d6", "-", "o", "all, K=16 (published)"),
    "leaf16": ("#eb6834", "--", "o", "leaf, K=16"),
    "all64": ("#1baf7a", "-", "s", "all, K=64"),
    "leaf64": ("#eda100", "--", "s", "leaf, K=64"),
}
GAP_STYLE = {
    "candidate_search": ("#2a78d6", "candidate-search gap  H_off - U(offline)"),
    "objective": ("#eb6834", "objective gap  U(offline) - U(label)"),
    "signal": ("#1baf7a", "signal gap  U(label) - U(learned)"),
    "achieved": ("#eda100", "achieved  U(learned) - U(lru)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _panel_grid(entries):
    traces = sorted({entry["trace"] for entry in entries})
    cells = sorted({(entry["l1_fraction"], entry["l2_multiplier"]) for entry in entries})
    return traces, cells


def _style_axis(axis):
    axis.grid(True, axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=7)


def ladder_figure(directory: Path, decomposition: list[dict]) -> None:
    traces, cells = _panel_grid(decomposition)
    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(7.0, 2.9 * len(cells) + 0.6), 2.9 * len(traces) + 1.0))
    positions = np.arange(len(RUNGS))
    index = {(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"],
              entry["mechanism"]): entry for entry in decomposition}
    for row_index, name in enumerate(traces):
        for column, (fraction, multiplier) in enumerate(cells):
            axis = axes[row_index][column]
            references = None
            for mechanism, (color, style, marker, label) in MECHANISM_STYLE.items():
                entry = index.get((name, fraction, multiplier, mechanism))
                if entry is None:
                    continue
                references = entry
                means = [entry[f"U_{rung}_points_mean"] for rung in RUNGS]
                lows = [entry[f"U_{rung}_points_min"] for rung in RUNGS]
                highs = [entry[f"U_{rung}_points_max"] for rung in RUNGS]
                axis.fill_between(positions, lows, highs, color=color, alpha=0.14, linewidth=0)
                axis.plot(positions, means, color=color, linestyle=style, marker=marker,
                          markersize=4.5, linewidth=1.6, label=label)
            if references is not None:
                axis.axhline(references["H_off_points_mean"], color=INK, linestyle=":",
                             linewidth=1.0, label="heap offline (H_off)")
                axis.axhline(references["H_lru_points_mean"], color=MUTED, linestyle="-.",
                             linewidth=1.0, label="heap LRU (H_lru)")
            axis.set_xticks(positions, RUNGS)
            axis.set_title(f"{name.replace('_trace', '')} · L1 {100 * fraction:g}% · L2 x{multiplier:g}",
                           fontsize=8, color=INK)
            if column == 0:
                axis.set_ylabel("extra avoided, input-token points", fontsize=7.5, color=INK)
            _style_axis(axis)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3 if len(cells) > 2 else 2,
                  fontsize=7, frameon=False)
    figure.suptitle("Score ladder under four L2 mechanisms (mean over seeds; band = seed min–max)",
                    fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.10 if len(traces) > 1 else 0.17, 1.0, 0.95))
    figure.savefig(directory / "ladder_by_mechanism.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def gap_figure(directory: Path, decomposition: list[dict]) -> None:
    traces, cells = _panel_grid(decomposition)
    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(7.0, 2.9 * len(cells) + 0.6), 2.9 * len(traces) + 1.2))
    mechanisms = list(MECHANISM_STYLE)
    positions = np.arange(len(mechanisms))
    index = {(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"],
              entry["mechanism"]): entry for entry in decomposition}
    for row_index, name in enumerate(traces):
        for column, (fraction, multiplier) in enumerate(cells):
            axis = axes[row_index][column]
            up = np.zeros(len(mechanisms))
            down = np.zeros(len(mechanisms))
            for gap, (color, label) in GAP_STYLE.items():
                values = np.array([
                    index[(name, fraction, multiplier, mechanism)][f"{gap}_points_mean"]
                    if (name, fraction, multiplier, mechanism) in index else 0.0
                    for mechanism in mechanisms])
                positive = np.where(values > 0, values, 0.0)
                negative = np.where(values < 0, values, 0.0)
                axis.bar(positions, positive, bottom=up, color=color, width=0.66,
                         edgecolor="white", linewidth=0.8, label=label)
                axis.bar(positions, negative, bottom=down, color=color, width=0.66,
                         edgecolor="white", linewidth=0.8)
                up += positive
                down += negative
            totals = [index[(name, fraction, multiplier, mechanism)]["T_points_mean"]
                      if (name, fraction, multiplier, mechanism) in index else math.nan
                      for mechanism in mechanisms]
            axis.plot(positions, totals, linestyle="none", marker="_", markersize=14,
                      markeredgewidth=1.6, color=INK, label="T = H_off - U(lru)")
            axis.axhline(0.0, color=MUTED, linewidth=0.8)
            axis.margins(y=0.08)
            axis.set_xticks(positions, mechanisms)
            axis.set_title(f"{name.replace('_trace', '')} · L1 {100 * fraction:g}% · L2 x{multiplier:g}",
                           fontsize=8, color=INK)
            if column == 0:
                axis.set_ylabel("input-token points", fontsize=7.5, color=INK)
            _style_axis(axis)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3 if len(cells) > 2 else 2,
                  fontsize=7, frameon=False)
    figure.suptitle("Headroom decomposition by mechanism (mean over seeds; negative terms below zero)",
                    fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.11 if len(traces) > 1 else 0.20, 1.0, 0.95))
    figure.savefig(directory / "gap_decomposition.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


README_TEXT = """# Mechanism control: score ladder under four L2 mechanisms

Pre-registration: `docs/mechanism-control-plan.md`. Rungs: `lru` (sampled LRU key), `learned` (frozen pi0 `next_use` ranker), `label` (exact training target at the decision), `offline` (heap offline comparator's key on the sampled path). Mechanisms: eligibility `all` | `leaf` x sample width K = 16 | 64; `all16` is the published one. U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. `H_off` and `H_lru` are the published heap rows of `decision_population_replay_seeds.csv`, not rerun. Seed intervals describe sampling-seed variability only.

- `replay_seeds.csv`: one row per replay (trace x cell x mechanism x rung x seed) with the replay counters, the Phase 0.98b attribution columns, the reproduction mark of the base mechanism, timing and worker memory.
- `replay.csv`: five-seed mean, std, 95% t half-width, min and max of the main replay metrics per trace x cell x mechanism x rung.
- `decomposition_seeds.csv`: per seed, the heap references, the four U's, the four terms of `T = H_off - U(lru)` in exact integer tokens, in points and as shares of T, the per-seed dominant label and the per-seed ladder pairs.
- `decomposition.csv`: the same aggregated over seeds, with the shares and the dominant label taken on the five-seed means, the seed sign counts of every term, and whether the dominant term has one sign in every seed.
- `mechanism_effects.csv`: seed-paired differences of U between mechanisms (`leaf16 - all16`, `all64 - all16`, `leaf64 - leaf16`, `leaf64 - all16`) for every rung, with the seed sign counts and the consistent-gain / consistent-loss / mixed reading.
- `ladder.csv` and `ladder_inversions.csv`: for every trace x cell x mechanism, how many seeds satisfy each adjacent pair of `lru <= learned <= label <= offline`, the ordered flag, and every pair that fails in at least one seed.
- `capacity_pattern.csv`: mean and seed signs of `U(learned) - U(lru)` and `U(label) - U(lru)` per trace x cell x mechanism.
- `ladder_by_mechanism.png`: U per rung, one line per mechanism with its seed min-max band, heap references as horizontal lines.
- `gap_decomposition.png`: the four terms of T per mechanism as stacked bars (negative terms below zero), T marked.
- `run_config.json`: plan and code commits, source manifest, trace, model and reference hashes, grid, check counts, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------


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

    status = _git("status", "--porcelain")
    differing = sources_differing_from_head()
    if not args.allow_dirty:
        if status:
            raise SystemExit(f"the working tree is not clean:\n{status}")
        if differing:
            raise SystemExit(f"execution source differs from HEAD: {differing}")
    head_start = _git("rev-parse", "HEAD")
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()
    if args.allow_dirty and (status or differing):
        print(f"WARNING smoke from a dirty tree: {len(status.splitlines())} status lines, "
              f"{len(differing)} execution files differ from HEAD", flush=True)

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
    rankers, model_files = load_models(names)
    reference_rows = load_reference_rows()
    reference_sha = sha256_path(REFERENCE_SEEDS)
    for name in names:
        print(f"  pi0 {name}: {model_files[name]['path']} sha256={model_files[name]['sha256'][:12]}",
              flush=True)

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_set)
    SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                  rankers=rankers)
    args.output_dir.mkdir(parents=True)

    tasks = build_tasks(names, cells, seeds)
    workers = min(args.workers, len(tasks))
    print(f"{len(tasks)} replays on {workers} workers "
          f"({len(names)} traces x {len(cells)} cells x {len(MECHANISMS)} mechanisms x "
          f"{len(RUNGS)} rungs x {len(seeds)} seeds)", flush=True)
    available_start = _proc_mib("/proc/meminfo", "MemAvailable")
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
                          f"last={row['trace']}/{row['cell']}/{row['mechanism']}/{row['rung']}"
                          f"/s{row['seed']} ({row['seconds']:.0f}s, "
                          f"worker peak {row['worker_peak_rss_mib']:.0f} MiB)", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        # Joined, so every worker is reaped and counted in RUSAGE_CHILDREN.
        pool.join()
    replay_seconds = time.time() - started
    children_peak = _peak_rss_mib(resource.RUSAGE_CHILDREN)
    worker_peak = max(float(row["worker_peak_rss_mib"]) for row in rows)
    worker_pss = max((float(row["worker_pss_mib_end"]) for row in rows
                      if math.isfinite(float(row["worker_pss_mib_end"]))), default=math.nan)
    print(f"  replays in {replay_seconds:.0f}s; peak worker RSS {worker_peak:.0f} MiB "
          f"(children max {children_peak:.0f} MiB), max worker PSS at replay end "
          f"{worker_pss:.0f} MiB, parent peak {_peak_rss_mib(resource.RUSAGE_SELF):.0f} MiB, "
          f"MemAvailable at start {available_start:.0f} MiB", flush=True)

    timing: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        timing[(MECHANISMS.index((row["eligibility"], row["width"])),
                RUNGS.index(row["rung"]))].append(float(row["seconds"]))
    print("  seconds per replay, by mechanism and rung:", flush=True)
    timing_rows = []
    for (mechanism_index, rung_index), values in sorted(timing.items()):
        label = mechanism_label(*MECHANISMS[mechanism_index])
        print(f"    {label:7s} {RUNGS[rung_index]:8s} n={len(values):4d} "
              f"mean={np.mean(values):7.1f} max={np.max(values):7.1f} "
              f"total={np.sum(values):8.1f}", flush=True)
        timing_rows.append({"mechanism": label, "rung": RUNGS[rung_index], "n": len(values),
                            "mean_seconds": float(np.mean(values)),
                            "max_seconds": float(np.max(values)),
                            "total_seconds": float(np.sum(values))})

    # --- checks: nothing is derived or published unless every one passes ---
    failures: list[str] = []
    expected_per_rung = len(names) * len(cells) * len(seeds)
    reproduction = check_reproduction(rows, reference_rows, expected_per_rung)
    for rung, report in reproduction.items():
        print(f"  reproduction (all16 {rung} vs published {report['reference']}): "
              f"{report['matched']} matched, {report['missing']} missing, "
              f"{report['mismatched']} mismatched (expected {report['expected']})", flush=True)
        for line in report["mismatches"]:
            print(f"    MISMATCH {line}", flush=True)
        if report["mismatched"] or report["missing"] or report["matched"] != report["expected"]:
            failures.append(f"reproduction of {report['reference']} failed")
    leaf_problems = check_leaf_closure(rows)
    leaf_rows = sum(1 for row in rows if row["eligibility"] == "leaf")
    print(f"  leaf closure: {leaf_rows - len(leaf_problems)}/{leaf_rows} leaf replays with zero "
          f"present-unusable tokens and blocks", flush=True)
    for line in leaf_problems:
        print(f"    PRESENT-UNUSABLE {line}", flush=True)
    if leaf_problems:
        failures.append("a leaf replay left a block present but unusable")
    cell_seeds, invariant_problems = check_invariants(rows, len(MECHANISMS) * len(RUNGS))
    print(f"  invariants over {cell_seeds} trace x cell x seed groups "
          f"({', '.join(INVARIANT_COLUMNS)}): "
          f"{'hold' if not invariant_problems else 'BROKEN'}", flush=True)
    for line in invariant_problems[:20]:
        print(f"    VARIES {line}", flush=True)
    if invariant_problems:
        failures.append("an arm-independent counter varies across arms")
    heap, heap_problems = select_heap_references(rows, reference_rows)
    print(f"  heap references: {sum(len(v) for v in heap.values())} rows for {len(heap)} "
          f"cell-seeds, {len(heap_problems)} problems", flush=True)
    for line in heap_problems[:20]:
        print(f"    HEAP {line}", flush=True)
    if heap_problems:
        failures.append("heap references missing or not matching")
    unexplained = sum(int(row["absent_unexplained_tokens"]) for row in rows)
    print(f"  unexplained per-block absent tokens over the run: {unexplained}", flush=True)
    if failures:
        raise SystemExit("checks failed, nothing derived or published: " + "; ".join(failures))

    # --- derived tables ---
    key = lambda row: (row["trace"], row["l1_fraction"], row["l2_multiplier"],
                       MECHANISMS.index((row["eligibility"], row["width"])),
                       RUNGS.index(row["rung"]), row["seed"])
    rows.sort(key=key)
    replay_summary = aggregate_replays(rows)
    decomposition_rows = decomposition_seeds(rows, heap)
    if not all(row["identity_exact"] for row in decomposition_rows):
        raise SystemExit("the telescoping identity failed on integer tokens; nothing published")
    decomposition = decomposition_summary(decomposition_rows)
    effects = mechanism_effects(rows)
    ladder, inversions = ladder_table(decomposition_rows)
    capacity = capacity_pattern(decomposition_rows)

    print("  decomposition (means over seeds, input-token points):", flush=True)
    for entry in decomposition:
        print(f"    {entry['trace'][:12]:12s} {entry['cell']:16s} {entry['mechanism']:7s} "
              + " ".join(f"U_{rung}={entry[f'U_{rung}_points_mean']:7.3f}" for rung in RUNGS)
              + f" | H_off={entry['H_off_points_mean']:7.3f} H_lru={entry['H_lru_points_mean']:7.3f}"
              + " | " + " ".join(f"{gap}={entry[f'{gap}_points_mean']:7.3f}" for gap in GAPS)
              + f" T={entry['T_points_mean']:7.3f} -> {entry['dominant']}", flush=True)
    total_cells = len(names) * len(cells)
    print("  ladder order (lru <= learned <= label <= offline in every seed):", flush=True)
    for eligibility, width in MECHANISMS:
        label = mechanism_label(eligibility, width)
        ordered = sum(1 for entry in ladder if entry["mechanism"] == label and entry["ordered"])
        print(f"    {label:7s} ordered in {ordered}/{total_cells} trace x cell", flush=True)
    for inversion in inversions:
        print(f"    INVERSION {inversion['trace']}/{inversion['cell']}/{inversion['mechanism']}: "
              f"{inversion['pair']} fails in {inversion['seeds_inverted']}/{inversion['seeds']} "
              f"seeds (mean diff {inversion['mean_diff_points']:.3f} points)", flush=True)

    # The source and HEAD must not have moved while the replays ran.
    manifest_end = source_manifest()
    head_end = _git("rev-parse", "HEAD")
    if manifest_end != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "mechanism_control",
        "smoke": bool(args.smoke),
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "git_dirty_at_start": bool(status),
        "execution_sources_differing_from_head": differing,
        "allow_dirty": bool(args.allow_dirty),
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "models": model_files,
        "model_manifests": {str(path.relative_to(REPOSITORY)): sha256_path(path)
                            for path in MODEL_MANIFESTS},
        "reference_seeds": {"path": str(REFERENCE_SEEDS.relative_to(REPOSITORY)),
                            "sha256": reference_sha},
        "traces": list(names), "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "eligibilities": list(ELIGIBILITIES), "widths": list(WIDTHS),
        "mechanisms": [mechanism_label(*m) for m in MECHANISMS],
        "base_mechanism": mechanism_label(*BASE_MECHANISM), "rungs": list(RUNGS),
        "contrasts": [f"{later}_minus_{earlier}" for later, earlier in CONTRASTS],
        "learned_target": LEARNED_TARGET, "label_horizon_seconds": LABEL_HORIZON_SECONDS,
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "arrival_protection": "none", "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "checks": {
            "reproduction": {rung: {k: v for k, v in report.items() if k != "mismatches"}
                             for rung, report in reproduction.items()},
            "leaf_replays": leaf_rows, "leaf_present_unusable_violations": len(leaf_problems),
            "invariant_groups": cell_seeds, "invariant_violations": len(invariant_problems),
            "invariant_columns": list(INVARIANT_COLUMNS),
            "heap_reference_cell_seeds": len(heap),
            "heap_reference_problems": len(heap_problems),
            "unexplained_absent_tokens": unexplained,
            "identity_exact_seed_rows": len(decomposition_rows),
        },
        "replays": len(rows), "workers": workers, "replay_seconds": replay_seconds,
        "wall_seconds": time.time() - clock,
        "seconds_by_arm": timing_rows,
        "memory": {
            "rss_unit": "MiB (Linux ru_maxrss KiB / 1024)",
            "peak_worker_rss_mib": worker_peak,
            "children_peak_rss_mib": children_peak,
            "max_worker_pss_mib_at_replay_end": worker_pss,
            "parent_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
            "mem_available_mib_at_start": available_start,
        },
        "note": "Descriptive control. Nothing is fitted; the learned rung is the frozen pi0 "
                "next_use ranker loaded by hash, the label rung reads the exact training target, "
                "the offline rung uses the heap comparator's key on the sampled path, and the heap "
                "references are the published Phase 0.97 rows. Seed intervals describe "
                "sampling-seed variability only.",
    }
    destinations = [args.output_dir]
    if args.paper_dir is not None and not args.smoke:
        destinations.append(args.paper_dir)
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=directory == args.output_dir)
        _write(directory / "replay_seeds.csv", rows)
        _write(directory / "replay.csv", replay_summary)
        _write(directory / "decomposition_seeds.csv", decomposition_rows)
        _write(directory / "decomposition.csv", decomposition)
        _write(directory / "mechanism_effects.csv", effects)
        _write(directory / "ladder.csv", ladder)
        if inversions:
            _write(directory / "ladder_inversions.csv", inversions)
        else:
            (directory / "ladder_inversions.csv").write_text(
                "trace,l1_fraction,l2_multiplier,cell,mechanism,pair,seeds_inverted,seeds,"
                "mean_diff_points,seed_signs\n", encoding="utf-8")
        _write(directory / "capacity_pattern.csv", capacity)
        ladder_figure(directory, decomposition)
        gap_figure(directory, decomposition)
        (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
        (directory / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"written to {', '.join(str(d) for d in destinations)}; done in "
          f"{time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
