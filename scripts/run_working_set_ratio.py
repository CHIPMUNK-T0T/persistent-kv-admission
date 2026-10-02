#!/usr/bin/env python3
"""Working-set ratio check: does a reuse working set locate the capacity reversals?

The pre-registration is `docs/working-set-ratio-plan.md`; nothing here may
change it. For each real trace and L1 fraction the L1 offer (victim) stream of
heap-LRU L1 is recorded exactly as the two-tier simulator produces it — the
Phase 0.97 L1-only call, `run_two_tier` with an L2 of zero bytes and a
`victim_hook` — and every returning offer gets its stack distance `D` in the
plan's admit-all, offer-ordered, exclusive reference tier
(`persistent_kv_admission.workingset`). The working set `W(f)` is the
token-weighted median of `D + size` over the offers whose return falls in the
evaluation window, and `gamma = W(f) / C_L2` per cell.

Three published outcomes are compared with the sign the ratio predicts: heap
`lru_2hit - lru` and sampled `lru_2hit_s - lru_s` from
`results/paper/decision_population_replay_seeds.csv`, and arrival protection
`all - none` from `results/paper/phase1_intervention_pairs_seeds.csv`. No L2
policy is run, nothing is fitted and nothing is proposed.

Before anything is derived the run checks, per trace and L1 fraction, that the
recorded stream has as many offers as every published replay has L1 evictions
and as the published mechanism-control rows have admissions plus rejections,
that the L1 capacity, requested tokens and L1-avoided tokens are the published
ones, that the L2 capacities are the published ones, and that `D + size <= C`
agrees with a direct simulation of the reference tier for every returning
offer at both L2 capacities. Any failure exits without writing a table.

The run refuses an existing output directory and an execution tree that is not
committed.
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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from persistent_kv_admission.decisionpop import horizon_for
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.mechanism import sign_counts
from persistent_kv_admission.onpolicy import sha256_path
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.workingset import (
    QUANTILES,
    THRESHOLD,
    TRANSITION,
    agreement,
    declinable,
    in_transition,
    interval_contains,
    observation,
    predicted_sign,
    ratio,
    record_offer_stream,
    reference_tier_held,
    share,
    stack_distances,
    threshold_intervals,
    verdict,
    working_set,
)

PHASE097_SCRIPT = REPOSITORY / "scripts/run_decision_population.py"
PLAN_PATH = REPOSITORY / "docs/working-set-ratio-plan.md"
# Published inputs, anchored at the repository and hashed into the run config.
DECISION_SEEDS = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"
PHASE1_PAIRS = REPOSITORY / "results/paper/phase1_intervention_pairs_seeds.csv"
MECHANISM_REPLAY = REPOSITORY / "results/paper/mechanism_control_001/replay.csv"
REFERENCES = (DECISION_SEEDS, PHASE1_PAIRS, MECHANISM_REPLAY)


def _load_phase097():
    """Import the Phase 0.97 runner by path, unchanged, for its capacity rule
    and its constants; `main` is behind the usual guard, so nothing runs."""
    spec = importlib.util.spec_from_file_location("run_decision_population", PHASE097_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rdp = _load_phase097()

# --- the fixed grid (docs/working-set-ratio-plan.md) ------------------------------------

TRACES = ("conversation_trace", "toolagent_trace")
FRACTIONS = (0.0025, 0.01, 0.02)
MULTIPLIERS = (1.0, 4.0)
CELLS = tuple((fraction, multiplier) for fraction in FRACTIONS for multiplier in MULTIPLIERS)
# "Per real trace x cell (12)": the denominator of every agreement count.
EXPECTED_CELLS = len(TRACES) * len(CELLS)
SEEDS = (0, 1, 2, 3, 4)

# The three outcomes, in the plan's order. `declining_sign` is the sign the
# outcome takes where declining writes helps (gamma > 1): 2-hit admission
# declines first-time writes (+), arrival protection forbids declining (-).
OUTCOMES = (
    {"outcome": "heap_lru_2hit_minus_lru", "number": 1, "declining_sign": 1,
     "label": "heap lru_2hit - lru", "source": DECISION_SEEDS, "paired": False},
    {"outcome": "sampled_lru_2hit_s_minus_lru_s", "number": 2, "declining_sign": 1,
     "label": "sampled lru_2hit_s - lru_s", "source": DECISION_SEEDS, "paired": True},
    {"outcome": "protection_all_minus_none", "number": 3, "declining_sign": -1,
     "label": "arrival protection all - none", "source": PHASE1_PAIRS, "paired": True},
)
OUTCOME = {spec["outcome"]: spec for spec in OUTCOMES}
# Phase 1 carries `all - none` for two targets of the B arm. The findings
# (docs/phase1-intervention-findings.md) name `next_use` the primary target
# ("The next-use result remains the primary verdict") and `binary` the fixed
# robustness target, so `binary` is a secondary row everywhere.
PHASE1_COMPARISON = ("all", "none", "all-none")
PHASE1_PRIMARY_TARGET = "next_use"
PHASE1_SECONDARY_TARGETS = ("binary",)
# Reading 4: the rejection share of these rungs under this mechanism.
DECLINABLE_RUNGS = ("label", "learned")
DECLINABLE_MECHANISM = "all16"
# Floats compare exactly with 0.5, 1 and 2 only below this (workingset.ratio).
EXACT_LIMIT = 2 ** 53
SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path, help="the two real Mooncake trace files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new run directory; refused if it exists")
    parser.add_argument("--workers", type=int, default=len(TRACES) * len(FRACTIONS))
    return parser.parse_args()


# --- provenance ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, check=True,
                          capture_output=True, text=True).stdout.strip()


def execution_sources() -> list[Path]:
    """Every file the run executes from this repository."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.update((Path(__file__).resolve(), PHASE097_SCRIPT))
    return sorted(paths)


def source_manifest() -> dict[str, object]:
    files = {str(path.relative_to(REPOSITORY)): sha256_path(path) for path in execution_sources()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def sources_differing_from_head() -> list[str]:
    """Execution files and published inputs whose working copy is not
    byte-identical to HEAD's (an untracked file differs by definition)."""
    differing = []
    for path in execution_sources() + list(REFERENCES):
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


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display(path: Path) -> str:
    """A path relative to the repository when it lies inside it."""
    try:
        return str(path.resolve().relative_to(REPOSITORY))
    except ValueError:
        return str(path.resolve())


# --- the offer streams --------------------------------------------------------------------


def stream_worker(task) -> dict[str, object]:
    """One L1 offer stream, its stack distances, its working set, and the
    reading-4 counts at each L2 capacity of its cells, with the equivalence of
    `D + size <= C` and the reference tier checked on every returning offer."""
    name, fraction = task
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    l1_bytes = rdp._capacity(name, fraction)
    started = time.time()
    offers, result = record_offer_stream(trace, rdp.L1_POLICY, l1_bytes, SHARED["groups"][name],
                                         measure_from_ms=split_ms)
    recorded = time.time()
    distances = stack_distances(offers)
    distance_seconds = time.time() - recorded
    summary = working_set(offers, distances, split_ms)
    capacities = {}
    for multiplier in MULTIPLIERS:
        capacity = rdp._capacity(name, fraction * multiplier)
        held = reference_tier_held(offers, capacity)
        violations = sum(
            1 for offer, distance, kept in zip(offers, distances, held)
            if offer.return_group is not None and (distance + offer.size_bytes <= capacity) != kept
        )
        capacities[multiplier] = {
            "l2_capacity_bytes": capacity,
            "tier_equivalence_violations": violations,
            "returning_held_in_reference_tier": sum(1 for kept in held if kept),
            **declinable(offers, distances, capacity, split_ms),
        }
    return {
        "trace": name, "l1_fraction": fraction, "l1_capacity_bytes": l1_bytes,
        "split_ms": split_ms,
        "replay_l1_capacity_bytes": result.l1_capacity_bytes,
        "l1_evictions": result.l1_evictions,
        "requested_tokens": result.requested_tokens,
        "l1_avoided_tokens": result.l1_avoided_tokens,
        "offered_bytes": sum(offer.size_bytes for offer in offers),
        "working_set": summary, "capacities": capacities,
        "max_distance_bytes": max((d for d in distances if d is not None), default=0),
        "record_seconds": recorded - started, "distance_seconds": distance_seconds,
        "seconds": time.time() - started,
        "worker_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
    }


def compute_streams(tasks, workers: int) -> list[dict]:
    """Every stream, in a fork pool (or in-process with one worker)."""
    if workers <= 1:
        return [stream_worker(task) for task in tasks]
    pool = mp.get_context("fork").Pool(min(workers, len(tasks)))
    try:
        streams = pool.map(stream_worker, tasks, chunksize=1)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return streams


# --- checks against the published tables ---------------------------------------------------


def _cell_key(row) -> tuple:
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]))


def check_streams(streams, decision_rows, mechanism_rows, phase1_rows) -> list[str]:
    """Every identifier the published replays share with these streams."""
    problems = []
    by_stream = {(s["trace"], float(s["l1_fraction"])): s for s in streams}
    for stream in streams:
        key = f"{stream['trace']}/{stream['l1_fraction']:g}"
        if stream["l1_evictions"] != stream["working_set"]["offers"]:
            problems.append(f"{key}: recorder saw {stream['working_set']['offers']} offers, "
                            f"the replay counted {stream['l1_evictions']} L1 evictions")
        if stream["replay_l1_capacity_bytes"] != stream["l1_capacity_bytes"]:
            problems.append(f"{key}: L1 capacity {stream['replay_l1_capacity_bytes']} != "
                            f"{stream['l1_capacity_bytes']}")
        for multiplier, entry in stream["capacities"].items():
            if entry["tier_equivalence_violations"]:
                problems.append(f"{key} x{multiplier:g}: D + size <= C disagrees with the "
                                f"reference tier on {entry['tier_equivalence_violations']} offers")
    matched = defaultdict(int)
    for row in decision_rows:
        trace, fraction, multiplier = _cell_key(row)
        stream = by_stream.get((trace, fraction))
        if stream is None or multiplier not in MULTIPLIERS:
            continue
        label = (f"decision_population {trace}/{rdp.cell_label(fraction, multiplier)}/"
                 f"{row['kind']}/{row['arm']}/{row['target'] or '-'}/s{row['seed']}")
        expected = {
            "l1_evictions": stream["l1_evictions"],
            "l1_capacity_bytes": stream["l1_capacity_bytes"],
            "l2_capacity_bytes": stream["capacities"][multiplier]["l2_capacity_bytes"],
            "requested_tokens": stream["requested_tokens"],
            "l1_avoided_tokens": stream["l1_avoided_tokens"],
        }
        for field, value in expected.items():
            if int(row[field]) != int(value):
                problems.append(f"{label}: {field} {row[field]} != {value}")
        for field, value in (("l1_policy", rdp.L1_POLICY), ("hit_model", rdp.HIT_MODEL),
                             ("closure", rdp.CLOSURE)):
            if row[field] != value:
                problems.append(f"{label}: {field} {row[field]!r} != {value!r}")
        matched["decision_population"] += 1
    for row in mechanism_rows:
        if row["mechanism"] != DECLINABLE_MECHANISM:
            continue
        trace, fraction, multiplier = _cell_key(row)
        stream = by_stream.get((trace, fraction))
        if stream is None or multiplier not in MULTIPLIERS:
            continue
        label = f"mechanism_control {trace}/{row['cell']}/{row['rung']}"
        offers = float(row["l2_admissions_mean"]) + float(row["l2_rejections_mean"])
        if abs(offers - stream["l1_evictions"]) > 1e-6:
            problems.append(f"{label}: admissions + rejections {offers} != "
                            f"{stream['l1_evictions']} offers")
        for field, value in (("l1_capacity_bytes", stream["l1_capacity_bytes"]),
                             ("l2_capacity_bytes", stream["capacities"][multiplier]["l2_capacity_bytes"]),
                             ("requested_tokens", stream["requested_tokens"]),
                             ("l1_avoided_tokens", stream["l1_avoided_tokens"])):
            if int(row[field]) != int(value):
                problems.append(f"{label}: {field} {row[field]} != {value}")
        matched["mechanism_control"] += 1
    for row in phase1_rows:
        trace, fraction, multiplier = _cell_key(row)
        stream = by_stream.get((trace, fraction))
        if stream is None:
            continue
        if int(row["requested_tokens"]) != stream["requested_tokens"]:
            problems.append(f"phase1 {trace}/{row['cell']}/{row['comparison']}/s{row['seed']}: "
                            f"requested_tokens {row['requested_tokens']} != {stream['requested_tokens']}")
        matched["phase1"] += 1
    for source in ("decision_population", "mechanism_control", "phase1"):
        if not matched[source]:
            problems.append(f"no {source} row matched any stream")
    return problems


# --- the published outcomes -----------------------------------------------------------------


def _points(tokens: int, requested: int) -> float:
    return 100.0 * tokens / max(requested, 1)


def heap_outcome(decision_rows, later: str = "lru_2hit", earlier: str = "lru") -> dict[tuple, dict]:
    """Outcome 1, deterministic: the heap rows are replicated once per seed in
    the published table; every copy must agree, and one is used."""
    found: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for row in decision_rows:
        if row["kind"] != "heap" or row["target"] != "" or row["arm"] not in (later, earlier):
            continue
        found[_cell_key(row)][row["arm"]].add(
            (int(row["avoided_prefill_tokens"]), int(row["requested_tokens"])))
    out = {}
    for key, arms in found.items():
        for arm in (later, earlier):
            if len(arms.get(arm, ())) != 1:
                raise SystemExit(f"{key}: heap {arm} has {len(arms.get(arm, ()))} distinct rows")
        (after, requested), (before, requested_before) = (next(iter(arms[later])),
                                                           next(iter(arms[earlier])))
        if requested != requested_before:
            raise SystemExit(f"{key}: requested tokens differ between heap arms")
        delta = after - before
        out[key] = {"tokens": [delta], "points": [_points(delta, requested)], "seeds": [],
                    "requested_tokens": requested}
    return out


def sampled_outcome(decision_rows, later: str = "lru_2hit_s", earlier: str = "lru_s") -> dict[tuple, dict]:
    """Outcome 2, seed-paired over the five published seeds."""
    found: dict[tuple, dict[str, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    for row in decision_rows:
        if row["kind"] != "sampled" or row["target"] != "" or row["arm"] not in (later, earlier):
            continue
        seeds = found[_cell_key(row)][row["arm"]]
        seed = int(row["seed"])
        if seed in seeds:
            raise SystemExit(f"{_cell_key(row)}: duplicate {row['arm']} seed {seed}")
        seeds[seed] = (int(row["avoided_prefill_tokens"]), int(row["requested_tokens"]))
    out = {}
    for key, arms in found.items():
        if set(arms.get(later, {})) != set(SEEDS) or set(arms.get(earlier, {})) != set(SEEDS):
            raise SystemExit(f"{key}: sampled seeds are not {list(SEEDS)} for both arms")
        tokens, points = [], []
        for seed in SEEDS:
            (after, requested), (before, requested_before) = arms[later][seed], arms[earlier][seed]
            if requested != requested_before:
                raise SystemExit(f"{key}/s{seed}: requested tokens differ between arms")
            tokens.append(after - before)
            points.append(_points(after - before, requested))
        out[key] = {"tokens": tokens, "points": points, "seeds": list(SEEDS),
                    "requested_tokens": arms[later][SEEDS[0]][1]}
    return out


def protection_outcome(phase1_rows, target: str) -> dict[tuple, dict]:
    """Outcome 3 for one target, seed-paired: `net_avoided_tokens` of the
    `all - none` comparison, checked against its own left and right columns
    and against the published points."""
    left, right, comparison = PHASE1_COMPARISON
    found: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in phase1_rows:
        if (row["comparison"], row["left_variant"], row["right_variant"]) != (comparison, left, right):
            continue
        if row["target"] != target:
            continue
        key, seed = _cell_key(row), int(row["seed"])
        if seed in found[key]:
            raise SystemExit(f"{key}: duplicate {comparison} {target} seed {seed}")
        net = int(row["net_avoided_tokens"])
        if net != int(row["left_avoided_tokens"]) - int(row["right_avoided_tokens"]):
            raise SystemExit(f"{key}/s{seed}: net != left - right")
        requested = int(row["requested_tokens"])
        if abs(_points(net, requested) - float(row["net_input_percentage_points"])) > 1e-9:
            raise SystemExit(f"{key}/s{seed}: published points disagree with the tokens")
        found[key][seed] = {"tokens": net, "requested_tokens": requested}
    out = {}
    for key, seeds in found.items():
        if set(seeds) != set(SEEDS):
            raise SystemExit(f"{key}: {comparison} {target} seeds are {sorted(seeds)}")
        tokens = [seeds[seed]["tokens"] for seed in SEEDS]
        requested = seeds[SEEDS[0]]["requested_tokens"]
        out[key] = {"tokens": tokens, "points": [_points(value, requested) for value in tokens],
                    "seeds": list(SEEDS), "requested_tokens": requested}
    return out


def phase1_inventory(phase1_rows) -> list[dict]:
    """Every (comparison, target) the Phase 1 pairs table carries, with its cells."""
    cells: dict[tuple, set] = defaultdict(set)
    for row in phase1_rows:
        cells[(row["comparison"], row["left_variant"], row["right_variant"], row["target"])].add(
            (row["trace"], row["cell"]))
    return [{"comparison": c, "left_variant": l, "right_variant": r, "target": t,
             "trace_cells": len(members),
             "cells": sorted({cell for _, cell in members})}
            for (c, l, r, t), members in sorted(cells.items())]


def observations(decision_rows, phase1_rows, traces=TRACES) -> list[dict]:
    """One entry per outcome variant: (outcome, target, primary) and its
    published cells inside the grid (the Phase 0.97 table also carries the
    synthetic trace, which is not part of this check)."""
    def grid(cells: dict[tuple, dict]) -> dict[tuple, dict]:
        return {key: value for key, value in cells.items()
                if key[0] in traces and (key[1], key[2]) in CELLS}

    out = [
        {"outcome": "heap_lru_2hit_minus_lru", "target": "", "primary": True,
         "cells": grid(heap_outcome(decision_rows))},
        {"outcome": "sampled_lru_2hit_s_minus_lru_s", "target": "", "primary": True,
         "cells": grid(sampled_outcome(decision_rows))},
        {"outcome": "protection_all_minus_none", "target": PHASE1_PRIMARY_TARGET, "primary": True,
         "cells": grid(protection_outcome(phase1_rows, PHASE1_PRIMARY_TARGET))},
    ]
    for target in PHASE1_SECONDARY_TARGETS:
        out.append({"outcome": "protection_all_minus_none", "target": target, "primary": False,
                    "cells": grid(protection_outcome(phase1_rows, target))})
    return out


# --- derived tables --------------------------------------------------------------------------


def _capacity_base(name: str) -> int:
    return int(rdp._SHARED["working_set"][name])


def working_set_rows(streams) -> list[dict]:
    out = []
    for stream in sorted(streams, key=lambda s: (s["trace"], s["l1_fraction"])):
        summary = stream["working_set"]
        base = _capacity_base(stream["trace"])
        row = {"trace": stream["trace"], "l1_fraction": stream["l1_fraction"],
               "l1_capacity_bytes": stream["l1_capacity_bytes"],
               "capacity_base_bytes": base, "measure_from_ms": stream["split_ms"],
               "offers": summary["offers"], "returning_offers": summary["returning_offers"],
               "never_returning_offers": summary["offers"] - summary["returning_offers"],
               "window_returning_offers": summary["window_returning_offers"],
               "window_returning_tokens": summary["window_returning_tokens"],
               "offered_bytes": stream["offered_bytes"],
               "max_stack_distance_bytes": stream["max_distance_bytes"]}
        for q in QUANTILES:
            value = summary["quantiles"][q]
            row[f"d_plus_size_q{int(round(100 * q)):02d}_bytes"] = value
            row[f"d_plus_size_q{int(round(100 * q)):02d}_of_base"] = value / base
        row["W_bytes"] = summary["quantiles"][0.5]
        row["W_of_base"] = row["W_bytes"] / base
        out.append(row)
    return out


def ratio_rows(streams) -> list[dict]:
    out = []
    for stream in sorted(streams, key=lambda s: (s["trace"], s["l1_fraction"])):
        quantiles = stream["working_set"]["quantiles"]
        for multiplier in MULTIPLIERS:
            capacity = stream["capacities"][multiplier]["l2_capacity_bytes"]
            for value in list(quantiles.values()) + [capacity]:
                if not (math.isfinite(value) and 0 < value < EXACT_LIMIT):
                    raise SystemExit(f"{stream['trace']}: {value} is not a finite positive "
                                     f"integer below 2**53")
            row = {"trace": stream["trace"], "l1_fraction": stream["l1_fraction"],
                   "l2_multiplier": multiplier,
                   "cell": rdp.cell_label(stream["l1_fraction"], multiplier),
                   "l2_capacity_bytes": capacity, "W_bytes": quantiles[0.5],
                   "gamma": ratio(quantiles[0.5], capacity)}
            for q in QUANTILES:
                row[f"gamma_q{int(round(100 * q)):02d}"] = ratio(quantiles[q], capacity)
            out.append(row)
    return out


def _sign_string(values) -> str:
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)


def agreement_rows(variants, ratios) -> list[dict]:
    """Reading 1, row by row: one row per outcome variant x published trace x cell."""
    gamma = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"])): row
             for row in ratios}
    out = []
    for variant in variants:
        spec = OUTCOME[variant["outcome"]]
        for key in sorted(variant["cells"]):
            if key not in gamma:
                raise SystemExit(f"{variant['outcome']}: published cell {key} has no ratio")
            cell = variant["cells"][key]
            ratio_row = gamma[key]
            sign, reading = observation(cell["points"])
            predicted = predicted_sign(ratio_row["gamma"], spec["declining_sign"])
            positive, zero, negative = sign_counts(cell["points"])
            out.append({
                "outcome": variant["outcome"], "number": spec["number"],
                "target": variant["target"], "primary": variant["primary"],
                "trace": key[0], "l1_fraction": key[1], "l2_multiplier": key[2],
                "cell": rdp.cell_label(key[1], key[2]),
                "gamma": ratio_row["gamma"], "gamma_q25": ratio_row["gamma_q25"],
                "gamma_q75": ratio_row["gamma_q75"],
                "in_transition": in_transition(ratio_row["gamma"]),
                "observed_points": float(np.mean(cell["points"])),
                "observed_points_min": float(np.min(cell["points"])),
                "observed_points_max": float(np.max(cell["points"])),
                "observed_tokens_mean": float(np.mean(cell["tokens"])),
                "pairing": "seed-paired" if cell["seeds"] else "deterministic",
                "values": len(cell["points"]),
                "seed_signs": _sign_string(cell["points"]),
                "n_pos": positive, "n_zero": zero, "n_neg": negative,
                "reading": reading, "observed_sign": sign,
                "declining_sign": spec["declining_sign"], "predicted_sign": predicted,
                "match": sign == predicted,
            })
    return out


def _variants(rows) -> list[tuple]:
    seen = []
    for row in rows:
        variant = (row["number"], row["outcome"], row["target"], row["primary"])
        if variant not in seen:
            seen.append(variant)
    return sorted(seen, key=lambda v: (v[0], not v[3], v[2]))


def verdict_rows(rows) -> list[dict]:
    """Readings 1 and 2 per outcome variant.

    `verdict` is the plan's rule, which needs all twelve trace x cells; where
    the published table has fewer it reads "not computable" and names the
    count. `verdict_on_compared_cells` applies the same rule to the cells that
    are published, so the two agree whenever all twelve are.
    """
    out = []
    for number, outcome, target, primary in _variants(rows):
        members = [row for row in rows if (row["outcome"], row["target"]) == (outcome, target)]
        counted = agreement(members)
        misses = [members[index] for index in counted["misses"]]
        gammas = [row["gamma"] for row in misses]
        on_compared = verdict(gammas)
        complete = counted["compared"] == EXPECTED_CELLS
        out.append({
            "outcome": outcome, "number": number, "target": target, "primary": primary,
            "cells_expected": EXPECTED_CELLS, "cells_compared": counted["compared"],
            "agreement": counted["agreement"], "misses": len(misses),
            "misses_in_transition": sum(1 for gamma in gammas if in_transition(gamma)),
            "miss_list": "; ".join(f"{row['trace']}/{row['cell']} gamma={row['gamma']:.4g} "
                                   f"({row['reading']})" for row in misses),
            "verdict": on_compared if complete else
            f"not computable: {counted['compared']} of {EXPECTED_CELLS} cells published",
            "verdict_on_compared_cells": on_compared,
        })
    return out


def threshold_rows(rows) -> list[dict]:
    """Reading 3: every maximising interval of thresholds on gamma."""
    out = []
    for number, outcome, target, primary in _variants(rows):
        members = [row for row in rows if (row["outcome"], row["target"]) == (outcome, target)]
        found = threshold_intervals(members)
        at_one = agreement(members, THRESHOLD)["agreement"]
        for index, interval in enumerate(found["intervals"]):
            out.append({
                "outcome": outcome, "number": number, "target": target, "primary": primary,
                "cells_compared": found["compared"], "max_agreement": found["max_agreement"],
                "agreement_at_1": at_one, "interval": index,
                "threshold_low": interval[0], "threshold_high": interval[1],
                "contains_1": interval_contains(interval, THRESHOLD),
            })
    return out


def rejection_shares(mechanism_rows) -> dict[tuple, dict[str, float]]:
    """Rejections / (admissions + rejections) of each rung under all16; the
    store decrements an admission when the arrival loses its own first round,
    so admissions + rejections is the number of offers."""
    out: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in mechanism_rows:
        if row["mechanism"] != DECLINABLE_MECHANISM or row["rung"] not in DECLINABLE_RUNGS:
            continue
        admissions = float(row["l2_admissions_mean"])
        rejections = float(row["l2_rejections_mean"])
        out[_cell_key(row)][row["rung"]] = share(rejections, admissions + rejections)
    return dict(out)


def declinable_rows(streams, mechanism_rows) -> list[dict]:
    shares = rejection_shares(mechanism_rows)
    out = []
    for stream in sorted(streams, key=lambda s: (s["trace"], s["l1_fraction"])):
        for multiplier in MULTIPLIERS:
            entry = stream["capacities"][multiplier]
            key = (stream["trace"], float(stream["l1_fraction"]), multiplier)
            row = {"trace": stream["trace"], "l1_fraction": stream["l1_fraction"],
                   "l2_multiplier": multiplier,
                   "cell": rdp.cell_label(stream["l1_fraction"], multiplier),
                   "l2_capacity_bytes": entry["l2_capacity_bytes"],
                   "declinable_share": share(entry["window_declinable_bytes"],
                                             entry["window_offered_bytes"]),
                   "never_return_share": share(entry["window_never_return_bytes"],
                                               entry["window_offered_bytes"]),
                   "beyond_capacity_share": share(entry["window_beyond_capacity_bytes"],
                                                  entry["window_offered_bytes"])}
            for rung in DECLINABLE_RUNGS:
                row[f"{rung}_rejection_share_{DECLINABLE_MECHANISM}"] = (
                    shares.get(key, {}).get(rung, math.nan))
            row["declinable_share_offers_whole_trace"] = share(entry["declinable_offers"],
                                                               entry["offers"])
            row["declinable_share_offers_window"] = share(entry["window_declinable_offers"],
                                                          entry["window_offers"])
            for field in ("window_offers", "window_offered_bytes", "window_declinable_offers",
                          "window_declinable_bytes", "window_never_return_bytes",
                          "window_beyond_capacity_bytes", "offers", "declinable_offers"):
                row[field] = entry[field]
            out.append(row)
    return out


# --- figure --------------------------------------------------------------------------------

# The first two slots of the mechanism-control palette; every series also
# differs in marker, and the CSVs are the table view.
TRACE_STYLE = {
    "conversation_trace": ("#2a78d6", "o", "conversation"),
    "toolagent_trace": ("#eb6834", "s", "tool-agent"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axis(axis):
    axis.grid(True, axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=7)


def ratio_figure(directory: Path, rows: list[dict]) -> None:
    """Observed effect against gamma, one panel per outcome; the primary rows
    filled, a secondary target hollow; bars span the seed minimum to maximum."""
    figure, axes = plt.subplots(1, len(OUTCOMES), figsize=(4.2 * len(OUTCOMES), 3.9), squeeze=False)
    for column, spec in enumerate(OUTCOMES):
        axis = axes[0][column]
        members = [row for row in rows if row["outcome"] == spec["outcome"]]
        for name, (color, marker, label) in TRACE_STYLE.items():
            for primary in (True, False):
                points = [row for row in members if row["trace"] == name and row["primary"] == primary]
                if not points:
                    continue
                gammas = np.array([row["gamma"] for row in points])
                means = np.array([row["observed_points"] for row in points])
                # Clamped: with equal seeds the mean can round a hair past the extremes.
                lows = np.maximum(means - np.array([row["observed_points_min"] for row in points]), 0.0)
                highs = np.maximum(np.array([row["observed_points_max"] for row in points]) - means, 0.0)
                target = points[0]["target"]
                text = label if primary else f"{label}, {target} target (secondary)"
                axis.errorbar(gammas, means, yerr=[lows, highs], linestyle="none", marker=marker,
                              markersize=5.5, color=color, ecolor=color, elinewidth=1.0, capsize=2,
                              markerfacecolor=color if primary else "white", markeredgewidth=1.2,
                              label=text)
                if primary:
                    for row in points:
                        axis.annotate(f"{100 * row['l1_fraction']:g}%x{row['l2_multiplier']:g}",
                                      (row["gamma"], row["observed_points"]), fontsize=5.5,
                                      color=MUTED, xytext=(4, 3), textcoords="offset points")
        axis.set_xscale("log")
        # Keep gamma = 1 inside the panel with room on both sides; plain 1-2-5
        # labels, no minor labels (they collide when the span is under a decade).
        gammas = [row["gamma"] for row in members] + [THRESHOLD]
        axis.set_xlim(min(gammas) / 1.6, max(gammas) * 1.6)
        axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.axvline(THRESHOLD, color=INK, linestyle=":", linewidth=1.0)
        axis.axhline(0.0, color=MUTED, linewidth=0.8)
        sign = "+" if spec["declining_sign"] > 0 else "-"
        other = "-" if spec["declining_sign"] > 0 else "+"
        primary_target = next((row["target"] for row in members if row["primary"] and row["target"]), "")
        axis.set_title(f"({spec['number']}) {spec['label']}"
                       + (f" ({primary_target} target)" if primary_target else "") + "\n"
                       f"predicted {other} where gamma <= 1, {sign} where gamma > 1",
                       fontsize=8, color=INK)
        axis.set_xlabel("gamma = W / C_L2 (log)", fontsize=7.5, color=INK)
        if column == 0:
            axis.set_ylabel("observed effect, input-token points", fontsize=7.5, color=INK)
        _style_axis(axis)
    # One legend for the figure, from the panel with the most series (outcome 3
    # carries the secondary target too).
    handles, labels = max((axis.get_legend_handles_labels() for axis in axes[0]),
                          key=lambda pair: len(pair[1]))
    figure.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))),
                  fontsize=7, frameon=False)
    figure.suptitle("Published sign outcomes against the working-set ratio "
                    "(dotted: gamma = 1; bars: seed min-max)", fontsize=9, color=INK)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    figure.savefig(directory / "effect_vs_gamma.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


README_TEXT = """# Working-set ratio check

Pre-registration: `docs/working-set-ratio-plan.md`. For each real trace and L1 fraction the L1 offer stream of heap-LRU L1 is recorded with the Phase 0.97 L1-only call (`run_two_tier` with an L2 of zero bytes and a `victim_hook`; L1 never consults L2, so the stream is the one every L2 arm received, and no L2 result is used). An offer's return is the first later request whose prefix contains its state. `D` is the largest total size, at any instant between the offer and its return, of the newer offers not yet returned; within a timestamp group every return precedes every offer, as in the simulator. Sizes are the simulator's packed sizes (block tokens x 2048 bytes). `W` is the token-weighted (block tokens) lower median of `D + size` over the offers whose return falls in the evaluation window; `gamma = W / C_L2` with the Phase 0.97 capacities. Observed effects are in input-token points (100 x tokens / window requested tokens); seed-paired outcomes are read as consistent gain, consistent loss or mixed. The predicted sign is that of declining writes helping where `gamma > 1` (outcomes 1 and 2 positive, outcome 3 negative) and the opposite where `gamma <= 1`.

- `working_set.csv`: per trace x L1 fraction, the offer counts and the token-weighted quartiles of `D + size` in bytes and as a fraction of the capacity base (the trace's packed working set, of which the L1 and L2 fractions are taken).
- `ratio.csv`: per trace x cell, `C_L2`, `W` and `gamma` at the median and at the quartiles.
- `agreement.csv`: per outcome x published trace x cell, the observed value (mean, seed min and max), the seed signs and reading, the predicted sign and the match. Outcome 3 (`all - none`, Phase 1 B arm) is published at three cells per trace; `next_use` is the primary target and `binary` a secondary row.
- `verdicts.csv`: per outcome, the agreement count, every miss with its gamma, and the verdict; `verdict` needs all twelve trace x cells and reads "not computable" otherwise, `verdict_on_compared_cells` applies the rule to the published cells.
- `threshold.csv`: per outcome, every interval `[low, high)` of thresholds on gamma that maximises agreement, the maximum, the agreement at 1, and whether the interval contains 1.
- `declinable.csv`: per trace x cell, the share of offered bytes in the window whose offer never returns or has `D + size > C_L2` (with its two parts), next to the rejection share `rejections / (admissions + rejections)` of the `label` and `learned` rungs under `all16` from `mechanism_control_001/replay.csv`. Those shares count offers over the whole trace; the count-based whole-trace and window declinable shares are given beside them for a like-for-like reading.
- `effect_vs_gamma.png`: observed effect against gamma (log axis, dotted line at 1) for the three outcomes and both traces.
- `run_config.json`: plan and code commits, source manifest, trace and reference hashes, grid, capacities, checks, timing and memory.
"""


# --- main ------------------------------------------------------------------------------------


def _write(path: Path, rows) -> None:
    rdp._write(path, rows)


def derive_tables(streams, decision_rows, phase1_rows, mechanism_rows) -> dict[str, list[dict]]:
    ratios = ratio_rows(streams)
    rows = agreement_rows(observations(decision_rows, phase1_rows), ratios)
    return {
        "working_set": working_set_rows(streams),
        "ratio": ratios,
        "agreement": rows,
        "verdicts": verdict_rows(rows),
        "threshold": threshold_rows(rows),
        "declinable": declinable_rows(streams, mechanism_rows),
    }


def publish(directory: Path, tables: dict[str, list[dict]], config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("working_set", "ratio", "agreement", "verdicts", "threshold", "declinable"):
        _write(directory / f"{name}.csv", tables[name])
    ratio_figure(directory, tables["agreement"])
    (directory / "README.md").write_text(README_TEXT, encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n",
                                               encoding="utf-8")


def main() -> None:
    args = parse_args()
    clock = time.time()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    status = _git("status", "--porcelain")
    if status:
        raise SystemExit(f"the working tree is not clean:\n{status}")
    differing = sources_differing_from_head()
    if differing:
        raise SystemExit(f"execution source or published input differs from HEAD: {differing}")
    head_start = _git("rev-parse", "HEAD")
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()

    traces, groups, splits, working_sets, trace_files = {}, {}, {}, {}, {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace {trace.name}")
        _, split_ms, _ = horizon_for(trace)
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = split_ms
        working_sets[trace.name] = working_set_bytes(trace)
        trace_files[trace.name] = {"path": str(path.resolve()), "sha256": sha256_path(path)}
        print(f"loaded {trace.name}: requests={len(trace.requests)} split={split_ms:.1f}ms "
              f"working set={working_sets[trace.name]} bytes", flush=True)
    if set(traces) != set(TRACES):
        raise SystemExit(f"traces {sorted(traces)} do not match the grid {sorted(TRACES)}")
    names = tuple(sorted(traces))
    decision_rows = _read(DECISION_SEEDS)
    phase1_rows = _read(PHASE1_PAIRS)
    mechanism_rows = _read(MECHANISM_REPLAY)
    inventory = phase1_inventory(phase1_rows)
    print("  Phase 1 comparisons published:", flush=True)
    for entry in inventory:
        print(f"    {entry['comparison']:18s} target={entry['target']:9s} "
              f"{entry['trace_cells']} trace x cells {entry['cells']}", flush=True)

    # The capacity rule is the Phase 0.97 function itself, reading this dict.
    rdp._SHARED.update(working_set=working_sets)
    SHARED.update(traces=traces, groups=groups, splits=splits)
    available_start = _proc_mib("/proc/meminfo", "MemAvailable")
    tasks = [(name, fraction) for name in names for fraction in FRACTIONS]
    started = time.time()
    streams = compute_streams(tasks, args.workers)
    stream_seconds = time.time() - started
    for stream in streams:
        summary = stream["working_set"]
        print(f"  {stream['trace']}/{stream['l1_fraction']:g}: {summary['offers']} offers, "
              f"{summary['returning_offers']} returning, {summary['window_returning_offers']} "
              f"returning in the window; W={summary['quantiles'][0.5]} bytes "
              f"(record {stream['record_seconds']:.1f}s, D {stream['distance_seconds']:.1f}s, "
              f"peak {stream['worker_peak_rss_mib']:.0f} MiB)", flush=True)

    problems = check_streams(streams, decision_rows, mechanism_rows, phase1_rows)
    for line in problems[:40]:
        print(f"    CHECK {line}", flush=True)
    if problems:
        raise SystemExit(f"{len(problems)} checks failed; nothing derived or published")
    tables = derive_tables(streams, decision_rows, phase1_rows, mechanism_rows)
    for row in tables["verdicts"]:
        print(f"  ({row['number']}) {row['outcome']}{'/' + row['target'] if row['target'] else ''}"
              f"{'' if row['primary'] else ' [secondary]'}: {row['agreement']}/"
              f"{row['cells_compared']} -> {row['verdict']} "
              f"(on compared cells: {row['verdict_on_compared_cells']})", flush=True)
        if row["miss_list"]:
            print(f"      misses: {row['miss_list']}", flush=True)

    # The source and HEAD must not have moved while the streams were computed.
    if source_manifest() != manifest_start:
        raise SystemExit("execution source changed during the run; nothing published")
    if _git("rev-parse", "HEAD") != head_start:
        raise SystemExit("HEAD changed during the run; nothing published")

    config = {
        "phase": "working_set_ratio",
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "source_manifest": manifest_start,
        "trace_files": trace_files,
        "references": {_display(path): sha256_path(path)
                       for path in (DECISION_SEEDS, PHASE1_PAIRS, MECHANISM_REPLAY)},
        "traces": list(names), "l1_fractions": list(FRACTIONS),
        "l2_multipliers": list(MULTIPLIERS), "cells": [list(cell) for cell in CELLS],
        "cells_expected": EXPECTED_CELLS, "seeds": list(SEEDS),
        "l1_policy": rdp.L1_POLICY, "offer_stream": "run_two_tier, L2 of zero bytes, victim_hook",
        "size_model": "packed block tokens x bytes_per_token (HYPERPARAMETERS)",
        "quantiles": list(QUANTILES), "weight": "offer block tokens",
        "threshold": THRESHOLD, "transition": list(TRANSITION),
        "outcomes": [{key: (_display(value) if isinstance(value, Path) else value)
                      for key, value in spec.items()} for spec in OUTCOMES],
        "phase1_comparison": "-".join(PHASE1_COMPARISON[:2]),
        "phase1_primary_target": PHASE1_PRIMARY_TARGET,
        "phase1_secondary_targets": list(PHASE1_SECONDARY_TARGETS),
        "phase1_inventory": inventory,
        "declinable_rungs": list(DECLINABLE_RUNGS), "declinable_mechanism": DECLINABLE_MECHANISM,
        "measure_from_ms": splits, "working_set_bytes": working_sets,
        "capacities": {f"{s['trace']}/{s['l1_fraction']:g}": {
            "l1_capacity_bytes": s["l1_capacity_bytes"],
            **{f"l2x{m:g}": s["capacities"][m]["l2_capacity_bytes"] for m in MULTIPLIERS}}
            for s in streams},
        "checks": {"problems": len(problems),
                   "tier_equivalence_violations": sum(
                       entry["tier_equivalence_violations"]
                       for s in streams for entry in s["capacities"].values()),
                   "returning_offers_checked_per_capacity": {
                       f"{s['trace']}/{s['l1_fraction']:g}": s["working_set"]["returning_offers"]
                       for s in streams}},
        "streams": [{key: value for key, value in s.items() if key not in ("working_set", "capacities")}
                    for s in streams],
        "workers": min(args.workers, len(tasks)), "stream_seconds": stream_seconds,
        "wall_seconds": time.time() - clock,
        "memory": {"rss_unit": "MiB (Linux ru_maxrss KiB / 1024)",
                   "peak_worker_rss_mib": max(s["worker_peak_rss_mib"] for s in streams),
                   "children_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_CHILDREN),
                   "parent_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
                   "mem_available_mib_at_start": available_start},
        "note": "Arithmetic on the L1 offer stream and on published tables. No L2 policy is "
                "run, nothing is fitted and nothing is proposed.",
    }
    publish(args.output_dir, tables, config)
    print(f"written to {args.output_dir}; done in {time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
