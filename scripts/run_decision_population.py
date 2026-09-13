#!/usr/bin/env python3
"""Phase 0.97: does training on the retention-decision population help?

Phases 0.5-0.95 fitted one causal history ranker on the global observed
population and found that its ranking quality falls as the population narrows
towards the states an eviction really chooses between, and that it does not beat
LRU or LFU under a byte budget. Two explanations survive that: the
representation carries no usable signal on the decision population, or the
training distribution was the wrong one. This script separates them by changing
the training population and nothing else.

The setting is the Phase 0.95 two-tier replay: heap-LRU L1 at 0.25 / 1 / 2 % of
the working set, every L1 eviction offered to an exclusive union L2 of 1x and 4x
L1 under the tree hit rule, measured from the Phase 0.5 split. Every arm that
needs a time-varying score runs through one sampled-eviction mechanism (the
arriving victim plus a uniform sample of 16 residents, lowest score leaves, the
arrival may be rejected); the generic policies run through it too, and their
heap versions plus the heap offline comparator are kept as references.

Training populations, same features, target, model, split and embargo:

* A_pd    - the Phase 0.9 global fit (control, per-decision standardisation);
* A_none  - the same global fit on raw features;
* B       - the L1 victim stream of the training window;
* C_lru / C_lfu / C_union - the decision sets logged while a causal behaviour
            policy (sampled L2-LRU, sampled L2-LFU, and their union) runs.

Outputs, written to --output-dir and copied to --paper-dir: the per-fit table
with standardised coefficients, the predictive table (train population x
evaluation population), the replay table per seed and aggregated, the run
configuration, and figures 14 (replay closure) and 15 (population ladder).

Nothing is tuned per trace and no policy is introduced.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.characterize import write_csv
from persistent_kv_admission.crossworkload import (
    HYPERPARAMETERS,
    TARGETS,
    FixedModelScorer,
    fit_fixed_model,
)
from persistent_kv_admission.decisionpop import (
    BEHAVIOUR_POLICIES,
    CANDIDATE_CAP,
    DECISION_CAP,
    MAX_TRAIN_ROWS,
    NEGATIVES_PER_POSITIVE,
    SAMPLE_WIDTH,
    SNAPSHOT_COUNT,
    TRAIN_POPULATIONS,
    CandidateLogger,
    RawFeatureScorer,
    Rows,
    VictimLogger,
    coefficient_row,
    concatenate,
    deduplicate,
    evaluate,
    fit_population_ranker,
    OnPolicyDecisionLogger,
    horizon_for,
    lexicographic_score,
    max_coefficient_deviation,
    observed_rows,
    population_metrics,
    scoring_for,
    state_indices,
    union_across_logs,
)
from persistent_kv_admission.gap import summarize, working_set_bytes
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

DEFAULT_L1_BUDGETS = (0.0025, 0.01, 0.02)
DEFAULT_MULTIPLIERS = (1, 4)
DEFAULT_SEEDS = 5
L1_POLICY = "lru"
HIT_MODEL = "tree"
CLOSURE = "union"
# Heap references, one deterministic run each.
HEAP_ARMS = ("lru", "lfu", "lru_2hit", "offline_next_use")
# Generic policies through the sampled mechanism, one run per seed.
GENERIC_SAMPLED = ("lru", "lfu", "lru_2hit")
# The closure floor and ceiling, fixed before the run.
CLOSURE_FLOOR = "lru_s"
CLOSURE_CEILING = "offline_next_use"
# Behaviour policies whose decision sets become training populations.
C_VARIANTS = {"C_lru": ("lru",), "C_lfu": ("lfu",), "C_union": ("lru", "lfu")}
# The cell the population-ladder figure is drawn at, and the smoke cell.
LADDER_BUDGET, LADDER_MULTIPLIER = 0.01, 4
# On-policy decisions are logged for one seed only: the point is what each arm's
# own stream looks like, not its seed dispersion, and every arm pays for it.
ONPOLICY_SEED = 0
# The Phase 0.9 fit table the A_pd control must reproduce exactly.
PHASE09_CONFIG = Path("results/paper/target_change_config.json")
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", type=Path, help="trace files (required unless --figure-only)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision_population"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--l1-budgets", default=",".join(str(v) for v in DEFAULT_L1_BUDGETS))
    parser.add_argument("--l2-multipliers", default=",".join(str(v) for v in DEFAULT_MULTIPLIERS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figures from the CSVs in <output-dir> without replaying")
    parser.add_argument("--smoke", action="store_true",
                        help="one trace, the 1%% x 4x cell, one seed, every arm: a timing probe")
    return parser.parse_args()


def _write(path: Path, rows) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fields} for row in rows], fields)


def _capacity(name: str, fraction: float) -> int:
    return max(1, round(_SHARED["working_set"][name] * fraction))


def cell_label(fraction: float, multiplier: float | None = None) -> str:
    if multiplier is None:
        return f"l1={fraction:g}"
    return f"l1={fraction:g},l2x{multiplier:g}"


def model_cell(population: str, fraction: float, multiplier: float) -> str:
    if population in ("A_pd", "A_none"):
        return ""
    if population == "B":
        return cell_label(fraction)
    return cell_label(fraction, multiplier)


def ranker_key(name: str, target: str, population: str, fraction: float, multiplier: float):
    return (name, target, population, model_cell(population, fraction, multiplier))


def victim_path(name: str, fraction: float) -> Path:
    return _SHARED["logs_dir"] / f"victims_{name}_{fraction:g}.npz"


def candidate_path(name: str, fraction: float, multiplier: float, policy: str) -> Path:
    return _SHARED["logs_dir"] / f"candidates_{name}_{fraction:g}_x{multiplier:g}_{policy}.npz"


# --- stage workers ------------------------------------------------------------


def _a_fit_worker(task):
    """The Phase 0.9 global fit, in its own convention and in the raw one."""
    name, target, standardization = task
    trace = _SHARED["traces"][name]
    started = time.time()
    ranker, horizon, split_ms, positives, rows = fit_fixed_model(
        trace, 600.0, snapshot_count=SNAPSHOT_COUNT, candidate_cap=CANDIDATE_CAP,
        max_train_rows=MAX_TRAIN_ROWS, negatives_per_positive=NEGATIVES_PER_POSITIVE,
        train_fraction=HYPERPARAMETERS["train_fraction"], seed=0,
        standardization=standardization, target=target,
    )
    population = "A_pd" if standardization == "per_decision" else "A_none"
    return {
        "trace": name, "target": target, "population": population, "cell": "",
        "standardization": standardization, "ranker_standardize": bool(ranker.standardize),
        "horizon_seconds": float(horizon), "split_ms": float(split_ms),
        "train_rows": int(rows), "train_positives": int(positives),
        "seconds": time.time() - started, "ranker": ranker,
    }


def _victim_worker(task):
    """One L1 victim stream (L2 of zero bytes) plus the B fits on its training half."""
    name, fraction = task
    trace = _SHARED["traces"][name]
    horizon, split_ms = _SHARED["horizons"][name], _SHARED["splits"][name]
    started = time.time()
    logger = VictimLogger(trace, horizon, state_index=_SHARED["state_index"][name])
    result = run_two_tier(
        trace, L1_POLICY, _capacity(name, fraction), 0, occurrence_groups=_SHARED["groups"][name],
        measure_from_ms=split_ms, observer=logger, victim_hook=logger,
    )
    rows = logger.rows()
    rows.save(victim_path(name, fraction))
    logged = time.time() - started
    train = rows.select(rows.train_mask(split_ms))
    fits = []
    for target in _SHARED["targets"]:
        began = time.time()
        ranker, positives, used = fit_population_ranker(train, target, seed=0)
        fits.append({
            "trace": name, "target": target, "population": "B", "cell": cell_label(fraction),
            "standardization": "ranker", "ranker_standardize": bool(ranker.standardize),
            "horizon_seconds": horizon, "split_ms": split_ms, "train_rows": int(used),
            "train_positives": int(positives), "seconds": time.time() - began,
            "l1_fraction": fraction, "population_rows": int(len(rows)),
            "train_pool_rows": int(len(train)), "ranker": ranker,
        })
    return {
        "task": task, "fits": fits, "seconds": logged,
        "baseline": {"trace": name, "l1_fraction": fraction,
                     "l1_capacity_bytes": result.l1_capacity_bytes,
                     "l1_avoided_tokens": result.l1_avoided_tokens,
                     "requested_tokens": result.requested_tokens,
                     "l1_evictions": result.l1_evictions,
                     "victim_rows": int(len(rows))},
    }


def _candidate_worker(task):
    """One behaviour-policy replay whose sampled L2 decisions are logged."""
    name, fraction, multiplier, policy = task
    trace = _SHARED["traces"][name]
    horizon, split_ms = _SHARED["horizons"][name], _SHARED["splits"][name]
    started = time.time()
    logger = CandidateLogger(trace, horizon, max_decisions=DECISION_CAP, seed=0,
                             state_index=_SHARED["state_index"][name])
    result = run_two_tier(
        trace, L1_POLICY, _capacity(name, fraction), _capacity(name, fraction * multiplier),
        policy, hit_model=HIT_MODEL, closure=CLOSURE, occurrence_groups=_SHARED["groups"][name],
        measure_from_ms=split_ms, l2_eviction="sampled", l2_sample_width=SAMPLE_WIDTH,
        l2_seed=0, l2_arm=f"{policy}_s", observer=logger, l2_decision_hook=logger,
    )
    rows = logger.rows()
    rows.save(candidate_path(name, fraction, multiplier, policy))
    return {
        "task": task, "seconds": time.time() - started,
        "meta": {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                 "behaviour_policy": policy, "decisions_seen": logger.decisions_seen,
                 "decisions_kept": len(logger.kept), "rows": int(len(rows)),
                 "l2_decisions": result.l2_decisions, "l2_admissions": result.l2_admissions,
                 "l2_rejections": result.l2_rejections, "l2_evictions": result.l2_evictions},
    }


def _c_fit_worker(task):
    """The three candidate-population fits of one cell, for every target."""
    name, fraction, multiplier = task
    split_ms = _SHARED["splits"][name]
    horizon = _SHARED["horizons"][name]
    logs = {}
    for policy in BEHAVIOUR_POLICIES:
        rows = Rows.load(candidate_path(name, fraction, multiplier, policy))
        logs[policy] = rows.select(rows.train_mask(split_ms))
    fits = []
    for population, policies in C_VARIANTS.items():
        if len(policies) == 1:
            train, dropped = logs[policies[0]], 0
        else:
            # Cross-log only: the second arm's copy of a (t, state) the first
            # already supplies is dropped; repeats inside one log stay, because
            # they are separate decisions the state really faced.
            train, dropped = union_across_logs([logs[p] for p in policies])
        # Measured, never applied: how often one log repeats a (t, state).
        repeats = sum(len(logs[p]) - len(deduplicate(logs[p])[0]) for p in policies)
        for target in _SHARED["targets"]:
            began = time.time()
            ranker, positives, used = fit_population_ranker(train, target, seed=0)
            fits.append({
                "trace": name, "target": target, "population": population,
                "cell": cell_label(fraction, multiplier), "standardization": "ranker",
                "ranker_standardize": bool(ranker.standardize), "horizon_seconds": horizon,
                "split_ms": split_ms, "train_rows": int(used), "train_positives": int(positives),
                "seconds": time.time() - began, "l1_fraction": fraction,
                "l2_multiplier": multiplier, "train_pool_rows": int(len(train)),
                "duplicate_rows_dropped": int(dropped),
                "repeated_time_state_rows": int(repeats),
                "rows_before_dedup": int(len(train) + dropped),
                "train_decisions": int(len(np.unique(train.group))) if len(train) else 0,
                "ranker": ranker,
            })
    return {"task": task, "fits": fits}


def _observed_worker(name):
    """The Phase 0.5 shared test snapshots of one trace, as one grouped population."""
    trace = _SHARED["traces"][name]
    started = time.time()
    rows = observed_rows(trace, _SHARED["horizons"][name], _SHARED["snapshots"][name],
                         candidate_cap=CANDIDATE_CAP, seed=0,
                         state_index=_SHARED["state_index"][name])
    return name, rows, time.time() - started


def _predictive_worker(task):
    """Every fitted ranker of one cell, scored on every population of that cell."""
    name, fraction, multiplier = task
    trace = _SHARED["traces"][name]
    split_ms, horizon = _SHARED["splits"][name], _SHARED["horizons"][name]
    started = time.time()
    victims = Rows.load(victim_path(name, fraction))
    victims = victims.select(victims.test_mask(split_ms, trace.end_ms))
    cell = cell_label(fraction, multiplier)
    populations = [("observed", "", "test", _SHARED["observed"][name], False, None),
                   ("victims", cell_label(fraction), "test", victims, False, None)]
    for policy in BEHAVIOUR_POLICIES:
        logged = Rows.load(candidate_path(name, fraction, multiplier, policy))
        # The same log supplies both splits, so a model that cannot rank its
        # own training decisions is visible next to its test number.
        populations.append((f"candidates_{policy}", cell, "test",
                            logged.select(logged.test_mask(split_ms, trace.end_ms)),
                            True, f"{policy}_key"))
        populations.append((f"candidates_{policy}_train", cell, "train",
                            logged.select(logged.train_mask(split_ms)), True, f"{policy}_key"))
    out = []
    for population_name, population_cell, split, rows, decision_sets, key_model in populations:
        if len(rows) == 0:
            continue
        shared = {"trace": name, "population": population_name,
                  "population_cell": population_cell, "split": split,
                  "horizon_seconds": horizon}
        for target in _SHARED["targets"]:
            metric = "auc" if target == "binary" else "spearman"
            for fitted_on in TRAIN_POPULATIONS:
                key = ranker_key(name, target, fitted_on, fraction, multiplier)
                ranker = _SHARED["rankers"].get(key)
                if ranker is None:
                    continue
                scoring = scoring_for(fitted_on, rows)
                out.append({
                    **shared, "target": target, "model": fitted_on,
                    "model_cell": model_cell(fitted_on, fraction, multiplier),
                    "scoring": scoring, "metric": metric,
                    **evaluate(ranker, rows, target, scoring, decision_sets),
                })
            if key_model is not None:
                # The behaviour policy's own ordering on its own decisions, from
                # the score tuples the store really compared (recorded, not
                # recomputed), so the learned arms have a like-for-like floor.
                scores = lexicographic_score(rows.arm_score, rows.arm_tiebreak)
                out.append({
                    **shared, "target": target, "model": key_model, "model_cell": cell,
                    "scoring": "recorded_arm_key", "metric": metric,
                    **population_metrics(scores, rows, target, decision_sets),
                })
    return {"task": task, "rows": out, "seconds": time.time() - started}


def _replay_worker(task):
    """One two-tier replay of one arm, with its own decisions logged at seed 0."""
    name, fraction, multiplier, kind, arm, target, seed = task
    trace = _SHARED["traces"][name]
    split_ms = _SHARED["splits"][name]
    started = time.time()
    shared = dict(hit_model=HIT_MODEL, closure=CLOSURE, occurrence_groups=_SHARED["groups"][name],
                  measure_from_ms=split_ms)
    l1_bytes = _capacity(name, fraction)
    l2_bytes = _capacity(name, fraction * multiplier)
    # Every arm that evicts by score gets its own decisions recorded once, so
    # its accuracy can be read on the stream it actually produced and not only
    # on the shared behaviour-policy logs.
    logger = None
    if kind in ("sampled", "learned") and seed == ONPOLICY_SEED:
        logger = OnPolicyDecisionLogger(
            trace, _SHARED["horizons"][name], split_ms, trace.end_ms,
            max_decisions=DECISION_CAP, seed=0, state_index=_SHARED["state_index"][name],
        )
    if kind == "heap":
        result = run_two_tier(trace, L1_POLICY, l1_bytes, l2_bytes, arm, l2_eviction="heap",
                              l2_arm=arm, **shared)
    elif kind == "sampled":
        result = run_two_tier(trace, L1_POLICY, l1_bytes, l2_bytes, arm, l2_eviction="sampled",
                              l2_sample_width=SAMPLE_WIDTH, l2_seed=seed, l2_arm=f"{arm}_s",
                              l2_decision_hook=logger, **shared)
    else:
        ranker = _SHARED["rankers"][ranker_key(name, target, arm, fraction, multiplier)]
        if arm == "A_pd":
            scorer = FixedModelScorer(
                trace, ranker, refresh_every=HYPERPARAMETERS["normalizer_refresh_every"],
                sample_size=HYPERPARAMETERS["normalizer_sample"], seed=seed,
                normalize_on="cached",
            )
        else:
            scorer = RawFeatureScorer(trace, ranker)
        result = run_two_tier(trace, L1_POLICY, l1_bytes, l2_bytes, "learned",
                              l2_eviction="sampled", l2_sample_width=SAMPLE_WIDTH, l2_seed=seed,
                              l2_scorer=scorer, l2_arm=arm, l2_decision_hook=logger, **shared)
    row = result.as_row()
    row.update(l1_fraction=fraction, l2_multiplier=multiplier, l2_fraction=fraction * multiplier,
               total_fraction=fraction * (1 + multiplier), kind=kind, arm=result.l2_arm,
               target=target, seed=seed, seconds=time.time() - started)
    onpolicy = []
    if logger is not None:
        for label_target in _SHARED["targets"]:
            onpolicy.append({
                "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                "cell": cell_label(fraction, multiplier), "arm": result.l2_arm, "kind": kind,
                "arm_target": target, "label_target": label_target,
                "own_target": bool(target) and label_target == target,
                "metric": "auc" if label_target == "binary" else "spearman",
                "seed": seed, "horizon_seconds": _SHARED["horizons"][name],
                "l2_decisions": result.l2_decisions,
                **logger.metrics(label_target),
            })
    return {"row": row, "onpolicy": onpolicy}


# --- derivation ---------------------------------------------------------------


def derive(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    """Per-seed extra tokens, input-token share, and headroom closure.

    Closure is measured per seed against that seed's own sampled L2-LRU, with
    the heap offline comparator as the ceiling, which is the Phase 0.5
    definition transported to the two-tier setting. Deterministic heap arms are
    replicated once per seed so that every arm has a closure for every seed.
    """
    expanded: list[dict] = []
    for row in rows:
        if row["kind"] == "heap":
            for seed in seeds:
                expanded.append(dict(row, seed=seed))
        else:
            expanded.append(row)
    by_key = {(r["trace"], float(r["l1_fraction"]), float(r["l2_multiplier"]), int(r["seed"]),
               r["arm"], r["target"]): r for r in expanded}
    for row in expanded:
        cell = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]), int(row["seed"]))
        row["extra_avoided_tokens"] = int(row["avoided_prefill_tokens"]) - int(row["l1_avoided_tokens"])
        row["extra_fraction_of_input"] = row["extra_avoided_tokens"] / max(int(row["requested_tokens"]), 1)
        floor = by_key.get(cell + (CLOSURE_FLOOR, ""))
        ceiling = by_key.get(cell + (CLOSURE_CEILING, ""))
        if floor is None or ceiling is None:
            row["headroom_closure"] = math.nan
            row["gain_vs_lru_s_fraction"] = math.nan
            continue
        reference = int(floor["avoided_prefill_tokens"])
        headroom = int(ceiling["avoided_prefill_tokens"]) - reference
        gain = int(row["avoided_prefill_tokens"]) - reference
        row["gain_vs_lru_s_fraction"] = gain / max(reference, 1)
        row["headroom_closure"] = gain / headroom if headroom > 0 else math.nan
    return expanded


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                row["arm"], row["target"], row["kind"])].append(row)
    out = []
    for (name, fraction, multiplier, arm, target, kind), members in sorted(groups.items()):
        entry: dict[str, object] = {
            "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier, "arm": arm,
            "target": target, "kind": kind, "cell": cell_label(fraction, multiplier),
            "seeds": len(members),
            "l1_avoided_tokens": int(members[0]["l1_avoided_tokens"]),
            "requested_tokens": int(members[0]["requested_tokens"]),
        }
        for metric in ("avoided_prefill_tokens", "extra_avoided_tokens", "extra_fraction_of_input",
                       "headroom_closure", "gain_vs_lru_s_fraction", "l2_hit_blocks",
                       "l2_admissions", "l2_rejections", "l2_evictions", "seconds"):
            stats = summarize([float(member[metric]) for member in members])
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


def phase09_reference() -> dict[tuple[str, str], dict]:
    """Stored Phase 0.9 fits keyed by (trace, f"{target}_h{horizon}")."""
    if not PHASE09_CONFIG.exists():
        return {}
    config = json.loads(PHASE09_CONFIG.read_text(encoding="utf-8"))
    out = {}
    for name, fits in config.get("fits", {}).items():
        for fit in fits:
            out[(name, f"{fit['target']}_h{int(fit['horizon'])}")] = fit
    return out


# --- figures ------------------------------------------------------------------


def _f(row, key) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


LEARNED_STYLE = (
    ("A_pd", "A: global, per-decision norm (Phase 0.9 control)", "tab:blue", "-"),
    ("A_none", "A: global, raw features", "tab:cyan", "--"),
    ("B", "B: L1 victims", "tab:orange", "-"),
    ("C_lru", "C: L2 candidates under sampled LRU", "tab:green", "-"),
    ("C_lfu", "C: L2 candidates under sampled LFU", "tab:red", "-"),
    ("C_union", "C: candidates, both behaviour policies", "tab:purple", "-"),
)
FLAT_STYLE = (("lfu_s", "sampled LFU", "0.35", "s"), ("lru_2hit_s", "sampled 2-hit LRU", "0.6", "^"))


# Which column of the predictive table each evaluation population is read from.
# A population with real decision points is read at the decision level, because
# that is the ranking a retention decision performs; a population without one
# can only be read pooled.
LADDER_COLUMNS = (
    ("observed", "pooled_metric", "observed\n(pooled)"),
    ("victims", "pooled_metric", "victims\n(pooled)"),
    ("candidates_lru", "within_decision_macro", "candidates_lru\n(within-decision)"),
    ("candidates_lfu", "within_decision_macro", "candidates_lfu\n(within-decision)"),
)


def make_figures(paper_dir: Path, replay: list[dict], predictive: list[dict],
                 onpolicy: list[dict], synthetic_names: set[str]) -> None:
    traces = sorted({r["trace"] for r in replay} - synthetic_names)
    targets = [t for t in TARGETS if any(r["target"] == t for r in replay)]
    if not traces or not targets:
        print("  no real-trace rows: figures skipped", flush=True)
        return
    cells = sorted({(_f(r, "l1_fraction"), _f(r, "l2_multiplier")) for r in replay})
    positions = np.arange(len(cells))
    labels = [f"{100 * f:g}%\nx{m:g}" for f, m in cells]

    figure, axes = plt.subplots(len(targets), len(traces),
                                figsize=(5.4 * len(traces), 3.6 * len(targets)), squeeze=False)
    for row_index, target in enumerate(targets):
        for column, name in enumerate(traces):
            axis = axes[row_index][column]
            for arm, label, color, style in LEARNED_STYLE:
                points, errors = [], []
                for cell in cells:
                    match = [r for r in replay if r["trace"] == name and r["arm"] == arm
                             and r["target"] == target and (_f(r, "l1_fraction"), _f(r, "l2_multiplier")) == cell]
                    points.append(_f(match[0], "headroom_closure_mean") if match else math.nan)
                    errors.append(_f(match[0], "headroom_closure_ci95_half") if match else math.nan)
                errors = [0.0 if math.isnan(e) else e for e in errors]
                axis.errorbar(positions, points, yerr=errors, marker="o", ms=3.5, color=color,
                              linestyle=style, label=label, capsize=2)
            for arm, label, color, marker in FLAT_STYLE:
                points = []
                for cell in cells:
                    match = [r for r in replay if r["trace"] == name and r["arm"] == arm
                             and (_f(r, "l1_fraction"), _f(r, "l2_multiplier")) == cell]
                    points.append(_f(match[0], "headroom_closure_mean") if match else math.nan)
                axis.plot(positions, points, linestyle="none", marker=marker, ms=5, color=color,
                          label=label)
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.axhline(1.0, color="0.5", linewidth=0.8)
            axis.set(xticks=positions, xticklabels=labels,
                     title=f"{name} — target: {target}",
                     xlabel="L1 budget x L2 multiplier",
                     ylabel="headroom closure vs sampled L2-LRU")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=5.5)
    figure.suptitle("L2 retention utility by training population")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig14_decision_population.png", dpi=170)
    plt.close(figure)

    # Figure 15: train population x evaluation population, one cell of the grid.
    ladder_cell = cell_label(LADDER_BUDGET, LADDER_MULTIPLIER)
    wanted = [r for r in predictive
              if r["population_cell"] in ("", cell_label(LADDER_BUDGET), ladder_cell)]
    evaluation = [column[0] for column in LADDER_COLUMNS]
    figure, axes = plt.subplots(len(targets), len(traces),
                                figsize=(5.0 * len(traces), 3.8 * len(targets)), squeeze=False)
    for row_index, target in enumerate(targets):
        for column, name in enumerate(traces):
            axis = axes[row_index][column]
            grid = np.full((len(TRAIN_POPULATIONS), len(evaluation)), math.nan)
            for i, model in enumerate(TRAIN_POPULATIONS):
                for j, (population, field, _label) in enumerate(LADDER_COLUMNS):
                    match = [r for r in wanted if r["trace"] == name and r["target"] == target
                             and r["model"] == model and r["population"] == population
                             and r.get("split", "test") == "test"
                             and r["model_cell"] in ("", cell_label(LADDER_BUDGET), ladder_cell)]
                    if match:
                        grid[i, j] = _f(match[0], field)
            finite = grid[np.isfinite(grid)]
            span = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
            image = axis.imshow(grid, cmap="viridis", aspect="auto", vmin=span[0], vmax=span[1])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if math.isfinite(grid[i, j]):
                        middle = 0.5 * (span[0] + span[1])
                        axis.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7,
                                  color="white" if grid[i, j] < middle else "black")
            axis.set(xticks=range(len(evaluation)), yticks=range(len(TRAIN_POPULATIONS)),
                     xticklabels=[column[2] for column in LADDER_COLUMNS],
                     yticklabels=list(TRAIN_POPULATIONS),
                     xlabel="evaluation population (metric)", ylabel="training population",
                     title=f"{name} — {target} ({'AUC' if target == 'binary' else 'Spearman'})")
            axis.tick_params(axis="x", labelrotation=25, labelsize=6)
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(f"Predictive quality: training x evaluation population ({ladder_cell}), "
                    "test split")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig15_population_ladder.png", dpi=170)
    plt.close(figure)

    # Figure 16: what each arm got right on the decisions it actually faced.
    if not onpolicy:
        return
    figure, axes = plt.subplots(len(targets), len(traces),
                                figsize=(5.4 * len(traces), 3.6 * len(targets)), squeeze=False)
    for row_index, target in enumerate(targets):
        for column, name in enumerate(traces):
            axis = axes[row_index][column]
            for arm, label, color, style in LEARNED_STYLE:
                points = []
                for cell in cells:
                    match = [r for r in onpolicy if r["trace"] == name and r["arm"] == arm
                             and r["label_target"] == target and r["arm_target"] == target
                             and (_f(r, "l1_fraction"), _f(r, "l2_multiplier")) == cell]
                    points.append(_f(match[0], "within_decision_macro") if match else math.nan)
                axis.plot(positions, points, style, marker="o", ms=3.5, color=color, label=label)
            for arm, label, color, marker in (("lru_s", "sampled LRU", "0.1", "o"),) + FLAT_STYLE:
                points = []
                for cell in cells:
                    match = [r for r in onpolicy if r["trace"] == name and r["arm"] == arm
                             and r["label_target"] == target
                             and (_f(r, "l1_fraction"), _f(r, "l2_multiplier")) == cell]
                    points.append(_f(match[0], "within_decision_macro") if match else math.nan)
                axis.plot(positions, points, linestyle="none", marker=marker, ms=5, color=color,
                          label=label)
            axis.axhline(0.5 if target == "binary" else 0.0, color="0.5", linewidth=0.8)
            axis.set(xticks=positions, xticklabels=labels,
                     title=f"{name} — label: {target}",
                     xlabel="L1 budget x L2 multiplier",
                     ylabel=("within-decision AUC (macro)" if target == "binary"
                             else "mean within-decision Spearman"))
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=5.5)
    figure.suptitle("On-policy decisions: each arm scored on the stream it produced (seed 0)")
    figure.tight_layout()
    figure.savefig(paper_dir / "fig16_onpolicy_decisions.png", dpi=170)
    plt.close(figure)


# --- main ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        load = lambda name: list(csv.DictReader((args.output_dir / name).open(encoding="utf-8")))
        replay = load("decision_population_replay.csv")
        predictive = load("decision_population_predictive.csv")
        onpolicy = load("decision_population_onpolicy.csv")
        synthetic = {r["trace"] for r in replay if "synthetic" in r["trace"]}
        make_figures(args.paper_dir, replay, predictive, onpolicy, synthetic)
        print("figures redrawn", flush=True)
        return
    if not args.traces:
        raise SystemExit("at least one trace file is required unless --figure-only is given")

    paths = list(args.traces)
    l1_budgets = tuple(float(v) for v in args.l1_budgets.split(","))
    multipliers = tuple(float(v) for v in args.l2_multipliers.split(","))
    targets = tuple(v for v in args.targets.split(",") if v)
    seeds = tuple(range(args.seeds))
    if args.smoke:
        paths, l1_budgets, multipliers, seeds = paths[:1], (LADDER_BUDGET,), (float(LADDER_MULTIPLIER),), (0,)
        print("smoke run: one trace, one cell, one seed, every arm", flush=True)
    for target in targets:
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target!r}")
    context = mp.get_context("fork")
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    traces, groups, splits, horizons, snapshots, working_set, indices = {}, {}, {}, {}, {}, {}, {}
    for path in paths:
        trace = load_mooncake_trace(path, 512)
        horizon, split_ms, test_snapshots = horizon_for(trace)
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        horizons[trace.name] = horizon
        splits[trace.name] = split_ms
        snapshots[trace.name] = test_snapshots
        working_set[trace.name] = working_set_bytes(trace)
        indices[trace.name] = state_indices(trace)
        print(f"loaded {trace.name}: H={horizon:g}s split={split_ms:.0f}ms "
              f"snapshots={len(test_snapshots)}", flush=True)
    _SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                   snapshots=snapshots, working_set=working_set, state_index=indices,
                   targets=targets, logs_dir=logs_dir)

    rankers: dict[tuple, object] = {}
    fit_rows: list[dict] = []
    reference = phase09_reference()
    clock = time.time()

    # 1. The global control fits (A_pd, A_none).
    tasks = [(name, target, standardization) for name in traces for target in targets
             for standardization in ("per_decision", "ranker")]
    started = time.time()
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for fit in pool.map(_a_fit_worker, tasks):
            _register(rankers, fit_rows, fit, reference)
    print(f"  {len(tasks)} global fits in {time.time()-started:.0f}s", flush=True)

    # 2. Victim streams (one per trace x L1 budget) and the B fits on them.
    tasks = [(name, fraction) for name in traces for fraction in l1_budgets]
    started = time.time()
    baselines = []
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for out in pool.map(_victim_worker, tasks):
            baselines.append(out["baseline"])
            for fit in out["fits"]:
                _register(rankers, fit_rows, fit, reference)
            print(f"    victims {out['task']}: {out['baseline']['victim_rows']} events "
                  f"in {out['seconds']:.0f}s", flush=True)
    print(f"  {len(tasks)} victim streams in {time.time()-started:.0f}s", flush=True)

    # 3. Behaviour-policy replays whose L2 decision sets are logged.
    tasks = [(name, fraction, multiplier, policy) for name in traces for fraction in l1_budgets
             for multiplier in multipliers for policy in BEHAVIOUR_POLICIES]
    started = time.time()
    candidate_meta = []
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for out in pool.map(_candidate_worker, tasks):
            candidate_meta.append(dict(out["meta"], seconds=out["seconds"]))
            print(f"    candidates {out['task']}: {out['meta']['decisions_seen']} decisions, "
                  f"{out['meta']['decisions_kept']} kept in {out['seconds']:.0f}s", flush=True)
    print(f"  {len(tasks)} decision logs in {time.time()-started:.0f}s", flush=True)

    # 4. The candidate-population fits, one worker per cell.
    tasks = [(name, fraction, multiplier) for name in traces for fraction in l1_budgets
             for multiplier in multipliers]
    started = time.time()
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for out in pool.map(_c_fit_worker, tasks):
            for fit in out["fits"]:
                _register(rankers, fit_rows, fit, reference)
    print(f"  {len(tasks) * len(C_VARIANTS) * len(targets)} candidate fits in "
          f"{time.time()-started:.0f}s", flush=True)
    _SHARED["rankers"] = rankers

    # 5. The observed test-snapshot population of each trace (shared by fork).
    started = time.time()
    observed = {}
    with context.Pool(min(args.workers, len(traces))) as pool:
        for name, rows, seconds in pool.map(_observed_worker, list(traces)):
            observed[name] = rows
            print(f"    observed {name}: {len(rows)} rows in {seconds:.0f}s", flush=True)
    _SHARED["observed"] = observed
    print(f"  observed populations in {time.time()-started:.0f}s", flush=True)

    # 6. Predictive evaluation: every fit x every population of every cell.
    tasks = [(name, fraction, multiplier) for name in traces for fraction in l1_budgets
             for multiplier in multipliers]
    started = time.time()
    predictive: dict[tuple, dict] = {}
    with context.Pool(min(args.workers, 8, len(tasks))) as pool:
        for out in pool.map(_predictive_worker, tasks):
            for row in out["rows"]:
                predictive[(row["trace"], row["target"], row["model"], row["model_cell"],
                            row["population"], row["population_cell"])] = row
    predictive_rows = [predictive[key] for key in sorted(predictive)]
    print(f"  {len(predictive_rows)} predictive rows in {time.time()-started:.0f}s", flush=True)

    # 7. The replay grid.
    tasks = []
    for name in traces:
        for fraction in l1_budgets:
            for multiplier in multipliers:
                for arm in HEAP_ARMS:
                    tasks.append((name, fraction, multiplier, "heap", arm, "", 0))
                for arm in GENERIC_SAMPLED:
                    for seed in seeds:
                        tasks.append((name, fraction, multiplier, "sampled", arm, "", seed))
                for target in targets:
                    for arm in TRAIN_POPULATIONS:
                        for seed in seeds:
                            tasks.append((name, fraction, multiplier, "learned", arm, target, seed))
    print(f"  {len(tasks)} replays on {args.workers} workers", flush=True)
    started = time.time()
    raw_rows: list[dict] = []
    raw_path = args.output_dir / "raw_replay.jsonl"
    onpolicy_rows: list[dict] = []
    with context.Pool(args.workers) as pool, raw_path.open("w", encoding="utf-8") as raw:
        for done, out in enumerate(pool.imap_unordered(_replay_worker, tasks, chunksize=1), 1):
            row = out["row"]
            onpolicy_rows.extend(out["onpolicy"])
            raw_rows.append(row)
            raw.write(json.dumps(row, default=str) + "\n")
            if done % 25 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {time.time()-started:.0f}s "
                      f"last={row['trace']}/{row['arm']}/{row['target']}/s{row['seed']} "
                      f"({row['seconds']:.0f}s)", flush=True)
    replay_seconds = time.time() - started
    print(f"  replays in {replay_seconds:.0f}s", flush=True)

    seed_rows = derive(raw_rows, seeds)
    seed_rows.sort(key=lambda r: (r["trace"], float(r["l1_fraction"]), float(r["l2_multiplier"]),
                                  r["target"], r["arm"], int(r["seed"])))
    summary = aggregate(seed_rows)
    fit_rows.sort(key=lambda r: (r["trace"], r["target"], r["population"], r["cell"]))

    timing = defaultdict(list)
    for row in raw_rows:
        timing[(row["kind"], row["arm"])].append(float(row["seconds"]))
    print("  seconds per replay, by arm:", flush=True)
    for (kind, arm), values in sorted(timing.items()):
        print(f"    {kind:8s} {arm:12s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)

    onpolicy_rows.sort(key=lambda r: (r["trace"], float(r["l1_fraction"]),
                                      float(r["l2_multiplier"]), r["arm"], r["arm_target"],
                                      r["label_target"]))
    for directory in (args.output_dir, args.paper_dir):
        _write(directory / "decision_population_fits.csv", fit_rows)
        _write(directory / "decision_population_predictive.csv", predictive_rows)
        _write(directory / "decision_population_replay.csv", summary)
        _write(directory / "decision_population_replay_seeds.csv", seed_rows)
        _write(directory / "decision_population_onpolicy.csv", onpolicy_rows)
    _write(args.output_dir / "decision_population_logs.csv", candidate_meta)
    _write(args.output_dir / "decision_population_l1_baseline.csv", baselines)

    config = {
        "onpolicy_seed": ONPOLICY_SEED,
        "l1_policy": L1_POLICY, "l1_budgets": l1_budgets, "l2_multipliers": multipliers,
        "seeds": list(seeds), "targets": list(targets), "hit_model": HIT_MODEL, "closure": CLOSURE,
        "sample_width": SAMPLE_WIDTH, "train_populations": list(TRAIN_POPULATIONS),
        "heap_arms": list(HEAP_ARMS), "generic_sampled_arms": list(GENERIC_SAMPLED),
        "closure_floor": CLOSURE_FLOOR, "closure_ceiling": CLOSURE_CEILING,
        "hyperparameters": HYPERPARAMETERS, "snapshot_count": SNAPSHOT_COUNT,
        "candidate_cap": CANDIDATE_CAP, "decision_cap": DECISION_CAP,
        "max_train_rows": MAX_TRAIN_ROWS, "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set, "smoke": bool(args.smoke),
        "replay_seconds": replay_seconds,
        "note": "Only the training population of the L2 ranker varies. Features, target, model, "
                "split, embargo, sampled candidate width, size model and seeds are the Phase 0.9 "
                "and 0.95 settings. A_pd keeps the Phase 0.9 standardisation (per-decision at fit "
                "time, FixedModelScorer normalising on the ranked set at replay time); every other "
                "fit takes raw features and is replayed through RawFeatureScorer.",
    }
    text = json.dumps(config, indent=2, default=str) + "\n"
    (args.output_dir / "run_config.json").write_text(text, encoding="utf-8")
    (args.paper_dir / "decision_population_config.json").write_text(text, encoding="utf-8")
    synthetic = {name for name in traces if "synthetic" in name}
    make_figures(args.paper_dir, summary, predictive_rows, onpolicy_rows, synthetic)
    print(f"done in {time.time()-clock:.0f}s", flush=True)


def _register(rankers: dict, fit_rows: list[dict], fit: dict, reference: dict) -> None:
    """Keep the fitted ranker and turn its metadata into one CSV row."""
    ranker = fit.pop("ranker")
    key = (fit["trace"], fit["target"], fit["population"], fit["cell"])
    rankers[key] = ranker
    row = dict(fit)
    # Convergence diagnostics, so a null result can be told apart from a fit
    # that never solved: Newton steps and the flag for the logistic ranker, the
    # conditioning of the solved normal matrix for the ridge one.
    row["fit_iterations"] = int(getattr(ranker, "iterations", -1))
    row["fit_converged"] = bool(getattr(ranker, "converged", False))
    row["fit_condition_number"] = float(getattr(ranker, "condition_number", math.nan))
    if fit["population"] == "A_pd":
        stored = reference.get((fit["trace"], f"{fit['target']}_h{int(fit['horizon_seconds'])}"))
        coefficients = stored.get("coefficients") if stored else None
        row["phase09_coefficients_found"] = bool(coefficients)
        row["phase09_max_coefficient_deviation"] = (
            max_coefficient_deviation(ranker, coefficients) if coefficients else math.nan
        )
    row.update(coefficient_row(ranker))
    fit_rows.append(row)
    print(f"    fit {fit['trace']}/{fit['target']}/{fit['population']}"
          f"{'/' + fit['cell'] if fit['cell'] else ''}: rows={fit['train_rows']} "
          f"positives={fit['train_positives']} in {fit['seconds']:.1f}s"
          + (f" devn={row['phase09_max_coefficient_deviation']:.2e}"
             if fit["population"] == "A_pd" else ""), flush=True)


if __name__ == "__main__":
    main()
