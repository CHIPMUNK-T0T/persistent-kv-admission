#!/usr/bin/env python3
"""Phase 0.98: which eviction decisions cost the reuse, and what they stranded?

Phase 0.97 refuted the decision-population hypothesis and left one narrower
question behind: on the cells where the learned or victim-trained L2 loses most,
*which* decisions lose it — declining the arriving victim, or dropping a
resident — and is the present-but-unusable KV a property of the sampled
mechanism or of the learned scores. This script answers that descriptively.
Nothing is fitted here that Phase 0.97 did not fit, no feature and no policy is
added, and the arms are replayed exactly as they were: the same fit procedure,
the same populations, the same row caps, the same seed 0 for every subsample.
The replays are therefore expected to reproduce
`results/paper/decision_population_replay_seeds.csv` bit for bit, which the run
asserts before it writes anything.

Three read-only hooks on `run_two_tier` carry the measurement (see
`persistent_kv_admission.attribution`): one sees each request's L1 prefix and
L2 hit sets, one sees every state that leaves L2 and why, and the existing
on-policy decision logger of Phase 0.97 supplies the decisions themselves. With
the hooks absent the replay is byte-identical, which is what makes the
reproduction check meaningful.

Grid, fixed before the run: the two real traces, the three cells the Phase 0.97
results picked out (0.25% x 1, 1% x 4, 2% x 4), the three generic sampled arms
and A_none / B / C_lru / C_union on the next-use and the binary target, five
seeds.

Outputs, written to --output-dir and copied to --paper-dir: the loss
attribution per seed and aggregated with the pre-registered readings, the
orphaning table, the decision-type regret and ranking split, the run
configuration, and figure 17.

Phase 0.98b addendum: the same grid is rerun with one more counter family and
nothing else. Phase 0.98 charged one block per broken chain — the root — and
covered only 22-41% of the L2-hit shortfall; the per-block charge sends every
absent block beyond the L1 prefix to its own last removal, so the whole absent
part of a request is attributed and the rejected / evicted split is taken on
the full shortfall. Two checks decide whether the run is trustworthy and are
run before anything is written: every pre-existing column of every seed row
must reproduce the committed Phase 0.98 table (integers equal, floats within
1e-9) on top of the Phase 0.97 reproduction, and the arm-independent parts of
the decomposition (compulsory tokens, blocks beyond the prefix, tokens L1
avoided) must be identical across the arms of a trace x cell x seed, which the
run reports either way and never corrects. The per-block readings sit in the
same CSVs next to the root-only ones, and figure 18 stacks them.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persistent_kv_admission.attribution import (
    DECISION_CATEGORIES,
    LOSS_CATEGORIES,
    AttributionCollector,
    decision_regret,
    ranking_split,
)
from persistent_kv_admission.crossworkload import HYPERPARAMETERS, TARGETS
from persistent_kv_admission.decisionpop import (
    BEHAVIOUR_POLICIES,
    DECISION_CAP,
    OnPolicyDecisionLogger,
    RawFeatureScorer,
    horizon_for,
    state_indices,
)
from persistent_kv_admission.gap import summarize, working_set_bytes
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

REPOSITORY = Path(__file__).resolve().parents[1]
PHASE097_SCRIPT = Path(__file__).resolve().parent / "run_decision_population.py"


def _load_phase097():
    """Import the Phase 0.97 runner as a module, unchanged, and reuse its stages.

    The fits this phase replays must be the ones Phase 0.97 produced, so they
    are produced by the same functions rather than by a copy of them. Importing
    the script by path leaves it exactly as it was published; `main` is behind
    the usual guard, so nothing runs on import.
    """
    spec = importlib.util.spec_from_file_location("run_decision_population", PHASE097_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rdp = _load_phase097()

# The cells were chosen from the Phase 0.97 results before this phase ran:
# 0.25% x 1 (largest on-policy inversion), 1% x 4 (C_lru below sampled LRU on
# binary / count, learned at parity with 2-hit), 2% x 4 (B collapse).
CELLS = ((0.0025, 1.0), (0.01, 4.0), (0.02, 4.0))
GENERIC_ARMS = ("lru", "lfu", "lru_2hit")
LEARNED_ARMS = ("A_none", "B", "C_lru", "C_union")
ATTRIBUTION_TARGETS = ("next_use", "binary")
DEFAULT_SEEDS = 5
# The mechanism control every difference and ratio is taken against.
LOSS_FLOOR = "lru_s"
# The smoke grid: one trace, one cell, one seed, three arms.
SMOKE_CELL = (0.01, 4.0)
SMOKE_GENERIC = ("lru",)
SMOKE_LEARNED = ("A_none", "B")
SMOKE_TARGETS = ("next_use",)
# The Phase 0.97 per-seed replay table this phase must reproduce exactly. Anchored
# at the repository, not at the working directory, and never at --paper-dir: a
# smoke run writing elsewhere is still checked against the published numbers.
PHASE097_SEEDS = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"
# The committed Phase 0.98 per-seed loss table. Phase 0.98b adds columns to it
# and must leave every column it already had untouched; anchored at the
# repository for the same reason as the Phase 0.97 table above.
PHASE098_SEEDS = REPOSITORY / "results/paper/decision_attribution_losses_seeds.csv"
# Compared with a relative-or-absolute tolerance: the pre-existing columns are
# integers or ratios of integers, so anything above this is a real change.
COLUMN_TOLERANCE = 1e-9
# Columns of the committed table that are not reproducible and are not checked.
VOLATILE_COLUMNS = ("seconds",)
# Per trace x cell x seed these must not depend on the arm: L1 is prefix-closed
# and never consults L2, and a block L2 never held is compulsory for every arm.
INVARIANT_COLUMNS = ("absent_compulsory_tokens", "beyond_prefix_tokens", "l1_avoided_tokens")

# Pre-registered reading thresholds (docs/experiment-plan.md, Phase 0.98).
DOMINANCE_SHARE = 0.5
ORPHAN_MECHANISM_RATIO = 1.2
ORPHAN_LEARNING_RATIO = 2.0
VICTIM_AUC_CEILING = 0.5
RESIDENTS_AUC_FLOOR = 0.6

# Token columns that also get a share-of-input column and a difference to lru_s.
TOKEN_COLUMNS = tuple(
    [f"root_{name}_tokens" for name in LOSS_CATEGORIES]
    + [f"unusable_after_{name}_tokens" for name in LOSS_CATEGORIES]
    + [f"absent_{name}_tokens" for name in LOSS_CATEGORIES]
    + [f"downstream_{name}_tokens" for name in LOSS_CATEGORIES]
    + ["downstream_absent_tokens", "l2_hit_tokens", "root_loss_tokens", "unusable_tokens",
       "decision_loss_tokens", "absent_loss_tokens", "perblock_decision_loss_tokens"]
)
# Block counts of the per-block charge. Aggregated over the seeds because the
# tokens-per-absent-block reading divides by them; no difference to sampled LRU
# is taken, since the reading is about the size of a block and not about an arm.
BLOCK_COLUMNS = tuple(
    [f"absent_{name}_blocks" for name in LOSS_CATEGORIES]
    + [f"downstream_{name}_blocks" for name in LOSS_CATEGORIES]
)
LOSS_METRICS = tuple(
    list(TOKEN_COLUMNS)
    + [f"{column}_share" for column in TOKEN_COLUMNS]
    + list(BLOCK_COLUMNS)
    + ["avoided_prefill_tokens", "l2_avoided_tokens", "extra_avoided_tokens",
       "extra_fraction_of_input", "unexplained_states", "seconds"]
)
DIFF_METRICS = tuple(
    [f"diff_{column}" for column in TOKEN_COLUMNS]
    + [f"diff_{column}_share" for column in TOKEN_COLUMNS]
    + ["diff_avoided_prefill_tokens", "diff_extra_fraction_of_input"]
)
ORPHAN_METRICS = (
    "removal_rejections", "removal_rejections_window", "removal_promotions",
    "resident_evictions", "resident_evictions_window", "evictions_with_orphans",
    "evictions_with_orphans_window", "share_evictions_with_orphans",
    "share_evictions_with_orphans_window", "orphaned_blocks", "orphaned_blocks_window",
    "orphaned_bytes", "orphaned_bytes_window", "orphaned_blocks_per_eviction",
    "l2_admissions", "l2_decisions", "l1_evictions",
)
DECISION_METRICS = (
    "decisions_offered", "decisions_in_window", "decisions_kept",
    "rejected_decisions", "rejected_reused_within_h_share", "rejected_avoidable_share",
    "evicted_decisions", "evicted_reused_within_h_share", "evicted_avoidable_share",
    "victim_vs_resident_metric", "victim_vs_resident_pairs", "first_round_decisions",
    "victim_rank_fraction_mean", "residents_only_metric", "residents_only_decisions",
    "residents_constant_label", "whole_set_metric", "whole_set_decisions_scored",
    "evicted_lowest_label_rate", "victim_matches_argmin_rate", "prevalence",
)
IDENTITY = ("trace", "l1_fraction", "l2_multiplier", "cell", "arm", "kind", "target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", type=Path,
                        help="trace files (required unless --figure-only)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision_attribution"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw fig17 from the CSVs in <output-dir> without replaying")
    parser.add_argument("--smoke", action="store_true",
                        help="one trace, the 1%% x 4x cell, seed 0, lru_s + A_none + B: a probe")
    return parser.parse_args()


def git_head() -> dict[str, object]:
    """The commit the run was made at, and whether the tree was clean."""
    def run(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=REPOSITORY, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"git_head": run("rev-parse", "HEAD"),
                "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": "", "git_dirty": None}


# --- the replay worker --------------------------------------------------------


def _attribution_worker(task):
    """One Phase 0.97 replay, with the three diagnostics attached to it."""
    name, fraction, multiplier, kind, arm, target, seed = task
    shared = rdp._SHARED
    trace = shared["traces"][name]
    split_ms = shared["splits"][name]
    started = time.time()
    collector = AttributionCollector(trace, measure_from_ms=split_ms)
    logger = OnPolicyDecisionLogger(
        trace, shared["horizons"][name], split_ms, trace.end_ms,
        max_decisions=DECISION_CAP, seed=0, state_index=shared["state_index"][name],
    )
    common = dict(
        hit_model=rdp.HIT_MODEL, closure=rdp.CLOSURE,
        occurrence_groups=shared["groups"][name], measure_from_ms=split_ms,
        l2_eviction="sampled", l2_sample_width=rdp.SAMPLE_WIDTH, l2_seed=seed,
        l2_decision_hook=logger, l2_request_hook=collector.on_request,
        l2_removal_hook=collector,
    )
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    if kind == "sampled":
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, arm,
                              l2_arm=f"{arm}_s", **common)
    else:
        ranker = shared["rankers"][rdp.ranker_key(name, target, arm, fraction, multiplier)]
        result = run_two_tier(trace, rdp.L1_POLICY, l1_bytes, l2_bytes, "learned",
                              l2_scorer=RawFeatureScorer(trace, ranker), l2_arm=arm, **common)
    # Every counter the replay keeps for itself must agree with the attribution
    # that was built beside it; a disagreement means the classification is wrong.
    collector.check_against(result)
    seconds = time.time() - started

    identity = {"trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                "cell": rdp.cell_label(fraction, multiplier), "arm": result.l2_arm,
                "kind": kind, "target": target, "seed": seed}
    loss = dict(
        identity,
        requested_tokens=result.requested_tokens,
        measured_requests=result.measured_requests,
        l1_avoided_tokens=result.l1_avoided_tokens,
        l2_avoided_tokens=result.l2_avoided_tokens,
        avoided_prefill_tokens=result.avoided_prefill_tokens,
        extra_avoided_tokens=result.avoided_prefill_tokens - result.l1_avoided_tokens,
        l2_present_unusable_tokens=result.l2_present_unusable_tokens,
        seconds=seconds,
        **collector.loss_row(),
    )
    requested = max(result.requested_tokens, 1)
    loss["extra_fraction_of_input"] = loss["extra_avoided_tokens"] / requested
    for column in TOKEN_COLUMNS:
        loss[f"{column}_share"] = loss[column] / requested

    orphaning = dict(
        identity, seconds=seconds,
        l2_admissions=result.l2_admissions, l2_decisions=result.l2_decisions,
        l1_evictions=result.l1_evictions, l2_rejections=result.l2_rejections,
        l2_evictions=result.l2_evictions,
        **collector.orphaning_row(),
    )

    regret = decision_regret(logger)
    decisions = []
    for label_target in ATTRIBUTION_TARGETS:
        decisions.append(dict(
            identity, seed=seed, label_target=label_target,
            metric="auc" if label_target == "binary" else "spearman",
            own_target=bool(target) and label_target == target,
            horizon_seconds=shared["horizons"][name],
            **regret, **ranking_split(logger, label_target),
        ))
    return {"loss": loss, "orphaning": orphaning, "decisions": decisions}


# --- reproduction check -------------------------------------------------------


def phase097_reference(path: Path) -> dict[tuple, int]:
    """Phase 0.97's avoided prefill tokens, keyed by trace / cell / arm / target / seed."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                   row["arm"], row["target"], int(row["seed"]))
            out[key] = int(row["avoided_prefill_tokens"])
    return out


def check_reproduction(rows: list[dict], reference: dict[tuple, int]) -> tuple[int, int, list[str]]:
    """Mark every replay against Phase 0.97 and return (matched, missing, mismatches)."""
    matched = missing = 0
    mismatches: list[str] = []
    for row in rows:
        key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
               row["arm"], row["target"], int(row["seed"]))
        expected = reference.get(key)
        if expected is None:
            missing += 1
            row["phase097_avoided_prefill_tokens"] = ""
            row["reproduces_phase097"] = ""
            continue
        row["phase097_avoided_prefill_tokens"] = expected
        same = int(row["avoided_prefill_tokens"]) == expected
        row["reproduces_phase097"] = same
        if same:
            matched += 1
        else:
            mismatches.append(
                f"{row['trace']}/{row['cell']}/{row['arm']}/{row['target'] or '-'}/s{row['seed']}: "
                f"{row['avoided_prefill_tokens']} != {expected}"
            )
    return matched, missing, mismatches


def phase098_reference(path: Path) -> dict[tuple, dict[str, str]]:
    """The committed Phase 0.98 seed rows, whole, keyed the same way."""
    if not path.exists():
        return {}
    out: dict[tuple, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                   row["arm"], row["target"], int(row["seed"]))
            out[key] = row
    return out


def _same_value(mine, theirs: str) -> bool:
    """Equal as the CSV holds them: exact as text, else within the tolerance.

    The text comparison comes first so that labels, booleans and empty cells are
    compared as themselves; only when it fails are both sides read as numbers,
    which is where the integers and the ratios of integers are settled.
    """
    if str(mine) == theirs:
        return True
    left, right = _number(mine), _number(theirs)
    if math.isnan(left) and math.isnan(right):
        return True
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= COLUMN_TOLERANCE * max(1.0, abs(left), abs(right))


def check_phase098_columns(rows: list[dict],
                           reference: dict[tuple, dict[str, str]]) -> tuple[int, int, list[str]]:
    """Every column Phase 0.98 published must come back unchanged from this run.

    The per-block counters are additions, so a pre-existing column that moved
    means the replay itself moved and the phase is not the rerun it claims to
    be. A column the committed table has and this run does not is a mismatch
    too: dropping a column is as much a change as changing one.
    """
    matched = missing = 0
    mismatches: list[str] = []
    for row in rows:
        key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
               row["arm"], row["target"], int(row["seed"]))
        expected = reference.get(key)
        if expected is None:
            missing += 1
            continue
        differing: list[str] = []
        for column, theirs in expected.items():
            if column in VOLATILE_COLUMNS:
                continue
            if column not in row:
                differing.append(f"{column}: absent from this run")
            elif not _same_value(row[column], theirs):
                differing.append(f"{column}: {row[column]} != {theirs}")
        if differing:
            mismatches.append(
                f"{row['trace']}/{row['cell']}/{row['arm']}/{row['target'] or '-'}/s{row['seed']}: "
                + "; ".join(differing[:4])
                + (f" (+{len(differing) - 4} more)" if len(differing) > 4 else "")
            )
        else:
            matched += 1
    return matched, missing, mismatches


def check_compulsory_invariance(rows: list[dict]) -> tuple[bool, int, list[str]]:
    """Are the arm-independent parts of the decomposition really arm-independent?

    L1 is prefix-closed and never consults L2, and a block L2 never held is
    compulsory whatever the arm decided, so within one trace x cell x seed the
    tokens L1 avoided, the tokens beyond the prefix and the compulsory part of
    the per-block charge should be one number shared by all arms. The
    pre-registration says to report the answer either way and not to correct
    it, so this counts the groups that vary and never raises.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["seed"])].append(row)
    variant_groups = 0
    lines: list[str] = []
    for key, members in sorted(groups.items()):
        varying = []
        for column in INVARIANT_COLUMNS:
            first = int(members[0][column])
            if any(int(member[column]) != first for member in members):
                low = min(int(member[column]) for member in members)
                high = max(int(member[column]) for member in members)
                varying.append(f"{column} in [{low}, {high}]")
        if varying:
            variant_groups += 1
            lines.append(f"{key[0]}/l1={key[1]:g},l2x{key[2]:g}/s{key[3]} over "
                         f"{len(members)} arms: " + "; ".join(varying))
    return variant_groups == 0, variant_groups, lines


# --- derivation ---------------------------------------------------------------


def derive_differences(rows: list[dict]) -> None:
    """Attach each row's difference to sampled LRU of the same cell and seed."""
    floor = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["seed"]): row
             for row in rows if row["arm"] == LOSS_FLOOR}
    if not floor:
        # Every difference, ratio and reading is defined against this arm, so
        # without it the derived columns are all nan and say so.
        print(f"  WARNING: {LOSS_FLOOR} is not in this grid; no difference to it can be taken",
              flush=True)
    for row in rows:
        base = floor.get((row["trace"], row["l1_fraction"], row["l2_multiplier"], row["seed"]))
        for column in list(TOKEN_COLUMNS) + [f"{c}_share" for c in TOKEN_COLUMNS] + [
                "avoided_prefill_tokens", "extra_fraction_of_input"]:
            row[f"diff_{column}"] = (row[column] - base[column]) if base is not None else math.nan


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...],
              carry: tuple[str, ...] = ()) -> list[dict]:
    """Mean, standard deviation, and CI95 half-width over the seeds of each group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, members in sorted(groups.items()):
        entry: dict[str, object] = dict(zip(keys, key))
        entry["seeds"] = len(members)
        for name in carry:
            entry[name] = members[0][name]
        for metric in metrics:
            stats = summarize([float(member[metric]) for member in members])
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
        out.append(entry)
    return out


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def attach_readings(losses: list[dict], orphaning: list[dict], decisions: list[dict]) -> None:
    """The three pre-registered readings, per arm and cell, against sampled LRU.

    * dominant failure — the decision type whose root plus present-unusable
      loss accounts for at least half of the arm's total loss difference to
      sampled LRU; "mixed" when neither does and "not_worse" when the arm's
      attributed loss is not above sampled LRU's at all.
    * orphaning — orphaned bytes within 1.2 x sampled LRU's read as borne by
      the sampling mechanism, at or above 2 x as borne by the learned score,
      between the two as unresolved. Taken on the evaluation window, which is
      the window every loss number is restricted to; the whole-trace ratio is
      reported next to it and is not what the reading uses.
    * ranking split — a victim-versus-resident AUC below 0.5 together with a
      residents-only AUC at or above 0.6 reads as a failure to place the
      arrival among the residents rather than to order the residents.

    Phase 0.98b adds the same dominant-failure rule on the per-block charge,
    next to the root-only one and never in place of it: `perblock_*` columns,
    whether the two labels agree, the share of the per-block difference carried
    by rejections, how much of the arm's L2-hit shortfall the per-block
    decision loss covers, and the mean tokens per absent block by category.
    """
    cell_of = lambda row: (row["trace"], row["l1_fraction"], row["l2_multiplier"])
    orphan_by_arm = {cell_of(row) + (row["arm"], row["target"]): row for row in orphaning}
    ranking_by_arm = {cell_of(row) + (row["arm"], row["target"]): row for row in decisions
                      if row["label_target"] == "binary"}
    floor_orphan = {cell_of(row): row for row in orphaning if row["arm"] == LOSS_FLOOR}
    for entry in losses:
        total = _number(entry["diff_decision_loss_tokens_mean"])
        parts = {}
        for name in DECISION_CATEGORIES:
            parts[name] = (_number(entry[f"diff_root_{name}_tokens_mean"])
                           + _number(entry[f"diff_unusable_after_{name}_tokens_mean"]))
            entry[f"diff_{name}_loss_tokens_mean"] = parts[name]
        entry["diff_total_loss_tokens_mean"] = total
        if not math.isfinite(total) or total <= 0:
            entry["dominant_failure"] = "not_worse" if math.isfinite(total) else ""
        else:
            dominant = [name for name, value in parts.items() if value >= DOMINANCE_SHARE * total]
            entry["dominant_failure"] = dominant[0] if len(dominant) == 1 else "mixed"

        perblock = {}
        for name in DECISION_CATEGORIES:
            perblock[name] = (_number(entry[f"diff_absent_{name}_tokens_mean"])
                              + _number(entry[f"diff_unusable_after_{name}_tokens_mean"]))
            entry[f"diff_{name}_perblock_loss_tokens_mean"] = perblock[name]
        perblock_total = _number(entry["diff_perblock_decision_loss_tokens_mean"])
        entry["diff_perblock_total_loss_tokens_mean"] = perblock_total
        if not math.isfinite(perblock_total) or perblock_total <= 0:
            entry["perblock_dominant_failure"] = (
                "not_worse" if math.isfinite(perblock_total) else ""
            )
            entry["perblock_rejected_share"] = math.nan
        else:
            dominant = [name for name, value in perblock.items()
                        if value >= DOMINANCE_SHARE * perblock_total]
            entry["perblock_dominant_failure"] = dominant[0] if len(dominant) == 1 else "mixed"
            entry["perblock_rejected_share"] = perblock["rejected"] / perblock_total
        entry["dominant_agrees"] = bool(
            entry["perblock_dominant_failure"] == entry["dominant_failure"]
        )
        # What the arm lost against sampled LRU is the L2 hits it did not get;
        # the coverage says how much of that the charged decisions account for.
        shortfall = -_number(entry["diff_l2_hit_tokens_mean"])
        entry["perblock_shortfall_coverage"] = (
            perblock_total / shortfall if shortfall > 0 else math.nan
        )
        for name in LOSS_CATEGORIES:
            blocks = _number(entry[f"absent_{name}_blocks_mean"])
            entry[f"tokens_per_absent_block_{name}"] = (
                _number(entry[f"absent_{name}_tokens_mean"]) / blocks if blocks else math.nan
            )

        key = cell_of(entry) + (entry["arm"], entry["target"])
        mine = orphan_by_arm.get(key)
        base = floor_orphan.get(cell_of(entry))
        for suffix in ("", "_window"):
            numerator = _number(mine[f"orphaned_bytes{suffix}_mean"]) if mine else math.nan
            denominator = _number(base[f"orphaned_bytes{suffix}_mean"]) if base else math.nan
            entry[f"orphaned_bytes{suffix}_mean"] = numerator
            entry[f"orphaning_ratio{suffix}"] = (
                numerator / denominator if denominator else math.nan
            )
        ratio = _number(entry["orphaning_ratio_window"])
        if not math.isfinite(ratio):
            entry["orphaning_reading"] = ""
        elif ratio <= ORPHAN_MECHANISM_RATIO:
            entry["orphaning_reading"] = "mechanism"
        elif ratio >= ORPHAN_LEARNING_RATIO:
            entry["orphaning_reading"] = "learning"
        else:
            entry["orphaning_reading"] = "unresolved"

        ranking = ranking_by_arm.get(key)
        victim = _number(ranking["victim_vs_resident_metric_mean"]) if ranking else math.nan
        residents = _number(ranking["residents_only_metric_mean"]) if ranking else math.nan
        entry["victim_vs_resident_auc_mean"] = victim
        entry["residents_only_auc_mean"] = residents
        if not (math.isfinite(victim) and math.isfinite(residents)):
            entry["ranking_reading"] = ""
        elif victim < VICTIM_AUC_CEILING and residents >= RESIDENTS_AUC_FLOOR:
            entry["ranking_reading"] = "arrival_placement"
        else:
            entry["ranking_reading"] = "other"


# --- figure -------------------------------------------------------------------


# One stack segment per attributed loss class, in the order they are stacked
# from the axis. `downstream_absent` is deliberately not a segment: it is the
# part of a request that the root loss already explains, it is charged to no
# decision, and at these budgets it is roughly 0.73 of the input tokens, so
# stacking it would compress every attributed class into the bottom twentieth
# of the panel. It is annotated above each bar instead, and is in the CSV.
FIGURE_STACK = (
    ("root_rejected_tokens_share", "root loss: rejected arrival", "tab:red"),
    ("root_evicted_tokens_share", "root loss: evicted resident", "tab:orange"),
    ("root_compulsory_tokens_share", "root loss: compulsory", "0.75"),
    ("unusable_after_rejected_tokens_share", "present-unusable after rejection", "tab:purple"),
    ("unusable_after_evicted_tokens_share", "present-unusable after eviction", "tab:pink"),
    ("unusable_after_compulsory_tokens_share", "present-unusable after compulsory", "0.55"),
)
FIGURE_ANNOTATION = "downstream_absent_tokens_share"
FIGURE_TARGET = "next_use"
# Figure 18, the Phase 0.98b counterpart: the same panels over the per-block
# charge. There is no annotation and no leftover class, because every absent
# block is now inside the stack; the two compulsory-and-unexplained classes of
# the present-unusable part are left out because they are zero by construction
# (nothing that L2 never held can be present in L2 and unusable).
PERBLOCK_STACK = (
    ("absent_rejected_tokens_share", "absent: rejected arrival", "tab:red"),
    ("absent_evicted_tokens_share", "absent: evicted resident", "tab:orange"),
    ("absent_compulsory_tokens_share", "absent: compulsory", "0.75"),
    ("unusable_after_rejected_tokens_share", "present-unusable after rejection", "tab:purple"),
    ("unusable_after_evicted_tokens_share", "present-unusable after eviction", "tab:pink"),
)


def figure_arms(rows: list[dict]) -> list[tuple[str, str]]:
    """(arm, target) in figure order: the mechanism control first, then the rest."""
    present = {(row["arm"], row["target"]) for row in rows}
    order = [(f"{arm}_s", "") for arm in GENERIC_ARMS]
    order += [(arm, FIGURE_TARGET) for arm in LEARNED_ARMS]
    return [item for item in order if item in present]


def _stacked_figure(paper_dir: Path, losses: list[dict], stack, annotation: str | None,
                    filename: str, suptitle: str) -> None:
    """One panel per trace x cell, one stacked bar per arm, shares of the input.

    Figures 17 and 18 are the same picture over two charges of the same
    requests, so they are drawn by the same code and differ only in the stack
    and in whether a leftover class is annotated above the bar.
    """
    rows = [row for row in losses
            if row["target"] in ("", FIGURE_TARGET) and "synthetic" not in str(row["trace"])]
    if not rows:
        print(f"  no real-trace rows: {filename} skipped", flush=True)
        return
    traces = sorted({row["trace"] for row in rows})
    cells = sorted({(_number(row["l1_fraction"]), _number(row["l2_multiplier"])) for row in rows})
    arms = figure_arms(rows)
    indexed = {(row["trace"], _number(row["l1_fraction"]), _number(row["l2_multiplier"]),
                row["arm"], row["target"]): row for row in rows}
    def share(name, cell, arm, target, field):
        entry = indexed.get((name, cell[0], cell[1], arm, target), {})
        return _number(entry.get(f"{field}_mean", math.nan))

    figure, axes = plt.subplots(len(traces), len(cells), squeeze=False,
                                figsize=(max(5.2, 3.8 * len(cells)), 1.2 + 3.6 * len(traces)))
    positions = np.arange(len(arms))
    for row_index, name in enumerate(traces):
        for column, cell in enumerate(cells):
            axis = axes[row_index][column]
            bottom = np.zeros(len(arms))
            for field, label, color in stack:
                values = np.nan_to_num(
                    np.array([share(name, cell, arm, target, field) for arm, target in arms]),
                    nan=0.0)
                axis.bar(positions, values, bottom=bottom, color=color, width=0.72,
                         label=label if (row_index == 0 and column == 0) else None)
                bottom += values
            headroom = 1.22 * max(bottom.max(), 1e-9)
            if annotation is not None:
                for position, (arm, target) in zip(positions, arms):
                    absent = share(name, cell, arm, target, annotation)
                    if math.isfinite(absent):
                        axis.text(position, bottom[position] + 0.02 * headroom, f"{absent:.2f}",
                                  ha="center", va="bottom", fontsize=5.5, color="0.35",
                                  rotation=90)
            axis.set(xticks=positions, xticklabels=[arm for arm, _target in arms],
                     ylim=(0.0, headroom),
                     title=f"{name} — {100 * cell[0]:g}% x{cell[1]:g}",
                     ylabel="attributed loss, share of window input" if column == 0 else "")
            axis.tick_params(axis="x", labelrotation=60, labelsize=6)
            axis.grid(True, axis="y", alpha=0.25)
    figure.legend(loc="lower center", ncol=2 if len(cells) < 2 else 3, fontsize=6.5,
                  frameon=False)
    figure.suptitle(suptitle, fontsize=9)
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.93))
    figure.savefig(paper_dir / filename, dpi=170)
    plt.close(figure)


def make_figure(paper_dir: Path, losses: list[dict]) -> None:
    _stacked_figure(
        paper_dir, losses, FIGURE_STACK, FIGURE_ANNOTATION,
        "fig17_decision_attribution.png",
        f"Where the reuse was lost ({FIGURE_TARGET} target, mean over seeds)\n"
        "above each bar: unattributed downstream-absent share",
    )


def make_perblock_figure(paper_dir: Path, losses: list[dict]) -> None:
    """Figure 18: the same panels with every absent block charged to its own removal."""
    _stacked_figure(
        paper_dir, losses, PERBLOCK_STACK, None,
        "fig18_perblock_attribution.png",
        f"Where the reuse was lost, block by block ({FIGURE_TARGET} target, mean over seeds)\n"
        "every block beyond the L1 prefix that L2 did not hold, charged to its last removal",
    )


def print_readings(losses: list[dict]) -> None:
    """The two dominant-failure labels side by side, so the log stands alone."""
    print("  readings per arm (difference to sampled LRU, mean over seeds):", flush=True)
    print(f"    {'trace':20s} {'cell':22s} {'arm':8s} {'target':8s} "
          f"{'root-only':10s} {'per-block':10s} {'rej.share':>9s} {'coverage':>9s}", flush=True)
    for entry in losses:
        print(f"    {str(entry['trace'])[:20]:20s} {str(entry['cell'])[:22]:22s} "
              f"{str(entry['arm'])[:8]:8s} {str(entry['target'] or '-')[:8]:8s} "
              f"{str(entry['dominant_failure'] or '-'):10s} "
              f"{str(entry['perblock_dominant_failure'] or '-'):10s} "
              f"{_number(entry['perblock_rejected_share']):9.3f} "
              f"{_number(entry['perblock_shortfall_coverage']):9.3f}", flush=True)


# --- main ---------------------------------------------------------------------


def build_grid(traces, cells, seeds, generic, learned, targets):
    tasks = []
    for name in traces:
        for fraction, multiplier in cells:
            for arm in generic:
                for seed in seeds:
                    tasks.append((name, fraction, multiplier, "sampled", arm, "", seed))
            for target in targets:
                for arm in learned:
                    for seed in seeds:
                        tasks.append((name, fraction, multiplier, "learned", arm, target, seed))
    return tasks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        with (args.output_dir / "decision_attribution_losses.csv").open(encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        make_figure(args.paper_dir, summary)
        make_perblock_figure(args.paper_dir, summary)
        print("fig17 and fig18 redrawn", flush=True)
        return
    if not args.traces:
        raise SystemExit("at least one trace file is required unless --figure-only is given")

    paths = list(args.traces)
    cells = CELLS
    seeds = tuple(range(args.seeds))
    generic, learned = GENERIC_ARMS, LEARNED_ARMS
    targets = ATTRIBUTION_TARGETS
    if args.smoke:
        paths, cells, seeds = paths[:1], (SMOKE_CELL,), (0,)
        generic, learned, targets = SMOKE_GENERIC, SMOKE_LEARNED, SMOKE_TARGETS
        print("smoke run: one trace, the 1% x 4 cell, seed 0, lru_s + A_none + B", flush=True)
    for target in targets:
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target!r}")
    fractions = sorted({fraction for fraction, _ in cells})
    context = mp.get_context("fork")
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    clock = time.time()

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
        print(f"loaded {trace.name}: H={horizon:g}s split={split_ms:.0f}ms", flush=True)
    # The Phase 0.97 stage workers read their inputs from this dict, so the
    # fits below are produced by exactly the code that produced them there.
    rdp._SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                       snapshots=snapshots, working_set=working_set, state_index=indices,
                       targets=targets, logs_dir=logs_dir)

    rankers: dict[tuple, object] = {}
    fit_rows: list[dict] = []
    reference_fits = rdp.phase09_reference()

    def register(fits):
        for fit in fits:
            rdp._register(rankers, fit_rows, fit, reference_fits)

    # 1. The global control fit on raw features (A_none), if any arm needs it.
    if "A_none" in learned:
        tasks = [(name, target, "ranker") for name in traces for target in targets]
        started = time.time()
        with context.Pool(min(args.workers, len(tasks))) as pool:
            register(pool.map(rdp._a_fit_worker, tasks))
        print(f"  {len(tasks)} global fits in {time.time()-started:.0f}s", flush=True)

    # 2. Victim streams and the B fits on their training half.
    if "B" in learned:
        tasks = [(name, fraction) for name in traces for fraction in fractions]
        started = time.time()
        with context.Pool(min(args.workers, len(tasks))) as pool:
            for out in pool.map(rdp._victim_worker, tasks):
                register(out["fits"])
        print(f"  {len(tasks)} victim streams in {time.time()-started:.0f}s", flush=True)

    # 3 and 4. Behaviour-policy decision logs and the candidate-population fits.
    if any(arm.startswith("C_") for arm in learned):
        tasks = [(name, fraction, multiplier, policy) for name in traces
                 for fraction, multiplier in cells for policy in BEHAVIOUR_POLICIES]
        started = time.time()
        with context.Pool(min(args.workers, len(tasks))) as pool:
            for out in pool.map(rdp._candidate_worker, tasks):
                print(f"    candidates {out['task']}: {out['meta']['decisions_seen']} decisions "
                      f"in {out['seconds']:.0f}s", flush=True)
        print(f"  {len(tasks)} decision logs in {time.time()-started:.0f}s", flush=True)
        tasks = [(name, fraction, multiplier) for name in traces
                 for fraction, multiplier in cells]
        started = time.time()
        with context.Pool(min(args.workers, len(tasks))) as pool:
            for out in pool.map(rdp._c_fit_worker, tasks):
                register(out["fits"])
        print(f"  candidate fits in {time.time()-started:.0f}s", flush=True)
    rdp._SHARED["rankers"] = rankers

    # 5. The attribution replays.
    tasks = build_grid(traces, cells, seeds, generic, learned, targets)
    print(f"  {len(tasks)} attribution replays on {args.workers} workers", flush=True)
    started = time.time()
    loss_rows, orphan_rows, decision_rows = [], [], []
    raw_path = args.output_dir / "raw_attribution.jsonl"
    with context.Pool(args.workers) as pool, raw_path.open("w", encoding="utf-8") as raw:
        for done, out in enumerate(pool.imap_unordered(_attribution_worker, tasks, chunksize=1), 1):
            loss_rows.append(out["loss"])
            orphan_rows.append(out["orphaning"])
            decision_rows.extend(out["decisions"])
            raw.write(json.dumps(out, default=str) + "\n")
            if done % 10 == 0 or done == len(tasks):
                row = out["loss"]
                print(f"  [{done}/{len(tasks)}] {time.time()-started:.0f}s "
                      f"last={row['trace']}/{row['cell']}/{row['arm']}/{row['target'] or '-'}"
                      f"/s{row['seed']} ({row['seconds']:.0f}s)", flush=True)
    replay_seconds = time.time() - started
    print(f"  replays in {replay_seconds:.0f}s", flush=True)

    timing = defaultdict(list)
    for row in loss_rows:
        timing[(row["kind"], row["arm"])].append(float(row["seconds"]))
    print("  seconds per replay, by arm:", flush=True)
    for (kind, arm), values in sorted(timing.items()):
        print(f"    {kind:8s} {arm:12s} n={len(values):4d} mean={np.mean(values):7.1f} "
              f"max={np.max(values):7.1f} total={np.sum(values):8.1f}", flush=True)

    # The whole phase rests on these being the Phase 0.97 replays, unchanged.
    matched, missing, mismatches = check_reproduction(loss_rows, phase097_reference(PHASE097_SEEDS))
    print(f"  Phase 0.97 reproduction: {matched} matched, {missing} not in the reference, "
          f"{len(mismatches)} mismatched", flush=True)
    for line in mismatches:
        print(f"    MISMATCH {line}", flush=True)
    if mismatches:
        raise SystemExit("the replays did not reproduce Phase 0.97; nothing was written")
    unexplained = sum(int(row["root_unexplained_tokens"]) for row in loss_rows)
    unexplained_absent = sum(int(row["absent_unexplained_tokens"]) for row in loss_rows)
    print(f"  unexplained root-loss tokens over the grid: {unexplained}", flush=True)
    print(f"  unexplained per-block absent tokens over the grid: {unexplained_absent}", flush=True)

    sort_key = lambda row: (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                            str(row["target"]), row["arm"], int(row["seed"]))
    derive_differences(loss_rows)
    loss_rows.sort(key=sort_key)
    orphan_rows.sort(key=sort_key)
    decision_rows.sort(key=lambda row: sort_key(row) + (row["label_target"],))

    # Phase 0.98b must be the Phase 0.98 run with counters added, so every
    # column that table already had has to come back unchanged.
    p98_matched, p98_missing, p98_mismatches = check_phase098_columns(
        loss_rows, phase098_reference(PHASE098_SEEDS))
    print(f"  Phase 0.98 column reproduction: {p98_matched} matched, "
          f"{p98_missing} not in the reference, {len(p98_mismatches)} mismatched", flush=True)
    for line in p98_mismatches:
        print(f"    MISMATCH {line}", flush=True)
    if p98_mismatches:
        raise SystemExit("the replays did not reproduce Phase 0.98; nothing was written")

    invariant, variant_groups, variant_lines = check_compulsory_invariance(loss_rows)
    print(f"  compulsory / prefix invariance across the arms: "
          f"{'holds' if invariant else 'BROKEN'}, {variant_groups} varying groups", flush=True)
    for line in variant_lines[:10]:
        print(f"    VARIES {line}", flush=True)

    loss_summary = aggregate(loss_rows, IDENTITY, LOSS_METRICS + DIFF_METRICS,
                             carry=("requested_tokens", "l1_avoided_tokens"))
    orphan_summary = aggregate(orphan_rows, IDENTITY, ORPHAN_METRICS)
    decision_summary = aggregate(decision_rows, IDENTITY + ("label_target", "metric"),
                                 DECISION_METRICS)
    attach_readings(loss_summary, orphan_summary, decision_summary)

    combined_decisions = (
        [dict(row, row_type="seed") for row in decision_rows]
        + [dict(row, row_type="aggregate") for row in decision_summary]
    )
    combined_orphaning = (
        [dict(row, row_type="seed") for row in orphan_rows]
        + [dict(row, row_type="aggregate") for row in orphan_summary]
    )
    for directory in (args.output_dir, args.paper_dir):
        rdp._write(directory / "decision_attribution_losses.csv", loss_summary)
        rdp._write(directory / "decision_attribution_losses_seeds.csv", loss_rows)
        rdp._write(directory / "decision_attribution_decisions.csv", combined_decisions)
        rdp._write(directory / "decision_attribution_orphaning.csv", combined_orphaning)

    config = {
        **git_head(),
        "phase": "0.98b",
        "l1_policy": rdp.L1_POLICY, "hit_model": rdp.HIT_MODEL, "closure": rdp.CLOSURE,
        "cells": [list(cell) for cell in cells], "seeds": list(seeds),
        "generic_arms": list(generic), "learned_arms": list(learned), "targets": list(targets),
        "sample_width": rdp.SAMPLE_WIDTH, "decision_cap": DECISION_CAP,
        "loss_floor": LOSS_FLOOR, "loss_categories": list(LOSS_CATEGORIES),
        "dominance_share": DOMINANCE_SHARE,
        "orphan_mechanism_ratio": ORPHAN_MECHANISM_RATIO,
        "orphan_learning_ratio": ORPHAN_LEARNING_RATIO,
        "victim_auc_ceiling": VICTIM_AUC_CEILING, "residents_auc_floor": RESIDENTS_AUC_FLOOR,
        "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons, "measure_from_ms": splits,
        "working_set_bytes": working_set, "smoke": bool(args.smoke),
        "replays": len(loss_rows), "replay_seconds": replay_seconds,
        "phase097_reference": str(PHASE097_SEEDS),
        "phase097_matched": matched, "phase097_missing": missing,
        "phase097_mismatched": len(mismatches),
        "phase098_reference": str(PHASE098_SEEDS),
        "phase098_matched": p98_matched, "phase098_missing": p98_missing,
        "phase098_mismatched": len(p98_mismatches),
        "column_tolerance": COLUMN_TOLERANCE,
        "compulsory_arm_invariant": invariant,
        "compulsory_variant_groups": variant_groups,
        "invariant_columns": list(INVARIANT_COLUMNS),
        "unexplained_root_loss_tokens": unexplained,
        "unexplained_absent_tokens": unexplained_absent,
        "note": "Descriptive only. No fit, feature, policy or threshold in this phase changes "
                "anything the replay does: the three hooks are read-only, the arms are the "
                "Phase 0.97 arms rebuilt by the Phase 0.97 code at seed 0, and every replay is "
                "checked against decision_population_replay_seeds.csv before anything is written. "
                "Phase 0.98b adds one read-only counter family to the same request hook - every "
                "absent block beyond the L1 prefix charged to its own last removal, next to the "
                "root-only charge, which the committed Phase 0.98 columns are checked to "
                "reproduce exactly.",
    }
    text = json.dumps(config, indent=2, default=str) + "\n"
    (args.output_dir / "run_config.json").write_text(text, encoding="utf-8")
    (args.paper_dir / "decision_attribution_config.json").write_text(text, encoding="utf-8")
    rdp._write(args.output_dir / "decision_attribution_fits.csv", fit_rows)
    make_figure(args.paper_dir, loss_summary)
    make_perblock_figure(args.paper_dir, loss_summary)
    print_readings(loss_summary)
    print(f"done in {time.time()-clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
