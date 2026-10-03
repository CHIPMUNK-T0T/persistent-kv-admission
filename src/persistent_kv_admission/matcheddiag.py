"""Matched-horizon diagnosis — the frozen ranker's errors at the reuse boundary that works for the cell.

`docs/matched-horizon-diagnosis-plan.md` is the pre-registration. The
ranker-error diagnosis (`errordiag`, `docs/ranker-error-diagnosis-plan.md`)
read the frozen `pi0` ranker's logged decisions with the reuse boundary at
600 seconds only; the horizon control and its fill-in then found the horizon
`h*` at which the exact reuse bit with recency reaches the exact `next_use`
label to be 60, 150 or 300 seconds in eight of the twelve trace x cell. This
module re-reads the same saved held-out decision logs with the horizon as a
parameter; it replays nothing and fits nothing.

Reused unchanged from `errordiag`: the decisions of a population at a horizon
(`decisions_from_arrays`, which takes the horizon as an argument, so the
population and its stored `horizon_seconds` are never touched), the victims
(logged, scorer, recency), `m3` / `m4` (`statistics`), the four error kinds
with the addendum's precedence (`class_rows`), the conditional rates against
recency and the uniform expectation (`conditional_rates`, `uniform_rates`)
and the seed reading against a baseline (`baseline_reading`, on
`mechanism.sign_reading`). At h = 600 every one of these is the diagnosis's
own computation on the same arrays.

New here:

* the horizons {60, 150, 300, 600} s and the matched horizon `h*` of each
  cell, the plan's table (the same on both traces);
* `m3` and `m4` by subset of decisions (every decision, the logged victim is
  the arrival, a resident): reading 4's bridge quantity is `m4` over every
  decision and over the resident-victim decisions;
* the within-decision concordance of a chooser's key for the reuse bit at a
  horizon: over the decisions with both a reusable and a non-reusable
  candidate, the share of the (reusable, non-reusable) candidate pairs in
  which the key places the non-reusable candidate first (a lower key is
  evicted first), an exact tie of the key counting one half, averaged over
  decisions with each decision weighing one. The ranker's key is (score,
  stored tie-break) compared lexicographically, recency's the stored
  tie-break alone; a uniformly drawn victim has 0.5 exactly;
* the plan's two registered predictions as rules (composition: the avoidable
  reusable eviction carries at least 0.99 of the excess at `h*`; separation:
  the `next_use` ranker's concordance for the `h*`-bit is below that for the
  600-second bit in the five-seed mean, in the trace x cell with `h*` < 600 s);
* Spearman's rank correlation with average ranks for ties (reading 4's
  description of twelve points; numpy only).

Not done here: loading, hashing and checking (the runner,
`scripts/run_matched_horizon_diagnosis.py`), any fit, any replay, any choice
of a horizon before the fact.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import errordiag
from .errordiag import Decisions
from .onpolicy import DecisionPopulation

# --- the fixed surface (docs/matched-horizon-diagnosis-plan.md) -------------------------------

HORIZONS_SECONDS = (60.0, 150.0, 300.0, 600.0)
# The diagnosis's horizon, where every statistic must reproduce its published value.
DIAGNOSIS_HORIZON_SECONDS = 600.0
# h* of each (L1 fraction, L2 multiplier), the plan's table: the horizon of the
# grids run that minimises S_h, the same on both traces.
MATCHED_HORIZON_SECONDS = {
    (0.0025, 1.0): 60.0,
    (0.0025, 4.0): 150.0,
    (0.01, 1.0): 150.0,
    (0.01, 4.0): 600.0,
    (0.02, 1.0): 300.0,
    (0.02, 4.0): 600.0,
}
# Reading 1's prediction: the avoidable reusable eviction carries at least this
# share of the label excess at h*.
COMPOSITION_SHARE = 0.99
# Reading 3: a uniformly drawn victim's concordance, exact.
UNIFORM_CONCORDANCE = 0.5
# Reading 3's registered prediction is about this target's ranker.
PREDICTION_TARGET = "next_use"
# Victim sources of the m3 / m4 table: the logged victim and the published
# pi0 model's recomputed one (which must be the logged one).
VICTIM_SOURCES = ("logged", "scorer_pi0")
# Decisions per block of the vectorised pairwise comparison (bounds memory).
_CONCORDANCE_CHUNK = 4096

assert set(HORIZONS_SECONDS) >= set(MATCHED_HORIZON_SECONDS.values())
assert DIAGNOSIS_HORIZON_SECONDS in HORIZONS_SECONDS


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def matched_horizon(fraction: float, multiplier: float) -> float:
    """h* of a cell, in seconds."""
    key = (float(fraction), float(multiplier))
    if key not in MATCHED_HORIZON_SECONDS:
        raise KeyError(f"no matched horizon for cell {key}")
    return MATCHED_HORIZON_SECONDS[key]


def is_matched(fraction: float, multiplier: float, horizon_seconds: float) -> bool:
    """Whether `horizon_seconds` is the cell's h*."""
    return float(horizon_seconds) == matched_horizon(fraction, multiplier)


# --- the decisions of a population at a horizon ----------------------------------------------


def decisions_at(population: DecisionPopulation, horizon_seconds: float) -> Decisions:
    """`errordiag.Decisions` of a population with the reuse boundary at
    `horizon_seconds`: `label_h = -log1p(min(next_use_delta_s, h))`, reusable
    when the next use is at most h away. The population is not changed; its
    stored `horizon_seconds` (the logger's, 600 s) is not read. At h = 600 this
    is `errordiag.decisions_of` on the same arrays."""
    return errordiag.decisions_from_arrays(
        population.group, population.next_use_delta_ms, population.arm_tiebreak,
        population.arriving, population.state_index, horizon_seconds=float(horizon_seconds))


# --- m3 and m4 by subset ---------------------------------------------------------------------


def subset_statistics(decisions: Decisions, victims: np.ndarray) -> dict[str, dict[str, float]]:
    """`m3` and `m4` with their sums for every decision (`errordiag.statistics`
    as it is) and for the decisions whose victim is the arrival or a resident
    (the diagnosis's subsets)."""
    arrival = decisions.arriving[victims]
    value = errordiag.excess(decisions, victims)
    flag = errordiag.avoidable(decisions, victims)
    out = {"all": errordiag.statistics(decisions, victims)}
    for subset, mask in (("arrival", arrival), ("resident", ~arrival)):
        count = int(mask.sum())
        excess_sum = float(value[mask].sum())
        avoidable_count = int(flag[mask].sum())
        out[subset] = {"decisions": count, "excess_sum": excess_sum,
                       "m3": _ratio(excess_sum, count), "avoidable_count": avoidable_count,
                       "m4": _ratio(avoidable_count, count)}
    return out


# --- reading 3: within-decision concordance --------------------------------------------------


def key_ranks(primary, secondary=None) -> np.ndarray:
    """A dense integer rank of every row's key: `(primary, secondary)` compared
    lexicographically, or `primary` alone. Equal keys (exactly equal floats,
    -0.0 equal to 0.0) get equal ranks and a lower key a lower rank, so any two
    rows compare as their keys do; only comparisons within a decision are used."""
    primary = np.asarray(primary, dtype=float)
    secondary = (np.zeros(len(primary)) if secondary is None
                 else np.asarray(secondary, dtype=float))
    if primary.shape != secondary.shape or primary.ndim != 1:
        raise ValueError("a key's components have different shapes")
    if not (np.isfinite(primary).all() and np.isfinite(secondary).all()):
        raise ValueError("a key is not finite")
    if not len(primary):
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((secondary, primary))
    first, second = primary[order], secondary[order]
    new = np.r_[True, (first[1:] != first[:-1]) | (second[1:] != second[:-1])]
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.cumsum(new) - 1
    return ranks


def ranker_key_ranks(scores: np.ndarray, decisions: Decisions) -> np.ndarray:
    """The ranker's key, `(score, stored tie-break)`."""
    return key_ranks(scores, decisions.tiebreak)


def recency_key_ranks(decisions: Decisions) -> np.ndarray:
    """Recency's key, the stored tie-break alone."""
    return key_ranks(decisions.tiebreak)


def concordance_by_decision(ranks: np.ndarray, decisions: Decisions) -> np.ndarray:
    """Each decision's concordance of the key whose row ranks are `ranks` for
    the reuse bit of `decisions` (nan where the decision does not have both a
    reusable and a non-reusable candidate).

    With R the reusable candidates and N the others, it is the share of the
    |R| x |N| pairs (r, n) whose key places n first (key(n) < key(r): n would be
    evicted before r), an exact tie of the key counting one half, computed as
    `(2 * first + ties) / (2 * |R| * |N|)` from integer counts, one rounding.
    Decisions are grouped by width and compared pairwise in blocks.
    """
    ranks = np.asarray(ranks)
    if ranks.shape != decisions.reusable.shape:
        raise ValueError("a key rank per candidate row is needed")
    out = np.full(len(decisions), np.nan)
    mixed = decisions.mixed
    for width in np.unique(decisions.widths[mixed]).tolist():
        chosen = np.flatnonzero(mixed & (decisions.widths == width))
        offsets = np.arange(width)
        for begin in range(0, len(chosen), _CONCORDANCE_CHUNK):
            part = chosen[begin:begin + _CONCORDANCE_CHUNK]
            rows = decisions.starts[part][:, None] + offsets
            rank = ranks[rows]
            reusable = decisions.reusable[rows]
            # [d, i, j]: candidate i is reusable and candidate j is not ...
            pair = reusable[:, :, None] & ~reusable[:, None, :]
            # ... and j's key is below i's (j first) or equal to it.
            first = rank[:, None, :] < rank[:, :, None]
            tied = rank[:, None, :] == rank[:, :, None]
            first_count = (pair & first).sum(axis=(1, 2))
            tie_count = (pair & tied).sum(axis=(1, 2))
            count = decisions.reusable_count[part]
            pairs = count * (width - count)
            out[part] = (2 * first_count + tie_count) / (2 * pairs)
    return out


def concordance(ranks: np.ndarray, decisions: Decisions) -> dict[str, float]:
    """The mean of `concordance_by_decision` over the decisions with both
    classes present, each weighing one, and their number (nan when none)."""
    values = concordance_by_decision(ranks, decisions)
    mixed = decisions.mixed
    return {"concordance_decisions": int(mixed.sum()),
            "concordance": float(np.mean(values[mixed])) if mixed.any() else math.nan}


# --- one population at one horizon -----------------------------------------------------------


def horizon_statistics(decisions: Decisions, logged: np.ndarray, recomputed: np.ndarray,
                       recency: np.ndarray, ranker_ranks: np.ndarray,
                       recency_ranks: np.ndarray) -> dict[str, object]:
    """Everything the readings need from one population at one horizon.

    `decisions` is the population at the horizon (`decisions_at`); the victim
    arrays and key ranks do not depend on the horizon and are computed once.
    Classes and rates are by the logged victim, as in the diagnosis; m3 / m4
    for both victim sources; concordance for the ranker's and recency's keys.
    """
    ranker = concordance(ranker_ranks, decisions)
    other = concordance(recency_ranks, decisions)
    if ranker["concordance_decisions"] != other["concordance_decisions"]:
        raise AssertionError("the two keys see different decisions")
    return {
        "statistics": {"logged": subset_statistics(decisions, logged),
                       "scorer_pi0": subset_statistics(decisions, recomputed)},
        "classes": errordiag.class_rows(decisions, logged),
        "rates": {"ranker": errordiag.conditional_rates(decisions, logged),
                  "recency": errordiag.conditional_rates(decisions, recency),
                  "uniform": errordiag.uniform_rates(decisions)},
        "concordance": {"decisions": ranker["concordance_decisions"],
                        "ranker": ranker["concordance"], "recency": other["concordance"],
                        "uniform": UNIFORM_CONCORDANCE},
    }


# --- the reading rules -----------------------------------------------------------------------


def composition_holds(avoidable_excess_share: float) -> bool:
    """Reading 1's prediction for one value: the avoidable reusable eviction
    carries at least 0.99 of the excess (exactly 0.99 holds; nan does not)."""
    return math.isfinite(avoidable_excess_share) and avoidable_excess_share >= COMPOSITION_SHARE


def prediction_registered(target: str, fraction: float, multiplier: float) -> bool:
    """Reading 3's prediction is registered for the `next_use` ranker in the
    trace x cell with h* < 600 s (eight of twelve)."""
    return (target == PREDICTION_TARGET
            and matched_horizon(fraction, multiplier) < DIAGNOSIS_HORIZON_SECONDS)


def concordance_prediction_holds(at_matched: float, at_600: float) -> bool:
    """Reading 3's prediction for one trace x cell: the five-seed mean
    concordance for the h*-bit is strictly below that for the 600-second bit
    (equal fails; a nan fails)."""
    return math.isfinite(at_matched) and math.isfinite(at_600) and at_matched < at_600


def baseline_reading(ranker: Sequence[float], baseline: Sequence[float]) -> str:
    """Reading 2's seed rule, `errordiag.baseline_reading` as it is."""
    return errordiag.baseline_reading(ranker, baseline)


# --- reading 4: Spearman ---------------------------------------------------------------------


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks, each group of exactly equal values getting the mean of
    the positions it occupies."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation: Pearson's correlation of the average ranks.
    nan with fewer than two points, a non-finite value, or a constant side."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Spearman needs two sequences of one length")
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return math.nan
    dx = average_ranks(x)
    dy = average_ranks(y)
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    denominator = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    return float((dx * dy).sum()) / denominator if denominator else math.nan
