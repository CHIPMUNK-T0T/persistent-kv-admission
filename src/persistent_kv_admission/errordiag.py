"""Ranker-error diagnosis — selection or population, and which resident errors.

`docs/ranker-error-diagnosis-plan.md` is the pre-registration. The error-location
control measured each ranker's mean victim label excess (`m3`) on the decisions
that ranker itself produced, so a change of `m3` between `pi0` and `pi3` mixed a
change in how the ranker chooses with a change in the candidate sets it is
shown. This module re-reads the saved held-out decision logs of the published
on-policy run (`onpolicy.DecisionPopulation`, reservoirs of whole decisions)
without any replay or fit.

A *victim array* holds one global row index per logged decision, in stored
decision order. Its sources:

* a scorer (a published ranker): the first minimum, in stored candidate order,
  of `(score on the stored features, stored tie-break)` — the published rule,
  with the score from `onpolicy.sequential_ranker_score` (the arithmetic of the
  replay's `score_row`) and the minimum from `onpolicy._first_tuple_argmin`, the
  function the published cross-score applied; no hybrid, protection or override;
* the logged victim (the population's `victim` flag);
* recency: the first minimum of the stored tie-break;
* the reference: the first minimum of `(label, stored tie-break)`.

A uniformly drawn candidate has no victim array: its rates are exact
expectations (each candidate 1 / width), with no random draw.

Labels are the `next_use` target, `-log1p(min(delta_s, H))`, and a candidate is
reusable when its next use is at most H away (the `binary` target), both from
`decisionpop.target_column` on the stored label primitives. Everything here is
a pure function of arrays; `scripts/run_ranker_error_diagnosis.py` loads, hashes
and checks. The seed-sign readings are `mechanism.sign_reading` and the "has the
sign of" rule is `errorloc.agrees`, used as they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .decisionpop import target_column
from .errorloc import agrees, contradicts
from .mechanism import sign_reading
from .onpolicy import DecisionPopulation, _first_tuple_argmin, sequential_ranker_score

LABEL_TARGET = "next_use"
REUSE_TARGET = "binary"
# Reading 3's classes, in the plan's order; the order is also the precedence
# where a decision meets two definitions (`class_memberships`).
CLASSES = ("reference_victim", "other_tiebreak", "avoidable_reusable", "order_error")
AVOIDABLE_CLASS = CLASSES.index("avoidable_reusable")
# Decisions by their logged victim: every decision, the arrival, a resident.
SUBSETS = ("all", "arrival", "resident")
# Reading 4: the two conditional rates and the three choosers.
RATES = ("avoidable", "order")
CHOOSERS = ("ranker", "recency", "uniform")
# Required check: the diagnosis stops as unresolved when a scorer's victims on
# its own population differ from the logged ones in more than 1 per mille of
# the decisions of any population.
STOP_PER_MILLE = 1
# Reading 5: the share of the evictions on the top 10% of victim states.
TOP_PERCENT = 10


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


# --- the decisions of one population ---------------------------------------------------------


@dataclass(frozen=True)
class Decisions:
    """The chooser-independent quantities of one population's decisions.

    Row arrays have one entry per candidate row and decision arrays one per
    decision, both in stored order; decision `d` spans the rows
    `starts[d] .. starts[d] + widths[d] - 1`. `reference` is the global row of
    the first minimum of `(label, tie-break)`.
    """

    starts: np.ndarray
    widths: np.ndarray
    label: np.ndarray
    reusable: np.ndarray
    tiebreak: np.ndarray
    arriving: np.ndarray
    state_index: np.ndarray
    min_label: np.ndarray
    reusable_count: np.ndarray
    above_min_count: np.ndarray
    reference: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def mixed(self) -> np.ndarray:
        """Decisions with both a reusable and a non-reusable candidate."""
        return (self.reusable_count > 0) & (self.reusable_count < self.widths)

    @property
    def all_reusable(self) -> np.ndarray:
        return self.reusable_count == self.widths

    @property
    def order_eligible(self) -> np.ndarray:
        """Decisions whose candidates are all reusable and not all of one label."""
        return self.all_reusable & (self.above_min_count > 0)


def decision_bounds(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First row and width of each decision, cut where `group` changes, as
    `DecisionPopulation.group_blocks` cuts them."""
    group = np.asarray(group)
    if not len(group):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(group)].astype(np.int64)
    return starts, ends - starts


def first_minimum(primary: np.ndarray, secondary: np.ndarray, starts: np.ndarray,
                  widths: np.ndarray) -> np.ndarray:
    """The global row of the first minimum of `(primary, secondary)` in each
    decision, in stored candidate order: `onpolicy._first_tuple_argmin` per
    decision, the published rule."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    out = np.empty(len(starts), dtype=np.int64)
    for index, (start, width) in enumerate(zip(starts.tolist(), widths.tolist())):
        block = slice(start, start + width)
        out[index] = start + _first_tuple_argmin(primary[block], secondary[block])
    return out


def decisions_from_arrays(group, next_use_delta_ms, tiebreak, arriving, state_index,
                          horizon_seconds: float) -> Decisions:
    """`Decisions` from a population's stored row arrays."""
    starts, widths = decision_bounds(group)
    if not len(starts):
        raise ValueError("a population without decisions")
    delta = np.asarray(next_use_delta_ms, dtype=float)
    if np.isnan(delta).any():
        raise ValueError("a next-use delta is nan")
    # Neither target reads the within-horizon count.
    unused = np.zeros(len(delta))
    label = target_column(delta, unused, LABEL_TARGET, horizon_seconds)
    reusable = target_column(delta, unused, REUSE_TARGET, horizon_seconds) == 1.0
    tiebreak = np.asarray(tiebreak, dtype=float)
    min_label = np.minimum.reduceat(label, starts)
    above = label > np.repeat(min_label, widths)
    return Decisions(
        starts=starts, widths=widths, label=label, reusable=reusable, tiebreak=tiebreak,
        arriving=np.asarray(arriving) == 1,
        state_index=np.asarray(state_index, dtype=np.int64),
        min_label=min_label,
        reusable_count=np.add.reduceat(reusable.astype(np.int64), starts),
        above_min_count=np.add.reduceat(above.astype(np.int64), starts),
        reference=first_minimum(label, tiebreak, starts, widths),
    )


def decisions_of(population: DecisionPopulation) -> Decisions:
    return decisions_from_arrays(population.group, population.next_use_delta_ms,
                                 population.arm_tiebreak, population.arriving,
                                 population.state_index, population.horizon_seconds)


# --- victims ---------------------------------------------------------------------------------


def scorer_scores(ranker, population: DecisionPopulation) -> np.ndarray:
    """A published ranker's score of every stored candidate row."""
    scores = sequential_ranker_score(ranker, population.features)
    if not np.isfinite(scores).all():
        raise ValueError("a scorer gave a non-finite score")
    return scores


def scorer_victims(scores: np.ndarray, decisions: Decisions) -> np.ndarray:
    """The first minimum of `(score, stored tie-break)` of each decision."""
    return first_minimum(scores, decisions.tiebreak, decisions.starts, decisions.widths)


def logged_victims(victim, decisions: Decisions) -> np.ndarray:
    """The row of each decision's logged victim flag."""
    rows = np.flatnonzero(np.asarray(victim) == 1)
    if len(rows) != len(decisions) or not (
            (rows >= decisions.starts) & (rows < decisions.starts + decisions.widths)).all():
        raise ValueError("the logged victims are not one per decision")
    return rows


def recency_victims(decisions: Decisions) -> np.ndarray:
    """The first minimum of the stored tie-break of each decision."""
    return first_minimum(decisions.tiebreak, np.zeros(len(decisions.tiebreak)),
                         decisions.starts, decisions.widths)


# --- m3 and m4 -------------------------------------------------------------------------------


def excess(decisions: Decisions, victims: np.ndarray) -> np.ndarray:
    """`label(victim) - min label` of each decision."""
    return decisions.label[victims] - decisions.min_label


def avoidable(decisions: Decisions, victims: np.ndarray) -> np.ndarray:
    """Whether the victim is reusable while some candidate is not."""
    return decisions.reusable[victims] & (decisions.reusable_count < decisions.widths)


def statistics(decisions: Decisions, victims: np.ndarray) -> dict[str, float]:
    """m3 (mean label excess of the victim) and m4 (share of avoidable reusable
    evictions) of a set of decisions, with their sums."""
    excess_sum = float(excess(decisions, victims).sum())
    avoidable_count = int(avoidable(decisions, victims).sum())
    count = len(decisions)
    return {"decisions": count, "excess_sum": excess_sum, "m3": _ratio(excess_sum, count),
            "avoidable_count": avoidable_count, "m4": _ratio(avoidable_count, count)}


# --- reading 3: the four classes -------------------------------------------------------------


def class_memberships(decisions: Decisions, victims: np.ndarray) -> np.ndarray:
    """The plan's four class definitions as written, one boolean row per class
    of `CLASSES`, one column per decision.

    They overlap in one case only: a victim whose next use is exactly at the
    horizon is reusable (delta <= H) and has the clipped label of a candidate
    with no reuse, so with such a candidate present the decision has no excess
    and is also an avoidable reusable eviction. Every other decision meets
    exactly one definition: a positive excess needs a victim label above the
    clipped one, i.e. a next use within the horizon.
    """
    value = excess(decisions, victims)
    if (value < 0).any():
        raise AssertionError("a victim's label lies below its decision's minimum")
    zero = value == 0
    reference = victims == decisions.reference
    return np.vstack((
        zero & reference,
        zero & ~reference,
        avoidable(decisions, victims),
        decisions.all_reusable & (value > 0),
    ))


def classify(decisions: Decisions, victims: np.ndarray) -> np.ndarray:
    """Each decision's class, as an index into `CLASSES`.

    The plan states that each decision falls in exactly one class. Where two
    definitions hold (the horizon case of `class_memberships`) the first in the
    plan's order is taken, so that decision counts as a no-excess class; the
    class table reports how many decisions this applies to.
    """
    memberships = class_memberships(decisions, victims)
    if not memberships.any(axis=0).all():
        raise AssertionError("a decision meets none of the four class definitions")
    return np.argmax(memberships, axis=0)


def class_rows(decisions: Decisions, victims: np.ndarray) -> list[dict[str, object]]:
    """Reading 3 for one population: per subset of decisions (every decision,
    victim is the arrival, victim is a resident) and class, the decisions and
    their label excess, as counts and as shares of the subset and of the
    population. `overlap_decisions` counts the decisions of the class that also
    meet a later definition."""
    memberships = class_memberships(decisions, victims)
    classes = classify(decisions, victims)
    overlap = memberships.sum(axis=0) > 1
    value = excess(decisions, victims)
    arrival = decisions.arriving[victims]
    subsets = {"all": np.ones(len(decisions), dtype=bool), "arrival": arrival,
               "resident": ~arrival}
    total_excess = float(value.sum())
    rows = []
    for subset in SUBSETS:
        mask = subsets[subset]
        subset_decisions = int(mask.sum())
        subset_excess = float(value[mask].sum())
        for index, name in enumerate(CLASSES):
            member = mask & (classes == index)
            count = int(member.sum())
            excess_sum = float(value[member].sum())
            rows.append({
                "subset": subset, "class": name, "decisions": count, "excess_sum": excess_sum,
                "overlap_decisions": int((member & overlap).sum()),
                "subset_decisions": subset_decisions, "subset_excess_sum": subset_excess,
                "decision_share": _ratio(count, subset_decisions),
                "excess_share": _ratio(excess_sum, subset_excess),
                "decision_share_of_population": _ratio(count, len(decisions)),
                "excess_share_of_population": _ratio(excess_sum, total_excess),
            })
    return rows


# --- reading 4: conditional rates ------------------------------------------------------------


def conditional_rates(decisions: Decisions, victims: np.ndarray) -> dict[str, float]:
    """The avoidable-eviction rate over the decisions with both a reusable and a
    non-reusable candidate, and the order-error rate over the decisions whose
    candidates are all reusable and not all of one label."""
    mixed = decisions.mixed
    eligible = decisions.order_eligible
    reusable = decisions.reusable[victims]
    above = decisions.label[victims] > decisions.min_label
    return {
        "avoidable_decisions": int(mixed.sum()),
        "avoidable_rate": _ratio(int((reusable & mixed).sum()), int(mixed.sum())),
        "order_decisions": int(eligible.sum()),
        "order_rate": _ratio(int((above & eligible).sum()), int(eligible.sum())),
    }


def uniform_rates(decisions: Decisions) -> dict[str, float]:
    """`conditional_rates` of a victim drawn uniformly from the candidates, as
    the exact expectation: the mean over the same decisions of the share of
    candidates that would count."""
    mixed = decisions.mixed
    eligible = decisions.order_eligible
    widths = decisions.widths.astype(float)
    return {
        "avoidable_decisions": int(mixed.sum()),
        "avoidable_rate": (float(np.mean(decisions.reusable_count[mixed] / widths[mixed]))
                           if mixed.any() else math.nan),
        "order_decisions": int(eligible.sum()),
        "order_rate": (float(np.mean(decisions.above_min_count[eligible] / widths[eligible]))
                       if eligible.any() else math.nan),
    }


# --- reading 5: concentration ----------------------------------------------------------------


def concentration(states: np.ndarray) -> dict[str, float]:
    """Over the victim states of a set of evictions: the number of distinct
    states, the share of the evictions on states that are a victim more than
    once, and the share on the `TOP_PERCENT`% of the distinct states with the
    most (rounded up, at least one state; which of several tied states is taken
    does not change the share)."""
    states = np.asarray(states)
    evictions = len(states)
    if not evictions:
        return {"evictions": 0, "distinct_states": 0, "repeated_states": 0,
                "repeat_share": math.nan, "top_states": 0, "top_share": math.nan}
    _, counts = np.unique(states, return_counts=True)
    distinct = len(counts)
    top = -(-distinct * TOP_PERCENT // 100)
    ordered = np.sort(counts)[::-1]
    return {
        "evictions": evictions, "distinct_states": distinct,
        "repeated_states": int((counts > 1).sum()),
        "repeat_share": float(counts[counts > 1].sum()) / evictions,
        "top_states": top, "top_share": float(ordered[:top].sum()) / evictions,
    }


def concentration_rows(decisions: Decisions, victims: np.ndarray) -> list[dict[str, object]]:
    """Reading 5 for one population: `concentration` over the decisions of the
    `avoidable_reusable` class (as `classify` assigns it), for every such
    decision and separately for those whose victim is the arrival or a
    resident. Each subset counts its own repeats and its own top states."""
    evictions = classify(decisions, victims) == AVOIDABLE_CLASS
    arrival = decisions.arriving[victims]
    subsets = {"all": evictions, "arrival": evictions & arrival, "resident": evictions & ~arrival}
    return [{"subset": subset, **concentration(decisions.state_index[victims[subsets[subset]]])}
            for subset in SUBSETS]


# --- the reading rules -----------------------------------------------------------------------


def exceeds_stop_rule(mismatches: int, decisions: int) -> bool:
    """Whether a population's own-scorer mismatches exceed 0.1% of its decisions
    (in integers: 1000 * mismatches > decisions)."""
    return 1000 * int(mismatches) > STOP_PER_MILLE * int(decisions)


def has_sign_of(value: float, reference: float) -> bool:
    """`value` has the sign of `reference` (zero matching zero only; a nan
    matches nothing): `errorloc.agrees`."""
    return agrees(value, reference)


def selection_reading(on_pi0_population: bool, on_pi3_population: bool) -> str:
    """Reading 1: the mean of `-Δsel(P)` has the sign of the mean `ΔU` on both
    populations, on exactly one, or on neither."""
    if on_pi0_population and on_pi3_population:
        return "carried_by_selection"
    if on_pi0_population or on_pi3_population:
        return "population_dependent"
    return "not_carried_by_selection"


def baseline_reading(ranker: Sequence[float], baseline: Sequence[float]) -> str:
    """Reading 4: "better" when the ranker's rate is lower than the baseline's
    in every seed, "worse" when higher in every seed, "mixed" otherwise (an equal
    rate or a nan in any seed is mixed). `mechanism.sign_reading` of the
    seed-paired differences."""
    if len(ranker) != len(baseline):
        raise ValueError("the ranker and the baseline have different seeds")
    reading = sign_reading([r - b for r, b in zip(ranker, baseline)])
    return {"consistent_loss": "better", "consistent_gain": "worse"}.get(reading, "mixed")


def window_contradicts(utility_reading: str, oriented_change: float) -> bool:
    """Reading 2's list: `ΔU_label` is consistent while the mean of `-Δown` has
    the other (non-zero) sign: `errorloc.contradicts`."""
    return contradicts(utility_reading, oriented_change)


def seed_signs(values: Sequence[float]) -> str:
    """One character per seed: +, - or 0, and n for a nan."""
    return "".join("n" if not math.isfinite(value) else "+" if value > 0
                   else "-" if value < 0 else "0" for value in values)
