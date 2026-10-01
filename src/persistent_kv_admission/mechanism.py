"""Mechanism control — a fixed score ladder under four L2 eviction mechanisms.

`docs/mechanism-control-plan.md` is the pre-registration. The learned L2 arms
of Phase 0.97 and later rank through one mechanism (the arrival plus up to 16
uniformly sampled residents, any of which may leave), and the published
headroom denominator compares that mechanism's LRU with a heap comparator that
scores every resident with the exact next use. This phase holds the score
fixed at four rungs — LRU, the frozen learned ranker, the exact label that
ranker was fitted to, and the offline comparator's own key — and varies only
the mechanism that applies it (`twotier.L2_ELIGIBILITIES` x sample width), so
the published headroom splits into the Phase 0.75 gaps:

    T_m = [H_off - U_m(offline)]        candidate-search gap
        + [U_m(offline) - U_m(label)]   objective gap
        + [U_m(label) - U_m(learned)]   signal gap
        + [U_m(learned) - U_m(lru)]     achieved part

with `H_off` the heap offline reference and `U_m(s)` the extra avoided tokens
of rung `s` under mechanism `m`. Nothing here fits, adds a feature, or changes
a policy: the label rung reads the trace's future on purpose, as the perfect
predictor of the training target, and the helpers below are pure arithmetic
on numbers the replays produced, kept here so that they can be tested.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .crossworkload import TARGETS
from .decisionpop import _Labeller, target_column
from .trace import Request, Trace

# The score ladder, from the weakest to the strongest score, in the order the
# plan states the ladder inequality `lru <= learned <= label <= offline`.
RUNGS = ("lru", "learned", "label", "offline")
# The four terms of the telescoping identity, from the top of the ladder down.
GAPS = ("candidate_search", "objective", "signal", "achieved")
# The reading attached to the term holding at least half of T. "achieved" is
# not a gap, but when the learned arm itself has closed half of the headroom
# no gap can be the dominant one and the label says so instead of "mixed".
DOMINANT_LABELS = {
    "candidate_search": "mechanism_bound",
    "objective": "objective_bound",
    "signal": "signal_bound",
    "achieved": "achieved",
}
DOMINANCE_SHARE = 0.5
# Adjacent pairs of the ladder inequality, lower rung first.
LADDER_PAIRS = (("lru", "learned"), ("learned", "label"), ("label", "offline"))


class ExactLabelScorer:
    """The exact training target of a state at the decision instant.

    The `label` rung: a perfect predictor of what the frozen `next_use` ranker
    was fitted to, and nothing more. The next-use delta is read exactly as
    `decisionpop._Labeller` reads it for training rows (occurrences at the
    current timestamp excluded, `math.inf` when there is none) and turned into
    the target by `decisionpop.target_column` itself, so the score is the
    training label by construction rather than a re-derivation of it. For
    `next_use` that is `-log1p(min(delta_s, H))`: a state without a reuse
    within H scores the clipped value, and recency (the store's second key)
    orders those among themselves, as it does for the learned arm.

    The score reads the future and ignores history, so `observe` does nothing.
    """

    time_varying = True

    def __init__(self, trace: Trace, horizon_seconds: float, target: str = "next_use") -> None:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}")
        self.label = _Labeller(trace, horizon_seconds)
        self.horizon_seconds = float(horizon_seconds)
        self.target = target

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        return None

    def score(self, state_id: str, timestamp_ms: float) -> float:
        delta, count = self.label(state_id, timestamp_ms)
        value = target_column(
            np.asarray([delta], dtype=float), np.asarray([count], dtype=float),
            self.target, self.horizon_seconds,
        )
        return float(value[0])


# --- the decomposition --------------------------------------------------------


def decompose(utilities: Mapping[str, float], h_off: float) -> dict[str, object]:
    """The telescoping split of one mechanism's headroom `T = H_off - U(lru)`.

    `utilities` maps every rung of `RUNGS` to its utility. Given integer token
    counts the four terms sum to `T` exactly, which is how the runner calls it
    (it converts to input-token points afterwards); given floats the identity
    holds up to rounding. Shares are of `T` and nan when `T` is zero.
    """
    missing = [rung for rung in RUNGS if rung not in utilities]
    if missing:
        raise ValueError(f"utilities lack the rungs {missing}")
    u = {rung: utilities[rung] for rung in RUNGS}
    gaps = {
        "candidate_search": h_off - u["offline"],
        "objective": u["offline"] - u["label"],
        "signal": u["label"] - u["learned"],
        "achieved": u["learned"] - u["lru"],
    }
    total = h_off - u["lru"]
    shares = {
        name: (gaps[name] / total if total != 0 else math.nan) for name in GAPS
    }
    return {
        "gaps": gaps,
        "total": total,
        "shares": shares,
        "dominant": dominant_label(gaps, total),
    }


def dominant_label(gaps: Mapping[str, float], total: float) -> str:
    """The pre-registered reading of one decomposition.

    When `T > 0`, the term holding at least half of `T` names the reading
    (`DOMINANT_LABELS`); a negative term can let two terms clear the bar at
    once, and then no single one dominates, so the reading is "mixed", as it
    is when none clears it. When `T <= 0` (or is not finite) there is no
    headroom to hold a share of, and the reading is "mixed" too; the table
    carries `T`, so that case stays visible.
    """
    if not (math.isfinite(total) and total > 0):
        return "mixed"
    holding = [name for name in GAPS if gaps[name] >= DOMINANCE_SHARE * total]
    if len(holding) == 1:
        return DOMINANT_LABELS[holding[0]]
    return "mixed"


def pair_name(lower: str, upper: str) -> str:
    return f"{lower}_le_{upper}"


def ladder_holds(utilities: Mapping[str, float]) -> dict[str, bool]:
    """Whether each adjacent pair of the ladder holds, lower rung <= upper rung."""
    return {
        pair_name(lower, upper): bool(utilities[lower] <= utilities[upper])
        for lower, upper in LADDER_PAIRS
    }


def ladder_ordered(per_seed: Sequence[Mapping[str, float]]) -> bool:
    """`lru <= learned <= label <= offline` in every seed given (and at least one)."""
    if not per_seed:
        return False
    return all(all(ladder_holds(utilities).values()) for utilities in per_seed)


def sign_counts(values: Sequence[float]) -> tuple[int, int, int]:
    """(positive, zero, negative) counts of seed-paired values."""
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    return positive, len(values) - positive - negative, negative


def sign_reading(values: Sequence[float]) -> str:
    """"consistent_gain" when every seed is > 0, "consistent_loss" when every
    seed is < 0, "mixed" otherwise (a zero breaks consistency, as does no seed)."""
    positive, _, negative = sign_counts(values)
    if values and positive == len(values):
        return "consistent_gain"
    if values and negative == len(values):
        return "consistent_loss"
    return "mixed"
