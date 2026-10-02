"""Horizon and class-order control — is a reuse label enough, at which horizon, and with whose order.

`docs/horizon-control-plan.md` is the pre-registration. It builds on the
error-location control (`errorloc`, `docs/error-location-plan.md`) and rides on
the same hooks, reused as they are:

* Part 1, mechanism `all16`: `label_binary_h` for h in {6, 15, 60, 300, 600}
  seconds, the exact `binary` target at horizon h as
  `decisionpop.target_column` defines it (1 if the next use is at most h
  seconds away, else 0), key `(binary_h, last_group)`. It is the label rung's
  scorer, `mechanism.ExactLabelScorer(trace, h, target="binary")`, at h; h =
  600 is the published `label_binary` arm through the new parameter.
* Part 2, mechanism `all16`: `evict_binary_learned` and
  `evict_binary_recency`. Both are "admission by X, eviction by Y" in the
  error-location sense, with X the frozen `next_use` ranker and Y a composite
  key, so `errorloc.HybridOverride` applies the plan's rule unchanged: in a
  first-round decision in which the arrival is a candidate, the arrival is
  rejected if and only if it is the first minimum of the ranker's keys
  `(ranker score, last_group)` over the candidates; otherwise the victim is
  the first minimum of Y's keys (the store's) over the candidates other than
  the arrival; every later round keeps the store's choice, which is Y's. Y is
  the store's own scorer:
  - `evict_binary_recency`: the exact 600-second reuse label itself
    (`ExactLabelScorer(trace, 600, "binary")`), so the store's key is
    `(reusable within 600 s, last_group)`; the ranker is shown the history
    through `errorloc.HybridScorer`, as in the error-location hybrids.
  - `evict_binary_learned`: `ReuseClassScorer`, whose score is the pair
    `(reusable within 600 s, ranker score)`, so the store's key
    `((reusable, ranker score), last_group)` orders exactly as the composite
    `(reusable, ranker score, last_group)`. The ranker inside it is the same
    object `HybridOverride` consults, observed once, through the composite.
  Both components are read at the decision instant: the store scores every
  candidate when it builds the decision set, and the override scores X at the
  same timestamp, before anything is removed.
* Part 3, mechanism `leaf16`: `adm_label` and `evict_label` exactly as
  `errorloc.arm_setup` builds them; only the store's eligibility changes. Under
  leaf eligibility the arrival is a candidate of its first round only when it
  is a leaf (`arriving_index == 0`); otherwise the store passes
  `arriving_index == -1` and `HybridOverride` keeps the store's (Y's) choice,
  as it does in every later round, so every index it returns is a candidate
  of the round it answers.

The identity arms of the plan's checks are built the same way: admission by
the exact 600-second reuse label with `evict_binary_recency`'s eviction (a
second, separate `ExactLabelScorer` in X's place) must be the
`label_binary_600` rung, and under `leaf16` the X/X hybrids of `errorloc` must
be the `learned` and `label` rungs, decision by decision.

Nothing here fits, adds a feature or target, or changes an existing replay
path; every constructed arm reads the trace's future on purpose and none is a
proposed policy. The reading helpers at the end are pure arithmetic on numbers
the replays produced, kept here so that they can be tested; the seed-sign
readings are `mechanism.sign_counts` / `mechanism.sign_reading` and the
location rule is `errorloc.location_label`, used as they are.
"""

from __future__ import annotations

import math
from typing import Mapping

from . import errorloc
from .decisionpop import RawFeatureScorer
from .errorloc import ArmSetup, HybridOverride, HybridScorer, location_label
from .mechanism import ExactLabelScorer
from .trace import Request, Trace

# --- the arms (docs/horizon-control-plan.md, "Fixed surface") ---------------------------------

BINARY_TARGET = errorloc.BINARY_TARGET            # "binary"
# Part 1: the horizon grid, in seconds. 600 is the published `label_binary`.
HORIZONS_SECONDS = (6.0, 15.0, 60.0, 300.0, 600.0)
# Part 2: "reusable within 600 s", fixed by the plan whatever the trace's own
# label horizon (which the runner also requires to be 600 s).
REUSE_HORIZON_SECONDS = 600.0


def horizon_arm(horizon_seconds: float) -> str:
    return f"label_binary_{horizon_seconds:g}"


HORIZON_ARMS = tuple(horizon_arm(h) for h in HORIZONS_SECONDS)
HORIZON_OF_ARM = dict(zip(HORIZON_ARMS, HORIZONS_SECONDS))
# The h = 600 rung, which must reproduce the published `label_binary` rows.
PUBLISHED_HORIZON_ARM = horizon_arm(REUSE_HORIZON_SECONDS)
PUBLISHED_LABEL_BINARY_ARM = errorloc.LABEL_BINARY_ARM     # "label_binary"
# Part 2: arm -> how residents are ordered within a reuse class.
CLASS_ORDER = {"evict_binary_learned": "learned", "evict_binary_recency": "recency"}
CLASS_ORDER_ARMS = tuple(CLASS_ORDER)
# Part 3: the error-location hybrids, under leaf eligibility.
LEAF_ARMS = errorloc.HYBRID_ARMS                  # ("adm_label", "evict_label")
# The nine arms, in table and figure order.
ARMS = HORIZON_ARMS + CLASS_ORDER_ARMS + LEAF_ARMS

# Mechanisms: label -> (l2_eligibility, l2_sample_width).
ALL16, LEAF16 = "all16", "leaf16"
MECHANISMS = {ALL16: ("all", 16), LEAF16: ("leaf", 16)}

# Identities of the plan's required checks: identity arm -> the rung it must
# equal decision by decision, under the identity arm's own mechanism.
BINARY_IDENTITY_ARM = "adm_label_binary_evict_binary_recency"
IDENTITY_ARMS = {
    BINARY_IDENTITY_ARM: PUBLISHED_HORIZON_ARM,
    "hybrid_learned_learned": "learned",
    "hybrid_label_label": "label",
}
# The leaf16 rungs the leaf identities are compared with. They are published
# rows of the mechanism control and are not part of the grid; the smoke
# replays them, so that the comparison can be decision by decision.
IDENTITY_REFERENCE_ARMS = ("learned", "label")
ARM_MECHANISMS = {
    **{arm: ALL16 for arm in HORIZON_ARMS + CLASS_ORDER_ARMS + (BINARY_IDENTITY_ARM,)},
    **{arm: LEAF16 for arm in LEAF_ARMS + ("hybrid_learned_learned", "hybrid_label_label")
       + IDENTITY_REFERENCE_ARMS},
}
assert len(ARMS) == 9 and len(set(ARMS)) == 9
assert set(ARM_MECHANISMS) == set(ARMS) | set(IDENTITY_ARMS) | set(IDENTITY_REFERENCE_ARMS)
assert all(ARM_MECHANISMS[arm] == ARM_MECHANISMS[reference]
           for arm, reference in IDENTITY_ARMS.items())


def arm_mechanism(arm: str) -> tuple[str, str, int]:
    """(mechanism label, l2_eligibility, l2_sample_width) of an arm."""
    try:
        mechanism = ARM_MECHANISMS[arm]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}") from None
    eligibility, width = MECHANISMS[mechanism]
    return mechanism, eligibility, width


class ReuseClassScorer:
    """The store's scorer of `evict_binary_learned`.

    The score is the pair `(reuse.score, within.score)`: the exact reuse label
    first (0 or 1, from `ExactLabelScorer(..., "binary")`), the ranker's score
    second. The store appends `last_group`, and tuples compare element by
    element, so its key `((reusable, ranker score), last_group)` orders the
    candidates exactly as the plan's composite `(reusable, ranker score,
    last_group)`: every non-reusable state below every reusable one, the
    ranker's order within a class, recency on a tie. Each distinct component is
    shown every observation once, so a history ranker inside it sees exactly
    what it would see as the store's own scorer.
    """

    time_varying = True

    def __init__(self, reuse, within) -> None:
        if reuse is within:
            raise ValueError("the reuse label and the within-class score must be distinct scorers")
        self.reuse = reuse
        self.within = within

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.reuse.observe(requests, timestamp_ms)
        self.within.observe(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> tuple[float, float]:
        return (self.reuse.score(state_id, timestamp_ms),
                self.within.score(state_id, timestamp_ms))


def reuse_label(trace: Trace) -> ExactLabelScorer:
    """The exact 600-second reuse label of Part 2: 1 if the next use is at most
    600 s away, else 0 (`decisionpop.target_column`, target `binary`)."""
    return ExactLabelScorer(trace, REUSE_HORIZON_SECONDS, target=BINARY_TARGET)


def class_order_setup(arm: str, trace: Trace, admission, order: str) -> ArmSetup:
    """"Admission by `admission`, eviction by the composite key" of Part 2.

    `order` is how residents are ordered within a reuse class: "learned" (the
    ranker's score, which must then be `admission` itself: the frozen ranker,
    observed once through the composite) or "recency" (`last_group` alone).
    """
    if order == "learned":
        scorer = ReuseClassScorer(reuse_label(trace), admission)
    elif order == "recency":
        scorer = HybridScorer(admission, reuse_label(trace))
    else:
        raise ValueError(f"unknown class order {order!r}")
    return ArmSetup(arm, "learned", scorer=scorer, override=HybridOverride(admission))


def arm_setup(arm: str, trace: Trace, horizon_seconds: float, seed: int,
              learned_ranker=None) -> ArmSetup:
    """The replay of `arm` on `trace` at sampling seed `seed`; the mechanism is
    `arm_mechanism(arm)`, applied by the caller.

    `horizon_seconds` is the trace's label horizon, used by the `next_use`
    label of the Part 3 hybrids and their rungs (600 s on the real traces); the
    Part 1 horizons and the Part 2 reuse horizon are the plan's, whatever it
    is. `learned_ranker` is the frozen pi0 `next_use` ranker of the trace. A
    fresh scorer is built per call, because the history scorers carry state
    across a replay.
    """
    if arm in HORIZON_OF_ARM:
        return ArmSetup(arm, "learned", scorer=ExactLabelScorer(trace, HORIZON_OF_ARM[arm],
                                                                target=BINARY_TARGET))
    if arm in CLASS_ORDER:
        if learned_ranker is None:
            raise ValueError(f"{arm} needs the learned ranker")
        return class_order_setup(arm, trace, RawFeatureScorer(trace, learned_ranker),
                                 CLASS_ORDER[arm])
    if arm == BINARY_IDENTITY_ARM:
        return class_order_setup(arm, trace, reuse_label(trace), "recency")
    if arm in ARM_MECHANISMS:
        # adm_label / evict_label, the X/X hybrids and the rungs, as built for
        # the error-location control.
        return errorloc.arm_setup(arm, trace, horizon_seconds, seed, learned_ranker=learned_ranker)
    raise ValueError(f"unknown arm {arm!r}")


# --- the readings (docs/horizon-control-plan.md, "Readings, fixed before the run") ------------

SUFFICES_SHORTFALL = 0.10
# Reading 1 is also counted over the trace x cell whose published S_600
# exceeds this.
PUBLISHED_SUBSET_SHORTFALL = 0.05
HORIZON_READINGS = ("reuse_label_suffices", "order_needed")
CLASS_ORDER_SHARE = 0.9
CLASS_ORDER_READINGS = ("reuse_identification_suffices", "ranker_order_costs")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def shortfall(u_label: float, u_arm: float, u_lru: float) -> float:
    """`S = (U(label) - U(arm)) / (U(label) - U(lru))`: the share of the
    label's gain over sampled LRU that the arm does not reach (0 at the label,
    1 at LRU), nan when the denominator is zero."""
    return _ratio(u_label - u_arm, u_label - u_lru)


def horizon_reading(shortfalls: Mapping[float, float]) -> tuple[str, float, tuple[float, ...]]:
    """Reading 1 of one trace x cell, from `horizon -> S_h` on five-seed means.

    "reuse_label_suffices" when the smallest S_h is at most 0.10 (exactly 0.10
    suffices), "order_needed" otherwise. Returns the label, the smallest S_h
    and every horizon attaining it, ascending (several when they tie exactly).
    A nan S_h (zero denominator) attains nothing; when every S_h is nan the
    smallest is nan, no horizon attains it, and the reading is "order_needed",
    the rule's "otherwise".
    """
    finite = {float(h): float(s) for h, s in shortfalls.items() if math.isfinite(s)}
    if not finite:
        return "order_needed", math.nan, ()
    smallest = min(finite.values())
    attaining = tuple(sorted(h for h, s in finite.items() if s == smallest))
    label = "reuse_label_suffices" if smallest <= SUFFICES_SHORTFALL else "order_needed"
    return label, smallest, attaining


def in_published_subset(published_s600: float) -> bool:
    """Whether a trace x cell belongs to the separately counted subset:
    published S_600 strictly above 0.05 (a nan belongs to nothing)."""
    return math.isfinite(published_s600) and published_s600 > PUBLISHED_SUBSET_SHORTFALL


def recovery(u_arm: float, u_learned: float, u_evict_label: float) -> float:
    """`R(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))`, nan when
    the denominator is zero."""
    return _ratio(u_arm - u_learned, u_evict_label - u_learned)


def class_order_reading(r_evict_binary_learned: float) -> str:
    """Reading 2: "reuse_identification_suffices" when R(evict_binary_learned)
    is at least 0.9 (exactly 0.9 suffices), "ranker_order_costs" otherwise
    (a nan R, from a zero denominator, is the rule's "otherwise")."""
    if math.isfinite(r_evict_binary_learned) and r_evict_binary_learned >= CLASS_ORDER_SHARE:
        return "reuse_identification_suffices"
    return "ranker_order_costs"


def leaf_location(g: float, a: float, e: float, published_all16: str) -> tuple[str, bool]:
    """Reading 3: the location label of `G`, `A`, `E` under `leaf16` by the
    error-location rule (`errorloc.location_label`), and whether it is the
    published `all16` label of the same trace x cell."""
    if published_all16 not in errorloc.LOCATIONS:
        raise ValueError(f"unknown published location {published_all16!r}")
    label = location_label(g, a, e)
    return label, label == published_all16
