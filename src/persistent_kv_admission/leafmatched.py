"""Leaf-matched horizon — the matched-horizon bit and the class order under leaf eligibility.

`docs/leaf-matched-horizon-plan.md` is the pre-registration. It reruns, under
the `leaf16` mechanism of the horizon control's Part 3 (`horizonctl`,
`docs/horizon-control-plan.md`), the arms that the horizon fill-in
(`horizonfill`) and the matched class-order control (`matchedorder`) ran under
`all16` at the cell's matched horizon `h*`, with `h*` carried over unchanged
(`matchedorder.MATCHED_HORIZON_SECONDS`; nothing here looks for a `leaf16`
horizon):

* `label_binary_{h*}`: the label rung's scorer at `h*`,
  `mechanism.ExactLabelScorer(trace, h*, target="binary")`, key
  `(1 if the next use is at most h* s away else 0, last_group)`, built by
  `horizonfill.arm_setup` exactly as under `all16`;
* `evict_binary_{h*}_learned` and `evict_binary_{h*}_recency`: admission by
  the frozen `next_use` ranker through `errorloc.HybridOverride`, eviction by
  `((reusable within h*, ranker score), last_group)` or
  `(reusable within h*, last_group)`, built by `matchedorder.arm_setup`
  exactly as under `all16`;
* `label`: the error-location `label` rung (`errorloc.arm_setup`), the
  reproduction anchor: under `leaf16` it is the published `leaf16` `label`
  row of the mechanism control.

Leaf eligibility is the store's alone. Every arm here is the object its
parent builds, unchanged; the runner passes `l2_eligibility="leaf"` and width
16 (`horizonctl.MECHANISMS[LEAF16]`) to `run_two_tier` and nothing else
differs. The store then restricts every decision set to leaf residents and
offers the arrival in its own first round only when the arrival is a leaf
(`arriving_index == 0`); a non-leaf arrival's first round has
`arriving_index == -1`, as every later round does. `HybridOverride` answers
only `arriving_index >= 0`, so the ranker's admission rule judges leaf
arrivals only and every other round keeps the store's choice, the class key's
first minimum; every index it returns is a candidate of the round it answers
(the horizon control's Part 3 paragraph). On a trace where no state has a
cached child the two eligibilities offer the same candidates in the same
draw order with the same generator draws, so the arms are their `all16`
selves decision by decision.

`matchedorder.ClassStatistics` at `h*` sits beside the error-location
statistics in every replay of the three `h*` arms (the plan's class check:
no resident eviction discards a state reusable within `h*` while a sampled
resident is not; no override outside a first round in which the arrival is a
candidate). For `label_binary_{h*}` both are zero by construction (its key's
first component is the class); the `label` anchor does not carry it, because
its key is the `next_use` label, under which a victim whose next use is
exactly 600 s away can tie with a state never reused (the label identities of
the error-location statistics already count those victims).

Reading helpers, pure arithmetic on numbers the replays produced:
`shortfall` and the 0.10 rule (`horizonctl.shortfall`,
`horizonctl.horizon_reading`, relabelled for one horizon), `recovery` and the
0.9 rule (`matchedorder.matched_recovery` / `matched_reading`), the seed-sign
readings (`mechanism.sign_counts` / `mechanism.sign_reading` through
`matchedorder.seed_signs`), the three prediction counts, and the share of
decisions in which the arrival is a candidate (`stat_decisions_admission /
stat_decisions` of the error-location statistics, the column the horizon
control's Reading 3 reported).

Nothing here fits, adds a feature or target, or changes an existing replay
path; every arm reads the trace's future on purpose and none is a proposed
policy. `leaf16` is a control on eligibility, not a proposed mechanism.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from . import errorloc, horizonfill, matchedorder
from .errorloc import ArmSetup
from .horizonctl import (
    LEAF16,
    MECHANISMS,
    SUFFICES_SHORTFALL,
    horizon_arm,
    horizon_reading,
    shortfall,
)
from .matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    MATCHED_READINGS,
    SIGN_READINGS,
    matched_horizon,
    matched_reading,
    matched_recovery,
    seed_signs,
)
from .trace import Trace

# --- the arms (docs/leaf-matched-horizon-plan.md, "Fixed surface") ---------------------------

MECHANISM = LEAF16
ELIGIBILITY, WIDTH = MECHANISMS[MECHANISM]
# The reproduction anchor: the error-location `label` rung, published under
# leaf16 by the mechanism control.
ANCHOR_ARM = "label"
# label_binary_h at every matched horizon (horizonfill builds all four).
RUNG_ARMS = tuple(horizon_arm(h) for h in MATCHED_HORIZONS_SECONDS)
RUNG_OF_HORIZON = dict(zip(MATCHED_HORIZONS_SECONDS, RUNG_ARMS))
CLASS_ORDER_ARMS = matchedorder.ARMS
ORDER_OF_ARM = matchedorder.ORDER_OF_ARM
HORIZON_OF_ARM = {**dict(zip(RUNG_ARMS, MATCHED_HORIZONS_SECONDS)), **matchedorder.HORIZON_OF_ARM}
ARMS = RUNG_ARMS + CLASS_ORDER_ARMS + (ANCHOR_ARM,)
FAMILIES = ("horizon", "class_order", "reproduction_anchor")
FAMILY_OF_ARM = {**dict.fromkeys(RUNG_ARMS, "horizon"),
                 **dict.fromkeys(CLASS_ORDER_ARMS, "class_order"),
                 ANCHOR_ARM: "reproduction_anchor"}
# The replays that carry `matchedorder.ClassStatistics` at h*: the three h* arms.
CLASS_STATISTIC_ARMS = RUNG_ARMS + CLASS_ORDER_ARMS
ARMS_PER_CELL = 4

assert (ELIGIBILITY, WIDTH) == ("leaf", 16)
assert RUNG_ARMS == ("label_binary_60", "label_binary_150", "label_binary_300", "label_binary_600")
assert all(arm in horizonfill.HORIZON_OF_ARM for arm in RUNG_ARMS)
assert len(ARMS) == 13 and len(set(ARMS)) == 13 and set(FAMILY_OF_ARM) == set(ARMS)


def cell_arms(fraction: float, multiplier: float) -> tuple[str, str, str, str]:
    """The four arms replayed at a cell: `label_binary_{h*}`,
    `evict_binary_{h*}_learned`, `evict_binary_{h*}_recency` and the `label`
    anchor."""
    h = matched_horizon(fraction, multiplier)
    learned, recency = matchedorder.cell_arms(fraction, multiplier)
    return RUNG_OF_HORIZON[h], learned, recency, ANCHOR_ARM


def arm_setup(arm: str, trace: Trace, horizon_seconds: float, seed: int,
              learned_ranker=None) -> ArmSetup:
    """The replay of `arm` on `trace`: the parent's own construction, which
    the store's leaf eligibility (applied by the caller) does not enter.

    `horizon_seconds` is the trace's label horizon (600 s on the real traces),
    read by the `label` anchor only; `learned_ranker` is the frozen pi0
    `next_use` ranker, needed by the class-order arms. A fresh scorer is built
    per call, because the history scorers carry state across a replay.
    """
    if arm in RUNG_ARMS:
        return horizonfill.arm_setup(arm, trace)
    if arm in CLASS_ORDER_ARMS:
        return matchedorder.arm_setup(arm, trace, learned_ranker)
    if arm == ANCHOR_ARM:
        return errorloc.arm_setup(arm, trace, horizon_seconds, seed, learned_ranker=learned_ranker)
    raise ValueError(f"unknown arm {arm!r}")


# --- the readings (docs/leaf-matched-horizon-plan.md, "Readings, fixed before the run") ------

assert SUFFICES_SHORTFALL == 0.10
# Reading 1: S_h* <= 0.10 (exactly 0.10 suffices; a nan falls short).
HORIZON_READINGS = ("reuse_label_at_matched_horizon_suffices",
                    "reuse_label_at_matched_horizon_falls_short")
_HORIZON_LABEL = {"reuse_label_suffices": HORIZON_READINGS[0],
                  "order_needed": HORIZON_READINGS[1]}
# Reading 2: matchedorder's labels of the 0.9 rule, so that the leaf16 and the
# published all16 readings carry the same names.
CLASS_READINGS = MATCHED_READINGS
# The trace x cell grid the three predictions count over.
GRID_TRACE_CELLS = 12
# Prediction 2 does not predict the reading at 0.25% x 1 (it is counted).
NOT_PREDICTED_CELLS = ((0.0025, 1.0),)
# Reading -> (the outcome counted, at least how many of the 12 trace x cell).
PREDICTIONS = {
    "1_horizon_under_leaf_eligibility": (HORIZON_READINGS[0], 12),
    "2_class_order_under_leaf_eligibility": (CLASS_READINGS[0], 10),
    "3_order_within_matched_class": ("consistent_loss", 8),
}
assert set(NOT_PREDICTED_CELLS) <= set(MATCHED_HORIZON_SECONDS)


def hstar_shortfall(u_label: float, u_rung: float, u_lru: float) -> float:
    """`S_h* = (U(label) - U(label_binary_h*)) / (U(label) - U(lru))`:
    `horizonctl.shortfall`, nan when the denominator is zero."""
    return shortfall(u_label, u_rung, u_lru)


def hstar_reading(s_hstar: float) -> str:
    """Reading 1 of one trace x cell: "reuse_label_at_matched_horizon_suffices"
    when S_h* is at most 0.10 (exactly 0.10 suffices),
    "reuse_label_at_matched_horizon_falls_short" otherwise, a nan included:
    `horizonctl.horizon_reading` on the one horizon, relabelled."""
    return _HORIZON_LABEL[horizon_reading({0.0: float(s_hstar)})[0]]


def class_recovery(u_arm: float, u_learned: float, u_evict_label: float) -> float:
    """`R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))`:
    `matchedorder.matched_recovery` (`horizonctl.recovery`)."""
    return matched_recovery(u_arm, u_learned, u_evict_label)


def class_reading(r_matched_learned: float) -> str:
    """Reading 2: `matchedorder.matched_reading`, the 0.9 rule (exactly 0.9
    suffices; a nan R is the rule's "otherwise")."""
    return matched_reading(r_matched_learned)


def predicted(reading: str, fraction: float, multiplier: float) -> bool:
    """Whether a prediction covers this cell's reading: every cell, except
    reading 2 at 0.25% x 1."""
    if reading not in PREDICTIONS:
        raise ValueError(f"no prediction for {reading!r}")
    cell = (float(fraction), float(multiplier))
    return not (reading == "2_class_order_under_leaf_eligibility" and cell in NOT_PREDICTED_CELLS)


def prediction_holds(reading: str, count: int, cells: int) -> bool | None:
    """Whether the count of the predicted outcome meets the prediction fixed
    before the run (12/12, at least 10/12, at least 8/12). None when the
    count is not over the 12 trace x cell of the grid: a prediction about 12
    is not evaluated on another number."""
    if reading not in PREDICTIONS:
        raise ValueError(f"no prediction for {reading!r}")
    if int(cells) != GRID_TRACE_CELLS:
        return None
    return int(count) >= PREDICTIONS[reading][1]


def outcome_counts(outcomes: Iterable[str], labels: tuple[str, ...]) -> dict[str, int]:
    """Counts of each label over trace x cell; an unknown label is refused."""
    counts = dict.fromkeys(labels, 0)
    for outcome in outcomes:
        if outcome not in counts:
            raise ValueError(f"unknown reading {outcome!r}")
        counts[outcome] += 1
    return counts


def arrival_candidate_share(decisions_admission, decisions) -> float:
    """Share of the evaluation-window decisions in which the arrival is a
    candidate: `stat_decisions_admission / stat_decisions` of the
    error-location statistics, nan without a decision."""
    decisions = int(decisions)
    return int(decisions_admission) / decisions if decisions else math.nan


def row_arrival_candidate_share(row: Mapping) -> float:
    """`arrival_candidate_share` of one replay row (or published row)."""
    return arrival_candidate_share(row["stat_decisions_admission"], row["stat_decisions"])

