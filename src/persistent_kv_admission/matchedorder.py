"""Matched class order — the frozen ranker given the exact reuse class at the horizon that works for the cell.

`docs/matched-class-order-plan.md` is the pre-registration. It is a follow-up
of Part 2 of the horizon and class-order control (`horizonctl`,
`docs/horizon-control-plan.md`) and of its fill-in (`horizonfill`,
`docs/horizon-fill-plan.md`), and rides on Part 2's arms with one parameter
changed: the reuse horizon of the class, 600 s there, is here the horizon
`h*` that works for the cell.

Reused unchanged:

* the two arms' construction, `horizonctl.class_order_setup` with the exact
  reuse label at `h` in place of the 600-second one: admission by the frozen
  `next_use` ranker through `errorloc.HybridOverride`; eviction by the store's
  key `((reusable within h, ranker score), last_group)` through
  `horizonctl.ReuseClassScorer` for the learned order, and
  `(reusable within h, last_group)` through `errorloc.HybridScorer` for
  recency. The reuse label is the label rung's scorer,
  `mechanism.ExactLabelScorer(trace, h, target="binary")` (1 if the next use
  is at most h seconds away, else 0), exactly as `horizonctl` and
  `horizonfill` build it for their horizon arms;
* `ArmSetup`, the frozen ranker as `decisionpop.RawFeatureScorer`, the
  recovery share `horizonctl.recovery`, the 0.9 rule of
  `horizonctl.class_order_reading`, and the seed-sign readings
  `mechanism.sign_counts` / `mechanism.sign_reading`.

New here:

* `MATCHED_HORIZON_SECONDS`, the plan's table of `h*` by cell, the same on
  both traces, and the arm names `evict_binary_{h}_learned` /
  `evict_binary_{h}_recency`. At h = 600 the arm *is* the published arm under
  another name: `evict_binary_600_learned` is `evict_binary_learned` and
  `evict_binary_600_recency` is `evict_binary_recency`
  (`PUBLISHED_ARMS`); the runner reruns them as a reproduction check.
* `ClassStatistics`, a read-only recorder for the plan's check that no
  resident eviction discards a state reusable within `h*` while a sampled
  resident is not, and that every overridden decision is a first-round
  decision in which the arrival is a candidate. It sits in the override slot
  through `errorloc.RecordingOverride`, unchanged, beside the error-location
  `DecisionStatistics`, which stays at the trace's label horizon (600 s) as in
  Part 2. "Reusable within h" is `DecisionStatistics`'s own rule constructed
  at h: next-use delta (`decisionpop._Labeller`, occurrences at the decision
  instant excluded) at most `h * 1000` ms; restricted to resident-only
  decisions of the evaluation window, its count is `DecisionStatistics`'s
  `m4_count_resident` at the same horizon. `DecisionStatistics` itself is not
  constructed at `h*`, because its horizon also sets the reference key `K*`
  of m1-m3, which Part 2 fixed at 600 s.
* Reading helpers: reading 1 relabels `class_order_reading` and counts the
  trace x cell with `h* < 600` ("new") and `h* = 600` ("reproduced")
  separately, never merged; readings 2-4 are seed-paired sign readings.

The plan's text counts "six" trace x cell with `h* < 600` and "six" (60
replays) with `h* = 600`. By its own table `h* < 600` holds at 0.25%x1,
0.25%x4, 1%x1 and 2%x1 on both traces (eight) and `h* = 600` at 1%x4 and 2%x4
(four, 40 replays). Nothing here hard-codes either count: every count is
computed from `MATCHED_HORIZON_SECONDS` and reported with its denominator.

Nothing here fits, adds a feature or target, or changes an existing replay
path; every arm reads the trace's future on purpose and none is a proposed
policy. `h*` is the best of grids run on these traces, read after the fact.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .decisionpop import RawFeatureScorer
from .errorloc import ArmSetup, HybridOverride, HybridScorer
from .horizonctl import (
    ALL16,
    BINARY_TARGET,
    CLASS_ORDER,
    CLASS_ORDER_READINGS,
    CLASS_ORDER_SHARE,
    REUSE_HORIZON_SECONDS,
    ReuseClassScorer,
    class_order_reading,
    horizon_arm,
    recovery,
)
from .mechanism import ExactLabelScorer, sign_counts, sign_reading
from .trace import Trace

# --- the arms (docs/matched-class-order-plan.md, "Fixed surface") ----------------------------

MECHANISM = ALL16
# The plan's table: (l1_fraction, l2_multiplier) -> h*, in seconds, the same on
# both traces (the horizon of the grids run that minimises S_h).
MATCHED_HORIZON_SECONDS = {
    (0.0025, 1.0): 60.0,
    (0.0025, 4.0): 150.0,
    (0.01, 1.0): 150.0,
    (0.01, 4.0): 600.0,
    (0.02, 1.0): 300.0,
    (0.02, 4.0): 600.0,
}
MATCHED_HORIZONS_SECONDS = tuple(sorted(set(MATCHED_HORIZON_SECONDS.values())))
# The published arms' reuse horizon: at h* = 600 the arm is the published one.
PUBLISHED_HORIZON_SECONDS = REUSE_HORIZON_SECONDS          # 600.0
# Within a reuse class: the ranker's score, or recency alone.
ORDERS = ("learned", "recency")
# Reading 1 counts the two roles separately and never merges them.
ROLES = ("new", "reproduced")


def matched_arm(horizon_seconds: float, order: str) -> str:
    if order not in ORDERS:
        raise ValueError(f"unknown class order {order!r}")
    return f"evict_binary_{float(horizon_seconds):g}_{order}"


ARMS = tuple(matched_arm(h, order) for h in MATCHED_HORIZONS_SECONDS for order in ORDERS)
HORIZON_OF_ARM = {matched_arm(h, order): h for h in MATCHED_HORIZONS_SECONDS for order in ORDERS}
ORDER_OF_ARM = {matched_arm(h, order): order for h in MATCHED_HORIZONS_SECONDS for order in ORDERS}
# h = 600: the published arm under another name (the reproduction targets).
PUBLISHED_ARMS = {matched_arm(PUBLISHED_HORIZON_SECONDS, order): published
                  for published, order in CLASS_ORDER.items()}
# The label rung each arm's recency counterpart is compared with (reading 3).
RUNG_OF_HORIZON = {h: horizon_arm(h) for h in MATCHED_HORIZONS_SECONDS}

assert MATCHED_HORIZONS_SECONDS == (60.0, 150.0, 300.0, 600.0)
assert PUBLISHED_HORIZON_SECONDS == 600.0
assert PUBLISHED_ARMS == {"evict_binary_600_learned": "evict_binary_learned",
                          "evict_binary_600_recency": "evict_binary_recency"}
assert len(ARMS) == 8 and len(set(ARMS)) == 8


def matched_horizon(fraction: float, multiplier: float) -> float:
    """h* of a cell, in seconds."""
    try:
        return MATCHED_HORIZON_SECONDS[(float(fraction), float(multiplier))]
    except KeyError:
        raise ValueError(f"no matched horizon for cell {(fraction, multiplier)!r}") from None


def cell_arms(fraction: float, multiplier: float) -> tuple[str, str]:
    """The two arms replayed at a cell: (learned order, recency) at its h*."""
    h = matched_horizon(fraction, multiplier)
    return tuple(matched_arm(h, order) for order in ORDERS)


def role_of_horizon(horizon_seconds: float) -> str:
    """"new" below 600 s, "reproduced" at 600 s (the published arm rerun)."""
    h = float(horizon_seconds)
    if h not in MATCHED_HORIZONS_SECONDS:
        raise ValueError(f"{h:g} s is not a matched horizon")
    return "reproduced" if h == PUBLISHED_HORIZON_SECONDS else "new"


def reuse_label(trace: Trace, horizon_seconds: float) -> ExactLabelScorer:
    """The exact reuse label at h: 1 if the next use is at most h seconds away,
    else 0 (`decisionpop.target_column`, target `binary`); at h = 600 it is
    `horizonctl.reuse_label`."""
    return ExactLabelScorer(trace, float(horizon_seconds), target=BINARY_TARGET)


def class_order_setup(arm: str, trace: Trace, admission, order: str,
                      horizon_seconds: float) -> ArmSetup:
    """`horizonctl.class_order_setup` with the exact reuse label at
    `horizon_seconds` in place of the 600-second one: "admission by
    `admission`, eviction by the composite key".

    `order` is how residents are ordered within a reuse class: "learned" (the
    ranker's score, which must then be `admission` itself, observed once
    through the composite) or "recency" (`last_group` alone). The scorer and
    override classes are Part 2's, unchanged.
    """
    if order == "learned":
        scorer = ReuseClassScorer(reuse_label(trace, horizon_seconds), admission)
    elif order == "recency":
        scorer = HybridScorer(admission, reuse_label(trace, horizon_seconds))
    else:
        raise ValueError(f"unknown class order {order!r}")
    return ArmSetup(arm, "learned", scorer=scorer, override=HybridOverride(admission))


def arm_setup(arm: str, trace: Trace, learned_ranker) -> ArmSetup:
    """The replay of a matched arm on `trace` under `all16` (applied by the
    caller): the frozen pi0 `next_use` ranker `learned_ranker` as
    `RawFeatureScorer`, as `horizonctl.arm_setup` builds Part 2's arms. A fresh
    scorer is built per call, because the history scorers carry state across
    a replay."""
    if arm not in HORIZON_OF_ARM:
        raise ValueError(f"unknown arm {arm!r}")
    if learned_ranker is None:
        raise ValueError(f"{arm} needs the learned ranker")
    return class_order_setup(arm, trace, RawFeatureScorer(trace, learned_ranker),
                             ORDER_OF_ARM[arm], HORIZON_OF_ARM[arm])


def identity_arm(horizon_seconds: float) -> str:
    return f"adm_label_binary_{float(horizon_seconds):g}_evict_binary_{float(horizon_seconds):g}_recency"


def identity_setup(trace: Trace, horizon_seconds: float) -> ArmSetup:
    """The plan's identity arm at h: admission by the exact h-second reuse
    label (a second, separate `ExactLabelScorer` in X's place) with the
    recency arm's eviction. It must be the `label_binary_h` rung decision by
    decision, as `horizonctl`'s identity arm is at 600 s."""
    return class_order_setup(identity_arm(horizon_seconds), trace,
                             reuse_label(trace, horizon_seconds), "recency", horizon_seconds)


# --- the class statistic (docs/matched-class-order-plan.md, "Required checks") -----------------

CLASS_SCOPES = ("seen", "window")
CLASS_COUNTS = ("decisions", "rejections", "resident_evictions", "violations", "overridden",
                "overridden_outside_admission")


class ClassStatistics:
    """Resident evictions against the exact reuse class at `horizon_seconds`.

    A recorder for `errorloc.RecordingOverride` (which calls `attach(store)`
    and `record(...)` with the store's and the final victim). For every
    decision of the replay ("seen", warm-up included) and for those at or
    after `measure_from_ms` ("window") it counts:

    * `decisions`, and `rejections`: first-round decisions whose final victim
      is the arrival;
    * `resident_evictions`: every other decision (the victim is a resident:
      a sampled resident of a first round, or any candidate of a later round,
      the admitted arrival included);
    * `violations`: resident evictions whose victim is reusable within h while
      some sampled resident (a candidate other than the arrival of a first
      round) is not;
    * `overridden`: decisions whose final victim differs from the store's,
      and `overridden_outside_admission`: those whose arrival is not a
      candidate (`arriving_index < 0`, a later round under `all16`).

    `m4_count_resident` is `violations` over the window's resident-only
    decisions (`arriving_index < 0`): `DecisionStatistics`'s
    `m4_count_resident` constructed at the same horizon.

    "Reusable within h" is `DecisionStatistics`'s rule: the next-use delta
    `ExactLabelScorer(trace, h, "binary").label` returns (the label rung's
    labeller) at most `h * 1000` ms, which is that scorer's own score. It
    reads only its arguments and the trace; it writes nothing the replay
    reads.
    """

    def __init__(self, trace: Trace, horizon_seconds: float,
                 measure_from_ms: float | None) -> None:
        self.horizon_seconds = float(horizon_seconds)
        self.reference = reuse_label(trace, self.horizon_seconds)
        # Exactly `DecisionStatistics`'s horizon_ms and `target_column`'s
        # binary threshold: delta <= h * 1000.
        self.horizon_ms = self.horizon_seconds * 1000.0
        self.measure_from_ms = measure_from_ms
        self.counts = {scope: dict.fromkeys(CLASS_COUNTS, 0) for scope in CLASS_SCOPES}
        self.m4_count_resident = 0

    def attach(self, store) -> None:
        """Nothing is read from the store; present for `RecordingOverride`."""
        return None

    def reusable(self, state_id: str, timestamp_ms: float) -> bool:
        return self.reference.label(state_id, timestamp_ms)[0] <= self.horizon_ms

    def record(self, candidates, keys, original_index: int, victim_index: int,
               timestamp_ms: float, group_index: int, arriving_index: int) -> None:
        rejected = arriving_index >= 0 and victim_index == arriving_index
        overridden = victim_index != original_index
        violation = False
        if not rejected:
            victim_reusable = self.reusable(candidates[victim_index], timestamp_ms)
            if victim_reusable:
                violation = any(not self.reusable(state_id, timestamp_ms)
                                for index, state_id in enumerate(candidates)
                                if index != arriving_index and index != victim_index)
        scopes = ["seen"]
        if self.measure_from_ms is None or timestamp_ms >= self.measure_from_ms:
            scopes.append("window")
            if arriving_index < 0 and violation:
                self.m4_count_resident += 1
        for scope in scopes:
            counts = self.counts[scope]
            counts["decisions"] += 1
            counts["rejections"] += rejected
            counts["resident_evictions"] += not rejected
            counts["violations"] += violation
            counts["overridden"] += overridden
            counts["overridden_outside_admission"] += overridden and arriving_index < 0

    def row(self) -> dict[str, object]:
        out: dict[str, object] = {"class_horizon_seconds": self.horizon_seconds}
        for scope in CLASS_SCOPES:
            suffix = "_seen" if scope == "seen" else ""
            for name in CLASS_COUNTS:
                out[f"class_{name}{suffix}"] = self.counts[scope][name]
        out["class_m4_count_resident"] = self.m4_count_resident
        return out


CLASS_COLUMNS = (("class_horizon_seconds",)
                 + tuple(f"class_{name}{'_seen' if scope == 'seen' else ''}"
                         for scope in CLASS_SCOPES for name in CLASS_COUNTS)
                 + ("class_m4_count_resident",))


# --- the readings (docs/matched-class-order-plan.md, "Readings, fixed before the run") -------

MATCHED_READINGS = ("reuse_identification_at_matched_horizon_suffices",
                    "ranker_order_costs_at_matched_horizon")
_MATCHED_LABEL = dict(zip(CLASS_ORDER_READINGS, MATCHED_READINGS))
SIGN_READINGS = ("consistent_gain", "consistent_loss", "mixed")
assert CLASS_ORDER_SHARE == 0.9


def matched_recovery(u_arm: float, u_learned: float, u_evict_label: float) -> float:
    """`R_h*(a) = (U(a) - U(learned)) / (U(evict_label) - U(learned))`:
    `horizonctl.recovery`, nan when the denominator is zero."""
    return recovery(u_arm, u_learned, u_evict_label)


def matched_reading(r_matched_learned: float) -> str:
    """Reading 1: "reuse_identification_at_matched_horizon_suffices" when
    R_h*(evict_binary_h*_learned) is at least 0.9 (exactly 0.9 suffices),
    "ranker_order_costs_at_matched_horizon" otherwise (a nan R is the rule's
    "otherwise"): `horizonctl.class_order_reading`, relabelled."""
    return _MATCHED_LABEL[class_order_reading(r_matched_learned)]


def reading_counts(entries: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Reading 1 counted per role from `(role, reading)` pairs, one per trace
    x cell: `role -> {"cells": n, <reading>: count, ...}` for "new" and
    "reproduced", never merged (no total is formed)."""
    counts = {role: {"cells": 0, **dict.fromkeys(MATCHED_READINGS, 0)} for role in ROLES}
    for role, reading in entries:
        if role not in counts:
            raise ValueError(f"unknown role {role!r}")
        if reading not in MATCHED_READINGS:
            raise ValueError(f"unknown reading {reading!r}")
        counts[role]["cells"] += 1
        counts[role][reading] += 1
    return counts


def prediction_holds(new_counts: Mapping[str, int]) -> bool:
    """The prediction fixed before the run: reading 1 suffices in every trace
    x cell with h* < 600 s ("new"). False when there is none."""
    cells = int(new_counts["cells"])
    return cells > 0 and int(new_counts[MATCHED_READINGS[0]]) == cells


def seed_signs(values: Sequence[float]) -> tuple[int, int, int, str]:
    """Readings 2-4 on seed-paired differences: (positive, zero, negative,
    reading) with `mechanism.sign_counts` / `mechanism.sign_reading` as they
    are ("consistent_gain" when every seed is > 0, "consistent_loss" when
    every seed is < 0, "mixed" otherwise, a zero or nan breaking
    consistency)."""
    positive, zero, negative = sign_counts(values)
    return positive, zero, negative, sign_reading(values)


def sign_reading_counts(readings: Iterable[str]) -> dict[str, int]:
    """Counts of consistent_gain / consistent_loss / mixed over trace x cell."""
    counts = dict.fromkeys(SIGN_READINGS, 0)
    for reading in readings:
        if reading not in counts:
            raise ValueError(f"unknown sign reading {reading!r}")
        counts[reading] += 1
    return counts
