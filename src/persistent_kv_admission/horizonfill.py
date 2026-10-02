"""Horizon fill-in — is there a reuse horizon between 60 and 300 seconds where none of five sufficed.

`docs/horizon-fill-plan.md` is the pre-registration. It is a follow-up of the
horizon and class-order control (`horizonctl`, `docs/horizon-control-plan.md`),
restricted to the four trace x cell that read "order needed" there, and rides
on that control's Part 1 arm unchanged:

* `label_binary_h`, key `(1 if the state's next use is at most h seconds away
  else 0, last_group)`: the label rung's scorer,
  `mechanism.ExactLabelScorer(trace, h, target="binary")`, built exactly as
  `horizonctl.arm_setup` builds it for its own horizons, under `all16`;
* fill horizons 90, 120, 150, 180 and 240 s, and the check horizons 60, 300
  and 600 s, which the horizon control already ran and which are rerun for a
  self-contained curve and as reproduction checks.

Nothing here fits, adds a feature or target, or changes an existing replay
path; every arm reads the trace's future on purpose and none is a proposed
policy. The reading helpers are pure arithmetic on numbers the replays
produced, kept here so that they can be tested. The threshold, its boundary
and the nan rule are `horizonctl.horizon_reading`'s, used as it is.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .errorloc import ArmSetup
from .horizonctl import ALL16, BINARY_TARGET, SUFFICES_SHORTFALL, horizon_arm, horizon_reading
from .mechanism import ExactLabelScorer
from .trace import Trace

# --- the grid (docs/horizon-fill-plan.md, "Fixed surface") ------------------------------------

# Multiples of 30 s on the 3-second timestamp grid, strictly between 60 and 300 s.
FILL_HORIZONS_SECONDS = (90.0, 120.0, 150.0, 180.0, 240.0)
# Rerun for a self-contained curve and as checks; 600 is the published `label_binary`.
CHECK_HORIZONS_SECONDS = (60.0, 300.0, 600.0)
HORIZONS_SECONDS = tuple(sorted(FILL_HORIZONS_SECONDS + CHECK_HORIZONS_SECONDS))
# Readings 2 and 3 are read over the seven horizons from 60 to 300 s.
CURVE_HORIZONS_SECONDS = tuple(h for h in HORIZONS_SECONDS if 60.0 <= h <= 300.0)
# (l1_fraction, l2_multiplier): 0.25% x 4 and 1% x 1, both an L2 of 1% of the base.
FILL_CELLS = ((0.0025, 4.0), (0.01, 1.0))
MECHANISM = ALL16
ROLES = ("fill", "check")
ROLE_OF_HORIZON = {h: "fill" if h in FILL_HORIZONS_SECONDS else "check" for h in HORIZONS_SECONDS}

ARMS = tuple(horizon_arm(h) for h in HORIZONS_SECONDS)
HORIZON_OF_ARM = dict(zip(ARMS, HORIZONS_SECONDS))
FILL_ARMS = tuple(horizon_arm(h) for h in FILL_HORIZONS_SECONDS)
CHECK_ARMS = tuple(horizon_arm(h) for h in CHECK_HORIZONS_SECONDS)
# The h = 600 arm, which must reproduce the published `label_binary` rows.
PUBLISHED_HORIZON_ARM = horizon_arm(600.0)

assert HORIZONS_SECONDS == (60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 600.0)
assert len(CURVE_HORIZONS_SECONDS) == 7 and len(set(ARMS)) == 8
assert all(60.0 < h < 300.0 for h in FILL_HORIZONS_SECONDS)


def arm_setup(arm: str, trace: Trace) -> ArmSetup:
    """The replay of `label_binary_h` on `trace`: the store's first minimum of
    `(binary_h, last_group)`, no override; the class and the target are the
    ones `horizonctl.arm_setup` builds for its horizon arms. A fresh scorer is
    built per call, as there."""
    if arm not in HORIZON_OF_ARM:
        raise ValueError(f"unknown arm {arm!r}")
    return ArmSetup(arm, "learned",
                    scorer=ExactLabelScorer(trace, HORIZON_OF_ARM[arm], target=BINARY_TARGET))


# --- the readings (docs/horizon-fill-plan.md, "Readings, fixed before the run") ---------------

FILL_READINGS = ("reuse_label_suffices_at_filled_horizon", "order_needed_stands")
# The prediction fixed before the run: a reuse label suffices at a filled
# horizon in all four trace x cell.
PREDICTED_SUFFICING = 4
_FILL_LABEL = {"reuse_label_suffices": FILL_READINGS[0], "order_needed": FILL_READINGS[1]}


def _require_horizons(shortfalls: Mapping[float, float], expected: tuple[float, ...],
                      name: str) -> dict[float, float]:
    """`shortfalls` keyed by float horizon, refused unless it holds exactly
    the horizons `expected` names."""
    values = {float(h): float(s) for h, s in shortfalls.items()}
    if set(values) != set(expected):
        unknown = sorted(set(values) - set(expected))
        missing = sorted(set(expected) - set(values))
        raise ValueError(f"{name} needs S_h at exactly {list(expected)}; "
                         f"unknown {unknown}, missing {missing}")
    return values


def fill_reading(shortfalls: Mapping[float, float]) -> tuple[str, float, tuple[float, ...]]:
    """Reading 1 of one trace x cell, from `h -> S_h` on five-seed means over
    the five fill horizons only.

    "reuse_label_suffices_at_filled_horizon" when the smallest S_h is at most
    0.10 (exactly 0.10 suffices), "order_needed_stands" otherwise. Returns the
    label, the smallest S_h and every fill horizon attaining it, ascending. A
    nan S_h attains nothing; when every S_h is nan the reading is
    "order_needed_stands" (the rule's "otherwise").
    """
    values = _require_horizons(shortfalls, FILL_HORIZONS_SECONDS, "fill_reading")
    label, smallest, attaining = horizon_reading(values)
    return _FILL_LABEL[label], smallest, attaining


def unimodal(values: Sequence[float]) -> bool:
    """Reading 2: whether the sequence is non-increasing up to some index and
    non-decreasing after it (a valley; flat and monotone sequences qualify).
    Any nan, or no value at all, is not unimodal.

    The longest non-increasing prefix ends at a minimiser; the sequence is
    unimodal exactly when the rest is non-decreasing from there.
    """
    values = [float(value) for value in values]
    if not values or any(math.isnan(value) for value in values):
        return False
    turn = 0
    while turn + 1 < len(values) and values[turn + 1] <= values[turn]:
        turn += 1
    return all(values[i + 1] >= values[i] for i in range(turn, len(values) - 1))


def curve_values(shortfalls: Mapping[float, float]) -> list[float]:
    """S_h over the seven curve horizons, in ascending h (unimodal's input)."""
    values = _require_horizons(shortfalls, CURVE_HORIZONS_SECONDS, "curve")
    return [values[h] for h in CURVE_HORIZONS_SECONDS]


def curve_minimisers(shortfalls: Mapping[float, float]) -> tuple[float, ...]:
    """Reading 2: every curve horizon attaining the smallest S_h over the seven
    from 60 to 300 s, ascending (several on an exact tie, none when all are nan)."""
    return horizon_reading(_require_horizons(shortfalls, CURVE_HORIZONS_SECONDS, "curve"))[2]


def same_minimiser(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    """Reading 2: whether two cells of a trace have the same minimising
    horizon: the same set of minimisers, and at least one."""
    return bool(left) and tuple(left) == tuple(right)


def sufficing_set(shortfalls: Mapping[float, float]) -> tuple[float, ...]:
    """Reading 3: the curve horizons (60 to 300 s) with S_h <= 0.10, ascending
    (a nan S_h does not suffice)."""
    values = _require_horizons(shortfalls, CURVE_HORIZONS_SECONDS, "sufficing_set")
    return tuple(h for h in CURVE_HORIZONS_SECONDS
                 if math.isfinite(values[h]) and values[h] <= SUFFICES_SHORTFALL)


def sufficing_width_seconds(shortfalls: Mapping[float, float]) -> float:
    """Reading 3: largest minus smallest member of the sufficing set, in
    seconds (0 for one member); nan when the set is empty. Members need not be
    contiguous; the set itself is published beside the width."""
    members = sufficing_set(shortfalls)
    return members[-1] - members[0] if members else math.nan
