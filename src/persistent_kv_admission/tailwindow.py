"""Evaluation-window check — do the matched-horizon readings hold when the last 600 seconds are excluded.

`docs/tail-window-check-plan.md` is the pre-registration. Every exact-label arm
treats a state with no later occurrence in the trace as "not reused", which at
the end of a finite trace is knowledge of the end. This check reruns, with
everything else unchanged, the seven arms behind the matched-horizon readings
and counts each replay on three windows:

* `full`: the published evaluation window, from the split to the trace end
  (the replay's own counters);
* `head`: from the split to `trace end - 600,000 ms`, inclusive, counted by
  `onpolicy.LabelWindowUtilityCollector(trace, split_ms, end_ms - 600,000)`
  exactly as the on-policy run counted its label window;
* `tail`: `full - head`, by subtraction of the integer counters.

Reused unchanged:

* every arm's construction, dispatched to the module that built its published
  rows: `lru`, `learned`, `label` and `evict_label` to `errorloc.arm_setup`
  (`results/paper/error_location_001/`); `label_binary_60`, `_300` and `_600`
  to `horizonctl.arm_setup` (`results/paper/horizon_control_001/`);
  `label_binary_150` to `horizonfill.arm_setup`
  (`results/paper/horizon_fill_001/`); `evict_binary_{h*}_learned` /
  `_recency` to `matchedorder.arm_setup`
  (`results/paper/matched_class_order_001/`). Nothing here builds a scorer or
  an override;
* `h*` of each cell (`matchedorder.MATCHED_HORIZON_SECONDS`), the arm names
  (`horizonctl.horizon_arm`, `matchedorder.cell_arms`), the shortfall
  `horizonctl.shortfall` with the 0.10 rule of `horizonctl.horizon_reading`,
  the recovery share and 0.9 rule of `matchedorder.matched_recovery` /
  `matched_reading`, and the sign rule `errorloc.agrees` (zero matches zero
  only, a nan matches nothing).

New here:

* `ChainedRequestHook`, which calls several read-only request hooks with the
  same arguments, so that the head collector sits beside the attribution
  collector in `run_two_tier`'s single `l2_request_hook` slot. It returns
  nothing and changes nothing;
* the window arithmetic (`full_row`, `head_row`, `tail_row`, the utility `U`
  of a window in points of that window's input tokens, and seed-paired
  differences on a window);
* the reading helpers of the plan on a window: the reading of one `S_h*`, the
  agreement of a head reading with the full one, the tail share of a token
  difference, and whether each of the four predictions holds.

Not done here: nothing fits, adds a feature or target, chooses `h*` again or
changes an existing replay path. Every arm reads the trace's future on purpose
and none is a policy. A difference between head and tail mixes the exact
labels' knowledge of the end with any change of the workload over time; no
helper attributes it to either.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from . import errorloc, horizonctl, horizonfill, matchedorder
from .errorloc import ArmSetup, agrees
from .horizonctl import HORIZON_READINGS, horizon_arm, horizon_reading, shortfall
from .matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    MATCHED_READINGS,
    matched_horizon,
    matched_reading,
    matched_recovery,
)
from .onpolicy import LabelWindowUtilityCollector
from .trace import Trace

# --- the arms (docs/tail-window-check-plan.md, "Fixed surface") ------------------------------

MECHANISM = horizonctl.ALL16
# The seven arms of every trace x cell, by role; the h* roles resolve per cell.
RUNG_ROLE = "label_binary_h*"
MATCHED_ROLES = {"evict_binary_h*_learned": "learned", "evict_binary_h*_recency": "recency"}
ERROR_LOCATION_ARMS = ("lru", "learned", "label", "evict_label")
ARMS = ERROR_LOCATION_ARMS + (RUNG_ROLE,) + tuple(MATCHED_ROLES)
# Column-safe names of the roles ("h*" is not a column name).
SLUGS = {"lru": "lru", "learned": "learned", "label": "label", "evict_label": "evict_label",
         RUNG_ROLE: "label_binary_hstar", "evict_binary_h*_learned": "matched_learned",
         "evict_binary_h*_recency": "matched_recency"}
# The published run each arm's rows come from, and the only one it is read from.
ERROR_LOCATION_SOURCE = "error_location_001"
HORIZON_CONTROL_SOURCE = "horizon_control_001"
HORIZON_FILL_SOURCE = "horizon_fill_001"
MATCHED_SOURCE = "matched_class_order_001"
SOURCES = (ERROR_LOCATION_SOURCE, HORIZON_CONTROL_SOURCE, HORIZON_FILL_SOURCE, MATCHED_SOURCE)
# The label rung of each matched horizon: 150 s only in the fill-in.
RUNG_SOURCES = {60.0: HORIZON_CONTROL_SOURCE, 150.0: HORIZON_FILL_SOURCE,
                300.0: HORIZON_CONTROL_SOURCE, 600.0: HORIZON_CONTROL_SOURCE}
RUNG_ARMS = {horizon_arm(h): h for h in MATCHED_HORIZONS_SECONDS}

assert len(ARMS) == 7 and len(set(ARMS)) == 7 and set(SLUGS) == set(ARMS)
assert set(RUNG_SOURCES) == set(MATCHED_HORIZONS_SECONDS)
assert all(h in horizonctl.HORIZONS_SECONDS for h, source in RUNG_SOURCES.items()
           if source == HORIZON_CONTROL_SOURCE)
assert all(h in horizonfill.FILL_HORIZONS_SECONDS for h, source in RUNG_SOURCES.items()
           if source == HORIZON_FILL_SOURCE)


def resolve_arm(role: str, horizon_seconds: float) -> str:
    """The arm a role names at horizon `h*`: the role itself for the four
    error-location arms, `label_binary_{h*}` and `evict_binary_{h*}_{order}`."""
    if role in ERROR_LOCATION_ARMS:
        return role
    if role == RUNG_ROLE:
        return horizon_arm(float(horizon_seconds))
    if role in MATCHED_ROLES:
        return matchedorder.matched_arm(horizon_seconds, MATCHED_ROLES[role])
    raise ValueError(f"unknown arm role {role!r}")


def cell_arms(fraction: float, multiplier: float) -> tuple[str, ...]:
    """The seven arms replayed at a cell, in `ARMS` order, at the cell's h*;
    the two class-order arms are `matchedorder.cell_arms`."""
    h = matched_horizon(fraction, multiplier)
    arms = tuple(resolve_arm(role, h) for role in ARMS)
    assert arms[-2:] == matchedorder.cell_arms(fraction, multiplier)
    return arms


def arm_role(arm: str) -> str:
    """The role of a concrete arm (inverse of `resolve_arm`)."""
    if arm in ERROR_LOCATION_ARMS:
        return arm
    if arm in RUNG_ARMS:
        return RUNG_ROLE
    if arm in matchedorder.ORDER_OF_ARM:
        return f"evict_binary_h*_{matchedorder.ORDER_OF_ARM[arm]}"
    raise ValueError(f"unknown arm {arm!r}")


def arm_horizon(arm: str) -> float | None:
    """The reuse horizon an arm is built at (seconds), None for the four
    error-location arms, which use the trace's label horizon."""
    if arm in RUNG_ARMS:
        return RUNG_ARMS[arm]
    if arm in matchedorder.HORIZON_OF_ARM:
        return matchedorder.HORIZON_OF_ARM[arm]
    if arm in ERROR_LOCATION_ARMS:
        return None
    raise ValueError(f"unknown arm {arm!r}")


def reference_source(arm: str) -> str:
    """The published run whose row the replay of `arm` must reproduce."""
    if arm in ERROR_LOCATION_ARMS:
        return ERROR_LOCATION_SOURCE
    if arm in RUNG_ARMS:
        return RUNG_SOURCES[RUNG_ARMS[arm]]
    if arm in matchedorder.HORIZON_OF_ARM:
        return MATCHED_SOURCE
    raise ValueError(f"unknown arm {arm!r}")


def is_class_order_arm(arm: str) -> bool:
    return arm in matchedorder.HORIZON_OF_ARM


def arm_setup(arm: str, trace: Trace, horizon_seconds: float, seed: int,
              learned_ranker) -> ArmSetup:
    """The replay of `arm` on `trace`, built by the module that built its
    published rows, under `all16` (applied by the caller):

    * `lru`, `learned`, `label`, `evict_label`: `errorloc.arm_setup`, with
      `horizon_seconds` the trace's label horizon (600 s on the real traces);
    * `label_binary_60` / `_300` / `_600`: `horizonctl.arm_setup`;
    * `label_binary_150`: `horizonfill.arm_setup`;
    * `evict_binary_{h}_learned` / `_recency`: `matchedorder.arm_setup`.

    `learned_ranker` is the frozen pi0 `next_use` ranker of the trace. A fresh
    scorer is built per call, by the callee, as in every parent.
    """
    if arm in ERROR_LOCATION_ARMS:
        return errorloc.arm_setup(arm, trace, horizon_seconds, seed,
                                  learned_ranker=learned_ranker)
    if arm in RUNG_ARMS:
        if RUNG_SOURCES[RUNG_ARMS[arm]] == HORIZON_FILL_SOURCE:
            return horizonfill.arm_setup(arm, trace)
        return horizonctl.arm_setup(arm, trace, horizon_seconds, seed,
                                    learned_ranker=learned_ranker)
    if arm in matchedorder.HORIZON_OF_ARM:
        return matchedorder.arm_setup(arm, trace, learned_ranker)
    raise ValueError(f"unknown arm {arm!r}")


# --- the second collector ---------------------------------------------------------------------


class ChainedRequestHook:
    """`l2_request_hook` that calls each of `hooks`, in order, with the
    arguments the replay passes, the same objects for every hook, and returns
    None. `run_two_tier` has one request-hook slot and never reads what it
    returns; the hooks chained here (`AttributionCollector.on_request`,
    `LabelWindowUtilityCollector.on_request`) only read their arguments."""

    def __init__(self, *hooks) -> None:
        if not hooks:
            raise ValueError("a chained request hook needs at least one hook")
        for hook in hooks:
            if not callable(hook):
                raise TypeError(f"{hook!r} is not callable")
        self.hooks = tuple(hooks)

    def __call__(self, *args, **kwargs) -> None:
        for hook in self.hooks:
            hook(*args, **kwargs)
        return None


# --- the windows (docs/tail-window-check-plan.md, "Fixed surface") ----------------------------

TAIL_SECONDS = 600.0
TAIL_MS = TAIL_SECONDS * 1000.0
WINDOWS = ("full", "head", "tail")
# The replay's window counters (`TwoTierResult` fields) and the head
# collector's column for each.
COUNTERS = ("measured_requests", "requested_tokens", "requested_blocks", "l1_avoided_tokens",
            "l2_avoided_tokens", "avoided_prefill_tokens", "l2_hit_blocks",
            "l2_present_unusable_tokens", "l2_present_unusable_blocks")
COLLECTOR_COLUMNS = {
    "measured_requests": "label_window_measured_requests",
    "requested_tokens": "label_window_requested_tokens",
    "requested_blocks": "label_window_requested_blocks",
    "l1_avoided_tokens": "label_window_l1_avoided_tokens",
    "l2_avoided_tokens": "label_window_l2_avoided_tokens",
    "avoided_prefill_tokens": "label_window_avoided_prefill_tokens",
    "l2_hit_blocks": "label_window_l2_hit_blocks",
    "l2_present_unusable_tokens": "label_window_present_unusable_tokens",
    "l2_present_unusable_blocks": "label_window_present_unusable_blocks",
}
# The head counters that cannot depend on the arm (the plan's check).
HEAD_INVARIANT_COUNTERS = ("measured_requests", "requested_tokens", "l1_avoided_tokens")
assert set(COLLECTOR_COLUMNS) == set(COUNTERS)
assert TAIL_MS == 600_000.0


def head_end_ms(trace: Trace) -> float:
    """The head window's last instant, inclusive: `trace end - 600,000 ms`."""
    return trace.end_ms - TAIL_MS


def window_bounds(trace: Trace, split_ms: float) -> dict[str, float]:
    """The three windows of a trace, in ms: full `[split, end]`, head
    `[split, end - 600 s]`, tail `(end - 600 s, end]`."""
    return {"split_ms": float(split_ms), "head_end_ms": head_end_ms(trace),
            "end_ms": float(trace.end_ms), "tail_seconds": TAIL_SECONDS,
            "full_seconds": (trace.end_ms - split_ms) / 1000.0,
            "head_seconds": (head_end_ms(trace) - split_ms) / 1000.0}


def head_collector(trace: Trace, split_ms: float) -> LabelWindowUtilityCollector:
    """The head window's counters, as the on-policy run counted its label
    window: `LabelWindowUtilityCollector(trace, split_ms, end_ms - 600,000)`."""
    return LabelWindowUtilityCollector(trace, split_ms, head_end_ms(trace))


def full_row(result) -> dict[str, int]:
    """The full window's counters: the replay's own."""
    return {counter: int(getattr(result, counter)) for counter in COUNTERS}


def head_row(collector: LabelWindowUtilityCollector) -> dict[str, int]:
    """The head window's counters, from the collector's `row()`."""
    row = collector.row()
    return {counter: int(row[COLLECTOR_COLUMNS[counter]]) for counter in COUNTERS}


def tail_row(full: Mapping[str, int], head: Mapping[str, int]) -> dict[str, int]:
    """`full - head`, counter by counter, in integers."""
    return {counter: int(full[counter]) - int(head[counter]) for counter in COUNTERS}


def extra_tokens(counters: Mapping[str, int]) -> int:
    """Avoided prefill tokens beyond L1 alone: `avoided - l1_avoided`."""
    return int(counters["avoided_prefill_tokens"]) - int(counters["l1_avoided_tokens"])


def points(tokens: int, requested: int) -> float:
    """`100 * tokens / requested`, with the parents' guard `max(requested, 1)`
    (`run_mechanism_control._points`, the same expression)."""
    return 100.0 * tokens / max(int(requested), 1)


def utility_points(counters: Mapping[str, int]) -> float:
    """`U` of a window: extra avoided tokens in points of the window's input
    tokens. On the full window it is the parents' `extra_points`."""
    return points(extra_tokens(counters), counters["requested_tokens"])


def window_columns(full: Mapping[str, int], head: Mapping[str, int],
                   tail: Mapping[str, int]) -> dict[str, object]:
    """The per-replay columns: `<window>_<counter>`, `<window>_extra_avoided_tokens`
    and `U_<window>_points` for full, head and tail."""
    out: dict[str, object] = {}
    for window, counters in zip(WINDOWS, (full, head, tail)):
        for counter in COUNTERS:
            out[f"{window}_{counter}"] = int(counters[counter])
        out[f"{window}_extra_avoided_tokens"] = extra_tokens(counters)
        out[f"U_{window}_points"] = utility_points(counters)
    return out


def counters_of(row: Mapping[str, object], window: str) -> dict[str, int]:
    """A window's counters read back from a row of `window_columns`."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}")
    return {counter: int(row[f"{window}_{counter}"]) for counter in COUNTERS}


def window_utility(row: Mapping[str, object], window: str) -> float:
    """`U` of a replay on a window, recomputed from its integer counters."""
    return utility_points(counters_of(row, window))


def window_difference(later: Mapping[str, object], earlier: Mapping[str, object],
                      window: str) -> tuple[int, float]:
    """`U(later) - U(earlier)` of one seed on a window: integer tokens and
    points of `later`'s window input tokens; on the full window this is
    `run_error_location._difference_points`."""
    tokens = (extra_tokens(counters_of(later, window))
              - extra_tokens(counters_of(earlier, window)))
    return tokens, points(tokens, int(later[f"{window}_requested_tokens"]))


# --- the readings (docs/tail-window-check-plan.md, "Readings, fixed before the run") ----------

SUFFICES_SHORTFALL = horizonctl.SUFFICES_SHORTFALL          # 0.10
CLASS_ORDER_SHARE = horizonctl.CLASS_ORDER_SHARE            # 0.9
# Prediction 2: below 0.9 on the head window at 0.25% x 1, on both traces.
PREDICTED_BELOW_CELL = (0.0025, 1.0)
# The differences of reading 5: name -> (later role, earlier role, reading).
DIFFERENCES = {
    "G": ("label", "learned", "5"),
    "label_minus_label_binary": ("label", RUNG_ROLE, "1"),
    "label_minus_lru": ("label", "lru", "1"),
    "matched_learned_minus_learned": ("evict_binary_h*_learned", "learned", "2"),
    "matched_recency_minus_learned": ("evict_binary_h*_recency", "learned", "2"),
    "evict_label_minus_learned": ("evict_label", "learned", "2"),
    "learned_minus_recency": ("evict_binary_h*_learned", "evict_binary_h*_recency", "3"),
    "label_binary_minus_recency": (RUNG_ROLE, "evict_binary_h*_recency", "4"),
}
assert SUFFICES_SHORTFALL == 0.10 and CLASS_ORDER_SHARE == 0.9
assert PREDICTED_BELOW_CELL in MATCHED_HORIZON_SECONDS
assert all(later in ARMS and earlier in ARMS for later, earlier, _ in DIFFERENCES.values())


def window_shortfall(u_label: float, u_rung: float, u_lru: float) -> float:
    """Reading 1 on a window: `S_h* = (U(label) - U(label_binary_h*)) /
    (U(label) - U(lru))`, `horizonctl.shortfall` (nan for a zero denominator)."""
    return shortfall(u_label, u_rung, u_lru)


def shortfall_reading(s: float) -> str:
    """"reuse_label_suffices" when `S_h*` is at most 0.10 (exactly 0.10
    suffices), "order_needed" otherwise, a nan included:
    `horizonctl.horizon_reading` on the one horizon h*."""
    return horizon_reading({0.0: float(s)})[0]


def suffices(s: float) -> bool:
    """Prediction 1's count: `S_h* <= 0.10`."""
    return shortfall_reading(s) == HORIZON_READINGS[0]


def window_recovery(u_arm: float, u_learned: float, u_evict_label: float) -> float:
    """Reading 2 on a window: `R_h*(a) = (U(a) - U(learned)) / (U(evict_label)
    - U(learned))`, nan for a zero denominator."""
    return matched_recovery(u_arm, u_learned, u_evict_label)


def class_order_reading(r: float) -> str:
    """Reading 2's label: `matchedorder.matched_reading` (R >= 0.9, exactly 0.9
    included, "suffices"; otherwise, a nan included, "costs")."""
    return matched_reading(r)


def recovers(r: float) -> bool:
    """Prediction 2's count: `R_h* >= 0.9`."""
    return class_order_reading(r) == MATCHED_READINGS[0]


def sign_agrees(value: float, reference: float) -> bool:
    """Readings 3 and 4: the sign of a five-seed mean on one window is the
    sign on the full window (`errorloc.agrees`: zero matches zero only, a nan
    matches nothing)."""
    return agrees(value, reference)


def tail_share(full_tokens: float, tail_tokens: float) -> float:
    """Reading 5: the share of a full-window quantity that falls in the tail,
    `tail / full`; nan when the full quantity is zero. Not clipped: a head and
    a tail of opposite signs give a share outside [0, 1]."""
    return tail_tokens / full_tokens if full_tokens else math.nan


def predicted_head_class_reading(cell: Sequence[float]) -> str:
    """Prediction 2 cell by cell: "costs" at 0.25% x 1, "suffices" elsewhere."""
    if (float(cell[0]), float(cell[1])) == PREDICTED_BELOW_CELL:
        return MATCHED_READINGS[1]
    return MATCHED_READINGS[0]


def horizon_prediction_holds(head_shortfalls: Iterable[float]) -> bool:
    """Prediction 1: `S_h* <= 0.10` on the head window in every trace x cell
    (False when there is none)."""
    values = list(head_shortfalls)
    return bool(values) and all(suffices(value) for value in values)


def class_order_prediction_holds(entries: Iterable[tuple[Sequence[float], str, str]]) -> bool:
    """Prediction 2 from `(cell, full reading, head reading)` per trace x cell:
    the head reading is the full window's in every trace x cell, and the head
    reading is "costs" exactly at 0.25% x 1 (False when there is none)."""
    entries = list(entries)
    if not entries:
        return False
    return all(head == full and head == predicted_head_class_reading(cell)
               for cell, full, head in entries)


def sign_prediction_holds(agreements: Iterable[bool]) -> bool:
    """Predictions 3 and 4: the sign of the head mean agrees with the full
    window's in every trace x cell (False when there is none)."""
    values = list(agreements)
    return bool(values) and all(bool(value) for value in values)
