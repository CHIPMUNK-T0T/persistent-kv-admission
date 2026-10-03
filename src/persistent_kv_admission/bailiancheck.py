"""External-workload check on the Qwen-Bailian traces — the matched horizons transplanted, and a sufficing horizon within a fixed grid.

`docs/bailian-external-check-plan.md` is the pre-registration (agreed and
committed alone, commit `553b026`); nothing here may change it. This module
holds the plan's fixed surface and the pure arithmetic of its readings; the
runner (`scripts/run_bailian_external_check.py`) does every replay and every
file. Nothing here fits, adds a feature or target, or changes an existing
replay path; every arm reads the trace's future on purpose and none is a
policy.

The arms, each the store's first minimum of its key with no override, built
by the module that built its published rows wherever that module supports the
arm (`arm_setup`, `arm_builder`):

* `lru` and `label`: `errorloc.arm_setup` (the error-location rungs, published
  under `all16` in `error_location_001`; under `leaf16` the mechanism control
  built the same objects inline, and the leaf-matched run rebuilt `label` with
  `errorloc.arm_setup` and reproduced them);
* `label_binary_h` for `h` in the grid `G`: under `all16` `horizonctl.arm_setup`
  at the horizon control's horizons (6, 15, 60, 300, 600 s) and
  `horizonfill.arm_setup` at 150 s; under `leaf16` `horizonfill.arm_setup` at
  the leaf-matched horizons (60, 150, 300, 600 s) and `horizonctl.arm_setup`
  at 6 and 15 s; at 1,200 s, which no module lists, the identical
  construction, `ExactLabelScorer(trace, 1200, target="binary")` with
  `l2_policy="learned"` and no override;
* `label_binary_random_{h*}`: `classmix.RandomClassScorer(ExactLabelScorer(
  trace, h*, "binary"), seed, arm)` as the store's scorer for admission and
  eviction alike, no override, key `((bit, u), last_group)`; its stream is
  `random.Random(classmix.random_stream_seed(seed, arm))`, separate from the
  store's sampling stream `random.Random(l2_seed)`.

The windows (`window_bounds`, `window_collectors`): the full window
`[split, end]` is the replay's own (`measure_from_ms = split`, requests with
`timestamp >= split`); the primary window `W = [split, end - 1,200,000 ms]`,
its halves `W1 = [split, mid]` and `W2 = (mid, end - 1,200,000 ms]` with `mid`
the midpoint of `W`, are counted by three `onpolicy.LabelWindowUtilityCollector`
(inclusive bounds), chained beside the attribution collector by
`tailwindow.ChainedRequestHook`, exactly as `tailwindow.head_collector` counts
its head window from the same split. `W2`'s collector starts at
`math.nextafter(mid, inf)`, so a request exactly at `mid` is in `W1` only and
`W1 + W2 = W` holds request by request.

Readings 1-4 and 6 and predictions 1-6 are the plan's; reading 7 and
prediction 7 are its Addendum 1 (2026-10-03, before any replay): per trace x
cell and mechanism, `Delta_m(h) = U_m(label) - U_m(label_binary_h)` seed-paired
within the mechanism at h* and at 600 s; a cell is "separated" when the
smallest leaf16 seed value exceeds the largest all16 seed value (strictly) and
"reversed" when both seed readings are consistent with opposite signs, counted
over the cells evaluable under both mechanisms; prediction 7 is separation at
h* in at least 10 of the 16 trace x cell with h* < 600 s. No arm, capacity,
seed, window or replay count changed.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

from . import classmix, errorloc, horizonctl, horizonfill, matchedorder, tailwindow
from .errorloc import ArmSetup
from .gap import summarize
from .horizonctl import ALL16, BINARY_TARGET, LEAF16, MECHANISMS, horizon_arm
from .mechanism import ExactLabelScorer, sign_counts, sign_reading
from .onpolicy import LabelWindowUtilityCollector
from .trace import Trace

# --- the fixed surface (docs/bailian-external-check-plan.md) ----------------------------------

# The grid G of question B, in seconds, fixed at agreement (decision 1).
GRID_SECONDS = (6.0, 15.0, 60.0, 150.0, 300.0, 600.0, 1200.0)
# The Mooncake cell -> h* table, carried over unchanged (question A).
HSTAR_SECONDS = {
    (0.0025, 1.0): 60.0,
    (0.0025, 4.0): 150.0,
    (0.01, 1.0): 150.0,
    (0.01, 4.0): 600.0,
    (0.02, 1.0): 300.0,
    (0.02, 4.0): 600.0,
}
FRACTIONS = (0.0025, 0.01, 0.02)
MULTIPLIERS = (1.0, 4.0)
CELLS = tuple((fraction, multiplier) for fraction in FRACTIONS for multiplier in MULTIPLIERS)
MECHANISM_NAMES = (ALL16, LEAF16)
SEEDS = (0, 1, 2, 3, 4)
# Control C's reduction when the 16-token smoke exceeds 20 minutes per replay.
REDUCED_SEEDS = (0, 1, 2)
# The four converted traces (file stems, the loader's trace names).
TRACES = ("bailian_toc_trace", "bailian_tob_trace", "bailian_thinking_trace",
          "bailian_coder_trace")
UPSTREAM_SOURCES = {"bailian_toc_trace": "qwen_traceA_blksz_16.jsonl",
                    "bailian_tob_trace": "qwen_traceB_blksz_16.jsonl",
                    "bailian_thinking_trace": "qwen_thinking_blksz_16.jsonl",
                    "bailian_coder_trace": "qwen_coder_blksz_16.jsonl"}
GRANULARITY_TRACE = "bailian_tob_trace"
BLOCK_TOKENS = 512
FINE_BLOCK_TOKENS = 16
# Split and label horizon as every replay runner (`decisionpop.horizon_for`).
LABEL_HORIZON_SECONDS = 600.0

assert HSTAR_SECONDS == matchedorder.MATCHED_HORIZON_SECONDS
assert set(HSTAR_SECONDS) == set(CELLS)
assert set(HSTAR_SECONDS.values()) <= set(GRID_SECONDS)
assert MECHANISMS[ALL16] == ("all", 16) and MECHANISMS[LEAF16] == ("leaf", 16)
assert tuple(sorted(UPSTREAM_SOURCES)) == tuple(sorted(TRACES))

# --- the arms ---------------------------------------------------------------------------------

LRU, LABEL = "lru", "label"
REFERENCE_ARMS = (LRU, LABEL)
GRID_ARMS = tuple(horizon_arm(h) for h in GRID_SECONDS)
HORIZON_OF_GRID_ARM = dict(zip(GRID_ARMS, GRID_SECONDS))
RANDOM_PREFIX = "label_binary_random_"


def random_arm(horizon_seconds: float) -> str:
    """`label_binary_random_{h:g}`."""
    return f"{RANDOM_PREFIX}{float(horizon_seconds):g}"


HSTAR_HORIZONS = tuple(sorted(set(HSTAR_SECONDS.values())))
RANDOM_ARMS = tuple(random_arm(h) for h in HSTAR_HORIZONS)
HORIZON_OF_RANDOM_ARM = dict(zip(RANDOM_ARMS, HSTAR_HORIZONS))
ARMS = REFERENCE_ARMS + GRID_ARMS + RANDOM_ARMS
FAMILIES = ("reference", "grid", "random")
PARTS = ("A", "B")
# Heap references: name -> the published heap policy (run_decision_population's).
HEAP = "heap"
HEAP_POLICIES = {"H_lru": "lru", "H_off": "offline_next_use"}
HEAP_ARMS = tuple(HEAP_POLICIES)

assert GRID_ARMS == ("label_binary_6", "label_binary_15", "label_binary_60", "label_binary_150",
                     "label_binary_300", "label_binary_600", "label_binary_1200")
assert RANDOM_ARMS == ("label_binary_random_60", "label_binary_random_150",
                       "label_binary_random_300", "label_binary_random_600")
assert len(set(ARMS)) == len(ARMS) == 13


def hstar(fraction: float, multiplier: float) -> float:
    """h* of a cell, in seconds (the Mooncake table)."""
    try:
        return HSTAR_SECONDS[(float(fraction), float(multiplier))]
    except KeyError:
        raise ValueError(f"no h* for cell {(fraction, multiplier)!r}") from None


def transplant_arm(fraction: float, multiplier: float) -> str:
    """`label_binary_{h*}` of a cell: the transplant arm of question A."""
    return horizon_arm(hstar(fraction, multiplier))


def cell_random_arm(fraction: float, multiplier: float) -> str:
    """`label_binary_random_{h*}` of a cell."""
    return random_arm(hstar(fraction, multiplier))


def a_arms(fraction: float, multiplier: float) -> tuple[str, str, str, str]:
    """The four arms of question A: lru, label, label_binary_h*, label_binary_random_h*."""
    return (LRU, LABEL, transplant_arm(fraction, multiplier),
            cell_random_arm(fraction, multiplier))


def b_arms(fraction: float, multiplier: float) -> tuple[str, ...]:
    """The six further grid points of question B (G without h*), ascending."""
    own = transplant_arm(fraction, multiplier)
    return tuple(arm for arm in GRID_ARMS if arm != own)


def cell_arms(fraction: float, multiplier: float) -> tuple[str, ...]:
    """The ten sampled arms of a cell: lru, label, the seven grid points
    ascending, and the random arm at h*."""
    return REFERENCE_ARMS + GRID_ARMS + (cell_random_arm(fraction, multiplier),)


def granularity_arms(fraction: float, multiplier: float) -> tuple[str, str, str]:
    """Control C's three arms: lru, label, label_binary_h*."""
    return (LRU, LABEL, transplant_arm(fraction, multiplier))


def part(arm: str, fraction: float, multiplier: float) -> str:
    """"A" for the four transplant arms of the cell, "B" for the other grid points."""
    if arm in a_arms(fraction, multiplier):
        return "A"
    if arm in b_arms(fraction, multiplier):
        return "B"
    raise ValueError(f"{arm} is not an arm of cell {(fraction, multiplier)}")


def arm_family(arm: str) -> str:
    if arm in REFERENCE_ARMS:
        return "reference"
    if arm in HORIZON_OF_GRID_ARM:
        return "grid"
    if arm in HORIZON_OF_RANDOM_ARM:
        return "random"
    raise ValueError(f"unknown arm {arm!r}")


def arm_horizon(arm: str) -> float | None:
    """The reuse horizon an arm is built at (seconds); None for lru and label
    (the label uses the trace's 600-second label horizon)."""
    if arm in HORIZON_OF_GRID_ARM:
        return HORIZON_OF_GRID_ARM[arm]
    if arm in HORIZON_OF_RANDOM_ARM:
        return HORIZON_OF_RANDOM_ARM[arm]
    if arm in REFERENCE_ARMS:
        return None
    raise ValueError(f"unknown arm {arm!r}")


# The modules that built the published rows of each label_binary_h, by
# mechanism: all16 -> horizon_control_001 (horizonctl) and horizon_fill_001
# (horizonfill, 150 s); leaf16 -> leaf_matched_horizon_001 (horizonfill at
# every matched horizon). A horizon no module publishes under a mechanism is
# built by the module that lists it, and 1,200 s by the identical construction.
_ALL16_FILL_HORIZONS = (150.0,)
_LEAF16_FILL_HORIZONS = HSTAR_HORIZONS


def arm_builder(arm: str, mechanism: str) -> str:
    """The name of the function that builds `arm` under `mechanism`."""
    if mechanism not in MECHANISM_NAMES:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if arm in REFERENCE_ARMS:
        return "errorloc.arm_setup"
    if arm in HORIZON_OF_GRID_ARM:
        h = HORIZON_OF_GRID_ARM[arm]
        fill = _ALL16_FILL_HORIZONS if mechanism == ALL16 else _LEAF16_FILL_HORIZONS
        if h in fill and arm in horizonfill.HORIZON_OF_ARM:
            return "horizonfill.arm_setup"
        if arm in horizonctl.HORIZON_OF_ARM:
            return "horizonctl.arm_setup"
        if arm in horizonfill.HORIZON_OF_ARM:
            return "horizonfill.arm_setup"
        return "ExactLabelScorer(trace, h, target='binary'), l2_policy='learned'"
    if arm in HORIZON_OF_RANDOM_ARM:
        return "classmix.RandomClassScorer(ExactLabelScorer(trace, h*, target='binary'), seed, arm)"
    raise ValueError(f"unknown arm {arm!r}")


def arm_setup(arm: str, trace: Trace, horizon_seconds: float, seed: int,
              mechanism: str = ALL16) -> ArmSetup:
    """The replay of `arm` on `trace` at sampling seed `seed` (the mechanism
    is applied by the caller; it only selects the builder, `arm_builder`).
    `horizon_seconds` is the trace's label horizon, read by `label` only. A
    fresh scorer is built per call, because the random stream carries state
    across a replay."""
    builder = arm_builder(arm, mechanism)
    if builder == "errorloc.arm_setup":
        return errorloc.arm_setup(arm, trace, horizon_seconds, seed)
    if builder == "horizonctl.arm_setup":
        return horizonctl.arm_setup(arm, trace, horizon_seconds, seed)
    if builder == "horizonfill.arm_setup":
        return horizonfill.arm_setup(arm, trace)
    if arm in HORIZON_OF_GRID_ARM:
        return ArmSetup(arm, "learned", scorer=ExactLabelScorer(
            trace, HORIZON_OF_GRID_ARM[arm], target=BINARY_TARGET))
    h = HORIZON_OF_RANDOM_ARM[arm]
    stream = classmix.RandomClassScorer(ExactLabelScorer(trace, h, target=BINARY_TARGET),
                                        seed, arm)
    return ArmSetup(arm, "learned", scorer=stream)


def random_stream(setup: ArmSetup) -> classmix.RandomClassScorer | None:
    """The random arm's stream (the store's scorer itself), None otherwise."""
    return setup.scorer if isinstance(setup.scorer, classmix.RandomClassScorer) else None


class CandidateCount:
    """A read-only recorder for `errorloc.RecordingOverride`: the decisions
    and the candidates of every decision of the replay (warm-up included). The
    store scores every candidate once per decision through `_sampled_key`, so
    a random arm's draw count must equal `candidates`. `attach` keeps the
    store's sampling generator, only to check afterwards that the random arm's
    stream is a different object; nothing is read from or written to it."""

    def __init__(self) -> None:
        self.decisions = 0
        self.candidates = 0
        self.store_rng = None

    def attach(self, store) -> None:
        self.store_rng = getattr(store, "rng", None)

    def record(self, candidates, keys, original_index: int, victim_index: int,
               timestamp_ms: float, group_index: int, arriving_index: int) -> None:
        self.decisions += 1
        self.candidates += len(candidates)


def stream_is_separate(stream: classmix.RandomClassScorer, counter: CandidateCount) -> bool:
    """The random arm's generator is its own `random.Random`, neither the
    store's sampling generator nor Python's global one."""
    return (isinstance(stream.rng, random.Random) and counter.store_rng is not None
            and stream.rng is not counter.store_rng
            and stream.rng is not getattr(random, "_inst", None))


# --- counts (docs/bailian-external-check-plan.md, "Arms" and the checks) -------------------------

COUNT_A = len(TRACES) * len(MECHANISM_NAMES) * len(CELLS) * len(SEEDS) * 4
COUNT_B = len(TRACES) * len(MECHANISM_NAMES) * len(CELLS) * len(SEEDS) * 6
COUNT_SAMPLED = COUNT_A + COUNT_B
COUNT_HEAP = len(TRACES) * len(CELLS) * len(HEAP_ARMS)
ANCHOR_TRACE = "conversation_trace"
ANCHOR_CELLS = ((0.0025, 1.0), (0.01, 4.0))
ANCHOR_ALL16_ARMS = (LRU, LABEL, horizon_arm(60.0), horizon_arm(600.0))


def anchor_arms(mechanism: str, fraction: float, multiplier: float) -> tuple[str, ...]:
    """The anchor's arms: under all16 lru, label, label_binary_60 and
    label_binary_600 at both cells; under leaf16 lru, label and the cell's
    label_binary_h* (the only leaf16 rung published)."""
    if mechanism == ALL16:
        return ANCHOR_ALL16_ARMS
    if mechanism == LEAF16:
        return (LRU, LABEL, transplant_arm(fraction, multiplier))
    raise ValueError(f"unknown mechanism {mechanism!r}")


# The published run(s) each anchor replay is held to, the first the primary.
# The leaf16 label row is published twice: with digests by the leaf-matched
# run and with every counter column (no digest) by the mechanism control.
ANCHOR_SOURCES = {
    (ALL16, LRU): ("error_location_001",),
    (ALL16, LABEL): ("error_location_001",),
    (ALL16, horizon_arm(60.0)): ("horizon_control_001",),
    (ALL16, horizon_arm(600.0)): ("horizon_control_001",),
    (LEAF16, LRU): ("mechanism_control_001",),
    (LEAF16, LABEL): ("leaf_matched_horizon_001", "mechanism_control_001"),
    (LEAF16, horizon_arm(60.0)): ("leaf_matched_horizon_001",),
    (LEAF16, horizon_arm(600.0)): ("leaf_matched_horizon_001",),
}


def anchor_replays(seeds: Sequence[int] = SEEDS,
                   cells: Sequence[tuple[float, float]] = ANCHOR_CELLS) -> list[tuple]:
    """(trace, fraction, multiplier, mechanism, arm, seed) of every anchor replay."""
    return [(ANCHOR_TRACE, fraction, multiplier, mechanism, arm, seed)
            for fraction, multiplier in cells for mechanism in MECHANISM_NAMES
            for arm in anchor_arms(mechanism, fraction, multiplier) for seed in seeds]


def main_replays(traces: Sequence[str] = TRACES, cells: Sequence[tuple[float, float]] = CELLS,
                 seeds: Sequence[int] = SEEDS,
                 mechanisms: Sequence[str] = MECHANISM_NAMES) -> list[tuple]:
    """(trace, fraction, multiplier, mechanism, arm, seed) of every sampled replay of A+B."""
    return [(name, fraction, multiplier, mechanism, arm, seed)
            for name in traces for fraction, multiplier in cells for mechanism in mechanisms
            for arm in cell_arms(fraction, multiplier) for seed in seeds]


def heap_replays(traces: Sequence[str] = TRACES,
                 cells: Sequence[tuple[float, float]] = CELLS) -> list[tuple]:
    """(trace, fraction, multiplier, "heap", H_lru|H_off, None) of every heap reference."""
    return [(name, fraction, multiplier, HEAP, arm, None)
            for name in traces for fraction, multiplier in cells for arm in HEAP_ARMS]


SMOKE_CELL = (0.01, 1.0)
SMOKE_SEED = 0


def smoke_replays(traces: Sequence[str] = TRACES) -> list[tuple]:
    """The smoke at 512 tokens: seed 0, cell 1% x 1, all16, the four A arms per trace."""
    return [(name, SMOKE_CELL[0], SMOKE_CELL[1], ALL16, arm, SMOKE_SEED)
            for name in traces for arm in a_arms(*SMOKE_CELL)]


def smoke_fine_replays() -> list[tuple]:
    """The smoke at 16 tokens: To-B, seed 0, cell 1% x 1 (512-token bytes), all16,
    the three C arms."""
    return [(GRANULARITY_TRACE, SMOKE_CELL[0], SMOKE_CELL[1], ALL16, arm, SMOKE_SEED)
            for arm in granularity_arms(*SMOKE_CELL)]


def granularity_replays(seeds: Sequence[int] = SEEDS,
                        cells: Sequence[tuple[float, float]] = CELLS) -> list[tuple]:
    """Control C: To-B at 16 tokens, all16, the three C arms, every cell and seed."""
    return [(GRANULARITY_TRACE, fraction, multiplier, ALL16, arm, seed)
            for fraction, multiplier in cells for arm in granularity_arms(fraction, multiplier)
            for seed in seeds]


assert (COUNT_A, COUNT_B, COUNT_SAMPLED, COUNT_HEAP) == (960, 1440, 2400, 48)
assert len(main_replays()) == COUNT_SAMPLED and len(heap_replays()) == COUNT_HEAP
assert sum(1 for task in main_replays() if part(task[4], task[1], task[2]) == "A") == COUNT_A
assert len(anchor_replays()) == 70
assert sum(1 for task in anchor_replays() if task[3] == ALL16) == 40
assert len(smoke_replays()) == 16 and len(smoke_fine_replays()) == 3
assert len(granularity_replays()) == 90 and len(granularity_replays(REDUCED_SEEDS)) == 54
assert all(key in ANCHOR_SOURCES for task in anchor_replays() for key in [(task[3], task[4])])

# --- the smoke's stop conditions (docs/bailian-external-check-plan.md, "Smoke, resources ...") --

MIB_PER_GIB = 1024.0
SMOKE_MAX_SECONDS = 15 * 60.0              # 512 tokens: a replay above this, the run is not started
SMOKE_MAX_RSS_MIB = 3 * MIB_PER_GIB        # 512 tokens: 3 GiB
FINE_MAX_SECONDS = 20 * 60.0               # 16 tokens: above this, C is reduced to seeds 0-2
FINE_MAX_RSS_MIB = 6 * MIB_PER_GIB         # 16 tokens: above this, the worker count is halved
FINE_MAX_PROJECTED_HOURS = 12.0            # the (reduced) control is not run above this
FINE_WORKERS = 4                           # C runs on at most 4 workers


def smoke_verdict(seconds: Sequence[float], rss_mib: Sequence[float],
                  integrity_ok: bool) -> dict[str, object]:
    """The 512-token stop condition: the run may start only when no smoke
    replay exceeds 15 minutes or 3 GiB peak RSS (strictly above stops it) and
    every integrity check passed."""
    seconds, rss_mib = [float(v) for v in seconds], [float(v) for v in rss_mib]
    max_seconds = max(seconds) if seconds else math.nan
    max_rss = max(rss_mib) if rss_mib else math.nan
    within = bool(seconds) and max_seconds <= SMOKE_MAX_SECONDS and max_rss <= SMOKE_MAX_RSS_MIB
    return {"replays": len(seconds), "max_seconds": max_seconds, "max_rss_mib": max_rss,
            "max_seconds_allowed": SMOKE_MAX_SECONDS, "max_rss_mib_allowed": SMOKE_MAX_RSS_MIB,
            "integrity_ok": bool(integrity_ok), "within_limits": within,
            "start": within and bool(integrity_ok)}


def fine_smoke_verdict(seconds: Sequence[float], rss_mib: Sequence[float],
                       integrity_ok: bool) -> dict[str, object]:
    """Control C's decision from the 16-token smoke: seeds 0-2 when a replay
    exceeds 20 minutes, half the workers (2 of 4) when a worker exceeds 6 GiB,
    and not run when the decided control is projected above 12 hours of wall
    time. The projection is `ceil(replays / workers) x` the longest smoke
    replay (the smoke runs one budget, 1% x 1)."""
    seconds, rss_mib = [float(v) for v in seconds], [float(v) for v in rss_mib]
    max_seconds = max(seconds) if seconds else math.nan
    max_rss = max(rss_mib) if rss_mib else math.nan
    reduce_seeds = bool(seconds) and max_seconds > FINE_MAX_SECONDS
    halve_workers = bool(rss_mib) and max_rss > FINE_MAX_RSS_MIB
    seeds = REDUCED_SEEDS if reduce_seeds else SEEDS
    workers = FINE_WORKERS // 2 if halve_workers else FINE_WORKERS
    replays = len(granularity_replays(seeds))
    projected_hours = (math.ceil(replays / workers) * max_seconds / 3600.0 if seconds
                       else math.nan)
    run = (bool(seconds) and bool(integrity_ok) and math.isfinite(projected_hours)
           and projected_hours <= FINE_MAX_PROJECTED_HOURS)
    return {"replays_measured": len(seconds), "max_seconds": max_seconds,
            "max_rss_mib": max_rss, "reduce_seeds": reduce_seeds,
            "halve_workers": halve_workers, "seeds": list(seeds), "max_workers": workers,
            "replays": replays, "projected_hours": projected_hours,
            "projection": "ceil(replays / max_workers) x max smoke seconds at 16 tokens",
            "integrity_ok": bool(integrity_ok), "run": run}


# --- the windows (docs/bailian-external-check-plan.md, "Inputs, fixed": Windows) ---------------

TAIL_EXCLUSION_SECONDS = 1200.0
TAIL_EXCLUSION_MS = TAIL_EXCLUSION_SECONDS * 1000.0
WINDOWS = ("full", "W", "W1", "W2")
PRIMARY_WINDOW = "W"
COLLECTED_WINDOWS = ("W", "W1", "W2")
# The window counters: the replay's (`TwoTierResult` fields) on the full
# window, the collector's on the others (`tailwindow`'s mapping).
COUNTERS = tailwindow.COUNTERS
# The window counters that cannot depend on the arm (the identifiers of a window).
WINDOW_IDENTIFIERS = ("measured_requests", "requested_tokens", "l1_avoided_tokens")
assert TAIL_EXCLUSION_MS == 1_200_000.0
assert set(WINDOW_IDENTIFIERS) <= set(COUNTERS)


def window_bounds(end_ms: float, split_ms: float) -> dict[str, float]:
    """The windows of a trace, in ms: full `[split, end]`; W `[split, end -
    1,200,000]`; W1 `[split, mid]`; W2 `[nextafter(mid), end - 1,200,000]`
    with `mid = split + (W end - split) / 2`. Refused when W is empty."""
    split, end = float(split_ms), float(end_ms)
    primary_end = end - TAIL_EXCLUSION_MS
    if not primary_end > split:
        raise ValueError(f"the primary window [{split}, {primary_end}] ms is empty: the trace "
                         f"ends {(end - split) / 1000.0:g} s after the split")
    mid = split + (primary_end - split) / 2.0
    second_start = math.nextafter(mid, math.inf)
    if second_start > primary_end:
        raise ValueError("the primary window is too short to halve")
    return {"split_ms": split, "primary_end_ms": primary_end, "mid_ms": mid,
            "second_start_ms": second_start, "end_ms": end,
            "full_seconds": (end - split) / 1000.0,
            "primary_seconds": (primary_end - split) / 1000.0,
            "first_half_seconds": (mid - split) / 1000.0,
            "second_half_seconds": (primary_end - mid) / 1000.0}


def window_ranges(bounds: Mapping[str, float]) -> dict[str, tuple[float, float]]:
    """Inclusive `(start_ms, end_ms)` of each window."""
    return {"full": (bounds["split_ms"], bounds["end_ms"]),
            "W": (bounds["split_ms"], bounds["primary_end_ms"]),
            "W1": (bounds["split_ms"], bounds["mid_ms"]),
            "W2": (bounds["second_start_ms"], bounds["primary_end_ms"])}


def window_of(timestamp_ms: float, bounds: Mapping[str, float]) -> tuple[str, ...]:
    """The windows a request at `timestamp_ms` is counted in."""
    return tuple(window for window, (start, end) in window_ranges(bounds).items()
                 if start <= timestamp_ms <= end)


def window_collectors(trace: Trace, split_ms: float) -> dict[str, LabelWindowUtilityCollector]:
    """The three read-only collectors of W, W1 and W2 (in that order)."""
    ranges = window_ranges(window_bounds(trace.end_ms, split_ms))
    return {window: LabelWindowUtilityCollector(trace, *ranges[window])
            for window in COLLECTED_WINDOWS}


def window_counters(result, collectors: Mapping[str, LabelWindowUtilityCollector]
                    ) -> dict[str, dict[str, int]]:
    """Every window's counters: the replay's own on the full window
    (`tailwindow.full_row`), each collector's on its window
    (`tailwindow.head_row`, the same column mapping)."""
    out = {"full": tailwindow.full_row(result)}
    for window in COLLECTED_WINDOWS:
        out[window] = tailwindow.head_row(collectors[window])
    return out


def extra_tokens(counters: Mapping[str, int]) -> int:
    """`U` in tokens: avoided prefill tokens beyond L1 alone."""
    return tailwindow.extra_tokens(counters)


def points(tokens: float, requested: int) -> float:
    """`100 * tokens / requested` with the parents' guard `max(requested, 1)`."""
    return tailwindow.points(tokens, requested)


def utility_points(counters: Mapping[str, int]) -> float:
    """`U` in points of the window's input tokens (`tailwindow.utility_points`)."""
    return tailwindow.utility_points(counters)


def window_columns(counters: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    """`<window>_<counter>`, `<window>_extra_avoided_tokens` and
    `U_<window>_points` for every window."""
    out: dict[str, object] = {}
    for window in WINDOWS:
        values = counters[window]
        for counter in COUNTERS:
            out[f"{window}_{counter}"] = int(values[counter])
        out[f"{window}_extra_avoided_tokens"] = extra_tokens(values)
        out[f"U_{window}_points"] = utility_points(values)
    return out


def counters_of(row: Mapping[str, object], window: str) -> dict[str, int]:
    """A window's counters read back from a row of `window_columns`."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}")
    return {counter: int(row[f"{window}_{counter}"]) for counter in COUNTERS}


def window_utility(row: Mapping[str, object], window: str) -> tuple[int, float]:
    """`U` of a replay on a window: integer tokens and points, from its counters."""
    counters = counters_of(row, window)
    return extra_tokens(counters), utility_points(counters)


def window_difference(later: Mapping[str, object], earlier: Mapping[str, object],
                      window: str) -> tuple[int, float]:
    """`U(later) - U(earlier)` of one seed on a window: integer tokens and
    points of `later`'s window input tokens."""
    tokens = (extra_tokens(counters_of(later, window))
              - extra_tokens(counters_of(earlier, window)))
    return tokens, points(tokens, int(later[f"{window}_requested_tokens"]))


def window_problems(counters: Mapping[str, Mapping[str, int]]) -> list[str]:
    """The plan's window check on one replay: every counter non-negative, W at
    most full, and W1 + W2 = W, counter by counter."""
    problems = []
    for counter in COUNTERS:
        values = {window: int(counters[window][counter]) for window in WINDOWS}
        for window, value in values.items():
            if value < 0:
                problems.append(f"{window} {counter} {value} < 0")
        if values["W"] > values["full"]:
            problems.append(f"W {counter} {values['W']} > full {values['full']}")
        if values["W1"] + values["W2"] != values["W"]:
            problems.append(f"W1 + W2 {counter} {values['W1']} + {values['W2']} != W "
                            f"{values['W']}")
    return problems


# --- the readings (docs/bailian-external-check-plan.md, "Readings, fixed before the run") -----

SUFFICES_SHORTFALL = horizonctl.SUFFICES_SHORTFALL               # 0.10
# A trace x cell is evaluable when the five-seed mean of U(label) - U(lru) on
# the primary window is at least 1.0 point; a ratio whose denominator is below
# 1.0 point is not computed ("no headroom"); no ratio is clipped.
EVALUABLE_POINTS = 1.0
MIN_DENOMINATOR_POINTS = 1.0
NO_HEADROOM = "no_headroom"
CLASSES = ("transplant_suffices", "retuned_grid_point_suffices", "none_in_grid", NO_HEADROOM)
DIRECTIONS = ("longer", "shorter", "equal")
FLAG_HORIZON_SECONDS = 1200.0
CONSISTENT_LOSS = "consistent_loss"
SIGN_READINGS = ("consistent_gain", CONSISTENT_LOSS, "mixed")
# Prediction 4's L2 levels: L2 = f x k of W, ascending; the two 1% cells are one level.
L2_LEVELS = (0.0025, 0.01, 0.02, 0.04, 0.08)
L2_LEVEL_OF_CELL = {(0.0025, 1.0): 0.0025, (0.0025, 4.0): 0.01, (0.01, 1.0): 0.01,
                    (0.02, 1.0): 0.02, (0.01, 4.0): 0.04, (0.02, 4.0): 0.08}
MIN_ASSESSABLE_LEVELS = 3
GRID_TRACE_CELLS = len(TRACES) * len(CELLS)                     # 24 per mechanism

assert SUFFICES_SHORTFALL == 0.10
assert set(L2_LEVEL_OF_CELL) == set(CELLS)
assert all(math.isclose(level, cell[0] * cell[1]) for cell, level in L2_LEVEL_OF_CELL.items())
assert tuple(sorted(set(L2_LEVEL_OF_CELL.values()))) == L2_LEVELS
assert GRID_TRACE_CELLS == 24

# The six predictions, fixed before any replay: absolute counts (they do not
# shrink when fewer cells are evaluable). Prediction 2 compares two counts.
PREDICTIONS = {
    "1_transplant_all16": {"mechanism": ALL16, "counted": "S_hstar_at_most_0.10",
                           "at_least": 12, "of": GRID_TRACE_CELLS, "unit": "trace x cell"},
    "2_direction_of_failure_all16": {"mechanism": ALL16,
                                     "counted": "h_best_longer_than_h_star_minus_shorter",
                                     "at_least": None, "of": None, "unit": "trace x cell",
                                     "rule": "longer > shorter among evaluable cells where "
                                             "S_h* > 0.10; 0/0 when there is none"},
    "3_grid_all16": {"mechanism": ALL16, "counted": "min_h_S_h_at_most_0.10",
                     "at_least": 20, "of": GRID_TRACE_CELLS, "unit": "trace x cell"},
    "4_monotonicity_all16": {"mechanism": ALL16, "counted": "h_best_nondecreasing_in_L2",
                             "at_least": 3, "of": len(TRACES), "unit": "trace"},
    "5_random_order_all16": {"mechanism": ALL16, "counted": "D_rand_consistent_loss",
                             "at_least": 16, "of": GRID_TRACE_CELLS, "unit": "trace x cell"},
    "6_grid_leaf16": {"mechanism": LEAF16, "counted": "min_h_S_h_at_most_0.10",
                      "at_least": 16, "of": GRID_TRACE_CELLS, "unit": "trace x cell"},
    # Addendum 1 (2026-10-03, before any replay): reading 7.
    "7_order_beyond_bit_separated": {"mechanism": f"{LEAF16}_vs_{ALL16}",
                                     "counted": "Delta_hstar_separated_leaf16_above_all16",
                                     "at_least": 10,
                                     "of": len(TRACES) * sum(1 for cell in CELLS
                                                             if HSTAR_SECONDS[cell] < 600.0),
                                     "unit": "trace x cell",
                                     "rule": "over the trace x cell with h* < 600 s evaluable "
                                             "under both mechanisms"},
}
assert [spec["at_least"] for spec in PREDICTIONS.values()] == [12, None, 20, 3, 16, 16, 10]


def mean(values: Iterable[float]) -> float:
    """The five-seed mean of the tables (`gap.summarize`, finite values only)."""
    return summarize(list(values))["mean"]


def ratio(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, not computed (nan) when the denominator is
    below 1.0 point or not finite; never clipped."""
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.nan
    if denominator < MIN_DENOMINATOR_POINTS:
        return math.nan
    return numerator / denominator


def shortfall(u_label: float, u_arm: float, u_lru: float) -> float:
    """`S = (U(label) - U(arm)) / (U(label) - U(lru))` in points; nan ("no
    headroom") when the denominator is below 1.0 point. Not clipped: above 1
    or below 0 is reported as is."""
    return ratio(u_label - u_arm, u_label - u_lru)


def evaluable(u_label_mean: float, u_lru_mean: float) -> bool:
    """The plan's rule: the five-seed mean of U(label) - U(lru) on the primary
    window is at least 1.0 point (exactly 1.0 is evaluable; a nan is not)."""
    headroom = u_label_mean - u_lru_mean
    return math.isfinite(headroom) and headroom >= EVALUABLE_POINTS


def suffices(s: float) -> bool:
    """`S <= 0.10` (exactly 0.10 suffices; a nan does not)."""
    return math.isfinite(s) and s <= SUFFICES_SHORTFALL


def best_horizon(shortfalls: Mapping[float, float]) -> tuple[float | None, float]:
    """`h_best = argmin_h S_h` over the computed S_h, ties to the smaller h,
    and `min_h S_h`; (None, nan) when no S_h is computed."""
    finite = {float(h): float(s) for h, s in shortfalls.items() if math.isfinite(s)}
    if not finite:
        return None, math.nan
    smallest = min(finite.values())
    return min(h for h, s in finite.items() if s == smallest), smallest


def sufficing_horizons(shortfalls: Mapping[float, float]) -> tuple[float, ...]:
    """The horizons with `S_h <= 0.10`, ascending."""
    return tuple(sorted(float(h) for h, s in shortfalls.items() if suffices(s)))


def rests_on_1200(h_best: float | None, sufficing: Sequence[float] = ()) -> bool:
    """The plan's flag: the reading rests on the 1,200-second point when
    `h_best` is 1,200 s or 1,200 s is the only sufficing horizon."""
    return (h_best == FLAG_HORIZON_SECONDS
            or tuple(float(h) for h in sufficing) == (FLAG_HORIZON_SECONDS,))


def classify(is_evaluable: bool, s_hstar: float, min_s: float) -> str:
    """The plan's classification of a cell, in its order: "transplant_suffices"
    (S_h* <= 0.10), "retuned_grid_point_suffices" (reading 1 fails, min_h S_h
    <= 0.10), "none_in_grid" (both fail), "no_headroom" (not evaluable)."""
    if not is_evaluable:
        return NO_HEADROOM
    if suffices(s_hstar):
        return CLASSES[0]
    if suffices(min_s):
        return CLASSES[1]
    return CLASSES[2]


def direction(h_star: float, h_best: float | None) -> str:
    """Where the best grid horizon lies relative to h*: longer, shorter or equal
    ("" when there is no best)."""
    if h_best is None:
        return ""
    if h_best > h_star:
        return "longer"
    if h_best < h_star:
        return "shorter"
    return "equal"


def direction_counts(entries: Iterable[tuple[str, float, float | None]]) -> dict[str, int]:
    """Prediction 2's counts over `(classification, h*, h_best)` of every cell:
    among the evaluable cells where the transplant does not suffice, how often
    the best grid horizon is longer, shorter or equal to h*."""
    counts = {"failing": 0, **dict.fromkeys(DIRECTIONS, 0)}
    for label, h_star, h_best in entries:
        if label not in CLASSES:
            raise ValueError(f"unknown classification {label!r}")
        if label in (CLASSES[0], NO_HEADROOM):
            continue
        counts["failing"] += 1
        counts[direction(h_star, h_best)] += 1
    return counts


def direction_holds(counts: Mapping[str, int]) -> bool | None:
    """Prediction 2: longer in more cells than shorter; None ("0/0") when no
    evaluable cell fails the transplant."""
    if counts["failing"] == 0:
        return None
    return counts["longer"] > counts["shorter"]


def level_bests(bests: Mapping[tuple[float, float], float | None]) -> dict[float, float | None]:
    """Prediction 4: the best grid horizon per L2 level from `cell -> h_best`
    (None for a cell that is not evaluable, which is left out); the two cells
    of the 1% level are one level, taking the smaller best. A level with no
    evaluable cell is None."""
    out: dict[float, float | None] = dict.fromkeys(L2_LEVELS)
    for cell, best in bests.items():
        level = L2_LEVEL_OF_CELL[(float(cell[0]), float(cell[1]))]
        if best is None:
            continue
        out[level] = best if out[level] is None else min(out[level], best)
    return out


def monotonicity(levels: Mapping[float, float | None]) -> dict[str, object]:
    """Prediction 4 for one trace: the evaluable levels in L2 order, whether
    the trace is assessable (at least three evaluable levels) and, if so,
    whether the best horizon is nondecreasing over them (None otherwise)."""
    sequence = [(level, levels[level]) for level in L2_LEVELS if levels.get(level) is not None]
    assessable = len(sequence) >= MIN_ASSESSABLE_LEVELS
    nondecreasing = None
    if assessable:
        bests = [best for _, best in sequence]
        nondecreasing = all(later >= earlier for earlier, later in zip(bests, bests[1:]))
    return {"evaluable_levels": len(sequence), "assessable": assessable,
            "nondecreasing": nondecreasing,
            "sequence": tuple(sequence),
            "rests_on_1200": any(best == FLAG_HORIZON_SECONDS for _, best in sequence)}


def calibration(first_half: Mapping[float, float],
                second_half: Mapping[float, float]) -> dict[str, object]:
    """Reading 4 (descriptive): `h_cal = argmin_h S_h` on W1 (ties to the
    smaller h), then `S_{h_cal}` on W2 and whether it is at most 0.10, beside
    `min_h S_h` on W2. `h_cal` is None when no S_h is computed on W1."""
    h_cal, s_cal_first = best_horizon(first_half)
    s_cal_second = second_half[h_cal] if h_cal is not None else math.nan
    h_best_second, min_second = best_horizon(second_half)
    return {"h_cal": h_cal, "S_hcal_W1": s_cal_first, "S_hcal_W2": s_cal_second,
            "hcal_suffices_W2": suffices(s_cal_second),
            "h_best_W2": h_best_second, "min_S_W2": min_second,
            "min_S_W2_suffices": suffices(min_second),
            "rests_on_1200": h_cal == FLAG_HORIZON_SECONDS or rests_on_1200(
                h_best_second, sufficing_horizons(second_half))}


def label_share_of_t(u_label: float, u_lru: float, u_hoff: float) -> float:
    """`(U(label) - U(lru)) / T` with `T = H_off - U(lru)`; not computed when T
    is below 1.0 point; never clipped."""
    return ratio(u_label - u_lru, u_hoff - u_lru)


def seed_reading(values: Sequence[float]) -> str:
    """consistent_gain / consistent_loss / mixed (`mechanism.sign_reading`)."""
    return sign_reading(list(values))


def seed_signs(values: Sequence[float]) -> tuple[int, int, int]:
    """(positive, zero, negative) seed counts (`mechanism.sign_counts`)."""
    return sign_counts(list(values))


def prediction_holds(name: str, count: int) -> bool:
    """Whether an absolute-count prediction holds: `count >= at_least`
    (prediction 2 is `direction_holds`)."""
    spec = PREDICTIONS[name]
    if spec["at_least"] is None:
        raise ValueError(f"{name} is not a count prediction")
    return int(count) >= int(spec["at_least"])


# --- reading 7, order beyond the bit, by mechanism (plan Addendum 1) ---------------------------

ORDER_HORIZON_ROLES = ("h_star", "600")
ORDER_REFERENCE_HORIZON = 600.0
# Prediction 7 counts the trace x cell whose h* is below 600 s (16 of 24).
PREDICTION_7_CELLS = tuple(cell for cell in CELLS if HSTAR_SECONDS[cell] < ORDER_REFERENCE_HORIZON)
PREDICTION_7_TRACE_CELLS = len(TRACES) * len(PREDICTION_7_CELLS)
CONSISTENT_READINGS = ("consistent_gain", CONSISTENT_LOSS)
assert len(PREDICTION_7_CELLS) == 4 and PREDICTION_7_TRACE_CELLS == 16
assert PREDICTIONS["7_order_beyond_bit_separated"]["of"] == PREDICTION_7_TRACE_CELLS


def order_horizon(role: str, fraction: float, multiplier: float) -> float:
    """The horizon of reading 7's role: h* of the cell, or 600 s."""
    if role == "h_star":
        return hstar(fraction, multiplier)
    if role == "600":
        return ORDER_REFERENCE_HORIZON
    raise ValueError(f"unknown reading 7 horizon role {role!r}")


def order_deltas(u_label: Sequence[float], u_rung: Sequence[float]) -> list[float]:
    """`Delta_m(h) = U_m(label) - U_m(label_binary_h)`, seed-paired within one
    mechanism (the same seed order on both sides)."""
    if len(u_label) != len(u_rung):
        raise ValueError("Delta needs the same seeds on both sides")
    return [label - rung for label, rung in zip(u_label, u_rung)]


def separated(leaf16: Sequence[float], all16: Sequence[float]) -> bool:
    """The smallest seed value of Delta under leaf16 exceeds the largest under
    all16 (strictly: a tie is not separated); False without values."""
    leaf16, all16 = list(leaf16), list(all16)
    if not leaf16 or not all16 or not all(math.isfinite(v) for v in leaf16 + all16):
        return False
    return min(leaf16) > max(all16)


def reversed_signs(leaf16_reading: str, all16_reading: str) -> bool:
    """Both seed readings consistent (every seed of one sign) and opposite."""
    for reading in (leaf16_reading, all16_reading):
        if reading not in SIGN_READINGS:
            raise ValueError(f"unknown seed reading {reading!r}")
    return (leaf16_reading in CONSISTENT_READINGS and all16_reading in CONSISTENT_READINGS
            and leaf16_reading != all16_reading)


def order_status(evaluable_all16: bool, evaluable_leaf16: bool) -> str:
    """"evaluable" when the cell is evaluable under both mechanisms (the plan's
    rule applied to each), "no_headroom" otherwise."""
    return "evaluable" if (evaluable_all16 and evaluable_leaf16) else NO_HEADROOM


def prediction_7_count(entries: Iterable[tuple[tuple[float, float], str, bool]]) -> dict[str, int]:
    """Prediction 7 from `(cell, status, separated at h*)` of every trace x
    cell: the h* < 600 s trace x cell, those evaluable under both mechanisms,
    and the separated ones among them."""
    counts = {"cells": 0, "evaluable": 0, "separated": 0}
    for cell, status, is_separated in entries:
        if (float(cell[0]), float(cell[1])) not in PREDICTION_7_CELLS:
            continue
        counts["cells"] += 1
        if status != "evaluable":
            continue
        counts["evaluable"] += 1
        counts["separated"] += bool(is_separated)
    return counts


def cell_window_values(u_points: Mapping[str, Sequence[float]], h_star: float,
                       u_hoff: float = math.nan) -> dict[str, object]:
    """Readings 1-3 of one trace x cell x mechanism on one window, from the
    per-seed U (points, seed order) of lru, label, every grid arm and the
    random arm at h*, and the heap offline reference `u_hoff` (points).

    Returns the five-seed means, the headroom `U(label) - U(lru)`, S_h at every
    grid point (on the means) with its per-seed values, h_best, min_h S_h,
    the sufficing horizons, S_h*, the 1,200-second flag, D_rand per seed with
    its reading, T and the label's share of T."""
    means = {arm: mean(values) for arm, values in u_points.items()}
    headroom = means[LABEL] - means[LRU]
    shortfalls = {h: shortfall(means[LABEL], means[horizon_arm(h)], means[LRU])
                  for h in GRID_SECONDS}
    per_seed = {h: [shortfall(label, arm, lru) for label, arm, lru in
                    zip(u_points[LABEL], u_points[horizon_arm(h)], u_points[LRU])]
                for h in GRID_SECONDS}
    h_best, min_s = best_horizon(shortfalls)
    sufficing = sufficing_horizons(shortfalls)
    rand = random_arm(h_star)
    d_rand = [later - earlier for later, earlier in zip(u_points[rand],
                                                        u_points[horizon_arm(h_star)])]
    return {"means": means, "headroom": headroom,
            "evaluable": evaluable(means[LABEL], means[LRU]),
            "S": shortfalls, "S_seeds": per_seed, "h_best": h_best, "min_S": min_s,
            "sufficing": sufficing, "S_hstar": shortfalls[float(h_star)],
            "rests_on_1200": rests_on_1200(h_best, sufficing),
            "D_rand": d_rand, "D_rand_reading": seed_reading(d_rand),
            "T": u_hoff - means[LRU],
            "label_share_of_T": label_share_of_t(means[LABEL], means[LRU], u_hoff)}
