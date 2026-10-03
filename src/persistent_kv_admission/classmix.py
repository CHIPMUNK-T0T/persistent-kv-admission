"""Class-order mix — which reuse class carries the learned order's loss at the matched horizon, and does recency carry information beyond the bit.

`docs/class-order-mix-plan.md` is the pre-registration. It is a follow-up of
the matched class-order control (`matchedorder`,
`docs/matched-class-order-plan.md`), whose two arms at the cell's matched
horizon `h*` ordered both reuse classes by the frozen `next_use` ranker
(`evict_binary_{h*}_learned`) or both by recency (`evict_binary_{h*}_recency`).
Three arms take them apart. All three have admission by the frozen ranker
through `errorloc.HybridOverride`, unchanged (in a first-round decision in
which the arrival is a candidate, the arrival is rejected if and only if it is
the first minimum of the ranker's keys `(ranker score, last_group)`; otherwise,
and in every later round, the victim is the first minimum of the store's
keys), and the store's key `((reusable within h*, within-class key),
last_group)`:

* `mix_out_learned_{h}`: the within-class key is the ranker's score for a
  candidate that is not reusable within h and a constant for one that is, so
  the non-reusable class is ordered by the ranker and the reusable class by
  recency (`MixedClassScorer` with `learned_class=0`);
* `mix_in_learned_{h}`: the reverse, the reusable class by the ranker and the
  non-reusable class by recency (`learned_class=1`);
* `random_within_class_{h}`: the within-class key is a uniform draw in [0, 1)
  per candidate per decision (`RandomClassScorer`), from a stream of its own,
  seeded by the replay seed and the arm name (`random_stream_seed`). The
  store's sampling stream (`_VictimStore.rng`, seeded by `l2_seed`) is never
  read or advanced by it, so the candidate sets are drawn by the store exactly
  as for the recency arm, given the same earlier choices.

Reused unchanged: the reuse label at h (`matchedorder.reuse_label`, the label
rung's `mechanism.ExactLabelScorer(trace, h, target="binary")`: 1 if the next
use is at most h seconds away, else 0), the frozen ranker as
`decisionpop.RawFeatureScorer`, `errorloc.ArmSetup`, `errorloc.HybridOverride`
and `errorloc.HybridScorer`, the matched horizons `h*` of
`matchedorder.MATCHED_HORIZON_SECONDS`, the recovery share
`horizonctl.recovery`, and the seed-sign readings `mechanism.sign_counts` /
`mechanism.sign_reading` (through `matchedorder.seed_signs`).

The mixed arms' construction contains the matched control's two arms as its
null settings: with the ranker on both classes `MixedClassScorer`'s score is
`horizonctl.ReuseClassScorer`'s, and with the ranker on neither class the
store's key orders exactly as `(reusable within h, last_group)`, the recency
arm's. The tests check both identities decision by decision.

The reading helpers at the end are pure arithmetic on numbers the replays and
the published rows produced, kept here so that they can be tested. Nothing
here fits, adds a feature or target, or changes an existing replay path;
every arm reads the trace's future on purpose and none is a proposed policy.
`h*` is the best of grids run on these traces, read after the fact.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

from .decisionpop import RawFeatureScorer
from .errorloc import ArmSetup, HybridOverride, HybridScorer, hash_bits
from .gap import summarize
# `recovery`, `seed_signs` and `sign_reading_counts` are re-exported for the
# runner and used as they are.
from .horizonctl import recovery
from .matchedorder import (
    MATCHED_HORIZON_SECONDS,
    MATCHED_HORIZONS_SECONDS,
    MECHANISM,
    RUNG_OF_HORIZON,
    SIGN_READINGS,
    matched_arm,
    matched_horizon,
    reuse_label,
    seed_signs,
    sign_reading_counts,
)
from .trace import Request, Trace

# --- the arms (docs/class-order-mix-plan.md, "Fixed surface") ----------------------------------

# The two reuse classes, as the reuse label scores them.
NON_REUSABLE, REUSABLE = 0, 1
CLASSES = (NON_REUSABLE, REUSABLE)
# The null settings of the mixed construction (tests only): the ranker on
# both classes is the matched learned arm, on neither the matched recency arm.
BOTH_CLASSES: tuple[int, ...] = (NON_REUSABLE, REUSABLE)
NEITHER_CLASS: tuple[int, ...] = ()
# The within-class key of the class the ranker does not order: one constant,
# so that `last_group` (recency), the store's second key, decides there.
WITHIN_CLASS_CONSTANT = 0.0
# The three arm kinds, in table order.
MIX_OUT, MIX_IN, RANDOM = "mix_out_learned", "mix_in_learned", "random_within_class"
KINDS = (MIX_OUT, MIX_IN, RANDOM)
# Mixed arm kind -> the class whose candidates the ranker orders.
LEARNED_CLASS = {MIX_OUT: NON_REUSABLE, MIX_IN: REUSABLE}
# The random arm's stream: `random.Random(random_stream_seed(seed, arm))`.
RANDOM_STREAM_TAG = "class_order_mix/random_within_class"


def mix_arm(kind: str, horizon_seconds: float) -> str:
    """`mix_out_learned_{h:g}`, `mix_in_learned_{h:g}` or `random_within_class_{h:g}`."""
    if kind not in KINDS:
        raise ValueError(f"unknown arm kind {kind!r}")
    return f"{kind}_{float(horizon_seconds):g}"


ARMS = tuple(mix_arm(kind, h) for h in MATCHED_HORIZONS_SECONDS for kind in KINDS)
HORIZON_OF_ARM = {mix_arm(kind, h): float(h) for h in MATCHED_HORIZONS_SECONDS for kind in KINDS}
KIND_OF_ARM = {mix_arm(kind, h): kind for h in MATCHED_HORIZONS_SECONDS for kind in KINDS}

assert MECHANISM == "all16"
assert MATCHED_HORIZONS_SECONDS == (60.0, 150.0, 300.0, 600.0)
assert len(ARMS) == 12 and len(set(ARMS)) == 12


def cell_arms(fraction: float, multiplier: float) -> tuple[str, str, str]:
    """The three arms replayed at a cell, at its h*: (mix_out, mix_in, random)."""
    h = matched_horizon(fraction, multiplier)
    return tuple(mix_arm(kind, h) for kind in KINDS)


def matched_reference_arms(fraction: float, multiplier: float) -> dict[str, str]:
    """The published arms the cell's readings use, by role: the matched
    control's two arms at h* and the label rung at h*."""
    h = matched_horizon(fraction, multiplier)
    return {"matched_learned": matched_arm(h, "learned"),
            "matched_recency": matched_arm(h, "recency"),
            "label_binary_hstar": RUNG_OF_HORIZON[h]}


# --- the scorers ------------------------------------------------------------------------------


def _classes(learned_class) -> frozenset:
    """`learned_class` as a set of classes: an int names one class, a tuple
    (the null settings `NEITHER_CLASS`, `BOTH_CLASSES`) several."""
    if isinstance(learned_class, int) and not isinstance(learned_class, bool):
        classes = (learned_class,)
    elif isinstance(learned_class, tuple):
        classes = learned_class
    else:
        raise ValueError(f"a reuse class is 0 or 1 (or a tuple of them), got {learned_class!r}")
    for value in classes:
        if isinstance(value, bool) or not isinstance(value, int) or value not in CLASSES:
            raise ValueError(f"a reuse class is 0 or 1, got {value!r}")
    return frozenset(classes)


def _reuse_bit(reuse, state_id: str, timestamp_ms: float) -> float:
    bit = reuse.score(state_id, timestamp_ms)
    if bit not in CLASSES:
        raise ValueError(f"the reuse label scored {state_id} {bit!r}, not 0 or 1")
    return bit


class MixedClassScorer:
    """The store's scorer of `mix_out_learned` and `mix_in_learned`.

    The score is the pair `(bit, within)`: `bit` the exact reuse label (0 or
    1, `ExactLabelScorer(..., "binary")`), and `within` the ranker's score
    `within.score(...)` when `bit` is the class the ranker orders
    (`learned_class`), else `WITHIN_CLASS_CONSTANT`. The store appends
    `last_group` and tuples compare element by element, so its key
    `((bit, within), last_group)` puts every non-reusable state below every
    reusable one; within the ranker's class it orders by the ranker's score
    (recency on a tie); within the other class every candidate carries the
    same constant, so `last_group` (recency) alone decides there. The
    constant is never compared across classes, because the bit differs.

    `learned_class` is 0 (`mix_out_learned`: the non-reusable class by the
    ranker) or 1 (`mix_in_learned`: the reusable class by the ranker). The
    construction's two null settings are given as tuples: `BOTH_CLASSES`,
    whose score is `horizonctl.ReuseClassScorer`'s exactly (the matched
    learned arm), and `NEITHER_CLASS`, whose key orders exactly as
    `(bit, last_group)` (the matched recency arm).

    `within` must be the ranker object `errorloc.HybridOverride` consults for
    admission. `observe` forwards every observation to the reuse label and to
    `within` once each, in `ReuseClassScorer`'s order, so the ranker sees
    exactly the history it would see as the store's own scorer, whichever
    class it orders. The ranker is scored only for candidates of its class;
    its score reads the observed history and changes nothing.
    """

    time_varying = True

    def __init__(self, reuse, within, learned_class) -> None:
        if reuse is within:
            raise ValueError("the reuse label and the within-class score must be distinct scorers")
        self.reuse = reuse
        self.within = within
        self.learned_classes = _classes(learned_class)

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.reuse.observe(requests, timestamp_ms)
        self.within.observe(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> tuple[float, float]:
        bit = _reuse_bit(self.reuse, state_id, timestamp_ms)
        if bit in self.learned_classes:
            return (bit, self.within.score(state_id, timestamp_ms))
        return (bit, WITHIN_CLASS_CONSTANT)


def random_stream_seed(seed: int, arm: str) -> int:
    """The random arm's stream seed: 64 bits of BLAKE2b over
    (`RANDOM_STREAM_TAG`, replay seed, arm name), `errorloc.hash_bits`. A pure
    function of its arguments, the same in every process, independent of the
    trace and of `PYTHONHASHSEED`."""
    return hash_bits(RANDOM_STREAM_TAG, int(seed), str(arm))


class RandomClassScorer:
    """The store's eviction scorer of `random_within_class`.

    The score is the pair `(bit, u)`: `bit` the exact reuse label (0 or 1) and
    `u` a fresh uniform draw in [0, 1), `random.Random.random()` of a stream
    owned by this object and seeded with `random_stream_seed(seed, arm)`. The
    store's key `((bit, u), last_group)` puts every non-reusable state below
    every reusable one and orders each class by `u` (`last_group` decides only
    on an exact tie of two draws).

    The store scores every candidate of a decision exactly once, in draw
    order, when it builds the decision set (`_VictimStore._sampled_key`), and
    nothing else calls this score (`HybridOverride` consults the admission
    ranker, the statistics their own labels): one draw per candidate per
    decision, counted in `draws`. The stream is a separate `random.Random`
    instance: it never reads or advances the store's sampling stream
    (`_VictimStore.rng`, seeded by `l2_seed`) or Python's global one, and this
    object holds no reference to the store (it has no `attach`).

    `observe` forwards to the reuse label only, which ignores history. The
    arm wraps this scorer in `errorloc.HybridScorer` with the admission ranker,
    as the recency arm wraps the reuse label, so the ranker is shown every
    observation exactly as there.
    """

    time_varying = True

    def __init__(self, reuse, seed: int, arm: str) -> None:
        self.reuse = reuse
        self.seed = int(seed)
        self.arm = str(arm)
        self.stream_seed = random_stream_seed(self.seed, self.arm)
        self.rng = random.Random(self.stream_seed)
        self.draws = 0

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.reuse.observe(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> tuple[float, float]:
        bit = _reuse_bit(self.reuse, state_id, timestamp_ms)
        self.draws += 1
        return (bit, self.rng.random())


def mixed_setup(arm: str, trace: Trace, admission, learned_class,
                horizon_seconds: float) -> ArmSetup:
    """"Admission by `admission`, eviction by `((bit, within), last_group)`"
    with the ranker on `learned_class` (`MixedClassScorer`); `admission` is the
    ranker and the within-class score both, the one object the override
    consults. The override is `HybridOverride(admission)`, as in
    `matchedorder.class_order_setup`."""
    scorer = MixedClassScorer(reuse_label(trace, horizon_seconds), admission, learned_class)
    return ArmSetup(arm, "learned", scorer=scorer, override=HybridOverride(admission))


def random_setup(arm: str, trace: Trace, admission, horizon_seconds: float, seed: int,
                 stream_arm: str | None = None) -> ArmSetup:
    """"Admission by `admission`, eviction by `((bit, u), last_group)`".

    The store's scorer is `HybridScorer(admission, RandomClassScorer(...))`,
    built as `matchedorder.class_order_setup(order="recency")` builds
    `HybridScorer(admission, reuse label)`: the admission ranker is observed
    first, then the eviction scorer (which forwards to the reuse label), and
    the store's score is the eviction scorer's. The stream is seeded by `seed`
    and `stream_arm` (default: `arm`)."""
    stream = RandomClassScorer(reuse_label(trace, horizon_seconds), seed,
                               arm if stream_arm is None else stream_arm)
    return ArmSetup(arm, "learned", scorer=HybridScorer(admission, stream),
                    override=HybridOverride(admission))


def arm_setup(arm: str, trace: Trace, learned_ranker, seed: int) -> ArmSetup:
    """The replay of `arm` on `trace` at sampling seed `seed` under `all16`
    (applied by the caller): the frozen pi0 `next_use` ranker `learned_ranker`
    as `RawFeatureScorer`, as `matchedorder.arm_setup` builds the matched arms.
    `seed` seeds the random arm's own stream (with the arm name); the mixed
    arms do not read it. A fresh scorer is built per call, because the history
    scorers and the random stream carry state across a replay."""
    if arm not in HORIZON_OF_ARM:
        raise ValueError(f"unknown arm {arm!r}")
    if learned_ranker is None:
        raise ValueError(f"{arm} needs the learned ranker")
    h = HORIZON_OF_ARM[arm]
    kind = KIND_OF_ARM[arm]
    admission = RawFeatureScorer(trace, learned_ranker)
    if kind == RANDOM:
        return random_setup(arm, trace, admission, h, seed)
    return mixed_setup(arm, trace, admission, LEARNED_CLASS[kind], h)


def random_scorer(setup: ArmSetup) -> RandomClassScorer | None:
    """The random stream of a `random_within_class` setup, None for the others."""
    eviction = getattr(setup.scorer, "eviction", None)
    return eviction if isinstance(eviction, RandomClassScorer) else None


# --- the readings (docs/class-order-mix-plan.md, "Readings, fixed before the run") -------------

# Seed-paired differences, in points: name -> (later, earlier), each one of
# the three arm kinds or the matched control's two arms (published rows).
DIFFERENCES = {
    "D_out": (MIX_OUT, "matched_recency"),
    "D_in": (MIX_IN, "matched_recency"),
    "D_rand": (RANDOM, "matched_recency"),
    "D_learned": ("matched_learned", "matched_recency"),
    "random_minus_learned": (RANDOM, "matched_learned"),
}
# Prediction 1's eight trace x cell, as the plan names them: those in which
# the learned order within the matched class was a consistent loss.
PREDICTION_ONE_CELLS = (
    ("conversation_trace", 0.0025, 1.0), ("conversation_trace", 0.0025, 4.0),
    ("conversation_trace", 0.01, 1.0), ("conversation_trace", 0.02, 1.0),
    ("toolagent_trace", 0.0025, 1.0), ("toolagent_trace", 0.01, 1.0),
    ("toolagent_trace", 0.01, 4.0), ("toolagent_trace", 0.02, 1.0),
)
# Prediction 2: D_rand a consistent loss in at least 8 of the 12 trace x cell.
PREDICTION_TWO_MINIMUM = 8
PREDICTION_TWO_CELLS = 12
CONSISTENT_LOSS = SIGN_READINGS[1]
assert len(PREDICTION_ONE_CELLS) == 8 and len(set(PREDICTION_ONE_CELLS)) == 8
assert all(cell[1:] in MATCHED_HORIZON_SECONDS for cell in PREDICTION_ONE_CELLS)
assert CONSISTENT_LOSS == "consistent_loss"


def in_prediction_one(cell: Sequence) -> bool:
    """Whether (trace, l1_fraction, l2_multiplier) is one of the eight named cells."""
    return (str(cell[0]), float(cell[1]), float(cell[2])) in PREDICTION_ONE_CELLS


def class_conditions(d_in: float, d_out: float) -> dict[str, bool]:
    """Reading 1 on five-seed means: `D_in < 0`, `D_out > D_in`, and both. A
    zero is not below zero, an equality is not above, and a nan satisfies
    nothing."""
    negative = bool(d_in < 0)
    above = bool(d_out > d_in)
    return {"D_in_negative": negative, "D_out_above_D_in": above,
            "both": negative and above}


def prediction_one_holds(conditions: Mapping[tuple, bool]) -> bool:
    """Prediction 1: `D_in < 0 and D_out > D_in` in each of the eight named
    cells. `conditions` maps (trace, fraction, multiplier) to that cell's
    joint condition; a named cell that is absent fails the prediction."""
    found = {(str(cell[0]), float(cell[1]), float(cell[2])): bool(value)
             for cell, value in conditions.items()}
    return all(found.get(cell, False) for cell in PREDICTION_ONE_CELLS)


def prediction_two_holds(consistent_losses: int) -> bool:
    """Prediction 2: `D_rand` a consistent loss in at least 8 of the 12 trace
    x cell (exactly 8 suffices)."""
    return int(consistent_losses) >= PREDICTION_TWO_MINIMUM


def seed_interval(values: Iterable[float]) -> tuple[float, float, float]:
    """(mean, low, high) of the 95% t seed interval of seed-paired values,
    `mean -/+ ci95_half` with `gap.summarize`'s half-width (the `_ci95_half`
    the tables carry); nan bounds with fewer than two finite values."""
    stats = summarize(list(values))
    return stats["mean"], stats["mean"] - stats["ci95_half"], stats["mean"] + stats["ci95_half"]


def additive(d_in: float, d_out: float, low: float, high: float) -> bool:
    """Reading 1, descriptive: `D_in + D_out` (five-seed means) within the seed
    interval `[low, high]` of `D_learned`, bounds included; False when any
    value is not finite."""
    total = d_in + d_out
    if not all(math.isfinite(value) for value in (total, low, high)):
        return False
    return low <= total <= high

