"""Error-location control — which decisions and which errors separate a ranker from its label.

`docs/error-location-plan.md` is the pre-registration. The mechanism control
held the score fixed at four rungs and found the exact training label far above
the frozen `next_use` ranker through the published sampled mechanism (`all16`:
the arrival plus up to 16 uniformly sampled residents, any of which may leave).
This phase keeps that mechanism and places arms between the two:

* the reference rungs and the real rankers the ladder left out (`pi3`, the
  `binary` target, and the exact `binary` label);
* hybrids that take the arrival's admission from one score and every eviction
  from the other (`HybridScorer` + `HybridOverride`);
* the label with a controlled Gaussian error (`NoisyScorer`);
* the label rung with its victim swapped at a controlled rate, uniformly or to
  the runner-up (`SwapOverride`).

Every arm is scored at the decision instant; every constructed arm reads the
trace's future on purpose, and none is a proposed policy. Each decision an arm
makes is compared with the reference key `K* = (label, last_group)`, the label
rung's own key, by `DecisionStatistics`, which `RecordingOverride` feeds with
the *final* victim of the decision (after any override). Noise and swap draws
come from a keyed hash, never from the store's sampling RNG or Python's global
one, so they cannot move a candidate draw.

Nothing here fits, adds a feature or target, or changes an existing replay
path: the arms ride on `run_two_tier`'s existing scorer and override slots, and
the reading helpers at the end are pure arithmetic on numbers the replays
produced, kept here so that they can be tested. The seed-sign readings
("consistent gain / loss / mixed") are `mechanism.sign_counts` and
`mechanism.sign_reading`, used as they are.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .decisionpop import RawFeatureScorer, spearman
from .mechanism import RUNGS, ExactLabelScorer
from .trace import Request, Trace

# --- the arms (docs/error-location-plan.md, "Fixed surface") --------------------------

# The label of the reference key and of the label rung; `binary` is the other
# on-policy target, "1 if the state is requested within the horizon".
LABEL_TARGET = "next_use"
BINARY_TARGET = "binary"
REFERENCE_ARMS = RUNGS                    # ("lru", "learned", "label", "offline")
# Real rankers the ladder did not include: arm -> (published policy, target).
REAL_RANKERS = {
    "pi3_next_use": ("pi3", "next_use"),
    "pi0_binary": ("pi0", "binary"),
    "pi3_binary": ("pi3", "binary"),
}
REAL_RANKER_ARMS = tuple(REAL_RANKERS)
LABEL_BINARY_ARM = "label_binary"
# "Admission by X, eviction by Y": arm -> (X, Y), each a score component below.
# The two identity hybrids (X = Y) are smoke and test controls, not grid arms.
HYBRIDS = {
    "adm_label": ("label", "learned"),
    "evict_label": ("learned", "label"),
    "hybrid_learned_learned": ("learned", "learned"),
    "hybrid_label_label": ("label", "label"),
}
HYBRID_ARMS = ("adm_label", "evict_label")
HYBRID_COMPONENTS = ("learned", "label")
NOISE_LEVELS = (0.5, 1.0, 2.0, 4.0)
SWAP_KINDS = ("uniform", "runnerup")
SWAP_PROBABILITIES = (0.25, 0.5)


def noise_arm(scale: float) -> str:
    return f"noise_{scale:g}"


def swap_arm(kind: str, probability: float) -> str:
    return f"swap_{kind}_{probability:g}"


NOISE_ARMS = tuple(noise_arm(scale) for scale in NOISE_LEVELS)
SWAP_ARMS = tuple(swap_arm(kind, p) for kind in SWAP_KINDS for p in SWAP_PROBABILITIES)
# The 18 arms, in table and figure order.
ARMS = (REFERENCE_ARMS + REAL_RANKER_ARMS + (LABEL_BINARY_ARM,) + HYBRID_ARMS
        + NOISE_ARMS + SWAP_ARMS)
# Reading 4's primary set: the twelve arms whose victim is the first minimum of
# a single key (no override); the secondary set is all 18.
PRIMARY_ARMS = REFERENCE_ARMS + REAL_RANKER_ARMS + (LABEL_BINARY_ARM,) + NOISE_ARMS
METRIC_SETS = {"primary": PRIMARY_ARMS, "secondary": ARMS}
# Identities of the plan's required checks: an arm built at the null setting of
# its construction must be its reference arm decision by decision.
IDENTITY_ARMS = {
    "hybrid_learned_learned": "learned",
    "hybrid_label_label": "label",
    noise_arm(0.0): "label",
    swap_arm("uniform", 0.0): "label",
    swap_arm("runnerup", 0.0): "label",
}
assert len(ARMS) == 18 and len(set(ARMS)) == 18 and len(PRIMARY_ARMS) == 12


@dataclass
class ArmSetup:
    """How one arm is replayed: the `run_two_tier` policy name, scorer and
    sampled-offline flag, and the arm's own override (None for the arms whose
    victim is the store's first minimum)."""

    arm: str
    l2_policy: str
    scorer: object | None = None
    override: object | None = None
    sampled_offline: bool = False

    def replay_arguments(self) -> dict[str, object]:
        """`run_two_tier` keyword arguments of the score (the override is passed
        by the caller, wrapped or not in a `RecordingOverride`)."""
        arguments: dict[str, object] = {"l2_policy": self.l2_policy}
        if self.scorer is not None:
            arguments["l2_scorer"] = self.scorer
        if self.sampled_offline:
            arguments["l2_sampled_offline"] = True
        return arguments


def _parse_parameter(arm: str, prefix: str) -> float:
    try:
        return float(arm[len(prefix):])
    except ValueError:
        raise ValueError(f"unknown arm {arm!r}") from None


def arm_setup(arm: str, trace: Trace, horizon_seconds: float, seed: int,
              learned_ranker=None, real_ranker=None) -> ArmSetup:
    """The replay of `arm` on `trace` at sampling seed `seed`.

    `learned_ranker` is the frozen pi0 `next_use` ranker of the trace (needed
    by `learned` and every hybrid containing it); `real_ranker` is the loaded
    model of a real-ranker arm. A fresh scorer is built per call, because the
    history scorers carry state across a replay.
    """
    def component(name: str):
        if name == "learned":
            if learned_ranker is None:
                raise ValueError(f"{arm} needs the learned ranker")
            return RawFeatureScorer(trace, learned_ranker)
        if name == "label":
            return ExactLabelScorer(trace, horizon_seconds, target=LABEL_TARGET)
        raise ValueError(f"unknown score component {name!r}")

    if arm == "lru":
        return ArmSetup(arm, "lru")
    if arm == "offline":
        return ArmSetup(arm, "offline_next_use", sampled_offline=True)
    if arm in ("learned", "label"):
        return ArmSetup(arm, "learned", scorer=component(arm))
    if arm in REAL_RANKERS:
        if real_ranker is None:
            raise ValueError(f"{arm} needs its published model")
        return ArmSetup(arm, "learned", scorer=RawFeatureScorer(trace, real_ranker))
    if arm == LABEL_BINARY_ARM:
        return ArmSetup(arm, "learned",
                        scorer=ExactLabelScorer(trace, horizon_seconds, target=BINARY_TARGET))
    if arm in HYBRIDS:
        admission, eviction = HYBRIDS[arm]
        # X = Y is one scorer object in both roles: the composite then observes
        # it once, and its admission keys are the store's own keys exactly.
        x = component(admission)
        y = x if eviction == admission else component(eviction)
        scorer = HybridScorer(x, y)
        return ArmSetup(arm, "learned", scorer=scorer, override=HybridOverride(scorer.admission))
    if arm.startswith("noise_"):
        scale = _parse_parameter(arm, "noise_")
        return ArmSetup(arm, "learned", scorer=NoisyScorer(component("label"), scale, seed))
    for kind in SWAP_KINDS:
        prefix = f"swap_{kind}_"
        if arm.startswith(prefix):
            probability = _parse_parameter(arm, prefix)
            return ArmSetup(arm, "learned", scorer=component("label"),
                            override=SwapOverride(kind, probability, seed))
    raise ValueError(f"unknown arm {arm!r}")


# --- keyed randomness ------------------------------------------------------------------

_STANDARD_NORMAL = NormalDist()


def hash_bits(tag: str, seed: int, *parts: str) -> int:
    """64 bits of BLAKE2b over (tag, seed, parts): a pure function of its
    arguments, the same in every process and on every run, which reads and
    advances no generator."""
    text = "\x1f".join((tag, str(int(seed))) + tuple(parts))
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _open_uniform(bits: int) -> float:
    """53 bits as a uniform on the open interval (0, 1)."""
    return ((bits >> 11) + 0.5) / float(1 << 53)


def noise_z(seed: int, state_id: str, timestamp_ms: float) -> float:
    """The standard normal draw of `state_id` at the decision instant
    `timestamp_ms`: fresh at every timestamp, identical when the same state is
    scored again at the same timestamp. The same for every noise level, as the
    plan makes it a function of (seed, state, timestamp) only."""
    bits = hash_bits("noise", seed, state_id, repr(float(timestamp_ms)))
    return _STANDARD_NORMAL.inv_cdf(_open_uniform(bits))


def swap_uniform(seed: int, decision_index: int) -> float:
    """The swap coin of one decision, a uniform on [0, 1); the decision is
    swapped when it falls below p, so the swapped decisions of p = 0.25 are a
    subset of those of p = 0.5 at the same decision index."""
    return (hash_bits("swap-coin", seed, str(int(decision_index))) >> 11) / float(1 << 53)


def swap_pick(seed: int, decision_index: int, count: int) -> int:
    """A uniform index in [0, count), by exact integer scaling of 64 bits."""
    if count <= 0:
        raise ValueError("nothing to pick from")
    return (hash_bits("swap-pick", seed, str(int(decision_index))) * count) >> 64


# --- scorers ------------------------------------------------------------------------------


class NoisyScorer:
    """The label with an independent Gaussian error: `label + s * z`.

    `z = noise_z(seed, state, timestamp)`. The store's key is then
    `(label + s * z, last_group)`. History is forwarded to the base scorer,
    which for the label ignores it.
    """

    time_varying = True

    def __init__(self, base, scale: float, seed: int) -> None:
        if not (math.isfinite(scale) and scale >= 0.0):
            raise ValueError(f"noise scale must be finite and non-negative, got {scale!r}")
        self.base = base
        self.scale = float(scale)
        self.seed = int(seed)

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        self.base.observe(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> float:
        return (self.base.score(state_id, timestamp_ms)
                + self.scale * noise_z(self.seed, state_id, timestamp_ms))


class HybridScorer:
    """The store's scorer of "admission by X, eviction by Y".

    `score` is Y's, so the store's own keys, its first minimum and every later
    round are Y's. X is consulted only by `HybridOverride`, at the same instant
    the store scores. Both are shown every observation (each distinct scorer
    once), so a history scorer in either role sees exactly the history it
    would see as the store's own scorer.
    """

    time_varying = True

    def __init__(self, admission, eviction) -> None:
        self.admission = admission
        self.eviction = eviction
        self.observed = [admission] if admission is eviction else [admission, eviction]

    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        for scorer in self.observed:
            scorer.observe(requests, timestamp_ms)

    def score(self, state_id: str, timestamp_ms: float) -> float:
        return self.eviction.score(state_id, timestamp_ms)


def first_minimum(keys: Sequence[tuple]) -> int:
    """The index the store removes: the first minimum in draw order."""
    return min(range(len(keys)), key=keys.__getitem__)


def reference_order(keys: Sequence[tuple]) -> list[int]:
    """Candidate indices by (key, draw index): position 0 is the first minimum,
    position 1 the runner-up in draw order."""
    return sorted(range(len(keys)), key=lambda index: (keys[index], index))


class HybridOverride:
    """`l2_override_hook` of "admission by X, eviction by Y".

    In a first-round decision in which the arrival is a candidate
    (`arriving_index >= 0`, an *admission* decision) the arrival is rejected
    if and only if it is the first minimum of X's keys `(X score, last_group)`
    over the candidates; otherwise the victim is the first minimum of Y's keys
    (the store's) over the candidates other than the arrival. Every later round
    keeps the store's choice, which is Y's. With X = Y this is the published
    rule: the arrival sits first, so it is the first minimum of Y over all
    candidates exactly when Y rejects it, and otherwise Y's first minimum over
    all candidates is its first minimum over the others.

    `last_group` is read from the second element of the store's key tuple
    `(Y score, last_group)`, the same value the store's key carries.
    """

    def __init__(self, admission_scorer) -> None:
        self.admission = admission_scorer

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index,
                 arriving_index):
        if arriving_index < 0:
            return None
        admission_keys = []
        for index, state_id in enumerate(candidates):
            if len(keys[index]) != 2:
                raise ValueError("a hybrid needs scored keys (score, last_group)")
            admission_keys.append((self.admission.score(state_id, timestamp_ms), keys[index][1]))
        if first_minimum(admission_keys) == arriving_index:
            return arriving_index
        return min((index for index in range(len(candidates)) if index != arriving_index),
                   key=keys.__getitem__)


class SwapOverride:
    """`l2_override_hook` of the victim swaps on the label rung.

    At every decision with at least two candidates, with probability `p`
    (`swap_uniform(seed, decision index) < p`), the label rung's victim is
    replaced: by a uniformly drawn other candidate (`"uniform"`), or by the
    candidate ranked second by `K*` in draw order (`"runnerup"`). On the label
    rung the store's key *is* `K* = (label, last_group)`, so the runner-up is
    read off the keys the hook receives. The decision index is the store's
    decision counter, counted here by calls (one per decision, in order) and
    cross-checked against the store once it is attached.
    """

    def __init__(self, kind: str, probability: float, seed: int) -> None:
        if kind not in SWAP_KINDS:
            raise ValueError(f"unknown swap kind {kind!r}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"swap probability must lie in [0, 1], got {probability!r}")
        self.kind = kind
        self.probability = float(probability)
        self.seed = int(seed)
        self.decisions = 0
        self.swaps = 0
        self.store = None

    def attach(self, l1, l2, counters) -> None:
        self.store = l2

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index,
                 arriving_index):
        index = self.decisions
        self.decisions += 1
        if self.store is not None and self.store.decisions - 1 != index:
            raise AssertionError(f"swap decision index {index} != store's {self.store.decisions - 1}")
        count = len(candidates)
        if count < 2 or swap_uniform(self.seed, index) >= self.probability:
            return None
        self.swaps += 1
        if self.kind == "uniform":
            pick = swap_pick(self.seed, index, count - 1)
            return pick if pick < victim_index else pick + 1
        return reference_order(keys)[1]


# --- the decision statistics ---------------------------------------------------------------

STATISTICS = ("m1", "m2", "m3", "m4")
# Oriented so that larger is better (reading 4): m1, m2, -m3, -m4.
ORIENTATION = {"m1": 1.0, "m2": 1.0, "m3": -1.0, "m4": -1.0}
# Decision types of m2-m4: "" is every decision of the window.
DECISION_TYPES = ("", "admission", "resident")


def _suffix(kind: str) -> str:
    return f"_{kind}" if kind else ""


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


class DecisionStatistics:
    """m1-m4 of one arm's own decisions against the reference key `K*`.

    `K*` of a candidate is the label rung's key, `(label, last_group)`: the
    label from the label rung's own scorer (`ExactLabelScorer`, target
    `next_use`) and `last_group` from the store's mapping, read-only (`attach`
    receives the store; a candidate's entry always exists, since the store
    reads it to build the candidate's own key). The reference victim is the
    first minimum of `K*` in draw order. Over the decisions whose timestamp is
    at or after `measure_from_ms`:

    * m1: over candidate pairs with different `K*`, the share the arm's key
      orders the same way, a tie in the arm's key counting one half (pooled
      over the pairs of all decisions);
    * m2: the share of decisions whose victim is the reference victim;
    * m3: the mean of `label(victim) - min label` over the decisions;
    * m4: the share of decisions whose victim is requested within the horizon
      (the `binary` target, next use <= H) while some candidate is not.

    m2-m4 are kept for every decision, for admission decisions (the arrival is
    a candidate) and for resident-only ones. Everything is a sum or a count so
    that seeds aggregate exactly; `row` adds the ratios. Every decision of the
    replay, warm-up included, also enters a SHA-256 of (timestamp, arrival
    index, final victim, candidates) and the rejection and eviction counts,
    which the runner checks against the replay's own counters.

    `record` is given both the store's original victim and the final one; the
    statistics are of the final victim. It only reads its arguments.
    """

    def __init__(self, trace: Trace, horizon_seconds: float,
                 measure_from_ms: float | None) -> None:
        self.reference = ExactLabelScorer(trace, horizon_seconds, target=LABEL_TARGET)
        # Exactly `target_column(..., "binary", H)`: delta <= H * 1000.
        self.horizon_ms = float(horizon_seconds) * 1000.0
        self.measure_from_ms = measure_from_ms
        self.last_group = None
        self.digest = hashlib.sha256()
        self.decisions_seen = 0
        self.rejections_seen = 0
        self.overridden_seen = 0
        self.pairs = 0
        self.concordant = 0
        self.tied = 0
        self.counts = {kind: {"decisions": 0, "agree": 0, "excess": 0.0, "avoidable": 0,
                              "at_horizon": 0, "overridden": 0, "single": 0}
                       for kind in DECISION_TYPES}

    def attach(self, store) -> None:
        self.last_group = store.last_group

    def _last_group(self, state_id: str) -> float:
        value = self.last_group.get(state_id) if self.last_group is not None else None
        if value is None:
            raise RuntimeError(f"no last_group for candidate {state_id}: attach the store first")
        return float(value)

    def record(self, candidates, keys, original_index: int, victim_index: int,
               timestamp_ms: float, group_index: int, arriving_index: int) -> None:
        self.decisions_seen += 1
        if arriving_index >= 0 and victim_index == arriving_index:
            self.rejections_seen += 1
        overridden = victim_index != original_index
        self.overridden_seen += overridden
        self.digest.update(
            ("|".join([repr(float(timestamp_ms)), str(arriving_index), str(victim_index)])
             + "|" + "\x1e".join(candidates) + "\n").encode("utf-8"))
        if self.measure_from_ms is not None and timestamp_ms < self.measure_from_ms:
            return
        count = len(candidates)
        labels = [self.reference.score(state_id, timestamp_ms) for state_id in candidates]
        deltas = [self.reference.label(state_id, timestamp_ms)[0] for state_id in candidates]
        kstar = [(labels[index], self._last_group(state_id))
                 for index, state_id in enumerate(candidates)]
        # m1, pooled over the pairs of this decision whose K* differ.
        for i in range(count):
            key_i, arm_i = kstar[i], keys[i]
            for j in range(i + 1, count):
                key_j = kstar[j]
                if key_i == key_j:
                    continue
                arm_j = keys[j]
                self.pairs += 1
                if arm_i == arm_j:
                    self.tied += 1
                elif (key_i < key_j) == (arm_i < arm_j):
                    self.concordant += 1
        reusable = [delta <= self.horizon_ms for delta in deltas]
        agree = victim_index == first_minimum(kstar)
        excess = labels[victim_index] - min(labels)
        avoidable = reusable[victim_index] and not all(reusable)
        at_horizon = avoidable and deltas[victim_index] == self.horizon_ms
        for kind in ("", "admission" if arriving_index >= 0 else "resident"):
            counts = self.counts[kind]
            counts["decisions"] += 1
            counts["agree"] += agree
            counts["excess"] += excess
            counts["avoidable"] += avoidable
            counts["at_horizon"] += at_horizon
            counts["overridden"] += overridden
            counts["single"] += count == 1

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index,
                 arriving_index) -> None:
        """As a plain `l2_decision_hook` (no override: the victim is final)."""
        self.record(candidates, keys, victim_index, victim_index, timestamp_ms, group_index,
                    arriving_index)

    def row(self) -> dict[str, object]:
        out: dict[str, object] = {
            "stat_decisions_seen": self.decisions_seen,
            "stat_rejections_seen": self.rejections_seen,
            "stat_evictions_seen": self.decisions_seen - self.rejections_seen,
            "overridden_decisions_seen": self.overridden_seen,
            "decision_sha256": self.digest.hexdigest(),
            "m1_pairs": self.pairs,
            "m1_concordant": self.concordant,
            "m1_tied": self.tied,
            "m1": _ratio(self.concordant + 0.5 * self.tied, self.pairs),
        }
        for kind in DECISION_TYPES:
            counts = self.counts[kind]
            suffix = _suffix(kind)
            decisions = counts["decisions"]
            out[f"stat_decisions{suffix}"] = decisions
            out[f"single_candidate_decisions{suffix}"] = counts["single"]
            out[f"overridden_decisions{suffix}"] = counts["overridden"]
            out[f"m2_agree{suffix}"] = counts["agree"]
            out[f"m2{suffix}"] = _ratio(counts["agree"], decisions)
            out[f"m3_excess_sum{suffix}"] = counts["excess"]
            out[f"m3{suffix}"] = _ratio(counts["excess"], decisions)
            out[f"m4_count{suffix}"] = counts["avoidable"]
            out[f"m4{suffix}"] = _ratio(counts["avoidable"], decisions)
            out[f"m4_victim_at_horizon{suffix}"] = counts["at_horizon"]
        return out


class RecordingOverride:
    """The `l2_override_hook` every arm of the run carries.

    It calls the arm's own override (if any), hands `DecisionStatistics` the
    decision with its *final* victim — the override's choice, else the store's
    — and returns the arm's own answer unchanged, so with no arm override it
    returns None and the decision is the store's. Its `attach(l1, l2,
    counters)`, which `run_two_tier` calls with the live stores before the
    replay, gives the statistics the store's `last_group` mapping and forwards
    to the arm's override. It writes nothing the replay reads.
    """

    def __init__(self, statistics: DecisionStatistics, override=None) -> None:
        self.statistics = statistics
        self.override = override

    def attach(self, l1, l2, counters) -> None:
        self.statistics.attach(l2)
        if self.override is not None and hasattr(self.override, "attach"):
            self.override.attach(l1, l2, counters)

    def __call__(self, candidates, keys, victim_index, timestamp_ms, group_index,
                 arriving_index):
        choice = None
        if self.override is not None:
            choice = self.override(candidates, keys, victim_index, timestamp_ms, group_index,
                                   arriving_index)
        final = victim_index if choice is None else choice
        self.statistics.record(candidates, keys, victim_index, final, timestamp_ms, group_index,
                               arriving_index)
        return choice


# --- the readings (docs/error-location-plan.md, "Readings, fixed before the run") ------------

LOCATION_SHARE = 0.5
LOCATIONS = ("admission_located", "eviction_located", "both", "neither")
ORDERS_THRESHOLD = 0.9


def location_label(g: float, a: float, e: float) -> str:
    """Reading 1 on five-seed means: `G = U(label) - U(learned)`, `A` and `E`
    the hybrids' gains over `learned`. A term "reaches half" when it is at
    least `0.5 * G`; the rule is applied as written for any sign of G (a nan
    reaches nothing)."""
    half = LOCATION_SHARE * g
    admission = a >= half
    eviction = e >= half
    if admission and eviction:
        return "both"
    if admission:
        return "admission_located"
    if eviction:
        return "eviction_located"
    return "neither"


def non_increasing(values: Sequence[float]) -> bool:
    """Every value at most the one before it (reading 2, in s)."""
    return all(later <= earlier for earlier, later in zip(values, values[1:]))


def crossing_bracket(levels: Sequence[float], values: Sequence[float], reference: float) -> str:
    """Reading 2: between which adjacent levels the values cross `reference`.

    A level is on the upper side when its value is at or above the reference.
    "above_all" / "below_all" when no adjacent pair changes side; otherwise
    every adjacent pair that does, as "s1-s2", joined by ";" (one for a
    monotone sequence)."""
    if len(levels) != len(values):
        raise ValueError("levels and values differ in length")
    upper = [value >= reference for value in values]
    if all(upper):
        return "above_all"
    if not any(upper):
        return "below_all"
    return ";".join(f"{levels[index]:g}-{levels[index + 1]:g}"
                    for index in range(len(values) - 1) if upper[index] != upper[index + 1])


def oriented(statistic: str, value: float) -> float:
    """The statistic with larger-is-better orientation (m1, m2, -m3, -m4)."""
    return ORIENTATION[statistic] * value


def oriented_name(statistic: str) -> str:
    return statistic if ORIENTATION[statistic] > 0 else f"-{statistic}"


def rank_correlation(statistic_values: Sequence[float], utilities: Sequence[float]) -> float:
    """Spearman over arms, average ranks for ties, nan when either side is
    constant (or has fewer than two arms) — `decisionpop.spearman` itself."""
    return spearman(np.asarray(statistic_values, dtype=float), np.asarray(utilities, dtype=float))


def orders_utility(rho: float) -> bool:
    """Reading 4: a statistic orders utility in a cell when rho >= 0.9."""
    return math.isfinite(rho) and rho >= ORDERS_THRESHOLD


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def agrees(statistic_change: float, utility_change: float) -> bool:
    """Reading 5: the statistic's mean change has the sign of the mean utility
    change (zero matching zero only; a nan matches nothing)."""
    if not (math.isfinite(statistic_change) and math.isfinite(utility_change)):
        return False
    return _sign(statistic_change) == _sign(utility_change)


def contradicts(utility_reading: str, statistic_change: float) -> bool:
    """Reading 5's list: utility changes consistently while the statistic's
    mean change has the opposite (non-zero) sign."""
    if utility_reading == "consistent_gain":
        return statistic_change < 0
    if utility_reading == "consistent_loss":
        return statistic_change > 0
    return False


def ceiling_ratio(u_pi0_binary: float, u_label_binary: float, u_lru: float) -> float:
    """Reading 6: `(U(pi0_binary) - U(lru)) / (U(label_binary) - U(lru))`, nan
    when the denominator is zero."""
    return _ratio(u_pi0_binary - u_lru, u_label_binary - u_lru)
