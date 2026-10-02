"""Working-set ratio check — stack distances of the L1 offer stream.

`docs/working-set-ratio-plan.md` is the pre-registration. With heap-LRU L1 at
a fixed fraction, every L1 eviction *offers* one state to L2; the offer stream
does not depend on the L2 arm. The plan's reference model is an admit-all
exclusive tier that keeps offers in offer order: an offer stays until it
*returns* (the first later request whose prefix contains its state) or is
pushed out by newer offers. Its stack distance `D` is the largest total size,
at any instant between the offer and its return, of the newer offers that have
not yet returned at that instant, so the offer is still held at its return in
a tier of capacity `C` if and only if `D + size <= C`. The working set `W` is
the token-weighted median of `D + size` over the offers whose return falls in
the evaluation window, and `gamma = W / C_L2` is compared with published sign
outcomes by a rule fixed in the plan.

The timeline, and the tie rule. The simulator serves one timestamp group in
two phases: every request of the group is served against the tiers as they
stood before the group (an L2 block it touches is promoted out of L2), and only
then are the group's blocks inserted into L1, whose evictions are the offers.
So inside one group *every return precedes every offer*, and the offers of a
group keep the order in which L1 evicted them (the order `victim_hook` is
called). An offer made at group `g` can return only at a group `r > g`. The
returns of one group are simultaneous; they only remove size, so their order
among themselves never changes a maximum.

Instants as positions. Only an offer increases the outstanding size, so the
maximum over "any instant" is attained just after some offer. Number the
offers `0 .. n-1` in stream order and let position `p` be the instant just
after offer `p`. Offer `j` is outstanding at `p` exactly when `j <= p < R_j`,
where `R_j` is the number of offers made before the return group of `j` (the
first offer index whose group is at least that return group; `n` if it never
returns or returns after the last offer). Then

    D_i = max over i <= p < R_i of  sum of size_j over i < j <= p < R_j

which is what `stack_distances` computes in O(n log n): offers are swept from
the newest to the oldest, offer `j` range-adds its size on `[j, R_j)` of a
segment tree, and offer `i` takes the range maximum over its own `[i, R_i)`
before adding itself. `stack_distances_bruteforce` and `reference_tier_held`
work on the literal event timeline instead (groups, returns before offers) and
are the references the tests hold the fast path to.

Everything here except `record_offer_stream`, which runs the published L1
replay with an empty L2 and records its victims, is pure arithmetic on plain
values, kept here so that it can be tested on constructed streams.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .crossworkload import HYPERPARAMETERS
from .mechanism import sign_reading
from .trace import Trace

# The quantiles of `D + size` reported; the median is the working set `W`.
QUANTILES = (0.25, 0.5, 0.75)
# The fixed rule's threshold on gamma, and the transition band of reading 2:
# a miss with 0.5 <= gamma <= 2 (both ends included) is "in the transition".
THRESHOLD = 1.0
TRANSITION = (0.5, 2.0)
VERDICTS = ("located", "located outside the transition", "not located")


@dataclass(frozen=True, slots=True)
class Offer:
    """One L1 eviction offered to L2, with its return if there is one.

    `group_index` and `timestamp_ms` are those of the timestamp group whose
    insertion caused the eviction; `return_group` / `return_timestamp_ms` are
    those of the first later group with a request whose prefix contains the
    state, or None when there is no such request before the trace ends.
    """

    state_id: str
    group_index: int
    timestamp_ms: float
    size_bytes: int
    tokens: int
    return_group: int | None = None
    return_timestamp_ms: float | None = None

    @property
    def returns(self) -> bool:
        return self.return_group is not None


# --- the offer stream ---------------------------------------------------------------


class OfferRecorder:
    """`victim_hook` of `run_two_tier`: every L1 eviction in call order."""

    def __init__(self) -> None:
        self.victims: list[tuple[str, float, int]] = []

    def __call__(self, state_id: str, timestamp_ms: float, group_index: int) -> None:
        self.victims.append((state_id, timestamp_ms, group_index))


def group_timestamps(trace: Trace) -> list[float]:
    """The timestamp of every timestamp group, by group index."""
    return [timestamp for timestamp, _ in trace.timestamp_groups()]


def offers_from_victims(
    trace: Trace,
    victims: Iterable[tuple[str, float, int]],
    occurrence_groups: Mapping[str, Sequence[int]],
    timestamps: Sequence[float],
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
) -> list[Offer]:
    """Attach size, tokens and return to each recorded L1 victim.

    The size is the expression `run_two_tier`'s store charges (`state_bytes`:
    the packed block tokens, or the full block under the unpacked model, times
    `bytes_per_token`); the weight is the state's own block tokens. The return
    is the first occurrence group after the offer group (`bisect_right` on the
    state's occurrence groups), the same lookup the offline comparator makes.
    A state cannot be offered again before its earlier offer returned (it is
    out of L1 until a request brings it back), which is checked.
    """
    states = trace.states
    pending_until: dict[str, int | None] = {}
    out: list[Offer] = []
    previous_group = -1
    for state_id, timestamp_ms, group_index in victims:
        if group_index < previous_group:
            raise ValueError("victims are not in group order")
        previous_group = group_index
        if state_id in pending_until:
            until = pending_until[state_id]
            if until is None or group_index < until:
                raise RuntimeError(f"{state_id} offered again before its earlier offer returned")
        if timestamps[group_index] != timestamp_ms:
            raise ValueError(f"victim timestamp {timestamp_ms} is not that of group {group_index}")
        meta = states[state_id]
        tokens = meta.block_tokens
        size = (tokens if size_model == "packed" else trace.block_size) * bytes_per_token
        occurrences = occurrence_groups.get(state_id, ())
        position = bisect.bisect_right(occurrences, group_index)
        if position < len(occurrences):
            return_group = int(occurrences[position])
            return_ms = float(timestamps[return_group])
        else:
            return_group, return_ms = None, None
        pending_until[state_id] = return_group
        out.append(Offer(state_id, int(group_index), float(timestamp_ms), int(size), int(tokens),
                         return_group, return_ms))
    return out


def record_offer_stream(
    trace: Trace,
    l1_policy: str,
    l1_capacity_bytes: int,
    occurrence_groups: Mapping[str, Sequence[int]] | None = None,
    measure_from_ms: float | None = None,
    l2_capacity_bytes: int = 0,
    l2_policy: str = "lru",
    **two_tier_options,
):
    """The L1 offer stream as the two-tier simulator produces it.

    Runs `twotier.run_two_tier` with a recorder as `victim_hook`, which the
    replay calls for every L1 eviction before the victim is offered to L2. The
    default L2 of zero bytes is the L1-only path the Phase 0.97 runner records
    its victim streams with (no store is built, so nothing is offered to one).
    The stream does not depend on L2 — L1 never consults it — and a caller may
    pass any L2 to check that. Returns the offers and the replay result.
    """
    from .replay import _occurrence_groups
    from .twotier import run_two_tier

    groups = occurrence_groups if occurrence_groups is not None else _occurrence_groups(trace)
    recorder = OfferRecorder()
    result = run_two_tier(trace, l1_policy, l1_capacity_bytes, l2_capacity_bytes, l2_policy,
                          occurrence_groups=groups, measure_from_ms=measure_from_ms,
                          victim_hook=recorder, **two_tier_options)
    if len(recorder.victims) != result.l1_evictions:
        raise RuntimeError("the recorder missed an L1 eviction")
    bytes_per_token = two_tier_options.get("bytes_per_token", HYPERPARAMETERS["bytes_per_token"])
    size_model = two_tier_options.get("size_model", HYPERPARAMETERS["size_model"])
    offers = offers_from_victims(trace, recorder.victims, groups, group_timestamps(trace),
                                 bytes_per_token, size_model)
    return offers, result


# --- stack distance -------------------------------------------------------------------


def return_positions(offers: Sequence[Offer]) -> list[int]:
    """`R_j` for every offer: the number of offers made before its return.

    Returns precede the offers of their own group, so `R_j` is the first offer
    index whose group is at least the return group, `n` when the offer never
    returns or returns after the last offer.
    """
    groups = [offer.group_index for offer in offers]
    if any(later < earlier for earlier, later in zip(groups, groups[1:])):
        raise ValueError("offers are not in group order")
    count = len(offers)
    out = []
    for index, offer in enumerate(offers):
        if offer.return_group is None:
            out.append(count)
            continue
        if offer.return_group <= offer.group_index:
            raise ValueError(f"offer {index} returns at group {offer.return_group}, "
                             f"not after its offer group {offer.group_index}")
        out.append(bisect.bisect_left(groups, offer.return_group))
    return out


class RangeAddMax:
    """Range add and range maximum over positions `[0, n)`, all starting at 0.

    The bottom-up segment tree with lazy additions: `tree[p]` is the maximum of
    node `p`'s subtree including every addition stored at `p` and below,
    `pending[p]` the addition stored at internal node `p` and not yet pushed to
    its children. Both operations take half-open ranges and O(log n) steps.
    """

    def __init__(self, size: int) -> None:
        self.height = max(1, (max(size, 1) - 1).bit_length())
        self.leaves = 1 << self.height
        self.tree = [0] * (2 * self.leaves)
        self.pending = [0] * self.leaves

    def _rebuild(self, node: int) -> None:
        tree, pending = self.tree, self.pending
        while node > 1:
            node >>= 1
            left, right = tree[2 * node], tree[2 * node + 1]
            tree[node] = (left if left > right else right) + pending[node]

    def _push(self, node: int) -> None:
        tree, pending, leaves = self.tree, self.pending, self.leaves
        for shift in range(self.height, 0, -1):
            parent = node >> shift
            value = pending[parent]
            if value:
                for child in (2 * parent, 2 * parent + 1):
                    tree[child] += value
                    if child < leaves:
                        pending[child] += value
                pending[parent] = 0

    def add(self, start: int, end: int, value: int) -> None:
        if start >= end:
            return
        tree, pending, leaves = self.tree, self.pending, self.leaves
        low, high = start + leaves, end + leaves
        first, last = low, high - 1
        while low < high:
            if low & 1:
                tree[low] += value
                if low < leaves:
                    pending[low] += value
                low += 1
            if high & 1:
                high -= 1
                tree[high] += value
                if high < leaves:
                    pending[high] += value
            low >>= 1
            high >>= 1
        self._rebuild(first)
        self._rebuild(last)

    def max(self, start: int, end: int) -> int:
        if start >= end:
            raise ValueError("empty range")
        tree, leaves = self.tree, self.leaves
        low, high = start + leaves, end + leaves
        self._push(low)
        self._push(high - 1)
        best = -math.inf
        while low < high:
            if low & 1:
                if tree[low] > best:
                    best = tree[low]
                low += 1
            if high & 1:
                high -= 1
                if tree[high] > best:
                    best = tree[high]
            low >>= 1
            high >>= 1
        return int(best)


def stack_distances(offers: Sequence[Offer]) -> list[int | None]:
    """`D` of every returning offer (None for an offer that never returns).

    The sweep runs from the newest offer to the oldest: when offer `i` is
    queried, exactly the newer offers have range-added their sizes over their
    own outstanding positions, so the maximum over `[i, R_i)` is the largest
    total of newer offers outstanding at any instant before `i` returns. At
    position `i` itself no newer offer exists yet, so `D >= 0`.
    """
    count = len(offers)
    positions = return_positions(offers)
    tree = RangeAddMax(count)
    out: list[int | None] = [None] * count
    for index in range(count - 1, -1, -1):
        offer = offers[index]
        end = positions[index]
        if offer.return_group is not None:
            out[index] = tree.max(index, end)
        tree.add(index, end, offer.size_bytes)
    return out


def _timeline(offers: Sequence[Offer]) -> list[tuple[int, int, int]]:
    """Events `(group, phase, offer index)` in simulator order: within a group,
    phase 0 (returns) before phase 1 (offers); offers in stream order."""
    events = []
    for index, offer in enumerate(offers):
        events.append((offer.group_index, 1, index))
        if offer.return_group is not None:
            if offer.return_group <= offer.group_index:
                raise ValueError(f"offer {index} does not return after its offer group")
            events.append((offer.return_group, 0, index))
    events.sort()
    return events


def stack_distances_bruteforce(offers: Sequence[Offer]) -> list[int | None]:
    """The definition read literally on the event timeline, in O(n^2).

    For each returning offer, walk every event from its own offer to its own
    return, keeping the set of newer offers made and not yet returned, and take
    the largest total size that set reaches. A reference for tests only.
    """
    events = _timeline(offers)
    out: list[int | None] = [None] * len(offers)
    start_of = {}
    for position, (_, phase, index) in enumerate(events):
        if phase == 1:
            start_of[index] = position
    for index, offer in enumerate(offers):
        if offer.return_group is None:
            continue
        outstanding: dict[int, int] = {}
        best = 0
        for _, phase, other in events[start_of[index]:]:
            if phase == 0 and other == index:
                break
            if other <= index:
                continue
            if phase == 1:
                outstanding[other] = offers[other].size_bytes
            else:
                outstanding.pop(other, None)
            total = sum(outstanding.values())
            if total > best:
                best = total
        out[index] = best
    return out


def reference_tier_held(offers: Sequence[Offer], capacity_bytes: int) -> list[bool | None]:
    """Brute-force reference tier: is each returning offer still held at its return?

    Admit-all, exclusive, offer-ordered tier of `capacity_bytes`: every offer
    enters; while the held total exceeds the capacity the oldest held offer is
    pushed out; a returning offer leaves. Events follow the simulator's order
    (`_timeline`). None for an offer that never returns. The queue drops a
    returned offer lazily, so the whole stream costs O(n log n) for the event
    sort and O(n) after it, which lets the runner check every real offer too.
    """
    queue: deque[int] = deque()          # offer order, oldest first; may hold left offers
    present = [False] * len(offers)
    total = 0
    out: list[bool | None] = [None] * len(offers)
    for _, phase, index in _timeline(offers):
        if phase == 0:
            out[index] = present[index]
            if present[index]:
                present[index] = False
                total -= offers[index].size_bytes
            continue
        queue.append(index)
        present[index] = True
        total += offers[index].size_bytes
        while total > capacity_bytes:
            oldest = queue.popleft()
            if present[oldest]:
                present[oldest] = False
                total -= offers[oldest].size_bytes
    return out


# --- the working set and the ratio ------------------------------------------------


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Lower weighted quantile: the smallest value whose cumulative weight
    reaches `q` of the total (the inverse of the weighted distribution
    function). Always an observed value; nan when there is no positive weight.
    The comparison is exact for integer weights and a binary-exact `q`."""
    if len(values) != len(weights):
        raise ValueError("values and weights differ in length")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile {q} outside [0, 1]")
    if any(weight < 0 for weight in weights):
        raise ValueError("negative weight")
    pairs = sorted((value, weight) for value, weight in zip(values, weights) if weight > 0)
    if not pairs:
        return math.nan
    total = sum(Fraction(weight) for _, weight in pairs)
    target = Fraction(q) * total
    cumulative = Fraction(0)
    for value, weight in pairs:
        cumulative += Fraction(weight)
        if cumulative >= target:
            return value
    return pairs[-1][0]


def weighted_quantiles(values, weights, quantiles=QUANTILES) -> dict[float, float]:
    return {q: weighted_quantile(values, weights, q) for q in quantiles}


def in_window(timestamp_ms: float | None, measure_from_ms: float | None) -> bool:
    """The simulator's window test: at or after `measure_from_ms`."""
    if timestamp_ms is None:
        return False
    return measure_from_ms is None or timestamp_ms >= measure_from_ms


def working_set(
    offers: Sequence[Offer],
    distances: Sequence[int | None],
    measure_from_ms: float | None,
    quantiles=QUANTILES,
) -> dict[str, object]:
    """Counts and token-weighted quantiles of `D + size` (bytes) over the
    offers whose return falls in the evaluation window. The median is `W`."""
    if len(offers) != len(distances):
        raise ValueError("offers and distances differ in length")
    values, weights = [], []
    returning = 0
    for offer, distance in zip(offers, distances):
        if offer.return_group is None:
            continue
        returning += 1
        if distance is None:
            raise ValueError("a returning offer has no stack distance")
        if in_window(offer.return_timestamp_ms, measure_from_ms):
            values.append(distance + offer.size_bytes)
            weights.append(offer.tokens)
    return {
        "offers": len(offers),
        "returning_offers": returning,
        "window_returning_offers": len(values),
        "window_returning_tokens": sum(weights),
        "quantiles": weighted_quantiles(values, weights, quantiles),
    }


def ratio(working_set_bytes: float, capacity_bytes: int) -> float:
    """`gamma = W / C_L2`. With both below 2**53 the float quotient compares
    exactly with 0.5, 1 and 2."""
    if capacity_bytes <= 0:
        raise ValueError("capacity must be positive")
    return float(working_set_bytes) / float(capacity_bytes)


# --- predicted and observed signs, agreement, verdict -------------------------------


def predicted_sign(gamma: float, declining_sign: int, threshold: float = THRESHOLD) -> int:
    """The source's rule. `declining_sign` is the sign the outcome takes when
    declining writes helps (+1 for 2-hit minus LRU, -1 for protection minus
    none). Where `gamma > threshold` that sign is predicted, else the opposite;
    `gamma == threshold` falls on the "<=" side."""
    if declining_sign not in (1, -1):
        raise ValueError("declining_sign must be +1 or -1")
    return declining_sign if gamma > threshold else -declining_sign


def observation(values: Sequence[float]) -> tuple[int, str]:
    """Observed sign and reading of one trace x cell.

    One value (a deterministic outcome) or seed-paired values read by
    `mechanism.sign_reading`: +1 for "consistent_gain", -1 for
    "consistent_loss", 0 otherwise. A zero or mixed observation has sign 0,
    which never equals a prediction, so it is a miss; "zero" names the case
    where every value is exactly 0.
    """
    values = list(values)
    reading = sign_reading(values)
    if reading == "mixed" and values and all(value == 0 for value in values):
        reading = "zero"
    sign = {"consistent_gain": 1, "consistent_loss": -1}.get(reading, 0)
    return sign, reading


def agreement(rows: Sequence[Mapping[str, object]], threshold: float = THRESHOLD) -> dict[str, object]:
    """Matches of the observed sign with the predicted sign over `rows`.

    Each row needs `gamma`, `observed_sign` and `declining_sign`. Returns the
    match count, the number of rows compared, and the indices of the misses.
    """
    misses = []
    for index, row in enumerate(rows):
        predicted = predicted_sign(float(row["gamma"]), int(row["declining_sign"]), threshold)
        if int(row["observed_sign"]) != predicted:
            misses.append(index)
    return {"agreement": len(rows) - len(misses), "compared": len(rows), "misses": misses}


def in_transition(gamma: float, band: tuple[float, float] = TRANSITION) -> bool:
    return band[0] <= gamma <= band[1]


def verdict(miss_gammas: Sequence[float], band: tuple[float, float] = TRANSITION) -> str:
    """Reading 2 on the compared cells: "located" with no miss, "located outside
    the transition" when every miss has `0.5 <= gamma <= 2`, else "not located".
    Whether every expected cell was compared is the caller's question."""
    if not miss_gammas:
        return VERDICTS[0]
    if all(in_transition(gamma, band) for gamma in miss_gammas):
        return VERDICTS[1]
    return VERDICTS[2]


def threshold_intervals(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Reading 3: the thresholds on gamma that maximise agreement, and that maximum.

    Agreement is constant between consecutive distinct gammas. With them sorted
    as `g_1 < ... < g_k`, the candidate intervals are `[0, g_1)`,
    `[g_m, g_m+1)` and `[g_k, inf)`: a threshold `t` in `[g_m, g_m+1)` puts
    every cell with `gamma <= g_m` on the "<=" side, as `predicted_sign` does.
    Adjacent maximising intervals are merged; each run is reported once as a
    half-open `(low, high)`.
    """
    if not rows:
        return {"max_agreement": 0, "compared": 0, "intervals": []}
    gammas = sorted({float(row["gamma"]) for row in rows})
    edges = [0.0] + gammas + [math.inf]
    scored = []
    for low, high in zip(edges, edges[1:]):
        scored.append((low, high, agreement(rows, threshold=low)["agreement"]))
    best = max(count for _, _, count in scored)
    intervals: list[tuple[float, float]] = []
    for low, high, count in scored:
        if count != best:
            continue
        if intervals and intervals[-1][1] == low:
            intervals[-1] = (intervals[-1][0], high)
        else:
            intervals.append((low, high))
    return {"max_agreement": best, "compared": len(rows), "intervals": intervals}


def interval_contains(interval: tuple[float, float], value: float) -> bool:
    return interval[0] <= value < interval[1]


# --- the declinable share -------------------------------------------------------------


def declinable(
    offers: Sequence[Offer],
    distances: Sequence[int | None],
    capacity_bytes: int,
    measure_from_ms: float | None,
) -> dict[str, int]:
    """Reading 4: offers an LRU tier of `capacity_bytes` could decline without
    losing a hit — the offer never returns, or `D + size > C`.

    In bytes and in offers, over the offers made in the evaluation window (the
    pre-registered share is `window_declinable_bytes / window_offered_bytes`)
    and over the whole trace.
    """
    if len(offers) != len(distances):
        raise ValueError("offers and distances differ in length")
    out = {key: 0 for key in (
        "window_offers", "window_offered_bytes", "window_declinable_offers",
        "window_declinable_bytes", "window_never_return_bytes", "window_beyond_capacity_bytes",
        "offers", "offered_bytes", "declinable_offers", "declinable_bytes",
    )}
    for offer, distance in zip(offers, distances):
        never = offer.return_group is None
        beyond = (not never) and distance + offer.size_bytes > capacity_bytes
        decline = never or beyond
        out["offers"] += 1
        out["offered_bytes"] += offer.size_bytes
        out["declinable_offers"] += int(decline)
        out["declinable_bytes"] += offer.size_bytes if decline else 0
        if in_window(offer.timestamp_ms, measure_from_ms):
            out["window_offers"] += 1
            out["window_offered_bytes"] += offer.size_bytes
            out["window_declinable_offers"] += int(decline)
            out["window_declinable_bytes"] += offer.size_bytes if decline else 0
            out["window_never_return_bytes"] += offer.size_bytes if never else 0
            out["window_beyond_capacity_bytes"] += offer.size_bytes if beyond else 0
    return out


def share(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan
