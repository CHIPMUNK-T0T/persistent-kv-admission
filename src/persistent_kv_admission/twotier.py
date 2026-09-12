"""Two-tier replay: a fixed upper tier spills evicted leaves into a lower tier.

The upper tier (L1) is the existing prefix-closed cache with deterministic
heap eviction under LRU or LFU. Every arriving block enters L1, hit or miss,
so L1's evolution and its victim stream do not depend on the lower tier: the
same event stream feeds every L2 arm exactly, and there is no closed loop to
approximate. The lower tier (L2) receives each L1 victim at eviction time and
decides what to keep. This is the population a persistent tier actually
decides about.

Three models, fixed before the run:

* ``closure="union"`` (primary). A block in L2 is reusable when its whole
  ancestor chain is present in L1 ∪ L2 at request time; L2 itself carries no
  structural constraint. L2 is exclusive: a block that is requested while in
  L2 is materialised in L1 by that request either way (served from L2 if
  usable, recomputed otherwise), so it leaves L2. A block therefore sits in
  L2 with the history it had at eviction, and its L2 priority never goes
  stale.
* ``hit_model="independent"`` (control). Same L2 contents rule, but a block
  in either tier counts as reused whether or not its ancestors are present.
  This is reachable only by a system that recomputes the holes and is used as
  the upper bound that isolates the cost of prefix dependency.
* ``closure="standalone"`` (sensitivity). L2 must be prefix-closed by itself:
  admitting a victim copies its missing ancestors from L1 and charges them to
  L2; L2 is inclusive and evicts leaves only.

Every L1 eviction is logged as one `VictimEvent` (unit: state × eviction
event, since the same state can be evicted several times). Fields known at
eviction time are causal; reuse-time fields are filled when the state is next
requested and are evaluation-only. An event whose state is not requested
again before the trace ends is right-censored, not "never reused".
"""

from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

from .crossworkload import HYPERPARAMETERS
from .replay import _PrefixClosedCache, _occurrence_groups, replay
from .trace import Request, Trace

L1_POLICIES = ("lru", "lfu")
L2_POLICIES = ("lru", "lfu", "lru_2hit", "offline_next_use")
HIT_MODELS = ("tree", "independent")
CLOSURES = ("union", "standalone")
# Tie-break between two L2 blocks with the same next use under the offline
# comparator. "prefix_first" (default, the published setting) keeps the longer
# prefix; "deeper_first" evicts it. Diagnostic only: the default is unchanged.
OFFLINE_TIEBREAKS = ("prefix_first", "deeper_first")
# Depth bins (1-based block depth) used for the allocation breakdowns.
DEPTH_BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, None))


def depth_bin(depth: int) -> str:
    for low, high in DEPTH_BINS:
        if high is None and depth >= low:
            return f"{low}+"
        if high is not None and low <= depth <= high:
            return f"{low}" if low == high else f"{low}-{high}"
    raise ValueError(f"depth {depth} outside bins")


DEPTH_LABELS = tuple(depth_bin(low) for low, _ in DEPTH_BINS)


@dataclass
class VictimEvent:
    """One L1 eviction. Causal fields first, evaluation-only fields after.

    The distance-to-reuse fields (`requests_to_reuse`, `arriving_bytes_to_reuse`,
    `distinct_bytes_to_reuse`) count the requests strictly after the eviction's
    timestamp group and strictly before the reuse's timestamp group. They
    therefore do not depend on the row order inside a timestamp group: every
    request of a group observes the same pre-batch cache, and the eviction
    itself happens after the whole group has been served. `reuse_order` is the
    row index of the first request of the reuse group that touches the state,
    so it does depend on that order; it is reported, never used as a distance
    boundary. The `*_to_end` fields run to the end of the trace.
    """

    event_index: int
    state_id: str
    timestamp_ms: float
    group_index: int
    measured: bool
    depth: int
    block_tokens: int
    prefix_tokens: int
    frequency: int
    seconds_since_last_use: float
    prior_evictions: int
    siblings_retained: int
    parent_becomes_leaf: bool
    children_observed: int
    # --- filled at the next request of the state (evaluation only) ---
    censored: bool = True
    reuse_timestamp_ms: float = math.nan
    reuse_group: int = -1
    reuse_order: int = -1
    reuse_measured: bool = False
    seconds_to_reuse: float = math.nan
    requests_to_reuse: int = -1
    arriving_bytes_to_reuse: int = -1
    distinct_bytes_to_reuse: int = -1
    l1_prefix_blocks_at_reuse: int = -1
    missing_ancestor_blocks: int = -1
    missing_ancestor_bytes: int = -1
    reuse_requests_in_group: int = 0
    rescue_tokens_self: int = 0
    rescue_tokens_with_ancestors: int = 0
    # --- censoring information ---
    seconds_to_end: float = math.nan
    requests_to_end: int = -1
    arriving_bytes_to_end: int = -1
    distinct_bytes_to_end: int = -1
    future_uses: int = 0


@dataclass
class TwoTierResult:
    """One two-tier replay.

    Restricted to the evaluation window (`measure_from_ms`, or the whole trace
    when it is None): `measured_requests`, `requested_tokens`,
    `requested_blocks`, `avoided_prefill_tokens`, `l1_avoided_tokens`,
    `l2_avoided_tokens`, `l1_hit_blocks`, `l2_hit_blocks`,
    `l2_present_unusable_blocks`, `l2_present_unusable_tokens`,
    `l2_avoided_tokens_by_depth`; `l2_byte_seconds` and
    `l2_byte_seconds_by_depth`, which integrate residency only over intervals
    that start inside the window; and `l2_admitted_bytes_by_depth`, which
    counts only admissions caused by an eviction inside the window.

    Whole-trace (the warm-up is part of the mechanism, so it is counted):
    `l1_evictions`, `l2_admissions`, `l2_rejections`, `l2_already_held`,
    `l2_evictions`, `l2_ancestor_copy_bytes`. The victim event log is
    whole-trace too; each event carries its own `measured` flag.
    """

    trace: str
    l1_policy: str
    l1_capacity_bytes: int
    l2_policy: str
    l2_capacity_bytes: int
    hit_model: str
    closure: str
    measured_requests: int
    requested_tokens: int
    requested_blocks: int
    avoided_prefill_tokens: int
    l1_avoided_tokens: int
    l2_avoided_tokens: int
    l1_hit_blocks: int
    l2_hit_blocks: int
    l2_present_unusable_blocks: int
    l2_present_unusable_tokens: int
    l1_evictions: int
    l2_admissions: int
    l2_rejections: int
    l2_evictions: int
    l2_ancestor_copy_bytes: int
    l2_byte_seconds: float
    # L1 evictions of a block the inclusive standalone L2 already holds: no
    # admission and no rejection, so the three add up to l1_evictions.
    l2_already_held: int = 0
    offline_tiebreak: str = "prefix_first"
    l2_byte_seconds_by_depth: dict[str, float] = field(default_factory=dict)
    l2_avoided_tokens_by_depth: dict[str, int] = field(default_factory=dict)
    l2_admitted_bytes_by_depth: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row = {k: v for k, v in asdict(self).items() if not isinstance(v, dict)}
        for label in DEPTH_LABELS:
            row[f"l2_byte_seconds_depth_{label}"] = self.l2_byte_seconds_by_depth.get(label, 0.0)
            row[f"l2_avoided_tokens_depth_{label}"] = self.l2_avoided_tokens_by_depth.get(label, 0)
            row[f"l2_admitted_bytes_depth_{label}"] = self.l2_admitted_bytes_by_depth.get(label, 0)
        return row


class _VictimStore:
    """Exclusive lower tier under union closure.

    Priorities are computed once at admission from the state's history at
    eviction (recency, lifetime frequency) or from its next occurrence
    (offline). They cannot go stale: any later request of the state moves it
    back to L1 and out of this store.
    """

    def __init__(
        self,
        trace: Trace,
        capacity_bytes: int,
        policy: str,
        occurrence_groups: dict[str, list[int]],
        state_bytes: Callable[[str], int],
        offline_tiebreak: str = "prefix_first",
    ) -> None:
        if policy not in L2_POLICIES:
            raise ValueError(f"unknown L2 policy {policy!r}")
        if offline_tiebreak not in OFFLINE_TIEBREAKS:
            raise ValueError(f"unknown offline tie-break {offline_tiebreak!r}")
        self.offline_tiebreak = offline_tiebreak
        self.trace = trace
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.groups = occurrence_groups
        self.state_bytes = state_bytes
        self.cached: dict[str, None] = {}
        self.current_bytes = 0
        self.heap: list[tuple[tuple[float, ...], int, int, str]] = []
        self.serial = 0
        self.versions: dict[str, int] = defaultdict(int)
        self.admissions = 0
        self.rejections = 0
        self.evictions = 0
        # Live residency by depth. Admitted bytes by depth are accumulated by
        # the caller, which knows whether the eviction that caused the
        # admission falls inside the evaluation window.
        self.bytes_by_depth: dict[str, int] = defaultdict(int)

    def _key(self, state_id: str, frequency: int, last_group: int, group_index: int) -> tuple[float, ...]:
        if self.policy in {"lru", "lru_2hit"}:
            return (float(last_group),)
        if self.policy == "lfu":
            return (float(frequency), float(last_group))
        occurrences = self.groups.get(state_id, [])
        position = bisect.bisect_right(occurrences, group_index)
        next_group = occurrences[position] if position < len(occurrences) else math.inf
        prefix_tokens = float(self.trace.states[state_id].prefix_tokens)
        # The smallest key is evicted first, so with equal next use
        # "prefix_first" evicts the shorter prefix and "deeper_first" the
        # longer one.
        if self.offline_tiebreak == "deeper_first":
            return (-float(next_group), -prefix_tokens)
        return (-float(next_group), prefix_tokens)

    def admit(self, state_id: str, frequency: int, last_group: int, group_index: int) -> bool:
        if self.policy == "lru_2hit" and frequency < 2:
            self.rejections += 1
            return False
        if state_id in self.cached:
            raise RuntimeError(f"{state_id} evicted from L1 while resident in exclusive L2")
        size = self.state_bytes(state_id)
        label = depth_bin(self.trace.states[state_id].depth)
        self.cached[state_id] = None
        self.current_bytes += size
        self.bytes_by_depth[label] += size
        self.admissions += 1
        self.serial += 1
        heapq.heappush(
            self.heap,
            (self._key(state_id, frequency, last_group, group_index), self.serial, self.versions[state_id], state_id),
        )
        while self.current_bytes > self.capacity_bytes and self.heap:
            _, _, version, candidate = heapq.heappop(self.heap)
            if candidate not in self.cached or version != self.versions[candidate]:
                continue
            self._remove(candidate)
            self.evictions += 1
        return True

    def _remove(self, state_id: str) -> None:
        del self.cached[state_id]
        size = self.state_bytes(state_id)
        self.current_bytes -= size
        self.bytes_by_depth[depth_bin(self.trace.states[state_id].depth)] -= size
        self.versions[state_id] += 1

    def promote(self, state_id: str) -> None:
        """The state is being materialised in L1 by a request; it leaves L2."""
        if state_id in self.cached:
            self._remove(state_id)


def _chain(trace: Trace, state_id: str) -> tuple[str, ...]:
    chain = []
    current: str | None = state_id
    while current is not None:
        chain.append(current)
        current = trace.states[current].parent_id
    chain.reverse()
    return tuple(chain)


def run_two_tier(
    trace: Trace,
    l1_policy: str,
    l1_capacity_bytes: int,
    l2_capacity_bytes: int,
    l2_policy: str = "lru",
    hit_model: str = "tree",
    closure: str = "union",
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
    measure_from_ms: float | None = None,
    occurrence_groups: dict[str, list[int]] | None = None,
    event_sink: list[VictimEvent] | None = None,
    offline_tiebreak: str = "prefix_first",
) -> TwoTierResult:
    """Replay L1 (fixed) and L2 (the arm) in one pass; optionally log every L1 eviction.

    `offline_tiebreak` selects how the offline L2 comparator breaks a tie
    between two blocks with the same next use inside the union-closure store;
    it is a diagnostic knob, ignored by every other L2 policy and by the
    standalone closure, which uses the replay engine's own offline key.
    `TwoTierResult` documents which returned fields are restricted to the
    evaluation window and which cover the whole trace.
    """
    if l1_policy not in L1_POLICIES:
        raise ValueError(f"unknown L1 policy {l1_policy!r}")
    if hit_model not in HIT_MODELS:
        raise ValueError(f"unknown hit model {hit_model!r}")
    if closure not in CLOSURES:
        raise ValueError(f"unknown closure {closure!r}")
    if offline_tiebreak not in OFFLINE_TIEBREAKS:
        raise ValueError(f"unknown offline tie-break {offline_tiebreak!r}")
    if closure == "standalone" and hit_model != "tree":
        raise ValueError("standalone closure is defined for the tree hit model only")
    use_l2 = l2_capacity_bytes > 0 and l2_policy != "none"
    if use_l2 and l2_policy not in L2_POLICIES:
        raise ValueError(f"unknown L2 policy {l2_policy!r}")
    if use_l2 and closure == "standalone" and l2_policy == "lru_2hit":
        raise ValueError("lru_2hit is not defined for the standalone closure")
    groups = occurrence_groups or _occurrence_groups(trace)
    states = trace.states

    def state_bytes(state_id: str) -> int:
        tokens = states[state_id].block_tokens if size_model == "packed" else trace.block_size
        return tokens * bytes_per_token

    counters = defaultdict(int)
    l2_bytes_by_depth: dict[str, int] = defaultdict(int)
    l2_admitted_by_depth: dict[str, int] = defaultdict(int)
    l2_avoided_by_depth: dict[str, int] = defaultdict(int)
    byte_seconds_by_depth: dict[str, float] = defaultdict(float)
    open_events: dict[str, VictimEvent] = {}
    eviction_counts: dict[str, int] = defaultdict(int)
    l2_union: _VictimStore | None = None
    l2_standalone: _PrefixClosedCache | None = None

    def on_l2_evict(state_id: str, timestamp_ms: float, group_index: int) -> None:
        counters["l2_evictions"] += 1
        l2_bytes_by_depth[depth_bin(states[state_id].depth)] -= state_bytes(state_id)

    if use_l2:
        if closure == "union":
            l2_union = _VictimStore(trace, l2_capacity_bytes, l2_policy, groups, state_bytes,
                                    offline_tiebreak=offline_tiebreak)
            l2_bytes_by_depth = l2_union.bytes_by_depth
        else:
            l2_standalone = _PrefixClosedCache(
                trace, l2_capacity_bytes, bytes_per_token, l2_policy, groups, size_model,
                eviction="heap", evict_hook=on_l2_evict,
            )
    l2_cached: dict[str, None] = (
        l2_union.cached if l2_union is not None else l2_standalone.cached if l2_standalone is not None else {}
    )

    def on_l1_evict(state_id: str, timestamp_ms: float, group_index: int) -> None:
        counters["l1_evictions"] += 1
        meta = states[state_id]
        # Admitted bytes by depth are attributed to the eviction that caused
        # them, so they are counted only for evictions inside the window.
        measured_eviction = measure_from_ms is None or timestamp_ms >= measure_from_ms
        if event_sink is not None:
            occurrences = trace.occurrences_ms[state_id]
            last_index = bisect.bisect_right(occurrences, timestamp_ms) - 1
            last_use = occurrences[last_index] if last_index >= 0 else timestamp_ms
            parent = meta.parent_id
            siblings = l1.cached_children[parent] - 1 if parent is not None else 0
            event = VictimEvent(
                event_index=len(event_sink),
                state_id=state_id,
                timestamp_ms=timestamp_ms,
                group_index=group_index,
                measured=measured_eviction,
                depth=meta.depth,
                block_tokens=meta.block_tokens,
                prefix_tokens=meta.prefix_tokens,
                frequency=l1.frequency[state_id],
                seconds_since_last_use=(timestamp_ms - last_use) / 1000.0,
                prior_evictions=eviction_counts[state_id],
                siblings_retained=max(siblings, 0),
                parent_becomes_leaf=parent is not None and siblings == 0,
                children_observed=len(l1.observed_children[state_id]),
            )
            event_sink.append(event)
            open_events[state_id] = event
        eviction_counts[state_id] += 1
        if l2_union is not None:
            if l2_union.admit(state_id, l1.frequency[state_id], l1.last_group[state_id], group_index):
                counters["l2_admissions"] += 1
                if measured_eviction:
                    l2_admitted_by_depth[depth_bin(meta.depth)] += state_bytes(state_id)
            else:
                counters["l2_rejections"] += 1
        elif l2_standalone is not None:
            chain = _chain(trace, state_id)
            missing = [s for s in chain if s not in l2_standalone.cached]
            if state_id in l2_standalone.cached:
                # Inclusive tier: the block is already held (it was reused from
                # L2 and never left). Neither an admission nor a rejection, so
                # it is counted on its own to keep the three exhaustive.
                counters["l2_already_held"] += 1
                return
            l2_standalone.insert_requests([
                Request(order=-1, timestamp_ms=timestamp_ms, input_length=meta.prefix_tokens,
                        output_length=0, hash_ids=chain)
            ])
            counters["l2_admissions"] += 1
            for s in missing:
                size = state_bytes(s)
                label = depth_bin(states[s].depth)
                if measured_eviction:
                    l2_admitted_by_depth[label] += size
                if s != state_id:
                    counters["l2_ancestor_copy_bytes"] += size
                # Inserted states may already have been evicted again by the
                # insertion's own leaf eviction; the eviction hook subtracted
                # them, so add unconditionally to keep the running total exact.
                l2_bytes_by_depth[label] += size

    l1 = _PrefixClosedCache(
        trace, l1_capacity_bytes, bytes_per_token, l1_policy, groups, size_model,
        eviction="heap", evict_hook=on_l1_evict,
    )

    previous_ms: float | None = None
    for group_index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        # Residency is integrated over intervals whose start falls inside the
        # evaluation window, so the byte-seconds match the window every token
        # counter is restricted to.
        if (previous_ms is not None and use_l2
                and (measure_from_ms is None or previous_ms >= measure_from_ms)):
            dt = (timestamp_ms - previous_ms) / 1000.0
            for label, held in l2_bytes_by_depth.items():
                byte_seconds_by_depth[label] += held * dt
        previous_ms = timestamp_ms
        measured = measure_from_ms is None or timestamp_ms >= measure_from_ms
        promote: dict[str, None] = {}
        # Simultaneous requests all observe the pre-batch cache, so every
        # request in the group that contains an evicted state would be served
        # from L2 if the state were kept; the event counts them all.
        duplicates: dict[str, int] = defaultdict(int)
        if open_events:
            for request in requests:
                for state_id in request.hash_ids:
                    if state_id in open_events:
                        duplicates[state_id] += 1
        for request in requests:
            ids = request.hash_ids
            prefix = 0
            while prefix < len(ids) and ids[prefix] in l1.cached:
                prefix += 1
            l2_hits: list[int] = []
            present: list[int] = []
            if use_l2:
                if hit_model == "tree":
                    index = prefix
                    while index < len(ids) and ids[index] in l2_cached:
                        l2_hits.append(index)
                        index += 1
                    present = l2_hits + [i for i in range(index, len(ids)) if ids[i] in l2_cached]
                else:
                    l2_hits = [i for i in range(prefix, len(ids)) if ids[i] in l2_cached]
                    present = l2_hits
            for index, state_id in enumerate(ids):
                event = open_events.pop(state_id, None)
                if event is None:
                    continue
                # The state is not in L1 (it was evicted and not reinserted),
                # so the L1 prefix ends at or before it.
                event.censored = False
                event.reuse_timestamp_ms = timestamp_ms
                event.reuse_group = group_index
                event.reuse_order = request.order
                event.reuse_measured = measured
                event.seconds_to_reuse = (timestamp_ms - event.timestamp_ms) / 1000.0
                event.l1_prefix_blocks_at_reuse = prefix
                event.missing_ancestor_blocks = index - prefix
                event.missing_ancestor_bytes = sum(state_bytes(s) for s in ids[prefix:index])
                event.reuse_requests_in_group = duplicates[state_id]
                event.rescue_tokens_self = event.block_tokens * duplicates[state_id] if prefix == index else 0
                event.rescue_tokens_with_ancestors = event.block_tokens * duplicates[state_id]
            if measured:
                counters["measured_requests"] += 1
                counters["requested_tokens"] += request.input_length
                counters["requested_blocks"] += len(ids)
                counters["l1_hit_blocks"] += prefix
                l1_tokens = states[ids[prefix - 1]].prefix_tokens if prefix else 0
                counters["l1_avoided_tokens"] += l1_tokens
                for index in l2_hits:
                    tokens = states[ids[index]].block_tokens
                    counters["l2_avoided_tokens"] += tokens
                    counters["l2_hit_blocks"] += 1
                    l2_avoided_by_depth[depth_bin(states[ids[index]].depth)] += tokens
                for index in present:
                    if index not in l2_hits:
                        counters["l2_present_unusable_blocks"] += 1
                        counters["l2_present_unusable_tokens"] += states[ids[index]].block_tokens
            if l2_union is not None:
                for index in range(prefix, len(ids)):
                    if ids[index] in l2_cached:
                        promote[ids[index]] = None
        if l2_union is not None:
            for state_id in promote:
                l2_union.promote(state_id)
        l1.update_history(requests, group_index, timestamp_ms)
        if l2_standalone is not None:
            l2_standalone.update_history(requests, group_index, timestamp_ms)
        l1.insert_requests(requests)

    return TwoTierResult(
        trace=trace.name,
        l1_policy=l1_policy,
        l1_capacity_bytes=l1_capacity_bytes,
        l2_policy=l2_policy if use_l2 else "none",
        l2_capacity_bytes=l2_capacity_bytes if use_l2 else 0,
        hit_model=hit_model,
        closure=closure,
        measured_requests=counters["measured_requests"],
        requested_tokens=counters["requested_tokens"],
        requested_blocks=counters["requested_blocks"],
        avoided_prefill_tokens=counters["l1_avoided_tokens"] + counters["l2_avoided_tokens"],
        l1_avoided_tokens=counters["l1_avoided_tokens"],
        l2_avoided_tokens=counters["l2_avoided_tokens"],
        l1_hit_blocks=counters["l1_hit_blocks"],
        l2_hit_blocks=counters["l2_hit_blocks"],
        l2_present_unusable_blocks=counters["l2_present_unusable_blocks"],
        l2_present_unusable_tokens=counters["l2_present_unusable_tokens"],
        l1_evictions=counters["l1_evictions"],
        l2_admissions=counters["l2_admissions"],
        l2_rejections=counters["l2_rejections"],
        l2_already_held=counters["l2_already_held"],
        offline_tiebreak=offline_tiebreak,
        l2_evictions=l2_union.evictions if l2_union is not None else counters["l2_evictions"],
        l2_ancestor_copy_bytes=counters["l2_ancestor_copy_bytes"],
        l2_byte_seconds=sum(byte_seconds_by_depth.values()),
        l2_byte_seconds_by_depth=dict(byte_seconds_by_depth),
        l2_avoided_tokens_by_depth=dict(l2_avoided_by_depth),
        l2_admitted_bytes_by_depth=dict(l2_admitted_by_depth),
    )


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int) -> None:
        index += 1
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def prefix(self, count: int) -> int:
        total = 0
        while count > 0:
            total += self.tree[count]
            count -= count & -count
        return total

    def range(self, start: int, end: int) -> int:
        """Sum over positions [start, end)."""
        if end <= start:
            return 0
        return self.prefix(end) - self.prefix(start)


def annotate_events(
    trace: Trace,
    events: list[VictimEvent],
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
) -> None:
    """Fill the policy-independent distance fields of every event in place.

    Distances to reuse run over the requests strictly after the eviction's
    timestamp group and strictly before the reuse's timestamp group; distances
    to the trace end run from the same start to the last request. Both ends are
    timestamp-group boundaries, so a distance never depends on the row order
    inside a group: the eviction happens after its whole group has been served,
    and every request of the reuse group observes the same pre-batch cache, so
    counting only part of that group as competing traffic would be an artefact
    of the file order. `distinct_bytes` counts each state once (its block
    bytes), whatever the number of requests that touch it.
    """
    requests = trace.requests
    count = len(requests)
    group_first: list[int] = []
    for _, group in trace.timestamp_groups():
        group_first.append(group[0].order)
    group_first.append(count)

    def state_bytes(state_id: str) -> int:
        tokens = trace.states[state_id].block_tokens if size_model == "packed" else trace.block_size
        return tokens * bytes_per_token

    cumulative = [0] * (count + 1)
    position_start = [0] * (count + 1)
    for request in requests:
        cumulative[request.order + 1] = cumulative[request.order] + request.input_length * bytes_per_token
        position_start[request.order + 1] = position_start[request.order] + len(request.hash_ids)
    total_positions = position_start[count]

    queries: list[tuple[int, int, VictimEvent, bool]] = []
    for event in events:
        start = group_first[event.group_index + 1]
        occurrences = trace.occurrences_ms[event.state_id]
        event.future_uses = len(occurrences) - bisect.bisect_right(occurrences, event.timestamp_ms)
        event.seconds_to_end = (trace.end_ms - event.timestamp_ms) / 1000.0
        event.requests_to_end = count - start
        event.arriving_bytes_to_end = cumulative[count] - cumulative[start]
        queries.append((position_start[count], position_start[start], event, True))
        if not event.censored:
            end = group_first[event.reuse_group]
            event.requests_to_reuse = end - start
            event.arriving_bytes_to_reuse = cumulative[end] - cumulative[start]
            queries.append((position_start[end], position_start[start], event, False))
    queries.sort(key=lambda q: q[0])

    tree = _Fenwick(total_positions)
    last_position: dict[str, int] = {}
    position = 0
    request_index = 0
    block_index = 0
    for right, left, event, to_end in queries:
        while position < right:
            request = requests[request_index]
            state_id = request.hash_ids[block_index]
            previous = last_position.get(state_id)
            size = state_bytes(state_id)
            if previous is not None:
                tree.add(previous, -size)
            tree.add(position, size)
            last_position[state_id] = position
            position += 1
            block_index += 1
            if block_index == len(request.hash_ids):
                block_index = 0
                request_index += 1
        value = tree.range(left, right)
        if to_end:
            event.distinct_bytes_to_end = value
        else:
            event.distinct_bytes_to_reuse = value


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def summarize_events(
    events: list[VictimEvent],
    l1_capacity_bytes: int,
    requested_tokens: int,
    multipliers: tuple[float, ...] = (1, 2, 4, 8),
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Aggregate an event log: one summary row and one row per depth bin.

    Event counts and reuse shares are over events whose eviction falls in the
    measured window; rescue tokens are over events whose reuse falls in it,
    which matches how the replay attributes avoided tokens.
    """
    measured = [e for e in events if e.measured]
    reused = [e for e in measured if not e.censored]
    rescued = [e for e in events if e.reuse_measured]

    def share(items, predicate) -> float:
        return sum(1 for e in items if predicate(e)) / len(items) if items else math.nan

    summary: dict[str, object] = {
        "events_total": len(events),
        "events_measured": len(measured),
        "distinct_states_measured": len({e.state_id for e in measured}),
        "share_repeat_evictions": share(measured, lambda e: e.prior_evictions > 0),
        "share_reused_before_end": share(measured, lambda e: not e.censored),
        "share_censored": share(measured, lambda e: e.censored),
        "share_reused_with_l1_ancestors": share(reused, lambda e: e.missing_ancestor_blocks == 0),
        "mean_depth": sum(e.depth for e in measured) / len(measured) if measured else math.nan,
        "mean_frequency": sum(e.frequency for e in measured) / len(measured) if measured else math.nan,
        "median_seconds_since_last_use": _quantile([e.seconds_since_last_use for e in measured], 0.5),
        "median_seconds_to_reuse": _quantile([e.seconds_to_reuse for e in reused], 0.5),
        "p90_seconds_to_reuse": _quantile([e.seconds_to_reuse for e in reused], 0.9),
        "median_requests_to_reuse": _quantile([e.requests_to_reuse for e in reused], 0.5),
        "median_arriving_bytes_to_reuse": _quantile([e.arriving_bytes_to_reuse for e in reused], 0.5),
        "median_distinct_bytes_to_reuse": _quantile([e.distinct_bytes_to_reuse for e in reused], 0.5),
        "p90_distinct_bytes_to_reuse": _quantile([e.distinct_bytes_to_reuse for e in reused], 0.9),
        "median_distinct_bytes_to_reuse_over_l1": _quantile(
            [e.distinct_bytes_to_reuse / l1_capacity_bytes for e in reused], 0.5),
        "median_missing_ancestor_blocks": _quantile([e.missing_ancestor_blocks for e in reused], 0.5),
        "mean_missing_ancestor_blocks": (
            sum(e.missing_ancestor_blocks for e in reused) / len(reused) if reused else math.nan
        ),
        "rescue_tokens_self": sum(e.rescue_tokens_self for e in rescued),
        "rescue_tokens_with_ancestors": sum(e.rescue_tokens_with_ancestors for e in rescued),
        "rescue_fraction_self": sum(e.rescue_tokens_self for e in rescued) / max(requested_tokens, 1),
        "rescue_fraction_with_ancestors": sum(e.rescue_tokens_with_ancestors for e in rescued) / max(requested_tokens, 1),
        "missing_ancestor_bytes_total": sum(e.missing_ancestor_bytes for e in rescued),
    }
    for multiplier in multipliers:
        limit = multiplier * l1_capacity_bytes
        summary[f"share_reused_within_{multiplier:g}xL1_distinct_bytes"] = share(
            reused, lambda e, limit=limit: e.distinct_bytes_to_reuse <= limit)
        summary[f"rescue_fraction_with_ancestors_within_{multiplier:g}xL1"] = sum(
            e.rescue_tokens_with_ancestors for e in rescued if e.distinct_bytes_to_reuse <= limit
        ) / max(requested_tokens, 1)
    by_depth = []
    for label in DEPTH_LABELS:
        bucket = [e for e in measured if depth_bin(e.depth) == label]
        bucket_reused = [e for e in bucket if not e.censored]
        bucket_rescued = [e for e in rescued if depth_bin(e.depth) == label]
        by_depth.append({
            "depth_bin": label,
            "events_measured": len(bucket),
            "share_of_events": len(bucket) / len(measured) if measured else math.nan,
            "evicted_bytes_share": (
                sum(e.block_tokens for e in bucket) / max(sum(e.block_tokens for e in measured), 1)
            ),
            "share_reused_before_end": share(bucket, lambda e: not e.censored),
            "share_reused_with_l1_ancestors": share(bucket_reused, lambda e: e.missing_ancestor_blocks == 0),
            "median_distinct_bytes_to_reuse": _quantile([e.distinct_bytes_to_reuse for e in bucket_reused], 0.5),
            "rescue_tokens_self": sum(e.rescue_tokens_self for e in bucket_rescued),
            "rescue_tokens_with_ancestors": sum(e.rescue_tokens_with_ancestors for e in bucket_rescued),
            "rescue_share_with_ancestors": (
                sum(e.rescue_tokens_with_ancestors for e in bucket_rescued)
                / max(sum(e.rescue_tokens_with_ancestors for e in rescued), 1)
            ),
            "missing_ancestor_bytes": sum(e.missing_ancestor_bytes for e in bucket_rescued),
        })
    return summary, by_depth


def single_tier_reference(
    trace: Trace,
    policy: str,
    capacity_bytes: int,
    measure_from_ms: float | None,
    bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
    size_model: str = HYPERPARAMETERS["size_model"],
    occurrence_groups: dict[str, list[int]] | None = None,
) -> int:
    """Avoided tokens of one prefix-closed cache holding L1 + L2 capacity (heap eviction)."""
    result = replay(
        trace, policy, capacity_bytes, 0.0, bytes_per_token, size_model=size_model,
        occurrence_groups=occurrence_groups, measure_from_ms=measure_from_ms, eviction="heap",
    )
    return result.avoided_prefill_tokens
