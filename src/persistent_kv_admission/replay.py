"""Prefix-dependency-aware fixed-budget trace replay."""

from __future__ import annotations

import bisect
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol

from .trace import Request, Trace

# Called once per sampled eviction decision with the candidate leaves, their
# score tuples in the same order, the index of the chosen victim, the current
# timestamp, and the current timestamp-group index.
DecisionHook = Callable[[list[str], list[tuple[float, ...]], int, float, int], None]
# Called after every timestamp group has been observed and inserted.
GroupHook = Callable[["_PrefixClosedCache", int, float], None]
# Called with the victim's state id, the current timestamp, and the current
# timestamp-group index just before a leaf is evicted, on either eviction path.
EvictHook = Callable[[str, float, int], None]


class StateScorer(Protocol):
    """Supplies a retention priority for a cached state at a point in time.

    Higher means more valuable. `time_varying` tells the cache whether a stored
    heap key can go stale between touches and therefore needs re-evaluation on
    the eviction path.
    """

    time_varying: bool

    def observe(self, requests: list[Request], timestamp_ms: float) -> None: ...

    def score(self, state_id: str, timestamp_ms: float) -> float: ...


# Maximum re-insertions allowed while searching for one eviction victim.
RESCORE_LIMIT = 64

POLICIES = (
    "lru",
    "lru_2hit",
    "lfu",
    "longest_first",
    "frequency_x_prefix_length",
    "structural",
    "offline_next_use",
)


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    size_model: str
    capacity_bytes: int
    capacity_fraction: float
    avoided_prefill_tokens: int
    reused_prefix_tokens: int
    block_hit_rate: float
    request_hit_rate: float
    avoided_tokens_per_cache_byte: float
    requested_tokens: int
    requested_blocks: int
    hit_blocks: int
    measured_requests: int = 0


class _PrefixClosedCache:
    def __init__(
        self,
        trace: Trace,
        capacity_bytes: int,
        bytes_per_token: int,
        policy: str,
        occurrence_groups: dict[str, list[int]],
        size_model: str,
        scorer: StateScorer | None = None,
        eviction: str = "heap",
        sample_width: int = 16,
        seed: int = 0,
        decision_hook: DecisionHook | None = None,
        evict_hook: EvictHook | None = None,
    ) -> None:
        self.trace = trace
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = bytes_per_token
        self.policy = policy
        self.occurrence_groups = occurrence_groups
        self.size_model = size_model
        # Insertion-ordered dicts, not sets: a set of strings iterates in an
        # order that depends on per-process hash randomisation, which would make
        # the sampled candidate draw, and so every seeded result, unrepeatable.
        self.cached: dict[str, None] = {}
        self.cached_children: dict[str, int] = defaultdict(int)
        self.current_bytes = 0
        self.frequency: dict[str, int] = defaultdict(int)
        self.last_group: dict[str, int] = defaultdict(lambda: -1)
        self.observed_children: dict[str, set[str]] = defaultdict(set)
        self.observed_terminals: dict[str, set[str]] = defaultdict(set)
        self.versions: dict[str, int] = defaultdict(int)
        self.heap: list[tuple[tuple[float, ...], int, int, str]] = []
        self.serial = 0
        self.group_index = -1
        self.scorer = scorer
        self.timestamp_ms = 0.0
        self.rescored_pops = 0
        self.eviction = eviction
        self.sample_width = sample_width
        self.rng = random.Random(seed)
        self.leaves: dict[str, None] = {}
        self.decision_hook = decision_hook
        self.evict_hook = evict_hook
        if scorer is not None and hasattr(scorer, "attach"):
            # Lets a learning scorer draw training examples from the population
            # it actually makes decisions over, rather than every state ever seen.
            scorer.attach(self)

    def _state_bytes(self, state_id: str) -> int:
        tokens = (
            self.trace.states[state_id].block_tokens
            if self.size_model == "packed"
            else self.trace.block_size
        )
        return tokens * self.bytes_per_token

    def _score(self, state_id: str) -> tuple[float, ...]:
        meta = self.trace.states[state_id]
        last = self.last_group[state_id]
        if self.scorer is not None:
            return (self.scorer.score(state_id, self.timestamp_ms), float(last))
        if self.policy in {"lru", "lru_2hit"}:
            return (float(last),)
        if self.policy == "lfu":
            return (float(self.frequency[state_id]), float(last))
        if self.policy == "longest_first":
            return (float(meta.prefix_tokens), float(last))
        if self.policy == "frequency_x_prefix_length":
            return (float(self.frequency[state_id] * meta.prefix_tokens), float(last))
        if self.policy == "structural":
            structural = len(self.observed_children[state_id]) + len(
                self.observed_terminals[state_id]
            )
            return (float(structural), float(last))
        if self.policy == "offline_next_use":
            groups = self.occurrence_groups[state_id]
            position = bisect.bisect_right(groups, self.group_index)
            next_group = groups[position] if position < len(groups) else math.inf
            return (-float(next_group), float(meta.prefix_tokens))
        raise ValueError(f"unknown policy {self.policy!r}")

    def _push(self, state_id: str) -> None:
        self.serial += 1
        heapq.heappush(
            self.heap,
            (self._score(state_id), self.serial, self.versions[state_id], state_id),
        )

    def update_history(
        self, requests: list[Request], group_index: int, timestamp_ms: float = 0.0
    ) -> None:
        self.group_index = group_index
        self.timestamp_ms = timestamp_ms
        if self.scorer is not None:
            self.scorer.observe(requests, timestamp_ms)
        # Insertion-ordered dict, not a set: iterating this assigns the heap
        # serials that break ties between equal scores, so the order must not
        # depend on per-process string-hash randomisation.
        touched: dict[str, None] = {}
        for request in requests:
            terminal = request.hash_ids[-1]
            for index, state_id in enumerate(request.hash_ids):
                self.frequency[state_id] += 1
                self.last_group[state_id] = group_index
                self.observed_terminals[state_id].add(terminal)
                if index + 1 < len(request.hash_ids):
                    self.observed_children[state_id].add(request.hash_ids[index + 1])
                touched[state_id] = None
        for state_id in touched:
            self.versions[state_id] += 1
            # Sampled eviction never reads the heap, so feeding it would only
            # cost memory and time.
            if self.eviction == "heap" and state_id in self.cached:
                self._push(state_id)

    def hit_prefix_blocks(self, request: Request) -> int:
        count = 0
        for state_id in request.hash_ids:
            if state_id not in self.cached:
                break
            count += 1
        return count

    def insert_requests(self, requests: list[Request]) -> None:
        # Insert complete chains before eviction. The cache is temporarily over
        # budget, then leaf-only eviction restores a prefix-closed retained set.
        for request in requests:
            for state_id in request.hash_ids:
                if self.policy == "lru_2hit" and self.frequency[state_id] < 2:
                    continue
                if state_id in self.cached:
                    continue
                parent = self.trace.states[state_id].parent_id
                if parent is not None and parent not in self.cached:
                    # A request is a complete root-to-leaf chain, so this means
                    # an earlier duplicate in the batch was pruned unexpectedly.
                    raise RuntimeError(f"cannot cache child {state_id} without parent {parent}")
                self.cached[state_id] = None
                self.current_bytes += self._state_bytes(state_id)
                if parent is not None:
                    self.cached_children[parent] += 1
                    self.leaves.pop(parent, None)
                self.cached_children.setdefault(state_id, 0)
                self.leaves[state_id] = None
                if self.eviction == "heap":
                    self._push(state_id)
        while self.current_bytes > self.capacity_bytes and self.cached:
            if self.eviction == "sampled":
                self._evict_one_leaf_sampled()
            else:
                self._evict_one_leaf()

    def _evict_one_leaf(self) -> None:
        rescores = 0
        while self.heap:
            key, _, version, state_id = heapq.heappop(self.heap)
            if state_id not in self.cached:
                continue
            if version != self.versions[state_id]:
                continue
            if self.cached_children[state_id] != 0:
                continue
            if self.scorer is not None and self.scorer.time_varying:
                # Heap keys are stale between touches for time-varying scores.
                # Re-evaluate the candidate and reinsert it when it is worth
                # more now than the stored key claimed. The budget bounds the
                # worst case; beyond it the current candidate is evicted.
                current = self._score(state_id)
                if current > key and rescores < RESCORE_LIMIT:
                    rescores += 1
                    self.rescored_pops += 1
                    self.serial += 1
                    heapq.heappush(
                        self.heap, (current, self.serial, self.versions[state_id], state_id)
                    )
                    continue
            if self.evict_hook is not None:
                self.evict_hook(state_id, self.timestamp_ms, self.group_index)
            self._remove(state_id, push_parent=True)
            return
        raise RuntimeError("no evictable leaf found in non-empty prefix-closed cache")

    def _remove(self, state_id: str, push_parent: bool) -> None:
        del self.cached[state_id]
        self.leaves.pop(state_id, None)
        self.current_bytes -= self._state_bytes(state_id)
        parent = self.trace.states[state_id].parent_id
        if parent is not None and parent in self.cached:
            self.cached_children[parent] -= 1
            if self.cached_children[parent] == 0:
                self.leaves[parent] = None
                if push_parent:
                    self._push(parent)

    def _evict_one_leaf_sampled(self) -> None:
        """Evict the worst of a small random sample of retained leaves.

        Heap keys are written when a state is touched, so with a time-varying
        score they describe the state as it looked then, not now. A stale key
        keeps an old state ranked as highly as when it was fresh, which inverts
        the policy. Sampling re-scores every candidate at the moment of the
        decision, so the score is always current. This is the approximation real
        caches use, and it costs O(sample_width) per eviction instead of a full
        re-ranking.
        """
        if not self.leaves:
            raise RuntimeError("no evictable leaf found in non-empty prefix-closed cache")
        leaves = self.leaves
        if len(leaves) <= self.sample_width:
            candidates = list(leaves)
        else:
            candidates = self.rng.sample(list(leaves), self.sample_width)
        scores = [self._score(state_id) for state_id in candidates]
        # First minimum in draw order, which is what min(candidates, key=...)
        # would pick; kept explicit so the hook sees the same scores.
        victim_index = min(range(len(candidates)), key=scores.__getitem__)
        if self.decision_hook is not None:
            self.decision_hook(candidates, scores, victim_index, self.timestamp_ms, self.group_index)
        if self.evict_hook is not None:
            self.evict_hook(candidates[victim_index], self.timestamp_ms, self.group_index)
        self._remove(candidates[victim_index], push_parent=False)


def _occurrence_groups(trace: Trace) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for group_index, (_, requests) in enumerate(trace.timestamp_groups()):
        # Ordered, not a set: the per-state lists are the same either way, but
        # the key order of the returned mapping would otherwise depend on hash
        # randomisation, and it is passed around as a plain dict.
        present: dict[str, None] = {}
        for request in requests:
            for state_id in request.hash_ids:
                present[state_id] = None
        for state_id in present:
            result[state_id].append(group_index)
    return dict(result)


def replay(
    trace: Trace,
    policy: str,
    capacity_bytes: int,
    capacity_fraction: float,
    bytes_per_token: int,
    size_model: str = "packed",
    occurrence_groups: dict[str, list[int]] | None = None,
    scorer: StateScorer | None = None,
    measure_from_ms: float | None = None,
    eviction: str = "heap",
    sample_width: int = 16,
    seed: int = 0,
    decision_hook: DecisionHook | None = None,
    group_hook: GroupHook | None = None,
    evict_hook: EvictHook | None = None,
) -> ReplayResult:
    if scorer is None and policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if size_model not in {"packed", "fixed_block"}:
        raise ValueError(f"unknown size model {size_model!r}")
    if eviction not in {"heap", "sampled"}:
        raise ValueError(f"unknown eviction mechanism {eviction!r}")
    groups = occurrence_groups or _occurrence_groups(trace)
    cache = _PrefixClosedCache(
        trace, capacity_bytes, bytes_per_token, policy, groups, size_model, scorer,
        eviction, sample_width, seed, decision_hook, evict_hook,
    )
    requested_tokens = 0
    requested_blocks = 0
    hit_blocks = 0
    reused_tokens = 0
    request_hits = 0
    measured_requests = 0

    for group_index, (timestamp_ms, requests) in enumerate(trace.timestamp_groups()):
        # Requests before measure_from_ms still warm the cache but are excluded
        # from the reported metrics, so every policy is scored on one window.
        measured = measure_from_ms is None or timestamp_ms >= measure_from_ms
        # All requests with the same timestamp observe the pre-batch cache.
        for request in requests:
            hits = cache.hit_prefix_blocks(request)
            if measured:
                measured_requests += 1
                hit_blocks += hits
                requested_blocks += len(request.hash_ids)
                requested_tokens += request.input_length
                tokens = min(request.input_length, hits * trace.block_size)
                reused_tokens += tokens
                request_hits += hits > 0
        cache.update_history(requests, group_index, timestamp_ms)
        cache.insert_requests(requests)
        if group_hook is not None:
            group_hook(cache, group_index, timestamp_ms)

    return ReplayResult(
        policy=policy,
        size_model=size_model,
        capacity_bytes=capacity_bytes,
        capacity_fraction=capacity_fraction,
        avoided_prefill_tokens=reused_tokens,
        reused_prefix_tokens=reused_tokens,
        block_hit_rate=hit_blocks / max(requested_blocks, 1),
        request_hit_rate=request_hits / max(measured_requests, 1),
        avoided_tokens_per_cache_byte=reused_tokens / max(capacity_bytes, 1),
        requested_tokens=requested_tokens,
        requested_blocks=requested_blocks,
        hit_blocks=hit_blocks,
        measured_requests=measured_requests,
    )
