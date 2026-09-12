"""Prefix-dependency-aware fixed-budget trace replay."""

from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

from .trace import Request, Trace


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


class _PrefixClosedCache:
    def __init__(
        self,
        trace: Trace,
        capacity_bytes: int,
        bytes_per_token: int,
        policy: str,
        occurrence_groups: dict[str, list[int]],
        size_model: str,
    ) -> None:
        self.trace = trace
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = bytes_per_token
        self.policy = policy
        self.occurrence_groups = occurrence_groups
        self.size_model = size_model
        self.cached: set[str] = set()
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

    def update_history(self, requests: list[Request], group_index: int) -> None:
        self.group_index = group_index
        touched: set[str] = set()
        for request in requests:
            terminal = request.hash_ids[-1]
            for index, state_id in enumerate(request.hash_ids):
                self.frequency[state_id] += 1
                self.last_group[state_id] = group_index
                self.observed_terminals[state_id].add(terminal)
                if index + 1 < len(request.hash_ids):
                    self.observed_children[state_id].add(request.hash_ids[index + 1])
                touched.add(state_id)
        for state_id in touched:
            self.versions[state_id] += 1
            if state_id in self.cached:
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
                self.cached.add(state_id)
                self.current_bytes += self._state_bytes(state_id)
                if parent is not None:
                    self.cached_children[parent] += 1
                self.cached_children.setdefault(state_id, 0)
                self._push(state_id)
        while self.current_bytes > self.capacity_bytes and self.cached:
            self._evict_one_leaf()

    def _evict_one_leaf(self) -> None:
        while self.heap:
            _, _, version, state_id = heapq.heappop(self.heap)
            if state_id not in self.cached:
                continue
            if version != self.versions[state_id]:
                continue
            if self.cached_children[state_id] != 0:
                continue
            self.cached.remove(state_id)
            self.current_bytes -= self._state_bytes(state_id)
            parent = self.trace.states[state_id].parent_id
            if parent is not None and parent in self.cached:
                self.cached_children[parent] -= 1
                if self.cached_children[parent] == 0:
                    self._push(parent)
            return
        raise RuntimeError("no evictable leaf found in non-empty prefix-closed cache")


def _occurrence_groups(trace: Trace) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for group_index, (_, requests) in enumerate(trace.timestamp_groups()):
        present: set[str] = set()
        for request in requests:
            present.update(request.hash_ids)
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
) -> ReplayResult:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if size_model not in {"packed", "fixed_block"}:
        raise ValueError(f"unknown size model {size_model!r}")
    groups = occurrence_groups or _occurrence_groups(trace)
    cache = _PrefixClosedCache(
        trace, capacity_bytes, bytes_per_token, policy, groups, size_model
    )
    requested_tokens = 0
    requested_blocks = 0
    hit_blocks = 0
    reused_tokens = 0
    request_hits = 0

    for group_index, (_, requests) in enumerate(trace.timestamp_groups()):
        # All requests with the same timestamp observe the pre-batch cache.
        for request in requests:
            hits = cache.hit_prefix_blocks(request)
            hit_blocks += hits
            requested_blocks += len(request.hash_ids)
            requested_tokens += request.input_length
            tokens = min(request.input_length, hits * trace.block_size)
            reused_tokens += tokens
            request_hits += hits > 0
        cache.update_history(requests, group_index)
        cache.insert_requests(requests)

    return ReplayResult(
        policy=policy,
        size_model=size_model,
        capacity_bytes=capacity_bytes,
        capacity_fraction=capacity_fraction,
        avoided_prefill_tokens=reused_tokens,
        reused_prefix_tokens=reused_tokens,
        block_hit_rate=hit_blocks / requested_blocks,
        request_hit_rate=request_hits / len(trace.requests),
        avoided_tokens_per_cache_byte=reused_tokens / max(capacity_bytes, 1),
        requested_tokens=requested_tokens,
        requested_blocks=requested_blocks,
        hit_blocks=hit_blocks,
    )
