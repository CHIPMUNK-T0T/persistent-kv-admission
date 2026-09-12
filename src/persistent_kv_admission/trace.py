"""Mooncake trace loading and cumulative-prefix tree reconstruction."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Request:
    order: int
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[str, ...]


@dataclass(frozen=True)
class StateMeta:
    state_id: str
    parent_id: str | None
    depth: int
    prefix_tokens: int
    block_tokens: int


@dataclass
class Trace:
    name: str
    requests: list[Request]
    states: dict[str, StateMeta]
    occurrences_ms: dict[str, list[float]]
    children: dict[str, set[str]]
    terminal_branches: dict[str, set[str]]
    block_size: int

    @property
    def start_ms(self) -> float:
        return self.requests[0].timestamp_ms if self.requests else 0.0

    @property
    def end_ms(self) -> float:
        return self.requests[-1].timestamp_ms if self.requests else 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def timestamp_groups(self) -> Iterator[tuple[float, list[Request]]]:
        start = 0
        while start < len(self.requests):
            timestamp = self.requests[start].timestamp_ms
            end = start + 1
            while end < len(self.requests) and self.requests[end].timestamp_ms == timestamp:
                end += 1
            yield timestamp, self.requests[start:end]
            start = end


def _canonical_hash(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"hash id must be an integer or string, got {value!r}")
    return f"{type(value).__name__}:{value}"


def load_mooncake_trace(path: str | Path, block_size: int = 512) -> Trace:
    """Load and strictly validate a Mooncake JSONL trace.

    Mooncake hash IDs are cumulative prefix hashes. Consequently, the hash at
    depth d identifies the full prefix through block d, while the materialized
    KV increment represented by the node is only that node's final block.
    """

    path = Path(path)
    requests: list[Request] = []
    states: dict[str, StateMeta] = {}
    occurrences: dict[str, list[float]] = defaultdict(list)
    children: dict[str, set[str]] = defaultdict(set)
    terminal_branches: dict[str, set[str]] = defaultdict(set)
    previous_timestamp = -math.inf

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            missing = {"timestamp", "input_length", "output_length", "hash_ids"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            timestamp = record["timestamp"]
            input_length = record["input_length"]
            output_length = record["output_length"]
            raw_hashes = record["hash_ids"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(f"{path}:{line_number}: timestamp must be numeric")
            if timestamp < previous_timestamp:
                raise ValueError(f"{path}:{line_number}: timestamps are not nondecreasing")
            if not isinstance(input_length, int) or input_length <= 0:
                raise ValueError(f"{path}:{line_number}: input_length must be positive")
            if not isinstance(output_length, int) or output_length < 0:
                raise ValueError(f"{path}:{line_number}: output_length must be nonnegative")
            if not isinstance(raw_hashes, list):
                raise ValueError(f"{path}:{line_number}: hash_ids must be a list")
            expected_blocks = math.ceil(input_length / block_size)
            if len(raw_hashes) != expected_blocks:
                raise ValueError(
                    f"{path}:{line_number}: got {len(raw_hashes)} hashes, expected "
                    f"ceil({input_length}/{block_size})={expected_blocks}"
                )

            hash_ids = tuple(_canonical_hash(value) for value in raw_hashes)
            request = Request(
                order=len(requests),
                timestamp_ms=float(timestamp),
                input_length=input_length,
                output_length=output_length,
                hash_ids=hash_ids,
            )
            requests.append(request)
            previous_timestamp = float(timestamp)
            terminal_id = hash_ids[-1]

            for index, state_id in enumerate(hash_ids):
                parent_id = hash_ids[index - 1] if index else None
                prefix_tokens = min(input_length, (index + 1) * block_size)
                block_tokens = prefix_tokens - index * block_size
                meta = StateMeta(
                    state_id=state_id,
                    parent_id=parent_id,
                    depth=index + 1,
                    prefix_tokens=prefix_tokens,
                    block_tokens=block_tokens,
                )
                existing = states.get(state_id)
                if existing is not None and existing != meta:
                    raise ValueError(
                        f"{path}:{line_number}: hash {state_id} has inconsistent chain metadata: "
                        f"{existing!r} vs {meta!r}"
                    )
                states[state_id] = meta
                occurrences[state_id].append(float(timestamp))
                terminal_branches[state_id].add(terminal_id)
                if parent_id is not None:
                    children[parent_id].add(state_id)

    if not requests:
        raise ValueError(f"{path}: trace is empty")
    return Trace(
        name=path.stem,
        requests=requests,
        states=states,
        occurrences_ms=dict(occurrences),
        children=dict(children),
        terminal_branches=dict(terminal_branches),
        block_size=block_size,
    )


def unique_timestamps(trace: Trace) -> list[float]:
    return [timestamp for timestamp, _ in trace.timestamp_groups()]


def iter_state_ids(requests: Iterable[Request]) -> Iterator[str]:
    for request in requests:
        yield from request.hash_ids
