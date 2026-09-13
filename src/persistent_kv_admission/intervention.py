"""Collectors and paired outcomes for the Phase 1 arrival intervention."""

from __future__ import annotations

from .attribution import AttributionCollector
from .trace import Trace

PROTECTION_EVENTS = (
    "protected_offer",
    "protected_overflow_offer",
    "first_round_override",
    "oversized_ineligible",
)


class Phase1Collector:
    """Compose unchanged Phase 0.98b attribution with Phase 1 diagnostics."""

    def __init__(self, trace: Trace, measure_from_ms: float | None = None) -> None:
        self.trace = trace
        self.measure_from_ms = measure_from_ms
        self.attribution = AttributionCollector(trace, measure_from_ms=measure_from_ms)
        self.events = {event: 0 for event in PROTECTION_EVENTS}
        self.events_window = {event: 0 for event in PROTECTION_EVENTS}
        self.request_index = 0
        self.outcomes: list[tuple[int, float, int, int, str, int, int, int]] = []

    def attach(self, store) -> None:
        self.attribution.attach(store)

    def __call__(
        self,
        state_id: str,
        kind: str,
        timestamp_ms: float,
        group_index: int,
        decision_index: int,
    ) -> None:
        self.attribution(state_id, kind, timestamp_ms, group_index, decision_index)

    def on_protection(self, event: str, timestamp_ms: float, group_index: int) -> None:
        if event not in self.events:
            raise ValueError(f"unknown protection event {event!r}")
        self.events[event] += 1
        if self.measure_from_ms is None or timestamp_ms >= self.measure_from_ms:
            self.events_window[event] += 1

    def on_request(
        self,
        ids,
        prefix,
        l2_hits,
        present,
        timestamp_ms,
        group_index,
        measured,
    ) -> None:
        if self.request_index >= len(self.trace.requests):
            raise AssertionError("request hook observed more requests than the trace")
        request = self.trace.requests[self.request_index]
        self.request_index += 1
        if request.timestamp_ms != timestamp_ms or request.hash_ids != tuple(ids):
            raise AssertionError(
                f"request hook order diverged at trace request {request.order}"
            )
        expected_measured = (
            self.measure_from_ms is None or timestamp_ms >= self.measure_from_ms
        )
        if bool(measured) != expected_measured:
            raise AssertionError(
                f"request window flag diverged at trace request {request.order}"
            )
        self.attribution.on_request(
            ids, prefix, l2_hits, present, timestamp_ms, group_index, measured
        )
        if not measured:
            return
        states = self.trace.states
        requested = sum(states[state_id].block_tokens for state_id in ids)
        if requested != request.input_length:
            raise AssertionError(
                f"request token partition diverged at trace request {request.order}"
            )
        l1_avoided = sum(states[ids[index]].block_tokens for index in range(prefix))
        l2_avoided = sum(states[ids[index]].block_tokens for index in l2_hits)
        self.outcomes.append(
            (
                request.order,
                timestamp_ms,
                requested,
                len(ids),
                ids[-1],
                prefix,
                l1_avoided,
                l1_avoided + l2_avoided,
            )
        )

    def finish(self, result) -> None:
        if self.request_index != len(self.trace.requests):
            raise AssertionError(
                f"request hook saw {self.request_index} of {len(self.trace.requests)} requests"
            )
        self.attribution.check_against(result)
        requested = sum(outcome[2] for outcome in self.outcomes)
        l1_avoided = sum(outcome[6] for outcome in self.outcomes)
        avoided = sum(outcome[7] for outcome in self.outcomes)
        checks = (
            ("requested_tokens", requested, result.requested_tokens),
            ("l1_avoided_tokens", l1_avoided, result.l1_avoided_tokens),
            ("avoided_prefill_tokens", avoided, result.avoided_prefill_tokens),
            ("measured_requests", len(self.outcomes), result.measured_requests),
        )
        for name, mine, replay in checks:
            if mine != replay:
                raise AssertionError(f"{name}: request vector {mine} != replay {replay}")

    def protection_row(self) -> dict[str, int]:
        row: dict[str, int] = {}
        for event in PROTECTION_EVENTS:
            row[f"{event}_count"] = self.events[event]
            row[f"{event}_window_count"] = self.events_window[event]
        return row


def paired_request_summary(
    left: list[tuple[int, float, int, int, str, int, int, int]],
    right: list[tuple[int, float, int, int, str, int, int, int]],
) -> dict[str, int | float]:
    """Summarize left-minus-right request outcomes after identity checks."""

    if len(left) != len(right):
        raise AssertionError(f"paired request lengths differ: {len(left)} != {len(right)}")
    saved_tokens = 0
    lost_tokens = 0
    saved_requests = 0
    lost_requests = 0
    unchanged_requests = 0
    requested_tokens = 0
    left_total = 0
    right_total = 0
    for left_outcome, right_outcome in zip(left, right):
        if left_outcome[:-1] != right_outcome[:-1]:
            raise AssertionError(
                f"paired request identity or L1 prefix differs at {left_outcome[0]}"
            )
        requested_tokens += left_outcome[2]
        left_total += left_outcome[7]
        right_total += right_outcome[7]
        delta = left_outcome[7] - right_outcome[7]
        if delta > 0:
            saved_tokens += delta
            saved_requests += 1
        elif delta < 0:
            lost_tokens -= delta
            lost_requests += 1
        else:
            unchanged_requests += 1
    net = left_total - right_total
    if net != saved_tokens - lost_tokens:
        raise AssertionError(
            f"paired net identity failed: {net} != {saved_tokens} - {lost_tokens}"
        )
    return {
        "measured_requests": len(left),
        "requested_tokens": requested_tokens,
        "left_avoided_tokens": left_total,
        "right_avoided_tokens": right_total,
        "saved_requests": saved_requests,
        "lost_requests": lost_requests,
        "unchanged_requests": unchanged_requests,
        "differing_requests": saved_requests + lost_requests,
        "saved_tokens": saved_tokens,
        "lost_tokens": lost_tokens,
        "net_avoided_tokens": net,
        "net_fraction_of_input": net / max(requested_tokens, 1),
        "net_input_percentage_points": 100.0 * net / max(requested_tokens, 1),
    }
