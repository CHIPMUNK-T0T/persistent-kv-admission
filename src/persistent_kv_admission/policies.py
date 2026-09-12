"""Policy interfaces for persistent KV admission / retention experiments."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class KVState:
    state_id: str
    size_bytes: int
    prefix_tokens: int
    reuse_count: int = 0
    fan_out: int = 0
    age: float = 0.0


class AdmissionPolicy(Protocol):
    def score(self, state: KVState) -> float:
        """Return a retention priority score. Higher means more valuable."""
        ...


class ValueAwarePolicy:
    """Placeholder for the first non-semantic value-aware baseline."""

    def score(self, state: KVState) -> float:
        # Deliberately simple initial form; replace after trace characterization.
        recompute_proxy = float(state.prefix_tokens)
        structural_signal = 1.0 + float(state.fan_out)
        temporal_signal = 1.0 / (1.0 + max(state.age, 0.0))
        expected_reuse_proxy = (1.0 + state.reuse_count) * structural_signal * temporal_signal
        return (expected_reuse_proxy * recompute_proxy) / max(state.size_bytes, 1)
