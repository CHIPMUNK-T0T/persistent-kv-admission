"""Online, workload-adaptive retention scoring.

The research question this serves is not "which signal is best". It is whether
a serving runtime can observe its own reuse behaviour and adapt what it keeps,
without per-workload tuning.

So this scorer:

* starts from zero weights, which makes it fall back to recency order until it
  has learned anything;
* builds its own labelled examples from observations that have already matured,
  so no future information is used;
* updates with AdaGrad, which removes the per-workload learning-rate choice;
* standardises features against the live state population, so the same
  hyperparameters transfer across traces with different scales.

Every hyperparameter here is fixed across all workloads. None is tuned per
trace. That restriction is the point of the experiment, not an oversight.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .temporal import FEATURE_NAMES, PopulationNormalizer, TemporalHistory
from .trace import Request, Trace


class OnlineAdaptiveScorer:
    time_varying = True

    def __init__(
        self,
        trace: Trace,
        horizon_seconds: float = 600.0,
        samples_per_group: int = 64,
        l2: float = 1e-4,
        learning_rate: float = 0.5,
        refresh_every: int = 16,
        normalizer_sample: int = 512,
        indices: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> None:
        self.history = TemporalHistory(trace)
        self.dimension = len(FEATURE_NAMES)
        self.indices = tuple(range(self.dimension)) if indices is None else tuple(indices)
        self.normalizer = PopulationNormalizer(self.dimension)
        self.horizon_ms = horizon_seconds * 1000.0
        self.samples_per_group = samples_per_group
        self.l2 = l2
        self.learning_rate = learning_rate
        self.refresh_every = refresh_every
        self.normalizer_sample = normalizer_sample
        self.rng = np.random.default_rng(seed)

        self.weights = np.zeros(len(self.indices))
        self.bias = 0.0
        self.gradient_squares = np.full(len(self.indices), 1e-8)
        self.bias_gradient_square = 1e-8

        self.pending: dict[str, deque[tuple[float, np.ndarray]]] = defaultdict(deque)
        self.maturity: deque[tuple[float, str]] = deque()
        self.positive_examples = 0
        self.negative_examples = 0
        self.updates = 0
        self.groups_seen = 0
        self.cache = None

    def attach(self, cache) -> None:
        """Receive the cache whose retained set this scorer ranks.

        Training examples are drawn from the retained set rather than from every
        state ever observed. The eviction decision only ever ranks retained
        states, and the two populations are very different: most observed states
        are long-dead single-use blocks, so a model fitted on them learns to
        separate dead from live, which carries almost no information among the
        live states the cache actually holds.
        """
        self.cache = cache

    # -- learning ---------------------------------------------------------
    def _update(self, row: np.ndarray, label: float) -> None:
        # Balance the two classes with the positive rate observed so far, so a
        # rare-positive workload does not collapse the ranking onto one class.
        positives = self.positive_examples + 1.0
        negatives = self.negative_examples + 1.0
        weight = (negatives / positives) if label > 0 else 1.0
        logit = float(row @ self.weights) + self.bias
        probability = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
        error = (probability - label) * weight
        gradient = error * row + self.l2 * self.weights
        self.gradient_squares += gradient * gradient
        self.weights -= self.learning_rate * gradient / np.sqrt(self.gradient_squares)
        self.bias_gradient_square += error * error
        self.bias -= self.learning_rate * error / math.sqrt(self.bias_gradient_square)
        self.updates += 1
        if label > 0:
            self.positive_examples += 1
        else:
            self.negative_examples += 1

    def _resolve_positive(self, state_id: str, now_ms: float) -> None:
        queue = self.pending.get(state_id)
        if not queue:
            return
        while queue:
            created_ms, row = queue[0]
            if created_ms >= now_ms:
                break
            queue.popleft()
            self._update(row, 1.0)
        if not queue:
            self.pending.pop(state_id, None)

    def _expire(self, now_ms: float) -> None:
        while self.maturity and self.maturity[0][0] <= now_ms:
            _, state_id = self.maturity.popleft()
            queue = self.pending.get(state_id)
            if not queue:
                continue
            created_ms, row = queue[0]
            if created_ms + self.horizon_ms > now_ms:
                continue
            queue.popleft()
            self._update(row, 0.0)
            if not queue:
                self.pending.pop(state_id, None)

    # -- scorer protocol --------------------------------------------------
    def observe(self, requests: list[Request], timestamp_ms: float) -> None:
        touched = {state_id for request in requests for state_id in request.hash_ids}
        for state_id in touched:
            self._resolve_positive(state_id, timestamp_ms)
        self._expire(timestamp_ms)

        self.history.observe_requests(requests, timestamp_ms)

        observed = self.history.observed_state_ids()
        if self.groups_seen % self.refresh_every == 0 and len(observed) >= 2:
            picks = (
                self.rng.choice(len(observed), size=self.normalizer_sample, replace=False)
                if len(observed) > self.normalizer_sample
                else np.arange(len(observed))
            )
            sample = [observed[int(position)] for position in picks]
            self.normalizer.refresh(self.history.feature_matrix(sample, timestamp_ms))

        population = observed
        if self.cache is not None and self.cache.cached:
            population = list(self.cache.cached)
        if population:
            count = min(self.samples_per_group, len(population))
            picks = self.rng.choice(len(population), size=count, replace=False)
            for position in picks:
                state_id = population[int(position)]
                row = np.asarray(
                    self.normalizer.apply_row(
                        self.history.feature_vector(state_id, timestamp_ms)
                    )
                )[list(self.indices)]
                self.pending[state_id].append((timestamp_ms, row))
                self.maturity.append((timestamp_ms + self.horizon_ms, state_id))
        self.groups_seen += 1

    def score(self, state_id: str, timestamp_ms: float) -> float:
        row = self.normalizer.apply_row(self.history.feature_vector(state_id, timestamp_ms))
        total = self.bias
        weights = self.weights
        for position, index in enumerate(self.indices):
            total += weights[position] * row[index]
        return total

    def learned_weights(self) -> list[tuple[str, float]]:
        return [
            (FEATURE_NAMES[index], float(self.weights[position]))
            for position, index in enumerate(self.indices)
        ]
