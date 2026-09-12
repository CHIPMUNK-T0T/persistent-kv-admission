"""Deliberately small linear predictors for future-reuse ranking.

The goal is to measure how much information a feature group carries, not to
build a strong model. Only L2-regularised logistic regression is used, fitted
with Newton steps on standardised features.

Training rows use case-control sampling: every positive is kept and negatives
are subsampled. Under case-control sampling the fitted intercept is biased but
the coefficient vector, and therefore the induced ranking, is not. All reported
metrics are ranking metrics, so no intercept correction is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticRanker:
    indices: tuple[int, ...]
    l2: float = 1.0
    max_iterations: int = 40
    tolerance: float = 1e-7
    # With standardize=False the penalty acts on raw-unit coefficients, so the
    # fit is a genuinely different model and not a reparameterisation.
    standardize: bool = True
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    converged: bool = False
    iterations: int = 0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticRanker":
        selected = features[:, self.indices].astype(float)
        labels = labels.astype(float)
        if self.standardize:
            self.mean = selected.mean(axis=0)
            scale = selected.std(axis=0)
            scale[scale < 1e-12] = 1.0
        else:
            self.mean = np.zeros(selected.shape[1])
            scale = np.ones(selected.shape[1])
        self.scale = scale
        design = np.column_stack(
            ((selected - self.mean) / scale, np.ones(len(selected)))
        )
        weights = np.zeros(design.shape[1])
        # Scale the penalty with the sample count so that l2 keeps the same
        # meaning as the training set grows. Without this the penalty vanishes
        # on large samples and collinear feature pairs fit large opposing
        # coefficients that do not transfer to a later window.
        penalty = np.full(design.shape[1], self.l2 * len(design))
        penalty[-1] = 0.0  # never penalise the intercept
        for iteration in range(1, self.max_iterations + 1):
            logits = np.clip(design @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (probabilities - labels) + penalty * weights
            variance = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            hessian = design.T @ (design * variance[:, None]) + np.diag(penalty)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            weights = weights - step
            self.iterations = iteration
            if np.max(np.abs(step)) < self.tolerance:
                self.converged = True
                break
        self.coefficients = weights[:-1]
        self.intercept = float(weights[-1])
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValueError("ranker is not fitted")
        selected = features[:, self.indices].astype(float)
        return ((selected - self.mean) / self.scale) @ self.coefficients + self.intercept

    def score_row(self, row: list[float]) -> float:
        """Score one raw feature vector. Used on the replay eviction path."""
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValueError("ranker is not fitted")
        total = self.intercept
        coefficients = self.coefficients
        mean = self.mean
        scale = self.scale
        for position, index in enumerate(self.indices):
            total += coefficients[position] * (row[index] - mean[position]) / scale[position]
        return total

    def standardized_coefficients(self, names: tuple[str, ...]) -> list[tuple[str, float]]:
        if self.coefficients is None:
            raise ValueError("ranker is not fitted")
        return [
            (names[index], float(value))
            for index, value in zip(self.indices, self.coefficients)
        ]


def case_control_sample(
    features: np.ndarray,
    labels: np.ndarray,
    negatives_per_positive: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every positive row and subsample negatives.

    Returns the original arrays unchanged when there is nothing to drop.
    """
    positive_mask = labels > 0
    positive_count = int(positive_mask.sum())
    negative_indices = np.flatnonzero(~positive_mask)
    if positive_count == 0 or len(negative_indices) == 0:
        return features, labels
    keep = int(min(len(negative_indices), round(negatives_per_positive * positive_count)))
    if keep >= len(negative_indices):
        return features, labels
    chosen = rng.choice(negative_indices, size=keep, replace=False)
    order = np.sort(np.concatenate((np.flatnonzero(positive_mask), chosen)))
    return features[order], labels[order]


@dataclass
class RidgeRanker:
    """L2 linear regression on standardised features, closed form.

    Used for graded targets (log reuse count, negative log next-use time) so
    that a target change is only a target change: the feature set, the
    standardisation, and the penalty scale match `LogisticRanker`.
    """

    indices: tuple[int, ...]
    l2: float = 1.0
    standardize: bool = True
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    converged: bool = True
    iterations: int = 1

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeRanker":
        selected = features[:, self.indices].astype(float)
        targets = targets.astype(float)
        if self.standardize:
            self.mean = selected.mean(axis=0)
            scale = selected.std(axis=0)
            scale[scale < 1e-12] = 1.0
        else:
            self.mean = np.zeros(selected.shape[1])
            scale = np.ones(selected.shape[1])
        self.scale = scale
        design = (selected - self.mean) / scale
        centred_target = targets - targets.mean()
        penalty = self.l2 * len(design) * np.eye(design.shape[1])
        gram = design.T @ design + penalty
        self.coefficients = np.linalg.solve(gram, design.T @ centred_target)
        self.intercept = float(targets.mean())
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValueError("ranker is not fitted")
        selected = features[:, self.indices].astype(float)
        return ((selected - self.mean) / self.scale) @ self.coefficients + self.intercept

    def score_row(self, row: list[float]) -> float:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValueError("ranker is not fitted")
        total = self.intercept
        for position, index in enumerate(self.indices):
            total += self.coefficients[position] * (row[index] - self.mean[position]) / self.scale[position]
        return total

    def standardized_coefficients(self, names: tuple[str, ...]) -> list[tuple[str, float]]:
        if self.coefficients is None:
            raise ValueError("ranker is not fitted")
        return [(names[index], float(value)) for index, value in zip(self.indices, self.coefficients)]
