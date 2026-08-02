from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _validate_parallel(*arrays: Sequence[object]) -> None:
    if not arrays or len({len(array) for array in arrays}) != 1:
        raise ValueError("inputs must have the same non-zero length")


def _pava(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators algorithm."""
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index), value * weight, weight])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = [0.0] * len(values)
    for start, end, total, weight in blocks:
        fitted[int(start) : int(end) + 1] = [total / weight] * (
            int(end) - int(start) + 1
        )
    return fitted


@dataclass(slots=True)
class IsotonicCalibrator:
    """One-dimensional probability calibration with monotone PAVA."""

    thresholds: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    def fit(
        self,
        scores: Sequence[float],
        outcomes: Sequence[int | bool],
        sample_weight: Sequence[float] | None = None,
    ) -> IsotonicCalibrator:
        _validate_parallel(scores, outcomes)
        weights = list(sample_weight or [1.0] * len(scores))
        _validate_parallel(scores, weights)
        rows = sorted(
            (float(score), float(bool(outcome)), float(weight))
            for score, outcome, weight in zip(scores, outcomes, weights, strict=True)
        )
        grouped_scores: list[float] = []
        grouped_rates: list[float] = []
        grouped_weights: list[float] = []
        for score, outcome, weight in rows:
            if weight <= 0:
                raise ValueError("sample weights must be positive")
            if grouped_scores and score == grouped_scores[-1]:
                total = grouped_rates[-1] * grouped_weights[-1] + outcome * weight
                grouped_weights[-1] += weight
                grouped_rates[-1] = total / grouped_weights[-1]
            else:
                grouped_scores.append(score)
                grouped_rates.append(outcome)
                grouped_weights.append(weight)
        fitted = _pava(grouped_rates, grouped_weights)
        self.thresholds = tuple(grouped_scores)
        self.values = tuple(max(0.0, min(1.0, value)) for value in fitted)
        return self

    def predict_one(self, score: float) -> float:
        if not self.thresholds:
            raise RuntimeError("calibrator is not fitted")
        index = bisect.bisect_right(self.thresholds, float(score)) - 1
        return self.values[max(0, min(index, len(self.values) - 1))]

    def predict(self, scores: Iterable[float]) -> list[float]:
        return [self.predict_one(score) for score in scores]


@dataclass(slots=True)
class MonotoneBudgetCalibrator:
    """Calibrate P(success within B) while enforcing non-decrease in B."""

    budgets: tuple[float, ...] = ()
    probabilities: tuple[float, ...] = ()

    def fit(
        self,
        budgets: Sequence[float],
        outcomes: Sequence[int | bool],
        sample_weight: Sequence[float] | None = None,
    ) -> MonotoneBudgetCalibrator:
        fitted = IsotonicCalibrator().fit(budgets, outcomes, sample_weight)
        self.budgets = fitted.thresholds
        self.probabilities = fitted.values
        return self

    def predict_one(self, budget: float) -> float:
        if not self.budgets:
            raise RuntimeError("calibrator is not fitted")
        index = bisect.bisect_right(self.budgets, float(budget)) - 1
        return self.probabilities[max(0, min(index, len(self.probabilities) - 1))]

    def predict(self, budgets: Iterable[float]) -> list[float]:
        return [self.predict_one(budget) for budget in budgets]


@dataclass(slots=True)
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(
        self,
        probabilities: Sequence[float],
        outcomes: Sequence[int | bool],
        *,
        iterations: int = 300,
        learning_rate: float = 0.05,
    ) -> PlattCalibrator:
        _validate_parallel(probabilities, outcomes)
        logits = [
            math.log(max(1e-6, min(1 - 1e-6, value)))
            - math.log1p(-max(1e-6, min(1 - 1e-6, value)))
            for value in probabilities
        ]
        slope, intercept = 1.0, 0.0
        for _ in range(iterations):
            gradient_slope = 0.0
            gradient_intercept = 0.0
            for logit, outcome in zip(logits, outcomes, strict=True):
                prediction = 1.0 / (1.0 + math.exp(-max(-30, min(30, slope * logit + intercept))))
                error = prediction - float(bool(outcome))
                gradient_slope += error * logit
                gradient_intercept += error
            slope -= learning_rate * gradient_slope / len(logits)
            intercept -= learning_rate * gradient_intercept / len(logits)
        self.slope, self.intercept, self.fitted = slope, intercept, True
        return self

    def predict_one(self, probability: float) -> float:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        probability = max(1e-6, min(1 - 1e-6, probability))
        logit = math.log(probability) - math.log1p(-probability)
        value = max(-30, min(30, self.slope * logit + self.intercept))
        return 1.0 / (1.0 + math.exp(-value))


def ipcw_weights(
    censor_probabilities: Sequence[float],
    censored: Sequence[bool],
    *,
    clip: float = 10.0,
) -> list[float]:
    _validate_parallel(censor_probabilities, censored)
    result: list[float] = []
    for probability, is_censored in zip(
        censor_probabilities, censored, strict=True
    ):
        if not 0.0 <= probability < 1.0:
            raise ValueError("censor probabilities must be in [0, 1)")
        result.append(0.0 if is_censored else min(clip, 1.0 / (1.0 - probability)))
    return result

