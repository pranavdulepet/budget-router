from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Mapping, Sequence


def _equal_length(*values: Sequence[object]) -> None:
    if not values or len({len(value) for value in values}) != 1:
        raise ValueError("inputs must have the same length")


def success_at_budget(
    costs: Sequence[float], successes: Sequence[bool], budget: float
) -> float:
    _equal_length(costs, successes)
    if not costs:
        return 0.0
    return sum(
        bool(success) and cost <= budget
        for cost, success in zip(costs, successes, strict=True)
    ) / len(costs)


def normalized_success_auc(
    budgets: Sequence[float], success_rates: Sequence[float]
) -> float:
    _equal_length(budgets, success_rates)
    if len(budgets) < 2:
        raise ValueError("AUC needs at least two budget points")
    rows = sorted(zip(map(float, budgets), map(float, success_rates), strict=True))
    width = rows[-1][0] - rows[0][0]
    if width <= 0:
        raise ValueError("budget range must be positive")
    area = sum(
        (right_budget - left_budget) * (left_rate + right_rate) / 2
        for (left_budget, left_rate), (right_budget, right_rate) in zip(
            rows, rows[1:]
        )
    )
    return area / width


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    _equal_length(probabilities, outcomes)
    if not probabilities:
        raise ValueError("Brier score needs observations")
    return mean(
        (float(probability) - float(bool(outcome))) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    )


def integrated_brier_score(
    probability_curves: Sequence[Sequence[float]],
    outcome_curves: Sequence[Sequence[bool]],
) -> float:
    if (
        not probability_curves
        or len(probability_curves) != len(outcome_curves)
        or any(
            len(probabilities) != len(outcomes)
            for probabilities, outcomes in zip(
                probability_curves, outcome_curves, strict=True
            )
        )
    ):
        raise ValueError("probability and outcome curves must align")
    return mean(
        brier_score(probabilities, outcomes)
        for probabilities, outcomes in zip(
            probability_curves, outcome_curves, strict=True
        )
    )


def log_loss(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    _equal_length(probabilities, outcomes)
    if not probabilities:
        raise ValueError("log loss needs observations")
    result = 0.0
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        probability = max(1e-12, min(1 - 1e-12, float(probability)))
        result -= (
            math.log(probability) if outcome else math.log1p(-probability)
        )
    return result / len(probabilities)


def calibration_intercept_slope(
    probabilities: Sequence[float], outcomes: Sequence[bool]
) -> tuple[float, float]:
    """Return OLS diagnostic intercept/slope of outcomes on prediction logits."""
    _equal_length(probabilities, outcomes)
    logits = [
        math.log(max(1e-6, min(1 - 1e-6, value)))
        - math.log1p(-max(1e-6, min(1 - 1e-6, value)))
        for value in probabilities
    ]
    target = [float(bool(value)) for value in outcomes]
    x_mean, y_mean = mean(logits), mean(target)
    denominator = sum((value - x_mean) ** 2 for value in logits)
    slope = (
        sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(logits, target, strict=True)
        )
        / denominator
        if denominator
        else 0.0
    )
    return y_mean - slope * x_mean, slope


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    error_rate: float
    false_feasible_rate: float


def risk_coverage(
    probabilities: Sequence[float], outcomes: Sequence[bool]
) -> tuple[RiskCoveragePoint, ...]:
    _equal_length(probabilities, outcomes)
    thresholds = sorted(set(map(float, probabilities)), reverse=True)
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        selected = [
            bool(outcome)
            for probability, outcome in zip(probabilities, outcomes, strict=True)
            if probability >= threshold
        ]
        if not selected:
            continue
        errors = sum(not outcome for outcome in selected)
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=len(selected) / len(probabilities),
                error_rate=errors / len(selected),
                false_feasible_rate=errors / len(selected),
            )
        )
    return tuple(points)


def false_feasible_rate(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    threshold: float = 0.8,
) -> float:
    _equal_length(probabilities, outcomes)
    selected = [
        bool(outcome)
        for probability, outcome in zip(probabilities, outcomes, strict=True)
        if probability >= threshold
    ]
    return sum(not outcome for outcome in selected) / len(selected) if selected else 0.0


def cost_summary(costs: Sequence[float], successes: Sequence[bool]) -> dict[str, float]:
    _equal_length(costs, successes)
    if not costs:
        return {"mean": 0.0, "p95": 0.0, "cost_per_solved": math.inf}
    ordered = sorted(map(float, costs))
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    solved = sum(map(bool, successes))
    return {
        "mean": mean(ordered),
        "p95": p95,
        "cost_per_solved": sum(ordered) / solved if solved else math.inf,
    }


def repository_bootstrap_interval(
    values: Sequence[float],
    repositories: Sequence[str],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    _equal_length(values, repositories)
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, repository in zip(values, repositories, strict=True):
        grouped[repository].append(float(value))
    names = sorted(grouped)
    if not names:
        raise ValueError("bootstrap needs observations")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(names) for _ in names]
        draw = [value for name in selected for value in grouped[name]]
        draws.append(float(statistic(draw)))
    draws.sort()
    alpha = (1 - confidence) / 2
    low = draws[int(alpha * (samples - 1))]
    high = draws[int((1 - alpha) * (samples - 1))]
    return low, high


def aggregate_episode_metrics(
    *,
    costs: Sequence[float],
    successes: Sequence[bool],
    latencies_ms: Sequence[float],
    tokens: Sequence[int],
    switches: Sequence[int],
    budgets: Sequence[float],
) -> Mapping[str, float]:
    _equal_length(costs, successes, latencies_ms, tokens, switches)
    rates = [success_at_budget(costs, successes, budget) for budget in budgets]
    summary = cost_summary(costs, successes)
    return {
        "success_auc": normalized_success_auc(budgets, rates),
        "mean_cost_usd": summary["mean"],
        "p95_cost_usd": summary["p95"],
        "cost_per_solved_usd": summary["cost_per_solved"],
        "mean_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
        "mean_tokens": mean(tokens) if tokens else 0.0,
        "mean_switches": mean(switches) if switches else 0.0,
    }
