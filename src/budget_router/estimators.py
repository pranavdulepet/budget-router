from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .calibration import MonotoneBudgetCalibrator


@dataclass(slots=True)
class DirectBudgetEstimator:
    """Reference monotone budget-conditioned success estimator."""

    calibrator: MonotoneBudgetCalibrator = field(default_factory=MonotoneBudgetCalibrator)

    def fit(
        self,
        observed_costs: Sequence[float],
        succeeded: Sequence[bool],
        sample_weight: Sequence[float] | None = None,
    ) -> DirectBudgetEstimator:
        # Each trajectory contributes whether success was observed at its cost.
        self.calibrator.fit(observed_costs, succeeded, sample_weight)
        return self

    def predict(self, budgets: Iterable[float]) -> list[float]:
        return self.calibrator.predict(budgets)


@dataclass(slots=True)
class CompetingRiskBudgetEstimator:
    """Discrete cumulative incidence of success vs terminal failure over dollars."""

    bin_edges: tuple[float, ...] = ()
    success_cif: tuple[float, ...] = ()
    failure_cif: tuple[float, ...] = ()

    def fit(
        self,
        costs: Sequence[float],
        outcomes: Sequence[str],
        *,
        bins: Sequence[float],
    ) -> CompetingRiskBudgetEstimator:
        if not costs or len(costs) != len(outcomes):
            raise ValueError("costs and outcomes must have equal non-zero length")
        edges = tuple(sorted(set(float(value) for value in bins)))
        if not edges:
            raise ValueError("at least one bin is required")
        at_risk = len(costs)
        success_cif = 0.0
        failure_cif = 0.0
        survival = 1.0
        success_values: list[float] = []
        failure_values: list[float] = []
        previous_edge = float("-inf")
        for edge in edges:
            success_events = sum(
                previous_edge < cost <= edge and outcome == "success"
                for cost, outcome in zip(costs, outcomes, strict=True)
            )
            failure_events = sum(
                previous_edge < cost <= edge and outcome == "failure"
                for cost, outcome in zip(costs, outcomes, strict=True)
            )
            censored = sum(
                previous_edge < cost <= edge and outcome == "censored"
                for cost, outcome in zip(costs, outcomes, strict=True)
            )
            if at_risk > 0:
                success_cif += survival * success_events / at_risk
                failure_cif += survival * failure_events / at_risk
                survival *= 1.0 - (success_events + failure_events) / at_risk
            at_risk -= success_events + failure_events + censored
            success_values.append(success_cif)
            failure_values.append(failure_cif)
            previous_edge = edge
        self.bin_edges = edges
        self.success_cif = tuple(success_values)
        self.failure_cif = tuple(failure_values)
        return self

    def predict(self, budget: float) -> tuple[float, float]:
        if not self.bin_edges:
            raise RuntimeError("estimator is not fitted")
        index = bisect.bisect_right(self.bin_edges, float(budget)) - 1
        if index < 0:
            return 0.0, 0.0
        index = min(index, len(self.bin_edges) - 1)
        return self.success_cif[index], self.failure_cif[index]


@dataclass(slots=True)
class ActionValueEstimator:
    """Calibrated empirical success and cost-to-go head per action."""

    smoothing: float = 1.0
    _success: dict[str, float] = field(default_factory=dict)
    _cost: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        actions: Sequence[str],
        outcomes: Sequence[bool],
        costs_to_go: Sequence[float],
    ) -> ActionValueEstimator:
        if not actions or len({len(actions), len(outcomes), len(costs_to_go)}) != 1:
            raise ValueError("training arrays must have equal non-zero length")
        successes: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        costs: dict[str, float] = defaultdict(float)
        global_rate = sum(map(bool, outcomes)) / len(outcomes)
        global_cost = sum(costs_to_go) / len(costs_to_go)
        for action, outcome, cost in zip(actions, outcomes, costs_to_go, strict=True):
            successes[action] += float(bool(outcome))
            counts[action] += 1
            costs[action] += float(cost)
        for action, count in counts.items():
            self._success[action] = (
                successes[action] + self.smoothing * global_rate
            ) / (count + self.smoothing)
            self._cost[action] = (
                costs[action] + self.smoothing * global_cost
            ) / (count + self.smoothing)
        return self

    def predict(self, action: str) -> tuple[float, float]:
        if action not in self._success:
            raise KeyError(f"unknown action {action!r}")
        return self._success[action], self._cost[action]

