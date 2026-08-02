from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

from .pricing import PriceSnapshot
from .types import GoalContext, TokenUsage


@dataclass(frozen=True, slots=True)
class MatrixOutcome:
    task_id: str
    repository: str
    model: str
    seed: int
    cost_usd: Decimal
    success: bool


class InitialModelRouter:
    def select(
        self,
        goal: GoalContext,
        *,
        budget_usd: Decimal,
        prices: PriceSnapshot,
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FixedModelRouter(InitialModelRouter):
    model: str

    def select(
        self,
        goal: GoalContext,
        *,
        budget_usd: Decimal,
        prices: PriceSnapshot,
    ) -> str:
        if self.model not in goal.allowed_models:
            raise ValueError(f"fixed model {self.model!r} is not allowed")
        return self.model


@dataclass(frozen=True, slots=True)
class CheapestModelRouter(InitialModelRouter):
    reference_input_tokens: int = 8_000
    reference_output_tokens: int = 2_000

    def select(
        self,
        goal: GoalContext,
        *,
        budget_usd: Decimal,
        prices: PriceSnapshot,
    ) -> str:
        return min(
            goal.allowed_models,
            key=lambda model: (
                prices.price_for(model).conservative_cost(
                    TokenUsage(
                        input_tokens=self.reference_input_tokens,
                        output_tokens=self.reference_output_tokens,
                    )
                ),
                model,
            ),
        )


@dataclass(frozen=True, slots=True)
class GlobalBestRouter(InitialModelRouter):
    validation_success_rates: Mapping[str, float]

    def select(
        self,
        goal: GoalContext,
        *,
        budget_usd: Decimal,
        prices: PriceSnapshot,
    ) -> str:
        return max(
            goal.allowed_models,
            key=lambda model: (self.validation_success_rates.get(model, 0.0), model),
        )


class FrequencyMatchedRandomRouter(InitialModelRouter):
    def __init__(self, frequencies: Mapping[str, float], *, seed: int = 0) -> None:
        if not frequencies or any(value < 0 for value in frequencies.values()):
            raise ValueError("frequencies must be non-negative and non-empty")
        self.frequencies = dict(frequencies)
        self.rng = random.Random(seed)

    def select(
        self,
        goal: GoalContext,
        *,
        budget_usd: Decimal,
        prices: PriceSnapshot,
    ) -> str:
        models = list(goal.allowed_models)
        weights = [self.frequencies.get(model, 0.0) for model in models]
        if not any(weights):
            weights = [1.0] * len(models)
        return self.rng.choices(models, weights=weights, k=1)[0]


@dataclass(slots=True)
class HindsightTaskOracle:
    """Upper-bound control. Never use it as a deployable policy."""

    outcomes: Sequence[MatrixOutcome]

    def select(self, task_id: str, budget_usd: Decimal) -> str | None:
        feasible = [
            row
            for row in self.outcomes
            if row.task_id == task_id and row.success and row.cost_usd <= budget_usd
        ]
        return (
            min(feasible, key=lambda row: (row.cost_usd, row.model)).model
            if feasible
            else None
        )


@dataclass(slots=True)
class BudgetOracle:
    """Training-matrix success-at-budget oracle, aggregated without test labels."""

    outcomes: Sequence[MatrixOutcome]

    def success_rates(self, budget_usd: Decimal) -> Mapping[str, float]:
        totals: dict[str, int] = defaultdict(int)
        solved: dict[str, int] = defaultdict(int)
        for row in self.outcomes:
            totals[row.model] += 1
            solved[row.model] += int(row.success and row.cost_usd <= budget_usd)
        return {
            model: solved[model] / total for model, total in totals.items() if total
        }

    def select(self, allowed_models: Sequence[str], budget_usd: Decimal) -> str:
        rates = self.success_rates(budget_usd)
        return max(allowed_models, key=lambda model: (rates.get(model, 0.0), model))


METHOD_REGISTRY: Mapping[str, tuple[str, ...]] = {
    "controls": (
        "fixed_each",
        "cheapest",
        "global_best",
        "frequency_matched_random",
        "hindsight_task_oracle",
        "budget_oracle",
    ),
    "static": (
        "rules_tags",
        "embedding_knn",
        "calibrated_logistic",
        "calibrated_gbm",
        "matrix_factorization",
        "zero_few_shot_llm",
        "tinker_lora",
    ),
    "sequential": (
        "cheap_first_cascade",
        "explore_continue_escalate",
        "factorized_temporal_value",
        "distributional_offline_if_ess_gate",
    ),
    "outer": ("linucb", "thompson_sampling", "retrieval_memory"),
}

