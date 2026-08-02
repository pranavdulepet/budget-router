from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Protocol

from .ledger import BudgetLedger
from .operations import eligible_models, eligible_operations
from .types import (
    ActionEstimate,
    FeasibilityPoint,
    GoalContext,
    Operation,
    PARALLEL_SHARES,
    ReasonCode,
    RouterAction,
    RouterDecision,
    RouterState,
)


@dataclass(frozen=True, slots=True)
class ModelEstimate:
    success_probability: float
    uncertainty: float
    cost_multiplier: float = 1.0


class ModelHead(Protocol):
    """Separate model-selection head; implementations may be learned."""

    def estimate(self, goal: GoalContext, state: RouterState, model: str) -> ModelEstimate: ...


class OperationHead(Protocol):
    """Separate operation-selection head; implementations may be learned."""

    def multiplier(self, goal: GoalContext, state: RouterState, operation: Operation) -> float: ...


@dataclass(slots=True)
class LookupModelHead:
    """Calibrated per-model priors with optional task-feature overrides."""

    success_priors: Mapping[str, float] = field(default_factory=dict)
    default_success: float = 0.5
    uncertainty: float = 0.25

    def estimate(self, goal: GoalContext, state: RouterState, model: str) -> ModelEstimate:
        base = float(self.success_priors.get(model, self.default_success))
        progress = min(0.08, 0.01 * sum(event.success for event in state.tool_events))
        failures = 0.025 * sum(not event.success for event in state.tool_events[-4:])
        probability = min(0.99, max(0.01, base + progress - failures))
        uncertainty = min(1.0, max(0.02, self.uncertainty * (0.98**state.turn)))
        return ModelEstimate(probability, uncertainty)


@dataclass(slots=True)
class HeuristicOperationHead:
    multipliers: Mapping[Operation, float] = field(
        default_factory=lambda: {
            Operation.CONTINUE: 1.0,
            Operation.REFLECT: 0.94,
            Operation.REPLAN: 0.98,
            Operation.VERIFY: 1.03,
            Operation.RESTART: 0.78,
            Operation.PARALLEL: 1.12,
            Operation.STOP: 0.0,
        }
    )

    def multiplier(self, goal: GoalContext, state: RouterState, operation: Operation) -> float:
        value = float(self.multipliers.get(operation, 1.0))
        recent_failures = sum(not event.success for event in state.tool_events[-3:])
        if operation is Operation.REFLECT and recent_failures >= 2:
            value += 0.12
        if operation is Operation.REPLAN and recent_failures >= 2:
            value += 0.10
        if operation is Operation.VERIFY and state.workspace.files_changed:
            value += 0.08
        if operation is Operation.RESTART and state.turn < 8:
            value -= 0.15
        return value


@dataclass(frozen=True, slots=True)
class RouterConfig:
    policy_version: str = "factorized-heuristic-v1"
    max_output_tokens: int = 8_000
    min_output_tokens: int = 256
    switch_hysteresis_turns: int = 2
    switch_penalty_usd: Decimal = Decimal("0.002")
    feasibility_threshold: float = 0.80
    stop_value_threshold: float = 0.02
    ablation: str = "joint"  # joint, model_only, operation_only

    def __post_init__(self) -> None:
        if self.ablation not in {"joint", "model_only", "operation_only"}:
            raise ValueError("ablation must be joint, model_only, or operation_only")
        if not 0.0 <= self.feasibility_threshold <= 1.0:
            raise ValueError("feasibility_threshold must be in [0, 1]")


class FactorizedRouterPolicy:
    """Factorized heads with a joint, hard-budget-constrained decision layer."""

    def __init__(
        self,
        model_head: ModelHead | None = None,
        operation_head: OperationHead | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.model_head = model_head or LookupModelHead()
        self.operation_head = operation_head or HeuristicOperationHead()
        self.config = config or RouterConfig()

    def _output_allowance(self, ledger: BudgetLedger, model: str, input_tokens: int) -> int:
        price = ledger.prices.price_for(model)
        input_quote = ledger.quote_tokens(model, input_tokens, 0)
        available_for_output = max(Decimal("0"), ledger.available_usd - input_quote)
        if price.output_per_million_usd == 0:
            return self.config.max_output_tokens
        affordable = int(
            available_for_output * Decimal("1000000") / price.output_per_million_usd
        )
        return max(0, min(self.config.max_output_tokens, affordable))

    def _candidate_actions(
        self, goal: GoalContext, state: RouterState, ledger: BudgetLedger
    ) -> tuple[list[RouterAction], dict[str, str], bool]:
        operations = eligible_operations(state, goal)
        models, hysteresis_masked = eligible_models(
            state,
            goal,
            switch_hysteresis_turns=self.config.switch_hysteresis_turns,
        )
        actions: list[RouterAction] = []
        masked: dict[str, str] = {}

        for operation in operations:
            if operation is Operation.STOP:
                actions.append(RouterAction(Operation.STOP, max_output_tokens=0))
                continue
            if operation is Operation.VERIFY:
                actions.append(RouterAction(Operation.VERIFY, max_output_tokens=0))
                for model in models:
                    allowance = self._output_allowance(
                        ledger, model, state.estimated_input_tokens
                    )
                    if allowance < self.config.min_output_tokens:
                        masked[f"{operation.value}:{model}"] = (
                            "insufficient_output_allowance"
                        )
                        continue
                    interpreted = RouterAction(
                        Operation.VERIFY,
                        model=model,
                        max_output_tokens=allowance,
                    )
                    if (
                        ledger.quote_action(
                            interpreted, state.estimated_input_tokens
                        )
                        > ledger.available_usd
                    ):
                        masked[interpreted.key] = "hard_budget_cap"
                    else:
                        actions.append(interpreted)
                continue
            for model in models:
                allowance = self._output_allowance(
                    ledger, model, state.estimated_input_tokens
                )
                if allowance < self.config.min_output_tokens:
                    masked[f"{operation.value}:{model}"] = "insufficient_output_allowance"
                    continue
                if operation is Operation.PARALLEL:
                    for share in PARALLEL_SHARES:
                        action = RouterAction(
                            operation,
                            model=model,
                            branch_share=share,
                            max_output_tokens=allowance,
                        )
                        reserve = ledger.available_usd * Decimal("2") * share
                        one_turn = ledger.quote_action(action, state.estimated_input_tokens)
                        if one_turn > ledger.available_usd * share:
                            masked[action.key] = "branch_subbudget_below_one_turn_quote"
                        elif reserve > ledger.available_usd:
                            masked[action.key] = "parallel_reserve_exceeds_budget"
                        else:
                            actions.append(action)
                else:
                    action = RouterAction(operation, model=model, max_output_tokens=allowance)
                    if ledger.quote_action(action, state.estimated_input_tokens) > ledger.available_usd:
                        masked[action.key] = "hard_budget_cap"
                    else:
                        actions.append(action)
        return actions, masked, hysteresis_masked

    def _estimate(
        self,
        goal: GoalContext,
        state: RouterState,
        ledger: BudgetLedger,
        action: RouterAction,
    ) -> ActionEstimate:
        if action.operation is Operation.STOP:
            success = 1.0 if state.verification_passed else 0.0
            return ActionEstimate(action, success, 1.0, Decimal("0"), 0.0, success)
        if action.operation is Operation.VERIFY and action.model is None:
            visible_signal = (
                0.85
                if state.workspace.files_changed and not any(
                    not test.passed for test in state.test_events[-2:]
                )
                else 0.35
            )
            return ActionEstimate(
                action,
                visible_signal,
                1.0,
                Decimal("0"),
                state.uncertainty,
                visible_signal + 0.03,
            )

        assert action.model is not None
        model = self.model_head.estimate(goal, state, action.model)
        operation_multiplier = self.operation_head.multiplier(
            goal, state, action.operation
        )
        if self.config.ablation == "model_only":
            operation_multiplier = 1.0
        if self.config.ablation == "operation_only":
            model = ModelEstimate(0.5, model.uncertainty)
        success = max(0.001, min(0.999, model.success_probability * operation_multiplier))
        one_turn_cost = ledger.quote_action(action, state.estimated_input_tokens)
        if action.operation is Operation.PARALLEL and action.branch_share is not None:
            branch_reserve = ledger.available_usd * Decimal("2") * action.branch_share
            # The reserve enforces the cap; expected spend remains model-sensitive.
            expected_cost = min(branch_reserve, one_turn_cost * Decimal("6"))
            feasibility = 1.0
        else:
            expected_turns = Decimal(str(max(1.0, 4.0 - min(state.turn, 20) / 8)))
            expected_cost = min(ledger.available_usd, one_turn_cost * expected_turns)
            ratio = (
                float(ledger.available_usd / expected_cost)
                if expected_cost > 0
                else 10.0
            )
            feasibility = 1.0 / (1.0 + math.exp(-3.0 * (ratio - 1.0)))
        switch_penalty = (
            float(self.config.switch_penalty_usd)
            if state.current_model is not None and action.model != state.current_model
            else 0.0
        )
        normalized_cost = float(expected_cost / max(ledger.hard_limit_usd, Decimal("0.01")))
        value = success * feasibility - 0.12 * normalized_cost - switch_penalty
        return ActionEstimate(
            action=action,
            success_probability=success,
            feasibility_probability=feasibility,
            expected_cost_to_go_usd=expected_cost,
            uncertainty=model.uncertainty,
            value=value,
        )

    @staticmethod
    def _curve(
        estimate: ActionEstimate, remaining: Decimal
    ) -> tuple[FeasibilityPoint, ...]:
        levels = (Decimal("0.1"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1"))
        cost = estimate.expected_cost_to_go_usd
        points: list[FeasibilityPoint] = []
        previous = 0.0
        for level in levels:
            budget = remaining * level
            if cost == 0:
                probability = estimate.feasibility_probability
            else:
                ratio = float(budget / cost)
                probability = estimate.feasibility_probability / (
                    1.0 + math.exp(-4 * (ratio - 1))
                )
            probability = max(previous, min(1.0, probability))
            previous = probability
            points.append(FeasibilityPoint(budget, probability))
        return tuple(points)

    def decide(
        self, goal: GoalContext, state: RouterState, ledger: BudgetLedger
    ) -> RouterDecision:
        actions, masked, hysteresis_masked = self._candidate_actions(goal, state, ledger)
        if not actions:
            actions = [RouterAction(Operation.STOP, max_output_tokens=0)]
        estimates = tuple(self._estimate(goal, state, ledger, action) for action in actions)

        if state.verification_passed:
            selected = next(e for e in estimates if e.action.operation is Operation.STOP)
            reasons = (ReasonCode.VERIFIED_SUBMIT,)
        elif state.turn >= 75:
            selected = next(e for e in estimates if e.action.operation is Operation.STOP)
            reasons = (ReasonCode.TURN_LIMIT,)
        else:
            feasible = [
                estimate
                for estimate in estimates
                if estimate.feasibility_probability >= self.config.feasibility_threshold
                and estimate.action.operation is not Operation.STOP
            ]
            if not feasible:
                selected = next(e for e in estimates if e.action.operation is Operation.STOP)
                reasons = (
                    ReasonCode.HARD_CAP_STOP
                    if masked
                    else ReasonCode.LOW_VALUE_ABSTAIN,
                )
            else:
                selected = max(
                    feasible,
                    key=lambda estimate: (
                        estimate.value,
                        estimate.success_probability,
                        -float(estimate.expected_cost_to_go_usd),
                        -float(estimate.action.branch_share or Decimal("0")),
                        estimate.action.key,
                    ),
                )
                reasons_list = [ReasonCode.BEST_FEASIBLE_VALUE]
                if selected.action.operation is Operation.PARALLEL:
                    reasons_list.append(ReasonCode.PARALLEL_VALUE)
                if hysteresis_masked:
                    reasons_list.append(ReasonCode.SWITCH_HYSTERESIS)
                if masked:
                    reasons_list.append(ReasonCode.BUDGET_MASKED)
                reasons = tuple(reasons_list)

        switch_cost = (
            self.config.switch_penalty_usd
            if (
                selected.action.model is not None
                and state.current_model is not None
                and selected.action.model != state.current_model
            )
            else Decimal("0")
        )
        return RouterDecision(
            action=selected.action,
            feasibility_curve=self._curve(selected, ledger.available_usd),
            action_estimates=estimates,
            uncertainty=selected.uncertainty,
            switch_cost_usd=switch_cost,
            reason_codes=reasons,
            policy_version=self.config.policy_version,
            masked_actions=masked,
        )
