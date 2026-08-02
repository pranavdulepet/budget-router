from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .types import GoalContext, Operation, RouterAction, RouterState, TerminalOutcome


@dataclass(frozen=True, slots=True)
class OuterAction:
    model: str
    mode: str = "direct"  # direct or probe

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "probe"}:
            raise ValueError("mode must be direct or probe")

    @property
    def key(self) -> str:
        return f"{self.model}:{self.mode}"


class TerminalOnlyBandit:
    """Diagonal contextual bandit updated only from terminal verified feedback."""

    def __init__(
        self,
        actions: Sequence[OuterAction],
        feature_names: Sequence[str],
        *,
        algorithm: str = "linucb",
        exploration: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not actions or not feature_names:
            raise ValueError("actions and feature_names must be non-empty")
        if algorithm not in {"linucb", "thompson"}:
            raise ValueError("algorithm must be linucb or thompson")
        self.actions = tuple(actions)
        self.feature_names = tuple(feature_names)
        self.algorithm = algorithm
        self.exploration = exploration
        self._rng = random.Random(seed)
        self._precision = {
            action.key: [1.0] * len(feature_names) for action in actions
        }
        self._reward_sum = {
            action.key: [0.0] * len(feature_names) for action in actions
        }

    def vectorize(self, context: Mapping[str, float]) -> list[float]:
        return [float(context.get(name, 0.0)) for name in self.feature_names]

    def _mean(self, action: OuterAction) -> list[float]:
        return [
            total / precision
            for total, precision in zip(
                self._reward_sum[action.key],
                self._precision[action.key],
                strict=True,
            )
        ]

    def scores(self, context: Mapping[str, float]) -> Mapping[str, float]:
        vector = self.vectorize(context)
        result: dict[str, float] = {}
        for action in self.actions:
            mean_weights = self._mean(action)
            mean_reward = sum(
                value * weight for value, weight in zip(vector, mean_weights, strict=True)
            )
            variance = sum(
                value * value / precision
                for value, precision in zip(
                    vector, self._precision[action.key], strict=True
                )
            )
            if self.algorithm == "linucb":
                score = mean_reward + self.exploration * math.sqrt(max(0.0, variance))
            else:
                score = mean_reward + self._rng.gauss(
                    0.0, self.exploration * math.sqrt(max(0.0, variance))
                )
            result[action.key] = score
        return result

    def select(self, context: Mapping[str, float]) -> OuterAction:
        scores = self.scores(context)
        return max(self.actions, key=lambda action: (scores[action.key], action.key))

    def update_terminal(
        self,
        context: Mapping[str, float],
        action: OuterAction,
        outcome: TerminalOutcome,
    ) -> None:
        if not isinstance(outcome, TerminalOutcome):
            raise TypeError("bandit updates require TerminalOutcome")
        if action.key not in self._precision:
            raise KeyError(f"unknown action {action.key}")
        reward = float(outcome.resolved and outcome.verified)
        vector = self.vectorize(context)
        for index, value in enumerate(vector):
            self._precision[action.key][index] += value * value
            self._reward_sum[action.key][index] += reward * value


@dataclass(slots=True)
class RetrievalMemory:
    k: int = 5
    _items: list[tuple[tuple[float, ...], str, float]] = field(default_factory=list)

    def add(self, features: Sequence[float], action_key: str, reward: float) -> None:
        self._items.append((tuple(map(float, features)), action_key, float(reward)))

    def select(self, features: Sequence[float], actions: Sequence[OuterAction]) -> OuterAction:
        if not self._items:
            return actions[0]
        query = tuple(map(float, features))
        neighbors = sorted(
            self._items,
            key=lambda item: sum(
                (left - right) ** 2
                for left, right in zip(query, item[0], strict=True)
            ),
        )[: self.k]
        totals: dict[str, list[float]] = {}
        for _, action, reward in neighbors:
            totals.setdefault(action, []).append(reward)
        return max(
            actions,
            key=lambda action: (
                sum(totals.get(action.key, [0.0])) / len(totals.get(action.key, [0.0])),
                action.key,
            ),
        )


@dataclass(slots=True)
class OuterBanditSessionAdapter:
    """Bridge explicit terminal session feedback to an outer task-level bandit."""

    bandit: TerminalOnlyBandit
    features: Callable[[GoalContext, RouterState], Mapping[str, float]]

    def update_terminal(
        self,
        goal: GoalContext,
        state: RouterState,
        action: RouterAction,
        outcome: TerminalOutcome,
    ) -> None:
        model = action.model or state.current_model
        if model is None:
            return
        mode = "probe" if action.operation is Operation.REFLECT else "direct"
        outer_action = OuterAction(model, mode)
        if outer_action.key not in {item.key for item in self.bandit.actions}:
            return
        self.bandit.update_terminal(
            self.features(goal, state),
            outer_action,
            outcome,
        )
