from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .policy import ModelEstimate
from .types import GoalContext, RouterState

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*|\d+|[^\s]")


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def hashed_text_features(
    text: str,
    *,
    dimension: int,
    seed: str,
) -> dict[int, float]:
    if dimension <= 0:
        raise ValueError("feature dimension must be positive")
    tokens = [token.lower() for token in _TOKEN.findall(text)[:4_000]]
    terms = [f"u:{token}" for token in tokens]
    terms.extend(
        f"b:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    )
    terms.append(f"length:{min(15, len(tokens) // 64)}")
    counts: dict[int, float] = {}
    for term in terms:
        digest = hashlib.sha256(f"{seed}:{term}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        counts[index] = counts.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm:
        return {index: value / norm for index, value in counts.items()}
    return {}


@dataclass(frozen=True, slots=True)
class BinaryTextExample:
    task_id: str
    text: str
    labels: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class HashedLinearFit:
    models: tuple[str, ...]
    dimension: int
    hash_seed: str
    biases: Mapping[str, float]
    weights: Mapping[str, Mapping[int, float]]
    training_examples: int

    def probabilities(self, text: str) -> dict[str, float]:
        features = hashed_text_features(
            text,
            dimension=self.dimension,
            seed=self.hash_seed,
        )
        return {
            model: _sigmoid(
                float(self.biases[model])
                + sum(
                    float(self.weights[model].get(index, 0.0)) * value
                    for index, value in features.items()
                )
            )
            for model in self.models
        }

    def to_artifact_model_head(self) -> dict[str, Any]:
        return {
            "kind": "hashed-linear-v1",
            "dimension": self.dimension,
            "hash_seed": self.hash_seed,
            "training_examples": self.training_examples,
            "models": {
                model: {
                    "bias": self.biases[model],
                    "weights": {
                        str(index): value
                        for index, value in sorted(self.weights[model].items())
                    },
                }
                for model in self.models
            },
        }


def fit_hashed_linear(
    examples: Iterable[BinaryTextExample],
    models: Iterable[str],
    *,
    dimension: int = 1_024,
    hash_seed: str = "budget-router-task-v1",
    learning_rate: float = 0.5,
    l2: float = 0.01,
    epochs: int = 200,
    random_seed: int = 0,
) -> HashedLinearFit:
    rows = list(examples)
    model_names = tuple(models)
    if not rows or not model_names:
        raise ValueError("training needs examples and models")
    if any(model not in row.labels for row in rows for model in model_names):
        raise ValueError("every example must have a label for every model")
    if learning_rate <= 0 or l2 < 0 or epochs <= 0:
        raise ValueError("invalid training hyperparameters")
    features = [
        hashed_text_features(row.text, dimension=dimension, seed=hash_seed)
        for row in rows
    ]
    biases: dict[str, float] = {}
    fitted_weights: dict[str, dict[int, float]] = {}
    for model_index, model in enumerate(model_names):
        labels = [1.0 if row.labels[model] else 0.0 for row in rows]
        positives = sum(labels)
        prior = (positives + 0.5) / (len(rows) + 1.0)
        bias = math.log(prior / (1.0 - prior))
        weights: dict[int, float] = {}
        positive_weight = len(rows) / (2.0 * positives) if positives else 1.0
        negatives = len(rows) - positives
        negative_weight = len(rows) / (2.0 * negatives) if negatives else 1.0
        rng = random.Random(random_seed + model_index)
        order = list(range(len(rows)))
        for epoch in range(epochs):
            rng.shuffle(order)
            gradient: dict[int, float] = {}
            bias_gradient = 0.0
            for row_index in order:
                feature_row = features[row_index]
                prediction = _sigmoid(
                    bias
                    + sum(
                        weights.get(index, 0.0) * value
                        for index, value in feature_row.items()
                    )
                )
                label = labels[row_index]
                class_weight = positive_weight if label else negative_weight
                error = (prediction - label) * class_weight
                bias_gradient += error
                for index, value in feature_row.items():
                    gradient[index] = gradient.get(index, 0.0) + error * value
            rate = learning_rate / math.sqrt(1.0 + epoch / 20.0)
            scale = 1.0 / len(rows)
            bias -= rate * bias_gradient * scale
            for index in set(weights) | set(gradient):
                updated = weights.get(index, 0.0) - rate * (
                    gradient.get(index, 0.0) * scale
                    + l2 * weights.get(index, 0.0)
                )
                if abs(updated) >= 1e-10:
                    weights[index] = updated
                else:
                    weights.pop(index, None)
        biases[model] = bias
        fitted_weights[model] = weights
    return HashedLinearFit(
        models=model_names,
        dimension=dimension,
        hash_seed=hash_seed,
        biases=biases,
        weights=fitted_weights,
        training_examples=len(rows),
    )


@dataclass(slots=True)
class HashedLinearModelHead:
    dimension: int
    hash_seed: str
    biases: Mapping[str, float]
    weights: Mapping[str, Mapping[int, float]]

    @classmethod
    def from_artifact(cls, model_head: Mapping[str, Any]) -> HashedLinearModelHead:
        if model_head.get("kind") != "hashed-linear-v1":
            raise ValueError("unsupported task-aware model head")
        models = model_head.get("models", {})
        if not models:
            raise ValueError("task-aware model head has no models")
        return cls(
            dimension=int(model_head["dimension"]),
            hash_seed=str(model_head["hash_seed"]),
            biases={
                str(model): float(parameters["bias"])
                for model, parameters in models.items()
            },
            weights={
                str(model): {
                    int(index): float(value)
                    for index, value in parameters["weights"].items()
                }
                for model, parameters in models.items()
            },
        )

    def estimate(
        self,
        goal: GoalContext,
        state: RouterState,
        model: str,
    ) -> ModelEstimate:
        if model not in self.biases:
            raise KeyError(f"model is not present in the learned head: {model}")
        features = hashed_text_features(
            goal.goal,
            dimension=self.dimension,
            seed=self.hash_seed,
        )
        probability = _sigmoid(
            float(self.biases[model])
            + sum(
                float(self.weights[model].get(index, 0.0)) * value
                for index, value in features.items()
            )
        )
        progress = min(0.05, 0.008 * sum(event.success for event in state.tool_events))
        failures = 0.03 * sum(not event.success for event in state.tool_events[-4:])
        probability = min(0.995, max(0.005, probability + progress - failures))
        uncertainty = min(1.0, 2.0 * min(probability, 1.0 - probability))
        return ModelEstimate(probability, uncertainty)
