from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .calibration import IsotonicCalibrator
from .serialization import stable_hash
from .task_model import (
    BinaryTextExample,
    HashedLinearFit,
    HashedLinearModelHead,
    fit_hashed_linear,
    hashed_text_features,
)


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Provider-neutral routing metadata for one candidate model."""

    name: str
    provider: str = ""
    expected_cost_usd: float | None = None
    input_cost_per_million_usd: float | None = None
    output_cost_per_million_usd: float | None = None
    context_tokens: int | None = None
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model card name must not be empty")
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        for value in (
            self.expected_cost_usd,
            self.input_cost_per_million_usd,
            self.output_cost_per_million_usd,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("model costs must be finite and non-negative")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token estimates must be non-negative")
        if (
            self.input_cost_per_million_usd is not None
            and self.output_cost_per_million_usd is not None
        ):
            return (
                input_tokens * self.input_cost_per_million_usd
                + output_tokens * self.output_cost_per_million_usd
            ) / 1_000_000
        if self.expected_cost_usd is None:
            raise ValueError(f"model {self.name!r} has no usable cost estimate")
        return self.expected_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "expected_cost_usd": self.expected_cost_usd,
            "input_cost_per_million_usd": self.input_cost_per_million_usd,
            "output_cost_per_million_usd": self.output_cost_per_million_usd,
            "context_tokens": self.context_tokens,
            "capabilities": sorted(self.capabilities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelCard:
        return cls(
            name=str(value["name"]),
            provider=str(value.get("provider", "")),
            expected_cost_usd=(
                float(value["expected_cost_usd"])
                if value.get("expected_cost_usd") is not None
                else None
            ),
            input_cost_per_million_usd=(
                float(value["input_cost_per_million_usd"])
                if value.get("input_cost_per_million_usd") is not None
                else None
            ),
            output_cost_per_million_usd=(
                float(value["output_cost_per_million_usd"])
                if value.get("output_cost_per_million_usd") is not None
                else None
            ),
            context_tokens=(
                int(value["context_tokens"])
                if value.get("context_tokens") is not None
                else None
            ),
            capabilities=frozenset(str(item) for item in value.get("capabilities", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SemanticOutcome:
    request_id: str
    text: str
    model: str
    acceptable: bool
    split: str
    cost_usd: float | None = None
    group: str = ""

    def __post_init__(self) -> None:
        if not self.request_id or not self.text.strip() or not self.model:
            raise ValueError("outcomes require request_id, text, and model")
        if self.split not in {"train", "calibration"}:
            raise ValueError("semantic training rows must use train or calibration split")
        if self.cost_usd is not None and (
            not math.isfinite(self.cost_usd) or self.cost_usd < 0
        ):
            raise ValueError("outcome cost must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticOutcome:
        acceptable = value.get("acceptable")
        if acceptable is None and value.get("score") is not None:
            acceptable = float(value["score"]) >= float(
                value.get("acceptable_threshold", 0.5)
            )
        if acceptable is None:
            raise ValueError("outcome needs acceptable or score")
        return cls(
            request_id=str(value["request_id"]),
            text=str(value.get("text", value.get("prompt", ""))),
            model=str(value["model"]),
            acceptable=bool(acceptable),
            split=str(value["split"]),
            cost_usd=(
                float(value["cost_usd"])
                if value.get("cost_usd") is not None
                else None
            ),
            group=str(value.get("group", "")),
        )


@dataclass(frozen=True, slots=True)
class SemanticRouteDecision:
    selected_model: str
    mode: str
    predicted_success: Mapping[str, float]
    expected_cost_usd: Mapping[str, float]
    utility: Mapping[str, float]
    rejected: Mapping[str, tuple[str, ...]]
    quality_floor: float | None
    policy_gate_passed: bool | None
    policy_gate_scope: str
    fallback_applied: bool
    artifact_hash: str


@dataclass(slots=True)
class LinearIsotonicCalibrator:
    """Sklearn-compatible isotonic interpolation stored without a pickle."""

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def predict_one(self, score: float) -> float:
        if not self.thresholds:
            raise RuntimeError("calibrator is not fitted")
        value = float(score)
        if value <= self.thresholds[0]:
            return self.values[0]
        if value >= self.thresholds[-1]:
            return self.values[-1]
        right = bisect.bisect_right(self.thresholds, value)
        left = right - 1
        width = self.thresholds[right] - self.thresholds[left]
        if width <= 0:
            return self.values[left]
        fraction = (value - self.thresholds[left]) / width
        return self.values[left] + fraction * (
            self.values[right] - self.values[left]
        )


SemanticCalibrator = IsotonicCalibrator | LinearIsotonicCalibrator


def _calibrator_dict(calibrator: SemanticCalibrator) -> dict[str, Any]:
    return {
        "kind": (
            "isotonic-linear-v1"
            if isinstance(calibrator, LinearIsotonicCalibrator)
            else "isotonic-pava-v1"
        ),
        "thresholds": list(calibrator.thresholds),
        "values": list(calibrator.values),
    }


def _calibrator_from_dict(value: Mapping[str, Any]) -> SemanticCalibrator:
    kind = value.get("kind")
    if kind not in {"isotonic-pava-v1", "isotonic-linear-v1"}:
        raise ValueError("unsupported semantic calibrator")
    thresholds = tuple(float(item) for item in value["thresholds"])
    values = tuple(float(item) for item in value["values"])
    if not thresholds or len(thresholds) != len(values):
        raise ValueError("invalid semantic calibrator")
    if kind == "isotonic-linear-v1":
        return LinearIsotonicCalibrator(thresholds=thresholds, values=values)
    return IsotonicCalibrator(thresholds=thresholds, values=values)


def _complete_examples(
    outcomes: Sequence[SemanticOutcome],
    models: Sequence[str],
    split: str,
) -> list[BinaryTextExample]:
    grouped: dict[str, dict[str, SemanticOutcome]] = {}
    for outcome in outcomes:
        if outcome.split != split:
            continue
        request = grouped.setdefault(outcome.request_id, {})
        if outcome.model in request:
            raise ValueError(
                f"duplicate semantic outcome: {(outcome.request_id, outcome.model)}"
            )
        request[outcome.model] = outcome
    examples: list[BinaryTextExample] = []
    for request_id, rows in grouped.items():
        if set(rows) != set(models):
            raise ValueError(f"incomplete model matrix for request {request_id!r}")
        texts = {row.text for row in rows.values()}
        if len(texts) != 1:
            raise ValueError(f"model rows disagree on text for request {request_id!r}")
        examples.append(
            BinaryTextExample(
                task_id=request_id,
                text=texts.pop(),
                labels={model: rows[model].acceptable for model in models},
            )
        )
    if not examples:
        raise ValueError(f"no complete {split} examples")
    return examples


def _mean_costs(
    outcomes: Sequence[SemanticOutcome],
    cards: Sequence[ModelCard],
) -> dict[str, float]:
    observed: dict[str, list[float]] = {card.name: [] for card in cards}
    for outcome in outcomes:
        if outcome.split == "train" and outcome.cost_usd is not None:
            observed[outcome.model].append(outcome.cost_usd)
    result: dict[str, float] = {}
    for card in cards:
        if observed[card.name]:
            result[card.name] = sum(observed[card.name]) / len(observed[card.name])
        elif card.expected_cost_usd is not None:
            result[card.name] = card.expected_cost_usd
        else:
            raise ValueError(
                f"model {card.name!r} needs training costs or expected_cost_usd"
            )
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _sklearn_hashed_features(text: str, dimension: int) -> dict[int, float]:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("this semantic artifact needs budget-router[ml]") from exc
    row = HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    ).transform([text])
    return {
        int(index): float(value)
        for index, value in zip(row.indices, row.data, strict=True)
    }


@dataclass(frozen=True, slots=True)
class SklearnHashedLinearFit:
    models: tuple[str, ...]
    dimension: int
    biases: Mapping[str, float]
    weights: Mapping[str, Mapping[int, float]]
    training_examples: int

    def probabilities(self, text: str) -> dict[str, float]:
        features = _sklearn_hashed_features(text, self.dimension)
        return {
            model: _sigmoid(
                self.biases[model]
                + sum(
                    self.weights[model].get(index, 0.0) * value
                    for index, value in features.items()
                )
            )
            for model in self.models
        }

    def to_artifact_model_head(self) -> dict[str, Any]:
        return {
            "kind": "sklearn-hashing-linear-v1",
            "dimension": self.dimension,
            "vectorizer": {
                "alternate_sign": False,
                "lowercase": True,
                "ngram_range": [1, 2],
                "norm": "l2",
            },
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

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> SklearnHashedLinearFit:
        if value.get("kind") != "sklearn-hashing-linear-v1":
            raise ValueError("unsupported sklearn semantic model head")
        vectorizer = value.get("vectorizer", {})
        if vectorizer != {
            "alternate_sign": False,
            "lowercase": True,
            "ngram_range": [1, 2],
            "norm": "l2",
        }:
            raise ValueError("unsupported sklearn hashing configuration")
        model_rows = value.get("models", {})
        if not model_rows:
            raise ValueError("sklearn semantic model head has no models")
        return cls(
            models=tuple(str(model) for model in model_rows),
            dimension=int(value["dimension"]),
            biases={
                str(model): float(parameters["bias"])
                for model, parameters in model_rows.items()
            },
            weights={
                str(model): {
                    int(index): float(weight)
                    for index, weight in parameters["weights"].items()
                }
                for model, parameters in model_rows.items()
            },
            training_examples=int(value.get("training_examples", 0)),
        )


def _fit_sklearn_hashed(
    examples: Sequence[BinaryTextExample],
    models: Sequence[str],
    *,
    dimension: int,
    random_seed: int,
) -> SklearnHashedLinearFit:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("the sklearn semantic backend needs budget-router[ml]") from exc

    matrix = HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    ).transform([example.text for example in examples])
    biases: dict[str, float] = {}
    weights: dict[str, dict[int, float]] = {}
    for model in models:
        labels = np.asarray(
            [bool(example.labels[model]) for example in examples],
            dtype=int,
        )
        if np.unique(labels).size < 2:
            prior = (float(labels.sum()) + 0.5) / (len(labels) + 1.0)
            biases[model] = math.log(prior / (1.0 - prior))
            weights[model] = {}
            continue
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2_000,
            tol=1e-4,
            class_weight="balanced",
            average=True,
            random_state=random_seed,
        )
        estimator.fit(matrix, labels)
        biases[model] = float(estimator.intercept_[0])
        weights[model] = {
            index: float(value)
            for index, value in enumerate(estimator.coef_[0])
            if abs(float(value)) >= 1e-10
        }
    return SklearnHashedLinearFit(
        models=tuple(models),
        dimension=dimension,
        biases=biases,
        weights=weights,
        training_examples=len(examples),
    )


@dataclass(frozen=True, slots=True)
class SharedHashedLinearFit:
    models: tuple[str, ...]
    dimension: int
    hash_seed: str
    intercept: float
    common_weights: Mapping[int, float]
    model_biases: Mapping[str, float]
    interaction_weights: Mapping[str, Mapping[int, float]]
    training_examples: int

    def probabilities(self, text: str) -> dict[str, float]:
        features = hashed_text_features(
            text,
            dimension=self.dimension,
            seed=self.hash_seed,
        )
        common = self.intercept + sum(
            self.common_weights.get(index, 0.0) * value
            for index, value in features.items()
        )
        result: dict[str, float] = {}
        for model in self.models:
            linear = common + self.model_biases[model] + sum(
                self.interaction_weights[model].get(index, 0.0) * value
                for index, value in features.items()
            )
            result[model] = (
                1.0 / (1.0 + math.exp(-linear))
                if linear >= 0
                else math.exp(linear) / (1.0 + math.exp(linear))
            )
        return result

    def to_artifact_model_head(self) -> dict[str, Any]:
        return {
            "kind": "shared-hashed-linear-v1",
            "dimension": self.dimension,
            "hash_seed": self.hash_seed,
            "training_examples": self.training_examples,
            "intercept": self.intercept,
            "common_weights": {
                str(index): value
                for index, value in sorted(self.common_weights.items())
            },
            "models": {
                model: {
                    "bias": self.model_biases[model],
                    "interaction_weights": {
                        str(index): value
                        for index, value in sorted(
                            self.interaction_weights[model].items()
                        )
                    },
                }
                for model in self.models
            },
        }

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> SharedHashedLinearFit:
        if value.get("kind") != "shared-hashed-linear-v1":
            raise ValueError("unsupported shared semantic model head")
        model_rows = value.get("models", {})
        if not model_rows:
            raise ValueError("shared semantic model head has no models")
        return cls(
            models=tuple(str(model) for model in model_rows),
            dimension=int(value["dimension"]),
            hash_seed=str(value["hash_seed"]),
            intercept=float(value["intercept"]),
            common_weights={
                int(index): float(weight)
                for index, weight in value["common_weights"].items()
            },
            model_biases={
                str(model): float(parameters["bias"])
                for model, parameters in model_rows.items()
            },
            interaction_weights={
                str(model): {
                    int(index): float(weight)
                    for index, weight in parameters[
                        "interaction_weights"
                    ].items()
                }
                for model, parameters in model_rows.items()
            },
            training_examples=int(value.get("training_examples", 0)),
        )


@dataclass(frozen=True, slots=True)
class SklearnSharedHashedLinearFit:
    models: tuple[str, ...]
    dimension: int
    intercept: float
    common_weights: Mapping[int, float]
    model_biases: Mapping[str, float]
    interaction_weights: Mapping[str, Mapping[int, float]]
    training_examples: int

    def probabilities(self, text: str) -> dict[str, float]:
        features = _sklearn_hashed_features(text, self.dimension)
        common = self.intercept + sum(
            self.common_weights.get(index, 0.0) * value
            for index, value in features.items()
        )
        return {
            model: _sigmoid(
                common
                + self.model_biases[model]
                + sum(
                    self.interaction_weights[model].get(index, 0.0) * value
                    for index, value in features.items()
                )
            )
            for model in self.models
        }

    def to_artifact_model_head(self) -> dict[str, Any]:
        return {
            "kind": "sklearn-shared-hashing-linear-v1",
            "dimension": self.dimension,
            "vectorizer": {
                "alternate_sign": False,
                "lowercase": True,
                "ngram_range": [1, 2],
                "norm": "l2",
            },
            "training_examples": self.training_examples,
            "intercept": self.intercept,
            "common_weights": {
                str(index): value
                for index, value in sorted(self.common_weights.items())
            },
            "models": {
                model: {
                    "bias": self.model_biases[model],
                    "interaction_weights": {
                        str(index): value
                        for index, value in sorted(
                            self.interaction_weights[model].items()
                        )
                    },
                }
                for model in self.models
            },
        }

    @classmethod
    def from_artifact(
        cls,
        value: Mapping[str, Any],
    ) -> SklearnSharedHashedLinearFit:
        if value.get("kind") != "sklearn-shared-hashing-linear-v1":
            raise ValueError("unsupported sklearn shared semantic model head")
        vectorizer = value.get("vectorizer", {})
        if vectorizer != {
            "alternate_sign": False,
            "lowercase": True,
            "ngram_range": [1, 2],
            "norm": "l2",
        }:
            raise ValueError("unsupported sklearn hashing configuration")
        model_rows = value.get("models", {})
        if not model_rows:
            raise ValueError("sklearn shared semantic model head has no models")
        return cls(
            models=tuple(str(model) for model in model_rows),
            dimension=int(value["dimension"]),
            intercept=float(value["intercept"]),
            common_weights={
                int(index): float(weight)
                for index, weight in value["common_weights"].items()
            },
            model_biases={
                str(model): float(parameters["bias"])
                for model, parameters in model_rows.items()
            },
            interaction_weights={
                str(model): {
                    int(index): float(weight)
                    for index, weight in parameters[
                        "interaction_weights"
                    ].items()
                }
                for model, parameters in model_rows.items()
            },
            training_examples=int(value.get("training_examples", 0)),
        )


def _fit_sklearn_shared_hashed(
    examples: Sequence[BinaryTextExample],
    models: Sequence[str],
    *,
    dimension: int,
    random_seed: int,
) -> SklearnSharedHashedLinearFit:
    try:
        import numpy as np
        from scipy import sparse
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("the sklearn semantic backend needs budget-router[ml]") from exc

    base = HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    ).transform([example.text for example in examples])
    model_count = len(models)
    shared = sparse.vstack([base] * model_count, format="csr")
    one_hot = sparse.kron(
        sparse.eye(model_count, format="csr"),
        np.ones((len(examples), 1)),
        format="csr",
    )
    interactions = sparse.block_diag([base] * model_count, format="csr")
    design = sparse.hstack((shared, one_hot, interactions), format="csr")
    labels = np.concatenate(
        [
            np.asarray(
                [bool(example.labels[model]) for example in examples],
                dtype=int,
            )
            for model in models
        ]
    )
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2_000,
        tol=1e-4,
        class_weight="balanced",
        average=True,
        random_state=random_seed,
    )
    estimator.fit(design, labels)
    coefficients = estimator.coef_[0]
    common = coefficients[:dimension]
    model_bias_values = coefficients[dimension : dimension + model_count]
    interaction_start = dimension + model_count
    return SklearnSharedHashedLinearFit(
        models=tuple(models),
        dimension=dimension,
        intercept=float(estimator.intercept_[0]),
        common_weights={
            index: float(value)
            for index, value in enumerate(common)
            if abs(float(value)) >= 1e-10
        },
        model_biases={
            model: float(model_bias_values[index])
            for index, model in enumerate(models)
        },
        interaction_weights={
            model: {
                feature_index: float(value)
                for feature_index, value in enumerate(
                    coefficients[
                        interaction_start
                        + model_index * dimension : interaction_start
                        + (model_index + 1) * dimension
                    ]
                )
                if abs(float(value)) >= 1e-10
            }
            for model_index, model in enumerate(models)
        },
        training_examples=len(examples),
    )


def train_semantic_router(
    outcomes: Iterable[SemanticOutcome],
    model_cards: Iterable[ModelCard],
    *,
    dimension: int = 4_096,
    hash_seed: str = "budget-router-semantic-v1",
    lambdas: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
    noninferiority_margin: float = 0.01,
    backend: str = "auto",
    aggregation: str = "auto",
    random_seed: int = 0,
) -> dict[str, Any]:
    """Fit calibrated independent model heads on an arbitrary complete matrix."""

    rows = list(outcomes)
    cards = list(model_cards)
    models = [card.name for card in cards]
    if not cards or len(models) != len(set(models)):
        raise ValueError("model cards must be non-empty and uniquely named")
    if {row.model for row in rows} != set(models):
        raise ValueError("outcome models must exactly match model cards")
    train = _complete_examples(rows, models, "train")
    calibration = _complete_examples(rows, models, "calibration")
    if aggregation not in {"auto", "micro", "macro_group"}:
        raise ValueError("aggregation must be auto, micro, or macro_group")
    group_by_request = {
        row.request_id: row.group
        for row in rows
        if row.split == "calibration"
    }
    calibration_groups = [group_by_request[example.task_id] for example in calibration]
    if aggregation == "auto":
        aggregation = (
            "macro_group"
            if calibration_groups and all(calibration_groups)
            else "micro"
        )
    if aggregation == "macro_group" and not all(calibration_groups):
        raise ValueError("macro_group aggregation requires every calibration group")

    def aggregate(values: Sequence[float]) -> float:
        if aggregation == "micro":
            return sum(values) / len(values)
        groups = sorted(set(calibration_groups))
        return sum(
            sum(
                value
                for value, row_group in zip(
                    values, calibration_groups, strict=True
                )
                if row_group == group
            )
            / sum(row_group == group for row_group in calibration_groups)
            for group in groups
        ) / len(groups)
    if backend not in {"auto", "stdlib", "sklearn"}:
        raise ValueError("backend must be auto, stdlib, or sklearn")
    active_backend = backend
    if active_backend == "auto":
        try:
            import sklearn  # noqa: F401
        except ImportError:
            active_backend = "stdlib"
        else:
            active_backend = "sklearn"
    fits: dict[
        str,
        HashedLinearFit
        | SharedHashedLinearFit
        | SklearnHashedLinearFit
        | SklearnSharedHashedLinearFit,
    ] = {}
    if active_backend == "sklearn":
        fits["independent_hashed_logistic"] = _fit_sklearn_hashed(
            train,
            models,
            dimension=dimension,
            random_seed=random_seed,
        )
        fits["shared_task_model_hashed_classifier"] = (
            _fit_sklearn_shared_hashed(
                train,
                models,
                dimension=dimension,
                random_seed=random_seed,
            )
        )
    else:
        fits["independent_hashed_logistic"] = fit_hashed_linear(
            train,
            models,
            dimension=dimension,
            hash_seed=hash_seed,
            learning_rate=0.5,
            l2=0.01,
            epochs=200,
            random_seed=0,
        )
    expected_costs = _mean_costs(rows, cards)

    labels = {model: [] for model in models}
    for example in calibration:
        for model in models:
            labels[model].append(example.labels[model])
    calibrated_by_method: dict[str, list[dict[str, float]]] = {}
    calibrators_by_method: dict[str, dict[str, SemanticCalibrator]] = {}
    for method, active_fit in fits.items():
        raw = {model: [] for model in models}
        for example in calibration:
            probabilities = active_fit.probabilities(example.text)
            for model in models:
                raw[model].append(probabilities[model])
        if active_backend == "sklearn":
            try:
                from sklearn.isotonic import IsotonicRegression
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "the sklearn semantic backend needs budget-router[ml]"
                ) from exc

            def fit_linear_isotonic(
                scores: Sequence[float],
                outcomes: Sequence[bool],
            ) -> LinearIsotonicCalibrator:
                estimator = IsotonicRegression(
                    out_of_bounds="clip",
                    y_min=0.0,
                    y_max=1.0,
                )
                estimator.fit(scores, outcomes)
                return LinearIsotonicCalibrator(
                    thresholds=tuple(float(item) for item in estimator.X_thresholds_),
                    values=tuple(float(item) for item in estimator.y_thresholds_),
                )

            global_calibrator: SemanticCalibrator = fit_linear_isotonic(
                [
                    raw[model][row_index]
                    for row_index in range(len(calibration))
                    for model in models
                ],
                [
                    labels[model][row_index]
                    for row_index in range(len(calibration))
                    for model in models
                ],
            )
            method_calibrators: dict[str, SemanticCalibrator] = {}
            for model in models:
                if len(set(labels[model])) >= 2 and len(set(raw[model])) >= 2:
                    method_calibrators[model] = fit_linear_isotonic(
                        raw[model],
                        labels[model],
                    )
                else:
                    method_calibrators[model] = global_calibrator
        else:
            global_calibrator = IsotonicCalibrator().fit(
                [value for model in models for value in raw[model]],
                [value for model in models for value in labels[model]],
            )
            method_calibrators = {}
            for model in models:
                if len(set(labels[model])) >= 2 and len(set(raw[model])) >= 2:
                    method_calibrators[model] = IsotonicCalibrator().fit(
                        raw[model],
                        labels[model],
                    )
                else:
                    method_calibrators[model] = global_calibrator
        calibrators_by_method[method] = method_calibrators
        calibrated_by_method[method] = [
            {
                model: method_calibrators[model].predict_one(
                    raw[model][row_index]
                )
                for model in models
            }
            for row_index in range(len(calibration))
        ]
    fixed_rates = {
        model: aggregate(
            [float(example.labels[model]) for example in calibration]
        )
        for model in models
    }
    best_fixed = max(
        models,
        key=lambda model: (
            fixed_rates[model],
            -expected_costs[model],
            model,
        ),
    )
    cost_scale = max(expected_costs.values(), default=1.0) or 1.0
    candidates: list[dict[str, Any]] = []
    for method, calibrated_rows in calibrated_by_method.items():
        for weight in lambdas:
            selected: list[str] = []
            for probabilities in calibrated_rows:
                selected.append(
                    max(
                        models,
                        key=lambda model: (
                            probabilities[model]
                            - float(weight) * expected_costs[model] / cost_scale,
                            probabilities[model],
                            -expected_costs[model],
                            model,
                        ),
                    )
                )
            paired = [
                float(example.labels[selected[index]])
                - float(example.labels[best_fixed])
                for index, example in enumerate(calibration)
            ]
            mean_difference = aggregate(paired)
            if aggregation == "micro":
                variance = (
                    sum((value - mean_difference) ** 2 for value in paired)
                    / max(1, len(paired) - 1)
                )
                standard_error = math.sqrt(variance / len(paired))
            else:
                groups = sorted(set(calibration_groups))
                variance_of_macro_mean = 0.0
                for group in groups:
                    local = [
                        value
                        for value, row_group in zip(
                            paired, calibration_groups, strict=True
                        )
                        if row_group == group
                    ]
                    local_mean = sum(local) / len(local)
                    local_variance = (
                        sum((value - local_mean) ** 2 for value in local)
                        / max(1, len(local) - 1)
                    )
                    variance_of_macro_mean += local_variance / len(local)
                standard_error = math.sqrt(
                    variance_of_macro_mean / (len(groups) ** 2)
                )
            one_sided_lower = mean_difference - 1.645 * standard_error
            success_rate = aggregate(
                [
                    float(calibration[index].labels[model])
                    for index, model in enumerate(selected)
                ]
            )
            mean_cost = aggregate(
                [expected_costs[model] for model in selected]
            )
            candidates.append(
                {
                    "method": method,
                    "lambda": float(weight),
                    "success_rate": success_rate,
                    "mean_expected_cost_usd": mean_cost,
                    "difference_vs_best_fixed": mean_difference,
                    "one_sided_95_lower_difference": one_sided_lower,
                    "passes_noninferiority": (
                        one_sided_lower >= -noninferiority_margin
                    ),
                }
            )
    passing = [candidate for candidate in candidates if candidate["passes_noninferiority"]]
    selected_candidate = (
        min(
            passing,
            key=lambda candidate: (
                candidate["mean_expected_cost_usd"],
                -candidate["success_rate"],
                (
                    0
                    if candidate["method"] == "independent_hashed_logistic"
                    else 1
                ),
                candidate["lambda"],
            ),
        )
        if passing
        else max(
            candidates,
            key=lambda candidate: (
                candidate["success_rate"],
                -candidate["mean_expected_cost_usd"],
                (
                    0
                    if candidate["method"] == "independent_hashed_logistic"
                    else -1
                ),
                -candidate["lambda"],
            ),
        )
    )

    artifact: dict[str, Any] = {
        "schema_version": "semantic-router-artifact-v1",
        "models": models,
        "model_cards": [card.to_dict() for card in cards],
        "model_head": fits[
            selected_candidate["method"]
        ].to_artifact_model_head(),
        "model_heads": {
            method: method_fit.to_artifact_model_head()
            for method, method_fit in fits.items()
        },
        "calibration": {
            model: _calibrator_dict(
                calibrators_by_method[selected_candidate["method"]][model]
            )
            for model in models
        },
        "calibration_heads": {
            method: {
                model: _calibrator_dict(method_calibrators[model])
                for model in models
            }
            for method, method_calibrators in calibrators_by_method.items()
        },
        "expected_cost_usd": expected_costs,
        "selector": {
            "kind": "predicted-success-minus-normalized-cost-v1",
            "default_lambda": selected_candidate["lambda"],
            "default_method": selected_candidate["method"],
            "lambda_candidates": [float(value) for value in lambdas],
            "noninferiority_margin": noninferiority_margin,
            "selection_rule": "paired-one-sided-95-lower-bound",
            "aggregation": aggregation,
            "calibration_best_fixed": best_fixed,
            "calibration_candidates": candidates,
            "gate_passed": bool(passing),
        },
        "training": {
            "train_requests": len(train),
            "calibration_requests": len(calibration),
            "complete_matrix_required": True,
            "online_learning": False,
            "backend": active_backend,
            "random_seed": random_seed,
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


class SemanticRouter:
    """Calibrated semantic model selection, separate from provider routing."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        artifact_without_hash = dict(artifact)
        expected_hash = str(artifact_without_hash.pop("artifact_hash", ""))
        if not expected_hash or stable_hash(artifact_without_hash) != expected_hash:
            raise ValueError("semantic router artifact hash mismatch")
        if artifact.get("schema_version") != "semantic-router-artifact-v1":
            raise ValueError("unsupported semantic router artifact")
        self.artifact_hash = expected_hash
        self.models = tuple(str(model) for model in artifact["models"])
        self.cards = {
            card.name: card
            for card in (
                ModelCard.from_dict(value) for value in artifact["model_cards"]
            )
        }
        if set(self.models) != set(self.cards):
            raise ValueError("artifact model cards do not match model head")
        head_value = artifact["model_head"]
        if head_value.get("kind") == "hashed-linear-v1":
            self.head: (
                HashedLinearModelHead
                | SharedHashedLinearFit
                | SklearnHashedLinearFit
                | SklearnSharedHashedLinearFit
            ) = (
                HashedLinearModelHead.from_artifact(head_value)
            )
        elif head_value.get("kind") == "shared-hashed-linear-v1":
            self.head = SharedHashedLinearFit.from_artifact(head_value)
        elif head_value.get("kind") == "sklearn-hashing-linear-v1":
            self.head = SklearnHashedLinearFit.from_artifact(head_value)
        elif head_value.get("kind") == "sklearn-shared-hashing-linear-v1":
            self.head = SklearnSharedHashedLinearFit.from_artifact(head_value)
        else:
            raise ValueError("unsupported semantic model head")
        self.calibrators = {
            str(model): _calibrator_from_dict(value)
            for model, value in artifact["calibration"].items()
        }
        self.expected_costs = {
            str(model): float(value)
            for model, value in artifact["expected_cost_usd"].items()
        }
        selector = artifact["selector"]
        self.default_lambda = float(selector["default_lambda"])
        self.default_method = str(selector["default_method"])
        self.default_policy_gate_passed = bool(selector.get("gate_passed", False))
        self.gate_passed = self.default_policy_gate_passed
        self.calibration_best_fixed = str(selector["calibration_best_fixed"])
        if self.calibration_best_fixed not in self.models:
            raise ValueError("calibration fallback model is not in the model pool")
        self.candidate_policy_gates = {
            (str(candidate["method"]), float(candidate["lambda"])): bool(
                candidate["passes_noninferiority"]
            )
            for candidate in selector.get("calibration_candidates", ())
        }

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> SemanticRouter:
        return cls(artifact)

    def predict(self, text: str) -> dict[str, float]:
        if not text.strip():
            raise ValueError("routing text must not be empty")
        if isinstance(
            self.head,
            (
                SharedHashedLinearFit,
                SklearnHashedLinearFit,
                SklearnSharedHashedLinearFit,
            ),
        ):
            raw = self.head.probabilities(text)
            return {
                model: self.calibrators[model].predict_one(raw[model])
                for model in self.models
            }
        raw_features = self.head.dimension
        # Reuse the artifact head's exact hashing and weights without fabricating
        # an agent state.
        from .task_model import hashed_text_features

        features = hashed_text_features(
            text,
            dimension=raw_features,
            seed=self.head.hash_seed,
        )
        result: dict[str, float] = {}
        for model in self.models:
            linear = self.head.biases[model] + sum(
                self.head.weights[model].get(index, 0.0) * value
                for index, value in features.items()
            )
            probability = (
                1.0 / (1.0 + math.exp(-linear))
                if linear >= 0
                else math.exp(linear) / (1.0 + math.exp(linear))
            )
            result[model] = self.calibrators[model].predict_one(probability)
        return result

    def route(
        self,
        text: str,
        *,
        mode: str = "balanced",
        allowed_models: Iterable[str] | None = None,
        required_capabilities: Iterable[str] = (),
        denied_providers: Iterable[str] = (),
        input_tokens: int = 0,
        output_tokens: int = 0,
        max_cost_usd: float | None = None,
        cost_weight: float | None = None,
        quality_threshold: float | None = None,
        max_quality_drop: float = 0.05,
        allow_ungated_policy: bool = False,
    ) -> SemanticRouteDecision:
        if mode not in {"quality", "balanced", "cost"}:
            raise ValueError("mode must be quality, balanced, or cost")
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("token estimates must be non-negative")
        if not 0.0 <= max_quality_drop <= 1.0:
            raise ValueError("max_quality_drop must be in [0, 1]")
        predictions = self.predict(text)
        allowed = set(allowed_models or self.models)
        required = set(required_capabilities)
        denied = set(denied_providers)
        rejected: dict[str, tuple[str, ...]] = {}
        costs: dict[str, float] = {}
        eligible: list[str] = []
        for model in self.models:
            card = self.cards[model]
            reasons: list[str] = []
            if model not in allowed:
                reasons.append("not_allowed")
            if card.provider in denied:
                reasons.append("provider_denied")
            if not required <= card.capabilities:
                reasons.append("missing_capability")
            if (
                card.context_tokens is not None
                and input_tokens + output_tokens > card.context_tokens
            ):
                reasons.append("context_limit")
            try:
                cost = card.estimate_cost(input_tokens, output_tokens)
            except ValueError:
                cost = self.expected_costs[model]
            costs[model] = cost
            if max_cost_usd is not None and cost > max_cost_usd:
                reasons.append("request_cost_limit")
            if reasons:
                rejected[model] = tuple(reasons)
            else:
                eligible.append(model)
        if not eligible:
            raise ValueError("no model satisfies the routing constraints")
        active_weight = self.default_lambda if cost_weight is None else float(cost_weight)
        if not self.default_policy_gate_passed:
            policy_gate_passed: bool | None = False
            policy_gate_scope = "default_policy_failed"
        elif mode == "balanced":
            policy_gate_passed = next(
                (
                    passed
                    for (method, weight), passed in self.candidate_policy_gates.items()
                    if method == self.default_method
                    and math.isclose(weight, active_weight, abs_tol=1e-12)
                ),
                None,
            )
            policy_gate_scope = (
                "frozen_default_policy"
                if math.isclose(active_weight, self.default_lambda, abs_tol=1e-12)
                else "calibration_curve_candidate"
                if policy_gate_passed is not None
                else "unevaluated_cost_weight"
            )
        elif mode == "quality":
            policy_gate_passed = self.candidate_policy_gates.get(
                (self.default_method, 0.0)
            )
            policy_gate_scope = (
                "calibration_curve_candidate"
                if policy_gate_passed is not None
                else "unevaluated_quality_mode"
            )
        else:
            policy_gate_passed = None
            policy_gate_scope = "unevaluated_cost_mode"
        if policy_gate_passed is not True and not allow_ungated_policy:
            if self.calibration_best_fixed not in eligible:
                raise ValueError(
                    "the requested policy lacks a passing calibration gate and "
                    "its fixed fallback is ineligible; change constraints or "
                    "explicitly set allow_ungated_policy=True"
                )
            selected = self.calibration_best_fixed
            return SemanticRouteDecision(
                selected_model=selected,
                mode=mode,
                predicted_success=predictions,
                expected_cost_usd=costs,
                utility={selected: predictions[selected]},
                rejected=rejected,
                quality_floor=None,
                policy_gate_passed=policy_gate_passed,
                policy_gate_scope=policy_gate_scope,
                fallback_applied=True,
                artifact_hash=self.artifact_hash,
            )

        cost_scale = max((costs[model] for model in eligible), default=1.0) or 1.0
        weight = active_weight
        utility: dict[str, float] = {}
        floor: float | None = None
        if mode == "quality":
            utility = {model: predictions[model] for model in eligible}
            selected = max(
                eligible,
                key=lambda model: (
                    predictions[model],
                    -costs[model],
                    model,
                ),
            )
        elif mode == "balanced":
            utility = {
                model: predictions[model] - weight * costs[model] / cost_scale
                for model in eligible
            }
            selected = max(
                eligible,
                key=lambda model: (
                    utility[model],
                    predictions[model],
                    -costs[model],
                    model,
                ),
            )
        else:
            best_probability = max(predictions[model] for model in eligible)
            floor = max(
                0.0,
                quality_threshold
                if quality_threshold is not None
                else best_probability - max_quality_drop,
            )
            qualified = [
                model for model in eligible if predictions[model] >= floor
            ]
            if qualified:
                selected = min(
                    qualified,
                    key=lambda model: (
                        costs[model],
                        -predictions[model],
                        model,
                    ),
                )
            else:
                selected = max(
                    eligible,
                    key=lambda model: (
                        predictions[model],
                        -costs[model],
                        model,
                    ),
                )
            utility = {
                model: -costs[model] if predictions[model] >= floor else -1.0e300
                for model in eligible
            }
        return SemanticRouteDecision(
            selected_model=selected,
            mode=mode,
            predicted_success=predictions,
            expected_cost_usd=costs,
            utility=utility,
            rejected=rejected,
            quality_floor=floor,
            policy_gate_passed=policy_gate_passed,
            policy_gate_scope=(
                f"{policy_gate_scope}_explicit_override"
                if policy_gate_passed is not True and allow_ungated_policy
                else policy_gate_scope
            ),
            fallback_applied=False,
            artifact_hash=self.artifact_hash,
        )
