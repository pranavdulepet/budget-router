from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD


RANDOM_SEED = 20260729
LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
METHOD_COMPLEXITY = {
    "independent_hashed_logistic": 0,
    "shared_task_model_hashed_classifier": 1,
    "text_knn": 2,
    "calibrated_gradient_boosting": 3,
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class Matrix:
    models: list[str]
    texts: list[str]
    datasets: np.ndarray
    route_splits: np.ndarray
    prompt_keys: list[str]
    scores: np.ndarray
    acceptable: np.ndarray
    costs: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.route_splits == split)


def load_matrix(matrix_path: Path, manifest_path: Path) -> tuple[Matrix, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(matrix_path) != manifest["matrix_sha256"]:
        raise ValueError("prepared matrix checksum does not match manifest")
    models = [str(model) for model in manifest["models"]]
    rows: list[dict[str, Any]] = []
    with matrix_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row["outcomes"]) != set(models):
                raise ValueError(f"matrix model order mismatch at line {line_number}")
            rows.append(row)
    if len(rows) != manifest["prompt_count"]:
        raise ValueError("prepared matrix row count does not match manifest")

    scores = np.asarray(
        [[row["outcomes"][model]["score"] for model in models] for row in rows],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[row["outcomes"][model]["cost_usd"] for model in models] for row in rows],
        dtype=np.float64,
    )
    acceptable = np.asarray(
        [[row["outcomes"][model]["acceptable"] for model in models] for row in rows],
        dtype=bool,
    )
    if not np.isfinite(scores).all() or not np.isfinite(costs).all():
        raise ValueError("matrix contains non-finite outcomes")
    return (
        Matrix(
            models=models,
            texts=[str(row["origin_query"]) for row in rows],
            datasets=np.asarray([row["dataset"] for row in rows], dtype=object),
            route_splits=np.asarray([row["route_split"] for row in rows], dtype=object),
            prompt_keys=[str(row["prompt_key"]) for row in rows],
            scores=scores,
            acceptable=acceptable,
            costs=costs,
        ),
        manifest,
    )


class _ConstantClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: Any) -> np.ndarray:
        count = features.shape[0]
        return np.column_stack(
            (
                np.full(count, 1.0 - self.probability),
                np.full(count, self.probability),
            )
        )


def _binary_estimator(features: Any, labels: np.ndarray) -> Any:
    if np.unique(labels).size < 2:
        return _ConstantClassifier(float(labels.mean()))
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2_000,
        tol=1e-4,
        class_weight="balanced",
        average=True,
        random_state=RANDOM_SEED,
    )
    estimator.fit(features, labels)
    return estimator


def _shared_features(base: sparse.spmatrix, model_count: int) -> sparse.csr_matrix:
    rows = base.shape[0]
    shared = sparse.vstack([base] * model_count, format="csr")
    identity = sparse.eye(model_count, format="csr")
    model_one_hot = sparse.kron(identity, np.ones((rows, 1)), format="csr")
    interactions = sparse.block_diag([base] * model_count, format="csr")
    return sparse.hstack(
        (shared, model_one_hot, interactions),
        format="csr",
        dtype=np.float64,
    )


def _model_major_to_matrix(values: np.ndarray, rows: int, models: int) -> np.ndarray:
    return values.reshape(models, rows).T


def _fit_isotonic(
    calibration_raw: np.ndarray,
    calibration_labels: np.ndarray,
) -> tuple[list[IsotonicRegression | None], IsotonicRegression]:
    global_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    global_calibrator.fit(calibration_raw.ravel(), calibration_labels.astype(float).ravel())
    calibrators: list[IsotonicRegression | None] = []
    for model_index in range(calibration_raw.shape[1]):
        raw = calibration_raw[:, model_index]
        labels = calibration_labels[:, model_index].astype(float)
        if np.unique(raw).size < 2 or np.unique(labels).size < 2:
            calibrators.append(None)
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw, labels)
        calibrators.append(calibrator)
    return calibrators, global_calibrator


def _apply_isotonic(
    raw: np.ndarray,
    calibrators: list[IsotonicRegression | None],
    global_calibrator: IsotonicRegression,
) -> np.ndarray:
    result = np.empty_like(raw, dtype=np.float64)
    for model_index, calibrator in enumerate(calibrators):
        active = calibrator or global_calibrator
        result[:, model_index] = np.clip(
            active.predict(raw[:, model_index]),
            0.0,
            1.0,
        )
    return result


def train_predictors(
    matrix: Matrix,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    train = matrix.indices("train")
    calibration = matrix.indices("calibration")
    prediction_rows = np.concatenate(
        (calibration, matrix.indices("id_test"), matrix.indices("ood_test"))
    )
    row_lookup = {int(row): index for index, row in enumerate(prediction_rows)}
    models = len(matrix.models)

    hashing = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )
    train_hashed = hashing.transform([matrix.texts[index] for index in train])
    prediction_hashed = hashing.transform(
        [matrix.texts[index] for index in prediction_rows]
    )

    independent_estimators: list[Any] = []
    independent_raw = np.empty((len(prediction_rows), models), dtype=np.float64)
    for model_index in range(models):
        estimator = _binary_estimator(
            train_hashed,
            matrix.acceptable[train, model_index],
        )
        independent_estimators.append(estimator)
        independent_raw[:, model_index] = estimator.predict_proba(prediction_hashed)[:, 1]

    shared_train = _shared_features(train_hashed, models)
    shared_labels = np.concatenate(
        [matrix.acceptable[train, model_index] for model_index in range(models)]
    )
    shared_estimator = _binary_estimator(shared_train, shared_labels)
    shared_prediction = _shared_features(prediction_hashed, models)
    shared_raw = _model_major_to_matrix(
        shared_estimator.predict_proba(shared_prediction)[:, 1],
        len(prediction_rows),
        models,
    )

    tfidf = TfidfVectorizer(
        max_features=50_000,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_tfidf = tfidf.fit_transform([matrix.texts[index] for index in train])
    prediction_tfidf = tfidf.transform(
        [matrix.texts[index] for index in prediction_rows]
    )
    neighbor_count = min(31, len(train))
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbors.fit(train_tfidf)
    distances, neighbor_indices = neighbors.kneighbors(prediction_tfidf)
    weights = 1.0 / (distances + 0.05)
    neighbor_labels = matrix.acceptable[train][neighbor_indices]
    knn_raw = (
        (neighbor_labels * weights[:, :, None]).sum(axis=1)
        / weights.sum(axis=1)[:, None]
    )

    components = min(96, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
    dense_projection = make_pipeline(
        TruncatedSVD(n_components=components, random_state=RANDOM_SEED),
        Normalizer(copy=False),
    )
    train_dense = dense_projection.fit_transform(train_tfidf).astype(np.float32)
    prediction_dense = dense_projection.transform(prediction_tfidf).astype(np.float32)
    train_costs = matrix.costs[train].mean(axis=0)
    normalized_costs = train_costs / max(float(train_costs.max()), 1e-12)
    train_priors = matrix.acceptable[train].mean(axis=0)

    def dense_task_model_features(base: np.ndarray) -> np.ndarray:
        row_count = len(base)
        repeated = np.tile(base, (models, 1))
        one_hot = np.repeat(np.eye(models, dtype=np.float32), row_count, axis=0)
        model_cost = np.repeat(normalized_costs, row_count)[:, None].astype(np.float32)
        model_prior = np.repeat(train_priors, row_count)[:, None].astype(np.float32)
        return np.hstack((repeated, one_hot, model_cost, model_prior))

    boosted_train = dense_task_model_features(train_dense)
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    boosted.fit(boosted_train, shared_labels)
    boosted_raw = _model_major_to_matrix(
        boosted.predict_proba(dense_task_model_features(prediction_dense))[:, 1],
        len(prediction_rows),
        models,
    )

    raw_predictions = {
        "independent_hashed_logistic": independent_raw,
        "shared_task_model_hashed_classifier": shared_raw,
        "text_knn": knn_raw,
        "calibrated_gradient_boosting": boosted_raw,
    }
    calibration_positions = np.asarray(
        [row_lookup[int(row)] for row in calibration],
        dtype=int,
    )
    calibrated: dict[str, np.ndarray] = {}
    calibration_state: dict[str, Any] = {}
    for method, raw in raw_predictions.items():
        calibrators, global_calibrator = _fit_isotonic(
            raw[calibration_positions],
            matrix.acceptable[calibration],
        )
        calibrated[method] = _apply_isotonic(raw, calibrators, global_calibrator)
        calibration_state[method] = {
            "per_model": calibrators,
            "global": global_calibrator,
        }

    bundle = {
        "schema_version": "public-router-model-bundle-v1",
        "models": matrix.models,
        "prediction_rows": prediction_rows,
        "hashing_vectorizer": hashing,
        "independent_estimators": independent_estimators,
        "shared_estimator": shared_estimator,
        "tfidf_vectorizer": tfidf,
        "nearest_neighbors": neighbors,
        "knn_train_labels": matrix.acceptable[train],
        "dense_projection": dense_projection,
        "gradient_boosting": boosted,
        "train_costs": train_costs,
        "train_priors": train_priors,
        "calibration": calibration_state,
    }
    metadata = {
        "prediction_rows": prediction_rows,
        "row_lookup": row_lookup,
        "train_costs": train_costs,
        "train_priors": train_priors,
    }
    return calibrated, bundle, metadata


def select_models(predictions: np.ndarray, expected_costs: np.ndarray, weight: float) -> np.ndarray:
    normalized_cost = expected_costs / max(float(expected_costs.max()), 1e-12)
    utilities = predictions - float(weight) * normalized_cost[None, :]
    # Stable tie-breaking: higher prediction, lower expected cost, lower index.
    selections = np.empty(len(predictions), dtype=int)
    for row_index, row in enumerate(utilities):
        selections[row_index] = max(
            range(len(expected_costs)),
            key=lambda model_index: (
                float(row[model_index]),
                float(predictions[row_index, model_index]),
                -float(expected_costs[model_index]),
                -model_index,
            ),
        )
    return selections


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    if len(probabilities) <= 1:
        return 0.0
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(counts)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not mask.any():
            continue
        value += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value) if total else float("nan")


def evaluate_selection(
    matrix: Matrix,
    indices: np.ndarray,
    selections: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    row_numbers = np.arange(len(indices))
    selected_scores = matrix.scores[indices, selections]
    selected_successes = matrix.acceptable[indices, selections].astype(float)
    selected_costs = matrix.costs[indices, selections]
    datasets = matrix.datasets[indices]
    unique_datasets = sorted(set(datasets.tolist()))
    dataset_metrics: dict[str, Any] = {}
    for dataset in unique_datasets:
        mask = datasets == dataset
        dataset_metrics[dataset] = {
            "prompts": int(mask.sum()),
            "mean_score": float(selected_scores[mask].mean()),
            "success_rate": float(selected_successes[mask].mean()),
            "mean_cost_usd": float(selected_costs[mask].mean()),
            "total_cost_usd": float(selected_costs[mask].sum()),
        }
    counts = np.bincount(selections, minlength=len(matrix.models))
    result = {
        "prompt_count": len(indices),
        "dataset_count": len(unique_datasets),
        "macro_score": float(
            np.mean([dataset_metrics[dataset]["mean_score"] for dataset in unique_datasets])
        ),
        "micro_score": float(selected_scores.mean()),
        "macro_success_rate": float(
            np.mean(
                [dataset_metrics[dataset]["success_rate"] for dataset in unique_datasets]
            )
        ),
        "micro_success_rate": float(selected_successes.mean()),
        "macro_mean_cost_usd": float(
            np.mean(
                [dataset_metrics[dataset]["mean_cost_usd"] for dataset in unique_datasets]
            )
        ),
        "mean_cost_usd": float(selected_costs.mean()),
        "total_cost_usd": float(selected_costs.sum()),
        "cost_per_success_usd": (
            float(selected_costs.sum() / selected_successes.sum())
            if selected_successes.sum()
            else None
        ),
        "route_counts": {
            matrix.models[index]: int(count)
            for index, count in enumerate(counts)
            if count
        },
        "route_entropy": _entropy(counts),
        "datasets": dataset_metrics,
    }
    if predictions is not None:
        selected_probabilities = predictions[row_numbers, selections]
        result["selected_brier"] = float(
            np.mean((selected_probabilities - selected_successes) ** 2)
        )
        result["selected_ece"] = _ece(selected_probabilities, selected_successes)
        result["pairwise_brier"] = float(
            np.mean(
                (
                    predictions
                    - matrix.acceptable[indices].astype(np.float64)
                )
                ** 2
            )
        )
    return result


def evaluate_mixture(
    matrix: Matrix,
    indices: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    expected_scores = matrix.scores[indices] @ weights
    expected_successes = matrix.acceptable[indices].astype(float) @ weights
    expected_costs = matrix.costs[indices] @ weights
    datasets = matrix.datasets[indices]
    unique_datasets = sorted(set(datasets.tolist()))
    per_dataset: dict[str, Any] = {}
    for dataset in unique_datasets:
        mask = datasets == dataset
        per_dataset[dataset] = {
            "prompts": int(mask.sum()),
            "mean_score": float(expected_scores[mask].mean()),
            "success_rate": float(expected_successes[mask].mean()),
            "mean_cost_usd": float(expected_costs[mask].mean()),
            "total_cost_usd": float(expected_costs[mask].sum()),
        }
    return {
        "prompt_count": len(indices),
        "dataset_count": len(unique_datasets),
        "macro_score": float(
            np.mean([per_dataset[dataset]["mean_score"] for dataset in unique_datasets])
        ),
        "micro_score": float(expected_scores.mean()),
        "macro_success_rate": float(
            np.mean([per_dataset[dataset]["success_rate"] for dataset in unique_datasets])
        ),
        "micro_success_rate": float(expected_successes.mean()),
        "macro_mean_cost_usd": float(
            np.mean([per_dataset[dataset]["mean_cost_usd"] for dataset in unique_datasets])
        ),
        "mean_cost_usd": float(expected_costs.mean()),
        "total_cost_usd": float(expected_costs.sum()),
        "cost_per_success_usd": float(expected_costs.sum() / expected_successes.sum()),
        "route_frequencies": {
            matrix.models[index]: float(weight)
            for index, weight in enumerate(weights)
            if weight
        },
        "route_entropy": _entropy(weights),
        "datasets": per_dataset,
        "random_policy_evaluation": "exact_expectation",
    }


def _fixed_metrics(matrix: Matrix, indices: np.ndarray, model_index: int) -> dict[str, Any]:
    return evaluate_selection(
        matrix,
        indices,
        np.full(len(indices), model_index, dtype=int),
    )


def _macro_training_model_scores(matrix: Matrix, indices: np.ndarray) -> np.ndarray:
    values = np.empty(len(matrix.models), dtype=float)
    datasets = sorted(set(matrix.datasets[indices].tolist()))
    for model_index in range(len(matrix.models)):
        values[model_index] = np.mean(
            [
                matrix.scores[
                    indices[matrix.datasets[indices] == dataset],
                    model_index,
                ].mean()
                for dataset in datasets
            ]
        )
    return values


def _dataset_lookup(
    matrix: Matrix,
    train: np.ndarray,
    test: np.ndarray,
    expected_costs: np.ndarray,
) -> tuple[np.ndarray, dict[str, str], str]:
    training_datasets = sorted(set(matrix.datasets[train].tolist()))
    global_scores = _macro_training_model_scores(matrix, train)
    global_model = max(
        range(len(matrix.models)),
        key=lambda model_index: (
            global_scores[model_index],
            -expected_costs[model_index],
            -model_index,
        ),
    )
    lookup: dict[str, int] = {}
    for dataset in training_datasets:
        local = train[matrix.datasets[train] == dataset]
        lookup[dataset] = max(
            range(len(matrix.models)),
            key=lambda model_index: (
                float(matrix.scores[local, model_index].mean()),
                -expected_costs[model_index],
                -model_index,
            ),
        )
    selections = np.asarray(
        [lookup.get(str(matrix.datasets[row]), global_model) for row in test],
        dtype=int,
    )
    return (
        selections,
        {dataset: matrix.models[index] for dataset, index in lookup.items()},
        matrix.models[global_model],
    )


def _oracle(matrix: Matrix, indices: np.ndarray) -> np.ndarray:
    selections = np.empty(len(indices), dtype=int)
    for position, row in enumerate(indices):
        selections[position] = max(
            range(len(matrix.models)),
            key=lambda model_index: (
                matrix.scores[row, model_index],
                -matrix.costs[row, model_index],
                -model_index,
            ),
        )
    return selections


def _paired_bootstrap(
    matrix: Matrix,
    indices: np.ndarray,
    primary: np.ndarray,
    fixed: np.ndarray,
    *,
    repetitions: int = 2_000,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    datasets = sorted(set(matrix.datasets[indices].tolist()))
    primary_score = matrix.scores[indices, primary]
    fixed_score = matrix.scores[indices, fixed]
    primary_cost = matrix.costs[indices, primary]
    fixed_cost = matrix.costs[indices, fixed]
    score_differences = np.empty(repetitions, dtype=float)
    cost_savings = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        dataset_score_differences: list[float] = []
        primary_total = 0.0
        fixed_total = 0.0
        for dataset in datasets:
            positions = np.flatnonzero(matrix.datasets[indices] == dataset)
            sample = rng.choice(positions, size=len(positions), replace=True)
            dataset_score_differences.append(
                float((primary_score[sample] - fixed_score[sample]).mean())
            )
            primary_total += float(primary_cost[sample].sum())
            fixed_total += float(fixed_cost[sample].sum())
        score_differences[repetition] = np.mean(dataset_score_differences)
        cost_savings[repetition] = 1.0 - primary_total / fixed_total
    return {
        "repetitions": repetitions,
        "stratification": "dataset",
        "macro_score_difference_ci95": [
            float(np.quantile(score_differences, 0.025)),
            float(np.quantile(score_differences, 0.975)),
        ],
        "cost_savings_fraction_ci95": [
            float(np.quantile(cost_savings, 0.025)),
            float(np.quantile(cost_savings, 0.975)),
        ],
        "probability_primary_macro_score_ge_fixed": float(
            np.mean(score_differences >= 0.0)
        ),
    }


def _predictions_for_split(
    predictions: np.ndarray,
    metadata: dict[str, Any],
    indices: np.ndarray,
) -> np.ndarray:
    positions = [metadata["row_lookup"][int(row)] for row in indices]
    return predictions[np.asarray(positions, dtype=int)]


def run_study(
    matrix_path: Path,
    manifest_path: Path,
    amendment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    matrix, manifest = load_matrix(matrix_path, manifest_path)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if matrix.models != amendment["models"]:
        raise ValueError("prepared model pool does not match frozen amendment")
    lambdas = tuple(float(value) for value in amendment["selector"]["lambdas"])
    if lambdas != LAMBDAS:
        raise ValueError("code selector grid does not match frozen amendment")

    train = matrix.indices("train")
    calibration = matrix.indices("calibration")
    id_test = matrix.indices("id_test")
    ood_test = matrix.indices("ood_test")
    predictions, bundle, metadata = train_predictors(matrix)
    expected_costs = np.asarray(metadata["train_costs"], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    fixed_calibration = {
        model: _fixed_metrics(matrix, calibration, model_index)
        for model_index, model in enumerate(matrix.models)
    }
    best_fixed_index = max(
        range(len(matrix.models)),
        key=lambda model_index: (
            fixed_calibration[matrix.models[model_index]]["macro_score"],
            -fixed_calibration[matrix.models[model_index]]["macro_mean_cost_usd"],
            -model_index,
        ),
    )
    best_fixed_model = matrix.models[best_fixed_index]
    best_fixed_score = fixed_calibration[best_fixed_model]["macro_score"]
    quality_floor = best_fixed_score - float(
        amendment["selector"]["calibration_noninferiority_margin"]
    )

    calibration_curves: dict[str, list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    for method in METHOD_COMPLEXITY:
        local_predictions = _predictions_for_split(
            predictions[method],
            metadata,
            calibration,
        )
        curve: list[dict[str, Any]] = []
        for weight in lambdas:
            selections = select_models(local_predictions, expected_costs, weight)
            metrics = evaluate_selection(
                matrix,
                calibration,
                selections,
                predictions=local_predictions,
            )
            point = {"lambda": weight, "metrics": metrics}
            curve.append(point)
            candidates.append(
                {
                    "method": method,
                    "lambda": weight,
                    "metrics": metrics,
                    "selections": selections,
                }
            )
        calibration_curves[method] = curve

    passing = [
        candidate
        for candidate in candidates
        if candidate["metrics"]["macro_score"] >= quality_floor
    ]
    gate_passed = bool(passing)
    if passing:
        primary = min(
            passing,
            key=lambda candidate: (
                candidate["metrics"]["macro_mean_cost_usd"],
                -candidate["metrics"]["macro_score"],
                candidate["metrics"]["pairwise_brier"],
                METHOD_COMPLEXITY[candidate["method"]],
                candidate["lambda"],
            ),
        )
    else:
        primary = max(
            candidates,
            key=lambda candidate: (
                candidate["metrics"]["macro_score"],
                -candidate["metrics"]["macro_mean_cost_usd"],
                -candidate["metrics"]["pairwise_brier"],
                -METHOD_COMPLEXITY[candidate["method"]],
                -candidate["lambda"],
            ),
        )

    freeze = {
        "schema_version": "public-router-freeze-v1",
        "study_id": amendment["parent_study_id"],
        "prepared_manifest_hash": manifest["manifest_hash"],
        "matrix_sha256": manifest["matrix_sha256"],
        "amendment_sha256": _sha256(amendment_path),
        "model_bundle_sha256": _sha256(bundle_path),
        "models": matrix.models,
        "expected_cost_usd_from_training": {
            model: float(expected_costs[index])
            for index, model in enumerate(matrix.models)
        },
        "training_success_priors": {
            model: float(metadata["train_priors"][index])
            for index, model in enumerate(matrix.models)
        },
        "calibration_best_fixed_model": best_fixed_model,
        "calibration_best_fixed_macro_score": best_fixed_score,
        "calibration_quality_floor": quality_floor,
        "noninferiority_gate_passed": gate_passed,
        "primary_method": primary["method"],
        "primary_lambda": primary["lambda"],
        "primary_calibration_metrics": primary["metrics"],
        "test_labels_accessed_for_selection": False,
    }
    freeze["freeze_hash"] = hashlib.sha256(_canonical(freeze).encode()).hexdigest()
    freeze_path = output_dir / "router_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")

    # The primary policy is frozen above. Test outcomes are accessed only below.
    results_by_split: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    for split_name, indices in (("id_test", id_test), ("ood_test", ood_test)):
        primary_predictions = _predictions_for_split(
            predictions[primary["method"]],
            metadata,
            indices,
        )
        primary_selections = select_models(
            primary_predictions,
            expected_costs,
            primary["lambda"],
        )
        fixed_selections = np.full(len(indices), best_fixed_index, dtype=int)

        learned_curves: dict[str, list[dict[str, Any]]] = {}
        curve_selections: dict[tuple[str, float], np.ndarray] = {}
        for method in METHOD_COMPLEXITY:
            local_predictions = _predictions_for_split(
                predictions[method],
                metadata,
                indices,
            )
            learned_curves[method] = []
            for weight in lambdas:
                selections = select_models(local_predictions, expected_costs, weight)
                curve_selections[(method, weight)] = selections
                learned_curves[method].append(
                    {
                        "lambda": weight,
                        "metrics": evaluate_selection(
                            matrix,
                            indices,
                            selections,
                            predictions=local_predictions,
                        ),
                    }
                )

        fixed = {
            model: _fixed_metrics(matrix, indices, model_index)
            for model_index, model in enumerate(matrix.models)
        }
        posthoc_best_fixed_model = max(
            matrix.models,
            key=lambda model: (
                fixed[model]["macro_score"],
                -fixed[model]["total_cost_usd"],
                model,
            ),
        )
        cheapest_index = int(np.argmin(expected_costs))
        lookup_selections, lookup, lookup_fallback = _dataset_lookup(
            matrix,
            train,
            indices,
            expected_costs,
        )
        oracle_selections = _oracle(matrix, indices)
        primary_metrics = evaluate_selection(
            matrix,
            indices,
            primary_selections,
            predictions=primary_predictions,
        )
        best_fixed_metrics = fixed[best_fixed_model]
        route_frequencies = np.bincount(
            primary_selections,
            minlength=len(matrix.models),
        ).astype(float)
        route_frequencies /= route_frequencies.sum()
        controls = {
            "fixed_each": fixed,
            "calibration_best_fixed": {
                "model": best_fixed_model,
                "metrics": best_fixed_metrics,
            },
            "posthoc_test_best_fixed": {
                "model": posthoc_best_fixed_model,
                "deployable_selection": False,
                "metrics": fixed[posthoc_best_fixed_model],
            },
            "cheapest_fixed": {
                "model": matrix.models[cheapest_index],
                "metrics": fixed[matrix.models[cheapest_index]],
            },
            "uniform_random": evaluate_mixture(
                matrix,
                indices,
                np.ones(len(matrix.models)),
            ),
            "frequency_matched_random": evaluate_mixture(
                matrix,
                indices,
                route_frequencies,
            ),
            "training_dataset_lookup": {
                "lookup": lookup,
                "unseen_dataset_fallback": lookup_fallback,
                "metrics": evaluate_selection(matrix, indices, lookup_selections),
            },
            "hindsight_oracle": {
                "deployable": False,
                "metrics": evaluate_selection(matrix, indices, oracle_selections),
            },
        }
        score_gain = primary_metrics["macro_score"] - best_fixed_metrics["macro_score"]
        cost_saving = 1.0 - (
            primary_metrics["total_cost_usd"] / best_fixed_metrics["total_cost_usd"]
        )
        descriptive_matched = [
            {
                "method": method,
                "lambda": point["lambda"],
                "metrics": point["metrics"],
            }
            for method, curve in learned_curves.items()
            for point in curve
            if point["metrics"]["macro_score"]
            >= best_fixed_metrics["macro_score"]
            - float(amendment["selector"]["calibration_noninferiority_margin"])
        ]
        descriptive_best = (
            min(
                descriptive_matched,
                key=lambda item: (
                    item["metrics"]["macro_mean_cost_usd"],
                    -item["metrics"]["macro_score"],
                    METHOD_COMPLEXITY[item["method"]],
                ),
            )
            if descriptive_matched
            else None
        )
        descriptive_bootstrap = None
        if descriptive_best is not None:
            descriptive_bootstrap = _paired_bootstrap(
                matrix,
                indices,
                curve_selections[
                    (descriptive_best["method"], descriptive_best["lambda"])
                ],
                fixed_selections,
            )
        posthoc_quality_floor = (
            fixed[posthoc_best_fixed_model]["macro_score"]
            - float(amendment["selector"]["calibration_noninferiority_margin"])
        )
        descriptive_posthoc_matches = [
            item
            for item in descriptive_matched
            if item["metrics"]["macro_score"] >= posthoc_quality_floor
        ]
        descriptive_posthoc_best = (
            min(
                descriptive_posthoc_matches,
                key=lambda item: (
                    item["metrics"]["macro_mean_cost_usd"],
                    -item["metrics"]["macro_score"],
                    METHOD_COMPLEXITY[item["method"]],
                ),
            )
            if descriptive_posthoc_matches
            else None
        )
        results_by_split[split_name] = {
            "primary": {
                "method": primary["method"],
                "lambda": primary["lambda"],
                "metrics": primary_metrics,
                "macro_score_gain_vs_best_fixed": score_gain,
                "cost_saving_fraction_vs_best_fixed": cost_saving,
            },
            "controls": controls,
            "learned_cost_quality_curves": learned_curves,
            "paired_bootstrap_vs_best_fixed": _paired_bootstrap(
                matrix,
                indices,
                primary_selections,
                fixed_selections,
            ),
            "descriptive_test_matched_quality_best": descriptive_best,
            "descriptive_test_matched_quality_bootstrap": descriptive_bootstrap,
            "descriptive_match_to_posthoc_test_best_fixed": descriptive_posthoc_best,
        }
        for position, row in enumerate(indices):
            model_index = int(primary_selections[position])
            decisions.append(
                {
                    "route_split": split_name,
                    "dataset": str(matrix.datasets[row]),
                    "prompt_key": matrix.prompt_keys[row],
                    "selected_model": matrix.models[model_index],
                    "predicted_success": float(primary_predictions[position, model_index]),
                    "score": float(matrix.scores[row, model_index]),
                    "acceptable": bool(matrix.acceptable[row, model_index]),
                    "realized_cost_usd": float(matrix.costs[row, model_index]),
                }
            )

    decisions_path = output_dir / "test_decisions.jsonl"
    decisions_path.write_text(
        "".join(_canonical(row) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "public-router-study-results-v1",
        "study_id": amendment["parent_study_id"],
        "prepared_manifest_hash": manifest["manifest_hash"],
        "router_freeze_hash": freeze["freeze_hash"],
        "model_bundle_sha256": freeze["model_bundle_sha256"],
        "matrix": {
            "models": len(matrix.models),
            "prompts": len(matrix.texts),
            "outcomes": len(matrix.texts) * len(matrix.models),
            "split_counts": {
                split: int((matrix.route_splits == split).sum())
                for split in ("train", "calibration", "id_test", "ood_test")
            },
        },
        "calibration": {
            "fixed_models": fixed_calibration,
            "learned_cost_quality_curves": calibration_curves,
            "selection": {
                key: value
                for key, value in freeze.items()
                if key
                in {
                    "calibration_best_fixed_model",
                    "calibration_best_fixed_macro_score",
                    "calibration_quality_floor",
                    "noninferiority_gate_passed",
                    "primary_method",
                    "primary_lambda",
                    "primary_calibration_metrics",
                }
            },
        },
        "test": results_by_split,
        "test_decisions_sha256": _sha256(decisions_path),
        "limitations": [
            "Public study is one-shot routing, not long-horizon agent routing.",
            "SWE-bench and tau2 were source-excluded because the release lacks one frozen-pool model file each.",
            "Costs are historical realized costs from the pinned benchmark release.",
            "OOD ArenaHard scores rely on the upstream benchmark evaluator.",
        ],
    }
    summary["results_hash"] = hashlib.sha256(_canonical(summary).encode()).hexdigest()
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, freeze, and evaluate the pinned public classifier router."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/public_router_v1/prepared/matrix.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/public_router_v1/prepared/manifest.json"),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("artifacts/active_router_protocol_amendment_005_public_matrix.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/public_router_v1/study"),
    )
    args = parser.parse_args()
    results = run_study(
        args.matrix,
        args.manifest,
        args.amendment,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "results_hash": results["results_hash"],
                "selection": results["calibration"]["selection"],
                "id_test_primary": results["test"]["id_test"]["primary"],
                "ood_test_primary": results["test"]["ood_test"]["primary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
