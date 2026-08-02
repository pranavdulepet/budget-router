"""Training and static evaluation helpers for the agent-step classifier."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agent_step import (
    FrozenAgentStepArtifact,
    FrozenAgentStepRouter,
    render_agent_prefix,
)
from .calibration import PlattCalibrator
from .task_model import hashed_text_features


@dataclass(frozen=True, slots=True)
class AgentTrainingRow:
    row_id: str
    benchmark: str
    instance_id: str
    step_index: int
    messages: tuple[Mapping[str, Any], ...]
    needs_strong: bool


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_twinrouter_rows(
    path: Path,
    *,
    excluded_swe_repository_prefixes: Iterable[str] = (),
) -> tuple[list[AgentTrainingRow], dict[str, int]]:
    excluded = set(excluded_swe_repository_prefixes)
    rows: list[AgentTrainingRow] = []
    seen: set[str] = set()
    counts = {
        "source_rows": 0,
        "excluded_swe_repository_rows": 0,
        "included_rows": 0,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            counts["source_rows"] += 1
            row_id = str(value["id"])
            if row_id in seen:
                raise ValueError(f"duplicate TwinRouterBench row: {row_id}")
            seen.add(row_id)
            benchmark = str(value["benchmark"])
            instance_id = str(value["instance_id"])
            if (
                benchmark == "swebench"
                and instance_id.split("__", 1)[0] in excluded
            ):
                counts["excluded_swe_repository_rows"] += 1
                continue
            raw_messages = value["messages"]
            if not isinstance(raw_messages, list) or not raw_messages:
                raise ValueError(f"row {row_id!r} has no router-visible messages")
            step_index = int(value["step_index"])
            if step_index <= 0:
                raise ValueError(f"row {row_id!r} has invalid one-based step index")
            target_tier_id = int(value["target_tier_id"])
            if target_tier_id not in {0, 1, 2, 3}:
                raise ValueError(f"row {row_id!r} has invalid tier")
            rows.append(
                AgentTrainingRow(
                    row_id=row_id,
                    benchmark=benchmark,
                    instance_id=instance_id,
                    step_index=step_index - 1,
                    messages=tuple(raw_messages),
                    needs_strong=target_tier_id > 0,
                )
            )
            counts["included_rows"] += 1
    if not rows:
        raise ValueError("TwinRouterBench source produced no usable rows")
    return rows, counts


def split_instances(
    rows: Sequence[AgentTrainingRow],
    *,
    seed: str,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict[str, tuple[str, ...]]:
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("split fractions must leave a static test")
    by_benchmark: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_benchmark[row.benchmark].add(row.instance_id)
    result: dict[str, list[str]] = {
        "train": [],
        "calibration": [],
        "static_test": [],
    }
    for benchmark in sorted(by_benchmark):
        instances = sorted(
            by_benchmark[benchmark],
            key=lambda instance_id: hashlib.sha256(
                f"{seed}:{benchmark}:{instance_id}".encode()
            ).hexdigest(),
        )
        count = len(instances)
        if count < 3:
            raise ValueError(f"benchmark {benchmark!r} has fewer than three instances")
        train_end = max(1, math.floor(count * train_fraction))
        calibration_count = max(1, math.floor(count * calibration_fraction))
        calibration_end = min(count - 1, train_end + calibration_count)
        result["train"].extend(instances[:train_end])
        result["calibration"].extend(instances[train_end:calibration_end])
        result["static_test"].extend(instances[calibration_end:])
    frozen = {name: tuple(sorted(values)) for name, values in result.items()}
    if any(not values for values in frozen.values()):
        raise ValueError("every split must contain instances")
    if set(frozen["train"]) & set(frozen["calibration"]):
        raise RuntimeError("train/calibration instance overlap")
    if set(frozen["train"]) & set(frozen["static_test"]):
        raise RuntimeError("train/static-test instance overlap")
    if set(frozen["calibration"]) & set(frozen["static_test"]):
        raise RuntimeError("calibration/static-test instance overlap")
    return frozen


def rows_for_split(
    rows: Sequence[AgentTrainingRow],
    instance_ids: Iterable[str],
) -> list[AgentTrainingRow]:
    allowed = set(instance_ids)
    return [row for row in rows if row.instance_id in allowed]


def _row_text(row: AgentTrainingRow) -> str:
    return render_agent_prefix(row.messages, step_index=row.step_index)[0]


def sparse_feature_matrix(
    rows: Sequence[AgentTrainingRow],
    *,
    dimension: int,
    hash_seed: str,
):
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("training requires budget-router[ml]") from exc
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        features = hashed_text_features(
            _row_text(row),
            dimension=dimension,
            seed=hash_seed,
        )
        for index, value in sorted(features.items()):
            indices.append(index)
            data.append(value)
        indptr.append(len(data))
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(rows), dimension),
    )
    labels = np.asarray([int(row.needs_strong) for row in rows], dtype=np.int8)
    return matrix, labels


def fit_logistic_with_grouped_cv(
    rows: Sequence[AgentTrainingRow],
    *,
    dimension: int,
    hash_seed: str,
    c_candidates: Sequence[float],
    fold_seed: str,
) -> tuple[float, float, dict[int, float], list[dict[str, Any]]]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("training requires budget-router[ml]") from exc
    if not rows or not c_candidates:
        raise ValueError("training rows and C candidates are required")
    matrix, labels = sparse_feature_matrix(
        rows,
        dimension=dimension,
        hash_seed=hash_seed,
    )
    folds = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{fold_seed}:{row.instance_id}".encode()
                ).digest()[:8],
                "big",
            )
            % 5
            for row in rows
        ],
        dtype=np.int8,
    )
    cv_rows: list[dict[str, Any]] = []
    for candidate in c_candidates:
        losses: list[float] = []
        for fold in range(5):
            train_mask = folds != fold
            validation_mask = folds == fold
            if not validation_mask.any():
                raise ValueError(f"grouped CV fold {fold} is empty")
            if len(set(labels[train_mask].tolist())) != 2:
                raise ValueError(f"grouped CV fold {fold} training has one class")
            model = LogisticRegression(
                C=float(candidate),
                class_weight="balanced",
                max_iter=2_000,
                solver="liblinear",
                random_state=0,
            )
            model.fit(matrix[train_mask], labels[train_mask])
            probabilities = model.predict_proba(matrix[validation_mask])[:, 1]
            losses.append(
                float(
                    log_loss(
                        labels[validation_mask],
                        probabilities,
                        labels=[0, 1],
                    )
                )
            )
        cv_rows.append(
            {
                "C": float(candidate),
                "fold_log_loss": losses,
                "mean_log_loss": sum(losses) / len(losses),
            }
        )
    selected = min(cv_rows, key=lambda item: (item["mean_log_loss"], item["C"]))
    selected_c = float(selected["C"])
    fitted = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        max_iter=2_000,
        solver="liblinear",
        random_state=0,
    )
    fitted.fit(matrix, labels)
    coefficients = fitted.coef_[0]
    weights = {
        int(index): float(value)
        for index, value in enumerate(coefficients)
        if abs(float(value)) >= 1e-12
    }
    return selected_c, float(fitted.intercept_[0]), weights, cv_rows


def fit_platt(
    raw_probabilities: Sequence[float],
    outcomes: Sequence[bool],
) -> tuple[float, float]:
    calibrator = PlattCalibrator().fit(
        raw_probabilities,
        outcomes,
        iterations=1_000,
        learning_rate=0.03,
    )
    return calibrator.slope, calibrator.intercept


def evaluate_static_rows(
    artifact: FrozenAgentStepArtifact,
    rows: Sequence[AgentTrainingRow],
) -> dict[str, Any]:
    by_instance: dict[str, list[AgentTrainingRow]] = defaultdict(list)
    for row in rows:
        by_instance[row.instance_id].append(row)
    decisions: list[tuple[AgentTrainingRow, str]] = []
    passing_instances = 0
    switch_count = 0
    forced_strong = 0
    for instance_id in sorted(by_instance):
        router = FrozenAgentStepRouter(artifact)
        instance_passed = True
        for row in sorted(by_instance[instance_id], key=lambda value: value.step_index):
            decision = router.select(row.messages, step_index=row.step_index)
            decisions.append((row, decision.model_id))
            forced_strong += int(
                decision.forced and decision.model_id == artifact.strong_model
            )
            if row.needs_strong and decision.model_id != artifact.strong_model:
                instance_passed = False
        passing_instances += int(instance_passed)
        switch_count += router.switch_count
    strong_rows = sum(row.needs_strong for row, _ in decisions)
    strong_correct = sum(
        row.needs_strong and model == artifact.strong_model
        for row, model in decisions
    )
    cheap_routes = sum(model == artifact.cheap_model for _, model in decisions)
    return {
        "rows": len(decisions),
        "instances": len(by_instance),
        "strong_rows": strong_rows,
        "strong_rows_routed_strong": strong_correct,
        "strong_recall": strong_correct / strong_rows if strong_rows else 1.0,
        "passing_instances": passing_instances,
        "trajectory_pass": passing_instances / len(by_instance),
        "cheap_routes": cheap_routes,
        "strong_routes": len(decisions) - cheap_routes,
        "cheap_route_share": cheap_routes / len(decisions),
        "switch_count": switch_count,
        "forced_strong_count": forced_strong,
    }


def select_conservative_threshold(
    *,
    artifact_factory,
    calibration_rows: Sequence[AgentTrainingRow],
    calibrated_probabilities: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    if len(calibration_rows) != len(calibrated_probabilities):
        raise ValueError("calibration rows and probabilities must align")
    candidates = sorted({0.0, *(float(value) for value in calibrated_probabilities)})
    valid: list[tuple[int, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = evaluate_static_rows(
            artifact_factory(threshold),
            calibration_rows,
        )
        if metrics["strong_recall"] == 1.0 and metrics["trajectory_pass"] == 1.0:
            valid.append((int(metrics["cheap_routes"]), threshold, metrics))
    if not valid:
        raise RuntimeError("no calibration threshold preserves strong rows")
    # Maximize cheap calls; if policies are behaviorally tied, keep the lower
    # threshold as the more conservative deterministic tie-break.
    cheap_routes, threshold, metrics = max(
        valid,
        key=lambda item: (item[0], -item[1]),
    )
    return threshold, {
        **metrics,
        "candidate_threshold_count": len(candidates),
        "valid_threshold_count": len(valid),
        "selected_cheap_routes": cheap_routes,
    }
