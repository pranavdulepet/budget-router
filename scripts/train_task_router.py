from __future__ import annotations

import argparse
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl, stable_hash, stable_json
from budget_router.task_model import BinaryTextExample, HashedLinearFit, fit_hashed_linear


def _group_rows(
    rows: list[dict[str, Any]],
    model_order: list[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    observed_models = {str(row["model"]) for row in rows}
    if model_order is None:
        models = sorted(observed_models)
    else:
        models = [str(model) for model in model_order]
        if len(models) != len(set(models)):
            raise ValueError("dataset manifest candidate models must be unique")
        if set(models) != observed_models:
            raise ValueError(
                "dataset manifest candidate models do not match supervision rows"
            )
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        task_id = str(row["task_id"])
        model = str(row["model"])
        if model in grouped[task_id]:
            raise ValueError(f"duplicate task/model supervision row: {(task_id, model)}")
        grouped[task_id][model] = dict(row)
    for task_id, model_rows in grouped.items():
        if set(model_rows) != set(models):
            raise ValueError(f"task has an incomplete model matrix: {task_id}")
        texts = {str(row["problem_statement"]) for row in model_rows.values()}
        splits = {str(row["split"]) for row in model_rows.values()}
        if len(texts) != 1 or len(splits) != 1:
            raise ValueError(f"task supervision rows disagree: {task_id}")
    return models, dict(grouped)


def _examples(
    grouped: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    split: str,
) -> list[BinaryTextExample]:
    examples: list[BinaryTextExample] = []
    for task_id, model_rows in sorted(grouped.items()):
        first = model_rows[models[0]]
        if first["split"] != split:
            continue
        examples.append(
            BinaryTextExample(
                task_id=task_id,
                text=str(first["problem_statement"]),
                labels={
                    model: bool(model_rows[model]["resolved"]) for model in models
                },
            )
        )
    return examples


def select_model(
    fit: HashedLinearFit,
    text: str,
    expected_costs: dict[str, Decimal],
    *,
    cost_penalty: float,
) -> tuple[str, dict[str, float]]:
    probabilities = fit.probabilities(text)
    cost_scale = max(expected_costs.values(), default=Decimal("1"))
    scores = {
        model: probabilities[model]
        - cost_penalty * float(expected_costs[model] / cost_scale)
        for model in fit.models
    }
    selected = max(
        fit.models,
        key=lambda model: (
            scores[model],
            probabilities[model],
            -expected_costs[model],
        ),
    )
    return selected, probabilities


def _evaluate(
    fit: HashedLinearFit,
    examples: list[BinaryTextExample],
    grouped: dict[str, dict[str, dict[str, Any]]],
    expected_costs: dict[str, Decimal],
    cost_penalty: float,
) -> dict[str, Any]:
    resolved = 0
    total_cost = Decimal("0")
    brier_sum = 0.0
    selections: dict[str, int] = defaultdict(int)
    for example in examples:
        selected, probabilities = select_model(
            fit,
            example.text,
            expected_costs,
            cost_penalty=cost_penalty,
        )
        selections[selected] += 1
        selected_row = grouped[example.task_id][selected]
        resolved += int(bool(selected_row["resolved"]))
        total_cost += Decimal(str(selected_row["conservative_cost_usd"]))
        for model in fit.models:
            label = 1.0 if example.labels[model] else 0.0
            brier_sum += (probabilities[model] - label) ** 2
    denominator = len(examples) * len(fit.models)
    return {
        "task_count": len(examples),
        "resolved_count": resolved,
        "resolution_rate": resolved / len(examples),
        "total_cost_usd": str(total_cost),
        "mean_cost_usd": str(total_cost / len(examples)),
        "brier": brier_sum / denominator,
        "selection_counts": dict(sorted(selections.items())),
    }


def train_router(
    rows: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_models = dataset_manifest.get("candidate_models")
    models, grouped = _group_rows(
        rows,
        [str(model) for model in manifest_models]
        if manifest_models is not None
        else None,
    )
    train = _examples(grouped, models, "train")
    calibration = _examples(grouped, models, "calibration")
    if not train or not calibration:
        raise ValueError("router training requires train and calibration tasks")
    expected_costs = {
        model: sum(
            (
                Decimal(str(grouped[example.task_id][model]["conservative_cost_usd"]))
                for example in train
            ),
            Decimal("0"),
        )
        / len(train)
        for model in models
    }

    candidates: list[tuple[tuple[Any, ...], dict[str, Any], HashedLinearFit]] = []
    for learning_rate in (0.2, 0.5):
        for l2 in (0.001, 0.01, 0.05):
            fit = fit_hashed_linear(
                train,
                models,
                dimension=1_024,
                hash_seed="budget-router-task-v1",
                learning_rate=learning_rate,
                l2=l2,
                epochs=200,
                random_seed=20260727,
            )
            for cost_penalty in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
                metrics = _evaluate(
                    fit,
                    calibration,
                    grouped,
                    expected_costs,
                    cost_penalty,
                )
                key = (
                    metrics["resolved_count"],
                    -Decimal(metrics["total_cost_usd"]),
                    -metrics["brier"],
                    -cost_penalty,
                    -l2,
                )
                candidates.append(
                    (
                        key,
                        {
                            "learning_rate": learning_rate,
                            "l2": l2,
                            "epochs": 200,
                            "cost_penalty": cost_penalty,
                            "calibration": metrics,
                        },
                        fit,
                    )
                )
    _, selected, fit = max(candidates, key=lambda item: item[0])
    fixed_calibration: dict[str, dict[str, Any]] = {}
    for model in models:
        resolved_count = sum(
            example.labels[model] for example in calibration
        )
        total_cost = sum(
            (
                Decimal(
                    str(
                        grouped[example.task_id][model][
                            "conservative_cost_usd"
                        ]
                    )
                )
                for example in calibration
            ),
            Decimal("0"),
        )
        fixed_calibration[model] = {
            "resolved_count": resolved_count,
            "resolution_rate": resolved_count / len(calibration),
            "total_cost_usd": str(total_cost),
            "mean_cost_usd": str(total_cost / len(calibration)),
        }
    artifact = {
        "schema_version": "task-aware-router-artifact-v1",
        "policy": "cost-aware-static-v1",
        "study_id": rows[0]["study_id"],
        "study_manifest_hash": rows[0]["study_manifest_hash"],
        "dataset_hash": dataset_manifest["dataset_hash"],
        "models": models,
        "model_head": fit.to_artifact_model_head(),
        "selector": {
            "kind": "predicted-success-minus-normalized-cost-v1",
            "expected_cost_usd": {
                model: str(expected_costs[model]) for model in models
            },
            "cost_penalty": selected["cost_penalty"],
            "tie_break": ["higher predicted success", "lower expected cost"],
        },
        "training": {
            "train_task_count": len(train),
            "calibration_task_count": len(calibration),
            "hyperparameters": {
                key: value
                for key, value in selected.items()
                if key != "calibration"
            },
            "feature_fields": ["problem_statement"],
            "repository_identity_feature": False,
            "test_labels_accessed": False,
        },
        "calibration_selection": selected["calibration"],
        "fixed_model_calibration": fixed_calibration,
        "operation_head": {"kind": "heuristic-v1", "learned": False},
        "online_learning": False,
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_dataset.jsonl"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_baseline_v2/task_router.json"),
    )
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    dataset_manifest = json.loads(
        args.dataset_manifest.read_text(encoding="utf-8")
    )
    artifact = train_router(rows, dataset_manifest)
    if scan_for_secrets(artifact):
        raise ValueError("refusing to write a router artifact containing a secret")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(artifact) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_hash": artifact["artifact_hash"],
                "models": artifact["models"],
                "calibration_selection": artifact["calibration_selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
