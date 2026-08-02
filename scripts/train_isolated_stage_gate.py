from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import read_jsonl, stable_hash, stable_json
from budget_router.task_model import (
    BinaryTextExample,
    HashedLinearFit,
    fit_hashed_linear,
)

HEADS = ("candidate_win", "fixed_loss")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grade_rows(
    payloads: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["task_results"]:
            key = (str(row["task_id"]), str(row["model"]))
            if key in rows:
                raise ValueError(f"duplicate official grade: {key}")
            rows[key] = dict(row)
    return rows


def build_training_rows(
    protocol: dict[str, Any],
    candidate_freeze: dict[str, Any],
    tasks_payload: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_grade_payloads: Iterable[dict[str, Any]],
    source_grade_payloads: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate_freeze.get("study_manifest_hash") != protocol["manifest_hash"]:
        raise ValueError("candidate freeze belongs to a different study")
    candidate = str(candidate_freeze["selected_policy_id"])
    fixed_model = str(protocol["fixed_policy"]["model"])
    training_ids = [
        *map(str, protocol["stages"]["screen"]["task_ids"]),
        *map(str, protocol["stages"]["expand"]["task_ids"]),
    ]
    if len(training_ids) != 35 or len(set(training_ids)) != 35:
        raise ValueError("gate training requires 35 unique tasks")
    tasks = {
        str(task["instance_id"]): dict(task)
        for task in tasks_payload["records"]
    }
    candidate_grades = _grade_rows(candidate_grade_payloads)
    source_grades = _grade_rows(source_grade_payloads)
    candidate_records: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("model") != candidate:
            continue
        if record.get("study_stage") not in {"screen", "expand"}:
            continue
        task_id = str(record["task_id"])
        if task_id in candidate_records:
            raise ValueError(f"duplicate candidate training episode: {task_id}")
        candidate_records[task_id] = dict(record)
    if set(candidate_records) != set(training_ids):
        raise ValueError("candidate training episode matrix is incomplete")

    rows: list[dict[str, Any]] = []
    for task_id in training_ids:
        candidate_key = (task_id, candidate)
        fixed_key = (task_id, fixed_model)
        if candidate_key not in candidate_grades or fixed_key not in source_grades:
            raise ValueError(f"official gate-training grades are incomplete: {task_id}")
        candidate_grade = candidate_grades[candidate_key]
        fixed_grade = source_grades[fixed_key]
        if (
            candidate_grade["status"] == "grader_error"
            or fixed_grade["status"] == "grader_error"
        ):
            raise ValueError(f"gate-training label has a grader error: {task_id}")
        candidate_resolved = candidate_grade["status"] == "resolved"
        fixed_resolved = fixed_grade["status"] == "resolved"
        task = tasks[task_id]
        rows.append(
            {
                "task_id": task_id,
                "repository": str(task["repo"]),
                "problem_statement": str(task["problem_statement"]),
                "candidate_resolved": candidate_resolved,
                "fixed_resolved": fixed_resolved,
                "candidate_win": candidate_resolved and not fixed_resolved,
                "fixed_loss": fixed_resolved and not candidate_resolved,
                "candidate_cost_usd": str(
                    candidate_records[task_id]["conservative_cost_usd"]
                ),
                "fixed_cost_usd": str(fixed_grade["conservative_cost_usd"]),
            }
        )
    return rows


def _examples(rows: list[dict[str, Any]]) -> list[BinaryTextExample]:
    return [
        BinaryTextExample(
            task_id=str(row["task_id"]),
            text=str(row["problem_statement"]),
            labels={head: bool(row[head]) for head in HEADS},
        )
        for row in rows
    ]


def _fit(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    learning_rate: float,
    l2: float,
    random_seed: int,
) -> HashedLinearFit:
    gate = protocol["gate"]
    return fit_hashed_linear(
        _examples(rows),
        HEADS,
        dimension=int(gate["dimension"]),
        hash_seed=str(gate["hash_seed"]),
        learning_rate=learning_rate,
        l2=l2,
        epochs=int(gate["epochs"]),
        random_seed=random_seed,
    )


def _policy_metrics(
    rows: list[dict[str, Any]],
    probabilities: dict[str, dict[str, float]],
    *,
    expected_saving_ratio: float,
    cost_weight: float,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    resolved = 0
    total_cost = Decimal("0")
    fixed_only_losses = 0
    candidate_only_wins = 0
    for row in rows:
        task_id = str(row["task_id"])
        prediction = probabilities[task_id]
        score = (
            prediction["candidate_win"]
            - prediction["fixed_loss"]
            + cost_weight * expected_saving_ratio
        )
        selected_candidate = score >= threshold
        selected_resolved = bool(
            row["candidate_resolved"]
            if selected_candidate
            else row["fixed_resolved"]
        )
        selected_cost = Decimal(
            str(
                row["candidate_cost_usd"]
                if selected_candidate
                else row["fixed_cost_usd"]
            )
        )
        resolved += int(selected_resolved)
        total_cost += selected_cost
        fixed_only_losses += int(
            selected_candidate
            and row["fixed_resolved"]
            and not row["candidate_resolved"]
        )
        candidate_only_wins += int(
            selected_candidate
            and row["candidate_resolved"]
            and not row["fixed_resolved"]
        )
        decisions.append(
            {
                "task_id": task_id,
                "repository": row["repository"],
                "candidate_win_probability": prediction["candidate_win"],
                "fixed_loss_probability": prediction["fixed_loss"],
                "score": score,
                "selected_candidate": selected_candidate,
                "resolved": selected_resolved,
                "cost_usd": str(selected_cost),
            }
        )
    metrics = {
        "task_count": len(rows),
        "resolved_count": resolved,
        "resolution_rate": resolved / len(rows),
        "total_cost_usd": str(total_cost),
        "mean_cost_usd": str(total_cost / len(rows)),
        "fixed_only_losses": fixed_only_losses,
        "candidate_only_wins": candidate_only_wins,
        "candidate_selection_count": sum(
            decision["selected_candidate"] for decision in decisions
        ),
    }
    return metrics, decisions


def train_gate(
    protocol: dict[str, Any],
    candidate_freeze: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(rows) != 35:
        raise ValueError("gate training requires exactly 35 paired rows")
    repositories = sorted({str(row["repository"]) for row in rows})
    if len(repositories) != 5:
        raise ValueError("leave-one-repository-out validation requires five repositories")
    mean_candidate_cost = sum(
        (Decimal(str(row["candidate_cost_usd"])) for row in rows),
        Decimal("0"),
    ) / len(rows)
    mean_fixed_cost = sum(
        (Decimal(str(row["fixed_cost_usd"])) for row in rows),
        Decimal("0"),
    ) / len(rows)
    cost_scale = max(mean_candidate_cost, mean_fixed_cost, Decimal("0.000001"))
    expected_saving_ratio = float(
        (mean_fixed_cost - mean_candidate_cost) / cost_scale
    )
    gate = protocol["gate"]
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for learning_rate in map(float, gate["learning_rates"]):
        for l2 in map(float, gate["l2_values"]):
            probabilities: dict[str, dict[str, float]] = {}
            for fold_index, held_out_repo in enumerate(repositories):
                train_rows = [
                    row for row in rows if row["repository"] != held_out_repo
                ]
                held_out_rows = [
                    row for row in rows if row["repository"] == held_out_repo
                ]
                fit = _fit(
                    train_rows,
                    protocol,
                    learning_rate=learning_rate,
                    l2=l2,
                    random_seed=20260728 + fold_index,
                )
                for row in held_out_rows:
                    probabilities[str(row["task_id"])] = fit.probabilities(
                        str(row["problem_statement"])
                    )
            if set(probabilities) != {str(row["task_id"]) for row in rows}:
                raise AssertionError("cross-validation predictions are incomplete")
            for cost_weight in map(float, gate["cost_weights"]):
                for threshold in map(float, gate["decision_thresholds"]):
                    metrics, decisions = _policy_metrics(
                        rows,
                        probabilities,
                        expected_saving_ratio=expected_saving_ratio,
                        cost_weight=cost_weight,
                        threshold=threshold,
                    )
                    key = (
                        metrics["resolved_count"],
                        -Decimal(metrics["total_cost_usd"]),
                        -metrics["fixed_only_losses"],
                        -metrics["candidate_selection_count"],
                        -learning_rate,
                        -l2,
                        -cost_weight,
                        threshold,
                    )
                    candidates.append(
                        (
                            key,
                            {
                                "learning_rate": learning_rate,
                                "l2": l2,
                                "cost_weight": cost_weight,
                                "threshold": threshold,
                                "cross_validation": metrics,
                            },
                            {
                                "decisions": decisions,
                                "probabilities": probabilities,
                            },
                        )
                    )
    _, selected, selected_cv = max(candidates, key=lambda item: item[0])
    final_fit = _fit(
        rows,
        protocol,
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        random_seed=20260728,
    )
    candidate = str(candidate_freeze["selected_policy_id"])
    artifact = {
        "schema_version": "isolated-stage-gate-artifact-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "selected_candidate_policy_id": candidate,
        "fixed_policy_id": protocol["fixed_policy"]["policy_id"],
        "model_head": final_fit.to_artifact_model_head(),
        "selector": {
            "kind": "predicted-quality-delta-plus-mean-cost-v1",
            "score": (
                "p(candidate_only_win) - p(fixed_only_loss) + "
                "cost_weight * expected_mean_saving_ratio"
            ),
            "select_candidate_when": "score >= threshold",
            "cost_weight": selected["cost_weight"],
            "threshold": selected["threshold"],
            "expected_candidate_cost_usd": str(mean_candidate_cost),
            "expected_fixed_cost_usd": str(mean_fixed_cost),
            "expected_mean_saving_ratio": expected_saving_ratio,
        },
        "training": {
            "task_count": len(rows),
            "repository_count": len(repositories),
            "repositories": repositories,
            "feature_fields": ["problem_statement"],
            "repository_identity_feature": False,
            "validation": "leave-one-repository-out",
            "hyperparameters": {
                "learning_rate": selected["learning_rate"],
                "l2": selected["l2"],
                "epochs": gate["epochs"],
            },
            "test_labels_accessed": False,
        },
        "cross_validation": selected["cross_validation"],
        "cross_validation_decisions": selected_cv["decisions"],
        "provenance": provenance or {},
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--candidate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/candidate_freeze.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes.jsonl"),
    )
    parser.add_argument("--candidate-grades", type=Path, action="append", required=True)
    parser.add_argument(
        "--source-grades",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isolated_stage_v1/task_gate.json"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/gate_freeze.json"),
    )
    args = parser.parse_args()
    protocol = _load(args.protocol)
    candidate_freeze = _load(args.candidate_freeze)
    candidate_grade_payloads = [_load(path) for path in args.candidate_grades]
    source_grade_payloads = [_load(path) for path in args.source_grades]
    rows = build_training_rows(
        protocol,
        candidate_freeze,
        _load(args.tasks),
        read_jsonl(args.records),
        candidate_grade_payloads,
        source_grade_payloads,
    )
    provenance = {
        "protocol_sha256": _sha256(args.protocol),
        "candidate_freeze_sha256": _sha256(args.candidate_freeze),
        "tasks_sha256": _sha256(args.tasks),
        "records_sha256": _sha256(args.records),
        "candidate_grade_sha256": {
            str(path): _sha256(path) for path in args.candidate_grades
        },
        "source_grade_sha256": {
            str(path): _sha256(path) for path in args.source_grades
        },
        "training_rows_hash": stable_hash(rows),
    }
    artifact = train_gate(
        protocol,
        candidate_freeze,
        rows,
        provenance=provenance,
    )
    artifact_serialized = stable_json(artifact) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != artifact_serialized:
        raise FileExistsError("refusing to overwrite a different trained gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact_serialized, encoding="utf-8")
    freeze = {
        "schema_version": "isolated-stage-gate-freeze-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "selected_candidate_policy_id": candidate_freeze["selected_policy_id"],
        "router_artifact_path": str(args.output),
        "router_artifact_sha256": hashlib.sha256(
            artifact_serialized.encode()
        ).hexdigest(),
        "router_artifact_hash": artifact["artifact_hash"],
        "test_labels_accessed": False,
    }
    freeze["freeze_hash"] = stable_hash(freeze)
    freeze_serialized = stable_json(freeze) + "\n"
    if args.freeze.exists() and args.freeze.read_text(encoding="utf-8") != freeze_serialized:
        raise FileExistsError("refusing to overwrite a different gate freeze")
    args.freeze.write_text(freeze_serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "artifact_hash": artifact["artifact_hash"],
                "freeze": str(args.freeze),
                "freeze_hash": freeze["freeze_hash"],
                "cross_validation": artifact["cross_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
