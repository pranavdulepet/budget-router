from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import stable_hash, stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(
    study: dict[str, Any],
    tasks_payload: dict[str, Any],
    grade_payloads: Iterable[dict[str, Any]],
    candidate_models: list[str],
    *,
    splits: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = {
        str(row["instance_id"]): dict(row) for row in tasks_payload["records"]
    }
    study_tasks = {
        str(row["task_id"]): dict(row) for row in study["tasks"]
    }
    grade_rows: dict[tuple[str, str], dict[str, Any]] = {}
    grade_hashes: list[str] = []
    for payload in grade_payloads:
        grade_hashes.append(stable_hash(payload))
        for row in payload["task_results"]:
            key = (str(row["task_id"]), str(row["model"]))
            if key in grade_rows:
                raise ValueError(f"duplicate grade row across inputs: {key}")
            grade_rows[key] = dict(row)
    unknown_models = sorted(set(candidate_models) - set(study["models"]))
    if unknown_models:
        raise ValueError(f"candidate models are outside the study: {unknown_models}")

    selected_task_ids = [
        task_id
        for task_id, metadata in study_tasks.items()
        if str(metadata["split"]) in splits
    ]
    expected = {
        (task_id, model)
        for task_id in selected_task_ids
        for model in candidate_models
    }
    missing = sorted(expected - set(grade_rows))
    if missing:
        raise ValueError(f"official grades are incomplete for dataset: {missing}")

    rows: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        task = tasks[task_id]
        metadata = study_tasks[task_id]
        for model in candidate_models:
            grade = grade_rows[(task_id, model)]
            rows.append(
                {
                    "schema_version": "router-supervision-row-v1",
                    "study_id": study["study_id"],
                    "study_manifest_hash": study["manifest_hash"],
                    "task_id": task_id,
                    "split": metadata["split"],
                    "repository": task["repo"],
                    "model": model,
                    "problem_statement": task["problem_statement"],
                    "resolved": grade["status"] == "resolved",
                    "grade_status": grade["status"],
                    "conservative_cost_usd": str(
                        grade["conservative_cost_usd"]
                    ),
                }
            )
    rows.sort(key=lambda row: (row["split"], row["task_id"], row["model"]))
    manifest = {
        "schema_version": "router-supervision-manifest-v1",
        "study_id": study["study_id"],
        "study_manifest_hash": study["manifest_hash"],
        "candidate_models": candidate_models,
        "splits": list(splits),
        "task_count": len(selected_task_ids),
        "row_count": len(rows),
        "task_text_fields": ["problem_statement"],
        "excluded_features": [
            "repository identity",
            "gold patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "held-out grader logs",
            "test split outcomes before router freeze",
        ],
        "grade_payload_hashes": grade_hashes,
    }
    manifest["dataset_hash"] = stable_hash({"manifest": manifest, "rows": rows})
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument("--grades", type=Path, action="append", required=True)
    parser.add_argument(
        "--screen-gate",
        type=Path,
        default=Path("outputs/router_baseline_v2/screen_gate.json"),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "calibration", "test"),
        default=[],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_dataset.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_dataset_manifest.json"),
    )
    args = parser.parse_args()
    study = _load(args.study)
    gate = _load(args.screen_gate)
    if (
        gate.get("study_manifest_hash") != study["manifest_hash"]
        or gate.get("screen_gate_passed") is not True
    ):
        raise PermissionError("router dataset requires a passing screen gate")
    splits = tuple(args.split or ("train", "calibration"))
    if "test" in splits:
        raise PermissionError(
            "test labels must remain locked until after the router is frozen"
        )
    rows, manifest = build_dataset(
        study,
        _load(args.tasks),
        [_load(path) for path in args.grades],
        [str(model) for model in gate["surviving_models"]],
        splits=splits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(stable_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
