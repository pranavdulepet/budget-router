#!/usr/bin/env python3
"""Build public-feature router supervision from official development grades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl, stable_hash, stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(
    *,
    protocol: dict[str, Any],
    task_manifest: dict[str, Any],
    tasks_payload: dict[str, Any],
    model_pool: dict[str, Any],
    records: list[dict[str, Any]],
    grading: dict[str, Any],
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage != "development":
        raise PermissionError(
            "router fitting accepts development labels only; held-out is analysis-only"
        )
    if grading.get("study_stage") != stage:
        raise ValueError("official grading stage mismatch")
    if not grading.get("complete_with_no_unclassified_errors"):
        raise ValueError("official grading contains unclassified errors")
    tasks = {
        str(row["instance_id"]): dict(row)
        for row in tasks_payload["records"]
    }
    task_ids = [str(row["task_id"]) for row in task_manifest[stage]]
    treatment_by_model = {
        str(row["model"]): dict(row) for row in model_pool["treatments"]
    }
    models = [
        str(row["model"])
        for row in sorted(
            model_pool["treatments"],
            key=lambda value: ("cheap", "medium", "strong").index(
                str(value["role"])
            ),
        )
    ]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        if row.get("study_stage") != stage:
            continue
        key = (str(row["task_id"]), str(row["model"]))
        if key in by_key:
            raise ValueError(f"duplicate fixed-matrix row: {key}")
        by_key[key] = row
    expected = {(task_id, model) for task_id in task_ids for model in models}
    if set(by_key) != expected:
        raise ValueError(
            f"development matrix is incomplete: {sorted(expected - set(by_key))}"
        )

    resolved_by_model: dict[str, set[str]] = {}
    for model in models:
        official = grading["models"].get(model)
        if official is None:
            raise ValueError(f"official grading is missing model: {model}")
        if int(official["expected_task_count"]) != len(task_ids):
            raise ValueError("official grading task count mismatch")
        resolved_by_model[model] = set(map(str, official["resolved_ids"]))

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = tasks[task_id]
        public_text = str(task["problem_statement"])
        for model in models:
            treatment = treatment_by_model[model]
            record = by_key[(task_id, model)]
            rows.append(
                {
                    "schema_version": "quality-cost-router-row-v1",
                    "amendment_id": protocol["amendment_id"],
                    "stage": stage,
                    "task_id": task_id,
                    "repository": str(task["repo"]),
                    "version": str(task.get("version", "")),
                    "problem_statement": public_text,
                    "model": model,
                    "model_role": str(treatment["role"]),
                    "model_family": str(treatment["family"]),
                    "tinker_size": str(treatment["tinker_size"]),
                    "context_tokens": int(treatment["context_tokens"]),
                    "renderer": str(treatment["renderer"]),
                    "resolved": task_id in resolved_by_model[model],
                    "submitted_patch": bool(record["submitted_patch"]),
                    "structurally_valid": bool(record["structurally_valid"]),
                    "conservative_cost_usd": str(
                        record["conservative_cost_usd"]
                    ),
                    "input_tokens": int(record["input_tokens"]),
                    "output_tokens": int(record["output_tokens"]),
                    "model_calls": int(record["model_calls"]),
                }
            )
    manifest = {
        "schema_version": "quality-cost-router-dataset-v1",
        "amendment_id": protocol["amendment_id"],
        "stage": stage,
        "task_manifest_hash": task_manifest["manifest_hash"],
        "candidate_models": models,
        "candidate_roles": ["cheap", "medium", "strong"],
        "task_count": len(task_ids),
        "row_count": len(rows),
        "feature_policy": {
            "allowed": [
                "public problem statement",
                "public repository and version",
                "frozen model-card/config metadata",
            ],
            "forbidden": [
                "gold patch",
                "official grader tests",
                "grader logs",
                "hidden reasoning",
                "held-out outcomes",
            ],
        },
        "official_grading_manifest_hash": stable_hash(grading),
        "heldout_labels_accessed": False,
    }
    manifest["dataset_hash"] = stable_hash(
        {"manifest": manifest, "rows": rows}
    )
    if scan_for_secrets({"manifest": manifest, "rows": rows}):
        raise ValueError("refusing to write a dataset containing a secret")
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_009_three_tier_quality_cost.json"
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/quality_cost_v1_task_manifest.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_quality_cost_v1.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/quality_cost_v1/fixed_matrix/episodes.jsonl"),
    )
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--stage", default="development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build_dataset(
        protocol=_load(args.protocol),
        task_manifest=_load(args.task_manifest),
        tasks_payload=_load(args.tasks),
        model_pool=_load(args.model_pool),
        records=read_jsonl(args.records),
        grading=_load(args.grading),
        stage=args.stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        stable_json(manifest) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
