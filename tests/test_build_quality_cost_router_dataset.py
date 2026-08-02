from __future__ import annotations

import pytest

from scripts.build_quality_cost_router_dataset import build_dataset


def test_builder_uses_official_outcomes_and_rejects_heldout() -> None:
    protocol = {"amendment_id": "a"}
    manifest = {
        "manifest_hash": "h",
        "development": [
            {"task_id": "t", "repository": "r"},
        ],
    }
    tasks = {
        "records": [
            {
                "instance_id": "t",
                "repo": "r",
                "version": "1",
                "problem_statement": "fix it",
            }
        ]
    }
    treatments = []
    records = []
    grading_models = {}
    for role, model in (
        ("cheap", "c"),
        ("medium", "m"),
        ("strong", "s"),
    ):
        treatments.append(
            {
                "role": role,
                "model": model,
                "family": role,
                "tinker_size": role,
                "context_tokens": 10,
                "renderer": "r",
            }
        )
        records.append(
            {
                "study_stage": "development",
                "task_id": "t",
                "model": model,
                "submitted_patch": True,
                "structurally_valid": True,
                "conservative_cost_usd": "1",
                "input_tokens": 1,
                "output_tokens": 1,
                "model_calls": 1,
            }
        )
        grading_models[model] = {
            "expected_task_count": 1,
            "resolved_ids": ["t"] if role == "medium" else [],
        }
    grading = {
        "study_stage": "development",
        "complete_with_no_unclassified_errors": True,
        "models": grading_models,
    }
    rows, result_manifest = build_dataset(
        protocol=protocol,
        task_manifest=manifest,
        tasks_payload=tasks,
        model_pool={"treatments": treatments},
        records=records,
        grading=grading,
        stage="development",
    )
    assert len(rows) == 3
    assert [row["resolved"] for row in rows] == [False, True, False]
    assert result_manifest["heldout_labels_accessed"] is False
    with pytest.raises(PermissionError, match="development labels only"):
        build_dataset(
            protocol=protocol,
            task_manifest=manifest,
            tasks_payload=tasks,
            model_pool={"treatments": treatments},
            records=records,
            grading=grading,
            stage="heldout",
        )
