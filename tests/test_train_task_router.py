from __future__ import annotations

from scripts.train_task_router import train_router


def test_train_router_produces_task_aware_frozen_artifact() -> None:
    rows = []
    for split, tasks in {
        "train": [
            ("a1", "parser json wrapper", True, False),
            ("a2", "parser yaml wrapper", True, False),
            ("b1", "symbolic theorem", False, True),
            ("b2", "symbolic integration", False, True),
        ],
        "calibration": [
            ("c1", "parser token wrapper", True, False),
            ("c2", "symbolic algebra theorem", False, True),
        ],
    }.items():
        for task_id, text, cheap_success, strong_success in tasks:
            for model, resolved, cost in (
                ("cheap", cheap_success, "0.10"),
                ("strong", strong_success, "0.50"),
            ):
                rows.append(
                    {
                        "study_id": "study",
                        "study_manifest_hash": "study-hash",
                        "task_id": task_id,
                        "split": split,
                        "model": model,
                        "problem_statement": text,
                        "resolved": resolved,
                        "conservative_cost_usd": cost,
                    }
                )
    artifact = train_router(
        rows,
        {
            "dataset_hash": "dataset-hash",
            "candidate_models": ["strong", "cheap"],
        },
    )

    assert artifact["schema_version"] == "task-aware-router-artifact-v1"
    assert artifact["training"]["test_labels_accessed"] is False
    assert artifact["model_head"]["kind"] == "hashed-linear-v1"
    assert artifact["models"] == ["strong", "cheap"]
    assert artifact["artifact_hash"]
