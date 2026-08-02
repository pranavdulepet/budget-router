from __future__ import annotations

import hashlib

from budget_router.serialization import stable_json
from scripts.evaluate_frozen_task_router import evaluate_frozen_router


def test_frozen_router_evaluation_uses_only_post_freeze_test_grades() -> None:
    artifact = {
        "artifact_hash": "artifact-hash",
        "model_head": {
            "kind": "hashed-linear-v1",
            "dimension": 16,
            "hash_seed": "seed",
            "models": {
                "cheap": {"bias": 2.0, "weights": {}},
                "strong": {"bias": -2.0, "weights": {}},
            },
        },
        "selector": {
            "expected_cost_usd": {"cheap": "0.1", "strong": "0.5"},
            "cost_penalty": 0.1,
        },
    }
    artifact_bytes = (stable_json(artifact) + "\n").encode()
    freeze = {
        "study_manifest_hash": "study-hash",
        "router_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "router_artifact_hash": "artifact-hash",
        "freeze_hash": "freeze-hash",
        "models": ["cheap", "strong"],
    }
    study = {
        "study_id": "study",
        "manifest_hash": "study-hash",
        "stages": {"test": {"task_ids": ["t1", "t2"]}},
    }
    tasks = {
        "records": [
            {"instance_id": "t1", "repo": "repo/a", "problem_statement": "easy"},
            {"instance_id": "t2", "repo": "repo/b", "problem_statement": "hard"},
        ]
    }
    grades = {
        "task_results": [
            {
                "task_id": task_id,
                "model": model,
                "status": "resolved" if resolved else "graded_unresolved",
                "conservative_cost_usd": cost,
            }
            for task_id, model, resolved, cost in (
                ("t1", "cheap", True, "0.1"),
                ("t1", "strong", True, "0.5"),
                ("t2", "cheap", False, "0.1"),
                ("t2", "strong", True, "0.5"),
            )
        ]
    }

    result = evaluate_frozen_router(
        study,
        freeze,
        artifact,
        tasks,
        [grades],
        artifact_bytes=artifact_bytes,
    )

    assert result["provenance"]["test_labels_accessed_after_freeze"] is True
    assert result["policies"]["router"]["selection_counts"] == {
        "cheap": 2,
        "strong": 0,
    }
    assert result["best_fixed_model"] == "strong"
    assert result["policies"]["fixed:cheap"]["resolution_rate_95_interval"] == [
        0.09453120573423074,
        0.9054687942657693,
    ]
    assert result["pairwise_fixed_model_comparisons"] == [
        {
            "left_model": "cheap",
            "right_model": "strong",
            "left_only_resolved": 0,
            "right_only_resolved": 1,
            "discordant_task_count": 1,
            "resolution_rate_delta": -0.5,
            "two_sided_exact_mcnemar_p": 1.0,
        }
    ]
