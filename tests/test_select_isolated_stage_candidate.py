from __future__ import annotations

from scripts.select_isolated_stage_candidate import select_candidate


def test_candidate_selection_uses_frozen_lexicographic_rule() -> None:
    tasks = [f"task-{index}" for index in range(6)]
    policies = ["candidate-a", "candidate-b"]
    protocol = {
        "study_id": "study",
        "manifest_hash": "freeze",
        "stages": {"screen": {"task_ids": tasks}},
        "candidate_policies": [{"policy_id": policy} for policy in policies],
        "fixed_policy": {"model": "strong"},
        "selection": {"lexicographic_order": ["quality", "cost"]},
    }
    records = [
        {
            "study_stage": "screen",
            "study_manifest_hash": "freeze",
            "task_id": task,
            "model": policy,
            "structurally_valid": True,
            "provider_failed": False,
            "submitted_patch": True,
            "conservative_cost_usd": "0.40" if policy == "candidate-a" else "0.30",
        }
        for task in tasks
        for policy in policies
    ]
    # Fixed succeeds on task 0 only. Candidate A preserves it; candidate B loses
    # it but gains two other tasks. The frozen rule prioritizes avoiding the loss.
    statuses = {
        ("candidate-a", "task-0"): "resolved",
        ("candidate-a", "task-2"): "resolved",
        ("candidate-b", "task-1"): "resolved",
        ("candidate-b", "task-2"): "resolved",
    }
    candidate_grades = {
        "task_results": [
            {
                "task_id": task,
                "model": policy,
                "status": statuses.get((policy, task), "graded_unresolved"),
            }
            for task in tasks
            for policy in policies
        ]
    }
    source_grades = {
        "task_results": [
            {
                "task_id": task,
                "model": "strong",
                "status": "resolved" if task == "task-0" else "graded_unresolved",
            }
            for task in tasks
        ]
    }

    freeze = select_candidate(
        protocol,
        records,
        candidate_grades,
        [source_grades],
    )

    assert freeze["selected_policy_id"] == "candidate-a"
    metrics = {
        row["policy_id"]: row for row in freeze["candidate_metrics"]
    }
    assert metrics["candidate-a"]["fixed_only_losses"] == 0
    assert metrics["candidate-b"]["fixed_only_losses"] == 1

