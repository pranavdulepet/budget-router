from __future__ import annotations

from scripts.record_isolated_stage_abandonment import build_abandonment_record


def test_abandonment_reserves_full_cap_and_counts_unresolved() -> None:
    protocol = {
        "study_id": "study",
        "manifest_hash": "freeze",
        "price_snapshot": "prices",
        "execution_policy": {
            "kind": "isolated",
            "total_hard_cap_usd": "0.90",
        },
        "fixed_policy": {"model": "strong"},
        "stages": {"expand": {"task_ids": ["task"]}},
        "tasks": [
            {
                "task_id": "task",
                "repository": "repo",
                "dataset_ordinal": 7,
            }
        ],
    }
    candidate_freeze = {
        "study_manifest_hash": "freeze",
        "selected_policy_id": "candidate",
    }

    record = build_abandonment_record(
        protocol,
        candidate_freeze,
        [],
        task_id="task",
        stage="expand",
        reason="watchdog",
        latency_lower_bound_seconds=1_200,
    )

    assert record["submitted_patch"] is False
    assert record["provider_failed"] is True
    assert record["structurally_valid"] is False
    assert record["conservative_cost_usd"] == "0.90"
    assert record["provider_failure_reserved_usd"] == "0.90"

