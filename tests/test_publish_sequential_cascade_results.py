from __future__ import annotations

from budget_router.serialization import stable_hash
from scripts.publish_sequential_cascade_results import build_public_result


def test_public_cascade_result_reconciles_cost_and_official_grade() -> None:
    protocol = {
        "study_id": "study",
        "design_timing": "frozen before outcomes",
        "cascade_policy": {"policy_id": "cascade"},
        "budget": {
            "recorded_prior_exposure_usd": "10",
            "abandoned_provider_hang_reserve_usd": "0.9",
            "maximum_new_cascade_cost_usd": "1.8",
            "working_limit_usd": "19",
            "absolute_limit_usd": "20",
        },
    }
    protocol["manifest_hash"] = stable_hash(protocol)
    evaluation = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "evidence_status": "exploratory",
        "task_count": 2,
        "fixed_comparator": {
            "resolved_count": 1,
            "total_cost_usd": "1.0",
        },
        "cascade": {
            "submitted_count": 1,
            "resolved_count": 1,
            "total_cost_usd": "0.8",
            "scout_total_cost_usd": "0.1",
            "finisher_total_cost_usd": "0.7",
            "scout_total_calls": 2,
            "finisher_total_calls": 3,
        },
        "paired_comparison": {"resolved_count_delta": 0},
        "cost_preserving_policy_oracle": {
            "deployable": False,
            "relative_cost_reduction_vs_fixed": 0.2,
        },
        "frozen_success_rule_passed": True,
        "task_results": [],
    }
    evaluation["evaluation_hash"] = stable_hash(evaluation)
    grades = {
        "grader": {
            "dataset": "dataset",
            "dataset_revision": "revision",
            "harness_commit": "commit",
        },
        "models": {
            "cascade": {
                "submitted_count": 1,
                "graded_count": 1,
                "resolved_count": 1,
                "grader_error_count": 0,
            }
        },
    }
    collection = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "completed_episodes": 2,
        "submitted_episodes": 1,
        "handoff_episodes": 2,
        "scout_early_submissions": 0,
        "cascade_conservative_cost_usd": "0.8",
        "total_exposure_with_prior_and_hang_reserve_usd": "11.7",
    }
    report = {
        "submitted_instances": 1,
        "resolved_instances": 1,
        "error_instances": 0,
    }

    result = build_public_result(
        protocol,
        evaluation,
        grades,
        collection,
        report,
        source_sha256={
            name: name
            for name in (
                "protocol",
                "evaluation",
                "grades",
                "collection",
                "official_report",
            )
        },
        dataset_snapshot_sha256="dataset-sha",
    )

    assert result["cost_accounting"]["reserve_adjusted_total_exposure_usd"] == "11.7"
    assert result["phase_behavior"]["scout_share_of_cascade_cost"] == 0.125
    assert result["conclusion"]["cascade_total_cost_reduction_usd"] == "0.2"
    assert result["result_hash"]
