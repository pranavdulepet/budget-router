from __future__ import annotations

from scripts.analyze_sequential_cascade import analyze_cascade


def test_paired_cascade_analysis_counts_quality_and_phase_costs() -> None:
    policy = "cascade:cheap->strong"
    protocol = {
        "study_id": "cascade-study",
        "manifest_hash": "cascade-hash",
        "task_ids": ["t1", "t2"],
        "comparator": {"policy": "fixed:strong"},
        "cascade_policy": {"policy_id": policy},
    }
    baseline = {
        "task_results": [
            {
                "task_id": "t1",
                "model": "strong",
                "status": "resolved",
                "conservative_cost_usd": "0.4",
            },
            {
                "task_id": "t2",
                "model": "strong",
                "status": "graded_unresolved",
                "conservative_cost_usd": "0.5",
            },
        ]
    }
    cascade = {
        "task_results": [
            {
                "task_id": "t1",
                "model": policy,
                "status": "resolved",
                "conservative_cost_usd": "0.3",
            },
            {
                "task_id": "t2",
                "model": policy,
                "status": "resolved",
                "conservative_cost_usd": "0.4",
            },
        ]
    }
    episodes = [
        {
            "task_id": task_id,
            "model": policy,
            "submitted_patch": True,
            "handoff_count": handoff,
            "scout_early_submission": not handoff,
            "scout_cost_usd": scout_cost,
            "finisher_cost_usd": finisher_cost,
            "scout_calls": scout_calls,
            "finisher_calls": finisher_calls,
        }
        for task_id, handoff, scout_cost, finisher_cost, scout_calls, finisher_calls in (
            ("t1", 0, "0.3", "0", 3, 0),
            ("t2", 1, "0.05", "0.35", 6, 4),
        )
    ]

    result = analyze_cascade(protocol, baseline, cascade, episodes)

    assert result["fixed_comparator"]["resolved_count"] == 1
    assert result["cascade"]["resolved_count"] == 2
    assert result["cascade"]["scout_early_submission_count"] == 1
    assert result["cascade"]["handoff_count"] == 1
    assert result["paired_comparison"]["resolved_count_delta"] == 1
    assert result["paired_comparison"]["total_cost_delta_usd"] == "-0.2"
    oracle = result["cost_preserving_policy_oracle"]
    assert oracle["deployable"] is False
    assert oracle["resolved_count"] == 2
    assert oracle["total_cost_usd"] == "0.7"
    assert oracle["selection_counts"] == {"fixed": 0, "cascade": 2}
    assert result["frozen_success_rule_passed"] is True
