from __future__ import annotations

from scripts.evaluate_router_screen import evaluate_screen


def _grade(task_id: str, model: str, resolved: bool, cost: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "model": model,
        "status": "resolved" if resolved else "graded_unresolved",
        "conservative_cost_usd": cost,
        "latency_seconds": 1.0,
    }


def test_screen_retains_qualified_cheap_tier_and_prunes_dominated_premium() -> None:
    cheap, value, premium = "cheap", "value", "premium"
    pilot_ids = ["p1", "p2"]
    screen_ids = ["s1", "s2", "s3"]
    models = [cheap, value, premium]
    study = {
        "study_id": "study",
        "manifest_hash": "manifest",
        "models": models,
        "reused_pilot": {"task_ids": pilot_ids, "models": [value, premium]},
        "stages": {"screen": {"task_ids": screen_ids}},
        "screen_gate": {
            "cheap_model": cheap,
            "minimum_cheap_structurally_valid_rate": 0.75,
            "minimum_safe_cheap_opportunities": 3,
            "minimum_oracle_unique_gain_tasks_over_best_fixed": 2,
            "minimum_oracle_gain_rate_over_best_fixed": 0.05,
        },
    }
    outcomes = {
        "p1": {cheap: True, value: True, premium: False},
        "p2": {cheap: True, value: True, premium: True},
        "s1": {cheap: True, value: True, premium: False},
        "s2": {cheap: False, value: True, premium: False},
        "s3": {cheap: False, value: False, premium: False},
    }
    costs = {cheap: "0.10", value: "0.40", premium: "0.70"}
    pilot_grades = {
        "task_results": [
            *[
                _grade(task_id, model, outcomes[task_id][model], costs[model])
                for task_id in pilot_ids
                for model in (value, premium)
            ],
            _grade("p1", "historical-extra", False, "0.90"),
        ]
    }
    new_grades = {
        "task_results": [
            *[
                _grade(task_id, cheap, outcomes[task_id][cheap], costs[cheap])
                for task_id in pilot_ids
            ],
            *[
                _grade(task_id, model, outcomes[task_id][model], costs[model])
                for task_id in screen_ids
                for model in models
            ],
        ]
    }
    episodes = [
        {
            "task_id": row["task_id"],
            "model": row["model"],
            "study_manifest_hash": "manifest",
            "structurally_valid": True,
        }
        for row in new_grades["task_results"]
    ]

    result = evaluate_screen(study, pilot_grades, new_grades, episodes)

    assert result["screen_gate_passed"] is True
    assert result["signals"]["safe_cheap_opportunity_count"] == 3
    assert result["surviving_models"] == [cheap, value]
    assert result["models"][premium]["strictly_pareto_dominated_by"] == [
        cheap,
        value,
    ]
