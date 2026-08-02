from scripts.evaluate_kimi128_strong_gate import evaluate_gate


def test_kimi_gate_requires_quality_and_unique_win() -> None:
    tasks = [{"task_id": f"t{i}"} for i in range(12)]
    amendment = {
        "amendment_id": "a",
        "candidate": {"model": "kimi"},
        "fixed_medium_reference": {
            "resolved_ids": [f"t{i}" for i in range(6)],
        },
        "gate": {
            "minimum_structurally_valid": 11,
            "minimum_official_resolutions": 8,
            "minimum_unique_resolutions_beyond_fixed_medium": 1,
        },
    }
    episodes = [
        {
            "task_id": f"t{i}",
            "model": "kimi",
            "structurally_valid": True,
            "submitted_patch": True,
            "provider_failed": False,
            "conservative_cost_usd": "1",
        }
        for i in range(12)
    ]
    smoke = {
        "all_structurally_valid": True,
        "observed_conservative_cost_usd": "0.01",
    }
    grading = {
        "models": {"kimi": {"resolved_ids": [f"t{i}" for i in range(8)]}},
        "unclassified_errors": [],
    }
    result = evaluate_gate(
        amendment=amendment,
        task_manifest={"tasks": tasks},
        smoke=smoke,
        episodes=episodes,
        grading=grading,
    )
    assert result["passed"]
    assert result["unique_beyond_fixed_medium_count"] == 2

    grading["models"]["kimi"]["resolved_ids"] = [f"t{i}" for i in range(6)]
    result = evaluate_gate(
        amendment=amendment,
        task_manifest={"tasks": tasks},
        smoke=smoke,
        episodes=episodes,
        grading=grading,
    )
    assert not result["passed"]
