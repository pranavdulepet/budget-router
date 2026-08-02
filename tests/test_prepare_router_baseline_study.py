from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path

from scripts.prepare_router_baseline_study import prepare_study


def test_frozen_router_study_has_disjoint_balanced_cohorts() -> None:
    root = Path(__file__).parents[1]
    study = prepare_study(
        root / "data/swebench_verified_tasks.json",
        root / "artifacts/swebench_verified_split.json",
        root / "artifacts/pilot_tasks.json",
        root / "configs/model_pool_coding_v7.json",
        root / "outputs/pilot_coding_v6_final/results.json",
        task_selection_seed=20260727,
        episode_seed_base=20260726,
        episode_hard_cap_usd=Decimal("0.90"),
        working_limit_usd=Decimal("190"),
        absolute_limit_usd=Decimal("200"),
    )

    task_ids = [task["task_id"] for task in study["tasks"]]
    assert len(task_ids) == len(set(task_ids)) == 100
    assert Counter(task["split"] for task in study["tasks"]) == {
        "train": 60,
        "calibration": 20,
        "test": 20,
    }
    assert study["reused_pilot"]["episodes"] == 36
    assert (
        study["budget_projection"]["maximum_new_baseline_episodes_before_pruning"]
        == 364
    )
    assert (
        Decimal(
            study["budget_projection"][
                "expected_new_baseline_cost_usd_before_pruning"
            ]
        )
        < Decimal("180")
    )
    assert study["stages"]["cheap_backfill"]["new_episodes"] == 12
    assert study["stages"]["cheap_backfill"]["models"] == ["Qwen/Qwen3-8B"]
    assert study["stages"]["test"]["requires"] == "router_frozen"
    assert study["stages"]["test"]["models"] == "screen_survivors"

    screen_repositories = [
        task["repository"]
        for task in study["tasks"]
        if task["task_id"] in set(study["stages"]["screen"]["task_ids"])
    ]
    assert set(Counter(screen_repositories).values()) == {5}

    test_repositories = [
        task["repository"]
        for task in study["tasks"]
        if task["task_id"] in set(study["stages"]["test"]["task_ids"])
    ]
    assert set(Counter(test_repositories).values()) == {5}
