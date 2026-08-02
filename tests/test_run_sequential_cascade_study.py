from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.run_sequential_cascade_study import (
    cascade_exposure,
    load_protocol,
    plan_cascade,
)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "source_study": root / "artifacts/router_baseline_study_v2.json",
        "router_freeze": root / "outputs/router_baseline_v2/router_freeze.json",
        "baseline_records": root / "outputs/router_baseline_v2/episodes.jsonl",
        "provider_hangs": root / "outputs/router_baseline_v2/provider_hangs.jsonl",
        "comparator_report": (
            root
            / "outputs/router_baseline_v2/swebench_grader/test"
            / "Qwen__Qwen3.6-35B-A3B.router-v2-test-qwen36-pinned.json"
        ),
        "model_pool": root / "configs/model_pool_coding_v7.json",
        "prices": root / "configs/tinker_prices_2026-07-27_coding_v3.json",
        "harness_prompt": root / "configs/harness_prompt.txt",
        "tasks": root / "data/swebench_verified_tasks.json",
    }


def test_full_cascade_plan_reserves_all_twenty_paired_episodes() -> None:
    root = Path(__file__).parents[1]
    protocol = load_protocol(
        root / "artifacts/sequential_cascade_study_v1.json",
        input_paths=_paths(root),
    )

    plan = plan_cascade(protocol, [], max_tasks=None)

    assert plan["pending_episode_count"] == 20
    assert plan["maximum_pending_cost_usd"] == Decimal("18.00")
    assert plan["projected_total_exposure_usd"] == Decimal("177.608486065")


def test_cascade_plan_resumes_without_repeating_tasks() -> None:
    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "artifacts/sequential_cascade_study_v1.json").read_text()
    )
    record = {
        "study_manifest_hash": protocol["manifest_hash"],
        "model": protocol["cascade_policy"]["policy_id"],
        "task_id": protocol["task_ids"][0],
        "conservative_cost_usd": "0.20",
    }

    plan = plan_cascade(protocol, [record], max_tasks=2)

    assert plan["pending_episode_count"] == 2
    assert protocol["task_ids"][0] not in plan["pending_task_ids"]
    assert cascade_exposure([record]) == Decimal("0.20")
