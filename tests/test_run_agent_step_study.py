from __future__ import annotations

import json
from pathlib import Path

from budget_router.serialization import read_jsonl
from scripts.run_agent_step_study import (
    FIXED_CHEAP,
    FIXED_STRONG,
    ROUTED,
    _plan,
    _policy_order,
    _recover_episode_shards,
)


def test_policy_order_is_deterministic_and_complete() -> None:
    first = _policy_order("repo__repo-1", 20260729)
    second = _policy_order("repo__repo-1", 20260729)

    assert first == second
    assert set(first) == {FIXED_CHEAP, FIXED_STRONG, ROUTED}


def test_plan_resumes_within_a_task_block() -> None:
    manifest = {"heldout": [{"task_id": "repo__repo-1"}]}
    records = [
        {
            "study_stage": "heldout",
            "task_id": "repo__repo-1",
            "policy_id": FIXED_CHEAP,
        }
    ]

    blocks = _plan(
        stage="heldout",
        manifest=manifest,
        records=records,
        max_task_blocks=None,
        treatment_order_seed=20260729,
    )

    assert len(blocks) == 1
    assert FIXED_CHEAP not in blocks[0]["policies"]
    assert set(blocks[0]["policies"]) == {FIXED_STRONG, ROUTED}


def test_completed_episode_shard_is_recovered_once(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    shard = (
        output_root
        / "episode_shards"
        / "heldout"
        / "repo__repo-1"
        / "fixed.json"
    )
    shard.parent.mkdir(parents=True)
    row = {
        "study_stage": "heldout",
        "task_id": "repo__repo-1",
        "policy_id": FIXED_CHEAP,
        "protocol_sha256": "protocol",
        "router_artifact_hash": "artifact",
        "task_manifest_hash": "tasks",
        "conservative_cost_usd": "0.1",
    }
    shard.write_text(json.dumps(row), encoding="utf-8")
    records_path = output_root / "dynamic_episodes.jsonl"
    records: list[dict] = []

    recovered = _recover_episode_shards(
        output_root=output_root,
        records_path=records_path,
        records=records,
        expected_protocol_sha256="protocol",
        expected_artifact_hash="artifact",
        expected_task_manifest_hash="tasks",
    )
    recovered_again = _recover_episode_shards(
        output_root=output_root,
        records_path=records_path,
        records=records,
        expected_protocol_sha256="protocol",
        expected_artifact_hash="artifact",
        expected_task_manifest_hash="tasks",
    )

    assert recovered == 1
    assert recovered_again == 0
    assert read_jsonl(records_path) == [row]
