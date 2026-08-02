from __future__ import annotations

from scripts.run_isolated_stage_test_blocks import next_test_block


def test_next_test_block_resumes_missing_policy() -> None:
    protocol = {
        "fixed_policy": {"policy_id": "fixed"},
        "stages": {"test": {"task_ids": ["task-1", "task-2"]}},
    }
    freeze = {"selected_candidate_policy_id": "candidate"}
    records = [
        {
            "study_stage": "test",
            "task_id": "task-1",
            "model": "candidate",
        }
    ]

    assert next_test_block(protocol, freeze, records) == (
        "task-1",
        ["fixed"],
    )
    records.append(
        {"study_stage": "test", "task_id": "task-1", "model": "fixed"}
    )
    assert next_test_block(protocol, freeze, records) == (
        "task-2",
        ["candidate", "fixed"],
    )
