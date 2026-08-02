from scripts.freeze_agent_step_followup_tasks import select_tasks


def test_select_tasks_is_deterministic_and_respects_used_ids() -> None:
    records = [
        {"instance_id": f"repo__repo-{index}", "repo": "owner/repo"}
        for index in range(6)
    ]
    first = select_tasks(
        records,
        used={"repo__repo-0"},
        quotas={"owner/repo": 3},
        seed="frozen",
    )
    second = select_tasks(
        records,
        used={"repo__repo-0"},
        quotas={"owner/repo": 3},
        seed="frozen",
    )

    assert first == second
    assert len(first) == 3
    assert "repo__repo-0" not in {row["task_id"] for row in first}
