from scripts.freeze_quality_cost_tasks import select_development_tasks


def test_select_development_tasks_is_deterministic_and_excludes_ids() -> None:
    rows = [
        {"instance_id": f"repo__repo-{index}", "repo": "owner/repo"}
        for index in range(8)
    ]
    kwargs = {
        "records": rows,
        "excluded": {"repo__repo-0", "repo__repo-1"},
        "quotas": {"owner/repo": 4},
        "seed": "locked",
    }
    first = select_development_tasks(**kwargs)
    second = select_development_tasks(**kwargs)

    assert first == second
    assert len(first) == 4
    assert not {"repo__repo-0", "repo__repo-1"} & {
        row["task_id"] for row in first
    }
