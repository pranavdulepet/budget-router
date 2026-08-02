from scripts.analyze_guarded_agent_step_followup import (
    _repository_cluster_bootstrap,
    _retrospective_oracle,
)


def test_repository_cluster_bootstrap_is_equal_weight_and_deterministic() -> None:
    task_ids = ["a1", "a2", "a3", "b1"]
    repositories = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    routed_quality = {"a1": 1, "a2": 1, "a3": 1, "b1": 0}
    fixed_quality = {"a1": 0, "a2": 0, "a3": 0, "b1": 1}
    routed_cost = {task_id: 0.5 for task_id in task_ids}
    fixed_cost = {task_id: 1.0 for task_id in task_ids}

    first = _repository_cluster_bootstrap(
        task_ids,
        repositories,
        routed_quality,
        fixed_quality,
        routed_cost,
        fixed_cost,
        samples=500,
        seed=7,
    )
    second = _repository_cluster_bootstrap(
        task_ids,
        repositories,
        routed_quality,
        fixed_quality,
        routed_cost,
        fixed_cost,
        samples=500,
        seed=7,
    )

    assert first == second
    assert first["point_quality_difference"] == 0.0
    assert first["point_cost_saving_fraction"] == 0.5
    assert first["repository_count"] == 2


def test_retrospective_oracle_chooses_cheapest_success_or_failure() -> None:
    policies = ("cheap", "strong")
    records = {
        ("a", "cheap"): {"conservative_cost_usd": "0.1"},
        ("a", "strong"): {"conservative_cost_usd": "0.5"},
        ("b", "cheap"): {"conservative_cost_usd": "0.2"},
        ("b", "strong"): {"conservative_cost_usd": "0.4"},
    }
    resolved = {"cheap": {"a"}, "strong": {"a"}}

    result = _retrospective_oracle(
        ["a", "b"],
        policies,
        records,
        resolved,
    )

    assert result["resolved_count"] == 1
    assert result["total_observed_cost_usd"] == "0.3"
    assert result["selected_policy_counts"] == {"cheap": 2}
