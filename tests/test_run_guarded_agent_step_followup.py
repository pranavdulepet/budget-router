from __future__ import annotations

from scripts.run_agent_step_study import FIXED_CHEAP, FIXED_STRONG
from scripts.run_guarded_agent_step_followup import (
    HELDOUT_POLICIES,
    ROUTED,
    _max_consecutive_cheap,
    _plan,
    _policy_order,
)


class _Decision:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def test_development_and_heldout_plans_are_disjoint_and_resume_safe() -> None:
    manifest = {
        "development": [{"task_id": "dev-1"}, {"task_id": "dev-2"}],
        "heldout": [{"task_id": "test-1"}],
    }
    records = [
        {
            "study_stage": "development",
            "task_id": "dev-1",
            "policy_id": ROUTED,
        }
    ]
    development = _plan(
        stage="development",
        manifest=manifest,
        records=records,
        max_task_blocks=None,
        treatment_order_seed=7,
    )
    heldout = _plan(
        stage="heldout",
        manifest=manifest,
        records=records,
        max_task_blocks=None,
        treatment_order_seed=7,
    )
    assert development == [{"task_id": "dev-2", "policies": [ROUTED]}]
    assert heldout[0]["task_id"] == "test-1"
    assert set(heldout[0]["policies"]) == set(HELDOUT_POLICIES)


def test_treatment_order_is_deterministic_and_complete() -> None:
    first = _policy_order("task", 20260729)
    assert first == _policy_order("task", 20260729)
    assert set(first) == {FIXED_CHEAP, FIXED_STRONG, ROUTED}


def test_maximum_cheap_burst() -> None:
    decisions = [
        _Decision("strong"),
        _Decision("cheap"),
        _Decision("cheap"),
        _Decision("strong"),
        _Decision("cheap"),
    ]
    assert _max_consecutive_cheap(decisions, "cheap") == 2
