from __future__ import annotations

import copy

import pytest

from budget_router.agent_step import FrozenAgentStepArtifact
from budget_router.guarded_agent_step import (
    GuardedAgentStepArtifact,
    GuardedAgentStepRouter,
)


def _artifact(
    *,
    bias: float = -10.0,
    threshold: float = 0.5,
    initial: int = 2,
    burst: int = 2,
    dwell: int = 2,
    context_guard: int = 24_000,
) -> GuardedAgentStepArtifact:
    base = FrozenAgentStepArtifact(
        cheap_model="cheap",
        strong_model="strong",
        dimension=64,
        hash_seed="guarded-test",
        bias=bias,
        weights={},
        platt_slope=1.0,
        platt_intercept=0.0,
        cheap_threshold=threshold,
        force_strong_context_tokens=context_guard,
        strong_minimum_dwell_calls=dwell,
        metadata={"source": "unit-test"},
    )
    value = GuardedAgentStepArtifact(
        base_classifier=FrozenAgentStepArtifact.from_dict(base.to_dict()),
        force_strong_initial_calls=initial,
        maximum_consecutive_cheap_calls=burst,
        metadata={"amendment": "008"},
    )
    return GuardedAgentStepArtifact.from_dict(value.to_dict())


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "small task"}]


def test_guarded_artifact_roundtrip_and_nested_hash_validation() -> None:
    payload = _artifact().to_dict()
    loaded = GuardedAgentStepArtifact.from_dict(payload)
    assert loaded.cheap_model == "cheap"
    assert loaded.artifact_hash == payload["artifact_hash"]

    outer_tampered = copy.deepcopy(payload)
    outer_tampered["guards"]["maximum_consecutive_cheap_calls"] = 3
    with pytest.raises(ValueError, match="hash mismatch"):
        GuardedAgentStepArtifact.from_dict(outer_tampered)

    nested_tampered = copy.deepcopy(payload)
    nested_tampered["base_classifier"]["classifier"]["bias"] = 5
    nested_tampered.pop("artifact_hash")
    from budget_router.serialization import stable_hash

    nested_tampered["artifact_hash"] = stable_hash(
        {key: value for key, value in nested_tampered.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        GuardedAgentStepArtifact.from_dict(nested_tampered)


def test_initial_and_cheap_burst_guards_bound_every_trajectory() -> None:
    router = GuardedAgentStepRouter(_artifact())
    decisions = [
        router.select(_messages(), step_index=step)
        for step in range(8)
    ]

    assert [decision.model_id for decision in decisions] == [
        "strong",
        "strong",
        "cheap",
        "cheap",
        "strong",
        "strong",
        "cheap",
        "cheap",
    ]
    assert decisions[0].reason == "force_strong_initial_calls"
    assert decisions[1].reason == "force_strong_initial_calls"
    assert decisions[4].reason == "force_strong_cheap_burst"
    assert decisions[5].reason == "force_strong_dwell"
    assert router.route_counts == {"cheap": 4, "strong": 4}
    assert router.switch_count == 3


def test_context_guard_precedes_burst_and_resets_cheap_run() -> None:
    router = GuardedAgentStepRouter(
        _artifact(initial=0, burst=2, context_guard=100)
    )
    assert router.select(_messages()).model_id == "cheap"
    guarded = router.select([{"role": "user", "content": "x" * 800}])
    dwell = router.select(_messages())
    resumed = router.select(_messages())

    assert guarded.reason == "force_strong_context_guard"
    assert dwell.reason == "force_strong_dwell"
    assert resumed.model_id == "cheap"


def test_failover_replaces_latest_cheap_and_restarts_strong_dwell() -> None:
    router = GuardedAgentStepRouter(_artifact(initial=0))
    cheap = router.select(_messages())
    failover = router.force_failover_to_strong(cheap)
    next_decision = router.select(_messages())

    assert failover.model_id == "strong"
    assert next_decision.reason == "force_strong_dwell"
    assert router.route_counts == {"cheap": 0, "strong": 2}


def test_invalid_input_fails_closed_to_strong() -> None:
    router = GuardedAgentStepRouter(_artifact(initial=0))
    decision = router.select([])
    assert decision.model_id == "strong"
    assert decision.reason == "force_strong_input_validation"
