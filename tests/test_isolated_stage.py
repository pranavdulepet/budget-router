from __future__ import annotations

import json
from decimal import Decimal

import pytest

from budget_router.isolated_stage import (
    build_visible_diagnostic_handoff,
    canonicalize_infrastructure_recoveries,
    is_zero_call_infrastructure_failure,
    isolated_policy_id,
    plan_stage_episodes,
    remaining_finisher_cap,
)


def test_visible_handoff_contains_only_bounded_sanitized_evidence() -> None:
    messages = [
        {"role": "system", "content": "private harness prompt"},
        {"role": "user", "content": "original issue text"},
        {
            "role": "assistant",
            "content": "<think>private reasoning</think>Inspecting the parser.",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "rg 'Parser' src"}
                        ),
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode>\n<output>src/parser.py</output>",
        },
        {
            "role": "user",
            "content": "Tool call error",
            "extra": {"interrupt_type": "FormatError"},
        },
        {"role": "exit", "content": "Submitted"},
    ]

    handoff = build_visible_diagnostic_handoff(messages, max_chars=10_000)

    assert "Inspecting the parser." in handoff
    assert "rg 'Parser' src" in handoff
    assert "src/parser.py" in handoff
    assert "Tool call error" in handoff
    assert "private reasoning" not in handoff
    assert "private harness prompt" not in handoff
    assert "original issue text" not in handoff
    assert "Submitted" not in handoff


def test_visible_handoff_is_deterministically_bounded() -> None:
    messages = [{"role": "tool", "content": "x" * 1_000}]

    first = build_visible_diagnostic_handoff(messages, max_chars=120)
    second = build_visible_diagnostic_handoff(messages, max_chars=120)

    assert first == second
    assert len(first) == 120
    assert "[VISIBLE HANDOFF ELIDED]" in first


def test_remaining_finisher_cap_uses_actual_scout_spend() -> None:
    assert remaining_finisher_cap(
        Decimal("0.90"),
        Decimal("0.037"),
    ) == Decimal("0.863")
    with pytest.raises(ValueError, match="exceeds"):
        remaining_finisher_cap(Decimal("0.90"), Decimal("0.91"))


def test_canonicalize_infrastructure_recovery_preserves_audit_overlay() -> None:
    failed = {
        "study_manifest_hash": "freeze",
        "study_stage": "expand",
        "task_id": "task-1",
        "model": "candidate",
        "model_calls": 0,
        "conservative_cost_usd": "0",
        "submitted_patch": False,
        "provider_failed": False,
        "structurally_valid": False,
        "error_type": "CalledProcessError",
    }
    recovered = {
        **failed,
        "model_calls": 4,
        "conservative_cost_usd": "0.4",
        "structurally_valid": True,
        "error_type": None,
    }

    assert is_zero_call_infrastructure_failure(failed)
    assert not is_zero_call_infrastructure_failure(recovered)
    assert canonicalize_infrastructure_recoveries(
        [failed],
        [recovered],
    ) == [
        {
            **recovered,
            "infrastructure_recovery": True,
            "replaces_error_type": "CalledProcessError",
        }
    ]


def test_stage_plan_is_append_only_and_reserves_every_policy() -> None:
    policies = [
        isolated_policy_id("cheap-a", "strong"),
        isolated_policy_id("cheap-b", "strong"),
    ]
    record = {
        "study_manifest_hash": "freeze",
        "study_stage": "screen",
        "task_id": "task-1",
        "model": policies[0],
    }

    plan = plan_stage_episodes(
        task_ids=["task-1", "task-2"],
        policy_ids=policies,
        records=[record],
        study_manifest_hash="freeze",
        stage="screen",
        hard_cap_usd=Decimal("0.90"),
    )

    assert plan["pending_episode_count"] == 3
    assert plan["maximum_pending_cost_usd"] == Decimal("2.70")
    assert plan["pending_blocks"][0] == {
        "task_id": "task-1",
        "policy_ids": [policies[1]],
    }
