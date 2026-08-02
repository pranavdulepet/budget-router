from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from budget_router.agent_step import (
    FrozenAgentStepArtifact,
    FrozenAgentStepRouter,
    render_agent_prefix,
)
from budget_router.mini_swe_tinker import TinkerAgentStepRouterMiniSweModel
from budget_router.providers import ProviderError
from budget_router.types import TokenUsage


def _artifact(
    *,
    bias: float = -10.0,
    threshold: float = 0.5,
    context_guard: int = 24_000,
    dwell: int = 2,
) -> FrozenAgentStepArtifact:
    value = FrozenAgentStepArtifact(
        cheap_model="cheap",
        strong_model="strong",
        dimension=64,
        hash_seed="test",
        bias=bias,
        weights={},
        platt_slope=1.0,
        platt_intercept=0.0,
        cheap_threshold=threshold,
        force_strong_context_tokens=context_guard,
        strong_minimum_dwell_calls=dwell,
        metadata={"source": "unit-test"},
    )
    return FrozenAgentStepArtifact.from_dict(value.to_dict())


def test_router_visible_prefix_is_bounded_and_strips_private_blocks() -> None:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Fix parser.py. Why does it fail?"},
        {
            "role": "assistant",
            "content": "<think>private reasoning</think>Visible note",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pytest -q"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>1</returncode><output>failed</output>",
        },
    ]
    rendered, summary = render_agent_prefix(messages, step_index=1)

    assert "private reasoning" not in rendered
    assert "Visible note" in rendered
    assert "pytest -q" in rendered
    assert summary.tool_call_count == 1
    assert summary.tool_result_count == 1
    assert summary.recent_nonzero_returncodes == 1
    assert summary.has_code_hint
    assert summary.has_question_hint


def test_artifact_roundtrip_and_hash_validation() -> None:
    payload = _artifact().to_dict()
    loaded = FrozenAgentStepArtifact.from_dict(payload)
    assert loaded.cheap_model == "cheap"
    assert loaded.artifact_hash == payload["artifact_hash"]

    payload["policy"]["cheap_threshold"] = 0.8
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenAgentStepArtifact.from_dict(payload)


def test_context_guard_and_asymmetric_strong_dwell() -> None:
    router = FrozenAgentStepRouter(
        _artifact(context_guard=100, dwell=2)
    )
    first = router.select(
        [{"role": "user", "content": "x" * 800}],
        step_index=0,
    )
    second = router.select(
        [{"role": "user", "content": "small task"}],
        step_index=1,
    )
    third = router.select(
        [{"role": "user", "content": "small task"}],
        step_index=2,
    )

    assert first.reason == "force_strong_context_guard"
    assert second.reason == "force_strong_dwell"
    assert third.model_id == "cheap"
    assert router.switch_count == 1


class _FakeChild:
    def __init__(self, name: str, *, fail_first: bool = False) -> None:
        self.config = SimpleNamespace(model_name=name, price_snapshot="prices")
        self.hard_limit_usd = Decimal("1")
        self.maximum_call_reservation_usd = Decimal("0.05")
        self.spent_usd = Decimal("0")
        self.usage = TokenUsage()
        self.calls = 0
        self.provider_failures = 0
        self.provider_failure_reserved_usd = Decimal("0")
        self.fail_first = fail_first

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        del messages
        if self.fail_first:
            self.fail_first = False
            self.provider_failures += 1
            self.provider_failure_reserved_usd += Decimal("0.05")
            self.spent_usd += Decimal("0.05")
            raise ProviderError("injected")
        self.calls += 1
        self.spent_usd += Decimal("0.01")
        self.usage += TokenUsage(input_tokens=10, output_tokens=5)
        return {
            "role": "assistant",
            "content": "work",
            "tool_calls": [],
            "extra": {"actions": [], "cost": 0.01},
        }

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del message, outputs, template_vars
        return []


def test_tinker_wrapper_fails_over_and_preserves_shared_cap() -> None:
    cheap = _FakeChild("cheap", fail_first=True)
    strong = _FakeChild("strong")
    wrapper = TinkerAgentStepRouterMiniSweModel(
        cheap,  # type: ignore[arg-type]
        strong,  # type: ignore[arg-type]
        router=FrozenAgentStepRouter(_artifact()),
        total_hard_limit_usd=Decimal("0.20"),
    )
    response = wrapper.query([{"role": "user", "content": "small task"}])

    route = response["extra"]["agent_step_router"]
    assert route["model_id"] == "strong"
    assert route["cheap_provider_failed_over"] is True
    assert wrapper.spent_usd == Decimal("0.06")
    assert wrapper.provider_failures == 1
    assert wrapper.router.decisions[-1].reason == (
        "force_strong_cheap_provider_failure"
    )
