from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from budget_router.mini_swe_tinker import TinkerCascadeMiniSweModel
from budget_router.types import TokenUsage


class _FakeModel:
    def __init__(self, name: str, *, hard_limit: str, cost: str) -> None:
        self.config = SimpleNamespace(
            model_name=name,
            price_snapshot="prices",
        )
        self.hard_limit_usd = Decimal(hard_limit)
        self.maximum_call_reservation_usd = Decimal("0.01")
        self.cost = Decimal(cost)
        self.spent_usd = Decimal("0")
        self.usage = TokenUsage()
        self.calls = 0
        self.provider_failures = 0
        self.provider_failure_reserved_usd = Decimal("0")

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        del messages
        self.calls += 1
        self.spent_usd += self.cost
        self.usage += TokenUsage(input_tokens=10, output_tokens=5)
        return {
            "role": "assistant",
            "content": "work",
            "tool_calls": [],
            "extra": {"actions": [], "cost": float(self.cost)},
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


def test_cascade_switches_after_fixed_scout_calls_with_visible_handoff() -> None:
    scout = _FakeModel("scout", hard_limit="0.06", cost="0.01")
    finisher = _FakeModel("finisher", hard_limit="0.90", cost="0.10")
    cascade = TinkerCascadeMiniSweModel(
        scout,  # type: ignore[arg-type]
        finisher,  # type: ignore[arg-type]
        total_hard_limit_usd=Decimal("0.90"),
        scout_max_calls=2,
    )
    messages: list[dict[str, Any]] = []

    assert cascade.query(messages)["extra"]["cascade_phase"] == "scout"
    assert cascade.query(messages)["extra"]["cascade_phase"] == "scout"
    response = cascade.query(messages)

    assert response["extra"]["cascade_phase"] == "finisher"
    assert messages[-1]["extra"]["cascade_handoff"] is True
    assert messages[-1]["extra"]["scout_calls"] == 2
    assert cascade.handoff_count == 1
    assert cascade.switch_reason == "scout_call_limit"
    assert cascade.spent_usd == Decimal("0.12")
    assert finisher.hard_limit_usd == Decimal("0.88")


def test_cascade_serialization_keeps_phase_costs_separate() -> None:
    scout = _FakeModel("scout", hard_limit="0.06", cost="0.01")
    finisher = _FakeModel("finisher", hard_limit="0.90", cost="0.10")
    cascade = TinkerCascadeMiniSweModel(
        scout,  # type: ignore[arg-type]
        finisher,  # type: ignore[arg-type]
        total_hard_limit_usd=Decimal("0.90"),
        scout_max_calls=1,
    )
    messages: list[dict[str, Any]] = []
    cascade.query(messages)
    cascade.query(messages)

    stats = cascade.serialize()["info"]["model_stats"]
    assert stats["scout_calls"] == 1
    assert stats["finisher_calls"] == 1
    assert stats["scout_spent_usd"] == "0.01"
    assert stats["finisher_spent_usd"] == "0.10"
