from __future__ import annotations

from budget_router.semantic import (
    ModelCard,
    SemanticOutcome,
    SemanticRouter,
    train_semantic_router,
)
from budget_router.serialization import stable_hash


def _artifact() -> dict:
    cards = [
        ModelCard(
            "cheap",
            provider="local",
            expected_cost_usd=0.01,
            context_tokens=4_000,
            capabilities=frozenset({"text"}),
        ),
        ModelCard(
            "strong",
            provider="remote",
            expected_cost_usd=0.20,
            context_tokens=16_000,
            capabilities=frozenset({"text", "tools"}),
        ),
    ]
    rows = []
    for split, count in (("train", 12), ("calibration", 8)):
        for index in range(count):
            request_id = f"{split}-{index}"
            text = "simple greeting" if index % 2 == 0 else "prove a hard theorem"
            rows.extend(
                [
                    SemanticOutcome(
                        request_id,
                        text,
                        "cheap",
                        acceptable=index % 2 == 0,
                        split=split,
                        cost_usd=0.01,
                    ),
                    SemanticOutcome(
                        request_id,
                        text,
                        "strong",
                        acceptable=True,
                        split=split,
                        cost_usd=0.20,
                    ),
                ]
            )
    return train_semantic_router(rows, cards, dimension=128)


def test_semantic_artifact_is_hash_checked_and_routes_all_modes() -> None:
    artifact = _artifact()
    router = SemanticRouter(artifact)

    quality = router.route("prove a hard theorem", mode="quality")
    cost = router.route("simple greeting", mode="cost", max_quality_drop=0.2)
    constrained = router.route(
        "use a tool",
        mode="balanced",
        required_capabilities={"tools"},
    )

    assert quality.selected_model in {"cheap", "strong"}
    assert cost.expected_cost_usd["cheap"] < cost.expected_cost_usd["strong"]
    assert constrained.selected_model == "strong"
    assert constrained.rejected["cheap"] == ("missing_capability",)
    assert constrained.policy_gate_passed is True
    assert constrained.policy_gate_scope == "frozen_default_policy"
    assert constrained.fallback_applied is False
    assert quality.artifact_hash == artifact["artifact_hash"]
    assert artifact["model_head"]["kind"] in {
        "sklearn-hashing-linear-v1",
        "sklearn-shared-hashing-linear-v1",
    }
    assert {
        value["kind"] for value in artifact["calibration"].values()
    } == {"isotonic-linear-v1"}


def test_semantic_router_applies_context_and_provider_constraints() -> None:
    router = SemanticRouter(_artifact())
    context_decision = router.route(
        "long request",
        input_tokens=5_000,
    )
    provider_decision = router.route(
        "private request",
        denied_providers={"remote"},
    )

    assert context_decision.selected_model == "strong"
    assert context_decision.rejected["cheap"] == ("context_limit",)
    assert provider_decision.selected_model == "cheap"
    assert provider_decision.rejected["strong"] == ("provider_denied",)


def test_failed_calibration_gate_falls_back_to_best_fixed_model() -> None:
    artifact = _artifact()
    artifact.pop("artifact_hash")
    artifact["selector"]["gate_passed"] = False
    artifact["selector"]["calibration_best_fixed"] = "strong"
    artifact["artifact_hash"] = stable_hash(artifact)
    router = SemanticRouter(artifact)

    balanced = router.route("simple greeting", mode="balanced")
    cost = router.route("simple greeting", mode="cost")
    quality = router.route("simple greeting", mode="quality")
    assert balanced.selected_model == "strong"
    assert cost.selected_model == "strong"
    assert quality.selected_model == "strong"
    assert balanced.policy_gate_passed is False
    assert balanced.fallback_applied is True
    assert cost.policy_gate_passed is False
    assert cost.fallback_applied is True

    override = router.route(
        "simple greeting",
        mode="cost",
        allow_ungated_policy=True,
    )
    assert override.fallback_applied is False
    assert override.policy_gate_scope.endswith("_explicit_override")

    try:
        router.route(
            "simple greeting",
            mode="balanced",
            denied_providers={"remote"},
        )
    except ValueError as error:
        assert "fixed fallback is ineligible" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("an ineligible fixed fallback must fail closed")
