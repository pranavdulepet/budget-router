from __future__ import annotations

from budget_router.semantic_agent import SemanticAgentRouter

from tests.test_semantic_router import _artifact


def test_semantic_agent_routes_visible_prefix_across_arbitrary_pool() -> None:
    router = SemanticAgentRouter(_artifact())
    messages = [
        {"role": "system", "content": "You are a tool-using agent."},
        {"role": "user", "content": "Prove a hard theorem with a tool."},
    ]

    decision = router.select(
        messages,
        mode="balanced",
        required_capabilities={"tools"},
        output_tokens=200,
    )

    assert decision.selected_model == "strong"
    assert decision.selected_provider == "remote"
    assert decision.rejected["cheap"] == ("missing_capability",)
    assert decision.prefix_summary.step_index == 0
    assert decision.input_tokens_for_cost > 0
    assert decision.artifact_hash == _artifact()["artifact_hash"]
    assert router.route_counts == {"cheap": 0, "strong": 1}
    assert router.switch_count == 0


def test_semantic_agent_tracks_switches_and_can_override_token_estimate() -> None:
    router = SemanticAgentRouter(_artifact())
    messages = [{"role": "user", "content": "A short request"}]

    first = router.select(
        messages,
        mode="quality",
        allowed_models={"strong"},
        input_tokens=123,
    )
    second = router.select(
        messages,
        mode="cost",
        allowed_models={"cheap"},
        allow_ungated_policy=True,
    )

    assert first.input_tokens_for_cost == 123
    assert first.previous_model is None
    assert not first.switched
    assert second.step_index == 1
    assert second.previous_model == "strong"
    assert second.switched
    assert router.switch_count == 1
    assert router.route_counts == {"cheap": 1, "strong": 1}
