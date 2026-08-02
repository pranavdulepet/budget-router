import budget_router


def test_public_api_contains_only_supported_router_interfaces() -> None:
    assert budget_router.__all__ == [
        "AgentPrefixSummary",
        "ModelCard",
        "SemanticAgentRouteDecision",
        "SemanticAgentRouter",
        "SemanticEvaluationOutcome",
        "SemanticOutcome",
        "SemanticRouteDecision",
        "SemanticRouter",
        "evaluate_semantic_router",
        "render_agent_prefix",
        "train_semantic_router",
    ]
