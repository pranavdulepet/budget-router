import budget_router


def test_public_api_preserves_supported_imports() -> None:
    assert budget_router.__all__ == [
        "AgentPrefixSummary",
        "ModelCard",
        "SemanticAgentRouteDecision",
        "SemanticAgentRouter",
        "SemanticEvaluationOutcome",
        "SemanticOutcome",
        "SemanticRouteDecision",
        "SemanticRouter",
        "approximate_context_tokens",
        "evaluate_semantic_router",
        "render_agent_prefix",
        "train_semantic_router",
    ]
