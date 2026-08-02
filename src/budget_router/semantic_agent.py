"""Provider-neutral, arbitrary-pool routing for individual agent LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .agent_step import AgentPrefixSummary, render_agent_prefix
from .semantic import SemanticRouteDecision, SemanticRouter


@dataclass(frozen=True, slots=True)
class SemanticAgentRouteDecision:
    """One auditable model choice made from the visible agent trajectory."""

    selected_model: str
    selected_provider: str
    step_index: int
    mode: str
    predicted_success: Mapping[str, float]
    expected_cost_usd: Mapping[str, float]
    utility: Mapping[str, float]
    rejected: Mapping[str, tuple[str, ...]]
    quality_floor: float | None
    policy_gate_passed: bool | None
    policy_gate_scope: str
    fallback_applied: bool
    artifact_hash: str
    prefix_summary: AgentPrefixSummary
    input_tokens_for_cost: int
    output_tokens_for_cost: int
    previous_model: str | None
    switched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "selected_provider": self.selected_provider,
            "step_index": self.step_index,
            "mode": self.mode,
            "predicted_success": dict(self.predicted_success),
            "expected_cost_usd": dict(self.expected_cost_usd),
            "utility": dict(self.utility),
            "rejected": {
                model: list(reasons) for model, reasons in self.rejected.items()
            },
            "quality_floor": self.quality_floor,
            "policy_gate_passed": self.policy_gate_passed,
            "policy_gate_scope": self.policy_gate_scope,
            "fallback_applied": self.fallback_applied,
            "artifact_hash": self.artifact_hash,
            "prefix_summary": self.prefix_summary.to_dict(),
            "input_tokens_for_cost": self.input_tokens_for_cost,
            "output_tokens_for_cost": self.output_tokens_for_cost,
            "previous_model": self.previous_model,
            "switched": self.switched,
        }


class SemanticAgentRouter:
    """Apply a trained arbitrary-model semantic router at every agent step.

    The classifier sees only the visible message/tool trajectory rendered by
    ``render_agent_prefix``. Provider clients remain outside this class.
    """

    def __init__(
        self,
        artifact_or_router: Mapping[str, Any] | SemanticRouter,
    ) -> None:
        self.semantic = (
            artifact_or_router
            if isinstance(artifact_or_router, SemanticRouter)
            else SemanticRouter(artifact_or_router)
        )
        self.decisions: list[SemanticAgentRouteDecision] = []

    @property
    def switch_count(self) -> int:
        return sum(decision.switched for decision in self.decisions)

    @property
    def route_counts(self) -> dict[str, int]:
        counts = {model: 0 for model in self.semantic.models}
        for decision in self.decisions:
            counts[decision.selected_model] += 1
        return counts

    def select(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        step_index: int | None = None,
        mode: str = "balanced",
        allowed_models: Iterable[str] | None = None,
        required_capabilities: Iterable[str] = (),
        denied_providers: Iterable[str] = (),
        input_tokens: int | None = None,
        output_tokens: int = 0,
        max_cost_usd: float | None = None,
        cost_weight: float | None = None,
        quality_threshold: float | None = None,
        max_quality_drop: float = 0.05,
        allow_ungated_policy: bool = False,
    ) -> SemanticAgentRouteDecision:
        """Choose a model for the next call without invoking any provider."""
        active_step = len(self.decisions) if step_index is None else step_index
        text, summary = render_agent_prefix(messages, step_index=active_step)
        tokens_for_cost = (
            summary.approximate_context_tokens
            if input_tokens is None
            else input_tokens
        )
        semantic_decision: SemanticRouteDecision = self.semantic.route(
            text,
            mode=mode,
            allowed_models=allowed_models,
            required_capabilities=required_capabilities,
            denied_providers=denied_providers,
            input_tokens=tokens_for_cost,
            output_tokens=output_tokens,
            max_cost_usd=max_cost_usd,
            cost_weight=cost_weight,
            quality_threshold=quality_threshold,
            max_quality_drop=max_quality_drop,
            allow_ungated_policy=allow_ungated_policy,
        )
        previous = (
            self.decisions[-1].selected_model if self.decisions else None
        )
        selected = semantic_decision.selected_model
        decision = SemanticAgentRouteDecision(
            selected_model=selected,
            selected_provider=self.semantic.cards[selected].provider,
            step_index=active_step,
            mode=semantic_decision.mode,
            predicted_success=semantic_decision.predicted_success,
            expected_cost_usd=semantic_decision.expected_cost_usd,
            utility=semantic_decision.utility,
            rejected=semantic_decision.rejected,
            quality_floor=semantic_decision.quality_floor,
            policy_gate_passed=semantic_decision.policy_gate_passed,
            policy_gate_scope=semantic_decision.policy_gate_scope,
            fallback_applied=semantic_decision.fallback_applied,
            artifact_hash=semantic_decision.artifact_hash,
            prefix_summary=summary,
            input_tokens_for_cost=tokens_for_cost,
            output_tokens_for_cost=output_tokens,
            previous_model=previous,
            switched=previous is not None and previous != selected,
        )
        self.decisions.append(decision)
        return decision
