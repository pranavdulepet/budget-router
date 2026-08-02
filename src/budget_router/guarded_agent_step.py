"""Guarded deployment policy for the live-calibrated agent-step router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .agent_step import (
    AgentPrefixSummary,
    AgentStepDecision,
    FrozenAgentStepArtifact,
    render_agent_prefix,
)
from .serialization import stable_hash


@dataclass(frozen=True, slots=True)
class GuardedAgentStepArtifact:
    """Portable classifier plus deterministic activation guards."""

    base_classifier: FrozenAgentStepArtifact
    force_strong_initial_calls: int = 2
    maximum_consecutive_cheap_calls: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        if self.force_strong_initial_calls < 0:
            raise ValueError("initial strong-call guard must be non-negative")
        if self.maximum_consecutive_cheap_calls <= 0:
            raise ValueError("maximum consecutive cheap calls must be positive")

    @property
    def cheap_model(self) -> str:
        return self.base_classifier.cheap_model

    @property
    def strong_model(self) -> str:
        return self.base_classifier.strong_model

    @property
    def cheap_threshold(self) -> float:
        return self.base_classifier.cheap_threshold

    @property
    def force_strong_context_tokens(self) -> int:
        return self.base_classifier.force_strong_context_tokens

    @property
    def strong_minimum_dwell_calls(self) -> int:
        return self.base_classifier.strong_minimum_dwell_calls

    def calibrated_probability_text(self, text: str) -> tuple[float, float]:
        return self.base_classifier.calibrated_probability_text(text)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "guarded-agent-step-router-artifact-v2",
            "base_classifier": self.base_classifier.to_dict(),
            "guards": {
                "force_strong_initial_calls": self.force_strong_initial_calls,
                "maximum_consecutive_cheap_calls": (
                    self.maximum_consecutive_cheap_calls
                ),
            },
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["artifact_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GuardedAgentStepArtifact:
        if value.get("schema_version") != "guarded-agent-step-router-artifact-v2":
            raise ValueError("unsupported guarded agent-step artifact")
        claimed_hash = str(value.get("artifact_hash", ""))
        payload = dict(value)
        payload.pop("artifact_hash", None)
        if claimed_hash != stable_hash(payload):
            raise ValueError("guarded agent-step artifact hash mismatch")
        guards = value["guards"]
        return cls(
            base_classifier=FrozenAgentStepArtifact.from_dict(
                value["base_classifier"]
            ),
            force_strong_initial_calls=int(
                guards["force_strong_initial_calls"]
            ),
            maximum_consecutive_cheap_calls=int(
                guards["maximum_consecutive_cheap_calls"]
            ),
            metadata=dict(value.get("metadata", {})),
            artifact_hash=claimed_hash,
        )


class GuardedAgentStepRouter:
    """Stateful classifier router with initial, context, dwell, and burst guards."""

    def __init__(self, artifact: GuardedAgentStepArtifact) -> None:
        self.artifact = artifact
        self.decisions: list[AgentStepDecision] = []
        self._strong_dwell_remaining = 0
        self._consecutive_cheap = 0

    def select(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        step_index: int | None = None,
    ) -> AgentStepDecision:
        current_step = len(self.decisions) if step_index is None else step_index
        try:
            text, summary = render_agent_prefix(messages, step_index=current_step)
            raw, calibrated = self.artifact.calibrated_probability_text(text)
            predicted = (
                self.artifact.cheap_model
                if calibrated <= self.artifact.cheap_threshold
                else self.artifact.strong_model
            )
            model_id = predicted
            reason = (
                "classifier_cheap"
                if predicted == self.artifact.cheap_model
                else "classifier_strong"
            )
            forced = False
            if len(self.decisions) < self.artifact.force_strong_initial_calls:
                model_id = self.artifact.strong_model
                reason = "force_strong_initial_calls"
                forced = True
            elif (
                summary.approximate_context_tokens
                >= self.artifact.force_strong_context_tokens
            ):
                model_id = self.artifact.strong_model
                reason = "force_strong_context_guard"
                forced = True
            elif (
                predicted == self.artifact.cheap_model
                and self.decisions
                and self.decisions[-1].model_id == self.artifact.strong_model
                and self._strong_dwell_remaining > 0
            ):
                model_id = self.artifact.strong_model
                reason = "force_strong_dwell"
                forced = True
            elif (
                predicted == self.artifact.cheap_model
                and self._consecutive_cheap
                >= self.artifact.maximum_consecutive_cheap_calls
            ):
                model_id = self.artifact.strong_model
                reason = "force_strong_cheap_burst"
                forced = True
        except Exception:
            summary = AgentPrefixSummary(
                step_index=current_step,
                message_count=len(messages),
                tool_call_count=0,
                tool_result_count=0,
                recent_nonzero_returncodes=0,
                approximate_context_tokens=1,
                original_request_chars=0,
                recent_visible_chars=0,
                has_code_hint=False,
                has_question_hint=False,
            )
            raw = calibrated = 1.0
            predicted = model_id = self.artifact.strong_model
            reason = "force_strong_input_validation"
            forced = True

        self._update_guard_state(model_id)
        decision = AgentStepDecision(
            step_index=current_step,
            model_id=model_id,
            predicted_model_id=predicted,
            raw_needs_strong_probability=raw,
            calibrated_needs_strong_probability=calibrated,
            cheap_threshold=self.artifact.cheap_threshold,
            reason=reason,
            forced=forced,
            feature_summary=summary,
        )
        self.decisions.append(decision)
        return decision

    def _update_guard_state(self, model_id: str) -> None:
        previous = self.decisions[-1].model_id if self.decisions else None
        if model_id == self.artifact.strong_model:
            self._consecutive_cheap = 0
            if previous != self.artifact.strong_model:
                self._strong_dwell_remaining = max(
                    0, self.artifact.strong_minimum_dwell_calls - 1
                )
            elif self._strong_dwell_remaining:
                self._strong_dwell_remaining -= 1
        else:
            self._consecutive_cheap += 1
            self._strong_dwell_remaining = 0

    @property
    def route_counts(self) -> dict[str, int]:
        return {
            self.artifact.cheap_model: sum(
                decision.model_id == self.artifact.cheap_model
                for decision in self.decisions
            ),
            self.artifact.strong_model: sum(
                decision.model_id == self.artifact.strong_model
                for decision in self.decisions
            ),
        }

    @property
    def switch_count(self) -> int:
        return sum(
            left.model_id != right.model_id
            for left, right in zip(self.decisions, self.decisions[1:], strict=False)
        )

    def force_failover_to_strong(
        self, decision: AgentStepDecision
    ) -> AgentStepDecision:
        """Replace a failed cheap call with the strong retry in router state."""
        if decision.model_id != self.artifact.cheap_model:
            raise ValueError("only a cheap decision can fail over to strong")
        if not self.decisions or self.decisions[-1] is not decision:
            raise ValueError("failover decision must be the latest route")
        failover = AgentStepDecision(
            step_index=decision.step_index,
            model_id=self.artifact.strong_model,
            predicted_model_id=decision.predicted_model_id,
            raw_needs_strong_probability=decision.raw_needs_strong_probability,
            calibrated_needs_strong_probability=(
                decision.calibrated_needs_strong_probability
            ),
            cheap_threshold=decision.cheap_threshold,
            reason="force_strong_cheap_provider_failure",
            forced=True,
            feature_summary=decision.feature_summary,
        )
        self.decisions[-1] = failover
        self._consecutive_cheap = 0
        self._strong_dwell_remaining = max(
            0, self.artifact.strong_minimum_dwell_calls - 1
        )
        return failover
