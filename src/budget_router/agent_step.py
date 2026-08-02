"""Portable, classifier-based routing for individual agent LLM calls."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .providers.base import strip_hidden_reasoning
from .serialization import stable_hash
from .task_model import hashed_text_features

_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
_CODE_HINT = re.compile(
    r"(```|traceback|exception|error|def |class |function|test_|pytest|"
    r"\.py\b|\.js\b|\.ts\b|git diff|/testbed)",
    re.IGNORECASE,
)
_QUESTION_HINT = re.compile(r"\b(what|why|how|when|where|which|who)\b|\?", re.IGNORECASE)


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n[ROUTER INPUT ELIDED]\n"
    available = max(0, limit - len(marker))
    left = available // 2
    return value[:left] + marker + value[-(available - left) :]


def _visible_content(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return strip_hidden_reasoning(content)


def _tool_call_text(message: Mapping[str, Any]) -> str:
    rendered: list[str] = []
    for call in message.get("tool_calls", ()) or ():
        if not isinstance(call, Mapping):
            continue
        function = call.get("function", {})
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", ""))
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        rendered.append(f"{name}({_bounded(arguments, 1200)})")
    return "\n".join(rendered)


def approximate_context_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Conservative tokenizer-independent context estimate used for guarding."""
    characters = 0
    for message in messages:
        characters += len(str(message.get("role", ""))) + len(_visible_content(message))
        characters += len(_tool_call_text(message))
    return max(1, math.ceil(characters / 4))


@dataclass(frozen=True, slots=True)
class AgentPrefixSummary:
    step_index: int
    message_count: int
    tool_call_count: int
    tool_result_count: int
    recent_nonzero_returncodes: int
    approximate_context_tokens: int
    original_request_chars: int
    recent_visible_chars: int
    has_code_hint: bool
    has_question_hint: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "tool_result_count": self.tool_result_count,
            "recent_nonzero_returncodes": self.recent_nonzero_returncodes,
            "approximate_context_tokens": self.approximate_context_tokens,
            "original_request_chars": self.original_request_chars,
            "recent_visible_chars": self.recent_visible_chars,
            "has_code_hint": self.has_code_hint,
            "has_question_hint": self.has_question_hint,
        }


def render_agent_prefix(
    messages: Sequence[Mapping[str, Any]],
    *,
    step_index: int,
) -> tuple[str, AgentPrefixSummary]:
    """Render a bounded, router-visible state for deterministic classification."""
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if not messages:
        raise ValueError("agent routing requires at least one visible message")

    first_user = next(
        (
            _visible_content(message)
            for message in messages
            if str(message.get("role", "")) == "user"
        ),
        "",
    )
    recent_parts: list[str] = []
    tool_calls = 0
    tool_results = 0
    recent_returncodes: list[int] = []
    for message in messages:
        calls = message.get("tool_calls", ()) or ()
        tool_calls += len(calls)
        if str(message.get("role", "")) == "tool":
            tool_results += 1
        for match in _RETURNCODE.findall(_visible_content(message)):
            recent_returncodes.append(int(match))

    for message in messages[-6:]:
        role = str(message.get("role", "unknown"))
        content = _bounded(_visible_content(message), 2_000)
        calls = _bounded(_tool_call_text(message), 1_500)
        if content:
            recent_parts.append(f"RECENT_ROLE_{role.upper()}\n{content}")
        if calls:
            recent_parts.append(f"RECENT_TOOL_CALL\n{calls}")

    recent = _bounded("\n\n".join(recent_parts), 8_000)
    original = _bounded(first_user, 8_000)
    context_tokens = approximate_context_tokens(messages)
    combined = f"{original}\n{recent}"
    summary = AgentPrefixSummary(
        step_index=step_index,
        message_count=len(messages),
        tool_call_count=tool_calls,
        tool_result_count=tool_results,
        recent_nonzero_returncodes=sum(code != 0 for code in recent_returncodes[-4:]),
        approximate_context_tokens=context_tokens,
        original_request_chars=len(first_user),
        recent_visible_chars=len(recent),
        has_code_hint=bool(_CODE_HINT.search(combined)),
        has_question_hint=bool(_QUESTION_HINT.search(original)),
    )
    relative_bucket = min(9, step_index // 5)
    context_bucket = min(15, context_tokens // 2_000)
    message_bucket = min(15, len(messages) // 5)
    metadata = " ".join(
        (
            f"__step_bucket_{relative_bucket}",
            f"__message_bucket_{message_bucket}",
            f"__context_bucket_{context_bucket}",
            f"__tool_calls_{min(20, tool_calls)}",
            f"__tool_results_{min(20, tool_results)}",
            f"__recent_failures_{summary.recent_nonzero_returncodes}",
            f"__has_code_{int(summary.has_code_hint)}",
            f"__has_question_{int(summary.has_question_hint)}",
        )
    )
    rendered = (
        f"ROUTER_METADATA {metadata}\n\n"
        f"RECENT_VISIBLE_STATE\n{recent}\n\n"
        f"ORIGINAL_USER_REQUEST\n{original}"
    )
    return rendered, summary


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True, slots=True)
class FrozenAgentStepArtifact:
    """JSON-portable binary tier classifier and frozen deployment policy."""

    cheap_model: str
    strong_model: str
    dimension: int
    hash_seed: str
    bias: float
    weights: Mapping[int, float]
    platt_slope: float
    platt_intercept: float
    cheap_threshold: float
    force_strong_context_tokens: int = 24_000
    strong_minimum_dwell_calls: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        if not self.cheap_model or not self.strong_model:
            raise ValueError("agent-step artifact requires two model IDs")
        if self.cheap_model == self.strong_model:
            raise ValueError("cheap and strong models must differ")
        if self.dimension <= 0 or self.force_strong_context_tokens <= 0:
            raise ValueError("artifact dimensions and context guard must be positive")
        if self.strong_minimum_dwell_calls <= 0:
            raise ValueError("strong dwell must be positive")
        if not 0.0 <= self.cheap_threshold <= 1.0:
            raise ValueError("cheap threshold must be in [0, 1]")

    def raw_probability_text(self, text: str) -> float:
        features = hashed_text_features(
            text,
            dimension=self.dimension,
            seed=self.hash_seed,
        )
        logit = self.bias + sum(
            float(self.weights.get(index, 0.0)) * value
            for index, value in features.items()
        )
        return _sigmoid(logit)

    def calibrated_probability_text(self, text: str) -> tuple[float, float]:
        raw = max(1e-6, min(1 - 1e-6, self.raw_probability_text(text)))
        raw_logit = math.log(raw) - math.log1p(-raw)
        calibrated = _sigmoid(self.platt_slope * raw_logit + self.platt_intercept)
        return raw, calibrated

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-step-router-artifact-v1",
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "feature_extractor": {
                "kind": "router-visible-prefix-v1",
                "dimension": self.dimension,
                "hash_seed": self.hash_seed,
            },
            "classifier": {
                "kind": "binary-hashed-logistic-v1",
                "bias": self.bias,
                "weights": {
                    str(index): value
                    for index, value in sorted(self.weights.items())
                    if value
                },
            },
            "calibration": {
                "kind": "platt-v1",
                "slope": self.platt_slope,
                "intercept": self.platt_intercept,
            },
            "policy": {
                "cheap_threshold": self.cheap_threshold,
                "force_strong_context_tokens": self.force_strong_context_tokens,
                "strong_minimum_dwell_calls": self.strong_minimum_dwell_calls,
            },
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["artifact_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenAgentStepArtifact:
        if value.get("schema_version") != "agent-step-router-artifact-v1":
            raise ValueError("unsupported agent-step artifact")
        claimed_hash = str(value.get("artifact_hash", ""))
        payload = dict(value)
        payload.pop("artifact_hash", None)
        actual_hash = stable_hash(payload)
        if claimed_hash != actual_hash:
            raise ValueError("agent-step artifact hash mismatch")
        features = value["feature_extractor"]
        classifier = value["classifier"]
        calibration = value["calibration"]
        policy = value["policy"]
        if features.get("kind") != "router-visible-prefix-v1":
            raise ValueError("unsupported agent-step feature extractor")
        if classifier.get("kind") != "binary-hashed-logistic-v1":
            raise ValueError("unsupported agent-step classifier")
        if calibration.get("kind") != "platt-v1":
            raise ValueError("unsupported agent-step calibrator")
        return cls(
            cheap_model=str(value["cheap_model"]),
            strong_model=str(value["strong_model"]),
            dimension=int(features["dimension"]),
            hash_seed=str(features["hash_seed"]),
            bias=float(classifier["bias"]),
            weights={
                int(index): float(weight)
                for index, weight in classifier["weights"].items()
            },
            platt_slope=float(calibration["slope"]),
            platt_intercept=float(calibration["intercept"]),
            cheap_threshold=float(policy["cheap_threshold"]),
            force_strong_context_tokens=int(
                policy["force_strong_context_tokens"]
            ),
            strong_minimum_dwell_calls=int(
                policy["strong_minimum_dwell_calls"]
            ),
            metadata=dict(value.get("metadata", {})),
            artifact_hash=claimed_hash,
        )


@dataclass(frozen=True, slots=True)
class AgentStepDecision:
    step_index: int
    model_id: str
    predicted_model_id: str
    raw_needs_strong_probability: float
    calibrated_needs_strong_probability: float
    cheap_threshold: float
    reason: str
    forced: bool
    feature_summary: AgentPrefixSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "model_id": self.model_id,
            "predicted_model_id": self.predicted_model_id,
            "raw_needs_strong_probability": self.raw_needs_strong_probability,
            "calibrated_needs_strong_probability": (
                self.calibrated_needs_strong_probability
            ),
            "cheap_threshold": self.cheap_threshold,
            "reason": self.reason,
            "forced": self.forced,
            "feature_summary": self.feature_summary.to_dict(),
        }


class FrozenAgentStepRouter:
    """Stateful deployment policy with conservative context and dwell guards."""

    def __init__(self, artifact: FrozenAgentStepArtifact) -> None:
        self.artifact = artifact
        self.decisions: list[AgentStepDecision] = []
        self._strong_dwell_remaining = 0

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
            reason = "classifier_cheap" if model_id == self.artifact.cheap_model else "classifier_strong"
            forced = False
            if (
                summary.approximate_context_tokens
                >= self.artifact.force_strong_context_tokens
            ):
                model_id = self.artifact.strong_model
                reason = "force_strong_context_guard"
                forced = model_id != predicted
            elif (
                predicted == self.artifact.cheap_model
                and self.decisions
                and self.decisions[-1].model_id == self.artifact.strong_model
                and self._strong_dwell_remaining > 0
            ):
                model_id = self.artifact.strong_model
                reason = "force_strong_dwell"
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

        previous = self.decisions[-1].model_id if self.decisions else None
        if model_id == self.artifact.strong_model:
            if previous != self.artifact.strong_model:
                self._strong_dwell_remaining = max(
                    0, self.artifact.strong_minimum_dwell_calls - 1
                )
            elif self._strong_dwell_remaining:
                self._strong_dwell_remaining -= 1
        else:
            self._strong_dwell_remaining = 0

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

    def force_failover_to_strong(self, decision: AgentStepDecision) -> AgentStepDecision:
        """Record a cheap-call provider failure as a strong retry decision."""
        if decision.model_id != self.artifact.cheap_model:
            raise ValueError("only a cheap decision can fail over to strong")
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
        self._strong_dwell_remaining = max(
            0, self.artifact.strong_minimum_dwell_calls - 1
        )
        return failover


def decimal_cost(value: Decimal | str | float) -> Decimal:
    """Normalize public wrapper inputs without accepting NaN/Infinity."""
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("cost must be finite and non-negative")
    return result
