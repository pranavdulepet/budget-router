"""Tinker model implementation for mini-swe-agent's provider-neutral protocol."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .agent_step import FrozenAgentStepRouter
from .pricing import ModelPrice
from .providers import ModelRequest, ProviderError
from .providers.tinker import TinkerSamplingAdapter
from .redaction import redact
from .types import Operation, TokenUsage, TranscriptMessage

BASH_TOOL_SPEC = {
    "name": "bash",
    "description": "Execute one non-interactive bash command in the isolated workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True, slots=True)
class MiniSweTinkerConfig:
    model_name: str
    renderer: str
    reasoning: str
    temperature: float
    top_p: float = 1.0
    top_k: int = -1
    request_timeout_seconds: float | None = None
    max_output_tokens: int = 8_000
    price_snapshot: str = ""


def _cookbook_tool_call(value: Mapping[str, Any]) -> Any:
    from tinker_cookbook.renderers import ToolCall

    function = value.get("function", {})
    return ToolCall(
        type="function",
        id=value.get("id"),
        function={
            "name": str(function.get("name", "")),
            "arguments": str(function.get("arguments", "{}")),
        },
    )


def _renderer_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "exit":
            continue
        item: dict[str, Any] = {
            "role": role,
            "content": message.get("content", ""),
        }
        if calls := message.get("tool_calls"):
            item["tool_calls"] = [_cookbook_tool_call(call) for call in calls]
        for field in ("tool_call_id", "name"):
            if message.get(field):
                item[field] = message[field]
        history.append(item)
    return history


class TinkerMiniSweModel:
    """Synchronous mini-swe model facade over the asynchronous Tinker adapter."""

    def __init__(
        self,
        adapter: TinkerSamplingAdapter,
        *,
        model_name: str,
        renderer: str,
        reasoning: str,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = -1,
        request_timeout_seconds: float | None = None,
        seed: int,
        hard_limit_usd: Decimal,
        price: ModelPrice,
        price_snapshot: str,
        max_output_tokens: int = 8_000,
    ) -> None:
        self.adapter = adapter
        self.config = MiniSweTinkerConfig(
            model_name=model_name,
            renderer=renderer,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            request_timeout_seconds=request_timeout_seconds,
            max_output_tokens=max_output_tokens,
            price_snapshot=price_snapshot,
        )
        self.seed = seed
        self.hard_limit_usd = Decimal(hard_limit_usd)
        self.price = price
        self.spent_usd = Decimal("0")
        self.usage = TokenUsage()
        self.calls = 0
        self.provider_failures = 0
        self.provider_failure_reserved_usd = Decimal("0")

    @property
    def maximum_call_reservation_usd(self) -> Decimal:
        return self.price.conservative_cost(
            TokenUsage(
                input_tokens=self.price.context_tokens - self.config.max_output_tokens,
                output_tokens=self.config.max_output_tokens,
            )
        )

    def _require_reservation(self) -> None:
        remaining = self.hard_limit_usd - self.spent_usd
        if self.maximum_call_reservation_usd > remaining:
            from minisweagent.exceptions import LimitsExceeded

            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "BudgetLimitExceeded",
                    "extra": {
                        "exit_status": "BudgetLimitExceeded",
                        "submission": "",
                    },
                }
            )

    def query(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self._require_reservation()
        request = ModelRequest(
            model=self.config.model_name,
            messages=(TranscriptMessage("user", "Continue the coding task."),),
            operation=Operation.CONTINUE,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            timeout_seconds=self.config.request_timeout_seconds,
            seed=self.seed + self.calls,
            tools=(BASH_TOOL_SPEC,),
            metadata={"context_tokens": self.price.context_tokens},
        )
        try:
            response = asyncio.run(
                self.adapter._generate_from_messages(
                    request,
                    _renderer_history(messages),
                )
            )
        except ProviderError:
            self.provider_failures += 1
            # The provider may have completed or billed a timed-out request even
            # when usage was not returned. Charge the full preflight reservation
            # so timeout accounting remains conservative and inside the hard cap.
            reservation = self.maximum_call_reservation_usd
            self.spent_usd += reservation
            self.provider_failure_reserved_usd += reservation
            raise
        charge = self.price.conservative_cost(response.usage)
        if charge > self.maximum_call_reservation_usd:
            raise RuntimeError("Tinker response exceeded the conservative reservation")
        if self.spent_usd + charge > self.hard_limit_usd:
            raise RuntimeError("Tinker response would violate the hard episode cap")
        self.calls += 1
        self.spent_usd += charge
        self.usage += response.usage
        tool_calls = [
            {
                "type": "function",
                "id": call.call_id or f"bash-{self.calls}-{index}",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            }
            for index, call in enumerate(response.tool_calls)
        ]
        actions = [
            {
                "command": str(call.arguments["command"]),
                "tool_call_id": tool_calls[index]["id"],
            }
            for index, call in enumerate(response.tool_calls)
            if call.name == "bash" and isinstance(call.arguments.get("command"), str)
        ]
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": tool_calls,
            "extra": {
                "actions": actions,
                "cost": float(charge),
                "conservative_cost_usd": str(charge),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "latency_ms": response.latency_ms,
                "provider": dict(response.provider_metadata),
            },
        }

    def format_message(
        self,
        *,
        role: str,
        content: str,
        extra: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "content": content,
            **({"extra": dict(extra)} if extra else {}),
            **kwargs,
        }

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del template_vars
        actions = list(message.get("extra", {}).get("actions", ()))
        if not actions:
            return [
                {
                    "role": "user",
                    "content": (
                        "Tool call error: every response must call the bash tool "
                        'with a string argument named "command".'
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            ]
        result: list[dict[str, Any]] = []
        for action, output in zip(actions, outputs, strict=False):
            raw = str(output.get("output", ""))
            if len(raw) > 10_000:
                raw = raw[:5_000] + "\n[OUTPUT ELIDED]\n" + raw[-5_000:]
            content = (
                f"<returncode>{output.get('returncode', -1)}</returncode>\n"
                f"<output>{raw}</output>"
            )
            if output.get("exception_info"):
                content += f"\n<exception>{output['exception_info']}</exception>"
            result.append(
                {
                    "role": "tool",
                    "name": "bash",
                    "tool_call_id": action["tool_call_id"],
                    "content": content,
                    "extra": {
                        "returncode": output.get("returncode"),
                        "exception_info": output.get("exception_info", ""),
                    },
                }
            )
        return result

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "remaining_budget_usd": str(self.hard_limit_usd - self.spent_usd),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return redact(
            {
                "info": {
                    "config": {
                        "model": {
                            "model_name": self.config.model_name,
                            "renderer": self.config.renderer,
                            "reasoning": self.config.reasoning,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "top_k": self.config.top_k,
                            "request_timeout_seconds": (
                                self.config.request_timeout_seconds
                            ),
                            "max_output_tokens": self.config.max_output_tokens,
                            "price_snapshot": self.config.price_snapshot,
                        },
                        "model_type": (
                            "budget_router.mini_swe_tinker.TinkerMiniSweModel"
                        ),
                    },
                    "model_stats": {
                        "conservative_spent_usd": str(self.spent_usd),
                        "input_tokens": self.usage.input_tokens,
                        "output_tokens": self.usage.output_tokens,
                        "api_calls": self.calls,
                        "provider_failures": self.provider_failures,
                        "provider_failure_reserved_usd": str(
                            self.provider_failure_reserved_usd
                        ),
                    },
                }
            }
        )


class TinkerCascadeMiniSweModel:
    """Cheap-first model facade with a visible, same-workspace handoff."""

    def __init__(
        self,
        scout: TinkerMiniSweModel,
        finisher: TinkerMiniSweModel,
        *,
        total_hard_limit_usd: Decimal,
        scout_max_calls: int,
    ) -> None:
        if total_hard_limit_usd <= 0:
            raise ValueError("cascade total hard limit must be positive")
        if scout.hard_limit_usd >= total_hard_limit_usd:
            raise ValueError("scout sub-cap must leave budget for the finisher")
        if scout_max_calls <= 0:
            raise ValueError("scout call limit must be positive")
        self.scout = scout
        self.finisher = finisher
        self.total_hard_limit_usd = Decimal(total_hard_limit_usd)
        self.scout_max_calls = scout_max_calls
        self.phase = "scout"
        self.switch_reason: str | None = None
        self.handoff_count = 0

    @property
    def spent_usd(self) -> Decimal:
        return self.scout.spent_usd + self.finisher.spent_usd

    @property
    def usage(self) -> TokenUsage:
        return self.scout.usage + self.finisher.usage

    @property
    def calls(self) -> int:
        return self.scout.calls + self.finisher.calls

    @property
    def provider_failures(self) -> int:
        return self.scout.provider_failures + self.finisher.provider_failures

    @property
    def provider_failure_reserved_usd(self) -> Decimal:
        return (
            self.scout.provider_failure_reserved_usd
            + self.finisher.provider_failure_reserved_usd
        )

    def _scout_can_continue(self) -> bool:
        return (
            self.scout.calls < self.scout_max_calls
            and self.scout.spent_usd + self.scout.maximum_call_reservation_usd
            <= self.scout.hard_limit_usd
        )

    def _handoff(self, messages: list[dict[str, Any]], reason: str) -> None:
        if self.phase == "finisher":
            return
        self.phase = "finisher"
        self.switch_reason = reason
        self.handoff_count += 1
        messages.append(
            {
                "role": "user",
                "content": (
                    "Visible cascade handoff: the cheap scout phase is complete "
                    f"after {self.scout.calls} model calls and "
                    f"${self.scout.spent_usd} conservative spend "
                    f"(reason: {reason}). Continue as the strong finisher in the "
                    "same workspace. Inspect the existing diff and visible tool "
                    "evidence, finish the issue, verify it, and submit only the "
                    "intended source patch."
                ),
                "extra": {
                    "cascade_handoff": True,
                    "from_model": self.scout.config.model_name,
                    "to_model": self.finisher.config.model_name,
                    "reason": reason,
                    "scout_calls": self.scout.calls,
                    "scout_cost_usd": str(self.scout.spent_usd),
                },
            }
        )

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if self.phase == "scout" and self._scout_can_continue():
            try:
                response = self.scout.query(messages)
            except ProviderError:
                self._handoff(messages, "scout_provider_failure")
            else:
                response.setdefault("extra", {})["cascade_phase"] = "scout"
                response["extra"]["active_model"] = self.scout.config.model_name
                return response
        elif self.phase == "scout":
            reason = (
                "scout_call_limit"
                if self.scout.calls >= self.scout_max_calls
                else "scout_budget_reservation"
            )
            self._handoff(messages, reason)

        remaining = self.total_hard_limit_usd - self.scout.spent_usd
        if remaining <= 0:
            from minisweagent.exceptions import LimitsExceeded

            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "BudgetLimitExceeded",
                    "extra": {
                        "exit_status": "BudgetLimitExceeded",
                        "submission": "",
                    },
                }
            )
        self.finisher.hard_limit_usd = remaining
        response = self.finisher.query(messages)
        response.setdefault("extra", {})["cascade_phase"] = "finisher"
        response["extra"]["active_model"] = self.finisher.config.model_name
        if self.spent_usd > self.total_hard_limit_usd:
            raise RuntimeError("cascade exceeded its total hard limit")
        return response

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return self.scout.format_message(**kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        model = (
            self.scout
            if message.get("extra", {}).get("cascade_phase") == "scout"
            else self.finisher
        )
        return model.format_observation_messages(
            message,
            outputs,
            template_vars,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_name": (
                f"{self.scout.config.model_name}->{self.finisher.config.model_name}"
            ),
            "remaining_budget_usd": str(
                self.total_hard_limit_usd - self.spent_usd
            ),
            "cascade_phase": self.phase,
            "scout_calls_remaining": max(
                0, self.scout_max_calls - self.scout.calls
            ),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return redact(
            {
                "info": {
                    "config": {
                        "model": {
                            "policy": "cheap-first-visible-handoff-v1",
                            "scout_model": self.scout.config.model_name,
                            "finisher_model": self.finisher.config.model_name,
                            "scout_max_calls": self.scout_max_calls,
                            "scout_hard_limit_usd": str(
                                self.scout.hard_limit_usd
                            ),
                            "total_hard_limit_usd": str(
                                self.total_hard_limit_usd
                            ),
                            "price_snapshot": self.scout.config.price_snapshot,
                        },
                        "model_type": (
                            "budget_router.mini_swe_tinker."
                            "TinkerCascadeMiniSweModel"
                        ),
                    },
                    "model_stats": {
                        "conservative_spent_usd": str(self.spent_usd),
                        "input_tokens": self.usage.input_tokens,
                        "output_tokens": self.usage.output_tokens,
                        "api_calls": self.calls,
                        "provider_failures": self.provider_failures,
                        "provider_failure_reserved_usd": str(
                            self.provider_failure_reserved_usd
                        ),
                        "scout_calls": self.scout.calls,
                        "finisher_calls": self.finisher.calls,
                        "scout_spent_usd": str(self.scout.spent_usd),
                        "finisher_spent_usd": str(self.finisher.spent_usd),
                        "handoff_count": self.handoff_count,
                        "switch_reason": self.switch_reason,
                    },
                }
            }
        )


class TinkerAgentStepRouterMiniSweModel:
    """Per-call classifier router over two Tinker mini-swe model facades."""

    def __init__(
        self,
        cheap: TinkerMiniSweModel,
        strong: TinkerMiniSweModel,
        *,
        router: FrozenAgentStepRouter,
        total_hard_limit_usd: Decimal,
    ) -> None:
        if total_hard_limit_usd <= 0:
            raise ValueError("agent-step total hard limit must be positive")
        if cheap.config.model_name != router.artifact.cheap_model:
            raise ValueError("cheap model does not match the frozen router artifact")
        if strong.config.model_name != router.artifact.strong_model:
            raise ValueError("strong model does not match the frozen router artifact")
        self.cheap = cheap
        self.strong = strong
        self.router = router
        self.total_hard_limit_usd = Decimal(total_hard_limit_usd)

    @property
    def spent_usd(self) -> Decimal:
        return self.cheap.spent_usd + self.strong.spent_usd

    @property
    def usage(self) -> TokenUsage:
        return self.cheap.usage + self.strong.usage

    @property
    def calls(self) -> int:
        return self.cheap.calls + self.strong.calls

    @property
    def provider_failures(self) -> int:
        return self.cheap.provider_failures + self.strong.provider_failures

    @property
    def provider_failure_reserved_usd(self) -> Decimal:
        return (
            self.cheap.provider_failure_reserved_usd
            + self.strong.provider_failure_reserved_usd
        )

    def _remaining(self) -> Decimal:
        return self.total_hard_limit_usd - self.spent_usd

    def _prepare_call(self, model: TinkerMiniSweModel) -> None:
        remaining = self._remaining()
        if remaining <= 0 or model.maximum_call_reservation_usd > remaining:
            from minisweagent.exceptions import LimitsExceeded

            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "BudgetLimitExceeded",
                    "extra": {
                        "exit_status": "BudgetLimitExceeded",
                        "submission": "",
                    },
                }
            )
        # Each child tracks only its own spend. Give it precisely the shared
        # remainder so its existing reservation guard enforces the joint cap.
        model.hard_limit_usd = model.spent_usd + remaining

    def _query_selected(
        self,
        model: TinkerMiniSweModel,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._prepare_call(model)
        response = model.query(messages)
        if self.spent_usd > self.total_hard_limit_usd:
            raise RuntimeError("agent-step router exceeded its total hard limit")
        return response

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        decision = self.router.select(messages)
        selected = (
            self.cheap
            if decision.model_id == self.cheap.config.model_name
            else self.strong
        )
        cheap_failed_over = False
        try:
            response = self._query_selected(selected, messages)
        except ProviderError:
            if selected is not self.cheap:
                raise
            decision = self.router.force_failover_to_strong(decision)
            cheap_failed_over = True
            response = self._query_selected(self.strong, messages)

        extra = response.setdefault("extra", {})
        extra["active_model"] = decision.model_id
        extra["agent_step_router"] = {
            **decision.to_dict(),
            "artifact_hash": self.router.artifact.artifact_hash,
            "cheap_provider_failed_over": cheap_failed_over,
            "cumulative_cost_usd": str(self.spent_usd),
        }
        return response

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return self.cheap.format_message(**kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        active = message.get("extra", {}).get("active_model")
        model = self.cheap if active == self.cheap.config.model_name else self.strong
        return model.format_observation_messages(message, outputs, template_vars)

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_name": (
                f"agent-step:{self.cheap.config.model_name}|"
                f"{self.strong.config.model_name}"
            ),
            "remaining_budget_usd": str(self._remaining()),
            "router_artifact_hash": self.router.artifact.artifact_hash,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        route_counts = self.router.route_counts
        return redact(
            {
                "info": {
                    "config": {
                        "model": {
                            "policy": "frozen-agent-step-classifier-v1",
                            "cheap_model": self.cheap.config.model_name,
                            "strong_model": self.strong.config.model_name,
                            "total_hard_limit_usd": str(
                                self.total_hard_limit_usd
                            ),
                            "router_artifact_hash": (
                                self.router.artifact.artifact_hash
                            ),
                            "price_snapshot": self.cheap.config.price_snapshot,
                        },
                        "model_type": (
                            "budget_router.mini_swe_tinker."
                            "TinkerAgentStepRouterMiniSweModel"
                        ),
                    },
                    "model_stats": {
                        "conservative_spent_usd": str(self.spent_usd),
                        "input_tokens": self.usage.input_tokens,
                        "output_tokens": self.usage.output_tokens,
                        "api_calls": self.calls,
                        "provider_failures": self.provider_failures,
                        "provider_failure_reserved_usd": str(
                            self.provider_failure_reserved_usd
                        ),
                        "cheap_calls": self.cheap.calls,
                        "strong_calls": self.strong.calls,
                        "cheap_spent_usd": str(self.cheap.spent_usd),
                        "strong_spent_usd": str(self.strong.spent_usd),
                        "route_counts": route_counts,
                        "switch_count": self.router.switch_count,
                        "decisions": [
                            decision.to_dict()
                            for decision in self.router.decisions
                        ],
                    },
                }
            }
        )
