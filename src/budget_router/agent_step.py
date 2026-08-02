"""Render the visible part of an agent trajectory for model selection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
_CODE_HINT = re.compile(
    r"(```|traceback|exception|error|def |class |function|test_|pytest|"
    r"\.py\b|\.js\b|\.ts\b|git diff|/testbed)",
    re.IGNORECASE,
)
_QUESTION_HINT = re.compile(r"\b(what|why|how|when|where|which|who)\b|\?", re.IGNORECASE)
_PRIVATE_BLOCKS = (
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<(?:reasoning|analysis)>.*?</(?:reasoning|analysis)>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"<(?:think|reasoning|analysis)>.*\Z", re.IGNORECASE | re.DOTALL),
)


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
    for pattern in _PRIVATE_BLOCKS:
        content = pattern.sub("", content)
    return content.strip()


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
        rendered.append(f"{name}({_bounded(arguments, 1_200)})")
    return "\n".join(rendered)


def approximate_context_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Return a conservative, tokenizer-independent context estimate."""
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
    """Convert visible messages and tool results into bounded classifier text."""
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
        recent_returncodes.extend(
            int(match) for match in _RETURNCODE.findall(_visible_content(message))
        )

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
    metadata = " ".join(
        (
            f"__step_bucket_{min(9, step_index // 5)}",
            f"__message_bucket_{min(15, len(messages) // 5)}",
            f"__context_bucket_{min(15, context_tokens // 2_000)}",
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
