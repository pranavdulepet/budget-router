from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..types import Operation, TokenUsage, TranscriptMessage


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    messages: tuple[TranscriptMessage, ...]
    operation: Operation
    max_output_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    timeout_seconds: float | None = None
    seed: int | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    renderer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not self.messages:
            raise ValueError("model and messages are required")
        if not 0 <= self.max_output_tokens <= 8_000:
            raise ValueError("max_output_tokens must be in [0, 8000]")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model: str
    content: str
    usage: TokenUsage
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    latency_ms: int = 0
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if _contains_hidden_reasoning(self.content):
            raise ValueError("ModelResponse.content must not contain hidden reasoning")


class ModelProvider(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_REASONING_BLOCK = re.compile(
    r"<(?:reasoning|analysis)>.*?</(?:reasoning|analysis)>",
    re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_PRIVATE_BLOCK = re.compile(
    r"<(?:think|reasoning|analysis)>.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _contains_hidden_reasoning(text: str) -> bool:
    return bool(
        _THINK_BLOCK.search(text)
        or _REASONING_BLOCK.search(text)
        or _UNCLOSED_PRIVATE_BLOCK.search(text)
    )


def strip_hidden_reasoning(text: str) -> str:
    """Remove model-private reasoning before handoff, state storage, or publication."""
    text = _THINK_BLOCK.sub("", text)
    text = _REASONING_BLOCK.sub("", text)
    text = _UNCLOSED_PRIVATE_BLOCK.sub("", text)
    return text.strip()


class MockProvider:
    """Deterministic provider for protocol and integration tests."""

    def __init__(
        self,
        responses: Sequence[str] = ("Done.",),
        *,
        input_tokens: int = 100,
        output_tokens: int = 20,
        fail_on_calls: Sequence[int] = (),
    ) -> None:
        self._responses = deque(responses)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.fail_on_calls = set(fail_on_calls)
        self.calls: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        call_number = len(self.calls)
        if call_number in self.fail_on_calls:
            raise ProviderError(f"injected provider failure on call {call_number}")
        raw = self._responses.popleft() if self._responses else "Done."
        content = strip_hidden_reasoning(raw)
        return ModelResponse(
            model=request.model,
            content=content,
            usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=min(self.output_tokens, request.max_output_tokens),
            ),
        )
