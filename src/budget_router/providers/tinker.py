from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Mapping

from ..tinker_diagnostics import render_tinker_prompt
from ..types import TokenUsage
from .base import (
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    strip_hidden_reasoning,
)

DEFAULT_TINKER_BASE_URL = (
    "https://tinker.thinkingmachines.dev/services/tinker-prod"
)
TINKER_SEED_MODULUS = 2**31


def bounded_tinker_seed(seed: int | None) -> int | None:
    """Keep sampling seeds in the backend's safe signed 32-bit range."""
    return seed % TINKER_SEED_MODULUS if seed is not None else None


def _require_tinker_key() -> None:
    # Intentionally check presence only. The credential is never copied or logged.
    if not os.environ.get("TINKER_API_KEY"):
        raise ProviderError(
            "TINKER_API_KEY is not set. Rotate any exposed key before creating this adapter."
        )


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _httpx_client_config() -> Mapping[str, Any]:
    """Fetch feature flags while forcing the SDK away from a stuck pyqwest transport."""
    try:
        import httpx
    except ImportError as exc:
        raise ProviderError("Tinker HTTP transport fallback requires httpx") from exc
    base_url = os.environ.get("TINKER_BASE_URL") or DEFAULT_TINKER_BASE_URL
    headers = {"X-API-Key": os.environ["TINKER_API_KEY"]}
    for environment_name, header_name in (
        ("CLOUDFLARE_ACCESS_CLIENT_ID", "CF-Access-Client-Id"),
        ("CLOUDFLARE_ACCESS_CLIENT_SECRET", "CF-Access-Client-Secret"),
    ):
        if value := os.environ.get(environment_name):
            headers[header_name] = value
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/v1/client/config",
            headers=headers,
            json={"sdk_version": version("tinker")},
            timeout=15,
        )
        response.raise_for_status()
        config = response.json()
    except Exception as exc:
        raise ProviderError(
            f"Tinker client configuration failed: {type(exc).__name__}"
        ) from exc
    config["use_pyqwest_transport"] = False
    return config


def _visible_content(content: Any) -> str:
    if isinstance(content, str):
        return strip_hidden_reasoning(content)
    if isinstance(content, list):
        return strip_hidden_reasoning(
            "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
        )
    return ""


def _parsed_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    result: list[ToolCall] = []
    for call in message.get("tool_calls", ()):
        function = getattr(call, "function", None)
        if function is None:
            continue
        raw_arguments = getattr(function, "arguments", "{}")
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(arguments, Mapping):
            continue
        result.append(
            ToolCall(
                name=str(getattr(function, "name", "")),
                arguments=dict(arguments),
                call_id=str(getattr(call, "id", "") or ""),
            )
        )
    return tuple(result)


_NEMOTRON_FUNCTION = re.compile(
    r"<function=(?P<name>[^>\s]+)>\s*"
    r"<parameter=command>\s*(?P<command>.*?)\s*</parameter>\s*"
    r"</function>",
    re.DOTALL,
)
_NEMOTRON_MARKER = re.compile(
    r"^\[Makes a tool call:\s*(?P<payload>\{.*\})\]\s*"
    r"(?:</tool_call>)?\s*$",
    re.DOTALL,
)


def recover_nemotron_tool_call(
    content: str,
) -> tuple[tuple[ToolCall, ...], str, str | None]:
    """Recover one unambiguous Nano/Super bash call missing its outer wrapper."""
    matches = list(_NEMOTRON_FUNCTION.finditer(content))
    if len(matches) == 1 and content.count("<function=") == 1:
        match = matches[0]
        suffix = content[match.end() :].strip()
        if suffix in ("", "</tool_call>"):
            name = match.group("name").strip()
            command = match.group("command").strip()
            if name == "bash" and command:
                visible = content[: match.start()].strip()
                return (
                    (ToolCall(name="bash", arguments={"command": command}),),
                    visible,
                    "nemotron_missing_outer_xml_v1",
                )

    stripped = content.strip()
    marker = _NEMOTRON_MARKER.fullmatch(stripped)
    payload_text = marker.group("payload") if marker else stripped
    try:
        payload = json.loads(payload_text)
    except (TypeError, json.JSONDecodeError):
        return (), content, None
    if not isinstance(payload, Mapping):
        return (), content, None
    if set(payload) == {"command"}:
        command = payload.get("command")
    elif set(payload) == {"name", "arguments"} and payload.get("name") == "bash":
        arguments = payload.get("arguments")
        command = arguments.get("command") if isinstance(arguments, Mapping) else None
    else:
        command = None
    if isinstance(command, str) and command.strip():
        return (
            (ToolCall(name="bash", arguments={"command": command.strip()}),),
            "",
            "nemotron_bare_json_v1",
        )
    return (), content, None


def render_with_context_budget(
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: tuple[Mapping[str, Any], ...],
    *,
    effort: float | None,
    context_tokens: int | None,
    max_output_tokens: int,
) -> tuple[Any, int]:
    """Render a sliding history while reserving the full requested output."""
    candidate = list(messages)
    dropped = 0
    while True:
        rendered = render_tinker_prompt(
            renderer,
            candidate,
            tools,
            effort=effort,
        )
        if (
            context_tokens is None
            or len(rendered.token_ids) + max_output_tokens <= context_tokens
        ):
            return rendered, dropped

        anchor_count = 0
        if candidate and candidate[0].get("role") == "system":
            anchor_count = 1
        if (
            len(candidate) > anchor_count
            and candidate[anchor_count].get("role") == "user"
        ):
            anchor_count += 1
        tail = candidate[anchor_count:]
        next_assistant = next(
            (
                index
                for index, message in enumerate(tail[1:], start=1)
                if message.get("role") == "assistant"
            ),
            None,
        )
        if next_assistant is None:
            raise ProviderError(
                "Tinker prompt exceeds context after safe history trimming"
            )
        dropped += next_assistant
        candidate = candidate[:anchor_count] + tail[next_assistant:]


@dataclass(slots=True)
class TinkerSamplingAdapter:
    """Thin adapter over Tinker's SamplingClient and cookbook renderer."""

    sampling_client: Any
    tokenizer: Any
    renderer: Any
    renderer_name: str
    effort: float | None = None

    @classmethod
    async def from_environment(
        cls,
        model: str,
        *,
        renderer_name: str | None = None,
        effort: float | None = None,
        force_httpx_transport: bool = False,
    ) -> TinkerSamplingAdapter:
        _require_tinker_key()
        try:
            import tinker
            from tinker_cookbook import model_info, renderers, tokenizer_utils
        except ImportError as exc:
            raise ProviderError(
                "Tinker support requires budget-router[tinker]"
            ) from exc
        service = (
            tinker.ServiceClient(_client_config=dict(_httpx_client_config()))
            if force_httpx_transport
            else tinker.ServiceClient()
        )
        create = getattr(service, "create_sampling_client_async", None)
        client = (
            await create(base_model=model)
            if create is not None
            else service.create_sampling_client(base_model=model)
        )
        tokenizer = (
            tokenizer_utils.get_tokenizer(model)
            if model.split(":", 1)[0] == "thinkingmachines/Inkling"
            else await _await_if_needed(client.get_tokenizer())
        )
        selected_renderer = renderer_name or model_info.get_recommended_renderer_name(model)
        renderer = renderers.get_renderer(
            selected_renderer,
            tokenizer,
            model_name=model,
        )
        return cls(client, tokenizer, renderer, selected_renderer, effort)

    async def _generate_from_messages(
        self,
        request: ModelRequest,
        messages: list[dict[str, Any]],
    ) -> ModelResponse:
        try:
            from tinker import types
        except ImportError as exc:
            raise ProviderError("Tinker SDK is not installed") from exc
        try:
            raw_context_tokens = request.metadata.get("context_tokens")
            context_tokens = (
                int(raw_context_tokens) if raw_context_tokens is not None else None
            )
            rendered, history_messages_dropped = render_with_context_budget(
                self.renderer,
                messages,
                request.tools,
                effort=self.effort,
                context_tokens=context_tokens,
                max_output_tokens=request.max_output_tokens,
            )
        except NotImplementedError as exc:
            raise ProviderError(
                f"renderer {self.renderer_name} does not support tools"
            ) from exc
        prompt = rendered.prompt
        input_tokens = len(rendered.token_ids)
        if not hasattr(prompt, "chunks"):
            prompt = types.ModelInput.from_ints(list(rendered.token_ids))
        params = types.SamplingParams(
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=bounded_tinker_seed(request.seed),
            stop=self.renderer.get_stop_sequences(),
        )
        started = time.monotonic()
        try:
            sample = getattr(self.sampling_client, "sample_async", None)
            if sample is not None:
                pending = sample(
                    prompt=prompt, sampling_params=params, num_samples=1
                )
            else:
                pending = _await_if_needed(
                    self.sampling_client.sample(
                        prompt=prompt, sampling_params=params, num_samples=1
                    )
                )
            result = (
                await asyncio.wait_for(pending, timeout=request.timeout_seconds)
                if request.timeout_seconds is not None
                else await pending
            )
        except Exception as exc:
            category = getattr(exc, "category", None)
            category_suffix = (
                f" category={getattr(category, 'value', category)}"
                if category is not None
                else ""
            )
            raise ProviderError(
                f"Tinker sampling failed: {type(exc).__name__}{category_suffix}"
            ) from exc
        sequences = getattr(result, "sequences", getattr(result, "samples", ()))
        if not sequences:
            raise ProviderError("Tinker returned no sampled sequence")
        sequence = sequences[0]
        tokens = sequence.tokens() if callable(sequence.tokens) else sequence.tokens
        try:
            parsed, termination = self.renderer.parse_response(list(tokens))
        except Exception as exc:
            raise ProviderError(
                f"Tinker renderer parse failed: {type(exc).__name__}"
            ) from exc
        content = _visible_content(parsed.get("content", ""))
        tool_calls = _parsed_tool_calls(parsed)
        tool_call_recovery: str | None = None
        if not tool_calls and self.renderer_name.startswith("nemotron3"):
            tool_calls, content, tool_call_recovery = recover_nemotron_tool_call(
                content
            )
        malformed_tool_calls = len(parsed.get("unparsed_tool_calls", ()))
        return ModelResponse(
            model=request.model,
            content=content,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=len(tokens)),
            tool_calls=tool_calls,
            finish_reason=str(getattr(sequence, "stop_reason", "stop")),
            latency_ms=int((time.monotonic() - started) * 1000),
            provider_metadata={
                "adapter": "tinker",
                "renderer": self.renderer_name,
                "effort": self.effort,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "seed_was_bounded": (
                    request.seed is not None
                    and request.seed != bounded_tinker_seed(request.seed)
                ),
                "parse_termination": str(termination),
                "malformed_tool_calls": malformed_tool_calls,
                "tool_call_recovery": tool_call_recovery,
                "context_tokens": context_tokens,
                "history_messages_dropped": history_messages_dropped,
            },
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        messages: list[dict[str, Any]] = [
            {
                "role": message.role,
                "content": message.content,
                **({"name": message.name} if message.name else {}),
            }
            for message in request.messages
        ]
        return await self._generate_from_messages(request, messages)


@dataclass(slots=True)
class TinkerTrainingAdapter:
    """Minimal LoRA training boundary; the caller owns batching and checkpoint policy."""

    training_client: Any

    @classmethod
    async def from_environment(
        cls,
        model: str,
        *,
        rank: int = 32,
    ) -> TinkerTrainingAdapter:
        _require_tinker_key()
        try:
            import tinker
        except ImportError as exc:
            raise ProviderError(
                "Tinker support requires budget-router[tinker]"
            ) from exc
        service = tinker.ServiceClient()
        create = getattr(service, "create_lora_training_client_async", None)
        if create is not None:
            client = await create(base_model=model, rank=rank)
        else:
            client = service.create_lora_training_client(base_model=model, rank=rank)
        return cls(client)

    async def train_batch(self, data: list[Any], *, loss_fn: str = "cross_entropy") -> Any:
        forward = getattr(self.training_client, "forward_backward_async", None)
        result = (
            await forward(data, loss_fn=loss_fn)
            if forward is not None
            else await _await_if_needed(
                self.training_client.forward_backward(data, loss_fn=loss_fn)
            )
        )
        step = getattr(self.training_client, "optim_step_async", None)
        if step is not None:
            await step()
        else:
            await _await_if_needed(self.training_client.optim_step())
        return result
