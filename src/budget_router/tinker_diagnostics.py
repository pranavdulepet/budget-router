from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RenderedTinkerPrompt:
    """A rendered prompt plus publication-safe structural diagnostics."""

    prompt: Any
    rendered_messages: tuple[Mapping[str, Any], ...]
    token_ids: tuple[int, ...]

    def diagnostics(
        self,
        *,
        context_tokens: int,
        max_output_tokens: int,
        stop_sequences: Sequence[int | str],
    ) -> dict[str, Any]:
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if max_output_tokens < 0:
            raise ValueError("max_output_tokens must be non-negative")
        encoded = b"".join(struct.pack(">I", token_id) for token_id in self.token_ids)
        chunks = tuple(getattr(self.prompt, "chunks", ()))
        chunk_types = [type(chunk).__name__ for chunk in chunks]
        return {
            "input_tokens": len(self.token_ids),
            "max_output_tokens": max_output_tokens,
            "context_tokens": context_tokens,
            "context_utilization": (
                (len(self.token_ids) + max_output_tokens) / context_tokens
            ),
            "within_context": (
                len(self.token_ids) + max_output_tokens <= context_tokens
            ),
            "minimum_token_id": min(self.token_ids, default=None),
            "maximum_token_id": max(self.token_ids, default=None),
            "token_sha256": hashlib.sha256(encoded).hexdigest(),
            "chunk_count": len(chunks),
            "chunk_types": chunk_types,
            "text_only": bool(chunks)
            and all(name == "EncodedTextChunk" for name in chunk_types),
            "rendered_roles": [
                str(message.get("role", "")) for message in self.rendered_messages
            ],
            "stop_sequences": list(stop_sequences),
        }


def render_tinker_prompt(
    renderer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    effort: float | None = None,
) -> RenderedTinkerPrompt:
    """Render exactly the payload used by the Tinker adapter without sampling.

    The returned diagnostics contain hashes and structure, not prompt text or
    credentials, so they can be included in a support bundle.
    """

    copied = [dict(message) for message in messages]
    if tools:
        system_prompt = "\n\n".join(
            str(message["content"])
            for message in copied
            if message.get("role") == "system"
        )
        copied = [
            message for message in copied if message.get("role") != "system"
        ]
        prefix = renderer.create_conversation_prefix_with_tools(
            [dict(tool) for tool in tools],
            system_prompt=system_prompt,
        )
        copied = list(prefix) + copied
    prompt = (
        renderer.build_generation_prompt(copied, effort=effort)
        if effort is not None
        else renderer.build_generation_prompt(copied)
    )
    token_ids = (
        tuple(int(token_id) for token_id in prompt.to_ints())
        if hasattr(prompt, "to_ints")
        else tuple(int(token_id) for token_id in prompt)
    )
    return RenderedTinkerPrompt(
        prompt=prompt,
        rendered_messages=tuple(copied),
        token_ids=token_ids,
    )
