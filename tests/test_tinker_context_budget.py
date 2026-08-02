from __future__ import annotations

from types import SimpleNamespace

import pytest

from budget_router.providers.base import ProviderError
from budget_router.providers import tinker as tinker_provider


def _fake_render(
    _renderer: object,
    messages: list[dict[str, object]],
    _tools: tuple[object, ...],
    *,
    effort: float | None,
) -> SimpleNamespace:
    del effort
    tokens = sum(len(str(message.get("content", ""))) for message in messages)
    return SimpleNamespace(token_ids=list(range(tokens)), prompt=object())


def test_context_budget_drops_oldest_complete_exchanges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tinker_provider, "render_tinker_prompt", _fake_render)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a" * 20},
        {"role": "tool", "content": "o" * 20},
        {"role": "assistant", "content": "b" * 10},
        {"role": "tool", "content": "p" * 10},
    ]
    rendered, dropped = tinker_provider.render_with_context_budget(
        object(),
        messages,
        (),
        effort=None,
        context_tokens=50,
        max_output_tokens=20,
    )
    assert dropped == 2
    assert len(rendered.token_ids) == 30


def test_context_budget_never_drops_task_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tinker_provider, "render_tinker_prompt", _fake_render)
    messages = [
        {"role": "system", "content": "s" * 20},
        {"role": "user", "content": "u" * 20},
        {"role": "assistant", "content": "latest"},
        {"role": "tool", "content": "result"},
    ]
    with pytest.raises(ProviderError, match="safe history trimming"):
        tinker_provider.render_with_context_budget(
            object(),
            messages,
            (),
            effort=None,
            context_tokens=50,
            max_output_tokens=20,
        )
