from __future__ import annotations

from budget_router.agent_step import render_agent_prefix


def test_visible_prefix_is_bounded_and_strips_private_blocks() -> None:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Fix parser.py. Why does it fail?"},
        {
            "role": "assistant",
            "content": "<think>private reasoning</think>Visible note",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pytest -q"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>1</returncode><output>failed</output>",
        },
    ]

    rendered, summary = render_agent_prefix(messages, step_index=1)

    assert "private reasoning" not in rendered
    assert "Visible note" in rendered
    assert "pytest -q" in rendered
    assert summary.tool_call_count == 1
    assert summary.tool_result_count == 1
    assert summary.recent_nonzero_returncodes == 1
    assert summary.has_code_hint
    assert summary.has_question_hint


def test_visible_prefix_limits_large_messages() -> None:
    rendered, summary = render_agent_prefix(
        [{"role": "user", "content": "x" * 50_000}],
        step_index=0,
    )

    assert "[ROUTER INPUT ELIDED]" in rendered
    assert len(rendered) < 17_000
    assert summary.original_request_chars == 50_000
