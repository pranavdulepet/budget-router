from __future__ import annotations

import pytest

from budget_router.providers.tinker import recover_nemotron_tool_call


@pytest.mark.parametrize(
    ("content", "expected_command", "expected_recovery"),
    [
        (
            (
                "I will inspect the file.\n"
                "<function=bash>\n"
                "<parameter=command>\nrg -n 'target' src\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            "rg -n 'target' src",
            "nemotron_missing_outer_xml_v1",
        ),
        (
            '{"command":"pwd"}',
            "pwd",
            "nemotron_bare_json_v1",
        ),
        (
            '{"name":"bash","arguments":{"command":"git diff"}}',
            "git diff",
            "nemotron_bare_json_v1",
        ),
        (
            '[Makes a tool call: {"command":"ls"}]\n</tool_call>',
            "ls",
            "nemotron_bare_json_v1",
        ),
    ],
)
def test_recovers_observed_nemotron_bash_variants(
    content: str,
    expected_command: str,
    expected_recovery: str,
) -> None:
    calls, visible, recovery = recover_nemotron_tool_call(content)
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"command": expected_command}
    assert recovery == expected_recovery
    assert "<function=" not in visible


@pytest.mark.parametrize(
    "content",
    [
        "<function=bash><parameter=command>a</parameter></function>"
        "<function=bash><parameter=command>b</parameter></function>",
        "<function=other><parameter=command>pwd</parameter></function>",
        '{"command":"pwd","extra":true}',
        '{"name":"other","arguments":{"command":"pwd"}}',
        "ordinary prose",
    ],
)
def test_recovery_rejects_ambiguous_or_out_of_scope_content(content: str) -> None:
    calls, visible, recovery = recover_nemotron_tool_call(content)
    assert calls == ()
    assert visible == content
    assert recovery is None
