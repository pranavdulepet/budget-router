"""Utilities for the isolated diagnostic-scout routing experiment."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .providers.base import strip_hidden_reasoning
from .redaction import redact


def isolated_policy_id(scout_model: str, finisher_model: str) -> str:
    """Return the stable public identifier for an isolated scout policy."""
    return f"isolated:{scout_model}->{finisher_model}"


def _tool_commands(message: Mapping[str, Any]) -> list[str]:
    commands: list[str] = []
    for call in message.get("tool_calls", ()):
        try:
            function = call["function"]
            arguments = json.loads(function["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if (
            function.get("name") == "bash"
            and isinstance(arguments, dict)
            and isinstance(arguments.get("command"), str)
        ):
            commands.append(arguments["command"])
    return commands


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[VISIBLE HANDOFF ELIDED]\n"
    available = limit - len(marker)
    if available <= 0:
        return marker[:limit]
    head = available // 2
    return text[:head] + marker + text[-(available - head) :]


def build_visible_diagnostic_handoff(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
) -> str:
    """Render only visible scout evidence for a clean-workspace finisher.

    System prompts, the original issue, terminal metadata, and private reasoning
    are excluded. Visible assistant text, bash commands, tool results, and
    format-error feedback are retained in chronological order.
    """
    if max_chars <= 0:
        raise ValueError("visible handoff character limit must be positive")
    sections: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role == "assistant":
            visible = strip_hidden_reasoning(str(message.get("content", "")))
            if visible:
                sections.append("SCOUT VISIBLE NOTE:\n" + visible)
            for command in _tool_commands(message):
                sections.append("SCOUT BASH COMMAND:\n" + command)
        elif role == "tool":
            visible = strip_hidden_reasoning(str(message.get("content", "")))
            if visible:
                sections.append("SCOUT TOOL RESULT:\n" + visible)
        elif role == "user" and message.get("extra", {}).get("interrupt_type"):
            visible = strip_hidden_reasoning(str(message.get("content", "")))
            if visible:
                sections.append("SCOUT HARNESS FEEDBACK:\n" + visible)
    rendered = "\n\n".join(sections).strip()
    if not rendered:
        rendered = "The scout produced no transferable visible evidence."
    # A tool output could accidentally expose a credential from the workspace.
    # Redact before the text is passed to a second provider or persisted.
    sanitized = str(redact(rendered))
    return _bounded_text(sanitized, max_chars)


def remaining_finisher_cap(
    total_cap_usd: Decimal,
    scout_spend_usd: Decimal,
) -> Decimal:
    """Compute a non-negative finisher cap from actual conservative spend."""
    total = Decimal(total_cap_usd)
    scout = Decimal(scout_spend_usd)
    if total <= 0:
        raise ValueError("total episode cap must be positive")
    if scout < 0:
        raise ValueError("scout spend cannot be negative")
    if scout > total:
        raise ValueError("scout spend exceeds total episode cap")
    return total - scout


def is_zero_call_infrastructure_failure(record: Mapping[str, Any]) -> bool:
    """Return whether a record was censored before either provider was called."""
    return bool(
        record.get("error_type") == "CalledProcessError"
        and int(record.get("model_calls", -1)) == 0
        and Decimal(str(record.get("conservative_cost_usd", "-1"))) == 0
        and not record.get("submitted_patch")
        and not record.get("provider_failed")
        and not record.get("structurally_valid")
    )


def canonicalize_infrastructure_recoveries(
    records: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay eligible recovery attempts without erasing the raw audit log."""
    base_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for record in records:
        key = (
            str(record.get("study_stage", "")),
            str(record["task_id"]),
            str(record["model"]),
        )
        if key in base_by_key:
            raise ValueError(f"duplicate base episode: {key}")
        base_by_key[key] = record
        ordered_keys.append(key)

    recovery_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for recovery in recoveries:
        key = (
            str(recovery.get("study_stage", "")),
            str(recovery["task_id"]),
            str(recovery["model"]),
        )
        if key in recovery_by_key:
            raise ValueError(f"duplicate infrastructure recovery: {key}")
        original = base_by_key.get(key)
        if original is None:
            raise ValueError(f"infrastructure recovery has no base episode: {key}")
        if not is_zero_call_infrastructure_failure(original):
            raise ValueError(f"base episode is not retry-eligible: {key}")
        if recovery.get("study_manifest_hash") != original.get(
            "study_manifest_hash"
        ):
            raise ValueError(f"infrastructure recovery freeze mismatch: {key}")
        recovered = dict(recovery)
        recovered["infrastructure_recovery"] = True
        recovered["replaces_error_type"] = original.get("error_type")
        recovery_by_key[key] = recovered

    return [
        dict(recovery_by_key.get(key, base_by_key[key]))
        for key in ordered_keys
    ]


def plan_stage_episodes(
    *,
    task_ids: Sequence[str],
    policy_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    study_manifest_hash: str,
    stage: str,
    hard_cap_usd: Decimal,
    max_task_blocks: int | None = None,
) -> dict[str, Any]:
    """Build an append-only, task-blocked stage plan."""
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("stage task IDs must be non-empty and unique")
    if not policy_ids or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("stage policy IDs must be non-empty and unique")
    if hard_cap_usd <= 0:
        raise ValueError("episode hard cap must be positive")
    if max_task_blocks is not None and max_task_blocks <= 0:
        raise ValueError("max task blocks must be positive")

    valid = {(task_id, policy_id) for task_id in task_ids for policy_id in policy_ids}
    completed: set[tuple[str, str]] = set()
    for record in records:
        if record.get("study_manifest_hash") != study_manifest_hash:
            raise ValueError("record belongs to a different frozen study")
        key = (str(record["task_id"]), str(record["model"]))
        if key in completed:
            raise ValueError(f"duplicate stage record: {key}")
        completed.add(key)
        if record.get("study_stage") == stage and key not in valid:
            raise ValueError(f"record is outside the frozen {stage} matrix: {key}")

    blocks: list[dict[str, Any]] = []
    for task_id in task_ids:
        pending = [
            policy_id
            for policy_id in policy_ids
            if (task_id, policy_id) not in completed
        ]
        if pending:
            blocks.append({"task_id": task_id, "policy_ids": pending})
    if max_task_blocks is not None:
        blocks = blocks[:max_task_blocks]
    episode_count = sum(len(block["policy_ids"]) for block in blocks)
    return {
        "pending_blocks": blocks,
        "pending_episode_count": episode_count,
        "maximum_pending_cost_usd": hard_cap_usd * episode_count,
    }
