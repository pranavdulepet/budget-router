from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .providers.base import strip_hidden_reasoning
from .serialization import stable_hash, stable_json
from .types import GoalContext, RouterState, TranscriptMessage


class TransferFormat(StrEnum):
    FULL_VISIBLE_TRANSCRIPT = "full_visible_transcript"
    STRUCTURED_CHECKPOINT = "structured_checkpoint"
    GOAL_WORKSPACE_ARTIFACTS = "goal_workspace_artifacts"


@dataclass(frozen=True, slots=True)
class Handoff:
    format: TransferFormat
    messages: tuple[TranscriptMessage, ...]
    metadata: Mapping[str, Any]


def build_handoff(
    format: TransferFormat,
    goal: GoalContext,
    state: RouterState,
) -> Handoff:
    """Build a switch payload exclusively from visible state and workspace artifacts."""
    if format is TransferFormat.FULL_VISIBLE_TRANSCRIPT:
        messages = tuple(
            TranscriptMessage(
                message.role,
                strip_hidden_reasoning(message.content),
                message.name,
            )
            for message in state.visible_transcript
        )
        metadata = {"diff_hash": state.workspace.patch_hash}
    elif format is TransferFormat.STRUCTURED_CHECKPOINT:
        checkpoint = {
            "goal": goal.goal,
            "repository": goal.repository,
            "turn": state.turn,
            "workspace": {
                "files_changed": state.workspace.files_changed,
                "insertions": state.workspace.insertions,
                "deletions": state.workspace.deletions,
                "patch_hash": state.workspace.patch_hash,
                "summary": state.workspace.summary,
            },
            "recent_tools": [
                {"name": event.name, "success": event.success, "summary": event.summary}
                for event in state.tool_events[-5:]
            ],
            "recent_tests": [
                {
                    "command": event.command,
                    "passed": event.passed,
                    "summary": event.summary,
                }
                for event in state.test_events[-3:]
            ],
        }
        messages = (
            TranscriptMessage(
                "user",
                "Continue from this visible structured checkpoint:\n"
                + json.dumps(checkpoint, sort_keys=True),
            ),
        )
        metadata = {"checkpoint_version": "v1"}
    else:
        messages = (
            TranscriptMessage("user", goal.goal),
            TranscriptMessage(
                "user",
                "Visible workspace summary: "
                + (state.workspace.summary or "No summary available.")
                + f"\nDiff hash: {state.workspace.patch_hash or 'none'}",
            ),
        )
        metadata = {"artifact_only": True}
    return Handoff(format=format, messages=messages, metadata=metadata)


def select_transfer_format(
    validation_scores: Mapping[TransferFormat, float],
) -> TransferFormat:
    required = set(TransferFormat)
    if set(validation_scores) != required:
        missing = required - set(validation_scores)
        raise ValueError(f"validation scores missing formats: {sorted(missing)}")
    # Stable tie-break prefers the compact structured checkpoint.
    preference = {
        TransferFormat.STRUCTURED_CHECKPOINT: 2,
        TransferFormat.GOAL_WORKSPACE_ARTIFACTS: 1,
        TransferFormat.FULL_VISIBLE_TRANSCRIPT: 0,
    }
    return max(
        TransferFormat,
        key=lambda item: (validation_scores[item], preference[item]),
    )


def freeze_transfer_selection(
    path: str | Path,
    validation_scores: Mapping[TransferFormat, float],
) -> TransferFormat:
    winner = select_transfer_format(validation_scores)
    payload = {
        "schema_version": "transfer-selection-v1",
        "scores": {format.value: validation_scores[format] for format in TransferFormat},
        "selected": winner.value,
    }
    payload["selection_hash"] = stable_hash(payload)
    destination = Path(path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError("refusing to overwrite frozen transfer selection")
        return winner
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json(payload) + "\n", encoding="utf-8")
    return winner
