"""Capture tracked repository changes before an ephemeral workspace is removed."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERMINAL_WORKSPACE_DIFF_COMMAND = (
    "git diff --binary --no-ext-diff HEAD --"
)


@dataclass(frozen=True, slots=True)
class TerminalWorkspacePatchCapture:
    """Result of a harness-owned terminal workspace diff."""

    attempted: bool
    patch: str
    returncode: int | None
    error: str | None

    @property
    def captured(self) -> bool:
        return bool(self.patch)

    @property
    def patch_bytes(self) -> int:
        return len(self.patch.encode("utf-8"))

    @property
    def patch_sha256(self) -> str | None:
        if not self.patch:
            return None
        return hashlib.sha256(self.patch.encode("utf-8")).hexdigest()


def capture_terminal_workspace_patch(
    environment: Any | None,
    *,
    timeout_seconds: int = 60,
) -> TerminalWorkspacePatchCapture:
    """Return the tracked workspace diff without masking the agent outcome."""

    if environment is None:
        return TerminalWorkspacePatchCapture(
            attempted=False,
            patch="",
            returncode=None,
            error=None,
        )
    try:
        result = environment.execute(
            {"command": TERMINAL_WORKSPACE_DIFF_COMMAND},
            timeout=timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        return TerminalWorkspacePatchCapture(
            attempted=True,
            patch="",
            returncode=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    returncode = int(result.get("returncode", -1))
    exception_info = str(result.get("exception_info", "") or "")
    if returncode != 0 or exception_info:
        detail = exception_info or f"git diff exited with status {returncode}"
        return TerminalWorkspacePatchCapture(
            attempted=True,
            patch="",
            returncode=returncode,
            error=detail,
        )
    return TerminalWorkspacePatchCapture(
        attempted=True,
        patch=str(result.get("output", "") or ""),
        returncode=returncode,
        error=None,
    )


def terminal_workspace_patch_path(trajectory_path: Path) -> Path:
    """Derive a stable sidecar path from a trajectory path."""

    return trajectory_path.with_name(
        f"{trajectory_path.stem}.terminal_workspace.patch"
    )


def persist_terminal_workspace_patch(
    capture: TerminalWorkspacePatchCapture,
    path: Path,
) -> Path | None:
    """Persist a non-empty captured patch and return its path."""

    if not capture.captured:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(capture.patch, encoding="utf-8")
    return path


def terminal_workspace_patch_record(
    capture: TerminalWorkspacePatchCapture,
    path: Path | None,
) -> dict[str, Any]:
    """Serialize capture metadata without embedding patch contents."""

    return {
        "terminal_workspace_patch_capture_attempted": capture.attempted,
        "terminal_workspace_patch_captured": capture.captured,
        "terminal_workspace_patch_sha256": capture.patch_sha256,
        "terminal_workspace_patch_bytes": capture.patch_bytes,
        "terminal_workspace_patch_path": str(path) if path else None,
        "terminal_workspace_patch_capture_returncode": capture.returncode,
        "terminal_workspace_patch_capture_error": capture.error,
    }


def capture_and_persist_terminal_workspace_patch(
    environment: Any | None,
    trajectory_path: Path,
    *,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Capture and persist a terminal patch without raising into the runner."""

    capture = capture_terminal_workspace_patch(
        environment,
        timeout_seconds=timeout_seconds,
    )
    path: Path | None = None
    persistence_error: str | None = None
    if capture.captured:
        try:
            path = persist_terminal_workspace_patch(
                capture,
                terminal_workspace_patch_path(trajectory_path),
            )
        except Exception as exc:  # pragma: no cover - defensive filesystem edge
            persistence_error = f"{type(exc).__name__}: {exc}"
    record = terminal_workspace_patch_record(capture, path)
    if persistence_error is not None:
        record["terminal_workspace_patch_capture_error"] = persistence_error
    return record
