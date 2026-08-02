from __future__ import annotations

import hashlib
from pathlib import Path

from budget_router.workspace_patch import (
    TERMINAL_WORKSPACE_DIFF_COMMAND,
    capture_and_persist_terminal_workspace_patch,
    capture_terminal_workspace_patch,
    persist_terminal_workspace_patch,
    terminal_workspace_patch_path,
    terminal_workspace_patch_record,
)


class FakeEnvironment:
    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.actions: list[tuple[dict, int | None]] = []

    def execute(self, action: dict, *, timeout: int | None = None) -> dict:
        self.actions.append((action, timeout))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_capture_and_persist_terminal_workspace_patch(tmp_path: Path) -> None:
    patch = "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n"
    environment = FakeEnvironment(
        {"output": patch, "returncode": 0, "exception_info": ""}
    )

    capture = capture_terminal_workspace_patch(environment)
    path = terminal_workspace_patch_path(tmp_path / "task.json")
    persisted = persist_terminal_workspace_patch(capture, path)
    record = terminal_workspace_patch_record(capture, persisted)

    assert environment.actions == [
        ({"command": TERMINAL_WORKSPACE_DIFF_COMMAND}, 60)
    ]
    assert persisted == tmp_path / "task.terminal_workspace.patch"
    assert persisted.read_text(encoding="utf-8") == patch
    assert record == {
        "terminal_workspace_patch_capture_attempted": True,
        "terminal_workspace_patch_captured": True,
        "terminal_workspace_patch_sha256": hashlib.sha256(
            patch.encode()
        ).hexdigest(),
        "terminal_workspace_patch_bytes": len(patch.encode()),
        "terminal_workspace_patch_path": str(persisted),
        "terminal_workspace_patch_capture_returncode": 0,
        "terminal_workspace_patch_capture_error": None,
    }


def test_empty_terminal_workspace_is_successful_without_sidecar(
    tmp_path: Path,
) -> None:
    capture = capture_terminal_workspace_patch(
        FakeEnvironment({"output": "", "returncode": 0, "exception_info": ""})
    )
    persisted = persist_terminal_workspace_patch(
        capture, tmp_path / "empty.patch"
    )
    record = terminal_workspace_patch_record(capture, persisted)

    assert persisted is None
    assert record["terminal_workspace_patch_capture_attempted"] is True
    assert record["terminal_workspace_patch_captured"] is False
    assert record["terminal_workspace_patch_capture_error"] is None


def test_capture_failure_does_not_raise_or_persist(tmp_path: Path) -> None:
    capture = capture_terminal_workspace_patch(
        FakeEnvironment(error=RuntimeError("container unavailable"))
    )
    persisted = persist_terminal_workspace_patch(
        capture, tmp_path / "failed.patch"
    )
    record = terminal_workspace_patch_record(capture, persisted)

    assert persisted is None
    assert record["terminal_workspace_patch_capture_attempted"] is True
    assert record["terminal_workspace_patch_captured"] is False
    assert record["terminal_workspace_patch_capture_returncode"] is None
    assert record["terminal_workspace_patch_capture_error"] == (
        "RuntimeError: container unavailable"
    )


def test_absent_environment_is_not_an_attempt() -> None:
    capture = capture_terminal_workspace_patch(None)

    assert capture.attempted is False
    assert capture.captured is False
    assert capture.error is None


def test_capture_and_persist_convenience_function(tmp_path: Path) -> None:
    patch = "diff --git a/a.py b/a.py\n"
    record = capture_and_persist_terminal_workspace_patch(
        FakeEnvironment(
            {"output": patch, "returncode": 0, "exception_info": ""}
        ),
        tmp_path / "trajectory.json",
    )

    assert record["terminal_workspace_patch_captured"] is True
    path = Path(record["terminal_workspace_patch_path"])
    assert path.name == "trajectory.terminal_workspace.patch"
    assert path.read_text(encoding="utf-8") == patch
