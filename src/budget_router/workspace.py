from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class IsolatedWorkspacePool:
    """Copy-on-fork workspaces that never mutate the source checkout.

    Selecting a branch only changes ``current``. A restart creates a fresh copy
    of the original snapshot while retaining budget state in the caller.
    """

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source).resolve()
        if not self.source.is_dir():
            raise ValueError("source workspace must be a directory")
        if self.source == Path(self.source.anchor) or self.source == Path.home():
            raise ValueError("refusing to snapshot a broad filesystem root")
        self._temporary = tempfile.TemporaryDirectory(prefix="budget-router-workspace-")
        self.root = Path(self._temporary.name)
        self.original = self.root / "original"
        self._copy(self.source, self.original)
        self.current = self.original
        self._branches: dict[str, Path] = {}
        self._restart_count = 0

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )

    def fork(self, names: tuple[str, ...] = ("a", "b")) -> dict[str, Path]:
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("fork needs at least two unique branch names")
        branches: dict[str, Path] = {}
        for name in names:
            destination = self.root / f"branch-{name}"
            if destination.exists():
                raise FileExistsError(f"branch {name!r} already exists")
            self._copy(self.current, destination)
            branches[name] = destination
            self._branches[name] = destination
        return branches

    def select(self, name: str) -> Path:
        try:
            self.current = self._branches[name]
        except KeyError as exc:
            raise KeyError(f"unknown branch {name!r}") from exc
        return self.current

    def restart(self) -> Path:
        self._restart_count += 1
        destination = self.root / f"restart-{self._restart_count}"
        self._copy(self.original, destination)
        self.current = destination
        return destination

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> IsolatedWorkspacePool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

