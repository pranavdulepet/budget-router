from __future__ import annotations

import json
from pathlib import Path


def test_release_is_small_and_results_match_readme() -> None:
    root = Path(__file__).parents[1]
    public_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]
    markdown = [path for path in public_files if path.suffix.lower() == ".md"]
    assert markdown == [root / "README.md"]

    results = json.loads((root / "results/results.json").read_text())
    readme = (root / "README.md").read_text()
    primary = results["novel_repository_primary"]
    operational = results["full_operational_set"]
    assert f"{primary['router_resolved']}/{primary['tasks']}" in readme
    assert f"{operational['router_resolved']}/{operational['tasks']}" in readme
    assert "−20.80%" in readme
    assert "−20.79%" in readme
