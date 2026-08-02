from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_quality_cost_compatibility_gate import evaluate_gate


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_gate_uses_official_resolution_and_frozen_thresholds(
    tmp_path: Path,
) -> None:
    protocol = {
        "amendment_id": "a",
        "strong_compatibility_gate": {
            "tasks": ["a", "b", "c"],
            "minimum_structurally_valid": 2,
            "minimum_official_resolutions": 2,
        },
    }
    smoke = {
        "all_structurally_valid": True,
        "observed_conservative_cost_usd": "0.01",
    }
    episodes = [
        {
            "task_id": task_id,
            "model": "m",
            "structurally_valid": True,
            "submitted_patch": True,
            "provider_failed": False,
            "conservative_cost_usd": "1",
        }
        for task_id in ("a", "b", "c")
    ]
    grading = {
        "models": {"m": {"resolved_ids": ["a", "b"]}},
        "unclassified_errors": [],
    }
    protocol_path = tmp_path / "protocol.json"
    smoke_path = tmp_path / "smoke.json"
    episodes_path = tmp_path / "episodes.jsonl"
    grading_path = tmp_path / "grading.json"
    _write(protocol_path, protocol)
    _write(smoke_path, smoke)
    episodes_path.write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )
    _write(grading_path, grading)
    result = evaluate_gate(
        protocol_path=protocol_path,
        smoke_path=smoke_path,
        episodes_path=episodes_path,
        grading_path=grading_path,
    )
    assert result["passed"]
    assert result["resolved_count"] == 2
    assert result["episode_cost_usd"] == "3"

    grading["models"]["m"]["resolved_ids"] = ["a"]
    _write(grading_path, grading)
    result = evaluate_gate(
        protocol_path=protocol_path,
        smoke_path=smoke_path,
        episodes_path=episodes_path,
        grading_path=grading_path,
    )
    assert not result["passed"]
    assert result["decision"] == "stop_amendment_009_paid_work"
