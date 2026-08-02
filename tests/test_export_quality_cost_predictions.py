from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_quality_cost_predictions import export_predictions


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_export_requires_complete_matrix_and_preserves_no_submission(
    tmp_path: Path,
) -> None:
    protocol = {
        "parent_study_id": "study",
        "amendment_id": "amendment",
    }
    manifest = {"tasks": [{"task_id": "a"}, {"task_id": "b"}]}
    patch = "diff --git a/a b/a\n"
    trajectory = tmp_path / "trajectory.json"
    _write_json(trajectory, {"info": {"submission": patch}})
    records = [
        {
            "task_id": "a",
            "model": "vendor/model",
            "submitted_patch": True,
            "submission_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "trajectory": str(trajectory),
            "structurally_valid": True,
            "conservative_cost_usd": "0.2",
        },
        {
            "task_id": "b",
            "model": "vendor/model",
            "submitted_patch": False,
            "submission_sha256": None,
            "trajectory": "unused",
            "structurally_valid": True,
            "conservative_cost_usd": "0.3",
        },
    ]
    protocol_path = tmp_path / "protocol.json"
    manifest_path = tmp_path / "tasks.json"
    records_path = tmp_path / "records.jsonl"
    _write_json(protocol_path, protocol)
    _write_json(manifest_path, manifest)
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    result = export_predictions(
        project_root=tmp_path,
        protocol_path=protocol_path,
        task_manifest_path=manifest_path,
        records_path=records_path,
        stage="compatibility",
        output_dir=tmp_path / "out",
        expected_models=["vendor/model"],
    )
    model = result["models"]["vendor/model"]
    assert model["submitted_count"] == 1
    assert model["no_submission_task_ids"] == ["b"]
    assert model["canonical_episode_cost_usd"] == "0.5"

    records_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical matrix mismatch"):
        export_predictions(
            project_root=tmp_path,
            protocol_path=protocol_path,
            task_manifest_path=manifest_path,
            records_path=records_path,
            stage="compatibility",
            output_dir=tmp_path / "bad",
            expected_models=["vendor/model"],
        )
