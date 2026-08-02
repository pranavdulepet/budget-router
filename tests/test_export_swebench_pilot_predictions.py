from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_swebench_pilot_predictions import export_predictions


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_export_predictions_groups_models_and_counts_unresolved(tmp_path: Path) -> None:
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    trajectory = tmp_path / "trajectories" / "task-1.json"
    _write_json(trajectory, {"info": {"submission": patch}})
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "model": "org/model",
                        "task_id": "task-1",
                        "study_stage": "test",
                        "submitted_patch": True,
                        "submission_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                        "trajectory": "trajectories/task-1.json",
                    }
                ),
                json.dumps(
                    {
                        "model": "org/model",
                        "task_id": "task-2",
                        "study_stage": "train",
                        "submitted_patch": False,
                        "trajectory": "trajectories/task-2.json",
                    }
                ),
                json.dumps(
                    {
                        "model": "other",
                        "task_id": "task-3",
                        "study_stage": "train",
                        "submitted_patch": False,
                        "trajectory": "trajectories/task-3.json",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "predictions"
    manifest = export_predictions(episodes, output_dir, project_root=tmp_path)

    assert manifest["episode_count"] == 3
    assert manifest["submitted_count"] == 1
    assert manifest["unresolved_without_submission_count"] == 2
    assert manifest["models"]["org/model"]["submitted_count"] == 1
    assert manifest["models"]["org/model"]["unresolved_without_submission_count"] == 1
    exported = [
        json.loads(line)
        for line in (output_dir / "org__model.jsonl").read_text().splitlines()
    ]
    assert exported == [
        {
            "instance_id": "task-1",
            "model_name_or_path": "org/model",
            "model_patch": patch,
        }
    ]

    test_manifest = export_predictions(
        episodes,
        tmp_path / "test-predictions",
        project_root=tmp_path,
        study_stage="test",
    )
    assert test_manifest["study_stage"] == "test"
    assert test_manifest["episode_count"] == 1
    assert test_manifest["submitted_count"] == 1


def test_export_predictions_rejects_hash_mismatch(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.json"
    _write_json(trajectory, {"info": {"submission": "patch"}})
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "model": "model",
                "task_id": "task",
                "submitted_patch": True,
                "submission_sha256": "not-the-hash",
                "trajectory": "trajectory.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        export_predictions(episodes, tmp_path / "output", project_root=tmp_path)
