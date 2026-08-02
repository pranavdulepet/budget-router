from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_guarded_agent_step_predictions import (
    LABELS,
    export_predictions,
)
from scripts.run_guarded_agent_step_followup import ROUTED


ROOT = Path(__file__).resolve().parents[1]


def test_development_export_requires_every_canonical_episode(
    tmp_path: Path,
) -> None:
    artifact_path = ROOT / "outputs/agent_step_router_v2/router_artifact.json"
    artifact = json.loads(artifact_path.read_text())
    protocol_path = (
        ROOT
        / "artifacts/"
        "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
    )
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_hash": "tasks",
                "development": [{"task_id": "task-1"}],
                "heldout": [],
            }
        )
    )
    trajectory_path = tmp_path / "trajectory.json"
    patch = "diff --git a/a.py b/a.py\n"
    trajectory_path.write_text(json.dumps({"info": {"submission": patch}}))
    records_path = tmp_path / "records.jsonl"
    record = {
        "study_stage": "development",
        "task_id": "task-1",
        "policy_id": ROUTED,
        "router_artifact_hash": artifact["artifact_hash"],
        "submitted_patch": True,
        "submission_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "trajectory": str(trajectory_path),
        "structurally_valid": True,
        "conservative_cost_usd": "0.1",
    }
    records_path.write_text(json.dumps(record) + "\n")

    result = export_predictions(
        stage="development",
        project_root=ROOT,
        protocol_path=protocol_path,
        task_manifest_path=manifest_path,
        artifact_path=artifact_path,
        records_path=records_path,
        output_dir=tmp_path / "predictions",
    )
    model = result["models"][LABELS[ROUTED]]
    assert model["submitted_count"] == 1
    assert result["study_stage"] == "development"

    records_path.write_text("")
    with pytest.raises(ValueError, match="matrix mismatch"):
        export_predictions(
            stage="development",
            project_root=ROOT,
            protocol_path=protocol_path,
            task_manifest_path=manifest_path,
            artifact_path=artifact_path,
            records_path=records_path,
            output_dir=tmp_path / "incomplete",
        )
