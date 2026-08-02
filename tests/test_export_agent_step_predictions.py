from __future__ import annotations

import json
from pathlib import Path

import pytest

from budget_router.agent_step import FrozenAgentStepArtifact
from budget_router.serialization import stable_json
from scripts.export_agent_step_predictions import LABELS, export_predictions
from scripts.run_agent_step_study import HELDOUT_POLICIES

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.local_research


def test_export_requires_and_materializes_complete_policy_matrix(
    tmp_path: Path,
) -> None:
    task_manifest = json.loads(
        (ROOT / "artifacts/active_router_task_manifest.json").read_text()
    )
    artifact = FrozenAgentStepArtifact.from_dict(
        json.loads(
            (
                ROOT / "outputs/agent_step_router_v1/router_artifact.json"
            ).read_text()
        )
    )
    records_path = tmp_path / "episodes.jsonl"
    rows = []
    for task in task_manifest["heldout"]:
        for policy_id in HELDOUT_POLICIES:
            rows.append(
                {
                    "study_stage": "heldout",
                    "task_id": task["task_id"],
                    "policy_id": policy_id,
                    "router_artifact_hash": artifact.artifact_hash,
                    "submitted_patch": False,
                    "structurally_valid": True,
                    "conservative_cost_usd": "0.01",
                }
            )
    records_path.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = export_predictions(
        project_root=ROOT,
        protocol_path=(
            ROOT
            / "artifacts/"
            "active_router_protocol_amendment_007_agent_step_routing.json"
        ),
        task_manifest_path=ROOT / "artifacts/active_router_task_manifest.json",
        artifact_path=(
            ROOT / "outputs/agent_step_router_v1/router_artifact.json"
        ),
        records_path=records_path,
        output_dir=tmp_path / "predictions",
    )

    assert set(result["models"]) == set(LABELS.values())
    assert all(
        value["expected_task_count"] == 20
        for value in result["models"].values()
    )
    assert all(
        value["submitted_count"] == 0 for value in result["models"].values()
    )
