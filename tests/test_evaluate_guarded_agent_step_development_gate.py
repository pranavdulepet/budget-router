from __future__ import annotations

import json
from pathlib import Path

from budget_router.guarded_agent_step import GuardedAgentStepArtifact
from scripts.evaluate_guarded_agent_step_development_gate import (
    evaluate_development_gate,
)
from scripts.export_guarded_agent_step_predictions import LABELS
from scripts.run_guarded_agent_step_followup import ROUTED


ROOT = Path(__file__).resolve().parents[1]


def test_development_gate_requires_activation_quality_and_grader_health() -> None:
    protocol = json.loads(
        (
            ROOT
            / "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ).read_text()
    )
    artifact = GuardedAgentStepArtifact.from_dict(
        json.loads(
            (
                ROOT / "outputs/agent_step_router_v2/router_artifact.json"
            ).read_text()
        )
    )
    manifest = {
        "manifest_hash": "tasks",
        "development": [{"task_id": f"task-{index}"} for index in range(20)],
    }
    records = [
        {
            "study_stage": "development",
            "policy_id": ROUTED,
            "task_id": f"task-{index}",
            "router_artifact_hash": artifact.artifact_hash,
            "cheap_calls": 1 if index < 15 else 0,
            "strong_calls": 4,
            "structurally_valid": index < 19,
            "provider_failed": False,
            "guard_compliant": True,
            "initial_guard_violations": 0,
            "maximum_observed_consecutive_cheap_calls": 1,
            "conservative_cost_usd": "0.1",
        }
        for index in range(20)
    ]
    grading = {
        "study_stage": "development",
        "complete_with_no_unclassified_errors": True,
        "unclassified_errors": [],
        "declared_but_absent_model_errors": [],
        "classified_model_caused_errors": [],
        "models": {
            LABELS[ROUTED]: {
                "expected_task_count": 20,
                "resolved_count": 6,
                "resolved_ids": [f"task-{index}" for index in range(6)],
            }
        },
    }
    result = evaluate_development_gate(
        protocol=protocol,
        task_manifest=manifest,
        artifact=artifact,
        records=records,
        grading=grading,
    )
    assert result["heldout_collection_authorized"] is True
    assert result["metrics"]["actual_cheap_call_share"] == 0.15 / 0.95

    records[0]["provider_failed"] = True
    blocked = evaluate_development_gate(
        protocol=protocol,
        task_manifest=manifest,
        artifact=artifact,
        records=records,
        grading=grading,
    )
    assert blocked["heldout_collection_authorized"] is False
    assert blocked["checks"]["maximum_provider_failures"] is False
