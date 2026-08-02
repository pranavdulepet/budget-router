from __future__ import annotations

import hashlib
import json

from budget_router.serialization import stable_hash, stable_json
from scripts.evaluate_isolated_stage_gate import evaluate_gate
from scripts.train_isolated_stage_gate import train_gate


def test_frozen_gate_evaluation_compares_both_complete_paired_arms() -> None:
    training_rows = []
    for index in range(35):
        candidate_win = index % 5 == 0
        fixed_loss = index % 7 == 0 and not candidate_win
        training_rows.append(
            {
                "task_id": f"train-{index}",
                "repository": f"repo-{index % 5}",
                "problem_statement": f"regression issue {index}",
                "candidate_resolved": candidate_win or not fixed_loss,
                "fixed_resolved": fixed_loss or index % 3 == 0,
                "candidate_win": candidate_win,
                "fixed_loss": fixed_loss,
                "candidate_cost_usd": "0.20",
                "fixed_cost_usd": "0.40",
            }
        )
    test_ids = [f"test-{index}" for index in range(39)]
    protocol = {
        "study_id": "study",
        "manifest_hash": "manifest",
        "fixed_policy": {"policy_id": "fixed"},
        "stages": {"test": {"task_ids": test_ids}},
        "gate": {
            "dimension": 64,
            "hash_seed": "test-gate",
            "learning_rates": [0.2],
            "l2_values": [0.01],
            "epochs": 5,
            "cost_weights": [0.1],
            "decision_thresholds": [0.0],
        },
    }
    artifact = train_gate(
        protocol,
        {"selected_policy_id": "candidate"},
        training_rows,
    )
    artifact_bytes = (stable_json(artifact) + "\n").encode()
    freeze = {
        "study_manifest_hash": "manifest",
        "selected_candidate_policy_id": "candidate",
        "router_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "router_artifact_hash": artifact["artifact_hash"],
        "freeze_hash": "gate-freeze",
    }
    tasks = {
        "records": [
            {
                "instance_id": task_id,
                "problem_statement": f"held out issue {index}",
                "repo": f"heldout-{index % 6}",
            }
            for index, task_id in enumerate(test_ids)
        ]
    }
    grades = {
        "task_results": [
            {
                "task_id": task_id,
                "model": policy,
                "status": (
                    "resolved"
                    if (
                        (policy == "candidate" and index % 2 == 0)
                        or (policy == "fixed" and index % 3 == 0)
                    )
                    else "graded_unresolved"
                ),
                "conservative_cost_usd": (
                    "0.20" if policy == "candidate" else "0.40"
                ),
            }
            for index, task_id in enumerate(test_ids)
            for policy in ("candidate", "fixed")
        ]
    }

    result = evaluate_gate(
        protocol,
        freeze,
        json.loads(artifact_bytes),
        tasks,
        [grades],
        artifact_bytes=artifact_bytes,
        episodes=[
            {
                "study_stage": "test",
                "model": "candidate",
                "task_id": task_id,
                "scout_cost_usd": "0.01",
                "finisher_cost_usd": "0.19",
                "scout_calls": 2,
                "finisher_calls": 3,
                "structurally_valid": True,
                "submitted_patch": index % 2 == 0,
                "visible_handoff_chars": 100,
            }
            for index, task_id in enumerate(test_ids)
        ],
    )

    assert result["test_task_count"] == 39
    assert result["policies"]["fixed"]["resolved_count"] == 13
    assert result["policies"]["universal_candidate"]["resolved_count"] == 20
    assert len(result["selections"]) == 39
    assert result["candidate_phase_behavior"]["scout_total_calls"] == 78
    assert result["candidate_phase_behavior"]["candidate_cheaper_task_count"] == 39
    unhashed = dict(result)
    claimed = unhashed.pop("evaluation_hash")
    assert stable_hash(unhashed) == claimed
