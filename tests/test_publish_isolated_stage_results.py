from __future__ import annotations

import hashlib

from budget_router.serialization import stable_hash, stable_json
from scripts.publish_isolated_stage_results import build_public_result


def test_public_isolated_result_reconciles_frozen_paired_study() -> None:
    candidate = "isolated:cheap->strong"
    fixed = "fixed:strong"
    protocol = {
        "study_id": "study",
        "design_timing": "frozen before outcomes",
        "research_question": "question",
        "sampling": {
            "gate_training_task_count": 35,
            "repository_disjoint_test": True,
        },
        "fixed_policy": {"policy_id": fixed},
        "candidate_policies": [{"policy_id": candidate}],
        "execution_policy": {"kind": "isolated"},
        "budget": {"maximum_new_cost_usd": "112.5"},
    }
    protocol["manifest_hash"] = stable_hash(protocol)
    candidate_freeze = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "freeze_hash": "candidate-freeze",
        "selected_policy_id": candidate,
        "test_outcomes_accessed": False,
        "screen_task_ids": ["screen"],
        "candidate_metrics": [
            {"total_conservative_cost_usd": "1.0"}
        ],
    }
    gate = {
        "training": {"task_count": 35},
        "cross_validation": {"resolved_count": 10},
        "selector": {"threshold": 0.0},
    }
    gate["artifact_hash"] = stable_hash(gate)
    gate_bytes = (stable_json(gate) + "\n").encode()
    gate_freeze = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "selected_candidate_policy_id": candidate,
        "freeze_hash": "gate-freeze",
        "router_artifact_hash": gate["artifact_hash"],
        "router_artifact_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "test_labels_accessed": False,
    }
    collection = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "completed_stage_episodes": 78,
        "submitted_stage_episodes": 47,
        "stage_conservative_cost_usd": "44",
        "total_incremental_exposure_usd": "73",
    }
    model_summary = {
        "submitted_count": 0,
        "graded_count": 0,
        "resolved_count": 0,
        "grader_error_count": 0,
        "unapplyable_patch_count": 0,
        "resolved_task_ids": [],
    }
    grades = {
        "grader": {
            "dataset": "dataset",
            "dataset_revision": "revision",
            "harness_commit": "commit",
            "submitted_patches_graded": 47,
            "grader_errors": 0,
            "unapplyable_model_patches": 0,
        },
        "overall": {
            "submitted_count": 47,
            "total_conservative_cost_usd": "44",
        },
        "models": {
            candidate: {
                **model_summary,
                "submitted_count": 19,
                "graded_count": 19,
                "resolved_count": 11,
            },
            fixed: {
                **model_summary,
                "submitted_count": 28,
                "graded_count": 28,
                "resolved_count": 10,
            },
        },
    }
    fixed_result = {"resolved_count": 10, "total_cost_usd": "20"}
    candidate_result = {"resolved_count": 11, "total_cost_usd": "22"}
    gate_result = {
        "resolved_count": 10,
        "total_cost_usd": "19",
        "candidate_selection_count": 5,
    }
    evaluation = {
        "study_id": "study",
        "study_manifest_hash": protocol["manifest_hash"],
        "candidate_policy_id": candidate,
        "fixed_policy_id": fixed,
        "test_task_count": 39,
        "policies": {
            "fixed": fixed_result,
            "universal_candidate": candidate_result,
            "learned_gate": gate_result,
            "post_hoc_task_oracle": {
                "resolved_count": 12,
                "total_cost_usd": "18",
            },
        },
        "paired_comparisons": {},
        "selections": [],
        "candidate_phase_behavior": {"scout_total_calls": 78},
        "provenance": {"test_labels_accessed_after_freeze": True},
    }
    evaluation["evaluation_hash"] = stable_hash(evaluation)
    grading_manifest = {
        "complete_with_no_unclassified_errors": True,
        "classified_model_caused_errors": [],
        "dataset_sha256": "dataset-sha",
    }
    reports = {
        candidate: {
            "submitted_instances": 19,
            "completed_instances": 19,
            "resolved_instances": 11,
            "error_instances": 0,
        },
        fixed: {
            "submitted_instances": 28,
            "completed_instances": 28,
            "resolved_instances": 10,
            "error_instances": 0,
        },
    }
    source_hashes = {
        "grading_manifest": "grading",
        "grades": "grades",
        "evaluation": "evaluation",
        f"official_report:{candidate}": "candidate-report",
        f"official_report:{fixed}": "fixed-report",
    }

    result = build_public_result(
        protocol,
        candidate_freeze,
        gate_freeze,
        gate,
        collection,
        grades,
        evaluation,
        grading_manifest,
        reports,
        gate_artifact_bytes=gate_bytes,
        source_sha256=source_hashes,
    )

    assert result["conclusion"]["learned_gate_beats_fixed"]
    assert result["conclusion"]["universal_candidate_beats_fixed"]
    assert result["conclusion"][
        "observed_oracle_incremental_resolutions_over_fixed"
    ] == 2
    assert result["cost_accounting"]["unspent_vs_frozen_ceiling_usd"] == "39.5"
    assert result["result_hash"]
