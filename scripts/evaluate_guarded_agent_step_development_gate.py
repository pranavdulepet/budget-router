#!/usr/bin/env python3
"""Evaluate the frozen Amendment 008 paid development activation gate."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.guarded_agent_step import GuardedAgentStepArtifact
from budget_router.serialization import read_jsonl

try:
    from scripts.export_guarded_agent_step_predictions import LABELS
    from scripts.run_guarded_agent_step_followup import ROUTED
except ModuleNotFoundError:
    from export_guarded_agent_step_predictions import LABELS  # type: ignore[no-redef]
    from run_guarded_agent_step_followup import ROUTED  # type: ignore[no-redef]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_development_gate(
    *,
    protocol: dict[str, Any],
    task_manifest: dict[str, Any],
    artifact: GuardedAgentStepArtifact,
    records: list[dict[str, Any]],
    grading: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = {str(row["task_id"]) for row in task_manifest["development"]}
    development = [
        row
        for row in records
        if row.get("study_stage") == "development"
        and row.get("policy_id") == ROUTED
    ]
    by_id = {str(row["task_id"]): row for row in development}
    if len(by_id) != len(development):
        raise ValueError("duplicate development routed records")
    if set(by_id) != expected_ids:
        raise ValueError("development routed records do not match frozen tasks")
    if any(
        row.get("router_artifact_hash") != artifact.artifact_hash
        for row in development
    ):
        raise ValueError("development record artifact mismatch")
    label = LABELS[ROUTED]
    if grading.get("study_stage") != "development":
        raise ValueError("official grading stage mismatch")
    if label not in grading.get("models", {}):
        raise ValueError("official grading is missing the guarded router")
    official = grading["models"][label]
    if int(official["expected_task_count"]) != len(expected_ids):
        raise ValueError("official grading task count mismatch")

    cheap_calls = sum(int(row["cheap_calls"]) for row in development)
    strong_calls = sum(int(row["strong_calls"]) for row in development)
    total_calls = cheap_calls + strong_calls
    cheap_share = cheap_calls / total_calls if total_calls else 0.0
    trajectories_with_cheap = sum(
        int(row["cheap_calls"]) > 0 for row in development
    )
    structurally_valid = sum(
        bool(row["structurally_valid"]) for row in development
    )
    provider_failures = sum(
        bool(row.get("provider_failed")) for row in development
    )
    guard_violations = [
        str(row["task_id"])
        for row in development
        if (
            row.get("guard_compliant") is not True
            or int(row.get("initial_guard_violations", 0)) != 0
            or int(row.get("maximum_observed_consecutive_cheap_calls", 0))
            > artifact.maximum_consecutive_cheap_calls
        )
    ]
    resolved = int(official["resolved_count"])
    unclassified_errors = grading.get("unclassified_errors", [])
    missing_declared = grading.get("declared_but_absent_model_errors", [])
    gate = protocol["development_gate"]
    checks = {
        "episode_count": len(development) == int(gate["tasks"]),
        "minimum_actual_cheap_call_share": (
            cheap_share >= float(gate["minimum_actual_cheap_call_share"])
        ),
        "maximum_actual_cheap_call_share": (
            cheap_share <= float(gate["maximum_actual_cheap_call_share"])
        ),
        "minimum_trajectories_with_cheap_call": (
            trajectories_with_cheap
            >= int(gate["minimum_trajectories_with_cheap_call"])
        ),
        "minimum_structurally_valid_episodes": (
            structurally_valid
            >= int(gate["minimum_structurally_valid_episodes"])
        ),
        "minimum_official_resolutions": (
            resolved >= int(gate["minimum_official_resolutions"])
        ),
        "maximum_provider_failures": (
            provider_failures <= int(gate["maximum_provider_failures"])
        ),
        "no_unclassified_grader_errors": (
            not unclassified_errors
            and not missing_declared
            and grading.get("complete_with_no_unclassified_errors") is True
        ),
        "guard_compliance": not guard_violations,
    }
    passed = all(checks.values())
    return {
        "schema_version": "guarded-agent-step-development-gate-v2",
        "status": (
            "heldout_collection_authorized"
            if passed
            else "heldout_collection_blocked"
        ),
        "heldout_collection_authorized": passed,
        "artifact_hash": artifact.artifact_hash,
        "task_manifest_hash": task_manifest["manifest_hash"],
        "checks": checks,
        "metrics": {
            "episodes": len(development),
            "cheap_calls": cheap_calls,
            "strong_calls": strong_calls,
            "actual_cheap_call_share": cheap_share,
            "trajectories_with_cheap_call": trajectories_with_cheap,
            "structurally_valid_episodes": structurally_valid,
            "official_resolved_count": resolved,
            "official_resolved_ids": official["resolved_ids"],
            "provider_failure_episodes": provider_failures,
            "guard_violation_task_ids": sorted(guard_violations),
            "unclassified_grader_errors": unclassified_errors,
            "classified_model_caused_errors": grading.get(
                "classified_model_caused_errors", []
            ),
            "total_conservative_cost_usd": str(
                sum(
                    (
                        Decimal(str(row["conservative_cost_usd"]))
                        for row in development
                    ),
                    Decimal("0"),
                )
            ),
        },
        "result_contingent_threshold_change_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/agent_step_followup_v2_task_manifest.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v2/router_artifact.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/agent_step_router_v2/dynamic_episodes.jsonl"),
    )
    parser.add_argument(
        "--grading",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v2/swebench_grader/development/"
            "grading_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v2/development_gate_official.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite development gate: {args.output}")
    result = evaluate_development_gate(
        protocol=_load(args.protocol),
        task_manifest=_load(args.task_manifest),
        artifact=GuardedAgentStepArtifact.from_dict(_load(args.artifact)),
        records=read_jsonl(args.records),
        grading=_load(args.grading),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
