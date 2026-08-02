#!/usr/bin/env python3
"""Export the complete frozen agent-step matrix for official SWE-bench grading."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.agent_step import FrozenAgentStepArtifact
from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl

try:
    from scripts.run_agent_step_study import (
        FIXED_CHEAP,
        FIXED_STRONG,
        HELDOUT_POLICIES,
        ROUTED,
    )
except ModuleNotFoundError:
    from run_agent_step_study import (  # type: ignore[no-redef]
        FIXED_CHEAP,
        FIXED_STRONG,
        HELDOUT_POLICIES,
        ROUTED,
    )

LABELS = {
    FIXED_CHEAP: "agent_step_fixed_gpt_oss_20b",
    FIXED_STRONG: "agent_step_fixed_qwen35",
    ROUTED: "agent_step_router",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_predictions(
    *,
    project_root: Path,
    protocol_path: Path,
    task_manifest_path: Path,
    artifact_path: Path,
    records_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    manifest = _load(task_manifest_path)
    artifact = FrozenAgentStepArtifact.from_dict(_load(artifact_path))
    task_ids = [str(row["task_id"]) for row in manifest["heldout"]]
    records = [
        row
        for row in read_jsonl(records_path)
        if row.get("study_stage") == "heldout"
    ]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["task_id"]), str(row["policy_id"]))
        if key in by_key:
            raise ValueError(f"duplicate held-out agent-step record: {key}")
        if row.get("router_artifact_hash") != artifact.artifact_hash:
            raise ValueError(f"record has a different router artifact: {key}")
        by_key[key] = row
    expected = {
        (task_id, policy_id)
        for task_id in task_ids
        for policy_id in HELDOUT_POLICIES
    }
    missing = sorted(expected - set(by_key))
    unexpected = sorted(set(by_key) - expected)
    if missing or unexpected:
        raise ValueError(
            f"held-out agent-step matrix mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    for policy_id in HELDOUT_POLICIES:
        label = LABELS[policy_id]
        predictions: list[dict[str, str]] = []
        no_submission: list[str] = []
        policy_records = [by_key[(task_id, policy_id)] for task_id in task_ids]
        for row in policy_records:
            task_id = str(row["task_id"])
            if not row.get("submitted_patch"):
                no_submission.append(task_id)
                continue
            trajectory_path = Path(str(row["trajectory"]))
            if not trajectory_path.is_absolute():
                trajectory_path = project_root / trajectory_path
            trajectory = _load(trajectory_path)
            patch = str(trajectory.get("info", {}).get("submission", ""))
            if not patch:
                raise ValueError(
                    f"{policy_id} {task_id} is submitted but its patch is absent"
                )
            if hashlib.sha256(patch.encode()).hexdigest() != row.get(
                "submission_sha256"
            ):
                raise ValueError(f"submission hash mismatch: {policy_id} {task_id}")
            if scan_for_secrets(patch):
                raise ValueError("refusing to export a patch containing a secret")
            predictions.append(
                {
                    "instance_id": task_id,
                    "model_name_or_path": label,
                    "model_patch": patch,
                }
            )
        predictions.sort(key=lambda value: value["instance_id"])
        predictions_path = output_dir / f"{label}.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
            encoding="utf-8",
        )
        models[label] = {
            "policy_id": policy_id,
            "predictions_path": str(predictions_path),
            "submitted_count": len(predictions),
            "no_submission_task_ids": sorted(no_submission),
            "structurally_valid_count": sum(
                bool(row["structurally_valid"]) for row in policy_records
            ),
            "expected_task_count": len(task_ids),
            "canonical_episode_cost_usd": str(
                sum(
                    (
                        Decimal(str(row["conservative_cost_usd"]))
                        for row in policy_records
                    ),
                    Decimal("0"),
                )
            ),
        }
    result = {
        "schema_version": "agent-step-predictions-v1",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "protocol_sha256": _sha256(protocol_path),
        "task_manifest_hash": manifest["manifest_hash"],
        "router_artifact_hash": artifact.artifact_hash,
        "router_artifact_file_sha256": _sha256(artifact_path),
        "study_stage": "heldout",
        "task_ids": task_ids,
        "models": models,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/active_router_protocol_amendment_007_agent_step_routing.json"
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/active_router_task_manifest.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v1/router_artifact.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/agent_step_router_v1/dynamic_episodes.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v1/swebench_grader/heldout/predictions"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = export_predictions(
        project_root=args.project_root,
        protocol_path=args.protocol,
        task_manifest_path=args.task_manifest,
        artifact_path=args.artifact,
        records_path=args.records,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
