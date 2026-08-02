#!/usr/bin/env python3
"""Export Amendment 009 episodes for the pinned official SWE-bench grader."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl, stable_hash


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("_")


def _task_ids(manifest: dict[str, Any], stage: str) -> list[str]:
    key = "tasks" if stage == "compatibility" else stage
    rows = manifest.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"task manifest has no list for stage: {stage}")
    result = [
        str(row["task_id"]) if isinstance(row, dict) else str(row)
        for row in rows
    ]
    if len(result) != len(set(result)):
        raise ValueError("task manifest contains duplicate task IDs")
    return result


def _manifest_identity(manifest: dict[str, Any]) -> str:
    claimed = manifest.get("manifest_hash")
    if claimed:
        unhashed = dict(manifest)
        unhashed.pop("manifest_hash", None)
        if stable_hash(unhashed) != claimed:
            raise ValueError("task manifest hash mismatch")
        return str(claimed)
    return stable_hash(manifest)


def export_predictions(
    *,
    project_root: Path,
    protocol_path: Path,
    task_manifest_path: Path,
    records_path: Path,
    stage: str,
    output_dir: Path,
    expected_models: list[str] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    protocol = _load(protocol_path)
    task_manifest = _load(task_manifest_path)
    task_ids = _task_ids(task_manifest, stage)
    records = read_jsonl(records_path)
    if stage != "compatibility":
        records = [
            row for row in records if str(row.get("study_stage")) == stage
        ]
    records = [row for row in records if str(row.get("task_id")) in task_ids]

    models = sorted(
        set(expected_models or ())
        or {str(row["model"]) for row in records}
    )
    if not models:
        raise ValueError("no models found in canonical records")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["task_id"]), str(row["model"]))
        if key in by_key:
            raise ValueError(f"duplicate canonical episode: {key}")
        by_key[key] = row
    expected = {(task_id, model) for task_id in task_ids for model in models}
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra:
        raise ValueError(
            f"canonical matrix mismatch: missing={missing}, extra={extra}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload: dict[str, Any] = {}
    for model in models:
        model_records = [by_key[(task_id, model)] for task_id in task_ids]
        predictions: list[dict[str, str]] = []
        no_submission: list[str] = []
        for row in model_records:
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
                    f"{task_id} is marked submitted but has no saved patch"
                )
            patch_hash = hashlib.sha256(patch.encode()).hexdigest()
            if patch_hash != row.get("submission_sha256"):
                raise ValueError(f"submission hash mismatch: {task_id}")
            if scan_for_secrets(patch):
                raise ValueError("refusing to export a patch containing a secret")
            predictions.append(
                {
                    "instance_id": task_id,
                    "model_name_or_path": model,
                    "model_patch": patch,
                }
            )
        prediction_path = output_dir / f"{_slug(model)}.jsonl"
        prediction_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in sorted(
                    predictions, key=lambda value: value["instance_id"]
                )
            ),
            encoding="utf-8",
        )
        model_payload[model] = {
            "predictions_path": str(prediction_path),
            "submitted_count": len(predictions),
            "no_submission_task_ids": sorted(no_submission),
            "structurally_valid_count": sum(
                bool(row.get("structurally_valid")) for row in model_records
            ),
            "expected_task_count": len(task_ids),
            "canonical_episode_cost_usd": str(
                sum(
                    (
                        Decimal(str(row["conservative_cost_usd"]))
                        for row in model_records
                    ),
                    Decimal("0"),
                )
            ),
        }

    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    result = {
        "schema_version": "quality-cost-predictions-v1",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "protocol_sha256": protocol_hash,
        "task_manifest_hash": _manifest_identity(task_manifest),
        "study_stage": stage,
        "task_ids": task_ids,
        "models": model_payload,
    }
    (output_dir / "manifest.json").write_text(
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
            "artifacts/"
            "active_router_protocol_amendment_009_three_tier_quality_cost.json"
        ),
    )
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("compatibility", "development", "heldout"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-model", action="append", default=[])
    args = parser.parse_args()
    result = export_predictions(
        project_root=args.project_root,
        protocol_path=args.protocol,
        task_manifest_path=args.task_manifest,
        records_path=args.records,
        stage=args.stage,
        output_dir=args.output_dir,
        expected_models=args.expected_model or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
