#!/usr/bin/env python3
"""Export canonical active-study submissions for the official SWE-bench grader."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl
try:
    from scripts.run_active_router_study import (
        _excluded_models,
        _invalidated_episode_keys,
        _load_json,
        _sha256,
        _stage_scope,
    )
except ModuleNotFoundError:
    from run_active_router_study import (  # type: ignore[no-redef]
        _excluded_models,
        _invalidated_episode_keys,
        _load_json,
        _sha256,
        _stage_scope,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("_")


def export_predictions(
    *,
    project_root: Path,
    protocol_path: Path,
    manifest_path: Path,
    records_path: Path,
    exclusions_path: Path,
    stage: str,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    task_manifest = _load_json(manifest_path)
    protocol_hash = _sha256(protocol_path)
    models, task_ids = _stage_scope(protocol, task_manifest, stage)
    excluded = _excluded_models(
        exclusions_path,
        study_id=str(protocol["study_id"]),
        protocol_sha256=protocol_hash,
        stage=stage,
    )
    models = [model for model in models if model not in excluded]
    invalidated = _invalidated_episode_keys(project_root, protocol)
    records = [
        row
        for row in read_jsonl(records_path)
        if row.get("study_stage") == stage
        and row.get("model") in models
        and row.get("task_id") in task_ids
        and (
            str(row.get("study_stage", "")),
            str(row["task_id"]),
            str(row["model"]),
            str(row.get("protocol_sha256", "")),
        )
        not in invalidated
    ]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["task_id"]), str(row["model"]))
        if key in by_key:
            raise ValueError(f"duplicate canonical episode: {key}")
        by_key[key] = row
    expected = {(task_id, model) for task_id in task_ids for model in models}
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError(f"canonical screen matrix is incomplete: {missing}")

    predictions: dict[str, list[dict[str, str]]] = defaultdict(list)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload: dict[str, Any] = {}
    for model in models:
        model_records = [by_key[(task_id, model)] for task_id in task_ids]
        no_submission = []
        for row in model_records:
            if not row.get("submitted_patch"):
                no_submission.append(str(row["task_id"]))
                continue
            trajectory_path = Path(str(row["trajectory"]))
            if not trajectory_path.is_absolute():
                trajectory_path = project_root / trajectory_path
            trajectory = _load_json(trajectory_path)
            patch = str(trajectory.get("info", {}).get("submission", ""))
            if not patch:
                raise ValueError(
                    f"{row['task_id']} is marked submitted but has no patch"
                )
            actual_hash = hashlib.sha256(patch.encode()).hexdigest()
            if actual_hash != row.get("submission_sha256"):
                raise ValueError(f"submission hash mismatch: {row['task_id']}")
            if scan_for_secrets(patch):
                raise ValueError("refusing to export a patch containing a secret")
            predictions[model].append(
                {
                    "instance_id": str(row["task_id"]),
                    "model_name_or_path": model,
                    "model_patch": patch,
                }
            )
        prediction_path = output_dir / f"{_slug(model)}.jsonl"
        prediction_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in sorted(
                    predictions[model],
                    key=lambda value: value["instance_id"],
                )
            ),
            encoding="utf-8",
        )
        model_payload[model] = {
            "predictions_path": str(prediction_path),
            "submitted_count": len(predictions[model]),
            "no_submission_task_ids": sorted(no_submission),
            "structurally_valid_count": sum(
                bool(row["structurally_valid"]) for row in model_records
            ),
            "expected_task_count": len(task_ids),
            "canonical_episode_cost_usd": str(
                sum(
                    (
                        __import__("decimal").Decimal(
                            str(row["conservative_cost_usd"])
                        )
                        for row in model_records
                    ),
                    __import__("decimal").Decimal("0"),
                )
            ),
        }
    result = {
        "schema_version": "active-router-predictions-v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "task_manifest_hash": task_manifest["manifest_hash"],
        "study_stage": stage,
        "task_ids": task_ids,
        "excluded_models": sorted(excluded),
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
        default=Path("artifacts/active_router_protocol.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/active_router_task_manifest.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/active_router_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--screen-exclusions",
        type=Path,
        default=Path("outputs/active_router_v1/screen_exclusions.json"),
    )
    parser.add_argument("--stage", default="cheap_medium_screen")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/active_router_v1/swebench_grader/"
            "cheap_medium_screen/predictions"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = export_predictions(
        project_root=args.project_root,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        records_path=args.records,
        exclusions_path=args.screen_exclusions,
        stage=args.stage,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
