from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model).strip("_")


def export_predictions(
    episodes_path: Path,
    output_dir: Path,
    *,
    project_root: Path,
    study_stage: str | None = None,
) -> dict[str, Any]:
    records = _read_jsonl(episodes_path)
    if study_stage is not None:
        records = [
            record
            for record in records
            if record.get("study_stage") == study_stage
        ]
        if not records:
            raise ValueError(f"no episodes found for study stage: {study_stage}")
    predictions: dict[str, list[dict[str, str]]] = defaultdict(list)
    unresolved: dict[str, int] = defaultdict(int)

    for record in records:
        model = record["model"]
        if not record.get("submitted_patch"):
            unresolved[model] += 1
            continue

        trajectory_path = Path(record["trajectory"])
        if not trajectory_path.is_absolute():
            trajectory_path = project_root / trajectory_path
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        patch = trajectory.get("info", {}).get("submission")
        if not patch:
            raise ValueError(
                f"{record['task_id']} is marked submitted but its trajectory has no submission"
            )

        expected_sha256 = record.get("submission_sha256")
        actual_sha256 = hashlib.sha256(patch.encode()).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"{record['task_id']} submission hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        predictions[model].append(
            {
                "instance_id": record["task_id"],
                "model_name_or_path": model,
                "model_patch": patch,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_files: dict[str, str] = {}
    submitted_counts: dict[str, int] = {}
    for model, model_predictions in sorted(predictions.items()):
        model_predictions.sort(key=lambda prediction: prediction["instance_id"])
        output_path = output_dir / f"{_model_slug(model)}.jsonl"
        output_path.write_text(
            "".join(
                json.dumps(prediction, sort_keys=True) + "\n"
                for prediction in model_predictions
            ),
            encoding="utf-8",
        )
        model_files[model] = str(output_path)
        submitted_counts[model] = len(model_predictions)

    models = sorted({record["model"] for record in records})
    manifest = {
        "schema_version": "swebench-pilot-predictions-v1",
        "episodes_path": str(episodes_path),
        "study_stage": study_stage,
        "episode_count": len(records),
        "submitted_count": sum(submitted_counts.values()),
        "unresolved_without_submission_count": sum(unresolved.values()),
        "models": {
            model: {
                "predictions_path": model_files.get(model),
                "submitted_count": submitted_counts.get(model, 0),
                "unresolved_without_submission_count": unresolved.get(model, 0),
            }
            for model in models
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export saved pilot submissions for the official SWE-bench grader."
    )
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--study-stage")
    args = parser.parse_args()

    manifest = export_predictions(
        args.episodes,
        args.output_dir,
        project_root=args.project_root,
        study_stage=args.study_stage,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
