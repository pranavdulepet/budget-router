#!/usr/bin/env python3
"""Run the pinned official SWE-bench grader for an active router stage."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.run_isolated_stage_official_grading import (
        EXPECTED_DATASET_SHA256,
        EXPECTED_HARNESS_COMMIT,
        _load,
        _read_jsonl,
        _sha256,
        validate_report,
    )
except ModuleNotFoundError:
    from run_isolated_stage_official_grading import (  # type: ignore[no-redef]
        EXPECTED_DATASET_SHA256,
        EXPECTED_HARNESS_COMMIT,
        _load,
        _read_jsonl,
        _sha256,
        validate_report,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def grade_predictions(
    *,
    predictions_manifest: Path,
    project_root: Path,
    dataset: Path,
    harness_checkout: Path,
    output_dir: Path,
    python: Path,
    timeout_seconds: int,
    model_caused_error_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    project_root = project_root.resolve()
    harness_checkout = harness_checkout.resolve()
    output_dir = output_dir.resolve()
    python = python.absolute()
    if _sha256(dataset) != EXPECTED_DATASET_SHA256:
        raise ValueError("dataset snapshot hash mismatch")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=harness_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_HARNESS_COMMIT:
        raise ValueError(f"official harness commit mismatch: {commit}")

    manifest = _load(predictions_manifest)
    stage = str(manifest["study_stage"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    declared_model_errors = set(model_caused_error_keys or ())
    classified_model_errors: set[tuple[str, str]] = set()
    unclassified_errors: set[tuple[str, str]] = set()
    for model, model_manifest in sorted(manifest["models"].items()):
        predictions_path = Path(str(model_manifest["predictions_path"]))
        if not predictions_path.is_absolute():
            predictions_path = (project_root / predictions_path).resolve()
        predictions = _read_jsonl(predictions_path)
        if len(predictions) != int(model_manifest["submitted_count"]):
            raise ValueError(f"prediction count mismatch for {model}")
        run_id = f"active-router-{_slug(stage)}-{_slug(model)}-v1"
        if predictions:
            source_report = (
                harness_checkout
                / f"{model.replace('/', '__')}.{run_id}.json"
            )
            command = [
                str(python),
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(dataset),
                "--split",
                "test",
                "--predictions_path",
                str(predictions_path),
                "--max_workers",
                "1",
                "--timeout",
                str(timeout_seconds),
                "--cache_level",
                "env",
                "--clean",
                "false",
                "--run_id",
                run_id,
                "--namespace",
                "swebench",
            ]
            reuse_report = False
            if source_report.exists():
                prior_report = _load(source_report)
                try:
                    validate_report(prior_report, predictions)
                except ValueError:
                    pass
                else:
                    reuse_report = True
            event = (
                "official_grading_reuse_terminal_report"
                if reuse_report
                else "official_grading_start"
            )
            print(
                json.dumps(
                    {
                        "event": event,
                        "model": model,
                        "submitted_count": len(predictions),
                        "run_id": run_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not reuse_report:
                completed = subprocess.run(command, cwd=harness_checkout)
                if completed.returncode:
                    raise subprocess.CalledProcessError(
                        completed.returncode, command
                    )
            report = _load(source_report)
            errors = validate_report(report, predictions)
            model_errors = sorted(
                task_id
                for task_id in errors
                if (model, task_id) in declared_model_errors
            )
            unexpected_errors = sorted(set(errors) - set(model_errors))
            classified_model_errors.update(
                (model, task_id) for task_id in model_errors
            )
            unclassified_errors.update(
                (model, task_id) for task_id in unexpected_errors
            )
            report_path = output_dir / source_report.name
            shutil.copy2(source_report, report_path)
            source_logs = harness_checkout / "logs" / "run_evaluation" / run_id
            if source_logs.exists():
                shutil.copytree(
                    source_logs,
                    output_dir / "logs" / "run_evaluation" / run_id,
                    dirs_exist_ok=True,
                )
            resolved_ids = sorted(map(str, report["resolved_ids"]))
            completed_count = len(report["completed_ids"])
            terminal_count = completed_count + len(errors)
            report_hash = _sha256(report_path)
        else:
            report_path = None
            resolved_ids = []
            completed_count = 0
            terminal_count = 0
            report_hash = None
            model_errors = []
            unexpected_errors = []
        results[model] = {
            "predictions_path": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "report_path": str(report_path) if report_path else None,
            "report_sha256": report_hash,
            "run_id": run_id,
            "submitted_count": len(predictions),
            "official_completed_count": completed_count,
            "official_terminal_count": terminal_count,
            "resolved_ids": resolved_ids,
            "resolved_count": len(resolved_ids),
            "model_caused_error_ids": model_errors,
            "unclassified_error_ids": unexpected_errors,
            "no_submission_task_ids": model_manifest["no_submission_task_ids"],
            "expected_task_count": model_manifest["expected_task_count"],
            "structurally_valid_count": model_manifest[
                "structurally_valid_count"
            ],
            "canonical_episode_cost_usd": model_manifest[
                "canonical_episode_cost_usd"
            ],
        }
        print(
            json.dumps(
                {
                    "event": "official_grading_model_complete",
                    "model": model,
                    "resolved_count": len(resolved_ids),
                    "model_caused_error_ids": model_errors,
                    "unclassified_error_ids": unexpected_errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    missing_declared_errors = declared_model_errors - classified_model_errors
    result = {
        "schema_version": "active-router-official-grading-v1",
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "task_manifest_hash": manifest["task_manifest_hash"],
        "study_stage": stage,
        "dataset_path": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "harness_checkout": str(harness_checkout),
        "harness_commit": commit,
        "predictions_manifest": str(predictions_manifest),
        "predictions_manifest_sha256": _sha256(predictions_manifest),
        "models": results,
        "complete_with_no_unclassified_errors": not (
            unclassified_errors or missing_declared_errors
        ),
        "classified_model_caused_errors": [
            {"model": model, "task_id": task_id}
            for model, task_id in sorted(classified_model_errors)
        ],
        "unclassified_errors": [
            {"model": model, "task_id": task_id}
            for model, task_id in sorted(unclassified_errors)
        ],
        "declared_but_absent_model_errors": [
            {"model": model, "task_id": task_id}
            for model, task_id in sorted(missing_declared_errors)
        ],
    }
    (output_dir / "grading_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if unclassified_errors or missing_declared_errors:
        raise RuntimeError(
            "official grading has unclassified or absent-declared errors; "
            "inspect copied run logs, classify model-caused patch failures, "
            "and rerun transient failures"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-manifest",
        type=Path,
        default=Path(
            "outputs/active_router_v1/swebench_grader/"
            "cheap_medium_screen/predictions/manifest.json"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/private/tmp/"
            "swebench_verified_91aa3ed51b709be6457e12d00300a6a596d4c6a3.json"
        ),
    )
    parser.add_argument(
        "--harness-checkout",
        type=Path,
        default=Path("/private/tmp/model-router-swebench-official"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/active_router_v1/swebench_grader/cheap_medium_screen"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--model-caused-error",
        action="append",
        default=[],
        metavar="POLICY_ID=TASK_ID",
        help=(
            "Official error confirmed from its saved run log to be caused by "
            "the submitted model patch. May be repeated."
        ),
    )
    args = parser.parse_args()
    model_caused_error_keys: set[tuple[str, str]] = set()
    for value in args.model_caused_error:
        try:
            model, task_id = value.rsplit("=", 1)
        except ValueError:
            parser.error("--model-caused-error must be POLICY_ID=TASK_ID")
        if not model or not task_id:
            parser.error("--model-caused-error must be POLICY_ID=TASK_ID")
        model_caused_error_keys.add((model, task_id))
    result = grade_predictions(
        predictions_manifest=args.predictions_manifest,
        project_root=args.project_root,
        dataset=args.dataset,
        harness_checkout=args.harness_checkout,
        output_dir=args.output_dir,
        python=args.python,
        timeout_seconds=args.timeout_seconds,
        model_caused_error_keys=model_caused_error_keys,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
