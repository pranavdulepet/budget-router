from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_HARNESS_COMMIT = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
EXPECTED_DATASET_SHA256 = (
    "e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(model: str) -> str:
    if model.startswith("fixed:"):
        return "isolated-stage-test-fixed-v1"
    if model.startswith("isolated:"):
        return "isolated-stage-test-candidate-v1"
    raise ValueError(f"unexpected isolated-stage test policy: {model}")


def _report_filename(model: str, run_id: str) -> str:
    return f"{model.replace('/', '__')}.{run_id}.json"


def validate_report(
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> list[str]:
    expected = {str(row["instance_id"]) for row in predictions}
    submitted = set(map(str, report.get("submitted_ids", [])))
    if submitted != expected:
        raise ValueError(
            "official grader report submission mismatch: "
            f"missing={sorted(expected - submitted)}, "
            f"extra={sorted(submitted - expected)}"
        )
    completed = set(map(str, report.get("completed_ids", [])))
    errors = set(map(str, report.get("error_ids", [])))
    if completed & errors:
        raise ValueError("official grader report has overlapping completed/error IDs")
    if completed | errors != expected:
        raise ValueError("official grader report is not terminal for every submission")
    resolved = set(map(str, report.get("resolved_ids", [])))
    unresolved = set(map(str, report.get("unresolved_ids", [])))
    if resolved | unresolved != completed or resolved & unresolved:
        raise ValueError("official grader report has inconsistent completed outcomes")
    return sorted(errors)


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
    predictions_manifest = predictions_manifest.resolve()
    output_dir = output_dir.resolve()
    # Keep the virtual-environment path intact. Resolving its interpreter
    # symlink can make a child process lose the environment's site-packages.
    python = python.absolute()

    dataset_hash = _sha256(dataset)
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"dataset snapshot hash mismatch: {dataset_hash}"
        )
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
    if manifest.get("study_stage") != "test":
        raise ValueError("prediction manifest is not the held-out test export")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    declared_model_errors = set(model_caused_error_keys or ())
    classified_model_errors: set[tuple[str, str]] = set()
    unclassified_errors: set[tuple[str, str]] = set()

    for model, model_manifest in sorted(manifest["models"].items()):
        raw_predictions_path = model_manifest.get("predictions_path")
        if not raw_predictions_path:
            raise ValueError(f"test policy has no submitted predictions: {model}")
        predictions_path = Path(str(raw_predictions_path))
        if not predictions_path.is_absolute():
            predictions_path = (project_root / predictions_path).resolve()
        predictions = _read_jsonl(predictions_path)
        if len(predictions) != int(model_manifest["submitted_count"]):
            raise ValueError(f"prediction count mismatch for {model}")
        run_id = _run_id(model)
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
        print(
            json.dumps(
                {
                    "event": "official_grading_start",
                    "model": model,
                    "run_id": run_id,
                    "submitted_count": len(predictions),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(command, cwd=harness_checkout)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)

        source_report = harness_checkout / _report_filename(model, run_id)
        report = _load(source_report)
        errors = validate_report(report, predictions)
        model_errors = sorted(
            task_id
            for task_id in errors
            if (model, task_id) in declared_model_errors
        )
        unexpected_errors = sorted(set(errors) - set(model_errors))
        classified_model_errors.update((model, task_id) for task_id in model_errors)
        unclassified_errors.update(
            (model, task_id) for task_id in unexpected_errors
        )
        destination_report = output_dir / source_report.name
        shutil.copy2(source_report, destination_report)
        source_logs = harness_checkout / "logs" / "run_evaluation" / run_id
        destination_logs = output_dir / "logs" / "run_evaluation" / run_id
        if source_logs.exists():
            shutil.copytree(source_logs, destination_logs, dirs_exist_ok=True)
        results[model] = {
            "predictions_path": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "report_path": str(destination_report),
            "report_sha256": _sha256(destination_report),
            "run_id": run_id,
            "submitted_count": len(predictions),
            "completed_count": len(report["completed_ids"]),
            "resolved_count": len(report["resolved_ids"]),
            "model_caused_error_ids": model_errors,
            "unclassified_error_ids": unexpected_errors,
        }
        print(
            json.dumps(
                {
                    "event": "official_grading_policy_complete",
                    "model": model,
                    "resolved_count": len(report["resolved_ids"]),
                    "model_caused_error_ids": model_errors,
                    "unclassified_error_ids": unexpected_errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    missing_declared_errors = declared_model_errors - classified_model_errors
    grading_manifest = {
        "schema_version": "isolated-stage-official-grading-v1",
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_hash,
        "harness_checkout": str(harness_checkout),
        "harness_commit": commit,
        "predictions_manifest": str(predictions_manifest),
        "predictions_manifest_sha256": _sha256(predictions_manifest),
        "models": results,
        "complete_without_official_error_ids": not (
            classified_model_errors or unclassified_errors
        ),
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
        json.dumps(grading_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if unclassified_errors or missing_declared_errors:
        raise RuntimeError(
            "official grading has unclassified or absent-declared errors; "
            "inspect copied run logs, classify model-caused patch failures, "
            "and rerun transient failures"
        )
    return grading_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe, one-worker official grading for the isolated-stage "
            "held-out test."
        )
    )
    parser.add_argument(
        "--predictions-manifest",
        type=Path,
        default=Path(
            "outputs/isolated_stage_v1/swebench_grader/test/predictions/manifest.json"
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
        default=Path("outputs/isolated_stage_v1/swebench_grader/test"),
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
