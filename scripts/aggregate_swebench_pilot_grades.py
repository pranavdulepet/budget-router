from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model).strip("_")


def aggregate_grades(
    episodes_path: Path,
    grader_dir: Path,
    *,
    dataset_revision: str,
    harness_commit: str,
    study_stage: str | None = None,
    model_caused_error_ids: set[str] | None = None,
    model_caused_error_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    declared_model_errors = set(model_caused_error_ids or ())
    declared_model_error_keys = set(model_caused_error_keys or ())
    classified_model_errors: set[str] = set()
    classified_model_error_keys: set[tuple[str, str]] = set()
    episodes = _read_jsonl(episodes_path)
    if study_stage is not None:
        episodes = [
            episode
            for episode in episodes
            if episode.get("study_stage") == study_stage
        ]
        if not episodes:
            raise ValueError(f"no episodes found for study stage: {study_stage}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_model.setdefault(episode["model"], []).append(episode)

    model_results: dict[str, dict[str, Any]] = {}
    task_results: list[dict[str, Any]] = []
    all_resolved_tasks: set[str] = set()

    for model, model_episodes in sorted(by_model.items()):
        report_prefixes = {
            model.replace("/", "__") + ".",
            _model_slug(model) + ".",
        }
        report_paths = sorted(
            {
                path
                for report_prefix in report_prefixes
                for path in grader_dir.glob(f"{report_prefix}*.json")
            }
        )
        submitted = [episode for episode in model_episodes if episode["submitted_patch"]]
        if submitted and len(report_paths) != 1:
            raise ValueError(
                f"expected one grader report for {model}, found {len(report_paths)}"
            )
        if not submitted and report_paths:
            raise ValueError(f"found an unexpected grader report for {model}")

        report = json.loads(report_paths[0].read_text()) if report_paths else {}
        submitted_ids = set(report.get("submitted_ids", []))
        expected_submitted_ids = {episode["task_id"] for episode in submitted}
        if submitted_ids != expected_submitted_ids:
            raise ValueError(f"submitted task mismatch for {model}")
        resolved_ids = set(report.get("resolved_ids", []))
        error_ids = set(report.get("error_ids", []))
        unapplyable_patch_ids = {
            task_id
            for task_id in error_ids
            if task_id in declared_model_errors
            or (model, task_id) in declared_model_error_keys
        }
        grader_error_ids = error_ids - unapplyable_patch_ids
        classified_model_errors.update(
            task_id
            for task_id in unapplyable_patch_ids
            if task_id in declared_model_errors
        )
        classified_model_error_keys.update(
            (model, task_id)
            for task_id in unapplyable_patch_ids
            if (model, task_id) in declared_model_error_keys
        )
        completed_ids = set(report.get("completed_ids", []))
        terminal_ids = completed_ids | error_ids
        if (
            terminal_ids != submitted_ids
            or completed_ids & error_ids
            or not resolved_ids <= completed_ids
        ):
            raise ValueError(
                f"grading was incomplete for {model}: "
                f"terminal={len(terminal_ids)}, submitted={len(submitted_ids)}, "
                f"errors={len(error_ids)}"
            )

        all_resolved_tasks.update(resolved_ids)
        total_cost = sum(
            (Decimal(str(episode["conservative_cost_usd"])) for episode in model_episodes),
            Decimal("0"),
        )
        total_latency = sum(float(episode["latency_seconds"]) for episode in model_episodes)
        resolved_episodes = [
            episode for episode in model_episodes if episode["task_id"] in resolved_ids
        ]
        resolved_count = len(resolved_episodes)
        model_results[model] = {
            "episode_count": len(model_episodes),
            "submitted_count": len(submitted_ids),
            "graded_count": len(terminal_ids),
            "resolved_count": resolved_count,
            "unresolved_submission_count": len(submitted_ids - resolved_ids),
            "unresolved_without_submission_count": len(model_episodes) - len(submitted_ids),
            "grader_error_count": len(grader_error_ids),
            "unapplyable_patch_count": len(unapplyable_patch_ids),
            "submission_pass_rate": _ratio(resolved_count, len(submitted_ids)),
            "overall_resolution_rate": _ratio(resolved_count, len(model_episodes)),
            "total_conservative_cost_usd": str(total_cost),
            "cost_per_resolved_task_usd": (
                str(total_cost / resolved_count) if resolved_count else None
            ),
            "total_episode_latency_seconds": total_latency,
            "latency_per_resolved_task_seconds": (
                total_latency / resolved_count if resolved_count else None
            ),
            "mean_latency_of_resolved_episodes_seconds": (
                sum(float(episode["latency_seconds"]) for episode in resolved_episodes)
                / resolved_count
                if resolved_count
                else None
            ),
            "resolved_task_ids": sorted(resolved_ids),
            "grader_report": str(report_paths[0]) if report_paths else None,
        }

        for episode in model_episodes:
            task_id = episode["task_id"]
            if task_id in resolved_ids:
                status = "resolved"
            elif task_id in unapplyable_patch_ids:
                status = "unapplyable_patch"
            elif task_id in grader_error_ids:
                status = "grader_error"
            elif task_id in submitted_ids:
                status = "graded_unresolved"
            else:
                status = "no_submission"
            task_results.append(
                {
                    "model": model,
                    "task_id": task_id,
                    "status": status,
                    "conservative_cost_usd": str(episode["conservative_cost_usd"]),
                    "latency_seconds": float(episode["latency_seconds"]),
                }
            )

    if classified_model_errors != declared_model_errors:
        missing = sorted(declared_model_errors - classified_model_errors)
        raise ValueError(
            "declared model-caused errors were not present in grader reports: "
            + ", ".join(missing)
        )
    if classified_model_error_keys != declared_model_error_keys:
        missing = sorted(
            declared_model_error_keys - classified_model_error_keys
        )
        raise ValueError(
            "declared model-qualified errors were not present in grader reports: "
            + ", ".join(f"{model}={task_id}" for model, task_id in missing)
        )

    total_cost = sum(
        (Decimal(str(episode["conservative_cost_usd"])) for episode in episodes),
        Decimal("0"),
    )
    resolved_count = sum(result["resolved_count"] for result in model_results.values())
    submitted_count = sum(result["submitted_count"] for result in model_results.values())
    total_latency = sum(float(episode["latency_seconds"]) for episode in episodes)
    best_model, best_result = max(
        model_results.items(),
        key=lambda item: (
            item[1]["resolved_count"],
            -Decimal(item[1]["total_conservative_cost_usd"]),
        ),
    )
    task_results.sort(key=lambda row: (row["task_id"], row["model"]))

    return {
        "schema_version": "swebench-pilot-grades-v1",
        "study_stage": study_stage,
        "grader": {
            "name": "official SWE-bench terminal harness",
            "dataset": "SWE-bench/SWE-bench_Verified",
            "dataset_revision": dataset_revision,
            "harness_commit": harness_commit,
            "submitted_patches_graded": submitted_count,
            "grader_errors": sum(
                result["grader_error_count"] for result in model_results.values()
            ),
            "unapplyable_model_patches": sum(
                result["unapplyable_patch_count"]
                for result in model_results.values()
            ),
        },
        "overall": {
            "episode_count": len(episodes),
            "submitted_count": submitted_count,
            "resolved_count": resolved_count,
            "unresolved_count": len(episodes) - resolved_count,
            "unresolved_submission_count": submitted_count - resolved_count,
            "unresolved_without_submission_count": len(episodes) - submitted_count,
            "submission_pass_rate": _ratio(resolved_count, submitted_count),
            "overall_resolution_rate": _ratio(resolved_count, len(episodes)),
            "total_conservative_cost_usd": str(total_cost),
            "cost_per_resolved_task_usd": (
                str(total_cost / resolved_count) if resolved_count else None
            ),
            "total_episode_latency_seconds": total_latency,
            "latency_per_resolved_task_seconds": (
                total_latency / resolved_count if resolved_count else None
            ),
            "unique_task_count": len({episode["task_id"] for episode in episodes}),
            "unique_resolved_task_count": len(all_resolved_tasks),
            "best_fixed_model": best_model,
            "best_fixed_model_resolved_count": best_result["resolved_count"],
            "oracle_incremental_unique_resolutions_over_best_fixed": (
                len(all_resolved_tasks) - best_result["resolved_count"]
            ),
        },
        "models": model_results,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate official SWE-bench reports with saved pilot costs."
    )
    parser.add_argument("episodes", type=Path)
    parser.add_argument("grader_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--study-stage")
    parser.add_argument(
        "--model-caused-error-id",
        action="append",
        default=[],
        help=(
            "Official error ID confirmed from the saved run log to be an "
            "unapplyable model patch; counted as an unresolved model outcome."
        ),
    )
    parser.add_argument(
        "--model-caused-error",
        action="append",
        default=[],
        metavar="MODEL=TASK_ID",
        help=(
            "Model-qualified official error confirmed from the saved run log "
            "to be an unapplyable model patch."
        ),
    )
    args = parser.parse_args()
    model_caused_error_keys: set[tuple[str, str]] = set()
    for value in args.model_caused_error:
        try:
            model, task_id = value.rsplit("=", 1)
        except ValueError:
            parser.error("--model-caused-error must be MODEL=TASK_ID")
        if not model or not task_id:
            parser.error("--model-caused-error must be MODEL=TASK_ID")
        model_caused_error_keys.add((model, task_id))

    summary = aggregate_grades(
        args.episodes,
        args.grader_dir,
        dataset_revision=args.dataset_revision,
        harness_commit=args.harness_commit,
        study_stage=args.study_stage,
        model_caused_error_ids=set(args.model_caused_error_id),
        model_caused_error_keys=model_caused_error_keys,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
