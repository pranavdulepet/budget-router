from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_swebench_pilot_grades import aggregate_grades


def test_aggregate_grades_includes_non_submissions_in_resolution_rate(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "model": "org/model",
                    "task_id": "task-1",
                    "study_stage": "test",
                    "submitted_patch": True,
                    "conservative_cost_usd": "0.4",
                    "latency_seconds": 10,
                },
                {
                    "model": "org/model",
                    "task_id": "task-2",
                    "study_stage": "train",
                    "submitted_patch": False,
                    "conservative_cost_usd": "0.6",
                    "latency_seconds": 20,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    grader_dir = tmp_path / "grader"
    grader_dir.mkdir()
    (grader_dir / "org__model.run.json").write_text(
        json.dumps(
            {
                "submitted_ids": ["task-1"],
                "completed_ids": ["task-1"],
                "resolved_ids": ["task-1"],
                "error_ids": [],
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_grades(
        episodes,
        grader_dir,
        dataset_revision="dataset-sha",
        harness_commit="harness-sha",
    )

    assert result["overall"]["resolved_count"] == 1
    assert result["overall"]["overall_resolution_rate"] == 0.5
    assert result["overall"]["submission_pass_rate"] == 1.0
    assert result["models"]["org/model"]["cost_per_resolved_task_usd"] == "1.0"
    assert result["models"]["org/model"]["latency_per_resolved_task_seconds"] == 30

    test_result = aggregate_grades(
        episodes,
        grader_dir,
        dataset_revision="dataset-sha",
        harness_commit="harness-sha",
        study_stage="test",
    )
    assert test_result["study_stage"] == "test"
    assert test_result["overall"]["episode_count"] == 1
    assert test_result["overall"]["overall_resolution_rate"] == 1.0


def test_aggregate_grades_counts_terminal_patch_error_as_unresolved(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "model": "org/model",
                "task_id": "task-1",
                "submitted_patch": True,
                "conservative_cost_usd": "0.4",
                "latency_seconds": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grader_dir = tmp_path / "grader"
    grader_dir.mkdir()
    (grader_dir / "org__model.run.json").write_text(
        json.dumps(
            {
                "submitted_ids": ["task-1"],
                "completed_ids": [],
                "resolved_ids": [],
                "error_ids": ["task-1"],
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_grades(
        episodes,
        grader_dir,
        dataset_revision="dataset-sha",
        harness_commit="harness-sha",
    )

    assert result["overall"]["resolved_count"] == 0
    assert result["grader"]["grader_errors"] == 1
    assert result["models"]["org/model"]["graded_count"] == 1
    assert result["task_results"][0]["status"] == "grader_error"


def test_aggregate_grades_classifies_confirmed_model_patch_error(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "model": "org/model",
                "task_id": "task-1",
                "submitted_patch": True,
                "conservative_cost_usd": "0.4",
                "latency_seconds": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grader_dir = tmp_path / "grader"
    grader_dir.mkdir()
    (grader_dir / "org__model.run.json").write_text(
        json.dumps(
            {
                "submitted_ids": ["task-1"],
                "completed_ids": [],
                "resolved_ids": [],
                "error_ids": ["task-1"],
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_grades(
        episodes,
        grader_dir,
        dataset_revision="dataset-sha",
        harness_commit="harness-sha",
        model_caused_error_ids={"task-1"},
    )

    assert result["grader"]["grader_errors"] == 0
    assert result["grader"]["unapplyable_model_patches"] == 1
    assert result["models"]["org/model"]["unapplyable_patch_count"] == 1
    assert result["task_results"][0]["status"] == "unapplyable_patch"


def test_model_qualified_patch_error_does_not_classify_other_policy(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps(
                {
                    "model": model,
                    "task_id": "task-1",
                    "submitted_patch": True,
                    "conservative_cost_usd": "0.4",
                    "latency_seconds": 10,
                }
            )
            for model in ("org/model-a", "org/model-b")
        )
        + "\n",
        encoding="utf-8",
    )
    grader_dir = tmp_path / "grader"
    grader_dir.mkdir()
    for model in ("a", "b"):
        (grader_dir / f"org__model-{model}.run.json").write_text(
            json.dumps(
                {
                    "submitted_ids": ["task-1"],
                    "completed_ids": [] if model == "a" else ["task-1"],
                    "resolved_ids": [],
                    "error_ids": ["task-1"] if model == "a" else [],
                }
            ),
            encoding="utf-8",
        )

    result = aggregate_grades(
        episodes,
        grader_dir,
        dataset_revision="dataset-sha",
        harness_commit="harness-sha",
        model_caused_error_keys={("org/model-a", "task-1")},
    )

    assert result["models"]["org/model-a"]["unapplyable_patch_count"] == 1
    assert result["models"]["org/model-b"]["unapplyable_patch_count"] == 0
