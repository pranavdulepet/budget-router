from __future__ import annotations

import pytest

from scripts.run_isolated_stage_official_grading import (
    _report_filename,
    _run_id,
    validate_report,
)


def test_policy_run_ids_and_report_filename() -> None:
    fixed = "fixed:Qwen/Qwen3.6-35B-A3B"
    candidate = "isolated:openai/gpt-oss-20b->Qwen/Qwen3.6-35B-A3B"
    assert _run_id(fixed) == "isolated-stage-test-fixed-v1"
    assert _run_id(candidate) == "isolated-stage-test-candidate-v1"
    assert _report_filename(fixed, "run") == (
        "fixed:Qwen__Qwen3.6-35B-A3B.run.json"
    )


def test_validate_report_requires_terminal_partition() -> None:
    predictions = [
        {"instance_id": "a"},
        {"instance_id": "b"},
        {"instance_id": "c"},
    ]
    report = {
        "submitted_ids": ["a", "b", "c"],
        "completed_ids": ["a", "b"],
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "error_ids": ["c"],
    }
    assert validate_report(report, predictions) == ["c"]


def test_validate_report_rejects_incomplete_submission() -> None:
    predictions = [{"instance_id": "a"}, {"instance_id": "b"}]
    report = {
        "submitted_ids": ["a", "b"],
        "completed_ids": ["a"],
        "resolved_ids": ["a"],
        "unresolved_ids": [],
        "error_ids": [],
    }
    with pytest.raises(ValueError, match="not terminal"):
        validate_report(report, predictions)
