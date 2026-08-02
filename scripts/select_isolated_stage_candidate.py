from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import read_jsonl, stable_hash, stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_rows(
    payloads: Iterable[dict[str, Any]],
    *,
    fixed_model: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["task_results"]:
            if row["model"] != fixed_model:
                continue
            task_id = str(row["task_id"])
            if task_id in rows:
                raise ValueError(f"duplicate fixed grade: {task_id}")
            rows[task_id] = dict(row)
    return rows


def select_candidate(
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_grades: dict[str, Any],
    source_grade_payloads: Iterable[dict[str, Any]],
    *,
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    screen_ids = [str(task_id) for task_id in protocol["stages"]["screen"]["task_ids"]]
    policy_ids = [
        str(policy["policy_id"]) for policy in protocol["candidate_policies"]
    ]
    expected = {(task_id, policy_id) for task_id in screen_ids for policy_id in policy_ids}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("study_stage") != "screen":
            continue
        if record.get("study_manifest_hash") != protocol["manifest_hash"]:
            raise ValueError("screen record belongs to a different study")
        key = (str(record["task_id"]), str(record["model"]))
        if key in by_key:
            raise ValueError(f"duplicate screen episode: {key}")
        by_key[key] = dict(record)
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise ValueError(f"screen episode matrix is incomplete: missing={missing}, extra={extra}")

    grade_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_grades["task_results"]:
        key = (str(row["task_id"]), str(row["model"]))
        if key in grade_rows:
            raise ValueError(f"duplicate screen grade: {key}")
        grade_rows[key] = dict(row)
    if set(grade_rows) != expected:
        missing = sorted(expected - set(grade_rows))
        extra = sorted(set(grade_rows) - expected)
        raise ValueError(f"screen grade matrix is incomplete: missing={missing}, extra={extra}")

    fixed_model = str(protocol["fixed_policy"]["model"])
    fixed = _fixed_rows(source_grade_payloads, fixed_model=fixed_model)
    if not set(screen_ids) <= set(fixed):
        raise ValueError("fixed-model screen outcomes are incomplete")
    if any(fixed[task_id]["status"] == "grader_error" for task_id in screen_ids):
        raise ValueError("fixed-model screen outcomes contain a grader error")

    candidates: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        policy_records = [by_key[(task_id, policy_id)] for task_id in screen_ids]
        policy_grades = [grade_rows[(task_id, policy_id)] for task_id in screen_ids]
        candidate_resolved = [
            row["status"] == "resolved" for row in policy_grades
        ]
        fixed_resolved = [
            fixed[task_id]["status"] == "resolved" for task_id in screen_ids
        ]
        total_cost = sum(
            (
                Decimal(str(record["conservative_cost_usd"]))
                for record in policy_records
            ),
            Decimal("0"),
        )
        fixed_only_losses = sum(
            fixed_ok and not candidate_ok
            for fixed_ok, candidate_ok in zip(
                fixed_resolved,
                candidate_resolved,
                strict=True,
            )
        )
        candidate_only_wins = sum(
            candidate_ok and not fixed_ok
            for fixed_ok, candidate_ok in zip(
                fixed_resolved,
                candidate_resolved,
                strict=True,
            )
        )
        structurally_valid_count = sum(
            bool(record["structurally_valid"]) for record in policy_records
        )
        provider_failure_count = sum(
            bool(record["provider_failed"]) for record in policy_records
        )
        grader_error_count = sum(
            row["status"] == "grader_error" for row in policy_grades
        )
        eligible = bool(
            structurally_valid_count == len(screen_ids)
            and provider_failure_count == 0
            and grader_error_count == 0
        )
        candidates.append(
            {
                "policy_id": policy_id,
                "eligible": eligible,
                "episode_count": len(screen_ids),
                "structurally_valid_count": structurally_valid_count,
                "provider_failure_count": provider_failure_count,
                "grader_error_count": grader_error_count,
                "submitted_count": sum(
                    bool(record["submitted_patch"]) for record in policy_records
                ),
                "resolved_count": sum(candidate_resolved),
                "fixed_resolved_count": sum(fixed_resolved),
                "fixed_only_losses": fixed_only_losses,
                "candidate_only_wins": candidate_only_wins,
                "total_conservative_cost_usd": str(total_cost),
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("no isolated scout policy passed the frozen screen gate")
    selected = min(
        eligible,
        key=lambda row: (
            row["fixed_only_losses"],
            -row["candidate_only_wins"],
            -row["resolved_count"],
            Decimal(row["total_conservative_cost_usd"]),
            row["policy_id"],
        ),
    )
    freeze = {
        "schema_version": "isolated-stage-candidate-freeze-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "selection_rule": protocol["selection"],
        "screen_task_ids": screen_ids,
        "candidate_metrics": candidates,
        "selected_policy_id": selected["policy_id"],
        "test_outcomes_accessed": False,
        "input_hashes": input_hashes or {},
    }
    freeze["freeze_hash"] = stable_hash(freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--candidate-grades",
        type=Path,
        default=Path("outputs/isolated_stage_v1/screen_grades.json"),
    )
    parser.add_argument(
        "--train-calibration-grades",
        type=Path,
        default=Path("outputs/router_baseline_v2/train_calibration_grades.json"),
    )
    parser.add_argument(
        "--pilot-grades",
        type=Path,
        default=Path("artifacts/pilot_coding_v6_swebench_grades.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isolated_stage_v1/candidate_freeze.json"),
    )
    args = parser.parse_args()
    protocol = _load(args.protocol)
    freeze = select_candidate(
        protocol,
        read_jsonl(args.records),
        _load(args.candidate_grades),
        [
            _load(args.train_calibration_grades),
            _load(args.pilot_grades),
        ],
        input_hashes={
            "protocol": _sha256(args.protocol),
            "records": _sha256(args.records),
            "candidate_grades": _sha256(args.candidate_grades),
            "train_calibration_grades": _sha256(args.train_calibration_grades),
            "pilot_grades": _sha256(args.pilot_grades),
        },
    )
    serialized = stable_json(freeze) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("refusing to overwrite a different candidate freeze")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
