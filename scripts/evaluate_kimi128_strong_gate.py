#!/usr/bin/env python3
"""Evaluate the frozen Amendment 011 Kimi 128K qualification gate."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import read_jsonl


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gate(
    *,
    amendment: dict[str, Any],
    task_manifest: dict[str, Any],
    smoke: dict[str, Any],
    episodes: list[dict[str, Any]],
    grading: dict[str, Any],
) -> dict[str, Any]:
    expected_tasks = [str(row["task_id"]) for row in task_manifest["tasks"]]
    if len(episodes) != len(expected_tasks):
        raise ValueError("Kimi qualification episode count is not terminal")
    by_task: dict[str, dict[str, Any]] = {}
    for row in episodes:
        task_id = str(row["task_id"])
        if task_id in by_task:
            raise ValueError(f"duplicate Kimi qualification episode: {task_id}")
        by_task[task_id] = row
    if set(by_task) != set(expected_tasks):
        raise ValueError("Kimi qualification task IDs do not match the manifest")
    candidate = amendment["candidate"]
    model = str(candidate["model"])
    if {str(row["model"]) for row in episodes} != {model}:
        raise ValueError("Kimi qualification contains an unexpected model")
    official = grading["models"].get(model)
    if official is None:
        raise ValueError("official grading does not contain Kimi 128K")

    gate = amendment["gate"]
    resolved_ids = set(map(str, official["resolved_ids"]))
    medium_ids = set(
        map(
            str,
            amendment["fixed_medium_reference"]["resolved_ids"],
        )
    )
    unique_ids = resolved_ids - medium_ids
    structurally_valid = sum(
        bool(row.get("structurally_valid")) for row in episodes
    )
    submitted = sum(bool(row.get("submitted_patch")) for row in episodes)
    provider_failures = sum(
        bool(row.get("provider_failed")) for row in episodes
    )
    unclassified = list(grading.get("unclassified_errors", []))
    checks = {
        "synthetic_smoke_structurally_valid": bool(
            smoke.get("all_structurally_valid")
        ),
        "minimum_structurally_valid": (
            structurally_valid >= int(gate["minimum_structurally_valid"])
        ),
        "minimum_official_resolutions": (
            len(resolved_ids) >= int(gate["minimum_official_resolutions"])
        ),
        "minimum_unique_beyond_fixed_medium": (
            len(unique_ids)
            >= int(gate["minimum_unique_resolutions_beyond_fixed_medium"])
        ),
        "no_provider_failures": provider_failures == 0,
        "no_unclassified_infrastructure_errors": not unclassified,
    }
    passed = all(checks.values())
    return {
        "schema_version": "kimi128-strong-qualification-result-v1",
        "amendment_id": amendment["amendment_id"],
        "model": model,
        "expected_task_count": len(expected_tasks),
        "structurally_valid_count": structurally_valid,
        "submitted_count": submitted,
        "resolved_count": len(resolved_ids),
        "resolved_ids": sorted(resolved_ids),
        "fixed_medium_resolved_count": len(medium_ids),
        "unique_beyond_fixed_medium_count": len(unique_ids),
        "unique_beyond_fixed_medium_ids": sorted(unique_ids),
        "provider_failure_count": provider_failures,
        "unclassified_errors": unclassified,
        "episode_cost_usd": str(
            sum(
                (
                    Decimal(str(row["conservative_cost_usd"]))
                    for row in episodes
                ),
                Decimal("0"),
            )
        ),
        "smoke_cost_usd": str(smoke["observed_conservative_cost_usd"]),
        "checks": checks,
        "passed": passed,
        "decision": (
            "admit_kimi128_and_freeze_full_study_continuation"
            if passed
            else "stop_further_tinker_strong_candidate_search"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_011_kimi128_strong_qualification.json"
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/pilot_tasks.json"),
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        default=Path("outputs/quality_cost_v2/kimi128_tool_smoke.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/quality_cost_v2/kimi128_gate/episodes.jsonl"),
    )
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_gate(
        amendment=_load(args.amendment),
        task_manifest=_load(args.task_manifest),
        smoke=_load(args.smoke),
        episodes=read_jsonl(args.episodes),
        grading=_load(args.grading),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
