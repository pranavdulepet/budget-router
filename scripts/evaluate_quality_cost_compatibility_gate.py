#!/usr/bin/env python3
"""Evaluate the pre-registered Amendment 009 strong-model gate."""

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
    protocol_path: Path,
    smoke_path: Path,
    episodes_path: Path,
    grading_path: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    gate = protocol["strong_compatibility_gate"]
    smoke = _load(smoke_path)
    episodes = read_jsonl(episodes_path)
    grading = _load(grading_path)
    expected_tasks = list(map(str, gate["tasks"]))
    expected_set = set(expected_tasks)
    if len(episodes) != len(expected_tasks):
        raise ValueError("compatibility episode count is not terminal")
    by_task: dict[str, dict[str, Any]] = {}
    for row in episodes:
        task_id = str(row["task_id"])
        if task_id in by_task:
            raise ValueError(f"duplicate compatibility episode: {task_id}")
        by_task[task_id] = row
    if set(by_task) != expected_set:
        raise ValueError("compatibility task IDs do not match the frozen gate")
    models = {str(row["model"]) for row in episodes}
    if len(models) != 1:
        raise ValueError("compatibility gate must contain exactly one model")
    model = next(iter(models))
    official = grading["models"].get(model)
    if official is None:
        raise ValueError("official grading does not contain the gated model")

    smoke_valid = bool(smoke.get("all_structurally_valid"))
    structurally_valid_count = sum(
        bool(row.get("structurally_valid")) for row in episodes
    )
    submitted_count = sum(bool(row.get("submitted_patch")) for row in episodes)
    provider_failure_count = sum(
        bool(row.get("provider_failed")) for row in episodes
    )
    unclassified_errors = list(grading.get("unclassified_errors", []))
    resolved_ids = list(map(str, official.get("resolved_ids", [])))
    minimum_structural = int(gate["minimum_structurally_valid"])
    minimum_resolved = int(gate["minimum_official_resolutions"])
    checks = {
        "synthetic_smoke_structurally_valid": smoke_valid,
        "minimum_agent_structural_validity": (
            structurally_valid_count >= minimum_structural
        ),
        "minimum_official_resolutions": len(resolved_ids) >= minimum_resolved,
        "no_provider_failures": provider_failure_count == 0,
        "no_unclassified_infrastructure_errors": not unclassified_errors,
    }
    passed = all(checks.values())
    return {
        "schema_version": "quality-cost-compatibility-gate-v1",
        "amendment_id": protocol["amendment_id"],
        "model": model,
        "expected_task_count": len(expected_tasks),
        "smoke_structurally_valid": smoke_valid,
        "structurally_valid_count": structurally_valid_count,
        "submitted_count": submitted_count,
        "resolved_count": len(resolved_ids),
        "resolved_ids": sorted(resolved_ids),
        "provider_failure_count": provider_failure_count,
        "unclassified_errors": unclassified_errors,
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
            "proceed_with_amendment_009_paid_matrix"
            if passed
            else "stop_amendment_009_paid_work"
        ),
    }


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
    parser.add_argument(
        "--smoke",
        type=Path,
        default=Path(
            "outputs/quality_cost_v1/qwen397_tool_smoke_network.json"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path(
            "outputs/quality_cost_v1/compatibility/episodes.jsonl"
        ),
    )
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_gate(
        protocol_path=args.protocol,
        smoke_path=args.smoke,
        episodes_path=args.episodes,
        grading_path=args.grading,
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
