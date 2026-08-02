from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import read_jsonl, stable_json

try:
    from scripts.run_tinker_pilot import _write_jsonl_record
except ModuleNotFoundError:
    from run_tinker_pilot import _write_jsonl_record  # type: ignore[no-redef]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_abandonment_record(
    protocol: dict[str, Any],
    candidate_freeze: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    task_id: str,
    stage: str,
    reason: str,
    latency_lower_bound_seconds: float,
) -> dict[str, Any]:
    if candidate_freeze.get("study_manifest_hash") != protocol["manifest_hash"]:
        raise ValueError("candidate freeze belongs to a different study")
    if task_id not in protocol["stages"][stage]["task_ids"]:
        raise ValueError("task is outside the frozen stage")
    policy_id = str(candidate_freeze["selected_policy_id"])
    if any(
        row.get("task_id") == task_id and row.get("model") == policy_id
        for row in records
    ):
        raise ValueError("refusing to duplicate an existing episode record")
    task = next(row for row in protocol["tasks"] if row["task_id"] == task_id)
    cap = Decimal(str(protocol["execution_policy"]["total_hard_cap_usd"]))
    record = {
        "schema_version": "isolated-stage-abandoned-episode-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "study_stage": stage,
        "task_id": task_id,
        "task_ordinal": int(task["dataset_ordinal"]),
        "repository": task["repository"],
        "model": policy_id,
        "policy": protocol["execution_policy"]["kind"],
        "scout_model": "openai/gpt-oss-20b",
        "finisher_model": protocol["fixed_policy"]["model"],
        "hard_limit_usd": str(cap),
        "price_snapshot": protocol["price_snapshot"],
        "emitted_tool_call": False,
        "arguments_valid": False,
        "tool_result_returned": False,
        "continued_after_tool": False,
        "terminal_record_valid": False,
        "scout_structurally_valid": True,
        "finisher_structurally_valid": False,
        "structurally_valid": False,
        "provider_failed": True,
        "error_type": "ExternalProviderHang",
        "exit_status": "AbandonedProviderHang",
        "submitted_patch": False,
        "submission_sha256": None,
        "conservative_cost_usd": str(cap),
        "provider_failure_reserved_usd": str(cap),
        "input_tokens": 0,
        "output_tokens": 0,
        "model_calls": 0,
        "scout_calls": 6,
        "finisher_calls": None,
        "scout_cost_usd": None,
        "finisher_cost_usd": None,
        "scout_workspace_discarded": True,
        "finisher_clean_workspace": True,
        "latency_seconds": latency_lower_bound_seconds,
        "latency_is_lower_bound": True,
        "trajectory": None,
        "abandonment_reason": reason,
        "accounting_note": (
            "The full episode cap is reserved because the provider call did "
            "not return final usage. This is conservative, not an observed charge."
        ),
    }
    if scan_for_secrets(record):
        raise ValueError("abandonment record contains a possible secret")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--candidate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/candidate_freeze.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--hang-log",
        type=Path,
        default=Path("outputs/isolated_stage_v1/provider_hangs.jsonl"),
    )
    parser.add_argument("--stage", choices=("expand", "test"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--latency-lower-bound-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.latency_lower_bound_seconds <= 0:
        raise ValueError("latency lower bound must be positive")
    protocol = _load(args.protocol)
    record = build_abandonment_record(
        protocol,
        _load(args.candidate_freeze),
        read_jsonl(args.records),
        task_id=args.task_id,
        stage=args.stage,
        reason=args.reason,
        latency_lower_bound_seconds=args.latency_lower_bound_seconds,
    )
    _write_jsonl_record(args.records, record)
    _write_jsonl_record(args.hang_log, record)
    print(stable_json(record))


if __name__ == "__main__":
    main()
