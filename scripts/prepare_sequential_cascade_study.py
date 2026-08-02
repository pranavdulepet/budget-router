from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import read_jsonl, stable_hash, stable_json

try:
    from scripts.run_router_baseline_study import incremental_exposure
except ModuleNotFoundError:
    from run_router_baseline_study import incremental_exposure  # type: ignore[no-redef]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_protocol(
    source_study: dict[str, Any],
    router_freeze: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    abandoned_hangs: list[dict[str, Any]],
    comparator_report: dict[str, Any],
    *,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    source_without_hash = dict(source_study)
    claimed_source_hash = str(source_without_hash.pop("manifest_hash", ""))
    if stable_hash(source_without_hash) != claimed_source_hash:
        raise ValueError("source study manifest hash is invalid")
    if router_freeze.get("study_manifest_hash") != claimed_source_hash:
        raise ValueError("router freeze does not match the source study")

    task_ids = [
        str(task_id) for task_id in source_study["stages"]["test"]["task_ids"]
    ]
    if len(task_ids) != 20 or len(set(task_ids)) != 20:
        raise ValueError("cascade arm requires the frozen 20-task test cohort")
    if comparator_report.get("error_ids"):
        raise ValueError("fixed-Qwen comparator report contains grader errors")
    if set(comparator_report.get("submitted_ids", ())) != (
        set(comparator_report.get("completed_ids", ()))
        | set(comparator_report.get("error_ids", ()))
    ):
        raise ValueError("fixed-Qwen comparator grading is incomplete")

    recorded_exposure = incremental_exposure(baseline_records)
    abandoned_reserve = sum(
        (
            Decimal(str(row["reserved_cost_usd"]))
            for row in abandoned_hangs
        ),
        Decimal("0"),
    )
    total_cap = Decimal("0.90")
    maximum_new = total_cap * len(task_ids)
    projected = recorded_exposure + abandoned_reserve + maximum_new
    working_limit = Decimal(
        str(source_study["incremental_working_limit_usd"])
    )
    absolute_limit = Decimal(
        str(source_study["incremental_absolute_limit_usd"])
    )
    if projected > working_limit or projected > absolute_limit:
        raise ValueError("full cascade reservation does not fit frozen limits")

    protocol = {
        "schema_version": "sequential-cascade-study-v1",
        "study_id": "cheap-scout-qwen-cascade-2026-07-28-v1",
        "source_static_study_id": source_study["study_id"],
        "source_static_study_manifest_hash": claimed_source_hash,
        "source_router_freeze_hash": router_freeze["freeze_hash"],
        "design_timing": (
            "The optional cascade arm and $18 reservation were preregistered "
            "before static test collection. Scout phase details were frozen "
            "after static results but before any cascade outcome; treat this "
            "as a secondary exploratory paired experiment."
        ),
        "task_ids": task_ids,
        "comparator": {
            "policy": "fixed:Qwen/Qwen3.6-35B-A3B",
            "source": "existing frozen-test episode and official grade",
            "task_count": len(task_ids),
            "submitted_count": len(comparator_report["submitted_ids"]),
            "resolved_count": len(comparator_report["resolved_ids"]),
            "report_sha256": _sha256(input_paths["comparator_report"]),
        },
        "cascade_policy": {
            "policy_id": (
                "cascade:Qwen/Qwen3-8B->Qwen/Qwen3.6-35B-A3B"
            ),
            "kind": "cheap-first-visible-handoff-v1",
            "scout_model": "Qwen/Qwen3-8B",
            "scout_max_calls": 6,
            "scout_hard_cap_usd": "0.06",
            "scout_max_output_tokens_per_call": 4000,
            "early_scout_submission_allowed": True,
            "finisher_model": "Qwen/Qwen3.6-35B-A3B",
            "finisher_max_output_tokens_per_call": 8000,
            "handoff": "full visible transcript plus same mutable workspace",
            "hidden_reasoning_transferred": False,
            "total_step_limit": 75,
            "total_hard_cap_usd": "0.90",
            "finisher_budget": "total cap minus observed scout spend",
            "provider_failure_behavior": (
                "charge the scout reservation and hand off to the finisher"
            ),
            "seed_policy": (
                "reuse the frozen full-pool per-treatment seed formula for "
                "each model phase on the same task ordinal"
            ),
        },
        "analysis": {
            "primary_metric": "paired official resolution-rate delta",
            "secondary_metrics": [
                "total conservative cost delta",
                "cost per resolved task",
                "submission rate",
                "latency",
                "scout early-finish count",
                "handoff count",
                "scout and finisher cost shares",
            ],
            "paired_test": "two-sided exact McNemar",
            "uncertainty": "task bootstrap with fixed seed 20260728",
            "success_rule": (
                "cascade resolves more tasks, or matches fixed-Qwen resolution "
                "with lower total conservative cost"
            ),
            "all_no_patch_episodes_count_unresolved": True,
        },
        "budget": {
            "recorded_prior_exposure_usd": str(recorded_exposure),
            "abandoned_provider_hang_reserve_usd": str(abandoned_reserve),
            "maximum_new_cascade_cost_usd": str(maximum_new),
            "projected_maximum_exposure_usd": str(projected),
            "working_limit_usd": str(working_limit),
            "absolute_limit_usd": str(absolute_limit),
        },
        "inputs": {
            name: _sha256(path) for name, path in sorted(input_paths.items())
        },
    }
    protocol["manifest_hash"] = stable_hash(protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--router-freeze",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_freeze.json"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("outputs/router_baseline_v2/episodes.jsonl"),
    )
    parser.add_argument(
        "--provider-hangs",
        type=Path,
        default=Path("outputs/router_baseline_v2/provider_hangs.jsonl"),
    )
    parser.add_argument(
        "--comparator-report",
        type=Path,
        default=Path(
            "outputs/router_baseline_v2/swebench_grader/test/"
            "Qwen__Qwen3.6-35B-A3B.router-v2-test-qwen36-pinned.json"
        ),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_coding_v7.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-27_coding_v3.json"),
    )
    parser.add_argument(
        "--harness-prompt",
        type=Path,
        default=Path("configs/harness_prompt.txt"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sequential_cascade_study_v1.json"),
    )
    args = parser.parse_args()
    input_paths = {
        "source_study": args.source_study,
        "router_freeze": args.router_freeze,
        "baseline_records": args.baseline_records,
        "provider_hangs": args.provider_hangs,
        "comparator_report": args.comparator_report,
        "model_pool": args.model_pool,
        "prices": args.prices,
        "harness_prompt": args.harness_prompt,
        "tasks": args.tasks,
    }
    protocol = build_protocol(
        _load(args.source_study),
        _load(args.router_freeze),
        read_jsonl(args.baseline_records),
        read_jsonl(args.provider_hangs),
        _load(args.comparator_report),
        input_paths=input_paths,
    )
    serialized = stable_json(protocol) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("refusing to overwrite a different cascade freeze")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
