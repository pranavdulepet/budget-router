#!/usr/bin/env python3
"""Resume-safe fixed-model matrix runner for Amendment 009."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.pricing import PriceSnapshot
from budget_router.serialization import read_jsonl, stable_hash

try:
    from scripts.run_active_router_study import validate_protocol_lock
    from scripts.run_tinker_pilot import (
        _adapter,
        _image_name,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )
except ModuleNotFoundError:
    from run_active_router_study import validate_protocol_lock  # type: ignore[no-redef]
    from run_tinker_pilot import (  # type: ignore[no-redef]
        _adapter,
        _image_name,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )


STAGE_BUDGET_KEY = {
    "development": "development_fixed_matrix",
    "heldout": "heldout_fixed_matrix",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_task_manifest(payload: dict[str, Any]) -> None:
    claimed = str(payload.get("manifest_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("manifest_hash", None)
    if stable_hash(unhashed) != claimed:
        raise ValueError("quality-cost task manifest hash mismatch")


def validate_execution_lock(project_root: Path, path: Path) -> None:
    payload = _load(path)
    claimed = str(payload.get("execution_lock_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("execution_lock_hash", None)
    if stable_hash(unhashed) != claimed:
        raise ValueError("quality-cost execution lock hash mismatch")
    for relative, expected in payload["files"].items():
        actual = hashlib.sha256(
            (project_root / relative).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"execution-locked file hash mismatch: {relative}")
    expected_version = str(payload["external"]["mini_swe_agent_version"])
    if importlib.metadata.version("mini-swe-agent") != expected_version:
        raise ValueError("mini-swe-agent version mismatch")
    from minisweagent import package_dir

    config = package_dir / "config" / "benchmarks" / "swebench.yaml"
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    if config_hash != payload["external"]["mini_swe_agent_swebench_yaml_sha256"]:
        raise ValueError("mini-swe-agent SWE-bench configuration hash mismatch")


def _stage_limit(protocol: dict[str, Any], stage: str) -> Decimal:
    limits = {
        str(row["stage"]): Decimal(str(row["maximum_usd"]))
        for row in protocol["ordered_budget_stages"]
    }
    return limits[STAGE_BUDGET_KEY[stage]]


def _treatments(pool: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(pool["treatments"]):
        role = str(row["role"])
        if role in result:
            raise ValueError(f"duplicate model role: {role}")
        result[role] = (index, dict(row))
    if set(result) != {"cheap", "medium", "strong"}:
        raise ValueError("quality-cost pool must contain cheap, medium, and strong")
    return result


def _role_order(task_id: str, roles: list[str], seed: int) -> list[str]:
    result = list(roles)
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def plan_blocks(
    *,
    task_ids: list[str],
    roles: list[str],
    records: list[dict[str, Any]],
    stage: str,
    seed: int,
    max_task_blocks: int | None,
) -> list[dict[str, Any]]:
    completed: set[tuple[str, str, str]] = set()
    for row in records:
        key = (
            str(row["study_stage"]),
            str(row["task_id"]),
            str(row["model_role"]),
        )
        if key in completed:
            raise ValueError(f"duplicate fixed-matrix episode: {key}")
        completed.add(key)
    blocks: list[dict[str, Any]] = []
    for task_id in task_ids:
        pending = [
            role
            for role in _role_order(task_id, roles, seed)
            if (stage, task_id, role) not in completed
        ]
        if pending:
            blocks.append({"task_id": task_id, "roles": pending})
    if max_task_blocks is not None:
        blocks = blocks[:max_task_blocks]
    return blocks


def _ensure_image(task_id: str) -> str:
    image = _image_name(task_id)
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if inspected.returncode:
        subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            check=True,
            timeout=1_800,
        )
    return image


def _remove_task_image(image: str) -> None:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if not inspected.returncode:
        subprocess.run(["docker", "image", "rm", image], check=True)


def _exposure(records: list[dict[str, Any]], stage: str) -> Decimal:
    return sum(
        (
            Decimal(str(row["conservative_cost_usd"]))
            for row in records
            if row.get("study_stage") == stage
        ),
        Decimal("0"),
    )


def validate_completed_task_block(
    records: list[dict[str, Any]],
    *,
    stage: str,
    task_id: str,
    roles: list[str],
) -> None:
    rows = [
        row
        for row in records
        if row.get("study_stage") == stage
        and row.get("task_id") == task_id
        and row.get("model_role") in roles
    ]
    observed = {str(row["model_role"]) for row in rows}
    if len(rows) != len(observed) or observed != set(roles):
        raise RuntimeError(f"task block is incomplete or duplicated: {task_id}")
    if any(
        not row.get("terminal_record_valid") or row.get("provider_failed")
        for row in rows
    ):
        raise RuntimeError(f"task block is not terminal and valid: {task_id}")
    for row in rows:
        if not Path(str(row["trajectory"])).exists():
            raise RuntimeError(f"task block trajectory is absent: {task_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "heldout"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_009_three_tier_quality_cost.json"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/active_router_protocol.sha256"),
    )
    parser.add_argument(
        "--execution-lock",
        type=Path,
        default=Path("artifacts/quality_cost_v1_fixed_execution_lock.json"),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/quality_cost_v1_task_manifest.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_quality_cost_v1.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-30_quality_cost_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/quality_cost_v1/fixed_matrix"),
    )
    parser.add_argument("--role", action="append", dest="selected_roles")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-task-blocks", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--prune-completed-task-images", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.workers <= 3:
        raise SystemExit("--workers must be in [1, 3]")
    if args.max_task_blocks is not None and args.max_task_blocks <= 0:
        raise SystemExit("--max-task-blocks must be positive")
    validate_protocol_lock(Path.cwd(), args.protocol_lock)
    validate_execution_lock(Path.cwd(), args.execution_lock)
    protocol = _load(args.protocol)
    manifest = _load(args.task_manifest)
    validate_task_manifest(manifest)
    task_payload = _load(args.tasks)
    tasks_by_id = {
        str(row["instance_id"]): dict(row) for row in task_payload["records"]
    }
    task_ids = [str(row["task_id"]) for row in manifest[args.stage]]
    missing = sorted(set(task_ids) - set(tasks_by_id))
    if missing:
        raise ValueError(f"task IDs are absent from the dataset: {missing}")

    pool = _load(args.model_pool)
    treatment_map = _treatments(pool)
    roles = (
        [str(role) for role in args.selected_roles]
        if args.selected_roles
        else ["cheap", "medium", "strong"]
    )
    if len(roles) != len(set(roles)) or not set(roles) <= set(treatment_map):
        raise ValueError("selected roles must be unique members of the frozen pool")
    if args.prune_completed_task_images and set(roles) != set(treatment_map):
        raise ValueError(
            "image pruning requires the complete cheap/medium/strong task block"
        )

    records_path = args.output_root / "episodes.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    if any(
        row.get("study_stage") == args.stage and row.get("provider_failed")
        for row in records
    ):
        raise RuntimeError("saved provider failure stops fixed-matrix collection")
    blocks = plan_blocks(
        task_ids=task_ids,
        roles=roles,
        records=records,
        stage=args.stage,
        seed=args.seed,
        max_task_blocks=args.max_task_blocks,
    )
    remaining_reservation = sum(
        (
            Decimal(str(treatment_map[role][1]["episode_hard_cap_usd"]))
            for block in blocks
            for role in block["roles"]
        ),
        Decimal("0"),
    )
    stage_spent = _exposure(records, args.stage)
    stage_limit = _stage_limit(protocol, args.stage)
    if stage_spent + remaining_reservation > stage_limit:
        raise PermissionError("fixed-matrix plan exceeds the frozen stage cap")

    plan = {
        "stage": args.stage,
        "tasks_in_scope": len(task_ids),
        "roles": roles,
        "completed_episodes": sum(
            row.get("study_stage") == args.stage for row in records
        ),
        "pending_task_blocks": len(blocks),
        "pending_episodes": sum(len(block["roles"]) for block in blocks),
        "stage_spent_usd": str(stage_spent),
        "maximum_remaining_reservation_usd": str(remaining_reservation),
        "stage_limit_usd": str(stage_limit),
        "execute": args.execute,
        "prune_completed_task_images": args.prune_completed_task_images,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute or not blocks:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise PermissionError("TINKER_API_KEY is required only for --execute")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    prices = PriceSnapshot.load(args.prices)
    adapters = {
        role: asyncio.run(_adapter(treatment_map[role][1])) for role in roles
    }
    ordinal = {task_id: index for index, task_id in enumerate(task_ids)}
    for block in blocks:
        task_id = str(block["task_id"])
        image = _ensure_image(task_id)
        pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(block["roles"]))
        ) as executor:
            for role in block["roles"]:
                treatment_index, treatment = treatment_map[role]
                trajectory_path = (
                    args.output_root
                    / "trajectories"
                    / args.stage
                    / role
                    / f"{task_id}.json"
                )
                pending[
                    executor.submit(
                        _run_episode,
                        tasks_by_id[task_id],
                        treatment,
                        adapters[role],
                        prices,
                        seed=args.seed + treatment_index * 100 + ordinal[task_id],
                        hard_limit_usd=Decimal(
                            str(treatment["episode_hard_cap_usd"])
                        ),
                        max_output_tokens=args.max_output_tokens,
                        request_timeout_seconds=args.request_timeout_seconds,
                        trajectory_path=trajectory_path,
                    )
                ] = role
            block_records: list[dict[str, Any]] = []
            for future in concurrent.futures.as_completed(pending):
                role = pending[future]
                record = future.result()
                record["study_id"] = protocol["parent_study_id"]
                record["amendment_id"] = protocol["amendment_id"]
                record["study_stage"] = args.stage
                record["model_role"] = role
                record["task_manifest_hash"] = manifest["manifest_hash"]
                record["protocol_hash"] = hashlib.sha256(
                    args.protocol.read_bytes()
                ).hexdigest()
                shard = (
                    args.output_root
                    / "episode_shards"
                    / args.stage
                    / task_id
                    / f"{role}.json"
                )
                _write_json(shard, record)
                _write_jsonl_record(records_path, record)
                records.append(record)
                block_records.append(record)
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "role": role,
                            "structurally_valid": record["structurally_valid"],
                            "submitted": record["submitted_patch"],
                            "provider_failed": record["provider_failed"],
                            "cost_usd": record["conservative_cost_usd"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if any(row["provider_failed"] for row in block_records):
            raise RuntimeError("provider failure stops fixed-matrix collection")
        if _exposure(records, args.stage) > stage_limit:
            raise RuntimeError("fixed-matrix stage exposure exceeded its hard cap")
        if args.prune_completed_task_images:
            validate_completed_task_block(
                records,
                stage=args.stage,
                task_id=task_id,
                roles=roles,
            )
            _remove_task_image(image)

    _write_json(
        args.output_root / f"{args.stage}_collection_summary.json",
        {
            **plan,
            "terminal": True,
            "final_stage_spent_usd": str(_exposure(records, args.stage)),
            "final_episode_count": sum(
                row.get("study_stage") == args.stage for row in records
            ),
        },
    )


if __name__ == "__main__":
    main()
