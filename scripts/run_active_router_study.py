#!/usr/bin/env python3
"""Run resume-safe paid stages from the checksum-locked active router protocol."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.pricing import PriceSnapshot
from budget_router.serialization import read_jsonl

try:
    from scripts.run_tinker_pilot import (
        _adapter,
        _image_name,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )
except ModuleNotFoundError:
    from run_tinker_pilot import (  # type: ignore[no-redef]
        _adapter,
        _image_name,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )


SCREEN_STAGES = ("cheap_medium_screen", "higher_cost_screen")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol_lock(root: Path, lock_path: Path) -> None:
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        target = root / relative
        if _sha256(target) != expected:
            raise ValueError(f"active protocol checksum mismatch: {relative}")


def incremental_exposure(records: list[dict[str, Any]]) -> Decimal:
    return sum(
        (Decimal(str(row["conservative_cost_usd"])) for row in records),
        Decimal("0"),
    )


def _stage_limit(protocol: dict[str, Any], stage: str) -> Decimal:
    limits = {
        str(row["stage"]): Decimal(str(row["maximum_usd"]))
        for row in protocol["ordered_budget_stages"]
    }
    # Amendments 009--010 replaced the active paid-stage table after the
    # original screens were completed. Preserve their frozen historical caps
    # so archived screen plans and exports remain reproducible.
    completed_screen_limits = {
        "cheap_medium_screen": Decimal("17.40"),
        "higher_cost_screen": Decimal("36.00"),
    }
    if stage in completed_screen_limits:
        return completed_screen_limits[stage]
    if stage not in limits:
        raise ValueError(f"stage is not budgeted by the active protocol: {stage}")
    return limits[stage]


def _stage_scope(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    stage: str,
) -> tuple[list[str], list[str]]:
    if stage == "cheap_medium_screen":
        group = "cheap_medium"
        task_key = "cheap_medium_task_ids"
    elif stage == "higher_cost_screen":
        group = "higher_cost"
        task_key = "higher_cost_task_ids"
    else:
        raise ValueError(
            "this collection runner currently accepts the two predeclared "
            "screen stages only"
        )
    models = [
        str(row["model"])
        for row in protocol["candidate_groups"][group]["models"]
    ]
    task_ids = [str(value) for value in manifest["screens"][task_key]]
    if len(models) != len(set(models)) or len(task_ids) != len(set(task_ids)):
        raise ValueError("active stage contains duplicate models or task IDs")
    return models, task_ids


def _invalidated_episode_keys(
    root: Path,
    protocol: dict[str, Any],
) -> set[tuple[str, str, str, str]]:
    result: set[tuple[str, str, str, str]] = set()
    for relative in protocol.get("active_amendments", ()):
        amendment = _load_json(root / str(relative))
        for row in amendment.get("invalidated_episode_keys", ()):
            result.add(
                (
                    str(row["study_stage"]),
                    str(row["task_id"]),
                    str(row["model"]),
                    str(row["protocol_sha256"]),
                )
            )
    return result


def _excluded_models(
    path: Path,
    *,
    study_id: str,
    protocol_sha256: str,
    stage: str,
) -> set[str]:
    if not path.exists():
        return set()
    payload = _load_json(path)
    compatible_hashes = {
        str(value)
        for value in payload.get("compatible_active_protocol_sha256", ())
    }
    if (
        payload.get("study_id") != study_id
        or (
            payload.get("protocol_sha256") != protocol_sha256
            and protocol_sha256 not in compatible_hashes
        )
    ):
        raise ValueError("screen exclusion artifact does not match the active protocol")
    return {
        str(row["model"])
        for row in payload.get("decisions", ())
        if row.get("stage") == stage
        and row.get("decision") == "exclude_remaining_screen_episodes"
    }


def _plan(
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    pool: dict[str, Any],
    stage: str,
    records: list[dict[str, Any]],
    max_task_blocks: int | None,
    invalidated_keys: set[tuple[str, str, str, str]] | None = None,
    excluded_models: set[str] | None = None,
) -> dict[str, Any]:
    models, task_ids = _stage_scope(protocol, manifest, stage)
    excluded = set(excluded_models or ())
    unknown_exclusions = sorted(excluded - set(models))
    if unknown_exclusions:
        raise ValueError(f"excluded models are outside the stage: {unknown_exclusions}")
    models = [model for model in models if model not in excluded]
    if not models:
        raise ValueError("screen exclusions removed every model from the stage")
    treatment_by_model = {
        str(row["model"]): (index, dict(row))
        for index, row in enumerate(pool["treatments"])
    }
    missing = sorted(set(models) - set(treatment_by_model))
    if missing:
        raise ValueError(f"active models missing from pool: {missing}")

    invalidated = set(invalidated_keys or ())
    valid_records = [
        row
        for row in records
        if (
            str(row.get("study_stage", "")),
            str(row["task_id"]),
            str(row["model"]),
            str(row.get("protocol_sha256", "")),
        )
        not in invalidated
    ]
    keys = [
        (str(row["task_id"]), str(row["model"])) for row in valid_records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("active collection records contain duplicate task/model keys")
    allowed_task_ids = {
        str(row["task_id"])
        for split in ("development", "heldout")
        for row in manifest[split]
    }
    allowed_models = set(treatment_by_model)
    unexpected = sorted(
        (task_id, model)
        for task_id, model in keys
        if task_id not in allowed_task_ids or model not in allowed_models
    )
    if unexpected:
        raise ValueError(f"active records contain keys outside the lock: {unexpected}")

    completed = set(keys)
    blocks = []
    for task_id in task_ids:
        pending_models = [
            model for model in models if (task_id, model) not in completed
        ]
        if pending_models:
            blocks.append({"task_id": task_id, "models": pending_models})
    if max_task_blocks is not None:
        blocks = blocks[:max_task_blocks]

    caps = {
        model: Decimal(str(treatment_by_model[model][1]["episode_hard_cap_usd"]))
        for model in models
    }
    stage_records = [
        row for row in records if str(row.get("study_stage")) == stage
    ]
    return {
        "models": models,
        "task_ids": task_ids,
        "treatment_by_model": treatment_by_model,
        "caps": caps,
        "pending_blocks": blocks,
        "pending_episodes": sum(len(block["models"]) for block in blocks),
        "maximum_pending_cost_usd": sum(
            (
                caps[model]
                for block in blocks
                for model in block["models"]
            ),
            Decimal("0"),
        ),
        "stage_exposure_usd": incremental_exposure(stage_records),
        "total_collection_exposure_usd": incremental_exposure(records),
        "invalidated_episode_count": len(records) - len(valid_records),
        "excluded_models": sorted(excluded),
    }


def _smoke_exposure(output_root: Path) -> Decimal:
    total = Decimal("0")
    for path in output_root.glob("smoke*.json"):
        total += Decimal(
            str(_load_json(path)["observed_conservative_cost_usd"])
        )
    return total


def _interrupted_exposure(path: Path, *, stage: str | None = None) -> Decimal:
    if not path.exists():
        return Decimal("0")
    return sum(
        (
            Decimal(str(row["reserved_usd"]))
            for row in _load_json(path).get("records", ())
            if stage is None or row.get("stage") == stage
        ),
        Decimal("0"),
    )


def _run_with_fresh_adapter(
    task: dict[str, Any],
    treatment: dict[str, Any],
    prices: PriceSnapshot,
    *,
    seed: int,
    hard_limit_usd: Decimal,
    max_output_tokens: int,
    request_timeout_seconds: float,
    trajectory_path: Path,
) -> dict[str, Any]:
    adapter = asyncio.run(_adapter(treatment))
    return _run_episode(
        task,
        treatment,
        adapter,
        prices,
        seed=seed,
        hard_limit_usd=hard_limit_usd,
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=request_timeout_seconds,
        trajectory_path=trajectory_path,
    )


def _ensure_task_images(task_ids: list[str]) -> None:
    for task_id in task_ids:
        image = _image_name(task_id)
        inspected = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
        )
        if inspected.returncode == 0:
            continue
        print(
            json.dumps(
                {
                    "event": "docker_image_pull_start",
                    "task_id": task_id,
                    "image": image,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(
            ["docker", "pull", image],
            check=True,
            timeout=1_800,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=SCREEN_STAGES, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/active_router_protocol.json"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/active_router_protocol.sha256"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/active_router_task_manifest.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_router_v1.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-29_router_v1.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/active_router_v1"),
    )
    parser.add_argument(
        "--screen-exclusions",
        type=Path,
        default=Path("outputs/active_router_v1/screen_exclusions.json"),
    )
    parser.add_argument(
        "--interrupted-exposure",
        type=Path,
        default=Path("outputs/active_router_v1/interrupted_exposure.json"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-task-blocks", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--approved-stage-credit-usd", type=Decimal)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1, 4]")
    if args.max_task_blocks is not None and args.max_task_blocks <= 0:
        raise SystemExit("--max-task-blocks must be positive")
    if not 1 <= args.max_output_tokens <= 8_000:
        raise SystemExit("--max-output-tokens must be in [1, 8000]")
    if args.request_timeout_seconds <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")

    root = Path.cwd()
    validate_protocol_lock(root, args.protocol_lock)
    protocol = _load_json(args.protocol)
    manifest = _load_json(args.manifest)
    pool = _load_json(args.model_pool)
    prices = PriceSnapshot.load(args.prices)
    if pool["price_snapshot"] != prices.snapshot_id:
        raise ValueError("model pool and price snapshot disagree")
    stage_limit = _stage_limit(protocol, args.stage)
    global_limit = Decimal(
        str(protocol["authorization"]["incremental_hard_ceiling_usd"])
    )
    declared_total = sum(
        (
            Decimal(str(row["maximum_usd"]))
            for row in protocol["ordered_budget_stages"]
        ),
        Decimal("0"),
    )
    if declared_total != global_limit:
        raise ValueError("active stage budgets do not equal the global hard ceiling")

    records_path = args.output_root / "episodes.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    invalidated_keys = _invalidated_episode_keys(root, protocol)
    protocol_hash = _sha256(args.protocol)
    excluded_models = _excluded_models(
        args.screen_exclusions,
        study_id=str(protocol["study_id"]),
        protocol_sha256=protocol_hash,
        stage=args.stage,
    )
    plan = _plan(
        protocol=protocol,
        manifest=manifest,
        pool=pool,
        stage=args.stage,
        records=records,
        max_task_blocks=args.max_task_blocks,
        invalidated_keys=invalidated_keys,
        excluded_models=excluded_models,
    )
    smoke_exposure = _smoke_exposure(args.output_root)
    interrupted_exposure = _interrupted_exposure(args.interrupted_exposure)
    stage_interrupted_exposure = _interrupted_exposure(
        args.interrupted_exposure,
        stage=args.stage,
    )
    maximum_total = (
        smoke_exposure
        + interrupted_exposure
        + plan["total_collection_exposure_usd"]
        + plan["maximum_pending_cost_usd"]
    )
    public_plan = {
        "study_id": protocol["study_id"],
        "protocol_sha256": _sha256(args.protocol),
        "stage": args.stage,
        "stage_limit_usd": str(stage_limit),
        "stage_exposure_usd": str(plan["stage_exposure_usd"]),
        "smoke_exposure_usd": str(smoke_exposure),
        "interrupted_exposure_usd": str(interrupted_exposure),
        "stage_interrupted_exposure_usd": str(stage_interrupted_exposure),
        "total_collection_exposure_usd": str(
            plan["total_collection_exposure_usd"]
        ),
        "pending_task_blocks": len(plan["pending_blocks"]),
        "pending_episodes": plan["pending_episodes"],
        "maximum_pending_cost_usd": str(plan["maximum_pending_cost_usd"]),
        "maximum_total_incremental_exposure_usd": str(maximum_total),
        "invalidated_episode_count": plan["invalidated_episode_count"],
        "excluded_models": plan["excluded_models"],
        "global_limit_usd": str(global_limit),
        "execute": args.execute,
    }
    print(json.dumps(public_plan, indent=2, sort_keys=True), flush=True)
    if not args.execute or not plan["pending_blocks"]:
        return
    if (
        plan["stage_exposure_usd"]
        + stage_interrupted_exposure
        + plan["maximum_pending_cost_usd"]
        > stage_limit
    ):
        raise PermissionError("pending work exceeds the active stage budget")
    if maximum_total > global_limit:
        raise PermissionError("pending work exceeds the active global budget")
    if (
        args.approved_stage_credit_usd is None
        or args.approved_stage_credit_usd < plan["maximum_pending_cost_usd"]
    ):
        raise PermissionError(
            "execution requires approval for the displayed maximum pending cost"
        )
    if args.approved_stage_credit_usd > stage_limit:
        raise PermissionError("approval exceeds the locked stage limit")
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", ".minisweagent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    tasks_payload = _load_json(args.tasks)
    tasks_by_id = {
        str(row["instance_id"]): dict(row)
        for row in tasks_payload["records"]
    }
    missing_tasks = sorted(set(plan["task_ids"]) - set(tasks_by_id))
    if missing_tasks:
        raise ValueError(f"locked tasks missing from sanitized dataset: {missing_tasks}")
    _ensure_task_images(
        [
            str(block["task_id"])
            for block in plan["pending_blocks"]
        ]
    )
    all_task_ids = [
        str(row["task_id"])
        for split in ("development", "heldout")
        for row in manifest[split]
    ]
    ordinal_by_task = {
        task_id: ordinal for ordinal, task_id in enumerate(all_task_ids)
    }
    stopped_for_budget = False
    if (
        plan["stage_exposure_usd"]
        + stage_interrupted_exposure
        + plan["maximum_pending_cost_usd"]
        > stage_limit
        or smoke_exposure
        + interrupted_exposure
        + plan["total_collection_exposure_usd"]
        + plan["maximum_pending_cost_usd"]
        > global_limit
    ):
        raise PermissionError("complete pending reservation exceeds a locked budget")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: dict[
            concurrent.futures.Future[dict[str, Any]],
            tuple[str, str],
        ] = {}
        for block in plan["pending_blocks"]:
            task_id = str(block["task_id"])
            for model in block["models"]:
                treatment_index, treatment = plan["treatment_by_model"][model]
                trajectory_path = (
                    args.output_root
                    / "trajectories"
                    / args.stage
                    / model.replace("/", "__").replace(":", "_")
                    / f"{task_id}.json"
                )
                pending[
                    executor.submit(
                        _run_with_fresh_adapter,
                        tasks_by_id[task_id],
                        treatment,
                        prices,
                        seed=(
                            202607290
                            + treatment_index * 100
                            + ordinal_by_task[task_id]
                        ),
                        hard_limit_usd=plan["caps"][model],
                        max_output_tokens=args.max_output_tokens,
                        request_timeout_seconds=args.request_timeout_seconds,
                        trajectory_path=trajectory_path,
                    )
                ] = (task_id, model)
        for future in concurrent.futures.as_completed(pending):
            task_id, _ = pending[future]
            record = future.result()
            record["study_id"] = protocol["study_id"]
            record["protocol_sha256"] = protocol_hash
            record["task_manifest_hash"] = manifest["manifest_hash"]
            record["study_stage"] = args.stage
            record["task_ordinal"] = ordinal_by_task[task_id]
            _write_jsonl_record(records_path, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "episode_complete",
                        "stage": args.stage,
                        "task_id": record["task_id"],
                        "model": record["model"],
                        "exit_status": record["exit_status"],
                        "submitted_patch": record["submitted_patch"],
                        "structurally_valid": record["structurally_valid"],
                        "cost_usd": record["conservative_cost_usd"],
                        "total_collection_exposure_usd": str(
                            incremental_exposure(records)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "schema_version": "active-router-collection-summary-v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "task_manifest_hash": manifest["manifest_hash"],
        "smoke_exposure_usd": str(smoke_exposure),
        "interrupted_exposure_usd": str(interrupted_exposure),
        "collection_exposure_usd": str(incremental_exposure(records)),
        "total_incremental_exposure_usd": str(
            smoke_exposure + interrupted_exposure + incremental_exposure(records)
        ),
        "completed_episodes_by_stage": {
            stage: sum(str(row.get("study_stage")) == stage for row in records)
            for stage in SCREEN_STAGES
        },
        "provider_failure_count": sum(
            bool(row.get("provider_failed")) for row in records
        ),
        "invalidated_episode_count": sum(
            (
                str(row.get("study_stage", "")),
                str(row["task_id"]),
                str(row["model"]),
                str(row.get("protocol_sha256", "")),
            )
            in invalidated_keys
            for row in records
        ),
        "excluded_models_by_stage": {
            args.stage: plan["excluded_models"],
        },
        "stopped_for_budget": stopped_for_budget,
    }
    _write_json(args.output_root / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
