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
from budget_router.serialization import read_jsonl, stable_hash

try:
    from scripts.run_tinker_pilot import (
        _adapter,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )
except ModuleNotFoundError:
    # Direct execution places the scripts directory, rather than the
    # repository root, on sys.path.
    from run_tinker_pilot import (  # type: ignore[no-redef]
        _adapter,
        _run_episode,
        _write_json,
        _write_jsonl_record,
    )


def incremental_exposure(records: list[dict[str, Any]]) -> Decimal:
    # TinkerMiniSweModel includes any conservative provider-failure reservation
    # in spent_usd, which is serialized as conservative_cost_usd. The
    # provider_failure_reserved_usd field is provenance, not an additive cost.
    return sum(
        (Decimal(str(record["conservative_cost_usd"])) for record in records),
        Decimal("0"),
    )


def _load_local_secrets(path: Path = Path(".env")) -> None:
    """Load a git-ignored local secret file without replacing process values."""
    if not path.exists():
        return
    from dotenv import load_dotenv

    load_dotenv(path, override=False)


def _load_study(path: Path) -> dict[str, Any]:
    study = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(study["manifest_hash"])
    unhashed = dict(study)
    unhashed.pop("manifest_hash")
    if stable_hash(unhashed) != expected_hash:
        raise ValueError("study manifest hash mismatch")
    return study


def _require_stage_gate(
    stage: str,
    study: dict[str, Any],
    *,
    screen_gate_path: Path,
    router_freeze_path: Path,
) -> None:
    requirement = study["stages"][stage]["requires"]
    if requirement is None:
        return
    if requirement == "screen_gate_passed":
        if not screen_gate_path.exists():
            raise PermissionError("expanded collection requires a completed screen gate")
        gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("study_manifest_hash") != study["manifest_hash"]
            or gate.get("screen_gate_passed") is not True
        ):
            raise PermissionError("screen gate did not pass for this study")
        return
    if requirement == "router_frozen":
        if not router_freeze_path.exists():
            raise PermissionError("test collection requires a frozen router")
        freeze = json.loads(router_freeze_path.read_text(encoding="utf-8"))
        if (
            freeze.get("study_manifest_hash") != study["manifest_hash"]
            or not freeze.get("router_artifact_sha256")
        ):
            raise PermissionError("router freeze record is invalid for this study")
        artifact_path = Path(str(freeze.get("router_artifact_path", "")))
        if not artifact_path.is_file():
            raise PermissionError("frozen router artifact is missing")
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha256 != freeze["router_artifact_sha256"]:
            raise PermissionError("frozen router artifact content has changed")
        return
    raise ValueError(f"unknown stage requirement: {requirement}")


def _models_for_stage(
    study: dict[str, Any],
    stage: str,
    *,
    screen_gate_path: Path,
) -> list[str]:
    specification = study["stages"][stage]["models"]
    if isinstance(specification, list):
        models = [str(model) for model in specification]
    elif specification == "screen_survivors":
        if not screen_gate_path.exists():
            raise PermissionError("stage requires screen-survivor model selection")
        gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("study_manifest_hash") != study["manifest_hash"]
            or gate.get("screen_gate_passed") is not True
        ):
            raise PermissionError("screen-survivor model selection is invalid")
        models = [str(model) for model in gate.get("surviving_models", ())]
    else:
        raise ValueError(f"unknown stage model specification: {specification!r}")
    if not models or len(models) != len(set(models)):
        raise ValueError("stage models must be non-empty and unique")
    unknown = sorted(set(models) - set(study["models"]))
    if unknown:
        raise ValueError(f"stage models are outside the study: {unknown}")
    return models


def _plan(
    study: dict[str, Any],
    stage: str,
    records: list[dict[str, Any]],
    *,
    stage_models: list[str] | None = None,
    model_pool: dict[str, Any],
    max_task_blocks: int | None,
) -> dict[str, Any]:
    models = (
        list(stage_models)
        if stage_models is not None
        else [str(model) for model in study["models"]]
    )
    treatments_by_model = {
        str(treatment["model"]): (index, dict(treatment))
        for index, treatment in enumerate(model_pool["treatments"])
    }
    missing = sorted(set(models) - set(treatments_by_model))
    if missing:
        raise ValueError(f"study models missing from model pool: {missing}")
    task_by_id = {
        str(task["task_id"]): dict(task)
        for task in study["tasks"]
    }
    stage_task_ids = [str(task_id) for task_id in study["stages"][stage]["task_ids"]]
    completed = {
        (str(record["task_id"]), str(record["model"]))
        for record in records
    }
    valid_keys = {
        (task_id, model)
        for task_id in task_by_id
        for model in study["models"]
    }
    unexpected = sorted(completed - valid_keys)
    if unexpected:
        raise ValueError(f"records contain keys outside the study: {unexpected}")

    pending_blocks: list[dict[str, Any]] = []
    for task_id in stage_task_ids:
        pending_models = [
            model for model in models if (task_id, model) not in completed
        ]
        if pending_models:
            pending_blocks.append(
                {
                    "task": task_by_id[task_id],
                    "models": pending_models,
                }
            )
    if max_task_blocks is not None:
        pending_blocks = pending_blocks[:max_task_blocks]
    hard_caps = {
        str(model): Decimal(str(cap))
        for model, cap in study["model_hard_caps_usd"].items()
    }
    return {
        "models": models,
        "treatments_by_model": treatments_by_model,
        "pending_blocks": pending_blocks,
        "pending_episodes": sum(len(block["models"]) for block in pending_blocks),
        "maximum_pending_cost_usd": sum(
            (
                hard_caps[model]
                for block in pending_blocks
                for model in block["models"]
            ),
            Decimal("0"),
        ),
        "current_exposure_usd": incremental_exposure(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
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
        "--records",
        type=Path,
        default=Path("outputs/router_baseline_v2/episodes.jsonl"),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("outputs/router_baseline_v2/trajectories"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/router_baseline_v2/collection_summary.json"),
    )
    parser.add_argument(
        "--screen-gate",
        type=Path,
        default=Path("outputs/router_baseline_v2/screen_gate.json"),
    )
    parser.add_argument(
        "--router-freeze",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_freeze.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("cheap_backfill", "screen", "expand", "calibration", "test"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-task-blocks", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--approved-incremental-credit-usd", type=Decimal)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1, 4]")
    if args.max_task_blocks is not None and args.max_task_blocks <= 0:
        raise SystemExit("--max-task-blocks must be positive")

    _load_local_secrets()
    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", ".minisweagent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    study = _load_study(args.study)
    _require_stage_gate(
        args.stage,
        study,
        screen_gate_path=args.screen_gate,
        router_freeze_path=args.router_freeze,
    )
    model_pool = json.loads(args.model_pool.read_text(encoding="utf-8"))
    if model_pool["pool_version"] != study["model_pool_version"]:
        raise ValueError("model pool version does not match study")
    tasks_payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks_by_id = {
        str(record["instance_id"]): dict(record)
        for record in tasks_payload["records"]
    }
    records = read_jsonl(args.records) if args.records.exists() else []
    stage_models = _models_for_stage(
        study,
        args.stage,
        screen_gate_path=args.screen_gate,
    )
    plan = _plan(
        study,
        args.stage,
        records,
        stage_models=stage_models,
        model_pool=model_pool,
        max_task_blocks=args.max_task_blocks,
    )
    working_limit = Decimal(str(study["incremental_working_limit_usd"]))
    absolute_limit = Decimal(str(study["incremental_absolute_limit_usd"]))
    public_plan = {
        "study_id": study["study_id"],
        "stage": args.stage,
        "pending_task_blocks": len(plan["pending_blocks"]),
        "pending_episodes": plan["pending_episodes"],
        "current_incremental_exposure_usd": str(plan["current_exposure_usd"]),
        "maximum_pending_cost_usd": str(plan["maximum_pending_cost_usd"]),
        "working_limit_usd": str(working_limit),
        "absolute_limit_usd": str(absolute_limit),
        "execute": args.execute,
    }
    print(json.dumps(public_plan, indent=2, sort_keys=True))
    if not args.execute or not plan["pending_blocks"]:
        return
    if (
        args.approved_incremental_credit_usd is None
        or args.approved_incremental_credit_usd < absolute_limit
    ):
        raise PermissionError(
            f"execution requires explicit approval for the ${absolute_limit} absolute limit"
        )
    if args.approved_incremental_credit_usd > Decimal("200"):
        raise PermissionError("this study refuses approval above the user-authorized $200 cap")
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    prices = PriceSnapshot.load(args.prices)
    if prices.snapshot_id != study["price_snapshot"]:
        raise ValueError("price snapshot does not match the frozen study")
    adapters = {
        model: asyncio.run(_adapter(plan["treatments_by_model"][model][1]))
        for model in plan["models"]
    }
    hard_caps = {
        str(model): Decimal(str(cap))
        for model, cap in study["model_hard_caps_usd"].items()
    }
    stopped_for_budget = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for block in plan["pending_blocks"]:
            models = list(block["models"])
            full_block_reservation = sum(
                (hard_caps[model] for model in models),
                Decimal("0"),
            )
            exposure = incremental_exposure(records)
            if (
                exposure + full_block_reservation > working_limit
                or exposure + full_block_reservation > absolute_limit
                or exposure + full_block_reservation
                > args.approved_incremental_credit_usd
            ):
                stopped_for_budget = True
                break
            task_meta = block["task"]
            task_id = str(task_meta["task_id"])
            task = tasks_by_id[task_id]
            pending: dict[
                concurrent.futures.Future[dict[str, Any]],
                str,
            ] = {}
            for model in models:
                treatment_index, treatment = plan["treatments_by_model"][model]
                trajectory_path = (
                    args.trajectory_dir
                    / args.stage
                    / model.replace("/", "__").replace(":", "_")
                    / f"{task_id}.json"
                )
                pending[
                    executor.submit(
                        _run_episode,
                        task,
                        treatment,
                        adapters[model],
                        prices,
                        seed=(
                            int(study["episode_seed_base"])
                            + treatment_index * 100
                            + int(task_meta["ordinal"])
                        ),
                        hard_limit_usd=hard_caps[model],
                        max_output_tokens=8_000,
                        request_timeout_seconds=args.request_timeout_seconds,
                        trajectory_path=trajectory_path,
                    )
                ] = model
            for future in concurrent.futures.as_completed(pending):
                record = future.result()
                record["study_id"] = study["study_id"]
                record["study_manifest_hash"] = study["manifest_hash"]
                record["study_stage"] = args.stage
                record["task_ordinal"] = int(task_meta["ordinal"])
                _write_jsonl_record(args.records, record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "stage": args.stage,
                            "task_id": record["task_id"],
                            "model": record["model"],
                            "exit_status": record["exit_status"],
                            "submitted_patch": record["submitted_patch"],
                            "cost_usd": record["conservative_cost_usd"],
                            "incremental_exposure_usd": str(
                                incremental_exposure(records)
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    completed_by_stage: dict[str, int] = {}
    for name in study["stages"]:
        completed_by_stage[name] = sum(
            record.get("study_stage") == name for record in records
        )
    summary = {
        "schema_version": "router-baseline-collection-summary-v1",
        "study_id": study["study_id"],
        "study_manifest_hash": study["manifest_hash"],
        "incremental_exposure_usd": str(incremental_exposure(records)),
        "working_limit_usd": str(working_limit),
        "absolute_limit_usd": str(absolute_limit),
        "completed_episodes_by_stage": completed_by_stage,
        "stopped_for_budget": stopped_for_budget,
        "provider_failure_count": sum(
            bool(record.get("provider_failed")) for record in records
        ),
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
