from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.run_active_router_study import (
    _excluded_models,
    _interrupted_exposure,
    _invalidated_episode_keys,
    _plan,
    _stage_limit,
    _stage_scope,
    incremental_exposure,
    validate_protocol_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_active_protocol_lock_validates() -> None:
    validate_protocol_lock(
        ROOT,
        ROOT / "artifacts/active_router_protocol.sha256",
    )


def test_cheap_screen_plan_matches_locked_budget() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    manifest = _json("artifacts/active_router_task_manifest.json")
    pool = _json("configs/model_pool_router_v1.json")
    models, task_ids = _stage_scope(
        protocol,
        manifest,
        "cheap_medium_screen",
    )
    plan = _plan(
        protocol=protocol,
        manifest=manifest,
        pool=pool,
        stage="cheap_medium_screen",
        records=[],
        max_task_blocks=None,
    )
    assert len(models) == 3
    assert len(task_ids) == 12
    assert plan["pending_episodes"] == 36
    assert plan["maximum_pending_cost_usd"] == Decimal("17.40")
    assert _stage_limit(protocol, "cheap_medium_screen") == Decimal("17.40")


def test_resume_removes_only_the_completed_task_model_pair() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    manifest = _json("artifacts/active_router_task_manifest.json")
    pool = _json("configs/model_pool_router_v1.json")
    task_id = manifest["screens"]["cheap_medium_task_ids"][0]
    model = protocol["candidate_groups"]["cheap_medium"]["models"][0]["model"]
    record = {
        "task_id": task_id,
        "model": model,
        "study_stage": "cheap_medium_screen",
        "conservative_cost_usd": "0.01",
    }
    plan = _plan(
        protocol=protocol,
        manifest=manifest,
        pool=pool,
        stage="cheap_medium_screen",
        records=[record],
        max_task_blocks=1,
    )
    assert plan["pending_episodes"] == 2
    assert plan["maximum_pending_cost_usd"] == Decimal("1.10")
    assert plan["stage_exposure_usd"] == Decimal("0.01")


def test_amendment_invalidates_only_the_pre_repair_nano_episode() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    manifest = _json("artifacts/active_router_task_manifest.json")
    pool = _json("configs/model_pool_router_v1.json")
    invalidated = _invalidated_episode_keys(ROOT, protocol)
    assert len(invalidated) == 4
    stage, task_id, model, old_hash = next(
        key
        for key in invalidated
        if key[2] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    record = {
        "task_id": task_id,
        "model": model,
        "study_stage": stage,
        "protocol_sha256": old_hash,
        "conservative_cost_usd": "0.286",
    }
    plan = _plan(
        protocol=protocol,
        manifest=manifest,
        pool=pool,
        stage=stage,
        records=[record],
        max_task_blocks=1,
        invalidated_keys=invalidated,
    )
    assert plan["pending_episodes"] == 3
    assert plan["invalidated_episode_count"] == 1
    assert plan["stage_exposure_usd"] == Decimal("0.286")


def test_screen_exclusion_removes_nano_from_future_plan() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    manifest = _json("artifacts/active_router_task_manifest.json")
    pool = _json("configs/model_pool_router_v1.json")
    protocol_hash = __import__("hashlib").sha256(
        (ROOT / "artifacts/active_router_protocol.json").read_bytes()
    ).hexdigest()
    excluded = _excluded_models(
        ROOT / "outputs/active_router_v1/screen_exclusions.json",
        study_id=protocol["study_id"],
        protocol_sha256=protocol_hash,
        stage="cheap_medium_screen",
    )
    plan = _plan(
        protocol=protocol,
        manifest=manifest,
        pool=pool,
        stage="cheap_medium_screen",
        records=[],
        max_task_blocks=1,
        excluded_models=excluded,
    )
    assert excluded == {
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    }
    assert plan["pending_episodes"] == 2
    assert plan["maximum_pending_cost_usd"] == Decimal("1.10")


def test_interrupted_reservations_are_conservatively_counted() -> None:
    path = ROOT / "outputs/active_router_v1/interrupted_exposure.json"
    assert _interrupted_exposure(path) == Decimal("4.80")
    assert _interrupted_exposure(
        path,
        stage="cheap_medium_screen",
    ) == Decimal("4.80")


def test_duplicate_records_are_rejected() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    manifest = _json("artifacts/active_router_task_manifest.json")
    pool = _json("configs/model_pool_router_v1.json")
    record = {
        "task_id": manifest["screens"]["cheap_medium_task_ids"][0],
        "model": protocol["candidate_groups"]["cheap_medium"]["models"][0][
            "model"
        ],
        "conservative_cost_usd": "0.01",
    }
    with pytest.raises(ValueError, match="duplicate task/model"):
        _plan(
            protocol=protocol,
            manifest=manifest,
            pool=pool,
            stage="cheap_medium_screen",
            records=[record, dict(record)],
            max_task_blocks=None,
        )


def test_incremental_exposure_uses_conservative_cost_once() -> None:
    assert incremental_exposure(
        [
            {
                "conservative_cost_usd": "0.4",
                "provider_failure_reserved_usd": "0.9",
            }
        ]
    ) == Decimal("0.4")
