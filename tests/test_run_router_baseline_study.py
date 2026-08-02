from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.run_router_baseline_study import (
    _load_local_secrets,
    _load_study,
    _models_for_stage,
    _plan,
    _require_stage_gate,
    incremental_exposure,
)


def test_incremental_exposure_does_not_double_count_failure_reservations() -> None:
    records = [
        {
            "conservative_cost_usd": "0.40",
            "provider_failure_reserved_usd": "0.00",
        },
        {
            "conservative_cost_usd": "0.10",
            "provider_failure_reserved_usd": "0.90",
        },
    ]
    assert incremental_exposure(records) == Decimal("0.50")


def test_local_secret_loader_does_not_override_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("TINKER_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("TINKER_API_KEY", "process-value")
    _load_local_secrets(secret_file)
    assert os.environ["TINKER_API_KEY"] == "process-value"


def test_screen_plan_uses_complete_four_model_blocks_with_tiered_caps() -> None:
    root = Path(__file__).parents[1]
    study = _load_study(root / "artifacts/router_baseline_study_v2.json")
    model_pool = json.loads(
        (root / "configs/model_pool_coding_v7.json").read_text(encoding="utf-8")
    )
    plan = _plan(
        study,
        "screen",
        [],
        model_pool=model_pool,
        max_task_blocks=2,
    )
    assert len(plan["pending_blocks"]) == 2
    assert plan["pending_episodes"] == 8
    assert plan["maximum_pending_cost_usd"] == Decimal("6.10")
    assert all(len(block["models"]) == 4 for block in plan["pending_blocks"])


def test_backfill_plan_only_runs_the_cheap_model() -> None:
    root = Path(__file__).parents[1]
    study = _load_study(root / "artifacts/router_baseline_study_v2.json")
    model_pool = json.loads(
        (root / "configs/model_pool_coding_v7.json").read_text(encoding="utf-8")
    )
    models = _models_for_stage(
        study,
        "cheap_backfill",
        screen_gate_path=root / "unused-screen-gate.json",
    )
    plan = _plan(
        study,
        "cheap_backfill",
        [],
        stage_models=models,
        model_pool=model_pool,
        max_task_blocks=2,
    )
    assert models == ["Qwen/Qwen3-8B"]
    assert plan["pending_episodes"] == 2
    assert plan["maximum_pending_cost_usd"] == Decimal("0.70")


def test_survivor_stage_accepts_prior_records_from_pruned_model() -> None:
    root = Path(__file__).parents[1]
    study = _load_study(root / "artifacts/router_baseline_study_v2.json")
    model_pool = json.loads(
        (root / "configs/model_pool_coding_v7.json").read_text(encoding="utf-8")
    )
    pruned_model = "Qwen/Qwen3-8B"
    survivor_models = [
        model for model in study["models"] if model != pruned_model
    ]
    prior_record = {
        "task_id": study["stages"]["cheap_backfill"]["task_ids"][0],
        "model": pruned_model,
        "conservative_cost_usd": "0.01",
        "provider_failure_reserved_usd": "0",
    }

    plan = _plan(
        study,
        "expand",
        [prior_record],
        stage_models=survivor_models,
        model_pool=model_pool,
        max_task_blocks=1,
    )

    assert plan["pending_episodes"] == 3
    assert plan["maximum_pending_cost_usd"] == Decimal("2.70")


def test_test_stage_requires_matching_router_freeze(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    study = _load_study(root / "artifacts/router_baseline_study_v2.json")
    with pytest.raises(PermissionError, match="frozen router"):
        _require_stage_gate(
            "test",
            study,
            screen_gate_path=tmp_path / "screen.json",
            router_freeze_path=tmp_path / "freeze.json",
        )
