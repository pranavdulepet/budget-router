from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_active_protocol_budget_and_candidate_configs_are_consistent() -> None:
    protocol = _json("artifacts/active_router_protocol.json")
    amendment = _json(
        "artifacts/active_router_protocol_amendment_009_three_tier_quality_cost.json"
    )
    pool = _json("configs/model_pool_quality_cost_v1.json")
    prices = _json("configs/tinker_prices_2026-07-30_quality_cost_v1.json")

    ceiling = Decimal(protocol["authorization"]["incremental_hard_ceiling_usd"])
    stage_total = sum(
        (Decimal(stage["maximum_usd"]) for stage in protocol["ordered_budget_stages"]),
        Decimal("0"),
    )
    assert stage_total == ceiling == Decimal("3000.00")
    assert (
        Decimal(amendment["authorization"]["incremental_hard_ceiling_usd"])
        == ceiling
    )

    treatments = {row["model"]: row for row in pool["treatments"]}
    assert set(treatments) == set(prices["models"])
    assert pool["price_snapshot"] == prices["snapshot_id"]

    declared = {row["model"] for row in amendment["models"]}
    assert declared == set(treatments)

    for model, treatment in treatments.items():
        price = prices["models"][model]
        assert treatment["context_tokens"] == price["context_tokens"]
        expected_cap = next(
            row["episode_cap_usd"]
            for row in amendment["models"]
            if row["model"] == model
        )
        assert treatment["episode_hard_cap_usd"] == expected_cap


def test_active_task_manifest_is_locked_and_repository_disjoint() -> None:
    manifest = _json("artifacts/active_router_task_manifest.json")
    embedded_hash = manifest.pop("manifest_hash")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == embedded_hash

    development = {row["task_id"] for row in manifest["development"]}
    heldout = {row["task_id"] for row in manifest["heldout"]}
    cheap_screen = set(manifest["screens"]["cheap_medium_task_ids"])
    strong_screen = set(manifest["screens"]["higher_cost_task_ids"])

    assert len(development) == 24
    assert len(heldout) == 20
    assert development.isdisjoint(heldout)
    assert cheap_screen <= development
    assert strong_screen <= development
    assert len(cheap_screen) == 12
    assert len(strong_screen) == 8

    development_repositories = {
        row["repository"] for row in manifest["development"]
    }
    heldout_repositories = {row["repository"] for row in manifest["heldout"]}
    assert development_repositories.isdisjoint(heldout_repositories)
    assert manifest["invariants"]["heldout_grades_opened"] is False
    assert manifest["invariants"]["heldout_exact_ids_previously_unexecuted"] is True


def test_active_protocol_checksum_lock_matches_files() -> None:
    lock = ROOT / "artifacts/active_router_protocol.sha256"
    for line in lock.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
