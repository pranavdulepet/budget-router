from __future__ import annotations

import json
from pathlib import Path

from budget_router.serialization import stable_hash


def test_frozen_isolated_stage_manifest_has_disjoint_splits_and_exact_budget() -> None:
    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "artifacts/isolated_stage_router_study_v1.json").read_text()
    )
    unhashed = dict(protocol)
    claimed = unhashed.pop("manifest_hash")

    assert stable_hash(unhashed) == claimed
    assert protocol["budget"]["maximum_new_cost_usd"] == "112.50"
    assert protocol["stages"]["screen"]["episode_count"] == 18
    assert protocol["stages"]["expand"]["episode_count"] == 29
    assert protocol["stages"]["test"]["episode_count"] == 78
    assert len(protocol["stages"]["screen"]["task_ids"]) == 6
    assert len(protocol["stages"]["expand"]["task_ids"]) == 29
    assert len(protocol["stages"]["test"]["task_ids"]) == 39

    training_repositories = {
        row["repository"]
        for row in protocol["tasks"]
        if row["stage"] in {"screen", "expand"}
    }
    test_repositories = {
        row["repository"]
        for row in protocol["tasks"]
        if row["stage"] == "test"
    }
    assert not training_repositories & test_repositories

    source = json.loads(
        (root / "artifacts/router_baseline_study_v2.json").read_text()
    )
    old_task_ids = {row["task_id"] for row in source["tasks"]}
    assert not old_task_ids & set(protocol["stages"]["test"]["task_ids"])

