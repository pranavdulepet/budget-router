from __future__ import annotations

import pytest

from budget_router.serialization import stable_hash
from scripts.route_task import route_issue


def _artifact() -> dict[str, object]:
    artifact: dict[str, object] = {
        "models": ["cheap", "strong"],
        "model_head": {
            "kind": "hashed-linear-v1",
            "dimension": 16,
            "hash_seed": "seed",
            "models": {
                "cheap": {"bias": 1.0, "weights": {}},
                "strong": {"bias": 0.0, "weights": {}},
            },
        },
        "selector": {
            "expected_cost_usd": {"cheap": "0.1", "strong": "0.5"},
            "cost_penalty": 0.2,
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def test_route_issue_returns_a_reproducible_cost_aware_decision() -> None:
    result = route_issue(_artifact(), "Fix the parser")

    assert result["selected_model"] == "cheap"
    assert result["expected_cost_usd"] == {"cheap": "0.1", "strong": "0.5"}
    assert result["repository_identity_used_as_feature"] is False


def test_route_issue_rejects_a_tampered_artifact() -> None:
    artifact = _artifact()
    artifact["selector"]["cost_penalty"] = 0.9  # type: ignore[index]

    with pytest.raises(ValueError, match="content hash"):
        route_issue(artifact, "Fix the parser")
