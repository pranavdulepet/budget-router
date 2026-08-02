from __future__ import annotations

from budget_router.serialization import stable_hash
from scripts.route_isolated_stage_policy import route_issue


def test_isolated_policy_router_uses_frozen_score() -> None:
    artifact = {
        "selected_candidate_policy_id": "candidate",
        "fixed_policy_id": "fixed",
        "selector": {
            "cost_weight": 0.1,
            "expected_mean_saving_ratio": 0.5,
            "threshold": -1.0,
        },
        "model_head": {
            "kind": "hashed-linear-v1",
            "dimension": 8,
            "hash_seed": "test",
            "models": {
                "candidate_win": {"bias": 0.0, "weights": {}},
                "fixed_loss": {"bias": 0.0, "weights": {}},
            },
            "training_examples": 1,
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)

    decision = route_issue(
        artifact,
        "Parser rejects a valid expression.",
        repository="example/repo",
    )

    assert decision["selected_policy_id"] == "candidate"
    assert decision["candidate_win_probability"] == 0.5
    assert decision["fixed_loss_probability"] == 0.5
    assert decision["cost_component"] == 0.05
    assert decision["repository_identity_used_as_feature"] is False
