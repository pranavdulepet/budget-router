from __future__ import annotations

from scripts.train_isolated_stage_gate import train_gate


def test_gate_training_uses_leave_one_repository_out_and_is_reproducible() -> None:
    repositories = [f"repo-{index}" for index in range(5)]
    rows = []
    for index in range(35):
        candidate_wins = index % 5 == 0
        fixed_losses = index % 7 == 0 and not candidate_wins
        rows.append(
            {
                "task_id": f"task-{index}",
                "repository": repositories[index % 5],
                "problem_statement": (
                    "parser regression" if candidate_wins else "database regression"
                )
                + f" issue {index}",
                "candidate_resolved": candidate_wins or not fixed_losses,
                "fixed_resolved": fixed_losses or index % 3 == 0,
                "candidate_win": candidate_wins,
                "fixed_loss": fixed_losses,
                "candidate_cost_usd": "0.20",
                "fixed_cost_usd": "0.40",
            }
        )
    protocol = {
        "study_id": "study",
        "manifest_hash": "freeze",
        "fixed_policy": {"policy_id": "fixed"},
        "gate": {
            "dimension": 64,
            "hash_seed": "test-gate",
            "learning_rates": [0.2],
            "l2_values": [0.01],
            "epochs": 5,
            "cost_weights": [0.0, 0.1],
            "decision_thresholds": [0.0, 0.1],
        },
    }
    candidate_freeze = {"selected_policy_id": "candidate"}

    first = train_gate(protocol, candidate_freeze, rows)
    second = train_gate(protocol, candidate_freeze, rows)

    assert first == second
    assert first["training"]["validation"] == "leave-one-repository-out"
    assert first["training"]["repository_count"] == 5
    assert first["training"]["test_labels_accessed"] is False
    assert len(first["cross_validation_decisions"]) == 35
    assert first["model_head"]["training_examples"] == 35

