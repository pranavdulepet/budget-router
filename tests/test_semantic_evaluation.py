from __future__ import annotations

import json
from pathlib import Path

from budget_router.cli import command_semantic_evaluate
from budget_router.semantic import SemanticRouter
from budget_router.semantic_evaluation import (
    SemanticEvaluationOutcome,
    evaluate_semantic_router,
)
from tests.test_semantic_router import _artifact


def _heldout() -> list[SemanticEvaluationOutcome]:
    rows: list[SemanticEvaluationOutcome] = []
    for request_id, text, cheap_ok, group in (
        ("test-1", "simple greeting", True, "easy"),
        ("test-2", "prove a hard theorem", False, "hard"),
    ):
        rows.extend(
            (
                SemanticEvaluationOutcome(
                    request_id,
                    text,
                    "cheap",
                    acceptable=cheap_ok,
                    cost_usd=0.01,
                    group=group,
                    input_tokens=100,
                    output_tokens=50,
                ),
                SemanticEvaluationOutcome(
                    request_id,
                    text,
                    "strong",
                    acceptable=True,
                    cost_usd=0.20,
                    group=group,
                    input_tokens=100,
                    output_tokens=50,
                ),
            )
        )
    return rows


def test_evaluator_compares_router_modes_with_every_fixed_model() -> None:
    report = evaluate_semantic_router(
        SemanticRouter(_artifact()),
        _heldout(),
        modes=("quality", "balanced", "cost"),
        allow_ungated_policy=True,
    )

    assert report["requests"] == 2
    assert set(report["policies"]) == {
        "fixed:cheap",
        "fixed:strong",
        "router:quality",
        "router:balanced",
        "router:cost",
    }
    assert report["policies"]["fixed:cheap"]["acceptable_count"] == 1
    assert report["policies"]["fixed:strong"]["acceptable_count"] == 2
    assert report["policies"]["fixed:strong"]["total_cost_usd"] == 0.4


def test_semantic_evaluate_command_writes_a_hash_linked_report(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    artifact_path = tmp_path / "router.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    outcomes_path = tmp_path / "heldout.jsonl"
    outcomes_path.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": row.request_id,
                    "text": row.text,
                    "model": row.model,
                    "acceptable": row.acceptable,
                    "cost_usd": row.cost_usd,
                    "group": row.group,
                    "input_tokens": row.input_tokens,
                    "output_tokens": row.output_tokens,
                }
            )
            + "\n"
            for row in _heldout()
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "evaluation.json"

    result = command_semantic_evaluate(
        artifact_path,
        outcomes_path,
        output_path,
    )

    assert output_path.exists()
    assert result["artifact_hash"] == artifact["artifact_hash"]
    assert result["source_sha256"]
    assert result["outcome_matrix_hash"]
    assert result["report_hash"]
    assert len(result["task_decisions"]) == 2


def test_token_priced_evaluation_requires_request_or_global_token_estimates() -> None:
    rows = [
        SemanticEvaluationOutcome(
            row.request_id,
            row.text,
            row.model,
            row.acceptable,
            cost_usd=row.cost_usd,
            group=row.group,
        )
        for row in _heldout()
    ]
    artifact = _artifact()
    for card in artifact["model_cards"]:
        card["input_cost_per_million_usd"] = 1.0
        card["output_cost_per_million_usd"] = 2.0
    artifact_without_hash = dict(artifact)
    artifact_without_hash.pop("artifact_hash")
    from budget_router.serialization import stable_hash

    artifact["artifact_hash"] = stable_hash(artifact_without_hash)

    try:
        evaluate_semantic_router(SemanticRouter(artifact), rows)
    except ValueError as error:
        assert "token-priced model cards require" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing token estimates must fail closed")


def test_token_priced_evaluation_prices_fixed_and_routed_policies() -> None:
    artifact = _artifact()
    for card in artifact["model_cards"]:
        multiplier = 1.0 if card["name"] == "cheap" else 10.0
        card["input_cost_per_million_usd"] = multiplier
        card["output_cost_per_million_usd"] = multiplier * 2
    artifact_without_hash = dict(artifact)
    artifact_without_hash.pop("artifact_hash")
    from budget_router.serialization import stable_hash

    artifact["artifact_hash"] = stable_hash(artifact_without_hash)
    rows = [
        SemanticEvaluationOutcome(
            row.request_id,
            row.text,
            row.model,
            row.acceptable,
            group=row.group,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
        )
        for row in _heldout()
    ]

    report = evaluate_semantic_router(SemanticRouter(artifact), rows)

    assert report["policies"]["fixed:cheap"]["total_cost_usd"] == 0.0004
    assert report["policies"]["fixed:strong"]["total_cost_usd"] == 0.004
