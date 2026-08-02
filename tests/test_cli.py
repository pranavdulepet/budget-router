from __future__ import annotations

import json
from pathlib import Path

from budget_router.cli import (
    command_agent_route,
    command_semantic_evaluate,
    command_semantic_route,
    command_semantic_train,
)


def test_cli_functions_run_the_complete_example(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    artifact = tmp_path / "router.json"
    evaluation = tmp_path / "evaluation.json"

    trained = command_semantic_train(
        root / "examples/outcomes.example.jsonl",
        root / "examples/model_cards.example.json",
        artifact,
    )
    routed = command_semantic_route(
        artifact,
        "Debug a difficult parser failure with tools.",
        required_capabilities=["tools"],
        input_tokens=1000,
        output_tokens=200,
    )
    evaluated = command_semantic_evaluate(
        artifact,
        root / "examples/heldout.example.jsonl",
        evaluation,
    )
    agent = command_agent_route(
        artifact,
        root / "examples/agent_messages.example.json",
        required_capabilities=["tools"],
    )

    assert trained["artifact_hash"]
    assert routed["selected_model"]
    assert evaluated["requests"] == 2
    assert agent["router_kind"] == "semantic_arbitrary_pool"
    assert json.loads(artifact.read_text())["artifact_hash"] == trained["artifact_hash"]
