"""Command-line interface for training, evaluating, and serving routers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import typer

from .redaction import scan_for_secrets
from .semantic import ModelCard, SemanticOutcome, SemanticRouter, train_semantic_router
from .semantic_agent import SemanticAgentRouter
from .semantic_evaluation import SemanticEvaluationOutcome, evaluate_semantic_router
from .serialization import read_jsonl, stable_json, to_jsonable

app = typer.Typer(
    add_completion=False,
    help="Train and run a provider-neutral model router.",
    no_args_is_help=True,
)


def _emit(value: Any) -> None:
    typer.echo(json.dumps(to_jsonable(value), indent=2, sort_keys=True))


def _write_checked(path: Path, value: Any) -> None:
    if scan_for_secrets(value):
        raise ValueError("refusing to write output that may contain a credential")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value) + "\n", encoding="utf-8")


def command_semantic_train(
    outcomes_path: Path,
    model_cards_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    card_data = json.loads(model_cards_path.read_text(encoding="utf-8"))
    card_rows = card_data.get("models", card_data) if isinstance(card_data, dict) else card_data
    if not isinstance(card_rows, list):
        raise ValueError("model cards must be a list or an object with a models list")
    cards = [ModelCard.from_dict(row) for row in card_rows]
    outcomes = [SemanticOutcome.from_dict(row) for row in read_jsonl(outcomes_path)]
    artifact = train_semantic_router(outcomes, cards)
    _write_checked(output_path, artifact)
    return {
        "output": str(output_path),
        "artifact_hash": artifact["artifact_hash"],
        "models": artifact["models"],
        "training": artifact["training"],
        "selector": artifact["selector"],
    }


def command_semantic_route(
    artifact_path: Path,
    text: str,
    *,
    mode: str = "balanced",
    allowed_models: list[str] | None = None,
    required_capabilities: list[str] | tuple[str, ...] = (),
    denied_providers: list[str] | tuple[str, ...] = (),
    input_tokens: int = 0,
    output_tokens: int = 0,
    max_cost_usd: float | None = None,
    cost_weight: float | None = None,
    quality_threshold: float | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    router = SemanticRouter(json.loads(artifact_path.read_text(encoding="utf-8")))
    return to_jsonable(
        router.route(
            text,
            mode=mode,
            allowed_models=allowed_models,
            required_capabilities=required_capabilities,
            denied_providers=denied_providers,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_cost_usd=max_cost_usd,
            cost_weight=cost_weight,
            quality_threshold=quality_threshold,
            max_quality_drop=max_quality_drop,
            allow_ungated_policy=allow_ungated_policy,
        )
    )


def command_semantic_evaluate(
    artifact_path: Path,
    outcomes_path: Path,
    output_path: Path,
    *,
    modes: list[str] | tuple[str, ...] = ("quality", "balanced"),
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    router = SemanticRouter(json.loads(artifact_path.read_text(encoding="utf-8")))
    outcomes = [
        SemanticEvaluationOutcome.from_dict(row) for row in read_jsonl(outcomes_path)
    ]
    report = evaluate_semantic_router(
        router,
        outcomes,
        modes=modes,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        max_quality_drop=max_quality_drop,
        allow_ungated_policy=allow_ungated_policy,
        source_sha256=hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
    )
    _write_checked(output_path, report)
    return {"output": str(output_path), **report}


def command_agent_route(
    artifact_path: Path,
    messages_path: Path,
    *,
    step_index: int = 0,
    mode: str = "balanced",
    allowed_models: list[str] | None = None,
    required_capabilities: list[str] | tuple[str, ...] = (),
    denied_providers: list[str] | tuple[str, ...] = (),
    input_tokens: int | None = None,
    output_tokens: int = 0,
    max_cost_usd: float | None = None,
    cost_weight: float | None = None,
    quality_threshold: float | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = json.loads(messages_path.read_text(encoding="utf-8"))
    messages = payload.get("messages", payload) if isinstance(payload, dict) else payload
    if not isinstance(messages, list) or not messages:
        raise ValueError("agent messages must be a list or an object with messages")
    decision = SemanticAgentRouter(artifact).select(
        messages,
        step_index=step_index,
        mode=mode,
        allowed_models=allowed_models,
        required_capabilities=required_capabilities,
        denied_providers=denied_providers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        max_cost_usd=max_cost_usd,
        cost_weight=cost_weight,
        quality_threshold=quality_threshold,
        max_quality_drop=max_quality_drop,
        allow_ungated_policy=allow_ungated_policy,
    )
    return {
        **decision.to_dict(),
        "router_kind": "semantic_arbitrary_pool",
        "execution": "decision_only",
    }


def command_serve_proxy(config_path: Path, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("install budget-router[proxy] to serve the proxy") from exc
    from .openai_proxy import app_from_config

    uvicorn.run(app_from_config(config_path), host=host, port=port)


@app.command("semantic-train")
def semantic_train(
    outcomes: Path = typer.Option(..., exists=True, readable=True),
    model_cards: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/semantic_router.json")),
) -> None:
    """Train a calibrated router from a complete request-by-model matrix."""
    _emit(command_semantic_train(outcomes, model_cards, output))


@app.command("semantic-route")
def semantic_route(
    artifact: Path = typer.Option(..., exists=True, readable=True),
    text: str = typer.Option(...),
    mode: str = typer.Option("balanced"),
    allowed_model: list[str] | None = typer.Option(None, "--allowed-model"),
    require_capability: list[str] | None = typer.Option(None, "--require-capability"),
    deny_provider: list[str] | None = typer.Option(None, "--deny-provider"),
    input_tokens: int = typer.Option(0, min=0),
    output_tokens: int = typer.Option(0, min=0),
    max_cost_usd: float | None = typer.Option(None),
    cost_weight: float | None = typer.Option(None),
    quality_threshold: float | None = typer.Option(None),
    max_quality_drop: float = typer.Option(0.05),
    allow_ungated_policy: bool = typer.Option(False),
) -> None:
    """Choose a model without calling a provider."""
    _emit(
        command_semantic_route(
            artifact,
            text,
            mode=mode,
            allowed_models=allowed_model,
            required_capabilities=require_capability or (),
            denied_providers=deny_provider or (),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_cost_usd=max_cost_usd,
            cost_weight=cost_weight,
            quality_threshold=quality_threshold,
            max_quality_drop=max_quality_drop,
            allow_ungated_policy=allow_ungated_policy,
        )
    )


@app.command("semantic-evaluate")
def semantic_evaluate(
    artifact: Path = typer.Option(..., exists=True, readable=True),
    outcomes: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/semantic_evaluation.json")),
    mode: list[str] | None = typer.Option(None, "--mode"),
    input_tokens: int | None = typer.Option(None, min=0),
    output_tokens: int | None = typer.Option(None, min=0),
    max_quality_drop: float = typer.Option(0.05),
    allow_ungated_policy: bool = typer.Option(False),
) -> None:
    """Compare routed policies with every fixed model on held-out outcomes."""
    _emit(
        command_semantic_evaluate(
            artifact,
            outcomes,
            output,
            modes=mode or ("quality", "balanced"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_quality_drop=max_quality_drop,
            allow_ungated_policy=allow_ungated_policy,
        )
    )


@app.command("agent-route")
def agent_route(
    artifact: Path = typer.Option(..., exists=True, readable=True),
    messages: Path = typer.Option(..., exists=True, readable=True),
    step_index: int = typer.Option(0, min=0),
    mode: str = typer.Option("balanced"),
    allowed_model: list[str] | None = typer.Option(None, "--allowed-model"),
    require_capability: list[str] | None = typer.Option(None, "--require-capability"),
    deny_provider: list[str] | None = typer.Option(None, "--deny-provider"),
    input_tokens: int | None = typer.Option(None, min=0),
    output_tokens: int = typer.Option(0, min=0),
    max_cost_usd: float | None = typer.Option(None),
    cost_weight: float | None = typer.Option(None),
    quality_threshold: float | None = typer.Option(None),
    max_quality_drop: float = typer.Option(0.05),
    allow_ungated_policy: bool = typer.Option(False),
) -> None:
    """Choose a model for one visible agent step."""
    _emit(
        command_agent_route(
            artifact,
            messages,
            step_index=step_index,
            mode=mode,
            allowed_models=allowed_model,
            required_capabilities=require_capability or (),
            denied_providers=deny_provider or (),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_cost_usd=max_cost_usd,
            cost_weight=cost_weight,
            quality_threshold=quality_threshold,
            max_quality_drop=max_quality_drop,
            allow_ungated_policy=allow_ungated_policy,
        )
    )


@app.command("serve-proxy")
def serve_proxy(
    config: Path = typer.Option(..., exists=True, readable=True),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65_535),
) -> None:
    """Serve an OpenAI-compatible routing proxy."""
    command_serve_proxy(config, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
