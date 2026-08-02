from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .agent_step import FrozenAgentStepArtifact, FrozenAgentStepRouter
from .guarded_agent_step import GuardedAgentStepArtifact, GuardedAgentStepRouter
from .calibration import IsotonicCalibrator
from .experiments import (
    FixedMatrixPlan,
    PilotGate,
    PilotModelResult,
    derive_budget_levels,
    project_matrix_cost,
)
from .ledger import BudgetLedger
from .metrics import (
    aggregate_episode_metrics,
    brier_score,
    calibration_intercept_slope,
    false_feasible_rate,
    log_loss,
    repository_bootstrap_interval,
    risk_coverage,
    success_at_budget,
)
from .policy import FactorizedRouterPolicy, LookupModelHead, RouterConfig
from .pricing import PriceSnapshot
from .redaction import scan_for_secrets
from .replay import replay
from .serialization import read_jsonl, stable_json, to_jsonable, write_jsonl
from .session import RouterSession
from .semantic import ModelCard, SemanticOutcome, SemanticRouter, train_semantic_router
from .semantic_agent import SemanticAgentRouter
from .semantic_evaluation import (
    SemanticEvaluationOutcome,
    evaluate_semantic_router,
)
from .task_model import HashedLinearModelHead
from .trace import TraceWriter
from .types import Budget, GoalContext, RouterState

INITIAL_MODELS = (
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen/Qwen3.6-35B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "thinkingmachines/Inkling",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
)


def load_default_prices() -> PriceSnapshot:
    resource = files("budget_router.data").joinpath("tinker_prices_2026-07-25.json")
    return PriceSnapshot.from_dict(json.loads(resource.read_text(encoding="utf-8")))


def _prices(path: Path | None) -> PriceSnapshot:
    return PriceSnapshot.load(path) if path else load_default_prices()


def _emit(value: Any) -> None:
    print(json.dumps(to_jsonable(value), indent=2, sort_keys=True))


def _pilot_result(item: dict[str, Any]) -> PilotModelResult:
    return PilotModelResult(
        model=str(item["model"]),
        episodes=int(item["episodes"]),
        structurally_valid=int(item["structurally_valid"]),
        projected_mean_cost_usd=Decimal(str(item["projected_mean_cost_usd"])),
        throughput_tasks_per_hour=float(item["throughput_tasks_per_hour"]),
        failure_rate=float(item["failure_rate"]),
    )


def command_pilot(
    results_path: Path,
    *,
    approved_credit_usd: Decimal | None = None,
) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data
    results = tuple(_pilot_result(item) for item in rows)
    pool_complete = len(results) == 5 and len({result.model for result in results}) == 5
    projected = project_matrix_cost(results)
    gate = PilotGate(
        results=results,
        projected_matrix_cost_usd=projected,
        required_credit_usd=projected,
        approved_credit_usd=approved_credit_usd,
    )
    return {
        "models": [to_jsonable(result) | {"passes": result.passes} for result in results],
        "model_pool_complete": pool_complete,
        "structural_gate_passed": gate.structurally_valid and pool_complete,
        "projected_matrix_cost_usd": str(projected),
        "required_credit_usd": str(gate.required_credit_usd),
        "credit_approved": gate.credit_approved,
        "may_collect": gate.may_collect and pool_complete,
        "next_action": (
            "collection_authorized"
            if gate.may_collect and pool_complete
            else "pause_for_credit_approval"
            if gate.structurally_valid and pool_complete
            else "replace_incompatible_models_and_repeat_pilot"
        ),
        "matrix_shape": "500×5×3",
    }


def command_collect(
    tasks_path: Path,
    output_path: Path,
    *,
    pilot_results_path: Path,
    approved_credit_usd: Decimal,
    reference_matrix: bool = True,
) -> dict[str, Any]:
    pilot_data = json.loads(pilot_results_path.read_text(encoding="utf-8"))
    rows = pilot_data["results"] if isinstance(pilot_data, dict) else pilot_data
    results = tuple(_pilot_result(item) for item in rows)
    if len(results) != 5 or len({result.model for result in results}) != 5:
        raise RuntimeError("collection requires five unique pilot-gated treatments")
    projection = project_matrix_cost(results)
    gate = PilotGate(results, projection, projection, approved_credit_usd)
    gate.require_collection_approval()
    tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))
    if isinstance(tasks_data, dict):
        task_rows = tasks_data.get("tasks", tasks_data.get("records"))
        if task_rows is None:
            raise ValueError("task input must contain a tasks or records list")
    else:
        task_rows = tasks_data
    task_ids = tuple(
        str(
            (
                item.get("task_id", item.get("instance_id"))
                if isinstance(item, dict)
                else item
            )
        )
        for item in task_rows
    )
    if any(task_id in {"", "None"} for task_id in task_ids):
        raise ValueError("every task row must provide task_id or instance_id")
    matrix = FixedMatrixPlan(
        task_ids=task_ids,
        models=tuple(result.model for result in results),
    )
    if reference_matrix:
        matrix.validate_reference_design()
    jobs = (
        {
            "task_id": task_id,
            "model": model,
            "seed": seed,
            "status": "pending",
            "policy": "fixed_model",
        }
        for task_id, model, seed in matrix.runs()
    )
    write_jsonl(output_path, jobs)
    return {
        "jobs_materialized": matrix.run_count,
        "output": str(output_path),
        "approved_credit_usd": str(approved_credit_usd),
        "projected_cost_usd": str(projection),
        "paid_calls_started": False,
        "next_action": "launch the provider runner explicitly with this immutable job queue",
    }


def command_train(traces_path: Path, output_path: Path) -> dict[str, Any]:
    traces = TraceWriter.read(traces_path)
    totals: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    costs: dict[str, list[float]] = defaultdict(list)
    for trace in traces:
        model = trace.model_version
        if not model or not trace.event or not trace.event.get("outcome"):
            continue
        outcome = trace.event["outcome"]
        totals[model] += 1
        successes[model] += int(bool(outcome.get("resolved") and outcome.get("verified")))
        costs[model].append(float(trace.conservative_spent_usd))
    if not totals:
        raise ValueError("training requires terminal trace records with model outcomes")
    global_rate = sum(successes.values()) / sum(totals.values())
    priors = {
        model: (successes[model] + global_rate) / (totals[model] + 1)
        for model in sorted(totals)
    }
    artifact = {
        "schema_version": "router-artifact-v1",
        "policy": "factorized_lookup",
        "model_head": {"success_priors": priors},
        "operation_head": {"kind": "heuristic-v1"},
        "ablations": ["model_only", "operation_only", "joint"],
        "training_records": sum(totals.values()),
        "mean_terminal_cost_usd": {
            model: mean(values) for model, values in costs.items()
        },
        "online_learning": False,
    }
    if scan_for_secrets(artifact):
        raise ValueError("refusing to write an artifact containing possible secrets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(artifact) + "\n", encoding="utf-8")
    return {"output": str(output_path), **artifact}


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
    outcomes = [
        SemanticOutcome.from_dict(row) for row in read_jsonl(outcomes_path)
    ]
    artifact = train_semantic_router(outcomes, cards)
    if scan_for_secrets(artifact):
        raise ValueError("refusing to write an artifact containing possible secrets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(artifact) + "\n", encoding="utf-8")
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
    allowed_models: Sequence[str] | None = None,
    required_capabilities: Sequence[str] = (),
    denied_providers: Sequence[str] = (),
    input_tokens: int = 0,
    output_tokens: int = 0,
    max_cost_usd: float | None = None,
    cost_weight: float | None = None,
    quality_threshold: float | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    router = SemanticRouter(artifact)
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
    modes: Sequence[str] = ("quality", "balanced"),
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    router = SemanticRouter(artifact)
    outcomes = [
        SemanticEvaluationOutcome.from_dict(row)
        for row in read_jsonl(outcomes_path)
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
    if scan_for_secrets(report):
        raise ValueError("refusing to write an evaluation containing possible secrets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(report) + "\n", encoding="utf-8")
    return {"output": str(output_path), **report}


def command_agent_route(
    artifact_path: Path,
    messages_path: Path,
    *,
    step_index: int = 0,
    mode: str = "balanced",
    allowed_models: Sequence[str] | None = None,
    required_capabilities: Sequence[str] = (),
    denied_providers: Sequence[str] = (),
    input_tokens: int | None = None,
    output_tokens: int = 0,
    max_cost_usd: float | None = None,
    cost_weight: float | None = None,
    quality_threshold: float | None = None,
    max_quality_drop: float = 0.05,
    allow_ungated_policy: bool = False,
) -> dict[str, Any]:
    """Select a model for one visible agent prefix without calling a provider."""
    artifact_value = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = json.loads(messages_path.read_text(encoding="utf-8"))
    messages = payload.get("messages", payload) if isinstance(payload, dict) else payload
    if not isinstance(messages, list) or not messages:
        raise ValueError("agent messages must be a list or an object with messages")
    schema = artifact_value.get("schema_version")
    if schema == "agent-step-router-artifact-v1":
        artifact = FrozenAgentStepArtifact.from_dict(artifact_value)
        decision = FrozenAgentStepRouter(artifact).select(
            messages,
            step_index=step_index,
        )
        return {
            **decision.to_dict(),
            "artifact_hash": artifact.artifact_hash,
            "router_kind": "frozen_binary_tier",
            "execution": "decision_only",
        }
    if schema == "guarded-agent-step-router-artifact-v2":
        artifact = GuardedAgentStepArtifact.from_dict(artifact_value)
        decision = GuardedAgentStepRouter(artifact).select(
            messages,
            step_index=step_index,
        )
        return {
            **decision.to_dict(),
            "artifact_hash": artifact.artifact_hash,
            "router_kind": "guarded_binary_tier",
            "execution": "decision_only",
        }
    if schema == "semantic-router-artifact-v1":
        decision = SemanticAgentRouter(artifact_value).select(
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
    raise ValueError(f"unsupported agent routing artifact schema: {schema!r}")


def command_serve_proxy(
    config_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    if not 0 < port < 65_536:
        raise ValueError("port must be in [1, 65535]")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install budget-router[proxy] to serve the OpenAI-compatible proxy"
        ) from exc
    from .openai_proxy import app_from_config

    uvicorn.run(app_from_config(config_path), host=host, port=port)


def command_evaluate(
    episodes_path: Path,
    *,
    budgets: Sequence[float] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(episodes_path)
    costs = [float(row["cost_usd"]) for row in rows]
    successes = [bool(row["success"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    if budgets is None:
        successful_costs = [cost for cost, success in zip(costs, successes, strict=True) if success]
        budgets = [float(value) for value in derive_budget_levels(successful_costs)]
    calibration = IsotonicCalibrator().fit(probabilities, successes)
    calibrated = calibration.predict(probabilities)
    metrics = aggregate_episode_metrics(
        costs=costs,
        successes=successes,
        latencies_ms=[float(row.get("latency_ms", 0)) for row in rows],
        tokens=[int(row.get("tokens", 0)) for row in rows],
        switches=[int(row.get("switches", 0)) for row in rows],
        budgets=budgets,
    )
    intercept, slope = calibration_intercept_slope(calibrated, successes)
    report = {
        "episodes": len(rows),
        "budgets_usd": list(budgets),
        "success_at_budget": {
            str(budget): success_at_budget(costs, successes, budget)
            for budget in budgets
        },
        **metrics,
        "brier": brier_score(calibrated, successes),
        "log_loss": log_loss(calibrated, successes),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "false_feasible_rate_at_0.8": false_feasible_rate(
            calibrated, successes, threshold=0.8
        ),
        "risk_coverage": to_jsonable(risk_coverage(calibrated, successes)),
        "calibration_wrapper": "isotonic-pava-v1",
    }
    if all("repository" in row for row in rows):
        values = [
            float(success and cost <= max(budgets))
            for cost, success in zip(costs, successes, strict=True)
        ]
        report["repository_bootstrap_success_interval"] = (
            repository_bootstrap_interval(
                values,
                [str(row["repository"]) for row in rows],
                samples=2_000,
                seed=0,
            )
        )
    return report


async def _decision(
    *,
    goal_text: str,
    repository: str,
    budget_usd: Decimal,
    prices_path: Path | None,
    artifact_path: Path | None,
    explain: bool,
) -> dict[str, Any]:
    prices = _prices(prices_path)
    priors: dict[str, float] = {}
    allowed_models = INITIAL_MODELS
    model_head: Any = LookupModelHead()
    if artifact_path:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_head = artifact["model_head"]
        if artifact_head.get("kind") == "hashed-linear-v1":
            allowed_models = tuple(str(model) for model in artifact["models"])
            model_head = HashedLinearModelHead.from_artifact(artifact_head)
        else:
            priors = {
                str(key): float(value)
                for key, value in artifact_head["success_priors"].items()
            }
            model_head = LookupModelHead(success_priors=priors)
    missing_prices = sorted(set(allowed_models) - set(prices.models))
    if missing_prices:
        raise ValueError(f"router models are missing from the price snapshot: {missing_prices}")
    goal = GoalContext(
        goal=goal_text,
        repository=repository,
        allowed_models=allowed_models,
        constraints={"max_parallel_operations": 1},
    )
    budget = Budget(budget_usd, budget_usd, prices.snapshot_id)
    ledger = BudgetLedger(budget, prices)
    policy = FactorizedRouterPolicy(
        model_head=model_head,
        config=RouterConfig(),
    )
    session = RouterSession(goal, ledger, policy=policy)
    state = RouterState(estimated_input_tokens=max(128, len(goal_text) // 3))
    decision = await session.decide(state)
    await session.cancel_pending()
    result: dict[str, Any] = {
        "selected_action": to_jsonable(decision.action),
        "reason_codes": [reason.value for reason in decision.reason_codes],
        "policy_version": decision.policy_version,
        "remaining_budget_usd": str(ledger.budget.remaining_usd),
        "price_snapshot": prices.snapshot_id,
        "execution": "decision_only",
    }
    if explain:
        result.update(
            {
                "feasibility_curve": to_jsonable(decision.feasibility_curve),
                "action_estimates": to_jsonable(decision.action_estimates),
                "masked_actions": dict(decision.masked_actions),
                "uncertainty": decision.uncertainty,
                "switch_cost_usd": str(decision.switch_cost_usd),
            }
        )
    return result


def command_replay(
    trace_path: Path,
    *,
    hard_limit_usd: Decimal,
    prices_path: Path | None = None,
) -> dict[str, Any]:
    report = replay(
        trace_path,
        _prices(prices_path),
        hard_limit_usd=hard_limit_usd,
    )
    return to_jsonable(report)


try:
    import typer
except ImportError:  # pragma: no cover - used in dependency-free source checkouts
    typer = None


if typer is not None:
    app = typer.Typer(
        name="budget-router",
        help="Train, evaluate, and serve provider-neutral model routers.",
        no_args_is_help=True,
    )

    @app.command()
    def pilot(
        results: Path = typer.Option(..., exists=True, readable=True),
        approved_credit_usd: str | None = typer.Option(None),
    ) -> None:
        """Validate the 12-task structural pilot and report the credit gate."""
        _emit(
            command_pilot(
                results,
                approved_credit_usd=(
                    Decimal(approved_credit_usd)
                    if approved_credit_usd is not None
                    else None
                ),
            )
        )

    @app.command()
    def collect(
        tasks: Path = typer.Option(..., exists=True, readable=True),
        output: Path = typer.Option(Path("outputs/fixed_matrix_jobs.jsonl")),
        pilot_results: Path = typer.Option(..., exists=True, readable=True),
        approved_credit_usd: str = typer.Option(...),
        reference_matrix: bool = typer.Option(True),
    ) -> None:
        """Materialize the approved fixed-policy matrix without starting paid calls."""
        _emit(
            command_collect(
                tasks,
                output,
                pilot_results_path=pilot_results,
                approved_credit_usd=Decimal(approved_credit_usd),
                reference_matrix=reference_matrix,
            )
        )

    @app.command()
    def train(
        traces: Path = typer.Option(..., exists=True, readable=True),
        output: Path = typer.Option(Path("outputs/router.json")),
    ) -> None:
        """Fit a reproducible lookup-head artifact from terminal traces."""
        _emit(command_train(traces, output))

    @app.command("semantic-train")
    def semantic_train(
        outcomes: Path = typer.Option(..., exists=True, readable=True),
        model_cards: Path = typer.Option(..., exists=True, readable=True),
        output: Path = typer.Option(Path("outputs/semantic_router.json")),
    ) -> None:
        """Train a calibrated router from any complete prompt/model outcome matrix."""
        _emit(command_semantic_train(outcomes, model_cards, output))

    @app.command("semantic-route")
    def semantic_route(
        artifact: Path = typer.Option(..., exists=True, readable=True),
        text: str = typer.Option(...),
        mode: str = typer.Option("balanced"),
        allowed_model: list[str] | None = typer.Option(None, "--allowed-model"),
        require_capability: list[str] | None = typer.Option(
            None, "--require-capability"
        ),
        deny_provider: list[str] | None = typer.Option(None, "--deny-provider"),
        input_tokens: int = typer.Option(0),
        output_tokens: int = typer.Option(0),
        max_cost_usd: float | None = typer.Option(None),
        cost_weight: float | None = typer.Option(None),
        quality_threshold: float | None = typer.Option(None),
        max_quality_drop: float = typer.Option(0.05),
        allow_ungated_policy: bool = typer.Option(False),
    ) -> None:
        """Choose a model without calling it; returns scores, costs, and filters."""
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
        input_tokens: int | None = typer.Option(None),
        output_tokens: int | None = typer.Option(None),
        max_quality_drop: float = typer.Option(0.05),
        allow_ungated_policy: bool = typer.Option(False),
    ) -> None:
        """Evaluate routed modes and every fixed model on an untouched matrix."""
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
        require_capability: list[str] | None = typer.Option(
            None, "--require-capability"
        ),
        deny_provider: list[str] | None = typer.Option(None, "--deny-provider"),
        input_tokens: int | None = typer.Option(None),
        output_tokens: int = typer.Option(0),
        max_cost_usd: float | None = typer.Option(None),
        cost_weight: float | None = typer.Option(None),
        quality_threshold: float | None = typer.Option(None),
        max_quality_drop: float = typer.Option(0.05),
        allow_ungated_policy: bool = typer.Option(False),
    ) -> None:
        """Choose from a binary or arbitrary model pool for one agent step."""
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
        """Serve the router as an OpenAI-compatible chat-completions gateway."""
        command_serve_proxy(config, host=host, port=port)

    @app.command()
    def evaluate(
        episodes: Path = typer.Option(..., exists=True, readable=True),
        budget: list[float] | None = typer.Option(None, "--budget"),
    ) -> None:
        """Evaluate calibrated success-versus-budget metrics."""
        _emit(command_evaluate(episodes, budgets=budget))

    @app.command("run")
    def run_command(
        goal: str = typer.Option(...),
        repository: str = typer.Option("."),
        budget_usd: str = typer.Option(...),
        prices: Path | None = typer.Option(None),
        artifact: Path | None = typer.Option(None),
    ) -> None:
        """Return the first reserved router action; harness execution is explicit."""
        _emit(
            asyncio.run(
                _decision(
                    goal_text=goal,
                    repository=repository,
                    budget_usd=Decimal(budget_usd),
                    prices_path=prices,
                    artifact_path=artifact,
                    explain=False,
                )
            )
        )

    @app.command()
    def explain(
        goal: str = typer.Option(...),
        repository: str = typer.Option("."),
        budget_usd: str = typer.Option(...),
        prices: Path | None = typer.Option(None),
        artifact: Path | None = typer.Option(None),
    ) -> None:
        """Explain candidates, estimates, masks, uncertainty, and reason codes."""
        _emit(
            asyncio.run(
                _decision(
                    goal_text=goal,
                    repository=repository,
                    budget_usd=Decimal(budget_usd),
                    prices_path=prices,
                    artifact_path=artifact,
                    explain=True,
                )
            )
        )

    @app.command("replay")
    def replay_command(
        trace: Path = typer.Option(..., exists=True, readable=True),
        hard_limit_usd: str = typer.Option(...),
        prices: Path | None = typer.Option(None),
    ) -> None:
        """Deterministically recompute trace spend and hard-cap violations."""
        _emit(
            command_replay(
                trace,
                hard_limit_usd=Decimal(hard_limit_usd),
                prices_path=prices,
            )
        )


def _fallback_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="budget-router")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "pilot",
        "collect",
        "train",
        "semantic-train",
        "semantic-route",
        "semantic-evaluate",
        "agent-route",
        "serve-proxy",
        "evaluate",
        "run",
        "explain",
        "replay",
    ):
        subparsers.add_parser(name)
    namespace, rest = parser.parse_known_args(argv)
    command = namespace.command
    child = argparse.ArgumentParser(prog=f"budget-router {command}")
    if command == "pilot":
        child.add_argument("--results", type=Path, required=True)
        child.add_argument("--approved-credit-usd", type=Decimal)
        args = child.parse_args(rest)
        _emit(command_pilot(args.results, approved_credit_usd=args.approved_credit_usd))
    elif command == "collect":
        child.add_argument("--tasks", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--pilot-results", type=Path, required=True)
        child.add_argument("--approved-credit-usd", type=Decimal, required=True)
        child.add_argument("--allow-non-reference-matrix", action="store_true")
        args = child.parse_args(rest)
        _emit(
            command_collect(
                args.tasks,
                args.output,
                pilot_results_path=args.pilot_results,
                approved_credit_usd=args.approved_credit_usd,
                reference_matrix=not args.allow_non_reference_matrix,
            )
        )
    elif command == "train":
        child.add_argument("--traces", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        args = child.parse_args(rest)
        _emit(command_train(args.traces, args.output))
    elif command == "semantic-train":
        child.add_argument("--outcomes", type=Path, required=True)
        child.add_argument("--model-cards", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        args = child.parse_args(rest)
        _emit(command_semantic_train(args.outcomes, args.model_cards, args.output))
    elif command == "semantic-route":
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--text", required=True)
        child.add_argument("--mode", choices=("quality", "balanced", "cost"), default="balanced")
        child.add_argument("--allowed-model", action="append")
        child.add_argument("--require-capability", action="append", default=[])
        child.add_argument("--deny-provider", action="append", default=[])
        child.add_argument("--input-tokens", type=int, default=0)
        child.add_argument("--output-tokens", type=int, default=0)
        child.add_argument("--max-cost-usd", type=float)
        child.add_argument("--cost-weight", type=float)
        child.add_argument("--quality-threshold", type=float)
        child.add_argument("--max-quality-drop", type=float, default=0.05)
        child.add_argument("--allow-ungated-policy", action="store_true")
        args = child.parse_args(rest)
        _emit(
            command_semantic_route(
                args.artifact,
                args.text,
                mode=args.mode,
                allowed_models=args.allowed_model,
                required_capabilities=args.require_capability,
                denied_providers=args.deny_provider,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                max_cost_usd=args.max_cost_usd,
                cost_weight=args.cost_weight,
                quality_threshold=args.quality_threshold,
                max_quality_drop=args.max_quality_drop,
                allow_ungated_policy=args.allow_ungated_policy,
            )
        )
    elif command == "semantic-evaluate":
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--outcomes", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument(
            "--mode",
            action="append",
            choices=("quality", "balanced", "cost"),
        )
        child.add_argument("--input-tokens", type=int)
        child.add_argument("--output-tokens", type=int)
        child.add_argument("--max-quality-drop", type=float, default=0.05)
        child.add_argument("--allow-ungated-policy", action="store_true")
        args = child.parse_args(rest)
        _emit(
            command_semantic_evaluate(
                args.artifact,
                args.outcomes,
                args.output,
                modes=args.mode or ("quality", "balanced"),
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                max_quality_drop=args.max_quality_drop,
                allow_ungated_policy=args.allow_ungated_policy,
            )
        )
    elif command == "agent-route":
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--messages", type=Path, required=True)
        child.add_argument("--step-index", type=int, default=0)
        child.add_argument(
            "--mode", choices=("quality", "balanced", "cost"), default="balanced"
        )
        child.add_argument("--allowed-model", action="append")
        child.add_argument("--require-capability", action="append", default=[])
        child.add_argument("--deny-provider", action="append", default=[])
        child.add_argument("--input-tokens", type=int)
        child.add_argument("--output-tokens", type=int, default=0)
        child.add_argument("--max-cost-usd", type=float)
        child.add_argument("--cost-weight", type=float)
        child.add_argument("--quality-threshold", type=float)
        child.add_argument("--max-quality-drop", type=float, default=0.05)
        child.add_argument("--allow-ungated-policy", action="store_true")
        args = child.parse_args(rest)
        _emit(
            command_agent_route(
                args.artifact,
                args.messages,
                step_index=args.step_index,
                mode=args.mode,
                allowed_models=args.allowed_model,
                required_capabilities=args.require_capability,
                denied_providers=args.deny_provider,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                max_cost_usd=args.max_cost_usd,
                cost_weight=args.cost_weight,
                quality_threshold=args.quality_threshold,
                max_quality_drop=args.max_quality_drop,
                allow_ungated_policy=args.allow_ungated_policy,
            )
        )
    elif command == "serve-proxy":
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--host", default="127.0.0.1")
        child.add_argument("--port", type=int, default=8000)
        args = child.parse_args(rest)
        command_serve_proxy(args.config, host=args.host, port=args.port)
    elif command == "evaluate":
        child.add_argument("--episodes", type=Path, required=True)
        child.add_argument("--budget", type=float, action="append")
        args = child.parse_args(rest)
        _emit(command_evaluate(args.episodes, budgets=args.budget))
    elif command in {"run", "explain"}:
        child.add_argument("--goal", required=True)
        child.add_argument("--repository", default=".")
        child.add_argument("--budget-usd", type=Decimal, required=True)
        child.add_argument("--prices", type=Path)
        child.add_argument("--artifact", type=Path)
        args = child.parse_args(rest)
        _emit(
            asyncio.run(
                _decision(
                    goal_text=args.goal,
                    repository=args.repository,
                    budget_usd=args.budget_usd,
                    prices_path=args.prices,
                    artifact_path=args.artifact,
                    explain=command == "explain",
                )
            )
        )
    else:
        child.add_argument("--trace", type=Path, required=True)
        child.add_argument("--hard-limit-usd", type=Decimal, required=True)
        child.add_argument("--prices", type=Path)
        args = child.parse_args(rest)
        _emit(
            command_replay(
                args.trace,
                hard_limit_usd=args.hard_limit_usd,
                prices_path=args.prices,
            )
        )


def main() -> None:
    if typer is not None:
        app()
    else:
        _fallback_main(sys.argv[1:])


if __name__ == "__main__":
    main()
