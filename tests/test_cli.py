import asyncio
import json
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import TestCase

from budget_router.cli import (
    _decision,
    command_agent_route,
    command_collect,
    command_evaluate,
    command_pilot,
    command_train,
)
from budget_router.agent_step import FrozenAgentStepArtifact
from budget_router.guarded_agent_step import GuardedAgentStepArtifact
from tests.test_semantic_router import _artifact as semantic_artifact
from budget_router.ledger import BudgetLedger
from budget_router.serialization import read_jsonl, write_jsonl
from budget_router.session import RouterSession
from budget_router.trace import RouterTrace, TraceWriter
from budget_router.types import (
    Budget,
    EventKind,
    GoalContext,
    Operation,
    RouterEvent,
    RouterState,
    TerminalOutcome,
)

from tests.helpers import make_prices


class CliWorkflowTests(TestCase):
    @staticmethod
    def pilot_rows() -> list[dict[str, object]]:
        return [
            {
                "model": f"model-{index}",
                "episodes": 12,
                "structurally_valid": 10,
                "projected_mean_cost_usd": "0.001",
                "throughput_tasks_per_hour": 2,
                "failure_rate": 0.01,
            }
            for index in range(5)
        ]

    def test_pilot_and_collection_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot_path = root / "pilot.json"
            pilot_path.write_text(
                json.dumps({"results": self.pilot_rows()}), encoding="utf-8"
            )
            report = command_pilot(
                pilot_path, approved_credit_usd=Decimal("10")
            )
            self.assertTrue(report["may_collect"])

            task_formats = (
                {"tasks": [{"task_id": "a"}, {"task_id": "b"}]},
                {"records": [{"instance_id": "a"}, {"instance_id": "b"}]},
            )
            for index, task_data in enumerate(task_formats):
                tasks_path = root / f"tasks-{index}.json"
                tasks_path.write_text(json.dumps(task_data), encoding="utf-8")
                output = root / f"jobs-{index}.jsonl"
                collected = command_collect(
                    tasks_path,
                    output,
                    pilot_results_path=pilot_path,
                    approved_credit_usd=Decimal("10"),
                    reference_matrix=False,
                )
                self.assertEqual(collected["jobs_materialized"], 30)
                self.assertEqual(len(read_jsonl(output)), 30)
                self.assertFalse(collected["paid_calls_started"])

    def test_evaluate_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "cost_usd": cost,
                        "success": success,
                        "probability": probability,
                        "latency_ms": 10,
                        "tokens": 100,
                        "switches": 0,
                    }
                    for cost, success, probability in (
                        (0.01, True, 0.8),
                        (0.02, False, 0.2),
                        (0.03, True, 0.7),
                        (0.04, False, 0.3),
                    )
                ],
            )
            report = command_evaluate(path, budgets=[0.01, 0.02, 0.04])
            self.assertEqual(report["episodes"], 4)
            self.assertIn("success_auc", report)
            self.assertIn("brier", report)

    def test_train_workflow_from_terminal_trace(self) -> None:
        async def build_trace(path: Path) -> None:
            prices = make_prices()
            goal = GoalContext("fix", "repo", ("cheap",), task_id="task")
            ledger = BudgetLedger(
                Budget(Decimal("1"), Decimal("1"), prices.snapshot_id),
                prices,
            )
            session = RouterSession(goal, ledger)
            state = RouterState(
                current_model="cheap",
                verification_passed=True,
            )
            decision = await session.decide(state)
            event = RouterEvent(
                EventKind.TERMINAL,
                turn=0,
                operation=Operation.STOP,
                outcome=TerminalOutcome(True, True, True),
            )
            await session.observe(event)
            TraceWriter(path).append(
                RouterTrace.from_turn(
                    task_id="task",
                    run_id="run",
                    goal=goal,
                    state=state,
                    decision=decision,
                    event=event,
                    remaining_budget_usd=ledger.budget.remaining_usd,
                    conservative_spent_usd=ledger.conservative_spent_usd,
                    billed_spent_usd=ledger.billed_spent_usd,
                    prices=prices,
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            asyncio.run(build_trace(trace_path))
            artifact_path = root / "router.json"
            result = command_train(trace_path, artifact_path)
            self.assertEqual(result["training_records"], 1)
            self.assertTrue(artifact_path.is_file())

    def test_decision_loads_task_aware_model_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "router.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "models": [
                            "Qwen/Qwen3-8B",
                            "Qwen/Qwen3.6-35B-A3B",
                        ],
                        "model_head": {
                            "kind": "hashed-linear-v1",
                            "dimension": 16,
                            "hash_seed": "seed",
                            "models": {
                                "Qwen/Qwen3-8B": {"bias": 2.0, "weights": {}},
                                "Qwen/Qwen3.6-35B-A3B": {
                                    "bias": -2.0,
                                    "weights": {},
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            root = Path(__file__).parents[1]
            result = asyncio.run(
                _decision(
                    goal_text="fix a parser",
                    repository="repo",
                    budget_usd=Decimal("0.9"),
                    prices_path=(
                        root / "configs/tinker_prices_2026-07-27_coding_v3.json"
                    ),
                    artifact_path=artifact_path,
                    explain=True,
                )
            )
            self.assertEqual(
                result["selected_action"]["model"],
                "Qwen/Qwen3-8B",
            )

    def test_agent_route_uses_portable_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "agent.json"
            artifact_path.write_text(
                json.dumps(
                    FrozenAgentStepArtifact(
                        cheap_model="cheap",
                        strong_model="strong",
                        dimension=32,
                        hash_seed="test",
                        bias=-10,
                        weights={},
                        platt_slope=1,
                        platt_intercept=0,
                        cheap_threshold=0.5,
                    ).to_dict()
                ),
                encoding="utf-8",
            )
            messages_path = root / "messages.json"
            messages_path.write_text(
                json.dumps(
                    {"messages": [{"role": "user", "content": "Fix parser.py"}]}
                ),
                encoding="utf-8",
            )

            result = command_agent_route(
                artifact_path,
                messages_path,
                step_index=0,
            )

            self.assertEqual(result["model_id"], "cheap")
            self.assertEqual(result["execution"], "decision_only")
            self.assertEqual(result["router_kind"], "frozen_binary_tier")

    def test_agent_route_uses_portable_guarded_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = FrozenAgentStepArtifact(
                cheap_model="cheap",
                strong_model="strong",
                dimension=32,
                hash_seed="test",
                bias=-10,
                weights={},
                platt_slope=1,
                platt_intercept=0,
                cheap_threshold=0.5,
            )
            guarded = GuardedAgentStepArtifact(
                base_classifier=FrozenAgentStepArtifact.from_dict(
                    base.to_dict()
                ),
                force_strong_initial_calls=2,
                maximum_consecutive_cheap_calls=2,
            )
            artifact_path = root / "guarded.json"
            artifact_path.write_text(
                json.dumps(guarded.to_dict()),
                encoding="utf-8",
            )
            messages_path = root / "messages.json"
            messages_path.write_text(
                json.dumps(
                    {"messages": [{"role": "user", "content": "Fix parser.py"}]}
                ),
                encoding="utf-8",
            )

            result = command_agent_route(
                artifact_path,
                messages_path,
                step_index=0,
            )

            self.assertEqual(result["model_id"], "strong")
            self.assertEqual(result["reason"], "force_strong_initial_calls")
            self.assertEqual(result["execution"], "decision_only")
            self.assertEqual(result["router_kind"], "guarded_binary_tier")

    def test_agent_route_accepts_arbitrary_pool_semantic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "semantic.json"
            artifact_path.write_text(
                json.dumps(semantic_artifact()),
                encoding="utf-8",
            )
            messages_path = root / "messages.json"
            messages_path.write_text(
                json.dumps(
                    {"messages": [{"role": "user", "content": "Use a tool"}]}
                ),
                encoding="utf-8",
            )

            result = command_agent_route(
                artifact_path,
                messages_path,
                required_capabilities=("tools",),
                output_tokens=200,
            )

            self.assertEqual(result["selected_model"], "strong")
            self.assertEqual(result["selected_provider"], "remote")
            self.assertEqual(result["router_kind"], "semantic_arbitrary_pool")
            self.assertEqual(result["execution"], "decision_only")
