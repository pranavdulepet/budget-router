import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from budget_router.handoff import (
    TransferFormat,
    build_handoff,
    freeze_transfer_selection,
    select_transfer_format,
)
from budget_router.parallel import BranchResult, ParallelBranchExecutor
from budget_router.providers import MockProvider, ModelRequest
from budget_router.runner import (
    MiniSweAgentCommand,
    ProviderHarness,
    ReferenceEpisodeRunner,
    run_mini_swe_agent,
)
from budget_router.ledger import BudgetLedger
from budget_router.session import RouterSession
from budget_router.types import (
    DiffSummary,
    Budget,
    EventKind,
    GoalContext,
    Operation,
    RouterAction,
    RouterEvent,
    RouterState,
    TerminalOutcome,
    TestEvent as RouterTestEvent,
    TranscriptMessage,
)
from budget_router.workspace import IsolatedWorkspacePool

from tests.helpers import make_prices


class WorkspaceTests(TestCase):
    def test_branches_are_isolated_and_restart_restores_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "file.txt").write_text("original", encoding="utf-8")
            with IsolatedWorkspacePool(source) as pool:
                branches = pool.fork()
                (branches["a"] / "file.txt").write_text("a", encoding="utf-8")
                self.assertEqual(
                    (branches["b"] / "file.txt").read_text(encoding="utf-8"),
                    "original",
                )
                pool.select("a")
                self.assertEqual((pool.current / "file.txt").read_text(), "a")
                restarted = pool.restart()
                self.assertEqual((restarted / "file.txt").read_text(), "original")
                self.assertEqual((source / "file.txt").read_text(), "original")

    def test_refuses_broad_root(self) -> None:
        with self.assertRaises(ValueError):
            IsolatedWorkspacePool(Path("/"))


class HandoffTests(TestCase):
    def setUp(self) -> None:
        self.goal = GoalContext("fix", "repo", ("m",))
        self.state = RouterState(
            visible_transcript=(
                TranscriptMessage("assistant", "<think>private</think>Visible"),
            ),
            workspace=DiffSummary(
                files_changed=1, patch_hash="abc", summary="changed parser"
            ),
        )

    def test_full_transcript_removes_hidden_reasoning(self) -> None:
        handoff = build_handoff(
            TransferFormat.FULL_VISIBLE_TRANSCRIPT, self.goal, self.state
        )
        self.assertEqual(handoff.messages[0].content, "Visible")

    def test_format_selection_tie_prefers_checkpoint(self) -> None:
        scores = {format: 0.5 for format in TransferFormat}
        self.assertEqual(
            select_transfer_format(scores),
            TransferFormat.STRUCTURED_CHECKPOINT,
        )

    def test_transfer_selection_freezes(self) -> None:
        scores = {format: 0.5 for format in TransferFormat}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            winner = freeze_transfer_selection(path, scores)
            self.assertEqual(winner, TransferFormat.STRUCTURED_CHECKPOINT)
            self.assertEqual(freeze_transfer_selection(path, scores), winner)


class ProviderAndParallelTests(IsolatedAsyncioTestCase):
    async def test_model_request_validates_nucleus_and_top_k_sampling(self) -> None:
        messages = (TranscriptMessage("user", "hello"),)
        with self.assertRaises(ValueError):
            ModelRequest(
                "m",
                messages,
                Operation.CONTINUE,
                10,
                top_p=0,
            )
        with self.assertRaises(ValueError):
            ModelRequest(
                "m",
                messages,
                Operation.CONTINUE,
                10,
                top_k=0,
            )
        with self.assertRaises(ValueError):
            ModelRequest(
                "m",
                messages,
                Operation.CONTINUE,
                10,
                timeout_seconds=0,
            )

    async def test_mock_provider_protocol_and_reasoning_redaction(self) -> None:
        provider = MockProvider(["<think>secret</think>Visible"], output_tokens=5)
        response = await provider.generate(
            ModelRequest(
                "m",
                (TranscriptMessage("user", "hello"),),
                Operation.CONTINUE,
                10,
            )
        )
        self.assertEqual(response.content, "Visible")
        self.assertEqual(response.usage.output_tokens, 5)

    async def test_truncated_private_reasoning_is_not_exposed(self) -> None:
        provider = MockProvider(["<think>unfinished private reasoning"])
        response = await provider.generate(
            ModelRequest(
                "m",
                (TranscriptMessage("user", "hello"),),
                Operation.CONTINUE,
                10,
            )
        )
        self.assertEqual(response.content, "")

    async def test_provider_failure_is_visible_event(self) -> None:
        provider = MockProvider(fail_on_calls=(1,))
        goal = GoalContext("fix", "repo", ("m",))
        harness = ProviderHarness(provider, goal)
        event = await harness.execute(
            RouterAction(Operation.CONTINUE, "m", max_output_tokens=100),
            RouterState(visible_transcript=(TranscriptMessage("user", "fix"),)),
        )
        self.assertEqual(event.kind, EventKind.PROVIDER_ERROR)
        self.assertIn("injected provider failure", event.error or "")

    async def test_visible_verifier_round_trip(self) -> None:
        provider = MockProvider()
        goal = GoalContext("fix", "repo", ("m",))
        harness = ProviderHarness(
            provider,
            goal,
            verify_commands=(("python3", "-c", "print('ok')"),),
        )
        event = await harness.execute(
            RouterAction(Operation.VERIFY, max_output_tokens=0),
            RouterState(turn=1),
        )
        self.assertTrue(event.test_events[0].passed)
        self.assertIn("ok", event.test_events[0].summary)

    async def test_optional_verifier_interpretation_is_metered(self) -> None:
        provider = MockProvider(["Visible diagnosis"], input_tokens=20, output_tokens=5)
        harness = ProviderHarness(
            provider,
            GoalContext("fix", "repo", ("m",)),
            verify_commands=(("python3", "-c", "raise SystemExit(1)"),),
        )
        event = await harness.execute(
            RouterAction(Operation.VERIFY, "m", max_output_tokens=100),
            RouterState(
                turn=1,
                visible_transcript=(TranscriptMessage("user", "fix"),),
            ),
        )
        self.assertFalse(event.test_events[0].passed)
        self.assertEqual(event.metadata["visible_content"], "Visible diagnosis")
        self.assertEqual(event.usage.output_tokens, 5)

    async def test_restart_restores_isolated_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "file").write_text("original", encoding="utf-8")
            with IsolatedWorkspacePool(source) as pool:
                pool.current = pool.fork(("work", "other"))["work"]
                (pool.current / "file").write_text("changed", encoding="utf-8")
                harness = ProviderHarness(
                    MockProvider(),
                    GoalContext("fix", "repo", ("m",)),
                    workspace_pool=pool,
                )
                event = await harness.execute(
                    RouterAction(Operation.RESTART, "m", max_output_tokens=100),
                    RouterState(turn=1),
                )
                self.assertIsNone(event.error)
                self.assertEqual((harness.workspace / "file").read_text(), "original")

    async def test_parallel_selection_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "file").write_text("x", encoding="utf-8")
            with IsolatedWorkspacePool(source) as pool:
                executor = ParallelBranchExecutor(pool)

                async def run_branch(
                    name: str, workspace: Path, budget: Decimal, max_turns: int
                ) -> BranchResult:
                    return BranchResult(
                        name=name,
                        workspace=workspace,
                        turns=max_turns,
                        conservative_cost_usd=budget / 2,
                        visible_tests_passed=name == "b",
                        value_estimate=0.5,
                    )

                result = await executor.execute(
                    RouterAction(Operation.PARALLEL, "m", Decimal("0.40")),
                    remaining_budget_usd=Decimal("1"),
                    run_branch=run_branch,
                )
                self.assertEqual(result.selected.name, "b")
                self.assertEqual(pool.current, result.selected.workspace)
                self.assertEqual(result.per_branch_budget_usd, Decimal("0.40"))

    async def test_mini_swe_adapter_is_dry_run_by_default(self) -> None:
        command = MiniSweAgentCommand(Path("config.yaml"), Path("output"))
        result = await run_mini_swe_agent(command)
        self.assertFalse(result["executed"])
        self.assertEqual(result["argv"][:2], ["mini-extra", "swebench"])

    async def test_reference_episode_round_trip(self) -> None:
        prices = make_prices()
        goal = GoalContext(
            "fix",
            "repo",
            ("cheap",),
            constraints={"max_parallel_operations": 0},
        )
        session = RouterSession(
            goal,
            BudgetLedger(
                Budget(Decimal("1"), Decimal("1"), prices.snapshot_id),
                prices,
            ),
        )

        class Harness:
            async def execute(
                self, action: RouterAction, state: RouterState
            ) -> RouterEvent:
                if action.operation is Operation.STOP:
                    return RouterEvent(
                        EventKind.TERMINAL,
                        state.turn,
                        operation=Operation.STOP,
                        outcome=TerminalOutcome(True, True, True),
                    )
                return RouterEvent(
                    EventKind.TEST,
                    state.turn,
                    model=action.model,
                    operation=action.operation,
                    test_events=(RouterTestEvent("visible", True),),
                )

        result = await ReferenceEpisodeRunner(session, Harness()).run(RouterState())
        self.assertTrue(result.outcome.resolved)
        self.assertEqual(result.turns, 2)
