from decimal import Decimal
from unittest import IsolatedAsyncioTestCase, TestCase

from budget_router.ledger import BudgetLedger
from budget_router.operations import eligible_models, eligible_operations
from budget_router.policy import (
    FactorizedRouterPolicy,
    HeuristicOperationHead,
    LookupModelHead,
    RouterConfig,
)
from budget_router.session import PendingDecisionError, RouterSession
from budget_router.types import (
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
    TokenUsage,
)

from tests.helpers import make_prices


class ActionValidationTests(TestCase):
    def test_parallel_share_is_discrete(self) -> None:
        with self.assertRaises(ValueError):
            RouterAction(Operation.PARALLEL, "cheap", Decimal("0.15"))

    def test_stop_strips_output_allowance(self) -> None:
        action = RouterAction(Operation.STOP)
        self.assertEqual(action.max_output_tokens, 0)

    def test_non_parallel_share_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RouterAction(Operation.CONTINUE, "cheap", Decimal("0.1"))


class EligibilityTests(TestCase):
    def setUp(self) -> None:
        self.goal = GoalContext("fix", "repo", ("cheap", "strong"))

    def test_initial_operations(self) -> None:
        operations = eligible_operations(RouterState(), self.goal)
        self.assertIn(Operation.CONTINUE, operations)
        self.assertIn(Operation.PARALLEL, operations)
        self.assertNotIn(Operation.RESTART, operations)

    def test_verified_state_can_only_stop(self) -> None:
        self.assertEqual(
            eligible_operations(RouterState(verification_passed=True), self.goal),
            (Operation.STOP,),
        )

    def test_switch_hysteresis(self) -> None:
        models, masked = eligible_models(
            RouterState(current_model="cheap", turn=3, last_switch_turn=2),
            self.goal,
            switch_hysteresis_turns=2,
        )
        self.assertEqual(models, ("cheap",))
        self.assertTrue(masked)


class SessionTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.prices = make_prices()
        self.goal = GoalContext(
            "fix",
            "repo",
            ("cheap", "strong"),
            constraints={"max_parallel_operations": 0},
        )

    def session(self, amount: str = "1") -> RouterSession:
        budget = Budget(Decimal(amount), Decimal(amount), self.prices.snapshot_id)
        policy = FactorizedRouterPolicy(
            LookupModelHead({"cheap": 0.6, "strong": 0.7}),
            config=RouterConfig(feasibility_threshold=0.5),
        )
        return RouterSession(
            self.goal,
            BudgetLedger(budget, self.prices),
            policy=policy,
        )

    async def test_decide_observe_settles_reservation(self) -> None:
        session = self.session()
        state = RouterState(estimated_input_tokens=100)
        decision = await session.decide(state)
        self.assertGreater(session.ledger.reserved_usd, 0)
        event = RouterEvent(
            EventKind.MODEL,
            turn=0,
            model=decision.action.model,
            operation=decision.action.operation,
            usage=TokenUsage(input_tokens=100, output_tokens=10),
            metadata={"visible_content": "Implemented a fix."},
        )
        next_state = await session.observe(event)
        self.assertEqual(session.ledger.reserved_usd, 0)
        self.assertGreater(next_state.spent_usd, 0)
        self.assertEqual(next_state.visible_transcript[-1].content, "Implemented a fix.")

    async def test_two_decisions_without_observe_rejected(self) -> None:
        session = self.session()
        state = RouterState(estimated_input_tokens=100)
        await session.decide(state)
        with self.assertRaises(PendingDecisionError):
            await session.decide(state)
        await session.cancel_pending()

    async def test_state_spend_must_match_ledger(self) -> None:
        session = self.session()
        with self.assertRaises(ValueError):
            await session.decide(RouterState(spent_usd=Decimal("0.1")))

    async def test_verified_state_submits(self) -> None:
        session = self.session()
        state = RouterState(verification_passed=True)
        decision = await session.decide(state)
        self.assertEqual(decision.action.operation, Operation.STOP)
        event = RouterEvent(
            EventKind.TERMINAL,
            turn=0,
            operation=Operation.STOP,
            outcome=TerminalOutcome(True, True, True),
        )
        next_state = await session.observe(event)
        self.assertTrue(next_state.terminal)

    async def test_all_visible_tests_must_pass(self) -> None:
        session = self.session()
        state = RouterState(turn=1, estimated_input_tokens=100)
        decision = await session.decide(state)
        event = RouterEvent(
            EventKind.TEST,
            turn=1,
            model=decision.action.model,
            operation=decision.action.operation,
            test_events=(
                RouterTestEvent("one", True),
                RouterTestEvent("two", False),
            ),
        )
        next_state = await session.observe(event)
        self.assertFalse(next_state.verification_passed)

    async def test_tiny_budget_stops_before_call(self) -> None:
        session = self.session("0.000001")
        decision = await session.decide(RouterState(estimated_input_tokens=100))
        self.assertEqual(decision.action.operation, Operation.STOP)
        await session.cancel_pending()

    async def test_restart_resets_visible_transcript_but_retains_spend(self) -> None:
        budget = Budget(Decimal("1"), Decimal("1"), self.prices.snapshot_id)
        policy = FactorizedRouterPolicy(
            LookupModelHead({"cheap": 0.6, "strong": 0.5}),
            HeuristicOperationHead({Operation.RESTART: 10.0}),
            RouterConfig(feasibility_threshold=0.5),
        )
        session = RouterSession(
            self.goal,
            BudgetLedger(budget, self.prices),
            policy=policy,
        )
        state = RouterState(
            turn=1,
            estimated_input_tokens=100,
            visible_transcript=(
                TranscriptMessage("assistant", "stale attempt"),
            ),
        )
        decision = await session.decide(state)
        self.assertEqual(decision.action.operation, Operation.RESTART)
        next_state = await session.observe(
            RouterEvent(
                EventKind.OPERATION,
                turn=1,
                model=decision.action.model,
                operation=Operation.RESTART,
            )
        )
        self.assertEqual(next_state.visible_transcript[0].content, self.goal.goal)
        self.assertEqual(next_state.spent_usd, Decimal("0"))

    def test_online_learning_requires_explicit_opt_in(self) -> None:
        budget = Budget(Decimal("1"), Decimal("1"), self.prices.snapshot_id)
        with self.assertRaises(ValueError):
            RouterSession(
                self.goal,
                BudgetLedger(budget, self.prices),
                online_adapter=object(),
            )
