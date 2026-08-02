import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from budget_router.ledger import BudgetLedger
from budget_router.redaction import REDACTED, redact, scan_for_secrets
from budget_router.replay import replay
from budget_router.session import RouterSession
from budget_router.trace import RouterTrace, TraceWriter
from budget_router.types import (
    Budget,
    EventKind,
    GoalContext,
    RouterEvent,
    RouterState,
    TokenUsage,
)

from tests.helpers import make_prices


class RedactionTests(TestCase):
    def test_recursive_redaction(self) -> None:
        value = {
            "api_key": "do-not-keep",
            "nested": [
                "Bearer abcdefghijklmnopqrstuv",
                "<think>private chain</think>visible",
            ],
        }
        result = redact(value)
        self.assertEqual(result["api_key"], REDACTED)
        self.assertEqual(result["nested"][0], REDACTED)
        self.assertNotIn("private chain", result["nested"][1])
        self.assertFalse(scan_for_secrets(result))

    def test_secret_scanner_finds_private_key(self) -> None:
        self.assertTrue(scan_for_secrets("-----BEGIN PRIVATE KEY-----"))

    def test_secret_scanner_finds_hyphenated_provider_keys(self) -> None:
        self.assertTrue(
            scan_for_secrets("tml-" + "exampleCredentialValue1234567890")
        )
        self.assertTrue(
            scan_for_secrets("sk-" + "exampleCredentialValue1234567890")
        )

    def test_secret_scanner_allows_trace_content_hashes(self) -> None:
        value = {
            "git_sha": "3ae92215dae8f55903f6bc6c8c063e5cb7498bac",
            "container_id": (
                "2f973ede7d3b2b6fbf19f380204a1a0e"
                "8c4f9cc68cc65da186d918d080225450"
            ),
            "public_url": (
                "https://github.com/matplotlib/matplotlib/commit/"
                "3ae92215dae8f55903f6bc6c8c063e5cb7498bac"
            ),
            "traceback_path": (
                "/Users/example/projects/model_router/src/budget_router/"
                "providers/tinker.py"
            ),
        }
        self.assertFalse(scan_for_secrets(value))

    def test_unknown_opaque_token_is_redacted_before_scan(self) -> None:
        opaque = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

        self.assertTrue(scan_for_secrets(opaque))
        result = redact({"tool_output": f"fixture token: {opaque}"})

        self.assertNotIn(opaque, result["tool_output"])
        self.assertIn(REDACTED, result["tool_output"])
        self.assertFalse(scan_for_secrets(result))


class TraceTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.prices = make_prices()
        self.goal = GoalContext(
            "fix",
            "repo",
            ("cheap",),
            task_id="task-1",
            constraints={"api_key": "must disappear", "max_parallel_operations": 0},
        )
        ledger = BudgetLedger(
            Budget(Decimal("1"), Decimal("1"), self.prices.snapshot_id),
            self.prices,
        )
        self.session = RouterSession(self.goal, ledger)
        self.state = RouterState(estimated_input_tokens=100)
        self.decision = await self.session.decide(self.state)
        self.event = RouterEvent(
            EventKind.MODEL,
            turn=0,
            model="cheap",
            operation=self.decision.action.operation,
            usage=TokenUsage(input_tokens=100, cached_input_tokens=50, output_tokens=10),
        )
        await self.session.observe(self.event)

    def make_trace(self) -> RouterTrace:
        return RouterTrace.from_turn(
            task_id="task-1",
            run_id="run-1",
            goal=self.goal,
            state=self.state,
            decision=self.decision,
            event=self.event,
            remaining_budget_usd=self.session.ledger.budget.remaining_usd,
            conservative_spent_usd=self.session.ledger.conservative_spent_usd,
            billed_spent_usd=self.session.ledger.billed_spent_usd,
            prices=self.prices,
        )

    async def test_trace_round_trip_and_hash(self) -> None:
        trace = self.make_trace()
        self.assertEqual(len(trace.record_hash), 64)
        self.assertEqual(trace.goal["constraints"]["api_key"], REDACTED)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            writer = TraceWriter(path)
            writer.append(trace)
            loaded = writer.read(path)
            self.assertEqual(loaded[0].record_hash, trace.record_hash)

    async def test_tampered_trace_rejected(self) -> None:
        item = self.make_trace().to_dict()
        item["turn_id"] = 4
        with self.assertRaises(ValueError):
            RouterTrace.from_dict(item)

    async def test_propensities_must_sum_to_one(self) -> None:
        item = self.make_trace().to_dict()
        item["record_hash"] = ""
        item["action_propensities"] = {
            key: 0.0 for key in item["action_propensities"]
        }
        with self.assertRaises(ValueError):
            RouterTrace.from_dict(item)

    async def test_replay_recomputes_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            TraceWriter(path).append(self.make_trace())
            report = replay(path, self.prices, hard_limit_usd=Decimal("1"))
            self.assertTrue(report.deterministic)
            self.assertEqual(report.hard_cap_violations, 0)
            self.assertEqual(report.conservative_spent_usd, Decimal("0.00012"))
