from decimal import Decimal
from unittest import TestCase

from budget_router.ledger import BudgetExceededError, BudgetLedger
from budget_router.pricing import ModelPrice
from budget_router.types import Budget, Operation, RouterAction, TokenUsage

from tests.helpers import make_prices


class PricingTests(TestCase):
    def test_conservative_ignores_cache_but_billed_uses_it(self) -> None:
        price = ModelPrice("m", Decimal("1"), Decimal("2"), Decimal("0.2"))
        usage = TokenUsage(
            input_tokens=1_000_000,
            cached_input_tokens=500_000,
            output_tokens=500_000,
        )
        self.assertEqual(price.conservative_cost(usage), Decimal("2"))
        self.assertEqual(price.billed_cost(usage), Decimal("1.6"))

    def test_invalid_cache_price_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelPrice("m", Decimal("1"), Decimal("2"), Decimal("1.1"))


class LedgerTests(TestCase):
    def setUp(self) -> None:
        self.prices = make_prices()

    def ledger(self, amount: str = "1") -> BudgetLedger:
        return BudgetLedger(
            Budget(Decimal(amount), Decimal(amount), self.prices.snapshot_id),
            self.prices,
        )

    def test_snapshot_must_match(self) -> None:
        with self.assertRaises(ValueError):
            BudgetLedger(Budget(Decimal("1"), Decimal("1"), "wrong"), self.prices)

    def test_reservation_and_settlement(self) -> None:
        ledger = self.ledger()
        action = RouterAction(Operation.CONTINUE, "cheap", max_output_tokens=1_000)
        reservation = ledger.reserve_action(action, input_tokens=2_000)
        self.assertEqual(reservation.amount_usd, Decimal("0.004"))
        conservative, billed = ledger.commit(
            reservation.reservation_id,
            usage=TokenUsage(
                input_tokens=2_000,
                cached_input_tokens=1_000,
                output_tokens=500,
            ),
        )
        self.assertEqual(conservative, Decimal("0.003"))
        self.assertEqual(billed, Decimal("0.0022"))
        self.assertEqual(ledger.available_usd, Decimal("0.997"))
        ledger.assert_invariants()

    def test_cannot_reserve_past_hard_cap(self) -> None:
        ledger = self.ledger("0.001")
        action = RouterAction(Operation.CONTINUE, "cheap", max_output_tokens=1_000)
        with self.assertRaises(BudgetExceededError):
            ledger.reserve_action(action, input_tokens=0)

    def test_output_beyond_bound_is_rejected(self) -> None:
        ledger = self.ledger()
        action = RouterAction(Operation.CONTINUE, "cheap", max_output_tokens=100)
        reservation = ledger.reserve_action(action, input_tokens=100)
        with self.assertRaises(BudgetExceededError):
            ledger.commit(
                reservation.reservation_id,
                usage=TokenUsage(input_tokens=100, output_tokens=101),
            )

    def test_parallel_reserves_equal_branch_shares(self) -> None:
        ledger = self.ledger()
        action = RouterAction(
            Operation.PARALLEL,
            "cheap",
            branch_share=Decimal("0.40"),
        )
        reservation = ledger.reserve_parallel(action)
        self.assertEqual(reservation.amount_usd, Decimal("0.80"))
        self.assertEqual(ledger.available_usd, Decimal("0.20"))
        ledger.release(reservation.reservation_id)
        self.assertEqual(ledger.available_usd, Decimal("1"))

    def test_budget_validates_remaining(self) -> None:
        with self.assertRaises(ValueError):
            Budget(Decimal("1"), Decimal("2"), "snapshot")
