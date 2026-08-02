from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from .pricing import PriceSnapshot
from .types import Budget, RouterAction, TokenUsage, as_usd


class BudgetExceededError(RuntimeError):
    """Raised before an action could cross the conservative hard cap."""


class UnknownReservationError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_id: str
    amount_usd: Decimal
    action_key: str
    model: str | None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


class BudgetLedger:
    """Conservative, reservation-based USD accounting.

    The hard-cap side always uses uncached list prices. ``billed_spent_usd`` is
    maintained separately and may use provider-reported cache hits.
    """

    def __init__(self, budget: Budget, prices: PriceSnapshot) -> None:
        if budget.price_snapshot != prices.snapshot_id:
            raise ValueError(
                f"budget uses {budget.price_snapshot!r}, prices are {prices.snapshot_id!r}"
            )
        already_spent = budget.hard_limit_usd - budget.remaining_usd
        self._limit = budget.hard_limit_usd
        self._conservative_spent = already_spent
        self._billed_spent = Decimal("0")
        self._prices = prices
        self._reservations: dict[str, Reservation] = {}

    @property
    def hard_limit_usd(self) -> Decimal:
        return self._limit

    @property
    def conservative_spent_usd(self) -> Decimal:
        return self._conservative_spent

    @property
    def billed_spent_usd(self) -> Decimal:
        return self._billed_spent

    @property
    def reserved_usd(self) -> Decimal:
        return sum(
            (reservation.amount_usd for reservation in self._reservations.values()),
            Decimal("0"),
        )

    @property
    def available_usd(self) -> Decimal:
        return self._limit - self._conservative_spent - self.reserved_usd

    @property
    def prices(self) -> PriceSnapshot:
        return self._prices

    @property
    def budget(self) -> Budget:
        return Budget(
            hard_limit_usd=self._limit,
            remaining_usd=self._limit - self._conservative_spent,
            price_snapshot=self._prices.snapshot_id,
        )

    def quote_tokens(self, model: str, input_tokens: int, output_tokens: int) -> Decimal:
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("token bounds must be non-negative")
        return self._prices.price_for(model).conservative_cost(
            TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        )

    def quote_action(self, action: RouterAction, input_tokens: int) -> Decimal:
        if action.model is None or action.max_output_tokens == 0:
            return Decimal("0")
        return self.quote_tokens(action.model, input_tokens, action.max_output_tokens)

    def reserve(
        self,
        action: RouterAction,
        amount_usd: Decimal,
        *,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Reservation:
        amount = as_usd(amount_usd)
        if amount < 0:
            raise ValueError("reservation amount must be non-negative")
        if amount > self.available_usd:
            raise BudgetExceededError(
                f"need ${amount}, only ${self.available_usd} remains under the hard cap"
            )
        reservation = Reservation(
            reservation_id=uuid4().hex,
            amount_usd=amount,
            action_key=action.key,
            model=action.model,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def reserve_action(self, action: RouterAction, input_tokens: int) -> Reservation:
        amount = self.quote_action(action, input_tokens)
        return self.reserve(
            action,
            amount,
            max_input_tokens=input_tokens if action.model else None,
            max_output_tokens=action.max_output_tokens if action.model else None,
        )

    def reserve_parallel(self, action: RouterAction) -> Reservation:
        if action.branch_share is None:
            raise ValueError("parallel action has no branch share")
        amount = self.available_usd * Decimal("2") * action.branch_share
        return self.reserve(action, amount)

    def release(self, reservation_id: str) -> None:
        try:
            del self._reservations[reservation_id]
        except KeyError as exc:
            raise UnknownReservationError(reservation_id) from exc

    def commit(
        self,
        reservation_id: str,
        *,
        usage: TokenUsage | None = None,
        model: str | None = None,
        conservative_cost_usd: Decimal | None = None,
        billed_cost_usd: Decimal | None = None,
    ) -> tuple[Decimal, Decimal]:
        try:
            reservation = self._reservations[reservation_id]
        except KeyError as exc:
            raise UnknownReservationError(reservation_id) from exc

        selected_model = model or reservation.model
        if usage is not None:
            if selected_model is None:
                raise ValueError("token usage requires a model")
            if (
                reservation.max_input_tokens is not None
                and usage.input_tokens > reservation.max_input_tokens
            ):
                raise BudgetExceededError("actual input exceeded the reserved token bound")
            if (
                reservation.max_output_tokens is not None
                and usage.output_tokens > reservation.max_output_tokens
            ):
                raise BudgetExceededError("actual output exceeded the reserved token bound")
            price = self._prices.price_for(selected_model)
            conservative = price.conservative_cost(usage)
            billed = price.billed_cost(usage)
        else:
            conservative = as_usd(conservative_cost_usd or Decimal("0"))
            billed = as_usd(
                billed_cost_usd if billed_cost_usd is not None else conservative
            )

        if conservative > reservation.amount_usd:
            raise BudgetExceededError(
                f"charge ${conservative} exceeded reservation ${reservation.amount_usd}"
            )
        if self._conservative_spent + conservative > self._limit:
            raise BudgetExceededError("charge would exceed the hard budget cap")
        del self._reservations[reservation_id]
        self._conservative_spent += conservative
        self._billed_spent += billed
        return conservative, billed

    def assert_invariants(self) -> None:
        if self._conservative_spent < 0 or self._billed_spent < 0:
            raise AssertionError("negative spend")
        if self._conservative_spent + self.reserved_usd > self._limit:
            raise AssertionError("hard cap violated")
