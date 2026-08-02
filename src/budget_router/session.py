from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from .ledger import BudgetLedger, Reservation
from .policy import FactorizedRouterPolicy
from .types import (
    EventKind,
    GoalContext,
    Operation,
    RouterDecision,
    RouterEvent,
    RouterState,
    TranscriptMessage,
)


class PendingDecisionError(RuntimeError):
    pass


class RouterSession:
    """Stateful async SDK boundary.

    ``decide`` reserves the worst-case list-price cost. ``observe`` settles
    actual usage and releases unused reservation. Learning is deliberately
    absent unless an explicit terminal-only adapter is supplied.
    """

    def __init__(
        self,
        goal: GoalContext,
        ledger: BudgetLedger,
        *,
        policy: FactorizedRouterPolicy | None = None,
        online_adapter: object | None = None,
        enable_online_learning: bool = False,
    ) -> None:
        if enable_online_learning and online_adapter is None:
            raise ValueError("online learning requires an explicit adapter")
        if online_adapter is not None and not enable_online_learning:
            raise ValueError("adapter supplied while online learning is disabled")
        self.goal = goal
        self.ledger = ledger
        self.policy = policy or FactorizedRouterPolicy()
        self.online_adapter = online_adapter
        self.enable_online_learning = enable_online_learning
        self._pending: Reservation | None = None
        self._decision: RouterDecision | None = None
        self._state: RouterState | None = None

    @property
    def pending_decision(self) -> RouterDecision | None:
        return self._decision

    async def decide(self, state: RouterState) -> RouterDecision:
        if self._pending is not None:
            raise PendingDecisionError("observe or cancel the prior decision before deciding again")
        if state.spent_usd != self.ledger.conservative_spent_usd:
            raise ValueError(
                "state spend does not match the conservative ledger: "
                f"{state.spent_usd} != {self.ledger.conservative_spent_usd}"
            )
        decision = self.policy.decide(self.goal, state, self.ledger)
        if decision.action.operation is Operation.PARALLEL:
            reservation = self.ledger.reserve_parallel(decision.action)
        else:
            reservation = self.ledger.reserve_action(
                decision.action, state.estimated_input_tokens
            )
        self._pending = reservation
        self._decision = decision
        self._state = state
        self.ledger.assert_invariants()
        return decision

    async def cancel_pending(self) -> None:
        if self._pending is not None:
            self.ledger.release(self._pending.reservation_id)
        self._pending = None
        self._decision = None
        self._state = None

    async def observe(self, event: RouterEvent) -> RouterState:
        if self._pending is None or self._decision is None or self._state is None:
            raise PendingDecisionError("observe requires a pending decision")
        action = self._decision.action
        if event.operation is not None and event.operation is not action.operation:
            raise ValueError("event operation does not match selected action")
        if event.model is not None and action.model is not None and event.model != action.model:
            raise ValueError("event model does not match selected model")

        if action.model is not None or event.usage.input_tokens or event.usage.output_tokens:
            self.ledger.commit(
                self._pending.reservation_id,
                usage=event.usage,
                model=event.model or action.model,
            )
        else:
            self.ledger.commit(
                self._pending.reservation_id,
                conservative_cost_usd=Decimal("0"),
            )

        previous = self._state
        new_model = event.model or action.model or previous.current_model
        model_selected_or_switched = (
            new_model is not None
            and new_model != previous.current_model
        )
        terminal = event.kind is EventKind.TERMINAL
        event_verification_passed = bool(event.test_events) and all(
            test.passed for test in event.test_events
        )
        verification_passed = previous.verification_passed or event_verification_passed
        if event.outcome is not None:
            verification_passed = event.outcome.verified
        transcript = previous.visible_transcript
        visible_content = event.metadata.get("visible_content")
        if isinstance(visible_content, str) and visible_content:
            transcript = transcript + (TranscriptMessage("assistant", visible_content),)
        if action.operation is Operation.RESTART:
            transcript = (TranscriptMessage("user", self.goal.goal),)
        next_state = replace(
            previous,
            visible_transcript=transcript,
            tool_events=previous.tool_events + event.tool_events,
            test_events=previous.test_events + event.test_events,
            current_model=new_model,
            turn=min(75, max(previous.turn + 1, event.turn + 1)),
            spent_usd=self.ledger.conservative_spent_usd,
            uncertainty=max(0.0, previous.uncertainty * 0.95),
            verification_passed=verification_passed,
            terminal=terminal,
            last_switch_turn=(
                previous.turn
                if model_selected_or_switched
                else previous.last_switch_turn
            ),
            restart_count=previous.restart_count
            + int(action.operation is Operation.RESTART),
            parallel_count=previous.parallel_count
            + int(action.operation is Operation.PARALLEL),
        )

        if self.enable_online_learning and event.kind is EventKind.TERMINAL:
            update = getattr(self.online_adapter, "update_terminal", None)
            if update is None:
                raise TypeError("online adapter must provide update_terminal")
            result = update(self.goal, previous, action, event.outcome)
            if hasattr(result, "__await__"):
                await result

        self._pending = None
        self._decision = None
        self._state = None
        self.ledger.assert_invariants()
        return next_state
