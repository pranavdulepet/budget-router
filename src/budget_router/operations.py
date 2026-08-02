from __future__ import annotations

from .types import GoalContext, Operation, RouterState


def eligible_operations(state: RouterState, goal: GoalContext) -> tuple[Operation, ...]:
    """Return operations whose semantic preconditions are visible and satisfied."""
    if state.terminal:
        return ()
    if state.verification_passed:
        return (Operation.STOP,)
    if state.turn >= 75:
        return (Operation.STOP,)

    operations: list[Operation] = [Operation.CONTINUE, Operation.REFLECT]
    if state.turn > 0:
        operations.append(Operation.REPLAN)
    if state.workspace.files_changed > 0 or state.turn > 0:
        operations.append(Operation.VERIFY)
    if state.turn > 0 and state.restart_count == 0:
        operations.append(Operation.RESTART)
    max_parallel = int(goal.constraints.get("max_parallel_operations", 1))
    if state.turn <= 69 and state.parallel_count < max_parallel:
        operations.append(Operation.PARALLEL)
    operations.append(Operation.STOP)
    return tuple(operations)


def eligible_models(
    state: RouterState,
    goal: GoalContext,
    *,
    switch_hysteresis_turns: int,
) -> tuple[tuple[str, ...], bool]:
    """Apply switch hysteresis and return ``(models, switch_was_masked)``."""
    if state.current_model is None:
        return goal.allowed_models, False
    if state.current_model not in goal.allowed_models:
        return goal.allowed_models, False
    if state.last_switch_turn is None:
        return goal.allowed_models, False
    if state.turn - state.last_switch_turn >= switch_hysteresis_turns:
        return goal.allowed_models, False
    return (state.current_model,), len(goal.allowed_models) > 1


def operation_requires_model(operation: Operation) -> bool:
    return operation not in {Operation.STOP, Operation.VERIFY}

