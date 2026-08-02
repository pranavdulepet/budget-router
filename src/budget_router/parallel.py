from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Awaitable, Callable

from .types import Operation, RouterAction, TokenUsage
from .workspace import IsolatedWorkspacePool


@dataclass(frozen=True, slots=True)
class BranchResult:
    name: str
    workspace: Path
    turns: int
    conservative_cost_usd: Decimal
    visible_tests_passed: bool
    value_estimate: float
    diff_hash: str | None = None
    error: str | None = None
    usage: TokenUsage = TokenUsage()

    def __post_init__(self) -> None:
        if not 0 <= self.turns <= 5:
            raise ValueError("parallel branches are limited to five turns")
        if self.conservative_cost_usd < 0:
            raise ValueError("branch cost must be non-negative")


BranchFunction = Callable[
    [str, Path, Decimal, int],
    Awaitable[BranchResult],
]


@dataclass(frozen=True, slots=True)
class ParallelResult:
    selected: BranchResult
    branches: tuple[BranchResult, BranchResult]
    per_branch_budget_usd: Decimal


class ParallelBranchExecutor:
    """Execute two equal, isolated branches and select by visible evidence."""

    def __init__(self, workspace_pool: IsolatedWorkspacePool) -> None:
        self.workspace_pool = workspace_pool

    async def execute(
        self,
        action: RouterAction,
        *,
        remaining_budget_usd: Decimal,
        run_branch: BranchFunction,
        max_turns: int = 5,
    ) -> ParallelResult:
        if action.operation is not Operation.PARALLEL or action.branch_share is None:
            raise ValueError("ParallelBranchExecutor requires a parallel action")
        if not 1 <= max_turns <= 5:
            raise ValueError("max_turns must be between one and five")
        per_branch = remaining_budget_usd * action.branch_share
        if remaining_budget_usd - Decimal("2") * per_branch < (
            remaining_budget_usd * Decimal("0.20")
        ):
            raise ValueError("parallel action did not reserve 20% for continuation")
        branches = self.workspace_pool.fork(("a", "b"))
        left, right = await asyncio.gather(
            run_branch("a", branches["a"], per_branch, max_turns),
            run_branch("b", branches["b"], per_branch, max_turns),
        )
        for result in (left, right):
            if result.conservative_cost_usd > per_branch:
                raise RuntimeError(f"branch {result.name} exceeded its sub-budget")
        selected = max(
            (left, right),
            key=lambda result: (
                result.visible_tests_passed,
                result.error is None,
                result.value_estimate,
                -result.turns,
                result.name,
            ),
        )
        self.workspace_pool.select(selected.name)
        return ParallelResult(
            selected=selected,
            branches=(left, right),
            per_branch_budget_usd=per_branch,
        )
