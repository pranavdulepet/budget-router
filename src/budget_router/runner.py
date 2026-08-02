from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Protocol

from .parallel import BranchFunction, ParallelBranchExecutor
from .providers import ModelProvider, ModelRequest, ProviderError
from .session import RouterSession
from .trace import RouterTrace, TraceWriter
from .types import (
    EventKind,
    GoalContext,
    Operation,
    RouterAction,
    RouterEvent,
    RouterState,
    TerminalOutcome,
    TestEvent,
    TokenUsage,
    TranscriptMessage,
)
from .workspace import IsolatedWorkspacePool


class CodingHarness(Protocol):
    async def execute(self, action: RouterAction, state: RouterState) -> RouterEvent: ...


@dataclass(slots=True)
class ParallelHarnessHandler:
    executor: ParallelBranchExecutor
    run_branch: BranchFunction
    remaining_budget: Callable[[], Decimal]
    max_turns: int = 5

    async def execute_action(
        self, action: RouterAction, state: RouterState
    ) -> RouterEvent:
        result = await self.executor.execute(
            action,
            remaining_budget_usd=self.remaining_budget(),
            run_branch=self.run_branch,
            max_turns=self.max_turns,
        )
        usage = result.branches[0].usage + result.branches[1].usage
        return RouterEvent(
            kind=EventKind.OPERATION,
            turn=state.turn,
            model=action.model,
            operation=Operation.PARALLEL,
            usage=usage,
            diff_hash=result.selected.diff_hash,
            error=result.selected.error,
            metadata={
                "selected_branch": result.selected.name,
                "per_branch_budget_usd": str(result.per_branch_budget_usd),
                "branches": [
                    {
                        "name": branch.name,
                        "turns": branch.turns,
                        "cost_usd": str(branch.conservative_cost_usd),
                        "visible_tests_passed": branch.visible_tests_passed,
                        "value_estimate": branch.value_estimate,
                        "diff_hash": branch.diff_hash,
                        "error": branch.error,
                    }
                    for branch in result.branches
                ],
            },
        )


@dataclass(slots=True)
class ProviderHarness:
    """Reference operation executor for provider-backed turns and visible tests."""

    provider: ModelProvider
    goal: GoalContext
    verify_commands: tuple[tuple[str, ...], ...] = ()
    workspace: Path | None = None
    workspace_pool: IsolatedWorkspacePool | None = None
    parallel_handler: object | None = None
    temperature: float = 0.7
    seed: int = 0

    async def _verify(self, action: RouterAction, state: RouterState) -> RouterEvent:
        tests: list[TestEvent] = []
        for command in self.verify_commands:
            started = time.monotonic()
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await process.communicate()
            summary = output.decode("utf-8", errors="replace")[-2_000:]
            tests.append(
                TestEvent(
                    command=shlex.join(command),
                    passed=process.returncode == 0,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    summary=summary,
                )
            )
        if not tests:
            tests.append(
                TestEvent(
                    command="<no verifier configured>",
                    passed=False,
                    summary="No visible deterministic verifier was configured.",
                )
            )
        metadata: dict[str, object] = {}
        usage = TokenUsage()
        latency_ms = sum(test.duration_ms for test in tests)
        error = None
        if action.model is not None:
            summary = "\n\n".join(
                f"$ {test.command}\npassed={test.passed}\n{test.summary}"
                for test in tests
            )
            try:
                interpretation = await self.provider.generate(
                    ModelRequest(
                        model=action.model,
                        messages=state.visible_transcript
                        + (
                            TranscriptMessage(
                                "user",
                                "Interpret these visible verifier results without editing "
                                "or using tools:\n" + summary,
                            ),
                        ),
                        operation=Operation.VERIFY,
                        max_output_tokens=action.max_output_tokens,
                        temperature=self.temperature,
                        seed=self.seed * 1_000 + state.turn,
                    )
                )
                usage = interpretation.usage
                latency_ms += interpretation.latency_ms
                metadata["visible_content"] = interpretation.content
                metadata["provider"] = dict(interpretation.provider_metadata)
                metadata["seed"] = self.seed * 1_000 + state.turn
                metadata["temperature"] = self.temperature
            except ProviderError as exc:
                error = str(exc)
        return RouterEvent(
            kind=EventKind.TEST,
            turn=state.turn,
            model=action.model,
            operation=Operation.VERIFY,
            usage=usage,
            latency_ms=latency_ms,
            test_events=tuple(tests),
            error=error,
            metadata=metadata,
        )

    async def execute(self, action: RouterAction, state: RouterState) -> RouterEvent:
        if action.operation is Operation.VERIFY:
            return await self._verify(action, state)
        if action.operation is Operation.RESTART:
            error = None
            if self.workspace_pool is not None:
                self.workspace = self.workspace_pool.restart()
            else:
                error = "restart requested without an isolated workspace pool"
            return RouterEvent(
                kind=EventKind.OPERATION,
                turn=state.turn,
                model=action.model,
                operation=action.operation,
                error=error,
            )
        if action.operation is Operation.PARALLEL:
            if self.parallel_handler is None:
                return RouterEvent(
                    kind=EventKind.OPERATION,
                    turn=state.turn,
                    model=action.model,
                    operation=action.operation,
                    error=(
                        "parallel requested without a branch handler; configure "
                        "ParallelHarnessHandler"
                    ),
                )
            execute = getattr(self.parallel_handler, "execute_action", None)
            if execute is None:
                raise TypeError("parallel_handler must provide execute_action")
            return await execute(action, state)
        if action.operation is Operation.STOP:
            return RouterEvent(
                kind=EventKind.TERMINAL,
                turn=state.turn,
                operation=Operation.STOP,
                outcome=TerminalOutcome(
                    resolved=state.verification_passed,
                    verified=state.verification_passed,
                    submitted=state.verification_passed,
                    abstained=not state.verification_passed,
                    reason="verified_submit" if state.verification_passed else "policy_abstain",
                ),
            )
        assert action.model is not None
        instruction = {
            Operation.CONTINUE: "Continue solving the goal using visible tools and workspace.",
            Operation.REFLECT: (
                "Diagnose the current approach without editing files or using tools. "
                "Return only a concise visible diagnosis."
            ),
            Operation.REPLAN: (
                "Return a structured replacement plan. Preserve the current workspace."
            ),
        }.get(action.operation, "Continue.")
        messages = state.visible_transcript + (
            TranscriptMessage("user", instruction),
        )
        try:
            response = await self.provider.generate(
                ModelRequest(
                    model=action.model,
                    messages=messages,
                    operation=action.operation,
                    max_output_tokens=action.max_output_tokens,
                    temperature=self.temperature,
                    seed=self.seed * 1_000 + state.turn,
                )
            )
        except ProviderError as exc:
            return RouterEvent(
                kind=EventKind.PROVIDER_ERROR,
                turn=state.turn,
                model=action.model,
                operation=action.operation,
                error=str(exc),
            )
        return RouterEvent(
            kind=EventKind.MODEL,
            turn=state.turn,
            model=response.model,
            operation=action.operation,
            usage=response.usage,
            latency_ms=response.latency_ms,
            metadata={
                "visible_content": response.content,
                "provider": dict(response.provider_metadata),
                "seed": self.seed * 1_000 + state.turn,
                "temperature": self.temperature,
            },
        )


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    outcome: TerminalOutcome
    final_state: RouterState
    turns: int


class ReferenceEpisodeRunner:
    def __init__(
        self,
        session: RouterSession,
        harness: CodingHarness,
        *,
        trace_writer: TraceWriter | None = None,
        run_id: str = "run",
    ) -> None:
        self.session = session
        self.harness = harness
        self.trace_writer = trace_writer
        self.run_id = run_id

    async def run(self, state: RouterState) -> EpisodeResult:
        while not state.terminal and state.turn < 75:
            before = state
            decision = await self.session.decide(before)
            event = await self.harness.execute(decision.action, before)
            state = await self.session.observe(event)
            if self.trace_writer is not None:
                self.trace_writer.append(
                    RouterTrace.from_turn(
                        task_id=self.session.goal.task_id,
                        run_id=self.run_id,
                        goal=self.session.goal,
                        state=before,
                        decision=decision,
                        event=event,
                        remaining_budget_usd=self.session.ledger.budget.remaining_usd,
                        conservative_spent_usd=self.session.ledger.conservative_spent_usd,
                        billed_spent_usd=self.session.ledger.billed_spent_usd,
                        prices=self.session.ledger.prices,
                    )
                )
        outcome = (
            event.outcome
            if "event" in locals() and event.outcome is not None
            else TerminalOutcome(
                resolved=False,
                verified=False,
                submitted=False,
                abstained=True,
                censored=state.turn >= 75,
                reason="turn_limit",
            )
        )
        return EpisodeResult(outcome=outcome, final_state=state, turns=state.turn)


@dataclass(frozen=True, slots=True)
class MiniSweAgentCommand:
    config: Path
    output: Path
    subset: str = "verified"
    split: str = "test"
    workers: int = 1
    executable: str = "mini-extra"

    def argv(self) -> tuple[str, ...]:
        return (
            self.executable,
            "swebench",
            "--config",
            str(self.config),
            "--subset",
            self.subset,
            "--split",
            self.split,
            "--workers",
            str(self.workers),
            "--output",
            str(self.output),
        )


async def run_mini_swe_agent(
    command: MiniSweAgentCommand,
    *,
    execute: bool = False,
) -> dict[str, object]:
    """Pinned mini-swe-agent v2 batch boundary. Execution is opt-in."""
    if not execute:
        return {"executed": False, "argv": list(command.argv())}
    command.output.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *command.argv(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    result = {
        "executed": True,
        "returncode": process.returncode,
        "output_tail": output.decode("utf-8", errors="replace")[-4_000:],
    }
    (command.output / "budget_router_invocation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(f"mini-swe-agent exited with {process.returncode}")
    return result
