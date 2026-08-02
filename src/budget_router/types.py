from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping, Sequence

MAX_OUTPUT_TOKENS = 8_000
MAX_TURNS = 75
PARALLEL_SHARES = (Decimal("0.10"), Decimal("0.20"), Decimal("0.30"), Decimal("0.40"))


def as_usd(value: Decimal | str | int | float) -> Decimal:
    """Convert a value to an exact USD Decimal without binary-float surprises."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


class Operation(StrEnum):
    CONTINUE = "continue"
    REFLECT = "reflect"
    REPLAN = "replan"
    VERIFY = "verify"
    RESTART = "restart"
    PARALLEL = "parallel"
    STOP = "stop"


class EventKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    TEST = "test"
    PROVIDER_ERROR = "provider_error"
    OPERATION = "operation"
    TERMINAL = "terminal"


class ReasonCode(StrEnum):
    BEST_FEASIBLE_VALUE = "best_feasible_value"
    BUDGET_MASKED = "budget_masked"
    HARD_CAP_STOP = "hard_cap_stop"
    VERIFIED_SUBMIT = "verified_submit"
    LOW_VALUE_ABSTAIN = "low_value_abstain"
    SWITCH_HYSTERESIS = "switch_hysteresis"
    PARALLEL_VALUE = "parallel_value"
    PROVIDER_RECOVERY = "provider_recovery"
    TURN_LIMIT = "turn_limit"
    NO_ELIGIBLE_ACTION = "no_eligible_action"


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    role: str
    content: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class DiffSummary:
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    patch_hash: str | None = None
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ToolEvent:
    name: str
    success: bool
    duration_ms: int = 0
    summary: str = ""


@dataclass(frozen=True, slots=True)
class TestEvent:
    command: str
    passed: bool
    duration_ms: int = 0
    summary: str = ""


@dataclass(frozen=True, slots=True)
class GoalContext:
    goal: str
    repository: str
    allowed_models: tuple[str, ...]
    task_id: str = ""
    repository_metadata: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    harness_version: str = "mini-swe-agent@pinned"
    prompt_version: str = "coding-agent-v1"

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if not self.allowed_models:
            raise ValueError("allowed_models must not be empty")
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise ValueError("allowed_models must be unique")


@dataclass(frozen=True, slots=True)
class Budget:
    hard_limit_usd: Decimal
    remaining_usd: Decimal
    price_snapshot: str

    def __post_init__(self) -> None:
        hard = as_usd(self.hard_limit_usd)
        remaining = as_usd(self.remaining_usd)
        object.__setattr__(self, "hard_limit_usd", hard)
        object.__setattr__(self, "remaining_usd", remaining)
        if hard < 0 or remaining < 0:
            raise ValueError("budget amounts must be non-negative")
        if remaining > hard:
            raise ValueError("remaining_usd cannot exceed hard_limit_usd")
        if not self.price_snapshot:
            raise ValueError("price_snapshot must be named")


@dataclass(frozen=True, slots=True)
class RouterState:
    visible_transcript: tuple[TranscriptMessage, ...] = ()
    workspace: DiffSummary = field(default_factory=DiffSummary)
    tool_events: tuple[ToolEvent, ...] = ()
    test_events: tuple[TestEvent, ...] = ()
    current_model: str | None = None
    turn: int = 0
    spent_usd: Decimal = Decimal("0")
    uncertainty: float = 1.0
    estimated_input_tokens: int = 0
    verification_passed: bool = False
    terminal: bool = False
    last_switch_turn: int | None = None
    restart_count: int = 0
    parallel_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spent_usd", as_usd(self.spent_usd))
        if not 0 <= self.turn <= MAX_TURNS:
            raise ValueError(f"turn must be between 0 and {MAX_TURNS}")
        if self.spent_usd < 0:
            raise ValueError("spent_usd must be non-negative")
        if self.estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be non-negative")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RouterAction:
    operation: Operation
    model: str | None = None
    branch_share: Decimal | None = None
    max_output_tokens: int = MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if not 0 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens must be in [0, {MAX_OUTPUT_TOKENS}]")
        if self.operation is Operation.PARALLEL:
            if self.model is None:
                raise ValueError("parallel requires a model")
            share = as_usd(self.branch_share) if self.branch_share is not None else None
            object.__setattr__(self, "branch_share", share)
            if share not in PARALLEL_SHARES:
                raise ValueError(f"parallel branch_share must be one of {PARALLEL_SHARES}")
            if share is not None and Decimal("2") * share > Decimal("0.80"):
                raise ValueError("parallel must reserve at least 20% for continuation")
        elif self.branch_share is not None:
            raise ValueError("branch_share is only valid for parallel")
        if self.operation is Operation.STOP:
            if self.model is not None:
                raise ValueError("stop must not select a model")
            if self.max_output_tokens != 0:
                object.__setattr__(self, "max_output_tokens", 0)

    @property
    def key(self) -> str:
        share = f":{self.branch_share}" if self.branch_share is not None else ""
        return f"{self.operation.value}:{self.model or '-'}{share}"


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    action: RouterAction
    success_probability: float
    feasibility_probability: float
    expected_cost_to_go_usd: Decimal
    uncertainty: float
    value: float

    def __post_init__(self) -> None:
        for name in ("success_probability", "feasibility_probability", "uncertainty"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(
            self, "expected_cost_to_go_usd", as_usd(self.expected_cost_to_go_usd)
        )


@dataclass(frozen=True, slots=True)
class FeasibilityPoint:
    budget_usd: Decimal
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget_usd", as_usd(self.budget_usd))
        if self.budget_usd < 0 or not 0.0 <= self.probability <= 1.0:
            raise ValueError("invalid feasibility point")


@dataclass(frozen=True, slots=True)
class RouterDecision:
    action: RouterAction
    feasibility_curve: tuple[FeasibilityPoint, ...]
    action_estimates: tuple[ActionEstimate, ...]
    uncertainty: float
    switch_cost_usd: Decimal
    reason_codes: tuple[ReasonCode, ...]
    policy_version: str
    masked_actions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "switch_cost_usd", as_usd(self.switch_cost_usd))
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.cached_input_tokens, self.output_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input cannot exceed total input")

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    resolved: bool
    verified: bool
    submitted: bool
    abstained: bool = False
    censored: bool = False
    reason: str = ""
    grader: str = ""

    def __post_init__(self) -> None:
        if self.resolved and not self.verified:
            raise ValueError("resolved outcomes must be verified")
        if self.submitted and self.abstained:
            raise ValueError("an outcome cannot be submitted and abstained")


@dataclass(frozen=True, slots=True)
class RouterEvent:
    kind: EventKind
    turn: int
    model: str | None = None
    operation: Operation | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0
    tool_events: tuple[ToolEvent, ...] = ()
    test_events: tuple[TestEvent, ...] = ()
    error: str | None = None
    diff_hash: str | None = None
    outcome: TerminalOutcome | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.kind is EventKind.TERMINAL and self.outcome is None:
            raise ValueError("terminal events require an outcome")


def ensure_tuple(value: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(value or ())

