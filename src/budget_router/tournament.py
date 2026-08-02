from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    name: str
    normalized_auc: float
    integrated_brier: float
    false_feasible_rate: float
    hard_cap_violations: int
    auc_interval: tuple[float, float]
    routing_latency_ms: float
    routing_cost_usd: float
    complexity_rank: int
    is_fixed_baseline: bool = False

    def __post_init__(self) -> None:
        if self.auc_interval[0] > self.auc_interval[1]:
            raise ValueError("invalid AUC interval")


@dataclass(frozen=True, slots=True)
class TournamentRules:
    feasibility_threshold: float = 0.80
    maximum_false_feasible_rate: float = 0.20
    require_strict_brier_improvement: bool = True


@dataclass(frozen=True, slots=True)
class TournamentResult:
    selected: PolicyEvaluation
    eligible: tuple[str, ...]
    excluded: dict[str, str]
    negative_result: bool
    reason: str


def _intervals_overlap(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def select_reference_router(
    evaluations: Sequence[PolicyEvaluation],
    *,
    budget_only_integrated_brier: float,
    rules: TournamentRules | None = None,
) -> TournamentResult:
    if not evaluations:
        raise ValueError("evaluations must not be empty")
    rules = rules or TournamentRules()
    eligible: list[PolicyEvaluation] = []
    excluded: dict[str, str] = {}
    for result in evaluations:
        if result.hard_cap_violations:
            excluded[result.name] = "hard_cap_violation"
        elif result.false_feasible_rate > rules.maximum_false_feasible_rate:
            excluded[result.name] = "risk_control_gate"
        elif (
            result.integrated_brier >= budget_only_integrated_brier
            if rules.require_strict_brier_improvement
            else result.integrated_brier > budget_only_integrated_brier
        ):
            excluded[result.name] = "brier_gate"
        else:
            eligible.append(result)
    if not eligible:
        fixed = [
            item
            for item in evaluations
            if item.is_fixed_baseline and item.hard_cap_violations == 0
        ]
        if not fixed:
            raise RuntimeError("no eligible policy and no fixed-model fallback")
        selected = max(fixed, key=lambda item: item.normalized_auc)
        return TournamentResult(
            selected=selected,
            eligible=(),
            excluded=excluded,
            negative_result=True,
            reason="all learned policies failed preregistered gates; fixed baseline retained",
        )

    top = max(eligible, key=lambda item: item.normalized_auc)
    tied = [
        item
        for item in eligible
        if _intervals_overlap(item.auc_interval, top.auc_interval)
    ]
    selected = min(
        tied,
        key=lambda item: (
            item.complexity_rank,
            item.routing_latency_ms,
            item.routing_cost_usd,
            item.name,
        ),
    )
    fixed = [item for item in eligible if item.is_fixed_baseline]
    best_fixed = max(fixed, key=lambda item: item.normalized_auc) if fixed else None
    learned_beats_fixed = (
        best_fixed is None
        or (
            not selected.is_fixed_baseline
            and selected.normalized_auc > best_fixed.normalized_auc
        )
    )
    negative = not learned_beats_fixed
    if negative and best_fixed is not None:
        selected = best_fixed
        reason = "no learned router beat the best fixed-model baseline"
    elif len(tied) > 1:
        reason = "bootstrap intervals overlapped; selected simpler lower-overhead policy"
    else:
        reason = "highest normalized held-out success-versus-budget AUC"
    return TournamentResult(
        selected=selected,
        eligible=tuple(item.name for item in eligible),
        excluded=excluded,
        negative_result=negative,
        reason=reason,
    )
