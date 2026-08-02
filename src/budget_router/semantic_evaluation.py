"""Held-out evaluation for provider-neutral semantic routers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .semantic import SemanticRouter
from .serialization import stable_hash


@dataclass(frozen=True, slots=True)
class SemanticEvaluationOutcome:
    """One held-out request-model outcome.

    Evaluation rows are intentionally separate from ``SemanticOutcome`` so a
    held-out label cannot be passed to ``train_semantic_router`` by accident.
    """

    request_id: str
    text: str
    model: str
    acceptable: bool
    cost_usd: float | None = None
    group: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.text.strip() or not self.model:
            raise ValueError("evaluation outcomes require request_id, text, and model")
        if self.cost_usd is not None and (
            not math.isfinite(self.cost_usd) or self.cost_usd < 0
        ):
            raise ValueError("evaluation cost must be finite and non-negative")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError(
                "evaluation rows must provide both input_tokens and output_tokens"
            )
        if (
            self.input_tokens is not None
            and min(self.input_tokens, self.output_tokens or 0) < 0
        ):
            raise ValueError("evaluation token estimates must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticEvaluationOutcome:
        acceptable = value.get("acceptable")
        if acceptable is None and value.get("score") is not None:
            acceptable = float(value["score"]) >= float(
                value.get("acceptable_threshold", 0.5)
            )
        if acceptable is None:
            raise ValueError("evaluation outcome needs acceptable or score")
        return cls(
            request_id=str(value["request_id"]),
            text=str(value.get("text", value.get("prompt", ""))),
            model=str(value["model"]),
            acceptable=bool(acceptable),
            cost_usd=(
                float(value["cost_usd"])
                if value.get("cost_usd") is not None
                else None
            ),
            group=str(value.get("group", "")),
            input_tokens=(
                int(value["input_tokens"])
                if value.get("input_tokens") is not None
                else None
            ),
            output_tokens=(
                int(value["output_tokens"])
                if value.get("output_tokens") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "model": self.model,
            "acceptable": self.acceptable,
            "cost_usd": self.cost_usd,
            "group": self.group,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _policy_metrics(
    selected: Sequence[str],
    requests: Sequence[Mapping[str, SemanticEvaluationOutcome]],
    *,
    request_costs: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    successes: list[float] = []
    costs: list[float] = []
    groups: list[str] = []
    route_counts: dict[str, int] = {}
    for model, request, costs_by_model in zip(
        selected, requests, request_costs, strict=True
    ):
        row = request[model]
        successes.append(float(row.acceptable))
        costs.append(float(costs_by_model[model]))
        groups.append(row.group)
        route_counts[model] = route_counts.get(model, 0) + 1
    group_rates = {
        group: sum(
            success
            for success, row_group in zip(successes, groups, strict=True)
            if row_group == group
        )
        / sum(row_group == group for row_group in groups)
        for group in sorted(set(groups))
        if group
    }
    return {
        "requests": len(requests),
        "acceptable_count": int(sum(successes)),
        "success_rate": sum(successes) / len(successes),
        "macro_group_success_rate": (
            sum(group_rates.values()) / len(group_rates) if group_rates else None
        ),
        "group_success_rate": group_rates,
        "total_cost_usd": sum(costs),
        "mean_cost_usd": sum(costs) / len(costs),
        "route_counts": route_counts,
    }


def _policy_series(
    selected: Sequence[str],
    requests: Sequence[Mapping[str, SemanticEvaluationOutcome]],
    *,
    request_costs: Sequence[Mapping[str, float]],
) -> tuple[list[float], list[float]]:
    successes: list[float] = []
    costs: list[float] = []
    for model, request, costs_by_model in zip(
        selected, requests, request_costs, strict=True
    ):
        row = request[model]
        successes.append(float(row.acceptable))
        costs.append(float(costs_by_model[model]))
    return successes, costs


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _paired_bootstrap(
    routed: tuple[Sequence[float], Sequence[float]],
    fixed: tuple[Sequence[float], Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    routed_success, routed_cost = routed
    fixed_success, fixed_cost = fixed
    if len(routed_success) != len(fixed_success):
        raise ValueError("paired policies have different request counts")
    randomizer = random.Random(seed)
    quality: list[float] = []
    cost_saving: list[float] = []
    for _ in range(samples):
        indices = [
            randomizer.randrange(len(routed_success))
            for _ in range(len(routed_success))
        ]
        quality.append(
            sum(
                routed_success[index] - fixed_success[index]
                for index in indices
            )
            / len(indices)
        )
        fixed_total = sum(fixed_cost[index] for index in indices)
        routed_total = sum(routed_cost[index] for index in indices)
        if fixed_total > 0:
            cost_saving.append((fixed_total - routed_total) / fixed_total)
    return {
        "samples": samples,
        "seed": seed,
        "quality_difference_95ci": [
            _percentile(quality, 0.025),
            _percentile(quality, 0.975),
        ],
        "cost_saving_fraction_95ci": (
            [
                _percentile(cost_saving, 0.025),
                _percentile(cost_saving, 0.975),
            ]
            if cost_saving
            else None
        ),
    }


def evaluate_semantic_router(
    router: SemanticRouter,
    outcomes: Iterable[SemanticEvaluationOutcome],
    *,
    modes: Sequence[str] = ("quality", "balanced"),
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    max_quality_drop: float = 0.05,
    bootstrap_samples: int = 5_000,
    bootstrap_seed: int = 0,
    allow_ungated_policy: bool = False,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare routed policies with every fixed model on an untouched matrix."""

    if not modes or any(mode not in {"quality", "balanced", "cost"} for mode in modes):
        raise ValueError("modes must contain quality, balanced, or cost")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if (input_tokens is None) != (output_tokens is None):
        raise ValueError(
            "global evaluation settings require both input_tokens and output_tokens"
        )
    if input_tokens is not None and min(input_tokens, output_tokens or 0) < 0:
        raise ValueError("global evaluation token estimates must be non-negative")
    rows = list(outcomes)
    grouped: dict[str, dict[str, SemanticEvaluationOutcome]] = {}
    order: list[str] = []
    for row in rows:
        if row.model not in router.models:
            raise ValueError(f"evaluation row uses unknown model {row.model!r}")
        if row.request_id not in grouped:
            grouped[row.request_id] = {}
            order.append(row.request_id)
        request = grouped[row.request_id]
        if row.model in request:
            raise ValueError(
                f"duplicate evaluation outcome: {(row.request_id, row.model)}"
            )
        request[row.model] = row
    if not grouped:
        raise ValueError("evaluation requires at least one request")
    requests = [grouped[request_id] for request_id in order]
    expected_models = set(router.models)
    for request_id, request in zip(order, requests, strict=True):
        if set(request) != expected_models:
            raise ValueError(f"incomplete model matrix for request {request_id!r}")
        if len({row.text for row in request.values()}) != 1:
            raise ValueError(f"model rows disagree on text for request {request_id!r}")
        if len({row.group for row in request.values()}) != 1:
            raise ValueError(f"model rows disagree on group for request {request_id!r}")
        if len(
            {
                (row.input_tokens, row.output_tokens)
                for row in request.values()
            }
        ) != 1:
            raise ValueError(
                f"model rows disagree on token estimates for request {request_id!r}"
            )

    uses_token_pricing = any(
        card.input_cost_per_million_usd is not None
        and card.output_cost_per_million_usd is not None
        for card in router.cards.values()
    )
    routing_tokens: list[tuple[int, int]] = []
    for request_id, request in zip(order, requests, strict=True):
        first = next(iter(request.values()))
        active_input = input_tokens if input_tokens is not None else first.input_tokens
        active_output = (
            output_tokens if output_tokens is not None else first.output_tokens
        )
        if active_input is None or active_output is None:
            if uses_token_pricing:
                raise ValueError(
                    "token-priced model cards require input_tokens and "
                    f"output_tokens for evaluation request {request_id!r}"
                )
            active_input, active_output = 0, 0
        routing_tokens.append((active_input, active_output))

    request_costs: list[dict[str, float]] = []
    for request, (request_input, request_output) in zip(
        requests, routing_tokens, strict=True
    ):
        costs_by_model: dict[str, float] = {}
        for model, row in request.items():
            if row.cost_usd is not None:
                costs_by_model[model] = row.cost_usd
                continue
            try:
                costs_by_model[model] = router.cards[model].estimate_cost(
                    request_input,
                    request_output,
                )
            except ValueError:
                costs_by_model[model] = router.expected_costs[model]
        request_costs.append(costs_by_model)

    policies: dict[str, Any] = {}
    selections: dict[str, list[str]] = {}
    route_decisions: dict[str, list[Any]] = {}
    for model in router.models:
        policy = f"fixed:{model}"
        selections[policy] = [model] * len(requests)
        policies[policy] = _policy_metrics(
            selections[policy],
            requests,
            request_costs=request_costs,
        )
    for mode in dict.fromkeys(modes):
        policy = f"router:{mode}"
        selections[policy] = []
        route_decisions[policy] = []
        for request, (request_input, request_output) in zip(
            requests, routing_tokens, strict=True
        ):
            decision = router.route(
                next(iter(request.values())).text,
                mode=mode,
                input_tokens=request_input,
                output_tokens=request_output,
                max_quality_drop=max_quality_drop,
                allow_ungated_policy=allow_ungated_policy,
            )
            route_decisions[policy].append(decision)
            selections[policy].append(decision.selected_model)
        policies[policy] = _policy_metrics(
            selections[policy],
            requests,
            request_costs=request_costs,
        )
    fixed_policies = [policy for policy in policies if policy.startswith("fixed:")]
    best_fixed = max(
        fixed_policies,
        key=lambda policy: (
            (
                policies[policy]["macro_group_success_rate"]
                if policies[policy]["macro_group_success_rate"] is not None
                else policies[policy]["success_rate"]
            ),
            -policies[policy]["total_cost_usd"],
            policy,
        ),
    )
    fixed_series = _policy_series(
        selections[best_fixed],
        requests,
        request_costs=request_costs,
    )
    comparisons: dict[str, Any] = {}
    for index, policy in enumerate(
        policy for policy in policies if policy.startswith("router:")
    ):
        routed_series = _policy_series(
            selections[policy],
            requests,
            request_costs=request_costs,
        )
        fixed_total = policies[best_fixed]["total_cost_usd"]
        comparisons[policy] = {
            "quality_difference": (
                policies[policy]["success_rate"]
                - policies[best_fixed]["success_rate"]
            ),
            "cost_saving_fraction": (
                (fixed_total - policies[policy]["total_cost_usd"]) / fixed_total
                if fixed_total > 0
                else None
            ),
            "paired_bootstrap": _paired_bootstrap(
                routed_series,
                fixed_series,
                samples=bootstrap_samples,
                seed=bootstrap_seed + index,
            ),
        }
    task_decisions: list[dict[str, Any]] = []
    for index, (request_id, request) in enumerate(
        zip(order, requests, strict=True)
    ):
        first = next(iter(request.values()))
        policy_rows: dict[str, Any] = {}
        for policy, selected in selections.items():
            model = selected[index]
            outcome = request[model]
            policy_rows[policy] = {
                "selected_model": model,
                "acceptable": outcome.acceptable,
                "cost_usd": request_costs[index][model],
                **(
                    {
                        "policy_gate_passed": route_decisions[policy][
                            index
                        ].policy_gate_passed,
                        "policy_gate_scope": route_decisions[policy][
                            index
                        ].policy_gate_scope,
                        "fallback_applied": route_decisions[policy][
                            index
                        ].fallback_applied,
                    }
                    if policy in route_decisions
                    else {}
                ),
            }
        task_decisions.append(
            {
                "request_id": request_id,
                "text_hash": stable_hash({"text": first.text}),
                "group": first.group,
                "input_tokens": routing_tokens[index][0],
                "output_tokens": routing_tokens[index][1],
                "policies": policy_rows,
            }
        )
    report = {
        "schema_version": "semantic-router-evaluation-v1",
        "artifact_hash": router.artifact_hash,
        "outcome_matrix_hash": stable_hash(
            [
                row.to_dict()
                for row in sorted(
                    rows,
                    key=lambda row: (row.request_id, row.model),
                )
            ]
        ),
        "source_sha256": source_sha256,
        "evaluation_config": {
            "modes": list(dict.fromkeys(modes)),
            "global_input_tokens": input_tokens,
            "global_output_tokens": output_tokens,
            "per_request_token_estimates": input_tokens is None,
            "max_quality_drop": max_quality_drop,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "allow_ungated_policy": allow_ungated_policy,
        },
        "default_policy_gate_passed": router.default_policy_gate_passed,
        "requests": len(requests),
        "models": list(router.models),
        "best_fixed_policy": best_fixed,
        "policies": policies,
        "comparisons_vs_best_fixed": comparisons,
        "task_decisions": task_decisions,
    }
    report["report_hash"] = stable_hash(report)
    return report
