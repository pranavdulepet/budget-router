#!/usr/bin/env python3
"""Analyze the frozen Amendment 008 guarded agent-step experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from budget_router.guarded_agent_step import (
    GuardedAgentStepArtifact,
    GuardedAgentStepRouter,
)
from budget_router.serialization import read_jsonl

try:
    from scripts.analyze_agent_step_results import (
        _quantile,
        mcnemar_exact_two_sided,
        paired_bootstrap,
    )
except ModuleNotFoundError:
    from analyze_agent_step_results import (  # type: ignore[no-redef]
        _quantile,
        mcnemar_exact_two_sided,
        paired_bootstrap,
    )


FIXED_CHEAP = "fixed:openai-gpt-oss-20b"
FIXED_STRONG = "fixed:qwen3.6-35b"
ROUTED = "router:guarded-live-calibrated-v2"
POLICIES = (FIXED_CHEAP, FIXED_STRONG, ROUTED)
LABELS = {
    FIXED_CHEAP: "agent_step_v2_fixed_gpt_oss_20b",
    FIXED_STRONG: "agent_step_v2_fixed_qwen35",
    ROUTED: "agent_step_v2_guarded_router",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal_sum(values: Iterable[Any]) -> Decimal:
    return sum((Decimal(str(value)) for value in values), Decimal("0"))


def _repository_cluster_bootstrap(
    task_ids: Sequence[str],
    repositories: Mapping[str, str],
    routed_quality: Mapping[str, int],
    fixed_quality: Mapping[str, int],
    routed_cost: Mapping[str, float],
    fixed_cost: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Equal-weight repositories and resample repository clusters."""
    by_repository: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        by_repository[repositories[task_id]].append(task_id)
    repository_ids = sorted(by_repository)
    if not repository_ids:
        raise ValueError("repository bootstrap requires tasks")

    def summaries(selected: Sequence[str]) -> tuple[float, float]:
        quality_delta = sum(
            sum(
                routed_quality[task_id] - fixed_quality[task_id]
                for task_id in by_repository[repository]
            )
            / len(by_repository[repository])
            for repository in selected
        ) / len(selected)
        routed_mean_cost = sum(
            sum(routed_cost[task_id] for task_id in by_repository[repository])
            / len(by_repository[repository])
            for repository in selected
        ) / len(selected)
        fixed_mean_cost = sum(
            sum(fixed_cost[task_id] for task_id in by_repository[repository])
            / len(by_repository[repository])
            for repository in selected
        ) / len(selected)
        saving = (
            1 - routed_mean_cost / fixed_mean_cost if fixed_mean_cost else 0.0
        )
        return quality_delta, saving

    point_quality, point_saving = summaries(repository_ids)
    rng = random.Random(seed)
    quality_samples: list[float] = []
    cost_samples: list[float] = []
    for _ in range(samples):
        selected = [
            repository_ids[rng.randrange(len(repository_ids))]
            for _ in repository_ids
        ]
        quality, saving = summaries(selected)
        quality_samples.append(quality)
        cost_samples.append(saving)
    return {
        "method": "equal_repository_weight_cluster_bootstrap",
        "repository_count": len(repository_ids),
        "repositories": repository_ids,
        "samples": samples,
        "seed": seed,
        "point_quality_difference": point_quality,
        "quality_difference_95ci": [
            _quantile(quality_samples, 0.025),
            _quantile(quality_samples, 0.975),
        ],
        "point_cost_saving_fraction": point_saving,
        "cost_saving_fraction_95ci": [
            _quantile(cost_samples, 0.025),
            _quantile(cost_samples, 0.975),
        ],
    }


def _replay_shadow_identity(
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    task_ids: Sequence[str],
    artifact: GuardedAgentStepArtifact,
    *,
    project_root: Path,
) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    total_calls = cheap_calls = switches = 0
    reason_counts: Counter[str] = Counter()
    for task_id in task_ids:
        record = records[(task_id, FIXED_STRONG)]
        trajectory_path = Path(str(record["trajectory"]))
        if not trajectory_path.is_absolute():
            trajectory_path = project_root / trajectory_path
        messages = _load(trajectory_path)["messages"]
        router = GuardedAgentStepRouter(artifact)
        prefix: list[Mapping[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant":
                router.select(prefix)
            prefix.append(message)
        if len(router.decisions) != int(record["model_calls"]):
            raise ValueError(
                f"shadow call count mismatch for {task_id}: "
                f"{len(router.decisions)} != {record['model_calls']}"
            )
        trajectory_cheap = sum(
            decision.model_id == artifact.cheap_model
            for decision in router.decisions
        )
        reason_counts.update(decision.reason for decision in router.decisions)
        total_calls += len(router.decisions)
        cheap_calls += trajectory_cheap
        switches += router.switch_count
        trajectories.append(
            {
                "task_id": task_id,
                "calls": len(router.decisions),
                "cheap_calls": trajectory_cheap,
                "strong_calls": len(router.decisions) - trajectory_cheap,
                "switch_count": router.switch_count,
                "would_select_only_qwen": trajectory_cheap == 0,
            }
        )
    only_qwen = sum(row["would_select_only_qwen"] for row in trajectories)
    return {
        "status": "descriptive_zero_cost_prefix_replay",
        "trajectory_source": "independently_sampled_fixed_qwen_arm",
        "trajectories": len(trajectories),
        "total_calls": total_calls,
        "cheap_calls": cheap_calls,
        "strong_calls": total_calls - cheap_calls,
        "cheap_call_share": cheap_calls / total_calls if total_calls else 0.0,
        "trajectories_with_cheap_call": len(trajectories) - only_qwen,
        "trajectories_selecting_only_qwen": only_qwen,
        "switch_count": switches,
        "decision_reasons": dict(sorted(reason_counts.items())),
        "trajectory_rows": trajectories,
    }


def _active_route_audit(
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    task_ids: Sequence[str],
    *,
    project_root: Path,
    price_snapshot: Mapping[str, Any],
    cheap_model: str,
    strong_model: str,
) -> dict[str, Any]:
    position_counts: Counter[int] = Counter()
    reason_counts: Counter[str] = Counter()
    guard_counts: Counter[str] = Counter()
    task_rows: list[dict[str, Any]] = []
    cheap_input = cheap_output = 0
    total_assistant_calls = 0
    for task_id in task_ids:
        record = records[(task_id, ROUTED)]
        trajectory_path = Path(str(record["trajectory"]))
        if not trajectory_path.is_absolute():
            trajectory_path = project_root / trajectory_path
        messages = _load(trajectory_path)["messages"]
        calls = [
            message for message in messages if message.get("role") == "assistant"
        ]
        if len(calls) != int(record["model_calls"]):
            raise ValueError(f"active call count mismatch for {task_id}")
        cheap_positions: list[int] = []
        for index, message in enumerate(calls):
            extra = message.get("extra", {})
            decision = extra.get("agent_step_router")
            if not isinstance(decision, Mapping):
                raise ValueError(f"missing route metadata for {task_id} call {index}")
            model = str(extra.get("active_model"))
            reason_counts[str(decision["reason"])] += 1
            if bool(decision.get("forced")):
                guard_counts[str(decision["reason"])] += 1
            if model == cheap_model:
                cheap_positions.append(index)
                position_counts[index] += 1
                usage = extra.get("usage", {})
                cheap_input += int(usage.get("input_tokens", 0))
                cheap_output += int(usage.get("output_tokens", 0))
            elif model != strong_model:
                raise ValueError(f"unknown active model in {task_id}: {model}")
        if len(cheap_positions) != int(record["cheap_calls"]):
            raise ValueError(f"active cheap-call mismatch for {task_id}")
        total_assistant_calls += len(calls)
        task_rows.append(
            {
                "task_id": task_id,
                "calls": len(calls),
                "cheap_calls": len(cheap_positions),
                "cheap_call_positions_zero_based": cheap_positions,
                "normalized_cheap_call_positions": [
                    position / max(1, len(calls) - 1)
                    for position in cheap_positions
                ],
            }
        )
    prices = price_snapshot["models"]
    cheap_prices = prices[cheap_model]
    strong_prices = prices[strong_model]

    def token_cost(tokens_in: int, tokens_out: int, model_prices: Mapping[str, Any]) -> Decimal:
        return (
            Decimal(tokens_in)
            * Decimal(str(model_prices["input_per_million_usd"]))
            + Decimal(tokens_out)
            * Decimal(str(model_prices["output_per_million_usd"]))
        ) / Decimal("1000000")

    observed_cheap_token_cost = token_cost(cheap_input, cheap_output, cheap_prices)
    same_tokens_at_strong_price = token_cost(
        cheap_input, cheap_output, strong_prices
    )
    routed_records = [records[(task_id, ROUTED)] for task_id in task_ids]
    return {
        "trajectories": len(task_rows),
        "trajectories_with_cheap_call": sum(
            bool(row["cheap_calls"]) for row in task_rows
        ),
        "total_calls": total_assistant_calls,
        "decision_reasons": dict(sorted(reason_counts.items())),
        "forced_guard_reasons": dict(sorted(guard_counts.items())),
        "cheap_calls_by_zero_based_position": {
            str(index): count for index, count in sorted(position_counts.items())
        },
        "guard_compliant_trajectories": sum(
            bool(row.get("guard_compliant")) for row in routed_records
        ),
        "cheap_substitution_token_accounting": {
            "cheap_input_tokens": cheap_input,
            "cheap_output_tokens": cheap_output,
            "observed_cheap_call_cost_usd": str(observed_cheap_token_cost),
            "same_tokens_at_strong_list_price_usd": str(
                same_tokens_at_strong_price
            ),
            "mechanical_same_token_saving_usd": str(
                same_tokens_at_strong_price - observed_cheap_token_cost
            ),
            "caveat": (
                "Same-token repricing isolates list-price substitution only; "
                "it is not a causal counterfactual because model choice changes "
                "future tokens and trajectory length."
            ),
        },
        "task_rows": task_rows,
    }


def _policy_metrics(
    task_ids: Sequence[str],
    policy_id: str,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    resolved: set[str],
    grading_model: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [records[(task_id, policy_id)] for task_id in task_ids]
    resolved_here = sorted(resolved.intersection(task_ids))
    total_cost = _decimal_sum(row["conservative_cost_usd"] for row in rows)
    total_calls = sum(int(row["model_calls"]) for row in rows)
    cheap_calls = sum(int(row.get("cheap_calls", 0)) for row in rows)
    strong_calls = sum(int(row.get("strong_calls", 0)) for row in rows)
    return {
        "episodes": len(rows),
        "submitted_count": sum(bool(row["submitted_patch"]) for row in rows),
        "resolved_count": len(resolved_here),
        "resolution_rate": len(resolved_here) / len(rows),
        "resolved_task_ids": resolved_here,
        "total_conservative_cost_usd": str(total_cost),
        "mean_cost_per_task_usd": str(total_cost / len(rows)),
        "cost_per_resolved_task_usd": (
            str(total_cost / len(resolved_here)) if resolved_here else None
        ),
        "model_calls": total_calls,
        "cheap_calls": cheap_calls,
        "strong_calls": strong_calls,
        "cheap_call_share": cheap_calls / total_calls if total_calls else 0.0,
        "switch_count": sum(int(row.get("switch_count", 0)) for row in rows),
        "provider_failure_count": sum(
            bool(row.get("provider_failed")) for row in rows
        ),
        "structurally_valid_count": sum(
            bool(row.get("structurally_valid")) for row in rows
        ),
        "model_caused_error_count": len(
            set(map(str, grading_model.get("model_caused_error_ids", [])))
            .intersection(task_ids)
        ),
    }


def _comparison(
    task_ids: Sequence[str],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    resolved_by_policy: Mapping[str, set[str]],
    *,
    comparator: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    routed_quality = [int(task_id in resolved_by_policy[ROUTED]) for task_id in task_ids]
    fixed_quality = [
        int(task_id in resolved_by_policy[comparator]) for task_id in task_ids
    ]
    routed_cost = [
        float(records[(task_id, ROUTED)]["conservative_cost_usd"])
        for task_id in task_ids
    ]
    fixed_cost = [
        float(records[(task_id, comparator)]["conservative_cost_usd"])
        for task_id in task_ids
    ]
    wins = sum(
        routed and not fixed
        for routed, fixed in zip(routed_quality, fixed_quality, strict=True)
    )
    losses = sum(
        fixed and not routed
        for routed, fixed in zip(routed_quality, fixed_quality, strict=True)
    )
    routed_total = _decimal_sum(
        records[(task_id, ROUTED)]["conservative_cost_usd"]
        for task_id in task_ids
    )
    fixed_total = _decimal_sum(
        records[(task_id, comparator)]["conservative_cost_usd"]
        for task_id in task_ids
    )
    return {
        "comparator": comparator,
        "tasks": len(task_ids),
        "routed_minus_fixed_resolved": sum(routed_quality) - sum(fixed_quality),
        "routed_minus_fixed_resolution_rate": (
            sum(routed_quality) - sum(fixed_quality)
        )
        / len(task_ids),
        "routed_cost_saving_usd": str(fixed_total - routed_total),
        "routed_cost_saving_fraction": (
            float(1 - routed_total / fixed_total) if fixed_total else 0.0
        ),
        "mcnemar_router_wins": wins,
        "mcnemar_router_losses": losses,
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(wins, losses),
        "paired_task_bootstrap": paired_bootstrap(
            routed_quality,
            fixed_quality,
            routed_cost,
            fixed_cost,
            samples=samples,
            seed=seed,
        ),
    }


def _retrospective_oracle(
    task_ids: Sequence[str],
    candidate_policies: Sequence[str],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    resolved_by_policy: Mapping[str, set[str]],
) -> dict[str, Any]:
    selected_counts: Counter[str] = Counter()
    resolved_ids: list[str] = []
    total_cost = Decimal("0")
    task_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        successful = [
            policy_id
            for policy_id in candidate_policies
            if task_id in resolved_by_policy[policy_id]
        ]
        eligible = successful or list(candidate_policies)
        selected = min(
            eligible,
            key=lambda policy_id: (
                Decimal(
                    str(
                        records[(task_id, policy_id)][
                            "conservative_cost_usd"
                        ]
                    )
                ),
                policy_id,
            ),
        )
        cost = Decimal(
            str(records[(task_id, selected)]["conservative_cost_usd"])
        )
        resolved = bool(successful)
        selected_counts[selected] += 1
        resolved_ids.extend([task_id] if resolved else [])
        total_cost += cost
        task_rows.append(
            {
                "task_id": task_id,
                "selected_policy": selected,
                "resolved": resolved,
                "cost_usd": str(cost),
            }
        )
    return {
        "candidate_policies": list(candidate_policies),
        "resolved_count": len(resolved_ids),
        "resolution_rate": len(resolved_ids) / len(task_ids),
        "resolved_task_ids": sorted(resolved_ids),
        "total_observed_cost_usd": str(total_cost),
        "selected_policy_counts": dict(sorted(selected_counts.items())),
        "definition": (
            "Uses terminal held-out outcomes to choose the cheapest observed "
            "successful arm for each solvable task and the cheapest observed "
            "arm when none succeeded. It is an unattainable retrospective "
            "ceiling, not a deployable policy or causal cost estimate."
        ),
        "task_rows": task_rows,
    }


def analyze(
    *,
    records_path: Path,
    grading_manifest_path: Path,
    protocol_path: Path,
    artifact_path: Path,
    task_manifest_path: Path,
    price_snapshot_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    artifact_value = _load(artifact_path)
    artifact = GuardedAgentStepArtifact.from_dict(artifact_value)
    task_manifest = _load(task_manifest_path)
    grading = _load(grading_manifest_path)
    price_snapshot = _load(price_snapshot_path)
    if grading.get("complete_with_no_unclassified_errors") is not True:
        raise ValueError("official grading is incomplete or has unclassified errors")

    all_records = read_jsonl(records_path)
    heldout_records = [
        row for row in all_records if row.get("study_stage") == "heldout"
    ]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in heldout_records:
        key = (str(row["task_id"]), str(row["policy_id"]))
        if key in records:
            raise ValueError(f"duplicate held-out record: {key}")
        records[key] = row

    heldout_rows = task_manifest["heldout"]
    task_ids = [str(row["task_id"]) for row in heldout_rows]
    repositories = {
        str(row["task_id"]): str(row["repository"]) for row in heldout_rows
    }
    subgroups = {
        str(row["task_id"]): str(row["subgroup"]) for row in heldout_rows
    }
    expected = {
        (task_id, policy_id) for task_id in task_ids for policy_id in POLICIES
    }
    if len(task_ids) != int(protocol["heldout"]["tasks"]):
        raise ValueError("task manifest differs from frozen held-out count")
    if set(records) != expected:
        raise ValueError("held-out policy matrix is incomplete")
    if any(not bool(row.get("terminal_record_valid")) for row in heldout_records):
        raise ValueError("held-out matrix contains a non-terminal record")
    if any(bool(row.get("provider_failed")) for row in heldout_records):
        raise ValueError("held-out matrix contains a provider failure")
    for row in heldout_records:
        if row["router_artifact_hash"] != artifact.artifact_hash:
            raise ValueError("mixed router artifact hashes")
        if row["protocol_sha256"] != grading["protocol_sha256"]:
            raise ValueError("mixed protocol hashes")
        if row["task_manifest_hash"] != grading["task_manifest_hash"]:
            raise ValueError("mixed task manifest hashes")

    resolved_by_policy = {
        policy_id: set(
            map(str, grading["models"][LABELS[policy_id]]["resolved_ids"])
        )
        for policy_id in POLICIES
    }
    samples = int(protocol["analysis"]["paired_bootstrap_resamples"])
    seed = int(protocol["analysis"]["paired_bootstrap_seed"])

    def slice_metrics(selected: Sequence[str]) -> dict[str, Any]:
        return {
            "tasks": len(selected),
            "policies": {
                policy_id: _policy_metrics(
                    selected,
                    policy_id,
                    records,
                    resolved_by_policy[policy_id],
                    grading["models"][LABELS[policy_id]],
                )
                for policy_id in POLICIES
            },
            "router_vs_fixed_strong": _comparison(
                selected,
                records,
                resolved_by_policy,
                comparator=FIXED_STRONG,
                samples=samples,
                seed=seed,
            ),
            "router_vs_fixed_cheap": _comparison(
                selected,
                records,
                resolved_by_policy,
                comparator=FIXED_CHEAP,
                samples=samples,
                seed=seed,
            ),
        }

    full = slice_metrics(task_ids)
    subgroup_results = {
        subgroup: slice_metrics(
            [task_id for task_id in task_ids if subgroups[task_id] == subgroup]
        )
        for subgroup in sorted(set(subgroups.values()))
    }
    repository_results = {
        repository: slice_metrics(
            [
                task_id
                for task_id in task_ids
                if repositories[task_id] == repository
            ]
        )
        for repository in sorted(set(repositories.values()))
    }

    routed_quality = {
        task_id: int(task_id in resolved_by_policy[ROUTED])
        for task_id in task_ids
    }
    strong_quality = {
        task_id: int(task_id in resolved_by_policy[FIXED_STRONG])
        for task_id in task_ids
    }
    routed_cost = {
        task_id: float(records[(task_id, ROUTED)]["conservative_cost_usd"])
        for task_id in task_ids
    }
    strong_cost = {
        task_id: float(
            records[(task_id, FIXED_STRONG)]["conservative_cost_usd"]
        )
        for task_id in task_ids
    }
    repository_macro = {
        "full": _repository_cluster_bootstrap(
            task_ids,
            repositories,
            routed_quality,
            strong_quality,
            routed_cost,
            strong_cost,
            samples=samples,
            seed=seed,
        ),
        "subgroups": {
            subgroup: _repository_cluster_bootstrap(
                [
                    task_id
                    for task_id in task_ids
                    if subgroups[task_id] == subgroup
                ],
                repositories,
                routed_quality,
                strong_quality,
                routed_cost,
                strong_cost,
                samples=samples,
                seed=seed,
            )
            for subgroup in sorted(set(subgroups.values()))
        },
    }

    active_audit = _active_route_audit(
        records,
        task_ids,
        project_root=project_root,
        price_snapshot=price_snapshot,
        cheap_model=artifact.cheap_model,
        strong_model=artifact.strong_model,
    )
    shadow_audit = _replay_shadow_identity(
        records,
        task_ids,
        artifact,
        project_root=project_root,
    )
    routed_policy = full["policies"][ROUTED]
    joint_components = {
        "router_resolved_at_least_fixed_strong": (
            routed_policy["resolved_count"]
            >= full["policies"][FIXED_STRONG]["resolved_count"]
        ),
        "router_cost_below_fixed_strong": (
            Decimal(routed_policy["total_conservative_cost_usd"])
            < Decimal(
                full["policies"][FIXED_STRONG]["total_conservative_cost_usd"]
            )
        ),
        "router_cheap_call_share_at_least_0_10": (
            routed_policy["cheap_call_share"] >= 0.10
        ),
        "at_least_30_routed_trajectories_with_cheap_call": (
            active_audit["trajectories_with_cheap_call"] >= 30
        ),
    }

    fixed_oracle = _retrospective_oracle(
        task_ids,
        (FIXED_CHEAP, FIXED_STRONG),
        records,
        resolved_by_policy,
    )
    all_arm_oracle = _retrospective_oracle(
        task_ids,
        POLICIES,
        records,
        resolved_by_policy,
    )
    development_cost = _decimal_sum(
        row["conservative_cost_usd"]
        for row in all_records
        if row.get("study_stage") == "development"
    )
    heldout_cost = _decimal_sum(
        row["conservative_cost_usd"] for row in heldout_records
    )
    return {
        "schema_version": "guarded-agent-step-followup-results-v2",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "inferential_status": protocol["inferential_status"],
        "protocol_sha256": grading["protocol_sha256"],
        "task_manifest_hash": grading["task_manifest_hash"],
        "router_artifact_hash": artifact.artifact_hash,
        "matrix_audit": {
            "canonical_episodes": len(heldout_records),
            "tasks": len(task_ids),
            "policies": len(POLICIES),
            "terminal_records": sum(
                bool(row["terminal_record_valid"]) for row in heldout_records
            ),
            "provider_failures": sum(
                bool(row["provider_failed"]) for row in heldout_records
            ),
            "official_grading_complete": True,
            "official_unclassified_errors": grading["unclassified_errors"],
        },
        "budget": {
            "development_cost_usd": str(development_cost),
            "heldout_cost_usd": str(heldout_cost),
            "amendment_008_exposure_usd": str(development_cost + heldout_cost),
            "maximum_new_exposure_usd": protocol["budget"][
                "maximum_new_exposure_usd"
            ],
        },
        "full_60_task_analysis": full,
        "subgroup_analysis": subgroup_results,
        "repository_analysis": repository_results,
        "repository_macro_router_vs_fixed_strong": repository_macro,
        "joint_success": {
            "criterion": protocol["analysis"]["joint_success"],
            "components": joint_components,
            "passed": all(joint_components.values()),
        },
        "active_router_audit": active_audit,
        "shadow_identity_audit": shadow_audit,
        "non_deployable_task_oracles": {
            "fixed_models_only": fixed_oracle,
            "all_three_arms": all_arm_oracle,
        },
        "task_results": [
            {
                "task_id": task_id,
                "repository": repositories[task_id],
                "subgroup": subgroups[task_id],
                "policies": {
                    policy_id: {
                        "resolved": (
                            task_id in resolved_by_policy[policy_id]
                        ),
                        "submitted": bool(
                            records[(task_id, policy_id)]["submitted_patch"]
                        ),
                        "cost_usd": str(
                            records[(task_id, policy_id)][
                                "conservative_cost_usd"
                            ]
                        ),
                        "model_calls": int(
                            records[(task_id, policy_id)]["model_calls"]
                        ),
                        "cheap_calls": int(
                            records[(task_id, policy_id)].get(
                                "cheap_calls", 0
                            )
                        ),
                        "strong_calls": int(
                            records[(task_id, policy_id)].get(
                                "strong_calls", 0
                            )
                        ),
                    }
                    for policy_id in POLICIES
                },
            }
            for task_id in task_ids
        ],
        "official_grading": {
            "dataset_sha256": grading["dataset_sha256"],
            "harness_commit": grading["harness_commit"],
            "predictions_manifest_sha256": grading[
                "predictions_manifest_sha256"
            ],
            "grading_manifest_sha256": _sha256(grading_manifest_path),
            "classified_model_caused_errors": grading[
                "classified_model_caused_errors"
            ],
            "models": {
                policy_id: {
                    "label": LABELS[policy_id],
                    "submitted_count": grading["models"][LABELS[policy_id]][
                        "submitted_count"
                    ],
                    "official_terminal_count": grading["models"][
                        LABELS[policy_id]
                    ]["official_terminal_count"],
                    "resolved_count": grading["models"][LABELS[policy_id]][
                        "resolved_count"
                    ],
                    "resolved_ids": grading["models"][LABELS[policy_id]][
                        "resolved_ids"
                    ],
                    "model_caused_error_ids": grading["models"][
                        LABELS[policy_id]
                    ]["model_caused_error_ids"],
                    "report_sha256": grading["models"][LABELS[policy_id]][
                        "report_sha256"
                    ],
                }
                for policy_id in POLICIES
            },
        },
        "source_hashes": {
            "records": _sha256(records_path),
            "protocol": _sha256(protocol_path),
            "router_artifact": _sha256(artifact_path),
            "task_manifest": _sha256(task_manifest_path),
            "price_snapshot": _sha256(price_snapshot_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/agent_step_router_v2/dynamic_episodes.jsonl"),
    )
    parser.add_argument(
        "--grading-manifest",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v2/swebench_grader/"
            "heldout/grading_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v2/router_artifact.json"),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/agent_step_followup_v2_task_manifest.json"),
    )
    parser.add_argument(
        "--price-snapshot",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-29_router_v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agent_step_router_v2_results.json"),
    )
    args = parser.parse_args()
    result = analyze(
        records_path=args.records,
        grading_manifest_path=args.grading_manifest,
        protocol_path=args.protocol,
        artifact_path=args.artifact,
        task_manifest_path=args.task_manifest,
        price_snapshot_path=args.price_snapshot,
        project_root=args.project_root,
    )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result artifact: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policies": result["full_60_task_analysis"]["policies"],
                "router_vs_fixed_strong": result["full_60_task_analysis"][
                    "router_vs_fixed_strong"
                ],
                "joint_success": result["joint_success"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
