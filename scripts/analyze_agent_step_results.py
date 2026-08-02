#!/usr/bin/env python3
"""Analyze official paired results for the frozen agent-step router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from budget_router.serialization import read_jsonl

try:
    from scripts.export_agent_step_predictions import LABELS
    from scripts.run_agent_step_study import (
        FIXED_CHEAP,
        FIXED_STRONG,
        HELDOUT_POLICIES,
        ROUTED,
    )
except ModuleNotFoundError:
    from export_agent_step_predictions import LABELS  # type: ignore[no-redef]
    from run_agent_step_study import (  # type: ignore[no-redef]
        FIXED_CHEAP,
        FIXED_STRONG,
        HELDOUT_POLICIES,
        ROUTED,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(len(ordered) - 1, left + 1)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def paired_bootstrap(
    routed_quality: Sequence[int],
    fixed_quality: Sequence[int],
    routed_costs: Sequence[float],
    fixed_costs: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    lengths = {
        len(routed_quality),
        len(fixed_quality),
        len(routed_costs),
        len(fixed_costs),
    }
    if len(lengths) != 1 or not routed_quality:
        raise ValueError("paired bootstrap inputs must have one non-zero length")
    size = len(routed_quality)
    rng = random.Random(seed)
    quality_differences: list[float] = []
    cost_savings: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        quality_differences.append(
            sum(routed_quality[index] - fixed_quality[index] for index in indices)
            / size
        )
        fixed_total = sum(fixed_costs[index] for index in indices)
        routed_total = sum(routed_costs[index] for index in indices)
        cost_savings.append(
            1 - routed_total / fixed_total if fixed_total else 0.0
        )
    return {
        "samples": samples,
        "seed": seed,
        "quality_difference_95ci": [
            _quantile(quality_differences, 0.025),
            _quantile(quality_differences, 0.975),
        ],
        "cost_saving_fraction_95ci": [
            _quantile(cost_savings, 0.025),
            _quantile(cost_savings, 0.975),
        ],
    }


def mcnemar_exact_two_sided(wins: int, losses: int) -> float:
    """Return the exact two-sided McNemar p-value for paired binary outcomes."""
    if wins < 0 or losses < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def analyze(
    *,
    records_path: Path,
    grading_manifest_path: Path,
    protocol_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    artifact = _load(artifact_path)
    grading = _load(grading_manifest_path)
    if grading.get("complete_with_no_unclassified_errors") is not True:
        raise ValueError("official grading has unclassified or incomplete errors")
    all_records = read_jsonl(records_path)
    records = [
        row for row in all_records if row.get("study_stage") == "heldout"
    ]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["task_id"]), str(row["policy_id"]))
        if key in by_key:
            raise ValueError(f"duplicate held-out record: {key}")
        by_key[key] = row
    task_ids = sorted({task_id for task_id, _ in by_key})
    expected = {
        (task_id, policy_id)
        for task_id in task_ids
        for policy_id in HELDOUT_POLICIES
    }
    if len(task_ids) != int(protocol["dynamic_test"]["heldout_tasks"]):
        raise ValueError("held-out task count differs from the frozen protocol")
    if set(by_key) != expected:
        raise ValueError("held-out policy matrix is incomplete")
    resolved_by_policy = {
        policy_id: set(
            map(str, grading["models"][LABELS[policy_id]]["resolved_ids"])
        )
        for policy_id in HELDOUT_POLICIES
    }

    policies: dict[str, Any] = {}
    for policy_id in HELDOUT_POLICIES:
        policy_records = [by_key[(task_id, policy_id)] for task_id in task_ids]
        resolved = resolved_by_policy[policy_id]
        total_cost = sum(
            (
                Decimal(str(row["conservative_cost_usd"]))
                for row in policy_records
            ),
            Decimal("0"),
        )
        total_calls = sum(int(row["model_calls"]) for row in policy_records)
        cheap_calls = sum(int(row.get("cheap_calls", 0)) for row in policy_records)
        strong_calls = sum(int(row.get("strong_calls", 0)) for row in policy_records)
        policies[policy_id] = {
            "episodes": len(policy_records),
            "submitted_count": sum(
                bool(row["submitted_patch"]) for row in policy_records
            ),
            "resolved_count": len(resolved),
            "resolution_rate": len(resolved) / len(policy_records),
            "resolved_task_ids": sorted(resolved),
            "total_conservative_cost_usd": str(total_cost),
            "cost_per_resolved_task_usd": (
                str(total_cost / len(resolved)) if resolved else None
            ),
            "model_calls": total_calls,
            "cheap_calls": cheap_calls,
            "strong_calls": strong_calls,
            "cheap_call_share": cheap_calls / total_calls if total_calls else 0.0,
            "switch_count": sum(
                int(row.get("switch_count", 0)) for row in policy_records
            ),
            "provider_failure_count": sum(
                bool(row.get("provider_failed")) for row in policy_records
            ),
            "structurally_valid_count": sum(
                bool(row.get("structurally_valid")) for row in policy_records
            ),
            "model_caused_error_count": len(
                grading["models"][LABELS[policy_id]].get(
                    "model_caused_error_ids", []
                )
            ),
            "model_caused_error_task_ids": sorted(
                map(
                    str,
                    grading["models"][LABELS[policy_id]].get(
                        "model_caused_error_ids", []
                    ),
                )
            ),
        }

    routed_quality = [
        int(task_id in resolved_by_policy[ROUTED]) for task_id in task_ids
    ]
    strong_quality = [
        int(task_id in resolved_by_policy[FIXED_STRONG]) for task_id in task_ids
    ]
    routed_costs = [
        float(by_key[(task_id, ROUTED)]["conservative_cost_usd"])
        for task_id in task_ids
    ]
    strong_costs = [
        float(by_key[(task_id, FIXED_STRONG)]["conservative_cost_usd"])
        for task_id in task_ids
    ]
    routed_total = Decimal(policies[ROUTED]["total_conservative_cost_usd"])
    strong_total = Decimal(policies[FIXED_STRONG]["total_conservative_cost_usd"])
    router_wins = sum(
        routed and not fixed
        for routed, fixed in zip(routed_quality, strong_quality, strict=True)
    )
    router_losses = sum(
        fixed and not routed
        for routed, fixed in zip(routed_quality, strong_quality, strict=True)
    )
    comparison = {
        "routed_minus_fixed_strong_resolved": (
            policies[ROUTED]["resolved_count"]
            - policies[FIXED_STRONG]["resolved_count"]
        ),
        "routed_minus_fixed_strong_resolution_rate": (
            policies[ROUTED]["resolution_rate"]
            - policies[FIXED_STRONG]["resolution_rate"]
        ),
        "routed_cost_saving_usd": str(strong_total - routed_total),
        "routed_cost_saving_fraction": (
            float(1 - routed_total / strong_total) if strong_total else 0.0
        ),
        "mcnemar_router_wins": router_wins,
        "mcnemar_router_losses": router_losses,
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(
            router_wins, router_losses
        ),
        "joint_success": (
            policies[ROUTED]["resolved_count"]
            >= policies[FIXED_STRONG]["resolved_count"]
            and routed_total < strong_total
        ),
        "paired_bootstrap": paired_bootstrap(
            routed_quality,
            strong_quality,
            routed_costs,
            strong_costs,
            samples=int(protocol["analysis"]["paired_bootstrap_resamples"]),
            seed=int(protocol["analysis"]["paired_bootstrap_seed"]),
        ),
    }
    routed_policy = policies[ROUTED]
    routing_mechanism_activated = (
        routed_policy["cheap_calls"] > 0
        and routed_policy["strong_calls"] > 0
    )
    routing_mechanism_audit = {
        "mixed_model_routing_activated": routing_mechanism_activated,
        "cheap_calls": routed_policy["cheap_calls"],
        "strong_calls": routed_policy["strong_calls"],
        "cheap_call_share": routed_policy["cheap_call_share"],
        "switch_count": routed_policy["switch_count"],
        "observed_cost_difference_attributable_to_model_routing": (
            routing_mechanism_activated
        ),
        "interpretation": (
            "The frozen policy used both models on held-out trajectories."
            if routing_mechanism_activated
            else (
                "The frozen policy selected only the strong model. The raw "
                "cost difference versus fixed strong therefore reflects "
                "trajectory/token variation, not cheaper-model substitution."
            )
        ),
    }
    heldout_total = sum(
        (
            Decimal(policy["total_conservative_cost_usd"])
            for policy in policies.values()
        ),
        Decimal("0"),
    )
    smoke_total = sum(
        (
            Decimal(str(row["conservative_cost_usd"]))
            for row in all_records
            if row.get("study_stage") == "smoke"
        ),
        Decimal("0"),
    )
    official_grading = {
        "complete_with_no_unclassified_errors": True,
        "dataset_sha256": grading["dataset_sha256"],
        "harness_commit": grading["harness_commit"],
        "predictions_manifest_sha256": grading[
            "predictions_manifest_sha256"
        ],
        "grading_manifest_sha256": _sha256(grading_manifest_path),
        "classified_model_caused_errors": grading[
            "classified_model_caused_errors"
        ],
        "reports": {
            policy_id: {
                "report_sha256": grading["models"][LABELS[policy_id]][
                    "report_sha256"
                ],
                "submitted_count": grading["models"][LABELS[policy_id]][
                    "submitted_count"
                ],
                "official_terminal_count": grading["models"][
                    LABELS[policy_id]
                ]["official_terminal_count"],
            }
            for policy_id in HELDOUT_POLICIES
        },
    }
    task_results = [
        {
            "task_id": task_id,
            "policies": {
                policy_id: {
                    "resolved": task_id in resolved_by_policy[policy_id],
                    "submitted": bool(by_key[(task_id, policy_id)]["submitted_patch"]),
                    "cost_usd": str(
                        by_key[(task_id, policy_id)]["conservative_cost_usd"]
                    ),
                    "model_calls": int(
                        by_key[(task_id, policy_id)]["model_calls"]
                    ),
                    "cheap_calls": int(
                        by_key[(task_id, policy_id)].get("cheap_calls", 0)
                    ),
                    "strong_calls": int(
                        by_key[(task_id, policy_id)].get("strong_calls", 0)
                    ),
                }
                for policy_id in HELDOUT_POLICIES
            },
        }
        for task_id in task_ids
    ]
    return {
        "schema_version": "agent-step-results-v1",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "protocol_sha256": grading["protocol_sha256"],
        "task_manifest_hash": grading["task_manifest_hash"],
        "router_artifact_hash": artifact["artifact_hash"],
        "inferential_status": protocol["inferential_status"],
        "primary_joint_criterion": protocol["analysis"]["joint_success"],
        "cost_accounting": {
            "smoke_cost_usd": str(smoke_total),
            "heldout_cost_usd": str(heldout_total),
            "new_exposure_usd": str(smoke_total + heldout_total),
            "frozen_maximum_new_exposure_usd": protocol["budget"][
                "maximum_new_exposure_usd"
            ],
        },
        "official_grading": official_grading,
        "policies": policies,
        "router_vs_fixed_strong": comparison,
        "routing_mechanism_audit": routing_mechanism_audit,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/agent_step_router_v1/dynamic_episodes.jsonl"),
    )
    parser.add_argument(
        "--grading-manifest",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v1/swebench_grader/"
            "heldout/grading_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/active_router_protocol_amendment_007_agent_step_routing.json"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v1/router_artifact.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agent_step_router_v1_results.json"),
    )
    args = parser.parse_args()
    result = analyze(
        records_path=args.records,
        grading_manifest_path=args.grading_manifest,
        protocol_path=args.protocol,
        artifact_path=args.artifact,
    )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result artifact: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["router_vs_fixed_strong"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
