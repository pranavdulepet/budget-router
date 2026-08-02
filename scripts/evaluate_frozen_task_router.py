from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import stable_hash, stable_json
from budget_router.task_model import HashedLinearModelHead
from budget_router.types import GoalContext, RouterState


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def _wilson_interval(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    z = 1.959963984540054
    rate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (rate + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _paired_exact_comparison(
    left: list[bool],
    right: list[bool],
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired comparison needs equal non-empty outcome vectors")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, count)
            for count in range(min(left_only, right_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "left_only_resolved": left_only,
        "right_only_resolved": right_only,
        "discordant_task_count": discordant,
        "resolution_rate_delta": (
            sum(left) / len(left) - sum(right) / len(right)
        ),
        "two_sided_exact_mcnemar_p": p_value,
    }


def _bootstrap_deltas(
    router_success: list[bool],
    baseline_success: list[bool],
    router_cost: list[Decimal],
    baseline_cost: list[Decimal],
    *,
    samples: int = 10_000,
    seed: int = 20260727,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    count = len(router_success)
    quality: list[float] = []
    cost: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        quality.append(
            sum(router_success[index] for index in indices) / count
            - sum(baseline_success[index] for index in indices) / count
        )
        cost.append(
            float(
                sum((router_cost[index] for index in indices), Decimal("0"))
                / count
                - sum((baseline_cost[index] for index in indices), Decimal("0"))
                / count
            )
        )
    return {
        "resolution_rate_delta_95_interval": [
            _percentile(quality, 0.025),
            _percentile(quality, 0.975),
        ],
        "mean_cost_delta_usd_95_interval": [
            _percentile(cost, 0.025),
            _percentile(cost, 0.975),
        ],
    }


def evaluate_frozen_router(
    study: dict[str, Any],
    freeze: dict[str, Any],
    artifact: dict[str, Any],
    tasks_payload: dict[str, Any],
    grade_payloads: Iterable[dict[str, Any]],
    *,
    artifact_bytes: bytes,
) -> dict[str, Any]:
    if freeze.get("study_manifest_hash") != study["manifest_hash"]:
        raise ValueError("router freeze does not match study")
    if hashlib.sha256(artifact_bytes).hexdigest() != freeze["router_artifact_sha256"]:
        raise ValueError("router artifact differs from the pre-test freeze")
    if artifact.get("artifact_hash") != freeze["router_artifact_hash"]:
        raise ValueError("router artifact hash differs from the pre-test freeze")
    models = [str(model) for model in freeze["models"]]
    test_ids = [str(task_id) for task_id in study["stages"]["test"]["task_ids"]]
    tasks = {
        str(row["instance_id"]): dict(row) for row in tasks_payload["records"]
    }
    grade_rows: dict[tuple[str, str], dict[str, Any]] = {}
    grade_hashes: list[str] = []
    for payload in grade_payloads:
        grade_hashes.append(stable_hash(payload))
        for row in payload["task_results"]:
            key = (str(row["task_id"]), str(row["model"]))
            if key in grade_rows:
                raise ValueError(f"duplicate official test grade: {key}")
            grade_rows[key] = dict(row)
    expected = {(task_id, model) for task_id in test_ids for model in models}
    missing = sorted(expected - set(grade_rows))
    if missing:
        raise ValueError(f"official test grades are incomplete: {missing}")

    head = HashedLinearModelHead.from_artifact(artifact["model_head"])
    expected_costs = {
        model: Decimal(str(cost))
        for model, cost in artifact["selector"]["expected_cost_usd"].items()
    }
    cost_penalty = float(artifact["selector"]["cost_penalty"])
    cost_scale = max(expected_costs.values())
    selections: list[dict[str, Any]] = []
    for task_id in test_ids:
        text = str(tasks[task_id]["problem_statement"])
        probabilities = {
            model: head.estimate(
                # The learned head uses only goal text; repository is present
                # for the public runtime contract but is not a feature.
                GoalContext(text, str(tasks[task_id]["repo"]), tuple(models)),
                RouterState(),
                model,
            ).success_probability
            for model in models
        }
        scores = {
            model: probabilities[model]
            - cost_penalty * float(expected_costs[model] / cost_scale)
            for model in models
        }
        selected = max(
            models,
            key=lambda model: (
                scores[model],
                probabilities[model],
                -expected_costs[model],
            ),
        )
        selected_grade = grade_rows[(task_id, selected)]
        selections.append(
            {
                "task_id": task_id,
                "selected_model": selected,
                "predicted_success": probabilities[selected],
                "scores": scores,
                "resolved": selected_grade["status"] == "resolved",
                "conservative_cost_usd": str(
                    selected_grade["conservative_cost_usd"]
                ),
            }
        )

    policies: dict[str, dict[str, Any]] = {}
    for model in models:
        model_rows = [grade_rows[(task_id, model)] for task_id in test_ids]
        resolved = [row["status"] == "resolved" for row in model_rows]
        costs = [Decimal(str(row["conservative_cost_usd"])) for row in model_rows]
        total = sum(costs, Decimal("0"))
        policies[f"fixed:{model}"] = {
            "resolved_count": sum(resolved),
            "resolution_rate": sum(resolved) / len(test_ids),
            "resolution_rate_95_interval": _wilson_interval(
                sum(resolved), len(test_ids)
            ),
            "total_cost_usd": str(total),
            "mean_cost_usd": str(total / len(test_ids)),
            "cost_per_resolved_task_usd": (
                str(total / sum(resolved)) if sum(resolved) else None
            ),
        }
    router_resolved = [bool(row["resolved"]) for row in selections]
    router_costs = [
        Decimal(str(row["conservative_cost_usd"])) for row in selections
    ]
    router_total = sum(router_costs, Decimal("0"))
    policies["router"] = {
        "resolved_count": sum(router_resolved),
        "resolution_rate": sum(router_resolved) / len(test_ids),
        "resolution_rate_95_interval": _wilson_interval(
            sum(router_resolved), len(test_ids)
        ),
        "total_cost_usd": str(router_total),
        "mean_cost_usd": str(router_total / len(test_ids)),
        "cost_per_resolved_task_usd": (
            str(router_total / sum(router_resolved))
            if sum(router_resolved)
            else None
        ),
        "selection_counts": {
            model: sum(row["selected_model"] == model for row in selections)
            for model in models
        },
    }
    best_fixed = max(
        models,
        key=lambda model: (
            policies[f"fixed:{model}"]["resolved_count"],
            -Decimal(policies[f"fixed:{model}"]["total_cost_usd"]),
        ),
    )
    best_rows = [grade_rows[(task_id, best_fixed)] for task_id in test_ids]
    best_success = [row["status"] == "resolved" for row in best_rows]
    best_cost = [
        Decimal(str(row["conservative_cost_usd"])) for row in best_rows
    ]
    oracle_count = sum(
        any(grade_rows[(task_id, model)]["status"] == "resolved" for model in models)
        for task_id in test_ids
    )
    fixed_success = {
        model: [
            grade_rows[(task_id, model)]["status"] == "resolved"
            for task_id in test_ids
        ]
        for model in models
    }
    pairwise_fixed = []
    for left_index, left_model in enumerate(models):
        for right_model in models[left_index + 1 :]:
            pairwise_fixed.append(
                {
                    "left_model": left_model,
                    "right_model": right_model,
                    **_paired_exact_comparison(
                        fixed_success[left_model],
                        fixed_success[right_model],
                    ),
                }
            )
    result = {
        "schema_version": "frozen-task-router-evaluation-v1",
        "study_id": study["study_id"],
        "study_manifest_hash": study["manifest_hash"],
        "router_freeze_hash": freeze["freeze_hash"],
        "test_task_count": len(test_ids),
        "models": models,
        "best_fixed_model": best_fixed,
        "oracle_resolved_count": oracle_count,
        "policies": policies,
        "pairwise_fixed_model_comparisons": pairwise_fixed,
        "router_vs_best_fixed": {
            "resolved_count_delta": (
                policies["router"]["resolved_count"]
                - policies[f"fixed:{best_fixed}"]["resolved_count"]
            ),
            "total_cost_delta_usd": str(
                router_total
                - Decimal(policies[f"fixed:{best_fixed}"]["total_cost_usd"])
            ),
            **_bootstrap_deltas(
                router_resolved,
                best_success,
                router_costs,
                best_cost,
            ),
        },
        "selections": selections,
        "provenance": {
            "test_labels_accessed_after_freeze": True,
            "grade_payload_hashes": grade_hashes,
        },
    }
    result["evaluation_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_freeze.json"),
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=Path("outputs/router_baseline_v2/task_router.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument("--grades", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_baseline_v2/test_evaluation.json"),
    )
    args = parser.parse_args()
    artifact_bytes = args.router.read_bytes()
    result = evaluate_frozen_router(
        _load(args.study),
        _load(args.freeze),
        json.loads(artifact_bytes),
        _load(args.tasks),
        [_load(path) for path in args.grades],
        artifact_bytes=artifact_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
