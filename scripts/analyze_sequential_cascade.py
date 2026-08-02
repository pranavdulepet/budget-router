from __future__ import annotations

import argparse
import json
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import read_jsonl, stable_hash, stable_json

try:
    from scripts.evaluate_frozen_task_router import (
        _paired_exact_comparison,
        _wilson_interval,
    )
except ModuleNotFoundError:
    from evaluate_frozen_task_router import (  # type: ignore[no-redef]
        _paired_exact_comparison,
        _wilson_interval,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def _paired_bootstrap(
    cascade_success: list[bool],
    fixed_success: list[bool],
    cascade_cost: list[Decimal],
    fixed_cost: list[Decimal],
    *,
    samples: int = 10_000,
    seed: int = 20260728,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    count = len(cascade_success)
    quality: list[float] = []
    cost: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        quality.append(
            sum(cascade_success[index] for index in indices) / count
            - sum(fixed_success[index] for index in indices) / count
        )
        cost.append(
            float(
                sum((cascade_cost[index] for index in indices), Decimal("0"))
                - sum((fixed_cost[index] for index in indices), Decimal("0"))
            )
        )
    return {
        "resolution_rate_delta_95_interval": [
            _percentile(quality, 0.025),
            _percentile(quality, 0.975),
        ],
        "total_cost_delta_usd_95_interval": [
            _percentile(cost, 0.025),
            _percentile(cost, 0.975),
        ],
    }


def analyze_cascade(
    protocol: dict[str, Any],
    baseline_grades: dict[str, Any],
    cascade_grades: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    task_ids = [str(task_id) for task_id in protocol["task_ids"]]
    policy_id = str(protocol["cascade_policy"]["policy_id"])
    fixed_model = str(protocol["comparator"]["policy"]).removeprefix("fixed:")
    baseline_rows = {
        str(row["task_id"]): dict(row)
        for row in baseline_grades["task_results"]
        if row["model"] == fixed_model and row["task_id"] in task_ids
    }
    cascade_rows = {
        str(row["task_id"]): dict(row)
        for row in cascade_grades["task_results"]
        if row["model"] == policy_id and row["task_id"] in task_ids
    }
    episode_rows = {
        str(row["task_id"]): dict(row)
        for row in episodes
        if row["model"] == policy_id and row["task_id"] in task_ids
    }
    expected = set(task_ids)
    for name, rows in (
        ("baseline grades", baseline_rows),
        ("cascade grades", cascade_rows),
        ("cascade episodes", episode_rows),
    ):
        if set(rows) != expected:
            raise ValueError(f"{name} do not match the frozen task cohort")

    fixed_success = [
        baseline_rows[task_id]["status"] == "resolved" for task_id in task_ids
    ]
    cascade_success = [
        cascade_rows[task_id]["status"] == "resolved" for task_id in task_ids
    ]
    fixed_cost = [
        Decimal(str(baseline_rows[task_id]["conservative_cost_usd"]))
        for task_id in task_ids
    ]
    cascade_cost = [
        Decimal(str(cascade_rows[task_id]["conservative_cost_usd"]))
        for task_id in task_ids
    ]
    fixed_total = sum(fixed_cost, Decimal("0"))
    cascade_total = sum(cascade_cost, Decimal("0"))
    fixed_resolved = sum(fixed_success)
    cascade_resolved = sum(cascade_success)
    fixed_summary = {
        "policy": f"fixed:{fixed_model}",
        "resolved_count": fixed_resolved,
        "resolution_rate": fixed_resolved / len(task_ids),
        "resolution_rate_95_interval": _wilson_interval(
            fixed_resolved, len(task_ids)
        ),
        "total_cost_usd": str(fixed_total),
        "mean_cost_usd": str(fixed_total / len(task_ids)),
        "cost_per_resolved_task_usd": (
            str(fixed_total / fixed_resolved) if fixed_resolved else None
        ),
    }
    cascade_summary = {
        "policy": policy_id,
        "resolved_count": cascade_resolved,
        "resolution_rate": cascade_resolved / len(task_ids),
        "resolution_rate_95_interval": _wilson_interval(
            cascade_resolved, len(task_ids)
        ),
        "total_cost_usd": str(cascade_total),
        "mean_cost_usd": str(cascade_total / len(task_ids)),
        "cost_per_resolved_task_usd": (
            str(cascade_total / cascade_resolved) if cascade_resolved else None
        ),
        "submitted_count": sum(
            bool(episode_rows[task_id]["submitted_patch"])
            for task_id in task_ids
        ),
        "handoff_count": sum(
            int(episode_rows[task_id]["handoff_count"] > 0)
            for task_id in task_ids
        ),
        "scout_early_submission_count": sum(
            bool(episode_rows[task_id]["scout_early_submission"])
            for task_id in task_ids
        ),
        "scout_total_cost_usd": str(
            sum(
                (
                    Decimal(str(episode_rows[task_id]["scout_cost_usd"]))
                    for task_id in task_ids
                ),
                Decimal("0"),
            )
        ),
        "finisher_total_cost_usd": str(
            sum(
                (
                    Decimal(str(episode_rows[task_id]["finisher_cost_usd"]))
                    for task_id in task_ids
                ),
                Decimal("0"),
            )
        ),
        "scout_total_calls": sum(
            int(episode_rows[task_id]["scout_calls"]) for task_id in task_ids
        ),
        "finisher_total_calls": sum(
            int(episode_rows[task_id]["finisher_calls"]) for task_id in task_ids
        ),
    }
    paired = _paired_exact_comparison(cascade_success, fixed_success)
    paired.update(
        {
            "resolved_count_delta": cascade_resolved - fixed_resolved,
            "total_cost_delta_usd": str(cascade_total - fixed_total),
            **_paired_bootstrap(
                cascade_success,
                fixed_success,
                cascade_cost,
                fixed_cost,
            ),
        }
    )
    oracle_selections: list[str] = []
    oracle_cost = Decimal("0")
    oracle_resolved = 0
    for index in range(len(task_ids)):
        if cascade_success[index] != fixed_success[index]:
            selection = "cascade" if cascade_success[index] else "fixed"
        elif cascade_cost[index] < fixed_cost[index]:
            selection = "cascade"
        else:
            selection = "fixed"
        oracle_selections.append(selection)
        if selection == "cascade":
            oracle_cost += cascade_cost[index]
            oracle_resolved += int(cascade_success[index])
        else:
            oracle_cost += fixed_cost[index]
            oracle_resolved += int(fixed_success[index])
    policy_oracle = {
        "description": (
            "Hindsight upper bound: on outcome-discordant tasks choose the "
            "successful policy; otherwise choose the lower-cost policy."
        ),
        "deployable": False,
        "resolved_count": oracle_resolved,
        "incremental_resolutions_over_fixed": oracle_resolved - fixed_resolved,
        "total_cost_usd": str(oracle_cost),
        "total_cost_delta_vs_fixed_usd": str(oracle_cost - fixed_total),
        "relative_cost_reduction_vs_fixed": float(
            (fixed_total - oracle_cost) / fixed_total
        ),
        "cost_per_resolved_task_usd": (
            str(oracle_cost / oracle_resolved) if oracle_resolved else None
        ),
        "selection_counts": {
            policy: oracle_selections.count(policy)
            for policy in ("fixed", "cascade")
        },
    }
    success = (
        cascade_resolved > fixed_resolved
        or (
            cascade_resolved == fixed_resolved
            and cascade_total < fixed_total
        )
    )
    result = {
        "schema_version": "sequential-cascade-evaluation-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "task_count": len(task_ids),
        "evidence_status": "secondary exploratory paired experiment",
        "fixed_comparator": fixed_summary,
        "cascade": cascade_summary,
        "paired_comparison": paired,
        "cost_preserving_policy_oracle": policy_oracle,
        "frozen_success_rule_passed": success,
        "task_results": [
            {
                "task_id": task_id,
                "fixed_resolved": fixed_success[index],
                "cascade_resolved": cascade_success[index],
                "fixed_cost_usd": str(fixed_cost[index]),
                "cascade_cost_usd": str(cascade_cost[index]),
                "scout_cost_usd": str(
                    episode_rows[task_id]["scout_cost_usd"]
                ),
                "finisher_cost_usd": str(
                    episode_rows[task_id]["finisher_cost_usd"]
                ),
                "handoff": bool(episode_rows[task_id]["handoff_count"]),
                "scout_early_submission": bool(
                    episode_rows[task_id]["scout_early_submission"]
                ),
            }
            for index, task_id in enumerate(task_ids)
        ],
    }
    result["evaluation_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/sequential_cascade_study_v1.json"),
    )
    parser.add_argument(
        "--baseline-grades",
        type=Path,
        default=Path("outputs/router_baseline_v2/test_grades.json"),
    )
    parser.add_argument(
        "--cascade-grades",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/grades.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/evaluation.json"),
    )
    args = parser.parse_args()
    result = analyze_cascade(
        _load(args.protocol),
        _load(args.baseline_grades),
        _load(args.cascade_grades),
        read_jsonl(args.episodes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
