from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import stable_hash


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _grade_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["task_results"]:
        key = (str(row["task_id"]), str(row["model"]))
        if key in rows:
            raise ValueError(f"duplicate grade row: {key}")
        rows[key] = dict(row)
    return rows


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    return (
        left["resolved_count"] >= right["resolved_count"]
        and Decimal(left["mean_cost_usd"]) <= Decimal(right["mean_cost_usd"])
        and (
            left["resolved_count"] > right["resolved_count"]
            or Decimal(left["mean_cost_usd"]) < Decimal(right["mean_cost_usd"])
        )
    )


def evaluate_screen(
    study: dict[str, Any],
    pilot_grades: dict[str, Any],
    new_grades: dict[str, Any],
    new_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    pilot_ids = [str(task_id) for task_id in study["reused_pilot"]["task_ids"]]
    screen_ids = [str(task_id) for task_id in study["stages"]["screen"]["task_ids"]]
    task_ids = pilot_ids + screen_ids
    models = [str(model) for model in study["models"]]
    cheap_model = str(study["screen_gate"]["cheap_model"])
    reused_models = [str(model) for model in study["reused_pilot"]["models"]]

    pilot_rows = _grade_rows(pilot_grades)
    new_rows = _grade_rows(new_grades)
    expected_pilot = {
        (task_id, model) for task_id in pilot_ids for model in reused_models
    }
    expected_new = {
        *((task_id, cheap_model) for task_id in pilot_ids),
        *((task_id, model) for task_id in screen_ids for model in models),
    }
    missing_pilot = sorted(expected_pilot - set(pilot_rows))
    if missing_pilot:
        raise ValueError(
            f"reused pilot grades are missing the frozen cohort: {missing_pilot}"
        )
    # The immutable pilot artifact also contains historical treatments that the
    # follow-up study did not preregister for reuse. Keep its full hash in
    # provenance, but restrict outcome analysis to the frozen cohort.
    pilot_rows = {key: pilot_rows[key] for key in expected_pilot}
    if set(new_rows) != expected_new:
        missing = sorted(expected_new - set(new_rows))
        extra = sorted(set(new_rows) - expected_new)
        raise ValueError(
            f"new screen grades are incomplete: missing={missing}, extra={extra}"
        )

    episode_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in new_episodes:
        if row.get("study_manifest_hash") != study["manifest_hash"]:
            raise ValueError("new episode has the wrong study manifest hash")
        key = (str(row["task_id"]), str(row["model"]))
        if key in episode_rows:
            raise ValueError(f"duplicate new episode: {key}")
        episode_rows[key] = dict(row)
    if set(episode_rows) != expected_new:
        raise ValueError("new episode records do not match the completed screen")

    matrix: dict[str, dict[str, dict[str, Any]]] = {
        task_id: {} for task_id in task_ids
    }
    for task_id in task_ids:
        for model in models:
            source = (
                new_rows
                if model == cheap_model or task_id in screen_ids
                else pilot_rows
            )
            row = source[(task_id, model)]
            matrix[task_id][model] = {
                "resolved": row["status"] == "resolved",
                "status": str(row["status"]),
                "cost_usd": str(row["conservative_cost_usd"]),
            }

    metrics: dict[str, dict[str, Any]] = {}
    for model in models:
        resolved_ids = [
            task_id for task_id in task_ids if matrix[task_id][model]["resolved"]
        ]
        total_cost = sum(
            (
                Decimal(matrix[task_id][model]["cost_usd"])
                for task_id in task_ids
            ),
            Decimal("0"),
        )
        metrics[model] = {
            "resolved_count": len(resolved_ids),
            "resolution_rate": len(resolved_ids) / len(task_ids),
            "total_cost_usd": str(total_cost),
            "mean_cost_usd": str(total_cost / len(task_ids)),
            "resolved_task_ids": sorted(resolved_ids),
        }

    best_fixed_model = max(
        models,
        key=lambda model: (
            metrics[model]["resolved_count"],
            -Decimal(metrics[model]["mean_cost_usd"]),
        ),
    )
    oracle_ids = {
        task_id
        for task_id in task_ids
        if any(matrix[task_id][model]["resolved"] for model in models)
    }
    oracle_gain = len(oracle_ids) - metrics[best_fixed_model]["resolved_count"]
    oracle_gain_rate = oracle_gain / len(task_ids)
    safe_cheap_ids = [
        task_id
        for task_id in task_ids
        if matrix[task_id][cheap_model]["resolved"]
        and matrix[task_id][best_fixed_model]["resolved"]
    ]

    cheap_episodes = [
        episode_rows[(task_id, cheap_model)] for task_id in task_ids
    ]
    cheap_structurally_valid_count = sum(
        bool(row.get("structurally_valid")) for row in cheap_episodes
    )
    cheap_structurally_valid_rate = (
        cheap_structurally_valid_count / len(cheap_episodes)
    )
    gate = study["screen_gate"]
    cheap_qualified = cheap_structurally_valid_rate >= float(
        gate["minimum_cheap_structurally_valid_rate"]
    )
    cheap_signal = (
        cheap_qualified
        and len(safe_cheap_ids) >= int(gate["minimum_safe_cheap_opportunities"])
    )
    quality_signal = (
        oracle_gain
        >= int(gate["minimum_oracle_unique_gain_tasks_over_best_fixed"])
        and oracle_gain_rate
        >= float(gate["minimum_oracle_gain_rate_over_best_fixed"])
    )

    for model in models:
        cheaper_models = [
            other
            for other in models
            if Decimal(metrics[other]["mean_cost_usd"])
            < Decimal(metrics[model]["mean_cost_usd"])
        ]
        unique_over_cheaper = [
            task_id
            for task_id in task_ids
            if matrix[task_id][model]["resolved"]
            and not any(
                matrix[task_id][other]["resolved"] for other in cheaper_models
            )
        ]
        dominated_by = [
            other
            for other in models
            if other != model and _dominates(metrics[other], metrics[model])
        ]
        metrics[model]["unique_resolutions_over_all_cheaper_models"] = len(
            unique_over_cheaper
        )
        metrics[model]["unique_task_ids_over_all_cheaper_models"] = sorted(
            unique_over_cheaper
        )
        metrics[model]["strictly_pareto_dominated_by"] = sorted(dominated_by)

    retained: set[str] = {best_fixed_model}
    if cheap_signal or best_fixed_model == cheap_model:
        retained.add(cheap_model)
    for model in models:
        if model in retained or model == cheap_model:
            continue
        if (
            metrics[model]["unique_resolutions_over_all_cheaper_models"] > 0
            or not metrics[model]["strictly_pareto_dominated_by"]
        ):
            retained.add(model)
    surviving_models = [model for model in models if model in retained]
    screen_gate_passed = (
        len(surviving_models) >= 2 and (cheap_signal or quality_signal)
    )

    result = {
        "schema_version": "router-screen-gate-v1",
        "study_id": study["study_id"],
        "study_manifest_hash": study["manifest_hash"],
        "screen_gate_passed": screen_gate_passed,
        "surviving_models": surviving_models,
        "best_fixed_model": best_fixed_model,
        "signals": {
            "cheap_model": cheap_model,
            "cheap_qualified": cheap_qualified,
            "cheap_structurally_valid_count": cheap_structurally_valid_count,
            "cheap_structurally_valid_rate": cheap_structurally_valid_rate,
            "safe_cheap_opportunity_count": len(safe_cheap_ids),
            "safe_cheap_task_ids": sorted(safe_cheap_ids),
            "cheap_cost_signal_passed": cheap_signal,
            "oracle_resolved_count": len(oracle_ids),
            "oracle_gain_tasks_over_best_fixed": oracle_gain,
            "oracle_gain_rate_over_best_fixed": oracle_gain_rate,
            "quality_complementarity_signal_passed": quality_signal,
        },
        "models": metrics,
        "provenance": {
            "pilot_grades_hash": stable_hash(pilot_grades),
            "new_grades_hash": stable_hash(new_grades),
            "new_episode_count": len(new_episodes),
        },
    }
    result["gate_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--pilot-grades",
        type=Path,
        default=Path("artifacts/pilot_coding_v6_swebench_grades.json"),
    )
    parser.add_argument(
        "--new-grades",
        type=Path,
        default=Path("outputs/router_baseline_v2/screen_grades.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/router_baseline_v2/episodes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_baseline_v2/screen_gate.json"),
    )
    args = parser.parse_args()
    result = evaluate_screen(
        _load(args.study),
        _load(args.pilot_grades),
        _load(args.new_grades),
        _read_jsonl(args.episodes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
