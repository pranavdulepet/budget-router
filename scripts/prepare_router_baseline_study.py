from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import stable_hash, stable_json

CHEAP_MODEL = "Qwen/Qwen3-8B"
REUSED_MODELS = (
    "Qwen/Qwen3.6-35B-A3B",
    "thinkingmachines/Inkling",
    "moonshotai/Kimi-K2.6",
)
STUDY_MODELS = (CHEAP_MODEL, *REUSED_MODELS)
MODEL_HARD_CAPS_USD = {
    CHEAP_MODEL: Decimal("0.35"),
    **{model: Decimal("0.90") for model in REUSED_MODELS},
}
CHEAP_PROJECTED_MEAN_COST_USD = Decimal("0.18")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _balanced_select(
    task_ids: list[str],
    tasks_by_id: dict[str, dict[str, Any]],
    *,
    count: int,
    seed: int,
    label: str,
) -> list[str]:
    grouped: dict[str, deque[str]] = defaultdict(deque)
    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{label}:{task_id}".encode()
        ).hexdigest(),
    )
    for task_id in ordered:
        grouped[str(tasks_by_id[task_id]["repo"])].append(task_id)
    repositories = sorted(
        grouped,
        key=lambda repository: hashlib.sha256(
            f"{seed}:{label}:{repository}".encode()
        ).hexdigest(),
    )
    selected: list[str] = []
    while len(selected) < count and any(grouped.values()):
        for repository in repositories:
            if grouped[repository]:
                selected.append(grouped[repository].popleft())
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"{label} has only {len(selected)} selectable tasks")
    return selected


def prepare_study(
    tasks_path: Path,
    split_path: Path,
    pilot_path: Path,
    model_pool_path: Path,
    pilot_results_path: Path,
    *,
    task_selection_seed: int,
    episode_seed_base: int,
    episode_hard_cap_usd: Decimal,
    working_limit_usd: Decimal,
    absolute_limit_usd: Decimal,
) -> dict[str, Any]:
    tasks_payload = _load(tasks_path)
    split_payload = _load(split_path)
    pilot_payload = _load(pilot_path)
    pool_payload = _load(model_pool_path)
    pilot_results = _load(pilot_results_path)
    tasks_by_id = {
        str(record["instance_id"]): dict(record)
        for record in tasks_payload["records"]
    }
    pilot_ids = [str(row["task_id"]) for row in pilot_payload["tasks"]]
    pilot_set = set(pilot_ids)
    if len(pilot_ids) != 12 or len(pilot_set) != 12:
        raise ValueError("the reusable pilot cohort must contain 12 unique tasks")
    if not pilot_set <= set(split_payload["train"]):
        raise ValueError("all reusable pilot tasks must be in the training split")

    treatments_by_model = {
        str(treatment["model"]): dict(treatment)
        for treatment in pool_payload["treatments"]
    }
    missing_models = sorted(set(STUDY_MODELS) - set(treatments_by_model))
    if missing_models:
        raise ValueError(f"study models missing from pool: {missing_models}")

    screen_train = _balanced_select(
        [task_id for task_id in split_payload["train"] if task_id not in pilot_set],
        tasks_by_id,
        count=20,
        seed=task_selection_seed,
        label="screen_train",
    )
    used_train = pilot_set | set(screen_train)
    expand_train = _balanced_select(
        [task_id for task_id in split_payload["train"] if task_id not in used_train],
        tasks_by_id,
        count=28,
        seed=task_selection_seed,
        label="expand_train",
    )
    calibration = _balanced_select(
        list(split_payload["calibration"]),
        tasks_by_id,
        count=20,
        seed=task_selection_seed,
        label="calibration",
    )
    test = _balanced_select(
        list(split_payload["test"]),
        tasks_by_id,
        count=20,
        seed=task_selection_seed,
        label="test",
    )

    pilot_means = {
        str(row["model"]): Decimal(str(row["projected_mean_cost_usd"]))
        for row in pilot_results["results"]
    }
    expected_reused_models_per_task = sum(
        pilot_means[model] for model in REUSED_MODELS
    )
    expected_all_models_per_task = (
        expected_reused_models_per_task + CHEAP_PROJECTED_MEAN_COST_USD
    )
    stages = {
        "cheap_backfill": {
            "split": "train",
            "task_ids": pilot_ids,
            "models": [CHEAP_MODEL],
            "requires": None,
            "new_episodes": len(pilot_ids),
        },
        "screen": {
            "split": "train",
            "task_ids": screen_train,
            "models": list(STUDY_MODELS),
            "requires": None,
            "new_episodes": len(screen_train) * len(STUDY_MODELS),
        },
        "expand": {
            "split": "train",
            "task_ids": expand_train,
            "models": "screen_survivors",
            "requires": "screen_gate_passed",
            "new_episodes": len(expand_train) * len(STUDY_MODELS),
        },
        "calibration": {
            "split": "calibration",
            "task_ids": calibration,
            "models": "screen_survivors",
            "requires": "screen_gate_passed",
            "new_episodes": len(calibration) * len(STUDY_MODELS),
        },
        "test": {
            "split": "test",
            "task_ids": test,
            "models": "screen_survivors",
            "requires": "router_frozen",
            "new_episodes": len(test) * len(STUDY_MODELS),
        },
    }
    for name, stage in stages.items():
        tasks = len(stage["task_ids"])
        if name == "cheap_backfill":
            expected = CHEAP_PROJECTED_MEAN_COST_USD * tasks
            maximum = MODEL_HARD_CAPS_USD[CHEAP_MODEL] * tasks
        else:
            # Expansion stages are conservatively projected with all four
            # models. The preregistered screen can only reduce this cost.
            expected = expected_all_models_per_task * tasks
            maximum = sum(MODEL_HARD_CAPS_USD.values()) * tasks
        stage["expected_new_cost_usd"] = str(expected)
        stage["maximum_new_cost_usd"] = str(maximum)
        stage["projection_note"] = (
            "all four models before preregistered screen pruning"
            if stage["models"] == "screen_survivors"
            else "explicit stage model set"
        )

    all_task_ids = pilot_ids + screen_train + expand_train + calibration + test
    if len(all_task_ids) != 100 or len(set(all_task_ids)) != 100:
        raise ValueError("study must contain exactly 100 unique tasks")
    task_records = [
        {
            "task_id": task_id,
            "repository": str(tasks_by_id[task_id]["repo"]),
            "split": (
                "train"
                if task_id in set(split_payload["train"])
                else "calibration"
                if task_id in set(split_payload["calibration"])
                else "test"
            ),
            "ordinal": ordinal,
        }
        for ordinal, task_id in enumerate(all_task_ids)
    ]
    new_task_count = len(all_task_ids) - len(pilot_ids)
    result = {
        "schema_version": "router-baseline-study-v2",
        "study_id": "cost-aware-router-paired-baselines-2026-07-27-v2",
        "task_selection_seed": task_selection_seed,
        "episode_seed_base": episode_seed_base,
        "source": tasks_payload["source"],
        "source_revision": tasks_payload["revision"],
        "split_manifest_hash": split_payload["manifest_hash"],
        "model_pool_version": pool_payload["pool_version"],
        "price_snapshot": pool_payload["price_snapshot"],
        "models": list(STUDY_MODELS),
        "treatments": [treatments_by_model[model] for model in STUDY_MODELS],
        "model_hard_caps_usd": {
            model: str(MODEL_HARD_CAPS_USD[model]) for model in STUDY_MODELS
        },
        "legacy_episode_hard_cap_usd": str(episode_hard_cap_usd),
        "incremental_working_limit_usd": str(working_limit_usd),
        "incremental_absolute_limit_usd": str(absolute_limit_usd),
        "reused_pilot": {
            "task_ids": pilot_ids,
            "models": list(REUSED_MODELS),
            "episodes": len(pilot_ids) * len(REUSED_MODELS),
            "new_spend_usd": "0",
        },
        "stages": stages,
        "tasks": task_records,
        "screen_gate": {
            "cheap_model": CHEAP_MODEL,
            "minimum_cheap_structurally_valid_rate": 0.75,
            "minimum_safe_cheap_opportunities": 3,
            "minimum_oracle_unique_gain_tasks_over_best_fixed": 2,
            "minimum_oracle_gain_rate_over_best_fixed": 0.05,
            "retain_model_if": [
                "it is the best fixed model",
                "it is the qualified cheap model with at least three safe opportunities",
                "it resolves a task that no cheaper retained model resolves",
                "it is not strictly Pareto dominated in resolved count and mean cost",
            ],
            "decision_rule": (
                "continue with at least two retained models when the screen shows "
                "either quality complementarity or predictable cheap-safe opportunities"
            ),
        },
        "router_protocol": {
            "kind": "cost-aware task selection plus separately evaluated cheap-scout cascade",
            "train_splits": ["train"],
            "tuning_split": "calibration",
            "locked_split": "test",
            "primary_metric": "resolution_rate",
            "secondary_metrics": [
                "cost_per_resolved_task_usd",
                "mean_cost_usd",
                "submission_rate",
                "calibration_brier",
            ],
            "comparators": [
                "each fixed model",
                "global best fixed model",
                "cheapest fixed model",
                "best fixed model under matched cost",
                "frequency-matched random",
                "task oracle",
            ],
            "tie_break": ["lower cost", "simpler policy"],
            "cascade": {
                "status": "optional_after_static_router_freeze",
                "cheap_scout": CHEAP_MODEL,
                "evaluation": "separate online arm; never reconstructed from fixed-model trajectories",
                "maximum_incremental_test_cost_usd": "18.00",
                "run_only_if_full_reservation_fits_absolute_budget": True,
            },
        },
        "budget_projection": {
            "new_tasks": new_task_count,
            "maximum_new_baseline_episodes_before_pruning": (
                len(pilot_ids) + new_task_count * len(STUDY_MODELS)
            ),
            "expected_new_baseline_cost_usd_before_pruning": str(
                expected_reused_models_per_task * new_task_count
                + CHEAP_PROJECTED_MEAN_COST_USD * len(all_task_ids)
            ),
            "cheap_cost_projection_provenance": (
                "preregistered conservative planning assumption; replace with observed "
                "mean after the screen"
            ),
            "maximum_if_every_episode_hits_cap_usd": str(
                sum(MODEL_HARD_CAPS_USD[model] for model in REUSED_MODELS)
                * new_task_count
                + MODEL_HARD_CAPS_USD[CHEAP_MODEL] * len(all_task_ids)
            ),
            "execution_guard": (
                "launch complete stage-model task blocks only when their full hard-cap "
                "reservation fits the incremental working and absolute limits"
            ),
            "optional_cascade_reservation_usd": "18.00",
        },
        "input_hashes": {
            "tasks": stable_hash(tasks_payload),
            "split": stable_hash(split_payload),
            "pilot": stable_hash(pilot_payload),
            "model_pool": stable_hash(pool_payload),
            "pilot_results": stable_hash(pilot_results),
        },
    }
    result["manifest_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("artifacts/swebench_verified_split.json"),
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("artifacts/pilot_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_coding_v7.json"),
    )
    parser.add_argument(
        "--pilot-results",
        type=Path,
        default=Path("outputs/pilot_coding_v6_final/results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument("--task-selection-seed", type=int, default=20260727)
    parser.add_argument("--episode-seed-base", type=int, default=20260726)
    parser.add_argument("--episode-hard-cap-usd", type=Decimal, default=Decimal("0.90"))
    parser.add_argument("--working-limit-usd", type=Decimal, default=Decimal("190"))
    parser.add_argument("--absolute-limit-usd", type=Decimal, default=Decimal("200"))
    args = parser.parse_args()
    if not Decimal("0") < args.working_limit_usd <= args.absolute_limit_usd:
        raise SystemExit("working limit must be positive and at most the absolute limit")

    study = prepare_study(
        args.tasks,
        args.split,
        args.pilot,
        args.model_pool,
        args.pilot_results,
        task_selection_seed=args.task_selection_seed,
        episode_seed_base=args.episode_seed_base,
        episode_hard_cap_usd=args.episode_hard_cap_usd,
        working_limit_usd=args.working_limit_usd,
        absolute_limit_usd=args.absolute_limit_usd,
    )
    serialized = stable_json(study) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to overwrite different frozen study: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(study, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
