from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from budget_router.isolated_stage import isolated_policy_id
from budget_router.serialization import stable_hash, stable_json


TRAIN_QUOTAS = {
    "astropy/astropy": 10,
    "django/django": 9,
    "matplotlib/matplotlib": 9,
    "mwaskom/seaborn": 2,
    "pylint-dev/pylint": 5,
}
TEST_QUOTAS = {
    "psf/requests": 3,
    "pydata/xarray": 7,
    "pytest-dev/pytest": 7,
    "scikit-learn/scikit-learn": 7,
    "sphinx-doc/sphinx": 7,
    "sympy/sympy": 8,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def _validate_source_study(study: dict[str, Any]) -> None:
    claimed = str(study.get("manifest_hash", ""))
    unhashed = dict(study)
    unhashed.pop("manifest_hash", None)
    if stable_hash(unhashed) != claimed:
        raise ValueError("source study manifest hash is invalid")


def _fixed_rows(
    payloads: Iterable[dict[str, Any]],
    *,
    fixed_model: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["task_results"]:
            if row["model"] != fixed_model:
                continue
            task_id = str(row["task_id"])
            if task_id in rows:
                raise ValueError(f"duplicate fixed-model grade: {task_id}")
            if row["status"] == "grader_error":
                continue
            rows[task_id] = dict(row)
    return rows


def _quota_sample(
    candidates: Iterable[dict[str, Any]],
    quotas: dict[str, int],
    *,
    seed: str,
) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in candidates:
        by_repo[str(task["repo"])].append(dict(task))
    selected: list[dict[str, Any]] = []
    for repo, count in quotas.items():
        ranked = sorted(
            by_repo.get(repo, ()),
            key=lambda task: (_rank(seed, str(task["instance_id"])), task["instance_id"]),
        )
        if len(ranked) < count:
            raise ValueError(f"not enough eligible tasks for {repo}: {len(ranked)} < {count}")
        selected.extend(ranked[:count])
    return sorted(
        selected,
        key=lambda task: (_rank(seed, str(task["instance_id"])), task["instance_id"]),
    )


def _screen_sample(
    gate_tasks: list[dict[str, Any]],
    *,
    seed: str,
) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in gate_tasks:
        by_repo[str(task["repo"])].append(task)
    selected = [
        min(
            tasks,
            key=lambda task: (_rank(seed, str(task["instance_id"])), task["instance_id"]),
        )
        for _, tasks in sorted(by_repo.items())
    ]
    remaining = [task for task in gate_tasks if task not in selected]
    selected.append(
        min(
            remaining,
            key=lambda task: (_rank(seed, str(task["instance_id"])), task["instance_id"]),
        )
    )
    return sorted(
        selected,
        key=lambda task: (_rank(seed, str(task["instance_id"])), task["instance_id"]),
    )


def build_protocol(
    source_study: dict[str, Any],
    tasks_payload: dict[str, Any],
    model_pool: dict[str, Any],
    prices: dict[str, Any],
    source_grade_payloads: Iterable[dict[str, Any]],
    *,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    _validate_source_study(source_study)
    if model_pool["price_snapshot"] != prices["snapshot_id"]:
        raise ValueError("model pool and price snapshot disagree")
    fixed_model = str(model_pool["finisher"]["model"])
    scouts = [dict(row) for row in model_pool["scouts"]]
    if len(scouts) != 3 or len({row["model"] for row in scouts}) != 3:
        raise ValueError("the screen requires exactly three unique scouts")
    for treatment in [*scouts, model_pool["finisher"]]:
        if treatment["model"] not in prices["models"]:
            raise ValueError(f"missing frozen price for {treatment['model']}")

    all_tasks = [dict(row) for row in tasks_payload["records"]]
    tasks_by_id = {str(row["instance_id"]): row for row in all_tasks}
    if len(tasks_by_id) != len(all_tasks):
        raise ValueError("SWE-bench task IDs are not unique")
    source_tasks = {
        str(row["task_id"]): dict(row)
        for row in source_study["tasks"]
    }
    source_train_ids = {
        task_id
        for task_id, row in source_tasks.items()
        if row["split"] == "train"
    }
    if len(source_train_ids) != 60:
        raise ValueError("expected the frozen source study to contain 60 train tasks")

    fixed_rows = _fixed_rows(source_grade_payloads, fixed_model=fixed_model)
    excluded_fixed_ids = sorted(source_train_ids - set(fixed_rows))

    gate_tasks = _quota_sample(
        (
            tasks_by_id[task_id]
            for task_id in source_train_ids
            if task_id in fixed_rows
        ),
        TRAIN_QUOTAS,
        seed="isolated-stage-gate-train-20260728-v1",
    )
    screen_tasks = _screen_sample(
        gate_tasks,
        seed="isolated-stage-screen-20260728-v1",
    )
    screen_ids = {str(task["instance_id"]) for task in screen_tasks}
    expansion_tasks = [
        task for task in gate_tasks if str(task["instance_id"]) not in screen_ids
    ]
    if len(screen_tasks) != 6 or len(expansion_tasks) != 29:
        raise ValueError("training split must be 6 screen plus 29 expansion tasks")

    used_ids = set(source_tasks)
    fresh_candidates = [
        task for task in all_tasks if str(task["instance_id"]) not in used_ids
    ]
    test_tasks = _quota_sample(
        fresh_candidates,
        TEST_QUOTAS,
        seed="isolated-stage-repository-heldout-test-20260728-v1",
    )
    if len(test_tasks) != 39:
        raise ValueError("test split must contain exactly 39 fresh tasks")
    train_repos = {str(task["repo"]) for task in gate_tasks}
    test_repos = {str(task["repo"]) for task in test_tasks}
    if train_repos & test_repos:
        raise ValueError("gate-training and held-out test repositories overlap")

    candidate_policies = [
        {
            "policy_id": isolated_policy_id(str(scout["model"]), fixed_model),
            "scout_model": str(scout["model"]),
            "finisher_model": fixed_model,
        }
        for scout in scouts
    ]
    if len({row["policy_id"] for row in candidate_policies}) != 3:
        raise ValueError("candidate policy IDs are not unique")

    total_cap = Decimal("0.90")
    screen_maximum = total_cap * 18
    expansion_maximum = total_cap * 29
    test_maximum = total_cap * 78
    maximum_new = screen_maximum + expansion_maximum + test_maximum
    if maximum_new != Decimal("112.50"):
        raise AssertionError("unexpected study reservation")

    dataset_ordinals = {
        str(task["instance_id"]): index for index, task in enumerate(all_tasks)
    }
    task_rows = []
    for stage, tasks in (
        ("screen", screen_tasks),
        ("expand", expansion_tasks),
        ("test", test_tasks),
    ):
        for task in tasks:
            task_id = str(task["instance_id"])
            task_rows.append(
                {
                    "task_id": task_id,
                    "repository": str(task["repo"]),
                    "stage": stage,
                    "dataset_ordinal": dataset_ordinals[task_id],
                }
            )

    protocol = {
        "schema_version": "isolated-stage-router-study-v1",
        "study_id": "isolated-stage-router-2026-07-28-v1",
        "source_static_study_id": source_study["study_id"],
        "source_static_study_manifest_hash": source_study["manifest_hash"],
        "design_timing": (
            "Frozen after the first static-router and same-workspace cascade "
            "studies, before any isolated-scout provider call or outcome."
        ),
        "research_question": (
            "Can a cheap isolated diagnostic model transfer useful visible "
            "evidence to a clean strong finisher, and can an issue-text gate "
            "retain fixed-Qwen quality at lower cost?"
        ),
        "model_pool_version": model_pool["pool_version"],
        "price_snapshot": prices["snapshot_id"],
        "candidate_policies": candidate_policies,
        "fixed_policy": {
            "policy_id": f"fixed:{fixed_model}",
            "model": fixed_model,
        },
        "execution_policy": {
            "kind": "isolated-diagnostic-visible-handoff-v1",
            "scout_max_calls": 6,
            "scout_hard_cap_usd": "0.08",
            "scout_max_output_tokens_per_call": 4_000,
            "scout_submission_discarded": True,
            "scout_workspace_discarded": True,
            "visible_handoff_max_chars": 60_000,
            "hidden_reasoning_transferred": False,
            "finisher_clean_workspace": True,
            "finisher_max_output_tokens_per_call": 8_000,
            "total_step_limit": 75,
            "total_hard_cap_usd": str(total_cap),
            "finisher_budget": "total cap minus actual conservative scout spend",
            "provider_failure_accounting": "charge full preflight reservation",
            "seed_base": 202_607_280,
            "seed_formula": (
                "fixed and every Qwen finisher use (seed_base + "
                "dataset_ordinal) mod 2147483647; scout uses (seed_base + "
                "(scout_index+1)*1000 + dataset_ordinal) mod 2147483647"
            ),
        },
        "stages": {
            "screen": {
                "task_ids": [str(task["instance_id"]) for task in screen_tasks],
                "policy_ids": [row["policy_id"] for row in candidate_policies],
                "episode_count": 18,
                "maximum_cost_usd": str(screen_maximum),
                "requires": None,
            },
            "expand": {
                "task_ids": [str(task["instance_id"]) for task in expansion_tasks],
                "policy_ids": "selected_candidate",
                "episode_count": 29,
                "maximum_cost_usd": str(expansion_maximum),
                "requires": "candidate_frozen",
            },
            "test": {
                "task_ids": [str(task["instance_id"]) for task in test_tasks],
                "policy_ids": "selected_candidate_plus_fixed",
                "episode_count": 78,
                "maximum_cost_usd": str(test_maximum),
                "requires": "gate_frozen",
            },
        },
        "selection": {
            "eligible_if": [
                "all six screen episodes structurally valid",
                "zero provider failures",
                "zero official grader errors",
            ],
            "lexicographic_order": [
                "fewest fixed-only losses",
                "most candidate-only wins",
                "most resolved tasks",
                "lowest total conservative cost",
                "policy_id ascending",
            ],
            "fixed_outcome_source": "previously collected official grades",
        },
        "gate": {
            "features": ["public problem_statement"],
            "excluded_features": [
                "repository identity",
                "gold patch",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
                "test outcomes",
                "test grader logs",
            ],
            "heads": ["candidate-only win", "fixed-only loss"],
            "model": "balanced hashed logistic regression",
            "dimension": 1_024,
            "hash_seed": "isolated-stage-gate-v1",
            "validation": "leave-one-repository-out cross-validation",
            "learning_rates": [0.2, 0.5],
            "l2_values": [0.001, 0.01, 0.05],
            "epochs": 200,
            "cost_weights": [0.0, 0.05, 0.1, 0.2, 0.4],
            "decision_thresholds": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            "selection_objective": [
                "highest cross-validated resolved count",
                "lowest cross-validated conservative cost",
                "fewest cross-validated fixed-only losses",
                "fewest candidate selections",
                "simpler hyperparameters",
            ],
            "freeze_before_test": True,
        },
        "analysis": {
            "primary_comparison": "learned gate versus fixed Qwen",
            "secondary_comparisons": [
                "universal selected candidate versus fixed Qwen",
                "learned gate versus universal selected candidate",
                "post-hoc task oracle versus both deployable policies",
            ],
            "quality_metric": "official SWE-bench Verified resolution rate",
            "cost_metric": "total conservative inference cost",
            "paired_test": "two-sided exact McNemar",
            "uncertainty": "task bootstrap with seed 20260728",
            "all_no_patch_episodes_count_unresolved": True,
        },
        "sampling": {
            "gate_training_quotas": TRAIN_QUOTAS,
            "test_quotas": TEST_QUOTAS,
            "gate_training_task_count": 35,
            "repository_disjoint_test": True,
            "test_tasks_unused_by_source_study": True,
            "source_train_tasks_without_valid_fixed_grade_excluded": (
                excluded_fixed_ids
            ),
        },
        "budget": {
            "screen_maximum_usd": str(screen_maximum),
            "expansion_maximum_usd": str(expansion_maximum),
            "test_maximum_usd": str(test_maximum),
            "maximum_new_cost_usd": str(maximum_new),
            "user_reported_available_credit_usd": "approximately 116",
            "runner_absolute_limit_usd": str(maximum_new),
        },
        "tasks": sorted(task_rows, key=lambda row: (row["stage"], row["task_id"])),
        "inputs": {
            name: _sha256(path) for name, path in sorted(input_paths.items())
        },
    }
    protocol["manifest_hash"] = stable_hash(protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_stage_v1.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-28_stage_v1.json"),
    )
    parser.add_argument(
        "--train-calibration-grades",
        type=Path,
        default=Path("outputs/router_baseline_v2/train_calibration_grades.json"),
    )
    parser.add_argument(
        "--pilot-grades",
        type=Path,
        default=Path("artifacts/pilot_coding_v6_swebench_grades.json"),
    )
    parser.add_argument(
        "--harness-prompt",
        type=Path,
        default=Path("configs/harness_prompt.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    args = parser.parse_args()
    input_paths = {
        "source_study": args.source_study,
        "tasks": args.tasks,
        "model_pool": args.model_pool,
        "prices": args.prices,
        "train_calibration_grades": args.train_calibration_grades,
        "pilot_grades": args.pilot_grades,
        "harness_prompt": args.harness_prompt,
    }
    protocol = build_protocol(
        _load(args.source_study),
        _load(args.tasks),
        _load(args.model_pool),
        _load(args.prices),
        [
            _load(args.train_calibration_grades),
            _load(args.pilot_grades),
        ],
        input_paths=input_paths,
    )
    serialized = stable_json(protocol) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("refusing to overwrite a different frozen study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
