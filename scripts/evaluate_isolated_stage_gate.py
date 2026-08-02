from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import stable_hash, stable_json
from budget_router.task_model import HashedLinearModelHead
from budget_router.types import GoalContext, RouterState

try:
    from scripts.evaluate_frozen_task_router import (
        _bootstrap_deltas,
        _paired_exact_comparison,
        _wilson_interval,
    )
except ModuleNotFoundError:
    from evaluate_frozen_task_router import (  # type: ignore[no-redef]
        _bootstrap_deltas,
        _paired_exact_comparison,
        _wilson_interval,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grade_rows(
    payloads: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["task_results"]:
            key = (str(row["task_id"]), str(row["model"]))
            if key in rows:
                raise ValueError(f"duplicate official test grade: {key}")
            rows[key] = dict(row)
    return rows


def _policy_summary(
    successes: list[bool],
    costs: list[Decimal],
) -> dict[str, Any]:
    resolved = sum(successes)
    total = sum(costs, Decimal("0"))
    return {
        "task_count": len(successes),
        "resolved_count": resolved,
        "resolution_rate": resolved / len(successes),
        "resolution_rate_95_interval": _wilson_interval(
            resolved,
            len(successes),
        ),
        "total_cost_usd": str(total),
        "mean_cost_usd": str(total / len(successes)),
        "cost_per_resolved_task_usd": (
            str(total / resolved) if resolved else None
        ),
    }


def evaluate_gate(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
    artifact: dict[str, Any],
    tasks_payload: dict[str, Any],
    grade_payloads: Iterable[dict[str, Any]],
    *,
    artifact_bytes: bytes,
    episodes: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grade_payloads = list(grade_payloads)
    if freeze.get("study_manifest_hash") != protocol["manifest_hash"]:
        raise ValueError("gate freeze belongs to a different study")
    if hashlib.sha256(artifact_bytes).hexdigest() != freeze["router_artifact_sha256"]:
        raise ValueError("gate artifact differs from the pre-test freeze")
    if artifact.get("artifact_hash") != freeze["router_artifact_hash"]:
        raise ValueError("gate artifact hash differs from the pre-test freeze")
    candidate = str(freeze["selected_candidate_policy_id"])
    fixed = str(protocol["fixed_policy"]["policy_id"])
    if artifact.get("selected_candidate_policy_id") != candidate:
        raise ValueError("gate artifact and freeze disagree on the candidate")
    test_ids = [str(task_id) for task_id in protocol["stages"]["test"]["task_ids"]]
    if len(test_ids) != 39 or len(set(test_ids)) != 39:
        raise ValueError("expected a frozen 39-task test cohort")
    tasks = {
        str(task["instance_id"]): dict(task)
        for task in tasks_payload["records"]
    }
    grades = _grade_rows(grade_payloads)
    expected = {
        (task_id, policy_id)
        for task_id in test_ids
        for policy_id in (candidate, fixed)
    }
    if set(grades) != expected:
        missing = sorted(expected - set(grades))
        extra = sorted(set(grades) - expected)
        raise ValueError(f"paired test grades are incomplete: missing={missing}, extra={extra}")
    if any(row["status"] == "grader_error" for row in grades.values()):
        raise ValueError("paired test grades contain a grader error")

    head = HashedLinearModelHead.from_artifact(artifact["model_head"])
    selector = artifact["selector"]
    cost_weight = float(selector["cost_weight"])
    threshold = float(selector["threshold"])
    expected_saving_ratio = float(selector["expected_mean_saving_ratio"])
    fixed_success: list[bool] = []
    candidate_success: list[bool] = []
    fixed_cost: list[Decimal] = []
    candidate_cost: list[Decimal] = []
    gate_success: list[bool] = []
    gate_cost: list[Decimal] = []
    oracle_success: list[bool] = []
    oracle_cost: list[Decimal] = []
    oracle_selections: list[str] = []
    selections: list[dict[str, Any]] = []
    for task_id in test_ids:
        problem = str(tasks[task_id]["problem_statement"])
        goal = GoalContext(
            problem,
            str(tasks[task_id]["repo"]),
            ("candidate_win", "fixed_loss"),
        )
        probabilities = {
            name: head.estimate(goal, RouterState(), name).success_probability
            for name in ("candidate_win", "fixed_loss")
        }
        score = (
            probabilities["candidate_win"]
            - probabilities["fixed_loss"]
            + cost_weight * expected_saving_ratio
        )
        selected_candidate = score >= threshold
        candidate_row = grades[(task_id, candidate)]
        fixed_row = grades[(task_id, fixed)]
        candidate_ok = candidate_row["status"] == "resolved"
        fixed_ok = fixed_row["status"] == "resolved"
        candidate_episode_cost = Decimal(str(candidate_row["conservative_cost_usd"]))
        fixed_episode_cost = Decimal(str(fixed_row["conservative_cost_usd"]))
        selected_ok = candidate_ok if selected_candidate else fixed_ok
        selected_cost = (
            candidate_episode_cost if selected_candidate else fixed_episode_cost
        )
        if candidate_ok or fixed_ok:
            if candidate_ok and (
                not fixed_ok or candidate_episode_cost < fixed_episode_cost
            ):
                oracle_selected_policy = candidate
                oracle_episode_cost = candidate_episode_cost
            else:
                oracle_selected_policy = fixed
                oracle_episode_cost = fixed_episode_cost
            oracle_ok = True
        else:
            if candidate_episode_cost < fixed_episode_cost:
                oracle_selected_policy = candidate
                oracle_episode_cost = candidate_episode_cost
            else:
                oracle_selected_policy = fixed
                oracle_episode_cost = fixed_episode_cost
            oracle_ok = False
        fixed_success.append(fixed_ok)
        candidate_success.append(candidate_ok)
        fixed_cost.append(fixed_episode_cost)
        candidate_cost.append(candidate_episode_cost)
        gate_success.append(selected_ok)
        gate_cost.append(selected_cost)
        oracle_success.append(oracle_ok)
        oracle_cost.append(oracle_episode_cost)
        oracle_selections.append(oracle_selected_policy)
        selections.append(
            {
                "task_id": task_id,
                "candidate_win_probability": probabilities["candidate_win"],
                "fixed_loss_probability": probabilities["fixed_loss"],
                "score": score,
                "selected_policy_id": candidate if selected_candidate else fixed,
                "resolved": selected_ok,
                "conservative_cost_usd": str(selected_cost),
                "candidate_resolved": candidate_ok,
                "fixed_resolved": fixed_ok,
                "candidate_cost_usd": str(candidate_episode_cost),
                "fixed_cost_usd": str(fixed_episode_cost),
                "candidate_cost_delta_vs_fixed_usd": str(
                    candidate_episode_cost - fixed_episode_cost
                ),
                "post_hoc_oracle_policy_id": oracle_selected_policy,
            }
        )

    policies = {
        "fixed": _policy_summary(fixed_success, fixed_cost),
        "universal_candidate": _policy_summary(
            candidate_success,
            candidate_cost,
        ),
        "learned_gate": {
            **_policy_summary(gate_success, gate_cost),
            "candidate_selection_count": sum(
                row["selected_policy_id"] == candidate for row in selections
            ),
            "fixed_selection_count": sum(
                row["selected_policy_id"] == fixed for row in selections
            ),
        },
        "post_hoc_task_oracle": {
            **_policy_summary(oracle_success, oracle_cost),
            "candidate_selection_count": oracle_selections.count(candidate),
            "fixed_selection_count": oracle_selections.count(fixed),
            "deployable": False,
        },
    }
    result = {
        "schema_version": "isolated-stage-gate-evaluation-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "gate_freeze_hash": freeze["freeze_hash"],
        "test_task_count": len(test_ids),
        "candidate_policy_id": candidate,
        "fixed_policy_id": fixed,
        "policies": policies,
        "paired_comparisons": {
            "learned_gate_vs_fixed": {
                **_paired_exact_comparison(gate_success, fixed_success),
                "resolved_count_delta": sum(gate_success) - sum(fixed_success),
                "total_cost_delta_usd": str(
                    sum(gate_cost, Decimal("0"))
                    - sum(fixed_cost, Decimal("0"))
                ),
                **_bootstrap_deltas(
                    gate_success,
                    fixed_success,
                    gate_cost,
                    fixed_cost,
                    seed=20260728,
                ),
            },
            "universal_candidate_vs_fixed": {
                **_paired_exact_comparison(candidate_success, fixed_success),
                "resolved_count_delta": (
                    sum(candidate_success) - sum(fixed_success)
                ),
                "total_cost_delta_usd": str(
                    sum(candidate_cost, Decimal("0"))
                    - sum(fixed_cost, Decimal("0"))
                ),
                **_bootstrap_deltas(
                    candidate_success,
                    fixed_success,
                    candidate_cost,
                    fixed_cost,
                    seed=20260728,
                ),
            },
            "learned_gate_vs_universal_candidate": {
                **_paired_exact_comparison(gate_success, candidate_success),
                "resolved_count_delta": (
                    sum(gate_success) - sum(candidate_success)
                ),
                "total_cost_delta_usd": str(
                    sum(gate_cost, Decimal("0"))
                    - sum(candidate_cost, Decimal("0"))
                ),
            },
        },
        "selections": selections,
        "provenance": {
            "test_labels_accessed_after_freeze": True,
            "grade_payload_hashes": [
                stable_hash(payload) for payload in grade_payloads
            ],
        },
    }
    if episodes is not None:
        candidate_episodes: dict[str, dict[str, Any]] = {}
        for episode in episodes:
            if (
                episode.get("study_stage") != "test"
                or episode.get("model") != candidate
            ):
                continue
            task_id = str(episode["task_id"])
            if task_id in candidate_episodes:
                raise ValueError(f"duplicate candidate test episode: {task_id}")
            candidate_episodes[task_id] = dict(episode)
        if set(candidate_episodes) != set(test_ids):
            raise ValueError("candidate phase episodes do not match held-out test")
        scout_cost = sum(
            (
                Decimal(str(candidate_episodes[task_id]["scout_cost_usd"]))
                for task_id in test_ids
            ),
            Decimal("0"),
        )
        finisher_cost = sum(
            (
                Decimal(str(candidate_episodes[task_id]["finisher_cost_usd"]))
                for task_id in test_ids
            ),
            Decimal("0"),
        )
        candidate_total = sum(candidate_cost, Decimal("0"))
        if scout_cost + finisher_cost != candidate_total:
            raise ValueError("candidate phase costs do not reconcile")
        candidate_cheaper = [
            left < right
            for left, right in zip(candidate_cost, fixed_cost, strict=True)
        ]
        result["candidate_phase_behavior"] = {
            "episode_count": len(test_ids),
            "structurally_valid_count": sum(
                bool(candidate_episodes[task_id]["structurally_valid"])
                for task_id in test_ids
            ),
            "submitted_count": sum(
                bool(candidate_episodes[task_id]["submitted_patch"])
                for task_id in test_ids
            ),
            "scout_total_calls": sum(
                int(candidate_episodes[task_id]["scout_calls"])
                for task_id in test_ids
            ),
            "finisher_total_calls": sum(
                int(candidate_episodes[task_id]["finisher_calls"])
                for task_id in test_ids
            ),
            "scout_total_cost_usd": str(scout_cost),
            "finisher_total_cost_usd": str(finisher_cost),
            "scout_share_of_candidate_cost": (
                float(scout_cost / candidate_total) if candidate_total else None
            ),
            "candidate_cheaper_task_count": sum(candidate_cheaper),
            "candidate_cheaper_or_equal_task_count": sum(
                left <= right
                for left, right in zip(candidate_cost, fixed_cost, strict=True)
            ),
            "candidate_cheaper_and_same_or_better_outcome_count": sum(
                cheaper and (candidate_ok or not fixed_ok)
                for cheaper, candidate_ok, fixed_ok in zip(
                    candidate_cheaper,
                    candidate_success,
                    fixed_success,
                    strict=True,
                )
            ),
            "candidate_cheaper_and_both_resolved_count": sum(
                cheaper and candidate_ok and fixed_ok
                for cheaper, candidate_ok, fixed_ok in zip(
                    candidate_cheaper,
                    candidate_success,
                    fixed_success,
                    strict=True,
                )
            ),
            "visible_handoff_total_chars": sum(
                int(candidate_episodes[task_id]["visible_handoff_chars"])
                for task_id in test_ids
            ),
        }
    result["evaluation_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/gate_freeze.json"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("outputs/isolated_stage_v1/task_gate.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument("--grades", type=Path, action="append", required=True)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes_canonical.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isolated_stage_v1/test_evaluation.json"),
    )
    args = parser.parse_args()
    artifact_bytes = args.gate.read_bytes()
    grade_payloads = [_load(path) for path in args.grades]
    result = evaluate_gate(
        _load(args.protocol),
        _load(args.freeze),
        json.loads(artifact_bytes),
        _load(args.tasks),
        grade_payloads,
        artifact_bytes=artifact_bytes,
        episodes=(
            json.loads(line)
            for line in args.episodes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
