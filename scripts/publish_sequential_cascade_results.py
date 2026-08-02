from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import stable_hash, stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_public_result(
    protocol: dict[str, Any],
    evaluation: dict[str, Any],
    grades: dict[str, Any],
    collection: dict[str, Any],
    official_report: dict[str, Any],
    *,
    source_sha256: dict[str, str],
    dataset_snapshot_sha256: str,
) -> dict[str, Any]:
    protocol_without_hash = dict(protocol)
    protocol_hash = str(protocol_without_hash.pop("manifest_hash", ""))
    if stable_hash(protocol_without_hash) != protocol_hash:
        raise ValueError("cascade protocol hash is invalid")
    evaluation_without_hash = dict(evaluation)
    evaluation_hash = str(evaluation_without_hash.pop("evaluation_hash", ""))
    if stable_hash(evaluation_without_hash) != evaluation_hash:
        raise ValueError("cascade evaluation hash is invalid")
    if (
        evaluation["study_manifest_hash"] != protocol_hash
        or collection["study_manifest_hash"] != protocol_hash
        or evaluation["study_id"] != protocol["study_id"]
        or collection["study_id"] != protocol["study_id"]
    ):
        raise ValueError("cascade artifacts do not share one frozen study")

    policy_id = str(protocol["cascade_policy"]["policy_id"])
    grade_summary = grades["models"].get(policy_id)
    if grade_summary is None:
        raise ValueError("cascade policy is absent from official grades")
    expected = {
        "submitted_instances": int(grade_summary["submitted_count"]),
        "resolved_instances": int(grade_summary["resolved_count"]),
        "error_instances": int(grade_summary["grader_error_count"]),
    }
    for field, value in expected.items():
        if int(official_report.get(field, -1)) != value:
            raise ValueError(f"official report disagrees on {field}")
    if (
        int(collection["completed_episodes"]) != evaluation["task_count"]
        or int(collection["submitted_episodes"])
        != evaluation["cascade"]["submitted_count"]
        or Decimal(str(collection["cascade_conservative_cost_usd"]))
        != Decimal(str(evaluation["cascade"]["total_cost_usd"]))
    ):
        raise ValueError("collection summary disagrees with the evaluation")

    prior = Decimal(str(protocol["budget"]["recorded_prior_exposure_usd"]))
    abandoned = Decimal(
        str(protocol["budget"]["abandoned_provider_hang_reserve_usd"])
    )
    cascade_cost = Decimal(str(evaluation["cascade"]["total_cost_usd"]))
    total_exposure = prior + abandoned + cascade_cost
    recorded_total = Decimal(
        str(collection["total_exposure_with_prior_and_hang_reserve_usd"])
    )
    if recorded_total != total_exposure:
        raise ValueError("total exposure does not reconcile")

    scout_cost = Decimal(str(evaluation["cascade"]["scout_total_cost_usd"]))
    result = {
        "schema_version": "sequential-cascade-v1-public-results-v1",
        "study": {
            "study_id": protocol["study_id"],
            "manifest_hash": protocol_hash,
            "design_timing": protocol["design_timing"],
            "evidence_status": evaluation["evidence_status"],
            "dataset": grades["grader"]["dataset"],
            "dataset_revision": grades["grader"]["dataset_revision"],
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "official_harness_commit": grades["grader"]["harness_commit"],
            "task_count": evaluation["task_count"],
            "paired_with_existing_fixed_comparator": True,
        },
        "policy": protocol["cascade_policy"],
        "official_evaluation": {
            "submitted_patches_graded": grade_summary["graded_count"],
            "grader_error_count": grade_summary["grader_error_count"],
            "official_report_sha256": source_sha256["official_report"],
            "grades_sha256": source_sha256["grades"],
            "evaluation_sha256": source_sha256["evaluation"],
            "evaluation_hash": evaluation_hash,
        },
        "results": {
            "fixed_comparator": evaluation["fixed_comparator"],
            "cascade": evaluation["cascade"],
            "paired_comparison": evaluation["paired_comparison"],
            "cost_preserving_policy_oracle": evaluation[
                "cost_preserving_policy_oracle"
            ],
            "frozen_success_rule_passed": evaluation[
                "frozen_success_rule_passed"
            ],
            "task_results": evaluation["task_results"],
        },
        "phase_behavior": {
            "handoff_episodes": collection["handoff_episodes"],
            "scout_early_submissions": collection["scout_early_submissions"],
            "scout_total_calls": evaluation["cascade"]["scout_total_calls"],
            "finisher_total_calls": evaluation["cascade"][
                "finisher_total_calls"
            ],
            "scout_total_cost_usd": str(scout_cost),
            "finisher_total_cost_usd": evaluation["cascade"][
                "finisher_total_cost_usd"
            ],
            "scout_share_of_cascade_cost": float(scout_cost / cascade_cost),
        },
        "cost_accounting": {
            "recorded_prior_exposure_usd": str(prior),
            "abandoned_provider_hang_reserve_usd": str(abandoned),
            "cascade_conservative_cost_usd": str(cascade_cost),
            "reserve_adjusted_total_exposure_usd": str(total_exposure),
            "maximum_new_cascade_cost_usd": protocol["budget"][
                "maximum_new_cascade_cost_usd"
            ],
            "working_limit_usd": protocol["budget"]["working_limit_usd"],
            "absolute_limit_usd": protocol["budget"]["absolute_limit_usd"],
        },
        "conclusion": {
            "cascade_beats_fixed_under_frozen_success_rule": bool(
                evaluation["frozen_success_rule_passed"]
            ),
            "cascade_total_cost_reduction_usd": str(
                Decimal(evaluation["fixed_comparator"]["total_cost_usd"])
                - cascade_cost
            ),
            "cascade_resolved_count_delta": evaluation["paired_comparison"][
                "resolved_count_delta"
            ],
            "observed_policy_routing_opportunity": (
                evaluation["cost_preserving_policy_oracle"][
                    "relative_cost_reduction_vs_fixed"
                ]
                > 0
            ),
            "deployable_stage_gate_trained": False,
            "claim_scope": (
                "This secondary exploratory, custom, budgeted, single-seed "
                "paired experiment is not a public SWE-bench leaderboard score. "
                "The hindsight policy oracle is an upper bound, not a trained "
                "or deployable router."
            ),
        },
        "source_sha256": source_sha256,
    }
    result["result_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/sequential_cascade_study_v1.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/evaluation.json"),
    )
    parser.add_argument(
        "--grades",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/grades.json"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/collection_summary.json"),
    )
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument(
        "--dataset-snapshot-sha256",
        default=(
            "545d417f42f5bff2112f8a4d7b283d51fd23c17d0805fc910046732c167a045c"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sequential_cascade_v1_results.json"),
    )
    args = parser.parse_args()
    paths = {
        "protocol": args.protocol,
        "evaluation": args.evaluation,
        "grades": args.grades,
        "collection": args.collection,
        "official_report": args.official_report,
    }
    result = build_public_result(
        _load(args.protocol),
        _load(args.evaluation),
        _load(args.grades),
        _load(args.collection),
        _load(args.official_report),
        source_sha256={name: _sha256(path) for name, path in paths.items()},
        dataset_snapshot_sha256=args.dataset_snapshot_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
