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


def _validate_stable_hash(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> str:
    unhashed = dict(payload)
    claimed = str(unhashed.pop(field, ""))
    if stable_hash(unhashed) != claimed:
        raise ValueError(f"{label} hash is invalid")
    return claimed


def _policy_wins(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_resolved = int(left["resolved_count"])
    right_resolved = int(right["resolved_count"])
    return left_resolved > right_resolved or (
        left_resolved == right_resolved
        and Decimal(str(left["total_cost_usd"]))
        < Decimal(str(right["total_cost_usd"]))
    )


def build_public_result(
    protocol: dict[str, Any],
    candidate_freeze: dict[str, Any],
    gate_freeze: dict[str, Any],
    gate_artifact: dict[str, Any],
    collection: dict[str, Any],
    grades: dict[str, Any],
    evaluation: dict[str, Any],
    grading_manifest: dict[str, Any],
    official_reports: dict[str, dict[str, Any]],
    *,
    gate_artifact_bytes: bytes,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    manifest_hash = _validate_stable_hash(
        protocol, "manifest_hash", "study manifest"
    )
    gate_artifact_hash = _validate_stable_hash(
        gate_artifact, "artifact_hash", "gate artifact"
    )
    evaluation_hash = _validate_stable_hash(
        evaluation, "evaluation_hash", "evaluation"
    )
    if hashlib.sha256(gate_artifact_bytes).hexdigest() != gate_freeze[
        "router_artifact_sha256"
    ]:
        raise ValueError("gate artifact bytes differ from the pre-test freeze")
    if gate_artifact_hash != gate_freeze["router_artifact_hash"]:
        raise ValueError("gate artifact hash differs from the pre-test freeze")

    study_id = str(protocol["study_id"])
    linked = (
        candidate_freeze,
        gate_freeze,
        collection,
        evaluation,
    )
    if any(
        payload.get("study_manifest_hash") != manifest_hash
        or payload.get("study_id") != study_id
        for payload in linked
    ):
        raise ValueError("isolated-stage artifacts do not share one frozen study")
    if candidate_freeze.get("test_outcomes_accessed") is not False:
        raise ValueError("candidate was not frozen before test outcomes")
    if gate_freeze.get("test_labels_accessed") is not False:
        raise ValueError("gate was not frozen before test labels")
    if evaluation.get("provenance", {}).get(
        "test_labels_accessed_after_freeze"
    ) is not True:
        raise ValueError("evaluation does not attest post-freeze test access")

    candidate = str(candidate_freeze["selected_policy_id"])
    fixed = str(protocol["fixed_policy"]["policy_id"])
    if (
        candidate != gate_freeze["selected_candidate_policy_id"]
        or candidate != evaluation["candidate_policy_id"]
        or fixed != evaluation["fixed_policy_id"]
    ):
        raise ValueError("candidate/fixed policy identity mismatch")

    if int(collection["completed_stage_episodes"]) != 78:
        raise ValueError("held-out collection does not contain 78 paired episodes")
    if int(collection["submitted_stage_episodes"]) != int(
        grades["overall"]["submitted_count"]
    ):
        raise ValueError("collection and official grades disagree on submissions")
    if Decimal(str(collection["stage_conservative_cost_usd"])) != Decimal(
        str(grades["overall"]["total_conservative_cost_usd"])
    ):
        raise ValueError("collection and official grades disagree on test cost")
    if grading_manifest.get("complete_with_no_unclassified_errors") is not True:
        raise ValueError("official grading contains an unclassified error")

    for policy in (candidate, fixed):
        grade_summary = grades["models"][policy]
        report = official_reports[policy]
        expected = {
            "submitted_instances": int(grade_summary["submitted_count"]),
            "completed_instances": (
                int(grade_summary["graded_count"])
                - int(grade_summary["grader_error_count"])
                - int(grade_summary["unapplyable_patch_count"])
            ),
            "resolved_instances": int(grade_summary["resolved_count"]),
            "error_instances": (
                int(grade_summary["grader_error_count"])
                + int(grade_summary["unapplyable_patch_count"])
            ),
        }
        for field, value in expected.items():
            if int(report.get(field, -1)) != value:
                raise ValueError(
                    f"official report for {policy} disagrees on {field}"
                )

    fixed_result = evaluation["policies"]["fixed"]
    candidate_result = evaluation["policies"]["universal_candidate"]
    gate_result = evaluation["policies"]["learned_gate"]
    oracle_result = evaluation["policies"]["post_hoc_task_oracle"]
    if int(grades["models"][fixed]["resolved_count"]) != int(
        fixed_result["resolved_count"]
    ):
        raise ValueError("fixed official grades disagree with paired evaluation")
    if int(grades["models"][candidate]["resolved_count"]) != int(
        candidate_result["resolved_count"]
    ):
        raise ValueError(
            "candidate official grades disagree with paired evaluation"
        )
    screen_cost = sum(
        (
            Decimal(str(row["total_conservative_cost_usd"]))
            for row in candidate_freeze["candidate_metrics"]
        ),
        Decimal("0"),
    )
    total_exposure = Decimal(str(collection["total_incremental_exposure_usd"]))
    test_cost = Decimal(str(collection["stage_conservative_cost_usd"]))
    pre_test_cost = total_exposure - test_cost

    result = {
        "schema_version": "isolated-stage-router-v1-public-results-v1",
        "study": {
            "study_id": study_id,
            "manifest_hash": manifest_hash,
            "design_timing": protocol["design_timing"],
            "research_question": protocol["research_question"],
            "dataset": grades["grader"]["dataset"],
            "dataset_revision": grades["grader"]["dataset_revision"],
            "dataset_snapshot_sha256": grading_manifest["dataset_sha256"],
            "official_harness_commit": grades["grader"]["harness_commit"],
            "gate_training_task_count": protocol["sampling"][
                "gate_training_task_count"
            ],
            "held_out_task_count": evaluation["test_task_count"],
            "repository_disjoint_test": protocol["sampling"][
                "repository_disjoint_test"
            ],
        },
        "policies": {
            "fixed": protocol["fixed_policy"],
            "selected_candidate": next(
                policy
                for policy in protocol["candidate_policies"]
                if policy["policy_id"] == candidate
            ),
            "execution_policy": protocol["execution_policy"],
        },
        "freezes": {
            "candidate_freeze_hash": candidate_freeze["freeze_hash"],
            "gate_freeze_hash": gate_freeze["freeze_hash"],
            "gate_artifact_hash": gate_artifact_hash,
            "gate_artifact_sha256": gate_freeze["router_artifact_sha256"],
            "test_outcomes_accessed_before_candidate_freeze": False,
            "test_labels_accessed_before_gate_freeze": False,
        },
        "screen": {
            "task_count": len(candidate_freeze["screen_task_ids"]),
            "candidate_metrics": candidate_freeze["candidate_metrics"],
            "selected_policy_id": candidate,
            "recorded_cost_usd": str(screen_cost),
        },
        "gate_training": {
            "training": gate_artifact["training"],
            "cross_validation": gate_artifact["cross_validation"],
            "selector": gate_artifact["selector"],
        },
        "official_evaluation": {
            "submitted_patches_graded": grades["grader"][
                "submitted_patches_graded"
            ],
            "grader_error_count": grades["grader"]["grader_errors"],
            "unapplyable_model_patches": grades["grader"][
                "unapplyable_model_patches"
            ],
            "classified_model_caused_errors": grading_manifest[
                "classified_model_caused_errors"
            ],
            "grading_manifest_sha256": source_sha256["grading_manifest"],
            "grades_sha256": source_sha256["grades"],
            "evaluation_sha256": source_sha256["evaluation"],
            "evaluation_hash": evaluation_hash,
            "official_report_sha256": {
                policy: source_sha256[f"official_report:{policy}"]
                for policy in (candidate, fixed)
            },
        },
        "results": {
            "fixed": fixed_result,
            "universal_candidate": candidate_result,
            "learned_gate": gate_result,
            "post_hoc_task_oracle": oracle_result,
            "paired_comparisons": evaluation["paired_comparisons"],
            "selections": evaluation["selections"],
            "candidate_phase_behavior": evaluation[
                "candidate_phase_behavior"
            ],
        },
        "cost_accounting": {
            "pre_test_screen_and_expansion_cost_usd": str(pre_test_cost),
            "held_out_test_cost_usd": str(test_cost),
            "total_incremental_exposure_usd": str(total_exposure),
            "maximum_frozen_exposure_usd": protocol["budget"][
                "maximum_new_cost_usd"
            ],
            "unspent_vs_frozen_ceiling_usd": str(
                Decimal(str(protocol["budget"]["maximum_new_cost_usd"]))
                - total_exposure
            ),
        },
        "conclusion": {
            "learned_gate_beats_fixed": _policy_wins(gate_result, fixed_result),
            "universal_candidate_beats_fixed": _policy_wins(
                candidate_result, fixed_result
            ),
            "candidate_selections_by_learned_gate": gate_result[
                "candidate_selection_count"
            ],
            "observed_oracle_incremental_resolutions_over_fixed": (
                int(oracle_result["resolved_count"])
                - int(fixed_result["resolved_count"])
            ),
            "deployable_gate_trained_and_held_out_tested": True,
            "claim_scope": (
                "This custom, budgeted, single-seed, repository-disjoint "
                "paired experiment is not a public SWE-bench leaderboard "
                "score. The post-hoc task oracle is not deployable."
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
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--candidate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/candidate_freeze.json"),
    )
    parser.add_argument(
        "--gate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/gate_freeze.json"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("outputs/isolated_stage_v1/task_gate.json"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("outputs/isolated_stage_v1/collection_summary.json"),
    )
    parser.add_argument(
        "--grades",
        type=Path,
        default=Path("outputs/isolated_stage_v1/test_grades.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("outputs/isolated_stage_v1/test_evaluation.json"),
    )
    parser.add_argument(
        "--grading-manifest",
        type=Path,
        default=Path(
            "outputs/isolated_stage_v1/swebench_grader/test/grading_manifest.json"
        ),
    )
    parser.add_argument(
        "--official-report",
        type=Path,
        action="append",
        required=True,
        help="One official report per held-out policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/isolated_stage_v1_results.json"),
    )
    args = parser.parse_args()

    grades = _load(args.grades)
    reports_by_submitted_ids = [
        (path, _load(path)) for path in args.official_report
    ]
    official_reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    for policy, model_summary in grades["models"].items():
        expected = set(model_summary["resolved_task_ids"])
        matches = [
            (path, report)
            for path, report in reports_by_submitted_ids
            if expected <= set(report.get("submitted_ids", []))
            and int(report.get("submitted_instances", -1))
            == int(model_summary["submitted_count"])
        ]
        if len(matches) != 1:
            raise ValueError(f"could not uniquely match official report for {policy}")
        report_paths[policy], official_reports[policy] = matches[0]

    paths = {
        "protocol": args.protocol,
        "candidate_freeze": args.candidate_freeze,
        "gate_freeze": args.gate_freeze,
        "gate": args.gate,
        "collection": args.collection,
        "grades": args.grades,
        "evaluation": args.evaluation,
        "grading_manifest": args.grading_manifest,
    }
    paths.update(
        {
            f"official_report:{policy}": path
            for policy, path in report_paths.items()
        }
    )
    gate_bytes = args.gate.read_bytes()
    result = build_public_result(
        _load(args.protocol),
        _load(args.candidate_freeze),
        _load(args.gate_freeze),
        json.loads(gate_bytes),
        _load(args.collection),
        grades,
        _load(args.evaluation),
        _load(args.grading_manifest),
        official_reports,
        gate_artifact_bytes=gate_bytes,
        source_sha256={name: _sha256(path) for name, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
