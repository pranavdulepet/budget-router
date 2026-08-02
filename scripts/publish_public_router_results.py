from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.serialization import stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "macro_score",
            "micro_score",
            "macro_success_rate",
            "total_cost_usd",
            "cost_per_success_usd",
            "prompt_count",
            "route_counts",
        )
        if key in value
    }


def publish(
    study_path: Path,
    artifact_evaluation_path: Path,
    arc_path: Path,
    cheap_gate_path: Path,
    higher_gate_path: Path,
    collection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    study = _load(study_path)
    artifact_evaluation = _load(artifact_evaluation_path)
    arc = _load(arc_path)
    cheap = _load(cheap_gate_path)
    higher = _load(higher_gate_path)
    collection = _load(collection_path)
    public_tests: dict[str, Any] = {}
    for split in ("id_test", "ood_test"):
        test = study["test"][split]
        public_tests[split] = {
            "confirmatory_primary": {
                "method": test["primary"]["method"],
                "lambda": test["primary"]["lambda"],
                "metrics": _metrics(test["primary"]["metrics"]),
                "quality_difference_vs_calibration_best_fixed": test[
                    "primary"
                ]["macro_score_gain_vs_best_fixed"],
                "cost_saving_fraction_vs_calibration_best_fixed": test[
                    "primary"
                ]["cost_saving_fraction_vs_best_fixed"],
                "paired_bootstrap": test["paired_bootstrap_vs_best_fixed"],
            },
            "calibration_best_fixed": {
                "model": test["controls"]["calibration_best_fixed"]["model"],
                "metrics": _metrics(
                    test["controls"]["calibration_best_fixed"]["metrics"]
                ),
            },
            "posthoc_test_best_fixed": {
                "model": test["controls"]["posthoc_test_best_fixed"]["model"],
                "metrics": _metrics(
                    test["controls"]["posthoc_test_best_fixed"]["metrics"]
                ),
                "deployable_selection": False,
            },
            "hindsight_oracle": {
                "metrics": _metrics(test["controls"]["hindsight_oracle"]["metrics"]),
                "deployable": False,
            },
        }
    deployable_tests = {
        split: {
            "selected": _metrics(artifact_evaluation["test"][split]["selected"]),
            "calibration_best_fixed_model": artifact_evaluation["test"][split][
                "calibration_best_fixed_model"
            ],
            "quality_difference": artifact_evaluation["test"][split][
                "macro_score_difference"
            ],
            "cost_saving_fraction": artifact_evaluation["test"][split][
                "cost_saving_fraction"
            ],
        }
        for split in ("id_test", "ood_test")
    }
    report = {
        "schema_version": "public-router-v1-sanitized-result",
        "study_id": study["study_id"],
        "inferential_summary": {
            "confirmatory_primary": "failed_on_id_test",
            "deployable_shared_classifier": (
                "prespecified_curve_point_selected_after_test_secondary_evidence"
            ),
            "arc_agi": "failed_sequential_external_replication",
        },
        "public_matrix": study["matrix"],
        "prepared_manifest_hash": study["prepared_manifest_hash"],
        "study_results_hash": study["results_hash"],
        "router_freeze_hash": study["router_freeze_hash"],
        "public_tests": public_tests,
        "deployable_artifact": {
            "artifact_hash": artifact_evaluation["artifact_hash"],
            "method": "shared_task_model_hashed_classifier",
            "lambda": artifact_evaluation["default_lambda"],
            "tests": deployable_tests,
        },
        "arc_agi_sequential_replication": {
            "inferential_status": arc["inferential_status"],
            "prompt_count": arc["prompt_count"],
            "selected": _metrics(arc["selected"]),
            "primary_fixed_model": arc["primary_fixed_model"],
            "primary_fixed": _metrics(arc["primary_fixed"]),
            "quality_difference": arc["quality_difference_vs_primary_fixed"],
            "cost_saving_fraction": arc[
                "cost_saving_fraction_vs_primary_fixed"
            ],
            "paired_bootstrap": arc["paired_bootstrap_vs_primary_fixed"],
            "posthoc_best_fixed_model": arc[
                "descriptive_posthoc_best_fixed_model"
            ],
            "manifest_hash": arc["manifest_hash"],
        },
        "paid_agent_screen": {
            "total_incremental_exposure_usd": collection[
                "total_incremental_exposure_usd"
            ],
            "stopped_for_budget": collection["stopped_for_budget"],
            "cheap_medium_models": cheap["models"],
            "higher_cost_models": higher["models"],
            "eligible_families": higher["eligible_families_after_both_screens"],
            "required_minimum_families": higher["required_minimum_families"],
            "matrix_expansion_stopped": higher[
                "tinker_matrix_expansion_stopped"
            ],
            "stop_rule": higher["stop_rule"],
        },
    }
    findings = scan_for_secrets(report)
    if findings:
        raise ValueError("refusing to publish a possible credential")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a small sanitized summary of the public router study."
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("outputs/public_router_v1/study/results.json"),
    )
    parser.add_argument(
        "--artifact-evaluation",
        type=Path,
        default=Path("outputs/public_router_v1/open_source/evaluation.json"),
    )
    parser.add_argument(
        "--arc",
        type=Path,
        default=Path("outputs/public_router_v1/arc_agi/results.json"),
    )
    parser.add_argument(
        "--cheap-gate",
        type=Path,
        default=Path("outputs/active_router_v1/cheap_medium_screen_gate.json"),
    )
    parser.add_argument(
        "--higher-gate",
        type=Path,
        default=Path("outputs/active_router_v1/higher_cost_screen_gate.json"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("outputs/active_router_v1/collection_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/public_router_v1_results.json"),
    )
    args = parser.parse_args()
    report = publish(
        args.study,
        args.artifact_evaluation,
        args.arc,
        args.cheap_gate,
        args.higher_gate,
        args.collection,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_hash": report["deployable_artifact"]["artifact_hash"],
                "id_score": report["deployable_artifact"]["tests"]["id_test"][
                    "selected"
                ]["macro_score"],
                "id_cost_saving_fraction": report["deployable_artifact"]["tests"][
                    "id_test"
                ]["cost_saving_fraction"],
                "arc_quality_difference": report[
                    "arc_agi_sequential_replication"
                ]["quality_difference"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
