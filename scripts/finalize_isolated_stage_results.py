from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from budget_router.serialization import read_jsonl, stable_json
try:
    from scripts.aggregate_swebench_pilot_grades import aggregate_grades
    from scripts.evaluate_isolated_stage_gate import evaluate_gate
    from scripts.publish_isolated_stage_results import build_public_result
except ModuleNotFoundError:
    from aggregate_swebench_pilot_grades import aggregate_grades
    from evaluate_isolated_stage_gate import evaluate_gate
    from publish_isolated_stage_results import build_public_result


DATASET_REVISION = "91aa3ed51b709be6457e12d00300a6a596d4c6a3"
HARNESS_COMMIT = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile official grades, evaluate the frozen isolated-stage "
            "gate, and emit sanitized public artifacts."
        )
    )
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
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes_canonical.jsonl"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("outputs/isolated_stage_v1/collection_summary.json"),
    )
    parser.add_argument(
        "--grader-dir",
        type=Path,
        default=Path("outputs/isolated_stage_v1/swebench_grader/test"),
    )
    parser.add_argument(
        "--grading-manifest",
        type=Path,
        default=Path(
            "outputs/isolated_stage_v1/swebench_grader/test/grading_manifest.json"
        ),
    )
    parser.add_argument(
        "--grades-output",
        type=Path,
        default=Path("outputs/isolated_stage_v1/test_grades.json"),
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=Path("outputs/isolated_stage_v1/test_evaluation.json"),
    )
    parser.add_argument(
        "--public-result-output",
        type=Path,
        default=Path("artifacts/isolated_stage_v1_results.json"),
    )
    parser.add_argument(
        "--public-gate-output",
        type=Path,
        default=Path("artifacts/isolated_stage_v1_task_gate.json"),
    )
    args = parser.parse_args()

    grading_manifest = _load(args.grading_manifest)
    if grading_manifest.get("complete_with_no_unclassified_errors") is not True:
        raise ValueError("official grading still contains an unclassified error")
    if grading_manifest.get("harness_commit") != HARNESS_COMMIT:
        raise ValueError("grading manifest uses the wrong harness commit")
    error_keys = {
        (str(row["model"]), str(row["task_id"]))
        for row in grading_manifest["classified_model_caused_errors"]
    }
    grades = aggregate_grades(
        args.episodes,
        args.grader_dir,
        dataset_revision=DATASET_REVISION,
        harness_commit=HARNESS_COMMIT,
        study_stage="test",
        model_caused_error_keys=error_keys,
    )
    args.grades_output.parent.mkdir(parents=True, exist_ok=True)
    args.grades_output.write_text(
        json.dumps(grades, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gate_bytes = args.gate.read_bytes()
    episodes = read_jsonl(args.episodes)
    evaluation = evaluate_gate(
        _load(args.protocol),
        _load(args.gate_freeze),
        json.loads(gate_bytes),
        _load(args.tasks),
        [grades],
        artifact_bytes=gate_bytes,
        episodes=episodes,
    )
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.write_text(
        stable_json(evaluation) + "\n",
        encoding="utf-8",
    )

    report_paths = {
        model: Path(str(model_result["report_path"]))
        for model, model_result in grading_manifest["models"].items()
    }
    official_reports = {
        model: _load(path) for model, path in report_paths.items()
    }
    sources = {
        "protocol": args.protocol,
        "candidate_freeze": args.candidate_freeze,
        "gate_freeze": args.gate_freeze,
        "gate": args.gate,
        "collection": args.collection,
        "grades": args.grades_output,
        "evaluation": args.evaluation_output,
        "grading_manifest": args.grading_manifest,
        **{
            f"official_report:{model}": path
            for model, path in report_paths.items()
        },
    }
    public_result = build_public_result(
        _load(args.protocol),
        _load(args.candidate_freeze),
        _load(args.gate_freeze),
        json.loads(gate_bytes),
        _load(args.collection),
        grades,
        evaluation,
        grading_manifest,
        official_reports,
        gate_artifact_bytes=gate_bytes,
        source_sha256={name: _sha256(path) for name, path in sources.items()},
    )
    args.public_result_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_result_output.write_text(
        stable_json(public_result) + "\n",
        encoding="utf-8",
    )
    args.public_gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_gate_output.write_bytes(gate_bytes)
    print(
        json.dumps(
            {
                "grades": str(args.grades_output),
                "evaluation": str(args.evaluation_output),
                "public_result": str(args.public_result_output),
                "public_gate": str(args.public_gate_output),
                "policies": evaluation["policies"],
                "paired_comparisons": evaluation["paired_comparisons"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
