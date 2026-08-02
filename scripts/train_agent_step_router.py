#!/usr/bin/env python3
"""Fit and freeze the public-data agent-step classifier before static test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from budget_router.agent_step import FrozenAgentStepArtifact, render_agent_prefix
from budget_router.agent_training import (
    evaluate_static_rows,
    fit_logistic_with_grouped_cv,
    fit_platt,
    load_twinrouter_rows,
    rows_for_split,
    select_conservative_threshold,
    sha256_file,
    split_instances,
)
from budget_router.serialization import stable_hash


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/active_router_protocol_amendment_007_agent_step_routing.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_step_router_v1"),
    )
    args = parser.parse_args()

    protocol = _load(args.protocol)
    source = protocol["public_training_source"]
    actual_question_hash = sha256_file(args.question_bank)
    actual_manifest_hash = sha256_file(args.source_manifest)
    if actual_question_hash != source["question_bank_sha256"]:
        raise SystemExit("TwinRouterBench question-bank hash mismatch")
    if actual_manifest_hash != source["manifest_sha256"]:
        raise SystemExit("TwinRouterBench manifest hash mismatch")

    output_paths = {
        "split": args.output_dir / "split_manifest.json",
        "artifact": args.output_dir / "router_artifact.json",
        "report": args.output_dir / "training_report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite frozen agent-step outputs: " + ", ".join(existing)
        )

    rows, source_counts = load_twinrouter_rows(
        args.question_bank,
        excluded_swe_repository_prefixes=source[
            "excluded_swe_repository_prefixes"
        ],
    )
    split_config = protocol["split"]
    split_ids = split_instances(
        rows,
        seed=str(split_config["seed"]),
        train_fraction=float(split_config["train_fraction"]),
        calibration_fraction=float(split_config["calibration_fraction"]),
    )
    row_counts = {
        name: len(rows_for_split(rows, instance_ids))
        for name, instance_ids in split_ids.items()
    }
    split_manifest: dict[str, Any] = {
        "schema_version": "agent-step-split-v1",
        "source_revision": source["revision"],
        "question_bank_sha256": actual_question_hash,
        "source_manifest_sha256": actual_manifest_hash,
        "split_seed": split_config["seed"],
        "source_counts": source_counts,
        "instance_ids": {
            name: list(instance_ids) for name, instance_ids in split_ids.items()
        },
        "instance_counts": {
            name: len(instance_ids) for name, instance_ids in split_ids.items()
        },
        "row_counts": row_counts,
    }
    split_manifest["manifest_hash"] = stable_hash(split_manifest)
    _write(output_paths["split"], split_manifest)

    train_rows = rows_for_split(rows, split_ids["train"])
    calibration_rows = rows_for_split(rows, split_ids["calibration"])
    classifier = protocol["classifier"]
    dimension = int(classifier["dimension"])
    hash_seed = str(classifier["hash_seed"])
    selected_c, bias, weights, cv_rows = fit_logistic_with_grouped_cv(
        train_rows,
        dimension=dimension,
        hash_seed=hash_seed,
        c_candidates=[
            float(value)
            for value in classifier["inverse_l2_strength_candidates"]
        ],
        fold_seed=str(split_config["seed"]),
    )

    uncalibrated = FrozenAgentStepArtifact(
        cheap_model=protocol["model_pool"]["cheap"],
        strong_model=protocol["model_pool"]["strong"],
        dimension=dimension,
        hash_seed=hash_seed,
        bias=bias,
        weights=weights,
        platt_slope=1.0,
        platt_intercept=0.0,
        cheap_threshold=0.0,
    )
    calibration_texts = [
        render_agent_prefix(row.messages, step_index=row.step_index)[0]
        for row in calibration_rows
    ]
    raw_probabilities = [
        uncalibrated.raw_probability_text(text) for text in calibration_texts
    ]
    platt_slope, platt_intercept = fit_platt(
        raw_probabilities,
        [row.needs_strong for row in calibration_rows],
    )
    calibrated_base = FrozenAgentStepArtifact(
        cheap_model=protocol["model_pool"]["cheap"],
        strong_model=protocol["model_pool"]["strong"],
        dimension=dimension,
        hash_seed=hash_seed,
        bias=bias,
        weights=weights,
        platt_slope=platt_slope,
        platt_intercept=platt_intercept,
        cheap_threshold=0.0,
        force_strong_context_tokens=int(
            protocol["runtime"]["force_strong_approximate_context_tokens"]
        ),
        strong_minimum_dwell_calls=int(
            protocol["runtime"]["strong_minimum_dwell_calls_before_deescalation"]
        ),
    )
    calibrated_probabilities = [
        calibrated_base.calibrated_probability_text(text)[1]
        for text in calibration_texts
    ]
    artifact_metadata = {
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "source": {
            "name": source["name"],
            "revision": source["revision"],
            "question_bank_sha256": actual_question_hash,
            "manifest_sha256": actual_manifest_hash,
        },
        "split_manifest_hash": split_manifest["manifest_hash"],
        "selected_C": selected_c,
        "cross_validation": cv_rows,
        "training_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "static_test_rows_sealed": row_counts["static_test"],
    }

    def artifact_factory(threshold: float) -> FrozenAgentStepArtifact:
        return FrozenAgentStepArtifact(
            cheap_model=protocol["model_pool"]["cheap"],
            strong_model=protocol["model_pool"]["strong"],
            dimension=dimension,
            hash_seed=hash_seed,
            bias=bias,
            weights=weights,
            platt_slope=platt_slope,
            platt_intercept=platt_intercept,
            cheap_threshold=threshold,
            force_strong_context_tokens=int(
                protocol["runtime"]["force_strong_approximate_context_tokens"]
            ),
            strong_minimum_dwell_calls=int(
                protocol["runtime"][
                    "strong_minimum_dwell_calls_before_deescalation"
                ]
            ),
            metadata=artifact_metadata,
        )

    threshold, threshold_report = select_conservative_threshold(
        artifact_factory=artifact_factory,
        calibration_rows=calibration_rows,
        calibrated_probabilities=calibrated_probabilities,
    )
    frozen = artifact_factory(threshold)
    artifact_payload = frozen.to_dict()
    _write(output_paths["artifact"], artifact_payload)

    # Re-load to prove the portable artifact hash and parser agree before any
    # static-test label is evaluated.
    reloaded = FrozenAgentStepArtifact.from_dict(_load(output_paths["artifact"]))
    training_metrics = evaluate_static_rows(reloaded, train_rows)
    calibration_metrics = evaluate_static_rows(reloaded, calibration_rows)
    report = {
        "schema_version": "agent-step-training-report-v1",
        "status": "artifact_frozen_static_test_unopened",
        "artifact_hash": reloaded.artifact_hash,
        "artifact_file_sha256": sha256_file(output_paths["artifact"]),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "selected_C": selected_c,
        "cross_validation": cv_rows,
        "platt": {
            "slope": platt_slope,
            "intercept": platt_intercept,
        },
        "threshold": threshold,
        "threshold_selection": threshold_report,
        "training_metrics": training_metrics,
        "calibration_metrics": calibration_metrics,
        "static_test": {
            "instances": len(split_ids["static_test"]),
            "rows": row_counts["static_test"],
            "outcomes_opened": False,
        },
    }
    _write(output_paths["report"], report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
