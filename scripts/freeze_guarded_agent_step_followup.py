#!/usr/bin/env python3
"""Fit, live-calibrate, and freeze the Amendment 008 guarded router."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from budget_router.agent_step import FrozenAgentStepArtifact, render_agent_prefix
from budget_router.agent_training import (
    AgentTrainingRow,
    fit_logistic_with_grouped_cv,
    fit_platt,
    load_twinrouter_rows,
    rows_for_split,
    sha256_file,
    split_instances,
)
from budget_router.guarded_agent_step import (
    GuardedAgentStepArtifact,
    GuardedAgentStepRouter,
)
from budget_router.serialization import stable_hash


@dataclass(frozen=True, slots=True)
class ShadowPrefix:
    calibrated_probability: float
    approximate_context_tokens: int


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replay_shadow_probabilities(
    trajectories: Sequence[Sequence[ShadowPrefix]],
    *,
    threshold: float,
    force_strong_initial_calls: int,
    maximum_consecutive_cheap_calls: int,
    force_strong_context_tokens: int,
    strong_minimum_dwell_calls: int,
) -> dict[str, Any]:
    """Replay the guarded state machine from precomputed classifier scores."""
    total_calls = 0
    cheap_calls = 0
    trajectories_with_cheap = 0
    guard_reasons: dict[str, int] = {
        "force_strong_initial_calls": 0,
        "force_strong_context_guard": 0,
        "force_strong_dwell": 0,
        "force_strong_cheap_burst": 0,
    }
    trajectory_rows: list[dict[str, Any]] = []
    maximum_observed_cheap_burst = 0
    initial_guard_violations = 0
    for trajectory_index, prefixes in enumerate(trajectories):
        previous = ""
        dwell_remaining = 0
        consecutive_cheap = 0
        trajectory_cheap = 0
        switches = 0
        for call_index, prefix in enumerate(prefixes):
            predicted_cheap = prefix.calibrated_probability <= threshold
            model = "cheap" if predicted_cheap else "strong"
            reason = "classifier_cheap" if predicted_cheap else "classifier_strong"
            if call_index < force_strong_initial_calls:
                model = "strong"
                reason = "force_strong_initial_calls"
            elif (
                prefix.approximate_context_tokens
                >= force_strong_context_tokens
            ):
                model = "strong"
                reason = "force_strong_context_guard"
            elif (
                predicted_cheap
                and previous == "strong"
                and dwell_remaining > 0
            ):
                model = "strong"
                reason = "force_strong_dwell"
            elif (
                predicted_cheap
                and consecutive_cheap >= maximum_consecutive_cheap_calls
            ):
                model = "strong"
                reason = "force_strong_cheap_burst"

            if reason in guard_reasons:
                guard_reasons[reason] += 1
            initial_guard_violations += int(
                call_index < force_strong_initial_calls and model != "strong"
            )
            switches += int(bool(previous) and previous != model)
            total_calls += 1
            if model == "cheap":
                cheap_calls += 1
                trajectory_cheap += 1
                consecutive_cheap += 1
                maximum_observed_cheap_burst = max(
                    maximum_observed_cheap_burst, consecutive_cheap
                )
                dwell_remaining = 0
            else:
                consecutive_cheap = 0
                if previous != "strong":
                    dwell_remaining = max(
                        0, strong_minimum_dwell_calls - 1
                    )
                elif dwell_remaining:
                    dwell_remaining -= 1
            previous = model
        trajectories_with_cheap += int(trajectory_cheap > 0)
        trajectory_rows.append(
            {
                "trajectory_index": trajectory_index,
                "calls": len(prefixes),
                "cheap_calls": trajectory_cheap,
                "switches": switches,
            }
        )
    return {
        "trajectories": len(trajectories),
        "total_calls": total_calls,
        "cheap_calls": cheap_calls,
        "strong_calls": total_calls - cheap_calls,
        "cheap_call_share": cheap_calls / total_calls if total_calls else 0.0,
        "trajectories_with_cheap_call": trajectories_with_cheap,
        "guard_reasons": guard_reasons,
        "maximum_observed_consecutive_cheap_calls": (
            maximum_observed_cheap_burst
        ),
        "initial_guard_violations": initial_guard_violations,
        "trajectory_rows": trajectory_rows,
    }


def select_live_threshold(
    trajectories: Sequence[Sequence[ShadowPrefix]],
    *,
    minimum_cheap_call_share: float,
    maximum_cheap_call_share: float,
    minimum_trajectories_with_cheap_call: int,
    force_strong_initial_calls: int,
    maximum_consecutive_cheap_calls: int,
    force_strong_context_tokens: int,
    strong_minimum_dwell_calls: int,
) -> tuple[float, dict[str, Any]]:
    probabilities = sorted(
        {
            prefix.calibrated_probability
            for trajectory in trajectories
            for prefix in trajectory
        }
    )
    if not probabilities:
        raise ValueError("live threshold selection requires shadow prefixes")
    qualifying: list[tuple[float, dict[str, Any]]] = []
    compact_candidates: list[dict[str, Any]] = []
    for threshold in probabilities:
        metrics = replay_shadow_probabilities(
            trajectories,
            threshold=threshold,
            force_strong_initial_calls=force_strong_initial_calls,
            maximum_consecutive_cheap_calls=maximum_consecutive_cheap_calls,
            force_strong_context_tokens=force_strong_context_tokens,
            strong_minimum_dwell_calls=strong_minimum_dwell_calls,
        )
        qualifies = (
            minimum_cheap_call_share
            <= metrics["cheap_call_share"]
            <= maximum_cheap_call_share
            and metrics["trajectories_with_cheap_call"]
            >= minimum_trajectories_with_cheap_call
            and metrics["initial_guard_violations"] == 0
            and metrics["maximum_observed_consecutive_cheap_calls"]
            <= maximum_consecutive_cheap_calls
        )
        if qualifies:
            qualifying.append((threshold, metrics))
        compact_candidates.append(
            {
                "threshold": threshold,
                "cheap_call_share": metrics["cheap_call_share"],
                "trajectories_with_cheap_call": metrics[
                    "trajectories_with_cheap_call"
                ],
                "qualifies": qualifies,
            }
        )
    if not qualifying:
        raise RuntimeError("no observed live threshold satisfies Amendment 008")
    threshold, selected = qualifying[0]
    return threshold, {
        "selection_rule": "lowest_qualifying_observed_probability",
        "candidate_threshold_count": len(probabilities),
        "qualifying_threshold_count": len(qualifying),
        "selected": selected,
        "candidate_summary": compact_candidates,
    }


def _guarded_static_metrics(
    artifact: GuardedAgentStepArtifact,
    rows: Sequence[AgentTrainingRow],
) -> dict[str, Any]:
    by_instance: dict[str, list[AgentTrainingRow]] = {}
    for row in rows:
        by_instance.setdefault(row.instance_id, []).append(row)
    total = strong_rows = strong_correct = cheap_routes = switches = 0
    passing_instances = 0
    for instance_id in sorted(by_instance):
        router = GuardedAgentStepRouter(artifact)
        passed = True
        for row in sorted(
            by_instance[instance_id], key=lambda item: item.step_index
        ):
            decision = router.select(row.messages, step_index=row.step_index)
            total += 1
            strong_rows += int(row.needs_strong)
            strong_correct += int(
                row.needs_strong
                and decision.model_id == artifact.strong_model
            )
            cheap_routes += int(decision.model_id == artifact.cheap_model)
            if row.needs_strong and decision.model_id != artifact.strong_model:
                passed = False
        passing_instances += int(passed)
        switches += router.switch_count
    return {
        "rows": total,
        "instances": len(by_instance),
        "strong_rows": strong_rows,
        "strong_rows_routed_strong": strong_correct,
        "strong_recall": strong_correct / strong_rows if strong_rows else 1.0,
        "passing_instances": passing_instances,
        "trajectory_pass": (
            passing_instances / len(by_instance) if by_instance else 0.0
        ),
        "cheap_routes": cheap_routes,
        "strong_routes": total - cheap_routes,
        "cheap_route_share": cheap_routes / total if total else 0.0,
        "switch_count": switches,
    }


def _shadow_trajectories(
    paths: Iterable[Path],
    *,
    classifier: FrozenAgentStepArtifact,
) -> tuple[list[list[ShadowPrefix]], list[dict[str, Any]]]:
    trajectories: list[list[ShadowPrefix]] = []
    source_rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        value = _load(path)
        prefixes: list[ShadowPrefix] = []
        messages = value.get("messages", [])
        for message_index, message in enumerate(messages):
            route = (
                message.get("extra", {}).get("agent_step_router")
                if isinstance(message, dict)
                else None
            )
            if not route:
                continue
            step_index = len(prefixes)
            text, summary = render_agent_prefix(
                messages[:message_index], step_index=step_index
            )
            _, calibrated = classifier.calibrated_probability_text(text)
            prefixes.append(
                ShadowPrefix(
                    calibrated_probability=calibrated,
                    approximate_context_tokens=(
                        summary.approximate_context_tokens
                    ),
                )
            )
        if not prefixes:
            raise ValueError(f"shadow trajectory has no routed calls: {path}")
        trajectories.append(prefixes)
        source_rows.append(
            {
                "task_id": str(value["instance_id"]),
                "path": str(path),
                "sha256": sha256_file(path),
                "calls": len(prefixes),
            }
        )
    return trajectories, source_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/agent_step_followup_v2_task_manifest.json"),
    )
    parser.add_argument(
        "--shadow-dir",
        type=Path,
        default=Path(
            "outputs/agent_step_router_v1/trajectories/heldout/"
            "router_frozen-agent-step-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_step_router_v2"),
    )
    args = parser.parse_args()

    protocol = _load(args.protocol)
    task_manifest = _load(args.task_manifest)
    source = protocol["public_training_source"]
    question_hash = sha256_file(args.question_bank)
    source_manifest_hash = sha256_file(args.source_manifest)
    if question_hash != source["question_bank_sha256"]:
        raise SystemExit("TwinRouterBench question-bank hash mismatch")
    if source_manifest_hash != source["manifest_sha256"]:
        raise SystemExit("TwinRouterBench manifest hash mismatch")
    if task_manifest["manifest_hash"] != stable_hash(
        {
            key: value
            for key, value in task_manifest.items()
            if key != "manifest_hash"
        }
    ):
        raise SystemExit("Amendment 008 task-manifest hash mismatch")

    output_paths = {
        "split": args.output_dir / "split_manifest.json",
        "artifact": args.output_dir / "router_artifact.json",
        "training": args.output_dir / "training_report.json",
        "activation": args.output_dir / "activation_gate.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite frozen guarded outputs: " + ", ".join(existing)
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
    split_rows = {
        name: rows_for_split(rows, instance_ids)
        for name, instance_ids in split_ids.items()
    }
    split_payload: dict[str, Any] = {
        "schema_version": "guarded-agent-step-split-v2",
        "source_revision": source["revision"],
        "question_bank_sha256": question_hash,
        "source_manifest_sha256": source_manifest_hash,
        "split_seed": split_config["seed"],
        "excluded_swe_repository_prefixes": source[
            "excluded_swe_repository_prefixes"
        ],
        "source_counts": source_counts,
        "instance_ids": {
            name: list(instance_ids) for name, instance_ids in split_ids.items()
        },
        "instance_counts": {
            name: len(instance_ids) for name, instance_ids in split_ids.items()
        },
        "row_counts": {
            name: len(values) for name, values in split_rows.items()
        },
    }
    split_payload["manifest_hash"] = stable_hash(split_payload)
    _write(output_paths["split"], split_payload)

    classifier_config = protocol["classifier"]
    dimension = int(classifier_config["dimension"])
    hash_seed = str(classifier_config["hash_seed"])
    selected_c, bias, weights, cv_rows = fit_logistic_with_grouped_cv(
        split_rows["train"],
        dimension=dimension,
        hash_seed=hash_seed,
        c_candidates=[
            float(value)
            for value in classifier_config[
                "inverse_l2_strength_candidates"
            ]
        ],
        fold_seed=str(split_config["seed"]),
    )
    uncalibrated = FrozenAgentStepArtifact(
        cheap_model=protocol["model_selection"]["cheap"],
        strong_model=protocol["model_selection"]["strong"],
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
        for row in split_rows["calibration"]
    ]
    raw_probabilities = [
        uncalibrated.raw_probability_text(text) for text in calibration_texts
    ]
    platt_slope, platt_intercept = fit_platt(
        raw_probabilities,
        [row.needs_strong for row in split_rows["calibration"]],
    )
    runtime = protocol["runtime"]
    classifier_base = FrozenAgentStepArtifact(
        cheap_model=protocol["model_selection"]["cheap"],
        strong_model=protocol["model_selection"]["strong"],
        dimension=dimension,
        hash_seed=hash_seed,
        bias=bias,
        weights=weights,
        platt_slope=platt_slope,
        platt_intercept=platt_intercept,
        cheap_threshold=0.0,
        force_strong_context_tokens=int(
            runtime["force_strong_approximate_context_tokens"]
        ),
        strong_minimum_dwell_calls=int(
            runtime["strong_minimum_dwell_calls_before_deescalation"]
        ),
    )

    shadow_paths = list(args.shadow_dir.glob("*.json"))
    expected_development = int(
        protocol["live_threshold"]["expected_development_trajectories"]
    )
    if len(shadow_paths) != expected_development:
        raise SystemExit(
            f"expected {expected_development} shadow trajectories, "
            f"found {len(shadow_paths)}"
        )
    expected_ids = {row["task_id"] for row in task_manifest["development"]}
    actual_ids = {path.stem for path in shadow_paths}
    if actual_ids != expected_ids:
        raise SystemExit("shadow trajectories do not match development tasks")
    trajectories, shadow_sources = _shadow_trajectories(
        shadow_paths, classifier=classifier_base
    )
    live_config = protocol["live_threshold"]
    threshold, threshold_report = select_live_threshold(
        trajectories,
        minimum_cheap_call_share=float(
            live_config["minimum_shadow_cheap_call_share"]
        ),
        maximum_cheap_call_share=float(
            live_config["maximum_shadow_cheap_call_share"]
        ),
        minimum_trajectories_with_cheap_call=int(
            live_config["minimum_shadow_trajectories_with_cheap_call"]
        ),
        force_strong_initial_calls=int(
            runtime["force_strong_initial_calls"]
        ),
        maximum_consecutive_cheap_calls=int(
            runtime["maximum_consecutive_cheap_calls"]
        ),
        force_strong_context_tokens=int(
            runtime["force_strong_approximate_context_tokens"]
        ),
        strong_minimum_dwell_calls=int(
            runtime["strong_minimum_dwell_calls_before_deescalation"]
        ),
    )
    base_frozen = FrozenAgentStepArtifact(
        cheap_model=classifier_base.cheap_model,
        strong_model=classifier_base.strong_model,
        dimension=classifier_base.dimension,
        hash_seed=classifier_base.hash_seed,
        bias=classifier_base.bias,
        weights=classifier_base.weights,
        platt_slope=classifier_base.platt_slope,
        platt_intercept=classifier_base.platt_intercept,
        cheap_threshold=threshold,
        force_strong_context_tokens=(
            classifier_base.force_strong_context_tokens
        ),
        strong_minimum_dwell_calls=(
            classifier_base.strong_minimum_dwell_calls
        ),
        metadata={
            "study_id": protocol["parent_study_id"],
            "amendment_id": protocol["amendment_id"],
            "split_manifest_hash": split_payload["manifest_hash"],
            "selected_C": selected_c,
            "training_rows": len(split_rows["train"]),
            "calibration_rows": len(split_rows["calibration"]),
            "threshold_source": "amendment_007_visible_prefixes_only",
        },
    )
    guarded = GuardedAgentStepArtifact(
        base_classifier=FrozenAgentStepArtifact.from_dict(base_frozen.to_dict()),
        force_strong_initial_calls=int(
            runtime["force_strong_initial_calls"]
        ),
        maximum_consecutive_cheap_calls=int(
            runtime["maximum_consecutive_cheap_calls"]
        ),
        metadata={
            "study_id": protocol["parent_study_id"],
            "amendment_id": protocol["amendment_id"],
            "task_manifest_hash": task_manifest["manifest_hash"],
            "shadow_trajectory_count": len(shadow_sources),
            "shadow_source_hash": stable_hash(shadow_sources),
        },
    )
    _write(output_paths["artifact"], guarded.to_dict())
    reloaded = GuardedAgentStepArtifact.from_dict(_load(output_paths["artifact"]))

    static_metrics = {
        name: _guarded_static_metrics(reloaded, values)
        for name, values in split_rows.items()
    }
    training_report = {
        "schema_version": "guarded-agent-step-training-report-v2",
        "status": "artifact_frozen_static_and_shadow_evaluation_complete",
        "artifact_hash": reloaded.artifact_hash,
        "artifact_file_sha256": sha256_file(output_paths["artifact"]),
        "split_manifest_hash": split_payload["manifest_hash"],
        "selected_C": selected_c,
        "cross_validation": cv_rows,
        "platt": {
            "slope": platt_slope,
            "intercept": platt_intercept,
        },
        "threshold": threshold,
        "static_metrics": static_metrics,
        "static_metrics_are_descriptive_not_threshold_selection": True,
    }
    _write(output_paths["training"], training_report)

    selected_shadow = threshold_report["selected"]
    activation_passed = (
        selected_shadow["cheap_call_share"]
        >= float(live_config["minimum_shadow_cheap_call_share"])
        and selected_shadow["cheap_call_share"]
        <= float(live_config["maximum_shadow_cheap_call_share"])
        and selected_shadow["trajectories_with_cheap_call"]
        >= int(live_config["minimum_shadow_trajectories_with_cheap_call"])
        and selected_shadow["initial_guard_violations"] == 0
        and selected_shadow["maximum_observed_consecutive_cheap_calls"]
        <= int(runtime["maximum_consecutive_cheap_calls"])
    )
    activation = {
        "schema_version": "guarded-agent-step-activation-gate-v2",
        "status": (
            "development_collection_authorized"
            if activation_passed
            else "development_collection_blocked"
        ),
        "development_collection_authorized": activation_passed,
        "official_outcomes_used": False,
        "artifact_hash": reloaded.artifact_hash,
        "task_manifest_hash": task_manifest["manifest_hash"],
        "selected_threshold": threshold,
        "threshold_selection": threshold_report,
        "shadow_sources": shadow_sources,
        "shadow_sources_hash": stable_hash(shadow_sources),
    }
    _write(output_paths["activation"], activation)
    print(json.dumps(activation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
