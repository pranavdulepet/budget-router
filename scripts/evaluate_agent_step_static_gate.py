#!/usr/bin/env python3
"""Evaluate the already-frozen agent-step artifact on its static test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from budget_router.agent_step import FrozenAgentStepArtifact
from budget_router.agent_training import (
    evaluate_static_rows,
    load_twinrouter_rows,
    rows_for_split,
    sha256_file,
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
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v1/router_artifact.json"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/agent_step_router_v1/split_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/agent_step_router_v1/static_gate.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite static gate: {args.output}")

    protocol = _load(args.protocol)
    source = protocol["public_training_source"]
    if sha256_file(args.question_bank) != source["question_bank_sha256"]:
        raise SystemExit("TwinRouterBench question-bank hash mismatch")
    if sha256_file(args.source_manifest) != source["manifest_sha256"]:
        raise SystemExit("TwinRouterBench manifest hash mismatch")
    artifact = FrozenAgentStepArtifact.from_dict(_load(args.artifact))
    split_manifest = _load(args.split_manifest)
    claimed_split_hash = str(split_manifest["manifest_hash"])
    split_without_hash = dict(split_manifest)
    split_without_hash.pop("manifest_hash")
    if stable_hash(split_without_hash) != claimed_split_hash:
        raise SystemExit("agent-step split-manifest hash mismatch")
    if artifact.metadata.get("split_manifest_hash") != claimed_split_hash:
        raise SystemExit("artifact belongs to a different agent-step split")

    rows, source_counts = load_twinrouter_rows(
        args.question_bank,
        excluded_swe_repository_prefixes=source[
            "excluded_swe_repository_prefixes"
        ],
    )
    static_rows = rows_for_split(
        rows,
        split_manifest["instance_ids"]["static_test"],
    )
    metrics = evaluate_static_rows(artifact, static_rows)
    frozen_gate = protocol["static_launch_gate"]
    checks = {
        "trajectory_pass": (
            metrics["trajectory_pass"]
            >= float(frozen_gate["minimum_trajectory_pass"])
        ),
        "strong_recall": (
            metrics["strong_recall"]
            >= float(frozen_gate["minimum_strong_recall"])
        ),
        "cheap_route_share": (
            metrics["cheap_route_share"]
            >= float(frozen_gate["minimum_cheap_route_share"])
        ),
        "artifact_and_source_hashes": True,
    }
    result = {
        "schema_version": "agent-step-static-gate-v1",
        "artifact_hash": artifact.artifact_hash,
        "artifact_file_sha256": sha256_file(args.artifact),
        "split_manifest_hash": claimed_split_hash,
        "source_counts": source_counts,
        "metrics": metrics,
        "thresholds": frozen_gate,
        "checks": checks,
        "dynamic_collection_authorized": all(checks.values()),
    }
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
