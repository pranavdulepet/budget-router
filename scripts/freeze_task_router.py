from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from budget_router.serialization import stable_hash, stable_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_router(
    study: dict[str, Any],
    screen_gate: dict[str, Any],
    dataset_manifest: dict[str, Any],
    router_artifact: dict[str, Any],
    *,
    router_artifact_bytes: bytes,
    router_artifact_path: str,
) -> dict[str, Any]:
    study_hash = str(study["manifest_hash"])
    if (
        screen_gate.get("study_manifest_hash") != study_hash
        or screen_gate.get("screen_gate_passed") is not True
    ):
        raise ValueError("screen gate is invalid for the frozen study")
    if router_artifact.get("study_manifest_hash") != study_hash:
        raise ValueError("router artifact is for a different study")
    if router_artifact.get("dataset_hash") != dataset_manifest.get("dataset_hash"):
        raise ValueError("router artifact does not match the supervision dataset")
    if "test" in dataset_manifest.get("splits", ()):
        raise ValueError("cannot freeze a router trained or tuned on test labels")
    if router_artifact.get("training", {}).get("test_labels_accessed") is not False:
        raise ValueError("router artifact lacks an explicit test-label leakage guard")
    if list(router_artifact.get("models", ())) != list(
        screen_gate.get("surviving_models", ())
    ):
        raise ValueError("router candidates do not match screen survivors")
    artifact_without_hash = dict(router_artifact)
    claimed_artifact_hash = str(artifact_without_hash.pop("artifact_hash", ""))
    if stable_hash(artifact_without_hash) != claimed_artifact_hash:
        raise ValueError("router artifact content hash is invalid")

    result = {
        "schema_version": "task-router-freeze-v1",
        "study_id": study["study_id"],
        "study_manifest_hash": study_hash,
        "screen_gate_hash": screen_gate["gate_hash"],
        "dataset_hash": dataset_manifest["dataset_hash"],
        "router_artifact_path": router_artifact_path,
        "router_artifact_hash": claimed_artifact_hash,
        "router_artifact_sha256": hashlib.sha256(router_artifact_bytes).hexdigest(),
        "models": list(router_artifact["models"]),
        "test_labels_accessed": False,
        "status": "frozen_before_test_collection",
    }
    result["freeze_hash"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--screen-gate",
        type=Path,
        default=Path("outputs/router_baseline_v2/screen_gate.json"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_dataset_manifest.json"),
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=Path("outputs/router_baseline_v2/task_router.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_freeze.json"),
    )
    args = parser.parse_args()
    router_bytes = args.router.read_bytes()
    result = freeze_router(
        _load(args.study),
        _load(args.screen_gate),
        _load(args.dataset_manifest),
        json.loads(router_bytes),
        router_artifact_bytes=router_bytes,
        router_artifact_path=str(args.router),
    )
    serialized = stable_json(result) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError("refusing to overwrite a different router freeze")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
