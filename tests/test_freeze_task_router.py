from __future__ import annotations

import pytest

from budget_router.serialization import stable_hash, stable_json
from scripts.freeze_task_router import freeze_router


def _fixtures() -> tuple[dict, dict, dict, dict, bytes]:
    study = {"study_id": "study", "manifest_hash": "study-hash"}
    gate = {
        "study_manifest_hash": "study-hash",
        "screen_gate_passed": True,
        "surviving_models": ["cheap", "strong"],
        "gate_hash": "gate-hash",
    }
    dataset = {"dataset_hash": "dataset-hash", "splits": ["train", "calibration"]}
    artifact = {
        "study_manifest_hash": "study-hash",
        "dataset_hash": "dataset-hash",
        "models": ["cheap", "strong"],
        "training": {"test_labels_accessed": False},
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    content = (stable_json(artifact) + "\n").encode()
    return study, gate, dataset, artifact, content


def test_freeze_router_records_content_hash_before_test() -> None:
    study, gate, dataset, artifact, content = _fixtures()
    result = freeze_router(
        study,
        gate,
        dataset,
        artifact,
        router_artifact_bytes=content,
        router_artifact_path="router.json",
    )
    assert result["test_labels_accessed"] is False
    assert result["models"] == ["cheap", "strong"]
    assert result["router_artifact_sha256"]


def test_freeze_router_rejects_test_label_leakage() -> None:
    study, gate, dataset, artifact, content = _fixtures()
    dataset["splits"].append("test")
    with pytest.raises(ValueError, match="test labels"):
        freeze_router(
            study,
            gate,
            dataset,
            artifact,
            router_artifact_bytes=content,
            router_artifact_path="router.json",
        )
