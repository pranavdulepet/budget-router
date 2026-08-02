#!/usr/bin/env python3
"""Freeze every input that can affect Amendment 008 paid execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from budget_router.serialization import stable_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/agent_step_router_v2/dynamic_lock.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite dynamic lock: {args.output}")

    paths = [
        Path("pyproject.toml"),
        Path("artifacts/active_router_protocol.json"),
        Path("artifacts/active_router_protocol.sha256"),
        Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ),
        Path(
            "docs/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.md"
        ),
        Path("artifacts/agent_step_followup_v2_task_manifest.json"),
        Path("configs/model_pool_router_v1.json"),
        Path("configs/tinker_prices_2026-07-29_router_v1.json"),
        Path("configs/harness_prompt.txt"),
        Path("data/swebench_verified_tasks.json"),
        Path("outputs/agent_step_router_v2/split_manifest.json"),
        Path("outputs/agent_step_router_v2/router_artifact.json"),
        Path("outputs/agent_step_router_v2/training_report.json"),
        Path("outputs/agent_step_router_v2/activation_gate.json"),
        Path("src/budget_router/agent_step.py"),
        Path("src/budget_router/guarded_agent_step.py"),
        Path("src/budget_router/mini_swe_tinker.py"),
        Path("src/budget_router/pricing.py"),
        Path("src/budget_router/redaction.py"),
        Path("src/budget_router/serialization.py"),
        Path("src/budget_router/task_model.py"),
        Path("src/budget_router/providers/tinker.py"),
        Path("scripts/run_tinker_pilot.py"),
        Path("scripts/run_active_router_study.py"),
        Path("scripts/run_agent_step_study.py"),
        Path("scripts/run_guarded_agent_step_followup.py"),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing guarded dynamic inputs: " + ", ".join(missing))

    from minisweagent import package_dir

    mini_config_path = package_dir / "config" / "benchmarks" / "swebench.yaml"
    mini_config = yaml.safe_load(mini_config_path.read_text(encoding="utf-8"))
    if not mini_config.get("agent", {}).get("system_template"):
        raise ValueError("mini-swe-agent system template is absent")
    activation = _load(
        Path("outputs/agent_step_router_v2/activation_gate.json")
    )
    if activation.get("development_collection_authorized") is not True:
        raise PermissionError("activation gate did not authorize dynamic freeze")
    artifact = _load(Path("outputs/agent_step_router_v2/router_artifact.json"))
    protocol = _load(
        Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "guarded-agent-step-dynamic-lock-v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "router_artifact_hash": artifact["artifact_hash"],
        "activation_gate_authorized": True,
        "development_paid_episodes_started": False,
        "heldout_paid_episodes_started": False,
        "heldout_official_grades_opened": False,
        "files": {str(path): _sha256(path) for path in paths},
        "external_runtime": {
            "mini_swe_agent_version": importlib.metadata.version(
                "mini-swe-agent"
            ),
            "mini_swe_agent_config_path": str(mini_config_path),
            "mini_swe_agent_config_sha256": _sha256(mini_config_path),
            "tinker_version": importlib.metadata.version("tinker"),
            "tinker_cookbook_version": importlib.metadata.version(
                "tinker-cookbook"
            ),
            "tml_renderers_version": importlib.metadata.version("tml-renderers"),
        },
    }
    payload["manifest_hash"] = stable_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
