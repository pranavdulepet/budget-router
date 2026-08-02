from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from budget_router.serialization import read_jsonl

try:
    from scripts.run_with_watchdog import run_with_watchdog
except ModuleNotFoundError:
    from run_with_watchdog import run_with_watchdog


def next_test_block(
    protocol: dict[str, Any],
    gate_freeze: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[str, list[str]] | None:
    policies = [
        str(gate_freeze["selected_candidate_policy_id"]),
        str(protocol["fixed_policy"]["policy_id"]),
    ]
    completed: set[tuple[str, str]] = set()
    for record in records:
        if record.get("study_stage") != "test":
            continue
        key = (str(record["task_id"]), str(record["model"]))
        if key in completed:
            raise ValueError(f"duplicate test episode: {key}")
        completed.add(key)
    for task_id in map(str, protocol["stages"]["test"]["task_ids"]):
        missing = [
            policy for policy in policies if (task_id, policy) not in completed
        ]
        if missing:
            return task_id, missing
    return None


def _image_name(task_id: str) -> str:
    from minisweagent.run.benchmarks.swebench import (
        get_swebench_docker_image_name,
    )

    return get_swebench_docker_image_name({"instance_id": task_id})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired held-out blocks with explicit image preflight and "
            "one-image-at-a-time Docker recycling."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--gate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/gate_freeze.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes.jsonl"),
    )
    parser.add_argument("--max-task-blocks", type=int, required=True)
    parser.add_argument("--watchdog-seconds", type=float, default=2700)
    parser.add_argument(
        "--approved-incremental-credit-usd",
        default="112.50",
    )
    args = parser.parse_args()
    if args.max_task_blocks <= 0:
        parser.error("--max-task-blocks must be positive")

    protocol = _load(args.protocol)
    gate_freeze = _load(args.gate_freeze)
    for block_index in range(args.max_task_blocks):
        records = read_jsonl(args.records) if args.records.exists() else []
        pending = next_test_block(protocol, gate_freeze, records)
        if pending is None:
            print(
                json.dumps(
                    {"event": "test_complete", "blocks_run": block_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        task_id, missing_policies = pending
        image = _image_name(task_id)
        print(
            json.dumps(
                {
                    "event": "preflight_image",
                    "task_id": task_id,
                    "missing_policies": missing_policies,
                    "image": image,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            check=True,
        )
        return_code = run_with_watchdog(
            [
                sys.executable,
                "scripts/run_isolated_stage_router_study.py",
                "--stage",
                "test",
                "--max-task-blocks",
                "1",
                "--workers",
                "1",
                "--approved-incremental-credit-usd",
                args.approved_incremental_credit_usd,
                "--execute",
            ],
            timeout_seconds=args.watchdog_seconds,
        )
        if return_code:
            raise SystemExit(return_code)

        updated = read_jsonl(args.records)
        next_pending = next_test_block(protocol, gate_freeze, updated)
        if next_pending is not None and next_pending[0] == task_id:
            raise RuntimeError(
                f"test runner returned without completing block: {task_id}"
            )
        subprocess.run(["docker", "image", "rm", image], check=True)
        print(
            json.dumps(
                {
                    "event": "block_complete_image_removed",
                    "task_id": task_id,
                    "canonical_episode_count": len(updated),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
