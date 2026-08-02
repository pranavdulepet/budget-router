#!/usr/bin/env python3
"""Freeze exact-task-unseen SWE-bench tasks for Amendment 008."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import stable_hash


TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_executed_task_ids(output_root: Path) -> set[str]:
    """Collect IDs with evidence that a prior paid episode actually ran."""
    used: set[str] = set()
    for path in output_root.rglob("*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            task_id = row.get("task_id") or row.get("instance_id")
            if (
                isinstance(task_id, str)
                and TASK_ID.fullmatch(task_id)
                and any(
                    key in row
                    for key in (
                        "conservative_cost_usd",
                        "submitted_patch",
                        "model_calls",
                        "latency_seconds",
                    )
                )
            ):
                used.add(task_id)
    for path in output_root.rglob("*.json"):
        if any(part in {"trajectories", "episode_shards"} for part in path.parts):
            if TASK_ID.fullmatch(path.stem):
                used.add(path.stem)
        if "swebench_grader" not in path.parts:
            continue
        try:
            report = _load(path)
        except (OSError, TypeError, ValueError):
            continue
        for key in ("submitted_ids", "completed_ids", "error_ids"):
            values = report.get(key, []) if isinstance(report, dict) else []
            for task_id in values:
                if isinstance(task_id, str) and TASK_ID.fullmatch(task_id):
                    used.add(task_id)
    return used


def select_tasks(
    records: Iterable[dict[str, Any]],
    *,
    used: set[str],
    quotas: dict[str, int],
    seed: str,
) -> list[dict[str, str]]:
    by_repository: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_id = str(row["instance_id"])
        repository = str(row["repo"])
        if task_id not in used and repository in quotas:
            by_repository[repository].append(row)
    selected: list[dict[str, str]] = []
    for repository, quota in sorted(quotas.items()):
        ordered = sorted(
            by_repository[repository],
            key=lambda row: hashlib.sha256(
                f"{seed}:{repository}:{row['instance_id']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise ValueError(
                f"{repository} has {len(ordered)} unused tasks; {quota} required"
            )
        selected.extend(
            {
                "task_id": str(row["instance_id"]),
                "repository": repository,
            }
            for row in ordered[:quota]
        )
    return sorted(selected, key=lambda row: row["task_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_008_guarded_agent_step_followup.json"
        ),
    )
    parser.add_argument(
        "--prior-task-manifest",
        type=Path,
        default=Path("artifacts/active_router_task_manifest.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agent_step_followup_v2_task_manifest.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite task manifest: {args.output}")

    protocol = _load(args.protocol)
    prior = _load(args.prior_task_manifest)
    tasks = _load(args.tasks)
    used = collect_executed_task_ids(args.output_root)
    heldout = protocol["heldout"]
    novel_quotas = {
        str(repository): int(count)
        for repository, count in heldout[
            "novel_repository_primary_subgroup"
        ]["repositories"].items()
    }
    familiar_quotas = {
        str(repository): int(count)
        for repository, count in heldout[
            "familiar_repository_secondary_subgroup"
        ]["repositories"].items()
    }
    seed = str(heldout["selection_seed"])
    novel = select_tasks(
        tasks["records"],
        used=used,
        quotas=novel_quotas,
        seed=seed,
    )
    familiar = select_tasks(
        tasks["records"],
        used=used | {row["task_id"] for row in novel},
        quotas=familiar_quotas,
        seed=seed,
    )
    development = [
        {
            "task_id": str(row["task_id"]),
            "repository": str(row["repository"]),
            "source": "amendment_007_heldout_now_development",
        }
        for row in prior["heldout"]
    ]
    selected = [
        {**row, "subgroup": "novel_repository_primary"} for row in novel
    ] + [
        {**row, "subgroup": "familiar_repository_secondary"}
        for row in familiar
    ]
    if len(selected) != int(heldout["tasks"]):
        raise RuntimeError("selected held-out task count differs from protocol")
    selected_ids = {row["task_id"] for row in selected}
    development_ids = {row["task_id"] for row in development}
    if selected_ids & used:
        raise RuntimeError("selected held-out tasks contain previously executed IDs")
    if selected_ids & development_ids:
        raise RuntimeError("development and held-out exact task IDs overlap")

    payload: dict[str, Any] = {
        "schema_version": "agent-step-followup-task-manifest-v2",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "selection_seed": seed,
        "source": {
            "dataset": tasks["source"],
            "revision": tasks["revision"],
            "content_hash": tasks["content_hash"],
        },
        "prior_execution_index": {
            "executed_task_count": len(used),
            "executed_task_ids_sha256": hashlib.sha256(
                "\n".join(sorted(used)).encode()
            ).hexdigest(),
        },
        "development": sorted(development, key=lambda row: row["task_id"]),
        "heldout": sorted(selected, key=lambda row: row["task_id"]),
        "subgroup_counts": {
            "novel_repository_primary": len(novel),
            "familiar_repository_secondary": len(familiar),
        },
        "repository_counts": {
            repository: sum(
                row["repository"] == repository for row in selected
            )
            for repository in sorted(
                {row["repository"] for row in selected}
            )
        },
        "invariants": {
            "heldout_exact_ids_previously_unexecuted": True,
            "development_and_heldout_exact_ids_disjoint": True,
            "heldout_tasks": len(selected),
            "development_tasks": len(development),
            "heldout_paid_episodes_started": False,
            "heldout_official_grades_opened": False,
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
