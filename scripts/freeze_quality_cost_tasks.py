#!/usr/bin/env python3
"""Freeze Amendment 009 development and exact-task-unseen held-out cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from budget_router.serialization import stable_hash

try:
    from scripts.freeze_agent_step_followup_tasks import (
        collect_executed_task_ids,
        select_tasks,
    )
except ModuleNotFoundError:
    from freeze_agent_step_followup_tasks import (  # type: ignore[no-redef]
        collect_executed_task_ids,
        select_tasks,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_development_tasks(
    records: Iterable[dict[str, Any]],
    *,
    excluded: set[str],
    quotas: dict[str, int],
    seed: str,
) -> list[dict[str, str]]:
    by_repository: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_id = str(row["instance_id"])
        repository = str(row["repo"])
        if task_id not in excluded and repository in quotas:
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
                f"{repository} has {len(ordered)} eligible tasks; {quota} required"
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
            "active_router_protocol_amendment_009_three_tier_quality_cost.json"
        ),
    )
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=Path("artifacts/quality_cost_v1_compatibility_tasks.json"),
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
        default=Path("artifacts/quality_cost_v1_task_manifest.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite task manifest: {args.output}")

    protocol = _load(args.protocol)
    compatibility = _load(args.compatibility_manifest)
    tasks = _load(args.tasks)
    design = protocol["data_design"]
    used = collect_executed_task_ids(args.output_root)
    gate_ids = {str(row["task_id"]) for row in compatibility["tasks"]}

    heldout_design = design["heldout"]
    heldout_quotas = {
        str(repository): int(count)
        for repository, count in heldout_design["repositories"].items()
    }
    heldout = select_tasks(
        tasks["records"],
        used=used | gate_ids,
        quotas=heldout_quotas,
        seed=str(heldout_design["selection_seed"]),
    )
    heldout_ids = {row["task_id"] for row in heldout}

    development_design = design["development"]
    development_quotas = {
        str(repository): int(count)
        for repository, count in development_design["repositories"].items()
    }
    development = select_development_tasks(
        tasks["records"],
        excluded=gate_ids | heldout_ids,
        quotas=development_quotas,
        seed=str(development_design["selection_seed"]),
    )
    development_ids = {row["task_id"] for row in development}

    if len(development) != int(development_design["tasks"]):
        raise RuntimeError("development task count differs from protocol")
    if len(heldout) != int(heldout_design["tasks"]):
        raise RuntimeError("held-out task count differs from protocol")
    if heldout_ids & used:
        raise RuntimeError("held-out cohort contains a previously executed task")
    if development_ids & heldout_ids or gate_ids & (development_ids | heldout_ids):
        raise RuntimeError("compatibility, development, and held-out IDs must be disjoint")
    development_repositories = {row["repository"] for row in development}
    heldout_repositories = {row["repository"] for row in heldout}
    if development_repositories & heldout_repositories:
        raise RuntimeError("development and held-out repositories must be disjoint")

    payload: dict[str, Any] = {
        "schema_version": "quality-cost-task-manifest-v1",
        "study_id": protocol["parent_study_id"],
        "amendment_id": protocol["amendment_id"],
        "source": {
            "dataset": tasks["source"],
            "revision": tasks["revision"],
            "content_hash": tasks["content_hash"],
        },
        "selection": {
            "development_seed": development_design["selection_seed"],
            "heldout_seed": heldout_design["selection_seed"],
        },
        "prior_execution_index": {
            "executed_task_count": len(used),
            "executed_task_ids_sha256": hashlib.sha256(
                "\n".join(sorted(used)).encode()
            ).hexdigest(),
        },
        "compatibility": sorted(gate_ids),
        "development": development,
        "heldout": heldout,
        "repository_counts": {
            "development": {
                repository: sum(
                    row["repository"] == repository for row in development
                )
                for repository in sorted(development_repositories)
            },
            "heldout": {
                repository: sum(row["repository"] == repository for row in heldout)
                for repository in sorted(heldout_repositories)
            },
        },
        "invariants": {
            "heldout_exact_ids_previously_unexecuted": True,
            "development_and_heldout_repositories_disjoint": True,
            "all_cohorts_exact_ids_disjoint": True,
            "development_tasks": len(development),
            "heldout_tasks": len(heldout),
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
