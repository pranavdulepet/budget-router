#!/usr/bin/env python3
"""Pin and sanitize SWE-bench Verified, then freeze split and pilot manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from budget_router.experiments import TaskRecord, repository_split
from budget_router.serialization import stable_hash, stable_json

DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
ALLOWED_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "version",
    "environment_setup_commit",
)
FORBIDDEN_FIELDS = ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")


def _revision(dataset_name: str, requested: str | None) -> str:
    if requested:
        return requested
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install the data extra before preparing SWE-bench") from exc
    return str(HfApi().dataset_info(dataset_name).sha)


def _load(dataset_name: str, revision: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the data extra before preparing SWE-bench") from exc
    return load_dataset(dataset_name, revision=revision, split="test")


def _sanitize(row: dict[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field) for field in ALLOWED_FIELDS if field in row}
    missing = [
        field for field in ("instance_id", "repo", "base_commit", "problem_statement")
        if not result.get(field)
    ]
    if missing:
        raise ValueError(f"dataset row is missing required fields: {missing}")
    if any(field in result for field in FORBIDDEN_FIELDS):
        raise ValueError("gold patch or grader fields entered the sanitized task record")
    return result


def _pilot_tasks(
    rows: list[dict[str, Any]],
    train_ids: set[str],
    *,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    eligible = [
        {"task_id": row["instance_id"], "repository": row["repo"]}
        for row in rows
        if row["instance_id"] in train_ids
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['task_id']}".encode()
        ).hexdigest()
    )
    for row in eligible:
        grouped[row["repository"]].append(row)
    repositories = sorted(
        grouped,
        key=lambda repo: hashlib.sha256(f"{seed}:{repo}".encode()).hexdigest(),
    )
    selected: list[dict[str, str]] = []
    while len(selected) < count and any(grouped.values()):
        for repository in repositories:
            if grouped[repository]:
                selected.append(grouped[repository].popleft())
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"training split has only {len(selected)} selectable tasks")
    return selected


def _write_new(path: Path, value: Any) -> None:
    serialized = stable_json(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite different frozen data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision")
    parser.add_argument("--tasks", type=Path, default=Path("data/swebench_verified_tasks.json"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/swebench_verified_split.json"),
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=Path("artifacts/pilot_tasks.json"),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--pilot-count", type=int, default=12)
    args = parser.parse_args()

    revision = _revision(args.dataset, args.revision)
    rows = [_sanitize(dict(row)) for row in _load(args.dataset, revision)]
    if len(rows) != 500:
        raise ValueError(f"SWE-bench Verified must contain 500 tasks, found {len(rows)}")
    identifiers = [str(row["instance_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("SWE-bench instance IDs must be unique")

    task_records = [
        TaskRecord(task_id=row["instance_id"], repository=row["repo"])
        for row in rows
    ]
    manifest = repository_split(task_records, seed=args.seed)
    manifest.freeze(args.split_manifest)
    pilot = _pilot_tasks(
        rows,
        set(manifest.train),
        count=args.pilot_count,
        seed=args.seed,
    )
    task_payload = {
        "schema_version": "swebench-visible-tasks-v1",
        "source": args.dataset,
        "revision": revision,
        "records": rows,
        "excluded_fields": list(FORBIDDEN_FIELDS),
        "content_hash": stable_hash(rows),
    }
    pilot_payload = {
        "schema_version": "pilot-task-manifest-v1",
        "source": args.dataset,
        "revision": revision,
        "split_manifest_hash": manifest.manifest_hash,
        "seed": args.seed,
        "tasks": pilot,
    }
    _write_new(args.tasks, task_payload)
    _write_new(args.pilot_manifest, pilot_payload)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "revision": revision,
                "tasks": len(rows),
                "train": len(manifest.train),
                "calibration": len(manifest.calibration),
                "test": len(manifest.test),
                "pilot_tasks": len(pilot),
                "split_manifest_hash": manifest.manifest_hash,
                "gold_fields_written": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
