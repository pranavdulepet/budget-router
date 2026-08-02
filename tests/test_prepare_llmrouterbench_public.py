from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_llmrouterbench_public import prepare


def _write_result(
    root: Path,
    dataset: str,
    split: str,
    model: str,
    records: list[dict],
    *,
    flat: bool = False,
) -> None:
    directory = root / dataset / model if flat else root / dataset / split / model
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "counts": len(records),
        "records": records,
    }
    (directory / f"{dataset}-{split}-{model}-20260729_000000.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_prepare_builds_only_complete_valid_prompt_matrices(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    amendment = {
        "source": {"archive_sha256": "unused"},
        "models": ["small", "large"],
        "id_sources": [{"dataset": "id", "split": "test"}],
        "ood_sources": [{"dataset": "ood", "split": "test"}],
        "split": {"seed": "unit-test"},
    }
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")

    base = [
        {
            "index": index,
            "origin_query": f"Question {index}",
            "score": float(index % 2),
            "cost": 0.01 * index,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        for index in range(1, 11)
    ]
    _write_result(root, "id", "test", "small", base)
    _write_result(
        root,
        "id",
        "test",
        "large",
        [
            {**row, "score": 1.0, "cost": 0.1}
            for row in base
            if row["index"] != 10
        ],
    )
    _write_result(root, "ood", "test", "small", base[:3], flat=True)
    _write_result(root, "ood", "test", "large", base[:3], flat=True)

    manifest = prepare(root, amendment_path, tmp_path / "out")
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "matrix.jsonl").read_text().splitlines()
    ]

    assert manifest["prompt_count"] == 12
    assert manifest["outcome_count"] == 24
    assert manifest["datasets"]["id"]["complete_prompt_count"] == 9
    assert {row["route_split"] for row in rows if row["dataset"] == "ood"} == {
        "ood_test"
    }
    assert all(set(row["outcomes"]) == {"small", "large"} for row in rows)
    assert sum(manifest["route_split_counts"].values()) == 12
