from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_query(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def _record_key(record: dict[str, Any]) -> str:
    if "index" not in record:
        raise ValueError("record is missing index")
    return _canonical(record["index"])


def _source_dir(root: Path, dataset: str, split: str, model: str) -> Path:
    normal = root / dataset / split / model
    if normal.is_dir():
        return normal
    flat = root / dataset / model
    if flat.is_dir():
        return flat
    raise FileNotFoundError(
        f"no result directory for dataset={dataset!r} split={split!r} model={model!r}"
    )


def _select_file(root: Path, dataset: str, split: str, model: str) -> Path:
    directory = _source_dir(root, dataset, split, model)
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no JSON result in {directory}")
    return candidates[-1]


def _valid_outcome(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        score = float(record["score"])
        cost = float(record["cost"])
        prompt_tokens = int(record.get("prompt_tokens", 0))
        completion_tokens = int(record.get("completion_tokens", 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "malformed_outcome"
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None, "invalid_score"
    if not math.isfinite(cost) or cost < 0.0:
        return None, "invalid_cost"
    if prompt_tokens < 0 or completion_tokens < 0:
        return None, "invalid_tokens"
    return (
        {
            "score": score,
            "acceptable": score >= 0.5,
            "cost_usd": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        None,
    )


def _load_result(path: Path, dataset: str, split: str, model: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("dataset_name") != dataset:
        raise ValueError(
            f"{path}: dataset_name={payload.get('dataset_name')!r}, expected {dataset!r}"
        )
    if payload.get("split") != split:
        raise ValueError(f"{path}: split={payload.get('split')!r}, expected {split!r}")
    if payload.get("model_name") != model:
        raise ValueError(
            f"{path}: model_name={payload.get('model_name')!r}, expected {model!r}"
        )
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: records is not a list")
    return payload


def _assignment(
    position: int,
    total: int,
    *,
    is_ood: bool,
) -> str:
    if is_ood:
        return "ood_test"
    if position < int(total * 0.6):
        return "train"
    if position < int(total * 0.8):
        return "calibration"
    return "id_test"


def prepare(
    root: Path,
    amendment_path: Path,
    output_dir: Path,
    *,
    archive: Path | None = None,
) -> dict[str, Any]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    source_lock = amendment["source"]
    if archive is not None:
        actual = _sha256_file(archive)
        expected = source_lock["archive_sha256"]
        if actual != expected:
            raise ValueError(f"archive SHA-256 mismatch: {actual} != {expected}")

    models = [str(model) for model in amendment["models"]]
    seed = str(amendment["split"]["seed"])
    sources = [
        (str(source["dataset"]), str(source["split"]), False)
        for source in amendment["id_sources"]
    ] + [
        (str(source["dataset"]), str(source["split"]), True)
        for source in amendment["ood_sources"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "matrix.jsonl"
    temporary_path = output_dir / "matrix.jsonl.tmp"
    selected_files: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    total_rows = 0
    split_counts: Counter[str] = Counter()

    with temporary_path.open("w", encoding="utf-8") as output:
        for dataset, split, is_ood in sources:
            source_paths: dict[str, Path] = {}
            missing_models: list[str] = []
            for model in models:
                try:
                    source_paths[model] = _select_file(root, dataset, split, model)
                except FileNotFoundError:
                    missing_models.append(model)
            if missing_models:
                dataset_summaries[dataset] = {
                    "source_split": split,
                    "ood": bool(is_ood),
                    "source_excluded": True,
                    "reason": "missing_model_result_file",
                    "missing_models": missing_models,
                    "complete_prompt_count": 0,
                    "route_split_counts": {},
                }
                continue

            reference_queries: dict[str, tuple[str, str]] = {}
            outcomes: dict[str, dict[str, dict[str, Any]]] = {}
            invalid_by_model: dict[str, Counter[str]] = {}
            duplicate_by_model: dict[str, int] = {}
            all_keys: set[str] = set()

            for model_index, model in enumerate(models):
                path = source_paths[model]
                payload = _load_result(path, str(dataset), str(split), model)
                selected_files.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "model": model,
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "records_declared": payload.get("counts"),
                        "records_loaded": len(payload["records"]),
                        "data_fingerprint": payload.get("data_fingerprint"),
                    }
                )
                model_rows: dict[str, dict[str, Any]] = {}
                invalid = Counter()
                duplicates = 0
                for record in payload["records"]:
                    try:
                        key = _record_key(record)
                    except ValueError:
                        invalid["missing_index"] += 1
                        continue
                    if key in model_rows:
                        duplicates += 1
                        continue
                    normalized = _normalized_query(record.get("origin_query", ""))
                    if not normalized:
                        invalid["empty_query"] += 1
                        continue
                    outcome, reason = _valid_outcome(record)
                    if reason is not None:
                        invalid[reason] += 1
                        continue
                    if model_index == 0:
                        reference_queries[key] = (
                            str(record.get("origin_query", "")),
                            normalized,
                        )
                    else:
                        reference = reference_queries.get(key)
                        if reference is None:
                            invalid["not_in_reference"] += 1
                            continue
                        if normalized != reference[1]:
                            invalid["query_mismatch"] += 1
                            continue
                    assert outcome is not None
                    model_rows[key] = outcome
                outcomes[model] = model_rows
                invalid_by_model[model] = invalid
                duplicate_by_model[model] = duplicates
                all_keys.update(model_rows)

            complete_keys = set(reference_queries)
            for model in models:
                complete_keys.intersection_update(outcomes[model])
            incomplete = len(all_keys - complete_keys)
            local_counts: Counter[str] = Counter()
            ordered_keys = sorted(
                complete_keys,
                key=lambda item: hashlib.sha256(
                    f"{seed}\x1f{dataset}\x1f{item}\x1f{reference_queries[item][1]}".encode()
                ).digest(),
            )
            for position, key in enumerate(ordered_keys):
                original, normalized = reference_queries[key]
                route_split = _assignment(
                    position,
                    len(ordered_keys),
                    is_ood=bool(is_ood),
                )
                row = {
                    "schema_version": "llmrouterbench-compact-matrix-v1",
                    "dataset": dataset,
                    "source_split": split,
                    "route_split": route_split,
                    "prompt_key": key,
                    "origin_query": original,
                    "outcomes": {model: outcomes[model][key] for model in models},
                }
                output.write(_canonical(row) + "\n")
                total_rows += 1
                local_counts[route_split] += 1
                split_counts[route_split] += 1
            dataset_summaries[str(dataset)] = {
                "source_split": split,
                "ood": bool(is_ood),
                "complete_prompt_count": len(complete_keys),
                "incomplete_prompt_keys": incomplete,
                "route_split_counts": dict(sorted(local_counts.items())),
                "invalid_by_model": {
                    model: dict(sorted(counts.items()))
                    for model, counts in invalid_by_model.items()
                    if counts
                },
                "duplicate_indices_by_model": {
                    model: count
                    for model, count in duplicate_by_model.items()
                    if count
                },
            }
    temporary_path.replace(rows_path)

    manifest = {
        "schema_version": "llmrouterbench-public-manifest-v1",
        "source": source_lock,
        "amendment_sha256": _sha256_file(amendment_path),
        "models": models,
        "model_count": len(models),
        "prompt_count": total_rows,
        "outcome_count": total_rows * len(models),
        "route_split_counts": dict(sorted(split_counts.items())),
        "datasets": dataset_summaries,
        "selected_files": selected_files,
        "matrix_path": rows_path.name,
        "matrix_sha256": _sha256_file(rows_path),
    }
    manifest["manifest_hash"] = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and compact the pinned public LLMRouterBench matrix."
    )
    parser.add_argument("root", type=Path, help="Extracted bench-release directory")
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("artifacts/active_router_protocol_amendment_005_public_matrix.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/public_router_v1/prepared"),
    )
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    manifest = prepare(
        args.root,
        args.amendment,
        args.output_dir,
        archive=args.archive,
    )
    print(
        json.dumps(
            {
                "prompt_count": manifest["prompt_count"],
                "outcome_count": manifest["outcome_count"],
                "route_split_counts": manifest["route_split_counts"],
                "matrix_sha256": manifest["matrix_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
