from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from budget_router.semantic import SemanticRouter
from budget_router.serialization import stable_json
from prepare_llmrouterbench_public import (
    _load_result,
    _normalized_query,
    _record_key,
    _select_file,
    _sha256_file,
    _valid_outcome,
)
from run_public_router_study import evaluate_selection, load_matrix


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def prepare_arc_agi(
    root: Path,
    amendment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    dataset = str(amendment["source"]["dataset"])
    split = str(amendment["source"]["split"])
    models = [str(model) for model in amendment["models"]]

    source_paths = {
        model: _select_file(root, dataset, split, model) for model in models
    }
    selected_files: list[dict[str, Any]] = []
    reference_queries: dict[str, tuple[str, str]] = {}
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    invalid_by_model: dict[str, Counter[str]] = {}
    duplicate_by_model: dict[str, int] = {}
    all_keys: set[str] = set()

    for model_index, model in enumerate(models):
        path = source_paths[model]
        payload = _load_result(path, dataset, split, model)
        selected_files.append(
            {
                "model": model,
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "records_loaded": len(payload["records"]),
                "data_fingerprint": payload.get("data_fingerprint"),
            }
        )
        model_rows: dict[str, dict[str, Any]] = {}
        invalid: Counter[str] = Counter()
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
    ordered_keys = sorted(complete_keys)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "matrix.jsonl"
    temporary_path = output_dir / "matrix.jsonl.tmp"
    with temporary_path.open("w", encoding="utf-8") as stream:
        for key in ordered_keys:
            row = {
                "schema_version": "llmrouterbench-compact-matrix-v1",
                "dataset": dataset,
                "source_split": split,
                "route_split": "external_replication",
                "prompt_key": key,
                "origin_query": reference_queries[key][0],
                "outcomes": {model: outcomes[model][key] for model in models},
            }
            stream.write(_canonical(row) + "\n")
    temporary_path.replace(matrix_path)

    manifest = {
        "schema_version": "arc-agi-replication-manifest-v1",
        "source": amendment["source"],
        "amendment_sha256": _sha256_file(amendment_path),
        "models": models,
        "model_count": len(models),
        "prompt_count": len(ordered_keys),
        "outcome_count": len(ordered_keys) * len(models),
        "route_split_counts": {"external_replication": len(ordered_keys)},
        "datasets": {
            dataset: {
                "source_split": split,
                "complete_prompt_count": len(ordered_keys),
                "incomplete_prompt_keys": len(all_keys - complete_keys),
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
        },
        "selected_files": selected_files,
        "matrix_path": matrix_path.name,
        "matrix_sha256": _sha256_file(matrix_path),
    }
    manifest["manifest_hash"] = hashlib.sha256(
        _canonical(manifest).encode()
    ).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _paired_bootstrap(
    selected_scores: np.ndarray,
    fixed_scores: np.ndarray,
    selected_costs: np.ndarray,
    fixed_costs: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = len(selected_scores)
    quality = np.empty(resamples, dtype=float)
    saving = np.empty(resamples, dtype=float)
    for bootstrap_index in range(resamples):
        sample = rng.integers(0, count, size=count)
        quality[bootstrap_index] = float(
            np.mean(selected_scores[sample] - fixed_scores[sample])
        )
        fixed_total = float(fixed_costs[sample].sum())
        selected_total = float(selected_costs[sample].sum())
        saving[bootstrap_index] = (
            1.0 - selected_total / fixed_total
            if fixed_total > 0
            else math.nan
        )
    return {
        "resamples": resamples,
        "seed": seed,
        "quality_difference_95_interval": [
            float(np.quantile(quality, 0.025)),
            float(np.quantile(quality, 0.975)),
        ],
        "cost_saving_fraction_95_interval": [
            float(np.nanquantile(saving, 0.025)),
            float(np.nanquantile(saving, 0.975)),
        ],
        "probability_quality_at_least_fixed": float(np.mean(quality >= 0.0)),
    }


def evaluate_arc_agi(
    artifact_path: Path,
    amendment_path: Path,
    matrix_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    expected_hash = str(amendment["frozen_router"]["artifact_hash"])
    if artifact.get("artifact_hash") != expected_hash:
        raise ValueError("router artifact does not match the replication freeze")
    router = SemanticRouter(artifact)
    matrix, manifest = load_matrix(matrix_path, manifest_path)
    if list(router.models) != matrix.models:
        raise ValueError("router and replication model pools differ")

    indices = np.arange(len(matrix.texts), dtype=int)
    model_index = {model: index for index, model in enumerate(matrix.models)}
    predictions = np.asarray(
        [
            [router.predict(matrix.texts[row])[model] for model in matrix.models]
            for row in indices
        ],
        dtype=float,
    )
    selected_models = [
        router.route(matrix.texts[row], mode="balanced").selected_model
        for row in indices
    ]
    selections = np.asarray(
        [model_index[model] for model in selected_models],
        dtype=int,
    )
    selected_metrics = evaluate_selection(
        matrix,
        indices,
        selections,
        predictions=predictions,
    )
    fixed_metrics = {
        model: evaluate_selection(
            matrix,
            indices,
            np.full(len(indices), index, dtype=int),
        )
        for index, model in enumerate(matrix.models)
    }
    primary_model = str(amendment["comparison"]["primary_fixed_model"])
    primary_index = model_index[primary_model]
    primary_fixed = fixed_metrics[primary_model]
    posthoc_best = max(
        matrix.models,
        key=lambda model: (
            fixed_metrics[model]["macro_score"],
            -fixed_metrics[model]["total_cost_usd"],
            model,
        ),
    )
    selected_scores = matrix.scores[indices, selections]
    selected_costs = matrix.costs[indices, selections]
    fixed_scores = matrix.scores[indices, primary_index]
    fixed_costs = matrix.costs[indices, primary_index]
    report = {
        "schema_version": "arc-agi-sequential-replication-v1",
        "inferential_status": amendment["inferential_status"],
        "replaces_confirmatory_primary": False,
        "artifact_hash": artifact["artifact_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "prompt_count": len(indices),
        "selected": selected_metrics,
        "primary_fixed_model": primary_model,
        "primary_fixed": primary_fixed,
        "quality_difference_vs_primary_fixed": (
            selected_metrics["macro_score"] - primary_fixed["macro_score"]
        ),
        "cost_saving_fraction_vs_primary_fixed": (
            1.0
            - selected_metrics["total_cost_usd"]
            / primary_fixed["total_cost_usd"]
        ),
        "paired_bootstrap_vs_primary_fixed": _paired_bootstrap(
            selected_scores,
            fixed_scores,
            selected_costs,
            fixed_costs,
            resamples=int(amendment["comparison"]["paired_bootstrap_resamples"]),
            seed=int(amendment["comparison"]["paired_bootstrap_seed"]),
        ),
        "fixed_models": fixed_metrics,
        "descriptive_posthoc_best_fixed_model": posthoc_best,
        "descriptive_posthoc_best_fixed": fixed_metrics[posthoc_best],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen ARC-AGI sequential router replication."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "artifacts/"
            "active_router_protocol_amendment_006_arc_agi_replication.json"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/public_router_v1/open_source/semantic_router.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/public_router_v1/arc_agi"),
    )
    args = parser.parse_args()
    manifest = prepare_arc_agi(args.root, args.amendment, args.output_dir)
    report = evaluate_arc_agi(
        args.artifact,
        args.amendment,
        args.output_dir / "matrix.jsonl",
        args.output_dir / "manifest.json",
        args.output_dir / "results.json",
    )
    print(
        json.dumps(
            {
                "artifact_hash": report["artifact_hash"],
                "manifest_hash": manifest["manifest_hash"],
                "prompt_count": report["prompt_count"],
                "primary_fixed_model": report["primary_fixed_model"],
                "router_macro_score": report["selected"]["macro_score"],
                "primary_fixed_macro_score": report["primary_fixed"][
                    "macro_score"
                ],
                "quality_difference": report[
                    "quality_difference_vs_primary_fixed"
                ],
                "cost_saving_fraction": report[
                    "cost_saving_fraction_vs_primary_fixed"
                ],
                "posthoc_best_fixed_model": report[
                    "descriptive_posthoc_best_fixed_model"
                ],
                "bootstrap": report["paired_bootstrap_vs_primary_fixed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
