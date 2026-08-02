from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from budget_router.semantic import SemanticRouter
from run_public_router_study import evaluate_selection, load_matrix


def evaluate_artifact(
    artifact_path: Path,
    matrix_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    router = SemanticRouter(artifact)
    matrix, manifest = load_matrix(matrix_path, manifest_path)
    if list(router.models) != matrix.models:
        raise ValueError("semantic artifact and matrix model pools differ")
    model_index = {model: index for index, model in enumerate(matrix.models)}
    results: dict[str, Any] = {}
    for split in ("id_test", "ood_test"):
        indices = matrix.indices(split)
        predictions = np.asarray(
            [
                [router.predict(matrix.texts[row])[model] for model in matrix.models]
                for row in indices
            ],
            dtype=float,
        )
        selections = np.asarray(
            [
                model_index[
                    router.route(
                        matrix.texts[row],
                        mode="balanced",
                    ).selected_model
                ]
                for row in indices
            ],
            dtype=int,
        )
        selected = evaluate_selection(
            matrix,
            indices,
            selections,
            predictions=predictions,
        )
        calibration_best = str(artifact["selector"]["calibration_best_fixed"])
        fixed = evaluate_selection(
            matrix,
            indices,
            np.full(len(indices), model_index[calibration_best], dtype=int),
        )
        results[split] = {
            "selected": selected,
            "calibration_best_fixed_model": calibration_best,
            "calibration_best_fixed": fixed,
            "macro_score_difference": (
                selected["macro_score"] - fixed["macro_score"]
            ),
            "cost_saving_fraction": (
                1.0 - selected["total_cost_usd"] / fixed["total_cost_usd"]
            ),
        }
    report = {
        "schema_version": "public-semantic-artifact-evaluation-v1",
        "artifact_hash": artifact["artifact_hash"],
        "prepared_manifest_hash": manifest["manifest_hash"],
        "default_lambda": artifact["selector"]["default_lambda"],
        "test": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen deployable semantic artifact."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/public_router_v1/open_source/semantic_router.json"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/public_router_v1/prepared/matrix.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/public_router_v1/prepared/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/public_router_v1/open_source/evaluation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_artifact(
                args.artifact,
                args.matrix,
                args.manifest,
                args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
