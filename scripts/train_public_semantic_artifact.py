from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from budget_router.redaction import scan_for_secrets
from budget_router.semantic import ModelCard, SemanticOutcome, train_semantic_router
from budget_router.serialization import stable_json


PROVIDERS = {
    "claude-sonnet-4": "anthropic",
    "deepseek-v3-0324": "deepseek",
    "deepseek-v3.1-terminus": "deepseek",
    "deepseek-r1-0528": "deepseek",
    "gemini-2.5-flash": "google",
    "gemini-2.5-pro": "google",
    "gpt-5-chat": "openai",
    "gpt-5": "openai",
    "qwen3-235b-a22b-2507": "qwen",
    "qwen3-235b-a22b-thinking-2507": "qwen",
    "glm-4.6": "z-ai",
    "kimi-k2-0905": "moonshot",
    "intern-s1": "internlm",
}


def train_public_artifact(
    matrix_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = [str(model) for model in manifest["models"]]
    costs: dict[str, list[float]] = defaultdict(list)
    outcomes: list[SemanticOutcome] = []
    with matrix_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["route_split"] not in {"train", "calibration"}:
                continue
            request_id = f"{row['dataset']}:{row['prompt_key']}"
            for model in models:
                outcome = row["outcomes"][model]
                outcomes.append(
                    SemanticOutcome(
                        request_id=request_id,
                        text=str(row["origin_query"]),
                        model=model,
                        acceptable=bool(outcome["acceptable"]),
                        split=str(row["route_split"]),
                        cost_usd=float(outcome["cost_usd"]),
                        group=str(row["dataset"]),
                    )
                )
                if row["route_split"] == "train":
                    costs[model].append(float(outcome["cost_usd"]))
    cards = [
        ModelCard(
            name=model,
            provider=PROVIDERS[model],
            expected_cost_usd=sum(costs[model]) / len(costs[model]),
            capabilities=frozenset({"text"}),
            metadata={"source": "LLMRouterBench pinned public matrix"},
        )
        for model in models
    ]
    artifact = train_semantic_router(
        outcomes,
        cards,
        dimension=65_536,
        backend="sklearn",
        lambdas=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
        noninferiority_margin=0.005,
        random_seed=20_260_729,
    )
    if scan_for_secrets(artifact):
        raise ValueError("refusing to write a possible secret")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "semantic_router.json"
    cards_path = output_dir / "model_cards.json"
    artifact_path.write_text(stable_json(artifact) + "\n", encoding="utf-8")
    cards_path.write_text(
        json.dumps(
            {"models": [card.to_dict() for card in cards]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "artifact": str(artifact_path),
        "artifact_hash": artifact["artifact_hash"],
        "model_cards": str(cards_path),
        "models": models,
        "training": artifact["training"],
        "selector": artifact["selector"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the deployable provider-neutral artifact on public data."
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
        "--output-dir",
        type=Path,
        default=Path("outputs/public_router_v1/open_source"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train_public_artifact(args.matrix, args.manifest, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
