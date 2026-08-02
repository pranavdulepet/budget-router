from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.serialization import stable_hash
from budget_router.task_model import HashedLinearModelHead
from budget_router.types import GoalContext, RouterState


def route_issue(
    artifact: dict[str, Any],
    issue_text: str,
    *,
    repository: str = "",
) -> dict[str, Any]:
    if not issue_text.strip():
        raise ValueError("issue text must not be empty")
    claimed_hash = str(artifact.get("artifact_hash", ""))
    artifact_without_hash = dict(artifact)
    artifact_without_hash.pop("artifact_hash", None)
    if stable_hash(artifact_without_hash) != claimed_hash:
        raise ValueError("router artifact content hash is invalid")

    models = [str(model) for model in artifact["models"]]
    head = HashedLinearModelHead.from_artifact(artifact["model_head"])
    expected_costs = {
        model: Decimal(str(cost))
        for model, cost in artifact["selector"]["expected_cost_usd"].items()
    }
    if set(models) != set(expected_costs):
        raise ValueError("router models and selector costs do not match")
    cost_penalty = float(artifact["selector"]["cost_penalty"])
    cost_scale = max(expected_costs.values())
    goal = GoalContext(issue_text, repository, tuple(models))
    probabilities = {
        model: head.estimate(goal, RouterState(), model).success_probability
        for model in models
    }
    scores = {
        model: probabilities[model]
        - cost_penalty * float(expected_costs[model] / cost_scale)
        for model in models
    }
    selected = max(
        models,
        key=lambda model: (
            scores[model],
            probabilities[model],
            -expected_costs[model],
        ),
    )
    return {
        "schema_version": "task-router-decision-v1",
        "artifact_hash": claimed_hash,
        "selected_model": selected,
        "selected_predicted_success": probabilities[selected],
        "selected_expected_cost_usd": str(expected_costs[selected]),
        "probabilities": probabilities,
        "scores": scores,
        "expected_cost_usd": {
            model: str(expected_costs[model]) for model in models
        },
        "repository_identity_used_as_feature": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route public issue text with a frozen static task router."
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=Path("artifacts/router_baseline_v2_task_router.json"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--issue", help="Public issue text to route.")
    source.add_argument(
        "--issue-file",
        type=Path,
        help="UTF-8 file containing public issue text.",
    )
    parser.add_argument(
        "--repository",
        default="",
        help="Optional repository label; recorded for the runtime contract only.",
    )
    args = parser.parse_args()
    issue_text = (
        str(args.issue)
        if args.issue is not None
        else args.issue_file.read_text(encoding="utf-8")
    )
    artifact = json.loads(args.router.read_text(encoding="utf-8"))
    print(
        json.dumps(
            route_issue(artifact, issue_text, repository=args.repository),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
