from __future__ import annotations

import argparse
import hashlib
import json
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
    unhashed = dict(artifact)
    claimed_hash = str(unhashed.pop("artifact_hash", ""))
    if stable_hash(unhashed) != claimed_hash:
        raise ValueError("gate artifact content hash is invalid")

    candidate = str(artifact["selected_candidate_policy_id"])
    fixed = str(artifact["fixed_policy_id"])
    selector = artifact["selector"]
    head = HashedLinearModelHead.from_artifact(artifact["model_head"])
    goal = GoalContext(issue_text, repository, (candidate, fixed))
    probabilities = {
        name: head.estimate(
            goal,
            RouterState(),
            name,
        ).success_probability
        for name in ("candidate_win", "fixed_loss")
    }
    cost_component = float(selector["cost_weight"]) * float(
        selector["expected_mean_saving_ratio"]
    )
    score = (
        probabilities["candidate_win"]
        - probabilities["fixed_loss"]
        + cost_component
    )
    threshold = float(selector["threshold"])
    selected = candidate if score >= threshold else fixed
    return {
        "schema_version": "isolated-stage-policy-decision-v1",
        "artifact_hash": claimed_hash,
        "selected_policy_id": selected,
        "candidate_policy_id": candidate,
        "fixed_policy_id": fixed,
        "candidate_win_probability": probabilities["candidate_win"],
        "fixed_loss_probability": probabilities["fixed_loss"],
        "cost_component": cost_component,
        "score": score,
        "threshold": threshold,
        "repository_identity_used_as_feature": False,
        "warning": (
            "Probabilities are cohort-level estimates from the frozen study, "
            "not guarantees for this task."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Choose fixed Qwen or the frozen isolated GPT-OSS-to-Qwen policy "
            "from public issue text without a provider call."
        )
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("artifacts/isolated_stage_v1_task_gate.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/isolated_stage_v1_results.json"),
        help="Sanitized result containing the pre-test gate authentication.",
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
        help="Optional repository label; recorded but not used as a feature.",
    )
    args = parser.parse_args()

    gate_bytes = args.gate.read_bytes()
    artifact = json.loads(gate_bytes)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    freeze = result["freezes"]
    if hashlib.sha256(gate_bytes).hexdigest() != freeze[
        "gate_artifact_sha256"
    ]:
        raise ValueError("gate bytes differ from the pre-test freeze")
    if artifact.get("artifact_hash") != freeze["gate_artifact_hash"]:
        raise ValueError("gate content hash differs from the pre-test freeze")
    issue_text = (
        str(args.issue)
        if args.issue is not None
        else args.issue_file.read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            route_issue(artifact, issue_text, repository=args.repository),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
