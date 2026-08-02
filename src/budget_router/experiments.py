from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .serialization import stable_hash, stable_json

SUPPORTED_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
POST_MATRIX_ALLOCATION = {
    "checkpoint_acquisition": Decimal("0.50"),
    "router_training": Decimal("0.10"),
    "on_policy_validation_test": Decimal("0.40"),
}
CHECKPOINT_ALLOCATION = {
    "model_forks": Decimal("0.50"),
    "operation_handoff_forks": Decimal("0.30"),
    "parallel_branches": Decimal("0.20"),
}


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    repository: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    train: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]
    repositories: Mapping[str, str]
    protocol_version: str = "swebench-repo-split-v1"
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        all_tasks = self.train + self.calibration + self.test
        if len(all_tasks) != len(set(all_tasks)):
            raise ValueError("tasks overlap across splits")
        payload = self.to_dict(include_hash=False)
        expected = stable_hash(payload)
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("split manifest hash mismatch")
        object.__setattr__(self, "manifest_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "protocol_version": self.protocol_version,
            "seed": self.seed,
            "train": list(self.train),
            "calibration": list(self.calibration),
            "test": list(self.test),
            "repositories": dict(self.repositories),
        }
        if include_hash:
            result["manifest_hash"] = self.manifest_hash
        return result

    def freeze(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("manifest_hash") != self.manifest_hash:
                raise FileExistsError("refusing to overwrite a different frozen manifest")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(stable_json(self.to_dict()) + "\n", encoding="utf-8")


def repository_split(
    tasks: Sequence[TaskRecord],
    *,
    seed: int = 20260725,
    targets: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> SplitManifest:
    if not tasks:
        raise ValueError("tasks must not be empty")
    if not math.isclose(sum(targets), 1.0):
        raise ValueError("split targets must sum to one")
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    for task in tasks:
        grouped[task.repository].append(task)
    rng = random.Random(seed)
    repositories = sorted(grouped)
    rng.shuffle(repositories)
    # Large repositories are allocated first; seeded hash breaks equal-size ties.
    repositories.sort(
        key=lambda repo: (
            -len(grouped[repo]),
            hashlib.sha256(f"{seed}:{repo}".encode()).hexdigest(),
        )
    )
    names = ("train", "calibration", "test")
    target_counts = {
        name: targets[index] * len(tasks) for index, name in enumerate(names)
    }
    counts = dict.fromkeys(names, 0)
    task_splits: dict[str, list[str]] = {name: [] for name in names}
    repository_splits: dict[str, str] = {}
    for repository in repositories:
        split = max(
            names,
            key=lambda name: (
                target_counts[name] - counts[name],
                -names.index(name),
            ),
        )
        repository_splits[repository] = split
        identifiers = sorted(task.task_id for task in grouped[repository])
        task_splits[split].extend(identifiers)
        counts[split] += len(identifiers)
    return SplitManifest(
        seed=seed,
        train=tuple(sorted(task_splits["train"])),
        calibration=tuple(sorted(task_splits["calibration"])),
        test=tuple(sorted(task_splits["test"])),
        repositories=repository_splits,
    )


def derive_budget_levels(
    successful_trajectory_costs: Sequence[Decimal | float | str],
) -> tuple[Decimal, ...]:
    if not successful_trajectory_costs:
        raise ValueError("successful training trajectories are required")
    ordered = sorted(Decimal(str(value)) for value in successful_trajectory_costs)
    if ordered[0] < 0:
        raise ValueError("costs must be non-negative")
    result: list[Decimal] = []
    for quantile in SUPPORTED_QUANTILES:
        index = max(0, math.ceil(quantile * len(ordered)) - 1)
        result.append(ordered[index].quantize(Decimal("0.01"), rounding=ROUND_CEILING))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FixedMatrixPlan:
    task_ids: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...] = (0, 1, 2)

    @property
    def run_count(self) -> int:
        return len(self.task_ids) * len(self.models) * len(self.seeds)

    def validate_reference_design(self) -> None:
        if len(self.task_ids) != 500:
            raise ValueError("reference fixed matrix requires all 500 tasks")
        if len(self.models) != 5:
            raise ValueError("reference fixed matrix requires five models")
        if len(self.seeds) != 3:
            raise ValueError("reference fixed matrix requires three seeds")

    def runs(self) -> Iterable[tuple[str, str, int]]:
        for task_id in self.task_ids:
            for model in self.models:
                for seed in self.seeds:
                    yield task_id, model, seed


@dataclass(frozen=True, slots=True)
class PilotModelResult:
    model: str
    episodes: int
    structurally_valid: int
    projected_mean_cost_usd: Decimal
    throughput_tasks_per_hour: float
    failure_rate: float

    @property
    def passes(self) -> bool:
        return self.episodes == 12 and self.structurally_valid >= 10


@dataclass(frozen=True, slots=True)
class PilotGate:
    results: tuple[PilotModelResult, ...]
    projected_matrix_cost_usd: Decimal
    required_credit_usd: Decimal
    approved_credit_usd: Decimal | None = None

    @property
    def structurally_valid(self) -> bool:
        return bool(self.results) and all(result.passes for result in self.results)

    @property
    def credit_approved(self) -> bool:
        return (
            self.approved_credit_usd is not None
            and self.approved_credit_usd >= self.required_credit_usd
        )

    @property
    def may_collect(self) -> bool:
        return self.structurally_valid and self.credit_approved

    def require_collection_approval(self) -> None:
        if not self.structurally_valid:
            failed = [result.model for result in self.results if not result.passes]
            raise RuntimeError(f"pilot protocol gate failed for: {', '.join(failed)}")
        if not self.credit_approved:
            raise PermissionError(
                "credit approval is required; the 500×5×3 matrix is never auto-reduced"
            )


def project_matrix_cost(
    results: Sequence[PilotModelResult],
    *,
    tasks: int = 500,
    seeds: int = 3,
    contingency: Decimal = Decimal("0.10"),
) -> Decimal:
    if not results:
        raise ValueError("pilot results required")
    raw = sum(
        (result.projected_mean_cost_usd * tasks * seeds for result in results),
        Decimal("0"),
    )
    return (raw * (Decimal("1") + contingency)).quantize(Decimal("0.01"))


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    checkpoint_id: str
    kind: str
    uncertainty: float
    expected_information_gain: float
    expected_value: float
    estimated_cost_usd: Decimal

    @property
    def acquisition_score(self) -> float:
        cost = max(float(self.estimated_cost_usd), 1e-6)
        kind_bonus = {
            "early": 1.05,
            "uncertain": 1.15,
            "failure": 1.10,
            "high_value": 1.08,
        }.get(self.kind, 1.0)
        return (
            kind_bonus
            * (0.45 * self.uncertainty + 0.4 * self.expected_information_gain
               + 0.15 * self.expected_value)
            / cost
        )


def select_checkpoint_counterfactuals(
    candidates: Sequence[CheckpointCandidate],
    *,
    budget_usd: Decimal,
) -> tuple[CheckpointCandidate, ...]:
    remaining = Decimal(budget_usd)
    selected: list[CheckpointCandidate] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.acquisition_score, item.checkpoint_id), reverse=True
    ):
        if candidate.estimated_cost_usd <= remaining:
            selected.append(candidate)
            remaining -= candidate.estimated_cost_usd
    return tuple(selected)


def allocate_credits(
    remaining_credit_usd: Decimal,
) -> dict[str, Decimal | dict[str, Decimal]]:
    total = Decimal(remaining_credit_usd)
    if total < 0:
        raise ValueError("remaining credit must be non-negative")
    top = {name: total * share for name, share in POST_MATRIX_ALLOCATION.items()}
    checkpoint_total = top["checkpoint_acquisition"]
    nested = {
        name: checkpoint_total * share for name, share in CHECKPOINT_ALLOCATION.items()
    }
    return {**top, "checkpoint_breakdown": nested}


def effective_sample_size(importance_weights: Sequence[float]) -> float:
    if not importance_weights:
        return 0.0
    if any(weight < 0 for weight in importance_weights):
        raise ValueError("importance weights must be non-negative")
    total = sum(importance_weights)
    squared = sum(weight * weight for weight in importance_weights)
    return total * total / squared if squared else 0.0


def offline_policy_coverage_gate(
    importance_weights: Sequence[float],
    *,
    minimum_effective_sample_size: float,
) -> bool:
    if minimum_effective_sample_size <= 0:
        raise ValueError("minimum ESS must be positive")
    return effective_sample_size(importance_weights) >= minimum_effective_sample_size
