from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from .experiments import PilotModelResult


@dataclass(frozen=True, slots=True)
class ToolProtocolEpisode:
    model: str
    task_id: str
    emitted_tool_call: bool
    arguments_valid: bool
    tool_result_returned: bool
    continued_after_tool: bool
    terminal_record_valid: bool
    cost_usd: Decimal
    latency_seconds: float
    provider_failed: bool = False

    @property
    def structurally_valid(self) -> bool:
        return (
            self.emitted_tool_call
            and self.arguments_valid
            and self.tool_result_returned
            and self.continued_after_tool
            and self.terminal_record_valid
            and not self.provider_failed
        )


def summarize_tool_protocol_pilot(
    episodes: Sequence[ToolProtocolEpisode],
    *,
    expected_tasks_per_model: int = 12,
) -> tuple[PilotModelResult, ...]:
    if not episodes:
        raise ValueError("pilot episodes are required")
    grouped: dict[str, list[ToolProtocolEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.model].append(episode)
    results: list[PilotModelResult] = []
    for model, rows in sorted(grouped.items()):
        if len(rows) != expected_tasks_per_model:
            raise ValueError(
                f"{model} has {len(rows)} episodes; expected {expected_tasks_per_model}"
            )
        total_seconds = sum(row.latency_seconds for row in rows)
        results.append(
            PilotModelResult(
                model=model,
                episodes=len(rows),
                structurally_valid=sum(row.structurally_valid for row in rows),
                projected_mean_cost_usd=sum(
                    (row.cost_usd for row in rows), Decimal("0")
                )
                / len(rows),
                throughput_tasks_per_hour=(
                    len(rows) * 3_600 / total_seconds if total_seconds > 0 else 0.0
                ),
                failure_rate=sum(row.provider_failed for row in rows) / len(rows),
            )
        )
    return tuple(results)


def replacement_for(
    incompatible_model: str,
    attempted_replacements: Sequence[str],
    *,
    replacement_order: Sequence[str] = (
        "Qwen/Qwen3.5-4B",
        "openai/gpt-oss-120b:peft:131072",
        "Qwen/Qwen3.5-397B-A17B",
    ),
) -> str:
    attempted = set(attempted_replacements)
    for model in replacement_order:
        if model not in attempted:
            return model
    raise RuntimeError(f"replacement tiers exhausted for {incompatible_model}")
