from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .pricing import PriceSnapshot
from .trace import RouterTrace, TraceWriter
from .types import TokenUsage


@dataclass(frozen=True, slots=True)
class ReplayReport:
    records: int
    conservative_spent_usd: Decimal
    billed_spent_usd: Decimal
    hard_cap_violations: int
    snapshot_id: str
    deterministic: bool


def replay(
    trace_path: str | Path,
    prices: PriceSnapshot,
    *,
    hard_limit_usd: Decimal | str,
) -> ReplayReport:
    records = TraceWriter.read(trace_path)
    conservative = Decimal("0")
    billed = Decimal("0")
    violations = 0
    seen: set[tuple[str, int]] = set()
    deterministic = True
    hard_limit = Decimal(str(hard_limit_usd))

    for trace in records:
        if trace.price_snapshot != prices.snapshot_id:
            raise ValueError(
                f"trace snapshot {trace.price_snapshot!r} != {prices.snapshot_id!r}"
            )
        identity = (trace.run_id, trace.turn_id)
        if identity in seen:
            deterministic = False
        seen.add(identity)
        if trace.event is None:
            continue
        usage_data = trace.event.get("usage", {})
        usage = TokenUsage(
            input_tokens=int(usage_data.get("input_tokens", 0)),
            cached_input_tokens=int(usage_data.get("cached_input_tokens", 0)),
            output_tokens=int(usage_data.get("output_tokens", 0)),
        )
        model = trace.event.get("model") or trace.selected_action.get("model")
        if model and (usage.input_tokens or usage.output_tokens):
            price = prices.price_for(str(model))
            conservative += price.conservative_cost(usage)
            billed += price.billed_cost(usage)
        if conservative > hard_limit:
            violations += 1
    return ReplayReport(
        records=len(records),
        conservative_spent_usd=conservative,
        billed_spent_usd=billed,
        hard_cap_violations=violations,
        snapshot_id=prices.snapshot_id,
        deterministic=deterministic and violations == 0,
    )

