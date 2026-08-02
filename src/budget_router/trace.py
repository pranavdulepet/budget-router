from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .pricing import PriceSnapshot
from .redaction import redact, scan_for_secrets
from .serialization import stable_hash, stable_json, to_jsonable
from .types import GoalContext, RouterDecision, RouterEvent, RouterState, as_usd

TRACE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class RouterTrace:
    task_id: str
    run_id: str
    turn_id: int
    goal: Mapping[str, Any]
    visible_state: Mapping[str, Any]
    eligible_actions: tuple[str, ...]
    action_propensities: Mapping[str, float]
    selected_action: Mapping[str, Any]
    action_estimates: tuple[Mapping[str, Any], ...]
    remaining_budget_usd: Decimal
    conservative_spent_usd: Decimal
    billed_spent_usd: Decimal
    price_snapshot: str
    price_provenance: str
    policy_version: str
    model_version: str | None = None
    event: Mapping[str, Any] | None = None
    parent_run_id: str | None = None
    stopping_reason: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = TRACE_SCHEMA_VERSION
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining_budget_usd", as_usd(self.remaining_budget_usd))
        object.__setattr__(
            self, "conservative_spent_usd", as_usd(self.conservative_spent_usd)
        )
        object.__setattr__(self, "billed_spent_usd", as_usd(self.billed_spent_usd))
        if self.turn_id < 0:
            raise ValueError("turn_id must be non-negative")
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {self.schema_version!r}")
        if not self.eligible_actions:
            raise ValueError("trace requires at least one eligible action")
        if set(self.action_propensities) != set(self.eligible_actions):
            raise ValueError("propensities must cover exactly the eligible actions")
        if any(not 0.0 <= value <= 1.0 for value in self.action_propensities.values()):
            raise ValueError("action propensities must be in [0, 1]")
        if abs(sum(self.action_propensities.values()) - 1.0) > 1e-6:
            raise ValueError("action propensities must sum to one")
        operation = str(self.selected_action.get("operation", ""))
        model = self.selected_action.get("model") or "-"
        share = self.selected_action.get("branch_share")
        selected_key = f"{operation}:{model}" + (
            f":{share}" if share is not None else ""
        )
        if (
            selected_key not in self.action_propensities
            or self.action_propensities[selected_key] <= 0
        ):
            raise ValueError("selected action must have positive logged propensity")
        payload = self.to_dict(include_hash=False)
        expected = stable_hash(payload)
        if self.record_hash and self.record_hash != expected:
            raise ValueError("record_hash does not match trace contents")
        object.__setattr__(self, "record_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "goal": to_jsonable(self.goal),
            "visible_state": to_jsonable(self.visible_state),
            "eligible_actions": list(self.eligible_actions),
            "action_propensities": dict(self.action_propensities),
            "selected_action": to_jsonable(self.selected_action),
            "action_estimates": to_jsonable(self.action_estimates),
            "remaining_budget_usd": str(self.remaining_budget_usd),
            "conservative_spent_usd": str(self.conservative_spent_usd),
            "billed_spent_usd": str(self.billed_spent_usd),
            "price_snapshot": self.price_snapshot,
            "price_provenance": self.price_provenance,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "event": to_jsonable(self.event),
            "stopping_reason": self.stopping_reason,
        }
        if include_hash:
            result["record_hash"] = self.record_hash
        return result

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> RouterTrace:
        return cls(
            task_id=str(item["task_id"]),
            run_id=str(item["run_id"]),
            parent_run_id=item.get("parent_run_id"),
            turn_id=int(item["turn_id"]),
            timestamp=str(item["timestamp"]),
            goal=item["goal"],
            visible_state=item["visible_state"],
            eligible_actions=tuple(item["eligible_actions"]),
            action_propensities={
                str(key): float(value)
                for key, value in item["action_propensities"].items()
            },
            selected_action=item["selected_action"],
            action_estimates=tuple(item["action_estimates"]),
            remaining_budget_usd=item["remaining_budget_usd"],
            conservative_spent_usd=item["conservative_spent_usd"],
            billed_spent_usd=item["billed_spent_usd"],
            price_snapshot=str(item["price_snapshot"]),
            price_provenance=str(item["price_provenance"]),
            policy_version=str(item["policy_version"]),
            model_version=item.get("model_version"),
            event=item.get("event"),
            stopping_reason=item.get("stopping_reason"),
            schema_version=str(item.get("schema_version", TRACE_SCHEMA_VERSION)),
            record_hash=str(item.get("record_hash", "")),
        )

    @classmethod
    def from_turn(
        cls,
        *,
        task_id: str,
        run_id: str,
        goal: GoalContext,
        state: RouterState,
        decision: RouterDecision,
        event: RouterEvent | None,
        remaining_budget_usd: Decimal,
        conservative_spent_usd: Decimal,
        billed_spent_usd: Decimal,
        prices: PriceSnapshot,
        propensities: Mapping[str, float] | None = None,
        parent_run_id: str | None = None,
    ) -> RouterTrace:
        sanitized_goal = redact(to_jsonable(goal))
        sanitized_state = redact(to_jsonable(state))
        sanitized_event = redact(to_jsonable(event)) if event else None
        estimates = tuple(to_jsonable(item) for item in decision.action_estimates)
        eligible = tuple(item.action.key for item in decision.action_estimates)
        if propensities is None:
            propensities = {
                action: float(action == decision.action.key) for action in eligible
            }
        trace = cls(
            task_id=task_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            turn_id=state.turn,
            goal=sanitized_goal,
            visible_state=sanitized_state,
            eligible_actions=eligible,
            action_propensities=propensities,
            selected_action=to_jsonable(decision.action),
            action_estimates=estimates,
            remaining_budget_usd=remaining_budget_usd,
            conservative_spent_usd=conservative_spent_usd,
            billed_spent_usd=billed_spent_usd,
            price_snapshot=prices.snapshot_id,
            price_provenance=prices.source_url,
            policy_version=decision.policy_version,
            model_version=(
                (event.model if event is not None else None)
                or decision.action.model
                or state.current_model
            ),
            event=sanitized_event,
            stopping_reason=(
                event.outcome.reason if event and event.outcome is not None else None
            ),
        )
        findings = scan_for_secrets(trace.to_dict())
        if findings:
            raise ValueError(f"trace contains possible secrets at: {', '.join(findings)}")
        return trace


class TraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: RouterTrace) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(stable_json(trace.to_dict()))
            stream.write("\n")

    @staticmethod
    def read(path: str | Path) -> list[RouterTrace]:
        traces: list[RouterTrace] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    traces.append(RouterTrace.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid trace at line {line_number}: {exc}") from exc
        return traces

    @staticmethod
    def to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet support requires budget-router[data]") from exc
        rows = [trace.to_dict() for trace in TraceWriter.read(jsonl_path)]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)


def query_traces(path: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Query JSONL or Parquet through DuckDB; ``trace`` is the table name."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB support requires budget-router[data]") from exc
    path = Path(path)
    escaped_path = str(path).replace("'", "''")
    reader = (
        f"read_parquet('{escaped_path}')"
        if path.suffix == ".parquet"
        else f"read_json_auto('{escaped_path}')"
    )
    connection = duckdb.connect(":memory:")
    connection.execute(f"CREATE VIEW trace AS SELECT * FROM {reader}")
    return connection.execute(sql).fetchall()
