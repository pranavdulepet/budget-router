#!/usr/bin/env python3
"""Run a tightly capped two-turn tool-protocol smoke check on the Tinker pool."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.pilot import ToolProtocolEpisode
from budget_router.pricing import PriceSnapshot
from budget_router.providers import ModelRequest, ProviderError
from budget_router.providers.tinker import TinkerSamplingAdapter
from budget_router.serialization import stable_json
from budget_router.types import Operation, TokenUsage, TranscriptMessage

TOOL_SPEC = {
    "name": "workspace_probe",
    "description": "Read one visible workspace path without modifying it.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
SYSTEM_PROMPT = (
    "This is a tool-protocol compatibility check. Never reveal private reasoning. "
    "Use only the declared tool and follow its schema exactly."
)
USER_PROMPT = (
    "Call workspace_probe exactly once with path README.md. "
    "Do not answer the request directly."
)


def _treatment_effort(treatment: dict[str, Any]) -> float | None:
    reasoning = str(treatment["reasoning"])
    return float(reasoning.removeprefix("effort=")) if reasoning.startswith("effort=") else None


def _context_tokens(
    treatment: dict[str, Any],
    pool: dict[str, Any],
) -> int:
    value = treatment.get("context_tokens")
    if value is None:
        value = pool.get("context_tokens")
    if value is None:
        raise ValueError(
            f"missing context_tokens for treatment {treatment.get('model', '<unknown>')}"
        )
    return int(value)


def _cost(
    prices: PriceSnapshot,
    model: str,
    usage: TokenUsage,
) -> Decimal:
    return prices.price_for(model).conservative_cost(usage)


def _valid_call(response: Any) -> bool:
    return (
        len(response.tool_calls) == 1
        and response.tool_calls[0].name == "workspace_probe"
        and response.tool_calls[0].arguments == {"path": "README.md"}
        and not response.provider_metadata.get("malformed_tool_calls")
    )


async def _episode(
    treatment: dict[str, Any],
    prices: PriceSnapshot,
    *,
    max_tokens: int,
    seed: int,
    request_timeout_seconds: float,
) -> tuple[ToolProtocolEpisode, dict[str, Any]]:
    model = treatment["model"]
    started = time.monotonic()
    total_usage = TokenUsage()
    emitted = False
    arguments_valid = False
    tool_result_returned = False
    continued = False
    provider_failed = False
    provider_failure_reserved_usd = Decimal("0")
    error_type: str | None = None
    error_category: str | None = None
    first_latency_ms = 0
    second_latency_ms = 0
    try:
        adapter = await TinkerSamplingAdapter.from_environment(
            model,
            renderer_name=treatment["renderer"],
            effort=_treatment_effort(treatment),
            force_httpx_transport=True,
        )
        request = ModelRequest(
            model=model,
            messages=(
                TranscriptMessage("system", SYSTEM_PROMPT),
                TranscriptMessage("user", USER_PROMPT),
            ),
            operation=Operation.CONTINUE,
            max_output_tokens=max_tokens,
            temperature=float(treatment["temperature"]),
            top_p=float(treatment.get("top_p", 1.0)),
            top_k=int(treatment.get("top_k", -1)),
            timeout_seconds=request_timeout_seconds,
            seed=seed,
            tools=(TOOL_SPEC,),
        )
        first = await adapter.generate(request)
        total_usage += first.usage
        first_latency_ms = first.latency_ms
        emitted = bool(first.tool_calls)
        arguments_valid = _valid_call(first)
        if arguments_valid:
            from tinker_cookbook.renderers import ToolCall as CookbookToolCall

            call = first.tool_calls[0]
            renderer_call = CookbookToolCall(
                type="function",
                id=call.call_id or "probe-1",
                function={
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            )
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
                {
                    "role": "assistant",
                    "content": first.content,
                    "tool_calls": [renderer_call],
                },
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call.call_id or "probe-1",
                    "content": json.dumps(
                        {
                            "path": "README.md",
                            "exists": True,
                            "first_heading": "Budget Router",
                        },
                        separators=(",", ":"),
                    ),
                },
                {
                    "role": "user",
                    "content": "Acknowledge the tool result with the single word COMPLETE.",
                },
            ]
            tool_result_returned = True
            second_request = ModelRequest(
                model=model,
                messages=(TranscriptMessage("user", "Continue after the tool result."),),
                operation=Operation.CONTINUE,
                max_output_tokens=max_tokens,
                temperature=float(treatment["temperature"]),
                top_p=float(treatment.get("top_p", 1.0)),
                top_k=int(treatment.get("top_k", -1)),
                timeout_seconds=request_timeout_seconds,
                seed=seed + 1,
                tools=(TOOL_SPEC,),
            )
            second = await adapter._generate_from_messages(second_request, history)
            total_usage += second.usage
            second_latency_ms = second.latency_ms
            continued = bool(second.content.strip() or second.tool_calls)
    except Exception as exc:
        provider_failed = True
        error_type = type(exc).__name__
        current: BaseException | None = exc
        while current is not None:
            category = getattr(current, "category", None)
            if category is not None:
                error_category = str(getattr(category, "value", category))
                break
            current = current.__cause__
        price = prices.price_for(model)
        provider_failure_reserved_usd = price.conservative_cost(
            TokenUsage(
                input_tokens=price.context_tokens - max_tokens,
                output_tokens=max_tokens,
            )
        )

    elapsed = time.monotonic() - started
    cost = _cost(prices, model, total_usage) + provider_failure_reserved_usd
    episode = ToolProtocolEpisode(
        model=model,
        task_id="synthetic-tool-protocol-v1",
        emitted_tool_call=emitted,
        arguments_valid=arguments_valid,
        tool_result_returned=tool_result_returned,
        continued_after_tool=continued,
        terminal_record_valid=True,
        cost_usd=cost,
        latency_seconds=elapsed,
        provider_failed=provider_failed,
    )
    record = {
        "model": model,
        "renderer": treatment["renderer"],
        "reasoning": treatment["reasoning"],
        "temperature": treatment["temperature"],
        "top_p": treatment.get("top_p", 1.0),
        "top_k": treatment.get("top_k", -1),
        "seed": seed,
        "max_output_tokens_per_call": max_tokens,
        "emitted_tool_call": emitted,
        "arguments_valid": arguments_valid,
        "tool_result_returned": tool_result_returned,
        "continued_after_tool": continued,
        "structurally_valid": episode.structurally_valid,
        "provider_failed": provider_failed,
        "provider_failure_reserved_usd": str(provider_failure_reserved_usd),
        "error_type": error_type,
        "error_category": error_category,
        "input_tokens": total_usage.input_tokens,
        "output_tokens": total_usage.output_tokens,
        "conservative_cost_usd": str(cost),
        "latency_seconds": elapsed,
        "first_latency_ms": first_latency_ms,
        "second_latency_ms": second_latency_ms,
    }
    return episode, record


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    pool = json.loads(args.model_pool.read_text(encoding="utf-8"))
    treatments = pool["treatments"]
    if args.model is not None:
        treatments = [
            treatment
            for treatment in treatments
            if treatment["model"] == args.model
        ]
        if not treatments:
            raise SystemExit(f"model is not in the configured pool: {args.model}")
    prices = PriceSnapshot.load(args.prices)
    theoretical = sum(
        (
            prices.price_for(treatment["model"]).conservative_cost(
                TokenUsage(
                    input_tokens=(
                        _context_tokens(treatment, pool) - args.max_tokens
                    ),
                    output_tokens=args.max_tokens,
                )
            )
            * 2
            for treatment in treatments
        ),
        Decimal("0"),
    )
    if theoretical > args.hard_cap_usd:
        raise PermissionError(
            f"worst-case reservation {theoretical} exceeds cap {args.hard_cap_usd}"
        )
    records: list[dict[str, Any]] = []
    total = Decimal("0")
    for index, treatment in enumerate(treatments):
        _, record = await _episode(
            treatment,
            prices,
            max_tokens=args.max_tokens,
            seed=args.seed + index * 10,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        total += Decimal(record["conservative_cost_usd"])
        if total > args.hard_cap_usd:
            raise RuntimeError("observed smoke cost exceeded the hard cap")
        records.append(record)
    return {
        "schema_version": "tinker-tool-protocol-smoke-v1",
        "pool_version": pool["pool_version"],
        "price_snapshot": prices.snapshot_id,
        "hard_cap_usd": str(args.hard_cap_usd),
        "worst_case_reserved_usd": str(theoretical),
        "observed_conservative_cost_usd": str(total),
        "all_structurally_valid": all(
            record["structurally_valid"] for record in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pool", type=Path, default=Path("configs/model_pool.json"))
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-25.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/tinker_smoke.json"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-seconds", type=float, default=180)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--hard-cap-usd", type=Decimal, default=Decimal("1.00"))
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8_000:
        raise SystemExit("--max-tokens must be in [1, 8000]")
    if args.request_timeout_seconds <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")
    result = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "records"
            },
            indent=2,
            sort_keys=True,
        )
    )
    for record in result["records"]:
        print(
            json.dumps(
                {
                    "model": record["model"],
                    "structurally_valid": record["structurally_valid"],
                    "provider_failed": record["provider_failed"],
                    "error_type": record["error_type"],
                    "cost_usd": record["conservative_cost_usd"],
                    "latency_seconds": round(record["latency_seconds"], 3),
                },
                sort_keys=True,
            )
        )
    if not result["all_structurally_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
