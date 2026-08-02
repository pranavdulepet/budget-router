#!/usr/bin/env python3
"""Exercise sliding context budgeting on one 32K and one 64K Tinker model."""

from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.mini_swe_tinker import BASH_TOOL_SPEC
from budget_router.pricing import PriceSnapshot
from budget_router.providers import ModelRequest
from budget_router.providers.tinker import TinkerSamplingAdapter
from budget_router.serialization import stable_json
from budget_router.types import Operation, TokenUsage, TranscriptMessage


MODELS = (
    "openai/gpt-oss-20b",
    "Qwen/Qwen3.6-35B-A3B",
)


def _effort(treatment: dict[str, Any]) -> float | None:
    value = str(treatment["reasoning"])
    return float(value.removeprefix("effort=")) if value.startswith("effort=") else None


def _long_history(exchanges: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Use the declared tool exactly as requested.",
        },
        {
            "role": "user",
            "content": "Maintain the coding workspace and follow later instructions.",
        },
    ]
    filler = "Prior visible diagnostic context. " + ("x" * 9_968)
    for index in range(exchanges):
        messages.extend(
            [
                {"role": "assistant", "content": f"{index}: {filler}"},
                {"role": "user", "content": "Continue."},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "Call bash exactly once with command pwd.",
        }
    )
    return messages


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    pool = json.loads(args.model_pool.read_text(encoding="utf-8"))
    treatments = {
        str(row["model"]): dict(row) for row in pool["treatments"]
    }
    prices = PriceSnapshot.load(args.prices)
    selected_models = tuple(args.models or MODELS)
    worst_case = sum(
        (
            prices.price_for(model).conservative_cost(
                TokenUsage(
                    input_tokens=(
                        int(treatments[model]["context_tokens"]) - args.max_tokens
                    ),
                    output_tokens=args.max_tokens,
                )
            )
            for model in selected_models
        ),
        Decimal("0"),
    )
    if worst_case > args.hard_cap_usd:
        raise PermissionError(
            f"worst-case reservation {worst_case} exceeds cap {args.hard_cap_usd}"
        )

    records = []
    total = Decimal("0")
    for index, model in enumerate(selected_models):
        treatment = treatments[model]
        adapter = await TinkerSamplingAdapter.from_environment(
            model,
            renderer_name=treatment["renderer"],
            effort=_effort(treatment),
            force_httpx_transport=True,
        )
        request = ModelRequest(
            model=model,
            messages=(TranscriptMessage("user", "Continue."),),
            operation=Operation.CONTINUE,
            max_output_tokens=args.max_tokens,
            temperature=float(treatment["temperature"]),
            top_p=float(treatment.get("top_p", 1.0)),
            top_k=int(treatment.get("top_k", -1)),
            timeout_seconds=args.request_timeout_seconds,
            seed=args.seed + index,
            tools=(BASH_TOOL_SPEC,),
            metadata={"context_tokens": treatment["context_tokens"]},
        )
        response = await adapter._generate_from_messages(
            request,
            _long_history(args.exchanges),
        )
        usage_cost = prices.price_for(model).conservative_cost(response.usage)
        total += usage_cost
        records.append(
            {
                "model": model,
                "context_tokens": treatment["context_tokens"],
                "history_messages_dropped": response.provider_metadata.get(
                    "history_messages_dropped"
                ),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "tool_call_valid": (
                    len(response.tool_calls) == 1
                    and response.tool_calls[0].name == "bash"
                    and response.tool_calls[0].arguments == {"command": "pwd"}
                ),
                "conservative_cost_usd": str(usage_cost),
            }
        )
    return {
        "schema_version": "tinker-context-budget-smoke-v1",
        "hard_cap_usd": str(args.hard_cap_usd),
        "worst_case_reserved_usd": str(worst_case),
        "observed_conservative_cost_usd": str(total),
        "all_passed": all(
            int(row["history_messages_dropped"] or 0) > 0
            and bool(row["tool_call_valid"])
            for row in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_router_v1.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-29_router_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/active_router_v1/smoke_context_budget.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODELS,
        dest="models",
    )
    parser.add_argument("--exchanges", type=int, default=60)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=202607390)
    parser.add_argument("--hard-cap-usd", type=Decimal, default=Decimal("0.05"))
    args = parser.parse_args()
    if args.exchanges <= 0:
        raise SystemExit("--exchanges must be positive")
    result = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )
    for row in result["records"]:
        print(json.dumps(row, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
