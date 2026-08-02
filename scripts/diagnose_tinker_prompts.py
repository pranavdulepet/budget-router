#!/usr/bin/env python3
"""Diagnose Tinker prompt-dependent sampling stalls with a capped A/B probe."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from budget_router.mini_swe_tinker import BASH_TOOL_SPEC
from budget_router.pricing import PriceSnapshot
from budget_router.providers import ModelRequest
from budget_router.providers.tinker import TinkerSamplingAdapter
from budget_router.redaction import redact, scan_for_secrets
from budget_router.serialization import stable_json
from budget_router.tinker_diagnostics import render_tinker_prompt
from budget_router.types import Operation, TokenUsage, TranscriptMessage

WORKSPACE_TOOL_SPEC = {
    "name": "workspace_probe",
    "description": "Read one visible workspace path without modifying it.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
CONTROL_SYSTEM = (
    "This is a tool-protocol compatibility check. Never reveal private reasoning. "
    "Use only the declared tool and follow its schema exactly."
)
CONTROL_USER = (
    "Call workspace_probe exactly once with path README.md. "
    "Do not answer the request directly."
)
SHORT_BASH_USER = "Call the bash tool exactly once with command pwd."


def _effort(treatment: Mapping[str, Any]) -> float | None:
    reasoning = str(treatment["reasoning"])
    return (
        float(reasoning.removeprefix("effort="))
        if reasoning.startswith("effort=")
        else None
    )


def _load_real_messages(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = [
        dict(message)
        for message in payload["messages"]
        if message.get("role") != "exit"
    ]
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("diagnostic trajectory must contain one system and one user message")
    return messages


def _neutral_user(target_characters: int) -> str:
    lines: list[str] = []
    index = 1
    while sum(len(line) + 1 for line in lines) < target_characters:
        lines.append(
            f"Neutral context line {index:04d}: documentation text for a latency measurement."
        )
        index += 1
    return "\n".join(lines)


def _prompt_ablation_cases(
    real_messages: list[dict[str, Any]],
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    real_system = str(real_messages[0]["content"])
    real_user = str(real_messages[1]["content"])
    return [
        {
            "name": "short_workspace_control",
            "purpose": "credential, transport, renderer, and small tool control",
            "max_output_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": CONTROL_SYSTEM},
                {"role": "user", "content": CONTROL_USER},
            ],
            "tools": [WORKSPACE_TOOL_SPEC],
        },
        {
            "name": "short_bash_schema",
            "purpose": "isolate the real bash tool schema",
            "max_output_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": CONTROL_SYSTEM},
                {"role": "user", "content": SHORT_BASH_USER},
            ],
            "tools": [BASH_TOOL_SPEC],
        },
        {
            "name": "long_neutral_bash",
            "purpose": "isolate prompt length with neutral text",
            "max_output_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": CONTROL_SYSTEM},
                {"role": "user", "content": _neutral_user(len(real_user))},
            ],
            "tools": [BASH_TOOL_SPEC],
        },
        {
            "name": "real_harness_short_task",
            "purpose": "isolate the mini-swe system harness",
            "max_output_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": real_system},
                {"role": "user", "content": SHORT_BASH_USER},
            ],
            "tools": [BASH_TOOL_SPEC],
        },
        {
            "name": "short_harness_real_task",
            "purpose": "isolate the SWE-bench task and instance instructions",
            "max_output_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": CONTROL_SYSTEM},
                {"role": "user", "content": real_user},
            ],
            "tools": [BASH_TOOL_SPEC],
        },
        {
            "name": "full_real_prompt",
            "purpose": "reproduce the complete failed first-turn prompt",
            "max_output_tokens": max_output_tokens,
            "messages": real_messages,
            "tools": [BASH_TOOL_SPEC],
        },
    ]


def _output_sweep_cases(
    real_messages: list[dict[str, Any]],
    output_token_allowances: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "name": f"full_real_prompt_{allowance}_tokens",
            "purpose": "isolate requested output allowance on the complete real prompt",
            "max_output_tokens": allowance,
            "messages": real_messages,
            "tools": [BASH_TOOL_SPEC],
        }
        for allowance in output_token_allowances
    ]


def _parse_output_token_allowances(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("output-token sweep must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError("output-token sweep must contain unique values")
    if any(value < 1 or value > 8_000 for value in values):
        raise ValueError("output-token sweep values must be in [1, 8000]")
    return values


def _local_renderer(treatment: Mapping[str, Any]) -> tuple[Any, Any]:
    from tinker_cookbook import renderers, tokenizer_utils

    tokenizer = tokenizer_utils.get_tokenizer(str(treatment["model"]))
    renderer = renderers.get_renderer(
        str(treatment["renderer"]),
        tokenizer,
        model_name=str(treatment["model"]),
    )
    return tokenizer, renderer


def _session_id(adapter: TinkerSamplingAdapter) -> str | None:
    value = getattr(adapter.sampling_client, "_sampling_session_id", None)
    return str(value) if value else None


async def _execute_case(
    case: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    max_output_tokens: int,
    timeout_seconds: float,
    seed: int,
) -> dict[str, Any]:
    started = time.monotonic()
    adapter = await TinkerSamplingAdapter.from_environment(
        str(treatment["model"]),
        renderer_name=str(treatment["renderer"]),
        effort=_effort(treatment),
        force_httpx_transport=True,
    )
    request = ModelRequest(
        model=str(treatment["model"]),
        messages=(TranscriptMessage("user", "diagnostic payload"),),
        operation=Operation.CONTINUE,
        max_output_tokens=max_output_tokens,
        temperature=float(treatment["temperature"]),
        top_p=float(treatment.get("top_p", 1.0)),
        top_k=int(treatment.get("top_k", -1)),
        timeout_seconds=timeout_seconds,
        seed=seed,
        tools=tuple(case["tools"]),
    )
    try:
        response = await adapter._generate_from_messages(
            request,
            [dict(message) for message in case["messages"]],
        )
    except Exception as exc:
        return {
            "completed": False,
            "provider_failed": True,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "provider_session_id": _session_id(adapter),
        }
    return {
        "completed": True,
        "provider_failed": False,
        "exception_type": None,
        "exception_message": None,
        "elapsed_seconds": time.monotonic() - started,
        "provider_latency_ms": response.latency_ms,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "finish_reason": response.finish_reason,
        "tool_call_count": len(response.tool_calls),
        "provider_session_id": _session_id(adapter),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    pool = json.loads(args.model_pool.read_text(encoding="utf-8"))
    matches = [
        treatment
        for treatment in pool["treatments"]
        if treatment["model"] == args.model
    ]
    if len(matches) != 1:
        raise ValueError("selected model must appear exactly once in the model pool")
    treatment = matches[0]
    prices = PriceSnapshot.load(args.prices)
    price = prices.price_for(args.model)
    real_messages = _load_real_messages(args.trajectory)
    output_token_allowances = (
        [args.max_output_tokens]
        if args.mode == "prompt-ablation"
        else _parse_output_token_allowances(args.output_token_sweep)
    )
    cases = (
        _prompt_ablation_cases(real_messages, args.max_output_tokens)
        if args.mode == "prompt-ablation"
        else _output_sweep_cases(real_messages, output_token_allowances)
    )
    maximum_case_costs = [
        price.conservative_cost(
            TokenUsage(
                input_tokens=price.context_tokens - int(case["max_output_tokens"]),
                output_tokens=int(case["max_output_tokens"]),
            )
        )
        for case in cases
    ]
    maximum_total_cost = sum(maximum_case_costs, Decimal("0"))
    if maximum_total_cost > args.hard_cap_usd:
        raise PermissionError(
            f"worst-case reservation {maximum_total_cost} exceeds cap {args.hard_cap_usd}"
        )

    _, renderer = _local_renderer(treatment)
    case_records: list[dict[str, Any]] = []
    conservative_cost = Decimal("0")
    for index, (case, maximum_case_cost) in enumerate(
        zip(cases, maximum_case_costs, strict=True)
    ):
        case_max_output_tokens = int(case["max_output_tokens"])
        rendered = render_tinker_prompt(
            renderer,
            case["messages"],
            case["tools"],
            effort=_effort(treatment),
        )
        record = {
            "name": case["name"],
            "purpose": case["purpose"],
            "prompt": rendered.diagnostics(
                context_tokens=price.context_tokens,
                max_output_tokens=case_max_output_tokens,
                stop_sequences=renderer.get_stop_sequences(),
            ),
            "executed": args.execute,
        }
        if args.execute:
            print(stable_json({"starting_case": case["name"]}), flush=True)
            outcome = await _execute_case(
                case,
                treatment,
                max_output_tokens=case_max_output_tokens,
                timeout_seconds=args.request_timeout_seconds,
                seed=args.seed + index,
            )
            record["outcome"] = outcome
            usage = (
                TokenUsage(
                    input_tokens=int(outcome["input_tokens"]),
                    output_tokens=int(outcome["output_tokens"]),
                )
                if outcome["completed"]
                else None
            )
            case_cost = (
                price.conservative_cost(usage)
                if usage is not None
                else maximum_case_cost
            )
            conservative_cost += case_cost
            record["conservative_cost_usd"] = str(case_cost)
            print(
                stable_json(
                    {
                        "finished_case": case["name"],
                        "completed": outcome["completed"],
                        "elapsed_seconds": outcome["elapsed_seconds"],
                        "conservative_cost_usd": str(case_cost),
                    }
                ),
                flush=True,
            )
        case_records.append(record)

    return {
        "schema_version": "tinker-prompt-diagnosis-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "renderer": treatment["renderer"],
        "reasoning": treatment["reasoning"],
        "sdk_versions": {
            name: importlib.metadata.version(name)
            for name in ("tinker", "tinker-cookbook", "mini-swe-agent")
        },
        "transport": "httpx",
        "trajectory_source": str(args.trajectory),
        "mode": args.mode,
        "max_output_tokens": args.max_output_tokens,
        "output_token_allowances": output_token_allowances,
        "request_timeout_seconds": args.request_timeout_seconds,
        "hard_cap_usd": str(args.hard_cap_usd),
        "maximum_case_reservation_usd": str(max(maximum_case_costs)),
        "maximum_total_reservation_usd": str(maximum_total_cost),
        "executed": args.execute,
        "conservative_cost_usd": str(conservative_cost),
        "cases": case_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_coding_v2.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-25_coding_v2.json"),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "outputs/pilot_coding_v2_128_probe/trajectories/"
            "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/"
            "matplotlib__matplotlib-24570.json"
        ),
    )
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("prompt-ablation", "output-sweep"),
        default="prompt-ablation",
    )
    parser.add_argument(
        "--output-token-sweep",
        default="64,128,256,512",
        help="Comma-separated allowances used when --mode=output-sweep.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=60)
    parser.add_argument("--hard-cap-usd", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tinker_prompt_diagnosis.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_output_tokens <= 8_000:
        raise SystemExit("--max-output-tokens must be in [1, 8000]")
    if args.request_timeout_seconds <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")
    if args.hard_cap_usd <= 0:
        raise SystemExit("--hard-cap-usd must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = redact(asyncio.run(_run(args)))
    findings = scan_for_secrets(result)
    if findings:
        raise ValueError("refusing to write a diagnostic artifact containing a secret")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(result) + "\n", encoding="utf-8")
    print(stable_json(result))


if __name__ == "__main__":
    main()
