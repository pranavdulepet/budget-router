#!/usr/bin/env python3
"""Run the frozen 12-task-per-model mini-swe/Tinker compatibility pilot."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import yaml

from budget_router.experiments import project_matrix_cost
from budget_router.mini_swe_tinker import TinkerMiniSweModel
from budget_router.pilot import ToolProtocolEpisode, summarize_tool_protocol_pilot
from budget_router.pricing import PriceSnapshot
from budget_router.providers.tinker import TinkerSamplingAdapter
from budget_router.redaction import redact, scan_for_secrets
from budget_router.serialization import read_jsonl, stable_json
from budget_router.workspace_patch import (
    capture_and_persist_terminal_workspace_patch,
)


def _write_jsonl_record(path: Path, value: Mapping[str, Any]) -> None:
    sanitized = redact(value)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist a pilot record containing a secret")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stable_json(sanitized) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    sanitized = redact(value)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist pilot output containing a secret")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(sanitized) + "\n", encoding="utf-8")


def _load_tasks(
    tasks_path: Path,
    pilot_path: Path,
    *,
    expected_task_count: int = 12,
) -> list[dict[str, Any]]:
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    pilot_payload = json.loads(pilot_path.read_text(encoding="utf-8"))
    by_id = {
        str(row["instance_id"]): dict(row)
        for row in tasks_payload["records"]
    }
    selected = [by_id[str(row["task_id"])] for row in pilot_payload["tasks"]]
    if (
        len(selected) != expected_task_count
        or len({row["instance_id"] for row in selected}) != expected_task_count
    ):
        raise ValueError(
            "pilot manifest must resolve to exactly "
            f"{expected_task_count} unique tasks"
        )
    return selected


def _load_agent_templates() -> tuple[str, str]:
    from minisweagent import package_dir

    built_in = yaml.safe_load(
        (package_dir / "config" / "benchmarks" / "swebench.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy = Path("configs/harness_prompt.txt").read_text(encoding="utf-8")
    system = policy.strip() + "\n\n" + built_in["agent"]["system_template"].strip()
    return system, str(built_in["agent"]["instance_template"])


def _select_treatments(
    treatments: list[dict[str, Any]],
    selected_models: list[str] | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Select treatments while retaining their frozen full-pool seed indices."""
    if not treatments:
        raise ValueError("the pilot requires at least one treatment")
    names = [str(treatment["model"]) for treatment in treatments]
    if len(names) != len(set(names)):
        raise ValueError("the reference pilot requires unique model treatments")
    if not selected_models:
        return list(enumerate(treatments))
    if len(selected_models) != len(set(selected_models)):
        raise ValueError("--model may select each treatment at most once")
    unknown = sorted(set(selected_models) - set(names))
    if unknown:
        raise ValueError(
            "selected model is not in the frozen pool: " + ", ".join(unknown)
        )
    requested = set(selected_models)
    return [
        (index, treatment)
        for index, treatment in enumerate(treatments)
        if str(treatment["model"]) in requested
    ]


def _image_name(task_id: str) -> str:
    from minisweagent.run.benchmarks.swebench import (
        get_swebench_docker_image_name,
    )

    return get_swebench_docker_image_name({"instance_id": task_id})


def _tool_structure(messages: list[dict[str, Any]]) -> dict[str, bool]:
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    tool_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    valid_arguments = True
    calls = 0
    for message in messages:
        for call in message.get("tool_calls", ()):
            calls += 1
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (KeyError, TypeError, json.JSONDecodeError):
                valid_arguments = False
                continue
            valid_arguments &= (
                call["function"].get("name") == "bash"
                and isinstance(arguments, dict)
                and isinstance(arguments.get("command"), str)
            )
    continued = any(
        assistant_index > tool_index
        for assistant_index in assistant_indices
        for tool_index in tool_indices
    )
    return {
        "emitted_tool_call": calls > 0,
        "arguments_valid": calls > 0 and valid_arguments,
        "tool_result_returned": bool(tool_indices),
        "continued_after_tool": continued,
    }


async def _adapter(treatment: dict[str, Any]) -> TinkerSamplingAdapter:
    reasoning = str(treatment["reasoning"])
    effort = (
        float(reasoning.removeprefix("effort="))
        if reasoning.startswith("effort=")
        else None
    )
    return await TinkerSamplingAdapter.from_environment(
        treatment["model"],
        renderer_name=treatment["renderer"],
        effort=effort,
        force_httpx_transport=True,
    )


def _run_episode(
    task: dict[str, Any],
    treatment: dict[str, Any],
    adapter: TinkerSamplingAdapter,
    prices: PriceSnapshot,
    *,
    seed: int,
    hard_limit_usd: Decimal,
    max_output_tokens: int,
    request_timeout_seconds: float,
    trajectory_path: Path,
) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment

    system_template, instance_template = _load_agent_templates()
    task_id = str(task["instance_id"])
    environment = None
    agent = None
    started = time.monotonic()
    error_type: str | None = None
    exit_status = ""
    submission = ""
    workspace_patch = capture_and_persist_terminal_workspace_patch(
        None, trajectory_path
    )
    model = TinkerMiniSweModel(
        adapter,
        model_name=treatment["model"],
        renderer=treatment["renderer"],
        reasoning=treatment["reasoning"],
        temperature=float(treatment["temperature"]),
        top_p=float(treatment.get("top_p", 1.0)),
        top_k=int(treatment.get("top_k", -1)),
        request_timeout_seconds=request_timeout_seconds,
        seed=seed,
        hard_limit_usd=hard_limit_usd,
        price=prices.price_for(treatment["model"]),
        price_snapshot=prices.snapshot_id,
        max_output_tokens=max_output_tokens,
    )
    try:
        environment = DockerEnvironment(
            image=_image_name(task_id),
            cwd="/testbed",
            timeout=60,
            pull_timeout=600,
            interpreter=["bash", "-c"],
            run_args=["--rm", "--platform", "linux/amd64"],
        )
        agent = DefaultAgent(
            model,
            environment,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=75,
            cost_limit=0,
            wall_time_limit_seconds=7_200,
            output_path=None,
        )
        result = agent.run(str(task["problem_statement"]))
        exit_status = str(result.get("exit_status", ""))
        submission = str(result.get("submission", ""))
    except Exception as exc:
        error_type = type(exc).__name__
        exit_status = error_type
    finally:
        workspace_patch = capture_and_persist_terminal_workspace_patch(
            environment, trajectory_path
        )
        if environment is not None:
            environment.cleanup()

    messages = list(agent.messages) if agent is not None else []
    structure = _tool_structure(messages)
    terminal_record_valid = bool(
        messages
        and messages[-1].get("role") == "exit"
        and messages[-1].get("extra", {}).get("exit_status")
    )
    trajectory = (
        agent.serialize(
            {
                "instance_id": task_id,
                "pilot": {
                    "seed": seed,
                    "hard_limit_usd": str(hard_limit_usd),
                    "price_snapshot": prices.snapshot_id,
                },
            }
        )
        if agent is not None
        else {
            "instance_id": task_id,
            "messages": [],
            "info": {"exit_status": exit_status},
        }
    )
    _write_json(trajectory_path, trajectory)
    latency = time.monotonic() - started
    provider_failed = model.provider_failures > 0
    episode = ToolProtocolEpisode(
        model=treatment["model"],
        task_id=task_id,
        emitted_tool_call=structure["emitted_tool_call"],
        arguments_valid=structure["arguments_valid"],
        tool_result_returned=structure["tool_result_returned"],
        continued_after_tool=structure["continued_after_tool"],
        terminal_record_valid=terminal_record_valid,
        cost_usd=model.spent_usd,
        latency_seconds=latency,
        provider_failed=provider_failed,
    )
    return {
        "schema_version": "tinker-pilot-episode-v1",
        "task_id": task_id,
        "repository": task["repo"],
        "model": treatment["model"],
        "renderer": treatment["renderer"],
        "reasoning": treatment["reasoning"],
        "temperature": treatment["temperature"],
        "top_p": treatment.get("top_p", 1.0),
        "top_k": treatment.get("top_k", -1),
        "seed": seed,
        "hard_limit_usd": str(hard_limit_usd),
        "max_output_tokens_per_turn": max_output_tokens,
        "request_timeout_seconds": request_timeout_seconds,
        "price_snapshot": prices.snapshot_id,
        "emitted_tool_call": episode.emitted_tool_call,
        "arguments_valid": episode.arguments_valid,
        "tool_result_returned": episode.tool_result_returned,
        "continued_after_tool": episode.continued_after_tool,
        "terminal_record_valid": episode.terminal_record_valid,
        "structurally_valid": episode.structurally_valid,
        "provider_failed": provider_failed,
        "error_type": error_type,
        "exit_status": exit_status,
        "submitted_patch": bool(submission),
        "submission_sha256": (
            hashlib.sha256(submission.encode()).hexdigest() if submission else None
        ),
        **workspace_patch,
        "conservative_cost_usd": str(model.spent_usd),
        "provider_failure_reserved_usd": str(
            model.provider_failure_reserved_usd
        ),
        "input_tokens": model.usage.input_tokens,
        "output_tokens": model.usage.output_tokens,
        "model_calls": model.calls,
        "latency_seconds": latency,
        "trajectory": str(trajectory_path),
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    pool_version: str,
    price_snapshot: str,
    models_in_scope: list[str],
    expected_tasks_per_model: int = 12,
) -> dict[str, Any]:
    episodes = [
        ToolProtocolEpisode(
            model=row["model"],
            task_id=row["task_id"],
            emitted_tool_call=bool(row["emitted_tool_call"]),
            arguments_valid=bool(row["arguments_valid"]),
            tool_result_returned=bool(row["tool_result_returned"]),
            continued_after_tool=bool(row["continued_after_tool"]),
            terminal_record_valid=bool(row["terminal_record_valid"]),
            cost_usd=Decimal(str(row["conservative_cost_usd"])),
            latency_seconds=float(row["latency_seconds"]),
            provider_failed=bool(row["provider_failed"]),
        )
        for row in records
    ]
    model_results = summarize_tool_protocol_pilot(
        episodes,
        expected_tasks_per_model=expected_tasks_per_model,
    )
    projected = project_matrix_cost(model_results)
    return {
        "schema_version": "tinker-pilot-summary-v1",
        "pool_version": pool_version,
        "price_snapshot": price_snapshot,
        "results": [
            {
                "model": result.model,
                "episodes": result.episodes,
                "structurally_valid": result.structurally_valid,
                "projected_mean_cost_usd": str(result.projected_mean_cost_usd),
                "throughput_tasks_per_hour": result.throughput_tasks_per_hour,
                "failure_rate": result.failure_rate,
                "passes": result.passes,
            }
            for result in model_results
        ],
        "projected_matrix_cost_usd": str(projected),
        "required_credit_usd": str(projected),
        "structural_gate_passed": all(result.passes for result in model_results),
        "pilot_scope_models": models_in_scope,
        "matrix_shape": f"500×{len(models_in_scope)}×3",
        "reference_matrix_shape": "500×5×3",
        "credit_approved": False,
        "paid_matrix_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=Path("artifacts/pilot_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-25.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/pilot/episodes.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/pilot/results.json"),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("outputs/pilot/trajectories"),
    )
    parser.add_argument("--episode-hard-cap-usd", type=Decimal, default=Decimal("3"))
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--expected-task-count", type=int, default=12)
    parser.add_argument(
        "--model",
        action="append",
        dest="selected_models",
        help=(
            "run only this frozen treatment; repeat to select several. "
            "Omit to run every configured treatment."
        ),
    )
    parser.add_argument("--approved-pilot-credit-usd", type=Decimal)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_output_tokens <= 8_000:
        raise SystemExit("--max-output-tokens must be in [1, 8000]")
    if args.request_timeout_seconds <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")
    if not 1 <= args.workers <= 5:
        raise SystemExit("--workers must be in [1, 5]")
    if args.max_new_runs is not None and args.max_new_runs <= 0:
        raise SystemExit("--max-new-runs must be positive when provided")
    if args.expected_task_count <= 0:
        raise SystemExit("--expected-task-count must be positive")

    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", ".minisweagent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    tasks = _load_tasks(
        args.tasks,
        args.pilot_manifest,
        expected_task_count=args.expected_task_count,
    )
    pool = json.loads(args.model_pool.read_text(encoding="utf-8"))
    try:
        selected = _select_treatments(
            list(pool["treatments"]),
            args.selected_models,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    treatments = [treatment for _, treatment in selected]
    treatment_indices = [index for index, _ in selected]
    selected_model_names = [str(treatment["model"]) for treatment in treatments]
    run_count = len(tasks) * len(treatments)
    maximum_credit = args.episode_hard_cap_usd * run_count
    existing = read_jsonl(args.records) if args.records.exists() else []
    existing_keys = [(row["task_id"], row["model"]) for row in existing]
    if len(existing_keys) != len(set(existing_keys)):
        raise RuntimeError("pilot records contain duplicate task/model keys")
    expected_task_ids = {str(task["instance_id"]) for task in tasks}
    expected_models = set(selected_model_names)
    unexpected_keys = sorted(
        (str(task_id), str(model))
        for task_id, model in existing_keys
        if str(task_id) not in expected_task_ids or str(model) not in expected_models
    )
    if unexpected_keys:
        raise RuntimeError(
            "pilot records contain task/model keys outside the selected scope: "
            f"{unexpected_keys}"
        )
    completed = set(existing_keys)
    remaining_runs = run_count - len(existing)
    if remaining_runs < 0:
        raise RuntimeError(f"pilot has {len(existing)} records; expected at most {run_count}")
    authorized_new_runs = (
        min(remaining_runs, args.max_new_runs)
        if args.max_new_runs is not None
        else remaining_runs
    )
    maximum_additional_credit = args.episode_hard_cap_usd * authorized_new_runs
    plan = {
        "pool_version": pool["pool_version"],
        "price_snapshot": pool["price_snapshot"],
        "runs": run_count,
        "tasks": len(tasks),
        "models": len(treatments),
        "selected_models": selected_model_names,
        "episode_hard_cap_usd": str(args.episode_hard_cap_usd),
        "maximum_pilot_credit_usd": str(maximum_credit),
        "completed_runs": len(existing),
        "remaining_runs": remaining_runs,
        "maximum_additional_credit_usd": str(maximum_additional_credit),
        "authorized_new_runs": authorized_new_runs,
        "max_output_tokens_per_turn": args.max_output_tokens,
        "request_timeout_seconds": args.request_timeout_seconds,
        "workers": args.workers,
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return
    if (
        args.approved_pilot_credit_usd is None
        or args.approved_pilot_credit_usd < maximum_additional_credit
    ):
        raise PermissionError(
            "pilot requires explicit approval for up to "
            f"{maximum_additional_credit} additional USD"
        )
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    prices = PriceSnapshot.load(args.prices)
    records = list(existing)
    adapters = [asyncio.run(_adapter(treatment)) for treatment in treatments]
    launched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for task_index, task in enumerate(tasks):
            pending: dict[
                concurrent.futures.Future[dict[str, Any]],
                tuple[int, dict[str, Any]],
            ] = {}
            for model_index, treatment, adapter in zip(
                treatment_indices,
                treatments,
                adapters,
                strict=True,
            ):
                if (
                    args.max_new_runs is not None
                    and launched >= args.max_new_runs
                ):
                    break
                key = (task["instance_id"], treatment["model"])
                if key in completed:
                    continue
                trajectory_path = (
                    args.trajectory_dir
                    / treatment["model"].replace("/", "__").replace(":", "_")
                    / f"{task['instance_id']}.json"
                )
                future = executor.submit(
                    _run_episode,
                    task,
                    treatment,
                    adapter,
                    prices,
                    seed=args.seed + model_index * 100 + task_index,
                    hard_limit_usd=args.episode_hard_cap_usd,
                    max_output_tokens=args.max_output_tokens,
                    request_timeout_seconds=args.request_timeout_seconds,
                    trajectory_path=trajectory_path,
                )
                pending[future] = (model_index, treatment)
                launched += 1
            for future in concurrent.futures.as_completed(pending):
                record = future.result()
                _write_jsonl_record(args.records, record)
                records.append(record)
                completed.add((record["task_id"], record["model"]))
                print(
                    json.dumps(
                        {
                            "completed": len(records),
                            "runs": run_count,
                            "task_id": record["task_id"],
                            "model": record["model"],
                            "structurally_valid": record["structurally_valid"],
                            "cost_usd": record["conservative_cost_usd"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.max_new_runs is not None and launched >= args.max_new_runs:
                break
    if len(records) != run_count and args.max_new_runs is not None:
        print(
            json.dumps(
                {
                    "partial_run_complete": True,
                    "completed_runs": len(records),
                    "remaining_runs": run_count - len(records),
                },
                sort_keys=True,
            )
        )
        return
    if len(records) != run_count:
        raise RuntimeError(f"pilot has {len(records)} records; expected {run_count}")
    summary = _summary(
        records,
        pool_version=str(pool["pool_version"]),
        price_snapshot=prices.snapshot_id,
        models_in_scope=selected_model_names,
        expected_tasks_per_model=args.expected_task_count,
    )
    _write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
