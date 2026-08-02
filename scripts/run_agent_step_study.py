#!/usr/bin/env python3
"""Resume-safe paid runner for the frozen agent-step SWE-bench study."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from budget_router.agent_step import FrozenAgentStepArtifact, FrozenAgentStepRouter
from budget_router.mini_swe_tinker import (
    TinkerAgentStepRouterMiniSweModel,
    TinkerMiniSweModel,
)
from budget_router.pricing import PriceSnapshot
from budget_router.redaction import redact, scan_for_secrets
from budget_router.serialization import read_jsonl, stable_hash
from budget_router.workspace_patch import (
    capture_and_persist_terminal_workspace_patch,
)

try:
    from scripts.run_active_router_study import validate_protocol_lock
    from scripts.run_tinker_pilot import (
        _adapter,
        _image_name,
        _load_agent_templates,
        _run_episode,
        _tool_structure,
        _write_json,
        _write_jsonl_record,
    )
except ModuleNotFoundError:
    from run_active_router_study import validate_protocol_lock  # type: ignore[no-redef]
    from run_tinker_pilot import (  # type: ignore[no-redef]
        _adapter,
        _image_name,
        _load_agent_templates,
        _run_episode,
        _tool_structure,
        _write_json,
        _write_jsonl_record,
    )


FIXED_CHEAP = "fixed:openai-gpt-oss-20b"
FIXED_STRONG = "fixed:qwen3.6-35b"
ROUTED = "router:frozen-agent-step-v1"
HELDOUT_POLICIES = (FIXED_CHEAP, FIXED_STRONG, ROUTED)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("_")


def _exposure(records: list[dict[str, Any]]) -> Decimal:
    return sum(
        (Decimal(str(row["conservative_cost_usd"])) for row in records),
        Decimal("0"),
    )


def validate_dynamic_lock(path: Path) -> dict[str, Any]:
    lock = _load(path)
    if lock.get("schema_version") != "agent-step-dynamic-lock-v1":
        raise ValueError("unsupported agent-step dynamic lock")
    claimed_hash = str(lock.get("manifest_hash", ""))
    payload = dict(lock)
    payload.pop("manifest_hash", None)
    if stable_hash(payload) != claimed_hash:
        raise ValueError("agent-step dynamic lock manifest hash mismatch")
    for relative, expected in lock["files"].items():
        target = Path(str(relative))
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"agent-step dynamic input changed: {relative}")
    from minisweagent import package_dir

    mini_config_path = package_dir / "config" / "benchmarks" / "swebench.yaml"
    if (
        _sha256(mini_config_path)
        != lock["external_runtime"]["mini_swe_agent_config_sha256"]
    ):
        raise ValueError("mini-swe-agent SWE-bench template changed")
    return lock


def _treatment_map(pool: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    result = {
        str(value["model"]): (index, dict(value))
        for index, value in enumerate(pool["treatments"])
    }
    required = {"openai/gpt-oss-20b", "Qwen/Qwen3.6-35B-A3B"}
    if not required <= set(result):
        raise ValueError("frozen agent-step models are missing from the model pool")
    return result


def _policy_order(task_id: str, seed: int) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = list(HELDOUT_POLICIES)
    rng.shuffle(values)
    return values


def _plan(
    *,
    stage: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    max_task_blocks: int | None,
    treatment_order_seed: int,
) -> list[dict[str, Any]]:
    if stage == "smoke":
        task_ids = ["astropy__astropy-7166"]
        policies = (ROUTED,)
    elif stage == "heldout":
        task_ids = [str(row["task_id"]) for row in manifest["heldout"]]
        policies = HELDOUT_POLICIES
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unsupported stage: {stage}")
    completed: set[tuple[str, str, str]] = set()
    for row in records:
        key = (
            str(row["study_stage"]),
            str(row["task_id"]),
            str(row["policy_id"]),
        )
        if key in completed:
            raise ValueError(f"duplicate agent-step episode: {key}")
        completed.add(key)
    blocks: list[dict[str, Any]] = []
    for task_id in task_ids:
        order = (
            _policy_order(task_id, treatment_order_seed)
            if stage == "heldout"
            else list(policies)
        )
        pending = [
            policy
            for policy in order
            if (stage, task_id, policy) not in completed
        ]
        if pending:
            blocks.append({"task_id": task_id, "policies": pending})
    if max_task_blocks is not None:
        blocks = blocks[:max_task_blocks]
    return blocks


def _recover_episode_shards(
    *,
    output_root: Path,
    records_path: Path,
    records: list[dict[str, Any]],
    expected_protocol_sha256: str,
    expected_artifact_hash: str,
    expected_task_manifest_hash: str,
) -> int:
    canonical = {
        (
            str(row["study_stage"]),
            str(row["task_id"]),
            str(row["policy_id"]),
        )
        for row in records
    }
    recovered = 0
    for path in sorted((output_root / "episode_shards").glob("*/*/*.json")):
        row = _load(path)
        key = (
            str(row["study_stage"]),
            str(row["task_id"]),
            str(row["policy_id"]),
        )
        if key in canonical:
            continue
        if (
            row.get("protocol_sha256") != expected_protocol_sha256
            or row.get("router_artifact_hash") != expected_artifact_hash
            or row.get("task_manifest_hash") != expected_task_manifest_hash
        ):
            raise ValueError(f"episode shard freeze mismatch: {path}")
        _write_jsonl_record(records_path, row)
        records.append(row)
        canonical.add(key)
        recovered += 1
    return recovered


def _caps(protocol: dict[str, Any]) -> dict[str, Decimal]:
    runtime = protocol["runtime"]
    return {
        FIXED_CHEAP: Decimal(str(runtime["fixed_cheap_episode_cap_usd"])),
        FIXED_STRONG: Decimal(str(runtime["fixed_strong_episode_cap_usd"])),
        ROUTED: Decimal(str(runtime["routed_episode_cap_usd"])),
    }


def _make_child(
    *,
    adapter: Any,
    treatment: dict[str, Any],
    prices: PriceSnapshot,
    seed: int,
    hard_limit_usd: Decimal,
    request_timeout_seconds: float,
    max_output_tokens: int,
) -> TinkerMiniSweModel:
    return TinkerMiniSweModel(
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


def _run_routed_episode(
    task: dict[str, Any],
    *,
    artifact: FrozenAgentStepArtifact,
    cheap_treatment: dict[str, Any],
    strong_treatment: dict[str, Any],
    cheap_treatment_index: int,
    strong_treatment_index: int,
    prices: PriceSnapshot,
    seed_base: int,
    task_ordinal: int,
    hard_limit_usd: Decimal,
    request_timeout_seconds: float,
    max_output_tokens: int,
    trajectory_path: Path,
) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment

    cheap_adapter = asyncio.run(_adapter(cheap_treatment))
    strong_adapter = asyncio.run(_adapter(strong_treatment))
    cheap = _make_child(
        adapter=cheap_adapter,
        treatment=cheap_treatment,
        prices=prices,
        seed=seed_base + cheap_treatment_index * 100 + task_ordinal,
        hard_limit_usd=hard_limit_usd,
        request_timeout_seconds=request_timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    strong = _make_child(
        adapter=strong_adapter,
        treatment=strong_treatment,
        prices=prices,
        seed=seed_base + strong_treatment_index * 100 + task_ordinal,
        hard_limit_usd=hard_limit_usd,
        request_timeout_seconds=request_timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
    routed = TinkerAgentStepRouterMiniSweModel(
        cheap,
        strong,
        router=FrozenAgentStepRouter(artifact),
        total_hard_limit_usd=hard_limit_usd,
    )
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
            routed,
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
                "agent_step_study": {
                    "policy_id": ROUTED,
                    "artifact_hash": artifact.artifact_hash,
                    "seed_base": seed_base,
                    "task_ordinal": task_ordinal,
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
    route_counts = routed.router.route_counts
    record = {
        "schema_version": "agent-step-episode-v1",
        "task_id": task_id,
        "repository": task["repo"],
        "policy_id": ROUTED,
        "policy_kind": "frozen_agent_step_classifier",
        "router_artifact_hash": artifact.artifact_hash,
        "cheap_model": cheap.config.model_name,
        "strong_model": strong.config.model_name,
        "seed_base": seed_base,
        "task_ordinal": task_ordinal,
        "hard_limit_usd": str(hard_limit_usd),
        "max_output_tokens_per_turn": max_output_tokens,
        "request_timeout_seconds": request_timeout_seconds,
        "price_snapshot": prices.snapshot_id,
        "emitted_tool_call": structure["emitted_tool_call"],
        "arguments_valid": structure["arguments_valid"],
        "tool_result_returned": structure["tool_result_returned"],
        "continued_after_tool": structure["continued_after_tool"],
        "terminal_record_valid": terminal_record_valid,
        "structurally_valid": bool(
            structure["emitted_tool_call"]
            and structure["arguments_valid"]
            and structure["tool_result_returned"]
            and terminal_record_valid
        ),
        "provider_failed": routed.provider_failures > 0,
        "error_type": error_type,
        "exit_status": exit_status,
        "submitted_patch": bool(submission),
        "submission_sha256": (
            hashlib.sha256(submission.encode()).hexdigest()
            if submission
            else None
        ),
        **workspace_patch,
        "conservative_cost_usd": str(routed.spent_usd),
        "provider_failure_reserved_usd": str(
            routed.provider_failure_reserved_usd
        ),
        "input_tokens": routed.usage.input_tokens,
        "output_tokens": routed.usage.output_tokens,
        "model_calls": routed.calls,
        "cheap_calls": cheap.calls,
        "strong_calls": strong.calls,
        "cheap_cost_usd": str(cheap.spent_usd),
        "strong_cost_usd": str(strong.spent_usd),
        "cheap_routes": route_counts[cheap.config.model_name],
        "strong_routes": route_counts[strong.config.model_name],
        "switch_count": routed.router.switch_count,
        "forced_strong_reasons": {
            reason: sum(decision.reason == reason for decision in routed.router.decisions)
            for reason in sorted(
                {
                    decision.reason
                    for decision in routed.router.decisions
                    if decision.forced
                }
            )
        },
        "latency_seconds": time.monotonic() - started,
        "trajectory": str(trajectory_path),
    }
    sanitized = redact(record)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist an agent-step record containing a secret")
    return sanitized


def _run_policy(
    task: dict[str, Any],
    *,
    policy_id: str,
    task_ordinal: int,
    artifact: FrozenAgentStepArtifact,
    treatments: dict[str, tuple[int, dict[str, Any]]],
    prices: PriceSnapshot,
    caps: dict[str, Decimal],
    seed_base: int,
    request_timeout_seconds: float,
    max_output_tokens: int,
    output_root: Path,
    stage: str,
) -> dict[str, Any]:
    trajectory_path = (
        output_root
        / "trajectories"
        / stage
        / policy_id.replace("/", "__").replace(":", "_")
        / f"{task['instance_id']}.json"
    )
    if policy_id == ROUTED:
        cheap_index, cheap_treatment = treatments[artifact.cheap_model]
        strong_index, strong_treatment = treatments[artifact.strong_model]
        return _run_routed_episode(
            task,
            artifact=artifact,
            cheap_treatment=cheap_treatment,
            strong_treatment=strong_treatment,
            cheap_treatment_index=cheap_index,
            strong_treatment_index=strong_index,
            prices=prices,
            seed_base=seed_base,
            task_ordinal=task_ordinal,
            hard_limit_usd=caps[policy_id],
            request_timeout_seconds=request_timeout_seconds,
            max_output_tokens=max_output_tokens,
            trajectory_path=trajectory_path,
        )

    model_name = (
        artifact.cheap_model if policy_id == FIXED_CHEAP else artifact.strong_model
    )
    treatment_index, treatment = treatments[model_name]
    adapter = asyncio.run(_adapter(treatment))
    record = _run_episode(
        task,
        treatment,
        adapter,
        prices,
        seed=seed_base + treatment_index * 100 + task_ordinal,
        hard_limit_usd=caps[policy_id],
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=request_timeout_seconds,
        trajectory_path=trajectory_path,
    )
    record["schema_version"] = "agent-step-episode-v1"
    record["policy_id"] = policy_id
    record["policy_kind"] = "fixed_model"
    record["router_artifact_hash"] = artifact.artifact_hash
    record["active_model"] = model_name
    record["task_ordinal"] = task_ordinal
    record["cheap_calls"] = (
        int(record["model_calls"]) if policy_id == FIXED_CHEAP else 0
    )
    record["strong_calls"] = (
        int(record["model_calls"]) if policy_id == FIXED_STRONG else 0
    )
    record["cheap_cost_usd"] = (
        record["conservative_cost_usd"] if policy_id == FIXED_CHEAP else "0"
    )
    record["strong_cost_usd"] = (
        record["conservative_cost_usd"] if policy_id == FIXED_STRONG else "0"
    )
    record["cheap_routes"] = record["cheap_calls"]
    record["strong_routes"] = record["strong_calls"]
    record["switch_count"] = 0
    record["forced_strong_reasons"] = {}
    sanitized = redact(record)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist a fixed record containing a secret")
    return sanitized


def _run_block(
    block: dict[str, Any],
    *,
    tasks_by_id: dict[str, dict[str, Any]],
    ordinal_by_task: dict[str, int],
    artifact: FrozenAgentStepArtifact,
    treatments: dict[str, tuple[int, dict[str, Any]]],
    prices: PriceSnapshot,
    caps: dict[str, Decimal],
    seed_base: int,
    request_timeout_seconds: float,
    max_output_tokens: int,
    output_root: Path,
    stage: str,
    record_context: dict[str, str],
) -> list[dict[str, Any]]:
    task_id = str(block["task_id"])
    results: list[dict[str, Any]] = []
    for policy_id in block["policies"]:
        record = _run_policy(
            tasks_by_id[task_id],
            policy_id=policy_id,
            task_ordinal=ordinal_by_task[task_id],
            artifact=artifact,
            treatments=treatments,
            prices=prices,
            caps=caps,
            seed_base=seed_base,
            request_timeout_seconds=request_timeout_seconds,
            max_output_tokens=max_output_tokens,
            output_root=output_root,
            stage=stage,
        )
        record.update(record_context)
        record["study_stage"] = stage
        shard_path = (
            output_root
            / "episode_shards"
            / stage
            / task_id
            / f"{_slug(policy_id)}.json"
        )
        _write_json(shard_path, record)
        results.append(record)
    return results


def _ensure_images(task_ids: list[str]) -> None:
    for task_id in task_ids:
        image = _image_name(task_id)
        inspected = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
        )
        if inspected.returncode:
            subprocess.run(
                ["docker", "pull", "--platform", "linux/amd64", image],
                check=True,
                timeout=1_800,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "heldout"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "artifacts/active_router_protocol_amendment_007_agent_step_routing.json"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/active_router_protocol.sha256"),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("artifacts/active_router_task_manifest.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/agent_step_router_v1/router_artifact.json"),
    )
    parser.add_argument(
        "--static-gate",
        type=Path,
        default=Path("outputs/agent_step_router_v1/static_gate.json"),
    )
    parser.add_argument(
        "--dynamic-lock",
        type=Path,
        default=Path("outputs/agent_step_router_v1/dynamic_lock.json"),
    )
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
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/agent_step_router_v1"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-task-blocks", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--approved-new-credit-usd", type=Decimal)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise SystemExit("--workers must be in [1, 3]")
    if args.max_task_blocks is not None and args.max_task_blocks <= 0:
        raise SystemExit("--max-task-blocks must be positive")

    validate_protocol_lock(Path.cwd(), args.protocol_lock)
    dynamic_lock = validate_dynamic_lock(args.dynamic_lock)
    protocol = _load(args.protocol)
    manifest = _load(args.task_manifest)
    artifact = FrozenAgentStepArtifact.from_dict(_load(args.artifact))
    gate = _load(args.static_gate)
    if (
        gate.get("dynamic_collection_authorized") is not True
        or gate.get("artifact_hash") != artifact.artifact_hash
        or dynamic_lock.get("router_artifact_hash") != artifact.artifact_hash
    ):
        raise PermissionError("the frozen static gate did not authorize collection")
    pool = _load(args.model_pool)
    prices = PriceSnapshot.load(args.prices)
    if pool["price_snapshot"] != prices.snapshot_id:
        raise ValueError("model pool and price snapshot disagree")
    treatments = _treatment_map(pool)
    caps = _caps(protocol)
    records_path = args.output_root / "dynamic_episodes.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    protocol_hash = _sha256(args.protocol)
    artifact_file_hash = _sha256(args.artifact)
    recovered_shards = _recover_episode_shards(
        output_root=args.output_root,
        records_path=records_path,
        records=records,
        expected_protocol_sha256=protocol_hash,
        expected_artifact_hash=artifact.artifact_hash,
        expected_task_manifest_hash=str(manifest["manifest_hash"]),
    )
    blocks = _plan(
        stage=args.stage,
        manifest=manifest,
        records=records,
        max_task_blocks=args.max_task_blocks,
        treatment_order_seed=int(
            protocol["dynamic_test"]["treatment_order_seed"]
        ),
    )
    pending_max = sum(
        (
            caps[policy_id]
            for block in blocks
            for policy_id in block["policies"]
        ),
        Decimal("0"),
    )
    current_new = _exposure(records)
    amendment_limit = Decimal(
        str(protocol["budget"]["maximum_new_exposure_usd"])
    )
    original_limit = Decimal(
        str(protocol["authorization"]["original_incremental_hard_ceiling_usd"])
    )
    prior = Decimal(
        str(protocol["budget"]["recorded_prior_active_router_exposure_usd"])
    )
    public_plan = {
        "stage": args.stage,
        "artifact_hash": artifact.artifact_hash,
        "dynamic_lock_hash": dynamic_lock["manifest_hash"],
        "pending_task_blocks": len(blocks),
        "pending_episodes": sum(len(block["policies"]) for block in blocks),
        "maximum_pending_cost_usd": str(pending_max),
        "current_new_exposure_usd": str(current_new),
        "maximum_new_exposure_after_plan_usd": str(current_new + pending_max),
        "maximum_cumulative_active_router_exposure_usd": str(
            prior + current_new + pending_max
        ),
        "amendment_limit_usd": str(amendment_limit),
        "original_limit_usd": str(original_limit),
        "execute": args.execute,
        "recovered_episode_shards": recovered_shards,
    }
    print(json.dumps(public_plan, indent=2, sort_keys=True), flush=True)
    if not args.execute or not blocks:
        return
    if current_new + pending_max > amendment_limit:
        raise PermissionError("pending plan exceeds the agent-step amendment cap")
    if prior + current_new + pending_max > original_limit:
        raise PermissionError("pending plan exceeds the original Tinker ceiling")
    if (
        args.approved_new_credit_usd is None
        or args.approved_new_credit_usd < pending_max
        or args.approved_new_credit_usd > amendment_limit
    ):
        raise PermissionError(
            "execution requires an approval covering the displayed pending maximum"
        )
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set in this process environment")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    tasks_payload = _load(args.tasks)
    tasks_by_id = {
        str(row["instance_id"]): dict(row)
        for row in tasks_payload["records"]
    }
    required_tasks = {str(block["task_id"]) for block in blocks}
    missing = sorted(required_tasks - set(tasks_by_id))
    if missing:
        raise ValueError(f"agent-step tasks are missing from source data: {missing}")
    all_task_ids = [
        str(row["task_id"]) for row in manifest["development"] + manifest["heldout"]
    ]
    ordinal_by_task = {
        task_id: ordinal for ordinal, task_id in enumerate(all_task_ids)
    }
    _ensure_images(sorted(required_tasks))
    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", ".minisweagent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

    record_context = {
        "study_id": str(protocol["parent_study_id"]),
        "amendment_id": str(protocol["amendment_id"]),
        "protocol_sha256": protocol_hash,
        "task_manifest_hash": str(manifest["manifest_hash"]),
        "router_artifact_file_sha256": artifact_file_hash,
        "dynamic_lock_hash": str(dynamic_lock["manifest_hash"]),
    }
    errors: list[dict[str, str]] = []
    consecutive_provider_failures = 0
    for previous in reversed(records):
        if previous.get("provider_failed"):
            consecutive_provider_failures += 1
        else:
            break
    stop_launching = consecutive_provider_failures >= 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        block_iterator = iter(blocks)
        pending: dict[
            concurrent.futures.Future[list[dict[str, Any]]],
            str,
        ] = {}

        def submit_next() -> bool:
            try:
                block = next(block_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _run_block,
                block,
                tasks_by_id=tasks_by_id,
                ordinal_by_task=ordinal_by_task,
                artifact=artifact,
                treatments=treatments,
                prices=prices,
                caps=caps,
                seed_base=202607290,
                request_timeout_seconds=args.request_timeout_seconds,
                max_output_tokens=args.max_output_tokens,
                output_root=args.output_root,
                stage=args.stage,
                record_context=record_context,
            )
            pending[future] = str(block["task_id"])
            return True

        if not stop_launching:
            for _ in range(args.workers):
                if not submit_next():
                    break
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                task_id = pending.pop(future)
                try:
                    completed_records = future.result()
                except Exception as exc:
                    errors.append(
                        {
                            "task_id": task_id,
                            "error_type": type(exc).__name__,
                        }
                    )
                    stop_launching = True
                    print(
                        json.dumps(
                            {
                                "event": "task_block_failed",
                                "task_id": task_id,
                                "error_type": type(exc).__name__,
                                "recovery": "completed episode shards retained",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                for record in completed_records:
                    _write_jsonl_record(records_path, record)
                    records.append(record)
                    if record.get("provider_failed"):
                        consecutive_provider_failures += 1
                    else:
                        consecutive_provider_failures = 0
                    if consecutive_provider_failures >= 2:
                        stop_launching = True
                    print(
                        json.dumps(
                            {
                                "event": "episode_complete",
                                "stage": args.stage,
                                "task_id": task_id,
                                "policy_id": record["policy_id"],
                                "exit_status": record["exit_status"],
                                "submitted_patch": record["submitted_patch"],
                                "cost_usd": record["conservative_cost_usd"],
                                "new_exposure_usd": str(_exposure(records)),
                                "consecutive_provider_failures": (
                                    consecutive_provider_failures
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if not stop_launching:
                    submit_next()
            if stop_launching:
                for future in list(pending):
                    if future.cancel():
                        pending.pop(future)
    if consecutive_provider_failures >= 2:
        raise RuntimeError("stopped after two consecutive provider failures")
    if errors:
        raise RuntimeError(f"agent-step task blocks failed: {errors}")


if __name__ == "__main__":
    main()
