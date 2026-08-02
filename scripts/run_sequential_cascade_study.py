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
from typing import Any

from budget_router.mini_swe_tinker import (
    TinkerCascadeMiniSweModel,
    TinkerMiniSweModel,
)
from budget_router.pricing import PriceSnapshot
from budget_router.redaction import redact, scan_for_secrets
from budget_router.serialization import read_jsonl, stable_hash
from budget_router.workspace_patch import (
    capture_and_persist_terminal_workspace_patch,
)

try:
    from scripts.run_router_baseline_study import _load_local_secrets
    from scripts.run_tinker_pilot import (
        _adapter,
        _image_name,
        _load_agent_templates,
        _tool_structure,
        _write_json,
        _write_jsonl_record,
    )
except ModuleNotFoundError:
    from run_router_baseline_study import _load_local_secrets  # type: ignore[no-redef]
    from run_tinker_pilot import (  # type: ignore[no-redef]
        _adapter,
        _image_name,
        _load_agent_templates,
        _tool_structure,
        _write_json,
        _write_jsonl_record,
    )


SCOUT_FINISHER_PROMPT = """

This episode uses a frozen cheap-first cascade. Qwen 8B is the initial scout
for at most six model calls and a $0.06 sub-cap. The scout may inspect, test,
edit, and submit if it can confidently finish. If it has not submitted, the
system will hand the same workspace and full visible transcript to Qwen 35B,
which must inspect the inherited work, finish, verify, and submit within the
remaining portion of the shared $0.90 total cap. Never transfer or request
private chain-of-thought; only visible messages, tool evidence, and workspace
artifacts cross the handoff.
""".strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(
    path: Path,
    *,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    protocol = _load_json(path)
    claimed_hash = str(protocol.get("manifest_hash", ""))
    without_hash = dict(protocol)
    without_hash.pop("manifest_hash", None)
    if stable_hash(without_hash) != claimed_hash:
        raise ValueError("cascade protocol manifest hash is invalid")
    for name, expected in protocol["inputs"].items():
        actual_path = input_paths.get(name)
        if actual_path is None or _sha256(actual_path) != expected:
            raise ValueError(f"cascade input differs from freeze: {name}")
    return protocol


def cascade_exposure(records: list[dict[str, Any]]) -> Decimal:
    return sum(
        (Decimal(str(row["conservative_cost_usd"])) for row in records),
        Decimal("0"),
    )


def plan_cascade(
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    max_tasks: int | None,
) -> dict[str, Any]:
    policy_id = str(protocol["cascade_policy"]["policy_id"])
    completed: set[str] = set()
    for row in records:
        if row.get("study_manifest_hash") != protocol["manifest_hash"]:
            raise ValueError("cascade record has a different protocol hash")
        if row.get("model") != policy_id:
            raise ValueError("cascade record has a different policy ID")
        task_id = str(row["task_id"])
        if task_id in completed:
            raise ValueError(f"duplicate cascade record: {task_id}")
        completed.add(task_id)
    task_ids = [str(task_id) for task_id in protocol["task_ids"]]
    unexpected = completed - set(task_ids)
    if unexpected:
        raise ValueError(f"cascade records contain unexpected tasks: {unexpected}")
    pending = [task_id for task_id in task_ids if task_id not in completed]
    if max_tasks is not None:
        pending = pending[:max_tasks]
    total_cap = Decimal(str(protocol["cascade_policy"]["total_hard_cap_usd"]))
    prior = Decimal(
        str(protocol["budget"]["recorded_prior_exposure_usd"])
    ) + Decimal(
        str(protocol["budget"]["abandoned_provider_hang_reserve_usd"])
    )
    current = prior + cascade_exposure(records)
    return {
        "pending_task_ids": pending,
        "pending_episode_count": len(pending),
        "maximum_pending_cost_usd": total_cap * len(pending),
        "current_total_exposure_usd": current,
        "projected_total_exposure_usd": current + total_cap * len(pending),
    }


def _run_cascade_episode(
    task: dict[str, Any],
    *,
    task_ordinal: int,
    protocol: dict[str, Any],
    scout_treatment: dict[str, Any],
    finisher_treatment: dict[str, Any],
    scout_adapter: Any,
    finisher_adapter: Any,
    prices: PriceSnapshot,
    seed_base: int,
    scout_treatment_index: int,
    finisher_treatment_index: int,
    request_timeout_seconds: float,
    trajectory_path: Path,
) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment

    policy = protocol["cascade_policy"]
    total_cap = Decimal(str(policy["total_hard_cap_usd"]))
    scout_cap = Decimal(str(policy["scout_hard_cap_usd"]))
    scout_seed = seed_base + scout_treatment_index * 100 + task_ordinal
    finisher_seed = seed_base + finisher_treatment_index * 100 + task_ordinal
    scout = TinkerMiniSweModel(
        scout_adapter,
        model_name=scout_treatment["model"],
        renderer=scout_treatment["renderer"],
        reasoning=scout_treatment["reasoning"],
        temperature=float(scout_treatment["temperature"]),
        top_p=float(scout_treatment.get("top_p", 1.0)),
        top_k=int(scout_treatment.get("top_k", -1)),
        request_timeout_seconds=request_timeout_seconds,
        seed=scout_seed,
        hard_limit_usd=scout_cap,
        price=prices.price_for(scout_treatment["model"]),
        price_snapshot=prices.snapshot_id,
        max_output_tokens=int(policy["scout_max_output_tokens_per_call"]),
    )
    finisher = TinkerMiniSweModel(
        finisher_adapter,
        model_name=finisher_treatment["model"],
        renderer=finisher_treatment["renderer"],
        reasoning=finisher_treatment["reasoning"],
        temperature=float(finisher_treatment["temperature"]),
        top_p=float(finisher_treatment.get("top_p", 1.0)),
        top_k=int(finisher_treatment.get("top_k", -1)),
        request_timeout_seconds=request_timeout_seconds,
        seed=finisher_seed,
        hard_limit_usd=total_cap - scout_cap,
        price=prices.price_for(finisher_treatment["model"]),
        price_snapshot=prices.snapshot_id,
        max_output_tokens=int(policy["finisher_max_output_tokens_per_call"]),
    )
    cascade = TinkerCascadeMiniSweModel(
        scout,
        finisher,
        total_hard_limit_usd=total_cap,
        scout_max_calls=int(policy["scout_max_calls"]),
    )

    system_template, instance_template = _load_agent_templates()
    system_template = system_template.strip() + "\n\n" + SCOUT_FINISHER_PROMPT
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
            cascade,
            environment,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=int(policy["total_step_limit"]),
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
                "cascade": {
                    "policy_id": policy["policy_id"],
                    "study_id": protocol["study_id"],
                    "study_manifest_hash": protocol["manifest_hash"],
                    "scout_seed": scout_seed,
                    "finisher_seed": finisher_seed,
                },
            }
        )
        if agent is not None
        else {
            "instance_id": task_id,
            "messages": [],
            "info": {"exit_status": exit_status, "submission": ""},
        }
    )
    _write_json(trajectory_path, trajectory)
    latency = time.monotonic() - started
    record = {
        "schema_version": "sequential-cascade-episode-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "study_stage": "cascade_test",
        "task_id": task_id,
        "task_ordinal": task_ordinal,
        "repository": task["repo"],
        "model": policy["policy_id"],
        "policy": policy["kind"],
        "scout_model": scout_treatment["model"],
        "finisher_model": finisher_treatment["model"],
        "scout_seed": scout_seed,
        "finisher_seed": finisher_seed,
        "hard_limit_usd": str(total_cap),
        "scout_hard_limit_usd": str(scout_cap),
        "scout_max_calls": int(policy["scout_max_calls"]),
        "max_output_tokens_per_turn": int(
            policy["finisher_max_output_tokens_per_call"]
        ),
        "request_timeout_seconds": request_timeout_seconds,
        "price_snapshot": prices.snapshot_id,
        "emitted_tool_call": structure["emitted_tool_call"],
        "arguments_valid": structure["arguments_valid"],
        "tool_result_returned": structure["tool_result_returned"],
        "continued_after_tool": structure["continued_after_tool"],
        "terminal_record_valid": terminal_record_valid,
        "structurally_valid": (
            structure["emitted_tool_call"]
            and structure["arguments_valid"]
            and structure["tool_result_returned"]
            and terminal_record_valid
        ),
        "provider_failed": cascade.provider_failures > 0,
        "error_type": error_type,
        "exit_status": exit_status,
        "submitted_patch": bool(submission),
        "submission_sha256": (
            hashlib.sha256(submission.encode()).hexdigest()
            if submission
            else None
        ),
        **workspace_patch,
        "conservative_cost_usd": str(cascade.spent_usd),
        "provider_failure_reserved_usd": str(
            cascade.provider_failure_reserved_usd
        ),
        "input_tokens": cascade.usage.input_tokens,
        "output_tokens": cascade.usage.output_tokens,
        "model_calls": cascade.calls,
        "scout_calls": scout.calls,
        "finisher_calls": finisher.calls,
        "scout_cost_usd": str(scout.spent_usd),
        "finisher_cost_usd": str(finisher.spent_usd),
        "handoff_count": cascade.handoff_count,
        "switch_reason": cascade.switch_reason,
        "scout_early_submission": bool(submission) and cascade.handoff_count == 0,
        "latency_seconds": latency,
        "trajectory": str(trajectory_path),
    }
    sanitized = redact(record)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist a cascade record containing a secret")
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/sequential_cascade_study_v1.json"),
    )
    parser.add_argument(
        "--source-study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--router-freeze",
        type=Path,
        default=Path("outputs/router_baseline_v2/router_freeze.json"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("outputs/router_baseline_v2/episodes.jsonl"),
    )
    parser.add_argument(
        "--provider-hangs",
        type=Path,
        default=Path("outputs/router_baseline_v2/provider_hangs.jsonl"),
    )
    parser.add_argument(
        "--comparator-report",
        type=Path,
        default=Path(
            "outputs/router_baseline_v2/swebench_grader/test/"
            "Qwen__Qwen3.6-35B-A3B.router-v2-test-qwen36-pinned.json"
        ),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_coding_v7.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-27_coding_v3.json"),
    )
    parser.add_argument(
        "--harness-prompt",
        type=Path,
        default=Path("configs/harness_prompt.txt"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/trajectories"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/sequential_cascade_v1/collection_summary.json"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--approved-incremental-credit-usd", type=Decimal)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1, 4]")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise SystemExit("--max-tasks must be positive")

    input_paths = {
        "source_study": args.source_study,
        "router_freeze": args.router_freeze,
        "baseline_records": args.baseline_records,
        "provider_hangs": args.provider_hangs,
        "comparator_report": args.comparator_report,
        "model_pool": args.model_pool,
        "prices": args.prices,
        "harness_prompt": args.harness_prompt,
        "tasks": args.tasks,
    }
    protocol = load_protocol(args.protocol, input_paths=input_paths)
    records = read_jsonl(args.records) if args.records.exists() else []
    plan = plan_cascade(protocol, records, max_tasks=args.max_tasks)
    public_plan = {
        "study_id": protocol["study_id"],
        "manifest_hash": protocol["manifest_hash"],
        "pending_episodes": plan["pending_episode_count"],
        "maximum_pending_cost_usd": str(plan["maximum_pending_cost_usd"]),
        "current_total_exposure_usd": str(plan["current_total_exposure_usd"]),
        "projected_total_exposure_usd": str(plan["projected_total_exposure_usd"]),
        "working_limit_usd": protocol["budget"]["working_limit_usd"],
        "absolute_limit_usd": protocol["budget"]["absolute_limit_usd"],
        "execute": args.execute,
    }
    print(json.dumps(public_plan, indent=2, sort_keys=True))
    if not args.execute or not plan["pending_task_ids"]:
        return

    absolute_limit = Decimal(str(protocol["budget"]["absolute_limit_usd"]))
    working_limit = Decimal(str(protocol["budget"]["working_limit_usd"]))
    if (
        args.approved_incremental_credit_usd is None
        or args.approved_incremental_credit_usd < absolute_limit
        or args.approved_incremental_credit_usd > Decimal("200")
    ):
        raise PermissionError(
            "cascade execution requires the existing explicit $200 approval"
        )
    if (
        plan["projected_total_exposure_usd"] > working_limit
        or plan["projected_total_exposure_usd"] > absolute_limit
    ):
        raise PermissionError("full pending cascade reservation exceeds frozen limits")

    _load_local_secrets()
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set")
    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", ".minisweagent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    source_study = _load_json(args.source_study)
    tasks_payload = _load_json(args.tasks)
    tasks_by_id = {
        str(row["instance_id"]): dict(row) for row in tasks_payload["records"]
    }
    ordinals = {
        str(row["task_id"]): int(row["ordinal"])
        for row in source_study["tasks"]
    }
    model_pool = _load_json(args.model_pool)
    treatments = [dict(row) for row in model_pool["treatments"]]
    treatment_by_model = {
        str(row["model"]): (index, row)
        for index, row in enumerate(treatments)
    }
    policy = protocol["cascade_policy"]
    scout_index, scout_treatment = treatment_by_model[policy["scout_model"]]
    finisher_index, finisher_treatment = treatment_by_model[
        policy["finisher_model"]
    ]
    prices = PriceSnapshot.load(args.prices)
    scout_adapter = asyncio.run(_adapter(scout_treatment))
    finisher_adapter = asyncio.run(_adapter(finisher_treatment))
    seed_base = int(source_study["episode_seed_base"])
    total_cap = Decimal(str(policy["total_hard_cap_usd"]))

    stopped_for_budget = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        for task_id in plan["pending_task_ids"]:
            current = (
                Decimal(str(protocol["budget"]["recorded_prior_exposure_usd"]))
                + Decimal(
                    str(protocol["budget"]["abandoned_provider_hang_reserve_usd"])
                )
                + cascade_exposure(records)
                + total_cap * len(futures)
            )
            if current + total_cap > working_limit or current + total_cap > absolute_limit:
                stopped_for_budget = True
                break
            trajectory_path = args.trajectory_dir / f"{task_id}.json"
            future = executor.submit(
                _run_cascade_episode,
                tasks_by_id[task_id],
                task_ordinal=ordinals[task_id],
                protocol=protocol,
                scout_treatment=scout_treatment,
                finisher_treatment=finisher_treatment,
                scout_adapter=scout_adapter,
                finisher_adapter=finisher_adapter,
                prices=prices,
                seed_base=seed_base,
                scout_treatment_index=scout_index,
                finisher_treatment_index=finisher_index,
                request_timeout_seconds=args.request_timeout_seconds,
                trajectory_path=trajectory_path,
            )
            futures[future] = task_id

        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            _write_jsonl_record(args.records, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "task_id": record["task_id"],
                        "submitted_patch": record["submitted_patch"],
                        "scout_calls": record["scout_calls"],
                        "finisher_calls": record["finisher_calls"],
                        "handoff_count": record["handoff_count"],
                        "cost_usd": record["conservative_cost_usd"],
                        "cascade_episode_count": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "schema_version": "sequential-cascade-collection-summary-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "completed_episodes": len(records),
        "submitted_episodes": sum(bool(row["submitted_patch"]) for row in records),
        "handoff_episodes": sum(int(row["handoff_count"] > 0) for row in records),
        "scout_early_submissions": sum(
            bool(row["scout_early_submission"]) for row in records
        ),
        "cascade_conservative_cost_usd": str(cascade_exposure(records)),
        "total_exposure_with_prior_and_hang_reserve_usd": str(
            Decimal(str(protocol["budget"]["recorded_prior_exposure_usd"]))
            + Decimal(
                str(protocol["budget"]["abandoned_provider_hang_reserve_usd"])
            )
            + cascade_exposure(records)
        ),
        "stopped_for_budget": stopped_for_budget,
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
