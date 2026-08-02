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

from budget_router.isolated_stage import (
    build_visible_diagnostic_handoff,
    canonicalize_infrastructure_recoveries,
    is_zero_call_infrastructure_failure,
    plan_stage_episodes,
    remaining_finisher_cap,
)
from budget_router.mini_swe_tinker import TinkerMiniSweModel
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
        _run_episode,
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
        _run_episode,
        _tool_structure,
        _write_json,
        _write_jsonl_record,
    )


SCOUT_SYSTEM_PROMPT = """
You are the disposable diagnostic scout in a frozen two-stage coding policy.
Use at most six model calls to inspect the repository, reproduce the issue,
locate relevant code, and gather concise visible evidence for another model.
Any edits and any submitted patch will be discarded with this container.
Prioritize commands, test output, file locations, and a concrete proposed fix.
Do not expose, request, or encode private chain-of-thought or credentials.
""".strip()

FINISHER_HANDOFF_PROMPT = """
This is the clean finisher phase of a frozen isolated-scout policy. You are in a
fresh checkout: none of the scout's edits exist here, and any scout submission
was intentionally discarded. Solve the original issue yourself, using the
bounded visible diagnostic evidence below only as potentially fallible context.
Inspect the clean repository, implement the correct patch, test it, and submit.
No hidden reasoning was transferred.

--- BEGIN VISIBLE SCOUT EVIDENCE ---
{handoff}
--- END VISIBLE SCOUT EVIDENCE ---
""".strip()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(
    path: Path,
    *,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    protocol = _load(path)
    claimed = str(protocol.get("manifest_hash", ""))
    unhashed = dict(protocol)
    unhashed.pop("manifest_hash", None)
    if stable_hash(unhashed) != claimed:
        raise ValueError("isolated-stage protocol manifest hash is invalid")
    for name, expected in protocol["inputs"].items():
        actual = input_paths.get(name)
        if actual is None or _sha256(actual) != expected:
            raise ValueError(f"isolated-stage input differs from freeze: {name}")
    return protocol


def study_exposure(records: list[dict[str, Any]]) -> Decimal:
    return sum(
        (Decimal(str(record["conservative_cost_usd"])) for record in records),
        Decimal("0"),
    )


def _load_freeze(
    path: Path,
    *,
    protocol: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    if not path.exists():
        raise PermissionError(f"{kind} freeze is required before this stage")
    freeze = _load(path)
    if freeze.get("study_manifest_hash") != protocol["manifest_hash"]:
        raise PermissionError(f"{kind} freeze belongs to a different study")
    if kind == "gate":
        artifact_path = Path(str(freeze.get("router_artifact_path", "")))
        if not artifact_path.is_file():
            raise PermissionError("frozen gate artifact is missing")
        if _sha256(artifact_path) != freeze.get("router_artifact_sha256"):
            raise PermissionError("frozen gate artifact content has changed")
    return freeze


def policies_for_stage(
    protocol: dict[str, Any],
    stage: str,
    *,
    candidate_freeze_path: Path,
    gate_freeze_path: Path,
) -> list[str]:
    if stage == "screen":
        return [
            str(policy["policy_id"])
            for policy in protocol["candidate_policies"]
        ]
    candidate_freeze = _load_freeze(
        candidate_freeze_path,
        protocol=protocol,
        kind="candidate",
    )
    selected = str(candidate_freeze.get("selected_policy_id", ""))
    valid = {
        str(policy["policy_id"]) for policy in protocol["candidate_policies"]
    }
    if selected not in valid:
        raise PermissionError("candidate freeze has no valid selected policy")
    if stage == "expand":
        return [selected]
    gate_freeze = _load_freeze(
        gate_freeze_path,
        protocol=protocol,
        kind="gate",
    )
    if gate_freeze.get("selected_candidate_policy_id") != selected:
        raise PermissionError("gate and candidate freezes disagree")
    return [selected, str(protocol["fixed_policy"]["policy_id"])]


def _terminal_valid(messages: list[dict[str, Any]]) -> bool:
    return bool(
        messages
        and messages[-1].get("role") == "exit"
        and messages[-1].get("extra", {}).get("exit_status")
    )


def _seed(value: int) -> int:
    return value % 2_147_483_647


def _run_isolated_episode(
    task: dict[str, Any],
    *,
    task_ordinal: int,
    stage: str,
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    scout_treatment: dict[str, Any],
    finisher_treatment: dict[str, Any],
    scout_adapter: Any,
    finisher_adapter: Any,
    prices: PriceSnapshot,
    scout_index: int,
    request_timeout_seconds: float,
    trajectory_path: Path,
) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment

    execution = protocol["execution_policy"]
    total_cap = Decimal(str(execution["total_hard_cap_usd"]))
    scout_cap = Decimal(str(execution["scout_hard_cap_usd"]))
    seed_base = int(execution["seed_base"])
    scout_seed = _seed(seed_base + (scout_index + 1) * 1_000 + task_ordinal)
    finisher_seed = _seed(seed_base + task_ordinal)
    task_id = str(task["instance_id"])
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
        max_output_tokens=int(execution["scout_max_output_tokens_per_call"]),
    )
    system_template, instance_template = _load_agent_templates()
    scout_environment = None
    scout_agent = None
    scout_error_type: str | None = None
    scout_exit_status = ""
    scout_submission = ""
    started = time.monotonic()
    try:
        scout_environment = DockerEnvironment(
            image=_image_name(task_id),
            cwd="/testbed",
            timeout=60,
            pull_timeout=600,
            interpreter=["bash", "-c"],
            run_args=["--rm", "--platform", "linux/amd64"],
        )
        scout_agent = DefaultAgent(
            scout,
            scout_environment,
            system_template=system_template.strip() + "\n\n" + SCOUT_SYSTEM_PROMPT,
            instance_template=instance_template,
            step_limit=int(execution["scout_max_calls"]),
            cost_limit=0,
            wall_time_limit_seconds=1_800,
            output_path=None,
        )
        scout_result = scout_agent.run(str(task["problem_statement"]))
        scout_exit_status = str(scout_result.get("exit_status", ""))
        scout_submission = str(scout_result.get("submission", ""))
    except Exception as exc:
        scout_error_type = type(exc).__name__
        scout_exit_status = scout_error_type
    finally:
        if scout_environment is not None:
            scout_environment.cleanup()

    scout_messages = list(scout_agent.messages) if scout_agent is not None else []
    handoff = build_visible_diagnostic_handoff(
        scout_messages,
        max_chars=int(execution["visible_handoff_max_chars"]),
    )
    finisher_cap = remaining_finisher_cap(total_cap, scout.spent_usd)
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
        hard_limit_usd=finisher_cap,
        price=prices.price_for(finisher_treatment["model"]),
        price_snapshot=prices.snapshot_id,
        max_output_tokens=int(execution["finisher_max_output_tokens_per_call"]),
    )
    finisher_environment = None
    finisher_agent = None
    finisher_error_type: str | None = None
    finisher_exit_status = ""
    submission = ""
    workspace_patch = capture_and_persist_terminal_workspace_patch(
        None, trajectory_path
    )
    try:
        finisher_environment = DockerEnvironment(
            image=_image_name(task_id),
            cwd="/testbed",
            timeout=60,
            pull_timeout=600,
            interpreter=["bash", "-c"],
            run_args=["--rm", "--platform", "linux/amd64"],
        )
        finisher_agent = DefaultAgent(
            finisher,
            finisher_environment,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=max(
                1,
                int(execution["total_step_limit"]) - scout.calls,
            ),
            cost_limit=0,
            wall_time_limit_seconds=7_200,
            output_path=None,
        )
        finisher_problem = (
            str(task["problem_statement"])
            + "\n\n"
            + FINISHER_HANDOFF_PROMPT.format(handoff=handoff)
        )
        finisher_result = finisher_agent.run(finisher_problem)
        finisher_exit_status = str(finisher_result.get("exit_status", ""))
        submission = str(finisher_result.get("submission", ""))
    except Exception as exc:
        finisher_error_type = type(exc).__name__
        finisher_exit_status = finisher_error_type
    finally:
        workspace_patch = capture_and_persist_terminal_workspace_patch(
            finisher_environment, trajectory_path
        )
        if finisher_environment is not None:
            finisher_environment.cleanup()

    finisher_messages = (
        list(finisher_agent.messages) if finisher_agent is not None else []
    )
    scout_structure = _tool_structure(scout_messages)
    finisher_structure = _tool_structure(finisher_messages)
    scout_structurally_valid = bool(
        scout_structure["emitted_tool_call"]
        and scout_structure["arguments_valid"]
        and scout_structure["tool_result_returned"]
        and _terminal_valid(scout_messages)
    )
    finisher_structurally_valid = bool(
        finisher_structure["emitted_tool_call"]
        and finisher_structure["arguments_valid"]
        and finisher_structure["tool_result_returned"]
        and _terminal_valid(finisher_messages)
    )
    total_spend = scout.spent_usd + finisher.spent_usd
    if total_spend > total_cap:
        raise RuntimeError("isolated policy exceeded its total episode cap")
    combined_trajectory = {
        "instance_id": task_id,
        "scout": (
            scout_agent.serialize(
                {
                    "instance_id": task_id,
                    "phase": "disposable_scout",
                    "submission_discarded": bool(scout_submission),
                }
            )
            if scout_agent is not None
            else {
                "messages": [],
                "info": {"exit_status": scout_exit_status},
            }
        ),
        "finisher": (
            finisher_agent.serialize(
                {
                    "instance_id": task_id,
                    "phase": "clean_finisher",
                }
            )
            if finisher_agent is not None
            else {
                "messages": [],
                "info": {"exit_status": finisher_exit_status},
            }
        ),
        "info": {
            "submission": submission,
            "exit_status": finisher_exit_status,
            "policy_id": candidate["policy_id"],
            "study_manifest_hash": protocol["manifest_hash"],
            "visible_handoff_sha256": hashlib.sha256(handoff.encode()).hexdigest(),
            "visible_handoff_chars": len(handoff),
        },
    }
    _write_json(trajectory_path, combined_trajectory)
    latency = time.monotonic() - started
    record = {
        "schema_version": "isolated-stage-episode-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "study_stage": stage,
        "task_id": task_id,
        "task_ordinal": task_ordinal,
        "repository": task["repo"],
        "model": candidate["policy_id"],
        "policy": execution["kind"],
        "scout_model": scout_treatment["model"],
        "finisher_model": finisher_treatment["model"],
        "scout_seed": scout_seed,
        "finisher_seed": finisher_seed,
        "hard_limit_usd": str(total_cap),
        "scout_hard_limit_usd": str(scout_cap),
        "scout_max_calls": int(execution["scout_max_calls"]),
        "max_output_tokens_per_turn": int(
            execution["finisher_max_output_tokens_per_call"]
        ),
        "request_timeout_seconds": request_timeout_seconds,
        "price_snapshot": prices.snapshot_id,
        "emitted_tool_call": finisher_structure["emitted_tool_call"],
        "arguments_valid": finisher_structure["arguments_valid"],
        "tool_result_returned": finisher_structure["tool_result_returned"],
        "continued_after_tool": finisher_structure["continued_after_tool"],
        "terminal_record_valid": _terminal_valid(finisher_messages),
        "scout_structurally_valid": scout_structurally_valid,
        "finisher_structurally_valid": finisher_structurally_valid,
        "structurally_valid": (
            scout_structurally_valid and finisher_structurally_valid
        ),
        "provider_failed": (
            scout.provider_failures + finisher.provider_failures > 0
        ),
        "scout_error_type": scout_error_type,
        "finisher_error_type": finisher_error_type,
        "error_type": finisher_error_type or scout_error_type,
        "scout_exit_status": scout_exit_status,
        "exit_status": finisher_exit_status,
        "scout_submitted_patch_discarded": bool(scout_submission),
        "submitted_patch": bool(submission),
        "submission_sha256": (
            hashlib.sha256(submission.encode()).hexdigest()
            if submission
            else None
        ),
        **workspace_patch,
        "conservative_cost_usd": str(total_spend),
        "provider_failure_reserved_usd": str(
            scout.provider_failure_reserved_usd
            + finisher.provider_failure_reserved_usd
        ),
        "input_tokens": scout.usage.input_tokens + finisher.usage.input_tokens,
        "output_tokens": scout.usage.output_tokens + finisher.usage.output_tokens,
        "model_calls": scout.calls + finisher.calls,
        "scout_calls": scout.calls,
        "finisher_calls": finisher.calls,
        "scout_cost_usd": str(scout.spent_usd),
        "finisher_cost_usd": str(finisher.spent_usd),
        "visible_handoff_chars": len(handoff),
        "visible_handoff_sha256": hashlib.sha256(handoff.encode()).hexdigest(),
        "scout_workspace_discarded": True,
        "finisher_clean_workspace": True,
        "latency_seconds": latency,
        "trajectory": str(trajectory_path),
    }
    sanitized = redact(record)
    if scan_for_secrets(sanitized):
        raise ValueError("refusing to persist an isolated-stage record containing a secret")
    return sanitized


def _run_fixed_episode(
    task: dict[str, Any],
    *,
    task_ordinal: int,
    stage: str,
    protocol: dict[str, Any],
    treatment: dict[str, Any],
    adapter: Any,
    prices: PriceSnapshot,
    request_timeout_seconds: float,
    trajectory_path: Path,
) -> dict[str, Any]:
    execution = protocol["execution_policy"]
    seed = _seed(int(execution["seed_base"]) + task_ordinal)
    record = _run_episode(
        task,
        treatment,
        adapter,
        prices,
        seed=seed,
        hard_limit_usd=Decimal(str(execution["total_hard_cap_usd"])),
        max_output_tokens=int(execution["finisher_max_output_tokens_per_call"]),
        request_timeout_seconds=request_timeout_seconds,
        trajectory_path=trajectory_path,
    )
    record.update(
        {
            "schema_version": "isolated-stage-fixed-episode-v1",
            "study_id": protocol["study_id"],
            "study_manifest_hash": protocol["manifest_hash"],
            "study_stage": stage,
            "task_ordinal": task_ordinal,
            "model": protocol["fixed_policy"]["policy_id"],
            "policy": "fixed-strong-model-v1",
            "underlying_model": treatment["model"],
            "finisher_seed": seed,
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/isolated_stage_router_study_v1.json"),
    )
    parser.add_argument(
        "--source-study",
        type=Path,
        default=Path("artifacts/router_baseline_study_v2.json"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/swebench_verified_tasks.json"),
    )
    parser.add_argument(
        "--model-pool",
        type=Path,
        default=Path("configs/model_pool_stage_v1.json"),
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("configs/tinker_prices_2026-07-28_stage_v1.json"),
    )
    parser.add_argument(
        "--train-calibration-grades",
        type=Path,
        default=Path("outputs/router_baseline_v2/train_calibration_grades.json"),
    )
    parser.add_argument(
        "--pilot-grades",
        type=Path,
        default=Path("artifacts/pilot_coding_v6_swebench_grades.json"),
    )
    parser.add_argument(
        "--harness-prompt",
        type=Path,
        default=Path("configs/harness_prompt.txt"),
    )
    parser.add_argument(
        "--candidate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/candidate_freeze.json"),
    )
    parser.add_argument(
        "--gate-freeze",
        type=Path,
        default=Path("outputs/isolated_stage_v1/gate_freeze.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes.jsonl"),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("outputs/isolated_stage_v1/trajectories"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/isolated_stage_v1/collection_summary.json"),
    )
    parser.add_argument(
        "--recovery-records",
        type=Path,
        default=Path(
            "outputs/isolated_stage_v1/infrastructure_recoveries.jsonl"
        ),
    )
    parser.add_argument(
        "--canonical-records",
        type=Path,
        default=Path("outputs/isolated_stage_v1/episodes_canonical.jsonl"),
    )
    parser.add_argument("--stage", choices=("screen", "expand", "test"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-task-blocks", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--approved-incremental-credit-usd", type=Decimal)
    parser.add_argument(
        "--retry-zero-call-infrastructure-failures",
        action="store_true",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1, 4]")
    if args.max_task_blocks is not None and args.max_task_blocks <= 0:
        raise SystemExit("--max-task-blocks must be positive")

    input_paths = {
        "source_study": args.source_study,
        "tasks": args.tasks,
        "model_pool": args.model_pool,
        "prices": args.prices,
        "train_calibration_grades": args.train_calibration_grades,
        "pilot_grades": args.pilot_grades,
        "harness_prompt": args.harness_prompt,
    }
    protocol = load_protocol(args.protocol, input_paths=input_paths)
    policies = policies_for_stage(
        protocol,
        args.stage,
        candidate_freeze_path=args.candidate_freeze,
        gate_freeze_path=args.gate_freeze,
    )
    records = read_jsonl(args.records) if args.records.exists() else []
    recovery_records = (
        read_jsonl(args.recovery_records)
        if args.recovery_records.exists()
        else []
    )
    frozen_task_ids = [
        str(task_id) for task_id in protocol["stages"][args.stage]["task_ids"]
    ]
    planning_records = records
    output_records_path = args.records
    if args.retry_zero_call_infrastructure_failures:
        if args.stage != "expand":
            raise PermissionError(
                "the documented infrastructure recovery is limited to expansion"
            )
        eligible_keys = {
            (str(record["task_id"]), str(record["model"]))
            for record in records
            if record.get("study_stage") == args.stage
            and is_zero_call_infrastructure_failure(record)
        }
        if not eligible_keys:
            raise PermissionError("there are no zero-call infrastructure failures")
        frozen_task_ids = [
            task_id
            for task_id in frozen_task_ids
            if any(key[0] == task_id for key in eligible_keys)
        ]
        for recovery in recovery_records:
            key = (str(recovery["task_id"]), str(recovery["model"]))
            if key not in eligible_keys:
                raise ValueError(
                    f"recovery record is not retry-eligible: {key}"
                )
        planning_records = recovery_records
        output_records_path = args.recovery_records
    hard_cap = Decimal(str(protocol["execution_policy"]["total_hard_cap_usd"]))
    plan = plan_stage_episodes(
        task_ids=frozen_task_ids,
        policy_ids=policies,
        records=planning_records,
        study_manifest_hash=protocol["manifest_hash"],
        stage=args.stage,
        hard_cap_usd=hard_cap,
        max_task_blocks=args.max_task_blocks,
    )
    current_exposure = study_exposure(records) + study_exposure(recovery_records)
    absolute_limit = Decimal(str(protocol["budget"]["runner_absolute_limit_usd"]))
    public_plan = {
        "study_id": protocol["study_id"],
        "manifest_hash": protocol["manifest_hash"],
        "stage": args.stage,
        "infrastructure_recovery": bool(
            args.retry_zero_call_infrastructure_failures
        ),
        "policies": policies,
        "pending_task_blocks": len(plan["pending_blocks"]),
        "pending_episodes": plan["pending_episode_count"],
        "maximum_pending_cost_usd": str(plan["maximum_pending_cost_usd"]),
        "current_incremental_exposure_usd": str(current_exposure),
        "projected_incremental_exposure_usd": str(
            current_exposure + plan["maximum_pending_cost_usd"]
        ),
        "absolute_limit_usd": str(absolute_limit),
        "execute": args.execute,
    }
    print(json.dumps(public_plan, indent=2, sort_keys=True))
    if not args.execute or not plan["pending_blocks"]:
        return
    if (
        args.approved_incremental_credit_usd is None
        or args.approved_incremental_credit_usd < absolute_limit
        or args.approved_incremental_credit_usd > Decimal("116")
    ):
        raise PermissionError(
            "execution requires explicit approval in [$112.50, $116.00]"
        )
    if current_exposure + plan["maximum_pending_cost_usd"] > absolute_limit:
        raise PermissionError("pending reservation exceeds the frozen study limit")

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

    tasks_payload = _load(args.tasks)
    tasks_by_id = {
        str(task["instance_id"]): dict(task)
        for task in tasks_payload["records"]
    }
    ordinals = {
        str(row["task_id"]): int(row["dataset_ordinal"])
        for row in protocol["tasks"]
    }
    model_pool = _load(args.model_pool)
    treatment_by_model = {
        str(row["model"]): dict(row)
        for row in [*model_pool["scouts"], model_pool["finisher"]]
    }
    policy_by_id = {
        str(row["policy_id"]): dict(row)
        for row in protocol["candidate_policies"]
    }
    scout_indices = {
        str(row["model"]): index
        for index, row in enumerate(model_pool["scouts"])
    }
    needed_models = {str(model_pool["finisher"]["model"])}
    needed_models.update(
        policy_by_id[policy_id]["scout_model"]
        for policy_id in policies
        if policy_id in policy_by_id
    )
    adapters = {
        model: asyncio.run(_adapter(treatment_by_model[model]))
        for model in sorted(needed_models)
    }
    prices = PriceSnapshot.load(args.prices)
    if prices.snapshot_id != protocol["price_snapshot"]:
        raise ValueError("price snapshot differs from the frozen study")

    stopped_for_budget = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for block in plan["pending_blocks"]:
            block_reservation = hard_cap * len(block["policy_ids"])
            accounting_records = [*records, *recovery_records]
            if study_exposure(accounting_records) + block_reservation > absolute_limit:
                stopped_for_budget = True
                break
            task_id = str(block["task_id"])
            futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
            for policy_id in block["policy_ids"]:
                trajectory_path = (
                    args.trajectory_dir
                    / (
                        f"{args.stage}_infrastructure_recovery"
                        if args.retry_zero_call_infrastructure_failures
                        else args.stage
                    )
                    / hashlib.sha256(policy_id.encode()).hexdigest()[:12]
                    / f"{task_id}.json"
                )
                if policy_id in policy_by_id:
                    candidate = policy_by_id[policy_id]
                    scout_model = str(candidate["scout_model"])
                    future = executor.submit(
                        _run_isolated_episode,
                        tasks_by_id[task_id],
                        task_ordinal=ordinals[task_id],
                        stage=args.stage,
                        protocol=protocol,
                        candidate=candidate,
                        scout_treatment=treatment_by_model[scout_model],
                        finisher_treatment=dict(model_pool["finisher"]),
                        scout_adapter=adapters[scout_model],
                        finisher_adapter=adapters[model_pool["finisher"]["model"]],
                        prices=prices,
                        scout_index=scout_indices[scout_model],
                        request_timeout_seconds=args.request_timeout_seconds,
                        trajectory_path=trajectory_path,
                    )
                else:
                    future = executor.submit(
                        _run_fixed_episode,
                        tasks_by_id[task_id],
                        task_ordinal=ordinals[task_id],
                        stage=args.stage,
                        protocol=protocol,
                        treatment=dict(model_pool["finisher"]),
                        adapter=adapters[model_pool["finisher"]["model"]],
                        prices=prices,
                        request_timeout_seconds=args.request_timeout_seconds,
                        trajectory_path=trajectory_path,
                    )
                futures[future] = policy_id
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                _write_jsonl_record(output_records_path, record)
                if args.retry_zero_call_infrastructure_failures:
                    recovery_records.append(record)
                else:
                    records.append(record)
                canonical_records = canonicalize_infrastructure_recoveries(
                    records,
                    recovery_records,
                )
                args.canonical_records.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                args.canonical_records.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in canonical_records
                    ),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "stage": args.stage,
                            "task_id": record["task_id"],
                            "policy": record["model"],
                            "submitted_patch": record["submitted_patch"],
                            "cost_usd": record["conservative_cost_usd"],
                            "total_episode_count": len(canonical_records),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    canonical_records = canonicalize_infrastructure_recoveries(
        records,
        recovery_records,
    )
    stage_records = [
        record
        for record in canonical_records
        if record["study_stage"] == args.stage
    ]
    summary = {
        "schema_version": "isolated-stage-collection-summary-v1",
        "study_id": protocol["study_id"],
        "study_manifest_hash": protocol["manifest_hash"],
        "stage": args.stage,
        "completed_stage_episodes": len(stage_records),
        "submitted_stage_episodes": sum(
            bool(record["submitted_patch"]) for record in stage_records
        ),
        "structurally_valid_stage_episodes": sum(
            bool(record["structurally_valid"]) for record in stage_records
        ),
        "provider_failed_stage_episodes": sum(
            bool(record["provider_failed"]) for record in stage_records
        ),
        "stage_conservative_cost_usd": str(study_exposure(stage_records)),
        "total_incremental_exposure_usd": str(
            study_exposure(records) + study_exposure(recovery_records)
        ),
        "infrastructure_recovery_attempt_count": len(recovery_records),
        "stopped_for_budget": stopped_for_budget,
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
