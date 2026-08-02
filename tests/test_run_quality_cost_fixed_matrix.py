import hashlib
import json
from pathlib import Path

import pytest

from budget_router.serialization import stable_hash
from scripts.run_quality_cost_fixed_matrix import (
    plan_blocks,
    validate_completed_task_block,
    validate_execution_lock,
)


def test_plan_blocks_is_resume_safe_and_task_blocked() -> None:
    records = [
        {
            "study_stage": "development",
            "task_id": "task-a",
            "model_role": "cheap",
        }
    ]
    blocks = plan_blocks(
        task_ids=["task-a", "task-b"],
        roles=["cheap", "medium", "strong"],
        records=records,
        stage="development",
        seed=7,
        max_task_blocks=None,
    )

    assert [block["task_id"] for block in blocks] == ["task-a", "task-b"]
    assert "cheap" not in blocks[0]["roles"]
    assert set(blocks[0]["roles"]) == {"medium", "strong"}
    assert set(blocks[1]["roles"]) == {"cheap", "medium", "strong"}


def test_execution_lock_validates_internal_and_file_hashes(
    tmp_path: Path,
) -> None:
    from importlib.metadata import version

    from minisweagent import package_dir

    target = tmp_path / "target.txt"
    target.write_text("locked", encoding="utf-8")
    config = package_dir / "config" / "benchmarks" / "swebench.yaml"
    payload = {
        "files": {
            "target.txt": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "external": {
            "mini_swe_agent_version": version("mini-swe-agent"),
            "mini_swe_agent_swebench_yaml_sha256": hashlib.sha256(
                config.read_bytes()
            ).hexdigest(),
        },
    }
    payload["execution_lock_hash"] = stable_hash(payload)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(payload), encoding="utf-8")
    validate_execution_lock(tmp_path, lock)
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        validate_execution_lock(tmp_path, lock)


def test_completed_task_block_requires_every_valid_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    trajectory = Path("trajectory.json")
    trajectory.write_text("{}", encoding="utf-8")
    rows = [
        {
            "study_stage": "development",
            "task_id": "task",
            "model_role": role,
            "terminal_record_valid": True,
            "provider_failed": False,
            "trajectory": str(trajectory),
        }
        for role in ("cheap", "medium", "strong")
    ]
    validate_completed_task_block(
        rows,
        stage="development",
        task_id="task",
        roles=["cheap", "medium", "strong"],
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_completed_task_block(
            rows[:-1],
            stage="development",
            task_id="task",
            roles=["cheap", "medium", "strong"],
        )
