from __future__ import annotations

import json
from pathlib import Path

from budget_router.serialization import read_jsonl, stable_hash
from scripts.prepare_sequential_cascade_study import build_protocol


def test_cascade_protocol_fits_the_frozen_budget_and_pins_inputs() -> None:
    root = Path(__file__).parents[1]
    paths = {
        "source_study": root / "artifacts/router_baseline_study_v2.json",
        "router_freeze": root / "outputs/router_baseline_v2/router_freeze.json",
        "baseline_records": root / "outputs/router_baseline_v2/episodes.jsonl",
        "provider_hangs": root / "outputs/router_baseline_v2/provider_hangs.jsonl",
        "comparator_report": (
            root
            / "outputs/router_baseline_v2/swebench_grader/test"
            / "Qwen__Qwen3.6-35B-A3B.router-v2-test-qwen36-pinned.json"
        ),
        "model_pool": root / "configs/model_pool_coding_v7.json",
        "prices": root / "configs/tinker_prices_2026-07-27_coding_v3.json",
        "harness_prompt": root / "configs/harness_prompt.txt",
        "tasks": root / "data/swebench_verified_tasks.json",
    }
    def load(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    protocol = build_protocol(
        load(paths["source_study"]),
        load(paths["router_freeze"]),
        read_jsonl(paths["baseline_records"]),
        read_jsonl(paths["provider_hangs"]),
        load(paths["comparator_report"]),
        input_paths=paths,
    )

    without_hash = dict(protocol)
    claimed_hash = without_hash.pop("manifest_hash")
    assert stable_hash(without_hash) == claimed_hash
    assert protocol["budget"]["recorded_prior_exposure_usd"] == "158.708486065"
    assert protocol["budget"]["projected_maximum_exposure_usd"] == "177.608486065"
    assert protocol["cascade_policy"]["scout_max_calls"] == 6
    assert len(protocol["task_ids"]) == 20
