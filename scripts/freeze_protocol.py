#!/usr/bin/env python3
"""Hash immutable protocol inputs before locked test inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
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
        "--output", type=Path, default=Path("artifacts/protocol-lock.json")
    )
    args = parser.parse_args()
    paths = [
        Path("pyproject.toml"),
        Path("configs/experiment.json"),
        args.model_pool,
        args.prices,
        Path("configs/harness_prompt.txt"),
        Path("docs/analysis_protocol.md"),
        args.split_manifest,
        Path("src/budget_router/providers/tinker.py"),
        Path("src/budget_router/mini_swe_tinker.py"),
        Path("scripts/run_tinker_pilot.py"),
        Path("scripts/tinker_protocol_smoke.py"),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing protocol inputs: {', '.join(missing)}")
    payload = {
        "schema_version": "protocol-lock-v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files": {str(path): digest(path) for path in paths},
        "locked_test_opened": False,
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
