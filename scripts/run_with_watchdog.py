from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Sequence


WATCHDOG_EXIT_CODE = 124


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    termination_grace_seconds: float,
) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=termination_grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_with_watchdog(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float = 5.0,
) -> int:
    if not command:
        raise ValueError("a command is required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if termination_grace_seconds < 0:
        raise ValueError("termination_grace_seconds must be non-negative")

    process = subprocess.Popen(list(command), start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"watchdog: process exceeded {timeout_seconds:g} seconds; "
            "terminating its process group",
            file=sys.stderr,
            flush=True,
        )
        _terminate_process_group(
            process,
            termination_grace_seconds=termination_grace_seconds,
        )
        return WATCHDOG_EXIT_CODE
    except KeyboardInterrupt:
        _terminate_process_group(
            process,
            termination_grace_seconds=termination_grace_seconds,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one command with a process-group wall-clock watchdog."
    )
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--termination-grace-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        return_code = run_with_watchdog(
            command,
            timeout_seconds=args.timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
