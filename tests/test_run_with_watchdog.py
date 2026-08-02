from __future__ import annotations

import sys
import time

from scripts.run_with_watchdog import WATCHDOG_EXIT_CODE, run_with_watchdog


def test_watchdog_returns_child_exit_code() -> None:
    assert (
        run_with_watchdog(
            [sys.executable, "-c", "raise SystemExit(7)"],
            timeout_seconds=1,
        )
        == 7
    )


def test_watchdog_terminates_timed_out_process_group() -> None:
    started = time.monotonic()
    result = run_with_watchdog(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_seconds=0.05,
        termination_grace_seconds=0.05,
    )
    assert result == WATCHDOG_EXIT_CODE
    assert time.monotonic() - started < 1
