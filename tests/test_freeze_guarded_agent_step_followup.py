from __future__ import annotations

from scripts.freeze_guarded_agent_step_followup import (
    ShadowPrefix,
    replay_shadow_probabilities,
    select_live_threshold,
)


def _trajectory(values: list[float]) -> list[ShadowPrefix]:
    return [
        ShadowPrefix(
            calibrated_probability=value,
            approximate_context_tokens=100,
        )
        for value in values
    ]


def test_shadow_replay_matches_initial_burst_and_dwell_guards() -> None:
    metrics = replay_shadow_probabilities(
        [_trajectory([0.1] * 8)],
        threshold=0.5,
        force_strong_initial_calls=2,
        maximum_consecutive_cheap_calls=2,
        force_strong_context_tokens=24_000,
        strong_minimum_dwell_calls=2,
    )
    assert metrics["cheap_calls"] == 4
    assert metrics["strong_calls"] == 4
    assert metrics["maximum_observed_consecutive_cheap_calls"] == 2
    assert metrics["guard_reasons"] == {
        "force_strong_initial_calls": 2,
        "force_strong_context_guard": 0,
        "force_strong_dwell": 1,
        "force_strong_cheap_burst": 1,
    }


def test_live_threshold_is_lowest_observed_value_meeting_constraints() -> None:
    trajectories = [
        _trajectory([0.9, 0.9, 0.10, 0.20, 0.9]),
        _trajectory([0.9, 0.9, 0.15, 0.25, 0.9]),
    ]
    threshold, report = select_live_threshold(
        trajectories,
        minimum_cheap_call_share=0.2,
        maximum_cheap_call_share=0.5,
        minimum_trajectories_with_cheap_call=2,
        force_strong_initial_calls=2,
        maximum_consecutive_cheap_calls=2,
        force_strong_context_tokens=24_000,
        strong_minimum_dwell_calls=2,
    )
    assert threshold == 0.15
    assert report["selected"]["cheap_calls"] == 2
    assert report["selected"]["trajectories_with_cheap_call"] == 2


def test_context_guard_never_routes_cheap() -> None:
    trajectories = [
        [
            ShadowPrefix(
                calibrated_probability=0.01,
                approximate_context_tokens=30_000,
            )
        ]
    ]
    metrics = replay_shadow_probabilities(
        trajectories,
        threshold=0.5,
        force_strong_initial_calls=0,
        maximum_consecutive_cheap_calls=2,
        force_strong_context_tokens=24_000,
        strong_minimum_dwell_calls=2,
    )
    assert metrics["cheap_calls"] == 0
    assert metrics["guard_reasons"]["force_strong_context_guard"] == 1
