from __future__ import annotations

from decimal import Decimal

import pytest

from scripts.run_tinker_pilot import _select_treatments, _summary


def _treatments() -> list[dict[str, str]]:
    return [{"model": f"model-{index}"} for index in range(5)]


def test_select_treatments_preserves_frozen_pool_indices() -> None:
    selected = _select_treatments(
        _treatments(),
        ["model-3", "model-1"],
    )

    assert [(index, row["model"]) for index, row in selected] == [
        (1, "model-1"),
        (3, "model-3"),
    ]


@pytest.mark.parametrize(
    "selected, message",
    [
        (["missing"], "not in the frozen pool"),
        (["model-1", "model-1"], "at most once"),
    ],
)
def test_select_treatments_rejects_invalid_scope(
    selected: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _select_treatments(_treatments(), selected)


def test_single_model_summary_labels_projected_contribution() -> None:
    records = [
        {
            "model": "model-1",
            "task_id": f"task-{index}",
            "emitted_tool_call": True,
            "arguments_valid": True,
            "tool_result_returned": True,
            "continued_after_tool": True,
            "terminal_record_valid": True,
            "conservative_cost_usd": str(Decimal("0.10")),
            "latency_seconds": 10.0,
            "provider_failed": False,
        }
        for index in range(12)
    ]

    summary = _summary(
        records,
        pool_version="pool-v1",
        price_snapshot="prices-v1",
        models_in_scope=["model-1"],
    )

    assert summary["pilot_scope_models"] == ["model-1"]
    assert summary["matrix_shape"] == "500×1×3"
    assert summary["reference_matrix_shape"] == "500×5×3"
    assert summary["structural_gate_passed"] is True
