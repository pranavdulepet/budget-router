from __future__ import annotations

import numpy as np

from scripts.run_public_router_study import (
    Matrix,
    evaluate_selection,
    select_models,
)


def _matrix() -> Matrix:
    return Matrix(
        models=["cheap", "strong"],
        texts=["easy", "hard"],
        datasets=np.asarray(["a", "b"], dtype=object),
        route_splits=np.asarray(["id_test", "id_test"], dtype=object),
        prompt_keys=["1", "2"],
        scores=np.asarray([[1.0, 1.0], [0.0, 1.0]]),
        acceptable=np.asarray([[True, True], [False, True]]),
        costs=np.asarray([[0.1, 1.0], [0.1, 1.0]]),
    )


def test_selector_trades_predicted_quality_for_expected_cost() -> None:
    predictions = np.asarray([[0.9, 0.95], [0.2, 0.9]])
    costs = np.asarray([0.1, 1.0])

    quality = select_models(predictions, costs, 0.0)
    balanced = select_models(predictions, costs, 0.2)

    assert quality.tolist() == [1, 1]
    assert balanced.tolist() == [0, 1]


def test_evaluation_reports_macro_quality_cost_and_routes() -> None:
    matrix = _matrix()
    indices = np.asarray([0, 1])
    metrics = evaluate_selection(
        matrix,
        indices,
        np.asarray([0, 1]),
        predictions=np.asarray([[0.8, 0.9], [0.2, 0.8]]),
    )

    assert metrics["macro_score"] == 1.0
    assert metrics["micro_success_rate"] == 1.0
    assert metrics["total_cost_usd"] == 1.1
    assert metrics["route_counts"] == {"cheap": 1, "strong": 1}
    assert 0.0 <= metrics["selected_brier"] <= 1.0
