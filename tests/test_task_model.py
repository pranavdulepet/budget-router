from __future__ import annotations

from budget_router.task_model import (
    BinaryTextExample,
    HashedLinearModelHead,
    fit_hashed_linear,
    hashed_text_features,
)


def test_hashed_features_are_reproducible_and_normalized() -> None:
    left = hashed_text_features("fix parser timeout", dimension=64, seed="seed")
    right = hashed_text_features("fix parser timeout", dimension=64, seed="seed")
    assert left == right
    assert abs(sum(value * value for value in left.values()) - 1.0) < 1e-9


def test_task_aware_head_learns_different_model_preferences() -> None:
    examples = [
        BinaryTextExample("a1", "parser json token wrapper", {"cheap": True, "strong": False}),
        BinaryTextExample("a2", "parser yaml token wrapper", {"cheap": True, "strong": False}),
        BinaryTextExample("b1", "symbolic integration theorem", {"cheap": False, "strong": True}),
        BinaryTextExample("b2", "symbolic algebra theorem", {"cheap": False, "strong": True}),
    ]
    fit = fit_hashed_linear(
        examples,
        ("cheap", "strong"),
        dimension=128,
        epochs=300,
        random_seed=7,
    )
    head = HashedLinearModelHead.from_artifact(fit.to_artifact_model_head())
    parser = head.probabilities("parser json wrapper")
    theorem = head.probabilities("symbolic theorem")

    assert parser["cheap"] > parser["strong"]
    assert theorem["strong"] > theorem["cheap"]
