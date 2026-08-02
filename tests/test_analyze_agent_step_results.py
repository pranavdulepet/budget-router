from scripts.analyze_agent_step_results import (
    mcnemar_exact_two_sided,
    paired_bootstrap,
)


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_bootstrap(
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [0.5, 0.4, 0.3, 0.2],
        [1.0, 0.8, 0.6, 0.4],
        samples=500,
        seed=7,
    )
    second = paired_bootstrap(
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [0.5, 0.4, 0.3, 0.2],
        [1.0, 0.8, 0.6, 0.4],
        samples=500,
        seed=7,
    )

    assert first == second
    assert first["cost_saving_fraction_95ci"] == [0.5, 0.5]
    assert first["quality_difference_95ci"][0] >= 0


def test_mcnemar_exact_two_sided() -> None:
    assert mcnemar_exact_two_sided(0, 0) == 1.0
    assert mcnemar_exact_two_sided(3, 2) == 1.0
    assert mcnemar_exact_two_sided(5, 0) == 0.0625
