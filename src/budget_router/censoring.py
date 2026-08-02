from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .metrics import brier_score


@dataclass(frozen=True, slots=True)
class CensoringStressResult:
    mechanism: str
    retained: int
    censored: int
    complete_case_brier: float
    original_brier: float
    absolute_shift: float


def censoring_stress_test(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    probability: float,
    outcome_odds_ratio: float = 1.0,
    seed: int = 0,
) -> CensoringStressResult:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes need equal non-zero length")
    if not 0 <= probability < 1 or outcome_odds_ratio <= 0:
        raise ValueError("invalid censoring parameters")
    rng = random.Random(seed)
    keep: list[bool] = []
    for outcome in outcomes:
        row_probability = probability
        if outcome:
            odds = probability / max(1e-12, 1 - probability)
            shifted_odds = odds * outcome_odds_ratio
            row_probability = shifted_odds / (1 + shifted_odds)
        keep.append(rng.random() >= row_probability)
    if not any(keep):
        keep[0] = True
    retained_probabilities = [
        value for value, retain in zip(probabilities, keep, strict=True) if retain
    ]
    retained_outcomes = [
        value for value, retain in zip(outcomes, keep, strict=True) if retain
    ]
    original = brier_score(probabilities, outcomes)
    complete = brier_score(retained_probabilities, retained_outcomes)
    mechanism = "random" if outcome_odds_ratio == 1 else "outcome_dependent"
    return CensoringStressResult(
        mechanism=mechanism,
        retained=sum(keep),
        censored=len(keep) - sum(keep),
        complete_case_brier=complete,
        original_brier=original,
        absolute_shift=abs(complete - original),
    )

