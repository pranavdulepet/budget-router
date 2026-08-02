from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Sequence

from .pricing import ModelPrice, PriceSnapshot


def apply_price_shock(
    snapshot: PriceSnapshot,
    model: str,
    *,
    multiplier: Decimal | str = Decimal("2"),
) -> PriceSnapshot:
    factor = Decimal(str(multiplier))
    if factor <= 0:
        raise ValueError("price multiplier must be positive")
    original = snapshot.price_for(model)
    shocked = replace(
        original,
        input_per_million_usd=original.input_per_million_usd * factor,
        cached_input_per_million_usd=original.cached_input_per_million_usd * factor,
        output_per_million_usd=original.output_per_million_usd * factor,
        training_per_million_usd=(
            original.training_per_million_usd * factor
            if original.training_per_million_usd is not None
            else None
        ),
    )
    models = dict(snapshot.models)
    models[model] = shocked
    return replace(
        snapshot,
        snapshot_id=f"{snapshot.snapshot_id}-shock-{model}-{factor}x",
        models=models,
        notes=f"{snapshot.notes} Preregistered {factor}x shock applied to {model}.",
    )


def remove_model(snapshot: PriceSnapshot, model: str) -> PriceSnapshot:
    snapshot.price_for(model)
    models = dict(snapshot.models)
    del models[model]
    if not models:
        raise ValueError("cannot remove the final model")
    return replace(
        snapshot,
        snapshot_id=f"{snapshot.snapshot_id}-without-{model}",
        models=models,
        notes=f"{snapshot.notes} Model removed: {model}.",
    )


def introduce_model(
    snapshot: PriceSnapshot,
    price: ModelPrice,
    *,
    calibration_probe_count: int,
    required_probe_count: int,
) -> PriceSnapshot:
    if calibration_probe_count < required_probe_count:
        raise PermissionError("held-out model requires the fixed calibration probe set")
    if price.model in snapshot.models:
        raise ValueError("model already exists in snapshot")
    models = dict(snapshot.models)
    models[price.model] = price
    return replace(
        snapshot,
        snapshot_id=f"{snapshot.snapshot_id}-plus-{price.model}",
        models=models,
        notes=f"{snapshot.notes} Held-out model introduced after fixed probes: {price.model}.",
    )


def cumulative_regret(
    rewards: Sequence[float],
    oracle_rewards: Sequence[float],
) -> tuple[float, ...]:
    if len(rewards) != len(oracle_rewards):
        raise ValueError("reward sequences must have equal length")
    total = 0.0
    result: list[float] = []
    for reward, oracle in zip(rewards, oracle_rewards, strict=True):
        total += float(oracle) - float(reward)
        result.append(total)
    return tuple(result)

