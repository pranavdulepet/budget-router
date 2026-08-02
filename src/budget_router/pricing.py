from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .types import TokenUsage, as_usd

MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    model: str
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal
    cached_input_per_million_usd: Decimal
    training_per_million_usd: Decimal | None = None
    context_tokens: int = 65_536

    def __post_init__(self) -> None:
        for name in (
            "input_per_million_usd",
            "output_per_million_usd",
            "cached_input_per_million_usd",
        ):
            value = as_usd(getattr(self, name))
            object.__setattr__(self, name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.training_per_million_usd is not None:
            training = as_usd(self.training_per_million_usd)
            object.__setattr__(self, "training_per_million_usd", training)
            if training < 0:
                raise ValueError("training price must be non-negative")
        if self.cached_input_per_million_usd > self.input_per_million_usd:
            raise ValueError("cached input price cannot exceed uncached input price")
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")

    def conservative_cost(self, usage: TokenUsage) -> Decimal:
        """Hard-cap charge: all prompt tokens are priced as uncached."""
        return (
            as_usd(usage.input_tokens) * self.input_per_million_usd
            + as_usd(usage.output_tokens) * self.output_per_million_usd
        ) / MILLION

    def billed_cost(self, usage: TokenUsage) -> Decimal:
        uncached = usage.input_tokens - usage.cached_input_tokens
        return (
            as_usd(uncached) * self.input_per_million_usd
            + as_usd(usage.cached_input_tokens) * self.cached_input_per_million_usd
            + as_usd(usage.output_tokens) * self.output_per_million_usd
        ) / MILLION


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    snapshot_id: str
    as_of: str
    source_url: str
    models: Mapping[str, ModelPrice]
    currency: str = "USD"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.as_of or not self.source_url:
            raise ValueError("snapshot_id, as_of, and source_url are required")
        if self.currency != "USD":
            raise ValueError("v1 only supports USD")
        if not self.models:
            raise ValueError("price snapshot needs at least one model")
        for key, price in self.models.items():
            if key != price.model:
                raise ValueError(f"price map key {key!r} does not match model {price.model!r}")

    def price_for(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError as exc:
            raise KeyError(f"model {model!r} is absent from snapshot {self.snapshot_id!r}") from exc

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceSnapshot:
        models = {
            model: ModelPrice(
                model=model,
                input_per_million_usd=item["input_per_million_usd"],
                cached_input_per_million_usd=item["cached_input_per_million_usd"],
                output_per_million_usd=item["output_per_million_usd"],
                training_per_million_usd=item.get("training_per_million_usd"),
                context_tokens=int(item.get("context_tokens", 65_536)),
            )
            for model, item in data["models"].items()
        }
        return cls(
            snapshot_id=data["snapshot_id"],
            as_of=data["as_of"],
            source_url=data["source_url"],
            models=models,
            currency=data.get("currency", "USD"),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> PriceSnapshot:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "as_of": self.as_of,
            "source_url": self.source_url,
            "currency": self.currency,
            "notes": self.notes,
            "models": {
                name: {
                    "input_per_million_usd": str(price.input_per_million_usd),
                    "cached_input_per_million_usd": str(
                        price.cached_input_per_million_usd
                    ),
                    "output_per_million_usd": str(price.output_per_million_usd),
                    "training_per_million_usd": (
                        str(price.training_per_million_usd)
                        if price.training_per_million_usd is not None
                        else None
                    ),
                    "context_tokens": price.context_tokens,
                }
                for name, price in self.models.items()
            },
        }

