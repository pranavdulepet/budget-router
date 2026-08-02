from decimal import Decimal

from budget_router.pricing import ModelPrice, PriceSnapshot


def make_prices() -> PriceSnapshot:
    return PriceSnapshot(
        snapshot_id="test-prices-v1",
        as_of="2026-01-01",
        source_url="https://example.test/prices",
        models={
            "cheap": ModelPrice(
                "cheap",
                input_per_million_usd=Decimal("1"),
                output_per_million_usd=Decimal("2"),
                cached_input_per_million_usd=Decimal("0.2"),
            ),
            "strong": ModelPrice(
                "strong",
                input_per_million_usd=Decimal("4"),
                output_per_million_usd=Decimal("8"),
                cached_input_per_million_usd=Decimal("0.8"),
            ),
        },
    )

