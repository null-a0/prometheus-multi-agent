"""Configurable model pricing registry.

No production prices are bundled. Applications or tests must register prices
explicitly before costs can be calculated.
"""
from __future__ import annotations

from prometheus.cost.models import Pricing


class PricingRegistry:
    def __init__(self, prices: list[Pricing] | None = None) -> None:
        self._prices: dict[tuple[str, str], Pricing] = {}
        for price in prices or []:
            self.register(price)

    def register(self, pricing: Pricing) -> None:
        self._prices[(pricing.provider.lower(), pricing.model)] = pricing

    def get_price(self, provider: str, model: str) -> Pricing | None:
        return self._prices.get((provider.lower(), model))

    def models(self) -> list[Pricing]:
        return list(self._prices.values())
