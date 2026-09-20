"""Token-cost calculation helpers."""
from __future__ import annotations

from prometheus.cost.models import Pricing


def calculate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: Pricing | None,
) -> tuple[float | None, float | None, float | None]:
    """Return input, output, and total costs; unknown inputs remain unknown."""
    if pricing is None:
        return None, None, None
    input_cost = (
        input_tokens * pricing.input_price_per_million / 1_000_000
        if input_tokens is not None
        else None
    )
    output_cost = (
        output_tokens * pricing.output_price_per_million / 1_000_000
        if output_tokens is not None
        else None
    )
    total_cost = input_cost + output_cost if input_cost is not None and output_cost is not None else None
    return input_cost, output_cost, total_cost
