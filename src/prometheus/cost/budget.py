"""Budget status and pre-call checking infrastructure."""
from __future__ import annotations

from prometheus.cost.models import BudgetStatus


class BudgetManager:
    def __init__(
        self,
        configured_budget: float | None = None,
        warning_threshold: float = 0.8,
        currency: str = "USD",
    ) -> None:
        self.configured_budget = configured_budget
        self.warning_threshold = warning_threshold
        self.currency = currency

    def check_budget(self, current_known_cost: float | None) -> BudgetStatus:
        if self.configured_budget is None or current_known_cost is None:
            return BudgetStatus(
                status="UNKNOWN",
                configured_budget=self.configured_budget,
                current_known_cost=current_known_cost,
                remaining_budget=None,
                currency=self.currency,
            )
        remaining = self.configured_budget - current_known_cost
        if current_known_cost >= self.configured_budget:
            status = "EXCEEDED"
        elif current_known_cost >= self.configured_budget * self.warning_threshold:
            status = "WARNING"
        else:
            status = "AVAILABLE"
        return BudgetStatus(
            status=status,
            configured_budget=self.configured_budget,
            current_known_cost=current_known_cost,
            remaining_budget=remaining,
            currency=self.currency,
        )

    def can_spend(self, current_known_cost: float | None, estimated_cost: float | None = None) -> BudgetStatus:
        status = self.check_budget(current_known_cost)
        if status.status == "UNKNOWN" or estimated_cost is None or status.remaining_budget is None:
            return status
        projected = (status.current_known_cost or 0.0) + estimated_cost
        return self.check_budget(projected)
