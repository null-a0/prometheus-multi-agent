"""Cost and usage data contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Pricing:
    provider: str
    model: str
    input_price_per_million: float
    output_price_per_million: float
    currency: str = "USD"
    effective_date: str | None = None


@dataclass
class LLMUsage:
    usage_id: str
    investigation_id: str | None
    task_id: str | None
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    currency: str
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "success"
    error_type: str | None = None
    token_usage_source: str = "provider_reported"


@dataclass(frozen=True)
class UsageSummary:
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    total_cost: float | None
    cost_by_model: dict[str, float]
    cost_by_provider: dict[str, float]
    average_latency_ms: float | None
    currency: str = "USD"


@dataclass(frozen=True)
class BudgetStatus:
    status: str
    configured_budget: float | None
    current_known_cost: float | None
    remaining_budget: float | None
    currency: str = "USD"
    details: dict[str, Any] = field(default_factory=dict)
