"""Provider-independent usage tracking and aggregation."""
from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from prometheus.cost.models import LLMUsage, UsageSummary


class UsageRepository:
    """Minimal repository seam; replaceable with durable storage later."""

    def save(self, usage: LLMUsage) -> None:
        raise NotImplementedError

    def list(self) -> list[LLMUsage]:
        raise NotImplementedError


class InMemoryUsageRepository(UsageRepository):
    def __init__(self) -> None:
        self._records: list[LLMUsage] = []
        self._lock = Lock()

    def save(self, usage: LLMUsage) -> None:
        with self._lock:
            self._records.append(usage)

    def list(self) -> list[LLMUsage]:
        with self._lock:
            return list(self._records)


class UsageTracker:
    def __init__(self, repository: UsageRepository | None = None) -> None:
        self.repository = repository or InMemoryUsageRepository()

    def record_usage(self, usage: LLMUsage) -> None:
        self.repository.save(usage)

    def records(self) -> list[LLMUsage]:
        return self.repository.list()

    def get_usage_summary(
        self,
        investigation_id: str | None = None,
        task_id: str | None = None,
    ) -> UsageSummary:
        records: Iterable[LLMUsage] = self.records()
        if investigation_id is not None:
            records = (item for item in records if item.investigation_id == investigation_id)
        if task_id is not None:
            records = (item for item in records if item.task_id == task_id)
        selected = list(records)
        successful = [item for item in selected if item.status == "success"]
        input_tokens = _sum_known(item.input_tokens for item in selected)
        output_tokens = _sum_known(item.output_tokens for item in selected)
        total_tokens = _sum_known(item.total_tokens for item in selected)
        costs = [item.total_cost for item in selected if item.total_cost is not None]
        latencies = [item.latency_ms for item in selected]
        return UsageSummary(
            total_calls=len(selected),
            successful_calls=len(successful),
            failed_calls=len(selected) - len(successful),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost=sum(costs) if costs else None,
            cost_by_model=_cost_group(selected, "model"),
            cost_by_provider=_cost_group(selected, "provider"),
            average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        )


def _sum_known(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _cost_group(records: list[LLMUsage], field: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in records:
        if item.total_cost is not None:
            key = getattr(item, field)
            result[key] = result.get(key, 0.0) + item.total_cost
    return result
