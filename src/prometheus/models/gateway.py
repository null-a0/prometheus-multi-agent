"""Provider-neutral model gateway and legacy client compatibility adapter."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from prometheus.cost.calculator import calculate_cost
from prometheus.cost.models import LLMUsage
from prometheus.cost.pricing import PricingRegistry
from prometheus.cost.tracker import UsageTracker
from prometheus.llm.base import BaseLLMClient
from prometheus.models.providers.openrouter import OpenRouterProvider
from prometheus.models.types import (
    ModelGatewayError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderResponse,
)

logger = logging.getLogger(__name__)


class LegacyClientProvider:
    """Adapt the existing BaseLLMClient seam to the provider interface."""

    provider_name = "legacy"

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client
        self.default_model = getattr(client, "model", "unknown")

    def generate(self, request: ModelRequest) -> ProviderResponse:
        system_prompt = None
        prompt = request.messages[-1]["content"] if request.messages else ""
        if request.messages and request.messages[0].get("role") == "system":
            system_prompt = request.messages[0]["content"]
        result = self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ProviderResponse(
            generated_text=result,
            provider=self.provider_name,
            model=request.model or self.default_model,
        )


class ModelGateway:
    """Single application boundary for all model invocations."""

    def __init__(
        self,
        provider: ModelProvider,
        usage_tracker: UsageTracker | None = None,
        pricing_registry: PricingRegistry | None = None,
        budget_manager: Any | None = None,
    ) -> None:
        self.provider = provider
        self.usage_tracker = usage_tracker or UsageTracker()
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.budget_manager = budget_manager

    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        investigation_id: str | None = None,
        task_id: str | None = None,
        budget_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        request_messages = messages or ([{"role": "user", "content": prompt or ""}])
        if messages is None and system_prompt:
            request_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt or ""},
            ]
        request = ModelRequest(
            messages=request_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            investigation_id=investigation_id,
            task_id=task_id,
            budget_context=budget_context,
        )
        usage_id = str(uuid.uuid4())
        provider_name = self.provider.provider_name
        selected_model = model or self.provider.default_model
        started = time.perf_counter()
        logger.info("Model invocation started", extra={"provider": provider_name, "model": selected_model})

        if self.budget_manager is not None:
            current_cost = self.usage_tracker.get_usage_summary().total_cost
            budget_status = self.budget_manager.check_budget(current_cost)
            if budget_status.status == "EXCEEDED":
                latency_ms = _elapsed_ms(started)
                self._record_failure(
                    usage_id, request, provider_name, selected_model, latency_ms, "budget_exceeded"
                )
                raise ModelGatewayError(
                    category="unavailable",
                    message="Configured model budget has been exceeded.",
                    provider=provider_name,
                    model=selected_model,
                    usage_id=usage_id,
                )

        try:
            provider_response = self.provider.generate(request)
        except ModelGatewayError as exc:
            latency_ms = _elapsed_ms(started)
            self._record_failure(usage_id, request, provider_name, selected_model, latency_ms, exc.category)
            exc.usage_id = usage_id
            logger.warning(
                "Model invocation failed",
                extra={"provider": provider_name, "model": selected_model, "failure_category": exc.category},
            )
            raise
        except Exception as exc:
            latency_ms = _elapsed_ms(started)
            self._record_failure(usage_id, request, provider_name, selected_model, latency_ms, "unknown")
            raise ModelGatewayError(
                category="unknown",
                message="Model invocation failed.",
                provider=provider_name,
                model=selected_model,
                usage_id=usage_id,
            ) from exc

        latency_ms = _elapsed_ms(started)
        pricing = self.pricing_registry.get_price(provider_response.provider, provider_response.model)
        input_cost, output_cost, total_cost = calculate_cost(
            provider_response.input_tokens,
            provider_response.output_tokens,
            pricing,
        )
        self.usage_tracker.record_usage(
            LLMUsage(
                usage_id=usage_id,
                investigation_id=request.investigation_id,
                task_id=request.task_id,
                provider=provider_response.provider,
                model=provider_response.model,
                input_tokens=provider_response.input_tokens,
                output_tokens=provider_response.output_tokens,
                total_tokens=provider_response.total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                currency=pricing.currency if pricing else "USD",
                latency_ms=latency_ms,
                token_usage_source=("provider_reported" if provider_response.total_tokens is not None else "unknown"),
            )
        )
        logger.info(
            "Model invocation completed",
            extra={
                "provider": provider_response.provider,
                "model": provider_response.model,
                "input_tokens": provider_response.input_tokens,
                "output_tokens": provider_response.output_tokens,
                "latency_ms": latency_ms,
                "cost_known": total_cost is not None,
            },
        )
        return ModelResponse(
            generated_text=provider_response.generated_text,
            provider=provider_response.provider,
            model=provider_response.model,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
            total_tokens=provider_response.total_tokens,
            latency_ms=latency_ms,
            request_id=provider_response.request_id,
            finish_reason=provider_response.finish_reason,
        )

    def _record_failure(
        self,
        usage_id: str,
        request: ModelRequest,
        provider: str,
        model: str,
        latency_ms: float,
        error_type: str,
    ) -> None:
        self.usage_tracker.record_usage(
            LLMUsage(
                usage_id=usage_id,
                investigation_id=request.investigation_id,
                task_id=request.task_id,
                provider=provider,
                model=model,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                input_cost=None,
                output_cost=None,
                total_cost=None,
                currency="USD",
                latency_ms=latency_ms,
                status="failed",
                error_type=error_type,
                token_usage_source="unknown",
            )
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


_default_gateway: ModelGateway | None = None


def get_model_gateway() -> ModelGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = ModelGateway(provider=OpenRouterProvider())
    return _default_gateway
