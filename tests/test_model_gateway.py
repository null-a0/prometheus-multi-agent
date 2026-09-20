from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prometheus.api.app import app
from prometheus.cost import BudgetManager, Pricing, PricingRegistry, UsageTracker, calculate_cost
from prometheus.cost.models import LLMUsage
from prometheus.models.gateway import ModelGateway
from prometheus.models.providers.openrouter import OpenRouterProvider
from prometheus.models.types import ModelGatewayError, ModelRequest, ProviderResponse
from prometheus.rag import pipeline


class FakeProvider:
    provider_name = "fake"
    default_model = "fake-model"

    def __init__(self, response: ProviderResponse | None = None, error: Exception | None = None):
        self.response = response or ProviderResponse(
            generated_text="fake answer",
            provider="fake",
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            request_id="req-1",
            finish_reason="stop",
        )
        self.error = error
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


def _gateway(provider: FakeProvider) -> ModelGateway:
    registry = PricingRegistry(
        [Pricing("fake", "fake-model", input_price_per_million=2.0, output_price_per_million=4.0)]
    )
    return ModelGateway(provider, usage_tracker=UsageTracker(), pricing_registry=registry)


def test_model_gateway_maps_request_and_response() -> None:
    provider = FakeProvider()
    response = _gateway(provider).generate(
        prompt="hello",
        model="fake-model",
        temperature=0.2,
        max_tokens=20,
        investigation_id="inv-1",
        task_id="task-1",
    )

    assert response.generated_text == "fake answer"
    assert response.provider == "fake"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.total_tokens == 15
    assert response.latency_ms >= 0
    assert provider.requests[0].messages == [{"role": "user", "content": "hello"}]
    assert _gateway(provider).usage_tracker.records() == []


def test_usage_recording_and_cost_aggregation() -> None:
    provider = FakeProvider()
    gateway = _gateway(provider)
    gateway.generate(prompt="hello", investigation_id="inv-1")
    summary = gateway.usage_tracker.get_usage_summary(investigation_id="inv-1")

    assert summary.total_calls == 1
    assert summary.successful_calls == 1
    assert summary.failed_calls == 0
    assert summary.total_input_tokens == 10
    assert summary.total_output_tokens == 5
    assert summary.total_tokens == 15
    assert summary.total_cost == pytest.approx(0.00004)
    assert summary.cost_by_model == {"fake-model": pytest.approx(0.00004)}
    assert summary.cost_by_provider == {"fake": pytest.approx(0.00004)}


def test_failed_usage_is_recorded_and_error_is_normalized() -> None:
    provider = FakeProvider(
        error=ModelGatewayError(
            category="rate_limit",
            message="provider unavailable",
            provider="fake",
            model="fake-model",
        )
    )
    gateway = _gateway(provider)

    with pytest.raises(ModelGatewayError) as raised:
        gateway.generate(prompt="hello")

    assert raised.value.category == "rate_limit"
    record = gateway.usage_tracker.records()[0]
    assert record.status == "failed"
    assert record.error_type == "rate_limit"
    assert record.total_cost is None
    assert record.input_tokens is None


def test_openrouter_adapter_maps_provider_response_and_usage() -> None:
    response = SimpleNamespace(
        id="openrouter-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    provider = OpenRouterProvider(api_key="test-key", model="test-model", client=sdk_client)

    result = provider.generate(ModelRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.generated_text == "answer"
    assert result.provider == "openrouter"
    assert result.model == "test-model"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.total_tokens == 5
    assert result.request_id == "openrouter-1"


def test_openrouter_provider_normalizes_authentication_failure() -> None:
    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    with patch.object(provider, "_get_client", side_effect=RuntimeError("bad auth")), pytest.raises(
        ModelGatewayError
    ) as raised:
        provider.generate(ModelRequest(messages=[{"role": "user", "content": "hi"}]))

    assert raised.value.category == "unknown"
    assert str(raised.value) == "Model provider request failed."


def test_missing_tokens_and_unknown_pricing_remain_unknown() -> None:
    assert calculate_cost(10, None, Pricing("fake", "model", 2.0, 4.0)) == (0.00002, None, None)
    assert calculate_cost(10, 5, None) == (None, None, None)

    gateway = ModelGateway(
        FakeProvider(
            response=ProviderResponse(
                generated_text="answer",
                provider="fake",
                model="unknown-model",
            )
        ),
        usage_tracker=UsageTracker(),
        pricing_registry=PricingRegistry(),
    )
    gateway.generate(prompt="hi")
    usage = gateway.usage_tracker.records()[0]
    assert usage.input_tokens is None
    assert usage.total_cost is None
    assert usage.token_usage_source == "unknown"


def test_budget_status_and_pre_call_budget_block() -> None:
    manager = BudgetManager(configured_budget=1.0, warning_threshold=0.8)
    assert manager.check_budget(0.5).status == "AVAILABLE"
    assert manager.check_budget(0.8).status == "WARNING"
    assert manager.check_budget(1.0).status == "EXCEEDED"
    assert manager.check_budget(None).status == "UNKNOWN"

    provider = FakeProvider()
    tracker = UsageTracker()
    tracker.record_usage(
        LLMUsage(
            usage_id="existing",
            investigation_id=None,
            task_id=None,
            provider="fake",
            model="fake-model",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            input_cost=1.0,
            output_cost=0.0,
            total_cost=1.0,
            currency="USD",
            latency_ms=1.0,
        )
    )
    gateway = ModelGateway(provider, usage_tracker=tracker, budget_manager=manager)
    with pytest.raises(ModelGatewayError, match="budget"):
        gateway.generate(prompt="blocked")
    assert provider.requests == []
    assert tracker.records()[-1].error_type == "budget_exceeded"


def test_multiple_models_and_providers_aggregate_separately() -> None:
    tracker = UsageTracker()
    for provider, model, cost in (("one", "m1", 1.0), ("two", "m2", 2.0)):
        tracker.record_usage(
            LLMUsage(
                usage_id=f"{provider}-1",
                investigation_id=None,
                task_id=None,
                provider=provider,
                model=model,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                input_cost=cost,
                output_cost=0.0,
                total_cost=cost,
                currency="USD",
                latency_ms=2.0,
            )
        )
    summary = tracker.get_usage_summary()
    assert summary.total_calls == 2
    assert summary.cost_by_provider == {"one": 1.0, "two": 2.0}
    assert summary.cost_by_model == {"m1": 1.0, "m2": 2.0}


def test_rag_generation_uses_model_gateway() -> None:
    provider = FakeProvider()
    gateway = _gateway(provider)
    candidate = {"id": "c-1", "text": "retrieved", "metadata": {}, "score": 1.0}
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[candidate]),
        patch.object(pipeline.reranker, "rerank", return_value=[candidate]),
        patch.object(pipeline.reranker, "apply_recency_and_credibility_filters", return_value=[candidate]),
    ):
        result = pipeline.query_rag("question", model_gateway=gateway)

    assert result.answer == "fake answer"
    assert provider.requests[0].messages[0]["role"] == "system"
    assert gateway.usage_tracker.get_usage_summary().total_calls == 1


def test_usage_summary_api_does_not_expose_prompt_data() -> None:
    client = TestClient(app)
    response = client.get("/api/usage/summary")
    assert response.status_code == 200
    assert "total_calls" in response.json()
    assert "prompt" not in response.text.lower()
