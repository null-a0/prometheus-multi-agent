from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from prometheus.cost import BudgetManager, UsageTracker
from prometheus.models.gateway import ModelGateway
from prometheus.models.types import ProviderResponse
from prometheus.rag import pipeline
from prometheus.routing import EscalationManager, EscalationRequest, RoutingPolicy, TaskRouter


class NoCallProvider:
    provider_name = "fake"
    default_model = "fake-model"

    def generate(self, _request):
        raise AssertionError("TaskRouter must not invoke a provider")


def test_deterministic_routing_needs_no_llm() -> None:
    plan = TaskRouter().route("List the uploaded documents")

    assert plan.route == "DETERMINISTIC"
    assert plan.requires_llm is False
    assert plan.requires_rag is False
    assert plan.max_allowed_llm_calls == 0
    assert plan.recommended_model_tier is None


def test_low_factual_and_summary_routing_uses_cheap_tier() -> None:
    factual = TaskRouter().route("What is the revenue in Q3?")
    summary = TaskRouter().route("Summarize the uploaded annual report.")

    assert factual.route == "LOW"
    assert factual.recommended_model_tier == "CHEAP"
    assert factual.requires_rag is True
    assert factual.requires_investigation is False
    assert summary.route == "LOW"


def test_medium_comparison_routing_uses_medium_tier() -> None:
    plan = TaskRouter().route("Compare the conclusions of the Q2 and Q3 reports.")

    assert plan.route == "MEDIUM"
    assert plan.recommended_model_tier == "MEDIUM"
    assert plan.max_allowed_llm_calls == 3
    assert plan.requires_investigation is False


def test_high_causal_routing_requires_investigation() -> None:
    plan = TaskRouter().route("Why did revenue decline?")

    assert plan.route == "HIGH"
    assert plan.recommended_model_tier == "STRONG"
    assert plan.requires_investigation is True
    assert plan.escalation_allowed is True


def test_external_research_is_detected() -> None:
    plan = TaskRouter().route(
        "Why did revenue decline and what external market factors contributed?"
    )

    assert plan.route == "HIGH"
    assert plan.requires_external_research is True
    assert plan.requires_investigation is True


def test_custom_policy_is_configurable() -> None:
    policy = RoutingPolicy(patterns={"deterministic": (r"\bping\b",)})
    plan = TaskRouter(policy=policy).route("Ping the service")

    assert plan.route == "DETERMINISTIC"


def test_budget_warning_reduces_high_route_depth_and_records_reason() -> None:
    budget = BudgetManager(configured_budget=1.0, warning_threshold=0.8)
    tracker = UsageTracker()
    tracker.get_usage_summary = Mock(  # type: ignore[method-assign]
        return_value=type("Summary", (), {"total_cost": 0.9})()
    )
    plan = TaskRouter(budget_manager=budget, usage_tracker=tracker).route("Why did revenue decline?")

    assert plan.route == "HIGH"
    assert plan.recommended_model_tier == "MEDIUM"
    assert plan.budget_policy == "reduced_depth"
    assert "budget" in plan.routing_reason.lower()


def test_escalation_manager_promotes_low_plan_without_provider_call() -> None:
    router = TaskRouter()
    plan = router.route("What is the revenue in Q3?")
    escalated = EscalationManager(router).escalate(
        plan,
        EscalationRequest(trigger="insufficient_evidence", reason="No grounded passage was found."),
    )

    assert escalated.route == "MEDIUM"
    assert escalated.escalated is True
    assert escalated.escalation_trigger == "insufficient_evidence"
    assert escalated.requires_investigation is False


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        TaskRouter().route("  ")


def test_router_does_not_invoke_model_gateway_or_provider() -> None:
    router = TaskRouter()
    gateway = ModelGateway(NoCallProvider())

    plan = router.route("Investigate customer churn")

    assert plan.requires_investigation is True
    assert gateway.usage_tracker.records() == []


def test_rag_response_contains_routing_metadata_without_changing_execution() -> None:
    candidate = {"id": "chunk", "text": "Revenue was stable.", "metadata": {}, "score": 1.0}
    fake_provider = type(
        "FakeProvider",
        (),
        {
            "provider_name": "fake",
            "default_model": "fake-model",
            "generate": lambda self, _request: ProviderResponse(
                generated_text="Revenue was stable [1].",
                provider="fake",
                model="fake-model",
            ),
        },
    )()
    gateway = ModelGateway(fake_provider)
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[candidate]),
        patch.object(pipeline.reranker, "rerank", return_value=[candidate]),
        patch.object(pipeline.reranker, "apply_recency_and_credibility_filters", return_value=[candidate]),
    ):
        result = pipeline.query_rag(
            "What is the revenue?",
            model_gateway=gateway,
            task_router=TaskRouter(),
        )

    assert result.answer == "Revenue was stable [1]."
    assert result.routing is not None
    assert result.routing.route == "LOW"
    assert result.routing.model_tier == "CHEAP"
    assert result.routing.requires_investigation is False
