"""Deterministic cost-aware task router."""
from __future__ import annotations

import re

from prometheus.cost.budget import BudgetManager
from prometheus.cost.tracker import UsageTracker
from prometheus.routing.models import EscalationRequest, ExecutionPlan
from prometheus.routing.policy import RoutingPolicy


class TaskRouter:
    """Classify requests without invoking an LLM or provider."""

    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        budget_manager: BudgetManager | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self.policy = policy or RoutingPolicy()
        self.budget_manager = budget_manager or BudgetManager()
        self.usage_tracker = usage_tracker or UsageTracker()

    def route(self, question: str, document_count: int | None = None) -> ExecutionPlan:
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("Question cannot be empty or whitespace-only.")

        normalized = clean_question.lower()
        if _matches_any(normalized, self.policy.signals_for("deterministic")):
            plan = ExecutionPlan(
                route="DETERMINISTIC",
                complexity="DETERMINISTIC",
                recommended_model_tier=None,
                requires_llm=False,
                requires_rag=False,
                requires_investigation=False,
                requires_external_research=False,
                max_allowed_llm_calls=0,
                budget_policy="not_applicable",
                routing_reason="Deterministic metadata or status operation.",
                confidence="high",
                escalation_allowed=False,
            )
            return plan

        external = _matches_any(normalized, self.policy.signals_for("external"))
        high = _matches_any(normalized, self.policy.signals_for("high"))
        medium = _matches_any(normalized, self.policy.signals_for("medium"))
        low_signal = _matches_any(normalized, self.policy.signals_for("low"))
        multiple_documents = document_count is not None and document_count > 1

        if external:
            high = high or any(token in normalized for token in ("why", "investigate", "research", "factors"))
            medium = medium or multiple_documents
        if high:
            route = "HIGH"
            reason = "Causal, investigative, recommendation, or competing-explanation request."
            tier = "STRONG"
            max_calls = 8
            requires_investigation = True
            confidence = "high"
        elif medium or multiple_documents:
            route = "MEDIUM"
            reason = "Synthesis or comparison across evidence is required."
            tier = "MEDIUM"
            max_calls = 3
            requires_investigation = False
            confidence = "medium"
        else:
            route = "LOW"
            reason = "Simple factual or summarization request suitable for document RAG."
            if not low_signal:
                reason = "Default document question routed to the lowest useful LLM tier."
            tier = "CHEAP"
            max_calls = 1
            requires_investigation = False
            confidence = "medium" if low_signal else "low"

        budget = self.budget_manager.check_budget(self.usage_tracker.get_usage_summary().total_cost)
        budget_policy = "within_budget" if budget.status in ("AVAILABLE", "WARNING") else "budget_unknown"
        if budget.status == "WARNING" and route == "HIGH":
            tier = "MEDIUM"
            max_calls = 3
            budget_policy = "reduced_depth"
            reason += " Strong synthesis was reduced to the medium tier because budget is near its warning threshold."
        elif budget.status == "EXCEEDED" and route == "HIGH":
            tier = "MEDIUM"
            max_calls = 1
            budget_policy = "human_confirmation_required"
            reason += " Strong synthesis was not selected because the configured budget is exceeded."

        return ExecutionPlan(
            route=route,
            complexity=route,
            recommended_model_tier=tier,
            requires_llm=True,
            requires_rag=True,
            requires_investigation=requires_investigation,
            requires_external_research=external,
            max_allowed_llm_calls=max_calls,
            budget_policy=budget_policy,
            routing_reason=reason,
            confidence=confidence,
            escalation_allowed=route != "DETERMINISTIC",
        )

    def escalate(self, plan: ExecutionPlan, request: EscalationRequest) -> ExecutionPlan:
        """Return a higher-level plan for a later execution stage to consume."""
        next_route = {"DETERMINISTIC": "LOW", "LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "HIGH"}[plan.route]
        if next_route == "HIGH":
            tier = "STRONG"
            calls = max(plan.max_allowed_llm_calls, 8)
            investigation = True
        elif next_route == "MEDIUM":
            tier = "MEDIUM"
            calls = max(plan.max_allowed_llm_calls, 3)
            investigation = False
        else:
            tier = "CHEAP"
            calls = max(plan.max_allowed_llm_calls, 1)
            investigation = False
        return plan.model_copy(
            update={
                "route": next_route,
                "complexity": next_route,
                "recommended_model_tier": tier,
                "requires_llm": True,
                "requires_rag": next_route != "DETERMINISTIC",
                "requires_investigation": investigation,
                "max_allowed_llm_calls": calls,
                "escalation_allowed": next_route != "HIGH",
                "escalated": True,
                "escalation_trigger": request.trigger,
                "routing_reason": f"Escalated from {plan.route}: {request.reason}",
            }
        )


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


_default_router: TaskRouter | None = None


def get_task_router() -> TaskRouter:
    global _default_router
    if _default_router is None:
        from prometheus.models.gateway import get_model_gateway

        gateway = get_model_gateway()
        _default_router = TaskRouter(usage_tracker=gateway.usage_tracker)
    return _default_router
