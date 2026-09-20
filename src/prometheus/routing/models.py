"""Task routing and escalation contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RouteLevel = Literal["DETERMINISTIC", "LOW", "MEDIUM", "HIGH"]
ModelTier = Literal["CHEAP", "MEDIUM", "STRONG"]
BudgetPolicy = Literal[
    "not_applicable",
    "budget_unknown",
    "within_budget",
    "reduced_depth",
    "human_confirmation_required",
]


class ExecutionPlan(BaseModel):
    """Transparent plan describing what an application should execute."""

    route: RouteLevel
    complexity: RouteLevel
    recommended_model_tier: ModelTier | None
    requires_llm: bool
    requires_rag: bool
    requires_investigation: bool
    requires_external_research: bool
    max_allowed_llm_calls: int
    budget_policy: BudgetPolicy
    routing_reason: str
    confidence: Literal["high", "medium", "low"] | None = None
    escalation_allowed: bool = False
    escalated: bool = False
    escalation_trigger: str | None = None

    def routing_metadata(self) -> RoutingMetadata:
        return RoutingMetadata(
            route=self.route,
            complexity=self.complexity,
            model_tier=self.recommended_model_tier,
            requires_investigation=self.requires_investigation,
            requires_external_research=self.requires_external_research,
            escalated=self.escalated,
            routing_reason=self.routing_reason,
            budget_policy=self.budget_policy,
        )


class RoutingMetadata(BaseModel):
    """Concise routing information safe to expose in an API response."""

    route: RouteLevel
    complexity: RouteLevel
    model_tier: ModelTier | None
    requires_investigation: bool
    requires_external_research: bool
    escalated: bool
    routing_reason: str
    budget_policy: BudgetPolicy


class EscalationRequest(BaseModel):
    trigger: Literal[
        "insufficient_evidence",
        "conflicting_evidence",
        "low_retrieval_quality",
        "deeper_research_requested",
        "ungrounded_answer",
        "previous_execution_failed",
    ]
    reason: str = Field(min_length=1)
