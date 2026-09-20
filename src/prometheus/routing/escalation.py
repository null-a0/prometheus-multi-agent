"""Escalation interface for later evidence-quality execution stages."""
from __future__ import annotations

from prometheus.routing.models import EscalationRequest, ExecutionPlan
from prometheus.routing.router import TaskRouter


class EscalationManager:
    """Delegates plan escalation without implementing investigation behavior."""

    def __init__(self, router: TaskRouter) -> None:
        self.router = router

    def escalate(self, plan: ExecutionPlan, request: EscalationRequest) -> ExecutionPlan:
        return self.router.escalate(plan, request)
