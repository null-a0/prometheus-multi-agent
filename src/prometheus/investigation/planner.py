"""Minimal rule-based investigation planner contract."""
from __future__ import annotations

import re
from typing import Protocol
from uuid import uuid4

from prometheus.investigation.models import (
    InvestigationPlan,
    InvestigationPlannerInput,
    InvestigationPlanStatus,
    InvestigationTask,
    InvestigationTaskType,
)


class InvestigationPlanner(Protocol):
    def create_plan(self, request: InvestigationPlannerInput) -> InvestigationPlan:
        ...


class RuleBasedInvestigationPlanner:
    """Create bounded task definitions without retrieval or model calls."""

    def __init__(self, default_max_tasks: int = 3) -> None:
        if default_max_tasks < 1:
            raise ValueError("default_max_tasks must be at least 1")
        self.default_max_tasks = default_max_tasks

    def create_plan(self, request: InvestigationPlannerInput) -> InvestigationPlan:
        query = request.query
        execution_plan = request.execution_plan
        investigation_id = f"investigation-{uuid4()}"

        if not execution_plan.requires_investigation:
            return InvestigationPlan(
                investigation_id=investigation_id,
                original_query=query,
                objective=query,
                tasks=[],
                max_tasks=0,
                requires_external_research=False,
                status=InvestigationPlanStatus.NO_INVESTIGATION_REQUIRED,
                planning_reason="The execution plan does not require an investigation.",
            )

        max_tasks = min(self.default_max_tasks, max(execution_plan.max_allowed_llm_calls, 1))
        tasks = [
            InvestigationTask(
                task_id="retrieve_internal_evidence",
                type=InvestigationTaskType.DOCUMENT_RETRIEVAL,
                question=f"Retrieve internal evidence relevant to: {query}",
                priority=1,
                required_evidence_types=["document passage"],
                source_scope=["domain_docs", "paper_corpus"],
                max_results=10,
            ),
            InvestigationTask(
                task_id="analyze_internal_evidence",
                type=InvestigationTaskType.EVIDENCE_ANALYSIS,
                question=f"Analyze documented explanations for: {query}",
                priority=2,
                required_evidence_types=["supporting passage"],
                depends_on=["retrieve_internal_evidence"],
                max_results=10,
            ),
        ]

        requires_external = execution_plan.requires_external_research
        if requires_external:
            tasks.append(
                InvestigationTask(
                    task_id="research_external_factors",
                    type=InvestigationTaskType.EXTERNAL_RESEARCH,
                    question=f"Identify external factors relevant to: {query}",
                    priority=3,
                    required_evidence_types=["external source"],
                    source_scope=["external_research"],
                    depends_on=["analyze_internal_evidence"],
                    max_results=10,
                )
            )

        tasks = tasks[:max_tasks]
        if requires_external and not any(task.type is InvestigationTaskType.EXTERNAL_RESEARCH for task in tasks):
            requires_external = False

        return InvestigationPlan(
            investigation_id=investigation_id,
            original_query=query,
            objective=_objective(query),
            tasks=tasks,
            max_tasks=max_tasks,
            requires_external_research=requires_external,
            status=InvestigationPlanStatus.READY,
            planning_reason="Created a bounded rule-based investigation plan from the execution route.",
        )


def _objective(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()
