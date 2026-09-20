from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from prometheus.investigation import (
    InMemoryEvidencePool,
    InvestigationPlan,
    InvestigationPlannerInput,
    InvestigationPlanStatus,
    InvestigationTask,
    InvestigationTaskType,
    RuleBasedInvestigationPlanner,
)
from prometheus.rag.models import Evidence
from prometheus.routing import TaskRouter


def execution_plan(query: str = "Why did revenue decline?", **overrides):
    plan = TaskRouter().route(query)
    return plan.model_copy(update=overrides)


def valid_task(**overrides) -> InvestigationTask:
    values = {
        "task_id": "task-1",
        "type": InvestigationTaskType.DOCUMENT_RETRIEVAL,
        "question": "Retrieve revenue evidence",
    }
    values.update(overrides)
    return InvestigationTask(**values)


def valid_plan(**overrides) -> InvestigationPlan:
    values = {
        "investigation_id": "investigation-1",
        "original_query": "Why did revenue decline?",
        "objective": "Explain the revenue decline.",
        "tasks": [valid_task()],
        "max_tasks": 1,
        "planning_reason": "High-complexity causal request.",
    }
    values.update(overrides)
    return InvestigationPlan(**values)


def test_valid_task() -> None:
    task = valid_task(required_evidence_types=["document passage"], source_scope=["domain_docs"])
    assert task.task_id == "task-1"
    assert task.priority == 1
    assert task.max_results == 10


@pytest.mark.parametrize("field", ["task_id", "question"])
def test_task_empty_or_whitespace_text_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        valid_task(**{field: "   "})


def test_task_invalid_priority_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_task(priority=0)


def test_task_invalid_max_results_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_task(max_results=0)


def test_task_list_entries_and_duplicate_dependencies_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_task(required_evidence_types=[" "])
    with pytest.raises(ValidationError):
        valid_task(depends_on=["other", "other"])


def test_valid_plan() -> None:
    plan = valid_plan()
    assert plan.status is InvestigationPlanStatus.READY


@pytest.mark.parametrize("field", ["investigation_id", "original_query", "objective", "planning_reason"])
def test_plan_empty_or_whitespace_text_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        valid_plan(**{field: "  "})


def test_plan_task_count_limit_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_plan(tasks=[valid_task(), valid_task(task_id="task-2")], max_tasks=1)


def test_plan_duplicate_task_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_plan(tasks=[valid_task(), valid_task(task_id="task-1")], max_tasks=2)


def test_plan_unknown_dependency_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_plan(tasks=[valid_task(depends_on=["missing"])])


def test_plan_self_dependency_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_plan(tasks=[valid_task(depends_on=["task-1"])])


def test_plan_external_task_requires_flag() -> None:
    with pytest.raises(ValidationError):
        valid_plan(
            tasks=[valid_task(type=InvestigationTaskType.EXTERNAL_RESEARCH)],
            requires_external_research=False,
        )


def test_plan_external_task_allowed_when_enabled() -> None:
    plan = valid_plan(
        tasks=[valid_task(type=InvestigationTaskType.EXTERNAL_RESEARCH)],
        requires_external_research=True,
    )
    assert plan.requires_external_research is True


def test_ready_plan_without_tasks_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_plan(tasks=[], max_tasks=0)


def test_planner_input_trims_query_and_reuses_execution_plan() -> None:
    plan = execution_plan()
    request = InvestigationPlannerInput(query="  Why did revenue decline?  ", execution_plan=plan)

    assert request.query == "Why did revenue decline?"
    assert request.execution_plan is plan


@pytest.mark.parametrize("query", ["", "   "])
def test_planner_input_empty_query_rejected(query: str) -> None:
    with pytest.raises(ValidationError):
        InvestigationPlannerInput(query=query, execution_plan=execution_plan())


def test_planner_input_does_not_reject_non_investigation_plan() -> None:
    plan = execution_plan("What is revenue in Q3?")
    request = InvestigationPlannerInput(query="What is revenue in Q3?", execution_plan=plan)

    assert request.execution_plan.requires_investigation is False


def test_investigation_request_produces_bounded_plan() -> None:
    request = InvestigationPlannerInput(
        query="Why did revenue decline in Q3 and what external factors contributed?",
        execution_plan=execution_plan(
            "Why did revenue decline in Q3 and what external factors contributed?"
        ),
    )
    plan = RuleBasedInvestigationPlanner().create_plan(request)

    assert plan.status is InvestigationPlanStatus.READY
    assert plan.requires_external_research is True
    assert len(plan.tasks) <= plan.max_tasks
    assert [task.type for task in plan.tasks] == [
        InvestigationTaskType.DOCUMENT_RETRIEVAL,
        InvestigationTaskType.EVIDENCE_ANALYSIS,
        InvestigationTaskType.EXTERNAL_RESEARCH,
    ]


def test_non_investigation_request_returns_no_investigation_required() -> None:
    query = "What is the revenue in Q3?"
    request = InvestigationPlannerInput(query=query, execution_plan=execution_plan(query))
    plan = RuleBasedInvestigationPlanner().create_plan(request)

    assert plan.status is InvestigationPlanStatus.NO_INVESTIGATION_REQUIRED
    assert plan.tasks == []
    assert plan.max_tasks == 0


def test_planner_does_not_call_an_llm_or_provider() -> None:
    provider = Mock()
    provider.generate.side_effect = AssertionError("planner must not call a provider")
    request = InvestigationPlannerInput(
        query="Why did revenue decline?",
        execution_plan=execution_plan(),
    )

    plan = RuleBasedInvestigationPlanner().create_plan(request)

    assert plan.status is InvestigationPlanStatus.READY
    provider.generate.assert_not_called()


def test_planner_respects_max_tasks() -> None:
    query = "Why did revenue decline and what external factors contributed?"
    request = InvestigationPlannerInput(
        query=query,
        execution_plan=execution_plan(query, max_allowed_llm_calls=2),
    )

    plan = RuleBasedInvestigationPlanner().create_plan(request)

    assert plan.max_tasks == 2
    assert len(plan.tasks) == 2
    assert plan.requires_external_research is False


def test_in_memory_evidence_pool_adds_and_filters_by_task() -> None:
    pool = InMemoryEvidencePool()
    first = Evidence(evidence_id="e1", text="first")
    second = Evidence(evidence_id="e2", text="second")

    pool.add(first, task_id="task-1")
    pool.add(second, task_id="task-2")

    assert pool.get(task_id="task-1") == [first]
    assert pool.get(task_id="task-2") == [second]


def test_in_memory_evidence_pool_get_all_and_clear() -> None:
    pool = InMemoryEvidencePool()
    pool.add(Evidence(evidence_id="e1", text="first"), task_id="task-1")
    pool.add(Evidence(evidence_id="e2", text="second"), task_id="task-2")

    assert len(pool.get_all()) == 2
    assert pool.get() == pool.get_all()
    pool.clear()
    assert pool.get_all() == []


def test_in_memory_evidence_pool_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError):
        InMemoryEvidencePool().add(Evidence(text="evidence"), task_id="  ")
