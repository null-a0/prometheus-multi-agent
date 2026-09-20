from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from prometheus.investigation import (
    InMemoryEvidencePool,
    InvestigationExecutionResult,
    InvestigationExecutionStatus,
    InvestigationTask,
    InvestigationTaskExecutionStatus,
    InvestigationTaskResult,
    InvestigationTaskType,
    SequentialInvestigationExecutor,
)
from prometheus.rag.models import Evidence


def task(task_id: str, task_type: InvestigationTaskType, priority: int = 1, depends_on=None, max_results=10):
    return InvestigationTask(
        task_id=task_id,
        type=task_type,
        question=f"Question for {task_id}",
        priority=priority,
        depends_on=depends_on or [],
        max_results=max_results,
    )


def plan(tasks: list[InvestigationTask], **overrides):
    values = {
        "investigation_id": "investigation-1",
        "original_query": "Why did revenue decline?",
        "objective": "Explain the decline.",
        "tasks": tasks,
        "max_tasks": len(tasks),
        "planning_reason": "Test plan",
    }
    values.update(overrides)
    from prometheus.investigation.models import InvestigationPlan

    return InvestigationPlan(**values)


def test_valid_execution_result_and_task_result() -> None:
    task_result = InvestigationTaskResult(
        task_id="task-1", status=InvestigationTaskExecutionStatus.COMPLETED, evidence_count=1
    )
    result = InvestigationExecutionResult(
        investigation_id="investigation-1",
        status=InvestigationExecutionStatus.COMPLETED,
        task_results=[task_result],
        evidence_count=1,
    )

    assert result.task_results[0].evidence_count == 1


@pytest.mark.parametrize("model", [InvestigationTaskResult, InvestigationExecutionResult])
def test_negative_evidence_count_rejected(model) -> None:
    values = {"task_id": "task-1", "status": InvestigationTaskExecutionStatus.COMPLETED} if model is InvestigationTaskResult else {
        "investigation_id": "investigation-1",
        "status": InvestigationExecutionStatus.COMPLETED,
        "task_results": [],
    }
    with pytest.raises(ValidationError):
        model(**values, evidence_count=-1)


def test_invalid_task_status_rejected() -> None:
    with pytest.raises(ValidationError):
        InvestigationTaskResult(task_id="task-1", status="unknown")


def test_dependencies_execute_before_dependents_and_priority_orders_ready_tasks() -> None:
    order: list[str] = []

    def retrieve(question: str, max_results: int, source_scope):
        order.append(question.split()[-1])
        return []

    tasks = [
        task("task-b", InvestigationTaskType.DOCUMENT_RETRIEVAL, priority=1, depends_on=["task-a"]),
        task("task-a", InvestigationTaskType.DOCUMENT_RETRIEVAL, priority=2),
        task("task-c", InvestigationTaskType.DOCUMENT_RETRIEVAL, priority=1),
    ]
    result = SequentialInvestigationExecutor(document_retriever=retrieve).execute(plan(tasks), InMemoryEvidencePool())

    assert result.status is InvestigationExecutionStatus.COMPLETED
    assert order.index("task-a") < order.index("task-b")
    assert order.index("task-c") < order.index("task-a")


def test_tie_breaking_uses_task_id() -> None:
    order: list[str] = []

    def retrieve(question: str, max_results: int, source_scope):
        order.append(question.split()[-1])
        return []

    tasks = [
        task("task-z", InvestigationTaskType.DOCUMENT_RETRIEVAL),
        task("task-a", InvestigationTaskType.DOCUMENT_RETRIEVAL),
    ]
    SequentialInvestigationExecutor(document_retriever=retrieve).execute(plan(tasks), InMemoryEvidencePool())

    assert order == ["task-a", "task-z"]


def test_document_retrieval_dispatches_and_respects_max_results() -> None:
    retriever = Mock(return_value=[
        {"id": "chunk-1", "text": "Revenue evidence", "metadata": {"document_id": "doc-1"}, "score": 0.8}
    ])
    pool = InMemoryEvidencePool()
    result = SequentialInvestigationExecutor(document_retriever=retriever).execute(
        plan([task("retrieve", InvestigationTaskType.DOCUMENT_RETRIEVAL, max_results=3)]), pool
    )

    retriever.assert_called_once_with("Question for retrieve", 3, [])
    assert result.status is InvestigationExecutionStatus.COMPLETED
    assert result.evidence_count == 1
    assert pool.get(task_id="retrieve")[0].document_id == "doc-1"


def test_external_research_reuses_injected_external_abstraction() -> None:
    researcher = Mock(
        return_value=[
            SimpleNamespace(
                source_id="W1",
                source="openalex",
                title="Market evidence",
                abstract="External market evidence.",
                authors=["Author"],
                year=2024,
                doi=None,
                url="https://example.test/W1",
            )
        ]
    )
    pool = InMemoryEvidencePool()
    result = SequentialInvestigationExecutor(external_researcher=researcher).execute(
        plan(
            [task("external", InvestigationTaskType.EXTERNAL_RESEARCH, max_results=2)],
            requires_external_research=True,
        ),
        pool,
    )

    researcher.assert_called_once_with("Question for external", 2)
    assert result.evidence_count == 1
    assert pool.get(task_id="external")[0].retrieval_method == "external_research"


def test_analysis_and_comparison_consume_existing_pool_without_fabricating_evidence() -> None:
    pool = InMemoryEvidencePool()
    existing = Evidence(evidence_id="existing", text="Existing evidence")
    pool.add(existing, task_id="source")
    tasks = [
        task("source", InvestigationTaskType.EVIDENCE_ANALYSIS),
        task("compare", InvestigationTaskType.COMPARISON, depends_on=["source"]),
    ]

    result = SequentialInvestigationExecutor().execute(plan(tasks), pool)

    assert result.status is InvestigationExecutionStatus.COMPLETED
    assert result.evidence_count == 1
    assert pool.get_all() == [existing]


def test_failed_task_does_not_crash_independent_task_and_dependent_is_skipped() -> None:
    calls: list[str] = []

    def retrieve(question: str, max_results: int, source_scope):
        task_id = question.split()[-1]
        calls.append(task_id)
        if task_id == "failed":
            raise RuntimeError("retrieval unavailable")
        return []

    tasks = [
        task("failed", InvestigationTaskType.DOCUMENT_RETRIEVAL),
        task("dependent", InvestigationTaskType.DOCUMENT_RETRIEVAL, depends_on=["failed"]),
        task("independent", InvestigationTaskType.DOCUMENT_RETRIEVAL),
    ]
    result = SequentialInvestigationExecutor(document_retriever=retrieve).execute(plan(tasks), InMemoryEvidencePool())
    statuses = {item.task_id: item.status for item in result.task_results}

    assert statuses["failed"] is InvestigationTaskExecutionStatus.FAILED
    assert statuses["dependent"] is InvestigationTaskExecutionStatus.SKIPPED
    assert statuses["independent"] is InvestigationTaskExecutionStatus.COMPLETED
    assert "dependent" not in calls
    assert result.status is InvestigationExecutionStatus.PARTIAL


def test_all_failed_tasks_produce_failed_result() -> None:
    def fail(_question: str, _max_results: int, _source_scope):
        raise RuntimeError("unavailable")

    result = SequentialInvestigationExecutor(document_retriever=fail).execute(
        plan([task("one", InvestigationTaskType.DOCUMENT_RETRIEVAL)]), InMemoryEvidencePool()
    )

    assert result.status is InvestigationExecutionStatus.FAILED
    assert result.error is not None


def test_cyclic_plan_is_rejected_defensively() -> None:
    from prometheus.investigation.models import InvestigationPlan

    cyclic = InvestigationPlan.model_construct(
        investigation_id="cycle",
        original_query="cycle",
        objective="cycle",
        tasks=[
            task("a", InvestigationTaskType.DOCUMENT_RETRIEVAL, depends_on=["b"]),
            task("b", InvestigationTaskType.DOCUMENT_RETRIEVAL, depends_on=["a"]),
        ],
        max_tasks=2,
        requires_external_research=False,
        planning_reason="test",
    )
    with pytest.raises(ValueError, match="acyclic"):
        SequentialInvestigationExecutor().execute(cyclic, InMemoryEvidencePool())


def test_duplicate_evidence_ids_are_deduplicated_but_task_associations_remain() -> None:
    pool = InMemoryEvidencePool()
    evidence = Evidence(evidence_id="same", text="same evidence")
    pool.add(evidence, task_id="task-a")
    pool.add(evidence, task_id="task-b")

    assert pool.get_all() == [evidence]
    assert pool.get(task_id="task-a") == [evidence]
    assert pool.get(task_id="task-b") == [evidence]


def test_clear_only_clears_this_pool() -> None:
    first = InMemoryEvidencePool()
    second = InMemoryEvidencePool()
    evidence = Evidence(evidence_id="e", text="evidence")
    first.add(evidence, task_id="task")
    second.add(evidence, task_id="task")

    first.clear()
    assert first.get_all() == []
    assert second.get_all() == [evidence]


def test_executor_source_has_no_provider_or_agent_framework_imports() -> None:
    from pathlib import Path

    source = Path("src/prometheus/investigation/executor.py").read_text(encoding="utf-8")
    assert "openai" not in source.lower()
    assert "openrouter" not in source.lower()
    assert "langgraph" not in source.lower()
    assert "ModelGateway" not in source
