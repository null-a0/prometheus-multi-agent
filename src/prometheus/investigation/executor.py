"""Deterministic sequential execution of validated investigation plans."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from prometheus.investigation.evidence_pool import EvidencePool
from prometheus.investigation.models import (
    InvestigationExecutionResult,
    InvestigationExecutionStatus,
    InvestigationPlan,
    InvestigationTask,
    InvestigationTaskExecutionStatus,
    InvestigationTaskResult,
    InvestigationTaskType,
)
from prometheus.rag.evidence import build_evidence
from prometheus.retrieval import hybrid_search
from prometheus.retrieval.sources.openalex_client import search_openalex

DocumentRetriever = Callable[[str, int, Sequence[str]], list[Mapping[str, Any]]]
ExternalResearcher = Callable[[str, int], list[Any]]


class InvestigationExecutor(Protocol):
    def execute(
        self,
        plan: InvestigationPlan,
        evidence_pool: EvidencePool,
    ) -> InvestigationExecutionResult:
        ...


class SequentialInvestigationExecutor:
    """Execute plan tasks once, sequentially, and store produced evidence."""

    def __init__(
        self,
        document_retriever: DocumentRetriever | None = None,
        external_researcher: ExternalResearcher | None = None,
    ) -> None:
        self.document_retriever = document_retriever or _retrieve_documents
        self.external_researcher = external_researcher or search_openalex

    def execute(
        self,
        plan: InvestigationPlan,
        evidence_pool: EvidencePool,
    ) -> InvestigationExecutionResult:
        ordered_tasks = _execution_order(plan.tasks)
        task_results: list[InvestigationTaskResult] = []
        completed: set[str] = set()
        failed_or_skipped: set[str] = set()

        for task in ordered_tasks:
            blocked_by = [dependency for dependency in task.depends_on if dependency in failed_or_skipped]
            if blocked_by:
                task_results.append(
                    InvestigationTaskResult(
                        task_id=task.task_id,
                        status=InvestigationTaskExecutionStatus.SKIPPED,
                        evidence_count=0,
                        error=f"Skipped because dependency failed or was skipped: {blocked_by[0]}",
                    )
                )
                failed_or_skipped.add(task.task_id)
                continue

            try:
                evidence_count = self._execute_task(task, evidence_pool)
            except Exception as exc:  # noqa: BLE001
                task_results.append(
                    InvestigationTaskResult(
                        task_id=task.task_id,
                        status=InvestigationTaskExecutionStatus.FAILED,
                        evidence_count=0,
                        error=_safe_error(exc),
                    )
                )
                failed_or_skipped.add(task.task_id)
                continue

            task_results.append(
                InvestigationTaskResult(
                    task_id=task.task_id,
                    status=InvestigationTaskExecutionStatus.COMPLETED,
                    evidence_count=evidence_count,
                )
            )
            completed.add(task.task_id)

        failures = [
            result
            for result in task_results
            if result.status is not InvestigationTaskExecutionStatus.COMPLETED
        ]
        if not failures:
            status = InvestigationExecutionStatus.COMPLETED
            error = None
        elif completed:
            status = InvestigationExecutionStatus.PARTIAL
            error = "One or more investigation tasks failed or were skipped."
        else:
            status = InvestigationExecutionStatus.FAILED
            error = "All investigation tasks failed or were skipped."

        return InvestigationExecutionResult(
            investigation_id=plan.investigation_id,
            status=status,
            task_results=task_results,
            evidence_count=len(evidence_pool.get_all()),
            error=error,
        )

    def _execute_task(self, task: InvestigationTask, evidence_pool: EvidencePool) -> int:
        if task.type is InvestigationTaskType.DOCUMENT_RETRIEVAL:
            results = self.document_retriever(task.question, task.max_results, task.source_scope)
            evidence = build_evidence(results, retrieval_method="hybrid")
            for item in evidence:
                evidence_pool.add(item, task_id=task.task_id)
            return len(evidence_pool.get(task_id=task.task_id))

        if task.type is InvestigationTaskType.EXTERNAL_RESEARCH:
            records = self.external_researcher(task.question, task.max_results)
            evidence = [_paper_to_evidence(record) for record in records]
            evidence = [item for item in evidence if item is not None]
            for item in evidence:
                evidence_pool.add(item, task_id=task.task_id)
            return len(evidence_pool.get(task_id=task.task_id))

        if task.type in (InvestigationTaskType.EVIDENCE_ANALYSIS, InvestigationTaskType.COMPARISON):
            return len(evidence_pool.get())

        raise ValueError(f"Unsupported investigation task type: {task.type}")


def _retrieve_documents(question: str, max_results: int, _source_scope: Sequence[str]) -> list[Mapping[str, Any]]:
    return hybrid_search.hybrid_search(query=question, top_k=max_results)


def _paper_to_evidence(record: Any):
    abstract = getattr(record, "abstract", None)
    source_id = getattr(record, "source_id", None)
    if not abstract or not source_id:
        return None
    from prometheus.rag.models import Evidence

    return Evidence(
        evidence_id=str(source_id),
        document_id=str(source_id),
        document_name=getattr(record, "title", None),
        chunk_id=str(source_id),
        text=abstract,
        source_type=getattr(record, "source", None),
        retrieval_method="external_research",
        metadata={
            "authors": getattr(record, "authors", []),
            "year": getattr(record, "year", None),
            "doi": getattr(record, "doi", None),
            "url": getattr(record, "url", None),
        },
    )


def _execution_order(tasks: list[InvestigationTask]) -> list[InvestigationTask]:
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("task IDs must be unique")
    remaining = {task.task_id: set(task.depends_on) for task in tasks}
    ordered: list[InvestigationTask] = []
    while remaining:
        ready = [task_id for task_id, dependencies in remaining.items() if not dependencies]
        if not ready:
            raise ValueError("task dependencies must form a directed acyclic graph")
        ready.sort(key=lambda task_id: (by_id[task_id].priority, task_id))
        for task_id in ready:
            ordered.append(by_id[task_id])
            remaining.pop(task_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return ordered


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else type(exc).__name__
