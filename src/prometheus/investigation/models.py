"""Pydantic contracts for future investigation planning."""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prometheus.routing.models import ExecutionPlan


class InvestigationTaskType(str, Enum):
    DOCUMENT_RETRIEVAL = "document_retrieval"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    COMPARISON = "comparison"
    EXTERNAL_RESEARCH = "external_research"


class EvidenceGraphNodeType(str, Enum):
    EVIDENCE = "evidence"
    CLAIM = "claim"


class EvidenceRelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class ClaimOrigin(str, Enum):
    CALLER_SUPPLIED = "caller_supplied"
    USER_INPUT = "user_input"
    DOCUMENT_EXTRACTED = "document_extracted"
    MANUAL_REVIEW = "manual_review"
    MODEL_GENERATED = "model_generated"
    VERIFICATION_LAYER = "verification_layer"


def _require_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be non-empty and not whitespace-only")
    return value.strip()


def _require_list_entries(values: list[str]) -> list[str]:
    cleaned = [_require_text(value) for value in values]
    return cleaned


class InvestigationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    type: InvestigationTaskType
    question: str
    priority: int = Field(default=1, ge=1)
    required_evidence_types: list[str] = Field(default_factory=list)
    source_scope: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    max_results: int = Field(default=10, ge=1)

    _validate_task_id = field_validator("task_id")(_require_text)
    _validate_question = field_validator("question")( _require_text)
    _validate_lists = field_validator("required_evidence_types", "source_scope", "depends_on")(
        _require_list_entries
    )

    @field_validator("depends_on")
    @classmethod
    def validate_unique_dependencies(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("depends_on must not contain duplicate task IDs")
        return values


class InvestigationPlanStatus(str, Enum):
    READY = "ready"
    NO_INVESTIGATION_REQUIRED = "no_investigation_required"
    INVALID = "invalid"


class InvestigationExecutionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class InvestigationTaskExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class InvestigationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: InvestigationTaskExecutionStatus
    evidence_count: int = Field(default=0, ge=0)
    error: str | None = None


class InvestigationExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation_id: str
    status: InvestigationExecutionStatus
    task_results: list[InvestigationTaskResult] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    error: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    origin: ClaimOrigin = ClaimOrigin.CALLER_SUPPLIED
    origin_detail: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)

    _validate_claim_text = field_validator("claim_id", "text")(_require_text)

    @field_validator("origin_detail")
    @classmethod
    def validate_origin_detail(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value)

    @field_validator("source_evidence_ids")
    @classmethod
    def validate_source_evidence_ids(cls, values: list[str]) -> list[str]:
        cleaned = _require_list_entries(values)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_evidence_ids must not contain duplicate IDs")
        return cleaned


class EvidenceGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str | None = None
    source_id: str
    target_id: str
    relation: EvidenceRelationType

    _validate_edge_ids = field_validator("edge_id", "source_id", "target_id")(_require_text)

    @model_validator(mode="after")
    def validate_edge(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must be different")
        if self.edge_id is None:
            self.edge_id = _stable_edge_id(self.source_id, self.target_id, self.relation)
        return self


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation_id: str
    original_query: str
    objective: str
    tasks: list[InvestigationTask] = Field(default_factory=list)
    max_tasks: int = Field(ge=0)
    requires_external_research: bool = False
    status: InvestigationPlanStatus = InvestigationPlanStatus.READY
    planning_reason: str

    _validate_text = field_validator(
        "investigation_id", "original_query", "objective", "planning_reason"
    )(_require_text)

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if len(self.tasks) > self.max_tasks:
            raise ValueError("tasks cannot exceed max_tasks")

        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique within an investigation plan")

        known_ids = set(task_ids)
        for task in self.tasks:
            for dependency in task.depends_on:
                if dependency not in known_ids:
                    raise ValueError(f"unknown dependency task ID: {dependency}")
                if dependency == task.task_id:
                    raise ValueError("a task cannot depend on itself")

        _validate_dependency_dag(self.tasks)

        if not self.requires_external_research and any(
            task.type is InvestigationTaskType.EXTERNAL_RESEARCH for task in self.tasks
        ):
            raise ValueError("external research tasks require requires_external_research=True")

        if self.status is InvestigationPlanStatus.READY and not self.tasks:
            raise ValueError("READY plans must contain at least one task")
        if self.status is InvestigationPlanStatus.NO_INVESTIGATION_REQUIRED and self.tasks:
            raise ValueError("NO_INVESTIGATION_REQUIRED plans must not contain tasks")
        return self


class InvestigationPlannerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    execution_plan: ExecutionPlan

    _validate_query = field_validator("query")(_require_text)


def _validate_dependency_dag(tasks: list[InvestigationTask]) -> None:
    dependencies = {task.task_id: set(task.depends_on) for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("task dependencies must form a directed acyclic graph")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in dependencies:
        visit(task_id)


def _stable_edge_id(
    source_id: str,
    target_id: str,
    relation: EvidenceRelationType,
) -> str:
    payload = f"{source_id}\x1f{relation.value}\x1f{target_id}".encode()
    return f"edge-{hashlib.sha256(payload).hexdigest()[:16]}"
