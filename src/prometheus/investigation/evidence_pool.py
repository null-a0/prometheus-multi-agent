"""Evidence storage contracts for future task execution."""
from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from prometheus.rag.models import Evidence


class EvidencePool(Protocol):
    def add(self, evidence: Evidence, *, task_id: str) -> None:
        ...

    def get(self, *, task_id: str | None = None) -> list[Evidence]:
        ...

    def get_all(self) -> list[Evidence]:
        ...

    def clear(self) -> None:
        ...


class InMemoryEvidencePool:
    """Non-persistent evidence pool representing one investigation."""

    def __init__(self) -> None:
        self._evidence_by_id: dict[str, Evidence] = {}
        self._task_to_evidence_ids: dict[str, list[str]] = defaultdict(list)

    def add(self, evidence: Evidence, *, task_id: str) -> None:
        clean_task_id = task_id.strip()
        if not clean_task_id:
            raise ValueError("task_id must be non-empty and not whitespace-only")
        evidence_id = evidence.evidence_id
        if evidence_id is None:
            evidence_id = f"anonymous:{id(evidence)}"
        if evidence_id not in self._evidence_by_id:
            self._evidence_by_id[evidence_id] = evidence
        if evidence_id not in self._task_to_evidence_ids[clean_task_id]:
            self._task_to_evidence_ids[clean_task_id].append(evidence_id)

    def get(self, *, task_id: str | None = None) -> list[Evidence]:
        if task_id is None:
            return self.get_all()
        evidence_ids = self._task_to_evidence_ids.get(task_id, [])
        return [self._evidence_by_id[evidence_id] for evidence_id in evidence_ids]

    def get_all(self) -> list[Evidence]:
        return list(self._evidence_by_id.values())

    def clear(self) -> None:
        self._evidence_by_id.clear()
        self._task_to_evidence_ids.clear()
