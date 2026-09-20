"""Fail-closed deterministic verification traceability assessment."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prometheus.investigation.findings import Finding, VerificationStatus
from prometheus.investigation.graph import EvidenceGraph
from prometheus.investigation.models import Claim, EvidenceGraphEdge, EvidenceRelationType
from prometheus.rag.models import Evidence


class VerificationAssessmentError(ValueError):
    """Raised when a Finding cannot be traced through an EvidenceGraph."""


class VerificationAssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    finding: Finding
    evidence_graph: EvidenceGraph


class VerificationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    finding_id: str
    claim_id: str
    status: VerificationStatus
    rationale: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    supporting_edge_ids: list[str] = Field(default_factory=list)
    contradicting_edge_ids: list[str] = Field(default_factory=list)
    unresolved_reasons: list[str] = Field(default_factory=list)

    @field_validator(
        "assessment_id",
        "finding_id",
        "claim_id",
        "rationale",
        mode="before",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be non-empty and not whitespace-only")
        return value.strip()


class VerificationAssessor(Protocol):
    def assess(self, request: VerificationAssessmentInput) -> VerificationAssessment:
        ...


class DeterministicVerificationAssessor:
    """Validate Finding status and expose only graph-traceable relationships."""

    def assess(self, request: VerificationAssessmentInput) -> VerificationAssessment:
        _validate_finding(request.finding)
        _validate_graph(request.evidence_graph)
        finding = request.finding
        graph = request.evidence_graph

        _validate_claim(graph, finding.claim_id)
        _validate_evidence_references(graph, finding.supporting_evidence_ids, "supporting")
        _validate_evidence_references(graph, finding.contradicting_evidence_ids, "contradicting")

        supporting = set(finding.supporting_evidence_ids)
        contradicting = set(finding.contradicting_evidence_ids)
        if supporting.intersection(contradicting):
            raise VerificationAssessmentError(
                "An evidence ID cannot support and contradict the same Finding."
            )

        if finding.verification_status is VerificationStatus.UNVERIFIED:
            if supporting or contradicting or finding.unresolved_reasons:
                raise VerificationAssessmentError(
                    "UNVERIFIED findings must not contain evidence or unresolved reasons."
                )
            return self._assessment(
                finding,
                rationale="The finding has not been assessed.",
            )

        if finding.verification_status is VerificationStatus.SUPPORTED:
            if not supporting:
                raise VerificationAssessmentError("SUPPORTED findings require supporting evidence.")
            if contradicting:
                raise VerificationAssessmentError(
                    "SUPPORTED findings cannot contain contradicting evidence."
                )
            support_edges = self._trace_evidence(
                graph, supporting, finding.claim_id, EvidenceRelationType.SUPPORTS
            )
            return self._assessment(
                finding,
                rationale=(
                    "The finding is supported by evidence with corresponding SUPPORTS "
                    "relationships in the evidence graph."
                ),
                supporting_edges=support_edges,
            )

        if finding.verification_status is VerificationStatus.CONFLICTING:
            if not supporting or not contradicting:
                raise VerificationAssessmentError(
                    "CONFLICTING findings require supporting and contradicting evidence."
                )
            if not finding.unresolved_reasons:
                raise VerificationAssessmentError(
                    "CONFLICTING findings require unresolved reasons."
                )
            support_edges = self._trace_evidence(
                graph, supporting, finding.claim_id, EvidenceRelationType.SUPPORTS
            )
            contradict_edges = self._trace_evidence(
                graph, contradicting, finding.claim_id, EvidenceRelationType.CONTRADICTS
            )
            return self._assessment(
                finding,
                rationale=(
                    "The finding contains both supporting and contradicting evidence "
                    "relationships for the claim; the conflict remains unresolved."
                ),
                supporting_edges=support_edges,
                contradicting_edges=contradict_edges,
            )

        if finding.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE:
            support_edges = self._trace_evidence(
                graph, supporting, finding.claim_id, EvidenceRelationType.SUPPORTS
            )
            contradict_edges = self._trace_evidence(
                graph, contradicting, finding.claim_id, EvidenceRelationType.CONTRADICTS
            )
            if not finding.unresolved_reasons:
                raise VerificationAssessmentError(
                    "INSUFFICIENT_EVIDENCE findings require unresolved reasons."
                )
            return self._assessment(
                finding,
                rationale=(
                    "The finding records that the currently available evidence is "
                    "insufficient for assessment."
                ),
                supporting_edges=support_edges,
                contradicting_edges=contradict_edges,
            )

        raise VerificationAssessmentError("Unsupported verification status.")

    def _trace_evidence(
        self,
        graph: EvidenceGraph,
        evidence_ids: set[str],
        claim_id: str,
        relation: EvidenceRelationType,
    ) -> list[EvidenceGraphEdge]:
        edges = [
            edge
            for edge in graph.get_relations(relation=relation)
            if edge.source_id in evidence_ids
            and edge.target_id == claim_id
            and edge.relation is relation
            and graph.get_evidence(edge.source_id) is not None
        ]
        missing = evidence_ids.difference({edge.source_id for edge in edges})
        if missing:
            relation_name = relation.value.upper()
            raise VerificationAssessmentError(
                f"Missing {relation_name} relationship for evidence: {', '.join(sorted(missing))}"
            )
        return sorted(edges, key=lambda edge: _edge_id(edge))

    def _assessment(
        self,
        finding: Finding,
        *,
        rationale: str,
        supporting_edges: list[EvidenceGraphEdge] | None = None,
        contradicting_edges: list[EvidenceGraphEdge] | None = None,
    ) -> VerificationAssessment:
        supporting_edges = supporting_edges or []
        contradicting_edges = contradicting_edges or []
        supporting_edge_ids = [_edge_id(edge) for edge in supporting_edges]
        contradicting_edge_ids = [_edge_id(edge) for edge in contradicting_edges]
        return VerificationAssessment(
            assessment_id=_assessment_id(
                finding=finding,
                supporting_edge_ids=supporting_edge_ids,
                contradicting_edge_ids=contradicting_edge_ids,
            ),
            finding_id=finding.finding_id,
            claim_id=finding.claim_id,
            status=finding.verification_status,
            rationale=rationale,
            supporting_evidence_ids=list(finding.supporting_evidence_ids),
            contradicting_evidence_ids=list(finding.contradicting_evidence_ids),
            supporting_edge_ids=supporting_edge_ids,
            contradicting_edge_ids=contradicting_edge_ids,
            unresolved_reasons=list(finding.unresolved_reasons),
        )


def _validate_finding(finding: Finding) -> None:
    if not isinstance(finding, Finding):
        raise VerificationAssessmentError("Finding is required.")
    try:
        Finding.model_validate(finding.model_dump())
    except Exception as exc:
        raise VerificationAssessmentError("Finding is invalid.") from exc


def _validate_graph(graph: EvidenceGraph) -> None:
    required_methods = (
        "get_claim",
        "get_evidence",
        "get_relations",
    )
    if graph is None or any(not callable(getattr(graph, method, None)) for method in required_methods):
        raise VerificationAssessmentError("EvidenceGraph is unavailable or invalid.")


def _validate_claim(graph: EvidenceGraph, claim_id: str) -> None:
    claim = graph.get_claim(claim_id)
    if claim is None:
        raise VerificationAssessmentError(
            "Claim referenced by Finding does not exist in EvidenceGraph."
        )
    if not isinstance(claim, Claim):
        raise VerificationAssessmentError("Finding claim_id does not reference a CLAIM node.")


def _validate_evidence_references(
    graph: EvidenceGraph,
    evidence_ids: list[str],
    role: str,
) -> None:
    for evidence_id in evidence_ids:
        evidence = graph.get_evidence(evidence_id)
        if evidence is None:
            raise VerificationAssessmentError(
                f"{role.title()} evidence referenced by Finding does not exist in EvidenceGraph."
            )
        if not isinstance(evidence, Evidence):
            raise VerificationAssessmentError(
                f"{role.title()} evidence ID does not reference an EVIDENCE node."
            )


def _assessment_id(
    *,
    finding: Finding,
    supporting_edge_ids: list[str],
    contradicting_edge_ids: list[str],
) -> str:
    payload = {
        "finding_id": finding.finding_id,
        "claim_id": finding.claim_id,
        "status": finding.verification_status.value,
        "supporting_evidence_ids": list(finding.supporting_evidence_ids),
        "contradicting_evidence_ids": list(finding.contradicting_evidence_ids),
        "supporting_edge_ids": supporting_edge_ids,
        "contradicting_edge_ids": contradicting_edge_ids,
        "unresolved_reasons": list(finding.unresolved_reasons),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"assessment-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _edge_id(edge: EvidenceGraphEdge) -> str:
    if not edge.edge_id:
        raise VerificationAssessmentError("Graph edge is missing a stable edge_id.")
    return edge.edge_id
