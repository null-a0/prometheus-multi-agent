"""Lightweight in-memory evidence/claim graph for one investigation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prometheus.investigation.evidence_pool import EvidencePool
from prometheus.investigation.models import (
    Claim,
    EvidenceGraphEdge,
    EvidenceGraphNodeType,
    EvidenceRelationType,
)
from prometheus.rag.models import Evidence


@runtime_checkable
class EvidenceGraph(Protocol):
    def add_evidence(self, evidence: Evidence) -> None:
        ...

    def add_claim(self, claim: Claim) -> None:
        ...

    def add_relation(self, edge: EvidenceGraphEdge) -> None:
        ...

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        ...

    def get_claim(self, claim_id: str) -> Claim | None:
        ...

    def get_claims_for_evidence(self, evidence_id: str) -> list[Claim]:
        ...

    def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        ...

    def get_relations(self, *, relation: EvidenceRelationType | None = None) -> list[EvidenceGraphEdge]:
        ...


class InMemoryEvidenceGraph:
    """Deterministic graph state scoped to one investigation."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._claims: dict[str, Claim] = {}
        self._relations: dict[tuple[str, str, EvidenceRelationType], EvidenceGraphEdge] = {}

    def add_evidence(self, evidence: Evidence) -> None:
        evidence_id = _require_id(evidence.evidence_id, "evidence_id")
        if evidence_id in self._claims:
            raise ValueError(f"node ID already belongs to a claim: {evidence_id}")
        self._evidence.setdefault(evidence_id, evidence)

    def add_claim(self, claim: Claim) -> None:
        claim_id = _require_id(claim.claim_id, "claim_id")
        if claim_id in self._evidence:
            raise ValueError(f"node ID already belongs to evidence: {claim_id}")
        missing = [evidence_id for evidence_id in claim.source_evidence_ids if evidence_id not in self._evidence]
        if missing:
            raise ValueError(f"claim references missing evidence IDs: {', '.join(sorted(missing))}")
        self._claims.setdefault(claim_id, claim)

    def add_relation(self, edge: EvidenceGraphEdge) -> None:
        source_type = self._node_type(edge.source_id)
        target_type = self._node_type(edge.target_id)
        if edge.relation is EvidenceRelationType.SUPPORTS:
            if source_type is not EvidenceGraphNodeType.EVIDENCE or target_type is not EvidenceGraphNodeType.CLAIM:
                raise ValueError("SUPPORTS relations must connect Evidence to Claim")
        elif edge.relation is EvidenceRelationType.CONTRADICTS:
            valid_claim_contradiction = (
                source_type is EvidenceGraphNodeType.CLAIM
                and target_type is EvidenceGraphNodeType.CLAIM
            )
            valid_evidence_contradiction = (
                source_type is EvidenceGraphNodeType.EVIDENCE
                and target_type is EvidenceGraphNodeType.CLAIM
            )
            if not (valid_claim_contradiction or valid_evidence_contradiction):
                raise ValueError("CONTRADICTS relations must connect Evidence or Claim to Claim")
        key = (edge.source_id, edge.target_id, edge.relation)
        duplicate_edge_id = next(
            (
                existing
                for existing_key, existing in self._relations.items()
                if existing_key != key and existing.edge_id == edge.edge_id
            ),
            None,
        )
        if duplicate_edge_id is not None:
            raise ValueError(f"edge ID already belongs to another relation: {edge.edge_id}")
        self._relations.setdefault(key, edge)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self._evidence.get(evidence_id)

    def get_claim(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def get_claims_for_evidence(self, evidence_id: str) -> list[Claim]:
        claim_ids = sorted(
            edge.target_id
            for edge in self.get_relations(relation=EvidenceRelationType.SUPPORTS)
            if edge.source_id == evidence_id
        )
        return [self._claims[claim_id] for claim_id in claim_ids]

    def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        evidence_ids = sorted(
            edge.source_id
            for edge in self.get_relations(relation=EvidenceRelationType.SUPPORTS)
            if edge.target_id == claim_id
        )
        return [self._evidence[evidence_id] for evidence_id in evidence_ids]

    def get_relations(self, *, relation: EvidenceRelationType | None = None) -> list[EvidenceGraphEdge]:
        edges = self._relations.values()
        if relation is not None:
            edges = (edge for edge in edges if edge.relation is relation)
        return sorted(edges, key=lambda edge: edge.edge_id or "")

    def _node_type(self, node_id: str) -> EvidenceGraphNodeType:
        if node_id in self._evidence:
            return EvidenceGraphNodeType.EVIDENCE
        if node_id in self._claims:
            return EvidenceGraphNodeType.CLAIM
        raise ValueError(f"graph node does not exist: {node_id}")


def build_evidence_graph(evidence_pool: EvidencePool, claims: Sequence[Claim]) -> EvidenceGraph:
    graph = InMemoryEvidenceGraph()
    for evidence in evidence_pool.get_all():
        graph.add_evidence(evidence)
    for claim in claims:
        graph.add_claim(claim)
        for evidence_id in claim.source_evidence_ids:
            graph.add_relation(
                EvidenceGraphEdge(
                    source_id=evidence_id,
                    target_id=claim.claim_id,
                    relation=EvidenceRelationType.SUPPORTS,
                )
            )
    return graph


def _require_id(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value
