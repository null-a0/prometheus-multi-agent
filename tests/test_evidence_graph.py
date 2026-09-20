from __future__ import annotations

import pytest
from pydantic import ValidationError

from prometheus.investigation import (
    Claim,
    ClaimOrigin,
    EvidenceGraphEdge,
    EvidenceRelationType,
    InMemoryEvidenceGraph,
    InMemoryEvidencePool,
    build_evidence_graph,
)
from prometheus.rag.models import Evidence


def evidence(evidence_id: str) -> Evidence:
    return Evidence(evidence_id=evidence_id, text=f"Evidence {evidence_id}")


def test_valid_claim_and_edge() -> None:
    claim = Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=["ev-1"])
    edge = EvidenceGraphEdge(
        source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS
    )

    assert claim.source_evidence_ids == ["ev-1"]
    assert claim.origin is ClaimOrigin.CALLER_SUPPLIED
    assert edge.edge_id
    assert edge.relation is EvidenceRelationType.SUPPORTS


def test_claim_provenance_is_explicit_and_validated() -> None:
    claim = Claim(
        claim_id="claim-1",
        text="Revenue increased.",
        origin=ClaimOrigin.DOCUMENT_EXTRACTED,
        origin_detail="extracted from annual report",
        source_evidence_ids=["ev-1"],
    )

    assert claim.origin is ClaimOrigin.DOCUMENT_EXTRACTED
    assert claim.origin_detail == "extracted from annual report"

    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim-1",
            text="Revenue increased.",
            origin_detail=" ",
        )


@pytest.mark.parametrize("field", ["claim_id", "text"])
def test_empty_claim_fields_rejected(field: str) -> None:
    values = {"claim_id": "claim-1", "text": "valid", "source_evidence_ids": []}
    values[field] = " "
    with pytest.raises(ValidationError):
        Claim(**values)


def test_duplicate_claim_evidence_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=["ev-1", "ev-1"])


def test_empty_claim_evidence_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=[" "])


def test_empty_edge_ids_and_self_relation_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceGraphEdge(source_id=" ", target_id="claim", relation=EvidenceRelationType.SUPPORTS)
    with pytest.raises(ValidationError):
        EvidenceGraphEdge(source_id="same", target_id="same", relation=EvidenceRelationType.CONTRADICTS)
    with pytest.raises(ValidationError):
        EvidenceGraphEdge(edge_id=" ", source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS)


def test_edge_id_is_stable_and_can_be_explicit() -> None:
    first = EvidenceGraphEdge(source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS)
    second = EvidenceGraphEdge(source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS)
    explicit = EvidenceGraphEdge(
        edge_id="edge-custom",
        source_id="ev-1",
        target_id="claim-1",
        relation=EvidenceRelationType.SUPPORTS,
    )

    assert first.edge_id == second.edge_id
    assert first.edge_id.startswith("edge-")
    assert explicit.edge_id == "edge-custom"


def test_graph_adds_and_retrieves_nodes_and_relationships() -> None:
    graph = InMemoryEvidenceGraph()
    ev = evidence("ev-1")
    claim = Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=["ev-1"])
    graph.add_evidence(ev)
    graph.add_claim(claim)
    graph.add_relation(EvidenceGraphEdge(source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS))

    assert graph.get_evidence("ev-1") == ev
    assert graph.get_claim("claim-1") == claim
    assert graph.get_claims_for_evidence("ev-1") == [claim]
    assert graph.get_evidence_for_claim("claim-1") == [ev]


def test_duplicate_nodes_and_relations_do_not_duplicate() -> None:
    graph = InMemoryEvidenceGraph()
    ev = evidence("ev-1")
    claim = Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=["ev-1"])
    edge = EvidenceGraphEdge(source_id="ev-1", target_id="claim-1", relation=EvidenceRelationType.SUPPORTS)
    graph.add_evidence(ev)
    graph.add_evidence(ev)
    graph.add_claim(claim)
    graph.add_claim(claim)
    graph.add_relation(edge)
    graph.add_relation(edge)

    assert len(graph.get_evidence_for_claim("claim-1")) == 1
    assert graph.get_relations() == [edge]


def test_duplicate_edge_id_for_different_relationship_is_rejected() -> None:
    graph = InMemoryEvidenceGraph()
    graph.add_evidence(evidence("ev-1"))
    graph.add_evidence(evidence("ev-2"))
    graph.add_claim(Claim(claim_id="claim-1", text="Claim", source_evidence_ids=[]))
    graph.add_relation(
        EvidenceGraphEdge(
            edge_id="edge-shared",
            source_id="ev-1",
            target_id="claim-1",
            relation=EvidenceRelationType.SUPPORTS,
        )
    )

    with pytest.raises(ValueError, match="edge ID"):
        graph.add_relation(
            EvidenceGraphEdge(
                edge_id="edge-shared",
                source_id="ev-2",
                target_id="claim-1",
                relation=EvidenceRelationType.SUPPORTS,
            )
        )


def test_missing_claim_evidence_and_invalid_edge_direction_rejected() -> None:
    graph = InMemoryEvidenceGraph()
    with pytest.raises(ValueError, match="missing evidence"):
        graph.add_claim(Claim(claim_id="claim-1", text="Orphan", source_evidence_ids=["missing"]))

    graph.add_evidence(evidence("ev-1"))
    graph.add_claim(Claim(claim_id="claim-1", text="Claim", source_evidence_ids=["ev-1"]))
    with pytest.raises(ValueError, match="SUPPORTS"):
        graph.add_relation(EvidenceGraphEdge(source_id="claim-1", target_id="ev-1", relation=EvidenceRelationType.SUPPORTS))


def test_relations_are_filtered_and_deterministically_sorted() -> None:
    graph = InMemoryEvidenceGraph()
    for evidence_id in ["ev-b", "ev-a"]:
        graph.add_evidence(evidence(evidence_id))
    for claim_id, evidence_id in [("claim-b", "ev-b"), ("claim-a", "ev-a")]:
        graph.add_claim(Claim(claim_id=claim_id, text=claim_id, source_evidence_ids=[evidence_id]))
        graph.add_relation(EvidenceGraphEdge(source_id=evidence_id, target_id=claim_id, relation=EvidenceRelationType.SUPPORTS))

    relations = graph.get_relations(relation=EvidenceRelationType.SUPPORTS)
    assert relations == graph.get_relations(relation=EvidenceRelationType.SUPPORTS)
    assert {edge.source_id for edge in relations} == {"ev-a", "ev-b"}
    assert [edge.edge_id for edge in relations] == sorted(edge.edge_id for edge in relations)


def test_build_evidence_graph_adds_support_edges_from_pool() -> None:
    pool = InMemoryEvidencePool()
    pool.add(evidence("ev-1"), task_id="task-1")
    claim = Claim(claim_id="claim-1", text="Revenue increased.", source_evidence_ids=["ev-1"])

    graph = build_evidence_graph(pool, [claim])

    assert graph.get_evidence_for_claim("claim-1") == [evidence("ev-1")]
    assert graph.get_relations(relation=EvidenceRelationType.SUPPORTS)[0].source_id == "ev-1"


def test_build_evidence_graph_rejects_missing_reference() -> None:
    with pytest.raises(ValueError, match="missing evidence"):
        build_evidence_graph(
            InMemoryEvidencePool(),
            [Claim(claim_id="claim-1", text="Orphan", source_evidence_ids=["missing"])],
        )
