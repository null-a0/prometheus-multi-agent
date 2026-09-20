from __future__ import annotations

from unittest.mock import patch

from prometheus.investigation import (
    Claim,
    ContradictionDetector,
    EvidenceRelationType,
    InMemoryEvidencePool,
    build_evidence_graph,
)
from prometheus.rag.models import Evidence


def graph_for_claims(*claims: Claim):
    pool = InMemoryEvidencePool()
    for claim in claims:
        for evidence_id in claim.source_evidence_ids:
            pool.add(Evidence(evidence_id=evidence_id, text=f"Evidence for {evidence_id}"), task_id="task")
    return build_evidence_graph(pool, list(claims))


def test_explicit_opposite_equal_change_is_detected() -> None:
    graph = graph_for_claims(
        Claim(claim_id="claim-a", text="Revenue increased by 10%.", source_evidence_ids=["ev-a"]),
        Claim(claim_id="claim-b", text="Revenue decreased by 10%.", source_evidence_ids=["ev-b"]),
    )

    edges = ContradictionDetector().detect(graph)

    assert len(edges) == 1
    assert edges[0].source_id == "claim-a"
    assert edges[0].target_id == "claim-b"
    assert edges[0].relation is EvidenceRelationType.CONTRADICTS


def test_non_contradictory_claims_produce_no_edges() -> None:
    graph = graph_for_claims(
        Claim(claim_id="claim-a", text="Revenue increased by 10%.", source_evidence_ids=["ev-a"]),
        Claim(claim_id="claim-b", text="Revenue increased by 10%.", source_evidence_ids=["ev-b"]),
        Claim(claim_id="claim-c", text="Revenue decreased by 5%.", source_evidence_ids=["ev-c"]),
    )

    assert ContradictionDetector().detect(graph) == []


def test_detector_does_not_mutate_graph() -> None:
    graph = graph_for_claims(
        Claim(claim_id="claim-a", text="Revenue increased by 10%.", source_evidence_ids=["ev-a"]),
        Claim(claim_id="claim-b", text="Revenue decreased by 10%.", source_evidence_ids=["ev-b"]),
    )
    before = graph.get_relations()

    ContradictionDetector().detect(graph)

    assert graph.get_relations() == before


def test_detector_does_not_call_an_llm() -> None:
    graph = graph_for_claims(
        Claim(claim_id="claim-a", text="Revenue increased by 10%.", source_evidence_ids=["ev-a"]),
        Claim(claim_id="claim-b", text="Revenue decreased by 10%.", source_evidence_ids=["ev-b"]),
    )
    with patch("prometheus.models.gateway.ModelGateway.generate") as generate:
        ContradictionDetector().detect(graph)
    generate.assert_not_called()


def test_detector_source_has_no_provider_imports() -> None:
    from pathlib import Path

    source = Path("src/prometheus/investigation/contradiction.py").read_text(encoding="utf-8").lower()
    assert "openai" not in source
    assert "openrouter" not in source
    assert "modelgateway" not in source
