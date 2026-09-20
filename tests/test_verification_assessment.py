from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from prometheus.investigation import (
    Claim,
    DeterministicVerificationAssessor,
    EvidenceGraphEdge,
    EvidenceRelationType,
    Finding,
    InMemoryEvidenceGraph,
    VerificationAssessment,
    VerificationAssessmentError,
    VerificationAssessmentInput,
    VerificationStatus,
)
from prometheus.rag.models import Evidence


def evidence(evidence_id: str) -> Evidence:
    return Evidence(evidence_id=evidence_id, text=f"Evidence {evidence_id}")


def finding(**overrides) -> Finding:
    values = {
        "finding_id": "finding-1",
        "claim_id": "claim-1",
        "verification_status": VerificationStatus.UNVERIFIED,
    }
    values.update(overrides)
    return Finding(**values)


def graph_with(
    *,
    evidence_ids: list[str] | None = None,
    claim_ids: list[str] | None = None,
    edges: list[EvidenceGraphEdge] | None = None,
) -> InMemoryEvidenceGraph:
    graph = InMemoryEvidenceGraph()
    for evidence_id in evidence_ids or ["ev-1"]:
        graph.add_evidence(evidence(evidence_id))
    for claim_id in claim_ids or ["claim-1"]:
        graph.add_claim(Claim(claim_id=claim_id, text=f"Claim {claim_id}"))
    for edge in edges or []:
        graph.add_relation(edge)
    return graph


def assess(result: Finding, graph: InMemoryEvidenceGraph | None = None) -> VerificationAssessment:
    request = VerificationAssessmentInput(
        finding=result,
        evidence_graph=graph or graph_with(),
    )
    return DeterministicVerificationAssessor().assess(request)


def edge(source_id: str, relation: EvidenceRelationType, target_id: str = "claim-1") -> EvidenceGraphEdge:
    return EvidenceGraphEdge(source_id=source_id, target_id=target_id, relation=relation)


def edge_id(source_id: str, relation: EvidenceRelationType, target_id: str = "claim-1") -> str:
    payload = f"{source_id}\x1f{relation.value}\x1f{target_id}".encode()
    return f"edge-{hashlib.sha256(payload).hexdigest()[:16]}"


def test_valid_input_and_extra_input_field_rejected() -> None:
    request = VerificationAssessmentInput(finding=finding(), evidence_graph=graph_with())

    result = DeterministicVerificationAssessor().assess(request)

    assert result.finding_id == "finding-1"
    with pytest.raises(ValidationError):
        VerificationAssessmentInput(
            finding=finding(),
            evidence_graph=graph_with(),
            assessment_id="caller-provided",
        )


def test_invalid_finding_missing_claim_and_invalid_graph_fail_closed() -> None:
    invalid = Finding.model_construct(
        finding_id="finding-1",
        claim_id=" ",
        verification_status=VerificationStatus.UNVERIFIED,
        supporting_evidence_ids=[],
        contradicting_evidence_ids=[],
        unresolved_reasons=[],
    )
    with pytest.raises(VerificationAssessmentError, match="Finding is invalid"):
        assess(invalid)

    with pytest.raises(VerificationAssessmentError, match="Claim referenced"):
        assess(finding(claim_id="missing"))

    with pytest.raises(VerificationAssessmentError, match="EvidenceGraph"):
        DeterministicVerificationAssessor().assess(
            VerificationAssessmentInput.model_construct(finding=finding(), evidence_graph=None)
        )


def test_unverified_mapping_and_failure_conditions() -> None:
    result = assess(finding())

    assert result.rationale == "The finding has not been assessed."
    assert result.supporting_evidence_ids == []
    assert result.contradicting_evidence_ids == []
    assert result.supporting_edge_ids == []
    assert result.contradicting_edge_ids == []
    assert result.unresolved_reasons == []

    with pytest.raises(VerificationAssessmentError):
        assess(finding(supporting_evidence_ids=["ev-1"]))
    with pytest.raises(VerificationAssessmentError):
        assess(finding(contradicting_evidence_ids=["ev-1"]))
    with pytest.raises(VerificationAssessmentError):
        assess(finding(unresolved_reasons=["Assessment has not happened"]))


def test_supported_mapping_multiple_edges_and_failure_conditions() -> None:
    graph = graph_with(
        evidence_ids=["ev-1", "ev-2"],
        edges=[
            edge("ev-2", EvidenceRelationType.SUPPORTS),
            edge("ev-1", EvidenceRelationType.SUPPORTS),
        ],
    )
    result = assess(
        finding(
            verification_status=VerificationStatus.SUPPORTED,
            supporting_evidence_ids=["ev-1", "ev-2"],
        ),
        graph,
    )

    assert result.rationale == (
        "The finding is supported by evidence with corresponding SUPPORTS relationships "
        "in the evidence graph."
    )
    assert result.supporting_evidence_ids == ["ev-1", "ev-2"]
    assert set(result.supporting_edge_ids) == {
        edge_id("ev-1", EvidenceRelationType.SUPPORTS),
        edge_id("ev-2", EvidenceRelationType.SUPPORTS),
    }

    with pytest.raises(ValidationError):
        finding(verification_status=VerificationStatus.SUPPORTED)
    with pytest.raises(VerificationAssessmentError, match="does not exist"):
        assess(
            finding(
                verification_status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=["missing"],
            )
        )
    with pytest.raises(VerificationAssessmentError, match="Missing SUPPORTS"):
        assess(
            finding(
                verification_status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=["ev-1"],
            )
        )
    with pytest.raises(VerificationAssessmentError, match="Missing SUPPORTS"):
        assess(
            finding(
                verification_status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=["ev-1"],
            ),
            graph_with(edges=[edge("ev-1", EvidenceRelationType.CONTRADICTS)]),
        )
    with pytest.raises(VerificationAssessmentError):
        assess(
            finding(
                verification_status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=["ev-1"],
                contradicting_evidence_ids=["ev-2"],
            ),
            graph_with(
                evidence_ids=["ev-1", "ev-2"],
                edges=[
                    edge("ev-1", EvidenceRelationType.SUPPORTS),
                    edge("ev-2", EvidenceRelationType.CONTRADICTS),
                ],
            ),
        )
    supported_with_reason = assess(
        finding(
            verification_status=VerificationStatus.SUPPORTED,
            supporting_evidence_ids=["ev-1"],
            unresolved_reasons=["Still open"],
        ),
        graph_with(edges=[edge("ev-1", EvidenceRelationType.SUPPORTS)]),
    )
    assert supported_with_reason.unresolved_reasons == ["Still open"]


def test_conflicting_mapping_and_failure_conditions() -> None:
    graph = graph_with(
        evidence_ids=["ev-support", "ev-conflict"],
        edges=[
            edge("ev-support", EvidenceRelationType.SUPPORTS),
            edge("ev-conflict", EvidenceRelationType.CONTRADICTS),
        ],
    )
    result = assess(
        finding(
            verification_status=VerificationStatus.CONFLICTING,
            supporting_evidence_ids=["ev-support"],
            contradicting_evidence_ids=["ev-conflict"],
            unresolved_reasons=["Sources disagree"],
        ),
        graph,
    )

    assert result.supporting_edge_ids == [edge_id("ev-support", EvidenceRelationType.SUPPORTS)]
    assert result.contradicting_edge_ids == [
        edge_id("ev-conflict", EvidenceRelationType.CONTRADICTS)
    ]
    assert result.unresolved_reasons == ["Sources disagree"]

    with pytest.raises(VerificationAssessmentError, match="supporting and contradicting"):
        assess(
            finding(
                verification_status=VerificationStatus.CONFLICTING,
                supporting_evidence_ids=["ev-support"],
                unresolved_reasons=["Sources disagree"],
            ),
            graph,
        )
    with pytest.raises(VerificationAssessmentError, match="supporting and contradicting"):
        assess(
            finding(
                verification_status=VerificationStatus.CONFLICTING,
                contradicting_evidence_ids=["ev-conflict"],
                unresolved_reasons=["Sources disagree"],
            ),
            graph,
        )
    with pytest.raises(ValidationError):
        finding(
            verification_status=VerificationStatus.CONFLICTING,
            supporting_evidence_ids=["ev-support"],
            contradicting_evidence_ids=["ev-conflict"],
        )
    with pytest.raises(VerificationAssessmentError, match="Missing SUPPORTS"):
        assess(
            finding(
                verification_status=VerificationStatus.CONFLICTING,
                supporting_evidence_ids=["ev-support"],
                contradicting_evidence_ids=["ev-conflict"],
                unresolved_reasons=["Sources disagree"],
            ),
            graph_with(
                evidence_ids=["ev-support", "ev-conflict"],
                edges=[edge("ev-conflict", EvidenceRelationType.CONTRADICTS)],
            ),
        )
    with pytest.raises(VerificationAssessmentError, match="Missing CONTRADICTS"):
        assess(
            finding(
                verification_status=VerificationStatus.CONFLICTING,
                supporting_evidence_ids=["ev-support"],
                contradicting_evidence_ids=["ev-conflict"],
                unresolved_reasons=["Sources disagree"],
            ),
            graph_with(
                evidence_ids=["ev-support", "ev-conflict"],
                edges=[edge("ev-support", EvidenceRelationType.SUPPORTS)],
            ),
        )


def test_insufficient_evidence_mapping_with_and_without_traceable_evidence() -> None:
    without_evidence = assess(
        finding(
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            unresolved_reasons=["No relevant evidence was found"],
        )
    )
    assert without_evidence.supporting_edge_ids == []

    with_evidence = assess(
        finding(
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=["ev-1"],
            contradicting_evidence_ids=["ev-2"],
            unresolved_reasons=["Evidence is incomplete"],
        ),
        graph_with(
            evidence_ids=["ev-1", "ev-2"],
            edges=[
                edge("ev-1", EvidenceRelationType.SUPPORTS),
                edge("ev-2", EvidenceRelationType.CONTRADICTS),
            ],
        ),
    )
    assert with_evidence.rationale == (
        "The finding records that the currently available evidence is insufficient for assessment."
    )
    assert with_evidence.supporting_edge_ids == [edge_id("ev-1", EvidenceRelationType.SUPPORTS)]
    assert with_evidence.contradicting_edge_ids == [
        edge_id("ev-2", EvidenceRelationType.CONTRADICTS)
    ]

    with pytest.raises(ValidationError):
        finding(verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE)
    with pytest.raises(VerificationAssessmentError, match="does not exist"):
        assess(
            finding(
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                supporting_evidence_ids=["missing"],
                unresolved_reasons=["Evidence is incomplete"],
            )
        )
    with pytest.raises(VerificationAssessmentError, match="Missing SUPPORTS"):
        assess(
            finding(
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                supporting_evidence_ids=["ev-1"],
                unresolved_reasons=["Evidence is incomplete"],
            )
        )


def test_traceability_invariants_and_unrelated_edges_are_not_included() -> None:
    graph = graph_with(
        evidence_ids=["ev-1", "ev-2"],
        claim_ids=["claim-1", "claim-2"],
        edges=[
            edge("ev-1", EvidenceRelationType.SUPPORTS, "claim-1"),
            edge("ev-1", EvidenceRelationType.SUPPORTS, "claim-2"),
            edge("ev-2", EvidenceRelationType.SUPPORTS, "claim-1"),
        ],
    )
    result = assess(
        finding(
            verification_status=VerificationStatus.SUPPORTED,
            supporting_evidence_ids=["ev-1"],
        ),
        graph,
    )

    assert result.finding_id == "finding-1"
    assert result.claim_id == "claim-1"
    assert result.status is VerificationStatus.SUPPORTED
    assert result.supporting_evidence_ids == ["ev-1"]
    assert result.supporting_edge_ids == [edge_id("ev-1", EvidenceRelationType.SUPPORTS)]


@pytest.mark.parametrize(
    "field",
    [
        "truth",
        "truth_score",
        "confidence",
        "confidence_score",
        "probability",
        "is_true",
        "is_verified",
        "source_reliability",
    ],
)
def test_boundary_fields_are_not_present(field: str) -> None:
    assert field not in VerificationAssessment.model_fields


def test_no_verified_status_exists() -> None:
    assert "VERIFIED" not in VerificationStatus.__members__


def test_determinism_no_mutation_no_resolution_and_no_model_calls() -> None:
    graph = graph_with(
        evidence_ids=["ev-support", "ev-conflict"],
        edges=[
            edge("ev-support", EvidenceRelationType.SUPPORTS),
            edge("ev-conflict", EvidenceRelationType.CONTRADICTS),
        ],
    )
    result_finding = finding(
        verification_status=VerificationStatus.CONFLICTING,
        supporting_evidence_ids=["ev-support"],
        contradicting_evidence_ids=["ev-conflict"],
        unresolved_reasons=["Sources disagree"],
    )
    finding_before = result_finding.model_dump()
    graph_before = graph.get_relations()

    with patch("prometheus.models.gateway.ModelGateway.generate") as generate:
        first = assess(result_finding, graph)
        second = assess(result_finding, graph)

    assert first == second
    assert result_finding.model_dump() == finding_before
    assert graph.get_relations() == graph_before
    assert first.status is VerificationStatus.CONFLICTING
    assert first.unresolved_reasons == ["Sources disagree"]
    generate.assert_not_called()


def test_verification_source_has_no_provider_or_retrieval_imports() -> None:
    source = Path("src/prometheus/investigation/verification.py").read_text(encoding="utf-8").lower()

    assert "openai" not in source
    assert "openrouter" not in source
    assert "modelgateway" not in source
    assert "hybrid_search" not in source
