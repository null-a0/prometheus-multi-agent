from __future__ import annotations

import pytest
from pydantic import ValidationError

from prometheus.investigation.findings import Finding, VerificationStatus


def finding(**overrides) -> Finding:
    values = {
        "finding_id": "finding-1",
        "claim_id": "claim-1",
        "verification_status": VerificationStatus.UNVERIFIED,
    }
    values.update(overrides)
    return Finding(**values)


def test_unverified_without_evidence_or_reasons_is_valid() -> None:
    result = finding()

    assert result.verification_status is VerificationStatus.UNVERIFIED
    assert result.supporting_evidence_ids == []
    assert result.unresolved_reasons == []


def test_supported_requires_supporting_evidence() -> None:
    result = finding(
        verification_status=VerificationStatus.SUPPORTED,
        supporting_evidence_ids=["e1", "e2"],
    )

    assert result.supporting_evidence_ids == ["e1", "e2"]


def test_conflicting_requires_evidence_and_reason() -> None:
    result = finding(
        verification_status=VerificationStatus.CONFLICTING,
        supporting_evidence_ids=["e1"],
        contradicting_evidence_ids=["e2"],
        unresolved_reasons=["Sources disagree on the reported value"],
    )

    assert result.verification_status is VerificationStatus.CONFLICTING


def test_insufficient_evidence_requires_reason() -> None:
    result = finding(
        verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        unresolved_reasons=["No sufficiently relevant evidence was retrieved"],
    )

    assert result.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_multiple_reasons_and_internal_spaces_are_valid() -> None:
    result = finding(
        finding_id="finding with spaces",
        claim_id="claim with spaces",
        verification_status=VerificationStatus.CONFLICTING,
        supporting_evidence_ids=["evidence one", "evidence two"],
        contradicting_evidence_ids=["evidence three"],
        unresolved_reasons=["Only one source was available", "Sources disagree"],
    )

    assert result.finding_id == "finding with spaces"
    assert len(result.unresolved_reasons) == 2


@pytest.mark.parametrize("field", ["finding_id", "claim_id"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_identity_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        finding(**{field: value})


@pytest.mark.parametrize("field", ["supporting_evidence_ids", "contradicting_evidence_ids"])
@pytest.mark.parametrize("value", [[""], ["   "]])
def test_empty_evidence_ids_rejected(field: str, value: list[str]) -> None:
    with pytest.raises(ValidationError):
        finding(**{field: value})


def test_duplicate_supporting_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(supporting_evidence_ids=["e1", "e1"])


def test_duplicate_contradicting_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(contradicting_evidence_ids=["e2", "e2"])


def test_same_evidence_cannot_support_and_contradict() -> None:
    with pytest.raises(ValidationError):
        finding(supporting_evidence_ids=["e1"], contradicting_evidence_ids=["e1"])


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_unresolved_reason_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        finding(
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            unresolved_reasons=[value],
        )


def test_supported_without_supporting_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(verification_status=VerificationStatus.SUPPORTED, contradicting_evidence_ids=["e2"])


def test_conflicting_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(
            verification_status=VerificationStatus.CONFLICTING,
            unresolved_reasons=["Sources were not comparable"],
        )


def test_conflicting_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(
            verification_status=VerificationStatus.CONFLICTING,
            supporting_evidence_ids=["e1"],
            contradicting_evidence_ids=["e2"],
        )


def test_insufficient_evidence_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        finding(verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE)


@pytest.mark.parametrize(
    "field",
    ["truth", "truth_score", "confidence", "confidence_score", "probability", "is_true", "is_verified", "arbitrary"],
)
def test_truth_and_unknown_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        finding(**{field: True})


def test_serialization_preserves_enum_values() -> None:
    for status in VerificationStatus:
        result = finding(
            verification_status=status,
            supporting_evidence_ids=["e1"] if status is VerificationStatus.SUPPORTED else [],
            contradicting_evidence_ids=["e2"] if status is VerificationStatus.CONFLICTING else [],
            unresolved_reasons=["Reason"] if status in {
                VerificationStatus.CONFLICTING,
                VerificationStatus.INSUFFICIENT_EVIDENCE,
            } else [],
        )
        assert result.model_dump(mode="json")["verification_status"] == status.value
