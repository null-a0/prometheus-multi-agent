"""Validated evidentiary state for an explicit claim."""
from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _require_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be non-empty and not whitespace-only")
    return value.strip()


def _validate_ids(values: list[str]) -> list[str]:
    cleaned = [_require_text(value) for value in values]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("evidence IDs must not contain duplicates")
    return cleaned


def _validate_reasons(values: list[str]) -> list[str]:
    return [_require_text(value) for value in values]


class Finding(BaseModel):
    """Current evidentiary status of a claim, without truth or confidence scoring."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    claim_id: str
    verification_status: VerificationStatus
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_reasons: list[str] = Field(default_factory=list)

    _validate_identity = field_validator("finding_id", "claim_id")(_require_text)
    _validate_supporting_ids = field_validator("supporting_evidence_ids")(_validate_ids)
    _validate_contradicting_ids = field_validator("contradicting_evidence_ids")(_validate_ids)
    _validate_reasons = field_validator("unresolved_reasons")(_validate_reasons)

    @model_validator(mode="after")
    def validate_evidentiary_state(self) -> Self:
        supporting = set(self.supporting_evidence_ids)
        contradicting = set(self.contradicting_evidence_ids)
        if supporting.intersection(contradicting):
            raise ValueError("an evidence ID cannot both support and contradict the same finding")

        if self.verification_status is VerificationStatus.SUPPORTED and not supporting:
            raise ValueError("SUPPORTED findings require supporting evidence")

        if self.verification_status is VerificationStatus.CONFLICTING:
            if not supporting and not contradicting:
                raise ValueError("CONFLICTING findings require supporting or contradicting evidence")
            if not self.unresolved_reasons:
                raise ValueError("CONFLICTING findings require unresolved reasons")

        if (
            self.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE
            and not self.unresolved_reasons
        ):
            raise ValueError("INSUFFICIENT_EVIDENCE findings require unresolved reasons")
        return self
