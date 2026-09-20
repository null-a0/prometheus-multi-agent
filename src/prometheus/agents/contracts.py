"""
Data contracts for the Phase 9 multi-agent redesign (see
docs/phase9-multi-agent-design.md section 4). These are what let
Research/Explainer/Quiz/LiteratureReview be built and tested in isolation:
each agent's node function takes one of the `*Request` models below and
returns the matching `*Result`, never a bare dict.

Style follows prometheus.rag.models: Pydantic, Field(description=...) on
every field so the schema is self-documenting.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from prometheus.rag.models import AcademicSourceRecord, SourceRecord

# ---------------------------------------------------------------------------
# Research Agent (Phase 10) â€” "What information/evidence do we need?"
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    topic: str = Field(description="User's topic or question.")
    document_id: str | None = Field(default=None, description="Optional scope to a single ingested document.")
    include_academic: bool = Field(default=False, description="Whether to query external paper sources (OpenAlex etc.).")
    max_sources: int = Field(default=10, description="Cap on returned evidence items.")


class Claim(BaseModel):
    statement: str = Field(description="A single factual claim extracted from evidence.")
    supporting_evidence_ids: list[str] = Field(
        description="chunk_id/source_id values from `evidence`/`academic_sources` backing this claim. Never empty."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Retrieval/rerank-derived confidence, not an LLM self-rating.")


class ResearchResult(BaseModel):
    topic: str
    evidence: list[SourceRecord] = Field(default_factory=list, description="Local retrieved+reranked chunks.")
    academic_sources: list[AcademicSourceRecord] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list, description="Every claim must cite at least one evidence/academic_source id.")
    has_sufficient_evidence: bool = Field(
        description="False if retrieval returned nothing usable; downstream agents must handle this explicitly, not silently proceed."
    )


# ---------------------------------------------------------------------------
# Explainer Agent (Phase 11) â€” "How can I understand this easily?"
# ---------------------------------------------------------------------------


class ExplainerRequest(BaseModel):
    research: ResearchResult = Field(
        description="Must come from a prior Research Agent call in the same session; never constructed from a bare topic string."
    )
    levels: list[Literal["eli5", "beginner", "technical"]] = Field(default=["eli5", "beginner", "technical"])


class LevelExplanation(BaseModel):
    level: str
    intuition: str = Field(description="Plain-language framing, no jargon.")
    example: str
    explanation: str
    key_takeaway: str


class ExplanationResult(BaseModel):
    topic: str
    levels: list[LevelExplanation]
    source_evidence_ids: list[str] = Field(
        description="Union of evidence ids actually referenced across all levels â€” lets the UI show 'grounded in N sources'."
    )


# ---------------------------------------------------------------------------
# Quiz Agent (Phase 12) â€” "Do I actually understand it?"
# ---------------------------------------------------------------------------


class QuizRequest(BaseModel):
    research: ResearchResult
    explanation: ExplanationResult | None = Field(default=None, description="Optional; if provided, questions may target specific levels.")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_count: int = Field(default=5, ge=1, le=20)
    question_type: Literal["mcq", "short_answer", "mixed"] = "mcq"


class QuizQuestion(BaseModel):
    question: str
    options: list[str] | None = Field(default=None, description="Set for mcq; null for short_answer.")
    answer: str
    explanation: str = Field(description="Why this is the answer, referencing the evidence.")
    source_evidence_ids: list[str] = Field(
        description="Must be a subset of research.claims[*].supporting_evidence_ids."
    )


class QuizResult(BaseModel):
    topic: str
    questions: list[QuizQuestion]


# ---------------------------------------------------------------------------
# Literature Review Agent (Phase 13) â€” "How do I systematically synthesize
# research on this topic?"
# ---------------------------------------------------------------------------


class LiteratureReviewRequest(BaseModel):
    question: str = Field(description="The broad research question.")
    sections: list[str] | None = Field(default=None, description="User-specified section list; None means the agent proposes a default structure.")
    citation_style: Literal["ieee", "apa", "plain"] = "plain"
    max_sub_topics: int = Field(default=5, ge=1, le=10)


class ReviewSection(BaseModel):
    heading: str
    content: str
    supporting_finding_ids: list[str] = Field(
        description="Report/Claim ids from prometheus.findings.models, per sub-topic sub-orchestrator run."
    )


class LiteratureReviewResult(BaseModel):
    question: str
    sections: list[ReviewSection]
    open_gaps: list[str] = Field(
        default_factory=list,
        description="Unresolved Critic objections or coverage gaps, surfaced rather than hidden.",
    )
    references: list[AcademicSourceRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Groundedness validators (docs/phase9-multi-agent-design.md section 6: these
# are the deterministic checks, not LLM-judge calls, and Phase 10-13 test
# suites should assert against them directly rather than reimplementing).
# ---------------------------------------------------------------------------


def research_claims_are_grounded(result: ResearchResult) -> bool:
    """Every Claim.supporting_evidence_ids id must resolve to a real
    evidence/academic_source id within the same ResearchResult, and no
    claim may cite zero ids."""
    known_ids = {chunk.chunk_id for chunk in result.evidence}
    known_ids |= {source.source_id for source in result.academic_sources}
    return all(
        claim.supporting_evidence_ids and all(cid in known_ids for cid in claim.supporting_evidence_ids)
        for claim in result.claims
    )


def quiz_is_grounded(research: ResearchResult, quiz: QuizResult) -> bool:
    """Every QuizQuestion.source_evidence_ids id must be a subset of the ids
    already cited by `research`'s own claims â€” the Quiz Agent must not
    introduce evidence the Research Agent never surfaced."""
    known_ids = {cid for claim in research.claims for cid in claim.supporting_evidence_ids}
    return all(
        question.source_evidence_ids and all(cid in known_ids for cid in question.source_evidence_ids)
        for question in quiz.questions
    )