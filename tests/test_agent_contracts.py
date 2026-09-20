"""
Unit tests for the Phase 9 agent contracts (prometheus.agents.contracts).

These test the data contracts and their deterministic groundedness
invariants (docs/phase9-multi-agent-design.md section 6) independently of
the agents themselves, which aren't implemented until Phases 10-13.
"""
import pytest

from prometheus.agents.contracts import (
    Claim,
    QuizQuestion,
    QuizResult,
    ResearchResult,
    quiz_is_grounded,
    research_claims_are_grounded,
)
from prometheus.agents.explainer_agent import run_explainer_agent
from prometheus.agents.literature_review_agent import run_literature_review_agent
from prometheus.agents.quiz_agent import run_quiz_agent
from prometheus.agents.research_agent import run_research_agent
from prometheus.rag.models import SourceRecord


def _evidence(chunk_id: str) -> SourceRecord:
    return SourceRecord(citation_index=1, document_id="doc-1", chunk_id=chunk_id, text_snippet="...")


def test_research_claims_grounded_in_own_evidence():
    result = ResearchResult(
        topic="RAG",
        evidence=[_evidence("c1"), _evidence("c2")],
        claims=[Claim(statement="RAG grounds generation in retrieval.", supporting_evidence_ids=["c1"], confidence=0.9)],
        has_sufficient_evidence=True,
    )
    assert research_claims_are_grounded(result)


def test_research_claim_citing_unknown_id_is_not_grounded():
    result = ResearchResult(
        topic="RAG",
        evidence=[_evidence("c1")],
        claims=[Claim(statement="fabricated", supporting_evidence_ids=["does-not-exist"], confidence=0.9)],
        has_sufficient_evidence=True,
    )
    assert not research_claims_are_grounded(result)


def test_research_claim_with_no_evidence_ids_is_not_grounded():
    result = ResearchResult(
        topic="RAG",
        evidence=[_evidence("c1")],
        claims=[Claim(statement="unsupported", supporting_evidence_ids=[], confidence=0.9)],
        has_sufficient_evidence=True,
    )
    assert not research_claims_are_grounded(result)


def test_quiz_grounded_in_research_claim_ids():
    research = ResearchResult(
        topic="RAG",
        evidence=[_evidence("c1")],
        claims=[Claim(statement="RAG grounds generation in retrieval.", supporting_evidence_ids=["c1"], confidence=0.9)],
        has_sufficient_evidence=True,
    )
    quiz = QuizResult(
        topic="RAG",
        questions=[
            QuizQuestion(
                question="What does RAG ground generation in?",
                answer="retrieval",
                explanation="Per the retrieved evidence.",
                source_evidence_ids=["c1"],
            )
        ],
    )
    assert quiz_is_grounded(research, quiz)


def test_quiz_citing_id_outside_research_is_not_grounded():
    research = ResearchResult(
        topic="RAG",
        evidence=[_evidence("c1")],
        claims=[Claim(statement="RAG grounds generation in retrieval.", supporting_evidence_ids=["c1"], confidence=0.9)],
        has_sufficient_evidence=True,
    )
    quiz = QuizResult(
        topic="RAG",
        questions=[
            QuizQuestion(
                question="fabricated question",
                answer="?",
                explanation="?",
                source_evidence_ids=["c-never-seen"],
            )
        ],
    )
    assert not quiz_is_grounded(research, quiz)


@pytest.mark.parametrize(
    "agent_fn",
    [run_research_agent, run_explainer_agent, run_quiz_agent, run_literature_review_agent],
)
def test_phase_10_to_13_agents_are_not_yet_implemented(agent_fn):
    with pytest.raises(NotImplementedError):
        agent_fn(None)