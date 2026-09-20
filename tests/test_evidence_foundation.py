from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prometheus.api.app import app
from prometheus.llm.base import LLMProviderError
from prometheus.llm.mock import MockLLMClient
from prometheus.models.types import ModelGatewayError
from prometheus.rag import pipeline
from prometheus.rag.evidence import build_evidence
from prometheus.rag.models import Evidence
from prometheus.retrieval import hybrid_search, reranker


def _candidate() -> dict:
    return {
        "id": "chunk-1",
        "text": "Evidence about hybrid retrieval.",
        "metadata": {
            "document_id": "doc-1",
            "filename": "notes.txt",
            "file_type": "txt",
            "page_number": 4,
            "source_type": "domain_doc",
            "heading": "Retrieval",
        },
        "score": 0.72,
        "rerank_score": 0.91,
        "final_score": 0.91,
    }


def test_evidence_model_preserves_provenance_and_scores() -> None:
    evidence = Evidence(
        evidence_id="chunk-1",
        document_id="doc-1",
        document_name="notes.txt",
        chunk_id="chunk-1",
        text="A passage.",
        page_number=4,
        source_type="domain_doc",
        retrieval_method="hybrid",
        retrieval_score=0.72,
        reranker_score=0.91,
        final_score=0.91,
        metadata={"file_type": "txt"},
    )

    assert evidence.document_id == "doc-1"
    assert evidence.document_name == "notes.txt"
    assert evidence.page_number == 4
    assert evidence.retrieval_score == 0.72
    assert evidence.reranker_score == 0.91
    assert evidence.final_score == 0.91


def test_build_evidence_does_not_fabricate_missing_provenance() -> None:
    evidence = build_evidence([{"text": "Unattributed passage", "metadata": {}}])[0]

    assert evidence.text == "Unattributed passage"
    assert evidence.evidence_id is None
    assert evidence.document_id is None
    assert evidence.document_name is None
    assert evidence.page_number is None
    assert evidence.metadata == {}


def test_build_evidence_preserves_retrieval_provenance() -> None:
    evidence = build_evidence([_candidate()])[0]

    assert evidence.evidence_id == "chunk-1"
    assert evidence.document_id == "doc-1"
    assert evidence.document_name == "notes.txt"
    assert evidence.chunk_id == "chunk-1"
    assert evidence.retrieval_method == "hybrid"
    assert evidence.retrieval_score == 0.72
    assert evidence.reranker_score == 0.91
    assert evidence.metadata["page_number"] == 4


def test_hybrid_retrieval_fuses_existing_retrieval_methods() -> None:
    bm25_hit = {"id": "bm25", "text": "lexical", "metadata": {}, "score": 0.5}
    dense_hit = {"id": "dense", "text": "semantic", "metadata": {}, "score": 0.4}
    with (
        patch.object(hybrid_search, "bm25_search", return_value=[bm25_hit]),
        patch.object(hybrid_search, "dense_search", return_value=[dense_hit]),
    ):
        results = hybrid_search.hybrid_search("query", top_k=2)

    assert {result["id"] for result in results} == {"bm25", "dense"}
    assert all("score" in result for result in results)


def test_reranker_preserves_candidates_and_adds_scores() -> None:
    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            assert pairs == [("query", "passage")]
            return [0.88]

    candidate = {"id": "chunk-1", "text": "passage", "metadata": {}, "score": 0.4}
    with patch.object(reranker, "_get_cross_encoder", return_value=FakeCrossEncoder()):
        results = reranker.rerank("query", [candidate])

    assert results[0] is candidate
    assert results[0]["rerank_score"] == 0.88


def test_query_maps_evidence_to_citations() -> None:
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "rerank", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "apply_recency_and_credibility_filters", return_value=[_candidate()]),
    ):
        result = pipeline.query_rag("What is retrieval?", llm_client=MockLLMClient())

    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "chunk-1"
    assert result.citations == result.sources
    assert result.citations[0].citation_index == 1
    assert result.evidence_status == "sufficient"


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.query_rag("   ")


def test_no_retrieval_results_do_not_call_llm() -> None:
    client = MockLLMClient()
    with patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[]):
        result = pipeline.query_rag("No matching evidence", llm_client=client)

    assert result.answer == pipeline.NO_EVIDENCE_MESSAGE
    assert result.evidence == []
    assert result.evidence_status == "no_evidence"
    assert client.call_history == []


def test_retrieved_but_filtered_evidence_is_insufficient() -> None:
    client = MockLLMClient()
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "rerank", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "apply_recency_and_credibility_filters", return_value=[]),
    ):
        result = pipeline.query_rag("Weak evidence", llm_client=client)

    assert result.answer == pipeline.INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.evidence == []
    assert result.evidence_status == "insufficient_evidence"
    assert client.call_history == []


def test_reranker_failure_is_not_converted_to_an_answer() -> None:
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "rerank", side_effect=RuntimeError("model unavailable")),
        pytest.raises(pipeline.RagPipelineError, match="reranking failed"),
    ):
        pipeline.query_rag("Needs ranking", llm_client=MockLLMClient())


def test_llm_failure_is_normalized_for_api_classification() -> None:
    client = MockLLMClient(custom_handler=lambda _prompt, _system: (_ for _ in ()).throw(LLMProviderError("upstream")))
    with (
        patch.object(pipeline.hybrid_search, "hybrid_search", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "rerank", return_value=[_candidate()]),
        patch.object(pipeline.reranker, "apply_recency_and_credibility_filters", return_value=[_candidate()]),
        pytest.raises(ModelGatewayError, match="Model invocation failed"),
    ):
        pipeline.query_rag("Needs generation", llm_client=client)


def test_api_maps_llm_failure_to_bad_gateway() -> None:
    client = TestClient(app)
    with patch("prometheus.api.app.query_rag", side_effect=LLMProviderError("upstream")):
        response = client.post("/api/rag/query", json={"question": "What happened?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "RAG answer generation failed."
