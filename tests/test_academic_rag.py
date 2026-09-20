"""
Unit and integration tests for optional OpenAlex academic evidence in the Prometheus RAG pipeline.
Covers:
1. Academic evidence disabled by default (OpenAlex is not queried).
2. Academic evidence enabled (OpenAlex queried, results mapped with [A], [B], [C] citations).
3. Academic context construction and metadata formatting.
4. Graceful degradation on empty OpenAlex results.
5. Graceful degradation on OpenAlex network/HTTP/API errors.
6. Prompt construction separating Local Document Evidence and External Academic Evidence.
7. Document isolation (document_id scoping) preserved alongside academic evidence.
8. FastAPI /api/rag/query endpoint forwards academic parameters.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import prometheus.retrieval.hybrid_search as hybrid_search_module
import prometheus.retrieval.vector_store as vector_store_module
from prometheus.api.app import app
from prometheus.llm.mock import MockLLMClient
from prometheus.rag.context import build_academic_context, build_context
from prometheus.rag.pipeline import query_rag
from prometheus.rag.prompts import build_rag_user_prompt
from prometheus.retrieval.sources.paper_record import PaperRecord

pytestmark = pytest.mark.unit


def _isolate_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate ChromaDB and BM25 persistent files to a clean temporary directory for testing."""
    isolated_settings = dataclasses.replace(
        vector_store_module.settings,
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    monkeypatch.setattr(vector_store_module, "settings", isolated_settings)
    monkeypatch.setattr(hybrid_search_module, "settings", isolated_settings)
    monkeypatch.setattr(vector_store_module, "_vector_store", None)


@pytest.fixture
def sample_papers() -> list[PaperRecord]:
    return [
        PaperRecord(
            source="openalex",
            source_id="https://openalex.org/W123456789",
            title="Attention Is All You Need",
            authors=["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki"],
            year=2017,
            doi="https://doi.org/10.48550/arxiv.1706.03762",
            url="https://arxiv.org/abs/1706.03762",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        ),
        PaperRecord(
            source="openalex",
            source_id="https://openalex.org/W987654321",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin, Jacob", "Chang, Ming-Wei"],
            year=2018,
            doi="https://doi.org/10.48550/arxiv.1810.04805",
            url="https://arxiv.org/abs/1810.04805",
            pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
            abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Academic Evidence Disabled by Default
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_academic_evidence_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Prometheus is a generative AI platform for research and analysis.", encoding="utf-8")
    ingest_document(str(doc_path), document_id="doc-prom-1")

    with patch("prometheus.rag.pipeline.search_openalex") as mock_openalex:
        result = query_rag(
            question="What is Prometheus?",
            llm_client=MockLLMClient(),
            # include_academic_evidence defaults to False
        )

        mock_openalex.assert_not_called()
        assert result.academic_sources == []
        assert len(result.sources) > 0
        assert result.sources[0].citation_index == 1


# ---------------------------------------------------------------------------
# 2. Academic Evidence Enabled Queries OpenAlex
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_academic_evidence_enabled_queries_openalex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_papers: list[PaperRecord],
):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Prometheus utilizes transformer architecture.", encoding="utf-8")
    ingest_document(str(doc_path), document_id="doc-prom-2")

    with patch("prometheus.rag.pipeline.search_openalex", return_value=sample_papers) as mock_openalex:
        result = query_rag(
            question="What architecture does Prometheus use?",
            include_academic_evidence=True,
            openalex_max_results=2,
            llm_client=MockLLMClient(),
        )

        mock_openalex.assert_called_once_with(
            query="What architecture does Prometheus use?",
            max_results=2,
        )
        assert len(result.academic_sources) == 2
        assert result.academic_sources[0].citation_label == "[A]"
        assert result.academic_sources[0].title == "Attention Is All You Need"
        assert result.academic_sources[0].year == 2017
        assert result.academic_sources[1].citation_label == "[B]"
        assert result.academic_sources[1].title == "BERT: Pre-training of Deep Bidirectional Transformers"


# ---------------------------------------------------------------------------
# 3. Academic Context Construction & Formatting
# ---------------------------------------------------------------------------

def test_build_academic_context_formatting(sample_papers: list[PaperRecord]):
    context_str, records = build_academic_context(sample_papers)

    assert '(A] Paper: "Attention Is All You Need"' in context_str or '[A] Paper: "Attention Is All You Need"' in context_str
    assert "Authors: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki" in context_str
    assert "(2017)" in context_str
    assert "DOI / Link: https://doi.org/10.48550/arxiv.1706.03762" in context_str
    assert "Abstract: The dominant sequence transduction models" in context_str

    assert '[B] Paper: "BERT: Pre-training of Deep Bidirectional Transformers"' in context_str
    assert "(2018)" in context_str

    assert len(records) == 2
    assert records[0].citation_label == "[A]"
    assert records[0].source == "openalex"
    assert records[0].source_id == "https://openalex.org/W123456789"
    assert records[0].doi == "https://doi.org/10.48550/arxiv.1706.03762"

    assert records[1].citation_label == "[B]"


def test_build_academic_context_empty():
    context_str, records = build_academic_context([])
    assert context_str == ""
    assert records == []


# ---------------------------------------------------------------------------
# 4. Graceful Degradation on Empty OpenAlex Results
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_academic_evidence_empty_results_handled_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Local document content only.", encoding="utf-8")
    ingest_document(str(doc_path), document_id="doc-local-only")

    with patch("prometheus.rag.pipeline.search_openalex", return_value=[]):
        result = query_rag(
            question="What is the local content?",
            include_academic_evidence=True,
            openalex_max_results=3,
            llm_client=MockLLMClient(),
        )

        assert result.academic_sources == []
        assert len(result.sources) > 0
        assert "Local document content" in result.answer or len(result.answer) > 0


# ---------------------------------------------------------------------------
# 5. Graceful Degradation on OpenAlex Network / HTTP Errors
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_academic_evidence_http_error_falls_back_to_local_rag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Local document content survives external failure.", encoding="utf-8")
    ingest_document(str(doc_path), document_id="doc-resilient")

    # Simulate network timeout / HTTP failure from OpenAlex
    mock_error = httpx.ConnectTimeout("OpenAlex API timed out after 10s")
    with patch("prometheus.rag.pipeline.search_openalex", side_effect=mock_error):
        result = query_rag(
            question="Does local RAG survive when OpenAlex fails?",
            include_academic_evidence=True,
            llm_client=MockLLMClient(),
        )

        # RAG should NOT crash
        assert result.academic_sources == []
        assert len(result.sources) > 0
        assert result.sources[0].filename == "sample.txt"
        # Warning should be logged
        assert any("Failed to fetch OpenAlex academic evidence" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 6. Prompt Construction Separation
# ---------------------------------------------------------------------------

def test_prompt_separates_local_and_academic_evidence(sample_papers: list[PaperRecord]):
    mock_chunks = [
        {
            "id": "c-1",
            "text": "Internal system notes on attention.",
            "metadata": {
                "document_id": "doc-notes",
                "filename": "notes.pdf",
                "page_number": 1,
            },
            "score": 0.9,
        }
    ]
    local_context, _ = build_context(mock_chunks)
    academic_context, _ = build_academic_context(sample_papers)

    prompt = build_rag_user_prompt(
        question="Explain transformer attention.",
        context_str=local_context,
        academic_context_str=academic_context,
    )

    assert "LOCAL DOCUMENT EVIDENCE:" in prompt
    assert "[1] Source: notes.pdf" in prompt
    assert "EXTERNAL ACADEMIC EVIDENCE (OPENALEX):" in prompt
    assert '[A] Paper: "Attention Is All You Need"' in prompt
    assert '[B] Paper: "BERT: Pre-training of Deep Bidirectional Transformers"' in prompt
    assert "User Question:" in prompt
    assert "Explain transformer attention." in prompt


def test_prompt_without_academic_evidence_preserves_legacy_format():
    local_context = "[1] Source: doc.pdf (Page 1) [DocID: doc-1]\nSingle source text."
    prompt = build_rag_user_prompt(
        question="What is this?",
        context_str=local_context,
        academic_context_str=None,
    )

    assert "Context Evidence:" in prompt
    assert "LOCAL DOCUMENT EVIDENCE:" not in prompt
    assert "EXTERNAL ACADEMIC EVIDENCE" not in prompt
    assert local_context in prompt


# ---------------------------------------------------------------------------
# 7. Document Isolation Preserved with Academic Evidence
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_document_scoping_preserved_with_academic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_papers: list[PaperRecord],
):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc1 = tmp_path / "paper_alpha.txt"
    doc1.write_text("Alpha discusses neural network weights and optimization.", encoding="utf-8")
    ingest_document(str(doc1), document_id="doc-alpha")

    doc2 = tmp_path / "paper_beta.txt"
    doc2.write_text("Beta discusses quantum computing and qubits.", encoding="utf-8")
    ingest_document(str(doc2), document_id="doc-beta")

    with patch("prometheus.rag.pipeline.search_openalex", return_value=sample_papers):
        # Query scoped strictly to doc-alpha
        result_alpha = query_rag(
            question="What is discussed?",
            document_id="doc-alpha",
            include_academic_evidence=True,
            llm_client=MockLLMClient(),
        )

        assert all(src.document_id == "doc-alpha" for src in result_alpha.sources)
        assert not any(src.document_id == "doc-beta" for src in result_alpha.sources)
        assert len(result_alpha.academic_sources) == 2


# ---------------------------------------------------------------------------
# 8. API Endpoint Supports Academic Parameters
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.model_backed
def test_api_endpoint_supports_academic_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_papers: list[PaperRecord],
):
    from prometheus.retrieval.ingestion import ingest_document

    _isolate_stores(tmp_path, monkeypatch)

    doc = tmp_path / "api_doc.txt"
    doc.write_text("API test document on deep learning.", encoding="utf-8")
    ingest_document(str(doc), document_id="doc-api")

    client = TestClient(app)

    with patch("prometheus.rag.pipeline.search_openalex", return_value=sample_papers) as mock_openalex:
        req_payload = {
            "question": "What is deep learning?",
            "document_id": "doc-api",
            "include_academic_evidence": True,
            "openalex_max_results": 2,
        }
        response = client.post("/api/rag/query", json=req_payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "academic_sources" in data
        assert len(data["academic_sources"]) == 2
        assert data["academic_sources"][0]["citation_label"] == "[A]"
        assert data["academic_sources"][0]["title"] == "Attention Is All You Need"
        assert data["academic_sources"][1]["citation_label"] == "[B]"
        mock_openalex.assert_called_once_with(
            query="What is deep learning?",
            max_results=2,
        )
