"""
Optional live integration test for Prometheus RAG pipeline against OpenRouter API.
Skipped automatically if OPENROUTER_API_KEY is not set.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.model_backed, pytest.mark.live_provider]


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is not set in environment.",
)
def test_live_openrouter_rag_query(tmp_path: Path):
    from prometheus.rag.pipeline import query_rag
    from prometheus.retrieval.ingestion import ingest_document

    doc_path = tmp_path / "live_doc.txt"
    doc_path.write_text(
        "Prometheus is a NotebookLM-like research system created to empower deep scientific investigation.\n\n"
        "It uses hybrid retrieval combining dense embeddings and BM25 with reciprocal rank fusion.",
        encoding="utf-8",
    )

    doc_id = ingest_document(str(doc_path), document_id="doc-live-test")

    result = query_rag(
        question="What is Prometheus and what retrieval technique does it use?",
        document_id=doc_id,
        top_k=2,
    )

    assert result.has_sufficient_context is True
    assert len(result.answer) > 20
    assert len(result.sources) >= 1
    assert result.sources[0].document_id == doc_id
    print(f"\n[Live OpenRouter Generation Result]\n{result.answer}\n")
    for src in result.sources:
        print(f"  Source [{src.citation_index}]: {src.filename} (score: {src.score})")
