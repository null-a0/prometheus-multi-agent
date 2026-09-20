"""Construction and citation helpers for retrieved RAG evidence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prometheus.rag.models import Evidence


def build_evidence(
    retrieval_results: Sequence[Mapping[str, Any]],
    retrieval_method: str = "hybrid",
) -> list[Evidence]:
    """Convert ranked retrieval results into typed evidence items.

    The builder copies only provenance and scores present in the retrieval
    result. Missing values remain ``None`` rather than being inferred.
    """
    evidence: list[Evidence] = []
    for result in retrieval_results:
        metadata = dict(result.get("metadata") or {})
        chunk_id = result.get("id")
        document_id = (
            metadata.get("document_id")
            or metadata.get("source_id")
            or metadata.get("project_id")
        )
        document_name = metadata.get("filename")
        if document_name is None and metadata.get("url"):
            document_name = metadata["url"]

        evidence.append(
            Evidence(
                evidence_id=str(chunk_id) if chunk_id is not None else None,
                document_id=str(document_id) if document_id is not None else None,
                document_name=str(document_name) if document_name is not None else None,
                chunk_id=str(chunk_id) if chunk_id is not None else None,
                text=str(result.get("text", "")),
                page_number=metadata.get("page_number"),
                source_type=metadata.get("source_type") or metadata.get("source"),
                retrieval_method=retrieval_method,
                retrieval_score=_as_float(result.get("score")),
                reranker_score=_as_float(result.get("rerank_score")),
                final_score=_as_float(result.get("final_score")),
                metadata=metadata,
            )
        )
    return evidence


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None
