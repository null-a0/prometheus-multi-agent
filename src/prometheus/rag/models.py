"""
Pydantic data contracts for the Prometheus RAG generation layer.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from prometheus.routing.models import RoutingMetadata


class Evidence(BaseModel):
    """One retrieved passage with provenance and ranking information."""

    evidence_id: str | None = Field(default=None, description="Stable identifier for the retrieved evidence item.")
    document_id: str | None = Field(default=None, description="Identifier of the source document, when available.")
    document_name: str | None = Field(default=None, description="Source document name, when available.")
    chunk_id: str | None = Field(default=None, description="Identifier of the retrieved chunk, when available.")
    text: str = Field(description="Retrieved evidence text.")
    page_number: int | None = Field(default=None, description="1-based source page, when available.")
    source_type: str | None = Field(default=None, description="Source classification, when available.")
    retrieval_method: str | None = Field(default=None, description="Retrieval path used to produce this item.")
    retrieval_score: float | None = Field(default=None, description="Fusion or retrieval score, when available.")
    reranker_score: float | None = Field(default=None, description="Cross-encoder reranker score, when available.")
    final_score: float | None = Field(default=None, description="Final score after ranking filters, when available.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive provenance metadata.")


class SourceRecord(BaseModel):
    """Structured attribution for a cited evidence chunk."""

    citation_index: int = Field(description="1-based citation index corresponding to [1], [2] in context.")
    document_id: str = Field(description="Identifier of the document this chunk belongs to.")
    filename: str = Field(default="unknown", description="Original filename or document name.")
    file_type: str | None = Field(default=None, description="Format of the source document (pdf, docx, etc.).")
    page_number: int | None = Field(default=None, description="1-based page number if available (PDF).")
    slide_number: int | None = Field(default=None, description="1-based slide number if available (PPTX).")
    heading: str | None = Field(default=None, description="Heading context under which this chunk falls.")
    chunk_id: str = Field(description="Unique chunk identifier.")
    score: float | None = Field(default=None, description="Final relevance/rerank score.")
    text_snippet: str = Field(description="Snippet or full text of the evidence passage.")


class AcademicSourceRecord(BaseModel):
    """Structured attribution for an external academic literature source (e.g. OpenAlex)."""

    citation_label: str = Field(description="Citation label corresponding to [A], [B], etc. in the prompt.")
    source: str = Field(default="openalex", description="Originating academic registry or source.")
    source_id: str = Field(description="Identifier within the academic source.")
    title: str = Field(description="Title of the paper.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    year: int | None = Field(default=None, description="Publication year.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    url: str | None = Field(default=None, description="Canonical web URL.")
    abstract: str | None = Field(default=None, description="Abstract text snippet.")


class RagResult(BaseModel):
    """Complete response returned by the Prometheus RAG pipeline."""

    answer: str = Field(description="Factually grounded answer synthesized by the LLM.")
    routing: RoutingMetadata | None = Field(
        default=None,
        description="Concise task-routing metadata for this request.",
    )
    citations: list[SourceRecord] = Field(
        default_factory=list,
        description="Citation records mapped to the evidence items used in the answer.",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence items retrieved and made available to answer generation.",
    )
    sources: list[SourceRecord] = Field(default_factory=list, description="List of local document source attributions.")
    academic_sources: list[AcademicSourceRecord] = Field(
        default_factory=list,
        description="External academic literature records (e.g. from OpenAlex) when enabled.",
    )
    retrieved_chunks: list[dict] = Field(default_factory=list, description="Raw retrieved and reranked chunks.")
    has_sufficient_context: bool = Field(
        default=True,
        description="False if the retrieval step yielded no evidence or if the context was insufficient.",
    )
    evidence_status: Literal["no_evidence", "insufficient_evidence", "sufficient"] = Field(
        default="sufficient",
        description="Whether retrieval produced no evidence, insufficient evidence, or sufficient context.",
    )


class RagQueryRequest(BaseModel):
    """Request payload for querying the RAG pipeline."""

    question: str = Field(..., description="User's query string.")
    document_id: str | None = Field(
        default=None,
        description="Optional document_id to isolate search strictly to a single document/notebook.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of evidence chunks to retrieve and consider.")
    include_academic_evidence: bool = Field(
        default=False,
        description="When True, queries OpenAlex for up to 3 academic papers to complement local document evidence.",
    )
    openalex_max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of external academic papers to query when academic evidence is enabled.",
    )
