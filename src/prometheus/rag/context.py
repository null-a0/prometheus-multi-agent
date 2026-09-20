from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prometheus.rag.evidence import build_evidence
from prometheus.rag.models import AcademicSourceRecord, Evidence, SourceRecord
from prometheus.retrieval.sources.paper_record import PaperRecord


def build_context(ranked_chunks: Sequence[Mapping[str, Any] | Evidence]) -> tuple[str, list[SourceRecord]]:
    """Format evidence into numbered prompt blocks and structured citations.

    Parameters
    ----------
    ranked_chunks : Sequence[dict]
        Typed evidence or legacy chunk dictionaries returned by retrieval and
        reranking. Legacy dictionaries are converted before citation mapping.

    Returns
    -------
    tuple[str, list[SourceRecord]]
        1. Context string ready for prompt injection with citations [1], [2], ...
        2. List of typed SourceRecords preserving full document provenance.
    """
    if not ranked_chunks:
        return "", []

    evidence = [
        item if isinstance(item, Evidence) else build_evidence([item])[0]
        for item in ranked_chunks
    ]
    return build_context_from_evidence(evidence)


def build_context_from_evidence(evidence: Sequence[Evidence]) -> tuple[str, list[SourceRecord]]:
    """Build prompt context and citations from already-normalized evidence."""
    context_blocks: list[str] = []
    sources: list[SourceRecord] = []

    for idx, item in enumerate(evidence, start=1):
        metadata = item.metadata
        text = item.text.strip()
        doc_id = item.document_id or "unknown"
        filename = item.document_name or "document"
        file_type = metadata.get("file_type")
        page_num = item.page_number
        slide_num = metadata.get("slide_number")
        heading = metadata.get("heading")
        score = item.final_score or item.reranker_score or item.retrieval_score

        # Build human-readable location tag
        location_parts = []
        if page_num is not None:
            location_parts.append(f"Page {page_num}")
        if slide_num is not None:
            location_parts.append(f"Slide {slide_num}")
        if heading:
            location_parts.append(f"Section: '{heading}'")

        location_str = f" ({', '.join(location_parts)})" if location_parts else ""

        # Format context passage
        passage_header = f"[{idx}] Source: {filename}{location_str} [DocID: {doc_id}]"
        context_blocks.append(f"{passage_header}\n{text}")

        # Create structured source record
        sources.append(
            SourceRecord(
                citation_index=idx,
                document_id=str(doc_id),
                filename=str(filename),
                file_type=file_type,
                page_number=page_num,
                slide_number=slide_num,
                heading=heading,
                chunk_id=str(item.chunk_id or item.evidence_id or f"chunk-{idx}"),
                score=float(score) if score is not None else None,
                text_snippet=text,
            )
        )

    full_context_str = "\n\n".join(context_blocks)
    return full_context_str, sources


_ACADEMIC_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")


def build_academic_context(
    records: Sequence[PaperRecord],
) -> tuple[str, list[AcademicSourceRecord]]:
    """Format OpenAlex PaperRecords into labeled academic context blocks and typed AcademicSourceRecords.

    Parameters
    ----------
    records : Sequence[PaperRecord]
        List of PaperRecord objects returned by search_openalex().

    Returns
    -------
    tuple[str, list[AcademicSourceRecord]]
        1. Formatted academic literature context string labeled with [A], [B], ...
        2. List of typed AcademicSourceRecords preserving full bibliographic metadata.
    """
    if not records:
        return "", []

    context_blocks: list[str] = []
    academic_sources: list[AcademicSourceRecord] = []

    for idx, paper in enumerate(records):
        label = f"[{_ACADEMIC_LABELS[idx]}]" if idx < len(_ACADEMIC_LABELS) else f"[Ref-{idx + 1}]"
        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown Authors"
        year_str = str(paper.year) if paper.year is not None else "n.d."
        doi_or_url = paper.doi or paper.url or "N/A"
        abstract_text = paper.abstract.strip() if paper.abstract and paper.abstract.strip() else "No abstract provided."

        header = (
            f"{label} Paper: \"{paper.title}\"\n"
            f"    Authors: {authors_str} ({year_str})\n"
            f"    DOI / Link: {doi_or_url} [OpenAlex ID: {paper.source_id}]"
        )
        context_blocks.append(f"{header}\n    Abstract: {abstract_text}")

        academic_sources.append(
            AcademicSourceRecord(
                citation_label=label,
                source=paper.source,
                source_id=paper.source_id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                doi=paper.doi,
                url=paper.url,
                abstract=paper.abstract,
            )
        )

    return "\n\n".join(context_blocks), academic_sources

