"""
Research Agent (per docs/phase9-multi-agent-design.md, section 4.1)

Responsibility:
  "What information/evidence do we need?" A single retrieve-assess pass
  (not a debate loop) that turns a topic into grounded evidence + claims.

Allowed to call:
  prometheus.retrieval.hybrid_search, prometheus.retrieval.reranker,
  prometheus.retrieval.sources.* (arXiv/S2/OpenAlex clients),
  prometheus.findings.repository.search_findings.

Must not:
  Simplify/rephrase for a target audience, or generate quiz content. Must
  not fabricate a claim with no supporting_evidence_ids.

TODO (Phase 10):
  - Wire prometheus.retrieval.hybrid_search + reranker for `evidence`
  - Extract Claims from the reranked evidence (LLM call), each with
    non-empty supporting_evidence_ids
  - Query prometheus.retrieval.sources.openalex_client when
    request.include_academic is set
  - Set has_sufficient_evidence=False rather than proceeding when
    retrieval returns nothing usable (mirrors
    prometheus.rag.pipeline.INSUFFICIENT_CONTEXT_MESSAGE)
"""
from __future__ import annotations

from prometheus.agents.contracts import ResearchRequest, ResearchResult


def run_research_agent(request: ResearchRequest) -> ResearchResult:
    """Entry point. topic/document scope in, grounded evidence + claims out."""
    raise NotImplementedError("Phase 10: implement Research Agent")