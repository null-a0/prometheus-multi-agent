"""
Literature Review Agent (per docs/phase9-multi-agent-design.md, section 4.4)

Responsibility:
  "How do I systematically synthesize research on this topic?" The one
  Research Mode agent; the only new agent allowed to spawn the existing
  Phase 3 sub-orchestrator (Planner -> Researcher/Retriever -> Critic <->
  Synthesizer debate loop), one run per identified sub-topic.

Allowed to call:
  Everything the Research Agent can, plus
  prometheus.graph.sub_orchestrator.build_sub_orchestrator_graph per
  sub-topic.

Must not:
  Silently drop a sub-topic whose Critic loop hit the max debate-round cap
  without recording it in open_gaps.

TODO (Phase 13):
  - Define def run_literature_review_agent(request: LiteratureReviewRequest) -> LiteratureReviewResult
  - Decompose request.question into up to request.max_sub_topics sub-topics
  - Spawn a sub-orchestrator run per sub-topic (reuses Phase 3 as-is)
  - Cluster/compare sub-topic findings into request.sections (or a default
    structure if request.sections is None), formatted per
    request.citation_style
  - A sub-topic whose sub-orchestrator run exceeds the Phase 3 max debate
    rounds gets an open_gaps entry instead of a synthesized section
"""
from __future__ import annotations

from prometheus.agents.contracts import LiteratureReviewRequest, LiteratureReviewResult


def run_literature_review_agent(request: LiteratureReviewRequest) -> LiteratureReviewResult:
    """Entry point. A research question (+ optional format spec) in, a
    structured, evidence-cited literature review out."""
    raise NotImplementedError("Phase 13: implement Literature Review Agent")