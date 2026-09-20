"""
Explainer Agent (per docs/phase9-multi-agent-design.md, section 4.2)

Responsibility:
  "How can I understand this easily?" Simplifies a prior Research Agent
  result into eli5/beginner/technical explanations.

Allowed to call:
  The model gateway/LLM client only.

Must not:
  Call retrieval, OpenAlex, or the Findings DB directly â€” its only source
  of truth is the ResearchResult it's handed. Must not introduce evidence
  ids that aren't already present in that ResearchResult.

TODO (Phase 11):
  - Define def run_explainer_agent(request: ExplainerRequest) -> ExplanationResult
  - Per level in request.levels: prompt for intuition/example/explanation/
    key_takeaway grounded strictly in request.research.claims
  - If request.research.claims is empty, return levels that say evidence
    was insufficient rather than explaining from parametric knowledge
  - source_evidence_ids must be the union of ids actually cited across
    levels, and must be a subset of request.research's own ids
"""
from __future__ import annotations

from prometheus.agents.contracts import ExplainerRequest, ExplanationResult


def run_explainer_agent(request: ExplainerRequest) -> ExplanationResult:
    """Entry point. A prior ResearchResult in, level-adjusted explanations out."""
    raise NotImplementedError("Phase 11: implement Explainer Agent")