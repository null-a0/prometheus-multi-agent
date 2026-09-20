"""
Quiz Agent (per docs/phase9-multi-agent-design.md, section 4.3)

Responsibility:
  "Do I actually understand it?" Generates assessment questions grounded
  in a prior Research Agent result (and optionally an Explainer result).

Allowed to call:
  The model gateway/LLM client only. Same restriction as Explainer â€” no
  independent retrieval.

Must not:
  Invent questions whose source_evidence_ids aren't a subset of
  request.research's claim evidence ids.

TODO (Phase 12):
  - Define def run_quiz_agent(request: QuizRequest) -> QuizResult
  - Generate request.question_count questions of request.question_type at
    request.difficulty, each citing source_evidence_ids drawn from
    request.research.claims[*].supporting_evidence_ids
  - If research has fewer claims than question_count can be grounded
    against, reduce question_count and say so in the result rather than
    inventing ungrounded questions
"""
from __future__ import annotations

from prometheus.agents.contracts import QuizRequest, QuizResult


def run_quiz_agent(request: QuizRequest) -> QuizResult:
    """Entry point. A prior ResearchResult (+ optional ExplanationResult) in,
    grounded quiz questions out."""
    raise NotImplementedError("Phase 12: implement Quiz Agent")