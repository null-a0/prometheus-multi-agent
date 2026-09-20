"""Narrow deterministic contradiction detection for structured claims."""
from __future__ import annotations

import re
from typing import Protocol

from prometheus.investigation.graph import EvidenceGraph
from prometheus.investigation.models import EvidenceGraphEdge, EvidenceRelationType


class ContradictionDetectorProtocol(Protocol):
    def detect(self, graph: EvidenceGraph) -> list[EvidenceGraphEdge]:
        ...


class ContradictionDetector:
    """Detect only explicit equal-magnitude increase/decrease claim pairs."""

    _CHANGE_PATTERN = re.compile(
        r"^(?P<subject>.+?)\s+(?P<direction>increased|decreased)\s+by\s+(?P<amount>\d+(?:\.\d+)?%?)\.?$",
        re.IGNORECASE,
    )

    def detect(self, graph: EvidenceGraph) -> list[EvidenceGraphEdge]:
        claims = _claims_from_graph(graph)
        parsed = {
            claim_id: match
            for claim_id, claim in claims.items()
            if (match := self._CHANGE_PATTERN.match(" ".join(claim.text.split())))
        }
        edges: list[EvidenceGraphEdge] = []
        for left_id, left_match in parsed.items():
            for right_id, right_match in parsed.items():
                if left_id >= right_id:
                    continue
                same_subject = left_match.group("subject").strip().lower() == right_match.group("subject").strip().lower()
                same_amount = left_match.group("amount") == right_match.group("amount")
                opposite_direction = left_match.group("direction").lower() != right_match.group("direction").lower()
                if same_subject and same_amount and opposite_direction:
                    edges.append(
                        EvidenceGraphEdge(
                            source_id=left_id,
                            target_id=right_id,
                            relation=EvidenceRelationType.CONTRADICTS,
                        )
                    )
        return edges


def _claims_from_graph(graph: EvidenceGraph) -> dict[str, object]:
    claim_ids = {
        edge.target_id
        for edge in graph.get_relations(relation=EvidenceRelationType.SUPPORTS)
    }
    claim_ids.update(
        edge.source_id
        for edge in graph.get_relations(relation=EvidenceRelationType.CONTRADICTS)
    )
    claim_ids.update(
        edge.target_id
        for edge in graph.get_relations(relation=EvidenceRelationType.CONTRADICTS)
    )
    return {
        claim_id: claim
        for claim_id in sorted(claim_ids)
        if (claim := graph.get_claim(claim_id)) is not None
    }
