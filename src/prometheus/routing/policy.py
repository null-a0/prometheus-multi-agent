"""Configurable, transparent routing signals."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingPolicy:
    deterministic_patterns: tuple[str, ...] = (
        r"\b(list|show|display)\b.*\b(documents?|files?|sources?)\b",
        r"\b(document|file)\s+(metadata|status)\b",
        r"\b(health|service)\s+(check|status)\b",
    )
    high_patterns: tuple[str, ...] = (
        r"\bwhy\b",
        r"\b(investigate|investigation|research)\b",
        r"\b(competing explanations?|root cause|causal factors?)\b",
        r"\b(recommend|recommendation|trade[- ]?offs?)\b",
        r"\bwhat evidence supports each\b",
    )
    medium_patterns: tuple[str, ...] = (
        r"\b(compare|comparison|contrast|versus|vs\.?|difference|changed|change)\b",
        r"\b(across|between)\b.*\b(reports?|documents?|files?|periods?)\b",
        r"\b(synthesize|synthesis)\b",
    )
    external_patterns: tuple[str, ...] = (
        r"\b(external|outside|market|industry|web|online|latest|recent literature|academic)\b",
        r"\b(externally|public sources?)\b",
    )
    low_patterns: tuple[str, ...] = (
        r"\b(what|who|when|where|which|how much|how many)\b",
        r"\b(summarize|summary|overview|mentioned|stated)\b",
    )
    patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def signals_for(self, name: str) -> tuple[str, ...]:
        if name in self.patterns:
            return self.patterns[name]
        return getattr(self, f"{name}_patterns")
