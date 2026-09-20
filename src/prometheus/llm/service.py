"""
High-level LLM service factory and generation functions.
"""
from __future__ import annotations

import os

from prometheus.llm.base import BaseLLMClient
from prometheus.llm.mock import MockLLMClient
from prometheus.llm.openrouter import OpenRouterClient

_default_client: BaseLLMClient | None = None


def get_llm_client(force_mock: bool = False) -> BaseLLMClient:
    """Get the active LLM client based on configuration.

    If force_mock is True, or if the environment variable PROMETHEUS_MOCK_LLM=true,
    returns a deterministic MockLLMClient. Otherwise returns OpenRouterClient.
    """
    global _default_client
    if force_mock or os.getenv("PROMETHEUS_MOCK_LLM", "").lower() in ("true", "1", "yes"):
        return MockLLMClient()

    if _default_client is None:
        _default_client = OpenRouterClient()
    return _default_client


def generate_answer(
    question: str,
    context: str,
    system_prompt: str | None = None,
    client: BaseLLMClient | None = None,
) -> str:
    """Convenience helper to generate an answer from a question and context."""
    from prometheus.models.gateway import LegacyClientProvider, ModelGateway, get_model_gateway

    prompt = f"Context Information:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    gateway = (
        ModelGateway(provider=LegacyClientProvider(client))
        if client is not None
        else get_model_gateway()
    )
    return gateway.generate(prompt=prompt, system_prompt=system_prompt).generated_text
