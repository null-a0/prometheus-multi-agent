"""Backward-compatible OpenRouter client facade.

Provider-specific SDK work lives in ``prometheus.models.providers.openrouter``.
Application code should call ``ModelGateway`` instead of this compatibility API.
"""
from __future__ import annotations

from prometheus.llm.base import BaseLLMClient, LLMConfigError, LLMProviderError
from prometheus.models.providers.openrouter import OpenRouterProvider
from prometheus.models.types import ModelGatewayError, ModelRequest


class OpenRouterClient(BaseLLMClient):
    """Legacy text-only facade over the provider adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._provider = OpenRouterProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        self.model = self._provider.default_model

    @property
    def provider_adapter(self) -> OpenRouterProvider:
        """Expose the gateway-compatible adapter without exposing SDK objects."""
        return self._provider

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._provider.generate(
                ModelRequest(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return response.generated_text
        except ModelGatewayError as exc:
            if exc.category == "authentication_error":
                raise LLMConfigError(str(exc)) from exc
            raise LLMProviderError(str(exc), provider="OpenRouter", status_code=exc.status_code) from exc
