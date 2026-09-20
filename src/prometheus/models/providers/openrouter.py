"""OpenRouter provider adapter for the model gateway."""
from __future__ import annotations

import logging
import os
from typing import Any

from prometheus.config import settings
from prometheus.models.types import ModelGatewayError, ModelRequest, ProviderResponse

logger = logging.getLogger(__name__)


class OpenRouterProvider:
    """Translate provider-neutral requests to the OpenAI-compatible OpenRouter API."""

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key).strip()
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or settings.openrouter_base_url).strip()
        self.default_model = (model or os.getenv("OPENROUTER_MODEL") or settings.openrouter_model).strip()
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if not self.api_key:
            raise ModelGatewayError(
                category="authentication_error",
                message="Model provider authentication is not configured.",
                provider=self.provider_name,
                model=self.default_model,
            )
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers={
                    "HTTP-Referer": "https://github.com/NamitCodes/prometheus",
                    "X-Title": "Prometheus Knowledge Platform",
                },
            )
        return self._client

    def generate(self, request: ModelRequest) -> ProviderResponse:
        model = request.model or self.default_model
        messages = request.messages
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None and request.max_tokens > 0:
            kwargs["max_tokens"] = request.max_tokens

        try:
            response = self._get_client().chat.completions.create(**kwargs)
            if not response.choices:
                raise ModelGatewayError(
                    category="provider_error",
                    message="Model provider returned no choices.",
                    provider=self.provider_name,
                    model=model,
                )

            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return ProviderResponse(
                generated_text=(getattr(choice.message, "content", None) or "").strip(),
                provider=self.provider_name,
                model=model,
                input_tokens=_usage_value(usage, "prompt_tokens"),
                output_tokens=_usage_value(usage, "completion_tokens"),
                total_tokens=_usage_value(usage, "total_tokens"),
                request_id=getattr(response, "id", None),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except ModelGatewayError:
            raise
        except Exception as exc:
            category = _error_category(exc)
            logger.warning("OpenRouter provider request failed", extra={"category": category, "model": model})
            raise ModelGatewayError(
                category=category,
                message="Model provider request failed.",
                provider=self.provider_name,
                model=model,
                status_code=getattr(exc, "status_code", None),
            ) from exc


def _usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    return int(value) if value is not None else None


def _error_category(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    if status_code in (401, 403):
        return "authentication_error"
    if status_code == 429:
        return "rate_limit"
    if "timeout" in name:
        return "timeout"
    if status_code in (400, 422):
        return "invalid_request"
    if status_code in (500, 502, 503, 504):
        return "unavailable"
    if status_code is not None:
        return "provider_error"
    return "unknown"
