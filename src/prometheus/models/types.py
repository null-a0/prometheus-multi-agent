"""Provider-neutral model gateway contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral generation request."""

    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    investigation_id: str | None = None
    task_id: str | None = None
    budget_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized provider response before gateway metadata is added."""

    generated_text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral response returned by ModelGateway."""

    generated_text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    request_id: str | None = None
    finish_reason: str | None = None
    status: str = "success"


class ModelProvider(Protocol):
    provider_name: str
    default_model: str

    def generate(self, request: ModelRequest) -> ProviderResponse:
        """Generate text using one provider-specific adapter."""


@dataclass
class ModelGatewayError(Exception):
    """Provider-independent model invocation failure."""

    category: str
    message: str
    provider: str = "unknown"
    model: str = "unknown"
    status_code: int | None = None
    usage_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message
