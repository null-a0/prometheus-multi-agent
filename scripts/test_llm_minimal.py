"""
Diagnostic script: Minimal real OpenRouter / Gemini generation request.

Verifies:
1. API key is loaded without exposing the secret.
2. The configured Gemini model on OpenRouter can generate a minimal completion.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

from prometheus.config import settings
from prometheus.models.gateway import ModelGateway
from prometheus.models.providers.openrouter import OpenRouterProvider
from prometheus.models.types import ModelGatewayError


def mask_key(key: str) -> str:
    """Safely mask key for display, showing only length and leading characters."""
    if not key:
        return "<EMPTY>"
    prefix = key[:6] if len(key) >= 6 else key[:2]
    return f"{prefix}...[HIDDEN, length={len(key)}]"


def main() -> int:
    print("=" * 60)
    print("PROMETHEUS -- MINIMAL OPENROUTER / GEMINI CONNECTIVITY TEST")
    print("=" * 60)

    # 1. Verify key loading without printing the secret
    raw_key = os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
    model = os.getenv("OPENROUTER_MODEL") or settings.openrouter_model
    base_url = os.getenv("OPENROUTER_BASE_URL") or settings.openrouter_base_url

    print(f"Base URL   : {base_url}")
    print(f"Target Model: {model}")
    print(f"API Key    : {mask_key(raw_key)}")

    if not raw_key or not raw_key.strip():
        print("\n[ERROR] OPENROUTER_API_KEY is not set or empty in .env!")
        print("Please add your key to E:\\prometheus\\.env:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        return 1

    # 2. Minimal generation test
    gateway = ModelGateway(
        provider=OpenRouterProvider(api_key=raw_key, model=model, base_url=base_url)
    )
    print("\nSending minimal generation request to OpenRouter...")
    try:
        response = gateway.generate(
            prompt="Hello! Please reply with exactly: 'PROMETHEUS_ONLINE'.",
            system_prompt="You are a concise connectivity verification bot. Output only the requested token.",
            max_tokens=20,
        )
        print("\n[SUCCESS] Generation response received:")
        print(f"  Response text: {response.generated_text.strip()!r}")
        print("\nOpenRouter client and Gemini model are fully operational.")
        return 0

    except ModelGatewayError as gateway_err:
        print(f"\n[MODEL ERROR] ({gateway_err.category})")
        print(f"  Message: {gateway_err}")
        print("\nTroubleshooting tips:")
        print("  - Check that the model name is supported on OpenRouter (e.g. google/gemini-2.0-flash-001)")
        print("  - Verify that your OpenRouter account has active credits / quota")
        print("  - Check network connectivity to https://openrouter.ai")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"\n[UNEXPECTED ERROR] {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
