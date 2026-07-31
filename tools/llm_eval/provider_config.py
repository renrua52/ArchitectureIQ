"""Environment-only configuration for OpenAI-compatible model providers."""

from __future__ import annotations

import os


class ProviderConfigError(RuntimeError):
    """Raised when an API endpoint or credential is not configured."""


def _first_nonempty(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def provider_base_url(*, default: str | None = None) -> str:
    """Return a configured OpenAI-compatible base URL without a trailing slash."""
    value = _first_nonempty("V_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL")
    if value:
        return value.rstrip("/")
    if default:
        return default.rstrip("/")
    raise ProviderConfigError(
        "Missing API base URL. Set V_API_BASE or OPENAI_API_BASE."
    )


def provider_api_key() -> str:
    """Return a local credential without ever embedding it in source code."""
    value = _first_nonempty("V_API_KEY", "OPENAI_API_KEY", "OPENAI_API_KEY_GPTGE")
    if value:
        return value
    raise ProviderConfigError(
        "Missing API credential. Set V_API_KEY or OPENAI_API_KEY."
    )
