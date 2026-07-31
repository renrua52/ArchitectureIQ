"""OpenAI-compatible chat-completions client using environment credentials."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from provider_config import provider_api_key, provider_base_url


class LLMClientError(RuntimeError):
    pass


_RETRYABLE_HTTP_STATUSES = frozenset({429, *range(500, 600)})


def _default_ssl_context() -> ssl.SSLContext:
    """Use the maintained certifi bundle when the system Python lacks one."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _retry_after_seconds(headers: Any) -> float | None:
    """Return a valid numeric Retry-After value, if the provider supplied one."""
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float = 0.0
    max_tokens: int = 16384
    top_p: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    raw: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    @property
    def assistant_message(self) -> dict[str, Any]:
        return self.raw["choices"][0]["message"]


_MESSAGE_TEXT_KEYS = (
    "reasoning_content",
    "reasoning",
    "thought",
    "thinking",
    "content",
)


def message_text(message: dict[str, Any]) -> str:
    """Merge all visible text fields from an API message before parsing."""
    chunks: list[str] = []
    for key in _MESSAGE_TEXT_KEYS:
        piece = message.get(key)
        if piece:
            chunks.append(str(piece))
    return "\n\n".join(chunks)


def message_parts(message: dict[str, Any]) -> dict[str, str]:
    parts: dict[str, str] = {}
    for key in _MESSAGE_TEXT_KEYS:
        piece = message.get(key)
        if piece:
            parts[key] = str(piece)
    return parts


def _token_limit_payload(config: ModelConfig) -> dict[str, int]:
    """Return a single token-limit field (APIs reject both at once)."""
    if "max_completion_tokens" in config.extra:
        return {"max_completion_tokens": int(config.extra["max_completion_tokens"])}
    if "max_tokens" in config.extra:
        return {"max_tokens": int(config.extra["max_tokens"])}

    param = os.environ.get("OPENAI_MAX_TOKENS_PARAM", "max_tokens").strip()
    if param == "max_completion_tokens":
        return {"max_completion_tokens": config.max_tokens}
    if param != "max_tokens":
        raise LLMClientError(
            "OPENAI_MAX_TOKENS_PARAM must be 'max_tokens' or 'max_completion_tokens', "
            f"got {param!r}"
        )
    return {"max_tokens": config.max_tokens}


class LLMClient:
    """Minimal caller for ``POST /chat/completions`` compatible APIs."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 8.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_base_delay_s < 0 or retry_max_delay_s < 0:
            raise ValueError("retry delays must be non-negative")
        self.base_url = (base_url or provider_base_url()).rstrip("/")
        self.api_key = api_key or provider_api_key()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_max_delay_s = retry_max_delay_s
        self._sleep = sleep_fn
        self._ssl_context = ssl_context or _default_ssl_context()

    def _retry_delay(self, retry_index: int, retry_after_s: float | None = None) -> float:
        backoff = min(
            self.retry_max_delay_s,
            self.retry_base_delay_s * (2**retry_index),
        )
        return min(self.retry_max_delay_s, max(backoff, retry_after_s or 0.0))

    def complete(self, prompt: str, config: ModelConfig) -> LLMCompletion:
        return self.complete_conversation([{"role": "user", "content": prompt}], config)

    def complete_conversation(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
    ) -> LLMCompletion:
        payload: dict[str, Any] = {
            "model": config.name,
            "messages": messages,
            "temperature": config.temperature,
            **_token_limit_payload(config),
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        for key, value in config.extra.items():
            if key not in {"max_tokens", "max_completion_tokens"}:
                payload[key] = value

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for retry_index in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_s,
                    context=self._ssl_context,
                ) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in _RETRYABLE_HTTP_STATUSES or retry_index >= self.max_retries:
                    raise LLMClientError(f"LLM API HTTP {exc.code}: {detail}") from exc
                delay = self._retry_delay(retry_index, _retry_after_seconds(exc.headers))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if retry_index >= self.max_retries:
                    raise LLMClientError(f"LLM API request failed: {exc}") from exc
                delay = self._retry_delay(retry_index)
            self._sleep(delay)

        try:
            choice = raw["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected LLM response shape: {raw!r}") from exc

        return LLMCompletion(
            content=message_text(message),
            raw=raw,
            finish_reason=choice.get("finish_reason"),
            usage=raw.get("usage"),
        )
