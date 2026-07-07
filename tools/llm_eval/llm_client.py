"""OpenAI-compatible chat-completions client using environment credentials."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class LLMClientError(RuntimeError):
    pass


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


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LLMClientError(f"Missing required environment variable {name}")
    return value


class LLMClient:
    """Minimal caller for ``POST /chat/completions`` compatible APIs."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = (base_url or _env("OPENAI_API_BASE")).rstrip("/")
        self.api_key = api_key or _env("OPENAI_API_KEY")
        self.timeout_s = timeout_s

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
            "max_tokens": config.max_tokens,
            "max_completion_tokens": config.max_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        payload.update(config.extra)

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(f"LLM API request failed: {exc}") from exc

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
