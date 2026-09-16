"""OpenAI-compatible chat-completions client using environment credentials."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class LLMClientError(RuntimeError):
    pass


class _PostPreservingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep chat-completion requests as POST across relay redirects."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if req.get_method() == "POST" and code in {301, 302, 307, 308}:
            redirected_headers = {
                key: value
                for key, value in req.headers.items()
                if key.lower() != "content-length"
            }
            return urllib.request.Request(
                newurl,
                data=req.data,
                headers=redirected_headers,
                origin_req_host=req.origin_req_host,
                unverifiable=True,
                method="POST",
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LLMClientError(f"{name} must be a boolean value, got {value!r}")


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


def _read_streaming_response(response: Any) -> dict[str, Any]:
    """Collect an OpenAI-compatible SSE response into the non-streaming shape."""
    metadata: dict[str, Any] = {}
    message: dict[str, Any] = {"role": "assistant"}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_choice = False
    plain_lines: list[str] = []

    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            if not line.startswith("event:"):
                plain_lines.append(line)
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Invalid streaming JSON event: {data!r}") from exc
        if not isinstance(chunk, dict):
            raise LLMClientError(f"Unexpected streaming event shape: {chunk!r}")
        error = chunk.get("error")
        if error:
            raise LLMClientError(f"LLM API error response: {error!r}")

        for key, value in chunk.items():
            if key not in {"choices", "usage"}:
                metadata[key] = value
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]

        choices = chunk.get("choices")
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMClientError(f"Unexpected streaming choice shape: {choice!r}")
        saw_choice = True
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            raise LLMClientError(f"Unexpected streaming delta shape: {delta!r}")
        if delta.get("role"):
            message["role"] = str(delta["role"])
        for key in _MESSAGE_TEXT_KEYS:
            piece = delta.get(key)
            if piece:
                message[key] = message.get(key, "") + str(piece)

    if not saw_choice:
        if plain_lines:
            try:
                raw = json.loads("\n".join(plain_lines))
            except json.JSONDecodeError as exc:
                raise LLMClientError("Streaming response contained no choices") from exc
            if isinstance(raw, dict):
                return raw
        raise LLMClientError("Streaming response contained no choices")
    raw = {
        **metadata,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        raw["usage"] = usage
    return raw


class LLMClient:
    """Minimal caller for ``POST /chat/completions`` compatible APIs."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        stream: bool | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = (base_url or _env("OPENAI_API_BASE")).rstrip("/")
        self.api_key = api_key or _env("OPENAI_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.stream = _bool_env("OPENAI_STREAM") if stream is None else stream

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
        if self.stream:
            payload["stream"] = True

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
        opener = urllib.request.build_opener(_PostPreservingRedirectHandler())
        for attempt in range(self.max_retries + 1):
            try:
                with opener.open(request, timeout=self.timeout_s) as response:
                    if self.stream:
                        raw = _read_streaming_response(response)
                    else:
                        raw = json.loads(response.read().decode("utf-8"))
                error = raw.get("error") if isinstance(raw, dict) else None
                if not isinstance(error, dict):
                    break
                try:
                    error_code = int(error.get("code", 0))
                except (TypeError, ValueError):
                    error_code = 0
                retryable = error_code == 429 or error_code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise LLMClientError(f"LLM API error response: {error!r}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise LLMClientError(f"LLM API HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= self.max_retries:
                    raise LLMClientError(f"LLM API request failed: {exc}") from exc
            if attempt < self.max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt) + random.random()))

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
