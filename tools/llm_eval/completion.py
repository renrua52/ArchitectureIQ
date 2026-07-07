"""Fetch a complete model response, continuing when the API stops early."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_client import LLMClient, LLMCompletion, ModelConfig, message_parts
from response_parser import parse_choice_letter

_LENGTH_STOP_REASONS = frozenset({"length", "max_tokens"})

_CONTINUATION_PROMPT = (
    "Your previous response was cut off before you finished. "
    "Continue from where you left off and end with your final choice in "
    "<answer>LETTER</answer> tags."
)


@dataclass(frozen=True)
class ModelExchange:
    model_response: str
    finish_reason: str | None
    usage: dict[str, Any] | None
    continuation_count: int
    truncated: bool
    message_parts: dict[str, str]


def fetch_model_response(
    client: LLMClient,
    prompt: str,
    config: ModelConfig,
    valid_letters: frozenset[str],
    *,
    max_continuations: int = 3,
) -> ModelExchange:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    chunks: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    parts: dict[str, str] = {}
    continuation_count = 0

    for attempt in range(max_continuations + 1):
        completion = client.complete_conversation(messages, config)
        if completion.content:
            chunks.append(completion.content)
        finish_reason = completion.finish_reason
        usage = completion.usage
        parts = message_parts(completion.assistant_message)

        full = "\n\n".join(chunks)
        if parse_choice_letter(full, valid_letters) is not None:
            return ModelExchange(
                model_response=full,
                finish_reason=finish_reason,
                usage=usage,
                continuation_count=continuation_count,
                truncated=False,
                message_parts=parts,
            )

        if finish_reason not in _LENGTH_STOP_REASONS or attempt >= max_continuations:
            break

        continuation_count += 1
        messages.append(completion.assistant_message)
        messages.append({"role": "user", "content": _CONTINUATION_PROMPT})

    full = "\n\n".join(chunks)
    truncated = finish_reason in _LENGTH_STOP_REASONS and parse_choice_letter(full, valid_letters) is None
    return ModelExchange(
        model_response=full,
        finish_reason=finish_reason,
        usage=usage,
        continuation_count=continuation_count,
        truncated=truncated,
        message_parts=parts,
    )
