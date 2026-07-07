"""Parse a multiple-choice letter from an LLM response."""

from __future__ import annotations

import re

_ANSWER_TAG = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)


def parse_choice_letter(response: str, valid_letters: frozenset[str]) -> str | None:
    matches = _ANSWER_TAG.findall(response)
    if not matches:
        return None
    letter = matches[-1].upper()
    if letter in valid_letters:
        return letter
    return None


def split_chain_of_thought(response: str, parsed_letter: str | None) -> str | None:
    del parsed_letter  # answer is extracted from tags, not the last line
    text = response.strip()
    if not text:
        return None
    body = _ANSWER_TAG.sub("", text).strip()
    return body or None
