"""Augment benchmark prompts with a structured answer format for LLM eval."""

from __future__ import annotations

ANSWER_FORMAT_SECTION = """\
---
## Response format

Explain your reasoning step by step before committing to an answer. Include the
main comparisons that led to your choice; do not reply with only the answer tag.

When you are ready to commit to a choice, put your final answer on its own line wrapped in tags:

<answer>{example}</answer>

Valid choices: {choices}. Use exactly one letter inside the tags. Always include the answer tag; if space is limited, keep reasoning brief and still output the tag.\
"""


def format_eval_prompt(base_prompt: str, valid_letters: frozenset[str]) -> str:
    letters = ", ".join(sorted(valid_letters))
    example = next(iter(sorted(valid_letters)))
    suffix = ANSWER_FORMAT_SECTION.format(example=example, choices=letters)
    return f"{base_prompt.rstrip()}\n\n{suffix}\n"
