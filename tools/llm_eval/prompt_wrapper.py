"""Augment benchmark prompts with a structured answer format for LLM eval."""

from __future__ import annotations

ANSWER_FORMAT_SECTION = """\
---
## Response format

Think through the problem step by step. You may use as much reasoning as you need.

When you are ready to commit to a choice, put your final answer on its own line wrapped in tags:

<answer>{example}</answer>

Valid choices: {choices}. Use exactly one letter inside the tags. Always include the answer tag; if space is limited, keep reasoning brief and still output the tag.\
"""


def format_eval_prompt(base_prompt: str, valid_letters: frozenset[str]) -> str:
    # Since v1.4 the rendered benchmark prompt asks for <answer>X</answer>
    # itself, so appending this section again would give the model two
    # near-identical response-format blocks. Older prompts on disk (and any
    # prompt built outside the renderer) still get the suffix.
    if "<answer>" in base_prompt:
        return f"{base_prompt.rstrip()}\n"
    letters = ", ".join(sorted(valid_letters))
    example = next(iter(sorted(valid_letters)))
    suffix = ANSWER_FORMAT_SECTION.format(example=example, choices=letters)
    return f"{base_prompt.rstrip()}\n\n{suffix}\n"
