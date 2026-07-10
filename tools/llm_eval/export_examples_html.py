"""Run a few ArchitectureIQ questions and export a self-contained HTML page.

The page is meant for easy sharing: open ``index.html`` directly in a browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from completion import fetch_model_response
from llm_client import LLMClient, ModelConfig
from question_loader import QuestionItem, list_questions
from response_parser import parse_choice_letter, split_chain_of_thought


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExampleResult:
    question_id: str
    family: str
    dataset_id: str
    question_type: str
    prompt_text: str
    question_dir: str
    parsed_letter: str | None
    ground_truth_letter: str
    correct: bool
    visible_rationale: str
    answer_content: str
    finish_reason: str | None
    truncated: bool
    continuation_count: int
    usage: dict[str, Any] | None
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "family": self.family,
            "dataset_id": self.dataset_id,
            "question_type": self.question_type,
            "prompt_text": self.prompt_text,
            "question_dir": self.question_dir,
            "parsed_letter": self.parsed_letter,
            "ground_truth_letter": self.ground_truth_letter,
            "correct": self.correct,
            "visible_rationale": self.visible_rationale,
            "answer_content": self.answer_content,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "continuation_count": self.continuation_count,
            "usage": self.usage,
            "raw_response": self.raw_response,
        }


def _default_data_root() -> Path:
    return ROOT / "data"


def _default_out_dir() -> Path:
    return ROOT / "outputs" / "gpt54_bigram_examples"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _find_items(data_root: Path, question_ids: list[str]) -> list[QuestionItem]:
    wanted = set(question_ids)
    items_by_id = {item.question_id: item for item in list_questions(data_root)}
    missing = [qid for qid in question_ids if qid not in items_by_id]
    if missing:
        raise FileNotFoundError(f"Question ids not found under {data_root}: {', '.join(missing)}")
    items = [items_by_id[qid] for qid in question_ids]
    non_bigram = [item.question_id for item in items if str(item.question.get("family")) != "bigram_lm"]
    if non_bigram:
        raise ValueError(f"These questions are not bigram_lm items: {', '.join(non_bigram)}")
    return items


def _example_prompt(item: QuestionItem) -> str:
    letters = ", ".join(sorted(item.valid_letters))
    instructions = f"""
---
## Response format

In the visible answer text, include exactly this structure:

Rationale:
- bullet 1
- bullet 2
- bullet 3

Then put the final choice on its own line wrapped in tags:
<answer>A</answer>

Requirements:
- Keep the rationale concise and concrete.
- Base it on architecture, optimizer, and sample-budget reasoning.
- Do not omit the answer tag.
- Valid choices: {letters}
"""
    return f"{item.prompt_text.rstrip()}\n{instructions}"


def _clean_visible_rationale(text: str | None) -> str:
    if not text:
        return "No visible rationale was returned in the normal answer text."
    cleaned = text.replace("<!-- -->", "").strip()
    if not cleaned:
        return "No visible rationale was returned in the normal answer text."
    return cleaned


def _run_one(item: QuestionItem, client: LLMClient, config: ModelConfig) -> ExampleResult:
    prompt = _example_prompt(item)
    exchange = fetch_model_response(client, prompt, config, item.valid_letters)
    parsed = parse_choice_letter(exchange.model_response, item.valid_letters)
    visible_rationale = _clean_visible_rationale(split_chain_of_thought(exchange.model_response, parsed))
    answer_content = exchange.message_parts.get("content", "")
    return ExampleResult(
        question_id=item.question_id,
        family=str(item.question.get("family", "?")),
        dataset_id=str(item.question.get("dataset_id", "?")),
        question_type=str(item.question.get("type", "?")),
        prompt_text=item.prompt_text,
        question_dir=str(item.question_dir),
        parsed_letter=parsed,
        ground_truth_letter=item.correct_letter,
        correct=parsed == item.correct_letter if parsed is not None else False,
        visible_rationale=visible_rationale,
        answer_content=answer_content,
        finish_reason=exchange.finish_reason,
        truncated=exchange.truncated,
        continuation_count=exchange.continuation_count,
        usage=exchange.usage,
        raw_response=exchange.model_response,
    )


def _usage_text(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "-"
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    total_tokens = usage.get("total_tokens", "?")
    return f"prompt {prompt_tokens} / completion {completion_tokens} / total {total_tokens}"


def _render_example_card(result: ExampleResult) -> str:
    status_class = "correct" if result.correct else "wrong"
    status_text = "Correct" if result.correct else "Wrong"
    parsed = result.parsed_letter or "Unparsed"
    return f"""
    <section class="card">
      <div class="card-head">
        <div>
          <div class="eyebrow">{escape(result.family)} · {escape(result.dataset_id)} · {escape(result.question_type)}</div>
          <h2>{escape(result.question_id)}</h2>
        </div>
        <div class="status {status_class}">{status_text}</div>
      </div>

      <div class="meta-grid">
        <div class="meta-item"><span>Model answer</span><strong>{escape(parsed)}</strong></div>
        <div class="meta-item"><span>Ground truth</span><strong>{escape(result.ground_truth_letter)}</strong></div>
        <div class="meta-item"><span>API finish</span><strong>{escape(result.finish_reason or "-")}</strong></div>
        <div class="meta-item"><span>Token usage</span><strong>{escape(_usage_text(result.usage))}</strong></div>
      </div>

      <div class="section">
        <h3>Question</h3>
        <pre>{escape(result.prompt_text)}</pre>
      </div>

      <div class="section">
        <h3>Visible rationale</h3>
        <pre>{escape(result.visible_rationale)}</pre>
      </div>

      <div class="section">
        <h3>Raw visible answer field</h3>
        <pre>{escape(result.answer_content or "(empty)")}</pre>
      </div>
    </section>
    """


def _render_html(*, title: str, model_name: str, created_at: str, results: list[ExampleResult]) -> str:
    cards = "\n".join(_render_example_card(result) for result in results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0b1020;
      --panel: #131a2c;
      --panel-border: #29324a;
      --text: #e8edf7;
      --muted: #9aa7bd;
      --accent: #7dd3fc;
      --good: #16a34a;
      --bad: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.55 Inter, Segoe UI, Arial, sans-serif;
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--panel-border);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .stack {{
      display: grid;
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 20px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .card-head h2 {{
      margin: 4px 0 0;
      font-size: 20px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .status {{
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 600;
      white-space: nowrap;
      border: 1px solid currentColor;
    }}
    .status.correct {{ color: #86efac; }}
    .status.wrong {{ color: #fca5a5; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .meta-item {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .meta-item span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .meta-item strong {{
      font-size: 13px;
      word-break: break-word;
    }}
    .section {{
      margin-top: 16px;
    }}
    .section h3 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 8px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0a0f1d;
      border: 1px solid var(--panel-border);
      font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>{escape(title)}</h1>
      <p>Model: {escape(model_name)} · Created: {escape(created_at)} · Questions: {len(results)}</p>
    </header>
    <div class="stack">
      {cards}
    </div>
  </main>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a few questions and export a self-contained HTML page.")
    parser.add_argument("data_root", nargs="?", default=str(_default_data_root()), help="Data root (default: data)")
    parser.add_argument("--out-dir", default=str(_default_out_dir()), help="Output directory for HTML and JSON")
    parser.add_argument("--model", default="gpt-5.4", help="Model name passed to the chat API")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--question-id",
        action="append",
        required=True,
        dest="question_ids",
        help="Question id to include. Repeat for multiple questions.",
    )
    parser.add_argument("--title", default="GPT-5.4 Bigram Examples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    items = _find_items(data_root, list(args.question_ids))
    client = LLMClient()
    config = ModelConfig(
        name=args.model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_p=args.top_p,
    )

    results = [_run_one(item, client, config) for item in items]
    created_at = _utc_now_iso()

    manifest = {
        "created_at": created_at,
        "model": config.to_dict(),
        "data_root": str(data_root),
        "question_ids": [result.question_id for result in results],
        "results": [result.to_dict() for result in results],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index.html").write_text(
        _render_html(title=args.title, model_name=args.model, created_at=created_at, results=results),
        encoding="utf-8",
    )
    print(f"Wrote HTML to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
