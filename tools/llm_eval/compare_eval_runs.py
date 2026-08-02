#!/usr/bin/env python3
"""Compare multiple ArchitectureIQ LLM eval runs and export a self-contained report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_loader import QuestionItem, list_questions  # noqa: E402


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "so",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "use",
    "using",
    "was",
    "were",
    "which",
    "with",
}

SIGNAL_KEYWORDS = {
    "metric_comparison": {
        "loss",
        "mse",
        "ce",
        "error",
        "accuracy",
        "metric",
        "validation",
        "test",
        "train",
        "curve",
        "perplexity",
    },
    "capacity_regularization": {
        "overfit",
        "underfit",
        "capacity",
        "parameter",
        "parameters",
        "regularization",
        "dropout",
        "depth",
        "width",
        "bottleneck",
    },
    "architecture_pattern": {
        "attention",
        "residual",
        "skip",
        "convolution",
        "conv",
        "linear",
        "mlp",
        "layer",
        "layers",
        "hidden",
        "embedding",
    },
    "bigram_probability": {
        "bigram",
        "probability",
        "transition",
        "token",
        "tokens",
        "context",
        "sequence",
        "likelihood",
        "count",
        "counts",
        "conditional",
    },
    "feature_interaction": {
        "interaction",
        "feature",
        "features",
        "variable",
        "variables",
        "cross",
        "nonlinear",
        "joint",
        "combine",
        "combination",
        "multivariate",
    },
    "trend_shape": {
        "slope",
        "curvature",
        "smooth",
        "plateau",
        "oscillation",
        "monotonic",
        "trend",
        "shape",
        "spike",
        "flat",
    },
}

SIGNAL_LABELS = {
    "metric_comparison": "metric comparison",
    "capacity_regularization": "capacity and regularization",
    "architecture_pattern": "architecture pattern matching",
    "bigram_probability": "bigram probability reasoning",
    "feature_interaction": "feature interaction reasoning",
    "trend_shape": "trend and shape reading",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class RunData:
    label: str
    run_dir: Path
    manifest: dict[str, Any]
    results: dict[str, dict[str, Any]]


def _load_questions(data_root: Path) -> tuple[list[str], dict[str, QuestionItem]]:
    items = list_questions(data_root)
    return [item.question_id for item in items], {item.question_id: item for item in items}


def _load_run(label: str, run_dir: Path) -> RunData:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "results").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[payload["question_id"]] = payload
    return RunData(label=label, run_dir=run_dir, manifest=manifest, results=results)


def _accuracy(correct: int, total: int) -> float | None:
    return (correct / total) if total else None


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _clean_reasoning(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<answer>\s*[A-Z]\s*</answer>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _signal_hits(text: str) -> set[str]:
    found: set[str] = set()
    lowered = {tok.lower() for tok in _tokens(text)}
    for signal, keywords in SIGNAL_KEYWORDS.items():
        if lowered & keywords:
            found.add(signal)
    return found


def _top_signal_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for signal in _signal_hits(_clean_reasoning(row.get("chain_of_thought") or row.get("model_response"))):
            counts[signal] += 1
    total = len(rows)
    output: list[dict[str, Any]] = []
    for signal, count in counts.most_common(3):
        output.append(
            {
                "signal": signal,
                "label": SIGNAL_LABELS.get(signal, signal),
                "count": count,
                "share": _accuracy(count, total),
            }
        )
    return output


def _top_tokens(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = _clean_reasoning(row.get("chain_of_thought") or row.get("model_response"))
        for tok in _tokens(text):
            if tok in STOPWORDS or len(tok) < 4:
                continue
            counts[tok] += 1
    return [{"token": token, "count": count} for token, count in counts.most_common(10)]


def _run_summary(run: RunData, question_ids: list[str], question_map: dict[str, QuestionItem]) -> dict[str, Any]:
    rows = [run.results[qid] for qid in question_ids]
    parsed = [row for row in rows if row.get("parsed_letter")]
    correct_rows = [row for row in parsed if row.get("correct")]
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finish_counts: Counter[str] = Counter()
    continuation_hist: Counter[int] = Counter()

    for row in rows:
        family = str(question_map[row["question_id"]].question.get("family", row.get("family", "?")))
        qtype = str(question_map[row["question_id"]].question.get("type", row.get("question_type", "?")))
        family_groups[family].append(row)
        type_groups[qtype].append(row)
        finish_counts[str(row.get("finish_reason") or "null")] += 1
        continuation_hist[int(row.get("continuation_count", 0))] += 1

    def summarize_bucket(bucket_rows: list[dict[str, Any]]) -> dict[str, Any]:
        parsed_rows = [row for row in bucket_rows if row.get("parsed_letter")]
        correct = sum(1 for row in parsed_rows if row.get("correct"))
        return {
            "total": len(bucket_rows),
            "parsed": len(parsed_rows),
            "correct": correct,
            "accuracy": _accuracy(correct, len(parsed_rows)),
        }

    by_family = {family: summarize_bucket(bucket_rows) for family, bucket_rows in sorted(family_groups.items())}
    by_type = {qtype: summarize_bucket(bucket_rows) for qtype, bucket_rows in sorted(type_groups.items())}

    rationale_by_family: dict[str, Any] = {}
    for family, bucket_rows in sorted(family_groups.items()):
        correct_bucket = [row for row in bucket_rows if row.get("correct")]
        wrong_bucket = [row for row in bucket_rows if row.get("parsed_letter") and not row.get("correct")]
        rationale_by_family[family] = {
            "correct_top_signals": _top_signal_summary(correct_bucket),
            "wrong_top_signals": _top_signal_summary(wrong_bucket),
            "correct_top_tokens": _top_tokens(correct_bucket),
            "wrong_top_tokens": _top_tokens(wrong_bucket),
        }

    total_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    usage_rows = 0
    for row in rows:
        usage = row.get("usage") or {}
        if usage:
            usage_rows += 1
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)

    return {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "model_name": run.manifest.get("model", {}).get("name"),
        "shared_questions": len(rows),
        "parsed": len(parsed),
        "unparsed": len(rows) - len(parsed),
        "correct": len(correct_rows),
        "accuracy": _accuracy(len(correct_rows), len(parsed)),
        "by_family": by_family,
        "by_type": by_type,
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "truncated_count": sum(1 for row in rows if row.get("truncated")),
        "continuation_histogram": {str(k): v for k, v in sorted(continuation_hist.items())},
        "usage_totals": {
            "rows_with_usage": usage_rows,
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
        "rationale_by_family": rationale_by_family,
    }


def _pairwise_summary(
    left: RunData,
    right: RunData,
    question_ids: list[str],
    question_map: dict[str, QuestionItem],
) -> dict[str, Any]:
    same_answer = 0
    different_answer = 0
    both_correct = 0
    only_left_correct = 0
    only_right_correct = 0
    neither_correct = 0
    correctness_flip_questions: list[dict[str, Any]] = []

    by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for qid in question_ids:
        lrow = left.results[qid]
        rrow = right.results[qid]
        family = str(question_map[qid].question.get("family", lrow.get("family", "?")))
        l_ans = lrow.get("parsed_letter")
        r_ans = rrow.get("parsed_letter")
        if l_ans == r_ans:
            same_answer += 1
            by_family[family]["same_answer"] += 1
        else:
            different_answer += 1
            by_family[family]["different_answer"] += 1

        l_correct = bool(lrow.get("correct"))
        r_correct = bool(rrow.get("correct"))
        if l_correct and r_correct:
            both_correct += 1
            by_family[family]["both_correct"] += 1
        elif l_correct and not r_correct:
            only_left_correct += 1
            by_family[family]["only_left_correct"] += 1
        elif r_correct and not l_correct:
            only_right_correct += 1
            by_family[family]["only_right_correct"] += 1
        else:
            neither_correct += 1
            by_family[family]["neither_correct"] += 1

        if l_correct != r_correct:
            correctness_flip_questions.append(
                {
                    "question_id": qid,
                    "family": family,
                    "type": str(question_map[qid].question.get("type", lrow.get("question_type", "?"))),
                    "ground_truth_letter": lrow.get("ground_truth_letter"),
                    "left_answer": l_ans,
                    "right_answer": r_ans,
                    "left_correct": l_correct,
                    "right_correct": r_correct,
                    "left_finish_reason": lrow.get("finish_reason"),
                    "right_finish_reason": rrow.get("finish_reason"),
                }
            )

    family_summary = {}
    for family, stats in sorted(by_family.items()):
        total = sum(
            stats.get(key, 0)
            for key in (
                "both_correct",
                "only_left_correct",
                "only_right_correct",
                "neither_correct",
            )
        )
        family_summary[family] = {
            **stats,
            "total": total,
            "correctness_flip_rate": _accuracy(
                stats.get("only_left_correct", 0) + stats.get("only_right_correct", 0),
                total,
            ),
        }

    return {
        "left": left.label,
        "right": right.label,
        "shared_questions": len(question_ids),
        "same_parsed_letter": same_answer,
        "different_parsed_letter": different_answer,
        "both_correct": both_correct,
        "only_left_correct": only_left_correct,
        "only_right_correct": only_right_correct,
        "neither_correct": neither_correct,
        "correctness_flip_rate": _accuracy(only_left_correct + only_right_correct, len(question_ids)),
        "family_breakdown": family_summary,
        "correctness_flip_questions": correctness_flip_questions,
    }


def _build_question_rows(
    runs: list[RunData],
    question_ids: list[str],
    question_map: dict[str, QuestionItem],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, qid in enumerate(question_ids, start=1):
        item = question_map[qid]
        entry = {
            "question_index": index,
            "question_id": qid,
            "family": item.question.get("family"),
            "question_type": item.question.get("type"),
            "ground_truth_letter": item.correct_letter,
            "prompt_text": item.prompt_text,
            "question_dir": str(item.question_dir),
            "runs": {},
        }
        for run in runs:
            row = run.results[qid]
            entry["runs"][run.label] = {
                "parsed_letter": row.get("parsed_letter"),
                "correct": bool(row.get("correct")),
                "finish_reason": row.get("finish_reason"),
                "truncated": bool(row.get("truncated")),
                "continuation_count": int(row.get("continuation_count", 0)),
                "usage": row.get("usage"),
                "history_entry_count": row.get("history_entry_count"),
                "history_context": row.get("history_context"),
                "chain_of_thought": row.get("chain_of_thought"),
                "model_response": row.get("model_response"),
            }
        output.append(entry)
    return output


def _svg_bar_chart(title: str, rows: list[tuple[str, float | None]], *, width: int = 900) -> str:
    usable_rows = [(label, value if value is not None else 0.0) for label, value in rows]
    height = 90 + 56 * len(usable_rows)
    bar_max = max((value for _, value in usable_rows), default=0.0)
    bar_max = max(bar_max, 0.01)
    svg_rows: list[str] = [
        f'<text x="20" y="28" font-size="20" font-weight="700" fill="#e5e7eb">{escape(title)}</text>'
    ]
    for idx, (label, value) in enumerate(usable_rows):
        y = 60 + idx * 56
        bar_width = int((width - 280) * (value / bar_max))
        svg_rows.append(
            f'<text x="20" y="{y + 18}" font-size="14" fill="#cbd5e1">{escape(label)}</text>'
        )
        svg_rows.append(
            f'<rect x="240" y="{y}" width="{width - 280}" height="24" rx="4" fill="#1f2937"></rect>'
        )
        svg_rows.append(
            f'<rect x="240" y="{y}" width="{bar_width}" height="24" rx="4" fill="#38bdf8"></rect>'
        )
        svg_rows.append(
            f'<text x="{width - 28}" y="{y + 18}" text-anchor="end" font-size="14" fill="#e5e7eb">{value * 100:.1f}%</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" class="chart">{"" .join(svg_rows)}</svg>'


def _render_signal_list(items: list[dict[str, Any]], *, share_key: str = "share") -> str:
    if not items:
        return "<li>none</li>"
    lines = []
    for item in items:
        share = item.get(share_key)
        if isinstance(share, (int, float)):
            suffix = f" ({share * 100:.1f}%)"
        else:
            suffix = ""
        label = item.get("label") or item.get("token") or item.get("signal")
        count = item.get("count")
        lines.append(f"<li>{escape(str(label))}: {count}{suffix}</li>")
    return "".join(lines)


def _render_html(
    *,
    summary: dict[str, Any],
    questions: list[dict[str, Any]],
    out_path: Path,
) -> None:
    run_summaries = summary["runs"]
    pairwise = summary["pairwise"]
    overall_rows = [(run["label"], run["accuracy"]) for run in run_summaries]
    family_names = sorted({family for run in run_summaries for family in run["by_family"]})

    family_charts = []
    for family in family_names:
        rows = []
        for run in run_summaries:
            rows.append((run["label"], run["by_family"].get(family, {}).get("accuracy")))
        family_charts.append(_svg_bar_chart(f"Accuracy by family: {family}", rows))

    run_cards = []
    for run in run_summaries:
        family_rows = []
        for family, metrics in run["by_family"].items():
            rationale = run["rationale_by_family"].get(family, {})
            family_rows.append(
                "<tr>"
                f"<td>{escape(family)}</td>"
                f"<td>{metrics['correct']}/{metrics['parsed']}</td>"
                f"<td>{_percent(metrics['accuracy'])}</td>"
                f"<td><ul>{_render_signal_list(rationale.get('correct_top_signals', []))}</ul></td>"
                f"<td><ul>{_render_signal_list(rationale.get('correct_top_tokens', []), share_key='token')}</ul></td>"
                "</tr>"
            )
        run_cards.append(
            "<section class='panel'>"
            f"<h2>{escape(run['label'])}</h2>"
            f"<p class='meta'>{escape(run['run_dir'])}</p>"
            "<table>"
            "<tr><th>Shared questions</th><th>Correct</th><th>Accuracy</th><th>Unparsed</th><th>Truncated</th><th>Total tokens</th><th>Reasoning tokens</th></tr>"
            "<tr>"
            f"<td>{run['shared_questions']}</td>"
            f"<td>{run['correct']}/{run['parsed']}</td>"
            f"<td>{_percent(run['accuracy'])}</td>"
            f"<td>{run['unparsed']}</td>"
            f"<td>{run['truncated_count']}</td>"
            f"<td>{run['usage_totals']['total_tokens']}</td>"
            f"<td>{run['usage_totals']['reasoning_tokens']}</td>"
            "</tr></table>"
            "<h3>By family</h3>"
            "<table><tr><th>Family</th><th>Score</th><th>Accuracy</th><th>Top correct-signal buckets</th><th>Top correct tokens</th></tr>"
            f"{''.join(family_rows)}"
            "</table>"
            "</section>"
        )

    pairwise_cards = []
    for cmp in pairwise:
        flip_rows = []
        for item in cmp["correctness_flip_questions"]:
            flip_rows.append(
                "<tr>"
                f"<td>{escape(item['question_id'])}</td>"
                f"<td>{escape(item['family'])}</td>"
                f"<td>{escape(item['ground_truth_letter'] or '')}</td>"
                f"<td>{escape(str(item['left_answer']))}</td>"
                f"<td>{'Y' if item['left_correct'] else 'N'}</td>"
                f"<td>{escape(str(item['right_answer']))}</td>"
                f"<td>{'Y' if item['right_correct'] else 'N'}</td>"
                "</tr>"
            )
        pairwise_cards.append(
            "<section class='panel'>"
            f"<h2>{escape(cmp['left'])} vs {escape(cmp['right'])}</h2>"
            "<table>"
            "<tr><th>Shared</th><th>Same answer</th><th>Different answer</th><th>Both correct</th><th>Only left correct</th><th>Only right correct</th><th>Neither correct</th><th>Flip rate</th></tr>"
            "<tr>"
            f"<td>{cmp['shared_questions']}</td>"
            f"<td>{cmp['same_parsed_letter']}</td>"
            f"<td>{cmp['different_parsed_letter']}</td>"
            f"<td>{cmp['both_correct']}</td>"
            f"<td>{cmp['only_left_correct']}</td>"
            f"<td>{cmp['only_right_correct']}</td>"
            f"<td>{cmp['neither_correct']}</td>"
            f"<td>{_percent(cmp['correctness_flip_rate'])}</td>"
            "</tr></table>"
            "<details><summary>Correctness flips</summary>"
            "<table><tr><th>Question</th><th>Family</th><th>GT</th><th>Left ans</th><th>Left correct</th><th>Right ans</th><th>Right correct</th></tr>"
            f"{''.join(flip_rows) if flip_rows else '<tr><td colspan=7>none</td></tr>'}"
            "</table></details>"
            "</section>"
        )

    question_sections = []
    for question in questions:
        run_blocks = []
        for run_label, row in question["runs"].items():
            usage = row.get("usage") or {}
            usage_line = (
                f"tokens={usage.get('total_tokens', 'n/a')}, "
                f"completion={usage.get('completion_tokens', 'n/a')}, "
                f"reasoning={(usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 'n/a')}"
            )
            history_count = row.get("history_entry_count")
            history_note = f"history_entries={history_count}" if history_count is not None else "history_entries=n/a"
            reasoning = _clean_reasoning(row.get("chain_of_thought") or row.get("model_response"))
            history_context = row.get("history_context") or ""
            run_blocks.append(
                "<div class='run-block'>"
                f"<h4>{escape(run_label)}</h4>"
                f"<p><strong>Answer:</strong> {escape(str(row.get('parsed_letter')))} | "
                f"<strong>Correct:</strong> {'Y' if row.get('correct') else 'N'} | "
                f"<strong>Finish:</strong> {escape(str(row.get('finish_reason')))} | "
                f"<strong>Truncated:</strong> {row.get('truncated')} | "
                f"<strong>Continuations:</strong> {row.get('continuation_count')} | "
                f"<strong>{escape(history_note)}</strong></p>"
                f"<p class='meta'>{escape(usage_line)}</p>"
                "<details><summary>Reasoning</summary>"
                f"<pre>{escape(reasoning)}</pre></details>"
                "<details><summary>Raw response</summary>"
                f"<pre>{escape(str(row.get('model_response') or ''))}</pre></details>"
                "<details><summary>History context</summary>"
                f"<pre>{escape(history_context)}</pre></details>"
                "</div>"
            )
        question_sections.append(
            "<section class='panel question'>"
            f"<h2>Q{question['question_index']}: {escape(question['question_id'])}</h2>"
            f"<p><strong>Family:</strong> {escape(str(question['family']))} | "
            f"<strong>Type:</strong> {escape(str(question['question_type']))} | "
            f"<strong>GT:</strong> {escape(question['ground_truth_letter'])}</p>"
            f"<p class='meta'>{escape(question['question_dir'])}</p>"
            "<details open><summary>Prompt</summary>"
            f"<pre>{escape(question['prompt_text'])}</pre></details>"
            f"{''.join(run_blocks)}"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ArchitectureIQ LLM Run Comparison</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #0b1220;
      --border: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #22c55e;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; }}
    .meta {{ color: var(--muted); word-break: break-all; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 0;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: var(--panel-2); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      max-height: 400px;
      overflow: auto;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
    }}
    .chart {{
      width: 100%;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 16px;
    }}
    .run-block {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin-top: 12px;
      background: var(--panel-2);
    }}
    ul {{ margin: 0; padding-left: 18px; }}
    .question {{ scroll-margin-top: 16px; }}
  </style>
</head>
<body>
<main>
  <section class="panel">
    <h1>ArchitectureIQ LLM run comparison</h1>
    <p>Shared question set: {summary['shared_question_count']} questions. Extra questions excluded from intersection: {escape(json.dumps(summary['nonshared_questions'], ensure_ascii=False))}</p>
    <p>This report compares independent runs against sequential-feedback runs, where each prompt includes revealed answers from earlier questions in the same run.</p>
  </section>

  <section class="panel">
    <h2>Overall Accuracy</h2>
    {_svg_bar_chart("Shared-set accuracy", overall_rows)}
  </section>

  <section class="panel">
    <h2>By Family</h2>
    {''.join(family_charts)}
  </section>

  <div class="grid">
    {''.join(run_cards)}
  </div>

  <div class="grid">
    {''.join(pairwise_cards)}
  </div>

  <section class="panel">
    <h2>Question-level records</h2>
    <p>Each question includes prompt text, all compared answers, and the full stored reasoning/raw response fields from the run artifacts.</p>
  </section>
  {''.join(question_sections)}
</main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", help="ArchitectureIQ data root")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named run directory to compare",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for summary.json, questions.json, and index.html")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids, question_map = _load_questions(data_root)

    runs: list[RunData] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec: {spec!r}; expected LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        runs.append(_load_run(label, Path(raw_path).expanduser().resolve()))

    shared_ids = set(ordered_ids)
    run_specific_only: dict[str, list[str]] = {}
    for run in runs:
        run_ids = set(run.results)
        shared_ids &= run_ids
        run_specific_only[run.label] = sorted(run_ids - set(ordered_ids))
    shared_question_ids = [qid for qid in ordered_ids if qid in shared_ids]

    nonshared_questions: dict[str, list[str]] = {}
    for run in runs:
        run_ids = set(run.results)
        missing = [qid for qid in ordered_ids if qid not in run_ids]
        extras = sorted(run_ids - set(ordered_ids))
        if missing or extras:
            nonshared_questions[run.label] = missing + extras

    run_summaries = [_run_summary(run, shared_question_ids, question_map) for run in runs]
    pairwise = []
    for idx in range(len(runs)):
        for jdx in range(idx + 1, len(runs)):
            pairwise.append(_pairwise_summary(runs[idx], runs[jdx], shared_question_ids, question_map))

    questions = _build_question_rows(runs, shared_question_ids, question_map)

    summary = {
        "data_root": str(data_root),
        "shared_question_count": len(shared_question_ids),
        "ordered_question_count_in_data_root": len(ordered_ids),
        "shared_question_ids": shared_question_ids,
        "nonshared_questions": nonshared_questions,
        "runs": run_summaries,
        "pairwise": pairwise,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "questions.json").write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")
    _render_html(summary=summary, questions=questions, out_path=out_dir / "index.html")


if __name__ == "__main__":
    main()
