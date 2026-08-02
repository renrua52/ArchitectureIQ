#!/usr/bin/env python3
"""Apply the small collection/provenance delta to the imported React quiz.

The upstream branch is intentionally kept recognizable.  This idempotent
adapter only changes ordered Next behavior and exposes collection provenance;
it is separate from the Streamlit Inspector frontend.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        if new in text:
            return
        raise RuntimeError(f"Expected source fragment not found: {path}")
    if count != 1:
        raise RuntimeError(f"Expected one source fragment, found {count}: {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_types() -> None:
    path = ROOT / "frontend" / "quiz" / "src" / "types.ts"
    replace_once(
        path,
        "  choices?: number;\n};",
        "  choices?: number;\n  profile?: string;\n  profileHash?: string;\n  track?: string;\n  order?: number;\n};",
    )
    replace_once(
        path,
        "  profile?: string;\n  budget:",
        "  profile?: string;\n  profileHash?: string;\n  track?: string;\n  sourceRun?: string | null;\n  provenance?: Record<string, unknown>;\n  budget:",
    )
    replace_once(
        path,
        "export type BakeFile = {\n  schema_version: number;\n  questions: QuestionSummary[];\n  byId: Record<string, BakedQuestion>;\n};",
        "export type BakeFile = {\n  schema_version: number;\n  ordered?: boolean;\n  collection?: Record<string, unknown> | null;\n  questions: QuestionSummary[];\n  byId: Record<string, BakedQuestion>;\n};",
    )


def patch_main() -> None:
    path = ROOT / "frontend" / "quiz" / "src" / "main.tsx"
    replace_once(
        path,
        "  function nextQuestion() {\n    if (!summaries.length) {\n      return;\n    }\n    leaveAndSwitch((index + 1) % summaries.length);\n  }",
        "  function nextQuestion() {\n    if (!summaries.length || (bake?.ordered && index >= summaries.length - 1)) {\n      return;\n    }\n    const next = bake?.ordered ? index + 1 : (index + 1) % summaries.length;\n    leaveAndSwitch(next);\n  }",
    )
    replace_once(
        path,
        "          <button type=\"button\" onClick={nextQuestion}>\n            Next\n          </button>",
        "          <button\n            type=\"button\"\n            onClick={nextQuestion}\n            disabled={Boolean(bake.ordered && index >= summaries.length - 1)}\n          >\n            {bake.ordered && index >= summaries.length - 1 ? \"End\" : \"Next\"}\n          </button>",
    )
    replace_once(
        path,
        "        <span className=\"tag\">{humanType(question.type)}</span>\n        <span className=\"dot\">·</span>\n        <span>{question.detail.choices.length} choices</span>",
        "        <span className=\"tag\">{humanType(question.type)}</span>\n        <span className=\"dot\">·</span>\n        <span className=\"tag\">{question.track ?? \"default\"}</span>\n        <span className=\"dot\">·</span>\n        <span>{question.detail.choices.length} choices</span>",
    )
    replace_once(
        path,
        "      <section className=\"stage-screen\" key={`${question.id}-${stage}`}>",
        "      <section className=\"stage-screen\" key={`${question.id}-${stage}`}>\n        <div className=\"provenance\" aria-label=\"Question provenance\">\n          <span>Track: {question.track ?? \"default\"}</span>\n          <span>Profile: {question.profile ?? \"legacy/unknown\"}</span>\n          <span>Hash: {question.profileHash ?? \"legacy/unknown\"}</span>\n        </div>",
    )
    replace_once(
        path,
        "              <span>\n                {humanFamily(item.family)} · {humanMetricByFamily(item.family)} ·{\" \"}\n                {humanType(item.type)} · {item.choices ?? \"?\"} choices\n              </span>",
        "              <span>\n                {humanFamily(item.family)} · {humanMetricByFamily(item.family)} ·{\" \"}\n                {humanType(item.type)} · {item.track ?? \"default\"} · {item.choices ?? \"?\"} choices\n              </span>",
    )
    replace_once(
        path,
        "        <button type=\"button\" className=\"cta\" onClick={onNext}>\n          Next question →\n        </button>",
        "        <button type=\"button\" className=\"cta\" onClick={onNext}>\n          Next question →\n        </button>",
    )


def main() -> int:
    patch_types()
    patch_main()
    print("patched collection-aware quiz frontend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
