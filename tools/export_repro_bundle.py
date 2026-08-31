#!/usr/bin/env python3
"""Export the downloadable "reproduce the ground truth" bundle for one question.

Mirror of the browser-side builder in `frontend/quiz/src/bundle.ts`: same layout,
same gating, same static `reproduce.py` / `README.md` (read from
`frontend/quiz/repro/`, so there is one copy of each). `tests/test_repro_bundle.py`
cross-checks the two implementations.

Reads the BakeFile only — like `tools/question_inspector/`, this deliberately does
not import `architecture_iq`, so it works against any deployed bake.

    python tools/export_repro_bundle.py --list
    python tools/export_repro_bundle.py --question q_502033 --out /tmp/q_502033 --answered
    python tools/export_repro_bundle.py --question q_502033 --out /tmp/bundle.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAKE = REPO_ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"
REPRO_DIR = REPO_ROOT / "frontend" / "quiz" / "repro"
BUNDLE_VERSION = 1


def _render_file(value: Any) -> str:
    """Bake `files` values are either raw source text or already-parsed JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2) + "\n"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _recorded_threads(reveal_files: dict[str, Any]) -> int | float | None:
    for letter in reveal_files:
        summary = _as_dict(_as_dict(reveal_files[letter]).get("summary.json"))
        threads = _number_or_none(_as_dict(summary.get("environment")).get("torch_threads_per_seed"))
        if threads is not None:
            return threads
    return None


def build_bundle_entries(question: dict[str, Any], answered: bool) -> dict[str, str]:
    """Return {path_inside_zip: text}, matching buildBundleEntries() in bundle.ts."""
    root = question["id"]
    detail = question["detail"]
    evaluation = _as_dict(question.get("evaluation"))
    dataset_files = _as_dict(detail["dataset"].get("files"))
    dataset_spec = _as_dict(dataset_files.get("dataset_spec.json"))
    significance = _as_dict(dataset_spec.get("significance"))
    metric = (
        detail["dataset"].get("selectionMetric")
        or evaluation.get("selection_metric")
        or question.get("metric")
        or "test_mse"
    )
    reveal_files = _as_dict(_as_dict(question.get("reveal")).get("files"))

    meta: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "question_id": question["id"],
        "family": question["family"],
        "dataset_id": question["datasetId"],
        "type": question["type"],
        "profile": question.get("profile"),
        "selection_metric": metric,
        "n_seeds": _number_or_none(evaluation.get("n_seeds")) or 10,
        "base_seed": _number_or_none(evaluation.get("base_seed")) or 0,
        "device": evaluation.get("device") or "cpu",
        "fail_threshold": _number_or_none(significance.get("fail_threshold")),
        "varying_axes": question.get("varyingAxes") or [],
        "invariant_axes": question.get("invariantAxes") or [],
        "answered": answered,
        "choices": [],
    }
    for choice in detail["choices"]:
        spec = _as_dict(_as_dict(choice.get("files")).get("candidate_spec.json"))
        budget = _as_dict(spec.get("budget"))
        meta["choices"].append(
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidateId"],
                "training_steps": _number_or_none(budget.get("training_steps")),
                "batch_size": _number_or_none(budget.get("batch_size")),
                "total_samples_seen": _number_or_none(budget.get("total_samples_seen")),
            }
        )

    threads = _recorded_threads(reveal_files)
    if threads is not None:
        meta["torch_threads_per_seed"] = threads
    if answered:
        reveal = _as_dict(question.get("reveal"))
        meta["correct_letter"] = reveal.get("correctLetter")
        meta["ranked"] = [entry["letter"] for entry in reveal.get("ranked") or []]

    entries: dict[str, str] = {
        f"{root}/README.md": (REPRO_DIR / "README.md").read_text(encoding="utf-8"),
        f"{root}/reproduce.py": (REPRO_DIR / "reproduce.py").read_text(encoding="utf-8"),
        f"{root}/question.json": json.dumps(meta, indent=2) + "\n",
        f"{root}/prompt.txt": detail.get("prompt") or "",
    }

    for name in sorted(dataset_files):
        entries[f"{root}/dataset/{name}"] = _render_file(dataset_files[name])

    for choice in detail["choices"]:
        letter = choice["letter"]
        files = _as_dict(choice.get("files"))
        for name in sorted(files):
            entries[f"{root}/choices/{letter}/{name}"] = _render_file(files[name])
        if not answered:
            continue
        reference = _as_dict(reveal_files.get(letter))
        for name in sorted(reference):
            entries[f"{root}/choices/{letter}/reference/{name}"] = _render_file(reference[name])

    return entries


def write_bundle(entries: dict[str, str], out: Path) -> Path:
    """Write to a .zip when `out` ends in .zip, otherwise to a directory tree."""
    if out.suffix == ".zip":
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for path in entries:
                archive.writestr(path, entries[path])
        return out
    for path, content in entries.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return out


def load_question(bake_path: Path, question_id: str) -> dict[str, Any]:
    with bake_path.open(encoding="utf-8") as handle:
        bake = json.load(handle)
    by_id = bake.get("byId") or {}
    if question_id not in by_id:
        raise SystemExit(
            f"{question_id!r} is not in {bake_path}. Available: {', '.join(sorted(by_id))}"
        )
    return by_id[question_id]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bake", type=Path, default=DEFAULT_BAKE, help=f"BakeFile (default {DEFAULT_BAKE})")
    parser.add_argument("--question", help="question id, e.g. q_502033")
    parser.add_argument("--out", type=Path, help="output directory, or a path ending in .zip")
    parser.add_argument(
        "--answered",
        action="store_true",
        help="include the reference results and the answer key (what the site gives after answering)",
    )
    parser.add_argument("--list", action="store_true", help="list question ids in the bake and exit")
    args = parser.parse_args(argv)

    if not args.bake.exists():
        raise SystemExit(f"BakeFile not found: {args.bake}")
    if args.list:
        with args.bake.open(encoding="utf-8") as handle:
            bake = json.load(handle)
        for summary in bake.get("questions") or []:
            print(f"{summary['id']}  {summary.get('family', '?')}  {summary.get('type', '?')}")
        return 0
    if not args.question or not args.out:
        parser.error("--question and --out are required (or use --list)")

    question = load_question(args.bake, args.question)
    entries = build_bundle_entries(question, bool(args.answered))
    target = write_bundle(entries, args.out)
    gating = "with reference results" if args.answered else "code only (pre-answer)"
    print(f"wrote {len(entries)} files to {target}  [{gating}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
