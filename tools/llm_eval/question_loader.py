"""Load question artifacts (no main-package imports)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_MANIFEST = "run.json"


@dataclass(frozen=True)
class QuestionItem:
    question_dir: Path
    question_id: str
    question: dict[str, Any]
    prompt_text: str
    prompt_hash: str
    valid_letters: frozenset[str]

    @property
    def correct_letter(self) -> str:
        return str(self.question["correct_letter"]).upper()


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def load_question_item(question_dir: Path) -> QuestionItem:
    question_dir = question_dir.resolve()
    question_file = question_dir / "question.json"
    if not question_file.is_file():
        raise FileNotFoundError(f"Missing question.json in {question_dir}")

    question = json.loads(question_file.read_text(encoding="utf-8"))
    prompt_rel = question.get("prompt", {}).get("rendered_path", "prompt.txt")
    prompt_path = question_dir / prompt_rel
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prompt file {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    letters = frozenset(str(c["letter"]).upper() for c in question["choices"])
    qid = question.get("question_id", question_dir.name)
    return QuestionItem(
        question_dir=question_dir,
        question_id=qid,
        question=question,
        prompt_text=prompt_text,
        prompt_hash=prompt_hash(prompt_text),
        valid_letters=letters,
    )


def _collect_question_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []

    if (root / "question.json").is_file():
        return [root]

    if (root / RUN_MANIFEST).is_file():
        return sorted(
            path.resolve()
            for path in root.iterdir()
            if path.is_dir() and (path / "question.json").is_file()
        )

    if root.name == "questions" and (root.parent / "dataset_spec.json").is_file():
        runs: list[Path] = []
        for run_dir in sorted(root.iterdir()):
            if run_dir.is_dir() and (run_dir / RUN_MANIFEST).is_file():
                runs.extend(_collect_question_dirs(run_dir))
        return runs

    if (root / "datasets").is_dir():
        found: list[Path] = []
        legacy = root / "questions"
        if legacy.is_dir():
            found.extend(_collect_question_dirs(legacy))
        for qfile in (root / "datasets").rglob("questions/*/*/question.json"):
            found.append(qfile.parent.resolve())
        return sorted(set(found))

    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and (path / "question.json").is_file()
    )


def list_questions(questions_root: Path) -> list[QuestionItem]:
    dirs = _collect_question_dirs(questions_root)
    return [load_question_item(path) for path in dirs]
