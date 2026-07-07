"""Read-only loaders for ArchitectureIQ output artifacts (no main-package imports)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CANDIDATE_FILES = (
    "summary.json",
    "candidate_spec.json",
    "loss.py",
    "model.py",
    "optimizer.py",
    "train.py",
)

DATASET_FILES = (
    "dataset_spec.json",
    "synthesize.py",
)


@dataclass(frozen=True)
class QuestionBundle:
    question_root: Path
    data_root: Path
    question: dict[str, Any]
    prompt_text: str
    dataset_dir: Path
    choices: list[dict[str, Any]]


def resolve_data_root(question_root: Path, data_root: Path | str | None = None) -> Path:
    question_root = question_root.resolve()
    if data_root is not None:
        return Path(data_root).resolve()
    current = question_root
    while True:
        if (current / "datasets").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    cwd_data = Path.cwd() / "data"
    if (cwd_data / "datasets").is_dir():
        return cwd_data
    raise FileNotFoundError(
        "Could not infer data root. Pass --data-root pointing at the directory "
        "that contains datasets/ and questions/."
    )


def load_question_bundle(question_path: Path | str, data_root: Path | str | None = None) -> QuestionBundle:
    question_root = Path(question_path).resolve()
    if question_root.is_file():
        question_root = question_root.parent
    question_file = question_root / "question.json"
    if not question_file.is_file():
        raise FileNotFoundError(f"Missing question.json in {question_root}")

    root = resolve_data_root(question_root, data_root)
    question = json.loads(question_file.read_text())
    prompt_path = question_root / question.get("prompt", {}).get("rendered_path", "prompt.txt")
    prompt_text = prompt_path.read_text() if prompt_path.is_file() else ""

    family = question["family"]
    dataset_id = question["dataset_id"]
    dataset_dir = root / "datasets" / family / dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    choices: list[dict[str, Any]] = []
    for choice in question["choices"]:
        candidate_dir = root / choice["candidate_path"]
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")
        choices.append({**choice, "candidate_dir": candidate_dir})

    return QuestionBundle(
        question_root=question_root,
        data_root=root,
        question=question,
        prompt_text=prompt_text,
        dataset_dir=dataset_dir,
        choices=choices,
    )


def read_text_file(path: Path) -> str:
    if not path.is_file():
        return f"(missing: {path.name})"
    return path.read_text()


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"error": f"missing {path.name}"}
    return json.loads(path.read_text())


def load_dataset_tensors(dataset_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.load(dataset_dir / "train.pt", weights_only=True)
    test = torch.load(dataset_dir / "test.pt", weights_only=True)
    return train["x"], train["y"], test["x"], test["y"]


def candidate_file_paths(candidate_dir: Path, *, include_summary: bool = True) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "candidate_spec.json": candidate_dir / "candidate_spec.json",
        "loss.py": candidate_dir / "loss.py",
        "model.py": candidate_dir / "model.py",
        "optimizer.py": candidate_dir / "optimizer.py",
        "train.py": candidate_dir / "train.py",
    }
    results_dir = candidate_dir / "results"
    summary = results_dir / "summary.json"
    if include_summary and summary.is_file():
        paths["summary.json"] = summary
        curves = results_dir / "curves.npz"
        if curves.is_file():
            paths["curves.npz"] = curves
    return paths


def dataset_file_paths(dataset_dir: Path) -> dict[str, Path]:
    return {name: dataset_dir / name for name in DATASET_FILES}


def _iter_question_json_paths(root: Path) -> list[Path]:
    """Collect question.json paths under legacy and dataset-scoped layouts."""
    found: list[Path] = []
    legacy = root / "questions"
    if legacy.is_dir():
        for path in legacy.iterdir():
            qfile = path / "question.json"
            if path.is_dir() and qfile.is_file():
                found.append(qfile)

    datasets_dir = root / "datasets"
    if datasets_dir.is_dir():
        for qfile in datasets_dir.rglob("questions/*/*/question.json"):
            found.append(qfile)

    return found


def list_question_dirs(data_root: Path | str) -> list[Path]:
    """Return sorted question directories under a data root."""
    root = Path(data_root).resolve()
    return sorted({qfile.parent.resolve() for qfile in _iter_question_json_paths(root)})


def question_label(question_dir: Path) -> str:
    question_file = question_dir / "question.json"
    if not question_file.is_file():
        return question_dir.name
    question = json.loads(question_file.read_text())
    qid = question.get("question_id", question_dir.name)
    qtype = question.get("type", "?")
    dataset_id = question.get("dataset_id", "?")
    budget = question.get("budget", {})
    if isinstance(budget, dict):
        samples = budget.get("total_samples_seen", "?")
    else:
        samples = budget
    return f"{qid} · {qtype} · {dataset_id} · n={samples}"


def format_metrics(summary: dict[str, Any]) -> str:
    metric = summary.get("selection_metric", "test_mse")
    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    mean = summary.get(mean_key)
    std = summary.get(std_key)
    if mean is None:
        return "Metrics unavailable"
    if std is None:
        return f"{metric}: {mean:.6f}"
    return f"{metric}: {mean:.6f} ± {std:.6f}"
