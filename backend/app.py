"""FastAPI service for the ArchitectureIQ React frontend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR_DIR = ROOT / "tools" / "question_inspector"
if str(INSPECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(INSPECTOR_DIR))

from artifact_loader import (  # noqa: E402
    QuestionBundle,
    candidate_file_paths,
    format_metrics,
    list_question_dirs,
    load_question_bundle,
    read_json_file,
    read_text_file,
)

DEFAULT_DATA_ROOT = ROOT / "data"

app = FastAPI(title="ArchitectureIQ API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _data_root(data_root: str | None) -> Path:
    return Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT


def _question_dirs(data_root: str | None = None) -> list[Path]:
    root = _data_root(data_root)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Data root not found: {root}")
    return list_question_dirs(root)


def _bundle(question_id: str, data_root: str | None = None) -> QuestionBundle:
    for question_dir in _question_dirs(data_root):
        qfile = question_dir / "question.json"
        question = read_json_file(qfile)
        if question.get("question_id") == question_id:
            return load_question_bundle(question_dir, _data_root(data_root))
    raise HTTPException(status_code=404, detail=f"Question not found: {question_id}")


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_fmt_value(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "-"
    return str(value)


def _flatten_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "training steps": spec.get("budget", {}).get("training_steps"),
        "batch size": spec.get("budget", {}).get("batch_size"),
        "total samples seen": spec.get("budget", {}).get("total_samples_seen"),
        "model type": spec.get("model", {}).get("type"),
        "layers": spec.get("model", {}).get("depth"),
        "width": spec.get("model", {}).get("width"),
        "residual": spec.get("model", {}).get("residual"),
        "layer norm": spec.get("model", {}).get("layer_norm"),
        "activations": spec.get("model", {}).get("activations"),
        "optimizer": spec.get("optimizer", {}).get("type"),
        "learning rate": spec.get("optimizer", {}).get("lr"),
        "weight decay": spec.get("optimizer", {}).get("weight_decay"),
        "betas": spec.get("optimizer", {}).get("betas"),
        "loss": spec.get("loss", {}).get("loss_id"),
        "lambda": spec.get("loss", {}).get("lambda"),
    }


def _candidate_specs(bundle: QuestionBundle) -> dict[str, dict[str, Any]]:
    return {
        choice["letter"]: read_json_file(choice["candidate_dir"] / "candidate_spec.json")
        for choice in bundle.choices
    }


def _shared_and_variant_specs(
    bundle: QuestionBundle,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    specs = _candidate_specs(bundle)
    flattened = {letter: _flatten_spec(spec) for letter, spec in specs.items()}
    letters = list(flattened.keys())
    shared: list[dict[str, str]] = []
    variant: dict[str, list[dict[str, str]]] = {letter: [] for letter in letters}

    if not letters:
        return shared, variant

    for key in flattened[letters[0]]:
        values = [flattened[letter].get(key) for letter in letters]
        if all(value == values[0] for value in values):
            if values[0] is not None:
                shared.append({"label": key, "value": _fmt_value(values[0])})
        else:
            for letter, value in zip(letters, values, strict=True):
                if value is not None:
                    variant[letter].append({"label": key, "value": _fmt_value(value)})

    return shared, variant


def _question_title(bundle: QuestionBundle) -> str:
    q = bundle.question
    metric = q.get("significance", {}).get("metric") or q.get("evaluation", {}).get("selection_metric")
    metric_label = str(metric or "test metric").replace("_", " ")
    return f"Which training setup will reach the lowest {metric_label}?"


def _safe_points(tensor: torch.Tensor, max_points: int = 180) -> list[float]:
    values = tensor.squeeze().detach().cpu().numpy().astype(float).tolist()
    if not isinstance(values, list):
        return [float(values)]
    if len(values) <= max_points:
        return values
    step = max(1, len(values) // max_points)
    return values[::step][:max_points]


def _dataset_points(bundle: QuestionBundle) -> dict[str, list[dict[str, float]]]:
    train = torch.load(bundle.dataset_dir / "train.pt", weights_only=True)
    test = torch.load(bundle.dataset_dir / "test.pt", weights_only=True)
    train_x = _safe_points(train["x"])
    train_y = _safe_points(train["y"])
    test_x = _safe_points(test["x"])
    test_y = _safe_points(test["y"])
    return {
        "train": [{"x": x, "y": y} for x, y in zip(train_x, train_y, strict=False)],
        "test": [{"x": x, "y": y} for x, y in zip(test_x, test_y, strict=False)],
    }


def _summary_row(choice: dict[str, Any]) -> dict[str, Any]:
    summary_path = choice["candidate_dir"] / "results" / "summary.json"
    summary = read_json_file(summary_path) if summary_path.is_file() else {}
    metric = summary.get("selection_metric", "test_mse")
    return {
        "letter": choice["letter"],
        "candidateId": choice["candidate_id"],
        "metric": metric,
        "mean": summary.get(f"mean_{metric}"),
        "std": summary.get(f"std_{metric}"),
        "label": format_metrics(summary) if summary and "error" not in summary else "Metrics unavailable",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/questions")
def list_questions(data_root: str | None = Query(default=None)) -> dict[str, Any]:
    questions = []
    for index, question_dir in enumerate(_question_dirs(data_root)):
        question = read_json_file(question_dir / "question.json")
        budget = question.get("budget", {})
        questions.append(
            {
                "id": question.get("question_id", question_dir.name),
                "index": index,
                "type": question.get("type"),
                "datasetId": question.get("dataset_id"),
                "budget": budget.get("total_samples_seen") if isinstance(budget, dict) else budget,
                "choices": question.get("num_choices", len(question.get("choices", []))),
            }
        )
    return {"questions": questions}


@app.get("/api/questions/{question_id}")
def get_question(question_id: str, data_root: str | None = Query(default=None)) -> dict[str, Any]:
    bundle = _bundle(question_id, data_root)
    q = bundle.question
    shared, variant = _shared_and_variant_specs(bundle)
    return {
        "id": q["question_id"],
        "title": _question_title(bundle),
        "family": q.get("family"),
        "datasetId": q.get("dataset_id"),
        "type": q.get("type"),
        "profile": q.get("profile"),
        "budget": q.get("budget"),
        "metric": q.get("significance", {}).get("metric")
        or q.get("evaluation", {}).get("selection_metric"),
        "evaluation": q.get("evaluation", {}),
        "invariantAxes": q.get("invariant_axes", []),
        "varyingAxes": q.get("varying_axes", []),
        "prompt": bundle.prompt_text,
        "dataset": _dataset_points(bundle),
        "shared": shared,
        "choices": [
            {
                "letter": choice["letter"],
                "candidateId": choice["candidate_id"],
                "variant": variant.get(choice["letter"], []),
            }
            for choice in bundle.choices
        ],
    }


@app.post("/api/questions/{question_id}/answer")
def answer_question(
    question_id: str,
    payload: dict[str, Any],
    data_root: str | None = Query(default=None),
) -> dict[str, Any]:
    bundle = _bundle(question_id, data_root)
    letter = str(payload.get("letter", "")).upper()
    valid = {choice["letter"] for choice in bundle.choices}
    if letter not in valid:
        raise HTTPException(status_code=400, detail="Invalid answer letter")

    ranked = [_summary_row(choice) for choice in bundle.choices]
    ranked.sort(key=lambda row: (row["mean"] is None, row["mean"] or float("inf")))
    return {
        "picked": letter,
        "correctLetter": bundle.question["correct_letter"],
        "correct": letter == bundle.question["correct_letter"],
        "ranked": ranked,
    }


@app.get("/api/questions/{question_id}/candidates/{letter}/files")
def candidate_files(
    question_id: str,
    letter: str,
    reveal: bool = Query(default=False),
    data_root: str | None = Query(default=None),
) -> dict[str, Any]:
    bundle = _bundle(question_id, data_root)
    selected = next((choice for choice in bundle.choices if choice["letter"] == letter.upper()), None)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Choice not found: {letter}")
    paths = candidate_file_paths(selected["candidate_dir"], include_summary=reveal)
    files = []
    for name, path in paths.items():
        files.append(
            {
                "name": name,
                "content": read_json_file(path) if name.endswith(".json") else read_text_file(path),
                "kind": "json" if name.endswith(".json") else "text",
            }
        )
    return {"files": files}
