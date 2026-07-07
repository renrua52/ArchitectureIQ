from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROFILES_DIR = ROOT / "profiles"
PROMPTS_DIR = ROOT / "prompts" / "templates"


def dataset_dir(family: str, dataset_id: str) -> Path:
    return DATA_DIR / "datasets" / family / dataset_id


def candidate_set_dir(dataset_path: Path, set_name: str) -> Path:
    return dataset_path / "candidates" / set_name


def candidate_in_set_dir(set_path: Path, candidate_id: str) -> Path:
    return set_path / candidate_id


def candidate_dir(family: str, dataset_id: str, budget: int, candidate_id: str) -> Path:
    """Legacy layout: candidates/budget_{n}/{candidate_id}."""
    return (
        dataset_dir(family, dataset_id)
        / "candidates"
        / f"budget_{budget}"
        / candidate_id
    )


def question_dir(question_id: str) -> Path:
    """Legacy flat layout: data/questions/{question_id}."""
    return DATA_DIR / "questions" / question_id
