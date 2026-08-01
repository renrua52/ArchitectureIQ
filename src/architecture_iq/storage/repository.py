"""Repository API for the columnar backend storage (``backend/data/``).

All paths are derived from IDs; consumers never store or construct paths
inside JSON. Writing is done by the generator suite; reading is shared
with the evaluation suite.
"""
from __future__ import annotations

from pathlib import Path

from architecture_iq.storage.schema import (
    CANDIDATES_DIR,
    CURVES_NPZ,
    PROBLEMS_DIR,
    PROBLEM_README_MD,
    PROBLEM_SPEC_JSON,
    RESULTS_DIR,
    SUMMARY_JSON,
    TRAINERS_DIR,
    TRAINER_SPEC_JSON,
    TRAINER_TRAIN_PY,
)
from architecture_iq.util import read_json, write_json

BACKEND_ROOT = Path(__file__).resolve().parents[3] / "backend"
DATA_ROOT = BACKEND_ROOT / "data"


def data_root() -> Path:
    return DATA_ROOT


# --- problems -------------------------------------------------------------

def problem_dir(problem_id: str) -> Path:
    return data_root() / PROBLEMS_DIR / problem_id


def problem_spec_path(problem_id: str) -> Path:
    return problem_dir(problem_id) / PROBLEM_SPEC_JSON


def problem_readme_path(problem_id: str) -> Path:
    return problem_dir(problem_id) / PROBLEM_README_MD


def list_problems() -> list[str]:
    root = data_root() / PROBLEMS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / PROBLEM_SPEC_JSON).is_file())


def read_problem_spec(problem_id: str) -> dict:
    return read_json(problem_spec_path(problem_id))


def write_problem_spec(problem_id: str, spec: dict) -> None:
    write_json(problem_spec_path(problem_id), spec)


# --- trainers -------------------------------------------------------------

def trainer_dir(trainer_id: str) -> Path:
    return data_root() / TRAINERS_DIR / trainer_id


def trainer_spec_path(trainer_id: str) -> Path:
    return trainer_dir(trainer_id) / TRAINER_SPEC_JSON


def trainer_train_path(trainer_id: str) -> Path:
    return trainer_dir(trainer_id) / TRAINER_TRAIN_PY


def list_trainers() -> list[str]:
    root = data_root() / TRAINERS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / TRAINER_SPEC_JSON).is_file())


def write_trainer(trainer_id: str, spec: dict, train_py: str) -> None:
    out_dir = trainer_dir(trainer_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / TRAINER_SPEC_JSON, spec)
    (out_dir / TRAINER_TRAIN_PY).write_text(train_py, encoding="utf-8")


def read_trainer_spec(trainer_id: str) -> dict:
    return read_json(trainer_spec_path(trainer_id))


# --- candidates -----------------------------------------------------------

def candidate_config_path(problem_id: str, candidate_id: str) -> Path:
    return data_root() / CANDIDATES_DIR / problem_id / f"{candidate_id}.json"


def list_candidate_ids(problem_id: str) -> list[str]:
    root = data_root() / CANDIDATES_DIR / problem_id
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def read_candidate_config(problem_id: str, candidate_id: str) -> dict:
    return read_json(candidate_config_path(problem_id, candidate_id))


def write_candidate_config(problem_id: str, config: dict) -> None:
    path = candidate_config_path(problem_id, config["candidate_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, config)


# --- results --------------------------------------------------------------

def results_dir(problem_id: str, candidate_id: str) -> Path:
    return data_root() / RESULTS_DIR / problem_id / candidate_id


def summary_path(problem_id: str, candidate_id: str) -> Path:
    return results_dir(problem_id, candidate_id) / SUMMARY_JSON


def curves_path(problem_id: str, candidate_id: str) -> Path:
    return results_dir(problem_id, candidate_id) / CURVES_NPZ


def list_results(problem_id: str) -> list[str]:
    root = data_root() / RESULTS_DIR / problem_id
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / SUMMARY_JSON).is_file())


def read_summary(problem_id: str, candidate_id: str) -> dict:
    return read_json(summary_path(problem_id, candidate_id))
