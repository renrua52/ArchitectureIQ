"""Schema constants and minimal validation for the backend storage layout.

Schemas are intentionally small and hand-validated (no jsonschema
dependency): the storage layer only enforces the fields the repository
itself needs. Family-specific fields (model/optimizer/loss internals)
are validated by the generator-side renderers.
"""
from __future__ import annotations

from typing import Any

# Column directories under ``backend/data/``.
PROBLEMS_DIR = "problems"
TRAINERS_DIR = "trainers"
CANDIDATES_DIR = "candidates"
RESULTS_DIR = "results"

# Schema versions written by this storage layer.
PROBLEM_SCHEMA_VERSION = "2.0"
CANDIDATE_SCHEMA_VERSION = "2.0"
TRAINER_SCHEMA_VERSION = "1.0"
RESULTS_SCHEMA_VERSION = "1.0"

# Fixed result filenames.
SUMMARY_JSON = "summary.json"
CURVES_NPZ = "curves.npz"
TRAINER_SPEC_JSON = "trainer_spec.json"
TRAINER_TRAIN_PY = "train.py"
PROBLEM_SPEC_JSON = "dataset_spec.json"
PROBLEM_README_MD = "README.md"

PROBLEM_SPEC_REQUIRED = ("schema_version", "problem_id", "family", "params", "selection_metric", "files", "dataset_id")
CANDIDATE_REQUIRED = ("schema_version", "problem_id", "candidate_id", "family", "budget", "model", "optimizer", "loss")
TRAINER_REQUIRED = ("schema_version", "trainer_id", "family", "version", "content_sha256", "source")
SUMMARY_REQUIRED = ("candidate_id", "selection_metric", "n_seeds", "mean_test_mse")


def _missing(data: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [k for k in required if k not in data]


def validate_problem_spec(data: dict[str, Any]) -> list[str]:
    """Return a list of missing/incorrect fields (empty means valid)."""
    errors = list(_missing(data, PROBLEM_SPEC_REQUIRED))
    if data.get("schema_version") != PROBLEM_SCHEMA_VERSION:
        errors.append("schema_version")
    return errors


def validate_candidate_config(data: dict[str, Any]) -> list[str]:
    errors = list(_missing(data, CANDIDATE_REQUIRED))
    if data.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        errors.append("schema_version")
    for key in ("budget", "model", "optimizer", "loss"):
        if key in data and not isinstance(data[key], dict):
            errors.append(key)
    return errors


def validate_trainer_spec(data: dict[str, Any]) -> list[str]:
    errors = list(_missing(data, TRAINER_REQUIRED))
    if data.get("schema_version") != TRAINER_SCHEMA_VERSION:
        errors.append("schema_version")
    return errors


def validate_summary(data: dict[str, Any]) -> list[str]:
    errors = list(_missing(data, SUMMARY_REQUIRED))
    if "mean_" + str(data.get("selection_metric", "")) not in data:
        errors.append(f"mean_{data.get('selection_metric')}")
    return errors
