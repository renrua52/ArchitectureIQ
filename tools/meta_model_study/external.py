"""Blind external-evaluation inputs and scoring for the meta-model study.

This module deliberately has no dependency on any fitted meta-model.  The
prediction phase reads only the sanitized questions and turns each public
choice into the same ``setting``/``derived`` shape used by the training data.
The answer key is opened only by :func:`score_predictions`, after unscored
predictions have already been written.

No function in this module resolves candidate paths or reads candidate result
files.  In particular, parameter counts are computed directly from the public
model spec through the registered :class:`~architecture_iq.models.base.ModelFamily`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from architecture_iq.registry import ensure_registries, get_model_type


EXTERNAL_INPUT_SCHEMA_VERSION = "meta_model_external_input_v1"
UNSCORED_PREDICTIONS_SCHEMA_VERSION = "meta_model_unscored_predictions_v1"
SCORED_PREDICTIONS_SCHEMA_VERSION = "meta_model_scored_predictions_v1"

# Each family is fitted independently, so external choices must be routed to
# the experiment trained for that family and fixed benchmark environment.
FAMILY_TO_EXPERIMENT: dict[str, str] = {
    "univariate_regression": "univariate_sym_62678b_b2048_bs32_mse",
    "multivariate_regression": "multivariate_mvar_c59a30_b5120_bs32_mse",
    "bigram_lm": "bigram_bg_0021c1_b5120_bs64_cross_entropy",
}

_QUESTION_FORBIDDEN_FIELDS = {
    "answer",
    "answer_key",
    "correct_letter",
    "ground_truth",
    "results",
    "seed_results",
    "significance",
    "summary",
    "target",
}
_CHOICE_FORBIDDEN_FIELDS = _QUESTION_FORBIDDEN_FIELDS | {
    "candidate_path",
    "candidate_set_path",
}
_UNSCORED_FORBIDDEN_FIELDS = {
    "answer_key",
    "correct_letter",
    "ground_truth",
    "is_correct",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_for_family(
    family: str,
    family_to_experiment: Mapping[str, str] = FAMILY_TO_EXPERIMENT,
) -> str:
    """Return the independently trained experiment serving ``family``."""

    try:
        experiment_id = family_to_experiment[family]
    except KeyError as exc:
        known = ", ".join(sorted(family_to_experiment))
        raise ValueError(f"No meta-model experiment for family {family!r}; known: {known}") from exc
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(f"Invalid experiment ID for family {family!r}: {experiment_id!r}")
    return experiment_id


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _reject_fields(value: Mapping[str, Any], forbidden: set[str], context: str) -> None:
    leaked = sorted(set(value).intersection(forbidden))
    if leaked:
        raise ValueError(f"{context} contains non-blind fields: {', '.join(leaked)}")


def _validate_budget(budget: dict[str, Any], context: str) -> None:
    required = ("training_steps", "batch_size", "total_samples_seen")
    missing = [field for field in required if field not in budget]
    if missing:
        raise ValueError(f"{context} is missing budget fields: {', '.join(missing)}")
    try:
        steps = int(budget["training_steps"])
        batch_size = int(budget["batch_size"])
        samples = int(budget["total_samples_seen"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has a non-integer training budget") from exc
    if min(steps, batch_size, samples) <= 0:
        raise ValueError(f"{context} training budget values must be positive")
    if steps * batch_size != samples:
        raise ValueError(
            f"{context} violates training_steps * batch_size = total_samples_seen"
        )


def choice_to_example(
    choice: Mapping[str, Any],
    *,
    include_parameter_count: bool = True,
) -> dict[str, Any]:
    """Build a target-free model example from one sanitized public choice.

    The exact parameter feature is obtained by constructing the registry model
    from ``choice['model']``.  The RNG fork prevents parameter initialization
    from perturbing the caller's Torch RNG stream.
    """

    choice_obj = _require_object(choice, "choice")
    _reject_fields(choice_obj, _CHOICE_FORBIDDEN_FIELDS, "choice")
    setting = {
        key: deepcopy(_require_object(choice_obj.get(key), f"choice.{key}"))
        for key in ("budget", "model", "optimizer", "loss")
    }
    _validate_budget(setting["budget"], "choice")
    derived: dict[str, Any] = {}
    if include_parameter_count:
        # Keep Torch lazy.  On macOS, importing PyTorch before XGBoost 3.2 can
        # crash the latter's native OpenMP runtime.  The blind input-preparation
        # phase runs separately from fitted-model prediction for this reason.
        import torch

        model_type = _require_nonempty_string(
            setting["model"].get("type"),
            "choice.model.type",
        )
        ensure_registries()
        with torch.random.fork_rng():
            module = get_model_type(model_type).build_module(setting["model"])
            total_params = sum(parameter.numel() for parameter in module.parameters())
            trainable_params = sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        if total_params <= 0:
            raise ValueError(f"choice.model {model_type!r} has no parameters")
        if not 0 <= trainable_params <= total_params:
            raise ValueError(
                "choice.model returned inconsistent trainable/total parameter counts"
            )
        derived = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "log_total_params": math.log(total_params),
        }

    return {
        "setting": setting,
        "derived": derived,
    }


def load_prediction_inputs(
    questions_sanitized_path: Path,
    *,
    family_to_experiment: Mapping[str, str] = FAMILY_TO_EXPERIMENT,
    include_parameter_count: bool = True,
) -> list[dict[str, Any]]:
    """Load sanitized questions as target-free, question-oriented inputs.

    Only ``questions_sanitized_path`` is opened.  Candidate IDs are retained as
    opaque audit identifiers; candidate paths are rejected and never followed.
    """

    raw = json.loads(questions_sanitized_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions_sanitized must be a non-empty JSON list")

    inputs: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for question_index, raw_question in enumerate(raw):
        context = f"questions_sanitized[{question_index}]"
        question = _require_object(raw_question, context)
        _reject_fields(question, _QUESTION_FORBIDDEN_FIELDS, context)
        question_id = _require_nonempty_string(
            question.get("question_id"), f"{context}.question_id"
        )
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate sanitized question_id: {question_id}")
        seen_question_ids.add(question_id)
        family = _require_nonempty_string(
            question.get("family"), f"{context}.family"
        )
        experiment_id = experiment_for_family(family, family_to_experiment)
        raw_choices = question.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"{context}.choices must be a non-empty list")

        choices: list[dict[str, Any]] = []
        seen_letters: set[str] = set()
        seen_candidates: set[str] = set()
        for choice_index, raw_choice in enumerate(raw_choices):
            choice_context = f"{context}.choices[{choice_index}]"
            choice = _require_object(raw_choice, choice_context)
            _reject_fields(choice, _CHOICE_FORBIDDEN_FIELDS, choice_context)
            letter = _require_nonempty_string(
                choice.get("letter"), f"{choice_context}.letter"
            )
            candidate_id = _require_nonempty_string(
                choice.get("candidate_id"), f"{choice_context}.candidate_id"
            )
            if letter in seen_letters:
                raise ValueError(f"Duplicate choice letter {letter!r} in {question_id}")
            if candidate_id in seen_candidates:
                raise ValueError(
                    f"Duplicate candidate_id {candidate_id!r} in {question_id}"
                )
            seen_letters.add(letter)
            seen_candidates.add(candidate_id)
            choices.append(
                {
                    "letter": letter,
                    "candidate_id": candidate_id,
                    "example": choice_to_example(
                        choice,
                        include_parameter_count=include_parameter_count,
                    ),
                }
            )

        inputs.append(
            {
                "schema_version": EXTERNAL_INPUT_SCHEMA_VERSION,
                "question_id": question_id,
                "question_run_id": question.get("question_run_id"),
                "family": family,
                "experiment_id": experiment_id,
                "dataset_id": _require_nonempty_string(
                    question.get("dataset_id"), f"{context}.dataset_id"
                ),
                "selection_metric": _require_nonempty_string(
                    question.get("selection_metric"),
                    f"{context}.selection_metric",
                ),
                "choices": choices,
            }
        )
    return inputs


def _walk_forbidden_unscored_fields(value: Any, context: str) -> None:
    if isinstance(value, dict):
        leaked = sorted(set(value).intersection(_UNSCORED_FORBIDDEN_FIELDS))
        if leaked:
            raise ValueError(
                f"{context} contains scored/answer fields: {', '.join(leaked)}"
            )
        for key, child in value.items():
            _walk_forbidden_unscored_fields(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_unscored_fields(child, f"{context}[{index}]")


def _validated_prediction_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("predictions must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        context = f"predictions[{index}]"
        row = deepcopy(_require_object(raw_row, context))
        _walk_forbidden_unscored_fields(row, context)
        question_id = _require_nonempty_string(
            row.get("question_id"), f"{context}.question_id"
        )
        if question_id in seen:
            raise ValueError(f"Duplicate prediction question_id: {question_id}")
        seen.add(question_id)
        _require_nonempty_string(
            row.get("predicted_letter"), f"{context}.predicted_letter"
        )
        normalized.append(row)
    return normalized


def _atomic_write_json(path: Path, value: Any) -> None:
    serialized = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_unscored_predictions(
    output_path: Path,
    predictions: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write target-free predictions and return their SHA-256.

    This function never accepts or reads an answer-key path.  The returned hash
    is over the exact installed bytes, making the pre-scoring artifact easy to
    record in a run manifest before the private key is opened.
    """

    rows = _validated_prediction_rows(list(predictions))
    metadata_copy = deepcopy(dict(metadata or {}))
    _walk_forbidden_unscored_fields(metadata_copy, "metadata")
    payload = {
        "schema_version": UNSCORED_PREDICTIONS_SCHEMA_VERSION,
        "num_questions": len(rows),
        "metadata": metadata_copy,
        "predictions": rows,
    }
    _atomic_write_json(output_path, payload)
    return sha256_file(output_path)


def _load_answer_key(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "questions" in raw:
        raw = raw["questions"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("answer_key must be a non-empty JSON list")
    return [_require_object(row, f"answer_key[{index}]") for index, row in enumerate(raw)]


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("predictions")
    return _validated_prediction_rows(raw)


def _index_unique(
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        question_id = _require_nonempty_string(
            row.get("question_id"), f"{source}[{index}].question_id"
        )
        if question_id in indexed:
            raise ValueError(f"Duplicate {source} question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def score_predictions(
    predictions_path: Path,
    answer_key_path: Path,
) -> dict[str, Any]:
    """Score a frozen prediction artifact against the separately opened key.

    The prediction and answer-key question sets must match exactly.  Scoring
    does not load sanitized questions, candidate specs, or candidate summaries.
    """

    predictions_by_id = _index_unique(
        _load_predictions(predictions_path), source="predictions"
    )
    key_by_id = _index_unique(_load_answer_key(answer_key_path), source="answer_key")
    prediction_ids = set(predictions_by_id)
    key_ids = set(key_by_id)
    if prediction_ids != key_ids:
        missing = sorted(key_ids - prediction_ids)
        unexpected = sorted(prediction_ids - key_ids)
        raise ValueError(
            "Prediction/answer-key question sets differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    questions: list[dict[str, Any]] = []
    family_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"num_questions": 0, "num_correct": 0}
    )
    for question_id, key in key_by_id.items():
        prediction = predictions_by_id[question_id]
        family = _require_nonempty_string(
            key.get("family"), f"answer_key[{question_id}].family"
        )
        predicted_family = prediction.get("family")
        if predicted_family is not None and predicted_family != family:
            raise ValueError(
                f"Family mismatch for {question_id}: "
                f"prediction={predicted_family!r}, answer_key={family!r}"
            )
        predicted_letter = _require_nonempty_string(
            prediction.get("predicted_letter"),
            f"predictions[{question_id}].predicted_letter",
        )
        correct_letter = _require_nonempty_string(
            key.get("correct_letter"),
            f"answer_key[{question_id}].correct_letter",
        )

        key_choices = key.get("choices", [])
        candidate_by_letter: dict[str, str] = {}
        if key_choices:
            if not isinstance(key_choices, list):
                raise TypeError(f"answer_key[{question_id}].choices must be a list")
            for choice_index, raw_choice in enumerate(key_choices):
                choice = _require_object(
                    raw_choice,
                    f"answer_key[{question_id}].choices[{choice_index}]",
                )
                letter = _require_nonempty_string(
                    choice.get("letter"),
                    f"answer_key[{question_id}].choices[{choice_index}].letter",
                )
                candidate_id = _require_nonempty_string(
                    choice.get("candidate_id"),
                    f"answer_key[{question_id}].choices[{choice_index}].candidate_id",
                )
                if letter in candidate_by_letter:
                    raise ValueError(
                        f"Duplicate answer-key choice letter {letter!r} in {question_id}"
                    )
                candidate_by_letter[letter] = candidate_id
            if predicted_letter not in candidate_by_letter:
                raise ValueError(
                    f"Invalid predicted letter {predicted_letter!r} for {question_id}"
                )
            if correct_letter not in candidate_by_letter:
                raise ValueError(
                    f"Invalid correct letter {correct_letter!r} for {question_id}"
                )
            supplied_candidate = prediction.get("predicted_candidate_id")
            if (
                supplied_candidate is not None
                and supplied_candidate != candidate_by_letter[predicted_letter]
            ):
                raise ValueError(
                    f"Predicted letter/candidate mismatch for {question_id}: "
                    f"{predicted_letter!r} maps to "
                    f"{candidate_by_letter[predicted_letter]!r}, got "
                    f"{supplied_candidate!r}"
                )

        is_correct = predicted_letter == correct_letter
        family_totals[family]["num_questions"] += 1
        family_totals[family]["num_correct"] += int(is_correct)
        questions.append(
            {
                "question_id": question_id,
                "family": family,
                "predicted_letter": predicted_letter,
                "predicted_candidate_id": candidate_by_letter.get(predicted_letter),
                "correct_letter": correct_letter,
                "correct_candidate_id": candidate_by_letter.get(correct_letter),
                "is_correct": is_correct,
            }
        )

    by_family = {}
    for family in sorted(family_totals):
        counts = family_totals[family]
        by_family[family] = {
            **counts,
            "accuracy": counts["num_correct"] / counts["num_questions"],
        }
    num_correct = sum(row["is_correct"] for row in questions)
    total = {
        "num_questions": len(questions),
        "num_correct": num_correct,
        "accuracy": num_correct / len(questions),
    }
    return {
        "schema_version": SCORED_PREDICTIONS_SCHEMA_VERSION,
        "predictions_sha256": sha256_file(predictions_path),
        "answer_key_sha256": sha256_file(answer_key_path),
        "total": total,
        "by_family": by_family,
        "questions": questions,
    }
