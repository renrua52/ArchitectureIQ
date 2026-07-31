"""Build a private, artifact-only surprise catalog for an attested release.

Only questions declared by :class:`release_manifest.QuizManifest` are read.
This module never discovers questions, imports generated training code, reruns
ground truth, or writes scores back into a release.  The returned rows are
private policy inputs: callers must expose only the answer-safe result produced
by :mod:`surprise_recommender`.
"""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

try:  # Package import in tools; top-level import in the Streamlit app/tests.
    from .feedback import compute_question_version
    from .release_manifest import ManifestQuestion, QuizManifest
    from .surprise_recommender import (
        RecommendationCandidate,
        QuestionIdentity,
        cold_start_prior,
    )
except ImportError:  # pragma: no cover - exercised by the app's import style.
    from feedback import compute_question_version
    from release_manifest import ManifestQuestion, QuizManifest
    from surprise_recommender import (
        RecommendationCandidate,
        QuestionIdentity,
        cold_start_prior,
    )


MIN_VALID_WIN_RATE = 0.7
_KNOWN_OPTIMIZERS = frozenset({"SGD", "Adam", "AdamW", "RMSprop", "Adagrad"})
_ADAPTIVE_OPTIMIZERS = frozenset({"Adam", "AdamW", "RMSprop", "Adagrad"})


class SurpriseCatalogError(ValueError):
    """Raised when a claimed catalog artifact is malformed or escapes its release."""


@dataclass(frozen=True, slots=True)
class SurpriseCatalogRow:
    """One private, auditable catalog row and its recommender-ready candidate."""

    candidate: RecommendationCandidate
    anti_heuristic: float | None
    ensemble_heuristic_wrong: bool | None
    heuristic_count: int
    invalid_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RecommendationCandidate):
            raise SurpriseCatalogError(
                "candidate must be a surprise_recommender.RecommendationCandidate"
            )
        if self.anti_heuristic is not None and not (
            math.isfinite(self.anti_heuristic) and 0.0 <= self.anti_heuristic <= 1.0
        ):
            raise SurpriseCatalogError("anti_heuristic must be None or in [0, 1]")
        if self.ensemble_heuristic_wrong is not None and not isinstance(
            self.ensemble_heuristic_wrong, bool
        ):
            raise SurpriseCatalogError(
                "ensemble_heuristic_wrong must be None or a bool"
            )
        if (
            isinstance(self.heuristic_count, bool)
            or not isinstance(self.heuristic_count, int)
            or self.heuristic_count < 0
        ):
            raise SurpriseCatalogError("heuristic_count must be a non-negative int")
        if self.heuristic_count == 0 and (
            self.anti_heuristic is not None or self.ensemble_heuristic_wrong is not None
        ):
            raise SurpriseCatalogError(
                "questions with no discriminative heuristics cannot claim signals"
            )
        if self.heuristic_count > 0 and (
            self.anti_heuristic is None or self.ensemble_heuristic_wrong is None
        ):
            raise SurpriseCatalogError(
                "discriminative heuristics require both private surprise signals"
            )
        if not isinstance(self.invalid_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.invalid_reasons
        ):
            raise SurpriseCatalogError("invalid_reasons must be non-empty strings")
        if self.candidate.valid is (bool(self.invalid_reasons)):
            raise SurpriseCatalogError(
                "candidate.valid must be the inverse of invalid_reasons"
            )

    @property
    def identity(self) -> QuestionIdentity:
        """Return the exact release/question/version identity."""
        return self.candidate.identity


@dataclass(frozen=True, order=True, slots=True)
class _OptimizerAggressiveness:
    """Lexicographic shortcut: adaptivity dominates learning rate."""

    adaptive: int
    learning_rate: float


@dataclass(frozen=True, slots=True)
class _ModelFeatures:
    parameter_count: int
    depth: int
    width: int


@dataclass(frozen=True, slots=True)
class _ChoiceFeatures:
    letter: str
    model: _ModelFeatures | None
    optimizer: _OptimizerAggressiveness | None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SurpriseCatalogError(f"{label} must be a regular, non-symlink file")
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except SurpriseCatalogError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise SurpriseCatalogError(
            f"cannot read strict JSON from {label}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise SurpriseCatalogError(f"{label} must contain a JSON object")
    return document


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SurpriseCatalogError(f"{field} must be a JSON object")
    return value


def _identifier(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise SurpriseCatalogError(
            f"{field} must be a non-empty identifier without surrounding whitespace"
        )
    return value


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SurpriseCatalogError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise SurpriseCatalogError(f"{field} must be at least {minimum}")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SurpriseCatalogError(f"{field} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise SurpriseCatalogError(f"{field} must be finite")
    return resolved


def _positive_dimension(value: Any, *, field: str) -> int:
    return _integer(value, field=field, minimum=1)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_release_path(
    root: Path,
    reference: Any,
    *,
    field: str,
    kind: str,
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise SurpriseCatalogError(f"{field} must be a non-empty relative path")
    if "\\" in reference:
        raise SurpriseCatalogError(f"{field} must use POSIX path separators")
    pure = PurePosixPath(reference)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != reference
    ):
        raise SurpriseCatalogError(f"{field} must be a normalized relative path")

    unresolved = root.joinpath(*pure.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise SurpriseCatalogError(f"cannot resolve {field}: {exc}") from exc
    if not _is_relative_to(resolved, root):
        raise SurpriseCatalogError(f"{field} escapes the release data root")

    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise SurpriseCatalogError(f"{field} cannot traverse a symbolic link")
    if kind == "dir" and not resolved.is_dir():
        raise SurpriseCatalogError(f"{field} must reference a directory")
    if kind == "file" and not resolved.is_file():
        raise SurpriseCatalogError(f"{field} must reference a file")
    return resolved


def _file_inside(candidate_dir: Path, relative: str, *, label: str) -> Path:
    path = candidate_dir / PurePosixPath(relative)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SurpriseCatalogError(f"cannot resolve {label}: {exc}") from exc
    if not _is_relative_to(resolved, candidate_dir):
        raise SurpriseCatalogError(f"{label} escapes its candidate directory")
    current = candidate_dir
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise SurpriseCatalogError(f"{label} cannot traverse a symbolic link")
    return resolved


def _mlp_features(model: Mapping[str, Any], *, field: str) -> _ModelFeatures:
    input_dim = _positive_dimension(
        model.get("input_dim", 1), field=f"{field}.input_dim"
    )
    depth = _positive_dimension(model.get("depth"), field=f"{field}.depth")
    width = _positive_dimension(model.get("width"), field=f"{field}.width")
    layer_norm = model.get("layer_norm")
    if (
        not isinstance(layer_norm, list)
        or len(layer_norm) != depth
        or not all(isinstance(item, bool) for item in layer_norm)
    ):
        raise SurpriseCatalogError(
            f"{field}.layer_norm must contain one bool for each MLP block"
        )

    # This mirrors the generated MLP exactly: input projection, ``depth``
    # width-to-width blocks, optional LayerNorms, and a scalar output head.
    parameter_count = input_dim * width + width
    parameter_count += depth * (width * width + width)
    parameter_count += 2 * width * sum(layer_norm)
    parameter_count += width + 1
    return _ModelFeatures(
        parameter_count=parameter_count,
        depth=depth,
        width=width,
    )


def _transformer_features(model: Mapping[str, Any], *, field: str) -> _ModelFeatures:
    d_model_value = model.get("d_model", model.get("embed_dim"))
    d_ff_value = model.get("d_ff", model.get("ff_dim"))
    d_model = _positive_dimension(d_model_value, field=f"{field}.d_model")
    d_ff = _positive_dimension(d_ff_value, field=f"{field}.d_ff")
    layers = _positive_dimension(model.get("num_layers"), field=f"{field}.num_layers")
    heads = _positive_dimension(model.get("num_heads"), field=f"{field}.num_heads")
    vocab = _positive_dimension(model.get("vocab_size"), field=f"{field}.vocab_size")
    context = _positive_dimension(
        model.get("context_length"), field=f"{field}.context_length"
    )
    if d_model % heads:
        raise SurpriseCatalogError(f"{field}.d_model must be divisible by num_heads")

    # PyTorch TransformerEncoderLayer parameters: Q/K/V + attention output,
    # two FFN matrices, their biases, and two affine LayerNorms.
    embeddings = vocab * d_model + context * d_model
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff
    per_layer += 9 * d_model + d_ff
    output_head = d_model * vocab + vocab
    return _ModelFeatures(
        parameter_count=embeddings + layers * per_layer + output_head,
        depth=layers,
        width=d_model,
    )


def _model_features(model: Mapping[str, Any], *, field: str) -> _ModelFeatures | None:
    model_type = model.get("type")
    if model_type == "mlp":
        return _mlp_features(model, field=field)
    if model_type in {"transformer", "transformer_lm"}:
        return _transformer_features(model, field=field)
    # Unknown plugins are not assigned zero-valued features.  Omitting these
    # shortcuts prevents an unsupported model from looking artificially small.
    return None


def _optimizer_aggressiveness(
    optimizer: Mapping[str, Any], *, field: str
) -> _OptimizerAggressiveness | None:
    optimizer_type = optimizer.get("type")
    if optimizer_type not in _KNOWN_OPTIMIZERS:
        return None
    learning_rate = _finite_number(optimizer.get("lr"), field=f"{field}.lr")
    if learning_rate < 0:
        raise SurpriseCatalogError(f"{field}.lr must be non-negative")
    return _OptimizerAggressiveness(
        adaptive=int(optimizer_type in _ADAPTIVE_OPTIMIZERS),
        learning_rate=learning_rate,
    )


def _winner_distribution(
    values: Mapping[str, Any], *, maximize: bool
) -> dict[str, Fraction] | None:
    if not values or any(value is None for value in values.values()):
        return None
    unique = set(values.values())
    if len(unique) == 1:
        return None
    target = max(unique) if maximize else min(unique)
    winners = frozenset(letter for letter, value in values.items() if value == target)
    share = Fraction(1, len(winners))
    return {letter: share if letter in winners else Fraction(0, 1) for letter in values}


def _surprise_signals(
    features: Mapping[str, _ChoiceFeatures], correct_letter: str
) -> tuple[float | None, bool | None, int]:
    model_available = all(item.model is not None for item in features.values())
    distributions: list[dict[str, Fraction]] = []
    if model_available:
        parameters = {
            letter: item.model.parameter_count  # type: ignore[union-attr]
            for letter, item in features.items()
        }
        depths = {
            letter: item.model.depth  # type: ignore[union-attr]
            for letter, item in features.items()
        }
        widths = {
            letter: item.model.width  # type: ignore[union-attr]
            for letter, item in features.items()
        }
        for values, maximize in (
            (parameters, True),
            (parameters, False),
            (depths, True),
            (widths, True),
        ):
            distribution = _winner_distribution(values, maximize=maximize)
            if distribution is not None:
                distributions.append(distribution)

    optimizers = {letter: item.optimizer for letter, item in features.items()}
    optimizer_distribution = _winner_distribution(optimizers, maximize=True)
    if optimizer_distribution is not None:
        distributions.append(optimizer_distribution)

    if not distributions:
        return None, None, 0

    heuristic_count = len(distributions)
    correct_probability = (
        sum(
            (distribution[correct_letter] for distribution in distributions),
            start=Fraction(0, 1),
        )
        / heuristic_count
    )
    anti_heuristic = float(1 - correct_probability)

    ensemble = {
        letter: sum(
            (distribution[letter] for distribution in distributions),
            start=Fraction(0, 1),
        )
        / heuristic_count
        for letter in features
    }
    correct_score = ensemble[correct_letter]
    best_other = max(
        score for letter, score in ensemble.items() if letter != correct_letter
    )
    # Conservative tie rule: a tied top score is not enough evidence to claim
    # that the ensemble is wrong.  Only a strictly better wrong choice is.
    ensemble_wrong = best_other > correct_score
    return anti_heuristic, ensemble_wrong, heuristic_count


def _question_evaluation(
    question: Mapping[str, Any], *, question_id: str
) -> tuple[str, int, int, tuple[str, ...]]:
    evaluation = _mapping(question.get("evaluation"), field=f"{question_id}.evaluation")
    metric = _identifier(
        evaluation.get("selection_metric"),
        field=f"{question_id}.evaluation.selection_metric",
    )
    n_seeds = _integer(
        evaluation.get("n_seeds"),
        field=f"{question_id}.evaluation.n_seeds",
        minimum=1,
    )
    base_seed = _integer(
        evaluation.get("base_seed"), field=f"{question_id}.evaluation.base_seed"
    )

    significance = _mapping(
        question.get("significance"), field=f"{question_id}.significance"
    )
    passed = significance.get("passed")
    if not isinstance(passed, bool):
        raise SurpriseCatalogError(f"{question_id}.significance.passed must be a bool")
    if significance.get("metric") != metric:
        raise SurpriseCatalogError(
            f"{question_id}.significance.metric must match the selection metric"
        )
    gap = _finite_number(
        significance.get("gap"), field=f"{question_id}.significance.gap"
    )
    win_rate = _finite_number(
        significance.get("win_rate"), field=f"{question_id}.significance.win_rate"
    )
    if not 0.0 <= win_rate <= 1.0:
        raise SurpriseCatalogError(
            f"{question_id}.significance.win_rate must be in [0, 1]"
        )

    invalid: list[str] = []
    if not passed:
        invalid.append("significance_not_passed")
    if gap <= 0.0:
        invalid.append("non_positive_significance_gap")
    if win_rate < MIN_VALID_WIN_RATE:
        invalid.append("win_rate_below_threshold")
    return metric, n_seeds, base_seed, tuple(invalid)


def _load_choice(
    root: Path,
    question: Mapping[str, Any],
    raw_choice: Any,
    *,
    index: int,
    metric: str,
    n_seeds: int,
    base_seed: int,
) -> tuple[_ChoiceFeatures, str, str, Path, tuple[str, ...]]:
    question_id = str(question["question_id"])
    field = f"{question_id}.choices[{index}]"
    choice = _mapping(raw_choice, field=field)
    letter = choice.get("letter")
    if not isinstance(letter, str) or len(letter) != 1 or not "A" <= letter <= "Z":
        raise SurpriseCatalogError(f"{field}.letter must be one uppercase ASCII letter")
    candidate_id = _identifier(
        choice.get("candidate_id"), field=f"{field}.candidate_id"
    )
    candidate_reference = choice.get("candidate_path")
    candidate_dir = _resolve_release_path(
        root,
        candidate_reference,
        field=f"{field}.candidate_path",
        kind="dir",
    )
    assert isinstance(candidate_reference, str)  # validated by resolver
    candidate_parts = PurePosixPath(candidate_reference).parts
    if (
        len(candidate_parts) < 6
        or candidate_parts[0] != "datasets"
        or candidate_parts[1] != question["family"]
        or candidate_parts[2] != question["dataset_id"]
        or candidate_parts[3] != "candidates"
        or candidate_parts[-1] != candidate_id
    ):
        raise SurpriseCatalogError(
            f"{field}.candidate_path does not match its family/dataset/candidate identity"
        )
    set_reference = choice.get("candidate_set_path")
    set_dir = _resolve_release_path(
        root,
        set_reference,
        field=f"{field}.candidate_set_path",
        kind="dir",
    )
    if candidate_dir.parent != set_dir:
        raise SurpriseCatalogError(
            f"{field}.candidate_path is not inside candidate_set_path"
        )

    spec = _read_json(
        _file_inside(candidate_dir, "candidate_spec.json", label=f"{field} spec"),
        label=f"{field}.candidate_spec.json",
    )
    for identity_field, expected in (
        ("candidate_id", candidate_id),
        ("family", question["family"]),
        ("dataset_id", question["dataset_id"]),
    ):
        if spec.get(identity_field) != expected:
            raise SurpriseCatalogError(
                f"{field} candidate_spec.{identity_field} does not match the question"
            )
    model = _mapping(spec.get("model"), field=f"{field}.candidate_spec.model")
    optimizer = _mapping(
        spec.get("optimizer"), field=f"{field}.candidate_spec.optimizer"
    )

    summary = _read_json(
        _file_inside(
            candidate_dir,
            "results/summary.json",
            label=f"{field} summary",
        ),
        label=f"{field}.results/summary.json",
    )
    if summary.get("candidate_id") != candidate_id:
        raise SurpriseCatalogError(f"{field} summary candidate_id does not match")
    if summary.get("selection_metric") != metric:
        raise SurpriseCatalogError(f"{field} summary selection_metric does not match")
    if summary.get("execution") != "candidate_py_files":
        raise SurpriseCatalogError(
            f"{field} summary must record execution='candidate_py_files'"
        )
    if (
        _integer(summary.get("n_seeds"), field=f"{field} summary.n_seeds", minimum=1)
        != n_seeds
        or _integer(summary.get("base_seed"), field=f"{field} summary.base_seed")
        != base_seed
    ):
        raise SurpriseCatalogError(f"{field} summary seed configuration does not match")
    failed_seeds = _integer(
        summary.get("failed_seeds"), field=f"{field} summary.failed_seeds", minimum=0
    )
    if failed_seeds > n_seeds:
        raise SurpriseCatalogError(f"{field} summary.failed_seeds exceeds n_seeds")
    excluded = summary.get("excluded")
    if not isinstance(excluded, bool):
        raise SurpriseCatalogError(f"{field} summary.excluded must be a bool")
    mean_value = summary.get(f"mean_{metric}")
    std_value = summary.get(f"std_{metric}")
    if failed_seeds == n_seeds:
        if mean_value is not None or std_value is not None:
            raise SurpriseCatalogError(
                f"{field} fully failed summary metrics must be null"
            )
    else:
        _finite_number(mean_value, field=f"{field} summary.mean_{metric}")
        std = _finite_number(std_value, field=f"{field} summary.std_{metric}")
        if std < 0:
            raise SurpriseCatalogError(
                f"{field} summary.std_{metric} must be non-negative"
            )

    invalid: list[str] = []
    choice_excluded = choice.get("excluded", False)
    if not isinstance(choice_excluded, bool):
        raise SurpriseCatalogError(f"{field}.excluded must be a bool when present")
    if choice_excluded:
        invalid.append("choice_marked_excluded")
    if failed_seeds:
        invalid.append("summary_failed_seeds")
    if excluded:
        invalid.append("summary_excluded")

    return (
        _ChoiceFeatures(
            letter=letter,
            model=_model_features(model, field=f"{field}.candidate_spec.model"),
            optimizer=_optimizer_aggressiveness(
                optimizer, field=f"{field}.candidate_spec.optimizer"
            ),
        ),
        candidate_id,
        candidate_reference,
        candidate_dir,
        tuple(invalid),
    )


def _load_question_row(
    manifest: QuizManifest,
    record: ManifestQuestion,
    directory: Path,
    *,
    exposure_count: int,
) -> SurpriseCatalogRow:
    root = manifest.data_root.resolve()
    question = _read_json(
        directory / "question.json", label=f"{record.path}/question.json"
    )
    for field, expected in (
        ("question_id", record.question_id),
        ("family", record.family),
        ("dataset_id", record.dataset_id),
    ):
        if question.get(field) != expected:
            raise SurpriseCatalogError(
                f"manifest question {record.question_id!r} {field} does not match question.json"
            )
    if directory.name != record.question_id:
        raise SurpriseCatalogError(
            f"manifest question {record.question_id!r} ID does not match its directory"
        )
    observed_version = compute_question_version(question)
    if observed_version != record.version:
        raise SurpriseCatalogError(
            f"manifest question {record.question_id!r} version does not match question.json"
        )
    if (
        manifest.release_id_for(
            record.question_id,
            record.version,
            question_path=directory,
        )
        != manifest.release_id
    ):
        raise SurpriseCatalogError(
            f"manifest question {record.question_id!r} has no exact release identity"
        )

    metric, n_seeds, base_seed, significance_invalid = _question_evaluation(
        question, question_id=record.question_id
    )
    raw_choices = question.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise SurpriseCatalogError(
            f"{record.question_id}.choices must contain at least two choices"
        )
    if question.get("num_choices") not in {None, len(raw_choices)}:
        raise SurpriseCatalogError(
            f"{record.question_id}.num_choices does not match choices"
        )

    features: dict[str, _ChoiceFeatures] = {}
    candidate_ids: set[str] = set()
    candidate_references: set[str] = set()
    candidate_dirs: set[Path] = set()
    invalid = list(significance_invalid)
    for index, raw_choice in enumerate(raw_choices):
        feature, candidate_id, reference, candidate_dir, choice_invalid = _load_choice(
            root,
            question,
            raw_choice,
            index=index,
            metric=metric,
            n_seeds=n_seeds,
            base_seed=base_seed,
        )
        if feature.letter in features:
            raise SurpriseCatalogError(
                f"{record.question_id}.choices contains duplicate letter {feature.letter!r}"
            )
        if candidate_id in candidate_ids:
            raise SurpriseCatalogError(
                f"{record.question_id}.choices contains duplicate candidate_id"
            )
        if reference in candidate_references or candidate_dir in candidate_dirs:
            raise SurpriseCatalogError(
                f"{record.question_id}.choices contains duplicate candidate path"
            )
        features[feature.letter] = feature
        candidate_ids.add(candidate_id)
        candidate_references.add(reference)
        candidate_dirs.add(candidate_dir)
        invalid.extend(choice_invalid)

    correct_letter = question.get("correct_letter")
    if not isinstance(correct_letter, str) or correct_letter not in features:
        raise SurpriseCatalogError(
            f"{record.question_id}.correct_letter must name exactly one choice"
        )
    anti_heuristic, ensemble_wrong, heuristic_count = _surprise_signals(
        features, correct_letter
    )
    if heuristic_count:
        posterior = cold_start_prior(
            anti_heuristic=anti_heuristic,
            ensemble_heuristic_wrong=ensemble_wrong,
        )
    else:
        # No discriminative supported shortcut is evidence neither for nor
        # against surprise.  A neutral 0.5 input maps to a symmetric Beta prior.
        posterior = cold_start_prior(anti_heuristic=0.5)

    identity = QuestionIdentity(
        release_id=manifest.release_id,
        question_id=record.question_id,
        question_version=record.version,
    )
    invalid_reasons = tuple(sorted(set(invalid)))
    candidate = RecommendationCandidate(
        identity=identity,
        family=record.family,
        posterior=posterior,
        exposure_count=exposure_count,
        valid=not invalid_reasons,
    )
    return SurpriseCatalogRow(
        candidate=candidate,
        anti_heuristic=anti_heuristic,
        ensemble_heuristic_wrong=ensemble_wrong,
        heuristic_count=heuristic_count,
        invalid_reasons=invalid_reasons,
    )


def _validated_exposures(
    exposure_counts: Mapping[QuestionIdentity, int] | None,
) -> dict[QuestionIdentity, int]:
    if exposure_counts is None:
        return {}
    if not isinstance(exposure_counts, Mapping):
        raise SurpriseCatalogError(
            "exposure_counts must map exact QuestionIdentity values to counts"
        )
    validated: dict[QuestionIdentity, int] = {}
    for identity, value in exposure_counts.items():
        if not isinstance(identity, QuestionIdentity):
            raise SurpriseCatalogError(
                "every exposure_counts key must be an exact QuestionIdentity"
            )
        validated[identity] = _integer(
            value,
            field=f"exposure_counts[{identity.question_id!r}]",
            minimum=0,
        )
    return validated


def build_surprise_catalog(
    manifest: QuizManifest,
    *,
    exposure_counts: Mapping[QuestionIdentity, int] | None = None,
) -> list[SurpriseCatalogRow]:
    """Load private cold-start rows for exactly one attested quiz manifest.

    Missing exposure entries default to zero.  Supplied keys must use the exact
    release/question/version identity, and keys outside this manifest are
    rejected so counts from another release cannot silently bleed into policy.
    """
    if not isinstance(manifest, QuizManifest):
        raise SurpriseCatalogError("manifest must be an attested QuizManifest")
    requested_root = manifest.data_root
    if requested_root.is_symlink():
        raise SurpriseCatalogError("manifest.data_root cannot be a symbolic link")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise SurpriseCatalogError(f"cannot resolve manifest.data_root: {exc}") from exc
    if not root.is_dir():
        raise SurpriseCatalogError("manifest.data_root must be a directory")

    records = manifest.questions
    declared_dirs = manifest.question_dirs()
    if not records or len(records) != len(declared_dirs):
        raise SurpriseCatalogError(
            "manifest questions and manifest.question_dirs() must be non-empty and aligned"
        )
    exposures = _validated_exposures(exposure_counts)
    rows: list[SurpriseCatalogRow] = []
    seen_identities: set[QuestionIdentity] = set()
    seen_paths: set[Path] = set()
    for index, (record, declared_dir) in enumerate(
        zip(records, declared_dirs, strict=True)
    ):
        if not isinstance(record, ManifestQuestion):
            raise SurpriseCatalogError(f"manifest.questions[{index}] is invalid")
        expected_dir = _resolve_release_path(
            root,
            record.path,
            field=f"manifest.questions[{index}].path",
            kind="dir",
        )
        try:
            observed_dir = Path(declared_dir).resolve(strict=True)
        except OSError as exc:
            raise SurpriseCatalogError(
                f"cannot resolve manifest.question_dirs()[{index}]: {exc}"
            ) from exc
        if observed_dir != expected_dir:
            raise SurpriseCatalogError(
                f"manifest.question_dirs()[{index}] does not match its question record"
            )
        identity = QuestionIdentity(
            manifest.release_id,
            record.question_id,
            record.version,
        )
        if identity in seen_identities:
            raise SurpriseCatalogError(f"duplicate manifest identity: {identity}")
        if expected_dir in seen_paths:
            raise SurpriseCatalogError(
                f"duplicate manifest question path: {record.path}"
            )
        seen_identities.add(identity)
        seen_paths.add(expected_dir)
        rows.append(
            _load_question_row(
                manifest,
                record,
                expected_dir,
                exposure_count=exposures.get(identity, 0),
            )
        )

    unknown_exposures = set(exposures) - seen_identities
    if unknown_exposures:
        first = min(unknown_exposures)
        raise SurpriseCatalogError(
            "exposure_counts contains an identity outside this manifest: "
            f"{first.question_id}/{first.question_version}"
        )
    return rows


def build_recommendation_candidates(
    manifest: QuizManifest,
    *,
    exposure_counts: Mapping[QuestionIdentity, int] | None = None,
) -> list[RecommendationCandidate]:
    """Return the catalog in the direct input shape expected by ``Next`` policy."""
    return [
        row.candidate
        for row in build_surprise_catalog(
            manifest,
            exposure_counts=exposure_counts,
        )
    ]
