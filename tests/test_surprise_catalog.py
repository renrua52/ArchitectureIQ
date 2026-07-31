"""Artifact fixtures for the private, manifest-scoped surprise catalog."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))

import release_manifest  # noqa: E402
import surprise_catalog as catalog  # noqa: E402
import surprise_recommender as recommender  # noqa: E402
from feedback import compute_question_version  # noqa: E402


FAMILY = "fixture_family"
DATASET_ID = "dataset_fixture"
RUN_ID = "run_fixture"
RELEASE_ID = "release_" + "a" * 64


def _mlp(*, depth: int, width: int, input_dim: int = 2) -> dict[str, Any]:
    return {
        "type": "mlp",
        "depth": depth,
        "width": width,
        "input_dim": input_dim,
        "layer_norm": [False] * depth,
        "activations": ["relu"] * depth,
        "residual": False,
    }


def _transformer(*, layers: int, width: int, d_ff: int) -> dict[str, Any]:
    return {
        "type": "transformer_lm",
        "vocab_size": 32,
        "context_length": 16,
        "d_model": width,
        "num_layers": layers,
        "num_heads": 2,
        "d_ff": d_ff,
    }


def _choice_definition(
    candidate_id: str,
    *,
    model: dict[str, Any],
    optimizer: str = "SGD",
    lr: float = 0.001,
    mean: float | None = 1.0,
    std: float | None = 0.1,
    failed_seeds: int = 0,
    excluded: bool = False,
    choice_excluded: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "model": model,
        "optimizer": {"type": optimizer, "lr": lr},
        "mean": mean,
        "std": std,
        "failed_seeds": failed_seeds,
        "excluded": excluded,
        "choice_excluded": choice_excluded,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _question_path(question_id: str) -> str:
    return f"datasets/{FAMILY}/{DATASET_ID}/questions/{RUN_ID}/{question_id}"


def _make_manifest(
    tmp_path: Path,
    *,
    question_id: str = "q_fixture",
    definitions: list[dict[str, Any]] | None = None,
    letters: list[str] | None = None,
    order: list[int] | None = None,
    correct_index: int = 0,
    metric: str = "test_mse",
    passed: bool = True,
    gap: float = 0.5,
    win_rate: float = 0.9,
) -> tuple[release_manifest.QuizManifest, Path, dict[str, Any]]:
    root = tmp_path / "release"
    default_definitions = [
        _choice_definition(
            "c_correct",
            model=_mlp(depth=1, width=4),
            optimizer="SGD",
            lr=0.001,
            mean=0.5,
        ),
        _choice_definition(
            "c_large",
            model=_mlp(depth=3, width=16),
            optimizer="Adam",
            lr=0.01,
            mean=1.0,
        ),
        _choice_definition(
            "c_medium",
            model=_mlp(depth=2, width=8),
            optimizer="SGD",
            lr=0.1,
            mean=1.5,
        ),
    ]
    definitions = deepcopy(definitions or default_definitions)
    letters = letters or [chr(ord("A") + index) for index in range(len(definitions))]
    order = order or list(range(len(definitions)))
    if len(letters) != len(definitions) or sorted(order) != list(
        range(len(definitions))
    ):
        raise AssertionError("invalid fixture permutation")

    choices_by_index: dict[int, dict[str, Any]] = {}
    for index, definition in enumerate(definitions):
        candidate_id = definition["candidate_id"]
        set_path = f"datasets/{FAMILY}/{DATASET_ID}/candidates/set_fixture"
        candidate_path = f"{set_path}/{candidate_id}"
        candidate_dir = root / candidate_path
        spec = {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "family": FAMILY,
            "dataset_id": DATASET_ID,
            "model": definition["model"],
            "optimizer": definition["optimizer"],
        }
        summary = {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "selection_metric": metric,
            "execution": "candidate_py_files",
            "n_seeds": 10,
            "base_seed": 0,
            "failed_seeds": definition["failed_seeds"],
            "excluded": definition["excluded"],
            f"mean_{metric}": definition["mean"],
            f"std_{metric}": definition["std"],
        }
        _write_json(candidate_dir / "candidate_spec.json", spec)
        _write_json(candidate_dir / "results" / "summary.json", summary)
        choice = {
            "letter": letters[index],
            "candidate_id": candidate_id,
            "candidate_path": candidate_path,
            "candidate_set_path": set_path,
        }
        if definition["choice_excluded"]:
            choice["excluded"] = True
        choices_by_index[index] = choice

    question = {
        "schema_version": "1.0",
        "question_id": question_id,
        "family": FAMILY,
        "dataset_id": DATASET_ID,
        "question_run_id": RUN_ID,
        "question_run_path": (f"datasets/{FAMILY}/{DATASET_ID}/questions/{RUN_ID}"),
        "type": "mixed",
        "num_choices": len(definitions),
        "choices": [choices_by_index[index] for index in order],
        "correct_letter": letters[correct_index],
        "evaluation": {
            "selection_metric": metric,
            "n_seeds": 10,
            "base_seed": 0,
        },
        "significance": {
            "passed": passed,
            "metric": metric,
            "gap": gap,
            "win_rate": win_rate,
        },
    }
    relative = _question_path(question_id)
    question_dir = root / relative
    _write_json(question_dir / "question.json", question)
    version = compute_question_version(question)
    record = release_manifest.ManifestQuestion(
        question_id=question_id,
        version=version,
        path=relative,
        family=FAMILY,
        dataset_id=DATASET_ID,
        source_run=RUN_ID,
        source_run_path=f"datasets/{FAMILY}/{DATASET_ID}/questions/{RUN_ID}",
    )
    manifest = release_manifest.QuizManifest(
        data_root=root,
        release_id=RELEASE_ID,
        generated_at=None,
        questions=(record,),
        source_run_count=1,
        manifest_sha256="b" * 64,
        artifact_count=1,
        artifact_bytes=1,
    )
    return manifest, question_dir, question


def _rewrite_question(
    manifest: release_manifest.QuizManifest,
    question_dir: Path,
    question: dict[str, Any],
) -> release_manifest.QuizManifest:
    _write_json(question_dir / "question.json", question)
    record = replace(manifest.questions[0], version=compute_question_version(question))
    return replace(manifest, questions=(record,))


def test_catalog_is_metric_letter_choice_and_path_order_independent(
    tmp_path: Path,
) -> None:
    first, _, _ = _make_manifest(tmp_path / "first", metric="test_mse")
    second, _, _ = _make_manifest(
        tmp_path / "second",
        metric="test_cross_entropy",
        letters=["C", "A", "B"],
        order=[2, 1, 0],
    )

    first_row = catalog.build_surprise_catalog(first)[0]
    second_row = catalog.build_surprise_catalog(second)[0]

    assert first_row.anti_heuristic == second_row.anti_heuristic == pytest.approx(0.8)
    assert first_row.ensemble_heuristic_wrong is True
    assert second_row.ensemble_heuristic_wrong is True
    assert first_row.heuristic_count == second_row.heuristic_count == 5
    assert first_row.candidate.posterior == second_row.candidate.posterior


def test_tied_heuristic_winners_split_probability_and_top_tie_is_conservative(
    tmp_path: Path,
) -> None:
    definitions = [
        _choice_definition("c_a", model=_mlp(depth=3, width=16)),
        _choice_definition("c_b", model=_mlp(depth=3, width=16)),
        _choice_definition("c_c", model=_mlp(depth=1, width=4)),
    ]
    manifest, _, _ = _make_manifest(tmp_path, definitions=definitions)

    row = catalog.build_surprise_catalog(manifest)[0]

    # Correct A shares max-parameter/deepest/widest with B (1/2 each), while
    # min-parameter picks C.  Optimizers are all equal and therefore omitted.
    assert row.heuristic_count == 4
    assert row.anti_heuristic == pytest.approx(5 / 8)
    assert row.ensemble_heuristic_wrong is False


def test_all_equal_heuristics_are_omitted_and_use_a_neutral_prior(
    tmp_path: Path,
) -> None:
    definitions = [
        _choice_definition("c_a", model=_mlp(depth=2, width=8)),
        _choice_definition("c_b", model=_mlp(depth=2, width=8)),
    ]
    manifest, _, _ = _make_manifest(tmp_path, definitions=definitions)

    row = catalog.build_surprise_catalog(manifest)[0]

    assert row.heuristic_count == 0
    assert row.anti_heuristic is None
    assert row.ensemble_heuristic_wrong is None
    assert row.candidate.posterior.mean == pytest.approx(0.5)


def test_transformer_features_and_unknown_models_do_not_get_zero_value_votes(
    tmp_path: Path,
) -> None:
    transformers = [
        _choice_definition(
            "c_small",
            model=_transformer(layers=1, width=32, d_ff=64),
            optimizer="SGD",
            lr=0.001,
        ),
        _choice_definition(
            "c_large",
            model=_transformer(layers=3, width=64, d_ff=128),
            optimizer="Adam",
            lr=0.001,
        ),
    ]
    transformer_manifest, _, _ = _make_manifest(
        tmp_path / "transformer",
        definitions=transformers,
    )
    transformer_row = catalog.build_surprise_catalog(transformer_manifest)[0]
    assert transformer_row.heuristic_count == 5
    assert transformer_row.anti_heuristic == pytest.approx(0.8)

    unknown = [
        _choice_definition(
            "c_unknown_a",
            model={"type": "future_plugin", "pretend_size": 1},
            optimizer="Adam",
            lr=0.01,
        ),
        _choice_definition(
            "c_unknown_b",
            model={"type": "future_plugin", "pretend_size": 999999},
            optimizer="SGD",
            lr=0.001,
        ),
    ]
    unknown_manifest, _, _ = _make_manifest(
        tmp_path / "unknown",
        definitions=unknown,
    )
    unknown_row = catalog.build_surprise_catalog(unknown_manifest)[0]
    assert unknown_row.heuristic_count == 1
    assert unknown_row.anti_heuristic == pytest.approx(0.0)
    assert unknown_row.ensemble_heuristic_wrong is False


@pytest.mark.parametrize(
    ("definition_change", "manifest_change", "expected_reasons"),
    [
        ({"failed_seeds": 1}, {}, {"summary_failed_seeds"}),
        (
            {"failed_seeds": 10, "mean": None, "std": None},
            {},
            {"summary_failed_seeds"},
        ),
        ({"excluded": True}, {}, {"summary_excluded"}),
        ({"choice_excluded": True}, {}, {"choice_marked_excluded"}),
        ({}, {"passed": False}, {"significance_not_passed"}),
        ({}, {"gap": 0.0}, {"non_positive_significance_gap"}),
        ({}, {"win_rate": 0.69}, {"win_rate_below_threshold"}),
    ],
)
def test_validity_hard_gates_keep_invalid_questions_out_of_next(
    tmp_path: Path,
    definition_change: dict[str, Any],
    manifest_change: dict[str, Any],
    expected_reasons: set[str],
) -> None:
    definition = _choice_definition("c_a", model=_mlp(depth=1, width=4))
    definition.update(definition_change)
    definitions = [
        definition,
        _choice_definition("c_b", model=_mlp(depth=2, width=8)),
    ]
    manifest, _, _ = _make_manifest(
        tmp_path,
        definitions=definitions,
        **manifest_change,
    )

    row = catalog.build_surprise_catalog(manifest)[0]

    assert row.candidate.valid is False
    assert expected_reasons.issubset(row.invalid_reasons)
    with pytest.raises(recommender.NoEligibleQuestionError):
        recommender.select_question([row.candidate], set(), seed=1)


def test_catalog_reads_only_manifest_question_dirs(tmp_path: Path) -> None:
    manifest, _, _ = _make_manifest(tmp_path)
    hidden = (
        manifest.data_root
        / "datasets"
        / FAMILY
        / DATASET_ID
        / "questions"
        / "another_run"
        / "q_not_published"
        / "question.json"
    )
    hidden.parent.mkdir(parents=True)
    hidden.write_text("this is deliberately invalid JSON", encoding="utf-8")

    rows = catalog.build_surprise_catalog(manifest)

    assert [row.identity.question_id for row in rows] == ["q_fixture"]


def test_exact_identity_drives_exposure_counts_and_candidate_output(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _make_manifest(tmp_path)
    record = manifest.questions[0]
    identity = recommender.QuestionIdentity(
        release_id=manifest.release_id,
        question_id=record.question_id,
        question_version=record.version,
    )

    candidates = catalog.build_recommendation_candidates(
        manifest,
        exposure_counts={identity: 7},
    )

    assert len(candidates) == 1
    assert candidates[0].identity == identity
    assert candidates[0].exposure_count == 7
    assert candidates[0].family == FAMILY

    wrong_version = replace(identity, question_version="qv1_wrong")
    with pytest.raises(catalog.SurpriseCatalogError, match="outside this manifest"):
        catalog.build_surprise_catalog(
            manifest,
            exposure_counts={wrong_version: 1},
        )
    with pytest.raises(catalog.SurpriseCatalogError, match="QuestionIdentity"):
        catalog.build_surprise_catalog(
            manifest,
            exposure_counts={record.question_id: 1},  # type: ignore[dict-item]
        )


def test_duplicate_manifest_identity_and_duplicate_choice_fail_closed(
    tmp_path: Path,
) -> None:
    manifest, question_dir, question = _make_manifest(tmp_path)
    duplicate_manifest = replace(
        manifest,
        questions=(manifest.questions[0], manifest.questions[0]),
    )
    with pytest.raises(catalog.SurpriseCatalogError, match="duplicate manifest"):
        catalog.build_surprise_catalog(duplicate_manifest)

    question["choices"][1]["letter"] = question["choices"][0]["letter"]
    manifest = _rewrite_question(manifest, question_dir, question)
    with pytest.raises(catalog.SurpriseCatalogError, match="duplicate letter"):
        catalog.build_surprise_catalog(manifest)


def test_missing_correct_letter_and_path_escape_fail_closed(tmp_path: Path) -> None:
    manifest, question_dir, question = _make_manifest(tmp_path / "correct")
    question["correct_letter"] = "Z"
    manifest = _rewrite_question(manifest, question_dir, question)
    with pytest.raises(catalog.SurpriseCatalogError, match="correct_letter"):
        catalog.build_surprise_catalog(manifest)

    escaped_manifest, escaped_dir, escaped_question = _make_manifest(
        tmp_path / "escape"
    )
    outside = escaped_manifest.data_root.parent / "outside"
    outside.mkdir(parents=True)
    escaped_question["choices"][0]["candidate_path"] = "../outside"
    escaped_manifest = _rewrite_question(
        escaped_manifest, escaped_dir, escaped_question
    )
    with pytest.raises(
        catalog.SurpriseCatalogError, match="normalized relative path|escapes"
    ):
        catalog.build_surprise_catalog(escaped_manifest)


def test_duplicate_json_keys_fail_closed_without_modifying_artifacts(
    tmp_path: Path,
) -> None:
    manifest, question_dir, question = _make_manifest(tmp_path)
    before_candidate = (
        manifest.data_root
        / question["choices"][0]["candidate_path"]
        / "candidate_spec.json"
    ).read_bytes()
    question_file = question_dir / "question.json"
    question_file.write_text(
        '{"question_id":"q_fixture","question_id":"q_other"}',
        encoding="utf-8",
    )

    with pytest.raises(catalog.SurpriseCatalogError, match="duplicate JSON"):
        catalog.build_surprise_catalog(manifest)

    assert (
        manifest.data_root
        / question["choices"][0]["candidate_path"]
        / "candidate_spec.json"
    ).read_bytes() == before_candidate
