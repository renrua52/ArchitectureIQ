"""Tests for candidate-disjoint evaluation collection tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILD = _load_tool("build_leakage_safe_collection")
_VALIDATE = _load_tool("validate_leakage_safe_collection")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_question_run(
    root: Path,
    questions: list[tuple[str, str, list[str]]],
) -> Path:
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run.json",
        {"question_ids": [question_id for question_id, _, _ in questions]},
    )
    for question_id, dataset_id, candidate_ids in questions:
        question_dir = run_dir / question_id
        question_dir.mkdir()
        choices = [
            {"letter": chr(ord("A") + index), "candidate_id": candidate_id}
            for index, candidate_id in enumerate(candidate_ids)
        ]
        _write_json(
            question_dir / "question.json",
            {
                "question_id": question_id,
                "family": "demo",
                "dataset_id": dataset_id,
                "choices": choices,
                "correct_letter": "A",
                "prompt": {"rendered_path": "prompt.txt"},
            },
        )
        (question_dir / "prompt.txt").write_text(
            f"Choose the best setting for {question_id}.",
            encoding="utf-8",
        )
    return run_dir


def _build(
    run_dir: Path,
    output_dir: Path,
    *,
    split_mode: str,
    support_count: int,
    holdout_count: int,
    seed: int = 7,
) -> dict[str, object]:
    return _BUILD.build_collection(
        [run_dir],
        output_dir,
        support_count=support_count,
        holdout_count=holdout_count,
        split_mode=split_mode,
        seed=seed,
    )


def test_id_collection_is_reproducible_and_holdout_datasets_are_seen(
    tmp_path: Path,
) -> None:
    run_dir = _write_question_run(
        tmp_path / "source",
        [
            ("q_d1_1", "d1", ["c01", "c02"]),
            ("q_d1_2", "d1", ["c03", "c04"]),
            ("q_d1_3", "d1", ["c05", "c06"]),
            ("q_d2_1", "d2", ["c07", "c08"]),
            ("q_d2_2", "d2", ["c09", "c10"]),
            ("q_d2_3", "d2", ["c11", "c12"]),
        ],
    )

    first = _build(
        run_dir,
        tmp_path / "first",
        split_mode="id",
        support_count=2,
        holdout_count=2,
        seed=19,
    )
    second = _build(
        run_dir,
        tmp_path / "second",
        split_mode="id",
        support_count=2,
        holdout_count=2,
        seed=19,
    )

    assert first["collection_id"] == second["collection_id"]
    assert first["support_question_ids"] == second["support_question_ids"]
    assert first["holdout_question_ids"] == second["holdout_question_ids"]
    result = _VALIDATE.validate_collection(tmp_path / "first")
    assert set(result["holdout_dataset_ids"]) <= set(result["support_dataset_ids"])


def test_ood_collection_uses_disjoint_support_and_holdout_datasets(
    tmp_path: Path,
) -> None:
    run_dir = _write_question_run(
        tmp_path / "source",
        [
            ("q_d1_1", "d1", ["c01", "c02"]),
            ("q_d1_2", "d1", ["c03", "c04"]),
            ("q_d2_1", "d2", ["c05", "c06"]),
            ("q_d2_2", "d2", ["c07", "c08"]),
        ],
    )
    _build(
        run_dir,
        tmp_path / "collection",
        split_mode="ood",
        support_count=1,
        holdout_count=1,
    )

    result = _VALIDATE.validate_collection(tmp_path / "collection")

    assert set(result["support_dataset_ids"]).isdisjoint(
        result["holdout_dataset_ids"]
    )


def test_overlapping_questions_are_excluded_and_shortage_fails(
    tmp_path: Path,
) -> None:
    run_dir = _write_question_run(
        tmp_path / "source",
        [
            ("q1", "d1", ["c1", "c2"]),
            ("q2", "d1", ["c2", "c3"]),
        ],
    )
    records = _BUILD.load_question_runs([run_dir])

    selected = _BUILD.select_candidate_disjoint(records)

    assert len(selected) == 1
    with pytest.raises(ValueError, match="Cannot build an ID split"):
        _build(
            run_dir,
            tmp_path / "collection",
            split_mode="id",
            support_count=1,
            holdout_count=1,
        )


def test_validator_rejects_candidate_reused_across_questions(tmp_path: Path) -> None:
    run_dir = _write_question_run(
        tmp_path / "source",
        [
            ("q1", "d1", ["c1", "c2"]),
            ("q2", "d1", ["c3", "c4"]),
        ],
    )
    collection_dir = tmp_path / "collection"
    _build(
        run_dir,
        collection_dir,
        split_mode="id",
        support_count=1,
        holdout_count=1,
    )
    manifest_path = collection_dir / "collection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][1]["candidate_ids"][0] = manifest["records"][0][
        "candidate_ids"
    ][0]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="candidate_id .* is reused"):
        _VALIDATE.validate_collection(collection_dir)


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    [
        (
            "support.json",
            lambda payload: payload[0].update({"correct_letter": "A"}),
            "exposes private fields",
        ),
        (
            "holdout.json",
            lambda payload: payload[0].update(
                {"prompt": "Inspect results/summary.json before answering."}
            ),
            "contains private marker",
        ),
    ],
)
def test_validator_rejects_public_key_or_prompt_marker_leakage(
    tmp_path: Path,
    filename: str,
    mutate: object,
    message: str,
) -> None:
    run_dir = _write_question_run(
        tmp_path / "source",
        [
            ("q1", "d1", ["c1", "c2"]),
            ("q2", "d1", ["c3", "c4"]),
        ],
    )
    collection_dir = tmp_path / "collection"
    _build(
        run_dir,
        collection_dir,
        split_mode="id",
        support_count=1,
        holdout_count=1,
    )
    public_path = collection_dir / filename
    payload = json.loads(public_path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(public_path, payload)

    with pytest.raises(ValueError, match=message):
        _VALIDATE.validate_collection(collection_dir)
