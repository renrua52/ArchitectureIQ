from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.meta_model_study.compare_agent_baseline import (
    ComparisonInputError,
    compare_agent_baseline,
    exact_mcnemar_two_sided_p_value,
    paired_cluster_bootstrap_ci,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _choices(question_id: str) -> list[dict[str, str]]:
    return [
        {"letter": "A", "candidate_id": f"c_{question_id}_a"},
        {"letter": "B", "candidate_id": f"c_{question_id}_b"},
    ]


def _public_questions() -> list[dict[str, object]]:
    return [
        {
            "question_id": question_id,
            "question_run_id": cluster_id,
            "family": "test_family",
            "dataset_id": f"dataset_{cluster_id}",
            "choices": _choices(question_id),
        }
        for question_id, cluster_id in (
            ("q1", "cluster_1"),
            ("q2", "cluster_1"),
            ("q3", "cluster_2"),
            ("q4", "cluster_2"),
        )
    ]


def _answer_key() -> list[dict[str, object]]:
    return [
        {
            "question_id": question["question_id"],
            "question_run_id": question["question_run_id"],
            "family": question["family"],
            "correct_letter": "A",
            "choices": question["choices"],
        }
        for question in _public_questions()
    ]


def _prediction_payload(
    letters: list[str],
    *,
    collection: str = "predictions",
) -> dict[str, object]:
    rows = []
    for question, letter in zip(_public_questions(), letters, strict=True):
        question_id = str(question["question_id"])
        candidate_id = dict(
            (choice["letter"], choice["candidate_id"]) for choice in question["choices"]
        )[letter]
        rows.append(
            {
                "question_id": question_id,
                "family": question["family"],
                "predicted_letter": letter,
                "predicted_candidate_id": candidate_id,
                "choice_predictions": question["choices"],
            }
        )
    return {collection: rows}


def _comparison_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        _write(tmp_path / "questions.json", _public_questions()),
        _write(tmp_path / "answer_key.json", _answer_key()),
        _write(
            tmp_path / "baseline.json",
            _prediction_payload(["A", "B", "A", "B"]),
        ),
        _write(
            tmp_path / "agent.json",
            _prediction_payload(["A", "A", "B", "B"], collection="records"),
        ),
    )


def test_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    questions, answer_key, baseline, agent = _comparison_paths(tmp_path)
    agent_payload = json.loads(agent.read_text(encoding="utf-8"))
    agent_payload["records"][0]["predicted_candidate_id"] = "c_wrong"
    _write(agent, agent_payload)

    with pytest.raises(ComparisonInputError, match="choice identity mismatch"):
        compare_agent_baseline(
            questions,
            answer_key,
            baseline,
            agent,
            reps=20,
        )


def test_paired_counts_and_top1_scores(tmp_path: Path) -> None:
    questions, answer_key, baseline, agent = _comparison_paths(tmp_path)

    result = compare_agent_baseline(
        questions,
        answer_key,
        baseline,
        agent,
        seed=11,
        reps=100,
    )

    assert result["agent"] == {"correct": 2, "total": 4, "accuracy": 0.5}
    assert result["baseline"] == {"correct": 2, "total": 4, "accuracy": 0.5}
    assert result["paired_2x2"] == {
        "both_correct": 1,
        "agent_correct_baseline_wrong": 1,
        "agent_wrong_baseline_correct": 1,
        "both_wrong": 1,
    }
    assert result["difference"]["correct"] == 0
    assert result["mcnemar_exact_two_sided"]["p_value"] == 1.0
    assert "55/60 versus 54/60" in result["interpretation"]["point_estimate_note"]


def test_missing_identity_requires_manifest_and_manifest_is_checked(
    tmp_path: Path,
) -> None:
    questions, answer_key, baseline, _ = _comparison_paths(tmp_path)
    agent = _write(
        tmp_path / "legacy_agent.json",
        {"answers": {f"legacy_{index}": "A" for index in range(1, 5)}},
    )

    with pytest.raises(ComparisonInputError, match="identity-manifest"):
        compare_agent_baseline(
            questions,
            answer_key,
            baseline,
            agent,
            reps=20,
        )

    manifest = _write(
        tmp_path / "agent_identity.json",
        {
            "questions": [
                {
                    "source_question_id": f"legacy_{index}",
                    "question_id": question["question_id"],
                    "choices": question["choices"],
                }
                for index, question in enumerate(_public_questions(), start=1)
            ]
        },
    )
    result = compare_agent_baseline(
        questions,
        answer_key,
        baseline,
        agent,
        agent_identity_manifest_path=manifest,
        reps=20,
    )

    assert result["agent"]["correct"] == 4
    assert result["input_adapters"]["agent"] == "answers_map"
    assert result["input_adapters"]["agent_identity_manifest_used"] is True


def test_exact_two_sided_mcnemar_known_values() -> None:
    assert exact_mcnemar_two_sided_p_value(1, 3) == pytest.approx(0.625)
    assert exact_mcnemar_two_sided_p_value(0, 4) == pytest.approx(0.125)
    assert exact_mcnemar_two_sided_p_value(0, 0) == 1.0


def test_cluster_paired_bootstrap_is_deterministic() -> None:
    rows = [
        {"cluster_id": "positive", "agent_correct": True, "baseline_correct": False},
        {"cluster_id": "positive", "agent_correct": True, "baseline_correct": False},
        {"cluster_id": "negative", "agent_correct": False, "baseline_correct": True},
        {"cluster_id": "negative", "agent_correct": False, "baseline_correct": True},
        {"cluster_id": "neutral", "agent_correct": True, "baseline_correct": True},
        {"cluster_id": "neutral", "agent_correct": False, "baseline_correct": False},
    ]

    first = paired_cluster_bootstrap_ci(rows, seed=314, reps=500)
    second = paired_cluster_bootstrap_ci(rows, seed=314, reps=500)

    assert first == second
    assert first["num_clusters"] == 3
    assert first["point_estimate"] == 0.0
    assert first["low"] <= 0.0 <= first["high"]
