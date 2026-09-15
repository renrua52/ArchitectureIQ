from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "llm_eval"))

from kb_pipeline import (  # noqa: E402
    ModelConfig,
    _upgrade_state,
    init_kb,
    list_questions,
    read_json,
    rebuild_credit_history,
    run_epoch,
)
from llm_client import LLMCompletion  # noqa: E402


class FakeClient:
    def __init__(self, responses: list[dict[str, Any] | str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, config: ModelConfig) -> LLMCompletion:
        self.prompts.append(prompt)
        payload = self.responses.pop(0)
        content = payload if isinstance(payload, str) else json.dumps(payload)
        raw = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        return LLMCompletion(content=content, raw=raw, finish_reason="stop")


def write_question(root: Path, question_id: str, correct: str = "A") -> None:
    qdir = root / question_id
    qdir.mkdir(parents=True)
    (qdir / "prompt.txt").write_text(
        "Choose the optimizer most likely to minimize test loss. Choices A and B.",
        encoding="utf-8",
    )
    (qdir / "question.json").write_text(
        json.dumps(
            {
                "question_id": question_id,
                "correct_letter": correct,
                "choices": [{"letter": "A"}, {"letter": "B"}],
            }
        ),
        encoding="utf-8",
    )


def configs() -> tuple[ModelConfig, ModelConfig]:
    return ModelConfig("solver"), ModelConfig("curator")


def test_list_questions_follows_direct_symlink_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    root = tmp_path / "questions"
    write_question(source, "q_1")
    root.mkdir()
    (root / "q_1").symlink_to(source / "q_1", target_is_directory=True)

    assert [item.question_id for item in list_questions(root)] == ["q_1"]


def test_correct_new_claim_is_added(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver = FakeClient(
        [
            {
                "answer": "A",
                "primary_claim": {"type": "new", "text": "Adam handles noisy gradients well."},
                "explanation": "The gradient scale varies.",
            }
        ]
    )
    curator = FakeClient(
        [{"existing_id": None, "canonical_text": "Adam can handle noisy gradients well."}]
    )
    solver_config, curator_config = configs()

    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=solver,
        curator_client=curator,
        solver_config=solver_config,
        curator_config=curator_config,
    )

    state = read_json(kb_dir / "claims.json")
    assert state["claims"] == [
        {
            "id": "K0001",
            "text": "Adam can handle noisy gradients well.",
            "support_count": 1,
            "failure_count": 0,
            "credit": 1,
            "created_epoch": 1,
            "last_supported_epoch": 1,
            "last_evaluated_epoch": 1,
        }
    ]
    assert "Ground-truth answer: A" in curator.prompts[0]


def test_frozen_snapshot_and_existing_claim_updates(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver_config, curator_config = configs()
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "Momentum smooths gradients."},
                    "explanation": "It reduces oscillation.",
                }
            ]
        ),
        curator_client=FakeClient(
            [{"existing_id": None, "canonical_text": "Momentum smooths noisy gradients."}]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    write_question(questions, "q_2", correct="A")
    solver = FakeClient(
        [
            {
                "answer": "B",
                "primary_claim": {"type": "kb", "id": "K0001"},
                "explanation": "This is the main consideration.",
            }
        ]
    )
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions / "q_2",
        solver_client=solver,
        curator_client=FakeClient([]),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    assert "K0001: Momentum smooths noisy gradients." in solver.prompts[0]
    assert read_json(kb_dir / "claims.json")["claims"] == [
        {
            "id": "K0001",
            "text": "Momentum smooths noisy gradients.",
            "support_count": 1,
            "failure_count": 1,
            "credit": 0,
            "created_epoch": 1,
            "last_supported_epoch": 1,
            "last_evaluated_epoch": 2,
        }
    ]
    epoch_one = read_json(kb_dir / "snapshots" / "kb_0001.json")
    assert epoch_one["claims"][0]["support_count"] == 1
    epoch_two = read_json(kb_dir / "snapshots" / "kb_0002.json")
    assert epoch_two["claims"][0]["credit"] == 0
    assert epoch_two["delta"]["penalized"] == [
        {
            "action": "penalized",
            "claim_id": "K0001",
            "claim_text": "Momentum smooths noisy gradients.",
            "credit_delta": -1,
            "credit_after": 0,
        }
    ]


def test_curator_can_deduplicate_new_text_to_existing_claim(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver_config, curator_config = configs()
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "Adam adapts each coordinate."},
                    "explanation": "Coordinate scales differ.",
                }
            ]
        ),
        curator_client=FakeClient(
            [{"existing_id": None, "canonical_text": "Adam adapts learning rates per parameter."}]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    write_question(questions, "q_2", correct="B")
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions / "q_2",
        solver_client=FakeClient(
            [
                {
                    "answer": "B",
                    "primary_claim": {"type": "new", "text": "Adam uses coordinate-wise rates."},
                    "explanation": "The parameters have different scales.",
                }
            ]
        ),
        curator_client=FakeClient([{"existing_id": "K0001", "canonical_text": None}]),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    claims = read_json(kb_dir / "claims.json")["claims"]
    assert len(claims) == 1
    assert claims[0]["support_count"] == 2
    assert claims[0]["failure_count"] == 0
    assert claims[0]["credit"] == 2
    assert len((kb_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_wrong_new_claim_never_enters_kb_or_curator(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="B")
    curator = FakeClient([])
    solver_config, curator_config = configs()

    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "A false proposition."},
                    "explanation": "This reasoning is wrong.",
                }
            ]
        ),
        curator_client=curator,
        solver_config=solver_config,
        curator_config=curator_config,
    )

    assert read_json(kb_dir / "claims.json")["claims"] == []
    snapshot = read_json(kb_dir / "snapshots" / "kb_0001.json")
    assert snapshot["delta"]["rejected"] == [
        {
            "action": "rejected",
            "claim_id": None,
            "claim_text": "A false proposition.",
            "credit_delta": 0,
            "credit_after": None,
        }
    ]
    assert curator.prompts == []


def test_positive_and_negative_credit_are_aggregated_within_epoch(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver_config, curator_config = configs()
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions / "q_1",
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "Momentum smooths gradients."},
                    "explanation": "It reduces oscillation.",
                }
            ]
        ),
        curator_client=FakeClient(
            [{"existing_id": None, "canonical_text": "Momentum smooths gradients."}]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    write_question(questions, "q_2", correct="A")
    write_question(questions, "q_3", correct="B")
    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "kb", "id": "K0001"},
                    "explanation": "The claim applies.",
                },
                {
                    "answer": "A",
                    "primary_claim": {"type": "kb", "id": "K0001"},
                    "explanation": "The claim applies.",
                },
            ]
        ),
        curator_client=FakeClient([]),
        solver_config=solver_config,
        curator_config=curator_config,
        skip_seen=True,
    )

    claim = read_json(kb_dir / "claims.json")["claims"][0]
    assert claim["support_count"] == 2
    assert claim["failure_count"] == 1
    assert claim["credit"] == 1
    assert claim["last_evaluated_epoch"] == 2


def test_upgrade_state_adds_credit_fields_to_v2_claims() -> None:
    state = {
        "schema_version": "architectureiq_kb_v2",
        "claims": [
            {
                "id": "K0001",
                "text": "A claim.",
                "support_count": 3,
                "created_epoch": 1,
                "last_supported_epoch": 2,
            }
        ],
    }

    upgraded = _upgrade_state(state)

    assert upgraded["schema_version"] == "architectureiq_kb_v3"
    assert upgraded["claims"][0]["failure_count"] == 0
    assert upgraded["claims"][0]["credit"] == 3
    assert upgraded["claims"][0]["last_evaluated_epoch"] == 2


def test_rebuild_credit_history_restores_hard_deleted_claim(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "credit"
    questions = tmp_path / "questions"
    init_kb(source)
    write_question(questions, "q_1", correct="A")
    solver_config, curator_config = configs()
    run_epoch(
        kb_dir=source,
        questions_root=questions / "q_1",
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "Momentum smooths gradients."},
                    "explanation": "It reduces oscillation.",
                }
            ]
        ),
        curator_client=FakeClient(
            [{"existing_id": None, "canonical_text": "Momentum smooths gradients."}]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )
    write_question(questions, "q_2", correct="B")
    run_epoch(
        kb_dir=source,
        questions_root=questions / "q_2",
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "kb", "id": "K0001"},
                    "explanation": "The claim applies.",
                }
            ]
        ),
        curator_client=FakeClient([]),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    old_state = read_json(source / "claims.json")
    old_state["schema_version"] = "architectureiq_kb_v2"
    old_state["claims"] = []
    (source / "claims.json").write_text(json.dumps(old_state), encoding="utf-8")
    old_snapshot = read_json(source / "snapshots" / "kb_0002.json")
    old_snapshot["schema_version"] = "architectureiq_kb_v2"
    old_snapshot["claims"] = []
    (source / "snapshots" / "kb_0002.json").write_text(
        json.dumps(old_snapshot), encoding="utf-8"
    )

    rebuilt = rebuild_credit_history(source, output)

    assert rebuilt["last_epoch"] == 2
    assert rebuilt["claims"] == [
        {
            "id": "K0001",
            "text": "Momentum smooths gradients.",
            "support_count": 1,
            "failure_count": 1,
            "credit": 0,
            "created_epoch": 1,
            "last_supported_epoch": 1,
            "last_evaluated_epoch": 2,
        }
    ]
    assert read_json(source / "snapshots" / "kb_0002.json")["claims"] == []
    assert read_json(output / "snapshots" / "kb_0002.json")["claims"][0]["credit"] == 0


def test_invalid_solver_format_is_saved_and_repaired_before_gt(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver = FakeClient(
        [
            "<answer>A</answer><explanation>Adam adapts coordinate scales.</explanation>",
            {
                "answer": "A",
                "primary_claim": {"type": "new", "text": "Adam adapts coordinate scales."},
                "explanation": "Adam adapts coordinate scales.",
            },
        ]
    )
    solver_config, curator_config = configs()

    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=solver,
        curator_client=FakeClient(
            [{"existing_id": None, "canonical_text": "Adam adapts coordinate scales."}]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    qdir = kb_dir / "runs" / "epoch_0001" / "questions" / "q_1"
    assert (qdir / "solver_response.json").is_file()
    assert (qdir / "solver_repair_response.json").is_file()
    assert read_json(qdir / "solver_locked.json")["format_repaired"] is True


def test_invalid_curator_format_is_saved_and_repaired(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    questions = tmp_path / "questions"
    init_kb(kb_dir)
    write_question(questions, "q_1", correct="A")
    solver_config, curator_config = configs()

    run_epoch(
        kb_dir=kb_dir,
        questions_root=questions,
        solver_client=FakeClient(
            [
                {
                    "answer": "A",
                    "primary_claim": {"type": "new", "text": "Adam adapts scales."},
                    "explanation": "Coordinate scales differ.",
                }
            ]
        ),
        curator_client=FakeClient(
            [
                "Canonical claim: Adam adapts coordinate scales.",
                {
                    "existing_id": None,
                    "canonical_text": "Adam adapts coordinate scales.",
                },
            ]
        ),
        solver_config=solver_config,
        curator_config=curator_config,
    )

    qdir = kb_dir / "runs" / "epoch_0001" / "questions" / "q_1"
    assert (qdir / "curator_raw_response.json").is_file()
    assert (qdir / "curator_repair_response.json").is_file()
    assert read_json(qdir / "curator_response.json")["format_repaired"] is True
