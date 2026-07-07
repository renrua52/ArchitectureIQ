"""Tests for the standalone LLM evaluation runner."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL = ROOT / "tools" / "llm_eval"
sys.path.insert(0, str(LLM_EVAL))

from completion import fetch_model_response  # noqa: E402
from llm_client import LLMCompletion, ModelConfig, message_text  # noqa: E402
from prompt_wrapper import format_eval_prompt  # noqa: E402
from question_loader import QuestionItem, load_question_item, prompt_hash  # noqa: E402
from response_parser import parse_choice_letter, split_chain_of_thought  # noqa: E402
from runner import QuestionResult, evaluate_question, run_evaluation, summarize_results  # noqa: E402


@dataclass
class FakeCompletion:
    content: str
    finish_reason: str = "stop"
    usage: dict[str, str] | None = None

    def to_llm_completion(self) -> LLMCompletion:
        return LLMCompletion(
            content=self.content,
            raw={
                "choices": [
                    {
                        "message": {"content": self.content},
                        "finish_reason": self.finish_reason,
                    }
                ]
            },
            finish_reason=self.finish_reason,
            usage=self.usage,
        )


class FakeClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, prompt: str, config: ModelConfig) -> LLMCompletion:
        return self.complete_conversation([{"role": "user", "content": prompt}], config)

    def complete_conversation(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
    ) -> LLMCompletion:
        self.calls.append(messages)
        key = prompt_hash(messages[0]["content"])
        if key not in self.responses:
            raise KeyError(f"No fake response for prompt hash {key}")
        return FakeCompletion(content=self.responses[key]).to_llm_completion()


@dataclass
class SequentialFakeClient:
    completions: list[FakeCompletion] = field(default_factory=list)
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def complete_conversation(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
    ) -> LLMCompletion:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.completions) - 1)
        return self.completions[idx].to_llm_completion()


def test_format_eval_prompt_appends_answer_tags() -> None:
    wrapped = format_eval_prompt("Pick one.", frozenset({"A", "B"}))
    assert "<answer>A</answer>" in wrapped
    assert "Valid choices: A, B" in wrapped


def test_parse_choice_letter_from_answer_tag() -> None:
    valid = frozenset({"A", "B", "C"})
    assert parse_choice_letter("Step by step...\n<answer>B</answer>", valid) == "B"
    assert parse_choice_letter("Draft <answer>A</answer> then <answer>C</answer>", valid) == "C"
    assert parse_choice_letter("Choice A and Choice B look plausible.", valid) is None
    assert parse_choice_letter("Maybe both?", valid) is None


def test_split_chain_of_thought_removes_answer_tag() -> None:
    text = "Because loss A overfits.\n<answer>B</answer>"
    assert split_chain_of_thought(text, "B") == "Because loss A overfits."


def test_message_text_merges_reasoning_and_content() -> None:
    merged = message_text(
        {
            "reasoning_content": "Internal reasoning.",
            "content": "Visible answer.\n<answer>A</answer>",
        }
    )
    assert "Internal reasoning." in merged
    assert "<answer>A</answer>" in merged


def test_fetch_model_response_continues_on_length() -> None:
    client = SequentialFakeClient(
        completions=[
            FakeCompletion("Partial reasoning without answer.", finish_reason="length"),
            FakeCompletion("More reasoning.\n<answer>A</answer>", finish_reason="stop"),
        ]
    )
    exchange = fetch_model_response(
        client,
        "prompt",
        ModelConfig(name="fake"),
        frozenset({"A", "B"}),
    )
    assert exchange.continuation_count == 1
    assert exchange.truncated is False
    assert "<answer>A</answer>" in exchange.model_response
    assert len(client.calls) == 2


def test_run_evaluation_concurrent(tmp_path: Path) -> None:
    questions_root = tmp_path / "questions"
    for qid, letter in (("q_a", "A"), ("q_b", "B"), ("q_c", "A")):
        qdir = questions_root / qid
        qdir.mkdir(parents=True)
        prompt = f"Pick for {qid}."
        (qdir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (qdir / "question.json").write_text(
            json.dumps(
                {
                    "question_id": qid,
                    "type": "mixed",
                    "family": "univariate_regression",
                    "correct_letter": letter,
                    "choices": [{"letter": "A"}, {"letter": "B"}],
                    "prompt": {"rendered_path": "prompt.txt"},
                }
            ),
            encoding="utf-8",
        )

    responses = {}
    for qid in ("q_a", "q_b", "q_c"):
        item = load_question_item(questions_root / qid)
        eval_prompt = format_eval_prompt(item.prompt_text, item.valid_letters)
        responses[prompt_hash(eval_prompt)] = f"<answer>{item.correct_letter}</answer>"

    client = FakeClient(responses)
    manifest = run_evaluation(
        questions_root=questions_root,
        run_dir=tmp_path / "run",
        model_config=ModelConfig(name="fake-model"),
        client=client,
        workers=3,
    )
    assert manifest["summary"]["total_questions"] == 3
    assert manifest["summary"]["correct"] == 3
    assert manifest["workers"] == 3
    assert len(list((tmp_path / "run" / "results").glob("*.json"))) == 3


def test_run_evaluation_writes_run_artifacts(tmp_path: Path) -> None:
    questions_root = tmp_path / "questions"
    qdir = questions_root / "q_test01"
    qdir.mkdir(parents=True)
    prompt = "Pick one.\nReply with a single letter (A, B)."
    (qdir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (qdir / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q_test01",
                "type": "mixed",
                "family": "univariate_regression",
                "correct_letter": "B",
                "choices": [{"letter": "A"}, {"letter": "B"}],
                "prompt": {"rendered_path": "prompt.txt"},
            }
        ),
        encoding="utf-8",
    )

    item = load_question_item(qdir)
    eval_prompt = format_eval_prompt(item.prompt_text, item.valid_letters)
    raw_response = "Reasoning here about Choice A and Choice B.\n<answer>B</answer>"
    client = FakeClient({prompt_hash(eval_prompt): raw_response})
    run_dir = tmp_path / "run"
    config = ModelConfig(name="fake-model", temperature=0.0)

    manifest = run_evaluation(
        questions_root=questions_root,
        run_dir=run_dir,
        model_config=config,
        client=client,
        limit=None,
        workers=1,
    )

    assert manifest["summary"]["correct"] == 1
    assert manifest["summary"]["accuracy"] == 1.0
    assert (run_dir / "run.json").is_file()
    assert "<answer>" in client.calls[0][0]["content"]

    result_path = run_dir / "results" / f"{item.question_id}.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ground_truth_letter"] == "B"
    assert payload["parsed_letter"] == "B"
    assert payload["model_response"] == raw_response
    assert payload["eval_prompt"] == eval_prompt
    assert payload["chain_of_thought"] == "Reasoning here about Choice A and Choice B."


def test_evaluate_question_preserves_full_model_response() -> None:
    from completion import ModelExchange

    item = QuestionItem(
        question_dir=Path("."),
        question_id="q_x",
        question={"type": "mixed", "family": "f", "correct_letter": "A"},
        prompt_text="prompt",
        prompt_hash="hash",
        valid_letters=frozenset({"A", "B"}),
    )
    raw = "Long reasoning that mentions Choice B.\n<answer>A</answer>"
    exchange = ModelExchange(
        model_response=raw,
        finish_reason="stop",
        usage=None,
        continuation_count=0,
        truncated=False,
        message_parts={"content": raw},
    )
    result = evaluate_question(item, exchange, "full prompt text")
    assert result.model_response == raw
    assert result.parsed_letter == "A"


def test_summarize_results_counts_unparsed() -> None:
    rows = [
        QuestionResult(
            question_id="q1",
            prompt_hash="abc",
            ground_truth_letter="A",
            parsed_letter="A",
            correct=True,
            model_response="<answer>A</answer>",
            chain_of_thought=None,
            question_type="mixed",
            family="f",
            eval_prompt="",
        ),
        QuestionResult(
            question_id="q2",
            prompt_hash="def",
            ground_truth_letter="B",
            parsed_letter=None,
            correct=False,
            model_response="unsure",
            chain_of_thought="unsure",
            question_type="mixed",
            family="f",
            eval_prompt="",
            truncated=True,
            finish_reason="length",
        ),
    ]
    summary = summarize_results(rows)
    assert summary["parsed"] == 1
    assert summary["unparsed"] == 1
    assert summary["accuracy"] == 1.0
