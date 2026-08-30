from __future__ import annotations

from pathlib import Path

from architecture_iq.prompts.formatters import (
    format_mlp_nl,
    format_training_schedule,
)
from architecture_iq.prompts.renderer import render_prompt


def test_format_mlp_nl_states_one_shared_activation() -> None:
    # The activation is a single per-network value, so the prompt must not
    # print a per-layer list like [gelu, gelu].
    text = format_mlp_nl(
        {
            "depth": 2,
            "width": 64,
            "residual": False,
            "layer_norm": [True, False],
            "activation": "gelu",
        }
    )
    assert "- Activation: gelu (one activation, shared by every layer)" in text
    assert "Activations: [" not in text
    assert "- Layer norm per layer: [True, False]" in text


def test_format_training_schedule_includes_product() -> None:
    text = format_training_schedule(
        {"training_steps": 8, "batch_size": 64, "total_samples_seen": 512}
    )
    assert "training_steps: 8" in text
    assert "batch_size: 64" in text
    assert "training_steps × batch_size" in text


def test_render_prompt_includes_reproduction_protocol() -> None:
    q_path = Path("data/questions/q_547c83")
    if not q_path.exists():
        return
    text = render_prompt(q_path)
    assert "Data splits and training protocol" in text
    assert "Target expression (canonical)" in text
    assert "with replacement" in text
    assert "Ground-truth ranking" in text or "Evaluation metric" in text
    assert "Activations: [" in text
    assert "def _activation" in text
    assert "## Sample budget (same for all choices)" in text


def test_render_prompt_matches_prompt_txt() -> None:
    q_path = Path("data/questions/q_547c83")
    if not q_path.exists():
        return
    from architecture_iq.util import read_json

    q = read_json(q_path / "question.json")
    if "evaluation" not in q:
        return
    rendered = render_prompt(q_path)
    on_disk = (q_path / "prompt.txt").read_text(encoding="utf-8")
    assert rendered == on_disk


def _write_minimal_question(root: Path) -> tuple[Path, Path]:
    """A two-choice univariate question on disk, enough for render_prompt."""
    import json

    dataset_path = root / "datasets" / "univariate_regression" / "sym_test"
    dataset_path.mkdir(parents=True)
    (dataset_path / "synthesize.py").write_text(
        "import torch\n\n\n"
        "def target(x):\n    return torch.sin(2 * torch.pi * x)\n\n\n"
        "def synthesize():\n    x = torch.rand(8, 1)\n    return x, target(x)\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset_spec.json").write_text(
        json.dumps(
            {
                "family": "univariate_regression",
                "dataset_id": "sym_test",
                "selection_metric": "test_mse",
                "params": {
                    "expression": "sin(2*pi*x)",
                    "domain": [0.0, 1.0],
                    "train_size": 8,
                    "test_size": 8,
                    "point_sampling": {"seed": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    set_path = dataset_path / "candidates" / "set_test"
    choices = []
    for index, letter in enumerate(("A", "B")):
        cand_path = set_path / f"c_{letter.lower()}"
        cand_path.mkdir(parents=True)
        (cand_path / "candidate_spec.json").write_text(
            json.dumps(
                {
                    "candidate_id": cand_path.name,
                    "family": "univariate_regression",
                    "budget": {
                        "training_steps": 64,
                        "batch_size": 16,
                        "total_samples_seen": 1024,
                    },
                    "model": {
                        "type": "mlp",
                        "depth": 1,
                        "width": 16 + index,
                        "residual": False,
                        "layer_norm": [False],
                        "activation": "relu",
                        "input_dim": 1,
                        "output_dim": 1,
                    },
                    "optimizer": {"type": "Adam", "lr": 1.0e-3},
                    "loss": {"loss_id": "mse"},
                    "execution": {"device": "cpu"},
                }
            ),
            encoding="utf-8",
        )
        choices.append(
            {
                "letter": letter,
                "candidate_path": str(cand_path.relative_to(root)),
            }
        )

    question_path = dataset_path / "questions" / "run_test" / "q_test"
    question_path.mkdir(parents=True)
    (question_path / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q_test",
                "family": "univariate_regression",
                "dataset_id": "sym_test",
                "type": "architecture_only",
                "budget": {
                    "training_steps": 64,
                    "batch_size": 16,
                    "total_samples_seen": 1024,
                },
                "correct_letter": "A",
                "choices": choices,
                "evaluation": {
                    "selection_metric": "test_mse",
                    "n_seeds": 10,
                    "base_seed": 0,
                    "device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    return question_path, dataset_path


def test_render_prompt_asks_for_answer_tags(tmp_path: Path) -> None:
    """The grader reads <answer>X</answer>, so the prompt must ask for that form.

    Before v1.4 the prompt said "Reply with a single letter" while
    tools/llm_eval/response_parser.py only accepted the tag, so a model that
    followed the prompt literally scored zero.
    """
    question_path, dataset_path = _write_minimal_question(tmp_path)
    text = render_prompt(
        question_path, dataset_path=dataset_path, artifact_root=tmp_path
    )

    assert "<answer></answer>" in text
    assert "<answer>A</answer>" in text
    # The old bare-letter instruction must be gone, not merely supplemented.
    assert "Reply with a single letter" not in text
