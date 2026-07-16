from __future__ import annotations

from pathlib import Path

from architecture_iq.ground_truth.runner import _sync_candidate_files
from architecture_iq.paths import PROMPTS_DIR, DATA_DIR, dataset_dir
from architecture_iq.profile import load_profile
from architecture_iq.prompts.code_excerpt import (
    excerpt_loss_py,
    excerpt_model_py,
    excerpt_optimizer_py,
    excerpt_synthesize_py,
)
from architecture_iq.prompts.formatters import (
    SINGLE_AXIS_TYPES,
    format_dataset_protocol,
    format_loss_nl,
    format_synthetic_tabular_classification_rule,
    format_model_nl,
    format_optimizer_nl,
    format_ranking_protocol,
    format_training_schedule,
)
from architecture_iq.util import read_json


def _read_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _question_total_samples_seen(budget: dict | int) -> int | None:
    if isinstance(budget, int):
        return budget
    if budget.get("mixed"):
        return None
    return int(budget["total_samples_seen"])


def _evaluation_meta(q: dict) -> dict:
    if "evaluation" in q:
        return q["evaluation"]
    profile = load_profile(q.get("profile", "v1"))
    return {
        "selection_metric": "test_mse",
        "n_seeds": profile.n_seeds,
        "base_seed": profile.base_seed,
        "device": "cpu",
    }


def render_prompt(question_path: Path) -> str:
    q = read_json(question_path / "question.json")
    dataset_path = dataset_dir(q["family"], q["dataset_id"])
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    params = dataset_spec["params"]
    eval_meta = _evaluation_meta(q)
    selection_metric = eval_meta.get("selection_metric", dataset_spec["selection_metric"])

    header = _read_template("header.md") or (
        "You are taking the ArchitectureIQ benchmark. "
        "Read each training setup and pick the choice that achieves the best "
        f"**{selection_metric}** after the stated training budget. Reply with a single letter."
    )

    dataset_nl = _read_template(f"dataset/{q['family']}.md")
    if not dataset_nl:
        dataset_nl = (
            "Univariate regression on [0, 1]. Input and target are 1-D scalars. "
            f"Train size: {params['train_size']}, test size: {params['test_size']}."
        )

    is_classification = q["family"] == "synthetic_tabular_classification"
    total_samples_seen = _question_total_samples_seen(q["budget"])
    single_axis = q["type"] in SINGLE_AXIS_TYPES and not (
        isinstance(q["budget"], dict) and q["budget"].get("mixed")
    )

    budget_heading = (
        "## Sample budget"
        if total_samples_seen is None
        else "## Sample budget (same for all choices)"
    )
    parts = [
        header.strip(),
        "",
        "## Dataset",
        dataset_nl.strip(),
        "",
    ]
    if is_classification:
        parts.extend(
            [
                "### Data-generating and classification rule",
                format_synthetic_tabular_classification_rule(params),
                "",
            ]
        )
    else:
        synth_source = (dataset_path / "synthesize.py").read_text(encoding="utf-8")
        synth_code = excerpt_synthesize_py(synth_source)
        parts.extend(["### Synthesis (PyTorch)", "```python", synth_code, "```", ""])
    parts.extend(
        [
            "### Data splits and training protocol",
            format_dataset_protocol(params, family=q["family"], device=str(eval_meta.get("device", "cpu"))),
            "",
            budget_heading,
        ]
    )
    if single_axis and q["choices"]:
        first_cand = read_json(
            DATA_DIR / q["choices"][0]["candidate_path"] / "candidate_spec.json"
        )
        parts.append(format_training_schedule(first_cand["budget"]))
    elif total_samples_seen is not None:
        parts.extend(
            [
                f"- total_samples_seen: {total_samples_seen}",
                "",
                "Each choice specifies its own `training_steps` and `batch_size` below; "
                "they must satisfy `training_steps × batch_size = total_samples_seen`.",
            ]
        )
    else:
        parts.extend(
            [
                "- budgets differ across choices",
                "",
                "Each choice specifies its own training budget below.",
            ]
        )
    parts.extend(
        [
            "",
            "## Evaluation metric",
            format_ranking_protocol(
                n_seeds=int(eval_meta["n_seeds"]),
                base_seed=int(eval_meta["base_seed"]),
                selection_metric=selection_metric,
                device=str(eval_meta.get("device", "cpu")),
            ),
            "",
            "## Choices",
        ]
    )

    for choice in q["choices"]:
        cand_path = DATA_DIR / choice["candidate_path"]
        cand_spec = read_json(cand_path / "candidate_spec.json")
        _sync_candidate_files(cand_path, cand_spec)
        model_code = excerpt_model_py((cand_path / "model.py").read_text(encoding="utf-8"))
        loss_code = excerpt_loss_py((cand_path / "loss.py").read_text(encoding="utf-8"))
        opt_code = excerpt_optimizer_py((cand_path / "optimizer.py").read_text(encoding="utf-8"))

        parts.extend(
            [
                "",
                f"### Choice {choice['letter']}",
                "",
            ]
        )
        if not single_axis:
            parts.extend(
                [
                    "**Training schedule**",
                    format_training_schedule(cand_spec["budget"]),
                    "",
                ]
            )
        parts.extend(
            [
                "**Model (natural language)**",
                format_model_nl(cand_spec["model"]),
                "",
                "**Model code**",
                "```python",
                model_code,
                "```",
                "",
                "**Optimizer**",
                format_optimizer_nl(cand_spec["optimizer"]),
                "",
                "```python",
                opt_code,
                "```",
                "",
                "**Loss**",
                format_loss_nl(cand_spec["loss"]),
                "",
                "```python",
                loss_code,
                "```",
            ]
        )

    parts.extend(
        [
            "",
            "## Your answer",
            f"Reply with a single letter ({', '.join(c['letter'] for c in q['choices'])}).",
        ]
    )
    return "\n".join(parts)


def write_prompt(question_path: Path) -> Path:
    text = render_prompt(question_path)
    out = question_path / "prompt.txt"
    out.write_text(text, encoding="utf-8")
    return out
