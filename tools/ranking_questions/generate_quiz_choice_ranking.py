#!/usr/bin/env python3
"""Generate 3-item ranking tasks from a sanitized 3-choice quiz artifact.

Each output question keeps the current quiz question's 3 choices as targets, and
adds calibration/reference settings sampled from other questions. The targets are
scored by inversion count against the true order of their mean held-out metric.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))

from architecture_iq.prompts.formatters import (  # noqa: E402
    format_dataset_protocol,
    format_loss_nl,
    format_model_nl,
    format_optimizer_nl,
    format_training_schedule,
)

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from common import candidate_metric, read_json, write_json  # noqa: E402
from generate import _plot_curve  # noqa: E402


@dataclass(frozen=True)
class ChoiceArtifact:
    source_index: int
    question_id: str
    family: str
    dataset_id: str
    original_letter: str
    candidate_id: str
    candidate_path: Path
    spec: dict[str, Any]
    summary: dict[str, Any]
    metric: str
    mean_metric: float
    std_metric: float | None

    @property
    def path(self) -> Path:
        return self.candidate_path


def _data_path(relative: str) -> Path:
    path = ROOT / "data" / relative
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


def _setting_markdown(spec: dict[str, Any]) -> str:
    lines = ["Training schedule"]
    lines.extend(format_training_schedule(spec["budget"]).splitlines())
    lines.append("")
    lines.append("Model")
    lines.extend(format_model_nl(spec["model"]).splitlines())
    lines.append("")
    lines.append("Optimizer")
    lines.extend(format_optimizer_nl(spec["optimizer"]).splitlines())
    lines.append("")
    lines.append("Loss")
    lines.extend(format_loss_nl(spec["loss"]).splitlines())
    return "\n".join(lines)


def _format_dataset_params(q: dict[str, Any]) -> str:
    return format_dataset_protocol(q.get("dataset_params", {}), family=q["family"])


def _load_source(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = read_json(source_dir / "questions_sanitized.json")
    answer_rows = read_json(source_dir / "answer_key.json")
    if not isinstance(questions, list) or not isinstance(answer_rows, list):
        raise ValueError("Expected list schemas for questions_sanitized.json and answer_key.json")
    answer_by_qid = {row["question_id"]: row for row in answer_rows}
    if len(questions) != len(answer_by_qid):
        raise ValueError("Question count and answer-key count differ")
    return questions, answer_by_qid


def _load_choices(
    questions: list[dict[str, Any]],
    answer_by_qid: dict[str, dict[str, Any]],
) -> dict[str, list[ChoiceArtifact]]:
    choices_by_qid: dict[str, list[ChoiceArtifact]] = {}
    for q_index, q in enumerate(questions, start=1):
        qid = q["question_id"]
        key_row = answer_by_qid[qid]
        path_by_letter = {choice["letter"]: choice["candidate_path"] for choice in key_row["choices"]}
        records: list[ChoiceArtifact] = []
        for choice in q["choices"]:
            candidate_path = _data_path(path_by_letter[choice["letter"]])
            summary = read_json(candidate_path / "results" / "summary.json")
            metric, mean_metric, std_metric = candidate_metric(summary)
            if metric != q["selection_metric"]:
                raise ValueError(f"Metric mismatch for {qid}/{choice['letter']}: {metric} != {q['selection_metric']}")
            records.append(
                ChoiceArtifact(
                    source_index=q_index,
                    question_id=qid,
                    family=q["family"],
                    dataset_id=q["dataset_id"],
                    original_letter=choice["letter"],
                    candidate_id=choice["candidate_id"],
                    candidate_path=candidate_path,
                    spec=choice,
                    summary=summary,
                    metric=metric,
                    mean_metric=mean_metric,
                    std_metric=std_metric,
                )
            )
        choices_by_qid[qid] = records
    return choices_by_qid


def _make_shards(num_questions: int, shard_size: int) -> list[list[int]]:
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    return [
        list(range(start, min(start + shard_size, num_questions + 1)))
        for start in range(1, num_questions + 1, shard_size)
    ]


def _sample_references(
    *,
    rng: random.Random,
    question: dict[str, Any],
    current_qid: str,
    excluded_candidate_ids: set[str],
    excluded_source_indices: set[int],
    all_choices: list[ChoiceArtifact],
    reference_size: int,
) -> list[ChoiceArtifact]:
    pool: list[ChoiceArtifact] = []
    seen_candidate_ids: set[str] = set()
    for choice in all_choices:
        if choice.family != question["family"]:
            continue
        if choice.dataset_id != question["dataset_id"]:
            continue
        if choice.question_id == current_qid:
            continue
        if choice.source_index in excluded_source_indices:
            continue
        if choice.candidate_id in excluded_candidate_ids:
            continue
        if choice.candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(choice.candidate_id)
        pool.append(choice)
    if len(pool) < reference_size:
        raise ValueError(
            f"Only {len(pool)} reference choices available for {current_qid}; "
            f"need {reference_size}"
        )
    pool.sort(key=lambda c: (c.source_index, c.original_letter, c.candidate_id))
    return rng.sample(pool, reference_size)


def _render_prompt(
    *,
    blind_id: str,
    question: dict[str, Any],
    reference_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# ArchitectureIQ 3-Item Ranking Question",
        "",
        "Rank the target settings from best to worst expected final held-out metric. "
        "Do not run experiments, inspect hidden result files, or inspect files outside "
        "this blind shard.",
        "",
        f"Question: `{blind_id}`",
        f"Family: `{question['family']}`",
        f"Dataset: `{question['dataset_id']}`",
        f"Metric: `{question['selection_metric']}`; lower is better.",
        "",
        "## Dataset Protocol",
        "",
        _format_dataset_params(question),
        "",
        "## Reference Settings",
        "",
        "These 5 settings are sampled from other questions. They include architecture, "
        "optimizer, loss, the full learning-curve image, and final mean metric.",
    ]
    for item in reference_records:
        lines.extend(
            [
                "",
                f"### {item['label']}",
                "",
                item["setting_markdown"],
                "",
                f"Final mean {question['selection_metric']}: {item['mean_metric']:.8g}",
                f"![{item['label']} learning curve]({item['curve_image']})",
            ]
        )
    lines.extend(
        [
            "",
            "## Targets To Rank",
            "",
            "Return only JSON in this exact shape: "
            f"{{\"{blind_id}\":[\"X2\",\"X1\",\"X3\"]}}",
        ]
    )
    for item in target_records:
        lines.extend(["", f"### {item['label']}", "", item["setting_markdown"]])
    return "\n".join(lines).strip() + "\n"


def generate(
    source_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    reference_size: int,
    shard_size: int,
    force: bool,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("Output directory must be outside the source quiz directory")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    questions, answer_by_qid = _load_source(source_dir)
    choices_by_qid = _load_choices(questions, answer_by_qid)
    all_choices = [choice for choices in choices_by_qid.values() for choice in choices]
    rng = random.Random(seed)
    shards = _make_shards(len(questions), shard_size)
    source_index_by_qid = {q["question_id"]: idx for idx, q in enumerate(questions, start=1)}
    target_candidate_ids_by_source_index = {
        source_index_by_qid[qid]: {choice.candidate_id for choice in choices}
        for qid, choices in choices_by_qid.items()
    }

    answers: dict[str, list[str]] = {}
    metadata_questions: list[dict[str, Any]] = []
    public_questions: list[dict[str, Any]] = []

    for shard_index, source_indices in enumerate(shards, start=1):
        shard_id = f"agent_{chr(ord('A') + shard_index - 1)}"
        shard_dir = output_dir / "blind_shards" / shard_id
        shard_dir.mkdir(parents=True)
        excluded_source_indices = set(source_indices)
        shard_target_candidate_ids = {
            candidate_id
            for source_index in source_indices
            for candidate_id in target_candidate_ids_by_source_index[source_index]
        }
        for source_index in source_indices:
            q = questions[source_index - 1]
            qid = q["question_id"]
            blind_id = f"bq_{source_index:02d}"
            qdir = shard_dir / blind_id
            curves_dir = qdir / "curves"
            curves_dir.mkdir(parents=True)
            targets = choices_by_qid[qid]
            references = _sample_references(
                rng=rng,
                question=q,
                current_qid=qid,
                excluded_candidate_ids=shard_target_candidate_ids,
                excluded_source_indices=excluded_source_indices,
                all_choices=all_choices,
                reference_size=reference_size,
            )

            reference_records: list[dict[str, Any]] = []
            for ref_index, ref in enumerate(references, start=1):
                label = f"K{ref_index}"
                curve_name = f"{label}.png"
                _plot_curve(ref, curves_dir / curve_name, label)
                reference_records.append(
                    {
                        "label": label,
                        "setting_markdown": _setting_markdown(ref.spec),
                        "curve_image": f"curves/{curve_name}",
                        "mean_metric": ref.mean_metric,
                    }
                )

            shuffled_labels = [f"X{i}" for i in range(1, len(targets) + 1)]
            rng.shuffle(shuffled_labels)
            label_by_letter = {
                target.original_letter: shuffled_labels[i]
                for i, target in enumerate(targets)
            }
            target_records = [
                {
                    "label": label_by_letter[target.original_letter],
                    "setting_markdown": _setting_markdown(target.spec),
                }
                for target in targets
            ]
            rng.shuffle(target_records)

            true_targets = sorted(targets, key=lambda item: item.mean_metric)
            true_order = [label_by_letter[target.original_letter] for target in true_targets]
            expected_winner = answer_by_qid[qid]["correct_letter"]
            if true_targets[0].original_letter != expected_winner:
                raise ValueError(
                    f"Mean-metric winner for {qid} is {true_targets[0].original_letter}, "
                    f"but quiz answer key says {expected_winner}"
                )
            answers[blind_id] = true_order
            prompt = _render_prompt(
                blind_id=blind_id,
                question=q,
                reference_records=reference_records,
                target_records=target_records,
            )
            (qdir / "prompt.md").write_text(prompt, encoding="utf-8")
            write_json(
                qdir / "manifest.json",
                {
                    "blind_id": blind_id,
                    "source_question_number": source_index,
                    "family": q["family"],
                    "dataset_id": q["dataset_id"],
                    "metric": q["selection_metric"],
                    "target_labels": [record["label"] for record in target_records],
                    "reference_labels": [record["label"] for record in reference_records],
                },
            )

            metadata_questions.append(
                {
                    "blind_id": blind_id,
                    "source_question_number": source_index,
                    "source_question_id": qid,
                    "shard_id": shard_id,
                    "family": q["family"],
                    "dataset_id": q["dataset_id"],
                    "metric": q["selection_metric"],
                    "original_significance": {
                        "gap": answer_by_qid[qid].get("gap"),
                        "win_rate": answer_by_qid[qid].get("win_rate"),
                    },
                    "true_order": true_order,
                    "targets": [
                        {
                            "label": label_by_letter[target.original_letter],
                            "original_letter": target.original_letter,
                            "candidate_id": target.candidate_id,
                            "mean_metric": target.mean_metric,
                            "std_metric": target.std_metric,
                        }
                        for target in targets
                    ],
                    "references": [
                        {
                            "label": f"K{i}",
                            "source_question_number": ref.source_index,
                            "source_question_id": ref.question_id,
                            "original_letter": ref.original_letter,
                            "candidate_id": ref.candidate_id,
                            "mean_metric": ref.mean_metric,
                        }
                        for i, ref in enumerate(references, start=1)
                    ],
                }
            )
            public_questions.append(
                {
                    "blind_id": blind_id,
                    "shard_id": shard_id,
                    "family": q["family"],
                    "dataset_id": q["dataset_id"],
                    "metric": q["selection_metric"],
                    "prompt": f"blind_shards/{shard_id}/{blind_id}/prompt.md",
                }
            )

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "quiz_choice_ranking3_v1",
            "source": str(source_dir.relative_to(ROOT)),
            "seed": seed,
            "num_questions": len(questions),
            "reference_size": reference_size,
            "target_size": 3,
            "shard_size": shard_size,
            "num_shards": len(shards),
            "questions": public_questions,
            "instructions": "Agents should inspect only their assigned blind_shards/<agent_*> directory.",
        },
    )
    write_json(
        output_dir / "answer_key.json",
        {
            "schema_version": "quiz_choice_ranking3_answer_key_v1",
            "answers": answers,
        },
    )
    write_json(
        output_dir / "private_metadata.json",
        {
            "schema_version": "quiz_choice_ranking3_private_metadata_v1",
            "questions": metadata_questions,
        },
    )
    for shard_index, source_indices in enumerate(shards, start=1):
        shard_id = f"agent_{chr(ord('A') + shard_index - 1)}"
        write_json(
            output_dir / "blind_shards" / shard_id / "manifest.json",
            {
                "schema_version": "quiz_choice_ranking3_blind_shard_v1",
                "shard_id": shard_id,
                "num_questions": len(source_indices),
                "question_ids": [f"bq_{idx:02d}" for idx in source_indices],
                "answer_format": {"answers": {"bq_01": ["X2", "X1", "X3"]}},
                "instructions": "Use only prompt.md and curves inside this shard. Do not run experiments.",
            },
        )
    return {
        "output_dir": str(output_dir),
        "num_questions": len(questions),
        "num_shards": len(shards),
        "reference_size": reference_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-size", type=int, default=5)
    parser.add_argument("--shard-size", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = generate(
            args.source_dir,
            args.output_dir,
            seed=args.seed,
            reference_size=args.reference_size,
            shard_size=args.shard_size,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result["output_dir"])
    print(f"questions={result['num_questions']} shards={result['num_shards']} refs={result['reference_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
