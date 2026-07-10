#!/usr/bin/env python3
"""Create a de-identified ranking-question bundle for less leaky agent eval."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from common import read_json, write_json  # noqa: E402


def _setting_block(item: dict[str, Any]) -> str:
    return item["setting_markdown"]


def _render_prompt(
    *,
    blind_id: str,
    question: dict[str, Any],
    calibration: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    omit_calibration: bool,
) -> str:
    lines = [
        "# ArchitectureIQ Ranking Question",
        "",
        "Rank the target settings from best to worst. Do not run experiments and do not "
        "inspect any files outside this blind bundle.",
        "",
        f"Blind question: `{blind_id}`",
        f"Task family: `{question['family']}`",
        f"Metric: `{question['metric']}`; lower is better.",
        "",
    ]
    if omit_calibration:
        lines.extend(
            [
                "No calibration examples or learning curves are provided in this version. "
                "Use only the target setting descriptions.",
                "",
            ]
        )
    else:
        lines.append(
            "Use only the calibration examples, their curve images, and the target "
            "setting descriptions below."
        )
        lines.extend(["", "## Calibration Examples"])
        for item in calibration:
            lines.extend(
                [
                    "",
                    f"### {item['label']}",
                    "",
                    _setting_block(item),
                    "",
                    f"Final mean {question['metric']}: {item['mean_metric']:.8g}",
                    f"![{item['label']} learning curve](curves/{item['label']}.png)",
                ]
            )
    lines.extend(
        [
            "",
            "## Targets To Rank",
            "",
            "Return only the target labels in best-to-worst order, for example: "
            "`X3,X1,X5,X2,X4`.",
        ]
    )
    for item in targets:
        lines.extend(["", f"### {item['label']}", "", _setting_block(item)])
    return "\n".join(lines).strip() + "\n"


def make_blind_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    omit_calibration: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir):
        raise ValueError("Blind output must be outside the source ranking run.")
    if run_dir.is_relative_to(output_dir):
        raise ValueError("Blind output cannot contain the source ranking run.")

    rng = random.Random(seed)
    manifest = read_json(run_dir / "manifest.json")
    full_answer_key = read_json(run_dir / "answer_key.json")["questions"]

    for qid in manifest["question_ids"]:
        source_qdir = run_dir / "questions" / qid
        if qid not in full_answer_key:
            raise ValueError(f"Missing answer key for {qid}")
        question_path = source_qdir / "ranking_question.json"
        if not question_path.is_file():
            raise FileNotFoundError(question_path)
        if not omit_calibration:
            question = read_json(question_path)
            for item in question["calibration"]:
                curve_path = source_qdir / item["curve_image"]
                if not curve_path.is_file():
                    raise FileNotFoundError(curve_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    blind_answers: dict[str, list[str]] = {}
    public_questions: list[dict[str, Any]] = []

    for i, qid in enumerate(manifest["question_ids"], start=1):
        blind_id = f"bq_{i:02d}"
        source_qdir = run_dir / "questions" / qid
        q = read_json(source_qdir / "ranking_question.json")
        qout = output_dir / blind_id
        qout.mkdir(parents=True, exist_ok=True)
        curves_out = qout / "curves"
        if not omit_calibration:
            curves_out.mkdir(parents=True, exist_ok=True)

        calibration = []
        if not omit_calibration:
            for idx, item in enumerate(q["calibration"], start=1):
                label = f"K{idx}"
                new_item = {
                    "label": label,
                    "setting_markdown": item["setting_markdown"],
                    "mean_metric": item["mean_metric"],
                }
                src_curve = source_qdir / item["curve_image"]
                shutil.copy2(src_curve, curves_out / f"{label}.png")
                calibration.append(new_item)

        raw_targets = []
        for idx, item in enumerate(q["targets"], start=1):
            raw_targets.append(
                {
                    "old_label": item["label"],
                    "setting_markdown": item["setting_markdown"],
                }
            )
        shuffled_labels = [f"X{idx}" for idx in range(1, len(raw_targets) + 1)]
        rng.shuffle(shuffled_labels)
        label_by_old = {
            item["old_label"]: shuffled_labels[idx]
            for idx, item in enumerate(raw_targets)
        }
        targets = [
            {"label": label_by_old[item["old_label"]], "setting_markdown": item["setting_markdown"]}
            for item in raw_targets
        ]
        rng.shuffle(targets)
        blind_answers[blind_id] = [
            label_by_old[old_label]
            for old_label in full_answer_key[qid]["true_order"]
        ]

        prompt = _render_prompt(
            blind_id=blind_id,
            question=q,
            calibration=calibration,
            targets=targets,
            omit_calibration=omit_calibration,
        )
        (qout / "prompt.md").write_text(prompt, encoding="utf-8")
        public_questions.append(
            {
                "blind_id": blind_id,
                "metric": q["metric"],
                "target_labels": [item["label"] for item in targets],
            }
        )

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ranking_blind_bundle_v1",
            "omit_calibration": omit_calibration,
            "num_questions": len(public_questions),
            "questions": public_questions,
            "instructions": "Use only prompt.md and curves in each blind question directory.",
            "answer_format": {"answers": {"bq_01": ["X3", "X1", "X5", "X2", "X4"]}},
        },
    )
    return {
        "schema_version": "ranking_blind_answer_key_v1",
        "answers": blind_answers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--omit-calibration",
        action="store_true",
        help="Write target-only prompts with no calibration examples or curve images.",
    )
    parser.add_argument(
        "--answer-key-output",
        type=Path,
        default=None,
        help="Optional path for writing the blind answer key. Omit during blind eval.",
    )
    parser.add_argument(
        "--print-answer-key",
        action="store_true",
        help="Print the blind answer key to stdout without writing it into the bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    answer_key_output = (
        args.answer_key_output.resolve() if args.answer_key_output is not None else None
    )
    if answer_key_output is None and not args.print_answer_key:
        print(
            "Specify --answer-key-output outside the blind bundle or use "
            "--print-answer-key.",
            file=sys.stderr,
        )
        return 1
    if answer_key_output is not None and (
        answer_key_output == output_dir or answer_key_output.is_relative_to(output_dir)
    ):
        print("Answer key output must be outside the blind bundle.", file=sys.stderr)
        return 1
    try:
        key = make_blind_bundle(
            args.run_dir,
            output_dir,
            seed=args.seed,
            omit_calibration=args.omit_calibration,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if answer_key_output is not None:
        write_json(answer_key_output, key)
    if args.print_answer_key:
        import json

        print(json.dumps(key, indent=2))
    else:
        print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
