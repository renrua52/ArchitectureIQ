#!/usr/bin/env python3
"""Generate ranking questions whose target candidates are pairwise significant."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))

from architecture_iq.profile import load_profile  # noqa: E402
from architecture_iq.significance.validator import validate_significance  # noqa: E402

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from common import compact_json, write_json  # noqa: E402
from generate import (  # noqa: E402
    _copy_sanitized_prompt_assets,
    _html_template,
    _load_candidates,
    _load_curves,
    _prompt_for_question,
    _question_from_groups,
    _repo_rel,
    _ui_data,
    read_json,
)


def _pairwise_passes(better: Any, worse: Any, profile: Any) -> bool:
    sig = validate_significance(
        [better.summary, worse.summary],
        profile,
        metric=better.metric,
    )
    return bool(sig.passed and sig.winner_index == 0)


def _pair_matrix(candidates: list[Any], profile: Any) -> list[list[bool]]:
    n = len(candidates)
    matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = _pairwise_passes(
                candidates[i],
                candidates[j],
                profile,
            )
    return matrix


def _is_clique(combo: tuple[int, ...], matrix: list[list[bool]]) -> bool:
    return all(matrix[i][j] for i, j in itertools.combinations(combo, 2))


def _find_disjoint_cliques(
    candidates: list[Any],
    matrix: list[list[bool]],
    *,
    target_size: int,
    num_questions: int,
    calibration_size: int,
    search_limit: int,
) -> list[tuple[int, ...]]:
    if num_questions * target_size + calibration_size > len(candidates):
        raise ValueError(
            "Not enough candidates for disjoint calibration and targets "
            f"({num_questions} * {target_size} + {calibration_size} > {len(candidates)})"
        )

    cliques: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(len(candidates)), target_size):
        if _is_clique(combo, matrix):
            cliques.append(combo)
    cliques.sort(
        key=lambda combo: (
            sum(candidates[i].mean_metric for i in combo),
            sum(combo),
            combo,
        )
    )

    best: list[tuple[int, ...]] = []
    nodes = 0

    def rec(start: int, used: set[int], chosen: list[tuple[int, ...]]) -> bool:
        nonlocal best, nodes
        nodes += 1
        if len(chosen) > len(best):
            best = list(chosen)
        if len(chosen) == num_questions:
            return True
        if nodes >= search_limit:
            return False
        remaining_slots = num_questions - len(chosen)
        if len(candidates) - len(used) < remaining_slots * target_size + calibration_size:
            return False
        for idx in range(start, len(cliques)):
            combo = cliques[idx]
            if any(item in used for item in combo):
                continue
            used.update(combo)
            chosen.append(combo)
            if rec(idx + 1, used, chosen):
                return True
            chosen.pop()
            for item in combo:
                used.remove(item)
        return False

    rec(0, set(), [])
    if len(best) < num_questions:
        raise RuntimeError(
            f"Found only {len(best)} disjoint pairwise-significant target groups; "
            f"need {num_questions}. Try fewer questions or a different candidate set."
        )
    return best


def _calibration_from_unused(
    candidates: list[Any],
    groups: list[tuple[int, ...]],
    calibration_size: int,
) -> list[Any]:
    used = {idx for group in groups for idx in group}
    unused = [candidate for idx, candidate in enumerate(candidates) if idx not in used]
    unused.sort(key=lambda candidate: candidate.mean_metric)
    if len(unused) < calibration_size:
        raise ValueError("Not enough unused candidates for calibration")
    if calibration_size == 1:
        return [unused[len(unused) // 2]]
    positions = [
        round(i * (len(unused) - 1) / (calibration_size - 1))
        for i in range(calibration_size)
    ]
    picked: list[Any] = []
    seen: set[str] = set()
    for pos in positions:
        candidate = unused[pos]
        if candidate.spec["candidate_id"] in seen:
            continue
        picked.append(candidate)
        seen.add(candidate.spec["candidate_id"])
    for candidate in unused:
        if len(picked) >= calibration_size:
            break
        if candidate.spec["candidate_id"] not in seen:
            picked.append(candidate)
            seen.add(candidate.spec["candidate_id"])
    return picked[:calibration_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_set", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ranking_questions")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", default="v1")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--calibration-size", type=int, default=6)
    parser.add_argument("--target-size", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--search-limit", type=int, default=250_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        print("run-name must be a single directory name", file=sys.stderr)
        return 1
    out_dir = args.output.resolve() / args.run_name
    if out_dir.exists():
        print(f"Output run already exists: {out_dir}", file=sys.stderr)
        return 1
    try:
        candidates = _load_candidates(
            args.candidate_set.resolve(),
            max_candidates=None if args.max_candidates == 0 else args.max_candidates,
        )
        candidates.sort(key=lambda candidate: candidate.mean_metric)
        for candidate in candidates:
            _load_curves(candidate)
        identities = {
            (candidate.spec["family"], candidate.spec["dataset_id"], candidate.metric)
            for candidate in candidates
        }
        if len(identities) != 1:
            raise ValueError("Candidates must share one family, dataset, and metric")
        profile = load_profile(args.profile)
        matrix = _pair_matrix(candidates, profile)
        edge_pairs = sum(
            1
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
            if matrix[i][j]
        )
        groups = _find_disjoint_cliques(
            candidates,
            matrix,
            target_size=args.target_size,
            num_questions=args.num_questions,
            calibration_size=args.calibration_size,
            search_limit=args.search_limit,
        )
        calibration = _calibration_from_unused(candidates, groups, args.calibration_size)
        set_manifest = read_json(args.candidate_set / "set.json")
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True)
    questions: list[dict[str, Any]] = []
    answer_by_id: dict[str, Any] = {}
    group_details: list[dict[str, Any]] = []
    for question_index, group in enumerate(groups, start=1):
        targets = [candidates[idx] for idx in group]
        question, answer_key = _question_from_groups(
            calibration=calibration,
            targets=targets,
            question_index=question_index,
            out_dir=out_dir,
            salt={
                "layout": "pairwise_significant",
                "group": [candidate.spec["candidate_id"] for candidate in targets],
            },
        )
        qdir = out_dir / "questions" / question["question_id"]
        write_json(qdir / "ranking_question.json", question)
        write_json(qdir / "answer_key.json", answer_key)
        (qdir / "prompt.md").write_text(_prompt_for_question(question), encoding="utf-8")
        questions.append(question)
        answer_by_id[question["question_id"]] = answer_key
        group_details.append(
            {
                "question_id": question["question_id"],
                "target_candidate_ids": [candidate.spec["candidate_id"] for candidate in targets],
                "target_mean_metrics": [candidate.mean_metric for candidate in targets],
                "target_std_metrics": [candidate.std_metric for candidate in targets],
            }
        )

    manifest = {
        "schema_version": "ranking_manifest_v1",
        "run_id": args.run_name,
        "candidate_set": _repo_rel(args.candidate_set.resolve()),
        "dataset_id": set_manifest["dataset_id"],
        "family": set_manifest["family"],
        "metric": questions[0]["metric"],
        "lower_is_better": True,
        "eligible_candidates": len(candidates),
        "num_questions": len(questions),
        "layout": "pairwise_significant",
        "calibration_size": args.calibration_size,
        "target_size": args.target_size,
        "pairwise_significance": {
            "profile": args.profile,
            "gap_min": profile.significance["gap_min"],
            "win_rate_min": profile.significance["win_rate_min"],
            "use_non_overlap": profile.significance.get("use_non_overlap", True),
            "required_pairs_per_question": args.target_size * (args.target_size - 1) // 2,
            "passing_pairs_in_pool": edge_pairs,
            "total_pairs_in_pool": len(candidates) * (len(candidates) - 1) // 2,
        },
        "question_ids": [q["question_id"] for q in questions],
        "question_paths": [f"questions/{q['question_id']}" for q in questions],
        "calibration_candidate_ids": [candidate.spec["candidate_id"] for candidate in calibration],
        "group_details": group_details,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "answer_key.json", {"questions": answer_by_id})

    llm_eval_dir = out_dir / "llm_eval"
    for qid in manifest["question_ids"]:
        _copy_sanitized_prompt_assets(out_dir / "questions" / qid, llm_eval_dir)
    write_json(
        llm_eval_dir / "README.json",
        {
            "instructions": (
                "Use only prompt.md and curve images in this directory. "
                "Do not inspect parent directories, candidate result files, or answer keys."
            ),
            "question_ids": manifest["question_ids"],
            "answer_format": {"rq_id": ["T3", "T1", "T5", "T2", "T4"]},
        },
    )
    (out_dir / "index.html").write_text(
        _html_template(_ui_data(manifest, questions, answer_by_id)),
        encoding="utf-8",
    )
    print(out_dir)
    print(compact_json(manifest["pairwise_significance"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
