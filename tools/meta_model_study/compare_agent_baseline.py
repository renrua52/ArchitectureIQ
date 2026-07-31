"""Fair, paired top-1 comparison of agent and meta-model answers.

The CLI opens only the explicitly supplied question, answer-key, answer, and
optional identity-manifest JSON files. Candidate paths are treated as opaque
data and are never resolved or opened.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "agent_baseline_comparison_v1"
IDENTITY_MANIFEST_HINT = (
    '{"questions": [{"source_question_id": "...", '
    '"question_id": "...", "choices": '
    '[{"letter": "A", "candidate_id": "c_..."}]}]}'
)


class ComparisonInputError(ValueError):
    """Raised when inputs cannot be paired without guessing identity."""


@dataclass(frozen=True)
class PublicQuestion:
    """Canonical public identity for one question."""

    question_id: str
    cluster_id: str
    cluster_field: str
    choices: Mapping[str, str]
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class IdentityEntry:
    """Mapping from one source-local question ID to canonical identity."""

    source_question_id: str
    question_id: str
    choices: Mapping[str, str]


@dataclass(frozen=True)
class Top1Answer:
    """One normalized top-1 answer with independently checkable identity."""

    question_id: str
    letter: str
    candidate_id: str


def _require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonInputError(f"{context} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonInputError(f"{context} must be a non-empty string")
    return value


def _load_json(path: Path, role: str) -> Any:
    path = Path(path)
    parts = path.parts
    if any(
        part == "results" and index > 0 and parts[index - 1].startswith("c_")
        for index, part in enumerate(parts)
    ):
        raise ComparisonInputError(
            f"{role} points inside a candidate results directory; candidate "
            "results are forbidden"
        )
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise ComparisonInputError(f"Could not read {role} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonInputError(f"Invalid JSON in {role} {path}: {exc}") from exc


def _unwrap_question_rows(value: Any, role: str) -> list[Mapping[str, Any]]:
    if isinstance(value, dict):
        value = value.get("questions")
    if not isinstance(value, list) or not value:
        raise ComparisonInputError(
            f"{role} must be a non-empty JSON list or an object with questions"
        )
    return [_require_object(row, f"{role}[{index}]") for index, row in enumerate(value)]


def _parse_choice_map(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        raise ComparisonInputError(
            f"{context} must contain a non-empty choice identity list"
        )
    choices: dict[str, str] = {}
    seen_candidates: set[str] = set()
    for index, raw_choice in enumerate(value):
        choice_context = f"{context}[{index}]"
        choice = _require_object(raw_choice, choice_context)
        letter = _require_nonempty_string(
            choice.get("letter"), f"{choice_context}.letter"
        )
        candidate_id = _require_nonempty_string(
            choice.get("candidate_id"), f"{choice_context}.candidate_id"
        )
        if letter in choices:
            raise ComparisonInputError(
                f"Duplicate choice letter {letter!r} in {context}"
            )
        if candidate_id in seen_candidates:
            raise ComparisonInputError(
                f"Duplicate candidate_id {candidate_id!r} in {context}"
            )
        choices[letter] = candidate_id
        seen_candidates.add(candidate_id)
    return choices


def _assert_choice_identity(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
    context: str,
) -> None:
    if dict(actual) != dict(expected):
        raise ComparisonInputError(
            f"{context} choice identity mismatch: "
            f"expected {dict(expected)!r}, got {dict(actual)!r}"
        )


def _index_public_questions(value: Any) -> dict[str, PublicQuestion]:
    questions: dict[str, PublicQuestion] = {}
    for index, row in enumerate(_unwrap_question_rows(value, "public questions")):
        context = f"public questions[{index}]"
        question_id = _require_nonempty_string(
            row.get("question_id"), f"{context}.question_id"
        )
        if question_id in questions:
            raise ComparisonInputError(f"Duplicate public question_id: {question_id}")

        cluster_field = ""
        cluster_id = ""
        for field in ("question_cluster_id", "cluster_id", "question_run_id"):
            if row.get(field) is not None:
                cluster_field = field
                cluster_id = _require_nonempty_string(row[field], f"{context}.{field}")
                break
        if not cluster_id:
            raise ComparisonInputError(
                f"{context} needs question_cluster_id, cluster_id, or "
                "question_run_id for cluster-paired bootstrap"
            )
        choices = _parse_choice_map(row.get("choices"), f"{context}.choices")
        metadata = {
            field: row[field]
            for field in ("question_run_id", "family", "dataset_id")
            if isinstance(row.get(field), str) and row[field]
        }
        questions[question_id] = PublicQuestion(
            question_id=question_id,
            cluster_id=cluster_id,
            cluster_field=cluster_field,
            choices=choices,
            metadata=metadata,
        )
    return questions


def _set_mismatch_message(
    role: str,
    expected_ids: set[str],
    actual_ids: set[str],
) -> str:
    return (
        f"{role} question_id set differs from public questions: "
        f"missing={sorted(expected_ids - actual_ids)!r}, "
        f"unexpected={sorted(actual_ids - expected_ids)!r}"
    )


def _load_answer_key(
    value: Any,
    public_questions: Mapping[str, PublicQuestion],
) -> dict[str, str]:
    expected_ids = set(public_questions)
    correct_by_id: dict[str, str] = {}
    for index, row in enumerate(_unwrap_question_rows(value, "private answer key")):
        context = f"private answer key[{index}]"
        question_id = _require_nonempty_string(
            row.get("question_id"), f"{context}.question_id"
        )
        if question_id in correct_by_id:
            raise ComparisonInputError(
                f"Duplicate private answer-key question_id: {question_id}"
            )
        if question_id not in public_questions:
            correct_by_id[question_id] = ""
            continue
        public = public_questions[question_id]
        key_choices = _parse_choice_map(row.get("choices"), f"{context}.choices")
        _assert_choice_identity(key_choices, public.choices, context)
        for field, public_value in public.metadata.items():
            key_value = row.get(field)
            if key_value is not None and key_value != public_value:
                raise ComparisonInputError(
                    f"{context}.{field} identity mismatch: expected "
                    f"{public_value!r}, got {key_value!r}"
                )
        correct_letter = _require_nonempty_string(
            row.get("correct_letter"), f"{context}.correct_letter"
        )
        if correct_letter not in public.choices:
            raise ComparisonInputError(
                f"{context}.correct_letter {correct_letter!r} is not a public choice"
            )
        supplied_candidate = row.get("correct_candidate_id")
        if supplied_candidate is not None:
            supplied_candidate = _require_nonempty_string(
                supplied_candidate, f"{context}.correct_candidate_id"
            )
            expected_candidate = public.choices[correct_letter]
            if supplied_candidate != expected_candidate:
                raise ComparisonInputError(
                    f"{context} correct choice identity mismatch: letter "
                    f"{correct_letter!r} maps to {expected_candidate!r}, got "
                    f"{supplied_candidate!r}"
                )
        correct_by_id[question_id] = correct_letter

    if set(correct_by_id) != expected_ids:
        raise ComparisonInputError(
            _set_mismatch_message(
                "private answer key", expected_ids, set(correct_by_id)
            )
        )
    return correct_by_id


def _load_identity_manifest(
    value: Any,
    public_questions: Mapping[str, PublicQuestion],
    role: str,
) -> dict[str, IdentityEntry]:
    rows = _unwrap_question_rows(value, f"{role} identity manifest")
    by_source: dict[str, IdentityEntry] = {}
    canonical_ids: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{role} identity manifest[{index}]"
        source_question_id = _require_nonempty_string(
            row.get("source_question_id"), f"{context}.source_question_id"
        )
        question_id = _require_nonempty_string(
            row.get("question_id"), f"{context}.question_id"
        )
        if source_question_id in by_source:
            raise ComparisonInputError(
                f"Duplicate {role} manifest source_question_id: {source_question_id}"
            )
        if question_id in canonical_ids:
            raise ComparisonInputError(
                f"Duplicate {role} manifest question_id: {question_id}"
            )
        if question_id not in public_questions:
            raise ComparisonInputError(
                f"{context}.question_id {question_id!r} is not public"
            )
        choices = _parse_choice_map(row.get("choices"), f"{context}.choices")
        _assert_choice_identity(choices, public_questions[question_id].choices, context)
        entry = IdentityEntry(source_question_id, question_id, choices)
        by_source[source_question_id] = entry
        canonical_ids.add(question_id)

    expected_ids = set(public_questions)
    if canonical_ids != expected_ids:
        raise ComparisonInputError(
            _set_mismatch_message(
                f"{role} identity manifest", expected_ids, canonical_ids
            )
        )
    return by_source


def _extract_answer_rows(
    value: Any,
    role: str,
) -> tuple[list[Mapping[str, Any]], str]:
    if isinstance(value, list):
        if not value:
            raise ComparisonInputError(f"{role} answer list must not be empty")
        return [
            _require_object(row, f"{role}[{index}]") for index, row in enumerate(value)
        ], "top_level_list"

    payload = _require_object(value, role)
    collection_keys = [
        key
        for key in ("predictions", "records", "answers", "questions")
        if key in payload
    ]
    if not collection_keys:
        raise ComparisonInputError(
            f"{role} has no per-question top-1 predictions/answers; aggregate "
            "score files cannot be paired"
        )
    if len(collection_keys) != 1:
        raise ComparisonInputError(
            f"{role} has ambiguous per-question collections: {collection_keys!r}"
        )
    collection_key = collection_keys[0]
    collection = payload[collection_key]
    if collection_key == "answers" and isinstance(collection, dict):
        if not collection:
            raise ComparisonInputError(f"{role}.answers must not be empty")
        rows: list[Mapping[str, Any]] = []
        for source_question_id, raw_answer in collection.items():
            source_question_id = _require_nonempty_string(
                source_question_id, f"{role}.answers key"
            )
            if isinstance(raw_answer, str):
                row: dict[str, Any] = {"answer": raw_answer}
            else:
                row = dict(
                    _require_object(
                        raw_answer, f"{role}.answers[{source_question_id!r}]"
                    )
                )
            row["_mapping_source_question_id"] = source_question_id
            rows.append(row)
        return rows, "answers_map"
    if not isinstance(collection, list) or not collection:
        raise ComparisonInputError(f"{role}.{collection_key} must be a non-empty list")
    return [
        _require_object(row, f"{role}.{collection_key}[{index}]")
        for index, row in enumerate(collection)
    ], collection_key


def _string_alias(
    row: Mapping[str, Any],
    fields: Sequence[str],
    context: str,
) -> str | None:
    values: list[tuple[str, str]] = []
    for field in fields:
        if row.get(field) is not None:
            values.append(
                (field, _require_nonempty_string(row[field], f"{context}.{field}"))
            )
    if not values:
        return None
    if len({value for _, value in values}) != 1:
        raise ComparisonInputError(f"{context} has conflicting aliases: {values!r}")
    return values[0][1]


def _normalize_answers(
    value: Any,
    public_questions: Mapping[str, PublicQuestion],
    role: str,
    manifest: Mapping[str, IdentityEntry] | None,
) -> tuple[dict[str, Top1Answer], str]:
    rows, adapter = _extract_answer_rows(value, role)
    normalized: dict[str, Top1Answer] = {}
    for index, row in enumerate(rows):
        context = f"{role}.{adapter}[{index}]"
        canonical_question_id = _string_alias(row, ("question_id",), context)
        source_question_id = _string_alias(
            row,
            (
                "_mapping_source_question_id",
                "source_question_id",
                "question",
            ),
            context,
        )
        manifest_lookup_id = source_question_id or canonical_question_id
        manifest_entry: IdentityEntry | None = None
        if manifest is not None:
            if manifest_lookup_id is None or manifest_lookup_id not in manifest:
                raise ComparisonInputError(
                    f"{context} has no matching entry in the {role} identity manifest"
                )
            manifest_entry = manifest[manifest_lookup_id]
            if (
                canonical_question_id is not None
                and canonical_question_id != manifest_entry.question_id
            ):
                raise ComparisonInputError(
                    f"{context} question identity mismatch: row has "
                    f"{canonical_question_id!r}, manifest maps to "
                    f"{manifest_entry.question_id!r}"
                )
            question_id = manifest_entry.question_id
        else:
            if canonical_question_id is None:
                if source_question_id in public_questions:
                    question_id = source_question_id
                else:
                    raise ComparisonInputError(
                        f"{context} lacks canonical question_id; provide "
                        f"--{role}-identity-manifest with {IDENTITY_MANIFEST_HINT}"
                    )
            else:
                question_id = canonical_question_id
            if (
                canonical_question_id is not None
                and source_question_id is not None
                and source_question_id != question_id
            ):
                raise ComparisonInputError(
                    f"{context} uses a source-local question ID without an identity "
                    f"manifest; provide --{role}-identity-manifest"
                )

        if question_id not in public_questions:
            raise ComparisonInputError(
                f"{context}.question_id {question_id!r} is not public"
            )
        public = public_questions[question_id]
        letter = _string_alias(
            row,
            ("predicted_letter", "selected_letter", "answer", "letter"),
            context,
        )
        if letter is None:
            raise ComparisonInputError(f"{context} lacks a top-1 answer letter")
        if letter not in public.choices:
            raise ComparisonInputError(
                f"{context} answer {letter!r} is not a public choice for {question_id}"
            )

        for choices_field in ("choice_predictions", "choices"):
            if row.get(choices_field) is not None:
                supplied_choices = _parse_choice_map(
                    row[choices_field], f"{context}.{choices_field}"
                )
                _assert_choice_identity(supplied_choices, public.choices, context)

        candidate_id = _string_alias(
            row,
            ("predicted_candidate_id", "selected_candidate_id", "candidate_id"),
            context,
        )
        if candidate_id is None:
            if manifest_entry is None:
                raise ComparisonInputError(
                    f"{context} lacks candidate_id for its top-1 answer; provide "
                    f"--{role}-identity-manifest with {IDENTITY_MANIFEST_HINT}"
                )
            candidate_id = manifest_entry.choices[letter]
        expected_candidate = public.choices[letter]
        if candidate_id != expected_candidate:
            raise ComparisonInputError(
                f"{context} top-1 choice identity mismatch: letter {letter!r} "
                f"maps to {expected_candidate!r}, got {candidate_id!r}"
            )
        if (
            manifest_entry is not None
            and manifest_entry.choices[letter] != candidate_id
        ):
            raise ComparisonInputError(
                f"{context} top-1 candidate_id disagrees with the identity manifest"
            )

        for field, public_value in public.metadata.items():
            supplied_value = row.get(field)
            if supplied_value is not None and supplied_value != public_value:
                raise ComparisonInputError(
                    f"{context}.{field} identity mismatch: expected "
                    f"{public_value!r}, got {supplied_value!r}"
                )
        if question_id in normalized:
            raise ComparisonInputError(f"Duplicate {role} question_id: {question_id}")
        normalized[question_id] = Top1Answer(question_id, letter, candidate_id)

    expected_ids = set(public_questions)
    if set(normalized) != expected_ids:
        raise ComparisonInputError(
            _set_mismatch_message(role, expected_ids, set(normalized))
        )
    return normalized, adapter


def exact_mcnemar_two_sided_p_value(
    agent_only_correct: int,
    baseline_only_correct: int,
) -> float:
    """Return the exact two-sided McNemar p-value for discordant pairs."""

    for name, value in (
        ("agent_only_correct", agent_only_correct),
        ("baseline_only_correct", baseline_only_correct),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    discordant = agent_only_correct + baseline_only_correct
    if discordant == 0:
        return 1.0
    tail_end = min(agent_only_correct, baseline_only_correct)
    tail = Fraction(
        sum(comb(discordant, k) for k in range(tail_end + 1)), 1 << discordant
    )
    return float(min(Fraction(1), 2 * tail))


def exact_mcnemar_p_value(
    agent_only_correct: int,
    baseline_only_correct: int,
) -> float:
    """Backward-friendly short alias for the exact two-sided test."""

    return exact_mcnemar_two_sided_p_value(
        agent_only_correct,
        baseline_only_correct,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def paired_cluster_bootstrap_ci(
    paired_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    reps: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap agent-minus-baseline accuracy by resampling whole clusters."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0:
        raise ValueError("reps must be a positive integer")

    clusters: dict[str, list[int]] = {}
    total_delta = 0
    total_questions = 0
    for index, row in enumerate(paired_rows):
        context = f"paired_rows[{index}]"
        cluster_id = _require_nonempty_string(
            row.get("cluster_id"), f"{context}.cluster_id"
        )
        agent_correct = row.get("agent_correct")
        baseline_correct = row.get("baseline_correct")
        if not isinstance(agent_correct, bool) or not isinstance(
            baseline_correct, bool
        ):
            raise ValueError(
                f"{context}.agent_correct and baseline_correct must be booleans"
            )
        delta = int(agent_correct) - int(baseline_correct)
        clusters.setdefault(cluster_id, []).append(delta)
        total_delta += delta
        total_questions += 1
    if not clusters:
        raise ValueError("paired_rows must not be empty")

    cluster_stats = [
        (sum(clusters[cluster_id]), len(clusters[cluster_id]))
        for cluster_id in sorted(clusters)
    ]
    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    for _ in range(reps):
        sampled_delta = 0
        sampled_questions = 0
        for _ in cluster_stats:
            delta, count = cluster_stats[rng.randrange(len(cluster_stats))]
            sampled_delta += delta
            sampled_questions += count
        bootstrap_differences.append(sampled_delta / sampled_questions)
    bootstrap_differences.sort()
    low = _percentile(bootstrap_differences, 0.025)
    high = _percentile(bootstrap_differences, 0.975)
    return {
        "method": "paired percentile bootstrap resampling whole question clusters",
        "confidence_level": 0.95,
        "seed": seed,
        "reps": reps,
        "num_clusters": len(cluster_stats),
        "point_estimate": total_delta / total_questions,
        "low": low,
        "high": high,
    }


def compare_agent_baseline(
    public_questions_path: Path,
    private_answer_key_path: Path,
    baseline_path: Path,
    agent_answers_path: Path,
    *,
    baseline_identity_manifest_path: Path | None = None,
    agent_identity_manifest_path: Path | None = None,
    seed: int = 0,
    reps: int = 10_000,
) -> dict[str, Any]:
    """Load, strictly pair, and compare agent and baseline top-1 answers."""

    public_questions = _index_public_questions(
        _load_json(public_questions_path, "public questions")
    )
    answer_key = _load_answer_key(
        _load_json(private_answer_key_path, "private answer key"),
        public_questions,
    )

    baseline_manifest = None
    if baseline_identity_manifest_path is not None:
        baseline_manifest = _load_identity_manifest(
            _load_json(baseline_identity_manifest_path, "baseline identity manifest"),
            public_questions,
            "baseline",
        )
    agent_manifest = None
    if agent_identity_manifest_path is not None:
        agent_manifest = _load_identity_manifest(
            _load_json(agent_identity_manifest_path, "agent identity manifest"),
            public_questions,
            "agent",
        )

    baseline, baseline_adapter = _normalize_answers(
        _load_json(baseline_path, "baseline"),
        public_questions,
        "baseline",
        baseline_manifest,
    )
    agent, agent_adapter = _normalize_answers(
        _load_json(agent_answers_path, "agent"),
        public_questions,
        "agent",
        agent_manifest,
    )

    paired_rows: list[dict[str, Any]] = []
    both_correct = 0
    agent_only = 0
    baseline_only = 0
    both_wrong = 0
    for question_id, public in public_questions.items():
        correct_letter = answer_key[question_id]
        agent_correct = agent[question_id].letter == correct_letter
        baseline_correct = baseline[question_id].letter == correct_letter
        both_correct += int(agent_correct and baseline_correct)
        agent_only += int(agent_correct and not baseline_correct)
        baseline_only += int(not agent_correct and baseline_correct)
        both_wrong += int(not agent_correct and not baseline_correct)
        paired_rows.append(
            {
                "cluster_id": public.cluster_id,
                "agent_correct": agent_correct,
                "baseline_correct": baseline_correct,
            }
        )

    total = len(public_questions)
    agent_correct_count = both_correct + agent_only
    baseline_correct_count = both_correct + baseline_only
    difference_count = agent_correct_count - baseline_correct_count
    p_value = exact_mcnemar_two_sided_p_value(agent_only, baseline_only)
    bootstrap = paired_cluster_bootstrap_ci(paired_rows, seed=seed, reps=reps)
    bootstrap["cluster_fields"] = sorted(
        {question.cluster_field for question in public_questions.values()}
    )
    bootstrap_excludes_zero = bootstrap["low"] > 0 or bootstrap["high"] < 0
    point_estimate_note = (
        "55/60 versus 54/60 is only a point-estimate difference; "
        "55/60 > 54/60 does not by itself establish statistical significance."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scoring": "paired top-1 accuracy",
        "num_questions": total,
        "agent": {
            "correct": agent_correct_count,
            "total": total,
            "accuracy": agent_correct_count / total,
        },
        "baseline": {
            "correct": baseline_correct_count,
            "total": total,
            "accuracy": baseline_correct_count / total,
        },
        "paired_2x2": {
            "both_correct": both_correct,
            "agent_correct_baseline_wrong": agent_only,
            "agent_wrong_baseline_correct": baseline_only,
            "both_wrong": both_wrong,
        },
        "difference": {
            "direction": "agent_minus_baseline",
            "correct": difference_count,
            "accuracy": difference_count / total,
            "percentage_points": 100.0 * difference_count / total,
        },
        "mcnemar_exact_two_sided": {
            "agent_only_correct": agent_only,
            "baseline_only_correct": baseline_only,
            "discordant_pairs": agent_only + baseline_only,
            "p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
        },
        "cluster_paired_bootstrap_95_ci": bootstrap,
        "interpretation": {
            "point_estimate_note": point_estimate_note,
            "mcnemar_significant_at_0_05": p_value < 0.05,
            "cluster_bootstrap_ci_excludes_zero": bootstrap_excludes_zero,
        },
        "input_adapters": {
            "baseline": baseline_adapter,
            "agent": agent_adapter,
            "baseline_identity_manifest_used": baseline_manifest is not None,
            "agent_identity_manifest_used": agent_manifest is not None,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only paired top-1 comparison of agent answers and a "
            "meta-model baseline. Results are printed as JSON to stdout."
        )
    )
    parser.add_argument(
        "--public-questions",
        "--questions",
        dest="public_questions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--private-answer-key",
        "--answer-key",
        dest="private_answer_key",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline",
        "--baseline-predictions",
        "--baseline-answers",
        dest="baseline",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--agent-answers", "--agent", dest="agent_answers", type=Path, required=True
    )
    parser.add_argument("--baseline-identity-manifest", type=Path)
    parser.add_argument("--agent-identity-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = compare_agent_baseline(
            args.public_questions,
            args.private_answer_key,
            args.baseline,
            args.agent_answers,
            baseline_identity_manifest_path=args.baseline_identity_manifest,
            agent_identity_manifest_path=args.agent_identity_manifest,
            seed=args.seed,
            reps=args.reps,
        )
    except (ComparisonInputError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
