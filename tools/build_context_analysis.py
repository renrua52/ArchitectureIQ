from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUIZ60 = ROOT / "artifacts" / "quiz_attempt_60"
ARCHIVE65 = ROOT / "artifacts" / "_archive_default_unused" / "quiz_attempt_65"
OUT = QUIZ60 / "context_analysis"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], ensure_ascii=False)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def choice_candidate_id(question: dict[str, Any], letter: str | None) -> str | None:
    if not letter:
        return None
    for choice in question.get("choices", []):
        if choice.get("letter") == letter:
            return choice.get("candidate_id")
    return None


def positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 1)


def parameter_count(model: dict[str, Any]) -> int | None:
    model_type = model.get("type")
    if model_type == "mlp":
        input_dim = positive_int(model.get("input_dim"), 1)
        depth = positive_int(model.get("depth"))
        width = positive_int(model.get("width"))
        layer_norm = model.get("layer_norm")
        if not isinstance(layer_norm, list):
            layer_norm = [False] * depth
        count = input_dim * width + width
        count += depth * (width * width + width)
        count += 2 * width * sum(1 for item in layer_norm[:depth] if item)
        count += width + 1
        return count
    if model_type in {"transformer", "transformer_lm"}:
        d_model = positive_int(model.get("d_model", model.get("embed_dim")))
        d_ff = positive_int(model.get("d_ff", model.get("ff_dim")))
        layers = positive_int(model.get("num_layers"))
        vocab = positive_int(model.get("vocab_size"))
        context = positive_int(model.get("context_length"))
        embeddings = vocab * d_model + context * d_model
        per_layer = 4 * d_model * d_model + 2 * d_model * d_ff
        per_layer += 9 * d_model + d_ff
        output_head = d_model * vocab + vocab
        return embeddings + layers * per_layer + output_head
    return None


def model_shape(model: dict[str, Any]) -> dict[str, Any]:
    params = parameter_count(model)
    if model.get("type") == "mlp":
        return {
            "type": "mlp",
            "depth": model.get("depth"),
            "width": model.get("width"),
            "residual": model.get("residual"),
            "trainable_params_est": params,
        }
    if model.get("type") in {"transformer", "transformer_lm"}:
        return {
            "type": "transformer_lm",
            "depth": model.get("num_layers"),
            "width": model.get("d_model", model.get("embed_dim")),
            "d_ff": model.get("d_ff", model.get("ff_dim")),
            "heads": model.get("num_heads"),
            "trainable_params_est": params,
        }
    return {"type": model.get("type"), "trainable_params_est": params}


def load_base() -> dict[str, Any]:
    questions = load_json(QUIZ60 / "questions_sanitized.json")
    answers = load_json(QUIZ60 / "answer_key.json")
    feedback = load_json(QUIZ60 / "learning_feedback_key.json")
    index = load_json(QUIZ60 / "index.json")
    q_by_id = {item["question_id"]: item for item in questions}
    a_by_id = {item["question_id"]: item for item in answers}
    f_by_id = {item["question_id"]: item for item in feedback}
    n_by_id = {item["question_id"]: i + 1 for i, item in enumerate(questions)}
    old_n_by_id = {item["question_id"]: item.get("old_n") for item in index}
    return {
        "questions": questions,
        "answers": answers,
        "feedback": feedback,
        "q_by_id": q_by_id,
        "a_by_id": a_by_id,
        "f_by_id": f_by_id,
        "n_by_id": n_by_id,
        "old_n_by_id": old_n_by_id,
    }


def build_question_features(base: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qmeta: dict[str, Any] = {}
    for question in base["questions"]:
        qid = question["question_id"]
        answer = base["a_by_id"][qid]
        feedback = base["f_by_id"][qid]
        path_by_letter = {choice["letter"]: choice["candidate_path"] for choice in answer["choices"]}
        metric_values = feedback.get("choice_mean_metrics", {})
        choice_rows = []
        for choice in question["choices"]:
            letter = choice["letter"]
            opt = choice.get("optimizer", {})
            shape = model_shape(choice.get("model", {}))
            choice_rows.append(
                {
                    "letter": letter,
                    "candidate_id": choice["candidate_id"],
                    "candidate_path": path_by_letter.get(letter),
                    "metric_value": metric_values.get(letter),
                    "is_correct": letter == answer["correct_letter"],
                    "model": choice.get("model", {}),
                    "model_shape": shape,
                    "optimizer": opt,
                    "optimizer_type": opt.get("type"),
                    "learning_rate": opt.get("lr"),
                    "weight_decay": opt.get("weight_decay", 0),
                    "loss": choice.get("loss", {}),
                    "budget": choice.get("budget", {}),
                }
            )
        max_params = max(
            row["model_shape"]["trainable_params_est"] or -1 for row in choice_rows
        )
        max_lr = max(row["learning_rate"] or -1 for row in choice_rows)
        max_param_letters = [
            row["letter"]
            for row in choice_rows
            if row["model_shape"]["trainable_params_est"] == max_params
        ]
        max_lr_letters = [row["letter"] for row in choice_rows if row["learning_rate"] == max_lr]
        max_param_high_lr_letters = sorted(set(max_param_letters) & set(max_lr_letters))
        winner_choice = next(
            row for row in choice_rows if row["letter"] == answer["correct_letter"]
        )
        ranking = sorted(
            [
                {"letter": row["letter"], "metric_value": row["metric_value"]}
                for row in choice_rows
                if isinstance(row["metric_value"], (int, float))
            ],
            key=lambda item: item["metric_value"],
        )
        row = {
            "n": base["n_by_id"][qid],
            "original_n": base["old_n_by_id"].get(qid),
            "question_id": qid,
            "question_run_id": question.get("question_run_id"),
            "family": question["family"],
            "dataset_id": question["dataset_id"],
            "question_type": question.get("question_type"),
            "varying_axes": question.get("varying_axes", []),
            "selection_metric": question.get("selection_metric"),
            "correct_letter": answer["correct_letter"],
            "correct_candidate_id": choice_candidate_id(question, answer["correct_letter"]),
            "gap": answer.get("gap"),
            "win_rate": answer.get("win_rate"),
            "metric_values": metric_values,
            "true_ranking": ranking,
            "choices": choice_rows,
            "max_param_letters": max_param_letters,
            "max_param_correct": answer["correct_letter"] in max_param_letters,
            "highest_lr_letters": max_lr_letters,
            "highest_lr_correct": answer["correct_letter"] in max_lr_letters,
            "optimizer_families": sorted(
                {row["optimizer_type"] for row in choice_rows if row.get("optimizer_type")}
            ),
            "winner_optimizer_family": winner_choice.get("optimizer_type"),
            "winner_learning_rate": winner_choice.get("learning_rate"),
            "winner_param_est": winner_choice["model_shape"].get("trainable_params_est"),
            "capacity_optimizer_interaction": {
                "max_param_letters": max_param_letters,
                "highest_lr_letters": max_lr_letters,
                "max_param_and_highest_lr_letters": max_param_high_lr_letters,
                "correct_is_max_param_and_highest_lr": answer["correct_letter"]
                in max_param_high_lr_letters,
                "winner_optimizer_family": winner_choice.get("optimizer_type"),
                "winner_learning_rate": winner_choice.get("learning_rate"),
                "winner_param_est": winner_choice["model_shape"].get("trainable_params_est"),
            },
            "heuristic_hits": {
                "max_param": answer["correct_letter"] in max_param_letters,
                "highest_lr": answer["correct_letter"] in max_lr_letters,
                "max_param_and_highest_lr": answer["correct_letter"]
                in max_param_high_lr_letters,
            },
            "candidate_paths": path_by_letter,
            "raw_question": question,
        }
        rows.append(row)
        qmeta[qid] = row
    return rows, qmeta


def normalize_prediction(
    *,
    row: dict[str, Any],
    question: dict[str, Any],
    qmeta: dict[str, Any],
    model: str,
    setting: str,
    source_path: Path,
    run_id: str,
    protocol: str,
    evidence_depth: str,
    lesson_key: str = "lesson",
    post_feedback_key: str = "post_feedback_note",
) -> dict[str, Any] | None:
    qid = row.get("question_id")
    if qid not in qmeta:
        return None
    predicted_letter = row.get("predicted_letter") or row.get("answer_before_feedback")
    predicted_candidate_id = row.get("predicted_candidate_id") or choice_candidate_id(
        question, predicted_letter
    )
    correct_letter = qmeta[qid]["correct_letter"]
    correct_candidate_id = qmeta[qid]["correct_candidate_id"]
    return {
        "model": model,
        "setting": setting,
        "run_id": run_id,
        "protocol": protocol,
        "evidence_depth": evidence_depth,
        "source_path": rel(source_path),
        "n": qmeta[qid]["n"],
        "original_n": qmeta[qid]["original_n"],
        "question_id": qid,
        "family": qmeta[qid]["family"],
        "predicted_letter": predicted_letter,
        "predicted_candidate_id": predicted_candidate_id,
        "correct_letter": correct_letter,
        "correct_candidate_id": correct_candidate_id,
        "is_correct": predicted_letter == correct_letter
        if row.get("is_correct") is None
        else bool(row.get("is_correct")),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "lesson": row.get(lesson_key),
        "post_feedback_note": row.get(post_feedback_key),
        "choice_mean_metrics": row.get("choice_mean_metrics"),
    }


def load_prediction_rows(base: dict[str, Any], qmeta: dict[str, Any]) -> list[dict[str, Any]]:
    q_by_id = base["q_by_id"]
    predictions: list[dict[str, Any]] = []

    def add(row: dict[str, Any], **kwargs: Any) -> None:
        qid = row.get("question_id")
        if qid not in q_by_id:
            return
        normalized = normalize_prediction(row=row, question=q_by_id[qid], qmeta=qmeta, **kwargs)
        if normalized:
            predictions.append(normalized)

    per_question_dir = ARCHIVE65 / "per_question_blind_gpt54_gpt55"
    for model, filename in [
        ("gpt-5.4", "gpt-54_rows.json"),
        ("gpt-5.5", "gpt-55_rows.json"),
    ]:
        path = per_question_dir / filename
        if path.exists():
            for row in load_json(path):
                add(
                    row,
                    model=model,
                    setting="single_question_blind",
                    source_path=path,
                    run_id=f"{model}:fresh_per_question",
                    protocol="one fresh subagent per question",
                    evidence_depth="letter_candidate_confidence_reason",
                )

    path = QUIZ60 / "gpt56_sol" / "single_question_blind" / "score.json"
    if path.exists():
        for row in load_json(path).get("records", []):
            add(
                row,
                model="gpt-5.6-sol",
                setting="single_question_blind",
                source_path=path,
                run_id="gpt-5.6-sol:fresh_per_question",
                protocol="one fresh subagent per question",
                evidence_depth="letter_candidate_confidence_reason",
            )

    fullset_gpt54_dir = ARCHIVE65 / "gpt54_blind_10"
    for path in sorted(fullset_gpt54_dir.glob("agent_*_scored.json")):
        payload = load_json(path)
        agent = payload.get("agent_label") or path.stem.replace("agent_", "").replace("_scored", "")
        for row in payload.get("predictions", []):
            add(
                row,
                model="gpt-5.4",
                setting="full_set_blind",
                source_path=path,
                run_id=f"agent_{agent}",
                protocol="one context sees all 60 questions, no feedback",
                evidence_depth="letter_candidate_confidence_reason",
            )

    for agent in ["A", "B", "C"]:
        path = ARCHIVE65 / f"replicate_blind_agent_{agent}.json"
        if path.exists():
            payload = load_json(path)
            rows = payload if isinstance(payload, list) else payload.get("predictions", [])
            for index, row in enumerate(rows, start=1):
                row = {**row, "n": index}
                add(
                    row,
                    model="gpt-5.5",
                    setting="full_set_blind",
                    source_path=path,
                    run_id=f"agent_{agent}",
                    protocol="one context sees all questions, no feedback",
                    evidence_depth="letter_candidate_reason",
                )

    fullset_gpt56_dir = QUIZ60 / "gpt56_sol" / "blind"
    for path in sorted(fullset_gpt56_dir.glob("agent_*_scored.json")):
        agent = path.stem.replace("agent_", "").replace("_scored", "")
        for row in load_json(path).get("predictions", []):
            add(
                row,
                model="gpt-5.6-sol",
                setting="full_set_blind",
                source_path=path,
                run_id=f"agent_{agent}",
                protocol="one context sees all 60 questions, no feedback",
                evidence_depth="letter_candidate_confidence_reason",
            )

    grouped_gpt54_dir = QUIZ60 / "gpt54" / "grouped_10"
    for path in sorted(grouped_gpt54_dir.glob("group_*/session.json")):
        group = path.parent.name
        for row in load_json(path).get("records", []):
            add(
                row,
                model="gpt-5.4",
                setting="grouped_10_feedback",
                source_path=path,
                run_id=group,
                protocol="six reset 10-question blocks, prediction before feedback",
                evidence_depth="letter_candidate_confidence_reason_optional_lesson",
            )

    path = QUIZ60 / "report_sources" / "sequential_group_results.json"
    if path.exists():
        for group in load_json(path):
            for row in group.get("results", []):
                qid = row.get("question_id")
                clean_n = qmeta[qid]["n"] if qid in qmeta else row.get("n")
                clean_group = ((int(clean_n) - 1) // 10) + 1
                add(
                    row,
                    model="gpt-5.5",
                    setting="grouped_10_feedback",
                    source_path=path,
                    run_id=f"clean_group_{clean_group}",
                    protocol="clean 60-question grouped view reconstructed from historical grouped trace",
                    evidence_depth="letter_confidence_only",
                )

    grouped_gpt56_dir = QUIZ60 / "gpt56_sol" / "grouped_10"
    for path in sorted(grouped_gpt56_dir.glob("group_*/agent_scored.json")):
        group = path.parent.name
        for row in load_json(path).get("records", []):
            add(
                row,
                model="gpt-5.6-sol",
                setting="grouped_10_feedback",
                source_path=path,
                run_id=group,
                protocol="six reset 10-question blocks, prediction before feedback",
                evidence_depth="letter_candidate_confidence_reason_post_feedback",
                lesson_key="post_feedback_note",
            )

    fair_dir = QUIZ60 / "fair_sequential_v2"
    for model_dir, model in [
        ("gpt54", "gpt-5.4"),
        ("gpt55", "gpt-5.5"),
        ("gpt56_sol", "gpt-5.6-sol"),
    ]:
        for path in sorted((fair_dir / model_dir).glob("run_*/session.json")):
            run = path.parent.name
            for row in load_json(path).get("records", []):
                add(
                    row,
                    model=model,
                    setting="full_sequential_feedback_fair",
                    source_path=path,
                    run_id=run,
                    protocol="one fresh agent turn, prediction before feedback, full 60 questions",
                    evidence_depth="letter_candidate_confidence_reason_lesson_metrics",
                )

    return sorted(
        predictions,
        key=lambda item: (item["model"], item["setting"], item["run_id"], item["n"]),
    )


def build_memory_features(
    base: dict[str, Any], qmeta: dict[str, Any], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["setting"] == "full_sequential_feedback_fair":
            by_run[(prediction["model"], prediction["setting"], prediction["run_id"])].append(
                prediction
            )

    rows: list[dict[str, Any]] = []
    for (model, setting, run_id), records in sorted(by_run.items()):
        seen: dict[str, dict[str, Any]] = {}
        for record in sorted(records, key=lambda item: item["n"]):
            question = qmeta[record["question_id"]]
            choice_paths = question["candidate_paths"]
            current_paths = [choice_paths[choice["letter"]] for choice in question["choices"]]
            known_paths = [path for path in current_paths if path in seen]
            correct_path = choice_paths[question["correct_letter"]]
            known_count = len(known_paths)
            recencies = {
                letter: record["n"] - seen[path]["last_seen_n"]
                for letter, path in choice_paths.items()
                if path in seen
            }
            known_metrics = {
                letter: seen[path]["last_metric"]
                for letter, path in choice_paths.items()
                if path in seen and seen[path].get("last_metric") is not None
            }
            memory_oracle_letter = None
            if len(known_metrics) == len(question["choices"]):
                memory_oracle_letter = min(known_metrics, key=known_metrics.get)
            partial_best_known_letter = None
            if known_metrics:
                partial_best_known_letter = min(known_metrics, key=known_metrics.get)
            winner_seen = correct_path in seen
            winner_recency = record["n"] - seen[correct_path]["last_seen_n"] if winner_seen else None
            row = {
                "model": model,
                "setting": setting,
                "run_id": run_id,
                "n": record["n"],
                "question_id": record["question_id"],
                "family": record["family"],
                "is_correct": record["is_correct"],
                "predicted_letter": record["predicted_letter"],
                "correct_letter": record["correct_letter"],
                "known_count": known_count,
                "some_seen": known_count > 0,
                "winner_seen": winner_seen,
                "all_seen": known_count == len(current_paths),
                "any_choice_seen_in_last8": any(value <= 8 for value in recencies.values()),
                "winner_seen_in_last8": winner_recency is not None and winner_recency <= 8,
                "winner_seen_ever_not_last8": winner_recency is not None and winner_recency > 8,
                "min_any_recency": min(recencies.values()) if recencies else None,
                "winner_recency": winner_recency,
                "known_letters": sorted(known_metrics),
                "memory_oracle_letter": memory_oracle_letter,
                "partial_best_known_letter": partial_best_known_letter,
                "predicted_matches_memory_oracle": record["predicted_letter"]
                == memory_oracle_letter
                if memory_oracle_letter
                else None,
                "predicted_matches_partial_best_known": record["predicted_letter"]
                == partial_best_known_letter
                if partial_best_known_letter
                else None,
                "source_path": record["source_path"],
            }
            rows.append(row)

            metrics = record.get("choice_mean_metrics") or question["metric_values"]
            for letter, path in choice_paths.items():
                seen[path] = {
                    "last_seen_n": record["n"],
                    "last_question_id": record["question_id"],
                    "last_metric": metrics.get(letter) if isinstance(metrics, dict) else None,
                    "last_letter": letter,
                    "last_was_correct": letter == question["correct_letter"],
                }
    return rows


def aggregate_bool(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    yes = [row for row in rows if row.get(key) is True]
    no = [row for row in rows if row.get(key) is False]
    return {
        "true": ratio(sum(1 for row in yes if row["is_correct"]), len(yes)),
        "false": ratio(sum(1 for row in no if row["is_correct"]), len(no)),
    }


def build_summary(
    qrows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence: dict[str, dict[str, Any]] = {}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction["model"], prediction["setting"])].append(prediction)
    for (model, setting), rows in sorted(grouped.items()):
        runs = sorted({row["run_id"] for row in rows})
        evidence[f"{model}::{setting}"] = {
            "model": model,
            "setting": setting,
            "records": len(rows),
            "runs": len(runs),
            "questions_covered": len({row["question_id"] for row in rows}),
            "correct": sum(1 for row in rows if row["is_correct"]),
            "accuracy": sum(1 for row in rows if row["is_correct"]) / len(rows)
            if rows
            else None,
            "has_reason": sum(1 for row in rows if row.get("reason")),
            "has_lesson": sum(1 for row in rows if row.get("lesson")),
            "has_confidence": sum(1 for row in rows if row.get("confidence") is not None),
            "evidence_depths": sorted({row["evidence_depth"] for row in rows}),
            "source_paths": sorted({row["source_path"] for row in rows})[:12],
        }

    memory_by_model: dict[str, Any] = {}
    for model in sorted({row["model"] for row in memory_rows}):
        rows = [row for row in memory_rows if row["model"] == model]
        memory_by_model[model] = {
            "none_seen": ratio(
                sum(1 for row in rows if row["known_count"] == 0 and row["is_correct"]),
                sum(1 for row in rows if row["known_count"] == 0),
            ),
            "some_seen": ratio(
                sum(1 for row in rows if row["known_count"] > 0 and row["is_correct"]),
                sum(1 for row in rows if row["known_count"] > 0),
            ),
            "winner_seen": ratio(
                sum(1 for row in rows if row["winner_seen"] and row["is_correct"]),
                sum(1 for row in rows if row["winner_seen"]),
            ),
            "all_seen": ratio(
                sum(1 for row in rows if row["all_seen"] and row["is_correct"]),
                sum(1 for row in rows if row["all_seen"]),
            ),
            "winner_seen_in_last8": aggregate_bool(rows, "winner_seen_in_last8"),
            "winner_seen_ever_not_last8": aggregate_bool(rows, "winner_seen_ever_not_last8"),
            "known_count": {
                str(count): ratio(
                    sum(1 for row in rows if row["known_count"] == count and row["is_correct"]),
                    sum(1 for row in rows if row["known_count"] == count),
                )
                for count in range(4)
            },
        }

    return {
        "question_count": len(qrows),
        "choice_refs": sum(len(row["choices"]) for row in qrows),
        "unique_candidate_paths": len(
            {choice["candidate_path"] for row in qrows for choice in row["choices"]}
        ),
        "families": Counter(row["family"] for row in qrows),
        "question_types": Counter(row["question_type"] for row in qrows),
        "heuristics": {
            "max_param_correct": ratio(sum(1 for row in qrows if row["max_param_correct"]), len(qrows)),
            "highest_lr_correct": ratio(
                sum(1 for row in qrows if row["highest_lr_correct"]), len(qrows)
            ),
            "gap_buckets": Counter(
                ">=0.25"
                if row["gap"] >= 0.25
                else "0.1-0.25"
                if row["gap"] >= 0.1
                else "0.05-0.1"
                for row in qrows
            ),
        },
        "evidence_matrix": evidence,
        "context_decomposition": memory_by_model,
    }


def majority(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    votes = Counter(row["predicted_letter"] for row in rows if row.get("predicted_letter"))
    if not votes:
        return None
    letter, count = votes.most_common(1)[0]
    return {
        "letter": letter,
        "votes": dict(votes),
        "top_votes": count,
        "total_votes": sum(votes.values()),
    }


def case_tags(question: dict[str, Any], predictions: list[dict[str, Any]], memory_rows: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    if question["family"] == "univariate_regression":
        tags.add("univariate_hard_family")
    if question["max_param_correct"]:
        tags.add("capacity_shortcut")
    else:
        tags.add("capacity_trap")
    if not question["highest_lr_correct"]:
        tags.add("high_lr_trap")
    high_conf_wrong = [
        row
        for row in predictions
        if row.get("confidence") is not None and row["confidence"] >= 0.75 and not row["is_correct"]
    ]
    if high_conf_wrong:
        tags.add("high_confidence_error")
    for model in ["gpt-5.4", "gpt-5.5", "gpt-5.6-sol"]:
        full = [
            row
            for row in predictions
            if row["model"] == model and row["setting"] == "full_set_blind"
        ]
        seq = [
            row
            for row in predictions
            if row["model"] == model and row["setting"] == "full_sequential_feedback_fair"
        ]
        full_maj = majority(full)
        seq_maj = majority(seq)
        if full_maj and full_maj["letter"] != question["correct_letter"] and seq_maj and seq_maj["letter"] == question["correct_letter"]:
            tags.add("feedback_recovery")
        if full_maj and full_maj["letter"] != question["correct_letter"] and full_maj["top_votes"] >= max(3, full_maj["total_votes"] - 1):
            tags.add("blind_consensus_wrong")
    if any(row["all_seen"] and row["is_correct"] for row in memory_rows):
        tags.add("exact_memory_hit")
    if any(row["winner_seen_ever_not_last8"] and row["is_correct"] for row in memory_rows):
        tags.add("implicit_long_context_candidate_memory")
    if any(row["winner_seen_in_last8"] and row["is_correct"] for row in memory_rows):
        tags.add("visible_recent_lesson_memory")
    return sorted(tags)


def build_case_atlas(
    qrows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    memory_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_q[prediction["question_id"]].append(prediction)
    for row in memory_rows:
        memory_by_q[row["question_id"]].append(row)

    cases = []
    for question in qrows:
        qid = question["question_id"]
        q_predictions = sorted(
            predictions_by_q[qid],
            key=lambda item: (item["setting"], item["model"], item["run_id"]),
        )
        q_memory = sorted(memory_by_q[qid], key=lambda item: (item["model"], item["run_id"]))
        tags = case_tags(question, q_predictions, q_memory)
        by_setting: dict[str, Any] = {}
        for setting in sorted({row["setting"] for row in q_predictions}):
            by_model = {}
            for model in sorted({row["model"] for row in q_predictions if row["setting"] == setting}):
                rows = [
                    row
                    for row in q_predictions
                    if row["setting"] == setting and row["model"] == model
                ]
                by_model[model] = {
                    "records": rows,
                    "majority": majority(rows),
                    "accuracy": sum(1 for row in rows if row["is_correct"]) / len(rows)
                    if rows
                    else None,
                }
            by_setting[setting] = by_model
        cases.append(
            {
                "n": question["n"],
                "original_n": question["original_n"],
                "question_id": qid,
                "family": question["family"],
                "dataset_id": question["dataset_id"],
                "question_type": question["question_type"],
                "selection_metric": question["selection_metric"],
                "gap": question["gap"],
                "win_rate": question["win_rate"],
                "correct_letter": question["correct_letter"],
                "correct_candidate_id": question["correct_candidate_id"],
                "true_ranking": question["true_ranking"],
                "choices": question["choices"],
                "heuristics": {
                    "max_param_letters": question["max_param_letters"],
                    "max_param_correct": question["max_param_correct"],
                    "highest_lr_letters": question["highest_lr_letters"],
                    "highest_lr_correct": question["highest_lr_correct"],
                    "optimizer_families": question["optimizer_families"],
                    "winner_optimizer_family": question["winner_optimizer_family"],
                    "winner_learning_rate": question["winner_learning_rate"],
                    "winner_param_est": question["winner_param_est"],
                    "capacity_optimizer_interaction": question[
                        "capacity_optimizer_interaction"
                    ],
                    "heuristic_hits": question["heuristic_hits"],
                },
                "tags": tags,
                "predictions_by_setting": by_setting,
                "memory_records": q_memory,
                "source_provenance": {
                    "question_source": "artifacts/quiz_attempt_60/questions_sanitized.json",
                    "answer_source": "artifacts/quiz_attempt_60/answer_key.json",
                    "feedback_source": "artifacts/quiz_attempt_60/learning_feedback_key.json",
                    "prediction_sources": sorted(
                        {row["source_path"] for row in q_predictions}
                    ),
                },
                "raw_question": question["raw_question"],
            }
        )

    representatives = {}
    for tag in [
        "high_confidence_error",
        "blind_consensus_wrong",
        "feedback_recovery",
        "exact_memory_hit",
        "capacity_shortcut",
        "capacity_trap",
        "high_lr_trap",
        "univariate_hard_family",
        "implicit_long_context_candidate_memory",
    ]:
        tagged = [case for case in cases if tag in case["tags"]]
        if tagged:
            representatives[tag] = {
                "count": len(tagged),
                "question_ids": [case["question_id"] for case in tagged[:8]],
                "ns": [case["n"] for case in tagged[:8]],
            }

    return {
        "schema_version": 1,
        "source": {
            "questions": rel(QUIZ60 / "questions_sanitized.json"),
            "answer_key": rel(QUIZ60 / "answer_key.json"),
            "feedback": rel(QUIZ60 / "learning_feedback_key.json"),
        },
        "cases": cases,
        "representatives": representatives,
        "subagent_review_protocol": {
            "packet_count": 60,
            "recommended_reviewers": {
                "per_question_case_reviewers": 60,
                "cross_case_pattern_reviewers": 20,
                "statistics_reproducibility_reviewers": 10,
                "literature_writing_reviewers": 10,
            },
            "packet_schema": [
                "question",
                "answer",
                "heuristics",
                "predictions_by_setting",
                "memory_records",
                "mechanism_tags",
                "source_provenance",
            ],
            "reviewer_output_schema": [
                "mechanism_labels",
                "evidence_quotes_or_fields",
                "counterexamples",
                "writeup_readiness",
            ],
        },
    }


def build_matrix(qrows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_q[prediction["question_id"]].append(prediction)
    rows = []
    for question in qrows:
        preds = predictions_by_q[question["question_id"]]
        rows.append(
            {
                "n": question["n"],
                "original_n": question["original_n"],
                "question_id": question["question_id"],
                "family": question["family"],
                "dataset_id": question["dataset_id"],
                "question_type": question["question_type"],
                "selection_metric": question["selection_metric"],
                "gap": question["gap"],
                "win_rate": question["win_rate"],
                "correct_letter": question["correct_letter"],
                "correct_candidate_id": question["correct_candidate_id"],
                "metric_values": question["metric_values"],
                "true_ranking": question["true_ranking"],
                "choices": question["choices"],
                "prediction_count": len(preds),
                "prediction_correct_count": sum(1 for row in preds if row["is_correct"]),
                "predictions": preds,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    qrows, qmeta = build_question_features(base)
    predictions = load_prediction_rows(base, qmeta)
    memory_rows = build_memory_features(base, qmeta, predictions)
    matrix_rows = build_matrix(qrows, predictions)
    summary = build_summary(qrows, predictions, memory_rows)
    atlas = build_case_atlas(qrows, predictions, memory_rows)

    write_jsonl(OUT / "per_question_matrix.jsonl", matrix_rows)
    write_csv(
        OUT / "per_question_matrix.csv",
        matrix_rows,
        [
            "n",
            "original_n",
            "question_id",
            "family",
            "dataset_id",
            "selection_metric",
            "gap",
            "win_rate",
            "correct_letter",
            "correct_candidate_id",
            "metric_values",
            "prediction_count",
            "prediction_correct_count",
        ],
    )
    write_jsonl(OUT / "memory_features.jsonl", memory_rows)
    write_csv(
        OUT / "memory_features.csv",
        memory_rows,
        [
            "model",
            "run_id",
            "n",
            "question_id",
            "family",
            "is_correct",
            "known_count",
            "winner_seen",
            "all_seen",
            "winner_seen_in_last8",
            "winner_seen_ever_not_last8",
            "min_any_recency",
            "winner_recency",
            "memory_oracle_letter",
            "predicted_matches_memory_oracle",
        ],
    )
    write_jsonl(OUT / "heuristic_features.jsonl", qrows)
    write_csv(
        OUT / "heuristic_features.csv",
        qrows,
        [
            "n",
            "question_id",
            "family",
            "gap",
            "correct_letter",
            "max_param_letters",
            "max_param_correct",
            "highest_lr_letters",
            "highest_lr_correct",
            "optimizer_families",
            "winner_optimizer_family",
            "winner_learning_rate",
            "winner_param_est",
            "capacity_optimizer_interaction",
            "heuristic_hits",
        ],
    )
    write_json(OUT / "case_atlas.json", atlas)
    write_json(OUT / "summary.json", summary)
    js_payload = {
        "summary": summary,
        "case_atlas": atlas,
    }
    (OUT / "context_analysis.js").write_text(
        "window.AIQ_CONTEXT_ANALYSIS = "
        + json.dumps(js_payload, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": rel(OUT),
                "questions": len(qrows),
                "predictions": len(predictions),
                "memory_rows": len(memory_rows),
                "unique_candidate_paths": summary["unique_candidate_paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
