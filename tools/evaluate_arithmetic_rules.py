from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUIZ_DIR = ROOT / "artifacts" / "quiz_attempt_60"
DEFAULT_ORDER_METRICS = (
    ROOT / "artifacts" / "order_parameter_analysis" / "candidate_metrics.csv"
)
DEFAULT_ANCHOR_METRICS = (
    ROOT / "artifacts" / "order_parameter_analysis" / "anchored_progress_metrics.csv"
)
DEFAULT_OUT = ROOT / "artifacts" / "order_parameter_analysis"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mlp_params(model: dict[str, Any]) -> int:
    input_dim = int(model.get("input_dim", 1))
    width = int(model["width"])
    depth = int(model["depth"])
    layer_norm = [bool(v) for v in model.get("layer_norm", [])]
    total = input_dim * width + width
    for i in range(depth):
        total += width * width + width
        if i < len(layer_norm) and layer_norm[i]:
            total += 2 * width
    total += width + 1
    return total


def transformer_params(model: dict[str, Any]) -> int:
    vocab = int(model["vocab_size"])
    context = int(model["context_length"])
    d_model = int(model.get("d_model", model.get("embed_dim")))
    d_ff = int(model.get("d_ff", model.get("ff_dim")))
    layers = int(model["num_layers"])
    embeddings = vocab * d_model + context * d_model
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 9 * d_model + d_ff
    head = d_model * vocab + vocab
    return embeddings + layers * per_layer + head


def count_params(model: dict[str, Any]) -> int:
    if model["type"] == "mlp":
        return mlp_params(model)
    if model["type"] == "transformer_lm":
        return transformer_params(model)
    raise ValueError(model["type"])


def minmax(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def choice_features(choice: dict[str, Any]) -> dict[str, float | str | bool]:
    model = choice["model"]
    opt = choice["optimizer"]
    loss = choice["loss"]
    params = count_params(model)
    lr = float(opt["lr"])
    depth = float(model.get("depth", model.get("num_layers", 0)))
    width = float(model.get("width", model.get("d_model", 0)))
    layer_norm = list(model.get("layer_norm", []))
    return {
        "candidate_id": choice["candidate_id"],
        "params": float(params),
        "log_params": math.log(params),
        "lr": lr,
        "log_lr": math.log(lr),
        "log10_lr": math.log10(lr),
        "optimizer": opt["type"],
        "momentum": float(opt.get("momentum", 0.0)),
        "weight_decay": float(opt.get("weight_decay", 0.0)),
        "depth": depth,
        "width": width,
        "residual": bool(model.get("residual", False)),
        "layer_norm_frac": sum(1 for value in layer_norm if value)
        / max(len(layer_norm), 1),
        "loss_id": loss.get("loss_id", ""),
        "loss_lambda": float(loss.get("lambda", 0.0)),
    }


def hand_score(question: dict[str, Any], choice: dict[str, Any]) -> float:
    feats = [choice_features(c) for c in question["choices"]]
    current = choice_features(choice)
    cap_z = dict(zip([f["candidate_id"] for f in feats], minmax([float(f["log_params"]) for f in feats])))
    lr_z = dict(zip([f["candidate_id"] for f in feats], minmax([float(f["log10_lr"]) for f in feats])))
    depth_z = dict(zip([f["candidate_id"] for f in feats], minmax([float(f["depth"]) for f in feats])))

    opt = str(current["optimizer"])
    opt_bonus = {
        "SGD": -0.25,
        "Adagrad": 0.05,
        "Adam": 0.22,
        "AdamW": 0.22,
        "RMSprop": 0.18,
    }[opt]
    if opt == "SGD" and float(current["momentum"]) >= 0.9:
        opt_bonus += 0.22

    lr = float(current["lr"])
    lr_penalty = 0.0
    if lr <= 1e-4:
        lr_penalty += 0.10
    if lr >= 1e-2 and opt in {"Adam", "AdamW", "RMSprop"}:
        lr_penalty += 0.05

    wd = float(current["weight_decay"])
    wd_penalty = 0.03 * math.log10(1 + wd * 10000)

    loss_penalty = 0.0
    loss_id = str(current["loss_id"])
    if loss_id.endswith("_l1"):
        loss_penalty += 0.08
    if loss_id.endswith("_l2"):
        loss_penalty += 0.04
    loss_penalty += 2.0 * float(current["loss_lambda"])

    residual_bonus = 0.05 if bool(current["residual"]) else 0.0
    norm_bonus = 0.04 * float(current["layer_norm_frac"])

    family = question["family"]
    if family == "bigram_lm":
        cap_weight = 0.72
        lr_weight = 0.33
    elif family == "multivariate_regression":
        cap_weight = 0.62
        lr_weight = 0.30
    else:
        cap_weight = 0.72
        lr_weight = 0.35

    return (
        cap_weight * cap_z[str(current["candidate_id"])]
        + lr_weight * lr_z[str(current["candidate_id"])]
        + 0.07 * depth_z[str(current["candidate_id"])]
        + opt_bonus
        + residual_bonus
        + norm_bonus
        - lr_penalty
        - wd_penalty
        - loss_penalty
    )


def eta_score(question: dict[str, Any], choice: dict[str, Any]) -> float:
    f = choice_features(choice)
    opt_mult = {
        "SGD": 0.35 + 0.65 * float(f["momentum"]),
        "Adagrad": 0.9,
        "Adam": 1.15,
        "AdamW": 1.15,
        "RMSprop": 1.25,
    }[str(f["optimizer"])]
    return math.log(float(f["lr"]) * opt_mult)


def params_score(question: dict[str, Any], choice: dict[str, Any]) -> float:
    return float(choice_features(choice)["log_params"])


def simple_product_score(question: dict[str, Any], choice: dict[str, Any]) -> float:
    f = choice_features(choice)
    return float(f["log_params"]) + 0.7 * eta_score(question, choice)


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return [load_json(item) for item in sorted(path.glob("*.json"))]
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")
    return payload


def load_answers(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Answer key must contain a JSON list: {path}")
    return {row["question_id"]: row["correct_letter"] for row in payload}


def metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in (
                "log_params",
                "lr",
                "weight_decay",
                "momentum",
                "depth",
                "width",
                "final_metric",
                "log_improvement",
                "early_slope",
                "anchored_progress_auc",
            ):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def design_row(row: dict[str, Any]) -> list[float]:
    opt = row["optimizer"]
    log_params = (
        float(row["log_params"])
        if "log_params" in row and row["log_params"] != ""
        else math.log(float(row["num_params"]))
    )
    return [
        1.0,
        math.log(float(row["lr"])),
        log_params,
        float(row.get("momentum", 0.0)),
        math.log10(1.0 + float(row.get("weight_decay", 0.0)) * 10000.0),
        1.0 if opt == "Adam" else 0.0,
        1.0 if opt == "AdamW" else 0.0,
        1.0 if opt == "RMSprop" else 0.0,
        1.0 if opt == "SGD" else 0.0,
    ]


def design_row_rich(row: dict[str, Any]) -> list[float]:
    base = design_row(row)
    depth = float(row.get("depth", 0.0) or 0.0)
    width = float(row.get("width", 0.0) or 0.0)
    residual_raw = row.get("residual", False)
    residual = (
        1.0
        if residual_raw is True or str(residual_raw).lower() == "true"
        else 0.0
    )
    log_width = math.log(width) if width > 0 else 0.0
    return [
        *base,
        depth,
        log_width,
        residual,
        base[1] * base[2],
        residual * depth,
    ]


def fit_linear(rows: list[dict[str, Any]], target: str, *, rich: bool = False) -> dict[str, np.ndarray]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row[target]) if target in row else float("nan")
        if np.isfinite(value) and (target != "final_metric" or value > 0):
            by_family[row["family"]].append(row)
    models = {}
    design = design_row_rich if rich else design_row
    for family, group in by_family.items():
        x = np.asarray([design(r) for r in group], dtype=float)
        if target == "final_metric":
            y = -np.log(np.asarray([float(r[target]) for r in group], dtype=float))
        else:
            y = np.asarray([float(r[target]) for r in group], dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        models[family] = coef
    return models


def fit_score(
    models: dict[str, np.ndarray],
    question: dict[str, Any],
    choice: dict[str, Any],
    *,
    rich: bool = False,
) -> float:
    f = choice_features(choice)
    row = {
        "optimizer": f["optimizer"],
        "lr": f["lr"],
        "log_params": f["log_params"],
        "momentum": f["momentum"],
        "weight_decay": f["weight_decay"],
        "depth": f["depth"],
        "width": f["width"],
        "residual": f["residual"],
    }
    coef = models[question["family"]]
    design = design_row_rich if rich else design_row
    return float(np.asarray(design(row), dtype=float) @ coef)


def evaluate(
    questions: list[dict[str, Any]],
    answers: dict[str, str],
    name: str,
    score_fn,
) -> dict[str, Any]:
    rows = []
    for q in questions:
        scores = [(c["letter"], score_fn(q, c)) for c in q["choices"]]
        pred = max(scores, key=lambda x: (x[1], -ord(x[0])))[0]
        correct = answers[q["question_id"]]
        rows.append(
            {
                "rule": name,
                "question_id": q["question_id"],
                "family": q["family"],
                "question_type": q["question_type"],
                "pred": pred,
                "correct": correct,
                "ok": pred == correct,
                "scores": dict(scores),
            }
        )
    return summarize(name, rows)


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def acc(group: list[dict[str, Any]]) -> float:
        return sum(1 for r in group if r["ok"]) / len(group) if group else float("nan")

    by_family = {k: acc(v) for k, v in groupby(rows, "family").items()}
    by_type = {k: acc(v) for k, v in groupby(rows, "question_type").items()}
    return {
        "rule": name,
        "n": len(rows),
        "correct": sum(1 for r in rows if r["ok"]),
        "accuracy": acc(rows),
        "by_family": by_family,
        "by_type": by_type,
        "mistakes": [r for r in rows if not r["ok"]],
    }


def groupby(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUIZ_DIR / "questions_sanitized.json",
        help="Question JSON list or a directory containing one JSON file per question.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=DEFAULT_QUIZ_DIR / "answer_key.json",
    )
    parser.add_argument("--order-metrics", type=Path, default=DEFAULT_ORDER_METRICS)
    parser.add_argument("--anchor-metrics", type=Path, default=DEFAULT_ANCHOR_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.questions.exists():
        print(f"Questions not found: {args.questions}", file=sys.stderr)
        return 1
    if not args.answer_key.is_file():
        print(f"Answer key not found: {args.answer_key}", file=sys.stderr)
        return 1

    questions = load_questions(args.questions)
    answers = load_answers(args.answer_key)
    if not questions:
        print("No questions found.", file=sys.stderr)
        return 1
    missing_answers = [q["question_id"] for q in questions if q["question_id"] not in answers]
    if missing_answers:
        print(f"Missing answers for {len(missing_answers)} question(s).", file=sys.stderr)
        return 1

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate(questions, answers, "max_params", params_score),
        evaluate(questions, answers, "max_eta", eta_score),
        evaluate(questions, answers, "log_params_plus_eta", simple_product_score),
        evaluate(questions, answers, "hand_balanced", hand_score),
    ]

    metric_data = metric_rows(args.order_metrics)
    anchor_data = metric_rows(args.anchor_metrics)
    if metric_data:
        final_models = fit_linear(metric_data, "final_metric")
        final_models_rich = fit_linear(metric_data, "final_metric", rich=True)
        improvement_models = fit_linear(metric_data, "log_improvement")
        results.append(
            evaluate(
                questions,
                answers,
                "fit_candidate_final_metric",
                lambda q, c: fit_score(final_models, q, c),
            )
        )
        results.append(
            evaluate(
                questions,
                answers,
                "fit_candidate_final_metric_rich",
                lambda q, c: fit_score(final_models_rich, q, c, rich=True),
            )
        )
        results.append(
            evaluate(
                questions,
                answers,
                "fit_candidate_log_improvement",
                lambda q, c: fit_score(improvement_models, q, c),
            )
        )
    if anchor_data:
        auc_models = fit_linear(anchor_data, "anchored_progress_auc")
        results.append(
            evaluate(
                questions,
                answers,
                "fit_anchored_progress_auc",
                lambda q, c: fit_score(auc_models, q, c),
            )
        )

    compact = [{k: v for k, v in r.items() if k != "mistakes"} for r in results]
    (out / "arithmetic_rule_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Arithmetic Rule Evaluation",
        "",
        f"Questions: {len(questions)}",
        "",
        "| rule | correct | accuracy | bigram | multivariate | univariate | mixed | architecture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in compact:
        bf = r["by_family"]
        bt = r["by_type"]
        lines.append(
            "| {rule} | {correct}/{n} | {acc:.3f} | {bigram:.3f} | {multi:.3f} | {uni:.3f} | {mixed:.3f} | {arch:.3f} |".format(
                rule=r["rule"],
                correct=r["correct"],
                n=r["n"],
                acc=r["accuracy"],
                bigram=bf.get("bigram_lm", float("nan")),
                multi=bf.get("multivariate_regression", float("nan")),
                uni=bf.get("univariate_regression", float("nan")),
                mixed=bt.get("mixed", float("nan")),
                arch=bt.get("architecture_only", float("nan")),
            )
        )
    lines.extend(["", "## Mistakes"])
    for r in results:
        lines.append("")
        lines.append(f"### {r['rule']}")
        if not r["mistakes"]:
            lines.append("")
            lines.append("No mistakes.")
            continue
        for m in r["mistakes"][:20]:
            lines.append(
                f"- {m['question_id']} ({m['family']}, {m['question_type']}): pred {m['pred']} vs {m['correct']}"
            )
        if len(r["mistakes"]) > 20:
            lines.append(f"- ... {len(r['mistakes']) - 20} more")
    (out / "arithmetic_rule_results.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
