from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QUIZ60 = ROOT / "artifacts" / "quiz_attempt_60"
TRACE = QUIZ60 / "report_sources"
OUT = QUIZ60 / "readme_case_assets"
CURVE_OUT = OUT / "curves"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def by_key(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row[key]: row for row in rows}


def compact_model(model: dict[str, Any]) -> str:
    if model.get("type") == "transformer_lm":
        return (
            f"transformer d={model.get('d_model')}, "
            f"layers={model.get('num_layers')}, heads={model.get('num_heads')}, "
            f"d_ff={model.get('d_ff')}"
        )
    if model.get("type") == "mlp":
        return (
            f"MLP depth={model.get('depth')}, width={model.get('width')}, "
            f"residual={model.get('residual')}, act={','.join(map(str, model.get('activations', [])))}"
        )
    return ", ".join(f"{k}={v}" for k, v in model.items() if k != "type")


def compact_optimizer(opt: dict[str, Any]) -> str:
    parts = [str(opt.get("type", "optimizer"))]
    if "lr" in opt:
        parts.append(f"lr={opt['lr']}")
    if "weight_decay" in opt:
        parts.append(f"wd={opt['weight_decay']}")
    if "momentum" in opt:
        parts.append(f"momentum={opt['momentum']}")
    return ", ".join(parts)


def dataset_summary(q: dict[str, Any]) -> str:
    params = q.get("dataset_params", {})
    if q["family"] == "bigram_lm":
        return (
            f"vocab={params.get('vocab_size')}, context={params.get('context_length')}, "
            f"train={params.get('train_size')}, test={params.get('test_size')}, "
            f"layout={params.get('layout')}"
        )
    if "expression" in params:
        return f"expr: {params['expression']}"
    return ", ".join(f"{k}={v}" for k, v in list(params.items())[:6])


def svg_curve(question: dict[str, Any], answer: dict[str, Any], out_path: Path) -> None:
    choices = answer["choices"]
    series = []
    for ch in choices:
        curve_file = ROOT / "data" / ch["candidate_path"] / "results" / "curves.npz"
        if not curve_file.exists():
            continue
        data = np.load(curve_file)
        samples = np.asarray(data["samples"], dtype=float)
        curves = np.asarray(data["curves"], dtype=float)
        mean = np.nanmean(curves, axis=0)
        finite = np.isfinite(samples) & np.isfinite(mean) & (mean > 0)
        samples = samples[finite]
        mean = mean[finite]
        if samples.size:
            series.append(
                {
                    "letter": ch["letter"],
                    "candidate_id": ch["candidate_id"],
                    "samples": samples.tolist(),
                    "values": mean.tolist(),
                }
            )

    width, height = 760, 320
    margin = {"left": 64, "right": 24, "top": 28, "bottom": 46}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    colors = {"A": "#2f6f9f", "B": "#b35f2a", "C": "#3f8f5e"}

    if not series:
        if out_path.is_file():
            return
        out_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="180">'
            '<text x="20" y="90" font-family="monospace" font-size="16">No curve data found</text></svg>',
            encoding="utf-8",
        )
        return

    all_x = [x for s in series for x in s["samples"]]
    all_y = [y for s in series for y in s["values"]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if math.isclose(min_y, max_y):
        max_y = min_y + 1
    pad_y = (max_y - min_y) * 0.08
    min_y = max(min_y - pad_y, 1e-12)
    max_y += pad_y

    def sx(x: float) -> float:
        return margin["left"] + (x - min_x) / (max_x - min_x or 1) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + (max_y - y) / (max_y - min_y) * plot_h

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin["left"]}" y="18" font-family="Arial, sans-serif" font-size="13" fill="#344054">'
        f'{html.escape(question["question_id"])} · lower {html.escape(answer["metric"])} wins</text>',
    ]

    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y_val = min_y + (max_y - min_y) * frac
        yy = sy(y_val)
        lines.append(
            f'<line x1="{margin["left"]}" y1="{yy:.1f}" x2="{width-margin["right"]}" y2="{yy:.1f}" '
            'stroke="#e6e8ef" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin["left"]-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="10" fill="#667085">{y_val:.3g}</text>'
        )
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        x_val = min_x + (max_x - min_x) * frac
        xx = sx(x_val)
        lines.append(
            f'<line x1="{xx:.1f}" y1="{margin["top"]}" x2="{xx:.1f}" y2="{height-margin["bottom"]}" '
            'stroke="#f1f3f7" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{height-18}" text-anchor="middle" '
            f'font-family="monospace" font-size="10" fill="#667085">{int(x_val)}</text>'
        )

    lines.append(
        f'<line x1="{margin["left"]}" y1="{height-margin["bottom"]}" x2="{width-margin["right"]}" '
        f'y2="{height-margin["bottom"]}" stroke="#98a2b3" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" '
        f'y2="{height-margin["bottom"]}" stroke="#98a2b3" stroke-width="1"/>'
    )

    for s in series:
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["samples"], s["values"]))
        winner = "true" if s["letter"] == answer["correct_letter"] else "false"
        width_px = 3 if winner == "true" else 2
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors.get(s["letter"], "#555")}" '
            f'stroke-width="{width_px}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        end_x = sx(s["samples"][-1])
        end_y = sy(s["values"][-1])
        lines.append(
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" fill="{colors.get(s["letter"], "#555")}"/>'
        )
        lines.append(
            f'<text x="{min(end_x + 8, width - 80):.1f}" y="{end_y + 4:.1f}" '
            f'font-family="monospace" font-size="11" fill="{colors.get(s["letter"], "#555")}">'
            f'{s["letter"]} {s["values"][-1]:.3g}</text>'
        )

    lx = margin["left"]
    ly = height - 4
    for i, s in enumerate(series):
        x0 = lx + i * 160
        label = f'{s["letter"]} {s["candidate_id"]}'
        if s["letter"] == answer["correct_letter"]:
            label += " winner"
        lines.append(f'<rect x="{x0}" y="{ly-12}" width="18" height="3" fill="{colors.get(s["letter"], "#555")}"/>')
        lines.append(
            f'<text x="{x0+24}" y="{ly-8}" font-family="monospace" font-size="10" fill="#344054">'
            f'{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def normalize_trace(
    row: dict[str, Any],
    label: str,
    *,
    raw_response: dict[str, Any] | None = None,
    source_path: str,
    record_note: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "n": row.get("n") or row.get("question_number") or row.get("original_n"),
        "question_id": row.get("question_id"),
        "predicted_letter": row.get("predicted_letter") or row.get("answer_before_feedback"),
        "predicted_candidate_id": row.get("predicted_candidate_id"),
        "correct_letter": row.get("correct_letter") or row.get("feedback_viewed_after_prediction", {}).get("correct_letter"),
        "correct_candidate_id": row.get("correct_candidate_id")
        or row.get("feedback_viewed_after_prediction", {}).get("correct_candidate_id"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "lesson": row.get("lesson"),
        "is_correct": row.get("is_correct") if "is_correct" in row else row.get("correct"),
        "cumulative_accuracy": row.get("cumulative_accuracy"),
        "protocol_confirmation": row.get("protocol_confirmation"),
        "raw_response": raw_response if raw_response is not None else row,
        "source_path": source_path,
        "record_note": record_note,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CURVE_OUT.mkdir(parents=True, exist_ok=True)

    questions = load_json(QUIZ60 / "questions_sanitized.json")
    answers = load_json(QUIZ60 / "answer_key.json")
    feedback = load_json(TRACE / "learning_feedback_key.json")
    q_by_clean_n = {i + 1: q for i, q in enumerate(questions)}
    a_by_id = by_key(answers, "question_id")
    f_by_id = by_key(feedback, "question_id")

    strict_spotcheck_path = TRACE / "per_question_blind_gpt55_strict_prompt_spotcheck.json"
    full_a_path = TRACE / "replicate_blind_agent_A.json"
    group_path = TRACE / "sequential_group_results.json"
    cli_b_path = TRACE / "audited_sequential_B_session.json"

    strict_spotcheck = load_json(strict_spotcheck_path)
    full_a_document = load_json(full_a_path)
    group_results = load_json(group_path)
    cli_b_document = load_json(cli_b_path)

    strict_rows = {int(row["original_n"]): row for row in strict_spotcheck["results"]}
    full_a = full_a_document["predictions"]
    cli_b = cli_b_document["records"]

    def clean_n(original_n: int) -> int:
        if 41 <= original_n <= 45:
            raise ValueError(f"original n={original_n} is excluded from the 60-question scope")
        return original_n if original_n <= 40 else original_n - 5

    def row_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {
            int(r.get("n") or r.get("question_number")): r
            for r in rows
            if r.get("n") or r.get("question_number")
        }

    rows = {
        "full_a": row_by_n([{**r, "n": i + 1} for i, r in enumerate(full_a)]),
        "cli_b": row_by_n(cli_b),
    }
    group_rows: dict[int, dict[str, Any]] = {}
    for group in group_results:
        for record in group["results"]:
            group_rows[int(record["n"])] = record

    def enrich_for_display(row: dict[str, Any], original_n: int) -> dict[str, Any]:
        q = q_by_clean_n[clean_n(original_n)]
        answer = a_by_id[q["question_id"]]
        predicted_letter = row.get("predicted_letter") or row.get("answer_before_feedback")
        predicted_choice = next(
            (choice for choice in q["choices"] if choice["letter"] == predicted_letter),
            None,
        )
        correct_choice = next(
            choice for choice in q["choices"] if choice["letter"] == answer["correct_letter"]
        )
        return {
            **row,
            "n": original_n,
            "predicted_candidate_id": row.get("predicted_candidate_id")
            or (predicted_choice or {}).get("candidate_id"),
            "correct_letter": answer["correct_letter"],
            "correct_candidate_id": correct_choice["candidate_id"],
            "is_correct": predicted_letter == answer["correct_letter"],
        }

    needed_ns = sorted(set([1, 33, 39, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]))
    question_payload: dict[str, Any] = {}
    for n in needed_ns:
        cn = clean_n(n)
        q = q_by_clean_n[cn]
        a = a_by_id[q["question_id"]]
        f = f_by_id[q["question_id"]]
        svg_name = f'orig_{n:02d}_clean_{cn:02d}_{q["question_id"]}.svg'
        svg_curve(q, a, CURVE_OUT / svg_name)
        choices = []
        metrics = f.get("choice_mean_metrics", {})
        for ch in q["choices"]:
            choices.append(
                {
                    "letter": ch["letter"],
                    "candidate_id": ch["candidate_id"],
                    "model": compact_model(ch.get("model", {})),
                    "optimizer": compact_optimizer(ch.get("optimizer", {})),
                    "budget": ch.get("budget", {}),
                    "metric_value": metrics.get(ch["letter"]),
                    "is_correct": ch["letter"] == a["correct_letter"],
                }
            )
        question_payload[str(n)] = {
            "original_n": n,
            "clean_n": cn,
            "question_id": q["question_id"],
            "family": q["family"],
            "dataset_id": q["dataset_id"],
            "metric": q["selection_metric"],
            "question_type": q["question_type"],
            "varying_axes": q.get("varying_axes", []),
            "dataset_summary": dataset_summary(q),
            "correct_letter": a["correct_letter"],
            "gap": a.get("gap"),
            "win_rate": a.get("win_rate"),
            "curve_svg": f"artifacts/quiz_attempt_60/readme_case_assets/curves/{svg_name}",
            "choices": choices,
            "raw_question": q,
        }

    def single_raw_response(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": row["question_id"],
            "original_n": row["original_n"],
            "predicted_letter": row["predicted_letter"],
            "predicted_candidate_id": row["predicted_candidate_id"],
            "confidence": row["confidence"],
            "reason": row["reason"],
            "protocol_confirmation": row["protocol_confirmation"],
        }

    group_six = next(group for group in group_results if int(group["group"]) == 6)

    studies = [
        {
            "id": "case-single-blind",
            "title": "单题盲答：fresh subagent 的原始最终回答",
            "subtitle": "三个严格协议 spot check；每个 agent 只收到当前一道 sanitized question。",
            "question_ns": [33, 39, 60],
            "trace_title": "GPT-5.5 high · 单题盲答原始回答",
            "traces": [
                normalize_trace(
                    strict_rows[n],
                    f"n={n} fresh blind",
                    raw_response=single_raw_response(strict_rows[n]),
                    source_path=str(strict_spotcheck_path.relative_to(ROOT)),
                    record_note="raw_response 是该 subagent 实际提交的完整 output_text；正确答案与 is_correct 由父 agent 在返回后追加。",
                )
                for n in [33, 39, 60]
            ],
        },
        {
            "id": "case-fullset-blind",
            "title": "整套盲答：一次看到 60 题后的横向比较",
            "subtitle": "同一个 Agent A 一次性输出全套答案；下方保留单题原始 prediction，并可展开完整历史文件。",
            "question_ns": [1, 33, 60],
            "trace_title": "GPT-5.5 high · full-set blind Agent A 原始回答",
            "traces": [
                normalize_trace(
                    enrich_for_display(rows["full_a"][n], n),
                    f"n={n} Agent A",
                    raw_response=full_a[n - 1],
                    source_path=str(full_a_path.relative_to(ROOT)),
                    record_note="历史原始 run 含 65 题；主报告排除原始 41-45 后按 60 道三选题计分。此处不改写原始 prediction。",
                )
                for n in [1, 33, 60]
            ],
            "full_raw_label": "完整历史输出：Agent A 的 65 条 predictions",
            "full_raw_note": "这是磁盘中的原始历史文件。报告主口径过滤 41-45，但这里为保证可追溯性保留全部记录。",
            "full_raw_source": str(full_a_path.relative_to(ROOT)),
            "full_raw_response": full_a_document,
        },
        {
            "id": "case-group-feedback",
            "title": "10题组内反馈：Group 6 的连续短程学习",
            "subtitle": "这个历史 artifact 只保存先答选项、置信度和反馈后的正确性；原文件没有 reason/lesson。",
            "question_ns": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            "trace_title": "GPT-5.5 high · Group 6 原始记录",
            "traces": [
                normalize_trace(
                    enrich_for_display(group_rows[n], n),
                    f"n={n} group 6",
                    raw_response=group_rows[n],
                    source_path=str(group_path.relative_to(ROOT)),
                    record_note="原始 artifact 未保存 reason、candidate_id 或 lesson；这些字段无法从现有汇总中恢复，因此不补写。",
                )
                for n in range(51, 61)
            ],
            "full_raw_label": "完整原始记录：Group 6 的 10 条 results",
            "full_raw_note": "以下内容与历史 JSON 一致；缺失字段就是当时没有落盘。",
            "full_raw_source": str(group_path.relative_to(ROOT)),
            "full_raw_response": group_six,
        },
        {
            "id": "case-sequential-feedback",
            "title": "全程顺序反馈：记录器强制版 n=55-57",
            "subtitle": "CLI 先保存预测与 reason，再返回真实指标；每条原始记录同时保留反馈后的 lesson。",
            "question_ns": [55, 56, 57],
            "trace_title": "GPT-5.5 high · CLI-audited sequential B 原始记录",
            "traces": [
                normalize_trace(
                    rows["cli_b"][n],
                    f"n={n} CLI B",
                    raw_response=rows["cli_b"][n],
                    source_path=str(cli_b_path.relative_to(ROOT)),
                    record_note="这是记录器落盘的完整逐题 record：包含先答、理由、置信度、当前题反馈、lesson 与累计正确率。",
                )
                for n in [55, 56, 57]
            ],
            "full_raw_label": "完整历史输出：CLI B 的 65 条 records",
            "full_raw_note": "可展开审计整个顺序反馈 run；主报告统计时再过滤原始 41-45。",
            "full_raw_source": str(cli_b_path.relative_to(ROOT)),
            "full_raw_response": cli_b_document,
        },
    ]

    data = {"questions": question_payload, "studies": studies}
    (OUT / "case_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    js = "window.README_CASE_STUDIES = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"
    (OUT / "case_data.js").write_text(js, encoding="utf-8")


if __name__ == "__main__":
    main()
