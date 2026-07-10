#!/usr/bin/env python3
"""Generate calibration-plus-ranking ArchitectureIQ questions.

Each ranking question uses a sliding window over a trained candidate set:

- first N candidates are calibration examples with their learning curves and final metric;
- next M candidates are target settings to rank from best to worst;
- the answer key is the target order by mean final metric, lower is better by default.
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))

from architecture_iq.prompts.formatters import (  # noqa: E402
    format_loss_nl,
    format_model_nl,
    format_optimizer_nl,
    format_training_schedule,
)
from architecture_iq.util import short_hash  # noqa: E402

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from common import candidate_metric, compact_json, read_json, write_json  # noqa: E402


@dataclass(frozen=True)
class CandidateArtifact:
    path: Path
    spec: dict[str, Any]
    summary: dict[str, Any]
    metric: str
    mean_metric: float
    std_metric: float | None


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _candidate_sort_key(candidate: CandidateArtifact) -> tuple[str, str]:
    return (candidate.path.parent.name, candidate.path.name)


def _load_candidates(candidate_set: Path, *, max_candidates: int | None) -> list[CandidateArtifact]:
    candidates: list[CandidateArtifact] = []
    for candidate_dir in sorted(p for p in candidate_set.iterdir() if p.is_dir()):
        spec_path = candidate_dir / "candidate_spec.json"
        summary_path = candidate_dir / "results" / "summary.json"
        curves_path = candidate_dir / "results" / "curves.npz"
        if not spec_path.is_file() or not summary_path.is_file() or not curves_path.is_file():
            continue
        spec = read_json(spec_path)
        summary = read_json(summary_path)
        if summary.get("excluded"):
            continue
        metric, mean_metric, std_metric = candidate_metric(summary)
        candidates.append(
            CandidateArtifact(
                path=candidate_dir.resolve(),
                spec=spec,
                summary=summary,
                metric=metric,
                mean_metric=mean_metric,
                std_metric=std_metric,
            )
        )

    candidates.sort(key=_candidate_sort_key)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def _load_curves(candidate: CandidateArtifact) -> tuple[np.ndarray, list[int]]:
    curves_path = candidate.path / "results" / "curves.npz"
    data = np.load(curves_path)
    curves = np.asarray(data["curves"], dtype=np.float64)
    if "samples" in data:
        samples = np.asarray(data["samples"], dtype=np.int64).tolist()
    else:
        budget = candidate.spec["budget"]
        batch_size = int(budget["batch_size"])
        total = int(budget["total_samples_seen"])
        steps = total // batch_size
        samples = [step * batch_size for step in range(1, steps + 1)]
    if curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] == 0:
        raise ValueError(f"Invalid empty curve array: {curves_path}")
    if not samples:
        raise ValueError(f"No sample positions found: {curves_path}")
    n_cols = min(curves.shape[1], len(samples))
    return curves[:, :n_cols], samples[:n_cols]


def _plot_curve(candidate: CandidateArtifact, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, samples = _load_curves(candidate)
    mean_curve = np.nanmean(curves, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    for row in curves:
        ax.plot(samples, row, color="#94a3b8", alpha=0.32, linewidth=1)
    ax.plot(samples, mean_curve, color="#0f172a", linewidth=2.3, label="mean across seeds")
    ax.scatter(
        [samples[-1]],
        [candidate.mean_metric],
        color="#dc2626",
        s=36,
        zorder=4,
        label=f"final mean {candidate.metric}",
    )
    ax.set_title(title)
    ax.set_xlabel("samples seen")
    ax.set_ylabel(candidate.metric)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _setting_lines(spec: dict[str, Any]) -> list[str]:
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
    return lines


def _setting_markdown(spec: dict[str, Any]) -> str:
    return "\n".join(_setting_lines(spec))


def _setting_html(spec: dict[str, Any]) -> str:
    escaped = [html.escape(line) for line in _setting_lines(spec)]
    chunks: list[str] = []
    for line in escaped:
        if not line:
            chunks.append("<br>")
        elif line in {"Training schedule", "Model", "Optimizer", "Loss"}:
            chunks.append(f"<div class=\"spec-heading\">{line}</div>")
        else:
            chunks.append(f"<div>{line}</div>")
    return "\n".join(chunks)


def _candidate_public_record(
    candidate: CandidateArtifact,
    *,
    label: str,
    include_metric: bool,
    curve_path: Path | None = None,
) -> dict[str, Any]:
    record = {
        "label": label,
        "candidate_id": candidate.spec["candidate_id"],
        "candidate_path": _repo_rel(candidate.path),
        "setting_markdown": _setting_markdown(candidate.spec),
        "setting_html": _setting_html(candidate.spec),
    }
    if include_metric:
        record["metric"] = candidate.metric
        record["mean_metric"] = candidate.mean_metric
        if candidate.std_metric is not None:
            record["std_metric"] = candidate.std_metric
    if curve_path is not None:
        record["curve_image"] = curve_path.as_posix()
    return record


def _prompt_for_question(question: dict[str, Any]) -> str:
    lines = [
        "# ArchitectureIQ Ranking Question",
        "",
        "You must rank the target settings from best to worst. Do not run experiments, "
        "do not inspect hidden result files, and do not use the answer key. Use only "
        "the calibration examples and target setting descriptions below.",
        "",
        f"Dataset: `{question['family']}/{question['dataset_id']}`",
        f"Metric: `{question['metric']}`; lower is better.",
        "",
        "## Calibration Examples",
        "",
        "These examples include the setting, the full learning-curve image, and the final mean metric.",
    ]
    for item in question["calibration"]:
        lines.extend(
            [
                "",
                f"### {item['label']} · {item['candidate_id']}",
                "",
                item["setting_markdown"],
                "",
                f"Final mean {item['metric']}: {item['mean_metric']:.8g}",
                f"![{item['label']} learning curve]({item['curve_image']})",
            ]
        )
    lines.extend(
        [
            "",
            "## Targets To Rank",
            "",
            "Return only the target labels in best-to-worst order, for example: "
            "`T3,T1,T5,T2,T4`.",
        ]
    )
    for item in question["targets"]:
        lines.extend(
            [
                "",
                f"### {item['label']} · {item['candidate_id']}",
                "",
                item["setting_markdown"],
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _question_from_window(
    *,
    candidates: list[CandidateArtifact],
    start: int,
    question_index: int,
    calibration_size: int,
    target_size: int,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    window_size = calibration_size + target_size
    window = [candidates[(start + offset) % len(candidates)] for offset in range(window_size)]
    return _question_from_groups(
        calibration=window[:calibration_size],
        targets=window[calibration_size:],
        question_index=question_index,
        out_dir=out_dir,
        salt={"start": start, "layout": "cyclic"},
    )


def _question_from_groups(
    *,
    calibration: list[CandidateArtifact],
    targets: list[CandidateArtifact],
    question_index: int,
    out_dir: Path,
    salt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    window = calibration + targets
    first = window[0].spec
    metric = window[0].metric

    body_for_hash = {
        **salt,
        "candidate_ids": [c.spec["candidate_id"] for c in window],
        "calibration_size": len(calibration),
        "target_size": len(targets),
    }
    question_id = f"rq_{question_index:02d}_{short_hash(body_for_hash)}"
    qdir = out_dir / "questions" / question_id
    curves_dir = qdir / "curves"

    calibration_records: list[dict[str, Any]] = []
    for i, candidate in enumerate(calibration, start=1):
        label = f"K{i}"
        curve_name = f"{label}_{candidate.spec['candidate_id']}.png"
        curve_path = curves_dir / curve_name
        _plot_curve(candidate, curve_path, f"{label} · {candidate.spec['candidate_id']}")
        calibration_records.append(
            _candidate_public_record(
                candidate,
                label=label,
                include_metric=True,
                curve_path=Path("curves") / curve_name,
            )
        )

    target_records = [
        _candidate_public_record(candidate, label=f"T{i}", include_metric=False)
        for i, candidate in enumerate(targets, start=1)
    ]
    answer_order = [
        record["label"]
        for _, record in sorted(
            zip(targets, target_records, strict=True),
            key=lambda pair: pair[0].mean_metric,
        )
    ]
    answer_key = {
        "question_id": question_id,
        "metric": metric,
        "lower_is_better": True,
        "true_order": answer_order,
        "targets": [
            {
                "label": record["label"],
                "candidate_id": candidate.spec["candidate_id"],
                "candidate_path": _repo_rel(candidate.path),
                "mean_metric": candidate.mean_metric,
                "std_metric": candidate.std_metric,
            }
            for candidate, record in zip(targets, target_records, strict=True)
        ],
    }
    question = {
        "schema_version": "ranking_v1",
        "question_id": question_id,
        "family": first["family"],
        "dataset_id": first["dataset_id"],
        "metric": metric,
        "lower_is_better": True,
        "calibration_size": len(calibration),
        "target_size": len(targets),
        "calibration": calibration_records,
        "targets": target_records,
        "answer_key_path": "answer_key.json",
    }
    return question, answer_key


def _html_template(ui_data: dict[str, Any]) -> str:
    payload = compact_json(ui_data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ArchitectureIQ Ranking Quiz</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #111827;
      --muted: #64748b;
      --line: #dbe3ee;
      --panel: #f8fafc;
      --accent: #2563eb;
      --good: #047857;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
      z-index: 5;
    }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .meta {{ margin-top: 5px; color: var(--muted); font-size: 14px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1.12fr) minmax(340px, 0.88fr);
      gap: 22px;
      padding: 22px 28px 44px;
    }}
    section {{ min-width: 0; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    h3 {{ font-size: 15px; margin: 0 0 8px; }}
    .question-tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .question-tabs button,
    .controls button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 8px 11px;
      font-weight: 650;
      cursor: pointer;
    }}
    .question-tabs button.active {{
      border-color: var(--accent);
      color: var(--accent);
      background: #eff6ff;
    }}
    .calibration-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      background: #fff;
      min-width: 0;
    }}
    .card img {{
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-top: 10px;
    }}
    .candidate-id {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .metric {{ color: var(--good); font-weight: 750; margin-top: 8px; }}
    .spec {{ font-size: 13px; line-height: 1.45; color: #1f2937; }}
    .spec-heading {{
      color: #0f172a;
      font-weight: 760;
      margin-top: 9px;
      margin-bottom: 2px;
    }}
    .rank-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .rank-card {{
      border: 1.5px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      cursor: grab;
    }}
    .rank-card.dragging {{ opacity: 0.45; }}
    .rank-card:active {{ cursor: grabbing; }}
    .rank-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .position {{
      color: var(--muted);
      font-weight: 700;
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      margin: 14px 0 12px;
      flex-wrap: wrap;
    }}
    .controls .primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .result {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 48px;
      background: #fff;
      font-size: 14px;
    }}
    .result.good {{ border-color: #86efac; background: #f0fdf4; }}
    .result.bad {{ border-color: #fecaca; background: #fef2f2; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; padding: 18px; }}
      .calibration-grid {{ grid-template-columns: 1fr; }}
      header {{ padding-left: 18px; padding-right: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ArchitectureIQ Ranking Quiz</h1>
    <div class="meta" id="header-meta"></div>
    <div class="question-tabs" id="tabs"></div>
  </header>
  <main>
    <section>
      <h2>Calibration</h2>
      <div class="calibration-grid" id="calibration"></div>
    </section>
    <section>
      <h2>Rank Targets</h2>
      <div class="meta">Drag from best expected final metric at the top to worst at the bottom.</div>
      <div class="controls">
        <button class="primary" id="check">Check</button>
        <button id="reset">Reset</button>
      </div>
      <ol class="rank-list" id="rank-list"></ol>
      <div class="result" id="result"></div>
    </section>
  </main>
  <script>
    const DATA = {payload};
    let currentIndex = 0;
    let originalOrders = {{}};

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function inversions(order, truth) {{
      const rank = Object.fromEntries(truth.map((label, idx) => [label, idx]));
      let count = 0;
      for (let i = 0; i < order.length; i++) {{
        for (let j = i + 1; j < order.length; j++) {{
          if (rank[order[i]] > rank[order[j]]) count += 1;
        }}
      }}
      return count;
    }}

    function renderTabs() {{
      const tabs = document.getElementById('tabs');
      tabs.innerHTML = '';
      DATA.questions.forEach((q, idx) => {{
        const button = document.createElement('button');
        button.textContent = `${{idx + 1}}`;
        button.className = idx === currentIndex ? 'active' : '';
        button.onclick = () => {{ currentIndex = idx; render(); }};
        tabs.appendChild(button);
      }});
    }}

    function renderCalibration(q) {{
      const root = document.getElementById('calibration');
      root.innerHTML = q.calibration.map(item => `
        <article class="card">
          <h3>${{escapeHtml(item.label)}} <span class="candidate-id">${{escapeHtml(item.candidate_id)}}</span></h3>
          <div class="spec">${{item.setting_html}}</div>
          <div class="metric">final mean ${{escapeHtml(item.metric)}}: ${{Number(item.mean_metric).toPrecision(5)}}</div>
          <img src="${{escapeHtml(q.base_path + '/' + item.curve_image)}}" alt="${{escapeHtml(item.label)}} learning curve">
        </article>
      `).join('');
    }}

    function targetCard(item, index) {{
      const li = document.createElement('li');
      li.className = 'rank-card';
      li.draggable = true;
      li.dataset.label = item.label;
      li.innerHTML = `
        <div class="rank-head">
          <h3>${{escapeHtml(item.label)}} <span class="candidate-id">${{escapeHtml(item.candidate_id)}}</span></h3>
          <span class="position">#${{index + 1}}</span>
        </div>
        <div class="spec">${{item.setting_html}}</div>
      `;
      li.addEventListener('dragstart', () => li.classList.add('dragging'));
      li.addEventListener('dragend', () => {{
        li.classList.remove('dragging');
        updatePositions();
      }});
      return li;
    }}

    function renderTargets(q) {{
      const list = document.getElementById('rank-list');
      const order = originalOrders[q.question_id] || q.targets.map(t => t.label);
      const byLabel = Object.fromEntries(q.targets.map(t => [t.label, t]));
      list.innerHTML = '';
      order.forEach((label, idx) => list.appendChild(targetCard(byLabel[label], idx)));
      list.ondragover = event => {{
        event.preventDefault();
        const after = getDragAfterElement(list, event.clientY);
        const dragging = document.querySelector('.dragging');
        if (!dragging) return;
        if (after == null) list.appendChild(dragging);
        else list.insertBefore(dragging, after);
      }};
    }}

    function getDragAfterElement(container, y) {{
      const elements = [...container.querySelectorAll('.rank-card:not(.dragging)')];
      return elements.reduce((closest, child) => {{
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) return {{ offset, element: child }};
        return closest;
      }}, {{ offset: Number.NEGATIVE_INFINITY, element: null }}).element;
    }}

    function currentOrder() {{
      return [...document.querySelectorAll('#rank-list .rank-card')].map(el => el.dataset.label);
    }}

    function updatePositions() {{
      [...document.querySelectorAll('#rank-list .rank-card')].forEach((el, idx) => {{
        el.querySelector('.position').textContent = `#${{idx + 1}}`;
      }});
    }}

    function render() {{
      const q = DATA.questions[currentIndex];
      document.getElementById('header-meta').textContent =
        `${{q.question_id}} · ${{q.family}}/${{q.dataset_id}} · metric: ${{q.metric}} (lower is better)`;
      renderTabs();
      renderCalibration(q);
      renderTargets(q);
      document.getElementById('result').className = 'result';
      document.getElementById('result').textContent = '';
    }}

    document.getElementById('check').onclick = () => {{
      const q = DATA.questions[currentIndex];
      const order = currentOrder();
      const inv = inversions(order, q.true_order);
      const maxInv = q.true_order.length * (q.true_order.length - 1) / 2;
      const result = document.getElementById('result');
      result.className = `result ${{inv === 0 ? 'good' : 'bad'}}`;
      result.innerHTML = `
        <strong>Inversions:</strong> ${{inv}} / ${{maxInv}}<br>
        <strong>Your order:</strong> <code>${{order.join(',')}}</code><br>
        <strong>True order:</strong> <code>${{q.true_order.join(',')}}</code>
      `;
    }};

    document.getElementById('reset').onclick = () => render();
    render();
  </script>
</body>
</html>
"""


def _ui_data(manifest: dict[str, Any], questions: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    ui_questions = []
    for question in questions:
        qid = question["question_id"]
        ui_questions.append(
            {
                **question,
                "base_path": f"questions/{qid}",
                "true_order": answers[qid]["true_order"],
            }
        )
    return {
        "schema_version": "ranking_ui_v1",
        "manifest": manifest,
        "questions": ui_questions,
    }


def _copy_sanitized_prompt_assets(question_dir: Path, eval_dir: Path) -> None:
    target_dir = eval_dir / question_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(question_dir / "prompt.md", target_dir / "prompt.md")
    curves_src = question_dir / "curves"
    curves_dst = target_dir / "curves"
    if curves_dst.exists():
        shutil.rmtree(curves_dst)
    shutil.copytree(curves_src, curves_dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_set", type=Path, help="Candidate set directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "ranking_questions",
        help="Output directory. A run subdirectory is created inside it unless the path is empty.",
    )
    parser.add_argument("--num-questions", type=int, default=12)
    parser.add_argument("--calibration-size", type=int, default=5)
    parser.add_argument("--target-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--layout",
        choices=["cyclic", "anchored"],
        default="cyclic",
        help=(
            "cyclic uses sliding windows; anchored fixes calibration examples and "
            "uses disjoint target chunks when possible."
        ),
    )
    parser.add_argument(
        "--allow-target-repeat",
        action="store_true",
        help="Allow anchored layout to wrap and repeat target candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=60,
        help="Use only the first N eligible candidates; use 0 for all.",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_set = args.candidate_set.resolve()
    if not candidate_set.is_dir():
        print(f"Candidate set not found: {candidate_set}", file=sys.stderr)
        return 1
    if args.calibration_size < 1 or args.target_size < 2:
        print("calibration-size must be >= 1 and target-size >= 2", file=sys.stderr)
        return 1
    if args.num_questions < 1:
        print("num-questions must be >= 1", file=sys.stderr)
        return 1
    if args.stride < 1:
        print("stride must be >= 1", file=sys.stderr)
        return 1
    if args.max_candidates < 0:
        print("max-candidates must be >= 0", file=sys.stderr)
        return 1
    try:
        candidate_set.relative_to(ROOT)
    except ValueError:
        print("Candidate set must be inside the repository.", file=sys.stderr)
        return 1

    max_candidates = None if args.max_candidates == 0 else args.max_candidates
    try:
        candidates = _load_candidates(candidate_set, max_candidates=max_candidates)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Could not load candidate set: {exc}", file=sys.stderr)
        return 1
    window_size = args.calibration_size + args.target_size
    if len(candidates) < window_size:
        print(
            f"Need at least {window_size} eligible candidates; found {len(candidates)}",
            file=sys.stderr,
        )
        return 1
    identities = {
        (candidate.spec["family"], candidate.spec["dataset_id"], candidate.metric)
        for candidate in candidates
    }
    if len(identities) != 1:
        print(
            "Eligible candidates must share one family, dataset, and metric.",
            file=sys.stderr,
        )
        return 1
    try:
        for candidate in candidates:
            _load_curves(candidate)
    except (KeyError, OSError, ValueError) as exc:
        print(f"Invalid candidate curve data: {exc}", file=sys.stderr)
        return 1

    set_manifest_path = candidate_set / "set.json"
    if not set_manifest_path.is_file():
        print(f"Missing candidate-set manifest: {set_manifest_path}", file=sys.stderr)
        return 1
    try:
        set_manifest = read_json(set_manifest_path)
    except (KeyError, ValueError) as exc:
        print(f"Invalid candidate-set manifest: {exc}", file=sys.stderr)
        return 1
    run_name = args.run_name or (
        "ranking_"
        + short_hash(
            {
                "candidate_set": _repo_rel(candidate_set),
                "n": len(candidates),
                "num_questions": args.num_questions,
                "calibration_size": args.calibration_size,
                "target_size": args.target_size,
                "stride": args.stride,
                "layout": args.layout,
                "allow_target_repeat": args.allow_target_repeat,
            }
        )
    )
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        print("run-name must be a single directory name", file=sys.stderr)
        return 1

    starts = [(i * args.stride) % len(candidates) for i in range(args.num_questions)]
    questions: list[dict[str, Any]] = []
    answer_by_id: dict[str, Any] = {}

    if args.layout == "anchored":
        target_pool = candidates[args.calibration_size :]
        max_disjoint = len(target_pool) // args.target_size
        if args.num_questions > max_disjoint and not args.allow_target_repeat:
            print(
                "Anchored layout without repeats can generate at most "
                f"{max_disjoint} questions from {len(candidates)} candidates "
                f"({args.calibration_size} calibration + "
                f"{args.target_size} targets/question).",
                file=sys.stderr,
            )
            return 1
        starts = [(i * args.target_size) % len(target_pool) for i in range(args.num_questions)]

    out_dir = args.output.resolve() / run_name
    if out_dir.exists():
        print(f"Output run already exists: {out_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True)

    for i, start in enumerate(starts, start=1):
        if args.layout == "cyclic":
            question, answer_key = _question_from_window(
                candidates=candidates,
                start=start,
                question_index=i,
                calibration_size=args.calibration_size,
                target_size=args.target_size,
                out_dir=out_dir,
            )
        else:
            target_pool = candidates[args.calibration_size :]
            targets = [
                target_pool[(start + offset) % len(target_pool)]
                for offset in range(args.target_size)
            ]
            question, answer_key = _question_from_groups(
                calibration=candidates[: args.calibration_size],
                targets=targets,
                question_index=i,
                out_dir=out_dir,
                salt={"start": start, "layout": "anchored"},
            )
        qdir = out_dir / "questions" / question["question_id"]
        write_json(qdir / "ranking_question.json", question)
        write_json(qdir / "answer_key.json", answer_key)
        (qdir / "prompt.md").write_text(_prompt_for_question(question), encoding="utf-8")
        questions.append(question)
        answer_by_id[question["question_id"]] = answer_key

    manifest = {
        "schema_version": "ranking_manifest_v1",
        "run_id": run_name,
        "candidate_set": _repo_rel(candidate_set),
        "dataset_id": set_manifest["dataset_id"],
        "family": set_manifest["family"],
        "metric": questions[0]["metric"],
        "lower_is_better": True,
        "eligible_candidates": len(candidates),
        "num_questions": len(questions),
        "layout": args.layout,
        "calibration_size": args.calibration_size,
        "target_size": args.target_size,
        "stride": args.stride,
        "window_size": window_size,
        "starts": starts,
        "question_ids": [q["question_id"] for q in questions],
        "question_paths": [f"questions/{q['question_id']}" for q in questions],
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

    ui_data = _ui_data(manifest, questions, answer_by_id)
    (out_dir / "index.html").write_text(_html_template(ui_data), encoding="utf-8")
    print(out_dir)
    print(f"Generated {len(questions)} ranking questions from {len(candidates)} candidates.")
    print(f"Human drag UI: {out_dir / 'index.html'}")
    print(f"LLM sanitized prompts: {llm_eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
