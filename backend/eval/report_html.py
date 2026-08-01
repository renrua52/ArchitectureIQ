"""Generate a self-contained HTML report for one eval run folder.

Reads the per-run folder written by ``batch_eval --run-dir``:

    artifacts/eval_runs/{run_dir}/
    ├── run.json
    ├── set_{label}_questions.jsonl
    ├── posthoc_audit_{model}_parsed.jsonl        (optional, from question audit)
    └── results/{model}/responses_{label}.jsonl

and writes ``report.html`` showing per-question cards (references with losses,
options, GT winner), every model's answer + full stored reasoning (content and
reasoning_content), plus the post-hoc audit table. Single-file HTML (inline
CSS/JS), no external dependencies.

Usage:
    .venv/bin/python -m backend.eval.report_html --run-dir run_claude_terra_50_20260801
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

OUT_ROOT = Path("artifacts/eval_runs")

SCHEMA_VERSION = "0.1"


def esc(s) -> str:
    return html.escape(str(s or ""))


def setting_line(cfg: dict) -> str:
    m = cfg.get("model", {})
    o = cfg.get("optimizer", {})
    depth = m.get("depth") or m.get("num_layers")
    width = m.get("width") or m.get("d_model")
    parts = [f"type={m.get('type')}", f"d={depth}", f"w={width}"]
    if "residual" in m:
        parts.append(f"resid={m['residual']}")
    if "activations" in m:
        parts.append(f"act={','.join(str(a)[:4] for a in m['activations'])}")
    parts += [f"opt={o.get('type')}", f"lr={o.get('lr')}", f"wd={o.get('weight_decay')}"]
    if "momentum" in o and o.get("momentum") is not None:
        parts.append(f"mom={o['momentum']}")
    b = cfg.get("budget", {})
    parts.append(f"steps={b.get('training_steps')}×bs={b.get('batch_size')}")
    return " · ".join(parts)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def build(run_dir: Path) -> str:
    root = OUT_ROOT / run_dir
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    manifests = run.get("models", [])
    sets = {}
    for m in manifests:
        sets.setdefault(m["label"], {"models": []})
        sets[m["label"]]["models"].append(m)
    question_files = sorted(root.glob("set_*_questions.jsonl"))
    q_by_label = {}
    for f in question_files:
        label = f.name[len("set_") : -len("_questions.jsonl")]
        q_by_label[label] = [json.loads(l) for l in f.open(encoding="utf-8")]

    model_dirs = sorted((root / "results").iterdir()) if (root / "results").exists() else []
    resp_by = {}
    for md in model_dirs:
        for rf in sorted(md.glob("responses_*.jsonl")):
            label = rf.name[len("responses_") : -len(".jsonl")]
            resp_by[(md.name, label)] = [json.loads(l) for l in rf.open(encoding="utf-8")]

    audit = load_jsonl(root / "posthoc_audit_claude_parsed.jsonl")
    audit_by = {a["question_id"]: a for a in audit}

    # ---- accuracy table ----
    rows_acc = []
    for (model, label), rows in sorted(resp_by.items()):
        n = len(rows)
        ok = sum(1 for r in rows if r["is_correct"])
        rows_acc.append((model, label, ok, n, ok / n if n else 0))
    acc_table = "\n".join(
        f"<tr><td>{esc(m)}</td><td>{esc(l)}</td><td>{ok}/{n}</td>"
        f"<td>{pct:.1%}</td></tr>"
        for m, l, ok, n, pct in rows_acc)

    # ---- question cards ----
    cards = []
    for label, qs in q_by_label.items():
        cards.append(f'<h2>题集 {esc(label)}（{len(qs)} 题）</h2>')
        for q in qs:
            qid = q.get("question_id") or q.get("task") or "?"
            st = q.get("statistics") or {}
            ratio = st.get("ratio")
            win = st.get("win_rate")
            correct = q.get("correct_letter")
            if correct is None and q.get("type") == "two_choice_loss_compare":
                correct = q.get("answer")
            cards.append(f'<div class="card" id="{esc(qid)}">')
            cards.append(f'<div class="qhead">'
                         f'<span class="qid">{esc(qid)}</span>'
                         f'<span class="tag">{esc(q.get("type") or q.get("task"))}</span>'
                         f'<span class="tag">{esc(q.get("problem_id"))}</span>'
                         f'<span class="tag">ratio={esc(ratio)}</span>'
                         f'<span class="tag">win_rate={esc(win)}</span>'
                         f'<span class="tag gt">GT={esc(correct)}</span></div>')

            # references
            refs = q.get("references") or q.get("demos") or []
            if refs:
                cards.append('<details open><summary>参考 settings（带 loss）</summary><ul>')
                for i, r in enumerate(refs, 1):
                    loss = r.get("loss")
                    cards.append(f'<li><b>Ref{i}</b> loss={esc(round(loss, 4) if isinstance(loss, (int, float)) else loss)} — '
                                 f'<code>{esc(setting_line(r.get("setting") or {}))}</code></li>')
                cards.append('</ul></details>')

            # options
            opts = q.get("options") or []
            if opts:
                cards.append('<details open><summary>选项</summary><ul>')
                for o in opts:
                    win_mark = " <span class='win'>← WINNER</span>" if o.get("letter") == correct else ""
                    cards.append(f'<li><b>{esc(o.get("letter"))}</b> <code>{esc(setting_line(o.get("setting") or {}))}</code>{win_mark}</li>')
                cards.append('</ul></details>')
            elif q.get("type") == "two_choice_loss_compare":
                cards.append('<details open><summary>选项</summary><ul>')
                for L in "AB":
                    t = (q.get("target") or {}).get(L) or {}
                    win_mark = " <span class='win'>← WINNER</span>" if L == correct else ""
                    cards.append(f'<li><b>{L}</b> <code>{esc(setting_line(t.get("setting") or {}))}</code>{win_mark}</li>')
                cards.append('</ul></details>')

            # prompt
            if q.get("prompt"):
                cards.append(f'<details><summary>完整 prompt</summary><pre>{esc(q["prompt"])}</pre></details>')

            # per-model answers + reasoning
            cards.append('<div class="answers">')
            for (model, label2), rows in sorted(resp_by.items()):
                if label2 != label:
                    continue
                row = next((r for r in rows if (r.get("question_id") or r.get("task")) == qid), None)
                if row is None:
                    continue
                ok = row.get("is_correct")
                badge = "ok" if ok else "bad"
                content = row.get("content") or row.get("raw_response") or ""
                reasoning = row.get("reasoning_content") or ""
                cards.append(f'<div class="ans {badge}">')
                cards.append(f'<span class="model">{esc(model)}</span> '
                             f'<b>答 {esc(row.get("answer"))}</b> vs GT {esc(row.get("correct"))} '
                             f'<span class="badge">{("正确" if ok else "错误")}</span>')
                if reasoning:
                    cards.append(f'<details><summary>reasoning_content（{len(reasoning)} 字符）</summary>'
                                 f'<pre>{esc(reasoning)}</pre></details>')
                if content:
                    cards.append(f'<details><summary>回答内容（{len(content)} 字符）</summary>'
                                 f'<pre>{esc(content)}</pre></details>')
                cards.append('</div>')
            cards.append('</div>')

            # post-hoc audit
            a = audit_by.get(qid)
            if a:
                cards.append(f'<div class="audit"><b>Claude 事后评审</b> — 盲答 {esc(a.get("answer"))} '
                             f'(GT {esc(a.get("gt"))}) · 可答性 {esc(a.get("answerable_1to5"))}/5 · '
                             f'{esc(a.get("fairness"))}<br>{esc(a.get("hardness_reason"))}</div>')
            cards.append('</div>')

    # ---- audit summary ----
    audit_rows = ""
    if audit:
        n_sb = sum(1 for a in audit if a["type"] == "select_best")
        ok_sb = sum(1 for a in audit if a["type"] == "select_best" and a.get("answer") == a.get("gt"))
        n_tc = sum(1 for a in audit if a["type"] == "two_choice")
        ok_tc = sum(1 for a in audit if a["type"] == "two_choice" and a.get("answer") == a.get("gt"))
        a5 = [a.get("answerable_1to5") or 0 for a in audit]
        audit_rows = (f'<tr><td>select_best</td><td>{n_sb}</td><td>{ok_sb}/{n_sb} ({ok_sb/n_sb:.0%})</td>'
                      f'<td>{sum(1 for a in audit if a["type"]=="select_best" and a.get("answerable_1to5",0)>=4)}/{n_sb}</td>'
                      f'<td>{sum(1 for a in audit if a["type"]=="select_best" and a.get("fairness")=="fair")}/{n_sb}</td></tr>'
                      f'<tr><td>two_choice</td><td>{n_tc}</td><td>{ok_tc}/{n_tc} ({ok_tc/n_tc:.0%})</td>'
                      f'<td>{sum(1 for a in audit if a["type"]=="two_choice" and a.get("answerable_1to5",0)>=4)}/{n_tc}</td>'
                      f'<td>{sum(1 for a in audit if a["type"]=="two_choice" and a.get("fairness")=="fair")}/{n_tc}</td></tr>')

    models_used = sorted({m for m, _ in resp_by})
    model_filter = "".join(f'<label class="mf"><input type="checkbox" class="fmodel" value="{esc(m)}" checked> {esc(m)}</label>'
                           for m in models_used)

    html_out = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>评测报告 {esc(run_dir)}</title>
<style>
 body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; background: #fafafa; color: #222; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; margin: 10px 0; }} th, td {{ border: 1px solid #ccc; padding: 4px 10px; font-size: 13px; }}
 .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin: 14px 0; }}
 .qhead {{ font-weight: 600; margin-bottom: 6px; }} .qid {{ font-family: monospace; }}
 .tag {{ background: #eee; border-radius: 10px; padding: 1px 8px; font-size: 12px; margin-left: 6px; }}
 .tag.gt {{ background: #d4edda; }} .win {{ color: #1a7f37; font-weight: 700; }}
 pre {{ background: #f6f8fa; border: 1px solid #eee; border-radius: 6px; padding: 10px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }}
 details {{ margin: 6px 0; }} summary {{ cursor: pointer; font-size: 13px; color: #444; }}
 .answers {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
 .ans {{ border: 1px solid #ccc; border-radius: 6px; padding: 6px 10px; flex: 1 1 300px; font-size: 13px; }}
 .ans.ok {{ border-color: #1a7f37; }} .ans.bad {{ border-color: #d73a49; }}
 .model {{ font-family: monospace; font-weight: 600; }}
 .badge {{ font-size: 12px; margin-left: 6px; }}
 .ok .badge {{ color: #1a7f37; }} .bad .badge {{ color: #d73a49; }}
 .audit {{ background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: 13px; }}
 .controls {{ position: sticky; top: 0; background: #fafafa; padding: 8px 0; border-bottom: 1px solid #ddd; }}
 .controls input[type=text] {{ width: 260px; padding: 4px 8px; }}
 .mf {{ margin-right: 12px; font-size: 13px; }}
 .card.hidden {{ display: none; }}
</style>
</head>
<body>
<h1>评测报告 · {esc(run_dir)}</h1>
<p>创建于 {esc(manifests[0].get("created_at", "") if manifests else "")} ·
模型: {esc(", ".join(models_used))} ·
题集: {esc(", ".join(sets))}</p>

<h2>准确率</h2>
<table><tr><th>模型</th><th>题集</th><th>正确/总数</th><th>准确率</th></tr>{acc_table}</table>

<h2>事后评审（claude-opus-5 盲审，不给 GT）</h2>
<table>
<tr><th>题型</th><th>题数</th><th>Claude 盲答正确</th><th>可答性≥4</th><th>判定 fair</th></tr>
{audit_rows or "<tr><td colspan=5>无评审数据</td></tr>"}
</table>
<p style="font-size:12px;color:#666">可答性 1-5：参考信息能否逻辑推出胜者（1=纯猜，5=可推理确定）。</p>

<div class="controls">
<input type="text" id="search" placeholder="搜索 question_id / problem / 关键词…">
<label><input type="checkbox" id="onlyWrong" checked> 只看错误</label>
<label><input type="checkbox" id="onlyAudit" > 只看有评审的题</label>
{model_filter}
</div>

{''.join(cards)}

<script>
const cards = [...document.querySelectorAll('.card')];
const search = document.getElementById('search');
const onlyWrong = document.getElementById('onlyWrong');
const onlyAudit = document.getElementById('onlyAudit');
const fm = [...document.querySelectorAll('.fmodel')];
function apply() {{
  const q = search.value.toLowerCase();
  const showWrong = onlyWrong.checked;
  const showAudit = onlyAudit.checked;
  const ms = new Set(fm.filter(x => x.checked).map(x => x.value));
  for (const c of cards) {{
    const ansDivs = [...c.querySelectorAll('.ans')];
    let visible = ansDivs.length === 0;
    for (const a of ansDivs) {{
      const m = a.querySelector('.model').textContent;
      if (!ms.has(m)) {{ a.style.display = 'none'; continue; }}
      a.style.display = '';
      if (a.classList.contains('bad')) visible = true;
      else if (a.classList.contains('ok') && !showWrong) visible = true;
    }}
    const text = c.textContent.toLowerCase();
    if (q && !text.includes(q)) visible = false;
    if (showAudit && !c.querySelector('.audit')) visible = false;
    if (!showWrong) {{
      const anyBad = [...c.querySelectorAll('.ans')].some(a => ms.has(a.querySelector('.model').textContent) && a.classList.contains('bad'));
      if (anyBad) visible = true;
    }}
    c.classList.toggle('hidden', !visible);
  }}
}}
search.addEventListener('input', apply);
onlyWrong.addEventListener('change', apply);
onlyAudit.addEventListener('change', apply);
fm.forEach(x => x.addEventListener('change', apply));
apply();
</script>
</body></html>"""
    return html_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    root = OUT_ROOT / args.run_dir
    out = build(args.run_dir)
    (root / "report.html").write_text(out, encoding="utf-8")
    print(f"wrote {root / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
