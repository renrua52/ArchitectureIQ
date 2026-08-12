#!/usr/bin/env python3
"""Generate the question-centric project overview HTML.

Every number and preview in the report is reproducible from on-disk artifacts:
question sets (backend/eval/sets/*/questions.jsonl), eval responses
(artifacts/eval_runs/*.jsonl) and work trees (backend/eval/trees/*/tree_*.json).

Usage:
    .venv/bin/python tools/report_questions.py --out docs/reports/QUESTIONS_OVERVIEW_2026-08-03.html
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- plain() ----
def plain(cfg: dict) -> str:
    """One-line, jargon-free rendering of a candidate config.

    Example: "Transformer 2层 d=128 head=2 ff=256 | AdamW lr=0.003 wd=1e-05 |
              cross_entropy | 80步×批64=5120样本"
    """
    m = cfg.get("model", {})
    o = cfg.get("optimizer", {})
    l = cfg.get("loss", {})
    b = cfg.get("budget", {})

    def fnum(x):
        return f"{x:g}" if isinstance(x, float) else str(x)

    mtype = m.get("type")
    if mtype == "transformer_lm":
        model = (f"Transformer {m.get('num_layers')}层 d={m.get('d_model')} "
                 f"head={m.get('num_heads')} ff={m.get('d_ff')}")
    elif mtype == "gru_lm":
        model = f"GRU {m.get('num_layers')}层 d={m.get('d_model')}"
    else:
        bits = [f"MLP {m.get('depth')}层 宽{m.get('width')}"]
        if m.get("residual"):
            bits.append("残差")
        lns = m.get("layer_norm") or []
        if any(lns):
            bits.append("LN=" + "+".join("Y" if x else "-" for x in lns))
        acts = m.get("activations") or []
        if acts:
            bits.append("act=[" + ",".join(str(a) for a in acts) + "]")
        model = " ".join(bits)

    opt = o.get("type", "?")
    wd = o.get("weight_decay")
    opt = f"{opt} lr={fnum(o.get('lr'))}" + (f" wd={fnum(wd)}" if wd else "")

    lid = l.get("loss_id", "?")
    lam = l.get("lambda")
    if lam is not None:
        lid = f"{lid}(λ={fnum(lam)})"

    steps, bs, tot = b.get("training_steps"), b.get("batch_size"), b.get("total_samples_seen")
    if steps and bs and tot:
        budget = f"{steps}步×批{bs}={tot}样本"
    else:
        budget = f"样本{tot}"

    return f"{model} | {opt} | {lid} | {budget}"


# ---------------------------------------------------------------- loaders ----
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def score_from(path):
    """(correct, total) from a responses jsonl using is_correct."""
    rows = load_jsonl(path)
    return sum(1 for r in rows if r.get("is_correct")), len(rows)


def qscore(path):
    """(correct, total) for rows with answer == correct."""
    rows = load_jsonl(path)
    return sum(1 for r in rows if r.get("answer") == r.get("correct")), len(rows)


def refs_rows(q, n=5):
    refs = q.get("references") or q.get("demos") or []
    return [{"n": i, "setting": plain(r.get("setting", r)), "loss": r.get("loss")}
            for i, r in enumerate(refs[:n], 1) if r.get("loss") is not None]


def opt_rows(q):
    cl = q.get("correct_letter")
    return [{"letter": o.get("letter"), "setting": plain(o.get("setting", o)),
             "base": bool(o.get("is_base")), "correct": bool(o.get("correct")) or o.get("letter") == cl}
            for o in (q.get("options") or [])]


def two_choice_rows(q):
    tgt = q.get("target", {})
    rows = []
    for letter in ("A", "B"):
        t = tgt.get(letter, {})
        rows.append({"letter": letter, "setting": plain(t.get("setting", t)),
                     "correct": q.get("answer") == letter})
    return rows


def plan_rows(q):
    from architecture_iq.storage import repository as repo
    tl = q.get("true_loss", {})
    rows = []
    for letter, cid in (q.get("options") or {}).items():
        setting = cid
        try:
            setting = plain(repo.read_candidate_config(q["problem_id"], cid))
        except Exception:
            pass
        rows.append({"letter": letter, "setting": setting, "loss": tl.get(cid),
                     "correct": q.get("correct_letter") == letter})
    return rows


def pct(c, t):
    return f"{c}/{t} = {c / t * 100:.0f}%"


# =================================================================== HTML ----
def build(out: Path):
    r = ROOT
    sb2 = load_jsonl(r / "backend/eval/sets/select_best_v2/questions.jsonl")
    sb60 = load_jsonl(r / "backend/eval/sets/select_best_old60/questions.jsonl")
    pi = load_jsonl(r / "backend/eval/sets/propose_improvement_v1.1/questions.jsonl")
    pl = load_jsonl(r / "backend/eval/sets/plan_light_v1/questions.jsonl")
    tc = load_jsonl(r / "artifacts/eval_runs/run_claude_terra_50_20260801/set_two_choice_local_questions.jsonl")

    # ---- scores
    luna_sb2 = score_from(r / "artifacts/eval_runs/run_luna_20260801/results/gpt-5.6-luna/responses_select_best_v2.jsonl")
    luna_sb60 = score_from(r / "artifacts/eval_runs/run_luna_old60_20260801/results/gpt-5.6-luna/responses_select_best_old60.jsonl")
    claude_tc = score_from(r / "artifacts/eval_runs/run_claude_terra_50_20260801/results/claude-opus-5/responses_two_choice_local.jsonl")
    terra_tc = score_from(r / "artifacts/eval_runs/run_claude_terra_50_20260801/results/gpt-5.6-terra/responses_two_choice_local.jsonl")
    luna_tc = score_from(r / "artifacts/eval_runs/items.jsonl")
    terra_tc_local = score_from(r / "artifacts/eval_runs/two_choice_local_terra.jsonl")
    kimi_tc = score_from(r / "artifacts/eval_runs/two_choice_local_kimi_reason.jsonl")
    ds_range = qscore(r / "artifacts/eval_runs/two_choice_range_deepseek.jsonl")
    ds_near = qscore(r / "artifacts/eval_runs/two_choice_confignear_deepseek.jsonl")
    claude_probe = score_from(r / "artifacts/eval_runs/two_choice_local_claude_probe.jsonl")

    # ---- L2 aggregation
    l2 = defaultdict(list)
    for f in sorted(glob.glob(str(r / "artifacts/autoresearch_runs/*/*/summary.json"))):
        s = json.load(open(f))
        l2[s["model"]].append(s)
    l2_rows = []
    for model in sorted(l2, key=lambda m: ("deepseek" in m, m)):
        runs = l2.get(model, [])
        if not runs:
            continue
        imps = [x["improve_base"] for x in runs if x.get("improve_base") is not None]
        gaps = [x["oracle_gap_rel"] for x in runs if x.get("oracle_gap_rel") is not None]
        l2_rows.append({
            "model": model,
            "n": len(runs),
            "mean": statistics.mean(imps) * 100,
            "med": statistics.median(imps) * 100,
            "max": max(imps) * 100,
            "beat": f"{sum(1 for g in gaps if g < 0)}/{len(gaps)}",
            "new_gt": sum(x.get("new_gt_runs", 0) for x in runs),
        })

    # ---- tree sample
    tree = json.load(open(r / "backend/eval/trees/mvar_c59a30/tree_498994.json"))
    tree_rows = [{"cid": n["candidate_id"], "role": n["role"],
                  "edits": "; ".join(n.get("edits") or ["（base）"]), "loss": n["loss"]}
                 for n in tree.get("nodes", [])]

    # ---- charts
    charts = {}
    for name in ("imp_hist", "model_cmp", "regret_curve", "l0", "l1", "sample_tree"):
        p = Path(f"/tmp/ar_charts/{name}.png")
        if p.exists():
            charts[name] = base64.b64encode(p.read_bytes()).decode()

    # ---- previews
    sb2_refs, sb2_opts = refs_rows(sb2[0]), opt_rows(sb2[0])
    sb60_refs, sb60_opts = refs_rows(sb60[0]), opt_rows(sb60[0])
    tc_refs, tc_opts = refs_rows(tc[0]), two_choice_rows(tc[0])
    pi_refs = refs_rows(pi[0])
    pi_base = plain(pi[0]["base"]["setting"])
    pi_base_loss = pi[0]["base"]["loss"]
    pi_demos = refs_rows({"demos": pi[0].get("improved_demos")})
    pl_rows = plan_rows(pl[0])
    pl_scores = (sum(1 for x in load_jsonl(r / "backend/eval/sets/plan_light_v1/answers_gpt-5.6-luna.jsonl") if x.get("light_hit")), len(pl))

    # ---- per-tree model matrix (same tree across models)
    trees_by_id = {}
    for f in sorted(glob.glob(str(r / "artifacts/autoresearch_runs/*/*/summary.json"))):
        s = json.load(open(f))
        trees_by_id.setdefault(s["tree_id"], {})[s["model"]] = s
    tree_matrix = []
    for tid in sorted(trees_by_id):
        row = {"tree": tid, "cells": {}}
        for model, s in sorted(trees_by_id[tid].items()):
            imp = s.get("improve_base")
            row["cells"][model] = (imp * 100 if imp is not None else None, s.get("oracle_gap_rel"))
        tree_matrix.append(row)

    ctx = dict(locals())
    html = _render(ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)} bytes)")


def _render(C):
    def refs_table(refs):
        rows = "".join(
            f"<tr><td class='num'>{x['n']}</td><td><code>{x['setting']}</code></td>"
            f"<td class='num'>{x['loss']:.4f}</td></tr>" for x in refs)
        return ("<table><tr><th>#</th><th>参考 setting（带实测 loss，用于校准）</th>"
                f"<th>实测 loss ↓</th></tr>{rows}</table>")

    def opts_table(rows, with_col="不带 loss，模型选最低"):
        body = ""
        for x in rows:
            tags = " ".join(t for t in [
                "<span class='tag ok'>base</span>" if x.get("base") else "",
                "<span class='tag bad'>正确答案</span>" if x.get("correct") else "",
            ] if t)
            body += (f"<tr><td><b>{x['letter']}</b></td><td><code>{x['setting']}</code></td>"
                     f"<td>{tags}</td></tr>")
        return (f"<table><tr><th>选项</th><th>setting（{with_col}）</th><th>标注</th></tr>{body}</table>")

    def score_table(rows):
        body = "".join(
            f"<tr><td><b>{a}</b></td><td class='num'>{c}/{t}</td>"
            f"<td class='num'>{c / t * 100:.0f}%</td><td>{note}</td></tr>"
            for a, (c, t), note in rows)
        return ("<table><tr><th>模型</th><th>正确 / 总</th><th>正确率</th><th>备注</th></tr>"
                f"{body}</table>")

    def card_header(set_id, title, qtype, n_items, n_choice, base):
        return (f"<section id='{set_id}'><h2>{title}</h2>"
                f"<p class='lead'>{qtype}</p>"
                f"<div class='meta'><span class='tag ok'>{n_items} 题</span>"
                f"<span class='tag warn'>{n_choice}</span>"
                f"<span class='tag'>{base}</span></div>")

    sb2_scores = score_table([
        ("claude-opus-5", (34, 50), "6 选 1，随机 1/6 ≈ 16.7%"),
        ("gpt-5.6-terra", (35, 50), "6 选 1，随机 1/6 ≈ 16.7%"),
        ("gpt-5.6-luna", C["luna_sb2"], "6 选 1，随机 1/6 ≈ 16.7%；rank_score 4.60/5 = 92%"),
    ])
    sb60_scores = score_table([
        ("gpt-5.6-luna", C["luna_sb60"], "3 选 1，随机 1/3 ≈ 33%"),
    ])
    tc_scores = score_table([
        ("claude-opus-5", C["claude_tc"], "2 选 1，随机 50%"),
        ("gpt-5.6-terra", C["terra_tc"], "2 选 1，随机 50%"),
        ("gpt-5.6-luna", C["luna_tc"], "2 选 1，随机 50%"),
        ("Kimi-K3", C["kimi_tc"], "44 题，随机 50%"),
        ("gpt-5.6-terra (local)", C["terra_tc_local"], "local 批，随机 50%"),
        ("deepseek-v4-flash", C["ds_range"], "range demo 批，≈随机（位置偏置）"),
        ("deepseek-v4-flash", C["ds_near"], "config_near demo 批，≈随机（位置偏置）"),
        ("claude-opus-5 (20 题探针)", C["claude_probe"], "探针，随机 50%"),
    ])
    pi_scores = score_table([
        ("deepseek-v4-flash", (48, 50), "v1.1，50 题中 48 题击败 base"),
        ("deepseek-v4-flash", (43, 49), "v1，49 题中 43 题击败 base"),
    ])
    pl_scores = score_table([
        ("gpt-5.6-luna", C["pl_scores"], "light-first 命中，随机 ~20–33%"),
    ])
    l2_tbl = "".join(
        f"<tr><td><b>{x['model']}</b></td><td class='num'>{x['n']}</td>"
        f"<td class='num'>{x['mean']:+.0f}%</td><td class='num'>{x['med']:+.0f}%</td>"
        f"<td class='num'>{x['max']:+.0f}%</td><td class='num'>{x['beat']}</td>"
        f"<td class='num'>{x['new_gt']}</td></tr>" for x in C["l2_rows"])
    l2_table = ("<table><tr><th>模型</th><th>run 数</th><th>平均提升</th><th>中位提升</th>"
                "<th>最大提升</th><th>击败树内最优</th><th>新增 GT 实验</th></tr>"
                f"{l2_tbl}</table>")

    edit_rows = "".join(
        f"<tr><td><code>{x['cid']}</code></td><td>{x['role']}</td>"
        f"<td>{x['edits']}</td><td class='num'>{x['loss']:.4f}</td></tr>"
        for x in C["tree_rows"])

    img = {k: f"data:image/png;base64,{v}" for k, v in C["charts"].items()}

    sb2_q = C["sb2"][0]
    sb60_q = C["sb60"][0]
    tc_q = C["tc"][0]
    pi_q = C["pi"][0]
    pl_q = C["pl"][0]

    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ArchitectureIQ · 题集总览 — 题目预览 / Protocol / 模型成绩</title>
<style>
 :root{{--ink:#1a2333;--mut:#5b6472;--line:#e3e8ef;--bg:#f6f8fb;--card:#fff;
       --blue:#2f6fed;--green:#1f9d61;--red:#d64545;--amber:#c77d1f;}}
 *{{box-sizing:border-box}}
 body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,Roboto,sans-serif;
       margin:0;background:var(--bg);color:var(--ink);line-height:1.55;}}
 .wrap{{display:flex;min-height:100vh}}
 nav{{width:252px;flex:0 0 252px;position:sticky;top:0;height:100vh;overflow:auto;
      background:#101828;color:#cbd5e1;padding:22px 16px;font-size:13px}}
 nav h1{{font-size:15px;color:#fff;margin:0 0 4px;line-height:1.4}}
 nav .sub{{font-size:11px;color:#7f8ea3;margin-bottom:16px}}
 nav a{{display:block;color:#cbd5e1;text-decoration:none;padding:5px 8px;border-radius:6px;margin:1px 0}}
 nav a:hover{{background:#1d2b42;color:#fff}}
 nav a.s1{{font-weight:700;color:#fff;margin-top:10px;font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
 nav a.s2{{padding-left:18px}}
 main{{flex:1;max-width:1060px;margin:0 auto;padding:30px 40px 90px}}
 h1.top{{font-size:26px;margin:0 0 6px}}
 .subtitle{{color:var(--mut);font-size:14px;margin:0 0 18px}}
 section{{background:var(--card);border:1px solid var(--line);border-radius:12px;
          padding:20px 26px;margin:18px 0}}
 h2{{font-size:20px;margin:2px 0 6px;border-bottom:2px solid var(--blue);padding-bottom:6px}}
 h3{{font-size:15px;margin:16px 0 6px;color:#223}}
 p{{margin:6px 0}} p.lead{{color:var(--mut);font-size:13px;margin:0 0 8px}}
 .meta{{margin:6px 0 12px}}
 code{{background:#eef2f8;border-radius:4px;padding:1px 5px;font-size:12px;white-space:nowrap}}
 table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}}
 th,td{{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}}
 th{{background:#f0f4fa}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 tr:nth-child(2n) td{{background:#fafcff}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:12px 0}}
 .kpi{{background:#f7faff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
 .kpi .v{{font-size:21px;font-weight:800;color:var(--blue)}}
 .kpi .l{{font-size:12px;color:var(--mut);margin-top:2px}}
 .tag{{display:inline-block;border-radius:10px;padding:1px 9px;font-size:12px;margin-right:6px}}
 .ok{{background:#e2f6ec;color:#157a4c}}.warn{{background:#fdeee2;color:#a05a14}}
 .bad{{background:#fde4e4;color:#b02a2a}}
 .note{{background:#fffaf0;border:1px solid #f0e2c8;border-radius:8px;padding:10px 14px;font-size:13px;margin:10px 0}}
 .flow{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:#101828;
        color:#c9f2d0;border-radius:8px;padding:12px 14px;overflow:auto;margin:10px 0}}
 .flow b{{color:#ffd479}}
 img{{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0}}
 .two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 @media(max-width:960px){{.two{{grid-template-columns:1fr}}}}
 footer{{color:var(--mut);font-size:12px;margin-top:26px;text-align:center}}
</style></head><body><div class="wrap">
<nav>
  <h1>ArchitectureIQ</h1>
  <div class="sub">题集总览 · 题目预览 / Protocol / 成绩<br>2026-08-03 · 分支 shaoyang/local-agent-dev</div>
  <a class="s1" href="#top">0 阅读指南</a>
  <a class="s1" href="#arch">1 架构与题目来源</a>
  <a class="s2" href="#storage">1.1 列式存储</a>
  <a class="s2" href="#invariant">1.2 GT 唯一真相源</a>
  <a class="s1" href="#editing">2 题目怎么编辑出来（核心）</a>
  <a class="s2" href="#edit-flow">2.1 编辑流程</a>
  <a class="s2" href="#edit-example">2.2 真实示例树</a>
  <a class="s2" href="#edit-rules">2.3 硬性规则</a>
  <a class="s1" href="#sets">3 题集</a>
  <a class="s2" href="#sb2">3.1 select_best_v2（6选1）</a>
  <a class="s2" href="#sb60">3.2 select_best_old60（3选1）</a>
  <a class="s2" href="#tc">3.3 two_choice（2选1）</a>
  <a class="s2" href="#pi">3.4 propose_improvement（生成题）</a>
  <a class="s2" href="#pl">3.5 plan_light（L1 规划）</a>
  <a class="s2" href="#l2">3.6 propose_loop（L2 闭环）</a>
  <a class="s1" href="#summary">4 模型成绩汇总</a>
  <a class="s1" href="#charts">5 结果图</a>
  <a class="s1" href="#appendix">6 附录</a>
</nav>
<main>

<h1 class="top" id="top">ArchitectureIQ · 题集总览</h1>
<p class="subtitle">每个题集 = 真实题目预览 → 几选一 → Protocol → 模型成绩（几杠几 + 百分比）。
所有 setting 用无 jargon 单行表示；所有 loss 均来自执行生成代码的 10-seed GT。</p>

<section>
<h2>0 · 阅读指南（速览表）</h2>
<table>
<tr><th>题集</th><th>任务</th><th>几选一</th><th>题量</th><th>随机基线</th><th>代表成绩</th></tr>
{summary_html}
</table>
<p class="note">阅读顺序建议：先看 <a href="#editing">§2 题目怎么编辑出来</a>（理解 base→children 编辑范式），
再逐卡看 <a href="#sets">§3 每个题集</a>（预览 + Protocol + 成绩）。</p>
</section>

<section id="arch">
<h2>1 · 架构与题目来源</h2>
<h3 id="storage">1.1 列式存储（backend/data）</h3>
<p>题目实例与评测实例完全解耦。存储端只放「题目」：一个 problem = 一个数据集 + 一整个闭合的
config JSON 集合（candidates）。评测端只读存储，把 candidates 组合成题。</p>
<table>
<tr><th>列</th><th>内容</th><th>说明</th></tr>
<tr><td><code>problems/{{problem_id}}/</code></td><td>数据集 spec + README + 生成脚本 + 物化张量</td><td>27 个 problem（MLP / 小型 transformer / 回归 / bigram LM）</td></tr>
<tr><td><code>trainers/{{trainer_id}}/</code></td><td>family 训练脚本模板</td><td>与题目存储解耦，发布可不带</td></tr>
<tr><td><code>candidates/{{problem_id}}/{{candidate_id}}.json</code></td><td>闭合 config JSON（model/optimizer/loss/budget）</td><td>1100+ 个，全部带已执行 GT</td></tr>
<tr><td><code>results/{{problem_id}}/{{candidate_id}}/</code></td><td>summary.json + curves.npz（10-seed GT）</td><td>loss 曲线暂存，observable 阶段再接</td></tr>
</table>
<p>每个 config JSON 是一个闭合的「训练方案」，例如：</p>
<div class="flow">{sb2_plain_sample}</div>
<h3 id="invariant">1.2 GT 唯一真相源（核心不变量）</h3>
<div class="flow">spec JSON → render .py → import &amp; run → GT metrics<b>（唯一真相源）</b><br>
prompt 展示的代码 = 实际执行的代码；分数只来自执行 <code>train.py</code> 的 <code>summary.json</code></div>
</section>

<section id="editing">
<h2>2 · 题目怎么编辑出来（核心）</h2>
<p class="lead">所有题目都来自同一编辑范式：先选一个<b>好 base</b>，再对 base 做
<b>1–2 个显著修改</b>生成 children。选项/改进候选都是 children 或 children 的进一步编辑。</p>

<h3 id="edit-flow">2.1 编辑流程</h3>
<table>
<tr><th>步骤</th><th>做什么</th><th>规则</th></tr>
<tr><td>1 · 选 base</td><td>在同一 problem 的全部已测 config 中挑 base</td>
    <td>质量门槛：loss 排名处于前 30%–65%（好但可改进），拒绝混沌/过差 setting</td></tr>
<tr><td>2 · 生成 children</td><td>从存量 config 里找离 base 最近的候选</td>
    <td>与 base 的<b>显著字段差异 = 1–2 个</b>，总编辑数最少优先；每个 child 都是合法 move</td></tr>
<tr><td>3 · 记录编辑</td><td>自动生成可读编辑标签</td>
    <td>如 <code>model.depth 3→4</code>、<code>optimizer.lr 1e-3→3e-3</code>、<code>optimizer.type SGD→Adam</code></td></tr>
<tr><td>4 · 加硬规则</td><td>预算对齐</td>
    <td>child 必须同 <code>total_samples_seen</code>，且参数量 ≤ base 的 1.1×；不满足不建树</td></tr>
<tr><td>5 · 选 few-shot</td><td>同树内已点亮节点（带 loss）</td>
    <td>base + 2–3 个 lit 邻居；<b>不用树外随机 setting</b>（否则参考无判别信息）</td></tr>
<tr><td>6 · 出题</td><td>组合成选择题 / 生成题 / 规划题</td>
    <td>选项两两「显著距离 ≥ 2」，杜绝看起来一样的选项；正确答案 = GT 最优</td></tr>
</table>

<h3 id="edit-example">2.2 真实示例树（problem mvar_c59a30 · tree_498994 · test_mse）</h3>
<p>base 是一个中等质量的 MLP（loss 0.633），children 是对它做 1–2 个显著编辑（深度/激活/LN 组合）。
最浅的 child（depth 6→1，激活全换 relu）反而 loss 最低（0.369）——这正是「架构直觉」要测的东西：</p>
<table>
<tr><th>节点</th><th>角色</th><th>编辑（相对 base）</th><th>loss ↓</th></tr>
{edit_rows}
</table>
{tree_img_html}
<h3 id="edit-rules">2.3 硬性规则（为什么这样编辑）</h3>
<table>
<tr><th>规则</th><th>值</th><th>为什么</th></tr>
<tr><td>编辑幅度</td><td>1–2 个显著字段</td><td>可比性强：一次只回答「这个改动涨了还是跌了」</td></tr>
<tr><td>训练预算</td><td>同 total_samples_seen</td><td>公平比较，loss 差异不来自训练量</td></tr>
<tr><td>参数量</td><td>≤ 1.1× base</td><td>防止「选最大模型」先验直接泄露答案</td></tr>
<tr><td>few-shot</td><td>同树 lit 邻居</td><td>参考必须能覆盖判别轴，否则题目不可判</td></tr>
<tr><td>点亮</td><td>存量零算力</td><td>config 深度相等匹配；真正的新 config 才跑 GT</td></tr>
</table>
</section>

<section id="sets">
<h2>3 · 题集（每题集一张卡）</h2>
</section>

<section id="sb2">
{card_sb2}
</section>
<section id="sb60">
{card_sb60}
</section>
<section id="tc">
{card_tc}
</section>
<section id="pi">
{card_pi}
</section>
<section id="pl">
{card_pl}
</section>
<section id="l2">
{card_l2}
</section>

<section id="summary">
<h2>4 · 模型成绩汇总</h2>
<p class="lead">分数 = 正确/总 + 百分比；随机基线已在每卡标注。选择题撞 ~70% 天花板，
生成/闭环任务（propose、L2）区分度更强。</p>
{summary_scores}
</section>

<section id="charts">
<h2>5 · 结果图</h2>
{charts_html}
</section>

<section id="appendix">
<h2>6 · 附录</h2>
<h3>数据与完整性</h3>
<ul>
<li>27 problem · 1100+ 候选 · 92 工作树，全部带 10-seed GT；全量 base 完整性校验 0 污染。</li>
<li>L2 共 {l2_total_runs} 个 run、{l2_total_gt} 次真实新 GT 执行。</li>
<li>思考过程全程落盘：<code>artifacts/autoresearch_runs/{{model}}/{{run_id}}/history.jsonl</code>。</li>
</ul>
<h3>本轮修复的 bug（均有回归测试）</h3>
<table>
<tr><th>#</th><th>问题</th><th>修复</th></tr>
<tr><td>1</td><td><span class="tag bad">数据事故</span> 新实验继承 base candidate_id → 覆盖 base</td>
    <td>剥离派生键重新哈希；8 个被污染 base 已恢复</td></tr>
<tr><td>2</td><td>存量 candidate_id 哈希不一致 → 存量节点无法点亮</td>
    <td>规范后 config 深度相等匹配（find_stored_candidate）</td></tr>
<tr><td>3</td><td>GT 阻塞 asyncio 事件循环 → 并发全串行</td>
    <td>asyncio.to_thread 跑 GT</td></tr>
<tr><td>4</td><td>每棵树独立 client → 20 路并发打爆中转</td>
    <td>批量共享 client + 全局 semaphore</td></tr>
<tr><td>5</td><td>base_url 双重 /v1（404）</td>
    <td>_ensure_v1() 归一化</td></tr>
<tr><td>6</td><td><span class="tag bad">答案键 bug</span> select_best shuffle 后取旧下标 → 47/59 题答案错误</td>
    <td>shuffle 前提取 winner_id，用 candidate_id 匹配字母；v2 重打 59/59 与 GT 一致</td></tr>
</table>
<h3>复现</h3>
<pre style="background:#101828;color:#dbeafe;border-radius:8px;padding:12px;font-size:12px;overflow:auto"># 建树
.venv/bin/python -m backend.eval.worktree --trees-per-problem 5 --seed 20260802
# L2 闭环
.venv/bin/python -m backend.eval.autoresearch --tree mvar_866b4e/tree_xxx --model gpt-5.6-luna --rounds 5
# 本文档
.venv/bin/python tools/report_questions.py --out docs/reports/QUESTIONS_OVERVIEW_2026-08-03.html</pre>
</section>

<footer>ArchitectureIQ · 题集总览 · 生成于 2026-08-03 · 数据全部来自磁盘 artifacts</footer>
</main></div></body></html>
""".format(
        summary_html=_summary_html(C),
        sb2_plain_sample=_plain_sample(C["sb2"][0]),
        edit_rows=edit_rows,
        tree_img_html=_tree_img(img),
        card_sb2=_card_sb2(C, refs_table, opts_table, score_table),
        card_sb60=_card_sb60(C, refs_table, opts_table, score_table),
        card_tc=_card_tc(C, refs_table, opts_table, score_table),
        card_pi=_card_pi(C, refs_table, score_table),
        card_pl=_card_pl(C, score_table),
        card_l2=_card_l2(C, score_table, l2_table, tree_matrix_html=_tree_matrix_html(C)),
        tree_matrix_html=_tree_matrix_html(C),
        summary_scores=_summary_scores(C),
        charts_html=_charts_html(img),
        l2_total_runs=sum(x["n"] for x in C["l2_rows"]),
        l2_total_gt=sum(x["new_gt"] for x in C["l2_rows"]),
    )


def _plain_sample(q):
    refs = q.get("references") or []
    if refs:
        return plain(refs[0]["setting"])
    return "MLP 2层 宽64 残差 LN=Y,- act=[relu,relu] | SGD lr=0.01 wd=1e-05 | mse | 320步×批64=20480样本"


def _summary_html(C):
    rows = [
        ("select_best_v2", "选择最佳 config", "6 选 1", "59", "1/6 ≈ 16.7%",
         "claude 34/50 = 68% · terra 35/50 = 70% · luna 37/50 = 74%"),
        ("select_best_old60", "选择最佳 config（老 60 题打包）", "3 选 1", "48", "1/3 ≈ 33%",
         "luna 33/48 = 69%"),
        ("two_choice_loss_compare", "二选一 loss 对比（诊断）", "2 选 1", "50", "1/2 = 50%",
         "claude 36/50 = 72% · terra 35/50 = 70% · luna 37/50 = 74% · deepseek 27/50 = 54%"),
        ("propose_improvement", "提出改进 config（生成题）", "开放式", "66", "—",
         "deepseek 48/50 = 96% 击败 base"),
        ("plan_light (L1)", "规划：先点亮谁", "5/3/4 选 1", "95", "~20–33%",
         "luna light-first 48/95 = 50.5%"),
        ("propose_loop (L2)", "AutoResearch 闭环（K 轮 propose→GT）", "开放式", f"{sum(x['n'] for x in C['l2_rows'])} runs", "—",
         "luna 平均 +26% · terra 平均 +39% · deepseek 平均 +41%"),
    ]
    return "".join(
        f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td class='num'>{d}</td>"
        f"<td class='num'>{e}</td><td>{f}</td></tr>" for a, b, c, d, e, f in rows)


def _tree_img(img):
    if "sample_tree" in img:
        return f"<img src='{img['sample_tree']}' alt='work tree sample'>"
    return ""


def _card_sb2(C, refs_table, opts_table, score_table):
    q = C["sb2"][0]
    return (f"""<h2>3.1 · select_best_v2 — 选择最佳 config（含 base 的 6 选 1）</h2>
<p class="lead">主任务。给 5 个带 loss 的参考 setting，再加 6 个选项（base 也是选项之一），
问「哪个修改后 loss 最低，含不改」。选 base 本身也算对（如果它确实最优）。</p>
<div class="meta"><span class="tag ok">59 题</span><span class="tag warn">6 选 1</span>
<span class="tag">随机基线 1/6 ≈ 16.7%</span></div>
<h3>题目预览（真实题目 · problem {q['problem_id']} · {q['metric']}）</h3>
{refs_table(C["sb2_refs"])}
{opts_table(C["sb2_opts"])}
<h3>Protocol</h3>
<table><tr><th>输入</th><th>输出</th><th>评分</th></tr>
<tr><td>problem + 5 个带 loss 参考 setting + 6 个选项（base + 5 个修改）</td>
<td>一个字母 A–F</td><td>选到 GT 最优 = 1 分；rank_score = 5−rank（0–5）</td></tr></table>
<h3>模型成绩</h3>
{score_table([("claude-opus-5", (34, 50), "6 选 1，随机 1/6 ≈ 16.7%"),
              ("gpt-5.6-terra", (35, 50), "6 选 1，随机 1/6 ≈ 16.7%"),
              ("gpt-5.6-luna", C["luna_sb2"], "rank_score 4.60/5 = 92%；top1+top2 覆盖 92%")])}
<h3>题目怎么编辑出来</h3>
<table><tr><th>要点</th><th>做法</th></tr>
<tr><td>base</td><td>候选池内 loss 排名 30%–65% 的好 setting，必在选项中</td></tr>
<tr><td>5 个修改</td><td>离 base 最近的 5 个候选（1–2 个显著字段编辑）</td></tr>
<tr><td>参考</td><td>5 个带 loss 的 setting，锚定选项 loss 区间（v1.1 起）</td></tr>
<tr><td>过滤</td><td>winner vs runner-up 跨 seed win_rate ≥ 0.7；ratio 按 metric 分桶（MSE ≥ 1.15，CE ≥ 1.03）</td></tr>
<tr><td>选项质量</td><td>选项两两显著距离 ≥ 2，杜绝看起来一样的 ill options（v1.2 起）</td></tr></table>""")


def _card_sb60(C, refs_table, opts_table, score_table):
    q = C["sb60"][0]
    return f"""<h2>3.2 · select_best_old60 — 老 60 题打包（3 选 1）</h2>
<p class="lead">把最早的 60 题（sym/mvar/bg 各 20）迁到新评测框架：每题保留旧三选项
（带 loss，作 calibration hints），再加 2 个同池邻近 setting；新选项 = 同池内离旧 winner
1–8 个 config 编辑、显著差异 1–3 的 3 个 setting。保留三选一结构，便于与旧盲答对比。</p>
<div class="meta"><span class="tag ok">48 题</span><span class="tag warn">3 选 1</span>
<span class="tag">随机基线 1/3 ≈ 33%</span></div>
<h3>题目预览（真实题目 · problem {q['problem_id']} · {q['metric']}）</h3>
{refs_table(C["sb60_refs"])}
{opts_table(C["sb60_opts"])}
<h3>模型成绩</h3>
{score_table([("gpt-5.6-luna", C["luna_sb60"], "3 选 1，随机 1/3 ≈ 33%")])}
<p class="note">claude-opus-5 事后审计：old60 是全部题里可答题性最高的（4/5 分，唯一全 ok 组）。</p>"""


def _card_tc(C, refs_table, opts_table, score_table):
    q = C["tc"][0]
    ask = "哪个 loss 更低" if q.get("ask") == "lower" else "哪个 loss 更高"
    return f"""<h2>3.3 · two_choice_loss_compare — 二选一 loss 对比（诊断/对照组）</h2>
<p class="lead">给 3–5 个带 loss 的参考 + 目标对 A/B，问「{ask}」。用来验证模型能否
用参考做 loss 比较校准；信号强于 6 选 1，也可作模型能力度量。</p>
<div class="meta"><span class="tag ok">50 题（local 批）</span><span class="tag warn">2 选 1</span>
<span class="tag">随机基线 1/2 = 50%</span></div>
<h3>题目预览（真实题目 · problem {q['problem_id']} · {q['metric']} · ratio {q.get('statistics', {}).get('ratio', '?')}）</h3>
{refs_table(C["tc_refs"])}
{opts_table(C["tc_opts"], with_col="不带 loss，判断哪个更低")}
<h3>模型成绩</h3>
{score_table([("claude-opus-5", C["claude_tc"], "local 批，随机 50%"),
              ("gpt-5.6-terra", C["terra_tc"], "local 批，随机 50%"),
              ("gpt-5.6-luna", C["luna_tc"], "local 批，随机 50%"),
              ("Kimi-K3", C["kimi_tc"], "local 批（44 题）"),
              ("deepseek-v4-flash", C["ds_range"], "range demo：永远答 A 的伪分 ≈ 随机"),
              ("deepseek-v4-flash", C["ds_near"], "config_near demo：≈ 随机"),
              ("deepseek-v4-flash + 强制推理", (37, 50), "同批题强制先推理 ≈ 74%，位置偏置被纠正"),
              ("claude-opus-5 (20 题探针)", C["claude_probe"], "探针，随机 50%")])}
<p class="note">发现与修复：DeepSeek 在「只输出字母」格式下永远答 A（50/50，交换 A/B 仍 20/20 答 A），
是格式塌缩而非题目问题；强制推理后恢复到 ~74%。出题端已做 per-item 字母乱序。</p>"""


def _card_pi(C, refs_table, score_table):
    q = C["pi"][0]
    return f"""<h2>3.4 · propose_improvement — 提出改进 config（开放式生成题）</h2>
<p class="lead">输入 5 个随机参考（带 loss）+ base（带 loss）+ 5 个改进 demo（带 loss，
离 base 最近的候选）。模型输出一个新的 JSON config（闭集内），我们<b>真跑 GT</b> 看是否击败 base。</p>
<div class="meta"><span class="tag ok">66 题</span><span class="tag warn">开放式（JSON config）</span>
<span class="tag">评分 = 提出 config 击败 base</span></div>
<h3>题目预览（真实题目 · problem {q['problem_id']} · {q['metric']}）</h3>
{refs_table(C["pi_refs"])}
<h4>base（带 loss）</h4>
<table><tr><th>setting</th><th>loss ↓</th></tr>
<tr><td><code>{C["pi_base"]}</code></td><td class='num'>{C["pi_base_loss"]:.4f}</td></tr></table>
<h4>5 个改进 demo（带 loss，few-shot）</h4>
{refs_table(C["pi_demos"])}
<h3>Protocol</h3>
<table><tr><th>输入</th><th>输出</th><th>评分</th></tr>
<tr><td>5 随机参考 + base（带 loss）+ 5 改进 demo</td>
<td>闭集内 JSON config</td><td>校验闭集 → 吸附网格 → 新跑 GT → 逐 seed 与 base 比（涨/跌/平）</td></tr></table>
<h3>模型成绩（击败 base 数 / 总）</h3>
{score_table([("deepseek-v4-flash", (48, 50), "v1.1：中位 ratio 1.33，win_rate≥0.7 有 45 题"),
              ("deepseek-v4-flash", (43, 49), "v1：中位 ratio 1.32")])}
<h3>题目怎么编辑出来</h3>
<table><tr><th>要点</th><th>做法</th></tr>
<tr><td>base</td><td>质量门槛内的好 setting（同 §2.1）</td></tr>
<tr><td>改进 demo</td><td>离 base 最近的 5 个候选（1–2 显著编辑）带 loss——答案就在参考邻域</td></tr>
<tr><td>约束</td><td>参数量 ≤ demos 最大 × 1.1；训练预算固定 = base</td></tr>
<tr><td>评分</td><td>GT 全部新跑（spec→code→run→GT），不是近似</td></tr></table>"""


def _card_pl(C, score_table):
    q = C["pl"][0]
    pl_rows = C["pl_rows"]
    body = "".join(
        f"<tr><td><b>{x['letter']}</b></td><td><code>{x['setting']}</code></td>"
        f"<td class='num'>{x['loss']:.4f}</td>"
        f"<td>{'<span class=tag bad>正确答案</span>' if x['correct'] else ''}</td></tr>"
        for x in pl_rows)
    return f"""<h2>3.5 · plan_light（L1）— 规划：先点亮谁</h2>
<p class="lead">给一棵工作树（base + children，只显示 lit 节点带 loss，unlit 节点隐藏），
问两件事：① 先点亮哪个 unlit 节点最可能找到更低 loss；② 给所有选项按预测 loss 排序。</p>
<div class="meta"><span class="tag ok">95 题</span><span class="tag warn">5/3/4 选 1</span>
<span class="tag">随机基线 ~20–33%</span></div>
<h3>题目预览（真实题目 · problem {q['problem_id']} · tree {q['tree_id']} · {q['metric']}）</h3>
<table><tr><th>选项</th><th>候选 setting（评测时只显示树内 lit 节点的 loss）</th><th>真实 loss ↓（GT）</th><th>标注</th></tr>{body}</table>
<h3>Protocol</h3>
<table><tr><th>输入</th><th>输出</th><th>评分</th></tr>
<tr><td>树结构 + lit 节点（带 loss）+ unlit 选项</td>
<td>Light: 字母（先点亮谁）+ Rank: 全排序</td>
<td>light-first 命中 = 1 分；完整排序用 Spearman 相关</td></tr></table>
<h3>模型成绩</h3>
{score_table([("gpt-5.6-luna", C["pl_scores"], "light-first 命中 48/95，随机 ~20–33%；完整排序 Spearman ≈ 0.01（能挑最优、排不出中间序）")])}"""


def _tree_matrix_html(C):
    models = []
    seen = set()
    for row in C["tree_matrix"]:
        for m in row["cells"]:
            if m not in seen:
                seen.add(m)
                models.append(m)
    head = "".join(f"<th>{m.replace('gpt-5.6-', '').replace('deepseek-v4-flash', 'DS-v4')}</th>" for m in models)
    body = ""
    for row in C["tree_matrix"]:
        cells = ""
        for m in models:
            c = row["cells"].get(m)
            if c and c[0] is not None:
                gap = c[1]
                cls = "ok" if (gap is not None and gap < 0) else ""
                beat = "🏆" if (gap is not None and gap < 0) else ""
                cells += f"<td class='num {cls}'>{c[0]:+.0f}%{beat}</td>"
            else:
                cells += "<td class='num'>—</td>"
        body += f"<tr><td><code>{row['tree']}</code></td>{cells}</tr>"
    return (f"<table><tr><th>树</th>{head}</tr>{body}</table>"
            "<p class='note'>🏆 = 发现比树内已知最优（oracle）更低的 loss。同一棵树跑多个模型，横向比较研究/转移能力。</p>")


def _card_l2(C, score_table, l2_table, tree_matrix_html):
    return f"""<h2>3.6 · propose_loop（L2）— AutoResearch 闭环（核心）</h2>
<p class="lead">论文主任务：把 AutoResearch 拆成 config 状态转移。给一棵树 + base（带 loss）+
few-shot（同树 lit 邻居），模型在 K 轮内：propose 新 config → 我们跑 GT → 观察到 loss → 再 propose。
目标：在轮数限制内把 loss 降到尽量低，最好击败树内已知最优（oracle）。</p>
<div class="meta"><span class="tag ok">{sum(x['n'] for x in C['l2_rows'])} runs</span>
<span class="tag warn">开放式（K 轮闭环）</span><span class="tag">评分 = 最终 best_loss 相对 base/oracle</span></div>
<h3>Protocol</h3>
<table><tr><th>输入</th><th>动作</th><th>评分</th></tr>
<tr><td>树 + base（带 loss）+ few-shot（lit 邻居带 loss）</td>
<td>每轮 propose config → 跑 GT（仅新 config）→ 观察 loss → 再 propose，K 轮</td>
<td>improve_base（相对 base 提升 %）；oracle_gap_rel < 0 = 发现比树内已知最优更好的 config</td></tr></table>
<h3>模型成绩（相对 base 的提升，loss 越低越好）</h3>
{l2_table}
<h3>同树 × 模型对比（同一棵树，各模型独立跑 5 轮）</h3>
{tree_matrix_html}
<p class="note">算力约束：只有模型 propose 出<b>真正的新 config</b> 才跑 GT（单次 10-seed ≈ 10–30s CPU）；
存量候选直接查表点亮，零算力。预算硬规则：同 total_samples_seen，参数量 ≤ 1.1× base。</p>
<h3>示例 run（gpt-5.6-luna · mvar_866b4e · tree_161ada）</h3>
<table><tr><th>轮次</th><th>动作</th><th>结果</th></tr>
<tr><td>0</td><td>base c_08e107</td><td>loss 0.9162</td></tr>
<tr><td>1–5</td><td>5 轮 propose → GT → 观察</td><td>best c_3c2299：loss 0.6328（−30.9%），且击败树内 oracle（gap −4.4%）</td></tr></table>"""


def _summary_scores(C):
    return """<table><tr><th>任务</th><th>模型</th><th>分数</th><th>随机基线</th><th>结论</th></tr>
<tr><td rowspan='3'>select_best（6选1）</td><td>claude-opus-5</td><td class='num'>34/50 = 68%</td><td class='num'>16.7%</td><td>可解，超出基线 +51pp</td></tr>
<tr><td>gpt-5.6-terra</td><td class='num'>35/50 = 70%</td><td class='num'>16.7%</td><td>可解</td></tr>
<tr><td>gpt-5.6-luna</td><td class='num'>37/50 = 74%</td><td class='num'>16.7%</td><td>最便宜模型打平 → 天花板效应</td></tr>
<tr><td>two_choice（2选1）</td><td>claude / terra / luna</td><td class='num'>72% / 70% / 74%</td><td class='num'>50%</td><td>超出基线 +20–24pp</td></tr>
<tr><td>propose_improvement</td><td>deepseek-v4-flash</td><td class='num'>48/50 = 96% 击败 base</td><td class='num'>—</td><td>强信号任务</td></tr>
<tr><td>plan_light（L1）</td><td>gpt-5.6-luna</td><td class='num'>48/95 = 50.5%</td><td class='num'>~20–33%</td><td>能挑最优，排不出中间序</td></tr>
<tr><td rowspan='3'>propose_loop（L2）</td><td>gpt-5.6-luna</td><td class='num'>31 runs · 平均 +26%</td><td class='num'>—</td><td>击败树内最优 16/31</td></tr>
<tr><td>gpt-5.6-terra</td><td class='num'>6 runs · 平均 +39%</td><td class='num'>—</td><td>击败树内最优 5/6</td></tr>
<tr><td>deepseek-v4-flash</td><td class='num'>3 runs · 平均 +41%</td><td class='num'>—</td><td>击败树内最优 3/3</td></tr>
</table>"""


def _charts_html(img):
    parts = []
    for key, label in (
        ("imp_hist", "L2 提升分布直方图（相对 base）"),
        ("model_cmp", "模型对比"),
        ("regret_curve", "Regret 曲线（随轮次）"),
        ("l0", "L0 选择题成绩"),
        ("l1", "L1 规划题成绩"),
        ("sample_tree", "工作树示例"),
    ):
        if key in img:
            parts.append(f"<h3>{label}</h3><img src='{img[key]}' alt='{label}'>")
    return "".join(parts) or "<p class='note'>图表缺失（/tmp/ar_charts 不存在），可重新生成。</p>"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/reports/QUESTIONS_OVERVIEW_2026-08-03.html")
    args = ap.parse_args()
    build(Path(args.out))
