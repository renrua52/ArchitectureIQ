"""Render an HTML report for AutoResearch loop runs.

Reads ``artifacts/autoresearch_runs/{model}/{run_id}/`` (summary.json +
history.jsonl) and writes a single self-contained HTML with per-round
proposal / observed loss / reasoning (collapsible).

Usage:
    .venv/bin/python -m backend.eval.report_autoresearch \
        [--out artifacts/autoresearch_report.html] [--models gpt-5.6-luna]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

OUT_ROOT = Path("artifacts/autoresearch_runs")


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt_loss(v) -> str:
    return "—" if v is None else f"{float(v):.4g}"


def render_run(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (run_dir / "history.jsonl").open(encoding="utf-8")]
    head = (
        f"<h3>{_esc(run_dir.parent.name)} · {_esc(summary['tree_id'])}</h3>"
        f"<p class='meta'>problem={_esc(summary['problem_id'])} · metric={_esc(summary['metric'])} · "
        f"base loss={_fmt_loss(summary['base_loss'])} · best={_fmt_loss(summary['best_loss'])} · "
        f"oracle={_fmt_loss(summary['oracle'])} · improve={summary.get('improve_base')} · "
        f"oracle_gap={summary.get('oracle_gap_rel')} · new_gt={summary.get('new_gt_runs')} · "
        f"best={_esc(summary.get('best_candidate'))}</p>"
    )
    rounds = []
    for r in rows:
        prop = r.get("proposal")
        prop_html = (f"<pre>{_esc(json.dumps(prop, indent=1, ensure_ascii=False))}</pre>"
                     if prop else "<p class='err'>no JSON parsed</p>")
        errs = "".join(f"<li>{_esc(e)}</li>" for e in r.get("errors", []))
        notes = "".join(f"<li>{_esc(n)}</li>" for n in r.get("notes", []))
        reasoning = r.get("reasoning") or ""
        rounds.append(f"""
        <div class="round">
          <div class="rhead">Round {r['round']}
            <span class="tag {'ok' if r['ok'] else 'bad'}">{'lit' if r['ok'] else 'failed'}</span>
            <span class="tag">{'wasted' if r.get('wasted') else 'moved'}</span>
            <span class="loss">loss={_fmt_loss(r.get('loss'))} best={_fmt_loss(r.get('best_loss'))}</span>
          </div>
          <details><summary>prompt</summary><pre class="prompt">{_esc(r.get('prompt', ''))}</pre></details>
          <div class="prop">{prop_html}</div>
          <ul class="notes">{notes}</ul>
          <ul class="errors">{errs}</ul>
          <details class="think"><summary>reasoning</summary><pre class="reason">{_esc(reasoning)}</pre></details>
        </div>""")
    return head + "".join(rounds)


def render(models: list[str] | None = None) -> str:
    model_dirs = [d for d in OUT_ROOT.iterdir() if d.is_dir()] if OUT_ROOT.is_dir() else []
    if models:
        model_dirs = [d for d in model_dirs if d.name in models]
    sections = []
    total = 0
    improves = []
    for md in sorted(model_dirs):
        runs = sorted(md.glob("*/summary.json"))
        if not runs:
            continue
        body = "".join(render_run(r.parent) for r in runs)
        n = len(runs)
        total += n
        for r in runs:
            s = json.loads(r.read_text(encoding="utf-8"))
            if s.get("improve_base") is not None:
                improves.append((md.name, s["improve_base"], s["oracle_gap_rel"]))
        sections.append(f"<section><h2>{_esc(md.name)} · {n} runs</h2>{body}</section>")
    stats = ""
    if improves:
        by = {}
        for m, imp, gap in improves:
            by.setdefault(m, []).append((imp, gap))
        rows = ""
        for m, vals in by.items():
            imps = [v[0] for v in vals]
            gaps = [v[1] for v in vals]
            rows += (f"<tr><td>{_esc(m)}</td><td>{len(vals)}</td>"
                     f"<td>{sum(imps)/len(imps):.3f}</td>"
                     f"<td>{max(imps):.3f}</td>"
                     f"<td>{sum(gaps)/len(gaps):.3f}</td></tr>")
        stats = f"""<section><h2>Aggregate</h2>
        <table><tr><th>model</th><th>runs</th><th>mean improve</th><th>best improve</th><th>mean oracle_gap</th></tr>
        {rows}</table></section>"""
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>AutoResearch propose-loop report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}}
 h2{{border-bottom:2px solid #333;padding-bottom:4px}}
 h3{{margin:18px 0 4px}}
 .meta{{color:#555;font-size:13px;margin:2px 0 10px}}
 .round{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}}
 .rhead{{font-weight:600;margin-bottom:6px}}
 .tag{{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;margin-left:6px}}
 .tag.ok{{background:#dfd;color:#070}} .tag.bad{{background:#fdd;color:#a00}}
 .loss{{float:right;color:#444;font-weight:400}}
 .prompt{{white-space:pre-wrap;background:#f6f8fa;padding:8px;font-size:12px}}
 .reason{{white-space:pre-wrap;background:#fffbe6;padding:8px;font-size:12px;max-height:300px;overflow:auto}}
 pre{{white-space:pre-wrap;background:#f6f8fa;padding:8px;font-size:12px;margin:4px 0}}
 .err{{color:#a00}} .errors{{color:#a00;margin:4px 0}} .notes{{color:#444;font-size:12px;margin:4px 0}}
 table{{border-collapse:collapse;margin:10px 0}} td,th{{border:1px solid #ccc;padding:4px 10px;font-size:13px}}
</style></head><body><h1>AutoResearch propose-loop report</h1>{stats}{''.join(sections)}</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/autoresearch_report.html")
    ap.add_argument("--models", default=None, help="comma-separated model dirs")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(models), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
