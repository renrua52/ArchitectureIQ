#!/usr/bin/env python3
"""Render the completed-GT setting-to-loss evaluation as a static HTML deck."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS = Path(
    "artifacts/high_budget_gpt54_eval/wide_completed_setting_to_loss_analysis.json"
)
DEFAULT_OUTPUT = Path("docs/0715_setting_to_loss.html")


def load_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number != 0.0 and (abs(number) >= 1e4 or abs(number) < 1e-3):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def method_row(analysis: dict[str, Any], condition: str, method: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in analysis["family_balanced_methods"][condition]
            if row["method"] == method
        ),
        None,
    )


def condition_table_rows(analysis: dict[str, Any]) -> str:
    with_rows = {
        row["method"]: row
        for row in analysis["family_balanced_methods"]["with_parameter_count"]
    }
    without_rows = {
        row["method"]: row
        for row in analysis["family_balanced_methods"]["without_parameter_count"]
    }
    preferred = [
        "xgboost",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "gradient_boosting",
        "rbf_svr",
        "mlp",
        "shallow_tree",
        "compact_polynomial_ridge",
        "full_elastic_net",
        "full_ridge",
        "compact_ridge",
        "optimizer_lr_ridge",
        "optimizer_lr_lookup",
        "params_polynomial_ridge",
        "params_ridge",
        "max_params_heuristic",
        "constant_mean",
    ]
    ordered = [name for name in preferred if name in with_rows or name in without_rows]
    ordered.extend(
        sorted((set(with_rows) | set(without_rows)).difference(ordered))
    )
    rendered = []
    for name in ordered:
        with_metric = with_rows.get(name, {}).get("family_macro", {})
        without_metric = without_rows.get(name, {}).get("family_macro", {})
        rendered.append(
            "<tr>"
            f"<td><code>{esc(name)}</code></td>"
            f"<td>{num(with_rows.get(name, {}).get('selection_cv_log_rmse'))}</td>"
            f"<td>{num(without_rows.get(name, {}).get('selection_cv_log_rmse'))}</td>"
            f"<td>{pct(with_metric.get('three_choice_accuracy'))}</td>"
            f"<td>{pct(without_metric.get('three_choice_accuracy'))}</td>"
            f"<td>{num(with_metric.get('log_rmse'))}</td>"
            f"<td>{num(without_metric.get('log_rmse'))}</td>"
            f"<td>{num(with_metric.get('spearman'))}</td>"
            f"<td>{num(without_metric.get('spearman'))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def environment_rows(analysis: dict[str, Any]) -> str:
    rendered = []
    for row in analysis["per_environment"]:
        with_metric = row["with_parameter_count"]
        without_metric = row["without_parameter_count"]
        with_detail = next(iter(with_metric["per_environment"].values()))
        without_detail = next(
            iter(without_metric["per_environment"].values())
        )
        rendered.append(
            "<tr>"
            f"<td><code>{esc(row['task_id'])}</code><br><span class='mini'>{esc(row['family'])}</span></td>"
            f"<td>{esc(with_metric['method'])}</td>"
            f"<td>{num(with_detail['raw']['rmse'])}</td>"
            f"<td>{num(with_detail['log']['rmse'])}</td>"
            f"<td>{num(with_detail['ranking']['spearman'])}</td>"
            f"<td>{pct(with_detail['three_choice']['all']['accuracy'])}</td>"
            f"<td>{pct(with_detail['three_choice']['gap_ge_0_05']['accuracy'])}</td>"
            f"<td>{num(with_detail['three_choice']['all']['regret']['log']['mean'])}</td>"
            f"<td>{esc(without_metric['method'])}</td>"
            f"<td>{pct(without_detail['three_choice']['all']['accuracy'])}</td>"
            f"<td>{pct(row['three_choice_delta_without_minus_with'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def family_rows(analysis: dict[str, Any]) -> str:
    rendered = []
    for row in analysis["family_pooled"]:
        with_metric = row["with_parameter_count"]
        without_metric = row["without_parameter_count"]
        rendered.append(
            "<tr>"
            f"<td>{esc(row['family'])}</td>"
            f"<td>{esc(with_metric['method'])}</td>"
            f"<td>{num(with_metric['within_environment']['macro']['log_rmse'])}</td>"
            f"<td>{num(with_metric['within_environment']['macro']['spearman'])}</td>"
            f"<td>{pct(with_metric['within_environment']['macro']['three_choice_accuracy'])}</td>"
            f"<td>{esc(without_metric['method'])}</td>"
            f"<td>{pct(without_metric['within_environment']['macro']['three_choice_accuracy'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def feature_cards(analysis: dict[str, Any]) -> str:
    cards = []
    for family, conditions in sorted(analysis["xgboost_feature_importance"].items()):
        top = conditions["with_parameter_count"][:6]
        parameter_feature = conditions.get("parameter_count_feature")
        items = "".join(
            f"<li><code>{esc(item['feature'])}</code><span>{pct(item['importance'])}</span></li>"
            for item in top
        )
        cards.append(
            f"<div class='evidence-card'><h3>{esc(family)}</h3><ol class='importance-list'>{items}</ol>"
            f"<p class='mini'>log_total_params: "
            f"{('rank ' + str(parameter_feature['rank']) + ', ' + pct(parameter_feature['importance'])) if parameter_feature else 'constant/unused'}</p></div>"
        )
    return "".join(cards)


def case_cards(analysis: dict[str, Any]) -> str:
    phenomena = analysis["phenomena"]
    cards = []
    for title, rows in (
        ("参数量帮助最大", phenomena["parameter_count_helps_cases"][:2]),
        ("参数量伤害最大", phenomena["parameter_count_hurts_cases"][:2]),
        ("XGBoost 最大误差", phenomena["worst_xgboost_log_errors"][:2]),
    ):
        body = []
        for row in rows:
            body.append(
                "<div class='case-line'>"
                f"<code>{esc(row['environment'])}/{esc(row['candidate_id_short'])}</code> "
                f"{esc(row['optimizer'])} lr={row['learning_rate']:.0e}, "
                f"params={row.get('total_params') or '—'}<br>"
                f"GT log-loss {num(row['actual_log_loss'])}; "
                f"with {num(row['predicted_log_loss_with_params'])}; "
                f"without {num(row['predicted_log_loss_without_params'])}."
                "</div>"
            )
        cards.append(
            f"<div class='evidence-card'><h3>{esc(title)}</h3>{''.join(body)}</div>"
        )
    return "".join(cards)


def environment_case_cards(analysis: dict[str, Any]) -> str:
    phenomena = analysis["phenomena"]
    hardest = phenomena["hardest_environments"][:3]
    unstable = phenomena["largest_cv_selection_gaps"][:3]
    return (
        "<div class='evidence-card'><h3>最难 environments</h3>"
        + "".join(
            f"<p><code>{esc(row['task_id'])}</code><br>"
            f"{esc(row['with_parameter_count']['method'])}: "
            f"{pct(row['with_parameter_count']['three_choice']['accuracy'])}</p>"
            for row in hardest
        )
        + "</div><div class='evidence-card'><h3>CV 选择最不稳定</h3>"
        + "".join(
            f"<p><code>{esc(row['task_id'])}</code><br>"
            f"locked oracle gap {pct(row['selection_gap_to_heldout_oracle'])}</p>"
            for row in unstable
        )
        + "</div>"
    )


def render(analysis: dict[str, Any], *, analysis_hash: str, snapshot_hash: str) -> str:
    corpus = analysis["corpus"]
    phenomena = analysis["phenomena"]
    with_selected = analysis["family_balanced_methods"]["with_parameter_count"][0]
    without_selected = analysis["family_balanced_methods"]["without_parameter_count"][0]
    max_params = method_row(analysis, "with_parameter_count", "max_params_heuristic")
    if max_params is None:
        raise ValueError("Required max-params row is missing")
    b1_macro = phenomena["balanced_b1_anchor"]
    improved = phenomena["parameter_count_ablation"][
        "environments_improved_without_parameter_count"
    ]
    hurt = phenomena["parameter_count_ablation"][
        "environments_hurt_without_parameter_count"
    ]
    tied = phenomena["parameter_count_ablation"][
        "environments_tied_without_parameter_count"
    ]
    max_params_by_family = max_params["per_family"]
    with_selected_ci = with_selected.get("bootstrap_95_ci") or {}
    without_selected_ci = without_selected.get("bootstrap_95_ci") or {}
    champion_ci = phenomena["family_balanced_champion_bootstrap_95_ci"]
    with_champion_ci = champion_ci.get("with_parameter_count") or {}
    without_champion_ci = champion_ci.get("without_parameter_count") or {}
    return f"""<!DOCTYPE html>
<html lang="zh-Hans" data-theme="academic">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArchitectureIQ Setting→Loss 评测报告</title>
<link rel="stylesheet" href="../templates/html_report_template/html模板/css/theme.css">
<link rel="stylesheet" href="../templates/html_report_template/html模板/css/themes.css">
<link rel="stylesheet" href="../templates/html_report_template/html模板/css/base.css">
<link rel="stylesheet" href="../templates/html_report_template/html模板/css/components.css">
<link rel="stylesheet" href="../templates/html_report_template/html模板/css/navigation.css">
<style>
  :root {{ --maxw: 1280px; }}
  .slide {{ padding-top: 40px; }}
  .slide.scrollable {{ scrollbar-gutter: stable; overscroll-behavior: contain; }}
  .hero {{ display:flex; min-height:78vh; flex-direction:column; justify-content:center; }}
  .hero h1 {{ max-width:1000px; font-size:50px; line-height:1.12; }}
  .hero-copy {{ max-width:900px; color:var(--dim); font-size:19px; line-height:1.7; }}
  .stats {{ grid-template-columns:repeat(4,minmax(0,1fr)); }}
  .clean-table {{ width:100%; table-layout:auto; }}
  .clean-table th,.clean-table td {{ vertical-align:top; word-break:break-word; }}
  .table-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
  .evidence-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:16px 0; }}
  .evidence-card {{ border:1px solid var(--border); border-radius:8px; background:var(--bg-panel); padding:14px 16px; min-width:0; }}
  .evidence-card h3 {{ margin:0 0 8px; color:var(--accent); font-family:var(--font-mono); font-size:13px; }}
  .importance-list {{ margin:0; padding-left:20px; }}
  .importance-list li {{ display:flex; justify-content:space-between; gap:10px; margin:6px 0; }}
  .case-line {{ margin:8px 0; line-height:1.55; }}
  .mini {{ color:var(--dim); font-family:var(--font-mono); font-size:11px; overflow-wrap:anywhere; word-break:break-word; }}
  .ok-text {{ color:var(--live); }} .warn-text {{ color:var(--dead); }}
  code {{ overflow-wrap:anywhere; word-break:break-word; }}
  @media (max-width:840px) {{
    .toc {{ display:none; }} .icon-btn {{ left:auto; right:12px; }}
    .slide {{ padding:36px 22px 88px; }} .hero h1 {{ font-size:36px; }}
    .stats,.evidence-grid {{ grid-template-columns:1fr; }}
    .table-scroll .clean-table {{ min-width:920px; }}
    .nav {{ left:12px; right:12px; width:auto; max-width:calc(100vw - 24px); }}
    .nav .dots {{ max-width:86px; overflow:hidden; }}
  }}
</style>
</head>
<body>
<div id="progress"></div><button class="icon-btn" id="fsBtn" title="全屏 (F)">⛶</button><div class="toc" id="toc"></div>
<div class="deck" id="deck">

<div class="slide active" data-title="封面"><div class="hero">
  <div class="act-label">ArchitectureIQ · Setting → Loss</div>
  <h1>新 Setting 的 Loss 可预测到什么水平</h1>
  <div class="hero-copy">输入是 architecture、optimizer、loss-function 与 budget condition；监督目标来自执行生成代码后保存的 <code>mean_loss</code>，模型拟合 <code>log(mean_loss)</code>。报告先逐 environment 评测，再做 family macro，最后做三类 family 等权汇总。</div>
  <div class="badge-line"><span class="badge">{corpus['n_environments']} environments</span><span class="badge blue">{corpus['n_selected_settings']:,} executed settings</span><span class="badge live">{corpus['n_validation']:,} locked validation</span><span class="badge">{corpus['n_benchmark_eligible']:,} benchmark-eligible</span><span class="badge">old-60 仅外部附录</span></div>
</div></div>

<div class="slide scrollable" data-title="状态与指标">
  <div class="act-label">Status first · Metrics first</div><h1>主对象是已完成 GT 的新 settings，不是老 60 题</h1>
  <div class="stats"><div class="stat"><div class="stat-num">{corpus['n_selected_settings']:,}</div><div class="stat-label">completed settings</div><div class="stat-sub">{corpus['n_train']:,} train + {corpus['n_validation']:,} locked</div></div><div class="stat"><div class="stat-num blue">log RMSE</div><div class="stat-label">选模与拟合</div><div class="stat-sub">train-only 5-fold CV 选模型</div></div><div class="stat"><div class="stat-num blue">Rank</div><div class="stat-label">排序能力</div><div class="stat-sub">Spearman / Kendall / concordance</div></div><div class="stat"><div class="stat-num live">3-choice</div><div class="stat-label">下游选优</div><div class="stat-sub">同 environment 全三元组与 gap≥0.05</div></div></div>
  <table class="clean-table"><thead><tr><th>层级</th><th>指标</th><th>聚合规则</th><th>状态</th></tr></thead><tbody><tr><td>Environment</td><td>raw/log MAE、RMSE、R²；rank；choice；regret</td><td>每个 dataset/budget/batch 独立</td><td class="ok-text">主评测</td></tr><tr><td>Family</td><td>environment-macro log RMSE、Spearman、choice</td><td>Bigram / Multi / Uni 分开</td><td class="ok-text">可比较</td></tr><tr><td>Overall</td><td>三类 family 等权 macro</td><td>禁止用 5,671 行 row-micro 掩盖 B2 的 Uni 偏置</td><td class="ok-text">主 headline</td></tr><tr><td>Raw loss</td><td>CE 或 MSE</td><td>只在同 environment 内报告，不跨数据集混合</td><td>尺度隔离</td></tr><tr><td>未完成 GT</td><td>13 个尚无完整 export 的 B2 环境</td><td>不进入训练、验证或分母</td><td class="warn-text">排除</td></tr></tbody></table>
  <div class="callout"><div class="callout-title">样本单位</div><p>这里的一条样本是一组 executed candidate setting，不是一个 <code>question.json</code>。三选指标是在同一 environment 的 locked-validation settings 内枚举候选三元组，由预测 loss 取最小值；它是下游选优诊断，不替代 loss 回归主任务。</p></div>
</div>

<div class="slide scrollable" data-title="核心结果">
  <div class="act-label">Executive results · Equal-family macro</div><h1>复杂特征模型明显优于参数量规则；精确参数量不是主要信号</h1>
  <div class="stats"><div class="stat"><div class="stat-num live">{pct(phenomena['family_balanced_macro_three_choice']['with_parameter_count'])}</div><div class="stat-label">各 family CV champion · 含参数量</div><div class="stat-sub">env-bootstrap 95% CI {pct(with_champion_ci.get('lower'))}–{pct(with_champion_ci.get('upper'))}</div></div><div class="stat"><div class="stat-num blue">{pct(phenomena['family_balanced_macro_three_choice']['without_parameter_count'])}</div><div class="stat-label">各 family CV champion · 无参数量</div><div class="stat-sub">95% CI {pct(without_champion_ci.get('lower'))}–{pct(without_champion_ci.get('upper'))}</div></div><div class="stat"><div class="stat-num">{pct(with_selected['family_macro']['three_choice_accuracy'])}</div><div class="stat-label">统一算法 · {esc(with_selected['method'])}</div><div class="stat-sub">train-CV 选择；95% CI {pct(with_selected_ci.get('lower'))}–{pct(with_selected_ci.get('upper'))}</div></div><div class="stat"><div class="stat-num">{pct(without_selected['family_macro']['three_choice_accuracy'])}</div><div class="stat-label">统一算法无参数 · {esc(without_selected['method'])}</div><div class="stat-sub">train-CV 选择；95% CI {pct(without_selected_ci.get('lower'))}–{pct(without_selected_ci.get('upper'))}</div></div><div class="stat"><div class="stat-num">{pct(max_params['family_macro']['three_choice_accuracy'])}</div><div class="stat-label">只选最大参数量</div><div class="stat-sub">zero-fit ranking rule</div></div><div class="stat"><div class="stat-num">{corpus['n_validation_triples']:,}</div><div class="stat-label">validation triples</div><div class="stat-sub">CI 按 environment bootstrap，不把 triples 当独立样本</div></div></div>
  <div class="table-scroll"><table class="clean-table"><thead><tr><th>Family</th><th>含参数量 champion</th><th>log RMSE macro</th><th>Spearman macro</th><th>三选 macro</th><th>无参数量 champion</th><th>无参数量三选</th></tr></thead><tbody>{family_rows(analysis)}</tbody></table></div>
  <div class="callout"><div class="callout-title">Selection discipline</div><p>统一算法按三 family 的 train-CV log RMSE 等权选择，不看 locked-validation 正确率；每个 family 的 champion 也只由该 family 的 train CV 决定。B1 的 9 个平衡环境单列为含参数量 {pct(b1_macro['with_parameter_count'])}、无参数量 {pct(b1_macro['without_parameter_count'])}；completed snapshot 更大，但 B2 当前完成顺序偏向 univariate，因此不能只看 row-micro。</p></div>
</div>

<div class="slide scrollable" data-title="逐数据集">
  <div class="act-label">Environment first</div><h1>{corpus['n_environments']} 个 dataset / budget / batch 环境逐项评测</h1>
  <div class="subtitle">每行模型仅使用该 environment 的 train rows 选超参数；locked validation 不参与选择。最后一列为“删除参数量后的三选变化”。</div>
  <div class="table-scroll"><table class="clean-table"><thead><tr><th>Environment</th><th>含参数量 champion</th><th>raw RMSE</th><th>log RMSE</th><th>Spearman</th><th>三选</th><th>gap≥0.05</th><th>log regret</th><th>无参数量 champion</th><th>无参数量三选</th><th>Δ</th></tr></thead><tbody>{environment_rows(analysis)}</tbody></table></div>
</div>

<div class="slide scrollable" data-title="全部方法">
  <div class="act-label">22 with params · 18 without</div><h1>所有机器学习方法在同一 locked protocol 下比较</h1>
  <div class="subtitle">表中是三 family 等权 macro。参数专用方法在无参数量轨道为空；方法按统一名称配对，不能把不同 family 的原始 CE/MSE 直接平均。</div>
  <div class="evidence-grid"><div class="evidence-card"><h3>零拟合 / 低容量</h3><p>constant mean、最大参数量、optimizer×LR lookup；params OLS/Ridge/polynomial 只看手算参数量。</p></div><div class="evidence-card"><h3>线性特征学习</h3><p>compact/full OLS、Ridge、ElasticNet、polynomial Ridge。区别是输入特征集合、正则化和是否显式加入二阶交互。</p></div><div class="evidence-card"><h3>非线性模型</h3><p>shallow tree、Random Forest、ExtraTrees、HistGB、GradientBoosting、RBF-SVR、MLP、XGBoost；全部只在 train folds 选超参数。</p></div></div>
  <div class="table-scroll"><table class="clean-table"><thead><tr><th>方法</th><th>train CV·含</th><th>train CV·不含</th><th>三选·含</th><th>三选·不含</th><th>log RMSE·含</th><th>log RMSE·不含</th><th>Spearman·含</th><th>Spearman·不含</th></tr></thead><tbody>{condition_table_rows(analysis)}</tbody></table></div>
</div>

<div class="slide scrollable" data-title="特征输入">
  <div class="act-label">Feature contract</div><h1>数值 condition 与类别 condition 如何进入同一个模型</h1>
  <div class="evidence-grid"><div class="evidence-card"><h3>数值 condition</h3><p><code>log10(lr)</code>、weight decay、momentum/betas、budget、batch size、loss-specific scalar（如 lambda）；MLP depth/width/residual/activation统计；Transformer d_model/d_ff/layers/heads 与比例。含参数量轨道额外使用唯一一列 <code>derived.log_total_params</code>。</p></div><div class="evidence-card"><h3>类别 condition</h3><p>optimizer type、loss id、model type、逐层 activation 等字符串由 <code>DictVectorizer</code> 在每个训练 fold 内学习 one-hot。未见类别不会提前展开词表。</p></div><div class="evidence-card"><h3>混合方式</h3><p>数值列与 one-hot 列进入同一个 dense matrix；常量列在 fold 内删除并标准化。线性模型依靠显式 <code>optimizer × log10(lr)</code>，树/XGBoost/MLP 学习非线性交互。</p></div></div>
  <table class="clean-table"><thead><tr><th>Feature set</th><th>包含内容</th><th>用于</th></tr></thead><tbody><tr><td><code>params</code></td><td>仅 log_total_params</td><td>OLS / Ridge / polynomial 参数量基线</td></tr><tr><td><code>optimizer_lr</code></td><td>参数量（含轨道）、optimizer、log10(lr)、显式交互</td><td>lookup / Ridge</td></tr><tr><td><code>compact</code></td><td>工程化 optimizer、budget、loss、architecture 摘要</td><td>线性、浅树、GB、SVR</td></tr><tr><td><code>full</code></td><td>compact + setting 中每个原始标量与列表位置</td><td>RF、ExtraTrees、HistGB、MLP、XGBoost</td></tr></tbody></table>
  <div class="callout"><div class="callout-title">参数量来源与边界</div><p>“手动参数量”以实际 GT 使用的 generated Model 执行 <code>parameters().numel()</code>，再与 registry 构建模型交叉核验；模型只接收 <code>log_total_params</code> 一列，避免 total/trainable 重复加权。无参数量轨道删除这列，但仍保留 width、depth、d_model 等容量代理。family-pooled 模型不接收任意 dataset ID；因此逐 environment 指标是首层证据，pooling 结果是共享规律测试。</p></div>
</div>

<div class="slide scrollable" data-title="参数量消融">
  <div class="act-label">Controlled ablation</div><h1>参数量有局部价值，但不是稳定的单调规律</h1>
  <div class="stats"><div class="stat"><div class="stat-num">{improved}/{corpus['n_environments']}</div><div class="stat-label">删除后反而提升</div><div class="stat-sub">environment CV champion</div></div><div class="stat"><div class="stat-num">{hurt}/{corpus['n_environments']}</div><div class="stat-label">删除后下降</div><div class="stat-sub">另有 {tied} 个完全持平</div></div><div class="stat"><div class="stat-num">{pct(phenomena['family_balanced_macro_three_choice']['with_parameter_count'])}</div><div class="stat-label">含参数量 champion</div><div class="stat-sub">三 family 等权</div></div><div class="stat"><div class="stat-num">{pct(phenomena['family_balanced_macro_three_choice']['without_parameter_count'])}</div><div class="stat-label">无参数量 champion</div><div class="stat-sub">同 protocol</div></div></div>
  <div class="evidence-grid"><div class="evidence-card"><h3>删除后提升最大</h3>{''.join(f"<p><code>{esc(row['task_id'])}</code> {pct(row['three_choice_delta_without_minus_with'])}</p>" for row in phenomena['parameter_count_ablation']['largest_improvements_without_parameter_count'])}</div><div class="evidence-card"><h3>删除后下降最大</h3>{''.join(f"<p><code>{esc(row['task_id'])}</code> {pct(row['three_choice_delta_without_minus_with'])}</p>" for row in phenomena['parameter_count_ablation']['largest_drops_without_parameter_count'])}</div><div class="evidence-card"><h3>解释</h3><p>同一列在不同 dataset 上既可能帮助，也可能诱导错误容量先验。应把 parameter count 当作普通特征，而不是默认决策规则。</p></div></div>
  <div class="table-scroll"><table class="clean-table"><thead><tr><th>Family</th><th>含参数量 champion</th><th>三选</th><th>无参数量 champion</th><th>三选</th><th>Δ 无−含</th></tr></thead><tbody>{''.join(f"<tr><td>{esc(row['family'])}</td><td>{esc(row['with_parameter_count_method'])}</td><td>{pct(row['with_parameter_count_accuracy'])}</td><td>{esc(row['without_parameter_count_method'])}</td><td>{pct(row['without_parameter_count_accuracy'])}</td><td>{pct(row['delta_without_minus_with'])}</td></tr>" for row in phenomena['parameter_count_ablation']['by_family'])}</tbody></table></div>
  <div class="callout"><div class="callout-title">Capacity shortcut 没有迁移</div><p>只选最大参数量的三选正确率为 Bigram {pct(max_params_by_family['bigram_lm']['within_environment']['macro']['three_choice_accuracy'])}、Multi {pct(max_params_by_family['multivariate_regression']['within_environment']['macro']['three_choice_accuracy'])}、Uni {pct(max_params_by_family['univariate_regression']['within_environment']['macro']['three_choice_accuracy'])}，三 family 等权仅 {pct(max_params['family_macro']['three_choice_accuracy'])}。这显著低于旧 60 题上的 66.7%，说明旧题的容量捷径不能当作新 settings 的一般规律。</p></div>
</div>

<div class="slide scrollable" data-title="现象与 Case">
  <div class="act-label">Case-driven analysis</div><h1>最大误差切片集中在 optimizer × LR 与极端训练区间</h1>
  <div class="evidence-grid">{environment_case_cards(analysis)}{case_cards(analysis)}</div>
  <div class="table-scroll"><table class="clean-table"><thead><tr><th>切片</th><th>样本数</th><th>XGBoost log MAE</th></tr></thead><tbody>{''.join(f"<tr><td>LR {esc(row['group'])}</td><td>{row['n']}</td><td>{num(row['log_mae'])}</td></tr>" for row in phenomena['xgboost_error_by_lr_band'])}{''.join(f"<tr><td>{esc(row['group'])}</td><td>{row['n']}</td><td>{num(row['log_mae'])}</td></tr>" for row in phenomena['xgboost_error_by_optimizer'])}</tbody></table></div>
</div>

<div class="slide scrollable" data-title="特征现象">
  <div class="act-label">XGBoost feature importance</div><h1>模型学到的首要规律因 family 而异</h1>
  <div class="evidence-grid">{feature_cards(analysis)}</div>
  <div class="callout"><div class="callout-title">阅读限制</div><p>这里是 fitted-tree impurity importance，只用于解释候选规律，不是因果效应。重复编码的 raw/log 列会分摊 importance；更严格的 permutation/SHAP 应作为下一轮解释实验。</p></div>
</div>

<div class="slide scrollable" data-title="有效性与来源">
  <div class="act-label">Validity · Provenance</div><h1>哪些结论可以说，哪些暂时不能说</h1>
  <table class="clean-table"><thead><tr><th>项目</th><th>状态</th><th>说明</th></tr></thead><tbody><tr><td>GT</td><td class="ok-text">可追溯</td><td><code>write_candidate → run_ground_truth → results/summary.json</code>；本报告不重算 GT。</td></tr><tr><td>Snapshot</td><td class="ok-text">冻结</td><td>{corpus['n_environments']} 环境、指纹全局唯一；manifest SHA-256 <code>{esc(snapshot_hash)}</code>。</td></tr><tr><td>Selection</td><td class="ok-text">locked</td><td>超参数只看 train-only CV；validation 不参与选模。held-out oracle 只作答案后诊断。</td></tr><tr><td>B2</td><td class="warn-text">partial</td><td>只纳入已有完整且逐文件 hash 校验通过的 8 个环境；其余 {analysis['status']['excluded_environment_count']} 个未完成环境完全排除，不能称为完整 B2。</td></tr><tr><td>Remote</td><td class="ok-text">audited</td><td><code>{esc(analysis['status']['remote_audit']['ref'])}</code> 确认 wide_v2 是当前主项目；远程没有新增完成分数。</td></tr><tr><td>Overall</td><td class="warn-text">macro only</td><td>当前完成顺序偏向 univariate，总体 headline 使用三 family 等权 macro。</td></tr><tr><td>Convergence</td><td class="warn-text">warning</td><td>ElasticNet/MLP 的部分搜索出现收敛 warning；运行完成并保留 checkpoint，但不据单行作强结论。</td></tr><tr><td>Old 60</td><td>external appendix</td><td>只用于检验 setting→loss 能否迁移为 choice；不是本报告主训练集或主指标。</td></tr></tbody></table>
  <p class="mini">Analysis SHA-256: {esc(analysis_hash)} · Snapshot SHA-256: {esc(snapshot_hash)}</p>
</div>

<div class="slide" data-title="旧60附录"><div class="hero"><div class="act-label">External appendix</div><h1>老 60 题只回答迁移问题</h1><div class="hero-copy">旧 60 的 54/60、无参数量 51/60、最大参数量 40/60 与 GPT context 行为仍保留在 <a class="source-link" href="0710_gpteval.html">0710_gpteval.html</a>。它们不再定义新 setting→loss 报告的主问题。</div></div></div>

</div>
<div class="nav"><button class="btn" id="prev">← 上一步</button><div class="dots" id="dots"></div><div style="display:flex;align-items:center;gap:18px;"><span class="counter" id="counter">1 / 1</span><button class="btn" id="next">下一步 →</button></div></div>
<div id="notesPanel"></div><div id="helpOverlay"><div class="panel"><h3>快捷键</h3><div class="row"><span><kbd>←</kbd> <kbd>→</kbd></span><span>上一步 / 下一步</span></div><div class="row"><span><kbd>↑</kbd> <kbd>↓</kbd></span><span>长页内滚动</span></div><div class="row"><span><kbd>F</kbd></span><span>全屏</span></div></div></div>
<script src="../templates/html_report_template/html模板/js/engine.js"></script>
</body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/wide_v2_completed_gt_snapshot.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render(
            load_json(args.analysis),
            analysis_hash=sha256(args.analysis),
            snapshot_hash=sha256(args.snapshot),
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
