# ArchitectureIQ 出题状态与计划 · Status & Plan

> 本文档登记出题工作的**当前状态、已完成事项、验证结果、以及后续计划**。它是运营视角的总账；出题的**操作逻辑与旋钮**见 [question_generation.md](./question_generation.md)，稳定架构与不变量见 [../AGENTS.md](../AGENTS.md)。
>
> 最近更新：2026-08-02（AutoResearch 转移评测启动：work-tree + propose-loop + plan_light 已实现并跑通首轮 pilot）。
> 历史：2026-07（大预算批次 + label-noise 难度工程 + 惊讶反馈/推荐核心）。

---

## 1. 一页速览

| 维度 | 现状 |
|------|------|
| **本地题库总量** | **261 道有效题**（architecture_only 95 / mixed 133 / optimizer_only 33），来自 20 个实例；这是 gitignored `data/` 的本地盘点，不是线上发布数 |
| ├ 干净题 | 227 道（无噪声光滑 target） |
| └ 带噪题 | 34 道（label-noise，难度更高） |
| **当前仓库 attested 发布包** | **60 道题**（3 family × 20），但尚未证明新 bundle 已在线上部署；线上/推荐候选池只能使用实际 runtime manifest 验证通过的 release，不能把本地 261 题默认为已发布 |
| **数据集实例** | 27 个（22 干净 + 5 带噪），3 个 family |
| **难度（可测量）** | 261 题中 **130 道击败启发式集成**（50%）；盲评 per-solver 正确率 干净 54% / 带噪 46%（随机 25%） |
| **惊讶/推荐** | 本地已有 tie-aware 冷启动 catalog、reveal 后 reaction、权威聚合 SQL，以及只在 attested release 内运行的 Beta + 20% ε-greedy `Next`；每次展示记录 policy/decision/propensity。尚无 hosted 验收、跨用户后验 serving 或 A/B 证据 |
| **有效性** | 全部题 GT 来自真实执行 10-seed 训练，通过 gap/win-rate/非重叠显著性；带噪题 test=真函数 |
| **核心不变量** | spec→code→run→GT 对齐，全程未破坏；`data/` gitignored |
| **测试** | 941 passed |
| **文档** | 本文件 + `question_generation.md`（操作逻辑）+ AGENTS.md（架构） |

**结论**：作为 pipeline 与高可信 benchmark 均已合格；难度已从"中等"推进到"部分题连专家盲解都系统性出错"。仍有明确的提难空间（见 §5）。

### 惊讶值的三层统一术语（规划）

- **`intrinsic_surprise_proxy`**：从存量 GT、反启发式和 blind-solver 失败派生的离线冷启动特征。当前 `tools/difficulty/score_questions.py` 只能算这一层的雏形。
- **`observed_surprise_rate`**：用户在答案/排名揭晓后显式点击“有惊讶 / 没有惊讶”得到的版本化统计，应使用平滑后验而不是小样本裸比例。
- **`predicted_personal_surprise`**：推荐策略结合前两者、当前 session 表现与题型新颖度，对“这个用户做这道题会不会惊讶”的预测。

三者不得混用。**惊讶 ≠ 答错 ≠ 点赞**：答错可能是疏忽，惊讶也可能是困惑或负反馈；如要优化“更愿意继续”或点赞，必须另行采集 like/继续行为并作为 guardrail，不得用正确性或 surprise 代替。

---

## 2. 本阶段完成事项（2026-07）

### 2.1 基线修复（正确性）
| 修复 | 文件 | 影响 |
|------|------|------|
| failed-seed 候选排除出题池 | `questions/generator.py::eligible_candidate_paths` | 杜绝"赢家有 seed 发散到 1e31、丢弃后仍判赢"与题面"10 seeds mean"矛盾 |
| bigram 物化 tensor 真正来自执行 `synthesize.py` | `families/bigram_lm/family.py::materialize` | 修复 single-source-of-truth 不变量违背 |
| bigram/正则 loss 采样 lambda（`_l1/_l2` 结尾） | `candidates/generator.py`, `interactive.py` | 修复 `None * l1` 渲染，救活 bigram loss 题 |
| 修 merge 带入的 failing test | `tests/test_question_inspector.py` | train/test 形状不必相等（bigram 800/200） |

### 2.2 大预算清洁批次
- **226 候选**（预算 40960–81920，旧主集的 2–20×），222 ok / 4 excluded → **42 道新题**。
- 补出上一个 Agent 中断的孤儿题（mvar_e3e90e 的 8 道 mixed）。
- 实测 **capacity shortcut 从旧主集 66.7% 降到 24%**（arch-only 35%）：大预算让结构合适的小网追平大网。

### 2.3 难度工程（本阶段重点）
把"难度"与"有效性"分成两个正交轴，围绕"**GT 稳定 × 盲解-启发式-集成 趋近随机**"这个北极星工作：

1. **难度过滤器** `tools/difficulty/score_questions.py`（零训练）：validity + anti-heuristic + blind-ensemble-wrong 三分。让"够不够难"可测量。
2. **label noise** `--noise-std`：只对训练标签加高斯噪声，测试集=真函数。解锁 double descent、Adam-vs-SGD 泛化、正则化价值；干掉"选 Adam/选最大网"捷径。严格遵守不变量（噪声进 content-addressed id、只改 synthesize.py、prompt 诚实、GT 路径不变、加回归测试）。
3. **`profiles/v2.yaml`**：lr 网格上探到 0.1（edge-of-stability），更宽 wd/lambda，更大 budget；v1 保持不动以保复现。
4. **带噪批次** 254 候选 → 34 道新题，覆盖 arch/optimizer/loss/mixed。

### 2.4 效率工具
- `tools/batch_generate/parallel_sets.py`：多进程、每进程单线程（`OMP_NUM_THREADS=1`）打满 10 核。实测这些 tiny 模型**单线程最快**（9.2s vs 6 线程 14.6s）；9 workers 约 832% CPU。**不重实现任何生成逻辑**，只并行化 GT 循环。

---

## 3. 验证结果（用 subagent 盲评 + 难度过滤器）

### 3.1 有效性（回应"会不会像符号回归只有平凡/混沌"）
清洁批次 13 道代表题盲评：**12/13 答案可由 architecture reasoning 推出（0 道"不可"）**，avg quality 3.77/5。—— 有效性成立，不是靠噪声/运气。

### 3.2 难度（回应"可推理 ≠ 太简单，要专家也叫不准"）
带噪批次 10 道最难题，**3 独立 solver 盲解**：

| 指标 | 干净批次 | 带噪批次 |
|------|----------|----------|
| per-solver 正确率 | 54% | **46%**（随机 25%） |
| architecture-only solver 正确 | 4/8 | **0/9（每个专家全被骗）** |
| 有捷径且捷径给**错误**答案 | 部分 | **10/10** |

judge 逐题确认被证伪的专家捷径：**"选最小抗噪"、"选最大/最有表达力"、"选最正则化"、"选自适应 optimizer/最高 lr" 全部→错**。这是"必须真跑才知道"的机制。实测三个真现象：赢家从不是最大网（double descent）；带噪下 SGD/Adagrad 排前四、Adam 掉到 9–14 名；`mse` 无正则变最差。

结果文件：`tools/batch_generate/_eval_results.json`（清洁）、`_eval2_noisy_hardness.json`（带噪）。

---

## 4. 已知边界与未决问题

| 问题 | 现状 | 原因 |
|------|------|------|
| **loss-only 难过阈** | 带噪后信号方向对了（`mse` 最差、正则更好），但 gap 仅 0.017 < gap_min 0.05 | 需更强噪声（std 0.3–0.5）+ 更高维 target 才能让正则化过阈 |
| **一元带噪题几乎不成题** | target 太易，含噪仍 ~0.002–0.006 打平 | 噪声只在够难/够高维（多元）target 上生效 |
| **SGD-beats-Adam 在一元 gap 太小** | 现象真实但一元上不过阈 | 应搬到多元 |
| **难度过滤器看不到"噪声机制"** | 它只用结构启发式（param/depth/width/opt），对带噪的判定要靠真 solver | 结构 proxy 无法感知 bias-variance；盲评是补充 |
| **`intrinsic_surprise_proxy` 不是用户惊讶** | 当前 hardness 只有 5 个启发式、手工权重，且尚未完整处理 metric direction、并列候选与 release 边界 | 只能作冷启动/人工筛选特征，不能命名为 `observed_surprise_rate` 或宣称会提高留存 |
| **曝光/propensity 只有本地与待部署事件** | `Next` 已写 `question_presented`（policy/decision/mode/source/position/propensity）并进入 browser outbox；尚未 hosted 持久化或形成 A/B 数据 | 线上验收前不得用本地 trace 宣称留存/推荐提升；续做与 like 仍需独立事件 |
| **惊讶已有独立标签，like/继续行为尚无** | 本地 reveal 后 reaction 已与 answer/comment 分离；还没有独立 like 或 continuation 事件 | 不从 comment 文本、答错或 surprise 自动推断喜欢/留存 |
| **跨预算题（multi-set）未系统验证** | budget/batch axis 判定可能把 schedule 不同的 choices 误标为单轴题 | 正式发布跨预算题前需先修/验证 |
| **bigram loss_only** | 无噪声下 gap≈0（同回归） | 需为 bigram 设计放大 loss 差异的机制 |

---

## 5. 后续计划

### P0 — 建立显式惊讶信号

- **SURPRISE-001（本地已实现，待 hosted 验收）**：答案和真实排名 reveal 后显示“出乎意料 / 符合预期”；严格 `question_reaction_submitted` 使用稳定 event ID，复用 session trace、browser outbox、单事件/批量上传和 409 isolation；`19000` forward migration 与 Edge 校验已具备。

### P1 — 权威惊讶统计与离线/session 选题

- **SURPRISE-002**：在服务端权威 release/question registry 上聚合 yes/no/count/response-rate，按 family/type 层级先验计算 `observed_surprise_rate` 后验；unknown/mismatch 事件只进 raw/quality，不回退使用客户端自报维度。
- **RECO-001（本地 session serving 已实现，出题排序/hosted 后验待接）**：manifest-only catalog 对当前 60 题做 validity/failed-seed 硬门与 tie-aware proxy；`Next` 排除本 attempt 已答/当前题、避免连续同 family，并用默认 20% ε-greedy 选择。每次展示记录精确 mixture propensity；失败退回顺序 `Next`。

### P2 — 个性化惊讶预测与实验

- **RECO-002**：数据充足后用分层 Beta-Binomial/Thompson Sampling 结合离线 proxy、显式惊讶、近期正确率和新颖度估计 `predicted_personal_surprise`；保留 15%–20% exploration，通过带 propensity 的 A/B 测试评估，读取服务失败时 fail safe 到现有 `Next`。

### P0 — 提高难度产出率（延续本阶段）
1. **更强噪声 + 高维 target 出 loss 题**：`--noise-std 0.3~0.5` + multivariate dim≥5，让 L2/L1 正则化真正过 gap_min；或对 loss 题单设更低 gap_min（family-specific significance）。
2. **double descent 专用 set**：固定除 width 外全部，**密集扫 width**（16→512），配噪声，直接呈现 test_mse 随容量的非单调峰。这是最"教科书级反直觉"的题型。
3. **SGD-vs-Adam 泛化题搬到多元**：带噪多元 optimizer-only，把"永远选 Adam→接近最差"做成稳定过阈的题。

### P1 — 消除已知捷径 / 修跨预算
4. **optimizer 题加对抗 distractor**：故意放调坏的 Adam + 调好的 SGD，逼真正的 optimizer-dynamics 推理（清洁批次里 optimizer 题 shortcut-risk 偏高）。
5. **修跨预算题型判定**：`_budget_field` / `choices_compatible` 对 batch/budget 轴的处理，发布跨预算题前先验证。

### P2 — 新数据集家族（最大工作量，最贴内核）
按 [question_generation.md §7](./question_generation.md) roadmap：
6. `spectral_regression`（ReLU-MLP / SIREN / Fourier-MLP）测 spectral bias——最易复用现有 regression pipeline。
7. `structured_memory_lm`（TCN / GRU / local-vs-global attention）测 receptive field / 记忆。
8. 之后 `translated_motifs`(CNN)、`anova_interaction`、`set_relations`(DeepSets)。

### 持续 — 发布前固定动作
9. 每批新题跑难度过滤器 + shortcut baseline（最大参数量/最深/最宽/固定 Adam），确保命中率不显著高于随机。
10. 每实例限 1–3 题，避免过拟合固定 test split；正式 GT 用独立 confirmation seeds/data 复核。

### 惊讶/推荐最小验收合同（部分本地实现）

1. **采集语义**：未提交答案时不得评价惊讶；reveal 后 yes/no 各生成严格 boolean reaction。同一 session/attempt/release/question/version/reaction 在 UI 中只能生成一个有效评价，重试保持同 event ID 并幂等，原始 append-only 事件仍可审计。
2. **可恢复上传**：未配 endpoint 时 reaction 仍进 session trace/browser outbox，可下载、恢复、批量重试；单事件、500-events/1-MiB 分批、严格 receipt 和同 ID 内容冲突 409 的现有语义不得退化。
3. **权威统计**：Reports 按 release/question version 显示 rating/yes/no/response counts、平滑后验与筛选后守恒；未登记 release/question 不进权威统计，但必须留在 raw/quality 可追踪。
4. **推荐安全性**：只能返回 runtime-attested manifest 中的题；不返回已答、无效、未登记或不同 release 题。同 policy snapshot + seed 可复现，tie 不依赖字母/文件路径，metric direction 正确，远端不可用时回退顺序 `Next`。
5. **无泄漏与可评估性**：动态分数不写入 `question.json`、GT 或 prompt，答题前 UI 不显示 winner/hardness/surprise 信号。每次策略选题必须记录 exposure、policy version、decision ID 和 selection propensity；无该证据不得宣称推荐提升。
6. **产品成功口径**：A/B 主指标是预先定义的 `observed_surprise_rate`；answer/continue rate、accuracy、family/type 覆盖与独立 like 是 guardrail。只有达到预先设定的样本量且区间/后验证据支持时，才能称“更好”。

---

## 6. 交付物索引

| 类型 | 路径 |
|------|------|
| 出题操作逻辑文档 | `docs/question_generation.md` |
| 本状态计划文档 | `docs/STATUS_AND_PLAN.md` |
| 惊讶/推荐产品决策与 Backlog | `docs/product-development.md`（SURPRISE-001/002、RECO-001/002） |
| 惊讶 reaction 协议与 UI | `tools/question_inspector/feedback.py`、`app.py`、`supabase/migrations/20260712019000_question_reactions.sql` |
| 推荐核心 | `tools/question_inspector/surprise_recommender.py`（冷启动 Beta、后验更新、可审计 ε-greedy） |
| Attested 冷启动 catalog | `tools/question_inspector/surprise_catalog.py`（只读 manifest 内题目、tie-aware、零重训/零写回） |
| 权威 Surprise SQL | `supabase/migrations/20260712020000_feedback_surprise_report.sql`（首条有效票、Beta(1,1)、质量守恒） |
| 难度过滤器 | `tools/difficulty/score_questions.py`（输出 `_scores*.json`） |
| 并行 GT 生成器 | `tools/batch_generate/parallel_sets.py` |
| 批次 plan / index | `tools/batch_generate/batch_plan.json`, `batch2_noisy_{v1,v2}.json`, `_batch*_index.json` |
| 盲评结果 | `tools/batch_generate/_eval_results.json`（清洁）, `_eval2_noisy_hardness.json`（带噪） |
| v2 profile | `profiles/v2.yaml` |
| label-noise 实现 | `families/{univariate,multivariate}_regression/family.py`, `cli.py`, `datasets.py`, `prompts/formatters.py` |
| 回归测试 | `tests/test_new_families.py::test_label_noise_train_only_and_reproducible` |
| AutoResearch 评测计划 | `docs/plan-autoresearch-eval.md`（工作树/L1/L2 设计） |
| 评测协议与题集 | `docs/eval-sets.md`（select_best / two_choice / propose / L1 / L2） |
| L2 pilot 报告（5 模型同树对比） | `docs/reports/AUTORESEARCH_PILOT_2026-08-02.md` |
| 题集总览 HTML（题目预览/Protocol/成绩） | `docs/reports/QUESTIONS_OVERVIEW_2026-08-03.html`（`tools/report_questions.py` 可复现生成） |
| L2 逐 run 报告 | `artifacts/autoresearch_report.html` + `artifacts/autoresearch_runs/`（summary + history.jsonl） |

### 本阶段改动的源文件（15 个，+221/−44）
`candidates/generator.py`, `questions/generator.py`, `ground_truth/runner.py`, `cli.py`, `datasets.py`, `families/{univariate,multivariate,bigram}*/family.py`, `interactive.py`, `prompts/formatters.py`, `prompts/templates/dataset/univariate_regression.md`, `tools/question_inspector/prompt_format.py`, 及对应测试。

---

## 7. 复现关键命令

```bash
AIQ=".venv/bin/architecture-iq"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 带噪实例（测试集=真函数）
$AIQ create-dataset --profile v2 --family multivariate_regression --seed 311 --input-dim 4 --noise-std 0.15

# 并行大批量生成候选 GT（打满 CPU）
.venv/bin/python tools/batch_generate/parallel_sets.py --plan <plan.json> --workers 9 --profile v2

# 出题
$AIQ generate-question <dataset> <set> --profile v2 --num-questions 5 --num-choices 4 --seed 50001

# 测难度
python tools/difficulty/score_questions.py --top 25
```

---

## 8. 2026-07-31 周会：题目实例/评测实例分离 + 评测落地

> 来源：飞书 “ArchitectureIQ 0714” 文档 → 7月31日 Weekly
> （[wiki 链接](https://gcn73xn49fre.feishu.cn/wiki/Fj8sw4saiiWICFkbg3gcReLMn6e)；
> 2026-07-31 与 2026-08-02 两轮均以 lark-cli 用户身份验证可读可写）。
> 目标产物：一个 benchmark + 一篇 paper，两周 + 两个 section。
> 对应架构文档：`plan-v2.md`；评测协议：`PROTOCOLS.md`。

### 8.1 总原则

- 题目实例与评测实例独立：
  - **题目实例 = 数据集 + demo settings**（demo settings 是一组 config JSON）；其余全部 offload 给评测端。
  - **评测实例 = 把 settings 组合成选择题 / config 修改题（架构生成题）**。
- Demo setting 是否筛选、如何筛选：先做数据质量与 demo 质量初筛；demo settings 可直接在 autoresearch 过程中产生。

### 8.2 评测实例（本轮重点，郭绍阳）

- 题型：
  1. **选择题**：选择哪个 setting loss 最低（可给 few shot）。
  2. **config 修改题**：约定闭集可调参数，模型输出 json config，目标最小化 loss（应给 few shot；实际做实验也有参考）。
- 限制：模型 size 与 flops 不得超过 demos 中最大的 1.1 倍。
- 筛选器（唐晨成）放评测端：删 ill setting；过于接近/悬殊的 setting 不组进同一题；可用指标让 LLM 判断题目是否合理。
- 先只支持**给大模型的评测**；meta-model（TabPFN）评测作为第三 section 后续再做。

### 8.3 题库实例

- 题库扩展目标 ~100 种题型；总结出好的题目形式后进入自动化阶段。
- 可用 Kaggle / 已有 benchmark 数据改造为我们的题型，配合 few shot 与 observable 辅助范式。
- 自动生成训练 baseline，避免 ill setting。

### 8.4 存储结构设计（2026-07-31 对齐，列式存储）

> **已实现**（2026-07-31，分支 `shaoyang/local-agent-dev`）：权威设计见 [`docs/backend-storage.md`](./backend-storage.md)，storage API 在 `src/architecture_iq/storage/`，迁移工具 `tools/storage/migrate_data_layout.py`，存量数据已迁移到 `backend/data/`（27 problems / 941 candidates / 940 results / 3 trainers，旧 `data/datasets` 保留待验证）。

- 后端一级目录 `backend/`，二级 `data / generator / eval`；`data` 即“题目实例仓库”，三级起按列组织：

```
backend/
├── data/                                # 题目实例仓库（列式存储）
│   ├── problems/{problem_id}/           # dataset_spec.json + README(介绍文档) + synthesize.py + 物化张量
│   ├── trainers/{trainer_id}/           # 训练脚本（train.py 模板，按 family/trainer 独立）
│   ├── candidates/{problem_id}/         # 该 problem 的 config JSON 闭集（每个 candidate 一个 json）
│   └── results/{problem_id}/{candidate_id}/  # summary.json + curves.npz（+ 可选 ckpt），与 candidates 一一对应
├── generator/                           # 生成套件（与存储系统解耦）：数据集/训练脚本/config 生成 + GT 执行
└── eval/                                # 评测端：组合选择题/config修改题、LLM 评测、后续 meta-model
```

- 列式组织：`problems / trainers / candidates / results` 在第二层并排，题目编号在各自文件夹内；一次可以取出“所有 problem / 所有 configs”，各列体积可以悬殊。
- 原来的一种“数据集实例” → 现在变成一个 `problem` + 一堆 `candidates`；**评测端**用 problem + candidates 组合出题目（选择题 / config 修改题），不再由题目实例自己产出问题。
- 闭合集要求：所有代码设计都包含在题目代码里（config 只描述 `model / optimizer / loss / budget`，渲染与训练代码由 generator 或 trainers 提供，config 不引入新逻辑）。
- demo settings = candidates 的子集（config 元数据加 `role: demo|eval` 标记，供 few-shot 使用）。

### 8.5 现状 → 目标映射与迁移评估（2026-07-31）

| 现状（`data/datasets/...`） | 目标 | 迁移动作 |
|---|---|---|
| `{family}/{dataset_id}/dataset_spec.json` + `synthesize.py` + `train.pt/test.pt` | `data/problems/{problem_id}/` | 目录搬家；ID（content-addressed）不变 |
| `candidates/generator.py` 内嵌 train 模板 + `c_{hash}/train.py` | `data/trainers/{trainer_id}/` | 训练代码从每候选生成改为按 trainer 落盘 |
| `set_{budget}_{axes}_{hash}/c_{hash}/candidate_spec.json` | `data/candidates/{problem_id}/{candidate_id}.json` | 拍平 set 层级；config 只存 JSON |
| `c_{hash}/results/{summary.json,curves.npz}` | `data/results/{problem_id}/{candidate_id}/` | 搬家；路径引用改 ID |
| `questions/run_*/q_*/`（在题目实例内） | 移到 `eval/`（组合+渲染在评测端） | question 只存 candidate_id 引用，不再存路径 |

- **可行性：高**。`candidate_spec.json` 已是闭合 config；ID 全部 content-addressed，搬目录不换 ID；`paths.py` 是唯一集中点。
- **主要成本**：布局被 `src + tools` 共 508 处直接引用，且 `question.json` 里存了相对路径（`candidate_path`）——需要一次协同迁移（storage repository 层 + 引用改 ID + 存量数据迁移脚本 + 测试更新）。
- **解耦方式**：新增 `storage` 层（repository API：写 problem/trainer/candidate、读 result）；`generator` 只通过它写，`eval` 只通过它读 + 写自己的评测产物；双方互不感知对方内部格式。
- **保持不变量**：spec→code→run→GT 一致；GT 只来自执行生成代码；non-repeating candidates；anti-shortcut gates。
