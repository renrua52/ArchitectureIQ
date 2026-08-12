# Plan: AutoResearch 转移评测（Config 工作树）

> 状态：**设计稿 v1**（2026-08-02，分支 `shaoyang/local-agent-dev`）
> 关联：`backend-storage.md`（列式存储）、`eval-sets.md`（题集）、`../AGENTS.md`（不变量）、
> 本地文献调研：`~/Desktop/workdir/LiuComfyResearch/autoresearch/docs/autoresearch_landscape_2026_05.md`、
> `paper_meta/OUTLINE.md`、`CA_project/survey/`。
> 论文定位（用户 8-02）：把 AutoResearch 拆成 **config 状态转移**，benchmark 落脚点是**单次转移**的质量；
> 算力必须小（toy model），核心看**模型对 architecture 的直觉**。

---

## 1. 论文叙事（Why）

AutoResearch / RL 研究很多，但没人能回答："**Agent 的哪一步做得好、哪一步做得差？**"
AutoResearch 的关键载体是 **Config（实验 Settings）的版本管理**：一次 AutoResearch 过程 =
一系列 **config → config 的状态转移**（propose → 实验 → 观察 → 再 propose）。

本 benchmark 的原子单元 = **单次状态转移**：

1. 给一个**好的 base config**（生产场景从好 setting 出发，而不是混沌 setting）。
2. 从 base 扰动出 **candidate configs**，构成**工作树**（work tree）。
3. 树中节点分**点亮（lit，已做实验、loss 已知）**与**未点亮（unlit，待尝试）**。
4. 任务：
   - **判断**：哪个已点亮/候选 config 的 loss 最低（含 base，测"是否值得改"）。
   - **规划**：优先点亮哪些 unlit 节点（决策树上的 few-shot）。
   - **研究**：K 次 propose→实验→观察 的闭环，在预算内逼近最优 config（AutoResearch 评测法）。

**与现有工作树的对照（本地调研结论）**：

| 系统 | 我们的差异 |
|------|-----------|
| AI Scientist v1/v2（2504.08066） | 端到端 tree-search 不可分解；我们原子化"单次转移"并可审计 |
| Karpathy AutoResearch | 无闭集 config 网格、无 GT 可复现；我们有 spec→code→run→GT 不变量 |
| PaperOrchestra / Sibyl | 论文写作流水线，评测用模拟 review；我们测的是"实验决策"本身 |
| MLAgentBench / MLE-bench | 端到端、算力大、步骤不可控；我们 toy compute + 可分解步骤 |
| ASI-Bench / ResearchGym | 隐藏评分开放式研究；我们提供**可见分数**主线 + **隐藏分数**对照变体 |
| FML-Bench / KernelBench 等 | 单领域计算优化；我们跨 family 的架构直觉 |

**结论句（论文用）**：在 toy 预算下，我们首次把 AutoResearch 拆成可逐点评测的
config 状态转移，让"架构直觉"成为一个可测量的能力轴。

---

## 2. 数据模型（复用现有 backend 列式存储）

```
backend/data/
├── problems/{problem_id}/            # 数据集 + README + synthesize.py + 物化张量（已实现）
├── trainers/{trainer_id}/            # 训练脚本（已实现，按 family）
├── candidates/{problem_id}/{candidate_id}.json   # 闭合 config JSON（已实现）
└── results/{problem_id}/{candidate_id}/          # summary.json + curves.npz（已实现）
```

**新增：工作树（work tree）**——不另起存储，作为 `backend/eval/trees/{problem_id}/{tree_id}.json`
的**评测端视图**：

```json
{
  "schema_version": "0.1",
  "problem_id": "mvar_866b4e",
  "metric": "test_mse",
  "base": {"candidate_id": "c_xxx", "loss": 0.1042, "role": "base"},
  "nodes": [
    {"candidate_id": "c_yyy", "edits": ["model.depth 3->4", "optimizer.lr 1e-3->3e-3"],
     "lit": true, "loss": 0.0987, "edge": ["base"]},
    {"candidate_id": "c_zzz", "edits": ["optimizer.type SGD->Adam"],
     "lit": false}
  ],
  "budget_rule": {"params_ratio_max": 1.1, "total_samples_seen": "fixed = base"},
  "few_shot": ["c_yyy", "c_www"]        # 决策树内的 few-shot 节点 id
}
```

- **base 选择**：同一 problem 内 loss 质量好的 setting（默认 loss 排名前 25%–50% 区间的候选，
  避免混沌）；用户宗旨：**base 好，扰动才有意义**。
- **children**：对 base 做 1–2 个 salient 编辑（复用 `questions.py` 的 salient 距离逻辑），
  优先复用已有 GT 的候选（零算力）；不足时闭集内合成新 config。
- **lit 状态**：`results/` 存在 summary.json 即 lit；评测时按任务把部分 lit 节点**隐藏**（视作 unlit）。
- **few-shot 必须来自同一棵树**：base + 2–3 个 lit 邻居（含 loss），不引入树外随机 setting。

---

## 3. 任务层次（L0 → L1 → L2）

### L0 `select_best`（已有，单次判断）
6 选 1（含 base），5 个带 loss 参考锚定选项区间。已实现 + 已跑通（claude 70% / terra 72%）。
→ 作为 L1/L2 的对照组与难度标尺。

### L1 `plan_light`（规划：先点亮谁 / 谁最好）
- **输入**：树视图（base + 部分 lit 节点带 loss + 若干 unlit 节点的 config 摘要，无 loss）。
- **任务 A（rank）**：给 unlit 节点按预测 loss 升序排序（或选 top-k 点亮顺序）。
- **任务 B（best-lit）**：从"含 base 的所有已点亮节点"中选出 loss 最低者（决策树内的 select_best）。
- **评分**：任务 A → 排序的 Spearman ρ / 点亮顺序的后悔值（点亮第一个节点后相对 base 的改进）；
  任务 B → 正确率（与 L0 同一把尺子，但 few-shot 来自决策树）。

### L2 `propose_loop`（AutoResearch 闭环，主任务）
- **输入**：树视图（base + lit 节点带 loss + unlit 节点摘要）+ 闭集约束 + 轮次上限 K。
- **循环**：模型 propose 新 config（JSON，闭集）→ normalize/吸附 → GT（已有结果直接查表，新的才跑）→
  点亮该节点并回显 loss → 下一轮。可选并行：每轮 propose `P` 个。
- **评分**（不只看正确率，看研究成果）：
  - `improve_base`：K 轮后最佳 loss 相对 base 的相对提升。
  - `oracle_gap`：K 轮后最佳 loss 相对**树内 oracle**（全量 lit 后最优）的 gap。
  - `regret@k`：第 k 轮最佳相对 oracle 的后悔曲线。
  - `win_rate_vs_base`：逐 seed 战胜 base 的比例。
  - `budget_ok`：是否全程遵守参数量 ≤ 1.1× base、训练预算固定。
- **隐藏评分变体（对照，呼应 ASI-Bench 思路）**：同一棵树，模型**看不到 lit 节点的 loss**，
  只能看到"该节点已被点亮"。对比可见/隐藏两档，回答"即时反馈是否带来稳定改进"。

### 题量
- 每个 problem 建 3–5 棵树（不同 base / 不同扰动子集），每个 L2 题目 K=5 轮、P=1。
- 单档评测 50 题量级（≈ 10 problem × 5 tree），多模型并发（eval key，50 并发）。

---

## 4. 计算预算（必须小）

| 项 | 预算 |
|----|------|
| 数据集 | 256–1024 样本的 toy 数据（现有 family 即可） |
| 模型 | MLP（≤6 层 ≤256 宽）/ transformer_lm（d_model≤128） |
| 单次 GT | 10-seed，~10–30s CPU（现有 runner，单线程最快） |
| 新增 GT | 只对**树外新 propose 且不在 results 里**的 config 跑；存量候选直接查表 |
| 一轮评测 | 50 题 × 5 轮 × 最多 ~2 个新 config/题 ≈ ≤500 次新 GT ≈ 2–4 核时 |

> 题目主体复用存量 GT（27 problems、~1000 候选已有点亮结果），新算力只花在"模型真的提出新 config"上。

---

## 5. 实现步骤（M0–M5）

- [x] **M0 工作树生成器** `backend/eval/worktree.py`：建树、选 base、选 children、lit/unlit 视图、few-shot 选择。
      —— 已实现 + 单测；预算过滤（同 total_samples_seen、params ≤ 1.1× base）保证每个节点都是合法 move。
- [x] **M1 闭环 runner** `backend/eval/autoresearch.py`：L2 循环（propose→normalize→GT→点亮→记录），
      复用 `score_proposal.normalize_proposal` + `write_candidate` + `run_ground_truth`。
      —— 已实现 + 单测；含 exact-config 点亮（存量结果零算力复用）与新实验 GT 去重。
- [x] **M2 L1 生成器** `backend/eval/plan_light.py`：`plan_light` 题（light-first + rank），与 L0 同评分线。
      —— 已实现；95 题已生成（每棵树 1 题）。
- [x] **M3 并发评测**：复用 `batch_eval.call_llm` 管道，eval key 并发；思考过程全文落盘
      `artifacts/autoresearch_runs/{run_id}/`（history.jsonl 含 prompt/raw/reasoning/loss）。
- [x] **M4 报告**：`backend/eval/report_autoresearch.py` 生成 HTML（每轮 propose/观察/思考过程 + 聚合表）。
- [ ] **M5 文献任务扩展**：按本地调研（OpenML/Kaggle/已有 bench 封装）下载多样化 toy 任务，
      迁移成 problem 实例；隐藏评分对照实验。

### 5.1 首轮 pilot 结果（2026-08-02 夜，gpt-5.6-luna）

- L2 propose-loop：10 棵树 × 2–5 轮全部跑通；多棵明显提升（+2.8% ~ +79.8%），
  3 棵发现比树内初始 oracle 更好的 config（oracle_gap 为负 = 击败已知最优）。
- 模型行为观察：gpt-5.6-luna 倾向于重复 propose 相似 config（近重复轮次），
  但第 3–5 轮开始探索新架构（如 depth1 width128 + Adam）；prompt 已加"wasted round"警告。
- **发现并修复 2 个 bug**（均已有回归测试）：
  1. `run_new_gt` 继承 base 的 candidate_id 导致新实验覆盖 base 存储 → 修复为剥离派生键后重新哈希；
     存量 8 个被污染的 base 已从 `data/datasets/` 原样恢复。
  2. 存量 candidate_id 的哈希来源与 `short_hash(spec)` 不一致 → 改用**规范后 config 深度相等匹配**
     （`find_stored_candidate`，忽略 candidate_id/problem_id/files 等派生键），存量节点点亮零算力。
- 指标定义：`oracle` = 树内初始最优（全部节点带 GT）；`oracle_gap_rel` 为负 = 模型发现比树内已知最优更好的 config。
- **三模型同树对比（luna/terra/deepseek-v4-flash）**：3 棵共享树上均大幅超越随机基线，
  deepseek-v4-flash 3/3 击败树内 oracle；评测能区分不同模型的研究/转移能力（详见
  `docs/reports/AUTORESEARCH_PILOT_2026-08-02.md`）。
- 修复：base_url 双重 `/v1`（404）→ `_ensure_v1()`；deepseek-v4-flash 实测可用（本地 key）。

---

## 6. 验收标准

1. `worktree.py` 单测：树结构、lit/unlit、few-shot 选择、base 质量门槛全部可断言。
2. L2 闭环单测：mock LLM 返回闭集 config → GT 跑通 → 树正确点亮 → 评分正确。
3. 端到端 pilot：≥3 个 problem × ≥2 棵树 × K=3，便宜模型跑通，产出评分 + 日志。
4. 与 L0 对照：L1/L2 的成绩必须能解释（至少不比随机差，且能定位"卡在哪一步"：
   判断错 / 规划错 / propose 的 config 太差）。
5. 全量 `pytest` 不回归。

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| propose 的 config 全是树外新点，算力爆炸 | 闭集吸附 + 每轮强制从"树内未点亮"里选一个作为保底（先点亮存量再 propose 新点） |
| base 不够好导致题目混沌 | base 质量门槛（排名前 25%–50% 且跨 seed 稳定）；跑不通的 problem 跳过 |
| 模型只会抄 few-shot 里最好的 | few-shot 与选项不同源（few-shot 用 lit 邻居，选项含 unlit + 新 propose） |
| 隐藏评分变体没人能解 | 作为诊断对照，不作为主指标 |
| 新 GT 结果与存量候选重复 | candidate_id 内容寻址天然去重，重复即查表 |

---

## 8. 与 Feishu 文档的同步

7-31 Weekly 文档（`https://gcn73xn49fre.feishu.cn/wiki/Fj8sw4saiiWICFkbg3gcReLMn6e`）已登记。
本文档成熟后（M4 产出首轮数据）在 Feishu 追加"AutoResearch 转移评测"一节。
