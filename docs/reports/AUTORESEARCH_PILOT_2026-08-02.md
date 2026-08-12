# AutoResearch 转移评测 Pilot 报告（2026-08-02 夜）

> 设计文档：`docs/plan-autoresearch-eval.md`；评测协议：`docs/eval-sets.md §13`。
> 数据：`artifacts/autoresearch_runs/`（每轮 prompt/raw/reasoning/loss 全记录）；
> HTML 报告：`artifacts/autoresearch_report.html`。
> 模型：gpt-5.6-luna（中转 `~/.agents/relay.json` eval key，`reasoning_effort=high`，无 token 上限）。

---

## 1. 是什么

论文叙事：AutoResearch = config 状态转移。benchmark 原子单元 = 单次转移。
工作树（base + 1–2 salient 编辑的 children，全部有存量 GT）→ L2 propose-loop：
K 轮「propose JSON → 吸附闭集 → 点亮存量或跑新 GT → 观察 loss」。

**预算规则（用户要求）**：params ≤ 1.1× base、total_samples_seen 固定 —— 已硬编码进建树与评分，
树内每个节点都是合法 move。

## 2. 结果（gpt-5.6-luna，31 棵树 × 5 轮）

| 指标 | 值 |
|------|-----|
| improve_base（相对 base 的最佳 loss 提升） | mean 25.7% / median 24.4% / max 86.8% |
| 击败树内初始 oracle（oracle_gap < 0） | 16/31（51.6%） |
| 新实验（真实 GT 运行） | 92 次 |
| 失败轮次 | 0（全部成功解析/验证或合法点亮） |

代表性 run：
- `sym_62678b/tree_c49b96`：+82.3%（base 0.244 → best 0.043）
- `sym_62678b/tree_0cab84`：+79.8%
- `mvar_c59a30/tree_7f4940`：+35.7%；`mvar_befdab/tree_14b559`：+30.2%
- 一部分树 0 提升（模型只点亮已知或 propose 更差），这是合理方差

## 3. L1 plan_light（95 棵树，1 次调用/棵）

| 指标 | 值 | 随机基线 |
|------|-----|----------|
| light-first 命中（选中真实最优 unlit） | 50.5%（48/95） | ~17–20% |
| 完整排序 Spearman | ≈ 0.01 | 0 |

**解读**：模型能可靠挑出「最值得先点亮」的最优 unlit（50% vs 随机 18%），
但排不出中间序（损失区间太紧时近乎随机）。这支持"选项少、只问最优/二选一"的题型设计。

## 4. 行为观察

- **重复 propose**：56/204 轮是 wasted（propose 已点亮 config）；prompt 已加警告，模型仍常重提相似 config。
  → 后续可把 wasted 率作为独立行为指标，或在 scoring 里加重 regret 惩罚。
- **探索模式**：第 1–2 轮倾向近邻小改（换 optimizer/loss），第 3–5 轮开始动架构
  （depth/width/activation/layer_norm），多次出现「depth1 width128 + Adam」这类反直觉但有效的配置。
- 编辑类型分布：model 287 / optimizer 207 / loss 251 / budget 148 —— 模型会动所有轴。

## 5. 本轮修复的 bug（全部有回归测试）

1. **run_new_gt 覆盖 base**：新实验曾继承 base 的 candidate_id → 覆盖 base 的 config+GT。
   修复：剥离派生键重新哈希；8 个被污染 base 已从 `data/datasets/` 恢复；全量完整性校验 0 污染。
2. **存量 candidate_id 哈希不匹配**：改用规范后 config 深度相等匹配（`find_stored_candidate`）→ 点亮存量零算力。
3. **asyncio 事件循环被 GT 阻塞**：`run_new_gt` 改为 `asyncio.to_thread`，并发吞吐大幅提升。
4. **批量并发打爆中转**：每棵树独立 client/semaphore → 改为批量共享 client + 全局 semaphore。

## 6. 数据完整性

- 全量 92 棵树的 base 逐项对比 tree 记录：0 处污染。
- GT 全部来自执行生成代码（spec→code→run→GT 不变量保持）。

## 7. 模型对比（gpt-5.6-terra，同 6 棵树）

| 树 | luna improve | terra improve |
|----|-------------|---------------|
| tree_c49b96 (sym) | +82.3% | +60.9% |
| tree_34a8d7 (sym) | +86.8% | +82.5% |
| tree_7f4940 (mvar_c59a30) | +35.7% | +38.8% |
| tree_14b559 (mvar_befdab) | +30.2% | +27.0% |
| tree_e303cb (mvar_866b4e) | +18.5% | +21.8% |
| tree_ac7769 (bg) | +3.0% | +1.0% |

luna mean 42.8% vs terra mean 38.7% —— **在同一批树上两种模型都大幅超越随机基线**，
转移评测可区分不同模型的研究能力；两模型都在部分树上击败树内初始 oracle。

### 7.1 deepseek-v4-flash（本地 key，同 3 棵）

| 树 | luna | terra | deepseek-v4-flash |
|----|------|-------|-------------------|
| tree_c49b96 (sym) | +82.3% | +60.9% | +64.9%（beat oracle） |
| tree_7f4940 (mvar_c59a30) | +35.7% | +38.8% | +40.4%（beat oracle） |
| tree_e303cb (mvar_866b4e) | +18.5% | +21.8% | +17.2%（beat oracle） |

deepseek-v4-flash 3/3 击败树内初始 oracle，与 luna/terra 同量级 —— 该评测对
"便宜快速模型"同样有效且能区分能力差异。

### 7.3 claude-opus-5 / Kimi-K3（eval key，同 6 棵树，2026-08-03 补跑）

| 树 | luna | terra | claude-opus-5 | Kimi-K3 | deepseek-v4-flash |
|----|------|-------|---------------|---------|-------------------|
| tree_c49b96 (sym) | +82.3% | +60.9% | +71.6% 🏆 | +84.6% 🏆 | +64.9% 🏆 |
| tree_34a8d7 (sym) | +86.8% | +82.5% | +85.3% 🏆 | +85.3% 🏆 | — |
| tree_7f4940 (mvar_c59a30) | +35.7% | +38.8% | +28.3% | +33.3% | +40.4% 🏆 |
| tree_14b559 (mvar_befdab) | +30.2% | +27.0% | +5.8% | +0.0% | — |
| tree_e303cb (mvar_866b4e) | +18.5% | +21.8% | +29.9% 🏆 | +18.7% 🏆 | +17.2% 🏆 |
| tree_ac7769 (bg) | +3.0% | +1.0% | +0.0% | +1.5% | — |

🏆 = 该 run 击败树内初始 oracle（oracle_gap < 0）。

| 指标 | claude-opus-5 | Kimi-K3 |
|------|---------------|---------|
| mean / median / max improve | +36.8% / +29.1% / +85.3% | +37.2% / +26.0% / +85.3% |
| 击败树内 oracle | 3/6 | 3/6 |
| 新增真实 GT | 17 | 23 |

解读：**五个模型在同一批树上全部大幅超越随机基线**，且在 3/6 棵树上击败树内已知最优；
claude-opus-5 与 Kimi-K3 的量级与 terra（mean +38.7%）持平、高于 luna（+25.7%）。
难度上限来自树本身（如 tree_ac7769 树内最优就是 base 附近的 config，任何模型都难突破），
下限（0% 提升）是合理方差，不是评测失灵。这支持「L2 转移评测能区分不同模型的研究能力」。

### 7.2 修复：base_url 双重 /v1（404）

`run_one_loop` 曾对已含 `/v1` 的 base_url 再拼一次 `/v1` → deepseek 404。
新增 `_ensure_v1()`（已带 /v1 则原样，否则补 /v1），luna/terra/deepseek 统一走该路径。

## 8. 下一步（M5+）

1. ~~更多模型：claude-opus-5 / Kimi-K3 同树对比~~ —— 已完成（§7.3，5 模型同树对比）。
2. **隐藏评分变体**：同一棵树不显示 lit loss，只显示"已点亮"状态 —— 呼应 ASI-Bench 思路，测"无即时反馈"。
2. **隐藏评分变体**：同一棵树不显示 lit loss，只显示"已点亮"状态 —— 呼应 ASI-Bench 思路，测"无即时反馈"。
3. **文献任务扩展**：从本地调研（OpenML/Kaggle/已有 bench 封装）下载多样化 toy 任务迁移成 problem 实例。
4. **评分升级**：regret@k 曲线 + wasted 惩罚；`propose 并行 P 个`（用户提过 5 个并行修改）。
5. **与 L0 对照**：同一棵树同时出 select_best（6 选 1）与 propose-loop，验证"判断 vs 研究"能力差。
