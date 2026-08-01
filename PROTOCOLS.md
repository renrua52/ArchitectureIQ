# ArchitectureIQ Benchmark Protocols

ArchitectureIQ 支持多种 evaluation protocol，每个 protocol 有不同的用途和约束。所有 protocol 共享相同的题目集，但**结果不可直接比较**——不同 protocol 的分数差异可能反映 protocol 本身的特性而非模型能力。

## 核心原则

1. **Non-repeating candidates**：每个 candidate 在整个 benchmark 中最多出现一次，杜绝 sequential memory 效应。
2. **Spec → Code → GT 一致**：prompt 中展示的代码就是实际执行产生 GT 的代码。
3. **Anti-shortcut 验证**：所有题目必须通过 anti-shortcut gates 才能进入最终 benchmark。

---

## Protocol 定义

### Protocol 1: Per-Question Blind（主 benchmark protocol）

**用途**：最终 benchmark 分数，测纯推理能力。

**输入**：单个 sanitized question JSON（不含答案、不含 metrics）。
**输出**：单个字母（A/B/C/D）。
**约束**：
- 每次调用只能看到一个 question，无任何上下文
- 无工具访问（不能读文件、跑代码、搜索）
- 无交叉比较（看不到其他 question 的 choices）
- 无 feedback（看不到正确答案）
- 同一模型的多题评测必须用独立 session

**随机 baseline**：三选一 → 33.3%，二选一 → 50.0%

**评分**：`accuracy = correct / total`

---

### Protocol 2: Full-Set Blind（开发阶段快速评估）

**用途**：开发阶段快速评估，检查系统性偏差。

**输入**：所有 sanitized question JSON（一次性）。
**输出**：每个 question 的答案（一个字母）。
**约束**：
- 所有 question 一次性可见，可交叉比较
- 无工具访问
- 无 feedback

**随机 baseline**：同 Protocol 1。

**与 Protocol 1 的差异**：agent 可以跨 question 比较 candidates 的模式（如"这个 optimizer 在好几个 question 里都赢了"），但没有 feedback 来确认猜测是否正确。

**评分**：同 Protocol 1。

---

### Protocol 3: Sequential Feedback（调试/分析/人类 baseline）

**用途**：调试题目质量、分析 model 学习行为、建立人类 baseline。

**输入**：逐题展示 question + 上题反馈。
**输出**：逐题答案 + lesson。
**约束**：
- 严格 prediction-before-feedback（先用 CLI 记录预测，再揭示正确答案）
- 反馈内容：该题的正确答案 + 各 choice 的 metrics
- Agent 可以记录 lesson 用于后续 question
- 禁止直接读取 answer key 或 GT 文件
- 禁止 spawn 子 agent 或 replacement

**评分**：同 Protocol 1，但需标注"sequential"标签。

**预期**：即使 non-repeating candidates，Sequential 分数可能仍高于 Blind，因为 agent 可以从 feedback 中学习架构规律。但差距不应超过统计噪声。

---

## Scoring 输出格式

所有 protocol 的 scoring 输出统一为：

```json
{
  "protocol": "blind_per_question",
  "model": "gpt-5.5-high",
  "total": 60,
  "correct": 42,
  "accuracy": 0.700,
  "random_baseline": 0.500,
  "by_family": {
    "bigram_lm": {"correct": 15, "total": 20, "accuracy": 0.750},
    "multivariate_regression": {"correct": 14, "total": 20, "accuracy": 0.700},
    "univariate_regression": {"correct": 13, "total": 20, "accuracy": 0.650}
  },
  "by_type": {
    "mixed": {"correct": 35, "total": 50, "accuracy": 0.700},
    "architecture_only": {"correct": 7, "total": 10, "accuracy": 0.700}
  },
  "questions": [
    {"question_id": "q_001", "response": "A", "correct": "B", "correct": false, "family": "bigram_lm", "type": "mixed"}
  ]
}
```

---

## 题目质量 Gate

所有最终 benchmark 题目必须通过以下 gate：

| Gate | 检查内容 | 拒绝条件 | 适用 Family |
|---|---|---|---|
| **Affine-Fit** | 目标函数是否近似线性 | R² > 0.95 | univariate_regression |
| **Capacity Shortcut** | max_params 规则是否命中正确答案 | 命中 | 全部 |
| **SNR** | 信噪比是否合理 | SNR < 1.0 | 有 noise 的 family |
| **Interaction** | 最大一阶项贡献是否过大 | 单维度贡献 > 85% | multivariate_regression |

Gate 检查工具：`tools/anti_shortcut_gates.py`

---

## 退化 Univariate 题目标记

部分 univariate 题目因采样随机性退化为近线性函数。这些题目在 `dataset_spec.json` 中标记 `quality_tags: ["degraded_linear"]`，分析时可按需排除。

目前的退化标记：
- `degraded_linear`：affine R² > 0.95
- `degraded_low_range`：y_range < 0.4

---

## 与旧版 Protocol 的差异

| 特性 | 旧版 | 新版 |
|---|---|---|
| Candidate 重复 | 允许（同一 ID 出现在多题） | 禁止（non-repeating） |
| Scoring 口径 | 无统一标准 | 统一 `harness.py --protocol` |
| 题目质量 | 无 gate | 必须通过 anti-shortcut gates |
| 随机 baseline | 未报告 | 明确报告 |
| Protocol 文档 | 分散在 artifacts README | 集中定义 |


---

## 评测端协议（2026-08 起，backend/eval）

题目由评测端从 `backend/data/` 的 problem + candidates 组合而成（题集 JSONL，
见 [`docs/eval-sets.md`](./docs/eval-sets.md)），不再由题目实例自带 question。

### Eval-A: select_best（选择题，主任务）

- **输入**：problem + 5 个带 loss 的参考 setting + 6 个选项（base + 5 个修改，含不改）。
- **输出**：一个字母（A–F）。
- **约束**：base 必在选项中；选项两两在显著字段上差异 ≥ 2；winner vs runner-up 跨 seed win_rate ≥ 0.7，
  且 ratio 满足分 metric 阈值（MSE ≥ 1.15，CE ≥ 1.03）。
- **随机 baseline**：1/6 ≈ 16.7%。探针实测 LLM 接近随机（v1 25% / v1.1 16.7%），
  属"地板效应"任务，用于测上限，需报告置信区间。
- **评分**：`accuracy = correct / total`。

### Eval-B: propose_improvement（config 修改题）

- **输入**：problem + 5 个带 loss 的参考 + base（带 loss）+ 5 个改进 demo（带 loss）。
- **输出**：闭集内的新 JSON config。
- **评分**：`backend/eval/score_proposal.py` 跑 GT（新执行），对比 base：`ratio`、`win_rate_vs_base`。
- **约束**：参数量 ≤ demos 最大参数量 × 1.1；预算固定为 base 的 `total_samples_seen`。
- 探针实测 2/2 次提案击败 base（ratio 1.14x / 1.20x）。

### Eval-C: two_choice_loss_compare（二选一，诊断）

- **输入**：problem + 3 个带 loss 的参考 + 目标对 A/B，问"哪个 loss 更高/更低"。
- **输出**：一个字母（A/B）。过滤：ratio ∈ [1.2, 5]，win_rate ≥ 0.8。
- **随机 baseline**：50%。探针实测 87.5%（8 题）——验证参考校准可解，防止系统性失效。
