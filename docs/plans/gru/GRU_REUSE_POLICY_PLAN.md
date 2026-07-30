# GRU Pilot：候选复用题集计划

## 状态（2026-07-29）

- 已完成：逐 run reuse policy、组题选择器、CLI、审计器和回归测试。
- 已生成：`data/datasets/bigram_lm/bg_fdc03b/questions/run_100q_2c_395272`，policy 为 `blind_pair_unique`。
- 已审计：100/100 题通过，100 个不同 GRU--Transformer pair；报告位于该 run 的 `audit_gate_3_4/`。
- 已生成 Inspector collection：该 run 的 `review_collection.json`；可直接交给 `tools/start_quiz.py --question-run`。
- 已验证：`sequential_bounded_reuse` 在同一显著 pair 图上可选出 100 题并满足 `max_candidate_uses=10`；尚未为它写入单独题集。
## 目标与边界

在既有 `bg_fdc03b` / `set_5120_var_fix_fix_17862b` 的 32 个已完成候选上组装题目；不训练新候选、不重跑 10-seed GT、不修改冻结的 `v2.4-gru-architecture-pilot` profile。

所有题目继续是二选一 `architecture_only`：同题固定数据集、Adam、cross-entropy、batch size 和 5120 budget，仅模型规格变化；候选必须构成 GRU-vs-Transformer 且通过既有显著性契约。

## 按 run 固化的复用策略

| policy | 用途 | 候选跨题复用 | 额外限制 |
|---|---|---|---|
| `globally_disjoint_within_run` | 既有 canonical run | 禁止 | 无 |
| `blind_pair_unique` | 无作答反馈的 blind pool | 允许，不设次数上限 | 同一 GRU--Transformer pair 不重复 |
| `sequential_bounded_reuse` | 答后揭晓的 review/practice pool | 允许 | 每个候选最多 10 次；pair 不重复 |

策略由每个 `run.json` 声明并永久随 artifact 保存；默认仍为既有的全局不复用。该机制不改变 profile，也不追溯改变历史 run。

## Phase 1：100 题 blind pool

1. 为组题器增加上述逐 run policy；`blind_pair_unique` 从已验证显著的 GRU-vs-Transformer pair 中选择 100 个不同 pair。
2. 写入独立 review artifact，manifest 至少记录：`run_purpose`、`canonical_blind_evaluation: false`、policy、pair 唯一规则、候选集、profile hash、组题 seed 和 question IDs。
3. 审计器继续逐题重算 provenance、架构-only 不变量、显著性、赢家与 prompt 泄漏；对 `blind_pair_unique` 不检查候选出现次数，只报告 usage histogram，并检查 pair 唯一与跨类型。
4. 执行生成器、审计器和相关单元测试；确认旧 canonical run 仍按严格 disjoint 规则通过。

## Phase 2：展示与人工审查

本 100 题集不用于比较人类与 agent 的分数，因此不要求严格的“全程不揭晓” blind UI；可直接在现有 Inspector 中打开并在选择后显示 GT/指标。它仍须在 manifest 与审计报告中标为 review/blind pool，而非 canonical blind evaluation。

若未来需要顺序学习与审题，可另建 `sequential_bounded_reuse` collection，并使用 `max_candidate_uses=10`；这不改变本 blind pool 的无上限候选复用规则。

## 验收与用户决策点

- 技术验收：100 题、100 个不同 pair、每题显著且 GRU-vs-Transformer；blind pool 不触发候选次数失败。
- 语义验收：manifest 和审计报告将该 artifact 标为 review/blind pool，而非 canonical blind evaluation；现有 Inspector 可直接用于当前人工审查。
