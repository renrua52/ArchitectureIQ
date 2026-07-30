# XOR Pilot：100 题候选复用计划

## 状态

- 提议中；用于对齐，不生成题目或重跑训练。
- 复用冻结的 `v2.5-xor-holdout` 训练与显著性语义；不新建 profile，不修改既有 6 题 preview。

## 目标与边界

在固定二维 XOR dataset、split、loss、optimizer、batch size 与 samples-seen 下，用随机 sampler 生成 `16 MLP + 16 KAN` 的起始候选池，重新训练后，从最多 256 个 MLP--KAN pair 中组装 100 道不同 pair 的 `architecture_only` 题。

同一候选可以跨题复用；同一无序 MLP--KAN pair 不可重复。该题集是 review/blind practice pool，不是 100 个统计独立发现，也不作为 canonical blind evaluation。

XOR 继续不使用 `gap_min`。每个 pair 仍必须满足：两边 test CE 有限、无训练失败、既定逐-seed 胜负规则，以及均值 ± 标准差不重叠。

## 按 run 固化的复用策略

本次 100 题 run 使用 `blind_pair_unique`：

- 允许候选在不同题中无限复用；
- 每个 MLP--KAN pair 至多出现一次；
- 审计器报告候选 usage histogram，但不以次数作为失败门槛；
- run metadata 标明 `run_purpose: review_blind_pool` 与 `canonical_blind_evaluation: false`。

若以后做答后揭晓的顺序练习，可在相同 GT 上另导出 `sequential_bounded_reuse`，每个候选最多 10 次；它不是本次 100 题 run 的约束。

## 最短执行路径

1. 分别调用 MLP sampler 与 KAN sampler，直到各得到 16 个去重 model spec；不得调用会先随机选择模型类型的 generic `sample_model()`。记录 family-specific sampler seed。显式固定 batch size `32`、cross-entropy 与统一 Adam (`lr=1e-3`、weight decay `0`、betas `[0.9, 0.999]`)；将这 32 个实际 materialized specs、seeds、shared training spec 与 hash 冻结为该 run 的 matrix。随机性只用于起始候选池，不在组题时随机重采样。
2. 每个唯一候选按固定 screening/holdout seed 分区重新训练一次；同一候选在多个 pair 中引用同一组 GT，禁止按 pair 重训或事后调参。
3. 从通过 holdout 的 pair 中选择 100 个：pair 唯一、全部跨 MLP--KAN，且任一赢家家族占比不得超过 70%。不要求严格 50/50。
4. 若某赢家家族超过 70%，先删除该方向的冗余 pair；若剩余不足 100，则用新的、记录在案的 family-specific sampler seed 补采样和训练候选。补充 matrix 必须记录其排除的既有 matrix hash，并排除同家族已冻结 spec；可只补赢家稀缺家族，并复用对方家族已有的 GT。不得降低显著性门槛、重训旧 pair 或修改 profile 以追求方向平衡。
5. 一次性生成 100 道题、审计报告和 Inspector-compatible `review_collection.json`。审计重算 provenance、architecture-only 不变量、赢家、显著性、pair 唯一、赢家占比和 prompt 泄漏；候选复用只记录直方图。
6. 将完整 100 题 run 在 Inspector 中逐题人工审查。交付时提供已验证的 PowerShell 启动命令，使启动后直接加载该 collection；不额外创建启动脚本、数据库或中间 collection 实体。

## 验收

- 100/100 题，100 个不同 MLP--KAN pair；
- 任一赢家家族占比不超过 70%；
- 每题满足 XOR 的非 `gap_min` 显著性契约；
- 同一候选可复用，pair 无重复，usage histogram 已记录；
- manifest 明确标为 review/blind practice pool，并有可直接加载它的 Inspector 命令。