# GRU / Transformer Pilot：采样与人工审题计划（v2.4）

## 当前状态

- profile：`v2.4-gru-architecture-pilot`（正式语义；不修改既有 `v2.3-gru-pilot`）；
- 已完成：候选配额、Adam 固定协议、GRU 配置池、CPU seed 并行的代码与定向验证；
- 下一步：新建 v2.4 的 `bigram_lm` dataset，生成一组均衡的 32 candidates；
- 停止点：生成并审计出 5–6 道题后，立刻交由人工盲审，不再自动扩大规模。

## 1. 已冻结的比较契约

本轮只验证能否形成值得审查的 `architecture_only` 题目，而非发布题库。

| 项目 | 固定选择 |
|---|---|
| 数据族 | `bigram_lm` |
| profile | `v2.4-gru-architecture-pilot` |
| 预算 | `total_samples_seen = 5120` |
| 每题 choices | 2 |
| 优化器 | Adam：`lr=1e-3`、`weight_decay=0`、`betas=(0.9, 0.999)` |
| loss / batch size / device | cross-entropy / 32 / CPU（同一题内固定） |
| 显著性 | 既有 `gap_min=0.05`、`win_rate_min=0.70`、non-overlap |
| 可变化项 | 仅完整 model spec |
| 参数量比例 | 仅诊断，不作为 `<=2` 的硬过滤条件 |

同一题绝不混 dataset、优化器、loss、batch size、budget 或 device。GT 必须继续执行每个 candidate 实际生成的 `train.py`；不使用手写训练替代物。

## 2. 候选池设计

一次生成精确均衡的一组 32 candidates：

- 16 个 `transformer_lm`；
- 16 个 `gru_lm`；
- `--vary model`，并由 manifest 记录该配额；
- GRU 采样池为 13 个 hidden widths × 8 个层数，即 104 个不同结构配置；
- Transformer 使用既有结构池；Adam、loss、batch size 全部固定。

每个 candidate 运行 10 个独立 seeds。完成后执行 Gate 1/2：profile hash、dataset、训练身份、GT 健康性与 GRU/Transformer 分布必须正确；然后统计 cross-type、candidate-disjoint 且显著的 pair 容量。参数量及 pair ratio 同时输出为审计诊断，但不否决其他合法 pair。

## 3. CPU seed 并行策略

正式采样默认使用 CPU seed 多进程，前提是仅在本对话运行命令中显式设置：

```powershell
$env:ARCHITECTURE_IQ_SEED_WORKERS = '8'
$env:ARCHITECTURE_IQ_SEED_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

实现约束：每个进程仅运行一个独立 seed 的生成 candidate 代码；每个 seed 仍在 `train.py` 内执行自己的 `torch.manual_seed(seed)`；`executor.map` 保持 seed 顺序；只有父进程写 `summary.json` 与 curves。CUDA 或需要逐步 inspector progress 的调用保持串行。

基准门槛为相对 4-thread 串行至少 1.5×。在同一 GRU candidate 的隔离测试中，8 个一线程 seed workers 相对 4-thread 串行得到约 **1.55×**；相同单线程执行下，10 个 seed 的失败状态与指标逐项相等。真实 runner 的并行 smoke 也与既有 smoke 的 10 个 seed 指标完全一致。

## 4. 生成、审计与停止点

1. 新建 v2.4 dataset instance（profile hash 已变化，不能把 v2.3 artifact 当作正式来源）。
2. 生成 32 个均衡 candidates，完整执行 GT。
3. Gate 1/2：汇总健康性、类型数、参数量、显著 pair 容量；若没有至少 5–6 个 candidate-disjoint 的合格 pair，报告瓶颈，不静默放宽阈值。
4. 生成最多 6 道、至少 5 道的 `architecture_only` 题；若自动生成少于 5 道，保留真实数量并报告。
5. Gate 3/4：prompt 不含任何 final metric、曲线、seed 统计或答案字母；题内只变化模型；同时确认静态 exporter / inspector 的 GRU 模型卡与 canonical prompt 一致。
6. **停止**，交给你盲审。不会在审题前继续生成更多题或调整 profile。

## 5. 你需要插手的位置

本轮只保留一个必要人工 gate：对最终 5–6 道题先盲审，再查看 GT。每道记录：你的选项与置信度、是否存在模型名称或参数量 shortcut、题面是否清楚公平、是否在考查合理的结构/训练直觉，以及 `pass` / `revise` / `reject` 和一句原因。

若 Gate 1/2 的 pair 容量不足，我会先给出具体数量、失败原因和训练成本，再请你决定是新增候选、换 dataset seed，还是建立新 profile；不会擅自改变已冻结的 v2.4 规则。