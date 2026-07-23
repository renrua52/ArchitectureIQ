# KAN 分支合并计划：V2.1

状态：已确认方案，待实施
适用仓库：`ArchitectureIQ-main-integration`
来源对话：`019f64d9-9172-72b3-9866-2a07ca2bbb25`

## 1. 目标

以最小、可审查、可回滚的方式，把 KAN 分支中已经完成的校准和工具进展接入当前主线，同时保持现有 V2 benchmark 语义和历史 artifacts 不变。

本轮目标是建立 `v2.1`，而不是修改冻结的 `v2`。

## 2. 已确认的决策

### 2.1 Profile 版本

- 当前 `profiles/v2.yaml` 保持冻结。
- 主线当前 V2 profile hash：`3993a8aef680d37c`。
- 新的 KAN 扩展进入 `profiles/v2.1.yaml`。
- V2 历史 candidate、question、review collection 不改名、不重算、不迁移。
- 新 profile 产生新的 profile hash 和新的 artifacts。

### 2.2 分类 KAN

- `v2.1` 允许 synthetic tabular classification 生成 KAN candidate。
- 分类 KAN 需要通过 profile-aware model gate 控制，不能让 family 的兼容列表无条件改变旧 profile 行为。
- 分类 KAN 不自动混入当前 24 题审计 collection。
- 分类 KAN 先作为新的 candidate/calibration 能力接入；是否生成独立题集另行决定。

### 2.3 受控 KAN–MLP pair

`kan_mlp_pairs` 只用于校准和受控对照，不属于普通 benchmark candidate pool。因此采用独立配置：

```text
configs/kan_mlp_pairs_v2.1.yaml
```

该配置包含 KAN/MLP 结构、优化器、loss、batch size 和预算。工具通过 `--pair-config` 显式读取，报告记录该配置的稳定 hash。

这样 pair 变化不会无必要地改变正式 profile hash，也不会让通用 candidate sampler 意外采用受控 pair。若未来将受控 pair 纳入正式 benchmark，再提升为新的 profile 语义。

### 2.4 显著性规则

- 本轮不修改 `gap_min`、`win_rate_min` 或 `use_non_overlap`。
- 当前 `gap_min=0.05` 的跨 metric 泛化问题另列为 `v2.2` 的评测协议议题。
- KAN 合并期间继续使用现有 validator，避免同时改变模型池和题目判定规则。

## 3. 合并边界

### 3.1 计划移植

根据目标分支实际差异，重点移植并适配：

- KAN 参数池收缩和分类 KAN 的配置语义；
- profile-aware family/model gating；
- `tools/generate_kan_mlp_demo.py` 的独立 pair 配置支持；
- `tools/generate_kan_mlp_multivariate_calibration.py`；
- `tools/generate_kan_mlp_classification_calibration.py`；
- `tools/kan_mlp_benchmark_report.py`；
- 对应的 KAN、分类 KAN、校准和报告测试；
- 阶段 3/4 校准报告中的可复现性说明。

### 3.2 明确不移植

- 整个 KAN 分支或整分支 cherry-pick；
- 与 KAN 无关的历史上下文删除、leakage 工具删除或 README 大范围改写；
- `outputs/`、`.tmp/`、pytest 临时目录、`__pycache__` 等运行产物；
- 旧 V2 candidate/question artifacts 的重新生成；
- `gap_min` 或整个 significance validator 的重构。

## 4. 实施阶段

### 阶段 A：基线和差异清点

1. 记录当前主线 dirty worktree，不覆盖用户已有修改。
2. 对目标分支逐文件生成差异清单。
3. 区分三类内容：
   - 可直接移植的新增工具/测试；
   - 需要改写为 V2.1 的 profile 变更；
   - 应排除的无关变更和运行产物。

完成标准：形成明确的文件级移植清单，且当前 V2 文件没有被改写。

### 阶段 B：代码与测试移植

1. 将校准、报告工具适配到主线当前 API。
2. 保留现有 `spec → generated code → GT` 管线，不新增旁路训练逻辑。
3. 确保分类 KAN 使用 `output_dim=2` 和 `cross_entropy`，并能输出 `test_ce` 与 accuracy。
4. 为独立 pair 配置增加加载、校验和稳定 hash。
5. 处理与当前 Inspector、参数量统计、progress callback 等 dirty 修改的兼容性。

完成标准：工具可以在不生成正式题目的情况下独立运行；新增测试覆盖配置读取、模型渲染、GT smoke 和报告输出。

### 阶段 C：建立 V2.1

1. 以当前 `v2.yaml` 为基线创建 `v2.1.yaml`。
2. 将已校准的普通 KAN pool 写入 V2.1。
3. 增加 profile-aware model gate，采样时使用：

```text
profile 允许的模型类型 ∩ family 支持的模型类型
```

4. 让 V2 继续只产生原有模型范围；V2.1 才开放分类 KAN。
5. 在 dataset/candidate/question manifest 中保留 profile name/hash provenance。

完成标准：

- `load_profile("v2")` 的内容和 hash 不变；
- `load_profile("v2.1")` 成功；
- V2 不会采样分类 KAN；
- V2.1 可以显式采样分类 KAN。

### 阶段 D：校准与报告

1. 使用独立 pair 配置运行一元、多元和分类 KAN 校准。
2. 记录每个候选的：
   - test metric 均值/标准差；
   - seed win rate；
   - failed seeds；
   - trainable parameter count；
   - elapsed time；
   - pair 参数量误差。
3. 生成 calibration report 和 benchmark report。
4. 校准产物写入新的、明确标记为 V2.1 的输出目录，不覆盖旧报告。

完成标准：KAN 在目标 family 上可稳定执行，且报告能复现 pair 配置和 profile provenance。

### 阶段 E：题目入口和网页审计

1. 不修改当前 24 题 review collection。
2. 如需审计 KAN 题，建立独立 collection，并明确标记为 `kan_v2.1_diagnostic`。
3. 先验证 Inspector 能读取 KAN prompt、模型参数、表达式和分类可视化。
4. 只有校准结果和题目分布通过检查后，才生成新的审计题。

## 5. 主要风险与处理

| 风险 | 处理方式 |
|---|---|
| family 兼容列表直接放开 KAN，导致旧 V2 语义改变 | 使用 profile-aware gate，不在 family 列表中无条件泄漏 |
| 修改 `v2.yaml` 导致 hash 和历史语义漂移 | 只新增 `v2.1.yaml` |
| pair 配置改变正式 profile hash | pair 配置独立存放并单独记录 hash |
| 校准脚本与当前 dirty runner/Inspector 冲突 | 文件级适配，先跑 focused tests |
| 把校准结果误当成正式题集 | calibration、diagnostic collection 与 24 题 collection 分离 |
| KAN 训练成本过高 | 先小规模 smoke，再决定是否进行完整 10-seed 运行 |
| KAN 类型成为答案启发式 | 分别统计 KAN/MLP 胜率，并报告简单 baseline |

## 6. 验证顺序

建议按以下顺序验证：

1. profile loading/hash 测试；
2. KAN model render/import/forward/backward；
3. regression KAN GT smoke；
4. classification KAN GT smoke；
5. pair config 和 calibration tool tests；
6. benchmark report tests；
7. prompt formatter/Inspector parity；
8. 少量 V2.1 candidate sampling；
9. 必要时再运行完整校准和新 question run。

不在本轮把全历史 artifacts 全部重跑作为完成条件。

## 7. 完成标准

本计划完成后应满足：

- V2 冻结且可继续读取；
- V2.1 可加载并具有新的 profile hash；
- 回归和分类 KAN 均能通过现有 GT 管线执行；
- KAN–MLP calibration 使用独立配置且可复现；
- 参数量、耗时和失败率进入报告；
- 当前 24 题审计 collection 不被改变；
- 新 KAN 题目若生成，则进入独立 collection；
- `gap_min` 语义保持不变，后续另行进入 V2.2 设计。

## 8. 后续另行决策

- 是否将分类 KAN 题纳入正式 benchmark；
- 是否把受控 KAN–MLP pair 提升为正式 diagnostic split；
- 是否在 V2.2 引入 metric/family-aware effect criterion；
- 是否建立计算成本匹配的独立评测协议；
- 是否引入 FastKAN、动态 grid 或其他新的 KAN variant。
