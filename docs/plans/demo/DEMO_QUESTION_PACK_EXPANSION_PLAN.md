# ArchitectureIQ Demo 题包扩展方案

> 状态：讨论汇总版
>
> 本文档记录截至 2026-07-21 已讨论形成的 demo 题包扩展方案。它用于指导 KAN 合并后的题目生成、筛选和人工审阅，不修改历史题目，也不把 LLM 审计嵌入正式题目生成器。

## 1. 目标与边界

目标是形成一组质量较高、可以展示 ArchitectureIQ 能力的二选一 demo 题包。题目数量不再受现有 candidate 容量限制；允许针对目标题包重新设计并训练 candidate。

本方案的边界如下：

- KAN 分支合并并完成 profile 冻结后再开始正式扩展。
- 先确定 profile、device、budget 和 seed 语义，再生成 candidate。
- 新题从真实执行的 candidate 和 GT 产生，继续遵守 `spec → generated code → execution → GT` 不变式。
- 不原地修改历史 `candidate_spec.json`、GT 或旧 question run。
- LLM/subagent 只作为 demo 题包的外部质量审计，不进入正式生成代码，也不决定随机采样。
- 最终 demo 题包由确定性 gate、LLM 审计和人工复核共同形成。

## 2. 当前 dataset inventory

当前 checkout 中已有 11 个 dataset instance：

| Family | Dataset instances | 当前可用模型类型 |
|---|---|---|
| `univariate_regression` | `sym_40f9b4`、`sym_62678b`、`sym_a2a02f`、`sym_f5e1e3` | MLP、KAN（以冻结后的 V2 为准） |
| `multivariate_regression` | `mvar_35761e`、`mvar_4e315c`、`mvar_e3e90e` | MLP、KAN（以冻结后的 V2 为准） |
| `bigram_lm` | `bg_e2f7c8` | Transformer LM |
| `synthetic_tabular_classification` | `stabcls_0a866e`、`stabcls_59e9e3`、`stabcls_edccef` | MLP |

当前已有 question artifact 的基线分布是：

- `architecture_only`：18 道，全部来自三个 classification instance；
- `mixed`：36 道，全部是 `model + optimizer`，且 dataset 分布不均衡；
- `optimizer_only`：当前没有已生成题目；
- `loss_only`：当前没有已生成题目。

这些现有题目作为历史或可复用资产，不代表最终 demo 题包的目标分布。

## 3. 目标题包

建议最终形成六个题包。普通题包以 30 道为目标，loss-only 因为只有 8 个 dataset 支持 loss 变化，先以 24 道为目标。

| 题包 | 目标数量 | 主要目的 |
|---|---:|---|
| `architecture` | 30 | 只比较 architecture/model 配置，覆盖全部 dataset instance |
| `optimizer` | 30 | 固定 model 和 loss，只比较 optimizer |
| `loss` | 24 | 固定 model 和 optimizer，只比较 loss |
| `mixed` | 30 | 比较多个训练轴的联合影响 |
| `architecture_easy` | 30 | architecture-only，结构和容量差异明显 |
| `architecture_hard` | 30 | architecture-only，参数量接近但答案仍稳定 |

总目标规模为 174 道题。实际数量以确定性 gate、candidate 容量和人工复核结果为准；可以先完成每包 20 道的 pilot，再扩展到目标数量。

## 4. 普通题包的 dataset 配额

`architecture`、`optimizer`、`mixed`、`architecture_easy`、`architecture_hard` 使用同一套覆盖模板：

| Dataset | 每个普通题包的题数 |
|---|---:|
| `sym_40f9b4` | 2 |
| `sym_62678b` | 2 |
| `sym_a2a02f` | 2 |
| `sym_f5e1e3` | 2 |
| `mvar_35761e` | 3 |
| `mvar_4e315c` | 3 |
| `mvar_e3e90e` | 2 |
| `bg_e2f7c8` | 6 |
| `stabcls_0a866e` | 3 |
| `stabcls_59e9e3` | 3 |
| `stabcls_edccef` | 2 |
| **合计** | **30** |

这个分配保证每个普通题包都覆盖全部 11 个 dataset instance。Bigram 当前只有一个 instance，因此暂时会占据较多题目；新增 Bigram instance 后再分散配额。

## 5. Loss-only 题包

当前 profile 中：

- 回归 family 支持 `mse`、`mse_l1`、`mse_l2`；
- Bigram LM 支持 `cross_entropy`、`cross_entropy_l1`、`cross_entropy_l2`；
- synthetic classification 当前只有 `cross_entropy`，不能生成真正的 loss-only 对比。

建议 loss-only 题包分配为：

- 4 个 univariate instance：每个 3 道，共 12 道；
- 3 个 multivariate instance：每个 3 道，共 9 道；
- `bg_e2f7c8`：3 道；
- classification：0 道。

每道 loss-only 题固定 model、optimizer、batch size、budget、device 和 dataset，只改变 loss spec。

## 6. 题型约束

### 6.1 Architecture-only

只允许 model 配置发生变化。optimizer、loss、batch size、total samples seen、device 和 dataset 必须一致。

这里的 architecture 可以包括：

- MLP 的 depth、width、activation、residual；
- KAN 的 depth、width、grid size 等；
- Transformer 的 d_model、层数、head 数、FFN 宽度。

因此，classification 虽然只有 MLP model family，仍然可以产生 MLP architecture-only 题，但不能把它描述为 MLP-vs-KAN 题。

### 6.2 Optimizer-only

固定 model、loss、batch size、budget 和 device，只改变 optimizer 及其对应参数。

不应通过故意给某个 optimizer 配置明显不合理的 learning rate 来制造简单答案。

### 6.3 Loss-only

固定 model、optimizer、batch size、budget 和 device，只改变 loss。L1/L2 正则化系数必须处在经过 plausibility audit 的范围内。

### 6.4 Mixed

`mixed` 不应只是“任意多个字段变化”。建议预先规定 varying axes 的组成，例如：

- 约 1/3：`model + optimizer`；
- 约 1/3：`model + loss`；
- 约 1/3：`model + optimizer + loss`。

具体比例可以根据 candidate 容量调整，但每道题的 `varying_axes` 必须明确记录。除非题包明确考察 budget/batch size，否则不要让 batch size 随意变化。

## 7. Easy 与 hard 的难度定义

难度应在同一 `family × question_type` 分层内判断，不使用跨 family 的统一原始 gap 阈值。

### 7.1 Architecture-easy

候选条件：

- 只有 architecture 变化；
- 参数量差异可以较大；
- architecture 结构差异明显；
- 性能 gap 位于同一分层的较高分位，初步可取前 25%；
- winner 具有较高 seed win rate 和较低波动；
- 不存在明显病态 hyperparameter。

easy 的目标是让差异足够明显，但不能退化为“只看参数量就能回答”的题目。

### 7.2 Architecture-hard

候选条件：

- 只有 architecture 变化；
- 参数量比例控制在相对接近的范围内，初始可放宽为 `max(params)/min(params) ≤ 2.0`；参数量只是辅助约束，不是 hard 难度的主要定义；
- architecture 存在实质结构差异，而不是只改一个无关字段；
- 性能 gap 位于同一 `family × question_type` 分层的中低区间；这是 hard 难度判断的主要依据；
- 仍通过 significance、win-rate 和 seed 稳定性检查；
- prompt 包含做出判断所需的完整信息。

hard 的困难来自 reasoning，而不是 GT 歧义。训练轨迹可以验证答案，因此不能通过制造发散、失败 seed 或统计不稳定来制造 hard 题。

### 7.3 Gap 与参数量阈值

参数量比例和 gap 分位数先作为候选筛选指标，不立即冻结为最终阈值。gap 分位数是主要难度指标，参数量比例是辅助的容量控制指标。等第一批 candidate 训练完成后，再观察每个 family/type 的实际分布。

建议初步规则：

- easy：gap 位于同一分层前 25%，winner 稳定；
- hard：gap 位于同一 `family × question_type` 分层的中低区间，但仍满足最低 significance 和 win-rate 门槛；
- borderline：虽然勉强通过统计门槛，但波动较大或接近不可区分，进入人工复核，不直接进入 hard 包。

## 8. 定向 candidate 生成与重新训练

允许重新训练后，不应只随机生成大量 candidate 再碰运气筛选，而应按题包和分层定向设计。

推荐流程：

1. 先确定 dataset、题型、变化轴和难度条件；
2. 生成满足结构条件的 candidate pair 或 candidate pool；
3. 使用正式 generated code 运行 GT；
4. 根据 gap、win-rate、seed 稳定性和参数约束筛选；
5. 将通过的 pair 组装为 candidate-disjoint questions；
6. 不足时只补训练对应分层，不重新生成全部数据。

每道二选一题至少需要两个 candidate。实际训练应预留余量，因为部分 candidate 会因 GT、显著性、配置合理性或人工审计失败而被排除。初步可以按目标 candidate 数的 2–3 倍准备训练池，之后根据实际通过率调整。

现有合格 candidate 可以复用；缺口再训练。历史 candidate 的 profile、profile hash、device 和生成语义必须经过审计，不能因为数量不足而混入不兼容 artifact。

## 9. 确定性检查

继续使用现有 question expansion 计划中的 Gate 1–4：

### Gate 1：配置、兼容性与 provenance

- dataset、family、candidate、profile、profile hash 完整；
- model、optimizer、loss 在 profile 中合法；
- model 与 family 兼容；
- budget arithmetic 正确；
- execution device 一致；
- candidate set 的 varying/invariant axes 与实际 spec 一致。

### Gate 2：GT 健康度

- summary 和 curves 完整；
- seed 数量和 seed 顺序正确；
- 非失败 seed 的指标有限；
- failed seed 数在允许范围内；
- selection metric 与 dataset 一致；
- 没有明显数值爆炸或全体失败。

### Gate 3：题目统计有效性

- choice compatibility 正确；
- winner、correct letter 和 metric 方向正确；
- gap、win-rate、non-overlap 等条件满足；
- 题目声明的 varying axes 与真实 spec 一致。

### Gate 4：泄露和 collection

- candidate 不在同一评测 collection 中重复；
- question ID、candidate ID、dataset ID 与 manifest 一致；
- public prompt 不包含答案、GT 或私有统计量；
- profile 和 device 语义不混用。

## 10. LLM/Subagent 外部审计

LLM 审计只用于 demo 题包的外部质量筛选，不写入正式题目生成代码，不参与随机采样，因此不会改变代码层面的可复现性。

### 10.1 审计目标

subagent 不是第二个答案裁判，重点检查题目是否符合设计精神：

- 题型标签是否与实际比较轴一致；
- 是否存在未计划的 shortcut；
- easy/hard 难度是否符合目标；
- 题面和代码摘录是否完整、自然、无歧义；
- 比较是否公平、有 reasoning value；
- 是否存在明显不合理的 hyperparameter setting；
- 题目是否适合作为 demo 展示。

subagent 答错不等于题目失败。答错只作为诊断信号；除非它暴露出题面缺失、shortcut、配置病态或其他严重设计问题，否则不能据此淘汰题目。

### 10.2 Candidate-level hyperparameter plausibility audit

在组装题目之前审查单个 candidate：

- learning rate 是否明显极端；
- weight decay、loss lambda 是否病态；
- batch size 与训练步数是否导致训练没有实际意义；
- optimizer 参数是否与对应 optimizer 合理匹配；
- model capacity、KAN grid 等是否明显异常；
- 是否存在数值发散或完全不学习的迹象；
- setting 是否适合该 dataset family 和题包目标。

必须区分：

- 非法配置：由确定性 gate 处理；
- 合法但性能较差：不自动淘汰，可能是有效对比；
- 能运行但明显病态：subagent 标记，通常不进入 demo；
- 配置本身合理但结果异常：人工复核或重新训练。

### 10.3 Question-level spirit audit

生成题包后，subagent 逐题检查：

- prompt 是否能独立理解；
- 题目是否真的考察目标能力；
- 是否存在只比较参数量或只比较某个表面字段的 shortcut；
- hard 题是否只是噪声大，而不是推理难；
- mixed 题是否变化轴过多而无法解释；
- 是否值得作为 demo 展示。

建议先做盲审，只给 public prompt，不给 correct letter、GT 和 significance。随后可以做内部对照审计，用于检查它提出的问题是否真实存在。

### 10.4 审计结果

每个 candidate 或 question 可以标记为：

- `keep`：无明显设计问题；
- `human_review`：存在疑点，需要人工判断；
- `retrain`：配置本身不值得保留，但题目目标仍可保留；
- `revise`：需要修改 prompt 或重新组装；
- `reject`：违反题包设计精神。

建议保存外部审计记录，至少包括 question/candidate ID、profile hash、审计模型、审计时间、问题类型、严重程度、subagent 理由和最终人工决定。该记录不参与题目生成，也不改变随机性，但保证 demo 题包可追溯。

## 11. Demo 题包工作流

```text
KAN 合并
→ 冻结 profile 与 profile hash
→ 确定题包 manifest 和 dataset 配额
→ 盘点可复用 candidate
→ 按题包目标定向生成并训练 candidate
→ Gate 1/2
→ 按 varying axes、gap 和参数量组装题目
→ Gate 3/4
→ candidate-level hyperparameter plausibility audit
→ question-level LLM spirit audit
→ 人工处理 review/retrain/revise/reject
→ 固定 demo 题包版本
```

## 12. 待最终确定的事项

以下事项应在 KAN 合并和 profile 冻结后确定：

1. 使用冻结后的 `v2`，还是建立专门的 `v2-demo` profile；
2. demo 的统一 execution device；
3. 各题包是否在整个 demo collection 内 global candidate-disjoint；
4. mixed 中各 varying-axis 组合的确切比例；
5. architecture-hard 的参数量比例上限，以及各分层的 gap 分位数区间；
6. 各 family/type 的 gap 分位数和最低可靠性门槛；
7. LLM 审计后由谁执行最终人工决定；
8. 第一轮 pilot 是每包 20 道还是直接按目标数量生成。

## 13. 阶段完成标准

第一版 demo 题包完成时，应至少满足：

- 所有题目来自冻结且可追溯的 profile；
- 每个普通题包覆盖预定的 dataset instance；
- loss-only 题目只使用支持多 loss 的 dataset；
- architecture easy/hard 的定义和实际筛选指标一致；
- 所有题目通过 Gate 1–4；
- 明显不合理的 hyperparameter setting 已被剔除或替换；
- 每个题包经过 subagent 外部审计；
- high-severity 问题经过人工处理；
- 最终题包 manifest、审计记录和版本信息可追溯；
- 历史题目、candidate 和 GT 没有被原地覆盖。
