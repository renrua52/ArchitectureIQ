# ArchitectureIQ KAN 与 V2 Profile：阶段 1—3 实施计划

> 状态：阶段 1—2 已完成，阶段 3 校准进行中
> 适用仓库：`ArchitectureIQ`
> 范围：冻结 V1、建立 V2、实现第一版 KAN、完成回归任务校准与受控 KAN–MLP 题目构建
> 不在本计划内：KAN 分类正式题集、FastKAN、符号化、剪枝、动态图网格、图像/语言任务中的 KAN 变体

## 1. 结论摘要

本计划采用以下方案：

1. 冻结原始 `profiles/v1.yaml`，其内容以当前分支 `HEAD`（`16e36af`）中已提交的版本为准。
2. 当前工作区中尚未提交的分类 profile 改动不丢弃，而是迁移到新的 `profiles/v2.yaml`。
3. KAN 也只进入 V2，不进入 V1。
4. 第一版 KAN 使用仓库内自包含、纯 PyTorch 的 B-spline 实现，采用 efficient-KAN 风格，不依赖 `pykan`。
5. 第一版只在一元和多元回归任务中启用 KAN；代码预留 `output_dim`，但阶段 3 完成前不正式启用分类 KAN。
6. 主评测仍使用 ArchitectureIQ 当前的“相同训练样本预算”协议；参数量和运行成本作为诊断指标单独报告，不混成一个综合分数。
7. 阶段 3 同时构建普通异构架构题和受控 KAN–MLP 对比题，并明确区分“严格 architecture-only”与“模型和训练配方联合选择”。

## 当前实施记录（2026-07-15）

- 阶段 1 已完成：原始 V1 恢复为冻结配置；分类与 KAN 进入 `profiles/v2.yaml`；candidate/set/question run 记录 `profile_hash`。
- 阶段 2 已完成：加入纯 PyTorch、自包含 `efficient_spline_v1` KAN；一元和多元回归可采样、渲染、导入并执行 Ground Truth；prompt formatter、inspector 和参数量工具已接通。
- 阶段 2 验证：KAN 专项和相关回归/inspector 测试通过；排除本机 NumPy/Matplotlib 进程级 abort 的两个既有测试文件后，其余测试通过。
- 阶段 3 初筛已完成：在小预算、2 seeds 的代表性回归配置中，KAN forward/backward 和训练均有限且不产生 NaN；但当前小配置的 MSE 仍可能高于全局 `fail_threshold=2.0`，因此正式 V2 KAN 参数池尚未冻结。
- 阶段 3 未完成项：正式多实例校准、参数量匹配候选、受控 KAN–MLP 题集、训练成本报告，以及是否调整 V2 的 KAN 失败阈值。

## 2. `v1.yaml` 是什么，为什么有 Git 仍然需要冻结

### 2.1 Git 版本与 profile 版本不是一回事

Git 负责记录代码和文件的历史，因此我们始终可以找到某个旧提交中的 `v1.yaml`。但 `profiles/v1.yaml` 还有另一层含义：它是一次 benchmark 运行所采用的**命名配置协议**，决定：

- 有哪些数据集 family；
- 有哪些模型类型；
- 模型、优化器、损失和预算从哪些池中采样；
- 每个候选跑多少个随机种子；
- 显著性阈值如何设置；
- 最终可能生成什么分布的题目。

如果持续修改同名的 `v1.yaml`，那么两个结果即使都写着“V1”，也可能来自完全不同的候选池。Git 能帮助事后追溯，但不能保证结果在命名和含义上天然可比。

因此，冻结 profile 的目的不是限制后端代码演进，而是保证：

> `V1` 始终指向同一套 benchmark 配置语义；后端可以继续修复和扩展，但新任务、新模型池和新采样协议进入 `V2`。

### 2.2 本计划中的冻结边界

冻结对象是 `profiles/v1.yaml` 的配置语义，不是整个仓库：

- 可以修复通用后端 bug；
- 可以增强注册表、渲染器、inspector 和测试；
- 可以继续读取和执行 V1 artifacts；
- 不再向 V1 增加分类 family、KAN 或新的候选池选项；
- 如果修复会改变 V1 已有候选的实际训练结果，必须明确记录为兼容性修复，不能悄悄覆盖旧结果。

### 2.3 V2 与 schema version 的区别

`profile: v2` 表示 benchmark 配置版本。`schema_version` 表示 JSON artifact 的结构版本，两者不应混为一谈。

第一阶段默认：

```yaml
profile: v2
schema_version: "1.0"
```

只要 `dataset_spec.json`、`candidate_spec.json` 等结构仍向后兼容，就继续使用 schema `1.0`。只有 artifact 字段结构发生不兼容变化时，才提升 schema version。

同时，V2 应在生成的 dataset、candidate set、candidate 和 question manifest 中记录：

- `profile_name`，例如 `v2`；
- `profile_hash`，即 profile 文件内容的稳定哈希；
- Git commit 和运行环境继续按现有机制记录。

这样即使未来出现 `v2.1` 式调整，也能精确确认每个 artifact 使用了哪份配置。

## 3. KAN 首批结果的公平性定义

“公平”没有唯一答案。不同匹配方式回答不同的研究问题，因此本计划不强行把它们压成一个分数。

### 3.1 主协议：相同训练样本预算

正式 ArchitectureIQ 题目继续沿用当前协议：

- 使用同一个物化后的数据集实例；
- 相同 `total_samples_seen`；
- 严格 `architecture_only` 题中，batch size、优化器、学习率、损失函数和训练预算完全相同；
- 使用相同的训练与测试数据；
- 使用相同随机种子集合；
- 按 family 的 selection metric 排名，回归任务仍为 `test_mse`；
- Ground Truth 必须来自执行候选目录中生成的 `model.py`、`optimizer.py`、`loss.py` 和 `train.py`。

这个协议回答：

> 在看过相同数量训练样本、采用相同训练配方时，哪种架构取得更好的泛化误差？

它的优点是与当前 ArchitectureIQ 预算定义一致，也容易生成严格的 architecture-only 问题。

它不保证以下量相同：

- 参数量；
- 每步 FLOPs；
- CPU 运行时间；
- 内存占用；
- 每个架构各自最优的学习率。

因此主协议能够比较架构在当前训练制度下的结果，但不能被解释为“相同计算成本下 KAN 更好/更差”。

### 3.2 诊断协议 A：参数量近似匹配

KAN 的每条连接包含多个 spline 系数，所以 KAN 的 `width=32` 与 MLP 的 `width=32` 并不等价。

阶段 3 为 KAN 候选寻找参数量接近的 MLP 对照：

- 实际参数量由执行模型的 `sum(p.numel() for p in model.parameters())` 得到，不用手写估算公式作为最终真值；
- 理想匹配误差不超过 10%；
- 如果离散宽度无法达到 10%，允许放宽到 20%，但必须在结果中标记；
- 参数匹配不改变主 selection metric，只用于分析和挑选受控问题。

这个协议回答：

> 在可训练参数规模近似相同时，KAN 与 MLP 的归纳偏置和参数效率有何差异？

### 3.3 诊断协议 B：运行成本

阶段 3 记录但不用于主排名：

- 单 seed 训练耗时；
- 峰值内存（如果可以稳定采集）；
- 每个最终成功候选的总运行时间；
- 失败 seed 数量；
- 单位成功候选的计算成本。

第一版不把 wall-clock time 变成正式预算，因为它对机器、PyTorch 版本、线程数和系统负载敏感。所有时间比较必须来自同一设备和相同运行环境。

这个协议回答：

> KAN 的效果收益是否值得它增加的真实运行成本？

### 3.4 训练配方公平性：共同配方与各自调优必须分开

KAN 与 MLP 的最优学习率可能不同。因此需要区分：

#### 严格 architecture-only

- 模型架构变化；
- optimizer、lr、weight decay、loss、batch size 和预算完全相同；
- 用于正式的 `architecture_only` 标签。

它回答“只换架构会怎样”。

#### Architecture + recipe / co-design

- 每个模型可以使用经校准后更适合自己的 optimizer 和 lr；
- KAN 和 MLP 都在相同调参预算内选择配置；
- 不能再标记为严格 `architecture_only`，应归入 `mixed` 或未来单独的 `architecture_recipe` 类型。

它回答“把模型作为一个完整训练系统时，哪个方案最好”。

两类结果都重要，但不能放在同一个分数中解释。

### 3.5 本计划采用的最终公平性决定

1. 正式题目主协议：相同样本预算。
2. 严格 architecture-only：同时固定训练配方和 batch size。
3. 参数匹配：作为受控题筛选条件和分析维度。
4. 运行时间：作为成本诊断，不参与正确答案判定。
5. 各架构独立调优：进入 co-design/mixed 分析，不冒充 architecture-only。

## 4. V2 的计划内容

新的 `profiles/v2.yaml` 以冻结后的 V1 为基础，并加入：

### 4.1 数据集 family

- `univariate_regression`
- `multivariate_regression`
- `bigram_lm`
- `synthetic_tabular_classification`

### 4.2 模型类型

- `mlp`
- `transformer_lm`
- `kan`

模型类型仍需与数据集 family 求兼容交集：

| Dataset family | 阶段 3 结束时允许的模型 |
|---|---|
| `univariate_regression` | `mlp`, `kan` |
| `multivariate_regression` | `mlp`, `kan` |
| `bigram_lm` | `transformer_lm` |
| `synthetic_tabular_classification` | `mlp`；KAN 暂不正式启用 |

### 4.3 第一版 KAN 参数空间

第一轮不设置过大的笛卡尔积。初始 profile 建议：

```yaml
kan:
  variant: efficient_spline_v1
  depth: [1, 2, 3]
  width: [4, 8, 16, 32]
  grid_size: [3, 5, 8]
  spline_order: [3]
  grid_range:
    - [-1.0, 1.0]
  base_activation: [silu]
```

阶段 2 的 smoke test 可以覆盖完整小网格；阶段 3 正式校准前必须根据失败率和成本收缩参数池。

### 4.4 第一版明确不加入的 KAN 功能

- 训练中动态更新 spline grid；
- 根据测试数据调整网格；
- symbolic formula 提取；
- pruning；
- KAN 专用 entropy/sparsity loss；
- LBFGS 专用训练循环；
- 外部 `pykan` 运行时依赖；
- FastKAN/RBF 近似；
- KAN Transformer、ConvKAN 或 KAN adapter。

这些能力以后可以作为独立、可审计的模型或训练协议加入，不能暗中改变 `efficient_spline_v1` 的实现含义。

## 5. 阶段 1：冻结 V1，建立 V2 配置边界

### 5.1 目标

让 V1 保持原有 benchmark 含义，并让分类和 KAN 的后续开发都进入 V2。

### 5.2 实施事项

1. 将工作区中的 `profiles/v1.yaml` 恢复为当前 `HEAD` 中已提交的 V1 内容。
2. 以当前 V1 为基底创建 `profiles/v2.yaml`。
3. 把当前尚未提交的分类 profile 内容迁移进 V2。
4. 在 V2 中加入 KAN 参数段和模型池声明。
5. 保持 V1 原有 family、模型池、预算、随机种子数和显著性设置不变。
6. 确保 `load_profile("v1")` 和 `load_profile("v2")` 都能工作。
7. 在生成 artifacts 时加入 `profile_name` 与 `profile_hash`。
8. 增加 profile 版本测试：
   - V1 的关键池内容与冻结快照一致；
   - V2 包含分类和 KAN；
   - V1 不包含分类或 KAN；
   - profile hash 对同一内容稳定。
9. 在 README 中简要说明 V1 与 V2 的区别，但不改写 AGENTS 中的通用架构规则。

### 5.3 预计涉及文件

- `profiles/v1.yaml`
- `profiles/v2.yaml`
- `src/architecture_iq/profile.py`
- dataset/candidate/question manifest 的构建位置
- profile 相关测试
- `README.md`

### 5.4 完成标准

- V1 文件恢复且无语义新增；
- V2 能被 CLI 正常加载；
- 当前分类功能只通过 V2 暴露；
- 新生成 artifact 能确认所用 profile 的名称和内容哈希；
- 旧 V1 artifact 仍可读取和执行。

## 6. 阶段 2：实现自包含 KAN ModelFamily

### 6.1 目标

实现一个能通过现有 spec → code → execution → GT 管线运行的纯 PyTorch KAN 模型插件。

### 6.2 模型定义

第一版 KAN 采用以下结构：

- 每层为 `KANLinear`；
- 每条输入—输出连接包含可训练 B-spline 系数；
- 保留 base activation 分支，默认 `SiLU`；
- spline 分支与 base 分支相加得到该层输出；
- 网络结构为输入投影、若干同宽隐藏层和输出层；
- 支持任意正整数 `input_dim` 与 `output_dim`；
- 所有 grid 和参数初始化由 model spec 决定；
- 构造模型前由现有 `train.py` 设置 PyTorch seed；
- forward 不执行随机操作，也不读取训练/测试数据以外的隐藏状态。

### 6.3 建议的 model spec

```json
{
  "type": "kan",
  "variant": "efficient_spline_v1",
  "input_dim": 1,
  "output_dim": 1,
  "depth": 2,
  "width": 16,
  "grid_size": 5,
  "spline_order": 3,
  "grid_range": [-1.0, 1.0],
  "base_activation": "silu"
}
```

其中：

- `depth` 与 MLP 的显示语义保持一致，表示同宽隐藏 block 数；
- 输入到 width、隐藏 width 到 width、width 到输出都使用 KAN layer；
- `variant` 冻结具体实现，避免未来优化代码后同名候选含义漂移；
- 回归输入 `[0,1]` 位于初始 `[-1,1]` 网格内；隐藏层越界由 base 分支提供连续输出，阶段 3 再验证固定网格是否足够稳定。

### 6.4 接入点

1. 新增 `src/architecture_iq/models/kan.py`。
2. 实现 `validate`、`build_module`、`render_model_py` 和 `sample_spec`。
3. 在 registry 中注册 `KanModelFamily`。
4. 在一元和多元回归 family 中加入 `kan` 兼容声明。
5. 为 package prompt formatter 增加 KAN 的中文无关、英文 benchmark 描述。
6. 同步 question inspector 的 formatter mirror，并保持 parity test。
7. 如果 custom settings UI 在阶段 2 同时支持 KAN，则增加 KAN 参数输入；否则先保证已有 artifact 可查看，并把手动编辑 KAN 设置留到阶段 3。
8. 不新增 KAN 专用 Ground Truth runner 或训练循环。

### 6.5 测试要求

- 合法/非法 model spec 校验；
- 一元输入 forward shape；
- 多元输入 forward shape；
- `output_dim > 1` 的结构预留测试；
- backward 后所有应训练参数梯度有限；
- 固定 seed 下初始化可复现；
- `build_module` 与生成的 `model.py` 结构和参数量一致；
- 生成代码可被 runtime loader 独立导入；
- 小预算回归 GT smoke test；
- prompt 能正确展示 KAN 参数和执行代码；
- inspector formatter 与 package formatter 保持一致。

### 6.6 Prompt 长度检查

现有 renderer 会展示 `model.py` 中所有顶层 class。KAN 实现比 MLP 长，因此阶段 2 必须记录：

- 单个 KAN choice 的模型代码字符数和 token 估计；
- 2-choice、4-choice 问题的完整 prompt 长度；
- KAN 公共实现是否在不同 choice 中大量重复。

阶段 2 先保证正确性。如果重复导致 prompt 明显过长，阶段 3 再做“公共实现展示一次、每个 choice 展示实例化参数”的去重设计；去重后仍必须让读者看到实际执行的完整公共实现。

### 6.7 完成标准

- 所有 KAN 单元测试通过；
- KAN 候选可由 V2 profile 采样；
- KAN 候选能执行完整 GT；
- 生成代码和执行代码一致；
- V1 行为不受影响；
- 尚未将 KAN 正式开放给分类 family。

## 7. 阶段 3：回归校准、成本测量与受控题目构建

### 7.1 目标

找到稳定、成本可接受、能产生有意义 KAN–MLP 对比的 V2 回归参数池，并生成第一批受控 architecture questions。

### 7.2 校准轮次 A：快速稳定性筛选

建议规模：

- 一元回归 4 个 dataset instances；
- 多元回归 4 个 dataset instances，覆盖至少 2、4、8 维；
- 2 个中小预算；
- 每个候选先跑 2 seeds；
- 重点覆盖 KAN 的 depth、width、grid size 和 Adam 学习率。

检查内容：

- 是否出现 NaN/Inf；
- 失败 seed 比例；
- 梯度或输出是否异常爆炸；
- KAN 相对 MLP 的参数量倍率；
- 每个训练 step 和完整候选的运行时间；
- 最大预算下是否可能不可接受地慢。

校准轮次 A 后删除明显不稳定或成本极高的参数组合，不直接把全部初始网格放进正式 V2 候选池。

### 7.3 校准轮次 B：收缩网格后的正式校准

建议规模：

- 每个回归 family 至少 10 个 dataset instances；
- 覆盖小、中、大 3 个训练预算；
- 选择 2—3 个通过轮次 A 的 KAN 配置族；
- 为每个 KAN 配置寻找参数量近似的 MLP 对照；
- 初筛使用 3 seeds；
- 只有进入问题候选池的配置再按正式 V2 `n_seeds` 重跑。

输出至少包括：

- `test_mse` 均值、标准差；
- seed 级胜率；
- 失败 seed 数量；
- 参数量；
- 训练耗时；
- family、输入维度、表达式结构、预算；
- KAN 的 depth、width、grid size；
- 对照 MLP 的 depth、width、activation 和参数匹配误差。

### 7.4 参数池收缩规则

满足以下条件的区域才进入正式 V2 KAN pool：

- 选定配置区域的 seed 成功率至少 90%；
- 没有系统性 NaN/Inf；
- 在多个 dataset instance 上可训练，而不是只对一个函数有效；
- 运行成本仍能支持 10-seed Ground Truth；
- 至少存在一部分 KAN 胜、一部分 MLP 胜的实例，避免形成“看到 KAN 就固定选择/排除”的平凡启发式；
- 参数池规模足以产生多样性，但不能大到让随机采样被大量无效组合占据。

如果 KAN 只在非常窄的 optimizer/lr 区域稳定，则先在 V2 中使用受限的 KAN 校准配置，不通过 generic sampler 硬编码模型特例。

### 7.5 题目分为两组

#### A. 普通异构架构题

- 从 V2 允许的模型池中正常采样；
- 可以出现 MLP–MLP、KAN–KAN 或 MLP–KAN；
- 反映完整候选池中的实际选择任务；
- 报告中统计不同 pair type 的占比。

#### B. 受控 KAN–MLP 题

- 每题至少包含一个 KAN 和一个 MLP；
- 同 dataset instance；
- 同 `total_samples_seen`；
- 同 batch size；
- 同 optimizer、lr、weight decay 和 loss；
- 参数量尽量匹配，并记录误差；
- 继续使用现有 gap、win-rate 和 non-overlap 显著性条件；
- 一元和多元回归分别统计，不能只给合并数字。

为保证跨模型 pair 的数量，阶段 3 可以增加 profile 驱动的分层采样或一个通用的“required model types”候选构建参数，但不能把 `kan` 名称硬编码进通用 question logic。

### 7.6 防止题目被简单启发式击穿

阶段 3 要检查：

- 是否几乎所有一元回归都由 KAN 获胜；
- 是否 KAN 因计算或训练不稳定而几乎总输；
- 是否参数量本身就能预测答案；
- 是否 grid size、width 或预算存在单调到过于明显的规律；
- 是否某个 optimizer 对 KAN 的系统性不适配决定了答案；
- 是否 prompt 中模型类型名称本身比函数结构更能预测答案。

至少报告这些简单 baseline：

- 总选 MLP；
- 总选 KAN；
- 总选参数更多者；
- 总选参数更少者；
- 按模型类型和预算查表；
- 按训练成本查表。

如果简单 baseline 已经接近满分，应优先调整实例分布、参数池或受控配对，而不是立即发布题集。

### 7.7 阶段 3 完成标准

- V2 中存在经过收缩的稳定 KAN 参数池；
- 一元、多元回归都能生成并执行 KAN 候选；
- 正式候选按 V2 seed 数运行；
- 参数量、耗时和失败率可追踪；
- 能生成普通异构题与受控 KAN–MLP 题；
- 严格 architecture-only 与 co-design/mixed 结果分开；
- 简单模型类型启发式不能轻易击穿整套受控题；
- prompt 长度在目标模型上下文限制内，或已经完成公共代码去重；
- 形成一份阶段 3 校准报告和最终 V2 KAN 参数表。

## 8. 阶段 1—3 的预期文件改动范围

预计新增：

- `profiles/v2.yaml`
- `src/architecture_iq/models/kan.py`
- KAN 模型和 V2 profile 测试
- KAN 校准脚本或通用模型对比脚本
- 阶段 3 校准结果文档

预计修改：

- `src/architecture_iq/profile.py`
- `src/architecture_iq/registry.py`
- 两个 regression family 的兼容模型列表
- `src/architecture_iq/prompts/formatters.py`
- `tools/question_inspector/prompt_format.py`
- 必要时修改 inspector custom settings UI
- candidate/dataset/question manifests 的 profile provenance
- `README.md`

原则上不应为 KAN 修改：

- selection metric 定义；
- regression loss 实现；
- Ground Truth 的执行来源；
- significance validator 的核心判定；
- V1 的候选池语义。

## 9. 风险与应对

| 风险 | 后果 | 应对 |
|---|---|---|
| KAN 生成代码过长 | prompt token 大幅增加 | 先测量，再做公共实现去重 |
| KAN 每步过慢 | 10 seeds 和大预算不可承受 | 小网格筛选、记录成本、收缩正式池 |
| 固定 grid 不适合隐藏表示 | 训练不稳定或性能差 | 先调范围/初始化；动态图更新留给新 variant |
| 同 width 导致严重参数失配 | 对比结论误导 | 使用实际参数量做受控配对 |
| KAN 只适合特定 lr | 随机池中大量失败 | profile 校准；不在 generic code 中硬编码 |
| KAN 几乎总赢或总输 | 题目被类型启发式击穿 | 扩充实例结构、调整参数池、平衡配对 |
| V1/V2 artifact 混淆 | 结果不可追溯 | 记录 profile name + content hash |
| 分类与 KAN 同时改动过多 | 难以定位问题 | 按阶段和提交拆分，回归 KAN 先闭环 |

## 10. 推荐提交顺序

1. `Freeze v1 profile and introduce v2 profile`
2. `Record profile provenance in generated artifacts`
3. `Add standalone spline KAN model family`
4. `Enable KAN for regression families in v2`
5. `Add KAN prompt and inspector support`
6. `Add KAN render, runtime, and GT tests`
7. `Add regression KAN calibration tooling`
8. `Add controlled cross-model candidate sampling`
9. `Finalize calibrated v2 KAN grids and report`

每个提交都应保持现有测试可运行，避免把分类迁移、KAN 实现、题目采样和 prompt 重构压成一个无法审查的大提交。

## 11. 决策日志

### 已决定

- `[已决定]` 冻结 V1，分类与 KAN 进入 V2。
- `[已决定]` Git 版本与 benchmark profile 版本分别管理。
- `[已决定]` 第一版使用纯 PyTorch、自包含 B-spline KAN。
- `[已决定]` 第一版不依赖 `pykan`。
- `[已决定]` 第一版不做动态 grid、符号化或剪枝。
- `[已决定]` 阶段 3 前只正式启用回归 KAN。
- `[已决定]` 主协议使用相同训练样本预算。
- `[已决定]` 参数量和运行时间作为独立诊断维度。
- `[已决定]` 严格 architecture-only 与 architecture+recipe/co-design 分开报告。

### 阶段 3 后再决定

- `[待决定]` 是否在分类 family 中正式开放 KAN。
- `[待决定]` 是否新增 FastKAN/RBF 模型类型。
- `[待决定]` 是否为动态 grid 建立新的 KAN variant。
- `[待决定]` 是否建立计算预算匹配的独立 benchmark protocol。
- `[待决定]` 是否将受控 KAN–MLP 题作为 V2 主榜的一部分，或单独作为 diagnostic split。

## 12. 阶段 3 结束时应交付的结果

1. 冻结且可继续运行的 V1 profile。
2. 包含分类配置与 KAN 配置的 V2 profile。
3. 自包含、可渲染、可执行的 KAN ModelFamily。
4. 一元与多元回归 KAN 的稳定参数池。
5. 普通异构题和受控 KAN–MLP 题生成能力。
6. 包含 MSE、胜率、参数量、失败率和运行成本的校准报告。
7. 清晰的 profile provenance，能追溯每个 artifact 的具体配置。
8. 对分类 KAN、FastKAN 和计算匹配协议是否进入下一阶段的证据基础。
