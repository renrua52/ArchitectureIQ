# ArchitectureIQ 题包扩展、前端同步与 Demo 发布统一计划

状态：M1–M2 已完成；M3 GPU 受硬件限制；M4–M5 按容量形成 46 题 pilot；M6 发布候选已通过代码/静态/API 验收，待人工审计
来源讨论：题包扩展对话 `019f8378-b3ce-7ca3-b16f-42dbb3cea35d`、KAN 合并对话 `019f64d9-9172-72b3-9866-2a07ca2bbb25`
适用主线：`ArchitectureIQ-main-integration`

## 1. 总目标

在不破坏历史 benchmark 语义和已有题目的前提下，完成一条可发布的 demo 闭环：

```text
远端 GitHub main 前端同步
→ collection 驱动的统一答题入口
→ 定向 candidate/GT 扩展
→ 题包生成与 Gate 1–4 审计
→ CPU 并行/GPU 性能评估
→ 人工与 Luna 影子审计
→ 固定 demo manifest
→ GitHub 发布前验证
```

本计划同时覆盖：

- 普通题包扩展；
- 分类 KAN 独立 diagnostic 题包；
- profile 和 gap 规则演进；
- GitHub 远端 `main` 的前端兼容；
- collection/API/静态导出；
- 最终 CI、构建和发布检查。

## 2. 已确定的边界

### 2.1 题包规模与顺序

第一轮每个题包生成 10 道二选一题。

按以下顺序执行：

1. `architecture`
2. `optimizer`
3. `mixed`
4. `loss`
5. `architecture_easy`
6. `architecture_hard`

分类 KAN diagnostic 作为独立题包，第一轮也以 10 道为目标，不自动计入普通 classification architecture 题包。

因此第一轮完整 pilot 的上限约为 70 道：6 个普通题包各 10 道，加 10 道分类 KAN diagnostic。实际数量以 Gate 和 candidate 容量为准。

### 2.2 Profile 组织

Profile 是 benchmark 语义合同，不是前端页面分类。

- `v1`：历史 V1 语义，保持不变；
- `v2`：冻结的当前 V2 语义，保持不变；
- `v2.1`：已接入 KAN 和分类 KAN 的扩展语义；
- `v2.2`：已用于 KAN 扩展 profile 和 classification KAN pool；保持该语义冻结；
- 若未来正式改变 metric/family/loss-aware significance 协议，则创建后续版本（`v2.3+`），不改写 `v1`/`v2`/`v2.1`/`v2.2`。

统一前端可以展示多个 track，但每道题和每个 question run 必须记录：

- `profile`；
- `profile_hash`；
- `track`；
- source run 和 collection manifest。

不同 profile 的 candidate 不得混入同一道题。历史题、普通 V2/V2.1 题和 KAN diagnostic 题可以在同一个前端入口中出现，但必须按 track 明确标识。

### 2.3 Collection 去重

最终 demo collection 要求跨所有题包 global candidate-disjoint：

- 一个 candidate 只能出现在最终 collection 的一道题中；
- 探索性校准和内部分析允许复用 candidate；
- pilot 生成阶段就提前执行全局去重，避免最后才发现容量不足。

当前 24 题审计 collection 保持历史独立，不与新 demo collection 共享去重范围。

### 2.4 Execution device 与性能实验

正式题目优先使用 CPU。

CPU 性能实验包括：

- 串行 candidate/seed；
- 多进程或多任务 candidate 并行；
- PyTorch 多线程设置；
- 多 seed 并行吞吐；
- 运行时间、峰值内存、失败率。

GPU 性能实验包括：

- 单 candidate latency；
- 多 seed 吞吐；
- CPU/GPU 加速比；
- 显存和失败率。

并行化只改变调度，不改变：

- candidate 内部 seed；
- seed 顺序和语义；
- 训练预算；
- 生成代码；
- GT 结果定义。

同一正式 collection 内不混用 CPU/GPU 生成的结果。若 CPU 多核并行已经达到实用吞吐，本轮正式生成使用 CPU；GPU 结果仅作为后续是否迁移计算方案的依据。

## 3. 前端同步策略

### 3.1 正确的同步对象

本轮同步对象是 GitHub 远端仓库的最新 `origin/main`，不是 `frontend/vanilla` 分支。

当前本地开发分支已有 KAN、Inspector、题包审计和 collection 相关 dirty changes，因此不能直接覆盖或整分支 merge。

采用：

```text
获取最新 origin/main
→ 在隔离 worktree 比较远端 main 与本地开发主线
→ 识别前端/后端 adapter 的增量
→ 选择性移植并适配现有 artifact schema
```

### 3.2 前端职责

React/远端 main 前端定位为可分享的 demo quiz；现有 Streamlit Inspector 继续承担：

- 深度题目审计；
- custom setting 训练；
- 训练进度和临时曲线；
- 历史 artifact 调试；
- 详细数据集和模型诊断。

两者共存，不把 Inspector 的全部能力强行搬到 React 第一版。

### 3.3 Collection 驱动 API

前端不能再通过“最新或题目最多的 run”自动选题，必须由 collection manifest 驱动。

目标 API：

```text
GET  /api/collections
GET  /api/collections/{collection_id}
GET  /api/collections/{collection_id}/questions
GET  /api/questions/{question_id}
POST /api/questions/{question_id}/answer
```

前端必须支持：

- collection/track 下拉切换；
- question 下拉快速跳转；
- Next 严格按 manifest 顺序推进；
- 可选的 Random practice 模式；
- 答案锁定前不显示 GT；
- 提交后显示结果和反馈；
- 页面显示 profile、profile hash、track 和 question provenance。

### 3.4 可视化兼容层

后端提供 family-aware visualization payload，前端不重新猜测数据集语义。

至少覆盖：

- 一元回归：train/test 曲线或散点；
- 多元回归：明确的二维投影/降维说明；
- synthetic classification：按类别着色；
- Bigram LM：transition/metric 专用视图；
- KAN：模型参数、参数量和可解释字段。

同时补齐：

- expression 显示；
- `trainable_parameter_count`；
- log-Y 稳健中心线和阴影；
- classification CE/accuracy；
- KAN grid、spline order、base activation。

### 3.5 本轮 React demo 已落地能力

本轮前端已把以下能力接入统一 quiz surface：

- 参数量：choice card 规范显示 `Trainable Parameter Count`；历史 `candidate_spec.json` 缺失参数量时显示 `—`；KAN 保留原有 model type、width、residual、layer norm、activations 等字段。
- 说明：reveal 前提供 family-aware task description 和完整 benchmark instructions；dataset/choice 的 `candidate_spec.json`、model、train、loss、optimizer 等说明文件可通过 info modal 查看。
- 评论与审核：reveal 后记录 confidence 1–5、keep/revise/reject 和自由文本 comment，并通过 `audit_feedback` telemetry 与 session export 保留。
- 分类可视化：synthetic classification 使用规则感知的二维 feature projection，按 class 着色 train 点、outlined test 点，并展示经验 `P(class 1)` 与 feature-pair 说明；bigram、回归和 learning-curve 视图继续按 family 使用各自的 payload。
## 4. 题包生成计划
本轮网页审计范围是本机可信研究人员审阅：static BakeFile 中的 reveal 仍用于本地答题后的即时展示，UI 在答题前不显示。它不等同于公开/匿名盲审的安全边界；若未来对外发布或要求参与者不可从静态资源查看答案，再单独实施 private BakeFile/API answer flow。


### 阶段 A：资产与容量盘点

对已有 dataset/candidate/question artifacts 运行 Gate 1/2 preflight，按以下维度统计容量：

- profile/track；
- dataset family 和 dataset instance；
- question type；
- loss；
- model type；
- budget/device；
- 可用 candidate 数；
- 通过 significance 的 pair 数；
- candidate 全局去重后的剩余容量。

输出：

- 可复用 candidate 清单；
- 需要重跑 GT 清单；
- 需要新 candidate 清单；
- 每个题包的容量预测；
- 失败原因统计。

### 阶段 B：定向 candidate 生成

不盲目随机扩充全部 candidate，而是先固定：

- dataset 配额；
- varying axes；
- budget；
- device；
- easy/hard 目标；
- global candidate-disjoint 预留。

每个题包按目标题数的约 2–3 倍准备候选池，允许 Gate、统计和人工审计淘汰部分 candidate。

分类 KAN diagnostic 使用 `v2.1`，固定独立 track；先验证：

- KAN/MLP 是否存在合理的胜负混合；
- 参数量是否成为答案 shortcut；
- prompt 是否足够表达 KAN；
- classification visualization 是否清晰。

### 阶段 C：题目生成与 Gate 3/4

每个题包生成独立 run 和 manifest，随后执行：

- Gate 3：compatibility、winner、gap、win-rate、seed 对齐；
- Gate 4：public 字段、candidate 去重、collection 泄露和顺序。

题目不足时记录 capacity shortage，不降低确定性门槛。

### 4.1 loss-only deferred 决策

本轮 demo/pilot 对 `loss_only` 明确采取 `deferred`：不生成正式题目，不把它伪装成已完成的零题题包，也不降低 significance gate。

- 1024 budget：7 个 Gate 1/2 pass candidate，显著 pair 为 0；
- 2048 budget：7 个 Gate 1/2 pass candidate，显著 pair 为 0；
- 因此当前冻结 collection 的正式 records 不含 `loss_only`，该项应作为 coverage gap/deferred capability 记录在发布说明或后续 manifest schema，而不是作为 `excluded` 题目记录。
- 只有在新 candidate/profile/run 形成可复核的显著 pair 后，才重新开启 loss-only 题包；不回写冻结 profile 和 collection。
## 5. Gap 与难度规则

### 5.1 历史语义

历史 V1/V2/V2.1 artifact 继续使用其原 profile 中的 significance 规则，不回写旧题。

### 5.2 新 pilot 的分层门槛

新 pilot 采用：

```text
family × selection_metric × question_type
```

必要时细分为：

```text
family × selection_metric × loss × question_type
```

每层分别计算：

- raw gap；
- relative gap；
- robust effect size；
- seed win-rate；
- non-overlap heuristic；
- gap 分位数。

初步规则：

- `gap_min`：最低安全门槛；
- easy：同层 gap 高分位且 winner 稳定；
- hard：同层 gap 中低分位但仍稳定；
- borderline：通过最低门槛但波动或可区分性不足，进入人工复核。

具体数值由阶段 A 的真实 gap 分布推导，不预设跨任务 universal raw gap。

如果最终将这些分层阈值写入正式 profile，则创建 `v2.2.yaml`，并记录：

- 阈值来源数据集；
- 统计分层；
- 版本和 hash；
- 与 V2/V2.1 的差异。

## 6. 人工与 Luna 审计

第一轮 pilot 通过确定性 Gate 后：

1. 用户在统一前端中盲做；
2. 记录答案、置信度、用时和质量判断；
3. Luna 只看 public prompt 做 blind audit；
4. 再提供答案和 Gate 摘要做 post-reveal audit；
5. 汇总 keep/revise/reject/retrain 原因。

Luna 不参与正式随机采样和自动准入。

## 7. GitHub 发布计划

### 阶段 D：代码和前端联调

- 以远端 `main` 为参考创建隔离 integration worktree；
- 选择性移植前端和 API adapter；
- 保留当前 Streamlit Inspector；
- 接入 collection manifest；
- 运行后端 API tests；
- 运行 React build；
- 对 pilot collection 做浏览器 smoke test。

### 阶段 E：发布前冻结

发布前冻结：

- profile hash；
- collection manifest；
- question 顺序；
- candidate 全局去重结果；
- public/private 字段边界；
- CPU/GPU 性能报告；
- 人工和 Luna 审计记录。

GitHub 中原则上提交：

- 代码；
- 配置和 profile；
- collection manifest；
- API/前端测试；
- 必要的静态 demo 数据。

不直接提交大规模原始训练 artifacts，除非另有发布需求。

## 8. 里程碑与验收

### M1：远端 main 同步基线

- 已获取并记录远端 main commit；
- 明确前端增量文件；
- 本地 dirty changes 已隔离；
- 没有整枝删除主线文件。

### M2：前端 collection/API 骨架

- collection 列表和顺序可用；
- question dropdown 和 Next 可用；
- answer lock/reveal 正常；
- API tests 通过。

### M3：性能基线

- CPU 串行/并行结果；
- GPU 加速结果；
- 并行不改变 GT 结果；
- 正式生成设备选择完成。

### M4：第一批题包

- architecture、optimizer、mixed 各 10 道；
- classification KAN diagnostic 10 道；
- 全局 candidate-disjoint；
- Gate 1–4 通过。

### M5：完整 pilot

- architecture_easy、architecture_hard 本轮未生成；loss-only 已 deferred，本轮不生成正式题目；
- 每题有 profile/track provenance；
- 人工和 Luna 审计完成；
- 形成修改和扩容清单。

### M6：GitHub 发布候选

- 远端 main 前端兼容完成；
- React build、API tests、Python focused tests 通过；
- static export 和浏览器 smoke 通过；
- demo manifest/profile hash 冻结；
- 发布内容和历史 artifacts 边界明确。

## 9. 默认执行方式

在不再增加决策的情况下，后续按以下顺序连续执行：

```text
远端 main 只读同步与差异报告
→ 前端/API collection 骨架
→ CPU/GPU 性能基线
→ Gate 1/2 容量盘点
→ architecture/optimizer/mixed pilot
→ classification KAN diagnostic pilot
→ architecture_easy/hard pilot（本轮容量截止，deferred；loss-only deferred，不生成正式题目）
→ 人工/Luna 审计
→ v2.2 阈值决定（如需要）
→ GitHub 发布前冻结与验证
```

每个里程碑只需一次集中验收；中间不需要频繁介入。
