# Question 扩展阶段计划：复用 Candidate/GT，重新生成并人工审阅题目

## 1. 阶段决策

本阶段采用“方案 C”：保留现有 dataset、candidate 和真实 ground truth，不修改历史 artifact；使用当前 question generator 从已有 candidate set 中重新生成 question run。

正式准入只包含 Gate 1、Gate 2、Gate 3、Gate 4。暂不建立由 LLM 自动决定题目去留的 Gate 5。

GPT-5.6-luna 高推理 subagent 可以参与“影子审计”：它负责独立作答、指出歧义、识别可能的 shortcut，并给出修改建议，但它的结论在本阶段不自动接受或淘汰题目。最终通过人工做题，把人的判断、Luna 的判断和确定性 Gate 结果放在一起比较，再决定下一阶段是否需要正式的 Gate 5。

## 2. 阶段目标

完成一个小而完整的 question 扩展闭环：

```text
现有 candidate_spec + results/summary.json
→ Gate 1 配置与 provenance 审计
→ Gate 2 运行结果健康审计
→ 使用当前 generator 重新生成 question
→ Gate 3 比较与统计有效性审计
→ Gate 4 泄露、公开字段与 collection 审计
→ 形成一批新题
→ 用户实际作答并评价题目质量
→ Luna 影子审计
→ 汇总需要修改的 prompt、配置池、难度和题型问题
```

阶段完成时应回答四个问题：

1. 现有 candidate/GT 中有多少可以安全复用？
2. 当前 generator 能生成哪些类型、多少道 candidate-disjoint 新题？
3. 新题在真实作答时是否清楚、公平、有推理价值？
4. 下一轮扩展应优先补 candidate、修改 prompt，还是调整 profile/显著性规则？

## 3. 本阶段不做的事情

- 不原地修改旧 `question.json`、`candidate_spec.json` 或 GT。
- 不为了凑题数立即大规模重新训练 candidate。
- 不把 Luna 的主观评分变成正式准入门槛。
- 不在本阶段冻结最终题库规模或最终 KAN 参数池。
- 不把 dataset_id 不重合之外的更强 OOD 定义静默加入 collection v1。
- 不优先扩展 cross-budget、联合 OOD 等更复杂题型；先把基础题目质量闭环跑通。

## 4. 方案 C 的实现方法

### 4.1 输入

每个待重新出题的 dataset instance 需要：

- `dataset_spec.json`；
- 一个或多个正式 candidate set，包含 `set.json`；
- 每个 candidate 的 `candidate_spec.json`；
- 每个 candidate 的 `results/summary.json`；
- candidate 所属 profile 与当前加载 profile 一致。

V1 和 V2 必须分开生成 question run，不能把不同 profile 语义的 candidate 混入同一道题。

### 4.2 生成命令

通用形式：

```powershell
architecture-iq generate-question `
  <dataset_path> `
  <candidate_set_path_1> `
  <candidate_set_path_2> `
  --num-questions <N> `
  --num-choices 2 `
  --profile <v1-or-v2> `
  --seed <seed>
```

这个过程不重跑训练。它读取已有 `candidate_spec.json` 和 `results/summary.json`，寻找通过 compatibility 和 significance 的 candidate 组合，使用回溯选择 candidate-disjoint subsets，然后生成新的：

```text
questions/run_<N>q_2c_<hash>/
  run.json
  q_<id>/question.json
  q_<id>/prompt.txt
```

`run.json` 应记录 profile、profile_hash、seed、source candidate sets 和 `candidate_reuse_policy=globally_disjoint_within_run`。

### 4.3 旧题的地位

旧 question run 继续保留，作为历史结果和对照材料。本阶段新题必须生成到新的 run 目录，不覆盖旧题。

若旧 question 与新协议不兼容，不修补旧题，而是在迁移报告中标记：

- `historical_only`：仅作为历史记录；
- `rebuildable_from_candidates`：candidate/GT 可复用，可重新出题；
- `candidate_audit_failed`：candidate 或 GT 不满足 Gate 1/2；
- `insufficient_capacity`：通过审计的 candidate 不足以组成目标题数。

## 5. Gate 1：配置、兼容性与 provenance

### 5.1 它要保证什么

Gate 1 回答：“这些 candidate 的配置是否合法、属于同一语义版本，并且可以进行公平比较吗？”

最低检查项：

- dataset、family、candidate_id、profile、profile_hash 完整；
- candidate 的 dataset_id/family 与目标 dataset 一致；
- model type 已注册并与 dataset family 兼容；
- model spec 能通过对应 model family 的 `validate()`；
- optimizer、loss、batch size、budget 位于对应 profile 的允许范围；
- `training_steps × batch_size = total_samples_seen`；
- 同题 candidate 使用相同 execution device；
- candidate set 的 varying/invariant axes 与其成员实际配置一致；
- candidate_id 和规范化 candidate spec 不发生冲突；
- V1 candidate 只由当前 V1 profile 重新出题，V2 同理。

### 5.2 当前已有实现

当前代码已经部分实现：

- candidate sampling 从 profile pool 取值；
- family 暴露 `compatible_model_types()`；
- MLP、KAN、Transformer 等 model family 有 `validate()`；
- `choices_compatible()` 拒绝没有差异、device 不一致和不符合指定单轴类型的比较；
- question generator 检查 candidate 的 dataset_id/family；
- question record 写入 profile、budget、device 等信息；
- run manifest 已记录 `profile_hash`。

### 5.3 本阶段需要增加

增加统一、可离线运行的 Gate 1 preflight，而不是只依赖生成过程中的零散异常。建议提供：

```text
tools/audit_question_inputs.py
```

输出每个 candidate/set 的：

```json
{
  "gate": 1,
  "status": "pass | fail | review",
  "candidate_id": "...",
  "checks": {},
  "reasons": []
}
```

正式新题只使用 `pass` candidate。`review` 不自动进入正式 collection。

### 5.4 Gate 1 完成标准

- 对目标 candidate sets 全量运行；
- 没有未解释的缺失 provenance；
- 不同 profile_hash 不会进入同一个 run；
- 配置不合法时在生成 question 前失败；
- 报告能说明每个被排除 candidate 的具体原因。

## 6. Gate 2：真实运行结果健康度

### 6.1 它要保证什么

Gate 2 回答：“配置不只是语法合法，它是否真的完成了可信的训练和评测？”

最低检查项：

- `results/summary.json` 存在且可解析；
- seed 数量符合 profile 要求；
- seed 标识和顺序可对齐；
- 失败 seed 数不超过 `max_failed_seeds`；
- candidate 未被标记为 `excluded`；
- 选择指标的 mean、std 和各 seed final metric 均为有限值；
- selection metric 与 dataset_spec 一致；
- 必需的 curves/summary 字段完整；
- 没有全体 seed 失败、恒定异常值或明显数值爆炸。

训练时间、显存和“是否学习得足够好”可以记录为诊断字段；本阶段不宜用一个武断的绝对性能阈值淘汰所有较弱 candidate，因为较弱但有效的 candidate 仍可能构成有意义的比较。

### 6.2 当前已有实现

当前 ground-truth runner 已经：

- 为每个 seed 记录 `failed`；
- 汇总 mean/std；
- 统计 `failed_seeds`；
- 根据 profile 的 `max_failed_seeds` 写入 `excluded`；
- question generator 在建立 candidate pool 时过滤 `excluded` candidate。

### 6.3 本阶段需要增加

在 `audit_question_inputs.py` 中加入 Gate 2，并显式检查：

- summary schema 和必需字段；
- seed 数量一致性；
- 非有限值；
- 不同 candidate 的 seed 对齐；
- dataset selection metric 与 summary key 对应；
- summary 与 candidate_spec/profile 的 provenance 对应。

不重新训练即可完成大部分 Gate 2 审计。只有发现 summary 缺失、失败或 provenance 不可信时，才进入“需要重跑 GT”清单。

### 6.4 Gate 2 完成标准

- 所有参与重新出题的 candidate 都有健康的真实 GT；
- 被排除 candidate 有可审计原因；
- 缺失或不健康的 GT 不会进入 question generator；
- 输出“可直接复用”和“需要重跑 GT”的分离清单。

## 7. Gate 3：题目比较与统计有效性

### 7.1 它要保证什么

Gate 3 回答：“这几项 candidate 放在一起是否构成一个有明确、稳定答案的问题？”

最低检查项：

- 至少两个 choice，且配置存在真实差异；
- choices 满足目标题型的 compatibility；
- 同题共享相同 dataset instance；
- budget 信息正确呈现；
- winner 根据 dataset selection metric 正确选择；
- top-2 gap 达到 `gap_min`；
- winner 的 per-seed win rate 达到 `win_rate_min`；
- 启用时通过 mean±std non-overlap heuristic；
- 每个 choice 的 seed 数量相同且按相同 seed 语义比较；
- correct_letter 与 winner candidate 一致。

### 7.2 当前已有实现

`validate_significance()` 当前已经实现：

- 排除 `excluded` candidate；
- mean metric 有限性检查；
- winner 和 runner-up gap；
- per-seed win rate；
- 可选 mean±std non-overlap；
- 返回 winner_index、gap、win_rate 和失败原因。

question generator 已经：

- 先运行 `choices_compatible()`；
- 枚举或随机搜索 significant subsets；
- 从通过的 subsets 中回溯选择 candidate-disjoint 题目；
- 将 significance、evaluation、correct_letter 写入私有 question artifact。

### 7.3 本阶段需要增加

生成后增加统一的 question-run 审计，建议提供：

```text
tools/audit_question_run.py
```

它重新读取 source spec/summary 独立复算并核对：

- choice compatibility；
- metric 方向和 selection metric；
- seed 数量与 seed 对齐；
- gap、win rate、non-overlap；
- correct candidate/letter；
- run 内题目数量和题型分布。

### 7.4 Gate 3 完成标准

- 每道新题均能从 source GT 独立复算出相同答案；
- 没有 seed 数或 seed 语义不一致的比较；
- 每道题记录题型、gap、win rate 和被接受原因；
- 题目不足时明确报告 capacity shortage，不降低门槛偷偷凑数。

## 8. Gate 4：采样泄露、公开内容与 collection

### 8.1 它要保证什么

Gate 4 回答：“题目进入 support/holdout 后，是否泄露答案或复用了已反馈的 candidate？”

最低检查项：

- 一个 question run 内 candidate_id 不重复；
- 一个 evaluation collection 内所有题目 candidate_id 全局不重复；
- public prompt 不包含 correct_letter、GT、mean metric、summary 路径等私有信息；
- public question 与 source prompt 完全一致；
- support/holdout question_id 和顺序与 manifest 一致；
- ID collection 满足 `holdout_dataset_ids ⊆ support_dataset_ids`；
- OOD collection 满足 support/holdout dataset_id 不重合；
- support 可以返回 feedback，holdout 不返回 feedback；
- holdout 开始前冻结 lessons。

### 8.2 当前已有实现

当前已有三层约束：

1. question generator 使用回溯选择，保证新 run 内 candidate-disjoint；
2. `build_leakage_safe_collection.py` 跨一个或多个 run 再做全局 candidate_id 去重并划分 ID/OOD；
3. `validate_leakage_safe_collection.py` 检查 manifest、source question、public 字段、candidate 复用和 dataset split 规则。

`leakage_safe_feedback_session.py` 已负责 support feedback、lesson 冻结和 holdout 无反馈状态机。

### 8.3 本阶段需要增加

- 将 `audit_question_run.py` 的通过作为 collection builder 的前置条件；
- 在迁移报告中记录 source run、new run、collection 的完整映射；
- 对生成后的新 run 和 collection 运行现有 validator；
- 保存 deterministic audit 报告，避免只保留终端输出。

### 8.4 Gate 4 完成标准

- 新 run 声明并满足 run 内 candidate 全局不复用；
- 新 collection 通过现有 leakage-safe validator；
- public artifact 中无私有答案/性能字段；
- ID/OOD 规则和 feedback 状态机检查通过。

## 9. 第一批新题计划

第一批目的是建立质量反馈，不追求大规模。默认使用 2 choices，降低 candidate 消耗，并让人工判断更容易定位问题。

### 9.1 建议题目轨道

| 轨道 | Profile | 数据/候选来源 | 希望观察的问题 |
|---|---|---|---|
| A. Regression mixed | V1 | 已有较大的 model+optimizer candidate sets | mixed 题是否信息过载、是否存在明显 optimizer shortcut |
| B. Bigram mixed/optimizer | V1 | 已有 bigram candidate/GT | prompt 是否过长，模型与 optimizer 信息是否足够区分 |
| C. Classification architecture-only | V2 | 3 个 classification dataset，各有 12-candidate model-varying set | architecture-only 是否清楚，KAN/MLP 比较是否合理 |
| D. KAN vs MLP regression | V2 | 已有多个 2-candidate 配对 set | KAN 配置是否可理解，题目是否只能猜测，参数描述是否需要调整 |

### 9.2 最低有效产出

- 先尝试生成 16～24 道新题；
- Gate 1–4 后至少保留 12 道可人工作答的新题；
- 至少覆盖 3 个 dataset family；
- 至少包含 architecture-only 和 mixed 两种题型；
- 尽可能包含 optimizer-only，但如果现有 candidate pool 无法构造，不降低 compatibility/significance 门槛；
- 所有进入同一 collection 的题目保持 candidate_id 全局不重复。

如果最终不足 12 道，阶段仍可完成，但必须输出准确的 candidate 缺口，作为下一轮扩展依据。

## 10. 用户人工做题与质量审阅

人工做题只针对本阶段重新生成并通过 Gate 1–4 的新题，不使用旧题代替。

### 10.1 作答顺序

每道题采用以下过程：

1. 只查看 public prompt；
2. 选择答案并记录理由、置信度和用时；
3. 提交答案后再查看 evaluator feedback；
4. 不只判断“答对没有”，还判断题目本身是否值得保留；
5. 记录具体需要修改的位置，而不是只写“题目不好”。

### 10.2 每题记录模板

```text
question_id:
answer:
confidence: 1-5
time_spent:

在揭晓答案前：
- 我是否理解题目要比较什么？
- 哪些信息真正影响了判断？
- 是否有缺失、歧义或不必要的信息？
- 我是在推理，还是只能猜测？

揭晓答案后：
- 正确答案是否让我觉得合理？
- GT 差距与题面呈现的难度是否匹配？
- 是否存在明显 shortcut？
- candidate 配置是否像有意义的实验，而不是随机拼接？
- prompt 是否过长或关键内容不突出？
- 如果修改，只改 prompt、配置池、题型，还是 significance 门槛？

quality_decision: keep | revise | reject
suggested_changes:
```

### 10.3 人工审阅重点

优先判断：

- 可理解性：题面是否能在合理时间内读懂；
- 公平性：必要信息是否都展示了；
- 推理价值：是否需要架构/优化知识，而不是查格式或猜随机数；
- 配置自然度：候选是否像真实可用的实验设置；
- 难度：太显然、合理困难，还是信息不足导致的伪困难；
- 信息密度：代码和自然语言是否过长、重复或遮蔽关键差异；
- 反馈价值：support feedback 是否能形成可迁移的 lesson。

## 11. Luna 高推理 subagent 影子审计

### 11.1 定位

Luna 在本阶段是 reviewer，不是正式 Gate。它的输出用于：

- 帮助发现人工可能漏掉的歧义；
- 判断题目是否存在强 shortcut；
- 检查候选配置是否看起来不自然；
- 提供 prompt 精简或重点突出建议；
- 与用户的真实做题体验进行对照。

### 11.2 每道题的两阶段调用

阶段一：盲审。

- subagent 只接收 public prompt；
- 不允许读取 repository、question.json、summary 或 GT；
- 输出答案、置信度、推理摘要、歧义、缺失信息、shortcut 风险和配置自然度；
- 不向它透露正确答案。

阶段二：揭晓后审计。

- 在阶段一输出冻结后，再提供 correct answer、Gate 1–4 摘要、gap/win-rate 等私有审计数据；
- 要求它判断错误来自知识不足、题面不足、配置异常还是统计难度；
- 输出 `keep | revise | reject` 建议及具体修改位置。

### 11.3 Luna 输出格式

```json
{
  "question_id": "...",
  "blind_answer": "A",
  "confidence": 0.0,
  "clarity": 1,
  "reasoning_value": 1,
  "configuration_plausibility": 1,
  "ambiguity_flags": [],
  "shortcut_flags": [],
  "post_reveal_diagnosis": "...",
  "recommendation": "keep | revise | reject",
  "suggested_changes": []
}
```

### 11.4 与人工判断的合并规则

- Gate 1–4 失败：题目不能因 Luna 或人工认为“看起来不错”而进入正式集合；
- 用户发现关键歧义或不公平：优先进入 `revise/reject`；
- 用户与 Luna 都指出同一问题：列为高优先级修改；
- 只有 Luna 指出问题：进入复查，不自动删除；
- 只有用户指出问题：保留为重要人工证据，不以 Luna 未发现为由忽略；
- 本阶段结束后统计人工与 Luna 的一致/分歧类型，再决定是否设计正式 Gate 5。

## 12. 实施工作包

### WP1：资产盘点与 Gate 1/2 preflight

产出：

- `question_input_audit.json`；
- `question_input_audit.md`；
- 可复用 candidate 清单；
- 需要重跑 GT/补 candidate 的清单。

### WP2：方案 C 重新生成 question

产出：

- 新 question run；
- 完整 run manifest；
- 生成命令和 seed 记录；
- 题目容量不足报告。

### WP3：Gate 3/4 后置审计与 collection

产出：

- `question_run_audit.json`；
- Gate 3/4 汇总；
- 一个用于人工审阅的 leakage-safe pilot collection；
- validator 输出。

### WP4：人工做题与 Luna 影子审计

产出：

- 用户逐题作答与质量记录；
- Luna blind/post-reveal 审计记录；
- 人工与 Luna 对照表；
- `keep/revise/reject` 建议。

### WP5：阶段结论

产出一份简短结论，至少包含：

- 可复用资产比例；
- Gate 1–4 主要失败原因；
- 新生成并保留的题目数量和覆盖；
- 人工实际作答发现的问题；
- Luna 与人工判断的一致和分歧；
- prompt、profile、candidate pool、显著性规则分别需要修改什么；
- 下一轮应该先修题还是扩 candidate/GT。

## 13. 验收标准

本阶段在以下条件全部满足后完成：

- [ ] 历史 artifact 未被覆盖或回填修改；
- [ ] Gate 1/2 对目标 candidate sets 完成全量 preflight；
- [ ] V1/V2 profile 和 profile_hash 边界明确；
- [ ] 使用现有 candidate/GT 生成了全新的 question run；
- [ ] 每道新题通过 Gate 3 独立复算；
- [ ] 新 run 和 pilot collection 通过 Gate 4；
- [ ] 至少形成一批可由用户实际作答的新题；
- [ ] 用户完成首轮人工质量判断；
- [ ] Luna 完成对应的影子审计，但未被当作自动 Gate；
- [ ] 形成具体的修改清单和下一轮 candidate 缺口；
- [ ] 没有为了达到数量目标降低 deterministic Gate。

## 14. 建议执行顺序

```text
先实现 Gate 1/2 审计工具
→ 审计现有 V1/V2 candidate sets
→ 用通过者重新生成少量 question
→ 实现/运行 Gate 3 run audit
→ 构建并验证 Gate 4 pilot collection
→ 将新题交给用户实际作答
→ 并行调用 Luna 做相同题目的影子审计
→ 汇总人工与 Luna 反馈
→ 决定 prompt/profile/candidate pool 的修改
→ 再进入更大规模题目扩展
```
