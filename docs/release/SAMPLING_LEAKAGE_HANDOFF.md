# Sampling Leakage 工作交接

## 1. 这次解决了什么问题

ArchitectureIQ 的一道题由多个 candidate 作为选项构成。过去生成多道题时，只保证题目的候选集合不完全相同，但不同题目仍可能共享部分 `candidate_id`。

例如：

```text
Q1: c1, c2, c3, c4
Q2: c3, c4, c5, c6
```

如果 Q1 位于 support 阶段，并在作答后公开正确答案或性能指标，那么模型就会提前获得 Q2 中 `c3`、`c4` 的信息。这会污染后续 holdout 结果。

本次实现采用更严格、容易审计的协议：

```text
对于 evaluation collection 中任意两道不同题目 Qi、Qj：

candidate_ids(Qi) ∩ candidate_ids(Qj) = ∅
```

也就是说，一个 `candidate_id` 在整个 collection 中最多出现一次。

在此基础上，collection 可以拆分为：

```text
support:
答题 → 返回反馈 → 记录 lesson

holdout:
读取冻结的 lessons → 答题 → 不返回反馈
```

最终分别报告：

- `blind_score`
- `support_sequential_score`
- `post_feedback_holdout_score`

三类分数不能混成一个 accuracy。

## 2. 当前 Git 状态

本功能建立在 hardening 后的 main 基线上，功能提交为：

```text
4b6a7ee feat: add leakage-safe collection tooling
91441c6 feat: enforce leakage-safe support and holdout protocol
```

最终合并状态：

- 功能已在 feature branch 完成；
- 已通过 `7628956 merge: add leakage-safe evaluation collections` 合入本地 `main`；
- 合并后的完整测试为 `165 passed`，Ruff 为 `All checks passed!`；
- 本地 `main` 当前领先 `origin/main`，尚未推送 GitHub。

两个功能提交仍保留在 merge 历史中，方便分别追踪：

1. collection builder/validator；
2. 核心生成约束和 support/holdout 协议。

## 3. 已确定的协议

### 3.1 Candidate 隔离

两个层面都实施了 candidate 隔离。

第一层是单个 question run：

```text
candidate_reuse_policy = globally_disjoint_within_run
```

核心 question generator 不再只排除完全相同的题目组合，而是使用回溯搜索，从所有满足 significance 条件的候选子集中选择多道 candidate-disjoint questions。

第二层是 evaluation collection：

```text
candidate_reuse_policy = globally_disjoint
```

collection 可以读取一个或多个 question runs。即使不同 run 之间重复使用过 candidate，builder 最终输出的 support 和 holdout 仍不能共享任何 `candidate_id`。

### 3.2 ID split

ID 模式允许 support 和 holdout 使用相同的 `dataset_id`，用于测量相同数据分布上的迁移能力。

约束为：

```text
holdout_dataset_ids ⊆ support_dataset_ids
```

例如：

```text
Support:
dataset d1，candidate c1/c2
dataset d2，candidate c3/c4

Holdout:
dataset d1，candidate c5/c6
dataset d2，candidate c7/c8
```

dataset 可以重复，但 candidate 不能重复。

### 3.3 OOD split

OOD 模式要求 support 与 holdout 的 `dataset_id` 完全不重合：

```text
support_dataset_ids ∩ holdout_dataset_ids = ∅
```

例如：

```text
Support:
dataset d1/d2

Holdout:
dataset d3/d4
```

candidate 全局不重复的约束仍然保留。

这里的 OOD 是基于 `dataset_id` 的协议级 OOD。它不自动保证 family、难度、生成参数或语义距离满足更细粒度的 OOD 定义。

### 3.4 Feedback 协议

Support 阶段：

```text
读取当前题目和已有 lessons
→ 提交预测
→ 返回正确答案和指标反馈
→ 记录 lesson
→ 进入下一题
```

Holdout 阶段：

```text
冻结 support lessons
→ 读取当前题目和 frozen lessons
→ 提交预测
→ 只记录预测，不返回答案或指标
→ 进入下一题
```

一旦开始提交 holdout 答案，就不能再修改 lessons。

## 4. 代码结构

### 核心题目生成

```text
src/architecture_iq/questions/generator.py
src/architecture_iq/questions/runs.py
```

`generator.py`：

- 使用 `_pick_candidate_disjoint_subsets()` 选择问题；
- 采用回溯搜索，而不是简单贪心；
- 保证同一个 question run 内 candidate 不重复；
- 如果找不到请求数量的 candidate-disjoint questions，会明确失败。

`runs.py` 在 `run.json` 中记录：

```json
{
  "candidate_reuse_policy": "globally_disjoint_within_run"
}
```

### Collection builder

```text
tools/build_leakage_safe_collection.py
```

职责：

- 读取一个或多个 question run；
- 加载问题、prompt、dataset 和 candidate provenance；
- 过滤 candidate overlap；
- 按 ID 或 OOD 规则构造 support/holdout；
- 使用固定 seed 保证可复现；
- 输出 collection manifest 以及公开问题文件。

### Collection validator

```text
tools/validate_leakage_safe_collection.py
```

职责：

- 检查 candidate 全局唯一；
- 检查 support/holdout 的 dataset 协议；
- 检查 manifest、源问题和公开问题相互一致；
- 检查公开文件没有答案、真实性能或私有路径；
- 对不完整或不一致的 collection fail closed。

### Feedback session

```text
tools/leakage_safe_feedback_session.py
```

职责：

- 按 support → holdout 顺序推进实验；
- support 答题后返回反馈；
- 记录 support lessons；
- 进入 holdout 时冻结 lessons；
- holdout 答题后不返回反馈；
- 分别汇总三类 score。

## 5. 从题目生成到评测的完整流程

```text
Candidate pool
    ↓
generate-question
    ↓
candidate-disjoint question run
    ↓
build_leakage_safe_collection.py
    ↓
collection.json + support.json + holdout.json
    ↓
validate_leakage_safe_collection.py
    ↓
leakage_safe_feedback_session.py
    ↓
blind / support-sequential / post-feedback-holdout 报告
```

### 第一步：生成题目

继续使用现有 ArchitectureIQ question generation pipeline。示例：

```powershell
architecture-iq generate-question `
  data/datasets/<family>/<dataset_id> `
  --candidate-set data/datasets/<family>/<dataset_id>/candidates/<set_name> `
  --num-questions 8 `
  --num-choices 4 `
  --seed 19
```

具体 candidate set 参数以当前 CLI 帮助和实际数据路径为准。

新的核心生成器会尝试选择 8 道 candidate-disjoint questions。如果候选池不足，会返回类似错误：

```text
Requested 8 candidate-disjoint questions but no valid selection exists...
Generate more candidates or request fewer questions.
```

此时不应降低验证标准或重新使用 candidate。可选处理方式是：

- 生成更多 candidates；
- 引入更多 candidate sets；
- 减少 `--num-questions`；
- 适当调整 significance 条件。

### 第二步：构建 ID collection

```powershell
python tools/build_leakage_safe_collection.py `
  --question-run data/datasets/<family>/<dataset_id>/questions/<run_name> `
  --output artifacts/leakage_collections/id_demo `
  --support-count 4 `
  --holdout-count 4 `
  --split-mode id `
  --seed 19
```

多个 source runs 可以重复指定 `--question-run`：

```powershell
python tools/build_leakage_safe_collection.py `
  --question-run data/datasets/<family>/<dataset_1>/questions/<run_1> `
  --question-run data/datasets/<family>/<dataset_2>/questions/<run_2> `
  --output artifacts/leakage_collections/id_multi_dataset `
  --support-count 8 `
  --holdout-count 8 `
  --split-mode id `
  --seed 19
```

### 第三步：构建 OOD collection

```powershell
python tools/build_leakage_safe_collection.py `
  --question-run data/datasets/<family>/<dataset_1>/questions/<run_1> `
  --question-run data/datasets/<family>/<dataset_2>/questions/<run_2> `
  --output artifacts/leakage_collections/ood_demo `
  --support-count 4 `
  --holdout-count 4 `
  --split-mode ood `
  --seed 19
```

OOD 至少需要两个不同的 `dataset_id`。

### 第四步：验证 collection

```powershell
python tools/validate_leakage_safe_collection.py `
  artifacts/leakage_collections/id_demo
```

成功时会输出类似：

```json
{
  "valid": true,
  "collection_id": "lc_0123456789ab",
  "split_mode": "id",
  "support_questions": 4,
  "holdout_questions": 4,
  "unique_candidates": 32,
  "support_dataset_ids": ["dataset_a"],
  "holdout_dataset_ids": ["dataset_a"]
}
```

正式实验前必须执行 validator。不要因为 collection 是由 builder 生成的就跳过验证。

## 6. Collection 产物

输出目录结构：

```text
<collection_dir>/
  collection.json
  support.json
  holdout.json
```

### 6.1 `collection.json`

这是私有审计 manifest，不应发送给模型或提供给可自由读取文件的 agent。

主要字段：

```json
{
  "schema_version": "leakage_safe_collection_v1",
  "collection_id": "lc_0123456789ab",
  "seed": 19,
  "split_mode": "id",
  "candidate_reuse_policy": "globally_disjoint",
  "dataset_policy": "holdout_datasets_seen_in_support",
  "source_runs": ["C:/.../questions/run_..."],
  "support_question_ids": ["q_001", "q_002"],
  "holdout_question_ids": ["q_003", "q_004"],
  "records": [
    {
      "question_id": "q_001",
      "family": "univariate_regression",
      "dataset_id": "sym_...",
      "candidate_ids": ["c_1", "c_2", "c_3", "c_4"],
      "source_question_dir": "C:/.../questions/run_.../q_001",
      "split": "support"
    }
  ]
}
```

其中包含 source 路径、candidate provenance 和 split 分配，用于审计和验证，不属于模型上下文。

### 6.2 `support.json` 和 `holdout.json`

这两个文件是可以发送给 API 的公开视图，格式相同：

```json
[
  {
    "question_id": "q_001",
    "family": "univariate_regression",
    "dataset_id": "sym_...",
    "prompt": "完整问题 prompt",
    "choices": [
      {"letter": "A", "candidate_id": "c_1"},
      {"letter": "B", "candidate_id": "c_2"}
    ]
  }
]
```

公开文件不包含：

- `correct_letter`
- `correct_candidate_id`
- `choice_mean_metrics`
- `ground_truth`
- `significance`
- `evaluation`
- `candidate_path`
- `candidate_set_path`
- `source_question_dir`
- `results/summary.json` 路径

## 7. Private feedback 文件

Feedback session 还需要 evaluator 私下持有的 feedback JSON。支持以下两种顶层格式：

```json
[
  {
    "question_id": "q_001",
    "correct_letter": "B",
    "metric": "test_mse",
    "choice_mean_metrics": {
      "A": 0.42,
      "B": 0.31
    }
  }
]
```

或者：

```json
{
  "questions": [
    {
      "question_id": "q_001",
      "correct_letter": "B",
      "metric": "test_mse",
      "choice_mean_metrics": {
        "A": 0.42,
        "B": 0.31
      }
    }
  ]
}
```

feedback 文件必须覆盖 support 和 holdout 的全部 `question_id`。虽然 holdout 阶段不会向模型返回反馈，但 evaluator 在最终 summary 时仍需要私有答案进行评分。

## 8. 运行 support/holdout session

### 8.1 初始化

```powershell
python tools/leakage_safe_feedback_session.py init `
  --session artifacts/leakage_sessions/demo_session.json `
  --collection artifacts/leakage_collections/id_demo `
  --feedback artifacts/private/id_demo_feedback.json `
  --experiment "id-demo"
```

如果 session 已存在，工具默认拒绝覆盖。只有明确重新开始时才使用 `--force`。初始化时会验证 feedback 是否覆盖全部题目。

### 8.2 获取当前题目

```powershell
python tools/leakage_safe_feedback_session.py current `
  --session artifacts/leakage_sessions/demo_session.json
```

返回当前 `phase`、公开问题和最多 12 条已有 lesson。Holdout 阶段使用冻结后的 lessons。

### 8.3 提交 support 答案

```powershell
python tools/leakage_safe_feedback_session.py answer `
  --session artifacts/leakage_sessions/demo_session.json `
  --letter B `
  --confidence 0.72 `
  --reason "候选 B 的结构更符合当前数据规模"
```

Support 阶段先记录预测，再返回：

```json
{
  "phase": "support",
  "recorded_prediction": {
    "question_id": "q_001",
    "predicted_letter": "B",
    "predicted_candidate_id": "c_2",
    "confidence": 0.72,
    "reason": "..."
  },
  "feedback": {
    "correct_letter": "B",
    "is_correct": true,
    "metric": "test_mse",
    "choice_mean_metrics": {"A": 0.42, "B": 0.31}
  }
}
```

### 8.4 记录 lesson

```powershell
python tools/leakage_safe_feedback_session.py lesson `
  --session artifacts/leakage_sessions/demo_session.json `
  --text "在该分布上，更深的模型没有稳定抵消优化难度。"
```

建议每道 support 题都完成：

```text
current → answer → lesson
```

### 8.5 提交 holdout 答案

命令与 support 相同：

```powershell
python tools/leakage_safe_feedback_session.py answer `
  --session artifacts/leakage_sessions/demo_session.json `
  --letter A `
  --confidence 0.64 `
  --reason "应用 support 阶段总结的规律"
```

Holdout 返回中没有 `feedback`、`correct_letter`、`is_correct`、`metric` 或 `choice_mean_metrics`：

```json
{
  "phase": "holdout",
  "recorded_prediction": {
    "question_id": "q_010",
    "predicted_letter": "A",
    "predicted_candidate_id": "c_40",
    "confidence": 0.64,
    "reason": "..."
  }
}
```

提交第一道 holdout 答案后，再调用 `lesson` 会被拒绝。

### 8.6 生成 summary

```powershell
python tools/leakage_safe_feedback_session.py summary `
  --session artifacts/leakage_sessions/demo_session.json `
  --blind-score 0.375
```

输出结构：

```json
{
  "experiment_name": "id-demo",
  "protocol": {
    "support_feedback": true,
    "holdout_feedback": false,
    "lessons_frozen_before_holdout": true
  },
  "blind_score": 0.375,
  "support_sequential_score": {
    "correct": 3,
    "total": 4,
    "accuracy": 0.75
  },
  "post_feedback_holdout_score": {
    "correct": 2,
    "total": 4,
    "accuracy": 0.5
  },
  "frozen_lessons": ["..."],
  "complete": true
}
```

`blind_score` 由外部 blind evaluation 提供；当前 session 工具不会自动运行 blind baseline。

## 9. API 与 Agent 的隔离边界

### 9.1 普通 API 调用

API 模型只能接收：

- `current` 返回的 question；
- 允许公开的 choices 和 metadata；
- support 阶段返回的 feedback；
- 已记录的 lessons。

API 不应接收：

- `collection.json`
- private feedback 文件
- session 文件
- 原始 `question.json`
- `results/summary.json`
- candidate results 目录
- evaluator 的文件路径

推荐结构：

```text
Evaluator process
├── private collection manifest
├── private feedback
├── private session state
└── safe current-question/result projection
        ↓
      API
```

### 9.2 可读取文件系统的 Agent

如果 agent 能自由读取仓库，仅仅“不把答案放进 prompt”并不足够。

以下文件都必须位于 agent 不可访问的位置：

- `collection.json`
- private feedback
- session JSON
- 原始带答案的 question artifacts
- candidate `results/`
- 任何可以定位上述文件的 manifest

可为 agent 建立只包含公开资料的隔离 workspace，限制文件权限，或由 evaluator 代理全部 session 操作。当前代码实现了信息边界和协议状态机，但不会自动配置操作系统级 sandbox。

## 10. Validator 检查范围

Validator 当前会检查：

- schema version 正确；
- manifest 声明 `candidate_reuse_policy=globally_disjoint`；
- records 非空；
- question ID 不重复；
- 每道题内部 candidate 不重复；
- 整个 collection 中 candidate 不重复；
- source question 和 prompt 存在；
- manifest 的 question ID、dataset ID、candidate IDs 与 source 一致；
- public question 顺序与 manifest 一致；
- public dataset ID、candidate IDs 和 prompt 与 source 一致；
- public JSON 不含已知私有字段；
- public prompt 不含已知私有 marker；
- ID 模式下所有 holdout datasets 均在 support 中出现；
- OOD 模式下 support/holdout datasets 不重叠。

任何检查失败，CLI 都以非零状态退出。

## 11. 测试与验证结果

功能完成后已验证：

```text
Focused tests: 28 passed
Full pytest:    165 passed in 32.86s
Ruff:           All checks passed
```

建议合并或后续修改后复跑：

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\tools\question_inspector"
$python = if ($env:ARCHITECTUREIQ_PYTHON) { $env:ARCHITECTUREIQ_PYTHON } else { "python" }

& $python `
  -m pytest `
  tests/test_leakage_safe_collection.py `
  tests/test_leakage_safe_feedback_session.py `
  tests/test_question_generation.py `
  -q `
  --basetemp .pytest-tmp-leakage
```

完整测试：

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\tools\question_inspector"
$python = if ($env:ARCHITECTUREIQ_PYTHON) { $env:ARCHITECTUREIQ_PYTHON } else { "python" }

& $python `
  -m pytest `
  -q `
  --basetemp .pytest-tmp-leakage-full
```

Ruff：

```powershell
$python = if ($env:ARCHITECTUREIQ_PYTHON) { $env:ARCHITECTUREIQ_PYTHON } else { "python" }

& $python `
  -m ruff check .
```

Windows 多 worktree 环境下应显式设置 `PYTHONPATH`，否则 pytest 可能导入另一个 checkout。

## 12. 设计上的重要细节

### 为什么使用回溯而不是贪心

简单贪心可能因为先选择了一个局部可用组合，而错过后续完整的合法组合。因此核心 question generator 使用回溯搜索：

```text
选择一个 subset
→ 标记其中 candidate IDs
→ 搜索剩余不重合 subset
→ 如果无法完成目标数量，回退并尝试其他选择
```

### 为什么 collection 层仍需再次检查

核心生成器只保证同一个 question run 内 candidate-disjoint。collection 支持组合多个 source runs，而不同 run 之间仍可能复用 candidate，因此 builder 和 validator 仍必须执行全局检查。

### 为什么保留 validator

Builder 负责构建正确产物；validator 负责确认 builder 没有回归、文件没有被后续脚本改坏、source artifacts 没有漂移、public projection 没有意外加入答案或路径。正式实验应把 validator 作为前置门禁。

## 13. 当前限制

### 13.1 以 `candidate_id` 为第一版身份边界

当前规则依赖一个实际 candidate 对应一个稳定、唯一的 `candidate_id`。当前版本不会额外计算 `candidate_spec_hash`，因此不能自动识别 ID 不同但 candidate spec 或底层训练结果完全相同的情况。

如果未来存在 candidate 复制、重新编号或跨数据导入，应增加 canonical spec hash 或 source result identity。

### 13.2 OOD 只按 dataset ID 定义

当前 OOD 表示 support/holdout 的 `dataset_id` 不重叠，不是完整的语义 OOD 定义。后续可以继续细分 dataset family、difficulty、model family、budget 或联合 OOD，但不应静默改变当前 `leakage_safe_collection_v1`。

### 13.3 Builder 不生成 candidates

`build_leakage_safe_collection.py` 只从既有 question runs 中构造 collection。候选数量不足时，需要回到正常 pipeline：

```text
create dataset
→ generate candidates
→ run GT
→ generate questions
→ build collection
```

不能通过重复使用 candidate 来满足题目数量。

### 13.4 Private 文件隔离由运行环境负责

代码可以检查 public JSON 是否包含已知私有字段，但不能阻止具有完整磁盘读取权限的 agent 主动打开其他目录。正式 agent benchmark 必须同时配置文件系统隔离。

### 13.5 Prompt marker 检查不是通用秘密扫描器

Validator 会阻止已知字段和路径 marker，但它不是自然语言级别的泄露检测模型。Ground truth 仍必须遵守仓库已有规范：prompt 不展示最终 metrics 或 curves，prompt 代码来自实际执行的生成代码，GT 来自真实执行路径。

### 13.6 Session 不直接调用远端 API

`leakage_safe_feedback_session.py` 是可审计协议状态机，不是远端 API runner。当前推荐由外层 evaluator 执行：

```text
current
→ 构造 API request
→ 调用模型
→ 解析 letter
→ answer
→ support 时生成 lesson
```

这样 private feedback 始终由 evaluator 持有。

## 14. 后续开发建议

### 第一阶段：真实数据 smoke 并冻结当前协议

- 生成至少一个真实 ID collection；
- 使用 validator 验证；
- 用一个模型完成 support/holdout smoke；
- 保存 collection ID、seed、模型配置和 summary。

### 第二阶段：扩展题目数量

新增题目时必须从候选池规模倒推容量。例如 20 道题、每题 4 个 choices，至少需要 80 个互不重复 candidate，实际还需要给 significance 和 compatibility 过滤留出余量。

建议记录：

- 每个 dataset 的 candidate 总量；
- significance 通过率；
- 最多可生成的 candidate-disjoint questions；
- ID/OOD 可用容量。

### 第三阶段：正式 OOD 设计

在题目数量足够后，再明确 OOD 轴：dataset instance、dataset family、model family、budget 或 combined OOD。每种 OOD 应单独命名并单独报告。

### 第四阶段：远端 API 自动化

可以在当前 session 状态机外增加 wrapper，支持 API 调用、raw response、模型配置、resume、answer parsing、prompt hash 和失败重试。wrapper 不应复制 feedback 和 split 逻辑，而应调用当前 session 工具。

## 15. Follow 路径

建议按以下顺序阅读：

1. `SAMPLING_LEAKAGE_HANDOFF.md`
2. `src/architecture_iq/questions/generator.py`
   - `_pick_candidate_disjoint_subsets`
3. `src/architecture_iq/questions/runs.py`
   - `candidate_reuse_policy`
4. `tools/build_leakage_safe_collection.py`
   - `build_collection`
   - `_split_id`
   - `_split_ood`
5. `tools/validate_leakage_safe_collection.py`
   - `validate_collection`
6. `tools/leakage_safe_feedback_session.py`
   - `init_session`
   - `current_question`
   - `submit_answer`
   - `record_lesson`
   - `build_summary`
7. `tests/test_question_generation.py`
8. `tests/test_leakage_safe_collection.py`
9. `tests/test_leakage_safe_feedback_session.py`

如果只想确认协议是否被正确实现，重点看三处：

```text
generator:
同一 run 内 candidate 不复用

validator:
整个 collection 内 candidate 不复用

feedback session:
support 返回反馈，holdout 不返回反馈
```

## 16. 合并检查清单

- [x] 目标 main 工作树干净；
- [x] `4b6a7ee` 和 `91441c6` 已进入 main 历史；
- [x] 没有把旧 historical sequential runner 的删除混入本次合并；
- [x] focused tests 通过；
- [x] full pytest 通过；
- [x] Ruff 通过；
- [ ] 生成一个真实 ID collection；
- [ ] validator 返回 `valid: true`；
- [x] support 答案会返回 feedback；
- [x] holdout 答案不会返回 feedback；
- [x] holdout 开始后不能继续修改 lesson；
- [ ] `collection.json`、private feedback 和 session 文件不会暴露给 agent；
- [x] 最终 main merge commit 已记录为 `7628956`；
- [ ] 再从新 main 同步后续题目扩展和 KAN 分支。

本次交付已经完成并合入采样泄露工作的第一版协议和工具链。下一步应先完成真实 collection 与模型 smoke，再在这个基线上扩展题目数量，之后推进 KAN 问题包和更细粒度的 OOD 测试。
