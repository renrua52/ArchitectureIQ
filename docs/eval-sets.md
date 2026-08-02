# 评测端设计 · Eval Sets（2026-08-01 起）

> 本文档描述**评测端（`backend/eval/`）**的题集格式、生成器、评分闭环与探针结果。
> 存储后端见 [`backend-storage.md`](./backend-storage.md)；评测协议汇总见 [`../PROTOCOLS.md`](../PROTOCOLS.md)。
> 设计原则：题目实例（problem+candidates）与评测实例（题集）解耦；评测端只读 `backend/data/`，
> 组合 candidates 出题；GT 全部来自已执行结果（`backend/data/results/`），不重新计算。

---

## 1. 题集格式（JSONL）

题集位于 `backend/eval/sets/{set_name}/`：

```
backend/eval/sets/{set_name}/
├── questions.jsonl     # 每行一个题目 JSON（评测实例）
└── manifest.json       # 生成参数、过滤条件、跳过的 problem
```

每行 JSON 的公共字段：

| 字段 | 说明 |
|------|------|
| `schema_version` | 题集 schema 版本（当前 `0.1`） |
| `type` | `select_best` / `propose_improvement` / `two_choice_loss_compare` |
| `question_id` | 内容寻址 id（`sb_*` / `pi_*`），由题目内容 hash 生成 |
| `problem_id` | 引用的题目实例（`backend/data/problems/{problem_id}/`） |
| `metric` | 选择指标（`test_mse` / `test_ce`，lower is better） |
| `references` | 5 个参考 setting + 实测 loss（few-shot 校准用） |
| `prompt` | 渲染好的完整 prompt 文本 |

题目只存 **candidate_id 引用**，不存路径——候选配置、GT 统一经
`src/architecture_iq/storage/repository.py` 按 ID 读取，与后端布局解耦。

---

## 2. 三种题型

### 2.1 `select_best`（选择题，主任务）

- **选项**：base setting + 5 个离 base 最近的候选（**base 一定是选项之一**，问"哪个修改后 loss 最低，含不改"），共 6 选 1；选项**不带 loss**。
- **参考**：5 个带 loss 的 setting（v1.1 起锚定选项 loss 区间，避免外推）。
- **过滤**：winner vs runner-up 跨 seed `win_rate >= 0.7`；ratio 按 metric 分桶（MSE ≥ 1.15，CE ≥ 1.03）。
- **v1.2 改进**：选项两两之间在**显著字段**（模型类型/深度/宽度/残差/优化器/lr/weight_decay/loss/batch_size）上的差异 ≥ 2，杜绝"看起来一样"的 ill options；每个选项相对 base 只改 1–2 个显著字段（可比性强）。

### 2.2 `propose_improvement`（config 修改题）

- **输入**：5 个随机参考（带 loss）+ base（带 loss）+ 5 个改进 demo（带 loss，离 base 最近的候选）。
- **输出**：模型给出一个新的 JSON config（闭集内）。
- **评分**：`backend/eval/score_proposal.py` 校验闭集 → 吸附到网格 → `write_candidate` + `run_ground_truth`（新跑 GT）→ 与 base 逐 seed 对比（涨/跌/平）。
- **约束**：模型参数量 ≤ demos 中最大参数量 × 1.1（预算固定为 base 的 `total_samples_seen`）。

### 2.3 `two_choice_loss_compare`（二选一，诊断/对照组）

- 3 个参考（近 min / median / max）+ 目标对 A/B，问"哪个 loss 更高/更低"。
- 过滤：ratio ∈ [1.2, 5]，win_rate ≥ 0.8，选项通过 `choices_have_contrast`。
- 用途：验证模型是否能做参考校准的 loss 比较；信号强于 6 选 1（见 §5 探针结果）。

---

## 3. 生成器用法

```bash
# select_best（v2，当前推荐；v1/v1.1/v1.2 因答案键 bug 已删除）
.venv/bin/python -m backend.eval.questions --type select_best \
    --items-per-problem 5 --set-name select_best_v2 --seed 20260804

# propose_improvement
.venv/bin/python -m backend.eval.questions --type propose_improvement \
    --items-per-problem 5 --set-name propose_improvement_v1

# two_choice（诊断）
.venv/bin/python -m backend.eval.two_choice --items-per-problem 3

# 评分 LLM 提出的 config（propose_improvement 闭环）
.venv/bin/python -m backend.eval.score_proposal \
    --question backend/eval/sets/propose_improvement_v1/questions.jsonl:0 \
    --proposal artifacts/eval_probe/propose_improvement_v1/pi_736c27_proposal.json \
    --out artifacts/eval_probe/propose_improvement_v1/pi_736c27_score.json

# 探针抽样（分层 tight/medium/loose，批次内 problem 不重复）
.venv/bin/python -m backend.eval.probe --set select_best_v2 --num-batches 2 --batch-size 6 --seed 20260805
# 探针评分（subagent 把答案写到 batches/batch_{i}_answers.jsonl 后）
.venv/bin/python -m backend.eval.probe --set select_best_v2 --score
```

---

## 4. 评分闭环（propose_improvement）

`score_proposal.py` 的流程严格保持 spec → code → run → GT 不变量：

```
base config + proposal overrides
  -> normalize（合并到 base；lr/weight_decay/batch_size/架构维度吸附到 v2 profile 闭集网格）
  -> 校验（model/optimizer 类型、loss 与 family 兼容、参数量 <= 1.1x demos 上限）
  -> candidate_spec（schema 2.0，content-addressed candidate_id）
  -> write_candidate(temp) -> run_ground_truth(temp, profile v2, problem_dir)
  -> 逐 seed 与 base 对比：proposal_loss / ratio / win_rate
```

闭集（`profiles/v2.yaml`）：`optimizer ∈ {SGD, Adam, AdamW, RMSprop, Adagrad}`，
`lr ∈ {1e-4..1e-1}`、`weight_decay ∈ {0..1e-2}`、`batch_size ∈ {16, 32, 64}`，
`model ∈ {mlp, transformer_lm}`（mlp depth 1–6 / width 16–256；transformer d_model 32–128 等）。

---

## 5. 探针结果（subagent 做题，2026-08-01）

> **作废声明**：本节所有 `select_best_v1/v1.1/v1.2` 的分数均基于坏答案键（见 §8），全部作废并已清理；机制性发现（DeepSeek 位置偏置、短答案解码塌缩、few-shot 影响）仍有效。有效分数以 §9–§12 为准。

| 题集 | 题目数 | 正确率 | 随机基线 | 备注 |
|------|--------|--------|----------|------|
| `two_choice_loss_compare`（诊断） | 8 | 7/8（87.5%） | 50% | 二选一 + 参考校准对 LLM 可解 |
| `propose_improvement_v1`（config 修改） | 2 次出题 | 2/2 击败 base | — | ratio 1.14x / 1.20x，win_rate 1.0 |

**关键发现**：

1. **任务格式决定可解性**（v2 修键后重评）：六选一 claude 70% / terra 72%（§8.2，随机 16.7%）、
   二选一 74–87.5%；选择题仍存在天花板效应（§10），主指标建议转 propose_improvement。
2. **propose_improvement 信号强**：subagent 依据参考（RMSprop/宽网络等）提出的 config 两次都真实打败 base
   （GT 新跑验证，非猜测）。这是最有生产意义的方向（贴近 autoresearch：提出修改 → 实测涨跌）。
3. **CE 轴压缩**：bigram CE 整体 max/min 仅 ~1.14x，单点修改区分度低；LLM 提修改时应避开 loss 轴
   （与分析脚本 `tools/analysis/analyze_loss_disparity.py` 结论一致）。
4. **ill settings 存在**：v1.1 有 21 题存在两选项只在非显著字段（activations/layer_norm）不同，
   视觉上像重复选项；v1.2 已用显著字段过滤（过滤后仍有 26/59 题某选项在池中无"单轴邻居"参考）。
5. **参考必须覆盖判别轴**：错题 `sb_fedea4`（ratio 3.4）中 GT 胜者是 2x32+残差，而参考里最接近的是
   1x32（loss 0.65，误导向"宽网络赢"）。六选一的参考若不能覆盖选项间的判别轴，就会系统性误导；
   池中"恰好差一个显著字段"的邻居平均每题 2–4 个、但 26/59 题存在某选项无邻居，按轴构造参考不可靠。

---

## 5.1 大规模批量评测（50 题/setting，2026-08-01 第二轮）

批量并发评测脚本：`backend/eval/batch_eval.py`（选择题）与 `backend/eval/batch_propose.py`
（config 修改题）；默认 provider 为**本地 DeepSeek key**（`~/.codex-deepseek/.deepseek_api_key`
→ `api.deepseek.com`，模型 `deepseek-chat` = deepseek-v4-flash），默认 50 并发，
支持 `--sets a,b` 混合并发。phybench 中转（`openai.phybench.cn`，gpt-5.6-terra）不稳定，
仅在其中转恢复窗口内补跑对照。

### DeepSeek-V4-Flash（50 题/setting）

| 题集 | 题量 | 正确 / 击败 | 随机基线 | 结论 |
|------|------|-------------|----------|------|
| `two_choice` seed1（range demo） | 50 | 27/50（54%） | 50% | 二选一 ≈ 随机 |
| `two_choice` seed2（range demo） | 50 | 26/50（52%） | 50% | 二选一 ≈ 随机 |
| `two_choice` config_near demo | 50 | 21/50（42%） | 50% | 配置邻居 demo 是负优化 |
| `propose_improvement_v1.1` | 50 | **48/50 击败 base（96%）** | — | 中位 ratio 1.33，win_rate≥0.7 有 45 题 |
| `propose_improvement_v1` | 50 | **43/49 击败 base（88%）** | — | 中位 ratio 1.32，win_rate≥0.7 有 39 题 |

### gpt-5.6-terra（phybench 中转恢复窗口，50 题/setting）

| 题集 | 题量 | 正确率 | 随机基线 |
|------|------|--------|----------|
| `two_choice` seed1（range demo） | 50 | 38/50（76%） | 50% |
| `two_choice` config_near demo | 50 | 31/50（62%） | 50% |

**50 题样本下的结论（替代小样本探针）**：

1. **select_best 六选一（v1.x 坏键版本）分数作废**：修键后 v2 为 claude 70% / terra 72%（§8.2）。
   选择题的主要问题是天花板效应而非地板（见 §10）。
2. **二选一区分两个模型**：gpt-5.6-terra 76% 显著高于 deepseek 52–54%（≈随机）。
   → 二选一可以作为**模型能力的有效度量**；六选一作为上限测试。
3. **config_near demo 在两个模型上都是负优化**（gpt 76%→62%，deepseek 54%→42%），
   保留 range（min/median/max）demo 策略。
4. **propose_improvement 是强信号任务**：deepseek 两组 50 题分别 96% / 88% 击败 base
   （GT 全部新跑），中位 ratio ~1.32；top 案例 ratio 数百倍（如 base 0.208 → 0.0003）。
   即使一个只会"抄参考中最优 config"的模型也能赢，说明任务对 LLM 友好、可度量。
5. **工程结论**：50 并发 + 多 setting 并行（`--sets`）稳定跑通（3 setting × 50 题 ≈ 10s API 时间）；
   DeepSeek 官方 API 无限流问题；phybench 中转不可靠（会整批空响应）。

## 5.2 为什么性能看起来这么差 —— DeepSeek vs GPT 回答过程根因（2026-08-01）

对 `select_best_v1.2`（各 50 题）与 `two_choice`（各 50 题）的 `raw_response` 逐条复核，
得到三类原因：**一个评分 bug、一个模型位置偏置、一个题面欠定问题**。前两者是"看起来更差"的
放大器，第三者是真实地板。

### 5.2.1 评分 bug：加粗答案解析失败（已修复）

`batch_eval.parse_answer` 只能识别 `Answer: B`，识别不了 `Answer: **B**`（DeepSeek 常用加粗）。
旧逻辑随后退化成"取全文中**第一个**独立字母"——而推理开头列选项必然先出现 `A`，于是系统性记成 A。
后果：DeepSeek select_best_v1.2 原始 2/50，修复后 **6/50（12%）**；同时造成"A 答案占 58%"的假象
（50 个 A 里约一半是解析产物，修复后真实 A 占比 17/50=34%）。GPT 只输出单个字母，不受影响。
已修复 `parse_answer` 并加测试 `tests/test_batch_eval_parse.py`（`Answer: **B**` → B、末行独立字母、
"answer is X"、末次关键词优先）。

### 5.2.2 DeepSeek 位置偏置：二选一永远答 A（模型行为，需出题端防御）

- `two_choice` 两批 50 题：DeepSeek 答 A **49/50 和 50/50**；控制实验把 A/B 顺序交换重测 20 题，
  依然 **20/20 答 A** —— 完全不看题目内容。
- 所以 DeepSeek 二选一的 54%/52% **不是"略高于随机"**，而是"永远选第一个选项"的伪分
  （≈ P(正确答案=A)：range 题 56%、confignear 题 42%）。
- 六选一里也有残余首选项偏置（修复解析后 A 仍占 16/45 ≈ 36%，均匀应为 17%）。
- 反观 GPT 二选一 76% 是**真实能力**：字母均衡（A:26 / B:24），按变化轴 67–100%，
  无单轴捷径（大模型=赢的先验不成立，参数量与答案无相关性）。

### 5.2.3 DeepSeek 真实推理 = "锚定最优参考"启发式

select_best 长回答显示其核心策略是**"找与最优参考配置最相似的选项"**
（例 `sb_a5b412`："best reference 是 Ref2 (RMSprop, d_model=128, 1层)… 看选项里谁最像它"）。
统计：修复解析后 12/45 的答案恰为"离最优参考最近的选项"，但该启发式命中正确答案只有 **4/50**；
唯一答对的 loose 题（`sb_b02947`，ratio 102）也是"F 用 Adam 最像最优参考"猜中，属外推恰好成立。
它从不检验"参考间的 loss 差异来自哪个轴"。

### 5.2.4 GPT 回答过程：无推理单字母，边际校准

- 50 题全部是单字母输出（temp=0，无可见推理/校准过程），9/50 = 18% ≈ 随机 16.7%。
- 答案边际分布近似正确答案边际（A 少答、F/D 多答）→ 只学到选项的边缘频率，没有判别信号。
- 按 ratio 分层：tight（<1.15）0/5、medium 3/27（11%）、loose（≥2）6/18（33%）——只有 loose 有真实信号。

### 5.2.5 根本原因：题面欠定，参考无法支撑答案

1. **参考 kNN 预测器只有 5/59（8%）**：用 5 个参考的 loss 按 config 距离加权外推 6 个选项的 loss、
   取预测最小者，正确率低于随机 —— 即"按题面信息不可答"。
2. **判别轴覆盖率 11%**：winner vs runner-up 的判别轴共 134 个，其中只有 15 个存在"单轴邻居"参考
   （与 winner 仅差该轴、且该轴取值不同）；残差 0/16、weight_decay 0/13、momentum 0/7 完全无覆盖；
   59 题里仅 **2 题**能靠参考完全校准。
3. **参考会系统性误导**：错题 `sb_fedea4`（ratio 3.4）中参考最优指向"深+宽"（loss 0.051/0.058），
   而真胜者是 depth=2/width=32+残差 —— 照着参考选必然错。
4. **信号被参考噪声淹没**：参考间 loss 波动 15–30%，而 winner vs runner-up 的差距常只有 5–15%
   （tight 题 <1.15）；prompt 又明示"不要依赖先验、用参考校准"，在参考无信息的题上正好把模型推向错误方向。
5. **对比 propose_improvement（96% 击败 base）**：propose 的 5 个改进 demo **带 loss** 且就在 base 邻域，
   DeepSeek 直接抄最优 demo 附近即可（策略与"锚定最优参考"相同，但此时答案就在参考邻域，且门槛是"胜过 base"而非 6 选 1）。
   同一启发式在 select_best 上失效，因为选项不在参考邻域 —— 证明不是模型不行，是 select_best 题面欠定。

### 5.2.6 建议

- **two_choice**：出题端 per-item 随机化 A/B 顺序（或要求 JSON 输出），并在评测时加"交换顺序一致性"
  作为模型健康度检查；否则 DeepSeek 类模型的首选项偏置会让二选一变成伪分。
- **select_best 参考改造**：参考必须覆盖选项的判别轴 —— 用"base + 1–2 点修改并测 loss"的 few-shot
  参考替代随机参考（即用户 07-31 计划的 propose 范式）；在覆盖之前，select_best 分数≈随机的结论
  不能解读为"模型不会做实验推断"。
- **报告规范**：分数按 ratio 分层并附随机基线；tight（<1.15）题建议只作诊断不作主指标。
- **评分端**：保留修复后的 `parse_answer`（已加测试），后续换 provider 时重跑一遍 50 题基线。


## 5.3 追问核查：题目有没有传进去？为什么"永远答 A"？few-shot 是否太 random？（2026-08-01 第三轮）

针对上一轮结论"DeepSeek 二选一永远答 A"，逐项复核并做了控制实验（新增 `demo_strategy="local"`）。

### 5.3.1 题目内容确实传入了（排除渲染 bug）

检查 `artifacts/eval_probe/prompts/*.txt` 与 `backend/eval/two_choice.py::render_prompt`：3 个参考
（带 loss）+ A/B 两个完整配置（模型/优化器/预算全字段）+ 问题行均正常渲染；batch 与本地渲染走同一
代码路径。**不是题目没传进去。**

### 5.3.2 "永远答 A" = 短答案格式的解码塌缩（非内容、非 few-shot 问题）

控制实验（同一批题，DeepSeek temp=0）：

| 条件 | 结果 |
|------|------|
| `Answer with the letter only.`（不要求推理） | **50/50 答 A**（A/B 交换后 20/20 仍答 A） |
| 追加"先逐步推理再作答" | 同批 50 题 3 轮：**74% / 68% / 76% / 76%**（均值 ~73.5%） |
| 同一 prompt 重复两轮（推理版） | 11/50 题答案翻转 → 推理输出在 temp=0 下仍不稳定 |

结论：模型被要求只输出字母时走了"直接选第一个选项"的捷径，**根本没有处理题目内容**；强制推理后
真实水平恢复到 ~74%。这是格式/解码层面的问题，不是评测题目本身的问题。修复方向：评测端要求模型
先推理再作答，或改用 JSON 输出结构。

### 5.3.3 few-shot 太 random 的影响：是次要因素，不是塌缩原因

- 原 range demo（min/median/max）离目标对**中位 6 次编辑**，1-NN(编辑距离) 预测器只有 51% ≈ 随机
  → 全局分位点 demo 确实不提供本地判别信息。
- 但**换成 local demo 后，不要求推理时依然 26/50 全答 A** → 塌缩与 few-shot 内容无关。
- 强制推理时（同 50 题配对、3 轮平均）：local demo 73.5% vs range demo 75.3% —— **无显著差异**；
  1-NN oracle 显示 local 携带更多信号（56% vs 42%），但模型实际走的是全局模式匹配，demo 选择影响不大。

### 5.3.4 用户提议的"主 candidate + 大改 + 4 扰动 = 5 demos"：已实现，结论如下

`two_choice.py` 新增 `--demo-strategy local`：
- demo = 离目标对最近的锚点候选 + 4 个最近邻（共 5 个，带实测 loss），并要求 A/B 都落在
  star 的 `LOCAL_TARGET_MAX_DIST=4` 次编辑内（保证题目可本地校准）。
- **候选池太稀疏**：严格"同锚 1 次大改 + 4 次 ≤2 编辑扰动"几乎不存在（池中候选最近邻居中位
  2–4 次编辑，84/270 对通过放宽后的 local 策略）。要做严格 star，需要现造 5 个候选并并行跑 GT
  （propose 式闭环，`write_candidate` + `run_ground_truth`），池子里的配置密度不够。
- **真正的提升来自题目对的可比性过滤**：旧 range 题两目标"共锚 ≤4 次编辑"只有 14/57，
  新 local 题 59/84；过滤后即使换回 range demo 也升到 76%（旧题上 range demo+推理仅 65%）。
  → 出题端应优先保证"目标对在配置空间里可比"（同一锚点邻域内），demo 是 star 还是分位点影响其次。


## 5.4 跨模型对照：gpt-5.6-terra / Kimi-K3 via gpt.ge 中转（2026-08-01 第四轮）

harness 按 AGENTS.md §10 改造：凭证/模型名从 `~/.agents/relay.json` 读取（`eval` key），
不设 token 上限，默认 `reasoning_effort=high`，并兼容 `reasoning_content`（Kimi/qwen 会把
答案放 content 或 reasoning_content，视推理模式而定；空 content 自动重试）。同批
`artifacts/eval_probe_local/items` 前 50 题 two_choice + `select_best_v1.2` 前 50 题：

| 模型 | two_choice local（标准） | two_choice local（强制推理） | 备注 |
|------|------|------|------|
| gpt-5.6-terra（gpt.ge） | **33/50 = 66%** | **33/50 = 66%** | 快（50 题 ~15s）、无塌缩、答案均衡 |
| Kimi-K3（gpt.ge） | —（慢/超时） | 32/50 = 64%（重试 1 轮后，2 题仍空） | 慢（50 题 13–15 min），长 prompt 大量空响应 |
| DeepSeek v4-flash（官方 API） | 26/50 = 52%（全 A 塌缩） | **~74%**（3 轮均值 73.5%） | 短答案格式塌缩，需强制推理 |
| gpt-5.6-terra（phybench，旧 range 题） | 76%（旧题集，不可直接比） | — | 中转恢复窗口 |
| claude-opus-5（gpt.ge，20 题探针） | — | 14/20 = 70%（子集，n 小） | 快（20 题 ~12s）、均衡、无塌缩 |

**结论**：

1. **两个"更强"模型在 two_choice 上并不更强**：DeepSeek+强制推理（74%）> terra（66%）≈
   Kimi（64%，重试后）；select_best 的 v1.x 分数作废，修键后见 §8.2（70%+）。
2. **terra 走 gpt.ge 工程上最顺**：快、稳定、无首选项塌缩、两种模式分数一致（66%）。它不做可见推理
   （`reasoning_content` 只有标题，被中转截断），只能看到最终字母，回答过程不透明。
3. **Kimi-K3 走 gpt.ge 目前不可靠**：单次 40–60s+，50 题并发 10 要 13–15 分钟；`select_best`（长
   prompt）空响应率 42%（重试后仍有 21/50 空）；two_choice 空响应率 12%。短 prompt 尚可，长 prompt
   建议换模型或等中转稳定。
4. **推理强度参数**：`reasoning_effort=high` 被中转接受（两模型 200）；对 terra 两种模式分数无差异，
   对 Kimi 是必须项（不带该参数更容易超时）。
5. **claude-opus-5 首跑探针（AGENTS.md §10 协议）**：20 题 two_choice 70%（n=20 子集，需扩到 50 确认）；
   select_best 修键后同题重评为 68%（§8.2）。
6. **harness 变更**：`backend/eval/batch_eval.py` 现在默认读 relay.json（`eval` key + `models.debug[0]`），
   支持 `--model Kimi-K3` 等任意模型名、`--reason-suffix`、空响应自动重试；不再设 token 上限。


## 5.5 显著性检验与"题目是否可答"：claude-opus-5 事后评审（2026-08-01 第五轮）

### 5.5.1 已做的显著性检验（出题端）

| 过滤 | select_best | two_choice |
|------|-------------|------------|
| 跨 seed win_rate | winner vs runner-up ≥ 0.7（10 seed 里 ≥7） | ≥ 0.8（10 seed 里 ≥8） |
| ratio 下限 | CE ≥ 1.03 / MSE ≥ 1.15 | [1.2, 5.0] |
| 结构过滤 | v1.2 显著字段两两差异 ≥ 2 | `choices_have_contrast` |

**关键局限**：这只保证"GT 胜者 vs 亚军在统计上有区分"，**不保证"参考 settings 能支撑测试者推出答案"**。
参考是随机采样，从不做"覆盖判别轴"校验——这正是 5.2 中 kNN oracle 只有 8%、判别轴覆盖率仅 11% 的原因。

### 5.5.2 两道有代表性的题

- **`sb_fedea4`（ratio 3.4，误导型）**：参考最优指向"深+宽"（d6w128 / d3w256，loss 0.05–0.06），
  真胜者是 **d2w32+残差**（C）。残差轴在参考里从不被单独检验，照参考选必错。
- **`sb_a5b412`（ratio 1.07，噪声型）**：5 个参考 loss 全在 3.19–3.63 的窄带内（80 步、接近 ln(32)=3.47
  随机基线），胜者只比亚军好 7%；参考里 SGD 表现最差（3.63），胜者却恰好是 SGD lr=0.003——优化器×lr
  轴在参考里无覆盖，信息不足且方向误导。

### 5.5.3 我的分析：规律几乎找不到，题目欠定（与模型强弱无关）

- 参考 kNN 外推预测器只有 **5/59（8%）**，低于随机 16.7%——按题面信息"不可答"是结构性事实。
- winner vs runner-up 判别轴 134 个，仅 15 个（11%）有单轴邻居参考；残差 0/16、weight_decay 0/13、
  momentum 0/7 完全无覆盖；59 题里仅 2 题可完全校准。
- 分 ratio 看：tight（<1.15）两模型 0/5；medium 11–15%（低于随机）；loose（≥2）33%（2 倍随机）。
  越紧的题参考越可能误导——就像复杂系统：5 个随机采样点无法约束高维 loss 曲面。

### 5.5.4 claude-opus-5 事后盲审（18 题：12 select_best + 6 two_choice，不给 GT）

| 题型 | 盲答正确 | 可答性均值(1-5) | 判定 fair |
|------|----------|------------------|-----------|
| select_best | **1/12（8%）** | 2.58 | 1/12 |
| two_choice | 3/6（50%） | 3.67 | 4/6 |

- **claude-opus-5 盲答 select_best 也是 8%，与随机持平、与 kNN oracle 一致**——不是模型不行，是信息不够。
- 典型评审意见（节选）：
  - "All losses sit within ~0.4 of the ln(32)=3.47 random baseline after only 80 steps… optimizer/lr
    effects are indistinguishable from run-to-run noise."（sb_a5b412）
  - "References only show SGD at a tiny lr… the optimizer/effective-lr axis that decides the winner is
    never probed, and the top candidates differ by amounts likely near run-to-run noise."（sb_c17bb6）
  - "References only show one learning rate per optimizer, so the lr-vs-optimizer interaction must be
    extrapolated…"（sb_d5f3cc）
  - "References only sample a coarse depth/width trend and never disambiguate the close cases…"（sb_4f62f7）
- 18 题里没有一题被判 unfair，13 题 borderline、5 题 fair——claude 措辞温和，但可答性 1–3 分占 13/18，
  与量化结论吻合。

### 5.5.5 结论与建议

1. **用户的判断成立**：多数 select_best 题像复杂系统，5 个随机参考无法逻辑推出胜者；claude-opus-5
   盲审（8%）与 kNN oracle（8%）互相印证。
2. **修题面**：参考必须覆盖选项的判别轴（base + 1–2 点修改并测 loss 的 few-shot，即 propose 范式）；
   或把 tight（<1.15）题移出主指标只作诊断。
3. **评测协议**：`--run-dir` 每次评测一个文件夹（run.json + 题目快照 + results/{model}/responses.jsonl
   + summary），`backend/eval/report_html.py` 生成含完整思考过程的 `report.html`（含事后评审表）。


## 6. 结论与建议（2026-08-01 修订，旧结论因答案键 bug 作废）

- **select_best_v2 可解**：修键后 claude 70% / terra 72% / luna 74%（6 选 1，随机 16.7%），
  rank_score 4.4–4.6/5；"接近随机是预期"的旧结论错误。
- **区分度问题仍在**：选择题（6 选 1 / 2 选 1）模型全挤在 66–74%，天花板效应（见 §10）；
  主指标建议转 propose_improvement（开放式生成、按落点/涨跌打分）。
- **参数量 prior 泄露**：select_best_v2 中 66% 的题 winner=参数量最大选项（中位数跨度 163×），
  违反 1.1× 对齐规则，题目需按 §12.4 修。
- **曲线暂存不展示**：GT 的 `curves.npz` 已随候选存于 `backend/data/results/`，observable 阶段再接入。

---

## 7. 待办

- [ ] propose_improvement 批量闭环：LLM 并行 propose 5 个修改 → GT → 分层入库（`tools/batch_generate` 已有并行骨架）
- [ ] 同数据集多题跨批次（避免跨题泄漏）的完整评测协议
- [ ] meta-model（TabPFN）评测（第三 section，暂不实现）

## 8. 勘误（2026-08-01）：select_best v1.2 答案键 bug —— 根因已定位并修复

### 8.1 现象

- 上一版结论（§5.5.5 / §6）认为 select_best 接近随机是"题目太难、复杂系统不可判"，**结论错误**。
- 实际根因：`backend/eval/questions.py` 的 `build_select_best` 存在**索引失效 bug**：
  1. 先按 `mean_{metric}` 算出 `winner_i` / `runner_i`（shuffle 前的下标）；
  2. `rng.shuffle(pool)` 之后，又用**旧下标**去取 `pool[winner_i][0]` 当作 winner；
  3. 结果 `winner_candidate` / `correct_letter` 是 shuffle 后的**随机选项**，与 GT 无关。
- 量化：select_best_v1.2 的 59 题中，嵌入答案键与当前 `summary.json`（执行 GT）一致的只有 **12/59**；
  47 题答案键是错的（≈ 随机标注，符合 1/6 预期）。`ratio` / `win_rate` 字段本身算对了，但挂在了错误的候选上。

### 8.2 影响（旧数字作废）

| 模型 | select_best v1.2（坏键） | select_best v2（修复后同题重打分） | two_choice local（对照，未受影响） |
|---|---|---|---|
| claude-opus-5 | 9/50 = 18% | **34/50 = 68%** | 36/50 = 72% |
| gpt-5.6-terra | 10/50 = 20% | **35/50 = 70%** | 35/50 = 70% |

- 在 47 道"键被改对"的题里，claude 答对 29 道、terra 答对 27 道——模型其实是**按当前 GT 选了真最优**，只是被判错。
- two_choice 路径（`backend/eval/two_choice.py`）无此 bug：84 个 local 题答案键 84/84 与当前 summary 一致。

### 8.3 修复

- `build_select_best`：在 `rng.shuffle(pool)` **之前**提取 `winner_id` / `runner_id`，shuffle 后仅用 candidate_id 匹配字母。
- 已用原 seed `20260804` 重建 `backend/eval/sets/select_best_v2/questions.jsonl`：59 题与 v1.2 **同池同参考**，
  仅答案键修正，59/59 与当前 `summary.json` 一致。坏键的 `select_best_v1/v1.1/v1.2` 题集、相关探针/打分记录（`artifacts/eval_probe/select_best_v1.x`、`artifacts/eval_runs/select_best_v1.2_*`、50 题 run 中的 select_best 部分）已于 2026-08-02 全部删除。

### 8.4 对后续出题的启示（与用户"好 base + 改进"方向一致）

- select_best 可解性比之前认为的好得多（~70%），但 5 个随机参考仍欠定：tight 题（<1.15）和
  判别轴未覆盖的题仍会拖低分数。
- 下一版题面按 §5.5.5 建议改造：base 取候选池最优/次优，参考改为"base + 1–2 点修改并带 loss"
  的 few-shot（即 propose 范式），选项包含 base 本身。

## 9. 排名计分（partial credit）—— 2026-08-01 决策

### 9.1 规则

- 每个选项的 GT 排名 = 该选项在当前 `results/summary.json`（执行 GT）上的选择指标排名
  （select_best 恒为 lower-is-better；two_choice 按 `ask` 字段决定 higher/lower）。
- 得分 = `n_options - rank`：6 选 1 时第 1 名 5 分、…、第 6 名 0 分；二选一时退化为 1/0（= 旧准确率）。
- 排名只依赖「选项 candidate_id + 当前 summary」，因此对 §8 的坏答案键免疫。
- 实现：`backend/eval/ranking.py`（`rank_result` / `score_rows` / `summarize`），
  `batch_eval.py` 的 live 打分与 `python -m backend.eval.ranking --responses <file> --set <set>` 离线重打分共用同一逻辑。
- 汇总指标：`mean_rank`、`mean_rank_score`（0..n-1）、`rank_score_norm`（%）、`top1/top2/top3`、`rank_dist`、按 ratio 分层的 mean_rank/top1。

### 9.2 二选一审计（用户询问"二选一是否也要修"）

- two_choice 无 shuffle 索引 bug：`artifacts/eval_probe_local/items` 84/84、
  `artifacts/eval_probe/items` 57/57、`items_confignear/items` 57/57、`items_seed2/items` 57/57，
  全部与当前 summary 一致，无需修复。

### 9.3 50 题重打分（rank 计分，与 §8 修复后的答案键一致）

| 模型 | select_best 6选1（50题） | two_choice 2选1（50题） |
|---|---|---|
| claude-opus-5 | mean_rank 1.60，mean_score 4.40/5 = **88%**，top1 70% / top2 80% / top3 94% | mean_rank 1.28，score 0.72/1 = **72%** |
| gpt-5.6-terra | mean_rank 1.56，mean_score 4.44/5 = **88.8%**，top1 72% / top2 84% / top3 94% | mean_rank 1.30，score 0.70/1 = **70%** |

- select_best 分层（claude）：tight(<1.15) top1 80%、medium 67%、loose 72%——修复答案键后
  tight 题不再是 0%，"tight 不可判"的旧结论作废。
- 排名分布（claude select_best）：rank1×35、rank2×5、rank3×7、rank4×2、rank6×1——模型很少落到倒数两名。

## 10. 为什么二选一 ≈ 六选一？+ 便宜模型校准（gpt-5.6-luna）

### 10.1 现象

修键后各模型 50 题 top1 正确率几乎都落在 70–74%，六选一并不比二选一"更难"：

| 模型 | select_best 6选1 top1 | select_best rank_score | two_choice 2选1 top1 |
|---|---|---|---|
| claude-opus-5 | 70% | 4.40/5 = 88% | 72% |
| gpt-5.6-terra | 72% | 4.44/5 = 88.8% | 70% |
| gpt-5.6-luna（最便宜 debug） | **74%** | 4.60/5 = 92% | **74%** |

### 10.2 原因

1. **两套题池难度不同**：two_choice local 的 50 题里 38/50（76%）是 ratio 1.2–2× 的 medium 对，
   medium 层正确率只有 68–71%（claude 68% / luna 71%）；select_best 各层 67–80%。
2. **信息量不对称**：select_best 每题给 5 个带 loss 的参考 setting，判别轴更丰富，六选一只比二选一"看起来难"；
   而 select_best 的随机基线是 16.7%，**超出基线 +53–57pp**；two_choice 基线 50%，只超出 +22–24pp。
   绝对正确率撞在 ~70% 是巧合。
3. **ask 方向**：two_choice 里 ask=lower 略难（claude 62% / terra 65% vs higher 83% / 75%），
   luna 无此差异（73% vs 75%），样本小（24–26 题），先不据此调题。

### 10.3 题目是否合理？（校准结论）

- **可解性与合理性 ✅**：远超随机基线、答案键修复后分数可信、模型几乎不落到倒数（rank_score 88–92%）。
- **区分度 ⚠️ 天花板效应**：最便宜的 luna 与 claude/terra 打平甚至略高（74% vs 70–72%），
  当前 6 选 1 / 二选一题面无法区分模型能力。
- **建议**：把 select_best/two_choice 保留为"可解性诊断"，主推进方向改为 propose_improvement
  （开放式生成、按涨跌打分），或调难选择题（减少参考、收紧 gap、去掉局部校准 demos）。

## 11. 旧 60 题 → 新评测框架 + 模型偏序（2026-08-01）

### 11.1 旧 60 题转新评测（用户方案：三选项进 hint）

- 旧 60 题 = `examples/quiz_demo/bundle/datasets/{family}/{dataset_id}/questions/run_20q_3c_*/`
  （sym_62678b / mvar_c59a30 / bg_0021c1 各 20 题，三选一）。
- 转换器 `backend/eval/old60.py`：每题保留旧的三选项（带 loss）作 calibration hints，
  再加 2 个同池邻近 setting；新选项 = 同池内离旧 winner 1–8 个 config 编辑、salient 1–3 的 3 个
  setting（保留三选一结构，便于与旧盲答数字对比）；选项 loss 不展示，模型选最低。
- 产出 `backend/eval/sets/select_best_old60/questions.jsonl`：48 题（uni 18 / multi 17 / bg 13；
  bg 压缩 CE 用放宽阈值 ratio≥1.02、wr≥0.6，MSE 用 ratio≥1.15、wr≥0.7）。
- 完整性：选项与 references 零重叠、correct key 全部与当前 summary 一致。

### 11.2 luna 结果（3 选 1 + hints，48 题，无 token 上限、reasoning high）

| 指标 | 值 |
|---|---|
| top1 正确率 | 33/48 = **68.8%**（random 33.3%） |
| mean_rank / rank_score | 1.44 / 1.56 of 2（78.1%）；top2 87.5% |
| 分层 | loose(≥2) 88.9% / medium 58.8% / tight 54% |
| 分 family | univariate 83.3% / multivariate 64.7% / bigram 53.8% |

对比：旧 60 盲答（无 hint）GPT-5.5 只有 46.7%、人类基线 70%——给 hint 后便宜模型即达人类盲答水平。

### 11.3 模型偏序（docs/0710_gpteval.html + 本地 llm_runs/vapi_* + 新评测）

**旧 60 盲答（3 选 1，无反馈，外部评分）：**
gemini-3.5-flash-thinking 48.3% ≈ gpt-5 46.7% ≈ gpt-5.5 46.7% > claude-opus-4-8 40.0%
≈ claude-sonnet-5 40.0% > o3 35.0% > gemini-2.5-pro 23.3%（跨度 25pp，能区分模型）；
人类 唐晨成 70%，只看参数量规则 66.7%。

**旧 60 全程顺序（先答后反馈）：** GPT-5.5 79.7% ≈ GPT-5.4 78.3% ≈ GPT-5.6-SOL 78.3%（无区分）。

**新题 high_budget 13/15 盲答（llm_runs/vapi_*；4 选 1，随机 25%）**：
gpt-5.6-terra 7/13 = 53.8% > deepseek-v4-pro 3/13 = 23.1% > claude-sonnet-5 1/13 = 7.7%；
deepseek-v4-pro budget15 5/15 = 33.3%（同题 GPT-5.4 盲答 20%、顺序 46.7%）；
deepseek 3x3binary180 108/180 = 60%。
⚠ 这些是 **4 选 1**（不是 6 选 1）；且 n=13 单次抽取噪声大——重跑控制见 §11.5。

**新评测框架 50 题：** two_choice_local：luna 74% ≈ deepseek 74% ≈ claude-opus-5 72%
≈ terra 70% > kimi 66.7%；select_best_v2：luna 74% ≈ terra 72% ≈ claude 70%。

**meta-model 对照（旧 60 external）：** CV champion / XGBoost 54/60 = 90% > MLP 76.7%
> 参数量规则 66.7%（删参数量列仅 -3 题，主信号在 optimizer/lr/architecture 交互）。

### 11.4 结论

1. **用户判断成立**：旧 60 盲答能拉开模型（25pp 跨度），新题 6 选 1/二选一挤在 66–74%（8pp），
   新题区分度差；旧题"更合理"的观感有数据支持。
2. 但一旦给 hint（11.2），旧题也被便宜模型推到 69%——区分度主要来自"盲答无参考"。
3. 建议：选择题都当"可解性诊断"，主指标走 propose_improvement（开放式、按落点/涨跌打分），
   或继续用旧题盲答格式做模型能力排序。

### 11.5 字母交换控制实验（2026-08-02，回答"claude-sonnet-5 为什么比随机差"）

把 vapi 的 13 个零样本 prompt 原样重跑 + A↔B 内容交换变体重跑（`tools/llm_eval/run_letter_swap_control.py`，
relay eval key，temp=0，记录保留在 gitignored `llm_runs/control_letterswap_20260802/`）：

| 模型 | vapi 原跑 | 重跑 original | 重跑 swapped_ab | 字母交换后同候选率 |
|---|---|---|---|---|
| claude-sonnet-5 | 1/13 = 7.7% | 2/13 = 15.4% | 2/13 = 15.4% | 6/13 = 46.2% |
| gpt-5.6-terra | 7/13 = 53.8% | 5/13 = 38.5% | 3/13 = 23.1% | 5/13 = 38.5% |

结论：

1. **7.7% 是低抽，不是稳定水平**：temp=0 重跑 claude 2/13（15.4%）；两轮合并 3/26 = 11.5%。
   terra 也从 53.8% 掉到 38.5%（两轮 12/26 = 46.2%）；terra vs claude 合并 Fisher 双尾 **p=0.013**，
   差距真实存在但远小于 46pp 的表观差。
2. **claude 有轻微"避开首选项 A"的位置偏置**：两轮 A 只选 2/26（均匀期望 6.5），P(X≤2)=0.026；
   terra 无此偏置（A 6/26，p=0.52）。这与 §5.2.2 的 DeepSeek 位置偏置是同类现象（方向相反），
   在出题端应做 per-item 字母乱序 + 一致性检查。
3. **模型主要是"位置/字母驱动"而非内容驱动**：字母交换后 claude 54% / terra 62% 的题换了候选
   （同候选率 46%/38%，随机基线 25%）——这批 4 选 1 题对 LLM 的内容信号很弱，
   与"部分题混沌/先验误导"的判断一致；唯一稳健可解的仍是 q_7dc8c1（bigram 最小模型题，
   交换字母后两模型都跟着内容走）。
4. 含义：high_budget 13 题的模型排序不可靠，需要 ≥50 题 + 字母乱序重跑才有意义；
   该 release 应并入"可解性诊断"，不作模型能力排序。

## 12. 同事推送的新题分析 + opus 事后审计（2026-08-01）

### 12.1 新题来源（origin/main，本地分支落后 51 个提交）

- **唐晨成（tcc）**：`benchmark_releases/question_packs/` 两个 100 题包（2 选 1、architecture_only）：
  - `xor-v2.5-100q-37b9da`：synthetic_tabular_classification，30 MLP / 70 KAN；
  - `gru-v2.5-100q-a48abc`：bigram_lm，50 transformer_lm / 50 gru_lm。
- **任梓睿（renrua52）**：`tools/benchmark_v1_build.py`（分层配额 + audit 报告）、
  `src/architecture_iq/questions/quality.py`（可选质量过滤器：gap_max/gap_worst_max/require_finite_mean/max_failed_seeds）、
  `profiles/v1.yaml`（统一池、num_choices: 2）、XOR/spiral 分类 family、TabPFN 管线。

### 12.2 程序化检查（两包共 200 题）

| 包 | 答案键==GT | ratio 中位数 | tight(<1.05) | 预算一致 | 参数量比值中位数 | winner=参数量最大 |
|---|---|---|---|---|---|---|
| XOR | 100/100 | 1.19 | 0% | 100/100 | 2.12×（max 18.7×） | 52%>2× 跨度 |
| GRU | 100/100 | **1.017** | **95%** | 100/100 | 3.61×（max 64×） | 78%>2× 跨度 |

- GRU 包：CE 压缩，winner-runner 差距中位数 1.7%，落在 seed 噪声内 → 本质不可判。
- XOR 包：KAN 赢 70%（= 集合标题 30 MLP / 70 KAN），参数量/模型族 prior 直接泄露答案。

### 12.3 claude-opus-5 事后审计（12 题：XOR 3 / GRU 3 / select_best_v2 3 / old60 3，优先 tight）

- 判定可答题性（1–5，5=可推理）：mean 3.08；GRU 全部 2 分、XOR 3–4 分、select_best_v2 3 分、**old60 全部 4 分（最高）**。
- 失败模式：`gap_too_small` ×5（GRU 3 + select_best_v2 2）、`type_prior` ×3（全部 XOR）、`ok` ×4（全部 old60 + 1）。
- opus 原话（GRU）："gap 需要明显超过 ~0.007 的 seed 噪声"；opus 原话（XOR）："答案取决于 task-specific fit 而不是
  'KAN 在低维光滑 tabular 上赢' 的一般 prior"。
- 结论与用户判断一致：**旧 60 风格的题（hints 版）最合理；GRU 包是噪声；XOR 包被 type/params prior 污染**。

### 12.4 参数量/预算 1.1× 对齐检查（用户规则：候选 ≤ 最大 baseline 的 1.1 倍）

- `score_proposal.py` 已实现：`MAX_PARAM_RATIO=1.1`，`total_samples_seen` 固定为 base 的（50 个提案里 47 个通过）。
- 但**题库选项没遵守**：select_best_v2 的 59 题里 43 题选项间参数量跨度 >1.1×（中位数 **163×**）、
  old60 48/48（中位数 47×）、propose_v1.1 的 demos 55/66（中位数 65×）；
  select_best_v2 里 **66% 的题 winner=参数量最大选项**（old60 62%）→ 分数主要来自"选最大模型"prior，不是推理。
- 训练预算 total_samples_seen：select_best_v2 / old60 全部对齐（0 mixed）；propose_v1.1 有 4/66 demo 预算混合。
