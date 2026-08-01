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
# select_best（v1.2，当前推荐）
.venv/bin/python -m backend.eval.questions --type select_best \
    --items-per-problem 5 --set-name select_best_v1.2 --seed 20260804

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
.venv/bin/python -m backend.eval.probe --set select_best_v1.2 --num-batches 2 --batch-size 6 --seed 20260805
# 探针评分（subagent 把答案写到 batches/batch_{i}_answers.jsonl 后）
.venv/bin/python -m backend.eval.probe --set select_best_v1.2 --score
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

| 题集 | 题目数 | 正确率 | 随机基线 | 备注 |
|------|--------|--------|----------|------|
| `select_best_v1`（参考按全量分位数） | 12 | 3/12（25%） | 16.7% | 参考区间外推难 |
| `select_best_v1.1`（参考锚定选项区间） | 12 | 2/12（16.7%） | 16.7% | 未见提升；选项含"看似相同"的 ill pairs |
| `select_best_v1.2`（显著字段过滤 + 分 metric 阈值） | 5 | 1/5（20%） | 16.7% | 结构更干净（中位 ratio 1.27 → 1.48），仍接近随机 |
| `select_best_v1.2` 细查 | 1（ratio 3.4 错题） | — | — | 参考未覆盖判别轴（残差），对选项有误导 |
| `two_choice_loss_compare`（诊断） | 8 | 7/8（87.5%） | 50% | 二选一 + 参考校准对 LLM 可解 |
| `propose_improvement_v1`（config 修改） | 2 次出题 | 2/2 击败 base | — | ratio 1.14x / 1.20x，win_rate 1.0 |

**关键发现**：

1. **任务格式决定可解性**：同样基于参考 loss 校准，二选一 87.5%、六选一 ≈ 随机。
   六选一更接近"鉴别 LLM 是否具备从实验中推断架构规律"的基准目标，但当前对 LLM 过难（地板效应）。
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
| `select_best_v1.1` | 50 | 9/50（18%） | 16.7% | 六选一 ≈ 随机 |
| `select_best_v1.2` | 50 | 4/50（8%，另一 run 2/50） | 16.7% | 六选一 ≈ 随机（或更低） |
| `two_choice` seed1（range demo） | 50 | 27/50（54%） | 50% | 二选一 ≈ 随机 |
| `two_choice` seed2（range demo） | 50 | 26/50（52%） | 50% | 二选一 ≈ 随机 |
| `two_choice` config_near demo | 50 | 21/50（42%） | 50% | 配置邻居 demo 是负优化 |
| `propose_improvement_v1.1` | 50 | **48/50 击败 base（96%）** | — | 中位 ratio 1.33，win_rate≥0.7 有 45 题 |
| `propose_improvement_v1` | 50 | **43/49 击败 base（88%）** | — | 中位 ratio 1.32，win_rate≥0.7 有 39 题 |

### gpt-5.6-terra（phybench 中转恢复窗口，50 题/setting）

| 题集 | 题量 | 正确率 | 随机基线 |
|------|------|--------|----------|
| `select_best_v1.2` | 50 | 9/50（18%） | 16.7% |
| `two_choice` seed1（range demo） | 50 | 38/50（76%） | 50% |
| `two_choice` config_near demo | 50 | 31/50（62%） | 50% |

**50 题样本下的结论（替代小样本探针）**：

1. **六选一 select_best 对两个模型都是地板**（8–18%，随机 16.7%）；gpt-5.6-terra 略高于随机，
   deepseek 偶尔低于随机（被参考误导）。
2. **二选一区分两个模型**：gpt-5.6-terra 76% 显著高于 deepseek 52–54%（≈随机）。
   → 二选一可以作为**模型能力的有效度量**；六选一作为上限测试。
3. **config_near demo 在两个模型上都是负优化**（gpt 76%→62%，deepseek 54%→42%），
   保留 range（min/median/max）demo 策略。
4. **propose_improvement 是强信号任务**：deepseek 两组 50 题分别 96% / 88% 击败 base
   （GT 全部新跑），中位 ratio ~1.32；top 案例 ratio 数百倍（如 base 0.208 → 0.0003）。
   即使一个只会"抄参考中最优 config"的模型也能赢，说明任务对 LLM 友好、可度量。
5. **工程结论**：50 并发 + 多 setting 并行（`--sets`）稳定跑通（3 setting × 50 题 ≈ 10s API 时间）；
   DeepSeek 官方 API 无限流问题；phybench 中转不可靠（会整批空响应）。

## 6. 结论与建议

- **select_best 保留为"难任务"**（v1.2 结构更干净），用于测 LLM 的上限；分数接近随机是预期现象，需要记录随机基线与置信区间。
- **two_choice 作为诊断/控制任务**：验证模型确实在做参考校准（87.5%），防止"所有任务都随机"的系统性失效。
- **propose_improvement 作为主推进方向**：闭环已验证（LLM propose → 校验 → GT → 涨跌），
  下一步可批量并行 propose 5 个修改跑 GT，按涨/跌/平分层入库。
- **曲线暂存不展示**：GT 的 `curves.npz` 已随候选存于 `backend/data/results/`，observable 阶段再接入，仅改评测端代码。

---

## 7. 待办

- [ ] v1.2 探针补测（大样本），确认显著字段过滤是否提升可答性
- [ ] propose_improvement 批量闭环：LLM 并行 propose 5 个修改 → GT → 分层入库（`tools/batch_generate` 已有并行骨架）
- [ ] 同数据集多题跨批次（避免跨题泄漏）的完整评测协议
- [ ] meta-model（TabPFN）评测（第三 section，暂不实现）
